"""``mast/**.py`` 随包带 —— 以及**凭据不许跟着走**。

为什么要带源码：PyInstaller 把全部 ``.py`` 编译进 ``MAST.exe`` 内嵌的 PYZ，
冻结树里 ``_internal/mast/`` 只剩十来个 data 文件。但 ``mast.pyexec`` 要把
``sitecustomize.py`` / ``mastdata.py`` / ``io/nanonis_files.py`` **拷进分析会话
目录**（子进程里没有 MAST，只能拿文件），skill 覆盖层的 eject 也要源文件原文。

为什么凭据是个真问题：``mast/update/_defaults.py`` 里有 **``DEFAULT_TOKEN``
（推送服务器明文 token）**。它随源码进包，再给 DP 的分析子进程一个只读视图，
就等于把推送凭据交给一个会写代码的 agent。``mast2_build.ps1`` Step 4 那道
「删掉打包进去的 api key\\*.env」纪律**管不到源码通道**。

这份测试里最值钱的是最后两条 —— 它们钉住实现时真踩到的两个坑，两个都会让守卫
**静默失效**（照样构建成功，凭据照样进包，没有一行输出提到）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = str(_REPO / "MASTv2" / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import bundle_sources as bs  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return bs.collect(_REPO / "MASTv2", _REPO)


# ── 收集面 ──────────────────────────────────────────────────────────
def test_the_whole_package_is_collected(report):
    assert report.n >= bs.MIN_FILES, (
        f"只收到 {report.n} 个 .py —— 走目录收进来的东西缺失时一条丢弃记录都不会有")
    assert not report.degraded, f"git 探测降级了：{report.degraded}"


def test_dest_layout_matches_what_pyexec_looks_for(report):
    """落点必须是 ``mast/_src/`` —— 那正是 ``_srcfiles.source_root()`` 找的地方。

    两边对不上的后果不是报错，是**打包版里 pyexec 静默不可用**。
    """
    assert bs.DEST_ROOT.replace("\\", "/") == "mast/_src"

    by_rel = {Path(src).name: dst for src, dst in report.files}
    assert "nanonis_files.py" in by_rel
    assert by_rel["nanonis_files.py"].replace("\\", "/") == "mast/_src/io"
    assert by_rel["child_audit.py"].replace("\\", "/") == "mast/_src/pyexec"


def test_the_files_pyexec_copies_are_all_there(report):
    """点名检查 pyexec 真正会拷的那几个。

    少任何一个，打包版的分析子进程就装不上审计钩子 —— 而那是「不能删掉原始测量
    数据」这条底线的唯一执行者。
    """
    names = {Path(src).name for src, _ in report.files}
    for must in ("child_audit.py", "child_sitecustomize.py", "runtime_helper.py",
                 "nanonis_files.py", "si_quantity.py"):
        assert must in names, f"{must} 没被收进源码副本"


# ── 凭据 ────────────────────────────────────────────────────────────
def test_no_credentials_in_what_would_ship(report):
    assert not report.violations, (
        "这些疑似凭据会随源码发出去：" +
        "；".join(f"{p}:{n}" for p, n, _ in report.violations))


def test_the_push_token_file_is_excluded(report, tmp_path):
    """公开树不含本地生成凭据；独立 git 夹具验证忽略文件确实被跳过。"""
    import subprocess

    assert "_defaults.py" not in {Path(src).name for src, _ in report.files}
    repo = tmp_path / "bundle_fixture"
    package = repo / "MASTv2" / "mast"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    ignored = package / "_defaults.py"
    ignored.write_text("LOCAL_VALUE = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text("_defaults.py\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True, capture_output=True)
    synthetic = bs.collect(repo / "MASTv2", repo)
    assert not synthetic.degraded
    assert ignored in synthetic.skipped_gitignored
    assert "_defaults.py" not in {Path(src).name for src, _ in synthetic.files}
    assert "__init__.py" in {Path(src).name for src, _ in synthetic.files}


@pytest.mark.parametrize("shape,src", [
    ("同一行", 'DEFAULT_TOKEN = "f4IPgvqjXY71aQ0zKm"\n'),
    ("反斜杠续行", 'DEFAULT_TOKEN = \\\n    "f4IPgvqjXY71aQ0zKm"\n'),
    ("括号续行", 'DEFAULT_TOKEN = (\n    "f4IPgvqjXY71aQ0zKm"\n)\n'),
])
def test_a_planted_token_is_caught(tmp_path, shape, src):
    """三种写法都要抓到。

    「括号续行」这条是**变异验证逼出来的**：为了消除跨行误报，一度把赋值号到
    引号之间也改成不许换行 —— 误报是没了，但这种真凭据从此漏掉，而当时**没有
    任何测试会红**。两个跨行子句作用相反，各配一条测试，缺一个都会被下一个人
    「简化」掉。
    """
    f = tmp_path / "leaky.py"
    f.write_text(src, encoding="utf-8")
    hits = bs.scan_credentials(f)
    assert hits and hits[0][0] == "DEFAULT_TOKEN", f"{shape}的凭据没抓到"


def test_the_public_key_is_allowed(tmp_path):
    """公钥公开是它的用途 —— 客户端拿它验发布签名。"""
    f = tmp_path / "ok.py"
    f.write_text('DEFAULT_SIGNING_PUBKEY = "9ec1ebda69b9aa11bb22cc33dd44"\n',
                 encoding="utf-8")
    assert bs.scan_credentials(f) == []


def test_placeholders_are_not_flagged(tmp_path):
    """占位符不是凭据。误报会让人学会忽略这道守卫，那比没有更糟。"""
    f = tmp_path / "ph.py"
    f.write_text('API_KEY_DEFAULT = "<your-key-here>"\n'
                 'DEFAULT_PASSWORD = "CHANGEME_BEFORE_USE"\n', encoding="utf-8")
    assert bs.scan_credentials(f) == []


def test_a_function_call_is_not_a_credential(tmp_path):
    """只认**字面量**赋值。``TOKEN = get_token()`` 是代码，不是凭据。"""
    f = tmp_path / "call.py"
    f.write_text('DEFAULT_TOKEN = _read_token_from_disk()\n', encoding="utf-8")
    assert bs.scan_credentials(f) == []


# ── 两个实测踩到的坑 ────────────────────────────────────────────────
def test_the_scanner_does_not_match_across_lines(tmp_path):
    """跨行匹配会把无关的字符串报成凭据。

    这段是 ``mast/update/defaults.py`` 的**真实结构**。正则写成 ``[^=]*?`` 时：
    ``DEFAULT_TOKEN = ""`` 的第一个引号匹配不到 12 个字符，于是回溯 —— 用**第二个**
    引号当开头，一路吃到几行之后 ``DEFAULT_SIGNING_PUBKEY = ""`` 的引号为止，
    报出一个根本不存在的「凭据」。

    ⚠️ 第一版这条测试写的是「TOKEN 赋空值 + 后面没有引号」，**变异验证时不变红**
    —— 因为没有后续引号就配不成对，那段内容根本触发不了这个 bug。
    一条不会红的测试等于没有测试；这里的关键是**后面必须再有一个引号**，
    而且中间隔着 12 个以上非引号字符。
    """
    f = tmp_path / "multi.py"
    f.write_text(
        'try:\n'
        '    from mast.update._defaults import DEFAULT_SERVER_URL, DEFAULT_TOKEN\n'
        'except ImportError:\n'
        '    DEFAULT_SERVER_URL = ""\n'
        '    DEFAULT_TOKEN = ""\n'
        '\n'
        'try:\n'
        '    from mast.update._defaults import DEFAULT_SIGNING_PUBKEY\n'
        'except ImportError:\n'
        '    DEFAULT_SIGNING_PUBKEY = ""\n',
        encoding="utf-8")
    assert bs.scan_credentials(f) == [], "跨行误报又回来了"


def test_the_gitignore_probe_survives_windows_paths(tmp_path):
    """``git check-ignore`` **必须**用 ``-z``。

    默认输出会把含特殊字符的路径**加引号并 C-style 转义**（``core.quotePath``）
    —— Windows 路径里每一个反斜杠都触发它，于是回来的是带引号、反斜杠双写的
    字符串，与传入的路径一个都对不上。表现：这道守卫**静默失效** ——
    ``_defaults.py`` 照样进包，而没有任何一行输出提到它。

    这里直接对真仓库问一次，断言那个已知被忽略的文件真的被认出来。
    """
    target = _REPO / "MASTv2" / "mast" / "update" / "_defaults.py"
    if not target.is_file():
        pytest.skip("本机没有 _defaults.py（未跑过构建脚本）")

    ignored, degraded = bs.gitignored(_REPO, [target])
    assert not degraded, degraded
    assert target in ignored, (
        "git 认为它被忽略，但我们的解析没认出来 —— 八成又是引号/转义问题")


def test_a_broken_git_probe_is_reported_not_swallowed(monkeypatch):
    """git 问不出来时要**报降级**，不能当成「没有被忽略的文件」。

    静默返回空集 = 第一道判据消失，而且没人知道。
    """
    import subprocess as _sp

    def _boom(*_a, **_kw):
        raise OSError("no git here")

    monkeypatch.setattr(_sp, "run", _boom)
    ignored, degraded = bs.gitignored(_REPO, [Path("x.py")])
    assert ignored == set()
    assert degraded, "降级了却没说 —— 那道判据会静默消失"


# ── spec 侧的声明 ───────────────────────────────────────────────────
def test_the_spec_actually_wires_this_in():
    """漏声明只能在 spec 层面测出来。

    收集逻辑再对，spec 里没调它，打包版里就一个源文件都没有 —— 而 pyexec 会在
    真机上抛 ``SourceFilesMissing``，那时才发现就晚了。
    """
    spec = (_REPO / "mast2.spec").read_text(encoding="utf-8")
    assert "bundle_sources" in spec, "mast2.spec 没有接入源码收集"
    assert "_src_report.violations" in spec, "spec 里没有凭据守卫"
    assert "MIN_FILES" in spec, "spec 里没有数量守卫"
    assert "data_files += _src_report.files" in spec, "收集了却没加进 data_files"
