"""打包版里源文件从哪来 —— 以及拿不到时必须**大声**。

pyexec 需要把若干 ``.py`` 当**文件**拷进会话目录（子进程里没有 MAST，只能拿到
文件）。dev 下那些文件就在 ``mast/`` 里；**打包版里一个都没有** ——
PyInstaller 把全部 ``.py`` 编译进 ``MAST.exe`` 内嵌的 PYZ 归档，
``_internal/mast/`` 下只剩十来个 data 文件。

第一版直接用 ``Path(__file__).parent`` 找，并且写的是 ``if src.is_file()``：
dev 下一切正常，打包版里 ``sitecustomize.py`` **静默**拷不过去 —— 审计钩子不装，
保护消失，而日志、返回值、测试全都看不出任何异常。这是最难发现的一类失败，所以
现在改成抛异常。

这份测试同时是 **A-P1.5 的前置说明**：打包时必须把 ``mast/**.py`` 作为 data 带进
``mast/_src/``，否则 ``source_root()`` 会在真机上抛。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mast.pyexec import _srcfiles
from mast.pyexec._srcfiles import SourceFilesMissing, pyexec_file, resolve, source_root


def test_dev_layout_resolves_to_the_package_itself():
    root = source_root()
    assert (root / "io" / "nanonis_files.py").is_file()
    assert resolve("mast.io.nanonis_files") == root / "io" / "nanonis_files.py"
    assert pyexec_file("child_audit.py").is_file()


def test_frozen_layout_prefers_the_bundled_src_copy(tmp_path, monkeypatch):
    """打包版：``<package_root>/_src/`` 优先。

    这条同时钉住 A-P1.5 的落点约定 —— spec 把源码带到别处（或者忘了带），
    这里就会红。
    """
    fake_pkg = tmp_path / "mast"
    src = fake_pkg / "_src"
    (src / "io").mkdir(parents=True)
    (src / "pyexec").mkdir(parents=True)
    (src / "io" / "nanonis_files.py").write_text("# bundled copy", encoding="utf-8")
    (src / "pyexec" / "child_audit.py").write_text("# hook", encoding="utf-8")

    monkeypatch.setattr("mast._runtime_paths.package_root", lambda: fake_pkg)
    assert source_root() == src
    assert resolve("mast.io.nanonis_files") == src / "io" / "nanonis_files.py"
    assert pyexec_file("child_audit.py") == src / "pyexec" / "child_audit.py"


def test_a_frozen_build_without_the_src_copy_fails_loudly(tmp_path, monkeypatch):
    """既没有 ``_src`` 也没有裸 ``.py`` ⇒ **抛**，而且要说清楚缺什么、怎么补。

    这正是打包版忘了带源码时的样子。静默降级在这里的代价是：分析子进程照跑，
    但**没有审计钩子** —— 一个 `rmtree` 打错路径就能删掉几小时的测量数据。
    """
    empty = tmp_path / "mast"
    empty.mkdir()
    monkeypatch.setattr("mast._runtime_paths.package_root", lambda: empty)

    with pytest.raises(SourceFilesMissing) as exc:
        source_root()
    msg = str(exc.value)
    assert "_src" in msg, "没告诉人该往哪儿放"
    assert "审计钩子" in msg or "mast2.spec" in msg, "没说清后果或怎么补"


def test_provisioning_a_session_raises_when_sources_are_missing(tmp_path, monkeypatch):
    """会话布置不下去要**炸**，不能建出一个没有钩子的会话。

    反例（这就是这条测试存在的理由）：``if src.is_file(): copy`` 会安静地建出一个
    看起来完好、实际毫无保护的会话目录 —— 之后每一次 py_run 都在裸奔，而没有任何
    一行日志会提到这件事。
    """
    from mast.pyexec.session import get_session

    monkeypatch.setenv("MAST_DP_SESSIONS_DIR", str(tmp_path / "sessions"))
    empty = tmp_path / "mast"
    empty.mkdir()
    monkeypatch.setattr("mast._runtime_paths.package_root", lambda: empty)

    with pytest.raises(SourceFilesMissing):
        get_session(experiment_id="e", thread_id="t")


def test_the_bundled_copy_would_be_byte_identical(tmp_path, monkeypatch):
    """``_src`` 副本必须是**字节拷贝** —— mastkit 的 sha256 校验依赖这一点。

    这里模拟一次打包：把真源码树拷成 ``_src`` 布局，然后让 ``verify`` 从那儿装。
    真产物上的对应检查在发布冒烟里（``verify_frozen_artifact.py``）。
    """
    from mast.pyexec import kit_manifest as km

    real_root = source_root()
    fake_pkg = tmp_path / "mast"
    bundled = fake_pkg / "_src"
    bundled.mkdir(parents=True)
    for dotted in km.KIT_MODULES:
        src = resolve(dotted)
        rel = src.relative_to(real_root)
        dst = bundled / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    monkeypatch.setattr("mast._runtime_paths.package_root", lambda: fake_pkg)
    _srcfiles.source_root.cache_clear() if hasattr(
        _srcfiles.source_root, "cache_clear") else None

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    km.install(session_dir)
    assert km.verify(session_dir) == [], "从 _src 装出来的副本对不上源文件"


def test_a_bundled_layout_actually_runs_a_script(tmp_path, monkeypatch):
    """**打包版能不能用** —— 这条是唯一的证据。

    前面几条证明「源解析找得到 `_src`」「副本字节一致」。但那都还没有回答：
    照着打包脚本收出来的那份 `_src`，能不能真的把会话布置起来、把审计钩子装上、
    把脚本跑通？

    这里走完整条：用 ``bundle_sources.collect``（**打包时用的同一份逻辑**）真的
    拷出 `_src` 布局 → 把 ``package_root`` 指过去 → 建会话 → 跑一段带 numpy 的
    脚本 → 断言钩子装上了、原始测量文件动不了。

    没有这条，「打包版里 pyexec 可用」就只是一个推论。
    """
    import shutil as _sh
    import sys as _sys

    from mast.pyexec.runtime import find_runtime

    rt = find_runtime()
    if rt is None:
        pytest.skip("本机没有分析运行时")

    scripts = str(Path(__file__).resolve().parents[4] / "MASTv2" / "scripts")
    if scripts not in _sys.path:
        _sys.path.insert(0, scripts)
    import bundle_sources as bs

    repo = Path(__file__).resolve().parents[4]
    rep = bs.collect(repo / "MASTv2", repo)
    assert not rep.violations and rep.n >= bs.MIN_FILES

    # 照 data_files 的 (src, dest) 摆出冻结树：<pkg>/_src/...
    fake_pkg = tmp_path / "app" / "mast"
    for src, dest in rep.files:
        # dest 形如 "mast/_src/io"；去掉开头的 "mast/" 落到 fake_pkg 下
        rel = Path(dest).relative_to("mast")
        d = fake_pkg / rel
        d.mkdir(parents=True, exist_ok=True)
        _sh.copy2(src, d / Path(src).name)

    monkeypatch.setattr("mast._runtime_paths.package_root", lambda: fake_pkg)
    monkeypatch.setenv("MAST_DP_SESSIONS_DIR", str(tmp_path / "sessions"))

    # 一份"原始测量数据"，用来验钩子真的在管事
    data = tmp_path / "data"
    data.mkdir()
    sxm = data / "probe.sxm"
    sxm.write_bytes(b":NANONIS_VERSION:\n2\n:SCANIT_END:\n" + b"\x00" * 256)

    from mast.pyexec import get_session, run, snapshot

    s = get_session(experiment_id="frozen", thread_id="t1")
    assert (s.root / "sitecustomize.py").is_file(), "打包布局下钩子没被拷进去"
    assert (s.root / "mast" / "io" / "nanonis_files.py").is_file()

    (s.code_dir / "step01.py").write_text(f'''
import numpy as np
from mast.io.nanonis_files import read_sxm          # 来自拷进来的 mastkit
import _mast_audit
print("HOOK", "installed" in _mast_audit.describe())
print("SUM", float(np.arange(10).sum()))
try:
    open(r"{sxm}", "wb").write(b"ruined")
    print("WROTE")
except PermissionError as e:
    print("BLOCKED", type(e).__name__)
''', encoding="utf-8")

    _ = snapshot(s)
    res = run(s, "code/step01.py", runtime=rt, timeout_s=120)
    assert res.ok, res.stderr[-1000:]
    assert "HOOK True" in res.stdout, "钩子没装上（打包布局下的静默失效就长这样）"
    assert "SUM 45.0" in res.stdout
    assert "BLOCKED MastDataProtection" in res.stdout, "钩子装上了但没在管事"
    assert sxm.stat().st_size > 200, "原始测量文件被改了"
