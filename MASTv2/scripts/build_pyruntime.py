"""构建数据处理 agent 的分析运行时（embeddable CPython 3.13 + 离线 wheel）。

在**开发机**上跑一次，产物进 ``MASTv2/pyruntime/``，由 ``mast2_build.ps1``
Step 4.7 拷进 ``dist\\MAST\\MASTv2\\pyruntime``，随全量安装包发出去。

    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/build_pyruntime.py
    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/build_pyruntime.py --selftest-only

为什么要一整套独立解释器
======================
PyInstaller onedir 把纯 Python 源编译进 ``MAST.exe`` 内嵌的 PYZ，``_internal/``
下只剩 ``.pyd``——``_internal/numpy/__init__.py`` 不存在，子进程 import 不了。
而 ``_internal/nanonis_spm/__init__.py`` **是真实文件**：把 ``_internal`` 放进
子进程的 ``sys.path`` 会让 B1（子进程里没有 nanonis_spm）当场失效。
所以独立运行时不是「更干净」的选择，是**唯一**的选择。

三个会静默出错的地方
==================
1. **``python313._pth``**。embeddable 出厂的 ``._pth`` 里 ``import site`` 是注释掉
   的，而且没有 ``Lib\\site-packages``。不改这两处，装进去的包**根本不在
   ``sys.path`` 上**，且 ``.pth`` 文件不被处理（numpy 在某些构建上靠 ``.pth`` 设
   DLL 目录）。症状是「装好了但 import numpy 失败」，而目录里明明有 numpy。
   好的副作用：``._pth`` 存在会让解释器**默认 isolated**（忽略 PYTHONPATH /
   PYTHONHOME、禁 user site）—— 这是 B1 的第二道独立机制。
2. **``.pyc`` 必须用运行时自己的 ``python.exe`` 编**。magic number 要匹配；而且
   运行时装在 Program Files 下，普通用户写不了 ``.pyc``——不预编译的话每次跑都
   重新 parse 整个 scipy。
3. **自检不过要让构建死掉**。带一个坏运行时出厂，会在操作员机器上产生困惑的失败
   （「分析功能有时候不工作」），而在这里失败只是构建红一次。

几乎什么都不 strip
=================
只删 ``__pycache__``（紧接着会用运行时自己的 python.exe 重新编）。第一版还删了
``tests/`` 和 ``.pyi``，省下 15 MB，**弄坏了 numpy.testing 和 skimage** ——
见 :data:`STRIP_DIR_NAMES` 上面那段。``.py`` 源当然要留：traceback 带行号和源码行
是这个功能的全部意义所在（``run_numpy_snippet`` 今天做不到的正是这个）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

PY_VERSION = "3.13.13"
PY_TAG = "cp313"
PLATFORM_TAG = "win_amd64"

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTV2 = REPO_ROOT / "MASTv2"
RUNTIME_DIR = MASTV2 / "pyruntime"
WHEEL_CACHE = MASTV2 / "artifacts" / "pyruntime-wheels"
REQUIREMENTS = MASTV2 / "requirements-pyruntime.txt"

EMBED_URL = (f"https://www.python.org/ftp/python/{PY_VERSION}/"
             f"python-{PY_VERSION}-embed-amd64.zip")

#: 从解包结果里删掉的东西 —— **只有 ``__pycache__``**。
#
# 第一版删 ``tests``/``test``/``testing`` 目录和 ``.pyi``/``.pyx``/``.pxd``/``.c``
# 头文件，省下 3678 个文件、约 15 MB。构建照样成功，**产物是坏的**：
#
#   · ``numpy/testing/`` 匹配了 ``"testing"`` —— 而 ``numpy.testing`` 是公开 API，
#     ``sklearn`` 在 import 期就用它。症状：``No module named 'numpy.testing'``。
#   · ``.pyi`` 被当成纯类型存根删掉 —— 而 ``skimage`` 用 ``lazy_loader``，它
#     **在运行时读 ``__init__.pyi``** 来决定延迟导入什么。症状：
#     ``Cannot load imports from non-existent stub``。
#
# 两个包都是「装了但坏了」，而不是「没装」。15 MB 换两个坏掉的库不划算，而且
# 下一次有人加一个包时，这份名单会不会再咬一口是没法预判的 —— 名单的形状本身
# 就错：它按名字猜哪些文件运行时用不到，而这件事只有那个包自己知道。
#
# ``__pycache__`` 删得掉是因为紧接着的 ``compile_all`` 会用**运行时自己的**
# python.exe 重新生成（magic number 要匹配）。
STRIP_DIR_NAMES = ("__pycache__",)
STRIP_SUFFIXES = ()

#: 必需包缺一个就算构建失败（同 ``pyexec.runtime.REQUIRED_PACKAGES``）。
REQUIRED_IMPORTS = ("numpy", "scipy", "matplotlib")
OPTIONAL_IMPORTS = ("pandas", "skimage", "sklearn")


class BuildError(RuntimeError):
    """构建失败。**一律抛出，不降级**——见模块 docstring 第 3 条。"""


@dataclass
class Report:
    runtime_dir: str = ""
    python_version: str = ""
    n_wheels: int = 0
    n_files: int = 0
    total_mb: float = 0.0
    packages: dict = field(default_factory=dict)
    pth_ok: bool = False
    isolated: bool = False
    leaks: list = field(default_factory=list)
    selftest: dict = field(default_factory=dict)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. embeddable 解释器
# ---------------------------------------------------------------------------

def fetch_embeddable(dest_zip: Path) -> Path:
    if dest_zip.is_file() and dest_zip.stat().st_size > 1_000_000:
        _log(f"  已有 {dest_zip.name}（{dest_zip.stat().st_size / 1e6:.1f} MB），跳过下载")
        return dest_zip
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    _log(f"  下载 {EMBED_URL}")
    tmp = dest_zip.with_suffix(".zip.part")
    try:
        with urllib.request.urlopen(EMBED_URL, timeout=180) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as exc:                              # noqa: BLE001
        raise BuildError(
            f"下载 embeddable Python 失败：{exc}\n"
            f"  URL：{EMBED_URL}\n"
            "  校园网慢的话可以手工下好放到 " + str(dest_zip)) from exc
    os.replace(tmp, dest_zip)
    _log(f"  下载完成（{dest_zip.stat().st_size / 1e6:.1f} MB）")
    return dest_zip


def unpack_embeddable(zip_path: Path, runtime_dir: Path) -> None:
    if runtime_dir.exists():
        _log(f"  清掉旧的 {runtime_dir}")
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(runtime_dir)
    exe = runtime_dir / "python.exe"
    if not exe.is_file():
        raise BuildError(f"解包后没有 {exe} —— 下载的包不对？")
    _log(f"  解包到 {runtime_dir}")


# ---------------------------------------------------------------------------
# 2. ._pth —— 最常见的静默失败点
# ---------------------------------------------------------------------------

def fix_pth(runtime_dir: Path) -> Path:
    """加 ``Lib\\site-packages`` 且取消注释 ``import site``。

    两处都要改。只加路径不开 site，``.pth`` 文件不被处理；只开 site 不加路径，
    ``site-packages`` 不在 ``sys.path`` 上。任一处漏掉的症状都是「装好了但
    import 不到」，而目录里明明有那个包。
    """
    cands = sorted(runtime_dir.glob("python*._pth"))
    if not cands:
        raise BuildError(f"{runtime_dir} 下没有 python*._pth —— 这不是 embeddable 包")
    pth = cands[0]
    lines = pth.read_text(encoding="utf-8").splitlines()
    out, seen_sp, seen_site = [], False, False
    for ln in lines:
        s = ln.strip()
        if s.replace("/", "\\").lower() == r"lib\site-packages":
            seen_sp = True
        if re.fullmatch(r"#\s*import\s+site", s):
            out.append("import site")
            seen_site = True
            continue
        if s == "import site":
            seen_site = True
        out.append(ln)
    if not seen_sp:
        # 放在 `import site` 之前：路径顺序即搜索顺序
        idx = next((i for i, ln in enumerate(out) if ln.strip() == "import site"),
                   len(out))
        out.insert(idx, r"Lib\site-packages")
    if not seen_site:
        out.append("import site")
    pth.write_text("\n".join(out) + "\n", encoding="utf-8")
    _log(f"  {pth.name} 已修正：\n" + "".join(f"      {x}\n" for x in out))
    return pth


def verify_pth(runtime_dir: Path) -> tuple[bool, str]:
    """光写对文件不算数 —— 问解释器自己 ``site-packages`` 在不在 ``sys.path`` 上。"""
    exe = runtime_dir / "python.exe"
    sp = (runtime_dir / "Lib" / "site-packages").resolve()
    code = ("import json,sys,site\n"
            "print('MASTPTH'+json.dumps({'path':[str(p) for p in sys.path],"
            "'has_site':hasattr(site,'ENABLE_USER_SITE')}))\n")
    r = subprocess.run([str(exe), "-c", code], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=60)
    m = re.search(r"MASTPTH(\{.*\})", r.stdout or "")
    if not m:
        return False, f"探测没有输出：{(r.stderr or '')[-400:]}"
    data = json.loads(m.group(1))
    on_path = any(Path(p).resolve() == sp for p in data["path"] if p)
    if not on_path:
        return False, ("site-packages 不在 sys.path 上。sys.path = "
                       + " | ".join(data["path"]))
    return True, ""


# ---------------------------------------------------------------------------
# 3. wheel：下载 → 解包（不用 pip，embeddable 本来就没有）
# ---------------------------------------------------------------------------

def download_wheels(cache: Path) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    _log(f"  pip download → {cache}")
    cmd = [sys.executable, "-m", "pip", "download",
           "-r", str(REQUIREMENTS), "-d", str(cache),
           "--only-binary=:all:",
           "--platform", PLATFORM_TAG,
           "--python-version", "313",
           "--implementation", "cp",
           "--no-deps"]
    r = subprocess.run(cmd, encoding="utf-8", errors="replace",
                       capture_output=True)
    if r.returncode != 0:
        raise BuildError("pip download 失败：\n" + (r.stdout or "") + (r.stderr or ""))
    whls = sorted(cache.glob("*.whl"))
    if not whls:
        raise BuildError(f"{cache} 下一个 .whl 都没有")
    _log(f"  {len(whls)} 个 wheel，共 "
         f"{sum(w.stat().st_size for w in whls) / 1e6:.0f} MB")
    return whls


def install_wheels(whls: list[Path], runtime_dir: Path) -> None:
    sp = runtime_dir / "Lib" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    for w in whls:
        with zipfile.ZipFile(w) as z:
            z.extractall(sp)
    _log(f"  {len(whls)} 个 wheel 解包进 {sp}")


def install_sitecustomize(runtime_dir: Path) -> Path:
    """把 ``pyexec/runtime_sitecustomize.py`` 装成运行时的 ``sitecustomize.py``。

    为什么不能只靠会话目录里那份：``._pth`` 让解释器 isolated，isolated 忽略
    ``PYTHONPATH``，于是会话目录不在 ``sys.path`` 上，``sitecustomize`` 和
    ``_mast_audit`` 都找不到 —— 审计钩子静默不装。开发环境走的是非 isolated 的
    venv，所以这个洞在测试里看不见，只在发出去的那个运行时上存在。

    装进去的那份只做引导（读 ``MAST_PYEXEC_SESSION`` → 加进 ``sys.path`` →
    装钩子），真正的逻辑仍在会话里的 ``_mast_audit.py``，随 MAST 版本更新。
    """
    src = MASTV2 / "mast" / "pyexec" / "runtime_sitecustomize.py"
    if not src.is_file():
        raise BuildError(
            f"找不到 {src} —— 没有它，随包运行时上的审计钩子不会安装，"
            "而 B2（不能覆盖已存在的测量文件）的全部机制就是那个钩子。")
    dst = runtime_dir / "Lib" / "site-packages" / "sitecustomize.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    _log(f"  装入 {dst.relative_to(runtime_dir)}")
    return dst


def verify_sitecustomize(runtime_dir: Path) -> tuple[bool, str]:
    """光拷进去不算数 —— 造一个假会话，看钩子**真的**装上了没有。

    这条是这个构建脚本里最要紧的一次验证：它检的不是文件在不在，是那条
    「env → sys.path → import → install」的链子通不通。
    """
    import tempfile

    exe = runtime_dir / "python.exe"
    audit_src = MASTV2 / "mast" / "pyexec" / "child_audit.py"
    if not audit_src.is_file():
        return False, f"找不到 {audit_src}"
    with tempfile.TemporaryDirectory(prefix="mast_pyrt_") as td:
        sess = Path(td)
        shutil.copy2(audit_src, sess / "_mast_audit.py")
        probe = sess / "probe.py"
        probe.write_text(
            "import json,sys\n"
            "import _mast_audit as A\n"
            "print('MASTHOOK'+json.dumps({'installed':A._INSTALLED,"
            "'on_path':any(p==sys.argv[1] for p in sys.path)}))\n",
            encoding="utf-8")
        env = dict(os.environ)
        env["MAST_PYEXEC_SESSION"] = str(sess)
        env.pop("PYTHONPATH", None)          # 故意不给 —— 就是要验另一条路
        r = subprocess.run([str(exe), "-s", "-B", str(probe), str(sess)],
                           capture_output=True, encoding="utf-8",
                           errors="replace", timeout=120, env=env)
        if r.returncode == 97:
            return False, "子进程以 EXIT_NO_HOOK(97) 退出：" + (r.stderr or "")[-500:]
        m = re.search(r"MASTHOOK(\{.*\})", r.stdout or "")
        if not m:
            return False, ("探测没有输出（rc=%d）：\n%s\n%s"
                           % (r.returncode, (r.stdout or "")[-400:],
                              (r.stderr or "")[-600:]))
        d = json.loads(m.group(1))
        if not d.get("on_path"):
            return False, "会话目录没进 sys.path —— sitecustomize 没跑或没生效"
        if not d.get("installed"):
            return False, "会话目录进了 sys.path，但钩子没装上"
    return True, ""


def strip_runtime(runtime_dir: Path) -> int:
    """删掉测试与头文件。**不动 ``.py``**。"""
    sp = runtime_dir / "Lib" / "site-packages"
    n = 0
    for d in sorted(sp.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and d.name in STRIP_DIR_NAMES:
            n += sum(1 for _ in d.rglob("*") if _.is_file())
            shutil.rmtree(d, ignore_errors=True)
    for f in sp.rglob("*"):
        if f.is_file() and f.suffix in STRIP_SUFFIXES:
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    _log(f"  strip 掉 {n} 个文件（tests/ 与头文件；.py 全部保留）")
    return n


def compile_all(runtime_dir: Path) -> None:
    """用**运行时自己的** python.exe 预编译。

    magic number 要匹配；而且运行时装在 Program Files 下普通用户写不了 ``.pyc``，
    不预编译则每次跑都重新 parse 整个 scipy（实测冷启动差出好几秒）。
    """
    exe = runtime_dir / "python.exe"
    sp = runtime_dir / "Lib" / "site-packages"
    r = subprocess.run(
        [str(exe), "-m", "compileall", "-q", "-j", "0", str(sp)],
        capture_output=True, encoding="utf-8", errors="replace", timeout=1800)
    # compileall 对少数文件报语法错是正常的（py2 残留、模板文件），不算失败；
    # 但**完全跑不起来**要抛。
    if r.returncode not in (0, 1):
        raise BuildError("compileall 失败：\n" + (r.stdout or "") + (r.stderr or ""))
    n = sum(1 for _ in sp.rglob("*.pyc"))
    if n == 0:
        raise BuildError("compileall 之后一个 .pyc 都没有 —— 预编译没生效")
    _log(f"  预编译 {n} 个 .pyc")


# ---------------------------------------------------------------------------
# 4. 自检 —— 不过就让构建死掉
# ---------------------------------------------------------------------------

_SELFTEST_SRC = r'''
import json, sys, os, tempfile
out = {"python": ".".join(str(x) for x in sys.version_info[:3]),
       "pkg": {}, "bad": {}, "leaks": [], "figure": "", "isolated": None}
for m in %r:
    try:
        mod = __import__(m)
        out["pkg"][m] = getattr(mod, "__version__", "?")
    except Exception as e:
        out["bad"][m] = "%%s: %%s" %% (type(e).__name__, e)
# B1：这些必须 import 不到
for m in ("nanonis_spm", "mast"):
    try:
        __import__(m)
        out["leaks"].append(m)
    except Exception:
        pass
# ._pth 存在 ⇒ 解释器默认 isolated
out["isolated"] = bool(sys.flags.isolated) or not sys.flags.no_site
# 画一张图存盘：一次同时证明 Agg 后端、字体栈、写盘三件事
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(np.arange(64).reshape(8, 8))
    ax.set_title("selftest")
    p = os.path.join(tempfile.gettempdir(), "mast_pyruntime_selftest.png")
    fig.savefig(p, dpi=60)
    plt.close(fig)
    out["figure"] = p if os.path.getsize(p) > 0 else "空文件"
except Exception as e:
    out["figure"] = "FAILED %%s: %%s" %% (type(e).__name__, e)
print("MASTSELFTEST" + json.dumps(out))
''' % (tuple(REQUIRED_IMPORTS + OPTIONAL_IMPORTS),)


def selftest(runtime_dir: Path) -> dict:
    exe = runtime_dir / "python.exe"
    if not exe.is_file():
        raise BuildError(f"{exe} 不存在")
    r = subprocess.run([str(exe), "-c", _SELFTEST_SRC], capture_output=True,
                       encoding="utf-8", errors="replace", timeout=300)
    m = re.search(r"MASTSELFTEST(\{.*\})", r.stdout or "", re.S)
    if not m:
        raise BuildError(
            "自检没有输出 —— 运行时起不来。\n"
            "stdout：" + (r.stdout or "")[-800:] + "\n"
            "stderr：" + (r.stderr or "")[-800:])
    d = json.loads(m.group(1))

    problems = []
    if d["bad"]:
        # **不分必需/可选。** 这些包全在 requirements-pyruntime.txt 里，构建刚刚
        # 把它们装进去了 —— 装完 import 不了就是装坏了，不是「缺失」。
        #
        # 第一版把可选包的失败降级成一行 `!` 提示，于是 skimage 和 sklearn 被
        # strip 弄坏之后构建照样返回 0，产物照样发得出去。「装坏了」被折叠成
        # 「没装」，而后者在运行时是允许的 —— 一个合理得没人会去核的报告。
        #
        # 「可选」的语义是**运行时缺了能降级**（RuntimeInfo.missing 如实报告），
        # 不是构建时可以装坏。
        for k, v in sorted(d["bad"].items()):
            tag = "必需包" if k in REQUIRED_IMPORTS else "可选包"
            problems.append(
                f"{tag} {k} import 失败：{v}\n"
                "      （它在 requirements-pyruntime.txt 里、刚装进去过 —— "
                "所以这是「装坏了」不是「没装」）")
    if d["leaks"]:
        problems.append(
            "B1 破了：子进程里能 import " + "、".join(d["leaks"])
            + "。「数据处理 agent 碰不到仪器」这条论断的**唯一**机制就是这些包"
              "不在这个解释器里 —— 而 DP 没有 SafetyGate 也没有 HITL。")
    if d["figure"].startswith("FAILED") or d["figure"] == "空文件":
        problems.append("画图存盘失败：" + d["figure"])
    if not d["python"].startswith("3.13"):
        problems.append(f"解释器版本是 {d['python']}，不是 3.13")
    if problems:
        raise BuildError("运行时自检不通过：\n  - " + "\n  - ".join(problems))
    return d


# ---------------------------------------------------------------------------
# 5. MANIFEST
# ---------------------------------------------------------------------------

def write_manifest(runtime_dir: Path, rep: Report, whls: list[Path]) -> Path:
    files = [p for p in runtime_dir.rglob("*") if p.is_file()]
    rep.n_files = len(files)
    rep.total_mb = round(sum(p.stat().st_size for p in files) / 1e6, 1)
    doc = {
        "schema": 1,
        "python_version": rep.python_version,
        "built_by": f"{platform.node()} / {platform.platform()}",
        "packages": rep.packages,
        "wheels": [{"name": w.name, "sha256": _sha256(w)} for w in whls],
        "n_files": rep.n_files,
        "total_mb": rep.total_mb,
        "pth_ok": rep.pth_ok,
        "isolated": rep.isolated,
        "leaks": rep.leaks,
        "note": ("这套运行时只被 mast.pyexec 起的分析子进程用，MAST 主进程一个都不 "
                 "import。leaks 必须是空的 —— 那是 B1（DP 碰不到仪器）的机制本身。"),
    }
    p = runtime_dir / "MANIFEST.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------

def build(*, skip_download: bool = False) -> Report:
    if not REQUIREMENTS.is_file():
        raise BuildError(f"没有 {REQUIREMENTS}")
    rep = Report(runtime_dir=str(RUNTIME_DIR))

    _log("① embeddable CPython " + PY_VERSION)
    zp = WHEEL_CACHE.parent / f"python-{PY_VERSION}-embed-amd64.zip"
    fetch_embeddable(zp)
    unpack_embeddable(zp, RUNTIME_DIR)

    _log("② 修 ._pth（site-packages + import site）")
    fix_pth(RUNTIME_DIR)

    _log("③ wheel")
    whls = (sorted(WHEEL_CACHE.glob("*.whl")) if skip_download
            else download_wheels(WHEEL_CACHE))
    if not whls:
        raise BuildError(f"{WHEEL_CACHE} 下没有 wheel（--skip-download 要求缓存已就绪）")
    rep.n_wheels = len(whls)
    install_wheels(whls, RUNTIME_DIR)

    _log("④ 验 ._pth 真的生效")
    ok, why = verify_pth(RUNTIME_DIR)
    rep.pth_ok = ok
    if not ok:
        raise BuildError(
            "._pth 没生效：" + why + "\n"
            "  这正是这类部署最常见的失败方式 —— 目录里明明有 numpy，import 却失败。")
    _log("  site-packages 在 sys.path 上 ✓")

    _log("⑤ 装审计钩子引导（sitecustomize）")
    install_sitecustomize(RUNTIME_DIR)
    ok, why = verify_sitecustomize(RUNTIME_DIR)
    if not ok:
        raise BuildError(
            "审计钩子在这个运行时上装不上：" + why + "\n"
            "  没有它，B2（不能覆盖或删除已存在的测量文件）就没有任何机制 —— "
            "而分析进程看起来会完全正常。")
    _log("  假会话验证：钩子装上了 ✓")

    _log("⑥ strip + 预编译")
    strip_runtime(RUNTIME_DIR)
    compile_all(RUNTIME_DIR)

    _log("⑦ 自检")
    d = selftest(RUNTIME_DIR)
    rep.python_version = d["python"]
    rep.packages = d["pkg"]
    rep.isolated = bool(d["isolated"])
    rep.leaks = list(d["leaks"])
    rep.selftest = d
    _log(f"  python {d['python']}；"
         + "、".join(f"{k} {v}" for k, v in sorted(d["pkg"].items())))
    _log(f"  B1 泄漏检查：{d['leaks'] or '干净（nanonis_spm / mast 都 import 不到）'}")
    _log(f"  画图存盘：{d['figure']}")

    _log("⑧ MANIFEST")
    mp = write_manifest(RUNTIME_DIR, rep, whls)
    _log(f"  {mp}（{rep.n_files} 个文件，{rep.total_mb} MB）")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建 DP 分析运行时")
    ap.add_argument("--skip-download", action="store_true",
                    help="用 artifacts/pyruntime-wheels/ 里已有的 wheel")
    ap.add_argument("--selftest-only", action="store_true",
                    help="只对已有的 MASTv2/pyruntime/ 跑一次自检")
    args = ap.parse_args(argv)

    try:
        if args.selftest_only:
            ok, why = verify_pth(RUNTIME_DIR)
            if not ok:
                raise BuildError("._pth 没生效：" + why)
            # 钩子这条也要验。第一版只验 ._pth 和六个库 —— 而 mast2_build.ps1
            # Step 4.7 就靠这个命令判断运行时好不好，于是「sitecustomize 装没装
            # 上」全靠 PS1 那边 Test-Path 一个文件。文件在不在 ≠ 那条
            # env → sys.path → import → install 的链子通不通，而后者才是 B2 的
            # 全部机制。
            ok, why = verify_sitecustomize(RUNTIME_DIR)
            if not ok:
                raise BuildError("审计钩子装不上：" + why)
            d = selftest(RUNTIME_DIR)
            _log("自检通过：python %s；%s；泄漏 %s；审计钩子装得上"
                 % (d["python"],
                    "、".join(f"{k} {v}" for k, v in sorted(d["pkg"].items())),
                    d["leaks"] or "无"))
            return 0
        rep = build(skip_download=args.skip_download)
    except BuildError as exc:
        print("\n构建失败：\n" + str(exc), file=sys.stderr)
        return 2
    _log(f"\n完成：{rep.runtime_dir}（{rep.total_mb} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
