"""找到一个能跑数据分析的 Python 解释器 —— 找不到就说清楚，绝不抛。

形状照抄 ``mast/net/tailscale.py:41-73`` 的 ``find_cli()``：PATH 优先、平台默认
位置兜底、**返回 None 而不是抛异常**，并且把「查了哪些地方」一起带回来 ——
「没找到」必须是一句能照着做的话，不是一个空值。

探测顺序
========

1. **bundled** —— 随安装包发的 CPython 3.13（``<exe目录>/MASTv2/pyruntime/``）。
   env ``MAST_PYRUNTIME_DIR`` 可覆盖。
2. **dev** —— 开发环境下就是 ``sys.executable``（``.venv-v2-py313`` 已有全部六库）。
   这条让 P1 不依赖打包也能完整跑，隔离保证因此**每次 CI 都被真实测到**，
   而不是等一个 335 MB 的构建。
3. **system** —— ``py -0p`` / ``shutil.which`` / 注册表里已装的 3.13。

每个候选都要**真的跑一遍** ``import numpy, scipy, matplotlib``。一个没有科学栈的
裸解释器对数据分析毫无用处，当作「没找到」处理，并把原因说出来。

一条路径陷阱（本仓踩过）
======================
frozen 时用的是 **exe 所在目录**，不是 ``project_root()``。后者在运行时被启动器
指向用户数据目录（``mast2_launcher.py`` 导出 ``MAST2_PROJECT_ROOT``），而随包发
的资产在 exe 旁边。搞反了就是「视觉模型加载超时」那个 bug 的复刻
（``vision/vigil_backend.py:165-172``）。所以这里抄的是
``mast/knowledge/paths.py:80-88`` 的 ``base_dir()``。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 没有这三个就不算一个可用的分析运行时 —— 缺任何一个都干不了本职工作。
REQUIRED_PACKAGES: tuple[str, ...] = ("numpy", "scipy", "matplotlib")
# 缺了仍然可用，但要**如实报告**：让 agent 撞上去才发现边界是在浪费它的回合。
OPTIONAL_PACKAGES: tuple[str, ...] = ("pandas", "skimage", "sklearn")

_PROBE_TIMEOUT_S = 30.0
_MIN_VERSION = (3, 11)

# 打包运行时的目录名（相对 base_dir）。与 mast2_build.ps1 的 Step 4.7 落点一致。
_BUNDLED_SUBPATH = ("MASTv2", "pyruntime")


@dataclass(frozen=True)
class RuntimeInfo:
    """一个已验证可用的解释器。

    ``origin`` 是给用户看的一句人话（这个解释器是哪来的），不是给代码判分支的；
    要分支用 ``kind``。
    """

    exe: Path
    kind: str                       # "bundled" | "dev" | "system"
    version: str                    # "3.13.13"
    origin: str
    packages: dict[str, str] = field(default_factory=dict)   # 名 -> 版本
    missing: tuple[str, ...] = ()   # OPTIONAL_PACKAGES 里没装上的
    leaks: tuple[str, ...] = ()     # 这个解释器里**能**import 到的 MAST/硬件包

    @property
    def isolated(self) -> bool:
        """B1 此刻成不成立 —— 一个可观察的事实，不是一个假设。

        ``True`` 只在这个解释器里 **既没有 nanonis_spm 也没有 mast** 时成立。
        随包发的运行时天然满足；但 **dev 回退（``sys.executable``）几乎必然不
        满足** —— 那就是 MAST 自己的 venv，里面当然装着 nanonis_spm。

        这一条必须被报出来而不是被假设：漏报的后果是「我们以为它碰不到仪器」，
        而那正是这整个功能敢存在的理由。
        """
        return not self.leaks

    @property
    def summary(self) -> str:
        libs = "、".join(f"{k} {v}" for k, v in sorted(self.packages.items()))
        s = f"Python {self.version}（{self.origin}）：{libs}"
        if self.missing:
            s += f"；**未安装**：{'、'.join(self.missing)}"
        if self.leaks:
            s += (f"；⚠️ 该解释器可 import {'、'.join(self.leaks)} —— "
                  "「进不了仪器」这条在本运行时上不成立")
        return s


@dataclass(frozen=True)
class ProbeReport:
    """一次探测的全部结果 —— 包括失败的那些。

    ``find_runtime()`` 返回 None 时，这份报告就是「我查了哪些地方、各自为什么
    不行」的答案。没有它，「找不到」是一个死胡同。
    """

    runtime: RuntimeInfo | None
    tried: tuple[tuple[str, str], ...] = ()   # (候选路径, 失败原因)

    def why_not(self) -> str:
        if self.runtime is not None:
            return ""
        if not self.tried:
            return ("未找到可用的分析 Python：既没有随包运行时，"
                    "也没有在 PATH / py launcher / 注册表里找到 Python 3.13。")
        lines = ["未找到可用的分析 Python。已查："]
        lines += [f"  · {p} —— {why}" for p, why in self.tried]
        lines.append("请安装 Python 3.13 并 pip install numpy scipy matplotlib，"
                     "或安装带分析运行时的 MAST 版本。")
        return "\n".join(lines)


# ── 缓存 ─────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_cached: ProbeReport | None = None


def reset_runtime_cache() -> None:
    """丢弃缓存（测试用，或运行时被重新安装之后）。"""
    global _cached
    with _lock:
        _cached = None


# ── 路径 ─────────────────────────────────────────────────────────────────
def _base_dir() -> Path:
    """随包资产的 base。见模块 docstring 里那条路径陷阱。

    **两条分支都不能用 ``project_root()``。** frozen 那条的理由在 docstring 里；
    dev 这条是同一个陷阱的另一半 —— ``MAST2_PROJECT_ROOT`` 一被设（启动器会设、
    测试会设、任何人想把数据放别处都会设），``project_root()`` 就指到数据目录，
    而 ``MASTv2/pyruntime`` 在**仓库**里。

    2026-08-20 实测撞到：一个设了 ``MAST2_PROJECT_ROOT`` 的端到端脚本悄悄落回
    dev 解释器，日志里如实写着「B1 不成立」—— 而如果没有那行警告，它看起来
    只是「跑通了」。

    dev 下从包自己的位置推：``mast/__init__.py`` 的 parents[2] 就是仓库根
    （``<repo>/MASTv2/mast/``）。这个推导不依赖任何环境变量。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    import mast
    return Path(mast.__file__).resolve().parents[2]


def bundled_dir() -> Path:
    raw = os.environ.get("MAST_PYRUNTIME_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return _base_dir().joinpath(*_BUNDLED_SUBPATH)


def _exe_name() -> str:
    return "python.exe" if sys.platform == "win32" else "python3"


# ── 探测一个候选 ─────────────────────────────────────────────────────────
# B1 的自查名单：这些在子进程里**必须**是 import 不到的。
# 「碰不到仪器」不是靠拦截，是靠那些包根本不在这个解释器里。
LEAK_MODULES: tuple[str, ...] = ("nanonis_spm", "mast")

_PROBE_SRC = (
    "import json,sys\n"
    "d={'python':'.'.join(str(x) for x in sys.version_info[:3]),'pkg':{},'bad':{},"
    "'leaks':[]}\n"
    "for m in %r:\n"
    "    try:\n"
    "        d['pkg'][m]=getattr(__import__(m),'__version__','?')\n"
    "    except Exception as e:\n"
    "        d['bad'][m]='%%s: %%s'%%(type(e).__name__,e)\n"
    "for m in %r:\n"
    "    try:\n"
    "        __import__(m)\n"
    "        d['leaks'].append(m)\n"
    "    except Exception:\n"
    "        pass\n"
    "print('MASTPROBE'+json.dumps(d))\n"
) % (tuple(REQUIRED_PACKAGES + OPTIONAL_PACKAGES), tuple(LEAK_MODULES))


def _run_kwargs() -> dict:
    kw: dict = {
        "capture_output": True,
        "timeout": _PROBE_TIMEOUT_S,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if sys.platform == "win32":
        # 无头服务下不要闪控制台窗口（同 tailscale._run_kwargs）。
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kw


def _probe(exe: Path) -> tuple[dict | None, str]:
    """跑一次探测。返回 ``(结果 dict | None, 失败原因)``。永不抛。"""
    if not exe.is_file():
        return None, "文件不存在"
    try:
        # -I（isolated）在这里就用上：探测要看到的是这个解释器**自己**的环境，
        # 不是继承来的 PYTHONPATH 让它假装有 numpy。
        proc = subprocess.run([str(exe), "-I", "-c", _PROBE_SRC], **_run_kwargs())
    except subprocess.TimeoutExpired:
        return None, f"探测超时（{_PROBE_TIMEOUT_S:g}s）"
    except OSError as exc:
        return None, f"无法调用：{exc}"

    raw = (proc.stdout or "")
    marker = raw.find("MASTPROBE")
    if marker < 0:
        tail = (proc.stderr or raw or "").strip().replace("\n", " ")[-200:]
        return None, f"探测无输出（exit {proc.returncode}）{(' — ' + tail) if tail else ''}"
    try:
        data = json.loads(raw[marker + len("MASTPROBE"):].strip())
    except Exception as exc:  # noqa: BLE001
        return None, f"探测输出不是 JSON：{exc}"

    ver = tuple(int(x) for x in str(data.get("python", "0")).split(".")[:2])
    if ver < _MIN_VERSION:
        return None, (f"Python {data.get('python')} 太旧（需要 "
                      f"{'.'.join(str(v) for v in _MIN_VERSION)}+）")
    missing_required = [m for m in REQUIRED_PACKAGES if m not in data.get("pkg", {})]
    if missing_required:
        return None, ("缺少必需的库：" + "、".join(missing_required) +
                      "（一个没有科学栈的解释器做不了数据分析）")
    return data, ""


def _make(exe: Path, kind: str, origin: str, data: dict) -> RuntimeInfo:
    pkg = dict(data.get("pkg", {}))
    return RuntimeInfo(
        exe=exe.resolve(), kind=kind, version=str(data.get("python", "?")),
        origin=origin, packages=pkg,
        missing=tuple(m for m in OPTIONAL_PACKAGES if m not in pkg),
        leaks=tuple(str(m) for m in (data.get("leaks") or [])),
    )


# ── 候选来源 ─────────────────────────────────────────────────────────────
def _system_candidates() -> list[tuple[Path, str]]:
    """(路径, 来源说明)。仓库里此前没有任何 Python 探测代码，所以保守着写。"""
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()

    def _add(raw, origin: str) -> None:
        if not raw:
            return
        p = Path(str(raw))
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append((p, origin))

    if sys.platform == "win32":
        # py launcher（PEP 397）能列出所有已装解释器 —— 比 PATH 全得多。
        py = shutil.which("py")
        if py:
            try:
                proc = subprocess.run([py, "-0p"], **_run_kwargs())
                for line in (proc.stdout or "").splitlines():
                    line = line.strip()
                    if not line.startswith("-V:3.1"):
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        _add(parts[1].strip().lstrip("*").strip(),
                             f"py launcher（{parts[0]}）")
            except Exception as exc:  # noqa: BLE001
                logger.debug("py -0p failed: %s", exc)

        try:
            import winreg  # noqa: PLC0415 — Windows-only
            for hive, hname in ((winreg.HKEY_CURRENT_USER, "HKCU"),
                                (winreg.HKEY_LOCAL_MACHINE, "HKLM")):
                for tag in ("3.13", "3.13-32", "3.13-arm64", "3.12"):
                    try:
                        key = winreg.OpenKey(
                            hive, rf"SOFTWARE\Python\PythonCore\{tag}\InstallPath",
                            0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY)
                    except OSError:
                        continue
                    with key:
                        try:
                            exe_path, _ = winreg.QueryValueEx(key, "ExecutablePath")
                            _add(exe_path, f"注册表 {hname} Python {tag}")
                            continue
                        except OSError:
                            pass
                        try:
                            root, _ = winreg.QueryValueEx(key, "")
                            _add(Path(root) / "python.exe",
                                 f"注册表 {hname} Python {tag}")
                        except OSError:
                            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("winreg probe failed: %s", exc)

    for name in ("python3.13", "python3.12", "python3", "python"):
        _add(shutil.which(name), f"PATH 上的 {name}")
    return out


# ── 主入口 ───────────────────────────────────────────────────────────────
def probe_runtime(*, refresh: bool = False) -> ProbeReport:
    """完整探测（含失败原因）。结果按进程缓存。**永不抛。**"""
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached

    tried: list[tuple[str, str]] = []

    def _try(exe: Path, kind: str, origin: str) -> RuntimeInfo | None:
        data, why = _probe(exe)
        if data is None:
            tried.append((str(exe), why))
            return None
        return _make(exe, kind, origin, data)

    found: RuntimeInfo | None = None

    # 1) 随包运行时
    bexe = bundled_dir() / _exe_name()
    if bexe.is_file():
        found = _try(bexe, "bundled", "随 MAST 安装包附带")

    # 2) 开发环境：本进程自己的解释器
    if found is None and not getattr(sys, "frozen", False):
        found = _try(Path(sys.executable), "dev", "开发环境（MAST 自己的 venv）")

    # 3) 系统上已装的
    if found is None:
        for exe, origin in _system_candidates():
            found = _try(exe, "system", origin)
            if found is not None:
                break

    report = ProbeReport(runtime=found, tried=tuple(tried))
    if found is not None:
        logger.info("pyexec runtime: %s", found.summary)
        if not found.isolated:
            # 大声，因为这条决定了「DP 碰不到仪器」这句话现在算不算数。
            logger.warning(
                "pyexec: 当前分析运行时（%s）可以 import %s —— **B1 不成立**。"
                "随包发的运行时不会有这个问题；这条路径只应出现在开发/CI，"
                "或者用户机器上缺少随包运行时的时候。",
                found.kind, "、".join(found.leaks))
        if found.missing:
            logger.info("pyexec runtime 缺少可选库：%s（脚本里 import 会失败，"
                        "这一点必须写进给 agent 的说明里）",
                        "、".join(found.missing))
    else:
        logger.warning("pyexec: %s", report.why_not())
    with _lock:
        _cached = report
    return report


def find_runtime(*, refresh: bool = False) -> RuntimeInfo | None:
    """可用的解释器，没有就 None。**永不抛**（``tailscale.find_cli`` 的契约）。"""
    return probe_runtime(refresh=refresh).runtime


# ── 自检 ─────────────────────────────────────────────────────────────────
# `_probe` 回答「这个解释器有没有那些包」。自检要回答的是**更实际**的一个问题：
# 「用它跑一次真正的分析，会不会成」。两者差在三样东西上，而这三样恰恰是最容易
# 在用户机器上第一次才暴露的：
#   · matplotlib 的 Agg 后端能不能出图（选到 GUI 后端 = 无头挂死）
#   · 字体栈在不在（缺字体时 savefig 会抛，而不是画出没有标签的图）
#   · 那个进程有没有权限写盘
# 所以自检画一张图存一次盘 —— 一次同时证明这三件事。

_SELFTEST_SRC = (
    "import json,os,sys,tempfile\n"
    "d={'python':'.'.join(str(x) for x in sys.version_info[:3]),'pkg':{},"
    "'bad':{},'leaks':[],'figure':'','isolated':None}\n"
    "for m in %r:\n"
    "    try:\n"
    "        d['pkg'][m]=getattr(__import__(m),'__version__','?')\n"
    "    except Exception as e:\n"
    "        d['bad'][m]='%%s: %%s'%%(type(e).__name__,e)\n"
    "for m in %r:\n"
    "    try:\n"
    "        __import__(m)\n"
    "        d['leaks'].append(m)\n"
    "    except Exception:\n"
    "        pass\n"
    "d['isolated']=bool(sys.flags.isolated) or not sys.flags.no_site\n"
    "try:\n"
    "    import matplotlib\n"
    "    matplotlib.use('Agg')\n"
    "    import matplotlib.pyplot as plt, numpy as np\n"
    "    f,ax=plt.subplots(figsize=(2,2))\n"
    "    ax.imshow(np.arange(64).reshape(8,8)); ax.set_title('selftest')\n"
    "    p=os.path.join(tempfile.gettempdir(),'mast_pyruntime_selftest.png')\n"
    "    f.savefig(p,dpi=60); plt.close(f)\n"
    "    d['figure']=p if os.path.getsize(p)>0 else 'EMPTY'\n"
    "except Exception as e:\n"
    "    d['figure']='FAILED %%s: %%s'%%(type(e).__name__,e)\n"
    "print('MASTSELFTEST'+json.dumps(d))\n"
) % (tuple(REQUIRED_PACKAGES + OPTIONAL_PACKAGES), tuple(LEAK_MODULES))

_SELFTEST_TIMEOUT_S = 180.0
_selftest_cache: dict = {}


@dataclass(frozen=True)
class SelftestResult:
    """一次自检。``ok=False`` 时 ``stderr`` 逐字保留 —— 一句「不可用」帮不了任何人。"""

    ok: bool = False
    exe: str = ""
    kind: str = ""
    python: str = ""
    packages: dict = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    leaks: tuple[str, ...] = ()
    figure: str = ""
    isolated: bool = False
    problems: tuple[str, ...] = ()
    stderr: str = ""

    def describe(self) -> str:
        if self.ok:
            libs = "、".join(f"{k} {v}" for k, v in sorted(self.packages.items()))
            tail = ("" if self.isolated else
                    "；⚠️ 这个解释器不是隔离的，B1 不成立（见日志）")
            return f"分析运行时可用（{self.kind}，python {self.python}）：{libs}{tail}"
        return "分析运行时不可用：" + "；".join(self.problems or ("原因不明",))

    def as_dict(self) -> dict:
        return {"ok": self.ok, "exe": self.exe, "kind": self.kind,
                "python": self.python, "packages": dict(self.packages),
                "missing": list(self.missing), "leaks": list(self.leaks),
                "figure": self.figure, "isolated": self.isolated,
                "problems": list(self.problems), "stderr": self.stderr}


def selftest(rt: "RuntimeInfo | None" = None, *,
             refresh: bool = False) -> SelftestResult:
    """跑一次真分析（import 六库 + Agg 画图 + 存盘）。**永不抛。**

    结果按 ``(exe, mtime)`` 进程内缓存：一次约 1.5 s，每次 ``py_run`` 都跑一遍
    是白花的。``refresh=True`` 强制重跑。

    ``rt=None`` 时自己去 :func:`find_runtime` 找。
    """
    if rt is None:
        rt = find_runtime()
    if rt is None:
        return SelftestResult(problems=(probe_runtime().why_not(),))

    exe = Path(rt.exe)
    try:
        key = (str(exe), exe.stat().st_mtime_ns)
    except OSError as exc:
        return SelftestResult(exe=str(exe), kind=rt.kind,
                              problems=(f"解释器不可读：{exc}",))
    with _lock:
        hit = _selftest_cache.get(key)
    if hit is not None and not refresh:
        return hit

    try:
        proc = subprocess.run([str(exe), "-c", _SELFTEST_SRC],
                              capture_output=True, timeout=_SELFTEST_TIMEOUT_S,
                              encoding="utf-8", errors="replace",
                              **({"creationflags": getattr(
                                  subprocess, "CREATE_NO_WINDOW", 0)}
                                 if sys.platform == "win32" else {}))
    except subprocess.TimeoutExpired:
        res = SelftestResult(exe=str(exe), kind=rt.kind,
                             problems=(f"自检超时（{_SELFTEST_TIMEOUT_S:g}s）—— "
                                       "多半是 matplotlib 选到了 GUI 后端在等窗口",))
        with _lock:
            _selftest_cache[key] = res
        return res
    except OSError as exc:
        return SelftestResult(exe=str(exe), kind=rt.kind,
                              problems=(f"起不来：{exc}",))

    err = (proc.stderr or "")[-4000:]
    m = re.search(r"MASTSELFTEST(\{.*\})", proc.stdout or "", re.S)
    if not m:
        res = SelftestResult(
            exe=str(exe), kind=rt.kind, stderr=err,
            problems=("自检脚本没有输出 —— 解释器本身起不来。"
                      "完整 stderr 见下（**不要**只报「不可用」）：",))
        with _lock:
            _selftest_cache[key] = res
        return res

    try:
        d = json.loads(m.group(1))
    except ValueError as exc:
        res = SelftestResult(exe=str(exe), kind=rt.kind, stderr=err,
                             problems=(f"自检输出不是 JSON：{exc}",))
        with _lock:
            _selftest_cache[key] = res
        return res

    problems: list[str] = []
    bad = d.get("bad") or {}
    for name in REQUIRED_PACKAGES:
        if name in bad:
            problems.append(f"必需库 {name} import 失败：{bad[name]}")
    fig = str(d.get("figure") or "")
    if fig.startswith("FAILED") or fig == "EMPTY":
        problems.append(
            "画图存盘失败：" + fig +
            "（这一步同时在验 Agg 后端、字体栈、写盘权限三件事）")

    res = SelftestResult(
        ok=not problems, exe=str(exe), kind=rt.kind,
        python=str(d.get("python") or ""), packages=dict(d.get("pkg") or {}),
        missing=tuple(m2 for m2 in OPTIONAL_PACKAGES
                      if m2 not in (d.get("pkg") or {})),
        leaks=tuple(d.get("leaks") or ()), figure=fig,
        isolated=bool(d.get("isolated")), problems=tuple(problems), stderr=err)
    with _lock:
        _selftest_cache[key] = res
    if res.ok:
        logger.info("pyexec 自检通过：%s", res.describe())
    else:
        # stderr 逐字进日志。这个功能最没用的形态就是一句「分析运行时不可用」。
        logger.error("pyexec 自检不通过：%s\nstderr：\n%s",
                     "；".join(res.problems), res.stderr)
    return res


def reset_selftest_cache() -> None:
    with _lock:
        _selftest_cache.clear()
