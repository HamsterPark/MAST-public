"""起子进程跑一段分析代码 —— 从零构造环境，超时能真杀。

为什么 env 是**从零构造**而不是 ``os.environ.copy()``
===================================================
拷贝当前环境会把三类东西带进去，每一类都出过事：

* ``_MEI*`` / ``_PYI*`` —— PyInstaller 注入的。带着它们起子进程会
  ``ERROR_BAD_EXE_FORMAT`` 直接死，而脚本本身完全没错
  （``mast2_launcher.py:1735-1747`` 的血泪记录）。
* ``PYTHONPATH`` / ``PYTHONHOME`` —— 会让分析解释器去 MAST 的 ``_internal``
  里找 stdlib。⚠️ 而且 ``_internal/nanonis_spm/__init__.py`` **是真实存在的文件**
  —— 把 ``_internal`` 放进子进程的 path 就等于亲手把仪器协议递过去。
* LAN 凭据、``MAST_*`` —— 分析进程没有理由知道它们。

白名单只留 Windows 加载 DLL 真正需要的那几个，其余全是我们显式设的。

为什么 stdout/stderr 用**文件**而不是 PIPE
========================================
PIPE 必须 ``communicate()``，否则缓冲区满了就死锁；而 ``communicate()`` 和「超时
后还想看看它输出到哪儿了」是冲突的。用文件三件好处一起拿：不可能死锁、**超时后
仍有部分输出**、模型可以用一行 Python 重读整个日志。

不阻塞 graph 节点
=================
这个模块被 ``agents/data_processing/tools.py`` 同步调用，带硬超时。理由和
``tools.py:1050-1052`` 那段注释一样：受控超时可以住在 ``tools.py``，但**不能**住
在任何 ``agents/**/graph.py``（hook ``buffer_no_blocking.py:24-26`` 禁止）。
LangChain 会把同步 ``@tool`` 丢进线程池，所以 ``ainvoke`` 不会卡事件循环。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 300.0        # 5 min
MAX_TIMEOUT_S = 3600.0           # 1 h —— 上限是拦失控的，不是当预算用的
STDOUT_TAIL = 4000
STDERR_TAIL = 4000               # 取**尾部**：traceback 在末尾

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

#: 这些是 Windows 加载 DLL / 找系统目录真正需要的。别的一律不继承。
_INHERIT = ("SystemRoot", "windir", "COMSPEC", "PATHEXT",
            "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "TEMP", "TMP")


@dataclass
class ExecResult:
    returncode: int | None            # None = 超时被杀
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    killed_reason: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    truncated: bool = False
    limits: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _omp_threads() -> str:
    """给 OpenMP 留一半核。

    **这是一台驱动 STM 的实验室机器。** 无界的 scipy/OpenBLAS 线程池会占满每个
    核，把 instrument_control 的监控循环饿死 —— 那不是「分析慢一点」，是仪器侧
    的读数迟到。但也不能钉死成 2：16 核机器上那是把分析能力砍到 1/8。
    """
    raw = os.environ.get("MAST_PYEXEC_THREADS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return raw
    try:
        return str(max(2, (os.cpu_count() or 4) // 2))
    except Exception:  # noqa: BLE001
        return "2"


def build_env(session, *, scan_dirs: str = "", blocked_ports: str = "") -> dict:
    """子进程的完整环境。见模块 docstring 里那三类被刻意剔掉的东西。"""
    env = {k: os.environ[k] for k in _INHERIT if k in os.environ}

    # PATH：只要系统目录。分析解释器自带 vcruntime，不需要别的。
    sysroot = env.get("SystemRoot") or r"C:\Windows"
    if sys.platform == "win32":
        env["PATH"] = os.pathsep.join(
            [str(Path(sysroot) / "System32"), sysroot,
             str(Path(sysroot) / "System32" / "Wbem")])
    else:
        env["PATH"] = "/usr/bin:/bin:/usr/local/bin"

    env.update({
        # 会话目录进 path：sitecustomize（审计钩子）+ mastdata + mast 子集都在这里。
        # 这条同时决定了钩子装不装得上 —— 所以命令行**不能**用 -I（它隐含 -E，
        # 会把这个变量整个忽略，钩子就静默不装了）。
        #
        # ⚠️ **但 PYTHONPATH 一条不够。** 随包运行时带 python313._pth，而 ._pth 的
        # 存在让解释器默认 isolated —— isolated 照样忽略 PYTHONPATH。同一个陷阱、
        # 第二次、从另一条路进来，而这次更坏：开发环境走非 isolated 的 venv，
        # 测试全绿，只有要发给用户的那个运行时上钩子会静默不装。
        # 所以会话目录**同时**用一个自定义变量送（isolated 不碰这类变量），由
        # 运行时里的 sitecustomize 接手。两条路都在，走哪条取决于解释器；
        # _mast_audit.install() 幂等，重复装无害。
        "PYTHONPATH": str(session.root),
        "MAST_PYEXEC_SESSION": str(session.root),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",     # 运行时目录可能在 Program Files 下只读
        "TEMP": str(session.tmp_dir),
        "TMP": str(session.tmp_dir),
        # matplotlib：无头必须 Agg，否则可能选 GUI 后端然后挂死。
        "MPLBACKEND": "Agg",
        # 字体缓存**共享**：第一次构建要 ~10 s，必须一辈子只发生一次，
        # 不能每个会话重来。
        "MPLCONFIGDIR": str(session.root.parent / ".mplcache"),
        "OMP_NUM_THREADS": _omp_threads(),
        "MKL_NUM_THREADS": _omp_threads(),
        "OPENBLAS_NUM_THREADS": _omp_threads(),
        "NUMEXPR_NUM_THREADS": _omp_threads(),
        # 审计钩子的配置。子进程在跑用户代码**之前**就把它们抓进闭包，
        # 脚本改 os.environ 无效。
        "MAST_PYEXEC_SESSION": str(session.root),
        "MAST_PYEXEC_BLOCKED_PORTS": blocked_ports,
        "MAST_PYEXEC_SCAN_DIRS": scan_dirs,
    })
    return env


def nanonis_ports() -> str:
    """从 config 读真实端口，**不硬编码** —— 那几个数是可配的。"""
    try:
        from mast.config import Config
        cfg = Config()
        ports = getattr(getattr(cfg, "nanonis", None), "ports", None)
        if ports:
            return ",".join(str(int(p)) for p in ports)
    except Exception:  # noqa: BLE001
        pass
    return "6501,6502,6503,6504"


def _tail(path: Path, limit: int) -> tuple[str, bool]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", False
    if len(raw) <= limit:
        return raw, False
    return raw[-limit:], True


def run(session, script_rel: str, *, runtime, timeout_s: float = DEFAULT_TIMEOUT_S,
        memory_bytes: int = 0) -> ExecResult:
    """跑 ``session/<script_rel>``，带硬超时和 Job Object。"""
    from mast.pyexec.jobobject import JobBox
    from mast.pyexec.staging import scan_dirs_for_env

    timeout_s = max(1.0, min(float(timeout_s or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))
    script = session.root / script_rel
    stem = Path(script_rel).stem
    out_path = session.logs_dir / f"{stem}.out"
    err_path = session.logs_dir / f"{stem}.err"
    session.logs_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(session, scan_dirs=scan_dirs_for_env(),
                    blocked_ports=nanonis_ports())
    # -s 禁 user site。**不用 -I** —— 见 build_env 里 PYTHONPATH 那条注释。
    cmd = [str(runtime.exe), "-s", "-B", "-X", "utf8", str(script)]

    flags = 0
    if sys.platform == "win32":
        flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP

    box = JobBox(memory_bytes=memory_bytes) if memory_bytes else JobBox()
    t0 = time.time()
    rc: int | None = None
    timed_out = False
    killed_reason = ""

    try:
        with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
            proc = subprocess.Popen(
                cmd, cwd=str(session.root), env=env, stdin=subprocess.DEVNULL,
                stdout=fo, stderr=fe, close_fds=True, creationflags=flags)
            box.assign(proc.pid)
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                killed_reason = f"超过 {timeout_s:g} 秒墙钟上限"
                box.kill(proc.pid)          # 整棵树，含 multiprocessing 子进程
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("pyexec: 进程 %d 在 kill 之后仍未退出", proc.pid)
                rc = None
    except OSError as exc:
        box.close()
        return ExecResult(returncode=-1, stderr=f"无法启动分析进程：{exc}",
                          duration_s=time.time() - t0,
                          killed_reason="spawn failed")
    finally:
        box.close()

    dur = time.time() - t0
    so, t1 = _tail(out_path, STDOUT_TAIL)
    se, t2 = _tail(err_path, STDERR_TAIL)

    if not timed_out and rc not in (0, None) and not se.strip():
        # 非零退出但 stderr 空 —— 十有八九是被 Job Object 的内存上限干掉的。
        # 说出来，否则表现成「脚本莫名其妙失败了」。
        killed_reason = (f"进程以 {rc} 退出且没有 traceback —— 很可能撞了内存上限"
                         f"（{box.memory_bytes / 1024**3:.1f} GiB）")

    return ExecResult(
        returncode=rc, stdout=so, stderr=se, duration_s=dur,
        timed_out=timed_out, killed_reason=killed_reason,
        stdout_path=str(out_path), stderr_path=str(err_path),
        truncated=t1 or t2,
        limits={"timeout_s": timeout_s,
                "memory_bytes": box.memory_bytes,
                "job_object": box.supported,
                "threads": env.get("OMP_NUM_THREADS", "")},
    )


__all__ = ["DEFAULT_TIMEOUT_S", "MAX_TIMEOUT_S", "ExecResult", "build_env",
           "nanonis_ports", "run"]
