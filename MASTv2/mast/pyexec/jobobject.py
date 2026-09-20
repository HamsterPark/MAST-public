"""Windows Job Object —— 让「能停下来」成为一个事实而不是一个希望。

**它在这里的角色是能力的前提，不是限制。** 因为整棵进程树被关在一个 job 里、
而且 ``KILL_ON_JOB_CLOSE`` 保证父进程崩了 OS 也会收干净，我们才敢让分析脚本
自由地 ``multiprocessing`` 并行。没有它就只能禁子进程 —— 那会挡掉「200 张图
并行处理」这种真需求。

它同时修掉一个真实的退步：``agents/data_processing/tools.py`` 的
``_maybe_set_memory_rlimit()`` 在 Windows 上**直接 return**（POSIX-only
``resource`` 模块），所以现有的 numpy 沙箱在真机上根本没有内存上限，而墙钟超时
杀不掉一个 CPython 线程。出进程 + Job Object 才让「资源可控」这句话成真。

上限给得很松（默认物理内存 50%、16 个进程）：上限是拦失控的，不是当预算用的。

已知竞态（写下来，不假装是零）
============================
``Popen`` 返回到 ``AssignProcessToJobObject`` 之间有几微秒。如果子进程恰好在那
个窗口里 spawn 了孙进程，那个孙进程会逃出 job。彻底消除要 ``CREATE_SUSPENDED``
+ ``ResumeThread``，但 ``subprocess.Popen`` 在 Windows 上会关掉主线程句柄，得
改用 ctypes 直接 ``CreateProcessW`` —— 不值得。兜底是 kill 时先用 ``psutil``
扫一遍进程树。威胁模型是「脚本写错了」，而一个写错的脚本不会精确命中那个窗口。

非 Windows 上整个模块降级成同接口的 no-op，Linux CI 照常跑（只跳内存那条）。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags
_LIMIT_ACTIVE_PROCESS = 0x00000008
_LIMIT_PROCESS_MEMORY = 0x00000100
_LIMIT_JOB_MEMORY = 0x00000200
_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

_JobObjectExtendedLimitInformation = 9

DEFAULT_ACTIVE_PROCESS_LIMIT = 16      # 要让 multiprocessing 用得起来
DEFAULT_MEMORY_FRACTION = 0.5          # 物理内存的一半；上限，不是预算


def _default_memory_bytes() -> int:
    """物理内存的一半，拿不到就退到一个宽松的常数。"""
    try:
        import psutil
        return int(psutil.virtual_memory().total * DEFAULT_MEMORY_FRACTION)
    except Exception:  # noqa: BLE001
        return 8 * 1024 ** 3


if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.restype = wintypes.BOOL
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]

    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001


@dataclass
class JobBox:
    """一个进程盒子。``supported=False`` 时所有方法是安全的 no-op。"""

    memory_bytes: int = 0
    active_process_limit: int = DEFAULT_ACTIVE_PROCESS_LIMIT
    _handle: int | None = None
    _closed: bool = False

    supported: bool = IS_WINDOWS
    reason: str = ""          # 不支持/建不起来时的人话原因

    def __post_init__(self) -> None:
        if not self.memory_bytes:
            self.memory_bytes = _default_memory_bytes()
        if not IS_WINDOWS:
            self.supported = False
            self.reason = f"{sys.platform} 上没有 Job Object（超时仍会杀主进程）"
            return
        try:
            h = _k32.CreateJobObjectW(None, None)
            if not h:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                _LIMIT_KILL_ON_JOB_CLOSE | _LIMIT_PROCESS_MEMORY
                | _LIMIT_JOB_MEMORY | _LIMIT_ACTIVE_PROCESS)
            info.BasicLimitInformation.ActiveProcessLimit = int(
                self.active_process_limit)
            info.ProcessMemoryLimit = int(self.memory_bytes)
            info.JobMemoryLimit = int(self.memory_bytes)
            ok = _k32.SetInformationJobObject(
                h, _JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info))
            if not ok:
                err = ctypes.get_last_error()
                _k32.CloseHandle(h)
                raise OSError(err, "SetInformationJobObject failed")
            self._handle = int(h)
            self.supported = True
        except Exception as exc:  # noqa: BLE001 — 建不起来就降级，不能挡住分析
            self.supported = False
            self.reason = f"Job Object 建立失败：{exc}"
            logger.warning("pyexec: %s —— 内存上限本次不生效（超时仍会杀进程树）",
                           self.reason)

    # ── 生命周期 ─────────────────────────────────────────────────────
    def assign(self, pid: int) -> bool:
        """把一个进程（和它此后的所有子孙）关进盒子。"""
        if not self.supported or self._handle is None:
            return False
        h_proc = _k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
                                  False, int(pid))
        if not h_proc:
            logger.warning("pyexec: OpenProcess(%d) 失败，本次进程不在 job 内", pid)
            return False
        try:
            ok = bool(_k32.AssignProcessToJobObject(self._handle, h_proc))
            if not ok:
                logger.warning("pyexec: AssignProcessToJobObject 失败（err %d）",
                               ctypes.get_last_error())
            return ok
        finally:
            _k32.CloseHandle(h_proc)

    def kill(self, pid: int | None = None) -> None:
        """杀掉整棵树。

        两条一起用，因为它们各自有盲区：``TerminateJobObject`` 覆盖 job 里的
        一切（包括我们不知道 pid 的孙进程），``psutil`` 兜住那个 assign 竞态窗口
        里逃出去的。
        """
        if pid is not None:
            self._psutil_kill_tree(pid)
        if self.supported and self._handle is not None:
            try:
                _k32.TerminateJobObject(self._handle, 1)
            except Exception:  # noqa: BLE001
                logger.debug("TerminateJobObject failed", exc_info=True)

    @staticmethod
    def _psutil_kill_tree(pid: int) -> None:
        try:
            import psutil
        except Exception:  # noqa: BLE001
            return
        try:
            proc = psutil.Process(pid)
        except Exception:  # noqa: BLE001 — 已经没了
            return
        victims = []
        try:
            victims = proc.children(recursive=True)
        except Exception:  # noqa: BLE001
            pass
        for p in [*victims, proc]:      # 先子后父：别让父进程死后子进程改挂 init
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        """关掉 job 句柄。``KILL_ON_JOB_CLOSE`` ⇒ 还活着的成员随之被 OS 收走。"""
        if self._closed:
            return
        self._closed = True
        if self.supported and self._handle is not None:
            try:
                _k32.CloseHandle(self._handle)
            except Exception:  # noqa: BLE001
                logger.debug("CloseHandle(job) failed", exc_info=True)
            self._handle = None

    def __enter__(self) -> "JobBox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["DEFAULT_ACTIVE_PROCESS_LIMIT", "DEFAULT_MEMORY_FRACTION",
           "IS_WINDOWS", "JobBox"]
