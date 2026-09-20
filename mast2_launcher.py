"""PyInstaller entry point for the MAST desktop bundle.

The Tk desktop launcher: a thin Python control panel that boots/stops the
**FastAPI service** (which serves the typed API + the bundled TypeScript React
SPA) and opens the browser to it. The launcher itself stays Python/Tk — it is
the PyInstaller entry that bootstraps the whole Python core (FastAPI + torch +
LangGraph + nanonis_spm); the TS rewrite converted the in-browser UI, not this
window. (Gradio was fully removed in the TS rewrite.)

Runs in two modes:

  Default (no flags) — **Launcher mode**: opens a Tk control panel with
    - 启动 / 重启 / 关闭 服务 按钮
    - GUI 网址 (一键浏览器打开)
    - API key 配置入口 (Kimi / DeepSeek / 阿里云 DashScope / Anthropic / …)
    - 状态指示灯 (key 是否就绪 + Nanonis 是否连接 + 服务是否在跑)
    - 子进程日志滚动框

  ``--service-mode``: skip the launcher window and boot the FastAPI service
    directly. The launcher Popen()'s the same exe with this flag to run the
    actual MAST service. In dev mode you can also call::

        .venv-v2-py313/Scripts/python.exe mast2_launcher.py --service-mode

Boot order in service mode (do not reorder):
    1. Resolve and chdir to the user-data root.
    2. Apply ``mast.core.platform_patch`` BEFORE heavy imports (Py3.14 +
       ``platform._wmi_query()`` hang).
    3. Make sure ``experiments/``, ``working-sessions/``, ``api key/``,
       ``artifacts/`` exist next to the exe so first-run is friction-free.
    4. Build the live FastAPI app (``mast.api.app.create_app`` +
       ``mast.api.bootstrap.build_live_context``) and serve it with uvicorn
       (HTTP/WS/SSE + the bundled ``frontend/dist`` SPA at ``/``).

This module is imported as the PyInstaller entry. It also runs fine in dev
(``.venv-v2-py313/Scripts/python.exe mast2_launcher.py``) which keeps the boot
path the same in both environments.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable

# In dev mode (not frozen), prepend MASTv2/ so `import mast` finds the v2
# package instead of the v1 one at the repo root. PyInstaller's spec
# handles this for the frozen build.
if not getattr(sys, "frozen", False):
    _repo = Path(__file__).resolve().parent
    _v2_root = _repo / "MASTv2"
    if _v2_root.exists():
        sys.path.insert(0, str(_v2_root))

# ── Per-provider key files (matches mast.config) ──────────────────────
PROVIDER_KEY_FILES: dict[str, str] = {
    "moonshot":  "kimi.env",       # 月之暗面 Kimi
    "deepseek":  "deepseek.env",
    "dashscope": "dashscope.env",  # 阿里云百炼（语音 / 嵌入 / 备用 LLM）
    "anthropic": "api key.env",    # Claude — legacy filename
    "minimax":   "minimax.env",    # MiniMax（Anthropic 兼容端点）
    "zhipu":     "glm.env",        # 智谱 GLM
}

PROVIDER_LABELS: dict[str, str] = {
    "moonshot":  "Kimi (Moonshot)",
    "deepseek":  "DeepSeek",
    "dashscope": "阿里云百炼 (语音/嵌入)",
    "anthropic": "Anthropic Claude",
    "minimax":   "MiniMax",
    "zhipu":     "智谱 GLM",
}

PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "moonshot":  ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "deepseek":  ("DEEPSEEK_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY", "ALIYUN_BAILIAN_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "minimax":   ("MINIMAX_API_KEY",),
    "zhipu":     ("ZHIPU_API_KEY", "GLM_API_KEY"),
}

# Nanonis main port from mast.config.NanonisConfig.port_main
NANONIS_HOST = "127.0.0.1"
NANONIS_PORT = 6501

# Gradio server — MAST2 uses 7862 / 7863 / 8766 to coexist with v1 on the
# same machine (v1 uses 7860/7861/8765). MASTConfig.server.server_port for v2
# remains 7860 by default (single-app dev workflow), but the launcher in
# bundled mode forces MAST2_LAUNCHER_PORT into the subprocess env so the
# service binds 7862.
GUI_HOST = "127.0.0.1"
GUI_PORT = 7862

# Push update server (intranet)
PUSH_SERVER_PORT = 8766

# Process-name signatures that suggest the Nanonis Mimea simulator (rather
# than a real V5e controller).  Compared case-insensitively against the
# names returned by `tasklist`.
_NANONIS_SIMULATOR_PROCESS_HINTS: tuple[str, ...] = (
    "MimeaV5e.exe",
    "Mimea_V5e.exe",
    "NanonisMimea.exe",
    "MimeaSimulator.exe",
    "Mimea-V5e.exe",
)

# Logo (used both for Tk window icon and PyInstaller exe icon). PNG path is
# what Tk's iconphoto() needs; ICO is what mast.spec / Inno Setup consume.
LOGO_PNG_RELPATH = Path("logo") / "MAST2_logo.png"
LOGO_ICO_RELPATH = Path("logo") / "MAST2_logo.ico"

# LAN access uses an auth file under `api key/` so it's preserved across
# upgrades (same as LLM provider keys). First non-comment line is the
# username, second non-comment line is the password.
LAN_AUTH_FILENAME = "lan_auth.env"

# Env vars passed from the launcher to its service / admin subprocesses to
# enable LAN binding + HTTP basic auth at the Gradio layer.
#
# SECURITY (v2.1.14): the LAN password must NOT travel through the child's
# environment block. On Windows a process's environment is readable by any
# other process running as the same user (Process Explorer, WMI
# Win32_Process, etc.) and lingers for the full lifetime of the long-running
# Gradio service. Instead we hand the child a *path* to a short-lived,
# owner-only-readable file (LAN_ENV_AUTH_FILE) holding `user\npassword`. The
# child reads it once at boot and immediately deletes it, so the exposure
# window is milliseconds and never touches the persistent env block.
#
# LAN_ENV_USER (the username — not a secret) and the legacy LAN_ENV_PASS are
# retained only so a *new* launcher can still drive an *old* child build; new
# launchers no longer populate LAN_ENV_PASS.
LAN_ENV_ENABLE    = "MAST2_LAUNCHER_LAN"
LAN_ENV_USER      = "MAST2_LAUNCHER_AUTH_USER"
LAN_ENV_PASS      = "MAST2_LAUNCHER_AUTH_PASS"        # legacy / back-compat only
LAN_ENV_AUTH_FILE = "MAST2_LAUNCHER_AUTH_FILE"        # path to ephemeral creds file

# Subdir (under user_root) holding ephemeral LAN-auth handoff files. Created
# owner-only where the OS supports it; swept clean on launcher start/exit.
LAN_AUTH_HANDOFF_DIR = ".mast2_lan_auth"

# Single-instance lock: <user_root>/.mast2_launcher.pid stores the active
# launcher PID. On startup, second instance finds existing PID, brings the
# original window forward via SendMessage WM_USER, then exits.
LAUNCHER_PID_FILE = ".mast2_launcher.pid"
LAUNCHER_PORT_FILE = ".mast2_launcher.port"     # win32 listener port for IPC
LAUNCHER_BRING_FOREGROUND_PORT_RANGE = (47860, 47880)

# Startup-with-Windows registry key (HKCU\Software\Microsoft\Windows\CurrentVersion\Run\MAST2)
STARTUP_REG_NAME = "MAST"


# ── Network helpers ─────────────────────────────────────────────────

def _lan_ip() -> str:
    """Return the host's primary outbound IPv4 address (or '' if offline).

    Uses a UDP socket trick so we don't require any actual network round-trip.
    Filters out 127.* loopback addresses.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        try:
            # Doesn't matter that 8.8.8.8 isn't reachable — we just want the
            # interface the OS would route through to it.
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    # Fallback: gethostbyname_ex
    try:
        host = socket.gethostname()
        for ip in socket.gethostbyname_ex(host)[2]:
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    return ""


# ── Single-instance lock (PID file + local IPC port) ──────────────────

def _pid_alive(pid: int) -> bool:
    """Check if a process with this PID is still alive (Windows-correct).

    NOTE: `os.kill(pid, 0)` does NOT work as a probe on Windows — it raises
    WinError 87 even for live processes that we lack TERMINATE rights on.
    The correct method is OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) +
    GetExitCodeProcess; STILL_ACTIVE (259) ⇒ alive.
    """
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_uint()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _read_launcher_lock(user_root: Path) -> tuple[int, int] | None:
    """Return (pid, ipc_port) from <user_root>/.mast2_launcher.pid or None."""
    f = user_root / LAUNCHER_PID_FILE
    if not f.exists():
        return None
    try:
        lines = f.read_text(encoding="utf-8").splitlines()
        pid = int(lines[0].strip())
        port = int(lines[1].strip()) if len(lines) > 1 else 0
    except (OSError, ValueError, IndexError):
        return None
    if not _pid_alive(pid):
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return (pid, port)


def _write_launcher_lock(user_root: Path, ipc_port: int) -> None:
    f = user_root / LAUNCHER_PID_FILE
    try:
        f.write_text(f"{os.getpid()}\n{ipc_port}\n", encoding="utf-8")
    except OSError as exc:
        print(f"[launcher] could not write lock {f}: {exc}")


def _clear_launcher_lock(user_root: Path) -> None:
    f = user_root / LAUNCHER_PID_FILE
    try:
        f.unlink(missing_ok=True)
    except OSError:
        pass


def _bind_ipc_port() -> tuple[socket.socket, int] | None:
    """Bind a TCP socket on a free port in BRING_FOREGROUND_PORT_RANGE.

    Returns (socket, port) or None if no port is free. The socket listens on
    127.0.0.1; second instance sends "BRING_FOREGROUND\n" to ask the first
    to surface its window.
    """
    lo, hi = LAUNCHER_BRING_FOREGROUND_PORT_RANGE
    for port in range(lo, hi):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            s.listen(4)
            s.setblocking(False)
            return (s, port)
        except OSError:
            continue
    return None


def _send_bring_foreground(port: int) -> bool:
    if port <= 0:
        return False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as c:
            c.sendall(b"BRING_FOREGROUND\n")
        return True
    except OSError:
        return False


# ── Startup-with-Windows toggle (HKCU registry) ─────────────────────────

def _startup_enabled() -> bool:
    """Check if MAST2 launcher is registered for HKCU Run."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_READ,
        )
        try:
            value, _ = winreg.QueryValueEx(key, STARTUP_REG_NAME)
            return bool(value)
        finally:
            winreg.CloseKey(key)
    except (FileNotFoundError, OSError):
        return False


def _startup_set(enabled: bool) -> bool:
    """Toggle startup with Windows. Returns True on success."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE,
        )
        try:
            if enabled:
                exe = sys.executable if _frozen() else f'"{sys.executable}" "{Path(__file__).resolve()}"'
                # Quote the path to handle spaces
                if _frozen():
                    exe = f'"{exe}"'
                # Boot-time launch starts MINIMIZED to the tray so it doesn't pop
                # a window on every login (resolves the auto-start ↔ ×-to-tray
                # interaction). The flag is consumed in main() / LauncherApp.
                winreg.SetValueEx(key, STARTUP_REG_NAME, 0, winreg.REG_SZ,
                                  exe + " --minimized")
            else:
                try:
                    winreg.DeleteValue(key, STARTUP_REG_NAME)
                except FileNotFoundError:
                    pass
            return True
        finally:
            winreg.CloseKey(key)
    except OSError as exc:
        print(f"[launcher] startup toggle failed: {exc}")
        return False


def _detect_nanonis_simulator() -> bool:
    """Return True if a Mimea-simulator process appears to be running locally.

    Uses Windows `tasklist /FO CSV /NH` so we don't pull in psutil. Returns
    False on any failure (caller treats unknown == real / external).
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return False
    if out.returncode != 0:
        return False
    text = out.stdout.lower()
    for hint in _NANONIS_SIMULATOR_PROCESS_HINTS:
        if hint.lower() in text:
            return True
    return False


# ── GPU detection (for the 视觉模型 / Vision toggle) ────────────────────
# These helpers shell out (nvidia-smi / PowerShell WMI) so they must NEVER be
# called on the Tk mainloop — the caller runs them on a background thread. They
# never raise: any failure returns the "unknown" shape so the dialog can still
# render an honest "未检测到显卡" message.

def _detect_gpu() -> dict:
    """Best-effort GPU probe WITHOUT importing torch.

    Returns a dict ``{"vendor","name","vram_gb","driver","source"}``.
    Tries ``nvidia-smi`` first (authoritative for NVIDIA: exact VRAM + driver),
    then falls back to Windows WMI (Win32_VideoController) which works for any
    adapter but caps AdapterRAM at ~4 GiB (uint32). On total failure returns the
    "unknown" shape. Swallows ALL exceptions — must never raise.
    """
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # 1. nvidia-smi — exact for NVIDIA cards.
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6,
            creationflags=no_window,
        )
        if out.returncode == 0 and out.stdout.strip():
            first = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 3:
                name = parts[0] or "NVIDIA GPU"
                try:
                    mib = float(parts[1])
                    vram_gb = round(mib / 1024.0, 1)
                except (ValueError, TypeError):
                    vram_gb = 0.0
                driver = parts[2]
                return {
                    "vendor": "NVIDIA", "name": name, "vram_gb": vram_gb,
                    "driver": driver, "source": "nvidia-smi",
                }
    except Exception:
        pass

    # 2. WMI fallback (any adapter, but VRAM is uint32-capped at ~4 GiB).
    try:
        ps = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -First 1 Name,AdapterRAM,DriverVersion | "
            "ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)|$($_.DriverVersion)\" }"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=8,
            creationflags=no_window,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            bits = line.split("|")
            name = (bits[0].strip() if len(bits) > 0 else "") or "未知"
            ram_raw = bits[1].strip() if len(bits) > 1 else ""
            driver = bits[2].strip() if len(bits) > 2 else ""
            lname = name.lower()
            if "nvidia" in lname:
                vendor = "NVIDIA"
            elif "amd" in lname or "radeon" in lname:
                vendor = "AMD"
            elif "intel" in lname:
                vendor = "Intel"
            else:
                vendor = "unknown"
            try:
                ram_bytes = float(ram_raw) if ram_raw else 0.0
                gib = round(ram_bytes / (1024.0 ** 3), 1)
            except (ValueError, TypeError):
                gib = 0.0
            # AdapterRAM is a uint32 → anything ≥ ~4 GiB reports as ~4.0 and is
            # not trustworthy; signal "unknown" with the >=4 sentinel.
            vram_gb = -1.0 if gib >= 4.0 else gib
            return {
                "vendor": vendor, "name": name, "vram_gb": vram_gb,
                "driver": driver, "source": "wmi",
            }
    except Exception:
        pass

    # 3. Nothing worked.
    return {"vendor": "unknown", "name": "未知", "vram_gb": 0.0,
            "driver": "", "source": "none"}


def _vision_support(gpu: dict) -> tuple[str, str]:
    """Map a ``_detect_gpu()`` result to (level, message).

    level ∈ {"ok","marginal","need_driver","unsupported"}. Message is a ready-to
    -display Chinese sentence. Never raises.
    """
    try:
        vendor = str(gpu.get("vendor", "unknown"))
        name = str(gpu.get("name", "未知")) or "未知"
        driver = str(gpu.get("driver", "") or "")
        source = str(gpu.get("source", "none"))
        try:
            vram_gb = float(gpu.get("vram_gb", 0.0))
        except (ValueError, TypeError):
            vram_gb = 0.0

        # Non-NVIDIA (or nothing detected) → CPU/Mock fallback only.
        if vendor != "NVIDIA" or source == "none":
            return ("unsupported",
                    f"未检测到 NVIDIA CUDA 显卡（{name}）。视觉模型将以 CPU/Mock "
                    "回退，速度很慢、能力受限。建议保持关闭。")

        # VRAM unknown (WMI uint32 cap) — can't size it precisely.
        if vram_gb < 0:
            return ("marginal",
                    f"检测到 {name}，但无法精确读取显存/驱动（非 NVIDIA 工具）。"
                    "若为较新的 NVIDIA 显卡通常可用；如加载失败会自动回退 Mock。")

        # Driver too old for CUDA 12.x. Parse leading int of e.g. "527.41".
        major = None
        try:
            m = "".join(ch for ch in driver.split(".")[0] if ch.isdigit())
            if m:
                major = int(m)
        except (ValueError, TypeError, IndexError):
            major = None
        if major is not None and major < 528:
            return ("need_driver",
                    f"驱动过旧（当前 {driver}）。本项目 CUDA 12.x 需要 NVIDIA "
                    "驱动 ≥ 528（建议 ≥ 555）。请到 NVIDIA 官网下载并自行更新驱动"
                    "后再开启视觉模型。")

        # Driver OK → size on VRAM.
        if vram_gb >= 6:
            return ("ok",
                    f"支持 ✓  {name} · 显存 {vram_gb} GB · 驱动 {driver}。"
                    "可加载完整 DINOv3(M12) 视觉模型。")
        if vram_gb >= 4:
            return ("marginal",
                    f"可用但显存偏紧（{vram_gb} GB）。建议使用 legacy 后端或小批量；"
                    "DINOv3 主干可能接近显存上限。")
        if vram_gb > 0:
            return ("unsupported",
                    f"显存不足（{vram_gb} GB < 4 GB），无法稳定加载视觉模型，"
                    "将回退 CPU/Mock。")
        # vram == 0 but NVIDIA detected (rare) — treat as unknown/marginal.
        return ("marginal",
                f"检测到 {name}，但无法读取显存大小。若为较新的 NVIDIA 显卡通常"
                "可用；如加载失败会自动回退 Mock。")
    except Exception:
        return ("unsupported",
                "显卡检测异常，视觉模型可能回退 CPU/Mock。建议保持关闭。")


# ── Vision on/off persistence (config/vision.json) ────────────────────
# The launcher remembers the user's choice across runs. DEFAULT True preserves
# the historical behaviour (the service always pre-warmed the vision model).

def _vision_config_path(root: Path) -> Path:
    return root / "config" / "vision.json"


def _load_vision_warn_suppressed(root: Path) -> bool:
    """Read config/vision.json → has the user ticked «下次不再提示» for the vision
    load warning? Best-effort; DEFAULT False (show the warning) on any error.

    Vision is ALWAYS on (it loads with the main service — there is no user-facing
    off, only an automatic Mock fallback inside VisionModule when a GPU can't load
    it). This flag only governs whether the GPU/load WARNING dialog pops on start."""
    try:
        import json as _json
        p = _vision_config_path(root)
        if not p.exists():
            return False
        data = _json.loads(p.read_text(encoding="utf-8"))
        return bool(data.get("warn_suppressed", False))
    except Exception:
        return False


def _save_vision_warn_suppressed(root: Path, suppressed: bool) -> None:
    """Persist the «下次不再提示» choice to config/vision.json. Swallows errors."""
    try:
        import json as _json
        p = _vision_config_path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps({"warn_suppressed": bool(suppressed)}),
                     encoding="utf-8")
    except Exception as exc:
        print(f"[launcher] could not save vision.json: {exc}")


# ── Persistent file logging ──────────────────────────────────────────
# Critical for diagnosing bundle startup failures. PyInstaller --noconsole
# (console=False in mast.spec) means sys.stdout/stderr go to NUL when the
# user double-clicks MAST.exe, so any uncaught exception during launcher
# init is silently swallowed. We tee everything to experiments/logs/ so
# the next launch can show the user "what went wrong last time".

class _Tee:
    """Forward writes to multiple file-like objects.

    Used to mirror stdout/stderr to a real file even when sys.__stdout__
    is None (PyInstaller --noconsole). Failures on any single stream are
    swallowed so logging never breaks the main code path.
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        n = 0
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
                n = len(data)
            except Exception:
                pass
        return n

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False


_LAUNCHER_LOG_FH = None  # kept open for the lifetime of the process


def _log_dir_for(root: Path) -> Path:
    """Return (and create) the per-install log directory."""
    p = root / "experiments" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _today_stamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d")


def _fmt_release_date(iso: str) -> str:
    """Trim an ISO-8601 stamp (e.g. RELEASED_AT / manifest.published_at) to a
    human YYYY-MM-DD. Returns '未知' for an empty/garbled value so the version
    dialog always renders an honest cell."""
    s = (iso or "").strip()
    if not s:
        return "未知"
    # Both RELEASED_AT and published_at are ISO-8601 (date part = first 10 chars).
    return s[:10] if len(s) >= 10 and s[4] == "-" and s[7] == "-" else s


def _setup_persistent_logging(user_root: Path) -> Path:
    """Tee sys.stdout/stderr → experiments/logs/launcher-YYYYMMDD.log.

    Idempotent — calling twice is a no-op (we keep one open file handle).
    Returns the log file path so the launcher can show it to the user.
    """
    global _LAUNCHER_LOG_FH
    if _LAUNCHER_LOG_FH is not None:
        return Path(_LAUNCHER_LOG_FH.name)

    import datetime, faulthandler, platform
    log_dir = _log_dir_for(user_root)
    log_path = log_dir / f"launcher-{_today_stamp()}.log"

    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    fh.write(
        "\n"
        + "=" * 70
        + "\n"
        + f"=== Launcher session start: {datetime.datetime.now().isoformat()}\n"
        + f"=== sys.executable : {sys.executable}\n"
        + f"=== sys.frozen     : {getattr(sys, 'frozen', False)}\n"
        + f"=== _MEIPASS       : {getattr(sys, '_MEIPASS', '(none)')}\n"
        + f"=== platform       : {platform.platform()}\n"
        + f"=== python         : {sys.version.split()[0]}\n"
        + f"=== user_root      : {user_root}\n"
        + f"=== argv           : {sys.argv}\n"
        + "=" * 70
        + "\n"
    )
    fh.flush()
    _LAUNCHER_LOG_FH = fh

    # Tee the original sys.stdout/stderr (which may be None in --noconsole
    # bundle) into our file. After this point, print() and uncaught
    # exceptions reach the log file.
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)

    # Hard-crash hook: catches segfaults and similar that don't go through
    # Python's exception machinery. faulthandler writes the C-level traceback
    # straight to fh.
    try:
        faulthandler.enable(file=fh, all_threads=True)
    except Exception:
        pass

    return log_path


# ── Path helpers ─────────────────────────────────────────────────────

def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _resource_path(relpath: Path | str) -> Path:
    """Resolve a path relative to the app, working in dev AND in PyInstaller.

    PyInstaller --onedir lays the source tree under sys._MEIPASS at runtime;
    in dev we just resolve relative to this file's parent. Caller passes a
    repo-relative path like ``Path("logo") / "MAST_logo.png"``.
    """
    base = Path(getattr(sys, "_MEIPASS", "")) if _frozen() else Path(__file__).resolve().parent
    p = base / Path(relpath)
    if p.exists():
        return p
    # Fallback: maybe the user moved the asset next to the .exe (e.g. zip
    # extract). Look there too so the launcher doesn't crash on a missing
    # resource — the logo is purely cosmetic.
    if _frozen():
        alt = Path(sys.executable).resolve().parent / Path(relpath)
        if alt.exists():
            return alt
    return p  # may not exist; caller checks


# When the Inno Setup installer runs it asks the user to pick a data
# directory (separate from the binary install path). The chosen path is
# written to <install>/data_dir.txt — one absolute path per file. Launcher
# reads this and uses it as the user_root so user data lives independently
# of the binary install (binary in C:\MAST\, data wherever the user picked).
DATA_DIR_MARKER = "data_dir.txt"


def _read_data_dir_marker(install_dir: Path) -> Path | None:
    """Read <install>/data_dir.txt, return resolved Path or None."""
    marker = install_dir / DATA_DIR_MARKER
    if not marker.exists():
        return None
    try:
        line = marker.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    if not line:
        return None
    p = Path(line).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return p.resolve()


def _user_root() -> Path:
    """Directory holding experiments/, api key/, models/, config/.

    Resolution order:
      1. ``$MAST2_USER_ROOT`` env var (escape hatch for testing / portable mode)
      2. ``<exe-parent>/data_dir.txt`` (written by the Inno Setup installer)
      3. ``<exe-parent>`` (legacy layout — v0.2.0..0.2.4 default; also zip/dev)
    """
    env = os.environ.get("MAST2_USER_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p.resolve()
        except OSError:
            pass

    install_dir = (
        Path(sys.executable).resolve().parent if _frozen()
        else Path(__file__).resolve().parent
    )
    marker_target = _read_data_dir_marker(install_dir)
    if marker_target is not None:
        return marker_target
    return install_dir


def _ensure_user_dirs(root: Path) -> None:
    for sub in ("experiments", "working-sessions", "api key", "models"):
        (root / sub).mkdir(parents=True, exist_ok=True)


# ── Key loading / saving ─────────────────────────────────────────────

def _read_key_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        pass
    return ""


def get_key(provider: str, root: Path) -> str:
    """env var first → file fallback. Empty string when missing."""
    for var in PROVIDER_ENV_VARS.get(provider, ()):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    fname = PROVIDER_KEY_FILES.get(provider)
    if not fname:
        return ""
    return _read_key_file(root / "api key" / fname)


def save_key(provider: str, root: Path, key: str) -> Path:
    fname = PROVIDER_KEY_FILES[provider]
    path = root / "api key" / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {PROVIDER_LABELS[provider]} API key. First non-comment line is used.\n"
        f"{key.strip()}\n",
        encoding="utf-8",
    )
    return path


# ── LAN auth (username + password) ───────────────────────────────────

def get_lan_auth(root: Path) -> tuple[str, str]:
    """Return (username, password) from `api key/lan_auth.env`, or ('','')."""
    path = root / "api key" / LAN_AUTH_FILENAME
    if not path.exists():
        return ("", "")
    user = ""
    pwd = ""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not user:
                user = line
            elif not pwd:
                pwd = line
                break
    except OSError:
        return ("", "")
    return (user, pwd)


def save_lan_auth(root: Path, username: str, password: str) -> Path:
    """Persist LAN auth to `api key/lan_auth.env` (preserved across upgrade)."""
    path = root / "api key" / LAN_AUTH_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# MAST 局域网访问凭据。第一行非注释 = 用户名，第二行 = 密码。\n"
        "# 仅用于启用 LAN 时的 Gradio HTTP basic auth；不影响其它 API key。\n"
        f"{username.strip()}\n"
        f"{password.strip()}\n",
        encoding="utf-8",
    )
    return path


def clear_lan_auth(root: Path) -> None:
    """Remove the LAN auth file (used when user disables LAN access)."""
    path = root / "api key" / LAN_AUTH_FILENAME
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# ── LAN auth handoff (ephemeral file → child, instead of env var) ─────
#
# These functions implement the secure launcher→child credential handoff:
# the password never enters the child's environment block. See the
# LAN_ENV_AUTH_FILE comment near the top of this module for the rationale.

def _restrict_file_to_owner(path: Path) -> None:
    """Best-effort: make ``path`` readable only by the current user.

    Windows: ``icacls`` removes inheritance and grants only the current
    user. POSIX: ``chmod 0600``. Any failure is swallowed — the file is
    short-lived and lives under the user's own data dir, so this is
    defence-in-depth, not the sole protection.
    """
    try:
        if sys.platform == "win32":
            user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if not user:
                return
            subprocess.run(
                ["icacls", str(path), "/inheritance:r",
                 "/grant:r", f"{user}:F"],
                capture_output=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.chmod(path, 0o600)
    except Exception:
        pass


def _lan_auth_handoff_dir(user_root: Path) -> Path:
    """Return (creating if needed) the ephemeral LAN-auth handoff dir."""
    d = user_root / LAN_AUTH_HANDOFF_DIR
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _sweep_lan_auth_handoff(user_root: Path) -> None:
    """Delete any leftover handoff files (e.g. a child that died pre-read).

    Called on launcher start and exit so stale credential files never
    accumulate on disk.
    """
    d = user_root / LAN_AUTH_HANDOFF_DIR
    if not d.exists():
        return
    try:
        for f in d.iterdir():
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass


def write_lan_auth_handoff(user_root: Path, username: str, password: str) -> Path:
    """Write `username\\npassword` to a unique owner-only temp file.

    Returns the file path; the consuming child is responsible for reading
    then deleting it (``consume_lan_auth_handoff``). The launcher also sweeps
    the directory on exit as a backstop against leaks.
    """
    import secrets
    d = _lan_auth_handoff_dir(user_root)
    path = d / f"auth-{os.getpid()}-{secrets.token_hex(8)}.tmp"
    # Create with restrictive mode up-front on POSIX (mkstemp-style); on
    # Windows we tighten the ACL right after writing.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, f"{username}\n{password}\n".encode("utf-8"))
    finally:
        os.close(fd)
    _restrict_file_to_owner(path)
    return path


def consume_lan_auth_handoff(path_str: str) -> tuple[str, str]:
    """Read (username, password) from a handoff file, then delete it.

    Returns ('', '') if the path is empty/unreadable. The file is unlinked
    in all cases (best-effort) so the secret does not outlive boot.
    """
    if not path_str:
        return ("", "")
    p = Path(path_str)
    user = ""
    pwd = ""
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        if lines:
            user = lines[0].strip()
        if len(lines) > 1:
            pwd = lines[1].strip()
    except OSError:
        pass
    finally:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    return (user, pwd)


def _read_lan_auth_from_env() -> tuple[str, str]:
    """Resolve LAN (username, password) for a service/admin child process.

    Preference order:
      1. ``LAN_ENV_AUTH_FILE`` — the secure ephemeral file (v2.1.14+). Read
         once and deleted; the password never lived in the env block.
      2. ``LAN_ENV_PASS`` — legacy plaintext env var, kept only so a new
         launcher can drive an old child or vice-versa.

    Username comes from the file when present, else ``LAN_ENV_USER``.
    Returns ('', '') when nothing usable is configured.
    """
    auth_file = os.environ.get(LAN_ENV_AUTH_FILE, "").strip()
    if auth_file:
        user, pwd = consume_lan_auth_handoff(auth_file)
        if user and pwd:
            return (user, pwd)
        # File missing/empty (already consumed by a sibling, or race) — fall
        # through to the legacy env path if present.
    user = os.environ.get(LAN_ENV_USER, "").strip()
    pwd  = os.environ.get(LAN_ENV_PASS, "").strip()
    return (user, pwd)


# ── Status checks ────────────────────────────────────────────────────

def port_open(host: str, port: int, *, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def gui_running() -> bool:
    return port_open(GUI_HOST, GUI_PORT)


def _pid_on_port(port: int) -> int | None:
    """Return the PID LISTENING on a local TCP *port* (Windows), else None.

    Used by the launcher's «关闭主服务并退出启动器» to stop a service that was
    started OUTSIDE this launcher (no Popen handle) — e.g. a directly-launched
    C:\\MAST\\MAST.exe. Parses `netstat -ano`; best-effort, never raises.
    """
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper() == "TCP" \
                    and parts[-1].isdigit() and "LISTEN" in line.upper() \
                    and (parts[1].endswith(f":{port}")):
                return int(parts[-1])
    except Exception:
        pass
    return None


def _http_shutdown_request(port: int, *, log=None) -> bool:
    """Best-effort POST /api/admin/shutdown to the LOCAL service so it closes the
    Nanonis pool + stops daemons + asks uvicorn to exit gracefully. A windowless
    (console=False) frozen service has no top-level window, so ``taskkill``
    WITHOUT ``/F`` can't reach it and the launcher falls through to a hard
    TerminateProcess mid-TCP — this HTTP path is what actually lets it stop
    cleanly. Loopback-only endpoint; tries http then https."""
    import ssl as _ssl
    import urllib.request
    for scheme in ("http", "https"):
        url = f"{scheme}://127.0.0.1:{port}/api/admin/shutdown"
        try:
            req = urllib.request.Request(url, method="POST", data=b"")
            ctx = _ssl._create_unverified_context() if scheme == "https" else None
            urllib.request.urlopen(req, timeout=3, context=ctx)  # noqa: S310 (loopback)
            return True
        except Exception:
            continue
    return False


def _graceful_stop_proc(proc, *, graceful_timeout: float = 15.0,
                        force_timeout: float = 5.0, log=None) -> str:
    """Stop a launcher-OWNED ``Popen`` GRACEFULLY, hard-killing only as a last
    resort.

    Why this is not just ``proc.terminate()``: on Windows ``Popen.terminate()``
    is ``TerminateProcess`` — an immediate HARD kill that gives the main service
    NO chance to run its shutdown handlers. The service holds the Nanonis V5e TCP
    connection, and a force-kill mid-TCP can corrupt the Nanonis port until
    Nanonis itself restarts (see 项目规约 / _make_pid_stopper). So we mirror the
    safe external-service path: first ask the process to stop *gracefully* via
    ``taskkill /PID`` WITHOUT ``/F`` (which posts a normal termination request the
    target can intercept to graceful-shutdown the TCP link), wait up to
    *graceful_timeout* seconds, and only if it refuses do we fall back to a hard
    kill — and even then we log it so the (rare) port-corruption window is
    auditable. On POSIX, ``taskkill`` does not exist, so we send SIGTERM
    (proc.terminate) as the graceful step and SIGKILL (proc.kill) as the fallback.

    Returns one of ``"graceful"`` / ``"forced"`` / ``"already-dead"`` / ``"error"``.
    """
    def _say(m: str) -> None:
        try:
            (log or logger.info)(m)
        except Exception:
            pass

    if proc is None or proc.poll() is not None:
        return "already-dead"
    pid = getattr(proc, "pid", None)

    # ── Graceful step 0: HTTP self-shutdown (works for a windowless service,
    #    which taskkill /PID without /F cannot reach). ─────────────────
    try:
        if _http_shutdown_request(GUI_PORT, log=log):
            _say("已请求主服务优雅自关闭（/api/admin/shutdown）。")
            for _ in range(max(1, int(min(graceful_timeout, 8.0) / 0.2))):
                if proc.poll() is not None:
                    return "graceful"
                time.sleep(0.2)
    except Exception as exc:
        _say(f"HTTP 优雅关闭请求失败（改用 taskkill）: {exc}")

    # ── Graceful step ────────────────────────────────────────────────
    try:
        if sys.platform == "win32" and pid:
            # taskkill WITHOUT /F → cooperative stop; the service can close its
            # Nanonis TCP link cleanly before exiting (no TerminateProcess).
            subprocess.run(
                ["taskkill", "/PID", str(pid)],
                capture_output=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            proc.terminate()  # POSIX: SIGTERM (graceful) — or Win fallback if no pid
    except Exception as exc:
        _say(f"优雅停止请求发送失败（将等待并在必要时强制结束）: {exc}")

    # Give it a GENEROUS window to flush + close the Nanonis port.
    deadline = max(1, int(graceful_timeout / 0.2))
    for _ in range(deadline):
        if proc.poll() is not None:
            return "graceful"
        time.sleep(0.2)

    # ── Forced fallback — only after graceful failed. Narrow the window. ──
    _say(f"主服务在 {graceful_timeout:.0f}s 内未优雅退出 — 将强制结束"
         f"（PID {pid}；注意：此时若正与 Nanonis 通信，端口可能需 Nanonis 重启恢复）。")
    try:
        proc.kill()  # Windows: TerminateProcess; POSIX: SIGKILL
    except Exception as exc:
        _say(f"强制结束失败: {exc}")
        return "error"
    for _ in range(max(1, int(force_timeout / 0.2))):
        if proc.poll() is not None:
            return "forced"
        time.sleep(0.2)
    return "forced"


#: ``nanonis_connected()`` 的短缓存：(墙钟时刻, 结果)。状态刷新会在服务刚起来的
#: 两分钟内被密集调用，没有它就是一串 netstat 子进程。
_NANONIS_PROBE: list = [0.0, False]
_NANONIS_PROBE_TTL_S = 1.5


def nanonis_connected() -> bool:
    """Nanonis 的 TCP 服务在不在（状态灯用）。

    ⚠️ **刻意不用 port_open()。** 那个函数会真的建一条 TCP 到 6501 再关掉，而这个
    函数在「运行主服务」之后的两分钟里会被轮询几十次（前 30 秒每 2 秒一次），等于
    对 Nanonis 主口反复建链断链 —— 那是本仓库反复吃过亏的一个端口（见
    既有教训：mid-TCP 被打断会永久损坏它，直到 Nanonis
    软件重启）。为了点亮一个状态灯而反复戳它，风险与收益完全不成比例。

    改成查本机有没有进程在 6501 上 LISTENING —— 回答的是同一个问题（Nanonis 起没
    起），但**一个字节都不发给它**。``NANONIS_HOST`` 写死是 127.0.0.1，所以本机
    netstat 一定看得到。

    带 1.5 秒缓存：``_pid_on_port`` 会起一个 netstat 子进程，轮询期照原样调用就是
    每 2 秒一个进程。
    """
    now = time.time()
    if (now - _NANONIS_PROBE[0]) < _NANONIS_PROBE_TTL_S:
        return bool(_NANONIS_PROBE[1])
    ok = _pid_on_port(NANONIS_PORT) is not None
    _NANONIS_PROBE[0], _NANONIS_PROBE[1] = now, ok
    return ok


# ── Service-mode entry (existing GUI boot path) ──────────────────────

def _lan_cert_hosts(extra_hosts=None) -> tuple[set, set]:
    """(dns_names, ip_addrs) the LAN cert must cover: loopback + this machine's
    hostname + all its resolved IPs + any explicit *extra_hosts* (the configured
    server address a client dials). Detecting the real LAN IP is the fix for the
    old localhost-only SAN — a client hitting https://<lan-ip> failed hostname
    verification because the cert didn't list that IP."""
    import socket as _sock

    names = {"localhost"}
    ips = {"127.0.0.1", "::1"}
    try:
        hn = _sock.gethostname()
        if hn:
            names.add(hn)
            try:
                for info in _sock.getaddrinfo(hn, None):
                    addr = info[4][0]
                    if addr:
                        ips.add(addr.split("%", 1)[0])  # strip IPv6 zone id
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    import ipaddress as _ip
    for h in (extra_hosts or []):
        h = str(h or "").strip().strip("[]")
        if not h or h in ("0.0.0.0", "::"):  # wildcard bind → not a dialable host
            continue
        try:
            _ip.ip_address(h)
            ips.add(h)
        except ValueError:
            names.add(h)
    return names, ips


def _cert_covers(cert: Path, names: set, ips: set) -> bool:
    """True iff *cert*'s SAN already lists every required name + ip."""
    try:
        from cryptography import x509
        c = x509.load_pem_x509_certificate(cert.read_bytes())
        san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        have_dns = {str(n).lower() for n in san.get_values_for_type(x509.DNSName)}
        have_ip = {str(i) for i in san.get_values_for_type(x509.IPAddress)}
        return {n.lower() for n in names} <= have_dns and set(ips) <= have_ip
    except Exception:  # noqa: BLE001 — unreadable/old cert → regenerate to be safe
        return False


def _ensure_lan_cert(cert: Path, key: Path, log, extra_hosts=None) -> bool:
    """Generate a self-signed TLS cert+key for LAN HTTPS if missing OR if the
    existing cert does not cover the required hosts (loopback + this machine's
    hostname/IPs + *extra_hosts*, e.g. the push-server address clients dial).

    Returns True when a usable cert/key pair exists afterward. Best-effort:
    needs `cryptography` (pinned in requirements); if absent, returns False and
    the caller falls back to the cleartext-warning path. The cert is long-lived
    (10y) and self-signed — fine for an intranet instrument console (the point is
    to encrypt the token/creds + traffic, not public PKI trust; clients pin this
    cert as their CA — see update client verify)."""
    names, ips = _lan_cert_hosts(extra_hosts)
    if cert.exists() and key.exists() and _cert_covers(cert, names, ips):
        return True
    if cert.exists():
        log.info("LAN cert missing required host(s) %s / %s — regenerating.", names, ips)
    try:
        import datetime as _dt
        import ipaddress as _ip

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "MAST LAN")])
        san_entries: list = [x509.DNSName(n) for n in sorted(names)]
        for ipstr in sorted(ips):
            try:
                san_entries.append(x509.IPAddress(_ip.ip_address(ipstr)))
            except ValueError:
                continue
        san = x509.SubjectAlternativeName(san_entries)
        now = _dt.datetime.now(_dt.timezone.utc)
        cert_obj = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(days=1))
            .not_valid_after(now + _dt.timedelta(days=3650))
            .add_extension(san, critical=False)
            .sign(k, hashes.SHA256())
        )
        cert.parent.mkdir(parents=True, exist_ok=True)
        key.write_bytes(k.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
        cert.write_bytes(cert_obj.public_bytes(serialization.Encoding.PEM))
        log.info("Generated self-signed LAN TLS cert at %s", cert)
        return True
    except Exception as exc:
        log.warning("LAN TLS cert auto-gen failed (%s) — install `cryptography` "
                    "or drop lan_cert.pem/lan_key.pem manually.", exc)
        return False


def run_service() -> int:
    """Boot the MAST service (FastAPI + TS SPA). Called when ``--service-mode`` is set."""
    user_root = _user_root()
    os.chdir(user_root)
    _ensure_user_dirs(user_root)
    os.environ.setdefault("MAST2_PROJECT_ROOT", str(user_root))

    # Belt (frozen service): the DINOv3 vision backbone is bundled offline next to
    # the exe, so timm/HF must NEVER reach the network. A blocked huggingface.co
    # (restricted network) otherwise makes timm.create_model(pretrained=True)
    # retry-hang for minutes → the background vision warm never signals DONE/FAILED
    # → the launcher's 3-min vision watcher times out and the user sees
    # "启动不起来" even though the service is up.
    # configure_backbone_cache also forces this; set it process-wide here in case
    # any code path imports timm before that runs.
    if getattr(sys, "frozen", False):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # MUST precede heavy imports (Py3.14 platform._wmi_query() hang on import).
    import mast.core.platform_patch  # noqa: F401

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("mast.service")

    # Surface Anthropic key into env if file exists (legacy compat).
    if not os.environ.get("ANTHROPIC_API_KEY"):
        k = get_key("anthropic", user_root)
        if k:
            os.environ["ANTHROPIC_API_KEY"] = k

    from mast.config import MASTConfig

    config = MASTConfig()
    # MAST2 binds 7862 to avoid colliding with v1 at 7860
    config.server.server_port = GUI_PORT
    host = "127.0.0.1"
    auth = None
    if os.environ.get(LAN_ENV_ENABLE) == "1":
        user, pwd = _read_lan_auth_from_env()
        if user and pwd:
            host = "0.0.0.0"
            auth = (user, pwd)
            log.info("LAN access ON — binding 0.0.0.0 with HTTP basic auth (user=%s).", user)
        else:
            log.warning("LAN access requested but auth empty — staying on 127.0.0.1.")
    # Scheme logged accurately after the TLS decision below; bind host/port here.
    log.info("Starting MAST service (FastAPI + TS SPA) on %s:%s ...", host, GUI_PORT)
    # Background update-checker. Always starts; reads server URL + token
    # from <data>/api key/ each cycle, no-op when neither is configured.
    try:
        from mast import __version__ as _mast_version
        from mast.update.client import UpdateChecker
        checker = UpdateChecker(
            user_root, _mast_version,
            interval_min=config.update.check_interval_min,
        )
        checker.start()
        log.info("Update checker started (interval=%dmin)",
                 config.update.check_interval_min)
    except Exception as exc:
        log.warning("Could not start update checker: %s", exc)

    # Pre-warm the M12 vision model in the background so the scan-progress
    # vision monitor's 12.5%→100% milestones fire promptly on the first scan
    # (the ~30s backbone cold-load otherwise races a short acquisition).
    #
    # Vision loads WITH the service (always on). Warm it in a daemon thread that
    # emits MAST_VISION_LOAD_{START,DONE,FAILED} markers so the launcher's progress
    # bar can track the load. CRITICAL: use the LOGGING channel, NOT print(). In the
    # frozen --noconsole build, the launcher Popen's the service with stdout=PIPE;
    # a bare print() to sys.stdout there interleaves unreliably with the logging
    # StreamHandler's stderr writes and the DONE marker was being LOST in the pipe
    # (the service loaded vision fine but the launcher never saw DONE → 3-min
    # "视觉模型加载超时" on ANY GPU, incl. a fast RTX 4060). Logging goes through the
    # same handler as every other line that DOES reach the pipe; the launcher
    # matches the marker as a substring, so "[INFO] mast.service: MAST_VISION_LOAD_DONE"
    # still triggers _vision_progress_done.
    if os.environ.get("MAST_VISION_BACKEND", "").strip().lower() != "mock":
        def _vision_warm() -> None:
            try:
                log.info("MAST_VISION_LOAD_START")
                from mast.vision.module import VisionModule
                vm = VisionModule.get()
                be = getattr(vm, "_backend", None)
                if be is not None and hasattr(be, "preload"):
                    be.preload()
                log.info("MAST_VISION_LOAD_DONE")
            except Exception as exc:  # never crash the service
                log.error("MAST_VISION_LOAD_FAILED %s", exc)
        try:
            threading.Thread(target=_vision_warm, daemon=True).start()
            log.info("vision model pre-warm kicked off (background)")
        except Exception as exc:
            log.info("vision pre-warm not started: %s", exc)
    else:
        log.info("vision backend forced to mock via env — skipping pre-warm")

    # Build the live FastAPI app (gradio-free CoreRuntime core + bundled TS SPA)
    # and serve it. This REPLACES the deleted Gradio app — same URL, same port.
    import uvicorn

    from mast.api.app import create_app
    from mast.api.bootstrap import build_live_context

    ctx = build_live_context(config)
    app = create_app(context=ctx, dev_cors=False, auth=auth)

    ssl_kw: dict = {}
    if host != "127.0.0.1":
        cert = user_root / "api key" / "lan_cert.pem"
        key = user_root / "api key" / "lan_key.pem"
        # Fold this machine's Tailscale host(s) into the cert SAN so a peer that
        # dials the MagicDNS name / 100.x IP gets a name match, not a mismatch
        # warning. Best-effort: no Tailscale → empty → cert unchanged.
        ts_extra: list[str] = []
        try:
            from mast.net.tailscale import extra_cert_hosts, get_status
            ts_extra = extra_cert_hosts(get_status())
            if ts_extra:
                log.info("Tailscale 远程访问就绪 — 证书 SAN 追加 %s", ts_extra)
        except Exception as exc:  # never block boot on detection
            log.info("Tailscale host detection skipped: %s", exc)
        if _ensure_lan_cert(cert, key, log, extra_hosts=ts_extra):
            ssl_kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
            log.info("TLS enabled for LAN (https).")
        else:
            log.warning(
                "LAN binding WITHOUT TLS — basic-auth creds travel in cleartext. "
                "Install `cryptography` (auto-gen) or drop lan_cert.pem/lan_key.pem "
                "in <data>/api key/ to enable HTTPS."
            )

    # Use uvicorn.Server (not uvicorn.run) so the in-process /api/admin/shutdown
    # endpoint can set server.should_exit for a clean graceful stop — the
    # launcher POSTs it before any taskkill fallback.
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=GUI_PORT, **ssl_kw))
    try:
        app.state.uvicorn_server = server
    except Exception:
        pass
    server.run()
    return 0


def _run_publish(setup_exe: str, version: str | None) -> int:
    """Admin CLI: publish a new MAST version to the push server's data dir.

    Mirrors ``python -m mast.update publish`` so admins can use the bundle's
    own MAST.exe without needing a separate Python install.
    """
    if not version:
        print("ERROR: --publish requires --version X.Y.Z", file=sys.stderr)
        return 2
    user_root = _user_root()
    _ensure_user_dirs(user_root)
    os.environ.setdefault("MAST2_PROJECT_ROOT", str(user_root))

    from mast.update.manifest import make_manifest, write_manifest
    from mast.update.server import push_dir
    import shutil

    src = Path(setup_exe).resolve()
    if not src.exists():
        print(f"ERROR: not found: {src}", file=sys.stderr)
        return 2
    pdir = push_dir(user_root)
    target = pdir / src.name
    if src.resolve() != target.resolve():
        print(f"Copying {src.name} -> {target}")
        shutil.copy2(src, target)
    print(f"Computing SHA256 of {target.name} ...")
    m = make_manifest(target, version=version.strip())
    write_manifest(pdir / "manifest.json", m)
    print(f"\nPublished v{m.version}")
    print(f"  filename:    {m.filename}")
    print(f"  sha256:      {m.sha256}")
    print(f"  size_bytes:  {m.size_bytes}")
    print(f"  manifest:    {pdir / 'manifest.json'}")
    return 0


def run_push_server() -> int:
    """Boot the intranet push-update server. Called via ``--push-server-mode``."""
    user_root = _user_root()
    os.chdir(user_root)
    _ensure_user_dirs(user_root)
    os.environ.setdefault("MAST2_PROJECT_ROOT", str(user_root))

    import mast.core.platform_patch  # noqa: F401
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("mast.update_server")

    from mast.config import MASTConfig
    from mast.update.server import run_server

    cfg = MASTConfig().update
    host = cfg.server_host
    # LAN TLS: encrypt the Bearer token + manifest + installer/delta bytes in
    # flight. Reuse the self-signed LAN cert, but ensure its SAN covers the
    # push-server address clients dial (server_host) so they don't get a hostname
    # mismatch. A loopback-only bind stays plaintext http (local dev). Clients
    # pin this cert as their CA — the admin copies lan_cert.pem to each client's
    # <data>/api key/update_server_ca.pem (out-of-band; avoids TOFU).
    ssl_cert = ssl_key = None
    if host not in ("127.0.0.1", "localhost", "::1", ""):
        cert = user_root / "api key" / "lan_cert.pem"
        key = user_root / "api key" / "lan_key.pem"
        if _ensure_lan_cert(cert, key, log, extra_hosts=[host]):
            ssl_cert, ssl_key = str(cert), str(key)
            log.info("push server TLS enabled (https). Distribute %s to clients as "
                     "'update_server_ca.pem' so they trust it.", cert)
            # Auto-pin the serving cert as the LOCAL client's trusted CA. The
            # client trusts a build-bundled CA that does NOT match this machine's
            # runtime self-signed cert, so without a pin the same-machine wishlist
            # relay + OTA download fail TLS verification (CERTIFICATE_VERIFY_FAILED
            # — the silent 收信/更新 白屏, 审查). Remote clients still
            # get the cert out-of-band; this only self-heals the admin's own box.
            try:
                ca_pin = user_root / "api key" / "update_server_ca.pem"
                if (not ca_pin.exists()) or ca_pin.read_bytes() != cert.read_bytes():
                    ca_pin.write_bytes(cert.read_bytes())
                    log.info("auto-pinned serving cert → %s (local client now trusts this server)", ca_pin)
            except Exception as exc:  # best-effort; remote clients unaffected
                log.warning("could not auto-pin serving cert for the local client: %s", exc)
        else:
            log.warning("push server WITHOUT TLS — token + update bytes travel in "
                        "cleartext; install `cryptography` to auto-gen a LAN cert.")
    # MAST2 binds 8766 to avoid colliding with v1's 8765 on the same admin host
    log.info("MAST push server starting on %s://%s:%d  (data: %s)",
             "https" if ssl_cert else "http", host, PUSH_SERVER_PORT, user_root)
    run_server(user_root, host=host, port=PUSH_SERVER_PORT,
               ssl_certfile=ssl_cert, ssl_keyfile=ssl_key)
    return 0


def _clear_pending_delta(pdir: Path) -> None:
    """Discard a staged pending delta (marker + zip + offline stage) so a stale /
    bad orphan isn't retried."""
    import shutil as _sh
    try:
        from mast.update.client import DELTA_MARKER
        (pdir / DELTA_MARKER).unlink(missing_ok=True)
        for z in pdir.glob("delta_*.zip"):
            z.unlink(missing_ok=True)
        stage = pdir / "offline_stage"
        if stage.exists():
            _sh.rmtree(stage, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[launcher] could not clear stale delta: {exc}")


def _pending_delta_from_current(man: dict, pdir: Path) -> bool:
    """True iff the pending delta's ``from_version`` matches our CURRENT version,
    so it is the right delta to apply here. A stale orphan (from a prior version —
    e.g. left in pending after a full-installer upgrade) has a mismatched
    from_version; applying it would mis-patch / downgrade the new install, so we
    skip AND clear it. Also requires a plausible newer
    to_version."""
    try:
        from mast import __version__ as cur
    except Exception:  # pragma: no cover
        cur = ""
    frm = str(man.get("from_version", ""))
    to = str(man.get("to_version", ""))
    if frm and frm == str(cur) and to and to != str(cur):
        return True
    print(f"[launcher] pending delta {frm!r}→{to!r} does not match current {cur!r} "
          f"— clearing stale orphan")
    _clear_pending_delta(pdir)
    return False


def _apply_pending_delta_if_safe(user_root: Path) -> None:
    """Hot-apply a staged OTA delta IFF it only refreshes data assets (the
    literature index / vision checkpoint) — safe to overwrite while MAST.exe is
    running. A delta touching code/exe/dll is left for the full-installer path
    (Windows can't replace the running exe in place). Fail-safe: any error leaves
    the delta untouched so the full installer remains the fallback.

    Applied BEFORE the full-installer check; a data-only delta leaves no .exe in
    pending, so load_pending() returns None and the launcher continues normally
    into the freshly-updated app.
    """
    try:
        import json as _json
        from mast.update.client import (
            DELTA_MARKER, apply_pending_delta, delta_is_data_only, pending_dir,
        )
        from mast.update.delta import read_delta_manifest
    except Exception as exc:
        print(f"[launcher] delta module unavailable: {exc}")
        return
    try:
        pdir = pending_dir(user_root)
        marker = pdir / DELTA_MARKER
        if not marker.exists():
            return
        info = _json.loads(marker.read_text(encoding="utf-8"))
        dz = pdir / str(info.get("filename", ""))
        if not dz.exists():
            return
        man = read_delta_manifest(dz)
        if not _pending_delta_from_current(man, pdir):
            return  # stale orphan (from a prior version) — skipped + cleared
        paths = (list(man.get("added", [])) + list(man.get("changed", []))
                 + list(man.get("removed", [])))
        if not delta_is_data_only(paths):
            # code/exe/base_library delta → handled by the offline applier next
            return
        install_root = (Path(sys.executable).resolve().parent if _frozen()
                        else user_root)
        status, detail = apply_pending_delta(user_root, install_root)
        print(f"[launcher] OTA delta apply: {status} — {detail}")
    except Exception as exc:
        print(f"[launcher] delta apply skipped (fail-safe): {exc}")


# Self-updater run AFTER MAST exits, so it can replace the locked exe /
# base_library.zip that a code delta touches (Windows can't overwrite a running
# image in place). Python-free (system PowerShell) so it doesn't depend on the
# _internal it's patching. Reads a plan JSON (paths already sha256-verified +
# traversal-checked by stage_offline_delta), backs up each file before
# overwriting, rolls back on any failure, then relaunches MAST either way
# (feature B, 2026-07-03).
_OFFLINE_APPLY_PS1 = r"""
param([Parameter(Mandatory=$true)][string]$PlanPath)
$ErrorActionPreference = 'Stop'
try {
  $plan  = Get-Content -Raw -LiteralPath $PlanPath | ConvertFrom-Json
  $stage = $plan.stage_dir
  $root  = $plan.install_root
  $exe   = Join-Path $root 'MAST.exe'
  $rootPrefix = if ($root.EndsWith('\')) { $root } else { $root + '\' }
  $log   = Join-Path $root ('_delta_apply_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
  $backup = Join-Path $root '_delta_backup'
  function Log($m) { try { Add-Content -LiteralPath $log -Value ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $m) } catch {} }
  Log ('offline delta apply: {0} -> {1}' -f $plan.from_version, $plan.to_version)

  # 1) Wait for every MAST process under the install root to exit (max 180s).
  $deadline = (Get-Date).AddSeconds(180)
  while ((Get-Date) -lt $deadline) {
    $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
      $_.Path -and $_.Path.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase) })
    if ($running.Count -eq 0) { break }
    Start-Sleep -Milliseconds 500
  }
  Start-Sleep -Seconds 2   # let file handles fully release

  # 2) Backup then overwrite staged files; backup then delete removed.
  if (Test-Path -LiteralPath $backup) { Remove-Item -Recurse -Force -LiteralPath $backup }
  New-Item -ItemType Directory -Force -Path $backup | Out-Null
  $failed = $null
  try {
    foreach ($rel in $plan.copies) {
      $src = Join-Path $stage $rel; $dst = Join-Path $root $rel
      $dstDir = Split-Path -Parent $dst
      if (-not (Test-Path -LiteralPath $dstDir)) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
      if (Test-Path -LiteralPath $dst) {
        $bdst = Join-Path $backup $rel; $bdir = Split-Path -Parent $bdst
        if (-not (Test-Path -LiteralPath $bdir)) { New-Item -ItemType Directory -Force -Path $bdir | Out-Null }
        Copy-Item -LiteralPath $dst -Destination $bdst -Force
      }
      Copy-Item -LiteralPath $src -Destination $dst -Force
    }
    foreach ($rel in $plan.removed) {
      $dst = Join-Path $root $rel
      if (Test-Path -LiteralPath $dst) {
        $bdst = Join-Path $backup $rel; $bdir = Split-Path -Parent $bdst
        if (-not (Test-Path -LiteralPath $bdir)) { New-Item -ItemType Directory -Force -Path $bdir | Out-Null }
        Copy-Item -LiteralPath $dst -Destination $bdst -Force
        Remove-Item -LiteralPath $dst -Force
      }
    }
  } catch { $failed = $_.Exception.Message }

  if ($failed) {
    Log ('apply FAILED: {0} -- rolling back' -f $failed)
    Get-ChildItem -Recurse -File -LiteralPath $backup -ErrorAction SilentlyContinue | ForEach-Object {
      $rel = $_.FullName.Substring($backup.Length).TrimStart('\')
      $dst = Join-Path $root $rel; $dstDir = Split-Path -Parent $dst
      if (-not (Test-Path -LiteralPath $dstDir)) { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
      Copy-Item -LiteralPath $_.FullName -Destination $dst -Force
    }
    Log 'rollback complete; relaunching previous version'
  } else {
    Log ('apply OK: {0} copied, {1} removed -> {2}' -f $plan.copies.Count, $plan.removed.Count, $plan.to_version)
    Remove-Item -Recurse -Force -LiteralPath $stage -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -LiteralPath $backup -ErrorAction SilentlyContinue
    $pdir = Split-Path -Parent $stage
    Remove-Item -Force -LiteralPath (Join-Path $pdir 'delta.json') -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $pdir -Filter 'delta_*.zip' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Remove-Item -Force -LiteralPath $PlanPath -ErrorAction SilentlyContinue
  }

  # 3) Relaunch MAST (new version on success, old on rollback).
  Log ('relaunching {0}' -f $exe)
  Start-Process -FilePath $exe -WorkingDirectory $root
} catch {
  try { Add-Content -LiteralPath (Join-Path $env:TEMP 'mast_delta_apply_error.log') -Value $_.Exception.Message } catch {}
  try { Start-Process -FilePath (Join-Path $plan.install_root 'MAST.exe') } catch {}
}
""".lstrip()


def _notify_offline_apply(version: str) -> None:
    """Best-effort 2.5s toast so the user knows MAST is restarting to update."""
    try:
        import tkinter as tk
        r = tk.Tk()
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        tk.Label(r, text=f"正在应用增量更新 → v{version}\nMAST 即将自动重启…",
                 font=("Segoe UI", 11), padx=26, pady=18).pack()
        r.update_idletasks()
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w, h = r.winfo_reqwidth(), r.winfo_reqheight()
        r.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        r.after(2500, r.destroy)
        r.mainloop()
    except Exception:  # noqa: BLE001 — cosmetic only
        pass


def _apply_pending_delta_offline_if_needed(user_root: Path) -> bool:
    """A pending delta that is NOT hot-appliable (touches code / exe /
    base_library.zip) can't be written while MAST runs. Stage it (verified) and
    spawn a self-updater that applies it AFTER MAST exits, then relaunches.
    Returns True iff the launcher should exit (applier spawned). Fail-safe: any
    problem returns False so the full-installer fallback still runs."""
    try:
        import json as _json
        from mast.update.client import (
            DELTA_MARKER, delta_is_data_only, pending_dir, stage_offline_delta,
        )
        from mast.update.delta import read_delta_manifest
    except Exception as exc:
        print(f"[launcher] offline delta module unavailable: {exc}")
        return False
    try:
        pdir = pending_dir(user_root)
        marker = pdir / DELTA_MARKER
        if not marker.exists():
            return False
        info = _json.loads(marker.read_text(encoding="utf-8"))
        dz = pdir / str(info.get("filename", ""))
        if not dz.exists():
            return False
        man = read_delta_manifest(dz)
        if not _pending_delta_from_current(man, pdir):
            return False  # stale orphan (from a prior version) — skipped + cleared
        paths = (list(man.get("added", [])) + list(man.get("changed", []))
                 + list(man.get("removed", [])))
        if delta_is_data_only(paths):
            return False  # already hot-applied live by _apply_pending_delta_if_safe
        install_root = (Path(sys.executable).resolve().parent if _frozen()
                        else user_root)
        status, plan = stage_offline_delta(user_root, install_root)
        if status != "staged" or not isinstance(plan, dict):
            print(f"[launcher] offline delta stage failed ({status}: {plan}) — full-installer fallback")
            return False
        plan_path = pdir / "apply_plan.json"
        ps1_path = pdir / "_offline_apply.ps1"
        plan_path.write_text(_json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        ps1_path.write_text(_OFFLINE_APPLY_PS1, encoding="utf-8")
        # Spawn the self-updater ROBUSTLY. The old DETACHED_PROCESS (no console) +
        # `-WindowStyle Hidden` + PyInstaller's injected env + no std handles made
        # the spawned powershell die with ERROR_BAD_EXE_FORMAT before it could run
        #. Fix: full path to powershell.exe, a CLEAN env (drop
        # the PyInstaller _MEI/_PYI vars), CREATE_NO_WINDOW (a hidden console — the
        # child still gets valid handles and survives the launcher exiting), and
        # DEVNULL std handles.
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        ps_exe = os.path.join(sysroot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
        if not os.path.exists(ps_exe):
            ps_exe = "powershell"
        child_env = {k: v for k, v in os.environ.items()
                     if not (k.startswith("_MEI") or k.startswith("_PYI"))}
        CREATE_NO_WINDOW = 0x08000000
        flags = CREATE_NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(
            [ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(ps1_path), str(plan_path)],
            cwd=str(user_root), creationflags=flags, close_fds=True, env=child_env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[launcher] offline delta applier spawned for v{plan.get('to_version')} — exiting to let it patch")
        _notify_offline_apply(str(plan.get("to_version", "")))
        return True
    except Exception as exc:
        print(f"[launcher] offline delta apply skipped (fail-safe): {exc}")
        return False


def _check_and_run_pending_update(user_root: Path) -> bool:
    """If a verified pending update exists, show modal + run installer.

    Returns True if the launcher should exit (because the installer was
    started); False if no update was applied and the launcher should
    continue normally.
    """
    try:
        from mast.update.client import archive_installed, load_pending
    except Exception as exc:
        # Module missing in older bundles or import error — never block startup.
        print(f"[launcher.main] update client unavailable: {exc}")
        return False

    manifest, setup_path = load_pending(user_root)
    if manifest is None or setup_path is None:
        return False

    # Check that we're actually older than the pending version (defensive —
    # avoid prompting if user manually downgrades or the file is stale).
    try:
        from mast.update.manifest import is_newer
        from mast import __version__ as current_version
    except Exception:
        return False
    if not is_newer(manifest.version, current_version):
        # Old or equal — clean it up so we don't keep nagging.
        archive_installed(user_root)
        return False

    # Show a Tk modal. Single button, no close.
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        # No Tk → just run the installer directly.
        _run_installer_and_exit(setup_path, user_root, manifest.version)
        return True

    _enable_dpi_awareness()

    root = tk.Tk()
    root.title("MAST 强制更新")
    root.resizable(False, False)
    # No window-close button: this is a forced update.
    root.protocol("WM_DELETE_WINDOW", lambda: None)
    try:
        logo = tk.PhotoImage(file=str(_resource_path(LOGO_PNG_RELPATH)))
        root.iconphoto(True, logo)
    except Exception:
        logo = None  # noqa: F841

    frm = ttk.Frame(root, padding=20)
    frm.grid()
    ttk.Label(
        frm, text=f"MAST 有新版本 v{manifest.version}",
        font=("Segoe UI", 14, "bold"),
    ).grid(row=0, column=0, sticky="w")
    ttk.Label(
        frm,
        text=(
            f"当前版本: v{current_version}\n"
            f"待安装版本: v{manifest.version}\n"
            f"发布时间: {manifest.published_at}\n"
            f"文件大小: {manifest.size_bytes // (1024*1024)} MB\n\n"
            "本次更新为**强制更新**，必须安装后才能继续使用 MAST。\n"
            "升级会保留所有用户数据（API key / 实验 / 模型 / 配置）。\n"
        ),
        justify="left",
    ).grid(row=1, column=0, sticky="w", pady=(8, 12))

    def _install():
        root.destroy()
        _run_installer_and_exit(setup_path, user_root, manifest.version)

    ttk.Button(
        frm, text="立即安装", command=_install,
    ).grid(row=2, column=0, sticky="e")

    # Center on screen
    root.update_idletasks()
    w = root.winfo_reqwidth(); h = root.winfo_reqheight()
    sw = root.winfo_screenwidth(); sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
    root.mainloop()
    return True


def _run_installer_and_exit(setup_path: Path, user_root: Path, version: str) -> None:
    """Spawn the Inno Setup installer (elevated) and exit the launcher.

    v0.3.5 fix: previously used ``subprocess.Popen([..., "/SILENT", ...])``
    which loses the UAC prompt to the background and silently fails when
    the user dismisses it (or doesn't see it). Now uses ShellExecuteW with
    the "runas" verb — Windows's standard "request elevation" call —
    which pops UAC to the foreground with focus. We also drop ``/SILENT``
    so the user sees the Inno Setup wizard's progress bar (and any error
    dialogs); ``/SUPPRESSMSGBOXES`` still auto-confirms the data-dir page.
    """
    print(f"[launcher.main] running installer: {setup_path} (v{version})")
    args = "/SP- /SUPPRESSMSGBOXES /NORESTART"
    rc = -1
    try:
        import ctypes
        # ShellExecuteW(hwnd, verb, file, params, dir, show_cmd)
        # Returns >32 on success (the HINSTANCE-shaped result).
        rc = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", str(setup_path), args, str(setup_path.parent), 1,
        )
    except Exception as exc:
        print(f"[launcher.main] ShellExecuteW failed: {exc}")
        rc = 0  # force fallback below

    if rc <= 32:
        # ShellExecuteW failed (UAC denied, file not found, etc.). Fall back
        # to plain Popen — the OS will still try its own UAC flow, just
        # without the foreground guarantee.
        print(f"[launcher.main] ShellExecuteW returned {rc}; falling back to Popen")
        try:
            subprocess.Popen(
                [str(setup_path), "/SP-", "/SUPPRESSMSGBOXES", "/NORESTART"],
                cwd=str(setup_path.parent),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except Exception as exc:
            print(f"[launcher.main] Popen fallback also failed: {exc}")
            return
    # IMPORTANT: do NOT call archive_installed here. ShellExecuteW returns
    # asynchronously; the Inno Setup process hasn't actually opened the
    # .exe file yet. Moving the file out from under it instantly causes
    # Windows to fail elevation with "系统找不到指定的文件" (ERROR_FILE_NOT_FOUND).
    # The next launcher start handles cleanup via _check_and_run_pending_update's
    # "is_newer(remote, current_version)" check — if we're already on the
    # version that pending_update advertises, archive_installed runs there
    # instead, well after the file is no longer needed.


def _enable_dpi_awareness() -> None:
    """Tell Windows we draw at native pixel scale so 200% DPI doesn't blur Tk.

    On Windows 10+ this asks for per-monitor v2 DPI awareness; older builds
    silently fall back to system-DPI awareness. Failure is non-fatal — we
    just live with bitmap-stretched fonts.
    """
    try:
        import ctypes
        # Per-monitor v2 (best on Win10+)
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(
                ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            )
            return
        except Exception:
            pass
        # Per-monitor v1 (Win8.1+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
        # System-DPI (Win 7+)
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ── Launcher window ──────────────────────────────────────────────────

class LauncherApp:
    """Tk control panel for the MAST service."""

    def __init__(self, root_dir: Path, *, start_minimized: bool = False) -> None:
        self.root_dir = root_dir
        # Auto-start (open-on-boot) launches with --minimized so the window
        # doesn't pop every login — it goes straight to the tray (resolves the
        # auto-start ↔ ×-to-tray interaction). Applied at the end of __init__.
        self._start_minimized = bool(start_minimized)
        # Clear any LAN-auth handoff files left behind by a prior session that
        # crashed between writing creds and the child consuming them.
        _sweep_lan_auth_handoff(root_dir)
        self.proc: subprocess.Popen | None = None
        self.push_proc: subprocess.Popen | None = None
        self.log_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._push_reader_thread: threading.Thread | None = None
        # LAN IP is detected once at start-up; manual refresh re-reads it.
        self._lan_ip: str = _lan_ip()
        # Tailscale status (cross-network remote access) is detected off-thread —
        # shelling out to the CLI can take up to a few seconds, so we never do it
        # on the Tk thread. None = "not yet detected"; a TailscaleStatus once known.
        self._ts_status = None
        # LAN access toggle is sticky across runs (persisted via lan_auth.env
        # presence). Treat "have credentials" as "user wants LAN enabled".
        u, p = get_lan_auth(root_dir)
        self._lan_enabled: bool = bool(u and p)
        self._lan_user: str = u
        self._lan_pass: str = p

        # Vision model is ALWAYS on — it loads WITH the main service (no separate
        # switch, no user-facing off: the software has no "vision off" run mode,
        # only VisionModule's automatic Mock fallback when a GPU can't load it).
        # Starting/stopping the service IS the vision on/off, and it is immediate
        # (the model dies with the service process → VRAM freed). On start we show
        # a GPU/load WARNING (unless «下次不再提示» was ticked) + a progress bar.
        self._vision_warn_suppressed: bool = _load_vision_warn_suppressed(root_dir)
        # Set while we're tracking a service warm-up via stdout markers.
        self._vision_watching: bool = False

        # Build UI
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk

        self.win = tk.Tk()
        # Title shows version + release date for at-a-glance "what am I running"
        try:
            from mast import __version__ as _ver, __release_date__ as _rel
            self.win.title(f"MAST 启动器 — v{_ver}  ({_rel})")
        except Exception:
            self.win.title("MAST 启动器")

        # ── DPI-aware sizing & font scaling ─────────────────────────
        # When DPI awareness is enabled (called from main() before us),
        # winfo_fpixels('1i') reports the OS-side DPI. Default is 96; 200%
        # display = 192. We use this to:
        #   1. set Tk's internal scaling so fonts auto-grow
        #   2. scale our default geometry (was 820 fixed)
        # We cap the result so a 4K screen doesn't get a giant window.
        try:
            dpi = float(self.win.winfo_fpixels("1i"))
        except Exception:
            dpi = 96.0
        self._dpi_scale = max(1.0, dpi / 96.0)  # 1.0=100%, 2.0=200%
        # Tk's default scaling is ~1.333 (96 dpi → 1.333 px/pt); we want it
        # proportional to actual DPI so text doesn't stay 96-dpi-sized.
        try:
            self.win.tk.call("tk", "scaling", 1.333 * self._dpi_scale)
        except Exception:
            pass
        # NOTE: ttk fonts are sized in *points*; Tk's scaling factor (set
        # above) auto-converts pt → px for the active DPI, so we just use
        # natural point sizes (10, 12, …) below and don't pre-multiply.
        # Scale-derived default geometry.  Base = 820×820 logical px @ 96 dpi.
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        base_w = int(820 * self._dpi_scale)
        base_h = int(820 * self._dpi_scale)
        # Cap at ~70% of screen so it never fills the monitor.
        win_w = max(820, min(base_w, int(sw * 0.70)))
        win_h = max(680, min(base_h, int(sh * 0.85)))
        self.win.geometry(f"{win_w}x{win_h}")
        self.win.minsize(int(720 * self._dpi_scale), int(680 * self._dpi_scale))
        # × never exits (Nanonis-style) — it minimizes to tray. The only real
        # exit is the «关闭启动器» button / tray «退出», which gracefully stops the
        # service + frees VRAM (see _minimize_to_tray / _shutdown_launcher).
        self.win.protocol("WM_DELETE_WINDOW", self._minimize_to_tray)

        # ── Logo (window icon + taskbar) ────────────────────────────
        # iconphoto needs a tk.PhotoImage. Failing silently is fine — the
        # logo is purely cosmetic; missing asset shouldn't crash startup.
        try:
            logo_path = _resource_path(LOGO_PNG_RELPATH)
            if logo_path.exists():
                self._logo_img = tk.PhotoImage(file=str(logo_path))
                self.win.iconphoto(True, self._logo_img)
        except Exception:
            self._logo_img = None  # noqa: F841

        # ── Bind IPC listener for single-instance "bring foreground" ──
        self._ipc_sock = None
        self._ipc_port = 0
        bind = _bind_ipc_port()
        if bind is not None:
            self._ipc_sock, self._ipc_port = bind
            # Write lock with our IPC port so a future second instance can
            # connect to us instead of starting up.
            _write_launcher_lock(self.root_dir, self._ipc_port)
            # Schedule periodic IPC poll
            self.win.after(500, self._poll_ipc)
        else:
            print("[launcher] could not bind IPC port — single-instance ping disabled")

        # ── Tray icon support (pystray + Pillow). Optional. ──
        self._tray_icon = None
        self._tray_thread = None
        self._init_tray()

        self._build_ui()
        self._refresh_status()
        # Kick off Tailscale detection in the background (updates the 远程访问
        # status/URL rows once it returns, without blocking UI startup).
        self._detect_tailscale_async()

        # Open-on-boot: go straight to the tray instead of popping the window.
        # Deferred so the tray icon thread + window are fully up first.
        if self._start_minimized:
            self.win.after(400, self._minimize_to_tray)

    # ── Layout ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        ttk = self.ttk
        tk = self.tk
        pad = {"padx": 12, "pady": 6}
        # Label wraplengths must scale with DPI (the window is ~820*dpi_scale wide).
        # A fixed 100%-px wraplength forces ugly early wrapping at 200% ("字换行乱").
        wl = lambda base: int(base * getattr(self, "_dpi_scale", 1.0))  # noqa: E731

        # Wrap the entire launcher UI in a Canvas + vertical Scrollbar so
        # the layout is usable on small / low-DPI screens (1366×768 laptop
        # was clipping the bottom buttons before this).
        scroll_container = ttk.Frame(self.win)
        scroll_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_container, highlightthickness=0,
                           borderwidth=0)
        canvas.pack(side="left", fill="both", expand=True)

        vscroll = ttk.Scrollbar(scroll_container, orient="vertical",
                                command=canvas.yview)
        vscroll.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=vscroll.set)

        outer = ttk.Frame(canvas, padding=12)
        outer_window = canvas.create_window((0, 0), window=outer, anchor="nw")

        def _on_outer_configure(_event=None):
            # Inner frame changed → recompute scroll region.
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Stretch the inner frame to match canvas width so widgets that
            # use `fill="x"` actually span the available width.
            canvas.itemconfigure(outer_window, width=event.width)

        outer.bind("<Configure>", _on_outer_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse-wheel scrolling — bind to the launcher's toplevel only so we
        # don't interfere with other Tk windows the user may open later.
        def _on_mousewheel(event):
            # event.delta is ±120 per notch on Windows; on macOS it's smaller.
            canvas.yview_scroll(int(-event.delta / 120), "units")
        self.win.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        # ── Service controls (main GUI only) ─────────────────────────
        svc = ttk.LabelFrame(outer, text="服务", padding=12)
        svc.pack(fill="x", **pad)
        for col in range(3):
            svc.columnconfigure(col, weight=1)

        self.svc_status_var = self.tk.StringVar(value="● 主服务: 检测中…")
        self.svc_status_lbl = ttk.Label(
            svc, textvariable=self.svc_status_var, font=("Segoe UI", 10, "bold")
        )
        self.svc_status_lbl.grid(row=0, column=0, columnspan=3, sticky="w")

        # Start / restart / stop the main GUI service.
        self.start_btn = ttk.Button(svc, text="启动主服务", command=self.start_service)
        self.start_btn.grid(row=1, column=0, padx=(0, 6), pady=(8, 0), sticky="we")
        self.restart_btn = ttk.Button(svc, text="重启主服务", command=self.restart_service)
        self.restart_btn.grid(row=1, column=1, padx=6, pady=(8, 0), sticky="we")
        self.stop_btn = ttk.Button(svc, text="关闭主服务", command=self.stop_service)
        self.stop_btn.grid(row=1, column=2, padx=(6, 0), pady=(8, 0), sticky="we")

        # Full-exit affordance lives HERE in 服务 (the window × only minimizes to
        # tray — see _minimize_to_tray — so this is the only real way out).
        # Gracefully stops the main service (frees VRAM) behind a progress bar.
        self.shutdown_btn = ttk.Button(
            svc, text="关闭主服务并退出启动器", command=self._shutdown_launcher)
        self.shutdown_btn.grid(row=2, column=0, columnspan=3, padx=0, pady=(8, 0),
                               sticky="we")

        # ── Vision model (DINOv3 / M12) — loads WITH the service, no switch ──
        # There is NO separate on/off (and no "vision off" run mode): the vision
        # model loads as part of «启动主服务» and dies with it (VRAM freed on stop).
        # This frame is purely informational — a status line + the load progress
        # bar. The GPU/load warning is shown by start_service (see that method).
        vis = ttk.LabelFrame(outer, text="视觉模型 (Vision)", padding=12)
        vis.pack(fill="x", **pad)
        vis.columnconfigure(0, weight=1)

        self.vision_status_var = self.tk.StringVar(
            value="视觉模型: 随主服务自动加载 (DINOv3 / M12)"
        )
        self.vision_status_lbl = ttk.Label(
            vis, textvariable=self.vision_status_var, font=("Segoe UI", 10, "bold"),
            foreground="#888", wraplength=wl(560), justify="left",
        )
        self.vision_status_lbl.grid(row=0, column=0, columnspan=2, sticky="w")

        # Indeterminate progress bar + stage label, hidden until a warm-up runs.
        self.vision_progress = ttk.Progressbar(vis, mode="indeterminate", length=240)
        self.vision_progress.grid(row=2, column=0, columnspan=2, sticky="we",
                                  pady=(8, 0))
        self.vision_progress.grid_remove()
        self.vision_progress_var = self.tk.StringVar(value="")
        self.vision_progress_lbl = ttk.Label(
            vis, textvariable=self.vision_progress_var, foreground="#666",
        )
        self.vision_progress_lbl.grid(row=3, column=0, columnspan=2, sticky="w",
                                      pady=(2, 0))
        self.vision_progress_lbl.grid_remove()

        # ── LAN access toggle (binds 0.0.0.0 + HTTP basic auth) ────
        lan = ttk.LabelFrame(outer, text="局域网 / 远程访问", padding=12)
        lan.pack(fill="x", **pad)
        lan.columnconfigure(1, weight=1)

        self.lan_enabled_var = self.tk.BooleanVar(value=self._lan_enabled)
        self.lan_chk = ttk.Checkbutton(
            lan, text="启用 (其它电脑可通过局域网 IP 或 Tailscale 远程访问 MAST)",
            variable=self.lan_enabled_var, command=self._on_lan_toggle,
        )
        self.lan_chk.grid(row=0, column=0, columnspan=3, sticky="w")

        self.lan_info_var = self.tk.StringVar(value="(未启用)")
        ttk.Label(lan, textvariable=self.lan_info_var, foreground="#666",
                  wraplength=wl(420), justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0),
        )
        ttk.Button(lan, text="修改账号密码", command=self._configure_lan_auth).grid(
            row=1, column=2, sticky="e", pady=(4, 0),
        )
        # Tailscale (跨网远程) status line — set by _refresh_status from the
        # off-thread detection. Reuses the SAME 0.0.0.0 bind as LAN access; this
        # line just tells the user the address to open on the OTHER computer.
        self.ts_info_var = self.tk.StringVar(value="🌐 Tailscale: 检测中…")
        ttk.Label(lan, textvariable=self.ts_info_var, foreground="#666",
                  wraplength=wl(470), justify="left").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0),
        )

        # ── GUI URLs (2 rows: main local/LAN) ──────────────────────
        gui = ttk.LabelFrame(outer, text="GUI 网址", padding=12)
        gui.pack(fill="x", **pad)
        gui.columnconfigure(1, weight=1)

        self.url_var     = self.tk.StringVar(value=f"{self._gui_scheme()}://{GUI_HOST}:{GUI_PORT}")
        self.lan_url_var = self.tk.StringVar(value="")
        # Tailscale (跨网) address to open on the OTHER computer; filled by
        # _refresh_status only when Tailscale is ready AND remote access is on.
        self.ts_url_var  = self.tk.StringVar(value="")

        # Each row: label, var, "open" callback (with running-check), "copy" callback.
        # require_running=None means we don't gate (browser will just fail).
        url_rows = [
            ("主服务 - 本机",     self.url_var,
             lambda: self._open_url(self.url_var.get(), require_running=gui_running),
             lambda: self._copy_url_var(self.url_var)),
            ("主服务 - 局域网",   self.lan_url_var,
             lambda: self._open_url(self.lan_url_var.get(), require_running=gui_running),
             lambda: self._copy_url_var(self.lan_url_var)),
            ("远程 - Tailscale",  self.ts_url_var,
             lambda: self._open_url(self.ts_url_var.get(), require_running=gui_running),
             lambda: self._copy_url_var(self.ts_url_var)),
        ]
        for i, (label, var, open_cb, copy_cb) in enumerate(url_rows):
            ttk.Label(gui, text=label, width=18, anchor="w").grid(
                row=i, column=0, sticky="w", padx=(0, 6), pady=2,
            )
            entry = ttk.Entry(gui, textvariable=var, state="readonly")
            entry.grid(row=i, column=1, sticky="we", padx=(0, 8), pady=2)
            ttk.Button(gui, text="浏览器打开", command=open_cb).grid(
                row=i, column=2, padx=(0, 6), pady=2,
            )
            ttk.Button(gui, text="复制", command=copy_cb).grid(
                row=i, column=3, padx=(0, 0), pady=2,
            )

        # ── Status indicators ───────────────────────────────────────
        st = ttk.LabelFrame(outer, text="状态指示", padding=12)
        st.pack(fill="x", **pad)
        st.columnconfigure(1, weight=1)

        # Header row: hint on left, manual refresh button on right.
        st_header = ttk.Frame(st)
        st_header.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 6))
        st_header.columnconfigure(0, weight=1)
        ttk.Label(
            st_header,
            text="（不自动刷新；启停服务后会自动更新；其它情况点 «刷新»）",
            foreground="#888", wraplength=wl(460), justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(st_header, text="刷新", command=self._manual_refresh).grid(
            row=0, column=1, sticky="e",
        )

        self.indicators: dict[str, dict] = {}
        rows: list[tuple[str, str]] = [
            ("anthropic", "Anthropic API key"),
            ("moonshot",  "Kimi API key"),
            ("deepseek",  "DeepSeek API key"),
            ("dashscope", "阿里云百炼 API key"),
            ("minimax",   "MiniMax API key"),
            ("zhipu",     "智谱 GLM API key"),
            ("nanonis",   "Nanonis 仪器"),
        ]
        for i, (key, label) in enumerate(rows, start=1):
            dot = ttk.Label(st, text="●", font=("Segoe UI", 12, "bold"), foreground="#888")
            dot.grid(row=i, column=0, sticky="w", padx=(0, 8), pady=2)
            text = ttk.Label(st, text=label)
            text.grid(row=i, column=1, sticky="w", pady=2)
            value = ttk.Label(st, text="—", foreground="#888")
            value.grid(row=i, column=2, sticky="e", pady=2)
            self.indicators[key] = {"dot": dot, "value": value, "text": text}

        # ── API key configuration ───────────────────────────────────
        cfg = ttk.LabelFrame(outer, text="API key 配置", padding=12)
        cfg.pack(fill="x", **pad)
        for col in (0, 1, 2, 3):
            cfg.columnconfigure(col, weight=1)
        ttk.Button(
            cfg, text="配置 Kimi", command=lambda: self._configure_provider("moonshot")
        ).grid(row=0, column=0, sticky="we", padx=4)
        ttk.Button(
            cfg, text="配置 DeepSeek", command=lambda: self._configure_provider("deepseek")
        ).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(
            cfg, text="配置 阿里云百炼", command=lambda: self._configure_provider("dashscope")
        ).grid(row=0, column=2, sticky="we", padx=4)
        ttk.Button(
            cfg, text="配置 Anthropic", command=lambda: self._configure_provider("anthropic")
        ).grid(row=0, column=3, sticky="we", padx=4)
        ttk.Button(
            cfg, text="配置 MiniMax", command=lambda: self._configure_provider("minimax")
        ).grid(row=1, column=0, sticky="we", padx=4, pady=(6, 0))
        ttk.Button(
            cfg, text="配置 智谱 GLM", command=lambda: self._configure_provider("zhipu")
        ).grid(row=1, column=1, sticky="we", padx=4, pady=(6, 0))

        # ── Auto update (client side only — server is admin/CLI-controlled) ──
        # The push SERVER (publishing new versions) is intentionally NOT
        # exposed in the launcher GUI from v0.3.2 onwards. End users
        # shouldn't see start/stop/publish controls. The admin runs the
        # server via the CLI:
        #   MAST.exe --push-server-mode
        #   MAST.exe --publish <setup.exe> --version X.Y.Z
        # The bundle ships with a baked-in DEFAULT_SERVER_URL +
        # DEFAULT_TOKEN (mast.update.defaults) so clients connect out of
        # the box. User can still override via the dialog below.
        upd = ttk.LabelFrame(outer, text="自动更新", padding=12)
        upd.pack(fill="x", **pad)
        for col in (0, 1, 2):
            upd.columnconfigure(col, weight=1)
        self.update_status_var = self.tk.StringVar(value="● 自动更新: 检测中…")
        self.update_status_lbl = ttk.Label(
            upd, textvariable=self.update_status_var, font=("Segoe UI", 10, "bold"),
        )
        self.update_status_lbl.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(
            upd, text="立即检查更新", command=self._check_update_now,
        ).grid(row=1, column=0, padx=4, pady=(8, 0), sticky="we")
        ttk.Button(
            upd, text="修改服务器/Token…", command=self._configure_update_client_dialog,
        ).grid(row=1, column=1, padx=4, pady=(8, 0), sticky="we")
        ttk.Button(
            upd, text="清除自定义 (恢复默认)", command=self._reset_update_overrides,
        ).grid(row=1, column=2, padx=4, pady=(8, 0), sticky="we")
        # Current-version info + release-notes button.
        try:
            from mast import __version__ as _cv, __release_date__ as _cd
        except Exception:
            _cv, _cd = "?", ""
        ttk.Label(
            upd,
            text=f"当前版本 v{_cv}  ({_fmt_release_date(_cd)})",
            foreground="#666",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(
            upd, text="当前版本更新说明", command=self._open_release_notes,
        ).grid(row=2, column=2, padx=4, pady=(8, 0), sticky="we")

        # ── 启动器选项: 系统托盘 + 开机自启 ────────────────────────
        opts = ttk.LabelFrame(outer, text="启动器选项", padding=12)
        opts.pack(fill="x", **pad)
        for col in (0, 1, 2):
            opts.columnconfigure(col, weight=1)

        self.autostart_var = self.tk.BooleanVar(value=_startup_enabled())
        ttk.Checkbutton(
            opts,
            text="开机自启 (HKCU Run)",
            variable=self.autostart_var,
            command=self._on_autostart_toggle,
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        ttk.Button(
            opts, text="最小化到托盘", command=self._minimize_to_tray,
        ).grid(row=0, column=1, sticky="we", padx=4)

        # 高级管理 (web tab) PIN is configured HERE in the launcher (local-only),
        # not in the browser. We store the sha256 hex to <root>/config/admin_pin.txt;
        # the web 高级管理 tab reads + compares it.
        ttk.Button(
            opts, text="设置高级管理密码", command=self._set_admin_pin,
        ).grid(row=0, column=2, sticky="we", padx=4)

        ttk.Label(
            opts,
            text=("× 关闭窗口 = 缩到状态栏，主服务继续运行（避免误关泄漏显存）；"
                  "要退出请点「服务」栏的「关闭主服务并退出启动器」。"
                  "「高级管理密码」设定网页「高级管理」标签的解锁口令（仅存本机）。"),
            foreground="#888",
            wraplength=wl(600),
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # ── Service log (scrolling text — large by default) ─────────
        log_frame = ttk.LabelFrame(outer, text="服务日志", padding=4)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(8, 4))

        # Header row with "open logs dir" button — survives launcher restarts
        # because logs are also written to disk under experiments/logs/.
        log_header = ttk.Frame(log_frame)
        log_header.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 4))
        log_header.columnconfigure(0, weight=1)
        ttk.Label(
            log_header,
            text=f"（同时持久化到 {self.root_dir / 'experiments' / 'logs'}）",
            foreground="#888", wraplength=wl(480), justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            log_header, text="打开日志目录", command=self._open_logs_dir,
        ).grid(row=0, column=1, sticky="e", padx=(0, 6))
        ttk.Button(
            log_header, text="清屏", command=self._clear_log_widget,
        ).grid(row=0, column=2, sticky="e")

        # height=20 makes the log meaningfully readable without scrolling.
        # fill="both" + expand=True lets it grow when the user enlarges the
        # window (was previously capped by sibling frames absorbing slack).
        self.log_text = self.tk.Text(
            log_frame, height=20, wrap="none", font=("Consolas", 10),
            state="disabled", background="#1e1e1e", foreground="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        yscroll.grid(row=1, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)
        xscroll.grid(row=2, column=0, sticky="we")
        self.log_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log("MAST 启动器就绪。")
        self._log(f"用户数据目录: {self.root_dir}")
        if self._lan_ip:
            self._log(f"检测到本机局域网 IP: {self._lan_ip}")
        else:
            self._log("未检测到局域网 IP（可能未联网或仅本地接口）。")
        if self._lan_enabled:
            self._log(f"局域网访问已启用 (用户名: {self._lan_user})")

    # ── Service control ──────────────────────────────────────────────

    def _make_subprocess_env(self) -> dict[str, str]:
        """Env for a child subprocess. Hands off LAN auth securely when enabled.

        The password is NOT placed in the env block (see LAN_ENV_AUTH_FILE
        rationale near the top of the module). Instead we write the creds to a
        fresh owner-only temp file per call and pass only its path; the child
        reads + deletes it on boot. Each call gets a unique file, so the main
        service and admin children never contend for the same one.
        """
        env = os.environ.copy()
        # Always strip any inherited LAN vars first so a stale parent env can
        # never leak into a child after the user toggles LAN off.
        for k in (LAN_ENV_ENABLE, LAN_ENV_USER, LAN_ENV_PASS, LAN_ENV_AUTH_FILE):
            env.pop(k, None)
        if self._lan_enabled and self._lan_user and self._lan_pass:
            env[LAN_ENV_ENABLE] = "1"
            env[LAN_ENV_USER]   = self._lan_user   # username is not a secret
            try:
                handoff = write_lan_auth_handoff(
                    self.root_dir, self._lan_user, self._lan_pass
                )
                env[LAN_ENV_AUTH_FILE] = str(handoff)
            except Exception as exc:
                # If we can't write the secure file, fall back to the legacy
                # plaintext env var so LAN access still works (the child
                # otherwise sees no creds and safely stays on 127.0.0.1, but
                # then the user's LAN toggle silently does nothing). Log the
                # downgrade so the weaker secret handling is visible.
                self._log(f"⚠️ 安全降级：无法写入临时凭据文件 ({exc})，"
                          "改用环境变量传递密码。")
                env[LAN_ENV_PASS] = self._lan_pass

        # Vision model is always on (loads with the service); we do NOT force a
        # backend here — VisionModule auto-selects vigil/legacy and falls back to
        # Mock by itself if the GPU can't load it. Strip any stale inherited value
        # so a parent env can't pin the backend unexpectedly.
        env.pop("MAST_VISION_BACKEND", None)
        # Force UTF-8 for the child's stdio + logging FileHandlers. Without this,
        # Windows defaults the service/push log files to the cp936 locale codec, so
        # Chinese + em-dashes were written as GBK bytes and rendered as mojibake
        # (��) whenever a UTF-8 reader (our log viewer / grep) read them back
        # (2026-07-06 feedback: "日志中文乱码"). PYTHONUTF8=1 flips the whole child
        # process to UTF-8 mode — stdout pipe AND every open()/FileHandler.
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def start_service(self) -> None:
        """Start the main GUI service (the vision model loads WITH it).

        Vision warms up inside the service (~1–3 min, occupies VRAM). Before
        starting we pop a GPU/load WARNING (unless the user ticked «下次不再提示»);
        confirming proceeds to _do_start_service, which also arms the load
        progress bar. There is no separate vision switch — this IS it.
        """
        if self.proc is not None and self.proc.poll() is None:
            self._log("主服务已在运行。")
            self._refresh_status()
            return
        if gui_running():
            self._log(f"端口 {GUI_PORT} 已被占用 — 可能有外部进程在跑 MAST 主服务。")
            self._refresh_status()
            return
        if self._vision_warn_suppressed:
            self._do_start_service()
        else:
            # Warn first (GPU detect + load caution); 继续启动 → _do_start_service.
            self._show_vision_warning_dialog(on_continue=self._do_start_service)

    def _do_start_service(self) -> None:
        """Actually spawn the service subprocess + arm the vision load progress."""
        cmd = self._build_service_cmd()
        self._log(f"启动主服务子进程: {' '.join(cmd)}")
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.root_dir),
                env=self._make_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except Exception as exc:
            self._log(f"主服务启动失败: {type(exc).__name__}: {exc}")
            self.proc = None
        else:
            self._reader_thread = threading.Thread(
                target=self._read_proc_output, daemon=True
            )
            self._reader_thread.start()
            # The vision model warms inside the service — track its load.
            self._begin_vision_progress()

        self._refresh_status()
        # Service takes ~5–15 s to bind its Gradio port. We removed the
        # always-on poll in v0.2.5 (UI hitch from tasklist), but for the
        # first 30 s after «启动» the user expects the status to flip from
        # 启动中 → 运行中 without manually clicking 刷新.
        self._schedule_post_start_refresh()

    def stop_service(self, *, quiet: bool = False) -> None:
        """Stop the main GUI service."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            if not quiet:
                self._log("主服务未在运行。")
            self.proc = None
        else:
            if not quiet:
                self._log("正在优雅关闭主服务（让其先安全断开 Nanonis 连接）…")
            try:
                # Graceful-first stop: taskkill WITHOUT /F so the service can
                # close the Nanonis TCP link cleanly. Hard-kill ONLY if it
                # refuses within the timeout (a force-kill mid-TCP can corrupt
                # the Nanonis port until Nanonis restarts).
                outcome = _graceful_stop_proc(
                    proc, graceful_timeout=15.0,
                    log=(None if quiet else self._log_threadsafe),
                )
                if not quiet and outcome == "forced":
                    self._log("主服务未优雅退出，已强制结束。")
            except Exception as exc:
                self._log(f"主服务关闭异常: {exc}")
            self.proc = None
            if not quiet:
                self._log("主服务已关闭。")

        self._refresh_status()

    def restart_service(self) -> None:
        """Restart the main GUI service (vision reloads; progress shown, no dialog)."""
        self._log("重启主服务…")
        self.stop_service(quiet=True)
        time.sleep(0.3)
        self._do_start_service()

    # ── Vision model load warning + progress ─────────────────────────

    def _show_vision_warning_dialog(self, on_continue=None) -> None:
        """Toplevel shown before starting the service: GPU detect (background
        thread) + load caution + a «下次不再提示» checkbox + 继续启动 / 取消.

        Mirrors the LAN auth dialog (transient + grab_set + centered, logo icon).
        On 继续启动: persist the suppress choice if ticked, close, then call
        ``on_continue`` (which actually starts the service + the progress bar).
        On 取消: just close (the service is NOT started).
        """
        tk = self.tk
        ttk = self.ttk

        dlg = tk.Toplevel(self.win)
        dlg.title("启动主服务 · 加载视觉模型")
        dlg.transient(self.win)
        dlg.grab_set()
        dlg.resizable(False, False)
        try:
            if getattr(self, "_logo_img", None) is not None:
                dlg.iconphoto(False, self._logo_img)
        except Exception:
            pass

        frm = ttk.Frame(dlg, padding=14)
        frm.grid()

        ttk.Label(
            frm, text="即将启动主服务并加载视觉模型 (DINOv3 / M12)",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # GPU info block — filled in once detection (on a worker thread) returns.
        gpu_var = tk.StringVar(value="检测显卡中…")
        ttk.Label(frm, textvariable=gpu_var, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        support_var = tk.StringVar(value="")
        support_lbl = ttk.Label(frm, textvariable=support_var, justify="left",
                                wraplength=460)
        support_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(
            frm,
            text=(
                "加载视觉模型（DINOv3 主干 ~1.2 GB + 检查点）进显存并预热需要约 "
                "1–3 分钟，期间会占用约 2–4 GB 显存；加载在后台进行，界面可继续"
                "使用。关闭主服务即可释放这部分显存。"
            ),
            justify="left", wraplength=460, foreground="#666",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))

        driver_note_var = tk.StringVar(value="")
        driver_note_lbl = ttk.Label(frm, textvariable=driver_note_var,
                                    justify="left", wraplength=460,
                                    foreground="#dc2626")
        driver_note_lbl.grid(row=4, column=0, columnspan=2, sticky="w")

        # «下次不再提示» — when ticked + 继续启动, future starts skip this dialog
        # (the load progress bar still shows every start).
        suppress_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm, text="下次启动不再显示此警告（仍会显示加载进度条）",
            variable=suppress_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=2, sticky="e", pady=(12, 0))

        state = {"closed": False}

        def _confirm() -> None:
            if suppress_var.get():
                self._vision_warn_suppressed = True
                _save_vision_warn_suppressed(self.root_dir, True)
                self._log("已记住：下次启动不再显示视觉模型警告。")
            state["closed"] = True
            try:
                dlg.destroy()
            except Exception:
                pass
            if callable(on_continue):
                try:
                    on_continue()
                except Exception as exc:
                    self._log(f"启动主服务异常: {exc}")

        def _cancel() -> None:
            state["closed"] = True
            try:
                dlg.destroy()
            except Exception:
                pass
            self._log("已取消启动主服务。")

        confirm_btn = ttk.Button(btns, text="继续启动", command=_confirm)
        confirm_btn.grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="取消", command=_cancel).grid(row=0, column=1, padx=4)
        dlg.bind("<Escape>", lambda _e: _cancel())

        # Center over the launcher window.
        dlg.update_idletasks()
        w = dlg.winfo_reqwidth(); h = dlg.winfo_reqheight()
        x = self.win.winfo_rootx() + (self.win.winfo_width() - w) // 2
        y = self.win.winfo_rooty() + 80
        dlg.geometry(f"{w}x{h}+{x}+{y}")

        # ── Run _detect_gpu() on a worker thread (it shells out) ──
        def _detect_worker() -> None:
            gpu = _detect_gpu()
            level, msg = _vision_support(gpu)

            def _apply() -> None:
                if state["closed"]:
                    return
                try:
                    vram = gpu.get("vram_gb", 0.0)
                    if isinstance(vram, (int, float)) and vram < 0:
                        vram_txt = "未知(>=4)"
                    elif vram:
                        vram_txt = f"{vram} GB"
                    else:
                        vram_txt = "未知"
                    gpu_var.set(
                        f"显卡：{gpu.get('name', '未知')}\n"
                        f"显存：{vram_txt}\n"
                        f"驱动：{gpu.get('driver') or '未知'}\n"
                        f"来源：{gpu.get('source', 'none')}"
                    )
                    support_var.set(msg)
                    color = {
                        "ok": "#22c55e", "marginal": "#d97706",
                        "need_driver": "#dc2626", "unsupported": "#dc2626",
                    }.get(level, "#666")
                    support_lbl.configure(foreground=color)
                    if level == "need_driver":
                        driver_note_var.set(
                            "注意：需要你先到 NVIDIA 官网下载并自行更新显卡驱动。"
                            "仍可继续开启，但很可能回退到 Mock 后端。"
                        )
                except Exception:
                    pass

            try:
                self.win.after(0, _apply)
            except Exception:
                pass

        threading.Thread(target=_detect_worker, daemon=True).start()

    def _begin_vision_progress(self) -> None:
        """Show the indeterminate progress bar + stage text and arm the
        stdout-marker watcher. A 3-min safety timeout stops the bar if no
        completion marker arrives."""
        try:
            self._vision_watching = True
            self.vision_progress.grid()
            self.vision_progress_lbl.grid()
            self.vision_progress.start(12)
            self.vision_progress_var.set("启动服务…")
        except Exception as exc:
            self._log(f"视觉进度初始化异常: {exc}")
            return

        def _timeout() -> None:
            if self._vision_watching:
                try:
                    self.vision_progress.stop()
                except Exception:
                    pass
                self.vision_progress_var.set("加载超时，请查看日志")
                self._vision_watching = False

        try:
            self.win.after(180000, _timeout)
        except Exception:
            pass

    def _vision_progress_stage(self, text: str) -> None:
        """Marshalled onto the Tk thread: update the stage label (if watching)."""
        if not self._vision_watching:
            return
        try:
            self.vision_progress_var.set(text)
        except Exception:
            pass

    def _vision_progress_done(self, ok: bool, detail: str = "") -> None:
        """Marshalled onto the Tk thread: finish the progress bar."""
        try:
            self.vision_progress.stop()
        except Exception:
            pass
        self._vision_watching = False
        try:
            if ok:
                self.vision_status_var.set("● 视觉模型: 开启（已加载）")
                self.vision_status_lbl.configure(foreground="#22c55e")
                self.vision_progress.grid_remove()
                self.vision_progress_lbl.grid_remove()
                self.vision_progress_var.set("")
                self._log("视觉模型加载完成。")
            else:
                self.vision_status_var.set("视觉模型加载失败（已回退）")
                self.vision_status_lbl.configure(foreground="#dc2626")
                self.vision_progress_var.set("加载失败（已回退 Mock）")
                self._log(detail or "视觉模型加载失败（已回退）。")
        except Exception:
            pass

    def _build_service_cmd(self) -> list[str]:
        if _frozen():
            # In a frozen build sys.executable IS MAST.exe.
            return [sys.executable, "--service-mode"]
        # In dev, run the same .py file with the current interpreter.
        return [sys.executable, str(Path(__file__).resolve()), "--service-mode"]

    def _read_proc_output(self) -> None:
        """Pump the child stdout into the log widget AND service-*.log."""
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        log_path = _log_dir_for(self.root_dir) / f"service-{_today_stamp()}.log"
        try:
            with open(log_path, "a", encoding="utf-8", buffering=1) as fh:
                fh.write(f"\n=== service start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log_threadsafe(line)
                        try:
                            fh.write(line + "\n")
                        except Exception:
                            pass
                        # Refresh status the moment we see Gradio bind the port —
                        # the user no longer has to wait for the polling tick.
                        # v0.3.5 polled every 2s for 30s, but slow Tk repaint or
                        # cold-start >30s left the status stuck at "启动中".
                        if "Running on local URL" in line or "Uvicorn running on" in line:
                            self.win.after(0, self._refresh_status)
                        # ── Vision warm-up progress (markers from run_service) ──
                        # All Tk updates marshalled via win.after — this runs on
                        # a background thread and must not touch widgets directly.
                        if self._vision_watching:
                            if "MAST_VISION_LOAD_START" in line:
                                self.win.after(
                                    0, self._vision_progress_stage, "加载视觉模型权重…")
                            elif "MAST_VISION_LOAD_DONE" in line:
                                self.win.after(0, self._vision_progress_done, True)
                            elif "MAST_VISION_LOAD_FAILED" in line:
                                self.win.after(
                                    0, self._vision_progress_done, False, line)
                            elif ("backbone" in line or "DINOv3" in line) and (
                                    "load" in line.lower()):
                                self.win.after(
                                    0, self._vision_progress_stage, "加载主干网络…")
        except Exception as exc:
            self._log_threadsafe(f"日志读取异常: {exc}")
        finally:
            rc = proc.poll()
            if rc is not None:
                self._log_threadsafe(f"子进程已退出 (rc={rc})")
                # Final status refresh on subprocess exit so the GUI reflects
                # the dead state.
                self.win.after(0, self._refresh_status)

    # ── Browser / clipboard ──────────────────────────────────────────

    def _gui_scheme(self) -> str:
        """"http", or "https" when the service is actually bound for LAN.

        run_service binds 0.0.0.0 with auto-TLS (https) exactly when LAN access is
        enabled AND creds exist; otherwise it stays 127.0.0.1 over plain http. The
        launcher URLs MUST match: opening http:// against the TLS listener makes the
        server drop the plaintext request with no reply → the browser shows
        ERR_EMPTY_RESPONSE / 白屏."""
        return "https" if (getattr(self, "_lan_enabled", False)
                           and getattr(self, "_lan_user", "")) else "http"

    def _open_url(self, url: str, *, require_running: Callable[[], bool] | None = None) -> None:
        if not url:
            self._log("URL 为空（可能未检测到局域网 IP 或服务未启动）。")
            return
        if require_running is not None and not require_running():
            self._log(f"对应服务未运行，无法打开: {url}")
            return
        try:
            webbrowser.open(url)
        except Exception as exc:
            self._log(f"打开浏览器失败: {exc}")

    def _copy_url_var(self, var) -> None:
        url = var.get() if hasattr(var, "get") else str(var)
        if not url:
            self._log("URL 为空，无法复制。")
            return
        try:
            self.win.clipboard_clear()
            self.win.clipboard_append(url)
            self._log(f"已复制到剪贴板: {url}")
        except Exception as exc:
            self._log(f"复制失败: {exc}")

    def _open_logs_dir(self) -> None:
        log_dir = _log_dir_for(self.root_dir)
        try:
            os.startfile(str(log_dir))  # Windows: opens File Explorer at path
        except Exception as exc:
            self._log(f"打开日志目录失败: {exc} (路径: {log_dir})")

    def _clear_log_widget(self) -> None:
        # Only clears the in-memory Text widget — the on-disk log file is
        # untouched, so we never lose history. Useful when the visible
        # buffer gets too noisy.
        try:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")
            self._log("(屏幕日志已清；磁盘日志未动)")
        except Exception as exc:
            self._log(f"清屏失败: {exc}")

    def _schedule_post_start_refresh(self) -> None:
        """After «启动» / «重启», keep refreshing status until the service is up.

        v0.3.6 fix: previous 30 s window left the status stuck at "启动中"
        when:
          - cold-start exceeded 30 s (rare but real on slow disks)
          - the polling tick fired between widget repaints

        Now we poll every 2 s for the first 30 s, then every 5 s up to 120 s
        total — covers very-slow boots without burning cycles when idle.
        Also self-cancelling: the moment the main subprocess looks fully alive,
        we stop. The stdout reader's "Running on local URL" detector fires a
        refresh immediately too (much faster than waiting for any tick).
        """
        # (interval_ms, tick_count)
        phases = [(2000, 15), (5000, 18)]   # 30 s + 90 s = 120 s window

        def _tick(phase_idx: int, remaining: int) -> None:
            try:
                self._refresh_status()
            except Exception:
                return
            main_up = (
                self.proc is not None and self.proc.poll() is None
                and gui_running()
            )
            if main_up:
                return
            if remaining > 0:
                self.win.after(phases[phase_idx][0],
                               lambda: _tick(phase_idx, remaining - 1))
            elif phase_idx + 1 < len(phases):
                self.win.after(phases[phase_idx + 1][0],
                               lambda: _tick(phase_idx + 1, phases[phase_idx + 1][1]))

        self.win.after(phases[0][0], lambda: _tick(0, phases[0][1]))

    def _detect_tailscale_async(self) -> None:
        """Detect Tailscale off the Tk thread (the CLI can block a few seconds),
        then marshal a status refresh back onto the UI thread."""
        def work() -> None:
            try:
                from mast.net.tailscale import get_status
                st = get_status()
            except Exception:  # detection is best-effort; never crash the UI
                st = None
            self._ts_status = st
            try:
                self.win.after(0, self._refresh_status)
            except Exception:
                pass
        try:
            threading.Thread(target=work, daemon=True).start()
        except Exception:
            pass

    def _manual_refresh(self) -> None:
        # Re-detect LAN IP — laptop may have swapped Wi-Fi networks since boot.
        new_lan = _lan_ip()
        if new_lan != self._lan_ip:
            self._lan_ip = new_lan
            if new_lan:
                self._log(f"局域网 IP 更新为 {new_lan}")
            else:
                self._log("局域网 IP 已失效（可能断网）。")
        # Tailscale may have come up / logged in / changed IP since last check.
        self._detect_tailscale_async()
        self._refresh_status()
        self._log("已手动刷新状态。")

    # ── LAN access toggle / auth dialog ──────────────────────────────

    def _on_lan_toggle(self) -> None:
        """User clicked the «启用 LAN» checkbox."""
        if self.lan_enabled_var.get():
            # Turning ON: prompt for credentials if we don't already have them.
            if not (self._lan_user and self._lan_pass):
                creds = self._prompt_lan_auth_dialog()
                if creds is None:
                    # Cancelled — revert checkbox.
                    self.lan_enabled_var.set(False)
                    self._refresh_status()
                    return
                user, pwd = creds
                save_lan_auth(self.root_dir, user, pwd)
                self._lan_user = user
                self._lan_pass = pwd
            self._lan_enabled = True
            self._log(f"局域网访问已启用 (用户名: {self._lan_user})。"
                      "下次启动主服务时生效；如服务在跑请重启。")
        else:
            # Turning OFF: keep credentials on disk so re-enabling doesn't
            # need re-typing, but stop advertising LAN URLs / setting env vars.
            self._lan_enabled = False
            self._log("局域网访问已关闭。重启服务后生效。")
        self._refresh_status()

    def _configure_lan_auth(self) -> None:
        """User clicked «修改账号密码» — re-prompt and overwrite."""
        creds = self._prompt_lan_auth_dialog()
        if creds is None:
            return
        user, pwd = creds
        save_lan_auth(self.root_dir, user, pwd)
        self._lan_user = user
        self._lan_pass = pwd
        if not self._lan_enabled:
            # Auto-enable when user explicitly sets credentials.
            self._lan_enabled = True
            self.lan_enabled_var.set(True)
        self._log(f"局域网凭据已更新 (用户名: {self._lan_user})。重启服务后生效。")
        self._refresh_status()

    def _prompt_lan_auth_dialog(self) -> tuple[str, str] | None:
        """Modal dialog returning (user, pwd) or None if cancelled."""
        tk = self.tk
        ttk = self.ttk

        dlg = tk.Toplevel(self.win)
        dlg.title("局域网访问 — 设置账号密码")
        dlg.transient(self.win)
        dlg.grab_set()
        dlg.resizable(False, False)

        result: dict[str, tuple[str, str] | None] = {"creds": None}

        frm = ttk.Frame(dlg, padding=14)
        frm.grid()

        ttk.Label(
            frm,
            text=(
                "设置局域网访问的账号密码（HTTP basic auth）。\n"
                "其他电脑访问主服务 GUI 时会要求输入。\n"
                "凭据保存到 api key/lan_auth.env，升级时保留。\n"
                f"⚠️ 安全：开启后 {GUI_PORT} 会绑定 0.0.0.0，仅在可信网络使用。"
            ),
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(frm, text="用户名:").grid(row=1, column=0, sticky="w", pady=(0, 4))
        user_entry = ttk.Entry(frm, width=36)
        user_entry.insert(0, self._lan_user or "admin")
        user_entry.grid(row=1, column=1, sticky="we", pady=(0, 4))
        user_entry.focus_set()

        ttk.Label(frm, text="密码:").grid(row=2, column=0, sticky="w", pady=(0, 4))
        pwd_show_var = tk.BooleanVar(value=False)
        pwd_entry = ttk.Entry(frm, width=36, show="*")
        if self._lan_pass:
            pwd_entry.insert(0, self._lan_pass)
        pwd_entry.grid(row=2, column=1, sticky="we", pady=(0, 4))

        def _toggle_pwd_show() -> None:
            pwd_entry.configure(show="" if pwd_show_var.get() else "*")
        ttk.Checkbutton(frm, text="显示密码", variable=pwd_show_var,
                        command=_toggle_pwd_show).grid(row=3, column=1, sticky="w")

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def _save() -> None:
            u = user_entry.get().strip()
            p = pwd_entry.get()
            if not u or not p:
                self._log("LAN 配置：用户名 / 密码不能为空，已取消。")
                result["creds"] = None
            else:
                result["creds"] = (u, p)
            dlg.destroy()

        def _cancel() -> None:
            result["creds"] = None
            dlg.destroy()

        ttk.Button(btns, text="保存", command=_save).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="取消", command=_cancel).grid(row=0, column=1, padx=4)
        dlg.bind("<Return>", lambda _e: _save())
        dlg.bind("<Escape>", lambda _e: _cancel())

        dlg.update_idletasks()
        w = dlg.winfo_reqwidth(); h = dlg.winfo_reqheight()
        x = self.win.winfo_rootx() + (self.win.winfo_width() - w) // 2
        y = self.win.winfo_rooty() + 100
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        dlg.wait_window()
        return result["creds"]

    # ── Intranet update push: server side ─────────────────────────────

    def toggle_push_server(self) -> None:
        if self.push_proc is not None and self.push_proc.poll() is None:
            self.stop_push_server()
        else:
            self.start_push_server()

    def start_push_server(self) -> None:
        if self.push_proc is not None and self.push_proc.poll() is None:
            self._log("推送服务器已在运行。")
            return
        if port_open(GUI_HOST, PUSH_SERVER_PORT):
            self._log(f"端口 {PUSH_SERVER_PORT} 已被占用 — 可能有外部进程在跑推送服务器。")
            return
        cmd = self._build_push_server_cmd()
        self._log(f"启动推送服务器: {' '.join(cmd)}")
        try:
            self.push_proc = subprocess.Popen(
                cmd, cwd=str(self.root_dir), env=self._make_subprocess_env(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except Exception as exc:
            self._log(f"推送服务器启动失败: {type(exc).__name__}: {exc}")
            self.push_proc = None
            return
        self._push_reader_thread = threading.Thread(
            target=self._read_push_output, daemon=True,
        )
        self._push_reader_thread.start()
        self._refresh_status()

    def stop_push_server(self, *, quiet: bool = False) -> None:
        proc = self.push_proc
        if proc is None or proc.poll() is not None:
            if not quiet:
                self._log("推送服务器未在运行。")
            self.push_proc = None
            self._refresh_status()
            return
        if not quiet:
            self._log("正在关闭推送服务器…")
        try:
            proc.terminate()
            for _ in range(20):
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            else:
                proc.kill()
        except Exception as exc:
            self._log(f"推送服务器关闭异常: {exc}")
        self.push_proc = None
        if not quiet:
            self._log("推送服务器已关闭。")
        self._refresh_status()

    def _build_push_server_cmd(self) -> list[str]:
        if _frozen():
            return [sys.executable, "--push-server-mode"]
        return [sys.executable, str(Path(__file__).resolve()), "--push-server-mode"]

    def _read_push_output(self) -> None:
        proc = self.push_proc
        if proc is None or proc.stdout is None:
            return
        log_path = _log_dir_for(self.root_dir) / f"push-server-{_today_stamp()}.log"
        try:
            with open(log_path, "a", encoding="utf-8", buffering=1) as fh:
                fh.write(f"\n=== push server start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log_threadsafe(f"[push] {line}")
                        try:
                            fh.write(line + "\n")
                        except Exception:
                            pass
        except Exception as exc:
            self._log_threadsafe(f"推送服务器日志异常: {exc}")
        finally:
            rc = proc.poll()
            if rc is not None:
                self._log_threadsafe(f"推送服务器已退出 (rc={rc})")

    def _publish_new_version_dialog(self) -> None:
        """Pick a setup.exe + version, then run mast.update.publish."""
        from tkinter import filedialog, simpledialog
        path = filedialog.askopenfilename(
            title="选择 MAST setup.exe",
            initialdir=str(Path("dist").resolve() if Path("dist").exists() else Path.home()),
            filetypes=[("Installer", "*.exe"), ("All", "*.*")],
        )
        if not path:
            return
        version = simpledialog.askstring(
            "发布版本号",
            "输入版本号 (例 0.3.1，注意要和 setup.exe 文件名里一致):",
            parent=self.win,
        )
        if not version:
            return
        try:
            from mast.update.manifest import make_manifest, write_manifest
            from mast.update.server import push_dir
            import shutil
            setup_src = Path(path).resolve()
            target = push_dir(self.root_dir) / setup_src.name
            if setup_src != target.resolve():
                self._log(f"复制 {setup_src.name} → {target}")
                shutil.copy2(setup_src, target)
            self._log(f"计算 SHA256 of {target.name} ...")
            m = make_manifest(target, version=version.strip())
            write_manifest(push_dir(self.root_dir) / "manifest.json", m)
            self._log(f"已发布 v{m.version}: sha256={m.sha256[:16]}…  size={m.size_bytes//(1024*1024)} MB")
        except Exception as exc:
            self._log(f"发布失败: {type(exc).__name__}: {exc}")
        self._refresh_status()

    def _show_server_token_dialog(self) -> None:
        """Show / regenerate the server-side push token."""
        try:
            from mast.update.server import _read_token, write_token, generate_token
        except Exception as exc:
            self._log(f"无法加载更新模块: {exc}")
            return
        cur = _read_token(self.root_dir)
        if not cur:
            cur = generate_token()
            write_token(self.root_dir, cur)
            self._log("已自动生成新的推送 token。")

        tk = self.tk
        ttk = self.ttk
        dlg = tk.Toplevel(self.win)
        dlg.title("推送服务器 token")
        dlg.transient(self.win); dlg.grab_set(); dlg.resizable(False, False)
        frm = ttk.Frame(dlg, padding=14); frm.grid()
        ttk.Label(frm, text="共享 token (客户端需填同一个值):", justify="left").grid(
            row=0, column=0, sticky="w", pady=(0, 6))
        entry = ttk.Entry(frm, width=60)
        entry.insert(0, cur); entry.configure(state="readonly")
        entry.grid(row=1, column=0, sticky="we")

        def _copy():
            self.win.clipboard_clear(); self.win.clipboard_append(cur)
            self._log("token 已复制到剪贴板。")

        def _regen():
            new_tok = generate_token()
            write_token(self.root_dir, new_tok)
            entry.configure(state="normal"); entry.delete(0, "end"); entry.insert(0, new_tok)
            entry.configure(state="readonly")
            self._log("token 已重新生成 — 所有客户端需要更新它们的 token 才能继续接收推送。")

        btns = ttk.Frame(frm); btns.grid(row=2, column=0, sticky="we", pady=(8, 0))
        ttk.Button(btns, text="复制", command=_copy).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="重新生成", command=_regen).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="关闭", command=dlg.destroy).grid(row=0, column=2, padx=4)
        dlg.update_idletasks()
        w = dlg.winfo_reqwidth(); h = dlg.winfo_reqheight()
        x = self.win.winfo_rootx() + (self.win.winfo_width() - w) // 2
        y = self.win.winfo_rooty() + 100
        dlg.geometry(f"{w}x{h}+{x}+{y}")

    # ── Intranet update push: client side ─────────────────────────────

    def _configure_update_client_dialog(self) -> None:
        """Set the remote push server URL + token (writes update_*.env files)."""
        try:
            from mast.update.client import (
                read_server_url, write_server_url,
                _read_token as _read_client_token, write_token as _write_client_token,
            )
        except Exception as exc:
            self._log(f"无法加载更新模块: {exc}")
            return

        tk = self.tk
        ttk = self.ttk
        dlg = tk.Toplevel(self.win)
        dlg.title("配置远程推送服务器")
        dlg.transient(self.win); dlg.grab_set(); dlg.resizable(False, False)
        frm = ttk.Frame(dlg, padding=14); frm.grid()

        ttk.Label(frm, text=(
            "配置本机作为「客户端」，从内网推送服务器拉取 MAST 升级。\n"
            "服务器 URL 例: http://<推送服务器地址>:8765\n"
            "Token 必须和服务器侧 update_server_token.env 中的值一致。"
        ), justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(frm, text="服务器 URL:").grid(row=1, column=0, sticky="w", pady=4)
        url_entry = ttk.Entry(frm, width=44)
        url_entry.insert(0, read_server_url(self.root_dir))
        url_entry.grid(row=1, column=1, sticky="we", pady=4)

        ttk.Label(frm, text="Token:").grid(row=2, column=0, sticky="w", pady=4)
        tok_entry = ttk.Entry(frm, width=44)
        tok_entry.insert(0, _read_client_token(self.root_dir))
        tok_entry.grid(row=2, column=1, sticky="we", pady=4)

        btns = ttk.Frame(frm); btns.grid(row=3, column=0, columnspan=2, sticky="e", pady=(10, 0))

        def _save():
            url = url_entry.get().strip()
            tok = tok_entry.get().strip()
            if url:
                write_server_url(self.root_dir, url)
            if tok:
                _write_client_token(self.root_dir, tok)
            self._log(f"客户端配置已保存：URL={url or '(空)'} / token={'***' if tok else '(空)'}")
            self._log("下次主服务启动时（或点 «立即检查更新») 生效。")
            dlg.destroy()
            self._refresh_status()

        ttk.Button(btns, text="保存", command=_save).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="取消", command=dlg.destroy).grid(row=0, column=1, padx=4)
        dlg.update_idletasks()
        w = dlg.winfo_reqwidth(); h = dlg.winfo_reqheight()
        x = self.win.winfo_rootx() + (self.win.winfo_width() - w) // 2
        y = self.win.winfo_rooty() + 100
        dlg.geometry(f"{w}x{h}+{x}+{y}")

    def _check_update_now(self) -> None:
        """One-shot client check with a download progress dialog.

        The check + download run in a worker thread; a Tk modal shows an
        indeterminate bar while fetching the manifest and a determinate
        MB progress bar during the (162 MB full / ~48 MB delta) download, so the
        user always sees what's happening instead of a silent wait.
        """
        import tkinter as tk
        from tkinter import ttk

        dlg = tk.Toplevel(self.win)
        dlg.title("检查更新")
        dlg.resizable(False, False)
        dlg.transient(self.win)
        frm = ttk.Frame(dlg, padding=16)
        frm.grid()
        head = ttk.Label(frm, text="正在检查更新…", font=("Segoe UI", 11, "bold"))
        head.grid(row=0, column=0, sticky="w")
        pb = ttk.Progressbar(frm, length=340, mode="indeterminate")
        pb.grid(row=1, column=0, pady=(10, 4))
        pb.start(12)
        info = ttk.Label(frm, text="联系推送服务器…")
        info.grid(row=2, column=0, sticky="w")
        dlg.update_idletasks()
        w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        sw, sh = dlg.winfo_screenwidth(), dlg.winfo_screenheight()
        dlg.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        def _progress(done, total):
            def _u():
                try:
                    if total and total > 0:
                        if str(pb["mode"]) != "determinate":
                            pb.stop()
                            pb.config(mode="determinate", maximum=total)
                        pb["value"] = done
                        info.config(text=f"下载中… {done // (1024*1024)} / "
                                         f"{total // (1024*1024)} MB "
                                         f"({100*done//max(total,1)}%)")
                    else:
                        info.config(text=f"下载中… {done // (1024*1024)} MB")
                except Exception:
                    pass
            try:
                self.win.after(0, _u)
            except Exception:
                pass

        def _bg():
            status, detail = "error", ""
            remote_ver = remote_date = ""
            cur_ver = cur_date = ""
            try:
                from mast.update.client import (
                    _read_token, read_server_url, check_and_download,
                )
                from mast import __version__, __release_date__
                cur_ver, cur_date = __version__, __release_date__
                url = read_server_url(self.root_dir)
                tok = _read_token(self.root_dir)
                if not url or not tok:
                    status, detail = "no-config", "未配置 URL / token，先点 «配置远程服务器…»。"
                else:
                    # Lightweight manifest peek for the latest version + release
                    # date to DISPLAY (independent of whether a download happens —
                    # check_and_download only returns status strings, #ux).
                    try:
                        import httpx
                        from mast.update.manifest import Manifest
                        with httpx.Client(timeout=httpx.Timeout(20.0, connect=10.0)) as _c:
                            _r = _c.get(url.rstrip("/") + "/manifest.json",
                                        headers={"Authorization": f"Bearer {tok}"})
                            if _r.status_code == 200:
                                _m = Manifest.from_dict(_r.json())
                                remote_ver = _m.version or ""
                                remote_date = getattr(_m, "published_at", "") or ""
                    except Exception as _exc:
                        self._log_threadsafe(f"manifest 预览失败: {_exc}")
                    self._log_threadsafe(f"立即检查 {url} ...")
                    status, detail = check_and_download(
                        url, tok, self.root_dir, current_version=__version__,
                        progress_cb=_progress,
                    )
                    self._log_threadsafe(f"检查结果：{status}  {detail}")
            except Exception as exc:
                status, detail = "error", f"{type(exc).__name__}: {exc}"
                self._log_threadsafe(f"检查异常: {detail}")
            finally:
                def _done():
                    try:
                        pb.stop()
                    except Exception:
                        pass
                    nice = {
                        "downloaded": "新版本已下载,下次启动将提示安装。",
                        "downloaded-delta": "增量更新已下载,下次启动将应用。",
                        "no-update": "已是最新版本。",
                        "no-server": "无法连接推送服务器。",
                        "no-config": detail,
                        "error": f"出错：{detail}",
                    }.get(status, f"{status}: {detail}")
                    head.config(text=nice)
                    pb.grid_remove()
                    info.grid_remove()
                    # ── 4-field version table (操作员要求) ──
                    # On "no-update" the manifest peek may be empty / equal to
                    # current → fall back to current so 最新版本 is still honest.
                    _rv = remote_ver or cur_ver
                    _rd = remote_date or cur_date
                    tbl = ttk.Frame(frm)
                    tbl.grid(row=3, column=0, sticky="w", pady=(10, 0))
                    _rows = (
                        ("当前版本", f"v{cur_ver}"),
                        ("当前版本发布时间", _fmt_release_date(cur_date)),
                        ("最新版本", f"v{_rv}"),
                        ("最新版本发布时间", _fmt_release_date(_rd)),
                    )
                    for _i, (_k, _v) in enumerate(_rows):
                        ttk.Label(tbl, text=_k + "：", foreground="#666").grid(
                            row=_i, column=0, sticky="w", padx=(0, 8))
                        ttk.Label(tbl, text=_v, font=("Segoe UI", 9, "bold")).grid(
                            row=_i, column=1, sticky="w")
                    ttk.Button(frm, text="关闭", command=dlg.destroy).grid(
                        row=4, column=0, sticky="e", pady=(10, 0))
                    self._refresh_status()
                try:
                    self.win.after(0, _done)
                except Exception:
                    pass
        threading.Thread(target=_bg, daemon=True).start()

    def _reset_update_overrides(self) -> None:
        """Delete the per-machine URL/token files so the bundle defaults take effect."""
        for fname in ("update_server_url.env", "update_client_token.env"):
            p = self.root_dir / "api key" / fname
            if p.exists():
                try:
                    p.unlink()
                    self._log(f"已删除自定义: {p}")
                except OSError as exc:
                    self._log(f"删除失败 {p}: {exc}")
        self._log("自动更新配置已恢复为安装包默认值。")
        self._refresh_status()

    def _open_release_notes(self) -> None:
        """Open the CURRENT version's release notes — ALWAYS a real .txt.

        Resolution order:
          1. bundled ``docs/release_notes_current.txt`` (stable name, current
             version — emitted by the build pipeline).
          2. ``docs/release_notes_v*.txt`` matching __version__ (then newest).
          3. (transition) the legacy ``*release_notes*.md`` if no .txt exists.
          4. Fallback: GENERATE a real .txt at runtime under
             experiments/logs/ with a version header so the button never
             dead-ends — it always opens a real .txt with os.startfile().
        """
        import os
        import re
        try:
            from mast import __version__ as cur_ver, __release_date__ as cur_date
        except Exception:
            cur_ver, cur_date = "?", ""

        cand_dir = None
        try:
            d = _resource_path("docs")
            if d and d.is_dir():
                cand_dir = d
        except Exception:
            cand_dir = None

        def _vkey(p: Path):
            m = re.findall(r"v?(\d+(?:\.\d+)+)", p.name)
            return tuple(int(x) for x in m[-1].split(".")) if m else (0,)

        target: Path | None = None

        # 1. Stable per-version current file.
        if cand_dir is not None:
            cur_txt = cand_dir / "release_notes_current.txt"
            if cur_txt.exists():
                target = cur_txt

        # 2. release_notes_v*.txt matching __version__, else newest.
        if target is None and cand_dir is not None:
            txts = sorted(cand_dir.glob("release_notes_v*.txt"))
            if txts:
                mm = ".".join(cur_ver.split(".")[:2]) if cur_ver != "?" else ""
                for key in (cur_ver, mm):
                    if not key:
                        continue
                    hit = [p for p in txts if key in p.name]
                    if hit:
                        target = sorted(hit, key=_vkey)[-1]
                        break
                if target is None:
                    target = sorted(txts, key=_vkey)[-1]

        # 3. (transition) legacy markdown release notes if no .txt exists.
        if target is None and cand_dir is not None:
            mds: list[Path] = []
            seen: set = set()
            for pat in ("*release_notes*.md", "mast2_release_notes*.md"):
                for p in sorted(cand_dir.glob(pat)):
                    if p not in seen:
                        seen.add(p)
                        mds.append(p)
            if mds:
                mm = ".".join(cur_ver.split(".")[:2]) if cur_ver != "?" else ""
                for key in (cur_ver, mm):
                    if not key:
                        continue
                    hit = [p for p in mds if key in p.name]
                    if hit:
                        target = sorted(hit, key=_vkey)[-1]
                        break
                if target is None:
                    target = sorted(mds, key=_vkey)[-1]

        # 4. Nothing bundled — generate a real .txt at runtime so the button
        #    NEVER dead-ends.
        if target is None or not target.exists():
            try:
                out_dir = _log_dir_for(self.root_dir)
                target = out_dir / f"release_notes_v{cur_ver}.txt"
                target.write_text(
                    f"MAST v{cur_ver} 更新说明\n"
                    f"发布日期: {_fmt_release_date(cur_date)}\n"
                    + "=" * 40
                    + "\n\n(本地未打包详细更新说明，完整说明见 docs/。)\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                self._log(f"生成更新说明失败: {exc}")
                return

        try:
            os.startfile(str(target))  # noqa: S606 — Windows default opener
            self._log(f"打开更新说明: {target.name}")
        except Exception as exc:
            self._log(f"打开更新说明失败: {exc}")

    def _open_update_log(self) -> None:
        """Open the push server's access.log for inspection."""
        try:
            from mast.update.server import push_dir
            log_path = push_dir(self.root_dir) / "access.log"
            if not log_path.exists():
                self._log(f"日志文件不存在: {log_path}")
                return
            os.startfile(str(log_path))
        except Exception as exc:
            self._log(f"打开日志失败: {exc}")

    # ── API key config dialog ────────────────────────────────────────

    def _configure_provider(self, provider: str) -> None:
        existing = get_key(provider, self.root_dir)
        new_key = self._prompt_key_dialog(provider, existing)
        if new_key is None:
            return  # cancelled
        if not new_key.strip():
            self._log(f"{PROVIDER_LABELS[provider]}: 留空 — 未保存。")
            return
        try:
            path = save_key(provider, self.root_dir, new_key)
            self._log(f"{PROVIDER_LABELS[provider]} key 已保存到 {path}")
        except Exception as exc:
            self._log(f"保存失败: {exc}")
        self._refresh_status()

    def _prompt_key_dialog(self, provider: str, existing: str) -> str | None:
        tk = self.tk
        ttk = self.ttk
        label = PROVIDER_LABELS[provider]
        fname = PROVIDER_KEY_FILES[provider]
        env_vars = ", ".join(PROVIDER_ENV_VARS.get(provider, ()))

        dlg = tk.Toplevel(self.win)
        dlg.title(f"配置 {label}")
        dlg.transient(self.win)
        dlg.grab_set()
        dlg.resizable(False, False)

        result: dict[str, str | None] = {"key": None}

        frm = ttk.Frame(dlg, padding=14)
        frm.grid()

        ttk.Label(
            frm,
            text=(
                f"为 {label} 配置 API key。\n"
                f"保存到：api key/{fname}\n"
                f"环境变量优先级（如已设则覆盖文件）：{env_vars}"
            ),
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(frm, text="API key:").grid(row=1, column=0, sticky="w", pady=(0, 4))
        show_var = tk.BooleanVar(value=False)
        entry = ttk.Entry(frm, width=52, show="*")
        if existing:
            entry.insert(0, existing)
        entry.grid(row=1, column=1, sticky="we", pady=(0, 4))
        entry.focus_set()

        def _toggle_show() -> None:
            entry.configure(show="" if show_var.get() else "*")

        ttk.Checkbutton(frm, text="显示", variable=show_var, command=_toggle_show).grid(
            row=2, column=1, sticky="w"
        )

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def _save() -> None:
            result["key"] = entry.get()
            dlg.destroy()

        def _cancel() -> None:
            result["key"] = None
            dlg.destroy()

        ttk.Button(btns, text="保存", command=_save).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="取消", command=_cancel).grid(row=0, column=1, padx=4)
        dlg.bind("<Return>", lambda _e: _save())
        dlg.bind("<Escape>", lambda _e: _cancel())

        dlg.update_idletasks()
        w = dlg.winfo_reqwidth(); h = dlg.winfo_reqheight()
        x = self.win.winfo_rootx() + (self.win.winfo_width() - w) // 2
        y = self.win.winfo_rooty() + 80
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        dlg.wait_window()
        return result["key"]

    # ── Status polling ───────────────────────────────────────────────

    def _refresh_status(self) -> None:
        # ── Keys ────────────────────────────────────────────────────
        for prov in ("anthropic", "moonshot", "deepseek", "dashscope", "minimax", "zhipu"):
            ind = self.indicators[prov]
            k = get_key(prov, self.root_dir)
            if k:
                ind["dot"].configure(foreground="#22c55e")  # green
                masked = (k[:6] + "..." + k[-4:]) if len(k) > 12 else "set"
                ind["value"].configure(text=f"已配置 ({masked})", foreground="#22c55e")
            else:
                ind["dot"].configure(foreground="#dc2626")  # red
                ind["value"].configure(text="缺失", foreground="#dc2626")

        # ── Nanonis (simulator vs real) ─────────────────────────────
        # We can only tell "simulator vs real" by process name on the local
        # machine, so the answer is approximate when the controller lives on
        # a LAN host. Treat any TCP-open + no Mimea-process as "实机/外部".
        nan = self.indicators["nanonis"]
        if nanonis_connected():
            if _detect_nanonis_simulator():
                nan["dot"].configure(foreground="#eab308")  # amber
                nan["value"].configure(
                    text=f"已连接 — 模拟器 ({NANONIS_HOST}:{NANONIS_PORT})",
                    foreground="#eab308",
                )
            else:
                nan["dot"].configure(foreground="#22c55e")
                nan["value"].configure(
                    text=f"已连接 — 实机 ({NANONIS_HOST}:{NANONIS_PORT})",
                    foreground="#22c55e",
                )
        else:
            nan["dot"].configure(foreground="#888")
            nan["value"].configure(
                text=f"未连接 (TCP {NANONIS_PORT} 未开)", foreground="#888",
            )

        # ── URL strings (2 rows) ────────────────────────────────────
        # Scheme MUST reflect the service's real binding — https in LAN mode
        # (0.0.0.0 + TLS), else http. A stale http:// against the TLS listener is
        # the 白屏/ERR_EMPTY_RESPONSE bug.
        _scheme = self._gui_scheme()
        self.url_var.set(f"{_scheme}://{GUI_HOST}:{GUI_PORT}")
        if not self._lan_enabled:
            self.lan_url_var.set("(局域网访问未启用 — 勾上方复选框)")
        elif not self._lan_ip:
            self.lan_url_var.set("(未检测到局域网 IP)")
        else:
            self.lan_url_var.set(f"{_scheme}://{self._lan_ip}:{GUI_PORT}")

        # ── LAN info line ───────────────────────────────────────────
        if self._lan_enabled:
            if self._lan_user:
                self.lan_info_var.set(
                    f"已启用 — 用户名: {self._lan_user} | "
                    f"绑定 0.0.0.0 (主 {GUI_PORT}) | 重启服务后生效"
                )
            else:
                self.lan_info_var.set("已启用但缺少凭据 — 点 «修改账号密码» 设置")
        else:
            self.lan_info_var.set("未启用（仅本机可访问）")

        # ── Tailscale remote-access line + URL ──────────────────────
        # Renders the cached off-thread detection. The SAME 0.0.0.0 bind + TLS as
        # LAN mode already makes the console reachable over the tailnet; this only
        # tells the user which address to open on the OTHER computer.
        st = self._ts_status
        if st is None:
            self.ts_info_var.set("🌐 Tailscale: 检测中…")
            self.ts_url_var.set("")
        elif not getattr(st, "installed", False):
            self.ts_info_var.set(
                "🌐 Tailscale: 未安装 —— 在两台电脑都装 Tailscale 并登录同一账号 "
                "(tailscale.com/download)"
            )
            self.ts_url_var.set("")
        elif not getattr(st, "ready", False):
            self.ts_info_var.set(f"🌐 Tailscale: {st.hint or st.backend_state or '未就绪'}")
            self.ts_url_var.set("")
        elif not self._lan_enabled:
            dev = st.device_name or st.preferred_host
            self.ts_info_var.set(
                f"🌐 Tailscale: 就绪（本机 {dev}）—— 勾选上方复选框即可开启跨网远程访问"
            )
            self.ts_url_var.set("")
        else:
            try:
                from mast.net.tailscale import remote_urls
                urls = remote_urls(st, GUI_PORT, _scheme)
            except Exception:
                urls = []
            self.ts_url_var.set(urls[0]["url"] if urls else "")
            extra = f"，另有 {len(urls) - 1} 个备用地址" if len(urls) > 1 else ""
            self.ts_info_var.set(
                "🌐 Tailscale: 就绪 —— 在另一台电脑（登录同一账号）浏览器打开下方"
                f"「远程 - Tailscale」地址{extra}；首次访问提示证书不受信任点「继续」即可。"
            )

        # ── Main service status ─────────────────────────────────────
        proc = self.proc
        if proc is not None and proc.poll() is None:
            if gui_running():
                self.svc_status_var.set(f"● 主服务: 运行中 (端口 {GUI_PORT})")
                self.svc_status_lbl.configure(foreground="#22c55e")
            else:
                self.svc_status_var.set("● 主服务: 启动中… (子进程已起，等待端口)")
                self.svc_status_lbl.configure(foreground="#eab308")
        elif gui_running():
            self.svc_status_var.set("● 主服务: 外部进程占用端口 (非启动器管理)")
            self.svc_status_lbl.configure(foreground="#eab308")
        else:
            self.svc_status_var.set("● 主服务: 未运行")
            self.svc_status_lbl.configure(foreground="#888")

        # ── Main button enabled-state ───────────────────────────────
        # Enable based purely on the main GUI service state.
        owned_main = proc is not None and proc.poll() is None
        any_alive  = owned_main or gui_running()
        self.start_btn.configure(
            state="normal" if not gui_running() else "disabled"
        )
        self.restart_btn.configure(state="normal" if owned_main else "disabled")
        self.stop_btn.configure(state="normal" if any_alive else "disabled")

        # ── Auto-update (client side) status ────────────────────────
        try:
            from mast.update.client import read_server_url, _read_token
            from mast.update.defaults import get_default_server_url, get_default_token
            url = read_server_url(self.root_dir)
            tok = _read_token(self.root_dir)
            default_url = get_default_server_url()
            default_tok = get_default_token()
            # Are we using the bundled defaults or a per-machine override?
            url_source = (
                "默认" if url == default_url and url
                else "自定义" if url
                else "未配置"
            )
            tok_source = "默认" if tok == default_tok and tok else ("自定义" if tok else "无")
            if url:
                self.update_status_var.set(
                    f"● 自动更新: 订阅 {url}  ({url_source} URL / token {tok_source})"
                )
                self.update_status_lbl.configure(foreground="#22c55e")
            else:
                self.update_status_var.set("● 自动更新: 未配置 (服务器 URL 为空)")
                self.update_status_lbl.configure(foreground="#888")
        except Exception as exc:
            self.update_status_var.set(f"● 自动更新: 检测失败 — {exc}")
            self.update_status_lbl.configure(foreground="#dc2626")

    # ── Logging ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        with self.log_lock:
            ts = time.strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
            # sys.stdout was teed to launcher-*.log in main(); print routes
            # the same message to the persistent on-disk log so it survives
            # launcher restarts / crashes.
            try:
                print(f"[launcher] {line}")
            except Exception:
                pass

    def _log_threadsafe(self, msg: str) -> None:
        # Tk widget calls must happen on the main thread.
        self.win.after(0, self._log, msg)

    # ── Lifecycle ────────────────────────────────────────────────────

    def _minimize_to_tray(self) -> None:
        """Window-close (×) handler — NEVER exits (Nanonis-style). Hides to the
        system tray if available, else iconifies. The ONLY real exit is the
        physical «关闭启动器» button / tray «退出启动器», which gracefully stops the
        service + frees VRAM behind a progress bar (see _shutdown_launcher). This
        prevents an accidental × from orphaning the service and leaking its VRAM."""
        try:
            if self._tray_icon is not None:
                self.win.withdraw()
                self._log("启动器已最小化到系统托盘（右下角图标双击/右键可恢复）。")
            else:
                # No tray → iconify so the window is never lost.
                self.win.iconify()
                self._log("启动器已最小化（托盘不可用）。关闭请用「关闭启动器」按钮。")
        except Exception as exc:
            self._log(f"最小化失败: {exc}")

    def _make_proc_stopper(self, proc):
        """Graceful-then-force stopper for a launcher-OWNED Popen.

        Uses :func:`_graceful_stop_proc`: a cooperative ``taskkill`` (no ``/F``)
        first so the service closes its Nanonis TCP link cleanly, hard-killing
        ONLY if it refuses within the timeout (force-kill mid-TCP can corrupt the
        Nanonis port until Nanonis restarts)."""
        def _stop():
            try:
                _graceful_stop_proc(proc, graceful_timeout=15.0,
                                    log=self._log_threadsafe)
            except Exception:
                pass
        return _stop

    def _make_pid_stopper(self, pid):
        """Graceful stopper for an EXTERNAL service (started outside this launcher,
        e.g. C:\\MAST\\MAST.exe). We have no Popen handle, so taskkill by PID —
        WITHOUT /F. We deliberately do NOT force-kill: a force-kill mid-TCP can
        corrupt the Nanonis port until Nanonis restarts. If it won't stop
        gracefully we leave it (the caller reports that)."""
        def _stop():
            try:
                subprocess.run(["taskkill", "/PID", str(pid)],
                               capture_output=True, timeout=10,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                for _ in range(60):       # up to ~12 s for a graceful exit
                    if not _pid_alive(pid):
                        break
                    time.sleep(0.2)
            except Exception:
                pass
        return _stop

    def _shutdown_launcher(self) -> None:
        """The ONLY full-exit path (the «关闭主服务并退出启动器» button + tray «退出»).
        Confirms, gracefully stops the main service (frees VRAM) behind a progress
        bar, then exits. The window × only minimizes — so this is the real exit."""
        from tkinter import messagebox
        stoppers: list[tuple[str, object]] = []
        for label, attr in (("主服务", "proc"), ("推送服务器", "push_proc")):
            proc = getattr(self, attr, None)
            if proc and proc.poll() is None:
                stoppers.append((label, self._make_proc_stopper(proc)))
        external = False
        if not any(lbl == "主服务" for lbl, _ in stoppers) and gui_running():
            # Service is running but NOT owned by this launcher (external start).
            pid = _pid_on_port(GUI_PORT)
            if pid:
                external = True
                stoppers.append((f"外部主服务 (PID {pid})", self._make_pid_stopper(pid)))

        if not stoppers:
            if messagebox.askyesno("退出启动器", "未检测到运行中的服务。确定退出启动器？",
                                    parent=self.win):
                self._exit_now()
            return

        if external:
            msg = ("检测到主服务由外部启动（非本启动器托管）。将尝试优雅停止它并释放"
                   "显存，然后退出。\n注意：为避免损坏 Nanonis 端口，不会强制结束——"
                   "若它拒绝优雅退出会保留运行（届时请到启动它的窗口关闭）。\n\n继续？")
        else:
            msg = ("将优雅停止主服务并释放显存（约数秒），然后退出启动器。\n"
                   "若只想隐藏窗口、让服务继续运行，请点窗口右上角 ×（缩到状态栏）。\n\n确定？")
        if not messagebox.askyesno("关闭主服务并退出启动器", msg, parent=self.win):
            return
        self._run_shutdown_with_progress(stoppers)

    def _run_shutdown_with_progress(self, stoppers) -> None:
        """Modal progress dialog while each (label, stop_callable) runs on a worker
        thread; on completion the launcher exits. Tk updates marshalled via
        win.after (the worker never touches widgets directly)."""
        tk = self.tk
        ttk = self.ttk
        dlg = tk.Toplevel(self.win)
        dlg.title("正在关闭 MAST…")
        dlg.transient(self.win)
        dlg.resizable(False, False)
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)  # can't dismiss mid-shutdown
        try:
            if getattr(self, "_logo_img", None) is not None:
                dlg.iconphoto(False, self._logo_img)
        except Exception:
            pass
        frm = ttk.Frame(dlg, padding=18)
        frm.grid()
        status = tk.StringVar(value="正在优雅停止主服务、释放显存…")
        ttk.Label(frm, textvariable=status, font=("Segoe UI", 10),
                  wraplength=360, justify="left").grid(row=0, column=0, pady=(0, 10))
        bar = ttk.Progressbar(frm, mode="indeterminate", length=320)
        bar.grid(row=1, column=0)
        bar.start(12)
        dlg.update_idletasks()
        w = dlg.winfo_reqwidth(); h = dlg.winfo_reqheight()
        x = self.win.winfo_rootx() + (self.win.winfo_width() - w) // 2
        y = self.win.winfo_rooty() + 100
        dlg.geometry(f"{w}x{h}+{x}+{y}")

        def _worker() -> None:
            for label, stop in stoppers:
                try:
                    self.win.after(0, lambda l=label: status.set(f"正在停止：{l} …"))
                except Exception:
                    pass
                try:
                    stop()
                except Exception:
                    pass
            # External service may still be up (we never force-kill it).
            leftover = gui_running() and (self.proc is None or self.proc.poll() is not None)

            def _finish() -> None:
                try:
                    bar.stop()
                except Exception:
                    pass
                if leftover:
                    from tkinter import messagebox
                    messagebox.showinfo(
                        "外部服务仍在运行",
                        "外部主服务未能优雅停止（已避免强制结束以保护 Nanonis 端口）。\n"
                        "启动器将退出；如需释放显存，请到启动该服务的窗口手动关闭。",
                        parent=self.win)
                self._exit_now()

            try:
                self.win.after(0, _finish)
            except Exception:
                self._exit_now()

        threading.Thread(target=_worker, name="launcher-shutdown",
                         daemon=True).start()

    def _exit_now(self) -> None:
        """Final teardown — stop any still-alive child, clear lock/IPC/tray,
        destroy the window. Idempotent (safe to call from the mainloop-exit
        finally as a backstop).

        Normally reached AFTER _run_shutdown_with_progress already graceful-stopped
        the main service, so the main service is already gone here. But this is
        also the mainloop-exit backstop, so a service may still be alive: give the
        Nanonis-holding main service a SHORT graceful window (taskkill no /F)
        before any hard kill to narrow the port-corruption window. push_proc (push
        server) holds no Nanonis port → plain terminate is fine."""
        proc = getattr(self, "proc", None)
        if proc and proc.poll() is None:
            try:
                # Keep this snappy (exit path): short graceful attempt, then kill.
                _graceful_stop_proc(proc, graceful_timeout=5.0, force_timeout=2.0)
            except Exception:
                pass
        push = getattr(self, "push_proc", None)
        if push and push.poll() is None:
            try:
                push.terminate()
            except Exception:
                pass
        _clear_launcher_lock(self.root_dir)
        _sweep_lan_auth_handoff(self.root_dir)
        self._close_ipc()
        self._destroy_tray()
        try:
            self.win.destroy()
        except Exception:
            pass

    # ── IPC: second instance asks first to bring window to foreground ──

    def _poll_ipc(self) -> None:
        """Non-blocking accept-loop drained from Tk's after() ticker."""
        if self._ipc_sock is None:
            return
        try:
            while True:
                try:
                    conn, _ = self._ipc_sock.accept()
                except BlockingIOError:
                    break
                try:
                    data = conn.recv(64).decode("ascii", errors="ignore")
                finally:
                    conn.close()
                if data.startswith("BRING_FOREGROUND"):
                    self._bring_to_foreground()
        except Exception as exc:
            print(f"[launcher] IPC poll error: {exc}")
        finally:
            self.win.after(500, self._poll_ipc)

    def _bring_to_foreground(self) -> None:
        """Surface the launcher window — useful when 2nd instance asked us."""
        try:
            self.win.deiconify()
            self.win.lift()
            self.win.attributes("-topmost", True)
            self.win.after(200, lambda: self.win.attributes("-topmost", False))
            self.win.focus_force()
        except Exception:
            pass

    def _close_ipc(self) -> None:
        try:
            if self._ipc_sock is not None:
                self._ipc_sock.close()
                self._ipc_sock = None
        except Exception:
            pass

    # ── System tray (pystray + Pillow). Optional dependency. ──

    def _init_tray(self) -> None:
        """Set up the system-tray icon. No-op if pystray/PIL unavailable."""
        try:
            import pystray
            from PIL import Image
        except Exception as exc:
            print(f"[launcher] tray disabled (pystray/PIL missing): {exc}")
            return
        try:
            logo_path = _resource_path(LOGO_PNG_RELPATH)
            if not logo_path.exists():
                print(f"[launcher] tray disabled (logo not found at {logo_path})")
                return
            image = Image.open(str(logo_path))
        except Exception as exc:
            print(f"[launcher] tray icon load failed: {exc}")
            return

        def _show(_icon=None, _item=None):
            self.win.after(0, self._bring_to_foreground)

        def _quit(_icon=None, _item=None):
            # Graceful full shutdown (warning + progress + VRAM release).
            self.win.after(0, self._shutdown_launcher)

        menu = pystray.Menu(
            pystray.MenuItem("显示窗口", _show, default=True),
            pystray.MenuItem("退出启动器", _quit),
        )
        self._tray_icon = pystray.Icon("MAST2", image, "MAST 启动器", menu)
        # Run in background thread (Icon.run is blocking)
        self._tray_thread = threading.Thread(
            target=self._tray_icon.run, daemon=True,
        )
        self._tray_thread.start()

    def _destroy_tray(self) -> None:
        try:
            if self._tray_icon is not None:
                self._tray_icon.stop()
                self._tray_icon = None
        except Exception:
            pass

    def _on_autostart_toggle(self) -> None:
        enabled = self.autostart_var.get()
        ok = _startup_set(enabled)
        if ok:
            self._log(f"开机自启 {'已启用' if enabled else '已关闭'}")
        else:
            self._log(f"开机自启切换失败，请检查注册表写入权限")
            self.autostart_var.set(_startup_enabled())  # revert UI

    def _admin_pin_path(self) -> Path:
        return self.root_dir / "config" / "admin_pin.txt"

    def _set_admin_pin(self) -> None:
        """Set / clear the web 「高级管理」 tab unlock PIN. Stored LOCAL-ONLY as the
        sha256 hex in <root>/config/admin_pin.txt; the web tab reads + compares it
        (the raw PIN never leaves this machine / never enters the browser)."""
        from tkinter import simpledialog, messagebox
        import hashlib
        p = self._admin_pin_path()
        has = p.exists() and p.read_text(encoding="utf-8").strip()
        pin = simpledialog.askstring(
            "高级管理密码",
            ("设置网页「高级管理」标签的解锁密码"
             + ("（当前已设置）" if has else "（当前未设置）") + "。\n"
             "留空并确定 = 清除密码（届时该标签会提示未设置）。"),
            show="*", parent=self.win)
        if pin is None:
            return  # cancelled
        pin = pin.strip()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if not pin:
                p.unlink(missing_ok=True)
                self._log("高级管理密码已清除。")
                messagebox.showinfo("高级管理密码", "已清除密码。", parent=self.win)
                return
            confirm = simpledialog.askstring(
                "高级管理密码", "再次输入以确认：", show="*", parent=self.win)
            if confirm is None:
                return
            if confirm.strip() != pin:
                messagebox.showerror("高级管理密码", "两次输入不一致，未保存。", parent=self.win)
                return
            p.write_text(hashlib.sha256(pin.encode("utf-8")).hexdigest(), encoding="utf-8")
            self._log("高级管理密码已更新（sha256 写入 config/admin_pin.txt）。")
            messagebox.showinfo(
                "高级管理密码",
                "密码已保存。\n（网页「高级管理」标签将用它解锁；重启主服务后生效。）",
                parent=self.win)
        except Exception as exc:
            self._log(f"设置高级管理密码失败: {exc}")
            messagebox.showerror("高级管理密码", f"保存失败：{exc}", parent=self.win)

    def run(self) -> int:
        try:
            self.win.mainloop()
        finally:
            # Backstop: if the mainloop ever exits without going through
            # _shutdown_launcher (e.g. an unexpected destroy), still tear down.
            self._exit_now()
        return 0


# ── Headless vision self-test (frozen-bundle verification) ───────────

def _run_selftest_vision() -> int:
    """Headless M12 vision self-test. Verifies the FROZEN bundle can load
    torch + timm + peft + the offline DINOv3 backbone and run real inference.

    Writes a one-line PASS/FAIL summary to ``<project_root>/vision_selftest_result.txt``
    (console is suppressed under --noconsole) and returns 0 on PASS, 1 on FAIL.
    Run via ``MAST2.exe --selftest-vision``.
    """
    import traceback

    # 与 run_service / _run_install_skillpack 同一句 —— 三条一次性入口对
    # 「用户数据在哪」必须给同一个答案。少了它，冻结包里 project_root()
    # 退回 exe 所在目录，结果文件会落到**安装目录**而不是数据根
    # （真机实测：日志进了 <data-root>\experiments\logs，
    #  而 pyruntime_selftest_result.txt 落在 C:\MAST\）。
    os.environ.setdefault("MAST2_PROJECT_ROOT", str(_user_root()))

    lines: list[str] = ["=== MAST M12 vision self-test ==="]
    rc = 1
    try:
        import numpy as np

        from mast.vision.module import VisionModule
        from mast.vision.vigil_backend import VIGILBackend

        VisionModule._instance = None
        vm = VisionModule(backend="vigil")  # explicit → honored even pre-flight
        be = vm._backend
        lines.append(f"backend: {type(be).__name__}")
        if not isinstance(be, VIGILBackend):
            lines.append(
                "FAIL: backend is not VIGILBackend — torch/timm/peft or the M12 "
                "ckpt / DINOv3 backbone cache is missing from the frozen bundle."
            )
        else:
            vm.set_scan_size_nm(10.0)
            H = W = 256
            terr = (np.mgrid[0:H, 0:W][1] // 85).astype(np.float32) * 30.0
            img = terr + np.random.RandomState(7).randn(H, W).astype(np.float32) * 2.0
            c = vm.assess_tip_coarse(img)
            s = vm.segment(img)
            lines.append(
                f"coarse: label={c.label} conf={c.confidence:.3f} "
                f"R_tip_nm={c.tip_radius_nm} sharp_log10={c.sharpness_log10}"
            )
            lines.append(
                f"segment: level={s.level} classes={s.classes} "
                f"px={sum(s.class_counts.values())}"
            )
            ok = (
                be.is_loaded()
                and c.label in ("good", "bad")
                and s.level == 1
                and len(s.classes) == 4
            )
            # Also verify the scan-progress auto-trigger pipeline is bundled +
            # functional in the frozen runtime: the deterministic translator
            # and the monitor module must import and produce a Chinese narration.
            try:
                from mast.agents.buffer_summarizer.node import describe
                from mast.vision.scan_monitor import ScanVisionMonitor  # noqa: F401
                narration = describe("tip_coarse", c.model_dump())
                lines.append(f"scan-vision: describe -> {narration}")
                ok = ok and isinstance(narration, str) and len(narration) > 0
            except Exception as exc:  # noqa: BLE001
                lines.append(f"scan-vision pipeline FAIL: {type(exc).__name__}: {exc}")
                ok = False
            lines.append("PASS: M12 + scan-vision pipeline OK in the frozen runtime" if ok
                         else "FAIL: backend loaded but outputs failed sanity checks")
            rc = 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FAIL: {type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
        rc = 1

    summary = "\n".join(lines)
    for ln in lines:
        print(ln)
    try:
        from mast._runtime_paths import project_root
        out = project_root() / "vision_selftest_result.txt"
        out.write_text(summary + f"\n\nexit={rc}\n", encoding="utf-8")
        print(f"[selftest] wrote {out}")
    except Exception:
        pass
    return rc


def _run_selftest_pyruntime() -> int:
    """Headless self-test for the DP analysis runtime (bundled CPython 3.13).

    Verifies the SHIPPED ``MASTv2/pyruntime`` can import numpy/scipy/matplotlib,
    render a figure through Agg and write it to disk — and that it CANNOT import
    ``nanonis_spm`` or ``mast``.

    That last check is not a nice-to-have. "The data-processing agent can't touch
    the instrument" is a claim whose ONLY mechanism is that those packages are
    absent from this interpreter — DP has no SafetyGate and no HITL. So a leak
    here is a FAIL, not a warning.

    Writes ``<project_root>/pyruntime_selftest_result.txt`` and returns 0/1.
    Run via ``MAST.exe --selftest-pyruntime``.
    """
    import traceback

    # 与 run_service / _run_install_skillpack 同一句 —— 三条一次性入口对
    # 「用户数据在哪」必须给同一个答案。少了它，冻结包里 project_root()
    # 退回 exe 所在目录，结果文件会落到**安装目录**而不是数据根
    # （真机实测：日志进了 <data-root>\experiments\logs，
    #  而 pyruntime_selftest_result.txt 落在 C:\MAST\）。
    os.environ.setdefault("MAST2_PROJECT_ROOT", str(_user_root()))

    lines: list[str] = ["=== MAST pyruntime self-test ==="]
    rc = 1
    try:
        from mast.pyexec import runtime as R

        rep = R.probe_runtime(refresh=True)
        rt = rep.runtime
        if rt is None:
            lines.append("FAIL: no usable analysis runtime found")
            lines.append(rep.why_not())
        else:
            lines.append(f"runtime: {rt.kind} @ {rt.exe}")
            res = R.selftest(rt, refresh=True)
            lines.append(f"python : {res.python}")
            for k, v in sorted(res.packages.items()):
                lines.append(f"  {k} {v}")
            if res.missing:
                lines.append("optional missing: " + ", ".join(res.missing))
            lines.append(f"figure : {res.figure}")
            lines.append(f"isolated: {res.isolated}")
            if res.leaks:
                # B1 破了 —— 这条比缺一个可选库严重得多，不能只 warn。
                lines.append(
                    "FAIL: B1 BROKEN — the analysis child can import "
                    + ", ".join(res.leaks)
                    + ". That is the whole mechanism behind 'DP cannot reach the "
                      "instrument'; DP has no SafetyGate and no HITL.")
            elif not res.ok:
                lines.append("FAIL: " + "; ".join(res.problems))
                if res.stderr:
                    lines.append("--- child stderr ---")
                    lines.append(res.stderr)
            elif rt.kind != "bundled":
                # 发布冒烟要跑在真产物上：落到 dev/system 说明随包运行时没进去。
                lines.append(
                    f"FAIL: resolved to '{rt.kind}', not the bundled runtime — "
                    "MASTv2/pyruntime is missing from this build (build.ps1 "
                    "Step 4.7). It would still work here, but the operator's "
                    "machine has no dev venv to fall back to.")
            else:
                lines.append("PASS: bundled analysis runtime OK, no instrument leak")
                rc = 0
    except Exception as exc:  # noqa: BLE001
        lines.append(f"FAIL: {type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
        rc = 1

    summary = "\n".join(lines)
    for ln in lines:
        print(ln)
    try:
        from mast._runtime_paths import project_root
        out = project_root() / "pyruntime_selftest_result.txt"
        out.write_text(summary + f"\n\nexit={rc}\n", encoding="utf-8")
        print(f"[selftest] wrote {out}")
    except Exception:
        pass
    return rc


def _run_install_skillpack(pack_path: str) -> int:
    """Install a signed skill pack from a local .zip and report as JSON.

    This is the entry point an out-of-band push script drives over SSH.
    It exists so that script does NOT have to poke at ``_internal`` to import
    mast — that would put ``nanonis_spm`` on the child's path, which is
    exactly the thing B1 rests on NOT happening.

    Prints one ``INSTALL{json}`` line so the caller can parse a result rather
    than scrape prose. Returns 0 only if the pack verified AND installed.

    ⚠️ **必须先把 ``MAST2_PROJECT_ROOT`` 指到 user root**，和 :func:`run_service`
    一样。少了这一句，``mast._runtime_paths.project_root()`` 在冻结包里退回
    **exe 所在目录**，于是覆盖层解析成 ``<安装目录>\\config\\skill_overlay``，
    而**跑着的服务**用的是 ``<user root>\\config\\skill_overlay``（launcher 通过
    ``data_dir.txt`` 定出来的那个）。

    2026-08-21 真机实测：装包回报
    ``INSTALL{"ok": true, "installed": ["_packs/verify-p12/builtins/history_query.py"]}``，
    而 ``<data-root>\\config\\skill_overlay`` 下**一个 ``_packs`` 都没有** ——
    东西全落在 ``C:\\MAST\\config\\skill_overlay``。服务永远看不见它，
    ``/api/skill-overlay/packs`` 报空。**一次报了成功、却什么也没送到的安装**，
    正是这条通道最不该有的形态。
    """
    import json
    import traceback

    # 与 run_service 同一句 —— 两条入口对同一个「用户数据在哪」必须给同一个答案。
    _root = _user_root()
    os.environ.setdefault("MAST2_PROJECT_ROOT", str(_root))

    rc = 1
    try:
        from mast.update import skillpack_client as SC

        res = SC.install_pack_file(pack_path)
        print("INSTALL" + json.dumps(res.as_dict(), ensure_ascii=False))
        print(res.describe())
        rc = 0 if res.ok else 1
    except Exception as exc:  # noqa: BLE001
        print("INSTALL" + json.dumps(
            {"ok": False, "reasons": [f"{type(exc).__name__}: {exc}"]},
            ensure_ascii=False))
        print(traceback.format_exc())
        rc = 1
    return rc


# ── Top-level entry ──────────────────────────────────────────────────

def _main_inner() -> int:
    """Real main, wrapped by main() in a try/except for crash logging."""
    parser = argparse.ArgumentParser(
        prog="MAST",
        description="MAST desktop launcher / service",
        add_help=True,
    )
    parser.add_argument(
        "--service-mode",
        action="store_true",
        help="skip the launcher window and boot the main Gradio service directly",
    )
    parser.add_argument(
        "--push-server-mode",
        action="store_true",
        help="boot the intranet push-update server (mast.update.server) on port 8765",
    )
    parser.add_argument(
        "--publish", metavar="SETUP_EXE",
        help="(admin) publish a new version: copy the given setup.exe + write manifest "
             "into <data>/push_distribution/. Requires --version.",
    )
    parser.add_argument(
        "--version", dest="publish_version",
        help="version string for --publish (e.g. 0.3.2)",
    )
    parser.add_argument(
        "--selftest-vision",
        action="store_true",
        help="headless M12 vision self-test (verify the frozen bundle loads "
             "torch/timm/peft + the DINOv3 backbone and infers); writes "
             "vision_selftest_result.txt and exits",
    )
    parser.add_argument(
        "--install-skillpack", metavar="ZIP", default="",
        help="install a signed skill pack from a local .zip (verifies the "
             "Ed25519 signature and every file hash) and exit; prints an "
             "INSTALL{json} line. Used by push_skillpack_to_rig.py",
    )
    parser.add_argument(
        "--selftest-pyruntime",
        action="store_true",
        help="headless self-test of the bundled DP analysis runtime (numpy/"
             "scipy/matplotlib import + Agg figure + NO nanonis_spm leak); "
             "writes pyruntime_selftest_result.txt and exits",
    )
    parser.add_argument(
        "--minimized",
        action="store_true",
        help="start the launcher minimized to the system tray (used by the "
             "open-on-boot auto-start entry so it doesn't pop a window each login)",
    )
    args, _unknown = parser.parse_known_args()

    if args.selftest_vision:
        return _run_selftest_vision()
    if args.selftest_pyruntime:
        return _run_selftest_pyruntime()
    if args.install_skillpack:
        return _run_install_skillpack(args.install_skillpack)
    if args.service_mode:
        return run_service()
    if args.push_server_mode:
        return run_push_server()
    if args.publish:
        return _run_publish(args.publish, args.publish_version)

    # Launcher mode (default). Before opening Tk, check for a pending update.
    # If one is sitting in <data>/pending_update/ + verifies, the user MUST
    # accept it before they can use MAST — that's the "force update" UX.
    user_root = _user_root()

    # ── Single-instance check: if another launcher is already running, ping
    # it to bring its window to the foreground and exit cleanly.
    existing = _read_launcher_lock(user_root)
    if existing is not None:
        existing_pid, existing_port = existing
        print(f"[launcher.main] another MAST launcher is running (PID {existing_pid}); "
              f"asking it to come to foreground")
        if _send_bring_foreground(existing_port):
            print("[launcher.main] foreground request sent — exiting this instance.")
            return 0
        else:
            # IPC failed but lock is fresh — best to surface a message and exit
            # rather than start a competing launcher
            print(f"[launcher.main] could not reach existing launcher on port {existing_port}; "
                  f"exiting to avoid duplicate. Run with --force-launch to override.")
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"MAST 启动器 (PID {existing_pid}) 已在运行，但无法被唤起。\n"
                    f"请在任务管理器关闭旧进程，或重启 MAST 启动器。",
                    "MAST 启动器",
                    0x30,  # MB_ICONWARNING
                )
            except Exception:
                pass
            return 0

    # OTA apply order (each step no-ops if nothing pending):
    #   1. Hot-apply a data-only / SPA / docs delta live (safe while MAST runs).
    #   2. Else if a CODE delta is pending, spawn the offline self-updater (waits
    #      for MAST to exit → swaps locked files → relaunches) and exit now.
    #   3. Else run a pending full installer.
    _apply_pending_delta_if_safe(user_root)
    if _apply_pending_delta_offline_if_needed(user_root):
        return 0
    if _check_and_run_pending_update(user_root):
        return 0

    # Launcher mode (default). DPI awareness MUST be set before Tk init,
    # otherwise Windows will bitmap-stretch all our widgets at 200% scale.
    _enable_dpi_awareness()

    try:
        import tkinter  # noqa: F401
    except Exception:
        print("tkinter unavailable — falling back to direct service mode.")
        return run_service()

    app = LauncherApp(user_root, start_minimized=bool(getattr(args, "minimized", False)))
    return app.run()


def main() -> int:
    """Top-level entry. Sets up file logging FIRST so any later crash is
    captured to experiments/logs/launcher-YYYYMMDD.log, then dispatches.

    Critical for diagnosing bundled-app startup failures: with PyInstaller
    --noconsole, sys.stdout/stderr go to NUL by default, so we have no
    other way to learn what crashed.
    """
    user_root = _user_root()
    try:
        _ensure_user_dirs(user_root)
        log_path = _setup_persistent_logging(user_root)
        print(f"[launcher.main] logging to: {log_path}")
    except Exception as exc:
        # If logging setup itself crashes, there is nothing useful we can do
        # except let the original error propagate.
        print(f"[launcher.main] logging setup failed: {exc}")
        log_path = None

    try:
        return _main_inner()
    except SystemExit:
        raise
    except BaseException:
        import traceback, datetime
        print(
            f"[launcher.main] FATAL EXCEPTION at "
            f"{datetime.datetime.now().isoformat()}:"
        )
        traceback.print_exc()
        # In bundled mode the user may not see any window at all, so try to
        # surface the log path via a Windows MessageBox before dying.
        if _frozen() and log_path is not None:
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"MAST 启动失败。详细日志已写入：\n\n{log_path}\n\n"
                    "请把这个文件发给开发者诊断。",
                    "MAST 启动失败",
                    0x10,  # MB_ICONERROR
                )
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
