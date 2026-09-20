"""Tailscale detection + remote-access URL helpers — 傻瓜式跨网远程控制.

Why this exists
---------------
MAST's *LAN access* mode already does the hard part: when enabled it binds the
service to ``0.0.0.0`` with HTTP basic-auth and an auto-generated self-signed TLS
cert. A peer on the **same tailnet** can therefore already reach the console over
the Tailscale interface (the 100.64.0.0/10 CGNAT range) with **no port
forwarding** — Tailscale (WireGuard) tunnels it across networks and NATs.

What was missing to make cross-network control *foolproof* is purely
informational + cosmetic, and lives here:

* **Which address** to open on the *other* computer — the Tailscale IP
  (``100.x.y.z``) or the MagicDNS name, **not** the ``192.168.*`` LAN IP the
  launcher used to show. Give it to us and we build the exact URL.
* **Is Tailscale even ready** — installed, the daemon up, logged in — with a
  concrete Chinese next-step when it is not.
* **Cert coverage** — the Tailscale host(s) to fold into the self-signed cert's
  SAN so the remote browser isn't warned about a hostname mismatch.

The single source of truth is the ``tailscale`` CLI's ``status --json`` (a
stable, documented interface that talks to the local daemon and needs no admin).
Everything degrades gracefully: not installed, daemon stopped, logged out, a
timeout, or malformed JSON all resolve to a :class:`TailscaleStatus` the UI can
render as a clear next step rather than raising.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


# ── executable discovery ────────────────────────────────────────────────────

def find_cli() -> Optional[str]:
    """Absolute path to the ``tailscale`` executable, or ``None`` if not found.

    Checks ``PATH`` first (Linux, Homebrew, or a Windows install that added
    itself), then the platform's default install location so a stock Windows /
    macOS GUI install (which does *not* touch ``PATH``) is still detected.
    """
    exe = shutil.which("tailscale")
    if exe:
        return exe

    candidates: list[str] = []
    if sys.platform == "win32":
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("ProgramW6432", r"C:\Program Files"),
        ):
            if base:
                candidates.append(os.path.join(base, "Tailscale", "tailscale.exe"))
    elif sys.platform == "darwin":
        candidates += [
            "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
            "/usr/local/bin/tailscale",
            "/opt/homebrew/bin/tailscale",
        ]
    else:
        candidates += ["/usr/bin/tailscale", "/usr/local/bin/tailscale"]

    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _looks_like_tailscale_ip(ip: str) -> bool:
    """True for a Tailscale-assigned address: IPv4 100.64.0.0/10 or the
    Tailscale IPv6 ULA (``fd7a:115c:a1e0::/48``)."""
    ip = (ip or "").strip()
    if not ip:
        return False
    if ip.lower().startswith("fd7a:115c:a1e0"):
        return True
    parts = ip.split(".")
    if len(parts) == 4 and parts[0] == "100":
        try:
            return 64 <= int(parts[1]) <= 127 and 0 <= int(parts[3]) <= 255
        except ValueError:
            return False
    return False


# ── status ──────────────────────────────────────────────────────────────────

@dataclass
class TailscaleStatus:
    """A snapshot of the local Tailscale node, from ``tailscale status --json``."""

    installed: bool = False
    running: bool = False                     # BackendState == "Running"
    backend_state: str = ""                   # Running / Stopped / NeedsLogin / …
    self_ips: list[str] = field(default_factory=list)   # Tailscale IPs (v4 first)
    magic_dns: str = ""                       # FQDN, no trailing dot (MagicDNS on)
    tailnet: str = ""                         # tailnet name (…​.ts.net)
    device_name: str = ""                     # this node's short hostname
    hint: str = ""                            # Chinese next-step when not ready

    @property
    def ready(self) -> bool:
        """Reachable *right now*: installed, daemon up, and holding an IP."""
        return self.installed and self.running and bool(self.self_ips)

    @property
    def preferred_host(self) -> str:
        """The nicest host to dial — MagicDNS name (also cert-stable) else IP."""
        return self.magic_dns or (self.self_ips[0] if self.self_ips else "")


def _hint_for_state(state: str, *, ready: bool = False) -> str:
    if ready:
        return ""
    s = (state or "").strip()
    if s == "Running":  # up but no IP yet
        return "Tailscale 已连接但尚未分配 IP —— 请稍候，或在 Tailscale 中重新登录。"
    if s in ("Stopped", ""):
        return "Tailscale 已安装但未连接 —— 打开 Tailscale 并点「Connect」，保持登录。"
    if s == "NeedsLogin":
        return "Tailscale 需要登录 —— 打开 Tailscale，用与另一台电脑**相同的账号**登录。"
    if s in ("NoState", "Starting"):
        return "Tailscale 正在启动，请稍候…"
    return f"Tailscale 当前状态：{s}"


def _parse_status(data: dict) -> TailscaleStatus:
    backend = str(data.get("BackendState", "") or "")
    self_ = data.get("Self") or {}

    ips_raw = self_.get("TailscaleIPs") or data.get("TailscaleIPs") or []
    v4 = [ip for ip in ips_raw if _looks_like_tailscale_ip(ip) and ":" not in ip]
    v6 = [ip for ip in ips_raw if _looks_like_tailscale_ip(ip) and ":" in ip]
    ips = v4 + v6

    # DNSName is a FQDN only when MagicDNS is enabled; a dot-less value → absent.
    dns = str(self_.get("DNSName", "") or "").rstrip(".")
    magic = dns if "." in dns else ""

    suffix = str(data.get("MagicDNSSuffix", "") or "").rstrip(".")
    ct = data.get("CurrentTailnet") or {}
    tailnet = str(ct.get("Name", "") or suffix or "")
    device = str(self_.get("HostName", "") or "")

    st = TailscaleStatus(
        installed=True,
        running=(backend == "Running"),
        backend_state=backend,
        self_ips=ips,
        magic_dns=magic,
        tailnet=tailnet,
        device_name=device,
    )
    st.hint = _hint_for_state(backend, ready=st.ready)
    return st


def _run_kwargs(timeout: float) -> dict:
    kw: dict = {
        "capture_output": True,
        "timeout": timeout,
        "encoding": "utf-8",   # tailscale emits UTF-8 JSON; don't let a cp936
        "errors": "replace",   # locale mangle a Chinese device name.
    }
    if sys.platform == "win32":
        # Avoid a console window flash when the launcher/service is windowless.
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kw


def get_status(timeout: float = 3.0, cli: Optional[str] = None) -> TailscaleStatus:
    """Query the local Tailscale daemon. Never raises — see module docstring.

    Blocks up to *timeout* seconds shelling out to the CLI, so callers on an
    event loop (FastAPI) should run it in a threadpool, and UI callers off the
    main thread.
    """
    exe = cli or find_cli()
    if not exe:
        return TailscaleStatus(
            installed=False,
            hint="未检测到 Tailscale —— 请在两台电脑都安装 Tailscale 并登录同一账号："
                 "https://tailscale.com/download",
        )
    try:
        proc = subprocess.run([exe, "status", "--json"], **_run_kwargs(timeout))
    except subprocess.TimeoutExpired:
        return TailscaleStatus(installed=True, backend_state="",
                               hint="Tailscale 响应超时 —— 请确认 Tailscale 正在运行。")
    except OSError as exc:
        return TailscaleStatus(installed=True, hint=f"无法调用 Tailscale：{exc}")

    raw = (proc.stdout or "").strip()
    if not raw:
        # A logged-out / stopped daemon sometimes prints only to stderr.
        return TailscaleStatus(installed=True, backend_state="Stopped",
                               hint=_hint_for_state("Stopped"))
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return TailscaleStatus(installed=True,
                               hint="无法解析 Tailscale 状态输出 —— 请更新 Tailscale 后重试。")
    return _parse_status(data)


# ── derived views (URLs + cert hosts) ───────────────────────────────────────

def remote_urls(status: TailscaleStatus, port: int, scheme: str = "https") -> list[dict]:
    """The URL(s) to open on *another* computer, best first (MagicDNS, then IP).

    Each entry is ``{"label", "host", "url"}``. IPv6 hosts are bracketed for the
    URL. Empty when the node isn't ready.
    """
    out: list[dict] = []
    seen: set[str] = set()

    def add(label: str, host: str) -> None:
        if not host or host in seen:
            return
        seen.add(host)
        h = f"[{host}]" if (":" in host and not host.startswith("[")) else host
        out.append({"label": label, "host": host, "url": f"{scheme}://{h}:{port}"})

    # IP FIRST — the recommended address. MAST serves over a SELF-SIGNED cert, and
    # `*.ts.net` is on the browser HSTS-preload list, so a browser HARD-REJECTS a
    # self-signed cert on the MagicDNS name (no "proceed anyway" — # 2026-07-25 "tailscale 连不上"). The Tailscale IP is NOT preloaded, so the
    # browser lets you click through the warning. The MagicDNS name only works in a
    # browser once a *real* cert is provisioned (`tailscale cert` / `tailscale
    # serve`); we still list it, clearly labelled, for that case + for CLI/curl.
    for ip in status.self_ips:
        add("Tailscale IP（推荐）" if ":" not in ip else "Tailscale IPv6", ip)
    if status.magic_dns:
        add("MagicDNS 名称（浏览器需 tailscale cert 真证书）", status.magic_dns)
    return out


def extra_cert_hosts(status: TailscaleStatus) -> list[str]:
    """Tailscale host(s) to fold into the LAN cert's SAN (MagicDNS + IPs), so the
    remote browser gets a name/IP match instead of a hostname-mismatch warning."""
    hosts: list[str] = []
    if status.magic_dns:
        hosts.append(status.magic_dns)
    hosts.extend(status.self_ips)
    return hosts


def summary(port: int, scheme: str = "https", *,
            status: Optional[TailscaleStatus] = None) -> dict:
    """One bundle for the ``/api/remote-access`` endpoint (status + URLs)."""
    st = status if status is not None else get_status()
    return {
        "tailscale": {
            "installed": st.installed,
            "running": st.running,
            "ready": st.ready,
            "backend_state": st.backend_state,
            "device_name": st.device_name,
            "magic_dns": st.magic_dns,
            "tailnet": st.tailnet,
            "ips": list(st.self_ips),
            "hint": st.hint,
        },
        "urls": remote_urls(st, port, scheme),
    }


__all__ = [
    "TailscaleStatus",
    "find_cli",
    "get_status",
    "remote_urls",
    "extra_cert_hosts",
    "summary",
]
