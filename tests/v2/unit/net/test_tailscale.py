"""Tailscale detection + remote-access URL helpers — mast.net.tailscale.

Fully offline: the ``tailscale`` CLI is mocked, so every state (not installed /
stopped / needs-login / running / timeout / malformed) is exercised without a
real daemon. See docs/v2/ (远程访问 / 跨网 Tailscale 控制).
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.net import tailscale as ts  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────

RUNNING_JSON = json.dumps({
    "BackendState": "Running",
    "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1234"],
    "Self": {
        "HostName": "mast-box",
        "DNSName": "mast-box.tail1a2b3c.ts.net.",
        "TailscaleIPs": ["100.101.102.103", "fd7a:115c:a1e0::1234"],
        "Online": True,
    },
    "MagicDNSSuffix": "tail1a2b3c.ts.net",
    "CurrentTailnet": {"Name": "tail1a2b3c.ts.net", "MagicDNSEnabled": True},
})


def _fake_run(stdout="", stderr="", raise_exc=None):
    def run(cmd, **kw):
        if raise_exc is not None:
            raise raise_exc
        return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=0)
    return run


# ── _looks_like_tailscale_ip ─────────────────────────────────────────────────

@pytest.mark.parametrize("ip,expected", [
    ("100.64.0.1", True),
    ("100.101.102.103", True),
    ("100.127.255.254", True),
    ("100.63.0.1", False),      # below the /10 range
    ("100.128.0.1", False),     # above the /10 range
    ("192.168.1.5", False),     # ordinary LAN
    ("10.0.0.4", False),
    ("fd7a:115c:a1e0::1", True),
    ("fe80::1", False),
    ("", False),
    ("not-an-ip", False),
])
def test_looks_like_tailscale_ip(ip, expected):
    assert ts._looks_like_tailscale_ip(ip) is expected


# ── find_cli ─────────────────────────────────────────────────────────────────

def test_find_cli_prefers_path(monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda _n: "/usr/bin/tailscale")
    assert ts.find_cli() == "/usr/bin/tailscale"


def test_find_cli_falls_back_to_install_dir(monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda _n: None)
    hit = "/opt/homebrew/bin/tailscale" if sys.platform != "win32" else \
          str(Path(ts.os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tailscale" / "tailscale.exe")
    monkeypatch.setattr(ts.os.path, "isfile", lambda p: p == hit)
    assert ts.find_cli() == hit


def test_find_cli_none_when_absent(monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda _n: None)
    monkeypatch.setattr(ts.os.path, "isfile", lambda _p: False)
    assert ts.find_cli() is None


# ── get_status ───────────────────────────────────────────────────────────────

def test_not_installed(monkeypatch):
    monkeypatch.setattr(ts, "find_cli", lambda: None)
    st = ts.get_status()
    assert st.installed is False and st.ready is False
    assert "安装" in st.hint and "tailscale.com" in st.hint


def test_running_ready(monkeypatch):
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(stdout=RUNNING_JSON))
    st = ts.get_status(cli="tailscale")
    assert st.installed and st.running and st.ready
    assert st.self_ips[0] == "100.101.102.103"          # v4 first
    assert st.self_ips[1].startswith("fd7a")            # v6 after
    assert st.magic_dns == "mast-box.tail1a2b3c.ts.net"  # trailing dot stripped
    assert st.tailnet == "tail1a2b3c.ts.net"
    assert st.device_name == "mast-box"
    assert st.hint == ""                                # ready → no nag
    assert st.preferred_host == "mast-box.tail1a2b3c.ts.net"


def test_needs_login(monkeypatch):
    body = json.dumps({"BackendState": "NeedsLogin", "Self": {}})
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(stdout=body))
    st = ts.get_status(cli="tailscale")
    assert st.installed and not st.running and not st.ready
    assert "登录" in st.hint


def test_stopped(monkeypatch):
    body = json.dumps({"BackendState": "Stopped", "Self": {}})
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(stdout=body))
    st = ts.get_status(cli="tailscale")
    assert st.installed and not st.ready
    assert "未连接" in st.hint


def test_empty_stdout_treated_as_stopped(monkeypatch):
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(stdout="   "))
    st = ts.get_status(cli="tailscale")
    assert st.installed and not st.ready
    assert st.backend_state == "Stopped"


def test_malformed_json(monkeypatch):
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(stdout="<html>not json</html>"))
    st = ts.get_status(cli="tailscale")
    assert st.installed and not st.ready
    assert "解析" in st.hint


def test_timeout(monkeypatch):
    exc = subprocess.TimeoutExpired(cmd="tailscale", timeout=3.0)
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(raise_exc=exc))
    st = ts.get_status(cli="tailscale")
    assert st.installed and not st.ready
    assert "超时" in st.hint


def test_os_error(monkeypatch):
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(raise_exc=OSError("boom")))
    st = ts.get_status(cli="tailscale")
    assert st.installed and not st.ready


def test_running_but_no_ip_not_ready(monkeypatch):
    body = json.dumps({"BackendState": "Running", "Self": {"HostName": "x"}})
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(stdout=body))
    st = ts.get_status(cli="tailscale")
    assert st.running is True and st.ready is False
    assert "IP" in st.hint


def test_magicdns_off_dotless_name_ignored(monkeypatch):
    body = json.dumps({
        "BackendState": "Running",
        "Self": {"HostName": "box", "DNSName": "box", "TailscaleIPs": ["100.90.1.2"]},
    })
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(stdout=body))
    st = ts.get_status(cli="tailscale")
    assert st.magic_dns == ""                     # a dot-less DNSName is not a FQDN
    assert st.preferred_host == "100.90.1.2"      # falls back to the IP


# ── remote_urls / extra_cert_hosts / summary ─────────────────────────────────

def _ready_status():
    return ts.TailscaleStatus(
        installed=True, running=True, backend_state="Running",
        self_ips=["100.101.102.103", "fd7a:115c:a1e0::1234"],
        magic_dns="mast-box.tail1a2b3c.ts.net", tailnet="tail1a2b3c.ts.net",
        device_name="mast-box",
    )


def test_remote_urls_ip_first_magicdns_last():
    # IP is recommended: MAST uses a self-signed cert and *.ts.net is HSTS-preloaded,
    # so a browser hard-rejects the self-signed cert on the MagicDNS name (no
    # click-through). The IP is not preloaded → click-through works. So IP first.
    urls = ts.remote_urls(_ready_status(), 7862, "https")
    assert urls[0]["url"] == "https://100.101.102.103:7862"              # IPv4 recommended
    assert urls[1]["url"] == "https://[fd7a:115c:a1e0::1234]:7862"        # v6 bracketed
    assert urls[2]["url"] == "https://mast-box.tail1a2b3c.ts.net:7862"    # MagicDNS last


def test_remote_urls_empty_when_not_ready():
    assert ts.remote_urls(ts.TailscaleStatus(), 7862) == []


def test_remote_urls_scheme_respected():
    urls = ts.remote_urls(_ready_status(), 8000, "http")
    assert all(u["url"].startswith("http://") for u in urls)


def test_extra_cert_hosts():
    hosts = ts.extra_cert_hosts(_ready_status())
    assert hosts[0] == "mast-box.tail1a2b3c.ts.net"
    assert "100.101.102.103" in hosts and "fd7a:115c:a1e0::1234" in hosts


def test_summary_shape(monkeypatch):
    monkeypatch.setattr(ts.subprocess, "run", _fake_run(stdout=RUNNING_JSON))
    s = ts.summary(7862, "https")
    assert s["tailscale"]["ready"] is True
    assert s["tailscale"]["device_name"] == "mast-box"
    assert s["tailscale"]["ips"] == ["100.101.102.103", "fd7a:115c:a1e0::1234"]
    assert s["urls"][0]["url"] == "https://100.101.102.103:7862"


def test_summary_accepts_injected_status():
    s = ts.summary(9000, "https", status=_ready_status())
    assert s["urls"][0]["host"] == "100.101.102.103"       # IP recommended, first
    assert s["tailscale"]["installed"] is True
