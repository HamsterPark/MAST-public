"""Contract tests for GET /api/remote-access (设置 → 远程访问).

Tailscale detection is monkeypatched, so the route is exercised across bind
states + Tailscale states with no daemon. See mast/net/tailscale.py.
"""
from __future__ import annotations

import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.remote_access import router
from mast.net.tailscale import TailscaleStatus


def _client(*, bound_host: str | None = None, ssl: bool = False, port: int = 7862) -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    if bound_host is not None:
        cfg = types.SimpleNamespace(
            host=bound_host, port=port,
            ssl_certfile=("cert.pem" if ssl else None),
        )
        app.state.uvicorn_server = types.SimpleNamespace(config=cfg)
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _ready() -> TailscaleStatus:
    return TailscaleStatus(
        installed=True, running=True, backend_state="Running",
        self_ips=["100.101.102.103", "fd7a:115c:a1e0::9"],
        magic_dns="mast-box.tail1a2b.ts.net", tailnet="tail1a2b.ts.net",
        device_name="mast-box",
    )


def test_not_bound_reports_local_only(monkeypatch):
    monkeypatch.setattr("mast.net.tailscale.get_status",
                        lambda *a, **k: TailscaleStatus(installed=False, hint="装 Tailscale"))
    body = _client().get("/api/remote-access").json()          # no uvicorn_server → not bound
    assert body["lan_enabled"] is False
    assert "远程访问未开启" in body["note"]
    assert body["urls"] == []
    assert body["tailscale"]["installed"] is False


def test_bound_and_ready_gives_url(monkeypatch):
    monkeypatch.setattr("mast.net.tailscale.get_status", lambda *a, **k: _ready())
    body = _client(bound_host="0.0.0.0", ssl=True, port=7862).get("/api/remote-access").json()
    assert body["lan_enabled"] is True
    assert body["scheme"] == "https"
    assert body["port"] == 7862
    assert body["tailscale"]["ready"] is True
    assert body["tailscale"]["device_name"] == "mast-box"
    # IP recommended first (self-signed + *.ts.net HSTS → MagicDNS blocked in browser)
    assert body["urls"][0]["url"] == "https://100.101.102.103:7862"
    assert body["urls"][-1]["url"] == "https://mast-box.tail1a2b.ts.net:7862"
    assert "就绪" in body["note"]


def test_bound_but_tailscale_stopped_surfaces_hint(monkeypatch):
    st = TailscaleStatus(installed=True, running=False, backend_state="Stopped",
                         hint="打开 Tailscale 并保持登录")
    monkeypatch.setattr("mast.net.tailscale.get_status", lambda *a, **k: st)
    body = _client(bound_host="0.0.0.0", ssl=False, port=7862).get("/api/remote-access").json()
    assert body["lan_enabled"] is True
    assert body["scheme"] == "http"          # no ssl_certfile → http
    assert body["urls"] == []
    assert body["note"] == "打开 Tailscale 并保持登录"


def test_ipv6_bind_counts_as_bound(monkeypatch):
    monkeypatch.setattr("mast.net.tailscale.get_status", lambda *a, **k: _ready())
    body = _client(bound_host="::", ssl=True).get("/api/remote-access").json()
    assert body["lan_enabled"] is True


def test_response_shape_is_typed(monkeypatch):
    monkeypatch.setattr("mast.net.tailscale.get_status", lambda *a, **k: _ready())
    body = _client(bound_host="0.0.0.0", ssl=True).get("/api/remote-access").json()
    for key in ("lan_enabled", "scheme", "port", "lan_ip", "tailscale", "urls", "note"):
        assert key in body
    for key in ("installed", "running", "ready", "backend_state", "device_name",
                "magic_dns", "tailnet", "ips", "hint"):
        assert key in body["tailscale"]
