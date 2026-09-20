"""修复项 (2026-06-11): custom-route session guard mirrors Gradio's login state.

Gradio's launch(auth=...) protects only its OWN routes via Depends(login_check);
hand-inserted Starlette routes (/agents/*, /api/*, …) and static Mounts
(/agents-ui) bypassed it. route_auth re-implements the same check (app.auth +
app.tokens + access-token cookie) for everything we mount ourselves.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/gui/test_route_auth.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.testclient import TestClient

from mast.webui.route_auth import AuthedStaticFiles, authed_route, authed_ws

COOKIE_ID = "testcookieid"
TOKEN = "tok-123"


async def _async_ep(request):
    return JSONResponse({"ok": "async"})


def _sync_ep(request):
    return JSONResponse({"ok": "sync"})


async def _ws_ep(websocket):
    await websocket.accept()
    await websocket.send_text("hello")
    await websocket.close()


def _make_app(tmp_path, *, auth_enabled: bool) -> Starlette:
    static_dir = tmp_path / "static"
    static_dir.mkdir(exist_ok=True)
    (static_dir / "index.html").write_text("<html>ui</html>", encoding="utf-8")
    app = Starlette(routes=[
        Route("/api/async", authed_route(_async_ep), methods=["GET"]),
        Route("/api/sync", authed_route(_sync_ep), methods=["GET"]),
        WebSocketRoute("/ws", authed_ws(_ws_ep)),
        Mount("/ui", app=AuthedStaticFiles(directory=str(static_dir), html=True)),
    ])
    # Mimic the gradio.routes.App attributes the real login flow populates.
    app.auth = ("user", "pass") if auth_enabled else None
    app.auth_dependency = None
    app.cookie_id = COOKIE_ID
    app.tokens = {TOKEN: "user"} if auth_enabled else {}
    return app


def _client(tmp_path, *, auth_enabled, with_cookie=False, unsecure=False):
    c = TestClient(_make_app(tmp_path, auth_enabled=auth_enabled))
    if with_cookie:
        prefix = "access-token-unsecure-" if unsecure else "access-token-"
        c.cookies.set(f"{prefix}{COOKIE_ID}", TOKEN)
    return c


class TestAuthDisabled:
    """localhost default — zero friction, everything passes."""

    def test_routes_pass(self, tmp_path):
        c = _client(tmp_path, auth_enabled=False)
        assert c.get("/api/async").status_code == 200
        assert c.get("/api/sync").status_code == 200
        assert c.get("/ui/index.html").status_code == 200

    def test_ws_passes(self, tmp_path):
        c = _client(tmp_path, auth_enabled=False)
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_text() == "hello"


class TestAuthEnabled:
    """LAN mode — only logged-in sessions pass."""

    def test_routes_401_without_session(self, tmp_path):
        c = _client(tmp_path, auth_enabled=True)
        assert c.get("/api/async").status_code == 401
        assert c.get("/api/sync").status_code == 401

    def test_static_401_without_session(self, tmp_path):
        c = _client(tmp_path, auth_enabled=True)
        assert c.get("/ui/index.html").status_code == 401

    def test_routes_pass_with_session_cookie(self, tmp_path):
        c = _client(tmp_path, auth_enabled=True, with_cookie=True)
        assert c.get("/api/async").json() == {"ok": "async"}
        assert c.get("/api/sync").json() == {"ok": "sync"}
        assert c.get("/ui/index.html").status_code == 200

    def test_unsecure_cookie_variant_accepted(self, tmp_path):
        """Gradio sets access-token-unsecure-* on plain HTTP — must work too."""
        c = _client(tmp_path, auth_enabled=True, with_cookie=True, unsecure=True)
        assert c.get("/api/async").status_code == 200

    def test_bogus_token_rejected(self, tmp_path):
        c = TestClient(_make_app(tmp_path, auth_enabled=True))
        c.cookies.set(f"access-token-{COOKIE_ID}", "forged-token")
        assert c.get("/api/async").status_code == 401

    def test_ws_rejected_without_session(self, tmp_path):
        c = _client(tmp_path, auth_enabled=True)
        with pytest.raises(Exception):  # handshake rejected / closed pre-accept
            with c.websocket_connect("/ws") as ws:
                ws.receive_text()

    def test_ws_passes_with_session(self, tmp_path):
        c = _client(tmp_path, auth_enabled=True, with_cookie=True)
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_text() == "hello"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
