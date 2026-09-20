"""Contract tests for Domain H — Wishlist + agent requests.

The router under test is NOT yet mounted in mast.api.app (the integrator integrates
that). We exercise it via a throwaway FastAPI app with a fresh AppContext, per
house style. Two axes are covered for every endpoint:

* the LIVE path — a real WishlistBoard wired onto ctx, pointed at a tmp dir so
  the suite never touches the shared artifacts board;
* the DEGRADED path — a board whose calls raise (or absent core) → every
  endpoint returns a valid empty/degraded response, never a 500.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.wishlist import router
from mast.wishlist.store import WishlistBoard


def _client(ctx: AppContext) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


class _BoomBoard:
    """A board whose every method raises — proves the handlers degrade."""

    def __getattr__(self, _name):
        def _raise(*_a, **_k):
            raise RuntimeError("boom")

        return _raise


@pytest.fixture()
def live_board(tmp_path) -> WishlistBoard:
    # isolated board dir → never writes the shared artifacts/wishlist.json
    return WishlistBoard(board_dir=str(tmp_path / "wl"))


@pytest.fixture()
def live_client(live_board) -> TestClient:
    ctx = AppContext()
    ctx.wishlist_board = live_board  # Phase-3-style wiring
    return _client(ctx)


@pytest.fixture()
def boom_client() -> TestClient:
    ctx = AppContext()
    ctx.wishlist_board = _BoomBoard()
    return _client(ctx)


# ── GET /api/wishlist ──────────────────────────────────────────────────────
def test_get_wishlist_empty_live(live_client: TestClient) -> None:
    r = live_client.get("/api/wishlist")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["wishes"] == []
    assert body["requests"] == []
    assert body["pending_count"] == 0


def test_get_wishlist_with_data(live_client: TestClient, live_board: WishlistBoard) -> None:
    live_board.add_wish("更亮的暗色主题", category="功能建议", client_version="3.2.0")
    live_board.post_agent_request("literature", "请上传 X 的全文", kind="upload")
    r = live_client.get("/api/wishlist")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert len(body["wishes"]) == 1
    w = body["wishes"][0]
    assert w["text"] == "更亮的暗色主题"
    assert w["category"] == "功能建议"
    assert w["status"] == "queued"
    assert w["id"].startswith("w-")
    assert len(body["requests"]) == 1
    req = body["requests"][0]
    assert req["agent_id"] == "literature"
    assert req["message"] == "请上传 X 的全文"
    assert req["status"] == "pending"
    assert req["id"].startswith("r-")
    assert body["pending_count"] == 1


def test_get_wishlist_degrades_on_boom(boom_client: TestClient) -> None:
    r = boom_client.get("/api/wishlist")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["wishes"] == [] and body["requests"] == []
    assert body["pending_count"] == 0


# ── POST /api/wishlist/wishes ──────────────────────────────────────────────
def test_submit_wish_live(live_client: TestClient, live_board: WishlistBoard) -> None:
    r = live_client.post(
        "/api/wishlist/wishes",
        json={"text": "支持导出 CSV", "category": "功能建议"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["degraded"] is False
    assert body["wish"] is not None
    assert body["wish"]["text"] == "支持导出 CSV"
    # status is the freshly-queued local record (async relay flips it later)
    assert body["wish"]["status"] == "queued"
    # the wish is actually persisted on the board
    assert any(w.get("text") == "支持导出 CSV" for w in live_board.list_wishes())


def test_submit_wish_empty_rejected(live_client: TestClient) -> None:
    r = live_client.post("/api/wishlist/wishes", json={"text": "   "})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["wish"] is None
    assert body["degraded"] is False  # rejected, not degraded


def test_submit_wish_degrades_on_boom(boom_client: TestClient) -> None:
    r = boom_client.post("/api/wishlist/wishes", json={"text": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True
    assert body["wish"] is None


# ── POST /api/wishlist/requests/{id}/resolve ───────────────────────────────
def test_resolve_request_done(live_client: TestClient, live_board: WishlistBoard) -> None:
    rec = live_board.post_agent_request("instrument", "请操作硬件", kind="action")
    rid = rec["id"]
    r = live_client.post(
        f"/api/wishlist/requests/{rid}/resolve",
        json={"action": "done", "note": "已处理"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["degraded"] is False
    assert body["request"]["status"] == "done"
    assert body["request"]["note"] == "已处理"
    assert body["request"]["id"] == rid


def test_resolve_request_dismissed(live_client: TestClient, live_board: WishlistBoard) -> None:
    rec = live_board.post_agent_request("instrument", "随便看看", kind="action")
    rid = rec["id"]
    r = live_client.post(
        f"/api/wishlist/requests/{rid}/resolve",
        json={"action": "dismissed"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["request"]["status"] == "dismissed"


def test_resolve_unknown_id_not_degraded(live_client: TestClient) -> None:
    r = live_client.post(
        "/api/wishlist/requests/r-999/resolve",
        json={"action": "done"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False  # not found ≠ subsystem down
    assert body["request"] is None


def test_resolve_invalid_action_422(live_client: TestClient) -> None:
    # "queued" is a wish status, not a valid resolve action → schema rejects it
    r = live_client.post(
        "/api/wishlist/requests/r-1/resolve",
        json={"action": "queued"},
    )
    assert r.status_code == 422


def test_resolve_degrades_on_boom(boom_client: TestClient) -> None:
    r = boom_client.post(
        "/api/wishlist/requests/r-1/resolve",
        json={"action": "done"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True
    assert body["request"] is None
