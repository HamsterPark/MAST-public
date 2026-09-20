"""Parity follow-up — literature-library *ext2* route contract tests.

Per the house test rule the router under test is NOT yet included in
``mast.api.app`` (integration wires that); we mount it on a throwaway
FastAPI app with a fresh AppContext. We assert:

  * every endpoint returns 200 + the schema-shaped body (never a 500);
  * the LIVE path through the real ``LibraryRegistry`` / ``FetchBoard`` (each
    pointed at a temp dir via the singleton-injection hooks) — activate /
    get-detail / remove-members / resolve all relay correctly;
  * honest domain errors (unknown library id, unknown request id, bad action)
    are ``ok=False, degraded=False`` with a message — NOT degraded, NOT 500;
  * a degraded path (library backend import made to fail) returns
    ``degraded=True`` empty-but-not-broken.

The registry/board are JSON-file singletons; we redirect their default to a
temp dir and reset the cached singleton around each test so we never touch the
real ``MASTv2/artifacts/literature_libs``.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.literature_ext2 import router


# ── throwaway app ──────────────────────────────────────────────────────


def _client(ctx: AppContext | None = None, *, wired: bool = True) -> TestClient:
    """A throwaway app around the ext2 router.

    ``wired=True`` marks the literature backend as present, which every handler now
    requires (2026-07-29): the ``_is_wired`` gate was missing here while
    ``routes/literature.py`` had it on every handler, so with no core wired the
    list endpoints reported ``degraded`` while activate / remove still rewrote
    ``registry.json`` for real. Pass ``wired=False`` to exercise the degraded
    contract.
    """
    app = FastAPI()
    c = ctx or AppContext()
    if wired:
        c.literature_wired = True
    app.state.ctx = c
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ── isolated registry + board on a temp dir ────────────────────────────


@pytest.fixture()
def isolated_backends(tmp_path, monkeypatch):
    """Point the libraries registry + fetch board singletons at a temp dir.

    Returns the two live modules so a test can seed them directly through the
    same backend the routes relay onto.

    Isolation goes through the ``MAST_LITERATURE_LIBS_DIR`` env override — the
    knob ``knowledge/paths.py`` publishes — **not** a ``monkeypatch.setattr`` on a
    module constant. This fixture used to patch ``lib_mod._DEFAULT_LIBS_DIR`` with
    ``raising=False``; when that constant was removed (paths converged to one
    resolver, 2026-07-29) the patch became a silent no-op and this test file wrote
    four fixture libraries straight into the operator's real ``registry.json``.
    ``raising=False`` on an isolation knob is a trap: it turns "the thing I am
    patching no longer exists" from a loud error into live-data corruption.
    """
    from mast.knowledge import fetch_board as board_mod
    from mast.knowledge import libraries as lib_mod

    libs_dir = tmp_path / "libs"
    monkeypatch.setenv("MAST_LITERATURE_LIBS_DIR", str(libs_dir))

    lib_mod.reset_default_registry()
    board_mod.reset_default_board()
    # Belt and braces: prove the isolation actually took before any test writes.
    assert lib_mod.get_registry()._dir == libs_dir
    assert board_mod.get_board()._dir == libs_dir
    try:
        yield lib_mod, board_mod
    finally:
        lib_mod.reset_default_registry()
        board_mod.reset_default_board()


# ── activate ───────────────────────────────────────────────────────────


def test_activate_library_live(isolated_backends):
    lib_mod, _ = isolated_backends
    rec = lib_mod.create_library("My Lab Papers")
    lib_id = rec["library_id"]

    r = _client().post(f"/api/literature/libraries/{lib_id}/activate")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["degraded"] is False
    assert body["library_id"] == lib_id
    assert body["library"]["library_id"] == lib_id
    # backend persisted the choice
    assert lib_mod.get_active()["library_id"] == lib_id


def test_activate_unknown_library_is_domain_error(isolated_backends):
    r = _client().post("/api/literature/libraries/does_not_exist/activate")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False  # honest "no such library", not degraded
    assert body["message"]


def test_activate_degraded_when_backend_raises(isolated_backends, monkeypatch):
    """An unexpected (non-LibraryError) raise from the backend ⇒ degraded, not 500."""
    lib_mod, _ = isolated_backends

    def boom(*_a, **_k):
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(lib_mod, "set_active_library", boom)
    r = _client().post("/api/literature/libraries/reading/activate")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


# ── get detail ─────────────────────────────────────────────────────────


def test_get_library_detail_live(isolated_backends):
    lib_mod, _ = isolated_backends
    rec = lib_mod.create_library("Detail Lib")
    lib_id = rec["library_id"]
    lib_mod.add_members(["W123", "W456"], library_id=lib_id, added_by="user")

    r = _client().get(f"/api/literature/libraries/{lib_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["degraded"] is False
    assert body["member_count"] == 2
    wids = {m["work_id"] for m in body["library"]["members"]}
    assert wids == {"W123", "W456"}


def test_get_library_detail_unknown_is_not_found(isolated_backends):
    r = _client().get("/api/literature/libraries/nope")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False
    assert body["degraded"] is False
    assert body["library"] is None


# ── remove members ─────────────────────────────────────────────────────


def test_remove_members_live(isolated_backends):
    lib_mod, _ = isolated_backends
    rec = lib_mod.create_library("Removable")
    lib_id = rec["library_id"]
    lib_mod.add_members(["W1", "W2", "W3"], library_id=lib_id, added_by="user")

    r = _client().post(
        f"/api/literature/libraries/{lib_id}/members/remove",
        json={"work_ids": ["W1", "W3"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["degraded"] is False
    assert body["n_removed"] == 2
    assert set(body["removed"]) == {"W1", "W3"}
    assert body["member_count"] == 1
    # confirm via the backend
    remaining = {m["work_id"] for m in lib_mod.get_library(lib_id)["members"]}
    assert remaining == {"W2"}


def test_remove_members_unknown_library_is_domain_error(isolated_backends):
    r = _client().post(
        "/api/literature/libraries/ghost/members/remove",
        json={"work_ids": ["W1"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False
    assert body["message"]


def test_remove_members_empty_is_noop_success(isolated_backends):
    lib_mod, _ = isolated_backends
    lib_id = lib_mod.create_library("EmptyRemove")["library_id"]
    r = _client().post(
        f"/api/literature/libraries/{lib_id}/members/remove",
        json={"work_ids": []},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["n_removed"] == 0


# ── resolve fetch request ──────────────────────────────────────────────


def test_resolve_fetch_done_live(isolated_backends):
    _, board_mod = isolated_backends
    req = board_mod.post_request("W999", title="Some paper", reason="need full text")
    rid = req["request_id"]

    r = _client().post(
        f"/api/literature/fetch-board/{rid}/resolve",
        json={"action": "done", "note": "uploaded pdf"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["degraded"] is False
    assert body["status"] == "fulfilled"
    assert body["work_id"] == "W999"
    assert body["note"] == "uploaded pdf"
    assert board_mod.get_board().get_request(rid)["status"] == "fulfilled"


def test_resolve_fetch_dismissed_live(isolated_backends):
    _, board_mod = isolated_backends
    rid = board_mod.post_request("W888")["request_id"]
    r = _client().post(
        f"/api/literature/fetch-board/{rid}/resolve",
        json={"action": "dismissed"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "dismissed"


def test_resolve_fetch_invalid_action_is_domain_error(isolated_backends):
    _, board_mod = isolated_backends
    rid = board_mod.post_request("W777")["request_id"]
    r = _client().post(
        f"/api/literature/fetch-board/{rid}/resolve",
        json={"action": "frobnicate"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False
    assert "invalid action" in body["message"]


def test_resolve_fetch_unknown_request_is_domain_error(isolated_backends):
    r = _client().post(
        "/api/literature/fetch-board/fr-99999/resolve",
        json={"action": "done"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is False
    assert body["message"]
