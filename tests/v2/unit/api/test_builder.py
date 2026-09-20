"""Contract tests for the composite/builder WRITE seam (routes/builder.py).

These cover the save/create/update/delete + validate + catalog endpoints the
new BuilderPage needs. Every endpoint must return its declared status with a
schema-shaped body, and every degraded path must be empty-not-broken
(``degraded=True``, never a 500). An invalid spec is NOT degradation — it is a
typed ``ok=False`` with the design-time report (fail-closed).

The router is tested via a throwaway app (integration wires it into
``mast.api.app`` separately). Write roundtrips use a uniquely-named composite
and clean up after themselves; they relay onto the real file-backed version
store, which works standalone.
"""

from __future__ import annotations

import builtins

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.builder import router

# A minimal spec that the design-time validator accepts (empty node list →
# no unknown-skill / bounds problems; needs no live registry).
_EMPTY_NODES_SPEC = {
    "name": "",  # filled per-test
    "description": "builder write seam test composite",
    "safety_level": "confirm",
    "params": [],
    "nodes": [],
    "tags": ["pytest"],
}

_TEST_NAME = "__test_builder_write_seam__"


def _spec(name: str) -> dict:
    s = dict(_EMPTY_NODES_SPEC)
    s["name"] = name
    return s


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture()
def client() -> TestClient:
    return _client()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the composite version store at a per-test tmp dir so write
    roundtrips (and the append-only ``_history`` snapshots the store keeps by
    design) never pollute the user's real ``config/composite_skills/``.

    Replaces the ``composite_panel`` singleton with a tmp-rooted store; the
    routes call ``composite_store()`` which returns it. Restored automatically."""
    from mast.skills.composite.version_store import CompositeVersionStore
    import mast.webui.composite_panel as cp

    store = CompositeVersionStore(root=tmp_path / "composite_skills")
    monkeypatch.setattr(cp, "_store", store, raising=False)
    yield store


@pytest.fixture()
def clean_store(isolated_store):
    """Compat alias — the store is already isolated per test; nothing to wipe."""
    yield isolated_store


# ── POST /api/composites (create) ─────────────────────────────────────────────


def test_create_name_from_url_injected(client: TestClient, clean_store) -> None:
    # A spec with no own "name" inherits the request ``name`` (URL/body) — the
    # store needs a name and the API aligns them without inventing logic.
    bare = {"description": "no name", "safety_level": "confirm",
            "params": [], "nodes": [], "tags": []}
    r = client.post("/api/composites", json={"name": _TEST_NAME, "spec": bare})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body
    assert body["name"] == _TEST_NAME
    assert body["degraded"] is False


def test_create_roundtrip(client: TestClient, clean_store) -> None:
    r = client.post("/api/composites",
                    json={"name": _TEST_NAME, "spec": _spec(_TEST_NAME)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body
    assert body["name"] == _TEST_NAME
    assert body["version"] >= 1
    assert body["degraded"] is False
    # Verify it actually persisted via the store.
    from mast.webui.composite_panel import composite_store
    assert composite_store().exists(_TEST_NAME)


def test_create_rejects_existing(client: TestClient, clean_store) -> None:
    first = client.post("/api/composites",
                        json={"name": _TEST_NAME, "spec": _spec(_TEST_NAME)})
    assert first.json()["ok"] is True
    again = client.post("/api/composites",
                        json={"name": _TEST_NAME, "spec": _spec(_TEST_NAME)})
    assert again.status_code == 200
    body = again.json()
    assert body["ok"] is False
    assert body["error"] == "already_exists"
    assert body["degraded"] is False


def test_create_name_mismatch(client: TestClient) -> None:
    r = client.post("/api/composites",
                    json={"name": "AAA", "spec": _spec("BBB")})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "name_mismatch"
    assert body["degraded"] is False


def test_create_degrades_on_backend_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.composite_panel":
            raise RuntimeError("store unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.post("/api/composites",
                    json={"name": _TEST_NAME, "spec": _spec(_TEST_NAME)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


# ── PUT /api/composites/{name} (update / overwrite) ───────────────────────────


def test_update_creates_new_version(client: TestClient, clean_store) -> None:
    created = client.post("/api/composites",
                          json={"name": _TEST_NAME, "spec": _spec(_TEST_NAME)})
    v1 = created.json()["version"]
    upd = client.put(f"/api/composites/{_TEST_NAME}",
                     json={"spec": _spec(_TEST_NAME)})
    assert upd.status_code == 200
    body = upd.json()
    assert body["ok"] is True, body
    # Overwrite writes forward as a NEW version (history retained).
    assert body["version"] > v1
    assert body["degraded"] is False


def test_update_version_conflict(client: TestClient, clean_store) -> None:
    created = client.post("/api/composites",
                          json={"name": _TEST_NAME, "spec": _spec(_TEST_NAME)})
    stored_v = created.json()["version"]
    # Pass a stale base_version → optimistic-concurrency rejection.
    stale = stored_v - 1 if stored_v > 1 else 0
    upd = client.put(f"/api/composites/{_TEST_NAME}",
                     json={"spec": _spec(_TEST_NAME), "base_version": stale})
    assert upd.status_code == 200
    body = upd.json()
    assert body["ok"] is False
    assert body["error"] == "version_conflict"
    assert body["stored_version"] == stored_v
    assert body["degraded"] is False


def test_update_name_mismatch(client: TestClient) -> None:
    r = client.put("/api/composites/AAA", json={"spec": _spec("BBB")})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "name_mismatch"


def test_update_degrades_on_backend_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.composite_panel":
            raise RuntimeError("store unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.put(f"/api/composites/{_TEST_NAME}",
                   json={"spec": _spec(_TEST_NAME)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


# ── DELETE /api/composites/{name} ─────────────────────────────────────────────


def test_delete_roundtrip(client: TestClient, clean_store) -> None:
    client.post("/api/composites",
                json={"name": _TEST_NAME, "spec": _spec(_TEST_NAME)})
    r = client.delete(f"/api/composites/{_TEST_NAME}")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["found"] is True
    assert body["degraded"] is False
    from mast.webui.composite_panel import composite_store
    assert not composite_store().exists(_TEST_NAME)


def test_delete_unknown(client: TestClient) -> None:
    r = client.delete("/api/composites/__no_such_composite_xyz__")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["found"] is False
    assert body["error"] == "not_found"
    assert body["degraded"] is False


def test_delete_degrades_on_backend_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.composite_panel":
            raise RuntimeError("store unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.delete(f"/api/composites/{_TEST_NAME}")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


# ── POST /api/builder/validate ────────────────────────────────────────────────


def test_validate_returns_report(client: TestClient) -> None:
    r = client.post("/api/builder/validate", json={"spec": _spec("X")})
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and isinstance(body["problems"], list)
    assert isinstance(body["steps"], list)
    assert isinstance(body["degraded"], bool)


def test_validate_empty_body(client: TestClient) -> None:
    r = client.post("/api/builder/validate", json={})
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and isinstance(body["problems"], list)


def test_validate_degrades_on_backend_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "mast.webui.builder_api":
            raise RuntimeError("validator unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.post("/api/builder/validate", json={"spec": _spec("X")})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["ok"] is False
    assert body["problems"]


# ── GET /api/builder/catalog ──────────────────────────────────────────────────


def test_catalog_degrades_unwired(client: TestClient) -> None:
    # No registry wired → empty but not broken.
    r = client.get("/api/builder/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["skills"] == []
    assert body["total"] == 0


def test_catalog_real_when_registry_wired(client: TestClient) -> None:
    import mast.webui.builder_api as ba

    client.app.state.ctx._skill_registry = object()
    fake = {
        "index": [
            {"name": "SetBias", "zh": "设置偏压", "category": "BIAS",
             "safety": "confirm", "level": 0, "source": "builtin",
             "source_zh": "内置", "tags": ["bias"], "domain": "偏压与电流"},
            {"name": "Scan", "zh": "扫描", "category": "SCAN",
             "safety": "auto", "level": 0, "source": "builtin",
             "source_zh": "内置", "tags": ["scan"], "domain": "扫描"},
        ],
        "cards": {},
    }
    orig = ba.get_catalog
    ba.get_catalog = lambda: fake  # type: ignore[assignment]
    try:
        r = client.get("/api/builder/catalog")
        assert r.status_code == 200
        body = r.json()
        assert body["degraded"] is False
        assert body["total"] == 2
        names = {s["name"] for s in body["skills"]}
        assert names == {"SetBias", "Scan"}
        # Filter contract: source/domain/q apply via filter_index.
        r2 = client.get("/api/builder/catalog", params={"q": "bias"})
        b2 = r2.json()
        assert {s["name"] for s in b2["skills"]} == {"SetBias"}
        # Carries the builder-only facets.
        sb = next(s for s in body["skills"] if s["name"] == "SetBias")
        assert sb["source_zh"] == "内置"
        assert sb["domain"] == "偏压与电流"
    finally:
        ba.get_catalog = orig  # type: ignore[assignment]
