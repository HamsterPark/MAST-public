"""Phase-3 cognition API contract tests (Domain F).

Covers the typed seam for memory CRUD + pin + search, dreaming, and brainstorm.
Two layers:

  * DEGRADED — a throwaway app with a bare ``AppContext`` (no live core wired):
    every endpoint must return 200 with a valid empty/degraded shape (never 500).
  * WIRED — a context with a real in-memory ``MemoryStore`` (backed by a temp
    SQLite DB via ExperimentStorage-shaped stub) so the CRUD round-trip + the
    offline (no-LLM) dream pass exercise the real backend contract.

The leader integrates this router into ``mast.api.app`` separately, so we mount a
throwaway app here exactly as the house style prescribes.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.cognition import router


# ── degraded (no live core) ───────────────────────────────────────────


def _degraded_client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_namespaces_degraded() -> None:
    c = _degraded_client()
    r = c.get("/api/memory/namespaces")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["namespaces"] == ["global"]


def test_list_degraded() -> None:
    c = _degraded_client()
    r = c.get("/api/memory/global")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["entries"] == [] and body["count"] == 0
    assert body["namespace"] == "global"


def test_list_with_filters_degraded() -> None:
    c = _degraded_client()
    r = c.get("/api/memory/global", params={"kind": "note", "query": "tip"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["entries"] == []


def test_read_degraded() -> None:
    c = _degraded_client()
    r = c.get("/api/memory/global/insights/tip.md")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["found"] is False
    assert body["entry"] is None


def test_write_degraded() -> None:
    c = _degraded_client()
    r = c.post("/api/memory/global/insights/tip.md", json={"content": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


def test_delete_degraded() -> None:
    c = _degraded_client()
    r = c.request("DELETE", "/api/memory/global/insights/tip.md")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


def test_pin_degraded() -> None:
    c = _degraded_client()
    r = c.post("/api/memory/global/insights/tip.md/pin", json={"on": True})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


def test_dream_degraded() -> None:
    c = _degraded_client()
    r = c.post("/api/cognition/dream", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["job"] == "dream"
    assert body["degraded"] is True
    assert body["status"] == "degraded"
    assert body["written"] == []


def test_brainstorm_degraded() -> None:
    c = _degraded_client()
    r = c.post(
        "/api/cognition/brainstorm",
        json={"topic": "tip shaping", "viewpoints": ["a", "b"], "max_rounds": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["job"] == "brainstorm"
    assert body["degraded"] is True


# ── wired (real MemoryStore on a temp DB) ─────────────────────────────


class _StorageStub:
    """Minimal ExperimentStorage shape the cognition route needs: just exposes
    a ``_db_path`` so ``MemoryStore.from_storage`` + the dream pass can open it."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path


def _wired_client(tmp_path) -> TestClient:
    from mast.memory.store import MemoryStore

    db = str(tmp_path / "exp.db")
    store = MemoryStore(db)  # creates tables
    ctx = AppContext()
    # The route resolves either ctx.memory_store (preferred) or builds one off
    # ctx.experiment_storage. Wire BOTH so dream's db_path resolution works too.
    ctx._settings_store = None  # noqa: SLF001 - keep standalone settings untouched
    ctx.memory_store = store  # type: ignore[attr-defined] - additive future wiring
    ctx._experiment_storage = _StorageStub(db)  # noqa: SLF001
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_memory_crud_roundtrip(tmp_path) -> None:
    c = _wired_client(tmp_path)

    # initially empty namespace listing
    r = c.get("/api/memory/global")
    assert r.status_code == 200
    assert r.json()["degraded"] is False
    assert r.json()["count"] == 0

    # write
    r = c.post(
        "/api/memory/global/insights/tip.md",
        json={"content": "use 50 pm/V", "title": "Tip tip", "kind": "insight",
              "tags": ["tip", "sts"], "pin": False},
    )
    assert r.status_code == 200
    w = r.json()
    assert w["ok"] is True and w["degraded"] is False
    assert w["namespace"] == "global" and w["path"] == "insights/tip.md"
    assert isinstance(w["id"], int)

    # namespaces now includes global (always first)
    r = c.get("/api/memory/namespaces")
    assert r.json()["degraded"] is False
    assert r.json()["namespaces"][0] == "global"

    # read back the full row
    r = c.get("/api/memory/global/insights/tip.md")
    assert r.status_code == 200
    d = r.json()
    assert d["found"] is True and d["degraded"] is False
    entry = d["entry"]
    assert entry["content"] == "use 50 pm/V"
    assert entry["title"] == "Tip tip"
    assert entry["kind"] == "insight"
    assert entry["tags"] == ["tip", "sts"]
    assert entry["pinned"] is False

    # list shows it
    r = c.get("/api/memory/global")
    body = r.json()
    assert body["count"] == 1
    assert body["entries"][0]["path"] == "insights/tip.md"

    # kind filter
    r = c.get("/api/memory/global", params={"kind": "insight"})
    assert r.json()["count"] == 1
    r = c.get("/api/memory/global", params={"kind": "note"})
    assert r.json()["count"] == 0

    # substring search
    r = c.get("/api/memory/global", params={"query": "50 pm"})
    assert r.json()["count"] == 1
    r = c.get("/api/memory/global", params={"query": "nonexistent-xyz"})
    assert r.json()["count"] == 0

    # pin it
    r = c.post("/api/memory/global/insights/tip.md/pin", json={"on": True})
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["pinned"] is True
    # confirm pin persisted
    r = c.get("/api/memory/global/insights/tip.md")
    assert r.json()["entry"]["pinned"] is True

    # unpin
    r = c.post("/api/memory/global/insights/tip.md/pin", json={"on": False})
    assert r.json()["pinned"] is False

    # delete
    r = c.request("DELETE", "/api/memory/global/insights/tip.md")
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["deleted"] is True
    # gone now
    r = c.get("/api/memory/global/insights/tip.md")
    assert r.json()["found"] is False
    # deleting again → ok but deleted=False
    r = c.request("DELETE", "/api/memory/global/insights/tip.md")
    assert r.json()["ok"] is True and r.json()["deleted"] is False


def test_pin_missing_memory_wired(tmp_path) -> None:
    c = _wired_client(tmp_path)
    r = c.post("/api/memory/global/nope.md/pin", json={"on": True})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is False
    assert "no such" in body["message"].lower()


def test_read_missing_memory_wired(tmp_path) -> None:
    c = _wired_client(tmp_path)
    r = c.get("/api/memory/global/nope.md")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False and body["degraded"] is False
    assert body["entry"] is None


def test_dream_wired_offline(tmp_path) -> None:
    # No experiments in the DB → rule-based consolidator writes nothing, but the
    # pass itself runs and acks (accepted, not degraded).
    c = _wired_client(tmp_path)
    r = c.post("/api/cognition/dream", json={"namespace": "global"})
    assert r.status_code == 200
    body = r.json()
    assert body["job"] == "dream"
    assert body["degraded"] is False
    assert body["status"] == "accepted"
    assert isinstance(body["written"], list)


def test_brainstorm_wired_ack(tmp_path) -> None:
    c = _wired_client(tmp_path)
    r = c.post(
        "/api/cognition/brainstorm",
        json={"experiment_id": None, "topic": "next step",
              "viewpoints": ["safety first"], "max_rounds": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["job"] == "brainstorm"
    # brainstorm backend is importable in this repo → accepted, not degraded.
    assert body["degraded"] is False
    assert body["status"] == "accepted"


def test_routes_registered() -> None:
    c = _degraded_client()
    spec = c.get("/openapi.json").json()
    paths = set(spec.get("paths", {}))
    assert "/api/memory/namespaces" in paths
    assert "/api/memory/{ns}" in paths
    assert "/api/memory/{ns}/{path}" in paths
    assert "/api/memory/{ns}/{path}/pin" in paths
    assert "/api/cognition/dream" in paths
    assert "/api/cognition/brainstorm" in paths


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
