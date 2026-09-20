"""Contract tests for the FULL-transcript cognition seam (additive).

Covers the NEW paths that return the transcript ``run_brainstorm`` already
computes, plus the dream-pass full-content read-back:

  * POST /api/cognition/brainstorm/full
  * POST /api/cognition/brainstorm/stream   (SSE; post-run replay, not live)
  * GET  /api/cognition/dream/full

Two layers, exactly like ``test_cognition.py``:
  * DEGRADED — bare ``AppContext`` (no live core): 200 + ``degraded=True`` empty
    transcript, never 500.
  * WIRED — a real in-memory ``MemoryStore`` on a temp SQLite DB so the offline
    (no-LLM) rule-based brainstorm/dream exercise the real backend contract.

A throwaway app mounts the router exactly as the house style prescribes.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.cognition_transcript import router


# ── degraded (no live core) ───────────────────────────────────────────


def _degraded_client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_brainstorm_full_degraded() -> None:
    c = _degraded_client()
    r = c.post(
        "/api/cognition/brainstorm/full",
        json={"topic": "tip shaping", "viewpoints": ["a"], "max_rounds": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["transcript"] == []
    assert body["summary"] == ""
    assert body["written_memory_ids"] == []


def test_brainstorm_stream_degraded() -> None:
    c = _degraded_client()
    r = c.post("/api/cognition/brainstorm/stream", json={"topic": "x"})
    assert r.status_code == 200
    frames = _parse_sse(r.text)
    kinds = [f["kind"] for f in frames]
    assert "error" in kinds
    assert frames[-1]["kind"] == "done"
    assert frames[-1]["streaming"] is False


def test_dream_full_degraded() -> None:
    c = _degraded_client()
    r = c.get("/api/cognition/dream/full")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["entries"] == [] and body["count"] == 0


# ── wired (real MemoryStore on a temp DB) ─────────────────────────────


class _StorageStub:
    """Minimal ExperimentStorage shape: just a ``_db_path`` so MemoryStore +
    the brainstorm/dream grounding can open the DB."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path


def _wired_client(tmp_path) -> TestClient:
    from mast.memory.store import MemoryStore

    db = str(tmp_path / "exp.db")
    store = MemoryStore(db)  # creates tables
    ctx = AppContext()
    ctx.memory_store = store  # type: ignore[attr-defined] - additive wiring
    ctx._experiment_storage = _StorageStub(db)  # noqa: SLF001
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_brainstorm_full_wired_offline(tmp_path) -> None:
    # No LLM wired → run_brainstorm runs the rule-based offline discussion and
    # returns a real, non-empty transcript + summary (degraded=False).
    c = _wired_client(tmp_path)
    r = c.post(
        "/api/cognition/brainstorm/full",
        json={"experiment_id": None, "topic": "next step",
              "viewpoints": ["safety first"], "max_rounds": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["summary"].strip() != ""
    assert len(body["transcript"]) > 0
    # transcript turns carry the documented shape
    t0 = body["transcript"][0]
    for key in ("round", "speaker", "role", "viewpoint", "content"):
        assert key in t0
    # at least one viewpoint turn has a non-null viewpoint == its role
    vp_turns = [t for t in body["transcript"] if t["viewpoint"] is not None]
    assert vp_turns, "expected viewpoint turns"
    assert all(t["viewpoint"] == t["role"] for t in vp_turns)
    # facilitator/user turns have viewpoint=None
    assert any(
        t["viewpoint"] is None and t["role"] in ("facilitator", "user")
        for t in body["transcript"]
    )
    # the summary was persisted → its deterministic memory path is reported
    assert body["written_memory_ids"] == ["brainstorms/next step.md"]


def test_brainstorm_full_user_viewpoints_seeded(tmp_path) -> None:
    c = _wired_client(tmp_path)
    r = c.post(
        "/api/cognition/brainstorm/full",
        json={"topic": "drift", "viewpoints": ["my opinion XYZ"], "max_rounds": 1},
    )
    body = r.json()
    assert body["degraded"] is False
    user_turns = [t for t in body["transcript"] if t["role"] == "user"]
    assert any("my opinion XYZ" in t["content"] for t in user_turns)


def test_brainstorm_stream_wired_offline(tmp_path) -> None:
    c = _wired_client(tmp_path)
    with c.stream(
        "POST", "/api/cognition/brainstorm/stream",
        json={"topic": "next step", "viewpoints": [], "max_rounds": 1},
    ) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    frames = _parse_sse(text)
    kinds = [f["kind"] for f in frames]
    assert "turn" in kinds
    assert "summary" in kinds
    assert frames[-1]["kind"] == "done"
    # honesty: streaming is advertised as NOT live (no per-round generator)
    assert frames[-1]["streaming"] is False
    assert "note" in frames[-1]
    # a turn frame carries the transcript shape
    turn = next(f for f in frames if f["kind"] == "turn")
    for key in ("round", "speaker", "role", "viewpoint", "content"):
        assert key in turn


def test_dream_full_wired_offline(tmp_path) -> None:
    # No experiments in the DB → rule-based consolidator writes nothing, but the
    # pass runs and returns a real (empty) entry list (degraded=False).
    c = _wired_client(tmp_path)
    r = c.get("/api/cognition/dream/full", params={"namespace": "global"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert isinstance(body["entries"], list)
    assert body["count"] == len(body["entries"])


def test_routes_registered() -> None:
    c = _degraded_client()
    spec = c.get("/openapi.json").json()
    paths = set(spec.get("paths", {}))
    assert "/api/cognition/brainstorm/full" in paths
    assert "/api/cognition/brainstorm/stream" in paths
    assert "/api/cognition/dream/full" in paths
    # must NOT collide with the existing ack endpoints
    assert "/api/cognition/brainstorm" not in paths
    assert "/api/cognition/dream" not in paths


# ── helpers ───────────────────────────────────────────────────────────


def _parse_sse(text: str) -> list[dict]:
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            out.append(json.loads(line[len("data:"):].strip()))
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
