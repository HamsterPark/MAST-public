"""Phase-3 realtime channel tests (degraded / standalone mode).

Verifies the WS/SSE/polling contract boots and degrades safely with no live
BufferService wired: snapshot is empty-not-broken, the WS sends a snapshot frame
first, and SSE opens as text/event-stream with a snapshot data frame.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.ws import router


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router)  # full paths, no prefix
    return TestClient(app)


def test_buffer_snapshot_degraded() -> None:
    c = _client()
    r = c.get("/api/buffer/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["tip"] is None
    assert body["recent_events"] == []


def test_ws_buffer_sends_snapshot_first() -> None:
    c = _client()
    with c.websocket_connect("/ws/buffer") as ws:
        frame = ws.receive_json()
        assert frame["kind"] == "snapshot"
        assert frame["snapshot"]["degraded"] is True


def test_sse_buffer_opens_with_snapshot() -> None:
    c = _client()
    with c.stream("GET", "/sse/buffer") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                assert payload["kind"] == "snapshot"
                assert payload["snapshot"]["degraded"] is True
                break
