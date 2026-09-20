"""Contract tests for the background-run routes (true parallelism).

These relay onto ``live_app._background_runs`` (a real BackgroundRunManager wired
with a FAKE run_fn here — no live LLM/graph). Guarantees:
  * STANDALONE (no live app) → spawn/list/abort degrade to a typed body, no 500;
  * LIVE → spawn returns a run record immediately; list shows it; abort stops it;
  * instrument_control can never be backgrounded (typed 4xx-style body, ok=False);
  * a run's streamed messages merge into the injected transcript sink, tagged.
"""
from __future__ import annotations

import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.orchestrator import router as orch_router
from mast.core.background_runs import BG_TAG, BackgroundRunManager


def _client(ctx=None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx if ctx is not None else AppContext()
    app.include_router(orch_router, prefix="/api")
    return TestClient(app)


class _Sink:
    def __init__(self):
        self.rows = []
        self._lock = threading.Lock()

    def __call__(self, cid, kind, agent_id, role, text):
        with self._lock:
            self.rows.append((cid, kind, agent_id, role, text))


class _BgApp:
    """Minimal live-app surface the routes need: a lazily-returned manager."""

    def __init__(self, mgr):
        self._mgr = mgr

    def _ensure_background_manager(self):
        return self._mgr


def _live_ctx(app) -> AppContext:
    ctx = AppContext()
    ctx.live_app = app  # type: ignore[attr-defined]
    return ctx


def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ── standalone degradation ───────────────────────────────────────────────────
def test_background_spawn_degrades_standalone():
    c = _client()
    r = c.post("/api/agents/run-task/background",
               json={"instruction": "survey graphene", "agents": ["literature"]})
    assert r.status_code == 200
    b = r.json()
    assert b["degraded"] is True and b["ok"] is False


def test_background_list_degrades_standalone():
    c = _client()
    r = c.get("/api/agents/background-runs")
    assert r.status_code == 200 and r.json()["degraded"] is True


# ── live: spawn / list / abort ───────────────────────────────────────────────
def test_background_spawn_list_abort_live():
    release = threading.Event()

    def run_fn(instruction, thread_id, agents, emit, abort):
        emit("literature", "agent", "searching…")
        release.wait(timeout=5)
        return "done surveying"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=_Sink())
    c = _client(_live_ctx(_BgApp(mgr)))

    r = c.post("/api/agents/run-task/background",
               json={"instruction": "survey graphene STM", "conversation_id": "conv-1",
                     "agents": ["literature"], "title": "石墨烯综述"})
    b = r.json()
    assert b["ok"] is True and b["degraded"] is False
    run_id = b["run"]["run_id"]
    assert b["run"]["thread_id"] == f"bg-{run_id}"
    assert b["run"]["status"] in ("queued", "running")

    # list shows the active run scoped to the conversation
    assert _wait(lambda: c.get("/api/agents/background-runs",
                               params={"conversation_id": "conv-1"}).json()["count"] == 1)
    lst = c.get("/api/agents/background-runs",
                params={"conversation_id": "conv-1", "active_only": True}).json()
    assert lst["runs"][0]["run_id"] == run_id

    # abort it
    ra = c.post(f"/api/agents/run-task/background/{run_id}/abort")
    ab = ra.json()
    assert ab["ok"] is True and ab["aborted"] is True
    release.set()
    assert _wait(lambda: c.get("/api/agents/background-runs").json()["runs"][0]["status"]
                 in ("aborted", "done"))


def test_background_rejects_instrument_control():
    mgr = BackgroundRunManager(run_fn=lambda *a, **k: "x", transcript_sink=_Sink())
    c = _client(_live_ctx(_BgApp(mgr)))
    r = c.post("/api/agents/run-task/background",
               json={"instruction": "scan the surface", "agents": ["instrument_control"]})
    b = r.json()
    # a caller error → ok False, NOT degraded (it reached the manager and was rejected)
    assert b["ok"] is False and b["degraded"] is False
    assert "instrument_control" in (b.get("detail") or "")


def test_background_results_merge_into_transcript():
    sink = _Sink()

    def run_fn(instruction, thread_id, agents, emit, abort):
        emit("literature", "agent", "found 3 relevant papers")
        return "survey complete"

    mgr = BackgroundRunManager(run_fn=run_fn, transcript_sink=sink)
    c = _client(_live_ctx(_BgApp(mgr)))
    r = c.post("/api/agents/run-task/background",
               json={"instruction": "survey", "conversation_id": "conv-x",
                     "agents": ["literature"]})
    run_id = r.json()["run"]["run_id"]
    assert _wait(lambda: mgr.get(run_id)["status"] == "done")

    rows = [row for row in sink.rows if row[0] == "conv-x"]
    assert rows, "no background transcript rows merged"
    assert all(BG_TAG in row[4] for row in rows)
    assert any(row[1] == "message" and "found 3 relevant papers" in row[4] for row in rows)
    assert rows[0][1] == "status" and rows[-1][1] == "status"   # start + end markers


def test_background_empty_instruction_rejected():
    mgr = BackgroundRunManager(run_fn=lambda *a, **k: "x", transcript_sink=_Sink())
    c = _client(_live_ctx(_BgApp(mgr)))
    r = c.post("/api/agents/run-task/background",
               json={"instruction": "  ", "agents": ["literature"]})
    b = r.json()
    assert b["ok"] is False and b["degraded"] is False


# ── auto-background: the run-task bridge spawns on the supervisor's marker ─────
def test_run_task_bridge_spawns_background_on_auto_marker():
    """When the foreground supervisor emits the AUTO_BACKGROUND marker, the
    run-task stream must spawn a DETACHED background run for the named agent and
    surface a status frame — the graph node can't start threads, so the bridge
    does. Drives the REAL _run_task_stream against a fake marker-emitting graph."""
    import json
    import threading
    from langchain_core.messages import AIMessage
    from mast.api.routes.orchestrator import _run_task_stream

    class _MarkerOrch:
        def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
            from langgraph.types import Command
            if isinstance(stream_input, Command):
                return
            yield ((), {"supervisor": {"messages": [
                AIMessage(content="[SUPERVISOR::AUTO_BACKGROUND] literature :: 互不依赖",
                          id="bg-marker-1"),
                AIMessage(content="[SUPERVISOR → instrument_control] 互不依赖",
                          id="disp-1"),
            ]}})

    spawned: list = []

    def _run_fn(instruction, thread_id, agents, emit, abort):
        spawned.append((instruction, tuple(agents)))
        return "survey done"

    mgr = BackgroundRunManager(run_fn=_run_fn, transcript_sink=_Sink())

    class _BridgeApp:
        _orchestrator = _MarkerOrch()
        _orch_abort = threading.Event()
        _orch_running = False
        _orch_interrupts = {"lock": threading.Lock(), "pending": {}, "resolved": {}, "events": {}}
        _conv_store = None
        _agents_api_state = {"lock": threading.Lock(), "holds": {}, "interjects": [], "task": None}

        def _ensure_background_manager(self):
            return mgr

    frames = []
    for line in _run_task_stream(_BridgeApp(), "扫一张图，同时查文献", "t-abg"):
        if line.startswith("data:"):
            frames.append(json.loads(line[len("data:"):].strip()))

    # the bridge spawned a literature background run with the run's instruction
    assert _wait(lambda: any(a == ("literature",) for _i, a in spawned)), spawned
    # and surfaced a status frame announcing the auto-background
    assert any(f.get("kind") == "status" and f.get("agent") == "literature"
               and "后台" in (f.get("text") or "") for f in frames), frames
    # the manager tracks the run
    assert any(r["agents"] == ["literature"] for r in mgr.list_runs())
