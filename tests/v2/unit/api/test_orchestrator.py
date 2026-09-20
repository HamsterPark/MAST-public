"""Contract tests for the multi-agent orchestrator run-task routes.

The router is not mounted in mast.api.app yet (integration wires that), so
each test builds a throwaway FastAPI app and includes the router under /api.

Guarantees asserted:
  * STANDALONE (no live app): POST /agents/run-task streams a SINGLE degraded
    error frame then done — never a 500, never a hang. The abort handler returns
    a typed degraded body.
  * NO ORCHESTRATOR (live app but no built graph + builder fails): same single
    degraded frame + done.
  * LIVE (a fake app exposing ``_orchestrator`` / ``_orch_abort`` /
    ``_orch_interrupts``): the stream buckets per-agent messages, surfaces an
    ``interrupt`` frame, BLOCKS until the EXISTING agents_control resolve endpoint
    drains the live store, then resumes to ``done``.
  * ABORT: POST /agents/run-task/abort sets the live abort Event + wakes any
    interrupt waiter.
"""

from __future__ import annotations

import json
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.orchestrator import router as orch_router
from mast.api.routes.agents_control import router as control_router


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx if ctx is not None else AppContext()
    app.include_router(orch_router, prefix="/api")
    app.include_router(control_router, prefix="/api")  # the existing resolve relay
    return TestClient(app)


def _frames(resp) -> list[dict]:
    out = []
    for line in resp.iter_lines():
        if line.startswith("data:"):
            out.append(json.loads(line[len("data:"):].strip()))
    return out


# ── fakes that mimic the live core's orchestrator surface ────────────────────
class _AIMessage:
    def __init__(self, content="", tool_calls=None, mid=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.id = mid


class _Interrupt:
    def __init__(self, value):
        self.value = value


class _FakeOrchestrator:
    """Streams a scripted sequence of (namespace, chunk) tuples. If ``with_hitl``,
    the FIRST stream emits an ``__interrupt__`` and stops; resuming via
    Command(resume=…) streams the completion."""

    def __init__(self, with_hitl=False):
        self.with_hitl = with_hitl
        self.resumed_with = None

    def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
        from langgraph.types import Command
        if isinstance(stream_input, Command):
            self.resumed_with = stream_input.resume
            yield (("instrument_control:abc",),
                   {"agent": {"messages": [_AIMessage("已设置偏压并完成扫描")]}})
            yield ((), {"supervisor": {
                "messages": [_AIMessage("[SUPERVISOR → __end__] 任务完成")],
                "active_agent": "__end__"}})
            return
        # supervisor routes
        yield ((), {"supervisor": {"messages": [_AIMessage("路由到 instrument_control")]}})
        if self.with_hitl:
            req = {
                "action_requests": [
                    {"name": "SetBias", "args": {"bias_v": 5.0},
                     "description": "设置危险偏压"}
                ],
                "review_configs": [
                    {"action_name": "SetBias",
                     "allowed_decisions": ["approve", "reject", "edit"]}
                ],
            }
            yield (("instrument_control:abc",),
                   {"__interrupt__": (_Interrupt(req),)})
            return
        yield (("instrument_control:abc",),
               {"agent": {"messages": [_AIMessage("扫描完成，结果良好")]}})
        # The supervisor's END note. Not decoration: since 2026-07-28 the bridge
        # treats an exhausted stream with no [SUPERVISOR → __end__] (and no
        # active_agent="__end__") as a fail-silent branch death and reports
        # failed=True — because in production that is exactly what it is.
        yield ((), {"supervisor": {
            "messages": [_AIMessage("[SUPERVISOR → __end__] 任务完成")],
            "active_agent": "__end__"}})


class _FakeLiveApp:
    def __init__(self, orchestrator=None, build_ok=False, conv_store=None):
        self._orchestrator = orchestrator
        self._build_ok = build_ok
        self._orch_abort = threading.Event()
        self._orch_running = False
        self._orch_interrupts = {
            "lock": threading.Lock(),
            "pending": {},
            "resolved": {},
            "events": {},
        }
        # Durable group transcript store (None = persistence unwired, degrade-safe).
        self._conv_store = conv_store
        self._agents_api_state = {
            "lock": threading.Lock(), "holds": {}, "interjects": [], "task": None,
        }

    def _build_orchestrator(self, **kwargs):
        if self._build_ok:
            self._orchestrator = _FakeOrchestrator()
        return self._orchestrator is not None

    # the resolve relay (agents_control) calls this to translate a verdict.
    @staticmethod
    def _build_decision(verdict, skill, base_args, payload_params, reason):
        v = (verdict or "").strip().lower()
        if v == "approve":
            return {"type": "approve"}
        if v == "reject":
            return {"type": "reject", "message": reason or "rejected"}
        if v == "edit":
            merged = dict(base_args or {})
            if isinstance(payload_params, dict):
                merged.update(payload_params)
            return {"type": "edit", "edited_action": {"name": skill, "args": merged}}
        return {"type": "reject", "message": f"unknown {verdict}"}


def _live_ctx(app: _FakeLiveApp) -> AppContext:
    ctx = AppContext()
    ctx.live_app = app  # type: ignore[attr-defined]
    return ctx


# ── STANDALONE degradation ───────────────────────────────────────────────────
def test_run_task_degrades_standalone() -> None:
    c = _client()
    with c.stream("POST", "/api/agents/run-task", json={"task": "scan it"}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        frames = _frames(r)
    kinds = [f["kind"] for f in frames]
    assert "error" in kinds
    assert frames[0]["degraded"] is True
    assert kinds[-1] == "done"


def test_run_task_empty_task_degrades() -> None:
    c = _client()
    with c.stream("POST", "/api/agents/run-task", json={"task": "   "}) as r:
        frames = _frames(r)
    assert frames[0]["kind"] == "error"
    assert frames[-1]["kind"] == "done"


def test_run_task_no_orchestrator_degrades() -> None:
    app = _FakeLiveApp(orchestrator=None, build_ok=False)
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "go"}) as r:
        frames = _frames(r)
    kinds = [f["kind"] for f in frames]
    assert "error" in kinds
    assert frames[0]["degraded"] is True
    assert kinds[-1] == "done"


def test_abort_degrades_standalone() -> None:
    c = _client()
    r = c.post("/api/agents/run-task/abort")
    assert r.status_code == 200
    b = r.json()
    assert b["degraded"] is True
    assert b["ok"] is False


# ── LIVE happy path ──────────────────────────────────────────────────────────
def test_run_task_streams_messages() -> None:
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(with_hitl=False))
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "do a scan"}) as r:
        assert r.status_code == 200
        frames = _frames(r)
    kinds = [f["kind"] for f in frames]
    assert kinds[0] == "start"
    assert "status" in kinds
    assert "message" in kinds
    assert kinds[-1] == "done"
    assert frames[-1]["aborted"] is False
    # per-agent bucketing
    msgs = [f for f in frames if f["kind"] == "message"]
    assert any(f["agent"] == "_supervisor" for f in msgs)
    assert any(f["agent"] == "instrument_control" for f in msgs)
    # busy flag cleared
    assert app._orch_running is False


def test_run_task_builds_orchestrator_when_absent() -> None:
    app = _FakeLiveApp(orchestrator=None, build_ok=True)
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "build then run"}) as r:
        frames = _frames(r)
    kinds = [f["kind"] for f in frames]
    assert "start" in kinds
    assert kinds[-1] == "done"
    assert app._orchestrator is not None


# ── LIVE HITL interrupt → resolve via the EXISTING endpoint → resume ─────────
def test_run_task_interrupt_then_resolve_resumes() -> None:
    """Drive the route's SSE generator DIRECTLY (not via TestClient) so the
    blocked stream and the resolve call run on separate threads — Starlette's
    TestClient serialises requests through one anyio portal, which would deadlock
    a blocked stream against a concurrent resolve POST. The generator IS the
    real handler; we still resolve through the EXISTING agents_control router."""
    from mast.api.routes.orchestrator import _run_task_stream
    from mast.api.routes.agents_control import resolve_interrupt
    from mast.api.schemas_agents_control import ResolveInterruptRequest

    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(with_hitl=True))
    ctx = _live_ctx(app)

    class _Req:
        def __init__(self):
            self.app = type("A", (), {})()
            self.app.state = type("S", (), {})()
            self.app.state.ctx = ctx

    # the SYNC SSE generator that backs run_task's StreamingResponse — driving it
    # directly avoids the TestClient single-portal deadlock (blocked stream vs.
    # concurrent resolve POST).
    body_iter = _run_task_stream(app, "dangerous scan", "task1", "")

    collected: list[dict] = []
    done = threading.Event()

    def _consume():
        for chunk in body_iter:
            s = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            for line in s.splitlines():
                if line.startswith("data:"):
                    collected.append(json.loads(line[len("data:"):].strip()))
        done.set()

    t = threading.Thread(target=_consume, daemon=True)
    t.start()

    # wait for the interrupt frame to surface (the generator blocks after it)
    interrupt_id = None
    for _ in range(100):  # up to ~10s
        for f in list(collected):
            if f["kind"] == "interrupt":
                interrupt_id = f["interrupt_id"]
                break
        if interrupt_id:
            break
        threading.Event().wait(0.1)
    assert interrupt_id, f"no interrupt frame; got {[f['kind'] for f in collected]}"

    intr = next(f for f in collected if f["kind"] == "interrupt")
    assert intr["skill"] == "SetBias"
    assert intr["params"] == {"bias_v": 5.0}
    assert "approve" in intr["allowed_decisions"]
    # pending entry is live in the shared store (what resolve drains)
    assert interrupt_id in app._orch_interrupts["pending"]

    # resolve via the EXISTING agents_control relay (we do NOT reimplement it)
    out = resolve_interrupt(
        "instrument_control", interrupt_id,
        ResolveInterruptRequest(decision="approve"), _Req(),
    )
    assert out.ok is True

    assert done.wait(timeout=10), "stream did not finish after resolve"
    t.join(timeout=5)

    kinds = [f["kind"] for f in collected]
    assert "interrupt" in kinds
    assert kinds[-1] == "done"
    assert collected[-1]["aborted"] is False
    # the resume value reached the orchestrator (approve → {"decisions":[...]})
    assert app._orchestrator.resumed_with == {"decisions": [{"type": "approve"}]}
    # store cleaned up
    assert interrupt_id not in app._orch_interrupts["pending"]


# ── DURABLE GROUP PERSISTENCE (the regression: 群聊 退化成 demo) ──────────────
def _conv_store(tmp_path):
    from mast.chat.store import ConversationStore
    return ConversationStore(tmp_path / "exp.sqlite")


def test_run_task_persists_group_transcript(tmp_path) -> None:
    """A 群聊 run must (1) create a durable group conversation, (2) return its id
    in the start frame, (3) persist the operator turn + every per-agent message +
    the terminal entry — so the conversation survives a tab switch / reload. This
    is the exact gap that made the multi-agent conversation an ephemeral demo."""
    store = _conv_store(tmp_path)
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(with_hitl=False), conv_store=store)
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "do a scan"}) as r:
        frames = _frames(r)

    start = next(f for f in frames if f["kind"] == "start")
    cid = start["conversation_id"]
    assert cid, "start frame must carry the durable group conversation id"

    # a durable group row exists
    groups = store.list(kind="group")
    assert any(g["conversation_id"] == cid for g in groups)

    # the transcript was persisted, in order, with the operator turn + messages
    entries = store.messages_for(cid)
    kinds = [e["kind"] for e in entries]
    assert kinds[0] == "operator"
    assert entries[0]["text"] == "do a scan"
    assert "message" in kinds
    assert kinds[-1] == "done"
    # per-agent attribution survives (so the per-agent bridge can read it)
    agents = {e["agent_id"] for e in entries if e["kind"] == "message"}
    assert "instrument_control" in agents


def test_run_task_transcript_endpoint_replays(tmp_path) -> None:
    """GET /run-task/transcript returns the durable entries the TS client replays
    on reconnect (what makes a tab switch / reload non-destructive)."""
    store = _conv_store(tmp_path)
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=store)
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "scan Au(111)"}) as r:
        frames = _frames(r)
    cid = next(f for f in frames if f["kind"] == "start")["conversation_id"]

    resp = c.get("/api/agents/group-transcript", params={"conversation_id": cid})
    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] is False
    assert body["conversation_id"] == cid
    assert body["count"] >= 2
    assert body["entries"][0]["kind"] == "operator"
    assert body["entries"][0]["text"] == "scan Au(111)"


def test_run_task_transcript_degrades_without_store() -> None:
    """No wired store ⇒ transcript GET degrades to empty, never 500."""
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=None)
    c = _client(_live_ctx(app))
    resp = c.get("/api/agents/group-transcript", params={"conversation_id": "x"})
    assert resp.status_code == 200
    assert resp.json()["degraded"] is True


def test_run_task_conversations_endpoint_lists_groups(tmp_path) -> None:
    store = _conv_store(tmp_path)
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=store)
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "first run"}) as r:
        _frames(r)
    resp = c.get("/api/agents/group-conversations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] is False
    assert body["count"] >= 1
    assert body["conversations"][0]["title"]  # title seeded from the instruction


def test_group_endpoints_not_shadowed_by_agent_param_route(tmp_path) -> None:
    """REGRESSION: /agents/group-transcript|group-conversations must NOT be
    captured by the path-param route /agents/{agent_id}/... which is registered
    FIRST in the real app. The throwaway app here mounts the agents router BEFORE
    the orchestrator router (mirroring app.py order) — the literal group-* paths
    must still resolve to the orchestrator handlers (the bug was /run-task/
    conversations binding to list_conversations(agent_id='run-task'))."""
    from mast.api.routes.agents import router as agents_router

    store = _conv_store(tmp_path)
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=store)
    ctx = _live_ctx(app)
    ctx.conversation_store = store  # type: ignore[attr-defined]
    fa = FastAPI()
    fa.state.ctx = ctx
    fa.include_router(agents_router, prefix="/api")  # registered FIRST (as in app.py)
    fa.include_router(orch_router, prefix="/api")
    c = TestClient(fa)
    with c.stream("POST", "/api/agents/run-task", json={"task": "shadow check"}) as r:
        cid = next(f for f in _frames(r) if f["kind"] == "start")["conversation_id"]

    # group-conversations must reach the orchestrator handler (group shape +
    # the real group row), NOT a degraded-but-false-healthy empty list.
    conv = c.get("/api/agents/group-conversations").json()
    assert conv["degraded"] is False
    assert any(g["conversation_id"] == cid for g in conv["conversations"])
    assert "active_conversation_id" in conv  # GroupConversationsResponse shape

    # group-transcript must reach the orchestrator handler too.
    tr = c.get("/api/agents/group-transcript", params={"conversation_id": cid}).json()
    assert tr["degraded"] is False
    assert tr["conversation_id"] == cid and tr["count"] >= 1


def test_run_task_refuses_concurrent_stream_on_active_conversation(tmp_path) -> None:
    """Concurrency guard: a second run-task on a group conversation a live run is
    already streaming into must degrade with a busy error frame, NOT open a second
    stream on the same checkpointed thread."""
    store = _conv_store(tmp_path)
    cid = store.create("_supervisor", kind="group", thread_id="agents-live")["conversation_id"]
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=store)
    # simulate a live run already streaming into this conversation
    app._agents_api_state["task"] = {"active": True, "conversation_id": cid}
    c = _client(_live_ctx(app))
    with c.stream(
        "POST", "/api/agents/run-task",
        json={"task": "second stream", "conversation_id": cid},
    ) as r:
        frames = _frames(r)
    kinds = [f["kind"] for f in frames]
    assert "error" in kinds
    assert frames[0]["degraded"] is True
    assert kinds[-1] == "done"
    # no new transcript was appended (the guard returned before streaming)
    assert store.messages_for(cid) == []


def test_run_task_abort_handover_waits_then_proceeds(tmp_path, monkeypatch) -> None:
    """Feedback 2026-07-10 #41: after 中止, the old stream may stay `active` until
    its current super-step ends. A new run on the same conversation must WAIT for
    the aborting stream to wind down (instead of bouncing with the busy error)
    and then proceed normally."""
    import mast.api.routes.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "_ABORT_HANDOVER_GRACE_S", 5.0)
    store = _conv_store(tmp_path)
    cid = store.create("_supervisor", kind="group",
                       thread_id="agents-live")["conversation_id"]
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=store)
    # simulate: operator aborted, but the old stream is still winding down
    app._orch_abort.set()
    app._agents_api_state["task"] = {"active": True, "conversation_id": cid}

    def _wind_down_soon():
        import time as _t
        _t.sleep(0.6)
        app._agents_api_state["task"]["active"] = False

    t = threading.Thread(target=_wind_down_soon, daemon=True)
    t.start()
    c = _client(_live_ctx(app))
    with c.stream(
        "POST", "/api/agents/run-task",
        json={"task": "continue after abort", "conversation_id": cid},
    ) as r:
        frames = _frames(r)
    t.join()
    kinds = [f["kind"] for f in frames]
    # the run proceeded: start frame + agent messages + clean done, no busy error
    assert "start" in kinds and "message" in kinds
    assert not any(f.get("degraded") for f in frames if f["kind"] == "error")
    assert frames[-1]["kind"] == "done" and frames[-1]["aborted"] is False


def test_run_task_abort_handover_times_out_with_guidance(tmp_path, monkeypatch) -> None:
    """If the aborting old stream does NOT wind down inside the grace window
    (e.g. stuck in a long hardware wait), the new run degrades with an
    actionable '中止收尾' message — and re-signals abort — instead of the
    plain busy error."""
    import mast.api.routes.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "_ABORT_HANDOVER_GRACE_S", 0.5)
    store = _conv_store(tmp_path)
    cid = store.create("_supervisor", kind="group",
                       thread_id="agents-live")["conversation_id"]
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=store)
    app._orch_abort.set()
    app._agents_api_state["task"] = {"active": True, "conversation_id": cid}
    c = _client(_live_ctx(app))
    with c.stream(
        "POST", "/api/agents/run-task",
        json={"task": "continue after abort", "conversation_id": cid},
    ) as r:
        frames = _frames(r)
    err = next(f for f in frames if f["kind"] == "error")
    assert "中止收尾" in err["message"]
    assert frames[-1]["kind"] == "done"
    assert app._orch_abort.is_set()  # abort re-signalled for the stuck stream


def test_run_task_resumes_group_conversation_appends(tmp_path) -> None:
    """Passing an existing group conversation_id reuses its thread AND appends to
    the SAME durable transcript (continuity across turns, not a fresh demo each
    time)."""
    store = _conv_store(tmp_path)
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=store)
    c = _client(_live_ctx(app))
    # first turn → creates the group row
    with c.stream("POST", "/api/agents/run-task", json={"task": "turn one"}) as r:
        frames = _frames(r)
    cid = next(f for f in frames if f["kind"] == "start")["conversation_id"]
    first_len = len(store.messages_for(cid))
    # second turn → resume the SAME conversation
    with c.stream(
        "POST", "/api/agents/run-task",
        json={"task": "turn two", "conversation_id": cid},
    ) as r:
        frames2 = _frames(r)
    assert next(f for f in frames2 if f["kind"] == "start")["conversation_id"] == cid
    # still exactly one group row (resumed, not duplicated)
    assert sum(1 for g in store.list(kind="group") if g["conversation_id"] == cid) == 1
    # transcript grew (the second turn appended)
    grown = store.messages_for(cid)
    assert len(grown) > first_len
    assert any(e["text"] == "turn two" for e in grown if e["kind"] == "operator")


def test_run_task_tags_group_with_current_experiment(tmp_path, monkeypatch) -> None:
    """A new 群聊 is tagged with the experiment it was born in (provenance), and
    that tag surfaces in the group-conversations list — but the chat is never
    scoped/filtered by it (it still lists regardless of the current experiment)."""
    import mast.api.routes.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "_current_experiment_id", lambda: "exp-123")
    store = _conv_store(tmp_path)
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator(), conv_store=store)
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "tagged run"}) as r:
        cid = next(f for f in _frames(r) if f["kind"] == "start")["conversation_id"]
    assert store.get(cid)["experiment_id"] == "exp-123"
    conv = c.get("/api/agents/group-conversations").json()
    assert any(g["conversation_id"] == cid and g["experiment_id"] == "exp-123"
               for g in conv["conversations"])


def test_run_task_surfaces_step_budget() -> None:
    """The super-step budget is surfaced so the UI can always show 步数 N/限:
    start carries step_limit, every message carries step + step_limit, done too.
    Asserts against the live module default rather than a hardcoded number
    (the 150 literal went stale when 4.6.1 raised the default to 500)."""
    from mast.api.routes.orchestrator import _RECURSION_LIMIT
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator())
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "go"}) as r:
        frames = _frames(r)
    start = next(f for f in frames if f["kind"] == "start")
    assert start["step_limit"] == _RECURSION_LIMIT
    msgs = [f for f in frames if f["kind"] == "message"]
    assert msgs and all(
        "step" in f and f["step_limit"] == _RECURSION_LIMIT for f in msgs)
    assert msgs[-1]["step"] >= 1
    assert frames[-1]["kind"] == "done" and frames[-1]["step_limit"] == _RECURSION_LIMIT


class _RecursionOrchestrator:
    """Streams one chunk then raises a recursion-limit error (an agent looping on
    a precondition/safety failure exhausts the limit)."""

    def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
        yield ((), {"supervisor": {"messages": [_AIMessage("routing")]}})
        raise RuntimeError(
            "Recursion limit of 150 reached without hitting a stop condition")


def test_run_task_recursion_limit_surfaces_friendly_frame() -> None:
    """A recursion-limit hit must surface as a READABLE degraded error frame, not
    a raw 'GraphRecursionError' the UI renders under '加载失败'."""
    app = _FakeLiveApp(orchestrator=_RecursionOrchestrator())
    c = _client(_live_ctx(app))
    with c.stream("POST", "/api/agents/run-task", json={"task": "loop forever"}) as r:
        frames = _frames(r)
    errs = [f for f in frames if f["kind"] == "error"]
    assert errs, f"expected an error frame; got {[f['kind'] for f in frames]}"
    assert errs[0]["degraded"] is True
    assert "步数达到上限" in errs[0]["message"]  # friendly, operator-actionable
    assert "RuntimeError" not in errs[0]["message"]
    assert frames[-1]["kind"] == "done"
    assert frames[-1]["aborted"] is True


# ── LIVE abort relay ─────────────────────────────────────────────────────────
def test_abort_sets_event_live() -> None:
    app = _FakeLiveApp(orchestrator=_FakeOrchestrator())
    # seed a pending interrupt to prove abort wakes its waiter event
    ev = threading.Event()
    app._orch_interrupts["pending"]["x"] = {"skill": "S"}
    app._orch_interrupts["events"]["x"] = ev
    c = _client(_live_ctx(app))
    r = c.post("/api/agents/run-task/abort")
    assert r.status_code == 200
    b = r.json()
    assert b["ok"] is True
    assert "orchestrator" in b["aborted"]
    assert b["degraded"] is False
    assert app._orch_abort.is_set()
    assert ev.is_set()
