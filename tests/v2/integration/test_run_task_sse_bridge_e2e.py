"""The SSE bridge, end to end — the layer the operator actually talks to.

The graph closes the loop when driven with ``app.stream()`` directly (real LLM,
90 super-steps, artifacts on disk). The operator does not drive it that way: the
browser POSTs ``/api/agents/run-task`` and reads the SSE stream. Everything
between those two — bucketing chunks into frames, the live ``_agents_api_state``
bookkeeping, HITL publish/resume, the terminal label, the durable transcript —
is bridge code, and it has been unit-tested only in pieces.

That gap has already cost a night: ``IndexError: list index out of range`` on
the LAST chunk of EVERY normally-finished run . The graph was
fine. The bridge read ``fanned[0]`` on the empty list an END produces, inside a
block guarded by ``if st is not None`` — and no test built ``st``, so the whole
block was dead in CI while it crashed every real run.

So these drive the REAL route with a STUB graph (fast, deterministic, CI-safe)
and a LIVE task dict, and assert the frame sequence the client depends on.

Chunk shape is copied from what the bridge consumes, not from memory:
``orchestrator.stream(..., stream_mode="updates", subgraphs=True)`` yields
``(namespace, {node_name: {"messages": [...]}})`` where namespace is ``()`` for
the top-level supervisor and ``("<agent>:<uuid>",)`` inside an agent subgraph
(``_agent_from_namespace`` splits on ":"), plus ``{"__interrupt__": (...)}``
for a HITL pause.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/integration/test_run_task_sse_bridge_e2e.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import json  # noqa: E402
import threading  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import mast.api.routes.orchestrator as O  # noqa: E402
from mast.api.routes.agents_control import router as control_router  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core import diagnostics as diag  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Stub graphs — chunk shapes copied from the bridge's consumption code
# ════════════════════════════════════════════════════════════════════════════

def _sup(text: str):
    """A top-level (supervisor) chunk: namespace () → _agent_from_namespace None
    → the caller falls back to _normalize_node("supervisor") = "_supervisor"."""
    return ((), {"supervisor": {"messages": [AIMessage(content=text)]}})


def _agent(agent_id: str, text: str):
    """A subgraph chunk: namespace ("<agent>:<uuid>",) → owner = <agent>."""
    return ((f"{agent_id}:abc123",),
            {"agent": {"messages": [AIMessage(content=text)]}})


class _ScriptedGraph:
    """Replays a fixed chunk script. ``on_chunk`` fires after each yield so a
    test can abort mid-stream at a real super-step boundary."""

    def __init__(self, chunks, on_chunk=None):
        self._chunks = list(chunks)
        self._on_chunk = on_chunk
        self.resumed_with = None
        self.stream_calls = 0

    def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
        self.stream_calls += 1
        for i, c in enumerate(self._chunks):
            yield c
            if self._on_chunk is not None:
                self._on_chunk(i)


class _CrashingGraph:
    """Emits one good chunk, then raises — a bridge/agent crash mid-run."""

    def stream(self, *_a, **_kw):
        yield _sup("[SUPERVISOR → instrument_control] 去扫图")
        raise IndexError("list index out of range")


class _Interrupt:
    def __init__(self, value):
        self.value = value
        self.id = "lg-intr-1"


class _HitlGraph:
    """First pass pauses on a DANGEROUS approval; the resume completes."""

    def __init__(self):
        self.resumed_with = None

    def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
        from langgraph.types import Command
        if isinstance(stream_input, Command):
            self.resumed_with = stream_input.resume
            yield _agent("instrument_control", "偏压已设置，扫描完成")
            yield _sup("[SUPERVISOR → __end__] 完成")
            return
        yield _sup("[SUPERVISOR → instrument_control] 需要设置危险偏压")
        yield (("instrument_control:abc123",), {"__interrupt__": (_Interrupt({
            "action_requests": [{"name": "SetBias", "args": {"bias_v": 5.0},
                                 "description": "设置危险偏压"}],
            "review_configs": [{"action_name": "SetBias",
                                "allowed_decisions": ["approve", "reject", "edit"]}],
        }),)})


# ════════════════════════════════════════════════════════════════════════════
# The live app — with the bookkeeping dict that CI has never built
# ════════════════════════════════════════════════════════════════════════════

def _live_app(graph, conv_store=None):
    live = types.SimpleNamespace()
    live._orchestrator = graph
    live._orch_abort = threading.Event()
    live._orch_running = False
    live._orch_interrupts = {"lock": threading.Lock(), "pending": {},
                             "resolved": {}, "events": {}}
    live._conv_store = conv_store
    # THE condition that hid the IndexError: with this dict absent, `st` is None
    # and the whole per-chunk bookkeeping block is skipped in tests while running
    # on every real request.
    live._agents_api_state = {"lock": threading.Lock(), "holds": {},
                              "interjects": [], "task": None}

    def _build_orchestrator(**_kw):
        return graph is not None

    def _build_decision(verdict, skill, base_args, payload_params, reason):
        v = (verdict or "").strip().lower()
        if v == "approve":
            return {"type": "approve"}
        if v == "reject":
            return {"type": "reject", "message": reason or "rejected"}
        return {"type": "edit", "args": dict(base_args or {})}

    live._build_orchestrator = _build_orchestrator
    live._build_decision = _build_decision
    return live


def _client(live, conv_store=None):
    app = FastAPI()
    app.include_router(O.router, prefix="/api")
    app.include_router(control_router, prefix="/api")
    ctx = types.SimpleNamespace(live_app=live)
    if conv_store is not None:
        ctx.conversation_store = conv_store
    app.state.ctx = ctx
    return TestClient(app)


def _frames(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload:
                out.append(json.loads(payload))
    return out


@pytest.fixture(autouse=True)
def _clean_diagnostics():
    diag.clear()
    yield
    diag.clear()


@pytest.fixture()
def conv_store(tmp_path):
    return ConversationStore(tmp_path / "conv.db")


# ════════════════════════════════════════════════════════════════════════════
# 1 — a run that finishes: the frame sequence the client is written against
# ════════════════════════════════════════════════════════════════════════════

_HAPPY_SCRIPT = [
    _sup("[SUPERVISOR → instrument_control] 去扫一张图"),
    _agent("instrument_control", "扫描完成，256×256，质量良好"),
    _sup("[SUPERVISOR → data_processing] 分析这张图"),
    _agent("data_processing", "晶格常数 0.384 nm，缺陷密度 2.1e12 cm^-2"),
    _sup("[SUPERVISOR → paper_writing] 写报告"),
    _agent("paper_writing", "报告已写入 drafts/report.md"),
    _sup("[SUPERVISOR → __end__] 任务完成"),
]


def test_a_finished_run_emits_start_messages_and_a_clean_done(conv_store):
    live = _live_app(_ScriptedGraph(_HAPPY_SCRIPT), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "扫图并写报告"}).text)
    kinds = [f["kind"] for f in frames]

    # ── the contract ──
    assert kinds[0] == "start", kinds
    assert kinds[-1] == "done", kinds
    assert "message" in kinds
    assert "error" not in kinds, [f for f in frames if f["kind"] == "error"]

    start = frames[0]
    assert start["task"] == "扫图并写报告"
    assert start["conversation_id"], "start must carry the durable group conversation id"
    assert start["step_limit"] > 0

    done = frames[-1]
    assert done["aborted"] is False
    assert done["failed"] is False
    assert done["stop_reason"] == ""
    assert done["conversation_id"] == start["conversation_id"]
    assert done["step"] == len(_HAPPY_SCRIPT), "every super-step must be counted"

    # THE #40/#41/#42/#48 regression, at the layer it lived on.
    assert "运行出错" not in c.post(
        "/api/agents/run-task", json={"task": "again"}).text


def test_every_agent_message_is_attributed_to_its_own_subgraph(conv_store):
    """Namespace → agent id. If this broke, the group chat would credit the
    supervisor with the instrument's words."""
    live = _live_app(_ScriptedGraph(_HAPPY_SCRIPT), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "t"}).text)
    msgs = [(f["agent"], f["text"]) for f in frames if f["kind"] == "message"]
    by_agent = dict(msgs)

    assert "扫描完成，256×256，质量良好" == by_agent["instrument_control"]
    assert "晶格常数 0.384 nm，缺陷密度 2.1e12 cm^-2" == by_agent["data_processing"]
    assert "报告已写入 drafts/report.md" == by_agent["paper_writing"]
    # the supervisor's route notes are its own
    assert any(a == "_supervisor" for a, _ in msgs)


def test_the_live_task_slot_records_the_route_and_then_closes(conv_store):
    """``st`` bookkeeping — the block that CI never executed."""
    live = _live_app(_ScriptedGraph(_HAPPY_SCRIPT), conv_store)
    c = _client(live, conv_store)
    c.post("/api/agents/run-task", json={"task": "t"})

    task = live._agents_api_state["task"]
    assert task["active"] is False, "the task slot must close on every exit path"
    kinds = [h.get("kind") for h in task["handoffs"]]
    assert "handoff" in kinds and "end" in kinds, task["handoffs"]
    ends = [h for h in task["handoffs"] if h.get("kind") == "end"]
    assert ends[-1]["targets"] == []
    assert task["active_agents"] == []
    assert live._orch_running is False


# ════════════════════════════════════════════════════════════════════════════
# 2 — a crash: an error frame, an honest label, AND a locatable diagnostic
# ════════════════════════════════════════════════════════════════════════════

def test_a_crash_surfaces_as_an_error_frame_and_a_failed_done(conv_store):
    live = _live_app(_CrashingGraph(), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "扫图"}).text)
    kinds = [f["kind"] for f in frames]

    assert "error" in kinds, kinds
    err = next(f for f in frames if f["kind"] == "error")
    assert "IndexError" in err["message"]
    assert err["degraded"] is True

    # the status frame that precedes the terminal one says WHY
    stops = [f for f in frames if f["kind"] == "status"
             and "运行出错，已停止" in f.get("text", "")]
    assert stops, [f.get("text") for f in frames if f["kind"] == "status"]

    done = frames[-1]
    assert done["kind"] == "done"
    assert done["failed"] is True, "a crash must not masquerade as an operator abort"
    assert done["aborted"] is True      # aborted == "did not complete"
    assert "IndexError" in done["stop_reason"]


def test_a_crash_leaves_a_locatable_diagnostic_with_a_traceback(conv_store):
    """The SSE error frame is gone once the page reloads. 记录 → 诊断 is the only
    place the operator can still find out WHERE it died — and until 2026-07-28
    the log held one line ("IndexError: list index out of range") with no frame,
    no file and no agent. This asserts the traceback actually lands."""
    live = _live_app(_CrashingGraph(), conv_store)
    c = _client(live, conv_store)
    c.post("/api/agents/run-task", json={"task": "扫图"})

    rows = diag.recent(kinds=("run_error",))
    assert rows, f"no run_error diagnostic recorded; got {diag.summary()}"
    row = rows[0]
    assert "IndexError" in row["reason"]
    tb = row.get("traceback")
    assert isinstance(tb, list) and tb, f"traceback must be a LIST of frames: {tb!r}"
    assert any("orchestrator.py" in ln or "test_run_task_sse_bridge" in ln
               or "_ScriptedGraph" in ln or "stream" in ln for ln in tb), tb


# ════════════════════════════════════════════════════════════════════════════
# 3 — an abort: 已中止, NOT an error label (the three-state terminal)
# ════════════════════════════════════════════════════════════════════════════

def test_an_operator_abort_is_labelled_an_abort_not_a_crash(conv_store):
    """``_terminal_label`` has three states on purpose : a crash, an
    unanswered approval and a real abort used to persist identically as 已中止,
    so the operator got blamed for all three. The inverse must hold too — a real
    abort must not be dressed up as 运行出错."""
    live = None

    def _abort_after_first(_i):
        live._orch_abort.set()

    live = _live_app(_ScriptedGraph(_HAPPY_SCRIPT, on_chunk=_abort_after_first),
                     conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "扫图"}).text)

    done = frames[-1]
    assert done["kind"] == "done", "an abort must still converge to a terminal frame"
    assert done["aborted"] is True
    assert done["failed"] is False, "an operator abort is not a failure"
    assert done["stop_reason"] == ""
    assert not any("运行出错" in f.get("text", "") for f in frames)

    # and the durable terminal row says 已中止, which is what a replay shows
    rows = conv_store.messages_for(done["conversation_id"])
    terminal = [r for r in rows if r["kind"] == "done"]
    assert terminal and terminal[-1]["text"] == "已中止", terminal


def test_a_finished_run_persists_the_completed_label(conv_store):
    """The other side of the same three-state contract."""
    live = _live_app(_ScriptedGraph(_HAPPY_SCRIPT), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "t"}).text)
    rows = conv_store.messages_for(frames[-1]["conversation_id"])
    terminal = [r for r in rows if r["kind"] == "done"]
    assert terminal and terminal[-1]["text"] == "完成", terminal


def test_a_crash_persists_a_crash_label_not_an_abort(conv_store):
    """THE assertion that separates the three-state label from the two-state one.

    A replay reads only the persisted terminal row — the error frame is gone
    once the page reloads. With two states a numpy crash, an unanswered approval
    and a real 中止 all persisted as 已中止, so 9 of 11 conversations in the
    2026-07-27 forensics blamed the operator for backend failures. Asserting the
    ABORT row says 已中止 does not test this: both versions produce that. Only
    the crash row tells them apart."""
    live = _live_app(_CrashingGraph(), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "扫图"}).text)
    rows = conv_store.messages_for(frames[-1]["conversation_id"])
    terminal = [r for r in rows if r["kind"] == "done"]
    assert terminal, "a crashed run left no terminal marker to replay"
    label = terminal[-1]["text"]
    assert label.startswith("运行出错，已停止"), (
        f"a crash persisted as {label!r} — a replay cannot tell it from an "
        "operator abort, which is exactly what got the operator blamed ")
    assert "IndexError" in label
    assert label != "已中止"


# ════════════════════════════════════════════════════════════════════════════
# 4 — HITL: interrupt frame → HTTP resolve → the run continues
# ════════════════════════════════════════════════════════════════════════════

def test_hitl_interrupt_resolves_over_http_and_the_run_continues(conv_store):
    """「批准和拒绝都点不了」. The chain is
    SSE interrupt frame → POST .../resolve → Command(resume=…) → stream finishes,
    and it has never been tested across those layers.

    The STREAM is driven off the TestClient portal on purpose: TestClient
    serialises every request through one anyio portal, so a blocked stream plus a
    concurrent resolve POST would deadlock. The resolve still goes over real
    HTTP, which is the link that was broken."""
    graph = _HitlGraph()
    live = _live_app(graph, conv_store)
    c = _client(live, conv_store)

    collected: list[dict] = []
    finished = threading.Event()
    stream = O._run_task_stream(live, "设置 5V 偏压", "task-hitl", "")

    def _consume():
        try:
            for chunk in stream:
                s = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
                for line in s.splitlines():
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload:
                            collected.append(json.loads(payload))
        finally:
            finished.set()

    t = threading.Thread(target=_consume, daemon=True)
    t.start()

    interrupt_id = None
    for _ in range(150):  # ≤15 s
        hit = next((f for f in list(collected) if f["kind"] == "interrupt"), None)
        if hit:
            interrupt_id = hit["interrupt_id"]
            break
        threading.Event().wait(0.1)
    assert interrupt_id, f"no interrupt frame; got {[f['kind'] for f in collected]}"

    intr = next(f for f in collected if f["kind"] == "interrupt")
    assert intr["skill"] == "SetBias"
    assert intr["params"] == {"bias_v": 5.0}
    assert set(intr["allowed_decisions"]) >= {"approve", "reject"}
    assert interrupt_id in live._orch_interrupts["pending"], "resolve has nothing to drain"

    # ── the operator clicks 批准, over real HTTP ──
    r = c.post(
        f"/api/agents/instrument_control/interrupts/{interrupt_id}/resolve",
        json={"decision": "approve"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True, r.json()

    assert finished.wait(timeout=15), "the stream never resumed after approval"
    t.join(timeout=5)

    kinds = [f["kind"] for f in collected]
    assert kinds[-1] == "done", kinds
    assert collected[-1]["aborted"] is False
    # the decision reached the graph, addressed by interrupt id
    assert graph.resumed_with in (
        {"decisions": [{"type": "approve"}]},
        {"lg-intr-1": {"decisions": [{"type": "approve"}]}},
    ), graph.resumed_with
    # and the pending entry is cleaned up (a stale pending blocks the next run)
    assert interrupt_id not in live._orch_interrupts["pending"]


def test_a_rejected_interrupt_also_unblocks_the_stream(conv_store):
    """拒绝 must be as live as 批准 — #21 named both buttons."""
    graph = _HitlGraph()
    live = _live_app(graph, conv_store)
    c = _client(live, conv_store)

    collected: list[dict] = []
    finished = threading.Event()
    stream = O._run_task_stream(live, "设置 5V 偏压", "task-hitl-2", "")

    def _consume():
        try:
            for chunk in stream:
                for line in str(chunk).splitlines():
                    if line.startswith("data:"):
                        p = line[len("data:"):].strip()
                        if p:
                            collected.append(json.loads(p))
        finally:
            finished.set()

    threading.Thread(target=_consume, daemon=True).start()
    interrupt_id = None
    for _ in range(150):
        hit = next((f for f in list(collected) if f["kind"] == "interrupt"), None)
        if hit:
            interrupt_id = hit["interrupt_id"]
            break
        threading.Event().wait(0.1)
    assert interrupt_id

    r = c.post(
        f"/api/agents/instrument_control/interrupts/{interrupt_id}/resolve",
        json={"decision": "reject", "comment": "偏压太高"},
    )
    assert r.status_code == 200 and r.json()["ok"] is True
    assert finished.wait(timeout=15), "a rejection must unblock the stream too"
    assert [f["kind"] for f in collected][-1] == "done"


# ════════════════════════════════════════════════════════════════════════════
# 5 — the durable transcript: what a reconnect replays
# ════════════════════════════════════════════════════════════════════════════

def test_the_run_is_readable_back_from_the_transcript_endpoint(conv_store):
    """A reload must not lose the run. NB: the readable-back endpoint is
    ``GET /api/agents/group-transcript`` — ``/api/agents/{agent_id}/messages``
    reads the CHECKPOINTER for a private chat and knows nothing about a 群聊."""
    live = _live_app(_ScriptedGraph(_HAPPY_SCRIPT), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "扫图并写报告"}).text)
    cid = frames[0]["conversation_id"]

    r = c.get("/api/agents/group-transcript", params={"conversation_id": cid})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    entries = body["entries"]
    assert entries, "the run left no durable transcript"

    kinds = [e["kind"] for e in entries]
    assert kinds[0] == "operator", kinds
    assert entries[0]["text"] == "扫图并写报告"
    assert "message" in kinds
    assert kinds[-1] == "done"

    # per-agent attribution survives the round trip (the per-agent 群聊 feed
    # filters on it)
    agents = {e["agent_id"] for e in entries if e["kind"] == "message"}
    assert {"instrument_control", "data_processing", "paper_writing"} <= agents, agents


def test_the_conversation_is_listed_and_resumable(conv_store):
    live = _live_app(_ScriptedGraph(_HAPPY_SCRIPT), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "扫图"}).text)
    cid = frames[0]["conversation_id"]

    listed = c.get("/api/agents/group-conversations").json()
    assert listed["degraded"] is False
    ids = [x["conversation_id"] for x in listed["conversations"]]
    assert cid in ids, ids

    # resuming the SAME conversation must reuse its thread, not fork a new row
    live._orchestrator = _ScriptedGraph(_HAPPY_SCRIPT)
    again = _frames(c.post("/api/agents/run-task",
                           json={"task": "继续", "conversation_id": cid}).text)
    assert again[0]["conversation_id"] == cid
    assert len(c.get("/api/agents/group-conversations").json()["conversations"]) == 1


# ════════════════════════════════════════════════════════════════════════════
# 5 — the FOURTH terminal state: a branch that dies without handing back
# ════════════════════════════════════════════════════════════════════════════
#
# Dispatch audit 2026-07-28 致命二. The parent graph has exactly ONE edge
# (START→supervisor) and adds each agent as a bare compiled subgraph with no
# outgoing edge, so the return trip depends entirely on the model calling a
# handoff tool. When it doesn't — the model just answers, StallGuard forces a
# bare AIMessage, ModelCallLimit/ToolCallLimit hit with exit_behavior="end" —
# the branch stops where it stands and ``graph.stream()`` runs dry WITHOUT
# raising. Measured on a topology copied from production: the supervisor runs
# once and nothing else happens.
#
# The bridge then reported `{"aborted": false, "failed": false}` and persisted
# 「完成」, because `completed` was initialised True and only four paths could
# flip it — none of them about the hand-back. A dead branch and a finished run
# were pixel-identical for the operator.

_NO_HANDBACK_SCRIPT = [
    _sup("[SUPERVISOR → instrument_control] 去扫一张图"),
    # The agent answers in prose and never calls handoff_to_supervisor. In
    # production the subgraph ends here and the parent has nowhere to go.
    _agent("instrument_control", "我需要先确认针尖状态，请问要继续吗？"),
]

_STALL_GUARD_SCRIPT = [
    _sup("[SUPERVISOR → instrument_control] 去扫一张图"),
    _agent("instrument_control", "⛔ 本回合被空转保护终止"),
]


@pytest.mark.parametrize("script,label", [
    (_NO_HANDBACK_SCRIPT, "model answered without handing back"),
    (_STALL_GUARD_SCRIPT, "StallGuard forced a bare AIMessage"),
])
def test_a_branch_that_never_hands_back_is_reported_as_failed(script, label,
                                                              conv_store):
    live = _live_app(_ScriptedGraph(script), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "扫图"}).text)

    done = frames[-1]
    assert done["kind"] == "done"
    assert done["failed"] is True, f"{label}: silent death reported as success"
    assert done["aborted"] is True          # aborted == "did not complete"
    assert done["stop_reason"], "a failure with no reason is not diagnosable"
    assert "没有交回控制权" in done["stop_reason"]

    # and it is visible in the stream, not only in the terminal frame
    assert any(f["kind"] == "status" and "没有交回控制权" in f.get("text", "")
               for f in frames), [f.get("text") for f in frames]


def test_a_silent_death_persists_a_failure_label_not_完成(conv_store):
    """A replay reads only the persisted terminal row."""
    live = _live_app(_ScriptedGraph(_NO_HANDBACK_SCRIPT), conv_store)
    c = _client(live, conv_store)
    frames = _frames(c.post("/api/agents/run-task", json={"task": "扫图"}).text)
    rows = conv_store.messages_for(frames[-1]["conversation_id"])
    terminal = [r for r in rows if r["kind"] == "done"]
    assert terminal
    assert terminal[-1]["text"] != "完成", (
        "the branch died and the transcript says the task finished")
    assert terminal[-1]["text"].startswith("运行出错，已停止")


def test_a_silent_death_leaves_a_diagnostic(conv_store):
    live = _live_app(_ScriptedGraph(_NO_HANDBACK_SCRIPT), conv_store)
    c = _client(live, conv_store)
    c.post("/api/agents/run-task", json={"task": "扫图"})
    rows = diag.recent(kinds=("fail_silent_end",))
    assert rows, f"no fail_silent_end diagnostic; got {diag.summary()}"


def test_a_degenerate_end_still_counts_as_a_real_end(conv_store):
    """The supervisor's loop-guard / budget / routing-error branches end WITHOUT
    a `[SUPERVISOR → __end__]` note, but every one of them writes
    active_agent="__end__". They are real terminations and must not be flagged."""
    script = [
        _sup("[SUPERVISOR → instrument_control] 去扫一张图"),
        _agent("instrument_control", "[HANDOFF] 交回"),
        ((), {"supervisor": {
            "messages": [AIMessage(content="[SUPERVISOR] Loop guard tripped "
                                           "(visit_count exceeded).")],
            "active_agent": "__end__"}}),
    ]
    live = _live_app(_ScriptedGraph(script), conv_store)
    c = _client(live, conv_store)
    done = _frames(c.post("/api/agents/run-task", json={"task": "t"}).text)[-1]
    assert done["failed"] is False, done.get("stop_reason")
    assert done["aborted"] is False


def test_an_operator_abort_is_not_relabelled_as_a_silent_death(conv_store):
    """The guard must not steal the abort's label: an aborted run has no END
    record either, and blaming the agents for the operator's Stop would undo
    the three-state terminal ."""
    live = None

    def _abort_after_first(_i):
        live._orch_abort.set()

    live = _live_app(_ScriptedGraph(_HAPPY_SCRIPT, on_chunk=_abort_after_first),
                     conv_store)
    c = _client(live, conv_store)
    done = _frames(c.post("/api/agents/run-task", json={"task": "扫图"}).text)[-1]
    assert done["aborted"] is True
    assert done["failed"] is False
    assert done["stop_reason"] == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
