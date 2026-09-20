"""Private chat's approval gate, end to end — the surface operators actually use.

The pieces all existed and were not connected: ``ConversationEngine`` had a
resume loop and a ``hitl_resolver`` hook that no caller ever passed; the modal
polled ``GET /api/agents/{id}/interrupts``; ``POST .../resolve`` translated a
verdict and woke a worker. A DANGEROUS skill in private chat therefore printed
"当前入口未接审批处理器" and stopped with the graph parked in its checkpoint,
while the panel it told the operator to use reported "当前无待处理中断" — and
every later message hit the same gate and got the same non-answer.

These drive the whole chain the way the browser does:

    chat turn (blocked on the gate)
      → GET /interrupts shows the card
      → POST .../resolve over real HTTP
      → Command(resume=…) addressed by interrupt id
      → the SAME conversation finishes and its history holds the result

The turn is consumed on a plain thread, NOT through TestClient: TestClient
serialises every request through one anyio portal, so a blocked stream plus a
concurrent resolve POST deadlocks in the harness (not in the product). The
resolve still goes over real HTTP, which is the link that was broken.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/integration/test_private_chat_hitl_e2e.py -q
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

import threading  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from mast.api.routes.agents_control import router as control_router  # noqa: E402
from mast.api.routes.agents_topology import router as topology_router  # noqa: E402
from mast.api.routes.chat_stream import _make_hitl_resolver  # noqa: E402
from mast.chat.engine import ConversationEngine  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Stubs — shapes copied from what the engine consumes
# ════════════════════════════════════════════════════════════════════════════

class _Interrupt:
    def __init__(self, value, id_="lg-intr-1"):
        self.value = value
        self.id = id_


_HITL_REQUEST = {
    "action_requests": [{"name": "SetBias", "args": {"bias_v": 5.0},
                         "description": "设置危险偏压"}],
    "review_configs": [{"action_name": "SetBias",
                        "allowed_decisions": ["approve", "reject", "edit"]}],
}


class _GatedGraph:
    """Pauses on a HITL interrupt, then finishes once resumed.

    ``get_state`` reports no pending task interrupts, so the engine's
    stale-approval cleanup correctly leaves this thread alone.
    """

    def __init__(self):
        self.resumed_with = None
        self._messages = [AIMessage(content="", tool_calls=[
            {"id": "tc1", "name": "SetBias", "args": {"bias_v": 5.0}}])]

    def stream(self, stream_input, config=None, stream_mode=None, subgraphs=None):
        resume = getattr(stream_input, "resume", None)
        if resume is not None:
            self.resumed_with = resume
            self._messages = [AIMessage(content="偏压已设置为 5 V。")]
            yield ((), {"model": {"messages": list(self._messages)}})
            return
        yield ((), {"model": {"messages": list(self._messages)}})
        yield (("instrument_control:abc123",),
               {"__interrupt__": (_Interrupt(_HITL_REQUEST),)})

    def get_state(self, cfg):
        return types.SimpleNamespace(values={"messages": list(self._messages)},
                                     next=(), tasks=())


def _store_dict():
    return {"lock": threading.Lock(), "pending": {}, "resolved": {}, "events": {}}


def _ctx(interrupts_store, conv_store):
    """The app context the routes read: live app + the interrupts snapshot."""
    live = types.SimpleNamespace(
        _orch_interrupts=interrupts_store,
        _agents_api_state={"lock": threading.RLock(), "interrupts": {}},
        agents_interrupts=lambda agent_id="__all__": [
            p for p in list(interrupts_store["pending"].values())
            if agent_id in ("__all__", "_supervisor") or p.get("agent_id") == agent_id
        ],
    )
    return types.SimpleNamespace(live_app=live, app=live,
                                 conversation_store=conv_store,
                                 agents_interrupts=live.agents_interrupts)


def _client(ctx):
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(control_router, prefix="/api")
    app.include_router(topology_router, prefix="/api")
    return TestClient(app)


@pytest.fixture
def wired(tmp_path):
    """A private conversation whose next turn will hit the gate."""
    conv_store = ConversationStore(str(tmp_path / "conv.db"))
    graph = _GatedGraph()
    engine = ConversationEngine(graph_factory=lambda aid: graph,
                                checkpointer=None, store=conv_store)
    interrupts = _store_dict()
    ctx = _ctx(interrupts, conv_store)
    cid = conv_store.create("instrument_control", kind="private")["conversation_id"]
    return types.SimpleNamespace(engine=engine, graph=graph, store=conv_store,
                                 interrupts=interrupts, ctx=ctx,
                                 client=_client(ctx), cid=cid)


def _run_turn(w, text="设置 5V 偏压", abort=None):
    """Drive one chat turn on a background thread (see the module docstring)."""
    snapshots: list = []
    done = threading.Event()
    resolver = _make_hitl_resolver(w.ctx, "instrument_control", w.cid, abort)
    assert resolver is not None, "a live store must produce a resolver"

    def _consume():
        try:
            for snap in w.engine.stream_turn(w.cid, text, abort=abort,
                                             hitl_resolver=resolver):
                snapshots.append(snap)
        finally:
            done.set()

    t = threading.Thread(target=_consume, daemon=True)
    t.start()
    return snapshots, done, t


def _wait_for_card(w, timeout=15.0):
    for _ in range(int(timeout * 10)):
        r = w.client.get("/api/agents/instrument_control/interrupts")
        assert r.status_code == 200, r.text
        body = r.json()
        if body.get("interrupts"):
            return body["interrupts"][0]
        threading.Event().wait(0.1)
    raise AssertionError("the approval never reached GET /interrupts")


def _text(snapshots) -> str:
    return "\n".join(m.get("content", "")
                     for snap in snapshots for m in (snap or []))


# ════════════════════════════════════════════════════════════════════════════
# The chain
# ════════════════════════════════════════════════════════════════════════════

def test_approval_reaches_the_panel_and_carries_the_conversation_through(wired):
    snapshots, done, t = _run_turn(wired)

    card = _wait_for_card(wired)
    assert card["skill"] == "SetBias"
    assert card["params"] == {"bias_v": 5.0}
    assert card["agent_id"] == "instrument_control", \
        "the modal filters by agent id — a mismatch means an invisible card"
    assert set(card["allowed_decisions"]) >= {"approve", "reject"}
    # The id the panel shows is the id the resolve endpoint drains.
    assert card["event_id"] in wired.interrupts["pending"]

    r = wired.client.post(
        f"/api/agents/instrument_control/interrupts/{card['event_id']}/resolve",
        json={"decision": "approve"})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True and r.json()["applied"] is True

    assert done.wait(timeout=15), "the turn never resumed after approval"
    t.join(timeout=5)

    # Addressed BY ID — a bare value is broadcast to every pending interrupt.
    assert wired.graph.resumed_with == {
        "lg-intr-1": {"decisions": [{"type": "approve"}]}}, wired.graph.resumed_with
    # The result landed in THIS conversation.
    assert "偏压已设置" in _text(snapshots[-1:])
    # No stale card left behind for the operator to click at nothing.
    assert not wired.interrupts["pending"]


def test_reject_uses_the_same_translation_and_leaves_the_chat_usable(wired):
    snapshots, done, t = _run_turn(wired)
    card = _wait_for_card(wired)

    r = wired.client.post(
        f"/api/agents/instrument_control/interrupts/{card['event_id']}/resolve",
        json={"decision": "reject", "comment": "偏压太高"})
    assert r.status_code == 200 and r.json()["decision_type"] == "reject"

    assert done.wait(timeout=15)
    t.join(timeout=5)
    resume = wired.graph.resumed_with["lg-intr-1"]["decisions"][0]
    assert resume["type"] == "reject"
    assert "偏压太高" in resume["message"], "the operator's reason must reach the graph"
    assert not wired.interrupts["pending"]


def test_edit_rewrites_the_arguments_before_the_graph_sees_them(wired):
    snapshots, done, t = _run_turn(wired)
    card = _wait_for_card(wired)

    r = wired.client.post(
        f"/api/agents/instrument_control/interrupts/{card['event_id']}/resolve",
        json={"decision": "edit", "edited_args": {"bias_v": 0.5}})
    assert r.status_code == 200, r.text

    assert done.wait(timeout=15)
    t.join(timeout=5)
    decision = wired.graph.resumed_with["lg-intr-1"]["decisions"][0]
    assert decision["type"] == "edit"
    # A NUMBER, not the string the JSON body carried: SafetyGate's re-check in
    # the core only inspects int/float, so a string would slip past it.
    assert decision["edited_action"]["args"]["bias_v"] == 0.5
    assert isinstance(decision["edited_action"]["args"]["bias_v"], float)


def test_a_second_message_while_waiting_is_refused_not_raced(wired):
    """Two streams over one checkpoint thread corrupt it, so the guard REFUSES
    the second — which is also the operator-visible answer to "why is nothing
    happening": the first turn is waiting on them."""
    snapshots, done, t = _run_turn(wired)
    _wait_for_card(wired)

    second = list(wired.engine.stream_turn(wired.cid, "在吗？"))
    assert "已有一个进行中的回合" in _text(second)

    card = _wait_for_card(wired)
    wired.client.post(
        f"/api/agents/instrument_control/interrupts/{card['event_id']}/resolve",
        json={"decision": "reject"})
    assert done.wait(timeout=15)
    t.join(timeout=5)


def test_stop_during_the_wait_ends_the_turn_and_clears_the_card(wired):
    abort = threading.Event()
    snapshots, done, t = _run_turn(wired, abort=abort)
    _wait_for_card(wired)

    abort.set()   # what POST /chat/abort fires
    assert done.wait(timeout=15), "Stop must end a turn that is waiting on approval"
    t.join(timeout=5)
    assert wired.graph.resumed_with is None, "an abandoned approval must not resume"
    # The card is gone: resolving it would wake nobody.
    assert not wired.interrupts["pending"]
    r = wired.client.get("/api/agents/instrument_control/interrupts")
    assert r.json()["interrupts"] == []


def test_the_endpoints_degrade_instead_of_500_without_a_live_core(tmp_path):
    """House rule: this router must boot standalone. A dev instance with no core
    answers "degraded", never a 500 that the UI shows as a red error."""
    ctx = types.SimpleNamespace(live_app=None, app=None)
    c = _client(ctx)

    r = c.get("/api/agents/instrument_control/interrupts")
    assert r.status_code == 200 and r.json()["degraded"] is True

    r = c.post("/api/agents/instrument_control/interrupts/nope/resolve",
               json={"decision": "approve"})
    assert r.status_code == 200
    assert r.json()["degraded"] is True and r.json().get("ok") is not True

    # …and with no store there is no resolver, so the engine keeps its honest
    # "nobody can approve this here" notice rather than pretending otherwise.
    assert _make_hitl_resolver(ctx, "instrument_control", "c1", None) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
