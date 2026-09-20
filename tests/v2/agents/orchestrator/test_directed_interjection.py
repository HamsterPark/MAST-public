"""Hard @agent routing — an operator who addresses a specific agent reaches it.

Before: an interjection targeted at an agent (POST /agents/<id>/interject) was
only softened into a "(指向 X)" text line and handed to the LLM router, which
often ignored it ("@agent 不管用", ).

Now: the control provider surfaces ``directed_targets`` and the supervisor
dispatches to them DETERMINISTICALLY (the guarded hint path), taking precedence
over the LLM router.

Pinned here against the REAL supervisor node:
  * a directed target is dispatched even when the router would END;
  * the operator target's hop is counted in visit_count (loop guard stays honest);
  * with no directed target the behaviour is unchanged (router decides);
  * an unknown / unwired directed target is dropped (falls through to the router).
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from typing import Any  # noqa: E402

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from mast.agents.orchestrator.graph import _supervisor_node_factory  # noqa: E402
from mast.agents.state import MASTState  # noqa: E402


def _base_state(**kw) -> dict:
    st: dict[str, Any] = {
        "messages": [HumanMessage(content="goal")],
        "executed_skills": [], "scan_paths": [], "scan_metadata": {},
        "error_log": [], "event_refs": [], "visit_count": {},
        "pending_approvals": {},
    }
    st.update(kw)
    return st


class _EndRouter:
    """A router that ALWAYS wants to END — so anything that still runs proves a
    non-router force dispatched it."""

    def with_structured_output(self, schema, method=None):
        class _S:
            def invoke(self, _msgs):
                return {"next_agents": ["__end__"], "reason": "router wants end"}
        return _S()

    def invoke(self, _msgs):
        return AIMessage(content="end")


class _OnceDirected:
    """control_provider: yields a directed target on the FIRST drain only."""

    def __init__(self, targets, text="做这个"):
        self._targets = list(targets)
        self._text = text
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls == 1 and self._targets:
            return {"interjections": [f"(指向 {self._targets[0]}) {self._text}"],
                    "directed_targets": list(self._targets)}
        return {}


def _agent_stub(name: str, ran: list):
    def node(state):
        ran.append(name)
        # hand straight back to the supervisor (no routing hint)
        return Command(goto="supervisor",
                       update={"messages": [AIMessage(content=f"{name} ran")],
                               "active_agent": "supervisor"})
    return node


def _build(router, control_provider, agents, ran):
    g: StateGraph = StateGraph(MASTState)
    g.add_node("supervisor", _supervisor_node_factory(
        router, control_provider, wired_agents=tuple(agents)))
    for a in agents:
        g.add_node(a, _agent_stub(a, ran))
    g.add_edge(START, "supervisor")
    return g.compile(checkpointer=InMemorySaver())


def test_directed_target_dispatched_even_when_router_would_end():
    ran: list = []
    app = _build(_EndRouter(), _OnceDirected(["instrument_control"]),
                 ["instrument_control", "literature"], ran)
    out = app.invoke(_base_state(),
                     config={"configurable": {"thread_id": "d1"}, "recursion_limit": 30})
    # IC ran because the operator addressed it — even though the router said END.
    assert "instrument_control" in ran, "operator @instrument_control was not honoured"
    # literature never ran (nobody asked for it).
    assert "literature" not in ran
    # the operator-directed hop was counted (loop guard honesty).
    assert out["visit_count"].get("instrument_control", 0) >= 1
    # the supervisor's route note names the directed agent.
    assert any("[SUPERVISOR → instrument_control]" in str(getattr(m, "content", ""))
               for m in out["messages"])


def test_no_directed_target_leaves_router_in_charge():
    ran: list = []
    # no directed target → the _EndRouter ends immediately, nothing runs.
    app = _build(_EndRouter(), lambda: {}, ["instrument_control", "literature"], ran)
    app.invoke(_base_state(),
               config={"configurable": {"thread_id": "d2"}, "recursion_limit": 30})
    assert ran == [], f"router said END but agents ran: {ran}"


def test_unknown_directed_target_is_dropped():
    ran: list = []
    # operator addresses an agent that is NOT wired → dropped, router (END) wins.
    app = _build(_EndRouter(), _OnceDirected(["nonexistent_agent"]),
                 ["instrument_control", "literature"], ran)
    app.invoke(_base_state(),
               config={"configurable": {"thread_id": "d3"}, "recursion_limit": 30})
    assert ran == [], f"an unwired directed target should be dropped, not dispatched: {ran}"


def test_directed_target_takes_precedence_over_a_stale_agent_hint():
    """If an agent handoff left routing_hints=[literature] AND the operator now
    addresses instrument_control, the operator's target dispatches too (union),
    and IC — the operator's explicit choice — is included."""
    ran: list = []
    app = _build(_EndRouter(), _OnceDirected(["instrument_control"]),
                 ["instrument_control", "literature"], ran)
    out = app.invoke(_base_state(routing_hints=["literature"]),
                     config={"configurable": {"thread_id": "d4"}, "recursion_limit": 30})
    assert "instrument_control" in ran, "operator's explicit @IC was dropped"
    # the hint channel is cleared after consumption (no resurrection).
    assert not out.get("routing_hints")
