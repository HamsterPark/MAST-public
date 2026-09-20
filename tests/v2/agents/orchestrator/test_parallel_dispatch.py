"""Parallel fan-out — the supervisor dispatches several agents at once .

The supervisor was strictly serial hub-and-spoke: ONE agent per hop, so a
literature survey could not run while the instrument scanned and a 6-phase
campaign paid the sum of every phase. The operator confirmed the safety
premise: instrument_control is the ONLY agent that touches hardware, so
concurrent agents cannot contend for the instrument.

What must hold (all exercised against REAL compiled graphs, not mocks):
  * a multi-target decision fans out via Send and every branch runs;
  * concurrent handoffs from those branches do NOT crash the state — every key
    a handoff writes has a reducer (this used to raise InvalidUpdateError);
  * each branch SEES what the supervisor injected this step (Send passes an
    explicit payload — a naive implementation silently starves the branches);
  * routing hints from several branches are collected, not lost;
  * instrument_control can never appear twice in one fan-out;
  * the width cap holds and the loop guard still bounds the run;
  * a router that only speaks the OLD single-agent shape still routes.
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

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, Send, interrupt  # noqa: E402

from mast.agents._shared.handoff import make_handoff  # noqa: E402
from mast.agents.orchestrator.graph import (  # noqa: E402
    _AGENT_NAMES,
    _MAX_PARALLEL,
    _coerce_targets,
    _parallel_route_decision,
    _supervisor_node_factory,
)
from mast.agents.state import MASTState, merge_routing_hints  # noqa: E402

_VALID = frozenset(list(_AGENT_NAMES) + ["__end__"])


def _base_state(**kw) -> MASTState:
    st: dict[str, Any] = {
        "messages": [HumanMessage(content="goal")],
        "executed_skills": [], "scan_paths": [], "scan_metadata": {},
        "error_log": [], "event_refs": [], "visit_count": {},
        "pending_approvals": {},
    }
    st.update(kw)
    return st  # type: ignore[return-value]


# ── router stubs ────────────────────────────────────────────────────────────
class _ParallelStub:
    """Speaks the NEW shape via function_calling (tier 1)."""

    def __init__(self, agents, reason="parallel"):
        self._out = {"next_agents": list(agents), "reason": reason}

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _msgs):
                return dict(outer._out)

        return _S()

    def invoke(self, _msgs):  # pragma: no cover — tier 1 succeeds
        raise AssertionError("should not reach the text tier")


class _LegacySingleStub:
    """A provider that only ever answers the OLD singular shape, in free text.

    Structured output fails (as it does on several real providers), the text
    tier must then salvage a single agent — parallel must never make routing
    WORSE than it was.
    """

    def with_structured_output(self, schema, method=None):
        class _S:
            def invoke(self, _msgs):
                raise RuntimeError("this provider does not support function_calling")

        return _S()

    def invoke(self, _msgs):
        return AIMessage(content='{"next_agent": "literature", "reason": "old shape"}')


# ═══════════════════════════════════════════════════════════════════════════
# target coercion — the invariant guard
# ═══════════════════════════════════════════════════════════════════════════

class TestCoerceTargets:
    def test_dedupes_instrument_control(self):
        """The hardware invariant: IC can never appear twice in one fan-out."""
        out = _coerce_targets(
            ["instrument_control", "instrument_control", "literature"], _VALID)
        assert out.count("instrument_control") == 1
        assert out == ["instrument_control", "literature"]

    def test_drops_unknown_agents(self):
        assert _coerce_targets(["literature", "not_an_agent"], _VALID) == ["literature"]

    def test_end_is_absorbing(self):
        """"finish" + "do more work" is incoherent — ending is the safe read."""
        assert _coerce_targets(["literature", "__end__"], _VALID) == ["__end__"]

    def test_width_capped(self):
        out = _coerce_targets(list(_AGENT_NAMES), _VALID)
        assert len(out) == _MAX_PARALLEL

    def test_accepts_a_bare_string(self):
        assert _coerce_targets("literature", _VALID) == ["literature"]

    def test_accepts_comma_separated_improvisation(self):
        assert _coerce_targets("literature, data_processing", _VALID) == [
            "literature", "data_processing"]

    def test_junk_yields_nothing(self):
        assert _coerce_targets(None, _VALID) == []
        assert _coerce_targets(123, _VALID) == []


class TestRoutingHintsReducer:
    def test_collects_from_parallel_branches(self):
        assert merge_routing_hints(["a"], ["b"]) == ["a", "b"]

    def test_none_clears(self):
        assert merge_routing_hints(["a", "b"], None) == []

    def test_idempotent(self):
        assert merge_routing_hints(["a"], ["a"]) == ["a"]


# ═══════════════════════════════════════════════════════════════════════════
# supervisor node — dispatch shape
# ═══════════════════════════════════════════════════════════════════════════

class TestSupervisorFanOut:
    def test_multi_target_emits_sends(self):
        node = _supervisor_node_factory(
            _ParallelStub(["instrument_control", "literature"]))
        cmd = node(_base_state())
        assert isinstance(cmd.goto, list)
        assert {s.node for s in cmd.goto} == {"instrument_control", "literature"}
        # every dispatched agent is counted — a 2-way fan-out is 2 agent-hops
        assert cmd.update["visit_count"] == {
            "supervisor": 1, "instrument_control": 1, "literature": 1}

    def test_single_target_stays_a_plain_goto(self):
        """The serial path must be untouched: no Send, no behaviour change."""
        node = _supervisor_node_factory(_ParallelStub(["literature"]))
        cmd = node(_base_state())
        assert cmd.goto == "literature"
        assert not isinstance(cmd.goto, list)

    def test_branches_see_the_supervisors_message(self):
        """Send passes an EXPLICIT payload — if we forwarded the pre-update state
        the branches would never see the routing note or an operator interjection."""
        node = _supervisor_node_factory(
            _ParallelStub(["literature", "data_processing"], reason="并行理由"))
        cmd = node(_base_state())
        for send in cmd.goto:
            texts = [str(getattr(m, "content", "")) for m in send.arg["messages"]]
            assert any("并行理由" in t for t in texts)
            assert any("goal" in t for t in texts)   # original user turn preserved

    def test_unwired_agent_dropped_from_fanout(self):
        node = _supervisor_node_factory(
            _ParallelStub(["literature", "paper_review"]),
            wired_agents=("literature",))
        cmd = node(_base_state())
        assert cmd.goto == "literature"   # the unwired one would crash LangGraph

    def test_legacy_single_shape_router_still_routes(self):
        node = _supervisor_node_factory(_LegacySingleStub())
        cmd = node(_base_state())
        assert cmd.goto == "literature"

    def test_loop_guard_still_bounds_a_fanout(self):
        node = _supervisor_node_factory(_ParallelStub(["literature", "paper_review"]))
        cmd = node(_base_state(visit_count={"literature": 41}))
        assert cmd.goto == END
        assert "Loop guard tripped" in cmd.update["messages"][0].content


# ═══════════════════════════════════════════════════════════════════════════
# END-TO-END on a real compiled graph — the crash this design had to survive
# ═══════════════════════════════════════════════════════════════════════════

def _agent(name: str, next_hop: str | None = None):
    """An agent node that does work and hands back through the REAL handoff tool."""
    handoff = make_handoff(next_hop or "supervisor", "done")

    def _node(state: MASTState) -> Command:
        cmd = handoff.func(reason=f"{name} finished", tool_call_id=f"{name}-tc")
        upd = dict(cmd.update)
        upd["messages"] = [AIMessage(content=f"{name} did work")] + list(upd["messages"])
        upd["executed_skills"] = [f"{name}_skill"]
        return Command(goto="supervisor", update=upd)

    return _node


class TestParallelEndToEnd:
    def _build(self, model, agents=("instrument_control", "literature")):
        g: StateGraph = StateGraph(MASTState)
        g.add_node("supervisor", _supervisor_node_factory(
            model, wired_agents=tuple(agents)))
        for a in agents:
            g.add_node(a, _agent(a))
        g.add_edge(START, "supervisor")
        return g.compile(checkpointer=InMemorySaver())

    def test_concurrent_handoffs_do_not_crash_state(self):
        """THE regression: two branches handing back in the same super-step both
        write active_agent + routing_hints + visit_count + messages. Before the
        reducers this raised InvalidUpdateError and killed the whole run."""
        class _OnceThenEnd:
            def __init__(self):
                self.calls = 0

            def with_structured_output(self, schema, method=None):
                outer = self

                class _S:
                    def invoke(self, _msgs):
                        outer.calls += 1
                        if outer.calls == 1:
                            return {"next_agents": ["instrument_control", "literature"],
                                    "reason": "扫描与文献互不依赖"}
                        return {"next_agents": ["__end__"], "reason": "done"}

                return _S()

            def invoke(self, _msgs):
                return AIMessage(content="ok")

        app = self._build(_OnceThenEnd())
        out = app.invoke(_base_state(),
                         config={"configurable": {"thread_id": "par-1"},
                                 "recursion_limit": 50})
        texts = [str(getattr(m, "content", "")) for m in out["messages"]]
        # BOTH agents really ran, in one fan-out
        assert any("instrument_control did work" in t for t in texts)
        assert any("literature did work" in t for t in texts)
        # both their handoffs landed (dedupe_append preserved both skills)
        assert set(out["executed_skills"]) == {
            "instrument_control_skill", "literature_skill"}
        # the hint channel was consumed and cleared
        assert not out.get("routing_hints")

    def test_parallel_branches_can_each_pause_for_hitl(self):
        """Both branches interrupt at once; each is resumed BY ID with its own
        decision. A broadcast resume would feed A's approval to B."""
        def _asking_agent(name: str):
            def _node(state: MASTState) -> Command:
                verdict = interrupt({"kind": "dangerous", "agent": name})
                return Command(goto="supervisor", update={
                    "messages": [AIMessage(content=f"{name} resumed with {verdict}")],
                    "active_agent": name,
                })
            return _node

        class _FanThenEnd:
            def __init__(self):
                self.calls = 0

            def with_structured_output(self, schema, method=None):
                outer = self

                class _S:
                    def invoke(self, _msgs):
                        outer.calls += 1
                        if outer.calls == 1:
                            return {"next_agents": ["instrument_control", "literature"],
                                    "reason": "fan out"}
                        return {"next_agents": ["__end__"], "reason": "done"}

                return _S()

            def invoke(self, _msgs):
                return AIMessage(content="ok")

        g: StateGraph = StateGraph(MASTState)
        g.add_node("supervisor", _supervisor_node_factory(
            _FanThenEnd(), wired_agents=("instrument_control", "literature")))
        g.add_node("instrument_control", _asking_agent("instrument_control"))
        g.add_node("literature", _asking_agent("literature"))
        g.add_edge(START, "supervisor")
        app = g.compile(checkpointer=InMemorySaver())
        cfg = {"configurable": {"thread_id": "par-hitl"}, "recursion_limit": 50}

        app.invoke(_base_state(), config=cfg)
        snap = app.get_state(cfg)
        assert len(snap.interrupts) == 2, "both parallel branches must pause"
        # resume each by ITS OWN id with a DIFFERENT verdict
        resume_map = {
            i.id: f"verdict-for-{i.value['agent']}" for i in snap.interrupts
        }
        out = app.invoke(Command(resume=resume_map), config=cfg)
        texts = [str(getattr(m, "content", "")) for m in out["messages"]]
        # each branch got ITS OWN decision — no cross-feeding
        assert any("instrument_control resumed with verdict-for-instrument_control" in t
                   for t in texts)
        assert any("literature resumed with verdict-for-literature" in t for t in texts)


class TestRouteDecisionTiers:
    def test_tier1_structured_list(self):
        d = _parallel_route_decision(
            _ParallelStub(["literature", "paper_review"]), [])
        assert d["next_agents"] == ["literature", "paper_review"]

    def test_tier3_falls_back_to_single_target_router(self):
        d = _parallel_route_decision(_LegacySingleStub(), [])
        assert d["next_agents"] == ["literature"]

    def test_text_tier_reads_a_next_agents_array(self):
        class _TextStub:
            def with_structured_output(self, schema, method=None):
                class _S:
                    def invoke(self, _m):
                        raise RuntimeError("no function_calling here")
                return _S()

            def invoke(self, _m):
                return AIMessage(content=(
                    'sure: {"next_agents": ["data_processing", "literature"], '
                    '"reason": "both ready"}'))

        d = _parallel_route_decision(_TextStub(), [])
        assert d["next_agents"] == ["data_processing", "literature"]
        assert d["reason"] == "both ready"
