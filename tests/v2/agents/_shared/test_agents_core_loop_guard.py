"""agents_core review fixes — findings #12, #14, #15 (2026-05-30).

Real-logic regression tests (no mocking of the objects under test):

  #12  visit_count was reduced with ``merge_dicts`` (right-side OVERWRITES), so
       the handoff's ``{target: 1}`` delta never accumulated — a repeatedly
       re-entered agent stayed pinned at 1 and the per-agent loop guard
       (``per_agent_max > 8``) could never trip. Fixed by the ADDING reducer
       ``sum_int_dicts`` + delta writes everywhere.

  #14  wrap_skill returned a ``str`` subclass; langgraph's ToolNode wraps a
       non-Command return into a bare ToolMessage and DISCARDS any extra
       attributes — so executed_skills / scan_paths / composite_progress /
       error_log never reached MASTState in the real agent runtime. Fixed by
       returning a real ``Command(update={...})`` (carrying an injected
       tool_call_id) so ToolNode applies the update.

  #15  Sibling handoffs returned ``Command(goto=<sibling>, graph=PARENT)`` and
       routed straight node→node in the parent graph, bypassing the supervisor
       node — the ONLY place the loop/budget guard runs. An A→B→A→B ping-pong
       was unbounded. Fixed by routing EVERY handoff through ``supervisor``
       (recording the intended next agent as ``routing_hint``), so the guard
       runs on every inter-agent hop.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/_shared/test_agents_core_loop_guard.py -x -v
"""
from __future__ import annotations

# ── path bootstrap (robust walk-up) ──
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command

from mast.agents._shared.handoff import make_handoff
from mast.agents._shared.skill_adapter import wrap_skill
from mast.agents.orchestrator.graph import _AGENT_NAMES, _supervisor_node_factory
from mast.agents.state import MASTState, merge_dicts, sum_int_dicts
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.bias import GetBias, SetBias


def _base_state(**kw) -> dict:
    s = {
        "visit_count": {}, "messages": [], "executed_skills": [],
        "scan_paths": [], "scan_metadata": {}, "error_log": [],
        "pending_approvals": {},
    }
    s.update(kw)
    return s


# ════════════════════════════════════════════════════════════════════
# #12 — sum_int_dicts reducer accumulates; visit_count uses it
# ════════════════════════════════════════════════════════════════════

class TestSumIntDictsReducer:
    def test_accumulates_per_key(self):
        assert sum_int_dicts({"ic": 3}, {"ic": 1}) == {"ic": 4}

    def test_new_keys_added(self):
        assert sum_int_dicts({"ic": 3}, {"dp": 2}) == {"ic": 3, "dp": 2}

    def test_none_left_is_an_empty_accumulator(self):
        assert sum_int_dicts(None, {"a": 1}) == {"a": 1}

    def test_none_RIGHT_clears_the_channel(self):
        """``None`` on the right is a RESET, not "no change" (changed 2026-07-29).

        An adding reducer can never be reset from the outside, so it needed the
        same escape hatch ``merge_routing_hints`` has. Without it, run_task's
        per-task "reset" — which wrote ``{}`` — was a NO-OP, hops accumulated for
        the lifetime of the group thread, and once the running total crossed the
        loop guard (40) every subsequent task in that conversation ended
        instantly. The conversation was bricked and nothing said why.
        """
        assert sum_int_dicts({"a": 7, "supervisor": 33}, None) == {}
        assert sum_int_dicts(None, None) == {}

    def test_an_empty_dict_is_still_a_no_op_not_a_reset(self):
        """The other half of the same contract — and the actual bug's shape.

        This is pinned so nobody 'simplifies' ``{}`` into meaning a reset: an
        empty delta legitimately means "this writer had nothing to add", and
        several branches of a fan-out send exactly that.
        """
        assert sum_int_dicts({"a": 7}, {}) == {"a": 7}

    def test_repeated_target_deltas_climb(self):
        # The exact failure mode of #12: nine separate {target:1} handoff deltas.
        vc: dict[str, int] = {}
        for _ in range(9):
            vc = sum_int_dicts(vc, {"paper_writing": 1})
        assert vc["paper_writing"] == 9  # NOT pinned at 1

    def test_contrast_with_old_merge_dicts_overwrite(self):
        # Proves WHY the bug existed: merge_dicts (the old reducer) overwrites,
        # so the count would have been stuck at 1 forever.
        vc: dict[str, int] = {}
        for _ in range(9):
            vc = merge_dicts(vc, {"paper_writing": 1})
        assert vc["paper_writing"] == 1  # old behavior — guard never trips

    def test_state_wires_sum_int_dicts(self):
        import typing
        hints = typing.get_type_hints(MASTState, include_extras=True)
        reducer = hints["visit_count"].__metadata__[0]
        assert reducer is sum_int_dicts, "visit_count must use the ADDING reducer"


# ════════════════════════════════════════════════════════════════════
# #15 — handoff routes through supervisor; supervisor honors routing_hint
# ════════════════════════════════════════════════════════════════════

class TestHandoffRoutesThroughSupervisor:
    def test_sibling_handoff_goes_to_supervisor_not_sibling(self):
        tool = make_handoff("paper_writing", "hand analysis to writing")
        cmd = tool.func(reason="analysis done", tool_call_id="tc1")
        # Must NOT jump straight to the sibling — must land on supervisor.
        assert cmd.goto == "supervisor"
        assert cmd.graph == Command.PARENT
        # Intended next agent preserved as a routing hint. PLURAL since
        # 2026-07-11: under parallel fan-out several agents hand back in the same
        # super-step, each contributing its own hint (and a scalar channel would
        # crash on that concurrent write).
        assert cmd.update["routing_hints"] == ["paper_writing"]
        # +1 delta for the target so the guard accumulates it.
        assert cmd.update["visit_count"] == {"paper_writing": 1}

    def test_return_to_supervisor_sets_no_hint(self):
        tool = make_handoff("supervisor", "done, decide next")
        cmd = tool.func(reason="complete", tool_call_id="tc2")
        assert cmd.goto == "supervisor"
        assert "routing_hints" not in cmd.update
        # Double-count fix: a RETURN handoff must NOT bump visit_count for the
        # supervisor — the supervisor NODE counts its own visit when it runs next.
        # Counting here too would double-count the supervisor (the binding key for
        # the per-agent cap=8 guard) and trip the loop guard at ~half budget.
        assert "visit_count" not in cmd.update

    def test_handoff_toolmessage_carries_tool_call_id(self):
        tool = make_handoff("data_processing", "x")
        cmd = tool.func(reason="r", tool_call_id="tc-xyz")
        msg = cmd.update["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc-xyz"


class TestSupervisorHonorsHint:
    def test_hint_dispatches_without_llm(self):
        # No model configured, but a routing hint is present → supervisor must
        # honor it (dispatch to the agent) rather than END.
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(routing_hints=["paper_writing"]))
        assert cmd.goto == "paper_writing"      # single hint → plain goto
        assert cmd.update["routing_hints"] is None  # None CLEARS the channel
        # supervisor's own visit is the only delta here (handoff already counted
        # the target), so we don't double-count the agent.
        assert cmd.update["visit_count"] == {"supervisor": 1}

    def test_multiple_hints_fan_out(self):
        """Several agents finishing in parallel each name a next hop — that IS a
        fan-out request, and the supervisor honours all of them (2026-07-11)."""
        from langgraph.types import Send
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(routing_hints=["paper_writing", "data_processing"]))
        assert isinstance(cmd.goto, list) and len(cmd.goto) == 2
        assert all(isinstance(s, Send) for s in cmd.goto)
        assert {s.node for s in cmd.goto} == {"paper_writing", "data_processing"}
        assert cmd.update["routing_hints"] is None
        # Each Send carries an explicit payload — the branches must SEE the
        # supervisor's routing note (Send bypasses the channel update).
        for s in cmd.goto:
            assert any("SUPERVISOR" in str(getattr(m, "content", ""))
                       for m in s.arg["messages"])

    def test_invalid_hint_falls_through_to_end(self):
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(routing_hints=["not_a_real_agent"]))
        assert cmd.goto == END  # bad hint ignored, no-model path → END

    def test_unwired_hint_dropped_from_fanout(self):
        """A mixed list keeps only the wired agents (an unwired name would crash
        LangGraph with a missing-node error)."""
        node = _supervisor_node_factory(
            supervisor_model=None, wired_agents=("paper_writing",))
        cmd = node(_base_state(routing_hints=["paper_writing", "data_processing"]))
        assert cmd.goto == "paper_writing"  # the one wired hint, dispatched alone

    def test_guard_runs_before_hint_is_honored(self):
        # Even with a valid hint, the loop guard must win (it runs first).
        # Per-agent cap removed 2026-06-29 → trip the TOTAL-hop guard (>40).
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(visit_count={"paper_review": 41},
                               routing_hints=["paper_writing"]))
        assert cmd.goto == END
        assert "Loop guard tripped" in cmd.update["messages"][0].content


class TestSiblingPingPongIsBounded:
    """End-to-end: a sibling A↔B ping-pong in a real compiled parent graph is
    now bounded by the supervisor guard, because every hop routes through it.

    Build a minimal parent graph: supervisor + two agent nodes that always hand
    off to each other VIA the real make_handoff Command (goto="supervisor",
    routing_hint=<sibling>, +1 visit_count delta). The only adaptation for this
    flat test graph is dropping ``graph=Command.PARENT`` — that escape hatch
    only matters when the handoff is returned from inside an agent SUBGRAPH (it
    routes the Command up to the parent supervisor); here supervisor already
    lives in the same graph, and ``graph=PARENT`` would try to escape to a
    non-existent grandparent. The routing_hint + supervisor-guard mechanism —
    the actual subject of finding #15 — is exercised faithfully and unchanged.

    Without the #12 fix the +1 deltas would overwrite (stuck at 1) and the guard
    could never trip; without the #15 fix the agents would Command(goto=sibling)
    and never re-enter the supervisor at all → unbounded.
    """

    def _agent_node(self, me: str, sibling: str):
        handoff = make_handoff(sibling, f"{me}->{sibling}")

        def _node(state: MASTState) -> Command:
            cmd = handoff.func(reason="ping", tool_call_id=f"{me}-tc")
            # Same update (routing_hints + visit_count delta + messages), routed
            # to supervisor within this flat graph (see class docstring).
            return Command(goto="supervisor", update=dict(cmd.update))

        return _node

    def _build(self):
        supervisor = _supervisor_node_factory(supervisor_model=None)
        g: StateGraph = StateGraph(MASTState)
        g.add_node("supervisor", supervisor)
        g.add_node("paper_writing", self._agent_node("paper_writing", "paper_review"))
        g.add_node("paper_review", self._agent_node("paper_review", "paper_writing"))
        g.add_edge(START, "supervisor")
        return g.compile()

    def test_ping_pong_terminates_via_guard(self):
        app = self._build()
        # Seed a routing hint so the supervisor dispatches into the ping-pong.
        result = app.invoke(
            _base_state(routing_hints=["paper_writing"]),
            config={"configurable": {"thread_id": "pingpong-1"},
                    "recursion_limit": 200},
        )
        # The loop guard must have ended the run — NOT a recursion error. The
        # final supervisor message says the guard tripped. (Whether the total>40
        # or per-agent>8 branch fires first depends on how the supervisor's own
        # visits accumulate; either way the run is BOUNDED, which is the point.)
        texts = [getattr(m, "content", "") for m in result["messages"]]
        assert any("Loop guard tripped" in t for t in texts), \
            "sibling ping-pong must be caught by the supervisor loop guard"
        vc = result["visit_count"]
        # #12 proof: the bouncing agents accumulated PAST 1 — the old merge_dicts
        # reducer would have pinned every agent at exactly 1 forever, and the
        # supervisor key (re-entered every hop) climbed well past 1 too.
        assert vc["paper_writing"] > 1 and vc["paper_review"] > 1
        assert vc["supervisor"] > 1
        # #15 proof: every inter-agent hop passed through the supervisor, so its
        # count is at least as high as the total agent visits (it ran for each).
        assert vc["supervisor"] >= vc["paper_writing"] + vc["paper_review"] - 1
        # Bounded overall (guard's total cap is 40).
        assert sum(vc.values()) <= 60


# ════════════════════════════════════════════════════════════════════
# #14 — wrap_skill state side-effects really persist through ToolNode
# ════════════════════════════════════════════════════════════════════

@dataclass
class _FakeCtx:
    """Minimal ExecutionContext substitute (the SKILL is the object under test,
    not mocked — only the hardware boundary safe_call is canned)."""
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            e = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=e.get("return_value"),
                error=e.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


class _ToolState(TypedDict):
    messages: Annotated[list, add_messages]
    executed_skills: Annotated[list, add]
    scan_paths: Annotated[list, add]


def _run_tool_through_toolnode(tool, tool_name, args):
    """Drive a wrapped skill through a REAL compiled ToolNode StateGraph — the
    actual agent runtime path (NOT a direct .func call)."""
    g = StateGraph(_ToolState)
    g.add_node("tools", ToolNode([tool]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    app = g.compile()
    ai = AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": "tc-real"}])
    return app.invoke({"messages": [ai], "executed_skills": [], "scan_paths": []})


class TestWrapSkillPersistsThroughToolNode:
    def test_tool_call_id_hidden_from_llm(self):
        tool = wrap_skill(GetBias, lambda: _FakeCtx())
        # The injected id must NOT appear in the LLM-facing schema.
        assert "tool_call_id" not in tool.args

    def test_executed_skills_reach_state(self):
        tool = wrap_skill(
            GetBias,
            lambda: _FakeCtx({"Bias_Get": {"return_value": ("", b"", [1.234])}}),
        )
        res = _run_tool_through_toolnode(tool, "GetBias", {})
        # THE FIX: executed_skills is persisted into state (was [] before #14).
        assert res["executed_skills"] == ["GetBias"]

    def test_toolmessage_has_matching_call_id_and_summary(self):
        tool = wrap_skill(
            GetBias,
            lambda: _FakeCtx({"Bias_Get": {"return_value": ("", b"", [1.234])}}),
        )
        res = _run_tool_through_toolnode(tool, "GetBias", {})
        tmsgs = [m for m in res["messages"] if isinstance(m, ToolMessage)]
        assert len(tmsgs) == 1
        assert tmsgs[0].tool_call_id == "tc-real"
        assert "1.234" in tmsgs[0].content or "bias_v" in tmsgs[0].content

    def test_wrapped_return_is_a_command(self):
        # The mechanism: a real langgraph Command is what makes ToolNode apply
        # the update. A plain str (the old behavior) would have been silently
        # discarded.
        tool = wrap_skill(
            SetBias,
            lambda: _FakeCtx({"Bias_Set": {"return_value": ("", b"", [])}}),
        )
        ret = tool.func(tool_call_id="t", bias_v=0.5)
        assert isinstance(ret, Command)
        assert ret.update["executed_skills"] == ["SetBias"]

    def test_scan_path_artifact_reaches_state(self):
        # A skill whose result carries a 'path' must surface it into scan_paths.
        @dataclass
        class _Result:
            success: bool = True
            data: dict = field(default_factory=lambda: {"path": "E:/data/scan_001.sxm"})
            error: str = ""
            summary: str = "scanned ok"

        class _FakeScanSkill:
            def metadata(self):
                from mast.core.types import SkillMetadata, SkillCategory, SafetyLevel
                return SkillMetadata(
                    name="FakeScan", description="d",
                    category=SkillCategory.WRITE,
                    safety_level=SafetyLevel.AUTO,
                    parameters=[],
                )
            def validate_params(self, kwargs):
                return []
            def execute(self, ctx, params):
                return _Result()

        tool = wrap_skill(_FakeScanSkill, lambda: _FakeCtx())
        res = _run_tool_through_toolnode(tool, "FakeScan", {})
        assert res["scan_paths"] == ["E:/data/scan_001.sxm"]
        assert res["executed_skills"] == ["FakeScan"]

    def test_backward_compat_str_and_in_on_return(self):
        # Old direct-call tests rely on str(result) / "x" in result returning
        # the human summary — those must still work on the Command subclass.
        tool = wrap_skill(
            GetBias,
            lambda: _FakeCtx({"Bias_Get": {"return_value": ("", b"", [1.234])}}),
        )
        ret = tool.func(tool_call_id="t")
        assert "1.234" in str(ret)
        assert "1.234" in ret  # __contains__ over the summary
        assert ret.update["executed_skills"] == ["GetBias"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
