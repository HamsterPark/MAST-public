"""Budget soft-limit warning — review MEDIUM #7 (2026-05-30).

The supervisor's budget/loop guard previously had ONLY hard gates. A run could
sail along and then get guillotined mid-thought with no warning. This adds a
NON-blocking, fire-once soft-limit warning injected into the message stream at
~80% consumption so the agent steers toward a conclusion first.

Retuned 2026-07-30 and the numbers here are now DERIVED, never spelled out
---------------------------------------------------------------------------
The ceilings changed (total 40 → 60; the per-agent dimension came back, split so
the ``supervisor`` key has its own much higher ceiling). Every threshold in this
file is computed from the constants it is testing, because the defect that
prompted the retune was exactly a hard-coded number: the live gate read a literal
``40`` while ``_HOP_HARD_CAP`` fed only the soft warning, so retuning the constant
changed nothing and every reader — comments and tests included — believed it had.
A test that repeats the literal cannot catch that; a test that derives from the
constant can, and ``TestConstantIsTheRealGate`` below pins it directly.

These tests drive the REAL supervisor_node (no mocking of the object under
test; supervisor_model=None is the genuine no-LLM/no-network path the code
already supports). They assert:
  - the warning fires once at the 80% hop threshold,
  - it does NOT fire below it,
  - it is NOT re-injected once already in the stream (idempotent),
  - the per-agent dimension warns again, and the ``supervisor`` key is excluded
    from it (counting it was half of why the old cap misfired),
  - the dollar-budget soft dimension is REMOVED: it keyed off
    budget_initial_usd, which is not a declared MASTState channel and was dropped
    on every real graph hop (dead code),
  - the HARD gates still win at the cap → END (soft limit never weakens them),
  - the hard gate follows the CONSTANT, not a literal,
  - end-to-end through a real compiled graph: warned exactly once, then the
    hard loop guard still ENDs the run.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/orchestrator/test_budget_softlimit.py -q -p no:randomly
"""
from __future__ import annotations

# ── path bootstrap (robust walk-up; matches sibling orchestrator tests) ──
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
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from mast.agents.orchestrator import graph as _orch_graph
from mast.agents.orchestrator.graph import (
    _AGENT_HARD_CAP,
    _AGENT_NAMES,
    _AGENT_SOFT_THRESHOLD,
    _HOP_HARD_CAP,
    _HOP_SOFT_THRESHOLD,
    _SOFTLIMIT_MARKER,
    _SUPERVISOR_HARD_CAP,
    _SUPERVISOR_KEY,
    _already_soft_warned,
    _budget_softlimit_warning,
    _guard_dimensions,
    _supervisor_node_factory,
)
from mast.agents.state import MASTState


def _base_state(**kw) -> dict:
    s = {
        "visit_count": {}, "messages": [], "executed_skills": [],
        "scan_paths": [], "scan_metadata": {}, "error_log": [],
        "pending_approvals": {},
    }
    s.update(kw)
    return s


#: A total safely inside the soft band: past the soft threshold, under the hard
#: cap, and low enough per agent that spreading it never trips another dimension.
_SOFT_TOTAL = _HOP_SOFT_THRESHOLD + 2


def _msgs_with_marker(messages) -> list[str]:
    """Contents of messages that carry the soft-limit marker."""
    out = []
    for m in messages or []:
        c = getattr(m, "content", "")
        if isinstance(c, str) and _SOFTLIMIT_MARKER in c:
            out.append(c)
    return out


def _spread_visit_count(total: int) -> dict[str, int]:
    """A visit_count totalling ``total`` spread across the real agents so no single
    agent reaches the per-agent loop-guard cap.

    Three ceilings can trip (total / one real agent / the supervisor key). To
    exercise the *total*-based soft limit we must keep every per-agent count under
    ``_AGENT_HARD_CAP`` while the running total climbs — which requires spreading
    over several agents. Realistic: a real thread visits several agents
    (literature → design → IC → data → writing → review).

    Deliberately does NOT put anything under the ``supervisor`` key: that key is
    excluded from the per-agent dimension, so including it would let a test pass
    for the wrong reason.
    """
    assert total <= _AGENT_HARD_CAP * len(_AGENT_NAMES), \
        "cannot spread without tripping the per-agent cap"
    vc: dict[str, int] = {name: 0 for name in _AGENT_NAMES}
    for i in range(total):
        vc[_AGENT_NAMES[i % len(_AGENT_NAMES)]] += 1
    return {k: v for k, v in vc.items() if v}


#: A visit_count that exceeds the TOTAL cap while leaving both other dimensions
#: comfortably inside theirs — so a test asserting on the total gate cannot pass
#: because a different gate happened to fire. Agents carry one less than their
#: cap; the supervisor key makes up the rest (that is also its real shape — the
#: supervisor is re-entered on every hop, so it is roughly half of any total).
def _visit_count_over_total_cap() -> dict[str, int]:
    per = _AGENT_HARD_CAP - 1
    vc = {name: per for name in _AGENT_NAMES}
    need = (_HOP_HARD_CAP + 1) - per * len(_AGENT_NAMES)
    if need > 0:
        assert need <= _SUPERVISOR_HARD_CAP, \
            "cannot exceed the total cap without tripping another dimension"
        vc[_SUPERVISOR_KEY] = need
    return vc


# ════════════════════════════════════════════════════════════════════
# Sanity on the constants — every relationship the guards depend on
# ════════════════════════════════════════════════════════════════════

class TestThresholdConstants:
    def test_soft_threshold_is_80pct_of_hard_cap(self):
        assert _HOP_SOFT_THRESHOLD == int(_HOP_HARD_CAP * 0.8)

    def test_soft_threshold_below_hard_cap(self):
        # The soft warning must trip strictly before the hard gate.
        assert _HOP_SOFT_THRESHOLD < _HOP_HARD_CAP

    def test_agent_soft_threshold_below_agent_hard_cap(self):
        assert _AGENT_SOFT_THRESHOLD < _AGENT_HARD_CAP

    def test_supervisor_ceiling_is_far_above_the_agent_one(self):
        """The whole reason the per-agent cap misfired before: the supervisor is
        re-entered on EVERY hop, so it is structurally the largest key. Sharing the
        agents' ceiling with it made the ceiling unreachable by real agents and
        trivially reachable by routing alone."""
        assert _SUPERVISOR_HARD_CAP > _AGENT_HARD_CAP * 2

    def test_the_roster_can_actually_reach_the_total_cap(self):
        """If every real agent capped out and the supervisor capped out and the sum
        were still under the total cap, the total dimension would be dead code."""
        reachable = _AGENT_HARD_CAP * len(_AGENT_NAMES) + _SUPERVISOR_HARD_CAP
        assert reachable > _HOP_HARD_CAP


class TestGuardDimensions:
    """`_guard_dimensions` is the single reader all three ceilings share, so the
    one thing that must never regress is its treatment of the supervisor key."""

    def test_supervisor_is_not_counted_as_an_agent(self):
        total, worst, name, sup = _guard_dimensions({_SUPERVISOR_KEY: 25,
                                                    "literature": 3})
        assert total == 28            # total DOES include it
        assert (worst, name) == (3, "literature")   # per-agent does NOT
        assert sup == 25

    def test_supervisor_only_state_has_no_worst_agent(self):
        total, worst, name, sup = _guard_dimensions({_SUPERVISOR_KEY: 9})
        assert (total, worst, name, sup) == (9, 0, "", 9)

    def test_empty_is_all_zeros(self):
        assert _guard_dimensions({}) == (0, 0, "", 0)

    def test_tolerates_none_values(self):
        # visit_count arriving with a None value must not crash the guard —
        # a routing decision is not the place to raise over bookkeeping.
        assert _guard_dimensions({"literature": None, _SUPERVISOR_KEY: 2}) == (
            2, 0, "literature", 2)


# ════════════════════════════════════════════════════════════════════
# Pure helper: _budget_softlimit_warning fires correctly (no graph)
# ════════════════════════════════════════════════════════════════════

class TestSoftWarningHelper:
    def test_no_warning_well_below_threshold(self):
        st = _base_state(visit_count={"literature": 5})
        assert _budget_softlimit_warning(st, total=5) is None

    def test_no_warning_just_below_threshold(self):
        n = _HOP_SOFT_THRESHOLD - 1
        st = _base_state(visit_count={"a": n})
        assert _budget_softlimit_warning(st, total=n) is None

    def test_warns_at_threshold(self):
        n = _HOP_SOFT_THRESHOLD
        st = _base_state(visit_count={"a": n})
        msg = _budget_softlimit_warning(st, total=n)
        assert isinstance(msg, AIMessage)
        assert _SOFTLIMIT_MARKER in msg.content
        assert str(n) in msg.content and str(_HOP_HARD_CAP) in msg.content

    def test_warns_above_threshold(self):
        n = _HOP_SOFT_THRESHOLD + 6
        st = _base_state(visit_count={"a": n})
        assert _budget_softlimit_warning(st, total=n) is not None

    def test_per_agent_dimension_warns_again(self):
        """Restored 2026-07-30. One agent re-entered over and over is the shape of a
        revision ping-pong, and the TOTAL can sit far below its ceiling while it
        happens — so without this dimension that loop is invisible until it is
        expensive."""
        st = _base_state(visit_count={"paper_writing": _AGENT_SOFT_THRESHOLD})
        msg = _budget_softlimit_warning(
            st, total=_AGENT_SOFT_THRESHOLD,
            worst_agent=_AGENT_SOFT_THRESHOLD, worst_agent_name="paper_writing")
        assert msg is not None
        assert "paper_writing" in msg.content, \
            "the warning must name the agent — 'some agent is looping' is unactionable"

    def test_per_agent_dimension_quiet_below_its_threshold(self):
        n = _AGENT_SOFT_THRESHOLD - 1
        st = _base_state(visit_count={"paper_writing": n})
        assert _budget_softlimit_warning(
            st, total=n, worst_agent=n, worst_agent_name="paper_writing") is None

    def test_supervisor_alone_does_not_trigger_the_agent_dimension(self):
        """The exact misfire that got the old cap deleted: the supervisor's count is
        large on every healthy run. `_guard_dimensions` excludes it, so a state
        where ONLY the supervisor is high must stay quiet on this dimension."""
        n = _AGENT_SOFT_THRESHOLD + 5
        vc = {_SUPERVISOR_KEY: n}
        total, worst, worst_name, _sup = _guard_dimensions(vc)
        st = _base_state(visit_count=vc)
        assert _budget_softlimit_warning(
            st, total=total, worst_agent=worst, worst_agent_name=worst_name) is None

    def test_idempotent_when_marker_already_present(self):
        # Already-warned stream → helper returns None even past threshold.
        n = _HOP_SOFT_THRESHOLD + 6
        prior = AIMessage(content=f"{_SOFTLIMIT_MARKER} 软限警告:earlier")
        st = _base_state(visit_count={"a": n}, messages=[prior])
        assert _budget_softlimit_warning(st, total=n) is None

    def test_dollar_dimension_removed_even_when_initial_seeded(self):
        # Review 2026-06-10: the "$ budget soft limit" was REMOVED — it keyed off
        # budget_initial_usd, which is NOT a declared MASTState channel and so was
        # dropped on every real graph hop (dead code). Even when the caller passes
        # a raw dict carrying both fields, the dollar dimension no longer warns;
        # only the (real, channel-backed) hop/per-agent dimensions can. With a tiny
        # total here, no warning fires.
        st = _base_state(
            visit_count={"a": 2},
            budget_remaining_usd=0.15,
            budget_initial_usd=1.0,
        )
        assert _budget_softlimit_warning(st, total=2) is None

    def test_dollar_dimension_silent_above_20pct(self):
        # Still silent (the dimension is gone) — kept to document the removal.
        st = _base_state(
            visit_count={"a": 2},
            budget_remaining_usd=0.50,
            budget_initial_usd=1.0,
        )
        assert _budget_softlimit_warning(st, total=2) is None

    def test_dollar_dimension_silent_without_initial_reference(self):
        # No budget_initial_usd → dollar dimension is gone; only the hop dimension
        # can warn (and here total is tiny).
        st = _base_state(visit_count={"a": 2}, budget_remaining_usd=0.05)
        assert _budget_softlimit_warning(st, total=2) is None

    def test_already_soft_warned_tolerates_nonstring_content(self):
        # Multimodal content (list) must not crash the dedup scan.
        weird = AIMessage(content=[{"type": "text", "text": "hi"}])
        assert _already_soft_warned([weird]) is False


# ════════════════════════════════════════════════════════════════════
# Through the supervisor_node (real node, no LLM)
# ════════════════════════════════════════════════════════════════════

class TestSupervisorInjectsSoftWarning:
    def test_warning_injected_on_normal_no_model_path(self):
        # Between the soft threshold and the hard cap → soft warn but NOT hard
        # gate. Spread across agents so the per-agent cap is not what trips. No
        # model → the node still routes to END but the warning rides the messages.
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(visit_count=_spread_visit_count(_SOFT_TOTAL)))
        assert cmd.goto == END  # no-model path still ends, hard gate untouched
        warned = _msgs_with_marker(cmd.update["messages"])
        assert len(warned) == 1, "soft-limit warning must be injected exactly once"
        # visit_count delta semantics preserved (supervisor +1, no double count).
        assert cmd.update["visit_count"] == {"supervisor": 1}

    def test_warning_injected_on_hint_dispatch_path(self):
        # With a valid routing hint the node dispatches to the agent (no LLM).
        # The soft warning must still ride along on that path.
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(visit_count=_spread_visit_count(_SOFT_TOTAL),
                               routing_hints=["paper_writing"]))
        assert cmd.goto == "paper_writing"
        assert cmd.update["routing_hints"] is None
        warned = _msgs_with_marker(cmd.update["messages"])
        assert len(warned) == 1
        # hint path's visit_count semantics untouched (supervisor +1 only).
        assert cmd.update["visit_count"] == {"supervisor": 1}

    def test_warning_rides_the_parallel_fanout_path_too(self):
        """The fan-out branch (2+ hints → Send list) must carry the soft warning
        into EVERY branch — Send passes an explicit payload, so a warning that
        only went into the channel update would be invisible to the agents."""
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(visit_count=_spread_visit_count(_SOFT_TOTAL),
                               routing_hints=["paper_writing", "data_processing"]))
        assert isinstance(cmd.goto, list) and len(cmd.goto) == 2
        assert len(_msgs_with_marker(cmd.update["messages"])) == 1
        for send in cmd.goto:
            assert _msgs_with_marker(send.arg["messages"]), \
                "each fanned-out branch must see the soft-limit warning"

    def test_no_warning_below_threshold(self):
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(visit_count=_spread_visit_count(10)))
        assert _msgs_with_marker(cmd.update["messages"]) == []

    def test_not_reinjected_when_already_in_stream(self):
        # Prior warning present → supervisor must NOT add a second one.
        node = _supervisor_node_factory(supervisor_model=None)
        prior = AIMessage(content=f"{_SOFTLIMIT_MARKER} 软限警告:earlier")
        cmd = node(_base_state(visit_count=_spread_visit_count(_SOFT_TOTAL),
                               messages=[prior]))
        # Only the messages the node ADDS this step are in cmd.update["messages"].
        assert _msgs_with_marker(cmd.update["messages"]) == []

    def test_hard_gate_still_wins_at_cap(self):
        # Over the TOTAL cap while BOTH other dimensions stay inside theirs, so the
        # total gate is demonstrably the cause. The soft warning must NOT appear
        # (the hard-gate branch returns before soft injection).
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(visit_count=_visit_count_over_total_cap()))
        assert cmd.goto == END
        body = cmd.update["messages"][0].content
        assert "Loop guard tripped" in body
        assert "总跳数" in body, "the message must say WHICH ceiling ended the run"
        assert _msgs_with_marker(cmd.update["messages"]) == []

    def test_per_agent_hard_gate_ends_the_run(self):
        """Restored dimension, hard side: one agent past its own ceiling ends the run
        even though the total is nowhere near its cap. That combination — modest
        total, one agent hammered — is precisely a PW⇄PR revision loop."""
        node = _supervisor_node_factory(supervisor_model=None)
        vc = {"paper_writing": _AGENT_HARD_CAP + 1}
        assert sum(vc.values()) <= _HOP_HARD_CAP, "total must NOT be the cause here"
        cmd = node(_base_state(visit_count=vc))
        assert cmd.goto == END
        body = cmd.update["messages"][0].content
        assert "Loop guard tripped" in body and "paper_writing" in body

    def test_supervisor_hard_gate_ends_the_run(self):
        """The supervisor's own ceiling is the ping-pong detector: the $30.66 runaway
        showed up as ~34 routing hops. Its count must NOT be judged by the agents'
        ceiling (that misfire is why the cap was deleted once), but it must still
        have one of its own."""
        node = _supervisor_node_factory(supervisor_model=None)
        vc = {_SUPERVISOR_KEY: _SUPERVISOR_HARD_CAP + 1}
        assert sum(vc.values()) <= _HOP_HARD_CAP, "total must NOT be the cause here"
        cmd = node(_base_state(visit_count=vc))
        assert cmd.goto == END
        assert "Loop guard tripped" in cmd.update["messages"][0].content

    def test_a_healthy_supervisor_count_does_not_end_the_run(self):
        """Guard against over-correcting: a clean six-stage pipeline puts the
        supervisor around 7, and a legitimate 4-agent run with a revision cycle
        around 16. Neither may be ended."""
        node = _supervisor_node_factory(supervisor_model=None, wired_agents=("literature",))
        cmd = node(_base_state(visit_count={_SUPERVISOR_KEY: 16, "literature": 4},
                               routing_hints=["literature"]))
        assert cmd.goto == "literature", "a legitimate run was killed by the guards"

    def test_hard_budget_gate_still_wins(self):
        # budget_remaining_usd <= 0 still ENDs immediately, regardless of soft.
        # Low visit_count keeps the visit guards quiet so the BUDGET gate is the
        # demonstrable cause (and the soft warning is suppressed by it).
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node(_base_state(visit_count=_spread_visit_count(_SOFT_TOTAL),
                               budget_remaining_usd=0.0))
        assert cmd.goto == END
        assert "Budget exhausted" in cmd.update["messages"][0].content
        assert _msgs_with_marker(cmd.update["messages"]) == []


# ════════════════════════════════════════════════════════════════════
# The constant IS the gate — the defect this retune was built around
# ════════════════════════════════════════════════════════════════════

class TestConstantIsTheRealGate:
    """Until 2026-07-30 the live gate compared against a literal ``40`` while
    ``_HOP_HARD_CAP`` fed only the soft warning. Retuning the constant therefore
    changed NOTHING, and every comment and test in the tree read as though it had.

    These tests move the constants and assert the behaviour follows. They are the
    only kind that can catch a re-introduced literal: a test that repeats the
    number is satisfied by the bug.
    """

    def test_total_gate_follows_the_constant(self, monkeypatch):
        lowered = 6
        monkeypatch.setattr(_orch_graph, "_HOP_HARD_CAP", lowered)
        node = _supervisor_node_factory(supervisor_model=None,
                                        wired_agents=("literature",))
        # Under the ORIGINAL cap this total is unremarkable; under the lowered one
        # it must end the run. Spread so no per-agent ceiling is involved.
        vc = _spread_visit_count(lowered + 1)
        cmd = node(_base_state(visit_count=vc, routing_hints=["literature"]))
        assert cmd.goto == END, (
            "the hard gate ignored _HOP_HARD_CAP — it is comparing against a "
            "literal again (the exact defect fixed on 2026-07-30)")
        assert str(lowered) in cmd.update["messages"][0].content

    def test_per_agent_gate_follows_the_constant(self, monkeypatch):
        lowered = 3
        monkeypatch.setattr(_orch_graph, "_AGENT_HARD_CAP", lowered)
        node = _supervisor_node_factory(supervisor_model=None,
                                        wired_agents=("literature",))
        cmd = node(_base_state(visit_count={"literature": lowered + 1},
                               routing_hints=["literature"]))
        assert cmd.goto == END
        assert str(lowered) in cmd.update["messages"][0].content

    def test_supervisor_gate_follows_the_constant(self, monkeypatch):
        lowered = 4
        monkeypatch.setattr(_orch_graph, "_SUPERVISOR_HARD_CAP", lowered)
        node = _supervisor_node_factory(supervisor_model=None,
                                        wired_agents=("literature",))
        cmd = node(_base_state(visit_count={_SUPERVISOR_KEY: lowered + 1},
                               routing_hints=["literature"]))
        assert cmd.goto == END
        assert str(lowered) in cmd.update["messages"][0].content


# ════════════════════════════════════════════════════════════════════
# The USD gate is finally CONNECTED to something (2026-07-30)
# ════════════════════════════════════════════════════════════════════

class TestBudgetProbeRefreshesTheGate:
    """The gate shipped in as "caller-seeded" and an audit found no
    caller anywhere in the tree ever seeded it — inert for its entire life while a
    measured run spent $30.66 on a loop of SUCCESSES (invisible to StallGuard,
    which keys off repeated failures). ``budget_probe`` is the connection.
    """

    def test_probe_can_end_a_run_that_state_thought_was_solvent(self):
        # State says $5 left; the probe — which reads real spend — says none. The
        # probe must win, otherwise the gate is one hop stale at all times.
        node = _supervisor_node_factory(supervisor_model=None,
                                        wired_agents=("literature",),
                                        budget_probe=lambda: 0.0)
        cmd = node(_base_state(visit_count={"literature": 1},
                               budget_remaining_usd=5.0,
                               routing_hints=["literature"]))
        assert cmd.goto == END
        assert "Budget exhausted" in cmd.update["messages"][0].content

    def test_probe_writes_the_refreshed_value_through_dispatch(self):
        node = _supervisor_node_factory(supervisor_model=None,
                                        wired_agents=("literature",),
                                        budget_probe=lambda: 3.25)
        cmd = node(_base_state(visit_count={"literature": 1},
                               budget_remaining_usd=8.0,
                               routing_hints=["literature"]))
        assert cmd.goto == "literature"
        assert cmd.update["budget_remaining_usd"] == 3.25, \
            "the refreshed budget must ride the dispatch, or the UI and the next " \
            "hop both read a stale number"

    def test_probe_returning_none_leaves_the_gate_alone(self):
        """None means 'spend could not be read'. Treating it as 0 would let a
        billing hiccup end a healthy run; treating it as infinite would disable the
        ceiling. It must mean 'do not touch'."""
        node = _supervisor_node_factory(supervisor_model=None,
                                        wired_agents=("literature",),
                                        budget_probe=lambda: None)
        cmd = node(_base_state(visit_count={"literature": 1},
                               budget_remaining_usd=8.0,
                               routing_hints=["literature"]))
        assert cmd.goto == "literature"
        assert "budget_remaining_usd" not in cmd.update or \
            cmd.update["budget_remaining_usd"] == 8.0

    def test_a_raising_probe_does_not_take_the_run_down(self):
        def _boom():
            raise RuntimeError("ledger locked")

        node = _supervisor_node_factory(supervisor_model=None,
                                        wired_agents=("literature",),
                                        budget_probe=_boom)
        cmd = node(_base_state(visit_count={"literature": 1},
                               routing_hints=["literature"]))
        assert cmd.goto == "literature", "billing must never break routing"

    def test_no_probe_behaves_exactly_as_before(self):
        # Absent probe + unseeded state → gate inert, run proceeds. This is the
        # pre-2026-07-30 behaviour and must be preserved for tests/library users.
        node = _supervisor_node_factory(supervisor_model=None,
                                        wired_agents=("literature",))
        cmd = node(_base_state(visit_count={"literature": 1},
                               routing_hints=["literature"]))
        assert cmd.goto == "literature"

    def test_an_unseeded_state_is_not_treated_as_zero_budget(self):
        """`budget_remaining_usd` must stay a NotRequired channel with NO reducer.
        Adding one (tried 2026-07-30) initialises it to 0.0, and for a `<= 0` gate
        that converts the honest inert default into an unconditional kill switch —
        every run ended on its first hop. Verified through a real compiled graph,
        because that default only materialises inside LangGraph's channel setup."""
        seen: list[Any] = []

        def _probe_state(state: MASTState):
            seen.append(state.get("budget_remaining_usd"))
            return Command(goto=END, update={})

        g: StateGraph = StateGraph(MASTState)
        g.add_node("peek", _probe_state)
        g.add_edge(START, "peek")
        g.compile().invoke(_base_state(visit_count={}))
        assert seen == [None], (
            f"budget_remaining_usd defaulted to {seen[0]!r} instead of being "
            "absent — a reducer was added to it; see the field's comment in "
            "agents/state.py")


# ════════════════════════════════════════════════════════════════════
# End-to-end through a real compiled graph: warn once, then hard-gate END
# ════════════════════════════════════════════════════════════════════

class TestEndToEndWarnOnceThenHardGate:
    """A minimal parent graph where each agent hands off to the NEXT agent in a
    6-agent round-robin. Cycling all six keeps every per-agent count well under
    the per-agent cap (8) while the running total climbs through the soft
    threshold (32) and on to the total hard cap (40). This is the only way to
    actually exercise the *total*-based soft limit end-to-end — a 2-agent
    ping-pong would trip the per-agent guard (8) at total ≈18, long before 32.

    Asserts the soft warning lands EXACTLY once across the whole run (fire-once
    idempotency through the real add_messages reducer + checkpointer), and the
    run is still bounded by the hard loop guard (END), never a recursion error.
    """

    def _agent_node(self, me: str, nxt: str):
        def _node(state: MASTState) -> Command:
            # Mimic a handoff: route back through supervisor with a hint at the
            # next agent and a +1 visit_count delta for it (sum_int_dicts adds).
            return Command(
                goto="supervisor",
                update={
                    "messages": [AIMessage(content=f"{me}->{nxt}")],
                    "routing_hints": [nxt],
                    "visit_count": {nxt: 1},
                },
            )
        return _node

    def _build(self):
        supervisor = _supervisor_node_factory(supervisor_model=None)
        g: StateGraph = StateGraph(MASTState)
        g.add_node("supervisor", supervisor)
        n = len(_AGENT_NAMES)
        for i, name in enumerate(_AGENT_NAMES):
            nxt = _AGENT_NAMES[(i + 1) % n]  # round-robin → next agent
            g.add_node(name, self._agent_node(name, nxt))
        g.add_edge(START, "supervisor")
        return g.compile()

    def test_warned_exactly_once_and_bounded_by_hard_gate(self):
        app = self._build()
        result = app.invoke(
            _base_state(routing_hints=[_AGENT_NAMES[0]]),
            config={"configurable": {"thread_id": "softlimit-e2e-1"},
                    "recursion_limit": 300},
        )
        texts = [getattr(m, "content", "") for m in result["messages"]]
        # Soft warning fired exactly once across the entire run (idempotent —
        # the marker scan over add_messages history blocks any re-injection).
        n_soft = sum(1 for t in texts if isinstance(t, str) and _SOFTLIMIT_MARKER in t)
        assert n_soft == 1, f"expected exactly one soft-limit warning, got {n_soft}"
        # Hard loop guard still terminated the run (NOT a recursion error).
        assert any("Loop guard tripped" in t for t in texts), \
            "hard loop guard must still END the run"
        # The soft warning preceded the hard-gate termination in the stream.
        soft_idx = next(i for i, t in enumerate(texts)
                        if isinstance(t, str) and _SOFTLIMIT_MARKER in t)
        guard_idx = next(i for i, t in enumerate(texts)
                         if isinstance(t, str) and "Loop guard tripped" in t)
        assert soft_idx < guard_idx, "soft warning must arrive before the hard gate"
        # Run is bounded (hard cap is 40 hop-total; supervisor re-runs each hop).
        assert sum(result["visit_count"].values()) <= 80


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-p", "no:randomly"])
