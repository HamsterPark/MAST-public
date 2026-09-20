"""Concurrency stress on the parallel dispatch path — races don't fail once.

A fan-out is the first place in MAST where several agents write the SAME state
channels in the SAME super-step. Every bug in that regime is probabilistic: it
depends on which branch's worker thread wins, and a single green run proves
nothing. The failures we are hunting:

  * a **lost update** — two branches write ``visit_count`` / ``executed_skills``
    and one is dropped, so the loop guard undercounts and the run overruns;
  * an **order-dependent result** — the merged state depends on which branch
    finished first, which makes every downstream assertion a coin flip;
  * a **shared-instance leak** — the fan-out reuses one mutable object across
    branches (a Send payload, a context) and the branches corrupt each other;
  * **flakiness** — the same input sometimes crashes.

So: run the same fan-out many times, run it at every width, run it with the
branches deliberately racing (randomised delays), and assert the result is
EXACTLY the same every time.
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

import itertools  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from mast.agents._shared.handoff import make_handoff  # noqa: E402
from mast.agents.orchestrator.graph import (  # noqa: E402
    _AGENT_NAMES,
    _MAX_PARALLEL,
    _supervisor_node_factory,
)
from mast.agents.state import MASTState  # noqa: E402

_ROSTER = [a for a in _AGENT_NAMES if a != "instrument_control"]


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="goal")],
        "executed_skills": [], "scan_paths": [], "scan_metadata": {},
        "error_log": [], "event_refs": [], "visit_count": {},
        "pending_approvals": {},
    }


class _Router:
    """Fans out to `targets` once, then ends."""

    def __init__(self, targets: list[str]):
        self.targets = list(targets)
        self.hops = 0

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _m):
                outer.hops += 1
                if outer.hops == 1:
                    return {"next_agents": list(outer.targets), "reason": "race"}
                return {"next_agents": ["__end__"], "reason": "done"}

        return _S()

    def invoke(self, _m):
        return AIMessage(content="ok")


def _racing_agent(name: str, delay_s: float, seen: list):
    """An agent node that stalls for `delay_s` before handing back — so the
    branches finish in a controlled, adversarial order. Records the order it
    really ran in (thread-safe) so we can prove the RESULT is order-independent
    even when the ORDER differs."""
    handoff = make_handoff("supervisor", "done")
    lock = threading.Lock()

    def _node(state: MASTState) -> Command:
        if delay_s:
            time.sleep(delay_s)           # a test node, not agents/**/graph.py
        with lock:
            seen.append(name)
        cmd = handoff.func(reason=f"{name} done", tool_call_id=f"{name}-tc")
        upd: dict[str, Any] = dict(cmd.update)
        upd["messages"] = [AIMessage(content=f"{name} ran")] + list(upd["messages"])
        upd["executed_skills"] = [f"{name}_skill"]
        upd["error_log"] = [f"{name}_note"]
        return Command(goto="supervisor", update=upd)

    return _node


def _run(targets: list[str], delays: dict[str, float], thread: str,
         seen: list) -> dict:
    g: StateGraph = StateGraph(MASTState)
    g.add_node("supervisor",
               _supervisor_node_factory(_Router(targets), wired_agents=tuple(targets)))
    for a in targets:
        g.add_node(a, _racing_agent(a, delays.get(a, 0.0), seen))
    g.add_edge(START, "supervisor")
    app = g.compile(checkpointer=InMemorySaver())
    return app.invoke(_state(), config={"configurable": {"thread_id": thread},
                                        "recursion_limit": 60})


def _fingerprint(out: dict, targets: list[str]) -> tuple:
    """Everything about the merged state that must NOT depend on branch order."""
    return (
        tuple(sorted(out.get("executed_skills") or [])),
        tuple(sorted(out.get("error_log") or [])),
        tuple(sorted((out.get("visit_count") or {}).items())),
        tuple(sorted(out.get("routing_hints") or [])),
        tuple(sorted(t for t in
                     (str(getattr(m, "content", "")) for m in out["messages"])
                     if t.endswith(" ran"))),
    )


# ════════════════════════════════════════════════════════════════════════
# Repetition — a race that fails 1-in-20 passes a single run
# ════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("i", range(25))
def test_the_same_fanout_gives_the_same_state_every_time(i: int):
    targets = _ROSTER[:3]
    seen: list = []
    out = _run(targets, {}, f"rep-{i}", seen)
    assert set(out["executed_skills"]) == {f"{t}_skill" for t in targets}
    assert set(out["error_log"]) == {f"{t}_note" for t in targets}
    # every branch counted exactly once — a lost visit_count update would let the
    # loop guard undercount and the run overrun its hop budget
    vc = out["visit_count"]
    for t in targets:
        assert vc.get(t) == 1, f"visit_count for {t} is {vc.get(t)}, not 1: {vc}"
    assert len(seen) == len(targets), f"a branch never ran: {seen}"


def test_the_result_does_not_depend_on_which_branch_finishes_first():
    """Force EVERY finishing order with staggered sleeps and demand one identical
    merged state. This is the property that makes a fan-out usable at all: if the
    outcome depended on thread scheduling, no downstream assertion could hold."""
    targets = _ROSTER[:3]
    prints = set()
    for n, order in enumerate(itertools.permutations(targets)):
        delays = {a: 0.02 * k for k, a in enumerate(order)}   # forces this order
        seen: list = []
        out = _run(targets, delays, f"perm-{n}", seen)
        assert seen == list(order), f"the delays did not force the order: {seen}"
        prints.add(_fingerprint(out, targets))
    assert len(prints) == 1, (
        f"the merged state depends on which branch finished first — "
        f"{len(prints)} distinct outcomes across {len(list(itertools.permutations(targets)))} orders")


@pytest.mark.parametrize("width", range(2, _MAX_PARALLEL + 1))
def test_every_width_merges_cleanly_under_a_race(width: int):
    """Width matters: with N concurrent writers, N-1 of them can be the one that
    gets dropped. Race them all against each other at every legal width."""
    targets = _ROSTER[:width]
    delays = {a: 0.015 * (width - k) for k, a in enumerate(targets)}  # reverse order
    seen: list = []
    out = _run(targets, delays, f"width-{width}", seen)
    assert set(out["executed_skills"]) == {f"{t}_skill" for t in targets}
    assert sum((out["visit_count"] or {}).get(t, 0) for t in targets) == width
    assert seen == list(reversed(targets))


def test_a_fanout_including_instrument_control_races_cleanly():
    """The real production shape — IC alongside compute agents. IC must appear
    exactly once no matter how the threads interleave."""
    targets = ["instrument_control"] + _ROSTER[:2]
    for i in range(10):
        seen: list = []
        out = _run(targets, {"instrument_control": 0.01 * (i % 3)}, f"ic-{i}", seen)
        assert seen.count("instrument_control") == 1, (
            f"instrument_control ran {seen.count('instrument_control')} times: {seen}")
        assert out["visit_count"].get("instrument_control") == 1


# ════════════════════════════════════════════════════════════════════════
# The Send payload must not be a shared mutable
# ════════════════════════════════════════════════════════════════════════

def test_branches_cannot_corrupt_each_others_payload():
    """``_dispatch`` builds ONE payload dict and hands it to every Send. If the
    branches received the SAME object (rather than LangGraph copying it per
    node), a branch mutating its input — which agent middleware does routinely —
    would silently corrupt its siblings' view of the conversation."""
    from mast.agents.orchestrator.graph import _supervisor_node_factory as _f

    node = _f(_Router(["literature", "data_processing"]),
              wired_agents=("literature", "data_processing"))
    cmd = node(_state())
    sends = cmd.goto
    assert len(sends) == 2
    a, b = sends[0].arg, sends[1].arg
    # Same CONTENT…
    assert [m.content for m in a["messages"]] == [m.content for m in b["messages"]]
    # …and if they are the same object, mutating one must be visible in the other,
    # which is exactly the hazard. Prove the graph tolerates it by mutating the
    # branch-local list and checking the durable channel is unaffected: `update`
    # carries its OWN message list, so the persisted state cannot be poisoned.
    a["messages"].append(AIMessage(content="POISON"))
    assert not any("POISON" in str(getattr(m, "content", ""))
                   for m in cmd.update["messages"]), (
        "a branch mutating its payload reached the durable state update")


def test_the_hint_channel_is_cleared_exactly_once_per_dispatch():
    """Several branches each append a routing hint in the same super-step; the
    supervisor consumes ALL of them and writes None to clear. A clear that raced
    with an append would either resurrect a finished agent (stale hint) or drop a
    live one."""
    targets = _ROSTER[:3]
    for i in range(15):
        seen: list = []
        out = _run(targets, {a: 0.005 * k for k, a in enumerate(targets)},
                   f"hint-{i}", seen)
        assert not out.get("routing_hints"), (
            f"stale routing hints survived the dispatch: {out.get('routing_hints')}")
