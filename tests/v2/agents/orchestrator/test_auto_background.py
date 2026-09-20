"""Conservative auto-backgrounding — the supervisor detaches an INDEPENDENT
literature survey paired with instrument_control to a background run so the fast
instrument turn isn't barrier-blocked.

Policy is deliberately narrow + PREFER-MISSING, pinned here:
  * whitelist is ONLY {literature} (a background run has isolated state, so
    paper_*/data_processing would run against empty state → excluded);
  * fires ONLY when instrument_control is in the SAME batch (foreground hardware
    to protect) AND the gate is ON;
  * instrument_control is NEVER backgroundable;
  * gate OFF (default) → routing byte-for-byte unchanged;
  * an operator @agent direction / agent handoff hint is never auto-backgrounded
    (different branch) — that is the override.
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
from langgraph.graph import START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from mast.agents.orchestrator.graph import (  # noqa: E402
    _AUTO_BG_MARKER,
    _split_auto_background,
    _supervisor_node_factory,
)
from mast.agents.state import MASTState  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# 1. _split_auto_background — the pure policy
# ═══════════════════════════════════════════════════════════════════════════

_ON = lambda: True   # noqa: E731
_OFF = lambda: False  # noqa: E731


class TestSplitPolicy:
    def test_no_gate_is_off(self):
        assert _split_auto_background(["instrument_control", "literature"], None) == (
            ["instrument_control", "literature"], [])

    def test_gate_off_is_unchanged(self):
        assert _split_auto_background(["instrument_control", "literature"], _OFF) == (
            ["instrument_control", "literature"], [])

    def test_literature_paired_with_ic_is_backgrounded(self):
        fg, bg = _split_auto_background(["instrument_control", "literature"], _ON)
        assert fg == ["instrument_control"] and bg == ["literature"]

    def test_only_whitelisted_agent_is_peeled(self):
        # data_processing is NOT whitelisted (needs the saved scan → stays foreground)
        fg, bg = _split_auto_background(
            ["instrument_control", "literature", "data_processing"], _ON)
        assert fg == ["instrument_control", "data_processing"] and bg == ["literature"]

    def test_literature_alone_is_not_backgrounded(self):
        # no instrument_control ⇒ literature IS the foreground ⇒ nothing to detach
        assert _split_auto_background(["literature"], _ON) == (["literature"], [])

    def test_no_whitelisted_agent_no_background(self):
        assert _split_auto_background(["instrument_control", "data_processing"], _ON) == (
            ["instrument_control", "data_processing"], [])

    def test_instrument_control_is_never_backgrounded(self):
        for targets in (
            ["instrument_control"],
            ["instrument_control", "literature"],
            ["instrument_control", "literature", "paper_writing"],
        ):
            _fg, bg = _split_auto_background(targets, _ON)
            assert "instrument_control" not in bg

    def test_gate_exception_fails_safe_to_foreground(self):
        def _boom():
            raise RuntimeError("settings glitch")

        assert _split_auto_background(["instrument_control", "literature"], _boom) == (
            ["instrument_control", "literature"], [])


class TestDurationRefinement:
    """item ② — the optional duration advisor refines (never breaks) the split.

    Advisor ``(agent_type) -> bool|None``: True=slow (detach), False=fast (keep
    foreground), None=too few samples (prefer-missing → whitelist decides).
    """

    _T = ["instrument_control", "literature"]

    def test_none_advisor_is_byte_for_byte_the_whitelist_policy(self):
        # the default (no advisor) must equal the pre-② behaviour
        assert _split_auto_background(self._T, _ON, None) == (
            ["instrument_control"], ["literature"])

    def test_slow_whitelisted_agent_is_detached(self):
        fg, bg = _split_auto_background(self._T, _ON, lambda t: True)
        assert fg == ["instrument_control"] and bg == ["literature"]

    def test_proven_fast_whitelisted_agent_is_kept_foreground(self):
        # advisor confidently says literature is FAST → not worth detaching → the
        # normal fan-out runs (nothing backgrounded)
        assert _split_auto_background(self._T, _ON, lambda t: False) == (self._T, [])

    def test_unknown_samples_falls_back_to_whitelist_detach(self):
        # None (too few samples) is prefer-missing → behaves like the whitelist
        fg, bg = _split_auto_background(self._T, _ON, lambda t: None)
        assert fg == ["instrument_control"] and bg == ["literature"]

    def test_advisor_never_promotes_a_non_whitelisted_agent(self):
        # even if data_processing is "slow", it is state-unsafe (not whitelisted)
        # and must NEVER be detached — the advisor only prunes, never promotes.
        fg, bg = _split_auto_background(
            ["instrument_control", "data_processing"], _ON, lambda t: True)
        assert bg == [] and fg == ["instrument_control", "data_processing"]

    def test_advisor_exception_fails_safe_to_detach(self):
        def _boom(_t):
            raise RuntimeError("stats glitch")
        # an advisor error is treated as unknown → whitelist detach (never drops work)
        fg, bg = _split_auto_background(self._T, _ON, _boom)
        assert fg == ["instrument_control"] and bg == ["literature"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. supervisor_node — the real routing node, gate on vs off
# ═══════════════════════════════════════════════════════════════════════════

class _Router:
    def __init__(self, agents):
        self._agents = list(agents)

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _msgs):
                return {"next_agents": list(outer._agents), "reason": "扫描与文献互不依赖"}
        return _S()

    def invoke(self, _msgs):
        return AIMessage(content="ok")


def _stub(name: str, ran: list):
    def node(state):
        ran.append(name)
        return Command(goto="supervisor",
                       update={"messages": [AIMessage(content=f"{name} ran")],
                               "active_agent": "supervisor"})
    return node


def _run(agents_out, gate, ran, thread):
    g: StateGraph = StateGraph(MASTState)
    g.add_node("supervisor", _supervisor_node_factory(
        _Router(agents_out), wired_agents=("instrument_control", "literature"),
        background_gate=gate))
    g.add_node("instrument_control", _stub("instrument_control", ran))
    g.add_node("literature", _stub("literature", ran))
    g.add_edge(START, "supervisor")
    app = g.compile(checkpointer=InMemorySaver())
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="扫一张图，同时查文献")],
        "executed_skills": [], "scan_paths": [], "scan_metadata": {},
        "error_log": [], "event_refs": [], "visit_count": {}, "pending_approvals": {},
    }
    # cap hops: the stubs bounce back to supervisor which will re-route; the
    # scripted router keeps returning the same set, so bound via recursion_limit
    # and just inspect the FIRST dispatch's effect.
    try:
        out = app.invoke(state, config={"configurable": {"thread_id": thread},
                                        "recursion_limit": 6})
    except Exception:
        # a runaway (router keeps dispatching) still leaves the checkpoint with the
        # messages we assert on; read them back.
        out = app.get_state({"configurable": {"thread_id": thread}}).values
    return out


def test_gate_on_backgrounds_literature_and_runs_only_ic_foreground():
    ran: list = []
    out = _run(["instrument_control", "literature"], _ON, ran, "abg-on")
    texts = [str(getattr(m, "content", "")) for m in out.get("messages", [])]
    # the marker was emitted for literature
    assert any(_AUTO_BG_MARKER in t and "literature" in t for t in texts), texts
    # literature did NOT run in THIS graph (it was peeled off to a background run);
    # instrument_control did.
    assert "instrument_control" in ran
    assert "literature" not in ran, "literature ran foreground despite being backgrounded"
    # the foreground dispatch note names only instrument_control
    assert any("[SUPERVISOR → instrument_control]" in t for t in texts)


def test_gate_off_runs_the_normal_fanout():
    ran: list = []
    out = _run(["instrument_control", "literature"], _OFF, ran, "abg-off")
    texts = [str(getattr(m, "content", "")) for m in out.get("messages", [])]
    # NO marker, and BOTH agents ran (unchanged parallel fan-out)
    assert not any(_AUTO_BG_MARKER in t for t in texts)
    assert "instrument_control" in ran and "literature" in ran


# ═══════════════════════════════════════════════════════════════════════════
# 3. marker parity — the bridge parses the SAME literal the supervisor emits
# ═══════════════════════════════════════════════════════════════════════════

def test_marker_parity_graph_and_bridge():
    from mast.api.routes.orchestrator import (
        _AUTO_BG_MARKER as bridge_marker,
        _parse_auto_background,
    )
    assert bridge_marker == _AUTO_BG_MARKER
    assert _parse_auto_background(f"{_AUTO_BG_MARKER} literature :: 理由") == ["literature"]
    assert _parse_auto_background(f"{_AUTO_BG_MARKER} literature,paper_writing :: x") == [
        "literature", "paper_writing"]
    assert _parse_auto_background("[SUPERVISOR → instrument_control] 普通派发") == []
    assert _parse_auto_background("random text") == []
