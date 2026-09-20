"""The inter-agent artifact channel, proven on REAL compiled subgraphs.

Why this file insists on the real orchestrator
----------------------------------------------
The defect this channel fixes was invisible to a green test suite for months,
and the reason is worth stating once, at the top, because it dictates the shape
of every test below.

An agent hands control back with ``Command(goto="supervisor",
graph=Command.PARENT)``. That SHORT-CIRCUITS out of the agent subgraph: only
``command.update`` reaches the parent, and everything the agent actually did —
its prose, its tool calls, its tool results — stays in the subgraph namespace and
is discarded. So "did the work reach the next agent?" is a question about the
PARENT/SUBGRAPH BOUNDARY, and it cannot be answered by a hand-written stub node
standing in for an agent. ``test_group_compaction_visibility.py`` measured
compaction against a hand-written orchestrator and a force-fired trigger, and
that is precisely why nobody noticed compaction's output was being thrown away
at this same boundary.

There is a second, sharper trap underneath. A ``create_agent`` subgraph compiles
with LangChain's bare ``AgentState`` unless it is given ``state_schema``. Writing
an undeclared channel from a tool is then **silently dropped — no exception, no
warning** (verified 2026-07-29). The skill adapter had been writing
``executed_skills`` / ``scan_paths`` / ``composite_progress`` into thin air for
exactly this reason. So ``test_every_agent_declares_the_state_schema`` below is
not boilerplate: without it the entire channel reverts to a no-op that every
other test in this file would still fail to notice if they used stubs.
"""
from __future__ import annotations

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        c = p / "MASTv2"
        if c.is_dir():
            return str(c)
        p = p.parent
    raise RuntimeError("MASTv2 not found above " + str(Path(__file__).resolve()))


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import inspect  # noqa: E402
import typing  # noqa: E402
from typing import Annotated  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import InjectedToolCallId, tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from mast.agents._shared import artifact_channel as ch  # noqa: E402
from mast.agents.orchestrator.graph import build as build_orchestrator  # noqa: E402
from mast.agents.state import DocRef, MASTState, sum_int_dicts  # noqa: E402

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of  # noqa: E402

# Derived, not transcribed: this list gates "every agent declares state_schema"
# and "CONSUMES names only real agents", and a hand-copy of the roster turns both
# into "today equals today" the moment an agent is added (2026-08-21,
# research_director). The import is the orchestrator's own roster, so a new agent
# is covered by these gates from the hour it is wired.
from mast.agents.orchestrator.graph import _AGENT_NAMES as _AGENTS  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════════════════

class _Rec(GenericFakeChatModel):
    """Fake chat model that records every message list it was asked to answer —
    that recording is how we prove what an agent REALLY received."""

    sink: object = None

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.sink is not None:
            self.sink.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager,
                                 **kwargs)


class _ScriptedRouter:
    """Structured-output router, same shape as test_parallel_real_graphs uses."""

    def __init__(self, script):
        self.script = list(script)
        self.hops = 0

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _msgs):
                i = min(outer.hops, len(outer.script) - 1)
                outer.hops += 1
                return {"next_agents": list(outer.script[i]), "reason": "测试路由"}

        return _S()

    def invoke(self, _msgs):
        return AIMessage(content="ok")


def _tc(name, args, tid):
    return {"name": name, "args": args, "id": tid, "type": "tool_call"}


def _make_publisher(tool_name: str, field: str, value):
    """A tool that publishes ``value`` on ``field`` — stands in for save_draft
    etc. without touching the document store."""

    @tool(tool_name)
    def _publish(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Publish a fake artifact."""
        return Command(update={
            field: value,
            "messages": [ToolMessage(content=f"published {field}",
                                     tool_call_id=tool_call_id, name=tool_name)],
        })

    return _publish


def _seed_state(goal: str = "查文献然后设计实验") -> dict:
    return {
        "messages": [HumanMessage(content=goal)],
        "executed_skills": [], "scan_paths": [], "scan_metadata": {},
        "error_log": [], "event_refs": [], "visit_count": None,
        "pending_approvals": {}, "composite_progress": {},
    }


_REPORT = DocRef(
    doc_id="rpt__abc123", version=2, kind="literature_report",
    title="NiI2 综述", path="experiments/E1/reports/rpt__abc123/v002.md",
    summary="偏压 -1.5~+1.5 V，setpoint 100 pA。", produced_by="literature")


# ═══════════════════════════════════════════════════════════════════════════
# The structural preconditions — each one, alone, silently disables the channel
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuralPreconditions:
    def test_every_agent_declares_the_state_schema(self):
        """Without a ``state_schema`` the artifact channels do not exist in the
        subgraph, and a tool writing one is dropped WITHOUT AN ERROR.

        Asserted by reading each agent's build source rather than by behaviour,
        because the failure is silent by nature: a behavioural test that forgot
        to check the channel would pass against a completely dead channel.
        """
        for name in _AGENTS:
            mod = __import__(f"mast.agents.{name}.graph", fromlist=["build"])
            src = inspect.getsource(mod)
            assert "state_schema=AgentSubState" in src, (
                f"{name}/graph.py does not pass state_schema=AgentSubState to "
                f"create_agent — every artifact write from its tools is being "
                f"silently discarded"
            )

    def test_subgraphs_do_NOT_get_the_parent_control_plane(self):
        """A subgraph must not hold the parent's loop-guard counter.

        It used to (``state_schema=MASTState``, 2026-07-29), and a subgraph that
        ends WITHOUT handing off writes its whole final state back through the
        parent's reducers — so ``visit_count``, whose reducer ADDS, was counted
        twice for one hop. Measured, then fixed by narrowing the schema.
        """
        import typing as _t

        from mast.agents.state import AgentSubState
        hints = _t.get_type_hints(AgentSubState, include_extras=True)
        for control_field in ("visit_count", "routing_hints", "active_agent",
                              "budget_remaining_usd"):
            assert control_field not in hints, (
                f"AgentSubState declares {control_field!r}, which belongs to the "
                f"parent's control plane. A no-handoff exit will merge the "
                f"subgraph's copy back into the parent's channel.")

    def test_every_subgraph_channel_is_idempotent_under_remerge(self):
        """The property that makes a no-handoff write-back harmless.

        The subgraph is SEEDED from the parent, so anything it hands back
        includes values the parent already has. Every reducer here must therefore
        absorb a repeat: ``add_messages`` de-dupes by id, ``dedupe_*`` by value,
        ``merge_dicts``/``last_wins`` overwrite. A bare ``operator.add`` would
        silently duplicate — which is exactly what ``scan_paths`` and
        ``error_log`` did until 2026-07-30.
        """
        import typing as _t

        from mast.agents.state import AgentSubState
        hints = _t.get_type_hints(AgentSubState, include_extras=True)
        forbidden = []
        for name, ann in hints.items():
            meta = getattr(ann, "__metadata__", None)
            # NotRequired[Annotated[...]] nests one level deeper.
            if meta is None:
                inner = _t.get_args(ann)
                if inner:
                    meta = getattr(inner[0], "__metadata__", None)
            for reducer in (meta or ()):
                if getattr(reducer, "__name__", "") == "add" or reducer is __import__(
                        "operator").add:
                    forbidden.append(name)
        assert not forbidden, (
            f"non-idempotent reducer on subgraph channel(s) {forbidden}: a "
            f"no-handoff exit will duplicate every entry")

    def test_every_carried_field_has_a_reducer(self):
        """The parallel-safety invariant, as an assertion instead of a comment.

        Under a fan-out several agents hand back in the SAME super-step. A key
        without a reducer is a bare LangGraph channel and raises
        ``InvalidUpdateError`` on that concurrent write — so a carried field
        missing its reducer does not fail rarely, it fails EVERY parallel run.
        """
        hints = typing.get_type_hints(MASTState, include_extras=True)
        for field in ch.CARRIED_FIELDS:
            assert field in hints, f"{field} is carried but not declared in MASTState"
            ann = hints[field]
            flat = str(ann)
            assert "last_wins" in flat or "Annotated" in flat, (
                f"MASTState.{field} is carried across the handoff but has no "
                f"reducer — every parallel fan-out that writes it will crash"
            )

    def test_consumed_fields_are_all_carried(self):
        """An agent cannot be shown a field nothing ever transports to it."""
        for agent, fields in ch.CONSUMES.items():
            for f in fields:
                assert f in ch.CARRIED_FIELDS, (
                    f"{agent} consumes {f!r}, which no handoff carries")

    def test_consumes_covers_the_real_agents_plus_the_scheduler(self):
        assert set(ch.CONSUMES) == set(_AGENTS) | {"supervisor"}

    def test_the_scheduler_sees_everything_it_could_route_on(self):
        """The supervisor was the ONE role that could not see the products, while
        being the role that decides who runs next. Its "每个阶段只走一遍" rule had
        to be a behavioural guess because it had no way to check."""
        assert set(ch.CONSUMES["supervisor"]) == set(ch.CARRIED_FIELDS)

    def test_the_router_is_actually_handed_the_block(self):
        """CONSUMES alone proves nothing: the supervisor node is a bare function
        outside the middleware stack, so the block has to be spliced into
        routing_messages by hand."""
        import mast.agents.orchestrator.graph as g
        src = inspect.getsource(g)
        assert '_render_upstream(state, "supervisor"' in src
        assert "routing_messages" in src

    def test_the_scheduler_gets_no_readback_instructions(self):
        """It holds no tools, so naming one would be the 2026-07-30 defect again."""
        block = ch.render_upstream_block(
            {"literature_report": _REPORT}, "supervisor", available_tools=set())
        assert "rpt__abc123" in block
        assert "load_document" not in block


# ═══════════════════════════════════════════════════════════════════════════
# THE falsification point: LIT's product reaches XD, through the real graph
# ═══════════════════════════════════════════════════════════════════════════

class TestProductCrossesTheParentBoundary:
    def _run(self, publisher, *, lit_sink, xd_sink):
        lit_llm = _Rec(messages=iter([
            AIMessage(content="", tool_calls=[_tc(publisher.name, {}, "l1")]),
            AIMessage(content="", tool_calls=[
                _tc("handoff_to_experiment_design", {"reason": "先验摘要已完成"}, "l2")]),
        ]), sink=lit_sink)
        xd_llm = _Rec(messages=iter([
            AIMessage(content="", tool_calls=[
                _tc("handoff_to_supervisor", {"reason": "方案已出"}, "x1")]),
        ]), sink=xd_sink)
        orch = build_orchestrator(
            buf=None,
            include_agents=("literature", "experiment_design"),
            supervisor_model=_ScriptedRouter([["literature"], ["__end__"]]),
            agent_model_overrides={"literature": lit_llm,
                                   "experiment_design": xd_llm},
            checkpointer=InMemorySaver(),
            memory_tools=[publisher],
        )
        return orch.invoke(_seed_state(),
                           config={"configurable": {"thread_id": "t-artifact"},
                                   "recursion_limit": 25})

    def test_the_pointer_reaches_the_parent_channel(self):
        lit_sink, xd_sink = [], []
        out = self._run(_make_publisher("fake_save_report", "literature_report", _REPORT),
                        lit_sink=lit_sink, xd_sink=xd_sink)
        got = out.get("literature_report")
        assert got is not None, (
            "the literature report never crossed the subgraph→parent boundary — "
            "this is the whole defect the artifact channel exists to fix")
        assert ch._get(got, "doc_id") == "rpt__abc123"
        assert ch._get(got, "version") == 2

    def test_the_next_agent_is_actually_TOLD_about_it(self):
        """Reaching the parent is necessary but not sufficient: the point is that
        the DOWNSTREAM agent sees it. This asserts against the real message list
        the model was handed, not against state."""
        lit_sink, xd_sink = [], []
        self._run(_make_publisher("fake_save_report", "literature_report", _REPORT),
                  lit_sink=lit_sink, xd_sink=xd_sink)
        assert xd_sink, "experiment_design never ran"
        system = [m for m in xd_sink[0] if m.__class__.__name__ == "SystemMessage"]
        assert system, "experiment_design got no system message at all"
        text = str(system[0].content)
        assert "rpt__abc123" in text, (
            "experiment_design was not told the report's doc_id — its prompt has "
            "always claimed a LIT summary would be available; this is what makes "
            "that true")
        assert "NiI2 综述" in text
        assert "100 pA" in text, "the key findings did not reach the consumer"

    def test_without_a_product_nothing_is_injected(self):
        """Empty must render NOTHING — no placeholder, no example. A fabricated
        artifact is strictly worse than an absent one (2026-07-27 coordinate
        incident)."""
        lit_sink, xd_sink = [], []

        @tool("noop_tool")
        def noop(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
            """Do nothing."""
            return Command(update={"messages": [
                ToolMessage(content="nothing", tool_call_id=tool_call_id,
                            name="noop_tool")]})

        self._run(noop, lit_sink=lit_sink, xd_sink=xd_sink)
        assert xd_sink
        system = [m for m in xd_sink[0] if m.__class__.__name__ == "SystemMessage"]
        text = str(system[0].content) if system else ""
        assert ch.UPSTREAM_BLOCK_HEADER.splitlines()[0] not in text, (
            "an empty artifact channel still injected its header — an operator "
            "debugging a run would be reading a section that describes nothing")


# ═══════════════════════════════════════════════════════════════════════════
# Rendering rules
# ═══════════════════════════════════════════════════════════════════════════

class TestReadbackHintsAreReal:
    """The 2026-07-30 defect class: telling an agent to call a tool it lacks.

    The channel shipped naming ``load_document`` in every document consumer's
    context. That tool existed NOWHERE in the tree. Two neighbours had the same
    shape (paper_writing shown its own draft with no reader; data_processing shown
    the plan whose tools live in the meta-tool set it never receives).

    Worse than a missing instruction: the model tries, "fails", and then reasons
    on from the failure as if it were information about the experiment.
    """

    @staticmethod
    def _real_tool_names() -> set[str]:
        """Every ``@tool("name")`` in the tree, by source scan.

        A scan rather than imports: the tools live behind factories that need a
        BufferService, a registry and a context provider, and a test that has to
        construct those to answer "does this name exist" would be skipped the
        first time one of them got harder to build.
        """
        import re
        root = Path(_MASTV2_ROOT) / "mast"
        names: set[str] = set()
        for p in root.rglob("*.py"):
            try:
                src = p.read_text(encoding="utf-8")
            except Exception:
                continue
            names.update(re.findall(r'@tool\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']', src))
        return names

    def test_every_readback_tool_actually_exists(self):
        real = self._real_tool_names()
        assert "load_document" in real, (
            "precondition: load_document must exist — the artifact channel names "
            "it to every document consumer")
        missing = {tool_name
                   for options in ch._READBACK.values()
                   for tool_name, _hint in options
                   if tool_name not in real}
        assert not missing, (
            f"_READBACK names tool(s) that do not exist anywhere: {sorted(missing)}. "
            f"An instruction to use a capability the agent does not have is worse "
            f"than no instruction.")

    def test_a_hint_is_suppressed_when_the_agent_lacks_the_tool(self):
        ref = DocRef(doc_id="d1", version=1, kind="literature_report", title="T",
                     path="/tmp/x.md", summary="要点")
        with_tool = ch.render_field("literature_report", ref, {"load_document"})
        without = ch.render_field("literature_report", ref, set())
        assert "load_document" in with_tool
        assert "load_document" not in without, (
            "the block advertised a tool the agent does not hold")
        # …and it still says what it CAN say.
        assert "要点" in without and "d1" in without

    def test_with_no_reader_it_names_the_file_instead_of_inventing_a_tool(self):
        ref = DocRef(doc_id="d1", kind="literature_report", path="/tmp/x.md")
        line = ch.render_field("literature_report", ref, set())
        assert "/tmp/x.md" in line

    def test_none_means_unknown_not_empty(self):
        """None (could not read the tool list) must NOT behave like an empty set.

        Collapsing them would make an upstream API change silently strip every
        readback instruction — a regression that no test would notice because the
        block would still render.
        """
        ref = DocRef(doc_id="d1", kind="literature_report")
        assert "load_document" in ch.render_field("literature_report", ref, None)
        assert ch.render_field("literature_report", ref, set()) is not None

    def test_the_plan_hint_never_offers_load_document(self):
        """A plan's id is a planning-DB ``plan_id``, not a documents-store doc_id
        (``create_plan`` sets ``doc_id=plan_id``). Offering load_document there
        would look plausible and fail — the worst kind of wrong hint."""
        for tool_name, _hint in ch._READBACK["plan"]:
            assert tool_name != "load_document"

    def test_middleware_reads_the_real_tool_list(self):
        """The gate is only real if the middleware actually supplies the names."""
        from mast.agents._shared.upstream_mw import UpstreamArtifactMiddleware

        class _Req:
            state = {"literature_report": DocRef(doc_id="d9", kind="literature_report",
                                                 title="T", summary="s")}
            system_message = None
            tools = []       # holds NOTHING

        block = UpstreamArtifactMiddleware("experiment_design")._block(_Req())
        assert "d9" in block
        assert "load_document" not in block, (
            "middleware did not pass the agent's real tool list through")


class TestRendering:
    def test_an_empty_docref_renders_nothing(self):
        assert ch.render_field("draft", DocRef(doc_id="")) is None

    def test_render_accepts_a_plain_dict(self):
        """A value round-tripped through the SQLite checkpointer can come back as
        a dict rather than the Pydantic model."""
        line = ch.render_field("draft", {"doc_id": "d1", "version": 3,
                                         "title": "T", "kind": "paper_draft"})
        assert line and "d1" in line and "T" in line

    def test_unknown_agent_renders_nothing(self):
        assert ch.render_upstream_block({"draft": _REPORT}, "nobody") == ""

    def test_block_names_how_to_read_the_body(self):
        """A pointer whose body cannot be fetched is a tease. Every rendered doc
        line must name the readback route."""
        block = ch.render_upstream_block({"literature_report": _REPORT},
                                         "experiment_design")
        assert "load_document" in block

    def test_a_render_failure_does_not_take_the_turn_down(self):
        class _Explodes:
            def __getattr__(self, item):
                raise RuntimeError("boom")

        assert ch.render_field("draft", _Explodes()) is None


class TestCarriedFrom:
    def test_skips_absent_and_empty(self):
        carried = ch.carried_from({"draft": None, "scan_id": "",
                                   "literature_report": _REPORT})
        assert set(carried) == {"literature_report"}

    def test_none_state_is_safe(self):
        assert ch.carried_from(None) == {}

    def test_only_known_fields_travel(self):
        carried = ch.carried_from({"literature_report": _REPORT,
                                   "some_other_key": "should not travel"})
        assert set(carried) <= set(ch.CARRIED_FIELDS)


class TestDocRefBounds:
    def test_summary_is_bounded_by_the_type_not_by_convention(self):
        """The checkpointer rewrites the whole channel every super-step, so an
        unbounded summary is paid on every hop. Enforced in a validator because
        'keep it short' as a convention has a 100% historical failure rate."""
        from mast.agents.state import SUMMARY_MAX_CHARS
        ref = DocRef(doc_id="d", summary="x" * (SUMMARY_MAX_CHARS * 3))
        assert len(ref.summary) <= SUMMARY_MAX_CHARS + 16
        assert ref.summary.endswith("（已截断）")

    def test_analysis_lists_are_bounded(self):
        from mast.agents.state import LIST_MAX_ITEMS, AnalysisResult
        a = AnalysisResult(figures=[f"f{i}.png" for i in range(LIST_MAX_ITEMS * 3)])
        assert len(a.figures) == LIST_MAX_ITEMS


# ═══════════════════════════════════════════════════════════════════════════
# Parallel fan-out — the case the missing reducer would have crashed
# ═══════════════════════════════════════════════════════════════════════════

class TestParallelFanOutWritesTheSameField:
    def test_two_branches_publishing_the_same_field_do_not_crash(self):
        """Both branches write ``last_scan`` in ONE super-step. Before the
        reducers were added this raised
        ``InvalidUpdateError: can receive only one value per step`` — not
        occasionally, but on every parallel run that carried an artifact."""
        from mast.agents.state import ScanResult

        pub_a = _make_publisher("pub_a", "last_scan",
                                ScanResult(handle="a", status="done",
                                           sxm_path="/tmp/a.sxm"))
        pub_b = _make_publisher("pub_b", "last_scan",
                                ScanResult(handle="b", status="done",
                                           sxm_path="/tmp/b.sxm"))

        def _llm(pub):
            return _Rec(messages=iter([
                AIMessage(content="", tool_calls=[_tc(pub.name, {}, f"{pub.name}-1")]),
                AIMessage(content="", tool_calls=[
                    _tc("handoff_to_supervisor", {"reason": "done"}, f"{pub.name}-2")]),
            ]))

        orch = build_orchestrator(
            buf=None,
            include_agents=("literature", "data_processing"),
            supervisor_model=_ScriptedRouter([["literature", "data_processing"],
                                              ["__end__"]]),
            agent_model_overrides={"literature": _llm(pub_a),
                                   "data_processing": _llm(pub_b)},
            checkpointer=InMemorySaver(),
            memory_tools=[pub_a, pub_b],
        )
        out = orch.invoke(_seed_state("同时做两件事"),
                          config={"configurable": {"thread_id": "t-fanout"},
                                  "recursion_limit": 25})
        got = out.get("last_scan")
        assert got is not None
        # last_wins: an arbitrary branch wins, and that is the documented
        # contract — what must NOT happen is a crash or a lost run.
        assert ch._get(got, "sxm_path") in ("/tmp/a.sxm", "/tmp/b.sxm")


# ═══════════════════════════════════════════════════════════════════════════
# visit_count: the session-bricking bug
# ═══════════════════════════════════════════════════════════════════════════

class TestNoHandoffExitDoesNotInflateTheLoopGuard:
    """The 2026-07-30 spike, kept as a regression test.

    An agent can leave its subgraph two ways, and they behave differently:

      * ``handoff`` → ``Command(graph=Command.PARENT)`` short-circuits; only
        ``command.update`` crosses;
      * **no handoff** (ModelCallLimit ``jump_to:end``, StallGuard forced stop, or
        the model just answering in prose) → the subgraph's FULL final state is
        merged into the parent through the parent's reducers.

    The second path is the common one — 621 ``fail_silent_end`` events in the
    diagnostics ledger — and while the subgraph held a copy of ``visit_count`` it
    charged the loop guard twice for one hop. Nothing in the tree covered it,
    which is why it shipped.
    """

    def _run(self, *, hand_off: bool):
        @tool("touch_audit")
        def touch_audit(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
            """Write the append-only audit channels, like a real skill does."""
            return Command(update={
                "scan_paths": ["/tmp/one.sxm"],
                "error_log": ["one error"],
                "messages": [ToolMessage(content="touched",
                                         tool_call_id=tool_call_id,
                                         name="touch_audit")],
            })

        if hand_off:
            script = [
                AIMessage(content="", tool_calls=[_tc("touch_audit", {}, "a1")]),
                AIMessage(content="", tool_calls=[
                    _tc("handoff_to_supervisor", {"reason": "done"}, "a2")]),
            ]
        else:
            script = [
                AIMessage(content="", tool_calls=[_tc("touch_audit", {}, "a1")]),
                AIMessage(content="就这样，我直接回答了。"),
            ]

        orch = build_orchestrator(
            buf=None,
            include_agents=("literature",),
            supervisor_model=_ScriptedRouter([["literature"], ["__end__"]]),
            agent_model_overrides={"literature": _Rec(messages=iter(script))},
            checkpointer=InMemorySaver(),
            memory_tools=[touch_audit],
        )
        return orch.invoke(
            _seed_state("go"),
            config={"configurable": {"thread_id": f"nh-{hand_off}"},
                    "recursion_limit": 25})

    def test_the_agent_is_counted_once_either_way(self):
        for hand_off in (True, False):
            vc = self._run(hand_off=hand_off).get("visit_count") or {}
            assert vc.get("literature") == 1, (
                f"hand_off={hand_off}: literature counted {vc.get('literature')} "
                f"times for ONE dispatch — the loop guard both caps read is "
                f"inflated ({dict(vc)})")

    def test_the_audit_logs_are_not_duplicated(self):
        out = self._run(hand_off=False)
        assert (out.get("scan_paths") or []).count("/tmp/one.sxm") == 1
        assert (out.get("error_log") or []).count("one error") == 1


class TestVisitCountReset:
    def test_empty_dict_is_a_no_op_and_None_clears(self):
        assert sum_int_dicts({"supervisor": 39}, {}) == {"supervisor": 39}
        assert sum_int_dicts({"supervisor": 39}, None) == {}

    def test_a_thread_past_the_hop_cap_is_usable_again_after_a_reset(self):
        """The lived failure: a long-running group conversation accumulated hops
        for its whole lifetime (the per-task 'reset' wrote ``{}`` into an ADDING
        reducer, a no-op). Once past the guard, EVERY later task ended instantly
        with no explanation. Seeding ``None`` is what makes a new task a new
        budget."""
        from mast.agents.orchestrator.graph import _HOP_HARD_CAP

        bricked = {"supervisor": _HOP_HARD_CAP + 5}
        assert sum(sum_int_dicts(bricked, {}).values()) > _HOP_HARD_CAP
        assert sum(sum_int_dicts(bricked, None).values()) == 0

    def test_run_task_seeds_the_clearing_sentinel(self):
        """The reducer's escape hatch is only worth anything if the caller uses
        it. Pinned against the source because the wrong value ( ``{}`` ) is
        indistinguishable from the right one at a glance."""
        import mast.api.routes.orchestrator as orch_routes
        src = inspect.getsource(orch_routes)
        assert '"visit_count": None' in src, (
            "run_task no longer seeds None — an empty dict is a NO-OP under the "
            "adding reducer, which is exactly how group threads got bricked")


# ═══════════════════════════════════════════════════════════════════════════
# Parent-channel pruning (ships WITH the visit_count fix — see the design doc)
# ═══════════════════════════════════════════════════════════════════════════

class TestParentPrune:
    def test_under_the_cap_nothing_is_touched(self):
        from mast.agents.orchestrator.graph import _plan_parent_prune
        msgs = [AIMessage(content=f"m{i}", id=str(i)) for i in range(10)]
        drop, note = _plan_parent_prune(msgs)
        assert drop == set() and note is None

    def test_over_the_cap_it_keeps_the_first_and_the_recent(self):
        from mast.agents.orchestrator.graph import (
            _PARENT_MSG_KEEP, _PARENT_MSG_SOFT_CAP, _plan_parent_prune,
        )
        n = _PARENT_MSG_SOFT_CAP + 40
        msgs = [AIMessage(content=f"m{i}", id=str(i)) for i in range(n)]
        drop, note = _plan_parent_prune(msgs)
        assert note is not None
        assert "0" not in drop, "the operator's original instruction was dropped"
        kept_tail = {str(i) for i in range(n - _PARENT_MSG_KEEP, n)}
        assert not (drop & kept_tail), "a recent message was dropped"
        assert len(drop) == n - _PARENT_MSG_KEEP - 1

    def test_messages_without_an_id_are_left_alone(self):
        """``RemoveMessage`` addresses by id; an id-less message cannot be removed
        and must not be counted as if it had been."""
        from mast.agents.orchestrator.graph import _PARENT_MSG_SOFT_CAP, _plan_parent_prune
        msgs = [AIMessage(content=f"m{i}") for i in range(_PARENT_MSG_SOFT_CAP + 40)]
        drop, note = _plan_parent_prune(msgs)
        assert drop == set()
        assert note is None

    def test_the_note_says_products_survived(self):
        """A trimmed transcript must not read like lost work: the products are in
        their own channels and on disk, and the note has to say so or an operator
        (and the model) will assume the run lost them."""
        from mast.agents.orchestrator.graph import _PARENT_MSG_SOFT_CAP, _plan_parent_prune
        msgs = [AIMessage(content=f"m{i}", id=str(i))
                for i in range(_PARENT_MSG_SOFT_CAP + 40)]
        _drop, note = _plan_parent_prune(msgs)
        assert "产物" in str(note.content)

    def test_planning_never_raises(self):
        from mast.agents.orchestrator.graph import _plan_parent_prune
        assert _plan_parent_prune(None) == (set(), None)

    def test_the_meta_key_spelling_matches_all_three_sites(self):
        """The key is written as a literal in three modules that deliberately do
        not import each other (the SSE bridge stays free of agent imports). A
        typo in any one of them fails silently — the indicator simply never
        lights up, which is indistinguishable from "nothing was trimmed"."""
        from mast.agents._shared.compaction_mw import COMPACTION_META_KEY
        from mast.agents.orchestrator.graph import _COMPACTION_META_KEY as graph_key
        from mast.api.routes.orchestrator import _COMPACTION_META_KEY as bridge_key
        assert COMPACTION_META_KEY == graph_key == bridge_key

    def test_the_prune_rides_the_visibility_chain(self):
        from mast.agents._shared.compaction_mw import COMPACTION_META_KEY
        from mast.agents.orchestrator.graph import _PARENT_MSG_SOFT_CAP, _plan_parent_prune
        msgs = [AIMessage(content=f"m{i}", id=str(i))
                for i in range(_PARENT_MSG_SOFT_CAP + 40)]
        _drop, note = _plan_parent_prune(msgs)
        event = note.additional_kwargs.get(COMPACTION_META_KEY)
        assert isinstance(event, dict) and event.get("mode") == "parent_prune"

    def test_a_prune_is_never_reported_as_a_summary(self):
        """A deterministic delete described as "已被摘要替代" would send an operator
        looking for a summary that does not exist."""
        from mast.api.routes.orchestrator import _compaction_line
        line = _compaction_line({"mode": "parent_prune", "removed": 12, "kept": 80})
        assert "摘要替代" not in line
        assert "12" in line and "产物" in line
        # …while a REAL summarisation still says so.
        assert "摘要替代" in _compaction_line({"removed": 12, "kept": 20})


# ═══════════════════════════════════════════════════════════════════════════
# Compaction is sized for the model the agent REALLY runs
# ═══════════════════════════════════════════════════════════════════════════

class TestCompactionSizing:
    def test_effective_model_follows_the_persisted_override(self, monkeypatch):
        """``get_model_id`` only ever knew the code default. Sizing compaction
        with it meant an agent overridden onto a 120k-window model still used the
        250k-window threshold (175 500 tokens) — i.e. compaction could only fire
        AFTER the provider had already rejected the request."""
        from mast.agents._shared import models as m

        class _FakeRegistry:
            @staticmethod
            def get():
                class _R:
                    @staticmethod
                    def get_agent_overrides():
                        return {"literature": {"model": "glm-5.2"}}
                return _R()

        monkeypatch.setattr("mast.admin.override_store.ConfigOverrideRegistry",
                            _FakeRegistry)
        assert m.resolve_effective_model_id("literature") == m.GLM_5_2
        assert m.get_model_id("literature") != m.GLM_5_2, (
            "precondition: the code default differs from the override")

    def test_it_degrades_to_the_default_rather_than_raising(self, monkeypatch):
        from mast.agents._shared import models as m

        class _Boom:
            @staticmethod
            def get():
                raise RuntimeError("registry unavailable")

        monkeypatch.setattr("mast.admin.override_store.ConfigOverrideRegistry", _Boom)
        assert m.resolve_effective_model_id("literature") == m.get_model_id("literature")

    def test_the_threshold_really_moves_with_the_window(self):
        from mast.agents._shared.compaction_mw import compaction_trigger_tokens
        big = compaction_trigger_tokens("kimi-k3")
        small = compaction_trigger_tokens("glm-5.2")
        assert small < big, "a smaller window must compact earlier"

    def test_group_build_sizes_per_agent(self):
        """Six agents used to share ONE compaction instance sized for the
        orchestrator's model — and shared its per-turn caches while running
        concurrently under a fan-out."""
        import mast.core.runtime as rt
        src = source_of(rt.CoreRuntime._group_agent_middleware)
        assert "_chat_agent_middleware(agent_id)" in src


# ═══════════════════════════════════════════════════════════════════════════
# Dead code that described mechanisms which did not exist
# ═══════════════════════════════════════════════════════════════════════════

class TestRemovedDeadCode:
    def test_supervisor_notice_template_is_gone(self):
        """Zero call sites for its whole life, while its docstring claimed "every
        agent's return path uses this template". A dead function that describes
        an intention reads like a mechanism already in place."""
        import mast.agents._shared.handoff as h
        assert not hasattr(h, "make_supervisor_notice")

    def test_the_orphan_token_module_is_gone(self):
        with pytest.raises(ImportError):
            __import__("mast.agents._shared.tokens", fromlist=["x"])

    def test_the_dead_per_agent_cap_name_is_never_referenced_as_code(self):
        """History: a per-agent ceiling named ``_PER_AGENT_HARD_CAP`` was removed
        2026-06-29 while two comment blocks went on explaining how it worked, one
        calling it "the binding constraint". A comment describing a mechanism that
        does not exist misleads exactly as much as wrong code, and costs a reader
        more to disprove.

        A per-agent ceiling is LIVE again since 2026-07-30, but under different
        names (``_AGENT_HARD_CAP`` for real agents, ``_SUPERVISOR_HARD_CAP`` for the
        supervisor key) — split precisely because sharing ONE ceiling with the
        supervisor is what made the old one misfire. So the old name must never
        return as an identifier, while prose may still mention it as history.
        """
        import ast

        import mast.agents.orchestrator.graph as g
        src = inspect.getsource(g)
        tree = ast.parse(src)
        used = [n for n in ast.walk(tree)
                if (isinstance(n, ast.Name) and n.id == "_PER_AGENT_HARD_CAP")
                or (isinstance(n, ast.Attribute) and n.attr == "_PER_AGENT_HARD_CAP")]
        # AST, not a line scan: a prose mention of the old name is fine (and
        # useful). Only an executable reference is a defect.
        assert not used, (
            "graph.py references _PER_AGENT_HARD_CAP as an identifier at line(s) "
            f"{[n.lineno for n in used]}, but that constant does not exist — the "
            "live names are _AGENT_HARD_CAP / _SUPERVISOR_HARD_CAP")

    def test_the_restored_ceilings_are_documented_where_they_are_defined(self):
        """The retune is only safe to touch again if the next reader can find out
        WHY the numbers are what they are. Both root causes of the original misfire
        — cross-task accumulation, and the supervisor sharing the agents' ceiling —
        must be recorded next to the constants, not left in a commit message."""
        import mast.agents.orchestrator.graph as g
        src = inspect.getsource(g)
        for needle in ("_AGENT_HARD_CAP", "_SUPERVISOR_HARD_CAP", "_SUPERVISOR_KEY"):
            assert needle in src, f"{needle} vanished — a ceiling is gone again"
        assert "binding constraint" in src, (
            "the reason the supervisor key needs its OWN ceiling is no longer "
            "recorded; that omission is what got the cap deleted last time")
