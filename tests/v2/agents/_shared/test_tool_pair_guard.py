"""Regression: the group orchestrator must never ship a provider an unmatched
tool message (— ``tool_call_id is not found`` 400).

Root cause (verified 2026-07-20): an agent hands back by CALLING
``handoff_to_supervisor``, whose ``Command(goto="supervisor", graph=PARENT)``
short-circuits the subgraph so ONLY the handoff ``ToolMessage`` reaches the
parent ``messages`` channel — the ``AIMessage`` that called it stays behind in
the subgraph namespace. The next agent, seeded with that parent history, ships
the orphan ``ToolMessage`` straight to Kimi/OpenAI-compat → 400.

Three layers of proof:
  1. ``repair_tool_pairs`` (pure) converts orphan results, satisfies orphan
     calls, and no-ops on well-formed input;
  2. ``ToolPairGuardMiddleware.wrap_model_call`` rewrites only the request;
  3. through the REAL orchestrator + REAL agent subgraphs, the SECOND agent's
     model is handed a clean, fully-paired message list — the exact bug.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from mast.agents._shared.tool_pair_guard_mw import (
    ToolPairGuardMiddleware,
    repair_tool_pairs,
)


def _tc(name: str, tid: str, args: dict | None = None) -> dict:
    return {"name": name, "args": args or {}, "id": tid, "type": "tool_call"}


def _orphan_results(msgs) -> list[str]:
    provided: set[str] = set()
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tid:
                provided.add(tid)
    return [m.tool_call_id for m in msgs
            if isinstance(m, ToolMessage) and m.tool_call_id
            and m.tool_call_id not in provided]


def _orphan_calls(msgs) -> list[str]:
    answered = {m.tool_call_id for m in msgs
                if isinstance(m, ToolMessage) and m.tool_call_id}
    out: list[str] = []
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tid and tid not in answered:
                out.append(tid)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 1. repair_tool_pairs — the pure repair
# ═══════════════════════════════════════════════════════════════════════════

class TestRepairToolPairs:
    def test_wellformed_is_left_untouched(self):
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[_tc("scan", "a")]),
            ToolMessage(content="ok", tool_call_id="a"),
        ]
        assert repair_tool_pairs(msgs) is None

    def test_orphan_result_is_converted_to_narration_preserving_text(self):
        msgs = [
            HumanMessage(content="go"),
            AIMessage(content="[SUPERVISOR → literature]"),
            ToolMessage(content="[HANDOFF → supervisor] 扫描已完成", tool_call_id="ic-h"),
        ]
        out = repair_tool_pairs(msgs)
        assert out is not None
        assert _orphan_results(out) == []
        # the ToolMessage is gone, its text survives as assistant narration
        assert not any(isinstance(m, ToolMessage) for m in out)
        assert any("扫描已完成" in str(m.content) for m in out)

    def test_contentless_orphan_result_is_dropped(self):
        msgs = [
            HumanMessage(content="go"),
            ToolMessage(content="", tool_call_id="ghost"),
        ]
        out = repair_tool_pairs(msgs)
        assert out is not None
        assert not any(isinstance(m, ToolMessage) for m in out)
        assert len(out) == 1

    def test_orphan_call_is_satisfied_with_synthetic_result(self):
        msgs = [
            HumanMessage(content="go"),
            AIMessage(content="", tool_calls=[_tc("scan", "x")]),  # never answered
        ]
        out = repair_tool_pairs(msgs)
        assert out is not None
        assert _orphan_calls(out) == []
        synth = [m for m in out if isinstance(m, ToolMessage) and m.tool_call_id == "x"]
        assert len(synth) == 1

    def test_partial_answer_keeps_real_and_synthesizes_missing(self):
        # one AIMessage with TWO calls; only the first is answered
        msgs = [
            AIMessage(content="", tool_calls=[_tc("a", "1"), _tc("b", "2")]),
            ToolMessage(content="done-1", tool_call_id="1"),
        ]
        out = repair_tool_pairs(msgs)
        assert out is not None
        assert _orphan_calls(out) == [] and _orphan_results(out) == []
        # the real answer for id=1 is preserved verbatim
        assert any(isinstance(m, ToolMessage) and m.tool_call_id == "1"
                   and "done-1" in str(m.content) for m in out)
        # a synthetic answer exists for id=2
        assert any(isinstance(m, ToolMessage) and m.tool_call_id == "2" for m in out)

    def test_matched_pair_amid_orphans_is_preserved(self):
        msgs = [
            AIMessage(content="", tool_calls=[_tc("real", "keep")]),
            ToolMessage(content="real-result", tool_call_id="keep"),
            ToolMessage(content="[HANDOFF] x", tool_call_id="orphan"),
        ]
        out = repair_tool_pairs(msgs)
        assert out is not None
        # the matched pair (keep) survives intact; the orphan is gone
        assert any(isinstance(m, ToolMessage) and m.tool_call_id == "keep" for m in out)
        assert _orphan_results(out) == [] and _orphan_calls(out) == []

    def test_empty_is_noop(self):
        assert repair_tool_pairs([]) is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. ToolPairGuardMiddleware.wrap_model_call — request-only rewrite
# ═══════════════════════════════════════════════════════════════════════════

class _FakeRequest:
    def __init__(self, messages):
        self.messages = list(messages)

    def override(self, *, messages):
        return _FakeRequest(messages)


class TestMiddleware:
    def test_wrap_model_call_repairs_via_override(self):
        seen: dict = {}

        def handler(req):
            seen["messages"] = req.messages
            return AIMessage(content="ok")

        req = _FakeRequest([
            AIMessage(content="[SUPERVISOR → literature]"),
            ToolMessage(content="[HANDOFF → supervisor] done", tool_call_id="ic-h"),
        ])
        ToolPairGuardMiddleware().wrap_model_call(req, handler)
        assert _orphan_results(seen["messages"]) == []
        # original request object is untouched (override returns a new one)
        assert _orphan_results(req.messages) == ["ic-h"]

    def test_wrap_model_call_noop_on_clean_history(self):
        req = _FakeRequest([HumanMessage(content="hi"), AIMessage(content="hey")])
        passed: dict = {}
        ToolPairGuardMiddleware().wrap_model_call(
            req, lambda r: passed.setdefault("r", r))
        # same object handed through unchanged
        assert passed["r"] is req

    def test_guard_never_raises_on_bad_request(self):
        class _Bad:
            messages = None
        # must not raise
        ToolPairGuardMiddleware().wrap_model_call(_Bad(), lambda r: r)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Through the REAL orchestrator + REAL agent subgraphs — the exact bug
# ═══════════════════════════════════════════════════════════════════════════

from mast.agents.orchestrator.graph import build as build_orchestrator  # noqa: E402
from mast.core.types import NanonisCallRecord  # noqa: E402

_CANNED = {"Bias_Get": {"return_value": (0.5, b"", [])}}


@dataclass
class _FakeInstrument:
    canned: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def safe_call(self, method, *args, role="main"):
        with self._lock:
            self.calls.append((method, args))
        entry = self.canned.get(method)
        if entry is None:
            return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")
        return NanonisCallRecord(method=method, args=args,
                                 return_value=entry.get("return_value"),
                                 error=entry.get("error", ""))


class _RecordingFake(GenericFakeChatModel):
    sink: Any = None

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.sink is not None:
            self.sink.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _ic_llm(sink):
    return _RecordingFake(sink=sink, messages=iter([
        AIMessage(content="", tool_calls=[_tc("GetBias", "ic-1")]),
        AIMessage(content="IC done.",
                  tool_calls=[_tc("handoff_to_supervisor", "ic-h", {"reason": "扫描已完成"})]),
    ]))


def _lit_llm(sink):
    return _RecordingFake(sink=sink, messages=iter([
        AIMessage(content="Lit done.",
                  tool_calls=[_tc("handoff_to_supervisor", "lit-h", {"reason": "文献已检索"})]),
    ]))


class _ScriptedSupervisor:
    def __init__(self, script):
        self.script = list(script)
        self.hops = 0

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _msgs):
                i = min(outer.hops, len(outer.script) - 1)
                outer.hops += 1
                return {"next_agents": list(outer.script[i]), "reason": "serial"}
        return _S()

    def invoke(self, _msgs):
        return AIMessage(content="ok")


def _state(goal: str) -> dict:
    return {
        "messages": [HumanMessage(content=goal)],
        "executed_skills": [], "scan_paths": [], "scan_metadata": {},
        "error_log": [], "event_refs": [], "visit_count": {}, "pending_approvals": {},
    }


def test_second_agent_receives_no_orphan_through_real_subgraphs():
    """The exact production bug: supervisor → instrument_control (hands back) →
    supervisor → literature. Without the guard, literature's model is handed the
    orphan handoff ToolMessage (tool_call_id 'ic-h') and Kimi 400s. The guard is
    wired by each agent's build(), so we pass NO explicit middleware here."""
    ic_sink: list = []
    lit_sink: list = []
    inst = _FakeInstrument(canned=dict(_CANNED))
    app = build_orchestrator(
        buf=None,
        supervisor_model=_ScriptedSupervisor(
            [["instrument_control"], ["literature"], ["__end__"]]),
        agent_model_overrides={
            "instrument_control": _ic_llm(ic_sink),
            "literature": _lit_llm(lit_sink),
        },
        include_agents=("instrument_control", "literature"),
        context_provider=lambda: inst,
        checkpointer=InMemorySaver(),
        enable_hitl=False,
    )
    app.invoke(_state("扫一张图，然后查文献"),
               config={"configurable": {"thread_id": "tpg-serial"}})

    assert lit_sink, "literature model never ran"
    seen = lit_sink[0]
    assert _orphan_results(seen) == [], (
        f"literature was handed orphan tool result(s) {_orphan_results(seen)} "
        "→ provider would 400 'tool_call_id is not found'")
    assert _orphan_calls(seen) == []
    # the handoff reason is preserved as context rather than lost
    assert any("扫描已完成" in str(getattr(m, "content", "")) for m in seen), (
        "the previous agent's handoff context was dropped instead of converted")


def test_same_agent_reentry_receives_no_orphan():
    """The re-entry variant: instrument_control runs, hands back, and is dispatched
    AGAIN. Its second entry is re-seeded from the parent channel that holds the
    orphan handoff ToolMessage — without the guard IC would 400 on its own history."""
    ic_sink: list = []
    inst = _FakeInstrument(canned=dict(_CANNED))
    ic = _RecordingFake(sink=ic_sink, messages=iter([
        AIMessage(content="", tool_calls=[_tc("GetBias", "ic-1")]),
        AIMessage(content="first", tool_calls=[_tc("handoff_to_supervisor", "ic-h1", {"reason": "一"})]),
        AIMessage(content="second", tool_calls=[_tc("handoff_to_supervisor", "ic-h2", {"reason": "二"})]),
    ]))
    app = build_orchestrator(
        buf=None,
        supervisor_model=_ScriptedSupervisor(
            [["instrument_control"], ["instrument_control"], ["__end__"]]),
        agent_model_overrides={"instrument_control": ic},
        include_agents=("instrument_control",),
        context_provider=lambda: inst,
        checkpointer=InMemorySaver(),
        enable_hitl=False,
    )
    app.invoke(_state("扫两次"), config={"configurable": {"thread_id": "tpg-reentry"}})

    # every message list IC's model was ever handed must be provider-safe
    for k, seen in enumerate(ic_sink):
        assert _orphan_results(seen) == [], f"IC call#{k} got orphan result(s)"
        assert _orphan_calls(seen) == [], f"IC call#{k} got orphan call(s)"


def test_guard_is_actually_wired_into_a_built_agent():
    """A refactor that drops the guard from an agent build must fail HERE, not on
    hardware. Behavioural (the compiled graph exposes no middleware list): build a
    REAL instrument_control agent, seed a state carrying an orphan tool result, and
    assert its model is handed a clean list — i.e. the guard is live in the stack."""
    from mast.agents.instrument_control.graph import build as build_ic
    inst = _FakeInstrument(canned=dict(_CANNED))
    sink: list = []
    # model just answers (no tool calls) so the ReAct loop ends after one call.
    model = _RecordingFake(sink=sink, messages=iter([AIMessage(content="done")]))
    agent = build_ic(None, context_provider=lambda: inst, model=model,
                     enable_hitl=False, standalone=True)
    state = _state("continue")
    # inject an orphan tool result (as the group parent channel would carry after
    # a prior agent's handoff) BEFORE the current turn's user message.
    state["messages"] = [
        AIMessage(content="[SUPERVISOR → instrument_control]"),
        ToolMessage(content="[HANDOFF → supervisor] 上一步已完成", tool_call_id="prev-h"),
        HumanMessage(content="continue"),
    ]
    agent.invoke(state, config={"configurable": {"thread_id": "tpg-wired"}})
    assert sink, "the IC model never ran"
    assert _orphan_results(sink[0]) == [], (
        "ToolPairGuardMiddleware is not active in the instrument_control build — "
        f"its model was handed orphan(s) {_orphan_results(sink[0])}")
