"""Parallel fan-out through the REAL agent subgraphs (#32/#35 — integration).

``test_parallel_dispatch.py`` proves the supervisor's dispatch *shape* against
hand-written stub nodes. That is not enough: the production targets of a
``Send`` are **compiled ``create_agent`` subgraphs** with their own state schema,
their own middleware stack, and a real tool node. A ``Send`` payload that a plain
function node happily accepts could be dropped, rejected, or mis-merged by a real
subgraph — and we would never know until an operator ran a fan-out on hardware.

So everything here runs against ``orchestrator.build()`` with REAL agents wired:

  * a fan-out really reaches two compiled subgraphs and BOTH run their model and
    their tool node (instrument_control executes a real skill against a fake
    Nanonis; literature runs a real library tool);
  * both branches return through the REAL ``make_handoff`` tool in the SAME
    super-step — the concurrent ``Command.PARENT`` write that used to raise
    ``InvalidUpdateError``;
  * a fanned-out subgraph SEES the supervisor's dispatch note (Send passes an
    explicit payload — the branches would otherwise be starved);
  * instrument_control cannot be fanned out twice (the hardware invariant), and
    the fake instrument records exactly ONE session of calls;
  * HITL still pauses a real IC subgraph *inside* a fan-out, and two pending
    approvals resume BY ID with different decisions — no cross-feeding;
  * the structural premise the whole design rests on: **no agent other than
    instrument_control holds a single instrument skill.**
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

import threading  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from mast.agents.orchestrator.graph import build as build_orchestrator  # noqa: E402
from mast.core.types import NanonisCallRecord  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Fakes — a fake instrument and per-agent fake LLMs that RECORD what they saw
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FakeInstrument:
    """Stands in for ExecutionContext. Records every Nanonis verb, thread-safely
    (a fan-out really does run branches on worker threads)."""

    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        with self._lock:
            self.calls.append((method, args))
        entry = self.canned.get(method)
        if entry is None:
            return NanonisCallRecord(method=method, args=args,
                                     error=f"unmocked: {method}")
        return NanonisCallRecord(method=method, args=args,
                                 return_value=entry.get("return_value"),
                                 error=entry.get("error", ""))

    def methods(self) -> list[str]:
        with self._lock:
            return [m for m, _ in self.calls]


_CANNED = {
    "Bias_Get": {"return_value": (0.5, b"", [])},
    "Bias_Pulse": {"return_value": ("", b"", [])},
}


class _RecordingFake(GenericFakeChatModel):
    """A scripted fake chat model that records every message list it was asked
    to answer — this is how we prove a fanned-out branch really SAW the
    supervisor's dispatch note rather than being handed a stale state."""

    sink: Any = None   # a list to append each seen message list to

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.sink is not None:
            self.sink.append(list(messages))
        return super()._generate(messages, stop=stop,
                                 run_manager=run_manager, **kwargs)


def _tc(name: str, args: dict, tid: str) -> dict:
    return {"name": name, "args": args, "id": tid, "type": "tool_call"}


def _ic_llm(sink: list, *, tool_calls: list[dict] | None = None):
    """instrument_control: run a real skill, then hand back through the real
    handoff tool."""
    first = tool_calls if tool_calls is not None else [_tc("GetBias", {}, "ic-1")]
    return _RecordingFake(sink=sink, messages=iter([
        AIMessage(content="", tool_calls=first),
        AIMessage(content="IC finished the instrument work.",
                  tool_calls=[_tc("handoff_to_supervisor",
                                  {"reason": "扫描已完成"}, "ic-h")]),
    ]))


def _lit_llm(sink: list):
    """literature: run a real library tool, then hand back."""
    return _RecordingFake(sink=sink, messages=iter([
        AIMessage(content="", tool_calls=[_tc("lib_list", {}, "lit-1")]),
        AIMessage(content="Literature survey done.",
                  tool_calls=[_tc("handoff_to_supervisor",
                                  {"reason": "文献已检索"}, "lit-h")]),
    ]))


class _ScriptedSupervisor:
    """Routes per a scripted list of decisions; the last one repeats forever."""

    def __init__(self, script: list[list[str]], reason: str = "并行:互不依赖"):
        self.script = list(script)
        self.reason = reason
        self.hops = 0

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, _msgs):
                i = min(outer.hops, len(outer.script) - 1)
                outer.hops += 1
                return {"next_agents": list(outer.script[i]), "reason": outer.reason}

        return _S()

    def invoke(self, _msgs):  # _direct_answer / text tier — never needed here
        return AIMessage(content="ok")


def _build(supervisor, *, ic_sink, lit_sink, instrument, get_mode=None,
           enable_hitl=False, ic_tool_calls=None):
    return build_orchestrator(
        buf=None,
        supervisor_model=supervisor,
        agent_model_overrides={
            "instrument_control": _ic_llm(ic_sink, tool_calls=ic_tool_calls),
            "literature": _lit_llm(lit_sink),
        },
        include_agents=("instrument_control", "literature"),
        context_provider=lambda: instrument,
        checkpointer=InMemorySaver(),
        enable_hitl=enable_hitl,
        get_mode=get_mode,
    )


def _state(goal: str = "扫一张图，同时查一下文献") -> dict:
    return {
        "messages": [HumanMessage(content=goal)],
        "executed_skills": [], "scan_paths": [], "scan_metadata": {},
        "error_log": [], "event_refs": [], "visit_count": {},
        "pending_approvals": {},
    }


def _texts(msgs) -> list[str]:
    return [str(getattr(m, "content", "")) for m in msgs]


# ═══════════════════════════════════════════════════════════════════════════
# THE integration test — a fan-out through two real compiled subgraphs
# ═══════════════════════════════════════════════════════════════════════════

class TestFanOutThroughRealSubgraphs:
    def test_both_real_subgraphs_run_and_hand_back_concurrently(self):
        """The production shape: Send → two compiled create_agent subgraphs.

        Everything a stub node could not prove:
          * both subgraphs run their model AND their tool node;
          * IC's real skill really reaches the (fake) instrument;
          * both return via the REAL handoff tool in ONE super-step — the
            concurrent Command.PARENT write that used to raise
            InvalidUpdateError before the state got its reducers.
        """
        ic_sink: list = []
        lit_sink: list = []
        inst = FakeInstrument(canned=dict(_CANNED))
        app = _build(
            _ScriptedSupervisor([["instrument_control", "literature"], ["__end__"]]),
            ic_sink=ic_sink, lit_sink=lit_sink, instrument=inst)

        out = app.invoke(_state(), config={"configurable": {"thread_id": "real-1"}})

        # 1. both branches really executed their model
        assert ic_sink, "instrument_control subgraph never ran its model"
        assert lit_sink, "literature subgraph never ran its model"

        # 2. IC's tool node really executed the skill against the instrument
        assert "Bias_Get" in inst.methods(), (
            f"GetBias never reached the instrument; calls={inst.methods()}")

        # 3. literature's tool node really ran too (a non-IC subgraph under Send).
        #    NB the parent's message channel does NOT carry a subgraph's internal
        #    messages: a handoff returns Command(graph=Command.PARENT), which
        #    short-circuits out of the subgraph, so only `command.update` reaches
        #    the parent (the agent's tool traffic lives in the subgraph's own
        #    checkpoint namespace and is surfaced live via astream(subgraphs=True)).
        #    The honest proof the ToolNode ran is therefore the model's SECOND
        #    turn: it can only exist because a lib_list ToolMessage came back.
        assert len(lit_sink) >= 2, "literature's ToolNode never ran under Send"
        second_turn = [getattr(m, "name", "") for m in lit_sink[1]]
        assert "lib_list" in second_turn, (
            f"literature's tool result never returned to its model; {second_turn}")

        # 4. BOTH handoffs landed in the same super-step (the InvalidUpdateError
        #    regression) — and the message channel merged them, losing neither.
        handoffs = [t for t in _texts(out["messages"]) if "[HANDOFF → supervisor]" in t]
        assert len(handoffs) == 2, f"expected 2 concurrent handoffs, got {handoffs}"
        assert any("扫描已完成" in t for t in handoffs)
        assert any("文献已检索" in t for t in handoffs)

        # 5. the fan-out was counted as 2 agent-hops, and the hint channel was
        #    consumed + cleared (a stale hint would resurrect a finished agent).
        vc = out["visit_count"]
        assert vc.get("instrument_control", 0) >= 1 and vc.get("literature", 0) >= 1
        assert not out.get("routing_hints")

    def test_fanned_out_subgraph_sees_the_supervisors_note(self):
        """Send passes an EXPLICIT payload. Forward the pre-update state instead
        and every branch is starved of the routing note and of any operator
        interjection injected this same step — silently, with no error."""
        ic_sink: list = []
        lit_sink: list = []
        app = _build(
            _ScriptedSupervisor([["instrument_control", "literature"], ["__end__"]],
                                reason="并行理由:扫描与文献互不依赖"),
            ic_sink=ic_sink, lit_sink=lit_sink,
            instrument=FakeInstrument(canned=dict(_CANNED)))
        app.invoke(_state(), config={"configurable": {"thread_id": "real-2"}})

        for name, sink in (("instrument_control", ic_sink), ("literature", lit_sink)):
            first_turn = _texts(sink[0])
            assert any("并行理由" in t for t in first_turn), (
                f"{name} never saw the supervisor's dispatch note")
            assert any("扫一张图" in t for t in first_turn), (
                f"{name} lost the original user goal")

    def test_instrument_control_cannot_be_fanned_out_twice(self):
        """The one hardware invariant of the whole design. A confused router names
        IC twice; de-duplication must collapse it, and the instrument must see a
        single agent's worth of traffic — not two concurrent Nanonis sessions."""
        ic_sink: list = []
        inst = FakeInstrument(canned=dict(_CANNED))
        app = _build(
            _ScriptedSupervisor([["instrument_control", "instrument_control",
                                  "literature"], ["__end__"]]),
            ic_sink=ic_sink, lit_sink=[], instrument=inst)
        out = app.invoke(_state(), config={"configurable": {"thread_id": "real-3"}})

        assert inst.methods().count("Bias_Get") == 1, (
            f"the instrument was driven twice in one fan-out: {inst.methods()}")
        ic_handoffs = [t for t in _texts(out["messages"])
                       if "[HANDOFF → supervisor] 扫描已完成" in t]
        assert len(ic_handoffs) == 1

    def test_serial_single_target_still_works_through_a_real_subgraph(self):
        """The pre-parallel path must be untouched: one target → a plain goto,
        no Send, same result."""
        ic_sink: list = []
        inst = FakeInstrument(canned=dict(_CANNED))
        app = _build(_ScriptedSupervisor([["instrument_control"], ["__end__"]]),
                     ic_sink=ic_sink, lit_sink=[], instrument=inst)
        out = app.invoke(_state(), config={"configurable": {"thread_id": "real-4"}})
        assert "Bias_Get" in inst.methods()
        assert any("[HANDOFF → supervisor]" in t for t in _texts(out["messages"]))


# ═══════════════════════════════════════════════════════════════════════════
# HITL inside a fan-out — through the REAL IC middleware stack
# ═══════════════════════════════════════════════════════════════════════════

class TestNoHitlPauseInsideAFanOut:
    """⑰(2026-08-08):扇出的分支里**不再有审批暂停** —— 两发脉冲都直接打出去。

    这个类原名 ``TestHitlInsideARealFanOut``,钉的是 SEMI 模式下
    ``ModeGatedPulseHITLMiddleware`` 从**扇出的真子图内部**发出的 interrupt 的
    **形状**:LangChain 的 HITL 会发 **一个** ``interrupt()`` 带一个
    ``action_requests`` 列表(每个被门控的工具调用一项),并期待**一个**
    ``{"decisions": [...]}`` 的 resume 值,按位置对齐。``routes/orchestrator._drive``
    正是靠这个契约把 N 张用户卡片的裁决合并回一个 resume 值的。

    那个中间件随整条确认框链路一起删除,于是**这个形状在树里没有生产者了**。
    契约本身没有作废(``_drive`` 的合并逻辑还在,``hitl_bridge`` 的分支也还在),
    只是没有东西会再产生它 —— 所以这里改钉两件当下为真的事:

      1. 两发脉冲**都执行了,而且按顺序**(旧测试里被门控住的那两发);
      2. 扇出的另一条分支照常完成(旧断言,原因变了:不是「不被审批拖累」,
         而是根本没有暂停这回事)。

    要把审批框加回来,需要观测到:一次「确认框拦下了真实损害、而 SafetyGate 的
    包络/深度硬闸接不住」的实例。截至 2026-08-08 零例。
    """

    def _pulses(self):
        return [_tc("BiasPulse", {"width_s": "10m", "bias_v": "3"}, "p-1"),
                _tc("BiasPulse", {"width_s": "10m", "bias_v": "4"}, "p-2")]

    def _run_app(self, inst, thread: str, lit_sink: list | None = None):
        app = _build(
            _ScriptedSupervisor([["instrument_control", "literature"], ["__end__"]]),
            ic_sink=[], lit_sink=lit_sink if lit_sink is not None else [],
            instrument=inst,
            get_mode=lambda: "semi", enable_hitl=True,
            ic_tool_calls=self._pulses())
        cfg = {"configurable": {"thread_id": thread}}
        app.invoke(_state("打两个脉冲，同时查文献"), config=cfg)
        return app, cfg

    def test_both_pulses_fire_without_any_pause(self):
        """**语义反转的正主。** 原名
        ``test_both_pulses_are_gated_as_action_requests_of_one_interrupt``。

        旧断言:两发都被门控,一个都没打出去。新断言:两发都打出去了,
        而且图没有停在任何 interrupt 上。

        ## 2026-08-10:顺序断言拿掉了 —— 它先天是 flaky 的

        原来这里断言的是 ``[p[2] for p in pulses] == [3.0, 4.0]``,**按完成顺序**。
        实测连跑 5 次:**3 红 2 绿**;摘掉无关的中间件再跑 6 次:**3 红 3 绿** ——
        同样的抖动率,与任何一次改动都无关。

        **这是一个扇出并行**(文件名、类名都写着)。两条并行分支的完成顺序
        **没有任何东西保证**,所以那条断言钉的是一个不存在的契约。

        这条测试真正要钉的东西写在它自己的名字里:
        **两发都打出去了 + 没有停在任何 interrupt 上**。值要对,顺序不是它的主题。

        ⚠️ 它的代价不是「偶尔要重跑」:**它让「全量闸门绿过一次」变成一次抛硬币。**
        2026-08-10 的构建产物就是在这上面被认证的(约五成概率),
        虽然那个产物因为别的原因作废了,没有实害。
        ⇒ 发布预检因此加了一条:构建前那一跑,**套件里不许有已知 flaky**
        。
        """
        inst = FakeInstrument(canned=dict(_CANNED))
        app, cfg = self._run_app(inst, "real-nohitl-shape")
        snap = app.get_state(cfg)

        assert not snap.interrupts, (
            f"扇出分支里又出现了审批暂停:{snap.interrupts}")
        pulses = [args for m, args in inst.calls if m == "Bias_Pulse"]
        assert len(pulses) == 2, f"两发脉冲都该打出去;实际 {pulses}"
        # Bias_Pulse(Wait, width_s, bias_v, z_hold, abs_rel) → arg[2] is bias_v
        # 排序比较:值必须都对、都只出现一次;**顺序在扇出里没有定义**。
        assert sorted(p[2] for p in pulses) == [3.0, 4.0], (
            f"两发的值不对(顺序不算数,扇出里它没有定义):{pulses}")

    def test_both_pulses_leave_notices(self):
        """执行了就必须留痕:两发各一条,而不是「有一条就算」。

        按 subject 计数差值,不读最新一条 —— ``diagnostics`` 是进程级环形缓冲,
        同一次 pytest 里别的文件写进去的行会污染「读最新」。"""
        from mast.core import diagnostics as diag

        def _n() -> int:
            return len([r for r in diag.recent(500, kinds=("notice_only",))
                        if r.get("subject") == "BiasPulse"])

        before = _n()
        inst = FakeInstrument(canned=dict(_CANNED))
        self._run_app(inst, "real-nohitl-notice")
        assert _n() - before == 2, "两发脉冲应当各留下一条通知"

    def test_the_other_branch_still_completes(self):
        """literature 分支照常完成。

        旧理由是「不该被 IC 的审批提示挟持」;现在没有提示可挟持,但这条仍然值得
        钉 —— 它守的是扇出本身,而 ⑰ 动过 IC 的中间件链,那正是会碰坏扇出的地方。"""
        lit_sink: list = []
        inst = FakeInstrument(canned=dict(_CANNED))
        app, cfg = self._run_app(inst, "real-nohitl-2", lit_sink=lit_sink)

        assert lit_sink, "literature never ran"
        vals = app.get_state(cfg).values
        assert any("文献已检索" in t for t in _texts(vals.get("messages", []))), (
            "literature's handoff was lost")


# ═══════════════════════════════════════════════════════════════════════════
# The structural premise the operator confirmed — pinned, not assumed
# ═══════════════════════════════════════════════════════════════════════════

def test_only_instrument_control_holds_instrument_skills():
    """"仪器控制只是一个 agent，其他 agent 都不会参与仪器控制，因此并行是完全安全的."

    That sentence is the entire safety argument for concurrency. Pin it: derive
    every agent's REAL tool list and assert no agent except instrument_control
    holds a single skill from the instrument SkillRegistry. The day someone gives
    data_processing a hardware skill, this test — not the hardware — must be what
    tells them.
    """
    from mast.agents._shared.artifacts import agent_tool_names
    from mast.agents.instrument_control.tools import discover_instrument_skills

    hardware = {m.name for m in discover_instrument_skills().list_skills()}
    assert len(hardware) > 100, "the skill registry failed to load — vacuous test"

    for agent, tools in agent_tool_names().items():
        if agent == "instrument_control":
            assert tools & hardware, "instrument_control lost its instrument skills"
            continue
        leaked = tools & hardware
        assert not leaked, (
            f"{agent} holds instrument skill(s) {sorted(leaked)} — parallel dispatch "
            f"is only safe because instrument_control is the SOLE hardware agent")


@pytest.mark.parametrize("width", [2, 3, 4])
def test_wider_fanouts_all_land(width: int):
    """A 3- and 4-way fan-out must merge cleanly too — the reducers are what make
    N concurrent handoffs safe, and N=2 can pass by luck."""
    from mast.agents.orchestrator.graph import _supervisor_node_factory
    from mast.agents.state import MASTState
    from langgraph.graph import START, StateGraph

    names = ("literature", "data_processing", "paper_writing", "paper_review")[:width]
    from mast.agents._shared.handoff import make_handoff

    def _node(name: str):
        h = make_handoff("supervisor", "done")

        def _f(state):
            cmd = h.func(reason=f"{name} ok", tool_call_id=f"{name}-tc")
            upd = dict(cmd.update)
            upd["messages"] = [AIMessage(content=f"{name} ran")] + list(upd["messages"])
            upd["executed_skills"] = [f"{name}_skill"]
            return Command(goto="supervisor", update=upd)

        return _f

    sup = _ScriptedSupervisor([list(names), ["__end__"]])
    g: StateGraph = StateGraph(MASTState)
    g.add_node("supervisor", _supervisor_node_factory(sup, wired_agents=names))
    for n in names:
        g.add_node(n, _node(n))
    g.add_edge(START, "supervisor")
    app = g.compile(checkpointer=InMemorySaver())

    out = app.invoke(_state(), config={"configurable": {"thread_id": f"w{width}"},
                                       "recursion_limit": 50})
    assert set(out["executed_skills"]) == {f"{n}_skill" for n in names}
    assert not out.get("routing_hints")
