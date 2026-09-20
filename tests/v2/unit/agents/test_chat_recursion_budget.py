"""私聊递归预算应从当前图结构与工具调用预算派生。

模型节点、工具节点与 before_model/after_model 中间件分别占用 super-step；
wrap_tool_call 装饰器不新增图节点。测试覆盖多工具请求、StallGuard 可达性，
并防止中间件增减使固定递归上限悄悄改变可执行工作量。"""
from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from mast.agents._shared.call_limits import (
    DEFAULT_MODEL_CALLS_PER_RUN,
    MIN_RECURSION_LIMIT,
    derive_recursion_limit,
    super_steps_per_tool_call,
    turn_super_steps,
)
from mast.agents.instrument_control.graph import build
from mast.core.types import NanonisCallRecord

#: The literal that was in ``chat/engine.py`` from 2026-06-16 to 2026-08-04.
_OLD_LITERAL = 50


# ── fakes ────────────────────────────────────────────────────────────────────
class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel + no-op bind_tools (create_agent calls it)."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@dataclass
class _FakeCtx:
    """ExecutionContext stand-in. ``fail`` makes every call return the SAME error,
    which is what a genuine StallGuard-visible spin looks like."""

    fail: str = ""
    calls: list = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if self.fail:
            return NanonisCallRecord(method=method, args=args, error=self.fail)
        return NanonisCallRecord(method=method, args=args,
                                 return_value=(3e-12, 5e-8, 6e-5))


class _StubQueue:
    def get_nowait(self):
        raise queue.Empty


class _StubBuffer:
    """Enough BufferService surface for BufferHITLMiddleware to mount + peek."""

    def subscribe(self, kind):
        return _StubQueue()

    def unsubscribe(self, *a, **k):
        pass

    def read_latest(self, *a, **k):
        return None


class _StubRecorder:
    def __call__(self, *a, **k):
        pass

    def record_turn(self, *a, **k):
        pass


def _tool_call_script(n: int) -> list:
    """n sequential single-tool-call turns, then the final text answer."""
    out = [AIMessage(content="", tool_calls=[{"name": "GetZCtrlGain", "args": {},
                                              "id": f"c{i}"}])
           for i in range(n)]
    out.append(AIMessage(content="五步结果如下…"))
    return out


def _production_extra_middleware():
    """Mirror ``runtime._chat_agent_middleware()``'s SHAPE with no network.

    Only the node-producing members matter for a super-step count; the
    ``wrap_model_call`` ones (memory recall / readback / tip context) are
    included anyway so the list stays honest about what private chat mounts.
    """
    from mast.agents._shared.compaction_mw import make_compaction_middleware
    from mast.agents._shared.request_readback_mw import RequestReplyReadbackMiddleware
    from mast.agents._shared.tip_context_mw import TipContextMiddleware
    from mast.agents._shared.tool_refine_mw import ToolRefinementMiddleware

    summ = _FakeChatModel(messages=iter([AIMessage(content="s")] * 999))
    return [
        make_compaction_middleware(model_id="claude-sonnet-4-5", summarizer_model=summ),
        ToolRefinementMiddleware(summarizer_model=summ, min_chars=600),
        RequestReplyReadbackMiddleware(),
        TipContextMiddleware(),
    ]


def _build_graph(ctx, script, *, production: bool):
    """``production=True`` = the shape private chat actually builds on the instrument."""
    kw = dict(
        buf=None, context_provider=lambda: ctx,
        model=_FakeChatModel(messages=iter(script)),
        checkpointer=InMemorySaver(), enable_hitl=False, standalone=True,
        extra_middleware=[], max_model_calls=None, max_tool_calls=None,
        max_model_calls_per_run=DEFAULT_MODEL_CALLS_PER_RUN,
        max_tool_calls_per_run=80,
    )
    if production:
        kw.update(
            buf=_StubBuffer(),
            enable_hitl=True,
            interrupt_on={"SetBias": {"allowed_decisions": ["approve", "reject"]}},
            get_mode=lambda: "auto",
            extra_middleware=_production_extra_middleware(),
            turn_recorder=_StubRecorder(),
        )
    return build(**kw)


def _run(graph, recursion_limit: int, thread: str, text: str = "做这五步"):
    """Stream one turn; return (super_steps, error_or_None)."""
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": recursion_limit}
    steps = 0
    try:
        for _ in graph.stream({"messages": [HumanMessage(content=text)]},
                              config=cfg, stream_mode="updates"):
            steps += 1
    except GraphRecursionError as exc:
        return steps, exc
    return steps, None


# ── 1. the exchange rate is real, and it is len(nodes) - 2 ───────────────────
@pytest.mark.parametrize("production, expect_min", [(False, 5), (True, 8)])
def test_super_steps_per_tool_call_equals_node_count_minus_two(production, expect_min):
    """MEASURED by streaming, not asserted from the same arithmetic under test.

    This is the whole basis of the fix: if a future middleware changes the graph
    shape, the derived budget must follow it automatically. That only holds if
    ``len(nodes) - 2`` really is the per-round price.
    """
    ctx = _FakeCtx()
    g1 = _build_graph(ctx, _tool_call_script(1), production=production)
    n_nodes = len(g1.get_graph().nodes)
    predicted = super_steps_per_tool_call(g1)
    assert predicted == n_nodes - 2

    s1, err1 = _run(g1, 500, f"pc-{production}-1")
    s3, err3 = _run(_build_graph(_FakeCtx(), _tool_call_script(3), production=production),
                    500, f"pc-{production}-3")
    assert err1 is None and err3 is None
    measured = (s3 - s1) // 2
    assert measured == predicted, (
        f"per-round price mismatch: streamed {measured}, len(nodes)-2 = {predicted}. "
        "derive_recursion_limit() is now sizing the budget off a wrong exchange rate."
    )
    assert predicted >= expect_min


def test_turn_super_steps_formula_is_exact():
    """``steps_per_cycle * (n + 1) - 1`` — the final answer-only model call has no
    ``tools`` node after it, which is where the -1 comes from."""
    ctx = _FakeCtx()
    g = _build_graph(ctx, _tool_call_script(1), production=True)
    per = super_steps_per_tool_call(g)
    for n in (1, 2, 3):
        steps, err = _run(_build_graph(_FakeCtx(), _tool_call_script(n), production=True),
                          500, f"fx-{n}")
        assert err is None
        assert steps == turn_super_steps(per, n), (n, steps, turn_super_steps(per, n))


# ── 2. THE REPRODUCTION — a five-step request under the old literal 50 ───────
def test_five_step_request_fails_under_the_old_literal_50():
    """无重复、无失败的五步合成工具序列也需要足够的图执行预算；不能把预算不足误判为循环。"""
    ctx = _FakeCtx()
    g = _build_graph(ctx, _tool_call_script(5), production=True)
    per = super_steps_per_tool_call(g)
    needed = turn_super_steps(per, 5)
    assert needed > _OLD_LITERAL, (
        f"five steps need {needed} super-steps at {per}/round; the literal was "
        f"{_OLD_LITERAL}")

    steps, err = _run(g, _OLD_LITERAL, "repro-50")
    assert isinstance(err, GraphRecursionError), (
        "expected the operator's GraphRecursionError; got a clean finish")
    # Every tool call that DID run succeeded — no spin, no repeated failure.
    assert ctx.calls, "nothing ran at all"
    assert len({m for m, _ in ctx.calls}) == 1  # same tool, but each a NEW round


def test_the_old_literal_50_buys_only_a_handful_of_tool_calls():
    """把「隐含预算」写成一个数字：第 5 次工具调用就已经越界。

    这是回答「多步请求是不是普遍会这样」的证据 —— 是，而且是确定性的，不是偶发。

    **⑰ 之前这条断言的是 ``[1, 2, 3]``**（汇率 11）。割掉两个 HITL middleware 之后
    汇率降到 9，同一个 50 多买到一次调用。数字变了，结论一个字没变：这个悬崖的位置
    由**图形**决定，而没有人会在增删中间件时想起去核对一个写死的 50。断崖点因此从
    源码派生，不再硬编码 —— 否则下一次形状变化又会让这条测试变成假话。
    """
    per = super_steps_per_tool_call(
        _build_graph(_FakeCtx(), _tool_call_script(1), production=True))
    affordable = [n for n in range(1, 12) if turn_super_steps(per, n) <= _OLD_LITERAL]
    assert affordable and affordable[0] == 1, (
        f"at {per} super-steps/round a 50-step budget affords {affordable}")
    assert affordable[-1] < 9, (
        f"50 步竟然买得到 {affordable[-1]} 次工具调用 —— 若图真的瘦到这个程度，"
        f"请重新核对本模块 docstring 里的推导（当前 per={per}）")

    ok_n = affordable[-1]           # 刚好买得起
    over_n = ok_n + 1               # 刚好越界
    for n, expect_err in ((ok_n, False), (over_n, True)):
        _steps, err = _run(_build_graph(_FakeCtx(), _tool_call_script(n), production=True),
                           _OLD_LITERAL, f"cliff-{n}")
        assert (err is not None) is expect_err, (n, err)


def test_the_notice_middleware_costs_no_super_steps():
    """⑰ 的形状钉子:``wrap_tool_call``-only 的中间件**不占节点**。

    这是「换掉审批框之后反而更便宜」的机制:``before_model`` / ``after_model``
    各自会被 ``create_agent`` 编成一个独立节点(所以每次模型调用都多走一个
    super-step),而 ``wrap_*`` 是包在既有节点外的装饰,一步都不加。

    钉住它有两个用处:(1) 记下汇率为什么从 11 变回 9;(2) 下一次要加「观察类」
    中间件时,这里写着该用哪个钩子 —— 用错钩子的代价是每回合每次调用 +1 步,
    而那笔账不会出现在任何一次 diff 里。
    """
    from langchain.agents.middleware import AgentMiddleware

    from mast.agents._shared.auto_approval_mw import AutoApprovalNoticeMiddleware

    # ``hasattr`` 在这里没有信息量:基类把每个钩子都定义好了,所以恒为 True。
    # 决定「要不要建节点」的是**有没有覆写**。
    def _overrides(cls, hook: str) -> bool:
        return getattr(cls, hook, None) is not getattr(AgentMiddleware, hook, None)

    cls = AutoApprovalNoticeMiddleware
    for hook in ("before_model", "after_model", "abefore_model", "aafter_model"):
        assert not _overrides(cls, hook), (
            f"AutoApprovalNoticeMiddleware 覆写了 {hook} —— 它会变成一个节点，"
            f"每次模型调用多花 1 个 super-step；观察类中间件应当只用 wrap_tool_call")
    assert _overrides(cls, "wrap_tool_call") and _overrides(cls, "awrap_tool_call")

    # 而它确实装在生产图里 —— 否则上面那条断言只是在证明一个没人用的类很便宜。
    g = _build_graph(_FakeCtx(), _tool_call_script(1), production=True)
    node_names = {str(n) for n in g.get_graph().nodes}
    assert not any("AutoApprovalNotice" in n for n in node_names), node_names


def test_derived_budget_completes_the_same_five_step_request():
    """同一张图、同一条五步脚本，改用派生预算 → 跑完，五次工具调用全部执行。"""
    ctx = _FakeCtx()
    g = _build_graph(ctx, _tool_call_script(5), production=True)
    rl = derive_recursion_limit(g, model_calls_per_run=DEFAULT_MODEL_CALLS_PER_RUN)
    steps, err = _run(g, rl, "repro-derived")
    assert err is None, f"still failed at derived recursion_limit={rl} (used {steps})"
    assert len(ctx.calls) == 5, f"expected 5 hardware rounds, got {len(ctx.calls)}"


# ── 3. StallGuard: wired, correct, and unreachable at 50 ─────────────────────
class _CaptureStallGuard(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "StallGuard" in msg:
            self.lines.append(msg)


def _spin(recursion_limit: int, thread: str):
    """Drive a genuine same-tool/same-error spin; report what StallGuard managed."""
    cap = _CaptureStallGuard()
    lg = logging.getLogger("mast.agents._shared.stall_guard_mw")
    lg.addHandler(cap)
    old = lg.level
    lg.setLevel(logging.WARNING)
    try:
        ctx = _FakeCtx(fail="Z-Controller module not running")
        g = _build_graph(ctx, _tool_call_script(40), production=True)
        _steps, err = _run(g, recursion_limit, thread, text="把增益设好")
    finally:
        lg.removeHandler(cap)
        lg.setLevel(old)
    return {
        "nudges": [x for x in cap.lines if "nudge" in x],
        "stopped": [x for x in cap.lines if "STOPPING" in x],
        "error": err,
        "rounds": len(ctx.calls),
    }


def test_stall_guard_forced_stop_became_reachable_when_the_graph_shrank():
    """#31 的那个 guard **原来**在私聊里够不着自己的最后一级 —— ⑰ 之后够得着了。

    原始事实(2026-08-04,汇率 11):StallGuard 要 5 轮同错才能走完
    nudge→nudge→强制结束,5 轮 ≈ 65 super-step;50 步预算下递归上限先到,于是用户
    拿到的是裸 ``GraphRecursionError`` —— 正是这个 guard 存在的意义所要替换掉的那条
    报错。**接线对、判据对、够不着**。

    ⑰(2026-08-08)割掉两个 HITL middleware,汇率 11 → 9,同样的 50 步现在够走完那
    条阶梯了。这条测试因此反过来钉:强制结束**确实到达**。

    为什么不删掉它:这个 guard 的可达性是由**图形**决定的偶然事实,而不是有人设计
    过的保证。今天变得够得着,明天加两个中间件又会够不着,而那次同样不会有人注意到。
    真正的修复始终是「预算按图形派生」(下一条测试),这条只是记录汇率变化的下游
    后果 —— 以及它有多容易被反转。
    """
    out = _spin(_OLD_LITERAL, "spin-50")
    assert out["nudges"], "StallGuard did not even nudge — it may be unwired"
    assert out["stopped"], (
        "强制结束又够不着了 —— 图形多半又长胖了。这不是把这条断言改回去就完事的："
        "请重新核对本模块 docstring 里的汇率推导，并确认派生预算那条路仍然有效。")


def test_stall_guard_forced_stop_lands_once_the_budget_is_derived():
    """预算按图形派生之后，同一个 spin 由 StallGuard **可读地**结束，而不是撞上限。"""
    g = _build_graph(_FakeCtx(), _tool_call_script(1), production=True)
    rl = derive_recursion_limit(g, model_calls_per_run=DEFAULT_MODEL_CALLS_PER_RUN)
    out = _spin(rl, "spin-derived")
    assert out["stopped"], (
        f"StallGuard still never forced a stop at recursion_limit={rl}")
    assert out["error"] is None, "the guard must end the turn BEFORE the cap does"


# ── 4. the derivation helper itself ──────────────────────────────────────────
def test_derive_recursion_limit_scales_with_graph_shape():
    bare = _build_graph(_FakeCtx(), _tool_call_script(1), production=False)
    prod = _build_graph(_FakeCtx(), _tool_call_script(1), production=True)
    rl_bare = derive_recursion_limit(bare, model_calls_per_run=30)
    rl_prod = derive_recursion_limit(prod, model_calls_per_run=30)
    assert rl_prod > rl_bare, (
        "a heavier graph MUST get a bigger budget — that is the whole fix")
    assert rl_prod == super_steps_per_tool_call(prod) * 31


def test_derive_recursion_limit_is_fail_safe_and_bounded():
    class _Broken:
        def get_graph(self):
            raise RuntimeError("no graph")

    assert super_steps_per_tool_call(_Broken()) is None
    assert derive_recursion_limit(_Broken(), model_calls_per_run=30, fallback=300) == 300
    # never below the floor, whatever nonsense arrives
    assert derive_recursion_limit(_Broken(), model_calls_per_run=1,
                                  fallback=1) == MIN_RECURSION_LIMIT


def test_model_call_limit_becomes_reachable_at_the_derived_budget():
    """派生预算的判据：让**可读的**那个 guard 先触发，递归上限退回兜底位。

    50 步下 ModelCallLimitMiddleware(run_limit=30) 需要 330+ super-step 才谈得上
    触发 —— 它在私聊里从来没有生效过，尽管 call_limits.py 的注释称它为
    "the real in-turn circuit-breaker"。
    """
    prod = _build_graph(_FakeCtx(), _tool_call_script(1), production=True)
    per = super_steps_per_tool_call(prod)
    needed_for_model_cap = turn_super_steps(per, DEFAULT_MODEL_CALLS_PER_RUN - 1)
    assert needed_for_model_cap > _OLD_LITERAL, "sanity"
    rl = derive_recursion_limit(prod, model_calls_per_run=DEFAULT_MODEL_CALLS_PER_RUN)
    assert rl >= needed_for_model_cap, (
        f"derived budget {rl} still cannot reach the model-call cap "
        f"({needed_for_model_cap} needed) — the readable guard stays dead")


# ── 5. the engine actually uses it (wiring, not just arithmetic) ─────────────
def test_conversation_engine_derives_the_limit_from_the_built_graph():
    """``ConversationEngine`` 必须按它**真正建出来的那张图**定预算。

    这条测试的意义是「不再有字面量」：换一张更重的图，engine 给出的
    recursion_limit 必须跟着变大；否则加 middleware 又会静默扣掉可用步数。
    """
    from mast.chat.engine import ConversationEngine

    prod = _build_graph(_FakeCtx(), _tool_call_script(1), production=True)
    bare = _build_graph(_FakeCtx(), _tool_call_script(1), production=False)

    eng = ConversationEngine(graph_factory=lambda aid: prod, checkpointer=None,
                             store=None,
                             call_limits_provider=lambda: {"max_model_calls_per_run": 30})
    rl_prod = eng._recursion_limit_for("instrument_control", prod)
    rl_bare = eng._recursion_limit_for("literature", bare)

    assert rl_prod == super_steps_per_tool_call(prod) * 31
    assert rl_prod > rl_bare > 0
    assert rl_prod > _OLD_LITERAL, (
        "the engine is still handing the private chat a budget that cannot "
        "finish a four-step request")

    cfg = eng._config_for("instrument_control", prod, "T1")
    assert cfg["recursion_limit"] == rl_prod
    assert cfg["configurable"]["thread_id"] == "T1"

    # An explicit int still pins it (tests / a future operator override).
    pinned = ConversationEngine(graph_factory=lambda aid: prod, checkpointer=None,
                                store=None, recursion_limit=17)
    assert pinned._recursion_limit_for("instrument_control", prod) == 17


def test_conversation_engine_reprices_after_invalidate():
    """模型换了 / 图重建了之后必须重新测量，不能拿旧图的价钱继续算。"""
    from mast.chat.engine import ConversationEngine

    prod = _build_graph(_FakeCtx(), _tool_call_script(1), production=True)
    bare = _build_graph(_FakeCtx(), _tool_call_script(1), production=False)
    eng = ConversationEngine(graph_factory=lambda aid: prod, checkpointer=None,
                             store=None)
    first = eng._recursion_limit_for("instrument_control", prod)
    # Without invalidate the cache answers (cheap path).
    assert eng._recursion_limit_for("instrument_control", bare) == first
    eng.invalidate("instrument_control")
    assert eng._recursion_limit_for("instrument_control", bare) < first


# ── 6. 同形状扫描：其它把「一轮 = 2 步」写死的地方 ──────────────────────────
def test_orchestrator_recursion_floor_is_not_three_tool_calls():
    """``orchestrator_recursion_limit`` 的下限曾是 50 —— 同一个陷阱。

    subgraph 按**值**继承这个数字并花在自己的计数器上（实测：child 要 20 步时，
    parent limit=10 失败、25 通过），所以这个下限就是**每个 agent 自己**能拿到的
    预算。50 让用户可以把每个 agent 钉死在 3 次工具调用上。
    """
    from mast.api.routes import orchestrator as orch

    assert orch._MIN_USEFUL_RECURSION_LIMIT >= 150

    class _App:
        _settings = type("S", (), {"get": staticmethod(lambda k: 1)})()

    assert orch._effective_recursion_limit(_App()) == orch._MIN_USEFUL_RECURSION_LIMIT

    # An 11-steps/round graph must afford StallGuard's full 5-round ladder.
    per = super_steps_per_tool_call(
        _build_graph(_FakeCtx(), _tool_call_script(1), production=True))
    assert turn_super_steps(per, 5) <= orch._MIN_USEFUL_RECURSION_LIMIT


def test_composite_agent_node_budget_can_only_grow():
    """``2*(mmc+mtc)+10`` 也是「一轮 = 2 步」的定价。今天没饿着谁（74 > 39），
    但它必须能跟着图形变重而变大 —— 且**绝不允许比旧公式小**。"""
    from mast.skills.composite import agent_node

    mmc, mtc = 8, 24
    floor = 2 * (mmc + mtc) + 10
    heavy = _build_graph(_FakeCtx(), _tool_call_script(1), production=True)
    light = _build_graph(_FakeCtx(), _tool_call_script(1), production=False)

    for g in (heavy, light):
        rl = max(floor, derive_recursion_limit(g, model_calls_per_run=mmc,
                                               fallback=floor))
        assert rl >= floor, "the sweep must never SHRINK an existing budget"
        assert rl >= turn_super_steps(super_steps_per_tool_call(g), mmc - 1)

    assert hasattr(agent_node, "derive_recursion_limit"), (
        "agent_node no longer derives its budget — the literal is back")


# ── 7. what the failure leaves behind (question 4) ───────────────────────────
def test_recursion_error_does_not_defer_hardware_to_the_next_message():
    """回合被递归上限打断后，checkpoint 停在某个节点上（``next`` 非空）。

    安全上真正要问的是：下一条消息会不会把那个**待执行的 tools 节点**补跑掉，
    也就是把上一回合的硬件命令延迟执行到用户已经改主意之后。实测：不会 ——
    新输入到达时 LangGraph 丢弃 pending task。这条测试把这个行为钉死，因为它
    是「失败后状态干不干净」里唯一会动硬件的那一半。
    """
    ctx = _FakeCtx()
    saver = InMemorySaver()

    def _mk(script):
        return build(buf=None, context_provider=lambda: ctx,
                     model=_FakeChatModel(messages=iter(script)),
                     checkpointer=saver, enable_hitl=False, standalone=True,
                     extra_middleware=[], max_model_calls=None, max_tool_calls=None,
                     max_model_calls_per_run=30, max_tool_calls_per_run=80)

    # 9 lands exactly on a pending `tools` node for the bare shape (5/round).
    g = _mk(_tool_call_script(40))
    cfg = {"configurable": {"thread_id": "resid"}, "recursion_limit": 9}
    with pytest.raises(GraphRecursionError):
        for _ in g.stream({"messages": [HumanMessage(content="五步")]},
                          config=cfg, stream_mode="updates"):
            pass
    assert g.get_state(cfg).next == ("tools",), "test no longer lands on tools"
    ran_before = len(ctx.calls)

    g2 = _mk([AIMessage(content="好的，不动了")] * 20)
    cfg2 = {"configurable": {"thread_id": "resid"}, "recursion_limit": 300}
    for _ in g2.stream({"messages": [HumanMessage(content="算了，先别动")]},
                       config=cfg2, stream_mode="updates"):
        pass
    assert len(ctx.calls) == ran_before, (
        "a hardware call from the ABANDONED turn executed on the next message")
