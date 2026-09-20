"""agent 的每条消息带上**发生时刻** —— 而且「不知道」不许被写成 0。

## 需求背景

两个要求:一是旁白顺序乱(那一半由 ``narration.event_t`` + 前端按 ``t`` 排修掉),
二是**agent 的发言也要带上时间**。
这一句要的是**转录侧也有同一根时间轴** —— 有了它,「一条排在前面的消息时间更晚」
才会变成一个**看得见**的错,而不是一句「感觉有点乱」。

这也正是 ``frontend/src/lib/narration.ts`` 写了很久的那句话:

    ⚠️ 已知近似……要让它变精确,**得给 render_history 的每条消息加稳定时间戳**,
       那是另一个改动。

## 这个文件钉的五件

1. 这一轮新产生的消息**被盖上**时刻;
2. **重启前就存在的历史一条都不盖** —— 我们不知道它们是什么时候说的,
   盖 ``time.time()`` 会造出「所有历史都发生在重启那一秒」的假历史;
3. 盖过的**绝不重盖** —— 否则每次刷新都把历史改写成「刚刚」;
4. ``render_history`` 把它吐出来,而且**没有就不带这个键**(不带 0);
5. **它对 recursion 预算的成本是 0** —— 见第 5 节,那是第一版栽的地方。

第 2、5 条是这次改动真正的难点。第 1 条谁都会写。

## 为什么全部走 ``wrap_model_call`` 而不是内部方法

「校验不能交给会犯这个错的那一方」。第一版用 ``after_model`` 实现,单测
直接调内部方法 —— 9 条全绿,而闸门在另一个文件里红了两条:那个钩子会给图**加
一个节点**,每次模型调用多花一个 super-step。**测试从生产入口进,才拦得住
「接错钩子」这类错。**
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from mast.agents._shared.message_clock_mw import (  # noqa: E402
    MESSAGE_TIME_KEY,
    MessageClockMiddleware,
    message_time,
)
from mast.chat.render import render_history  # noqa: E402


class _Req:
    """最小 ModelRequest 替身:只有 ``messages``(和可选 runtime)。"""

    def __init__(self, messages, thread_id: str = "t1"):
        self.messages = messages
        self.runtime = type("_RT", (), {
            "config": {"configurable": {"thread_id": thread_id}}})()


# ── 1. 这一轮新产生的被盖上 ────────────────────────────────────────────────

def test_the_model_reply_is_stamped():
    """模型刚返回的那条**必须**盖上 —— 它是用户最常看的那一条。

    它的时刻是全流程里最确定的:handler 刚返回,现在就是它产生的时刻。
    所以它**不受基线约束**(基线防的是「把历史写成刚刚」,而这条本来就是刚刚)。
    """
    mw = MessageClockMiddleware()
    reply = AIMessage(content="好的")
    t0 = time.time()
    out = mw.wrap_model_call(_Req([HumanMessage(content="跑")]), lambda _r: reply)
    assert out is reply, "wrap 必须原样把结果传下去"
    t = message_time(reply)
    assert t is not None, "模型刚返回的这条没盖上时刻"
    assert t0 - 1 <= t <= time.time() + 1


def test_tool_results_arriving_next_round_are_stamped():
    """第二轮入站消息里多出来的那些(刚返回的工具结果)要盖上。"""
    mw = MessageClockMiddleware()
    h, a1 = HumanMessage(content="扫一张"), AIMessage(content="")
    mw.wrap_model_call(_Req([h, a1]), lambda _r: a1)
    tm = ToolMessage(content="ok", tool_call_id="c1")
    mw.wrap_model_call(_Req([h, a1, tm]), lambda _r: AIMessage(content="扫完了"))
    assert message_time(tm) is not None, "这一轮新到的工具结果没盖上时刻"


# ── 2. 历史一条都不盖(这条最重要)────────────────────────────────────────

def test_history_from_before_the_restart_is_never_stamped():
    """从 checkpoint 读回来的旧消息 ⇒ **不知道**它们是什么时候说的。

    给它们盖 ``time.time()`` 会造出一段「所有历史消息都发生在重启那一秒」的
    假历史,而前端会把它当真的画出来。**「不知道」和「等于现在」是两句话。**
    """
    mw = MessageClockMiddleware()
    # 真实 checkpoint 的历史长这样:一问一答收尾在 AIMessage 上。
    # ⚠️ 这里刻意**不用**「五条连续 HumanMessage」——那不是任何一次真实运行会
    # 产生的形状,而拿它当替身会让本条测试和「末尾 HumanMessage 要盖」那条例外
    # 打架,打输的是**测试**,不是代码(见 test_the_exception_does_not_reopen_the_history)。
    history = []
    for i in range(3):
        history += [HumanMessage(content=f"旧问 {i}"), AIMessage(content=f"旧答 {i}")]
    fresh = AIMessage(content="新的")
    mw.wrap_model_call(_Req(list(history)), lambda _r: fresh)
    for i, m in enumerate(history):
        assert message_time(m) is None, (
            f"第 {i} 条历史消息被盖上了「刚刚」—— 这就是那段假历史")
    assert message_time(fresh) is not None, "而模型刚返回的那条必须盖上"


def test_two_threads_do_not_share_a_baseline():
    """两个会话交替跑时,共用一个计数会让其中一个的基线永远偏。"""
    mw = MessageClockMiddleware()
    a_hist = [HumanMessage(content="A 旧"), AIMessage(content="A 旧答")]
    b_hist = []
    for i in range(4):
        b_hist += [HumanMessage(content=f"B 旧问 {i}"), AIMessage(content=f"B 旧答 {i}")]
    mw.wrap_model_call(_Req(a_hist, "A"), lambda _r: AIMessage(content="A 新"))
    b_new = AIMessage(content="B 新")
    mw.wrap_model_call(_Req(b_hist, "B"), lambda _r: b_new)
    assert message_time(b_new) is not None
    for m in b_hist:
        assert message_time(m) is None, (
            "B 的历史被 A 的基线波及了 —— A 只有 2 条而 B 有 8 条,"
            "共用一个计数会让 B 的前 2 条被当成「已见过」而后 6 条被当成新增")


# ── 3. 不重盖 ──────────────────────────────────────────────────────────────

def test_an_existing_stamp_is_never_overwritten():
    """否则每次刷新都会把整段历史改写成「刚刚」。"""
    mw = MessageClockMiddleware()
    old_t = 1_700_000_000.0
    m = AIMessage(content="早就说过了")
    m.additional_kwargs[MESSAGE_TIME_KEY] = old_t
    for _ in range(3):
        mw.wrap_model_call(_Req([m]), lambda _r: m)
    assert message_time(m) == old_t


def test_stamping_never_raises():
    """一个时间戳绝不许弄坏一个回合 —— 请求侧和返回侧都是。"""
    mw = MessageClockMiddleware()
    for bad_req in (type("_X", (), {"messages": None})(),
                    type("_X", (), {"messages": []})(),
                    type("_X", (), {"messages": [object()]})(),
                    type("_X", (), {})()):
        mw.wrap_model_call(bad_req, lambda _r: AIMessage(content="ok"))
    for weird in (None, 42, "text", {"nope": 1}, object()):
        # 上游返回形状变了 ⇒ 记一条 debug,**不抛**。
        mw.wrap_model_call(_Req([HumanMessage(content="x")]),
                           lambda _r, w=weird: w)


# ── 4. render_history 吐出来,没有就不带 ──────────────────────────────────

def test_render_history_carries_the_time_when_it_is_known():
    m = AIMessage(content="打完了")
    m.additional_kwargs[MESSAGE_TIME_KEY] = 1_786_541_601.9
    out = render_history([HumanMessage(content="跑"), m])
    ai = [e for e in out if e["role"] == "assistant"]
    assert ai and ai[-1].get("t") == pytest.approx(1_786_541_601.9)


def test_render_history_omits_the_key_rather_than_writing_zero():
    """**没有就不带这个键。** 一个 0 会被前端画成 1970 年。

    前端 ``fmtClock`` 对无效值返回空串,但那是第二道防线 —— 第一道是这里
    压根不产出一个假数。两道都要有:本仓的兜底值一次次落在「没什么可担心的」
    那一侧,靠的就是「上游不产假值 + 下游不信假值」两条一起。
    """
    out = render_history([HumanMessage(content="跑"), AIMessage(content="好")])
    for e in out:
        assert "t" not in e, f"没有时刻却带出了 t={e.get('t')!r}"


def test_the_key_spelling_matches_what_render_reads():
    """平价测试:``render.py`` 刻意不 import agent 包,两处拼写必须钉在一起。

    与 ``COMPACTION_META_KEY`` 同一条规矩 —— 而那条规矩存在的理由正是
    「一个写错的字面量不会报错,只会让这个字段永远缺席」。
    """
    from mast.chat import render as render_mod
    m = AIMessage(content="x")
    m.additional_kwargs[MESSAGE_TIME_KEY] = 123.0
    assert render_mod._msg_t(m) == 123.0, (
        f"render 读不到 {MESSAGE_TIME_KEY} —— 两处拼写分家了")


def test_the_api_schema_carries_it_too():
    """``ChatMessage`` 只搬它声明过的字段 —— 少一行,这个键在传输层被静默丢掉。

    本仓「生产方接上了、消费方不存在」的同形第五次;这次是先查了才写。
    """
    from mast.api.schemas_agents import ChatMessage
    assert ChatMessage(role="a", content="b", t=1.5).model_dump()["t"] == 1.5
    # 不知道就不带这个键(而不是带一个 0)
    assert "t" not in ChatMessage(role="a", content="b").model_dump(exclude_none=True)


# ── 5. 它对 recursion 预算的成本必须是 0 ─────────────────────────────────
#
# 第一版用了 ``after_model``,闸门当场红了两条(``test_chat_recursion_budget``),
# 其中一条的断言消息逐字写着:
#
#     「强制结束又够不着了 —— 图形多半又长胖了。**这不是把这条断言改回去就
#       完事的**:请重新核对本模块 docstring 里的汇率推导。」
#
# 汇率是 ``len(nodes) - 2``:每一个覆写了 ``before_model`` / ``after_model`` 的
# 中间件都会被 ``create_agent`` 编成**自己的一个节点**,于是每次模型调用多走一个
# super-step。而 ``wrap_*`` 是包在既有节点**外面**的装饰,一步都不加。
#
# 这条测试照抄 ``test_the_notice_middleware_costs_no_super_steps`` 的形状 ——
# 那里逐字写着「以后加中间件时该照抄的形状」,而我没照抄,所以补一条同形的钉子。

def test_the_clock_middleware_costs_no_super_steps():
    """时间戳是**观察类**中间件 ⇒ 只许用 ``wrap_*``,一个节点都不许加。"""
    from langchain.agents.middleware import AgentMiddleware

    def _overrides(cls, hook: str) -> bool:
        # ``hasattr`` 没有信息量:基类把每个钩子都定义好了,恒为 True。
        # 决定「要不要建节点」的是**有没有覆写**。
        return getattr(cls, hook, None) is not getattr(AgentMiddleware, hook, None)

    cls = MessageClockMiddleware
    for hook in ("before_model", "after_model", "abefore_model", "aafter_model"):
        assert not _overrides(cls, hook), (
            f"MessageClockMiddleware 覆写了 {hook} —— 它会变成一个节点,"
            "每次模型调用多花 1 个 super-step,而那笔账不会出现在任何一次 diff 里。"
            "观察类中间件只许用 wrap_*。")
    assert _overrides(cls, "wrap_model_call"), "根本没接上任何钩子"
    assert _overrides(cls, "awrap_model_call"), (
        "缺异步孪生 —— LangChain 基类的 awrap_model_call 在只定义同步版时会抛,"
        "于是每一次异步 dispatch 都崩(prefill_guard / tool_pair_guard 同一条教训)")


def test_the_operators_own_new_message_gets_a_time():
    """新会话里用户**自己刚发的那句**必须有时间。

    基线规则(首次见到一个 thread 就不盖)在「重启后的历史」上是对的,
    但它顺手把这一条也挡了 —— 真图实测:6 条消息盖 5 条,漏的正是它。
    而它是**确定**的:模型此刻正被调用,而消息列表以一条用户消息结尾
    ⇒ 那就是触发这一轮的那条。
    """
    mw = MessageClockMiddleware()
    ask = HumanMessage(content="跑一下")
    mw.wrap_model_call(_Req([ask]), lambda _r: AIMessage(content="好"))
    assert message_time(ask) is not None, (
        "新会话里用户自己那句没有时间 —— 而那是他最想对得上的一条")


def test_the_exception_does_not_reopen_the_history():
    """但这条例外**只适用于末尾那一条** —— 历史仍然一条都不盖。

    没有这一条,上面那个例外很容易被写成「把所有 HumanMessage 都盖上」,
    而那正是那段假历史。
    """
    mw = MessageClockMiddleware()
    old_human = [HumanMessage(content=f"旧问题 {i}") for i in range(3)]
    tail = HumanMessage(content="新问题")
    mw.wrap_model_call(_Req([*old_human, AIMessage(content="旧回答"), tail]),
                       lambda _r: AIMessage(content="新回答"))
    assert message_time(tail) is not None
    for i, m in enumerate(old_human):
        assert message_time(m) is None, f"第 {i} 条历史提问被盖上了「刚刚」"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# 会话时间戳基线必须按真实 Runtime 提供的会话标识隔离。
# 测试替身不得自行增加生产 Runtime 没有的 config 字段。
# 若标识读取退化为空字符串，新会话会误用其他会话的计数，漏掉首条消息时间戳。


def _real_runtime():
    """真的 `langgraph.runtime.Runtime` —— 不是替身。"""
    from langgraph.runtime import Runtime

    return Runtime(context=None, store=None, stream_writer=None, previous=None)


class _RealShapeReq:
    """`ModelRequest` 的最小形状,但 `runtime` 是**真的** Runtime 对象。"""

    def __init__(self, messages):
        self.messages = messages
        self.runtime = _real_runtime()


def test_the_double_never_gives_runtime_an_attribute_the_real_one_lacks():
    """闸门自检:`_Req` 造出来的 runtime 不许比真货多东西。

    这条测试存在的全部理由是:上面那个替身多了一个 `config`,而多出来的那一层
    正好盖住了唯一的失效路径。**替身比真货宽容,断言就是空的。**
    """
    real = _real_runtime()
    fake = _Req([HumanMessage(content="x")], "t1").runtime
    extra = [a for a in ("config", "configurable")
             if hasattr(fake, a) and not hasattr(real, a)]
    assert extra, (
        "替身现在不再多出 config/configurable 了 —— 那说明它被改成真形状了,"
        "这条自检可以删。但**别默默删**:先确认 _thread_of 仍有一条走得通的路。")
    assert not hasattr(real, "config"), (
        "真的 Runtime 现在有 config 了 —— `_thread_of` 的第一条路通了,"
        "下面那几条测试的前提变了")


def test_the_baseline_key_survives_the_real_runtime_object():
    """有会话 id 在手时,账本键**必须**是那条会话,不是空串。"""
    from mast.core.turn_context import turn_scope

    with turn_scope(conversation_id="conv-A", run_id="r1"):
        assert MessageClockMiddleware._thread_of(_real_runtime()) == "conv-A"
    with turn_scope(conversation_id="conv-B", run_id="r1"):
        assert MessageClockMiddleware._thread_of(_real_runtime()) == "conv-B"


def test_a_fresh_conversation_stamps_the_operators_own_first_line():
    """**这一条就是那个 bug。**

    先跑一条长会话(把基线推高),再开一条新会话 —— 新会话里用户自己发的
    第一句必须有时间。在修好之前它是 None,而那正是「上下文注入显示不出来」。
    """
    from mast.core.turn_context import turn_scope

    mw = MessageClockMiddleware()
    long_hist = []
    for i in range(6):
        long_hist += [HumanMessage(content=f"旧问 {i}"), AIMessage(content=f"旧答 {i}")]
    with turn_scope(conversation_id="conv-old", run_id="r1"):
        mw.wrap_model_call(_RealShapeReq(long_hist),
                           lambda _r: AIMessage(content="旧回答"))

    ask = HumanMessage(content="针尖坏了,请自主恢复")
    with turn_scope(conversation_id="conv-new", run_id="r2"):
        mw.wrap_model_call(_RealShapeReq([ask]),
                           lambda _r: AIMessage(content="好的"))
    assert message_time(ask) is not None, (
        "新会话里用户自己那句没有时间 —— 注入折叠块会永远对不上它,"
        "而且会把原因说成「重启前的历史」")


def test_the_two_conversations_do_not_share_a_baseline_for_real():
    """`test_two_threads_do_not_share_a_baseline` 的真形状孪生。

    那一条用的是带 `config` 的替身,所以它证明的是替身的行为。
    """
    from mast.core.turn_context import turn_scope

    mw = MessageClockMiddleware()
    a_hist = [HumanMessage(content="A 旧"), AIMessage(content="A 旧答")]
    b_hist = []
    for i in range(4):
        b_hist += [HumanMessage(content=f"B 旧问 {i}"), AIMessage(content=f"B 旧答 {i}")]
    with turn_scope(conversation_id="A", run_id="r1"):
        mw.wrap_model_call(_RealShapeReq(a_hist), lambda _r: AIMessage(content="A 新"))
    b_new = AIMessage(content="B 新")
    with turn_scope(conversation_id="B", run_id="r1"):
        mw.wrap_model_call(_RealShapeReq(b_hist), lambda _r: b_new)
    assert message_time(b_new) is not None
    for m in b_hist:
        assert message_time(m) is None, (
            "B 的历史被 A 的基线波及了 —— 两条会话共用了一个计数")


def test_the_trailing_human_is_stamped_even_when_the_key_is_unknown():
    """账本键取不到(没有会话 id)时,末尾那条用户消息**仍然**要盖上。

    这是一道安全网:键的解析是会退化的(它已经退化过一次,整整一段时间),
    而「模型正在被调用、列表以用户消息结尾」这个事实与键无关。
    """
    mw = MessageClockMiddleware()
    mw.wrap_model_call(_RealShapeReq([HumanMessage(content="先跑一条")]),
                       lambda _r: AIMessage(content="好"))
    ask = HumanMessage(content="再开一条")
    mw.wrap_model_call(_RealShapeReq([ask]), lambda _r: AIMessage(content="好"))
    assert message_time(ask) is not None


def test_the_safety_net_still_does_not_stamp_the_history():
    """安全网只管**末尾那一条** —— 历史仍然一条都不盖。

    没有这条,上面那道网很容易被写成「把所有 HumanMessage 都盖上」,
    而那正是这个模块从第一天就在防的那段假历史。
    """
    mw = MessageClockMiddleware()
    old = [HumanMessage(content=f"旧问 {i}") for i in range(3)]
    tail = HumanMessage(content="新问题")
    mw.wrap_model_call(_RealShapeReq([*old, AIMessage(content="旧答"), tail]),
                       lambda _r: AIMessage(content="新答"))
    assert message_time(tail) is not None
    for i, m in enumerate(old):
        assert message_time(m) is None, f"第 {i} 条历史提问被盖上了「刚刚」"
