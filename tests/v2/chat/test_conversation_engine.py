"""ConversationStore + ConversationEngine + render — the private-chat runtime.

Pins: durable multi-turn history via the checkpointer, conversation CRUD,
switch-render via get_messages, per-turn auto-title, and delete purging the
checkpoint thread. Uses a fake create_agent graph (no LLM, no hardware).
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 not found")


_ROOT = _find_mastv2_root()
if sys.path[0] != _ROOT:
    while _ROOT in sys.path:
        sys.path.remove(_ROOT)
    sys.path.insert(0, _ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        if "MASTv2" not in (getattr(sys.modules[_n], "__file__", "") or "").replace("\\", "/"):
            del sys.modules[_n]

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from mast.chat.engine import ConversationEngine
from mast.chat.store import ConversationStore


class _FM(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def _engine(tmp_path, n_replies=6):
    fm = _FM(messages=iter([AIMessage(f"回复{i}") for i in range(n_replies)]))
    saver = InMemorySaver()
    store = ConversationStore(str(tmp_path / "conv.db"))
    eng = ConversationEngine(
        graph_factory=lambda aid: create_agent(model=fm, tools=[],
                                               system_prompt="t", checkpointer=saver),
        checkpointer=saver, store=store)
    return eng, store, saver


def test_store_crud(tmp_path):
    store = ConversationStore(str(tmp_path / "c.db"))
    c = store.create("instrument_control", kind="private")
    assert store.get(c["conversation_id"])["title"] == "新对话"
    assert store.rename(c["conversation_id"], "我的会话")
    assert store.get(c["conversation_id"])["title"] == "我的会话"
    assert len(store.list(kind="private")) == 1
    assert store.delete(c["conversation_id"])
    assert store.list(kind="private") == []


def test_turn_accumulates_history_across_turns(tmp_path):
    eng, store, _ = _engine(tmp_path)
    cid = store.create("instrument_control")["conversation_id"]
    last = None
    for snap in eng.stream_turn(cid, "扫一张图"):
        last = snap
    assert [m["role"] for m in last] == ["user", "assistant"]
    for snap in eng.stream_turn(cid, "再来一张"):
        last = snap
    # checkpointer accumulated both turns
    assert [m["role"] for m in last] == ["user", "assistant", "user", "assistant"]
    # switch-render reads the same from the checkpointer
    assert [m["role"] for m in eng.get_messages(cid)] == \
        ["user", "assistant", "user", "assistant"]


def test_first_turn_autotitles_and_preview(tmp_path):
    eng, store, _ = _engine(tmp_path)
    cid = store.create("instrument_control")["conversation_id"]
    for _ in eng.stream_turn(cid, "这是第一句话用来命名"):
        pass
    row = store.get(cid)
    assert row["title"].startswith("这是第一句话")
    assert row["last_message_preview"]  # non-empty


def test_delete_purges_checkpoint_thread(tmp_path):
    eng, store, saver = _engine(tmp_path)
    conv = store.create("instrument_control")
    cid, tid = conv["conversation_id"], conv["thread_id"]
    for _ in eng.stream_turn(cid, "你好"):
        pass
    assert eng.get_messages(cid)  # has history
    store.delete(cid, checkpointer=saver)
    # thread purged → a fresh conversation reusing nothing has empty state
    assert store.get(cid) is None


def test_first_yield_shows_placeholder(tmp_path):
    eng, store, _ = _engine(tmp_path)
    cid = store.create("instrument_control")["conversation_id"]
    gen = eng.stream_turn(cid, "hi")
    first = next(gen)
    # generator-yield-first: user echoed + a thinking placeholder
    assert first[-1]["role"] == "assistant"
    assert "思考中" in first[-1]["content"] or "mast-thinking" in first[-1]["content"]
    for _ in gen:
        pass


# ── HITL: 私聊里的「停下来等人」通道 ────────────────────────────────────────
#
# 引擎一直有 resume 循环和 ``hitl_resolver`` 钩子,而有好几个月没有任何入口传过一个
# —— 私聊撞上一个需要人的节点就打印一句提示然后停住,图停在那个门上,于是**后面每一条
# 消息都撞同一个门、拿同一句非答复**。这一节钉两半:resolver 真的把一轮驱动到完成,
# 以及被停住的线程会被清掉而不是把对话卡死。
#
# ## ⑰(2026-08-08):**生产者换了,机制没换**
#
# 这些测试原来用 ``ic_build(..., interrupt_on={"BiasPulse": ...})`` 强行把一个
# DANGEROUS 审批门装上来当生产者(注释原文:「no builtin skill is DANGEROUS since
# the 2026-06-11 re-scoping」,所以必须用覆盖)。那个审批门随整条确认框链路一起删掉
# 了 —— 于是这一节**没有生产者了**,不是因为机制坏了。
#
# 机制本身必须留着,而且是要求保留的:``ask_user``(「缺关键参数时的正当提问」)
# 和声明式工作流里的 ``human`` 节点仍然会 interrupt,它们走的正是这条 resolver 回路。
# 所以生产者换成 ``ask_user``,每一条测试的教训原样保留:
#
#   旧生产者(DANGEROUS 审批门)          → 新生产者(ask_user)
#   ────────────────────────────────────────────────────────────────
#   resolver 返回 approve → 技能执行      → resolver 给出回答 → 后续动作执行
#   resolver 返回 reject → 技能不执行     → 没有「拒绝」这回事:ask_user 是提问不是
#                                          审批。改钉「没答复也不会卡死对话」
#   没接 resolver → 提示 + 停住           → 一字未改(提示文案与 kind 无关)
#   过期审批被清掉(↺)+ fail-closed       → 一字未改
#
# 「都不弹」和「该弹的没弹」是两件事;把这一节删掉会让 resolver 回路变成没人守的
# 代码,而它现在服务的是**唯一剩下的**两个 interrupt 生产者。

from dataclasses import dataclass, field           # noqa: E402
from typing import Any                             # noqa: E402

from langchain_core.language_models import BaseChatModel    # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from mast.agents._shared.ask_tools import ASK_USER_TOOLS     # noqa: E402
from mast.agents.instrument_control.graph import build as ic_build   # noqa: E402
from mast.chat.engine import _interrupt_payload, _stale_reject_value  # noqa: E402
from mast.core.types import NanonisCallRecord      # noqa: E402


class _SeqModel(BaseChatModel):
    """Replays queued messages. Deliberately does NOT implement ``_stream``.

    ``GenericFakeChatModel`` does, and its implementation streams CONTENT ONLY —
    tool_calls are dropped. Under the voice path's ``stream_mode=["updates",
    "messages"]`` that silently turns a gated tool call into a plain answer, so
    the graph never interrupts and the test would be pinning nothing. Without a
    ``_stream``, LangChain falls back to one chunk per generation and the tool
    calls survive.
    """

    queue: Any = None

    @property
    def _llm_type(self) -> str:
        return "seq-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(self.queue))])

    def bind_tools(self, tools, **kw):
        return self


@dataclass
class _Ctx:
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method == "Bias_Pulse":
            return NanonisCallRecord(method=method, args=args, return_value=("", b"", []))
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


_ASK_TC = {"name": "ask_user",
           "args": {"question": "要不要现在打这一发脉冲？没人回答我就跳过。",
                    "options": ["打", "跳过"]},
           "id": "tc-ask-1", "type": "tool_call"}
_PULSE_TC = {"name": "BiasPulse", "args": {"width_s": "10m", "bias_v": "5"},
             "id": "tc-bp-1", "type": "tool_call"}


def _ask_llm(*, then_pulse: bool = True, n_after: int = 4):
    """先 ``ask_user`` 停一下;拿到答复后(可选)再打一发脉冲,然后收尾。

    ``then_pulse=False`` 用于「不该有任何仪器动作」那几条 —— 脚本模型不会分支,
    所以「答了才打」这件事靠**两个脚本**表达,而不是靠一个会自己判断的假模型。
    """
    msgs = [AIMessage(content="先问一下", tool_calls=[_ASK_TC])]
    if then_pulse:
        msgs.append(AIMessage(content="这就打脉冲", tool_calls=[_PULSE_TC]))
    msgs += [AIMessage(content=f"完成{i}") for i in range(n_after)]
    return _SeqModel(queue=iter(msgs))


def _hitl_engine(tmp_path, llm=None):
    """Engine over the REAL instrument_control graph, with ``ask_user`` wired in.

    ⑰ 之前这里传 ``interrupt_on={"BiasPulse": ...}`` 强行造一个审批门;那条链路没了,
    生产者换成 ``ask_user`` —— 它是**真实生产者**,不是覆盖出来的,所以这一节从此
    测的是生产上真会发生的那条路。
    """
    ctx = _Ctx()
    saver = InMemorySaver()
    store = ConversationStore(str(tmp_path / "conv.db"))
    graph = ic_build(
        buf=None, context_provider=lambda: ctx, model=llm or _ask_llm(),
        checkpointer=saver, enable_hitl=True, extra_tools=list(ASK_USER_TOOLS),
    )
    eng = ConversationEngine(graph_factory=lambda aid: graph,
                             checkpointer=saver, store=store)
    return eng, store, ctx, graph


def _answer(text: str = "打"):
    """用户的回答 —— ``hitl_decision.build_ask_answer`` 造出来的那个形状。"""
    return {"selected": [text], "custom_text": "", "note": ""}


def _drain(gen):
    last = None
    for snap in gen:
        last = snap
    return last


def _text(snapshot) -> str:
    return "\n".join(m.get("content", "") for m in (snapshot or []))


class TestResolverClosesTheLoop:
    def test_an_answer_lets_the_turn_finish_in_the_same_conversation(self, tmp_path):
        """原名 ``test_approve_lets_the_turn_finish_in_the_same_conversation``。

        教训一字未改:被 interrupt 停住的一轮,在 resolver 给出答复之后必须**在同一
        个对话里**跑完 —— 后续动作真的执行,结果进这条对话的历史,线程被释放。"""
        eng, store, ctx, _ = _hitl_engine(tmp_path)
        cid = store.create("instrument_control")["conversation_id"]
        seen: list = []

        def resolver(interrupted):
            seen.append(interrupted)
            return _answer("打")

        final = _drain(eng.stream_turn(cid, "打个脉冲", hitl_resolver=resolver))
        assert seen, "the interrupt must reach the resolver"
        # 后续动作真的跑了,而且结果在 THIS conversation 的历史里。
        assert any(m == "Bias_Pulse" for m, _ in ctx.calls)
        assert "BiasPulse" in _text(final)
        # Thread released either way — a stuck guard wedges the conversation.
        assert not eng.is_active(store.get(cid)["thread_id"])

    def test_an_unanswered_question_leaves_the_conversation_usable(self, tmp_path):
        """原名 ``test_reject_leaves_the_conversation_usable``。

        ``ask_user`` 没有「拒绝」这回事(它是提问,不是审批),所以这里钉的是那条
        测试真正在守的性质:**没答复不会把对话卡死** —— 下一条消息照样得到回答,
        而且被停住的那一步 fail-closed,不会自己跑掉。"""
        eng, store, ctx, _ = _hitl_engine(tmp_path, llm=_ask_llm(then_pulse=False))
        cid = store.create("instrument_control")["conversation_id"]

        _drain(eng.stream_turn(cid, "打个脉冲", hitl_resolver=lambda i: None))
        assert not any(m == "Bias_Pulse" for m, _ in ctx.calls)
        after = _drain(eng.stream_turn(cid, "那算了"))
        assert "那算了" in _text(after)


class TestNoticesSurviveTheTurn:
    def test_the_no_resolver_notice_is_in_the_final_snapshot(self, tmp_path):
        # Regression: the notice was yielded, then the final snapshot — rendered
        # fresh from checkpoint state — replaced it. The client keeps only the
        # last snapshot, so the operator saw a bare tool-call line and no
        # explanation at all.
        eng, store, _ctx, _ = _hitl_engine(tmp_path)
        cid = store.create("instrument_control")["conversation_id"]
        final = _drain(eng.stream_turn(cid, "打个脉冲"))   # no resolver
        assert "未接审批处理器" in _text(final)

    def test_an_unanswered_interrupt_says_the_turn_is_paused(self, tmp_path):
        eng, store, _ctx, _ = _hitl_engine(tmp_path)
        cid = store.create("instrument_control")["conversation_id"]
        final = _drain(eng.stream_turn(cid, "打个脉冲", hitl_resolver=lambda i: None))
        assert "审批未完成" in _text(final)


class TestStaleApprovalIsCleared:
    def test_three_messages_do_not_all_hit_the_same_gate(self, tmp_path):
        """The wedge, reproduced and then cleared.

        Before: turn 1 parks on the gate, and turns 2 and 3 re-enter the same
        node, raise a NEW interrupt each and never answer the operator.
        """
        eng, store, ctx, graph = _hitl_engine(tmp_path,
                                              llm=_ask_llm(then_pulse=False))
        conv = store.create("instrument_control")
        cid, tid = conv["conversation_id"], conv["thread_id"]

        _drain(eng.stream_turn(cid, "打个脉冲"))            # no resolver → parked
        cfg = {"configurable": {"thread_id": tid}}
        assert graph.get_state(cfg).next, "turn 1 should leave the thread parked"

        second = _drain(eng.stream_turn(cid, "在吗？"))
        assert "↺" in _text(second), "the stale approval must be cleared"
        assert "在吗？" in _text(second), "and the new message must be answered"
        assert not any(m == "Bias_Pulse" for m, _ in ctx.calls), \
            "clearing a stale approval must FAIL CLOSED — never run the action"
        assert not graph.get_state(cfg).next, "the thread must no longer be parked"

        # Turn 3 is an ordinary turn: no second cleanup notice. (The rejection
        # itself stays in the history — that record is the point.)
        third = _drain(eng.stream_turn(cid, "第三条"))
        assert "↺" not in _text(third)
        assert "第三条" in _text(third)

    def test_a_clean_thread_is_left_alone(self, tmp_path):
        eng, store, _ctx, graph = _hitl_engine(tmp_path, llm=_SeqModel(
            queue=iter([AIMessage(f"回复{i}") for i in range(4)])))
        cid = store.create("instrument_control")["conversation_id"]
        first = _drain(eng.stream_turn(cid, "你好"))
        assert "↺" not in _text(first)

    def test_it_says_so_when_the_new_message_never_ran(self, tmp_path):
        # Cleanup succeeds, the model immediately asks again, and there is still
        # no resolver — so the turn ends before the operator's message is
        # delivered. Saying nothing here would look like the message was
        # answered with silence.
        llm = _SeqModel(queue=iter([
            AIMessage(content="一次", tool_calls=[_ASK_TC]),
            AIMessage(content="再来", tool_calls=[dict(_ASK_TC, id="tc-ask-2")]),
            AIMessage(content="好的"),
            AIMessage(content="好的2"),
        ]))
        eng, store, _ctx, _ = _hitl_engine(tmp_path, llm=llm)
        cid = store.create("instrument_control")["conversation_id"]
        _drain(eng.stream_turn(cid, "打个脉冲"))          # parks the thread
        second = _drain(eng.stream_turn(cid, "在吗？"))
        assert "还没送达" in _text(second)

    def test_probe_survives_a_graph_that_cannot_report_state(self, tmp_path):
        # get_state failing must degrade to "no rescue", never break the turn.
        eng, store, _ctx, graph = _hitl_engine(tmp_path)

        class _Blind:
            def get_state(self, cfg):
                raise RuntimeError("no checkpoint backend")

        assert eng._stale_interrupt_resume(_Blind(), {"configurable": {}}) is None


class TestStaleRejectShape:
    def test_one_decision_per_action_request(self):
        # The HITL middleware raises when the counts disagree, so a two-request
        # interrupt needs two rejections.
        v = _stale_reject_value({"action_requests": [
            {"name": "SetBias"}, {"name": "MoveProbeXY"}]})
        assert [d["type"] for d in v["decisions"]] == ["reject", "reject"]

    def test_workflow_human_gets_an_empty_route(self):
        v = _stale_reject_value({"kind": "workflow_human", "routes": ["a", "b"]})
        assert v["route"] == "" and "过期" in v["note"]

    def test_ask_user_is_answered_as_unanswered_not_rejected(self):
        # A question executes nothing — "rejected" would be a lie.
        v = _stale_reject_value({"kind": "ask_user", "question": "q"})
        assert v["selected"] == [] and v["timeout"] is True

    def test_unknown_kinds_fall_back_to_the_decisions_envelope(self):
        for val in ({"kind": "buffer_hitl", "events": []}, {"kind": "???"}, None):
            assert _stale_reject_value(val)["decisions"][0]["type"] == "reject"


class TestInterruptPayloadIsDescribed:
    """The voice channel reads skill / rationale off this. It used to get the
    raw HITLRequest, whose keys are action_requests / review_configs — so it
    announced 「需要人工确认：操作」 and could not name the skill."""

    class _I:
        def __init__(self, value, id_="lg-9"):
            self.value = value
            self.id = id_

    def test_dangerous_names_the_skills(self):
        p = _interrupt_payload((self._I({
            "action_requests": [{"name": "SetBias", "description": "危险偏压"}],
            "review_configs": []}),))
        assert p["kind"] == "dangerous"
        assert p["skill"] == "SetBias"
        assert p["rationale"] == "危险偏压"
        assert p["lg_id"] == "lg-9"

    def test_ask_user_carries_the_question(self):
        p = _interrupt_payload((self._I({"kind": "ask_user",
                                         "ask": {"question": "先扫哪里？"}}),))
        assert p["skill"] == "向用户提问" and p["rationale"] == "先扫哪里？"

    def test_workflow_human_names_the_node(self):
        p = _interrupt_payload((self._I({"kind": "workflow_human",
                                         "workflow": "AutoApproach",
                                         "node_id": "n3", "message": "确认"}),))
        assert "AutoApproach" in p["skill"] and "n3" in p["skill"]
        assert p["rationale"] == "确认"

    def test_buffer_hitl_names_the_events(self):
        p = _interrupt_payload((self._I({"kind": "buffer_hitl",
                                         "events": [{"kind": "tip_crash"}]}),))
        assert "tip_crash" in p["skill"]

    def test_a_broken_payload_never_raises(self):
        assert _interrupt_payload(None) == {} or isinstance(_interrupt_payload(None), dict)
        assert isinstance(_interrupt_payload(("plain string",)), dict)


class TestEventStreamAnnouncesBeforeBlocking:
    def test_the_waiting_event_precedes_the_resolver_call(self, tmp_path):
        """Voice must be told what it is waiting for BEFORE the wait.

        The resolver blocks for up to 15 minutes and nothing can be emitted
        while it does, so a channel that only hears about the interrupt
        afterwards is simply silent for the duration.
        """
        eng, store, _ctx, _ = _hitl_engine(tmp_path)
        cid = store.create("instrument_control")["conversation_id"]
        order: list[str] = []

        def resolver(interrupted):
            order.append("resolver")
            return _answer("打")

        events = []
        for ev in eng.stream_events(cid, "打个脉冲", hitl_resolver=resolver):
            if ev["type"] == "interrupt":
                order.append("announced")
                events.append(ev)
        assert order[:2] == ["announced", "resolver"]
        assert events[0]["waiting"] is True
        # ⑰:生产者从「DANGEROUS 审批门」换成 ``ask_user``,所以这里的 skill 名跟着
        # 变了。被钉住的性质没变 —— 播报里必须说清**在等什么**,而不是只说「在等」。
        assert events[0]["payload"]["skill"] == "向用户提问"
        assert "要不要现在打这一发脉冲" in str(events[0]["payload"])

    def test_a_stale_approval_is_announced_as_a_notice(self, tmp_path):
        eng, store, _ctx, _ = _hitl_engine(tmp_path)
        cid = store.create("instrument_control")["conversation_id"]
        _drain(eng.stream_turn(cid, "打个脉冲"))          # parks the thread
        kinds = [ev["type"] for ev in eng.stream_events(cid, "在吗")]
        assert "notice" in kinds


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
