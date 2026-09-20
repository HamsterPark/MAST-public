"""``ConversationEngineV2`` —— 私聊回合跑在 ``agentruntime.AgentLoop`` 上。

退出 LangGraph 的**第三个**切换面（Step 2）。公开面与
:class:`~mast.chat.engine.ConversationEngine` **同形**：``voice/session.py`` 与
``api/routes/chat_stream.py`` 都是 duck-typed 消费，两边一行都不用改。

换掉的只有两样
--------------
* ``graph.stream(...)`` 循环 → ``AgentLoop.run(...)`` 的事件流；
* ``graph.get_state(cfg)`` 读历史 → ``MessageStore`` 读一段行。

**没换的**（刻意）：回合并发闸、abort 的三处补丁（缺陷⑬：同步 generator 被
``iterate_in_threadpool`` 消费时每次 ``next()`` 可能落在不同 worker 线程上）、旁白锚点、
``_finish`` 收尾、turn_context 重设。那些与运行时无关，是消费侧的事实。

★ 硬件安全：没有 SafetyGate 的 IC 私聊**不可能**
------------------------------------------------
``instrument_control`` 的中间件栈里有 SafetyGate（794 行状态前置条件 + 包络裁决）、
BufferHITL、AlertDelivery、ModeBelief —— 那张表由 ``instrument_control/graph.py`` 在
建图时按运行时注入的 ``get_state`` / ``get_mode`` / ``registry`` / ``buf`` 组装，而
``record_build`` 只记名字不记实例，取不回来。

在这里照抄那张表 = 制造第二个真源，而它的漂移方式是**守卫看起来在、实际没挂**。
所以这个引擎**拒绝**为 instrument_control 建循环，而不是「先不挂、以后补」：

    「不挂安全件就跑仪器 agent」必须是不可能，不是不推荐。

非 IC 的五个 agent（文献 / 数据 / 实验设计 / 写作 / 评审）不驱动仪器，它们的私聊在这条
路上是完整的。IC 私聊等工作树落定后走专门装配 —— 那时 SafetyGate 的移植要先过安全线
测试组才准上真机。
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)

#: 绝不允许在没有硬件安全中间件的情况下起私聊的 agent。
#: 这是一份**名单**而不是一个 `if agent_id == "instrument_control"`：将来多一个
#: 驱动仪器的 agent 时，加名字的人会看见旁边这段理由。
HARDWARE_AGENTS: frozenset[str] = frozenset({"instrument_control"})


class UnsafeAgentRefused(RuntimeError):
    """为一个驱动仪器的 agent 起 v2 私聊 —— 拒绝。"""


class ConversationEngineV2:
    """私聊运行时（v2）。

    ``loop_factory(agent_id) -> AgentLoop`` 是注入的：装配一个 agent 的方式
    （中间件栈、预算、给不给交棒工具）随入口不同而不同，而回合驱动逻辑对此不该
    有意见。默认用 :func:`~mast.agentruntime.assembly.build_agent_loop`。
    """

    def __init__(self, *, store, message_store,
                 loop_factory: Callable[[str], Any] | None = None,
                 ic_loop_factory: Callable[[], Any] | None = None,
                 record_sink: Callable[[str, str, str], None] | None = None):
        self._store = store
        self._messages = message_store
        self._loop_factory = loop_factory or self._default_loop_factory
        # IC 的装配需要运行时注入的 ``get_state`` / ``get_mode`` / ``registry`` /
        # ``buf`` —— 那些只有 CoreRuntime 有。**不接 = IC 私聊被拒绝**，而不是
        # 「用一个少了安全件的循环凑合」。
        self._ic_factory = ic_loop_factory
        self._record_sink = record_sink
        self._loops: dict[str, Any] = {}
        self._loops_lock = threading.Lock()
        self._tl = threading.local()
        self._active_threads: set[str] = set()
        self._active_lock = threading.Lock()
        # 会话 → 本回合的 Stop 事件。**不是 thread-local 那一份的备份，是它的补丁**
        # （缺陷⑬：用户按了停止，长技能停不下来）。原样搬自 v1 引擎。
        self._abort_by_conv: dict[str, threading.Event] = {}
        self._abort_by_conv_lock = threading.Lock()

    # ── 装配 ───────────────────────────────────────────────────────────
    @staticmethod
    def _default_loop_factory(agent_id: str):
        from mast.agentruntime.assembly import build_agent_loop

        return build_agent_loop(agent_id, buf=None)

    def _loop_for(self, agent_id: str):
        if agent_id in HARDWARE_AGENTS and self._ic_factory is None:
            # 没有专门装配就拒绝。**不是「先不挂安全件跑起来」** —— 一个安全件没挂上
            # 但照跑的 IC，症状是它一切正常，直到某次动作本该被拒。
            raise UnsafeAgentRefused(
                f"{agent_id} 驱动仪器，它的私聊必须带完整的硬件安全中间件"
                "（SafetyGate / AlertDelivery / ModeBelief，有 buffer 时还有 "
                "BufferHITL）。这个引擎实例没有接 IC 专门装配（"
                "``ic_loop_factory``），所以拒绝。")
        with self._loops_lock:
            loop = self._loops.get(agent_id)
            if loop is None:
                loop = (self._ic_factory() if agent_id in HARDWARE_AGENTS
                        else self._loop_factory(agent_id))
                self._loops[agent_id] = loop
            return loop

    def invalidate(self, agent_id: str | None = None) -> None:
        """丢掉缓存的循环（改了技能表 / 提示词 / 模型之后）。"""
        with self._loops_lock:
            if agent_id is None:
                self._loops.clear()
            else:
                self._loops.pop(agent_id, None)

    # ── 回合身份与中止（与 v1 同形，消费方 duck-typed） ────────────────
    def active_abort_event(self):
        return getattr(self._tl, "abort", None)

    def active_run_id(self) -> str:
        return getattr(self._tl, "run_id", "") or ""

    def active_conversation_id(self) -> str:
        return getattr(self._tl, "conversation_id", "") or ""

    def is_active(self, thread_id: str) -> bool:
        with self._active_lock:
            return thread_id in self._active_threads

    def _register_abort(self, conversation_id: str, abort) -> None:
        if abort is None:
            return
        with self._abort_by_conv_lock:
            self._abort_by_conv[conversation_id] = abort

    def _unregister_abort(self, conversation_id: str) -> None:
        with self._abort_by_conv_lock:
            self._abort_by_conv.pop(conversation_id, None)

    def abort_conversation(self, conversation_id: str) -> bool:
        """从**别的线程**停掉一个正在跑的会话（Stop 端点走这条）。"""
        with self._abort_by_conv_lock:
            ev = self._abort_by_conv.get(conversation_id)
        if ev is None:
            return False
        ev.set()
        return True

    #: 上一轮没写完时补的那句话。**一处定义** —— 显示侧与落库侧读的是同一个字面量，
    #: 否则两边会慢慢分叉成两句意思相近但不一样的话，而没人会去对它们。
    INTERRUPTED_TURN_NOTE = (
        "（上一轮在产出回答前中断了 —— 多半是服务重启。它没有被恢复："
        "新引擎不保留暂停中的运行。请把上一个请求再说一遍。）")

    def _turn_was_interrupted(self, conversation_id: str, msgs: list) -> bool:
        """上一轮是不是没写完就没了。

        判据**不需要新状态**：历史的最后一条是用户消息 ⇒ 那一轮没产出任何回答。
        正常回合的末尾一定是 assistant（哪怕「本回合提前结束」那条 notice 也是
        assistant 声部）。

        ⚠️ 还要排除「**此刻正在跑**」：一个在飞的回合的历史末尾同样是用户消息，
        把它报成「中断了」会在每次刷新时吓人一跳。

        ⚠️ ``is_active`` 收的是 **thread_id，不是 conversation_id**（两个引擎一致）。
        第一版直接把 conversation_id 传了进去 —— 永远返回 False，于是**在飞的回合
        每次刷新都被报成「中断了」**，而单元测试当场抓到。两个 id 在同一个函数里
        并存时，「哪个是哪个」不能靠名字长得像来判断。
        """
        from langchain_core.messages import HumanMessage

        if not msgs or not isinstance(msgs[-1], HumanMessage):
            return False
        conv = self._store.get(conversation_id) or {}
        thread_id = conv.get("thread_id") or conversation_id
        return not self.is_active(thread_id)

    # ── 读历史 ─────────────────────────────────────────────────────────
    def get_messages(self, conversation_id: str) -> list[dict]:
        """渲染整段历史。**读的是 ``chat_messages``，不是 checkpoint。**

        上一轮没写完的话，这里**临时**渲染一句说明 —— 不落库。

        为什么显示侧要单独做一次：落库那一次发生在**下一轮开始时**，而用户困惑的
        那一刻是他**打开会话**的时候。只做落库那一半，他会盯着一个没有回答的问题，
        直到自己再打一句话才知道发生了什么。

        为什么**不在这里落库**：这是读路径。「查一下不是免费的」是本仓记过的形状 ——
        一次刷新写一行，翻十次历史就多十行。落库交给下一轮（那时它确实成为历史的
        一部分）。
        """
        from mast.agentruntime.persist import rows_to_messages
        from mast.chat.render import render_history

        try:
            rows = self._messages.load(conversation_id)
            msgs = rows_to_messages(rows)
            if self._turn_was_interrupted(conversation_id, msgs):
                msgs = [*msgs, _system_note(self.INTERRUPTED_TURN_NOTE)]
            return render_history(msgs)
        except Exception as exc:  # noqa: BLE001 — 打不开的历史不该让页面白屏
            logger.warning("get_messages failed for %s: %s", conversation_id, exc)
            return []

    # ── 驱动一个回合 ───────────────────────────────────────────────────
    def stream_turn(self, conversation_id: str, user_text: str, *,
                    abort=None, hitl_resolver: Callable[[object], object] | None = None,
                    ) -> Generator[list[dict], None, None]:
        """每产生一条消息就 yield 一次**当前全量渲染**（与 v1 的契约相同）。

        外层做的唯一一件事与 v1 一样：把回合标识在**每次恢复执行前**重设一遍 ——
        同步 generator 被 ``iterate_in_threadpool`` 消费时每次 ``next()`` 可能落在
        不同 worker 线程上，只在开头设一次会让并发的两条流互相读到对方的
        conversation_id。
        """
        from mast.core.turn_context import reassert_turn_each_resume

        holder: dict = {}
        return reassert_turn_each_resume(
            self._stream_turn_impl(conversation_id, user_text, abort=abort,
                                   hitl_resolver=hitl_resolver,
                                   turn_holder=holder),
            holder)

    def stream_events(self, conversation_id: str, user_text: str, *,
                      abort=None,
                      hitl_resolver: Callable[[object], object] | None = None):
        """跑一个回合，yield **结构化事件**（不是渲染快照）。

        ★ 这个方法一度不存在，而**语音只调它**（2026-08-27 补）
        -------------------------------------------------------
        ``voice/session.py`` 调 ``stream_events`` 四处、一次 ``stream_turn`` 都不调。
        v2 上没有这个方法 ⇒ 翻开 ``engine_v2_private_chat``（**boot 级**开关，
        语音与私聊共用一个引擎）会让语音整条链断掉。

        而那条「公开面与 v1 逐个签名比对」的测试一路全绿 —— 因为它用的是一张**手写
        的 7 个方法名的清单**，而 ``stream_events`` 不在上面。判据已改成派生。

        事件词汇表与 v1 **逐字相同**（消费方按 ``type`` 分支）::

            {"type":"token",      "text": <文本增量>}
            {"type":"tool_start", "name": …, "args_preview": str}
            {"type":"tool_end",   "name": …, "result_preview": str}
            {"type":"interrupt",  "payload": {...}}
            {"type":"final",      "text": …, "aborted"?: bool}
            {"type":"error",      "message": str}
        """
        from mast.core.turn_context import reassert_turn_each_resume

        holder: dict = {}
        return reassert_turn_each_resume(
            self._stream_events_impl(conversation_id, user_text, abort=abort,
                                     hitl_resolver=hitl_resolver,
                                     turn_holder=holder),
            holder)

    def _stream_events_impl(self, conversation_id: str, user_text: str, *,
                            abort=None, hitl_resolver=None, turn_holder=None):
        """把 :meth:`_stream_turn_impl` 的事件流翻译成 v1 的事件词汇表。

        ★ **不复制生命周期**：并发闸、abort 登记、三个 thread-local、落库、收尾
        全在 ``_stream_turn_impl`` 里，这里只做翻译。抄第二遍的话，两条路会在
        「新建回合带不带 run_id」这种地方慢慢分叉，而那种分叉的症状没人会想到去
        那里找。

        机理：sink 在 ``_stream_turn_impl`` 里是**同步调用**的，且发生在它 yield
        渲染快照**之前** —— 所以每次 ``next()`` 回来时，队列里已经是这一步该出的
        事件了。渲染快照本身这里丢掉（语音不需要）。
        """
        queued: list = []
        final_text = ""
        aborted = False
        try:
            gen = self._stream_turn_impl(
                conversation_id, user_text, abort=abort,
                hitl_resolver=hitl_resolver, turn_holder=turn_holder,
                event_sink=queued.append, stream_tokens=True)
            last_render: list[dict] = []
            for snapshot in gen:
                while queued:
                    frame = _event_frame(queued.pop(0))
                    if frame:
                        yield frame
                last_render = snapshot or last_render
            while queued:                      # 收尾时残留的那几个
                frame = _event_frame(queued.pop(0))
                if frame:
                    yield frame
            for row in reversed(last_render or []):
                if row.get("role") == "assistant" and row.get("content"):
                    final_text = str(row["content"])
                    break
            aborted = bool(abort is not None and abort.is_set())
        except Exception as exc:  # noqa: BLE001 — 一次回合失败不该毁掉会话
            logger.warning("v2 stream_events failed for %s: %s",
                           conversation_id, exc)
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            return
        yield {"type": "final", "text": final_text, "aborted": aborted}

    def _stream_turn_impl(self, conversation_id: str, user_text: str, *,
                          abort=None, hitl_resolver=None, turn_holder=None,
                          event_sink=None, stream_tokens: bool = False):
        from langchain_core.messages import HumanMessage

        from mast.agentruntime.context import RunContext
        from mast.agentruntime.persist import (
            message_to_row,
            rows_to_messages,
            source_key,
        )
        from mast.chat.render import render_history

        conv = self._store.get(conversation_id)
        if conv is None:
            return
        agent_id = conv.get("agent_id", "")
        thread_id = conv.get("thread_id", "") or conversation_id

        with self._active_lock:
            if thread_id in self._active_threads:
                # 同一会话不许两条流同时跑 —— 它们会交错写同一段历史。
                logger.info("conversation %s already streaming; refusing a second",
                            conversation_id)
                return
            self._active_threads.add(thread_id)

        abort = abort if abort is not None else threading.Event()
        run_id = f"chat-{thread_id}-{uuid.uuid4().hex[:8]}"
        self._register_abort(conversation_id, abort)
        self._tl.abort = abort
        self._tl.run_id = run_id
        self._tl.conversation_id = conversation_id
        if turn_holder is not None:
            turn_holder["conversation_id"] = conversation_id
            turn_holder["run_id"] = run_id

        final_render: list[dict] = []
        try:
            loop = self._loop_for(agent_id)

            # 历史 + 这一轮的问题。**先把问题落库**：回合中途崩掉时，用户问了
            # 什么必须还在 —— 而不是「他记得自己问过，系统里没有」。
            history = rows_to_messages(self._messages.load_working_set(conversation_id))

            # ★ 上一轮**没写完**就补一句（2026-08-27 实测补上）。
            #
            # 计划里第三条★刻意变更是「重启不保暂停中的 run」，并写着「明说不恢复
            # **+ 注入 notice**」。前半做到了（v2 没有 checkpoint，run 就是没了），
            # **后半从没实现**。实测：进程在回合中途消失之后，用户回来看到的是
            # 自己那句话孤零零挂着 —— 没有回答，也没有任何解释；再发一条消息一切
            # 正常。**那一轮被静默吞掉了。**
            #
            # 对比 v1：checkpoint 存着暂停的 thread，下一轮重新抛 interrupt，
            # 问题会再问一遍（``api/hitl_bridge`` 的模块 docstring 写着这件事）。
            # 所以这不只是「少一句提示」，是**一个 v1 有、v2 没有的保障**，而计划
            # 承认了这个降级并要求把它说出来。
            #
            # 判据不需要新状态：**历史的最后一条是用户消息** ⇒ 上一轮没产出任何
            # 回答。正常回合的末尾一定是 assistant（哪怕是「本回合提前结束」那条
            # notice 也是 assistant 声部）。
            if history and isinstance(history[-1], HumanMessage):
                lost = _system_note(self.INTERRUPTED_TURN_NOTE)
                self._append(conversation_id, lost, kind="notice")
                history = [*history, lost]

            user_msg = HumanMessage(content=user_text)
            self._append(conversation_id, user_msg)

            ctx = RunContext(run_id=run_id, conversation_id=conversation_id,
                             thread_id=thread_id, agent_id=agent_id, abort=abort,
                             ask_human=_as_ask_human(hitl_resolver))
            if event_sink is not None:
                # 逐 token 走 ``ctx.emit``（中间件链是普通调用，yield 不出东西），
                # 与工具事件汇到**同一个 sink**，顺序因此与发生顺序一致。
                ctx.emit = event_sink
            if stream_tokens:
                loop = _with_token_streaming(loop)

            gen = loop.run([*history, user_msg], ctx)
            produced: list = [user_msg]
            while True:
                try:
                    ev = next(gen)
                except StopIteration as stop:
                    result = stop.value
                    break
                # ★ 事件旁路：``stream_events``（语音那条）传一个 sink 进来，
                #   工具事件因此能**边发生边出去**，而不必等回合结束。渲染快照这条
                #   路不需要它们，所以默认没有 sink。
                #
                #   显式参数而不是 thread-local：这个生成器被
                #   ``iterate_in_threadpool`` 消费时，每次 ``next()`` 可能落在不同
                #   worker 线程上（``reassert_turn_each_resume`` 存在正是为此）——
                #   挂在 thread-local 上的 sink 会在第二次恢复时**悄悄消失**。
                if event_sink is not None:
                    try:
                        event_sink(ev)
                    except Exception as exc:  # noqa: BLE001 — 收端坏了不该毁掉回合
                        logger.debug("event sink failed: %s", exc)
                # 每个事件之后重渲染一次 —— v1 的契约是「回合中每一步都能看到进展」。
                snapshot = render_history(produced)
                if snapshot:
                    final_render = snapshot
                    yield snapshot

            for msg in (result.new_messages or []):
                self._append(conversation_id, msg)
                produced.append(msg)

            final_render = render_history(produced) or final_render
            if final_render:
                yield final_render

            if result.outcome not in ("final", "handoff"):
                # ★ 非正常结局要**出现在对话里**。旧引擎下这类回合看起来就像
                #   「助手没说话」，用户无从知道是限流、停机还是报错。
                note = f"（本回合提前结束：{result.stop_reason or result.outcome}）"
                self._append(conversation_id,
                             _system_note(note), kind="notice")
                produced.append(_system_note(note))
                final_render = render_history(produced)
                yield final_render
        except Exception as exc:  # noqa: BLE001 — 一次回合失败不该毁掉会话
            logger.warning("v2 chat turn failed for %s: %s", conversation_id, exc)
            note = _system_note(f"（本回合出错：{type(exc).__name__}: {exc}）")
            self._append(conversation_id, note, kind="notice")
            final_render = render_history([note])
            yield final_render
        finally:
            with self._active_lock:
                self._active_threads.discard(thread_id)
            self._unregister_abort(conversation_id)
            self._tl.abort = None
            self._tl.run_id = ""
            self._tl.conversation_id = ""
            self._finish(conversation_id, conv, user_text, final_render)

    # ── 落库 ───────────────────────────────────────────────────────────
    def _append(self, conversation_id: str, msg, *, kind: str = "message") -> None:
        from mast.agentruntime.persist import message_to_row, source_key

        try:
            row = message_to_row(msg)
            row.setdefault("meta", {})
            row["meta"]["src_key"] = source_key(msg)
            if kind != "message":
                row["meta"]["notice"] = True
            self._messages.append(conversation_id, **row)
        except Exception as exc:  # noqa: BLE001 — 落库失败不该中断回合
            logger.debug("append to chat_messages failed: %s", exc)

    def _finish(self, conversation_id, conv, user_text, final) -> None:
        try:
            preview = ""
            for m in reversed(final or []):
                if m.get("role") == "assistant":
                    preview = m.get("content", "")
                    break
            title = None
            if (conv.get("title") in ("新对话", "新任务", "")
                    and (conv.get("last_message_preview") or "") == ""):
                title = (user_text or "").strip().replace("\n", " ")[:40] or None
            self._store.touch(conversation_id, preview=preview, title=title)
        except Exception as exc:  # noqa: BLE001
            logger.debug("touch after v2 turn failed: %s", exc)

        try:
            from mast.chat.store import publish_private_turn_finished

            publish_private_turn_finished(
                conversation_id,
                conv.get("agent_id", "") if isinstance(conv, dict) else "")
        except Exception as exc:  # noqa: BLE001 — 通知绝不许毁掉一轮对话
            logger.debug("private turn notify failed: %s", exc)

        if self._record_sink is not None:
            try:
                agent_id = conv.get("agent_id", "") if isinstance(conv, dict) else ""
                preview = ""
                for m in reversed(final or []):
                    if m.get("role") == "assistant":
                        preview = m.get("content", "")
                        break
                self._record_sink(agent_id, user_text or "", preview or "")
            except Exception as exc:  # noqa: BLE001
                logger.debug("v2 record_sink failed: %s", exc)


def _system_note(text: str):
    """给用户看的一句话（限流停住、出错、拒绝）。

    ⚠️ 必须是 ``AIMessage`` 而不是 ``SystemMessage``。``render_history`` 只渲染
    Human / AI / Tool 三类，``SystemMessage`` **被整条丢掉** —— 第一版用了它，
    于是「用来让异常结局可见的那条说明，自己是不可见的」，三条测试当场变红。
    这条 notice 是系统对用户说话，assistant 声部本来就是对的。

    （对照：``compaction`` 的摘要用 ``SystemMessage`` 是**刻意**的——那段不是说给
    用户听的，是喂给模型的上下文。两处不同的选择，两个不同的读者。）
    """
    from langchain_core.messages import AIMessage

    return AIMessage(content=text)


def _with_token_streaming(loop):
    """要一个开着 ``stream_tokens`` 的同款循环。

    不原地改 ``loop.stream_tokens``：同一个 ``AgentLoop`` 对象可能被并发用于不同的
    run（编排器 fan-out 就是这么用的，类 docstring 明写「无状态」）。原地翻开关会
    让另一条 run 跟着流式 —— 而那条 run 没有人收 token。
    """
    import copy

    try:
        clone = copy.copy(loop)          # 浅拷贝：工具表/中间件栈共享，标志位独立
        clone.stream_tokens = True
        return clone
    except Exception as exc:  # noqa: BLE001 — 拷不动就退回非流式，不是致命的
        logger.debug("cannot clone loop for streaming: %s", exc)
        return loop


#: 会**阻塞等用户**的工具。它们的 ``tool_start`` 翻译成 ``interrupt(waiting=True)``，
#: 好让语音在等待期间说话（见 :func:`_event_frame`）。
#:
#: 写成集合而不是判 `name == "ask_user"`：判据是「**这个工具会不会挂起等人**」，
#: 而不是「它叫不叫 ask_user」—— 将来多一个会等人的工具，漏掉它的症状是
#: 「语音又静默了」，而那种漏法查起来极贵。
_ASKING_TOOLS = frozenset({"ask_user"})


def _event_frame(ev) -> dict | None:
    """一个 v2 ``RunEvent`` → v1 的事件 dict。认不出的返回 ``None``（丢掉）。

    翻译表刻意**窄**：语音只按 ``type`` 分支处理 token / tool_start / tool_end /
    interrupt / final / error 六种，多发的帧它会忽略，而每一种「多发但没人认」的帧
    都是一次白付的队列往返。
    """
    kind = getattr(ev, "kind", "")
    if kind == "token":
        text = getattr(ev, "text", "")
        return {"type": "token", "text": text} if text else None
    if kind == "tool_start":
        name = getattr(ev, "name", "tool")
        if name in _ASKING_TOOLS:
            # ★ **语音在等用户的那段时间里必须说话**（2026-08-27 实测补）。
            #
            # v1 里这句由 ``interrupt`` 事件驱动（``voice/session._announce_interrupt``：
            # 「需要人工确认…我等你的结果」）。v2 的 HITL 是**原地阻塞** —— 没有
            # interrupt 事件，实测语音在整个等待期间**完全静默**，最长 900 秒。
            # 而这条通道上用户看不见任何东西，**沉默就是「它死了」** —— 正是现场
            # #33/#34 那个症状，落在最难自查的一条链上。
            #
            # 翻译成 ``interrupt`` 而不是往语音的播报词表里加一条：那张表是**两个
            # 引擎共用**的，加进去会让 v1 说两遍（它已经有 announce），而 v1 现在
            # 是生产。这里翻译 ⇒ 只影响 v2，且用上语音已有的那套措辞。
            #
            # ``waiting=True`` 是关键：语音据此说「我等你的结果」而**不**把这一轮
            # 判为结束。时刻也对得上 —— ``tool_start`` 现在在工具执行**之前**发出
            # （见 ``agentruntime/loop._run_one_tool``），也就是阻塞开始之前。
            args = _short(getattr(ev, "preview", ""))
            return {"type": "interrupt", "waiting": True,
                    "payload": {"kind": "ask_user", "skill": "向用户提问",
                                "rationale": args}}
        return {"type": "tool_start", "name": name,
                "args_preview": _short(getattr(ev, "args_preview", "")
                                       or getattr(ev, "preview", ""))}
    if kind == "tool_end":
        return {"type": "tool_end", "name": getattr(ev, "name", "tool"),
                "result_preview": _short(getattr(ev, "preview", ""))}
    if kind == "interrupt":
        return {"type": "interrupt",
                "payload": dict(getattr(ev, "payload", None) or {})}
    return None


def _short(v, n: int = 220) -> str:
    s = v if isinstance(v, str) else str(v)
    return s if len(s) <= n else s[:n] + "…"


def _as_ask_human(hitl_resolver):
    """把 v1 的 ``hitl_resolver`` 适配成 :meth:`RunContext.ask` 要的形状。

    v1 的 resolver 收一个「pending」对象、返回用户的决定；新签名收 payload dict
    返回答案 dict。两边的**语义**一样（阻塞等人），所以适配是纯形状转换。
    """
    if hitl_resolver is None:
        return None

    def _ask(payload: dict):
        try:
            return hitl_resolver(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hitl_resolver failed: %s", exc)
            return None
    return _ask


__all__ = ["ConversationEngineV2", "UnsafeAgentRefused", "HARDWARE_AGENTS"]

