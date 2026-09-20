"""ConversationEngine — drive a real per-agent private chat on the existing graphs.

The engine builds (once, cached per agent_id) a STANDALONE compiled graph for an
agent via an injected ``graph_factory`` — the SAME ``create_agent`` graph the
orchestrator routes to, just compiled with the shared checkpointer so a private
conversation has its own durable ``thread_id``. Conversation identity lives
entirely in ``thread_id``; the checkpointer is the source of truth for history,
so the engine renders progress by reading ``graph.get_state`` between super-steps.

Decoupled from the GUI: ``graph_factory(agent_id) -> CompiledGraph`` is supplied
by the app (closing over the shared providers + checkpointer); HITL is surfaced
via an injected ``hitl_resolver`` callback so the same approval modal serves
私聊 and 群聊. Per-conversation abort is isolated **per conversation** so aborting
one chat never kills a concurrent run.

> 这句话原本写的是「靠 thread-local 隔离」,而那正是缺陷⑬ 的成因:同步 generator
> 每次恢复执行可能落在**另一个** worker 线程上,那里读不到回合开头设的 thread-local
> ⇒ 用户的「停止」到不了正在跑的技能。隔离的**单位是会话**,thread-local 只是它
> 在同线程时的快路径。见 :meth:`ConversationEngine.active_abort_event`。
"""

from __future__ import annotations

import html
import json
import logging
import threading
from typing import Callable, Generator

from mast.chat.render import extract_ai_parts, render_history

logger = logging.getLogger(__name__)

_THINKING_PLACEHOLDER = '<span class="mast-thinking">思考中…</span>'


def _esc(text) -> str:
    return html.escape(str(text if text is not None else ""), quote=False)


class ConversationEngine:
    """Builds/caches standalone agent graphs and drives conversation turns."""

    def __init__(
        self,
        graph_factory: Callable[[str], object],
        checkpointer,
        store,
        *,
        recursion_limit: int | None = None,
        call_limits_provider: Callable[[], dict] | None = None,
        record_sink: Callable[[str, str, str], None] | None = None,
        message_store=None,
    ):
        self._graph_factory = graph_factory
        self._checkpointer = checkpointer
        self._store = store
        # 结构化正文的第二个家（``agentruntime.persist.MessageStore``）。
        #
        # 今天私聊的正文**只**活在 checkpointer 里，所以：别的窗口收不到这一轮
        # （转录推送读的是 ConversationStore 那张表，私聊在里面没有正文），而且
        # 换掉 checkpointer 历史就归零 —— 没有第二份副本。
        #
        # 这是退出 LangGraph 的 strangler 第 0 步：**只写不读**。读路径仍然是
        # checkpointer，双写只是让新家先攒下数周的历史，等对账确认无误之后再切
        # 读端。None = 未接线（老的构造方式、测试），双写整个跳过。
        self._message_store = message_store
        # None (production) = derive per agent from the built graph's shape; an
        # explicit int pins it (tests, and an operator override if one is ever
        # added). See _recursion_limit_for for why a literal is the wrong thing
        # to write down here.
        self._rl_override = recursion_limit
        self._call_limits_provider = call_limits_provider
        self._rl_cache: dict[str, int] = {}
        # Optional (agent_id, user_text, assistant_text) → None sink that mirrors
        # each finished turn into the experiment record (storage.log_conversation).
        # Was never wired in the TS rewrite, so chat never reached the experiment
        # record. Best-effort; failures never break a turn.
        self._record_sink = record_sink
        self._graphs: dict[str, object] = {}
        self._graphs_lock = threading.Lock()
        self._tl = threading.local()
        self._active_threads: set[str] = set()
        self._active_lock = threading.Lock()
        # 会话 → 本回合的 Stop 事件。**不是 thread-local 的那一份的备份,是它的补丁。**
        # 见 active_abort_event() 里那段(缺陷⑬:用户按了停止,长技能停不下来)。
        self._abort_by_conv: "dict[str, threading.Event]" = {}
        self._abort_by_conv_lock = threading.Lock()

    # ── graph cache ────────────────────────────────────────────────────
    def _graph_for(self, agent_id: str):
        with self._graphs_lock:
            g = self._graphs.get(agent_id)
            if g is None:
                g = self._graph_factory(agent_id)
                self._graphs[agent_id] = g
            return g

    def invalidate(self, agent_id: str | None = None) -> None:
        """Drop cached graph(s) so the next turn rebuilds (e.g. after a model swap)."""
        with self._graphs_lock:
            if agent_id is None:
                self._graphs.clear()
                self._rl_cache.clear()
            else:
                self._graphs.pop(agent_id, None)
                self._rl_cache.pop(agent_id, None)

    # ── per-turn super-step budget ─────────────────────────────────────
    def _recursion_limit_for(self, agent_id: str, graph) -> int:
        """This agent's ``recursion_limit``, DERIVED from its graph shape.

        It used to be the literal 50, written on 2026-06-16 when a chat graph was
        7 nodes and one tool call cost 5 super-steps — i.e. 50 bought ~9 tool
        calls. Six ``before_model``/``after_model`` middlewares were added since;
        each one is an extra node on EVERY model call, so the price went 5 → 11
        and the same 50 silently became **3 tool calls per turn**.

        Nobody edited a limit, and two things stopped working (2026-08-04,
        the rig):

          * a five-step operator request died with ``GraphRecursionError`` — it
            needs 65 super-steps;
          * ``StallGuardMiddleware`` could no longer complete its
            nudge→nudge→stop ladder (5 failing rounds = 65 super-steps), so a
            genuine spin ended in the raw recursion error the guard exists to
            replace. Wired, correct, unreachable.

        Deriving it means the next middleware re-prices the budget instead of
        confiscating a third of it. Cached per agent and cleared by
        :meth:`invalidate`, so a rebuilt graph is re-measured.
        """
        if self._rl_override is not None:
            return int(self._rl_override)
        cached = self._rl_cache.get(agent_id)
        if cached is not None:
            return cached
        from mast.agents._shared.call_limits import (
            DEFAULT_MODEL_CALLS_PER_RUN,
            derive_recursion_limit,
        )
        calls = DEFAULT_MODEL_CALLS_PER_RUN
        if self._call_limits_provider is not None:
            try:
                calls = int((self._call_limits_provider() or {}).get(
                    "max_model_calls_per_run") or DEFAULT_MODEL_CALLS_PER_RUN)
            except Exception as exc:  # noqa: BLE001 — a budget estimate never breaks a turn
                logger.debug("call-limits provider failed: %s", exc)
        rl = derive_recursion_limit(graph, model_calls_per_run=calls)
        self._rl_cache[agent_id] = rl
        logger.info("chat[%s]: recursion_limit=%d (derived; %d model calls/run)",
                    agent_id, rl, calls)
        return rl

    def _config_for(self, agent_id: str, graph, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": self._recursion_limit_for(agent_id, graph)}

    # ── per-conversation abort isolation ───────────────────────────────
    def active_abort_event(self):
        """本回合的 Stop 事件(读不到返回 None)。

        app 的 per-agent ``context_provider`` 调它,好让 composite 的 ``check_abort()``
        对上**这个会话**的停止按钮,而不是一个会连带杀掉并发回合的全局 abort。

        ## 为什么不能只读 thread-local(缺陷⑬,2026-08-06 实机,两天撞三次)

        回合开头在**某一个** worker 线程上设了 ``self._tl.abort``。但这条流是同步
        generator,交给 ``StreamingResponse`` 后由 ``iterate_in_threadpool`` 消费 ——
        **每次 ``next()`` 单独走一趟 ``anyio.to_thread.run_sync``,不保证落在同一个
        线程上**(``turn_context.reassert_turn_each_resume`` 的 docstring 里有实测
        记录:两条并发流互相读到对方的值)。

        于是:恢复执行落到线程 B 时,``self._tl.abort`` 在 B 上**根本没有** ⇒ 这一步
        里建的 ``ExecutionContext`` 的 abort 并集里只剩 ``_orch_abort``(E_STOP)⇒
        **用户按下的「停止」到不了正在跑的技能**。等待循环确实每个 poll 都在查
        ``check_abort()`` —— 它查的那个事件集里没有用户那一个。

        这也解释了为什么是「三次」而不是「每次」:落在哪个线程是随机的。

        当初为 ``conversation_id`` / ``run_id`` 修过同一个坑(每次恢复前重设),
        **唯独漏了 abort**。这里补的是同一条思路的另一半:按**会话**取,而会话 id
        正是那次修复保证了每次恢复前都重设的东西。

        顺序上先读 thread-local:同线程时它一定是对的,且不需要拿锁。
        """
        ev = getattr(self._tl, "abort", None)
        if ev is not None:
            return ev
        from mast.core.turn_context import current_turn
        cid = (current_turn() or {}).get("conversation_id")
        if not cid:
            return None
        with self._abort_by_conv_lock:
            return self._abort_by_conv.get(str(cid))

    def _register_abort(self, conversation_id: str, abort) -> None:
        """把本回合的 Stop 事件登记到会话名下(线程无关的那一份)。"""
        if abort is None or not conversation_id:
            return
        with self._abort_by_conv_lock:
            self._abort_by_conv[str(conversation_id)] = abort

    def _unregister_abort(self, conversation_id: str) -> None:
        """回合结束就摘掉 —— **必须在 finally 里调**。

        留着的话,下一个回合在事件被 ``_abort_event()`` clear 之前的那一小段里,
        会读到上一回合的事件;更糟的是一个已经 set 过的陈旧事件会让新回合的第一个
        技能**立刻自我中止**,而症状是「它什么都没干就说被停了」。
        """
        if not conversation_id:
            return
        with self._abort_by_conv_lock:
            self._abort_by_conv.pop(str(conversation_id), None)

    def active_run_id(self) -> str:
        """This turn's OWN run id — never the orchestrator's.

        Composite step-progress sidecars are keyed ``(composite_name, run_id)``.
        The private-chat ExecutionContext used to borrow ``app._orch_run_id``,
        which is written when a group run starts and cleared NOWHERE in the
        tree: a chat composite therefore ran under a stranger's run id, and two
        concurrent AutoApproach runs (one per entry point) shared one sidecar.
        The resume guard only rejects TERMINAL and STALE (30 min) progress, and
        a run in flight is neither — so the second one resumed the first one's
        progress and skipped steps it had never executed. That is exactly the
        2026-07-10 「假进针」 accident, reopened along the cross-chain axis.

        One id per TURN (not per conversation): a composite's progress is only
        ever legitimately resumed inside the turn that created it.
        """
        return str(getattr(self._tl, "run_id", "") or "")

    def active_conversation_id(self) -> str:
        """本回合所属的对话 id。

        这个值早就写在 thread-local 里了（``stream_turn`` / ``stream_events`` 开头），
        但一直**没有公共访问器、零消费者**。文档保存需要它来回答「这份报告是哪次
        对话产出的」，所以补上 —— 与 :meth:`active_run_id` 同一形状。

        真正的落点是 ``mast.core.turn_context``（中立、谁都能读）；这里保留一个
        访问器是为了对称，也方便直接持有 engine 的调用方。
        """
        return str(getattr(self._tl, "conversation_id", "") or "")

    def _set_turn_context(self, conversation_id: str, run_id: str) -> None:
        """把本回合的来历放进进程级 thread-local，供任意工具读取。

        provenance 是 best-effort：写失败绝不能影响这一轮对话，所以整段包在
        try 里（见 ``core/turn_context`` 的三条纪律）。

        **只设一次是不够的** —— 见 :func:`reassert_turn_each_resume`。
        """
        try:
            from mast.core.turn_context import set_turn
            set_turn(conversation_id=conversation_id, run_id=run_id)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _clear_turn_context() -> None:
        try:
            from mast.core.turn_context import clear_turn
            clear_turn()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _set_narration_anchor(conversation_id: str, n: int) -> None:
        """告诉旁白侧「此刻这个会话已经渲染出 n 条消息」。

        旁白（``mast/chat/narration.py``）落在转录表里，刷新之后要重新插回消息
        之间 —— 而 :func:`render_history` 的输出**没有时间戳**（``chat/render.py``
        只给 ``{role, content}``），所以按 wall-clock 对齐是做不到的。``anchor``
        就是替代品：发出那一刻的消息条数。

        放在这里是**零额外开销**：``cur`` 每个 super-step 本来就在算。
        best-effort，与本类其他 provenance 写入同一条纪律。
        """
        try:
            from mast.chat import narration
            narration.set_anchor(conversation_id, n)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _clear_narration_anchor(conversation_id: str) -> None:
        """回合结束时摘掉 —— 留着的话，下一轮开头发的旁白会带上上一轮的条数。"""
        try:
            from mast.chat import narration
            narration.clear_anchor(conversation_id)
        except Exception:  # noqa: BLE001
            pass

    def _new_run_id(self, thread_id: str) -> str:
        import uuid

        return f"chat-{str(thread_id)[:12]}-{uuid.uuid4().hex[:8]}"

    def is_active(self, thread_id: str) -> bool:
        with self._active_lock:
            return thread_id in self._active_threads

    # ── read history (for conversation switch) ─────────────────────────
    def _state_messages(self, graph, cfg) -> list:
        try:
            snap = graph.get_state(cfg)
            return list((snap.values or {}).get("messages", []) or [])
        except Exception as exc:  # noqa: BLE001 — no checkpoint yet / read glitch
            logger.debug("get_state failed: %s", exc)
            return []

    def _stale_interrupt_resume(self, graph, cfg):
        """A ``Command`` that fail-closed rejects an interrupt left over from an
        EARLIER turn — or ``None`` when the thread is clean.

        A private chat that hit an approval nobody answered (no resolver wired,
        the operator pressed Stop, the server restarted) left the checkpoint
        parked ON the gate. Every later message then re-entered the same node,
        raised a NEW interrupt and got the same non-answer: the conversation
        was wedged, and from the operator's chair it looked like the assistant
        had started repeating one strange sentence at them (spike 2026-08-01).

        A late resume still drives such a thread to completion, so the fix is to
        clear the stale pause before the new message rather than to detect and
        refuse. It rejects — it never re-publishes for approval: an approval
        request that has outlived its turn has lost its context, and a hardware
        action must not run on a verdict given to a question nobody remembers.

        Why this is safe to call unconditionally: it runs AFTER the concurrency
        guard, so no other stream can be mid-turn on this thread, and anything
        found paused here therefore belongs to a turn that is already over.
        """
        try:
            snap = graph.get_state(cfg)
        except Exception as exc:  # noqa: BLE001 — no checkpoint yet / read glitch
            logger.debug("stale-interrupt probe failed: %s", exc)
            return None
        stale = []
        for task in (getattr(snap, "tasks", None) or ()):
            stale.extend(getattr(task, "interrupts", None) or ())
        if not stale:
            # NB: a non-empty ``next`` without interrupts is an aborted stream's
            # ordinary residue — the graph resumes from it on the next input.
            # Only a pending interrupt wedges the thread.
            return None
        from langgraph.types import Command
        by_id = {}
        for intr in stale:
            iid = getattr(intr, "id", None)
            if not iid:
                by_id = {}
                break
            by_id[iid] = _stale_reject_value(getattr(intr, "value", None))
        logger.warning("chat: auto-rejecting %d stale interrupt(s) on %s",
                       len(stale), cfg.get("configurable", {}).get("thread_id"))
        if by_id:
            return Command(resume=by_id)
        # No ids (older LangGraph): the private graph is serial, so a broadcast
        # reject reaches the one pause that exists.
        return Command(resume=_stale_reject_value(getattr(stale[0], "value", None)))

    def get_messages(self, conversation_id: str) -> list[dict]:
        """Render a conversation's full history for display on switch. Best-effort."""
        conv = self._store.get(conversation_id)
        if conv is None:
            return []
        try:
            graph = self._graph_for(conv["agent_id"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_messages: graph build failed for %s: %s",
                           conv["agent_id"], exc)
            return []
        cfg = self._config_for(conv["agent_id"], graph, conv["thread_id"])
        return render_history(self._state_messages(graph, cfg))

    # ── drive a turn ───────────────────────────────────────────────────
    def stream_turn(
        self,
        conversation_id: str,
        user_text: str,
        *,
        abort=None,
        hitl_resolver: Callable[[object], object] | None = None,
    ) -> Generator[list[dict], None, None]:
        """见 :meth:`_stream_turn_impl`。这层只做一件事：把回合标识在**每次恢复执行
        前**重新设一遍 —— 同步 generator 被 ``iterate_in_threadpool`` 消费时每次
        ``next()`` 可能落在不同 worker 线程上，只在开头设一次会让并发的两条流互相
        读到对方的 conversation_id（详见 ``core/turn_context.reassert_turn_each_resume``）。"""
        from mast.core.turn_context import reassert_turn_each_resume
        holder: dict = {}
        return reassert_turn_each_resume(
            self._stream_turn_impl(conversation_id, user_text, abort=abort,
                                   hitl_resolver=hitl_resolver, turn_holder=holder),
            holder)

    def _stream_turn_impl(
        self,
        conversation_id: str,
        user_text: str,
        *,
        abort=None,
        hitl_resolver: Callable[[object], object] | None = None,
        turn_holder: dict | None = None,
    ) -> Generator[list[dict], None, None]:
        """Run one turn; yield full rendered Chatbot history snapshots as it streams.

        Yields ``[{role, content}]`` lists (the GUI replaces the Chatbot with each
        yield). First yield shows the user turn + a thinking placeholder so the UI
        never looks frozen.
        """
        conv = self._store.get(conversation_id)
        if conv is None:
            yield [{"role": "assistant", "content": "_(会话不存在)_"}]
            return
        agent_id, thread_id = conv["agent_id"], conv["thread_id"]
        try:
            graph = self._graph_for(agent_id)
        except Exception as exc:  # noqa: BLE001
            yield [{"role": "assistant",
                    "content": f"**无法构建 `{agent_id}` 引擎**\n\n```\n{exc}\n```"}]
            return

        cfg = self._config_for(agent_id, graph, thread_id)
        from langchain_core.messages import HumanMessage

        base = render_history(self._state_messages(graph, cfg))
        user_disp = {"role": "user", "content": _esc(user_text)}

        # Concurrency guard: a second POST for the SAME
        # conversation (double-click / two clients) would otherwise drive TWO
        # streams over ONE LangGraph checkpoint thread — interleaved writes
        # corrupt the checkpoint. Refuse the second run instead of racing.
        with self._active_lock:
            if thread_id in self._active_threads:
                yield base + [user_disp, {
                    "role": "assistant",
                    "content": "_该会话已有一个进行中的回合，请等待它完成或点停止后再发送。_"}]
                return
            self._active_threads.add(thread_id)

        # first yield: user turn echoed + placeholder (generator-yield-first rule)
        yield base + [user_disp, {"role": "assistant", "content": _THINKING_PLACEHOLDER}]

        self._tl.abort = abort
        # 同时按**会话**登记一份:generator 恢复执行可能落在别的 worker 线程上,
        # 那里读不到 thread-local 的这一份(缺陷⑬,见 active_abort_event)。
        self._register_abort(conversation_id, abort)
        self._tl.conversation_id = conversation_id
        # This turn's OWN run id — see active_run_id() for why borrowing the
        # orchestrator's was a sidecar collision waiting to happen.
        self._tl.run_id = self._new_run_id(thread_id)
        self._set_turn_context(conversation_id, self._tl.run_id)
        if turn_holder is not None:
            # 外层包装每次恢复执行前从这里重读并重设（见 reassert_turn_each_resume）。
            turn_holder["conversation_id"] = conversation_id
            turn_holder["run_id"] = self._tl.run_id
        # 旁白的落点：base + 刚发出去的那条用户消息。第一个 super-step 之前就得有
        # 值，否则这段时间里发出的旁白会挂着**上一轮**留下的条数。
        # 只在这条（打字聊天）路上设：旁白目前只画在第一个标签页的仪器 chat 里，
        # 语音那条流的客户端根本不渲染它。
        self._set_narration_anchor(conversation_id, len(base) + 1)

        # Notices must OUTLIVE the turn. Every intermediate yield here is a full
        # history snapshot and the client replaces its list with each one, so a
        # notice yielded before the loop ends is wiped by the final snapshot
        # rendered from checkpoint state — which is why "当前入口未接审批处理器"
        # was never actually readable: it flashed and was gone, leaving a bare
        # tool-call line as the only trace (2026-08-01).
        notices: list[str] = []
        # Inputs for this turn, in order: a stale-approval cleanup (when the
        # thread was left parked on a gate) runs BEFORE the new message, and each
        # HITL resume is pushed to the front as it is answered.
        # Items are (is_user_message, input) so the code below can tell whether
        # what the operator typed actually reached the graph.
        pending_inputs: list[tuple[bool, object]] = []
        rescue = self._stale_interrupt_resume(graph, cfg)
        if rescue is not None:
            notices.append("↺ _上一轮遗留的人工审批已过期，已按拒绝自动处理；下面继续这条新消息_")
            yield base + [user_disp, {"role": "assistant", "content": notices[-1]}]
            pending_inputs.append((False, rescue))
        pending_inputs.append((True, {"messages": [HumanMessage(content=user_text)]}))
        aborted = False
        try:
            while pending_inputs:  # one pass per input; a HITL resume adds one
                _is_user, stream_input = pending_inputs.pop(0)
                interrupted = None
                for _ns, chunk in graph.stream(
                    stream_input, config=cfg,
                    stream_mode="updates", subgraphs=True,
                ):
                    if abort is not None and abort.is_set():
                        aborted = True
                        break
                    if isinstance(chunk, dict) and "__interrupt__" in chunk:
                        interrupted = chunk["__interrupt__"]
                        break
                    cur = render_history(self._state_messages(graph, cfg))
                    # 每个 super-step 更新一次旁白落点。零额外开销 —— cur 本来就在算。
                    self._set_narration_anchor(conversation_id, len(cur))
                    if cur:
                        yield cur
                if aborted:
                    break
                if interrupted is not None:
                    if hitl_resolver is None:
                        # nothing can approve it here — leave paused, surface notice
                        notices.append("⚠️ _需要人工审批，但当前入口未接审批处理器_")
                        cur = render_history(self._state_messages(graph, cfg))
                        cur.append({"role": "assistant", "content": notices[-1]})
                        yield cur
                        break
                    # The resolver BLOCKS until the operator answers, and nothing
                    # can be yielded while it does. Say what is being waited for
                    # first, or the chat reads as hung for up to 15 minutes.
                    # (The connection itself stays alive on api/sse.with_heartbeat's
                    # independent pump; this is about the operator, not the socket.)
                    waiting = render_history(self._state_messages(graph, cfg))
                    waiting.append({
                        "role": "assistant",
                        "content": "⏸ _正在等待你的处理 —— 请打开上方「人工介入」查看并回应_",
                    })
                    yield waiting
                    try:
                        decision = hitl_resolver(interrupted)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("hitl_resolver failed: %s", exc)
                        decision = None
                    if decision is None:
                        # Abort has its own "_已中止_" line below; anything else
                        # (a timeout the asker declared should stop, a resolver
                        # error) needs to say the turn is paused AND that the
                        # next message clears it — otherwise the operator has no
                        # way to know the conversation is still usable.
                        if abort is None or not abort.is_set():
                            notices.append(
                                "⚠️ _审批未完成，本轮到此为止；下一条消息会自动拒绝这条过期审批后继续_")
                        break
                    from langgraph.types import Command
                    pending_inputs.insert(0, (False, Command(resume=decision)))
                    continue
            # The user's message is still queued only when the cleanup pass
            # above ended the turn before it ran. Say so — silently dropping
            # what someone typed is worse than the wedge this replaced.
            if any(is_user for is_user, _ in pending_inputs) and not aborted:
                notices.append("⚠️ _你这条消息还没送达（先处理过期审批时本轮就停了），请重新发送_")
        except Exception as exc:  # noqa: BLE001 — surface the real error in chat
            logger.error("stream_turn(%s) failed: %s", conversation_id, exc)
            final = render_history(self._state_messages(graph, cfg))
            final.append({"role": "assistant",
                          "content": f"**执行出错**\n\n```\n{type(exc).__name__}: {exc}\n```"})
            self._finish(conversation_id, conv, thread_id, user_text, final)
            yield final
            return
        finally:
            self._tl.abort = None
            self._unregister_abort(conversation_id)
            self._tl.run_id = ""
            self._clear_turn_context()
            self._clear_narration_anchor(conversation_id)
            # ALWAYS release the active-run slot — even on client disconnect
            # (GeneratorExit) — so the concurrency guard can't permanently wedge
            # a conversation as "in progress". _finish also
            # discards (idempotent).
            with self._active_lock:
                self._active_threads.discard(thread_id)

        final = render_history(self._state_messages(graph, cfg))
        if not final:
            final = base + [user_disp, {"role": "assistant", "content": "_(无回复)_"}]
        # Re-attach what happened OUTSIDE the graph's own history — see `notices`.
        for note in notices:
            final.append({"role": "assistant", "content": note})
        if aborted:
            final.append({"role": "assistant", "content": "_已中止_"})
        self._finish(conversation_id, conv, thread_id, user_text, final)
        yield final

    # ── drive a turn as STRUCTURED EVENTS (voice / rich clients) ────────
    def stream_events(
        self,
        conversation_id: str,
        user_text: str,
        *,
        abort=None,
        hitl_resolver: Callable[[object], object] | None = None,
    ):
        """见 :meth:`_stream_events_impl`；这层与 :meth:`stream_turn` 同理，
        每次恢复执行前重设回合标识。"""
        from mast.core.turn_context import reassert_turn_each_resume
        holder: dict = {}
        return reassert_turn_each_resume(
            self._stream_events_impl(conversation_id, user_text, abort=abort,
                                     hitl_resolver=hitl_resolver,
                                     turn_holder=holder),
            holder)

    def _stream_events_impl(
        self,
        conversation_id: str,
        user_text: str,
        *,
        abort=None,
        hitl_resolver: Callable[[object], object] | None = None,
        turn_holder: dict | None = None,
    ) -> Generator[dict, None, None]:
        """Run one turn, yielding STRUCTURED events (not rendered snapshots):

          {"type":"token",      "text": <assistant text delta>}
          {"type":"tool_start", "name": <tool>, "args_preview": str}
          {"type":"tool_end",   "name": <tool>, "result_preview": str}
          {"type":"interrupt",  "payload": <interrupt value dict>}
          {"type":"final",      "text": <clean assistant reply>, "aborted"?: bool}
          {"type":"error",      "message": str}

        Same graph / thread / checkpointer / abort / concurrency-guard as
        ``stream_turn`` — a voice turn and a typed turn are interchangeable. The
        ``final`` text is authoritative (read from checkpoint state with tool /
        thinking blocks stripped) so a caller may TTS it even if it dropped every
        token. Used by the voice channel; the text SSE path keeps ``stream_turn``.
        """
        conv = self._store.get(conversation_id)
        if conv is None:
            yield {"type": "error", "message": "会话不存在"}
            return
        agent_id, thread_id = conv["agent_id"], conv["thread_id"]
        try:
            graph = self._graph_for(agent_id)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"无法构建 {agent_id} 引擎: {exc}"}
            return

        cfg = self._config_for(agent_id, graph, thread_id)
        from langchain_core.messages import HumanMessage

        with self._active_lock:
            if thread_id in self._active_threads:
                yield {"type": "error", "message": "该会话已有进行中的回合，请稍候。"}
                return
            self._active_threads.add(thread_id)

        self._tl.abort = abort
        # 同时按**会话**登记一份:generator 恢复执行可能落在别的 worker 线程上,
        # 那里读不到 thread-local 的这一份(缺陷⑬,见 active_abort_event)。
        self._register_abort(conversation_id, abort)
        self._tl.conversation_id = conversation_id
        # This turn's OWN run id — see active_run_id() for why borrowing the
        # orchestrator's was a sidecar collision waiting to happen.
        self._tl.run_id = self._new_run_id(thread_id)
        self._set_turn_context(conversation_id, self._tl.run_id)
        if turn_holder is not None:
            # 外层包装每次恢复执行前从这里重读并重设（见 reassert_turn_each_resume）。
            turn_holder["conversation_id"] = conversation_id
            turn_holder["run_id"] = self._tl.run_id

        seen_calls: set[str] = set()
        seen_results: set[str] = set()
        pending_inputs: list[object] = []
        # Same stale-approval cleanup as the typed path: a voice turn that was
        # barged in on while an approval was pending parks the thread on the
        # gate, and every later utterance would hit it again.
        rescue = self._stale_interrupt_resume(graph, cfg)
        if rescue is not None:
            yield {"type": "notice",
                   "text": "上一轮遗留的人工审批已过期，已按拒绝自动处理，现在继续。"}
            pending_inputs.append(rescue)
        pending_inputs.append({"messages": [HumanMessage(content=user_text)]})
        aborted = False
        try:
            while pending_inputs:  # one pass per input; a HITL resume adds one
                stream_input: object = pending_inputs.pop(0)
                interrupted = None
                for item in graph.stream(
                    stream_input, config=cfg,
                    stream_mode=["updates", "messages"], subgraphs=True,
                ):
                    if abort is not None and abort.is_set():
                        aborted = True
                        break
                    mode, data = _unpack_stream_item(item)
                    if mode == "messages":
                        delta = _chunk_text(data)
                        if delta:
                            yield {"type": "token", "text": delta}
                    else:  # "updates"
                        if isinstance(data, dict) and "__interrupt__" in data:
                            interrupted = data["__interrupt__"]
                            break
                        yield from _tool_events(data, seen_calls, seen_results)
                if aborted:
                    break
                if interrupted is not None:
                    payload = _interrupt_payload(interrupted)
                    decision = None
                    if hitl_resolver is not None:
                        # Announce BEFORE blocking. The resolver waits on the
                        # operator for up to 15 minutes and this generator can
                        # emit nothing while it does, so a caller that has not
                        # been told what is happening is a silent channel — for
                        # voice, a session that simply stops talking mid-answer.
                        yield {"type": "interrupt", "payload": payload, "waiting": True}
                        try:
                            decision = hitl_resolver(interrupted)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("voice hitl_resolver failed: %s", exc)
                            decision = None
                    if decision is None:
                        # Nobody can (or did) answer: announce and stop. The
                        # thread stays parked, and the next turn's stale-approval
                        # cleanup clears it.
                        yield {"type": "interrupt", "payload": payload}
                        break
                    from langgraph.types import Command
                    pending_inputs.insert(0, Command(resume=decision))
                    continue
        except Exception as exc:  # noqa: BLE001 — surface as an event, never raise out
            logger.error("stream_events(%s) failed: %s", conversation_id, exc)
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            self._tl.abort = None
            self._unregister_abort(conversation_id)
            self._tl.run_id = ""
            self._clear_turn_context()
            with self._active_lock:
                self._active_threads.discard(thread_id)
            # 出错也要收尾。这条路径原本直接 return，于是一轮**出错的语音对话**
            # 既不 touch（会话列表的 updated_at 不动）也不广播 private_turn ——
            # 两个信号源同时缺席，别的窗口再也看不到用户刚说的那句话（它已经在
            # checkpointer 里了）。``stream_turn`` 的同一条路径一直是调 _finish 的
            # （见上文 except 分支），两边不对称纯属遗漏。/#36 的同族。
            try:
                final_render = render_history(self._state_messages(graph, cfg))
            except Exception:  # noqa: BLE001
                final_render = []
            self._finish(conversation_id, conv, thread_id, user_text, final_render)
            return
        finally:
            self._tl.abort = None
            self._unregister_abort(conversation_id)
            self._tl.run_id = ""
            self._clear_turn_context()
            with self._active_lock:
                self._active_threads.discard(thread_id)

        final_text = self._final_assistant_text(graph, cfg)
        yield {"type": "final", "text": final_text, "aborted": aborted}
        try:
            final_render = render_history(self._state_messages(graph, cfg))
        except Exception:  # noqa: BLE001
            final_render = []
        self._finish(conversation_id, conv, thread_id, user_text, final_render)

    def _final_assistant_text(self, graph, cfg) -> str:
        """The last assistant turn's clean spoken text (no tool / thinking blocks)."""
        try:
            from langchain_core.messages import AIMessage
        except Exception:  # noqa: BLE001
            return ""
        for m in reversed(self._state_messages(graph, cfg)):
            if isinstance(m, AIMessage):
                _thinking, text, _tc = extract_ai_parts(m)
                if text and text.strip():
                    return text.strip()
        return ""

    # ── turn bookkeeping ───────────────────────────────────────────────
    def _finish(self, conversation_id, conv, thread_id, user_text, final) -> None:
        with self._active_lock:
            self._active_threads.discard(thread_id)
        try:
            preview = ""
            for m in reversed(final):
                if m.get("role") == "assistant":
                    preview = m.get("content", ""); break
            title = None
            if (conv.get("title") in ("新对话", "新任务", "")
                    and (conv.get("last_message_preview") or "") == ""):
                title = (user_text or "").strip().replace("\n", " ")[:40] or None
            self._store.touch(conversation_id, preview=preview, title=title)
        except Exception as exc:  # noqa: BLE001
            logger.debug("touch after turn failed: %s", exc)
        # 把这一轮的结构化正文镜像进 chat_messages（strangler 第 0 步：只写不读）。
        # 放在 touch 之后、广播之前：收到通知的人去读新家时，这一轮已经在里面了。
        self._mirror_turn(conversation_id, conv, thread_id)
        # 私聊的正文落在 checkpointer 里，不经过 ConversationStore.append_message，
        # 所以那条转录推送对它一次都不会发 —— 别的窗口（另一台机器、同一台的
        # 另一个标签页）因此永远看不到这一轮。。
        # 放在 touch 之后：先落库再广播，收到通知的人读到的一定是新状态。
        try:
            from mast.chat.store import publish_private_turn_finished
            publish_private_turn_finished(
                conversation_id,
                conv.get("agent_id", "") if isinstance(conv, dict) else "")
        except Exception as exc:  # noqa: BLE001 — 通知绝不许毁掉一轮对话
            logger.debug("private turn notify failed: %s", exc)
        # Mirror the finished turn into the experiment record (best-effort).
        if self._record_sink is not None:
            try:
                agent_id = conv.get("agent_id", "") if isinstance(conv, dict) else ""
                self._record_sink(agent_id, user_text or "", preview or "")
            except Exception as exc:  # noqa: BLE001 — never break a turn
                logger.debug("conversation record_sink failed: %s", exc)

    def _mirror_turn(self, conversation_id, conv, thread_id) -> int:
        """把 checkpointer 里这一轮的结构化正文增量复制进 ``chat_messages``。

        **只写不读**（strangler 第 0 步）。读路径仍是 checkpointer，所以这段代码
        出任何问题最坏是「新家少一段历史」——导入账本事后补得回来——而绝不能反过来
        毁掉一轮真实对话。因此整段包在一个 except 里，且 :func:`persist.mirror`
        自己也不抛。

        去重按**消息身份**而不是行数：压缩会让 checkpointer 里的列表变短，按位置
        比对会在压缩之后把整段历史重镜像一遍。
        """
        store = getattr(self, "_message_store", None)
        if store is None:
            return 0
        try:
            from mast.agentruntime.persist import mirror

            agent_id = conv.get("agent_id", "") if isinstance(conv, dict) else ""
            graph = self._graph_for(agent_id)
            cfg = self._config_for(agent_id, graph, thread_id)
            msgs = self._state_messages(graph, cfg)
            if not msgs:
                return 0
            return mirror(store, conversation_id, msgs)
        except Exception as exc:  # noqa: BLE001 — 双写绝不许毁掉一轮对话
            logger.debug("mirror turn into chat_messages failed: %s", exc)
            return 0


# ── structured-event stream helpers (module-level, pure) ───────────────────
def _unpack_stream_item(item):
    """Normalize a ``graph.stream(stream_mode=[...], subgraphs=True)`` item to
    ``(mode, data)``. subgraphs+multi-mode → (namespace, mode, data); be tolerant
    of a 2-tuple (single-mode subgraphs) or a bare update dict."""
    if isinstance(item, tuple):
        if len(item) == 3:
            return item[1], item[2]
        if len(item) == 2:
            return "updates", item[1]
    return "updates", item


def _chunk_text(data) -> str:
    """Extract the assistant TEXT delta from a 'messages'-mode item
    ``(AIMessageChunk, metadata)`` — skips thinking / reasoning blocks."""
    chunk = data[0] if isinstance(data, tuple) and data else data
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)
    return ""


def _short(x, n: int = 80) -> str:
    if isinstance(x, str):
        s = x
    else:
        try:
            s = json.dumps(x, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            s = str(x)
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def _tool_events(data, seen_calls: set, seen_results: set):
    """Yield tool_start / tool_end events from an 'updates'-mode node dict."""
    if not isinstance(data, dict):
        return
    try:
        from langchain_core.messages import AIMessage, ToolMessage
    except Exception:  # noqa: BLE001
        return
    for _node, update in data.items():
        msgs = update.get("messages") if isinstance(update, dict) else None
        if not isinstance(msgs, list):
            continue
        for m in msgs:
            if isinstance(m, AIMessage):
                for tc in (getattr(m, "tool_calls", None) or []):
                    cid = tc.get("id") or f"{tc.get('name')}:{len(seen_calls)}"
                    if cid in seen_calls:
                        continue
                    seen_calls.add(cid)
                    yield {"type": "tool_start", "name": tc.get("name", "tool"),
                           "args_preview": _short(tc.get("args"))}
            elif isinstance(m, ToolMessage):
                rid = getattr(m, "tool_call_id", "") or f"r{len(seen_results)}"
                if rid in seen_results:
                    continue
                seen_results.add(rid)
                body = m.content if isinstance(m.content, str) else str(m.content)
                yield {"type": "tool_end",
                       "name": getattr(m, "name", "tool") or "tool",
                       "result_preview": _short(body)}


def _interrupt_payload(interrupted) -> dict:
    """A DESCRIBED interrupt: the raw value plus ``kind`` / ``skill`` /
    ``rationale`` / ``lg_id``.

    This used to return the raw HITLRequest, whose keys are ``action_requests``
    and ``review_configs`` — so the voice channel, which reads ``skill`` and
    ``rationale``, announced 「需要人工确认：操作」 and could not name what it
    was asking about. The naming here mirrors ``api/hitl_bridge.publish_interrupts``
    so the spoken sentence and the approval card describe the same thing.

    ``event_id`` is deliberately absent: it is minted when the resolver PUBLISHES
    the interrupt, which happens after this. ``lg_id`` (LangGraph's own id) is
    what identifies this pause here.
    """
    try:
        first = interrupted[0] if isinstance(interrupted, (list, tuple)) else interrupted
        val = getattr(first, "value", first)
        lg_id = str(getattr(first, "id", "") or "")
        if not isinstance(val, dict):
            return {"kind": "unknown", "skill": _short(val, 60), "rationale": "",
                    "lg_id": lg_id, "value": str(val)}
        out = dict(val)
        out["lg_id"] = lg_id
        kind = val.get("kind")
        if kind == "workflow_human":
            out["skill"] = (f"工作流 {val.get('workflow', '?')} · 节点 "
                            f"{val.get('node_id', '?')}")
            out["rationale"] = str(val.get("message", ""))
        elif kind == "ask_user":
            ask = val.get("ask") if isinstance(val.get("ask"), dict) else {}
            out["skill"] = "向用户提问"
            out["rationale"] = str(ask.get("question") or val.get("question") or "")
        elif kind == "buffer_hitl":
            events = val.get("events") or []
            kinds = ", ".join(str(e.get("kind", "?")) for e in events
                              if isinstance(e, dict)) or "?"
            out["skill"] = f"缓冲区关键事件：{kinds}"
            out["rationale"] = f"检测到 {len(events)} 个关键硬件事件（{kinds}），需要确认后继续"
        else:
            reqs = [a for a in (val.get("action_requests") or []) if isinstance(a, dict)]
            out["kind"] = "dangerous"
            out["skill"] = "、".join(str(a.get("name") or "?") for a in reqs) or "未知 skill"
            out["rationale"] = str((reqs[0].get("description") if reqs else "") or "")
        return out
    except Exception:  # noqa: BLE001
        return {}


# The wording an operator sees when a stale approval is auto-rejected. It says
# what happened AND that nothing ran — "已过期" alone reads like the action might
# have gone through.
_STALE_NOTE = ("上一轮等待人工审批时会话被中断，该审批已过期 — 已按拒绝处理，"
               "未执行该动作。如仍需要，请重新发起。")


def _stale_reject_value(val) -> object:
    """The fail-CLOSED resume value for an interrupt left over from a dead turn.

    Shape follows the kind, because each pause reads its answer differently and
    a mismatched shape either crashes the node or is silently ignored (which
    would leave the gate open — the 2026-07-28 bug where approve and reject had
    identical outcomes). A DANGEROUS pause needs ONE decision PER action_request:
    the middleware raises when the counts disagree.
    """
    if isinstance(val, dict):
        kind = val.get("kind")
        if kind == "workflow_human":
            # No route name is universally safe; an empty one trips the
            # interpreter's own out-of-set guard and stops the workflow.
            return {"route": "", "note": _STALE_NOTE}
        if kind == "ask_user":
            # A question executes nothing, so there is nothing to reject — the
            # honest answer is "nobody answered", same shape as its timeout.
            return {"selected": [], "custom_text": "", "timeout": True,
                    "note": _STALE_NOTE}
        reqs = [a for a in (val.get("action_requests") or []) if isinstance(a, dict)]
        if reqs:
            return {"decisions": [{"type": "reject", "message": _STALE_NOTE}
                                  for _ in reqs]}
    # buffer_hitl and anything unrecognised: the decisions envelope is what both
    # the HITL middleware and the buffer gate can read.
    return {"decisions": [{"type": "reject", "message": _STALE_NOTE}]}


__all__ = ["ConversationEngine"]
