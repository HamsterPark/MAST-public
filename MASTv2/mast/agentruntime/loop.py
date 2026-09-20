"""``AgentLoop`` —— model-tool 循环。这是整次迁移的心脏。

它替代的是 ``langchain.agents.create_agent`` 编译出来的那张 ReAct 子图。同样的一件
事，少掉一层图：没有节点、没有通道、没有 super-step，因此也没有那一层带来的六条
隐性语义。

四条与旧循环的实质差别
----------------------
**① 结局是返回值，不是「什么都不说」。**
今天 agent 子图作为**没有出边的裸节点**挂在父图上：子图一结束，那条分支就停了——
supervisor 不会被重新进入，``graph.stream()`` 跑干而不抛异常。诊断台账记了 621 次
``fail_silent_end``，而驱动层报的是 ``{"aborted": false, "failed": false}``，UI 显示
「任务完成」。**一条死掉的分支和一次真正的完成，在像素层面无法区分。**
这里每一次 ``run()`` 都以一个 :data:`~mast.agentruntime.events.Outcome` 结束，并发
一条 ``BranchDoneEvent``。编排器可以选择怎么处置，但**不可能不知道**。

**② 预算的单位就是被预算的东西。**
``recursion_limit`` 数的是 super-step，而每个实现 ``before_model``/``after_model``
的中间件都会变成一个图节点，于是「一次工具往返值几步」随中间件数量漂移——同一个
字面量 50 从买约 9 次工具调用变成买 3 次，**没有任何一次 diff 里出现过「限额被改小」**。
这里直接数 ``model_calls`` 与 ``tool_calls``，加中间件不改变任何限额的含义。

**③ 工具在调用线程上顺序执行。**
``ToolNode`` 每次调用都起新线程，而 ``copy_context()`` 复制 contextvars、**不复制
threading.local**，于是用户按下的「停止」到不了正在跑的技能。这里工具就在循环所在
的线程上跑，``RunContext`` 沿参数传下去。顺序执行也符合硬件现实：仪器动作本来就必须
串行。

**④ 交棒不短路，孤儿 tool_call 在构造上不可能。**
每一条 ``AIMessage(tool_calls=…)`` 的每一个 tool_call，都会在同一个 ``run()`` 里得到
配对的 tool 消息——包括交棒那一次。旧世界靠 ``Command(graph=PARENT)`` 短路，配对的
AIMessage 留在子图里蒸发，父通道剩下孤儿 tool result，provider 400（复发四次）。

保留的东西
----------
中间件的四个挂点、挂载顺序、以及它们改请求不改 state 的纪律，全部照旧——那 14 个要
移植的中间件是四个月现场反馈的沉淀，不是框架适配层。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable

from mast.agentruntime.context import RunAborted, RunContext, use_context
from mast.agentruntime.events import (
    BranchDoneEvent,
    MessageEvent,
    Outcome,
    RunEvent,
    StepEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from mast.agentruntime.middleware import (
    Middleware,
    MiddlewareStack,
    ToolCallView,
    TurnView,
)
from mast.agentruntime.model import ChatModelPort, ModelRequest, ModelResponse
from mast.agentruntime.tools import ToolResult, ToolSpec, coerce_result

logger = logging.getLogger(__name__)


@dataclass
class CallLimits:
    """一次 ``run()`` 的预算。**单位是真实单位。**

    对比 ``derive_recursion_limit``：那个函数要靠内省编译图的节点数把「几次工具调用」
    翻译成「几步」，而翻译率随中间件数量变化。这里没有翻译。
    """

    max_model_calls: int = 25
    max_tool_calls: int = 60

    def exceeded(self, counters: "RunCounters") -> str:
        if counters.model_calls > self.max_model_calls:
            return (f"本回合模型调用已达上限（{self.max_model_calls} 次）。"
                    "请把已完成的部分交接出去，别继续往下试。")
        if counters.tool_calls > self.max_tool_calls:
            return (f"本回合工具调用已达上限（{self.max_tool_calls} 次）。"
                    "请把已完成的部分交接出去，别继续往下试。")
        return ""


@dataclass
class RunCounters:
    model_calls: int = 0
    tool_calls: int = 0


@dataclass
class AgentRunResult:
    """一次 ``run()`` 的全部产出。**每个字段都是显式的**。"""

    outcome: Outcome = "final"
    new_messages: list = field(default_factory=list)
    handoff: Any = None                     # HandoffRequest | None
    state_delta: dict = field(default_factory=dict)
    final_text: str = ""
    counters: RunCounters = field(default_factory=RunCounters)
    stop_reason: str = ""

    @property
    def handed_off(self) -> bool:
        return self.outcome == "handoff"


class StallSignal(Exception):
    """中间件用它主动停机（StallGuard 判定原地打转时）。

    是异常而不是返回值，因为它要能从 ``wrap_model_call`` 的深处直接终止本回合；
    循环把它转成 ``outcome="stalled"`` —— 对外**仍然是一个显式结局**。
    """

    def __init__(self, reason: str = ""):
        super().__init__(reason or "stalled")
        self.reason = reason


class AgentLoop:
    """一个 agent 的 model-tool 循环。

    **无状态**：会话身份在 ``messages`` 与 ``RunContext`` 里，同一个 loop 对象可以
    被并发用于不同的 run（编排器 fan-out 就是这么用的）。中间件实例是共享的，所以
    它们不许把 per-run 的东西存在 ``self`` 上——与今天的约定一致。
    """

    def __init__(self, *, name: str, model: ChatModelPort,
                 tools: Iterable[ToolSpec] = (), system_prompt: str = "",
                 middleware: Iterable[Middleware] = (),
                 limits: CallLimits | None = None,
                 stream_tokens: bool = False):
        self.name = name
        self.model = model
        self.tools: dict[str, ToolSpec] = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.stack = MiddlewareStack(middleware)
        self.limits = limits or CallLimits()
        #: 逐 token 发 :class:`TokenEvent`。**默认关**，只有语音链需要它。
        #:
        #: 仍然是**一次**模型调用：``ChatModelPort.stream`` 是个生成器，yield 文本
        #: 增量、return 组装好的 ``ModelResponse``（含 ``tool_calls``）。所以开着
        #: 与关着的差别只在「文本是一次给还是逐段给」，不多付一次往返。
        #:
        #: 那为什么还要开关：不是每个 provider 的流式都靠谱（六家各有各的脾气，见
        #: ``docs/api_providers/``），而只有语音真的需要「第一句话早几秒出来」。
        #: 默认走 ``invoke`` = 默认走那条被六家实测过的路。
        #:
        #: 语音是**唯一**的消费者（``voice/session.py`` 只读 ``type=="token"``），
        #: 它走 ``ConversationEngineV2.stream_events``，那条路会把它打开。
        self.stream_tokens = stream_tokens

    # ── 主循环 ─────────────────────────────────────────────────────────
    def run(self, messages: list, ctx: RunContext | None = None,
            ) -> Generator[RunEvent, None, AgentRunResult]:
        """跑一轮，yield 事件，返回 :class:`AgentRunResult`。

        同步生成器：**惰性正是「暂停」原语**——调用方不拉下一个事件，这一跳就停在
        那里。今天的 hold 功能靠的就是 ``graph.stream()`` 的这条性质，换掉运行时
        不能把它弄丢。
        """
        ctx = ctx or RunContext(agent_id=self.name)
        counters = RunCounters()
        convo = list(messages)
        new_messages: list = []
        # seed from the caller's carried state (multi-turn drivers put the previous
        # run's state_delta in ctx.extra["state"]) so loaded tool packs survive a turn
        _seed = (getattr(ctx, "extra", None) or {}).get("state")
        state_delta: dict = dict(_seed) if isinstance(_seed, dict) else {}
        final_text = ""

        try:
            while True:
                if ctx.aborted():
                    return (yield from self._done(ctx, "aborted", new_messages, None,
                                      state_delta, final_text, counters,
                                      "用户中止了本次运行。"))

                # ── 模型调用 ───────────────────────────────────────────
                turn = TurnView(list(convo), agent_id=self.name)
                self.stack.before_model(turn)
                convo = list(turn.messages)
                if turn.compaction:
                    yield self._compaction_event(ctx, turn.compaction)

                request = ModelRequest(
                    system_prompt=self.system_prompt,
                    messages=list(convo),
                    tools=[t for t in self.tools.values()],
                    state=dict(state_delta),      # loaded_tool_packs 等要让中间件看见
                )
                counters.model_calls += 1
                over = self.limits.exceeded(counters)
                if over:
                    return (yield from self._done(ctx, "limit", new_messages, None,
                                      state_delta, final_text, counters, over))

                try:
                    response = self.stack.call_model(
                        request, self._model_caller(ctx))
                except StallSignal as sig:
                    return (yield from self._done(ctx, "stalled", new_messages, None,
                                      state_delta, final_text, counters,
                                      sig.reason or "原地打转，已停机。"))
                except RunAborted:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] model call failed: %s", self.name, exc)
                    return (yield from self._done(ctx, "error", new_messages, None,
                                      state_delta, final_text, counters,
                                      f"模型调用失败：{type(exc).__name__}: {exc}"))

                self.stack.after_model(turn, response)
                msg = response.message
                if msg is not None:
                    convo.append(msg)
                    new_messages.append(msg)
                if response.text:
                    final_text = response.text
                    yield MessageEvent(agent=self.name, role="assistant",
                                       text=response.text, t=time.time())

                yield StepEvent(agent=self.name, model_calls=counters.model_calls,
                                tool_calls=counters.tool_calls,
                                limit=self.limits.max_model_calls,
                                unit="model_call", t=time.time())

                # ── 没有工具调用 = 这一跳的交付物 ──────────────────────
                if not response.tool_calls:
                    return (yield from self._done(ctx, "final", new_messages, None,
                                      state_delta, final_text, counters, ""))

                # ── 逐个执行工具（顺序、同线程） ──────────────────────
                handoff = None
                batch = list(response.tool_calls)
                answered: set[str] = set()
                try:
                    for idx, call in enumerate(batch):
                        if ctx.aborted():
                            self._pair_unanswered(
                                convo, new_messages, batch, answered,
                                "本次运行已被用户中止，此调用未执行。")
                            return (yield from self._done(
                                ctx, "aborted", new_messages, None, state_delta,
                                final_text, counters, "用户中止了本次运行。"))
                        counters.tool_calls += 1
                        over = self.limits.exceeded(counters)
                        if over:
                            # ★ 已经发出的 tool_call **每一个**都必须有配对的 tool
                            #   消息。之前这里只补了「当前这个」，批次里排在它后面
                            #   的仍然是孤儿 —— 见 :meth:`_pair_unanswered`。
                            self._pair_unanswered(convo, new_messages, batch,
                                                  answered, over)
                            return (yield from self._done(
                                ctx, "limit", new_messages, None, state_delta,
                                final_text, counters, over))

                        # ★ 先把「要开始了」发出去，**再**执行 —— 消费者（语音的
                        #   「开始扫描」、面板的「正在扫描」）需要的是这个时刻，
                        #   而不是执行完之后的追认。此处 yield 不会让技能半途挂起：
                        #   它还没开始。见 ``_run_one_tool`` 的 docstring。
                        yield self._start_event(call, self.name)

                        result, evs = self._run_one_tool(call, ctx)
                        for ev in evs:
                            yield ev
                        tool_msg = self._tool_message(call, result.text)
                        convo.append(tool_msg)
                        new_messages.append(tool_msg)
                        answered.add(_call_id(call))
                        if result.state_delta:
                            state_delta.update(result.state_delta)
                        if result.handoff is not None and handoff is None:
                            handoff = result.handoff
                        _ = idx
                except RunAborted:
                    # 工具体里抛出来的中止 —— 同样要把这一批**剩下的**补齐配对，
                    # 否则下一轮的历史里就有孤儿。
                    self._pair_unanswered(
                        convo, new_messages, batch, answered,
                        "本次运行已被用户中止，此调用未执行。")
                    raise

                if handoff is not None:
                    return (yield from self._done(ctx, "handoff", new_messages, handoff,
                                      state_delta, final_text, counters, ""))

        except RunAborted:
            return (yield from self._done(ctx, "aborted", new_messages, None, state_delta,
                              final_text, counters, "用户中止了本次运行。"))

    # ── 内部 ───────────────────────────────────────────────────────────
    def _pair_unanswered(self, convo: list, new_messages: list, batch: list,
                         answered: set, note: str) -> None:
        r"""给这一批里**还没有配对回应**的 tool_call 各补一条 tool 消息。

        ★ 为什么需要它（2026-08-27 实测发现）
        -----------------------------------
        这个循环刻意**不挂** ``ToolPairGuardMiddleware``，理由写在 ``assembly.py``
        里：「新循环里孤儿构造上不可能」。那句话对**一次跑完的 run** 成立 —— 每个
        tool_call 在同一次 ``run()`` 里得到配对回应。

        **但历史是跨 run 的。** 实测：模型在一个批次里要了两个工具调用，第一个跑
        完时用户按了停止 ⇒ 循环中止，第二个永远没有配对消息 —— 而带着两个
        ``tool_calls`` 的那条 AIMessage **已经落库**。下一轮把这段历史送给
        OpenAI-compat provider，就是一个孤儿 tool_call ⇒ **400**。

        这正是本仓复发过四次、最后靠 ToolPairGuard 才压住的那一类；而它现在发生在
        用户最常做的动作上：**按停止，然后继续这个对话**。

        三条早退路径都要补：中止（循环顶部）、撞上限、以及**工具体里抛出来的
        中止**。之前只有「撞上限」补了，而且只补了当前这一个 —— 批次里排在它后面
        的仍然是孤儿。

        补一条「未执行」的说明而不是把 tool_call 从 AIMessage 里摘掉：模型确实
        要求过这次调用，那是历史事实；而下一轮的模型看到「你要过、因为停止没跑」
        比看到「你没要过」更接近真相。
        """
        for call in batch:
            cid = _call_id(call)
            if cid in answered:
                continue
            msg = self._tool_message(call, note)
            convo.append(msg)
            new_messages.append(msg)
            answered.add(cid)

    def _invoke_model(self, request: ModelRequest) -> ModelResponse:
        return self.model.invoke(request)

    def _model_caller(self, ctx: RunContext):
        """中间件链最里面那一层 —— 按 :attr:`stream_tokens` 选 invoke 还是 stream。

        ★ token 走 ``ctx.emit`` 这条**旁路**，不走 ``yield``（2026-08-27）
        ------------------------------------------------------------------
        中间件链（``wrap_model_call``）是普通的函数调用，**它中间 yield 不出东西**。
        而 token 的全部价值就在于「边生成边出来」，攒到调用结束再发等于没流式。

        所以增量走 ``RunContext.emit``：消费方（``ConversationEngineV2.stream_events``）
        在工作线程里跑这个循环，用队列收 —— 语音本来就是这个形状
        （``voice/session.py::_iter_turn_events`` 把同步生成器桥进 async 也是一条队列）。

        判据是 :attr:`stream_tokens`，**不去猜 ``emit`` 接没接**：``RunContext.emit``
        的默认值是一个 no-op lambda 而不是 ``None``，「有没有人收」在这里根本判不出来
        —— 而 ``stream_tokens=True`` 本身就是调用方的意愿声明。
        """
        if not self.stream_tokens or ctx is None:
            return self._invoke_model

        def _call(request: ModelRequest) -> ModelResponse:
            gen = self.model.stream(request)
            while True:
                try:
                    delta = next(gen)
                except StopIteration as stop:
                    resp = stop.value
                    break
                if delta:
                    try:
                        ctx.emit(TokenEvent(agent=self.name, text=delta,
                                            t=time.time()))
                    except Exception as exc:  # noqa: BLE001 — 收端坏了不该毁掉回合
                        logger.debug("[%s] token sink failed: %s", self.name, exc)
            # 端口的 stream 保证 return 一个 ModelResponse；万一某个实现没给，
            # **退回一次 invoke** 而不是把 None 往下传 —— 后者会在 after_model
            # 里以 AttributeError 收场，而那个报错指向错的地方。
            if resp is None:
                logger.warning("[%s] stream() 没有返回 ModelResponse，退回 invoke",
                               self.name)
                return self._invoke_model(request)
            return resp

        return _call

    @staticmethod
    def _start_event(call: Any, agent: str) -> "ToolStartEvent":
        """「要开始跑这个工具了」—— 由调用方在**执行之前** yield 出去。"""
        name = str(call.get("name") if isinstance(call, dict)
                   else getattr(call, "name", ""))
        args = (call.get("args") if isinstance(call, dict)
                else getattr(call, "args", None)) or {}
        call_id = str(call.get("id") if isinstance(call, dict)
                      else getattr(call, "id", "") or "")
        return ToolStartEvent(agent=agent, name=name, tool_call_id=call_id,
                              preview=_preview(args),
                              args=dict(args) if isinstance(args, dict) else {},
                              t=time.time())

    def _run_one_tool(self, call: Any, ctx: RunContext):
        """执行一个工具调用，返回 ``(ToolResult, [事件])``。

        ★ ``tool_start`` **不在这里** —— 它由调用方在执行前先 yield（2026-08-27）
        ------------------------------------------------------------------------
        原来它和结束事件一起攒着，等工具跑完再一并交出去。实测：

            工具 开始     0.000s
            工具 结束     0.601s
            事件 tool_start 0.601s   ← 迟到了整整一次执行

        后果是具体的：语音靠 ``tool_start`` 播「开始扫描」，于是那句话在**扫描结束
        之后**才说出口 —— 真机上晚几十秒，而且**说出口那一刻它是假话**。群聊面板
        同理，「正在扫描」的状态在扫描已经结束时才亮起来。

        攒着的**理由本身是对的**（「一个技能跑到一半因为没人拉事件而挂起，是硬件
        系统里绝不能有的形状」），但它**用过了头**：那条不变式说的是**执行期间**
        不能被拉取节奏打断。在工具**还没开始**的时候 yield，技能不可能半途挂起 ——
        而那正好是 hold 想要的语义（「停在下一个工具之前」），循环的 docstring 本来
        就把「不拉事件 = 暂停」当作原语。

        所以剩下的事件（tool_end、未知工具的说明）仍然攒着：它们发生在执行**之后**，
        没有理由把它们拆开。
        """
        name = str(call.get("name") if isinstance(call, dict) else getattr(call, "name", ""))
        args = (call.get("args") if isinstance(call, dict)
                else getattr(call, "args", None)) or {}
        call_id = str(call.get("id") if isinstance(call, dict)
                      else getattr(call, "id", "") or "")
        events: list[RunEvent] = []

        spec = self.tools.get(name)
        if spec is None:
            # 模型调了一个不存在的工具。**告诉它**，而不是静默失败——
            # 「指向拿不到的工具」会让模型把失败当成关于实验的信息继续推理。
            text = (f"没有名为 {name} 的工具。可用的是："
                    f"{', '.join(sorted(self.tools)) or '（无）'}。")
            events.append(ToolEndEvent(agent=self.name, name=name, ok=False,
                                       tool_call_id=call_id, preview=text,
                                       t=time.time()))
            return ToolResult(text=text, ok=False), events

        view = ToolCallView(name=name, args=args, tool_call_id=call_id,
                            agent_id=self.name)
        try:
            # ★ 桥接进来的 langchain 工具收不到 ctx（它们的签名里没这个位置），
            #   而 ``ask_user`` 之类需要它才能把问题送到用户面前。挂在
            #   ContextVar 上是安全的，**因为工具就在这条线程上顺序跑** ——
            #   v1 用 threading.local 之所以失效，是 ToolNode 每次调用都换线程。
            #   见 ``context.use_context`` 的注释。
            with use_context(ctx):
                raw = self.stack.call_tool(
                    view, ctx,
                    (lambda v: spec.fn(v.args, ctx, tool_call_id=v.tool_call_id))
                    if getattr(spec, "wants_call_id", False)
                    else (lambda v: spec.fn(v.args, ctx)))
            result = coerce_result(raw)
        except RunAborted:
            raise
        except Exception as exc:  # noqa: BLE001
            # 工具抛异常是**常态**（硬件会拒绝、参数会越界）。把它变成模型读得懂
            # 的一句话继续对话，比让整个回合崩掉有用得多。
            logger.info("[%s] tool %s raised: %s", self.name, name, exc)
            result = ToolResult(text=f"{name} 执行失败：{type(exc).__name__}: {exc}",
                                ok=False)
        events.append(ToolEndEvent(agent=self.name, name=name, ok=result.ok,
                                   tool_call_id=call_id,
                                   preview=result.preview or _preview(result.text),
                                   t=time.time()))
        return result, events

    def _tool_message(self, call: Any, text: str):
        """造一条与 ``call`` 配对的 tool 消息。

        **配对是这里的全部要点。** 每一个 tool_call 都在同一个 ``run()`` 里得到
        它的回应，所以孤儿 tool result 在构造上不可能出现——``ToolPairGuard`` 这类
        「送 provider 之前剥孤儿」的防御层因此整个不需要。

        ⚠️ 「构造上不可能」这句话是**有条件的**（2026-08-27 实测补正）：它要求
        每条早退路径都把这一批**剩下的** tool_call 也配对掉。中止与撞上限曾经
        漏过 —— 见 :meth:`_pair_unanswered`。那句话现在仍然成立，但成立是**靠那个
        方法**，不是靠循环的形状自动成立。
        """
        from langchain_core.messages import ToolMessage

        call_id = str(call.get("id") if isinstance(call, dict)
                      else getattr(call, "id", "") or "")
        name = str(call.get("name") if isinstance(call, dict)
                   else getattr(call, "name", "") or "")
        return ToolMessage(content=text or "", tool_call_id=call_id, name=name)

    def _compaction_event(self, ctx: RunContext, info: dict):
        from mast.agentruntime.events import CompactionEvent

        return CompactionEvent(agent=self.name, removed=int(info.get("removed") or 0),
                               kept=int(info.get("kept") or 0), t=time.time())

    def _done(self, ctx: RunContext, outcome: Outcome, new_messages: list,
              handoff, state_delta: dict, final_text: str,
              counters: RunCounters, stop_reason: str,
              ) -> Generator[RunEvent, None, AgentRunResult]:
        """收尾：**必发** ``BranchDoneEvent``，然后返回结果。

        「必发」是这次迁移最想要的那件事。今天不交棒的分支什么都不说，于是
        「死分支」与「任务完成」在驱动层看来完全一样。

        ⚠️ 它是 ``yield`` 出去的，**不是** ``ctx.send`` 出去的 —— 事件只有一条
        通道。第一版让终结事件走 ctx 而其余事件走 yield，结果是：一个只 drain
        生成器的消费者收不到这次迁移最重要的那个事件，**而且没有任何报错**
        （后台编排的接线上真的这么漏过一次）。两条通道还意味着同时接了两边的
        消费者会看到重复的行。

        用法：``result = yield from self._done(...)``。
        """
        yield BranchDoneEvent(agent=self.name, outcome=outcome,
                              stop_reason=stop_reason, final_text=final_text,
                              t=time.time())
        return AgentRunResult(outcome=outcome, new_messages=new_messages,
                              handoff=handoff, state_delta=state_delta,
                              final_text=final_text, counters=counters,
                              stop_reason=stop_reason)


def _preview(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _call_id(call: Any) -> str:
    """一个 tool_call 的 id。dict 与对象两种形状都接 —— provider 给的是前者，
    langchain 的 ``ToolCall`` 是后者，而这个循环两种都会遇到。"""
    return str(call.get("id") if isinstance(call, dict)
               else getattr(call, "id", "") or "")


__all__ = ["AgentLoop", "AgentRunResult", "CallLimits", "RunCounters", "StallSignal"]
