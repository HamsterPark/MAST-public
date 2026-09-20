"""``RunContext`` —— 一次运行的身份、闸门与出口，**显式地传下去**。

为什么是参数而不是环境
----------------------
今天这些东西靠 ``threading.local`` 与 contextvar 传播，而它们**穿不过执行器**：

* LangGraph 的 ``ToolNode`` 每次调用都 ``get_executor_for_config(config)`` 起线程，
  ``ContextThreadPoolExecutor.submit`` 做的是 ``copy_context().run(...)`` ——复制
  contextvars，**不复制 threading.local**。技能就住在这后面，于是用户按下的
  「停止」到不了正在跑的技能（缺陷⑬，两天撞三次，修了三个 commit）。
* 计费捕获挂在 ``on_chat_model_start`` 上，那个回调拿不到 conversation_id；要它拿到
  就得把 id 从 HTTP 路由穿到模型工厂，中间隔着执行器——**穿不过去的请求会安静地
  不带标记**，于是「对不上账」和「没记录」变成同一种表现。

两处是同一个病：**关键身份走的是环境，而环境有边界**。新循环把它们放进一个显式对象，
沿调用链传参。工具在调用线程上顺序执行（见 ``loop.py``），所以传参这条路不会断。

兼容：``turn_context`` 仍然照写
------------------------------
深处还有一批读 ``core.turn_context`` 的代码（技能、旧适配层）。工具适配器在调用技能
之前照旧设置它，但**运行时自身的正确性不再依赖它**——那是给别人的便利，不是我们的
依据。
"""
from __future__ import annotations

import contextlib
import contextvars
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

#: 一次 ``ask_human`` 最多等多久（秒）。与 ``api/hitl_bridge`` 的
#: ``MAX_APPROVAL_WAIT_S`` 同值——那一层整体保留，两处不能各说各话。
DEFAULT_ASK_TIMEOUT_S = 900.0


@dataclass
class RunContext:
    """一次 run（群聊任务 / 私聊回合 / 后台运行 / composite 委托）的运行期上下文。

    刻意是可变 dataclass 而不是 frozen：``emit`` 与 ``ask_human`` 在装配的不同阶段
    被接上（编排器建好 sink 之后才知道往哪儿发），而 run 的身份在构造时就定了。
    """

    run_id: str = ""
    conversation_id: str = ""
    thread_id: str = ""
    agent_id: str = ""

    #: 用户的「停止」。**唯一的中止真源**——不是轮询某个全局标志，也不是
    #: thread-local 的一份拷贝。
    abort: threading.Event = field(default_factory=threading.Event)

    #: 事件出口。循环产出的每个 :class:`~mast.agentruntime.events.RunEvent` 都经过
    #: 它。默认丢弃：一个没接 sink 的 run 仍然要能跑完（测试、CLI）。
    emit: Callable[[Any], None] = lambda _event: None

    #: 阻塞式提问。接上的是 ``api/hitl_bridge``（发布 → 等 Event → 拿答案）。
    #: **None = 这条入口没有提问通道**——工具据此降级成「按保守默认继续并说明」，
    #: 而不是假装问过了。读不到不是答案。
    ask_human: Callable[[dict], dict | None] | None = None

    #: 附加信息（experiment_id / sample_id / 模型覆写…）。刻意松散：这里不该变成
    #: 第二个 state schema。
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``abort=None`` 是调用方在说「这条入口没有中止通道」（CLI、测试、
        # composite 委托），意思是「从未中止」——不是崩溃。显式传 None 会盖掉
        # ``default_factory``，所以在这里补一个。
        #
        # 这是「读不到不该被折叠成一个具体的值」的**反向**用法：这里读不到的是
        # 一个通道而不是一个读数，而「没有停止按钮」的正确语义恰好就是「没人按过」。
        if self.abort is None:
            self.abort = threading.Event()

    # ── 闸门 ───────────────────────────────────────────────────────────
    def aborted(self) -> bool:
        return self.abort.is_set()

    def raise_if_aborted(self) -> None:
        if self.abort.is_set():
            raise RunAborted(self.run_id)

    # ── 出口 ───────────────────────────────────────────────────────────
    def send(self, event) -> None:
        """发一个事件。**永不抛**——一个坏掉的 sink 不该毁掉一次真实运行。"""
        try:
            self.emit(event)
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).debug(
                "event sink raised; dropping event %r", getattr(event, "kind", "?"))

    def ask(self, payload: dict, *, timeout_s: float = DEFAULT_ASK_TIMEOUT_S):
        """问用户一个问题并**原地阻塞**等答案。

        返回 ``None`` 表示「没有答案」——没接提问通道、超时、或者运行被中止。
        三种都是「没答案」，调用方**必须**把它当成「读不到」而不是某个具体的值：
        这个仓库最贵的一类 bug 就是「读不到」被折叠成一个具体的值。

        没有重放。LangGraph 需要 ``interrupt()`` + 重放整个工具调用，是因为图节点
        不允许长期占用执行器；而我们四条驱动链全在可以阻塞的工作线程上。

        ★ **谁在问，由问的那个 ctx 自己报**（2026-08-27 实测修）
        -----------------------------------------------------
        群聊那条链上 ``attach()`` 是在**编排器的** ctx 上做的一次，``owner`` 因此
        被写死成 ``_supervisor``；而真正提问的是某条分支里的 agent。实测：
        literature 问的问题，卡片上写着 ``agent_id='_supervisor'``。

        旧路径踩过同一个坑的**另一面**并留下了规则 —— ``publish_interrupts``：

            Self-reported owner wins: … otherwise `owner` would fall back to
            instrument_control — **attributing the supervisor's question to an
            agent that never asked it**.

        在一台某个 agent 会动机器的机子上，「谁在问」不是装饰：用户看到
        「instrument_control 问：要不要进针」和「_supervisor 问…」，判断依据不一样。

        所以身份在**这里**注入 —— ``for_agent()`` 派生出来的分支 ctx 带着自己的
        ``agent_id``，一处修好全链受益。payload 里已经写了的（调用方明确指定）优先。
        """
        if self.ask_human is None:
            return None
        try:
            enriched = {**payload, "_timeout_s": timeout_s}
            if not enriched.get("agent_id") and self.agent_id:
                enriched["agent_id"] = self.agent_id
            return self.ask_human(enriched)
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning("ask_human failed: %s", exc)
            return None

    # ── 派生 ───────────────────────────────────────────────────────────
    def for_agent(self, agent_id: str) -> "RunContext":
        """给一条分支用的上下文：换 agent 身份，**共享同一个 abort 与 sink**。

        共享 abort 是要点：用户按一次停止，所有在飞的分支都要停。每条分支各自
        持一个 Event 的设计试过一次，结果是「停止」只停住了当时那一条。
        """
        return RunContext(
            run_id=self.run_id, conversation_id=self.conversation_id,
            thread_id=self.thread_id, agent_id=agent_id,
            abort=self.abort, emit=self.emit, ask_human=self.ask_human,
            extra=dict(self.extra))


class RunAborted(RuntimeError):
    """用户中止了这次运行。**是控制流，不是错误**——沿途每一层都要放行。

    与 ``GraphInterrupt`` 在旧世界里的地位相同：五处 ``except Exception`` 曾经把
    它吞掉，导致「停止」按钮按下去没反应。
    """

    def __init__(self, run_id: str = ""):
        super().__init__(f"run aborted: {run_id}" if run_id else "run aborted")
        self.run_id = run_id


# ─────────────────────────────────────────────────────────────────────
# 「正在跑的那个 ctx」—— 只给**桥接进来的 langchain 工具**用
# ─────────────────────────────────────────────────────────────────────
#
# ★ 为什么要有它（2026-08-27 实测发现）
# ------------------------------------
# 新写的工具走 ``ToolSpec.fn(args, ctx)``，上下文是**显式参数**——那是这一层的设计
# 立场，不变。但经 ``spec_from_langchain_tool`` 桥接进来的约 156 个既有 ``@tool``
# 收不到 ctx（langchain 的工具签名里没有这个位置）。
#
# 后果实测出来了，而且是这次迁移最该抓住的那一类：``ask_user`` 在 v2 上调
# langgraph 的 ``interrupt()`` → ``KeyError``（不在图运行时里）→ 它自己把这个异常
# 优雅地转成一句「这个入口没有接提问通道」→ **这一轮以 ``outcome="final"`` 正常
# 收场**。agent 问了问题、用户永远看不到、现场看起来完全正常。
#
# ⚠️ 为什么这里可以用 ContextVar，而 v1 的 ``threading.local`` 不行
# ----------------------------------------------------------------
# v1 的技能靠 ``threading.local`` 拿 run 身份与 abort，而 langgraph 的 ``ToolNode``
# **每次调用都换线程**，thread-local 传不过去（缺陷⑬：用户的「停止」到不了正在
# 跑的技能）。新循环**在自己的线程上顺序执行工具**（``test_agent_loop`` 钉着这一
# 条），所以「设一个、调用、清掉」在这里是可靠的。
#
# 这不是把显式上下文改回隐式：它只服务于**桥**，是一期的过渡物。工具移植成
# ``ToolSpec.fn(args, ctx)`` 之后，对应的读取点就该删掉。
_current: contextvars.ContextVar["RunContext | None"] = contextvars.ContextVar(
    "mast_agentruntime_ctx", default=None)


def current_context() -> "RunContext | None":
    """当前正在执行的工具所属的 ``RunContext``；不在工具调用里则为 ``None``。"""
    return _current.get()


@contextlib.contextmanager
def use_context(ctx: "RunContext | None"):
    """在一次工具调用期间把 ``ctx`` 挂上去，**无论如何都还原**。

    用 token 还原而不是设回 ``None``：嵌套调用（一个工具内部又驱动一个子循环）时
    设回 None 会把外层的也抹掉，而那种「问着问着通道没了」最难查。
    """
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


__all__ = ["RunContext", "RunAborted", "DEFAULT_ASK_TIMEOUT_S",
           "current_context", "use_context"]
