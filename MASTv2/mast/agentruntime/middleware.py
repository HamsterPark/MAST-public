"""中间件 —— 四个挂点，**全同步**。

与 langchain 的 ``AgentMiddleware`` 的关系
------------------------------------------
形状刻意保持一致（``before_model`` / ``wrap_model_call`` / ``wrap_tool_call`` /
``after_model``），因为要移植的 14 个中间件的**逻辑**都值得原样保留——它们是四个月
现场反馈的沉淀，不是框架适配层。变的只有两件事：

1. **类型归我们自己**（``ModelRequest`` 见 ``model.py``），二期换 provider 时中间件
   不用改签名；
2. **没有 async 双胞胎**。

第 2 条值得多说一句
-------------------
langchain 1.2 的基类不会把 async 钩子代理到同步钩子：实现了 ``wrap_model_call`` 而
没实现 ``awrap_model_call`` 的中间件，在 async 驱动下直接 ``NotImplementedError``。
全仓唯一的 async 消费者是 CLI 的一行 ``await graph.ainvoke()``，它却让**每一个**
中间件都背上写第二份实现的义务——21 份。这份义务被履行到一半过两次，两次的症状都是
「CLI 首次派发即崩」，而同步测试全绿。

CLI 已改同步（``pipeline/main.py``），义务的根源消失。这里不提供 async 挂点，是为了
让「再引入一个 async 消费者」变成一件必须正面讨论的事，而不是某次重构的副作用。
``tests/v2/agents/contract/test_no_async_graph_consumer.py`` 是那道闸门。

栈序
----
列表在前的先跑 ``before_model``，并在 ``wrap_*`` 的洋葱里处于**外层**。与今天
``create_agent`` 的语义一致，移植时挂载顺序可以照抄。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from mast.agentruntime.model import ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class TurnView:
    """``before_model`` / ``after_model`` 看到的回合视图。

    刻意不是 dataclass：它要能**替换消息列表**（压缩用），而替换是一个动作不是一个
    字段赋值——压缩要同时报告它删了多少、留了多少，好让转录里那条 compaction 行说
    得出话。
    """

    __slots__ = ("messages", "agent_id", "scratch", "_replaced")

    def __init__(self, messages: list, agent_id: str = "", scratch: dict | None = None):
        self.messages = messages
        self.agent_id = agent_id
        self.scratch: dict = scratch if scratch is not None else {}
        self._replaced: dict | None = None

    def replace_messages(self, messages: list, *, removed: int = 0,
                         kept: int = 0, note: str = "") -> None:
        """换掉本回合送给模型的消息列表（压缩的落点）。

        记下 removed/kept 是**契约**而不是可选的调试信息：一段被摘要替换过的历史，
        用户有权知道它被替换过，而转录里那条 compaction 行的数字就来自这里。
        """
        self.messages = list(messages)
        self._replaced = {"removed": int(removed), "kept": int(kept), "note": note}

    @property
    def compaction(self) -> dict | None:
        return self._replaced


class Middleware:
    """基类。四个挂点全部可选——不实现的就是不参与。

    ``name`` 用于日志与去重。默认取类名，与 langchain 一致；两个同名实例挂在一起
    是配置错误（上游直接拒绝），这里也拒绝，理由相同：两个同名的中间件，日志里
    分不出是哪一个在说话。
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    # ── 挂点 ───────────────────────────────────────────────────────────
    def before_model(self, turn: TurnView) -> None:
        """模型调用之前。改 ``turn.messages``（或 ``replace_messages``）。"""

    def wrap_model_call(
        self, request: ModelRequest,
        call_next: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """包住模型调用。**改请求、不改 state** —— 与今天 14 个 ``wrap_model_call``
        中间件的纪律一致（``ToolPairGuard`` 那条注释写得最清楚：只改发给 provider
        的请求，不动通道）。"""
        return call_next(request)

    def wrap_tool_call(self, call: "ToolCallView", ctx,
                       call_next: Callable[["ToolCallView"], Any]) -> Any:
        """包住工具调用。SafetyGate 与审批留痕挂在这里。"""
        return call_next(call)

    def after_model(self, turn: TurnView, response: ModelResponse) -> None:
        """模型返回之后。记录、留痕、更新信念。"""


class ToolCallView:
    """``wrap_tool_call`` 看到的一次工具调用。**参数可改**（SI 归一、参数精修）。"""

    __slots__ = ("name", "args", "tool_call_id", "agent_id", "scratch")

    def __init__(self, name: str, args: dict, tool_call_id: str = "",
                 agent_id: str = ""):
        self.name = name
        self.args = dict(args or {})
        self.tool_call_id = tool_call_id
        self.agent_id = agent_id
        self.scratch: dict = {}


class MiddlewareStack:
    """把一串中间件组合成可调用的洋葱。

    组合在**构造时**做一次而不是每次调用现搭：一次工具调用要穿过整条栈，每次
    重建闭包在 IC 的长回合里是白烧的 CPU。
    """

    def __init__(self, middleware: Iterable[Middleware] | None = None):
        self._mw: list[Middleware] = list(middleware or [])
        seen: set[str] = set()
        for m in self._mw:
            if m.name in seen:
                raise ValueError(
                    f"两个中间件同名：{m.name}。日志里分不出是哪一个在说话，"
                    "请给其中一个换个类名。")
            seen.add(m.name)

    def __len__(self) -> int:
        return len(self._mw)

    def __iter__(self):
        return iter(self._mw)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self._mw]

    # ── 驱动 ───────────────────────────────────────────────────────────
    def before_model(self, turn: TurnView) -> None:
        for m in self._mw:
            try:
                m.before_model(turn)
            except Exception as exc:  # noqa: BLE001
                logger.warning("middleware %s.before_model failed: %s", m.name, exc)

    def after_model(self, turn: TurnView, response: ModelResponse) -> None:
        for m in self._mw:
            try:
                m.after_model(turn, response)
            except Exception as exc:  # noqa: BLE001
                logger.warning("middleware %s.after_model failed: %s", m.name, exc)

    def call_model(self, request: ModelRequest,
                   invoke: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """穿过 ``wrap_model_call`` 洋葱调一次模型。

        ⚠️ 这里**不**吞异常。``before_model`` / ``after_model`` 的失败可以降级
        （少注入一段提示，回合还是能跑），但一个包住模型调用的中间件抛出来，意味着
        它对这次调用的判断没有生效——继续调用等于绕过它。SafetyGate 就在这条链上，
        「守卫抛异常所以放行」是这个仓库最不该有的形状。
        """
        chain = invoke
        for m in reversed(self._mw):
            chain = _bind_model_wrapper(m, chain)
        return chain(request)

    def call_tool(self, call: ToolCallView, ctx,
                  invoke: Callable[[ToolCallView], Any]) -> Any:
        """穿过 ``wrap_tool_call`` 洋葱调一次工具。同样不吞异常，理由同上。"""
        chain = invoke
        for m in reversed(self._mw):
            chain = _bind_tool_wrapper(m, ctx, chain)
        return chain(call)


def _bind_model_wrapper(m: Middleware, nxt):
    def _call(request: ModelRequest) -> ModelResponse:
        return m.wrap_model_call(request, nxt)
    return _call


def _bind_tool_wrapper(m: Middleware, ctx, nxt):
    def _call(call: ToolCallView):
        return m.wrap_tool_call(call, ctx, nxt)
    return _call


__all__ = ["Middleware", "MiddlewareStack", "TurnView", "ToolCallView"]
