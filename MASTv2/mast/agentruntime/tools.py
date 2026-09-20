"""工具协议 —— 循环调工具、工具回话的那一层。

三个刻意的选择
--------------
1. **交棒是返回值，不是短路**。今天 agent 交棒靠
   ``Command(goto="supervisor", graph=Command.PARENT, ...)``，它**短路退出子图**：
   发起交棒的那条 ``AIMessage(tool_calls=…)`` 留在子图命名空间不传播，只有
   ``ToolMessage`` 进了父通道 → 孤儿 tool result → OpenAI-compat provider 400
   （复发四次，最后靠 ``ToolPairGuardMiddleware`` 在送 provider 前剥掉孤儿）。
   这里交棒只是 ``ToolResult.handoff`` 一个字段：循环收到之后正常收尾，**配对的
   ``ToolMessage`` 由循环自己生成**，孤儿在构造上不可能出现。

2. **state 更新是返回值，不是注入**。今天工具要写 state 得返回 ``Command(update=…)``，
   而这需要子图声明 ``state_schema``；不声明就**静默丢弃、无异常无警告**——
   ``skill_adapter`` 的四组写入因此空转了两个多月。这里 ``ToolResult.state_delta``
   是个普通字段，循环把它累加进 ``AgentRunResult``；丢不掉，因为没有第三方在中间
   决定要不要接。

3. **保留 ``SkillToolResult`` 的兼容技巧**。约 175 个既有单测直接 ``tool.func(...)``
   然后 ``str(result)`` / ``"x" in result``。那些断言仍然有价值，不该为了换基类而
   集体改写。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HandoffRequest:
    """「我做完了，交回编排器」。"""

    target: str = "supervisor"
    reason: str = ""


@dataclass
class ToolResult:
    """一次工具调用的结果。

    ``text`` 是给模型看的那段话（会变成配对的 tool 消息）。其余字段是给运行时的。
    """

    text: str = ""
    handoff: HandoffRequest | None = None
    state_delta: dict = field(default_factory=dict)
    ok: bool = True
    #: 给事件流的一句摘要（工具面板显示用），空则用 ``text`` 的前若干字。
    preview: str = ""


@dataclass
class ToolSpec:
    """循环认识一个工具所需的全部信息。

    ``fn`` 接收 ``(args, ctx)`` —— **上下文是显式参数**。今天技能靠
    ``threading.local`` 拿 run 身份与 abort，而 ``ToolNode`` 每次调用都换线程，
    thread-local 传不过去（缺陷⑬：用户的「停止」到不了正在跑的技能）。
    """

    name: str
    description: str = ""
    #: JSON Schema。一期由 langchain 的 ``convert_to_openai_tool`` 生成，
    #: 二期由 provider 层生成——两边都是 JSON Schema，所以这个字段不用变。
    schema: dict = field(default_factory=dict)
    fn: Callable[[dict, Any], Any] = None  # type: ignore[assignment]
    #: 这个工具是否驱动仪器。编排器据此做**派发级**互斥（见 orchestrator 的
    #: 「在飞集合」），技能级令牌 ``core.instrument_lock`` 仍是兜底。
    touches_instrument: bool = False
    #: ``fn`` 接受 ``tool_call_id=`` 且需要模型给的那个 id（带 ``InjectedToolCallId``
    #: 的 langchain 工具必须用完整 ToolCall 调用）。循环据此决定怎么调 ``fn``。
    wants_call_id: bool = False


class SkillToolResult(ToolResult):
    """工具的返回类型 —— ``_SkillToolReturn(Command)`` 与 ``ArtifactToolReturn(Command)``
    的**直接替身**。

    保留四个兼容技巧，因为约 175 个既有单测依赖它们：

    * ``.update`` —— ``state_delta`` 的别名（旧代码读这个名字）；
    * ``__str__`` —— 返回摘要文本（``str(result)`` 到处都是）；
    * ``__contains__`` —— 对摘要做子串测试（``"扫描完成" in result``）；
    * ``__len__`` —— 摘要长度（``ArtifactToolReturn`` 有这一个）。

    这些不是技术债，是**保住既有断言的价值**：那些测试问的是「工具回话说了什么」，
    那个问题在新运行时下一字未变。

    签名对齐（2026-08-27）
    ---------------------
    第二个位置参数叫 ``update``，并接受 ``tool_call_id`` / ``name`` 两个关键字 ——
    因为这个类将来要**原地替掉**那两个 ``Command`` 子类，而它们的调用点
    有约 156 处。签名对不上，那次替换就从「改两个基类」变成「改 156 个调用点」。

    ``tool_call_id`` / ``name`` 被**接受但忽略**：在旧世界里它们用来往
    ``update["messages"]`` 里塞一条配对 ``ToolMessage``（``Command`` 靠那个把 tool
    回应送进通道）。新循环自己生成配对消息（见 ``AgentLoop._tool_message``），所以
    这里不需要它们 —— 但**必须能接住**，否则调用点就得改。
    """

    def __init__(self, summary: str = "", update: dict | None = None, *,
                 tool_call_id: str = "", name: str = "", **kw):
        # 显式吞掉这两个：见 docstring。留一条注释而不是默默忽略 —— 「参数收下了
        # 却什么也不做」如果没写下来，读的人会以为它有用。
        _ = (tool_call_id, name)
        # ★ 过滤在**构造处**，不只在 ``coerce_result`` 里（2026-08-27 副本试验）。
        #
        # 之前 ``messages`` 那个传输载体只在 ``coerce_result`` 走 ``_state_only()``
        # 时被摘掉。而 ``coerce_result`` 的第一条就是「已经是 ToolResult 就原样
        # 放行」—— 于是任何**直接构造** ``SkillToolResult(s, {"messages": […]})``
        # 的路径都绕过了那道过滤，``messages`` 变成一个假产物字段一路进
        # ``TaskState.artifacts``。
        #
        # 未来这条路径会从「少见」变成「唯一」：那两个 Command 子类
        # 换成本类的别名之后，工具返回的就是本类的实例，``coerce_result`` 一律
        # 原样放行。**过滤器要待在数据被造出来的地方**，不是待在某一条读它的路上。
        super().__init__(text=summary, state_delta=_state_only(update), **kw)

    @property
    def update(self) -> dict:
        return self.state_delta

    def __str__(self) -> str:
        return self.text or ""

    def __contains__(self, needle: object) -> bool:
        return str(needle) in (self.text or "")

    def __len__(self) -> int:
        return len(self.text or "")


#: ``update`` 里**不是** state 的键。``messages`` 是 langgraph 的配对 ToolMessage
#: 载体（``Command`` 靠它把 tool 回应塞进通道）；新循环自己生成配对消息，所以这一项
#: 到这里就该被摘掉，否则它会变成一个名叫 "messages" 的假产物字段。
_NON_STATE_UPDATE_KEYS = frozenset({"messages"})


def coerce_result(raw: Any) -> ToolResult:
    """把工具返回的任何东西规整成 :class:`ToolResult`。

    宽进严出：技能返回字符串、dict、``ToolResult``、langgraph 的 ``Command``、
    甚至 ``None`` 都能接住。**不接受的是静默丢弃**。

    ★ 为什么必须认得 ``Command``（2026-08-27 实测修复）
    ---------------------------------------------------
    今天约 156 个 ``@tool`` 返回的是 ``ArtifactToolReturn`` / ``_SkillToolReturn``
    —— 两个 ``langgraph.types.Command`` 的子类，产物写在 ``.update`` 里。第一版这里
    只认字符串 / dict / ToolResult，于是 ``Command`` 落进最后那条 ``str(raw)``：

    * ``text`` 完全正确（那两个类的 ``__str__`` 返回摘要）；
    * ``state_delta`` **空**。

    也就是说：agent 产出了综述、扫了图、写了草稿，而**产物一个都到不了 TaskState，
    下游 agent 什么也看不到，且全程没有任何报错**。这正是
    「不声明 ``state_schema`` ⇒ ``Command(update=…)`` 被静默丢弃」的翻版 —— 上一次
    这类问题空转了两个多月才被发现，而它之所以能藏那么久，就是因为文本看起来完全正常。

    在替代品里重演被替代者的病，是这次迁移最不该有的结局。
    """
    if isinstance(raw, ToolResult):
        return raw
    if raw is None:
        return ToolResult(text="")
    if isinstance(raw, str):
        return ToolResult(text=raw)
    if isinstance(raw, dict):
        # 允许 {"text": …, "update": …} 这种朴素形状
        if "text" in raw or "update" in raw or "state_delta" in raw:
            return ToolResult(
                text=str(raw.get("text", "")),
                state_delta=_state_only(raw.get("update")
                                        or raw.get("state_delta")),
            )

    # langgraph ``Command`` 形状：有 ``update`` 字典（可能还有 ``goto``）。
    # 用鸭子类型而不是 isinstance：二期 langgraph 会消失，而那时仍可能有别的
    # 东西长成这个样子；而且 import 一个即将被删的类只为做类型判断很别扭。
    # langchain returns a ToolMessage when a tool is invoked with a full ToolCall
    # (which the bridge now always does); the text is its content, not its repr.
    if type(raw).__name__ == "ToolMessage" and hasattr(raw, "content"):
        content = raw.content
        if isinstance(content, list):
            content = "\n".join(
                (c.get("text", "") if isinstance(c, dict) else str(c)) for c in content)
        return ToolResult(text=str(content or ""),
                          ok=(getattr(raw, "status", "success") != "error"))
    update = getattr(raw, "update", None)
    if isinstance(update, dict):
        handoff = None
        goto = getattr(raw, "goto", None)
        if isinstance(goto, str) and goto and goto != "__end__":
            handoff = HandoffRequest(target=goto)
        return ToolResult(text=str(raw), state_delta=_state_only(update),
                          handoff=handoff)

    return ToolResult(text=str(raw))


def _state_only(update: Any) -> dict:
    """``update`` 里真正属于 state 的部分（摘掉 ``messages`` 那类传输载体）。"""
    if not isinstance(update, dict):
        return {}
    return {k: v for k, v in update.items() if k not in _NON_STATE_UPDATE_KEYS}


def make_handoff_tool(target: str, description: str = "") -> ToolSpec:
    """造一个交棒工具。

    对比旧世界：``make_handoff`` 要构造 ``Command(goto=…, graph=Command.PARENT,
    update=carried_from(state))``，还得靠 ``InjectedState`` 把产物从子图里捞出来当
    「产物海关」。这里产物走 ``AgentRunResult``，交棒只需要说一句「交给谁、为什么」。
    """
    def _fn(args: dict, ctx) -> ToolResult:
        reason = str((args or {}).get("reason") or "").strip()
        return ToolResult(
            text=f"已交回编排器（{target}）：{reason}" if reason
                 else f"已交回编排器（{target}）。",
            handoff=HandoffRequest(target=target, reason=reason),
        )

    return ToolSpec(
        name=f"handoff_to_{target}",
        description=(description or
                     f"把控制权交回 {target}，并说明你做完了什么、下一步建议做什么。"),
        schema={"type": "object",
                "properties": {"reason": {
                    "type": "string",
                    "description": "交接说明：你做完了什么、结论是什么、建议下一步。"}},
                "required": ["reason"]},
        fn=_fn,
    )


def spec_from_langchain_tool(tool: Any, *, touches_instrument: bool = False) -> ToolSpec:
    """把一个 langchain ``BaseTool`` 包成 :class:`ToolSpec`（一期的桥）。

    ⚠️ 这里**不**传 ``ctx``：langchain 工具不认识它。这正是一期的边界——已有的 156
    个 ``@tool`` 照旧工作，而新写的工具用 ``ToolSpec.fn(args, ctx)`` 拿到显式上下文。
    移植 ``wrap_skill`` 时会走后一条路。
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    try:
        schema = convert_to_openai_tool(tool).get("function", {}).get("parameters", {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("cannot derive schema for %s: %s",
                       getattr(tool, "name", tool), exc)
        schema = {"type": "object", "properties": {}}

    def _fn(args: dict, _ctx, tool_call_id: str = "") -> ToolResult:
        # Invoke with a FULL ToolCall, never bare args: a langchain tool that declares
        # ``InjectedToolCallId`` (meta_tools, handoff, campaign/environment tools …)
        # refuses bare args with "tool must always be invoked with a full model
        # ToolCall". 2026-08-28: every such call failed at dispatch on the first
        # real-provider run of this loop, and the agent correctly reported "运行时
        # 故障" instead of touching the instrument. The loop hands us its id via
        # ``wants_call_id`` (see AgentLoop._run_tool_call).
        import uuid

        call = {"type": "tool_call", "name": getattr(tool, "name", "") or str(tool),
                "args": dict(args or {}), "id": tool_call_id or f"call_{uuid.uuid4().hex[:12]}"}
        return coerce_result(tool.invoke(call))

    return ToolSpec(
        name=getattr(tool, "name", "") or str(tool),
        description=getattr(tool, "description", "") or "",
        schema=schema, fn=_fn, touches_instrument=touches_instrument,
        wants_call_id=True,
    )


__all__ = [
    "ToolSpec",
    "ToolResult",
    "SkillToolResult",
    "HandoffRequest",
    "coerce_result",
    "make_handoff_tool",
    "spec_from_langchain_tool",
]
