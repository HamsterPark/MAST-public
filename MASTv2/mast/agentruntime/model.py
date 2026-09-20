"""模型端口 —— 循环与 provider 之间的那一层。

一期：包一层 langchain 的 ``BaseChatModel``（``models.py`` 的六家工厂原样复用，含
``reasoning_content`` 补丁与计费 callback）。二期：换成 ``mast/llm/`` 的自研 client，
**只换这个文件里的实现，循环与中间件一个字不动**。

为什么值得先立这道端口
----------------------
今天中间件直接改的是 langchain 的 ``ModelRequest``；那是一个我们不拥有的类型，它的
字段增删由上游决定。把请求换成我们自己的 dataclass，两件事立刻成立：

* 二期换 provider 层时，21 个中间件不需要跟着改签名；
* ``with_structured_output`` 那个坑（六家 provider 五种挂法）可以在**这一层**用能力
  表分层降级解决，而不是散在调用点。

``tools`` 是请求的一个普通字段
------------------------------
按需工具加载（``tool_visibility_mw``）今天要与「工具表建图时冻结」博弈；在这里它只是
一个中间件改一下 ``request.tools``。这不是新功能，是同一件事换个更自然的位置。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ModelRequest:
    """一次模型调用的全部输入。中间件改的就是它。

    可变 dataclass：中间件的常规动作就是往 ``system_prompt`` 后面追加一段、或者
    过滤 ``messages`` / ``tools``。让它们改一个副本比让每个中间件学会构造完整请求
    更不容易出错。
    """

    system_prompt: str = ""
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    #: provider 级参数（temperature / thinking / max_tokens…）。刻意是自由 dict：
    #: 各家能力不同，把它收敛成固定字段会在第一个新 provider 上破功。
    settings: dict = field(default_factory=dict)
    #: 给中间件放临时结论的地方（本回合是否已注入过记忆之类）。不进 provider。
    scratch: dict = field(default_factory=dict)
    #: 本次 run 到目前为止累计的 state（工具结果的 ``state_delta`` 合并而成，例如
    #: ``loaded_tool_packs``）。桥接层把它并进 langgraph 风格的 ``state`` 给老中间件
    #: 读 —— 2026-08-28 之前这里什么都没有，于是 ``load_tool_pack`` 在 v2 循环里
    #: 「加载成功」却永远不生效（可见性中间件读不到那个键）。不进 provider。
    state: dict = field(default_factory=dict)

    def copy(self) -> "ModelRequest":
        return ModelRequest(
            system_prompt=self.system_prompt, messages=list(self.messages),
            tools=list(self.tools), settings=dict(self.settings),
            scratch=dict(self.scratch), state=dict(self.state))


@dataclass
class ModelResponse:
    """一次模型调用的结果。

    ``message`` 是 provider 返回的原始消息对象（一期是 langchain 的 ``AIMessage``）；
    循环只读它的 ``tool_calls`` 与文本，别的都不碰——这样二期换类型时循环不用动。
    """

    message: Any = None
    #: 便于中间件与循环判断，避免每处各写一遍取值逻辑。
    tool_calls: list = field(default_factory=list)
    text: str = ""
    usage: dict = field(default_factory=dict)


@runtime_checkable
class ChatModelPort(Protocol):
    """循环对模型的全部要求。**只有两个方法**——刻意的。

    端口越窄，二期换实现时要重踩的坑越少。``with_structured_output`` 不在里面：
    它不可移植（六家五种挂法），结构化输出走 ``agents/_shared/llm_route.py`` 的
    分层降级，那一层是 provider 中立的。
    """

    def invoke(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> Iterator[str]:
        """逐 token 文本增量。只有语音链消费。"""
        ...


class LangChainModelPort:
    """一期实现：把 ``BaseChatModel`` 包成 :class:`ChatModelPort`。

    ``bind_tools`` 在**每次调用时**做，而不是建图时冻结一次——这正是按需工具加载
    需要的形状，而且它本来就便宜（``bind`` 只是记一下 kwargs）。
    """

    def __init__(self, chat_model: Any):
        self._model = chat_model

    # ── 内部：把我们的请求翻译成 langchain 的调用 ──────────────────────
    def _bound(self, request: ModelRequest):
        model = self._model
        if request.tools:
            # ToolSpec is OUR type; langchain's bind_tools wants dicts / BaseTools /
            # pydantic models. Passing ToolSpecs raised "Unsupported function …",
            # the except below logged "calling bare", and every v2 run against a real
            # provider was a model with NO tools (2026-08-28, found by the STM-Bench
            # driver — the first real-provider run of this port).
            bound_tools = [
                t if isinstance(t, dict) or not hasattr(t, "schema") else {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": getattr(t, "description", "") or "",
                        "parameters": (getattr(t, "schema", None)
                                       or {"type": "object", "properties": {}}),
                    },
                }
                for t in request.tools
            ]
            try:
                model = model.bind_tools(bound_tools)
            except NotImplementedError:
                # 测试替身常常不实现它；真实 provider 都实现。
                logger.debug("model does not implement bind_tools; calling bare")
            except Exception as exc:  # noqa: BLE001
                logger.warning("bind_tools failed (%s); calling bare", exc)
        if request.settings:
            try:
                model = model.bind(**request.settings)
            except Exception as exc:  # noqa: BLE001
                logger.warning("model.bind(%s) failed: %s",
                               sorted(request.settings), exc)
        return model

    def _payload(self, request: ModelRequest) -> list:
        from langchain_core.messages import SystemMessage

        msgs = list(request.messages)
        if request.system_prompt:
            msgs = [SystemMessage(content=request.system_prompt), *msgs]
        return msgs

    # ── 端口 ───────────────────────────────────────────────────────────
    def invoke(self, request: ModelRequest) -> ModelResponse:
        msg = self._bound(request).invoke(self._payload(request))
        return ModelResponse(
            message=msg,
            tool_calls=list(getattr(msg, "tool_calls", None) or []),
            text=_text_of(msg),
            usage=dict(getattr(msg, "usage_metadata", None) or {}),
        )

    def stream(self, request: ModelRequest) -> Iterator[str]:
        """★ 生成器：yield 文本增量，**return** 组装好的 :class:`ModelResponse`。

        2026-08-27 改形。前一版只 yield 文本、把 chunk 丢掉 —— 于是**工具调用没了**，
        流式路径只能拿到一段话，agent 在语音里就再也不会调工具。而它当时零消费者，
        所以这件事一直没暴露。

        为什么不是「先 stream 拿文本、再 invoke 一次拿结构」：那要**付两次调用**，
        而且两次的内容不保证一致（温度不为 0 时几乎必然不一致，于是念出来的话和
        实际做的事对不上）。langchain 的 chunk 支持 ``+`` 累加，累出来的消息带着
        ``tool_calls`` —— 一次调用给两样，与 langgraph 的
        ``stream_mode=["updates","messages"]`` 拿到的是同一批东西。

        用法（与 ``AgentLoop.run`` 同一个惯用法）::

            gen = port.stream(req)
            while True:
                try:
                    delta = next(gen)
                except StopIteration as stop:
                    response = stop.value
                    break
        """
        acc = None
        for chunk in self._bound(request).stream(self._payload(request)):
            acc = chunk if acc is None else acc + chunk
            piece = _text_of(chunk)
            if piece:
                yield piece
        if acc is None:
            # 一个 chunk 都没有 —— 不是错误（模型可以什么都不说），但也不能返回
            # None 让调用方去猜。给一个空响应，它与「模型说了空话」是同一件事。
            return ModelResponse(message=None, tool_calls=[], text="", usage={})
        return ModelResponse(
            message=acc,
            tool_calls=list(getattr(acc, "tool_calls", None) or []),
            text=_text_of(acc),
            usage=dict(getattr(acc, "usage_metadata", None) or {}),
        )


def _text_of(msg: Any) -> str:
    """从一条消息里取出**人要读的那部分文本**。

    内容可能是字符串，也可能是块列表。块列表里要跳过 ``thinking``：那是模型的
    草稿纸，混进正文会让用户读到一段本不该给他看的自言自语（语音链尤其明显——
    它会被念出来）。
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(str(block.get("text", "")))
            elif isinstance(block, str):
                out.append(block)
        return "".join(out)
    return str(content or "")


__all__ = ["ModelRequest", "ModelResponse", "ChatModelPort", "LangChainModelPort"]
