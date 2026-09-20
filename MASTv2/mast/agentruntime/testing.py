"""测试替身 —— 脚本化的 :class:`~mast.agentruntime.model.ChatModelPort`。

为什么它值得是生产包里的一个模块，而不是各测试文件各写一份
----------------------------------------------------------
今天大约 24 个测试文件直接 import langchain 的 ``GenericFakeChatModel``，而且几乎每个
都要为它补一个 ``bind_tools``（基类直接 raise）。二期把 langchain 的 ChatModel 层换掉时，
那 24 处要一起改——**收敛到一处，那次改动就只有一处**。

更重要的是替身的**层次**变了。``GenericFakeChatModel`` 替的是一个 ChatModel：一个我们
不拥有、字段由上游决定、还要额外补方法才能用的类型。:class:`ScriptedModel` 替的是
**我们自己的端口**（只有 ``invoke`` / ``stream`` 两个方法）——脚本化一个窄端口比脚本化
一个宽类简单得多，这本身就是端口窄的好处。

替身纪律
--------
这个仓库栽过两次同型的跟头：**替身太顺**让负例恒绿（``MagicMock`` 的 ``len()`` 是 0、
``startswith()`` 恒真），**替身多一层**让失效路径根本到不了。所以这里：

* 脚本演完了就说「演完了」，**不是**无限重复最后一条——一个永远有话说的模型会让
  「循环停不下来」这类缺陷测不出来；
* ``invoke`` 记下**每一次**收到的请求（``requests``），断言可以打在「模型到底看到了
  什么」上，而不是只打在最终结果上；
* 不实现端口之外的任何方法。测试如果需要 ``bind_tools``，说明被测代码在越过端口。
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator

from mast.agentruntime.model import ModelRequest, ModelResponse

#: 脚本演完之后的回答。刻意是一句**看得出来是替身**的话：一个测试如果在这句话上
#: 通过了，那它测的不是它以为的东西。
EXHAUSTED_TEXT = "（脚本已演完）"


class ScriptedModel:
    """按脚本作答的 ``ChatModelPort``。

    脚本项两种形状::

        "一段文本"                                    → 不带工具调用的答复
        {"tool": "Scan", "args": {...}, "id": "t1"}   → 一次工具调用
        [{"tool": ...}, {"tool": ...}]                → 同一轮里的多次工具调用

    用法::

        model = ScriptedModel([{"tool": "Scan"}, "扫好了。"])
        loop = AgentLoop(name="ic", model=model, tools=[...])
    """

    def __init__(self, script: Iterable[Any] = ()):
        self._script: list = list(script)
        #: 每一次 ``invoke`` 收到的请求（副本）。断言「模型看到了什么」用它。
        self.requests: list[ModelRequest] = []

    # ── 端口 ───────────────────────────────────────────────────────────
    def invoke(self, request: ModelRequest) -> ModelResponse:
        from langchain_core.messages import AIMessage

        self.requests.append(request.copy())
        if not self._script:
            return ModelResponse(message=AIMessage(content=EXHAUSTED_TEXT),
                                 text=EXHAUSTED_TEXT)

        item = self._script.pop(0)
        if isinstance(item, str):
            return ModelResponse(message=AIMessage(content=item), text=item)

        calls = item if isinstance(item, list) else [item]
        tool_calls = [{"name": c["tool"], "args": c.get("args", {}),
                       "id": c.get("id", f"tc-{i}"), "type": "tool_call"}
                      for i, c in enumerate(calls)]
        return ModelResponse(
            message=AIMessage(content="", tool_calls=tool_calls),
            tool_calls=tool_calls, text="")

    def stream(self, request: ModelRequest) -> Iterator[str]:
        text = self.invoke(request).text
        if text:
            yield text

    # ── 便利断言 ───────────────────────────────────────────────────────
    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def exhausted(self) -> bool:
        return not self._script

    def last_system_prompt(self) -> str:
        return self.requests[-1].system_prompt if self.requests else ""

    def last_tool_names(self) -> list[str]:
        if not self.requests:
            return []
        return [getattr(t, "name", "") for t in self.requests[-1].tools]


class ScriptedRouter:
    """路由用的替身 —— 只实现 ``llm_route`` 会用到的两个方法。

    与 :class:`ScriptedModel` 分开，因为路由走的是另一条路（``with_structured_output``
    的 function_calling 那一档 + 纯文本 JSON 降级），而那条路上每一家 provider 的挂法
    都不一样。想测「结构化输出挂了会怎样」就用 ``structured_fails=True``。
    """

    def __init__(self, decisions: Iterable[dict] = (), *,
                 structured_fails: bool = False, text_answer: str = ""):
        self._decisions: list[dict] = list(decisions)
        self._structured_fails = structured_fails
        self._text_answer = text_answer
        #: 文本降级那一轮真正收到的消息（用来验 json_hint / 结尾是不是 user 轮）。
        self.text_messages: list | None = None

    def _next(self) -> dict:
        return (self._decisions.pop(0) if self._decisions
                else {"next_agent": "__end__", "reason": "脚本已演完"})

    def with_structured_output(self, schema, method=None):
        if self._structured_fails:
            raise RuntimeError("This response_format type is unavailable now")
        outer = self

        class _Bound:
            def invoke(self, messages):
                return outer._next()
        return _Bound()

    def invoke(self, messages):
        self.text_messages = list(messages)
        import json

        payload = self._text_answer or json.dumps(self._next(), ensure_ascii=False)

        class _Resp:
            content = payload
        return _Resp()


__all__ = ["ScriptedModel", "ScriptedRouter", "EXHAUSTED_TEXT"]
