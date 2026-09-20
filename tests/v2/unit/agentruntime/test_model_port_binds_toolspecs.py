"""``LangChainModelPort`` must hand langchain OpenAI-style tool dicts, not ToolSpecs.

Before 2026-08-28 it passed ``ToolSpec`` objects to ``bind_tools``; langchain raised
"Unsupported function", the port logged "calling bare" and the model ran with NO
tools — silently, on every real provider.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage


class _FakeChat:
    def __init__(self):
        self.bound = None

    def bind_tools(self, tools, **kw):
        # what langchain would do with a ToolSpec
        for t in tools:
            if not isinstance(t, dict) and not hasattr(t, "args_schema"):
                raise ValueError(f"Unsupported function {t!r}")
        self.bound = tools
        return self

    def bind(self, **kw):
        return self

    def invoke(self, msgs):
        return AIMessage(content="ok")


def test_toolspecs_are_bound_as_openai_tool_dicts():
    from mast.agentruntime.model import LangChainModelPort, ModelRequest
    from mast.agentruntime.tools import ToolSpec

    fake = _FakeChat()
    port = LangChainModelPort(fake)
    spec = ToolSpec(name="SetBias", description="set bias",
                    schema={"type": "object", "properties": {"bias_v": {"type": "number"}}},
                    fn=lambda a, c: None)
    resp = port.invoke(ModelRequest(system_prompt="s", messages=[HumanMessage(content="hi")],
                                    tools=[spec]))
    assert resp.text == "ok"
    assert fake.bound is not None, "bind_tools was not reached — the model ran bare"
    assert fake.bound[0]["type"] == "function"
    assert fake.bound[0]["function"]["name"] == "SetBias"
    assert fake.bound[0]["function"]["parameters"]["properties"]["bias_v"]["type"] == "number"
