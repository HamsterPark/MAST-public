"""The langchain→ToolSpec bridge must invoke tools with a FULL ToolCall.

A ``@tool`` that declares ``InjectedToolCallId`` (meta_tools, handoff, campaign …)
refuses bare args: "When tool includes an InjectedToolCallId argument, tool must
always be invoked with a full model ToolCall". 2026-08-28: on the first real-provider
run of the v2 loop every such call failed at dispatch. The loop passes its call id
through ``ToolSpec.wants_call_id``; a plain-return tool comes back as a ToolMessage
and must still coerce to text.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool


@tool
def needs_id(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """echo the injected id"""
    return f"x={x} id={tool_call_id}"


@tool
def plain(x: int) -> str:
    """plain tool"""
    return f"plain {x}"


def test_bridge_passes_the_models_call_id_and_coerces_toolmessage():
    from mast.agentruntime.tools import spec_from_langchain_tool

    spec = spec_from_langchain_tool(needs_id)
    assert spec.wants_call_id
    res = spec.fn({"x": 3}, None, tool_call_id="call_abc")
    assert res.ok and "x=3" in res.text and "call_abc" in res.text
    res2 = spec_from_langchain_tool(plain).fn({"x": 5}, None, tool_call_id="call_z")
    assert res2.ok and res2.text.strip() == "plain 5"


def test_loop_hands_the_call_id_to_specs_that_want_it():
    from mast.agentruntime.loop import AgentLoop
    from mast.agentruntime.testing import ScriptedModel
    from mast.agentruntime.tools import spec_from_langchain_tool
    from mast.agentruntime.context import RunContext

    loop = AgentLoop(name="t", model=ScriptedModel([]), tools=[spec_from_langchain_tool(needs_id)])
    result, events = loop._run_one_tool(
        {"name": "needs_id", "args": {"x": 1}, "id": "call_777"}, RunContext(run_id="r"))
    assert result.ok, result.text
    assert "call_777" in result.text
