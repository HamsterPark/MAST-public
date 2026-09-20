"""A tool's ``state_delta`` (e.g. ``loaded_tool_packs``) must reach bridged middleware
on the NEXT model call, and must survive across runs via ``ctx.extra["state"]``.

2026-08-28, first real-provider run of the v2 loop: ``load_tool_pack`` answered
"已加载" but ``ToolVisibilityMiddleware`` never saw ``state["loaded_tool_packs"]`` —
the bridge built its state from messages alone — so the model only ever saw the core
tools and the provider "repaired" StartScan into StopScan.
"""
from __future__ import annotations


class _RecordingLCMiddleware:
    """Duck-typed langchain-style middleware: records the state it is shown."""

    def __init__(self):
        self.seen: list[dict] = []

    def wrap_model_call(self, request, handler):
        self.seen.append(dict(getattr(request, "state", {}) or {}))
        return handler(request)


def _loop(model, mw):
    from mast.agentruntime.compat import bridge_all
    from mast.agentruntime.loop import AgentLoop
    from mast.agentruntime.tools import ToolResult, ToolSpec

    load = ToolSpec(name="load_pack", description="load",
                    schema={"type": "object", "properties": {}},
                    fn=lambda a, c: ToolResult(text="已加载 scan", state_delta={"loaded_tool_packs": ["scan"]}))
    return AgentLoop(name="ic", model=model, tools=[load],
                     middleware=bridge_all([mw], agent_id="ic"))


def _run(loop, ctx=None):
    from langchain_core.messages import HumanMessage

    gen = loop.run([HumanMessage(content="go")], ctx)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


def test_state_delta_is_visible_on_the_next_model_call():
    from mast.agentruntime.testing import ScriptedModel

    mw = _RecordingLCMiddleware()
    loop = _loop(ScriptedModel([{"tool": "load_pack"}, "done"]), mw)
    result = _run(loop)
    assert result.state_delta.get("loaded_tool_packs") == ["scan"]
    assert len(mw.seen) == 2
    assert "loaded_tool_packs" not in mw.seen[0]
    assert mw.seen[1].get("loaded_tool_packs") == ["scan"]


def test_carried_state_seeds_the_next_run():
    from mast.agentruntime.context import RunContext
    from mast.agentruntime.testing import ScriptedModel

    mw = _RecordingLCMiddleware()
    loop = _loop(ScriptedModel(["ok"]), mw)
    ctx = RunContext(run_id="r2")
    ctx.extra["state"] = {"loaded_tool_packs": ["tip"]}
    _run(loop, ctx)
    assert mw.seen[0].get("loaded_tool_packs") == ["tip"]
