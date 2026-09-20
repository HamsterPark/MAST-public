"""Invoke an agent tool the way the graph invokes it.

Some tools declare an ``Annotated[str, InjectedToolCallId]`` argument so they can
return a ``Command`` that carries a state write alongside the human-readable
summary — that is how a saved document's pointer reaches
``MASTState.draft`` / ``.review`` / ``.literature_report`` and, through the
handoff, the next agent (see
``docs/v2/design/agent_communication_context_redesign.md``).

LangChain then REFUSES a bare-args ``tool.invoke({"title": ...})``: a tool with
an injected call id must be handed the full ToolCall envelope, exactly as
``ToolNode`` does at runtime. That refusal is a feature — the 2026-05-30 incident
(every skill's state update silently discarded) survived a full green test suite
precisely because the tests called tools through a path the runtime never uses.

So: call tools through this helper rather than reaching for ``.func`` to dodge
the envelope. ``.func`` skips argument validation too, which is the other half of
what the runtime actually does.
"""
from __future__ import annotations

from typing import Any


def tool_call(t: Any, args: dict | None = None, *, call_id: str = "test-tool-call") -> dict:
    """The ToolCall envelope ``ToolNode`` would build for ``t`` with ``args``."""
    return {
        "name": getattr(t, "name", str(t)),
        "args": dict(args or {}),
        "id": call_id,
        "type": "tool_call",
    }


def tool_text(result: Any) -> str:
    """The human-readable text of a tool result, whatever shape it came back in.

    Invoked through a ToolCall envelope, one tool can return either shape:

      * a ``Command`` (it also wrote state) — comes back as-is, and its ``str()``
        is the summary;
      * a plain ``str`` (an early rejection: empty body, bad verdict) — LangChain
        wraps it in a ``ToolMessage`` before the caller sees it.

    Both are correct; asserting on the text should not have to care which path a
    given input took.
    """
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(b) for b in content)
    return str(result)


__all__ = ["tool_call", "tool_text"]
