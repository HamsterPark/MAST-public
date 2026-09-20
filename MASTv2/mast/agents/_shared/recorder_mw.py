"""RecorderMiddleware — capture each agent turn (reasoning + tool_calls + usage).

RFC §3 (思考链 / agent_turn). An ``after_model`` hook attached to every one of the
6 sub-agents. After the model produces its AIMessage, this reads that message and
emits one ``agent_turn`` trajectory step via an INJECTED ``recorder`` callback — it
NEVER imports ``mast.webui`` / ``mast.logging`` (the no-cross-boundary invariant): the
GUI closes a sink over ``recorder`` at build() time, exactly like ``post_hook`` /
``safety_recorder``.

Cross-provider reasoning extraction (RFC risk #7 — ``reasoning_content`` is NOT
portable):
  * OpenAI-compatible (Kimi K2.x / DeepSeek V4 / Qwen3.x / GLM): the round-tripped
    reasoning lands on ``AIMessage.additional_kwargs["reasoning_content"]`` (a str),
    courtesy of ReasoningPreservingChatOpenAI.
  * Anthropic-compatible (Claude adaptive thinking / MiniMax): extended-thinking
    arrives as ``type == "thinking"`` blocks INSIDE ``AIMessage.content`` (a list),
    alongside the ``type == "text"`` answer blocks.
A provider that surfaces no reasoning records ``reasoning=None`` rather than
crashing — capture is best-effort and FAIL-SAFE: any extraction error is swallowed
so the agent run is never affected. The capture is read-only; ``after_model``
returns ``None`` (no state delta), so it can never alter the conversation.

Sanitization / no-tensor: the payload holds only JSON primitives; the StepRepo
write layer (``_json_safe``) is the final no-tensor gate, and gives the
``agent_turn`` output a wider per-string cap so multi-KB reasoning survives.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def _extract_reasoning(msg: AIMessage) -> Any | None:
    """Return the turn's reasoning trace across providers, or None.

    Tries the OpenAI-compatible ``additional_kwargs.reasoning_content`` first, then
    the Anthropic-compatible ``type == "thinking"`` content blocks. Encrypted
    ``redacted_thinking`` blocks carry no readable text and are skipped."""
    ak = getattr(msg, "additional_kwargs", None) or {}
    rc = ak.get("reasoning_content")
    if isinstance(rc, str):
        return rc if rc.strip() else None
    if isinstance(rc, (dict, list)) and rc:
        return rc  # rare non-string shape — keep it, StepRepo sanitizes

    content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                t = block.get("thinking")
                if isinstance(t, str) and t:
                    parts.append(t)
        if parts:
            return "\n".join(parts)
    return None


def _extract_text(msg: AIMessage) -> str:
    """The visible answer text (str content, or the ``text`` blocks of a list)."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return "" if content is None else str(content)


def _extract_tool_calls(msg: AIMessage) -> list[dict]:
    out: list[dict] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        if isinstance(tc, dict):
            out.append({"name": tc.get("name"), "args": tc.get("args"),
                        "id": tc.get("id")})
        else:
            out.append({"name": getattr(tc, "name", None),
                        "args": getattr(tc, "args", None),
                        "id": getattr(tc, "id", None)})
    return out


def _extract_usage(msg: AIMessage) -> dict | None:
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        return {"input_tokens": um.get("input_tokens"),
                "output_tokens": um.get("output_tokens"),
                "total_tokens": um.get("total_tokens")}
    return None


def _extract_meta(msg: AIMessage) -> tuple[Any | None, Any | None]:
    rm = getattr(msg, "response_metadata", None) or {}
    model_id = rm.get("model_name") or rm.get("model")
    finish = rm.get("finish_reason") or rm.get("stop_reason")
    return model_id, finish


def build_turn_payload(agent_id: str, msg: AIMessage) -> dict:
    """Flatten one AIMessage into the agent_turn step payload (JSON-primitives)."""
    model_id, finish = _extract_meta(msg)
    return {
        "agent_id": agent_id,
        "model_id": model_id,
        "reasoning": _extract_reasoning(msg),
        "content": _extract_text(msg),
        "tool_calls": _extract_tool_calls(msg),
        "usage": _extract_usage(msg),
        "finish_reason": finish,
        "msg_id": getattr(msg, "id", None),
    }


class RecorderMiddleware(AgentMiddleware):
    """Emit one ``agent_turn`` step per model call via an injected recorder.

    Both the sync (``after_model``) and async (``aafter_model``) hooks are wired:
    the GUI drives agents with ``graph.stream()`` (sync) while the CLI uses
    ``await graph.ainvoke()`` (async). The base class leaves both as no-ops, so a
    sync-only middleware would simply never fire on the async path — both are
    implemented to capture on either.
    """

    def __init__(self, *, agent_id: str, recorder: Callable[[dict], None] | None):
        super().__init__()
        self._agent_id = agent_id
        self._recorder = recorder

    def _capture(self, state: Any) -> None:
        if self._recorder is None:
            return
        try:
            msgs = (state.get("messages") if isinstance(state, dict)
                    else getattr(state, "messages", None))
            if not msgs:
                return
            msg = msgs[-1]
            if not isinstance(msg, AIMessage):
                return
            self._recorder(build_turn_payload(self._agent_id, msg))
        except Exception:  # noqa: BLE001 — logging must never break the agent run
            logger.debug("RecorderMiddleware capture failed (swallowed)", exc_info=True)

    def after_model(self, state: Any, runtime: Any = None) -> None:
        self._capture(state)
        return None

    async def aafter_model(self, state: Any, runtime: Any = None) -> None:
        self._capture(state)
        return None


__all__ = ["RecorderMiddleware", "build_turn_payload"]
