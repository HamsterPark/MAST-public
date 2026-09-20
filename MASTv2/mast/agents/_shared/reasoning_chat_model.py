"""ChatOpenAI subclass that preserves `reasoning_content` across turns.

Background
----------
Moonshot Kimi K2.6 and DeepSeek V4 Pro are "thinking" / reasoning models.
After producing tool_calls in turn 1 the server keeps state (reasoning
trace) that must be echoed back on turn 2 via the assistant message's
``reasoning_content`` field. If you don't, turn 2 fails with::

    400: thinking is enabled but reasoning_content is missing
    in assistant tool call message at index N

LangChain ``langchain-openai 1.2.x``'s ``_convert_dict_to_message`` does
not extract ``reasoning_content`` from the API response, and
``_convert_message_to_dict`` does not write it back when serializing for
the next turn. The standard ``enable_thinking=False`` workaround the
Moonshot docs mention only suppresses *output* of reasoning trace — the
server-side thinking pipeline is still engaged for these models and
still requires the round-trip.

This module provides ``ReasoningPreservingChatOpenAI``, a thin subclass
that:

1. Patches ``_create_chat_result`` to extract ``reasoning_content`` from
   the raw response and store it on ``AIMessage.additional_kwargs``.
2. Patches ``_get_request_payload`` to copy ``additional_kwargs.reasoning_content``
   back into the assistant message dict before sending it to the API.

This lets multi-turn tool-calling flows (LangGraph create_agent style)
work transparently with Kimi K2.6 / DeepSeek V4 Pro.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

logger = logging.getLogger(__name__)


def _extract_reasoning(raw_msg: Any) -> str | dict | list | None:
    """Find a `reasoning_content`-shaped field on an OpenAI choice.message."""
    if raw_msg is None:
        return None
    # openai SDK returns a pydantic model; reasoning_content may be a top-level
    # attribute, an attribute on `model_extra`, or a key inside the raw dict.
    val = getattr(raw_msg, "reasoning_content", None)
    if val:
        return val
    extras = getattr(raw_msg, "model_extra", None)
    if isinstance(extras, dict):
        v = extras.get("reasoning_content")
        if v:
            return v
    # Also try the dict-flavoured response path
    if isinstance(raw_msg, dict):
        return raw_msg.get("reasoning_content")
    return None


def _coerce_response_to_dict(response: Any) -> dict | None:
    """Best-effort coerce an openai response object to dict for inspection."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    return None


def make_reasoning_preserving_chat_openai_class():
    """Lazily build the subclass — avoids importing langchain_openai at module load."""
    from langchain_openai import ChatOpenAI

    class ReasoningPreservingChatOpenAI(ChatOpenAI):
        """Thin wrapper that round-trips `reasoning_content` between turns."""

        def _create_chat_result(
            self,
            response: dict | Any,
            generation_info: dict | None = None,
        ) -> ChatResult:
            result = super()._create_chat_result(response, generation_info)
            # Walk the response's choices and attach reasoning_content
            # to the corresponding generation's message.additional_kwargs.
            response_dict = _coerce_response_to_dict(response)
            if response_dict and "choices" in response_dict:
                choices = response_dict["choices"]
            else:
                # Fallback: pydantic-style with .choices attribute
                choices = getattr(response, "choices", None) or []
            for i, gen in enumerate(result.generations):
                if i >= len(choices):
                    break
                ch = choices[i]
                raw_msg = ch.get("message") if isinstance(ch, dict) else getattr(ch, "message", None)
                rc = _extract_reasoning(raw_msg)
                if rc:
                    msg = gen.message
                    if isinstance(msg, AIMessage):
                        msg.additional_kwargs.setdefault("reasoning_content", rc)
            return result

        def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> dict:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            # Re-attach reasoning_content fields onto the serialized
            # assistant messages so the server's thinking machinery is
            # happy. We match by position: messages from
            # _convert_input(input_).to_messages() and payload["messages"]
            # should be the same length and same order.
            try:
                src_messages = self._convert_input(input_).to_messages()
            except Exception:
                return payload
            api_messages = payload.get("messages")
            if not isinstance(api_messages, list) or len(api_messages) != len(src_messages):
                return payload
            for src, api_msg in zip(src_messages, api_messages):
                if not isinstance(src, AIMessage):
                    continue
                if api_msg.get("role") != "assistant":
                    continue
                rc = src.additional_kwargs.get("reasoning_content")
                if rc and "reasoning_content" not in api_msg:
                    api_msg["reasoning_content"] = rc
            return payload

    return ReasoningPreservingChatOpenAI


__all__ = ["make_reasoning_preserving_chat_openai_class"]
