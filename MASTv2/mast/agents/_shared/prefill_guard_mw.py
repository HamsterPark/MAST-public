"""ClaudePrefillGuardMiddleware — never ask the model to prefill an assistant turn.

Some Claude models (e.g. claude-sonnet-4-6) reject a conversation that ends on an
assistant message: ``This model does not support assistant message prefill. The
conversation must end with a user message.`` In the multi-agent orchestrator, an
agent node's FIRST model call receives the accumulated graph state, which ends in
the supervisor's ``[SUPERVISOR → <agent>]`` AIMessage → a 400 before the agent can
act. (Found 2026-06-08 in the full v2 test: with this guard the F5 routing fix
reaches 6/6 providers; without it sonnet routes but then dies at the agent node.)

Fix: when the outgoing message list ends on an AIMessage, append a minimal
HumanMessage so the model sees a user turn last. It is a no-op for the normal
create_agent loop (which ends on a ToolMessage after tool calls) and harmless for
providers that already tolerate an assistant-last conversation (Kimi/DeepSeek/etc.).
Only the per-call request is modified — graph state is untouched.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from mast.agents._shared.inject import append_new_human

#: 登记表条目 id。
PROMPT_ID = "mw.prefill_guard.continue"

logger = logging.getLogger(__name__)

_CONTINUE = "Proceed with the task above based on the routing decision."


class ClaudePrefillGuardMiddleware(AgentMiddleware):
    """Append a user turn when the outgoing messages end on an AIMessage.

    BOTH hooks are implemented: ``wrap_model_call`` (sync, used by the GUI's
    ``graph.stream()``) and ``awrap_model_call`` (async, used by the CLI's
    ``await graph.ainvoke()`` in pipeline/main.py). LangChain 1.2's base
    ``awrap_model_call`` raises NotImplementedError when only the sync hook is
    defined, so a sync-only middleware would crash every async agent dispatch —
    the async twin below is mandatory, not optional.
    """

    @staticmethod
    def _guarded(request: Any) -> Any:
        """Return a request whose message list never ends on an AIMessage.

        Modifies only the per-call request (via ``override`` when available, else
        a direct assignment), never the persisted graph state. No-op unless the
        last message is an AIMessage (the supervisor-handoff case).
        """
        try:
            msgs = getattr(request, "messages", None)
            if msgs and isinstance(msgs[-1], AIMessage):
                from mast.prompts.registry import resolve as resolve_prompt
                return append_new_human(
                    request, PROMPT_ID,
                    HumanMessage(content=resolve_prompt(PROMPT_ID, _CONTINUE)))
        except Exception as exc:  # noqa: BLE001 — never let the guard break a call
            logger.debug("ClaudePrefillGuard skipped: %s", exc)
        return request

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._guarded(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._guarded(request))


__all__ = ["ClaudePrefillGuardMiddleware"]
