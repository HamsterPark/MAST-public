"""ToolPairGuardMiddleware — never send a provider an unmatched tool message.

Some providers 400 the moment the outgoing message list contains a tool message
whose ``tool_call_id`` is not answered by a preceding assistant ``tool_calls``
entry (Kimi/OpenAI-compat: ``Invalid request: tool_call_id is not found``;
MiniMax: ``tool result's tool id … not found``), or an assistant ``tool_calls``
entry with no following tool result.

Where the orphans come from (the group orchestrator, verified 2026-07-20):
  An agent hands control back by CALLING the ``handoff_to_supervisor`` tool,
  which returns ``Command(goto="supervisor", graph=Command.PARENT,
  update={messages:[ToolMessage(...)]})``. ``graph=Command.PARENT`` SHORT-CIRCUITS
  out of the agent subgraph, so ONLY ``command.update`` reaches the parent
  ``messages`` channel — the ``AIMessage`` that CONTAINED the handoff tool_call
  stays behind in the subgraph's own namespace and never propagates up. The
  parent transcript is then left with a bare handoff ``ToolMessage`` (an orphan
  tool RESULT). On the NEXT hop the supervisor seeds the next agent (or the same
  agent re-entering) with that parent history, and the agent's model call ships
  the orphan straight to the provider → 400. It recurs because every inter-agent
  hop after the first re-seeds the growing orphan-laden history.

The fix is provider-portable and lives at the LAST boundary before the model:
this middleware rewrites ONLY the per-call request (never the persisted graph
state — mirrors ``ClaudePrefillGuardMiddleware``), so it also rescues the
orphans ALREADY sitting in existing checkpoints without a migration.

Two orphan shapes are repaired:
  1. Orphan tool RESULT — a ``ToolMessage`` whose ``tool_call_id`` is answered by
     no assistant ``tool_calls`` in the request. CONVERTED to a plain
     ``AIMessage`` carrying the same text (the handoff reason — "扫描已完成" — is
     real context the next agent should keep) so the pairing constraint is
     satisfied without losing information. A content-less orphan is dropped.
  2. Orphan tool CALL — an assistant ``tool_calls`` entry with no answering
     ``ToolMessage`` later in the request (e.g. a run aborted mid-tool). A
     synthetic ``ToolMessage`` is inserted right after the assistant message so
     the call is satisfied. We repair rather than surgically edit the assistant
     message, whose content-block shape is provider-specific.

Fail-safe: any error leaves the request untouched (a guard must never break a
call). No-op when the request is already well-formed.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger(__name__)

_SYNTH_TOOL_RESULT = "[工具结果在上下文组装时不可用]"


def _tool_call_ids(msg: Any) -> list[str]:
    out: list[str] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        if tid:
            out.append(tid)
    return out


def repair_tool_pairs(messages: list) -> "list | None":
    """Return a provider-safe copy of ``messages`` (every tool message matched,
    every tool_call answered), or ``None`` when nothing needed changing.

    Pure function (no I/O) so it is unit-testable in isolation.
    """
    if not messages:
        return None

    # Pass 1 — every tool_call id an assistant message PROVIDES, and the message
    # object that provides each (so a synthetic answer can be inserted right after
    # it, preserving provider ordering).
    provider_of: dict[str, Any] = {}
    for m in messages:
        for tid in _tool_call_ids(m):
            provider_of.setdefault(tid, m)

    # Pass 2 — drop/convert orphan tool RESULTS; record which provided ids get a
    # real answer so pass 3 can synthesize the rest.
    answered: set[str] = set()
    stage: list = []
    changed = False
    for m in messages:
        if isinstance(m, ToolMessage):
            tid = getattr(m, "tool_call_id", None)
            if tid and tid in provider_of:
                answered.add(tid)
                stage.append(m)
            else:
                # Orphan tool result. Keep its text as assistant narration (the
                # handoff reason is real context) rather than lose it; drop it
                # entirely only when there is nothing to keep.
                changed = True
                text = getattr(m, "content", "")
                if isinstance(text, list):
                    text = " ".join(str(x) for x in text)
                text = str(text).strip()
                if text:
                    stage.append(AIMessage(content=text))
        else:
            stage.append(m)

    # Pass 3 — satisfy orphan tool CALLS: for a provided id never answered, insert
    # a synthetic ToolMessage immediately after the assistant message that made it.
    unanswered_by_provider: dict[int, list[str]] = {}
    for tid, prov in provider_of.items():
        if tid not in answered:
            unanswered_by_provider.setdefault(id(prov), []).append(tid)
    if unanswered_by_provider:
        changed = True
        from mast.prompts.registry import resolve as resolve_prompt
        synth = resolve_prompt("mw.tool_pair_guard.synth", _SYNTH_TOOL_RESULT)
        out: list = []
        for m in stage:
            out.append(m)
            extra = unanswered_by_provider.get(id(m))
            if extra:
                for tid in extra:
                    out.append(ToolMessage(content=synth, tool_call_id=tid))
        stage = out

    return stage if changed else None


class ToolPairGuardMiddleware(AgentMiddleware):
    """Repair unmatched tool messages in the OUTGOING request (see module doc).

    BOTH hooks are implemented: ``wrap_model_call`` (sync — the GUI's
    ``graph.stream()``) and ``awrap_model_call`` (async — the CLI's
    ``await graph.ainvoke()``). LangChain's base ``awrap_model_call`` raises when
    only the sync hook is defined, so a sync-only middleware would crash every
    async dispatch — the async twin is mandatory (same lesson as prefill_guard).
    """

    @staticmethod
    def _guarded(request: Any) -> Any:
        try:
            msgs = getattr(request, "messages", None)
            if not msgs:
                return request
            repaired = repair_tool_pairs(list(msgs))
            if repaired is None:
                return request
            override = getattr(request, "override", None)
            if callable(override):
                return override(messages=repaired)
            request.messages = repaired
        except Exception as exc:  # noqa: BLE001 — a guard must never break a call
            logger.debug("ToolPairGuard skipped: %s", exc)
        return request

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._guarded(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._guarded(request))


__all__ = ["ToolPairGuardMiddleware", "repair_tool_pairs"]
