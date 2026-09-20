"""RequestReplyReadbackMiddleware — auto-inject operator answers to 心愿单 requests.

⑦ (2026-07 conversation analysis): ``request_user_action`` posts a request to the
board and the agent is expected to poll ``check_my_requests`` later — but a BLOCKED
agent that stopped has no turn on which to poll, and the operator's answer sat on
the board unread (the agent showed 「无记忆 / 无匹配」, forcing the operator to paste
the path into chat). Telling the agent in the prompt to "remember to poll" is the
same failure mode as telling it to "remember to check the active controller": the
LLM forgets. The fix is the framework HANDING the certain information over, not the
agent rediscovering it — so this middleware injects any answered-but-undelivered
request into the agent's system message on its NEXT turn, automatically.

Mirrors :class:`MemoryRecallMiddleware`:

  * implements sync ``wrap_model_call`` + async ``awrap_model_call``;
  * appends a block to ``request.system_message`` so the agent sees the answer
    WITHOUT calling a tool;
  * **per-USER-turn cache** keyed by the latest human message, so the injected
    block is STABLE across the sub-steps of one turn (a changing system message
    each sub-step would bust the Anthropic prompt cache). The board is consulted —
    and the answers marked delivered — only ONCE per turn (on a key change), so
    each answer is handed over exactly once: the next turn recomputes, finds them
    delivered, and injects nothing.

Attached via ``_chat_agent_middleware()`` — the SINGLE list shared by BOTH the
群聊 orchestrator agents and the 私聊 (main chat) agents, so one wiring point
covers both paths (the route-level supervisor injection cannot reach 私聊).

Best-effort: any failure leaves the request untouched. agent_id defaults to ""
(match every agent): in production all agents post under the shared id "agent",
and over-surfacing an answer is far better than a blocked agent never seeing it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage

from mast.agents._shared.inject import append_human_block, append_system_block

#: 登记表条目 id。
PROMPT_ID = "mw.request_readback"

logger = logging.getLogger(__name__)


def _latest_human_text(messages) -> str | None:
    for msg in reversed(list(messages or [])):
        if isinstance(msg, HumanMessage):
            c = msg.content
            if isinstance(c, str):
                return c.strip() or None
            if isinstance(c, list):
                parts = [b.get("text", "") for b in c if isinstance(b, dict)]
                t = "".join(parts).strip()
                return t or None
    return None


class RequestReplyReadbackMiddleware(AgentMiddleware):
    """Append operator answers the agent has not been told about yet.

    Covers BOTH ask-the-operator channels, because they have the same failure
    mode and the agent should not have to know which board its answer landed on:

      * 心愿单 — answered requests the agent has not read back;
      * 取文请求板 — papers that arrived without auto-resume managing to announce
        them (a request raised in a background run has no conversation to return
        to, so nothing ever told anyone it had been satisfied).

    The fetch half is gated on ``fetch_for_agent`` so a paper arriving does not
    interrupt an instrument conversation with literature news.
    """

    def __init__(self, *, agent_id: str = "", within_hours: float | None = 72.0,
                 fetch_for_agent: str = ""):
        super().__init__()
        self._agent_id = agent_id
        self._within_hours = within_hours
        self._fetch_for_agent = fetch_for_agent
        # (latest_human_text) -> the block computed for that turn. Sentinel object
        # so the first call (no human message yet) is distinct from a cached "".
        self._last_key: object = object()
        self._last_block: str = ""

    def _wishlist_block(self) -> str:
        try:
            from mast.agents._shared.resume_context import build_request_reply_block
            from mast.wishlist import get_board

            board = get_board()
            block, ids = build_request_reply_block(
                board, agent_id=self._agent_id, within_hours=self._within_hours
            )
            if block and ids:
                try:
                    board.mark_delivered(ids)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("mark_delivered failed: %s", exc)
            return block or ""
        except Exception as exc:  # noqa: BLE001 — readback never breaks a turn
            logger.debug("wishlist readback build failed: %s", exc)
            return ""

    def _fetch_block(self) -> str:
        if not self._fetch_for_agent:
            return ""
        try:
            from mast.agents._shared.resume_context import build_fetch_arrival_block
            from mast.knowledge.fetch_board import get_board

            board = get_board()
            block, ids = build_fetch_arrival_block(
                board, requested_by=self._fetch_for_agent,
                within_hours=self._within_hours,
            )
            if block and ids:
                try:
                    board.mark_announced(ids)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("mark_announced failed: %s", exc)
            return block or ""
        except Exception as exc:  # noqa: BLE001 — readback never breaks a turn
            logger.debug("fetch readback build failed: %s", exc)
            return ""

    def _build_and_consume(self) -> str:
        """Collect both channels' unread answers and mark them read.

        The mark happens HERE (once per turn, on a cache miss) so the same answer
        is never injected on a later turn.
        """
        blocks = [b for b in (self._wishlist_block(), self._fetch_block()) if b]
        return "\n\n".join(blocks)

    def _block_for(self, request) -> str:
        key = _latest_human_text(getattr(request, "messages", None))
        if key == self._last_key:
            return self._last_block
        block = self._build_and_consume()
        self._last_key = key
        self._last_block = block
        return block

    def _apply(self, request):
        try:
            block = self._block_for(request)
        except Exception as exc:  # noqa: BLE001
            logger.debug("RequestReplyReadbackMiddleware failed: %s", exc)
            return request
        if not block:
            return request
        # 用户刚答复的内容，逐轮不同，而且本来就更像 human 侧的话 → 挂最后一条
        # human 消息（同 memory_recall / live_state 的 cache 理由）。
        out = append_human_block(request, PROMPT_ID, block)
        if out is not None:
            return out
        return append_system_block(request, PROMPT_ID, block)

    def wrap_model_call(self, request, handler: Callable[[Any], Any]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._apply(request))


def make_request_readback_middleware(**kw) -> RequestReplyReadbackMiddleware:
    return RequestReplyReadbackMiddleware(**kw)


__all__ = ["RequestReplyReadbackMiddleware", "make_request_readback_middleware"]
