"""MemoryRecallMiddleware — inject relevant long-term memory into the prompt.

Mirrors ``LiveStateMiddleware``: implements both sync ``wrap_model_call`` and
async ``awrap_model_call`` and appends a block to ``request.system_message`` so
the agent sees relevant cross-conversation memory WITHOUT having to call a tool.

Two cost/caching safeguards (see plan gotchas G2/G3/G5):

  * **Recall runs once per USER turn, not per tool sub-step.** The block is keyed
    by (experiment, latest-human-message) and cached, so the system message is
    STABLE across the sub-steps of one turn — otherwise a different block each
    sub-step would bust the Anthropic prompt cache and pay a DashScope embed per
    hop.
  * The cheap MEMORY.md index header is always included (no embedding); the
    semantic ``knn`` recall (network/local embed) only runs when the user message
    changes.

Best-effort: any failure leaves the request untouched (a recall glitch never
breaks a turn). Shared by 私聊 (private chat) and 群聊 (orchestrator) — both run
the same agents over the same ``CognitionContext``/namespaces.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage

from mast.agents._shared.inject import append_human_block, append_system_block

#: 登记表条目 id。
PROMPT_ID = "mw.memory_recall"

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


class MemoryRecallMiddleware(AgentMiddleware):
    """Append a 'relevant long-term memory' block to the system message."""

    def __init__(
        self,
        cog,
        namespace_provider: Callable[[], str | None],
        *,
        k: int = 5,
        char_budget: int = 2400,
        include_index: bool = True,
        index_lines: int = 12,
    ):
        super().__init__()
        self._cog = cog
        self._ns_provider = namespace_provider
        self._k = k
        self._char_budget = char_budget
        self._include_index = include_index
        self._index_lines = index_lines
        self._last_key: tuple | None = None
        self._last_block: str = ""

    # ── block construction ─────────────────────────────────────────────
    def _experiment_id(self) -> str | None:
        try:
            return self._ns_provider()
        except Exception:  # noqa: BLE001
            return None

    def _build_block(self, query: str, eid: str | None) -> str:
        parts: list[str] = []
        if self._include_index:
            try:
                idx = self._cog.memory_index(experiment_id=eid,
                                             max_lines=self._index_lines)
                if idx and "空" not in idx[:40]:
                    parts.append(idx.strip())
            except Exception as exc:  # noqa: BLE001
                logger.debug("memory_index failed: %s", exc)
        try:
            rows = self._cog.recall(query, experiment_id=eid, k=self._k) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("recall failed: %s", exc)
            rows = []
        if rows:
            bullets = ["## 相关长期记忆 (语义召回 · 仅供参考 · 🌙=AI推断非实测)"]
            for r in rows:
                title = r.get("title") or (r.get("content", "")[:40])
                excerpt = " ".join(str(r.get("content", "")).split())[:160]
                ns = r.get("namespace", "")
                tag = "🌙 " if r.get("kind") == "dream" else ""
                bullets.append(f"- {tag}[{ns}] `{r.get('path','')}` "
                               f"({r.get('kind','note')}) — {title}: {excerpt}")
            parts.append("\n".join(bullets))
        block = "\n\n".join(p for p in parts if p).strip()
        if len(block) > self._char_budget:
            block = block[:self._char_budget] + "\n…(记忆已截断)"
        return block

    def _block_for(self, request) -> str:
        query = _latest_human_text(getattr(request, "messages", None))
        if not query:
            return ""
        eid = self._experiment_id()
        key = (eid, query)
        if key == self._last_key:
            return self._last_block
        block = self._build_block(query, eid)
        self._last_key = key
        self._last_block = block
        return block

    def _apply(self, request):
        try:
            block = self._block_for(request)
        except Exception as exc:  # noqa: BLE001 — recall never breaks a turn
            logger.debug("MemoryRecallMiddleware failed: %s", exc)
            return request
        if not block:
            return request
        # 逐轮不同（检索键 = 本轮提问）→ 挂最后一条 human 消息，别动 system。
        # 理由与 live_state 同：Anthropic 的 cache 断点打在 system 末尾，system
        # 一变，整段 system + 全部历史每轮 miss。没有 human 可挂才退回 system。
        out = append_human_block(request, PROMPT_ID, block)
        if out is not None:
            return out
        return append_system_block(request, PROMPT_ID, block)

    def wrap_model_call(self, request, handler: Callable[[Any], Any]) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._apply(request))


def make_memory_recall_middleware(cog, namespace_provider, **kw) -> MemoryRecallMiddleware:
    return MemoryRecallMiddleware(cog, namespace_provider, **kw)


__all__ = ["MemoryRecallMiddleware", "make_memory_recall_middleware"]
