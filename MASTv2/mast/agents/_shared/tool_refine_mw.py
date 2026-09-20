"""Per-call ToolMessage refinement middleware (before_model hook).

Bounds multi-turn context growth from tool returns. Before every model call, the
OLDER completed ToolMessages in history are refined -- verbose prose is
compressed while ALL numeric / coordinate / status / failure info is preserved
verbatim. The MOST RECENT tool exchange is left untouched so the agent still
sees the full raw result for its current decision.

Why before_model (not after_agent): in the group/orchestrator path an agent
turn ends with a handoff Command(goto=PARENT) that BYPASSES after_agent
(verified 2026-07-07), so an after_agent hook would never fire in group chat.
before_model fires reliably in BOTH private (standalone) and group (subgraph)
ReAct loops.

Five invariants (2026-07-07 impact survey):
  1. Reuse the original ToolMessage .id so add_messages REPLACES in place (a
     new/absent id would APPEND an orphan tool result -> provider 400).
  2. Idempotent: mark refined messages (additional_kwargs) and skip them next
     time -- else every call re-runs the LLM and thrashes the prompt cache.
  3. Preserve structure: keep status / name / tool_call_id and the failed: /
     precondition_failed: markers (StallGuard reads them).
  4. Return ONLY the refined ToolMessages under the messages key.
  5. Fail-safe: on error / empty / refusal / inflation, keep the original.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

_MARKER = "mast_refined"
_REFUSAL_PREFIXES = ("error", "sorry", "i cannot", "i can not", "as an ai",
                     "抱歉", "无法")

_REFINE_PROMPT = (
    "你是仪器实验对话的“工具返回精炼器”。"
    "下面是一条工具调用的返回内容，"
    "请在**不丢失任何信息价值**的前提下压缩它：\n"
    "- 必须**原样保留**所有数值、坐标、测量值、"
    "单位、文件路径、状态字段，"
    "以及 failed:/precondition_failed:/rolled_back: 等标记；\n"
    "- 只压缩冗长的自然语言叙述与重复说明；\n"
    "- 直接输出精炼后的结果本身。\n\n"
    "工具返回：\n"
)


class ToolRefinementMiddleware(AgentMiddleware):
    """Refine older ToolMessage contents before each model call (see module doc)."""

    def __init__(self, *, summarizer_model: Any, min_chars: int = 600,
                 keep_recent: int = 1, max_per_call: int = 4):
        super().__init__()
        self._summ = summarizer_model
        self._min_chars = max(1, int(min_chars))
        self._keep_recent = max(0, int(keep_recent))
        self._max_per_call = max(1, int(max_per_call))

    def before_model(self, state, runtime):  # type: ignore[override]
        msgs = state.get("messages") or []
        tool_idx = [i for i, m in enumerate(msgs) if isinstance(m, ToolMessage)]
        if len(tool_idx) <= self._keep_recent:
            return None
        cand_idx = tool_idx[:-self._keep_recent] if self._keep_recent else tool_idx
        refined: list = []
        for i in cand_idx:
            m = msgs[i]
            if m.additional_kwargs.get(_MARKER):
                continue
            if not isinstance(m.content, str):
                continue
            if len(m.content) < self._min_chars:
                continue
            new_content = self._refine(m.content)
            if new_content is None:
                continue
            refined.append(ToolMessage(
                content=new_content,
                tool_call_id=m.tool_call_id,
                id=m.id,
                name=m.name,
                status=m.status,
                additional_kwargs={**m.additional_kwargs, _MARKER: True},
            ))
            if len(refined) >= self._max_per_call:
                break
        return {"messages": refined} if refined else None

    async def abefore_model(self, state, runtime):  # type: ignore[override]
        return self.before_model(state, runtime)

    def _refine(self, content: str) -> "str | None":
        try:
            from mast.prompts.registry import resolve as resolve_prompt
            resp = self._summ.invoke(
                resolve_prompt("sub.tool_refine", _REFINE_PROMPT) + content)
            text = getattr(resp, "content", None)
            text = text.strip() if isinstance(text, str) else ""
            if not text:
                return None
            if len(text) >= len(content):
                return None
            low = text[:40].lower()
            if any(low.startswith(p) for p in _REFUSAL_PREFIXES):
                return None
            return text
        except Exception as exc:  # noqa: BLE001 -- refinement must never break a turn
            logger.debug("tool refine failed (kept original): %s", exc)
            return None


__all__ = ["ToolRefinementMiddleware"]
