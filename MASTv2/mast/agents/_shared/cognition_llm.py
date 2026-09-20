"""Provider-portable LLM functions for the cognition layer.

Two injectables that replace the dependency-free rule-based defaults inside
``CognitionContext`` when a model is available:

  * ``make_llm_phase_summarizer(llm)`` → ``fn(messages) -> str`` for
    ``CognitionContext.set_summarizer`` (PhaseManager phase summaries).
  * ``make_llm_dream_consolidator(llm)`` → ``fn(context) -> list[dict]`` for
    ``CognitionContext.set_consolidator`` (DreamingService background insight).

Both use a plain ``llm.invoke`` (NOT ``with_structured_output`` — that is not
provider-portable across MAST's 6 providers) and degrade gracefully: the
PhaseManager / DreamingService already fall back to the rule-based path if the
injected function raises, so these stay best-effort.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _msg_text(resp) -> str:
    """Extract assistant text from a ChatModel response across providers."""
    txt = getattr(resp, "text", None)
    if isinstance(txt, str):
        return txt.strip()
    if callable(txt):
        try:
            return str(txt()).strip()
        except Exception:  # noqa: BLE001
            pass
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        return "".join(parts).strip()
    return str(content).strip()


def make_llm_phase_summarizer(llm) -> Callable[[list[dict]], str]:
    """Build a phase summariser ``fn(messages: list[{role,content}]) -> str``."""

    def _summarize(messages: list[dict]) -> str:
        convo = "\n".join(
            f"{m.get('role', '?')}: {str(m.get('content', ''))[:600]}"
            for m in (messages or []))[:8000]
        prompt = (
            "下面是一段 STM 实验对话的片段。用 2-4 句中文提炼这一阶段的关键内容:"
            "用户意图、做出的关键决定/结论、产生的产物或当前状态。只输出摘要正文。\n\n"
            f"{convo}")
        resp = llm.invoke(prompt)
        return _msg_text(resp) or "(本阶段无可提炼内容)"

    return _summarize


def make_llm_dream_consolidator(llm) -> Callable[[dict], list]:
    """Build a dream consolidator ``fn(context: dict) -> list[{path,title,content,kind}]``."""

    def _consolidate(context: dict) -> list:
        exps = context.get("experiments", [])
        if not exps:
            return []
        skills = sorted(context.get("skill_counts", {}).items(),
                        key=lambda kv: kv[1], reverse=True)[:10]
        statuses = context.get("statuses", {})
        phases = context.get("phase_summaries", [])
        ctx_lines = [
            f"近期实验数: {len(exps)}",
            "状态分布: " + ", ".join(f"{k}×{v}" for k, v in statuses.items()),
            "高频技能: " + ", ".join(f"{s}×{n}" for s, n in skills),
        ]
        for p in phases[:6]:
            ctx_lines.append(f"阶段摘要 - {p.get('title') or '阶段'}: {(p.get('summary') or '')[:160]}")
        prompt = (
            "你在对 STM 自治实验系统的近期活动做『睡眠固化』:从下列统计与摘要中"
            "提炼可复用的跨实验洞见(成功模式、反复出现的失败共因、值得固化为协议的"
            "参数)。用中文 markdown 输出,3-6 条要点。这是基于历史数据的【推断】,"
            "不是实测,请勿编造具体数值。\n\n" + "\n".join(ctx_lines))
        try:
            text = _msg_text(llm.invoke(prompt))
        except Exception as exc:  # noqa: BLE001 — dreaming catches, but be explicit
            logger.debug("dream consolidator LLM failed: %s", exc)
            return []
        if not text:
            return []
        try:
            from mast.memory.dreaming import DREAM_TAG
            body = f"{DREAM_TAG}\n\n{text}"
        except Exception:  # noqa: BLE001
            body = text
        return [{"path": "dreams/consolidation.md",
                 "title": "做梦复盘 (LLM 固化)", "kind": "dream",
                 "content": body}]

    return _consolidate


__all__ = ["make_llm_phase_summarizer", "make_llm_dream_consolidator"]
