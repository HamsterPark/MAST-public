"""Agent-facing cognition tools — brainstorm + dream as LangChain tools.

Exposes the cognition-panel actions (multi-perspective brainstorm + memory
consolidation "dream") as tools any tool-running agent can call, mirroring
``agents/_shared/memory_tools.py``. The supervisor is a PURE ROUTER (no tool
list), so it reaches cognition by routing to an agent that holds these tools
(e.g. experiment_design / data_processing). Lives under ``agents/_shared`` so it
can attach to several agents without crossing the agent boundary.

A ``provider`` callable supplies the live db_path / MemoryStore / experiment id /
optional LLM, re-evaluated per call so the context follows the current
experiment without rebuilding the graph.
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# provider() -> {"db_path": str, "store": MemoryStore|None,
#                "experiment_id": str|None, "llm": chat-model|callable|None}
Provider = Callable[[], dict]


def make_cognition_tools(provider: Provider) -> list:
    """Build the [brainstorm, dream] cognition tools bound to *provider*."""

    def _ctx() -> dict:
        try:
            return provider() or {}
        except Exception as exc:  # pragma: no cover - provider best-effort
            logger.debug("cognition provider failed: %s", exc)
            return {}

    @tool("brainstorm")
    def brainstorm(topic: str, viewpoints: list | None = None,
                   max_rounds: int = 2) -> str:
        """发起一轮多视角头脑风暴(离线规则兜底;有 LLM 时更深入)。

        topic: 议题,如 "如何在 Au(111) 上提升针尖质量"。
        viewpoints: 可选,要纳入的具体观点/约束列表。
        max_rounds: 讨论轮数(默认 2)。
        返回简要结论摘要;完整 transcript 写入记忆(可用 memory_search 检索)。"""
        ctx = _ctx()
        try:
            from mast.agents.brainstorm.graph import run_brainstorm
            r = run_brainstorm(
                ctx.get("db_path"), ctx.get("experiment_id"),
                topic=topic, user_viewpoints=viewpoints or [],
                max_rounds=int(max_rounds or 2),
                llm=ctx.get("llm"), memory_store=ctx.get("store"),
            ) or {}
            summary = r.get("summary", "") or "(无摘要)"
            n = len(r.get("transcript", []) or [])
            return f"头脑风暴完成（{n} 条发言）：\n{summary}"
        except Exception as exc:  # noqa: BLE001 — tool loop must not crash
            return f"头脑风暴失败: {exc}"

    @tool("dream")
    def dream() -> str:
        """对实验记忆做一次"做梦"整合:聚类近期记忆→提炼洞见,写回记忆。

        无参数;返回本次新增/更新的整合条目摘要。"""
        ctx = _ctx()
        store = ctx.get("store")
        if store is None:
            return "记忆 store 不可用,无法做梦"
        try:
            from mast.memory.dreaming import DreamingService
            svc = DreamingService(ctx.get("db_path"), store)
            entries = svc.dream_once() or []
            if not entries:
                return "🌙 做梦完成：暂无可整合的新记忆"
            titles = "; ".join(str(e.get("title") or e.get("path") or "?")
                               for e in entries[:6])
            return f"🌙 做梦完成：新增/更新 {len(entries)} 条整合记忆 — {titles}"
        except Exception as exc:  # noqa: BLE001
            return f"做梦失败: {exc}"

    return [brainstorm, dream]


__all__ = ["make_cognition_tools"]
