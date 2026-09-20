"""Agent-facing persistent-memory tools (docs/v2/design/agentic-cognition.md §1.3).

These wrap :class:`~mast.memory.store.MemoryStore` as LangChain tools so any
agent can proactively read/write structured memory that survives across
sessions. A ``provider`` callable supplies the live store + the current
namespace / experiment so a tool call lands in the right place without the LLM
having to manage that.

Lives under ``agents/_shared`` (a shared module) — not inside any single agent —
so it can be attached to several agents without crossing the agent boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# provider() -> {"store": MemoryStore, "namespace": str,
#                "experiment_id": str|None, "author": str}
Provider = Callable[[], dict]


def make_memory_tools(provider: Provider) -> list:
    """Build the [write, read, list, search] memory tools bound to *provider*."""

    def _ctx() -> dict:
        try:
            return provider() or {}
        except Exception as exc:  # pragma: no cover - provider best-effort
            logger.debug("memory provider failed: %s", exc)
            return {}

    @tool("memory_write")
    def memory_write(path: str, content: str, kind: str = "note",
                     tags: list | None = None, pin: bool = False) -> str:
        """持久化一条记忆(跨 session 保存,下次也能读到)。

        path: 文件式路径,如 'insights/tip-conditioning.md'(同 path 覆盖=更新)。
        content: markdown 正文(经验/假设/协议/结论)。
        kind: note|insight|summary|hypothesis|protocol。
        勿存原始扫描数据或大数组——只存可复用的结构化知识。"""
        ctx = _ctx()
        store = ctx.get("store")
        if store is None:
            return "memory store unavailable"
        try:
            # Prefer the cognition `writer` (cog.remember) when supplied so the
            # write is ALSO indexed into the semantic vector store; fall back to a
            # bare store.write when no writer is wired (offline / store-only).
            writer = ctx.get("writer")
            ns = ctx.get("namespace", "global")
            if callable(writer):
                r = writer(ns, path, content, kind=kind, tags=tags or [],
                           experiment_id=ctx.get("experiment_id"),
                           author=ctx.get("author", "agent"), pinned=bool(pin))
            else:
                r = store.write(ns, path, content, kind=kind, tags=tags or [],
                                experiment_id=ctx.get("experiment_id"),
                                author=ctx.get("author", "agent"), pinned=bool(pin))
            return f"已保存记忆 {r['namespace']}/{r['path']}"
        except Exception as exc:
            return f"记忆写入失败: {exc}"

    @tool("memory_read")
    def memory_read(path: str) -> str:
        """读取一条记忆(按 path)。返回正文,不存在则返回提示。"""
        ctx = _ctx()
        store = ctx.get("store")
        if store is None:
            return "memory store unavailable"
        r = store.read(ctx.get("namespace", "global"), path)
        if r is None:
            # also try global as a fallback
            r = store.read("global", path)
        if r is None:
            return f"(无此记忆: {path})"
        return f"# {r.get('title') or r['path']} ({r['kind']})\n{r['content']}"

    @tool("memory_list")
    def memory_list(kind: str | None = None) -> str:
        """列出当前命名空间(+global)的记忆索引(path + 标题),可按 kind 过滤。"""
        ctx = _ctx()
        store = ctx.get("store")
        if store is None:
            return "memory store unavailable"
        ns = ctx.get("namespace", "global")
        rows = store.list(ns, kind=kind, limit=80)
        if ns != "global":
            rows = rows + store.list("global", kind=kind, limit=40)
        if not rows:
            return "(无记忆)"
        return "\n".join(
            f"- `{r['path']}` ({r['kind']}) — {r.get('title') or r['content'][:50]}"
            for r in rows)

    @tool("memory_search")
    def memory_search(query: str) -> str:
        """全文搜索记忆(标题/正文/标签),返回匹配的 path + 摘要。

        只搜当前命名空间(+global 共享),不跨实验泄漏其他实验的记忆。"""
        ctx = _ctx()
        store = ctx.get("store")
        if store is None:
            return "memory store unavailable"
        ns = ctx.get("namespace", "global")
        # Prefer the cognition `recaller` (semantic knn, namespace-scoped) when
        # wired; fall back to substring search over the caller's namespace
        # (+global) so an agent in experiment A can't read experiment B's memory.
        recaller = ctx.get("recaller")
        if callable(recaller):
            try:
                rows = recaller(query) or []
            except Exception:  # noqa: BLE001
                rows = []
            if rows:
                return "\n".join(
                    f"- `{r['path']}` — {r.get('title') or r['content'][:60]}"
                    for r in rows[:15])
        rows = store.search(query, namespace=ns, limit=15)
        if ns != "global":
            seen = {r["path"] for r in rows}
            rows = rows + [r for r in store.search(query, namespace="global", limit=10)
                           if r["path"] not in seen]
        if not rows:
            return f"(无匹配: {query})"
        return "\n".join(
            f"- `{r['path']}` — {r.get('title') or r['content'][:60]}"
            for r in rows[:15])

    return [memory_write, memory_read, memory_list, memory_search]


__all__ = ["make_memory_tools"]
