"""GUI helpers for the cognition tab (memory / dreaming / brainstorm).

Surfaces the (already-built, committed) cognition backend as a Lab Console tab:
  * browse / edit / pin / delete persistent memories per namespace,
  * see conversation-phase summaries (sharding output),
  * trigger a dreaming-consolidation pass on demand,
  * launch a multi-agent brainstorm about the experiment + progress.

All functions are pure-ish (take a ``MemoryStore`` / ids, return HTML or a
status string) so ``app.py`` just wires them to Gradio components. Everything
rendered into ``gr.HTML`` is escaped — memory content is arbitrary text and the
``🌙 dream`` / brainstorm outputs may contain anything.
"""

from __future__ import annotations

import html
import logging

from mast.memory.store import KINDS, MemoryStore

logger = logging.getLogger(__name__)

# Visual marker per memory kind.
_KIND_ICON = {
    "note": "📝", "insight": "💡", "summary": "🗂", "hypothesis": "🔬",
    "protocol": "📐", "dream": "🌙", "brainstorm": "🧠",
}


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


# ── namespaces / paths (for dropdowns) ────────────────────────────────

def memory_namespaces(store: MemoryStore | None) -> list[str]:
    """Existing namespaces, always including 'global' first."""
    if store is None:
        return ["global"]
    try:
        ns = store.namespaces()
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("memory_namespaces failed: %s", exc)
        ns = []
    out = ["global"] + [n for n in ns if n != "global"]
    # de-dup preserving order
    seen, res = set(), []
    for n in out:
        if n not in seen:
            seen.add(n); res.append(n)
    return res


def memory_paths(store: MemoryStore | None, namespace: str | None) -> list[str]:
    """Paths in a namespace (pinned first), for the editor dropdown."""
    if store is None:
        return []
    try:
        return [r["path"] for r in store.list(namespace or "global", limit=200)]
    except Exception as exc:  # pragma: no cover
        logger.debug("memory_paths failed: %s", exc)
        return []


# ── list / detail rendering ───────────────────────────────────────────

def render_memory_list_html(store: MemoryStore | None, namespace: str | None,
                            *, kind: str | None = None,
                            query: str | None = None) -> str:
    """A compact table of memories in a namespace (pinned first)."""
    if store is None:
        return '<div style="color:#94a3b8;padding:12px;">记忆存储未就绪。</div>'
    ns = namespace or "global"
    try:
        if query:
            rows = store.search(query, namespace=ns, limit=200)
            if kind:
                rows = [r for r in rows if r.get("kind") == kind]
        else:
            rows = store.list(ns, kind=kind, limit=200)
    except Exception as exc:  # pragma: no cover
        logger.debug("render_memory_list_html failed: %s", exc)
        return f'<div style="color:#ef4444;padding:12px;">读取失败: {_esc(exc)}</div>'
    if not rows:
        return ('<div style="color:#94a3b8;padding:12px;">'
                f'命名空间 <code>{_esc(ns)}</code> 暂无记忆。</div>')
    items = []
    for r in rows:
        icon = _KIND_ICON.get(r.get("kind"), "📄")
        star = "📌 " if r.get("pinned") else ""
        title = r.get("title") or (r.get("content", "")[:60].replace("\n", " "))
        author = r.get("author") or ""
        items.append(
            '<div style="padding:6px 10px;border-bottom:1px solid rgba(148,163,184,0.15);">'
            f'<span>{star}{icon} <code>{_esc(r.get("path"))}</code> '
            f'<span style="color:#64748b;font-size:0.85em;">({_esc(r.get("kind"))}'
            f'{" · " + _esc(author) if author else ""})</span></span>'
            f'<div style="color:#cbd5e1;font-size:0.9em;margin-top:2px;">{_esc(title)}</div>'
            '</div>'
        )
    return (f'<div style="font-size:0.95em;">'
            f'<div style="color:#94a3b8;padding:4px 10px;">'
            f'{len(rows)} 条 · 命名空间 <code>{_esc(ns)}</code></div>'
            + "".join(items) + '</div>')


def render_memory_detail_html(store: MemoryStore | None, namespace: str | None,
                              path: str | None) -> str:
    if store is None or not path:
        return '<div style="color:#94a3b8;padding:12px;">选择一条记忆查看详情。</div>'
    r = store.read(namespace or "global", path)
    if r is None:
        return f'<div style="color:#94a3b8;padding:12px;">无此记忆: {_esc(path)}</div>'
    icon = _KIND_ICON.get(r.get("kind"), "📄")
    tags = " ".join(f'<span style="background:rgba(99,102,241,0.15);border-radius:3px;'
                    f'padding:0 5px;margin-right:3px;">{_esc(t)}</span>'
                    for t in (r.get("tags") or []))
    return (
        f'<div style="padding:10px;">'
        f'<div style="font-weight:600;">{"📌 " if r.get("pinned") else ""}{icon} '
        f'{_esc(r.get("title") or r.get("path"))}</div>'
        f'<div style="color:#64748b;font-size:0.82em;margin:4px 0;">'
        f'{_esc(r.get("kind"))} · {_esc(r.get("author") or "?")} · '
        f'更新 {_esc((r.get("updated_at") or "")[:19])}</div>'
        f'<div style="margin:4px 0;">{tags}</div>'
        f'<pre style="white-space:pre-wrap;background:rgba(15,23,42,0.4);'
        f'padding:8px;border-radius:5px;font-size:0.9em;">{_esc(r.get("content"))}</pre>'
        f'</div>'
    )


# ── editor handlers (return status strings) ───────────────────────────

def save_memory(store: MemoryStore | None, namespace: str | None, path: str,
                content: str, *, kind: str = "note", title: str = "",
                pin: bool = False) -> str:
    if store is None:
        return "❌ 记忆存储未就绪"
    if not (path or "").strip():
        return "❌ 需要 path（如 insights/tip.md）"
    k = kind if kind in KINDS else "note"
    try:
        r = store.write(namespace or "global", path, content or "", title=title,
                        kind=k, author="user", pinned=bool(pin))
        return f"✅ 已保存 {r['namespace']}/{r['path']}"
    except Exception as exc:
        return f"❌ 保存失败: {exc}"


def delete_memory(store: MemoryStore | None, namespace: str | None,
                  path: str | None) -> str:
    if store is None or not path:
        return "❌ 选择要删除的记忆"
    try:
        ok = store.delete(namespace or "global", path)
        return "🗑 已删除" if ok else "（无此记忆）"
    except Exception as exc:
        return f"❌ 删除失败: {exc}"


def toggle_pin_memory(store: MemoryStore | None, namespace: str | None,
                      path: str | None, on: bool = True) -> str:
    if store is None or not path:
        return "❌ 选择记忆"
    r = store.read(namespace or "global", path)
    if r is None:
        return "（无此记忆）"
    try:
        store.pin(int(r["id"]), bool(on))
        return "📌 已置顶" if on else "已取消置顶"
    except Exception as exc:
        return f"❌ 失败: {exc}"


# ── phase summaries (sharding) ────────────────────────────────────────

def render_phase_summaries_html(pm, experiment_id: str | None) -> str:
    """List conversation-phase summaries for an experiment (sharding output)."""
    if pm is None:
        return '<div style="color:#94a3b8;padding:12px;">分片管理未就绪。</div>'
    try:
        phases = pm.list_phases(experiment_id)
    except Exception as exc:  # pragma: no cover
        return f'<div style="color:#ef4444;padding:12px;">读取失败: {_esc(exc)}</div>'
    if not phases:
        return ('<div style="color:#94a3b8;padding:12px;">'
                '尚无对话阶段。长对话会自动/手动分片，每段产出压缩摘要。</div>')
    items = []
    for p in phases:
        title = p.get("title") or f"阶段 {p.get('phase_index')}"
        summary = p.get("summary") or "(未摘要)"
        open_ = p.get("ended_msg_id") is None
        items.append(
            '<div style="padding:6px 10px;border-bottom:1px solid rgba(148,163,184,0.15);">'
            f'<span style="font-weight:600;">#{_esc(p.get("phase_index"))} {_esc(title)}'
            f'{" · 进行中" if open_ else ""}</span>'
            f'<div style="color:#cbd5e1;font-size:0.9em;margin-top:2px;">{_esc(summary)}</div>'
            '</div>'
        )
    return '<div style="font-size:0.95em;">' + "".join(items) + '</div>'


# ── dreaming (on-demand) ──────────────────────────────────────────────

def run_dream_now(db_path, store: MemoryStore | None, *, consolidator=None) -> str:
    """Run one dreaming-consolidation pass now and report what it wrote."""
    if store is None:
        return "❌ 记忆存储未就绪"
    try:
        from mast.memory.dreaming import DreamingService
        svc = DreamingService(db_path, store, consolidator=consolidator)
        written = svc.dream_once()
    except Exception as exc:
        return f"❌ 做梦失败: {exc}"
    if not written:
        return "🌙 做梦完成：无新的可固化内容（或已固化、被去重）。"
    paths = ", ".join(w.get("path", "?") for w in written)
    return f"🌙 做梦完成：写入 {len(written)} 条记忆（{paths}）。标注「非实测，仅供参考」。"


# ── brainstorm (lazy — backend may land slightly later) ───────────────

def run_brainstorm_panel(db_path, experiment_id: str | None, *, topic: str = "",
                         user_viewpoints: str = "", max_rounds: int = 2,
                         llm=None, store: MemoryStore | None = None) -> tuple[str, str]:
    """Run a brainstorm and return (transcript_html, summary_text).

    Imports the brainstorm graph lazily so this module is usable even before the
    brainstorm backend is wired; degrades to a clear message if absent.
    """
    try:
        from mast.agents.brainstorm.graph import run_brainstorm
    except Exception:
        return ('<div style="color:#94a3b8;padding:12px;">头脑风暴后端尚在构建中。</div>', "")
    vps = [v.strip() for v in (user_viewpoints or "").splitlines() if v.strip()]
    try:
        res = run_brainstorm(db_path, experiment_id, topic=topic,
                             user_viewpoints=vps, max_rounds=max_rounds,
                             llm=llm, memory_store=store)
    except Exception as exc:
        return (f'<div style="color:#ef4444;padding:12px;">头脑风暴失败: {_esc(exc)}</div>', "")
    transcript = res.get("transcript", []) if isinstance(res, dict) else []
    summary = res.get("summary", "") if isinstance(res, dict) else ""
    rows = []
    for turn in transcript:
        spk = _esc(turn.get("speaker") or turn.get("role") or "?")
        content = _esc(turn.get("content") or "")
        rows.append(
            '<div style="padding:6px 10px;border-bottom:1px solid rgba(148,163,184,0.12);">'
            f'<span style="color:#a78bfa;font-weight:600;">{spk}</span>'
            f'<div style="color:#cbd5e1;font-size:0.92em;margin-top:2px;">{content}</div>'
            '</div>'
        )
    html_out = ('<div style="font-size:0.95em;">'
                + ("".join(rows) or '<div style="color:#94a3b8;padding:12px;">（空）</div>')
                + '</div>')
    return html_out, summary


__all__ = [
    "memory_namespaces", "memory_paths", "render_memory_list_html",
    "render_memory_detail_html", "save_memory", "delete_memory",
    "toggle_pin_memory", "render_phase_summaries_html", "run_dream_now",
    "run_brainstorm_panel",
]
