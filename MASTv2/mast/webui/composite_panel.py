"""GUI panel for the declarative composite-skill builder.

Surfaces the (already-built) composite backend — version store, templates,
clone, and the offline flow-graph renderer — as a Lab Console tab:
  * a list of every stored composite,
  * its control-flow rendered as a graph (steps / conditionals / loops),
  * its version history with one-click rollback,
  * "clone as template" to copy a composite as a blueprint for a new one.

All functions are pure-ish (take the store / names, return HTML or status) so
app.py just wires them to Gradio components.
"""

from __future__ import annotations

import logging

from mast.skills.composite.graph_render import spec_to_html, spec_to_mermaid
from mast.skills.composite.version_store import (
    CompositeVersionStore, VersionStoreError,
)

logger = logging.getLogger(__name__)

_store: CompositeVersionStore | None = None

# 修复项 (2026-06-11): the app's LIVE SkillRegistry, wired at startup right
# after load_spec_skills(). Store mutations (clone/restore/delete) then hot-
# (un)register so the running agent sees the change without a restart —
# previously a freshly cloned composite was NOT runnable until restart, a
# rollback left the OLD version registered, and a delete left a ghost skill.
_live_registry = None
_agent_refresh = None


def set_live_registry(registry) -> None:
    """Wire the live SkillRegistry for hot-(un)registration (best-effort)."""
    global _live_registry
    _live_registry = registry


def set_agent_refresh(fn) -> None:
    """Wire a callback that refreshes the AGENT tool list after a registry
    change (修复项 review fix: the orchestrator freezes wrapped tools at build
    time — registry hot-registration alone does not reach a built agent).
    The callback returns a UI suffix describing what it did."""
    global _agent_refresh
    _agent_refresh = fn


def _refresh_agents() -> str:
    # Any registry change also invalidates the builder palette catalog (P1).
    try:
        from mast.webui.builder_api import invalidate_catalog
        invalidate_catalog()
    except Exception:  # pragma: no cover — builder API optional
        pass
    fn = _agent_refresh
    if fn is None:
        return ""
    try:
        return fn() or ""
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("agent tools refresh failed: %s", exc)
        return ""


def _hot_register(name: str) -> str:
    """(Re-)register *name* from the store into the live registry.
    Returns a short UI suffix describing what happened ('' if not wired)."""
    reg = _live_registry
    if reg is None:
        return ""
    try:
        from mast.skills.composite.loader import register_spec
        spec = composite_store().load(name)
        problems = spec.validate()
        if problems:
            return f" 注意：未热注册（spec 校验失败：{'; '.join(problems)}）。"
        register_spec(reg, spec)
        return " 已热注册。" + _refresh_agents()
    except ValueError as exc:
        if "collide" in str(exc):
            # Restart won't help — load_spec_skills hits the same refusal.
            logger.warning("composite hot-register refused for %s: %s", name, exc)
            return (" 注意：与现有非组合技能重名，无法注册为可执行技能"
                    "（重启也不会生效）——请改名后使用。")
        logger.warning("composite hot-register failed for %s: %s", name, exc)
        return f" 热注册失败（{exc}），重启后生效。"
    except Exception as exc:
        logger.warning("composite hot-register failed for %s: %s", name, exc)
        return f" 热注册失败（{exc}），重启后生效。"


def _hot_unregister(name: str) -> str:
    reg = _live_registry
    if reg is None:
        return ""
    try:
        # only_subclass_of guards name collisions: a composite named like a
        # builtin (e.g. "SetBias") must never deregister the builtin.
        from mast.skills.composite.interpreter import SpecComposite
        if reg.unregister(name, only_subclass_of=SpecComposite):
            return " 已从运行注册表移除。" + _refresh_agents()
        return ""
    except Exception as exc:
        logger.warning("composite hot-unregister failed for %s: %s", name, exc)
        return f" 运行注册表移除失败（{exc}），重启后消失。"


def composite_store() -> CompositeVersionStore:
    """Singleton store, seeded with built-in templates on first use."""
    global _store
    if _store is None:
        _store = CompositeVersionStore()
        try:
            from mast.skills.composite.templates import seed_templates
            seed_templates(_store)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("composite template seed failed: %s", exc)
        # P5: also seed the declarative twins of the built-in composites
        # (FullScan / ConditionTip / ShapeTipOnSurface / …) so they are
        # openable + forkable in the builder.
        try:
            from mast.skills.composite.builtin_composites import (
                seed_builtin_composites,
            )
            seed_builtin_composites(_store)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("builtin composite seed failed: %s", exc)
    return _store


def composite_choices() -> list[str]:
    try:
        return [s["name"] for s in composite_store().list_specs()]
    except Exception as exc:
        logger.debug("composite_choices failed: %s", exc)
        return []


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_composite_html(name: str | None) -> str:
    """Full HTML for one composite: metadata + flow graph + version history."""
    if not name:
        return ('<div style="color:#94a3b8;padding:12px;">选择一个复杂技能查看其'
                '流程图与版本历史。内置模版可"克隆为新技能"作为蓝本。</div>')
    store = composite_store()
    try:
        spec = store.load(name)
    except VersionStoreError:
        return f'<div style="color:#dc2626;">未找到复杂技能 {_esc(name)}</div>'
    safety_color = {"auto": "#16a34a", "confirm": "#b45309",
                    "dangerous": "#dc2626"}.get(spec.safety_level, "#64748b")
    params = "".join(
        f'<li><code>{_esc(p.name)}</code> ({_esc(p.type)})'
        + (f' = {_esc(p.default)}' if p.default is not None else "")
        + (' <span style="color:#dc2626">*</span>' if p.required else "")
        + (f' — {_esc(p.description)}' if p.description else "") + "</li>"
        for p in spec.params)
    versions = store.list_versions(name)
    vhist = "".join(
        f'<li>v{v["version"]} '
        f'<span style="color:#94a3b8;font-size:0.85em;">{_esc(v.get("saved_at",""))[:19]} '
        f'· {v.get("n_nodes",0)} 节点</span></li>'
        for v in versions)
    return (
        '<div style="font-size:0.95em;">'
        f'<div style="display:flex;align-items:baseline;gap:8px;">'
        f'<h3 style="margin:0;">{_esc(spec.name)}</h3>'
        f'<span style="color:#64748b;">v{spec.version}</span>'
        f'<span style="color:{safety_color};font-weight:600;">● {_esc(spec.safety_level)}</span>'
        f'</div>'
        f'<div style="color:#475569;margin:4px 0 8px;">{_esc(spec.description)}</div>'
        + (f'<div><b>参数</b><ul style="margin:4px 0 8px;">{params}</ul></div>' if params else "")
        + '<div style="font-weight:600;margin:8px 0 4px;">流程图</div>'
        + '<div style="border:1px solid #e2e8f0;border-radius:8px;padding:10px;background:#fff;">'
        + spec_to_html(spec) + '</div>'
        + (f'<div style="font-weight:600;margin:10px 0 4px;">版本历史</div>'
           f'<ul style="margin:4px 0;">{vhist}</ul>' if vhist else "")
        + '</div>')


def composite_mermaid(name: str | None) -> str:
    if not name:
        return ""
    try:
        return spec_to_mermaid(composite_store().load(name))
    except Exception:
        return ""


def clone_composite(src: str | None, new_name: str | None, author: str = "") -> str:
    if not src:
        return "请先选择要克隆的复杂技能。"
    if not new_name or not new_name.strip():
        return "请填写新技能名称。"
    try:
        c = composite_store().clone(src, new_name.strip(), author=author)
        return (f"已克隆 {src} → {c.name}（v{c.version}）。可在列表中选择并编辑。"
                + _hot_register(c.name))
    except VersionStoreError as exc:
        return f"克隆失败：{exc}"
    except Exception as exc:  # pragma: no cover
        return f"克隆失败：{exc}"


def restore_composite(name: str | None, version) -> str:
    if not name or version in (None, ""):
        return "请选择技能与要回滚到的版本。"
    try:
        v = int(version)
    except (ValueError, TypeError):
        return f"无效版本号：{version}"
    try:
        s = composite_store().restore(name, v)
        return (f"已回滚 {name} 到 v{v} 的内容（写为新版本 v{s.version}，历史保留）。"
                + _hot_register(s.name))
    except VersionStoreError as exc:
        return f"回滚失败：{exc}"
    except Exception as exc:  # pragma: no cover — e.g. WinError sharing violation
        return f"回滚失败：{exc}"


def delete_composite(name: str | None) -> str:
    if not name:
        return "请选择要删除的复杂技能。"
    try:
        composite_store().delete(name)
        return f"已删除 {name} 的当前版本（历史快照保留)。" + _hot_unregister(name)
    except Exception as exc:
        return f"删除失败：{exc}"


__all__ = [
    "composite_store", "composite_choices", "render_composite_html",
    "composite_mermaid", "clone_composite", "restore_composite", "delete_composite",
    "set_live_registry", "set_agent_refresh",
]
