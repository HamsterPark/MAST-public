"""Domain ``guidance_ext`` — the three guidance sub-surfaces beyond per-skill
annotations (decision trees / workflow recipes / measurement templates).

The old ``技能指导`` admin tab had FOUR sub-tabs; the Gradio→TS rewrite only
re-wired the per-skill annotation one (``skill_extra``, in
``settings_admin_write.py``). This module re-exposes the other three as a single
parametric read/write pair:

* ``GET  /admin/guidance-extra/{kind}`` — effective config (code default merged
  with the persisted override) + the raw override layer.
* ``POST /admin/guidance-extra/{kind}`` — persist the whole-section override +
  hot-reload (empty payload ⇒ reset to code defaults).

``kind ∈ {decision_trees, recipes, templates}`` maps to ONE top-level key inside
``guidance_overrides.json`` plus its code-default constant in
``mast.knowledge.skill_guidance`` (single source for the mapping below). The
effective-merge semantics mirror the old ``admin/tabs/guidance/__init__``
``effective_*`` helpers exactly:

* decision_trees / measurement_templates → per-id deep-ish merge (override entry
  updates the default entry of the same id);
* workflow_recipes → whole-list REPLACE if an override list exists.

The API layer is a THIN passthrough: it only RELAYS to the kept
``ConfigOverrideRegistry`` (the very registry ``settings_admin_write.py`` uses
for the ``skill_extra`` sub-key). NO merge/validation/safety authority lives
here — that stays in core (R6); the presentation merge below is only so GET
returns what the editor should show.

GRACEFUL DEGRADATION is mandatory: this app must boot STANDALONE with no live
core wired. Every endpoint checks ctx for the registry; if it's absent or any
call raises, it returns a valid empty/degraded body (``degraded: true``) — never
a 500, never a crash on import. Heavy core modules are LAZY-imported INSIDE the
handler in try/except (mirrors routes/admin.py + routes/settings_admin_write.py).

All three kinds HAVE a write seam in core (same override file, distinct key), so
none is read-only; ``writable`` stays True throughout.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_guidance_ext import (
    GuidanceExtraResponse,
    GuidanceExtraWriteRequest,
    GuidanceExtraWriteResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["guidance_ext"])

# kind → (override key in guidance_overrides.json, code-default constant name,
# merge mode). Single source so GET / POST agree. The constant is resolved
# LAZILY inside the handler (degrade-safe import). Merge mode:
#   "dict"  → per-id merge (default ⊕ override, override entries win per key);
#   "list"  → whole-list REPLACE when an override list is present.
# Mirrors admin/tabs/guidance/__init__.effective_{decision_trees,recipes,
# templates} and the keys the old sub-tabs wrote.
_KIND_MAP: dict[str, tuple[str, str, str]] = {
    "decision_trees": ("decision_trees", "DECISION_TREES", "dict"),
    "recipes": ("workflow_recipes", "WORKFLOW_RECIPES", "list"),
    "templates": ("measurement_templates", "MEASUREMENT_TEMPLATES", "dict"),
}


# ── helpers ────────────────────────────────────────────────────────────────────
def _override_registry(ctx: Any):
    """Best-effort handle to a live ConfigOverrideRegistry.

    Prefers one already wired onto the context (set at integration
    time); otherwise None. We do NOT construct one here in standalone mode — that
    would touch the shared ``config/overrides`` dir and start mutating real files
    from a dev process. Absent ⇒ degrade. Mirrors routes/admin.py +
    routes/settings_admin_write.py."""
    return getattr(ctx, "override_registry", None)


def _kind_defaults(const_name: str) -> Any:
    """Best-effort code defaults for one guidance kind (for the GET merge).

    Lazy-imported so a missing/heavy knowledge module degrades to None rather
    than 500. The editor renders ``data`` (effective); if defaults are
    unavailable it falls back to the raw override alone."""
    try:
        from mast.knowledge import skill_guidance  # type: ignore

        return getattr(skill_guidance, const_name, None)
    except Exception as exc:  # any import issue ⇒ no defaults (raw override only)
        logger.debug("guidance defaults unavailable (%s): %s", const_name, exc)
        return None


def _effective(default: Any, override: Any, mode: str) -> Any:
    """Effective value = code default with the persisted override applied.

    Mirrors ``admin/tabs/guidance/__init__``: dict kinds merge per-id (override
    entry updates / adds the default entry of the same id); the list kind
    (recipes) is replaced wholesale when an override list exists. NO business
    logic — only the presentation merge so GET returns what the editor shows."""
    if mode == "list":
        if override is not None:
            return copy.deepcopy(override)
        return copy.deepcopy(default) if default is not None else None
    # dict per-id merge
    if not isinstance(default, dict):
        # No usable default → effective is the raw override (or None).
        return copy.deepcopy(override) if override is not None else None
    result = copy.deepcopy(default)
    if isinstance(override, dict):
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key].update(val)
            else:
                result[key] = copy.deepcopy(val)
    return result


# ── GET /admin/guidance-extra/{kind} ────────────────────────────────────────────
@router.get("/admin/guidance-extra/{kind}", response_model=GuidanceExtraResponse)
def get_guidance_extra(request: Request, kind: str) -> GuidanceExtraResponse:
    """Read one guidance kind's effective config (code defaults + override).

    ``kind ∈ {decision_trees, recipes, templates}``. Reads the top-level key
    inside ``guidance_overrides.json`` via the registry and merges it over the
    code-default constant the old sub-tab edited. Unknown kind / no registry ⇒
    degraded (still surfaces the code defaults when reachable)."""
    ctx = request.app.state.ctx
    spec = _KIND_MAP.get(kind)
    if spec is None:
        return GuidanceExtraResponse(kind=kind, degraded=True)
    override_key, const_name, mode = spec
    default = _kind_defaults(const_name)

    reg = _override_registry(ctx)
    if reg is None:
        # No registry → still surface code defaults so the editor renders.
        return GuidanceExtraResponse(
            kind=kind,
            data=_effective(default, None, mode),
            override=None,
            has_override=False,
            degraded=True,
        )
    try:
        from mast.admin.override_store import GUIDANCE_OVERRIDES

        data = reg.get_raw(GUIDANCE_OVERRIDES) or {}
        override = data.get(override_key)
        return GuidanceExtraResponse(
            kind=kind,
            data=_effective(default, override, mode),
            override=override,
            has_override=override_key in data,
            degraded=False,
        )
    except Exception as exc:
        logger.warning("guidance-extra read failed (%s): %s", kind, exc)
        # Degrade to code defaults (the read failed, the editor still renders).
        return GuidanceExtraResponse(
            kind=kind, data=_effective(default, None, mode), degraded=True
        )


# ── POST /admin/guidance-extra/{kind} ───────────────────────────────────────────
@router.post("/admin/guidance-extra/{kind}", response_model=GuidanceExtraWriteResponse)
def write_guidance_extra(
    request: Request, kind: str, body: GuidanceExtraWriteRequest
) -> GuidanceExtraWriteResponse:
    """Persist one guidance kind's whole-section override + hot-reload.

    Relays to the registry: the payload is written into ``guidance_overrides.json``
    under this kind's top-level key and saved with an in-process reload (mirrors
    the old sub-tabs' ``save_overrides``). Empty payload (None / {} / []) ⇒ that
    kind's override is dropped (reset to code defaults; the file is deleted when
    no overrides remain). Unknown kind / no registry ⇒ degraded no-op."""
    ctx = request.app.state.ctx
    spec = _KIND_MAP.get(kind)
    if spec is None:
        return GuidanceExtraWriteResponse(ok=False, kind=kind, degraded=True)
    override_key, _const_name, _mode = spec

    reg = _override_registry(ctx)
    if reg is None:
        return GuidanceExtraWriteResponse(ok=False, kind=kind, degraded=True)
    try:
        from mast.admin.override_store import GUIDANCE_OVERRIDES

        data = reg.get_raw(GUIDANCE_OVERRIDES) or {}
        if body.data in (None, {}, []):
            data.pop(override_key, None)
        else:
            data[override_key] = body.data
        # save_or_delete_and_reload persists (or deletes the file when empty)
        # THEN fires the registered reload hooks — same path the old sub-tab used.
        fired = reg.save_or_delete_and_reload(GUIDANCE_OVERRIDES, data)
        saved = (reg.get_raw(GUIDANCE_OVERRIDES) or {}).get(override_key)
        return GuidanceExtraWriteResponse(
            ok=True, kind=kind, reloaded=bool(fired), override=saved, degraded=False
        )
    except Exception as exc:
        logger.warning("guidance-extra write failed (%s): %s", kind, exc)
        return GuidanceExtraWriteResponse(ok=False, kind=kind, degraded=True)
