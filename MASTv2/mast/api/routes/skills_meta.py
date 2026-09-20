"""Residual READ-ONLY parity gaps — skill metadata WRITE + effective safety
checks + read-only knowledge reference constants (TS-rewrite parity rebuild
follow-up).

Closes the three gaps flagged in the parity rebuild where the old admin GUI had
write / effective-merge / reference surfaces that the read-only API mirror was
missing:

  1. POST /api/skills/{name}/override
       Persist a per-skill metadata override (params / preconditions /
       postconditions / estimated_duration_s / rollback_skill / safety_level)
       into the SKILL override category of ConfigOverrideRegistry
       (skill_overrides.json). Mirrors admin/tabs/skills_tab.py:_on_save, which
       merged the override into the whole-file dict + save_and_reload. THIN
       relay only — the merge + SafetyLevel parsing + reload-hook fan-out is the
       registry's / core's job (apply_skill_metadata_override). NO merge / safety
       logic here (R6).

  2. GET /api/safety/checks/effective
       Return code-default _GLOBAL_CHECKS merged with the override layer, each
       row tagged default / modified / addition so the UI can render the
       "modified default vs new addition" override dot. Mirrors the merge in
       admin/tabs/safety/checks.py:_effective_checks (which the old HTML table
       used) — re-implemented over the same (overrides / additions / removals)
       override format so the merge does not live in the frontend.

  3. GET /api/knowledge/reference/{kind}
       Relay read-only reference constants the admin Knowledge tab showed
       (material safety constraints, scan speed rule, constant-height
       prerequisites, reference experiments, anomaly-response protocols, …) from
       mast.knowledge.{safety_constraints,experiment_design}. Verbatim, never
       merged with overrides (advisory code constants).

HOUSE STYLE (mirrors routes/admin.py): module-level ``router``; every handler
takes ``request: Request`` and reads ``ctx = request.app.state.ctx``;
``response_model`` on every handler; GRACEFUL DEGRADATION (boots standalone,
returns ``degraded=true`` — never 500); heavy backends LAZY-imported inside the
handler in try/except.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_skills_meta import (
    EffectiveCheckRow,
    EffectiveChecksResponse,
    KnowledgeReferenceResponse,
    SkillOverrideRequest,
    SkillOverrideResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills-meta"])

# skill_overrides.json — the SKILL override category. Named here (not imported
# from admin.override_store at module load) to stay degrade-safe on import.
_SKILL_OVERRIDES_FILE = "skill_overrides.json"

# Reference kind → (module, attribute). Single source for the read-only relay so
# the route and the schema Literal agree. Lazy-imported per request.
_REFERENCE_CONSTANTS: dict[str, tuple[str, str]] = {
    "safety_constraints": ("mast.knowledge.safety_constraints", "SAFETY_CONSTRAINTS"),
    "material_bias_limits": ("mast.knowledge.safety_constraints", "MATERIAL_BIAS_LIMITS"),
    "material_current_limits": (
        "mast.knowledge.safety_constraints",
        "MATERIAL_CURRENT_LIMITS",
    ),
    "tip_type_limits": ("mast.knowledge.safety_constraints", "TIP_TYPE_LIMITS"),
    "scan_speed_rule": ("mast.knowledge.safety_constraints", "SCAN_SPEED_RULE"),
    "constant_height_prerequisites": (
        "mast.knowledge.safety_constraints",
        "CONSTANT_HEIGHT_PREREQUISITES",
    ),
    "reference_experiments": (
        "mast.knowledge.experiment_design",
        "REFERENCE_EXPERIMENTS",
    ),
    "anomaly_response": ("mast.knowledge.experiment_design", "ANOMALY_RESPONSE"),
    "measurement_strategies": (
        "mast.knowledge.experiment_design",
        "MEASUREMENT_STRATEGIES",
    ),
}


def _override_registry(ctx: Any):
    """Best-effort handle to a live ConfigOverrideRegistry.

    Prefers one already wired onto the context (set at integration
    time); otherwise None. We do NOT construct one in standalone mode — that
    would touch the shared ``config/overrides`` dir from a dev process. Absent ⇒
    degrade. Mirrors routes/admin.py:_override_registry."""
    return getattr(ctx, "override_registry", None)


# ── 1. Skill metadata override (WRITE) ───────────────────────────────────────
@router.post("/skills/{name}/override", response_model=SkillOverrideResponse)
def write_skill_override(
    request: Request, name: str, body: SkillOverrideRequest
) -> SkillOverrideResponse:
    """Persist a per-skill metadata override into the SKILL override category.

    Forwards the override dict into the whole-file skill_overrides.json (one
    entry per skill, keyed by name) and triggers the core hot-reload. An empty
    payload for a skill that has no other fields ⇒ the skill's entry is cleared.
    The merge / SafetyLevel parsing / reload fan-out is the registry's job — the
    API only forwards the bytes (R6). Degrade-safe: no live registry ⇒ typed
    no-op with ``degraded=true``."""
    ctx = request.app.state.ctx
    reg = _override_registry(ctx)
    if reg is None:
        return SkillOverrideResponse(ok=False, name=name, degraded=True)
    try:
        override = body.to_override_dict()
        # Merge into the whole-file dict (mirrors skills_tab.py:_on_save which
        # read get_all_skill_overrides(), set [name], then save_and_reload).
        all_overrides = dict(reg.get_all_skill_overrides() or {})
        if override:
            all_overrides[name] = override
            # save_or_delete_and_reload persists the whole file (deletes it only
            # when the file becomes empty) THEN fires the registered reload hooks
            # (SkillRegistry metadata cache / SafetyGate derived caches).
            fired = reg.save_or_delete_and_reload(_SKILL_OVERRIDES_FILE, all_overrides)
        else:
            # Empty payload ⇒ clear this skill's entry (revert to code defaults).
            all_overrides.pop(name, None)
            fired = reg.save_or_delete_and_reload(_SKILL_OVERRIDES_FILE, all_overrides)
        stored = dict((reg.get_all_skill_overrides() or {}).get(name, {}))
        return SkillOverrideResponse(
            ok=True,
            name=name,
            reloaded=bool(fired),
            override=stored,
            fields=sorted(stored.keys()),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("skill override write failed (%s): %s", name, exc)
        return SkillOverrideResponse(ok=False, name=name, degraded=True)


# ── 2. Effective safety checks (READ, merged) ────────────────────────────────
@router.get("/safety/checks/effective", response_model=EffectiveChecksResponse)
def get_effective_checks(request: Request) -> EffectiveChecksResponse:
    """Return _GLOBAL_CHECKS merged with the override layer, row-tagged.

    Mirrors admin/tabs/safety/checks.py:_effective_checks: walk the code-default
    rules applying overrides + removals, then append additions. Each row is
    tagged default / modified / addition so the UI can show the override dot.
    Degrade-safe: if the safety module or the override store can't be imported we
    fall back to whatever we can read (code defaults if available, else empty +
    degraded)."""
    ctx = request.app.state.ctx

    # Code-default rules — without these we cannot build anything meaningful.
    try:
        from mast.core.safety import _GLOBAL_CHECKS

        defaults = list(_GLOBAL_CHECKS)
    except Exception as exc:
        logger.warning("effective checks: code defaults unavailable: %s", exc)
        return EffectiveChecksResponse(degraded=True)

    # Override layer (overrides / additions / removals). A live registry wired on
    # the context is preferred; otherwise the merge is just the code defaults.
    ovr: dict[str, Any] = {}
    has_override = False
    reg = _override_registry(ctx)
    if reg is not None:
        try:
            ovr = dict(reg.get_safety_checks() or {})
            has_override = bool(ovr)
        except Exception as exc:
            logger.warning("effective checks: override read failed: %s", exc)
            ovr = {}

    override_list = ovr.get("overrides", []) or []
    additions = ovr.get("additions", []) or []
    removals = set(ovr.get("removals", []) or [])
    override_map = {
        r["pattern"]: r for r in override_list if isinstance(r, dict) and "pattern" in r
    }

    rows: list[EffectiveCheckRow] = []
    for pattern, unit, min_a, max_a in defaults:
        if pattern in removals:
            continue
        if pattern in override_map:
            o = override_map[pattern]
            rows.append(
                EffectiveCheckRow(
                    pattern=str(o.get("pattern", pattern)),
                    unit=str(o.get("unit", unit)),
                    min_attr=str(o.get("min_attr", min_a)),
                    max_attr=str(o.get("max_attr", max_a)),
                    origin="modified",
                    overridden=True,
                )
            )
        else:
            rows.append(
                EffectiveCheckRow(
                    pattern=pattern,
                    unit=unit,
                    min_attr=min_a,
                    max_attr=max_a,
                    origin="default",
                    overridden=False,
                )
            )
    for a in additions:
        if not isinstance(a, dict):
            continue
        try:
            rows.append(
                EffectiveCheckRow(
                    pattern=str(a["pattern"]),
                    unit=str(a["unit"]),
                    min_attr=str(a["min_attr"]),
                    max_attr=str(a["max_attr"]),
                    origin="addition",
                    overridden=True,
                )
            )
        except (KeyError, TypeError):
            logger.warning("effective checks: dropping malformed addition: %r", a)

    return EffectiveChecksResponse(
        rows=rows,
        count=len(rows),
        has_override=has_override,
        degraded=False,
    )


# ── 3. Knowledge reference constants (READ-only) ─────────────────────────────
@router.get(
    "/knowledge/reference/{kind}", response_model=KnowledgeReferenceResponse
)
def get_knowledge_reference(request: Request, kind: str) -> KnowledgeReferenceResponse:
    """Relay one read-only knowledge reference constant verbatim.

    The constant is advisory code data the admin Knowledge tab showed read-only;
    it is NEVER merged with overrides. Unknown ``kind`` ⇒ degraded (empty). A
    knowledge-module import failure ⇒ degraded (never 500)."""
    target = _REFERENCE_CONSTANTS.get(kind)
    if target is None:
        return KnowledgeReferenceResponse(kind=kind, degraded=True)
    module_name, attr = target
    try:
        import importlib

        module = importlib.import_module(module_name)
        data = getattr(module, attr, None)
        if data is None:
            return KnowledgeReferenceResponse(kind=kind, degraded=True)
        try:
            count = len(data)
        except TypeError:
            count = 0
        return KnowledgeReferenceResponse(
            kind=kind, data=data, count=count, degraded=False
        )
    except Exception as exc:
        logger.warning("knowledge reference read failed (%s): %s", kind, exc)
        return KnowledgeReferenceResponse(kind=kind, degraded=True)
