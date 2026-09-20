"""Read-only Codex / knowledge reference endpoints (TS-rewrite parity).

Re-exposes the static reference DATA whose only UI in the Gradio app was a set of
HTML builders (``gui/html_builders.build_workflows_html`` / ``build_guide_html``
/ ``build_experiment_strategies_html`` / ``build_hardware_reference_html`` /
``build_advisor_overview_html``) and the old admin "技能 → IC" sub-views
(``admin/tabs/knowledge/*``, ``admin/tabs/encyclopedia/hierarchy``). Those
builders read plain Python constants and had NO API; the TS UI needs the raw
structured data so it can render its own markup.

Single dispatch endpoint::

    GET /api/knowledge/codex/{view}

with ``view`` in {workflows, decision_trees, guide, advisor,
experiment_strategies, hardware_reference, intent_map}. Each view RELAYS the kept
core constants/functions read-only and reshapes nothing of substance (only the
``options`` tuples of decision trees and the derived advisor card fields are
flattened to JSON-friendly objects).

GRACEFUL DEGRADATION is mandatory (house rule): this router boots STANDALONE with
no live core wired. Every needed constant/module is LAZY-imported INSIDE the
handler in try/except; if absent or a call raises, the view returns a valid
degraded body (``degraded=True``) with empty payload fields — NEVER a 500. No
business/safety logic lives here; authority stays in the knowledge core.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mast.api.schemas_codex_reference import (
    AdvisorCategory,
    AdvisorResponse,
    DecisionNode,
    DecisionOption,
    DecisionTree,
    DecisionTreesResponse,
    ExperimentStrategiesResponse,
    GuideResponse,
    HardwareReferenceResponse,
    IntentEntry,
    IntentMapResponse,
    WorkflowRecipe,
    WorkflowsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["codex_reference"])

# View ids accepted by the dispatch endpoint (kept here so the unknown-view
# branch can advertise the valid set without importing handlers).
_VIEWS = (
    "workflows",
    "decision_trees",
    "guide",
    "advisor",
    "experiment_strategies",
    "hardware_reference",
    "intent_map",
)


# ── relay helpers (all lazy-import + never raise) ──────────────────────


def _load_skill_guidance() -> tuple[list, dict]:
    """``(WORKFLOW_RECIPES, DECISION_TREES)`` or empty defaults on absence."""
    try:
        from mast.knowledge.skill_guidance import DECISION_TREES, WORKFLOW_RECIPES

        return list(WORKFLOW_RECIPES or []), dict(DECISION_TREES or {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("codex_reference: skill_guidance unavailable: %s", exc)
        return [], {}


def _load_encyclopedia() -> tuple[dict, list]:
    """``(COMPOSITE_HIERARCHY, INTENT_MAPPING)`` or empty defaults on absence."""
    try:
        from mast.webui.encyclopedia import COMPOSITE_HIERARCHY, INTENT_MAPPING

        return dict(COMPOSITE_HIERARCHY or {}), list(INTENT_MAPPING or [])
    except Exception as exc:
        logger.warning("codex_reference: encyclopedia constants unavailable: %s", exc)
        return {}, []


def _to_recipes(raw: list) -> list[WorkflowRecipe]:
    out: list[WorkflowRecipe] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        out.append(
            WorkflowRecipe(
                name=str(r.get("name", "")),
                desc=str(r.get("desc", "")),
                chain=[str(s) for s in (r.get("chain") or [])],
                params=str(r.get("params", "")),
            )
        )
    return out


def _to_trees(raw: dict) -> list[DecisionTree]:
    out: list[DecisionTree] = []
    for tree_id, tree in (raw or {}).items():
        if not isinstance(tree, dict):
            continue
        nodes: list[DecisionNode] = []
        for node in tree.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            opts: list[DecisionOption] = []
            for pair in node.get("options") or []:
                # Source options are (label, skill) tuples.
                try:
                    label, skill = pair
                except (TypeError, ValueError):
                    label, skill = (str(pair), "")
                opts.append(DecisionOption(label=str(label), skill=str(skill)))
            nodes.append(DecisionNode(q=str(node.get("q", "")), options=opts))
        out.append(
            DecisionTree(
                id=str(tree_id),
                title=str(tree.get("title", tree_id)),
                nodes=nodes,
            )
        )
    return out


def _to_intents(raw: list) -> list[IntentEntry]:
    out: list[IntentEntry] = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        out.append(
            IntentEntry(
                keywords=str(e.get("keywords", "")),
                skill=str(e.get("skill", "")),
                note=str(e.get("note", "")),
            )
        )
    return out


# ── view builders ──────────────────────────────────────────────────────


def _view_workflows() -> WorkflowsResponse:
    recipes_raw, _ = _load_skill_guidance()
    hierarchy, _ = _load_encyclopedia()
    recipes = _to_recipes(recipes_raw)
    degraded = not recipes and not hierarchy
    return WorkflowsResponse(
        view="workflows",
        degraded=degraded,
        detail="no workflow data available" if degraded else None,
        recipes=recipes,
        composite_hierarchy=hierarchy,
    )


def _view_decision_trees() -> DecisionTreesResponse:
    _, trees_raw = _load_skill_guidance()
    hierarchy, _ = _load_encyclopedia()
    trees = _to_trees(trees_raw)
    degraded = not trees and not hierarchy
    return DecisionTreesResponse(
        view="decision_trees",
        degraded=degraded,
        detail="no decision-tree data available" if degraded else None,
        trees=trees,
        composite_hierarchy=hierarchy,
    )


def _view_guide() -> GuideResponse:
    _, trees_raw = _load_skill_guidance()
    hierarchy, intents_raw = _load_encyclopedia()
    trees = _to_trees(trees_raw)
    intents = _to_intents(intents_raw)
    degraded = not trees and not intents and not hierarchy
    return GuideResponse(
        view="guide",
        degraded=degraded,
        detail="no guide data available" if degraded else None,
        intent_mapping=intents,
        decision_trees=trees,
        composite_hierarchy=hierarchy,
    )


def _view_advisor() -> AdvisorResponse:
    try:
        from mast.knowledge import get_all_categories, get_materials_in
    except Exception as exc:
        logger.warning("codex_reference: advisor lookups unavailable: %s", exc)
        return AdvisorResponse(
            view="advisor", degraded=True, detail=str(exc), categories=[]
        )

    try:
        cats = get_all_categories() or []
    except Exception as exc:
        logger.warning("codex_reference: get_all_categories failed: %s", exc)
        return AdvisorResponse(
            view="advisor", degraded=True, detail=str(exc), categories=[]
        )

    out: list[AdvisorCategory] = []
    for cat in cats:
        if not isinstance(cat, dict):
            continue
        cat_id = str(cat.get("id", ""))
        try:
            materials = get_materials_in(cat_id) or {}
        except Exception:
            materials = {}
        out.append(
            AdvisorCategory(
                id=cat_id,
                name=str(cat.get("name", "")),
                name_en=str(cat.get("name_en", "")),
                description=str(cat.get("description", "")),
                completeness=str(cat.get("completeness", "stub")),
                n_phases=len(cat.get("phases", []) or []),
                materials=[str(m) for m in materials.keys()],
                raw=cat,
            )
        )

    degraded = not out
    return AdvisorResponse(
        view="advisor",
        degraded=degraded,
        detail="no sample-type categories loaded" if degraded else None,
        categories=out,
    )


def _view_experiment_strategies() -> ExperimentStrategiesResponse:
    try:
        from mast.knowledge.experiment_design import (
            ANOMALY_RESPONSE,
            MEASUREMENT_STRATEGIES,
            REFERENCE_EXPERIMENTS,
        )
    except Exception as exc:
        logger.warning("codex_reference: experiment_design unavailable: %s", exc)
        return ExperimentStrategiesResponse(
            view="experiment_strategies", degraded=True, detail=str(exc)
        )

    ms = dict(MEASUREMENT_STRATEGIES or {})
    re_ = dict(REFERENCE_EXPERIMENTS or {})
    ar = dict(ANOMALY_RESPONSE or {})
    degraded = not ms and not re_ and not ar
    return ExperimentStrategiesResponse(
        view="experiment_strategies",
        degraded=degraded,
        detail="no experiment-design data available" if degraded else None,
        measurement_strategies=ms,
        reference_experiments=re_,
        anomaly_response=ar,
    )


def _view_hardware_reference() -> HardwareReferenceResponse:
    try:
        from mast.knowledge.hardware_profile import (
            CONTROLLERS,
            MOTOR_TYPES,
            NOISE_FORMULAS,
            PIEZO_MATERIALS,
            PREAMPLIFIERS,
        )
    except Exception as exc:
        logger.warning("codex_reference: hardware_profile unavailable: %s", exc)
        return HardwareReferenceResponse(
            view="hardware_reference", degraded=True, detail=str(exc)
        )

    piezo = dict(PIEZO_MATERIALS or {})
    preamp = dict(PREAMPLIFIERS or {})
    ctrl = dict(CONTROLLERS or {})
    motors = dict(MOTOR_TYPES or {})
    noise = dict(NOISE_FORMULAS or {})
    degraded = not any((piezo, preamp, ctrl, motors, noise))
    return HardwareReferenceResponse(
        view="hardware_reference",
        degraded=degraded,
        detail="no hardware-reference data available" if degraded else None,
        piezo_materials=piezo,
        preamplifiers=preamp,
        controllers=ctrl,
        motor_types=motors,
        noise_formulas=noise,
    )


def _view_intent_map() -> IntentMapResponse:
    _, intents_raw = _load_encyclopedia()
    intents = _to_intents(intents_raw)

    constraints: list[dict] = []
    scan_speed_rule: dict = {}
    try:
        from mast.knowledge.safety_constraints import (
            SAFETY_CONSTRAINTS,
            SCAN_SPEED_RULE,
        )

        constraints = [c for c in (SAFETY_CONSTRAINTS or []) if isinstance(c, dict)]
        scan_speed_rule = dict(SCAN_SPEED_RULE or {})
    except Exception as exc:
        logger.warning("codex_reference: safety_constraints unavailable: %s", exc)

    degraded = not intents and not constraints and not scan_speed_rule
    return IntentMapResponse(
        view="intent_map",
        degraded=degraded,
        detail="no intent-map data available" if degraded else None,
        intent_mapping=intents,
        safety_constraints=constraints,
        scan_speed_rule=scan_speed_rule,
    )


_DISPATCH = {
    "workflows": _view_workflows,
    "decision_trees": _view_decision_trees,
    "guide": _view_guide,
    "advisor": _view_advisor,
    "experiment_strategies": _view_experiment_strategies,
    "hardware_reference": _view_hardware_reference,
    "intent_map": _view_intent_map,
}


# ── GET /api/knowledge/codex/{view} ────────────────────────────────────


@router.get(
    "/knowledge/codex/{view}",
    response_model=None,
    summary="Read-only Codex / knowledge reference data for one view.",
)
def get_codex_reference(view: str, request: Request):
    """Return the reference data for ``view`` as raw structured JSON.

    ``view`` selects one of the kept knowledge constants/lookups; the body shape
    is the per-view model in ``schemas_codex_reference``. An unknown view returns
    a 404 (typed) — every KNOWN view degrades gracefully (``degraded=True`` with
    empty payload) rather than 500-ing when its backing constant is absent.

    ``request.app.state.ctx`` is read for house-style parity; these views are
    process-global static reference data and need no live subsystem, so a missing
    ctx never affects the result."""
    # House-style: touch ctx (no live subsystem is required for static data).
    _ = getattr(request.app.state, "ctx", None)

    builder = _DISPATCH.get(view)
    if builder is None:
        return JSONResponse(
            status_code=404,
            content={
                "view": view,
                "degraded": True,
                "detail": f"unknown view (expected one of {list(_VIEWS)})",
            },
        )

    try:
        return builder()
    except Exception as exc:  # pragma: no cover - last-resort guard
        logger.exception("codex_reference: view %r failed unexpectedly", view)
        return JSONResponse(
            status_code=200,
            content={"view": view, "degraded": True, "detail": str(exc)},
        )
