"""Pydantic response models for the read-only Codex / knowledge reference slice.

These shapes expose static reference DATA whose only UI in the Gradio app was a
set of HTML builders (``gui/html_builders.build_workflows_html`` /
``build_guide_html`` / ``build_experiment_strategies_html`` /
``build_hardware_reference_html`` / ``build_advisor_overview_html``) plus the
old admin "技能 → IC" sub-views (``admin/tabs/knowledge/*``,
``admin/tabs/encyclopedia/hierarchy``). The TS UI renders its own markup, so the
API returns the RAW structured constants from the kept Python core — never HTML:

* ``workflows``               — ``WORKFLOW_RECIPES`` (+ ``COMPOSITE_HIERARCHY``)
                                from ``knowledge.skill_guidance`` / ``webui.encyclopedia``.
* ``decision_trees``          — ``DECISION_TREES`` (+ ``COMPOSITE_HIERARCHY``).
* ``guide``                   — the LLM decision guide: intent map + decision
                                trees + composite hierarchy in one payload.
* ``advisor``                 — sample-type categories via ``knowledge.lookups``
                                (``get_all_categories`` / ``get_materials_in``).
* ``experiment_strategies``   — ``MEASUREMENT_STRATEGIES`` / ``REFERENCE_EXPERIMENTS``
                                / ``ANOMALY_RESPONSE`` from ``knowledge.experiment_design``.
* ``hardware_reference``      — ``PIEZO_MATERIALS`` / ``PREAMPLIFIERS`` /
                                ``CONTROLLERS`` / ``MOTOR_TYPES`` / ``NOISE_FORMULAS``
                                from ``knowledge.hardware_profile``.
* ``intent_map``              — ``INTENT_MAPPING`` (+ ``SAFETY_CONSTRAINTS`` /
                                ``SCAN_SPEED_RULE`` from ``knowledge.safety_constraints``).

Per the house rules this file is the SINGLE SOURCE OF TYPES for this slice. Every
endpoint has a ``response_model`` and every body carries ``degraded`` so the
frontend can render an empty-but-not-broken state when a constant/module is
absent. The payloads are deliberately permissive (``Any``-valued maps/lists):
the source constants are large hand-authored reference dicts that evolve, and
this layer only RELAYS them read-only — it never reshapes or validates content.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ── shared base ────────────────────────────────────────────────────────


class CodexViewBase(BaseModel):
    """Common envelope for every Codex reference view.

    ``view`` echoes the requested view id. ``degraded`` is ``True`` whenever a
    backing constant/module was absent or a relay call raised — the typed fields
    then carry empty defaults instead of a 500. ``detail`` optionally explains
    the degradation (for diagnostics; the UI may ignore it)."""

    view: str
    degraded: bool = False
    detail: Optional[str] = None


# ── view: workflows ────────────────────────────────────────────────────


class WorkflowRecipe(BaseModel):
    """One workflow recipe card (mirrors a ``WORKFLOW_RECIPES`` entry)."""

    name: str = ""
    desc: str = ""
    chain: list[str] = Field(default_factory=list)
    params: str = ""


class WorkflowsResponse(CodexViewBase):
    """``GET /api/knowledge/codex/workflows`` — recipe cards + composite map."""

    recipes: list[WorkflowRecipe] = Field(default_factory=list)
    composite_hierarchy: dict[str, list[str]] = Field(default_factory=dict)


# ── view: decision_trees ───────────────────────────────────────────────


class DecisionOption(BaseModel):
    """A single (label → skill) branch of a decision-tree node."""

    label: str = ""
    skill: str = ""


class DecisionNode(BaseModel):
    """One question node of a decision tree."""

    q: str = ""
    options: list[DecisionOption] = Field(default_factory=list)


class DecisionTree(BaseModel):
    """A named decision tree (mirrors a ``DECISION_TREES`` value)."""

    id: str = ""
    title: str = ""
    nodes: list[DecisionNode] = Field(default_factory=list)


class DecisionTreesResponse(CodexViewBase):
    """``GET /api/knowledge/codex/decision_trees`` — decision trees + composite map."""

    trees: list[DecisionTree] = Field(default_factory=list)
    composite_hierarchy: dict[str, list[str]] = Field(default_factory=dict)


# ── view: guide ────────────────────────────────────────────────────────


class IntentEntry(BaseModel):
    """One intent → skill mapping row (mirrors an ``INTENT_MAPPING`` entry)."""

    keywords: str = ""
    skill: str = ""
    note: str = ""


class GuideResponse(CodexViewBase):
    """``GET /api/knowledge/codex/guide`` — the full LLM decision guide.

    Combines the three sections the old ``build_guide_html`` rendered: the
    intent→skill mapping table, the decision trees, and the composite hierarchy.
    """

    intent_mapping: list[IntentEntry] = Field(default_factory=list)
    decision_trees: list[DecisionTree] = Field(default_factory=list)
    composite_hierarchy: dict[str, list[str]] = Field(default_factory=dict)


# ── view: advisor ──────────────────────────────────────────────────────


class AdvisorCategory(BaseModel):
    """A sample-type category overview card.

    Mirrors a ``knowledge.lookups.get_all_categories`` entry plus the derived
    material name list / counts the old ``build_advisor_overview_html`` showed.
    ``raw`` carries the full source dict for any field the TS UI wants."""

    id: str = ""
    name: str = ""
    name_en: str = ""
    description: str = ""
    completeness: str = "stub"
    n_phases: int = 0
    materials: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class AdvisorResponse(CodexViewBase):
    """``GET /api/knowledge/codex/advisor`` — sample-type category overview."""

    categories: list[AdvisorCategory] = Field(default_factory=list)


# ── view: experiment_strategies ────────────────────────────────────────


class ExperimentStrategiesResponse(CodexViewBase):
    """``GET /api/knowledge/codex/experiment_strategies``.

    Relays the three hand-authored reference dicts verbatim (kept as permissive
    maps; the TS UI renders strategy cards + reference-experiment + anomaly
    tables)."""

    measurement_strategies: dict[str, Any] = Field(default_factory=dict)
    reference_experiments: dict[str, Any] = Field(default_factory=dict)
    anomaly_response: dict[str, Any] = Field(default_factory=dict)


# ── view: hardware_reference ───────────────────────────────────────────


class HardwareReferenceResponse(CodexViewBase):
    """``GET /api/knowledge/codex/hardware_reference``.

    Relays the STM hardware reference constants verbatim (piezo materials,
    preamplifiers, controllers, motors, noise formulas)."""

    piezo_materials: dict[str, Any] = Field(default_factory=dict)
    preamplifiers: dict[str, Any] = Field(default_factory=dict)
    controllers: dict[str, Any] = Field(default_factory=dict)
    motor_types: dict[str, Any] = Field(default_factory=dict)
    noise_formulas: dict[str, Any] = Field(default_factory=dict)


# ── view: intent_map ───────────────────────────────────────────────────


class IntentMapResponse(CodexViewBase):
    """``GET /api/knowledge/codex/intent_map``.

    The intent→skill mapping plus the advisory safety reference (constraints +
    the scan-speed rule) the old admin reference tables surfaced."""

    intent_mapping: list[IntentEntry] = Field(default_factory=list)
    safety_constraints: list[dict[str, Any]] = Field(default_factory=list)
    scan_speed_rule: dict[str, Any] = Field(default_factory=dict)
