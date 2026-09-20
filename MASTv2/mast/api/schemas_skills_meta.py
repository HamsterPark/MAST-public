"""Pydantic request/response models for the residual READ-ONLY parity gaps —
skill metadata WRITE + effective safety checks + read-only knowledge reference
constants (TS-rewrite parity rebuild follow-up).

These three surfaces close the gaps flagged in the parity rebuild:

* **POST /api/skills/{name}/override** — the live "Skills Management" tab
  (admin/tabs/skills_tab.py) persisted per-skill metadata overrides
  (params / preconditions / postconditions / estimated_duration_s /
  rollback_skill / safety_level) into the SKILL override category of the
  ConfigOverrideRegistry (skill_overrides.json). The API is a THIN relay: it
  forwards the override dict into the registry and lets CORE own the merge +
  validation + safety derivation (R6). No merge/safety logic lives here.

* **GET /api/safety/checks/effective** — the live "全局检查规则" sub-tab
  (admin/tabs/safety/checks.py) showed _GLOBAL_CHECKS (code defaults) merged
  with the override layer, marking each row "modified default vs new addition"
  via the admin-override dot. This endpoint returns the same effective merge so
  the UI can render the dot WITHOUT the merge living in the frontend.

* **GET /api/knowledge/reference/{kind}** — the live Knowledge tab surfaced a
  handful of READ-ONLY reference constants (material safety constraints, scan
  speed rule, constant-height prerequisites, reference experiments, anomaly
  response protocols). This endpoint relays them verbatim, read-only.

EVERY response carries ``degraded`` so a standalone API process (no live core /
no admin module importable) returns an empty-but-valid body instead of 500-ing.
No safety / override business logic lives here — these models only carry data.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ── 1. Skill metadata override (WRITE) ───────────────────────────────────────

# The override fields the Skills Management tab could edit. All optional: only
# the fields the caller actually changed ride along (mirrors _collect_override
# in admin/tabs/skills_tab.py which emits only the differing keys).
SkillOverrideField = Literal[
    "parameters",
    "preconditions",
    "postconditions",
    "estimated_duration_s",
    "rollback_skill",
    "safety_level",
    "description",
]


class SkillOverrideRequest(BaseModel):
    """One skill's metadata override payload.

    Shape matches what admin/tabs/skills_tab.py persisted into the per-skill
    entry of skill_overrides.json (the SKILL override category):

      * ``parameters``  — {param_name: {field: value, ...}} (only differing
        fields per param; field set = type/unit/required/default/min_value/
        max_value/allowed_values/description)
      * ``preconditions`` / ``postconditions`` — list[str]
      * ``estimated_duration_s`` — float
      * ``rollback_skill`` — str | None
      * ``safety_level`` — str ("auto"/"confirm"/"dangerous")
      * ``description`` — str

    The API forwards this verbatim; the merge + SafetyLevel parsing + reload is
    the registry's / core's job (apply_skill_metadata_override)."""

    parameters: Optional[dict[str, dict[str, Any]]] = None
    preconditions: Optional[list[str]] = None
    postconditions: Optional[list[str]] = None
    estimated_duration_s: Optional[float] = None
    rollback_skill: Optional[str] = None
    safety_level: Optional[str] = None
    description: Optional[str] = None

    def to_override_dict(self) -> dict[str, Any]:
        """Collect only the explicitly-set fields into the override dict.

        ``rollback_skill=None`` is a legitimate clearing value, so we only drop
        fields the caller never sent (tracked by ``model_fields_set``)."""
        out: dict[str, Any] = {}
        for field in self.model_fields_set:
            out[field] = getattr(self, field)
        return out


class SkillOverrideResponse(BaseModel):
    """Result of persisting (or clearing) one skill's override."""

    ok: bool = False
    name: str
    reloaded: bool = False
    # The skill's full override payload as stored after the write (empty if the
    # write cleared it / degraded).
    override: dict[str, Any] = Field(default_factory=dict)
    fields: list[str] = Field(default_factory=list)
    degraded: bool = False


# ── 2. Effective safety checks (READ, merged) ────────────────────────────────


class EffectiveCheckRow(BaseModel):
    """One effective global-check rule = (pattern, unit, min_attr, max_attr)
    with provenance vs the code defaults.

    ``origin`` mirrors the admin-override dot logic in
    admin/tabs/safety/checks.py:
      * ``default``  — identical to a code default (no dot)
      * ``modified`` — a code-default pattern with overridden fields (dot)
      * ``addition`` — a pattern not present in _GLOBAL_CHECKS (dot)
    ``overridden`` is True for modified/addition (the dot)."""

    pattern: str
    unit: str
    min_attr: str
    max_attr: str
    origin: Literal["default", "modified", "addition"] = "default"
    overridden: bool = False


class EffectiveChecksResponse(BaseModel):
    """The full effective global-check rule set (code defaults merged with the
    override layer), plus the raw default set for reference."""

    rows: list[EffectiveCheckRow] = Field(default_factory=list)
    count: int = 0
    has_override: bool = False
    degraded: bool = False


# ── 3. Knowledge reference constants (READ-only) ─────────────────────────────

# The read-only reference kinds the live Knowledge tab surfaced. Each maps to a
# top-level constant in mast.knowledge.{safety_constraints,experiment_design}.
KnowledgeReferenceKind = Literal[
    "safety_constraints",
    "material_bias_limits",
    "material_current_limits",
    "tip_type_limits",
    "scan_speed_rule",
    "constant_height_prerequisites",
    "reference_experiments",
    "anomaly_response",
    "measurement_strategies",
]


class KnowledgeReferenceResponse(BaseModel):
    """One read-only reference constant relayed verbatim from the knowledge base.

    ``data`` is the raw constant (a dict or list depending on ``kind``). It is
    NEVER merged with overrides — these are advisory code constants the admin
    tab showed read-only."""

    kind: str
    data: Any = None
    count: int = 0
    degraded: bool = False
