"""Pydantic request/response models for Domain C (skills detail + composites +
encyclopedia) of the typed API seam.

These mirror the data shapes already produced by ``mast.webui.builder_api``
(catalog card / validate report), ``mast.skills.composite.version_store``
(composite summary / version history) and ``mast.webui.encyclopedia`` (DOMAINS /
INTENT_MAPPING). They are the SINGLE SOURCE OF TYPES exported via
``/openapi.json`` and consumed by the frontend type generators — never redefine
a shape that the core already owns; import it.

Every response carries a ``degraded`` boolean: True means the live core
(SkillRegistry / composite store) was not wired or a call raised, so the body is
an empty-but-valid placeholder rather than a 500.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ── skill card (full detail for /api/skills/{name}) ──────────────────────────


class SkillParam(BaseModel):
    """One parameter row of a skill card (mirrors builder_api._param_dict /
    the registry SkillParameter)."""

    name: str
    type: str = ""
    description: str = ""
    unit: Optional[str] = None
    required: bool = False
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    allowed_values: Optional[list[Any]] = None


class SkillCard(BaseModel):
    """Full skill card — the on-demand detail behind the lightweight catalog
    index. Shape mirrors ``builder_api.build_catalog()['cards'][name]``.

    ``found`` is False (with ``degraded=True``) when the live registry is not
    wired or the name is unknown — the frontend shows an empty-but-not-broken
    state, never a 404 crash for the standalone dev seam.
    """

    name: str = ""
    zh: str = ""
    category: str = ""
    safety: str = ""
    level: int = 0
    source: str = "other"
    source_zh: str = ""
    tags: list[str] = Field(default_factory=list)
    domain: str = "其他"
    version: str = ""
    description: str = ""
    description_zh: str = ""
    parameters: list[SkillParam] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    estimated_duration_s: Optional[float] = None
    rollback_skill: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
    outputs: list[Any] = Field(default_factory=list)
    found: bool = False
    degraded: bool = False


# ── composites (list / detail / versions) ────────────────────────────────────


class CompositeSummary(BaseModel):
    """One row of the composite list (mirrors version_store.list_specs())."""

    name: str
    version: int = 1
    description: str = ""
    safety_level: str = "confirm"
    n_nodes: int = 0
    tags: list[Any] = Field(default_factory=list)


class CompositesResponse(BaseModel):
    composites: list[CompositeSummary] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


class CompositeVersionEntry(BaseModel):
    """One row of a composite's version history (mirrors
    version_store.list_versions())."""

    version: int
    saved_at: str = ""
    n_nodes: int = 0
    description: str = ""


class CompositeDetailResponse(BaseModel):
    """Full composite: its current spec (raw to_dict) + version history.

    ``found`` is False when the name is unknown; ``degraded`` is True when the
    file store could not be reached at all (import / IO failure)."""

    name: str = ""
    spec: Optional[dict[str, Any]] = None
    versions: list[CompositeVersionEntry] = Field(default_factory=list)
    found: bool = False
    degraded: bool = False


class CompositeVersionsResponse(BaseModel):
    name: str = ""
    versions: list[CompositeVersionEntry] = Field(default_factory=list)
    found: bool = False
    degraded: bool = False


# ── composite validate (design-time lint report) ─────────────────────────────


class ValidateStepReport(BaseModel):
    """Per-step lint result (mirrors builder_api.validate_spec_payload steps)."""

    id: str = "?"
    skill: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    spec: dict[str, Any] = Field(default_factory=dict)


class ValidateResponse(BaseModel):
    """Full design-time validation report (mirrors validate_spec_payload()).

    ``degraded`` True means the validator backend (composite spec module) could
    not be imported — the API returns ``ok=False`` with an explanatory problem
    rather than crashing."""

    ok: bool = False
    problems: list[str] = Field(default_factory=list)
    steps: list[ValidateStepReport] = Field(default_factory=list)
    degraded: bool = False


# ── composite write results (clone / restore) ────────────────────────────────


class CloneRequest(BaseModel):
    """Clone a stored composite as a fresh blueprint. ``new_name`` is the target
    name; ``author`` is an optional stamp."""

    new_name: str = ""
    author: str = ""


class CompositeWriteResult(BaseModel):
    """Typed result of a composite write path (clone / restore). When the live
    core is absent or a call raises, ``ok=False`` + ``degraded=True`` (never a
    500). Business logic + hot-registration + safety passthrough are wired to
    the live singletons at integration time — the API only calls into the core.
    """

    ok: bool = False
    name: str = ""
    version: Optional[int] = None
    message: str = ""
    error: Optional[str] = None
    degraded: bool = False


# ── encyclopedia (domains / intent mapping) ──────────────────────────────────


class EncyclopediaDomain(BaseModel):
    """One curated domain group (mirrors encyclopedia.DOMAINS)."""

    id: str
    name: str
    desc: str = ""
    skills: list[str] = Field(default_factory=list)
    agent: Optional[str] = None


class EncyclopediaDomainsResponse(BaseModel):
    domains: list[EncyclopediaDomain] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


class IntentMappingEntry(BaseModel):
    """One intent → skill recipe row (mirrors encyclopedia.INTENT_MAPPING)."""

    keywords: str = ""
    skill: str = ""
    note: str = ""


class IntentMappingResponse(BaseModel):
    mapping: list[IntentMappingEntry] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


__all__ = [
    "SkillParam",
    "SkillCard",
    "CompositeSummary",
    "CompositesResponse",
    "CompositeVersionEntry",
    "CompositeDetailResponse",
    "CompositeVersionsResponse",
    "ValidateStepReport",
    "ValidateRequest",
    "ValidateResponse",
    "CloneRequest",
    "CompositeWriteResult",
    "EncyclopediaDomain",
    "EncyclopediaDomainsResponse",
    "IntentMappingEntry",
    "IntentMappingResponse",
]
