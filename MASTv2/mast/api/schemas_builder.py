"""Pydantic request/response models for the composite/builder WRITE seam.

The read side of composites (list / detail / versions / validate / clone /
restore) already lives in ``schemas_skills_ext.py`` + ``routes/skills_ext.py``.
This module adds ONLY the shapes for the WRITE endpoints the new BuilderPage
needs to save / create / delete a composite (the "不能用" blockers):

  * POST   /api/composites               create a new composite from a spec
  * PUT    /api/composites/{name}         overwrite an existing composite (new version)
  * DELETE /api/composites/{name}         delete a composite (history retained)
  * POST   /api/builder/validate          name-agnostic design-time lint
  * GET    /api/builder/catalog           paginated builder skill catalog

These mirror the data shapes produced by the kept core
``mast.webui.builder_api`` (validate report / catalog index) and
``mast.skills.composite.version_store`` (save → new version, _history
snapshots). No business/safety logic lives here — the API only relays.

Every response carries a ``degraded`` boolean: True means the live core
(composite store / SkillRegistry / validator) was not wired or a call raised,
so the body is an empty-but-valid placeholder rather than a 500.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ── per-step lint report (mirrors validate_spec_payload steps) ────────────────


class BuilderValidateStep(BaseModel):
    """Per-step lint result (mirrors builder_api.validate_spec_payload steps)."""

    id: str = "?"
    skill: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ── create / update (save a builder spec as a composite) ──────────────────────


class CompositeCreateRequest(BaseModel):
    """Publish a builder spec as a NEW composite. ``name`` is the target name
    (must equal ``spec['name']`` when both are given; ``name`` wins if spec omits
    it). ``spec`` is the raw CompositeSpec dict from the editor canvas."""

    name: str = ""
    spec: dict[str, Any] = Field(default_factory=dict)


class CompositeUpdateRequest(BaseModel):
    """Overwrite an existing composite (writes a new version, like the old save).

    ``base_version`` (optional) enables optimistic concurrency: pass the version
    the editor LOADED and the save is rejected with ``version_conflict`` if the
    stored composite changed underneath. ``None`` = last-write-wins."""

    spec: dict[str, Any] = Field(default_factory=dict)
    base_version: Optional[int] = None


class CompositeSaveResult(BaseModel):
    """Typed result of a create/update (save) write path.

    On success ``ok=True`` with the new ``version`` (the store bumped + archived
    a ``_history/<name>.vN.json`` snapshot). On a rejected/invalid spec ``ok``
    is False with ``error`` + the design-time ``report`` (problems/steps) — the
    same fail-closed contract as the old builder save. On a missing backend or
    unexpected failure ``ok=False`` + ``degraded=True`` (never a 500)."""

    ok: bool = False
    name: str = ""
    version: Optional[int] = None
    message: str = ""
    error: Optional[str] = None
    # Optimistic-concurrency: the version currently stored (on version_conflict).
    stored_version: Optional[int] = None
    hot_registered: bool = False
    # Design-time validation report echoed back (problems + per-step lint).
    problems: list[str] = Field(default_factory=list)
    steps: list[BuilderValidateStep] = Field(default_factory=list)
    degraded: bool = False


class CompositeDeleteResult(BaseModel):
    """Typed result of a delete. ``ok=True`` once the current spec is removed
    (history snapshots are retained by the store). ``found`` is False when the
    name does not exist; ``degraded`` True when the store is unreachable."""

    ok: bool = False
    name: str = ""
    found: bool = True
    message: str = ""
    error: Optional[str] = None
    degraded: bool = False


# ── builder validate (name-agnostic design-time lint) ─────────────────────────


class BuilderValidateRequest(BaseModel):
    spec: dict[str, Any] = Field(default_factory=dict)


class BuilderValidateResponse(BaseModel):
    """Full design-time validation report (mirrors validate_spec_payload()).

    ``degraded`` True means the validator backend could not be imported — the
    API returns ``ok=False`` with an explanatory problem rather than crashing."""

    ok: bool = False
    problems: list[str] = Field(default_factory=list)
    steps: list[BuilderValidateStep] = Field(default_factory=list)
    degraded: bool = False


# ── builder catalog (paginated skill index for the palette) ───────────────────


class BuilderCatalogEntry(BaseModel):
    """One lightweight index row (mirrors builder_api.build_catalog index entry —
    richer than /api/skills/catalog: carries source/source_zh/domain/level)."""

    name: str = ""
    zh: str = ""
    category: str = ""
    safety: str = ""
    level: int = 0
    source: str = "other"
    source_zh: str = ""
    tags: list[str] = Field(default_factory=list)
    domain: str = "其他"

    #: 在用户的订阅面上吗（未定制时恒 True）。**join 在路由层，不进目录缓存** ——
    #: 订阅的变化频率远高于技能集合，进了缓存就得每次订阅写都把整份目录作废。
    subscribed: bool = True
    #: 必装项（界面上该禁用退订开关）。名单真源是
    #: ``mast.skills.subscription.MANDATORY_SKILLS``，这里只是把它带到前端。
    mandatory: bool = False


class BuilderCatalogResponse(BaseModel):
    """Paginated builder palette catalog (mirrors builder_api ``_catalog``
    route). ``degraded`` True when the live registry is unwired/raises — empty
    list, never a 500."""

    total: int = 0
    page: int = 1
    skills: list[BuilderCatalogEntry] = Field(default_factory=list)
    degraded: bool = False


__all__ = [
    "BuilderValidateStep",
    "CompositeCreateRequest",
    "CompositeUpdateRequest",
    "CompositeSaveResult",
    "CompositeDeleteResult",
    "BuilderValidateRequest",
    "BuilderValidateResponse",
    "BuilderCatalogEntry",
    "BuilderCatalogResponse",
]
