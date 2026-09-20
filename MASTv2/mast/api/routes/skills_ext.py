"""Domain C of the typed API seam — skill detail + composites + encyclopedia.

Endpoints beyond the lightweight ``/api/skills/catalog`` index:

  GET  /api/skills/{name}                       full SkillCard
  GET  /api/composites                          list every stored composite
  GET  /api/composites/{name}                   one composite (spec + history)
  GET  /api/composites/{name}/versions          version history only
  POST /api/composites/{name}/validate          design-time lint report
  POST /api/composites/{name}/clone             copy a composite as a blueprint
  POST /api/composites/{name}/restore/{version} roll a composite back forward
  GET  /api/encyclopedia/domains                curated domain groups
  GET  /api/encyclopedia/intent-mapping         intent → skill recipes

GRACEFUL DEGRADATION is mandatory: this module must import and the API must
boot STANDALONE (no live core wired). Every handler LAZY-imports the heavy core
helpers (``mast.webui.builder_api`` for the catalog/validator, the composite file
store, ``mast.webui.encyclopedia`` for the curated data) INSIDE the handler,
wrapped in try/except — exactly like ``routes/skills.py``. On any absence or
error it returns a valid empty/degraded response (``degraded=True``); it NEVER
500s and NEVER crashes on import. Composite reads are file-backed and work
standalone, so they degrade only when the store itself is unreachable.

WRITE paths (clone / restore) are defined for contract completeness and also
degrade safely — they call straight into the core file store; the live-singleton
hot-(un)registration + safety passthrough are wired at integration
time. No business logic or safety checks live in this API layer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from mast.api.schemas_skills_ext import (
    CloneRequest,
    CompositeDetailResponse,
    CompositesResponse,
    CompositeSummary,
    CompositeVersionEntry,
    CompositeVersionsResponse,
    CompositeWriteResult,
    EncyclopediaDomain,
    EncyclopediaDomainsResponse,
    IntentMappingEntry,
    IntentMappingResponse,
    SkillCard,
    SkillParam,
    ValidateRequest,
    ValidateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills_ext"])


# ── helpers ──────────────────────────────────────────────────────────────────


def _card_from_raw(raw: dict) -> SkillCard:
    """Build a SkillCard from one builder_api catalog ``cards[name]`` dict."""
    params = [
        SkillParam(
            name=p.get("name", ""),
            type=p.get("type", ""),
            description=p.get("description", "") or "",
            unit=p.get("unit"),
            required=bool(p.get("required", False)),
            default=p.get("default"),
            min=p.get("min"),
            max=p.get("max"),
            allowed_values=p.get("allowed_values"),
        )
        for p in (raw.get("parameters") or [])
        if isinstance(p, dict)
    ]
    return SkillCard(
        name=raw.get("name", ""),
        zh=raw.get("zh", "") or "",
        category=raw.get("category", "") or "",
        safety=raw.get("safety", "") or "",
        level=int(raw.get("level", 0) or 0),
        source=raw.get("source", "other") or "other",
        source_zh=raw.get("source_zh", "") or "",
        tags=list(raw.get("tags") or []),
        domain=raw.get("domain", "其他") or "其他",
        version=str(raw.get("version", "") or ""),
        description=raw.get("description", "") or "",
        description_zh=raw.get("description_zh", "") or "",
        parameters=params,
        preconditions=list(raw.get("preconditions") or []),
        postconditions=list(raw.get("postconditions") or []),
        estimated_duration_s=raw.get("estimated_duration_s"),
        rollback_skill=raw.get("rollback_skill"),
        extra=dict(raw.get("extra") or {}),
        outputs=list(raw.get("outputs") or []),
        found=True,
        degraded=False,
    )


def _versions(store, name: str) -> list[CompositeVersionEntry]:
    out: list[CompositeVersionEntry] = []
    for v in store.list_versions(name) or []:
        out.append(
            CompositeVersionEntry(
                version=int(v.get("version", 0) or 0),
                saved_at=str(v.get("saved_at", "") or ""),
                n_nodes=int(v.get("n_nodes", 0) or 0),
                description=str(v.get("description", "") or ""),
            )
        )
    return out


# ── skill detail (full card) ─────────────────────────────────────────────────


@router.get("/skills/{name}", response_model=SkillCard)
def get_skill_card(name: str, request: Request) -> SkillCard:
    """Full skill card for one skill. Backed by the live SkillRegistry via
    builder_api's cached catalog; degrades to an empty card when unwired."""
    ctx = request.app.state.ctx
    if getattr(ctx, "skill_registry", None) is None:
        # Standalone dev / not yet wired — empty but not broken.
        return SkillCard(name=name, found=False, degraded=True)
    try:
        from mast.webui.builder_api import get_catalog  # type: ignore[attr-defined]

        raw = get_catalog().get("cards", {}).get(name)
        if not isinstance(raw, dict):
            return SkillCard(name=name, found=False, degraded=False)
        return _card_from_raw(raw)
    except Exception as exc:  # any wiring/shape mismatch → degrade, never 500
        logger.warning("skill card build failed for %s: %s", name, exc)
        return SkillCard(name=name, found=False, degraded=True)


# ── composites: list / detail / versions ─────────────────────────────────────


@router.get("/composites", response_model=CompositesResponse)
def list_composites(request: Request) -> CompositesResponse:
    """Every stored composite (file-backed; works standalone)."""
    try:
        from mast.webui.composite_panel import composite_store

        rows = composite_store().list_specs() or []
        composites = [
            CompositeSummary(
                name=str(r.get("name", "")),
                version=int(r.get("version", 1) or 1),
                description=str(r.get("description", "") or ""),
                safety_level=str(r.get("safety_level", "confirm") or "confirm"),
                n_nodes=int(r.get("n_nodes", 0) or 0),
                tags=list(r.get("tags") or []),
            )
            for r in rows
            if r.get("name")
        ]
        return CompositesResponse(
            composites=composites, count=len(composites), degraded=False
        )
    except Exception as exc:
        logger.warning("composite list failed: %s", exc)
        return CompositesResponse(degraded=True)


@router.get("/composites/{name}", response_model=CompositeDetailResponse)
def get_composite(name: str, request: Request) -> CompositeDetailResponse:
    """One composite: its current spec (raw dict) + version history."""
    try:
        from mast.webui.composite_panel import composite_store
        from mast.skills.composite.version_store import VersionStoreError

        store = composite_store()
        try:
            spec = store.load(name)
        except VersionStoreError:
            return CompositeDetailResponse(name=name, found=False, degraded=False)
        return CompositeDetailResponse(
            name=name,
            spec=spec.to_dict(),
            versions=_versions(store, name),
            found=True,
            degraded=False,
        )
    except Exception as exc:
        logger.warning("composite get failed for %s: %s", name, exc)
        return CompositeDetailResponse(name=name, found=False, degraded=True)


@router.get("/composites/{name}/versions", response_model=CompositeVersionsResponse)
def get_composite_versions(name: str, request: Request) -> CompositeVersionsResponse:
    """Version history of one composite (newest first)."""
    try:
        from mast.webui.composite_panel import composite_store
        from mast.skills.composite.version_store import VersionStoreError

        store = composite_store()
        if not store.exists(name):
            return CompositeVersionsResponse(name=name, found=False, degraded=False)
        try:
            versions = _versions(store, name)
        except VersionStoreError:
            return CompositeVersionsResponse(name=name, found=False, degraded=False)
        return CompositeVersionsResponse(
            name=name, versions=versions, found=True, degraded=False
        )
    except Exception as exc:
        logger.warning("composite versions failed for %s: %s", name, exc)
        return CompositeVersionsResponse(name=name, found=False, degraded=True)


# ── composite validate (design-time lint) ────────────────────────────────────


@router.post("/composites/{name}/validate", response_model=ValidateResponse)
async def validate_composite(
    name: str, request: Request, body: ValidateRequest
) -> ValidateResponse:
    """Design-time lint report for a composite spec (delegates to the core
    validator). The validator reuses the live registry to check skill existence
    /bounds; with no registry wired it still parses the spec and reports
    structural problems."""
    spec_dict = body.spec if isinstance(body.spec, dict) else {}
    try:
        from mast.webui.builder_api import validate_spec_payload

        report = validate_spec_payload(spec_dict)
        steps = report.get("steps") or []
        return ValidateResponse(
            ok=bool(report.get("ok", False)),
            problems=list(report.get("problems") or []),
            steps=[
                {
                    "id": str(s.get("id", "?")),
                    "skill": str(s.get("skill", "")),
                    "errors": list(s.get("errors") or []),
                    "warnings": list(s.get("warnings") or []),
                }
                for s in steps
                if isinstance(s, dict)
            ],
            degraded=False,
        )
    except Exception as exc:
        logger.warning("composite validate failed for %s: %s", name, exc)
        return ValidateResponse(
            ok=False,
            problems=[f"校验后端不可用：{exc}"],
            degraded=True,
        )


# ── composite write paths (clone / restore) — degrade-safe ───────────────────


@router.post("/composites/{name}/clone", response_model=CompositeWriteResult)
async def clone_composite(
    name: str, request: Request, body: CloneRequest
) -> CompositeWriteResult:
    """Clone composite ``name`` to a fresh blueprint ``new_name`` (v1).

    Calls straight into the core file store. Live-singleton hot-registration +
    safety passthrough are wired at integration time; this layer holds no
    business logic. Degrades to ``ok=false, degraded=true`` when the store is
    unreachable."""
    new_name = (body.new_name or "").strip()
    if not new_name:
        return CompositeWriteResult(
            ok=False, name=name, error="missing_new_name", degraded=False
        )
    try:
        from mast.webui.composite_panel import composite_store
        from mast.skills.composite.version_store import VersionStoreError
    except Exception as exc:
        logger.warning("composite clone backend unavailable: %s", exc)
        return CompositeWriteResult(ok=False, name=name, error=str(exc), degraded=True)
    try:
        cloned = composite_store().clone(name, new_name, author=body.author or "")
        return CompositeWriteResult(
            ok=True,
            name=cloned.name,
            version=cloned.version,
            message=f"已克隆 {name} → {cloned.name}（v{cloned.version}）。",
            degraded=False,
        )
    except VersionStoreError as exc:
        return CompositeWriteResult(ok=False, name=name, error=str(exc), degraded=False)
    except Exception as exc:
        logger.warning("composite clone failed for %s: %s", name, exc)
        return CompositeWriteResult(ok=False, name=name, error=str(exc), degraded=True)


@router.post(
    "/composites/{name}/restore/{version}", response_model=CompositeWriteResult
)
async def restore_composite(
    name: str, version: int, request: Request
) -> CompositeWriteResult:
    """Roll composite ``name`` back to ``version`` by writing it forward as a new
    version (non-destructive). Calls the core store; degrades safely."""
    try:
        from mast.webui.composite_panel import composite_store
        from mast.skills.composite.version_store import VersionStoreError
    except Exception as exc:
        logger.warning("composite restore backend unavailable: %s", exc)
        return CompositeWriteResult(ok=False, name=name, error=str(exc), degraded=True)
    try:
        saved = composite_store().restore(name, int(version))
        return CompositeWriteResult(
            ok=True,
            name=saved.name,
            version=saved.version,
            message=f"已回滚 {name} 到 v{version}（写为新版本 v{saved.version}）。",
            degraded=False,
        )
    except VersionStoreError as exc:
        return CompositeWriteResult(ok=False, name=name, error=str(exc), degraded=False)
    except Exception as exc:
        logger.warning("composite restore failed for %s v%s: %s", name, version, exc)
        return CompositeWriteResult(ok=False, name=name, error=str(exc), degraded=True)


# ── encyclopedia (curated, static) ───────────────────────────────────────────


@router.get("/encyclopedia/domains", response_model=EncyclopediaDomainsResponse)
def encyclopedia_domains(request: Request) -> EncyclopediaDomainsResponse:
    """Curated domain groups (encyclopedia.DOMAINS) with owning-agent annotation."""
    try:
        from mast.webui.encyclopedia import AGENT_BY_DOMAIN, DOMAINS

        domains = [
            EncyclopediaDomain(
                id=str(d.get("id", "")),
                name=str(d.get("name", "")),
                desc=str(d.get("desc", "") or ""),
                skills=list(d.get("skills") or []),
                agent=AGENT_BY_DOMAIN.get(d.get("id", ""), "instrument_control"),
            )
            for d in DOMAINS
            if d.get("id")
        ]
        return EncyclopediaDomainsResponse(
            domains=domains, count=len(domains), degraded=False
        )
    except Exception as exc:
        logger.warning("encyclopedia domains failed: %s", exc)
        return EncyclopediaDomainsResponse(degraded=True)


@router.get("/encyclopedia/intent-mapping", response_model=IntentMappingResponse)
def encyclopedia_intent_mapping(request: Request) -> IntentMappingResponse:
    """Intent-keyword → skill recipes (encyclopedia.INTENT_MAPPING)."""
    try:
        from mast.webui.encyclopedia import INTENT_MAPPING

        mapping = [
            IntentMappingEntry(
                keywords=str(m.get("keywords", "") or ""),
                skill=str(m.get("skill", "") or ""),
                note=str(m.get("note", "") or ""),
            )
            for m in INTENT_MAPPING
            if isinstance(m, dict)
        ]
        return IntentMappingResponse(
            mapping=mapping, count=len(mapping), degraded=False
        )
    except Exception as exc:
        logger.warning("encyclopedia intent mapping failed: %s", exc)
        return IntentMappingResponse(degraded=True)
