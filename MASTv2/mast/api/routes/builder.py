"""Composite/builder WRITE seam — the save/create/delete the new BuilderPage needs.

The READ side of composites (list / detail / versions / validate / clone /
restore) already lives in ``routes/skills_ext.py``. What was missing — and what
made the BuilderPage "不能用" — is the ability to PUBLISH a builder spec back as a
composite skill: create, overwrite (new version), and delete. This module adds
exactly those, plus a name-agnostic ``/api/builder/validate`` and a richer
``/api/builder/catalog`` for the palette.

Endpoints:
  POST   /api/composites               create a new composite from a builder spec
  PUT    /api/composites/{name}         overwrite an existing composite (new version)
  DELETE /api/composites/{name}         delete a composite (history retained)
  POST   /api/builder/validate          name-agnostic design-time lint report
  GET    /api/builder/catalog           paginated builder skill catalog (palette)

THIN RELAY ONLY. Each handler calls straight into the kept core:
``mast.webui.builder_api`` (``validate_spec_payload`` / ``resolved_skills_snapshot``
/ ``save_extra_meta`` / ``get_catalog`` / ``filter_index`` / ``invalidate_catalog``)
and ``mast.webui.composite_panel`` (``composite_store`` + the live-singleton
hot-(un)register helpers ``_hot_register`` / ``_hot_unregister``). The
``_history`` versioning + the optimistic-concurrency (``base_version`` →
``VersionConflictError``) gate are owned by ``version_store.save`` and preserved
unchanged. No validation/safety logic is reimplemented here.

GRACEFUL DEGRADATION is mandatory (house rule): this router imports + the API
boots STANDALONE. Every handler LAZY-imports the heavy core INSIDE a try/except
and returns a valid degraded body (``degraded=True``) on any absence or failure —
it NEVER 500s. An invalid spec is NOT degradation: it returns ``ok=False`` with
the design-time report (fail-closed, mirroring the old builder save).

NOTE — deliberately NOT ported here (need a subsystem unreachable standalone):
  * POST /api/builder/generate  (NL → spec) — needs a live LLM chat model
    (``generate_spec_sync``); a standalone API has no provider keys wired.
  * POST /api/builder/share/{name} (publish to lab) — needs the cloud push
    server + token (``share_to_lab_sync``); the push-back gate is a
    separate live-only path. Both stay in the live app; they are not the
    save/create/update/delete blockers this seam exists to close.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request

from mast.api.schemas_builder import (
    BuilderCatalogEntry,
    BuilderCatalogResponse,
    BuilderValidateRequest,
    BuilderValidateResponse,
    BuilderValidateStep,
    CompositeCreateRequest,
    CompositeDeleteResult,
    CompositeSaveResult,
    CompositeUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["builder"])


# ── helpers ────────────────────────────────────────────────────────────────


def _report_steps(report: dict) -> list[BuilderValidateStep]:
    """Coerce a validate_spec_payload ``steps`` list into typed rows."""
    out: list[BuilderValidateStep] = []
    for s in report.get("steps") or []:
        if not isinstance(s, dict):
            continue
        out.append(
            BuilderValidateStep(
                id=str(s.get("id", "?")),
                skill=str(s.get("skill", "")),
                errors=list(s.get("errors") or []),
                warnings=list(s.get("warnings") or []),
            )
        )
    return out


def _save_spec(
    name: str,
    spec_dict: dict,
    *,
    base_version: int | None,
    require_new: bool,
) -> CompositeSaveResult:
    """Relay one save onto the kept core, mirroring builder_api._save_sync.

    Validate (design-time) → CompositeSpec.from_dict → store.save (which bumps
    the version + archives a _history snapshot, and enforces the base_version
    optimistic-concurrency gate) → hot-register the live singleton → invalidate
    the palette catalog. All business/safety logic stays in the core; this only
    sequences the kept calls and maps outcomes to a typed result.

    ``require_new`` (POST /composites) rejects an already-existing name so a
    create never silently overwrites; the update path (PUT) allows it."""
    if not isinstance(spec_dict, dict):
        return CompositeSaveResult(ok=False, name=name, error="missing_spec",
                                   degraded=False)
    # The spec's own name is authoritative for the store; align with the URL.
    spec_dict = dict(spec_dict)
    if not spec_dict.get("name"):
        spec_dict["name"] = name
    if name and spec_dict.get("name") != name:
        return CompositeSaveResult(
            ok=False, name=name, error="name_mismatch",
            message="URL 中的名字与 spec.name 不一致", degraded=False)
    name = spec_dict["name"]

    # Lazy-import the kept core; absence ⇒ degrade (never 500).
    try:
        from mast.skills.composite.spec import CompositeSpec
        from mast.skills.composite.version_store import (
            VersionConflictError,
            VersionStoreError,
        )
        from mast.webui.builder_api import (
            invalidate_catalog,
            resolved_skills_snapshot,
            save_extra_meta,
            validate_spec_payload,
        )
        from mast.webui.composite_panel import _hot_register, composite_store
    except Exception as exc:
        logger.warning("composite save backend unavailable: %s", exc)
        return CompositeSaveResult(ok=False, name=name, error=str(exc),
                                   degraded=True)

    store = composite_store()
    try:
        if require_new and store.exists(name):
            return CompositeSaveResult(
                ok=False, name=name, error="already_exists",
                message=f"组合技能 {name!r} 已存在——改用更新或换名",
                degraded=False)

        # Design-time lint (fail-closed; an invalid spec is NOT degradation).
        report = validate_spec_payload(spec_dict)
        if not report.get("ok"):
            return CompositeSaveResult(
                ok=False, name=name, error="invalid_spec",
                problems=list(report.get("problems") or []),
                steps=_report_steps(report), degraded=False)

        try:
            spec = CompositeSpec.from_dict(spec_dict)
        except Exception as exc:
            return CompositeSaveResult(
                ok=False, name=name, error="invalid_spec",
                problems=[str(exc)], degraded=False)

        # Sync-ready record (RFC §7): author/machine stamps + resolved versions.
        extra = save_extra_meta()
        extra["_resolved_skills"] = resolved_skills_snapshot(spec_dict)
        try:
            saved = store.save(spec, base_version=base_version, extra_meta=extra)
        except VersionConflictError as exc:
            stored = None
            try:
                stored = store.load(name).version
            except Exception:
                pass
            return CompositeSaveResult(
                ok=False, name=name, error="version_conflict",
                message=str(exc), stored_version=stored, degraded=False)
        except VersionStoreError as exc:
            return CompositeSaveResult(
                ok=False, name=name, error="invalid_spec",
                problems=[str(exc)], degraded=False)

        # Hot-register the live singleton (best-effort; no-op when unwired).
        msg = ""
        try:
            msg = _hot_register(saved.name) or ""
        except Exception as exc:  # never let registration failure break the save
            logger.warning("hot-register failed for %s: %s", saved.name, exc)
        try:
            invalidate_catalog()
        except Exception:
            pass
        return CompositeSaveResult(
            ok=True, name=saved.name, version=saved.version,
            hot_registered="已热注册" in msg, message=msg.strip(),
            problems=list(report.get("problems") or []),
            steps=_report_steps(report), degraded=False)
    except Exception as exc:
        logger.warning("composite save failed for %s: %s", name, exc)
        return CompositeSaveResult(ok=False, name=name, error=str(exc),
                                   degraded=True)


# ── POST /api/composites — create a new composite ─────────────────────────────


@router.post("/composites", response_model=CompositeSaveResult)
async def create_composite(
    body: CompositeCreateRequest, request: Request
) -> CompositeSaveResult:
    """Publish a builder spec as a NEW composite skill.

    Rejects an already-existing name (use PUT to overwrite). Writes v1 + a
    ``_history/<name>.v1.json`` snapshot via the kept version store, then
    hot-registers the live singleton. Backing: ``version_store.save`` (+
    builder_api validate/snapshot/stamp + composite_panel hot-register)."""
    name = (body.name or "").strip() or str(
        (body.spec or {}).get("name", "")).strip()
    return _save_spec(name, body.spec, base_version=None, require_new=True)


# ── PUT /api/composites/{name} — overwrite an existing composite ──────────────


@router.put("/composites/{name}", response_model=CompositeSaveResult)
async def update_composite(
    name: str, body: CompositeUpdateRequest, request: Request
) -> CompositeSaveResult:
    """Overwrite an existing composite (writes a new version, like the old save).

    The version store bumps the integer version and archives the prior content
    under ``_history/`` — nothing is lost. ``base_version`` (optional) enforces
    optimistic concurrency: a ``version_conflict`` result (with ``stored_version``)
    means someone saved in between. Backing: ``version_store.save``."""
    return _save_spec(name, body.spec, base_version=body.base_version,
                      require_new=False)


# ── DELETE /api/composites/{name} — delete a composite ────────────────────────


@router.delete("/composites/{name}", response_model=CompositeDeleteResult)
async def delete_composite(name: str, request: Request) -> CompositeDeleteResult:
    """Delete a composite's current spec (history snapshots are retained).

    Also hot-unregisters the live singleton + invalidates the palette catalog.
    Backing: ``version_store.delete`` (+ composite_panel ``_hot_unregister``)."""
    try:
        from mast.skills.composite.version_store import VersionStoreError
        from mast.webui.builder_api import invalidate_catalog
        from mast.webui.composite_panel import _hot_unregister, composite_store
    except Exception as exc:
        logger.warning("composite delete backend unavailable: %s", exc)
        return CompositeDeleteResult(ok=False, name=name, error=str(exc),
                                     degraded=True)
    store = composite_store()
    try:
        if not store.exists(name):
            return CompositeDeleteResult(ok=False, name=name, found=False,
                                         error="not_found", degraded=False)
        store.delete(name)
    except VersionStoreError as exc:  # invalid name etc. — not a 500 oracle
        return CompositeDeleteResult(ok=False, name=name, found=False,
                                     error=str(exc), degraded=False)
    except Exception as exc:
        logger.warning("composite delete failed for %s: %s", name, exc)
        return CompositeDeleteResult(ok=False, name=name, error=str(exc),
                                     degraded=True)
    msg = ""
    try:
        msg = _hot_unregister(name) or ""
    except Exception as exc:
        logger.warning("hot-unregister failed for %s: %s", name, exc)
    try:
        invalidate_catalog()
    except Exception:
        pass
    return CompositeDeleteResult(ok=True, name=name, found=True,
                                 message=msg.strip(), degraded=False)


# ── POST /api/builder/validate — name-agnostic design-time lint ────────────────


@router.post("/builder/validate", response_model=BuilderValidateResponse)
async def validate_builder_spec(
    body: BuilderValidateRequest, request: Request
) -> BuilderValidateResponse:
    """Design-time validate a builder spec (name-agnostic — for the live canvas
    before a name is chosen). Reuses ``builder_api.validate_spec_payload``; the
    validator checks skill existence / param bounds against the live registry
    when wired, and still reports structural problems standalone."""
    spec_dict = body.spec if isinstance(body.spec, dict) else {}
    try:
        from mast.webui.builder_api import validate_spec_payload

        report = validate_spec_payload(spec_dict)
        return BuilderValidateResponse(
            ok=bool(report.get("ok", False)),
            problems=list(report.get("problems") or []),
            steps=_report_steps(report),
            degraded=False,
        )
    except Exception as exc:
        logger.warning("builder validate failed: %s", exc)
        return BuilderValidateResponse(
            ok=False, problems=[f"校验后端不可用：{exc}"], degraded=True)


# ── GET /api/builder/catalog — paginated palette catalog ──────────────────────


@router.get("/builder/catalog", response_model=BuilderCatalogResponse)
def builder_catalog(
    request: Request,
    q: str = Query(default=""),
    category: str = Query(default=""),
    tag: str = Query(default=""),
    source: str = Query(default=""),
    safety: str = Query(default=""),
    level: str = Query(default=""),
    domain: str = Query(default=""),
    subscribed: str = Query(
        default="",
        description='"1" 只看已订阅 / "0" 只看未订阅 / "" 全部（默认，见 docstring）'),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50000, ge=1, le=50000),
) -> BuilderCatalogResponse:
    """Builder palette catalog — richer than ``/api/skills/catalog``: every row
    carries source/source_zh/domain/level (the builder's grouping facets) and
    the full query-param filter contract. Reuses ``builder_api.get_catalog`` +
    ``filter_index`` (in-memory, off the live registry). Degrades to an empty
    list when the registry is unwired/raises — never a 500.

    **订阅（2026-08-26）**：每行带 ``subscribed`` / ``mandatory`` 两列，但
    ``subscribed`` 过滤**默认关**。刻意的：BuilderPage 还没有「只看订阅 / 全市场」
    的开关，这时候把默认改成只显示订阅项，用户会看到 palette 无缘无故少了一半
    而界面上没有任何东西解释为什么 —— 比不过滤糟得多。开关和默认一起进（二期）。

    palette 与 agent 工具面在这里**可以**不同步，而且没有安全后果：composite 的
    子步走 ``ExecutionContext.run``，那条路本来就不看订阅（也不看硬件门）。
    """
    ctx = request.app.state.ctx
    if getattr(ctx, "skill_registry", None) is None:
        # Standalone / not wired — empty but not broken (catalog needs the live
        # registry; builder_api.build_catalog returns {} without one).
        return BuilderCatalogResponse(page=page, degraded=True)
    try:
        from mast.webui.builder_api import filter_index, get_catalog

        cat = get_catalog()
        idx = filter_index(
            cat.get("index", []), q=q, category=category, tag=tag,
            source=source, safety=safety, level=level, domain=domain)

        # 订阅状态在路由层 join（不进目录缓存 —— 见 BuilderCatalogEntry 的注释）。
        # 订阅子系统缺席时 fail-open：全部当已订阅，palette 一个不少。
        try:
            from mast.skills.subscription import MANDATORY_SKILLS, subscribed_names
            subs = subscribed_names()
            mandatory = MANDATORY_SKILLS
        except Exception as exc:  # noqa: BLE001
            logger.warning("builder catalog: 订阅状态读不到（%s）—— 全部按已订阅列", exc)
            subs, mandatory = None, frozenset()

        def _is_sub(name: str) -> bool:
            return True if subs is None else (name in subs or name in mandatory)

        if subscribed in ("0", "1"):
            want = subscribed == "1"
            idx = [e for e in idx if _is_sub(str(e.get("name") or "")) is want]

        total = len(idx)
        items = idx[(page - 1) * page_size: page * page_size]
        skills = [
            BuilderCatalogEntry(
                name=str(e.get("name", "")),
                zh=str(e.get("zh", "") or ""),
                category=str(e.get("category", "") or ""),
                safety=str(e.get("safety", "") or ""),
                level=int(e.get("level", 0) or 0),
                source=str(e.get("source", "other") or "other"),
                source_zh=str(e.get("source_zh", "") or ""),
                tags=list(e.get("tags") or []),
                domain=str(e.get("domain", "其他") or "其他"),
                subscribed=_is_sub(str(e.get("name", ""))),
                mandatory=str(e.get("name", "")) in mandatory,
            )
            for e in items
            if isinstance(e, dict)
        ]
        return BuilderCatalogResponse(
            total=total, page=page, skills=skills, degraded=False)
    except Exception as exc:
        logger.warning("builder catalog failed: %s", exc)
        return BuilderCatalogResponse(page=page, degraded=True)
