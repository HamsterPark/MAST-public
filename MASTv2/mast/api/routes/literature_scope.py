"""Experiment-scoped literature libraries — ensure + copy.

设计文档：``docs/v2/design/document_and_library_management.md`` §4

Two endpoints, both thin relays onto :mod:`mast.knowledge.experiment_library`:

  * ``POST /api/literature/experiments/{experiment_id}/library`` — lazily create
    (or just return) that experiment's own library. **Idempotent**: the owner's
    model is one library per experiment with a derived id (``exp_<id8>``), so
    "create" is really "ensure" and calling it twice is not an error.
  * ``POST /api/literature/libraries/{library_id}/copy`` — copy a library's
    members into an experiment's library. This is the product's answer to
    cross-experiment sharing: libraries are never shared, you take a copy, and
    library count growing is explicitly fine .

Why there is no "bind" endpoint. The old design specified an
``experiment_library_binding`` table with its own audit events; under one
library per experiment the binding is an *identity* relation — the id is derived
from the experiment id — so the mapping table disappears entirely and nothing
can drift out of sync with it.

Why there is no "switch the current library" endpoint here either. The effective
library is a pull: ``resolve_effective_library()`` reads the active scope on every
call. Activating an experiment (``routes/scope.py``) therefore switches the
library with **zero** changes on this side — no subscription, no event, nothing to
forget to fire. ``POST /literature/libraries/{id}/activate`` (in
``literature_ext2``) still exists but is demoted: it sets the manual pointer used
only while no experiment is active.

GRACEFUL DEGRADATION (house rule 2): the API boots standalone with no live core.
The backend is lazy-imported inside each handler; absence or any raise returns a
typed ``degraded`` body, never a 500. An honest domain error (no such library,
no active experiment) is ``ok=False, degraded=False`` with a ``message``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from mast.api.schemas_literature import (
    CopyLibraryRequest,
    CopyLibraryResponse,
    EnsureExperimentLibraryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["literature"])


# ── POST /api/literature/experiments/{experiment_id}/library ────────────


@router.post(
    "/literature/experiments/{experiment_id}/library",
    response_model=EnsureExperimentLibraryResponse,
)
def ensure_experiment_library(
    experiment_id: str, request: Request
) -> EnsureExperimentLibraryResponse:
    """Ensure the experiment's own literature library exists. Idempotent.

    Normally nothing needs to call this — the library is created on first real use
    (an add, an ingest, a copy target). The endpoint exists so the UI can show the
    library section for an experiment that has not curated anything yet without
    having to fake an empty one.
    """
    _ = request.app.state.ctx
    eid = (experiment_id or "").strip()
    if not eid:
        return EnsureExperimentLibraryResponse(
            ok=False, message="experiment_id must be non-empty")
    try:
        from mast.knowledge import experiment_library as expl
        from mast.knowledge import libraries as lib_mod
    except Exception as exc:
        logger.warning("ensure_experiment_library: backend unavailable: %s", exc)
        return EnsureExperimentLibraryResponse(
            ok=False, degraded=True, experiment_id=eid, message=str(exc))

    lib_id = expl.experiment_library_id(eid)
    try:
        existed = lib_id in {r.get("library_id") for r in lib_mod.list_libraries()}
    except Exception:  # noqa: BLE001 — unknown ⇒ report created=False, not a failure
        existed = False

    try:
        lib_id = expl.ensure_experiment_library(eid)
    except Exception as exc:  # noqa: BLE001 — the core is degrade-safe; belt and braces
        logger.warning("ensure_experiment_library(%s) failed: %s", eid, exc)
        return EnsureExperimentLibraryResponse(
            ok=False, degraded=True, experiment_id=eid, message=str(exc))
    if not lib_id:
        return EnsureExperimentLibraryResponse(
            ok=False, experiment_id=eid,
            message="无法为该实验解析出专属库 id（experiment_id 为空或无效）。")

    path = expl.members_path(eid)
    members = expl.current_members(eid)
    if path is None:
        # No experiment folder (the experiment row is gone / undreadable). The
        # library still exists in the registry, so curation is not blocked — say so
        # instead of implying the folder-authoritative bibliography is in place.
        msg = (f"已就绪：{lib_id}（**仅 registry**；该实验的文件夹解析不到，"
               f"书目暂时没有文件夹权威，实验行恢复后会自动补上）")
    else:
        msg = f"已就绪：{lib_id}（书目权威：{path}）"
    return EnsureExperimentLibraryResponse(
        ok=True, degraded=False, library_id=lib_id, experiment_id=eid,
        created=not existed, member_count=len(members),
        members_path="" if path is None else str(path), message=msg,
    )


# ── POST /api/literature/libraries/{library_id}/copy ───────────────────


@router.post(
    "/literature/libraries/{library_id}/copy",
    response_model=CopyLibraryResponse,
)
def copy_library(
    library_id: str, body: CopyLibraryRequest, request: Request
) -> CopyLibraryResponse:
    """Copy *library_id*'s members into an experiment's own library.

    ``to_experiment_id`` empty = the currently active experiment. The source may be
    any library (global / custom / another experiment's). Members already in the
    target are reported as ``skipped`` and left untouched — overwriting their
    ``reason`` with the source's wording would quietly destroy the target
    experiment's own annotations.
    """
    _ = request.app.state.ctx
    src = (library_id or "").strip()
    if not src:
        return CopyLibraryResponse(ok=False, message="library_id must be non-empty")
    try:
        from mast.knowledge import experiment_library as expl
    except Exception as exc:
        logger.warning("copy_library: backend unavailable: %s", exc)
        return CopyLibraryResponse(
            ok=False, degraded=True, src_library_id=src, message=str(exc))
    try:
        res = expl.copy_library(src, (body.to_experiment_id or "").strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("copy_library(%s) failed: %s", src, exc)
        return CopyLibraryResponse(
            ok=False, degraded=True, src_library_id=src, message=str(exc))

    if not res.get("ok"):
        # honest domain error (unknown src, same src/dst, no active experiment)
        return CopyLibraryResponse(
            ok=False, degraded=False, src_library_id=src,
            message=str(res.get("error") or "copy failed"))

    copied = [str(x) for x in (res.get("copied") or [])]
    rejected = [str(x) for x in (res.get("rejected") or [])]
    parts = [f"已复制 {len(copied)} 条到 {res.get('library_id', '')}"]
    if res.get("skipped"):
        parts.append(f"{len(res['skipped'])} 条目标库里已有（未改动）")
    if rejected:
        parts.append(f"{len(rejected)} 条未能加入")
    if res.get("at_cap"):
        parts.append("目标库已到 500 成员上限")
    if res.get("note"):
        parts.append(str(res["note"]))
    return CopyLibraryResponse(
        ok=True, degraded=False,
        src_library_id=str(res.get("src_library_id", src)),
        library_id=str(res.get("library_id", "")),
        to_experiment_id=str(res.get("to_experiment_id", "")),
        copied=copied,
        skipped=[str(x) for x in (res.get("skipped") or [])],
        rejected=rejected,
        member_count=int(res.get("member_count", 0) or 0),
        at_cap=bool(res.get("at_cap", False)),
        message="；".join(parts) + "。",
    )
