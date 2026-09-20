"""数据图库 —— thin relay onto ``mast.gallery``.


House rules this module holds:

* **Lazy imports, graceful degradation.** Every handler imports ``mast.gallery``
  inside a try and answers ``degraded=True`` + ``detail`` on any failure. Never 500.
* **GET is read-only.** No handler behind a GET creates the state directory or
  writes a file (trap T3): a contract test that walks every GET must not be able
  to leave a ``_gallery`` folder in the operator's real data root.
* **The build never runs on a request thread.** ``POST /gallery/build`` starts (or
  reports) the single background build and returns immediately.
* **Own prefix.** ``/gallery/*`` overlaps nothing; this repo has twice had a
  ``/{param}`` route capture a literal segment and answer 200 from the wrong
  handler (see the registration comments in ``api/app.py``).
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from mast.api.schemas_gallery import (
    GalleryBuildRequest,
    GalleryBuildStatus,
    GalleryConfigResponse,
    GalleryConfigUpdate,
    GalleryIndex,
    GalleryMarksDoc,
    GalleryMarksImportRequest,
    GalleryMarksImportResult,
    GalleryMarksPatch,
    GalleryMarksPatchResult,
    GalleryRoot,
    GalleryRootSuggestion,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["gallery"])


# ── config ─────────────────────────────────────────────────────────────


def _config_response() -> GalleryConfigResponse:
    from mast.gallery import config as gcfg
    from mast.gallery import paths as gpaths

    cfg = gcfg.load_config()
    return GalleryConfigResponse(
        ok=True,
        state_dir=str(gpaths.state_dir()),
        roots=[
            GalleryRoot(name=r.name, path=r.path, enabled=r.enabled,
                        exists=os.path.isdir(r.path))
            for r in cfg.roots
        ],
        suggestions=[GalleryRootSuggestion(path=p, why=why)
                     for p, why in gcfg.suggest_roots()],
        workers=cfg.workers,
    )


@router.get("/gallery/config", response_model=GalleryConfigResponse)
def gallery_config() -> GalleryConfigResponse:
    try:
        return _config_response()
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("gallery config read failed: %s", exc)
        return GalleryConfigResponse(ok=False, degraded=True, detail=str(exc))


@router.post("/gallery/config", response_model=GalleryConfigResponse)
def gallery_config_update(body: GalleryConfigUpdate) -> GalleryConfigResponse:
    """Replace the root list. Invalid input writes nothing and says why."""
    try:
        from mast.gallery import config as gcfg

        roots, errors = gcfg.normalise_roots([r.model_dump() for r in body.roots])
        if errors:
            current = _config_response()
            current.ok = False
            current.detail = "；".join(errors)
            return current
        workers = body.workers if body.workers is not None else gcfg.load_config().workers
        gcfg.save_config(gcfg.GalleryConfig(roots=roots, workers=workers))
        return _config_response()
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery config write failed: %s", exc)
        return GalleryConfigResponse(ok=False, degraded=True, detail=str(exc))


# ── build ──────────────────────────────────────────────────────────────


@router.get("/gallery/status", response_model=GalleryBuildStatus)
def gallery_status() -> GalleryBuildStatus:
    try:
        from mast.gallery import service

        return GalleryBuildStatus(**service.get_status())
    except Exception as exc:  # noqa: BLE001
        return GalleryBuildStatus(degraded=True, detail=str(exc))


@router.post("/gallery/build", response_model=GalleryBuildStatus)
def gallery_build(body: GalleryBuildRequest) -> GalleryBuildStatus:
    """Start the background build, or report the one already running."""
    try:
        from mast.gallery import service

        return GalleryBuildStatus(**service.start_build(force=body.force))
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery build start failed: %s", exc)
        return GalleryBuildStatus(degraded=True, detail=str(exc))


@router.post("/gallery/build/cancel", response_model=GalleryBuildStatus)
def gallery_build_cancel() -> GalleryBuildStatus:
    try:
        from mast.gallery import service

        return GalleryBuildStatus(**service.cancel_build())
    except Exception as exc:  # noqa: BLE001
        return GalleryBuildStatus(degraded=True, detail=str(exc))


# ── index + thumbnails ─────────────────────────────────────────────────


@router.get("/gallery/index", response_model=GalleryIndex)
def gallery_index(request: Request):
    """The whole index in one response, served from disk (design D9).

    Declared as ``GalleryIndex`` so the frontend gets types; returned as the
    pre-serialised bytes so 5000+ items are not re-validated per page load. When
    the client accepts gzip and the build wrote ``index.json.gz``, those bytes go
    out with ``Content-Encoding: gzip``."""
    try:
        from mast.gallery import index as gindex

        accepts_gzip = "gzip" in (request.headers.get("accept-encoding") or "").lower()
        got = gindex.read_index_bytes(prefer_gzip=accepts_gzip)
        if got is None:
            return GalleryIndex(built=False)
        data, gzipped = got
        headers = {"Cache-Control": "no-store"}
        if gzipped:
            headers["Content-Encoding"] = "gzip"
        return Response(content=data, media_type="application/json", headers=headers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery index read failed: %s", exc)
        return GalleryIndex(degraded=True, detail=str(exc))


@router.get("/gallery/thumb/{path:path}")
def gallery_thumb(path: str, v: int | None = None):
    """One rendered thumbnail. ``v`` is a cache-buster the index puts on every URL,
    so a URL that carries it can be cached forever (design D10)."""
    try:
        from mast.gallery import paths as gpaths

        base = gpaths.thumbs_dir().resolve()
        target = (base / path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return JSONResponse({"detail": "not found"}, status_code=404)
        cache = ("public, max-age=31536000, immutable" if v is not None else "no-cache")
        return FileResponse(target, headers={"Cache-Control": cache})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"detail": str(exc)}, status_code=404)


# ── marks ──────────────────────────────────────────────────────────────


@router.get("/gallery/marks", response_model=GalleryMarksDoc)
def gallery_marks():
    """The marks document as stored (design D11/D9: not re-serialised)."""
    try:
        from mast.gallery import marks as gmarks

        return JSONResponse(gmarks.load_marks(), headers={"Cache-Control": "no-store"})
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery marks read failed: %s", exc)
        return GalleryMarksDoc(degraded=True, detail=str(exc))


@router.post("/gallery/marks/patch", response_model=GalleryMarksPatchResult)
def gallery_marks_patch(body: GalleryMarksPatch) -> GalleryMarksPatchResult:
    """Apply one batch of edits. Only the keys the client sent are stored —
    ``exclude_unset`` keeps a mark from growing every default field."""
    try:
        from mast.gallery import marks as gmarks

        patch = body.model_dump(by_alias=True, exclude_unset=True)
        rev, updated = gmarks.patch_marks(patch)
        return GalleryMarksPatchResult(ok=True, rev=rev, updated=updated)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery marks patch failed: %s", exc)
        return GalleryMarksPatchResult(ok=False, degraded=True, detail=str(exc))


@router.post("/gallery/marks/import", response_model=GalleryMarksImportResult)
def gallery_marks_import(body: GalleryMarksImportRequest) -> GalleryMarksImportResult:
    try:
        from mast.gallery import marks as gmarks

        res = gmarks.import_marks(body.doc, key_prefix=body.key_prefix)
        return GalleryMarksImportResult(ok=True, **res)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery marks import failed: %s", exc)
        return GalleryMarksImportResult(ok=False, degraded=True, detail=str(exc))


@router.get("/gallery/marks/export/{fmt}")
def gallery_marks_export(fmt: Literal["json", "md", "csv", "series_csv"]):
    """Download one rendering of the marks. Read-only: generated on the fly when
    nothing has been saved yet."""
    try:
        from mast.gallery import marks as gmarks

        data, media_type, filename = gmarks.export_file(fmt)
        return Response(
            content=data, media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"',
                     "Cache-Control": "no-store"},
        )
    except Exception as exc:  # noqa: BLE001
        # 503 rather than a 200 JSON body: this is a download link, and a browser
        # would otherwise save the error message under the name marks.md.
        return JSONResponse({"degraded": True, "detail": str(exc)}, status_code=503)
