"""数据图库 → 出图 —— thin relay onto ``mast.gallery.figures``.


Same house rules as ``routes/gallery.py``: lazy imports and degraded bodies,
never 500; list/status GETs are read-only; the figure job runs in its own
single background slot and ``POST /run`` returns at once.

The one GET that WRITES is ``/preview/…``: it renders a 480 px JPEG next to the
figure the first time it is asked for. It only ever writes when the source file
exists, so a cold machine (or the boot-smoke walk with ``path=nope``) gets a 404
and no directory is created.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from mast.api.schemas_gallery_figures import (
    GalleryFigureJobStatus,
    GalleryFigureRequest,
    GalleryFiguresList,
    GalleryStsLinePlan,
    GalleryStsLinePlanRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["gallery"])


@router.get("/gallery/figures", response_model=GalleryFiguresList)
def gallery_figures() -> GalleryFiguresList:
    try:
        from mast.gallery.figures import store

        return GalleryFiguresList(**store.list_figures())
    except Exception as exc:  # noqa: BLE001 — degrade, never 500
        logger.warning("gallery figures list failed: %s", exc)
        return GalleryFiguresList(degraded=True, detail=str(exc))


@router.get("/gallery/figures/status", response_model=GalleryFigureJobStatus)
def gallery_figures_status() -> GalleryFigureJobStatus:
    try:
        from mast.gallery.figures import service

        return GalleryFigureJobStatus(**service.get_status())
    except Exception as exc:  # noqa: BLE001
        return GalleryFigureJobStatus(degraded=True, detail=str(exc))


@router.post("/gallery/figures/run", response_model=GalleryFigureJobStatus)
def gallery_figures_run(body: GalleryFigureRequest) -> GalleryFigureJobStatus:
    """Start a figure job, or report the one already running."""
    try:
        from mast.gallery.figures import service

        return GalleryFigureJobStatus(**service.start(
            kind=body.kind, ids=list(body.ids), series=list(body.series),
            options=dict(body.options)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery figure job start failed: %s", exc)
        return GalleryFigureJobStatus(degraded=True, detail=str(exc))


@router.post("/gallery/figures/cancel", response_model=GalleryFigureJobStatus)
def gallery_figures_cancel() -> GalleryFigureJobStatus:
    try:
        from mast.gallery.figures import service

        return GalleryFigureJobStatus(**service.cancel())
    except Exception as exc:  # noqa: BLE001
        return GalleryFigureJobStatus(degraded=True, detail=str(exc))


@router.post("/gallery/figures/sts_lines/plan", response_model=GalleryStsLinePlan)
def gallery_figures_sts_plan(body: GalleryStsLinePlanRequest) -> GalleryStsLinePlan:
    """Dry run of the station inference for an sts_lines figure (index only)."""
    try:
        from mast.gallery.figures import lines

        return GalleryStsLinePlan(**lines.plan_from_store(list(body.series), dict(body.options)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("gallery sts line plan failed: %s", exc)
        return GalleryStsLinePlan(ok=False, degraded=True, detail=str(exc))


@router.get("/gallery/figures/file/{path:path}")
def gallery_figures_file(path: str, v: int | None = None):
    """One product file. ``store.figure_file`` refuses anything outside figures/."""
    try:
        from mast.gallery.figures import store

        target = store.figure_file(path)
        if target is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        cache = "public, max-age=31536000, immutable" if v is not None else "no-cache"
        return FileResponse(target, filename=target.name, headers={"Cache-Control": cache})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"detail": str(exc)}, status_code=404)


@router.get("/gallery/figures/preview/{path:path}")
def gallery_figures_preview(path: str, v: int | None = None):
    """A 480 px JPEG of one image product, made on first request (see module doc)."""
    try:
        from mast.gallery.figures import store

        target = store.preview_file(path)
        if target is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        cache = "public, max-age=31536000, immutable" if v is not None else "no-cache"
        return FileResponse(target, media_type="image/jpeg", headers={"Cache-Control": cache})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"detail": str(exc)}, status_code=404)
