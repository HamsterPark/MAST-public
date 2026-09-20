"""Parity follow-up — literature-library *ext2* routes.

Re-exposes the library-management actions whose UI was lost in the Gradio→TS
rewrite but whose LOGIC is fully kept in the Python core:

  * ``mast.knowledge.libraries`` — ``set_active_library`` / ``get_library`` /
    ``remove_members`` (the JSON-file-backed ``LibraryRegistry`` singleton);
  * ``mast.knowledge.fetch_board`` — ``resolve`` (the full-text fetch board).

Each handler is a THIN relay onto that kept backend; no business/membership/
safety logic lives here. These mirror the kept GUI handlers in the (now
removed) ``gui/literature_panel.py`` — ``switch_library_h`` /
``render_library_members_html`` / ``remove_members_h`` / ``dismiss_request_h``.

Endpoints:
  * POST /api/literature/libraries/{library_id}/activate  — set the active
        (default) library for subsequent add/search calls → ``set_active_library``.
  * GET  /api/literature/libraries/{library_id}           — one library record
        incl. its member pointer-set + meta → ``get_library``.
  * POST /api/literature/libraries/{library_id}/members/remove — drop the named
        ``work_id`` pointers → ``remove_members``.
  * POST /api/literature/fetch-board/{request_id}/resolve — mark a fetch request
        done/dismissed → ``fetch_board.resolve``.

GRACEFUL DEGRADATION is mandatory (house rule 2): this router boots STANDALONE
with no live core wired. Every handler checks ``_is_wired(ctx)`` FIRST, then
lazy-imports its backend inside try/except; any absence or raise returns a valid
degraded body (``degraded=True``) — never a 500. An honest domain error (unknown
id, bad action) is reported as ``ok=False, degraded=False`` with a ``message``
(the backend either raises ``LibraryError`` or returns ``{"error": ...}``).

The ``_is_wired`` gate was **missing here** until 2026-07-29 while every handler
in ``routes/literature.py`` had it. The result was a half-dead router: with no
core wired the list endpoints correctly reported ``degraded``, yet "set active
library" and "remove pointers" fell straight through to the JSON-backed singleton
and really rewrote ``registry.json`` on disk. An unwired API that still mutates
data is worse than one that refuses — the UI shows a degraded banner while the
writes land anyway.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from mast.api.routes.literature import _is_wired
from mast.api.schemas_literature import LibraryDetail
from mast.api.schemas_literature_ext2 import (
    ActivateLibraryResponse,
    LibraryDetailResponse,
    RemoveMembersRequest,
    RemoveMembersResponse,
    ResolveFetchRequest,
    ResolveFetchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["literature"])

# Operator-facing fetch-board actions → board status (mirrors the kept GUI
# verbs: ``done`` button → fulfilled, ``dismiss`` button → dismissed).
_ACTION_TO_STATUS = {"done": "fulfilled", "dismissed": "dismissed"}


def _library_to_detail(rec: dict) -> LibraryDetail:
    """Coerce a ``LibraryRegistry`` record dict into the typed detail shape.

    Pydantic drops the unmodelled member fields (``original_local_id`` from a
    merge-map rewrite); the modelled pointer fields ride through unchanged."""
    return LibraryDetail.model_validate(rec)


# ── POST /api/literature/libraries/{library_id}/activate ───────────────


@router.post(
    "/literature/libraries/{library_id}/activate",
    response_model=ActivateLibraryResponse,
)
def activate_library(library_id: str, request: Request) -> ActivateLibraryResponse:
    """Set the active (default) library for subsequent add/search calls.

    Relays ``libraries.set_active_library``. The registry persists the choice +
    returns the now-active record. Unknown id ⇒ honest domain error
    (``LibraryError``); backend absent / any other raise ⇒ degraded."""
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return ActivateLibraryResponse(
            ok=False, degraded=True, library_id=library_id,
            message="literature backend not wired")
    try:
        from mast.knowledge import libraries as lib_mod
    except Exception as exc:
        logger.warning("activate_library: library backend unavailable: %s", exc)
        return ActivateLibraryResponse(
            ok=False, degraded=True, library_id=library_id, message=str(exc)
        )
    try:
        rec = lib_mod.set_active_library(library_id)
    except lib_mod.LibraryError as exc:
        # honest domain error (e.g. unknown library_id) — not degraded
        return ActivateLibraryResponse(
            ok=False, degraded=False, library_id=library_id, message=str(exc)
        )
    except Exception as exc:
        logger.warning("activate_library failed (%s): %s", library_id, exc)
        return ActivateLibraryResponse(
            ok=False, degraded=True, library_id=library_id, message=str(exc)
        )
    return ActivateLibraryResponse(
        ok=True,
        degraded=False,
        library_id=rec.get("library_id", library_id),
        library=_library_to_detail(rec),
        message=f"active library set to {rec.get('name', library_id)!r}",
    )


# ── GET /api/literature/libraries/{library_id} ─────────────────────────


@router.get(
    "/literature/libraries/{library_id}",
    response_model=LibraryDetailResponse,
)
def get_library_detail(library_id: str, request: Request) -> LibraryDetailResponse:
    """Return one library record incl. its member pointer-set + meta.

    Relays ``libraries.get_library``. Unknown id ⇒ ``found=False`` (still 200,
    not degraded); backend absent / any other raise ⇒ degraded."""
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return LibraryDetailResponse(
            found=False, degraded=True, message="literature backend not wired")
    try:
        from mast.knowledge import libraries as lib_mod
    except Exception as exc:
        logger.warning("get_library_detail: library backend unavailable: %s", exc)
        return LibraryDetailResponse(found=False, degraded=True, message=str(exc))
    try:
        rec = lib_mod.get_library(library_id)
    except lib_mod.LibraryError as exc:
        # no such library — honest "not found", not degraded
        return LibraryDetailResponse(found=False, degraded=False, message=str(exc))
    except Exception as exc:
        logger.warning("get_library_detail failed (%s): %s", library_id, exc)
        return LibraryDetailResponse(found=False, degraded=True, message=str(exc))
    members = rec.get("members") or []
    return LibraryDetailResponse(
        found=True,
        degraded=False,
        library=_library_to_detail(rec),
        member_count=len(members),
    )


# ── POST /api/literature/libraries/{library_id}/members/remove ─────────


@router.post(
    "/literature/libraries/{library_id}/members/remove",
    response_model=RemoveMembersResponse,
)
def remove_library_members(
    library_id: str, body: RemoveMembersRequest, request: Request
) -> RemoveMembersResponse:
    """Drop the named ``work_id`` pointers from a library.

    Relays ``libraries.remove_members`` (dedup/validation owned by the core).
    Unknown id ⇒ honest domain error; backend absent / any other raise ⇒
    degraded. An empty ``work_ids`` is a no-op success (nothing removed)."""
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return RemoveMembersResponse(
            ok=False, degraded=True, library_id=library_id,
            message="literature backend not wired")
    try:
        from mast.knowledge import libraries as lib_mod
    except Exception as exc:
        logger.warning("remove_library_members: backend unavailable: %s", exc)
        return RemoveMembersResponse(
            ok=False, degraded=True, library_id=library_id, message=str(exc)
        )
    try:
        res = lib_mod.remove_members(list(body.work_ids or []), library_id=library_id)
    except lib_mod.LibraryError as exc:
        return RemoveMembersResponse(
            ok=False, degraded=False, library_id=library_id, message=str(exc)
        )
    except Exception as exc:
        logger.warning("remove_library_members failed (%s): %s", library_id, exc)
        return RemoveMembersResponse(
            ok=False, degraded=True, library_id=library_id, message=str(exc)
        )
    removed = res.get("removed", []) if isinstance(res, dict) else []
    n_removed = (
        res.get("n_removed", len(removed)) if isinstance(res, dict) else len(removed)
    )
    return RemoveMembersResponse(
        ok=True,
        degraded=False,
        library_id=res.get("library_id", library_id) if isinstance(res, dict) else library_id,
        removed=removed,
        member_count=res.get("member_count", 0) if isinstance(res, dict) else 0,
        n_removed=n_removed,
        message=f"removed {n_removed} pointer(s)",
    )


# ── POST /api/literature/fetch-board/{request_id}/resolve ──────────────


@router.post(
    "/literature/fetch-board/{request_id}/resolve",
    response_model=ResolveFetchResponse,
)
def resolve_fetch_request(
    request_id: str, body: ResolveFetchRequest, request: Request
) -> ResolveFetchResponse:
    """Mark a full-text fetch request done (→ fulfilled) or dismissed.

    Relays ``fetch_board.resolve``. The backend returns ``{"error": ...}`` for
    an unknown request id / invalid status (honest domain error → not degraded);
    backend absent / any raise ⇒ degraded. ``action`` maps ``done→fulfilled``
    / ``dismissed→dismissed`` (the kept GUI verbs)."""
    ctx = request.app.state.ctx
    if not _is_wired(ctx):
        return ResolveFetchResponse(
            ok=False, degraded=True, request_id=request_id,
            message="literature backend not wired")
    action = (body.action or "").strip().lower()
    target_status = _ACTION_TO_STATUS.get(action)
    if target_status is None:
        return ResolveFetchResponse(
            ok=False,
            degraded=False,
            request_id=request_id,
            message=f"invalid action {body.action!r} (expected 'done' or 'dismissed')",
        )
    try:
        from mast.knowledge import fetch_board as board_mod
    except Exception as exc:
        logger.warning("resolve_fetch_request: board backend unavailable: %s", exc)
        return ResolveFetchResponse(
            ok=False, degraded=True, request_id=request_id, message=str(exc)
        )
    try:
        out = board_mod.resolve(request_id, target_status, note=body.note or "")
    except Exception as exc:
        logger.warning("resolve_fetch_request failed (%s): %s", request_id, exc)
        return ResolveFetchResponse(
            ok=False, degraded=True, request_id=request_id, message=str(exc)
        )
    if not isinstance(out, dict) or out.get("error"):
        # honest domain error from the board (no such request / invalid status)
        msg = out.get("error", "resolve failed") if isinstance(out, dict) else "resolve failed"
        return ResolveFetchResponse(
            ok=False, degraded=False, request_id=request_id, message=msg
        )
    return ResolveFetchResponse(
        ok=True,
        degraded=False,
        request_id=out.get("request_id", request_id),
        status=out.get("status", target_status),
        work_id=out.get("work_id", ""),
        note=out.get("note", ""),
        resolved_at=out.get("resolved_at"),
        message=f"request {out.get('request_id', request_id)} → {out.get('status', target_status)}",
    )
