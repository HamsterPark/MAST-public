"""Pydantic request/response models for the literature-library *ext2* seam.

Parity follow-up to ``schemas_literature.py``: these are the shapes for the
library routes whose UI was lost in the Gradio→TS rewrite but whose LOGIC still
lives in the Python core (``mast.knowledge.libraries`` + ``mast.knowledge.
fetch_board``). Each model mirrors the JSON-compatible dict the kept backend
already returns — the routes only relay.

Like the rest of ``mast.api.schemas*`` these are the SINGLE SOURCE OF TYPES
(``/openapi.json`` → frontend type generators). Every response carries a
``degraded`` boolean so the standalone API (no live core wired / a core call
that raises) returns an empty-but-not-broken body instead of a 500.

Covered endpoints (see routes/literature_ext2.py):
  * POST /literature/libraries/{id}/activate    — ``set_active_library``;
  * GET  /literature/libraries/{id}             — ``get_library`` (members + meta);
  * POST /literature/libraries/{id}/members/remove — ``remove_members``;
  * POST /literature/fetch-board/{rid}/resolve  — ``fetch_board.resolve``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# The member-pointer + library-detail shapes are identical to the originals in
# schemas_literature.py; re-export them here so this seam is self-contained for
# its consumers without redefining the field set (single source of the shape).
from mast.api.schemas_literature import LibraryDetail, LibraryMember

__all__ = [
    "LibraryMember",
    "LibraryDetail",
    "ActivateLibraryResponse",
    "LibraryDetailResponse",
    "RemoveMembersRequest",
    "RemoveMembersResponse",
    "ResolveFetchRequest",
    "ResolveFetchResponse",
]


# ── POST /literature/libraries/{id}/activate ──────────────────────────


class ActivateLibraryResponse(BaseModel):
    """Outcome of ``set_active_library`` — the now-active library record.

    ``ok=False, degraded=True`` when the library backend is absent/unreachable
    (standalone dev) or the call raised; ``ok=False, degraded=False`` for an
    honest domain error such as an unknown ``library_id`` (``LibraryError``)."""

    ok: bool = False
    degraded: bool = False
    library_id: Optional[str] = None
    library: Optional[LibraryDetail] = None
    message: str = ""


class LibraryDetailResponse(BaseModel):
    """A single library record incl. its member pointer-set (``get_library``).

    ``found=False`` for an unknown ``library_id`` (still 200, not degraded);
    ``degraded=True`` when the backend is absent or the lookup raised."""

    found: bool = False
    degraded: bool = False
    library: Optional[LibraryDetail] = None
    member_count: int = 0
    message: str = ""


# ── POST /literature/libraries/{id}/members/remove ────────────────────


class RemoveMembersRequest(BaseModel):
    """Remove the named ``work_id`` pointers from a library."""

    work_ids: list[str] = Field(default_factory=list)


class RemoveMembersResponse(BaseModel):
    """Outcome of ``remove_members`` (mirrors the backend dict)."""

    ok: bool = False
    degraded: bool = False
    library_id: Optional[str] = None
    removed: list[str] = Field(default_factory=list)
    member_count: int = 0
    n_removed: int = 0
    message: str = ""


# ── POST /literature/fetch-board/{rid}/resolve ────────────────────────


class ResolveFetchRequest(BaseModel):
    """Resolve a full-text fetch request.

    The board statuses are ``pending | fulfilled | failed | dismissed``; the
    operator-facing actions here are ``done`` (→ ``fulfilled``) and
    ``dismissed`` (→ ``dismissed``), matching the kept GUI verbs."""

    action: str = Field(description="done | dismissed")
    note: str = ""


class ResolveFetchResponse(BaseModel):
    """Outcome of ``fetch_board.resolve`` — the updated request record.

    ``ok=False, degraded=True`` when the board backend is absent or raised;
    ``ok=False, degraded=False`` for an honest domain error (unknown request id
    or an invalid action), with ``message`` carrying the reason."""

    ok: bool = False
    degraded: bool = False
    request_id: Optional[str] = None
    status: str = ""
    work_id: str = ""
    note: str = ""
    resolved_at: Optional[str] = None
    message: str = ""
