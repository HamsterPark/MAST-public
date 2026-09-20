"""Pydantic request/response models for the 文献库 (literature library) seam.

Domain G of the typed FastAPI rewrite (TS Phase 3). These shapes mirror the
JSON-compatible dicts the live core already returns:

  * libraries — ``mast.knowledge.libraries`` (``list_libraries`` /
    ``get_library`` / ``create_library`` / ``rename_library`` /
    ``delete_library`` / ``add_members`` / ``remove_members``);
  * semantic search — ``mast.knowledge.literature_index.search`` (degrade-safe:
    falls back to keyword matching, flagged per-row with ``degraded``);
  * abstract lookup — ``mast.knowledge.literature_index.fetch_abstract``.

Like the rest of ``mast.api.schemas*`` these are the SINGLE SOURCE OF TYPES:
exported via ``/openapi.json`` and consumed by the frontend type generators.
Every response carries a ``degraded`` boolean so the standalone API (no live
core wired) returns empty-but-not-broken payloads instead of 500-ing.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ── library records ───────────────────────────────────────────────────


class LibrarySummary(BaseModel):
    """One row of the library list (``LibraryRegistry.list_libraries``)."""

    library_id: str
    name: str = ""
    scope: str = Field(default="custom", description="global | experiment | custom")
    experiment_id: Optional[str] = Field(
        default=None,
        description="Set only for scope=experiment: the v1 experiment this library "
                    "belongs to. Its members are authoritative in that experiment's "
                    "folder (library/members.jsonl), not in registry.json.")
    experiment_name: str = Field(
        default="",
        description="Display name of that experiment, resolved for the UI; empty "
                    "when the experiment row is gone or this is not an "
                    "experiment library.")
    created_at: str = ""
    index_dir: Optional[str] = Field(
        default=None,
        description="Legacy per-library vector index. Always null: that feature is "
                    "dormant (design §4.6) and nothing writes this field.")
    member_count: int = 0
    is_active: bool = Field(
        default=False,
        description="Matches registry.active_library_id — the MANUAL pointer, only "
                    "consulted when no experiment is active. To pre-select a row, "
                    "use effective_library_id from the response instead.")


class LibraryMember(BaseModel):
    """A pointer into the big library (a member of a user library, §2.2)."""

    work_id: str
    doi: str = ""
    added_by: str = "user"
    added_at: str = ""
    reason: str = ""
    source: str = Field(default="openalex",
                        description="openalex | user_pdf | user_url | user_manual — "
                                    "where the bibliographic record came from. A "
                                    "copy between libraries does NOT change it; see "
                                    "copied_from.")
    fulltext_status: str = Field(default="none", description="none | requested | ingested")
    fulltext_ref: Optional[str] = Field(
        default=None,
        description="Relative path to the stored full text, e.g. 'papers/W123_au111'. "
                    "Relative on purpose: full text is a machine-level asset that "
                    "does not travel with the experiment folder.")
    copied_from: str = Field(
        default="",
        description="Source library_id when this pointer arrived via lib_copy.")


class LibraryDetail(BaseModel):
    """A single library record incl. its member pointer-set (``get_library``)."""

    library_id: str
    name: str = ""
    scope: str = "custom"
    experiment_id: Optional[str] = None
    created_at: str = ""
    index_dir: Optional[str] = None
    members: list[LibraryMember] = Field(default_factory=list)


class LibrariesResponse(BaseModel):
    """The library list for the 库管理 panel. ``degraded`` is True when the
    library backend is absent/unreachable (standalone dev) — empty, not broken.

    **The id to pre-select is ``effective_library_id``**, not
    ``active_library_id``. The latter is now only the manual fallback pointer used
    when no experiment is active; with an experiment running, adds and searches
    default to that experiment's own library instead, and pre-selecting the manual
    pointer would show the operator a library nothing is writing to.
    """

    libraries: list[LibrarySummary] = Field(default_factory=list)
    active_library_id: Optional[str] = Field(
        default=None,
        description="registry.active_library_id — the manual pointer. Consulted "
                    "only when no experiment is active.")
    effective_library_id: Optional[str] = Field(
        default=None,
        description="The library that an add/search with no explicit library_id "
                    "actually targets right now. Pre-select THIS.")
    effective_source: str = Field(
        default="",
        description="How effective_library_id was reached: 'experiment' (the "
                    "active experiment's own library), 'manual' "
                    "(active_library_id), or 'fallback' (the reading library, "
                    "because the manual pointer was unusable).")
    count: int = 0
    degraded: bool = False


# ── write request bodies ──────────────────────────────────────────────


class CreateLibraryRequest(BaseModel):
    name: str
    scope: str = Field(default="custom", description="custom | experiment (global is reserved)")


class PatchLibraryRequest(BaseModel):
    """Currently only renaming is supported (``library_id`` never changes)."""

    new_name: str


class AddMembersRequest(BaseModel):
    """Add ``work_id`` / DOI pointers into a library (defaults to the effective one)."""

    work_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    added_by: str = Field(default="user", description="user | agent")


# ── experiment-scoped libraries (routes/literature_scope.py) ──────────


class EnsureExperimentLibraryResponse(BaseModel):
    """Outcome of ``POST /literature/experiments/{id}/library`` (idempotent).

    ``created`` distinguishes "minted just now" from "already there"; both are
    ``ok=True`` — the endpoint is a lazy ensure, not a create."""

    ok: bool = False
    degraded: bool = False
    library_id: str = ""
    experiment_id: str = ""
    created: bool = False
    member_count: int = 0
    members_path: str = Field(
        default="",
        description="Absolute path of the authoritative library/members.jsonl; "
                    "empty when the experiment folder could not be resolved (the "
                    "library then lives registry-only, which the message says).")
    message: str = ""


class CopyLibraryRequest(BaseModel):
    """Copy a library's members into an experiment's own library.

    Copying is how the product replaces cross-experiment sharing :
    libraries are never shared, and growth in library count is explicitly fine."""

    to_experiment_id: str = Field(
        default="",
        description="Target experiment. Empty = the currently active experiment.")


class CopyLibraryResponse(BaseModel):
    """Outcome of a copy. ``at_cap`` / ``rejected`` are reported, never swallowed:
    a silently truncated copy would leave the operator believing papers are in a
    library that does not hold them."""

    ok: bool = False
    degraded: bool = False
    src_library_id: str = ""
    library_id: str = ""
    to_experiment_id: str = ""
    copied: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(
        default_factory=list,
        description="Already present in the target; left untouched (their existing "
                    "reason is not overwritten by the source's).")
    rejected: list[str] = Field(default_factory=list)
    member_count: int = 0
    at_cap: bool = False
    message: str = ""


# ── write results (degrade-safe) ──────────────────────────────────────


class MutationResult(BaseModel):
    """Generic write outcome. ``ok=False, degraded=True`` when the live core is
    not wired (contract present, real mutation happens at integration time)."""

    ok: bool = False
    degraded: bool = False
    message: str = ""
    library: Optional[LibrarySummary] = None


class MemberMutationResult(BaseModel):
    """Outcome of an add/remove-members call (mirrors ``add_members``)."""

    ok: bool = False
    degraded: bool = False
    library_id: Optional[str] = None
    added: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    member_count: int = 0
    at_cap: bool = False
    message: str = ""


# ── search ────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Semantic search over the big library, optionally filtered to a library's
    member pointer-set (the panel does big-library search → member filter)."""

    query: str
    k: int = Field(default=8, ge=1, le=200)
    material: str = ""
    library_id: Optional[str] = None


class SearchHit(BaseModel):
    """One search result row (mirrors ``literature_index.search`` dicts)."""

    work_id: str = ""
    title: str = ""
    year: int = 0
    journal: str = ""
    doi: str = ""
    cited: int = 0
    score: float = 0.0
    source: str = ""
    abstract_excerpt: str = ""
    retrieval: str = Field(default="semantic", description="semantic | keyword")


class SearchResponse(BaseModel):
    """Search results. ``degraded`` is True when either the backend is absent
    OR the semantic index fell back to keyword matching (per-row flag) — the UI
    shows a soft warning, results are still usable."""

    hits: list[SearchHit] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


# ── abstract ──────────────────────────────────────────────────────────


class AbstractResponse(BaseModel):
    """Full abstract + bibliographic context for one work
    (``literature_index.fetch_abstract``). ``found=False`` for an unknown id or
    an unprovisioned index; ``degraded`` flags the backend being absent."""

    work_id: str = ""
    found: bool = False
    title: str = ""
    abstract: str = ""
    user_abstract: str = ""
    abstract_provenance: str = ""
    authors: str = ""
    first_author: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    source: str = ""
    cited_by_count: int = 0
    note: str = ""
    degraded: bool = False


__all__ = [
    "LibrarySummary",
    "LibraryMember",
    "LibraryDetail",
    "LibrariesResponse",
    "CreateLibraryRequest",
    "PatchLibraryRequest",
    "AddMembersRequest",
    "EnsureExperimentLibraryResponse",
    "CopyLibraryRequest",
    "CopyLibraryResponse",
    "MutationResult",
    "MemberMutationResult",
    "SearchRequest",
    "SearchHit",
    "SearchResponse",
    "AbstractResponse",
]
