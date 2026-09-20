"""Pydantic request/response models for the literature_cognition parity seam.

Wave-A parity rebuild: the Gradio→TS SPA rewrite lost visibility of several
literature + cognition controls whose LOGIC still lives in the Python core. This
module is the typed contract for the endpoints that re-expose them:

  * POST  /api/literature/ingest       — PDF upload → ``knowledge.ingest.ingest_pdf``
                                          (abstract promoted into the big library)
  * POST  /api/literature/fetch        — DOI/URL → ``knowledge.fetch.try_fetch_fulltext``
  * GET   /api/literature/fetch-board  — the agent-asks-user fetch request board
                                          (``knowledge.fetch_board.list_requests``)
  * GET   /api/literature/ingest-status — subsystem readiness probe for the panel
  * GET   /api/cognition/phases        — conversation-phase summaries
                                          (``CognitionContext.phases.list_phases``)

These shapes mirror the data the old ``gui/literature_panel.py`` /
``gui/cognition_panel.py`` rendered, restated as JSON-serializable models.

EVERY response carries a ``degraded`` boolean so a standalone API process (no
live core wired) returns an empty-but-valid body instead of 500-ing (R: the seam
must boot standalone; the UI must never freeze). Safety / business logic NEVER
lives here — these models only carry data to and from the core.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── POST /api/literature/ingest ──────────────────────────────────────────────
class IngestRequest(BaseModel):
    """Promote a PDF (already on disk) into the big literature library.

    Mirrors ``literature_panel`` ingest: the panel uploaded a PDF and called
    ``ingest.ingest_pdf(pdf_path, work_id, library_id="")`` — passing an EMPTY
    ``library_id`` so the abstract is PROMOTED into the big library and the active
    library only gains a *pointer* (owner directive: 大库是真的库). ``work_id`` is
    optional; the core resolves / mints one (§7.4) and never refuses.

    ``pdf_path`` is a server-visible path (the SPA uploads via the file endpoint
    first, then references the stored path here) — no multipart parsing lives in
    this thin seam.
    """

    pdf_path: str = Field(description="Server-visible path to a PDF already on disk.")
    work_id: str = Field(default="", description="Explicit OpenAlex work_id; resolved/minted if blank.")
    library_id: str = Field(
        default="",
        description="Per-library ATTACH target; blank → promote-only (pointer model).",
    )
    source: str = Field(default="user_pdf", description="user_pdf | user_url | agent.")
    promote: bool = Field(default=True, description="Promote the abstract into the big library.")
    first_author: str = Field(default="", description="Hint used only when minting a synthetic id.")
    year: str = Field(default="", description="Hint used only when minting a synthetic id.")


class IngestResponse(BaseModel):
    """The serializable ``IngestResult`` (ingest.py) plus the seam's degraded flag."""

    ok: bool = False
    work_id: str = ""
    source: str = ""
    n_chunks: int = 0
    slug_dir: str = ""
    promoted: bool = False
    status: str = ""  # ingested | replaced | noop | warning:* | error:*
    doi: str = ""
    title: str = ""
    sha256: str = ""
    detail: str = ""
    library_id: str = ""
    ocr_used: bool = False  # text recovered via qwen-vl-ocr (scanned-PDF fallback)
    degraded: bool = False
    # Post-ingest library curation (set by the upload/manual endpoints, not the
    # bare ingest relay): whether a pointer to the resulting work_id was added to
    # the chosen library, and how many open fetch-board requests it satisfied.
    pointer_added: bool = False
    pointer_library_id: str = Field(
        default="",
        description="The library the pointer ACTUALLY landed in. Echoed because an "
                    "empty request library_id no longer means 'the active library' "
                    "— it means the effective one, i.e. the current experiment's "
                    "own library. The caller must be able to see where its paper "
                    "went without guessing.")
    pointer_library_source: str = Field(
        default="",
        description="How that library was chosen: 'request' (explicit library_id), "
                    "'experiment', 'manual', or 'fallback'.")
    fulltext_ref: str = Field(
        default="",
        description="Relative full-text path recorded on the member, e.g. "
                    "'papers/W123_au111'. Empty when no member row was annotated.")
    fulfilled_requests: int = 0


# ── POST /api/literature/manual-entry ────────────────────────────────────────
class ManualEntryRequest(BaseModel):
    """Register a paper from TYPED metadata — no PDF (#125 '直接管理条目').

    The operator hand-enters a paper that is not in OpenAlex and has no PDF at
    hand. The core mints a synthetic ``local:<hash>`` work_id (unless one is
    given), stores a rich abstract row (retrievable via fetch_abstract with NO
    embed key), and — best-effort — embeds the abstract so the entry is
    semantically searchable too. A pointer is then added to the chosen library.
    """

    title: str = Field(default="", description="Paper title (title OR abstract required).")
    abstract: str = Field(default="", description="Abstract / summary text (title OR abstract required).")
    work_id: str = Field(default="", description="Explicit work_id; a local:<hash> is minted if blank.")
    doi: str = Field(default="", description="Optional DOI.")
    first_author: str = Field(default="", description="Optional first author (also a mint hint).")
    authors: str = Field(default="", description="Optional full author list (free text).")
    year: str = Field(default="", description="Optional publication year.")
    journal: str = Field(default="", description="Optional journal / venue.")
    library_id: str = Field(
        default="", description="Library to add the pointer to; blank → active library."
    )


# ── SI attachments (POST /literature/attach-si, GET /literature/attachments) ──
class AttachmentEntry(BaseModel):
    """One supplementary file belonging to a paper."""

    file: str = Field(default="", description="Stored filename inside <slug>/attachments/.")
    label: str = Field(default="", description="Operator-facing description, e.g. 'Supplementary Note 3'.")
    n_chars: int = Field(default=0, description="Characters of text extracted at upload time.")
    ocr_used: bool = Field(default=False, description="Whether the text came from OCR (no text layer).")
    uploaded_at: str = Field(default="", description="UTC ISO timestamp.")
    registered: bool = Field(
        default=True,
        description="False for a PDF dropped into the directory by hand (no manifest row). "
                    "Still readable — an unregistered file is listed rather than ignored so "
                    "what the system reads matches what the operator can see.")


class AttachSiResponse(BaseModel):
    """Result of attaching one SI file."""

    ok: bool = False
    slug: str = Field(default="", description="The paper's directory name under data/papers/.")
    file: str = Field(default="", description="Stored filename.")
    label: str = ""
    n_chars: int = 0
    ocr_used: bool = False
    duplicate: bool = Field(
        default=False,
        description="True when a byte-identical file was already attached (no second copy made).")
    error: str = ""
    degraded: bool = False


class AttachmentsResponse(BaseModel):
    attachments: list[AttachmentEntry] = Field(default_factory=list)
    slug: str = ""
    degraded: bool = False


# ── POST /api/literature/fetch ───────────────────────────────────────────────
class FetchRequest(BaseModel):
    """Best-effort, ToS-respecting full-text fetch for a DOI / URL.

    Mirrors the literature panel's experimental "paste DOI/URL → MAST tries to
    fetch" control, which called ``fetch.try_fetch_fulltext(doi_or_url,
    auto_oa=True)``. NO paywall bypass / auth exists in the core by construction;
    this seam only relays the input.
    """

    doi_or_url: str = Field(description="A DOI (10.x/… or https://doi.org/…) or an http(s) URL.")
    oa_url: str = Field(default="", description="Optional known open-access PDF URL to try first.")
    auto_oa: bool = Field(default=True, description="Resolve an OA PDF from the DOI before fetching.")


class FetchResponse(BaseModel):
    """The canonical ``try_fetch_fulltext`` result dict, typed."""

    status: str = "unavailable"  # ok_pdf | ok_metadata | blocked | error | unavailable
    pdf_path: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    message: str = ""
    url: str = ""
    source: str = "user_url"
    degraded: bool = False


# ── GET /api/literature/fetch-board ──────────────────────────────────────────
class FetchRequestEntry(BaseModel):
    """One full-text fetch request the literature agent posted for the operator
    (``fetch_board`` record). Open requests surface first in the panel."""

    request_id: str = ""
    work_id: str = ""
    doi: str = ""
    title: str = ""
    reason: str = ""
    requested_by: str = "agent"
    experiment_id: Optional[str] = None
    status: str = "pending"  # pending | fulfilled | failed | dismissed
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None
    note: str = ""


class FetchBoardResponse(BaseModel):
    requests: list[FetchRequestEntry] = Field(default_factory=list)
    count: int = 0
    pending_count: int = 0
    degraded: bool = False


# ── GET /api/literature/ingest-status ────────────────────────────────────────
class IngestStatusResponse(BaseModel):
    """Readiness probe so the panel can show/hide the ingest controls and warn
    about missing deps before the user uploads a PDF (mirrors the panel's honest
    degradation messaging)."""

    available: bool = False  # numpy+pandas present → can embed/index
    can_extract_pdf: bool = False  # PyMuPDF (fitz) present
    can_fetch: bool = False  # httpx present (experimental fetch)
    has_embedder_key: bool = False  # a DashScope key resolved (else keyword fallback)
    can_ocr: bool = False  # fitz + httpx + DashScope key → scanned PDFs OCR-recoverable
    detail: str = ""
    degraded: bool = False


# ── GET /api/cognition/phases ────────────────────────────────────────────────
class PhaseSummary(BaseModel):
    """One conversation-phase summary (sharding output) — mirrors the cognition
    panel's ``render_phase_summaries_html`` rows."""

    phase_index: Optional[int] = None
    title: str = ""
    summary: str = ""
    experiment_id: Optional[str] = None
    started_msg_id: Optional[Any] = None
    ended_msg_id: Optional[Any] = None
    open: bool = False  # ended_msg_id is None → phase still in progress
    created_at: Optional[str] = None


class PhasesResponse(BaseModel):
    phases: list[PhaseSummary] = Field(default_factory=list)
    count: int = 0
    experiment_id: Optional[str] = None
    degraded: bool = False


__all__ = [
    "IngestRequest",
    "IngestResponse",
    "ManualEntryRequest",
    "FetchRequest",
    "FetchResponse",
    "FetchRequestEntry",
    "FetchBoardResponse",
    "IngestStatusResponse",
    "PhaseSummary",
    "PhasesResponse",
]
