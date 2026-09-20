"""Wave-A parity rebuild — literature_cognition domain.

The UI was rewritten Gradio→TS SPA and lost visibility of functionality whose
LOGIC still lives in the Python core. This module re-exposes the missing pieces
as typed FastAPI endpoints so the frontend can render them again. It is ADDITIVE
to the existing ``routes/literature.py`` (libraries + search + abstract) and
``routes/cognition.py`` (memory CRUD + dream + brainstorm) — it only adds the
gaps those two do not cover.

Endpoints (each wraps a kept backend fn; this seam is a THIN relay only):

  * POST  /api/literature/ingest        → ``knowledge.ingest.ingest_pdf``
        PDF (on disk) → abstract promoted into the big library (panel passed an
        empty ``library_id`` so the active library only gains a pointer).
  * POST  /api/literature/fetch         → ``knowledge.fetch.try_fetch_fulltext``
        DOI/URL → best-effort, ToS-respecting full-text fetch (auto_oa=True like
        the panel). No paywall bypass exists in the core by construction.
  * GET   /api/literature/fetch-board   → ``knowledge.fetch_board.list_requests``
        the agent-asks-user fetch request board (+ pending_count).
  * GET   /api/literature/ingest-status → dep/key readiness probe for the panel.
  * GET   /api/cognition/phases         → ``CognitionContext.phases.list_phases``
        conversation-phase summaries (sharding output).

GRACEFUL DEGRADATION is mandatory (house rule 2): the API boots STANDALONE with
no live core wired. Every handler checks ctx / lazy-imports its heavy backend in
try/except; if the backend is absent or any call raises, it returns a valid
empty/degraded body (``degraded=True`` / ``ok=False``) — NEVER a 500. Safety /
business logic NEVER lives here (R6) — it stays in the core; the seam only relays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, Query, Request, UploadFile

from mast.api.schemas_literature_cognition import (
    AttachmentEntry,
    AttachmentsResponse,
    AttachSiResponse,
    FetchBoardResponse,
    FetchRequest,
    FetchRequestEntry,
    FetchResponse,
    IngestRequest,
    IngestResponse,
    IngestStatusResponse,
    ManualEntryRequest,
    PhaseSummary,
    PhasesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["literature_cognition"])


# ── db-path resolution for cognition phases (graceful) ──────────────────


def _db_path_from(ctx: Any) -> Optional[str]:
    """Best-effort DB path for building a PhaseManager / CognitionContext.

    Order: an explicit ``ctx.cognition`` / ``ctx.memory_store`` (future wiring) →
    the wired ``ctx.experiment_storage`` (they share the experiment DB file).
    Returns None (→ degraded) when nothing is wired."""
    for attr in ("cognition", "memory_store", "experiment_storage"):
        src = getattr(ctx, attr, None)
        if src is None:
            continue
        p = getattr(src, "_db_path", None)
        if p is not None:
            return str(p)
    return None


# ── POST /api/literature/ingest ─────────────────────────────────────────


@router.post("/literature/ingest", response_model=IngestResponse)
def ingest_pdf_endpoint(request: Request, body: IngestRequest) -> IngestResponse:
    """Ingest a PDF (already on disk) and promote its abstract into the big
    library (mirrors literature_panel's upload control → ``ingest.ingest_pdf``).

    The core never refuses on missing deps / bad input — it returns an
    ``IngestResult`` with an ``error:*`` / ``warning:*`` status, which we surface
    as ``ok`` driven by that status. A truly absent backend → typed degraded."""
    try:
        from mast.knowledge import ingest as ingest_mod  # heavy: numpy/pandas/fitz
    except Exception as exc:  # backend unavailable → degrade, never 500
        logger.warning("literature ingest backend import failed: %s", exc)
        return IngestResponse(ok=False, status="error:unavailable", degraded=True)

    try:
        res = ingest_mod.ingest_pdf(
            body.pdf_path,
            work_id=body.work_id,
            library_id=body.library_id,
            source=body.source or "user_pdf",
            promote=body.promote,
            first_author=body.first_author,
            year=body.year,
        )
        data = res.to_dict() if hasattr(res, "to_dict") else dict(res)
    except Exception as exc:  # the core is degrade-safe, but never 500 from here
        # logger.exception 而非 warning:入库失败是**数据依赖**的 —— 同一份 PDF
        # 在别的机器上好好的。没有栈,一个可复现的缺陷就变成猜谜(agent 路径当时
        # 已经记栈,这条路由没有,于是多绕了一轮才查清)。
        logger.exception("literature ingest failed: %s", exc)
        return IngestResponse(ok=False, status="error:ingest_failed", detail=str(exc), degraded=True)

    status = str(data.get("status", "") or "")
    ok = status in ("ingested", "replaced", "noop")
    work_id = str(data.get("work_id", "") or "")

    # Close the board and wake whoever asked. This endpoint is the third way a
    # paper arrives — the operator's "PDF 摄取（服务器路径）" control, and the
    # landing spot after a successful DOI fetch hands back a pdf_path — and it
    # was the one that never ran the curation tail. A request satisfied through
    # it stayed 'pending' on the board forever, so the operator saw an unanswered
    # ask for a paper they had just supplied.
    cur: dict = {}
    if ok and work_id:
        slug_name = ""
        try:
            from pathlib import Path as _Path
            slug_name = _Path(str(data.get("slug_dir", "") or "")).name
        except Exception:  # pragma: no cover - defensive
            slug_name = ""
        cur = _add_pointer_and_close_board(
            work_id, body.library_id, source=body.source or "user_pdf",
            reason="server-path ingest", slug=slug_name)

    # ``ingest_pdf`` already recorded the full-text ref against the effective
    # library; echo which one that was so an empty request library_id does not
    # leave the caller guessing where its paper was filed.
    eff, eff_src = "", ""
    try:
        from mast.knowledge.experiment_library import resolve_effective_library
        eff, eff_src = resolve_effective_library()
    except Exception as exc:  # pragma: no cover - informational only
        logger.info("effective library unresolved: %s", exc)
    return IngestResponse(
        ok=ok,
        work_id=work_id,
        source=str(data.get("source", "") or ""),
        n_chunks=int(data.get("n_chunks", 0) or 0),
        slug_dir=str(data.get("slug_dir", "") or ""),
        promoted=bool(data.get("promoted", False)),
        status=status,
        doi=str(data.get("doi", "") or ""),
        title=str(data.get("title", "") or ""),
        sha256=str(data.get("sha256", "") or ""),
        detail=str(data.get("detail", "") or ""),
        library_id=(str(data.get("library_id", "") or "")
                    or str(cur.get("pointer_library_id", "") or "") or eff),
        pointer_added=bool(cur.get("pointer_added", False)),
        pointer_library_source=("request" if body.library_id
                                else (str(cur.get("pointer_library_source", "") or "")
                                      or eff_src)),
        fulltext_ref=str(cur.get("fulltext_ref", "") or ""),
        fulfilled_requests=int(cur.get("fulfilled_requests", 0) or 0),
        ocr_used=bool(data.get("ocr_used", False)),
        degraded=False,
    )


# ── shared curation: add a library pointer + close any open fetch request ──


def _add_pointer_and_close_board(
    work_id: str, library_id: str, *, source: str, reason: str, slug: str = "",
) -> dict:
    """Route-facing shim over :func:`knowledge.fulfilment.curate_ingested_fulltext`.

    The implementation moved down to the knowledge layer so the literature agent
    can run the same tail (``agents → api`` would be backwards). This wrapper
    stays for the response shape the upload/manual endpoints build from, and
    pins ``added_by="user"`` because everything reaching it came from an operator
    action. Still never raises.
    """
    try:
        from mast.knowledge.fulfilment import curate_ingested_fulltext
    except Exception as exc:  # pragma: no cover - backend absent → degrade
        logger.warning("fulfilment backend unavailable: %s", exc)
        return {"pointer_added": False, "pointer_library_id": "",
                "pointer_library_source": "", "fulltext_ref": "",
                "fulfilled_requests": 0, "resumed": 0}
    return curate_ingested_fulltext(
        work_id, library_id, source=source, reason=reason, slug=slug,
        added_by="user")


# ── POST /api/literature/upload-pdf (multipart) ─────────────────────────


@router.post("/literature/upload-pdf", response_model=IngestResponse)
async def upload_pdf_endpoint(
    request: Request,
    file: UploadFile = File(..., description="The PDF file to ingest."),
    library_id: str = Form(default=""),
    work_id: str = Form(default=""),
    promote: bool = Form(default=True),
    first_author: str = Form(default=""),
    year: str = Form(default=""),
) -> IngestResponse:
    """Receive a browser-uploaded PDF, save it to a temp path, ingest it into the
    big library, and add a POINTER to the chosen library (#125 附加 PDF).

    This is the browser-facing counterpart of ``POST /literature/ingest`` (which
    takes a server-visible path the SPA cannot produce). The upload is streamed to
    a temp file, ``ingest.ingest_pdf`` copies it into ``data/papers/<slug>/`` and
    the temp file is deleted. Degrade-safe: any dep/import problem → typed
    degraded body, never a 500."""
    import os
    import tempfile

    filename = (file.filename or "").strip()
    if filename and not filename.lower().endswith(".pdf"):
        return IngestResponse(
            ok=False, status="error:not_a_pdf",
            detail=f"仅支持 PDF 上传（收到 {filename!r}）。", degraded=False,
        )

    try:
        from mast.knowledge import ingest as ingest_mod  # heavy: numpy/pandas/fitz
    except Exception as exc:
        logger.warning("upload-pdf ingest backend import failed: %s", exc)
        return IngestResponse(ok=False, status="error:unavailable", degraded=True)

    # Stream the upload to a temp file (bounded read; UploadFile spools large
    # bodies to disk already, so .read() is memory-safe for typical papers).
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="mast_upload_")
        with os.fdopen(fd, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:
        logger.warning("upload-pdf save failed: %s", exc)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return IngestResponse(ok=False, status="error:upload_failed", detail=str(exc), degraded=False)
    finally:
        try:
            await file.close()
        except Exception:
            pass

    try:
        # library_id="" on purpose (pointer model): PROMOTE into the big library,
        # then add a pointer to the chosen library below (mirrors ingest_pdf_h).
        res = ingest_mod.ingest_pdf(
            tmp_path, work_id=(work_id or "").strip(), library_id="",
            source="user_pdf", promote=promote,
            first_author=(first_author or "").strip(), year=(year or "").strip(),
        )
        data = res.to_dict() if hasattr(res, "to_dict") else dict(res)
    except Exception as exc:
        logger.exception("upload-pdf ingest failed: %s", exc)
        return IngestResponse(ok=False, status="error:ingest_failed", detail=str(exc), degraded=True)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    status = str(data.get("status", "") or "")
    ok = status in ("ingested", "replaced", "noop")
    wid = str(data.get("work_id", "") or "")

    cur: dict = {}
    if ok and wid:
        cur = _add_pointer_and_close_board(
            wid, library_id, source="user_pdf", reason="ingested full text",
            slug=Path(str(data.get("slug_dir", "") or "")).name,
        )

    return IngestResponse(
        ok=ok,
        work_id=wid,
        source=str(data.get("source", "") or ""),
        n_chunks=int(data.get("n_chunks", 0) or 0),
        slug_dir=str(data.get("slug_dir", "") or ""),
        promoted=bool(data.get("promoted", False)),
        status=status,
        doi=str(data.get("doi", "") or ""),
        title=str(data.get("title", "") or ""),
        sha256=str(data.get("sha256", "") or ""),
        detail=str(data.get("detail", "") or ""),
        # Echo where the pointer really went, not the (possibly empty) request value.
        library_id=str(cur.get("pointer_library_id") or data.get("library_id", "") or ""),
        ocr_used=bool(data.get("ocr_used", False)),
        degraded=False,
        pointer_added=bool(cur.get("pointer_added", False)),
        pointer_library_id=str(cur.get("pointer_library_id", "") or ""),
        pointer_library_source=str(cur.get("pointer_library_source", "") or ""),
        fulltext_ref=str(cur.get("fulltext_ref", "") or ""),
        fulfilled_requests=int(cur.get("fulfilled_requests", 0) or 0),
    )


# ── SI attachments ──────────────────────────────────────────────────────


@router.post("/literature/attach-si", response_model=AttachSiResponse)
async def attach_si_endpoint(
    request: Request,
    file: UploadFile = File(..., description="The supplementary PDF."),
    # Blank is answered with a readable error rather than a 422, like every other
    # endpoint in this file — the SPA renders `error`, not validation envelopes.
    work_id: str = Form(default="", description="The paper this supplement belongs to."),
    label: str = Form(default="", description="e.g. 'Supplementary Note 3'."),
    ocr: bool = Form(default=True, description="Allow OCR when there is no text layer."),
) -> AttachSiResponse:
    """Attach a supplementary-information PDF to an already-ingested paper.

    Methods details increasingly live in the SI rather than the body, so reading
    only the main PDF and reporting "the paper does not state the setpoint" is
    often simply wrong about the paper.

    The supplement is deliberately NOT ingested as a paper: no work_id, no
    abstract row, nothing promoted into the big library. It belongs to one paper
    and is read when that paper is read. Text (including the OCR fallback for
    scanned supplements) is extracted here, once, so that reading stays cheap.
    """
    filename = (file.filename or "").strip()
    if filename and not filename.lower().endswith(".pdf"):
        return AttachSiResponse(ok=False, error=f"仅支持 PDF（收到 {filename!r}）。")
    if not (work_id or "").strip():
        return AttachSiResponse(ok=False, error="需要 work_id。")

    try:
        from mast.knowledge import attachments as att_mod
    except Exception as exc:
        logger.warning("attachments backend import failed: %s", exc)
        return AttachSiResponse(ok=False, error="附件后端不可用", degraded=True)

    import os
    import tempfile

    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="mast_si_")
        with os.fdopen(fd, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:
        logger.warning("attach-si save failed: %s", exc)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return AttachSiResponse(ok=False, error=str(exc))
    finally:
        try:
            await file.close()
        except Exception:
            pass

    try:
        res = att_mod.attach_si((work_id or "").strip(), tmp_path,
                                label=label, ocr=ocr, filename=filename)
    except Exception as exc:  # pragma: no cover — the core never raises
        logger.warning("attach-si failed: %s", exc)
        return AttachSiResponse(ok=False, error=str(exc), degraded=True)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return AttachSiResponse(
        ok=bool(res.get("ok", False)),
        slug=str(res.get("slug", "") or ""),
        file=str(res.get("file", "") or ""),
        label=str(res.get("label", "") or ""),
        n_chars=int(res.get("n_chars", 0) or 0),
        ocr_used=bool(res.get("ocr_used", False)),
        duplicate=bool(res.get("duplicate", False)),
        error=str(res.get("error", "") or ""),
    )


@router.get("/literature/attachments", response_model=AttachmentsResponse)
def list_attachments_endpoint(
    request: Request,
    work_id: str = Query(..., description="The paper whose supplements to list."),
) -> AttachmentsResponse:
    """List a paper's supplementary files. Empty (not degraded) when it has none."""
    try:
        from mast.knowledge import attachments as att_mod
    except Exception as exc:
        logger.warning("attachments backend import failed: %s", exc)
        return AttachmentsResponse(degraded=True)
    try:
        rows = att_mod.list_attachments((work_id or "").strip()) or []
        adir = att_mod.attachments_dir((work_id or "").strip())
    except Exception as exc:  # pragma: no cover — the core never raises
        logger.warning("attachment listing failed: %s", exc)
        return AttachmentsResponse(degraded=True)
    return AttachmentsResponse(
        attachments=[AttachmentEntry(**r) for r in rows],
        slug=(adir.parent.name if adir is not None else ""),
    )


# ── POST /api/literature/manual-entry ───────────────────────────────────


@router.post("/literature/manual-entry", response_model=IngestResponse)
def manual_entry_endpoint(request: Request, body: ManualEntryRequest) -> IngestResponse:
    """Register a paper from TYPED metadata (no PDF) and add a pointer to the
    chosen library (#125 '直接管理条目').

    Wraps ``ingest.ingest_manual``: mint/resolve a work_id, store a rich abstract
    row (retrievable with no embed key), best-effort embed for semantic search,
    then add a pointer. Degrade-safe."""
    try:
        from mast.knowledge import ingest as ingest_mod
    except Exception as exc:
        logger.warning("manual-entry backend import failed: %s", exc)
        return IngestResponse(ok=False, status="error:unavailable", degraded=True)

    if not (body.title or "").strip() and not (body.abstract or "").strip():
        return IngestResponse(
            ok=False, status="error:empty",
            detail="需要至少提供标题或摘要。", degraded=False,
        )

    try:
        res = ingest_mod.ingest_manual(
            title=body.title, abstract=body.abstract, work_id=body.work_id,
            doi=body.doi, first_author=body.first_author, authors=body.authors,
            year=body.year, journal=body.journal, source="user_manual",
        )
        data = res.to_dict() if hasattr(res, "to_dict") else dict(res)
    except Exception as exc:
        logger.exception("manual-entry ingest failed: %s", exc)
        return IngestResponse(ok=False, status="error:ingest_failed", detail=str(exc), degraded=True)

    status = str(data.get("status", "") or "")
    ok = status in ("ingested", "ingested_no_embed")
    wid = str(data.get("work_id", "") or "")

    cur: dict = {}
    if ok and wid:
        # No slug: a manual entry has no PDF, so there is no full text to point at
        # and fulltext_status honestly stays "none".
        cur = _add_pointer_and_close_board(
            wid, body.library_id, source="user_manual", reason="manual entry",
        )

    return IngestResponse(
        ok=ok,
        work_id=wid,
        source=str(data.get("source", "") or ""),
        promoted=bool(data.get("promoted", False)),
        status=status,
        doi=str(data.get("doi", "") or ""),
        title=str(data.get("title", "") or ""),
        detail=str(data.get("detail", "") or ""),
        library_id=str(cur.get("pointer_library_id", "") or ""),
        degraded=False,
        pointer_added=bool(cur.get("pointer_added", False)),
        pointer_library_id=str(cur.get("pointer_library_id", "") or ""),
        pointer_library_source=str(cur.get("pointer_library_source", "") or ""),
        fulfilled_requests=int(cur.get("fulfilled_requests", 0) or 0),
    )


# ── POST /api/literature/fetch ──────────────────────────────────────────


@router.post("/literature/fetch", response_model=FetchResponse)
def fetch_fulltext_endpoint(request: Request, body: FetchRequest) -> FetchResponse:
    """Best-effort, ToS-respecting full-text fetch for a DOI/URL (mirrors the
    panel's experimental control → ``fetch.try_fetch_fulltext`` with
    ``auto_oa=True``). The core always returns a JSON dict, never raises, and
    never bypasses a paywall — this seam only relays the input + result."""
    try:
        from mast.knowledge import fetch as fetch_mod  # httpx is optional inside
    except Exception as exc:
        logger.warning("literature fetch backend import failed: %s", exc)
        return FetchResponse(
            status="unavailable",
            message="网络组件不可用，无法自动获取。请手动下载 PDF 后上传。",
            degraded=True,
        )

    try:
        res = fetch_mod.try_fetch_fulltext(
            body.doi_or_url,
            oa_url=body.oa_url or "",
            auto_oa=body.auto_oa,
        )
        data = dict(res) if isinstance(res, dict) else {}
    except Exception as exc:  # defensive — the core is documented as never-raise
        logger.warning("literature fetch failed: %s", exc)
        return FetchResponse(status="error", message=str(exc), degraded=True)

    return FetchResponse(
        status=str(data.get("status", "unavailable") or "unavailable"),
        pdf_path=data.get("pdf_path"),
        metadata=data.get("metadata"),
        message=str(data.get("message", "") or ""),
        url=str(data.get("url", "") or ""),
        source=str(data.get("source", "user_url") or "user_url"),
        degraded=False,
    )


# ── GET /api/literature/fetch-board ─────────────────────────────────────


@router.get("/literature/fetch-board", response_model=FetchBoardResponse)
def fetch_board_endpoint(
    request: Request,
    status: str | None = Query(default=None, description="pending|fulfilled|failed|dismissed"),
) -> FetchBoardResponse:
    """The agent-asks-user full-text fetch request board (mirrors the panel's
    request list → ``fetch_board.list_requests``). Open requests sort first. A
    missing backend / unreadable board → typed empty-not-broken response."""
    try:
        from mast.knowledge import fetch_board as board_mod
    except Exception as exc:
        logger.warning("fetch board backend import failed: %s", exc)
        return FetchBoardResponse(degraded=True)

    try:
        rows = board_mod.list_requests(status) or []
        pending = int(board_mod.pending_count())
    except Exception as exc:
        logger.warning("fetch board read failed: %s", exc)
        return FetchBoardResponse(degraded=True)

    entries = [
        FetchRequestEntry(
            request_id=str(r.get("request_id", "") or ""),
            work_id=str(r.get("work_id", "") or ""),
            doi=str(r.get("doi", "") or ""),
            title=str(r.get("title", "") or ""),
            reason=str(r.get("reason", "") or ""),
            requested_by=str(r.get("requested_by", "agent") or "agent"),
            experiment_id=r.get("experiment_id"),
            status=str(r.get("status", "pending") or "pending"),
            created_at=r.get("created_at"),
            resolved_at=r.get("resolved_at"),
            note=str(r.get("note", "") or ""),
        )
        for r in rows
        if isinstance(r, dict)
    ]
    return FetchBoardResponse(
        requests=entries, count=len(entries), pending_count=pending, degraded=False
    )


# ── GET /api/literature/ingest-status ───────────────────────────────────


@router.get("/literature/ingest-status", response_model=IngestStatusResponse)
def ingest_status_endpoint(request: Request) -> IngestStatusResponse:
    """Readiness probe for the ingest/fetch controls: are numpy/pandas (embed +
    index), PyMuPDF (PDF extract), httpx (experimental fetch), and a DashScope
    embedder key available? Lets the panel warn honestly before an upload. Never
    raises — any probe failure degrades to the conservative 'unavailable'."""
    try:
        from mast.knowledge import ingest as ingest_mod

        available = ingest_mod.np is not None and ingest_mod.pd is not None
    except Exception as exc:
        logger.warning("ingest-status probe failed: %s", exc)
        return IngestStatusResponse(detail=str(exc), degraded=True)

    can_extract = False
    try:
        import fitz  # noqa: F401  (PyMuPDF)

        can_extract = True
    except Exception:
        can_extract = False

    can_fetch = False
    try:
        from mast.knowledge import fetch as fetch_mod

        can_fetch = fetch_mod.httpx is not None
    except Exception:
        can_fetch = False

    has_key = False
    try:
        ingest_mod._load_dashscope_key()
        has_key = True
    except Exception:
        has_key = False

    # Scanned-PDF OCR readiness (fitz + httpx + a DashScope key). Never raises.
    can_ocr = False
    try:
        from mast.knowledge import ocr as ocr_mod

        can_ocr = bool(ocr_mod.ocr_available())
    except Exception:
        can_ocr = False

    if available:
        detail = "就绪：可嵌入并写入大库。" if has_key else "可用，但缺 DashScope key → 语义检索降级为关键词匹配。"
        if can_ocr:
            detail += " 扫描版 PDF 可自动 OCR。"
    else:
        detail = "numpy/pandas 不可用 — 无法嵌入/索引 PDF。"
    return IngestStatusResponse(
        available=bool(available),
        can_extract_pdf=bool(can_extract),
        can_fetch=bool(can_fetch),
        has_embedder_key=bool(has_key),
        can_ocr=can_ocr,
        detail=detail,
        degraded=False,
    )


# ── GET /api/cognition/phases ───────────────────────────────────────────


@router.get("/cognition/phases", response_model=PhasesResponse)
def cognition_phases_endpoint(
    request: Request,
    experiment_id: str | None = Query(default=None, description="Scope to one experiment."),
) -> PhasesResponse:
    """Conversation-phase summaries (sharding output) — mirrors the cognition
    panel's ``render_phase_summaries_html`` over ``PhaseManager.list_phases``.

    Resolves a live ``CognitionContext.phases`` from the app context; with no DB
    wired we degrade to an empty list (never 500). The PhaseManager is read-only
    here. Heavy ``mast.memory.sharding`` is lazy-imported inside the handler."""
    ctx = request.app.state.ctx

    # Prefer an already-wired CognitionContext (has the assembled PhaseManager).
    pm = None
    cog = getattr(ctx, "cognition", None)
    if cog is not None:
        pm = getattr(cog, "phases", None)

    if pm is None:
        db_path = _db_path_from(ctx)
        if db_path is None:
            return PhasesResponse(experiment_id=experiment_id, degraded=True)
        try:
            from mast.memory.sharding import PhaseManager

            pm = PhaseManager(db_path)
        except Exception as exc:
            logger.warning("phase manager build failed: %s", exc)
            return PhasesResponse(experiment_id=experiment_id, degraded=True)

    try:
        phases = pm.list_phases(experiment_id) or []
    except Exception as exc:
        logger.warning("list phases failed: %s", exc)
        return PhasesResponse(experiment_id=experiment_id, degraded=True)

    out: list[PhaseSummary] = []
    for p in phases:
        if not isinstance(p, dict):
            continue
        out.append(
            PhaseSummary(
                phase_index=p.get("phase_index"),
                title=str(p.get("title") or "") or f"阶段 {p.get('phase_index')}",
                summary=str(p.get("summary") or "") or "(未摘要)",
                experiment_id=p.get("experiment_id"),
                started_msg_id=p.get("started_msg_id"),
                ended_msg_id=p.get("ended_msg_id"),
                open=p.get("ended_msg_id") is None,
                created_at=p.get("created_at"),
            )
        )
    return PhasesResponse(
        phases=out, count=len(out), experiment_id=experiment_id, degraded=False
    )
