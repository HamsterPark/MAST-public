"""Cognition FULL-transcript seam (additive to ``routes/cognition.py``).

``routes/cognition.py`` exposes ``POST /api/cognition/brainstorm`` and
``POST /api/cognition/dream`` as ack/job-only endpoints — they never return the
transcript that ``run_brainstorm`` already computes in a single call. This module
adds NEW paths that DO return the full transcript:

  * POST /api/cognition/brainstorm/full   → run ``run_brainstorm`` and return the
        whole ``{summary, transcript[...], written_memory_ids, degraded}``.
  * POST /api/cognition/brainstorm/stream → SSE shape for parity, BUT the core
        ``run_brainstorm`` only returns at the END (no per-round generator). We
        therefore emit the complete transcript as transcript frames *after* the
        synchronous run plus a final note that per-round streaming is unavailable
        without a generator — we do NOT fake per-round timing.
  * GET  /api/cognition/dream/full        → run one dream pass and return each
        written entry's FULL content (read back from the store).

GRACEFUL DEGRADATION (house rule): the API boots STANDALONE. Every handler
resolves the live ``MemoryStore`` + DB path from the app context; if absent or any
backend call raises, it returns ``degraded=True`` with an empty transcript (JSON)
or a single error/degraded SSE frame (streaming) — NEVER 500. Heavy core modules
(``mast.agents.brainstorm.graph`` / ``mast.memory.*``) are LAZY-imported inside
the handlers. No business/orchestration logic lives here — the seam only relays to
the kept core.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from mast.api.sse import SSE_HEADERS, with_heartbeat
from mast.api.schemas_cognition_transcript import (
    BrainstormFullRequest,
    BrainstormFullResponse,
    BrainstormTurn,
    DreamEntry,
    DreamFullResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cognition"])

# Roles in the brainstorm transcript that are NOT a viewpoint agent.
_NON_VIEWPOINT_ROLES = {"facilitator", "user", ""}


# ── live-store / db resolution (graceful) — mirrors routes/cognition.py ─


def _resolve_store(ctx: Any) -> Optional[Any]:
    """Best-effort resolve a live ``MemoryStore`` from the app context.

    Order: explicit ``ctx.memory_store`` → build one off ``ctx.experiment_storage``
    (they share the experiment DB file). Returns ``None`` (→ degraded) when
    nothing is wired or anything raises. Heavy import is lazy + guarded so the API
    still boots standalone."""
    store = getattr(ctx, "memory_store", None)
    if store is not None:
        return store
    storage = getattr(ctx, "experiment_storage", None)
    if storage is None:
        return None
    try:
        from mast.memory.store import MemoryStore

        return MemoryStore.from_storage(storage)
    except Exception as exc:  # any wiring/shape mismatch → degrade, never 500
        logger.warning("memory store resolve failed: %s", exc)
        return None


def _db_path_from(ctx: Any, store: Any) -> Optional[str]:
    """Best-effort DB path for grounding (brainstorm/dream read the experiment DB)."""
    for src in (store, getattr(ctx, "experiment_storage", None)):
        if src is None:
            continue
        p = getattr(src, "_db_path", None)
        if p is not None:
            return str(p)
    return None


def _resolve_llm(ctx: Any) -> Optional[Any]:
    """Best-effort live cognition LLM (a langchain chat model OR callable(str)->str).

    ``run_brainstorm`` accepts ``llm=None`` and runs a dependency-free rule-based
    discussion (no network / API key), so a missing LLM is NOT degradation — it
    just produces the offline transcript. We only relay a model the app may have
    wired; never construct one here."""
    for attr in ("cognition_llm", "llm", "chat_model"):
        m = getattr(ctx, attr, None)
        if m is not None:
            return m
    return None


def _to_turns(transcript: list) -> list[BrainstormTurn]:
    """Coerce core transcript dicts into typed turns (role→viewpoint surfacing)."""
    out: list[BrainstormTurn] = []
    for t in transcript or []:
        if not isinstance(t, dict):
            continue
        role = str(t.get("role") or "")
        viewpoint = None if role in _NON_VIEWPOINT_ROLES else role
        out.append(
            BrainstormTurn(
                round=int(t.get("round", 0) or 0),
                speaker=str(t.get("speaker") or ""),
                role=role,
                viewpoint=viewpoint,
                content=str(t.get("content") or ""),
            )
        )
    return out


def _brainstorm_memory_path(topic: str) -> str:
    """The deterministic path ``run_brainstorm`` writes its summary to when a
    memory store is supplied (mirrors graph.run_brainstorm's slug rule — relay,
    not reimplementation of the write itself)."""
    slug = (topic or "session").strip().replace("/", "_")[:40] or "session"
    return f"brainstorms/{slug}.md"


def _run_brainstorm(ctx: Any, body: BrainstormFullRequest) -> tuple[dict, bool, str]:
    """Resolve the store/DB and run ``run_brainstorm`` once.

    Returns ``(result, degraded, message)`` where ``result`` is
    ``{"transcript", "summary"}`` (empty on degrade). Never raises."""
    store = _resolve_store(ctx)
    db_path = _db_path_from(ctx, store)
    if store is None or db_path is None:
        return {"transcript": [], "summary": ""}, True, (
            "memory store / experiment DB not wired"
        )
    try:
        from mast.agents.brainstorm.graph import run_brainstorm
    except Exception as exc:  # backend unavailable → degrade, never 500
        logger.warning("brainstorm backend unavailable: %s", exc)
        return {"transcript": [], "summary": ""}, True, (
            "brainstorm backend not available"
        )
    try:
        result = run_brainstorm(
            db_path,
            body.experiment_id or None,
            topic=body.topic or "",
            user_viewpoints=list(body.viewpoints or []),
            max_rounds=max(1, int(body.max_rounds)),
            llm=_resolve_llm(ctx),
            memory_store=store,
        )
    except Exception as exc:  # the core is documented as never-raise; be defensive
        logger.warning("brainstorm run failed: %s", exc)
        return {"transcript": [], "summary": ""}, True, f"brainstorm failed: {exc}"
    # Validate the core's contract: a result missing transcript/summary is a
    # malformed backend response → surface as degraded, not a silent empty 200.
    if not isinstance(result, dict) or "transcript" not in result or "summary" not in result:
        logger.warning("brainstorm result malformed: %r", type(result).__name__)
        return {"transcript": [], "summary": ""}, True, "brainstorm 返回结构异常（缺 transcript/summary）"
    return result, False, ""


# ── POST /api/cognition/brainstorm/full ────────────────────────────────


@router.post("/cognition/brainstorm/full", response_model=BrainstormFullResponse)
def cognition_brainstorm_full(
    request: Request, body: BrainstormFullRequest
) -> BrainstormFullResponse:
    """Run a facilitated brainstorm and return the FULL transcript + summary.

    ``run_brainstorm`` computes the whole discussion in one call (it has no
    per-round generator), so this synchronous endpoint returns everything at once.
    Degrade-safe: no store/DB → ``degraded=True`` with an empty transcript."""
    result, degraded, message = _run_brainstorm(request.app.state.ctx, body)
    if degraded:
        return BrainstormFullResponse(degraded=True, message=message)

    summary = str(result.get("summary") or "")
    turns = _to_turns(result.get("transcript") or [])
    # run_brainstorm persists the summary to memory itself when a store is given;
    # report the deterministic path it used (best-effort, relay only).
    written: list[str] = []
    if summary.strip():
        written.append(_brainstorm_memory_path(body.topic or ""))
    return BrainstormFullResponse(
        summary=summary,
        transcript=turns,
        written_memory_ids=written,
        degraded=False,
    )


# ── POST /api/cognition/brainstorm/stream (SSE shape; no real per-round) ─


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/cognition/brainstorm/stream")
def cognition_brainstorm_stream(
    request: Request, body: BrainstormFullRequest
) -> StreamingResponse:
    """SSE variant for transcript parity. IMPORTANT: the core ``run_brainstorm``
    only returns at the END (no per-round generator), so we run it synchronously
    and then emit the already-computed transcript as one frame per turn, followed
    by a ``done`` frame. We do NOT fabricate per-round streaming timing.

    Frames: {kind:"turn", ...} per transcript line → {kind:"summary", content} →
    {kind:"done", streaming:false, ...}. Degrades to a single error frame when no
    store/DB is wired — never 500 mid-stream."""
    ctx = request.app.state.ctx

    def gen():
        result, degraded, message = _run_brainstorm(ctx, body)
        if degraded:
            yield _sse({"kind": "error", "degraded": True, "message": message})
            yield _sse({"kind": "done", "degraded": True, "streaming": False})
            return
        try:
            for turn in _to_turns(result.get("transcript") or []):
                yield _sse({"kind": "turn", **turn.model_dump()})
            yield _sse({"kind": "summary", "content": str(result.get("summary") or "")})
        except Exception as exc:  # surface as a frame, never 500 mid-stream
            logger.warning("brainstorm stream emit failed: %s", exc)
            yield _sse({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            # Honesty: true per-round streaming needs a generator the core does
            # not expose; this is a post-hoc replay of the full transcript.
            yield _sse({
                "kind": "done",
                "streaming": False,
                "note": "run_brainstorm has no per-round generator; "
                        "transcript emitted post-run, not streamed live.",
            })

    # Worst-case silence in the codebase: `_run_brainstorm` runs to completion
    # BEFORE the first frame, so a multi-round brainstorm is minutes of dead air
    # on the socket — exactly what an idle reaper on a Tailscale link cuts .
    return StreamingResponse(
        with_heartbeat(gen(), label="brainstorm"),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ── GET /api/cognition/dream/full ──────────────────────────────────────


@router.get("/cognition/dream/full", response_model=DreamFullResponse)
def cognition_dream_full(
    request: Request,
    namespace: str = Query(default="global", description="Where dream entries land."),
) -> DreamFullResponse:
    """Run one dreaming-consolidation pass and return each written entry's FULL
    content (read back from the store; ``dream_once`` itself returns only paths).

    Degrade-safe: no store/DB → ``degraded=True`` empty. The offline rule-based
    consolidator runs with no LLM key, so a wired store yields a real (possibly
    empty, when there are no experiments) result without ever 500-ing."""
    ctx = request.app.state.ctx
    store = _resolve_store(ctx)
    db_path = _db_path_from(ctx, store)
    if store is None or db_path is None:
        return DreamFullResponse(degraded=True, message="memory store / experiment DB not wired")

    try:
        from mast.memory.dreaming import DreamingService

        svc = DreamingService(db_path, store, namespace=namespace or "global")
        written = svc.dream_once() or []
    except Exception as exc:
        logger.warning("dream pass failed: %s", exc)
        return DreamFullResponse(degraded=True, message=f"dream failed: {exc}")

    entries: list[DreamEntry] = []
    for w in written:
        if not isinstance(w, dict):
            continue
        path = str(w.get("path") or "")
        row = None
        try:
            row = store.read(namespace or "global", path)
        except Exception as exc:  # read-back best-effort; never break the response
            logger.debug("dream entry read-back failed for %s: %s", path, exc)
            row = None
        entries.append(
            DreamEntry(
                path=path,
                title=str((row or {}).get("title") if row else w.get("title", "") or ""),
                kind=str((row or {}).get("kind", "dream") if row else "dream") or "dream",
                content=str((row or {}).get("content", "") if row else ""),
            )
        )
    return DreamFullResponse(entries=entries, count=len(entries), degraded=False)


__all__ = ["router"]
