"""Cognition domain (Domain F) — memory CRUD + pin + search, dreaming, brainstorm.

Mirrors the live Gradio handlers in ``mast.webui.cognition_panel`` and the backend
contracts in ``mast.memory.store`` / ``mast.memory.dreaming`` /
``mast.agents.brainstorm.graph`` — but exposes them as a typed FastAPI seam.

Graceful degradation is mandatory (the API must boot STANDALONE with no live core
wired): every handler resolves the live ``MemoryStore`` through the app context;
if it is absent, or any backend call raises, the handler returns a valid
empty/degraded response (``degraded=True``) — NEVER 500, NEVER crashing on
import. Heavy core modules are LAZY-imported INSIDE handlers, exactly like
``routes/skills.py`` imports ``builder_api``.

No business logic / safety checks live here — the API only forwards to the core.
Long ops (dream / brainstorm) currently return an ack/job shape; SSE streaming
and real-singleton wiring are integrated later at integration.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Query, Request

from mast.api.schemas_cognition import (
    BrainstormRequest,
    CognitionJobAck,
    DreamRequest,
    MemoryDeleteResult,
    MemoryDetailResponse,
    MemoryEntry,
    MemoryListResponse,
    MemoryNamespacesResponse,
    MemoryPinRequest,
    MemoryPinResult,
    MemoryWriteRequest,
    MemoryWriteResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cognition"])


# ── live-store resolution (graceful) ──────────────────────────────────


def _resolve_store(ctx: Any) -> Optional[Any]:
    """Best-effort resolve a live ``MemoryStore`` from the app context.

    Order: an explicit ``ctx.memory_store`` (future wiring) → build one off the
    wired ``ctx.experiment_storage`` (the store shares the experiment DB file).
    Returns ``None`` (→ degraded) when nothing is wired or anything raises. The
    heavy ``mast.memory.store`` import is lazy + guarded so the API still boots
    standalone with no memory subsystem present.
    """
    # 1) explicit memory store (leader may wire this in later, additively).
    store = getattr(ctx, "memory_store", None)
    if store is not None:
        return store

    # 2) derive from the wired experiment storage (they share the DB file).
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
    """Best-effort DB path for the long ops (dream / brainstorm grounding)."""
    for src in (store, getattr(ctx, "experiment_storage", None)):
        if src is None:
            continue
        p = getattr(src, "_db_path", None)
        if p is not None:
            return str(p)
    return None


def _to_entry(row: dict, *, namespace: str) -> MemoryEntry:
    """Coerce a backend memory ``_row`` dict into the typed schema (tolerant)."""
    return MemoryEntry(
        id=row.get("id"),
        namespace=row.get("namespace") or namespace,
        path=row.get("path") or "",
        title=row.get("title") or "",
        content=row.get("content") or "",
        kind=row.get("kind") or "note",
        tags=list(row.get("tags") or []),
        experiment_id=row.get("experiment_id"),
        author=row.get("author") or "",
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        pinned=bool(row.get("pinned")),
    )


# ── memory: namespaces ────────────────────────────────────────────────


@router.get("/memory/namespaces", response_model=MemoryNamespacesResponse)
def memory_namespaces(request: Request) -> MemoryNamespacesResponse:
    """List existing memory namespaces ('global' always first)."""
    store = _resolve_store(request.app.state.ctx)
    if store is None:
        return MemoryNamespacesResponse(namespaces=["global"], degraded=True)
    try:
        ns = store.namespaces()
    except Exception as exc:
        logger.warning("memory namespaces failed: %s", exc)
        return MemoryNamespacesResponse(namespaces=["global"], degraded=True)
    # 'global' first, then the rest, de-duped (mirrors cognition_panel).
    out = ["global"] + [n for n in (ns or []) if n != "global"]
    seen: set[str] = set()
    res: list[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            res.append(n)
    return MemoryNamespacesResponse(namespaces=res, degraded=False)


# ── memory: list / search ─────────────────────────────────────────────


@router.get("/memory/{ns}", response_model=MemoryListResponse)
def memory_list(
    request: Request,
    ns: str,
    kind: Optional[str] = Query(default=None),
    query: Optional[str] = Query(default=None),
) -> MemoryListResponse:
    """List memories in a namespace (pinned first). ``query`` does a substring
    search; ``kind`` filters by memory kind."""
    store = _resolve_store(request.app.state.ctx)
    if store is None:
        return MemoryListResponse(namespace=ns, degraded=True)
    try:
        if query:
            rows = store.search(query, namespace=ns, limit=200)
            if kind:
                rows = [r for r in rows if r.get("kind") == kind]
        else:
            rows = store.list(ns, kind=kind, limit=200)
    except Exception as exc:
        logger.warning("memory list failed: %s", exc)
        return MemoryListResponse(namespace=ns, degraded=True)
    entries = [_to_entry(r, namespace=ns) for r in (rows or [])]
    return MemoryListResponse(
        namespace=ns, entries=entries, count=len(entries), degraded=False
    )


# ── memory: pin ───────────────────────────────────────────────────────
# NOTE: the pin route is registered BEFORE the generic ``{path:path}`` routes
# below so the literal ``/pin`` suffix wins routing (Starlette matches routes in
# registration order; the greedy ``{path:path}`` would otherwise swallow ``pin``).


@router.post("/memory/{ns}/{path:path}/pin", response_model=MemoryPinResult)
def memory_pin(
    request: Request, ns: str, path: str, body: MemoryPinRequest
) -> MemoryPinResult:
    """Pin / unpin a memory by ``(namespace, path)``."""
    store = _resolve_store(request.app.state.ctx)
    if store is None:
        return MemoryPinResult(ok=False, degraded=True, message="memory store not wired")
    try:
        row = store.read(ns, path)
        if not row:
            return MemoryPinResult(ok=False, degraded=False, message="no such memory")
        store.pin(int(row["id"]), bool(body.on))
    except Exception as exc:
        logger.warning("memory pin failed: %s", exc)
        return MemoryPinResult(ok=False, degraded=True, message=f"pin failed: {exc}")
    return MemoryPinResult(
        ok=True, pinned=bool(body.on), degraded=False,
        message="pinned" if body.on else "unpinned",
    )


# ── memory: detail / write / delete (path may contain '/') ────────────


@router.get("/memory/{ns}/{path:path}", response_model=MemoryDetailResponse)
def memory_read(request: Request, ns: str, path: str) -> MemoryDetailResponse:
    """Read one memory by ``(namespace, path)``."""
    store = _resolve_store(request.app.state.ctx)
    if store is None:
        return MemoryDetailResponse(namespace=ns, path=path, degraded=True)
    try:
        row = store.read(ns, path)
    except Exception as exc:
        logger.warning("memory read failed: %s", exc)
        return MemoryDetailResponse(namespace=ns, path=path, degraded=True)
    if not row:
        return MemoryDetailResponse(namespace=ns, path=path, found=False, degraded=False)
    return MemoryDetailResponse(
        namespace=ns, path=path, entry=_to_entry(row, namespace=ns),
        found=True, degraded=False,
    )


@router.post("/memory/{ns}/{path:path}", response_model=MemoryWriteResult)
def memory_write(
    request: Request, ns: str, path: str, body: MemoryWriteRequest
) -> MemoryWriteResult:
    """Upsert a memory at ``(namespace, path)``. The core store sanitises the path
    and coerces the kind — the API adds no business logic."""
    store = _resolve_store(request.app.state.ctx)
    if store is None:
        return MemoryWriteResult(
            ok=False, degraded=True, namespace=ns, path=path,
            message="memory store not wired",
        )
    try:
        res = store.write(
            ns, path, body.content or "", title=body.title or "",
            kind=body.kind or "note", tags=list(body.tags or []),
            author="api", pinned=bool(body.pin),
        )
    except Exception as exc:
        logger.warning("memory write failed: %s", exc)
        return MemoryWriteResult(
            ok=False, degraded=True, namespace=ns, path=path,
            message=f"write failed: {exc}",
        )
    return MemoryWriteResult(
        ok=True, degraded=False,
        namespace=res.get("namespace") or ns,
        path=res.get("path") or path,
        id=res.get("id"),
        message="saved",
    )


@router.delete("/memory/{ns}/{path:path}", response_model=MemoryDeleteResult)
def memory_delete(request: Request, ns: str, path: str) -> MemoryDeleteResult:
    """Delete a memory by ``(namespace, path)``."""
    store = _resolve_store(request.app.state.ctx)
    if store is None:
        return MemoryDeleteResult(ok=False, degraded=True, message="memory store not wired")
    try:
        deleted = bool(store.delete(ns, path))
    except Exception as exc:
        logger.warning("memory delete failed: %s", exc)
        return MemoryDeleteResult(ok=False, degraded=True, message=f"delete failed: {exc}")
    return MemoryDeleteResult(
        ok=True, deleted=deleted, degraded=False,
        message="deleted" if deleted else "no such memory",
    )


# ── cognition: dreaming (long; ack/job for now) ───────────────────────


@router.post("/cognition/dream", response_model=CognitionJobAck)
def cognition_dream(request: Request, body: DreamRequest) -> CognitionJobAck:
    """Trigger a dreaming-consolidation pass. Long op → returns an ack/job.

    Degrade-safe: with no live store/DB this just acks ``degraded=True``. When a
    store IS wired the offline rule-based consolidator runs synchronously (no LLM
    key needed) and the written paths are returned in the ack; SSE progress is
    layered on later at integration.
    """
    ctx = request.app.state.ctx
    store = _resolve_store(ctx)
    db_path = _db_path_from(ctx, store)
    if store is None or db_path is None:
        return CognitionJobAck(
            job="dream", status="degraded", degraded=True,
            message="memory store / experiment DB not wired",
        )
    try:
        from mast.memory.dreaming import DreamingService

        svc = DreamingService(db_path, store, namespace=body.namespace or "global")
        written = svc.dream_once()
    except Exception as exc:
        logger.warning("dream pass failed: %s", exc)
        return CognitionJobAck(
            job="dream", status="degraded", degraded=True,
            message=f"dream failed: {exc}",
        )
    paths = [w.get("path", "?") for w in (written or [])]
    return CognitionJobAck(
        job="dream", status="accepted", degraded=False,
        message=f"consolidated {len(paths)} memory entr(ies)",
        written=paths,
    )


# ── cognition: brainstorm (long; ack/job for now) ─────────────────────


@router.post("/cognition/brainstorm", response_model=CognitionJobAck)
def cognition_brainstorm(request: Request, body: BrainstormRequest) -> CognitionJobAck:
    """Launch a facilitated multi-agent brainstorm. Long op → returns an ack/job.

    Degrade-safe: with no live store/DB this acks ``degraded=True``. Real
    execution (and SSE streaming of the transcript) is wired at integration; for
    now the ack confirms the contract and parameters were accepted.
    """
    ctx = request.app.state.ctx
    store = _resolve_store(ctx)
    db_path = _db_path_from(ctx, store)
    if store is None or db_path is None:
        return CognitionJobAck(
            job="brainstorm", status="degraded", degraded=True,
            message="memory store / experiment DB not wired",
        )
    # Confirm the brainstorm backend is importable; never run the (long) graph
    # inline in the request — that is the integrator's SSE job. Ack acceptance here.
    try:
        from mast.agents.brainstorm.graph import run_brainstorm  # noqa: F401
    except Exception as exc:
        logger.warning("brainstorm backend unavailable: %s", exc)
        return CognitionJobAck(
            job="brainstorm", status="degraded", degraded=True,
            message="brainstorm backend not available",
        )
    return CognitionJobAck(
        job="brainstorm", status="accepted", degraded=False,
        message=(
            f"brainstorm accepted (topic={body.topic!r}, "
            f"rounds={max(1, int(body.max_rounds))}, "
            f"viewpoints={len(body.viewpoints)})"
        ),
    )
