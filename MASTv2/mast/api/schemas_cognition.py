"""Pydantic request/response models for the cognition domain (Domain F).

Memory CRUD + pin + search, plus the two long-running cognition ops (dreaming
consolidation, multi-agent brainstorm) which — for now — return a typed ack/job
shape (SSE streaming is layered on at integration later).

These mirror the real backend data shapes:
  * memory rows: ``mast.memory.store.MemoryStore`` (``_row`` dict — id, namespace,
    path, title, content, kind, tags, experiment_id, author, created_at,
    updated_at, pinned);
  * dreaming: ``mast.memory.dreaming.DreamingService.dream_once`` → list of
    ``{path, title}``;
  * brainstorm: ``mast.agents.brainstorm.graph.run_brainstorm`` →
    ``{transcript, summary}``.

House style: every response has a ``degraded`` boolean so the API can boot
STANDALONE (no live memory store wired) and return empty-but-not-broken shapes
instead of 500-ing. Write endpoints degrade to ``ok=False, degraded=True``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# Mirror the backend's allowed memory kinds (single source = the store). Imported
# lazily-tolerant: if the store module is unavailable at import time we still want
# the schema module to load, so we fall back to the known tuple.
try:  # pragma: no cover - import guard
    from mast.memory.store import KINDS as _STORE_KINDS

    MEMORY_KINDS: tuple[str, ...] = tuple(_STORE_KINDS)
except Exception:  # pragma: no cover - defensive: schema must import standalone
    MEMORY_KINDS = (
        "note", "insight", "summary", "hypothesis", "protocol", "dream",
        "brainstorm",
    )


# ── memory rows ───────────────────────────────────────────────────────


class MemoryEntry(BaseModel):
    """One memory row (mirrors ``MemoryStore._row``). All fields optional past the
    address pair so a thin/degraded row still validates."""

    id: Optional[int] = None
    namespace: str
    path: str
    title: str = ""
    content: str = ""
    kind: str = "note"
    tags: list[str] = Field(default_factory=list)
    experiment_id: Optional[str] = None
    author: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pinned: bool = False


class MemoryNamespacesResponse(BaseModel):
    """Existing namespaces ('global' always first). ``degraded`` when no store."""

    namespaces: list[str] = Field(default_factory=lambda: ["global"])
    degraded: bool = False


class MemoryListResponse(BaseModel):
    """Memories in a namespace (pinned first), optionally filtered by kind /
    substring query. Empty-but-not-broken when the store is absent."""

    namespace: str = "global"
    entries: list[MemoryEntry] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False


class MemoryDetailResponse(BaseModel):
    """A single memory by ``(namespace, path)``. ``found`` is False when the row
    does not exist; ``degraded`` when the store itself is absent."""

    namespace: str = "global"
    path: str = ""
    entry: Optional[MemoryEntry] = None
    found: bool = False
    degraded: bool = False


# ── write payloads ────────────────────────────────────────────────────


class MemoryWriteRequest(BaseModel):
    """Upsert body for ``POST /api/memory/{ns}/{path}`` (path comes from the URL).

    No business logic / safety here — the API only forwards to the core store,
    which sanitises the path and coerces the kind."""

    content: str = ""
    title: str = ""
    kind: str = "note"
    tags: list[str] = Field(default_factory=list)
    pin: bool = False


class MemoryPinRequest(BaseModel):
    """Body for ``POST /api/memory/{ns}/{path}/pin`` — pin on/off (default on)."""

    on: bool = True


class MemoryWriteResult(BaseModel):
    """Result of a memory upsert. ``ok=False, degraded=True`` when no live store
    is wired (write deferred to integration time)."""

    ok: bool = False
    degraded: bool = False
    namespace: Optional[str] = None
    path: Optional[str] = None
    id: Optional[int] = None
    message: str = ""


class MemoryDeleteResult(BaseModel):
    """Result of a memory delete. ``deleted`` is True only when a row was removed."""

    ok: bool = False
    deleted: bool = False
    degraded: bool = False
    message: str = ""


class MemoryPinResult(BaseModel):
    """Result of a pin toggle."""

    ok: bool = False
    pinned: bool = False
    degraded: bool = False
    message: str = ""


# ── long-running cognition ops (ack/job for now) ──────────────────────


class DreamRequest(BaseModel):
    """Trigger a dreaming-consolidation pass. ``namespace`` is where the dream
    entries land (defaults to the backend's 'global')."""

    namespace: str = "global"


class CognitionJobAck(BaseModel):
    """Ack for a long op. ``status`` is 'accepted' when a job is (or would be)
    started, 'degraded' when the live subsystem is absent. Concrete results
    (written paths / transcript) stream over SSE later; for now an ack-shaped
    response keeps the contract stable. ``written`` carries any synchronous
    degrade-safe results when available."""

    job: str
    status: str = "accepted"
    degraded: bool = False
    message: str = ""
    written: list[str] = Field(default_factory=list)


class BrainstormRequest(BaseModel):
    """Launch a facilitated multi-agent brainstorm (mirrors ``run_brainstorm``)."""

    experiment_id: Optional[str] = None
    topic: str = ""
    viewpoints: list[str] = Field(default_factory=list)
    max_rounds: int = 2


__all__ = [
    "MEMORY_KINDS",
    "MemoryEntry",
    "MemoryNamespacesResponse",
    "MemoryListResponse",
    "MemoryDetailResponse",
    "MemoryWriteRequest",
    "MemoryPinRequest",
    "MemoryWriteResult",
    "MemoryDeleteResult",
    "MemoryPinResult",
    "DreamRequest",
    "CognitionJobAck",
    "BrainstormRequest",
]
