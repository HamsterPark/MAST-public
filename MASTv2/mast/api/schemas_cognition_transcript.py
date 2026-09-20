"""Pydantic models for the FULL-transcript cognition endpoints (additive seam).

The existing ``routes/cognition.py`` brainstorm/dream endpoints return an
ack/job shape only — they never surface the transcript that ``run_brainstorm``
already computes in one shot. These schemas back the additive ``*/full`` (and a
no-op SSE-shaped) endpoints that DO return the complete transcript.

Backend data shapes mirrored here (single source = the core):
  * brainstorm: ``mast.agents.brainstorm.graph.run_brainstorm`` →
    ``{"transcript": [{speaker, role, content, round}, ...], "summary": str}``.
    The ``role`` of a viewpoint turn IS the viewpoint key (design/safety/...);
    facilitator/user turns carry role="facilitator"/"user". We expose ``role``
    verbatim and ALSO surface it as ``viewpoint`` for viewpoint turns.
  * dream: ``mast.memory.dreaming.DreamingService.dream_once`` writes entries and
    returns ``[{path, title}]``; the full content is read back from the store.

House style: every response carries ``degraded`` so the API boots STANDALONE
(no live memory store / DB wired) and returns an empty-but-not-broken body
instead of 500-ing.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── brainstorm: full transcript ───────────────────────────────────────


class BrainstormFullRequest(BaseModel):
    """Run a facilitated multi-agent brainstorm and return the FULL transcript
    (mirrors ``run_brainstorm`` parameters)."""

    topic: str = ""
    viewpoints: list[str] = Field(default_factory=list)
    experiment_id: Optional[str] = None
    max_rounds: int = 2


class BrainstormTurn(BaseModel):
    """One line of the brainstorm transcript.

    ``speaker`` is the display name (主持人 / 用户 / 视角名). ``role`` is the raw
    role/viewpoint key from the core (``facilitator`` / ``user`` / a viewpoint
    key like ``design``). ``viewpoint`` repeats the viewpoint key for viewpoint
    turns and is ``None`` for facilitator/user turns (frontend convenience)."""

    round: int = 0
    speaker: str = ""
    role: str = ""
    viewpoint: Optional[str] = None
    content: str = ""


class BrainstormFullResponse(BaseModel):
    """The whole brainstorm: synthesised ``summary`` + every transcript turn.

    ``degraded=True`` (empty transcript/summary) when the memory store /
    experiment DB is not wired — never a 500. ``written_memory_ids`` lists the
    memory paths the summary was persisted to (best-effort; empty when no store
    write happened)."""

    summary: str = ""
    transcript: list[BrainstormTurn] = Field(default_factory=list)
    written_memory_ids: list[str] = Field(default_factory=list)
    degraded: bool = False
    message: str = ""


# ── dream: full transcript (per-entry content) ────────────────────────


class DreamEntry(BaseModel):
    """One consolidated dream memory entry (read back after the pass wrote it)."""

    path: str = ""
    title: str = ""
    kind: str = "dream"
    content: str = ""


class DreamFullResponse(BaseModel):
    """The dream pass result with each written entry's FULL content.

    ``degraded=True`` when the store / DB is absent. The offline rule-based
    consolidator runs with no LLM key, so a wired store yields a real (possibly
    empty, if there are no experiments) transcript without ever 500-ing."""

    entries: list[DreamEntry] = Field(default_factory=list)
    count: int = 0
    degraded: bool = False
    message: str = ""


__all__ = [
    "BrainstormFullRequest",
    "BrainstormTurn",
    "BrainstormFullResponse",
    "DreamEntry",
    "DreamFullResponse",
]
