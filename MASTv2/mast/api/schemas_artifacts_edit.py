"""Pydantic schemas for the workspace-artifact *editor* routes (save / revert /
history / diff / export).

These endpoints edit the REAL FILE on disk. The paper agents persist their work
(``paper_writing.save_draft`` → ``data/drafts/<title>_vNNN.md``;
``paper_review.save_review`` → ``data/reviews/…``) and read it back
(``load_draft`` / ``load_review``), so an operator edit saved here is genuinely
seen by the next agent that opens the document.

They used to write into an in-process dict (``live_app._agents_api_state
["artifact_edits"]``) that these schemas described as "the very dict the
orchestrator already reads on its next super-step". **Nothing has ever read it.**
The operator's edit was echoed back to them and then dropped on the floor. See
``routes/artifacts_edit.py`` for the full post-mortem.

``artifact_id`` is the per-FILE id — ``draft:<stem>`` / ``review:<stem>``, as
returned by ``GET /api/artifacts`` and ``GET /api/documents``. The artifact CLASS
("draft") cannot address one file among five, and a save that guesses is worse
than one that refuses.

Every handler degrades to a typed body (``degraded=True``) rather than raising —
an unknown id, a missing directory, a non-editable artifact.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── POST /api/artifacts/{artifact_id}/edit ───────────────────────────────────
class ArtifactEditRequest(BaseModel):
    """The operator's new body for a document. Saved as a NEW VERSION — the
    agent's own version is never overwritten."""

    body: str = Field(
        ...,
        description="Full Markdown body to save as the next version of this "
        "document. The agents read it back via load_draft / load_review.",
    )


class ArtifactEditResponse(BaseModel):
    ok: bool = False
    artifact_id: str = ""      # id of the NEW version (draft:<stem>_vNNN)
    body: Optional[str] = None
    t: Optional[float] = None  # mtime of the version just written
    history_len: int = 0       # versions that existed before this one
    detail: Optional[str] = None
    degraded: bool = True


# ── DELETE /api/artifacts/{artifact_id}/edit ─────────────────────────────────
class ArtifactRevertResponse(BaseModel):
    """Undo = re-commit the previous version as a new one. Never a deletion —
    revert must not be the one button in the app that destroys work."""

    ok: bool = False
    artifact_id: str = ""
    reverted: bool = False     # False when there is no prior version to go back to
    history_len: int = 0
    detail: Optional[str] = None
    degraded: bool = True


# ── GET /api/artifacts/{artifact_id}/history ─────────────────────────────────
class ArtifactHistoryEntry(BaseModel):
    body: str = ""
    t: Optional[float] = None


class ArtifactHistoryResponse(BaseModel):
    """Every saved version of the document, read from DISK (oldest → newest).
    The old in-memory history was empty on every fresh boot."""

    artifact_id: str = ""
    entries: list[ArtifactHistoryEntry] = Field(default_factory=list)
    count: int = 0
    current: Optional[ArtifactHistoryEntry] = None  # the newest version
    detail: Optional[str] = None
    degraded: bool = True


# ── GET /api/artifacts/{artifact_id}/diff ────────────────────────────────────
class ArtifactDiffResponse(BaseModel):
    """Unified diff of the previous version against the current one."""

    artifact_id: str = ""
    has_edit: bool = False      # ≥2 versions exist (something was edited)
    has_original: bool = False  # a prior version exists to diff against
    changed: bool = False
    diff: str = ""
    detail: Optional[str] = None
    degraded: bool = True


# ── GET /api/artifacts/{artifact_id}/export ──────────────────────────────────
class ArtifactExportResponse(BaseModel):
    """JSON fallback for the export (the route streams text/markdown by default;
    this typed body is returned when ``format=json`` or the file is missing)."""

    ok: bool = False
    artifact_id: str = ""
    body: Optional[str] = None
    filename: Optional[str] = None
    source: Optional[str] = None  # "file" | None
    detail: Optional[str] = None
    degraded: bool = True


__all__ = [
    "ArtifactEditRequest",
    "ArtifactEditResponse",
    "ArtifactRevertResponse",
    "ArtifactHistoryEntry",
    "ArtifactHistoryResponse",
    "ArtifactDiffResponse",
    "ArtifactExportResponse",
]
