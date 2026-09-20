"""Pydantic request/response models for the Wishlist + agent-request seam.

Domain H of the typed FastAPI seam (TS-rewrite Phase 3). Mirrors the record
shapes produced by :mod:`mast.wishlist.store` (``WishlistBoard``) — the same
JSON board the Gradio 心愿单 tab and the agents (agent→user requests) share.

Read-only shapes (GET /api/wishlist) plus the two write contracts (submit a
wish, resolve an agent request). Every response carries a ``degraded`` flag so
the frontend renders empty-but-not-broken when the live board is unwired.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Mirror the board's status vocabularies (store.WISH_STATUSES / REQUEST_STATUSES)
# as Literals so the TS typegen gets a closed enum. The board itself stays the
# single authority — the API never invents statuses.
WishStatus = Literal["queued", "sent", "failed"]
RequestStatus = Literal["pending", "done", "dismissed"]
# Only the two terminal actions are a valid *resolution* target from the UI.
ResolveAction = Literal["done", "dismissed"]


class Wish(BaseModel):
    """One user-submitted wish / feedback row (mirror of board ``add_wish``)."""

    id: str
    text: str = ""
    category: str = "feature"
    client_version: str = ""
    status: WishStatus = "queued"
    created_at: Optional[str] = None
    server_ack: Optional[str] = None
    error: Optional[str] = None


class AgentRequest(BaseModel):
    """One agent→user request row (e.g. literature: "请上传全文")."""

    id: str
    agent_id: str = "agent"
    message: str = ""
    kind: str = "action"
    experiment_id: Optional[str] = None
    status: RequestStatus = "pending"
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None
    note: Optional[str] = None
    # The operator's answer to "where is the file?" — a field, not prose .
    # Surfaced back on the row so the operator can see exactly what the agent will
    # read via check_my_requests.
    path: Optional[str] = None
    delivered: bool = False   # the agent has read this answer back


class WishlistResponse(BaseModel):
    """The whole 心愿单 board: user wishes (newest first) + agent requests
    (open first). ``degraded`` True when the live board is unwired/unreadable —
    the frontend shows an empty-but-not-broken state, never an error."""

    wishes: list[Wish] = Field(default_factory=list)
    requests: list[AgentRequest] = Field(default_factory=list)
    pending_count: int = 0
    degraded: bool = False


class WishCreate(BaseModel):
    """Submit a wish. ``category`` defaults to the GUI's "功能建议"."""

    text: str = Field(description="the wish / feedback body (trimmed + bounded by the board)")
    category: str = "功能建议"
    client_version: str = Field(
        default="",
        description="optional client version tag; the live core fills it from mast.__version__",
    )


class WishCreateResponse(BaseModel):
    """Ack for a submitted wish. The local record is created synchronously
    (status ``queued``); the relay to the update server's /feedback endpoint is
    fired asynchronously (non-blocking), so the returned ``wish`` is the freshly
    queued record. ``ok`` False + ``degraded`` True when the board is unwired."""

    ok: bool = False
    wish: Optional[Wish] = None
    message: str = ""
    degraded: bool = False


class RequestResolve(BaseModel):
    """Resolve an agent request: an action, an optional note, and — the point of
    the exercise — an optional FILE PATH.

    Feedback 2026-07-10 #97: "心愿单备注中提供路径 agent 读不到；心愿单中又没有
    显式由用户提供路径的地方." The agent's single most common ask is *"where is the
    file?"*, and the only place to answer was a free-text note the agent could not
    reliably parse and — outside a live run — never even received. A path typed
    into prose is not a channel. Give it a field."""

    action: ResolveAction = Field(description="done → 已完成; dismissed → 已忽略")
    note: str = ""
    path: str = Field(
        "",
        description="用户提供的文件/目录路径（例如智能体索要的 .sxm）。"
        "作为独立字段传递，智能体可直接读取，无需从备注文本里猜。",
    )


class RequestResolveResponse(BaseModel):
    """Result of resolving an agent request. ``ok`` False + ``degraded`` True
    when the board is unwired; ``ok`` False without ``degraded`` when the id was
    not found / the action was rejected by the core."""

    ok: bool = False
    request: Optional[AgentRequest] = None
    message: str = ""
    degraded: bool = False


__all__ = [
    "WishStatus",
    "RequestStatus",
    "ResolveAction",
    "Wish",
    "AgentRequest",
    "WishlistResponse",
    "WishCreate",
    "WishCreateResponse",
    "RequestResolve",
    "RequestResolveResponse",
]
