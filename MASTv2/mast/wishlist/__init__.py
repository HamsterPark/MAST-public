"""Wishlist / feedback board — user wishes (→ update server) + agent→user requests.

A top-level module (NOT under gui/) so BOTH the GUI 心愿单 tab and the agents can
use it without crossing the agent boundary. See ``store.py``.
"""
from mast.wishlist.store import (  # noqa: F401
    WishlistBoard,
    add_wish,
    get_board,
    get_request,
    list_agent_requests,
    list_wishes,
    mark_wish_sent,
    post_agent_request,
    pending_request_count,
    reset_default_board,
    resolve_agent_request,
)

__all__ = [
    "WishlistBoard", "get_board", "reset_default_board",
    "add_wish", "list_wishes", "mark_wish_sent",
    "post_agent_request", "list_agent_requests", "resolve_agent_request",
    "get_request", "pending_request_count",
]
