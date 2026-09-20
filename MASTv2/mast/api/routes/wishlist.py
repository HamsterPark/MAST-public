"""Wishlist + agent-request endpoints (Domain H of the typed FastAPI seam).

Mirrors the Gradio 心愿单 tab over the shared :mod:`mast.wishlist` board:

* ``GET  /api/wishlist``                       — wishes[] + requests[] (read)
* ``POST /api/wishlist/wishes``                — submit a wish (record locally,
  async-relay to the update server, ack immediately)
* ``POST /api/wishlist/requests/{id}/resolve`` — resolve an agent→user request
  (action = done | dismissed, + note)

GRACEFUL DEGRADATION: the API must boot standalone with no live core wired.
The board is fetched through ``ctx`` first (Phase-3 wiring); failing that, the
heavy core module is LAZY-imported INSIDE each handler, wrapped in try/except —
any absence/raise returns a valid degraded response (``degraded=True``), never a
500. No business logic or safety lives here: handlers only call into the core.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_wishlist import (
    AgentRequest,
    RequestResolve,
    RequestResolveResponse,
    Wish,
    WishCreate,
    WishCreateResponse,
    WishlistResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wishlist"])


# ── board access (ctx wiring → lazy core import → None) ────────────────────
def _get_board(ctx: Any) -> Any:
    """Return a live WishlistBoard or ``None`` (degraded).

    Prefers a board wired onto the context (Phase 3+); otherwise LAZY-imports
    the shared module-level board. Any import/init failure → ``None`` so the
    caller degrades instead of 500-ing.
    """
    board = getattr(ctx, "wishlist_board", None)
    if board is not None:
        return board
    try:
        from mast.wishlist import get_board  # heavy-ish core, imported lazily

        return get_board()
    except Exception as exc:  # absent / unreadable → degrade, never crash
        logger.warning("wishlist board unavailable: %s", exc)
        return None


def _to_wish(raw: dict) -> Wish:
    return Wish(
        id=str(raw.get("id", "")),
        text=str(raw.get("text", "")),
        category=str(raw.get("category", "feature")),
        client_version=str(raw.get("client_version", "")),
        status=raw.get("status", "queued"),  # type: ignore[arg-type]
        created_at=raw.get("created_at"),
        server_ack=raw.get("server_ack"),
        error=raw.get("error"),
    )


def _to_request(raw: dict) -> AgentRequest:
    return AgentRequest(
        id=str(raw.get("id", "")),
        agent_id=str(raw.get("agent_id", "agent")),
        message=str(raw.get("message", "")),
        kind=str(raw.get("kind", "action")),
        experiment_id=raw.get("experiment_id"),
        status=raw.get("status", "pending"),  # type: ignore[arg-type]
        created_at=raw.get("created_at"),
        resolved_at=raw.get("resolved_at"),
        note=raw.get("note"),
        path=raw.get("path"),
        delivered=bool(raw.get("delivered")),
    )


# ── GET /api/wishlist ──────────────────────────────────────────────────────
@router.get("/wishlist", response_model=WishlistResponse)
def get_wishlist(request: Request) -> WishlistResponse:
    board = _get_board(request.app.state.ctx)
    if board is None:
        return WishlistResponse(degraded=True)
    try:
        wishes = [_to_wish(w) for w in (board.list_wishes() or [])]
        requests = [_to_request(r) for r in (board.list_agent_requests() or [])]
        # Upgrade ideas ride the same table but ask nothing of the operator —
        # ``report_upgrade_idea`` records "we should build X one day". Counting
        # them as pending work put "please go change the sample" and "here is an
        # idea for later" behind one number, and the nav badge that reads this
        # count would nag about a to-do list nobody has to do.
        pending = sum(1 for r in requests
                      if r.status == "pending" and r.kind != "upgrade")
        return WishlistResponse(
            wishes=wishes,
            requests=requests,
            pending_count=pending,
            degraded=False,
        )
    except Exception as exc:  # any board call raises → degrade
        logger.warning("wishlist read failed: %s", exc)
        return WishlistResponse(degraded=True)


# ── POST /api/wishlist/wishes ──────────────────────────────────────────────
def _relay_wish_async(board: Any, rec: dict, payload: WishCreate) -> None:
    """Fire-and-forget relay of a queued wish to the update server.

    Mirrors the GUI ``_wl_do_submit`` offload: the /feedback POST is a slow
    httpx round-trip, so it runs OFF the request path; on completion the board
    record is flipped to sent | failed via ``mark_wish_sent``. All best-effort —
    the user already has their ack. The update client is LAZY-imported here.
    """

    def _work() -> None:
        status, detail = "no-config", ""
        try:
            from mast.update.client import (  # lazy: heavy + optional
                _read_token,
                post_feedback,
                read_server_url,
            )

            try:
                from mast._runtime_paths import project_root

                root = project_root()
            except Exception:
                root = None
            url = read_server_url(root)
            tok = _read_token(root)
            status, detail = post_feedback(
                url,
                tok,
                text=payload.text,
                category=payload.category or "功能建议",
                client_version=payload.client_version or rec.get("client_version", ""),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort relay
            status, detail = "error", str(exc)
        ok = status == "ok"
        if not ok:
            # Was fully silent before (2026-07-06 test): a cert / token / offline
            # relay failure left the user believing their wish reached the admin.
            # Log it so failing relays are diagnosable; the board row is marked
            # failed below so the UI can surface a 未送达/重试 state.
            logger.warning(
                "wishlist relay to update server FAILED (%s): %s — wish kept "
                "locally, board marked failed", status, str(detail)[:200])
        try:
            board.mark_wish_sent(
                rec.get("id", ""),
                ok=ok,
                server_ack=detail if ok else "",
                error="" if ok else f"{status}: {detail}",
            )
        except Exception as exc:  # board flip best-effort
            logger.warning("mark_wish_sent failed: %s", exc)

    threading.Thread(target=_work, name="wl-relay", daemon=True).start()


@router.post("/wishlist/wishes", response_model=WishCreateResponse)
def submit_wish(payload: WishCreate, request: Request) -> WishCreateResponse:
    text = (payload.text or "").strip()
    if not text:
        return WishCreateResponse(ok=False, message="请先填写内容")

    board = _get_board(request.app.state.ctx)
    if board is None:
        return WishCreateResponse(ok=False, message="心愿单暂不可用", degraded=True)

    try:
        rec = board.add_wish(
            text,
            category=payload.category or "功能建议",
            client_version=payload.client_version or "",
        )
    except Exception as exc:
        logger.warning("add_wish failed: %s", exc)
        return WishCreateResponse(ok=False, message="提交失败", degraded=True)

    if not isinstance(rec, dict) or rec.get("error"):
        msg = rec.get("error") if isinstance(rec, dict) else "提交失败"
        return WishCreateResponse(ok=False, message=str(msg or "提交失败"))

    # Recorded locally (queued). Relay to the update server runs OFF the request
    # path so a slow/offline network never pins the handler.
    try:
        _relay_wish_async(board, rec, payload)
    except Exception as exc:  # spawning the thread must never break the ack
        logger.warning("wish relay spawn failed: %s", exc)

    return WishCreateResponse(
        ok=True,
        wish=_to_wish(rec),
        message="已提交，正在上报到更新服务器…",
        degraded=False,
    )


# ── POST /api/wishlist/requests/{id}/resolve ───────────────────────────────
@router.post(
    "/wishlist/requests/{request_id}/resolve",
    response_model=RequestResolveResponse,
)
def resolve_request(
    request_id: str, payload: RequestResolve, request: Request
) -> RequestResolveResponse:
    board = _get_board(request.app.state.ctx)
    if board is None:
        return RequestResolveResponse(ok=False, message="心愿单暂不可用", degraded=True)

    try:
        rec = board.resolve_agent_request(
            request_id, payload.action,
            note=payload.note or "", path=payload.path or "",
        )
    except Exception as exc:
        logger.warning("resolve_agent_request failed: %s", exc)
        return RequestResolveResponse(ok=False, message="处理失败", degraded=True)

    if not isinstance(rec, dict) or rec.get("error"):
        msg = rec.get("error") if isinstance(rec, dict) else "处理失败"
        return RequestResolveResponse(ok=False, message=str(msg or "处理失败"))

    # Deliver the operator's answer BACK to the asking agent.
    #
    # TWO paths now, because the live one is not enough : the interjection
    # relay below only fires if a run happens to be STREAMING at the moment the
    # operator clicks resolve. Resolve it after the run ends — the normal case,
    # since the agent asked precisely because it was blocked and stopped — and the
    # answer sat on the board forever, answered and unseen. The record is now also
    # persisted with a first-class ``path`` and an undelivered flag, so the agent's
    # ``check_my_requests`` tool can read it back on its NEXT turn, whenever that
    # is. The relay is the fast path; the tool is the one that always works.
    delivery = ""
    answer_text = "；".join(
        t for t in ((payload.path or "").strip(), (payload.note or "").strip()) if t
    )
    note_text = answer_text
    if note_text:
        delivered = False
        try:
            app = getattr(request.app.state.ctx, "live_app", None) or getattr(
                request.app.state.ctx, "app", None)
            st = getattr(app, "_agents_api_state", None)
            task = st.get("task") if isinstance(st, dict) else None
            if isinstance(st, dict) and isinstance(task, dict) and task.get("active"):
                import contextlib as _ctxlib
                import time as _t
                lock = st.get("lock")
                with (lock or _ctxlib.nullcontext()):
                    q = st.get("interjects")
                    if isinstance(q, list):
                        q.append({
                            "id": f"wish_{request_id}",
                            "agent_id": str(rec.get("agent_id") or "_supervisor"),
                            "text": (f"[心愿单答复 {request_id}] "
                                     f"{payload.action}: {note_text}"),
                            "t": _t.time(),
                        })
                        delivered = True
        except Exception as exc:  # noqa: BLE001 — delivery is best-effort
            logger.debug("wishlist note delivery failed: %s", exc)
        delivery = ("；答复已实时转发给运行中的智能体" if delivered else
                    "；当前无运行中的任务——答复已保存，智能体下次运行时会通过"
                    "check_my_requests 读到（不必再手动复述）")

    return RequestResolveResponse(
        ok=True,
        request=_to_request(rec),
        message=f"已将 {request_id} 标记为 {payload.action}{delivery}。",
        degraded=False,
    )
