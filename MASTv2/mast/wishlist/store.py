"""Thread-safe JSON-backed wishlist / feedback board.

Two record kinds in one file (``artifacts/wishlist/wishlist.json``):

* **wishes**  — user-submitted wishes / feedback. status: queued → sent | failed
  (sent = relayed to the update server's /feedback endpoint).
* **requests** — agent→user requests (e.g. literature agent: "请上传全文 X";
  instrument agent: "请去操作硬件 …"). status: pending → done | dismissed.

Mirrors :mod:`mast.knowledge.fetch_board` (atomic temp+replace, RLock, bounded,
degrades gracefully on a corrupt file, stdlib-only). Lives at module top so the
GUI 心愿单 tab AND agents (via agents/_shared/request_tools) share one board.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WISH_STATUSES = ("queued", "sent", "failed")
REQUEST_STATUSES = ("pending", "done", "dismissed")
_REQ_OPEN = "pending"
MAX_ITEMS = 1000   # bound each list so a runaway client/agent can't grow it forever


def _default_dir() -> Path:
    """Where the board lives — resolved LAZILY, through the sanctioned resolver.

    This used to be a module-level constant built from a private ``parents[3]``
    walk off ``__file__``:

        _DEFAULT_DIR = Path(__file__).resolve().parents[3] / "MASTv2" / "artifacts" / "wishlist"

    Two problems, one of which bit immediately:

    1. **It ignored ``MAST2_PROJECT_ROOT``.** ``project_root()`` honours that env
       override; a private walk does not. So every test that thought it had
       redirected the board with ``monkeypatch.setenv("MAST2_PROJECT_ROOT", …)``
       had in fact redirected nothing, and wrote into the operator's REAL
       wishlist. It accumulated 25 junk "请提供 Au111 那张图的完整路径" rows there
       before anyone looked. (Same failure the composite-skill store had — see
       tests/v2/conftest.py.)

    2. **It was import-time**, so even a correct override applied after import
       could not take effect. Resolution now happens per call.

    A private repo-root walk is also how #100 happened (``load_draft: drafts
    directory not found at C:\\MAST\\data\\drafts`` in the frozen build). Dev and
    frozen agree today only by coincidence; ``project_root()`` makes them agree by
    construction — env override → frozen exe dir → dev repo root.
    """
    from mast._runtime_paths import project_root

    return project_root() / "MASTv2" / "artifacts" / "wishlist"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(t: Any, limit: int = 4000) -> str:
    s = str(t or "").strip()
    if len(s) > limit:
        s = s[:limit]
    # strip control chars (defence in depth)
    return "".join(c for c in s if ord(c) >= 32 or c in "\n\t")


class WishlistBoard:
    """JSON-backed board: user wishes + agent→user requests."""

    def __init__(self, board_dir: str | os.PathLike[str] | None = None) -> None:
        self._dir = Path(board_dir) if board_dir is not None else _default_dir()
        self._path = self._dir / "wishlist.json"
        self._lock = threading.RLock()
        self._wishes: list[dict] = []
        self._requests: list[dict] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────
    def _load(self) -> None:
        with self._lock:
            self._wishes, self._requests = [], []
            try:
                if self._path.exists():
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                    self._wishes = list(raw.get("wishes") or [])
                    self._requests = list(raw.get("requests") or [])
            except Exception as exc:  # corrupt → start empty, never crash
                logger.warning("wishlist load failed (%s); starting empty", exc)
                self._wishes, self._requests = [], []

    def _save(self) -> None:
        with self._lock:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".json.tmp")
                payload = {"wishes": self._wishes, "requests": self._requests}
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, self._path)
            except Exception as exc:  # best-effort
                logger.warning("wishlist save failed: %s", exc)

    @staticmethod
    def _next_id(items: list[dict], prefix: str) -> str:
        n = 0
        for it in items:
            rid = it.get("id", "")
            if isinstance(rid, str) and rid.startswith(prefix):
                try:
                    n = max(n, int(rid[len(prefix):]))
                except ValueError:
                    pass
        return f"{prefix}{n + 1}"

    # ── user wishes ──────────────────────────────────────────────────
    def add_wish(self, text: str, *, category: str = "feature",
                 client_version: str = "") -> dict:
        text = _clean_text(text)
        if not text:
            return {"error": "empty wish"}
        with self._lock:
            if len(self._wishes) >= MAX_ITEMS:
                self._wishes = self._wishes[-(MAX_ITEMS - 1):]
            rec = {
                "id": self._next_id(self._wishes, "w-"),
                "text": text, "category": (category or "feature").strip(),
                "client_version": client_version, "status": "queued",
                "created_at": _now_iso(), "server_ack": None, "error": "",
            }
            self._wishes.append(rec)
            self._save()
            return dict(rec)

    def list_wishes(self) -> list[dict]:
        with self._lock:
            return [dict(w) for w in reversed(self._wishes)]  # newest first

    def mark_wish_sent(self, wish_id: str, *, ok: bool,
                       server_ack: str = "", error: str = "") -> dict:
        with self._lock:
            for rec in self._wishes:
                if rec.get("id") == wish_id:
                    rec["status"] = "sent" if ok else "failed"
                    rec["server_ack"] = server_ack or None
                    rec["error"] = "" if ok else _clean_text(error, 500)
                    self._save()
                    return dict(rec)
            return {"error": "no such wish"}

    # ── agent → user requests ────────────────────────────────────────
    def post_agent_request(self, agent_id: str, message: str, *,
                           kind: str = "action", experiment_id: str | None = None,
                           origin_conversation_id: str = "") -> dict:
        """Post (or return the existing OPEN) request.

        ``origin_conversation_id`` records **which conversation asked**, so that
        answering it can wake that conversation back up instead of leaving the
        answer to be noticed on some later turn that, for an agent that stopped,
        never comes. Empty when the caller has no conversation, which degrades to
        the old behaviour: the answer waits for the next turn to read it back.
        """
        message = _clean_text(message)
        if not message:
            return {"error": "empty request"}
        with self._lock:
            # dedup: identical open request from the same agent → return existing
            for rec in self._requests:
                if (rec.get("agent_id") == agent_id and rec.get("message") == message
                        and rec.get("status") == _REQ_OPEN):
                    return dict(rec)
            if len(self._requests) >= MAX_ITEMS:
                self._requests = self._requests[-(MAX_ITEMS - 1):]
            rec = {
                "id": self._next_id(self._requests, "r-"),
                "agent_id": (agent_id or "agent").strip() or "agent",
                "message": message, "kind": (kind or "action").strip(),
                "experiment_id": experiment_id, "status": _REQ_OPEN,
                "origin_conversation_id": (origin_conversation_id or "").strip(),
                "created_at": _now_iso(), "resolved_at": None, "note": "",
            }
            self._requests.append(rec)
            self._save()
            return dict(rec)

    def list_agent_requests(self, status: str | None = None) -> list[dict]:
        with self._lock:
            rows = [dict(r) for r in self._requests
                    if status is None or r.get("status") == status]
        # open first, then most-recent
        rows.sort(key=lambda r: (r.get("status") != _REQ_OPEN,
                                 r.get("created_at") or ""), reverse=False)
        return rows

    def resolve_agent_request(self, request_id: str, status: str, *,
                              note: str = "", path: str = "") -> dict:
        """Resolve one agent request.

        ``path`` is a FIRST-CLASS field, not prose. The agent's most
        common ask is "where is the file?"; the answer used to be typed into a free
        note that the agent could not parse and — unless a run happened to be live
        at that moment — never received at all. It is persisted here so
        ``check_my_requests`` can hand it back later, whenever the agent next runs.
        """
        if status not in REQUEST_STATUSES:
            return {"error": f"invalid status {status!r}"}
        answered = None
        with self._lock:
            for rec in self._requests:
                if rec.get("id") == request_id:
                    rec["status"] = status
                    rec["note"] = _clean_text(note, 500)
                    rec["path"] = _clean_text(path, 400)
                    rec["resolved_at"] = _now_iso()
                    rec["delivered"] = False   # not yet read back by the agent
                    self._save()
                    answered = dict(rec)
                    break
        if answered is None:
            return {"error": "no such request"}

        # Wake the conversation that asked, if it named itself. Without this the
        # answer waits for a turn that, for an agent which stopped when it hit
        # the blocker, never arrives — the readback middleware can only inject
        # into a turn that is already happening.
        if answered.get("origin_conversation_id"):
            try:
                from mast.core import fetch_resume
                res = fetch_resume.notify_request_answered([answered]) or {}
                if int(res.get("resumed", 0) or 0) > 0:
                    # Told directly; keep the middleware from repeating it.
                    self.mark_delivered([request_id])
                    answered["delivered"] = True
            except Exception as exc:  # noqa: BLE001 — never fail a resolve over this
                logger.info("request-answer resume for %s failed: %s", request_id, exc)
        return answered

    def resolved_requests_for(self, agent_id: str = "",
                             undelivered_only: bool = True) -> list[dict]:
        """Requests the operator has ANSWERED, that the agent has not read yet.

        This is the half of #97 that was missing entirely: there was no agent tool
        that could READ the wishlist. An answer only reached an agent if a
        supervisor run happened to be streaming at the moment the operator clicked
        resolve — otherwise it sat on the board, answered, forever unseen."""
        with self._lock:
            out = []
            for r in self._requests:
                if r.get("status") not in ("done", "dismissed"):
                    continue
                if agent_id and str(r.get("agent_id") or "") != agent_id:
                    continue
                if undelivered_only and r.get("delivered"):
                    continue
                out.append(dict(r))
            return out

    def mark_delivered(self, request_ids: list[str]) -> int:
        """Mark answers as read back, so an agent isn't handed the same one twice."""
        ids = set(request_ids or ())
        n = 0
        with self._lock:
            for r in self._requests:
                if r.get("id") in ids and not r.get("delivered"):
                    r["delivered"] = True
                    n += 1
            if n:
                self._save()
        return n

    def pending_request_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._requests if r.get("status") == _REQ_OPEN)

    def get_request(self, request_id: str) -> dict | None:
        """One request by id, or None. A NON-destructive point read (does not
        touch the ``delivered`` flag), so an agent can POLL a specific request it
        posted — "has r-4 been answered yet?" — as many times as it likes without
        consuming the answer the way ``check_my_requests`` does (feedback ⑦)."""
        rid = (request_id or "").strip()
        if not rid:
            return None
        with self._lock:
            for r in self._requests:
                if r.get("id") == rid:
                    return dict(r)
        return None


# ── module-level singleton + free funcs (mirror fetch_board) ──────────
_BOARD: WishlistBoard | None = None
_BOARD_LOCK = threading.Lock()


def get_board() -> WishlistBoard:
    global _BOARD
    with _BOARD_LOCK:
        if _BOARD is None:
            _BOARD = WishlistBoard()
        return _BOARD


def reset_default_board() -> None:
    """Test hook."""
    global _BOARD
    with _BOARD_LOCK:
        _BOARD = None


def _b(board: WishlistBoard | None) -> WishlistBoard:
    return board if board is not None else get_board()


def add_wish(text: str, *, board: WishlistBoard | None = None, **kw) -> dict:
    return _b(board).add_wish(text, **kw)


def list_wishes(*, board: WishlistBoard | None = None) -> list[dict]:
    return _b(board).list_wishes()


def mark_wish_sent(wish_id: str, *, board: WishlistBoard | None = None, **kw) -> dict:
    return _b(board).mark_wish_sent(wish_id, **kw)


def post_agent_request(agent_id: str, message: str, *,
                       board: WishlistBoard | None = None, **kw) -> dict:
    return _b(board).post_agent_request(agent_id, message, **kw)


def list_agent_requests(status: str | None = None, *,
                        board: WishlistBoard | None = None) -> list[dict]:
    return _b(board).list_agent_requests(status)


def resolve_agent_request(request_id: str, status: str, *,
                          board: WishlistBoard | None = None, **kw) -> dict:
    return _b(board).resolve_agent_request(request_id, status, **kw)


def pending_request_count(*, board: WishlistBoard | None = None) -> int:
    return _b(board).pending_request_count()


def get_request(request_id: str, *, board: WishlistBoard | None = None) -> dict | None:
    return _b(board).get_request(request_id)


__all__ = [
    "WishlistBoard", "get_board", "reset_default_board",
    "add_wish", "list_wishes", "mark_wish_sent",
    "post_agent_request", "list_agent_requests", "resolve_agent_request",
    "get_request", "pending_request_count", "WISH_STATUSES", "REQUEST_STATUSES",
]
