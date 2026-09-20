"""Agent-asks-user full-text fetch-request board (literature design §8 / G5).

The HITL piece of literature fetching: the literature agent can POST a request
"please get the full text of work_id X" (when its semantic search surfaces a
paper whose full text it needs but only the abstract is in the big library).
The request surfaces in the GUI 文献库 tab; the operator fulfils it by uploading
a PDF or triggering the experimental DOI/URL fetch, which promotes the paper
into the big library and resolves the request — and the agent can poll the board
to learn the outcome.

Persistence mirrors :mod:`mast.knowledge.libraries`: a JSON file at
``artifacts/literature_libs/fetch_requests.json`` (atomic temp+replace, RLock).
Zero non-stdlib deps; degrades gracefully on a corrupt/unreadable file.
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

STATUSES = ("pending", "fulfilled", "failed", "dismissed")
_OPEN = "pending"
MAX_REQUESTS = 500  # bound the board so a runaway agent can't grow it unboundedly


def _default_dir() -> Path:
    """The board's directory, via the ONE shared resolver (``knowledge/paths.py``).

    Was a private ``parents[3]`` walk frozen into a module constant — one of five
    such copies in ``knowledge/*``, and the exact shape of . Resolved
    per call now, so ``MAST_LITERATURE_LIBS_DIR`` is honoured even when set after
    import; the default is unchanged.
    """
    from mast.knowledge.paths import libs_dir
    return libs_dir()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _active_experiment_id() -> str:
    """The currently active experiment id, or ``""``. **Never raises** — the board
    must keep working with no runtime, no DB and no experiment."""
    try:
        from mast.documents.paths import current_scope
        eid, _sid = current_scope()
        return str(eid or "")
    except Exception:  # noqa: BLE001
        return ""


def _is_valid_work_id(work_id: Any) -> bool:
    if not isinstance(work_id, str):
        return False
    w = work_id.strip()
    if not w or len(w) > 512:
        return False
    # reject control chars / path separators (defence in depth, same as libraries)
    return not any(ord(c) < 32 for c in w) and "\\" not in w


class FetchBoard:
    """Thread-safe JSON-backed board of full-text fetch requests."""

    def __init__(self, board_dir: str | os.PathLike[str] | None = None) -> None:
        self._dir = Path(board_dir) if board_dir is not None else _default_dir()
        self._path = self._dir / "fetch_requests.json"
        self._lock = threading.RLock()
        self._items: dict[str, dict] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────
    def _load(self) -> None:
        with self._lock:
            self._items = {}
            try:
                if self._path.exists():
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                    for rec in (raw.get("requests") or []):
                        rid = rec.get("request_id")
                        if rid:
                            self._items[rid] = rec
            except Exception as exc:  # corrupt file → start empty, don't crash
                logger.warning("fetch board load failed (%s); starting empty", exc)
                self._items = {}

    def _save(self) -> None:
        with self._lock:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".json.tmp")
                payload = {"requests": list(self._items.values())}
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, self._path)
            except Exception as exc:  # best-effort; never crash the caller
                logger.warning("fetch board save failed: %s", exc)

    # ── api ──────────────────────────────────────────────────────────
    def _next_id(self) -> str:
        # monotonic-ish id without Date.now(): max existing index + 1
        n = 0
        for rid in self._items:
            if rid.startswith("fr-"):
                try:
                    n = max(n, int(rid[3:]))
                except ValueError:
                    pass
        return f"fr-{n + 1}"

    def post_request(self, work_id: str, *, doi: str = "", title: str = "",
                     reason: str = "", requested_by: str = "agent",
                     experiment_id: str | None = None,
                     origin_conversation_id: str = "") -> dict:
        """Post (or return the existing OPEN) request for ``work_id``.

        ``experiment_id`` is **frozen at creation** (design trap ⑯). Left as
        ``None`` it captures whichever experiment is active right now. The reason
        it cannot be resolved later, at fulfilment time: a request raised while
        experiment A ran may sit on the board for days and be fulfilled while B is
        active — filing the paper into B's library because B happens to be active
        is simply the wrong drawer, and nothing afterwards would reveal the error.

        Dedup is keyed on ``(work_id, experiment_id)``, not ``work_id`` alone: if
        two experiments each need the same paper they each get their own row, so
        each gets its own filing when the upload arrives. One upload still closes
        all of them (``resolve_work_id``).

        ``origin_conversation_id`` records **which conversation asked**, so that
        fulfilling the request can wake that conversation back up and let the
        agent carry on (``mast.core.fetch_resume``). It is stored verbatim and
        does NOT participate in dedup: a second conversation asking for the same
        paper in the same experiment still gets the first row back, so only the
        first asker is resumed (documented limitation — splitting the dedup key
        would put duplicate rows on the operator's board, which is worse).
        Empty when the caller has no conversation (background runs), which
        degrades to today's behaviour: the board closes, nobody is woken.
        """
        if not _is_valid_work_id(work_id):
            return {"error": "invalid work_id"}
        if experiment_id is None:
            experiment_id = _active_experiment_id() or None
        with self._lock:
            # dedup: an open request for the same work_id AND experiment
            for rec in self._items.values():
                if (rec.get("work_id") == work_id
                        and rec.get("status") == _OPEN
                        and (rec.get("experiment_id") or None) == (experiment_id or None)):
                    return dict(rec)
            if sum(1 for r in self._items.values() if r.get("status") == _OPEN) >= MAX_REQUESTS:
                return {"error": "fetch board is full (too many open requests)"}
            rid = self._next_id()
            rec = {
                "request_id": rid, "work_id": work_id.strip(),
                "doi": (doi or "").strip(), "title": (title or "").strip(),
                "reason": (reason or "").strip(),
                "requested_by": (requested_by or "agent").strip() or "agent",
                "experiment_id": experiment_id,
                "origin_conversation_id": (origin_conversation_id or "").strip(),
                "status": _OPEN, "created_at": _now_iso(),
                "resolved_at": None, "note": "",
            }
            self._items[rid] = rec
            self._save()
            return dict(rec)

    def list_requests(self, status: str | None = None) -> list[dict]:
        with self._lock:
            rows = [dict(r) for r in self._items.values()
                    if status is None or r.get("status") == status]
        # open first, then most-recent
        rows.sort(key=lambda r: (r.get("status") != _OPEN, r.get("created_at") or ""),
                  reverse=False)
        return rows

    def get_request(self, request_id: str) -> dict | None:
        with self._lock:
            rec = self._items.get(request_id)
            return dict(rec) if rec else None

    def resolve(self, request_id: str, status: str, *, note: str = "") -> dict:
        if status not in STATUSES:
            return {"error": f"invalid status {status!r}"}
        with self._lock:
            rec = self._items.get(request_id)
            if rec is None:
                return {"error": "no such request"}
            rec["status"] = status
            rec["note"] = (note or "").strip()
            rec["resolved_at"] = _now_iso()
            self._save()
            return dict(rec)

    def resolve_work_id(self, work_id: str, status: str, *, note: str = "") -> int:
        """Resolve all OPEN requests for a work_id (used when an ingest/fetch of
        that paper succeeds). Returns the number resolved."""
        n = 0
        with self._lock:
            for rec in self._items.values():
                if rec.get("work_id") == work_id and rec.get("status") == _OPEN:
                    rec["status"] = status
                    rec["note"] = (note or "").strip()
                    rec["resolved_at"] = _now_iso()
                    n += 1
            if n:
                self._save()
        return n

    def open_requests(self, work_id: str) -> list[dict]:
        """Deep copies of every OPEN request for ``work_id``.

        Same **read-before-resolve** discipline as :meth:`open_experiment_ids`:
        once ``resolve_work_id`` has run, the rows are no longer OPEN and this
        returns nothing — so a fulfilment path that wants to tell the askers
        "your paper arrived" must take this snapshot FIRST. Copies, not the live
        records, so a caller passing them around cannot mutate the board.
        """
        out: list[dict] = []
        with self._lock:
            for rec in self._items.values():
                if rec.get("work_id") == work_id and rec.get("status") == _OPEN:
                    out.append(dict(rec))
        out.sort(key=lambda r: r.get("created_at") or "")
        return out

    def open_experiment_ids(self, work_id: str) -> list[str]:
        """The frozen ``experiment_id``s of every OPEN request for ``work_id``.

        Call this **before** ``resolve_work_id`` — once resolved you have lost the
        list of who was waiting. Each id is a library the fulfilled paper should be
        filed into; ``""`` in the list means that request carried no experiment
        (posted with none active) and belongs in the manual pointer's library.
        Order follows creation. Duplicates are collapsed.
        """
        out: list[str] = []
        with self._lock:
            for rec in self._items.values():
                if rec.get("work_id") == work_id and rec.get("status") == _OPEN:
                    eid = str(rec.get("experiment_id") or "")
                    if eid not in out:
                        out.append(eid)
        return out

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._items.values() if r.get("status") == _OPEN)

    # ── telling the asker their paper arrived ─────────────────────────
    #
    # Auto-resume (``core.fetch_resume``) is the primary route: the conversation
    # that asked gets driven one more turn the moment the paper lands. It cannot
    # always fire — a request posted from a background run has no conversation to
    # return to, the engine may be down, the operator may have switched it off —
    # and in those cases the answer used to sit on the board exactly as the
    # wishlist answers used to, waiting for an agent to think of polling.
    #
    # So the board records whether an arrival was ever ANNOUNCED, mirroring the
    # wishlist's ``delivered`` flag. Whatever announces it marks it; the readback
    # middleware sweeps up whatever nothing announced.

    def unannounced_fulfilled(self, requested_by: str = "") -> list[dict]:
        """Fulfilled requests whose arrival nobody has told the agent about yet."""
        with self._lock:
            out = []
            for r in self._items.values():
                if r.get("status") != "fulfilled" or r.get("announced"):
                    continue
                if requested_by and str(r.get("requested_by") or "") != requested_by:
                    continue
                out.append(dict(r))
        out.sort(key=lambda r: r.get("resolved_at") or "")
        return out

    def mark_announced(self, request_ids: list[str]) -> int:
        """Mark arrivals as told, so the agent is not handed the same one twice."""
        ids = set(request_ids or ())
        n = 0
        with self._lock:
            for rid, rec in self._items.items():
                if rid in ids and not rec.get("announced"):
                    rec["announced"] = True
                    n += 1
            if n:
                self._save()
        return n


# ── module-level singleton + injection (mirrors libraries.py) ─────────
_BOARD: FetchBoard | None = None
_BOARD_LOCK = threading.Lock()


def get_board() -> FetchBoard:
    global _BOARD
    with _BOARD_LOCK:
        if _BOARD is None:
            _BOARD = FetchBoard()
        return _BOARD


def reset_default_board() -> None:
    """Test hook: drop the cached singleton."""
    global _BOARD
    with _BOARD_LOCK:
        _BOARD = None


def _b(board: FetchBoard | None) -> FetchBoard:
    return board if board is not None else get_board()


def post_request(work_id: str, *, board: FetchBoard | None = None, **kw) -> dict:
    return _b(board).post_request(work_id, **kw)


def list_requests(status: str | None = None, *, board: FetchBoard | None = None) -> list[dict]:
    return _b(board).list_requests(status)


def resolve(request_id: str, status: str, *, note: str = "",
            board: FetchBoard | None = None) -> dict:
    return _b(board).resolve(request_id, status, note=note)


def resolve_work_id(work_id: str, status: str, *, note: str = "",
                    board: FetchBoard | None = None) -> int:
    return _b(board).resolve_work_id(work_id, status, note=note)


def pending_count(*, board: FetchBoard | None = None) -> int:
    return _b(board).pending_count()


def open_experiment_ids(work_id: str, *, board: FetchBoard | None = None) -> list[str]:
    return _b(board).open_experiment_ids(work_id)


def open_requests(work_id: str, *, board: FetchBoard | None = None) -> list[dict]:
    return _b(board).open_requests(work_id)


def unannounced_fulfilled(requested_by: str = "", *,
                          board: FetchBoard | None = None) -> list[dict]:
    return _b(board).unannounced_fulfilled(requested_by)


def mark_announced(request_ids: list[str], *,
                   board: FetchBoard | None = None) -> int:
    return _b(board).mark_announced(request_ids)


__all__ = [
    "FetchBoard", "get_board", "reset_default_board", "post_request",
    "list_requests", "resolve", "resolve_work_id", "pending_count",
    "open_experiment_ids", "open_requests", "unannounced_fulfilled",
    "mark_announced", "STATUSES",
]
