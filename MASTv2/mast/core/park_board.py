"""The durable board of parked agents — a park that outlives its run.

``docs/v2/design/wakeup_scheduling.md`` §4. W2 parked agents in ``MASTState``, which
is enough inside one run and useless for the thing the mechanism is FOR: being woken
later. When the system is idle no run exists, so no state exists, so nothing holds
the park. This board does.

Authority, and why it matters that it is singular
-------------------------------------------------
**The board is the authority; ``MASTState.pending_activations`` is a cache.** Same
split as ``versions.jsonl`` (authority) vs the documents DB index (cache), and for
the same reason: two writers appear here on two different threads — the graph parks
an agent while the wake scheduler increments its decline count — and if both wrote
both places they would drift within a day. So the rule is narrower than "board is
authority": **every WRITE goes to the board only**, and each run seeds its state
copy from the board at the start.

experiment_id is frozen at creation
-----------------------------------
Copied deliberately from ``knowledge/fetch_board.py``'s design trap, because
this board has exactly the same hazard and a longer time horizon: a park raised while
experiment A ran may sit here for days and be woken while B is active. Waking it into
B would run the agent against the wrong experiment's data, and nothing afterwards
would reveal the error. The experiment is captured when the park is created and never
re-resolved.

``woken`` is not the end
-----------------------
The status set has ``done`` as well, and that is not bookkeeping neatness. A woken run
is a detached background run on an ``InMemorySaver`` — if it dies mid-way, its work is
gone, and a board that treated ``woken`` as terminal would have no record that the
chain broke. The park is only finished when the run's products come back (W5's return
inbox) or the run reaches a terminal state. Otherwise this mechanism would leak tasks
into silence, which is precisely the failure mode it is most at risk of.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: ``waiting`` → asked-and-chose-to-wait, or a hard dependency was missing.
#: ``woken``   → a run was spawned for it; NOT finished (see the module docstring).
#: ``done``    → the woken run's products came back. The only clean终点.
#: ``expired`` → the deadline passed. Escalated to the operator and, critically, NOT
#:               resumed automatically and NOT deleted — it stays visible until a
#:               human acknowledges it.
#: ``cancelled`` → withdrawn (the operator, or the product arriving another way).
#: ``done_by_goal`` → 这份等待所服务的**目标已经达成**，所以它不必再醒（2026-08-27）。
#:               与 ``done`` 分开是因为两者回答的是不同的问题：``done`` 说「它醒了、
#:               干完了、产物回来了」，``done_by_goal`` 说「它一次都没醒，而且不用醒
#:               了」。合成一个，用户就再也答不出「那件事到底是谁做的」。
STATUSES = ("waiting", "woken", "done", "done_by_goal", "expired", "cancelled")
_OPEN = "waiting"

#: Bound the board. A runaway parking loop must not grow a file without limit.
MAX_ENTRIES = 500

#: Default wait before a park escalates to the operator. 24 h matches the rhythm of
#: STM work (an overnight run is normal); the operator can retune it.
DEFAULT_TTL_S = 24 * 3600


def _board_path() -> Path:
    """Where the board lives: under the USER DATA root, never the install dir.

    Anchored on ``_runtime_paths.project_root()`` — the same anchor the billing
    ledger and the experiment folders use — and **deliberately not** on
    ``knowledge.paths.base_dir()``, which the neighbouring boards use. That
    distinction is the whole point of this function:

      * ``base_dir()`` frozen resolves to the directory holding the EXE
        (``C:\\MAST``), and its own docstring says why: ``MASTv2/artifacts/`` there is
        **assets shipped with the exe**. An OTA update replaces that tree.
      * A park is USER STATE with a multi-day lifetime — its entire purpose is to
        still be there when the thing it waits for finally arrives.

    Putting it under the install dir would mean an update quietly deletes every
    pending activation, and "quietly deletes a waiting agent" is precisely the
    silent-death failure this whole subsystem is built to prevent. So it lives
    beside the ledger in the configured user data root.

    Resolved per call, never frozen into a module constant: that pattern was Finding
    #137 (five private ``parents[3]`` walks in ``knowledge/*``), and it means an env
    override set after import is silently ignored — which in tests is how a suite
    starts writing into the operator's real data.
    """
    override = os.environ.get("MAST_PARK_BOARD_DIR")
    if override:
        return Path(override) / "park_board.json"
    try:
        from mast._runtime_paths import project_root

        return Path(project_root()) / "artifacts" / "activation" / "park_board.json"
    except Exception:  # noqa: BLE001
        return Path("artifacts") / "activation" / "park_board.json"


def _jsonable(value: Any) -> Any:
    """Convert an artifact payload into something ``json.dumps`` accepts.

    Found by test, 2026-07-30, and it was silent: the product channel carries
    ``DocRef`` / ``ScanResult`` / ``AnalysisResult`` — Pydantic models, not dicts —
    and ``json.dumps`` refuses them. ``_save`` is best-effort by design, so the write
    failed, logged a debug-level warning, and ``post_return`` still reported success.
    The inbox exists precisely to survive until the mainline next dispatches, which
    may be after a restart, so "it worked until the process ended" is the worst
    possible failure mode for it.

    Converting on the way IN (rather than teaching the reader to handle models) is
    the right side: a checkpoint would serialise these to plain dicts anyway, and
    every reader already goes through ``artifact_channel._get``, which handles dicts
    and objects identically.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:  # noqa: BLE001
                pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _active_experiment_id() -> str:
    """Currently active experiment id, or ``""``. **Never raises** — the board has to
    keep working with no runtime, no DB and no experiment."""
    try:
        from mast.documents.paths import current_scope

        eid, _sid = current_scope()
        return str(eid or "")
    except Exception:  # noqa: BLE001
        return ""


class ParkBoard:
    """Thread-safe JSON-backed board of parked agent activations.

    Shape copied from :class:`mast.knowledge.fetch_board.FetchBoard` — RLock, load on
    construction, atomic temp+replace on write, corrupt file degrades to empty rather
    than crashing. Reusing a proven shape rather than inventing a second persistence
    idiom for the same job.
    """

    def __init__(self, path: "str | os.PathLike[str] | None" = None) -> None:
        self._path = Path(path) if path is not None else _board_path()
        self._lock = threading.RLock()
        self._items: dict[str, dict] = {}
        self._inbox: list[dict] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────
    def _load(self) -> None:
        with self._lock:
            self._items = {}
            self._inbox = []
            try:
                if self._path.exists():
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                    for rec in (raw.get("parks") or []):
                        pid = rec.get("park_id")
                        if pid:
                            self._items[pid] = rec
                    self._inbox = list(raw.get("inbox") or [])
            except Exception as exc:  # noqa: BLE001 — corrupt file → empty, not crash
                logger.warning("park board load failed (%s); starting empty", exc)
                self._items = {}
                self._inbox = []

    def _save(self) -> None:
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".json.tmp")
                payload = {"parks": list(self._items.values()),
                           "inbox": list(self._inbox)}
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                os.replace(tmp, self._path)
            except Exception as exc:  # noqa: BLE001 — best effort; never crash a caller
                logger.warning("park board save failed: %s", exc)

    # ── the return inbox ─────────────────────────────────────────────
    #
    # A woken run produces something; without a way back, it produced it for nobody.
    # It cannot write the mainline's checkpoint (checkpoints are per-thread; two
    # writers corrupt it), so its products are POSTED here and the mainline drains
    # them at its next dispatch. Same file and same lock as the parks because it is
    # the same fact from the other end — and because a second persistence idiom for
    # the same job is a second thing to get wrong.

    def post_return(self, *, run_id: str, park_id: str, agent: str,
                    artifacts: dict, experiment_id: str = "") -> dict:
        """File a woken run's products for the mainline to pick up.

        Pointers only — the same ``carried_from`` payload a handoff would carry —
        because the bodies are already on disk and this file is not an archive.

        Deduped on ``run_id``: a re-post (a retry, or a run whose completion is
        observed twice) must not enqueue the same products twice.
        """
        if not artifacts:
            return {}
        rec = {
            "entry_id": f"rt-{uuid.uuid4().hex[:10]}",
            "run_id": str(run_id or ""),
            "park_id": str(park_id or ""),
            "agent": str(agent or ""),
            "experiment_id": str(experiment_id or ""),
            # Converted here, not at read time — see _jsonable.
            "artifacts": _jsonable(dict(artifacts)),
            "posted_at": time.time(),
            "drained_at": 0.0,
        }
        with self._lock:
            for existing in self._inbox:
                if (existing.get("run_id") == rec["run_id"]
                        and not float(existing.get("drained_at") or 0)):
                    return dict(existing)
            self._inbox.append(rec)
            # Bound it: undrained entries accumulate if the mainline never runs again.
            if len(self._inbox) > MAX_ENTRIES:
                del self._inbox[:-MAX_ENTRIES]
            self._save()
        logger.info("return inbox: %s posted %s from run %s",
                    rec["agent"], sorted(artifacts), rec["run_id"])
        return dict(rec)

    def peek_returns(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._inbox
                    if not float(r.get("drained_at") or 0)]

    def drain_returns(self) -> list[dict]:
        """Take every undrained entry and mark it drained. Idempotent by construction.

        Marking inside the same lock as the read is what makes a concurrent second
        drain (a fan-out where two branches both dispatch) return nothing rather than
        the same products twice.
        """
        now = time.time()
        with self._lock:
            out = [dict(r) for r in self._inbox
                   if not float(r.get("drained_at") or 0)]
            if not out:
                return []
            taken = {r["entry_id"] for r in out}
            for r in self._inbox:
                if r.get("entry_id") in taken:
                    r["drained_at"] = now
            self._save()
        return out

    # ── writes ───────────────────────────────────────────────────────
    def park(self, agent: str, *, waiting_for: "list[str] | tuple[str, ...]",
             reason: str = "", instruction: str = "", hard: bool = False,
             experiment_id: "str | None" = None,
             artifact_snapshot: "dict | None" = None,
             ttl_s: float = DEFAULT_TTL_S,
             conversation_id: str = "", campaign_id: str = "") -> dict:
        """Park ``agent``, or REFRESH its existing open park. Returns the entry.

        Dedup is on ``(agent, experiment_id)`` while the park is open: an agent the
        supervisor declines to dispatch three hops running is one wait, not three, and
        three rows would each carry their own deadline and decline count — the UI would
        show a queue where there is one fact.

        Refreshing deliberately does NOT reset ``created_at`` or ``deadline_at``. The
        wait started when it started; letting a re-park slide the deadline forward is
        how a bounded wait becomes an unbounded one.

        ``campaign_id`` is FROZEN here (2026-08-27), for the same reason
        ``experiment_id`` is: when the wake scheduler later asks "is the goal this
        park serves already met?", nothing in an idle process can re-derive which
        research campaign this wait belonged to. The v1 experiment row's
        ``v2_campaign_id`` only ever points at the GUI-scope campaign, and a campaign
        founded by research_director is linked to no experiment at all — so a guess
        would silently resolve to the wrong programme. Empty is honest and simply
        means "no goal check for this park".
        """
        eid = _active_experiment_id() if experiment_id is None else str(experiment_id or "")
        now = time.time()
        with self._lock:
            for rec in self._items.values():
                if (rec.get("agent") == agent
                        and (rec.get("experiment_id") or "") == eid
                        and rec.get("status") == _OPEN):
                    rec["waiting_for"] = list(waiting_for)
                    if reason:
                        rec["reason"] = reason
                    rec["hard"] = bool(hard)
                    rec["updated_at"] = now
                    # 非空才覆盖：一次拿不到 campaign 的重新 park 不该把已经
                    # 冻结的归属抹掉（那会让这份 park 的目标判据从此判不了）。
                    if campaign_id:
                        rec["campaign_id"] = str(campaign_id)
                    if artifact_snapshot is not None:
                        rec["artifact_snapshot"] = _jsonable(dict(artifact_snapshot))
                    self._save()
                    return dict(rec)
            if sum(1 for r in self._items.values() if r.get("status") == _OPEN) >= MAX_ENTRIES:
                logger.warning("park board full (%d open); refusing new park for %s",
                               MAX_ENTRIES, agent)
                return {"error": "park board is full"}
            rec = {
                "park_id": f"pk-{uuid.uuid4().hex[:10]}",
                "agent": str(agent),
                # FROZEN here, never re-resolved — see the module docstring.
                "experiment_id": eid,
                "conversation_id": str(conversation_id or ""),
                # FROZEN, like experiment_id above — see the docstring.
                "campaign_id": str(campaign_id or ""),
                "waiting_for": list(waiting_for),
                "reason": str(reason or ""),
                "instruction": str(instruction or ""),
                "hard": bool(hard),
                "artifact_snapshot": _jsonable(dict(artifact_snapshot or {})),
                "created_at": now,
                "updated_at": now,
                "deadline_at": now + max(60.0, float(ttl_s or DEFAULT_TTL_S)),
                "declines": 0,
                # PER-KIND versions, not one scalar: ``waiting_for`` is a list, and a
                # document's version is an int while a scan "version" is a count and an
                # mtime. One number could not say which of several waits had moved.
                "asked_at_versions": {},
                "asked_at": 0.0,
                "status": _OPEN,
                "woken_run_id": "",
                "woken_at": 0.0,
                "acknowledged_at": 0.0,
                "history": [],
            }
            self._items[rec["park_id"]] = rec
            self._save()
            logger.info("parked %s waiting for %s (park %s, exp %s)",
                        agent, list(waiting_for), rec["park_id"], eid or "-")
            return dict(rec)

    def _mutate(self, park_id: str, **fields) -> dict:
        with self._lock:
            rec = self._items.get(park_id)
            if rec is None:
                return {"error": "no such park"}
            rec.update(fields)
            rec["updated_at"] = time.time()
            self._save()
            return dict(rec)

    def note_decision(self, park_id: str, *, woke: bool, reason: str,
                      versions: "dict | None" = None) -> dict:
        """Record one wake decision, whichever way it went.

        Every "the agent chose NOT to wake" is written down with its reason. Without
        that the board shows a park getting older with no explanation, which is the
        same unreadable state as a hang. ``declines`` also feeds back into the next
        question — an agent that does not know it has declined three times cannot
        decide better than it did the first time.
        """
        with self._lock:
            rec = self._items.get(park_id)
            if rec is None:
                return {"error": "no such park"}
            entry = {"at": time.time(), "woke": bool(woke),
                     "reason": str(reason or "")[:500]}
            rec.setdefault("history", []).append(entry)
            # Bound the history: a park polled for days must not grow without limit.
            if len(rec["history"]) > 50:
                del rec["history"][:-50]
            rec["asked_at"] = entry["at"]
            if versions:
                rec["asked_at_versions"] = dict(versions)
            if not woke:
                rec["declines"] = int(rec.get("declines") or 0) + 1
            rec["updated_at"] = entry["at"]
            self._save()
            return dict(rec)

    def mark_woken(self, park_id: str, run_id: str) -> dict:
        """A run was spawned. NOT terminal — see the module docstring."""
        return self._mutate(park_id, status="woken", woken_run_id=str(run_id or ""),
                            woken_at=time.time())

    def mark_done(self, park_id: str, *, note: str = "") -> dict:
        """The woken run's products came back. The only clean终点."""
        return self._mutate(park_id, status="done", note=str(note or ""),
                            done_at=time.time())

    def mark_done_by_goal(self, park_id: str, *, reason: str = "",
                          campaign_id: str = "") -> dict:
        """这份等待所服务的目标已经达成 —— 它不必再醒。

        与 :meth:`mark_done` 分开：那个说「醒了、干完了」，这个说「一次都没醒，
        而且不用醒了」。终态，所以 ``open_parks`` 之后不会再看到它。
        """
        return self._mutate(park_id, status="done_by_goal",
                            note=str(reason or "")[:500],
                            goal_campaign_id=str(campaign_id or ""),
                            done_at=time.time())

    def note_goal_check(self, park_id: str, *, verdict: str,
                        reason: str = "") -> dict:
        """记下最近一次目标判据求值（**只在结论变化时落盘**）。

        调度器每 60 s 走一趟；每趟都写一次磁盘只是为了记下「还是没满足」，那是
        一分钟一次的无谓写入，也会把 ``updated_at`` 变成一个没有信息量的字段。
        判据本身不由这里决定 —— 这里只负责让用户在面板上看得见「目标判据：
        还差 X」或「读不到（为什么）」。
        """
        with self._lock:
            rec = self._items.get(park_id)
            if rec is None:
                return {"error": "no such park"}
            prev = rec.get("goal_check") or {}
            if (prev.get("verdict") == verdict
                    and str(prev.get("reason") or "") == str(reason or "")):
                return dict(rec)          # 没变 —— 不写盘
            rec["goal_check"] = {"verdict": str(verdict),
                                 "reason": str(reason or "")[:500],
                                 "checked_at": time.time()}
            self._save()
            return dict(rec)

    def mark_expired(self, park_id: str) -> dict:
        """The deadline passed. Escalated to the operator — **not** resumed, **not**
        deleted. It remains on the board until acknowledged, because in the idle case
        (the main case!) an SSE frame has no receiver and a transcript line is in a
        conversation nobody has open. A one-shot notification is not a delivery."""
        return self._mutate(park_id, status="expired", expired_at=time.time())

    def cancel(self, park_id: str, *, note: str = "") -> dict:
        return self._mutate(park_id, status="cancelled", note=str(note or ""))

    def acknowledge(self, park_id: str) -> dict:
        """The operator has seen it. Stops the row demanding attention; does not
        delete it (the record of a wait that timed out is worth keeping)."""
        return self._mutate(park_id, acknowledged_at=time.time())

    def sweep_expired(self, *, now: float | None = None) -> list[dict]:
        """Expire every open park past its deadline. Returns the newly-expired rows."""
        t = time.time() if now is None else float(now)
        out: list[dict] = []
        with self._lock:
            for rec in self._items.values():
                if rec.get("status") == _OPEN and float(rec.get("deadline_at") or 0) <= t:
                    rec["status"] = "expired"
                    rec["expired_at"] = t
                    rec["updated_at"] = t
                    out.append(dict(rec))
            if out:
                self._save()
        for rec in out:
            logger.warning("park %s (%s waiting for %s) EXPIRED — escalating",
                           rec["park_id"], rec["agent"], rec.get("waiting_for"))
        return out

    # ── reads ────────────────────────────────────────────────────────
    def list_parks(self, *, status: "str | None" = None,
                   experiment_id: "str | None" = None,
                   include_terminal: bool = True) -> list[dict]:
        with self._lock:
            rows = [dict(r) for r in self._items.values()]
        if status is not None:
            rows = [r for r in rows if r.get("status") == status]
        if experiment_id is not None:
            rows = [r for r in rows if (r.get("experiment_id") or "") == str(experiment_id or "")]
        if not include_terminal:
            rows = [r for r in rows if r.get("status") in (_OPEN, "woken", "expired")]
        # Attention first: expired, then waiting, then everything else; newest first
        # within a group.
        rank = {"expired": 0, _OPEN: 1, "woken": 2}
        rows.sort(key=lambda r: (rank.get(r.get("status"), 9),
                                 -float(r.get("created_at") or 0)))
        return rows

    def get(self, park_id: str) -> "dict | None":
        with self._lock:
            rec = self._items.get(park_id)
            return dict(rec) if rec else None

    def open_parks(self, experiment_id: "str | None" = None) -> list[dict]:
        return self.list_parks(status=_OPEN, experiment_id=experiment_id)

    def needs_attention(self) -> list[dict]:
        """Expired and not yet acknowledged — the rows the UI must keep highlighting."""
        return [r for r in self.list_parks(status="expired")
                if not float(r.get("acknowledged_at") or 0)]

    def as_state_cache(self, experiment_id: "str | None" = None) -> dict:
        """The ``MASTState.pending_activations`` value to seed a run with.

        The board is the authority; this is the read that makes a fresh run aware of
        parks raised before it existed. Without it every new run would re-ask every
        parked agent — paying again for a decision already on disk.
        """
        out: dict[str, Any] = {}
        for r in self.open_parks(experiment_id):
            out[r["agent"]] = {
                "status": "waiting",
                "waiting_for": list(r.get("waiting_for") or []),
                "reason": r.get("reason") or "",
                "at": r.get("created_at") or 0.0,
                "park_id": r.get("park_id") or "",
                "hard": bool(r.get("hard")),
            }
        return out


_BOARD: "ParkBoard | None" = None
_BOARD_LOCK = threading.Lock()


def board() -> ParkBoard:
    """Process-wide board. Lazily built so importing this module touches no disk."""
    global _BOARD
    with _BOARD_LOCK:
        if _BOARD is None:
            _BOARD = ParkBoard()
        return _BOARD


def set_board_for_test(b: "ParkBoard | None") -> None:
    """Install (or clear) the process board.

    Exists because this repo has polluted the operator's real data four times, always
    through the same gap: a test redirects one env var while the store resolves a
    different one. An explicit injection point removes the guesswork.
    """
    global _BOARD
    with _BOARD_LOCK:
        _BOARD = b


__all__ = ["ParkBoard", "STATUSES", "DEFAULT_TTL_S", "MAX_ENTRIES",
           "board", "set_board_for_test"]
