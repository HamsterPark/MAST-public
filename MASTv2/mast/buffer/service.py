"""BufferService — s-scale handshake between Vision (ms) and Agents (min).

Phase 2 implementation. Per compass §3.2:

  - Producer API (sync, called from VisionProducer thread):
      next_seq()                  → monotonic int
      put_tip_status(ts)
      put_region(rm)
      put_progress(p)
      emit_event(ev)              edge-triggered

  - Consumer API (async, called from agent coroutines):
      get_latest_tip_status()     → (TipStatus|None, seqno)
      get_latest_progress()
      get_tip_history(since_seq)
      subscribe(kind)             → asyncio.Queue (drop-oldest on full)

Cross-thread correctness (compass §3.2 bullet list):
  - asyncio.Queue is NOT thread-safe → producer→subscriber goes through
    `loop.call_soon_threadsafe(_nowait_put, q, ev)`
  - aiosqlite WAL writes scheduled via `asyncio.run_coroutine_threadsafe`
  - `_LatestSlot` pairs `(value, seqno)` writes under `threading.Lock` so
    consumers never see a torn pair
  - `_maybe_emit_quality_drop` uses rising-edge detection: emit only when
    quality transitions good → degraded|bad (not every frame)
  - Slow subscriber policy: drop oldest on QueueFull rather than block the
    vision thread

"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Generic, TypeVar

import aiosqlite

from mast.buffer.schemas import (
    RegionMap,
    ScanProgress,
    Severity,
    TipQuality,
    TipStatus,
    VisionEvent,
    VisionEventType,
)

logger = logging.getLogger(__name__)


def _safe_view_tip(ts: "TipStatus | None") -> "TipStatus | None":
    """The SAFE-mode READ view of a stored tip status.

    Writing "good" at the producer covers everything published *while* SAFE is
    on. It does not cover what was already in the buffer when the operator
    switched — and that is the likeliest case of all: **the operator switches to
    SAFE precisely because the system just called the tip bad**, so at the moment
    of the switch the latest TipStatus is almost certainly BAD. Without this the
    agent's very first ``read_latest_tip_status`` in SAFE returns "bad", and
    ``full_scan`` attaches its "consider ConditionTip/TipPulse" note to every
    scan — the exact incentive SAFE exists to remove.

    Applies to the value the SYSTEM ACTS ON. The真实 verdict stays in the
    milestone event payload (``tip_coarse.safe_mode_raw``), which no agent tool
    reads and every UI surface can.

    Never raises, never touches the stored object (TipStatus is frozen).
    """
    if ts is None:
        return None
    try:
        if ts.quality in (TipQuality.BAD, TipQuality.DEGRADED) and _safe_mode_active_quiet():
            return ts.model_copy(update={"quality": TipQuality.GOOD, "safe_mode": True})
    except Exception:  # noqa: BLE001 — a read path must never fail on this
        pass
    return ts


def _safe_mode_active_quiet() -> bool:
    """SAFE-mode predicate for the quality-drop safety net; never raises.

    Kept as a function (not a module-level import binding) so the buffer keeps
    working if ``mast.core`` is unavailable in a stripped-down embedding.
    """
    try:
        from mast.core.operating_mode import safe_mode_active
        return safe_mode_active()
    except Exception:  # noqa: BLE001 — publisher thread; escalation is the default
        return False


T = TypeVar("T")


class _LatestSlot(Generic[T]):
    """Lock-paired (value, seqno) slot. Reads never observe a torn pair."""

    __slots__ = ("_lock", "_value", "_seqno")

    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._value: T | None = None
        self._seqno: int = -1

    def put(self, value: T, seqno: int) -> None:
        with self._lock:
            self._value = value
            self._seqno = seqno

    def get(self) -> tuple[T | None, int]:
        with self._lock:
            return self._value, self._seqno

    def clear(self) -> None:
        """Reset to the empty (None, -1) state under the lock."""
        with self._lock:
            self._value = None
            self._seqno = -1


class BufferService:
    """In-process asyncio + aiosqlite WAL handshake layer.

    Lifecycle (called from agent main coroutine, NOT vision thread):
        buf = BufferService(wal_path=...)
        await buf.start()
        ... attach VisionProducer to buf ...
        ... agents subscribe / read ...
        await buf.stop()

    The async event loop becomes "the buffer's loop" — vision thread uses it
    to schedule subscriber notifications and WAL writes via threadsafe APIs.
    """

    def __init__(
        self,
        wal_path: Path | None = None,
        history_size: int = 100,
        queue_max_size: int = 1024,
        wal_enabled: bool = True,
    ):
        self._wal_path = Path(wal_path) if wal_path else None
        self._history_size = history_size
        self._queue_max_size = queue_max_size
        self._wal_enabled = wal_enabled and self._wal_path is not None

        # Single shared lock for all latest-slots (light contention is fine —
        # only writes under it; reads also under it briefly).
        self._slot_lock = threading.Lock()
        self._tip_slot: _LatestSlot[TipStatus] = _LatestSlot(self._slot_lock)
        self._region_slot: _LatestSlot[RegionMap] = _LatestSlot(self._slot_lock)
        self._progress_slot: _LatestSlot[ScanProgress] = _LatestSlot(self._slot_lock)

        # History ring for TipStatus (what subscribers might query for catch-up)
        self._tip_history: deque[TipStatus] = deque(maxlen=history_size)
        self._tip_history_lock = threading.Lock()

        # Subscribers per VisionEventType (asyncio.Queue's, populated lazily)
        self._subs: dict[VisionEventType, list[asyncio.Queue]] = defaultdict(list)
        # Fanout-all subscribers (receive every event regardless of kind).
        # Used by the GUI live-log pane.
        self._subs_all: list[asyncio.Queue] = []
        self._subs_lock = threading.Lock()

        # SYNCHRONOUS safety hooks : callbacks invoked inline from
        # emit_event on the PUBLISHER thread, before any asyncio fanout. This
        # is the physical E_STOP path — it must keep working even when the
        # asyncio loop is dead (history-only degraded mode) or busy, so it
        # deliberately does not go through subscriber queues.
        self._critical_hooks: list = []
        self._critical_hooks_lock = threading.Lock()

        # In-memory event history ring (compass §3.2; compose with WAL catchup).
        # Bounded deque — drop-oldest is automatic via maxlen.
        self._event_history: deque[VisionEvent] = deque(maxlen=history_size)
        self._event_history_lock = threading.Lock()

        # Rising-edge wait support (compass §5.2: predicate on monotonic id, not
        # value equality — avoids the asyncio.Condition double-wakeup race).
        # Guarded by `_event_history_lock` — written from the vision producer
        # thread in emit_event() and read from the agent loop thread in
        # wait_for_event(); a bare dict relied on the GIL (finding [107]).
        self._latest_event_seqno: dict[VisionEventType, int] = {}
        # `_cond` is lazily bound in start() because Condition needs a running loop.
        self._cond: asyncio.Condition | None = None
        # Set by stop() under `_cond` to tell parked wait_for_event() waiters to
        # give up and return None instead of re-parking. A bare notify_all() is
        # NOT enough: the waiter's predicate loop re-evaluates _try_match()
        # (which finds nothing new) and calls cond.wait() again, hanging forever
        # past stop(). Cleared on (re)start().
        self._stopping = False

        # Monotonic sequence counter (producer thread bumps; agents read)
        self._seqno: int = 0
        self._seqno_lock = threading.Lock()

        # State for edge-triggered emission
        self._last_emitted_quality: TipQuality | None = None

        # asyncio loop reference (set in start())
        self._loop: asyncio.AbstractEventLoop | None = None

        # WAL DB connection (set in start())
        self._wal_db: aiosqlite.Connection | None = None

        # Track active state
        self._started = False

        # Stats counters (cheap monotonic ints; reads are best-effort consistent).
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "events_published": 0,
            "events_dropped_oldest": 0,
            "events_fanout_failed": 0,
            "wal_writes_failed": 0,
            "wal_tip_writes": 0,
            "wal_event_writes": 0,
            "subscribers_active": 0,
        }

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def _reset_volatile_state(self) -> None:
        """Clear per-session in-memory edge / history / latest state.

        Called on stop() and at the head of a real start() so a restarted
        instance never replays a previous session's edges, events, or latest
        slots (finding [59]). Deliberately does NOT touch:
          - `_seqno`: a single monotonic counter restored from WAL on start();
            resetting it would break post-crash continuity.
          - `_stats`: cumulative counters are useful across restarts.
          - subscribers: managed separately in start()/stop().
        """
        self._last_emitted_quality = None
        with self._event_history_lock:
            self._event_history.clear()
            self._latest_event_seqno.clear()
        with self._tip_history_lock:
            self._tip_history.clear()
        # Reset the latest-value slots so get_latest_* don't return stale data
        # from the previous session. Each clear() takes `_slot_lock` itself.
        self._tip_slot.clear()
        self._region_slot.clear()
        self._progress_slot.clear()

    async def start(self) -> None:
        """Open WAL, capture asyncio loop. Idempotent."""
        if self._started:
            return
        # Fresh session: drop any in-memory edge/history state left over from a
        # prior start()/stop() cycle on this instance (finding [59]). Runs
        # BEFORE the WAL seqno restore below so the restored counter is never
        # clobbered (_reset_volatile_state leaves _seqno untouched).
        self._reset_volatile_state()
        self._stopping = False
        self._loop = asyncio.get_running_loop()
        # Condition needs a running loop — bind it here, not in __init__.
        self._cond = asyncio.Condition()
        if self._wal_enabled and self._wal_path is not None:
            self._wal_path.parent.mkdir(parents=True, exist_ok=True)
            self._wal_db = await aiosqlite.connect(str(self._wal_path))
            await self._wal_db.execute("PRAGMA journal_mode=WAL")
            await self._wal_db.execute(
                """
                CREATE TABLE IF NOT EXISTS tip_status_journal (
                    seqno INTEGER PRIMARY KEY,
                    t_mono_ns INTEGER NOT NULL,
                    quality TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    scan_id TEXT,
                    frame_idx INTEGER,
                    safe_mode INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # `CREATE TABLE IF NOT EXISTS` is a no-op on an existing DB, so an
            # older journal keeps the 6-column shape and every INSERT below would
            # fail (silently, at WARNING). Add the column in place. NOT the same
            # as a migration framework — it is the one additive column this table
            # has ever grown, and a missing `safe_mode` on an old row correctly
            # means "recorded before SAFE could rewrite anything".
            try:
                cur = await self._wal_db.execute("PRAGMA table_info(tip_status_journal)")
                cols = {row[1] for row in await cur.fetchall()}
                await cur.close()
                if "safe_mode" not in cols:
                    await self._wal_db.execute(
                        "ALTER TABLE tip_status_journal "
                        "ADD COLUMN safe_mode INTEGER NOT NULL DEFAULT 0")
                    logger.info("tip_status_journal: added safe_mode column to an existing WAL")
            except Exception as exc:  # noqa: BLE001 — never block startup on this
                logger.warning("tip_status_journal safe_mode column check failed: %s", exc)
            await self._wal_db.execute(
                """
                CREATE TABLE IF NOT EXISTS event_journal (
                    event_id TEXT PRIMARY KEY,
                    seqno INTEGER NOT NULL,
                    t_mono_ns INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    payload_json TEXT
                )
                """
            )
            await self._wal_db.commit()
            # Restore last seqno from WAL so post-crash continuity is preserved.
            # next_seq() is a SINGLE shared monotonic counter feeding BOTH
            # put_tip_status (tip_status_journal) and emit_event (event_journal),
            # so recovery MUST read the global max across both journals. Reading
            # only tip_status_journal under-restores when the last writes before a
            # crash were events (E_STOP / VISION_ERROR / edge drops) or when a
            # session emitted events but never wrote a tip status — causing seqno
            # collisions and silent INSERT OR REPLACE overwrite of earlier rows.
            # 审查 finding [17].
            async with self._wal_db.execute(
                "SELECT MAX(s) FROM ("
                "SELECT MAX(seqno) AS s FROM tip_status_journal "
                "UNION ALL "
                "SELECT MAX(seqno) AS s FROM event_journal)"
            ) as cur:
                row = await cur.fetchone()
                if row and row[0] is not None:
                    with self._seqno_lock:
                        self._seqno = int(row[0])
                        logger.info("BufferService restored seqno=%d from WAL", self._seqno)
        self._started = True

    async def stop(self) -> None:
        """Close WAL, clear subscribers, and release parked wait_for_event()
        waiters.

        Releasing waiters is the load-bearing part. A parked waiter is sitting
        inside ``_waiter()``'s ``while True: ... await cond.wait()`` loop. A bare
        ``notify_all()`` wakes it for ONE predicate re-check — which finds no new
        event and immediately ``cond.wait()``s again, so the waiter (especially a
        ``timeout=None`` one) would hang forever past stop(). We instead set
        ``_stopping`` UNDER the condition lock, THEN notify_all: each woken waiter
        re-checks the predicate, sees ``_stopping`` is set, and returns ``None``.

        We keep ``_cond`` alive across the notify so woken waiters can finish
        their ``async with self._cond`` block cleanly (nulling it out first would
        let a concurrent waiter hit ``None`` mid-flight → AttributeError). The
        reference is dropped only after a yield gives the awoken waiters a chance
        to unwind. ``wait_for_event()`` callers that arrive AFTER stop() get the
        clean ``RuntimeError('not started')`` from its own ``_cond is None`` guard.
        """
        if not self._started:
            return
        # Flip started off first so any NEW wait_for_event() arriving during
        # teardown short-circuits (it re-reads _cond, which we null at the end).
        self._started = False
        with self._subs_lock:
            self._subs.clear()
            self._subs_all.clear()
        # Wake every parked waiter and tell it to give up (return None) rather
        # than re-park. _stopping is read by _try_match()'s caller under _cond.
        cond = self._cond
        if cond is not None:
            self._stopping = True
            async with cond:
                cond.notify_all()
            # Yield control so the awoken waiters actually run their predicate
            # re-check + return before we drop the WAL / cond references.
            await asyncio.sleep(0)
        if self._wal_db is not None:
            await self._wal_db.close()
            self._wal_db = None
        self._cond = None
        # Drop per-session edge / history / latest state so a subsequent
        # start() on this same instance does not replay stale edges or events
        # (finding [59]). _seqno survives (WAL continuity); stats survive.
        self._reset_volatile_state()

    # ─────────────────────────────────────────────────────────────────
    # Producer API (sync — called from VisionProducer thread)
    # ─────────────────────────────────────────────────────────────────

    def next_seq(self) -> int:
        with self._seqno_lock:
            self._seqno += 1
            return self._seqno

    def put_tip_status(self, ts: TipStatus) -> None:
        """Atomically update latest tip slot + history; maybe emit edge event."""
        self._tip_slot.put(ts, ts.seqno)
        with self._tip_history_lock:
            self._tip_history.append(ts)
        # WAL journal append (async, scheduled). Tolerate a closed loop —
        # GUI-initiated BufferService instances start via `asyncio.run()`
        # which tears down the loop after returning; history-only mode is
        # acceptable for poll-based consumers.
        if self._wal_db is not None and self._loop is not None and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._wal_append_tip(ts), self._loop
                )
            except RuntimeError:
                pass
        # Edge-triggered event
        self._maybe_emit_quality_drop(ts)

    def put_region(self, rm: RegionMap) -> None:
        self._region_slot.put(rm, rm.seqno)

    def put_progress(self, p: ScanProgress) -> None:
        self._progress_slot.put(p, p.seqno)

    def emit_event(self, ev: VisionEvent) -> None:
        """Push an event to all subscribers of ev.kind. Non-blocking.

        Behaviour by thread:
          - Same thread as the asyncio loop → fan out synchronously (immediate;
            `q.get_nowait()` right after put_tip_status will see the event)
          - Cross-thread (vision producer thread) → schedule via
            `call_soon_threadsafe` (callback runs on next loop iteration)
        """
        # Track in-memory history (used by GUI / get_event_history) and the
        # per-kind rising-edge id under the SAME lock. Both are read by
        # wait_for_event() on the agent loop thread while this runs on the
        # vision producer thread; the shared lock makes the (history, id) pair
        # consistent and removes the GIL-only guard (finding [107]). Deque
        # auto-trims via maxlen, so this is O(1) even under churn.
        with self._event_history_lock:
            self._event_history.append(ev)
            # Per-kind monotonic id for rising-edge wait_for_event.
            if ev.seqno > self._latest_event_seqno.get(ev.kind, -1):
                self._latest_event_seqno[ev.kind] = ev.seqno
        with self._stats_lock:
            self._stats["events_published"] += 1
        # SYNCHRONOUS safety hooks  — run inline on the publisher thread
        # so an E_STOP reaches threading-level consumers (e.g. the composite
        # abort Event) immediately, with no asyncio-loop or LLM-call-boundary
        # dependency. Hooks must be fast and non-blocking; a failing hook is
        # logged and never breaks fanout.
        with self._critical_hooks_lock:
            hooks = list(self._critical_hooks)
        for fn in hooks:
            try:
                fn(ev)
            except Exception:  # pragma: no cover — defensive
                logger.exception("critical event hook failed for %s", ev.kind)
        with self._subs_lock:
            queues = list(self._subs.get(ev.kind, ()))
            all_queues = list(self._subs_all)
        # Targeted subscribers + subscribe_all fanout.
        # If the captured loop has been closed (e.g. BufferService started
        # via `asyncio.run(start())` which tore down its loop), we degrade
        # to history-only mode: ring buffer + stats + latest_event_seqno
        # still updated, but cross-thread/coroutine scheduling is skipped.
        # GUI poll-based consumers (get_event_history / get_stats) keep
        # working; subscribers / wait_for_event / WAL writes do not.
        loop_alive = self._loop is not None and not self._loop.is_closed()
        fan = queues + all_queues
        if fan and loop_alive:
            if self._on_loop_thread():
                self._fanout_to_queues(fan, ev)
            else:
                try:
                    self._loop.call_soon_threadsafe(self._fanout_to_queues, fan, ev)
                except RuntimeError:
                    pass
        # Notify rising-edge waiters.
        if loop_alive and self._cond is not None:
            if self._on_loop_thread():
                self._loop.create_task(self._notify_cond())
            else:
                try:
                    self._loop.call_soon_threadsafe(
                        lambda: self._loop.create_task(self._notify_cond())  # type: ignore[union-attr]
                    )
                except RuntimeError:
                    pass
        # Persist to WAL (best-effort, always async because aiosqlite is async).
        if self._wal_db is not None and loop_alive:
            try:
                asyncio.run_coroutine_threadsafe(self._wal_append_event(ev), self._loop)
            except RuntimeError:
                pass

    async def _notify_cond(self) -> None:
        """Wake all rising-edge waiters. Must run on the asyncio loop."""
        if self._cond is None:
            return
        async with self._cond:
            self._cond.notify_all()

    def _on_loop_thread(self) -> bool:
        """True if caller is running on the BufferService's asyncio loop thread."""
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    # ─────────────────────────────────────────────────────────────────
    # Consumer API
    # ─────────────────────────────────────────────────────────────────

    def get_latest_tip_status(self) -> tuple[TipStatus | None, int]:
        ts, seq = self._tip_slot.get()      # one read — the slot can change between calls
        return _safe_view_tip(ts), seq

    def get_latest_progress(self) -> tuple[ScanProgress | None, int]:
        return self._progress_slot.get()

    def get_latest_region(self) -> tuple[RegionMap | None, int]:
        return self._region_slot.get()

    def get_tip_history(self, since_seq: int) -> list[TipStatus]:
        """Return TipStatus entries with seqno > since_seq (chronological order)."""
        with self._tip_history_lock:
            snapshot = list(self._tip_history)
        return [_safe_view_tip(t) for t in snapshot if t.seqno > since_seq]

    def register_critical_hook(self, fn) -> None:
        """Register a SYNCHRONOUS callback invoked inline from emit_event.

        修复项 (2026-06-11): the physical E_STOP abort path. Unlike subscribe(),
        hooks fire on the publisher thread before any asyncio fanout, so they
        work even when the loop is dead/busy. ``fn(ev)`` must be fast and
        non-blocking (typical use: set a ``threading.Event``); exceptions are
        swallowed (logged) so a bad hook can never break event fanout.
        Idempotent: registering the same callable twice is a no-op.
        """
        with self._critical_hooks_lock:
            if fn not in self._critical_hooks:
                self._critical_hooks.append(fn)

    def unregister_critical_hook(self, fn) -> None:
        """Remove a hook previously added by register_critical_hook()."""
        with self._critical_hooks_lock:
            try:
                self._critical_hooks.remove(fn)
            except ValueError:
                pass

    def subscribe(
        self, kind: VisionEventType, max_size: int | None = None
    ) -> asyncio.Queue:
        """Subscribe to events of `kind`. Returns asyncio.Queue (drop-oldest policy).

        Caller `await queue.get()` to consume. Caller is responsible for
        unsubscribing via `unsubscribe(kind, queue)` when done.
        """
        size = max_size if max_size is not None else self._queue_max_size
        q: asyncio.Queue = asyncio.Queue(maxsize=size)
        with self._subs_lock:
            self._subs[kind].append(q)
        with self._stats_lock:
            self._stats["subscribers_active"] += 1
        return q

    def unsubscribe(self, kind: VisionEventType, queue: asyncio.Queue) -> None:
        with self._subs_lock:
            try:
                self._subs[kind].remove(queue)
                with self._stats_lock:
                    self._stats["subscribers_active"] = max(
                        0, self._stats["subscribers_active"] - 1
                    )
            except ValueError:
                pass

    def subscribe_all(self, max_size: int | None = None) -> asyncio.Queue:
        """Subscribe to events of *every* kind. Used by GUI live-log pane.

        Drop-oldest policy identical to per-kind subscribe.
        """
        size = max_size if max_size is not None else self._queue_max_size
        q: asyncio.Queue = asyncio.Queue(maxsize=size)
        with self._subs_lock:
            self._subs_all.append(q)
        with self._stats_lock:
            self._stats["subscribers_active"] += 1
        return q

    def unsubscribe_all(self, queue: asyncio.Queue) -> None:
        """Detach a queue previously returned by subscribe_all()."""
        with self._subs_lock:
            try:
                self._subs_all.remove(queue)
                with self._stats_lock:
                    self._stats["subscribers_active"] = max(
                        0, self._stats["subscribers_active"] - 1
                    )
            except ValueError:
                pass

    # ─────────────────────────────────────────────────────────────────
    # Rising-edge wait / catch-up / history / stats (compass §5.2 + §6)
    # ─────────────────────────────────────────────────────────────────

    async def wait_for_event(
        self,
        kind: VisionEventType,
        since_seqno: int = -1,
        timeout: float | None = None,
    ) -> VisionEvent | None:
        """Wait until an event of `kind` with seqno > since_seqno is published.

        Rising-edge semantics (compass §5.2): comparison uses monotonic id, not
        value equality, to avoid the asyncio.Condition double-wakeup race
        documented by the Inngest "asyncio primitives get wrong about shared
        state" blog. If an event with `seqno > since_seqno` already exists when
        called, returns immediately with the most recent matching event from
        in-memory history; otherwise blocks on the Condition until notify_all.

        Returns None on timeout. Returns the matching VisionEvent on success.
        """
        if self._cond is None:
            raise RuntimeError("BufferService not started — call start() first")

        def _try_match() -> VisionEvent | None:
            # Read the latest-id table and scan history under the SAME lock so
            # the (id, history) pair stays consistent with concurrent emit_event
            # writes on the vision thread (finding [107]).
            with self._event_history_lock:
                # Fast path: latest-id table cheaply tells us if a match exists.
                if self._latest_event_seqno.get(kind, -1) <= since_seqno:
                    return None
                # Scan history newest-to-oldest, return first event of kind with
                # higher seqno. History is bounded but always covers the most
                # recent event of every active kind (we record on every emit).
                for ev in reversed(self._event_history):
                    if ev.kind == kind and ev.seqno > since_seqno:
                        return ev
            return None

        # Optimistic non-blocking check before taking the Condition.
        hit = _try_match()
        if hit is not None:
            return hit

        async def _waiter() -> VisionEvent | None:
            assert self._cond is not None
            async with self._cond:
                while True:
                    # stop() sets _stopping under this same lock then notifies;
                    # checking it here (after wait() returns / before re-parking)
                    # lets a parked waiter unwind to None instead of hanging
                    # forever once the service is shutting down.
                    if self._stopping:
                        return None
                    got = _try_match()
                    if got is not None:
                        return got
                    await self._cond.wait()

        try:
            if timeout is None:
                return await _waiter()
            return await asyncio.wait_for(_waiter(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def catchup(
        self,
        since_seqno: int,
        kinds: list[VisionEventType] | None = None,
    ) -> list[VisionEvent]:
        """Replay events with seqno > since_seqno from the WAL event_journal.

        Compass §6 calls this on LISTEN reconnect ("Not durable — missed
        events are gone"); the in-process equivalent here is aiosqlite WAL
        replay so a consumer restarting can resume from its last-seen seqno.

        If WAL is disabled, falls back to the in-memory history ring (which
        is bounded to history_size — older events may be lost).
        """
        if kinds is not None:
            kind_filter = {k.value for k in kinds}
        else:
            kind_filter = None

        if self._wal_db is not None:
            return await self._catchup_from_wal(since_seqno, kind_filter)
        # Fallback: in-memory history (best effort).
        with self._event_history_lock:
            snapshot = list(self._event_history)
        out = [
            ev for ev in snapshot
            if ev.seqno > since_seqno
            and (kind_filter is None or ev.kind.value in kind_filter)
        ]
        out.sort(key=lambda e: e.seqno)
        return out

    async def _catchup_from_wal(
        self, since_seqno: int, kind_filter: set[str] | None
    ) -> list[VisionEvent]:
        assert self._wal_db is not None
        import json
        if kind_filter is None:
            sql = (
                "SELECT event_id, seqno, t_mono_ns, kind, severity, payload_json "
                "FROM event_journal WHERE seqno > ? ORDER BY seqno"
            )
            args: tuple = (since_seqno,)
        else:
            placeholders = ",".join("?" * len(kind_filter))
            sql = (
                "SELECT event_id, seqno, t_mono_ns, kind, severity, payload_json "
                f"FROM event_journal WHERE seqno > ? AND kind IN ({placeholders}) "
                "ORDER BY seqno"
            )
            args = (since_seqno, *sorted(kind_filter))
        out: list[VisionEvent] = []
        async with self._wal_db.execute(sql, args) as cur:
            async for row in cur:
                event_id, seqno, t_mono_ns, kind, severity, payload_json = row
                payload = json.loads(payload_json) if payload_json else {}
                out.append(
                    VisionEvent(
                        event_id=event_id,
                        seqno=int(seqno),
                        t_mono_ns=int(t_mono_ns),
                        kind=VisionEventType(kind),
                        severity=Severity(severity),
                        payload=payload,
                    )
                )
        return out

    def get_event_history(
        self,
        since_seqno: int = -1,
        limit: int = 100,
    ) -> list[VisionEvent]:
        """Synchronous read of recent events from the in-memory ring buffer.

        Used by the GUI for live tabular display — does *not* hit the WAL.
        The ring is bounded by `history_size` (constructor arg, default 100);
        older events are not retrievable here — call `catchup()` instead.

        Returns events with seqno > since_seqno, chronological order, at most
        `limit` items (truncated newest-first if exceeded).
        """
        if limit <= 0:
            return []
        with self._event_history_lock:
            snapshot = list(self._event_history)
        matched = [ev for ev in snapshot if ev.seqno > since_seqno]
        if len(matched) > limit:
            matched = matched[-limit:]
        return matched

    def get_stats(self) -> dict[str, int]:
        """Snapshot of internal counters (events_published, events_dropped_oldest,
        wal_writes_failed, etc.). Safe to call from any thread."""
        with self._stats_lock:
            return dict(self._stats)

    # ─────────────────────────────────────────────────────────────────
    # Internal: edge detection + fanout
    # ─────────────────────────────────────────────────────────────────

    def _maybe_emit_quality_drop(self, ts: TipStatus) -> None:
        """Emit TIP_QUALITY_DROP only on rising edge good → degraded|bad."""
        prev = self._last_emitted_quality
        cur = ts.quality
        is_drop = (
            (prev in (None, TipQuality.GOOD, TipQuality.UNKNOWN))
            and (cur in (TipQuality.DEGRADED, TipQuality.BAD))
        )

        # ── SAFE-mode safety net (2026-08-01) ──
        # In SAFE every tip verdict is rewritten to "good" at its producer
        # (VisionModule.assess_tip_coarse → scan_monitor._publish_coarse is the
        # only path that writes a TipStatus), so reaching here in SAFE means some
        # producer bypassed the override. Do not escalate — a CRITICAL here halts
        # the running composite and raises a HITL interrupt, precisely what SAFE
        # promises not to do — but SAY SO loudly: this branch firing is the
        # designed signal that the override chain has a hole.
        suppressed = is_drop and _safe_mode_active_quiet()

        # The latch records what the alarm chain last ACTED on. A suppressed drop
        # was not acted on, so it must not move the latch: parking it at BAD would
        # make the first genuine bad tip after switching back to auto a bad→bad
        # transition — no rising edge, no event, no alarm. (Same reasoning that
        # makes SAFE write "good" to the store in the first place; this branch is
        # the one path that could still have poisoned it.)
        if not suppressed:
            self._last_emitted_quality = cur
        if not is_drop:
            return

        severity = Severity.CRITICAL if cur == TipQuality.BAD else Severity.WARN
        # ``source`` —— 谁判的。同一个 kind 下有两个判定方(电流监控 vs 视觉),
        # 处方、可信度、以及 ``runtime.tip_halt_source`` 给它们的中止权都不同。
        # 不写这个字段,``ReadHardwareEvents`` 的摘要就印成「来源 ?」——
        # 缺少来源的摘要不足以支持 agent 决策。
        payload = {"quality": cur.value, "confidence": ts.confidence,
                   "scan_id": ts.scan_id, "frame_idx": ts.frame_idx,
                   "source": "vision_tip_status"}
        if suppressed:
            logger.warning(
                "SAFE 模式下仍收到 %s TipStatus（seqno=%d, scan_id=%s）——针尖判定覆写链有漏，"
                "已降级为 WARN 不中止实验。请检查是否有绕过 VisionModule 门面的判定产出方。",
                cur.value, ts.seqno, ts.scan_id)
            severity = Severity.WARN
            payload["safe_mode_suppressed"] = True

        ev = VisionEvent(
            seqno=ts.seqno,
            kind=VisionEventType.TIP_QUALITY_DROP,
            severity=severity,
            payload=payload,
            cause_ref=f"tip_status#{ts.seqno}",
        )
        self.emit_event(ev)

    def _fanout_to_queues(
        self, queues: list[asyncio.Queue], ev: VisionEvent
    ) -> None:
        """Runs on the asyncio loop. put_nowait to each queue; drop-oldest on full."""
        for q in queues:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # Drop oldest, then push new (slow-subscriber policy)
                try:
                    q.get_nowait()
                    with self._stats_lock:
                        self._stats["events_dropped_oldest"] += 1
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    logger.warning("Subscriber queue full even after drop-oldest; dropping event")
                    with self._stats_lock:
                        self._stats["events_fanout_failed"] += 1

    # ─────────────────────────────────────────────────────────────────
    # WAL helpers (run on asyncio loop, awaited from threadsafe schedule)
    # ─────────────────────────────────────────────────────────────────

    async def _wal_append_tip(self, ts: TipStatus) -> None:
        if self._wal_db is None:
            return
        try:
            await self._wal_db.execute(
                "INSERT OR REPLACE INTO tip_status_journal "
                "(seqno, t_mono_ns, quality, confidence, scan_id, frame_idx, safe_mode) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts.seqno, ts.t_mono_ns, ts.quality.value, ts.confidence,
                 ts.scan_id, ts.frame_idx, int(bool(ts.safe_mode))),
            )
            await self._wal_db.commit()
            with self._stats_lock:
                self._stats["wal_tip_writes"] += 1
        except Exception as e:
            logger.warning("WAL append failed for tip seqno=%d: %s", ts.seqno, e)
            with self._stats_lock:
                self._stats["wal_writes_failed"] += 1

    async def _wal_append_event(self, ev: VisionEvent) -> None:
        if self._wal_db is None:
            return
        try:
            import json
            await self._wal_db.execute(
                "INSERT OR REPLACE INTO event_journal "
                "(event_id, seqno, t_mono_ns, kind, severity, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ev.event_id, ev.seqno, ev.t_mono_ns, ev.kind.value,
                 ev.severity.value, json.dumps(ev.payload)),
            )
            await self._wal_db.commit()
            with self._stats_lock:
                self._stats["wal_event_writes"] += 1
        except Exception as e:
            logger.warning("WAL append failed for event %s: %s", ev.event_id, e)
            with self._stats_lock:
                self._stats["wal_writes_failed"] += 1


__all__ = ["BufferService", "_LatestSlot"]
