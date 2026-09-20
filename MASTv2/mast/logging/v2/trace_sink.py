"""Fire-and-forget TraceSink for agent training/usage trajectories.

RFC: ``docs/v2/design/agent_training_log_rfc.md`` §2.

The agent graph (skill_adapter / orchestrator / safety_mw / HITL) records steps
through this sink. Calls MUST NOT block the graph: every write is enqueued and
performed on a single background daemon thread (``queue.put_nowait`` — drops on
overload rather than blocking, so a slow/locked DB can never stall a node). IDs
are minted synchronously (ULID) and returned immediately, so a caller can
reference a trajectory/step id before its row is actually written.

Injected by the GUI at ``build()`` time (like ``instrument_post_hook``); the
agents never import this module — they receive a recorder callback. This keeps
the v2 invariants: *graph.py does not block* and *no cross-agent import*.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Protocol

from mast.logging.v2.ulid import ulid_now

logger = logging.getLogger(__name__)

# Re-log a repeating worker failure at most this often (the first one is always
# logged, with its traceback).
_FAIL_QUIET_S = 60.0


class TraceSink(Protocol):
    """The recorder interface injected into the agent layer."""

    def begin_trajectory(self, *, thread_id: str, **kw: Any) -> str: ...
    def record_step(self, *, trajectory_id: str, step_type: str, **kw: Any) -> str: ...
    def end_trajectory(self, trajectory_id: str, **kw: Any) -> None: ...
    def set_quality(self, trajectory_id: str, quality: dict) -> None: ...


class NullTraceSink:
    """No-op sink (training log disabled). Still mints ids so callers never break."""

    def begin_trajectory(self, *, thread_id: str, **kw: Any) -> str:
        return ulid_now()

    def record_step(self, *, trajectory_id: str, step_type: str, **kw: Any) -> str:
        return ulid_now()

    def end_trajectory(self, trajectory_id: str, **kw: Any) -> None:
        pass

    def set_quality(self, trajectory_id: str, quality: dict) -> None:
        pass


class QueuedTraceSink:
    """Never-blocking TraceSink: writes go to a single background daemon thread.

    ``begin_trajectory`` / ``record_step`` mint a ULID synchronously and return
    it immediately; the actual INSERT runs on the worker. The worker is FIFO, so
    a trajectory is always written before the steps that reference it.
    """

    def __init__(self, repos: Any, *, maxsize: int = 10000):
        self._repos = repos
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = object()
        self.dropped = 0            # queue overflow
        self.failed = 0             # writes the DB rejected
        self.orphan_dropped = 0     # steps skipped because their trajectory is dead
        self.last_error: str | None = None
        # Trajectories whose INSERT was rejected. Touched ONLY from the worker
        # thread (both the add and the read happen inside worker-run closures),
        # so it needs no lock.
        self._dead: set[str] = set()
        self._fail_log: dict[str, list] = {}
        self._worker = threading.Thread(
            target=self._run, name="mast-trace-sink", daemon=True)
        self._worker.start()

    def _enqueue(self, label: str, fn) -> None:
        try:
            self._q.put_nowait((label, fn))
        except queue.Full:
            self.dropped += 1  # NEVER block the agent graph — drop on overload
            self._note_failure(label + ":overflow", None,
                               "trace sink queue full — dropping %s" % label)

    def _note_failure(self, key: str, exc: BaseException | None, msg: str) -> None:
        """Log a worker failure LOUDLY the first time, then at most once a minute
        with a running count.

        Deliberately not `logger.debug`: this sink swallowed a 100%-failure rate
        (every INSERT hit a FOREIGN KEY constraint) for a full day and the
        service log carried not one line about it, so the agent training log
        looked merely 'empty' instead of broken.
        Swallowing is about not crashing the graph — never about being silent.
        """
        st = self._fail_log.setdefault(key, [0, 0.0])
        st[0] += 1
        now = time.monotonic()
        if st[0] == 1:
            st[1] = now
            logger.warning("%s — trajectory logging is LOSING DATA%s", msg,
                           ("" if exc is None else ": %r" % (exc,)),
                           exc_info=exc is not None)
        elif now - st[1] >= _FAIL_QUIET_S:
            st[1] = now
            logger.warning("%s — still failing (%d times so far)%s", msg, st[0],
                           ("" if exc is None else ": %r" % (exc,)))

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is self._stop:
                    return
                label, fn = item
                try:
                    fn()
                except Exception as exc:  # must never crash the worker
                    self.failed += 1
                    self.last_error = "%s: %s: %s" % (label, type(exc).__name__, exc)
                    self._note_failure("%s:%s" % (label, type(exc).__name__), exc,
                                       "trace sink %s failed" % label)
            finally:
                self._q.task_done()

    # ── public API — id minted sync, write deferred to the worker ──
    def begin_trajectory(self, *, thread_id: str, operator_intent: dict | None = None,
                         context_snapshot: dict | None = None,
                         experiment_id: str | None = None,
                         campaign_id: str | None = None,
                         sample_id: str | None = None) -> str:
        tid = ulid_now()

        def _op() -> None:
            try:
                self._repos.trajectories.begin(
                    thread_id=thread_id, operator_intent=operator_intent,
                    context_snapshot=context_snapshot, experiment_id=experiment_id,
                    campaign_id=campaign_id, sample_id=sample_id, trajectory_id=tid)
            except Exception:
                # The id was already handed to the caller, so its steps are
                # coming. Remember that this trajectory does not exist, so the
                # steps are dropped as orphans (counted, once) instead of each
                # raising its own foreign-key error.
                self._dead.add(tid)
                raise

        self._enqueue("begin_trajectory", _op)
        return tid

    def record_step(self, *, trajectory_id: str, step_type: str, **kw: Any) -> str:
        sid = ulid_now()

        def _op() -> None:
            if trajectory_id in self._dead:
                self.orphan_dropped += 1
                self._note_failure("orphan_step", None,
                                   "trace sink dropping steps for trajectory %s "
                                   "(its INSERT was rejected)" % trajectory_id)
                return
            self._repos.steps.record(
                trajectory_id=trajectory_id, step_type=step_type, step_id=sid, **kw)

        self._enqueue("record_step", _op)
        return sid

    def end_trajectory(self, trajectory_id: str, *, exit_status: str | None = None,
                       final_outcome: dict | None = None) -> None:
        def _op() -> None:
            if trajectory_id in self._dead:
                return          # already reported by the begin failure
            self._repos.trajectories.end(
                trajectory_id, exit_status=exit_status, final_outcome=final_outcome)

        self._enqueue("end_trajectory", _op)

    def set_quality(self, trajectory_id: str, quality: dict) -> None:
        self._enqueue("set_quality", lambda: self._repos.trajectories.set_quality(
            trajectory_id, quality))

    def stats(self) -> dict:
        """Observable health of the sink. ``failed``/``orphan_dropped`` > 0 means
        trajectory rows are being lost right now."""
        return {"dropped": self.dropped, "failed": self.failed,
                "orphan_dropped": self.orphan_dropped,
                "dead_trajectories": len(self._dead),
                "last_error": self.last_error}

    # ── lifecycle — tests / shutdown only, NEVER from the agent graph ──
    def flush(self) -> None:
        """Block until the queue drains. For tests / graceful shutdown only."""
        self._q.join()

    def close(self) -> None:
        try:
            self._q.put_nowait(self._stop)
        except queue.Full:
            self.dropped += 1
        self._worker.join(timeout=5.0)
