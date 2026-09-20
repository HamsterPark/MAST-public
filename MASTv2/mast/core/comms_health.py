"""Communication health circuit breaker for the Nanonis TCP link.

WHY (field trace, s82–130): when the Nanonis TCP link goes down, every
hardware read/write times out ON ITS OWN (5 s each) before the agent gives up.
The trace shows GetBias → GetCurrent → GetZPosition → SetBias → ZControllerOnOff
each raising ``TimeoutError`` — nineteen separate ~5 s stalls, one tool at a time,
before the agent finally handed off. That is ~90 s of the operator watching the
system bang its head against a dead socket, and it hammers the *fragile* Nanonis
port (force-kill / repeated abrupt reconnects can wedge it until Nanonis restarts).

This module is the circuit breaker that stops that march. It observes only
TCP-level failures (timeouts / connection errors / failed reconnects — NOT a
Nanonis application error string, which means the round-trip SUCCEEDED). After
``fail_threshold`` consecutive TCP failures it OPENS: subsequent calls are
short-circuited *without touching the socket* for a cooldown, so the caller gets
an immediate, clear "comms down" verdict instead of a fresh 5 s stall, and the
port is left alone. After the cooldown one probe is allowed (HALF-OPEN); its
result either CLOSES the breaker (Nanonis recovered) or re-OPENS it.

SAFETY: the breaker never issues a command. It only decides whether a command is
allowed to be *attempted*. Its whole job is to make the system STOP sending and
report, which is exactly the safe thing to do when the link is dead — a stop /
withdraw cannot reach a Nanonis that is not answering TCP anyway, so failing fast
is strictly better than a per-command 5 s hang.

Process-level, thread-safe, holds only primitives (no sockets / tensors) — safe
to import anywhere in the core layer. One breaker lives on each
:class:`~mast.core.connection.ConnectionPool`.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

# Defaults (module constants so the whole system agrees; overridable per-instance
# for tests). Kept conservative on the Nanonis-port-fragility side: a short
# threshold trips fast, and the cooldown is long enough that we are not
# re-probing a dead port every few hundred ms.
_FAIL_THRESHOLD = 3          # consecutive TCP failures that OPEN the breaker
_OPEN_COOLDOWN_S = 20.0      # how long the breaker stays open before a probe
_STREAK_WINDOW_S = 30.0      # a failure older than this starts a fresh streak

# States
CLOSED = "closed"        # healthy — calls flow normally
OPEN = "open"            # tripped — calls short-circuited, port left alone
HALF_OPEN = "half_open"  # cooldown elapsed — exactly one probe in flight


class CommsCircuitBreaker:
    """Consecutive-TCP-failure circuit breaker.

    The pool consults :meth:`allow` before every ``safe_call`` and reports the
    outcome via :meth:`record_success` / :meth:`record_failure`. ``record_*`` is
    driven ONLY by TCP-level outcomes — a Nanonis app-error string is a
    successful round-trip and counts as a success, so a run that legitimately
    pokes an unavailable module does not trip the link breaker.
    """

    def __init__(
        self,
        *,
        fail_threshold: int = _FAIL_THRESHOLD,
        open_cooldown_s: float = _OPEN_COOLDOWN_S,
        streak_window_s: float = _STREAK_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fail_threshold = max(1, int(fail_threshold))
        self._open_cooldown_s = float(open_cooldown_s)
        self._streak_window_s = float(streak_window_s)
        self._clock = clock
        self._lock = threading.RLock()
        # State
        self._streak = 0                # consecutive TCP failures
        self._last_fail_at = 0.0
        self._open_until = 0.0           # monotonic deadline while OPEN
        self._probe_in_flight = False    # HALF-OPEN: one probe allowed at a time
        self._tripped_total = 0          # how many times we have OPENed (telemetry)
        self._last_reason = ""

    # ── state introspection ──────────────────────────────────────────────
    def _state_locked(self) -> str:
        if self._streak < self._fail_threshold:
            return CLOSED
        if self._probe_in_flight:
            return HALF_OPEN
        if self._clock() >= self._open_until:
            return HALF_OPEN  # cooldown elapsed — the next allow() probes
        return OPEN

    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def is_open(self) -> bool:
        with self._lock:
            return self._state_locked() == OPEN

    # ── the gate ─────────────────────────────────────────────────────────
    def allow(self) -> bool:
        """True if a call may be ATTEMPTED right now.

        CLOSED → always True. OPEN → False until the cooldown elapses, then
        exactly one probe is let through (the breaker goes HALF-OPEN with a
        probe outstanding; further calls are refused until that probe reports
        back via record_success / record_failure).
        """
        with self._lock:
            if self._streak < self._fail_threshold:
                return True  # CLOSED
            if self._probe_in_flight:
                return False  # a probe is already out — hold everyone else
            if self._clock() >= self._open_until:
                # Cooldown elapsed: release ONE probe.
                self._probe_in_flight = True
                return True
            return False  # still OPEN, inside cooldown

    def record_success(self) -> None:
        """A TCP round-trip completed (even if Nanonis returned an app error).
        Resets the streak and closes the breaker."""
        with self._lock:
            self._streak = 0
            self._last_fail_at = 0.0
            self._open_until = 0.0
            self._probe_in_flight = False
            self._last_reason = ""

    def record_failure(self, reason: str = "") -> None:
        """A TCP-level failure (timeout / connection error / failed reconnect).

        Extends the consecutive-failure streak (a stale failure older than the
        window starts a fresh streak rather than counting toward an unrelated
        one) and, at/over the threshold, (re)arms the OPEN cooldown."""
        with self._lock:
            now = self._clock()
            if self._streak and (now - self._last_fail_at) > self._streak_window_s:
                # The previous failures are too old to be "consecutive" with this
                # one — start the streak over so a slow drip never trips it.
                self._streak = 1
            else:
                self._streak += 1
            self._last_fail_at = now
            self._last_reason = str(reason or "")[:200]
            was_probe = self._probe_in_flight
            self._probe_in_flight = False
            if self._streak >= self._fail_threshold:
                # (Re)open. A failed HALF-OPEN probe re-arms the full cooldown.
                if self._open_until <= now or was_probe:
                    self._tripped_total += 1
                self._open_until = now + self._open_cooldown_s

    def cooldown_remaining_s(self) -> float:
        with self._lock:
            if self._streak < self._fail_threshold:
                return 0.0
            return max(0.0, self._open_until - self._clock())

    def snapshot(self) -> dict:
        """Plain-dict telemetry (JSON-safe) for diagnostics / GUI."""
        with self._lock:
            return {
                "state": self._state_locked(),
                "streak": self._streak,
                "fail_threshold": self._fail_threshold,
                "cooldown_remaining_s": round(
                    max(0.0, self._open_until - self._clock()), 2)
                if self._streak >= self._fail_threshold else 0.0,
                "tripped_total": self._tripped_total,
                "last_reason": self._last_reason,
            }


# Operator-facing message returned in a short-circuited call's record.error. The
# agent reads this on the tool path — it must say plainly "stop retrying, hand
# off", because the whole point is to end the one-tool-at-a-time march.
COMMS_DOWN_MESSAGE = (
    "comms_circuit_open: Nanonis TCP 通信连续超时(≥{n} 次)已熔断,暂停发命令 "
    "~{cd:.0f}s 以免反复重连撞坏脆弱的 Nanonis 端口。通信已判定中断——请勿逐个工具"
    "重试,应立即停止硬件操作、向用户/上层报告并 handoff,等待连接恢复后再继续。"
)


def format_comms_down(breaker: "CommsCircuitBreaker") -> str:
    snap = breaker.snapshot()
    return COMMS_DOWN_MESSAGE.format(
        n=snap.get("fail_threshold", _FAIL_THRESHOLD),
        cd=snap.get("cooldown_remaining_s", _OPEN_COOLDOWN_S) or _OPEN_COOLDOWN_S,
    )


__all__ = [
    "CommsCircuitBreaker",
    "COMMS_DOWN_MESSAGE",
    "format_comms_down",
    "CLOSED",
    "OPEN",
    "HALF_OPEN",
]
