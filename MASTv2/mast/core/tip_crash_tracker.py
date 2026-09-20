"""Same-region tip-crash tracker — the state machine that stops in-place spinning.

WHY (field trace, s289–323): after a tip crash the agent looped
PreScanCheck → FullScan (NaN) → ConditionTip → TipPulse at the SAME XY point for
~5 minutes. Conditioning a tip in place does not fix a spot that keeps crashing it
(debris / a step edge / a contaminated patch): the answer is to LEAVE — withdraw
and coarse-move to a fresh region — not to pulse harder at the same coordinates.
Nothing tracked "this location has already crashed the tip twice", so nothing
could refuse the third attempt, and the run burned time going nowhere.

This process-level tracker closes that gap. FullScan's crash detector RECORDS a
crash at its scan centre; the scan / conditioning composites CONSULT it before
they start, and once a region has crashed the tip ``block_threshold`` times
(default 2) they refuse to scan/condition there again and tell the agent to
escape (withdraw + coarse lateral move). A coarse lateral move (the "换区" action)
clears the blocks — that IS the recovery — and a stale crash expires after a TTL
so an overnight run can never wedge permanently.

Holds only primitives (quantised int cell keys + counts + timestamps); no tensors,
no hardware handles. Thread-safe, importable anywhere in the core layer.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

# Two crashes at one spot is the signal to move (team directive "同点连续 crash≥2 次").
_BLOCK_THRESHOLD = 2
# "Same region" tolerance. The field loop re-scanned the identical centre, but a
# few-nm jitter should still count as the same bad spot — a test scan is ~10 nm, so
# 8 nm cells collapse "essentially the same place" together without swallowing a
# genuine relocation.
_TOL_M = 8e-9
# A crash older than this no longer blocks — a deliberate escape (coarse move)
# clears immediately, but this guarantees an overnight run self-heals even if the
# agent never issues one. Mirrors the 30-min staleness window used elsewhere.
_TTL_S = 1800.0


class TipCrashTracker:
    """Counts tip crashes per quantised XY region; blocks a region at threshold."""

    def __init__(
        self,
        *,
        block_threshold: int = _BLOCK_THRESHOLD,
        tol_m: float = _TOL_M,
        ttl_s: float = _TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._block_threshold = max(1, int(block_threshold))
        self._tol_m = float(tol_m)
        self._ttl_s = float(ttl_s)
        self._clock = clock
        self._lock = threading.RLock()
        # cell -> {"count": int, "last": monotonic}
        self._cells: dict[tuple, dict] = {}

    # ── keying ───────────────────────────────────────────────────────────
    def _cell(self, x_m, y_m) -> tuple:
        """Quantise a position to a cell key. Unknown coords collapse to one
        sentinel cell so repeated crashes at an unknown position still count."""
        if x_m is None or y_m is None:
            return ("?", "?")
        try:
            return (round(float(x_m) / self._tol_m), round(float(y_m) / self._tol_m))
        except (TypeError, ValueError):
            return ("?", "?")

    def _prune(self, now: float) -> None:
        dead = [c for c, r in self._cells.items()
                if (now - r.get("last", 0.0)) > self._ttl_s]
        for c in dead:
            self._cells.pop(c, None)

    # ── record / query ───────────────────────────────────────────────────
    def record_crash(self, x_m=None, y_m=None) -> int:
        """Record one crash at (x_m, y_m); return the region's running count."""
        cell = self._cell(x_m, y_m)
        now = self._clock()
        with self._lock:
            self._prune(now)
            rec = self._cells.setdefault(cell, {"count": 0, "last": 0.0})
            rec["count"] += 1
            rec["last"] = now
            return int(rec["count"])

    def crash_count(self, x_m=None, y_m=None) -> int:
        cell = self._cell(x_m, y_m)
        now = self._clock()
        with self._lock:
            self._prune(now)
            rec = self._cells.get(cell)
            return int(rec["count"]) if rec else 0

    def is_blocked(self, x_m=None, y_m=None) -> bool:
        """True once this region has crashed the tip ``block_threshold`` times —
        the caller must escape (withdraw + coarse move), not retry here."""
        return self.crash_count(x_m, y_m) >= self._block_threshold

    def crash_points(self) -> "tuple[list[tuple[float, float, int]], int]":
        """Every remembered crash as ``([(x_m, y_m, count), …], unlocated)``.

        WHY this exists at all: ``crash_count`` answers "have you crashed HERE",
        which only helps a caller that already has a point to ask about. The spot
        PICKER has the opposite problem — it is choosing the point, so it needs
        the crashes enumerated up front to keep out of them. Without this the
        tracker's knowledge was unreachable by anyone selecting a position, and
        the only crash history a picker could consult was the persisted map (see
        ``map_scope.crash_memory_markers``, the one caller).

        Coordinates are the CELL CENTRE, not the exact crash point: the tracker
        only ever stored a quantised key. The reconstruction is off by at most
        ``tol_m / 2`` (4 nm at the 8 nm default) — two orders below the crash
        avoidance radius that consumes these points, so it changes no decision.
        Deliberately NOT compensated with a padding term: the radius belongs to
        the avoidance model (``AnalysisConfig.crash_r_m``), and a second, private
        fudge here would be exactly the duplicate keep-out convention this repo
        keeps paying for.

        ``unlocated`` counts crashes recorded with no usable position (the
        ``("?", "?")`` sentinel — a crash whose scan centre could not be read).
        They are returned SEPARATELY and never as a point, because a keep-out
        circle at a guessed coordinate would be a fabricated fact. A caller that
        cannot avoid them geometrically must say so out loud rather than let
        "nothing to avoid" stand in for "we know it happened somewhere here".
        """
        now = self._clock()
        located: list[tuple[float, float, int]] = []
        unlocated = 0
        with self._lock:
            self._prune(now)
            for cell, rec in self._cells.items():
                count = int(rec.get("count", 0))
                if count <= 0:
                    continue
                if cell == ("?", "?"):
                    unlocated += count
                    continue
                try:
                    located.append((cell[0] * self._tol_m,
                                    cell[1] * self._tol_m, count))
                except (TypeError, ValueError):  # pragma: no cover — key shape
                    unlocated += count
        return located, unlocated

    # ── recovery ─────────────────────────────────────────────────────────
    def note_recovery(self, x_m=None, y_m=None) -> None:
        """Clear crash history — the escape happened.

        With coords: clear just that region (e.g. a clean scan proved the tip
        works there). Without coords: clear everything — a coarse lateral move is
        a deliberate "换区", after which every prior bad spot is behind us."""
        with self._lock:
            if x_m is None and y_m is None:
                self._cells.clear()
                return
            self._cells.pop(self._cell(x_m, y_m), None)

    def clear(self) -> None:
        with self._lock:
            self._cells.clear()

    def snapshot(self) -> dict:
        now = self._clock()
        with self._lock:
            self._prune(now)
            return {
                "block_threshold": self._block_threshold,
                "tracked_regions": len(self._cells),
                "blocked_regions": sum(
                    1 for r in self._cells.values()
                    if r["count"] >= self._block_threshold),
                "max_count": max((r["count"] for r in self._cells.values()),
                                 default=0),
            }


# ── process-level singleton ──────────────────────────────────────────────
_singleton_lock = threading.Lock()
_singleton: "TipCrashTracker | None" = None


def get_tip_crash_tracker() -> TipCrashTracker:
    """The shared tracker consulted by the scan / conditioning composites."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = TipCrashTracker()
    return _singleton


def reset_tip_crash_tracker() -> None:
    """Test / new-run helper — wipe the shared tracker's state."""
    get_tip_crash_tracker().clear()


# ── the escape directive ─────────────────────────────────────────────────
def crash_escape_message(count: int, x_m=None, y_m=None) -> str:
    """The operator/agent-facing instruction to leave a repeatedly-crashing spot."""
    where = ""
    if x_m is not None and y_m is not None:
        try:
            where = f"(≈{float(x_m):.3e}, {float(y_m):.3e} m)"
        except (TypeError, ValueError):
            where = ""
    return (
        f"repeated_crash_escape_required: 同一区域{where}已连续检测到针尖 crash "
        f"{count} 次。原地扫描/修针/打脉冲无效——停止在此处重试。请用 "
        f"**RelocateCoarseXY** 换到新区域(先用 get_coarse_map 看该往哪走多少步);"
        f"它会自己收压电、用粗动马达退针清障并逐级自检、核对真空与驱动电压、"
        f"分块移动时看着电流,最后重新进针。"
        f"**不要直接调 MotorMove 做横向移动** —— 它只检查压电是否收到顶(约 1 µm 余量,"
        f"而且读不到状态时会放行),这正是又一次撞针的来路。"
        f"切勿在同点继续 ConditionTip/TipPulse。"
    )


def crash_guard(context, x_m=None, y_m=None) -> "str | None":
    """Return an escape directive if this region is crash-blocked, else None.

    Composites call this at the top of run_composite. On a block it also writes a
    diagnostics breadcrumb so the "why did it refuse to scan" is recorded (the
    refusal-ledger contract), then the caller early-exits with the message.
    """
    tracker = get_tip_crash_tracker()
    if not tracker.is_blocked(x_m, y_m):
        return None
    count = tracker.crash_count(x_m, y_m)
    msg = crash_escape_message(count, x_m, y_m)
    try:
        from mast.core.diagnostics import record
        record("tip_crash", "repeated_same_region",
               "同区域连续 crash≥阈值,拒绝原地重试——需退针+粗动换区",
               count=count, x_m=x_m, y_m=y_m)
    except Exception:  # noqa: BLE001 — diagnostics never breaks a skill
        pass
    return msg


__all__ = [
    "TipCrashTracker",
    "get_tip_crash_tracker",
    "reset_tip_crash_tracker",
    "crash_escape_message",
    "crash_guard",
]
