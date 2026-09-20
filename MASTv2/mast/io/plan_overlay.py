"""PlanOverlay — the live planning route drawn on the scan map.

Car-navigation analogy: a planning skill (or the operator) publishes a sequence
of positioned steps; the map draws them as a dashed route ahead. As real
operations execute, ``advance()`` consumes the matching pending step so the route
shrinks toward the destination — the operator watches the plan being carried out
in real time.

Process-global singleton (like ``logging.experiment_log`` set_active_log): the
agent-side meta-tool publishes into it, the GUI map reads from it. Thread-safe;
holds only plain MapMarker values (no hardware handles, never checkpointed).
"""
from __future__ import annotations

import threading
from dataclasses import replace

from mast.io.exp_map import MapMarker


class PlanOverlay:
    """Thread-safe holder for the current planned route (list of MapMarker)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: list[MapMarker] = []
        self._title: str = ""

    def set_plan(self, steps: list[MapMarker], *, title: str = "") -> None:
        """Replace the planned route. Every step is forced to status=planned,
        source=plan so it renders as a ghost regardless of how it was built.

        Copies each input marker (dataclasses.replace) instead of mutating it —
        the caller keeps ownership of its own objects ()."""
        cleaned: list[MapMarker] = []
        for s in steps or []:
            if s is None or not s.has_xy:
                continue
            cleaned.append(replace(
                s, status="planned", source="plan", kind=(s.kind or "plan")))
        with self._lock:
            self._steps = cleaned
            self._title = title or ""

    def clear(self) -> None:
        with self._lock:
            self._steps = []
            self._title = ""

    def snapshot(self) -> list[MapMarker]:
        """A shallow copy of the pending route (oldest/next-first)."""
        with self._lock:
            return list(self._steps)

    @property
    def title(self) -> str:
        with self._lock:
            return self._title

    def is_empty(self) -> bool:
        with self._lock:
            return not self._steps

    def advance(self, x_m: float, y_m: float, *, tol_m: float | None = None) -> bool:
        """Consume the nearest pending step within *tol_m* of (x_m, y_m).

        Called when a real operation lands at (x_m, y_m). Returns True if a step
        was consumed (route progressed). *tol_m* defaults to 8% of the route's
        spatial span (min 2 nm) so a step counts as "reached" when the operation
        is anywhere near it — like a nav app advancing once you reach the turn."""
        with self._lock:
            if not self._steps:
                return False
            if tol_m is None:
                xs = [s.x_m for s in self._steps]
                ys = [s.y_m for s in self._steps]
                span = max(max(xs) - min(xs), max(ys) - min(ys), 0.0)
                tol_m = max(span * 0.08, 2e-9)
            best_i = -1
            best_d2 = None
            for i, s in enumerate(self._steps):
                d2 = (s.x_m - x_m) ** 2 + (s.y_m - y_m) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            if best_i >= 0 and best_d2 is not None and best_d2 <= tol_m ** 2:
                self._steps.pop(best_i)
                return True
        return False


_OVERLAY = PlanOverlay()


def get_plan_overlay() -> PlanOverlay:
    """Return the process-global PlanOverlay singleton."""
    return _OVERLAY


__all__ = ["PlanOverlay", "get_plan_overlay"]
