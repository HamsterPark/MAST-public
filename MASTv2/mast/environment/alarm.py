"""Over-limit alarm evaluation for environment sensors.

A single :class:`AlarmSpec` expresses both directions so it works for a vacuum
gauge (pressure must stay *below* a ceiling — a rising pressure is a degraded
vacuum = alarm) and a thermometer (temperature must stay *within* a band).

Status precedence (worst wins): ``alarm`` > ``warning`` > base. The hard
bounds (``max`` / ``min``) trip ``alarm``; the soft bounds (``warn_max`` /
``warn_min``) trip ``warning``. Any bound left as ``None`` is simply not
checked, so a spec with only ``max`` set is the common "vacuum ceiling" case.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlarmSpec:
    """Threshold band for one sensor. All bounds optional."""

    max: float | None = None        # value > max      -> "alarm"
    min: float | None = None        # value < min      -> "alarm"
    warn_max: float | None = None   # value > warn_max -> "warning"
    warn_min: float | None = None   # value < warn_min -> "warning"

    def is_empty(self) -> bool:
        return all(b is None for b in (self.max, self.min, self.warn_max, self.warn_min))

    def evaluate(self, value: float, base_status: str = "ok") -> str:
        """Return the worst applicable status for *value*.

        ``base_status`` is returned when no threshold is crossed (lets a sensor
        that already reported ``error`` / ``unavailable`` keep that status).
        """
        if base_status in ("error", "unavailable"):
            return base_status
        try:
            v = float(value)
        except (TypeError, ValueError):
            return base_status
        if self.max is not None and v > self.max:
            return "alarm"
        if self.min is not None and v < self.min:
            return "alarm"
        if self.warn_max is not None and v > self.warn_max:
            return "warning"
        if self.warn_min is not None and v < self.warn_min:
            return "warning"
        return base_status

    @classmethod
    def from_dict(cls, d: dict | None) -> "AlarmSpec":
        if not d:
            return cls()
        def _f(k):
            val = d.get(k)
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None
        return cls(max=_f("max"), min=_f("min"),
                   warn_max=_f("warn_max"), warn_min=_f("warn_min"))

    def to_dict(self) -> dict:
        out: dict = {}
        for k in ("max", "min", "warn_max", "warn_min"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out


# Worst-wins ordering used by EnvironmentMonitor when aggregating sensors.
_SEVERITY_ORDER = {"ok": 0, "unavailable": 1, "warning": 2, "error": 3, "alarm": 4}


def worst_status(*statuses: str) -> str:
    """Return the highest-severity status among *statuses* (alarm worst)."""
    worst = "ok"
    for s in statuses:
        if _SEVERITY_ORDER.get(s, 0) > _SEVERITY_ORDER.get(worst, 0):
            worst = s
    return worst


__all__ = ["AlarmSpec", "worst_status"]
