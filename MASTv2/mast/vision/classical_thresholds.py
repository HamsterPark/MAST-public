"""Per-instrument tunable thresholds for the network-free classical vision tools.

Mirrors :mod:`mast.vision.thresholds` (the deployed model's valves): a process-
level live-read holder keeps the ACTIVE snapshot; the classical tools read it
lazily on every call, and the API/runtime layer writes it (startup hydration +
each ``POST /api/settings``). A change takes effect on the very next call — no
reload. The vision layer never imports settings; the wiring is one-way.

These control the *transparent classical good/bad verdict* (:mod:`tip_quality`)
and a couple of detector cut-offs — the interpretable knobs an operator retunes
per instrument/sample, exactly what the diagnostic recommended (the deployed
model's fused cut was miscalibrated and un-retunable). See
``docs/v2/benchmarks/vision_v25_diagnostic/``.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, fields, replace

# Keys surfaced in the 设置 UI (per-instrument classical tip-quality knobs).
EDITABLE_KEYS: tuple[str, ...] = (
    "tq_instability_max",   # fwd-bwd instability above this → unstable (bad)
    "tq_sharpness_min",     # FFT sharpness below this AND no lattice → no surface (bad)
    "tq_double_threshold",  # double-tip replica score (REPORTED only — no longer a bad rule)
    "tq_change_threshold",  # mid-scan tip-change calibrated z (v2); 0 = auto by scale
    "oscillation_threshold",  # feedback-ringing on/off-axis peak ratio
    "iz_barrier_min",       # I(z) apparent-barrier lower bound for "clean" (eV)
    "iz_barrier_max",       # I(z) apparent-barrier upper bound for "clean" (eV)
)

FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "tq_instability_max": (0.0, 1.0),
    "tq_sharpness_min": (0.0, 1000.0),
    "tq_double_threshold": (0.0, 2.0),
    "tq_change_threshold": (0.0, 100.0),
    "oscillation_threshold": (1.0, 1000.0),
    "iz_barrier_min": (0.0, 20.0),
    "iz_barrier_max": (0.0, 20.0),
}


@dataclass(frozen=True)
class ClassicalThresholds:
    """Immutable classical-tool thresholds; commission them for each instrument.

    Shipped values are algorithm defaults and do not establish performance on an
    installation. tq_change_threshold applies to null-calibrated z, not the old
    max-t statistic. Zero selects the scale-dependent default; persisted values
    from earlier versions must be interpreted in the statistic now in use.
    """
    tq_instability_max: float = 0.50
    tq_sharpness_min: float = 8.0
    tq_double_threshold: float = 0.18
    tq_change_threshold: float = 0.0
    oscillation_threshold: float = 3.0
    iz_barrier_min: float = 0.5
    iz_barrier_max: float = 8.0

    def to_mapping(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_mapping(cls, m: dict | None) -> "ClassicalThresholds":
        """Build from a (partial) mapping — tolerant of persisted settings.
        Unknown keys ignored, missing keys fall back to default, recognised
        numeric values clamped to :data:`FIELD_BOUNDS`."""
        if not m:
            return cls()
        known = {f.name for f in fields(cls)}
        clean: dict[str, float] = {}
        for k, v in m.items():
            if k in known and isinstance(v, (int, float)) and not isinstance(v, bool):
                lo, hi = FIELD_BOUNDS.get(k, (float("-inf"), float("inf")))
                clean[k] = float(min(hi, max(lo, float(v))))
        return replace(cls(), **clean)


_LOCK = threading.Lock()
_ACTIVE = ClassicalThresholds()


def get_classical_thresholds() -> ClassicalThresholds:
    """Return the active immutable snapshot (lock-free atomic reference read)."""
    return _ACTIVE


def set_classical_thresholds(m: "dict | ClassicalThresholds | None") -> ClassicalThresholds:
    """Swap the active snapshot. ``None`` / empty resets to defaults."""
    global _ACTIVE
    new = m if isinstance(m, ClassicalThresholds) else ClassicalThresholds.from_mapping(m)
    with _LOCK:
        _ACTIVE = new
    return new


__all__ = [
    "ClassicalThresholds",
    "get_classical_thresholds",
    "set_classical_thresholds",
    "EDITABLE_KEYS",
    "FIELD_BOUNDS",
]
