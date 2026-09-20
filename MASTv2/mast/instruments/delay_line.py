"""Pump-probe delay line: stage position ↔ optical delay conversion.

Physics: a retroreflector on a linear stage changes the optical path by
**twice** the stage travel, so 1 mm of stage motion is

    Δt = 2 mm / c = 2e-3 / 299792458 s ≈ 6.6713 ps

``ps_per_mm`` is configurable for double-pass geometries (4×) or direct
single-pass paths (3.3356). ``zero_offset_mm`` is the calibrated stage
position of pump-probe overlap (Δt = 0); ``sign`` flips which travel
direction means "probe later".

All limit safety stays in the underlying :class:`MotionAxis` — this
wrapper only converts units, so a delay outside the stage's travel raises
the axis's :class:`TravelLimitError` with the native positions in the
message, plus a pre-flight range hint from :meth:`delay_range_ps`.
"""

from __future__ import annotations

from dataclasses import dataclass

from mast.instruments.base import AxisStatus, MotionAxis

__all__ = ["PS_PER_MM_RETRO", "DelayLineConfig", "DelayLine"]

#: optical delay per stage mm for the standard retroreflector (round trip)
PS_PER_MM_RETRO = 6.671281904


@dataclass(frozen=True)
class DelayLineConfig:
    device_id: str
    axis: str
    unit_per_mm: float = 1000.0    # axis native units per mm (µm axes: 1000)
    ps_per_mm: float = PS_PER_MM_RETRO
    zero_offset_mm: float = 0.0
    sign: int = 1                  # +1: larger stage pos ⇒ later probe

    def __post_init__(self) -> None:
        if self.unit_per_mm <= 0:
            raise ValueError("unit_per_mm must be > 0")
        if self.ps_per_mm <= 0:
            raise ValueError("ps_per_mm must be > 0")
        if self.sign not in (1, -1):
            raise ValueError("sign must be +1 or -1")

    @classmethod
    def from_dict(cls, raw: dict) -> "DelayLineConfig":
        return cls(
            device_id=str(raw["device_id"]),
            axis=str(raw["axis"]),
            unit_per_mm=float(raw.get("unit_per_mm", 1000.0)),
            ps_per_mm=float(raw.get("ps_per_mm", PS_PER_MM_RETRO)),
            zero_offset_mm=float(raw.get("zero_offset_mm", 0.0)),
            sign=int(raw.get("sign", 1)),
        )


class DelayLine:
    """High-level pump-probe delay handle over a configured MotionAxis."""

    def __init__(self, axis: MotionAxis, config: DelayLineConfig):
        self._axis = axis
        self.config = config

    # -- conversions -----------------------------------------------------------

    def delay_to_position(self, delay_ps: float) -> float:
        """Optical delay (ps) → axis native position."""
        c = self.config
        mm = c.zero_offset_mm + c.sign * (delay_ps / c.ps_per_mm)
        return mm * c.unit_per_mm

    def position_to_delay(self, position: float) -> float:
        """Axis native position → optical delay (ps)."""
        c = self.config
        mm = position / c.unit_per_mm
        return c.sign * (mm - c.zero_offset_mm) * c.ps_per_mm

    @property
    def delay_range_ps(self) -> tuple[float, float]:
        """Reachable delay window implied by the axis soft limits."""
        lo = self.position_to_delay(self._axis.config.min_pos)
        hi = self.position_to_delay(self._axis.config.max_pos)
        return (min(lo, hi), max(lo, hi))

    # -- motion ----------------------------------------------------------------

    def move_to_delay_ps(
        self, delay_ps: float, *, wait: bool = True, timeout: float | None = None
    ) -> AxisStatus:
        return self._axis.move_abs(
            self.delay_to_position(delay_ps), wait=wait, timeout=timeout
        )

    def get_delay_ps(self) -> float:
        return self.position_to_delay(self._axis.get_position())

    def home(self, *, wait: bool = True, timeout: float | None = None) -> AxisStatus:
        return self._axis.home(wait=wait, timeout=timeout)

    def stop(self) -> None:
        self._axis.stop()

    @property
    def axis(self) -> MotionAxis:
        return self._axis
