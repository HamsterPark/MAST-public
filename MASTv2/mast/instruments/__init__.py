"""mast.instruments — optical-bench motion hardware drivers (TERS / THz).

New in 2026-07 (optics integration): Python-native drivers replacing the
lab's LabVIEW VIs for the pump-probe delay line and optical stages.

Layering:

- :mod:`mast.instruments.base` — ``MotionController`` / ``MotionAxis``
  abstractions with driver-level soft travel limits (Layer-0: enforced in
  the base class, never bypassable by callers or LLM output).
- :mod:`mast.instruments.pztc_nm003` — ZhuoJu PZTC nm003 closed-loop piezo
  stage controller (Modbus RTU over USB-RS422).
- :mod:`mast.instruments.pi_gcs` — Physik Instrumente GCS controllers
  (E-816 piezo, E-861 NEXACT stepper-piezo) over serial.
- :mod:`mast.instruments.thorlabs_kinesis` — Thorlabs Kinesis .NET stack
  (Benchtop Piezo et al.) via pythonnet, lazily imported.
- :mod:`mast.instruments.delay_line` — pump-probe delay line wrapper
  (stage mm ↔ optical delay ps).
- :mod:`mast.instruments.registry` — declarative device inventory from
  config; the single lookup point used by skills.

Design constraints (inherited from mast.environment):

1. **No-hardware must not crash.** Importing this package, constructing a
   driver, or querying the registry on a machine with no serial ports, no
   pythonnet and no devices must never raise at import/construct time —
   only on actual use, and then with :class:`InstrumentUnavailable`.
2. **Testable without hardware.** Serial drivers speak through the
   ``SerialLike`` protocol (mast.environment.serial_transport) so tests
   inject fake transports with canned frames.
"""

from mast.instruments.base import (
    AxisConfig,
    AxisStatus,
    InstrumentError,
    InstrumentUnavailable,
    MotionAxis,
    MotionController,
    MotionTimeout,
    TravelLimitError,
)
from mast.instruments.delay_line import DelayLine, DelayLineConfig
from mast.instruments.registry import (
    InstrumentRegistry,
    get_instrument_registry,
    reset_instrument_registry,
)

__all__ = [
    "AxisConfig",
    "AxisStatus",
    "DelayLine",
    "DelayLineConfig",
    "InstrumentError",
    "InstrumentRegistry",
    "InstrumentUnavailable",
    "MotionAxis",
    "MotionController",
    "MotionTimeout",
    "TravelLimitError",
    "get_instrument_registry",
    "reset_instrument_registry",
]
