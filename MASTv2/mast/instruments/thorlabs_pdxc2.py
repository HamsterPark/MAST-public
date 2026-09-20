"""Thorlabs PDXC2 piezo-inertia stage controller — via the Kinesis C DLL (ctypes).

The lab's Thorlabs stage is a **PDXC2** (``BenchtopPDXC2Control``, serial
112511920 in the pump-probe VIs), NOT a Benchtop Piezo (BPC) — a different
Kinesis device family with a different control model (piezo inertia "slip-stick"
motor, open-loop jog steps + closed-loop position). This driver calls the
documented C API in ``Thorlabs.MotionControl.Benchtop.Piezo.dll`` via ctypes
(no pythonnet), which is deterministic from the vendor header signatures
(``Thorlabs.MotionControl.Benchtop.Piezo.PDXC2.h``).

Closed-loop position flow (from the header)::

    TLI_BuildDeviceList → PDXC2_Open(serial) → PDXC2_StartPolling →
    PDXC2_Enable → PDXC2_SetPositionControlMode(PZ_CloseLoop)
    per move:  PDXC2_SetClosedLoopTarget(target) → PDXC2_MoveStart →
               poll PDXC2_GetPosition

VERIFY ON HARDWARE — three points that need the real stage (this driver cannot
be unit-tested end-to-end without it; everything else degrades gracefully):

  1. **Position unit.** ``PDXC2_SetClosedLoopTarget`` / ``GetPosition`` use
     int32 device units (Kinesis PDXC2 is typically nm). ``counts_per_unit``
     converts the axis native unit → device units; ``OpticalStageWiggle``'s
     ``scale_ratio`` confirms it in one nudge.
  2. **GetPosition convention.** Used here as the int-returning form
     (Kinesis standard). If the firmware exports only the ``int32*`` out-param
     form, flip :meth:`ThorlabsPDXC2Axis._get_position_raw`.
  3. **On-target.** The PDXC2 status word is not decoded here (bit map
     unverified), so a move reports settled and relies on ``AxisConfig.settle_s``
     for the dwell. Decode ``PDXC2_GetStatusBits`` for a true on-target once the
     bit meanings are confirmed.

Lazy like the rest of ``mast.instruments``: no ctypes load at import; a missing
Kinesis DLL / absent device surfaces as :class:`InstrumentUnavailable`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mast.instruments.base import (
    AxisConfig,
    AxisStatus,
    InstrumentError,
    InstrumentUnavailable,
    MotionAxis,
    MotionController,
)

logger = logging.getLogger(__name__)

__all__ = ["ThorlabsPDXC2Controller", "ThorlabsPDXC2Axis", "DEFAULT_KINESIS_DIR"]

DEFAULT_KINESIS_DIR = Path(r"C:\Program Files\Thorlabs\Kinesis")
_DLL_NAME = "Thorlabs.MotionControl.Benchtop.Piezo.dll"

# PZ_ControlModeTypes (Kinesis): Undefined=0, OpenLoop=1, CloseLoop=2,
# OpenLoopSmooth=3, CloseLoopSmooth=4.
_PZ_CLOSE_LOOP = 2


def _load_pdxc2_dll(dll_dir: Path):
    """Load the Kinesis Benchtop.Piezo DLL and set PDXC2 prototypes.

    Raises :class:`InstrumentUnavailable` if ctypes, the directory, or the DLL
    is missing — never a bare OSError."""
    try:
        import ctypes
    except Exception as exc:  # pragma: no cover - ctypes is stdlib
        raise InstrumentUnavailable(f"ctypes unavailable: {exc}") from exc

    if not dll_dir.is_dir():
        raise InstrumentUnavailable(
            f"Kinesis directory not found: {dll_dir} — install Thorlabs Kinesis "
            "or set dll_dir in the device config"
        )
    dll_path = dll_dir / _DLL_NAME
    if not dll_path.is_file():
        raise InstrumentUnavailable(f"Kinesis DLL missing: {dll_path}")

    # let the loader resolve sibling Kinesis DLLs (DeviceManager, etc.)
    try:
        os.add_dll_directory(str(dll_dir))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - non-Windows / already added
        pass

    try:
        lib = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        raise InstrumentUnavailable(f"failed to load {dll_path.name}: {exc}") from exc

    c = ctypes
    proto = {
        "TLI_BuildDeviceList": ([], c.c_short),
        "PDXC2_Open": ([c.c_char_p], c.c_short),
        "PDXC2_Close": ([c.c_char_p], None),
        "PDXC2_Enable": ([c.c_char_p], c.c_short),
        "PDXC2_Disable": ([c.c_char_p], c.c_short),
        "PDXC2_StartPolling": ([c.c_char_p, c.c_int], c.c_bool),
        "PDXC2_StopPolling": ([c.c_char_p], None),
        "PDXC2_SetPositionControlMode": ([c.c_char_p, c.c_short], c.c_short),
        "PDXC2_SetClosedLoopTarget": ([c.c_char_p, c.c_int], c.c_short),
        "PDXC2_MoveStart": ([c.c_char_p], c.c_short),
        "PDXC2_MoveStop": ([c.c_char_p], c.c_short),
        "PDXC2_RequestPosition": ([c.c_char_p], c.c_short),
        "PDXC2_GetPosition": ([c.c_char_p], c.c_int),
        "PDXC2_GetStatusBits": ([c.c_char_p], c.c_uint),
        "PDXC2_Home": ([c.c_char_p], c.c_short),
    }
    try:
        for name, (argtypes, restype) in proto.items():
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    except AttributeError as exc:
        raise InstrumentUnavailable(
            f"{dll_path.name} is missing a PDXC2 export ({exc}); wrong Kinesis "
            "version?"
        ) from exc
    return lib


class ThorlabsPDXC2Axis(MotionAxis):
    """The single PDXC2 stage axis. Native unit set by ``counts_per_unit``
    (device int32 units per axis unit)."""

    def __init__(self, config: AxisConfig, controller: "ThorlabsPDXC2Controller",
                 *, counts_per_unit: float = 1.0):
        super().__init__(config, controller)
        if counts_per_unit <= 0:
            raise ValueError("counts_per_unit must be > 0")
        self._pdxc: ThorlabsPDXC2Controller = controller
        self._cpu = float(counts_per_unit)

    def _move_abs_raw(self, target: float) -> None:
        lib, s = self._pdxc._lib_serial()
        counts = int(round(target * self._cpu))
        self._pdxc._check(lib.PDXC2_SetClosedLoopTarget(s, counts), "SetClosedLoopTarget")
        self._pdxc._check(lib.PDXC2_MoveStart(s), "MoveStart")

    def _get_position_raw(self) -> float:
        lib, s = self._pdxc._lib_serial()
        lib.PDXC2_RequestPosition(s)
        return int(lib.PDXC2_GetPosition(s)) / self._cpu

    def _get_status_raw(self) -> AxisStatus:
        # on_target=None → base wait_until_settled treats "not moving" as settled
        # and applies AxisConfig.settle_s. VERIFY: decode PDXC2_GetStatusBits.
        return AxisStatus(position=self._get_position_raw(), moving=False,
                          on_target=None)

    def _stop_raw(self) -> None:
        try:
            lib, s = self._pdxc._lib_serial()
            lib.PDXC2_MoveStop(s)
        except Exception:  # noqa: BLE001 - panic path must not raise
            pass

    def _home_raw(self) -> None:
        lib, s = self._pdxc._lib_serial()
        self._pdxc._check(lib.PDXC2_Home(s), "Home")


class ThorlabsPDXC2Controller(MotionController):
    """One PDXC2 controller, addressed by serial number (single stage)."""

    POLL_MS = 200

    def __init__(
        self,
        *,
        serial_no: str,
        axes: list[AxisConfig] | None = None,
        dll_dir: str | Path | None = None,
        counts_per_unit: float = 1.0,
        closed_loop: bool = True,
    ):
        super().__init__()
        if not serial_no:
            raise ValueError("ThorlabsPDXC2Controller needs the device serial_no")
        self._serial_str = str(serial_no)
        self._serial = self._serial_str.encode("ascii")
        self._axis_configs = list(axes or [])
        self._dll_dir = Path(dll_dir) if dll_dir else DEFAULT_KINESIS_DIR
        self._counts_per_unit = float(counts_per_unit)
        self._closed_loop = bool(closed_loop)
        self._lib = None

    # -- lifecycle -----------------------------------------------------------

    def _connect_raw(self) -> None:
        lib = _load_pdxc2_dll(self._dll_dir)
        if lib.TLI_BuildDeviceList() != 0:
            raise InstrumentUnavailable("TLI_BuildDeviceList failed")
        if lib.PDXC2_Open(self._serial) != 0:
            raise InstrumentUnavailable(
                f"PDXC2_Open({self._serial_str}) failed — device off / cable / "
                "another program holding it?"
            )
        try:
            lib.PDXC2_StartPolling(self._serial, self.POLL_MS)
            lib.PDXC2_Enable(self._serial)
            if self._closed_loop:
                lib.PDXC2_SetPositionControlMode(self._serial, _PZ_CLOSE_LOOP)
        except Exception as exc:
            try:
                lib.PDXC2_Close(self._serial)
            except Exception:  # noqa: BLE001
                pass
            raise InstrumentUnavailable(
                f"PDXC2 {self._serial_str}: init failed: {exc}"
            ) from exc
        self._lib = lib

    def _close_raw(self) -> None:
        lib = self._lib
        self._lib = None
        if lib is not None:
            for call in (
                lambda: lib.PDXC2_StopPolling(self._serial),
                lambda: lib.PDXC2_Disable(self._serial),
                lambda: lib.PDXC2_Close(self._serial),
            ):
                try:
                    call()
                except Exception:  # noqa: BLE001
                    pass

    def _build_axes(self) -> dict[str, MotionAxis]:
        return {
            cfg.name: ThorlabsPDXC2Axis(
                cfg, self, counts_per_unit=self._counts_per_unit
            )
            for cfg in self._axis_configs
        }

    # -- helpers -------------------------------------------------------------

    def _lib_serial(self):
        if self._lib is None:
            raise InstrumentUnavailable(f"PDXC2 {self._serial_str} not connected")
        return self._lib, self._serial

    def _check(self, code: int, op: str) -> None:
        if code != 0:
            raise InstrumentError(
                f"PDXC2 {self._serial_str}: {op} returned error code {code}"
            )
