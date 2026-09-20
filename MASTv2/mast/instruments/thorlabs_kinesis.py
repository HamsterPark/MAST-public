"""Thorlabs Kinesis motion hardware via the .NET CLI assemblies.

Replaces ``Stage control.vi``, which drives a **Benchtop Piezo** (BPC
series) through ``Thorlabs.MotionControl.Benchtop.PiezoCLI.dll``.

Loading strategy — everything is LAZY:

- No ``clr`` / ``System`` import at module level. The .NET runtime is
  only bootstrapped inside :meth:`ThorlabsBenchtopPiezo._connect_raw`.
- Missing pythonnet, missing Kinesis DLLs, or no device present all
  surface as :class:`InstrumentUnavailable` at connect time; importing
  this module and constructing the driver never raises.
- DLL directory is configurable; defaults to the standard Kinesis
  install path. (Do NOT point it at a random DLL copy — Kinesis
  assemblies resolve siblings from their own directory.)

Packaging note (PyInstaller): pythonnet + Kinesis DLLs are NOT bundled;
this driver requires the Kinesis software installed on the host, which is
the same requirement the LabVIEW VI had.

Other Kinesis families (Benchtop BrushlessMotor / DCServo for long-travel
delay stages) follow the identical Connect/GetChannel/Enable pattern —
add a sibling controller class here when that hardware enters service.
"""

from __future__ import annotations

import logging
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

__all__ = ["ThorlabsBenchtopPiezo", "ThorlabsPiezoAxis", "DEFAULT_KINESIS_DIR"]

DEFAULT_KINESIS_DIR = Path(r"C:\Program Files\Thorlabs\Kinesis")

#: assemblies needed for the Benchtop Piezo stack, loaded in this order
_PIEZO_ASSEMBLIES = (
    "Thorlabs.MotionControl.DeviceManagerCLI",
    "Thorlabs.MotionControl.GenericPiezoCLI",
    "Thorlabs.MotionControl.Benchtop.PiezoCLI",
)


def _bootstrap_dotnet(dll_dir: Path) -> dict:
    """Import pythonnet + the Kinesis piezo assemblies. Returns the .NET
    namespace objects; raises InstrumentUnavailable on any missing layer."""
    try:
        import clr  # type: ignore  # pythonnet
    except Exception as exc:
        raise InstrumentUnavailable(
            "pythonnet (clr) is not installed — Thorlabs Kinesis driver "
            "unavailable"
        ) from exc

    if not dll_dir.is_dir():
        raise InstrumentUnavailable(
            f"Kinesis directory not found: {dll_dir} — install Thorlabs "
            "Kinesis or set dll_dir in the device config"
        )

    import sys

    if str(dll_dir) not in sys.path:
        sys.path.append(str(dll_dir))
    for name in _PIEZO_ASSEMBLIES:
        dll = dll_dir / f"{name}.dll"
        if not dll.is_file():
            raise InstrumentUnavailable(f"Kinesis assembly missing: {dll}")
        try:
            clr.AddReference(str(dll))
        except Exception as exc:
            raise InstrumentUnavailable(
                f"failed to load Kinesis assembly {dll.name}: {exc}"
            ) from exc

    try:
        from System import Decimal  # type: ignore
        from Thorlabs.MotionControl.DeviceManagerCLI import (  # type: ignore
            DeviceManagerCLI,
        )
        from Thorlabs.MotionControl.Benchtop.PiezoCLI import (  # type: ignore
            BenchtopPiezo,
        )
        from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import (  # type: ignore
            PiezoControlModeTypes,
        )
    except Exception as exc:
        raise InstrumentUnavailable(
            f"Kinesis .NET namespaces failed to import: {exc}"
        ) from exc

    return {
        "Decimal": Decimal,
        "DeviceManagerCLI": DeviceManagerCLI,
        "BenchtopPiezo": BenchtopPiezo,
        "PiezoControlModeTypes": PiezoControlModeTypes,
    }


class ThorlabsPiezoAxis(MotionAxis):
    """One channel of a Benchtop Piezo. Native unit: µm (closed loop)."""

    def __init__(self, config: AxisConfig, controller: "ThorlabsBenchtopPiezo"):
        super().__init__(config, controller)
        self._tl: ThorlabsBenchtopPiezo = controller

    def _channel(self):
        return self._tl._channel_for(int(self.config.channel))

    def _move_abs_raw(self, target: float) -> None:
        net = self._tl._net
        ch = self._channel()
        try:
            ch.SetPosition(net["Decimal"](float(target)))
        except Exception as exc:
            raise InstrumentError(
                f"Thorlabs piezo ch{self.config.channel}: SetPosition failed: {exc}"
            ) from exc

    def _get_position_raw(self) -> float:
        net = self._tl._net
        ch = self._channel()
        try:
            return float(net["Decimal"].ToDouble(ch.GetPosition()))
        except Exception as exc:
            raise InstrumentError(
                f"Thorlabs piezo ch{self.config.channel}: GetPosition failed: {exc}"
            ) from exc

    def _get_status_raw(self) -> AxisStatus:
        pos = self._get_position_raw()
        # Kinesis piezo channels settle fast; the .NET status word exposes
        # no portable "moving" bit at the CLI level, so report settled and
        # let AxisConfig.settle_s provide the dwell after each move.
        return AxisStatus(position=pos, moving=False, on_target=None)

    def _stop_raw(self) -> None:
        ch = self._channel()
        try:
            # Piezo channels have no motion to abort mid-flight at CLI
            # level; disconnect-safe no-op keeps the panic path harmless.
            ch.StopPolling()
            ch.StartPolling(self._tl.POLL_MS)
        except Exception:  # noqa: BLE001 - panic path must not raise
            pass


class ThorlabsBenchtopPiezo(MotionController):
    """One Benchtop Piezo controller (BPC30x), addressed by serial number."""

    POLL_MS = 250
    SETTINGS_INIT_TIMEOUT_MS = 5000

    def __init__(
        self,
        *,
        serial_no: str,
        axes: list[AxisConfig] | None = None,
        dll_dir: str | Path | None = None,
        closed_loop: bool = True,
    ):
        super().__init__()
        if not serial_no:
            raise ValueError("ThorlabsBenchtopPiezo needs the device serial_no")
        self._serial_no = str(serial_no)
        self._axis_configs = list(axes or [])
        self._dll_dir = Path(dll_dir) if dll_dir else DEFAULT_KINESIS_DIR
        self._closed_loop = bool(closed_loop)
        self._net: dict = {}
        self._device = None
        self._channels: dict[int, object] = {}

    # -- lifecycle -----------------------------------------------------------

    def _connect_raw(self) -> None:
        self._net = _bootstrap_dotnet(self._dll_dir)
        net = self._net
        try:
            net["DeviceManagerCLI"].BuildDeviceList()
            device = net["BenchtopPiezo"].CreateBenchtopPiezo(self._serial_no)
            device.Connect(self._serial_no)
        except InstrumentUnavailable:
            raise
        except Exception as exc:
            raise InstrumentUnavailable(
                f"Thorlabs piezo {self._serial_no}: connect failed "
                f"(device off / cable / another program holding it?): {exc}"
            ) from exc
        self._device = device

        for cfg in self._axis_configs:
            ch_no = int(cfg.channel)
            try:
                ch = device.GetChannel(ch_no)
                if not ch.IsSettingsInitialized():
                    ch.WaitForSettingsInitialized(self.SETTINGS_INIT_TIMEOUT_MS)
                ch.StartPolling(self.POLL_MS)
                ch.EnableDevice()
                if self._closed_loop:
                    ch.SetPositionControlMode(
                        net["PiezoControlModeTypes"].CloseLoop
                    )
                self._channels[ch_no] = ch
            except Exception as exc:
                self._close_raw()
                raise InstrumentUnavailable(
                    f"Thorlabs piezo {self._serial_no} ch{ch_no}: init failed: {exc}"
                ) from exc

    def _close_raw(self) -> None:
        for ch in self._channels.values():
            try:
                ch.StopPolling()
                ch.DisableDevice()
            except Exception:  # noqa: BLE001
                pass
        self._channels = {}
        device, self._device = self._device, None
        if device is not None:
            try:
                device.Disconnect(True)
            except Exception:  # noqa: BLE001
                pass

    def _build_axes(self) -> dict[str, MotionAxis]:
        return {cfg.name: ThorlabsPiezoAxis(cfg, self) for cfg in self._axis_configs}

    def _channel_for(self, ch_no: int):
        ch = self._channels.get(ch_no)
        if ch is None:
            raise InstrumentError(
                f"Thorlabs piezo {self._serial_no}: channel {ch_no} not "
                f"initialised (configured: {sorted(self._channels)})"
            )
        return ch
