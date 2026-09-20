"""InstrumentRegistry — the single lookup point for optical-bench devices.

Skills and GUI panels obtain hardware exclusively through the module-level
singleton (:func:`get_instrument_registry`) — the same pattern as
``mast.io.plan_overlay``. Nothing here is stored in agent checkpoints
(controllers hold sockets/locks → the no-tensor/no-handle invariant), and
constructing the registry never touches hardware: drivers are built lazily
on first use and connect lazily on first motion.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from mast.instruments import config as inst_config
from mast.instruments.base import (
    AxisConfig,
    InstrumentError,
    MotionAxis,
    MotionController,
)
from mast.instruments.delay_line import DelayLine, DelayLineConfig

logger = logging.getLogger(__name__)

__all__ = ["InstrumentRegistry", "get_instrument_registry", "reset_instrument_registry"]


def _axis_configs(entry: dict[str, Any]) -> list[AxisConfig]:
    out: list[AxisConfig] = []
    for raw in entry.get("axes", []) or []:
        out.append(
            AxisConfig(
                name=str(raw["name"]),
                channel=raw.get("channel", 1),
                min_pos=float(raw["min_pos"]),
                max_pos=float(raw["max_pos"]),
                unit=str(raw.get("unit", "um")),
                default_speed=(
                    float(raw["default_speed"])
                    if raw.get("default_speed") is not None
                    else None
                ),
                settle_s=float(raw.get("settle_s", 0.0)),
            )
        )
    return out


def _build_controller(entry: dict[str, Any]) -> MotionController:
    """Instantiate (but do not connect) the driver for one device entry."""
    dev_type = str(entry.get("type", ""))
    axes = _axis_configs(entry)
    if dev_type == "pztc_nm003":
        from mast.instruments.pztc_nm003 import PztcController, PztcRegisterMap

        reg_map = PztcRegisterMap.from_dict(entry.get("register_map") or {})
        return PztcController(
            port=entry.get("port"),
            baudrate=int(entry.get("baudrate", 115200)),
            slave=int(entry.get("slave", 1)),
            register_map=reg_map,
            axes=axes,
            counts_per_unit=float(entry.get("counts_per_unit", 1.0)),
            position_tolerance=float(entry.get("position_tolerance", 5.0)),
        )
    if dev_type == "pi_gcs":
        from mast.instruments.pi_gcs import PiGcsController

        return PiGcsController(
            port=entry.get("port"),
            baudrate=int(entry.get("baudrate", 115200)),
            model=str(entry.get("model", "E-861")),
            axes=axes,
            servo_on_connect=bool(entry.get("servo_on_connect", True)),
        )
    if dev_type == "thorlabs_benchtop_piezo":
        from mast.instruments.thorlabs_kinesis import ThorlabsBenchtopPiezo

        return ThorlabsBenchtopPiezo(
            serial_no=str(entry.get("serial_no", "")),
            axes=axes,
            dll_dir=entry.get("dll_dir") or None,
            closed_loop=bool(entry.get("closed_loop", True)),
        )
    if dev_type == "thorlabs_pdxc2":
        from mast.instruments.thorlabs_pdxc2 import ThorlabsPDXC2Controller

        return ThorlabsPDXC2Controller(
            serial_no=str(entry.get("serial_no", "")),
            axes=axes,
            dll_dir=entry.get("dll_dir") or None,
            counts_per_unit=float(entry.get("counts_per_unit", 1.0)),
            closed_loop=bool(entry.get("closed_loop", True)),
        )
    raise InstrumentError(
        f"unknown optical device type {dev_type!r} "
        f"(supported: {inst_config.DEVICE_TYPES})"
    )


class InstrumentRegistry:
    """Config-driven device inventory with lazily-built, cached drivers."""

    def __init__(self, config_path: Path | None = None):
        self._config_path = config_path
        self._lock = threading.RLock()
        self._config = inst_config.load_config(config_path)
        self._controllers: dict[str, MotionController] = {}

    # -- inventory -------------------------------------------------------------

    def list_devices(self) -> list[dict[str, Any]]:
        """Inventory summary for UI/skills. Never touches hardware."""
        out: list[dict[str, Any]] = []
        with self._lock:
            for entry in self._config.get("devices", []):
                if not isinstance(entry, dict):
                    continue
                dev_id = str(entry.get("id", ""))
                ctrl = self._controllers.get(dev_id)
                out.append(
                    {
                        "id": dev_id,
                        "name": entry.get("name", dev_id),
                        "type": entry.get("type", "?"),
                        "enabled": bool(entry.get("enabled", True)),
                        "connected": bool(ctrl.is_connected) if ctrl else False,
                        "axes": [
                            {
                                "name": a.get("name"),
                                "unit": a.get("unit", "um"),
                                "min_pos": a.get("min_pos"),
                                "max_pos": a.get("max_pos"),
                                "role": a.get("role"),
                            }
                            for a in entry.get("axes", []) or []
                        ],
                    }
                )
        return out

    # -- lookups ---------------------------------------------------------------

    def controller(self, device_id: str) -> MotionController:
        with self._lock:
            ctrl = self._controllers.get(device_id)
            if ctrl is not None:
                return ctrl
            entry = inst_config.device_by_id(self._config, device_id)
            if entry is None:
                known = [
                    str(e.get("id"))
                    for e in self._config.get("devices", [])
                    if isinstance(e, dict)
                ]
                raise InstrumentError(
                    f"no optical device with id {device_id!r}; configured: {known}"
                )
            if not entry.get("enabled", True):
                raise InstrumentError(f"optical device {device_id!r} is disabled")
            ctrl = _build_controller(entry)
            self._controllers[device_id] = ctrl
            return ctrl

    def axis(self, device_id: str, axis_name: str) -> MotionAxis:
        return self.controller(device_id).axis(axis_name)

    def find_axes_by_role(self, role: str) -> list[tuple[str, str, dict[str, Any]]]:
        """(device_id, axis_name, axis_entry) for every axis tagged *role*.
        Config-level only — does not build drivers."""
        hits: list[tuple[str, str, dict[str, Any]]] = []
        with self._lock:
            for entry in self._config.get("devices", []):
                if not isinstance(entry, dict) or not entry.get("enabled", True):
                    continue
                for a in entry.get("axes", []) or []:
                    if isinstance(a, dict) and a.get("role") == role:
                        hits.append((str(entry.get("id")), str(a.get("name")), a))
        return hits

    def delay_line(self) -> DelayLine:
        with self._lock:
            raw = self._config.get("delay_line")
        if not raw:
            raise InstrumentError(
                "no delay_line section in config/optical_instruments.json — "
                "configure the pump-probe delay stage first"
            )
        cfg = DelayLineConfig.from_dict(raw)
        return DelayLine(self.axis(cfg.device_id, cfg.axis), cfg)

    # -- lifecycle ---------------------------------------------------------------

    def stop_all(self) -> None:
        """Panic-stop every built controller. Best-effort, never raises."""
        with self._lock:
            controllers = list(self._controllers.values())
        for ctrl in controllers:
            try:
                ctrl.stop_all()
            except Exception:  # noqa: BLE001
                pass

    def close_all(self) -> None:
        with self._lock:
            controllers, self._controllers = list(self._controllers.values()), {}
        for ctrl in controllers:
            try:
                ctrl.close()
            except Exception:  # noqa: BLE001
                pass

    def reload(self) -> None:
        """Re-read the manifest; closes existing connections first."""
        self.close_all()
        with self._lock:
            self._config = inst_config.load_config(self._config_path)


# ── module-level singleton (skills' access path; plan_overlay precedent) ──

_registry: InstrumentRegistry | None = None
_registry_lock = threading.Lock()


def get_instrument_registry() -> InstrumentRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = InstrumentRegistry()
    return _registry


def reset_instrument_registry(new: InstrumentRegistry | None = None) -> None:
    """Swap/clear the singleton (tests; config reload from GUI)."""
    global _registry
    with _registry_lock:
        old, _registry = _registry, new
    if old is not None:
        old.close_all()
