"""Live ENVIRONMENT readings — relays the latest cached sensor values to the
right-rail ENVIRONMENT panel (TS-rewrite seam).

The frontend right rail shows four headline gauges (vacuum / temperature /
helium_level / noise_level) plus the full sensor list. This route is a THIN
relay over the LIVE :class:`~mast.environment.monitor.EnvironmentMonitor` that
``api.bootstrap`` wires onto ``ctx.environment_monitor``: it reads the monitor's
LATEST cached readings (``get_latest()`` — populated by the monitor's own
background thread; we never open a serial port from this request) and reshapes
them into the headline gauges + sensor rows.

GRACEFUL DEGRADATION is mandatory (house rule 2): the app must boot standalone
with no live core wired. With no monitor on ``ctx`` — or if any call raises —
the endpoint returns a valid body with every headline value ``None`` and
``degraded=True`` so the rail renders ``N/A`` — never a 500, never a crash on
import. The heavy environment backend (``SensorReading`` type) is only touched
via duck-typing on the relayed readings; nothing here opens hardware.

NO business logic lives here — the monitor owns reading/alarm/archival; this is
a read-only relay (R6).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from mast.api.schemas_environment_readings import (  # noqa: F401 (re-exported members)
    EnvironmentAlarmEntry,
    EnvironmentAlarmsResponse,
    EnvironmentReadingsResponse,
    SensorEntry,
    SensorValue,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["environment"])

# The four headline gauges keyed by their canonical sensor name. These names are
# the single source of truth shared with environment/placeholders.py (the
# placeholder sensors are named exactly "vacuum" / "temperature" /
# "helium_level" / "noise_level"), so a configured DL-7 / Lakeshore sensor only
# lands in a headline slot if the user named it canonically — otherwise it still
# appears in the full ``sensors`` list AND is matched into a headline slot by
# driver type (see ``_headline_by_type``).
_HEADLINE_NAMES = ("vacuum", "temperature", "helium_level", "noise_level")

# Driver class name → headline slot, so an auto-detected gauge named e.g.
# "DL-7 真空计 (COM15)" or "MODEL336 (COM13)" still feeds the vacuum/temperature
# rail. Matched by class name (substring) to avoid importing the heavy drivers
# here (degrade-safe; no serial backend pulled in on a standalone boot).
_TYPE_TO_HEADLINE = {
    "DL7VacuumSensor": "vacuum",
    "VacuumSensor": "vacuum",
    "LakeshoreTemperatureSensor": "temperature",
    "TemperatureSensor": "temperature",
    "HeliumLevelSensor": "helium_level",
    "NoiseSensor": "noise_level",
}


def _driver_type(sensor: Any) -> str | None:
    """Best-effort driver-class name for a sensor object (unwraps the
    autodetect ``_RenamedSensor`` adapter). None when unknowable."""
    if sensor is None:
        return None
    inner = getattr(sensor, "_inner", sensor)  # unwrap autodetect._RenamedSensor
    try:
        return type(inner).__name__
    except Exception:  # pragma: no cover - defensive
        return None


def _is_placeholder(sensor: Any) -> bool:
    """这个传感器对象是**占位实现**吗（``environment/placeholders.py``）。

    判据是 ``isinstance``，不是类名字符串：类名判据在改名时静默变假，而这条的
    失效方式恰好是「一个占位又装回读数行里」—— 与它要修的缺陷同一个形状。

    为什么要区分：占位与真驱动读失败今天都报 ``unavailable``。前者插什么都不会
    好（MAST 里没有这个量的驱动），后者插回去就好。面板只该为后者留一行。
    """
    if sensor is None:
        return False
    inner = getattr(sensor, "_inner", sensor)  # unwrap autodetect._RenamedSensor
    try:
        from mast.environment.placeholders import PlaceholderSensor
    except Exception:  # pragma: no cover - degrade-safe (never import-fatal)
        return False
    return isinstance(inner, PlaceholderSensor)


def _connected(status: str) -> bool:
    """A sensor is 'connected' if its last reading wasn't a hard
    no-hardware/garbled state. ``ok`` / ``warning`` / ``alarm`` are live;
    ``unavailable`` (no port) and ``error`` (garbled / read failure) are not."""
    return status not in ("unavailable", "error", "")


def _value_for(reading: Any) -> tuple[float | None, str | None, str]:
    """(value, unit, status) off a relayed SensorReading. ``value`` is None when
    the reading is unavailable/error so the rail shows N/A instead of a
    misleading 0."""
    status = str(getattr(reading, "status", "unavailable") or "unavailable")
    unit = getattr(reading, "unit", None) or None
    if not _connected(status):
        return None, unit, status
    raw = getattr(reading, "value", None)
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        value = None
    return value, unit, status


@router.get("/environment/readings", response_model=EnvironmentReadingsResponse)
def get_environment_readings(request: Request) -> EnvironmentReadingsResponse:
    """Snapshot the live environment sensors for the right-rail ENVIRONMENT panel.

    Relays the monitor's LATEST cached readings (no serial I/O on this request).
    Degrades to all-``None`` headline gauges + ``degraded=True`` when no live
    monitor is wired (standalone boot) — never a 500."""
    ctx = request.app.state.ctx
    monitor = getattr(ctx, "environment_monitor", None)
    if monitor is None:
        # No live monitor → every gauge N/A, but a valid body (never 500).
        return EnvironmentReadingsResponse(degraded=True)

    try:
        latest = monitor.get_latest() or {}
    except Exception as exc:
        logger.warning("environment readings snapshot failed: %s", exc)
        return EnvironmentReadingsResponse(degraded=True)

    # Best-effort per-sensor driver type, read off the monitor's private sensor
    # map under getattr so a future API change can't 500 us.
    sensor_objs: dict[str, Any] = {}
    try:
        sensor_objs = dict(getattr(monitor, "_sensors", {}) or {})
    except Exception:  # pragma: no cover - defensive
        sensor_objs = {}

    headline: dict[str, SensorValue] = {}
    sensors: list[SensorEntry] = []
    for name, reading in latest.items():
        value, unit, status = _value_for(reading)
        connected = value is not None
        sensor_obj = sensor_objs.get(name)
        stype = _driver_type(sensor_obj)
        # Report the canonical sensor kind so the panel can associate a sensor
        # entry with its headline gauge. Prefer its canonical name, then driver type.
        slot = name if name in _HEADLINE_NAMES else _TYPE_TO_HEADLINE.get(stype or "")
        sensors.append(
            SensorEntry(
                name=str(name),
                type=stype,
                kind=slot,
                placeholder=_is_placeholder(sensor_obj),
                value=value,
                unit=unit,
                status=status,
                connected=connected,
            )
        )
        # Prefer a connected reading when multiple sensors map to one headline
        # slot. Iteration order must not hide a valid reading behind an unavailable
        # entry. An unavailable sensor still fills an otherwise empty slot so a
        # missing measurement remains visible as N/A.
        if slot is not None and (slot not in headline
                                 or (connected and not headline[slot].connected)):
            headline[slot] = SensorValue(
                value=value, unit=unit, status=status, connected=connected
            )

    try:
        overall = str(monitor.overall_status() or "unavailable")
    except Exception:
        overall = "unavailable"

    return EnvironmentReadingsResponse(
        vacuum=headline.get("vacuum", SensorValue()),
        temperature=headline.get("temperature", SensorValue()),
        helium_level=headline.get("helium_level", SensorValue()),
        noise_level=headline.get("noise_level", SensorValue()),
        sensors=sensors,
        overall_status=overall,
        degraded=False,
    )


@router.get("/environment/alarms", response_model=EnvironmentAlarmsResponse)
def get_environment_alarms(request: Request) -> EnvironmentAlarmsResponse:
    """Recent environment alarm/warning transitions for the Lab Console banner.

    Reads the live app's capped in-memory alarm log (populated by the
    EnvironmentMonitor → CoreRuntime._on_env_alarm stop-loss path). A hard alarm
    also aborts the autonomous run + retracts the tip; this endpoint just makes
    the history visible so an unattended vacuum/thermal fault isn't silent.
    Never 500s."""
    ctx = request.app.state.ctx
    app_handle = getattr(ctx, "live_app", None) or getattr(ctx, "app", None)
    log = getattr(app_handle, "_env_alarm_log", None)
    if not isinstance(log, list):
        return EnvironmentAlarmsResponse(degraded=True)
    entries: list[EnvironmentAlarmEntry] = []
    for row in log[-50:]:
        try:
            entries.append(EnvironmentAlarmEntry(
                t=str(row.get("t", "")),
                sensor=str(row.get("sensor", "")),
                status=str(row.get("status", "alarm")),
                value=row.get("value"),
                unit=row.get("unit"),
                prev=str(row.get("prev", "ok")),
            ))
        except Exception:  # pragma: no cover - defensive per-row
            continue
    monitor = getattr(ctx, "environment_monitor", None)
    active = "ok"
    if monitor is not None:
        try:
            active = str(monitor.overall_status() or "ok")
        except Exception:
            active = "ok"
    return EnvironmentAlarmsResponse(
        alarms=entries, active_status=active, degraded=False)
