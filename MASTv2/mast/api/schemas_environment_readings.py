"""Pydantic response models for the live ENVIRONMENT readings seam.

This slice powers the right-rail ENVIRONMENT panel: it relays the LATEST cached
sensor readings off the live :class:`~mast.environment.monitor.EnvironmentMonitor`
(wired onto ``ctx.environment_monitor`` by ``api.bootstrap``) so the four headline
gauges — vacuum / temperature / helium_level / noise_level — show REAL numbers
when DL-7 vacuum / Lakeshore temperature drivers (or any configured sensor) are
connected, and degrade to ``N/A`` (``value=None``) otherwise.

Per the house rules these models are the SINGLE SOURCE OF TYPES for this slice.
The read endpoint carries ``response_model=EnvironmentReadingsResponse`` and
degrades safely (``degraded=True``, all headline values ``None``) when no live
monitor is wired — never a 500.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SensorValue(BaseModel):
    """One headline gauge value (vacuum / temperature / helium / noise).

    ``value`` is ``None`` when the gauge is unavailable / not wired so the
    frontend renders ``N/A`` instead of a misleading ``0``. ``status`` mirrors
    ``SensorReading.status`` ("ok" / "warning" / "alarm" / "error" /
    "unavailable")."""

    value: Optional[float] = None
    unit: Optional[str] = None
    status: str = "unavailable"
    connected: bool = False


class SensorEntry(BaseModel):
    """One row in the full sensor list (every sensor the monitor watches).

    Flattened from the live monitor's ``name -> SensorReading`` map plus the
    sensor's declared driver type. ``value`` is ``None`` when the reading is
    unavailable so the panel shows ``N/A``."""

    name: str
    #: Driver CLASS name (``LakeshoreTemperatureSensor``) — diagnostics only.
    type: Optional[str] = None
    #: Which headline gauge this sensor IS: vacuum / temperature / helium_level
    #: / noise_level, or None for an extra gauge that maps to none of them.
    #:
    #: Added 2026-07-28 . The panel tried to de-duplicate the rail against
    #: ``type`` — comparing a driver class name to "temperature" — so the test
    #: never matched and EVERY sensor was printed twice: once as the headline
    #: row, once again as a raw row. The operator saw 77.42 K as both
    #: "Temperature" and "MODEL335 (COM13)", with vacuum / helium_level /
    #: noise_level duplicated underneath as bare type strings. The mapping
    #: already existed server-side; it just was not being told to the client.
    kind: Optional[str] = None
    #: 这一行是**占位实现**，不是一个读不到的传感器。
    #:
    #: 两件事今天在 ``status`` 上长得一模一样：一台 COM 口拔掉的 DL-7 与一个
    #: 从来没返回过数字的 ``NoiseSensor``，都报 ``unavailable``。但它们是不同的
    #: 事实 ——「表接着但读不到」可以插回去，「MAST 里根本没有这个量的驱动」
    #: 插什么都没用 —— 而只有前者值得在面板上占一行等着变好。
    #: 噪声这一路一直是 unavailable，看起来像坏了。
    placeholder: bool = False
    value: Optional[float] = None
    unit: Optional[str] = None
    status: str = "unavailable"
    connected: bool = False


class EnvironmentAlarmEntry(BaseModel):
    """One recorded environment alarm/warning transition (most recent last).

    Surfaced so the Lab Console can show a banner when a sensor crossed into an
    alert state — otherwise a vacuum failure / thermal runaway during an
    unattended run is completely invisible in the UI."""

    t: str = ""            # HH:MM:SS wall clock of the transition
    sensor: str = ""
    status: str = "alarm"  # "alarm" | "warning" | "error"
    value: Optional[float] = None
    unit: Optional[str] = None
    prev: str = "ok"       # status it transitioned FROM


class EnvironmentAlarmsResponse(BaseModel):
    """GET /api/environment/alarms — recent alert-state transitions.

    ``active`` is the current worst status across live sensors (for a persistent
    banner); ``alarms`` is the recent transition history (capped). ``degraded``
    means no live monitor is wired."""

    alarms: list[EnvironmentAlarmEntry] = Field(default_factory=list)
    active_status: str = "ok"
    degraded: bool = False


class EnvironmentReadingsResponse(BaseModel):
    """GET /api/environment/readings — live right-rail ENVIRONMENT snapshot.

    The four named gauges are the headline rail; ``sensors`` is the full set
    (including any extra configured / auto-detected gauges and the helium /
    noise placeholders). ``degraded=True`` means no live monitor is wired (boot
    standalone) — every headline value is ``None`` and the rail shows ``N/A``."""

    vacuum: SensorValue = Field(default_factory=SensorValue)
    temperature: SensorValue = Field(default_factory=SensorValue)
    helium_level: SensorValue = Field(default_factory=SensorValue)
    noise_level: SensorValue = Field(default_factory=SensorValue)
    sensors: list[SensorEntry] = Field(default_factory=list)
    overall_status: str = "unavailable"
    degraded: bool = False


__all__ = [
    "SensorValue",
    "SensorEntry",
    "EnvironmentReadingsResponse",
    "EnvironmentAlarmEntry",
    "EnvironmentAlarmsResponse",
]
