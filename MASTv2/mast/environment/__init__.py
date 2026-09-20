"""Environment: sensor abstraction and continuous monitoring.

Real serial drivers (auto-detected or configured) replace the placeholders when
hardware is present:
  * DL-7 vacuum gauge  — RS-485 / Modbus-RTU  (mast.environment.dl7_vacuum)
  * Lakeshore monitor  — SCPI over serial     (mast.environment.lakeshore_temp)

With nothing connected every sensor degrades to ``status="unavailable"`` and the
program runs normally — see mast.environment.autodetect.build_environment_sensors.
"""

from __future__ import annotations

from mast.environment.alarm import AlarmSpec, worst_status
from mast.environment.autodetect import (
    autodetect_sensors,
    build_environment_sensors,
    build_sensors_from_config,
)
from mast.environment.base import EnvironmentSensor
from mast.environment.dl7_vacuum import DL7VacuumSensor, probe_dl7
from mast.environment.lakeshore_temp import (
    LakeshoreTemperatureSensor,
    probe_lakeshore,
)
from mast.environment.monitor import EnvironmentMonitor
from mast.environment.placeholders import (
    HeliumLevelSensor,
    NoiseSensor,
    PlaceholderSensor,
    TemperatureSensor,
    VacuumSensor,
)
from mast.environment.serial_transport import (
    HAS_PYSERIAL,
    SerialSettings,
    SerialTransport,
    list_serial_ports,
)

__all__ = [
    "EnvironmentSensor",
    "EnvironmentMonitor",
    "PlaceholderSensor",
    "VacuumSensor",
    "HeliumLevelSensor",
    "TemperatureSensor",
    "NoiseSensor",
    # serial sensors
    "DL7VacuumSensor",
    "probe_dl7",
    "LakeshoreTemperatureSensor",
    "probe_lakeshore",
    # assembly + config
    "build_environment_sensors",
    "build_sensors_from_config",
    "autodetect_sensors",
    "AlarmSpec",
    "worst_status",
    # transport
    "HAS_PYSERIAL",
    "SerialSettings",
    "SerialTransport",
    "list_serial_ports",
]
