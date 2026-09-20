"""Placeholder sensors for hardware not yet connected."""

from __future__ import annotations

from mast.core.types import SensorReading
from mast.environment.base import EnvironmentSensor


class PlaceholderSensor(EnvironmentSensor):
    """Placeholder for sensors not yet connected."""

    def __init__(self, sensor_name: str, unit: str = ""):
        self._name = sensor_name
        self._unit = unit

    def name(self) -> str:
        return self._name

    def read(self) -> SensorReading:
        return SensorReading(value=0.0, unit=self._unit, status="unavailable")


class VacuumSensor(PlaceholderSensor):
    def __init__(self) -> None:
        super().__init__("vacuum", "mbar")


class HeliumLevelSensor(PlaceholderSensor):
    def __init__(self) -> None:
        super().__init__("helium_level", "%")


class TemperatureSensor(PlaceholderSensor):
    def __init__(self) -> None:
        super().__init__("temperature", "K")


class NoiseSensor(PlaceholderSensor):
    def __init__(self) -> None:
        super().__init__("noise_level", "pm")
