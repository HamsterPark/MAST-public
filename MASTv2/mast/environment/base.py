"""Abstract base class for environment sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mast.core.types import SensorReading


class EnvironmentSensor(ABC):
    """Abstract base for environment sensors."""

    @abstractmethod
    def name(self) -> str:
        """Unique sensor identifier."""
        ...

    @abstractmethod
    def read(self) -> SensorReading:
        """Take a single reading. Must not block for long."""
        ...

    def is_healthy(self) -> bool:
        """Default: read and check status."""
        try:
            reading = self.read()
            return reading.status in ("ok", "warning")
        except Exception:
            return False
