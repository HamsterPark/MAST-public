"""Shared fakes for the optical-instrument driver tests.

FakeTransport implements SerialLike (same style as
tests/v2/environment/test_sensors.py); FakeAxis/FakeController implement
the mast.instruments.base contract in-memory so registry/skill tests run
the REAL base-class template methods (limit checks, wait loops) without
hardware.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.environment.serial_transport import SerialUnavailable
from mast.instruments.base import (
    AxisConfig,
    AxisStatus,
    MotionAxis,
    MotionController,
)


class FakeTransport:
    """Canned-reply SerialLike. ``replies`` pops one frame per transact."""

    def __init__(self, reply: bytes = b"", *, unavailable: bool = False,
                 replies: list[bytes] | None = None):
        self._reply = reply
        self._replies = list(replies) if replies is not None else None
        self.unavailable = unavailable
        self.sent: list[bytes] = []
        self.closed = False

    def transact(self, payload, *, read_size=None, read_until=None, timeout=None):
        if self.unavailable:
            raise SerialUnavailable("fake: no port")
        self.sent.append(bytes(payload))
        if self._replies is not None:
            return self._replies.pop(0) if self._replies else b""
        return self._reply

    def close(self):
        self.closed = True


class FakeAxis(MotionAxis):
    """In-memory axis: instant moves, real base-class safety layer."""

    def __init__(self, config: AxisConfig, controller: "FakeController",
                 *, start: float = 0.0):
        super().__init__(config, controller)
        self.position = float(start)
        self.moves: list[float] = []
        self.stopped = 0
        self.homed_calls = 0

    def _move_abs_raw(self, target: float) -> None:
        self.position = target
        self.moves.append(target)

    def _get_position_raw(self) -> float:
        return self.position

    def _get_status_raw(self) -> AxisStatus:
        return AxisStatus(position=self.position, moving=False, on_target=True)

    def _stop_raw(self) -> None:
        self.stopped += 1

    def _home_raw(self) -> None:
        self.homed_calls += 1
        self.position = 0.0


class FakeController(MotionController):
    def __init__(self, axis_configs: list[AxisConfig] | None = None):
        super().__init__()
        self._axis_configs = axis_configs or []
        self.connected_calls = 0
        self.closed_calls = 0

    def _connect_raw(self) -> None:
        self.connected_calls += 1

    def _close_raw(self) -> None:
        self.closed_calls += 1

    def _build_axes(self):
        return {c.name: FakeAxis(c, self) for c in self._axis_configs}


@pytest.fixture
def fake_controller():
    def _make(**axis_kwargs) -> FakeController:
        defaults = dict(name="x", channel=1, min_pos=0.0, max_pos=100.0, unit="um")
        defaults.update(axis_kwargs)
        return FakeController([AxisConfig(**defaults)])

    return _make
