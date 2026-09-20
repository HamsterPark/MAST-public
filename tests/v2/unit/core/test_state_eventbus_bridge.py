"""v2 regression test: state.refresh() must still publish HARDWARE_STATE.

v2 GUI Dashboard subscribes via WebSocket to EventBus.HARDWARE_STATE events.
An earlier port iteration removed `EventBus.publish_hardware_state(...)` from
`InstrumentState.refresh()` (planning to switch to BufferService), but that
left the Dashboard live update silently broken until BufferService landed.
This test pins the bridge until VisionProducer wiring takes over.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_state_eventbus_bridge.py -x -v
"""
from __future__ import annotations

# ── path bootstrap (must come BEFORE any mast.* import) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass

import pytest

from mast.core.events import Event, EventBus, EventType
from mast.core.state import InstrumentState
from mast.core.types import NanonisCallRecord


@dataclass
class FakePool:
    """Minimal ConnectionPool stub returning canned values."""
    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        # All Nanonis getters return (error_str, raw_bytes, parsed_payload)
        canned = {
            "Bias_Get": ("", b"", [0.5]),
            "ZCtrl_StatusGet": ("", b"", [2]),  # 2 = ON
            "ZCtrl_SetpntGet": ("", b"", [1e-10]),
            "Current_Get": ("", b"", [5e-12]),
            "ZCtrl_ZPosGet": ("", b"", [1.5e-7]),
            "FolMe_XYPosGet": ("", b"", [0.0, 0.0]),
            "ZCtrl_LimitsGet": ("", b"", [3e-7, 0.0]),
            "Scan_StatusGet": ("", b"", [0]),
            "Scan_FrameGet": ("", b"", [0.0, 0.0, 1e-7, 1e-7, 0.0]),
        }
        ret = canned.get(method, ("", b"", None))
        return NanonisCallRecord(method=method, args=args, return_value=ret)


def test_state_refresh_publishes_hardware_state():
    """The Dashboard depends on this event firing every time state refreshes."""
    bus = EventBus.get()
    received: list[Event] = []
    bus.subscribe(received.append)
    try:
        state = InstrumentState(FakePool())
        state.refresh()
    finally:
        bus.unsubscribe(received.append)
    assert any(e.type == EventType.HARDWARE_STATE for e in received), (
        "state.refresh() must publish HARDWARE_STATE for the v2 Dashboard "
        "WebSocket subscriber to keep working."
    )


def test_state_refresh_event_payload_carries_bias():
    """The published event must carry the latest bias reading."""
    bus = EventBus.get()
    received: list[Event] = []
    bus.subscribe(received.append)
    try:
        state = InstrumentState(FakePool())
        state.refresh()
    finally:
        bus.unsubscribe(received.append)
    hw_events = [e for e in received if e.type == EventType.HARDWARE_STATE]
    assert hw_events, "no HARDWARE_STATE event captured"
    payload = hw_events[-1].data
    assert payload.get("bias_v") == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
