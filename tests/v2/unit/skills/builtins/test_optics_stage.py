"""Optics stage skills — execute paths over a fake instrument registry.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_optics_stage.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
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

import pytest

from mast.instruments.base import (
    AxisConfig,
    AxisStatus,
    InstrumentError,
    MotionAxis,
    MotionController,
)
from mast.instruments.delay_line import DelayLine, DelayLineConfig
from mast.instruments.registry import (
    InstrumentRegistry,
    reset_instrument_registry,
)
from mast.skills.builtins.optics_stage import (
    DelayLineGetDelay,
    DelayLineMoveTo,
    HomeOpticalStage,
    ListOpticalDevices,
    OpticalStageGetPos,
    OpticalStageMove,
    OpticalStageWiggle,
    StopOpticalStage,
)


# ── in-memory fakes (real base-class template methods) ────────────────────
class _Axis(MotionAxis):
    def __init__(self, config, controller, start=0.0):
        super().__init__(config, controller)
        self.position = start
        self.stopped = 0

    def _move_abs_raw(self, target):
        self.position = target

    def _get_position_raw(self):
        return self.position

    def _get_status_raw(self):
        return AxisStatus(position=self.position, moving=False, on_target=True,
                          homed=True)

    def _stop_raw(self):
        self.stopped += 1

    def _home_raw(self):
        self.position = 0.0


class _Controller(MotionController):
    def __init__(self, axis_configs):
        super().__init__()
        self._axis_configs = axis_configs

    def _connect_raw(self):
        pass

    def _close_raw(self):
        pass

    def _build_axes(self):
        return {c.name: _Axis(c, self) for c in self._axis_configs}


class FakeRegistry(InstrumentRegistry):
    """Bypasses the JSON manifest: fixed controller + delay line."""

    def __init__(self, *, with_delay_line=True):
        # note: deliberately NOT calling super().__init__ (no config file)
        import threading

        self._lock = threading.RLock()
        self._config = {"devices": [], "delay_line": None}
        self._controllers = {}
        self._ctrl = _Controller([
            AxisConfig(name="x", channel=1, min_pos=0.0, max_pos=100.0, unit="um"),
        ])
        self._delay_ctrl = _Controller([
            AxisConfig(name="delay", channel=1, min_pos=0.0, max_pos=15000.0,
                       unit="um"),
        ])
        self._with_delay_line = with_delay_line

    def list_devices(self):
        return [{"id": "stage1", "name": "台", "type": "fake", "enabled": True,
                 "connected": self._ctrl.is_connected,
                 "axes": [{"name": "x", "unit": "um", "min_pos": 0.0,
                           "max_pos": 100.0, "role": None}]}]

    def controller(self, device_id):
        if device_id == "stage1":
            return self._ctrl
        if device_id == "dl1":
            return self._delay_ctrl
        raise InstrumentError(f"no optical device with id {device_id!r}")

    def delay_line(self):
        if not self._with_delay_line:
            raise InstrumentError("no delay_line section")
        cfg = DelayLineConfig(device_id="dl1", axis="delay", unit_per_mm=1000.0,
                              zero_offset_mm=0.0)
        return DelayLine(self._delay_ctrl.axis("delay"), cfg)


@pytest.fixture
def fake_registry():
    reg = FakeRegistry()
    reset_instrument_registry(reg)
    yield reg
    reset_instrument_registry(None)


class TestListOpticalDevices:
    def test_lists_devices_and_delay_line(self, fake_registry):
        result = ListOpticalDevices().execute(None, {})
        assert result.success
        assert result.data["devices"][0]["id"] == "stage1"
        assert result.data["delay_line"]["device_id"] == "dl1"

    def test_missing_delay_line_still_succeeds(self):
        reset_instrument_registry(FakeRegistry(with_delay_line=False))
        try:
            result = ListOpticalDevices().execute(None, {})
            assert result.success
            assert result.data["delay_line"] is None
        finally:
            reset_instrument_registry(None)


class TestStageMoveAndRead:
    def test_absolute_move(self, fake_registry):
        result = OpticalStageMove().execute(
            None, {"device_id": "stage1", "axis": "x", "position": 42.0}
        )
        assert result.success
        assert result.data["position"] == 42.0
        assert result.data["unit"] == "um"

    def test_relative_move(self, fake_registry):
        OpticalStageMove().execute(
            None, {"device_id": "stage1", "axis": "x", "position": 40.0}
        )
        result = OpticalStageMove().execute(
            None,
            {"device_id": "stage1", "axis": "x", "position": 2.5, "relative": True},
        )
        assert result.success
        assert result.data["position"] == 42.5

    def test_soft_limit_rejection_is_failure_not_exception(self, fake_registry):
        result = OpticalStageMove().execute(
            None, {"device_id": "stage1", "axis": "x", "position": 200.0}
        )
        assert not result.success
        assert "soft travel limit" in result.error

    def test_unknown_device_fails_cleanly(self, fake_registry):
        result = OpticalStageMove().execute(
            None, {"device_id": "ghost", "axis": "x", "position": 1.0}
        )
        assert not result.success
        assert "ghost" in result.error

    def test_get_pos(self, fake_registry):
        fake_registry.controller("stage1").axis("x").move_abs(7.0)
        result = OpticalStageGetPos().execute(
            None, {"device_id": "stage1", "axis": "x"}
        )
        assert result.success
        assert result.data["position"] == 7.0
        assert result.data["limits"] == [0.0, 100.0]


class TestStopAndHome:
    def test_stop_single_device(self, fake_registry):
        ax = fake_registry.controller("stage1").axis("x")
        result = StopOpticalStage().execute(None, {"device_id": "stage1"})
        assert result.success
        assert ax.stopped == 1
        assert result.data["stopped"] == "stage1"

    def test_stop_all(self, fake_registry):
        fake_registry.controller("stage1")  # build
        result = StopOpticalStage().execute(None, {})
        assert result.success
        assert result.data["stopped"] == "all"

    def test_home(self, fake_registry):
        ax = fake_registry.controller("stage1").axis("x")
        ax.move_abs(50.0)
        result = HomeOpticalStage().execute(
            None, {"device_id": "stage1", "axis": "x"}
        )
        assert result.success
        assert ax.position == 0.0


class TestDelayLineSkills:
    def test_get_delay(self, fake_registry):
        result = DelayLineGetDelay().execute(None, {})
        assert result.success
        assert result.data["delay_ps"] == pytest.approx(0.0)
        lo, hi = result.data["delay_range_ps"]
        assert lo == pytest.approx(0.0) and hi > 99.0

    def test_move_to_delay(self, fake_registry):
        result = DelayLineMoveTo().execute(None, {"delay_ps": 10.0})
        assert result.success
        assert result.data["delay_ps"] == pytest.approx(10.0, abs=1e-6)

    def test_out_of_range_delay_pre_checked(self, fake_registry):
        result = DelayLineMoveTo().execute(None, {"delay_ps": 1e6})
        assert not result.success
        assert "outside reachable range" in result.error

    def test_unconfigured_delay_line_fails_cleanly(self):
        reset_instrument_registry(FakeRegistry(with_delay_line=False))
        try:
            result = DelayLineGetDelay().execute(None, {})
            assert not result.success
            assert "delay_line" in result.error
        finally:
            reset_instrument_registry(None)


class TestWiggle:
    def test_alive_axis_reports_scale_and_direction(self, fake_registry):
        result = OpticalStageWiggle().execute(
            None, {"device_id": "stage1", "axis": "x", "delta": 5.0}
        )
        assert result.success
        assert result.data["moved"] is True
        assert result.data["direction_ok"] is True
        assert result.data["scale_ratio"] == pytest.approx(1.0)
        assert result.data["returned_to_start"] is True

    def test_dead_axis_reports_not_moved(self, fake_registry):
        ax = fake_registry.controller("stage1").axis("x")
        ax._move_abs_raw = lambda target: None  # ignores commands = dead
        result = OpticalStageWiggle().execute(
            None, {"device_id": "stage1", "axis": "x", "delta": 5.0}
        )
        assert not result.success
        assert result.data["moved"] is False
        assert "did not move" in result.error

    def test_reversed_encoder_detected(self, fake_registry):
        ax = fake_registry.controller("stage1").axis("x")
        ax.move_abs(20.0)  # park mid-range so ±5 stays in [0, 100]

        def mirror(target, _ax=ax, _base=20.0):
            _ax.position = 2 * _base - target  # readback moves opposite

        ax._move_abs_raw = mirror
        result = OpticalStageWiggle().execute(
            None, {"device_id": "stage1", "axis": "x", "delta": 5.0}
        )
        assert result.success  # it IS alive...
        assert result.data["moved"] is True
        assert result.data["direction_ok"] is False  # ...but miswired
        assert "reversed" in result.summary

    def test_near_limit_flips_direction(self, fake_registry):
        ax = fake_registry.controller("stage1").axis("x")
        ax.move_abs(100.0)  # park at max → forward +delta would exceed
        result = OpticalStageWiggle().execute(
            None, {"device_id": "stage1", "axis": "x", "delta": 5.0}
        )
        assert result.success
        assert result.data["commanded_delta"] == -5.0
        assert result.data["observed_delta"] == pytest.approx(-5.0)

    def test_both_directions_blocked_fails_cleanly(self, fake_registry):
        ax = fake_registry.controller("stage1").axis("x")
        ax.move_abs(50.0)
        result = OpticalStageWiggle().execute(
            None, {"device_id": "stage1", "axis": "x", "delta": 200.0}
        )
        assert not result.success
        assert "soft limits" in result.error

    def test_no_return_leaves_axis_moved(self, fake_registry):
        ax = fake_registry.controller("stage1").axis("x")
        result = OpticalStageWiggle().execute(
            None,
            {"device_id": "stage1", "axis": "x", "delta": 5.0,
             "return_to_start": False},
        )
        assert result.success
        assert result.data["returned_to_start"] is None
        assert ax.position == pytest.approx(5.0)

    def test_zero_delta_rejected(self, fake_registry):
        result = OpticalStageWiggle().execute(
            None, {"device_id": "stage1", "axis": "x", "delta": 0.0}
        )
        assert not result.success
        assert "non-zero" in result.error

    def test_unknown_device_fails_cleanly(self, fake_registry):
        result = OpticalStageWiggle().execute(
            None, {"device_id": "ghost", "axis": "x", "delta": 5.0}
        )
        assert not result.success
        assert "ghost" in result.error


class TestMetadataInvariants:
    def test_all_skills_have_safety_level_and_instantiate_bare(self):
        for cls in (ListOpticalDevices, OpticalStageGetPos, OpticalStageMove,
                    OpticalStageWiggle, StopOpticalStage, HomeOpticalStage,
                    DelayLineGetDelay, DelayLineMoveTo):
            meta = cls().metadata()
            assert meta.safety_level is not None
            assert meta.name
            assert meta.description

    def test_param_validation_via_base(self, fake_registry):
        skill = OpticalStageMove()
        errors = skill.validate_params({"device_id": "stage1"})  # axis+position missing
        assert any("axis" in e for e in errors)
        assert any("position" in e for e in errors)
