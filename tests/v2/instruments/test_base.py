"""MotionAxis/MotionController base contract — Layer-0 soft limits,
relative-move resolution, wait loops, panic-stop semantics.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/instruments/test_base.py -x -v
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

from mast.instruments.base import (
    AxisConfig,
    AxisStatus,
    InstrumentError,
    MotionTimeout,
    TravelLimitError,
)

from .conftest import FakeAxis, FakeController


class TestAxisConfig:
    def test_inverted_limits_rejected(self):
        with pytest.raises(ValueError):
            AxisConfig(name="x", channel=1, min_pos=10.0, max_pos=1.0)

    def test_equal_limits_rejected(self):
        with pytest.raises(ValueError):
            AxisConfig(name="x", channel=1, min_pos=5.0, max_pos=5.0)


class TestSoftLimits:
    def test_move_inside_limits_ok(self, fake_controller):
        ctrl = fake_controller()
        status = ctrl.axis("x").move_abs(50.0)
        assert status.position == 50.0

    def test_move_above_max_refused_before_motion(self, fake_controller):
        ctrl = fake_controller()
        ax = ctrl.axis("x")
        with pytest.raises(TravelLimitError):
            ax.move_abs(100.001)
        assert ax.moves == []  # driver primitive never reached

    def test_move_below_min_refused(self, fake_controller):
        ctrl = fake_controller(min_pos=-5.0)
        with pytest.raises(TravelLimitError):
            ctrl.axis("x").move_abs(-5.1)

    def test_limits_inclusive_at_bounds(self, fake_controller):
        ctrl = fake_controller()
        ax = ctrl.axis("x")
        assert ax.move_abs(0.0).position == 0.0
        assert ax.move_abs(100.0).position == 100.0

    def test_relative_move_checks_final_target(self, fake_controller):
        ctrl = fake_controller()
        ax = ctrl.axis("x")
        ax.move_abs(90.0)
        with pytest.raises(TravelLimitError):
            ax.move_rel(11.0)  # 90 + 11 = 101 > 100
        assert ax.move_rel(10.0).position == 100.0

    def test_stop_is_never_limit_checked(self, fake_controller):
        ctrl = fake_controller()
        ax = ctrl.axis("x")
        ax.stop()
        assert ax.stopped == 1


class TestWaiting:
    def test_wait_until_settled_times_out(self, fake_controller):
        ctrl = fake_controller()
        ax = ctrl.axis("x")
        ax.POLL_INTERVAL_S = 0.001

        def never_settled():
            return AxisStatus(position=0.0, moving=True, on_target=False)

        ax._get_status_raw = never_settled
        with pytest.raises(MotionTimeout):
            ax.move_abs(10.0, wait=True, timeout=0.02)

    def test_not_moving_without_on_target_counts_as_settled(self, fake_controller):
        ctrl = fake_controller()
        ax = ctrl.axis("x")
        ax._get_status_raw = lambda: AxisStatus(
            position=10.0, moving=False, on_target=None
        )
        status = ax.move_abs(10.0, wait=True, timeout=1.0)
        assert status.position == 10.0

    def test_settle_extra_dwell_applied(self, fake_controller):
        ctrl = fake_controller(settle_s=0.01)
        ax = ctrl.axis("x")
        import time

        t0 = time.monotonic()
        ax.move_abs(1.0, wait=True)
        assert time.monotonic() - t0 >= 0.01


class TestController:
    def test_lazy_connect_on_axis_access(self, fake_controller):
        ctrl = fake_controller()
        assert not ctrl.is_connected
        ctrl.axis("x")
        assert ctrl.is_connected
        assert ctrl.connected_calls == 1

    def test_unknown_axis_lists_configured(self, fake_controller):
        ctrl = fake_controller()
        with pytest.raises(InstrumentError, match="configured axes"):
            ctrl.axis("nope")

    def test_close_idempotent(self, fake_controller):
        ctrl = fake_controller()
        ctrl.connect()
        ctrl.close()
        ctrl.close()
        assert ctrl.closed_calls == 1

    def test_home_without_support_raises(self, fake_controller):
        cfg = AxisConfig(name="y", channel=2, min_pos=0.0, max_pos=1.0)
        ctrl = FakeController([cfg])

        class NoHomeAxis(FakeAxis):
            def _home_raw(self):
                raise InstrumentError("no homing support")

        ctrl._build_axes = lambda: {"y": NoHomeAxis(cfg, ctrl)}
        with pytest.raises(InstrumentError, match="no homing"):
            ctrl.axis("y").home()

    def test_stop_all_swallows_axis_errors(self, fake_controller):
        ctrl = fake_controller()
        ax = ctrl.axis("x")

        def boom():
            raise RuntimeError("dead axis")

        ax._stop_raw = boom
        ctrl.stop_all()  # must not raise
