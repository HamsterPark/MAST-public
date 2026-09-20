"""AcquireSignalPoint (atomic) + OpticalStageScan (1D/2D raster) skills.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_optics_scan.py -x -v
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

import csv

import pytest

from mast.core.types import NanonisCallRecord
from mast.instruments.base import AxisConfig, AxisStatus, MotionAxis, MotionController
from mast.instruments.registry import InstrumentRegistry, reset_instrument_registry
from mast.skills.builtins import optics_scan as osc
from mast.skills.builtins.optics_scan import AcquireSignalPoint, OpticalStageScan


# ── in-memory fakes ────────────────────────────────────────────────────────
class _Axis(MotionAxis):
    def __init__(self, config, controller):
        super().__init__(config, controller)
        self.position = config.min_pos
        self.moves: list[float] = []

    def _move_abs_raw(self, target):
        self.position = target
        self.moves.append(target)

    def _get_position_raw(self):
        return self.position

    def _get_status_raw(self):
        return AxisStatus(position=self.position, moving=False, on_target=True)

    def _stop_raw(self):
        pass


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
    def __init__(self):
        import threading

        self._lock = threading.RLock()
        self._config = {"devices": [], "delay_line": None}
        self._controllers = {}
        self._ctrl = _Controller([
            AxisConfig(name="x", channel=1, min_pos=0.0, max_pos=100.0, unit="um"),
            AxisConfig(name="y", channel=2, min_pos=0.0, max_pos=100.0, unit="um"),
        ])

    def axis(self, device_id, axis_name):
        if device_id == "stage":
            return self._ctrl.axis(axis_name)
        from mast.instruments.base import InstrumentError

        raise InstrumentError(f"no optical device {device_id!r}")


class FakeCtx:
    def __init__(self, *, current=1.5e-9, signal=2.5e-6, fail_method="",
                 abort_after=None):
        self.calls: list[tuple] = []
        self._current = current
        self._signal = signal
        self._fail = fail_method
        self._abort_after = abort_after
        self._n = 0

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        if method == self._fail:
            return NanonisCallRecord(method=method, args=args,
                                     error=f"canned failure for {method}")
        if method == "Current_Get":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("ok", b"", [self._current]))
        if method == "Signals_ValGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("ok", b"", [self._signal]))
        if method in ("DigLines_PropsSet", "DigLines_Pulse"):
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("ok", b"", []))
        return NanonisCallRecord(method=method, args=args,
                                 error=f"unmocked: {method}")

    def check_abort(self):
        if self._abort_after is not None:
            self._n += 1
            if self._n > self._abort_after:
                return True
        return False


@pytest.fixture
def rig(monkeypatch, tmp_path):
    reg = FakeRegistry()
    reset_instrument_registry(reg)
    monkeypatch.setattr(osc, "project_root", lambda: tmp_path)
    yield reg, tmp_path
    reset_instrument_registry(None)


_FAST = dict(samples_per_point=2, sample_interval_s=0.0, settle_extra_s=0.0)


# ── AcquireSignalPoint ─────────────────────────────────────────────────────
class TestAcquireSignalPoint:
    def test_reads_current_and_signal(self):
        ctx = FakeCtx()
        res = AcquireSignalPoint().execute(ctx, {
            "read_current": True, "signal_indices": "14", "samples": 3,
            "sample_interval_s": 0.0,
        })
        assert res.success
        assert res.data["current_a"] == pytest.approx(1.5e-9)
        assert res.data["sig14_mean"] == pytest.approx(2.5e-6)
        assert res.data["n_samples"] == 3
        assert any(m == "Signals_ValGet" and a and a[0] == 14 for m, a in ctx.calls)

    def test_nothing_to_acquire_rejected(self):
        res = AcquireSignalPoint().execute(FakeCtx(), {
            "read_current": False, "signal_indices": "",
        })
        assert not res.success and "nothing to acquire" in res.error

    def test_nanonis_failure_is_failure(self):
        ctx = FakeCtx(fail_method="Current_Get")
        res = AcquireSignalPoint().execute(ctx, {"read_current": True, "samples": 1})
        assert not res.success and "Current_Get" in res.error

    def test_is_read_auto(self):
        m = AcquireSignalPoint().metadata()
        assert m.category.value == "read" and m.safety_level.name == "AUTO"


# ── OpticalStageScan ───────────────────────────────────────────────────────
class TestOpticalStageScan1D:
    def test_line_scan_writes_csv(self, rig):
        reg, tmp_path = rig
        ctx = FakeCtx()
        res = OpticalStageScan().execute(ctx, {
            "device1": "stage", "axis1": "x", "start1": 0.0, "stop1": 40.0,
            "points1": 5, **_FAST,
        })
        assert res.success
        assert res.data["n_points"] == 5 and res.data["dimensions"] == 1
        rows = list(csv.DictReader(open(res.data["path"], encoding="utf-8")))
        assert len(rows) == 5
        assert float(rows[0]["pos1"]) == pytest.approx(0.0)
        assert float(rows[-1]["pos1"]) == pytest.approx(40.0)
        assert "current_a" in rows[0]

    def test_returns_to_start(self, rig):
        reg, _ = rig
        ax = reg.axis("stage", "x")
        OpticalStageScan().execute(FakeCtx(), {
            "device1": "stage", "axis1": "x", "start1": 10.0, "stop1": 40.0,
            "points1": 4, **_FAST,
        })
        assert ax.moves[-1] == pytest.approx(10.0)   # parked back at start1

    def test_signal_indices_columns(self, rig):
        res = OpticalStageScan().execute(FakeCtx(), {
            "device1": "stage", "axis1": "x", "start1": 0.0, "stop1": 10.0,
            "points1": 3, "read_current": False, "signal_indices": "14", **_FAST,
        })
        assert res.success and res.data["signal_indices"] == [14]
        rows = list(csv.DictReader(open(res.data["path"], encoding="utf-8")))
        assert "sig14_mean" in rows[0]

    def test_unknown_device_fails(self, rig):
        res = OpticalStageScan().execute(FakeCtx(), {
            "device1": "ghost", "axis1": "x", "start1": 0.0, "stop1": 10.0,
            "points1": 3, **_FAST,
        })
        assert not res.success and "fast axis unavailable" in res.error


class TestOpticalStageScan2D:
    def test_grid_scan_shape_and_serpentine(self, rig):
        reg, _ = rig
        res = OpticalStageScan().execute(FakeCtx(), {
            "device1": "stage", "axis1": "x", "start1": 0.0, "stop1": 20.0, "points1": 3,
            "device2": "stage", "axis2": "y", "start2": 0.0, "stop2": 10.0, "points2": 2,
            **_FAST,
        })
        assert res.success
        assert res.data["dimensions"] == 2
        assert res.data["shape"] == [2, 3]
        assert res.data["n_points"] == 6
        rows = list(csv.DictReader(open(res.data["path"], encoding="utf-8")))
        assert len(rows) == 6
        # row 0 forward (col 0,1,2), row 1 serpentine reversed (col 2,1,0)
        r0 = [r for r in rows if int(r["row"]) == 0]
        r1 = [r for r in rows if int(r["row"]) == 1]
        assert [int(r["col"]) for r in r0] == [0, 1, 2]
        assert [int(r["col"]) for r in r1] == [2, 1, 0]
        assert "pos2" in rows[0]

    def test_dry_run_no_acquisition(self, rig):
        _, tmp_path = rig
        ctx = FakeCtx()
        res = OpticalStageScan().execute(ctx, {
            "device1": "stage", "axis1": "x", "start1": 0.0, "stop1": 20.0, "points1": 3,
            "device2": "stage", "axis2": "y", "start2": 0.0, "stop2": 10.0, "points2": 2,
            "dry_run": True, **_FAST,
        })
        assert res.success and res.data["dry_run"] is True
        assert res.data["n_points"] == 6 and res.data["shape"] == [2, 3]
        assert "path" not in res.data
        assert all(m not in ("Current_Get", "Signals_ValGet") for m, _ in ctx.calls)

    def test_trigger_pulsed_each_point(self, rig):
        ctx = FakeCtx()
        res = OpticalStageScan().execute(ctx, {
            "device1": "stage", "axis1": "x", "start1": 0.0, "stop1": 10.0, "points1": 3,
            "trigger_enable": True, "trigger_port": 0, "trigger_line": 2, **_FAST,
        })
        assert res.success
        assert sum(1 for m, _ in ctx.calls if m == "DigLines_PropsSet") == 1
        assert sum(1 for m, _ in ctx.calls if m == "DigLines_Pulse") == 3

    def test_abort_keeps_partial(self, rig):
        ctx = FakeCtx(abort_after=2)
        res = OpticalStageScan().execute(ctx, {
            "device1": "stage", "axis1": "x", "start1": 0.0, "stop1": 40.0,
            "points1": 5, **_FAST,
        })
        assert not res.success
        assert res.data["n_points"] == 2
        assert Path(res.data["path"]).is_file()
