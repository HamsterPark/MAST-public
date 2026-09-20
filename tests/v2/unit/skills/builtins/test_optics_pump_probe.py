"""PumpProbeScan — full sweep over fake delay line + canned Nanonis replies.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_optics_pump_probe.py -x -v
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
from mast.instruments.delay_line import DelayLine, DelayLineConfig
from mast.instruments.registry import InstrumentRegistry, reset_instrument_registry
from mast.skills.builtins import optics_pump_probe as opp
from mast.skills.builtins.optics_pump_probe import (
    PumpProbeScan,
    _nanonis_scalar,
    _signal_names,
)

_SIGNAL_NAMES = ["Current (A)", "Bias (V)", "LI Demod 1 X (A)", "LI Demod 1 Y (A)",
                 "LI Demod 2 X (A)", "Z (m)"]


# ── fakes ─────────────────────────────────────────────────────────────────
class _Axis(MotionAxis):
    def __init__(self, config, controller):
        super().__init__(config, controller)
        self.position = 0.0
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
            AxisConfig(name="delay", channel=1, min_pos=0.0, max_pos=15000.0,
                       unit="um"),
        ])
        self._sec = _Controller([
            AxisConfig(name="s2", channel=1, min_pos=-100.0, max_pos=100.0,
                       unit="mm"),
        ])

    def delay_line(self):
        cfg = DelayLineConfig(device_id="dl1", axis="delay", unit_per_mm=1000.0)
        return DelayLine(self._ctrl.axis("delay"), cfg)

    def axis(self, device_id, axis_name):
        if device_id == "stage2":
            return self._sec.axis(axis_name)
        from mast.instruments.base import InstrumentError
        raise InstrumentError(f"no optical device {device_id!r}")


class FakeCtx:
    """context.safe_call with canned Nanonis replies (test_motor.py style)."""

    def __init__(self, *, current=1.5e-9, lockin=2.5e-6, fail_method="",
                 abort_after: int | None = None):
        self.calls: list[tuple] = []
        self._current = current
        self._lockin = lockin
        self._fail = fail_method
        self._abort_after = abort_after
        self._n_aborts = 0

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        if method == self._fail:
            return NanonisCallRecord(method=method, args=args,
                                     error=f"canned failure for {method}")
        if method == "Signals_NamesGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("ok", b"", [len(_SIGNAL_NAMES), _SIGNAL_NAMES]),
            )
        if method == "Current_Get":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("ok", b"", [self._current]))
        if method == "Signals_ValGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("ok", b"", [self._lockin]))
        if method in ("DigLines_PropsSet", "DigLines_Pulse", "DigLines_OutStatusSet"):
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("ok", b"", []))
        return NanonisCallRecord(method=method, args=args,
                                 error=f"unmocked: {method}")

    def check_abort(self):
        if self._abort_after is not None:
            self._n_aborts += 1
            if self._n_aborts > self._abort_after:
                raise RuntimeError("operator abort")


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """Fake registry + artifacts redirected into tmp_path."""
    reg = FakeRegistry()
    reset_instrument_registry(reg)
    monkeypatch.setattr(opp, "project_root", lambda: tmp_path)
    yield reg, tmp_path
    reset_instrument_registry(None)


_FAST = dict(samples_per_point=3, sample_interval_s=0.0, settle_extra_s=0.0)


class TestHelpers:
    def test_nanonis_scalar_single_value_shape(self):
        # Current_Get / Signals_ValGet reply: ("header", raw, [value])
        rec = NanonisCallRecord(method="x", return_value=("h", b"", [1.5e-9]))
        assert _nanonis_scalar(rec) == 1.5e-9

    def test_nanonis_scalar_tolerates_nesting(self):
        rec = NanonisCallRecord(method="x", return_value=("h", b"", [[2.0]]))
        assert _nanonis_scalar(rec) == 2.0

    def test_nanonis_scalar_skips_bools(self):
        rec = NanonisCallRecord(method="x", return_value=("h", b"", [True, 2.0]))
        assert _nanonis_scalar(rec) == 2.0

    def test_nanonis_scalar_raises_when_empty(self):
        rec = NanonisCallRecord(method="x", return_value=("h", b"", []))
        with pytest.raises(ValueError):
            _nanonis_scalar(rec)

    def test_signal_names_extraction(self):
        rec = NanonisCallRecord(
            method="Signals_NamesGet",
            return_value=("ok", b"", [6, _SIGNAL_NAMES]),
        )
        assert _signal_names(rec) == _SIGNAL_NAMES


class TestPumpProbeScan:
    def test_full_sweep_writes_csv_and_data(self, rig):
        _, tmp_path = rig
        ctx = FakeCtx()
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 10.0, "points": 5, **_FAST,
        })
        assert result.success, result.error
        assert result.data["n_points"] == 5
        # auto-discovered "LI Demod 1 X (A)" at index 2
        assert result.data["lockin_signal_index"] == 2

        csv_path = Path(result.data["path"])
        assert csv_path.is_file()
        assert csv_path.parent == tmp_path / "artifacts" / "pump_probe"
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 5
        assert float(rows[0]["delay_ps"]) == pytest.approx(0.0)
        assert float(rows[-1]["delay_ps"]) == pytest.approx(10.0)
        assert float(rows[2]["current_a"]) == pytest.approx(1.5e-9)
        assert float(rows[2]["lockin_v"]) == pytest.approx(2.5e-6)

    def test_returns_to_start_after_sweep(self, rig):
        reg, _ = rig
        ax = reg.delay_line().axis
        ctx = FakeCtx()
        PumpProbeScan().execute(ctx, {
            "delay_start_ps": 2.0, "delay_stop_ps": 8.0, "points": 3, **_FAST,
        })
        # last commanded move parks back at delay_start_ps's stage position
        dl = reg.delay_line()
        assert ax.moves[-1] == pytest.approx(dl.delay_to_position(2.0))

    def test_range_preflight_rejects(self, rig):
        result = PumpProbeScan().execute(FakeCtx(), {
            "delay_start_ps": 0.0, "delay_stop_ps": 1e6, "points": 3, **_FAST,
        })
        assert not result.success
        assert "outside reachable range" in result.error

    def test_explicit_lockin_index_skips_discovery(self, rig):
        ctx = FakeCtx()
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2,
            "lockin_signal_index": 40, **_FAST,
        })
        assert result.success
        assert result.data["lockin_signal_index"] == 40
        assert all(m != "Signals_NamesGet" for m, _ in ctx.calls)

    def test_current_only_mode(self, rig):
        ctx = FakeCtx()
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2,
            "read_lockin": False, **_FAST,
        })
        assert result.success
        assert "lockin_signal_index" not in result.data
        assert all(m != "Signals_ValGet" for m, _ in ctx.calls)

    def test_nothing_to_acquire_rejected(self, rig):
        result = PumpProbeScan().execute(FakeCtx(), {
            "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2,
            "read_current": False, "read_lockin": False, **_FAST,
        })
        assert not result.success
        assert "nothing to acquire" in result.error

    def test_nanonis_failure_keeps_partial_data(self, rig):
        ctx = FakeCtx()
        calls = {"n": 0}
        orig = ctx.safe_call

        def flaky(method, *args, **kw):
            if method == "Current_Get":
                calls["n"] += 1
                if calls["n"] > 6:  # fail during the 3rd point (3 samples/point)
                    return NanonisCallRecord(method=method, error="TCP died")
            return orig(method, *args, **kw)

        ctx.safe_call = flaky
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 10.0, "points": 5, **_FAST,
        })
        assert not result.success
        assert result.data["n_points"] == 2          # two clean points kept
        assert result.data["interrupted_after_points"] == 2
        assert Path(result.data["path"]).is_file()   # partial CSV still written

    def test_operator_abort_keeps_partial_data(self, rig):
        ctx = FakeCtx(abort_after=2)
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 10.0, "points": 5, **_FAST,
        })
        assert not result.success
        assert result.data["n_points"] == 2
        assert "operator abort" in result.error

    def test_lockin_discovery_failure_suggests_explicit_index(self, rig):
        ctx = FakeCtx(fail_method="Signals_NamesGet")
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2, **_FAST,
        })
        assert not result.success
        assert "lockin_signal_index" in result.error

    def test_missing_delay_line_fails_cleanly(self, monkeypatch, tmp_path):
        class NoDelayRegistry(FakeRegistry):
            def delay_line(self):
                from mast.instruments.base import InstrumentError

                raise InstrumentError("no delay_line section")

        reset_instrument_registry(NoDelayRegistry())
        monkeypatch.setattr(opp, "project_root", lambda: tmp_path)
        try:
            result = PumpProbeScan().execute(FakeCtx(), {
                "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2, **_FAST,
            })
            assert not result.success
            assert "delay line unavailable" in result.error
        finally:
            reset_instrument_registry(None)

    def test_tag_sanitised_into_filename(self, rig):
        result = PumpProbeScan().execute(FakeCtx(), {
            "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2,
            "tag": "样品A/#7 run", **_FAST,
        })
        assert result.success
        name = Path(result.data["path"]).name
        assert "/" not in name and "#" not in name and " " not in name


class TestPumpProbeEnhancements:
    def test_dry_run_moves_but_records_nothing(self, rig):
        _, tmp_path = rig
        ctx = FakeCtx()
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 10.0, "points": 5,
            "dry_run": True, **_FAST,
        })
        assert result.success
        assert result.data["dry_run"] is True
        assert result.data["n_points"] == 5
        assert len(result.data["reached_delays_ps"]) == 5
        assert "path" not in result.data                    # no file written
        # no acquisition or trigger calls at all
        assert all(m not in ("Current_Get", "Signals_ValGet", "Signals_NamesGet",
                             "DigLines_Pulse")
                   for m, _ in ctx.calls)
        out = tmp_path / "artifacts" / "pump_probe"
        assert not out.exists() or not list(out.glob("*.csv"))

    def test_dry_run_allowed_with_all_reads_off(self, rig):
        result = PumpProbeScan().execute(FakeCtx(), {
            "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2,
            "read_current": False, "read_lockin": False, "dry_run": True, **_FAST,
        })
        assert result.success and result.data["dry_run"] is True

    def test_extra_signal_indices_recorded(self, rig):
        ctx = FakeCtx()
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 5.0, "points": 3,
            "read_lockin": False, "extra_signal_indices": "14", **_FAST,
        })
        assert result.success
        assert result.data["extra_signal_indices"] == [14]
        rows = list(csv.DictReader(open(result.data["path"], encoding="utf-8")))
        assert "sig14_mean" in rows[0] and "sig14_std" in rows[0]
        assert any(m == "Signals_ValGet" and a and a[0] == 14 for m, a in ctx.calls)

    def test_trigger_configures_once_and_pulses_each_point(self, rig):
        ctx = FakeCtx()
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 5.0, "points": 3,
            "read_lockin": False, "trigger_enable": True,
            "trigger_port": 0, "trigger_line": 2, **_FAST,
        })
        assert result.success
        assert sum(1 for m, _ in ctx.calls if m == "DigLines_PropsSet") == 1
        pulses = [a for m, a in ctx.calls if m == "DigLines_Pulse"]
        assert len(pulses) == 3                       # one per delay point
        assert pulses[0][0] == 0 and pulses[0][1] == [2]   # port, [line]
        assert result.data["triggered_line"] == {"port": 0, "line": 2}

    def test_trigger_config_failure_aborts_cleanly(self, rig):
        ctx = FakeCtx(fail_method="DigLines_PropsSet")
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2,
            "trigger_enable": True, **_FAST,
        })
        assert not result.success
        assert "trigger line" in result.error

    def test_secondary_stage_parked_before_sweep(self, rig):
        reg, _ = rig
        result = PumpProbeScan().execute(FakeCtx(), {
            "delay_start_ps": 0.0, "delay_stop_ps": 5.0, "points": 3,
            "read_lockin": False,
            "secondary_device_id": "stage2", "secondary_axis": "s2",
            "secondary_position": 3.0, **_FAST,
        })
        assert result.success
        assert reg._sec.axis("s2").position == 3.0

    def test_missing_secondary_stage_fails_cleanly(self, rig):
        result = PumpProbeScan().execute(FakeCtx(), {
            "delay_start_ps": 0.0, "delay_stop_ps": 1.0, "points": 2,
            "read_lockin": False,
            "secondary_device_id": "ghost", "secondary_axis": "s2", **_FAST,
        })
        assert not result.success
        assert "secondary stage" in result.error


class TestDualStageScan:
    """双台同时扫: 2nd stage rastered as a second delay axis (2D map)."""

    def test_secondary_scan_2d_map(self, rig):
        reg, _ = rig
        ctx = FakeCtx()
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 5.0, "points": 3,
            "read_lockin": False,
            "secondary_device_id": "stage2", "secondary_axis": "s2",
            "secondary_scan": True,
            "secondary_start": 0.0, "secondary_stop": 10.0, "secondary_points": 2,
            **_FAST,
        })
        assert result.success, result.error
        assert result.data["two_d"] is True
        assert result.data["shape"] == [2, 3]          # 2 sec-pos × 3 delays
        assert result.data["n_points"] == 6
        assert result.data["requested_points"] == 6
        # the 2nd stage really visited BOTH scan positions (not just parked)
        assert set(reg._sec.axis("s2").moves) >= {0.0, 10.0}
        # CSV carries both axes
        rows = list(csv.DictReader(open(result.data["path"], encoding="utf-8")))
        assert len(rows) == 6
        assert "sec_position" in rows[0] and "row" in rows[0] and "col" in rows[0]
        # 6 rows = 2 outer sec rows × 3 inner delays; last row is (sec=10, last delay)
        assert float(rows[-1]["sec_position"]) == pytest.approx(10.0)
        assert int(rows[-1]["row"]) == 1 and int(rows[-1]["col"]) == 2

    def test_secondary_scan_dry_run_2d(self, rig):
        result = PumpProbeScan().execute(FakeCtx(), {
            "delay_start_ps": 0.0, "delay_stop_ps": 5.0, "points": 3,
            "read_current": False, "read_lockin": False, "dry_run": True,
            "secondary_device_id": "stage2", "secondary_axis": "s2",
            "secondary_scan": True,
            "secondary_start": -5.0, "secondary_stop": 5.0, "secondary_points": 3,
            **_FAST,
        })
        assert result.success
        assert result.data["dry_run"] is True
        assert result.data["two_d"] is True
        assert result.data["n_points"] == 9           # 3 sec × 3 delay
        assert result.data["shape"] == [3, 3]
        assert "path" not in result.data

    def test_secondary_scan_abort_keeps_partial_2d(self, rig):
        # abort after 4 acquisitions worth of check_abort — partial 2D kept
        ctx = FakeCtx(abort_after=4)
        result = PumpProbeScan().execute(ctx, {
            "delay_start_ps": 0.0, "delay_stop_ps": 5.0, "points": 3,
            "read_lockin": False,
            "secondary_device_id": "stage2", "secondary_axis": "s2",
            "secondary_scan": True,
            "secondary_start": 0.0, "secondary_stop": 10.0, "secondary_points": 2,
            **_FAST,
        })
        assert not result.success
        assert "operator abort" in result.error
        assert 0 < result.data["n_points"] < 6         # stopped before the full grid
        assert Path(result.data["path"]).is_file()     # partial CSV written

    def test_secondary_park_still_1d_when_scan_off(self, rig):
        # secondary_scan absent → legacy park-once behaviour, NOT a 2D map
        reg, _ = rig
        result = PumpProbeScan().execute(FakeCtx(), {
            "delay_start_ps": 0.0, "delay_stop_ps": 5.0, "points": 3,
            "read_lockin": False,
            "secondary_device_id": "stage2", "secondary_axis": "s2",
            "secondary_position": 7.0, **_FAST,
        })
        assert result.success
        assert "two_d" not in result.data
        assert result.data["n_points"] == 3
        assert reg._sec.axis("s2").position == 7.0     # parked once
