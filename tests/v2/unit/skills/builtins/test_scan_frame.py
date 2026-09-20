"""G1 governed wrappers: GrabScanFrameData + CheckScanForCrash.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_scan_frame.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np

from mast.skills.builtins.scan_frame import (
    CheckScanForCrash,
    ComputeDriftVector,
    GrabScanFrameData,
    ParseRegions,
)


class _Rec:
    def __init__(self, error="", return_value=None):
        self.error = error
        self.return_value = return_value


class FakeCtx:
    """Mock ExecutionContext with safe_call for Scan_FrameDataGrab.

    frames: {(channel, direction): samples_list | None | "error"}.
    """
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def safe_call(self, method, *args):
        self.calls.append((method, args))
        if method == "Scan_FrameDataGrab":
            ch, direction = int(args[0]), int(args[1])
            val = self.frames.get((ch, direction), None)
            if val == "error":
                return _Rec(error="read failed", return_value=None)
            if val is None:
                return _Rec(error="", return_value=None)
            return _Rec(error="", return_value=("", None, list(val)))
        return _Rec(error="", return_value=None)


# ── GrabScanFrameData ───────────────────────────────────────────────────
class TestGrabScanFrameData:
    def test_grab_saves_npy(self, tmp_path):
        out = tmp_path / "fwd.npy"
        ctx = FakeCtx({(0, 1): [1.0, 2.0, 3.0]})
        res = GrabScanFrameData().execute(ctx, {
            "channel_index": 0, "direction": 1, "save_path": str(out)})
        assert res.success
        assert res.data["frame_path"] == str(out)
        assert res.data["n_samples"] == 3
        assert out.exists()
        assert list(np.load(out)) == [1.0, 2.0, 3.0]
        assert ctx.calls == [("Scan_FrameDataGrab", (0, 1))]

    def test_no_data_fails(self, tmp_path):
        ctx = FakeCtx({(0, 1): None})
        res = GrabScanFrameData().execute(ctx, {
            "channel_index": 0, "direction": 1, "save_path": str(tmp_path / "x.npy")})
        assert res.success is False
        assert "no usable samples" in res.error

    def test_read_error_fails(self, tmp_path):
        ctx = FakeCtx({(0, 1): "error"})
        res = GrabScanFrameData().execute(ctx, {
            "channel_index": 0, "direction": 1, "save_path": str(tmp_path / "x.npy")})
        assert res.success is False
        assert "read failed" in res.error


# ── CheckScanForCrash ───────────────────────────────────────────────────
class TestCheckScanForCrash:
    def test_ok_when_channels_have_variance(self):
        ctx = FakeCtx({(0, 1): [1.0, 2.0, 3.0], (14, 1): [4.0, 5.0, 6.0]})
        res = CheckScanForCrash().execute(ctx, {"channels": "0,14"})
        assert res.success
        assert res.data["crash_indicator"] is False
        assert res.data["status"] == "ok"
        assert res.data["per_channel"] == {"ch0": "ok", "ch14": "ok"}

    def test_crash_on_flat_channel(self):
        # ch0 looks plausible but Z (ch14) is flattened → crash via Z.
        ctx = FakeCtx({(0, 1): [1.0, 2.0, 3.0], (14, 1): [5.0, 5.0, 5.0]})
        res = CheckScanForCrash().execute(ctx, {"channels": "0,14"})
        assert res.data["crash_indicator"] is True
        assert res.data["status"] == "crash"
        assert res.data["crash_channel"] == "ch14"
        assert res.data["per_channel"]["ch14"] == "crash"

    def test_crash_on_nan(self):
        ctx = FakeCtx({(0, 1): [1.0, float("nan"), 3.0]})
        res = CheckScanForCrash().execute(ctx, {"channels": "0"})
        assert res.data["crash_indicator"] is True
        assert res.data["status"] == "crash"

    def test_skipped_when_nothing_readable(self):
        # No channel returns usable data → inconclusive, never "ok".
        ctx = FakeCtx({(0, 1): None, (14, 1): None})
        res = CheckScanForCrash().execute(ctx, {"channels": "0,14"})
        assert res.success
        assert res.data["crash_indicator"] is False
        assert res.data["status"] == "skipped"

    def test_default_channels(self):
        ctx = FakeCtx({(0, 1): [1.0, 2.0], (14, 1): [3.0, 4.0]})
        res = CheckScanForCrash().execute(ctx, {})
        assert res.data["channels_probed"] == [0, 14]
        assert res.data["status"] == "ok"


# ── ComputeDriftVector ──────────────────────────────────────────────────
class TestComputeDriftVector:
    def test_zero_drift_for_identical_frames(self, tmp_path):
        # Odd-sized image with a central peak → autocorrelation peak sits exactly
        # on the shape//2 center, so identical frames give exactly zero drift.
        ref = np.zeros((5, 5), dtype=float)
        ref[2, 2] = 10.0
        refp = tmp_path / "ref.npy"
        np.save(refp, ref)
        ctx = FakeCtx({(0, 1): list(ref.flatten())})  # current == reference
        res = ComputeDriftVector().execute(ctx, {"ref_path": str(refp), "scan_width_m": 5e-9})
        assert res.success
        assert res.data["drift_x_m"] == 0.0
        assert res.data["drift_y_m"] == 0.0

    def test_size_mismatch_returns_zero(self, tmp_path):
        refp = tmp_path / "ref.npy"
        np.save(refp, np.zeros((4, 4)))
        ctx = FakeCtx({(0, 1): [1.0, 2.0, 3.0]})  # 3 != 16
        res = ComputeDriftVector().execute(ctx, {"ref_path": str(refp), "scan_width_m": 4e-9})
        assert res.success
        assert res.data["drift_x_m"] == 0.0
        assert "mismatch" in res.data["note"]

    def test_missing_reference_fails(self, tmp_path):
        ctx = FakeCtx({(0, 1): [1.0]})
        res = ComputeDriftVector().execute(ctx, {"ref_path": str(tmp_path / "nope.npy"), "scan_width_m": 1e-9})
        assert res.success is False
        assert "reference" in res.error


# ── ParseRegions ────────────────────────────────────────────────────────
class TestParseRegions:
    def test_valid_normalised(self):
        import json
        regions = json.dumps([
            {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9, "height_m": 50e-9},
            {"center_x_m": 1e-7, "center_y_m": 0.0, "width_m": 50e-9, "height_m": 50e-9,
             "angle_deg": 30, "label": "spot"},
        ])
        res = ParseRegions().execute(None, {"regions": regions})
        assert res.success
        assert res.data["count"] == 2
        assert res.data["regions"][0]["angle_deg"] == 0.0  # default added
        assert res.data["regions"][0]["label"] == "R1"     # default label
        assert res.data["regions"][1]["label"] == "spot"

    def test_bad_json_fails(self):
        res = ParseRegions().execute(None, {"regions": "{not json"})
        assert res.success is False
        assert "invalid" in res.error

    def test_missing_field_fails(self):
        import json
        res = ParseRegions().execute(None, {"regions": json.dumps([{"center_x_m": 0.0}])})
        assert res.success is False

    def test_out_of_range_center_fails(self):
        import json
        res = ParseRegions().execute(None, {"regions": json.dumps(
            [{"center_x_m": 1.0, "center_y_m": 0, "width_m": 50e-9, "height_m": 50e-9}])})
        assert res.success is False
        assert "out of range" in res.error


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
