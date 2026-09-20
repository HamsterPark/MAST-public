"""Feedback ⑨ (validator half) + the cheap .sxm header reader.

product_validity.assess_array / assess_scan_file are the reusable check behind
the composite product-validity gate: a saved scan that is all-NaN / dead-flat /
unreadable is NOT a usable product even though it "was saved". Thresholds are
conservative — a real noisy scan must always read as valid.
"""
from __future__ import annotations

# ── path bootstrap ──
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

import numpy as np

from mast.io.product_validity import assess_array, assess_scan_file


def _make_sxm(path: Path, fill, nx=8, ny=8) -> Path:
    header = (
        ":SCAN_PIXELS:\n" f"{nx} {ny}\n"
        ":SCAN_OFFSET:\n" "-8.53E-8 9.244E-8\n"
        ":SCAN_RANGE:\n" "5E-8 5E-8\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.0\t0.0\n"
        "\n:SCANIT_END:\n"
    )
    frame = np.asarray(fill, dtype=">f4")
    path.write_bytes(header.encode() + b"\x1a\x04" + frame.tobytes() + frame.tobytes())
    return path


class TestAssessArray:
    def test_real_scan_is_valid(self):
        rng = np.random.default_rng(0)
        v = assess_array(rng.normal(size=(16, 16)))
        assert v["valid"] is True

    def test_all_nan_is_invalid(self):
        v = assess_array(np.full((8, 8), np.nan))
        assert v["valid"] is False and "nan" in v["reason"].lower()

    def test_empty_is_invalid(self):
        v = assess_array(np.array([]))
        assert v["valid"] is False

    def test_dead_flat_is_invalid(self):
        v = assess_array(np.zeros((8, 8)))
        assert v["valid"] is False and "flat" in v["reason"].lower()

    def test_mostly_nan_is_invalid(self):
        a = np.full(1000, np.nan)
        a[0] = 1.0  # 99.9% NaN
        v = assess_array(a)
        assert v["valid"] is False

    def test_a_single_finite_value_is_not_flagged_flat(self):
        # size==1 → constant but not "dead-flat over many samples"
        v = assess_array(np.array([3.0]))
        assert v["valid"] is True

    def test_faint_but_real_signal_stays_valid(self):
        # tiny variance, but nonzero → a real (if quiet) scan, must NOT downgrade
        a = np.zeros((8, 8))
        a[0, 0] = 1e-9
        v = assess_array(a)
        assert v["valid"] is True


class TestAssessScanFile:
    def test_valid_npy(self, tmp_path):
        p = tmp_path / "ok.npy"
        np.save(p, np.random.default_rng(1).normal(size=(8, 8)))
        v = assess_scan_file(p)
        assert v["valid"] is True and v["loaded"] is True

    def test_all_nan_npy_invalid(self, tmp_path):
        p = tmp_path / "nan.npy"
        np.save(p, np.full((8, 8), np.nan))
        v = assess_scan_file(p)
        assert v["valid"] is False and v["loaded"] is True

    def test_missing_file_invalid(self, tmp_path):
        v = assess_scan_file(tmp_path / "gone.npy")
        assert v["valid"] is False and v["loaded"] is False

    def test_crashed_flat_sxm_invalid(self, tmp_path):
        p = _make_sxm(tmp_path / "crash.sxm", np.zeros(64))
        v = assess_scan_file(p)
        assert v["valid"] is False

    def test_real_sxm_valid(self, tmp_path):
        p = _make_sxm(tmp_path / "good.sxm", np.arange(64))
        v = assess_scan_file(p)
        assert v["valid"] is True


class TestReadSxmHeader:
    def test_header_only_reader_gets_channels_and_geometry(self, tmp_path):
        from mast.io.nanonis_files import read_sxm_header, sxm_frame_meta
        p = _make_sxm(tmp_path / "h.sxm", np.arange(64))
        hdr = read_sxm_header(str(p))
        assert hdr.get("channel_names") == ["Z"]
        assert hdr.get("scan_pixels") == [8, 8]
        fm = sxm_frame_meta(hdr)
        assert fm["channels"] == ["Z"]
        assert fm["scan_pixels"] == [8, 8]

    def test_header_reader_is_cheap_and_safe_on_garbage(self, tmp_path):
        from mast.io.nanonis_files import read_sxm_header
        p = tmp_path / "junk.sxm"
        p.write_bytes(b"not really an sxm file, no marker here")
        assert read_sxm_header(str(p)) == {}
