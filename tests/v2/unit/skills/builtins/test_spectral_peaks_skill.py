"""``FindSpectralPeaks``: the IO shell — which column it reads, and what it says when there
is no lock-in channel to read.

The judge is covered in ``tests/v2/unit/vision/test_spectral_peaks.py``.
"""
from __future__ import annotations

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

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.skills.builtins.spectral_peaks import FindSpectralPeaks  # noqa: E402

PEAKS_V = (-0.37, -0.28)


def _curve(v):
    y = np.ones_like(v) + 0.3 * (v - v[0])
    for x0 in PEAKS_V:
        y = y + 0.03 ** 2 / ((v - x0) ** 2 + 0.03 ** 2)
    return y


def _dat(path: Path, *, lockin: bool = True, n: int = 600) -> Path:
    v = np.linspace(-0.5, 0.3, n)
    y = _curve(v)
    if lockin:
        names, cols = "Bias calc (V)\tLI Demod 1 X (A)", (v, y * 1e-12)
    else:
        # only a current column: the shell has to differentiate it and say that it did
        names, cols = "Bias calc (V)\tCurrent (A)", (v, np.cumsum(y) * (v[1] - v[0]) * 1e-12)
    lines = ["Experiment\tbias spectroscopy\t", "X (m)\t1.000000E-08\t",
             "Y (m)\t2.000000E-08\t", "", "[DATA]", names]
    lines += [f"{a:.6E}\t{b:.6E}" for a, b in zip(*cols)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_it_reads_the_lockin_column_and_finds_both_resonances(tmp_path):
    res = FindSpectralPeaks().execute(None, {"dat_path": str(_dat(tmp_path / "a.dat"))})
    assert res.success, res.error
    assert res.data["verdict"] == "peaks"
    assert res.data["didv_source"] == "lockin"
    got = sorted(res.data["energies_mev"])
    for want in PEAKS_V:
        assert min(abs(g - want * 1e3) for g in got) < 12.0, got


def test_without_a_lockin_channel_it_differentiates_the_current_and_says_so(tmp_path):
    res = FindSpectralPeaks().execute(None, {"dat_path": str(_dat(tmp_path / "b.dat",
                                                                  lockin=False))})
    assert res.success and res.data["didv_source"] == "numeric"
    assert res.data["verdict"] in ("peaks", "none")


def test_the_bias_window_is_honoured(tmp_path):
    p = _dat(tmp_path / "c.dat")
    res = FindSpectralPeaks().execute(None, {"dat_path": str(p), "bias_min_v": -0.32,
                                             "bias_max_v": 0.0})
    assert res.success
    assert all(-320.0 <= e <= 0.0 for e in res.data["energies_mev"]), res.data["energies_mev"]


def test_the_position_of_the_spectrum_comes_back_with_the_peaks(tmp_path):
    """A resonance is only a measurement if you know where it was taken."""
    res = FindSpectralPeaks().execute(None, {"dat_path": str(_dat(tmp_path / "d.dat"))})
    assert res.data["dat_x_m"] == pytest.approx(1e-8, rel=1e-6)
    assert res.data["dat_y_m"] == pytest.approx(2e-8, rel=1e-6)


def test_a_missing_file_is_a_failure_not_an_empty_peak_list(tmp_path):
    res = FindSpectralPeaks().execute(None, {"dat_path": str(tmp_path / "nope.dat")})
    assert res.success is False and "不存在" in (res.error or "")


def test_a_window_with_too_few_points_is_undecidable(tmp_path):
    p = _dat(tmp_path / "e.dat")
    res = FindSpectralPeaks().execute(None, {"dat_path": str(p), "bias_min_v": -0.3005,
                                             "bias_max_v": -0.3})
    assert res.success and res.data["verdict"] == "undecidable"
