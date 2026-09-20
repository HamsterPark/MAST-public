"""``find_peaks_1d``: resonances in a dI/dV curve, and the discipline to find none.

Two failure modes matter more than the peak positions. Finding peaks in noise turns a
quantum corral into whatever the noise happened to do, and merging two close resonances into
one loses the measurement the P3 scenario is scored on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mast.vision.spectral_peaks import find_peaks_1d  # noqa: E402


def lorentzian(x, x0, fwhm, amp=1.0):
    g = fwhm / 2.0
    return amp * g * g / ((x - x0) ** 2 + g * g)


def curve(peaks, *, n=400, lo=-0.5, hi=0.3, noise=0.0, seed=0, background=1.0):
    x = np.linspace(lo, hi, n)
    y = np.full_like(x, background) + 0.3 * (x - lo)
    for x0, fwhm, amp in peaks:
        y = y + lorentzian(x, x0, fwhm, amp)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, n)
    return x, y


def test_two_well_separated_resonances_come_back_in_order():
    x, y = curve([(-0.32, 0.03, 1.0), (-0.12, 0.04, 0.8)], noise=0.01, seed=1)
    res = find_peaks_1d(x, y)
    assert res.verdict == "peaks", res.reasons
    got = [p.energy_ev for p in res.peaks[:2]]
    assert got == sorted(got)
    assert got[0] == pytest.approx(-0.32, abs=0.01)
    assert got[1] == pytest.approx(-0.12, abs=0.01)


def test_two_close_resonances_are_not_merged_into_one():
    """(−0.370, −0.280) is the pair a real corral puts closest together; merging them would
    report one resonance where the paper reports two."""
    x, y = curve([(-0.370, 0.030, 1.0), (-0.280, 0.030, 0.9)], n=600, noise=0.008, seed=2)
    res = find_peaks_1d(x, y)
    assert res.verdict == "peaks"
    got = [p.energy_ev for p in res.peaks]
    assert len(got) >= 2, got
    assert min(got, key=lambda v: abs(v + 0.370)) == pytest.approx(-0.370, abs=0.012)
    assert min(got, key=lambda v: abs(v + 0.280)) == pytest.approx(-0.280, abs=0.012)


def test_pure_noise_yields_no_peaks():
    """The bar rises with the number of points, so a long noisy trace does not manufacture
    resonances out of its own wiggles."""
    for seed in range(6):
        x = np.linspace(-0.5, 0.3, 400)
        y = 1.0 + np.random.default_rng(seed).normal(0.0, 0.02, x.size)
        res = find_peaks_1d(x, y)
        assert res.verdict == "none", (seed, [p.energy_ev for p in res.peaks])


def test_the_noise_scale_is_reported_and_is_about_right():
    x, y = curve([(-0.3, 0.03, 1.0)], noise=0.02, seed=5)
    res = find_peaks_1d(x, y)
    assert res.noise_sigma == pytest.approx(0.02, rel=0.5)


def test_a_window_keeps_peaks_outside_it_from_being_reported():
    x, y = curve([(-0.45, 0.03, 1.0), (-0.10, 0.03, 1.0)], noise=0.005, seed=7)
    res = find_peaks_1d(x, y)
    inside = [p.energy_ev for p in res.peaks if -0.30 <= p.energy_ev <= 0.0]
    assert inside and all(abs(v + 0.10) < 0.02 for v in inside)


def test_negative_polarity_finds_dips():
    x, y = curve([(-0.25, 0.03, -1.0)], noise=0.005, seed=9, background=2.0)
    up = find_peaks_1d(x, y)
    down = find_peaks_1d(x, y, polarity="negative")
    assert down.verdict == "peaks"
    assert down.peaks[0].energy_ev == pytest.approx(-0.25, abs=0.01)
    assert not any(abs(p.energy_ev + 0.25) < 0.01 for p in up.peaks)


def test_a_short_trace_is_undecidable_rather_than_a_guess():
    res = find_peaks_1d([0.0, 0.1, 0.2], [1.0, 2.0, 1.0])
    assert res.verdict == "undecidable" and "too_few_points" in res.reasons


def test_max_peaks_truncates_by_energy_order():
    x, y = curve([(-0.40, 0.02, 1.0), (-0.30, 0.02, 1.0), (-0.20, 0.02, 1.0),
                  (-0.10, 0.02, 1.0)], n=800, noise=0.004, seed=4)
    res = find_peaks_1d(x, y, max_peaks=2)
    assert len(res.peaks) == 2
    assert [p.energy_ev for p in res.peaks] == sorted(p.energy_ev for p in res.peaks)
