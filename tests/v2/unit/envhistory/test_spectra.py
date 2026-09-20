"""Spectrum compression and accumulation.

The log-binning is checked against an ANALYTIC spectrum (a 1/f law with a known
exponent) rather than against a golden array: a golden array locks in whatever
the implementation did on the day it was written, including its mistakes.

The accumulator's gates get direct coverage because each one, if wrong, produces
a plausible-looking curve that is quietly wrong: too few segments makes a noisy
median, mixing sample rates averages different frequencies together, and
accepting non-quiet segments records tip-shaping as if it were the machine's
noise floor.
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

import math

import numpy as np

from mast.envhistory.spectra import SpectrumAccumulator, log_bin

FS = 20000.0
NF = 10000


def _analytic(alpha=-1.0, scale=1e-22):
    """PSD = scale · f**alpha on a linear grid, as _psd_of_runs would return."""
    freqs = np.arange(1, NF + 1) * (FS / 2.0 / NF)
    psd = scale * freqs ** alpha
    return freqs.tolist(), psd.tolist()


def test_log_bin_preserves_a_power_law():
    freqs, psd = _analytic(alpha=-1.0)
    f, p = log_bin(freqs, psd, 240)
    assert len(f) >= 100
    # Recover the exponent from the compressed curve.
    slope = np.polyfit(np.log10(f), np.log10(p), 1)[0]
    assert abs(slope - (-1.0)) < 0.02


def test_log_bin_is_spaced_logarithmically():
    freqs, psd = _analytic()
    f, _ = log_bin(freqs, psd, 240)
    ratios = np.diff(np.log10(f))
    # Spacing is uniform in log f except at the sparse low end, where empty
    # bins are dropped rather than interpolated.
    assert np.median(ratios) > 0
    assert f[0] >= 0.5
    assert f[-1] <= FS / 2.0


def test_log_bin_drops_empty_bins_instead_of_interpolating():
    freqs, psd = _analytic()
    f, p = log_bin(freqs, psd, 480)
    assert len(f) == len(p)
    assert len(f) < 480          # low-decade bins are narrower than df
    assert all(math.isfinite(v) for v in p)


def _spike_ratio(probe_hz, factor=1e4, bins=240):
    """How much of a single-bin spike at *probe_hz* survives compression."""
    freqs, psd = _analytic()
    spiked = list(psd)
    spiked[int(round(probe_hz / (FS / 2.0 / NF))) - 1] *= factor
    _, p0 = log_bin(freqs, psd, bins)
    f1, p1 = log_bin(freqs, spiked, bins)
    j = int(np.argmin(np.abs(np.asarray(f1) - probe_hz)))
    return p1[j] / p0[j]


def test_a_real_mains_line_survives_compression():
    """50 Hz must still be VISIBLE in an archived spectrum.

    At 1 Hz resolution a log bin near 50 Hz holds ~2 raw bins, so the in-bin
    median cannot (and must not) average a genuine spectral line away. An
    archived noise spectrum whose whole purpose is to show mains pickup and
    pump resonances would be worthless if compression flattened them.

    (The docstring of spectra.py used to claim the opposite. This test is what
    corrected it.)
    """
    assert _spike_ratio(50.0) > 1e3


def test_isolated_high_frequency_bins_are_smoothed_away():
    """Above a few hundred Hz a bin holds tens to hundreds of raw bins, and
    there the median does its job: single-bin periodogram flukes — ~100%
    variance each — are suppressed so the noise floor is readable."""
    assert _spike_ratio(500.0) < 1.5
    assert _spike_ratio(5000.0) < 1.5


def test_log_bin_ignores_non_finite_samples():
    freqs, psd = _analytic()
    psd[100] = float("nan")
    psd[200] = float("inf")
    f, p = log_bin(freqs, psd, 240)
    assert all(math.isfinite(v) for v in p)


def test_log_bin_refuses_degenerate_input():
    assert log_bin([], [], 240) == ([], [])
    assert log_bin([1.0], [1.0], 240) == ([], [])
    assert log_bin([1.0, 2.0], [1.0, 2.0], 1) == ([], [])


def test_accumulator_needs_min_segments_before_emitting():
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    for i in range(3):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i, bins=240)
    # Window elapsed, but only three segments in hand.
    assert acc.maybe_emit(1000.0 + 2000.0, interval_s=1800.0, min_segments=8) is None
    for i in range(3, 10):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i, bins=240)
    snap = acc.maybe_emit(1000.0 + 2000.0, interval_s=1800.0, min_segments=8)
    assert snap is not None
    assert snap.n_segments == 10
    assert snap.channel == "current"
    assert snap.fs_hz == FS


def test_accumulator_waits_for_the_interval():
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    for i in range(10):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i, bins=240)
    assert acc.maybe_emit(1005.0, interval_s=1800.0, min_segments=8) is None


def test_emitting_clears_the_accumulator():
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    for i in range(10):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i, bins=240)
    acc.maybe_emit(3000.0, interval_s=1800.0, min_segments=8)
    assert acc.n_accum == 0
    assert acc.maybe_emit(9000.0, interval_s=1800.0, min_segments=1) is None


def test_changing_the_sample_rate_clears_the_accumulator():
    """Two frequency grids are not commensurable; a per-point median across
    them would average different frequencies together."""
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    for i in range(5):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i, bins=240)
    assert acc.n_accum == 5
    half = (np.asarray(freqs) / 2.0).tolist()
    acc.add(half, psd, fs_hz=FS / 2.0, ts=1010.0, bins=240)
    assert acc.n_accum == 1


def test_median_across_segments_rejects_an_outlier_segment():
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    loud = (np.asarray(psd) * 1000.0).tolist()
    for i in range(9):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i, bins=240)
    acc.add(freqs, loud, fs_hz=FS, ts=1010.0, bins=240)   # one bad segment
    snap = acc.maybe_emit(3000.0, interval_s=1800.0, min_segments=8)
    _, expected = log_bin(freqs, psd, 240)
    assert np.allclose(snap.psd, expected, rtol=1e-9)


def test_subsampling_gate():
    acc = SpectrumAccumulator("current")
    assert acc.should_accum(1000.0, 10.0) is True
    freqs, psd = _analytic()
    acc.add(freqs, psd, fs_hz=FS, ts=1000.0, bins=240)
    assert acc.should_accum(1005.0, 10.0) is False
    assert acc.should_accum(1010.0, 10.0) is True


def test_accumulator_is_bounded():
    """A very long interval must not let the ring grow without limit."""
    from mast.envhistory.spectra import _MAX_ACCUM
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    for i in range(_MAX_ACCUM + 40):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i, bins=240)
    assert acc.n_accum == _MAX_ACCUM


def test_add_rejects_unusable_input():
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    assert acc.add(freqs, psd, fs_hz=0.0, ts=1.0, bins=240) is False
    assert acc.add([], [], fs_hz=FS, ts=1.0, bins=240) is False
    assert acc.n_accum == 0


def test_snapshot_carries_span_and_context():
    acc = SpectrumAccumulator("z", unit="m^2/Hz")
    freqs, psd = _analytic()
    for i in range(8):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i * 10, bins=240,
                ctx={"ctx_bias_v": 0.5})
    snap = acc.maybe_emit(3000.0, interval_s=1800.0, min_segments=8)
    assert snap.unit == "m^2/Hz"
    assert abs(snap.span_s - 70.0) < 1e-6
    assert snap.ctx["ctx_bias_v"] == 0.5
    assert snap.ctx_stable is True


def test_a_working_point_that_moved_mid_window_is_flagged():
    """一条谱可以横跨半小时 —— 中途改了偏压，它就不再是一个工作点上的测量。

    要害不是「记哪一个值」，是**说不说**。只留一个数的话，一条「-1.2 V」的谱
    可能有一半是在 +0.5 V 下攒的，而没有任何字段能让人发现这件事。
    """
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    for i in range(4):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i * 10, bins=240,
                ctx={"ctx_bias_v": -1.2, "ctx_setpoint_a": 100e-12})
    for i in range(4, 8):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i * 10, bins=240,
                ctx={"ctx_bias_v": 0.5, "ctx_setpoint_a": 100e-12})
    snap = acc.maybe_emit(3000.0, interval_s=1800.0, min_segments=8)
    assert snap.ctx_stable is False
    # 留的是**起点**的值：它和 span_s 说的是同一个起点，两个字段因此描述同一件事。
    assert snap.ctx["ctx_bias_v"] == -1.2


def test_context_churn_that_does_not_change_the_working_point_stays_stable():
    """``ctx_skill`` / ``ctx_scanning`` 一个窗口里本来就会来回变。

    拿它们判的话 ``ctx_stable`` 永远是 False，于是这个字段什么也不再区分 ——
    一个恒为「有问题」的告警和没有这个告警是一回事。
    """
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    for i in range(8):
        acc.add(freqs, psd, fs_hz=FS, ts=1000.0 + i * 10, bins=240,
                ctx={"ctx_bias_v": -1.2, "ctx_skill": f"skill_{i}",
                     "ctx_scanning": bool(i % 2)})
    snap = acc.maybe_emit(3000.0, interval_s=1800.0, min_segments=8)
    assert snap.ctx_stable is True


def test_reset_clears_the_stability_flag():
    """不清的话，一条谱的「变过」会传染给下一条 —— 而下一条可能一直很稳。"""
    acc = SpectrumAccumulator("current")
    freqs, psd = _analytic()
    acc.add(freqs, psd, fs_hz=FS, ts=1000.0, bins=240, ctx={"ctx_bias_v": -1.2})
    acc.add(freqs, psd, fs_hz=FS, ts=1010.0, bins=240, ctx={"ctx_bias_v": 0.5})
    acc.reset()
    for i in range(8):
        acc.add(freqs, psd, fs_hz=FS, ts=2000.0 + i * 10, bins=240,
                ctx={"ctx_bias_v": 0.5})
    snap = acc.maybe_emit(4000.0, interval_s=1800.0, min_segments=8)
    assert snap.ctx_stable is True
