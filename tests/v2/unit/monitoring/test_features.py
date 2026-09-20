"""Feature extraction pinned against physically-generated traces.

The assertions that matter most are the negative ones: a quiet Gaussian trace
must NOT report telegraph switching, and a clean trace must NOT report spikes.
Those are the false positives that would turn the monitor into an alarm nobody
trusts.

Run:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/monitoring/test_features.py -q
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

import numpy as np
import pytest

from mast.monitoring import features as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import synth_current  # noqa: E402

FS = 20000.0


# ── amplitude / shape ────────────────────────────────────────────────────────

def test_basic_stats_recovers_mean_and_rms():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=2e-12, seed=1)
    st = F.basic_stats(y)
    assert st["mean_a"] == pytest.approx(100e-12, rel=0.05)
    # true RMS includes DC, so it tracks the mean, not the noise
    assert st["rms_a"] == pytest.approx(100e-12, rel=0.05)
    assert st["ptp_a"] > 4 * 2e-12


def test_detrended_rms_matches_injected_noise_and_ignores_drift():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=3e-12,
                      drift_a_per_s=50e-12, seed=2)
    out = F.detrended_rms(y, FS)
    # the 50 pA/s ramp is removed, so RMS reflects the 3 pA noise alone
    assert out["rms_detrended_a"] == pytest.approx(3e-12, rel=0.10)
    assert out["slope_a_per_s"] == pytest.approx(50e-12, rel=0.10)


def test_moments_gaussian_is_flat():
    y = synth_current(fs_hz=FS, dur_s=1.0, white_rms_a=2e-12, seed=3)
    m = F.moments(y)
    assert abs(m["kurtosis"]) < 0.2
    assert abs(m["skewness"]) < 0.1


def test_spikes_are_counted_per_burst_not_per_sample():
    y = synth_current(fs_hz=FS, dur_s=1.0, white_rms_a=1e-12,
                      spikes=4, spike_amp_a=60e-12, seed=4)
    sp = F.spike_metrics(y, FS, k=8.0)
    assert 3 <= sp["spike_count"] <= 8          # bursts, not the ~100 samples each
    assert sp["spike_max_sigma"] > 20


def test_clean_trace_reports_no_spikes():
    y = synth_current(fs_hz=FS, dur_s=1.0, white_rms_a=2e-12, seed=5)
    assert F.spike_metrics(y, FS, k=8.0)["spike_count"] == 0


def test_jump_metrics_flag_a_step():
    y = synth_current(fs_hz=FS, dur_s=1.0, white_rms_a=1e-12, seed=6)
    y[10000:] += 200e-12                        # one abrupt level change
    out = F.jump_metrics(y, FS)
    assert out["jump_count"] >= 1
    assert out["max_step_a"] > 100e-12


def test_jump_detection_does_not_fire_on_clean_noise():
    """The default k is calibrated to this: at k=5 a quiet 1 s segment at 20 kHz
    reports ~6 jumps/s purely from the Gaussian tail, which would sit above any
    sane rate threshold forever."""
    for seed in range(8):
        y = synth_current(fs_hz=FS, dur_s=1.0, white_rms_a=2e-12, seed=200 + seed)
        assert F.jump_metrics(y, FS)["jump_rate_hz"] == 0.0, f"seed {seed}"


def test_jump_detection_still_finds_every_injected_step():
    """The other half of the calibration: the quiet floor must not cost recall."""
    rng = np.random.default_rng(0)
    y = synth_current(fs_hz=FS, dur_s=1.0, white_rms_a=2e-12, seed=7)
    for i in np.sort(rng.choice(int(FS), 20, replace=False)):
        y[i:] += rng.choice([-1.0, 1.0]) * 40e-12
    assert F.jump_metrics(y, FS)["jump_count"] >= 20


# ── saturation / freeze ──────────────────────────────────────────────────────

def test_saturation_fraction_and_rail():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=0.0, white_rms_a=30e-12,
                      line_amp_a=100e-12, sat_rail_a=50e-12, seed=7)
    out = F.saturation_metrics(y, 50e-12)
    assert out["sat_frac"] > 0.1                # hum drives it into the rail
    assert out["railed_frac"] > 0.0             # and holds there in runs


def test_no_saturation_when_well_inside_range():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=2e-12, seed=8)
    assert F.saturation_metrics(y, 90e-9)["sat_frac"] == 0.0


def test_frozen_readout_detected():
    frozen = np.full(2000, 1.234e-12)
    assert F.freeze_metrics(frozen)["frozen"] == 1
    live = synth_current(fs_hz=FS, dur_s=0.1, white_rms_a=1e-12, seed=9)
    assert F.freeze_metrics(live)["frozen"] == 0
    assert F.freeze_metrics(live)["unique_frac"] > 0.9


# ── spectral ─────────────────────────────────────────────────────────────────

def test_pink_noise_gives_inverse_f_slope_near_minus_one():
    """Per-segment, across seeds — the estimator must be usable on ONE segment,
    not only on an ensemble average."""
    for seed in range(6):
        y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=0.0,
                          white_rms_a=0.2e-12, pink_rms_a=20e-12, seed=10 + seed)
        out = F.psd_features([y], FS)
        assert out["inv_f_slope"] is not None
        assert -1.25 < out["inv_f_slope"] < -0.75, f"seed {seed}"
        assert out["inv_f_r2"] > 0.7


def test_white_noise_slope_is_flat():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=0.0, white_rms_a=5e-12, seed=11)
    out = F.psd_features([y], FS)
    assert abs(out["inv_f_slope"]) < 0.4        # no 1/f tilt on white noise


def test_mains_contamination_raises_line_ratio():
    clean = synth_current(fs_hz=FS, dur_s=1.0, mean_a=0.0, white_rms_a=5e-12, seed=12)
    dirty = synth_current(fs_hz=FS, dur_s=1.0, mean_a=0.0, white_rms_a=5e-12,
                          line_amp_a=20e-12, seed=12)
    assert F.psd_features([clean], FS)["line_ratio"] < 10
    assert F.psd_features([dirty], FS)["line_ratio"] > 50
    # the power lands in the mains band, not its neighbours
    d = F.psd_features([dirty], FS)
    assert d["band_45_65_a2"] > d["band_10_45_a2"]


def test_line_ratio_stays_quiet_across_seeds_without_mains():
    """The false-alarm guard. A peak-over-median definition measured p99 ≈ 12 on
    mains-free traces — 5% of healthy segments over a threshold of 10."""
    vals = []
    for seed in range(30):
        y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=2e-12,
                          pink_rms_a=1.5e-12, seed=300 + seed)
        r = F.psd_features([y], FS)["line_ratio"]
        if r is not None:
            vals.append(r)
    assert vals
    assert max(vals) < 8.0, f"noise-only line_ratio reached {max(vals):.1f}"


def test_line_ratio_still_catches_a_small_hum():
    """The other half: quieting it down must not cost sensitivity."""
    for seed in range(5):
        y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=2e-12,
                          pink_rms_a=1.5e-12, line_amp_a=1e-12, seed=400 + seed)
        assert F.psd_features([y], FS)["line_ratio"] > 10


def test_psd_averages_runs_without_splicing_across_a_gap():
    a = synth_current(fs_hz=FS, dur_s=0.5, mean_a=0.0, white_rms_a=5e-12, seed=13)
    b = synth_current(fs_hz=FS, dur_s=0.5, mean_a=0.0, white_rms_a=5e-12, seed=14)
    one = F.psd_features([np.concatenate([a, b])], FS)
    two = F.psd_features([a, b], FS)
    # same underlying process → same white floor whether or not it was split
    assert two["white_floor_a2hz"] == pytest.approx(one["white_floor_a2hz"], rel=0.5)


def test_psd_degrades_on_short_input():
    assert F.psd_features([np.zeros(10)], FS)["inv_f_slope"] is None


# ── RTN ──────────────────────────────────────────────────────────────────────

def test_telegraph_is_detected_with_plausible_rate():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=1e-12,
                      rtn_gap_a=20e-12, rtn_rate_hz=12.0, seed=15)
    out = F.rtn_metrics(y, FS)
    assert out["rtn_score"] > 0.7
    assert out["rtn_gap_a"] == pytest.approx(20e-12, rel=0.3)
    assert 4 < out["rtn_rate_hz"] < 40          # order of magnitude, not exact
    assert out["rtn_transitions"] > 3
    assert out["rtn_dwell_hi_ms"] > 0 and out["rtn_dwell_lo_ms"] > 0


def test_pure_gaussian_does_not_report_telegraph():
    """The false-positive gate: fitting two clusters to one always 'works'."""
    for seed in range(6):
        y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12,
                          white_rms_a=3e-12, seed=100 + seed)
        out = F.rtn_metrics(y, FS)
        assert out["rtn_score"] < 0.2, f"unimodal trace flagged as RTN (seed {seed})"
        assert out["rtn_transitions"] == 0


def test_pink_noise_alone_does_not_report_telegraph():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12,
                      white_rms_a=1e-12, pink_rms_a=15e-12, seed=16)
    assert F.rtn_metrics(y, FS)["rtn_score"] < 0.5


# ── envelope ─────────────────────────────────────────────────────────────────

def test_envelope_shape_and_bounds():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=5e-12, seed=17)
    env = F.envelope(y, FS, buckets_per_s=100)
    assert env.shape[0] == 2
    assert 90 <= env.shape[1] <= 110
    assert env.dtype == np.float32
    assert np.all(env[0] <= env[1])             # min never above max
    assert env[1].max() == pytest.approx(y.max(), rel=1e-5)


def test_envelope_keeps_a_spike_that_averaging_would_lose():
    y = np.zeros(20000)
    y[12345] = 1e-9
    env = F.envelope(y, FS, buckets_per_s=100)
    assert env[1].max() == pytest.approx(1e-9, rel=1e-5)


# ── aggregation contract ─────────────────────────────────────────────────────

def test_compute_segment_features_covers_every_declared_column():
    y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=2e-12,
                      pink_rms_a=3e-12, line_amp_a=1e-12, spikes=1, seed=18)
    feats = F.compute_segment_features([y], FS, F.FeatureParams())
    missing = [c for c in F.FEATURE_COLUMNS if c not in feats]
    assert not missing, f"features missing declared columns: {missing}"


def test_compute_segment_features_empty_input_is_empty_not_a_crash():
    assert F.compute_segment_features([], FS) == {}
    assert F.compute_segment_features([np.array([])], FS) == {}


def test_compute_segment_features_short_segment_degrades_to_none():
    feats = F.compute_segment_features([np.ones(5) * 1e-12], FS)
    assert feats                                # still returns a row
    assert feats["rtn_score"] is None           # but no fabricated verdicts
    assert feats["inv_f_slope"] is None
