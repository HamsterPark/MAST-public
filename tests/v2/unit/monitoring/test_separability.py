"""Do these features actually tell a good tip from a bad one?

Everything else in this directory tests that a function computes what it says.
This file tests the point of the subsystem: the user wants to judge tip health
from the tunnelling current, and if the features cannot separate a stable tip
from an unstable one there is nothing downstream worth building.

Both classes are synthesised from physical mechanisms (see synth.py) rather than
"noisy" vs "clean" — a bad tip here means telegraph switching between two
tunnelling configurations, elevated flicker noise and occasional discharges,
which is what a bad tip actually does.
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
N = 12          # segments per class; enough for a d′ estimate, quick enough to run


def _stable(seed: int) -> np.ndarray:
    """A tip that is behaving: low white noise, mild flicker, nothing else."""
    return synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12,
                         white_rms_a=2e-12, pink_rms_a=1.5e-12, seed=seed)


def _unstable(seed: int) -> np.ndarray:
    """A tip that is not: telegraph switching, heavy flicker, discharges."""
    return synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12,
                         white_rms_a=3e-12, pink_rms_a=6e-12,
                         rtn_gap_a=40e-12, rtn_rate_hz=15.0,
                         spikes=3, spike_amp_a=80e-12, seed=1000 + seed)


@pytest.fixture(scope="module")
def classes():
    good = [F.compute_segment_features([_stable(s)], FS) for s in range(N)]
    bad = [F.compute_segment_features([_unstable(s)], FS) for s in range(N)]
    return good, bad


def _dprime(good: list[dict], bad: list[dict], key: str) -> float:
    g = np.array([f[key] for f in good if f.get(key) is not None], dtype=float)
    b = np.array([f[key] for f in bad if f.get(key) is not None], dtype=float)
    if g.size < 3 or b.size < 3:
        return 0.0
    pooled = float(np.sqrt(0.5 * (g.var() + b.var())))
    return float(abs(b.mean() - g.mean()) / pooled) if pooled > 0 else float("inf")


@pytest.mark.parametrize("key,floor", [
    ("rms_detrended_a", 4.0),      # measured d′ ≈ 16
    ("ptp_a", 3.0),                # ≈ 11
    ("inv_f_slope", 2.0),          # ≈ 9
    ("jump_rate_hz", 2.0),         # ≈ 5
    ("band_1_10_a2", 2.0),         # ≈ 4
])
def test_feature_separates_a_bad_tip_from_a_good_one(classes, key, floor):
    """Each of these carries the signal on its own. The floors are far below the
    measured values so ordinary variation cannot fail the test — what they catch
    is a feature going flat or losing its scale."""
    good, bad = classes
    d = _dprime(good, bad, key)
    assert d > floor, f"{key}: d′={d:.2f}, expected > {floor}"


def test_a_single_threshold_on_noise_classifies_both_classes(classes):
    """The simplest possible rule — is the AC noise above a cut — should already
    work. If it does not, no downstream classifier will either."""
    good, bad = classes
    g = np.array([f["rms_detrended_a"] for f in good])
    b = np.array([f["rms_detrended_a"] for f in bad])
    cut = 0.5 * (g.mean() + b.mean())
    correct = int((g < cut).sum() + (b >= cut).sum())
    assert correct == 2 * N, f"{correct}/{2 * N} correct with a mid-point cut"


def test_the_stable_class_raises_no_alerts(classes):
    """A monitor that flags a healthy tip is one the operator switches off."""
    from mast.monitoring.alerts import AlertEngine
    from mast.monitoring.thresholds import MonitorThresholds

    good, _ = classes
    engine = AlertEngine(lambda: MonitorThresholds())
    verdicts = [engine.evaluate(f).level for f in good]
    assert all(v == "ok" for v in verdicts), f"false alarms on a good tip: {verdicts}"


@pytest.mark.parametrize("spacing_over_noise,expect_detection", [
    (12.0, True),    # comfortably resolved
    (8.0, True),
    (2.0, False),    # the two populations genuinely overlap
])
def test_rtn_detection_follows_its_documented_operating_range(
        spacing_over_noise, expect_detection):
    """Pins the range in rtn_metrics' docstring, in both directions: it must
    find telegraph noise when the levels are resolved, and must NOT guess when
    they are not."""
    noise = 3e-12
    gap = noise * spacing_over_noise
    hits = 0
    for seed in range(8):
        y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12, white_rms_a=2e-12,
                          pink_rms_a=noise, rtn_gap_a=gap, rtn_rate_hz=15.0,
                          seed=5000 + seed)
        if F.rtn_metrics(y, FS)["rtn_score"] >= 0.5:
            hits += 1
    if expect_detection:
        assert hits >= 6, f"only {hits}/8 detected at spacing/noise={spacing_over_noise}"
    else:
        assert hits <= 1, f"{hits}/8 flagged where the levels overlap — guessing"


def test_rtn_never_fires_on_noise_alone_at_any_level():
    """The false-alarm rate that the operating range is bought with."""
    for noise in (1e-12, 3e-12, 8e-12, 12e-12):
        for seed in range(6):
            y = synth_current(fs_hz=FS, dur_s=1.0, mean_a=100e-12,
                              white_rms_a=2e-12, pink_rms_a=noise, seed=6000 + seed)
            score = F.rtn_metrics(y, FS)["rtn_score"]
            assert score < 0.5, f"noise-only trace flagged as RTN (pink={noise}, {score=})"
