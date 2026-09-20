"""Synthetic checks for line scoring; no recorded instrument frames are shipped."""
import numpy as np
import pytest

from mast.vision.atomic_lines import (
    ADVISORY_STREAK, MIN_PIXELS, frame_line_advisory, line_score, usable_rows,
)

SCALE = 0.025


def _rows(periodic):
    rng = np.random.default_rng(314159)
    x = np.arange(512) * SCALE
    rows = rng.normal(0, 0.15, (16, 512))
    if periodic:
        rows += 3 * np.sin(2 * np.pi * x / 0.32)
    return rows


def test_periodic_signal_is_stronger_than_broadband_noise():
    noise = frame_line_advisory(_rows(False), SCALE)
    signal = frame_line_advisory(_rows(True), SCALE)
    assert noise["ok"] and signal["ok"]
    assert signal["line_snr_median"] > 5 * noise["line_snr_median"]
    assert signal["period_median_nm"] == pytest.approx(0.32, rel=0.1)


@pytest.mark.parametrize("periodic", [False, True])
def test_line_advice_never_claims_two_dimensional_acceptance(periodic):
    text = frame_line_advisory(_rows(periodic), SCALE)["advisory"]
    assert "建议" in text
    for banned in ("确认", "已达到原子分辨", "判定为", "验收通过"):
        assert banned not in text


def test_short_or_nonfinite_data_is_not_scored_as_zero():
    for values in (np.zeros(MIN_PIXELS - 1), np.full(64, np.nan)):
        result = line_score(values, SCALE)
        assert not result["ok"]
        assert "line_snr" not in result


def test_coarse_pixel_scale_cannot_support_the_frequency_band():
    result = line_score(np.random.default_rng(1).normal(size=64), 5.0)
    assert not result["ok"]


def test_unscanned_rows_are_not_treated_as_measurements():
    assert not usable_rows(np.zeros((5, 64))).any()
    out = frame_line_advisory(np.zeros((5, 64)), SCALE)
    assert not out["ok"]


def test_advice_requires_repeated_observations():
    assert ADVISORY_STREAK > 1
