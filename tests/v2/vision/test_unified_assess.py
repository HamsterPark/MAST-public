"""Unified one-call tip/scan assessment — VisionModule.assess().

Fuses every network-free classical detector (tip_quality + scan_artifacts +
optional I(z)/I(V)) with the learned stm_quality_v1 scorer into one verdict +
sub-results + merged reasons. See docs/v2/benchmarks/vision_v25_diagnostic/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
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


def _vm():
    from mast.vision.module import VisionModule
    return VisionModule(backend="mock")


def _lattice_pair(n=160, period=8, seed=0):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    base = np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period)
    return np.stack([(base + 0.03 * rng.randn(n, n)).astype(np.float32),
                     (base + 0.03 * rng.randn(n, n)).astype(np.float32)])


def _noise_pair(n=160):
    return np.stack([np.random.RandomState(1).randn(n, n),
                     np.random.RandomState(2).randn(n, n)]).astype(np.float32)


def test_good_stable_lattice_overall_good():
    from mast.vision.module import ScanArtifactsResult, TipQualityResult, UnifiedTipAssessment

    r = _vm().assess(_lattice_pair(), scan_size_nm=8.0, use_learned=False)
    assert isinstance(r, UnifiedTipAssessment)
    assert r.overall == "good"
    assert r.reasons == []
    assert isinstance(r.classical, TipQualityResult)
    assert isinstance(r.artifacts, ScanArtifactsResult)
    assert r.artifacts.drift_px == 0.0            # lattice must NOT alias into drift
    assert r.dl_quality_score is None             # use_learned=False


def test_noise_overall_bad():
    r = _vm().assess(_noise_pair(), scan_size_nm=8.0, use_learned=False)
    assert r.overall == "bad"
    assert any("no resolved surface" in x or "unstable" in x for x in r.reasons)


def test_ringing_flagged_as_artifact():
    n = 160
    xx = np.mgrid[0:n, 0:n][1]
    ring = (1.5 * np.sin(xx * 2 * np.pi / 6) + 0.05 * np.random.RandomState(0).randn(n, n)).astype(np.float32)
    r = _vm().assess(ring, use_learned=False)
    assert r.overall == "bad"
    assert r.artifacts.oscillation is True
    assert any("oscillation" in x for x in r.reasons)


def test_real_drift_flagged():
    from scipy import ndimage as ndi
    n = 160
    yy, xx = np.mgrid[0:n, 0:n]
    ap = np.zeros((n, n), np.float32)
    for y, x in zip(np.random.RandomState(3).randint(20, n - 20, 20),
                    np.random.RandomState(4).randint(20, n - 20, 20)):
        ap += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 9.0))
    fwd = ap + 0.03 * np.random.RandomState(5).randn(n, n).astype(np.float32)
    bwd = ndi.shift(ap, (3, 4), mode="constant") + 0.03 * np.random.RandomState(6).randn(n, n).astype(np.float32)
    r = _vm().assess(np.stack([fwd, bwd]).astype(np.float32), use_learned=False)
    assert r.artifacts.drift_px is not None and abs(r.artifacts.drift_px - 5.0) <= 2.0
    assert r.overall == "bad"


def test_spectroscopy_attached():
    z = np.linspace(0, 0.5, 60)
    Icur = np.exp(-21.7 * z) * (1 + 0.01 * np.random.RandomState(0).randn(60))
    V = np.linspace(-1, 1, 101)
    Iv = np.sinh(3 * V) + 0.01 * np.random.RandomState(1).randn(101)
    r = _vm().assess(_lattice_pair(), scan_size_nm=8.0, iz=(z, Icur), iv=(V, Iv), use_learned=False)
    assert r.iz is not None and r.iz.is_clean_exponential is True
    assert r.iv is not None and r.iv.is_stable is True


def test_bad_spectroscopy_makes_reasons():
    z = np.linspace(0, 0.5, 60)
    Icur = np.exp(-21.7 * z).copy()
    Icur[30:] *= 0.2                              # tip jump mid I(z)
    r = _vm().assess(_lattice_pair(), scan_size_nm=8.0, iz=(z, Icur), use_learned=False)
    assert any("I(z)" in x for x in r.reasons)


def test_single_channel_no_drift_signal():
    """A single (H,W) image (no retrace) → drift is None, still assessable."""
    r = _vm().assess(_lattice_pair()[0], scan_size_nm=8.0, use_learned=False)
    assert r.artifacts.drift_px is None
    assert r.overall in ("good", "usable", "bad")


@pytest.mark.skipif(
    os.environ.get("MAST_TEST_QUALITY_MODEL") != "1",
    reason="set MAST_TEST_QUALITY_MODEL=1 (+ stm_quality_v1 weights + backbone cache) to run",
)
def test_learned_channel_end_to_end():
    """use_learned=True actually loads stm_quality_v1 and fills the DL fields."""
    r = _vm().assess(_lattice_pair(), scan_size_nm=8.0, use_learned=True)
    assert r.dl_quality_score is not None
    assert r.dl_quality_tier in ("bad", "marginal", "excellent")
