"""Algorithmic double-/multi-tip detection — mast.vision.double_tip.

A double tip is a convolution echo; the detector finds the off-centre replica in
the autocorrelation of the lattice-subtracted residual and recovers the tip
separation. Exact on clean feature-bearing frames (validated AUROC 1.0, 100 %
d-recovery on synthetic ghosts). See docs/v2/benchmarks/vision_v25_diagnostic/.
"""
from __future__ import annotations

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
from scipy import ndimage as ndi  # noqa: E402

from mast.vision.double_tip import detect_double_tip  # noqa: E402


def _ideal(seed, n=192, k=14):
    """Flat terrace + sharp adsorbates — the regime a double tip clearly ghosts."""
    rng = np.random.RandomState(seed)
    img = np.zeros((n, n), np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    for y, x in zip(rng.randint(28, n - 28, k), rng.randint(28, n - 28, k)):
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 3.0 ** 2))
    return img + 0.03 * rng.randn(n, n).astype(np.float32)


def _double(img, a, dy, dx):
    return (1 - a) * img + a * ndi.shift(img, (dy, dx), order=1, mode="reflect")


def test_double_scores_higher_than_single():
    """The robust claim: a double tip scores strictly higher than the single."""
    wins = 0
    for seed in range(8):
        img = _ideal(seed)
        s_single = detect_double_tip(img).score
        s_double = detect_double_tip(_double(img, 0.5, 15, 11)).score
        wins += s_double > s_single + 0.05
    assert wins >= 7          # near-perfect separation on clean images


def test_recovers_separation_vector():
    img = _ideal(3)
    r = detect_double_tip(_double(img, 0.5, 16, 12), nm_per_px=0.1)
    dy, dx = r.separation_px
    assert abs(dy - 16) <= 2 and abs(dx - 12) <= 2      # exact-ish recovery
    assert r.separation_nm is not None
    assert abs(r.separation_nm - np.hypot(16, 12) * 0.1) < 0.3


def test_single_below_threshold_double_above():
    img = _ideal(5)
    assert detect_double_tip(img, threshold=0.18).is_double is False
    assert detect_double_tip(_double(img, 0.5, 14, 13), threshold=0.18).is_double is True


def test_flat_image_returns_not_double():
    r = detect_double_tip(np.ones((64, 64), np.float32))
    assert r.is_double is False and r.score == 0.0


def test_accepts_trace_retrace_pair_shape():
    img = _ideal(7)
    pair = np.stack([_double(img, 0.5, 14, 12), img])   # (2,H,W) → uses forward
    r = detect_double_tip(pair)
    assert r.separation_px[0] >= 0 and r.method == "autocorr"


def test_vision_module_detect_double_tip_delegates():
    from mast.vision.module import DoubleTipResult, VisionModule

    vm = VisionModule(backend="mock")
    r = vm.detect_double_tip(_double(_ideal(1), 0.5, 15, 12))
    assert isinstance(r, DoubleTipResult)
    assert r.is_double is True


def _clean_lattice(n=192, period=8, seed=0):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    return (np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period)
            + 0.03 * rng.randn(n, n)).astype(np.float32)


def test_clean_defectfree_lattice_not_double():
    """A periodic lattice must NOT masquerade as a ghost (Bragg subtraction)."""
    r = detect_double_tip(_clean_lattice())
    assert r.is_double is False


def test_method_variants_run():
    dbl = _double(_ideal(2), 0.5, 15, 12)
    for method in ("autocorr", "cepstrum", "fft", "combined"):
        r = detect_double_tip(dbl, method=method)
        assert r.method in ("autocorr", "cepstrum")
        assert r.score >= 0.0


def test_combined_at_least_autocorr():
    dbl = _double(_ideal(4), 0.5, 16, 11)
    a = detect_double_tip(dbl, method="autocorr").score
    c = detect_double_tip(dbl, method="combined").score
    assert c >= a - 1e-6
