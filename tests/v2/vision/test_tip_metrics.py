"""Cheap classical tip-quality signals — mast.vision.tip_metrics.

These repair the learned model's documented blind spots — most importantly a
pure-noise frame scores LOW here (no lattice, high fwd-bwd instability) instead
of the deployed model's "good, Q≈71". See docs/v2/benchmarks/vision_v25_diagnostic/.
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

from mast.vision.tip_metrics import assess_tip_classical  # noqa: E402


def _lattice(n=128, period=8, seed=0, noise=0.03):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period)
    return (img + noise * rng.randn(n, n)).astype(np.float32)


def _noise(n=128, seed=0):
    return np.random.RandomState(seed).randn(n, n).astype(np.float32)


def test_lattice_detected_sharp():
    m = assess_tip_classical(_lattice(), nm_per_px=0.03)
    assert m.has_lattice is True
    assert m.fft_sharpness > 20.0
    assert m.resolution_nm is not None and m.resolution_nm > 0


def test_pure_noise_is_not_good():
    """The key fix: pure noise must NOT look like a sharp, resolved surface."""
    m = assess_tip_classical(_noise(), nm_per_px=0.03)
    assert m.has_lattice is False
    assert m.fft_sharpness < 10.0


def test_noise_much_blunter_than_lattice():
    ml = assess_tip_classical(_lattice(), nm_per_px=0.03)
    mn = assess_tip_classical(_noise(), nm_per_px=0.03)
    assert ml.fft_sharpness > 5.0 * mn.fft_sharpness


def test_stable_fwd_bwd_low_instability():
    fwd = _lattice(seed=1)
    bwd = fwd + 0.02 * np.random.RandomState(9).randn(*fwd.shape).astype(np.float32)
    m = assess_tip_classical(np.stack([fwd, bwd]), nm_per_px=0.03)   # (2,H,W)
    assert m.fwd_bwd_instability is not None
    assert m.fwd_bwd_instability < 0.3


def test_uncorrelated_fwd_bwd_high_instability():
    m = assess_tip_classical(np.stack([_noise(seed=1), _noise(seed=2)]), nm_per_px=0.03)
    assert m.fwd_bwd_instability is not None
    assert m.fwd_bwd_instability > 0.7


def test_stable_tip_with_piezo_offset_stays_low():
    """Real-hardware regression (Agent-B, 2026-07-23): trace/retrace of a STABLE
    tip are laterally offset by piezo hysteresis (~6-7 px, fast axis). A zero-shift
    metric saturates and false-alarms; the shift-tolerant metric must stay low."""
    from scipy import ndimage as ndi
    fwd = _lattice(seed=1)
    bwd = ndi.shift(_lattice(seed=1), (0, 6), mode="reflect").astype(np.float32)  # 6-px fast-axis offset
    m = assess_tip_classical(np.stack([fwd, bwd]), nm_per_px=0.03)
    assert m.fwd_bwd_instability is not None
    assert m.fwd_bwd_instability < 0.3            # stable despite the offset


def test_instability_none_without_bwd():
    m = assess_tip_classical(_lattice(), nm_per_px=0.03)   # single channel
    assert m.fwd_bwd_instability is None


def test_separate_bwd_argument():
    fwd = _lattice(seed=3)
    bwd = fwd.copy()
    m = assess_tip_classical(fwd, bwd=bwd, nm_per_px=0.03)
    assert m.fwd_bwd_instability is not None and m.fwd_bwd_instability < 0.05


def test_flat_image_graceful():
    m = assess_tip_classical(np.ones((64, 64), np.float32))
    assert m.has_lattice is False and m.flatness == 1.0


def _step(n=128, seed=0):
    yy, xx = np.mgrid[0:n, 0:n]
    return ((xx > n // 2) * 5.0 + 0.05 * np.random.RandomState(seed).randn(n, n)).astype(np.float32)


def test_step_edge_resolution_measured():
    m = assess_tip_classical(_step(), nm_per_px=0.1)
    assert m.edge_resolution_px is not None and m.edge_resolution_px > 0
    assert m.edge_resolution_nm is not None
    assert m.n_terrace_levels == 2                 # low + high plateau


def test_noise_has_no_edge_resolution():
    m = assess_tip_classical(_noise(), nm_per_px=0.1)
    assert m.edge_resolution_px is None            # no coherent edge in noise
    assert m.n_terrace_levels == 1


def test_lattice_has_no_step_edge():
    m = assess_tip_classical(_lattice(), nm_per_px=0.03)
    assert m.edge_resolution_px is None            # a flat lattice has no big step


def test_vision_module_assess_tip_classical_delegates():
    from mast.vision.module import TipMetricsResult, VisionModule

    vm = VisionModule(backend="mock")
    m = vm.assess_tip_classical(np.stack([_lattice(), _lattice(seed=1)]), nm_per_px=0.03)
    assert isinstance(m, TipMetricsResult)
    assert m.has_lattice is True
    assert m.n_terrace_levels is not None
