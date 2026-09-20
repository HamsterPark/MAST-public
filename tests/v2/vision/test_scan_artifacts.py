"""Scan-artifact detection — mast.vision.scan_artifacts.

Feedback oscillation/ringing (on-axis FFT streak vs off-axis lattice), thermal
drift (fwd↔bwd cross-correlation), bad scan-lines / spikes. Network-free.
See docs/v2/benchmarks/vision_v25_diagnostic/.
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

from mast.vision.scan_artifacts import detect_scan_artifacts  # noqa: E402


def _lattice(n=128, period=8, seed=0, noise=0.03):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    return (np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period)
            + noise * rng.randn(n, n)).astype(np.float32)


def _adsorbates(n=128, k=20, seed=1):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), np.float32)
    for y, x in zip(rng.randint(20, n - 20, k), rng.randint(20, n - 20, k)):
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 9.0))
    return img.astype(np.float32)


def _ringing(n=128, period=6, seed=0):
    """Feedback ringing = a coherent ripple along the fast-scan axis on an
    otherwise flat terrace (the canonical scenario)."""
    xx = np.mgrid[0:n, 0:n][1]
    return (1.5 * np.sin(xx * 2 * np.pi / period)
            + 0.05 * np.random.RandomState(seed).randn(n, n)).astype(np.float32)


def test_ringing_detected():
    """A coherent horizontal ripple (feedback oscillation) is flagged."""
    r = detect_scan_artifacts(_ringing())
    assert r.oscillation is True
    assert r.has_artifact is True
    assert r.oscillation_cycles_per_line is not None


def test_clean_lattice_not_ringing():
    """A 2-D lattice puts peaks OFF the axes → not mistaken for ringing."""
    r = detect_scan_artifacts(_lattice(seed=2))
    assert r.oscillation is False


def test_bad_rows_detected():
    img = _lattice()
    img[40] += 8.0
    img[41] -= 8.0
    r = detect_scan_artifacts(img.astype(np.float32))
    assert r.bad_row_frac > 0.0


def test_drift_from_fwd_bwd_offset():
    """Thermal drift = fwd↔bwd registration offset (aperiodic content)."""
    ap = _adsorbates()
    fwd = ap + 0.03 * np.random.RandomState(3).randn(*ap.shape).astype(np.float32)
    bwd = ndi.shift(ap, (3, 4), mode="constant") + 0.03 * np.random.RandomState(4).randn(*ap.shape).astype(np.float32)
    r = detect_scan_artifacts(np.stack([fwd, bwd]).astype(np.float32))
    assert r.drift_px is not None
    assert abs(r.drift_px - 5.0) <= 2.0     # true |(3,4)| = 5


def test_drift_none_without_bwd():
    r = detect_scan_artifacts(_lattice())
    assert r.drift_px is None


def test_clean_image_no_artifact():
    r = detect_scan_artifacts(_adsorbates())
    assert r.has_artifact is False


def test_flat_image_graceful():
    r = detect_scan_artifacts(np.ones((64, 64), np.float32))
    assert r.has_artifact is False


def test_vision_module_detect_scan_artifacts_delegates():
    from mast.vision.module import ScanArtifactsResult, VisionModule

    vm = VisionModule(backend="mock")
    r = vm.detect_scan_artifacts(_ringing())
    assert isinstance(r, ScanArtifactsResult)
    assert r.oscillation is True
