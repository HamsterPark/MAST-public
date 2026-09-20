"""Transparent classical good/bad tip verdict — mast.vision.tip_quality.

A fused, interpretable alternative to the deployed opaque coarse label. Fixes its
blind spots by construction: pure noise / no resolved surface → bad; double tip →
bad; mid-scan change → bad; each with a stated reason. See
docs/v2/benchmarks/vision_v25_diagnostic/.
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

from mast.vision.tip_quality import assess_tip_quality_classical as assess  # noqa: E402


def _lattice_pair(n=160, period=8, seed=0):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    base = np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period)
    fwd = (base + 0.03 * rng.randn(n, n)).astype(np.float32)
    bwd = (base + 0.03 * rng.randn(n, n)).astype(np.float32)   # stable: fwd≈bwd
    return np.stack([fwd, bwd])


def test_good_stable_lattice_is_good():
    r = assess(_lattice_pair(), nm_per_px=8 / 160)
    assert r.label == "good"
    assert r.reasons == []
    assert r.has_lattice is True
    assert r.confidence > 0.5


def test_pure_noise_is_bad_with_reasons():
    """The key fix the learned model misses: pure noise must be BAD."""
    n = 160
    noise = np.stack([np.random.RandomState(1).randn(n, n),
                      np.random.RandomState(2).randn(n, n)]).astype(np.float32)
    r = assess(noise, nm_per_px=8 / n)
    assert r.label == "bad"
    assert any("no resolved surface" in x for x in r.reasons)
    assert r.has_lattice is False


def test_double_tip_reported_but_not_a_bad_rule():
    """double_tip was DEMOTED from the BAD rules (2026-07-27): physics-truth
    validation measured AUC 0.442-0.625 — unusable as an alarm (misses ~4 of 5
    at FPR 5 %). The field is still reported for a human who asks, but it must
    never flip the verdict or appear in the reasons.

    FIXTURE STRENGTHENED 2026-08-11. It used to be 160 px / 14 blobs, which sits
    right on the detector's evidence bar (replica significance 6.2 vs a required
    6.0). ``detect_double_tip`` now also demands the replica be present in more
    than one half of the frame before it will claim anything, precisely so that a
    marginal peak stops being reported as a finding — and that turned this
    fixture's answer into a coin flip. 256 px / 30 blobs is the same physics,
    unambiguously detected: 12/12 seeds ``multi_tip`` for the ghost and 12/12
    ``single_tip`` for its parent. The assertions below are unchanged."""
    n = 256
    rng = np.random.RandomState(3)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), np.float32)
    for y, x in zip(rng.randint(28, n - 28, 30), rng.randint(28, n - 28, 30)):
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 9.0))
    img = img + 0.03 * rng.randn(n, n).astype(np.float32)
    dbl = (0.5 * img + 0.5 * ndi.shift(img, (16, 12), order=1, mode="nearest")).astype(np.float32)
    r = assess(dbl, nm_per_px=0.1)
    assert r.is_double is True                      # still reported as data
    assert not any("double" in x for x in r.reasons)  # never a reason
    # the verdict is whatever the REMAINING rules say — is_double alone must
    # not force "bad" (here the frame is clean features, so label is good
    # unless another rule fires; accept either but not via double-tip)
    if r.label == "bad":
        assert r.reasons and all("double" not in x for x in r.reasons)


def test_unstable_fwd_bwd_is_bad():
    n = 160
    base = _lattice_pair(n)[0]
    unstable = np.stack([base, np.random.RandomState(7).randn(n, n).astype(np.float32)])  # retrace ≠ trace
    r = assess(unstable, nm_per_px=8 / n)
    assert r.label == "bad"
    assert any("unstable" in x or "no resolved" in x for x in r.reasons)


def test_reasons_are_human_readable_strings():
    r = assess(np.stack([np.random.RandomState(1).randn(128, 128),
                         np.random.RandomState(2).randn(128, 128)]).astype(np.float32))
    assert all(isinstance(x, str) and len(x) > 0 for x in r.reasons)


def test_vision_module_assess_tip_quality_delegates():
    from mast.vision.module import TipQualityResult, VisionModule

    vm = VisionModule(backend="mock")
    r = vm.assess_tip_quality(_lattice_pair(), nm_per_px=8 / 160)
    assert isinstance(r, TipQualityResult)
    assert r.label == "good"
