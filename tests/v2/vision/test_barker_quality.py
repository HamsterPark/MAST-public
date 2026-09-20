"""Barker-style tip-quality criteria — mast.vision.barker_quality.

Validated on VIGIL physics truth (GOOD/BAD AUC: CCR 0.638, circularity 0.617
— both above the 0.586 fft_sharpness baseline); these tests pin the
production mechanics: patch extraction, bank bootstrap/persistence, and the
ordering property (a distorted-tip frame scores worse than a good-tip frame).
Tier judgement only — never a radius measurement (see TipMetricsResult).
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

from mast.vision.barker_quality import (  # noqa: E402
    build_template_bank,
    ccr_score,
    circularity_score,
    feature_patches,
    load_bank,
    save_bank,
)


def _dot_frame(n=192, n_dots=12, seed=0, sigma=2.2, elong=1.0):
    """Point adsorbates on a flat background; ``elong`` stretches them the way
    a distorted tip images every feature the same wrong shape."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), np.float64)
    for y, x in zip(rng.randint(24, n - 24, n_dots), rng.randint(24, n - 24, n_dots)):
        img += np.exp(-(((yy - y) / (sigma * elong)) ** 2
                        + ((xx - x) / sigma) ** 2) / 2)
    return (img + 0.02 * rng.randn(n, n)).astype(np.float32)


def test_feature_patches_found_on_dot_frame():
    feats = feature_patches(_dot_frame())
    assert len(feats) >= 5
    p, y, x, amp = feats[0]
    assert p.shape == (24, 24) and amp > 0
    assert abs(float(np.sqrt((p * p).sum())) - 1.0) < 1e-6   # L2-normalised


def test_feature_patches_empty_on_flat():
    assert feature_patches(np.zeros((128, 128))) == []


def test_bank_bootstrap_and_ccr_ordering():
    """Frames imaged by a distorted tip (every feature elongated) must score a
    lower CCR against a good-tip bank than good-tip frames do."""
    good_lists = [feature_patches(_dot_frame(seed=s)) for s in (0, 1, 2)]
    bank = build_template_bank(good_lists, k=5)
    assert 1 <= len(bank) <= 5

    good_scores = [ccr_score(_dot_frame(seed=s), bank)[0] for s in (3, 4)]
    bad_scores = [ccr_score(_dot_frame(seed=s, elong=3.0), bank)[0] for s in (3, 4)]
    assert all(g is not None for g in good_scores)
    assert all(b is not None for b in bad_scores)
    assert min(good_scores) > max(bad_scores)


def test_ccr_none_without_bank():
    assert ccr_score(_dot_frame(), []) == (None, 0)


def test_circularity_round_beats_elongated():
    round_dev, n1 = circularity_score(_dot_frame(seed=5))
    elong_dev, n2 = circularity_score(_dot_frame(seed=5, elong=3.0))
    assert n1 > 0 and n2 > 0
    assert round_dev is not None and elong_dev is not None
    assert round_dev < elong_dev            # smaller = rounder = better tip


def test_bank_save_load_roundtrip(tmp_path):
    bank = build_template_bank([feature_patches(_dot_frame(seed=s)) for s in (0, 1)], k=3)
    p = tmp_path / "bank.npz"
    save_bank(bank, p)
    loaded = load_bank(p)
    assert len(loaded) == len(bank)
    for a, b in zip(bank, loaded):
        assert np.allclose(a, b, atol=1e-6)


def test_load_bank_missing_file_empty():
    assert load_bank("Z:/definitely/not/there.npz") == []


def test_tip_metrics_integration():
    """assess_tip_classical carries the Barker fields; CCR appears only with a
    bank, circularity always (when features exist)."""
    from mast.vision.tip_metrics import assess_tip_classical

    frame = _dot_frame()
    m0 = assess_tip_classical(frame, nm_per_px=0.05)
    assert m0.barker_ccr is None            # no bank given
    assert m0.circularity_dev is not None and m0.circularity_n > 0

    bank = build_template_bank([feature_patches(_dot_frame(seed=1))], k=3)
    m1 = assess_tip_classical(frame, nm_per_px=0.05, template_bank=bank)
    assert m1.barker_ccr is not None and m1.barker_ccr_n > 0


def test_sharpness_scale_gate():
    from mast.vision.tip_metrics import assess_tip_classical

    frame = _dot_frame()
    assert assess_tip_classical(frame, nm_per_px=0.01).sharpness_scale == "full"
    assert assess_tip_classical(frame, nm_per_px=0.03).sharpness_scale == "reduced"
    assert assess_tip_classical(frame, nm_per_px=0.2).sharpness_scale == "off"
    assert assess_tip_classical(frame).sharpness_scale is None
