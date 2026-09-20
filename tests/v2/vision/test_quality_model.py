"""stm_quality_v1 — learned quality scorer (mast.vision.quality_model).

Frozen DINOv3 + a ridge head trained on the 778 REAL human labels. On a locked
group-disjoint fold of the real gold set: Spearman 0.684 / AUROC 0.946, vs the
deployed v2.5 head's 0.196 / 0.628 on the same frames. See
docs/v2/benchmarks/vision_v25_diagnostic/ and docs/v2/AGENT_COLLAB.md.

The heavy end-to-end test (loads an 86 MB backbone, ~90 s) is opt-in via
MAST_TEST_QUALITY_MODEL=1 — mirroring the MAST_TEST_M12_REAL pattern — so the
default suite stays fast. Everything else here is pure-numpy and always runs.
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

from mast.vision.quality_model import (  # noqa: E402
    DEFAULT_MODEL,
    QualityScoreResult,
    STMQualityModelMissing,
    STMQualityScorer,
    plane_subtract,
    render_like_real,
)

_ENABLED = os.environ.get("MAST_TEST_QUALITY_MODEL") == "1"
_HEAD = STMQualityScorer().model_path
heavy = pytest.mark.skipif(
    not (_ENABLED and _HEAD.exists()),
    reason="set MAST_TEST_QUALITY_MODEL=1 (and provide the head + backbone cache) to run",
)


def _lattice(n=256, period=8, tilt=0.0, seed=0):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.sin(xx * 2 * np.pi / period) * np.sin(yy * 2 * np.pi / period)
    return (img + tilt * xx + 0.03 * rng.randn(n, n)).astype(np.float32)


# ── modality-matching render (the P0c result) ────────────────────────────────
def test_plane_subtract_removes_a_plane():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    ramp = 3.0 * xx + 2.0 * yy + 7.0
    out = plane_subtract(ramp)
    assert abs(float(out.std())) < 1e-3          # a pure plane flattens to ~0


def test_render_removes_tilt_and_keeps_texture():
    """The whole point: a global tilt must not crush the surface texture."""
    flat = _lattice(tilt=0.0)
    tilted = _lattice(tilt=0.05)
    r_flat, r_tilt = render_like_real(flat), render_like_real(tilted)
    # after plane-flattening the two renders have comparable contrast
    assert abs(float(r_flat.std()) - float(r_tilt.std())) / (float(r_flat.std()) + 1e-9) < 0.35


def test_render_output_range_and_dtype():
    r = render_like_real(_lattice())
    assert r.dtype == np.float32 and r.min() >= 0.0 and r.max() <= 255.0


def test_render_accepts_trace_retrace_and_rgb():
    pair = np.stack([_lattice(seed=1), _lattice(seed=2)])          # (2,H,W)
    assert render_like_real(pair).shape == (256, 256)
    rgb = np.repeat(_lattice(n=64)[:, :, None], 3, axis=2)          # (H,W,3)
    assert render_like_real(rgb).shape == (64, 64)


def test_render_flat_image_is_zeros():
    assert not render_like_real(np.ones((32, 32), np.float32) * 5).any()


def test_render_rejects_bad_shape():
    with pytest.raises(ValueError):
        render_like_real(np.zeros((4, 4, 4, 4), np.float32))


# ── wiring / failure modes (no model load) ───────────────────────────────────
def test_missing_head_raises_clearly(tmp_path):
    sc = STMQualityScorer(model_path=tmp_path / "nope.joblib")
    with pytest.raises(STMQualityModelMissing):
        sc.score(_lattice())


def test_default_head_path_is_the_registered_artifact():
    assert STMQualityScorer().model_path.name == Path(DEFAULT_MODEL).name


def test_result_type_is_frozen_and_serialisable():
    r = QualityScoreResult(score=1.2, tier="excellent", scan_size_nm=8.0)
    assert r.model_dump()["tier"] == "excellent"
    with pytest.raises(Exception):
        r.score = 2.0                                              # frozen


# ── end-to-end (opt-in: loads the backbone) ──────────────────────────────────
@heavy
def test_scores_noise_worse_than_lattice_and_survives_tilt():
    sc = STMQualityScorer()
    s_lat = sc.score(_lattice(tilt=0.05), scan_size_nm=8.0)
    s_noise = sc.score(np.random.RandomState(1).randn(256, 256).astype(np.float32),
                       scan_size_nm=8.0)
    assert isinstance(s_lat, QualityScoreResult)
    assert s_lat.scale_corrected is True
    # the deployed v2.5 head called pure noise "good" — this one must not
    assert s_noise.score < s_lat.score
    assert s_noise.tier == "bad"


@heavy
def test_vision_module_assess_quality_delegates():
    from mast.vision.module import VisionModule

    vm = VisionModule(backend="mock")          # additive: independent of backend
    r = vm.assess_quality(_lattice(tilt=0.05), scan_size_nm=8.0)
    assert isinstance(r, QualityScoreResult) and r.tier in ("bad", "marginal", "excellent")
