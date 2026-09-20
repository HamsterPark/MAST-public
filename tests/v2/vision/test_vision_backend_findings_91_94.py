"""Regression tests for vision-backend findings #91–#94 (2026-05-30).

#91 vigil_backend.assess_tip_coarse: scan_size NaN (operator gave no scale)
    must not crash Pydantic's `ge=0` on TipCoarseResult.scan_size_nm — it is
    surfaced as None.
#92 _mock_backend: a model-less mock must be fail-safe (label='bad',
    is_usable=False, quality_pred=0.0) — never silently auto-approve a tip.
#93 vigil_backend._segment_l2: no phantom class_27, and contiguous model
    channels map to the real v0.4 codec surface codes ({0..14, 16..26}),
    not to raw contiguous indices straddling the reserved code 15.
#94 _legacy_wrapper.segment: non-square input must yield a mask whose shape
    equals the original (h, w), not the square forward-pass tensor.

All tests run WITHOUT torch / network / real checkpoints — heavy paths are
exercised via lightweight stubs / monkeypatch.
"""
from __future__ import annotations

# ── path bootstrap (canonical block) ─────────────────────────────────
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.vision import vigil_backend as vb  # noqa: E402
from mast.vision._legacy_wrapper import (  # noqa: E402
    LegacyBackend,
    _resize_mask_nn,
    _rle_encode,
)
from mast.vision._mock_backend import MockBackend  # noqa: E402
from mast.vision.module import (  # noqa: E402
    SegmentationResult,
    TipCoarseResult,
)


# ═════════════════════════════════════════════════════════════════════
# #91 — NaN scan size must not crash TipCoarseResult validation
# ═════════════════════════════════════════════════════════════════════

def test_tipcoarseresult_rejects_nan_scan_size():
    """Anchor the failure mode: NaN really does fail Pydantic ge=0."""
    with pytest.raises(Exception):
        TipCoarseResult(label="good", confidence=0.5, scan_size_nm=float("nan"))


def _make_vigil_with_stub_infer(monkeypatch, *, scan_nm, infer_result=None):
    """Build an M12 VIGILBackend whose forward (``_infer``) is fully stubbed
    (no torch weights / no backbone load), so we exercise the real
    assess_tip_coarse result-construction logic — including the
    "no explicit scale → scale-dependent fields are None" handling.

    ``scan_nm=None`` means the operator never called ``set_scan_size_nm``."""
    be = vb.VIGILBackend(checkpoint_path="artifacts/__never__.pt")
    be._loaded = True  # skip the real (1.2 GB) load
    if scan_nm is not None:
        be.set_scan_size_nm(scan_nm)
    # v2.5 (ssl_sf09c1) 6-head result dict. Default = a clean/sharp tip:
    # q high, all bad-signals (n/k/t) low, round apex → fusion = good.
    res = infer_result or {
        "q_score": 78.0, "t_p": 0.05, "n_p": 0.05, "k_p": 0.05,
        "s_axis_ratio": 0.90, "s_asym_p": 0.10,
        "seg_map": np.zeros((4, 4), dtype=np.uint8),
        "seg_frac": {}, "cls_sha": "deadbeef",
    }
    monkeypatch.setattr(be, "_infer", lambda image: res)
    return be


def test_assess_tip_coarse_no_scale_nulls_scale_fields_not_crash(monkeypatch):
    """#91: with no operator scale, scale-DEPENDENT M12 fields stay None (the
    v2.5 model has no R_tip estimate at all — both are always None); the fused
    label + the rich per-head signals are still produced and never NaN-crash."""
    be = _make_vigil_with_stub_infer(monkeypatch, scan_nm=None)
    res = be.assess_tip_coarse(np.zeros((8, 8), dtype=np.float32))
    assert isinstance(res, TipCoarseResult)
    assert res.scan_size_nm is None       # not fabricated
    assert res.tip_radius_nm is None      # v2.5 produces no R_tip
    assert res.sharpness_log10 is None    # v2.5 produces no log10(R/scan)
    assert res.quality_score == 78.0      # soft-ordinal Q surfaced
    assert res.multi_apex_prob == 0.05    # rich per-head signal surfaced
    assert res.label == "good"            # clean tip → fused good


def test_assess_tip_coarse_explicit_scale_is_preserved(monkeypatch):
    """#91: a genuine scale is echoed back unchanged (it conditions every head
    via ScaleEmbedding); v2.5 has no R_tip so tip_radius_nm stays None."""
    be = _make_vigil_with_stub_infer(monkeypatch, scan_nm=50.0)
    res = be.assess_tip_coarse(np.zeros((8, 8), dtype=np.float32))
    assert res.scan_size_nm == 50.0
    assert res.tip_radius_nm is None
    assert res.quality_score == 78.0


def test_assess_tip_coarse_bad_when_multi_apex_dominant(monkeypatch):
    """Coarse good/bad: a strong multi-apex signal (N, the #1 STM quality
    killer) drives the fusion to BAD even with a middling quality score."""
    be = _make_vigil_with_stub_infer(monkeypatch, scan_nm=10.0, infer_result={
        "q_score": 65.0, "t_p": 0.10, "n_p": 0.80, "k_p": 0.20,
        "s_axis_ratio": 0.50, "s_asym_p": 0.40,
        "seg_map": np.zeros((4, 4), dtype=np.uint8),
        "seg_frac": {}, "cls_sha": "abad1dea",
    })
    res = be.assess_tip_coarse(np.zeros((8, 8), dtype=np.float32))
    assert res.label == "bad"
    assert res.multi_apex_prob == 0.80
    assert 0.5 < res.confidence <= 1.0


# ═════════════════════════════════════════════════════════════════════
# #92 — Mock backend is fail-safe
# ═════════════════════════════════════════════════════════════════════

def test_mock_coarse_is_fail_safe():
    r = MockBackend().assess_tip_coarse(np.zeros((32, 32), dtype=np.float32))
    assert r.label == "bad"        # never auto-approves
    assert r.confidence == 0.5


def test_mock_fine_not_usable():
    r = MockBackend().assess_tip_fine(np.zeros((32, 32), dtype=np.float32))
    assert r.is_usable is False
    assert r.label == "unknown"


def test_mock_partial_does_not_encourage_early_stop():
    lines = np.zeros((100, 40), dtype=np.float32)
    r = MockBackend().partial_assess(lines, 50)
    assert r.quality_pred == 0.0
    assert r.coarse_label == "unknown"
    assert r.frac_acquired == 0.5


# ═════════════════════════════════════════════════════════════════════
# (#93 removed — it tested the planned 27-class Head C-L2. The M12 model
#  ships only Head C-L1 (4 classes), so vigil_backend has no _segment_l2.
#  The codec reserved-code-15 invariant remains covered by test_mask_codec.py
#  — test_decode_rejects_reserved_surface_code_15 / test_write_surface_rejects_reserved_15.)
# ═════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════
# #94 — Legacy segment restores non-square (h, w) geometry
# ═════════════════════════════════════════════════════════════════════

def test_resize_mask_nn_changes_shape_preserves_labels():
    m = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    out = _resize_mask_nn(m, (4, 6))
    assert out.shape == (4, 6)
    # Nearest-neighbour: only the original label values appear.
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_resize_mask_nn_identity_when_same_shape():
    m = np.array([[1, 0, 1]], dtype=np.uint8)
    out = _resize_mask_nn(m, (1, 3))
    assert out.shape == (1, 3)
    assert np.array_equal(out, m)


def test_legacy_segment_returns_original_nonsquare_shape(monkeypatch):
    """#94: a 96x160 input must yield shape (96, 160), and the RLE must
    decode to exactly that many pixels — not the square forward-pass size."""
    import torch

    be = object.__new__(LegacyBackend)  # bypass checkpoint-probing __init__
    be._device = torch.device("cpu")
    be._models_loaded = True

    h, w = 96, 160

    # Stub the segmenter to return a square (1,1,S,S) logit map.
    class _StubSeg:
        def __call__(self, x):
            s = x.shape[-1]
            out = torch.full((1, 1, s, s), -5.0)
            out[:, :, : s // 2, :] = 5.0  # top half "defect"
            return out

    be._segmenter = _StubSeg()
    monkeypatch.setattr(be, "_ensure_loaded", lambda: None)

    img = np.random.rand(h, w).astype(np.float32)
    res = be.segment(img)

    assert isinstance(res, SegmentationResult)
    assert res.shape == (h, w)                       # original geometry restored
    from mast.vision.seg_utils import decode_rle
    decoded = decode_rle(res.mask_rle, res.shape)
    assert decoded.shape == (h, w)                   # mask covers (h, w) pixels
    total = res.class_counts["TERRACE"] + res.class_counts["POINT_DEFECT_BRIGHT"]
    assert total == h * w


def test_legacy_segment_none_segmenter_already_uses_hw(monkeypatch):
    """When the segmenter is absent the wrapper already returns (h, w);
    keep that path green after the fix."""
    import torch

    be = object.__new__(LegacyBackend)
    be._device = torch.device("cpu")
    be._models_loaded = True
    be._segmenter = None
    monkeypatch.setattr(be, "_ensure_loaded", lambda: None)

    res = be.segment(np.zeros((40, 70), dtype=np.float32))
    assert res.shape == (40, 70)


# ═════════════════════════════════════════════════════════════════════
# Vision-backend confirmed-bug fixes (2026-06-10)
# ═════════════════════════════════════════════════════════════════════

# ── Fix 2: non-tiled L1 segmentation reports the scan's NATIVE resolution,
#    not the 256×256 model-input size (downstream coordinate scaling). ──

def test_segment_l1_reports_native_resolution(monkeypatch):
    """_segment_l1 must map the 256×256 model seg_map back to the input (H, W);
    otherwise RegionMap coordinates are off by the resize ratio."""
    be = vb.VIGILBackend(checkpoint_path="artifacts/__never__.pt")
    be._loaded = True
    # The model always emits a 256×256 seg_map regardless of input size.
    monkeypatch.setattr(be, "_infer", lambda image: {
        "seg_map": np.zeros((256, 256), dtype=np.uint8),
    })
    # Non-tiled path (input below AUTO_TILE_THRESHOLD) on a 300×420 scan.
    res = be.segment(np.zeros((300, 420), dtype=np.float32), tile=0)
    assert res.shape == (300, 420)          # native, NOT (256, 256)
    assert res.level == 1
    assert sum(res.class_counts.values()) == 300 * 420


def test_segment_l1_native_resolution_channel_pair(monkeypatch):
    """Same fix for a (2,H,W) trace/retrace input — geometry comes from H,W."""
    be = vb.VIGILBackend(checkpoint_path="artifacts/__never__.pt")
    be._loaded = True
    monkeypatch.setattr(be, "_infer", lambda image: {
        "seg_map": np.zeros((256, 256), dtype=np.uint8),
    })
    res = be.segment(np.zeros((2, 200, 360), dtype=np.float32), tile=0)
    assert res.shape == (200, 360)


# ── Fix 3: a failed lazy load falls back to MockBackend (does NOT re-attempt
#    the expensive load every call, does NOT throw forever). ──

def test_failed_load_first_call_raises_then_falls_back_to_mock(tmp_path):
    """First inference after a missing-checkpoint load raises (operator signal);
    subsequent inferences degrade to fail-safe Mock results, and the expensive
    load is NOT re-attempted (the cached error is reused)."""
    be = vb.VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    img = np.zeros((32, 32), dtype=np.float32)

    # First call surfaces the real cause.
    with pytest.raises(vb.VIGILCheckpointMissing):
        be.assess_tip_coarse(img)
    assert be._load_error is not None
    assert be._mock is not None            # Mock fallback was built

    # Second call no longer raises — it returns the Mock's fail-safe result.
    r = be.assess_tip_coarse(img)
    assert r.label == "bad"                # MockBackend fail-safe
    assert r.confidence == 0.5

    # Fine + segment + partial all degrade to Mock too (no throw).
    assert be.assess_tip_fine(img).is_usable is False
    seg = be.segment(img)                  # level-1 → mock empty mask
    assert seg.shape == (32, 32)
    pa = be.partial_assess(np.zeros((40, 40), dtype=np.float32), 20)
    assert pa.coarse_label in ("unknown", "bad")


def test_failed_load_does_not_reattempt_load(tmp_path, monkeypatch):
    """Once a load fails, _do_load must NOT be invoked again (no repeated 1.2 GB
    load attempts) — the cached error short-circuits _ensure_loaded."""
    be = vb.VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    img = np.zeros((16, 16), dtype=np.float32)

    calls = {"n": 0}
    real_do_load = be._do_load

    def counting_do_load():
        calls["n"] += 1
        real_do_load()  # still raises VIGILCheckpointMissing

    monkeypatch.setattr(be, "_do_load", counting_do_load)

    with pytest.raises(vb.VIGILCheckpointMissing):
        be.assess_tip_coarse(img)
    # Several more inferences — none of them should re-enter _do_load.
    for _ in range(5):
        be.assess_tip_coarse(img)
    assert calls["n"] == 1


def test_failed_load_segment_channel_pair_mock_shape(tmp_path):
    """Fallback segment on a (2,H,W) frame must collapse to (H,W) before the
    Mock (which reads shape[0]/[1]) so the mask shape is (H,W), not (2,H)."""
    be = vb.VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    pair = np.zeros((2, 48, 64), dtype=np.float32)
    with pytest.raises(vb.VIGILCheckpointMissing):
        be.segment(pair)                   # first call raises, arms fallback
    seg = be.segment(pair)                 # now Mock
    assert seg.shape == (48, 64)           # NOT (2, 48)


# ── Fix 1: artifact resource path resolution covers dev / OTA / frozen. ──

def test_resource_resolution_prefers_existing_meipass(tmp_path, monkeypatch):
    """When the artifact is absent at project_root but present under a frozen
    _MEIPASS resource tree, _resolve_resource picks the _MEIPASS copy."""
    import sys as _sys

    proj = tmp_path / "proj"
    meipass = tmp_path / "meipass"
    rel = Path("MASTv2/artifacts/mast_vision_m12.pt")
    (meipass / rel.parent).mkdir(parents=True)
    (meipass / rel).write_bytes(b"\x00")

    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(proj))
    monkeypatch.setattr(_sys, "_MEIPASS", str(meipass), raising=False)
    resolved = vb.VIGILBackend._resolve_resource(rel)
    assert resolved == meipass / rel


def test_resource_resolution_falls_back_to_project_root(tmp_path, monkeypatch):
    """With no _MEIPASS and nothing on disk, the primary (project_root) candidate
    is returned so the eventual is_file() check reports a meaningful real path."""
    import sys as _sys

    proj = tmp_path / "proj"
    rel = Path("MASTv2/artifacts/mast_vision_m12.pt")
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(proj))
    monkeypatch.delattr(_sys, "_MEIPASS", raising=False)
    resolved = vb.VIGILBackend._resolve_resource(rel)
    assert resolved == proj / rel


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
