"""Test MockBackend behaviour — used as graceful fallback when torch/checkpoints missing."""

from __future__ import annotations

import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

# IMPORTANT: import mast.* at MODULE level so the cached `mast` becomes v2
# before pytest's rootdir-based sys.path injection can swap in v1.
import numpy as np
import pytest

from mast.vision import module as vmod
from mast.vision._mock_backend import MockBackend


def test_mock_backend_assess_tip_coarse():
    # finding #92: a model-less mock must be fail-safe, not report 'good'.
    b = MockBackend(reason="test")
    r = b.assess_tip_coarse(np.zeros((64, 64), dtype=np.float32))
    assert r.label == "bad"
    assert r.confidence == 0.5
    assert r.embedding_sha is None


def test_mock_backend_assess_tip_fine():
    # finding #92: fine assessment with no model is not usable.
    b = MockBackend()
    r = b.assess_tip_fine(np.zeros((64, 64), dtype=np.float32))
    assert r.label == "unknown"
    assert r.top2 == []
    assert r.is_usable is False


def test_mock_backend_segment_shape_propagates():
    b = MockBackend()
    img = np.zeros((128, 192), dtype=np.float32)
    r = b.segment(img)
    assert r.shape == (128, 192)
    assert r.mask_rle == b""
    assert r.class_counts == {"TERRACE": 128 * 192}


def test_mock_backend_partial_assess_fraction():
    b = MockBackend()
    lines = np.zeros((100, 50), dtype=np.float32)
    r = b.partial_assess(lines, 25)
    assert r.frac_acquired == 0.25
    assert r.coarse_label == "unknown"
    # finding #92: fail-safe — no early-stop encouragement from a mock.
    assert r.quality_pred == 0.0


def test_vision_module_fallback_to_mock_when_no_torch(monkeypatch):
    """If LegacyBackend raises on construction, VisionModule should fall back to MockBackend."""
    vmod.VisionModule._instance = None
    monkeypatch.setenv("MAST_VISION_BACKEND", "legacy")
    vm = vmod.VisionModule(legacy_artifacts_dir="nonexistent-models-dir-for-test")
    assert isinstance(vm._backend, MockBackend)
    assert vm._backend_name == "mock"
    r = vm.assess_tip_coarse(np.zeros((64, 64), dtype=np.float32))
    # finding #92: fail-safe mock reports 'bad', never auto-approves the tip.
    assert r.label == "bad"


def test_vision_module_explicit_mock_backend():
    """`backend='mock'` should select MockBackend without trying torch."""
    vmod.VisionModule._instance = None
    vm = vmod.VisionModule(backend="mock")
    assert isinstance(vm._backend, MockBackend)
    assert vm._backend_name == "mock"


def test_resolve_backend_accepts_mock_env(monkeypatch):
    monkeypatch.setenv("MAST_VISION_BACKEND", "mock")
    assert vmod.VisionModule._resolve_backend(None) == "mock"


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError):
        vmod.VisionModule._resolve_backend("foobar")
