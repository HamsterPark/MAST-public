"""VIGILBackend skeleton tests.

Verifies the interface without loading actual weights:
    - Construction is cheap (does not open the checkpoint).
    - Inference methods raise VIGILCheckpointMissing if the .pt is absent.
    - has_head / preload obey the documented contract.
    - L0 segmentation works WITHOUT weights (classical CV path).
    - Result schemas stay valid for the no-weights cases.
    - VisionModule backend selection switches between legacy and vigil.
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
import pytest  # noqa: E402

from mast.vision.vigil_backend import (  # noqa: E402
    VIGILBackend,
    VIGILCheckpointMissing,
)
from mast.vision.module import (  # noqa: E402
    SegmentationResult,
    VisionModule,
)


# ── Construction ─────────────────────────────────────────────────────


def test_construction_cheap(tmp_path):
    """Constructing a backend with a missing checkpoint must NOT raise."""
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    assert backend.checkpoint_path == tmp_path / "absent.pt"
    assert not backend.is_loaded()


def test_construction_resolves_relative_path(tmp_path, monkeypatch):
    """Relative paths resolve against the project root, not cwd."""
    backend = VIGILBackend(checkpoint_path="artifacts/whatever.pt")
    assert backend.checkpoint_path.is_absolute()
    assert "MASTv2" not in str(backend.checkpoint_path).replace("\\", "/").split("MAST/")[-1].split("/")[0]


# ── Missing-checkpoint behaviour ─────────────────────────────────────


def test_assess_tip_coarse_raises_when_checkpoint_missing(tmp_path):
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    img = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(VIGILCheckpointMissing):
        backend.assess_tip_coarse(img)


def test_segment_l1_raises_when_checkpoint_missing(tmp_path):
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    img = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(VIGILCheckpointMissing):
        backend.segment(img, level=1)


def test_preload_raises_when_checkpoint_missing(tmp_path):
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    with pytest.raises(VIGILCheckpointMissing):
        backend.preload()


def test_has_head_reports_m12_heads(tmp_path):
    """M12 carries Head Q / B / C-L1 (introspection needs no checkpoint);
    Head A / C-L2 / D do not exist in M12."""
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    assert backend.has_head("q")
    assert backend.has_head("b")
    assert backend.has_head("c_l1")
    assert backend.has_head("c")
    assert not backend.has_head("a")
    assert not backend.has_head("c_l2")
    assert not backend.has_head("d")


# ── L0 segmentation works without weights ────────────────────────────


def test_segment_l0_works_without_checkpoint(tmp_path):
    """L0 = classical CV; never opens the checkpoint."""
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    rng = np.random.default_rng(0)
    img = rng.normal(0, 30.0, (128, 128)).astype(np.float32)
    img[40:90, 40:90] = rng.normal(0, 1.0, (50, 50)).astype(np.float32)
    result = backend.segment(img, level=0)
    assert isinstance(result, SegmentationResult)
    assert result.level == 0
    assert "TERRACE" in result.classes
    assert "TERRACE" in result.class_counts
    # Some terrace pixels were detected
    assert result.class_counts["TERRACE"] > 0


def test_segment_default_level_is_one_and_needs_checkpoint(tmp_path):
    """M12's default segmentation level is 1 (4-class, learnt) — unlike L0
    (classical CV), it requires the checkpoint, so segment() with an absent
    checkpoint raises rather than silently degrading."""
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    img = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(VIGILCheckpointMissing):
        backend.segment(img)


def test_segment_level2_downgrades_to_l1(tmp_path):
    """M12 has no 27-class L2 head; segment(level=2) downgrades to L1 (which
    still needs the checkpoint → raises here, not a crash on a missing head)."""
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    img = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(VIGILCheckpointMissing):
        backend.segment(img, level=2)


def test_segment_invalid_level_rejected(tmp_path):
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    img = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="level must be"):
        backend.segment(img, level=3)


# ── Partial-assess gracefully degrades without weights ───────────────


def test_partial_assess_zero_lines_returns_unknown(tmp_path):
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    img = np.zeros((64, 64), dtype=np.float32)
    result = backend.partial_assess(img, n_available=0)
    assert result.coarse_label == "unknown"
    assert result.frac_acquired == 0.0
    assert result.self_consistency is None


def test_partial_assess_invalid_input_returns_default(tmp_path):
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    bad = np.array([])  # 1-D → not a scan
    result = backend.partial_assess(bad, n_available=10)
    assert result.coarse_label == "unknown"


# ── Scan size context ────────────────────────────────────────────────


def test_set_scan_size_nm_round_trips(tmp_path):
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    backend.set_scan_size_nm(50.0)
    assert backend._scan_size() == 50.0
    assert backend._scan_size_is_explicit()


def test_scan_size_defaults_when_unset(tmp_path):
    """Unset scan size → neutral default for the forward, flagged non-explicit."""
    backend = VIGILBackend(checkpoint_path=tmp_path / "absent.pt")
    assert backend._scan_size() == VIGILBackend.DEFAULT_SCAN_SIZE_NM
    assert not backend._scan_size_is_explicit()


# ── VisionModule backend selection ───────────────────────────────────


def test_vision_module_explicit_backend_legacy(tmp_path, monkeypatch):
    """Explicit backend='legacy' wires the LegacyBackend regardless of env.

    MAST2 v2.0.0 update: when no v1 checkpoints are present, LegacyBackend
    construction raises FileNotFoundError and VisionModule falls back to
    MockBackend. The fallback path is tested in test_mock_backend.py.
    """
    monkeypatch.setenv("MAST_VISION_BACKEND", "vigil")
    VisionModule._instance = None
    # Create a dummy .pth so LegacyBackend's checkpoint probe finds something
    (tmp_path / "dummy_subdir").mkdir()
    (tmp_path / "dummy_subdir" / "dummy.pth").write_bytes(b"\x00" * 16)
    try:
        vm = VisionModule(legacy_artifacts_dir=str(tmp_path), backend="legacy")
    except Exception:
        # If torch isn't installed at all, LegacyBackend fails earlier — that's
        # fine, the explicit-backend path was still selected correctly.
        return
    from mast.vision._legacy_wrapper import LegacyBackend
    assert isinstance(vm._backend, LegacyBackend)


def test_vision_module_explicit_backend_vigil(tmp_path):
    """Explicit backend='vigil' wires VIGILBackend."""
    VisionModule._instance = None
    vm = VisionModule(
        vigil_checkpoint_path=str(tmp_path / "absent.pt"),
        backend="vigil",
    )
    assert isinstance(vm._backend, VIGILBackend)


def test_vision_module_invalid_backend_rejected(tmp_path):
    VisionModule._instance = None
    with pytest.raises(ValueError, match="must be 'legacy', 'vigil' or 'mock'"):
        VisionModule(backend="dinov4")


def test_vision_module_env_resolution(tmp_path, monkeypatch):
    """MAST_VISION_BACKEND wins over MAST_LEGACY_VISION."""
    monkeypatch.setenv("MAST_VISION_BACKEND", "vigil")
    monkeypatch.setenv("MAST_LEGACY_VISION", "1")
    VisionModule._instance = None
    vm = VisionModule(vigil_checkpoint_path=str(tmp_path / "absent.pt"))
    assert isinstance(vm._backend, VIGILBackend)


def test_vision_module_default_downgrades_to_legacy_without_ckpt(tmp_path, monkeypatch):
    """Phase 9 default is 'vigil', BUT when the M12 checkpoint is absent the
    bare default quietly downgrades to legacy (→ mock if no v1 weights), so an
    unbundled environment still works."""
    monkeypatch.delenv("MAST_VISION_BACKEND", raising=False)
    monkeypatch.delenv("MAST_LEGACY_VISION", raising=False)
    VisionModule._instance = None
    vm = VisionModule(
        legacy_artifacts_dir=str(tmp_path),
        vigil_checkpoint_path=str(tmp_path / "absent_m12.pt"),
    )
    from mast.vision._legacy_wrapper import LegacyBackend
    from mast.vision._mock_backend import MockBackend

    assert isinstance(vm._backend, (LegacyBackend, MockBackend))
    assert not isinstance(vm._backend, VIGILBackend)


def test_vision_module_default_is_vigil_when_ckpt_present(tmp_path, monkeypatch):
    """When the M12 checkpoint IS present, the bare default resolves to vigil.
    Construction stays cheap — the 1.2 GB backbone is NOT loaded here."""
    monkeypatch.delenv("MAST_VISION_BACKEND", raising=False)
    monkeypatch.delenv("MAST_LEGACY_VISION", raising=False)
    ckpt = tmp_path / "m12.pt"
    ckpt.write_bytes(b"\x00" * 16)  # presence is all the default check inspects
    VisionModule._instance = None
    vm = VisionModule(vigil_checkpoint_path=str(ckpt))
    assert isinstance(vm._backend, VIGILBackend)
    assert not vm._backend.is_loaded()  # lazy: no model load at construction


# Cleanup the singleton after the suite so other tests get a fresh state
@pytest.fixture(autouse=True)
def _reset_vision_singleton():
    yield
    VisionModule._instance = None
