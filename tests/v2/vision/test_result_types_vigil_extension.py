"""Vision result-type extensions remain backwards-compatible.

The VIGIL extensions (morph/switching/drift/perturbation on TipFineResult,
level/classes/tipflag_*_rle on SegmentationResult, self_consistency on
PartialAssessmentResult, scan_size_nm on TipCoarseResult) are all Optional
with safe defaults so the legacy backend's existing payloads still validate.
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

import pytest  # noqa: E402

from mast.vision.module import (  # noqa: E402
    PartialAssessmentResult,
    SegmentationResult,
    TipCoarseResult,
    TipFineResult,
)


def test_tip_coarse_legacy_payload_validates():
    r = TipCoarseResult(label="good", confidence=0.9)
    assert r.scan_size_nm is None


def test_tip_coarse_vigil_payload_validates():
    r = TipCoarseResult(
        label="bad",
        confidence=0.6,
        embedding_sha="abc123",
        scan_size_nm=50.0,
    )
    assert r.scan_size_nm == 50.0


def test_tip_fine_legacy_payload_validates():
    """LegacyBackend always returns label='unknown' with no extra fields."""
    r = TipFineResult(label="unknown", top2=[], is_usable=True)
    assert r.morph is None
    assert r.switching is None
    assert r.drift is None
    assert r.perturbation is None


def test_tip_fine_vigil_payload_validates():
    r = TipFineResult(
        label="M2",
        top2=[("M2", 0.7), ("M0", 0.2)],
        is_usable=False,
        morph="M2",
        switching=True,
        drift=False,
        perturbation=False,
    )
    assert r.morph == "M2"
    assert r.switching is True


def test_tip_fine_morph_literal_rejects_invalid():
    with pytest.raises(Exception):  # pydantic ValidationError
        TipFineResult(label="x", morph="M5")  # type: ignore[arg-type]


def test_segmentation_legacy_payload_validates():
    r = SegmentationResult(
        mask_rle=b"abc",
        shape=(64, 64),
        class_counts={"TERRACE": 100},
    )
    assert r.level is None
    assert r.classes == []
    assert r.tipflag_stability_rle == b""
    assert r.tipflag_transition_rle == b""


def test_segmentation_l0_payload_validates():
    r = SegmentationResult(
        mask_rle=b"xyz",
        shape=(128, 128),
        class_counts={"TERRACE": 5000, "NOT_TERRACE": 11384},
        level=0,
        classes=["NOT_TERRACE", "TERRACE"],
    )
    assert r.level == 0


def test_segmentation_l2_with_tipflags_validates():
    r = SegmentationResult(
        mask_rle=b"x",
        shape=(64, 64),
        class_counts={"class_00": 10, "class_01": 20},
        level=2,
        classes=["class_00", "class_01"],
        tipflag_stability_rle=b"\x01\x02",
        tipflag_transition_rle=b"\x03\x04",
    )
    assert r.level == 2
    assert r.tipflag_stability_rle == b"\x01\x02"


def test_segmentation_invalid_level_rejected():
    with pytest.raises(Exception):
        SegmentationResult(mask_rle=b"", shape=(1, 1), level=5)  # type: ignore[arg-type]


def test_partial_legacy_payload_validates():
    r = PartialAssessmentResult(
        quality_pred=0.8,
        coarse_label="good",
        frac_acquired=0.5,
    )
    assert r.self_consistency is None


def test_partial_vigil_payload_validates():
    r = PartialAssessmentResult(
        quality_pred=0.7,
        coarse_label="good",
        frac_acquired=0.3,
        self_consistency=0.85,
    )
    assert r.self_consistency == 0.85


def test_partial_self_consistency_clamped():
    """Self-consistency must be in [0, 1]."""
    with pytest.raises(Exception):
        PartialAssessmentResult(
            quality_pred=0.5,
            coarse_label="good",
            frac_acquired=0.5,
            self_consistency=1.2,  # invalid
        )
