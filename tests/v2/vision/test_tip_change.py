"""Mid-scan tip-change detection v2 — mast.vision.tip_change.

v2 (2026-07-27): lag-k differenced row channels, null-library calibrated
against VIGIL physics ground truth (production FPR-1 % operating point;
re-verified: visible-event recall 90 %, negative FPR 0.25 %, score correlation
with the validated prototype 0.9999). The old max-t detector was AUC 0.510
with an 83 % false-alarm rate — these tests pin the v2 semantics.

Synthetic frames here are PHYSICALLY PLAUSIBLE (rotated incommensurate
lattice + noise floor): a perfect axis-aligned sin·sin lattice has whole rows
crossing zero, which produces a pathological adjacent-row-NCC structure no
real instrument produces (see test_pathological_synthetic_is_bounded).
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

from mast.vision.tip_change import cusum_online, detect_tip_change, lod_dc  # noqa: E402


def _lattice(n=160, period=7.3, seed=0, noise=0.05):
    """Rotated, incommensurate, phase-offset lattice + noise — no whole-row
    zero crossings (those are a synthetic artefact real scans don't have)."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    th = 0.3
    x2 = xx * np.cos(th) + yy * np.sin(th)
    y2 = -xx * np.sin(th) + yy * np.cos(th)
    img = np.sin(x2 * 2 * np.pi / period + 0.7) * np.sin(y2 * 2 * np.pi / (period * 1.13) + 1.1)
    return (img + noise * rng.randn(n, n)).astype(np.float32)


# ── detection: the z-offset event mode (the physical main mode on VIGIL) ──

def test_detects_z_offset_change():
    img = _lattice(seed=3)
    img[80:] += 0.5                       # apex length change → row-DC jump
    r = detect_tip_change(img)
    assert r.changed is True
    assert r.change_row is not None and abs(r.change_row - 80) <= 12
    assert r.score > 7.0
    assert r.calib == "vigil-c1"


def test_detects_on_short_early_crop():
    """A scan-monitor crop at the first milestone can be ~64 rows."""
    img = _lattice(64, seed=9)
    img[40:] += 0.6
    r = detect_tip_change(img)
    assert r.changed is True
    assert r.change_row is not None and abs(r.change_row - 40) <= 10


def test_accepts_trace_retrace_pair():
    img = _lattice(128)
    img2 = img.copy()
    img[80:] += 3.0                       # jump on the trace only
    r = detect_tip_change(np.stack([img, img2]))
    assert r.changed is True
    assert "tr" in r.channel_scores       # retrace enabled the tr channel


# ── null behaviour: the failure modes that killed v1 ──

def test_uniform_scan_no_change():
    r = detect_tip_change(_lattice(seed=2))
    assert r.changed is False
    assert r.change_row is None


def test_slow_drift_is_null():
    """v1's root defect: a slow trend grows max-t like √H → 83 % FPR. v2 puts
    slow trends INSIDE the null hypothesis."""
    yy = np.mgrid[0:512, 0:512][0]
    img = _lattice(512, seed=5) + (yy / 512.0) ** 2 * 2.0
    r = detect_tip_change(img)
    assert r.changed is False


def test_pure_noise_no_change():
    r = detect_tip_change(np.random.RandomState(7).randn(256, 256).astype(np.float32))
    assert r.changed is False


def test_flat_image_graceful():
    r = detect_tip_change(np.ones((64, 64), np.float32))
    assert r.changed is False and r.score == 0.0


def test_tiny_image_graceful():
    r = detect_tip_change(np.random.RandomState(0).randn(10, 32).astype(np.float32))
    assert r.changed is False


# ── new v2 surface: LOD, calibration table, threshold semantics ──

def test_lod_reported_in_input_units():
    img = _lattice(seed=2)
    r = detect_tip_change(img)
    assert r.lod is not None and r.lod > 0
    # LOD scales with the noise floor: a 10× noisier frame → ~10× the LOD
    r10 = detect_tip_change(_lattice(seed=2, noise=0.5))
    assert r10.lod is not None and r10.lod > 3 * r.lod
    # helper agrees with the result field
    assert abs(lod_dc(np.asarray(img, np.float64)) - r.lod) / r.lod < 0.3


def test_scale_selects_calibration_table():
    atomic = detect_tip_change(_lattice(seed=2), nm_per_px=0.01)
    meso = detect_tip_change(_lattice(seed=2), nm_per_px=0.2)
    assert atomic.calib == "vigil-c1" and atomic.threshold == 7.0
    assert meso.calib == "vigil-c2" and meso.threshold == 10.0


def test_explicit_threshold_overrides_auto():
    img = _lattice(seed=3)
    img[80:] += 0.5
    r_auto = detect_tip_change(img)
    assert r_auto.changed is True
    r = detect_tip_change(img, threshold=99.0)    # same frame, stricter gate
    assert r.changed is False and r.threshold == 99.0


def test_legacy_margin_frac_arg_accepted():
    r = detect_tip_change(_lattice(seed=2), None, 0.12)   # v1 positional call
    assert r.changed is False


def test_pathological_synthetic_is_bounded():
    """Perfect axis-aligned sin·sin lattice: whole rows cross zero → an NCC
    structure no real scan has. Documented edge: the score may graze the
    threshold, but must stay finite and small — never explode into the
    hundreds the way an unguarded MAD collapse would."""
    yy, xx = np.mgrid[0:160, 0:160]
    img = (np.sin(xx * 2 * np.pi / 8) * np.sin(yy * 2 * np.pi / 8)
           + 0.03 * np.random.RandomState(2).randn(160, 160)).astype(np.float32)
    r = detect_tip_change(img)
    assert np.isfinite(r.score) and r.score < 15.0


# ── online path ──

def test_cusum_separates_event_from_null():
    img = _lattice(seed=3)
    img[80:] += 0.5
    s_event, T = cusum_online(img)
    s_null, _ = cusum_online(_lattice(seed=2))
    assert s_event > 5 * max(s_null, 1.0)
    assert abs(int(T.argmax()) - 80) <= 20


def test_cusum_flat_graceful():
    s, T = cusum_online(np.ones((64, 64), np.float32))
    assert s == 0.0 and T.shape == (64,)


# ── facade ──

def test_vision_module_detect_tip_change_delegates():
    from mast.vision.module import TipChangeResult, VisionModule

    vm = VisionModule(backend="mock")
    img = _lattice(128, seed=4)
    img[70:] += 0.8
    r = vm.detect_tip_change(img, nm_per_px=0.01)
    assert isinstance(r, TipChangeResult)
    assert r.changed is True
