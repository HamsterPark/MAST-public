"""Tunable vision thresholds — zero-regression + holder + leniency tests.

The tip-quality discrimination thresholds moved out of VIGILBackend into
:mod:`mast.vision.thresholds` (adjustable from 设置). This suite locks the
**zero-regression invariant**: with the shipped defaults the pure decision
functions are bit-identical to the historical hard-coded expressions. It also
covers the process-level holder (atomic snapshot swap, thread-safe) and that
loosening a threshold actually flips the expected decisions.

No GPU / checkpoint needed — the pure functions take a raw head dict.
"""
from __future__ import annotations

import sys
import threading
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

from mast.vision.thresholds import (  # noqa: E402
    EDITABLE_KEYS,
    VisionThresholds,
    decide_fine,
    fuse_coarse,
    get_thresholds,
    set_thresholds,
)


# ── golden reference = the OLD hard-coded logic (verbatim) ────────────────────
def _old_fuse_quality(r: dict) -> tuple[str, float]:
    n = float(r["n_p"]); k = float(r["k_p"]); t = float(r["t_p"])
    ar = float(r["s_axis_ratio"]); q = float(r["q_score"])
    clean = (1.0 - n) * (1.0 - k) * (1.0 - t)
    q_norm = min(1.0, max(0.0, (q - 55.0) / 35.0))
    good = 0.55 * clean + 0.30 * q_norm + 0.15 * ar
    good = min(1.0, max(0.0, good))
    label = "good" if good >= 0.5 else "bad"
    return label, (good if label == "good" else 1.0 - good)


def _old_assess_fine(r: dict) -> dict:
    n_p = float(r["n_p"]); t_p = float(r["t_p"]); k_p = float(r["k_p"])
    q = float(r["q_score"]); ar = float(r["s_axis_ratio"])
    multi_tip = bool(n_p > 0.5)
    perturbation = bool(t_p > 0.5)
    contaminated = bool(k_p > 0.5)
    if multi_tip:
        morph = "M2"
    elif contaminated or perturbation:
        morph = "M3"
    elif q >= 72.0 and ar >= 0.6:
        morph = "M0"
    else:
        morph = "M1"
    is_usable = morph in ("M0", "M1") and not (perturbation or multi_tip or contaminated)
    return {
        "morph": morph, "multi_tip": multi_tip, "perturbation": perturbation,
        "contaminated": contaminated, "is_usable": is_usable,
    }


# The OLD hard-coded cuts — now the reference baseline for the zero-regression
# check, because the shipped VisionThresholds() defaults are intentionally
# looser (2026-07-10). q-norm anchors + fusion weights keep their defaults.
_STRICT = VisionThresholds(
    coarse_good_threshold=0.5,
    multi_apex_p_max=0.5,
    instability_p_max=0.5,
    contam_p_max=0.5,
    m0_quality_min=72.0,
    m0_axis_ratio_min=0.6,
)


# Grid straddles every threshold boundary (0.5 for n/k/t, 0.6 for ar, 72 for q).
_PROBS = [0.0, 0.49, 0.5, 0.51, 0.9, 1.0]
_ARS = [0.0, 0.59, 0.6, 0.61, 1.0]
_QS = [40.0, 54.0, 55.0, 71.0, 72.0, 90.0, 100.0]


def _grid():
    for n in _PROBS:
        for k in _PROBS:
            for t in _PROBS:
                for ar in _ARS:
                    for q in _QS:
                        yield {"n_p": n, "k_p": k, "t_p": t,
                               "s_axis_ratio": ar, "q_score": q}


@pytest.fixture(autouse=True)
def _reset_active_thresholds():
    """Restore the process-global holder to defaults after each test."""
    yield
    set_thresholds(None)


# ── zero-regression: the pure fns with the OLD cuts == old hard-coded math ────
def test_fuse_coarse_bit_identical_to_old_at_strict():
    for r in _grid():
        new_label, new_conf = fuse_coarse(r, _STRICT)
        old_label, old_conf = _old_fuse_quality(r)
        assert new_label == old_label, r
        assert new_conf == old_conf, r  # exact — the formula is verbatim


def test_decide_fine_bit_identical_to_old_at_strict():
    for r in _grid():
        assert decide_fine(r, _STRICT) == _old_assess_fine(r), r


def test_q_norm_span_and_weights_unchanged():
    th = VisionThresholds()
    # expert anchors/weights are NOT part of the loosening — still the originals
    assert th.coarse_q_hi - th.coarse_q_lo == 35.0
    assert th.coarse_w_clean + th.coarse_w_quality + th.coarse_w_axis == pytest.approx(1.0)


def test_shipped_defaults_are_the_loosened_set():
    """The shipped defaults are the 2026-07-10 loosened valves (behaviour change
    is intentional, not accidental drift)."""
    th = VisionThresholds()
    assert th.coarse_good_threshold == 0.35
    assert th.multi_apex_p_max == 0.70
    assert th.instability_p_max == 0.70
    assert th.contam_p_max == 0.70
    assert th.m0_quality_min == 62.0
    assert th.m0_axis_ratio_min == 0.45
    # and they ARE looser than the old strict baseline on every valve
    assert th.coarse_good_threshold < _STRICT.coarse_good_threshold
    assert th.multi_apex_p_max > _STRICT.multi_apex_p_max
    assert th.m0_quality_min < _STRICT.m0_quality_min


# ── leniency: loosening a valve flips the decision the intended way ───────────
def test_lower_good_cut_promotes_borderline_bad_to_good():
    # r lands at good≈0.375 → "bad" under the strict 0.5 cut, "good" once lowered.
    r = {"n_p": 0.4, "k_p": 0.0, "t_p": 0.0, "s_axis_ratio": 0.3, "q_score": 55.0}
    assert fuse_coarse(r, _STRICT)[0] == "bad"
    assert fuse_coarse(r, VisionThresholds(coarse_good_threshold=0.30))[0] == "good"


def test_raise_multi_apex_tolerance_stops_rejecting_borderline_multitip():
    r = {"n_p": 0.55, "k_p": 0.0, "t_p": 0.0, "s_axis_ratio": 0.9, "q_score": 90.0}
    d0 = decide_fine(r, _STRICT)
    assert d0["multi_tip"] and d0["morph"] == "M2" and not d0["is_usable"]
    d1 = decide_fine(r, VisionThresholds(multi_apex_p_max=0.65))
    assert not d1["multi_tip"] and d1["is_usable"] and d1["morph"] == "M0"


def test_lower_m0_quality_floor_promotes_m1_to_m0():
    r = {"n_p": 0.0, "k_p": 0.0, "t_p": 0.0, "s_axis_ratio": 0.9, "q_score": 66.0}
    assert decide_fine(r, _STRICT)["morph"] == "M1"
    assert decide_fine(r, VisionThresholds(m0_quality_min=65.0))["morph"] == "M0"


# ── from_mapping: tolerant + clamped ──────────────────────────────────────────
def test_from_mapping_ignores_unknown_and_fills_defaults():
    th = VisionThresholds.from_mapping({"coarse_good_threshold": 0.4, "bogus": 9})
    assert th.coarse_good_threshold == 0.4
    assert th.multi_apex_p_max == VisionThresholds().multi_apex_p_max  # default kept
    assert not hasattr(th, "bogus")


def test_from_mapping_clamps_out_of_range():
    th = VisionThresholds.from_mapping({"coarse_good_threshold": 5.0, "multi_apex_p_max": -3.0})
    assert th.coarse_good_threshold == 1.0
    assert th.multi_apex_p_max == 0.0


def test_from_mapping_rejects_bool_values():
    # bool is a subclass of int — must not be coerced into a threshold.
    th = VisionThresholds.from_mapping({"coarse_good_threshold": True})
    assert th.coarse_good_threshold == VisionThresholds().coarse_good_threshold


def test_empty_mapping_is_defaults():
    assert VisionThresholds.from_mapping({}) == VisionThresholds()
    assert VisionThresholds.from_mapping(None) == VisionThresholds()


def test_editable_keys_are_real_fields():
    names = set(VisionThresholds().to_mapping())
    assert set(EDITABLE_KEYS) <= names
    assert len(EDITABLE_KEYS) == 6


# ── holder: set/get + concurrency ─────────────────────────────────────────────
def test_holder_set_get_roundtrip():
    assert get_thresholds() == VisionThresholds()  # default at start
    snap = set_thresholds({"coarse_good_threshold": 0.4})
    assert get_thresholds() is snap
    assert get_thresholds().coarse_good_threshold == 0.4
    set_thresholds(None)  # reset
    assert get_thresholds() == VisionThresholds()


def test_holder_get_never_sees_partial_update():
    """Concurrent set/get always yields one COMPLETE snapshot (immutable swap)."""
    a = VisionThresholds()
    b = VisionThresholds(coarse_good_threshold=0.4, multi_apex_p_max=0.65,
                         contam_p_max=0.65, instability_p_max=0.65,
                         m0_quality_min=65.0, m0_axis_ratio_min=0.5)
    candidates = {a, b}
    seen: list[VisionThresholds] = []
    stop = threading.Event()

    def setter():
        i = 0
        while not stop.is_set():
            set_thresholds(a if i % 2 == 0 else b)
            i += 1

    def getter():
        while not stop.is_set():
            seen.append(get_thresholds())

    threads = [threading.Thread(target=setter) for _ in range(4)]
    threads += [threading.Thread(target=getter) for _ in range(4)]
    for th in threads:
        th.start()
    threading.Event().wait(0.05)
    stop.set()
    for th in threads:
        th.join()

    assert seen  # got some reads
    for s in seen:
        assert s in candidates  # every read is a whole, valid snapshot
