"""instrument_profile (core data + render) tests.

Pins: sanitize (config clamp / choice enum / calib passthrough), holder
round-trip, get_config spec-default fallback, retract dir-code / z-extend-sign
mapping, learned-calibration EWMA vs replace + persist-sink firing,
clear_calibration, and the always-non-empty mechanism block.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/core/test_instrument_profile.py -x -v
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.core import instrument_profile as ip


@pytest.fixture(autouse=True)
def _clean_holder():
    ip.set_persist_sink(None)
    ip.set_profile({})
    yield
    ip.set_persist_sink(None)
    ip.set_profile({})


# ── sanitize ────────────────────────────────────────────────────────────────

def test_sanitize_clamps_config_numbers():
    out = ip.sanitize({"lockin_mod_amp_v": 5.0,        # > 1.0 hi → clamp
                       "retract_step_max": 99999,      # > 1000 hi → clamp
                       "lockin_mod_freq_hz": 973.0})
    assert out["lockin_mod_amp_v"] == 1.0
    assert out["retract_step_max"] == 1000
    assert out["lockin_mod_freq_hz"] == 973.0


def test_sanitize_coerces_int_field():
    out = ip.sanitize({"lockin_signal_index": "8"})
    assert out["lockin_signal_index"] == 8
    assert isinstance(out["lockin_signal_index"], int)


def test_sanitize_drops_unknown_and_junk():
    out = ip.sanitize({"bogus": 1, "lockin_mod_amp_v": "not-a-number",
                       "retract_total_steps": float("nan")})
    assert "bogus" not in out
    assert "lockin_mod_amp_v" not in out
    assert "retract_total_steps" not in out


def test_sanitize_validates_choice_enum():
    assert ip.sanitize({"retract_motor_dir": "z-"})["retract_motor_dir"] == "z-"
    assert ip.sanitize({"retract_motor_dir": "sideways"}) == {}
    assert ip.sanitize({"z_extend_sign": "-1"})["z_extend_sign"] == "-1"


def test_sanitize_passes_calibration_through():
    out = ip.sanitize({"didv_at_contact_v": 1.5e-3, "didv_cal_bias_v": 0.5})
    assert out["didv_at_contact_v"] == pytest.approx(1.5e-3)
    assert out["didv_cal_bias_v"] == pytest.approx(0.5)


def test_sanitize_non_dict_returns_empty():
    assert ip.sanitize(None) == {}
    assert ip.sanitize("x") == {}


# ── holder + get_config ─────────────────────────────────────────────────────

def test_holder_roundtrip():
    stored = ip.set_profile({"retract_motor_dir": "z-", "retract_total_steps": 5000})
    assert stored["retract_motor_dir"] == "z-"
    assert ip.get_profile()["retract_total_steps"] == 5000


def test_get_config_falls_back_to_spec_default():
    # nothing set → spec defaults
    assert ip.get_config("lockin_mod_amp_v") == 0.02
    assert ip.get_config("retract_total_steps") == 3000
    assert ip.get_config("retract_motor_dir") == "z+"
    # unset int with None spec-default → the caller default
    assert ip.get_config("lockin_signal_index", 42) == 42


def test_get_retract_dir_code_maps_direction():
    assert ip.get_retract_dir_code() == 4          # default z+ → 4
    ip.set_profile({"retract_motor_dir": "z-"})
    assert ip.get_retract_dir_code() == 5          # z- → 5


def test_get_z_extend_sign():
    assert ip.get_z_extend_sign() == 1
    ip.set_profile({"z_extend_sign": "-1"})
    assert ip.get_z_extend_sign() == -1


# ── learned calibration ─────────────────────────────────────────────────────

def test_set_calibration_records_and_binds():
    cal = ip.set_calibration(1.2e-3, bias_v=0.5, setpoint_a=1e-10, mod_amp_v=0.02)
    assert cal["didv_at_contact_v"] == pytest.approx(1.2e-3)
    assert cal["didv_cal_bias_v"] == 0.5
    assert cal["didv_cal_setpoint_a"] == 1e-10
    assert cal["didv_cal_mod_amp_v"] == 0.02
    assert "didv_cal_updated_at" in cal


def test_set_calibration_ewma_on_compatible_condition():
    ip.set_calibration(1e-3, bias_v=0.5, setpoint_a=1e-10)
    cal = ip.set_calibration(2e-3, bias_v=0.5, setpoint_a=1e-10)   # same condition
    # EWMA alpha=0.3 → 0.3*2e-3 + 0.7*1e-3 = 1.3e-3
    assert cal["didv_at_contact_v"] == pytest.approx(1.3e-3, rel=1e-6)


def test_set_calibration_replaces_on_different_bias():
    ip.set_calibration(1e-3, bias_v=0.5, setpoint_a=1e-10)
    cal = ip.set_calibration(5e-3, bias_v=1.5, setpoint_a=1e-10)   # bias far off
    assert cal["didv_at_contact_v"] == pytest.approx(5e-3)         # replaced, not EWMA


def test_set_calibration_rejects_nonpositive():
    ip.set_calibration(1e-3, bias_v=0.5, setpoint_a=1e-10)
    cal = ip.set_calibration(0.0, bias_v=0.5, setpoint_a=1e-10)    # ignored
    assert cal["didv_at_contact_v"] == pytest.approx(1e-3)


def test_set_calibration_fires_persist_sink():
    captured = {}
    ip.set_persist_sink(lambda snap: captured.update(snap))
    ip.set_calibration(1.1e-3, bias_v=0.5, setpoint_a=1e-10)
    assert captured.get("didv_at_contact_v") == pytest.approx(1.1e-3)


def test_clear_calibration_keeps_config():
    ip.set_profile({"retract_motor_dir": "z-"})
    ip.set_calibration(1e-3, bias_v=0.5, setpoint_a=1e-10)
    ip.clear_calibration()
    assert ip.get_calibration() == {}
    assert ip.get_config("retract_motor_dir") == "z-"   # config survives


# ── render ──────────────────────────────────────────────────────────────────

def test_format_block_always_nonempty_with_mechanism():
    block = ip.format_profile_block({})
    assert block                       # never empty (mechanism is always relevant)
    assert "退针" in block
    assert "dI/dV" in block
    assert "尚未标定" in block          # no calibration yet


def test_format_block_shows_calibration_when_set():
    ip.set_calibration(1.5e-3, bias_v=0.5, setpoint_a=1e-10)
    block = ip.format_profile_block(ip.get_profile())
    assert "尚未标定" not in block
    assert "1.500 mV" in block         # _fmt_didv(1.5e-3)


# ── free text (2026-07-31, 信号链) ──────────────────────────────────────────

def test_sanitize_keeps_free_text_and_strips_it():
    out = ip.sanitize({"preamp_model": "  FEMTO DLPCA-200  "})
    assert out["preamp_model"] == "FEMTO DLPCA-200"


def test_sanitize_caps_free_text_length():
    out = ip.sanitize({"preamp_model": "x" * 500})
    assert len(out["preamp_model"]) == ip._TEXT_SPEC["preamp_model"][1]


def test_sanitize_drops_empty_and_non_string_text():
    """A dict/list here means the caller sent the wrong thing — str()-ing it would
    put "{'a': 1}" in the prompt as a preamp model, which is worse than nothing."""
    assert "preamp_model" not in ip.sanitize({"preamp_model": "   "})
    assert "preamp_model" not in ip.sanitize({"preamp_model": {"a": 1}})
    assert "preamp_model" not in ip.sanitize({"preamp_model": 42})
    assert "preamp_model" not in ip.sanitize({"preamp_model": None})


def test_get_config_reads_text_keys():
    """get_config only knew CONFIG/CHOICE — a text key would read back None even
    when stored."""
    ip.set_profile({"preamp_model": "SR570"})
    assert ip.get_config("preamp_model") == "SR570"
    ip.set_profile({})
    assert ip.get_config("preamp_model") is None
    assert ip.get_config("preamp_model", "fallback") == "fallback"


def test_text_keys_are_editable_and_in_all_keys():
    assert "preamp_model" in ip.EDITABLE_KEYS
    assert "preamp_model" in ip.ALL_KEYS


def test_preamp_gain_is_clamped_to_a_plausible_range():
    assert ip.sanitize({"preamp_gain_v_per_a": 1e20})["preamp_gain_v_per_a"] == 1e13
    assert ip.sanitize({"preamp_gain_v_per_a": 1.0})["preamp_gain_v_per_a"] == 1e3
    assert ip.sanitize({"preamp_gain_v_per_a": 1e9})["preamp_gain_v_per_a"] == 1e9


def test_bias_applied_to_defaults_to_unknown():
    """NOT "sample": defaulting to the common convention is exactly how a dataset
    gets read backwards without anyone noticing."""
    assert ip.get_config("bias_applied_to") == "unknown"
    assert ip.sanitize({"bias_applied_to": "sample"})["bias_applied_to"] == "sample"
    assert "bias_applied_to" not in ip.sanitize({"bias_applied_to": "somewhere"})


# ── qPlus 实测共振写回 ──────────────────────────────────────────────────────

def test_set_qplus_resonance_stores_and_fires_sink():
    seen = {}
    ip.set_persist_sink(lambda p: seen.update(p))
    ip.set_qplus_resonance(32701.0, 21000.0)
    prof = ip.get_profile()
    assert prof["qplus_f0_measured_hz"] == 32701.0
    assert prof["qplus_q_measured"] == 21000.0
    assert prof["qplus_fq_updated_at"] > 0
    assert seen.get("qplus_f0_measured_hz") == 32701.0


def test_set_qplus_resonance_rejects_junk():
    """A failed sweep returns 0 / NaN — storing that is worse than storing nothing,
    because every later read treats it as the truth."""
    for f0, q in ((0, 100), (-1, 100), (32768, 0), (float("nan"), 100),
                  ("x", 100), (32768, None)):
        ip.set_qplus_resonance(f0, q)
        assert "qplus_f0_measured_hz" not in ip.get_profile(), (f0, q)


# ── 换针:选择性清除 ────────────────────────────────────────────────────────

_TIP_BOUND_SAMPLE = {
    "didv_at_contact_v": 2.5e-3, "didv_cal_bias_v": 0.5,
    "didv_cal_setpoint_a": 1e-10, "didv_cal_mod_amp_v": 0.02,
    "didv_cal_updated_at": 111.0,
    "qplus_amplitude_baseline": 12.0,
    "qplus_f0_measured_hz": 32768.0, "qplus_q_measured": 4000.0,
    "qplus_fq_updated_at": 222.0,
}
_SURVIVORS = {
    "tilt_cal_g11": 1.0, "tilt_cal_g12": 0.0,
    "tilt_cal_g21": 0.0, "tilt_cal_g22": 1.0,
    "tilt_cal_cond": 1.2, "tilt_cal_updated_at": 333.0,
    "qplus_amplitude_signal_index": 7,
    "retract_motor_dir": "z-",
}


def test_get_tip_bound_state_returns_exactly_the_bound_keys():
    ip.set_profile({**_TIP_BOUND_SAMPLE, **_SURVIVORS})
    got = ip.get_tip_bound_state()
    assert set(got) == set(_TIP_BOUND_SAMPLE)


def test_clear_tip_bound_state_spares_tilt_and_signal_index():
    """The whole reason this is not clear_calibration(): tilt response belongs to
    the sample/holder and the signal index is Nanonis wiring. Neither changes
    when you swap the tip, and re-earning the tilt matrix costs a calibration run."""
    ip.set_profile({**_TIP_BOUND_SAMPLE, **_SURVIVORS})
    dropped = ip.clear_tip_bound_state()

    assert set(dropped) == set(_TIP_BOUND_SAMPLE), "returned archive must be complete"
    prof = ip.get_profile()
    for gone in _TIP_BOUND_SAMPLE:
        assert gone not in prof, f"{gone} is bound to the old tip and must be cleared"
    for kept, val in _SURVIVORS.items():
        assert prof[kept] == val, f"{kept} must survive a tip swap"


def test_clear_tip_bound_state_is_a_noop_when_nothing_is_bound():
    ip.set_profile({"retract_motor_dir": "z-"})
    assert ip.clear_tip_bound_state() == {}
    assert ip.get_config("retract_motor_dir") == "z-"


def test_clear_tip_bound_state_persists():
    seen = {}
    ip.set_persist_sink(lambda p: seen.update({"profile": dict(p)}))
    ip.set_profile({**_TIP_BOUND_SAMPLE, **_SURVIVORS})
    ip.clear_tip_bound_state()
    assert "didv_at_contact_v" not in seen["profile"]
    assert seen["profile"]["tilt_cal_g11"] == 1.0


def test_tip_bound_keys_and_clear_calibration_differ_deliberately():
    """clear_calibration() is _CALIB_KEYS-wide (it would take tilt with it) and
    misses the qPlus baseline (a _CONFIG_SPEC key). Pin that they are different
    sets so a future edit cannot quietly collapse them into one."""
    assert "qplus_amplitude_baseline" in ip.TIP_BOUND_KEYS
    assert "qplus_amplitude_baseline" not in ip._CALIB_KEYS
    assert set(ip.TIP_BOUND_KEYS) & {"tilt_cal_g11", "qplus_amplitude_signal_index"} == set()


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
