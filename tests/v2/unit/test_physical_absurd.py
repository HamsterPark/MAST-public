"""Physically-absurd value precheck (#118 deepening).

A physically-impossible magnitude (1.5 A tunnelling setpoint, 1.5 kV bias) is a
unit/exponent slip, not an "out of the configured envelope" violation. It must be
caught FIRST, on both the agent path (SafetyGate.check_global_bounds) and the
composite/manual path (SafetyGuard.check_parameter_bounds), with a message that
tells the model to fix the magnitude — distinct from a normal bounds message
that (correctly) reads as "raise the limit or lower the value".

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_physical_absurd.py -q
"""
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

from mast.agents._shared.safety_mw import SafetyGate
from mast.config import SafetyLimits
from mast.core.safety import SafetyGuard, physically_absurd_violations
from mast.skills.builtins.bias import SetBias
from mast.skills.builtins.zcontrol import SetSetpoint

SP = SetSetpoint().metadata()
BI = SetBias().metadata()


def _is_absurd(msgs) -> bool:
    return any("物理荒谬" in m for m in msgs)


# ── pure function ────────────────────────────────────────────────────────

def test_setpoint_1p5_amp_is_absurd():
    v = physically_absurd_violations(SP, {"setpoint_a": 1.5})
    assert _is_absurd(v)


def test_normal_setpoints_are_plausible():
    for sp in (1e-12, 1e-10, 1.5e-9, 1e-7, 5e-8):
        assert physically_absurd_violations(SP, {"setpoint_a": sp}) == []


def test_milliamp_setpoint_is_absurd():
    # 2 mA at a tunnel junction is impossible (real STM tops out ~100 µA).
    assert _is_absurd(physically_absurd_violations(SP, {"setpoint_a": 2e-3}))


def test_insane_bias_is_absurd():
    # The bias floor is generous (10 kV) — ordinary over-range is the ±10 V
    # bound's job; only a truly-insane hallucination trips the physical floor.
    assert _is_absurd(physically_absurd_violations(BI, {"bias_v": 1e6}))
    # ...and a merely-large 1.5 kV is NOT absurd (left to the bounds layer).
    assert physically_absurd_violations(BI, {"bias_v": 1500.0}) == []


def test_out_of_admin_bounds_but_physical_is_NOT_absurd():
    # 12 V exceeds the ±10 V admin envelope but is physically plausible — it must
    # NOT be flagged absurd (that path is the ordinary bounds check's job).
    assert physically_absurd_violations(BI, {"bias_v": 12.0}) == []


def test_string_value_is_coerced():
    assert _is_absurd(physically_absurd_violations(SP, {"setpoint_a": "1.5"}))


def test_unrelated_param_ignored():
    assert physically_absurd_violations(BI, {"slew_rate_v_per_s": 50.0}) == []


# ── wired into both safety paths, and fires FIRST ────────────────────────

def test_agent_path_reports_absurd_first():
    gate = SafetyGate(SafetyLimits())
    v = gate.check_global_bounds(SP, {"setpoint_a": 1.5})
    assert _is_absurd(v)
    # It stands alone — no confusing "above global safety maximum" pile-on.
    assert all("global safety maximum" not in m for m in v)


def test_composite_path_reports_absurd_first():
    guard = SafetyGuard(SafetyLimits())
    v = guard.check_parameter_bounds(SP, {"setpoint_a": 1.5})
    assert _is_absurd(v)


def test_plausible_but_out_of_bounds_still_uses_normal_bounds_message():
    gate = SafetyGate(SafetyLimits())
    v = gate.check_global_bounds(BI, {"bias_v": 12.0})
    assert v  # still rejected
    assert not _is_absurd(v)  # but via the ordinary bounds path, not absurd
