"""A bound is inclusive to within float noise on BOTH validation paths.

'100n' parses to 1.0000000000000001e-07 and was refused against a 1e-07 ceiling — the value
the operator meant IS the ceiling (2026-09-11, STM-Bench agent trials, SetSetpoint). One part
in 1e9 of slack is far below any physical resolution and keeps a genuinely-over value out.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_bound_float_slack.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.agents._shared.safety_mw import SafetyGate
from mast.config import SafetyLimits
from mast.core.safety import SafetyGuard
from mast.core.si_quantity import parse_quantity
from mast.skills.builtins.zcontrol import SetSetpoint

SP = SetSetpoint().metadata()
CEILING = 100e-9


def test_the_si_string_at_the_ceiling_is_a_hair_above_it_in_float():
    assert parse_quantity("100n", strict=False, what="setpoint_a") > CEILING   # the reason this test exists


def test_guard_accepts_the_ceiling_written_as_an_si_string_and_refuses_over():
    guard = SafetyGuard(SafetyLimits())
    assert guard.check_parameter_bounds(SP, {"setpoint_a": "100n"}) == []
    assert guard.check_parameter_bounds(SP, {"setpoint_a": CEILING}) == []
    over = guard.check_parameter_bounds(SP, {"setpoint_a": "101n"})
    assert over and any("maximum" in m for m in over)


def test_gate_accepts_the_ceiling_written_as_an_si_string_and_refuses_over():
    gate = SafetyGate(SafetyLimits())
    assert gate.check_global_bounds(SP, {"setpoint_a": "100n"}) == []
    over = gate.check_global_bounds(SP, {"setpoint_a": "101n"})
    assert over and any("maximum" in m for m in over)
