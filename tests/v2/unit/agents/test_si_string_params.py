"""Dimensioned parameters travel as SI strings — and every guard still sees them.

WHY THE TYPE CHANGED
====================
Measured on the real path (kimi-k3, tool calling, tool_choice=auto, 12 trials
each, 2026-08-04)::

    number-typed tool argument    0/12 — 3e-12 arrived as 3, 1.5e-10 as 1.5
    string-typed tool argument   12/12 — byte-identical

Not a preference: on the channel every hardware parameter uses, numbers do not
survive at all. The conversion happens once, in ``skill_adapter``, so no skill
and no composite caller had to change.

THE PART THAT IS EASY TO GET WRONG
==================================
Every safety check that did ``float(value)`` and ``continue``d on failure would
now SKIP a string instead of checking it — a change that turns a fix into four
new holes, silently, because nothing errors. Half this file is about that.
(既有教训: "看着在防护、其实没有".)

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/agents/test_si_string_params.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
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

from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from mast.agents._shared.skill_adapter import _si_params, wrap_skill
from mast.core.safety import (
    SafetyGuard,
    physically_absurd_violations,
)
from mast.core.si_quantity import needs_strict_prefix
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.bias import SetBias
from mast.skills.builtins.zcontrol import SetSetpoint
from mast.skills.builtins.zctrl_gain import SetZCtrlGain


@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        entry = self.canned.get(method)
        if entry is not None:
            return NanonisCallRecord(method=method, args=args,
                                     return_value=entry.get("return_value"),
                                     error=entry.get("error", ""))
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def _provider(canned=None):
    canned = canned or {}
    instances: list[FakeCtx] = []

    def provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    return provider, instances


def _props(cls) -> dict:
    return convert_to_openai_tool(
        wrap_skill(cls, _provider()[0])
    )["function"]["parameters"]["properties"]


# ── Who participates ─────────────────────────────────────────────────────────

def test_a_dimensioned_float_is_a_string():
    assert _props(SetSetpoint)["setpoint_a"]["type"] == "string"
    assert _props(SetBias)["bias_v"]["type"] == "string"


def test_a_dimensionless_float_is_left_alone():
    """No unit means no exponent to lose — quoting it would be noise."""
    from mast.skills.builtins.atom_track import ConfigureAtomTrack
    p = _props(ConfigureAtomTrack)["integral_gain"]
    assert p["type"] == "number", "a dimensionless gain does not need quoting"


def test_counts_and_flags_are_left_alone():
    from mast.skills.builtins.motor import MotorMove
    from mast.skills.builtins.zcontrol import ZControllerOnOff
    assert _props(MotorMove)["steps"]["type"] == "integer"
    assert _props(ZControllerOnOff)["enable"]["type"] == "boolean"


def test_an_enum_stays_an_enum():
    """allowed_values is already exact; a prefix would add nothing."""
    from mast.skills.builtins.motor import MotorMove
    assert "enum" in _props(MotorMove)["direction"]


# ── Strict vs lenient ────────────────────────────────────────────────────────

@pytest.mark.parametrize("lo,hi,strict", [
    (0.0, 1e-6, True),        # p_gain — 1 is never a legitimate metre-gain
    (1e-12, 1e-7, True),      # setpoint_a
    (-1.5e-6, 1.5e-6, True),  # a piezo coordinate
    (-10.0, 10.0, False),     # bias_v — "-2" is exactly right
    (1.0, 1800.0, False),     # a timeout in seconds
    (0.0, None, False),       # unknown ceiling is not evidence
    (None, None, False),
])
def test_the_prefix_is_demanded_only_where_a_bare_number_is_impossible(lo, hi, strict):
    assert needs_strict_prefix(lo, hi) is strict


def test_the_gains_are_strict_and_the_bias_is_not():
    assert _si_params(SetZCtrlGain().metadata())["p_gain"] is True
    assert _si_params(SetBias().metadata())["bias_v"] is False


def test_a_lenient_parameter_accepts_the_natural_form():
    canned = {"Bias_Set": {"return_value": None},
              "Bias_Get": {"return_value": ("", b"", [-2.0])}}
    provider, instances = _provider(canned)
    tool = wrap_skill(SetBias, provider)
    tool.func(tool_call_id="t", state={}, bias_v="-2")
    assert any(c[0] == "Bias_Set" for c in instances[-1].calls)


# ── The guards must still see the value ──────────────────────────────────────
#
# These are the tests that would have caught turning the fix into a hole.

def test_the_physical_absurdity_table_reads_si_strings():
    """It used to float() and `continue` — which would skip every string."""
    meta = SetZCtrlGain().metadata()
    assert physically_absurd_violations(meta, {"p_gain": "3"}), (
        "a 3-metre gain written as text walked past the absurdity table")
    assert physically_absurd_violations(meta, {"p_gain": "3000m"})
    assert not physically_absurd_violations(meta, {"p_gain": "3p"})


def test_the_global_envelope_reads_si_strings():
    from mast.config import SafetyLimits
    guard = SafetyGuard(SafetyLimits())
    meta = SetSetpoint().metadata()
    assert guard.check_parameter_bounds(meta, {"setpoint_a": "1500m"}), (
        "1.5 A written as text walked past the bounds check")
    assert not guard.check_parameter_bounds(meta, {"setpoint_a": "100p"})


def test_the_safety_gate_middleware_reads_si_strings():
    """SafetyGate sees the RAW tool-call args, which are strings now."""
    from mast.agents._shared.safety_mw import SafetyGate
    from mast.config import SafetyLimits

    gate = SafetyGate(SafetyLimits())
    meta = SetSetpoint().metadata()
    assert gate.check_global_bounds(meta, {"setpoint_a": "1500m"}), (
        "the gate stopped checking the parameters most in need of checking")
    assert not gate.check_global_bounds(meta, {"setpoint_a": "100p"})


def test_the_coarse_approach_classifier_reads_si_strings():
    """A Z target written as text must still count as a Z move."""
    from mast.core.safety import is_coarse_sample_approach

    assert is_coarse_sample_approach(
        "MotorMoveClosedLoop", {"target_z_m": "-5u", "absolute": False}) is True
    assert is_coarse_sample_approach(
        "MotorMoveClosedLoop", {"target_z_m": "0", "absolute": False}) is False


# ── End to end ───────────────────────────────────────────────────────────────

def test_the_value_reaching_hardware_is_the_value_that_was_written():
    canned = {"ZCtrl_GainSet": {"return_value": ("", b"", [])},
              "ZCtrl_GainGet": {"return_value": ("", b"", [3e-12, 1.6667e-05, 1.8e-07])}}
    provider, instances = _provider(canned)
    tool = wrap_skill(SetZCtrlGain, provider)
    tool.func(tool_call_id="t", state={},
              p_gain="3p", time_constant_s="16.667u", i_gain="180n")
    sent = [c[1] for c in instances[-1].calls if c[0] == "ZCtrl_GainSet"][0]
    assert sent[0] == pytest.approx(3e-12)
    assert sent[2] == pytest.approx(1.8e-07)


def test_the_skill_itself_never_sees_a_string():
    """Coercion happens above validate_params, so nothing downstream changed."""
    seen: dict = {}

    class _Spy(SetZCtrlGain):
        def execute(self, context, params):
            seen.update(params)
            return super().execute(context, params)

    canned = {"ZCtrl_GainSet": {"return_value": ("", b"", [])},
              "ZCtrl_GainGet": {"return_value": ("", b"", [3e-12, 1.6667e-05, 1.8e-07])}}
    tool = wrap_skill(_Spy, _provider(canned)[0])
    tool.func(tool_call_id="t", state={},
              p_gain="3p", time_constant_s="16.667u", i_gain="180n")
    assert all(isinstance(v, float) for v in seen.values()), seen


def test_an_unparseable_magnitude_costs_no_connection():
    """Rejected before the context provider is even called."""
    provider, instances = _provider({})
    tool = wrap_skill(SetZCtrlGain, provider)
    r = tool.func(tool_call_id="t", state={},
                  p_gain="3", time_constant_s="16.667u", i_gain="180n")
    assert instances == []
    assert "precondition_failed" in str(r)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
