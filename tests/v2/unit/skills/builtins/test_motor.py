"""v2 unit tests for mast.skills.builtins.motor.

Skills covered (6):
  MotorMove (CONFIRM), MotorGetPos (AUTO), StopMotor (AUTO),
  SetMotorFreqAmp (CONFIRM), MotorMoveClosedLoop (CONFIRM),
  GetMotorStepCounter (AUTO).

2026-06-11 safety re-scoping: the ONLY physically-dangerous action is an
open-loop coarse Z step TOWARD the sample (MotorMove direction='z-approach').
MotorMove is now CONFIRM (baseline); the z-approach danger is gated at runtime
by mast.core.safety.is_coarse_sample_approach (human approval on the executor
path, fail-closed block on the autonomous agent path), NOT by safety_level.
No builtin skill is DANGEROUS anymore.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_motor.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
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
from typing import Any, get_args

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.motor import (
    GetMotorStepCounter,
    MotorGetPos,
    MotorMove,
    MotorMoveClosedLoop,
    SetMotorFreqAmp,
    StopMotor,
)


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("skill_cls,expected_name,expected_danger", [
    # 2026-06-11 safety re-scoping: MotorMove is CONFIRM (baseline), NOT
    # DANGEROUS. The only physically-dangerous case — an open-loop coarse Z
    # step TOWARD the sample (direction='z-approach') — is gated at runtime by
    # mast.core.safety.is_coarse_sample_approach (human approval on the executor
    # path, fail-closed block on the autonomous agent path), independent of the
    # safety_level. No builtin skill is DANGEROUS anymore. MotorMoveClosedLoop
    # keeps a feedback target (not the open-loop danger) and stays CONFIRM.
    (MotorMove, "MotorMove", "CONFIRM"),
    (MotorGetPos, "MotorGetPos", "AUTO"),
    (StopMotor, "StopMotor", "AUTO"),
    (SetMotorFreqAmp, "SetMotorFreqAmp", "CONFIRM"),
    (MotorMoveClosedLoop, "MotorMoveClosedLoop", "CONFIRM"),
    (GetMotorStepCounter, "GetMotorStepCounter", "AUTO"),
])
def test_motor_skill_shape(skill_cls, expected_name, expected_danger):
    tool = wrap_skill(skill_cls, make_provider())
    assert tool.name == expected_name
    assert tool.metadata["danger_level"] == expected_danger
    assert tool.metadata["skill_source"].endswith(".motor")


def test_motor_move_required_fields():
    tool = wrap_skill(MotorMove, make_provider())
    fields = tool.args_schema.model_fields
    assert "direction" in fields
    assert "steps" in fields
    assert fields["direction"].is_required()
    assert fields["steps"].is_required()
    # direction carries allowed_values, so it reaches the model as an enum rather
    # than a bare string it would have to guess the legal values for. The Z names
    # say what they DO ('z-approach' / 'z-retract'), which is the whole reason the
    # enum is worth showing the model — 'z+' would leave it guessing which way is
    # toward the sample.
    assert set(get_args(fields["direction"].annotation)) == {
        "x+", "x-", "y+", "y-", "z-approach", "z-retract",
    }
    assert fields["steps"].annotation is int


def test_motor_move_closed_loop_required_fields():
    tool = wrap_skill(MotorMoveClosedLoop, make_provider())
    fields = tool.args_schema.model_fields
    for req in ("target_x_m", "target_y_m", "target_z_m"):
        assert req in fields
        assert fields[req].is_required()
    for opt in ("absolute", "wait", "group"):
        assert opt in fields
        assert not fields[opt].is_required()


def test_set_motor_freq_amp_required_fields():
    tool = wrap_skill(SetMotorFreqAmp, make_provider())
    fields = tool.args_schema.model_fields
    assert fields["frequency_hz"].is_required()
    assert fields["amplitude_v"].is_required()
    assert not fields["axis"].is_required()


# ── Execution tests ───────────────────────────────────────────────────────────

def test_motor_move_executes():
    canned = {"Motor_StartMove": {"return_value": None}}
    tool = wrap_skill(MotorMove, make_provider(canned))
    result = _invoke(tool, direction="x+", steps=5)
    assert result.update["executed_skills"] == ["MotorMove"]


@pytest.mark.parametrize("direction,expected_code", [
    ("x+", 0),
    ("x-", 1),
    ("y+", 2),
    ("y-", 3),
    # 2026-06-11: 'z-approach' = TOWARD sample = Z- = code 5;
    #             'z-retract'  = AWAY from sample = Z+ = code 4.
    ("z-approach", 5),
    ("z-retract", 4),
])
def test_motor_move_direction_encoding(direction, expected_code):
    """Each allowed direction maps to the correct Nanonis dir_code."""
    canned = {"Motor_StartMove": {"return_value": None}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(MotorMove, capturing_provider)
    result = _invoke(tool, direction=direction, steps=3)
    # Every allowed direction must actually execute (not blocked at the skill layer;
    # the z-approach danger gate lives in the executor/agent middleware, not here).
    assert result.update["executed_skills"] == ["MotorMove"]
    ctx = instances[-1]
    calls = [c for c in ctx.calls if c[0] == "Motor_StartMove"]
    assert calls[0][1][0] == expected_code  # dir_code
    assert calls[0][1][1] == 3  # steps


def test_motor_move_allowed_directions():
    """allowed_values now includes the two Z coarse directions."""
    tool = wrap_skill(MotorMove, make_provider())
    spec = next(
        p for p in tool.metadata["skill_metadata"].parameters
        if p.name == "direction"
    )
    assert spec.allowed_values == ["x+", "x-", "y+", "y-", "z-approach", "z-retract"]


def test_motor_move_z_approach_is_coarse_sample_approach():
    """direction='z-approach' is flagged by is_coarse_sample_approach; others not."""
    from mast.core.safety import is_coarse_sample_approach

    assert is_coarse_sample_approach("MotorMove", {"direction": "z-approach"}) is True
    for d in ("z-retract", "x+", "x-", "y+", "y-"):
        assert is_coarse_sample_approach("MotorMove", {"direction": d}) is False
    # Closed-loop coarse moves are positional-feedback (no tip-contact stop): ANY
    # Z-bearing move is conservatively gated, but a pure-XY move is not.
    assert is_coarse_sample_approach("MotorMoveClosedLoop", {"target_z_m": -1e-7}) is True
    assert is_coarse_sample_approach(
        "MotorMoveClosedLoop", {"target_x_m": 1e-6, "target_y_m": 0.0}) is False
    assert is_coarse_sample_approach("MotorMoveClosedLoop", {"target_z_m": 0.0}) is False
    assert is_coarse_sample_approach("MotorMove", None) is False


def test_motor_move_closed_loop_target_z_has_safety_ceiling():
    """target_z_m carries a defensive coarse safety ceiling (±1e-4 m); x/y do
    NOT (they are already clamped by the global x_m/y_m bounds check)."""
    spec_by_name = {
        p.name: p for p in MotorMoveClosedLoop().metadata().parameters
    }
    tz = spec_by_name["target_z_m"]
    assert tz.min_value == -1e-4
    assert tz.max_value == 1e-4
    # x/y deliberately unbounded at the spec level (global check clamps them).
    assert spec_by_name["target_x_m"].min_value is None
    assert spec_by_name["target_x_m"].max_value is None
    assert spec_by_name["target_y_m"].min_value is None
    assert spec_by_name["target_y_m"].max_value is None
    # And the bound actually rejects an absurd Z target via validate_params.
    errs = MotorMoveClosedLoop().validate_params(
        {"target_x_m": 0.0, "target_y_m": 0.0, "target_z_m": 1.0}  # 1 metre!
    )
    assert any("target_z_m" in e and "maximum" in e for e in errs)


def test_motor_get_pos_executes():
    canned = {"Motor_PosGet": {"return_value": ("", b"", [1e-3, 2e-3, 3e-3])}}
    tool = wrap_skill(MotorGetPos, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["MotorGetPos"]


def test_stop_motor_executes():
    canned = {"Motor_StopMove": {"return_value": None}}
    tool = wrap_skill(StopMotor, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["StopMotor"]


def test_set_motor_freq_amp_executes():
    canned = {"Motor_FreqAmpSet": {"return_value": None}}
    tool = wrap_skill(SetMotorFreqAmp, make_provider(canned))
    result = _invoke(tool, frequency_hz=1000.0, amplitude_v=30.0)
    assert result.update["executed_skills"] == ["SetMotorFreqAmp"]


def test_motor_move_closed_loop_executes():
    canned = {"Motor_StartClosedLoop": {"return_value": None}}
    tool = wrap_skill(MotorMoveClosedLoop, make_provider(canned))
    result = _invoke(tool, target_x_m=0.0, target_y_m=0.0, target_z_m=0.0)
    assert result.update["executed_skills"] == ["MotorMoveClosedLoop"]


def test_get_motor_step_counter_executes():
    canned = {"Motor_StepCounterGet": {"return_value": ("", b"", [10, 20, 30])}}
    tool = wrap_skill(GetMotorStepCounter, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetMotorStepCounter"]


def test_motor_move_error_propagates():
    canned = {"Motor_StartMove": {"error": "controller not ready"}}
    tool = wrap_skill(MotorMove, make_provider(canned))
    result = _invoke(tool, direction="y-", steps=1)
    assert result.update["messages"][0].status == "error"


def test_motor_move_missing_required():
    tool = wrap_skill(MotorMove, make_provider())
    result = _invoke(tool)  # missing direction + steps
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "Missing required" in msg_content
        or "direction" in msg_content
        or "steps" in msg_content
    )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
