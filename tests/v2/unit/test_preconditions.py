"""BaseSkill.check_preconditions recognises z_controller_off (motor/Z skills).

审查 HIGH: 'z_controller_off' was unknown → MotorMove /
MotorMoveClosedLoop / SetZPosition always failed AND the controller-off
tip-crash safety check never ran.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_preconditions.py -x -v
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

from mast.core.types import (
    HardwareState, SafetyLevel, SkillCategory, SkillMetadata, SkillResult,
)
from mast.skills.base import BaseSkill


class _MotorLike(BaseSkill):
    def metadata(self):
        return SkillMetadata(
            name="FakeMotor", version="1.0.0", category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS, description="",
            parameters=[], preconditions=["z_controller_off"],
        )
    def execute(self, ctx, params):
        return SkillResult(skill_name="FakeMotor", success=True)


def test_z_controller_off_recognised():
    s = _MotorLike()
    # controller OFF → precondition met (no "cannot verify", no unmet)
    assert s.check_preconditions(HardwareState(z_controller_on=False)) == []
    # controller ON → precondition correctly NOT met (a real safety block)
    unmet = s.check_preconditions(HardwareState(z_controller_on=True))
    assert len(unmet) == 1
    assert "not met" in unmet[0]
    assert "cannot verify" not in unmet[0].lower()   # it IS recognised now


def test_no_longer_unverifiable():
    from mast.skills.base import _PRECONDITION_CHECKS
    assert "z_controller_off" in _PRECONDITION_CHECKS
    assert _PRECONDITION_CHECKS["z_controller_off"] == ("z_controller_on", False)


class _ApproachLike(BaseSkill):
    def metadata(self):
        return SkillMetadata(
            name="FakeApproach", version="1.0.0", category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS, description="",
            parameters=[], preconditions=["bias_nonzero"],
        )
    def execute(self, ctx, params):
        return SkillResult(skill_name="FakeApproach", success=True)


def test_bias_nonzero_recognised():
    # 审查: 'bias_nonzero' lived ONLY in the SafetyGate substring
    # rules, NOT the exact-name dict base.py consulted, so EVERY approach got
    # "Cannot verify precondition: 'bias_nonzero'" even with the bias at 1.0 V —
    # the agent-path approach was fully blocked (feedback: 进针功能调用失败).
    s = _ApproachLike()
    # bias non-zero → precondition met (no "cannot verify", no unmet)
    assert s.check_preconditions(HardwareState(bias_v=1.0)) == []
    # bias unknown (None) → passes (mirrors the original is-not-None semantics)
    assert s.check_preconditions(HardwareState(bias_v=None)) == []
    # bias exactly zero → correctly NOT met (the real block: 0 V grinds the tip)
    unmet = s.check_preconditions(HardwareState(bias_v=0.0))
    assert len(unmet) == 1
    assert "bias is zero" in unmet[0]
    assert "cannot verify" not in unmet[0].lower()


def test_precondition_recognized_helper():
    from mast.core.preconditions import precondition_recognized
    assert precondition_recognized("bias_nonzero") is True       # substring rule
    assert precondition_recognized("z_controller_off") is True   # exact-name dict
    assert precondition_recognized("totally_unknown_xyz") is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
