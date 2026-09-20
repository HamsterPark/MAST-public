"""v2 unit tests for mast.skills.builtins.folme.

Skills covered (8):
  SetTipSpeed (CONFIRM), GetTipSpeed (AUTO), SetFolMeOversampling (CONFIRM),
  StopFolMe (AUTO), GetPointShootOnOff (AUTO), SetPointShootOnOff (CONFIRM),
  SetPointShootExperiment (CONFIRM), GetPointShootProps (AUTO).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_folme.py -x -v
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
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.folme import (
    GetPointShootOnOff,
    GetPointShootProps,
    GetTipSpeed,
    SetFolMeOversampling,
    SetPointShootExperiment,
    SetPointShootOnOff,
    SetTipSpeed,
    StopFolMe,
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
    (SetTipSpeed, "SetTipSpeed", "CONFIRM"),
    (GetTipSpeed, "GetTipSpeed", "AUTO"),
    (SetFolMeOversampling, "SetFolMeOversampling", "CONFIRM"),
    (StopFolMe, "StopFolMe", "AUTO"),
    (GetPointShootOnOff, "GetPointShootOnOff", "AUTO"),
    (SetPointShootOnOff, "SetPointShootOnOff", "CONFIRM"),
    (SetPointShootExperiment, "SetPointShootExperiment", "CONFIRM"),
    (GetPointShootProps, "GetPointShootProps", "AUTO"),
])
def test_folme_skill_shape(skill_cls, expected_name, expected_danger):
    tool = wrap_skill(skill_cls, make_provider())
    assert tool.name == expected_name
    assert tool.metadata["danger_level"] == expected_danger
    assert tool.metadata["skill_source"].endswith(".folme")


def test_set_tip_speed_required_field():
    tool = wrap_skill(SetTipSpeed, make_provider())
    fields = tool.args_schema.model_fields
    assert "speed_m_s" in fields
    assert fields["speed_m_s"].is_required()
    assert "custom_speed" in fields
    assert not fields["custom_speed"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["speed_m_s"].annotation is str


def test_set_folme_oversampling_required_field():
    tool = wrap_skill(SetFolMeOversampling, make_provider())
    fields = tool.args_schema.model_fields
    assert "oversampling" in fields
    assert fields["oversampling"].is_required()
    assert fields["oversampling"].annotation is int


def test_set_point_shoot_on_off_required_field():
    tool = wrap_skill(SetPointShootOnOff, make_provider())
    fields = tool.args_schema.model_fields
    assert "enable" in fields
    assert fields["enable"].is_required()
    assert fields["enable"].annotation is bool


# ── Execution tests ───────────────────────────────────────────────────────────

def test_set_tip_speed_executes():
    canned = {"FolMe_SpeedSet": {"return_value": None}}
    tool = wrap_skill(SetTipSpeed, make_provider(canned))
    result = _invoke(tool, speed_m_s=1e-9)
    assert result.update["executed_skills"] == ["SetTipSpeed"]


def test_set_tip_speed_calls_correct_args():
    canned = {"FolMe_SpeedSet": {"return_value": None}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetTipSpeed, capturing_provider)
    _invoke(tool, speed_m_s=2e-9, custom_speed=True)
    ctx = instances[-1]
    speed_calls = [c for c in ctx.calls if c[0] == "FolMe_SpeedSet"]
    assert len(speed_calls) == 1
    assert speed_calls[0][1][0] == 2e-9
    assert speed_calls[0][1][1] == 1  # custom_speed=True → int(True)=1


def test_get_tip_speed_executes():
    canned = {"FolMe_SpeedGet": {"return_value": ("", b"", [2e-9, 1])}}
    tool = wrap_skill(GetTipSpeed, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetTipSpeed"]


def test_stop_folme_executes():
    canned = {"FolMe_Stop": {"return_value": None}}
    tool = wrap_skill(StopFolMe, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["StopFolMe"]


def test_set_folme_oversampling_executes():
    canned = {"FolMe_OversamplSet": {"return_value": None}}
    tool = wrap_skill(SetFolMeOversampling, make_provider(canned))
    result = _invoke(tool, oversampling=4)
    assert result.update["executed_skills"] == ["SetFolMeOversampling"]


def test_get_point_shoot_on_off_executes():
    canned = {"FolMe_PSOnOffGet": {"return_value": ("", b"", [1])}}
    tool = wrap_skill(GetPointShootOnOff, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetPointShootOnOff"]


def test_set_point_shoot_on_off_executes():
    canned = {"FolMe_PSOnOffSet": {"return_value": None}}
    tool = wrap_skill(SetPointShootOnOff, make_provider(canned))
    result = _invoke(tool, enable=True)
    assert result.update["executed_skills"] == ["SetPointShootOnOff"]


def test_set_point_shoot_on_off_enable_maps_to_1():
    canned = {"FolMe_PSOnOffSet": {"return_value": None}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetPointShootOnOff, capturing_provider)
    _invoke(tool, enable=True)
    ctx = instances[-1]
    calls = [c for c in ctx.calls if c[0] == "FolMe_PSOnOffSet"]
    assert calls[0][1][0] == 1  # enable=True → status=1


def test_set_point_shoot_experiment_executes():
    canned = {"FolMe_PSExpSet": {"return_value": None}}
    tool = wrap_skill(SetPointShootExperiment, make_provider(canned))
    result = _invoke(tool, experiment_index=0)
    assert result.update["executed_skills"] == ["SetPointShootExperiment"]


def test_get_point_shoot_props_executes():
    canned = {"FolMe_PSPropsGet": {"return_value": ("", b"", [1, 0, 5, "scan", 0, "", 0.5])}}
    tool = wrap_skill(GetPointShootProps, make_provider(canned))
    result = _invoke(tool)
    assert result.update["executed_skills"] == ["GetPointShootProps"]


def test_set_tip_speed_error_propagates():
    canned = {"FolMe_SpeedSet": {"error": "TCP timeout"}}
    tool = wrap_skill(SetTipSpeed, make_provider(canned))
    result = _invoke(tool, speed_m_s=1e-9)
    msg = result.update["messages"][0]
    assert msg.status == "error"


# ── Realistic-triplet parse tests (return_value = [err, raw, Variables]) ─────

def test_get_tip_speed_parses_real_triplet():
    """FolMe_SpeedGet ResponseTypes ["f","I"] → [speed, custom_speed]."""
    canned = {"FolMe_SpeedGet": {"return_value": ("", b"\x00", [3.5e-9, 1])}}
    skill = GetTipSpeed()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["speed_m_s"] == 3.5e-9
    assert res.data["custom_speed"] is True


def test_get_point_shoot_on_off_parses_real_triplet():
    """FolMe_PSOnOffGet ResponseTypes ["I"] → [enabled]."""
    canned = {"FolMe_PSOnOffGet": {"return_value": ("", b"\x00", [0])}}
    skill = GetPointShootOnOff()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["enabled"] is False


def test_get_point_shoot_props_parses_real_triplet():
    """FolMe_PSPropsGet ResponseTypes ["I","I","i","*-c","i","*-c","f"]:
    [auto_resume, use_own_basename, bn_size, basename, vi_size, vi, delay]."""
    canned = {
        "FolMe_PSPropsGet": {
            "return_value": (
                "", b"\x00",
                [1, 0, 4, "scan", 0, "", 0.5],
            )
        }
    }
    skill = GetPointShootProps()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["auto_resume"] is True
    assert res.data["use_own_basename"] is False
    assert res.data["basename"] == "scan"
    assert res.data["pre_measure_delay_s"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
