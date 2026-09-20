"""v2 unit tests for mast.skills.builtins.zcontrol.

Skills covered: GetZPosition (AUTO), SetSetpoint (CONFIRM),
  ZControllerOnOff (CONFIRM), SetZPosition (CONFIRM), SetTipLift (CONFIRM),
  GetTipLift (AUTO), SetZLimitsEnabled (CONFIRM), GetZLimitsEnabled (AUTO),
  GetZCtrlList (AUTO), GetHomeProps (AUTO), SetHomeProps (CONFIRM),
  SetSwitchOffDelay (CONFIRM), GetWithdrawRate (AUTO) — 13 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_zcontrol.py -x -v
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
from mast.skills.builtins.zcontrol import (
    GetHomeProps,
    GetTipLift,
    GetWithdrawRate,
    GetZCtrlList,
    GetZLimitsEnabled,
    GetZPosition,
    SetHomeProps,
    SetSetpoint,
    SetSwitchOffDelay,
    SetTipLift,
    SetZLimitsEnabled,
    SetZPosition,
    ZControllerOnOff,
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

def test_get_z_position_shape():
    tool = wrap_skill(GetZPosition, make_provider())
    assert tool.name == "GetZPosition"
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_set_setpoint_shape():
    tool = wrap_skill(SetSetpoint, make_provider())
    assert tool.name == "SetSetpoint"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "setpoint_a" in fields
    assert fields["setpoint_a"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["setpoint_a"].annotation is str
    assert "A" in (fields["setpoint_a"].description or "")


def test_z_controller_on_off_shape():
    tool = wrap_skill(ZControllerOnOff, make_provider())
    assert tool.name == "ZControllerOnOff"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "enable" in fields
    assert fields["enable"].is_required()
    assert fields["enable"].annotation is bool


def test_set_z_position_shape():
    tool = wrap_skill(SetZPosition, make_provider())
    assert tool.name == "SetZPosition"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "z_pos_m" in fields
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["z_pos_m"].annotation is str


def test_set_home_props_shape():
    tool = wrap_skill(SetHomeProps, make_provider())
    fields = tool.args_schema.model_fields
    assert "rel_or_abs" in fields
    assert "home_position_m" in fields
    assert fields["rel_or_abs"].annotation is int
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["home_position_m"].annotation is str


def test_set_switch_off_delay_shape():
    tool = wrap_skill(SetSwitchOffDelay, make_provider())
    fields = tool.args_schema.model_fields
    assert "delay_s" in fields
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["delay_s"].annotation is str
    assert "s" in (fields["delay_s"].description or "").lower()


def test_skill_source_points_to_zcontrol_module():
    tool = wrap_skill(GetZPosition, make_provider())
    assert tool.metadata["skill_source"].endswith(".zcontrol")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_get_z_position_succeeds():
    canned = {
        "ZCtrl_ZPosGet": {"return_value": ("", b"", [1.5e-9])}
    }
    tool = wrap_skill(GetZPosition, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["GetZPosition"]
    msg_content = update["messages"][0].content
    assert "z_pos_m" in msg_content or "1.5" in msg_content


def test_set_setpoint_calls_correct_method():
    canned = {"ZCtrl_SetpntSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetSetpoint, capturing_provider)
    _invoke(tool, setpoint_a=100e-12)
    last_ctx = instances[-1]
    sp_calls = [c for c in last_ctx.calls if c[0] == "ZCtrl_SetpntSet"]
    assert len(sp_calls) == 1
    assert abs(sp_calls[0][1][0] - 100e-12) < 1e-20


def test_z_controller_on_off_executes():
    canned = {"ZCtrl_OnOffSet": {"return_value": ("", b"", [])}}
    tool = wrap_skill(ZControllerOnOff, make_provider(canned))
    result = _invoke(tool, enable=True)
    assert result.update["executed_skills"] == ["ZControllerOnOff"]


def test_get_z_position_error_propagated():
    canned = {"ZCtrl_ZPosGet": {"error": "Z position read failed"}}
    tool = wrap_skill(GetZPosition, make_provider(canned))
    result = _invoke(tool)
    msg_content = result.update["messages"][0].content
    assert (
        "Z position read failed" in msg_content
        or "False" in msg_content
        or "error" in msg_content.lower()
    )


# ── Triple-fixture tests for the three previously-broken reads ────────────────
#
# nanonis_spm methods return [error_string, raw_bytes, Variables]; the real data
# is Variables == return_value[2], laid out per the method's ResponseTypes. The
# FakeCtx fixtures below feed the *exact* triple shape the parser produces so
# these tests fail on the old [0]/[1]-indexed code and pass on the fix.
#
#   ZCtrl.HomePropsGet   ResponseTypes ["H", "f"]            -> [rel_or_abs, home_pos_m]
#   ZCtrl.WithdrawRateGet ResponseTypes ["f"]                -> [rate]
#   ZCtrl.CtrlListGet    ResponseTypes ["i", "i", "*+c", "i"] ->
#       [list_size, num_controllers, [names...], active_index]


def _exec(skill_cls, canned, **params):
    """Run skill.execute() directly against a FakeCtx; return the SkillResult."""
    ctx = FakeCtx(canned=canned)
    return skill_cls().execute(ctx, params)


def test_get_home_props_reads_variables_triple():
    # Variables [0]=relative(1), [1]=home position 2.5 nm.
    canned = {"ZCtrl_HomePropsGet": {"return_value": ["", b"raw", [1, 2.5e-9]]}}
    res = _exec(GetHomeProps, canned)
    assert res.success is True
    assert res.data["mode"] == "relative"
    assert abs(res.data["home_position_m"] - 2.5e-9) < 1e-18


def test_get_home_props_absolute_mode():
    canned = {"ZCtrl_HomePropsGet": {"return_value": ["", b"raw", [0, 0.0]]}}
    res = _exec(GetHomeProps, canned)
    assert res.success is True
    assert res.data["mode"] == "absolute"
    assert res.data["home_position_m"] == 0.0


def test_get_home_props_does_not_crash_via_wrapper():
    # Old code: int(parsed[0]) == int('') -> ValueError -> ToolMessage status=error.
    canned = {"ZCtrl_HomePropsGet": {"return_value": ["", b"", [1, 2.5e-9]]}}
    tool = wrap_skill(GetHomeProps, make_provider(canned))
    result = _invoke(tool)
    msg = result.update["messages"][0]
    assert msg.status == "success"
    assert "rolled_back" not in msg.content
    assert "ValueError" not in msg.content


def test_get_withdraw_rate_reads_variables_triple():
    # Variables [0]=rate 1.2e-6 m/s.
    canned = {"ZCtrl_WithdrawRateGet": {"return_value": ["", b"raw", [1.2e-6]]}}
    res = _exec(GetWithdrawRate, canned)
    assert res.success is True
    assert abs(res.data["withdraw_rate_m_per_s"] - 1.2e-6) < 1e-15


def test_get_withdraw_rate_does_not_crash_via_wrapper():
    # Old code: float(parsed[0]) == float('') -> ValueError.
    canned = {"ZCtrl_WithdrawRateGet": {"return_value": ["", b"", [3.4e-7]]}}
    tool = wrap_skill(GetWithdrawRate, make_provider(canned))
    result = _invoke(tool)
    msg = result.update["messages"][0]
    assert msg.status == "success"
    assert "rolled_back" not in msg.content
    assert "3.4" in msg.content or "withdraw_rate_m_per_s" in msg.content


def test_get_zctrl_list_parses_names_and_active_index():
    # Variables: list_size, num_controllers, names[], active_index.
    canned = {
        "ZCtrl_CtrlListGet": {
            "return_value": ["", b"raw", [24, 2, ["Controller 1", "Log"], 1]]
        }
    }
    res = _exec(GetZCtrlList, canned)
    assert res.success is True
    assert res.data["controllers"] == ["Controller 1", "Log"]
    assert res.data["active_index"] == 1


def test_get_zctrl_list_empty_when_no_names():
    canned = {"ZCtrl_CtrlListGet": {"return_value": ["", b"raw", [0, 0, [], 0]]}}
    res = _exec(GetZCtrlList, canned)
    assert res.success is True
    assert res.data["controllers"] == []
    assert res.data["active_index"] == 0


def test_get_zctrl_list_active_index_via_wrapper():
    canned = {
        "ZCtrl_CtrlListGet": {
            "return_value": ["", b"", [12, 1, ["Controller 1"], 0]]
        }
    }
    tool = wrap_skill(GetZCtrlList, make_provider(canned))
    result = _invoke(tool)
    msg = result.update["messages"][0]
    assert msg.status == "success"
    assert "Controller 1" in msg.content


# ── Safety-fix tests: unbounded Z config + default-on Z soft-limits ───────────


def _spec(skill_cls, param_name):
    return next(
        p for p in skill_cls().metadata().parameters if p.name == param_name
    )


def test_set_tip_lift_has_um_bounds():
    # Safety fix #5: tip_lift_m gains a ±1 µm hard cap (mirrors TipShape).
    spec = _spec(SetTipLift, "tip_lift_m")
    assert spec.min_value == -1e-6
    assert spec.max_value == 1e-6
    # The bound actually rejects an absurd lift via validate_params.
    errs = SetTipLift().validate_params({"tip_lift_m": 5.0})  # 5 metres!
    assert any("tip_lift_m" in e and "maximum" in e for e in errs)
    assert SetTipLift().validate_params({"tip_lift_m": -2e-9}) == []


def test_set_home_props_position_has_um_bounds():
    # Safety fix #5: home_position_m gains a ±1 µm sanity cap.
    spec = _spec(SetHomeProps, "home_position_m")
    assert spec.min_value == -1e-6
    assert spec.max_value == 1e-6
    errs = SetHomeProps().validate_params(
        {"rel_or_abs": 1, "home_position_m": 1.0}  # 1 metre!
    )
    assert any("home_position_m" in e and "maximum" in e for e in errs)


def test_set_z_limits_enabled_defaults_on():
    # Safety fix #4: `enabled` is optional and defaults to True; a bare call
    # ENABLES the Z soft-limits and calls LimitsEnabledSet(1).
    spec = _spec(SetZLimitsEnabled, "enabled")
    assert spec.required is False
    assert spec.default is True

    canned = {"ZCtrl_LimitsEnabledSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetZLimitsEnabled, capturing_provider)
    result = _invoke(tool)  # no enabled arg
    assert result.update["messages"][0].status == "success"
    calls = [c for c in instances[-1].calls if c[0] == "ZCtrl_LimitsEnabledSet"]
    assert len(calls) == 1
    assert calls[0][1] == (1,)  # defaulted on


def test_set_z_limits_enabled_explicit_disable_still_works():
    # The disable path is intentionally NOT gated: enabled=False → LimitsEnabledSet(0).
    canned = {"ZCtrl_LimitsEnabledSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetZLimitsEnabled, capturing_provider)
    result = _invoke(tool, enabled=False)
    assert result.update["messages"][0].status == "success"
    calls = [c for c in instances[-1].calls if c[0] == "ZCtrl_LimitsEnabledSet"]
    assert calls[0][1] == (0,)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
