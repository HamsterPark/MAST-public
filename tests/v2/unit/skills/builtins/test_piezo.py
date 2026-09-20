"""v2 unit tests for mast.skills.builtins.piezo.

Skills covered: SetDriftCompensation (CONFIRM), GetDriftCompensation (AUTO),
  SetPiezoTilt (CONFIRM), GetPiezoTilt (AUTO), SetPiezoRange (CONFIRM),
  SetPiezoSensitivity (CONFIRM), GetPiezoSensitivity (AUTO),
  GetPiezoHVAInfo (AUTO), GetPiezoHVAStatusLED (AUTO),
  GetPiezoXYZLimits (AUTO), SetPiezoHysteresisOnOff (CONFIRM),
  SetPiezoHysteresisValues (CONFIRM), LoadPiezoHysteresisFile (CONFIRM)
  — 13 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_piezo.py -x -v
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
from mast.skills.builtins.piezo import (
    GetDriftCompensation,
    GetPiezoHVAInfo,
    GetPiezoHVAStatusLED,
    GetPiezoSensitivity,
    GetPiezoTilt,
    GetPiezoXYZLimits,
    LoadPiezoHysteresisFile,
    SetDriftCompensation,
    SetPiezoHysteresisOnOff,
    SetPiezoHysteresisValues,
    SetPiezoRange,
    SetPiezoSensitivity,
    SetPiezoTilt,
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

def test_set_drift_compensation_shape():
    tool = wrap_skill(SetDriftCompensation, make_provider())
    assert tool.name == "SetDriftCompensation"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "enable" in fields
    assert fields["enable"].is_required()
    assert fields["enable"].annotation is bool
    assert "vx" in fields
    assert not fields["vx"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["vx"].annotation is str


def test_get_drift_compensation_shape():
    tool = wrap_skill(GetDriftCompensation, make_provider())
    assert tool.name == "GetDriftCompensation"
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_set_piezo_tilt_shape():
    tool = wrap_skill(SetPiezoTilt, make_provider())
    assert tool.name == "SetPiezoTilt"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "tilt_x_deg" in fields
    assert "tilt_y_deg" in fields
    assert fields["tilt_x_deg"].is_required()
    assert fields["tilt_y_deg"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["tilt_x_deg"].annotation is str
    assert "deg" in (fields["tilt_x_deg"].description or "")


def test_set_piezo_range_shape():
    tool = wrap_skill(SetPiezoRange, make_provider())
    assert tool.name == "SetPiezoRange"
    fields = tool.args_schema.model_fields
    assert "range_x_m" in fields
    assert "range_y_m" in fields
    assert "range_z_m" in fields
    for f in ("range_x_m", "range_y_m", "range_z_m"):
        assert fields[f].is_required()
        # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
        assert fields[f].annotation is str


def test_set_piezo_sensitivity_shape():
    tool = wrap_skill(SetPiezoSensitivity, make_provider())
    fields = tool.args_schema.model_fields
    assert "sens_x" in fields
    assert "sens_y" in fields
    assert "sens_z" in fields
    assert "m/V" in (fields["sens_x"].description or "")


def test_set_piezo_hysteresis_values_shape():
    tool = wrap_skill(SetPiezoHysteresisValues, make_provider())
    fields = tool.args_schema.model_fields
    assert "fast_x" in fields
    assert "fast_y" in fields
    assert "slow_x" in fields
    assert "slow_y" in fields
    for f in ("fast_x", "fast_y", "slow_x", "slow_y"):
        assert fields[f].is_required()
        assert fields[f].annotation is str


def test_load_piezo_hysteresis_file_shape():
    tool = wrap_skill(LoadPiezoHysteresisFile, make_provider())
    fields = tool.args_schema.model_fields
    assert "file_path" in fields
    assert fields["file_path"].is_required()
    assert fields["file_path"].annotation is str


def test_read_skills_are_auto():
    for cls in (GetDriftCompensation, GetPiezoTilt, GetPiezoSensitivity,
                GetPiezoHVAInfo, GetPiezoHVAStatusLED, GetPiezoXYZLimits):
        tool = wrap_skill(cls, make_provider())
        assert tool.metadata["danger_level"] == "AUTO", f"{cls.__name__} should be AUTO"


def test_skill_source_points_to_piezo_module():
    tool = wrap_skill(SetDriftCompensation, make_provider())
    assert tool.metadata["skill_source"].endswith(".piezo")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_set_drift_compensation_enable():
    canned = {"Piezo_DriftCompSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetDriftCompensation, capturing_provider)
    result = _invoke(tool, enable=True, vx=1e-10, vy=2e-10, vz=0.0)
    assert result.update["executed_skills"] == ["SetDriftCompensation"]
    last_ctx = instances[-1]
    dc_calls = [c for c in last_ctx.calls if c[0] == "Piezo_DriftCompSet"]
    assert len(dc_calls) == 1
    # First positional arg should be 1 (enable)
    assert dc_calls[0][1][0] == 1


def test_set_drift_compensation_disable_passes_2():
    canned = {"Piezo_DriftCompSet": {"return_value": ("", b"", [])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SetDriftCompensation, capturing_provider)
    _invoke(tool, enable=False)
    last_ctx = instances[-1]
    dc_calls = [c for c in last_ctx.calls if c[0] == "Piezo_DriftCompSet"]
    assert dc_calls[0][1][0] == 2  # 2 = Off


def test_set_piezo_tilt_executes():
    canned = {"Piezo_TiltSet": {"return_value": ("", b"", [])}}
    tool = wrap_skill(SetPiezoTilt, make_provider(canned))
    result = _invoke(tool, tilt_x_deg=0.5, tilt_y_deg=-0.3)
    assert result.update["executed_skills"] == ["SetPiezoTilt"]


def test_get_drift_compensation_error_propagated():
    canned = {"Piezo_DriftCompGet": {"error": "piezo module not responding"}}
    tool = wrap_skill(GetDriftCompensation, make_provider(canned))
    result = _invoke(tool)
    msg_content = result.update["messages"][0].content
    assert (
        "piezo module not responding" in msg_content
        or "False" in msg_content
        or "error" in msg_content.lower()
    )


# ── Real-triplet data-extraction tests ─────────────────────────────────────────
# Nanonis quickSend returns [error_string, raw_bytes, Variables]; Variables
# (index [2]) is ordered exactly by each method's ResponseTypes array. These
# tests feed realistic triplets and assert each field lands on the right key.


def _run(cls, method, variables, **kwargs):
    """Invoke skill cls with a canned triplet ("", b"...", variables)."""
    canned = {method: {"return_value": ("", b"\x00", variables)}}
    skill = cls()
    ctx = FakeCtx(canned=canned)
    return skill.execute(ctx, kwargs)


def test_get_drift_compensation_field_order():
    # ResponseTypes ["I","f","f","f","I","I","I","f"]:
    # [status, Vx, Vy, Vz, X_sat, Y_sat, Z_sat]
    # status=1 (On); velocities deliberately distinct so a shift is caught.
    res = _run(
        GetDriftCompensation, "Piezo_DriftCompGet",
        [1, 1.1e-10, 2.2e-10, 3.3e-10, 0, 1, 0],
    )
    assert res.success
    # status (index 0) is the on/off flag — NOT a velocity.
    assert res.data["enabled"] is True
    assert res.data["vx"] == 1.1e-10
    assert res.data["vy"] == 2.2e-10
    assert res.data["vz"] == 3.3e-10
    assert res.data["x_saturated"] is False
    assert res.data["y_saturated"] is True
    assert res.data["z_saturated"] is False


def test_get_drift_compensation_disabled_status():
    # status=0 must report enabled=False even though Vz is non-zero.
    res = _run(
        GetDriftCompensation, "Piezo_DriftCompGet",
        [0, 0.0, 0.0, 9.9e-10, 0, 0, 0],
    )
    assert res.success
    assert res.data["enabled"] is False
    assert res.data["vz"] == 9.9e-10


def test_get_piezo_tilt_field_order():
    # ResponseTypes ["f","f"] -> [tilt_x, tilt_y]
    res = _run(GetPiezoTilt, "Piezo_TiltGet", [0.5, -0.3])
    assert res.success
    assert res.data["tilt_x_deg"] == 0.5
    assert res.data["tilt_y_deg"] == -0.3


def test_get_piezo_sensitivity_field_order():
    # ResponseTypes ["f","f","f"] -> [sx, sy, sz]
    res = _run(GetPiezoSensitivity, "Piezo_SensGet", [1e-8, 2e-8, 3e-8])
    assert res.success
    assert res.data["sens_x"] == 1e-8
    assert res.data["sens_y"] == 2e-8
    assert res.data["sens_z"] == 3e-8


def test_get_piezo_hva_info_field_order():
    # ResponseTypes ["f","f","f","f","I","I","I"]:
    # [gain_aux, gain_x, gain_y, gain_z, xy_en, z_en, aux_en]
    res = _run(
        GetPiezoHVAInfo, "Piezo_HVAInfoGet",
        [10.0, 14.5, 15.5, 16.5, 1, 0, 1],
    )
    assert res.success
    assert res.data["gain_aux"] == 10.0
    assert res.data["gain_x"] == 14.5
    assert res.data["gain_y"] == 15.5
    assert res.data["gain_z"] == 16.5
    assert res.data["xy_enabled"] is True
    assert res.data["z_enabled"] is False
    assert res.data["aux_enabled"] is True


def test_get_piezo_hva_status_led_field_order():
    # ResponseTypes ["I","I","I","I"]:
    # [overheated, hv_supply, high_temperature, output_connector]
    res = _run(
        GetPiezoHVAStatusLED, "Piezo_HVAStatusLEDGet",
        [0, 1, 0, 1],
    )
    assert res.success
    assert res.data["overheated"] is False
    assert res.data["hv_supply"] is True
    assert res.data["high_temperature"] is False
    assert res.data["output_connector"] is True


def test_get_piezo_xyz_limits_field_order():
    # ResponseTypes ["H","f","f","f","f","f","f"]:
    # [enabled, x_low, x_high, y_low, y_high, z_low, z_high]
    res = _run(
        GetPiezoXYZLimits, "Piezo_XYZLimitsGet",
        [1, -5.0, 5.0, -4.0, 4.0, -3.0, 3.0],
    )
    assert res.success
    assert res.data["limits_enabled"] is True
    assert res.data["x_low_v"] == -5.0
    assert res.data["x_high_v"] == 5.0
    assert res.data["y_low_v"] == -4.0
    assert res.data["y_high_v"] == 4.0
    assert res.data["z_low_v"] == -3.0
    assert res.data["z_high_v"] == 3.0


def test_load_piezo_hysteresis_file_passes_single_arg():
    # Piezo.HystFileLoad(File_path) — wire ["+*c"] auto-prepends size.
    # We must pass ONLY the path string, not (len, path).
    canned = {"Piezo_HystFileLoad": {"return_value": ("", b"", [])}}
    ctx = FakeCtx(canned=canned)
    res = LoadPiezoHysteresisFile().execute(ctx, {"file_path": "C:/hyst.csv"})
    assert res.success
    load_calls = [c for c in ctx.calls if c[0] == "Piezo_HystFileLoad"]
    assert len(load_calls) == 1
    # Exactly one positional arg, equal to the path (no leading length int).
    assert load_calls[0][1] == ("C:/hyst.csv",)


# ── Safety-fix tests #3: calibration/scale setters reject 0/neg/absurd ────────


def _piezo_spec(skill_cls, param_name):
    return next(
        p for p in skill_cls().metadata().parameters if p.name == param_name
    )


def test_set_piezo_range_has_finite_positive_bounds():
    # Was min 0.0 only (allowed a degenerate 0 range). Now a small positive …
    # 1e-3 m sanity ceiling on every axis.
    for axis in ("range_x_m", "range_y_m", "range_z_m"):
        spec = _piezo_spec(SetPiezoRange, axis)
        assert spec.min_value == 1e-12
        assert spec.max_value == 1e-3
    # 0 and negatives are now rejected; absurd values too.
    errs = SetPiezoRange().validate_params(
        {"range_x_m": 0.0, "range_y_m": -1e-6, "range_z_m": 1.0}
    )
    assert any("range_x_m" in e for e in errs)   # 0 below minimum
    assert any("range_y_m" in e for e in errs)   # negative below minimum
    assert any("range_z_m" in e and "maximum" in e for e in errs)  # 1 m absurd
    # A realistic range passes.
    assert SetPiezoRange().validate_params(
        {"range_x_m": 1e-6, "range_y_m": 1e-6, "range_z_m": 5e-7}
    ) == []


def test_set_piezo_sensitivity_has_finite_positive_bounds():
    for axis in ("sens_x", "sens_y", "sens_z"):
        spec = _piezo_spec(SetPiezoSensitivity, axis)
        assert spec.min_value == 1e-12
        assert spec.max_value == 1e-3
    errs = SetPiezoSensitivity().validate_params(
        {"sens_x": 0.0, "sens_y": -1e-9, "sens_z": 1.0}
    )
    assert any("sens_x" in e for e in errs)
    assert any("sens_y" in e for e in errs)
    assert any("sens_z" in e and "maximum" in e for e in errs)
    assert SetPiezoSensitivity().validate_params(
        {"sens_x": 1e-8, "sens_y": 1e-8, "sens_z": 1e-8}
    ) == []


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
