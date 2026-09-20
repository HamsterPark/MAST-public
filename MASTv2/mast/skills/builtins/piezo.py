"""Piezo calibration, drift compensation, and HVA skills.

vendored from v1 mast/skills/builtins/piezo.py 2026-04-23"""

from __future__ import annotations

from typing import List

from mast.io.nanonis_files import decode_reply
from mast.skills.base import BaseSkill
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)


class SetDriftCompensation(BaseSkill):
    """Enable/disable piezo drift compensation."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetDriftCompensation",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="启用或禁用 piezo 漂移补偿。",
            parameters=[
                ParameterSpec(
                    name="enable",
                    type="bool",
                    description="True 为启用漂移补偿",
                    required=True,
                ),
                # Drift speeds are signed (drift can go either way). Real thermal
                # drift is ≪ 1 nm/s; ±1 µm/s is a generous cap that still rejects
                # a hallucinated absurd value (e.g. vx=1.0 = 1 m/s would rip the
                # tip across the sample) (2026-07-03 review).
                ParameterSpec(
                    name="vx",
                    type="float",
                    description="X 向漂移速度（m/s）；默认 0.0（不做 X 轴补偿）。",
                    unit="m/s",
                    required=False,
                    default=0.0,
                    min_value=-1e-6,
                    max_value=1e-6,
                ),
                ParameterSpec(
                    name="vy",
                    type="float",
                    description="Y 向漂移速度（m/s）；默认 0.0（不做 Y 轴补偿）。",
                    unit="m/s",
                    required=False,
                    default=0.0,
                    min_value=-1e-6,
                    max_value=1e-6,
                ),
                ParameterSpec(
                    name="vz",
                    type="float",
                    description="Z 向漂移速度（m/s）；默认 0.0（不做 Z 轴补偿）。",
                    unit="m/s",
                    required=False,
                    default=0.0,
                    min_value=-1e-6,
                    max_value=1e-6,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "drift", "compensation"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        enable = params["enable"]
        all_calls = []

        # Piezo_DriftCompSet(Compensation_on_off, Vx_m_s, Vy_m_s, Vz_m_s, Sat_Lim)
        # on_off: 0=no change, 1=On, 2=Off; Sat_Lim: saturation limit (0=no limit)
        if enable:
            vx = params.get("vx", 0.0)
            vy = params.get("vy", 0.0)
            vz = params.get("vz", 0.0)
            rec = context.safe_call("Piezo_DriftCompSet", 1, vx, vy, vz, 0)
        else:
            rec = context.safe_call("Piezo_DriftCompSet", 2, 0.0, 0.0, 0.0, 0)
        all_calls.append(rec)

        if rec.error:
            return SkillResult(
                skill_name="SetDriftCompensation",
                success=False,
                error=rec.error,
                nanonis_calls=all_calls,
            )

        return SkillResult(
            skill_name="SetDriftCompensation",
            success=True,
            data={"enabled": enable},
            nanonis_calls=all_calls,
        )


class GetDriftCompensation(BaseSkill):
    """Read current drift compensation settings."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetDriftCompensation",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 piezo 漂移补偿的开关状态与各向速度。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "drift", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Piezo_DriftCompGet")
        if record.error:
            return SkillResult(
                skill_name="GetDriftCompensation",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # Piezo.DriftCompGet ResponseTypes = ["I","f","f","f","I","I","I","f"]
        # parsed[2] = [status, Vx, Vy, Vz, X_sat, Y_sat, Z_sat]
        # NOTE: status (the On/Off flag) is index 0, NOT a velocity. The
        # velocities are indices 1..3. Reading vals[0] as Vx (and vals[3] as
        # "enabled") mislabels the on/off switch as a drift speed and silently
        # reports Vz as the compensation status.
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 4:
                data = {
                    "enabled": bool(int(vals[0])),
                    "vx": float(vals[1]),
                    "vy": float(vals[2]),
                    "vz": float(vals[3]),
                }
                if len(vals) >= 7:
                    data["x_saturated"] = bool(int(vals[4]))
                    data["y_saturated"] = bool(int(vals[5]))
                    data["z_saturated"] = bool(int(vals[6]))
        return SkillResult(
            skill_name="GetDriftCompensation",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_TiltSet / Piezo_TiltGet
# ---------------------------------------------------------------------------


class SetPiezoTilt(BaseSkill):
    """Set piezo tilt correction angles."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPiezoTilt",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 piezo X、Y 轴的倾斜校正角度。",
            parameters=[
                ParameterSpec(
                    name="tilt_x_deg",
                    type="float",
                    description="X 向的倾斜校正角度（度）",
                    unit="deg",
                    required=True,
                ),
                ParameterSpec(
                    name="tilt_y_deg",
                    type="float",
                    description="Y 向的倾斜校正角度（度）",
                    unit="deg",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "tilt", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        tilt_x = params["tilt_x_deg"]
        tilt_y = params["tilt_y_deg"]
        # Piezo.TiltSet(Tilt_X_deg, Tilt_Y_deg)
        record = context.safe_call("Piezo_TiltSet", tilt_x, tilt_y)
        if record.error:
            return SkillResult(
                skill_name="SetPiezoTilt",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPiezoTilt",
            success=True,
            data={"tilt_x_deg": tilt_x, "tilt_y_deg": tilt_y},
            nanonis_calls=[record],
        )


class GetPiezoTilt(BaseSkill):
    """Read piezo tilt correction angles."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPiezoTilt",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取 piezo X、Y 轴的倾斜校正角度。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "tilt", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Piezo_TiltGet")
        if record.error:
            return SkillResult(
                skill_name="GetPiezoTilt",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # Piezo.TiltGet ResponseTypes = ["f","f"] -> parsed[2] = [tilt_x, tilt_y]
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                data = {
                    "tilt_x_deg": float(vals[0]),
                    "tilt_y_deg": float(vals[1]),
                }
        return SkillResult(
            skill_name="GetPiezoTilt",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_RangeSet
# ---------------------------------------------------------------------------


class SetPiezoRange(BaseSkill):
    """Set piezo range for X, Y, Z axes."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPiezoRange",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置全部 3 个轴的 piezo 量程（m）。"
                "改量程的同时也会改 sensitivity（HV gain 不变）。"
            ),
            parameters=[
                ParameterSpec(
                    name="range_x_m",
                    type="float",
                    description="X 轴的 piezo 量程（m）",
                    unit="m",
                    required=True,
                    # Sanity ceiling (rig-tunable), NOT the true scanner range.
                    # Was min 0.0 (allowed a degenerate 0 range). Bound to a
                    # small positive … 1e-3 m (1 mm) so the V↔m calibration can't
                    # be set to 0/negative or an absurd value that defeats the
                    # ±1.5 µm XY position bounds.
                    min_value=1e-12,
                    max_value=1e-3,
                ),
                ParameterSpec(
                    name="range_y_m",
                    type="float",
                    description="Y 轴的 piezo 量程（m）",
                    unit="m",
                    required=True,
                    min_value=1e-12,   # sanity ceiling (rig-tunable); reject 0/neg
                    max_value=1e-3,
                ),
                ParameterSpec(
                    name="range_z_m",
                    type="float",
                    description="Z 轴的 piezo 量程（m）",
                    unit="m",
                    required=True,
                    min_value=1e-12,   # sanity ceiling (rig-tunable); reject 0/neg
                    max_value=1e-3,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "range", "calibration", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rx = params["range_x_m"]
        ry = params["range_y_m"]
        rz = params["range_z_m"]
        # Piezo.RangeSet(Range_X, Range_Y, Range_Z)
        record = context.safe_call("Piezo_RangeSet", rx, ry, rz)
        if record.error:
            return SkillResult(
                skill_name="SetPiezoRange",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPiezoRange",
            success=True,
            data={"range_x_m": rx, "range_y_m": ry, "range_z_m": rz},
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_SensSet / Piezo_SensGet
# ---------------------------------------------------------------------------


class SetPiezoSensitivity(BaseSkill):
    """Set piezo sensitivity (m/V) for X, Y, Z axes."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPiezoSensitivity",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置全部 3 个轴的 piezo sensitivity（m/V）。"
                "改 sensitivity 的同时也会改量程（HV gain 不变）。"
            ),
            parameters=[
                ParameterSpec(
                    name="sens_x",
                    type="float",
                    description="X 轴的 sensitivity（m/V）",
                    unit="m/V",
                    required=True,
                    # Sanity ceiling (rig-tunable), NOT a precise calibration
                    # window. Was min 0.0 (allowed a degenerate 0 sensitivity).
                    # Bound to a small positive … 1e-3 m/V so the V↔m calibration
                    # can't be set to 0/negative or an absurd value that defeats
                    # the ±1.5 µm position bounds. Real piezos are ~1e-9–1e-7 m/V.
                    min_value=1e-12,
                    max_value=1e-3,
                ),
                ParameterSpec(
                    name="sens_y",
                    type="float",
                    description="Y 轴的 sensitivity（m/V）",
                    unit="m/V",
                    required=True,
                    min_value=1e-12,   # sanity ceiling (rig-tunable); reject 0/neg
                    max_value=1e-3,
                ),
                ParameterSpec(
                    name="sens_z",
                    type="float",
                    description="Z 轴的 sensitivity（m/V）",
                    unit="m/V",
                    required=True,
                    min_value=1e-12,   # sanity ceiling (rig-tunable); reject 0/neg
                    max_value=1e-3,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "sensitivity", "calibration", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        sx = params["sens_x"]
        sy = params["sens_y"]
        sz = params["sens_z"]
        # Piezo.SensSet(Calibration_X, Calibration_Y, Calibration_Z)
        record = context.safe_call("Piezo_SensSet", sx, sy, sz)
        if record.error:
            return SkillResult(
                skill_name="SetPiezoSensitivity",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPiezoSensitivity",
            success=True,
            data={"sens_x": sx, "sens_y": sy, "sens_z": sz},
            nanonis_calls=[record],
        )


class GetPiezoSensitivity(BaseSkill):
    """Read piezo sensitivity (m/V) for X, Y, Z axes."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPiezoSensitivity",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取全部 3 个轴的 piezo sensitivity（m/V）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "sensitivity", "calibration", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Piezo_SensGet")
        if record.error:
            return SkillResult(
                skill_name="GetPiezoSensitivity",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # Piezo.SensGet ResponseTypes = ["f","f","f"] -> parsed[2] = [sx, sy, sz]
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                data = {
                    "sens_x": float(vals[0]),
                    "sens_y": float(vals[1]),
                    "sens_z": float(vals[2]),
                }
        return SkillResult(
            skill_name="GetPiezoSensitivity",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_HVAInfoGet
# ---------------------------------------------------------------------------


class GetPiezoHVAInfo(BaseSkill):
    """Read HVA (High Voltage Amplifier) gain information."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPiezoHVAInfo",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读取 AUX、X、Y、Z 各轴的 HVA gain 回读信息，"
                "以及它们的启用状态。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "hva", "gain", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Piezo_HVAInfoGet")
        if record.error:
            return SkillResult(
                skill_name="GetPiezoHVAInfo",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # Piezo.HVAInfoGet ResponseTypes = ["f","f","f","f","I","I","I"]
        # parsed[2] = [gain_aux, gain_x, gain_y, gain_z, xy_en, z_en, aux_en]
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 7:
                data = {
                    "gain_aux": float(vals[0]),
                    "gain_x": float(vals[1]),
                    "gain_y": float(vals[2]),
                    "gain_z": float(vals[3]),
                    "xy_enabled": bool(int(vals[4])),
                    "z_enabled": bool(int(vals[5])),
                    "aux_enabled": bool(int(vals[6])),
                }
        return SkillResult(
            skill_name="GetPiezoHVAInfo",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_HVAStatusLEDGet
# ---------------------------------------------------------------------------


class GetPiezoHVAStatusLED(BaseSkill):
    """Read HVA status LED information."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPiezoHVAStatusLED",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读取 HVA 的 LED 状态：过热、HV 供电、"
                "高温、输出接口。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "hva", "status", "led", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Piezo_HVAStatusLEDGet")
        if record.error:
            return SkillResult(
                skill_name="GetPiezoHVAStatusLED",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # Piezo.HVAStatusLEDGet ResponseTypes = ["I","I","I","I"]
        # parsed[2] = [overheated, hv_supply, high_temperature, output_connector]
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 4:
                data = {
                    "overheated": bool(int(vals[0])),
                    "hv_supply": bool(int(vals[1])),
                    "high_temperature": bool(int(vals[2])),
                    "output_connector": bool(int(vals[3])),
                }
        return SkillResult(
            skill_name="GetPiezoHVAStatusLED",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_XYZLimitsGet
# ---------------------------------------------------------------------------


class GetPiezoXYZLimits(BaseSkill):
    """Read XYZ voltage limits from Piezo Calibration module."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetPiezoXYZLimits",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="从 Piezo Calibration 读取 XYZ 的电压上限与启用状态。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "limits", "voltage", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Piezo_XYZLimitsGet")
        if record.error:
            return SkillResult(
                skill_name="GetPiezoXYZLimits",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # Piezo.XYZLimitsGet ResponseTypes = ["H","f","f","f","f","f","f"]
        # parsed[2] = [enabled, x_low, x_high, y_low, y_high, z_low, z_high]
        parsed = record.return_value
        data = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 7:
                data = {
                    "limits_enabled": bool(int(vals[0])),
                    "x_low_v": float(vals[1]),
                    "x_high_v": float(vals[2]),
                    "y_low_v": float(vals[3]),
                    "y_high_v": float(vals[4]),
                    "z_low_v": float(vals[5]),
                    "z_high_v": float(vals[6]),
                }
        return SkillResult(
            skill_name="GetPiezoXYZLimits",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_HystOnOffSet
# ---------------------------------------------------------------------------


class SetPiezoHysteresisOnOff(BaseSkill):
    """Enable or disable piezo hysteresis compensation."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPiezoHysteresisOnOff",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="启用或禁用 Piezo Configuration 中的 hysteresis 补偿。",
            parameters=[
                ParameterSpec(
                    name="enable",
                    type="bool",
                    description="True 为启用 hysteresis 补偿，False 为禁用",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["piezo", "hysteresis", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        enable = params["enable"]
        # Piezo.HystOnOffSet(On/Off) — 0=Off, 1=On
        on_off = 1 if enable else 0
        record = context.safe_call("Piezo_HystOnOffSet", on_off)
        if record.error:
            return SkillResult(
                skill_name="SetPiezoHysteresisOnOff",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPiezoHysteresisOnOff",
            success=True,
            data={"enabled": enable},
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_HystValsSet
# ---------------------------------------------------------------------------


class SetPiezoHysteresisValues(BaseSkill):
    """Set and apply hysteresis compensation values."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetPiezoHysteresisValues",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在 Piezo Calibration 模块中，为 fast 轴与 slow 轴设置并应用 "
                "hysteresis 补偿点。"
            ),
            parameters=[
                ParameterSpec(
                    name="fast_x",
                    type="str",
                    description="fast 轴 X 的 hysteresis 补偿点，用浮点数 JSON 列表表示",
                    required=True,
                ),
                ParameterSpec(
                    name="fast_y",
                    type="str",
                    description="fast 轴 Y 的 hysteresis 补偿点，用浮点数 JSON 列表表示",
                    required=True,
                ),
                ParameterSpec(
                    name="slow_x",
                    type="str",
                    description="slow 轴 X 的 hysteresis 补偿点，用浮点数 JSON 列表表示",
                    required=True,
                ),
                ParameterSpec(
                    name="slow_y",
                    type="str",
                    description="slow 轴 Y 的 hysteresis 补偿点，用浮点数 JSON 列表表示",
                    required=True,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["piezo", "hysteresis", "calibration", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import json

        try:
            fast_x: List[float] = json.loads(params["fast_x"])
            fast_y: List[float] = json.loads(params["fast_y"])
            slow_x: List[float] = json.loads(params["slow_x"])
            slow_y: List[float] = json.loads(params["slow_y"])
        except (json.JSONDecodeError, TypeError) as exc:
            return SkillResult(
                skill_name="SetPiezoHysteresisValues",
                success=False,
                error=f"Invalid JSON in hysteresis points: {exc}",
                nanonis_calls=[],
            )

        # VALID JSON IS NOT ENOUGH — `"5"` parses to the integer 5, and the length
        # checks below then raise TypeError, which reaches the agent as an unhandled
        # exception rather than a SkillResult. Same trap as SetPatternCloud.
        for label, seq in (("fast_x", fast_x), ("fast_y", fast_y),
                           ("slow_x", slow_x), ("slow_y", slow_y)):
            if not isinstance(seq, list):
                return SkillResult(
                    skill_name="SetPiezoHysteresisValues",
                    success=False,
                    error=(f"{label} 必须是 JSON 数组，例如 \"[0.1, 0.2]\"；"
                           f"收到的是 {type(seq).__name__}（{params[label]!r}）"),
                    nanonis_calls=[],
                )

        # Paired arrays must match in length: the wire prefixes EACH array with
        # its own size, and Nanonis treats fast_x/fast_y (and slow_x/slow_y) as
        # paired hysteresis points. Validate up-front so a length mismatch fails
        # clearly instead of corrupting the request body.
        if len(fast_x) != len(fast_y) or len(slow_x) != len(slow_y):
            return SkillResult(
                skill_name="SetPiezoHysteresisValues",
                success=False,
                error=(
                    "Hysteresis point arrays must be paired and equal-length: "
                    f"len(fast_x)={len(fast_x)} vs len(fast_y)={len(fast_y)}, "
                    f"len(slow_x)={len(slow_x)} vs len(slow_y)={len(slow_y)}."
                ),
                nanonis_calls=[],
            )

        # Piezo.HystValsSet(N_fast, fast_x[], N_fast, fast_y[],
        #                   N_slow, slow_x[], N_slow, slow_y[])
        # Each array is wire-prefixed by its OWN length — use the per-array count.
        # (Was: len(fast_x) reused for fast_y and len(slow_x) for slow_y, which
        # would mis-size the y arrays on mismatched-length input — fixed 2026-06-26.)
        record = context.safe_call(
            "Piezo_HystValsSet",
            len(fast_x), fast_x,
            len(fast_y), fast_y,
            len(slow_x), slow_x,
            len(slow_y), slow_y,
        )
        if record.error:
            return SkillResult(
                skill_name="SetPiezoHysteresisValues",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetPiezoHysteresisValues",
            success=True,
            data={
                "fast_axis_points": len(fast_x),
                "slow_axis_points": len(slow_x),
            },
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Piezo_HystFileLoad
# ---------------------------------------------------------------------------


class LoadPiezoHysteresisFile(BaseSkill):
    """Load hysteresis compensation values from a CSV file."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="LoadPiezoHysteresisFile",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在 Piezo Configuration 模块中，从一个 .csv 文件载入并应用"
                "两个轴的 hysteresis 补偿值。"
            ),
            parameters=[
                ParameterSpec(
                    name="file_path",
                    type="str",
                    description=".csv hysteresis 文件的路径",
                    required=True,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["piezo", "hysteresis", "file", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        file_path = params["file_path"]
        # Piezo.HystFileLoad(File_path) — single arg. The wire format is
        # ["+*c"]: quickSend auto-prepends the int32 path-size, so we must NOT
        # pass len(file_path) ourselves (that injects a spurious leading int and
        # corrupts the request body).
        record = context.safe_call(
            "Piezo_HystFileLoad", file_path,
        )
        if record.error:
            return SkillResult(
                skill_name="LoadPiezoHysteresisFile",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="LoadPiezoHysteresisFile",
            success=True,
            data={"file_path": file_path},
            nanonis_calls=[record],
        )
