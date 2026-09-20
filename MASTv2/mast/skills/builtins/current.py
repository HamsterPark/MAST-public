"""Current amplifier gain skill.

vendored from v1 mast/skills/builtins/current.py 2026-04-23"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill


class SetCurrentGain(BaseSkill):
    """Set the current amplifier gain."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetCurrentGain",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置电流放大器的增益档位与滤波器。",
            parameters=[
                ParameterSpec(
                    name="gain_index",
                    type="int",
                    description="增益档位索引，取自 Current.GainsGet 返回的列表",
                    required=True,
                    min_value=0,
                ),
                ParameterSpec(
                    name="filter_index",
                    type="int",
                    description="滤波器索引",
                    required=False,
                    default=0,
                    min_value=0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["current", "gain", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        gain = params["gain_index"]
        filt = params.get("filter_index", 0)
        # Current_GainSet(Gain_index, Filter_Index)
        record = context.safe_call("Current_GainSet", gain, filt)
        if record.error:
            return SkillResult(
                skill_name="SetCurrentGain",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetCurrentGain",
            success=True,
            data={"gain_index": gain, "filter_index": filt},
            nanonis_calls=[record],
        )


# ---------------------------------------------------------------------------
# Current — BEEMGet, CalibrSet
# ---------------------------------------------------------------------------


class GetCurrentGains(BaseSkill):
    """读回前放的增益档:有哪些档、现在是哪一档、这一档的满量程是多少。

    ``SetCurrentGain`` 一直都在,读回却没有 —— 于是「先把量程开够再降设定点」这一步做不了:
    量程不知道,就只能要么盲改、要么不改。横向操纵要 50 nA 量级的设定点,而成像量程往往是
    10 nA;设定点超出量程时电流读数一路饱和,Z 环看不到它到达目标,就一直伸下去直到撞上表面。
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetCurrentGains",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读回电流前放的增益档位表、当前档位与该档的满量程(安培)。只读,不改硬件。"
                "增益名是跨阻(V/A),满量程 = 10 V / 跨阻。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["read", "readback", "current", "preamp", "gain"],
            parameters=[],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Current_GainsGet")
        if record.error:
            return SkillResult(skill_name="GetCurrentGains", success=False,
                               error=record.error, nanonis_calls=[record])
        data: dict = {}
        parsed = record.return_value
        vals = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(vals, (list, tuple)) and len(vals) >= 4:
            gains = vals[2] if isinstance(vals[2], (list, tuple)) else []
            data["gains"] = [str(g) for g in gains]
            try:
                idx = int(vals[3])
            except (TypeError, ValueError):
                idx = None
            data["gain_index"] = idx
            if idx is not None and 0 <= idx < len(gains):
                data["gain"] = str(gains[idx])
                try:
                    # the gain name is a transimpedance in V/A; the DAC swings +-10 V
                    data["full_scale_a"] = 10.0 / float(str(gains[idx]).replace(" ", ""))
                except (TypeError, ValueError):
                    data["full_scale_a"] = None
        fs = data.get("full_scale_a")
        summary = ("增益档 %s(第 %s 档),满量程 %.3g A"
                   % (data.get("gain"), data.get("gain_index"), fs)) if fs else "读回增益档"
        return SkillResult(skill_name="GetCurrentGains", success=True, data=data,
                           summary=summary, nanonis_calls=[record])


class GetCurrentBEEM(BaseSkill):
    """Get the BEEM current value."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetCurrentBEEM",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="从 Current 模块读取 BEEM 电流值。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["current", "beem", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Current_BEEMGet")
        if record.error:
            return SkillResult(
                skill_name="GetCurrentBEEM",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, parsed_list).
        # Current.BEEMGet ResponseTypes=["f"] -> parsed[2][0] is the BEEM
        # current (A). Reading parsed[0] gives the empty error string and
        # float("") raised ValueError on real hardware (call always failed).
        parsed = record.return_value
        beem_a = 0.0
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                beem_a = float(vals[0])
            elif isinstance(vals, (int, float)):
                beem_a = float(vals)
        elif isinstance(parsed, (int, float)):
            beem_a = float(parsed)
        return SkillResult(
            skill_name="GetCurrentBEEM",
            success=True,
            data={"beem_current_a": beem_a},
            nanonis_calls=[record],
        )


class SetCurrentCalibration(BaseSkill):
    """Set the calibration and offset for a current gain."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetCurrentCalibration",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="为 Current 模块中选定的增益档位设置标定值与偏移量。",
            parameters=[
                ParameterSpec(
                    name="gain_index",
                    type="int",
                    description="增益档位索引（-1 = 当前选中的增益档）",
                    required=False,
                    default=-1,
                ),
                ParameterSpec(
                    name="calibration",
                    type="float",
                    description="标定值（float64，A/A 倍率）",
                    required=True,
                    # Sanity ceiling (rig-tunable), NOT a precise calibration
                    # window. Nominal is ~1.0; bound to 1e-3 … 1e3 so an agent
                    # can't set an extreme multiplier that silently rescales the
                    # measured current and defeats the setpoint bounds. Rejects
                    # 0 and negatives.
                    min_value=1e-3,
                    max_value=1e3,
                ),
                ParameterSpec(
                    name="offset",
                    type="float",
                    description="偏移量（float64，A）",
                    unit="A",
                    required=True,
                    # Sanity ceiling (rig-tunable): a huge additive offset also
                    # corrupts the current reading. ±1.0 A is generous-but-finite
                    # (real STM currents are pA…nA), so it only rejects absurd
                    # values.
                    min_value=-1.0,
                    max_value=1.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["current", "calibration", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        gain_index = params.get("gain_index", -1)
        calibration = params["calibration"]
        offset = params["offset"]
        record = context.safe_call(
            "Current_CalibrSet", gain_index, calibration, offset,
        )
        if record.error:
            return SkillResult(
                skill_name="SetCurrentCalibration",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetCurrentCalibration",
            success=True,
            data={
                "gain_index": gain_index,
                "calibration": calibration,
                "offset": offset,
            },
            nanonis_calls=[record],
        )
