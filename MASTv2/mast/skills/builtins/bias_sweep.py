"""Bias Sweeper + the last few loose Nanonis calls MAST never wrapped.

Three small gaps, closed together because none of them is worth a file:

  * **``BiasSwp_*`` (5 methods)** — the Bias Sweeper. Distinct from
    ``BiasSpectr_*`` (bias spectroscopy, which MAST does cover): the sweeper ramps
    the bias and records the acquisition channels along the way, without the
    spectroscopy module's Z-control gymnastics. It is what you want for an I–V
    curve at a fixed height, or a bias ramp while a lock-in watches.

  * **``Signals_CalibrGet`` / ``Signals_AddRTSet``** — read a signal's calibration
    (so a raw value can be turned into physical units), and pick the two extra
    real-time signals Nanonis streams. MAST covered 6 of the 8 ``Signals_*`` calls
    and stopped one short of being able to say what a number MEANS.

  * **``Util_AcqPeriodSet``** — the controller's acquisition period. MAST could
    READ it (``GetAcqPeriod``) and not set it, which is an odd place to stop.
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill


def _values(record) -> list:
    rv = getattr(record, "return_value", None)
    if isinstance(rv, (list, tuple)) and len(rv) > 2 and isinstance(rv[2], (list, tuple)):
        return list(rv[2])
    return []


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


class RunBiasSweep(BaseSkill):
    """Ramp the bias and record the acquisition channels along the way."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunBiasSweep",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "运行 Nanonis 的 BIAS SWEEPER：把偏压在两个限值之间 ramp，并"
                "记录采集通道。它**不是** bias spectroscopy"
                "（BiasSpectr / RunSTS）—— sweeper 只管 ramp 和记录，没有 "
                "spectroscopy 模块的那套 Z-control 时序。用它做定高的 I–V，"
                "或者在 lock-in 盯着时做一次偏压 ramp。\n\n"
                "针尖留在原地不动。挑限值时要像用 SetBias 那样谨慎 —— "
                "一次停在 5 V 的 sweep，会把偏压就留在 5 V。"
            ),
            parameters=[
                ParameterSpec(
                    name="lower_limit_v", type="float",
                    description="sweep 的偏压下限（V）",
                    unit="V", required=True, min_value=-10.0, max_value=10.0,
                ),
                ParameterSpec(
                    name="upper_limit_v", type="float",
                    description="sweep 的偏压上限（V）",
                    unit="V", required=True, min_value=-10.0, max_value=10.0,
                ),
                ParameterSpec(
                    name="steps", type="int",
                    description="sweep 的点数",
                    required=False, default=256, min_value=2, max_value=65535,
                ),
                ParameterSpec(
                    name="period_ms", type="int",
                    description="每个点的耗时（ms）",
                    unit="ms", required=False, default=10,
                    min_value=1, max_value=65535,
                ),
                ParameterSpec(
                    name="z_controller_off", type="bool",
                    description=(
                        "sweep 期间断开 Z feedback 环路。做定高 I–V 时通常选 "
                        "True；选 False 则偏压变动时 feedback 会一直追着 "
                        "setpoint 走。"
                    ),
                    required=False, default=True,
                ),
                ParameterSpec(
                    name="sweep_direction", type="int",
                    description="1 = 下限→上限，0 = 上限→下限",
                    required=False, default=1, min_value=0, max_value=1,
                ),
                ParameterSpec(
                    name="autosave", type="bool",
                    description="把这次 sweep 存成文件，落在 Nanonis 那台机器上",
                    required=False, default=True,
                ),
            ],
            preconditions=[],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["bias", "sweep", "spectroscopy", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        lo = float(params["lower_limit_v"])
        hi = float(params["upper_limit_v"])
        if lo > hi:
            lo, hi = hi, lo
        steps = int(params.get("steps", 256) or 256)
        period = int(params.get("period_ms", 10) or 10)
        zoff = 1 if params.get("z_controller_off", True) else 0
        direction = int(params.get("sweep_direction", 1) or 1)
        autosave = 1 if params.get("autosave", True) else 0
        calls = []

        rec = context.safe_call("BiasSwp_Open")
        calls.append(rec)
        if rec.error:
            return _fail("RunBiasSweep", f"BiasSwp_Open failed: {rec.error}", calls)

        rec = context.safe_call("BiasSwp_LimitsSet", lo, hi)
        calls.append(rec)
        if rec.error:
            return _fail("RunBiasSweep", f"BiasSwp_LimitsSet failed: {rec.error}", calls)

        # BiasSwp_PropsSet(Number_of_steps, Period_ms, Autosave, Save_dialog, Settling_ms)
        rec = context.safe_call("BiasSwp_PropsSet", steps, period, autosave, 0, period)
        calls.append(rec)
        if rec.error:
            return _fail("RunBiasSweep", f"BiasSwp_PropsSet failed: {rec.error}", calls)

        # BiasSwp_Start(Get_data, Sweep_direction, Z_Controller_status, Save_base_name, Reset_bias)
        rec = context.safe_call("BiasSwp_Start", 1, direction, zoff, "mast_biasswp", 1)
        calls.append(rec)
        if rec.error:
            return _fail("RunBiasSweep", f"BiasSwp_Start failed: {rec.error}", calls)

        return SkillResult(
            skill_name="RunBiasSweep", success=True,
            data={"lower_limit_v": lo, "upper_limit_v": hi, "steps": steps,
                  "period_ms": period, "z_controller_off": bool(zoff),
                  "data": _values(rec)},
            nanonis_calls=calls,
        )


class GetSignalCalibration(BaseSkill):
    """What does this signal's number MEAN? (gain + offset → physical units)"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSignalCalibration",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读取某路信号的标定（gain + offset）—— 也就是原始值的一个单位"
                "对应多少物理量。MAST 一直能列出信号、也能读到它们的值，却说不出"
                "这些数字**到底是什么意思**。在解读一条不是你配置的原始通道之前，"
                "先用它。"
            ),
            parameters=[
                ParameterSpec(
                    name="signal_index", type="int",
                    description="信号索引（0..127）—— 见 ListSignalNames",
                    required=True, min_value=0, max_value=127,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["signals", "calibration", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx = int(params["signal_index"])
        rec = context.safe_call("Signals_CalibrGet", idx)
        if rec.error:
            return _fail("GetSignalCalibration", rec.error, [rec])
        v = _values(rec)
        return SkillResult(
            skill_name="GetSignalCalibration", success=True,
            data={"signal_index": idx,
                  "calibration": float(v[0]) if len(v) > 0 else None,
                  "offset": float(v[1]) if len(v) > 1 else None},
            nanonis_calls=[rec],
        )


class SetAdditionalRealtimeSignals(BaseSkill):
    """Choose the two extra signals Nanonis streams in real time."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetAdditionalRealtimeSignals",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "选择 Nanonis 额外计算并输出的那**两路**实时信号"
                "（在固定的那几路之外）。它只是配置 —— 改变的是测什么，"
                "绝不改变仪器做什么。"
            ),
            parameters=[
                ParameterSpec(
                    name="signal_1", type="int",
                    description="第一路额外 RT 信号的索引",
                    required=True, min_value=0, max_value=127,
                ),
                ParameterSpec(
                    name="signal_2", type="int",
                    description="第二路额外 RT 信号的索引",
                    required=True, min_value=0, max_value=127,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["signals", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        s1 = int(params["signal_1"])
        s2 = int(params["signal_2"])
        rec = context.safe_call("Signals_AddRTSet", s1, s2)
        if rec.error:
            return _fail("SetAdditionalRealtimeSignals", rec.error, [rec])
        return SkillResult(
            skill_name="SetAdditionalRealtimeSignals", success=True,
            data={"signal_1": s1, "signal_2": s2}, nanonis_calls=[rec])


class SetAcquisitionPeriod(BaseSkill):
    """Set the controller's acquisition period."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetAcquisitionPeriod",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "设置 Nanonis 的采集周期（控制器的采样间隔）。MAST 以前只能"
                "**读**它（GetAcqPeriod），设不了。调小它采样更快、也更吃带宽；"
                "它影响控制器做的**每一次**测量，所以要有意识地改。"
            ),
            parameters=[
                ParameterSpec(
                    name="period_s", type="float",
                    description="采集周期，单位秒",
                    unit="s", required=True, min_value=1e-6, max_value=1.0,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["util", "acquisition", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        p = float(params["period_s"])
        rec = context.safe_call("Util_AcqPeriodSet", p)
        if rec.error:
            return _fail("SetAcquisitionPeriod", rec.error, [rec])
        return SkillResult(skill_name="SetAcquisitionPeriod", success=True,
                           data={"period_s": p}, nanonis_calls=[rec])


__all__ = ["RunBiasSweep", "GetSignalCalibration",
           "SetAdditionalRealtimeSignals", "SetAcquisitionPeriod"]
