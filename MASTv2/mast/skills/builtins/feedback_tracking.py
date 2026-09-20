# -*- coding: utf-8 -*-
"""AssessFeedbackTracking：读取扫描帧的 Z 与 Current 通道并报告反馈跟随误差。

本层负责 IO，数值判据见 mast.vision.feedback_lag。恒流模式下 Z 反映形貌，
电流反映反馈误差。电流对地形斜率的响应除以设定点后可与文件头中的
v / I_gain 交叉检查；不一致时应检查模型适用性与通道读数。

κ 无默认值，调用方应提供适用于当前数据的隧穿衰减常数。未提供 κ 时
仍报告 lag_ratio，依赖它的 fidelity 保持 None。该技能不直接调整增益。
"""
from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

_NAME = "AssessFeedbackTracking"

__all__ = ["AssessFeedbackTracking"]


class AssessFeedbackTracking(BaseSkill):
    """从一帧的 Z + 电流通道量出 Z 反馈的跟随误差。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读一帧 .sxm 的 Z 与电流两个通道，量 Z 反馈在这个扫描速度下跟得有多紧："
                "电流误差对地形斜率的斜率，除以设定点即 v/I_gain（可与文件头对账）。"
                "给了势垒常数 κ 还能换算出晶格频率上的跟随保真度。"
                "**不判针尖好坏，也不建议增益该设多少** —— 只给「现在这一档的代价」。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path", type="str",
                    description="要分析的 .sxm 文件路径（必须含 Current 通道）。",
                    required=True),
                ParameterSpec(
                    name="direction", type="str",
                    description="用正扫还是反扫。forward / backward。",
                    required=False, default="forward",
                    allowed_values=["forward", "backward"]),
                ParameterSpec(
                    name="kappa_per_nm", type="float",
                    description=(
                        "隧穿衰减常数，每纳米。**刻意没有默认值**：它只有 "
                        "MeasureBarrierHeight 量得出（κ = √φ·5.123，φ 单位 eV；"
                        "干净真空结约 10 /nm）。不给时跟随保真度留空，"
                        "lag_ratio 照样给。"),
                    unit="1/nm", required=False,
                    min_value=0.5, max_value=30.0),
                ParameterSpec(
                    name="lattice_period_nm", type="float",
                    description=(
                        "算保真度用的空间周期，纳米。留空则从这一帧自己量出的原胞取"
                        "较短的那个基矢。"),
                    unit="nm", required=False,
                    min_value=0.05, max_value=100.0),
            ],
            estimated_duration_s=3.0,
            composition_level=1,
            tags=["feedback", "current", "analysis", "read", "zcontrol"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from mast.skills.builtins._sxm_frame import load_frame
        from mast.vision.feedback_lag import measure_feedback_lag
        from mast.vision.lattice_cell import measure_cell

        path = str(params.get("scan_path") or "")
        direction = str(params.get("direction") or "forward")
        kappa = params.get("kappa_per_nm")
        period = params.get("lattice_period_nm")

        fr = load_frame(path)
        if fr.error:
            return SkillResult(skill_name=_NAME, success=False, error=fr.error)
        z = fr.forward if direction == "forward" else fr.backward
        cur = (fr.current_forward if direction == "forward"
               else fr.current_backward)
        if z is None or cur is None:
            return SkillResult(
                skill_name=_NAME, success=True,
                data={"verdict": "undetermined", "scan_path": path,
                      "direction": direction,
                      "reason": "missing_channel"},
                summary=("这一帧没有 %s 方向的 Z 或 Current 通道 —— 跟随误差只能从"
                         "电流通道读，Z 里没有这个信息。扫描时把 Current 加进采集"
                         "通道即可（ConfigureScan 的 channels 参数）。" % direction))

        period_src = "caller"
        if period is None:
            cell = measure_cell(z, fr.nm_per_px, scan_dir=fr.scan_dir)
            if cell.ok:
                period = cell.a2_nm
                period_src = "measured_from_frame"
            else:
                period_src = "unavailable"

        res = measure_feedback_lag(
            z, cur, nm_per_px=fr.nm_per_px, setpoint_a=fr.setpoint_a,
            speed_m_s=fr.speed_m_s, i_gain_m_s=fr.i_gain_m_s,
            lattice_period_nm=period,
            kappa_per_nm=float(kappa) if kappa is not None else None)

        data: dict = {
            "scan_path": path, "direction": direction,
            "verdict": "measured" if res.ok else "undetermined",
            "reason": res.reason,
            "nm_per_px": fr.nm_per_px, "width_nm": fr.width_nm,
            "setpoint_a": fr.setpoint_a, "bias_v": fr.bias_v,
            "speed_m_s": fr.speed_m_s, "i_gain_m_s": fr.i_gain_m_s,
            "p_gain_m": fr.p_gain_m, "scan_dir": fr.scan_dir,
            "lattice_period_nm": period, "lattice_period_source": period_src,
            "slope_a": res.slope_a, "lag_ratio": res.lag_ratio,
            "expected_lag_ratio": res.expected_lag_ratio,
            "ratio_disagreement": res.ratio_disagreement,
            "fit_r2": res.r2, "correlation": res.correlation,
            "residual_rms_a": res.residual_rms_a,
            "current_rms_a": res.current_rms_a,
            "omega_tau": res.omega_tau, "fidelity": res.fidelity,
            "warnings": list(res.warnings),
        }
        if kappa is None:
            data["fidelity_note"] = (
                "没给 kappa_per_nm，所以不给跟随保真度。κ 只有 MeasureBarrierHeight "
                "量得出；这不是缺省值能补的东西 —— 一个编出来的 κ 会让「这台机器没测过"
                "势垒」看不见，而保真度对 κ 是线性敏感的。")
        # 「把增益翻倍会怎样」按同一个模型外推 —— 它是预测不是建议。
        if res.ok and res.lag_ratio and res.omega_tau:
            wt2 = res.omega_tau / 2.0
            data["if_gain_doubled"] = {
                "lag_ratio": round(res.lag_ratio / 2.0, 4),
                "omega_tau": round(wt2, 4),
                "fidelity": round(1.0 / (1.0 + wt2 * wt2) ** 0.5, 4),
                "note": ("同一个一阶模型的外推：I_gain 翻倍 ⇒ lag_ratio 减半。"
                         "**这是预测，不是建议** —— 增益提高也会把电流噪声更多地"
                         "灌进 Z，那一半本技能量不了（要看扣掉滞后项之后的"
                         "residual_rms_a 在改动前后变没变）。"),
            }
        return SkillResult(skill_name=_NAME, success=True, data=data,
                           summary=_summary(data))


def _summary(data: dict) -> str:
    if data.get("verdict") != "measured":
        return "量不了跟随误差：%s。" % (data.get("reason") or "原因未知")
    s = "反馈滞后 lag_ratio %.3f" % (data.get("lag_ratio") or 0.0)
    exp = data.get("expected_lag_ratio")
    if exp:
        s += "（文件头的 v/I_gain 算得 %.3f，%s）" % (
            exp, "对上了" if (data.get("ratio_disagreement") or 1) < 0.3 else "**对不上**")
    fid = data.get("fidelity")
    if fid is not None and data.get("lattice_period_nm"):
        s += "；在 %.3f nm 的周期上 Z 只跟到真实起伏的 %.0f%%" % (
            data["lattice_period_nm"], 100 * fid)
    r2 = data.get("fit_r2")
    if r2 is not None and r2 < 0.25:
        s += "。拟合优度只有 %.2f —— 电流里主导的不是滞后项，这个数别拿去推增益" % r2
    return s + "。"


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(AssessFeedbackTracking, context_provider)
