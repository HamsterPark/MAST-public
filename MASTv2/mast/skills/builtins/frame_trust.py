# -*- coding: utf-8 -*-
"""AssessFrameTrust：报告逐行跳动，辅助解释扫描稳定性。

整帧 RMS、长宽比与晶格指标都依赖帧内容；坑、团簇或只占局部的晶格
可能改变这些指标，不能把所有异常分数都归因于针尖。

逐行中位高度差分的标准差用于观察整行抬落。它提供与整体 RMS 不同的信息，
仍需结合地形、扫描方向与噪声解释；此模块不提供样品标定的好坏分档。

晶格可信度由 AssessAtomicResolution 的 reasons 报告。
条纹与结构可能共存，单一频域能量比例无法替代对候选晶格峰的脊判据。
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

# 逐行 MAD 的工作流分档阈值（pm）。公开快照不包含样品标定数据，
# 这些阈值不能作为已验证的仪器验收标准，使用前需按当前条件验证。
_ROW_GOOD_PM = 40.0
_ROW_USABLE_PM = 250.0

#: 「大跳变」的判定：超过 5×MAD 且至少 50 pm。两条并且 —— 只用倍数会让极干净的帧
#: 把普通噪声数成跳变，只用绝对值会让粗糙帧漏报。
_BIG_JUMP_SIGMAS = 5.0
_BIG_JUMP_MIN_PM = 50.0


def _row_medians(z_pm):
    import numpy as np

    med = np.nanmedian(z_pm, axis=1)
    return med[np.isfinite(med)]


def row_jump_mad_pm(z_pm):
    """逐行中位高度差分的 MAD，乘 1.4826 换算为正态分布 σ 当量，单位 pm。与标准差相比，它较少受到少数大跳变影响；台阶与持续行抖动应结合图像上下文区分。"""
    import numpy as np

    med = _row_medians(z_pm)
    if med.size < 3:
        return None
    d = np.diff(med)
    return float(np.nanmedian(np.abs(d - np.nanmedian(d))) * 1.4826)


def row_big_jumps(z_pm):
    """大跳变的次数与总行数。

    ⚠️ **它分不开「针尖跳了一下」和「扫过一道真实台阶」** —— 两者在逐行中位高度上
    是同一个形状。要区分只能靠别的证据：同一位置重扫（针尖跳变不重复，台阶重复）、
    或者看图。这里只把次数报出来，**不据此下针尖的结论**。
    """
    import numpy as np

    med = _row_medians(z_pm)
    if med.size < 3:
        return None, None
    d = np.diff(med)
    mad = float(np.nanmedian(np.abs(d - np.nanmedian(d))) * 1.4826)
    thr = max(_BIG_JUMP_SIGMAS * mad, _BIG_JUMP_MIN_PM)
    return int((np.abs(d) > thr).sum()), int(d.size)


def row_jump_sigma_pm(z_pm):
    """差分的标准差（pm）。**辅助量** —— 它对少数几个大跳变敏感，
    所以既能反映针尖偶发跳变，也会被真实台阶抬高。判针尖看 MAD。"""
    import numpy as np

    med = _row_medians(z_pm)
    if med.size < 3:
        return None
    return float(np.nanstd(np.diff(med)))


class AssessFrameTrust(BaseSkill):
    """Judge whether a frame's badness is the tip's or the terrain's."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessFrameTrust",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "对 .sxm 文件报告逐行中位高度差分的标准差 σ，辅助观察整行跳动，只读文件。RMS 和长宽比也受地形与帧内容影响，不应把它们直接解释成针尖好坏；逐行指标同样需要结合扫描条件。这里不提供已标定的好坏分档。晶格可信度请读取 AssessAtomicResolution 的 reasons，例如 peaks_are_ridges、not_a_lattice 或 scale_gate。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path", type="str",
                    description="要评估的 .sxm 路径。", required=True),
                ParameterSpec(
                    name="channel", type="str",
                    description="形貌通道（'Z' 是标准选择）。",
                    required=False, default="Z"),
                ParameterSpec(
                    name="direction", type="str",
                    description="'forward' 或 'backward'。",
                    required=False, default="forward"),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["analysis", "tip", "quality", "针尖", "stability", "trust"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import numpy as np

        path = str(params.get("scan_path") or "")
        channel = str(params.get("channel") or "Z")
        direction = str(params.get("direction") or "forward")
        try:
            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
            from mast.vision.tilt import plane_subtract

            frames = sxm_oriented_frames(read_sxm(path), channel=channel)
            raw = frames.get(direction)
            if raw is None:
                return SkillResult(
                    skill_name="AssessFrameTrust", success=False,
                    error="通道 %r 里没有 %r 方向（有：%s）"
                          % (channel, direction, list(frames)))
            z = np.asarray(plane_subtract(np.asarray(raw, float)), float) * 1e12
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name="AssessFrameTrust", success=False,
                               error="读不了 %s：%s" % (path, exc))

        finite = np.isfinite(z)
        if finite.sum() < 100:
            return SkillResult(
                skill_name="AssessFrameTrust", success=True,
                data={"tip_verdict": "undetermined",
                      "tip_message": "有限像素不足 100 个 —— 判不了，这不是「针尖坏」。"})

        mad = row_jump_mad_pm(z)
        sigma = row_jump_sigma_pm(z)
        n_big, n_rows = row_big_jumps(z)
        rms = float(np.nanstd(z))
        data = {
            "row_jump_mad_pm": mad,          # ← 主指标
            "row_jump_sigma_pm": sigma,      # ← 辅助
            "big_jumps": n_big,
            "n_rows": n_rows,
            "rms_pm": rms,
            "pv_pm": float(np.ptp(z[finite])),
            "nan_fraction": float(1.0 - finite.mean()),
            "row_good_pm": _ROW_GOOD_PM,
            "row_usable_pm": _ROW_USABLE_PM,
        }

        if mad is None:
            data["tip_verdict"] = "undetermined"
            data["tip_message"] = "行数太少，量不出逐行跳动 —— 判不了。"
        elif mad <= _ROW_GOOD_PM:
            data["tip_verdict"] = "stable"
            data["tip_message"] = (
                "逐行 MAD %.1f pm —— 针尖稳。这一帧的 RMS %.0f pm 若很大，"
                "那是**地形**，不是针尖。" % (mad, rms))
        elif mad <= _ROW_USABLE_PM:
            data["tip_verdict"] = "usable_coarse"
            data["tip_message"] = (
                "逐行 MAD %.1f pm —— 大视野形貌还能用，原子级的活做不了。" % mad)
        else:
            data["tip_verdict"] = "unstable"
            data["tip_message"] = (
                "逐行 MAD %.1f pm 超过当前工作流的高档阈值。请检查行跳动、针尖状态与地形等因素；本阈值不是公开快照提供的仪器标定。"
                 % mad)

        if n_big:
            data["big_jump_note"] = (
                "另有 %d/%d 行出现大跳变。**这既可能是针尖跳，也可能是扫过真实台阶** —— "
                "两者在逐行中位高度上是同一个形状，分不开。要区分就在同一位置重扫一遍："
                "针尖跳变不重复，台阶重复。" % (n_big, n_rows))
        return SkillResult(skill_name="AssessFrameTrust", success=True, data=data)
