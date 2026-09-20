# -*- coding: utf-8 -*-
"""多帧原子分辨：一致性判定与压电定标。

这一组和 ``atomic_lattice`` 的单帧版是**两种问题**，不是「多扫几张更准」：

* ``CalibratePiezoFromLattice``（单帧）解得出仿射矩阵，但其中的剪切**归属未定**
  —— 压电非正交与慢轴热漂移在一帧里产生完全相同的图像畸变。
* ``AssessAtomicResolution``（单帧）能排除白噪声与准周期抖动，但排除不掉
  「针尖恰好在这一帧里以某个空间频率抖」。

多帧能解决的是这两件单帧**在原理上**做不到的事：换扫描角时压电畸变跟着转、
热漂移不转；真晶格跨帧重现同一组格矢、抖动不重现。推导见
``mast.vision.lattice_multiframe``。
"""
from __future__ import annotations

import logging
from typing import Any

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

__all__ = ["AssessAtomicConsistency", "CalibratePiezoMultiAngle"]


def _split_paths(raw: str) -> list[str]:
    """逗号分隔的路径串 → 列表。Windows 路径里没有逗号，所以这样切是安全的。"""
    return [p.strip().strip('"') for p in str(raw or "").split(",") if p.strip()]


def _load_one(path: str, channel: str):
    """``(image, nm_per_px, angle_deg, line_time_s, width_nm, error)``。

    ``angle_deg`` **从文件头读**（``scan_angle``），不让调用方手填 —— 手填的角度
    与文件对不上时，多角度求解会得到一个残差很小的错解，而错在输入上，看代码
    是看不出来的。
    """
    from pathlib import Path

    if not Path(path).exists():
        return None, None, None, None, None, f"文件不存在: {path}"
    try:
        import numpy as np

        from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    except ImportError as exc:  # noqa: BLE001
        return None, None, None, None, None, f"缺依赖: {exc}"
    try:
        scan = read_sxm(path)
    except Exception as exc:  # noqa: BLE001
        return None, None, None, None, None, f".sxm 读取失败: {exc}"

    fr = sxm_oriented_frames(scan, channel)
    img = fr.get("forward")
    if img is None:
        return None, None, None, None, None, f"没有通道 {channel!r} 的正扫数据"
    img = np.asarray(img, dtype=np.float64)
    nmpp = fr.get("nm_per_px")
    if not nmpp:
        return None, None, None, None, None, "文件头里没有像素标度"

    header = scan.get("header") or {}
    ang = header.get("scan_angle")
    try:
        angle = float(str(ang).split()[0]) if ang is not None else None
    except Exception:  # noqa: BLE001
        angle = None
    if angle is None:
        return None, None, None, None, None, "文件头里没有 scan_angle"

    # scan_time 存的是「每行每方向」的时间，一行往返是它的两倍 —— 慢轴推进
    # 一行所花的正是往返时间，漂移换算要用后者。
    line_t = None
    try:
        parts = str(header.get("scan_time", "")).split()
        if parts:
            line_t = 2.0 * float(parts[0])
    except Exception:  # noqa: BLE001
        line_t = None

    width_nm = fr.get("width_nm")
    return (img, float(nmpp), angle, line_t,
            float(width_nm) if width_nm else None, "")


class AssessAtomicConsistency(BaseSkill):
    """多帧原子相一致性：那个晶格是不是**真的**。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessAtomicConsistency",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "跨帧判断原子晶格是否真实：真晶格重现同一组格矢，针尖抖动不重现。"
                "单帧的角向集中度排除不掉「这一帧恰好抖出了周期」。"
            ),
            parameters=[
                ParameterSpec(name="scan_paths", type="str", required=True,
                              description="多个 .sxm 路径，逗号分隔（至少两个）"),
                ParameterSpec(name="channel", type="str", required=False,
                              default="Z", description="形貌通道"),
            ],
            estimated_duration_s=6.0,
            tags=["atomic", "lattice", "multiframe", "analysis"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from mast.vision.lattice_multiframe import assess_atomic_consistency

        paths = _split_paths(params.get("scan_paths", ""))
        channel = str(params.get("channel") or "Z")
        if len(paths) < 2:
            return SkillResult(
                skill_name=self.metadata().name, success=False,
                error="至少要两帧。单帧请用 AssessAtomicResolution —— "
                      "它给的是「这一帧像不像晶格」，不是「这个晶格是不是真的」。")

        imgs, angles, bad, nmpp = [], [], [], None
        for p in paths:
            img, mpp, ang, _lt, _w, err = _load_one(p, channel)
            if err:
                bad.append({"path": p, "error": err})
                continue
            if nmpp is None:
                nmpp = mpp
            elif abs(mpp - nmpp) / max(nmpp, 1e-12) > 0.02:
                bad.append({"path": p,
                            "error": "像素标度与第一帧差 %.1f%%，不是同一种取图"
                                     % (abs(mpp - nmpp) / nmpp * 100)})
                continue
            imgs.append(img)
            angles.append(ang)

        if len(imgs) < 2:
            return SkillResult(
                skill_name=self.metadata().name, success=False,
                error="能用的帧不足两张", data={"rejected": bad})

        r = assess_atomic_consistency(imgs, nmpp, angles_deg=angles)
        return SkillResult(
            skill_name=self.metadata().name, success=True,
            data={
                "verdict": r.verdict,
                "n_frames": r.n_frames,
                "n_atomic": r.n_atomic,
                "n_unusable": r.n_unusable,
                "n_no_lattice": r.n_no_lattice,
                "period_spread": r.period_spread,
                "angle_spread_deg": r.angle_spread_deg,
                "reason": r.reason,
                "per_frame": r.per_frame,
                "warnings": list(r.warnings),
                "rejected": bad,
                "angles_deg": angles,
            })


class CalibratePiezoMultiAngle(BaseSkill):
    """由多个扫描角的原子分辨帧，把压电畸变与热漂移**分开**。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CalibratePiezoMultiAngle",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "多扫描角原子分辨帧 → 压电 x/y 尺度与非正交，且与热漂移分离。"
                "单帧版给不出这个分离（剪切归属未定），这是它存在的唯一理由。"
            ),
            parameters=[
                ParameterSpec(name="scan_paths", type="str", required=True,
                              description="多个 .sxm 路径，逗号分隔。扫描角从文件头读，"
                                          "不用填；角度要张开 ≥15°"),
                ParameterSpec(name="channel", type="str", required=False,
                              default="Z", description="形貌通道"),
                ParameterSpec(name="surface", type="str", required=False,
                              default="Au(111)",
                              description="表面（决定理论晶格常数）"),
            ],
            estimated_duration_s=10.0,
            tags=["piezo", "calibration", "lattice", "multiframe"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from mast.vision.lattice_multiframe import (
            calibrate_multi_angle,
            collect_observation,
        )

        name = self.metadata().name
        paths = _split_paths(params.get("scan_paths", ""))
        channel = str(params.get("channel") or "Z")
        surface = str(params.get("surface") or "Au(111)")
        if len(paths) < 2:
            return SkillResult(
                skill_name=name, success=False,
                error="至少要两个扫描角。单帧请用 CalibratePiezoFromLattice，"
                      "但它报出的剪切**归属未定**（压电还是漂移分不开）。")

        obs, bad, line_t, width_nm = [], [], 0.0, 0.0
        for p in paths:
            img, mpp, ang, lt, w, err = _load_one(p, channel)
            if err:
                bad.append({"path": p, "error": err})
                continue
            o = collect_observation(img, mpp, ang, label=p, line_time_s=lt or 0.0)
            if o is None:
                bad.append({"path": p, "error": "这一帧上量不到晶格"})
                continue
            obs.append(o)
            if lt:
                line_t = lt
            if w:
                width_nm = w

        if len(obs) < 2:
            return SkillResult(
                skill_name=name, success=False,
                error="能用的帧不足两张（量到晶格的才算）",
                data={"rejected": bad, "n_ok": len(obs)})

        r = calibrate_multi_angle(obs, surface,
                                  line_time_s=line_t, slow_axis_nm=width_nm)
        data: dict[str, Any] = {
            "ok": r.ok,
            "reason": r.reason,
            "n_frames": r.n_frames,
            "angles_deg": [o.angle_deg for o in obs],
            "angle_spread_deg": r.angle_spread_deg,
            "condition_number": r.condition_number,
            "residual_rel": r.residual_rel,
            "per_frame_residual": r.per_frame_residual,
            "warnings": list(r.warnings),
            "rejected": bad,
        }
        if r.ok:
            data.update({
                "x_scale": r.x_scale,
                "y_scale": r.y_scale,
                "piezo_shear_deg": r.shear_deg,
                "drift_shear_equiv_deg": r.drift_shear_deg,
                "drift_nm_per_s": r.drift_nm_per_s,
                "drift_direction_deg": r.drift_direction_deg,
                "detail": r.detail,
                "suggestion": (
                    "把 X / Y 压电灵敏度分别乘以 x_scale / y_scale。"
                    "**这个 skill 不写仪器** —— 改灵敏度要人来决定，"
                    "而且改之前请确认这批帧确实是同一块区域、同一根针尖。"),
            })
        # ok=False 也返回 success=True：解不出来是一个**结论**（角度不够张开、
        # 帧上没有晶格），不是这次调用失败。把它当失败会让上层重试，而重试
        # 拿同一批帧只会得到同一个结论。
        return SkillResult(skill_name=name, success=True, data=data)
