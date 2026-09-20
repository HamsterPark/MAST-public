"""AnalyzeFrameTilt：只读已保存的 .sxm，计算倾斜、台阶主导与表面起伏。

为 AutoTilt 提供 surface_rms_m，并输出可用于目标仪器阈值验证的统计量。
通用默认阈值不等于目标仪器标定。慢扫方向混入采集期间漂移，不能直接把帧的
视在倾斜当成真实静态倾斜；需要两方向信息时使用独立的 TiltProbeCircle 测量。"""

from __future__ import annotations

import logging
from pathlib import Path

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class AnalyzeFrameTilt(BaseSkill):
    """Measure tilt, step dominance and surface roughness from a saved .sxm."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AnalyzeFrameTilt",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "量出样品倾斜、台阶是不是主导了高度分布、以及表面自身的粗糙度 —— "
                "全都只从一份已保存的 .sxm 文件里读。只读,不碰硬件。"
                "用它来拿到 AutoTilt 需要的 surface_rms_m(也就是「斜坡有没有把形貌"
                "淹没」那条判据),以及在决定要不要调平之前先看看图是不是肉眼可见地"
                "倾斜。**注意**:它报出来的倾斜**只有沿快扫轴那一个方向可信**:"
                "慢扫轴横跨整帧的采集时长,那个方向上的热漂移与真实倾斜无法区分。"
                "要两个方向都可信的倾斜测量,用 TiltProbeCircle。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path",
                    type="str",
                    description="要分析的 .sxm 文件路径。",
                    required=True,
                ),
                ParameterSpec(
                    name="channel",
                    type="str",
                    description="形貌通道('Z' 是标准选择)。",
                    required=False,
                    default="Z",
                ),
                ParameterSpec(
                    name="check_steps",
                    type="bool",
                    description=(
                        "跑「台阶主导」这道否决闸。台阶主导的时候,倾斜拟合量到的"
                        "是台阶包络而不是表面,所以倾斜会被报成**无效**,而不是给出"
                        "一个数。只有在想看原始拟合结果时才关掉它。"
                    ),
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=3.0,
            composition_level=2,
            tags=["scan", "tilt", "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        scan_path = str(params["scan_path"])
        channel_name = params.get("channel") or "Z"
        check_steps = bool(params.get("check_steps", True))

        if not Path(scan_path).exists():
            return SkillResult(
                skill_name="AnalyzeFrameTilt", success=False,
                error=f"文件不存在: {scan_path}")

        try:
            import numpy as np

            from mast.io.mosaic import parse_xy_meta
            from mast.io.nanonis_files import read_sxm
            from mast.vision.tilt import (
                detrend_quadratic,
                estimate_tilt,
                noise_floor,
                step_dominance_multiscale,
                structure_dominance,
            )
        except ImportError as exc:
            return SkillResult(skill_name="AnalyzeFrameTilt", success=False,
                               error=f"缺依赖: {exc}")

        try:
            scan = read_sxm(scan_path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name="AnalyzeFrameTilt", success=False,
                               error=f".sxm 读取失败: {exc}")

        channels = scan.get("channels", {}) or {}
        ch = channels.get(channel_name) or next(iter(channels.values()), None)
        if ch is None:
            return SkillResult(
                skill_name="AnalyzeFrameTilt", success=False,
                error=f"文件里没有可用通道(要的是 {channel_name!r})")
        img = ch.get("forward")
        if img is None:
            img = ch.get("backward")
        if img is None:
            return SkillResult(skill_name="AnalyzeFrameTilt", success=False,
                               error="通道里没有正扫/反扫数据")

        arr = np.asarray(img, dtype=np.float64)

        meta = None
        try:
            meta = parse_xy_meta(scan.get("header", {}) or {})
        except Exception:  # noqa: BLE001
            meta = None
        if not meta:
            return SkillResult(
                skill_name="AnalyzeFrameTilt", success=False,
                error=("从 .sxm 头里解析不出扫描几何(offset/range)—— 没有物理尺寸"
                       "就换算不出角度,拒绝给出无意义的数字"),
                data={"scan_path": scan_path})

        width_m, height_m = float(meta["w"]), float(meta["h"])
        angle_deg = float(meta.get("angle") or 0.0)
        ny, nx = arr.shape
        nm_per_px = (width_m / nx) * 1e9 if nx else None

        est = estimate_tilt(arr, width_m=width_m, height_m=height_m,
                            scan_angle_deg=angle_deg, nm_per_px=nm_per_px,
                            check_steps=check_steps)

        # surface_rms_m 在扣除倾斜与曲率后描述整帧起伏，作为 AutoTilt 的表面幅度尺度。
        # 若不去趋势，倾斜会同时抬高分子与分母，使判据自我抵消。
        # 二阶面也可能吸收单个台阶的一部分，因此分母可能偏小，应结合图像解释。
        # local_texture_rms_m 是分块局部纹理尺度，回答不同问题，不能直接替代前者作为 AutoTilt 输入。
        flat = detrend_quadratic(arr)
        finite = flat[np.isfinite(flat)]
        surface_rms = float(np.std(finite)) if finite.size else 0.0
        try:
            _g, local_sigma, _r = (0.0, 0.0, 1.0)
            tiles = []
            h, w = flat.shape
            tile = 32
            for i in range(h // tile):
                for j in range(w // tile):
                    blk = flat[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
                    blk = blk[np.isfinite(blk)]
                    if blk.size >= 2:
                        tiles.append(float(blk.std()))
            local_sigma = float(np.median(tiles)) if tiles else surface_rms
        except Exception:  # noqa: BLE001
            local_sigma = surface_rms

        ratio_max, by_tile = step_dominance_multiscale(flat)

        data = {
            "scan_path": scan_path,
            "channel": channel_name,
            "geometry": {
                "width_m": width_m, "height_m": height_m,
                "angle_deg": angle_deg,
                "pixels": [int(nx), int(ny)],
                "nm_per_px": nm_per_px,
            },
            # —— 喂给 AutoTilt 的就是这一个 ——
            "surface_rms_m": surface_rms,
            "local_texture_rms_m": local_sigma,
            "noise_floor_m": noise_floor(arr),
            "step": {
                "ratio_multiscale": ratio_max,
                "ratio_by_tile": by_tile,
                "ratio_single_tile32": structure_dominance(flat, 32),
            },
            "tilt": est.as_dict(),
        }

        if est.valid:
            summary = (
                f"倾斜 {est.slope_mag_deg:.4f}°(快扫轴 {est.tilt_fast_deg:+.4f}°,"
                f"慢扫轴 {est.tilt_slow_deg:+.4f}° ← 混漂移不可信);"
                f"帧内吃掉 Z {est.z_span_m * 1e9:.2f} nm;"
                f"表面起伏 {surface_rms * 1e12:.1f} pm")
        else:
            summary = (
                f"倾斜无法测定({est.invalid_reason});"
                f"表面起伏 {surface_rms * 1e12:.1f} pm,"
                f"台阶主导比 {ratio_max:.2f}")

        return SkillResult(skill_name="AnalyzeFrameTilt", success=True,
                           data=data, summary=summary)
