"""``LocateStepEdge`` —— 一帧形貌里最主要的那道台阶边在哪、朝哪。

``MeasureStepHeight`` 走 Z 直方图,回答的是「这道台阶多高」;本技能回答的是另一个问题:
**这道台阶在哪**。要沿台阶法向布一排谱(表面态驻波、边缘态、台阶处的势垒),先得有这条线。

判据本体在 :func:`mast.vision.step_edge.locate_step_edge`(纯函数、零 IO)。

坐标与角度:``edge_x_m`` / ``edge_y_m`` 是**扫描坐标**(米),直接可以喂给
``SpectroscopyAtPositions`` 或 ``FitDispersion``。``edge_angle_deg`` 是**图像坐标**里的方向
(0° = 快扫轴),与 ``AssessHerringbone`` 的 ``stripe_angle_deg`` 同一套;
``edge_angle_scan_deg`` 是同一条边在扫描坐标里的方向。两个都报:只报一个的那一次,用错的人
不会发现,而一排落在错方向上的谱,每一条都会「成功」。
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

_NAME = "LocateStepEdge"


class LocateStepEdge(BaseSkill):
    """找出一帧里最主要的台阶边,给出边上一点的坐标与边的方向。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在一张 .sxm 的形貌通道里找出最主要的那道台阶边,报出边上一点的扫描坐标"
                "与边的方向。只读文件,不碰硬件。verdict 取 'step_edge' / 'no_step' / "
                "'undecidable'。一帧里分不出两个够大的台面、或者边不直(弯边、几道台阶叠在"
                "一起),就回 undecidable —— 编出来的一条边会让整排谱落在错地方,而每条谱"
                "都会「成功」。角度同时给图像坐标与扫描坐标两个版本。"
            ),
            estimated_duration_s=3.0,
            composition_level=2,
            tags=["analysis", "step", "terrace", "edge", "geometry", "scan", "read"],
            parameters=[
                ParameterSpec(name="scan_path", type="str",
                              description=".sxm 扫描文件路径。", required=True),
                ParameterSpec(name="channel", type="str",
                              description="用哪一路通道,形貌一般是 Z。",
                              required=False, default="Z"),
                ParameterSpec(name="direction", type="str",
                              description="用正扫还是反扫的那一幅。",
                              required=False, default="forward",
                              allowed_values=["forward", "backward"]),
                # 刻意没有 default:缺席就用判据自己的值,写死在这里等于把它抄了两遍。
                ParameterSpec(name="edge_sigma", type="float",
                              description=(
                                  "边候选的门限:比梯度中位数高出这么多个稳健 σ(缺省 6)。"
                                  "调小能找到更弱的边,代价是噪声更容易冒充台阶。"),
                              required=False, min_value=1.0, max_value=50.0),
                ParameterSpec(name="max_curvature", type="float",
                              description=(
                                  "边界像素离拟合直线的 RMS 距离上限,以帧短边为单位"
                                  "(0.06 = 6%)。超过就判 undecidable。"),
                              required=False, min_value=0.005, max_value=0.5),
            ],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import math
        from pathlib import Path

        path = str(params.get("scan_path") or "")
        if not path or not Path(path).exists():
            return SkillResult(skill_name=_NAME, success=False, error=f"文件不存在: {path}")
        try:
            import numpy as np

            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
            from mast.vision.step_edge import locate_step_edge
        except ImportError as exc:
            return SkillResult(skill_name=_NAME, success=False, error=f"缺依赖: {exc}")
        try:
            scan = read_sxm(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=_NAME, success=False, error=f".sxm 读取失败: {exc}")

        channel = str(params.get("channel") or "Z")
        fr = sxm_oriented_frames(scan, channel)
        arr = fr.get(str(params.get("direction") or "forward"))
        if arr is None:
            return SkillResult(skill_name=_NAME, success=False,
                               error=f"文件里没有可用的 {channel!r} 通道数据")
        arr = np.asarray(arr, dtype=float)

        kw = {}
        if params.get("edge_sigma") is not None:
            kw["edge_sigma"] = float(params["edge_sigma"])
        if params.get("max_curvature") is not None:
            kw["max_straightness"] = float(params["max_curvature"])
        res = locate_step_edge(arr, **kw)

        nm_per_px = fr.get("nm_per_px")
        header = scan.get("header") or {}
        off = header.get("scan_offset") or header.get("SCAN_OFFSET")
        if isinstance(off, str):
            off = off.split()
        try:
            cx_m, cy_m = float(off[0]), float(off[1])
        except (TypeError, ValueError, IndexError):
            cx_m = cy_m = 0.0

        data = {"verdict": res.verdict, "step_height_pm": (None if res.step_height_m is None
                                                           else res.step_height_m * 1e12),
                "edge_angle_deg": res.angle_deg, "edge_angle_scan_deg": res.angle_scan_deg,
                "straightness": res.straightness, "n_edge_px": res.n_edge_px,
                "upper_fraction": res.upper_fraction, "nm_per_px": nm_per_px,
                "reasons": list(res.reasons), "warnings": list(res.warnings),
                "scan_path": path, "channel": channel}
        if res.x_px is not None and nm_per_px:
            ny, nx = arr.shape
            # image pixel -> scan coordinates: column grows with +x, and row 0 is the high-y
            # edge of the window, so the row index runs along -y
            data["edge_x_m"] = cx_m + (float(res.x_px) - (nx - 1) / 2.0) * float(nm_per_px) * 1e-9
            data["edge_y_m"] = cy_m - (float(res.y_px) - (ny - 1) / 2.0) * float(nm_per_px) * 1e-9
        if res.verdict == "step_edge":
            summary = (f"台阶边过 ({data.get('edge_x_m', 0) * 1e9:.1f}, "
                       f"{data.get('edge_y_m', 0) * 1e9:.1f}) nm,扫描系方向 "
                       f"{res.angle_scan_deg:.1f}°,高 {res.step_height_m * 1e12:.0f} pm")
        elif res.verdict == "no_step":
            summary = "这一帧里没有两个够大的台面 —— 没有台阶边可报"
        else:
            summary = f"判不了:{'、'.join(res.reasons) or '未知'}"
        return SkillResult(skill_name=_NAME, success=True, data=data, summary=summary)


__all__ = ["LocateStepEdge"]
