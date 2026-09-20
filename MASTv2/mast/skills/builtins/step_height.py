"""MeasureStepHeight —— 从一张带台阶的 .sxm 量单原子台阶高度（只读）。

**为什么要单独有这个技能**：

调平那条链路里已经有一个够用的稳健平面拟合 —— ``mast.vision.tilt.fit_plane_robust``，
它的内点阈**随噪声底自适应**（3 × noise_floor）。台阶上的点因此成为外点，拟合只落在
单个台面上，减掉之后**台阶被完整保留**。但它一直只被 ``AnalyzeFrameTilt`` 内部使用，
而那个技能只报告倾斜角度、**不交出去斜后的数据**。

于是 agent 手上唯一能调的去斜入口是 ``SubtractPlane_RANSAC``，它用的是
``mast.skills.paper.data_processing.ransac_plane_subtract`` —— 那里的
``residual_threshold`` **写死 1e-10（100 pm）**。``fit_plane_robust`` 的注释早就点名
说过这件事：「在原子级平整的表面上 100 pm 比整个高度起伏还大，于是几乎所有点都算
内点，RANSAC 退化成普通最小二乘，稳健性白给」。

2026-08-26 合成对照（注入 3 台面 × 235.455 pm，倾斜 0.8 pm/px）：

    噪声      SubtractPlane_RANSAC          fit_plane_robust
             inlier  峰间距                 inlier  峰间距
     2 pm    0.795   161.8 pm  (−31%)       0.581   235.5 pm  (−0.02%)
     4 pm    0.794   164.9 pm  (−30%)       0.581   235.4 pm  (−0.02%)
     8 pm    0.790   145.8 pm  (−38%)       0.581   234.7 pm  (−0.3%)
    16 pm    0.786   115.0 pm  (−51%)       0.581   233.9 pm  (−0.7%)

**能调到的那个把台阶低估三到五成，而且噪声越大错得越离谱。** 台阶高度是 Z 压电标定
的基准（Au(111) d111 = a₀/√3 = 235.455 pm），用错的那个去标定，Z 会被标歪同样的幅度。

**它不做理论值归一化。** 报出来的就是测到的峰间距。拿 d111 去算「层数」再除以层数，
任何数都会变回 d111 —— 那是自我实现，不是测量（2026-08-26 的一版就这么错过，正反扫
差 0.1 pm「好得可疑」正是这么来的）。要不要跟某个理论值比，是调用方的事。

只读文件、不碰硬件。
"""

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

#: 直方图找台面用的默认设置。min_sep 取得比半个台阶大 —— 再小就会把同一个台面的
#: 肩部当成第二个台面（2026-08-26 踩过：一版 min_sep=80 pm 报出一堆 0.65×d111）。
_BINS = 220
_REL_HEIGHT = 0.10
_MIN_SEP_PM = 110.0


def _levels(z_pm, bins: int = _BINS, rel: float = _REL_HEIGHT,
            min_sep: float = _MIN_SEP_PM):
    """高度直方图上的台面位置 [(高度 pm, 峰高)…]，按高度排序。"""
    import numpy as np

    v = z_pm[np.isfinite(z_pm)]
    if v.size < 500:
        return []
    lo, hi = np.percentile(v, [0.2, 99.8])
    if not (hi > lo):
        return []
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centre = 0.5 * (edges[1:] + edges[:-1])
    smooth = np.convolve(hist.astype(float), np.ones(3) / 3.0, mode="same")
    peaks: list[tuple[float, float]] = []
    top = float(smooth.max()) if smooth.size else 0.0
    for i in range(1, len(smooth) - 1):
        if not (smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]):
            continue
        if smooth[i] <= top * rel:
            continue
        near = [k for k, p in enumerate(peaks) if abs(centre[i] - p[0]) < min_sep]
        if not near:
            peaks.append((float(centre[i]), float(smooth[i])))
        else:
            k = near[0]
            if smooth[i] > peaks[k][1]:
                peaks[k] = (float(centre[i]), float(smooth[i]))
    return sorted(peaks)


class MeasureStepHeight(BaseSkill):
    """Measure single-atom step height from a saved .sxm (robust plane fit)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MeasureStepHeight",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从一张带台阶的 .sxm 量**单原子台阶高度** —— 用内点阈随噪声底自适应的"
                "稳健平面拟合调平,台阶因此被保留而不是被当成斜坡减掉。这是 **Z 压电"
                "标定的基准**(Au(111) d111 = 235.455 pm)。"
                "\n\n**别用 SubtractPlane_RANSAC 去斜再自己量**:那个技能的内点阈写死"
                "100 pm,在原子级平整的表面上几乎所有点都算内点,RANSAC 退化成普通最小"
                "二乘。2026-08-26 合成对照:它把台阶低估 30–51%,噪声越大错得越离谱。"
                "\n\n**它不做理论值归一化** —— 报出来的就是测到的峰间距。要不要跟 d111"
                "比是调用方的事。正扫与反扫**分开报**,两者之差是 Z 反馈滞后的直接读数"
                "(扫得越快差越大),想要标定就该把它压到接近 0 再取值。"
                "\n\n只读文件、不碰硬件。"
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
                    name="sigma_pm",
                    type="float",
                    description=(
                        "覆盖自适应内点阈所用的噪声尺度(pm)。**留空是默认路径** —— "
                        "由 noise_floor 自己量。只有当表面自身的精细结构(如 Au(111) 的"
                        "herringbone,~20 pm)让内点率掉到 0.2 以下时才手动给一个,"
                        "取值应在噪声底与台阶高度之间。"
                    ),
                    required=False,
                    default=None,
                    min_value=0.01,
                    max_value=10000.0,
                ),
                ParameterSpec(
                    name="min_gap_pm",
                    type="float",
                    description=(
                        "只报告间距 ≥ 这个值的台面对(pm)。用来滤掉同一台面的肩部,"
                        "**不是**用来把答案往某个理论值上靠。"
                    ),
                    required=False,
                    default=120.0,
                    min_value=1.0,
                    max_value=100000.0,
                ),
                ParameterSpec(
                    name="max_gap_pm",
                    type="float",
                    description=(
                        "只报告间距 ≤ 这个值的台面对(pm),用来滤掉跨多层的大跳变。"
                        "默认 400 pm 覆盖到常见金属的单层台阶(Au 235 / Cu 209 / Ag 236 / "
                        "Si(111) 314)。要量多层就调大它。"
                    ),
                    required=False,
                    default=400.0,
                    min_value=1.0,
                    max_value=100000.0,
                ),
            ],
            estimated_duration_s=3.0,
            composition_level=2,
            tags=["scan", "step", "height", "calibration", "z", "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        scan_path = str(params["scan_path"])
        channel_name = params.get("channel") or "Z"
        sigma_pm = params.get("sigma_pm")
        min_gap = float(params.get("min_gap_pm") or 120.0)
        max_gap = float(params.get("max_gap_pm") or 400.0)

        if not Path(scan_path).exists():
            return SkillResult(skill_name="MeasureStepHeight", success=False,
                               error=f"文件不存在: {scan_path}")

        try:
            import numpy as np

            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
            from mast.vision.tilt import fit_plane_robust
        except ImportError as exc:
            return SkillResult(skill_name="MeasureStepHeight", success=False,
                               error=f"缺依赖: {exc}")

        try:
            scan = read_sxm(scan_path)
            frames = sxm_oriented_frames(scan, channel=channel_name)
        except Exception as exc:  # noqa: BLE001 — 文件坏了要说清是哪一步
            return SkillResult(skill_name="MeasureStepHeight", success=False,
                               error=f"读取失败: {exc}")

        if frames.get("forward") is None:
            return SkillResult(skill_name="MeasureStepHeight", success=False,
                               error=f"通道 {channel_name!r} 在这个文件里没有数据")

        header = scan.get("header") or {}
        try:
            acq = float(str(header.get("acq_time", "0")).split()[0] or 0)
        except (TypeError, ValueError):
            acq = 0.0

        sigma_m = float(sigma_pm) * 1e-12 if sigma_pm else None
        out: dict = {
            "scan_path": scan_path,
            "channel": channel_name,
            "width_nm": frames.get("width_nm"),
            "nm_per_px": frames.get("nm_per_px"),
        }

        arr0 = np.asarray(frames["forward"], dtype=float)
        lines = int(arr0.shape[0]) if arr0.ndim == 2 else 0
        out["line_time_s"] = (acq / (2 * lines)) if (acq > 0 and lines) else None

        per_dir: dict[str, dict] = {}
        for direction in ("forward", "backward"):
            block = frames.get(direction)
            if block is None:
                continue
            arr = np.asarray(block, dtype=float)
            fit = fit_plane_robust(arr, sigma=sigma_m)
            if fit is None:
                per_dir[direction] = {"ok": False, "why": "平面拟合失败"}
                continue
            a, b, c, inlier = fit
            ny, nx = arr.shape
            gy, gx = np.mgrid[:ny, :nx]
            flat_pm = (arr - (a * gx + b * gy + c)) * 1e12
            peaks = _levels(flat_pm)
            gaps = [peaks[i + 1][0] - peaks[i][0] for i in range(len(peaks) - 1)]
            kept = [g for g in gaps if min_gap <= g <= max_gap]
            entry = {
                "ok": bool(kept),
                "inlier_ratio": round(float(inlier), 4),
                "n_levels": len(peaks),
                "levels_pm": [round(p[0], 2) for p in peaks],
                "gaps_pm": [round(g, 2) for g in gaps],
                "kept_gaps_pm": [round(g, 2) for g in kept],
                "step_pm": round(float(np.median(kept)), 2) if kept else None,
            }
            if not kept:
                entry["why"] = (
                    "没有落在 [%.0f, %.0f] pm 里的台面对" % (min_gap, max_gap)
                    if peaks else "找不到台面(高度分布是单峰)")
            per_dir[direction] = entry
        out["per_direction"] = per_dir

        fwd = per_dir.get("forward", {}).get("step_pm")
        bwd = per_dir.get("backward", {}).get("step_pm")
        if fwd is not None and bwd is not None:
            out["step_pm"] = round((fwd + bwd) / 2.0, 2)
            out["fwd_bwd_diff_pm"] = round(fwd - bwd, 2)
            out["hysteresis_note"] = (
                "正反扫之差是 Z 反馈滞后的直接读数;扫得越快差越大。"
                "拿来做 Z 标定之前应当把它压到接近 0(降低每线速度)。")
        elif fwd is not None or bwd is not None:
            out["step_pm"] = fwd if fwd is not None else bwd
            out["fwd_bwd_diff_pm"] = None
            out["hysteresis_note"] = "只有一个扫描方向可用,量不出反馈滞后。"
        else:
            out["step_pm"] = None
            out["fwd_bwd_diff_pm"] = None

        low_inlier = [d for d, v in per_dir.items()
                      if v.get("inlier_ratio") is not None and v["inlier_ratio"] < 0.2]
        if low_inlier:
            out["inlier_warning"] = (
                "内点率偏低(%s) —— 表面自身的精细结构比噪声底大得多(Au(111) 的 "
                "herringbone 约 20 pm),自适应阈值把台面上的点也判成了外点。"
                "给一个 sigma_pm(噪声底与台阶高度之间)再跑一次。"
                % ", ".join("%s %.3f" % (d, per_dir[d]["inlier_ratio"]) for d in low_inlier))

        if out["step_pm"] is None:
            summary = "量不出台阶高度:" + "; ".join(
                "%s %s" % (d, v.get("why") or "-") for d, v in per_dir.items())
            return SkillResult(skill_name="MeasureStepHeight", success=True,
                               data=out, summary=summary)

        bits = ["台阶 %.2f pm" % out["step_pm"]]
        if out.get("fwd_bwd_diff_pm") is not None:
            bits.append("正反扫差 %+.2f pm" % out["fwd_bwd_diff_pm"])
        if out["line_time_s"]:
            bits.append("每线 %.2f s" % out["line_time_s"])
        bits.append("内点率 %s" % "/".join(
            "%.3f" % v["inlier_ratio"] for v in per_dir.values()
            if v.get("inlier_ratio") is not None))
        if out.get("inlier_warning"):
            bits.append("⚠ 内点率低,结果存疑")
        return SkillResult(skill_name="MeasureStepHeight", success=True,
                           data=out, summary="; ".join(bits))
