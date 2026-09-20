# -*- coding: utf-8 -*-
"""``AssessScanTexture`` —— 一帧上「晶格有多强、条纹有多强、哪一块清楚」。

判据本体在 :mod:`mast.vision.frame_texture`（纯函数、零 IO）。这一层只做 IO
与阈值取用，一个判据逻辑都不写 —— 与 ``AssessFrameCorrugation`` /
``AssessHerringbone`` 同一个分工。

## 它与既有判据的分界（三条，都不重叠）

* ``AssessAtomicResolution`` / ``AssessAtomicPhase`` 回答**有没有**原子相（三态）。
  本技能不回答这个，它假设你已经知道有，来问**有多强、在哪儿**。
* ``AssessFrameCorrugation`` 量的是**整帧**起伏（``judge_frame`` 的
  ``corrugation_rms_m``，全仓单一真源），不分空间频率 —— 一道台阶就能把它抬起来。
  本技能只量晶格那一个频带。
* ``AssessFrameTrust`` 的逐行跳动 MAD 与本技能的条纹幅值是**同一个现象的两个口径**：
  前者是稳健统计量（抗台阶）、无量纲于频率；后者是带限幅值，与晶格幅值**同单位可比**。
  要判「针尖稳不稳」用前者；要回答「为什么这帧看着脏」用后者，因为那句话的完整形式是
  「晶格 4.7 pm，而条纹 5.6 pm」。

## ``good_ratio`` 没有默认值，这一条是硬的

块级比值的阈值目前只在**一个样品、一根针尖、一夜**上标定过，好帧与条纹区的
取值范围都很窄，不能代表其它样品/针尖状态。给它一个 ``ParameterSpec`` 默认值，
pydantic 会在调用方没传时替它填上，于是「这台机器上没标定过」这件事再也看不见
—— 本仓已为这个形状付过好几次学费。

不传 ``good_ratio`` 时**照样出数**：``tile_grid`` 与 ``tile_median_ratio`` 是测量，
不需要阈值；只有 ``tile_good_fraction`` 留 ``None`` 并在 ``notes`` 里说明为什么。
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

_NAME = "AssessScanTexture"

__all__ = ["AssessScanTexture"]


class AssessScanTexture(BaseSkill):
    """量一帧的晶格幅值、条纹幅值与逐块晶格可辨度。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "量一帧 .sxm 上晶格起伏有多强（逐方向，皮米）、逐行条纹噪声有多强"
                "（同单位，可直接与晶格比），以及把帧切成小块后有多少块的晶格"
                "压得住其余结构。回答的是「有多好、哪一块好」，"
                "**不回答「有没有原子分辨」** —— 那是 AssessAtomicResolution 的事。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path", type="str",
                    description="要分析的 .sxm 文件路径。",
                    required=True),
                ParameterSpec(
                    name="channel", type="str",
                    description="形貌通道名，默认 Z。",
                    required=False, default="Z"),
                ParameterSpec(
                    name="tile_nm", type="float",
                    description=(
                        "分块的边长，纳米。默认 4 nm —— 它不是阈值而是几何选择："
                        "块里至少要装下 8 个晶格周期，装不下就整个不给块图。"),
                    unit="nm", required=False, default=4.0,
                    min_value=0.5, max_value=100.0),
                ParameterSpec(
                    name="good_ratio", type="float",
                    description=(
                        "一块算「晶格清楚」的比值门槛（晶格带幅值 / 同块其余结构幅值）。"
                        "**刻意没有默认值**：0.6 只在真机的 WO₂I₂ 上标过一夜"
                        "（好帧 0.77-1.41、条纹区 0.10-0.35），换体系必须重标。"
                        "不传时仍给出逐块比值与中位数，只是不给「好块占比」。"),
                    required=False,
                    min_value=0.0, max_value=10.0),
            ],
            estimated_duration_s=3.0,
            composition_level=1,
            tags=["scan", "quality", "lattice", "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import numpy as np

        from mast.skills.builtins._sxm_frame import load_frame
        from mast.vision.frame_texture import (
            lattice_amplitude_pm,
            streak_amplitude_pm,
            tile_lattice_map,
        )
        from mast.vision.lattice_cell import measure_cell

        path = str(params.get("scan_path") or "")
        channel = str(params.get("channel") or "Z")
        tile_nm = float(params.get("tile_nm") or 4.0)
        good_ratio = params.get("good_ratio")

        fr = load_frame(path, channel)
        if fr.error:
            return SkillResult(skill_name=_NAME, success=False, error=fr.error)

        notes: list[str] = []
        data: dict = {
            "scan_path": path, "channel": channel,
            "nm_per_px": fr.nm_per_px, "width_nm": fr.width_nm,
            "bias_v": fr.bias_v, "setpoint_a": fr.setpoint_a,
            "scan_dir": fr.scan_dir, "rec_time": fr.rec_time,
        }

        # 晶格方向从整帧上量一次；块内不重新找峰（一块里只有几个周期，
        # 找峰会锁到噪声上）。
        cell = measure_cell(fr.forward, fr.nm_per_px, scan_dir=fr.scan_dir)
        if not cell.ok:
            data["verdict"] = "undetermined"
            data["reason"] = cell.reason
            data["streak_pm"] = streak_amplitude_pm(fr.forward, fr.nm_per_px)
            notes.append(
                "找不到可用的晶格方向（%s）—— 晶格幅值与块图都无从谈起，因为它们"
                "都是「沿着那个方向」的量。条纹幅值仍然给了：它不需要知道晶格在哪。"
                % cell.reason)
            data["notes"] = notes
            return SkillResult(skill_name=_NAME, success=True, data=data,
                               summary=_summary(data))

        data["verdict"] = "measured"
        dirs = [(cell.a1_angle_deg, cell.a1_nm),
                (cell.a1_angle_deg + cell.gamma_deg, cell.a2_nm)]
        amps = lattice_amplitude_pm(fr.forward, fr.nm_per_px, dirs)
        data["lattice_directions"] = [
            {"angle_deg": round(a.angle_deg, 2), "period_nm": round(a.period_nm, 4),
             "bandpass_pm": round(a.amplitude_pm, 2),
             "coherent_pm": round(a.coherent_pm, 2),
             "coherence": round(a.coherence, 3)}
            for a in amps]
        if amps:
            data["lattice_bandpass_pm"] = round(
                float(max(a.amplitude_pm for a in amps)), 2)
            data["lattice_coherent_pm"] = round(
                float(max(a.coherent_pm for a in amps)), 2)
        data["streak_pm"] = streak_amplitude_pm(fr.forward, fr.nm_per_px)
        # 条纹的绝对值随整帧起伏一起涨，单独看会把「地形丰富」读成「脏」。
        # 有意义的是与**同口径**（都是带通峰峰值）的晶格幅值之比。
        if data.get("lattice_bandpass_pm") and data.get("streak_pm"):
            data["streak_to_lattice"] = round(
                float(data["streak_pm"]) / float(data["lattice_bandpass_pm"]), 3)

        tiles = tile_lattice_map(
            fr.forward, fr.nm_per_px, [(a.angle_deg, a.period_nm) for a in amps],
            tile_nm=tile_nm,
            good_ratio=float(good_ratio) if good_ratio is not None else 0.6)
        data["tile_ok"] = bool(tiles.ok)
        data["tile_reason"] = tiles.reason
        data["tile_nm"] = tiles.tile_nm
        data["tile_periods_per_tile"] = (round(tiles.periods_per_tile, 2)
                                         if tiles.periods_per_tile else None)
        if tiles.ok:
            data["tile_grid"] = [[round(v, 3) for v in row] for row in tiles.grid]
            data["tile_median_ratio"] = round(tiles.median_ratio, 3)
            # 主报数：局域原子起伏。逐块算再取中位，与视野无关 ——
            # 整帧的相干分量会因帧内失相而随视野塌掉（30 nm 上只剩 1/30）。
            data["lattice_local_pm"] = round(tiles.coherent_median_pm, 2)
            if good_ratio is None:
                data["tile_good_fraction"] = None
                notes.append(
                    "没传 good_ratio，所以不给「好块占比」—— 那个数需要一个本机标定过的"
                    "门槛，而随包发布的 0.6 只在一个样品上标过。逐块比值与中位数"
                    "（%.2f）是测量，不需要门槛，已经给了。" % tiles.median_ratio)
            else:
                data["tile_good_fraction"] = round(tiles.good_fraction, 3)
                data["tile_good_ratio_used"] = float(good_ratio)
            # 几何上下 → 扫描先后：``up`` 的第一行是帧**底**。
            top, bot = tiles.top_band_median, tiles.bottom_band_median
            first, last = ((bot, top) if fr.scan_dir.startswith("up") else (top, bot))
            data["tile_first_scanned_median"] = round(first, 3)
            data["tile_last_scanned_median"] = round(last, 3)
            if first > 0 and (last / first < 0.5 or first / last < 0.5):
                notes.append(
                    "先扫的那一排块中位 %.2f、后扫的 %.2f —— 针尖在这一帧里变了。"
                    "**这一帧的读数是两段不同状态的平均**，拿它代表「现在的针尖」"
                    "会指向过去式（与 assess_atomic_phase 的半帧检查同一件事）。"
                    % (first, last))
        else:
            notes.extend(tiles.warnings)
        data["notes"] = notes
        data["cell"] = {"a1_nm": round(cell.a1_nm, 4), "a2_nm": round(cell.a2_nm, 4),
                        "gamma_deg": round(cell.gamma_deg, 2),
                        "indexed": cell.indexed, "indexed_total": cell.indexed_total}
        return SkillResult(skill_name=_NAME, success=True, data=data,
                           summary=_summary(data))


def _summary(data: dict) -> str:
    if data.get("verdict") == "undetermined":
        return "这一帧上找不到晶格方向，量不了晶格幅值与块图。"
    lat = data.get("lattice_local_pm")
    bp = data.get("lattice_bandpass_pm")
    ratio = data.get("streak_to_lattice")
    bits = []
    if lat is not None:
        bits.append("局域原子起伏 %.1f pm（逐块相干，与视野无关）" % lat)
    elif data.get("lattice_coherent_pm") is not None and bp is not None:
        bits.append("整帧相干起伏 %.1f pm（帧内失相会压低它；带通上界 %.1f pm）"
                    % (data["lattice_coherent_pm"], bp))
    if ratio is not None:
        bits.append("条纹/晶格 %.2f（同为带通口径）" % ratio)
    if data.get("tile_good_fraction") is not None:
        bits.append("%.0f%% 的块晶格清楚" % (100 * data["tile_good_fraction"]))
    elif data.get("tile_median_ratio") is not None:
        bits.append("块比值中位 %.2f（未给门槛，不算占比）" % data["tile_median_ratio"])
    s = "；".join(bits) if bits else "量不出可报的量"
    if ratio is not None and ratio > 1.5:
        s += "。条纹带的功率超过晶格带 —— 这帧「脏」的来源是逐行短划，不是晶格弱。"
    return s


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(AssessScanTexture, context_provider)
