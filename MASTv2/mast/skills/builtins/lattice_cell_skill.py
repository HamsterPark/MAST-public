# -*- coding: utf-8 -*-
"""``MeasureLatticeCell`` —— **未知**表面的原胞：a₁、a₂、测出来的 γ。

判据本体在 :mod:`mast.vision.lattice_cell`。这一层只做 IO 与聚合。

## 与三个既有技能的分界

| 技能 | 它问什么 | 需要已知晶格常数吗 |
|---|---|---|
| ``AnalyseAtomicLattice`` | 三个方向的周期、六重对称、取向 | 要（``surface``，与理论值比对） |
| ``CalibratePiezoFromLattice`` | 扫描器把已知晶格扭曲了多少 | **要**（查 ``SURFACE_LATTICE_NM``） |
| ``CalibratePiezoMultiAngle`` | 多角度分离压电畸变与热漂移 | 要 |
| **本技能** | **这个表面的原胞是多少** | **不要** |

新体系上线时手里只有图：``SURFACE_LATTICE_NM`` 里没有它的表项，而「先测后定」
正是它进表的前提。测出来之后，上面三个才用得上。

两处技术差别写在 ``vision.lattice_cell`` 的模块注释里：γ 是测量值而不是被强制成
120°；峰位做亚像素精修（既有的取整数 bin，径向量化误差约 1/r）。

## 给多帧、并且分上下扫，理由不是「多了更准」

慢轴漂移会把 a₂ 沿慢轴拉伸或压缩，而上扫与下扫**反号**。给两个方向的帧，
取平均即抵消，差值本身就是漂移应变 —— 同一批数据多出一个漂移速率的读数。
只给一个方向也能出数，但那时 a₂ 里的漂移**留在结果里且看不出来**，
所以返回体里会明说这一条。

## 超结构检验默认开，因为它的陷阱在默认关的时候最危险

「在半序位置细搜取最大值」是有偏统计量，纯噪声上也给正数。所以它永远同时在
**对照波矢**上计算同一个统计量，报告相对对照的比值与判决。只有候选振幅
为正不代表存在超结构；必须检查它是否能与对照区分。
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

_NAME = "MeasureLatticeCell"

__all__ = ["MeasureLatticeCell"]


class MeasureLatticeCell(BaseSkill):
    """从一批原子分辨帧量出实空间原胞，并检验有没有超结构。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从一批原子分辨 .sxm 量出表面的实空间原胞 a₁ / a₂ / γ —— "
                "**不需要事先知道晶格常数**（那正是 CalibratePiezoFromLattice 要而"
                "这里不要的东西）。给上扫与下扫两个方向的帧时，慢轴漂移在 a₂ 上的"
                "应变会被抵消，并顺带给出漂移速率。同时检验半序位置有没有超结构，"
                "带空白对照。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_paths", type="str",
                    description=(
                        "一个或多个 .sxm 路径，逗号或换行分隔。"
                        "**上扫与下扫都给**才能消掉慢轴漂移（文件头的 SCAN_DIR "
                        "自动分组，不用手工标）。"),
                    required=True),
                ParameterSpec(
                    name="channel", type="str",
                    description="形貌通道名，默认 Z。",
                    required=False, default="Z"),
                ParameterSpec(
                    name="min_indexed", type="int",
                    description=(
                        "一帧要被计入平均，它选出的基矢至少要指标上这么多个观测峰。"
                        "默认 3 = 「除了它们自己，至少还有一个峰印证」。"
                        "设成 2 会放进「只解释了自己」的帧，那种帧上「a₂ 被报成一半」"
                        "查不出来。"),
                    required=False, default=3, min_value=2, max_value=8),
                ParameterSpec(
                    name="superstructure", type="bool",
                    description="是否做半序超结构检验（带空白对照）。默认做。",
                    required=False, default=True),
            ],
            estimated_duration_s=20.0,
            composition_level=1,
            tags=["lattice", "analysis", "read", "calibration"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import numpy as np

        from mast.skills.builtins._sxm_frame import load_frame, split_paths
        from mast.vision.lattice_cell import (
            combine_up_down,
            measure_cell,
            superstructure_test,
        )

        paths = split_paths(str(params.get("scan_paths") or ""))
        channel = str(params.get("channel") or "Z")
        min_indexed = int(params.get("min_indexed") or 3)
        do_super = bool(params.get("superstructure", True))
        if not paths:
            return SkillResult(skill_name=_NAME, success=False,
                               error="没有给 scan_paths")

        ups, downs, per_frame, rejected = [], [], [], []
        heights, times, best = [], [], None
        n_io_failed = 0
        for p in paths:
            fr = load_frame(p, channel)
            if fr.error:
                rejected.append({"path": p, "reason": fr.error})
                n_io_failed += 1
                continue
            cell = measure_cell(fr.forward, fr.nm_per_px, scan_dir=fr.scan_dir)
            if not cell.ok:
                rejected.append({"path": p, "reason": cell.reason})
                continue
            per_frame.append({
                "name": fr.name, "scan_dir": fr.scan_dir,
                "width_nm": fr.width_nm, "bias_v": fr.bias_v,
                "setpoint_a": fr.setpoint_a,
                "a1_nm": round(cell.a1_nm, 4), "a2_nm": round(cell.a2_nm, 4),
                "gamma_deg": round(cell.gamma_deg, 2),
                "a1_angle_deg": round(cell.a1_angle_deg, 2),
                "indexed": cell.indexed, "indexed_total": cell.indexed_total,
                "n_ridge": cell.n_ridge,
            })
            (ups if fr.scan_dir.startswith("up") else downs).append(cell)
            if fr.width_nm:
                heights.append(float(fr.width_nm))
            if fr.line_time_s and fr.forward is not None:
                times.append(float(fr.line_time_s) * 2.0 * fr.forward.shape[0])
            if best is None or cell.indexed > best[0].indexed:
                best = (cell, fr)

        if n_io_failed == len(paths):
            # 一个文件都没读进来 = **这件事没做成**，与「读进来了但量不出晶格」
            # 是两回事。后者是判不了（success=True + undetermined），
            # 前者是技能失败 —— 混在一起，调用方就分不清该去修路径还是该换帧。
            return SkillResult(
                skill_name=_NAME, success=False,
                error=("%d 个路径一个都读不进来：%s"
                       % (len(paths), "；".join(r["reason"] for r in rejected[:3]))),
                data={"rejected": rejected})
        if not per_frame:
            return SkillResult(
                skill_name=_NAME, success=True,
                data={"verdict": "undetermined", "rejected": rejected,
                      "n_frames": 0},
                summary=("这一批帧上一个原胞都量不出来 —— 逐帧原因见 rejected。"
                         "最常见的是像素太粗（scale_off）或谱被条纹主导。"))

        data: dict = {
            "n_frames": len(per_frame), "n_up": len(ups), "n_down": len(downs),
            "per_frame": per_frame, "rejected": rejected,
        }
        combined = combine_up_down(
            ups, downs, min_indexed=min_indexed,
            frame_height_nm=float(np.median(heights)) if heights else None,
            frame_time_s=float(np.median(times)) if times else None)
        data["combined"] = {k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in combined.items()}

        if combined.get("ok"):
            data["verdict"] = "measured"
            data["a1_nm"] = round(combined["a1_nm"], 4)
            data["a2_nm"] = round(combined["a2_nm"], 4)
            data["gamma_deg"] = (round(combined["gamma_deg"], 2)
                                 if combined.get("gamma_deg") else None)
        else:
            # 只有一个扫描方向：仍然出数，但 a₂ 里含漂移，必须说出来。
            usable = [c for c in (ups + downs) if int(c.indexed or 0) >= min_indexed]
            if not usable:
                data["verdict"] = "undetermined"
                data["reason"] = combined.get("reason", "")
            else:
                data["verdict"] = "measured_one_direction"
                data["a1_nm"] = round(float(np.median([c.a1_nm for c in usable])), 4)
                data["a2_nm"] = round(float(np.median([c.a2_nm for c in usable])), 4)
                data["gamma_deg"] = round(
                    float(np.median([c.gamma_deg for c in usable])), 2)
                data["one_direction_caveat"] = (
                    "这一批只有%s扫。慢轴热漂移会把 a₂ 沿慢轴拉伸或压缩，而这个偏差"
                    "**只有换慢轴方向才看得见** —— 现在它留在 a₂ 里，量级通常是零点几"
                    "个百分点。要消掉就补几帧反方向的（Nanonis 扫描方向 up ↔ down）。"
                    % ("上" if ups else "下"))

        if do_super and best is not None:
            cell, fr = best
            res = superstructure_test(fr.forward, fr.nm_per_px, cell)
            data["superstructure_frame"] = fr.name
            data["superstructure"] = [
                {"order": r.label, "period_nm": r.period_nm,
                 "amplitude_pm": round((r.amplitude or 0.0) * 1e12, 3),
                 "control_max_pm": round((r.control_max or 0.0) * 1e12, 3),
                 "ratio_to_control": (round(r.ratio_to_control, 2)
                                      if r.ratio_to_control else None),
                 "verdict": r.verdict, "note": r.note}
                for r in res]

        data["shear_caveat"] = (
            "γ 是测量值，但它含着**未校正的扫描器剪切** —— 单帧分不开「表面本来就不是"
            "直角」与「扫描器把直角扭了」。要分开就用 CalibratePiezoMultiAngle"
            "（换扫描角）或 calibrate_up_down（换慢轴方向），两者分离的是剪切角，"
            "与本技能用上下扫消掉的慢轴长度应变是同一个漂移矢量的两个分量。")
        return SkillResult(skill_name=_NAME, success=True, data=data,
                           summary=_summary(data))


def _summary(data: dict) -> str:
    v = data.get("verdict")
    if v == "undetermined":
        return ("量不出原胞（%d 帧全部无法定基矢）。"
                % data.get("n_frames", 0))
    head = ("原胞 a₁ = %.4f nm、a₂ = %.4f nm、γ = %s°"
            % (data.get("a1_nm", float("nan")), data.get("a2_nm", float("nan")),
               data.get("gamma_deg")))
    c = data.get("combined") or {}
    if v == "measured":
        head += ("（上扫 %d 帧 / 下扫 %d 帧取中位，慢轴漂移已抵消"
                 % (c.get("n_up", 0), c.get("n_down", 0)))
        if c.get("drift_nm_per_h") is not None:
            head += "，顺带测得漂移 %.2f nm/h" % c["drift_nm_per_h"]
        head += "）。"
    else:
        head += "（只有一个扫描方向，a₂ 里含未抵消的漂移）。"
    sup = data.get("superstructure") or []
    present = [s for s in sup if s.get("verdict") == "present"]
    if sup:
        head += (" 超结构：%s。"
                 % ("、".join("%s 处有（对照的 %.1f 倍）"
                              % (s["order"], s["ratio_to_control"] or 0)
                              for s in present)
                    if present else "半序位置与空白对照不可区分，没有"))
    return head


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(MeasureLatticeCell, context_provider)
