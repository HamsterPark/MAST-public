"""``VerifyAdatomAt`` —— 目标位置上到底有没有一个原子。

横向操纵之后要回答的问题只有一个:原子到了没有。这一条读一帧复扫图,用 ``ExtractClusters``
找出所有团簇,再看离目标最近的那个有多远。

它是独立的分析技能,而不是折进 ``MoveAtomTo``:复合技能在只给原语的工具面里会被过滤掉,
而「移完再验」这一步在那种模式下也得做得到。

判定是四态加一个「判不了」:``at_target`` / ``ambiguous``(两个候选一样近,说不清是哪个)
/ ``displaced``(找到了,但不在位置上)/ ``not_found`` / ``undecidable``(目标压根不在这
一帧里)。``tolerance_m`` 刻意没有 default:缺席时要能回落到衬底的最近邻距离一半 ——
那才是「同一个晶格位」的意思。
"""

from __future__ import annotations

import logging
import math

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

_NAME = "VerifyAdatomAt"


class VerifyAdatomAt(BaseSkill):
    """复扫图上,目标位置有没有一个吸附原子。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读一帧已保存的 .sxm,判断给定目标位置上有没有一个吸附原子,并给出最近团簇"
                "到目标的残差。只读文件,不碰硬件。verdict 取 'at_target' / 'ambiguous' / "
                "'displaced' / 'not_found' / 'undecidable'。目标不在这一帧的范围内时是"
                "'undecidable' —— 那是**这一帧答不了**,不是「原子不在」。容差留空则取衬底"
                "最近邻距离的一半,也就是「同一个晶格位」。"
            ),
            parameters=[
                ParameterSpec(name="scan_path", type="str",
                              description="复扫帧的 .sxm 路径。", required=True),
                ParameterSpec(name="target_x_m", type="float", unit="m",
                              description="目标位置 x,例如 '12.5n'(SI 前缀必须写)。",
                              required=True, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(name="target_y_m", type="float", unit="m",
                              description="目标位置 y,例如 '-3n'。",
                              required=True, min_value=-1.5e-6, max_value=1.5e-6),
                # 刻意没有 default:缺席要能回落到衬底晶格。
                ParameterSpec(name="tolerance_m", type="float", unit="m",
                              description=("算「到位」的半径,例如 '150p'。留空则取衬底最近邻"
                                           "距离的一半。"),
                              required=False, min_value=1e-11, max_value=5e-9),
                ParameterSpec(name="channel", type="str",
                              description="形貌通道。", required=False, default="Z"),
                ParameterSpec(name="polarity", type="str",
                              description="原子是亮的还是暗的。", required=False,
                              default="bright", allowed_values=["auto", "bright", "dark"]),
                ParameterSpec(name="min_peak_height_m", type="float", unit="m",
                              description=("比这更矮的团簇不算原子,例如 '30p'。"
                                           "留空则不按高度筛。"),
                              required=False, min_value=1e-12, max_value=5e-9),
                ParameterSpec(name="expected_count", type="int",
                              description="这一帧里预期有几个原子(对不上只给 warning)。",
                              required=False, min_value=0, max_value=1000),
            ],
            estimated_duration_s=5.0,
            composition_level=2,
            tags=["analysis", "cluster", "atom", "manipulation", "scan", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        path = str(params.get("scan_path") or "")
        if not path or not Path(path).exists():
            return SkillResult(skill_name=_NAME, success=False, error=f"文件不存在: {path}")
        tol, tol_src = self._tolerance(params)
        if tol is None:
            return SkillResult(
                skill_name=_NAME, success=True,
                data={"verdict": "undecidable", "reasons": ["no_tolerance"],
                      "scan_path": path},
                summary="没有容差可用:传 tolerance_m,或者先声明衬底")
        try:
            from mast.skills.builtins.cluster_extract import ExtractClusters
        except ImportError as exc:
            return SkillResult(skill_name=_NAME, success=False, error=f"缺依赖: {exc}")

        res = ExtractClusters().execute(context, {
            "scan_path": path, "channel": params.get("channel") or "Z",
            "polarity": params.get("polarity") or "bright", "level": "plane"})
        if not res.success:
            return SkillResult(skill_name=_NAME, success=False,
                               error=f"团簇提取失败: {res.error}")
        data_in = res.data if isinstance(res.data, dict) else {}
        clusters = list(data_in.get("clusters") or [])
        tx = float(params["target_x_m"])
        ty = float(params["target_y_m"])
        inside = self._frame_contains(data_in, tx, ty)
        if inside is False:
            return SkillResult(
                skill_name=_NAME, success=True,
                data={"verdict": "undecidable", "reasons": ["target_outside_frame"],
                      "frame_contains_target": False, "n_clusters": len(clusters),
                      "tolerance_m": tol, "tolerance_source": tol_src, "scan_path": path},
                summary="目标不在这一帧里,这一帧答不了")

        min_h = params.get("min_peak_height_m")
        cands = []
        for c in clusters:
            x, y = c.get("x_m"), c.get("y_m")
            if x is None or y is None:
                continue
            if min_h is not None and (c.get("peak_height_m") or 0.0) < float(min_h):
                continue
            cands.append((math.hypot(float(x) - tx, float(y) - ty), c))
        cands.sort(key=lambda t: t[0])

        warns: list[str] = []
        exp = params.get("expected_count")
        if exp is not None and len(clusters) != int(exp):
            warns.append("count_mismatch")
        base = {"tolerance_m": tol, "tolerance_source": tol_src, "n_clusters": len(clusters),
                "n_candidates": len(cands), "frame_contains_target": True,
                "scan_path": path, "warnings": warns,
                "others": [{"x_m": c.get("x_m"), "y_m": c.get("y_m"), "dist_m": d,
                            "peak_height_m": c.get("peak_height_m")} for d, c in cands[1:6]]}
        if not cands:
            return SkillResult(skill_name=_NAME, success=True,
                               data={"verdict": "not_found", **base},
                               summary="这一帧的目标附近没有团簇")
        d0, best = cands[0]
        n_close = sum(1 for d, _ in cands if d <= 2 * tol)
        base.update({"residual_m": d0, "residual_nm": d0 * 1e9,
                     "found_x_m": best.get("x_m"), "found_y_m": best.get("y_m"),
                     "nearest_cluster": best, "n_candidates_within_2tol": n_close})
        if d0 <= tol and n_close <= 1:
            verdict, summary = "at_target", f"原子在目标上,残差 {d0 * 1e12:.0f} pm"
        elif d0 <= 2 * tol or n_close > 1:
            verdict, summary = "ambiguous", f"目标附近有 {n_close} 个候选,说不清是哪一个"
        elif d0 <= 10 * tol:
            verdict, summary = "displaced", f"找到了,但偏了 {d0 * 1e9:.2f} nm"
        else:
            verdict, summary = "not_found", f"最近的团簇在 {d0 * 1e9:.1f} nm 外"
        return SkillResult(skill_name=_NAME, success=True, data={"verdict": verdict, **base},
                           summary=summary)

    @staticmethod
    def _tolerance(params: dict) -> tuple[float | None, str]:
        given = params.get("tolerance_m")
        if given is not None:
            return float(given), "param"
        try:
            from mast.core.sample_facts import resolve_substrate

            nn = getattr(resolve_substrate(None), "nearest_neighbor_nm", None)
            if nn:
                return float(nn) * 1e-9 / 2.0, "sample_facts"
        except Exception:  # noqa: BLE001
            pass
        return None, "none"

    @staticmethod
    def _frame_contains(data: dict, x: float, y: float) -> bool | None:
        """Is the target inside the frame's footprint? None = the geometry is unreadable.

        ``ExtractClusters`` puts the geometry in a ``frame`` block; looking for it at the top
        level finds nothing, and the check then silently never runs. A target the frame does
        not cover would come back as ``not_found`` — "the atom is not there" — when the honest
        answer is "this frame cannot say".
        """
        geom = data.get("frame") if isinstance(data.get("frame"), dict) else data
        cx, cy = geom.get("cx_m"), geom.get("cy_m")
        w, h = geom.get("w_m"), geom.get("h_m")
        if None in (cx, cy, w, h):
            return None
        return abs(x - float(cx)) <= float(w) / 2 and abs(y - float(cy)) <= float(h) / 2
