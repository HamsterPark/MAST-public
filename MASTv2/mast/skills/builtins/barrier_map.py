# -*- coding: utf-8 -*-
"""MapBarrierHeight：比较位置间散布与同位置重复散布。

没有重复对照，空间差异可能只是测量噪声。判读使用两类散布的比值，
超过或低于配置的边界分别提示 surface_side 或 tip_side，中间范围返回 inconclusive。
这些是工作流解释，需要结合测量条件验证，不能由单点结论推导污染来源。

本技能仅在给定位置测量 I–Z，不执行修针或粗动。
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

#: 判读的两条边（位置间散布 ÷ 同点重复散布）。中间留一段「不硬判」。
_TIP_SIDE_MAX = 2.0
_SURFACE_SIDE_MIN = 3.0

#: 重复次数下限。少于这个数，「重复散布」本身就没有意义 ——
#: 而它是整个判读的分母，分母不可信则结论不可信。
_MIN_REPEATS = 4

#: 位置数下限。三个点算不出可信的空间散布。
_MIN_SITES = 4


def _rel_spread(values):
    """相对散布（σ/μ）。少于 2 个值返回 None（**不是 0**）—— 那是「没有」不是「很小」。"""
    import numpy as np

    v = [x for x in values if x is not None]
    if len(v) < 2:
        return None
    a = np.asarray(v, dtype=float)
    m = float(a.mean())
    return float(a.std() / m) if m > 0 else None


def verdict_from(site_spread, repeat_spread):
    """位置间散布与重复散布之比 → 判读。

    两个都要有值才判 —— 缺任何一个都是 `undetermined`，不是「没差别」。
    """
    if site_spread is None or repeat_spread is None:
        return "undetermined", None
    if repeat_spread <= 0:
        return "undetermined", None
    ratio = site_spread / repeat_spread
    if ratio <= _TIP_SIDE_MAX:
        return "tip_side", ratio
    if ratio >= _SURFACE_SIDE_MIN:
        return "surface_side", ratio
    return "inconclusive", ratio


class MapBarrierHeight(BaseSkill):
    """Map the tunnelling barrier across sites, with a repeat-at-one-site noise ruler."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="MapBarrierHeight",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在多个位置测量 I–Z 势垒，并在其中一个位置重复测量作为噪声对照。只有比较空间变化与重复测量散布，才能解释差异是否超过测量波动。\n\n重复对照是必需项。surface_side 和 tip_side 分别提示优先检查表面或针尖，证据不足时返回 inconclusive。\n\n每个位置调用 MeasureBarrierHeight，由它确认退开方向、筛除触底点并在可用点不足时拒答；本技能如实计入这些拒答。"
            ),
            parameters=[
                ParameterSpec(
                    name="sites_nm", type="str",
                    description=(
                        "位置列表，形如 'x1,y1; x2,y2; …'（nm）。至少 %d 个。"
                        "建议横跨几百 nm —— 挨得太近量到的是同一块。" % _MIN_SITES),
                    required=True),
                ParameterSpec(
                    name="repeats", type="int",
                    description=(
                        "在**第一个位置**重复测的次数，用作噪声标尺。至少 %d 次。"
                        "**这是判读的分母，省掉它整个结论就没有意义。**" % _MIN_REPEATS),
                    required=False, default=6,
                    min_value=_MIN_REPEATS, max_value=20),
                ParameterSpec(
                    name="bias_v", type="float", unit="V",
                    description="测量偏压。留空 = 用当前偏压。", required=False,
                    min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="hop_size_m", type="float", unit="m",
                    description="位置之间用来挪针尖的小扫描视野。",
                    required=False, default=3e-9,
                    min_value=1e-9, max_value=1e-7),
            ],
            estimated_duration_s=1800.0,
            composition_level=2,
            tags=["spectroscopy", "barrier", "map", "diagnostic", "势垒", "针尖还是表面"],
        )

    @staticmethod
    def _parse_sites(raw):
        sites, bad = [], []
        for chunk in str(raw or "").split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(",")
            if len(parts) != 2:
                bad.append(chunk)
                continue
            try:
                sites.append((float(parts[0]), float(parts[1])))
            except ValueError:
                bad.append(chunk)
        return sites, bad

    def validate_params(self, params: dict) -> list[str]:
        errors = list(super().validate_params(params) or [])
        sites, bad = self._parse_sites(params.get("sites_nm"))
        if bad:
            errors.append("sites_nm 里有解析不了的项：%s（每项写成 `x,y`，分号分隔）"
                          % "; ".join(bad[:4]))
        elif len(sites) < _MIN_SITES:
            errors.append("至少要 %d 个位置才算得出可信的空间散布（给了 %d）"
                          % (_MIN_SITES, len(sites)))
        return errors

    def _measure_at(self, context, x_nm, y_nm, bias_v, hop_m):
        context.run("ScanAt", {
            "center_x_m": x_nm * 1e-9, "center_y_m": y_nm * 1e-9,
            "size_m": hop_m, "pixels": 64, "line_time_s": 0.04,
            "purpose": "survey",
        })
        p = {}
        if bias_v is not None:
            p["bias_v"] = float(bias_v)
        res = context.run("MeasureBarrierHeight", p)
        d = getattr(res, "data", None) or {}
        return {"xy_nm": [x_nm, y_nm], "verdict": d.get("verdict"),
                "phi_ev": d.get("phi_ev"), "kappa_per_nm": d.get("kappa_per_nm"),
                "n_fit": d.get("n_fit")}

    def execute(self, context, params: dict) -> SkillResult:
        # ⚠ **不要假设 validate_params 一定跑过。** execute 可能被直接调用
        # （诊断路径、守卫测试、别的 composite 内部）。抛异常的 skill 到不了
        # agent 的错误处理路径 —— 没有 SkillResult、没有诊断记录、没有 HITL、
        # 没有恢复，只剩一个死掉的回合。所以坏参数在这里也要**返回**而不是炸。
        # （2026-08-27 由 test_all_skills_execute 抓到：sites_nm='0' ⇒ IndexError。）
        sites, bad = self._parse_sites(params.get("sites_nm"))
        if bad or len(sites) < _MIN_SITES:
            return SkillResult(
                skill_name="MapBarrierHeight", success=False,
                error=("sites_nm 至少要 %d 个位置，每项写成 `x,y`（nm），分号分隔。"
                       "收到 %d 个可用位置%s" % (
                           _MIN_SITES, len(sites),
                           "，解析不了的：%s" % "; ".join(bad[:4]) if bad else "")))
        repeats = int(params.get("repeats") or 6)
        if repeats < _MIN_REPEATS:
            return SkillResult(
                skill_name="MapBarrierHeight", success=False,
                error=("repeats 至少 %d —— 它是判读的**分母**（位置间散布 ÷ 重复散布），"
                       "次数太少这个分母本身就不可信，整个结论跟着不可信。" % _MIN_REPEATS))
        bias_v = params.get("bias_v")
        hop_m = float(params.get("hop_size_m") or 3e-9)

        # ① 噪声标尺：第一个位置重复测
        rx, ry = sites[0]
        rep = [self._measure_at(context, rx, ry, bias_v, hop_m) for _ in range(repeats)]
        rep_phi = [r["phi_ev"] for r in rep if r.get("phi_ev") is not None]
        repeat_spread = _rel_spread(rep_phi)

        # ② 各位置各一条
        per_site = [self._measure_at(context, x, y, bias_v, hop_m) for x, y in sites]
        site_phi = [r["phi_ev"] for r in per_site if r.get("phi_ev") is not None]
        site_spread = _rel_spread(site_phi)

        verdict, ratio = verdict_from(site_spread, repeat_spread)
        n_undet = sum(1 for r in per_site + rep if r.get("verdict") == "undetermined")

        msg = {
            "surface_side": (
                "位置间散布是同点重复的 %.1f 倍，提示存在空间变化。应优先检查表面与局部测量条件，不能假定改变针尖形状即可消除差异。"
                ),
            "tip_side": (
                "位置间散布只有同点重复的 %.1f 倍 ⇒ **各处一样，跟着针尖走**。"
                "该处理的是针尖，换地方没用。"),
            "inconclusive": (
                "位置间散布是同点重复的 %.1f 倍，落在 %.0f–%.0f 之间的灰带里 —— **不硬判**。"
                "「针尖侧」和「表面侧」驱动的下一步相反（换针尖 vs 换样品），"
                "猜错要赔一整轮。加位置、加重复次数再来。"),
        }.get(verdict)
        if msg and verdict == "inconclusive":
            message = msg % (ratio, _TIP_SIDE_MAX, _SURFACE_SIDE_MIN)
        elif msg:
            message = msg % ratio
        else:
            message = ("判不了：%s。" % (
                "重复测没得出可用的 φ" if repeat_spread is None
                else "各位置没得出足够的 φ" if site_spread is None else "散布算不出来")
                + "**这不是「没有差别」** —— 没有噪声标尺，位置间的散布读不出任何意义。")

        import statistics
        return SkillResult(
            skill_name="MapBarrierHeight", success=True,
            data={
                "verdict": verdict,
                "message": message,
                "spread_ratio": ratio,
                "site_spread": site_spread,
                "repeat_spread": repeat_spread,
                "repeat_site_nm": [rx, ry],
                "n_repeats": len(rep_phi),
                "n_sites_measured": len(site_phi),
                "n_sites_requested": len(sites),
                "n_undetermined": n_undet,
                "phi_median_ev": statistics.median(site_phi) if site_phi else None,
                "phi_min_ev": min(site_phi) if site_phi else None,
                "phi_max_ev": max(site_phi) if site_phi else None,
                "per_site": per_site,
                "repeats": rep,
                "tip_side_max_ratio": _TIP_SIDE_MAX,
                "surface_side_min_ratio": _SURFACE_SIDE_MIN,
            })
