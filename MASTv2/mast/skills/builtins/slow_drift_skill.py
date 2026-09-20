# -*- coding: utf-8 -*-
"""慢扰动（0.001–1 Hz）的测量与判别。

这一段频率是仪器诊断里最尴尬的一段：比机械振动慢，比热漂移快，而现有的
每一路数据都恰好错过它 ——

* 电流监控 2 kHz，但按秒分段，段内谱看不到 60 s 的东西；
* aux 通道 0.25 s 间隔够快，**窗口只有 300 s**（60 s 的周期只有五个）；
* 环境历史的原始读数约 3.8 s 间隔、够快，但 API 只暴露 **60 s 统计桶** ——
  Nyquist 正好 120 s，把这一段整个滤掉。

**扫描图正好补上这个洞**：慢轴每行 ``line_time`` 秒，一帧就是一条几百秒长、
采样率 0.4 Hz 左右的时间序列。而且它给出一个别处拿不到的判据 ——
**换扫描角，它转不转**：样品上的结构跟着转，时间性的扰动不转。

两条必须一起看的检验
──────────────────
1. **跨扫描角**：同一周期出现在两个以上不同扫描角上 ⇒ 时间性的。
2. **跨帧长**：真周期在秒轴上不动；谱泄漏的「周期」是帧长的整数分之一，
   会跟着帧长走。**所有帧同长时这一条做不了** —— 那时本技能只标可疑，
   不下结论，并告诉你还缺什么。

第 2 条是 2026-08-19 花了一轮才想明白的：六帧同为 629 s，「发现」的周期是
209.7 / 125.8 / 84.4 / 41.9 s，全是 629/3、629/5、629/7.4、629/15。一张
看起来很确定的周期表，里面一半是窗函数的旁瓣。
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

__all__ = ["AnalyseSlowDrift"]


class AnalyseSlowDrift(BaseSkill):
    """从一批扫描图里找慢扰动，并判断它是不是时间性的。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AnalyseSlowDrift",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从扫描图的行序列提取 0.001–1 Hz 的慢扰动，用跨扫描角判断它是否"
                "时间性的、用跨帧长排除谱泄漏。这一段频率其他数据通道都够不着。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_paths", type="str", required=True,
                    description=("多个 .sxm 路径，逗号分隔。**扫描角与帧长越杂越好** "
                                 "—— 同角度只是重复，同帧长排除不了谱泄漏"),
                ),
                ParameterSpec(name="channel", type="str", required=False,
                              default="Z", description="通道（恒流模式用 Z）"),
                ParameterSpec(
                    name="min_coverage", type="float", required=False,
                    default=0.9, min_value=0.1, max_value=1.0,
                    description="残帧的行序列有缺口，会在谱上生出假峰；低于这个"
                                "有效像素比例的帧直接不用",
                ),
                ParameterSpec(
                    name="max_components", type="int", required=False,
                    default=4, min_value=1, max_value=12,
                    description="每帧最多报几条成分"),
            ],
            estimated_duration_s=15.0,
            tags=["drift", "environment", "diagnostics", "multiframe"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import hashlib

        import numpy as np

        from mast.skills.builtins.atomic_multiframe import _load_one, _split_paths
        from mast.vision.slow_drift import analyse_slow_drift, combine_frames

        name = self.metadata().name
        paths = _split_paths(params.get("scan_paths", ""))
        channel = str(params.get("channel") or "Z")
        min_cov = float(params.get("min_coverage") or 0.9)
        maxc = int(params.get("max_components") or 4)
        if not paths:
            return SkillResult(skill_name=name, success=False,
                               error="没有给 scan_paths")

        seen: set[str] = set()
        results, rejected, used = [], [], []
        for p in paths:
            img, mpp, ang, lt, w, err = _load_one(p, channel)
            if err:
                rejected.append({"path": p, "reason": err})
                continue
            cov = float(np.isfinite(img).mean())
            if cov < min_cov:
                rejected.append({"path": p,
                                 "reason": "只有 %.0f%% 的行有数据" % (cov * 100)})
                continue
            if not lt or lt <= 0:
                rejected.append({"path": p, "reason": "文件头里没有每行时间"})
                continue
            # 同一帧被存成多份是常事（autosave + 手工 SaveScan）。把重复当成
            # 独立证据会让「几个角度上都出现」这句话凭空变强。
            h = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()
            if h in seen:
                rejected.append({"path": p, "reason": "与已用帧逐位相同（重复存盘）"})
                continue
            seen.add(h)
            label = str(p).replace("\\", "/").split("/")[-1]
            results.append(analyse_slow_drift(img, lt, scan_angle_deg=ang,
                                              label=label, max_components=maxc))
            used.append({"path": p, "angle_deg": ang, "line_time_s": lt,
                         "span_s": results[-1].span_s})

        if not results:
            return SkillResult(skill_name=name, success=False,
                               error="没有可用的帧", data={"rejected": rejected})

        comb = combine_frames(results)
        groups = comb.get("groups", [])
        # 「站得住」的门槛分两级，**取决于这批数据里有没有实质不同的帧长**：
        #
        #   有（≥25% 差异）⇒ 必须 span_invariant：跨扫描角 + 跨帧长周期不变。
        #                     这是能给出的最强证据。
        #   没有           ⇒ 退回到「跨角度且不是帧长谐波」，并且**明说**
        #                     帧长这一关没考过。
        #
        # 第一版只用 time_locked，于是一条在 627 s 帧上是 209.7 s、在 313 s 帧
        # 上是 104.9 s 的成分（教科书式的泄漏，周期严格跟着帧长减半）被列进
        # 了 confirmed，还因为「6 个扫描角」排在最前面。跨角度排除的是样品
        # 结构，它对窗函数造出来的东西完全无能为力 —— 两道关拦的不是同一样
        # 东西，不能拿一道去顶另一道。
        has_two_spans = any(g.get("spans_really_differ") for g in groups)
        if has_two_spans:
            solid = [g for g in groups if g.get("span_invariant")]
        else:
            solid = [g for g in groups if g.get("time_locked")
                     and not g.get("leakage_suspect")]

        data = {
            "n_frames_used": len(results),
            "frames": used,
            "rejected": rejected,
            "frame_spans_s": comb.get("frame_spans_s"),
            "single_frame_length": comb.get("single_frame_length"),
            "components": groups,
            "confirmed": solid,
            "n_leakage_suspect": comb.get("n_leakage_suspect"),
            "residual_rms_m": comb.get("residual_rms_m"),
            "trend_nm_per_h": comb.get("trend_nm_per_h"),
            "warnings": list(comb.get("warnings") or []),
        }

        angles = sorted({round(float(u["angle_deg"]), 1) for u in used})
        if len(angles) < 2:
            data["warnings"].append(
                "所有帧都是同一个扫描角（%s°）—— **分不开「时间性扰动」和「样品上的"
                "结构」**，因为后者只有换角度才会露出马脚。这一批只能算重复。"
                % angles[0] if angles else "?")
        data["has_two_distinct_frame_lengths"] = has_two_spans
        if not has_two_spans:
            data["warnings"].append(
                "这批帧没有**实质不同**的帧长（差异都不到 25%），所以「不是谱"
                "泄漏」这一关没考过 —— 列出来的成分只经过了跨扫描角那一关。"
                "补一批帧长差一半的帧，判据会强得多。")
        if solid:
            best = solid[0]
            data["advice"] = (
                "站得住的最强成分：周期 %.0f s（%.4f Hz），幅度中位 %.1f pm，"
                "在 %d 个不同扫描角上出现且不是帧长的整数分之一。"
                "接下来该问的是「什么东西以这个周期在动」—— 它在慢轴上的幅度是 Z 的"
                "共模起伏，可以拿去和环境历史（温度 / 真空 / 氦位）对时间。"
                % (best["period_s"], best["freq_hz"],
                   best["amplitude_m_median"] * 1e12, best["n_distinct_angles"]))
        elif groups:
            data["advice"] = (
                "找到 %d 条候选成分，但没有一条同时满足「跨扫描角」和「不是帧长"
                "谐波」。%s"
                % (len(groups),
                   "先补一批**不同帧长**的帧（改 line_time 或行数）。"
                   if comb.get("single_frame_length") else
                   "先补一批**不同扫描角**的帧。"))
        else:
            data["advice"] = (
                "这批帧上没有高过本底的周期成分 —— 慢扰动在这段时间里不显著。"
                "残差 rms 中位 %.1f pm 可以当作「这段时间的慢起伏总量」。"
                % (float(np.median(comb.get("residual_rms_m") or [0])) * 1e12))
        return SkillResult(skill_name=name, success=True, data=data)
