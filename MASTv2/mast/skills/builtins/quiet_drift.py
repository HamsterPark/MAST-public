# -*- coding: utf-8 -*-
"""静置长采集：把 0.001–1 Hz 这段频率量准。

**这个技能不采数据。** aux 通道本来就在以 ~0.25 s 的间隔连续记录 Z / qPlus 振幅
/ Δf / dI-dV / 偏压，并保留 7 天（``cm_aux_keep_hours``，默认 168）。所以要的
不是"再采一路"，而是三件别的事：

1. **保证这段时间是安静的** —— 停扫描，并在事后核对整段里确实没有扫描
   （aux 每一行都带 ``ctx_scanning``）。扫描时 Z 里装的是形貌和慢轴运动，
   拿它做环境谱等于在量扫描器自己。
2. **等够长** —— 要分辨 60 s 的东西，30 分钟给 30 个周期、频率分辨率 5.6e-4 Hz；
   而一帧扫描图只有 600 秒、10 个周期。这是本技能相对 ``AnalyseSlowDrift``
   的唯一优势，也是它值得占机时的唯一理由。
3. **事后取历史并做谱** —— 从 ``/monitoring/aux/series`` 拉回来，扣趋势，
   报周期成分。

与 ``AnalyseSlowDrift`` 的分工
────────────────────────────
* ``AnalyseSlowDrift`` 从**已有的扫描图**里找，免费，而且有一个这里没有的
  判据：换扫描角它转不转（区分时间性扰动与样品结构）。代价是帧长带来的谱泄漏
  ——那个陷阱在它的模块注释里有完整记录。
* 本技能占机时，但没有帧长这个人为周期，频率分辨率高一个量级。

**两者是互补的，不是替代**：扫描图能证明"不是样品结构"，静置能证明"周期是多少"。
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

__all__ = ["CharacteriseQuietDrift"]

#: 要分辨一个周期，至少得装下这么多个。10 个周期时频率分辨率约为频率的 10%，
#: 再少就谈不上"测出周期"，只能说"看起来有起伏"。
_MIN_CYCLES = 10.0


class CharacteriseQuietDrift(BaseSkill):
    """静置一段时间，然后从 aux 历史里量慢扰动。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CharacteriseQuietDrift",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "停扫描、静置若干分钟，再从 aux 历史做 0.001–1 Hz 的谱。"
                "比从扫描图里找频率分辨率高一个量级，且没有帧长造成的谱泄漏。"
            ),
            parameters=[
                ParameterSpec(
                    name="minutes", type="float", unit="min", required=False,
                    default=30.0, min_value=2.0, max_value=240.0,
                    description=("静置多久。要量 T 秒的周期，至少给 %.0f*T 秒；"
                                 "30 分钟够 60 s 的东西（30 个周期）" % _MIN_CYCLES),
                ),
                ParameterSpec(
                    name="stop_scan", type="bool", required=False, default=True,
                    description=("先停扫描。**默认 True** —— 扫描时 Z 里装的是"
                                 "形貌和慢轴运动，做出来的谱是扫描器自己的"),
                ),
                ParameterSpec(
                    name="column", type="str", required=False, default="z_m",
                    description="要分析的 aux 列（z_m / amp_m / df_hz / bias_v）",
                ),
                ParameterSpec(
                    name="analyse_only", type="bool", required=False, default=False,
                    description=("跳过等待，直接分析**最近** minutes 分钟的历史。"
                                 "机器已经静置过时用它，不必再占机时"),
                ),
                ParameterSpec(
                    name="max_scanning_fraction", type="float", required=False,
                    default=0.02, min_value=0.0, max_value=1.0,
                    description=("整段里允许有多少比例在扫描。超过就判这段不干净 "
                                 "—— 不是失败，是**这段数据回答不了这个问题**"),
                ),
            ],
            estimated_duration_s=1800.0,
            tags=["drift", "environment", "diagnostics", "quiet"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import time

        import numpy as np

        name = self.metadata().name
        minutes = float(params.get("minutes") or 30.0)
        column = str(params.get("column") or "z_m")
        analyse_only = bool(params.get("analyse_only", False))
        max_scan_frac = float(params.get("max_scanning_fraction") or 0.02)

        t_end_target = time.time()
        if not analyse_only:
            if bool(params.get("stop_scan", True)) and context is not None:
                try:
                    context.run("StopScan", {})
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: StopScan 失败: %s", name, exc)
            t0 = time.time()
            # 用基类的可中断等待 —— 直接 sleep 会让 abort 按钮变成摆设
            self.abortable_sleep(context, minutes * 60.0)
            t_end_target = time.time()
            if t_end_target - t0 < minutes * 60.0 * 0.9:
                return SkillResult(
                    skill_name=name, success=False,
                    error="等待被中断（只过了 %.1f 分钟，要 %.1f）—— 这段太短，"
                          "做出来的谱分辨不了目标频率。"
                          % ((t_end_target - t0) / 60.0, minutes))

        since = t_end_target - minutes * 60.0
        rows = self._fetch(context, since, t_end_target, column)
        if rows is None:
            return SkillResult(skill_name=name, success=False,
                               error="取不到 aux 历史（monitoring 没在跑？）")
        t, y, scanning = rows
        if len(t) < 64:
            return SkillResult(
                skill_name=name, success=False,
                error="这段里只有 %d 个 aux 样本，做不了谱。检查 cm_aux_enabled "
                      "与 cm_aux_keep_hours。" % len(t))

        scan_frac = float(np.mean(scanning)) if len(scanning) else 0.0
        span = float(t[-1] - t[0])
        dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.0
        data: dict = {
            "column": column, "n_samples": len(t), "span_s": span,
            "dt_s": dt, "fs_hz": (1.0 / dt) if dt > 0 else 0.0,
            "scanning_fraction": scan_frac,
            "freq_resolution_hz": (1.0 / span) if span > 0 else 0.0,
            "longest_resolvable_period_s": (span / _MIN_CYCLES) if span > 0 else 0.0,
            "warnings": [],
        }
        if scan_frac > max_scan_frac:
            data["warnings"].append(
                "这段里有 %.0f%% 的样本处于扫描中 —— 扫描时 Z 装的是形貌与慢轴"
                "运动，谱峰会是扫描器的节奏而不是环境的。**这段回答不了这个问题**，"
                "重来一次并确保停扫描。" % (scan_frac * 100))
            data["clean"] = False
            return SkillResult(skill_name=name, success=True, data=data)
        data["clean"] = True

        comps, resid, trend = self._spectrum(t, y, span, dt)
        data["components"] = comps
        data["residual_rms"] = resid
        data["trend_per_hour"] = trend
        if comps:
            top = comps[0]
            data["advice"] = (
                "最强成分：周期 %.1f s（%.5f Hz），幅度 %.3g，本底的 %.1f 倍。"
                "本段能分辨到 %.0f s 为止（%.0f 秒 / %.0f 个周期）。"
                "**这一条没有「换扫描角它转不转」的证据** —— 要排除它是样品上的"
                "结构，配合 AnalyseSlowDrift 跑一批不同扫描角的帧。"
                % (top["period_s"], top["freq_hz"], top["amplitude"],
                   top["over_floor"], data["longest_resolvable_period_s"],
                   span, _MIN_CYCLES))
        else:
            data["advice"] = (
                "没有高过本底的周期成分。残差 rms %.3g 就是这段时间里 %s 的慢起伏"
                "总量 —— 一个干净的否定，比一张勉强凑出来的周期表有用。"
                % (resid, column))
        return SkillResult(skill_name=name, success=True, data=data)

    # ------------------------------------------------------------------

    @staticmethod
    def _fetch(context, since: float, until: float, column: str):
        """从 monitoring store 直接读 —— 不绕 HTTP，也就没有 max_points 抽稀。"""
        import numpy as np

        try:
            from mast.monitoring.store import get_store_if_exists
            store = get_store_if_exists()
            if store is None:
                return None
            rows, _total, _thinned = store.aux_query(
                since=since, until=until, limit=200000, max_points=200000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("aux_query 失败: %s", exc)
            return None
        t, y, sc = [], [], []
        for r in rows or []:
            v = r.get(column)
            if v is None:
                continue
            t.append(float(r.get("ts") or 0.0))
            y.append(float(v))
            sc.append(bool(r.get("ctx_scanning")))
        if not t:
            return None
        return np.asarray(t), np.asarray(y), np.asarray(sc)

    @staticmethod
    def _spectrum(t, y, span: float, dt: float):
        import numpy as np

        # aux 有 busy 跳过，间隔不完全均匀 —— 先插到均匀栅格，否则 FFT 的
        # 频率轴是错的（而它不会报错，只会给出一个偏移了的峰）。
        grid = np.arange(t[0], t[-1], dt if dt > 0 else 1.0)
        v = np.interp(grid, t, y)
        n = len(v)
        if n < 64:
            return [], float("nan"), float("nan")
        coef = np.polyfit(np.arange(n), v, 1)
        v = v - np.polyval(coef, np.arange(n))
        trend_per_h = float(coef[0]) / max(dt, 1e-9) * 3600.0

        F = np.abs(np.fft.rfft(v * np.hanning(n)))
        f = np.fft.rfftfreq(n, d=dt)
        amp = 2.0 * F / (n * 0.5)
        # 下限由「至少 _MIN_CYCLES 个周期」定，上限留在 Nyquist 之下
        band = (f >= _MIN_CYCLES / span) & (f <= 0.45 / max(dt, 1e-9))
        if band.sum() < 8:
            return [], float(np.std(v)), trend_per_h
        fb, ab = f[band], amp[band]
        floor = float(np.median(ab))
        # ═══════════════════════════════════════════════════════════════
        # 阈值必须随**频点数**走，不能是一个常数
        # ═══════════════════════════════════════════════════════════════
        # 纯噪声谱里 N 个点的最大值不是 floor，而是约 floor*sqrt(2*ln N) ——
        # N≈1000 时就是 3.7 倍。所以一个固定的「3 倍本底」在长序列上**必然
        # 常态误报**：2026-08-20 的 59 分钟静置数据（真正什么都没有，残差只有
        # 1.00 pm）被它报出六条「成分」，全部落在 3.3–3.6 倍，而 advice 还会
        # 一本正经地说「最强成分周期 2.2 s」。
        #
        # 留 1.25 的余量：噪声本底不是严格瑞利（相邻频点相关、窗函数有旁瓣），
        # 而漏报一条弱峰的代价远小于报出一张全是噪声的周期表。
        n_bins = int(ab.size)
        thresh = floor * 1.25 * float(np.sqrt(2.0 * np.log(max(n_bins, 2))))
        out, shown = [], []
        for i in np.argsort(-ab):
            if len(out) >= 6:
                break
            if ab[i] < thresh:
                break
            if any(abs(fb[i] - s) < 2.0 / span for s in shown):
                continue
            shown.append(fb[i])
            out.append({"freq_hz": float(fb[i]), "period_s": float(1.0 / fb[i]),
                        "amplitude": float(ab[i]),
                        "over_floor": float(ab[i] / floor) if floor else 0.0,
                        "threshold_over_floor": float(thresh / floor) if floor else 0.0})
        return out, float(np.std(v)), trend_per_h
