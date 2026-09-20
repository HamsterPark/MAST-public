# -*- coding: utf-8 -*-
"""长期监控历史的查询入口 —— 给 agent 用的那一个。

**为什么需要它**
──────────────
这台机器一直在存两套长期数据：

* ``aux`` 辅助读数 —— Z / qPlus 振幅 / Δf / dI-dV / 偏压，约 0.25 s 一行，
  保留 7 天（``cm_aux_keep_hours``，默认 168）；
* ``env_history`` 环境历史 —— 温度（SPM / Magnet）、真空、氦位、噪声电平、
  隧道电流，60 s 统计桶**永久保留**，原始读数保留 14 天。

而 2026-08-20 查下来：**agent 侧没有任何入口**。数据处理 agent 的三十来个工具
全是扫描图与谱的处理；两个 agent 的提示词里一次都没提过这两套数据存在。
也就是说，一台连续记录了半个月环境参数的仪器，它自己的 agent 够不着那些数据。

这不是"少了个便利函数"。它决定了一整类问题能不能被问出口：
「昨晚成像变差的时候温度在做什么」「这个 60 秒的起伏在 Z 上有没有」
「氦位掉下去之后噪声本底变了吗」—— 没有入口，这些问题连提都提不出来。

**查哪一套**
────────────
按你要的时间尺度选，别按名字选：

* 想看**秒到分钟**的东西（振动、慢扰动、Z 的起伏）⇒ ``source="aux"``。
  它约 0.25 s 一行，Nyquist 2 Hz。
* 想看**小时到天**的东西（温度漂移、氦位、真空趋势）⇒ ``source="env"``。
  它是 60 s 桶，**Nyquist 是 120 s** —— 比这更快的东西在这里根本看不见，
  拿它去找 60 秒周期的扰动会一无所获，而且不会有任何报错。
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

__all__ = ["QueryMonitorHistory"]

#: env_history 的聚合粒度（秒）。它决定了那一套数据能回答的最快频率。
_ENV_BUCKET_S = 60.0


class QueryMonitorHistory(BaseSkill):
    """查 aux 辅助读数 / 环境历史，并说清楚这套数据能回答什么。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="QueryMonitorHistory",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "查长期监控历史：aux 辅助读数（Z/振幅/Δf/偏压，~0.25 s，留 7 天）"
                "或环境历史（温度/真空/氦位，60 s 桶，永久）。返回统计摘要，"
                "并说明这套数据的时间分辨率能回答什么频率。"
            ),
            parameters=[
                ParameterSpec(
                    name="source", type="str", required=False, default="aux",
                    description=("aux = 辅助读数（秒~分钟尺度）；"
                                 "env = 环境历史（小时~天尺度，60 s 桶，"
                                 "**看不见比 120 s 更快的东西**）"),
                ),
                ParameterSpec(
                    name="column", type="str", required=False, default="z_m",
                    description=("aux 列名（z_m / amp_m / df_hz / bias_v / didv）"
                                 "或 env 传感器名（'sample_temperature' / 'vacuum' / "
                                 "'helium_level' / 'noise_level'）"),
                ),
                ParameterSpec(
                    name="hours", type="float", unit="h", required=False,
                    default=1.0, min_value=0.01, max_value=8760.0,
                    description="往回看多少小时",
                ),
                ParameterSpec(
                    name="until_ts", type="float", required=False, default=None,
                    description="窗口右端（Unix 秒）；不填就是现在。查过去某段用它",
                ),
                ParameterSpec(
                    name="max_points", type="int", required=False, default=2000,
                    min_value=10, max_value=100000,
                    description="返回的采样点上限（只影响返回，不影响统计）",
                ),
                ParameterSpec(
                    name="include_series", type="bool", required=False, default=False,
                    description=("是否把序列本身返回。默认只回统计摘要 —— "
                                 "几万个点塞进对话没有意义"),
                ),
            ],
            estimated_duration_s=5.0,
            tags=["monitoring", "history", "environment", "diagnostics"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import time

        name = self.metadata().name
        source = str(params.get("source") or "aux").strip().lower()
        column = str(params.get("column") or "z_m")
        hours = float(params.get("hours") or 1.0)
        until = params.get("until_ts")
        until = float(until) if until else time.time()
        since = until - hours * 3600.0
        mp = int(params.get("max_points") or 2000)
        want_series = bool(params.get("include_series", False))

        if source not in ("aux", "env"):
            return SkillResult(
                skill_name=name, success=False,
                error="source 只能是 aux 或 env。想看秒~分钟尺度用 aux；"
                      "想看小时~天尺度用 env。")

        if source == "aux":
            out = self._aux(column, since, until, mp)
        else:
            out = self._env(column, since, until, mp)
        if isinstance(out, str):
            return SkillResult(skill_name=name, success=False, error=out)

        if not want_series:
            out.pop("t_s", None)
            out.pop("values", None)
        out["source"] = source
        out["column"] = column
        out["window_h"] = hours
        return SkillResult(skill_name=name, success=True, data=out)

    # ------------------------------------------------------------------

    @staticmethod
    def _stats(t, v):
        import numpy as np

        t = np.asarray(t, float)
        v = np.asarray(v, float)
        good = np.isfinite(v)
        t, v = t[good], v[good]
        if v.size == 0:
            return None
        d = {
            "n": int(v.size),
            "span_s": float(t[-1] - t[0]) if v.size > 1 else 0.0,
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "first": float(v[0]), "last": float(v[-1]),
        }
        if v.size > 2 and d["span_s"] > 0:
            k = float(np.polyfit(t - t[0], v, 1)[0])
            d["slope_per_s"] = k
            d["slope_per_hour"] = k * 3600.0
            d["dt_median_s"] = float(np.median(np.diff(t)))
            # 「这套数据能回答什么」不是脚注，是决定结论成不成立的前提
            d["nyquist_period_s"] = 2.0 * d["dt_median_s"]
            d["freq_resolution_hz"] = 1.0 / d["span_s"]
        return d

    def _aux(self, column, since, until, mp):
        try:
            from mast.monitoring.store import get_store_if_exists
            store = get_store_if_exists()
            if store is None:
                return ("monitoring store 不可用 —— 电流监控没在跑，"
                        "aux 历史也就没有。")
            rows, total, thinned = store.aux_query(
                since=since, until=until, limit=200000, max_points=mp)
        except Exception as exc:  # noqa: BLE001
            return "aux 查询失败: %s" % exc
        t, v, sc = [], [], []
        for r in rows or []:
            x = r.get(column)
            if x is None:
                continue
            t.append(float(r.get("ts") or 0.0))
            v.append(float(x))
            sc.append(bool(r.get("ctx_scanning")))
        if not t:
            cols = sorted(set((rows or [{}])[0].keys())) if rows else []
            return ("这段时间里没有 %r 这一列的数据。可用的列：%s"
                    % (column, ", ".join(c for c in cols if not c.startswith("ctx_"))[:300]))
        st = self._stats(t, v)
        out = {"stats": st, "total_rows": int(total), "thinned": bool(thinned),
               "t_s": t, "values": v}
        frac = sum(sc) / len(sc) if sc else 0.0
        out["scanning_fraction"] = frac
        out["notes"] = []
        if frac > 0.02:
            out["notes"].append(
                "这段里有 %.0f%% 的样本处于**扫描中**。扫描时 Z 装的是形貌与慢轴"
                "运动，不是环境起伏 —— 要做环境谱就换一段安静的，或用 "
                "CharacteriseQuietDrift。" % (frac * 100))
        if st and st.get("nyquist_period_s"):
            out["notes"].append(
                "这套数据能回答的最快周期是 %.1f s（采样间隔 %.2f s 的两倍），"
                "能分辨的最长周期约 %.0f s。"
                % (st["nyquist_period_s"], st["dt_median_s"], st["span_s"] / 10.0))
        return out

    def _env(self, sensor, since, until, mp):
        try:
            from mast.envhistory.store import get_store as _env_store  # type: ignore
            store = _env_store()
        except Exception:  # noqa: BLE001
            store = None
        if store is None:
            try:
                from mast.envhistory import get_store as _g  # type: ignore
                store = _g()
            except Exception as exc:  # noqa: BLE001
                return "环境历史 store 不可用: %s" % exc
        try:
            d = store.series(str(sensor), since=since, until=until, max_points=mp)
        except Exception as exc:  # noqa: BLE001
            return "环境历史查询失败: %s" % exc
        pts = (d or {}).get("points") or []
        if not pts:
            try:
                names = [s.get("sensor") for s in (store.sensors() or [])]
            except Exception:  # noqa: BLE001
                names = []
            return ("传感器 %r 在这段时间里没有数据。可用的：%s"
                    % (sensor, ", ".join(str(n) for n in names)[:300]))
        # 存在时间桶但全部读取被排除，与该时间窗没有记录是不同状态。
        # 前者需要检查传感器或连接，后者需要检查记录时间窗；必须明确报告，不能只返回空统计。
        t_all = [float(p.get("ts") or 0.0) for p in pts]
        v = [float(p.get("mean")) for p in pts if p.get("mean") is not None]
        t = [tt for tt, p in zip(t_all, pts) if p.get("mean") is not None]
        n_excluded = sum(int(p.get("n_excluded") or 0) for p in pts)
        n_kept = sum(int(p.get("n") or 0) for p in pts)
        statuses = {str(p.get("worst_status") or "") for p in pts}
        if not v:
            worst = ", ".join(sorted(x for x in statuses if x)) or "未知"
            return ("传感器 %r 在这段时间里有 %d 个统计桶，但**每一个都是空的**"
                    "（读到 0 次、排除 %d 次，状态：%s）—— 记录器在跑、也在采，"
                    "是**传感器本身读不到值**。这不是「没有数据」，"
                    "该去查的是传感器与接线，不是换时间窗。"
                    % (sensor, len(pts), n_excluded, worst))
        st = self._stats(t, v)
        return {
            "stats": st, "unit": (d or {}).get("unit"),
            "bucket_s_effective": (d or {}).get("bucket_s_effective"),
            "thinned": bool((d or {}).get("thinned")),
            "total": int((d or {}).get("total") or 0),
            "t_s": t, "values": v,
            "n_readings_kept": n_kept, "n_readings_excluded": n_excluded,
            "worst_statuses": sorted(x for x in statuses if x),
            "notes": [
                ("这段里有 %d 次读取被排除（传感器不可用），保留 %d 次 —— "
                 "统计只基于保留的那些。" % (n_excluded, n_kept))
                if n_excluded else
                "所有读取都可用（没有被排除的采样）。",
                "环境历史是 **%.0f 秒统计桶**，Nyquist 因此是 %.0f 秒 —— 比这更快的"
                "东西在这套数据里根本不出现，而且不会有任何报错。要看秒~分钟尺度"
                "请改用 source='aux'。" % (_ENV_BUCKET_S, 2 * _ENV_BUCKET_S),
            ],
        }
