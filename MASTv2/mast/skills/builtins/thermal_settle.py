# -*- coding: utf-8 -*-
"""WaitForThermalSettle：用温度变化速率判断是否趋稳。

温度低于某个绝对值时仍可能快速变化，绝对阈值不能单独代表热稳定。
接近目标温度的缓慢漂移同样需要由速率描述；速率由最近多个点的线性拟合估计。
可选的绝对温度上限与速率上限同时检查，默认值须按测量条件验证。

只读取温度并等待，不控温、不操作加热器。读不到温度时明确报告不可用，
不能把未知状态当作稳定。
"""

from __future__ import annotations

import logging
import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

# 默认稳定速率上限（K/min），为未标定的工作流参数；应按当前热环境与测量要求验证。
_DEFAULT_RATE_K_PER_MIN = 0.03

#: 拟合速率至少要几个采样点 —— 两点连线算出来的「速率」全是噪声。
_MIN_SAMPLES = 5

#: 连续读不到温度多少次就放弃（每次都当「稳了」是最危险的降级）。
_MAX_MISSES = 5


def rate_k_per_min(samples):
    """对 (t_s, T_K) 线性拟合，返回 K/min。点不够返回 None（**不是 0**）。"""
    import numpy as np

    pts = [(t, v) for t, v in samples if v is not None]
    if len(pts) < _MIN_SAMPLES:
        return None
    a = np.asarray(pts, dtype=float)
    if a[-1, 0] - a[0, 0] <= 0:
        return None
    return float(np.polyfit(a[:, 0], a[:, 1], 1)[0] * 60.0)


class WaitForThermalSettle(BaseSkill):
    """Wait until the temperature stops changing, not until it drops below a number."""

    #: 轮询间隔（s）。类属性，测试可以缩短。
    _poll_s = 25.0

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="WaitForThermalSettle",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "等待温度变化趋稳，使用变化速率而非仅靠绝对温度。低于某个温度上限时仍可能快速变化。\n\n同时检查可选的 max_temp_k 与 max_rate_k_per_min：绝对温度上限可留空，速率按最近多点线性拟合，少于 %d 点时无法判定。\n\n读取失败如实报告，不假装稳定。只读温度并等待，不控温、不操作加热器。"
                 % _MIN_SAMPLES
            ),
            parameters=[
                ParameterSpec(
                    name="max_rate_k_per_min", type="float", unit="K/min",
                    description=("稳定速率上限的绝对值，默认 %.2f。此为工作流参数，使用前须按当前热环境与测量要求验证。"
                                 % _DEFAULT_RATE_K_PER_MIN),
                    required=False, default=_DEFAULT_RATE_K_PER_MIN,
                    min_value=0.001, max_value=5.0),
                ParameterSpec(
                    name="max_temp_k", type="float", unit="K",
                    description=("温度上限。**留空 = 只看速率**（比如在 77 K 上工作时"
                                 "就不该要求它降到 5 K）。给了值就是两条**并且**。"),
                    required=False, min_value=0.0, max_value=500.0),
                ParameterSpec(
                    name="timeout_s", type="float", unit="s",
                    description="最长等待。到点仍未稳就如实返回未稳，**不假装成功**。",
                    required=False, default=3600.0,
                    min_value=10.0, max_value=86400.0),
                ParameterSpec(
                    name="window", type="int",
                    description="拟合速率用的采样点数（滑动窗口）。",
                    required=False, default=6,
                    min_value=_MIN_SAMPLES, max_value=60),
            ],
            estimated_duration_s=600.0,
            composition_level=0,
            tags=["thermal", "wait", "settle", "温度", "换样品"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        max_rate = abs(float(params.get("max_rate_k_per_min")
                             or _DEFAULT_RATE_K_PER_MIN))
        max_temp = params.get("max_temp_k")
        max_temp = float(max_temp) if max_temp is not None else None
        budget = float(params.get("timeout_s") or 3600.0)
        window = max(_MIN_SAMPLES, int(params.get("window") or 6))

        t0 = time.monotonic()
        samples, misses = [], 0
        while time.monotonic() - t0 < budget:
            res = context.run("GetTemperature", {})
            val = (getattr(res, "data", None) or {}).get("value_k")
            if val is None:
                misses += 1
                if misses >= _MAX_MISSES:
                    return SkillResult(
                        skill_name="WaitForThermalSettle", success=False,
                        error=("连续 %d 次读不到温度 —— **不当成「稳了」**。"
                               "读不到与稳定是两件事，把前者当后者会让调用方"
                               "在未知热状态下开工。" % misses),
                        data={"samples": samples, "misses": misses})
            else:
                misses = 0
                samples.append((time.monotonic() - t0, float(val)))
                samples = samples[-window:]
                rate = rate_k_per_min(samples)
                if rate is not None:
                    temp_ok = (max_temp is None) or (val <= max_temp)
                    if temp_ok and abs(rate) <= max_rate:
                        return SkillResult(
                            skill_name="WaitForThermalSettle", success=True,
                            data={"settled": True, "temperature_k": float(val),
                                  "rate_k_per_min": rate,
                                  "elapsed_s": time.monotonic() - t0,
                                  "n_samples": len(samples),
                                  "max_rate_k_per_min": max_rate,
                                  "max_temp_k": max_temp,
                                  "message": ("%.3f K，速率 %+.3f K/min —— 稳了，"
                                              "可以开工。" % (val, rate))})
            time.sleep(self._poll_s)

        last = samples[-1][1] if samples else None
        rate = rate_k_per_min(samples)
        return SkillResult(
            skill_name="WaitForThermalSettle", success=False,
            error=("等了 %.0f min 仍未稳：%s，速率 %s。**没有假装成功** —— "
                   "在这个状态下开工，读到的东西会跟着温度走。"
                   % (budget / 60.0,
                      ("%.3f K" % last) if last is not None else "读不到温度",
                      ("%+.3f K/min" % rate) if rate is not None else "算不出")),
            data={"settled": False, "temperature_k": last,
                  "rate_k_per_min": rate, "elapsed_s": time.monotonic() - t0,
                  "n_samples": len(samples)})
