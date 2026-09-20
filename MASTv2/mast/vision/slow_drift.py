# -*- coding: utf-8 -*-
"""从扫描图里提取慢扰动（0.001–1 Hz）。

**为什么用扫描图当探针**
────────────────────
这个频段最难测：aux 通道够快（0.25 s）但只留 300 s 窗口（60 s 的东西只有五个
周期）；环境历史的原始读数间隔约 3.8 s、够快，但 API 只暴露 60 s 统计桶 ——
Nyquist 正好 120 s，**恰好把这个频段滤掉**。

而扫描图天生就是一台慢采样器：慢轴每推进一行花 ``line_time`` 秒，一帧 256 行
就是一条 629 秒、采样率 0.4 Hz 的时间序列。更要紧的是它提供了一个**别的数据源
给不了的判据**：

    换扫描角，它转不转。

样品上的结构会跟着扫描角一起转；时间性的扰动不会 —— 它永远沿图像慢轴。
2026-08-19 就是靠这一条认出那个 57–63 s 的东西的：它在 0°/60°/120° 三个扫描角
上都出现，而且永远在 ±90°。单帧、或者任何单通道时间序列，都做不到这个区分。

**读数的物理含义**
────────────────
恒流模式下 Z 是反馈的输出，所以行均值的起伏 = 针尖-样品距离的慢变化，单位是
米。它包含真实的样品倾斜与热漂移（那是低频端的斜坡，先扣掉），剩下的周期性
成分才是这里要找的。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DriftComponent",
    "SlowDriftResult",
    "row_time_series",
    "analyse_slow_drift",
    "combine_frames",
]

#: 低于这个频率的成分当作漂移/倾斜，不当周期性扰动。一帧只有几百秒，
#: 比它更低的"周期"在一帧里连一个整周期都凑不出，谈不上周期。
_MIN_CYCLES_IN_FRAME = 2.5

# 帧长需充分不同，泄漏位置 span/k 才会移动。
# 接近相同的帧长不能当成独立佐证；相对差异门是通用软件阈值，需结合输入解释。
_SPAN_DISTINCT_MIN = 0.25

# 跨不同帧时长比较周期时允许的相对散布。
# 用于区分保持自身时间尺度的变化与随帧长移动的泄漏分量；应用前须验证容差。
_SPAN_INVARIANT_TOL = 0.15

#: 一个峰要高过局部本底多少倍才算数。3 倍是谱峰检测的常规下限；
#: 这个值不敏感，因为真实的周期成分通常高一个数量级。
_PEAK_OVER_FLOOR = 3.0


@dataclass
class DriftComponent:
    """一条周期性成分。"""

    freq_hz: float
    period_s: float
    amplitude_m: float
    over_floor: float
    #: 这一条是从哪一帧上量到的
    label: str = ""
    #: 该帧的扫描角。**跨角度比较靠它**。
    scan_angle_deg: float = 0.0
    #: 该帧的时长（秒）。**跨帧长比较靠它** —— 判泄漏时必须知道这一条成分
    #: 是从多长的帧上量到的，用「所有帧的帧长」去算比值是错的。
    span_s: float = 0.0


@dataclass
class SlowDriftResult:
    ok: bool = False
    reason: str = ""
    n_frames: int = 0
    fs_hz: float = 0.0
    span_s: float = 0.0
    components: list = field(default_factory=list)
    #: 行均值序列在扣掉线性趋势之后的 rms（m）—— 「慢扰动一共有多大」
    residual_rms_m: float = float("nan")
    #: 线性趋势本身，换算成 nm/小时。它是热漂移，不是扰动。
    trend_nm_per_h: float = float("nan")
    warnings: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def row_time_series(image, line_time_s: float):
    """一帧 → 一条时间序列（行均值），并扣掉线性趋势。

    返回 ``(t, y, trend_slope_m_per_s)``。行均值把快轴方向平均掉，留下的就是
    慢轴（= 时间）方向的共模起伏 —— 原子级的横向结构在这一步被平均没了，
    正好，这里要的不是它。
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 8:
        return None, None, float("nan")
    with np.errstate(invalid="ignore"):
        y = np.nanmean(a, axis=1)
    good = np.isfinite(y)
    if good.sum() < 8:
        return None, None, float("nan")
    n = len(y)
    t = np.arange(n, dtype=np.float64) * float(line_time_s)
    # 缺行用线性插值补上，否则 FFT 会在缺口处生出假峰
    if not good.all():
        y = np.interp(t, t[good], y[good])
    coef = np.polyfit(t, y, 1)
    return t, y - np.polyval(coef, t), float(coef[0])


def analyse_slow_drift(image, line_time_s: float, *, scan_angle_deg: float = 0.0,
                       label: str = "", max_components: int = 4) -> SlowDriftResult:
    """单帧的慢扰动谱。"""
    out = SlowDriftResult(n_frames=1)
    t, y, slope = row_time_series(image, line_time_s)
    if t is None:
        out.reason = "too_few_rows"
        return out

    n = len(y)
    out.fs_hz = 1.0 / float(line_time_s)
    out.span_s = float(t[-1])
    out.residual_rms_m = float(np.std(y))
    out.trend_nm_per_h = float(slope) * 1e9 * 3600.0

    win = np.hanning(n)
    F = np.abs(np.fft.rfft(y * win))
    freqs = np.fft.rfftfreq(n, d=float(line_time_s))
    # Hanning 的相干增益 0.5 —— 不除掉的话幅度系统性偏小一半
    amp = 2.0 * F / (n * 0.5)

    f_min = _MIN_CYCLES_IN_FRAME / out.span_s if out.span_s > 0 else 0.0
    band = freqs >= f_min
    if band.sum() < 4:
        out.reason = "frame_too_short"
        out.warnings.append(
            "这一帧只有 %.0f s，装不下 %.1f 个周期的最低频成分 —— 扫慢一点或"
            "多扫几行才谈得上「周期」。" % (out.span_s, _MIN_CYCLES_IN_FRAME))
        return out

    fb, ab = freqs[band], amp[band]
    floor = float(np.median(ab))
    order = np.argsort(-ab)
    comps = []
    for i in order:
        if len(comps) >= max_components:
            break
        if ab[i] < floor * _PEAK_OVER_FLOOR:
            break
        # 同一个峰的旁瓣不要重复报
        if any(abs(fb[i] - c.freq_hz) < 1.5 / out.span_s for c in comps):
            continue
        comps.append(DriftComponent(
            freq_hz=float(fb[i]), period_s=float(1.0 / fb[i]) if fb[i] else float("inf"),
            amplitude_m=float(ab[i]), over_floor=float(ab[i] / floor) if floor else 0.0,
            label=label, scan_angle_deg=float(scan_angle_deg),
            span_s=float(out.span_s)))
    out.components = comps
    out.detail = {"floor_m": floor, "n_rows": n, "f_min_hz": f_min}
    out.ok = True
    if not comps:
        out.warnings.append(
            "这一帧上没有高过本底 %.0f 倍的周期成分 —— 慢扰动在这一帧里不显著。"
            % _PEAK_OVER_FLOOR)
    return out


def combine_frames(results, *, tol_rel: float = 0.20) -> dict:
    """把多帧的成分归并，并回答**它随扫描角转不转**。

    这是本模块唯一做得到、而任何单通道时间序列做不到的判断：

    * 一个成分若在多个**不同扫描角**的帧上都出现在同一频率 ⇒ 它固定在时间轴上，
      是环境/仪器的扰动；
    * 若只在同一个扫描角下出现 ⇒ 无法与样品上的结构区分（那种结构在图像里的
      空间频率随扫描角变，但你只看了一个角度）。

    频率相同、而扫描角不同，是**判据**；频率相同、扫描角也相同，只是**重复**。
    """
    ok = [r for r in results if r is not None and r.ok]
    if not ok:
        return {"ok": False, "reason": "no_usable_frame", "groups": []}

    # ═══════════════════════════════════════════════════════════════════
    # 归并要在**频率**上做，容差由帧长决定 —— 不能用固定的周期百分比
    # ═══════════════════════════════════════════════════════════════════
    # 一条谱线的频率分辨率是 1/span，换算到周期就是 T²/span：短帧上同样的
    # 频率误差对应大得多的周期误差。用固定的 20% 周期容差，2026-08-19 就把
    # A 帧的 125.8 s（=627/5，泄漏）和 B 帧的 104.9 s（=312/3，泄漏）并成了
    # 一条「跨两种帧长、周期不变」的假证据 —— **两个不同的泄漏合成了一个
    # 看起来最有说服力的结论**。这比漏掉一条真周期糟得多。
    #
    # 频率域里两条线算同一条的条件：间距小于各自分辨率之和（再留一点余量）。
    allc = [c for r in ok for c in r.components]
    groups = []
    for c in sorted(allc, key=lambda x: -x.amplitude_m):
        for g in groups:
            df = abs(c.freq_hz - g["freq_hz"])
            res = 1.0 / max(c.span_s, 1e-9) + 1.0 / max(g["span_min_s"], 1e-9)
            if df <= 1.5 * res:
                g["members"].append(c)
                g["freq_hz"] = float(np.mean([m.freq_hz for m in g["members"]]))
                g["period_s"] = 1.0 / g["freq_hz"] if g["freq_hz"] else float("inf")
                g["span_min_s"] = min(g["span_min_s"], c.span_s)
                break
        else:
            groups.append({"period_s": c.period_s, "freq_hz": c.freq_hz,
                           "span_min_s": c.span_s, "members": [c]})

    spans = sorted({round(float(r.span_s), 1) for r in ok})
    out = []
    for g in groups:
        angles = sorted({round(m.scan_angle_deg, 1) for m in g["members"]})
        amps = [m.amplitude_m for m in g["members"]]
        periods = [m.period_s for m in g["members"]]
        out.append({
            "period_s": float(np.mean(periods)),
            "period_spread_s": float(np.std(periods)),
            "freq_hz": float(1.0 / np.mean(periods)) if np.mean(periods) else 0.0,
            "amplitude_m_median": float(np.median(amps)),
            "amplitude_m_min": float(np.min(amps)),
            "amplitude_m_max": float(np.max(amps)),
            # 帧数按不同帧标识去重，不能把同一帧的多个谱成分当成多帧佐证。
            # 成员数量单独报告；现有判据仍使用各自的跨帧长、时间锁定与泄漏条件。
            "n_frames": len({m.label for m in g["members"]}),
            "n_members": len(g["members"]),
            "scan_angles_deg": angles,
            "n_distinct_angles": len(angles),
            # 两个以上不同扫描角上出现同一周期 ⇒ 它不随扫描角转 ⇒ 时间性的
            "time_locked": len(angles) >= 2,
            "frames": sorted({m.label for m in g["members"]}),
            # 成员与所属结果一起保存；结果排序后不能再靠 zip(out, groups) 的位置对应关联。
            "_members": g["members"],
        })
    out.sort(key=lambda d: -d["amplitude_m_median"])

    # ═══════════════════════════════════════════════════════════════════
    # 谱泄漏检验：**真周期与帧长无关，泄漏的"周期"是帧长的整数分之一**
    # ═══════════════════════════════════════════════════════════════════
    # 曾经差点报出一批伪影：六帧同为 629 s，"发现"的周期是
    # 209.7 / 125.8 / 84.4 / 67.1 s —— 全是 629/3、629/5、629/7.5、629/9.4。
    # 有限长序列扣掉线性趋势之后，残留的低频能量必然堆在这些位置上。
    #
    # 关键在于：**所有帧同长时，这个检验做不了**。谱峰在不在 span/k 上，
    # 真周期和泄漏给出的答案一样。要分开就得有**不同时长**的帧（换 line_time
    # 或换行数），真周期在秒轴上不动，泄漏会跟着帧长走。
    #
    # 所以这里不下结论，只标记可疑并说清楚缺什么。给出一个"看起来很确定"
    # 的周期表，比说"这批数据判不了"更糟。
    single_span = len(spans) <= 1
    for d in out:
        members = d["_members"]
        # ⚠ 比值只能拿**这条成分自己出现过的那些帧长**去算。
        # 第一版拿 spans（所有帧的帧长）去算，于是 209.7 s 那条 —— 它只在
        # 627 s 的帧上出现、是标准的 span/3 泄漏 —— 因为 312/209.7=1.49
        # 不是整数，被 all() 判成了「不是泄漏」。**加进第二种帧长反而让判据
        # 失效**，这是最坏的一种缺陷：数据变好了，结论变差了。
        own = sorted({round(float(m.span_s), 1) for m in members})
        ratios = [sp / d["period_s"] for sp in own if d["period_s"] > 0]
        near_int = (all(abs(r - round(r)) < 0.12 and round(r) >= 2 for r in ratios)
                    if ratios else False)
        d["own_frame_spans_s"] = own
        d["n_distinct_spans"] = len(own)
        d["frame_span_ratio"] = [round(r, 2) for r in ratios]
        d["harmonic_of_frame_length"] = bool(near_int)
        # 跨帧长不变需要直接比较不同帧长的代表周期，不能只看是否被频率分辨率合并。
        # 按帧长分组后，取幅度最大的成员作代表，避免主峰旁瓣平均拉偏位置。
        # 比较代表周期散布，区分周期一致与分辨率不足导致的表面重合。
        by_span: dict = {}
        for m in members:
            k = round(float(m.span_s), 1)
            cur = by_span.get(k)
            if cur is None or float(m.amplitude_m) > cur[1]:
                by_span[k] = (float(m.period_s), float(m.amplitude_m))
        reps = [v[0] for v in by_span.values()]
        if len(reps) >= 2 and np.mean(reps) > 0:
            spread = (max(reps) - min(reps)) / float(np.mean(reps))
        else:
            spread = float("nan")
        d["period_by_span_s"] = {k: round(v[0], 1) for k, v in sorted(by_span.items())}
        d["period_spread_across_spans"] = None if spread != spread else round(spread, 3)
        # 帧长必须具有实质差异，才可能移动泄漏位置；近乎相同的帧长不能作为两份独立证据。
        distinct = len(own) >= 2 and (max(own) - min(own)) / max(own) >= _SPAN_DISTINCT_MIN
        d["spans_really_differ"] = bool(distinct)
        d["span_invariant"] = bool(
            distinct and not near_int
            and spread == spread and spread <= _SPAN_INVARIANT_TOL)
        d["leakage_suspect"] = bool(near_int) and len(own) <= 1
    n_susp = sum(1 for d in out if d["leakage_suspect"])
    n_inv = sum(1 for d in out if d.get("span_invariant"))

    warn = []
    if single_span and out:
        warn.append(
            "所有帧时长相同（%.0f s），**分不开真实周期与谱泄漏** —— 两者都会把"
            "能量堆在 span/k 上。要判就得再采一批**不同时长**的帧（改 line_time "
            "或改行数）：真周期在秒轴上不动，泄漏会跟着帧长走。"
            % (spans[0] if spans else 0.0))
    if n_inv:
        warn.append(
            "%d 条成分在**两种以上帧长**上都出现且周期不随帧长变 —— 这是真周期"
            "最强的证据（比跨扫描角更强：扫描角只排除样品结构，帧长排除的是"
            "窗函数自己造出来的东西）。" % n_inv)
    if n_susp:
        warn.append(
            "%d/%d 个成分的周期正好是帧长的整数分之一（已标 leakage_suspect）"
            "—— 在只有一种帧长的数据上，这些**默认当泄漏看**。" % (n_susp, len(out)))

    for d in out:
        d.pop("_members", None)
    return {"ok": True, "reason": "", "groups": out,
            "n_frames": len(ok),
            "frame_spans_s": spans,
            "single_frame_length": single_span,
            "n_leakage_suspect": n_susp,
            "n_span_invariant": n_inv,
            "warnings": warn,
            "residual_rms_m": [r.residual_rms_m for r in ok],
            "trend_nm_per_h": [r.trend_nm_per_h for r in ok]}
