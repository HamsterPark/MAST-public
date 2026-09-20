"""报告级串扰导航 —— 把 lock-in X 读数对照参考曲线换算成「大约还剩多少步」。

**只报告,不驱动任何决策**(2026-08-06 定稿)。用户今天人肉用这条曲线导航过一次
退针;这里做的是把那个人肉动作自动化,**不是**把它接到电机上。决策级要等报告级在
真机上跑一阵、看它说的和实际发生的对不对得上,再议。

## 为什么报的是「≈N ± band 步」而不是一个整数

参考曲线末段极平:9800 → 10000 步之间 X 只掉 0.027 pA,而单点 σ ≈ 0.05 pA。照直
插值能算出「≈9967 步」—— 但那个数字后三位是噪声。误差带由**局部斜率**和读数自身的
散布算出来:``band ≈ sd / |dX/dstep|``。一个没有误差带的距离,读的人会当刻度用。

## 三种「不报」,每种都说得出是哪一种

* **调制没开** ⇒ X 上没有信号可读(这不是故障,是这台机器此刻不在做 lock-in);
* **信号索引没配** ⇒ 不知道 X 走哪一路 RT 信号,**不猜**(猜错读到的是另一路信号,
  而算出来的距离看上去一样合理);
* **没有条件匹配的曲线** ⇒ 原样报读数并说明是哪个条件对不上,连两边的值一起说 ——
  于是「换了针没重采曲线」这件事是**看得见**的,而不是一个安静的空转。

第三条尤其要紧:如果只报「未翻译」,一个永远匹配不上的曲线库和一个正常工作、只是今天
条件不符的曲线库长得一模一样。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from mast.core import calibration_curves as cc
from mast.core import instrument_profile as ip
from mast.core import tip_state

logger = logging.getLogger(__name__)

#: 低于这个幅度按「没有可用信号」处理(与 AutoPhase 同一条下界守卫)。
MIN_SIGNAL_A = 1e-13

#: 报出来的步数取整到这个粒度。曲线本身就是每 100 步一个点,报 9967 是在说一个
#: 采样根本达不到的分辨率。
STEP_GRANULARITY = 100


def live_tip_type() -> "str | None":
    """当前针尖的可比性标签,没登记就是 ``None``(不猜)。

    配方(**曲线里的 tip_type 必须按同一个配方写**):受控词表里的材料,qPlus 传感器
    追加 ``/qPlus``。`tip_state.is_qplus()` 在没登记针尖时返回 False(fail-open),
    所以这里先判「有没有登记」——否则一支没登记的针会拿到一个像模像样的标签。
    """
    row = tip_state.get_current_tip()
    if not row:
        return None
    mat = str(row.get("material") or "").strip()
    if not mat:
        return None
    return f"{mat}/qPlus" if tip_state.is_qplus() else mat


def live_conditions(ctx) -> "dict[str, Any]":
    """此刻的可比性条件 + 调制是否开着。读不到的键留 ``None``(不是 0)。"""
    out: "dict[str, Any]" = {"mod_amp_v": None, "mod_freq_hz": None,
                             "tip_type": live_tip_type(), "mod_on": None}
    try:
        res = ctx.run("GetLockInConfig", {})
    except Exception:  # noqa: BLE001
        logger.debug("读 lock-in 配置失败(条件按读不到处理)", exc_info=True)
        return out
    data = getattr(res, "data", None) or {}
    # ⚠️ 键名取自 GetLockInConfig 的**实际输出**:幅度那个键叫 `amplitude`,
    # 没有单位后缀(此前就有照着自己的参数名 `amplitude_v` 去查,查不到的缺陷)。
    # 有一条钉子对着真技能核这三个名字。
    amp, freq = data.get("amplitude"), data.get("frequency_hz")
    if isinstance(amp, (int, float)) and not isinstance(amp, bool):
        out["mod_amp_v"] = float(amp)
    if isinstance(freq, (int, float)) and not isinstance(freq, bool):
        out["mod_freq_hz"] = float(freq)
    mod_on = data.get("mod_on")
    if isinstance(mod_on, bool):
        out["mod_on"] = mod_on
    return out


def sample_x(ctx, *, n: int = 5, interval_s: float = 0.15
             ) -> "tuple[float, float, int] | None":
    """窗口内平均的 X 读数 ``(mean_a, sd_a, n)``;读不到返回 ``None``。

    取 RT 信号(``Signals_ValsGet``),索引来自用户填的 ``lockin_x_signal_index``
    —— 与 AutoPhase 同一个来源,同一条「不猜」的纪律。
    """
    idx = ip.get_config("lockin_x_signal_index", None)
    if idx in (None, ""):
        return None
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    vals: "list[float]" = []
    for i in range(max(1, int(n))):
        try:
            rec = ctx.safe_call("Signals_ValsGet", [idx], 0)
        except Exception:  # noqa: BLE001
            break
        if getattr(rec, "error", ""):
            break
        v = _first_value(getattr(rec, "return_value", None))
        if v is not None:
            vals.append(v)
        if i + 1 < n and interval_s > 0:
            time.sleep(interval_s)
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    sd = 0.0
    if len(vals) > 1:
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return mean, sd, len(vals)


def _first_value(rv) -> "float | None":
    """``Signals_ValsGet`` 回包里的第一个浮点。两种回包形态都接(§2.21/§2.31)。"""
    if not isinstance(rv, (list, tuple)) or len(rv) <= 2:
        return None
    for field in rv[2]:
        if isinstance(field, (list, tuple)) and field:
            item = field[0]
            if isinstance(item, (list, tuple)) and item:   # 1-元组形态
                item = item[0]
            try:
                return float(item)
            except (TypeError, ValueError):
                return None
    return None


def steps_band(curve: "dict[str, Any]", value: float, sd: float) -> "float | None":
    """读数散布 ``sd`` 折算成步数的不确定度;算不出返回 ``None``。

    用**局部斜率**:曲线平的地方,同样的 sd 对应的步数band 大得多 —— 而那正是这条
    曲线末段的情形。算不出斜率(落在覆盖范围外 / 平段)时返回 None,由调用方说
    「给不出误差带」,不是给一个 0。
    """
    if sd is None or sd <= 0:
        return None
    vk, dk = cc.curve_keys(curve)
    pairs = sorted(((float(p[vk]), float(p[dk])) for p in curve.get("points", [])
                    if isinstance(p.get(vk), (int, float))
                    and isinstance(p.get(dk), (int, float))),
                   key=lambda t: t[0])
    if len(pairs) < 2:
        return None
    for (v0, d0), (v1, d1) in zip(pairs, pairs[1:]):
        if v0 <= value <= v1:
            if v1 == v0:
                return None
            slope = abs((d1 - d0) / (v1 - v0))     # 步 / 安培
            return slope * sd
    return None


def _round_steps(x: float) -> int:
    g = STEP_GRANULARITY
    return int(round(x / g) * g)


def position_report(value_a: float, *, sd_a: float = 0.0,
                    now: "dict[str, Any] | None" = None,
                    curves: "list[dict[str, Any]] | None" = None
                    ) -> "dict[str, Any]":
    """把一个 X 读数对照曲线库,给出「大约在曲线的哪个位置」。

    **永远带着原始读数**:翻译不出来的时候,读数本身仍然是用户要的东西。
    """
    out: "dict[str, Any]" = {"crosstalk_x_a": value_a, "crosstalk_sd_a": sd_a,
                             "crosstalk_steps": None, "crosstalk_steps_band": None}
    lib = cc.load_curves(cc.KIND_RETRACT_LOCKIN) if curves is None else list(curves)
    if not lib:
        out["crosstalk_note"] = (
            f"X = {value_a:.4g} A;曲线库里没有退针参考曲线,**未翻译**。"
            "(采一条:退针过程中每批记 X,存进标定目录。)")
        return out

    reasons: "list[str]" = []
    for curve in lib:
        steps, why = cc.translate(curve, value_a, now=dict(now or {}))
        if steps is None:
            reasons.append(why)
            continue
        band = steps_band(curve, value_a, sd_a)
        out["crosstalk_steps"] = _round_steps(steps)
        out["crosstalk_steps_band"] = None if band is None else _round_steps(band)
        out["crosstalk_curve"] = cc.describe(curve)
        band_txt = (f" ± {_round_steps(band)}" if band is not None
                    else "(给不出误差带:曲线在此处太平或算不出斜率)")
        out["crosstalk_note"] = (
            f"X = {value_a:.4g} A ≈ 参考曲线上的 **{_round_steps(steps)}"
            f"{band_txt} 步**(相对隧穿点的退针步数)。"
            f"依据:{cc.describe(curve)} **仅供参考,不驱动任何动作。**")
        for caveat in (curve.get("caveats") or [])[:3]:
            out.setdefault("crosstalk_caveats", []).append(caveat)
        return out

    # 一条都没匹配上 —— 把**为什么**说全,连两边的值。否则「换了针没重采曲线」
    # 和「功能坏了」长得一模一样。
    out["crosstalk_note"] = (
        f"X = {value_a:.4g} A;**未翻译成步数** —— " + ";".join(reasons[:3]))
    return out


def crosstalk_report(ctx, *, n: int = 5, interval_s: float = 0.15
                     ) -> "dict[str, Any]":
    """一次调用拿到「此刻的串扰读数 + 它在参考曲线上的位置」。

    调制没开就**不读**(那时 X 上没有信号,一个噪声读数翻译出来的距离是编的)。
    永不抛异常:这是报告,不是任何流程的目的。
    """
    try:
        now = live_conditions(ctx)
        if now.get("mod_on") is not True:
            state = {True: "开", False: "关", None: "读不到"}[now.get("mod_on")]
            return {"crosstalk_modulation_off": True,
                    "crosstalk_skipped": (
                        f"调制{state} —— 不读 X,也就不翻译。"
                        "⚠️ 进针/退针流程**开跑前会自动关调制**(用完即关,`d097541`),"
                        "所以默认情况下这条报告在这些流程里永远是这一句 —— "
                        "**这不是故障,是两个机制的取舍**:调制开着时电流通道上那层"
                        "纹波会污染进退针的形态判据,"
                        "而串扰导航恰恰需要它开着。要用导航就得为这些流程保留调制,"
                        "那是一次要单独拍板的取舍(见 KNOWN_ISSUES §2.36)。")}
        got = sample_x(ctx, n=n, interval_s=interval_s)
        if got is None:
            return {"crosstalk_skipped": (
                "读不到 lock-in X(索引未配置或读取失败)—— **不猜**索引:"
                "猜错读到的是另一路信号,而算出来的距离看上去一样合理。"
                "请在仪器档案里填 lockin_x_signal_index。")}
        mean, sd, count = got
        if abs(mean) < MIN_SIGNAL_A:
            return {"crosstalk_skipped": (
                f"X = {mean:.3g} A 在噪声底以下(下界 {MIN_SIGNAL_A:.1g})—— 不翻译。")}
        rep = position_report(mean, sd_a=sd, now=now)
        rep["crosstalk_samples"] = count
        return rep
    except Exception as exc:  # noqa: BLE001 — 报告失败不该带走流程
        logger.debug("串扰报告失败(跳过):%s", exc, exc_info=True)
        return {"crosstalk_skipped": f"串扰报告出错(已跳过):{exc}"}


__all__ = ["MIN_SIGNAL_A", "STEP_GRANULARITY", "live_tip_type", "live_conditions",
           "sample_x", "steps_band", "position_report", "crosstalk_report"]
