# -*- coding: utf-8 -*-
"""扫描帧的 Current 通道用于估计 Z 反馈跟随误差。

恒流扫描中 Z 记录形貌，电流记录反馈误差，两者服务于不同的分析问题。
log 模式下 dz/dt = I_gain · ln(I/I₀)，结合地形跟随速率 v · dZ/dx，
在小误差近似下有 ΔI/I₀ ≈ (v / I_gain) · dZ/dx。

将电流图拟合为 Z 图快轴梯度的响应，得到 lag_ratio = slope / I₀。
它可与文件头中的 v / I_gain 比较，用来检查模型与读数的一致性。

若调用方提供隧穿衰减常数 κ 与空间周期 λ，还可计算
ωτ = lag_ratio · π / (κ · λ) 及幅值保真度 1/√(1+(ωτ)²)。
κ 没有默认值；未提供时只报告 lag_ratio，fidelity 保持 None。

该测量不判断针尖质量，也不决定增益设置。单帧不能完全区分滞后与电流噪声，
因此另行报告扣除拟合滞后项后的 residual_rms_a。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "FeedbackLagResult",
    "measure_feedback_lag",
    "expected_lag_ratio",
    "tracking_fidelity",
]

#: 拟合需要地形本身有起伏。快轴梯度的 RMS 低于它 ⇒ 分母接近零，
#: 斜率是 0/0。单位 m/m（无量纲）。1e-5 = 每 nm 起伏 0.01 pm，
#: 比任何真实表面都平，只有死平帧会落到这里。
_MIN_SLOPE_RMS = 1e-5

#: 拟合优度下限。低于它说明电流里主导的不是滞后项（可能是针尖闪变、
#: 50 Hz、或者根本没在恒流模式），斜率不可信。
_MIN_R2 = 0.25


@dataclass(frozen=True)
class FeedbackLagResult:
    """一帧上的 Z 反馈跟随误差。``ok=False`` 时 ``reason`` 说明为什么量不了。"""

    ok: bool = False
    reason: str = ""
    #: ``ΔI = slope · dZ/dx`` 的斜率，单位安培（dZ/dx 无量纲）。
    slope_a: Optional[float] = None
    #: ``slope / setpoint``。log 模式下它**等于** ``v / I_gain``。
    lag_ratio: Optional[float] = None
    #: 拟合优度。低了说明电流里主导的不是滞后项。
    r2: Optional[float] = None
    #: 电流与 dZ/dx 的相关系数。
    #:
    #: ⚠️ 它的**符号跟着电流极性走**，不是一个固定值：负偏压下电流为负，相关也为负；
    #: 正偏压下两者都翻过来，所以不能要求相关系数固定为负。
    #: 与极性无关的不变量是 ``slope / I₀ > 0``：针尖落在上坡后面 ⇒ 间隙变小 ⇒
    #: 电流**幅值**变大，无论 I₀ 是正是负。它在 :attr:`sign_consistent` 里。
    correlation: Optional[float] = None
    #: ``slope / 平均电流 > 0``（两者**同号**）是否成立 —— 与偏压极性无关的那条。
    #: ``False`` = 通道配错或定向没做（未定向的反扫电流配上正扫 Z，镜像会翻号）。
    sign_consistent: Optional[bool] = None
    #: 电流图自己的均值（安培，**带符号**）。文件头里的设定点是无符号的幅值，
    #: 定不了符号，所以极性只能从图上取。
    mean_current_a: Optional[float] = None
    #: 扣掉滞后项之后剩下的电流起伏 —— 与反馈快慢无关的那一半噪声。
    residual_rms_a: Optional[float] = None
    #: 原始电流起伏（未扣滞后项），用来和上面一项比。
    current_rms_a: Optional[float] = None
    #: 文件头算出来的 ``v / I_gain``。与 ``lag_ratio`` 对不上是**自检失败**。
    expected_lag_ratio: Optional[float] = None
    #: 两者的相对差。``None`` = 文件头缺料，这道自检没做。
    ratio_disagreement: Optional[float] = None
    #: 在 ``lattice_period_nm`` 上未跟上的正交分量 ωτ。需要 κ。
    omega_tau: Optional[float] = None
    #: 幅值保真度 ``1/√(1+(ωτ)²)`` —— 图上的起伏是真实起伏的几成。需要 κ。
    fidelity: Optional[float] = None
    warnings: tuple[str, ...] = ()


def _detrend_rows(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """逐**行**减一次多项式。形状显式，方阵上也不会悄悄转置。"""
    h, w = arr.shape
    x = np.arange(w, dtype=np.float64)
    X = np.vstack([x, np.ones(w)]).T                  # (w, 2)
    coef, *_ = np.linalg.lstsq(X, arr.T, rcond=None)  # (2, h)
    return arr - (X @ coef).T                         # (h, w)


def expected_lag_ratio(speed_m_s: Optional[float],
                       i_gain_m_s: Optional[float]) -> Optional[float]:
    """文件头那一路的 ``v / I_gain``。缺任一项就返回 ``None``（不猜）。"""
    if not speed_m_s or not i_gain_m_s:
        return None
    if speed_m_s <= 0 or i_gain_m_s <= 0:
        return None
    return float(speed_m_s) / float(i_gain_m_s)


def tracking_fidelity(lag_ratio: Optional[float], period_nm: Optional[float],
                      kappa_per_nm: Optional[float]) -> tuple[Optional[float],
                                                              Optional[float]]:
    """``(ωτ, 幅值保真度)``。三个入参缺一就都返回 ``None``。

    ``ωτ = lag_ratio · π / (κ·λ)``，见模块注释的推导。κ 只有
    ``MeasureBarrierHeight`` 给得出，所以这里**不设默认值**。
    """
    if lag_ratio is None or not period_nm or not kappa_per_nm:
        return None, None
    if period_nm <= 0 or kappa_per_nm <= 0:
        return None, None
    wt = float(lag_ratio) * math.pi / (float(kappa_per_nm) * float(period_nm))
    return wt, float(1.0 / math.sqrt(1.0 + wt * wt))


def measure_feedback_lag(
    z_image: npt.ArrayLike,
    current_image: npt.ArrayLike,
    *,
    nm_per_px: float,
    setpoint_a: Optional[float] = None,
    speed_m_s: Optional[float] = None,
    i_gain_m_s: Optional[float] = None,
    lattice_period_nm: Optional[float] = None,
    kappa_per_nm: Optional[float] = None,
) -> FeedbackLagResult:
    """把电流图拟合成 Z 图快轴梯度的线性响应。

    两张图必须是**同一帧的同一扫描方向**，且都已经过
    :func:`mast.io.nanonis_files.sxm_oriented_frames` 定向 —— 反扫是镜像存储的，
    拿未定向的反扫电流去配正扫 Z，斜率的符号就是假的。

    纯函数：不读文件、不碰硬件、不抛异常。
    """
    z = np.asarray(z_image, dtype=np.float64)
    cur = np.asarray(current_image, dtype=np.float64)
    warns: list[str] = []

    if z.ndim != 2 or cur.ndim != 2:
        return FeedbackLagResult(reason="need_2d_images")
    if z.shape != cur.shape:
        return FeedbackLagResult(reason="shape_mismatch")
    if min(z.shape) < 16:
        return FeedbackLagResult(reason="image_too_small")
    if not (nm_per_px and nm_per_px > 0):
        return FeedbackLagResult(reason="unknown_pixel_size")

    good = np.isfinite(z) & np.isfinite(cur)
    rows = good.all(axis=1)
    if rows.sum() < 8:
        return FeedbackLagResult(
            reason="too_few_complete_rows",
            warnings=("只有 %d 行两个通道都完整 —— 未扫完的帧量不了跟随误差"
                      % int(rows.sum()),))
    z = z[rows]
    cur = cur[rows]

    # 逐行去趋势：慢轴的漂移与倾斜与反馈跟随无关，留着它们会给分子分母
    # 同时加上一个共同的低频项，把相关系数抬向 1。
    #
    # ⚠️ 显式写成最小二乘、不用 ``polyfit(x, arr.T, 1)`` 那种写法：方阵上后者的
    # 转置错误可能不报错却去掉逐列趋势，改变梯度相关性。
    zc = _detrend_rows(z)
    cc = _detrend_rows(cur)

    # dZ/dx **无量纲**（米每米），所以斜率的单位就是安培，
    # 而 slope/setpoint 直接可与 v/I_gain 比。
    dz = np.gradient(zc, float(nm_per_px) * 1e-9, axis=1)

    denom = float(np.sum(dz * dz))
    slope_rms = float(np.sqrt(np.mean(dz * dz)))
    if denom <= 0 or slope_rms < _MIN_SLOPE_RMS:
        return FeedbackLagResult(
            reason="frame_too_flat",
            warnings=("快轴梯度 RMS 只有 %.2e（下限 %.0e）—— 地形太平，"
                      "斜率是 0/0。这不是「反馈很好」，是**量不了**。"
                      % (slope_rms, _MIN_SLOPE_RMS),))

    slope = float(np.sum(cc * dz) / denom)
    resid = cc - slope * dz
    ss_tot = float(np.sum(cc * cc))
    r2 = float(1.0 - np.sum(resid * resid) / ss_tot) if ss_tot > 0 else 0.0
    denom_corr = math.sqrt(float(np.sum(cc * cc)) * denom)
    corr = float(np.sum(cc * dz) / denom_corr) if denom_corr > 0 else 0.0

    if r2 < _MIN_R2:
        warns.append(
            "拟合优度只有 %.2f（下限 %.2f）—— 电流里主导的不是滞后项。"
            "常见来源：针尖闪变、工频、或者这一帧根本不在恒流模式。"
            "斜率照报，但不要拿它去推增益。" % (r2, _MIN_R2))

    # 极性从电流图自身取得：文件头设定点为无符号幅值，不能据此判断电流符号。
    i_mean = float(np.mean(cur))
    i0 = abs(i_mean)
    if i0 <= 0 and setpoint_a:
        i0 = abs(float(setpoint_a))
        warns.append("电流图均值为零，改用文件头的设定点定标 —— 符号一致性这一关做不了。")
    lag_ratio = (abs(slope) / i0) if i0 > 0 else None

    sign_ok: Optional[bool] = None
    if i_mean != 0.0:
        sign_ok = bool((slope / i_mean) > 0)
        if not sign_ok:
            warns.append(
                "斜率与平均电流**异号**（slope %.3e A，平均电流 %.3e A）—— "
                "与极性无关的不变量是两者**同号**：针尖落在上坡后面 ⇒ 间隙变小 ⇒ "
                "电流**幅值**变大，所以 I₀ 为负时电流更负、斜率也为负。"
                "异号多半是把未定向的反扫电流配上了正扫 Z（镜像会把快轴梯度翻号），"
                "或者通道选错了。" % (slope, i_mean))

    exp_ratio = expected_lag_ratio(speed_m_s, i_gain_m_s)
    disagree = None
    if lag_ratio is not None and exp_ratio:
        disagree = float(abs(lag_ratio - exp_ratio) / exp_ratio)
        if disagree > 0.30:
            warns.append(
                "实测 lag_ratio %.3f 与文件头算出的 v/I_gain %.3f 差 %.0f%% —— "
                "这是一道**自检**，差这么多说明有一边不对（线速度或增益读错、"
                "不是 log 模式、或者帧里主导的不是滞后项），不是「反馈变差了」。"
                % (lag_ratio, exp_ratio, 100 * disagree))

    wt, fid = tracking_fidelity(lag_ratio, lattice_period_nm, kappa_per_nm)
    if lattice_period_nm and not kappa_per_nm:
        warns.append(
            "没给 kappa_per_nm，跟随保真度算不了 —— 它需要隧穿衰减常数，"
            "而那个数只有 MeasureBarrierHeight 量得出（κ = √φ·5.123，φ 单位 eV）。"
            "不给默认值是刻意的：默认值会让「没测过势垒」这件事看不见。")

    return FeedbackLagResult(
        ok=True,
        slope_a=slope,
        lag_ratio=lag_ratio,
        r2=r2,
        correlation=corr,
        sign_consistent=sign_ok,
        mean_current_a=i_mean,
        residual_rms_a=float(np.sqrt(np.mean(resid * resid))),
        current_rms_a=float(np.sqrt(np.mean(cc * cc))),
        expected_lag_ratio=exp_ratio,
        ratio_disagreement=disagree,
        omega_tau=wt,
        fidelity=fid,
        warnings=tuple(warns),
    )
