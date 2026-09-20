# -*- coding: utf-8 -*-
"""两帧之间的横向位移：亚像素、带米制、带**置信度**，并把上下扫回程差分出来。

## 与既有三处的分工

| 位置 | 输入 | 精度 | 失败时 |
|---|---|---|---|
| ``ComputeDriftVector`` / ``TrackDrift_ReferenceScan`` | 实时抓帧 + 一张 ``.npy`` | **整像素** | 静默 ``(0.0, 0.0)`` |
| ``DiffScans_ChangeDetect`` | 两个文件 | 整像素配准 | 出差图，不出位移矢量 |
| ``data.processors.drift_estimate`` | 两个数组 | 亚像素（相位相关 ×10） | 抛异常 |
| **本模块** | 两个数组 + 标度 | 亚像素 | ``ok=False`` + 原因 |

位移本身**直接复用** ``data.processors.drift_estimate``（skimage 的
``phase_cross_correlation``，已是亚像素）—— 不再写第四份互相关。本模块加的是
三件它没有的东西：

1. **米制换算与置信度**。整像素那两处最要命的地方不是精度，是
   「测不出来」被表达成「没有漂移」（``_compute_drift`` 的每一条异常路径都
   ``return 0.0, 0.0``）。这里判不了就说判不了。
2. **上下扫回程差的分离**。全仓没有任何位移函数处理 ``:SCAN_DIR:``：
   相邻两帧一上一下时，测到的 Δy 里装着「回程差」（正负交替的常数）与
   「净漂移」（同号累积）之和。两帧滑动平均消掉前者，差的一半就是它。

## 为什么这里必须用亚像素法（一次方法对照，值得记）

2026-09-03 同一批 30 nm 帧，同样先低通，两种取峰方式：

    对           整像素 FFT 峰 dy(px)     相位相关(×10) dy(px)
    0114->0115            0                    +0.0
    0115->0116          108                    −5.9
    0116->0117           −9                    −5.6
    0117->0118          230                    +7.4
    0118->0119         −104                    −0.2

整像素峰在这类帧上会跳到**伪极大**（108、230 px 都超过半帧）。当时的分析脚本
靠「相关 ≥0.25 且 |位移| <2 nm」把那些值筛掉，于是活下来的都是小值，
得出「回程差 0.12 nm/帧」——**那个数是筛选的产物，不是测量**。
用同一批帧、同一低通、改用亚像素相位相关之后，回程差与净漂移都落到 0.01 nm 量级。

⇒ 教训不是「整像素不够精」，是**先筛后统计会把方法的失败伪装成一个小而可信的数**。
所以本模块的置信度门是**返回 ``ok=False``**，而不是让调用方去筛。
3. **平坦帧的拒答**。相位相关在两张没有共同特征的图上照样给出一个峰。

## 什么时候**不要**用这条路（真机上被这批数据教的）

同一批 30 nm 帧上，本模块与它的所有前身都测不出可信的漂移：
换六种低通尺度（不低通到 σ=1 nm），同一批 9 对帧给出的 Δy 在 +3.6 / −3.2 /
−0.5 / −2.1 / +7.5 nm 之间跳，而且有三对的归一化互相关是**负的**
（「对齐」之后反相关）。原因不是算法，是**那批帧上没有可配准的东西**：
表面是同一个台面、非周期内容只有几个会动的吸附物、而针尖状态帧帧在变。

同一批帧上，**从晶格量漂移**是稳的：上下扫的 a₂ 应变给出 0.3-0.6 nm/h
（:func:`mast.vision.lattice_cell.combine_up_down`）。

⇒ 选路的规则：**有非周期特征（台阶、团簇、缺陷）就配准，只有晶格就量应变。**
本模块负责在前一种情况不成立时**说不知道**，而不是给一个数。

## 谁来做低通

传进来的图应当**已经去过趋势**（``scan_prep.poly_subtract`` 或
``tip_metrics._detrend``）。本模块不替调用方选去趋势方式：不同的判据要的
去趋势不同（条纹判据要保留行偏置，配准判据不要），在这里替它们决定会重演
``streak_amplitude_pm`` 那一处的冲突。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["PairDisplacement", "pair_displacement", "separate_hysteresis"]

#: 归一化互相关峰低于它就不给位移 —— 两张图没有共同的特征。
_MIN_CONFIDENCE = 0.25

#: 位移超过帧宽的这个比例时不给结论：相位相关的峰会绕回（wrap），
#: 一个「半帧」的位移与「负半帧」不可分。
_MAX_SHIFT_FRAC = 0.25


@dataclass(frozen=True)
class PairDisplacement:
    """一对帧之间的位移。``ok=False`` 时 ``reason`` 说明为什么量不了。"""

    ok: bool = False
    reason: str = ""
    dx_nm: Optional[float] = None
    dy_nm: Optional[float] = None
    #: 归一化互相关峰值 ∈ [0, 1]。低 ⇒ 两帧没有共同特征，位移无意义。
    confidence: Optional[float] = None
    dx_px: Optional[float] = None
    dy_px: Optional[float] = None
    warnings: tuple[str, ...] = ()


def _ncc_peak(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
              dy: float, dx: float) -> float:
    """在给定整数位移附近的归一化互相关值 —— 相位相关自己不给这个数。"""
    iy, ix = int(round(dy)), int(round(dx))
    ny, nx = a.shape
    ys = slice(max(0, iy), min(ny, ny + iy))
    xs = slice(max(0, ix), min(nx, nx + ix))
    ys2 = slice(max(0, -iy), min(ny, ny - iy))
    xs2 = slice(max(0, -ix), min(nx, nx - ix))
    pa = a[ys, xs]
    pb = b[ys2, xs2]
    if pa.size < 64 or pa.shape != pb.shape:
        return 0.0
    pa = pa - pa.mean()
    pb = pb - pb.mean()
    den = math.sqrt(float(np.sum(pa * pa)) * float(np.sum(pb * pb)))
    return float(np.sum(pa * pb) / den) if den > 0 else 0.0


def _ncc_shift(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
               limit_px: float) -> tuple[float, float, float]:
    """把 ``b`` 配准到 ``a`` 上要施加的位移 ``(dy_px, dx_px)``，以及那里的归一化互相关。

    在 ``±limit_px`` 的整数位移范围内，对**重叠区**做归一化互相关（零填充，不绕回），
    取峰后沿两轴各做一次抛物线亚像素插值。约定与 skimage 的
    ``phase_cross_correlation`` 及 :func:`_ncc_peak` 相同：``b`` 平移 ``(dy, dx)`` 后与
    ``a`` 对齐，特征本身从 ``a`` 到 ``b`` 挪动了 ``(-dy, -dx)``。
    """
    from scipy.signal import correlate

    ny, nx = a.shape
    lim = max(1, int(limit_px))
    a0 = a - float(a.mean())
    b0 = b - float(b.mean())
    ones = np.ones_like(a0)
    # correlate(x, y)[k] 的滞后 τ = k − (N − 1)：Σ_l x[l]·y[l − τ]，峰在 τ 处 ⇒ y 平移 τ 与 x 对齐
    cross = correlate(a0, b0, mode="full", method="fft")
    e_a = correlate(a0 * a0, ones, mode="full", method="fft")
    e_b = correlate(ones, b0 * b0, mode="full", method="fft")
    cy, cx = ny - 1, nx - 1
    win = (slice(cy - lim, cy + lim + 1), slice(cx - lim, cx + lim + 1))
    den = np.sqrt(np.clip(e_a[win], 0.0, None) * np.clip(e_b[win], 0.0, None))
    ncc = np.where(den > 0, cross[win] / np.where(den > 0, den, 1.0), 0.0)
    k = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    peak = float(ncc[k])
    # 反相关的对齐（衬度反转的两帧）：最强的「对齐」是负的。报那个位移和它的负相关，
    # 让调用方按 ``conf < 0`` 拒答，而不是在一片弱正相关里挑一个假峰。
    k_min = np.unravel_index(int(np.argmin(ncc)), ncc.shape)
    if -float(ncc[k_min]) > peak:
        k, peak = k_min, float(ncc[k_min])
    dy, dx = float(k[0] - lim), float(k[1] - lim)
    # 峰要**尖**：归一化互相关对一对几乎没有结构的图（重低通后的晶格帧）处处接近 1，
    # argmax 落在哪里都是噪声决定的。用峰与几个像素外的环的差衡量峰的锐度，
    # 调用方据此拒答（reason ``ambiguous``）。
    ring_r = max(2, int(round(0.04 * min(ny, nx))))
    ys, xs = np.mgrid[0:ncc.shape[0], 0:ncc.shape[1]]
    ring = (np.abs(ys - k[0]) == ring_r) | (np.abs(xs - k[1]) == ring_r)
    ring &= (np.abs(ys - k[0]) <= ring_r) & (np.abs(xs - k[1]) <= ring_r)
    sharpness = float(peak - float(ncc[ring].mean())) if ring.any() else 0.0

    def _sub(axis: int) -> float:
        idx = list(k)
        if not (0 < k[axis] < ncc.shape[axis] - 1):
            return 0.0
        idx[axis] = k[axis] - 1
        c_m = float(ncc[tuple(idx)])
        idx[axis] = k[axis] + 1
        c_p = float(ncc[tuple(idx)])
        curv = c_m - 2.0 * peak + c_p
        if curv >= 0:
            return 0.0
        off = 0.5 * (c_m - c_p) / curv
        return float(max(-0.5, min(0.5, off)))

    return dy + _sub(0), dx + _sub(1), peak, sharpness


#: 峰至少要比几个像素外的环高这么多，否则相关面是平的，位移由噪声决定
_MIN_SHARPNESS = 0.05

# 配准前使用低通抑制周期晶格，以突出缺陷、团簇和台阶等非周期内容。
# 周期结构会在多个格矢平移处产生相关峰；帧间晶格相位变化也会影响峰值。
# 低通尺度为纳米量，须结合输入分辨率与待配准结构选择。
# scan_artifacts._drift_px 另以偏移峰相对零延迟峰的强度约束周期结构的假位移。
_SMOOTH_NM = 0.5


def pair_displacement(first: npt.ArrayLike, second: npt.ArrayLike, *,
                      nm_per_px: float,
                      min_confidence: float = _MIN_CONFIDENCE,
                      smooth_nm: Optional[float] = _SMOOTH_NM) -> PairDisplacement:
    """``second`` 相对 ``first`` 的位移，纳米。两帧必须同尺寸、同标度、同视野。

    ``smooth_nm`` 是配准前的低通尺度，见 :data:`_SMOOTH_NM` —— 传 ``None``
    可以关掉，但那样量的是「晶格对没对上」而不是「样品挪了多远」。

    纯函数：不读文件、不碰硬件、不抛异常。
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        return PairDisplacement(reason="need_2d_images")
    if a.shape != b.shape:
        return PairDisplacement(reason="shape_mismatch")
    if min(a.shape) < 32:
        return PairDisplacement(reason="image_too_small")
    if not (nm_per_px and nm_per_px > 0):
        return PairDisplacement(reason="unknown_pixel_size")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return PairDisplacement(
            reason="non_finite",
            warnings=("图里有 NaN/Inf —— 未扫完的帧要先裁掉那些行"
                      "（``scan_prep.acquired_row_span``）。",))
    if a.std() <= 0 or b.std() <= 0:
        return PairDisplacement(
            reason="flat_frame",
            warnings=("有一帧是死平的 —— 相位相关在没有共同特征的两张图上"
                      "**照样会给出一个峰**，那个数没有意义。",))

    reg_a, reg_b = a, b
    if smooth_nm and smooth_nm > 0:
        try:
            from scipy.ndimage import gaussian_filter
            sig = float(smooth_nm) / float(nm_per_px)
            if sig >= 0.5:
                reg_a = gaussian_filter(a, sig)
                reg_b = gaussian_filter(b, sig)
        except ImportError:  # pragma: no cover — scipy 是硬依赖，这里只是不崩
            pass
    limit = _MAX_SHIFT_FRAC * min(a.shape)
    try:
        # 有界、零填充的归一化互相关，**不是**循环的相位相关（2026-09-13 改）。
        #
        # 之前用 skimage 的 ``phase_cross_correlation(normalization=None)``。它做的是
        # FFT 循环相关：位移超出的那部分从帧的另一边绕回来。STM 帧沿慢轴不是周期的
        # （行与行之间有蠕变、行偏置、起扫瞬态），绕回的那一条带把行方向的真峰压掉，
        # 而快轴方向左右两端接得上，不受影响。模拟器上实测：扫描框沿 x 挪 3 nm 量到
        # 2.9 nm；沿 y 挪 3 nm 量到 −0.06 nm，而按重叠区做的归一化互相关在 +9.6 px
        # 处给出 0.997 的峰。加 Hann 窗、二阶去趋势都救不回来，只有不绕回才行。
        #
        # 约定不变：返回的是把 ``second`` 配准到 ``first`` 上要施加的位移
        # （与 skimage 和 :func:`_ncc_peak` 同向），**特征本身挪动的量是它的相反数**。
        dy_px, dx_px, peak, sharpness = _ncc_shift(reg_a, reg_b, limit)
    except Exception as exc:  # noqa: BLE001 — 纯函数不抛
        return PairDisplacement(reason="correlation_failed",
                                warnings=("互相关失败：%s" % exc,))
    if peak >= 0 and sharpness < _MIN_SHARPNESS:
        return PairDisplacement(
            reason="ambiguous", dx_px=dx_px, dy_px=dy_px, confidence=peak,
            warnings=("相关面是平的（峰只比 %d px 外高 %.3f）—— 两帧在低通之后几乎没有"
                      "可配准的结构，位移由噪声决定。**这不是「没有漂移」**：晶格主导的帧"
                      "该从晶格量应变（MeasureLatticeCell）。"
                      % (max(2, int(round(0.04 * min(a.shape)))), sharpness),))

    # 置信度在**低通之后**的图上算 —— 与定位用的是同一份内容，
    # 否则报的是「晶格对没对上」，而位移根本不是靠晶格定出来的。
    conf = _ncc_peak(reg_a, reg_b, dy_px, dx_px)
    warns: list[str] = []
    if abs(dy_px) >= limit or abs(dx_px) >= limit:
        return PairDisplacement(
            reason="shift_too_large", dx_px=dx_px, dy_px=dy_px, confidence=conf,
            warnings=("位移 (%.1f, %.1f) px 超过帧的 %.0f%% —— 相位相关的峰会绕回，"
                      "这时「大位移」与「反向的大位移」不可分。要跟这么大的漂移，"
                      "得缩短帧间隔或扩大视野。"
                      % (dx_px, dy_px, 100 * _MAX_SHIFT_FRAC),))
    if conf < 0:
        return PairDisplacement(
            reason="anti_correlated", dx_px=dx_px, dy_px=dy_px, confidence=conf,
            warnings=("在找到的位移上两帧是**反**相关的（%.2f）—— 那不是一个对齐，"
                      "是相位相关在没有共同特征的两张图上凑出来的峰。"
                      "常见于「同一个台面 + 会动的吸附物 + 针尖在变」的帧。" % conf,))
    if conf < float(min_confidence):
        return PairDisplacement(
            reason="low_confidence", dx_px=dx_px, dy_px=dy_px, confidence=conf,
            warnings=("归一化互相关只有 %.2f（下限 %.2f）—— 两帧没有足够的共同特征。"
                      "**这不是「没有漂移」**：位移量不出来和位移为零是两件事，"
                      "上游那两个整像素实现把它们折叠成了同一个 0.0。"
                      % (conf, float(min_confidence)),))
    return PairDisplacement(
        ok=True, dx_nm=float(dx_px) * float(nm_per_px),
        dy_nm=float(dy_px) * float(nm_per_px),
        dx_px=float(dx_px), dy_px=float(dy_px), confidence=conf,
        warnings=tuple(warns))


def separate_hysteresis(dy_nm: Sequence[Optional[float]]) -> dict:
    """把交替扫描方向序列里的**回程差**与**净漂移**分开。

    相邻帧一上一下时，每一对的 Δy 都是 ``净漂移 ± 回程差``，符号随扫描方向交替。
    两两滑动平均消掉交替项，剩下的是净漂移；两者之差的一半就是回程差。

    ``dy_nm`` 里允许有 ``None``（那一对没量出来）—— 它们**不参与**平均，
    而不是当成 0。这正是上游那两个实现折叠掉的区别。
    """
    vals = list(dy_nm)
    net: list[float] = []
    for i in range(len(vals) - 1):
        a, b = vals[i], vals[i + 1]
        if a is None or b is None:
            continue
        net.append(0.5 * (float(a) + float(b)))
    known = [float(v) for v in vals if v is not None]
    out: dict = {
        "n_pairs": len(known),
        "n_unmeasured": sum(1 for v in vals if v is None),
        "net_drift_per_frame_nm": float(np.median(net)) if net else None,
        "hysteresis_nm": (float(np.median(np.abs(known))) if known else None),
    }
    if net and known:
        out["note"] = (
            "净漂移 %.3f nm/帧、回程差中位 %.3f nm/帧。两者混在一起时，"
            "一个纯往复的回程差会被读成正负交替的「漂移」——"
            "取两帧滑动平均正是为了把那一项消掉。"
            % (out["net_drift_per_frame_nm"], out["hysteresis_nm"]))
    return out
