# -*- coding: utf-8 -*-
"""用统一幅值口径描述晶格、逐行短划和分块晶格可辨度。

整体角向集中度回答是否存在周期结构；方向分辨的幅值和网格图则描述
信号强度、方向不均衡及其空间位置。条纹与晶格带通使用相同的 pm 单位，
可比较同块同窗下的相对强度。

环带采用 0.65·f0 < fr < 1.35·f0，与 herringbone.bandpass 共享口径。
正弦等效峰峰值为 2√2 × rms；角度从快扫轴向行号增大方向测量，mod 180。

每块必须容纳 _MIN_PERIODS_PER_TILE 个周期，不足时不生成块图。
块级判断采用同块内部的晶格带功率/其余结构功率比，避免直接比较
不同尺寸的角向集中度。good_ratio 为显式参数，须针对目标成像条件验证。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "DirectionAmplitude",
    "TileLatticeMap",
    "ring_mask",
    "directional_bandpass",
    "lattice_amplitude_pm",
    "streak_amplitude_pm",
    "tile_lattice_map",
]

#: 一块里至少要装下这么多个晶格周期。``_check_halves`` 用的是 ≈8
#: （2.0 nm / 0.25 nm），这里沿用同一个数量级。
_MIN_PERIODS_PER_TILE = 8.0

#: 条纹（逐行横向短划）在快轴上的相干范围：|fx| 小于这么多个频率格。
#: 1.5 格 = 特征沿快轴延展超过帧宽的 2/3，正是「一整行被抬起来」的样子。
_STREAK_FX_BINS = 1.5

#: 条纹在慢轴上的周期带，单位**行数**（不是纳米）。
#:
#: 用行数是因为这个伪影是**逐行采集**的产物，与样品的长度尺度无关：一次针尖闪变
#: 影响的是「接下来几行」。同一个 16 行的窗在 5 nm 帧上是 0.16 nm、在 30 nm 帧上是
#: 0.94 nm —— 用纳米定带会让同一台仪器的同一种伪影在不同视野下量到不同的东西。
#:
#: 上界限制长周期地形混入短划统计；下界是 Nyquist（2 行）。
#: max_rows 应按目标采集条件验证，不能把地形起伏直接解释为针尖状态。
_STREAK_ROWS = 16.0

#: 「其余结构」的周期带（nm）—— 块级比值的分母。它比晶格带宽得多，
#: 覆盖条纹、点缺陷、团簇边缘，但不含 DC 与整块倾斜。
_OTHER_PERIOD_NM = (0.15, 2.5)


@dataclass(frozen=True)
class DirectionAmplitude:
    """一个晶格方向上的幅值，**两个口径一起给**。

    两个数差得远是常态，而且那个差本身有意义（见 :attr:`coherence`）。
    只报其中一个都会被读错，所以它们绑在一起。
    """

    angle_deg: float
    period_nm: float
    #: **带通**峰峰值起伏，皮米（``2√2 × rms``，与 ``stripe_corrugation_pm``
    #: 同口径，可以直接和 herringbone 那边的数比）。
    #:
    #: ⚠️ 它是**上界**：环带宽 ±35%，带内一切都算进来，包括条纹在这个频率附近的
    #: 那一份。把它当成「原子起伏有这么高」会系统性高估。
    amplitude_pm: float
    #: **整帧相干**峰峰值起伏，皮米：把整幅图投影到那一个复指数上
    #: （``2|⟨z·e^{-i k·r}⟩|`` 的峰峰值）。
    #:
    #: ⚠️ **它不是「原子起伏」，它是「整帧的相位一致性」。** 真实晶格在几十纳米
    #: 上会因帧内漂移与压电非线性慢慢失相，于是这个数随视野变大而塌掉：
    #: 同一根针尖，5 nm 帧上 4.42 pm、20 nm 帧 1.92 pm、30 nm 帧 0.13 pm。
    #: 要「局域原子起伏」请看 :attr:`TileLatticeMap.coherent_median_pm`
    #: —— 逐块算再取中位，块内失相可以忽略，于是它与视野无关。
    coherent_pm: float = 0.0
    #: ``coherent_pm / amplitude_pm`` ∈ (0, 1]。同上，它随视野变小，
    #: 读作「这个方向在整帧尺度上有多相干」，不是「有多干净」。
    coherence: float = 0.0


@dataclass(frozen=True)
class TileLatticeMap:
    """逐块的「晶格 / 其余」比值图。``ok=False`` 时 ``reason`` 说明为什么没给。"""

    ok: bool = False
    reason: str = ""
    #: 行优先的二维比值表（``grid[iy][ix]``），行号随 y 增大 —— 与图像同序。
    grid: tuple[tuple[float, ...], ...] = ()
    tile_nm: Optional[float] = None
    periods_per_tile: Optional[float] = None
    #: 比值 ≥ ``good_ratio`` 的块占比。
    good_fraction: Optional[float] = None
    median_ratio: Optional[float] = None
    good_ratio: Optional[float] = None
    #: **局域**原子起伏：逐块把图投影到晶格波矢上，取各块峰峰值的中位数，皮米。
    #:
    #: 这是本模块里唯一一个可以当「原子起伏有多高」引用的数。块只有几纳米，
    #: 块内的相位失相可以忽略，所以它**不随视野变化** —— 而整帧的
    #: ``DirectionAmplitude.coherent_pm`` 会随视野变大而塌掉。
    coherent_median_pm: Optional[float] = None
    #: 帧**几何**上下两排块各自的中位比值。
    #:
    #: ⚠️ 刻意**不叫**「先扫 / 后扫」：定向之后 row 0 恒为帧顶，而它是先扫还是后扫
    #: 取决于 ``:SCAN_DIR:``（``up`` 的第一行是帧**底**）。这个纯函数看不到文件头，
    #: 所以只报几何位置，扫描顺序由技能层按 ``scan_dir`` 映射。
    #: 两者差得多 ⇒ 针尖在这一帧里变了（与 ``atomic_phase._check_halves`` 问的是
    #: 同一件事，但这里只**报告**，判定留给调用方）。
    top_band_median: Optional[float] = None
    bottom_band_median: Optional[float] = None
    warnings: tuple[str, ...] = ()


def ring_mask(shape: tuple[int, int], period_px: float) -> npt.NDArray[np.bool_]:
    """``0.65·f0 < fr < 1.35·f0`` 的环 —— 与 ``herringbone.bandpass`` 逐字同源。

    单独抽出来是为了和角向扇区求交；环本身的定义**不在这里改**。
    """
    H, W = int(shape[0]), int(shape[1])
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    fr = np.hypot(fy, fx)
    f0 = 1.0 / float(period_px)
    return (fr > 0.65 * f0) & (fr < 1.35 * f0)


def _angle_mask(shape: tuple[int, int], angle_deg: float,
                half_width_deg: float) -> npt.NDArray[np.bool_]:
    """以 ``angle_deg`` 为中心的双侧角向扇区（±k 都留）。

    角度约定与 ``herringbone.stripe_peak`` 一致：从快扫轴（+列）量到行号增大的
    方向。因为实信号的谱有 ``F(-k) = F*(k)``，扇区必须成对留，否则 ifft 出来是复的。
    """
    H, W = int(shape[0]), int(shape[1])
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    ang = np.degrees(np.arctan2(fy, fx))
    d = np.abs((ang - float(angle_deg) + 90.0) % 180.0 - 90.0)
    return d <= float(half_width_deg)


def directional_bandpass(image: npt.ArrayLike, period_px: float,
                         angle_deg: Optional[float] = None,
                         half_width_deg: float = 15.0) -> npt.NDArray[np.float64]:
    """环带 ∩ 角向扇区。``angle_deg=None`` ⇒ 整环（退化成 ``herringbone.bandpass``）。"""
    h = np.asarray(image, dtype=np.float64)
    keep = ring_mask(h.shape, period_px)
    if angle_deg is not None:
        keep = keep & _angle_mask(h.shape, angle_deg, half_width_deg)
    F = np.fft.fft2(h - h.mean())
    return np.real(np.fft.ifft2(F * keep))


def _pp_pm(band: npt.NDArray[np.float64]) -> float:
    """带通分量 → 峰峰值皮米。与 ``stripe_corrugation_pm`` 同一行换算。"""
    rms = float(np.sqrt(np.mean(band * band)))
    return float(2.0 * math.sqrt(2.0) * rms * 1e12)


def lattice_amplitude_pm(
    image_m: npt.ArrayLike,
    nm_per_px: float,
    directions: Sequence[tuple[float, float]],
    *,
    half_width_deg: float = 15.0,
) -> tuple[DirectionAmplitude, ...]:
    """**逐方向**的晶格起伏，峰峰值皮米。输入必须是米。

    ``directions`` 是 ``(角度°, 周期nm)`` 的序列 —— 通常来自
    ``lattice_calibration.find_lattice_peaks`` 或 :mod:`mast.vision.lattice_cell`。
    本函数**不自己找峰**：找峰有一份经过两批标定的实现（含脊点剔除），
    再写一份只会让两处慢慢漂开。

    量不了的方向（周期太小、图太小）会被跳过，不返回占位值。
    """
    h = np.asarray(image_m, dtype=np.float64)
    out: list[DirectionAmplitude] = []
    if h.ndim != 2 or min(h.shape) < 16 or not (nm_per_px and nm_per_px > 0):
        return ()
    ny, nx = h.shape
    finite = np.isfinite(h)
    if not finite.any():
        return ()
    h = np.where(finite, h, float(np.nanmean(h[finite])))
    yy, xx = np.mgrid[0:ny, 0:nx]
    rx = xx.ravel() * float(nm_per_px)
    ry = yy.ravel() * float(nm_per_px)
    hz = (h - float(np.mean(h))).ravel()
    for ang, per_nm in directions:
        per_px = float(per_nm) / float(nm_per_px)
        if not (per_px >= 3.0):          # 与 stripe_corrugation_pm 同一道门
            continue
        band = directional_bandpass(h, per_px, float(ang), half_width_deg)
        amp = _pp_pm(band)
        if not math.isfinite(amp):
            continue
        # 相干分量：投影到那一个复指数上。**与带通同样不加窗** —— 两个口径必须
        # 一致，否则比值不是「带内有多少是周期的」而是「两种窗的差」。
        # （曾经给相干那一侧加了 Hanning、带通那一侧没加，于是纯正弦上
        # coherence 得到 1.03 —— 一个定义上不可能超过 1 的量超过了 1。）
        th = math.radians(float(ang))
        kx = math.cos(th) / float(per_nm)
        ky = math.sin(th) / float(per_nm)
        ph = np.exp(-2j * np.pi * (rx * kx + ry * ky))
        # 2|⟨·⟩| 是振幅，×2 得峰峰值 ⇒ 共 4|⟨·⟩|/N。
        amp_c = float(4.0 * abs(np.sum(hz * ph)) / hz.size * 1e12)
        out.append(DirectionAmplitude(
            angle_deg=float(ang) % 180.0,
            period_nm=float(per_nm),
            amplitude_pm=amp,
            coherent_pm=amp_c,
            coherence=float(min(1.0, amp_c / amp)) if amp > 0 else 0.0,
        ))
    return tuple(out)


def streak_amplitude_pm(image_m: npt.ArrayLike, nm_per_px: float,
                        *, max_rows: float = _STREAK_ROWS,
                        fx_bins: float = _STREAK_FX_BINS) -> Optional[float]:
    """**逐行横向短划**的起伏，峰峰值皮米。输入必须是米。

    抓的形状：沿快轴延展（|fx| ≤ 几个频率格）、沿慢轴在 2–``max_rows`` 行之间
    起伏（行与行之间不相干）。针尖顶端在两个态之间闪变时，图上就是这样一条条横道。

    ⚠️ **这个数单独看没有意义，必须与同口径的晶格带通幅值一起读。**
    它随地形起伏变化，绝对幅值不能独自归因于针尖质量。
    有意义的是比值 ``streak / lattice_bandpass``，或者更好的是
    :func:`tile_lattice_map` 的逐块比值 —— 分子分母同块同窗，尺寸依赖自己抵消。
    这与 ``atomic_phase`` 模块注释里那条撤掉的「绝对起伏下限」是同一个教训。

    与既有两件的分工：``scan_artifacts._spike_frac`` 只抓 ≤2 px 的孤立毛刺（抓不到
    多像素短划）；``herringbone.slow_axis_power_ratio`` 给的是环上的方向比（无量纲，
    且必须在未做行对齐的图上算）。**这里给的是一个能与晶格幅值直接相减的 pm。**

    ⚠️ 它**不区分成因**：逐行短划可以是针尖闪变，也可以是真实的一维台阶边缘。
    区分要靠正反扫是否重现（同一根针尖的闪变不重现，表面结构重现）。

    ⚠️ 去趋势用的是 ``scan_prep.poly_subtract``（二维多项式面），**不是**
    ``tip_metrics._detrend``。后者是全仓判据的单一真源，但它第一步就减掉**行中值**
    —— 而行与行之间的偏置**正是这里要量的东西**，用它会把信号连同背景一起清零。
    ``herringbone.slow_axis_power_ratio`` 的文档里有同一条限制（「必须在不做行对齐
    的图上算」）。这不是绕开单一真源，是这道判据与那一份去趋势不兼容。
    """
    from mast.vision.scan_prep import poly_subtract

    h = np.asarray(image_m, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 16 or not (nm_per_px and nm_per_px > 0):
        return None
    # 背景倾斜/弯曲落在慢轴低频，与条纹带重叠。不减掉的话量到的是样品的倾斜。
    h = np.asarray(poly_subtract(h, order=1), dtype=np.float64)
    # 慢轴加窗：帧里装着非整数个周期的地形（例如 5 nm 帧上一个 4 nm 的起伏）
    # 会从帧边沿泄漏到**所有** fy 上，把条纹带填满。实测：30 pm 的平滑正弦
    # 不加窗时被读成 7.7 pm 的「条纹」。窗按 rms 归一，幅值标定不变。
    H, W = h.shape
    wy = np.hanning(H)[:, None]
    h = h * wy
    h = h / float(np.sqrt(np.mean(wy * wy)))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    # |fx| 以频率格计：一格 = 1/W。
    keep_x = np.abs(fx) <= (float(fx_bins) / float(W))
    fy_abs = np.abs(fy)
    # ``fy`` 的单位是「每行的周数」，所以周期（行）= 1/|fy| —— 与 nm_per_px 无关。
    keep_y = (fy_abs >= 1.0 / float(max_rows)) & (fy_abs <= 0.5)
    keep = keep_x & keep_y
    if not keep.any():
        return None
    F = np.fft.fft2(h - h.mean())
    band = np.real(np.fft.ifft2(F * keep))
    amp = _pp_pm(band)
    return float(amp) if math.isfinite(amp) else None


def _tile_coherent_pm(tile: npt.NDArray[np.float64], nm_per_px: float,
                      directions: Sequence[tuple[float, float]]) -> float:
    """这一块里最强的那个晶格方向的**局域**峰峰值起伏，皮米。

    块只有几纳米，块内相位基本不失相 —— 所以这个数才是「原子起伏有多高」，
    而整帧的同名量量的是「整帧相位一致性」，会随视野变大而塌掉。
    """
    n_y, n_x = tile.shape
    yy, xx = np.mgrid[0:n_y, 0:n_x]
    rx = xx.ravel() * float(nm_per_px)
    ry = yy.ravel() * float(nm_per_px)
    hz = (tile - float(np.mean(tile))).ravel()
    best = 0.0
    for ang, per_nm in directions:
        th = math.radians(float(ang))
        ph = np.exp(-2j * np.pi * (rx * math.cos(th) / float(per_nm)
                                   + ry * math.sin(th) / float(per_nm)))
        amp = float(4.0 * abs(np.sum(hz * ph)) / hz.size * 1e12)
        best = max(best, amp)
    return best


def _tile_ratio(tile: npt.NDArray[np.float64], nm_per_px: float,
                directions: Sequence[tuple[float, float]],
                half_width_deg: float) -> float:
    """一块里「晶格带功率 / 其余结构功率」的平方根（= 幅值比）。

    分子分母**同一块、同一个窗**，所以块的尺寸与位置在比值里抵消 ——
    这正是它能跨帧尺寸比较、而角向集中度不能的原因。
    """
    n = tile.shape[0]
    w = np.outer(np.hanning(n), np.hanning(tile.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2((tile - tile.mean()) * w))) ** 2
    H, W = tile.shape
    fy = (np.arange(H) - H // 2)[:, None] / float(H)
    fx = (np.arange(W) - W // 2)[None, :] / float(W)
    fr = np.hypot(fy, fx)
    with np.errstate(divide="ignore"):
        per = np.where(fr > 0, float(nm_per_px) / np.maximum(fr, 1e-12), np.inf)
    lat = np.zeros_like(fr, dtype=bool)
    for ang, per_nm in directions:
        f0 = float(nm_per_px) / float(per_nm)
        ring = (fr > 0.65 * f0) & (fr < 1.35 * f0)
        angm = np.degrees(np.arctan2(fy, fx))
        d = np.abs((angm - float(ang) + 90.0) % 180.0 - 90.0)
        lat |= ring & (d <= float(half_width_deg))
    other = (per >= _OTHER_PERIOD_NM[0]) & (per <= _OTHER_PERIOD_NM[1]) & ~lat
    p_lat = float(P[lat].sum()) if lat.any() else 0.0
    p_oth = float(P[other].sum()) if other.any() else 0.0
    if p_oth <= 0:
        return 0.0
    return float(math.sqrt(p_lat / p_oth))


def tile_lattice_map(
    image: npt.ArrayLike,
    nm_per_px: float,
    directions: Sequence[tuple[float, float]],
    *,
    tile_nm: float = 4.0,
    good_ratio: float = 0.6,
    half_width_deg: float = 15.0,
) -> TileLatticeMap:
    """把帧切成方块，逐块问「这一块的晶格压不压得住其余结构」。

    ``directions`` 与 :func:`lattice_amplitude_pm` 同源，一般是整帧上找到的两三个
    布拉格方向 —— **块内不重新找峰**：一块里只有几个周期，找峰会锁到噪声上，
    而「整帧知道晶格在哪个方向，逐块问它在不在」才是这张图要回答的问题。

    纯函数。给不出图时返回 ``ok=False`` + ``reason``，**不返回一张全零的图**。
    """
    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or not (nm_per_px and nm_per_px > 0):
        return TileLatticeMap(reason="bad_input")
    if not directions:
        return TileLatticeMap(reason="no_directions",
                              warnings=("没有给晶格方向 —— 块图问的是「这一块上"
                                        "那个方向的晶格在不在」，方向未知时无从问起。",))
    finite = np.isfinite(h)
    if finite.mean() < 0.5:
        return TileLatticeMap(reason="incomplete_frame")
    h = np.where(finite, h, float(np.nanmean(h[finite])))

    per_min_nm = min(float(p) for _a, p in directions)
    t_px = int(round(float(tile_nm) / float(nm_per_px)))
    periods = float(tile_nm) / per_min_nm
    if periods < _MIN_PERIODS_PER_TILE:
        return TileLatticeMap(
            reason="tile_too_small",
            tile_nm=float(tile_nm), periods_per_tile=periods,
            warnings=("每块只装得下 %.1f 个周期（下限 %.0f）—— 块再小，"
                      "块内谱就分不开晶格与噪声。要么把 tile_nm 调大，"
                      "要么这一帧的视野本来就不够切块。"
                      % (periods, _MIN_PERIODS_PER_TILE),))
    if t_px < 16:
        return TileLatticeMap(
            reason="tile_too_few_pixels", tile_nm=float(tile_nm),
            periods_per_tile=periods,
            warnings=("每块只有 %d 像素（下限 16）—— 像素太少，块内谱没有分辨率。"
                      % t_px,))
    ny, nx = h.shape[0] // t_px, h.shape[1] // t_px
    if ny < 1 or nx < 1:
        return TileLatticeMap(reason="frame_smaller_than_tile",
                              tile_nm=float(tile_nm), periods_per_tile=periods)

    grid = np.zeros((ny, nx), dtype=np.float64)
    coh = np.zeros((ny, nx), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            sub = h[iy * t_px:(iy + 1) * t_px, ix * t_px:(ix + 1) * t_px]
            grid[iy, ix] = _tile_ratio(sub, float(nm_per_px), directions,
                                       half_width_deg)
            coh[iy, ix] = _tile_coherent_pm(sub, float(nm_per_px), directions)
    good = float((grid >= float(good_ratio)).mean())
    return TileLatticeMap(
        ok=True,
        grid=tuple(tuple(float(v) for v in row) for row in grid),
        tile_nm=float(tile_nm),
        periods_per_tile=periods,
        good_fraction=good,
        median_ratio=float(np.median(grid)),
        good_ratio=float(good_ratio),
        coherent_median_pm=float(np.median(coh)),
        top_band_median=float(np.median(grid[0])),
        bottom_band_median=float(np.median(grid[-1])),
    )
