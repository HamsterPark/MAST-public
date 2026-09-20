# -*- coding: utf-8 -*-
"""从未知表面的二维谱峰测量实空间原胞 a₁、a₂ 和夹角 γ。

与 lattice_calibration / lattice_multiframe 的区别：后两者用已知晶格
作为外部标尺估计扫描器畸变；这里直接测量原胞，不预设六角或矩形对称性。
夹角保留为测量值，峰位使用功率加权质心做亚像素精修。候选峰复用
find_lattice_peaks 及其谱脊剔除规则，避免重复实现不同的找峰口径。

combine_up_down 对上下扫分别测量，再组合原胞参数；慢轴反向的差异可用于
描述漂移应变。没有独立扫描器标定时，晶格非直角与扫描器剪切仍可能混淆，
不能把未校正夹角直接解释为材料对称性的结论。

superstructure_test 对候选超结构波矢与空白对照应用相同的搜索统计量。
在邻域内取最大值对纯噪声也有正偏，且依赖搜索范围，因此同时报告对照
比值与判决，避免把孤立的半序位置幅值当作结构证据。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "CellResult",
    "SuperstructureResult",
    "measure_cell",
    "combine_up_down",
    "superstructure_test",
]

#: 两个基矢的夹角落在这个区间之外就不算「独立」——太接近 0/180° 时
#: 求逆是病态的，量出来的第二个基矢是噪声。
_MIN_INDEPENDENT_DEG = 20.0

#: 对照波矢的个数。取够多才能看出候选是不是落在对照的分布里。
_N_CONTROLS = 8


@dataclass(frozen=True)
class CellResult:
    """一帧上量出的实空间原胞。``ok=False`` 时 ``reason`` 说明为什么量不了。"""

    ok: bool = False
    reason: str = ""
    #: 较长的那个基矢，纳米。
    a1_nm: Optional[float] = None
    #: 较短的那个基矢，纳米。
    a2_nm: Optional[float] = None
    #: 两个基矢的夹角，度。**测量值**，没有被强制成 60/90/120°。
    gamma_deg: Optional[float] = None
    #: a₁ 相对快扫轴（+列）的方向，度，mod 180。
    a1_angle_deg: Optional[float] = None
    #: 原胞面积，nm²。
    area_nm2: Optional[float] = None
    #: 两个基矢各自的谱峰信噪（峰功率 / 带内中位功率）。
    snr: tuple[float, float] = (0.0, 0.0)
    #: 选中的这一对基矢把几个 / 共几个候选峰指标成了整数。
    #: ``indexed == 2`` 意味着「只解释了它们自己」—— 没有任何独立证据支持这一对
    #: 就是原胞基矢，结果要当成弱证据看。
    indexed: int = 0
    indexed_total: int = 0
    #: 参与选基矢的峰数，以及被 ``_is_ridge_point`` 剔掉的脊点数。
    #: 后者大而前者小 ⇒ 「看了，看到的全是脊」，与「没法看」是两件事
    #: （``LatticeResult.n_ridge`` 存在的理由，这里照样带出来）。
    n_peaks: int = 0
    n_ridge: int = 0
    scan_dir: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuperstructureResult:
    """一个候选超结构波矢的检验结果。"""

    label: str = ""
    period_nm: Optional[float] = None
    #: 候选处的相干幅值（与输入同单位；输入为米时即米）。
    amplitude: Optional[float] = None
    #: 对照波矢上**同一个统计量**的中位数与最大值。
    control_median: Optional[float] = None
    control_max: Optional[float] = None
    #: ``amplitude / control_max``。判决就看它。
    #:
    ratio_to_control: Optional[float] = None
    #: 三态：``present`` / ``absent`` / ``undetermined``。
    verdict: str = "undetermined"
    note: str = ""


def _spectrum(image: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], int, int]:
    """去均值 + Hanning + ``fftshift(fft2)`` 的**功率**谱。

    用功率（``|F|²``）而不是幅值，与 ``herringbone.stripe_peak`` 的质心口径一致
    （``find_lattice_peaks`` 内部用的是 ``np.abs``，那是它自己的取峰口径，
    这里只借它的**峰位**，精修在本模块的谱上做）。
    """
    h = np.asarray(image, dtype=np.float64)
    ny, nx = h.shape
    finite = np.isfinite(h)
    if not finite.all():
        h = np.where(finite, h, float(np.nanmean(h[finite])) if finite.any() else 0.0)
    win = np.outer(np.hanning(ny), np.hanning(nx))
    F = np.fft.fftshift(np.fft.fft2((h - h.mean()) * win))
    return np.abs(F) ** 2, ny, nx


def _refine(P: npt.NDArray[np.float64], iy: int, ix: int) -> tuple[float, float]:
    """3×3 功率加权质心 → ``(fy, fx)``，单位「每像素的周数」。

    与 ``herringbone.stripe_peak`` 的精修逐行同构：非方帧上两轴各自按轴长归一。
    """
    ny, nx = P.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, iy - 1), min(ny, iy + 2)
    x0, x1 = max(0, ix - 1), min(nx, ix + 2)
    w = P[y0:y1, x0:x1]
    tot = float(w.sum())
    if tot <= 0:  # pragma: no cover — argmax 处权重和不会是 0
        return (float(iy - cy) / ny, float(ix - cx) / nx)
    yy = np.arange(y0, y1, dtype=np.float64)[:, None]
    xx = np.arange(x0, x1, dtype=np.float64)[None, :]
    fy = (float((w * yy).sum() / tot) - cy) / float(ny)
    fx = (float((w * xx).sum() / tot) - cx) / float(nx)
    return fy, fx


def _gauss_reduce(a1: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把一对基矢约化成最短的一对（Gauss / Lagrange 约化）。

    没有这一步，「选了哪两个峰」会改变报出来的 a₁/a₂ —— 同一个晶格可以由无穷多
    组基矢张成。约化之后结果是**规范的**，两帧才可比。

    ⚠️ 方向不能反：每一轮必须让 **``a1`` 是较短的那个**，再拿它去约 ``a2``。
    反过来（拿长的去约短的）``mu`` 恒为 0，循环第一步就 break，函数看起来跑完了
    却什么都没约化。若输入基矢尚未约化，这会把超胞误报为原胞。
    """
    a1 = np.asarray(a1, dtype=float).copy()
    a2 = np.asarray(a2, dtype=float).copy()
    for _ in range(32):
        if float(np.dot(a1, a1)) > float(np.dot(a2, a2)):
            a1, a2 = a2, a1                      # a1 = 较短的那个
        denom = float(np.dot(a1, a1))
        if denom <= 0:
            break
        mu = int(round(float(np.dot(a1, a2)) / denom))
        if mu == 0:
            break
        a2 = a2 - mu * a1
    return a1, a2

# 向 find_lattice_peaks 请求的峰数决定搜索预算。
# 被剔除的谱脊极大也会消耗迭代次数；预算不足可能在到达孤立峰之前用尽。
# 增加预算不依赖样品周期，优于为某种材料收窄通用搜索带。
# 其他调用方使用不同预算时，条纹较重的输入仍可能返回 too_few_peaks。
_PEAK_BUDGET = 12


#: 指标化判据：一个观测峰的分数指标离整数多远还算「被指标上」。
_INDEX_TOL = 0.15

# 一个周期所需的最少像素数，限制欠采样时的原胞测量。
# 使用每周期像素数，不直接沿用按绝对 nm/px 定义的门，以适应不同晶格周期。
# 4 像素在 Nyquist 的 2 像素基础上保留余量，并覆盖 3×3 功率质心的精修邻域。
_MIN_PX_PER_PERIOD = 4.0

# 可信峰所需的最少重复周期数。
# 长周期峰接近谱心，容易被残余低频背景或条纹脊主导。
# 因此搜索带的长周期端须随实际帧宽限制，不能只用固定纳米上界。
_MIN_PERIODS_IN_FRAME = 12.0

#: 参与配对枚举的最强峰个数。再多只是噪声峰互相配对。
_MAX_PAIR_CANDIDATES = 8

#: 一对基矢要算「解释得了这张谱」，被它指标上的峰必须占到候选总功率的这么多。
#: 0.90 留出 10% 给杂峰（条纹脊的谐波、双针尖回声）——它们功率小，
#: 漏掉它们不该改变原胞的选择。
_INDEX_POWER_MIN = 0.90


def _best_indexing_pair(cands):
    """从候选峰里挑出**真正的原胞基矢**：能把其余观测峰指标成整数的那一对。

    ## 为什么不能只按功率取最强的两个

    真机上这么做会错三种，而且每一种都给出一个**看起来完全合理**的晶格常数：

    * 挑中二阶峰 ``2b₂`` ⇒ 实空间 a₂ 变成真值的**一半**（#0104：0.1807 nm）；
    * 挑中条纹在带边留下的弱峰 ⇒ a₁ 变成两倍（#0110：0.8345 nm）；
    * 挑中一个噪声方向 ⇒ γ 从 89° 变成 46.8°。

    这三帧的第二个峰信噪是 4202 / 4066 / 1219，而**正确**的帧里最低的是 3641 ——
    区间重叠，所以「信噪阈值」分不开它们。试过，不行，别再试。

    ## 判据：一对基矢对不对，问的是它能不能解释**别的**峰

    倒空间里每个观测峰都应当是 ``h·k₁ + l·k₂``（h、l 整数）。把所有候选峰在待选
    基底下求分数指标，数一数有多少落在整数附近（容差 ``_INDEX_TOL``）——
    错误的一对解释不了其余的峰，这与信噪无关，是几何。

    ## 得分必须按**功率**记，不能数个数

    数个数时，一个功率微弱的杂峰会左右结果：条纹脊在 0.73 nm 处留下一个弱峰，
    基底取 ``(b₁, b₂/2)`` 能把它连同真峰一起指标上（真 b₂ 落在 (0,2)），
    比正确的 ``(b₁, b₂)`` **多解释一个峰**，于是永远赢。实测后果：a₁ 被报成
    0.72-0.75 nm，正好是真值 0.364 的两倍，而 γ 完全正确（88.8 ± 0.4°）——
    一个只错在一个维度上、看起来极其可信的结果。

    改成「被指标上的**功率**占总功率的比例 ≥ ``_INDEX_POWER_MIN``」之后，
    漏掉那个弱峰只损失不到 1% 的功率，正确的基底照样合格，再由行列式挑出最粗的。

    ## 第二道：基矢**自己**必须是强峰

    只有指标化这一道时，行列式往哪个方向取都救不了 —— 两种错法各占一边，
    实测都出现过：

    * 选中 ``b₂/2``（长周期端的弱杂峰）⇒ 实空间原胞**翻倍**，a₁ 报成 0.708 nm；
    * 选中 ``2b₂``（二阶谐波）⇒ 实空间原胞**减半**，a₂ 报成 0.182 nm。

    第一种要行列式取大才排得掉，第二种要取小 —— 同一个旋钮不可能同时满足。

    真正区分它们的是**强度**：基频永远比自己的谐波强，也比条纹留下的杂峰强。
    所以并列判据是「两个基矢的功率之和最大」。这不是启发式，是衍射的常识。
    行列式只留作精确并列时的最后一道。

    ⚠️ **它救不了一种情形**：真正的 ``b₂`` 根本没被观测到（太弱）。那时
    ``(b₁, 2b₂)`` 能把看到的全部指标上，于是 a₂ 仍会被报成一半。这不是算法的
    缺陷，是数据里就没有那条信息 —— 判据只能与观测一样强。
    """
    pool = cands[:_MAX_PAIR_CANDIDATES]
    total_p = sum(max(c[2], 0.0) for c in pool) or 1.0
    qualified: list[tuple] = []          # (基矢功率和, det, k1, k2, n_indexed)
    fallback: tuple = (-1.0, None, None, 0, -1.0)   # (frac, k1, k2, n_indexed, det)
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            k1, k2 = pool[i], pool[j]
            cosang = min(1.0, abs(float(np.dot(k1[1], k2[1])) / (k1[0] * k2[0])))
            if math.degrees(math.acos(cosang)) < _MIN_INDEPENDENT_DEG:
                continue
            B = np.array([k1[1], k2[1]], dtype=float)
            det = abs(float(np.linalg.det(B)))
            if det < 1e-12:
                continue
            Binv = np.linalg.inv(B)
            n_idx = 0
            p_idx = 0.0
            for c in pool:
                hl = c[1] @ Binv
                if float(np.max(np.abs(hl - np.round(hl)))) <= _INDEX_TOL:
                    n_idx += 1
                    p_idx += max(c[2], 0.0)
            frac = p_idx / total_p
            if frac >= _INDEX_POWER_MIN:
                qualified.append((max(k1[2], 0.0) + max(k2[2], 0.0), det,
                                  k1, k2, n_idx))
            base_p = max(k1[2], 0.0) + max(k2[2], 0.0)
            if frac > fallback[0] or (frac == fallback[0] and base_p > fallback[4]):
                fallback = (frac, k1, k2, n_idx, base_p)
    if qualified:
        # 先按两个基矢的功率和（基频最强），精确并列时再看行列式。
        qualified.sort(key=lambda t: (-t[0], -t[1]))
        _p, _det, k1, k2, n_idx = qualified[0]
        return k1, k2, n_idx, len(pool)
    return fallback[1], fallback[2], fallback[3], len(pool)


def measure_cell(image: npt.ArrayLike, nm_per_px: float, *,
                 band_nm: Optional[tuple[float, float]] = None,
                 max_peaks: int = _PEAK_BUDGET,
                 scan_dir: str = "") -> CellResult:
    """从一帧原子分辨图量出实空间原胞。

    图必须已经过 :func:`mast.io.nanonis_files.sxm_oriented_frames` 定向。
    纯函数：不读文件、不碰硬件、不抛异常。
    """
    from mast.vision.lattice_calibration import PEAK_BAND_NM, find_lattice_peaks

    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 32:
        return CellResult(reason="image_too_small")
    if not (nm_per_px and nm_per_px > 0):
        return CellResult(reason="unknown_pixel_size")

    band = tuple(band_nm) if band_nm else PEAK_BAND_NM
    if band_nm is None:
        # 长周期端跟着帧宽收 —— 见 _MIN_PERIODS_IN_FRAME 的说明。
        width_nm = float(nm_per_px) * float(min(h.shape))
        cap = width_nm / _MIN_PERIODS_IN_FRAME
        if cap <= band[0]:
            return CellResult(
                reason="frame_too_small_for_band",
                warnings=("视野只有 %.2f nm，除以最少周期数 %.0f 之后连搜索带的下界"
                          "（%.2f nm）都够不到 —— 这一帧装不下足够多的周期，"
                          "量出来的「晶格」会是直流裙边。"
                          % (width_nm, _MIN_PERIODS_IN_FRAME, band[0]),))
        band = (band[0], min(band[1], cap))
    found = find_lattice_peaks(h, float(nm_per_px), band_nm=band, max_peaks=max_peaks)
    if not found.ok or len(found.peaks) < 2:
        extra = ()
        if found.n_ridge >= 2 * max(1, len(found.peaks)):
            extra = ("带内的极大有 %d 个被判成脊点、只剩 %d 个候选 —— 这一帧的谱被"
                     "条纹主导。「看到的全是脊」与「没法看」是两件事：前者说明"
                     "针尖在闪，后者说明帧本身不可用。"
                     % (found.n_ridge, len(found.peaks)),)
        return CellResult(reason=found.reason or "too_few_peaks",
                          n_peaks=len(found.peaks), n_ridge=int(found.n_ridge),
                          warnings=tuple(found.warnings) + extra)

    P, ny, nx = _spectrum(h)
    cy, cx = ny // 2, nx // 2
    # 带内中位功率 —— 每个峰的信噪分母。
    fy_g = (np.arange(ny) - cy)[:, None] / float(ny)
    fx_g = (np.arange(nx) - cx)[None, :] / float(nx)
    fr = np.hypot(fy_g, fx_g)
    with np.errstate(divide="ignore"):
        per = np.where(fr > 0, float(nm_per_px) / np.maximum(fr, 1e-12), np.inf)
    in_band = (per >= band[0]) & (per <= band[1])
    bg = float(np.median(P[in_band])) if in_band.any() else 0.0

    # 精修每一个候选峰，换成 nm^-1 的波矢。上半平面代表元去掉 ±k 重复。
    cands: list[tuple[float, np.ndarray, float]] = []   # (|k|, k_nm^-1, snr)
    for pk in found.peaks:
        iy = int(round(cy + pk.ky))
        ix = int(round(cx + pk.kx))
        if not (1 <= iy < ny - 1 and 1 <= ix < nx - 1):
            continue
        fy, fx = _refine(P, iy, ix)
        if fy < 0 or (fy == 0 and fx < 0):     # 上半平面代表元
            fy, fx = -fy, -fx
        k = np.array([fx / float(nm_per_px), fy / float(nm_per_px)], dtype=float)
        norm = float(np.hypot(*k))
        if norm <= 0:
            continue
        snr = float(P[iy, ix] / bg) if bg > 0 else 0.0
        if any(abs(norm - c[0]) < 1e-9 and
               abs(float(np.dot(k, c[1])) - norm * c[0]) < 1e-9 for c in cands):
            continue
        cands.append((norm, k, snr))
    if len(cands) < 2:
        return CellResult(reason="too_few_refined_peaks",
                          n_peaks=len(found.peaks), n_ridge=int(found.n_ridge))

    cands.sort(key=lambda c: -c[2])
    k1, k2, idx_score, idx_total = _best_indexing_pair(cands)
    if k2 is None:
        return CellResult(
            reason="no_independent_pair",
            n_peaks=len(found.peaks), n_ridge=int(found.n_ridge),
            warnings=("找到的峰全都近乎共线 —— 这是**一维条纹**的样子，不是二维晶格。"
                      "一维结构没有原胞可言。",))

    B = np.array([k1[1], k2[1]], dtype=float)
    det = float(np.linalg.det(B))
    if abs(det) < 1e-12:
        return CellResult(reason="singular_basis")
    # 每周期像素数 —— 用**选中的**基矢来判，而不是用一个绝对的 nm/px 阈值。
    px_per_period = min(1.0 / (k1[0] * float(nm_per_px)),
                        1.0 / (k2[0] * float(nm_per_px)))
    if px_per_period < _MIN_PX_PER_PERIOD:
        return CellResult(
            reason="too_few_pixels_per_period",
            n_peaks=len(found.peaks), n_ridge=int(found.n_ridge),
            warnings=("最短的那个周期只占 %.1f 个像素（下限 %.0f）—— 这么少的采样点"
                      "凑得出峰但量不准周期，亚像素精修也无从做起。"
                      "把视野缩小或把像素加密。"
                      % (px_per_period, _MIN_PX_PER_PERIOD),))
    A = np.linalg.inv(B).T            # 行 = 实空间基矢，纳米
    a1, a2 = _gauss_reduce(A[0].copy(), A[1].copy())
    l1, l2 = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
    if l1 < l2:                        # 约定：a₁ 是较长的那个
        a1, a2 = a2, a1
        l1, l2 = l2, l1
    cosg = float(np.dot(a1, a2)) / (l1 * l2)
    gamma = math.degrees(math.acos(max(-1.0, min(1.0, cosg))))
    if gamma > 90.0:                   # 取锐角代表元（等价原胞）
        a2 = -a2
        gamma = 180.0 - gamma
    warns = list(found.warnings)
    from mast.vision.atomic_phase import scale_gate
    gate = scale_gate(float(nm_per_px))
    if gate == "off":
        warns.append(
            "像素尺度 %.3f nm/px 在 ``atomic_phase.scale_gate`` 那里是 ``off``，"
            "所以 ``AssessAtomicResolution`` / ``AssessAtomicPhase`` 会拒判这一帧。"
            "本模块按**每周期像素数**判（这里 %.1f px/周期，够），两者不冲突："
            "那道门的 0.05 nm/px 是在 0.25 nm 的晶格上标的，换到 %.2f nm 的周期上"
            "对应的应当是 %.3f nm/px。"
            % (float(nm_per_px), px_per_period, 1.0 / k2[0],
               0.05 * (1.0 / k2[0]) / 0.25))
    if idx_score <= 2 and idx_total > 2:
        warns.append(
            "这一对基矢只指标上了它自己（%d/%d 个候选峰）—— 没有独立的峰来印证它"
            "就是原胞基矢。这时 a₂ 被报成真值一半（选中了二阶峰）这类错误"
            "查不出来，结果只能当弱证据。" % (idx_score, idx_total))
    warns.append(
        "γ = %.2f° 是**测量值**，但它含着未校正的扫描器剪切 —— 单帧分不开"
        "「表面本来就不是直角」与「扫描器把直角扭了」。要分开就换慢轴方向或"
        "换扫描角重测（calibrate_up_down / calibrate_multi_angle）。" % gamma)
    return CellResult(
        ok=True,
        a1_nm=l1, a2_nm=l2, gamma_deg=gamma,
        a1_angle_deg=float(math.degrees(math.atan2(a1[1], a1[0])) % 180.0),
        area_nm2=float(abs(l1 * l2 * math.sin(math.radians(gamma)))),
        snr=(k1[2], k2[2]),
        indexed=int(idx_score), indexed_total=int(idx_total),
        n_peaks=len(found.peaks), n_ridge=int(found.n_ridge),
        scan_dir=str(scan_dir or ""),
        warnings=tuple(warns),
    )


def combine_up_down(up: Sequence[CellResult], down: Sequence[CellResult], *,
                    frame_height_nm: Optional[float] = None,
                    frame_time_s: Optional[float] = None,
                    min_indexed: int = 3) -> dict:
    """上下扫成对平均，消掉慢轴漂移在 a₂ 上的应变。

    上扫与下扫的慢轴推进方向相反，漂移对沿慢轴长度的拉伸/压缩因此**反号**，
    取平均即抵消；差值本身就是漂移应变。这与 ``calibrate_up_down`` 用同一条对称性，
    但那里分离的是**剪切角**（漂移的快轴分量），这里是**长度**（慢轴分量）——
    两者是同一个漂移矢量的两个分量，互补而不重复。

    ``frame_height_nm`` 与 ``frame_time_s`` 都给了才换算漂移速率；缺任一项就只报
    应变，**不猜**。

    ``min_indexed`` 只收「有独立峰印证」的帧：``indexed == 2`` 的那一对基矢只解释了
    它自己，这种帧上「a₂ 被报成一半」查不出来。2026-09-03 的 28 帧里正好 4 帧如此，
    而且**全部**是离群值 —— 它们自己报出了低置信，聚合层照着筛就是了。

    统计量用**中位数**而不是均值：一帧的错解会把均值拖走，而这种错解不是罕见事故，
    是「峰没找全」的常态后果。
    """
    def _usable(rows):
        return [c for c in rows
                if c.ok and c.a2_nm and int(c.indexed or 0) >= int(min_indexed)]

    u, d = _usable(up), _usable(down)
    n_drop = (len([c for c in up if c.ok]) + len([c for c in down if c.ok])
              - len(u) - len(d))
    out: dict = {"ok": False, "n_up": len(u), "n_down": len(d),
                 "n_dropped_low_indexed": n_drop, "min_indexed": int(min_indexed)}
    if not u or not d:
        out["reason"] = ("可用的上扫 %d 帧 / 下扫 %d 帧（另有 %d 帧因指标化证据不足被"
                         "剔除）—— 两个方向都要有才谈得上抵消。只有一个方向时，"
                         "a₂ 里的漂移应变留在结果里且看不出来。"
                         % (len(u), len(d), n_drop))
        return out

    def _mean(rows, attr):
        vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        return float(np.median(vals)) if vals else None

    a2_u, a2_d = _mean(u, "a2_nm"), _mean(d, "a2_nm")
    a1_u, a1_d = _mean(u, "a1_nm"), _mean(d, "a1_nm")
    g_u, g_d = _mean(u, "gamma_deg"), _mean(d, "gamma_deg")
    a2 = 0.5 * (a2_u + a2_d)
    a1 = 0.5 * (a1_u + a1_d)
    strain = float((a2_u - a2_d) / (2.0 * a2)) if a2 else None
    out.update({
        "ok": True,
        "a1_nm": a1, "a2_nm": a2,
        "gamma_deg": 0.5 * (g_u + g_d) if (g_u is not None and g_d is not None) else None,
        "a1_up_nm": a1_u, "a1_down_nm": a1_d,
        "a2_up_nm": a2_u, "a2_down_nm": a2_d,
        # 定义是**操作性**的：``(a₂_up − a₂_down) / 2a₂``。它的**大小**就是漂移
        # 在慢轴上造成的应变；它的**正负**对应哪个物理漂移方向，取决于「样品往哪
        # 走」与「图上周期变长还是变短」之间的符号链，本仓没有用一次已知方向的
        # 漂移钉过 —— 所以别拿这个符号去推方向。
        "slow_axis_drift_strain": strain,
    })
    # 帧间散布，作为不确定度的下限（**不是**测量精度：帧间还有真实的针尖差异）。
    for tag, rows in (("up", u), ("down", d)):
        for attr in ("a1_nm", "a2_nm", "gamma_deg"):
            vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
            out["%s_%s_sd" % (tag, attr)] = (float(np.std(vals, ddof=1))
                                             if len(vals) > 1 else None)
    if strain is not None and frame_height_nm and frame_time_s:
        v = abs(strain) * float(frame_height_nm) / float(frame_time_s)
        out["drift_nm_per_h"] = float(v * 3600.0)
        out["drift_note"] = (
            "漂移速率 %.2f nm/h 的**大小**站得住（它就是上下扫的不对称）；"
            "**方向的符号未经真机确认** —— 定向翻转与慢轴推进方向的符号约定要用"
            "一次已知方向的漂移去钉，本仓对角度符号也是这么处理的。"
            % out["drift_nm_per_h"])
    return out


def _max_over_neighbourhood(hw, x_nm, y_nm, wsum, k0, span, step) -> float:
    """在 ``k0`` 附近的细网格上取相干幅值的最大 —— **有偏**统计量，必须配对照。

    ``hw`` 是已经乘好窗的图。用**可分离**的两次矩阵乘法算整片网格：

        A(kx,ky) = Σ_y e^{-2πi·y·ky} · [ Σ_x hw(y,x)·e^{-2πi·x·kx} ]

    逐点двойной循环在 512² 的图上要 6 亿次复数运算（一次单元测试跑几分钟）；
    分离之后是两次矩阵乘，数学上完全等价。
    """
    n = int(round(span / step))
    offs = np.arange(-n, n + 1, dtype=float) * float(step)
    kx = k0[0] + offs
    ky = k0[1] + offs
    Ex = np.exp(-2j * np.pi * np.outer(x_nm, kx))        # (nx, nk)
    Ey = np.exp(-2j * np.pi * np.outer(ky, y_nm))        # (nk, ny)
    A = Ey @ (hw @ Ex)                                    # (nk_y, nk_x)
    return float(2.0 * np.max(np.abs(A)) / wsum) if wsum > 0 else 0.0


def superstructure_test(
    image: npt.ArrayLike,
    nm_per_px: float,
    cell: CellResult,
    *,
    fractions: Sequence[tuple[float, float]] = ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5)),
    search_span_per_nm: float = 0.06,
    search_step_per_nm: float = 0.005,
    n_controls: int = _N_CONTROLS,
    seed: int = 0,
) -> tuple[SuperstructureResult, ...]:
    """半序（或任意分数序）位置上有没有真实的调制 —— **带空白对照**。

    对每个候选分数序波矢，在其邻域细网格上取相干幅值的最大值；再对
    ``n_controls`` 个**不可能有结构**的对照波矢做同一件事。判决只看候选与对照
    最大值的比值：

    * ``> 1.5``  ⇒ ``present``
    * ``< 1.2``  ⇒ ``absent``
    * 其间       ⇒ ``undetermined``

    这两个数不是标定出来的阈值，而是「候选必须明显跳出对照的分布」的一个朴素
    表述；结果里带着 ``control_median`` / ``control_max``，调用方可以自己复核。
    """
    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or not cell.ok or cell.a1_nm is None:
        return ()
    ny, nx = h.shape
    finite = np.isfinite(h)
    if not finite.all():
        h = np.where(finite, h, float(np.nanmean(h[finite])) if finite.any() else 0.0)
    h = h - float(np.mean(h))
    win = np.outer(np.hanning(ny), np.hanning(nx))
    hw = h * win
    x_nm = np.arange(nx, dtype=float) * float(nm_per_px)
    y_nm = np.arange(ny, dtype=float) * float(nm_per_px)
    wsum = float(np.sum(win))

    ang = math.radians(cell.a1_angle_deg or 0.0)
    g = math.radians(cell.gamma_deg or 90.0)
    A = np.array([
        [cell.a1_nm * math.cos(ang), cell.a1_nm * math.sin(ang)],
        [cell.a2_nm * math.cos(ang + g), cell.a2_nm * math.sin(ang + g)],
    ], dtype=float)
    B = np.linalg.inv(A).T            # 行 = 倒格矢 b1, b2

    rng = np.random.default_rng(seed)
    controls: list[float] = []
    for _ in range(int(n_controls)):
        # 对照点：分数坐标取在明显不是 0 / ½ / 1 的地方，且落在同一个 |k| 量级上。
        while True:
            u = float(rng.uniform(0.15, 0.85))
            v = float(rng.uniform(0.15, 0.85))
            if min(abs(u - 0.5), abs(v - 0.5)) > 0.12 and (u + v) > 0.25:
                break
        k0 = u * B[0] + v * B[1]
        controls.append(_max_over_neighbourhood(hw, x_nm, y_nm, wsum, k0,
                                                search_span_per_nm,
                                                search_step_per_nm))
    c_med = float(np.median(controls)) if controls else None
    c_max = float(np.max(controls)) if controls else None

    out: list[SuperstructureResult] = []
    for fu, fv in fractions:
        k0 = float(fu) * B[0] + float(fv) * B[1]
        norm = float(np.hypot(*k0))
        amp = _max_over_neighbourhood(hw, x_nm, y_nm, wsum, k0,
                                      search_span_per_nm, search_step_per_nm)
        ratio = (amp / c_max) if (c_max and c_max > 0) else None
        if ratio is None:
            verdict, note = "undetermined", "没有对照，判不了"
        elif ratio > 1.5:
            verdict = "present"
            note = ("候选幅值是对照最大值的 %.2f 倍 —— 跳出了噪声本底。" % ratio)
        elif ratio < 1.2:
            verdict = "absent"
            note = ("候选幅值只有对照最大值的 %.2f 倍 —— 与「什么都没有」不可区分。"
                    "注意这不是「幅值为零」：细网格取最大在纯噪声上也给正数，"
                    "所以能说的是**在这个本底之上没有**。" % ratio)
        else:
            verdict = "undetermined"
            note = ("候选 / 对照 = %.2f，落在两可之间。要下结论就加长积分"
                    "（更大的帧）或换更干净的针尖。" % ratio)
        out.append(SuperstructureResult(
            label="(%g,%g)" % (fu, fv),
            period_nm=float(1.0 / norm) if norm > 0 else None,
            amplitude=amp, control_median=c_med, control_max=c_max,
            ratio_to_control=ratio, verdict=verdict, note=note))
    return tuple(out)
