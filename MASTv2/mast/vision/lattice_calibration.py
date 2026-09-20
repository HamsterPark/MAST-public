"""从原子分辨图估计二维布拉格峰、晶格常数与扫描器的仿射畸变。

先由 atomic_phase 判别原子周期性，再测量晶格。快扫方向上的一维投影周期
依赖晶格取向，不能直接用作二维定标；这里取二维谱的局部极大并保持峰间距。

仿射拟合使用六角晶格的三个等长倒格矢及其夹角约束，结合所选参考面的
晶格常数确定尺度。没有独立角度基准时整体旋转不可辨识，因此采用对称
畸变矩阵，并在多个解中选取最接近单位阵的分支。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: 已知表面的**最近邻原子间距** a（nm）。
#:
#: ⚠️ FFT 一阶峰对应的**不是** a，而是原子行间距 d = √3/2·a（六角面）——
#: 混用这两个数会带来 15.5% 的系统偏差，比这个模块要测的畸变还大。
#: :func:`first_order_period_nm` 是唯一的换算入口。
SURFACE_LATTICE_NM: dict[str, float] = {
    "Au(111)": 0.2884,
    "Ag(111)": 0.2889,
    "Cu(111)": 0.2556,
    "Pt(111)": 0.2775,
    "HOPG": 0.2464,
    "NaCl(100)": 0.3990,      # 正方，见 SQUARE_SURFACES
    "Si(111)-1x1": 0.3840,
}

#: 正方晶格的表面（其余按六角处理）。正方面的一阶峰周期就是 a 本身，
#: 没有 √3/2 那一步，而且「三个倒格矢夹角 120°」这条约束不适用。
SQUARE_SURFACES: frozenset[str] = frozenset({"NaCl(100)"})

#: 一阶峰的搜索带（纳米周期）。与 ``atomic_phase.ATOMIC_BAND_NM`` 同一区间 ——
#: 两处若漂开，判别通过的帧会在这里找不到峰，反之亦然。
PEAK_BAND_NM: tuple[float, float] = (0.18, 0.80)

#: 峰之间的最小间隔（像素）。小于它的两个极大是同一个峰的肩膀。
_PEAK_MIN_SEP_PX = 8

#: 认定「六重对称」时，相邻峰夹角与 60° 的最大偏差。
#: 5° 是宽的：热漂移会把峰拉成椭圆，而我们正是要测那个畸变 —— 拿一个严到
#: 排除畸变的阈值去筛，会把唯一有信息的那些帧筛掉。
_HEX_ANGLE_TOL_DEG = 8.0

#: 区分孤立布拉格峰与谱脊的分数阈值；分数定义见 ``_is_ridge_point``。
_RIDGE_SCORE_MAX = 0.145
#: 取脊邻居的半径范围（像素）。下界必须大于 ``_PEAK_MIN_SEP_PX`` 才不会量到峰自己。
_RIDGE_SPAN_PX = (9, 18)


def first_order_period_nm(surface: str) -> Optional[float]:
    """该表面 FFT 一阶峰对应的实空间周期（nm），未知表面返回 None。

    六角面是原子**行间距** d = √3/2·a，不是最近邻距离 a。这一步是本模块最容易
    被跳过、跳过后又最难发现的地方：两者差 15.5%，而典型的压电偏差是 10%，
    错了会得到一个看起来很合理的错数。
    """
    a = SURFACE_LATTICE_NM.get(surface)
    if not a:
        return None
    return a if surface in SQUARE_SURFACES else a * math.sqrt(3) / 2.0


@dataclass(frozen=True)
class LatticePeak:
    """倒空间里的一个一阶峰。``kx``/``ky`` 以像素为单位，原点在谱中心。"""

    kx: float
    ky: float
    power: float
    period_nm: float
    angle_deg: float


@dataclass(frozen=True)
class LatticeResult:
    """一帧上的晶格测量。``ok=False`` 时 ``reason`` 说明为什么量不了。"""

    ok: bool = False
    reason: str = ""
    n_peaks: int = 0
    peaks: tuple[LatticePeak, ...] = ()
    hexagonal: bool = False
    #: 三组独立方向各自的周期（nm）。它们**本该相等** —— 不等的程度就是畸变。
    periods_nm: tuple[float, ...] = ()
    period_mean_nm: Optional[float] = None
    #: 三个周期的相对散布。它是「这帧值不值得拿去定标」的第一判据。
    period_spread: Optional[float] = None
    angles_deg: tuple[float, ...] = ()
    lattice_angle_deg: Optional[float] = None
    #: 三个一阶方向的**幅值平衡度** = 最弱 / 最强。
    #:
    #: 读作「**原子圆不圆**」:六方晶格各向同性成像时三个方向幅值该差不多
    #: (接近 1);某一个方向被压低 ⇒ 图上原子**成行不成点**。
    #:
    #: 与 angular_concentration 联读：后者取环上最大 bin / 中位 bin，
    #: 多方向能量越均衡，这个比值可能越低。两者分别描述周期性的方向集中度
    #: 与各晶向幅值平衡，因此 direction_balance 仅报告，不单独作闸门。
    direction_balance: Optional[float] = None
    #: 被 :func:`_is_ridge_point` 剔掉的那些极大**本身**。
    #:
    #: 剔掉它们是为了**定标**:混进 ``peaks`` 会污染 ``period_spread``。
    #: 但「这一帧到底是不是一个晶格」是另一个问题,而回答那个问题需要看
    #: **全部**局部极大 —— 一条条纹的极大散布在各个半径上,那正是它的破绽。
    #:
    #: 保留被剔除的峰，供下游条纹判据核对全部局部极大的半径分布。
    ridge_peaks: tuple[LatticePeak, ...] = ()
    #: 被 :func:`_is_ridge_point` 剔掉的极大个数。
    #: **必须是字段，不能只写进 ``warnings``** —— 下游要用它把
    #: 「看了，看到的全是脊」与「没法看」分开，而那句告警是给人读的中文。
    #: 即使定标不可用，下游也应检查这个计数，而非把缺峰当作未见条纹。
    n_ridge: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationResult:
    """由一帧原子分辨图反推的扫描器仿射畸变。

    ``x_scale`` / ``y_scale`` 是**实际尺度 ÷ 标称尺度**：1.10 意味着仪器实际扫过
    的范围比它以为的大 10%，于是量出来的一切长度都偏小 10%，压电常数应当乘上它。

    ``nonorthogonality_deg`` **不能全算在压电头上** —— 慢轴方向的热漂移在图上
    表现为完全相同的剪切，单帧分不开。要分开就换扫描角度重测：压电的非正交随
    扫描角旋转，漂移引起的不随。
    """

    ok: bool = False
    reason: str = ""
    surface: str = ""
    expected_period_nm: Optional[float] = None
    x_scale: Optional[float] = None
    y_scale: Optional[float] = None
    nonorthogonality_deg: Optional[float] = None
    lattice_angle_deg: Optional[float] = None
    residual: Optional[float] = None
    #: 校正前三个周期的相对散布，以及校正后的（后者应该 ~0，否则解没收敛）。
    spread_before: Optional[float] = None
    spread_after: Optional[float] = None
    matrix: tuple[tuple[float, float], tuple[float, float]] | None = None
    warnings: tuple[str, ...] = ()


# ── 峰提取 ──────────────────────────────────────────────────────────────────

def _plane_subtract(img: np.ndarray) -> np.ndarray:
    """去一次平面。倾斜的背景在谱心附近堆起一座低频山，会淹掉一阶峰。"""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ok = np.isfinite(img)
    if ok.sum() < 16:
        return img - np.nanmean(img)
    A = np.column_stack([xx[ok].ravel(), yy[ok].ravel(), np.ones(int(ok.sum()))])
    coef, *_ = np.linalg.lstsq(A, img[ok].ravel(), rcond=None)
    return img - (coef[0] * xx + coef[1] * yy + coef[2])


def _is_ridge_point(F, cy, cx, kx, ky) -> bool:
    """判断局部极大是否属于谱脊，而非紧致的孤立布拉格峰。

    沿横纵轴分别取峰外 _RIDGE_SPAN_PX 范围内邻居的中位数与峰值之比，
    以较大者为分数。谱脊的邻居同样偏高，紧致峰的邻居通常较低。
    中位数可抑制落入窗口的相邻孤立峰；邻域下界大于峰间最小距离。
    检验适用于全部峰，不限制在穿过原点的坐标轴上，以识别平移后的谱脊。
    """
    import numpy as _np

    ny, nx = F.shape
    xi, yi = int(round(cx + kx)), int(round(cy + ky))
    if not (0 <= xi < nx and 0 <= yi < ny):
        return False
    pw = float(F[yi, xi])
    if not (pw > 0):
        return False
    lo, hi = _RIDGE_SPAN_PX
    offs = list(range(lo, hi + 1))
    offs = offs + [-o for o in offs]
    vy = [F[yi + d, xi] for d in offs if 0 <= yi + d < ny]
    vx = [F[yi, xi + d] for d in offs if 0 <= xi + d < nx]
    if not vy or not vx:
        return False
    score = max(float(_np.median(vy)) / pw, float(_np.median(vx)) / pw)
    return score >= _RIDGE_SCORE_MAX


def find_lattice_peaks(image, nm_per_px: float, *,
                       band_nm: tuple[float, float] = PEAK_BAND_NM,
                       max_peaks: int = 6) -> LatticeResult:
    """二维功率谱里的一阶布拉格峰。

    峰取的是**二维局部极大**，不是径向剖面的极大 —— 径向会把三组方向平均掉，
    而三组方向之间的差异正是畸变的全部信息。
    """
    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 32:
        return LatticeResult(reason="image_too_small")
    if not (nm_per_px and nm_per_px > 0):
        return LatticeResult(reason="unknown_pixel_size")
    finite = np.isfinite(h)
    if finite.mean() < 0.5:
        # 半张图都没有的帧做不了二维谱：缺的那半会被当成常数，谱上多出一片
        # 与扫描无关的结构。这正是「扫了几行就停」的帧的样子。
        return LatticeResult(reason="incomplete_frame",
                             warnings=("有效像素只有 %.0f%%" % (100 * finite.mean()),))
    h = np.where(finite, h, np.nanmean(h[finite]))
    flat = _plane_subtract(h)
    ny, nx = flat.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    F = np.abs(np.fft.fftshift(np.fft.fft2(flat * win)))
    cy, cx = ny // 2, nx // 2
    F[cy - 3:cy + 4, cx - 3:cx + 4] = 0.0        # DC 与最低频

    ky, kx = np.mgrid[0:ny, 0:nx]
    kx = kx - cx
    ky = ky - cy
    r = np.hypot(kx, ky)
    with np.errstate(divide="ignore"):
        # 周期（nm）= 视场 / |k|。视场沿两轴可能不同，这里用平均值定带，
        # 带只是搜索范围，不参与定量。
        span_px = 0.5 * (nx + ny)
        period = np.where(r > 0, span_px * float(nm_per_px) / np.maximum(r, 1e-9), np.inf)
    band = (period >= band_nm[0]) & (period <= band_nm[1])
    if not band.any():
        return LatticeResult(reason="band_empty")

    work = np.where(band, F, 0.0)
    peaks: list[LatticePeak] = []
    ridged: list[LatticePeak] = []
    n_ridge = 0
    # 循环上限给 3 倍而不是 2 倍：被剔掉的脊点也要消耗迭代次数，
    # 上限给太紧会让真峰在噪声重的帧上被饿死。
    for _ in range(max_peaks * 3):
        idx = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[idx] <= 0:
            break
        yi, xi = idx
        y0, y1 = max(0, yi - _PEAK_MIN_SEP_PX), yi + _PEAK_MIN_SEP_PX + 1
        x0, x1 = max(0, xi - _PEAK_MIN_SEP_PX), xi + _PEAK_MIN_SEP_PX + 1
        # 一条脊上的点**不是一阶峰**。混进来会把 period_mean_nm 与
        # period_spread 一起污染，而后者是「这帧值不值得拿去定标」的第一判据。
        # 脊分与邻域定义见 _is_ridge_point。
        if _is_ridge_point(F, cy, cx, float(xi - cx), float(yi - cy)):
            n_ridge += 1
            ridged.append(LatticePeak(
                kx=float(xi - cx), ky=float(yi - cy), power=float(F[idx]),
                period_nm=float(period[idx]),
                angle_deg=float(np.degrees(np.arctan2(yi - cy, xi - cx)))))
            work[y0:y1, x0:x1] = 0.0
            continue
        peaks.append(LatticePeak(
            kx=float(xi - cx), ky=float(yi - cy), power=float(F[idx]),
            period_nm=float(period[idx]),
            angle_deg=float(np.degrees(np.arctan2(yi - cy, xi - cx)))))
        work[y0:y1, x0:x1] = 0.0
        if len(peaks) >= max_peaks:
            break
    if len(peaks) < 2:
        # **``n_ridge`` 必须跟着这条路一起出去。** 「一个峰都没剩下，因为四个
        # 候选全是脊」和「这帧上本来就什么都没有」是完全不同的两件事,而下游
        # 只有拿到这个数才分得开。少带它 = 把一条证据折叠成「读不到」。
        return LatticeResult(reason="too_few_peaks", n_peaks=len(peaks),
                             peaks=tuple(peaks), n_ridge=n_ridge,
                             ridge_peaks=tuple(ridged),
                             warnings=(("剔掉 %d 个脊上的点后不足两个一阶峰"
                                        % n_ridge),) if n_ridge else ())

    # 每个峰有一个 ±k 的孪生。取角度落在 [0,180) 的那一半作为独立方向。
    uniq: list[LatticePeak] = []
    for p in peaks:
        a = p.angle_deg % 180.0
        if not any(min(abs(a - q.angle_deg % 180.0),
                       180 - abs(a - q.angle_deg % 180.0)) < 10.0 for q in uniq):
            uniq.append(p)
    uniq.sort(key=lambda p: -p.power)
    dirs = uniq[:3]
    periods = tuple(p.period_nm for p in dirs)
    angles = tuple(p.angle_deg % 180.0 for p in dirs)
    mean_p = float(np.mean(periods))
    spread = float((max(periods) - min(periods)) / mean_p) if mean_p else None

    # 三方向幅值平衡度 —— 「原子圆不圆」。见字段自述里那张 conc 对照表。
    _dir_bal = None
    if len(dirs) >= 3:
        _p = [float(d.power) for d in dirs if d.power > 0]
        if len(_p) >= 3 and max(_p) > 0:
            _dir_bal = float(min(_p) / max(_p))

    hexagonal = False
    warns: list[str] = []
    if n_ridge:
        # **说出来。** 静默剔除会让「为什么这帧只有两个方向」无从回答，
        # 而且掩盖了「这台机器的行噪声大到能挤进一阶峰」这条真实信息。
        warns.append("剔掉 %d 个脊上的点（是一条脊的一段，不是一阶峰）" % n_ridge)
    if len(dirs) >= 3:
        a = sorted(angles)
        gaps = [a[1] - a[0], a[2] - a[1], 180 - (a[2] - a[0])]
        hexagonal = all(abs(g - 60.0) <= _HEX_ANGLE_TOL_DEG for g in gaps)
        if not hexagonal:
            warns.append("三个方向的夹角 %s 偏离 60° 超过 %.0f°"
                         % (np.round(gaps, 1).tolist(), _HEX_ANGLE_TOL_DEG))
    else:
        warns.append("只找到 %d 个独立方向，六重对称无从判起" % len(dirs))

    return LatticeResult(
        ok=True, n_peaks=len(peaks), peaks=tuple(peaks), hexagonal=hexagonal,
        periods_nm=periods, period_mean_nm=mean_p, period_spread=spread,
        angles_deg=angles,
        lattice_angle_deg=float(min(angles)) if angles else None,
        n_ridge=n_ridge, ridge_peaks=tuple(ridged),
        direction_balance=_dir_bal, warnings=tuple(warns))


# ── 仿射求解 ────────────────────────────────────────────────────────────────

#: 接受一个分支所需的最大方程残差。二次方程组的真解残差在 1e-20 量级，
#: 这个阈值只用来剔除 fsolve 没收敛的那些返回值。
_AFFINE_RES_TOL = 1e-6

#: 上一次 ``solve_affine`` 的分支诊断。**不是线程安全的**，只给同进程内紧接着
#: 读一次用（``calibrate_from_lattice`` / ``calibrate_multi_angle``）；它存在的
#: 理由是「选了哪一支、还有没有别的支」不该只活在日志里。
_LAST_SOLVE: dict = {}


def solve_affine(k1: Sequence[float], k2: Sequence[float],
                 k_ideal_px: float) -> tuple[np.ndarray | None, float]:
    """解对称矩阵 ``W`` 使 ``W·k1``、``W·k2``、``W·(k1+k2)`` 等长且夹角 120°。

    返回 ``(W, 残差)``；解不出来时 ``(None, inf)``。

    为什么是三个约束、三个未知：``W`` 取对称（``[[a,b],[b,c]]``）等于假设扫描器
    不引入纯旋转。不这么假设的话整体旋转不可定 —— 见模块注释里那个
    ``Y 缩放 = -1.90`` 的退化解。
    """
    K1 = np.asarray(k1, dtype=np.float64)
    K2 = np.asarray(k2, dtype=np.float64)
    if K1.shape != (2,) or K2.shape != (2,) or k_ideal_px <= 0:
        return None, float("inf")

    # ── 输入必须是夹角 120° 的一对 ──────────────────────────────────────
    # 第三个方程写的是 ``u·v = -target²/2``，也就是 **cos120°**。六角晶格的
    # 六个峰里，任取两个的夹角要么 60° 要么 120°，而 ±K 是同一个方向 —— 于是
    # 「挑两个独立方向」这件事有两种同样合理的挑法，只有一种满足这里的方程。
    #
    # 传进来 60° 的一对时，方程组仍然有解：W 会去把 60° **掰成** 120°，代价是
    # 人为剪切；即使方程残差很小，几何解释也可能错误。
    #
    # 调用方（``calibrate_from_lattice`` 走的是三方向里天然 120° 的那对）本来
    # 就满足；新调用方不一定知道。所以在这里翻号自救，而不是指望每个调用方
    # 都读到上面那行 ``-target²/2``。
    n1, n2 = float(np.linalg.norm(K1)), float(np.linalg.norm(K2))
    if n1 < 1e-12 or n2 < 1e-12:
        return None, float("inf")
    if float(K1 @ K2) / (n1 * n2) > 0.0:      # 夹角 < 90° ⇒ 是 60° 那种挑法
        K2 = -K2

    target = float(k_ideal_px)

    def eqs(p):
        W = np.array([[p[0], p[1]], [p[1], p[2]]])
        u, v = W @ K1, W @ K2
        return [u @ u - target ** 2, v @ v - target ** 2, u @ v + target ** 2 / 2.0]

    try:
        from scipy.optimize import fsolve
    except Exception:  # noqa: BLE001 — scipy 不在就老实说解不了
        logger.debug("scipy unavailable for affine solve", exc_info=True)
        return None, float("inf")

    # ═══════════════════════════════════════════════════════════════════
    # 这个方程组**有多个解**，而且它们的残差都是 ~1e-30
    # ═══════════════════════════════════════════════════════════════════
    #
    # 「W·K1、W·K2、W·(K1+K2) 等长且互成 120°」对对称的 W 是 3 方程 3 未知，
    # 但它是二次的：若 W 是解，把 (Aᵀ)⁻¹ 乘上一个旋转再对称化往往给出另一个解，
    # 六角晶格自身的 60° 对称还会产生其它分支。仅凭残差或行列式
    # 不能区分这些解，因为相差保面积旋转的解可能同时满足方程。
    #
    # 所以这里要一个**物理先验**来定分支：扫描器的畸变离单位阵不远。压电常数
    # 的标定偏差是百分之几到十几，非正交是几度 —— 一个 -37° 的剪切不是「另一
    # 种可能」，它是数学分支，不是仪器。取 ‖M - I‖ 最小的那支。
    #
    # 但**先验不等于事实**：如果第二支离得也不远，那就是真的分不开，必须说出来
    # 而不是默默选一个。``residual`` 之外多返回的那点信息由调用方决定怎么用。
    sols: list[tuple[float, np.ndarray]] = []
    for s in (1.0, target / max(np.linalg.norm(K1), 1e-9), 0.5, 2.0):
        for sh in (0.0, 0.05, -0.05):
            try:
                sol = fsolve(eqs, [s, sh, s], full_output=False)
            except Exception:  # noqa: BLE001
                continue
            res = float(np.max(np.abs(eqs(sol))))
            W = np.array([[sol[0], sol[1]], [sol[1], sol[2]]])
            # 只接受不翻号的解：行列式为负意味着镜像，扫描器不会做这件事。
            if np.linalg.det(W) <= 0 or not np.all(np.isfinite(W)):
                continue
            if res > _AFFINE_RES_TOL:
                continue
            try:
                M = np.linalg.inv(W)
            except Exception:  # noqa: BLE001
                continue
            dist = float(np.linalg.norm(M - np.eye(2)))
            if not any(np.allclose(W, w, rtol=1e-4, atol=1e-9) for _, w in sols):
                sols.append((dist, W))
    if not sols:
        return None, float("inf")
    sols.sort(key=lambda t: t[0])
    best = sols[0][1]
    best_res = float(np.max(np.abs(eqs([best[0, 0], best[0, 1], best[1, 1]]))))
    _LAST_SOLVE.clear()
    _LAST_SOLVE.update({
        "n_branches": len(sols),
        "distances": [round(d, 4) for d, _ in sols[:4]],
        "ambiguous": len(sols) > 1 and sols[1][0] < sols[0][0] * 2.0,
    })
    return best, best_res


def calibrate_from_lattice(image, nm_per_px: float, surface: str = "Au(111)",
                           *, expected_period_nm: float | None = None,
                           max_spread: float = 0.25) -> CalibrationResult:
    """由一帧原子分辨图反推扫描器的仿射畸变。

    ``max_spread``：三个方向周期的相对散布超过它就拒绝定标。散布大意味着要么
    这不是单一晶格（moiré、两个畴、针尖多重像），要么畸变已经大到线性模型撑不住
    —— 两种情况下拟合都会给出一个能算但没意义的数。
    """
    warns: list[str] = []
    d = expected_period_nm or first_order_period_nm(surface)
    if not d:
        return CalibrationResult(reason="unknown_surface", surface=surface)

    lat = find_lattice_peaks(image, nm_per_px)
    if not lat.ok:
        return CalibrationResult(reason=lat.reason, surface=surface,
                                 expected_period_nm=d, warnings=lat.warnings)
    if len(lat.peaks) < 3 or len(lat.periods_nm) < 3:
        return CalibrationResult(reason="need_three_directions", surface=surface,
                                 expected_period_nm=d)
    if not lat.hexagonal and surface not in SQUARE_SURFACES:
        warns.append("六重对称没通过 —— 定标结果按疑处理")
    if lat.period_spread is not None and lat.period_spread > max_spread:
        return CalibrationResult(
            reason="period_spread_too_large", surface=surface,
            expected_period_nm=d, spread_before=lat.period_spread,
            warnings=tuple(list(lat.warnings) + [
                "三个方向周期散布 %.1f%% 超过 %.0f%%：不是单一晶格，或畸变已超出"
                "线性模型" % (100 * lat.period_spread, 100 * max_spread)]))

    ny, nx = np.asarray(image).shape
    span_px = 0.5 * (nx + ny)
    k_ideal = span_px * float(nm_per_px) / d

    # 取功率最强的两个独立方向作为基；第三个由它们的和给出，不额外拟合。
    ps = sorted(lat.peaks, key=lambda p: -p.power)
    basis: list[LatticePeak] = []
    for p in ps:
        if all(abs((p.angle_deg - q.angle_deg) % 180.0) > 15.0 and
               abs((p.angle_deg - q.angle_deg) % 180.0) < 165.0 for q in basis):
            basis.append(p)
        if len(basis) == 2:
            break
    if len(basis) < 2:
        return CalibrationResult(reason="need_two_independent_k", surface=surface,
                                 expected_period_nm=d)
    K1 = np.array([basis[0].kx, basis[0].ky])
    K2 = np.array([basis[1].kx, basis[1].ky])
    # 六角的两个基矢夹角应是 60° 或 120°；若测到的是 60°，取 -K2 让它变成 120°，
    # 这样 K1+K2 才是第三个一阶峰（而不是二阶）。
    cosang = float(K1 @ K2 / (np.linalg.norm(K1) * np.linalg.norm(K2) + 1e-30))
    if cosang > 0:
        K2 = -K2

    W, res = solve_affine(K1, K2, k_ideal)
    if W is None:
        return CalibrationResult(reason="affine_no_solution", surface=surface,
                                 expected_period_nm=d,
                                 spread_before=lat.period_spread)
    M = np.linalg.inv(W)          # 实空间畸变：实际 = M · 标称
    sx = float(np.linalg.norm(M[:, 0]))
    sy = float(np.linalg.norm(M[:, 1]))
    ortho = float(np.degrees(np.arccos(
        np.clip(M[:, 0] @ M[:, 1] / (sx * sy + 1e-30), -1.0, 1.0))))

    u, v = W @ K1, W @ K2
    after = [np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(u + v)]
    spread_after = float((max(after) - min(after)) / np.mean(after))

    if not (0.5 < sx < 2.0 and 0.5 < sy < 2.0):
        warns.append("尺度因子 %.3f / %.3f 落在 0.5–2 之外 —— 多半是晶格常数选错了"
                     "（比如把最近邻距离当成了 FFT 一阶周期）" % (sx, sy))
    warns.append("非正交 %.2f° 里含慢轴热漂移，单帧分不开；换扫描角重测可分离"
                 % (ortho - 90.0))

    return CalibrationResult(
        ok=True, surface=surface, expected_period_nm=d,
        x_scale=sx, y_scale=sy, nonorthogonality_deg=ortho - 90.0,
        lattice_angle_deg=lat.lattice_angle_deg, residual=res,
        spread_before=lat.period_spread, spread_after=spread_after,
        matrix=((float(M[0, 0]), float(M[0, 1])), (float(M[1, 0]), float(M[1, 1]))),
        warnings=tuple(list(lat.warnings) + warns))


__all__ = [
    "SURFACE_LATTICE_NM", "SQUARE_SURFACES", "PEAK_BAND_NM",
    "LatticePeak", "LatticeResult", "CalibrationResult",
    "first_order_period_nm", "find_lattice_peaks", "solve_affine",
    "calibrate_from_lattice", "calibrate_forward_backward", "ForwardBackward",
    "calibrate_up_down",
]


@dataclass(frozen=True)
class ForwardBackward:
    """正反扫合并的定标 —— 一致性检查与平均。**它分不开漂移与压电非正交。**

    ⚠️ 这个类曾经声称能分离，那是错的，写在这里以免有人再推一遍：

    正扫与反扫的快扫方向相反，所以 Nanonis 把 backward **镜像存储**；
    ``io.sxm_oriented_frames`` 会把镜像还原，两帧于是落在同一个实空间坐标系里。
    而在实空间坐标系中，**压电非正交**（几何属性）与**慢轴热漂移**（时间连续、
    两次扫描同向）**都是同号的** —— 差值恒等于零，分不开。
    未定向反扫的镜像会翻转剪切符号；这个符号变化不能用来区分两种物理贡献。

    真正能分开的是**慢轴反向**：``:SCAN_DIR: up`` 与 ``down`` 各扫一帧。
    漂移剪切随慢轴方向翻号，压电非正交不随。见 :func:`calibrate_up_down`。

    所以这里只做两件老实事：**两帧一致性检查**（尺度本该相同，不同就说明有一帧
    的峰找错了）与**平均**（降噪）。
    """

    ok: bool = False
    reason: str = ""
    x_scale: Optional[float] = None
    y_scale: Optional[float] = None
    #: 两帧平均的非正交。**归属未定** —— 压电几何 + 慢轴热漂移之和，
    #: 要拆开必须再扫一帧慢轴反向的（:func:`calibrate_up_down`）。
    shear_deg: Optional[float] = None
    #: 两帧非正交之差。定向正确时它应当 ~0；显著不为零说明定向没做对，
    #: 或者两帧之间发生了别的变化。它是一道**自检**，不是物理量。
    shear_disagreement_deg: Optional[float] = None
    scale_agreement: Optional[float] = None
    forward: Optional[CalibrationResult] = None
    backward: Optional[CalibrationResult] = None
    warnings: tuple[str, ...] = ()


def calibrate_forward_backward(forward, backward, nm_per_px: float,
                               surface: str = "Au(111)",
                               *, max_scale_mismatch: float = 0.02) -> ForwardBackward:
    """正反扫的一致性检查与平均。**传进来的两帧必须已经过 ``sxm_oriented_frames``
    定向**，否则 backward 是镜像的，算出来的剪切符号是假的。

    ``max_scale_mismatch``：两个方向给出的尺度因子相差超过它就不给结论。
    尺度在正反扫上**没有任何理由不同**（它是同一段压电走过的同一段距离），
    真差了说明其中一次的峰找错了，或者帧内漂移大到线性模型不成立。
    """
    f = calibrate_from_lattice(forward, nm_per_px, surface)
    b = calibrate_from_lattice(backward, nm_per_px, surface)
    if not f.ok or not b.ok:
        return ForwardBackward(reason=("forward:%s backward:%s"
                                       % (f.reason or "ok", b.reason or "ok")),
                               forward=f, backward=b)
    dx = abs(f.x_scale - b.x_scale) / max(f.x_scale, 1e-9)
    dy = abs(f.y_scale - b.y_scale) / max(f.y_scale, 1e-9)
    agree = float(max(dx, dy))
    warns: list[str] = []
    if agree > max_scale_mismatch:
        return ForwardBackward(
            reason="scale_mismatch", forward=f, backward=b, scale_agreement=agree,
            warnings=("正反扫的尺度因子相差 %.1f%%（上限 %.0f%%）—— 尺度在两个方向上"
                      "没有理由不同，多半是有一次的峰找错了" % (100 * agree,
                                                              100 * max_scale_mismatch),))
    mean_shear = 0.5 * (f.nonorthogonality_deg + b.nonorthogonality_deg)
    disagree = float(f.nonorthogonality_deg - b.nonorthogonality_deg)
    if abs(disagree) > 0.5:
        warns.append("两帧的剪切差了 %.2f° —— 定向正确时它应当 ~0。检查是不是把"
                     "未定向的 backward 传进来了（镜像会翻转剪切符号）" % disagree)
    warns.append("剪切 %.2f° 的归属未定：压电几何与慢轴热漂移在这两帧上同号，"
                 "分不开。要拆开就再扫一帧慢轴反向的（SCAN_DIR up ↔ down）"
                 % mean_shear)
    return ForwardBackward(
        ok=True, x_scale=0.5 * (f.x_scale + b.x_scale),
        y_scale=0.5 * (f.y_scale + b.y_scale),
        shear_deg=float(mean_shear), shear_disagreement_deg=disagree,
        scale_agreement=agree, forward=f, backward=b,
        warnings=tuple(warns + list(f.warnings)))


def calibrate_up_down(up_frames: tuple, down_frames: tuple, nm_per_px: float,
                      surface: str = "Au(111)") -> dict:
    """用**慢轴反向**的两帧分离压电非正交与热漂移。

    这是唯一便宜且有效的分离办法。参数各是一对 ``(forward, backward)``，都必须
    已经过 ``sxm_oriented_frames`` 定向。

    * 慢轴热漂移的剪切随慢轴方向**翻号**；
    * 压电非正交是几何的，**不随**。

    所以 ``(up + down)/2`` 是压电那一份，``(up − down)/2`` 是漂移那一份。
    """
    u = calibrate_forward_backward(up_frames[0], up_frames[1], nm_per_px, surface)
    d = calibrate_forward_backward(down_frames[0], down_frames[1], nm_per_px, surface)
    if not u.ok or not d.ok:
        return {"ok": False, "reason": "up:%s down:%s" % (u.reason or "ok", d.reason or "ok")}
    piezo = 0.5 * (u.shear_deg + d.shear_deg)
    drift = 0.5 * (u.shear_deg - d.shear_deg)
    return {
        "ok": True,
        "x_scale": 0.5 * (u.x_scale + d.x_scale),
        "y_scale": 0.5 * (u.y_scale + d.y_scale),
        "piezo_nonorthogonality_deg": float(piezo),
        "drift_shear_deg": float(drift),
        "up": u, "down": d,
        "note": ("压电非正交 %.2f° / 热漂移 %.2f°。两者在单一慢轴方向上是同号叠加的，"
                 "只有换慢轴方向才分得开。" % (piezo, drift)),
    }
