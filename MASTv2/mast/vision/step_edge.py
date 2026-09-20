"""找出一帧形貌里最主要的那道台阶边,给出边上的一点与它的方向。

判据本体:纯函数、零 IO、不抛异常、阈值全是参数。技能壳只负责读文件与像素尺度。

**为什么需要它。** 台阶是表面态驻波最干净的散射体(Crommie/Hasegawa 1993 就是在台阶边测的),
可是要沿台阶法向布一排谱,先得知道台阶在哪、朝哪。``MeasureStepHeight`` 走的是 Z 直方图,
回的是**高度**而不是一条线;本模块补的正是这条线。

**做法。** 梯度模最大的那一撮像素 → 用结构张量(倍角平均)定出这些边**共同的方向** → 把候选
像素投到法向上,取最密的那一簇(一帧里常有好几道平行台阶)→ 对这一簇拟合直线,并在它两侧各取
一小段量高度差。找不到相干的梯度脊就回 ``no_step``,那簇不直就回 ``undecidable`` ——
一条编出来的边会让整排谱落在错的地方,而每条谱都会「成功」。

**为什么不去平面再分台面。** 一列规则的台阶在最小二乘意义下**就是**一个斜面:去平面会把楼梯
本身减掉,剩下的锯齿再按高度分类,分出来的不是台面。台阶与斜面的区别不在高度分布,在于台阶是
分段平的、跳变是突然的 —— 那是梯度里的东西。

**角度约定。** ``angle_deg`` 是**图像坐标**里的方向(0° = 快扫轴,角度随行号增大的方向增大),
与 ``assess_herringbone`` 的 ``stripe_angle_deg`` 同一套。``angle_scan_deg`` 是同一条边在
**扫描/样品坐标**里的方向:读图的人把行 0 放在窗口的高 y 边,数组的慢轴因此沿 −y,两者差一个
负号(模 180°)。两个都报,是因为只报一个的那一次,用错的人不会发现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

#: 边候选的门限:梯度中位数之上这么多个稳健 σ。用「比典型陡多少」而不是固定分位数 ——
#: 一道边在一帧里往往只占 1% 的像素,取「最陡的 4%」会把门限压进噪声里。
EDGE_SIGMA = 6.0
#: 边界像素到拟合直线的 RMS 距离上限,以帧短边为单位。超过 = 这不是一条直边
MAX_STRAIGHTNESS = 0.06
#: 边界像素至少要有这么多
MIN_EDGE_PIXELS = 24
#: 候选像素的梯度至少要有中位梯度的这么多倍,否则那只是斜面或噪声
MIN_SEPARATION_SIGMA = 4.0


@dataclass(frozen=True)
class StepEdgeResult:
    """step_edge / no_step / undecidable,外加这条边是什么样的。"""

    verdict: str
    #: 边上的一点(图像坐标,像素)
    x_px: float | None = None
    y_px: float | None = None
    #: 图像坐标里的方向(度,模 180)
    angle_deg: float | None = None
    #: 同一条边在扫描坐标里的方向(度,模 180)
    angle_scan_deg: float | None = None
    step_height_m: float | None = None
    straightness: float | None = None
    n_edge_px: int = 0
    upper_fraction: float | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "step_edge"


def _plane_removed(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ny, nx = a.shape
    i, j = np.mgrid[0:ny, 0:nx]
    m = np.c_[i.ravel(), j.ravel(), np.ones(i.size)]
    good = np.isfinite(a).ravel()
    if good.sum() < 8:
        return np.zeros_like(a)
    coef, *_ = np.linalg.lstsq(m[good], a.ravel()[good], rcond=None)
    return (np.nan_to_num(a).ravel() - m @ coef).reshape(a.shape)


def _box(a: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    """n x n 均值平滑,用累积和做,不引依赖。"""
    if n <= 1:
        return a
    pad = n // 2
    b = np.pad(a, pad, mode="edge")
    c = np.cumsum(np.cumsum(b, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    ny, nx = a.shape
    out = (c[n:n + ny, n:n + nx] - c[0:ny, n:n + nx]
           - c[n:n + ny, 0:nx] + c[0:ny, 0:nx])
    return out / float(n * n)


def _dominant_direction(gx: npt.NDArray[np.float64], gy: npt.NDArray[np.float64],
                        w: npt.NDArray[np.float64]) -> tuple[float, float]:
    """边的方向(单位向量),由梯度的**倍角**平均得到。

    梯度在一道边的两侧方向相反,直接平均会互相抵消;倍角平均把 +g 与 −g 当成同一个方向,
    这正是「无向的线」该有的算法。"""
    a2 = np.sum(w * (gx * gx - gy * gy))
    b2 = np.sum(w * 2.0 * gx * gy)
    th_g = 0.5 * math.atan2(b2, a2)          # mean gradient direction, mod pi
    return -math.sin(th_g), math.cos(th_g)   # the edge runs perpendicular to it


def locate_step_edge(image_m: npt.ArrayLike, *,
                     edge_sigma: float = EDGE_SIGMA,
                     max_straightness: float = MAX_STRAIGHTNESS,
                     min_edge_pixels: int = MIN_EDGE_PIXELS,
                     min_separation_sigma: float = MIN_SEPARATION_SIGMA) -> StepEdgeResult:
    """一帧形貌里最主要的那道台阶边(像素坐标)。"""
    a = np.asarray(image_m, dtype=float)
    if a.ndim != 2 or min(a.shape) < 32:
        return StepEdgeResult("undecidable", reasons=("frame_too_small",))
    if not np.isfinite(a).any():
        return StepEdgeResult("undecidable", reasons=("frame_all_nan",))
    z = _box(np.where(np.isfinite(a), a, np.nanmean(a)), 3)
    gy, gx = np.gradient(z)
    g = np.hypot(gx, gy)
    med = float(np.median(g))
    mad = 1.4826 * float(np.median(np.abs(g - med)))
    thr = med + float(edge_sigma) * max(mad, 1e-30)
    cand = g >= thr
    if int(cand.sum()) < int(min_edge_pixels):
        return StepEdgeResult("no_step", n_edge_px=int(cand.sum()),
                              reasons=("no_gradient_ridge",))
    sigma = med
    if float(np.median(g[cand])) < float(min_separation_sigma) * max(sigma, 1e-30):
        # the strongest "edge" is barely above the typical slope: this is a tilt, not a step
        return StepEdgeResult("no_step", n_edge_px=int(cand.sum()),
                              reasons=("terraces_not_separated",),
                              notes={"ridge_over_median_gradient": thr / max(sigma, 1e-30)})

    ex, ey = _dominant_direction(gx[cand], gy[cand], g[cand])
    ii, jj = np.nonzero(cand)
    # project onto the normal; parallel steps separate into clusters along it
    proj = (jj - jj.mean()) * (-ey) + (ii - ii.mean()) * ex
    span = float(proj.max() - proj.min()) or 1.0
    nb = max(8, int(span / 3.0))
    hist, edges = np.histogram(proj, bins=nb)
    k = int(np.argmax(hist))
    lo, hi = edges[k], edges[k + 1]
    pad = 0.5 * (hi - lo)
    keep = (proj >= lo - pad) & (proj <= hi + pad)
    ii, jj = ii[keep], jj[keep]
    if ii.size < int(min_edge_pixels):
        return StepEdgeResult("undecidable", n_edge_px=int(ii.size),
                              reasons=("too_few_edge_pixels",))

    # the height across it: the mean of the raw frame a few pixels either side
    y0, x0 = float(ii.mean()), float(jj.mean())
    off = max(4, int(0.05 * min(a.shape)))
    ny, nx = a.shape
    hi_i = np.clip(np.round(ii + ex * off).astype(int), 0, ny - 1)
    hi_j = np.clip(np.round(jj - ey * off).astype(int), 0, nx - 1)
    lo_i = np.clip(np.round(ii - ex * off).astype(int), 0, ny - 1)
    lo_j = np.clip(np.round(jj + ey * off).astype(int), 0, nx - 1)
    sep = float(abs(np.nanmean(a[hi_i, hi_j]) - np.nanmean(a[lo_i, lo_j])))
    frac = float(cand.mean())

    # total least squares: the direction is the principal axis of the boundary pixels
    y0, x0 = float(ii.mean()), float(jj.mean())
    pts = np.c_[jj - x0, ii - y0]
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    dx, dy = float(vt[0, 0]), float(vt[0, 1])
    resid = pts @ np.array([-dy, dx])
    straight = float(np.sqrt(np.mean(resid ** 2))) / float(min(a.shape))
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    res = StepEdgeResult(
        "step_edge", x_px=x0, y_px=y0, angle_deg=angle,
        angle_scan_deg=(-angle) % 180.0, step_height_m=sep,
        straightness=straight, n_edge_px=int(ii.size), upper_fraction=frac,
        notes={"ridge_gradient_threshold": thr, "median_gradient": sigma,
               "edge_fraction": frac})
    if straight > float(max_straightness):
        return StepEdgeResult("undecidable", x_px=x0, y_px=y0, angle_deg=angle,
                              angle_scan_deg=(-angle) % 180.0, step_height_m=sep,
                              straightness=straight, n_edge_px=int(ii.size),
                              upper_fraction=frac, reasons=("edge_not_straight",),
                              notes=res.notes)
    return res
