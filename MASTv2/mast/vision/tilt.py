"""表面倾斜与台阶主导判据 —— 自动调平的判断原语。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

在这个模块之前,全仓库没有任何函数能回答「这个表面倾斜了多少度」。有四份平面
拟合代码,三份把系数算完就丢掉,只有 ``ransac_plane_subtract`` 把系数放进了返回
值 —— 但那是 **z 单位 / 像素**,换成物理角度要除以像素的物理尺寸,而这一步全仓库
没有。

## 两个必须说清楚的物理限制

**一、基于扫图算平面,只有快扫方向是准的。**

一帧 512 线、每线 2 秒的图要扫 34 分钟。沿**慢扫轴**,图像顶部和底部相隔半小时,
这段时间里的热漂移会原样表现为一个视在倾斜 —— 和真实的样品倾斜无法区分。沿**快扫
轴**,一条线只用一两秒,漂移可忽略。

所以帧法测出来的 ``slope_slow`` 是「真实倾斜 + 漂移」,不是倾斜。本模块**如实
标注**这一点(``slow_axis_trusted=False``),不假装两个方向一样可信。这也正是
Nanonis 的 SmarTilt 用恒流跑一个内接圆而不是拟合整帧的原因:圆在几秒内跑完,两个
方向是在同一个时间尺度上测的(见 :mod:`mast.skills.builtins.tilt_probe`)。

**二、台阶密集区不能拿倾斜当判据。**

跨台阶拟合测到的是「包络 + 台阶采样噪声」的混合物。2026-07-28 的审计里,一个只含
单个 240 pm 台阶的合成面把 y 方向拟合斜率带偏到真值的 13 倍。所以任何倾斜测量都
必须先过台阶否决。

两个台阶判据的盲区**正好互补,必须一起用**:

  * 多尺度结构主导比 —— 固定 32 px 分块在台面宽度小于分块尺寸时会塌回 1.00,与
    纯平表面无法区分(而且是往「看起来很干净」的方向失效)。扫多个尺度取最大值
    才能覆盖密集台阶。
  * 分割器的 STEP 占比 —— 看不见**完全平行于快扫轴**的台阶(逐行中位数差分对齐
    会把纯 y 向阶梯整个吸收掉)。

前者不做行对齐,水平台阶抓得到;后者在密集区抓得到。单用任一个都会在某个真实场景
里静默失败。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── 常量(物理常数级:改动 = 改代码 + 重新验证) ──────────────────────────────

#: 结构主导比的触发阈。
#:
#: **实测标定(2026-07-30,物理合成:15.4 pm 噪声 / 240 pm 台阶)**。判据 =
#: 二阶去趋势 + 多尺度分块,统计量是各尺度比值的最大值。
#:
#:   纯高斯噪声   270 次(3 种帧尺寸 × 3 种噪声幅度 × 30 seed)最大 **1.0664**
#:   纯倾斜面     1.0611
#:   压电弯曲     1.061(4 倍强曲率也是 1.061 —— 二阶去趋势吃掉了它)
#:   台面 ≥16 px  ≥ 3.02
#:   台面 8 px    1.653
#:   台面 ≤5 px   1.06  ← 与噪声无法区分(见下面的硬限制)
#:
#: 1.4 落在实测误报天花板(1.0664)之上 31%、最弱真信号(1.653)之下 16%。
#:
#: 为什么是二阶去趋势:只扣一阶平面时,压电弯曲会把比值抬到 **1.808** —— 高于
#: 台面 8 px 的密集台阶(1.717)。也就是说在一阶下**不存在**任何阈值能既抓住密集
#: 台阶又不把压电弯曲误判成台阶。二次面能拟合弯曲、拟合不了阶梯,这一步把两者
#: 彻底分开。
STRUCTURE_RATIO_THRESHOLD = 1.4

#: **硬限制**:台面宽度小于约 6 个像素时,本判据与纯噪声无法区分(实测 1.06)。
#: 分块再小也没用 —— 分块必须装得下若干个像素才能估出局部 σ,而那个尺寸已经跨过
#: 台阶了。这个区间由分割器那一路(KDE 台面层分析)负责,这也正是两个判据必须取
#: 「或」的原因之一。以 100 nm / 256 px 的帧算,6 px ≈ 2.3 nm 台面宽。
DOMINANCE_MIN_TERRACE_PX = 6

#: 多尺度扫描用的分块边长(px)。固定 32 只在台面宽度 > 32 px 时起得来;台面更窄
#: 时要更小的分块才看得见台阶。实测的 tile × 台面宽度二维表显示比值只在
#: 「台面宽度 > 分块尺寸」时抬得起来,所以覆盖一整排尺度。
DOMINANCE_TILES = (4, 8, 16, 32, 64)

#: MAD → σ 的一致性因子(高斯分布下 σ = 1.4826 × MAD)。
MAD_TO_SIGMA = 1.4826

#: RANSAC 的内点阈 = 这个倍数 × 噪声底。3σ 覆盖 99.7% 的高斯噪声。
RANSAC_SIGMA_MULT = 3.0

#: RANSAC 迭代次数与最少内点比。
RANSAC_TRIALS = 200
MIN_INLIER_RATIO = 0.5

#: 帧短边小于这个像素数时不做拟合 —— 点太少,拟合出来的斜率没有意义。
MIN_FRAME_PX = 64

#: NaN 占比超过这个值就判无效(实时扫描中未采集的行是 NaN)。
MAX_NAN_FRAC = 0.20


@dataclass
class StepVerdict:
    """台阶主导判据的结果。"""

    step_dominated: bool
    ratio_multiscale: float
    ratio_by_tile: dict[int, float] = field(default_factory=dict)
    step_area_frac: float = 0.0
    step_present: bool = False
    #: 哪个判据触发的(诊断用):dominance / segmentation / both / none
    triggered_by: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_dominated": self.step_dominated,
            "ratio_multiscale": self.ratio_multiscale,
            "ratio_by_tile": dict(self.ratio_by_tile),
            "step_area_frac": self.step_area_frac,
            "step_present": self.step_present,
            "triggered_by": self.triggered_by,
        }


@dataclass
class TiltEstimate:
    """一帧图的倾斜估计。

    ``valid`` 为 False 时其余数值**不可用于调平决策** —— ``invalid_reason``
    说明为什么。这是刻意做成显式字段而不是「返回 None」的:调用方拿到一个带
    原因的结构,能把「为什么不调平」如实报告给用户。
    """

    valid: bool
    invalid_reason: str = ""
    #: 压电坐标系下的倾斜角(度)。scan_angle 已旋转。
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    #: 帧坐标系下的原始值(诊断用)。fast = 快扫轴(列方向),slow = 慢扫轴(行方向)。
    tilt_fast_deg: float = 0.0
    tilt_slow_deg: float = 0.0
    #: 合成倾斜幅度(度)与它在整帧对角线上吃掉的 Z 量程(米)。
    slope_mag_deg: float = 0.0
    z_span_m: float = 0.0
    #: **慢扫轴的斜率不可信** —— 它混着整帧时长内的热漂移。帧法恒为 False。
    slow_axis_trusted: bool = False
    inlier_ratio: float = 0.0
    noise_floor_m: float = 0.0
    #: scan_angle ≠ 0 时为 True:帧→压电的旋转用了未在真机上核实的方向约定。
    rotation_applied: bool = False
    step: "StepVerdict | None" = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "tilt_x_deg": self.tilt_x_deg,
            "tilt_y_deg": self.tilt_y_deg,
            "tilt_fast_deg": self.tilt_fast_deg,
            "tilt_slow_deg": self.tilt_slow_deg,
            "slope_mag_deg": self.slope_mag_deg,
            "z_span_m": self.z_span_m,
            "slow_axis_trusted": self.slow_axis_trusted,
            "inlier_ratio": self.inlier_ratio,
            "noise_floor_m": self.noise_floor_m,
            "rotation_applied": self.rotation_applied,
        }
        if self.step is not None:
            out["step"] = self.step.as_dict()
        return out


# ── 噪声底 ────────────────────────────────────────────────────────────────────

def noise_floor(image: np.ndarray) -> float:
    """从行内相邻像素差分估计噪声底(与 image 同单位)。

    用**行内差分**而不是整帧统计:差分把任何低频成分(倾斜、台阶包络、曲率)
    差掉,剩下的才是逐点噪声。除以 √2 是因为两个独立噪声样本相减方差翻倍。

    用 MAD 而不是标准差:台阶跨过的那些像素在差分里是巨大的离群值,标准差会被
    它们带走,中位数不会。
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return 0.0
    diffs = np.diff(arr, axis=1)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    return mad * MAD_TO_SIGMA / math.sqrt(2.0)


# ── 平面拟合 ──────────────────────────────────────────────────────────────────

def _lstsq_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """最小二乘拟合 z = a·x + b·y + c,返回 (a, b, c) 或 None。"""
    if z.size < 3:
        return None
    design = np.column_stack([x, y, np.ones_like(x)])
    try:
        coeff, *_ = np.linalg.lstsq(design, z, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coeff)):
        return None
    return float(coeff[0]), float(coeff[1]), float(coeff[2])


def fit_plane_robust(image: np.ndarray, *, sigma: float | None = None,
                     trials: int = RANSAC_TRIALS, seed: int = 42):
    """噪声自适应的稳健平面拟合。返回 ``(a, b, c, inlier_ratio)`` 或 None。

    系数单位是 **z 单位 / 像素**(换物理角度要再除以像素的物理尺寸)。

    与既有的 ``ransac_plane_subtract`` 的关键差别是内点阈:那里写死了
    ``residual_threshold=1e-10``(100 pm)。在原子级平整的表面上 100 pm 比整个
    高度起伏还大,于是几乎所有点都算内点,RANSAC 退化成普通最小二乘,稳健性
    白给;而在起伏大的表面上它又太小,内点少到拟合不稳。这里的阈值随图自适应:
    3 × 噪声底。
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        return None
    ny, nx = arr.shape
    gy, gx = np.mgrid[:ny, :nx]
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        return None

    x = gx[finite].astype(np.float64)
    y = gy[finite].astype(np.float64)
    z = arr[finite]

    sig = noise_floor(arr) if sigma is None else float(sigma)
    # 噪声底为 0(合成的无噪数据 / 常数面)时退回一个由高度尺度导出的小阈值,
    # 否则内点判据 |r| < 0 永远不成立,RANSAC 一个内点都找不到。
    if not np.isfinite(sig) or sig <= 0:
        spread = float(np.nanmax(z) - np.nanmin(z)) if z.size else 0.0
        sig = spread * 1e-6 if spread > 0 else 1.0
    thresh = RANSAC_SIGMA_MULT * sig

    rng = np.random.default_rng(seed)
    n = z.size
    best_plane = None
    best_count = -1
    for _ in range(trials):
        idx = rng.choice(n, size=3, replace=False)
        design = np.column_stack([x[idx], y[idx], np.ones(3)])
        try:
            plane = np.linalg.solve(design, z[idx])
        except np.linalg.LinAlgError:
            continue
        resid = np.abs(z - (plane[0] * x + plane[1] * y + plane[2]))
        count = int(np.count_nonzero(resid < thresh))
        if count > best_count:
            best_count = count
            best_plane = plane

    if best_plane is None:
        fit = _lstsq_plane(x, y, z)
        return None if fit is None else (*fit, 1.0)

    # 用全部内点重拟合一次(RANSAC 的三点解只是用来找内点集的)。
    resid = np.abs(z - (best_plane[0] * x + best_plane[1] * y + best_plane[2]))
    inliers = resid < thresh
    if int(inliers.sum()) >= 3:
        refined = _lstsq_plane(x[inliers], y[inliers], z[inliers])
        if refined is not None:
            return (*refined, float(inliers.sum()) / float(n))
    return (float(best_plane[0]), float(best_plane[1]), float(best_plane[2]),
            float(max(best_count, 0)) / float(n))


def plane_subtract(image: np.ndarray) -> np.ndarray:
    """扣掉一阶平面。拟合失败时原样返回。"""
    arr = np.asarray(image, dtype=np.float64)
    fit = fit_plane_robust(arr)
    if fit is None:
        return arr
    a, b, c, _ = fit
    ny, nx = arr.shape
    gy, gx = np.mgrid[:ny, :nx]
    return arr - (a * gx + b * gy + c)


def detrend_quadratic(image: np.ndarray) -> np.ndarray:
    """扣掉二阶曲面(给结构主导判据做前处理)。拟合失败时退回一阶。

    为什么台阶判据前面要扣二次面而不是平面:压电扫描管的弯曲是二次的,只扣平面
    时它会在残差里留下一个大尺度起伏,把结构主导比抬到 **1.808** —— 高于台面
    8 px 的密集台阶(1.717)。在那种情况下**不存在**任何阈值能既抓住密集台阶又
    不把弯曲误判成台阶。

    二次面能拟合弯曲、拟合不了阶梯,所以这一步只吃掉曲率:实测 4 倍强曲率也被
    压回噪声底 1.061,而台阶信号几乎不动(8 台阶 4.29→4.11、16 台阶 3.10→3.01)。

    用普通最小二乘而不是稳健拟合:这里的目的是**移除低频背景**,不是估计一个要
    拿去调硬件的物理量;台阶把二次拟合带偏一点,残差里仍然留着台阶,判据照样
    看得见。
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        return arr
    ny, nx = arr.shape
    gy, gx = np.mgrid[:ny, :nx]
    finite = np.isfinite(arr)
    if finite.sum() < 6:
        return arr
    x = gx[finite].astype(np.float64)
    y = gy[finite].astype(np.float64)
    z = arr[finite]
    design = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    try:
        coeff, *_ = np.linalg.lstsq(design, z, rcond=None)
    except np.linalg.LinAlgError:
        return plane_subtract(arr)
    if not np.all(np.isfinite(coeff)):
        return plane_subtract(arr)
    full = np.column_stack([
        (gx * gx).ravel(), (gy * gy).ravel(), (gx * gy).ravel(),
        gx.ravel(), gy.ravel(), np.ones(arr.size),
    ])
    return (arr.ravel() - full @ coeff).reshape(ny, nx)


# ── 台阶主导判据 ──────────────────────────────────────────────────────────────

def structure_dominance(flat: np.ndarray, tile: int) -> float:
    """单一尺度的结构主导比 = 整帧 σ / 分块 σ 的中位数。

    分块 σ 取**中位数**:多数分块只看到局部纹理(晶格 + 噪声),中位数对少数几个
    跨台阶的分块免疫。比值说明高度起伏里有多少是大尺度结构。
    """
    arr = np.asarray(flat, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    g = float(finite.std())
    h, w = arr.shape
    ny, nx = h // tile, w // tile
    if ny < 2 or nx < 2:
        return 1.0
    stds = []
    for i in range(ny):
        for j in range(nx):
            block = arr[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
            block = block[np.isfinite(block)]
            if block.size >= 2:
                stds.append(float(block.std()))
    if not stds:
        return 1.0
    local = float(np.median(stds))
    return g / local if local > 0 else 1.0


def step_dominance_multiscale(flat: np.ndarray) -> tuple[float, dict[int, float]]:
    """多尺度结构主导比:扫一排分块尺度,取最大值。

    单一 32 px 分块有个往「看起来很干净」方向失效的盲区:台面宽度小于分块尺寸时,
    每个分块内部都含台阶,局部 σ 追平整帧 σ,比值塌回 1.00 —— 与纯平表面无法区分。
    而那正是要识别的密集台阶区。实测(512²、100 nm 视野、台阶高 240 pm):

        分块    台面宽=128  =64   =32   =16    =8
          8        4.39    4.05  3.10  1.89  1.00
         16        3.97    3.07  1.88  1.00  1.00
         32        3.08    1.88  1.00  1.00  1.00   ← 单尺度的现役参数
         64        1.91    1.01  1.00  1.00  1.00

    比值只在**台面宽度 > 分块尺寸**时抬得起来,所以要覆盖一整排尺度。
    """
    arr = np.asarray(flat, dtype=np.float64)
    short_side = min(arr.shape) if arr.ndim == 2 else 0
    by_tile: dict[int, float] = {}
    for tile in DOMINANCE_TILES:
        if tile * 2 > short_side:
            continue
        by_tile[tile] = structure_dominance(arr, tile)
    if not by_tile:
        return 1.0, {}
    return max(by_tile.values()), by_tile


def _segmentation_step_signal(image: np.ndarray,
                              nm_per_px: float | None) -> tuple[bool, float]:
    """经典分割给出的 (STEP 是否存在, STEP 面积占比)。

    分割器不可用(缺依赖 / 图太小 / 抛异常)时返回 ``(False, 0.0)`` —— 判据组合
    是「或」,所以缺这一路只会让判据**更宽松**,而另一路(多尺度主导比)仍在。
    这是刻意的:一个分析组件坏掉不该让调平例程整个失效。
    """
    try:
        from mast.vision.seg_scale_adaptive import (
            segment_scale_adaptive,
            summarize_segmentation,
        )
        seg, _info = segment_scale_adaptive(
            np.asarray(image, dtype=np.float64), nm_per_px=nm_per_px)
        summary = summarize_segmentation(seg, nm_per_px or 1.0)
        step = summary.get("STEP", {}) if isinstance(summary, dict) else {}
        return bool(step.get("present", False)), float(step.get("area_frac", 0.0))
    except Exception as exc:  # noqa: BLE001 - 分割坏掉不能让调平失效
        logger.debug("台阶分割不可用(判据退回单路): %s", exc)
        return False, 0.0


def assess_steps(image: np.ndarray, *, nm_per_px: float | None = None,
                 use_segmentation: bool = True) -> StepVerdict:
    """台阶是否主导这幅图的高度起伏(= 倾斜拟合是否可信)。

    两个判据取**或**:盲区互补,单用任一个都会在某个真实场景里静默失败。
    """
    # 二阶去趋势:压电弯曲必须先扣掉,否则它会伪装成台阶(见 detrend_quadratic)。
    flat = detrend_quadratic(image)
    ratio, by_tile = step_dominance_multiscale(flat)
    dominance_hit = ratio >= STRUCTURE_RATIO_THRESHOLD

    seg_hit, area = (False, 0.0)
    if use_segmentation:
        seg_hit, area = _segmentation_step_signal(image, nm_per_px)

    if dominance_hit and seg_hit:
        trigger = "both"
    elif dominance_hit:
        trigger = "dominance"
    elif seg_hit:
        trigger = "segmentation"
    else:
        trigger = "none"

    return StepVerdict(
        step_dominated=bool(dominance_hit or seg_hit),
        ratio_multiscale=ratio,
        ratio_by_tile=by_tile,
        step_area_frac=area,
        step_present=seg_hit,
        triggered_by=trigger,
    )


# ── 倾斜估计 ──────────────────────────────────────────────────────────────────

def _rotate_slope(slope_fast: float, slope_slow: float, angle_deg: float):
    """帧坐标系的斜率向量 → 压电坐标系。

    扫描框相对压电坐标系转了 ``angle_deg``,斜率向量随之转回去。**符号约定未在
    真机上核实** —— 所以调平例程自己扫的探察帧一律用 ``scan_angle=0``,把这个
    未知量从闭环里彻底消掉;这里的旋转只用于解释既有的、非零角度的帧。
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (slope_fast * cos_t - slope_slow * sin_t,
            slope_fast * sin_t + slope_slow * cos_t)


def estimate_tilt(
    image: np.ndarray,
    *,
    width_m: float,
    height_m: float,
    scan_angle_deg: float = 0.0,
    nm_per_px: float | None = None,
    check_steps: bool = True,
) -> TiltEstimate:
    """从一帧高度图估计表面倾斜(物理角度,压电坐标系)。

    ``width_m`` / ``height_m`` 是**帧的物理尺寸**,用来把「z 单位 / 像素」的拟合
    系数换成无量纲斜率再取反正切。少了这一步,系数是没有物理意义的数字 —— 而这
    正是既有的 ``ransac_plane_subtract`` 停下来的地方。

    角度换算必须在 Python 里做:声明式工作流的表达式白名单没有 ``atan`` /
    ``degrees``(``**0.5`` 能开方,反三角算不了),所以这一步不可能写在 spec 里。
    """
    arr = np.asarray(image, dtype=np.float64)

    if arr.ndim != 2:
        return TiltEstimate(valid=False, invalid_reason="frame_not_2d")
    ny, nx = arr.shape
    if min(ny, nx) < MIN_FRAME_PX:
        return TiltEstimate(valid=False, invalid_reason="frame_too_small")

    finite = np.isfinite(arr)
    nan_frac = 1.0 - float(finite.sum()) / float(arr.size)
    if nan_frac > MAX_NAN_FRAC:
        return TiltEstimate(valid=False, invalid_reason="too_many_nan")

    if not (width_m > 0 and height_m > 0):
        return TiltEstimate(valid=False, invalid_reason="geometry_missing")

    sigma = noise_floor(arr)

    step = assess_steps(arr, nm_per_px=nm_per_px) if check_steps else None
    if step is not None and step.step_dominated:
        # 台阶主导时拒绝给出倾斜数字。给一个「大概齐」的角度比不给更糟:下游会
        # 拿它去调硬件,而 2026-07-28 的审计实例里这个数字偏了 13 倍。
        return TiltEstimate(
            valid=False, invalid_reason="step_dense",
            noise_floor_m=sigma, step=step,
        )

    fit = fit_plane_robust(arr, sigma=sigma)
    if fit is None:
        return TiltEstimate(valid=False, invalid_reason="fit_failed",
                            noise_floor_m=sigma, step=step)
    a, b, _c, inlier_ratio = fit

    if inlier_ratio < MIN_INLIER_RATIO:
        return TiltEstimate(
            valid=False, invalid_reason="low_inliers",
            inlier_ratio=inlier_ratio, noise_floor_m=sigma, step=step,
        )

    # 系数 → 无量纲斜率:除以像素的物理边长。
    # 列方向(x/nx)是快扫轴,行方向(y/ny)是慢扫轴。
    m_per_px_x = float(width_m) / float(nx)
    m_per_px_y = float(height_m) / float(ny)
    slope_fast = a / m_per_px_x
    slope_slow = b / m_per_px_y

    slope_x, slope_y = (slope_fast, slope_slow)
    rotated = abs(float(scan_angle_deg)) > 1e-9
    if rotated:
        slope_x, slope_y = _rotate_slope(slope_fast, slope_slow,
                                         float(scan_angle_deg))

    mag = math.hypot(slope_x, slope_y)
    diag_m = math.hypot(float(width_m), float(height_m))

    return TiltEstimate(
        valid=True,
        tilt_x_deg=math.degrees(math.atan(slope_x)),
        tilt_y_deg=math.degrees(math.atan(slope_y)),
        tilt_fast_deg=math.degrees(math.atan(slope_fast)),
        tilt_slow_deg=math.degrees(math.atan(slope_slow)),
        slope_mag_deg=math.degrees(math.atan(mag)),
        z_span_m=diag_m * mag,
        # 帧法**恒为 False**:慢扫轴上图像顶部与底部相隔整帧时长,那段时间里的
        # 热漂移与真实倾斜无法区分。要两个方向都可信,用恒流内接圆(TiltProbeCircle)。
        slow_axis_trusted=False,
        inlier_ratio=inlier_ratio,
        noise_floor_m=sigma,
        rotation_applied=rotated,
        step=step,
    )


@dataclass
class CircleTilt:
    """恒流内接圆测量的拟合结果(:mod:`mast.skills.builtins.tilt_probe` 用)。"""

    valid: bool
    invalid_reason: str = ""
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    slope_mag_deg: float = 0.0
    #: 下坡方向(度,从 +x 轴逆时针)。
    downhill_deg: float = 0.0
    #: 拟合残差诊断。**否决用的是后两个**,``residual_ratio``(RMS 比)分辨力
    #: 太低,只作展示 —— 台阶的基频会被正弦拟合吸收成假倾斜,RMS 里只剩高次谐波。
    residual_rms_m: float = 0.0
    residual_ratio: float = 0.0
    max_residual_ratio: float = 0.0
    max_jump_ratio: float = 0.0
    #: 由圆闭合差估出的线性漂移速率(米/秒),已从数据里扣除。
    drift_rate_m_s: float = 0.0
    n_points: int = 0
    #: **两个方向同样可信** —— 整圈在几秒内跑完,不像帧法那样慢轴混着漂移。
    slow_axis_trusted: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "tilt_x_deg": self.tilt_x_deg,
            "tilt_y_deg": self.tilt_y_deg,
            "slope_mag_deg": self.slope_mag_deg,
            "downhill_deg": self.downhill_deg,
            "residual_rms_m": self.residual_rms_m,
            "residual_ratio": self.residual_ratio,
            "max_residual_ratio": self.max_residual_ratio,
            "max_jump_ratio": self.max_jump_ratio,
            "drift_rate_m_s": self.drift_rate_m_s,
            "n_points": self.n_points,
            "slow_axis_trusted": self.slow_axis_trusted,
        }


# ── 圆上的台阶否决判据(物理合成评测) ────────────────────────────────
#
# **残差 RMS 几乎没有分辨力,不能用它当主判据。** 原因是台阶的基频分量会被正弦
# 拟合**吸收成一个假倾斜**:半圈抬高 h 的方波,其基频振幅是 0.64h,剩给残差的只有
# 高次谐波。实测(24 点、15.4 pm 噪声、240 pm 原子台阶):
#
#                        残差RMS/σ   max|残差|/σ   最大相邻跳变/σ
#     平面(40 seed 最大)     1.20        3.56          5.18
#     120 pm 台阶(最小)      1.31        2.77          4.42   ← 半个原子台阶,测不出
#     240 pm 台阶(最小)      2.55        6.00         11.14
#     500 pm 污染物(最小)     5.95       26.81         32.01
#
# 所以主判据是**残差里的相邻点跳变**:台阶是一个不连续,而正弦拟合无论如何都
# 消不掉不连续。它在 240 pm 台阶上给出 2.1 倍余量,RMS 只有 2.1 倍中的一点点。
#
# 已知极限:**半个原子台阶(~120 pm)穿圆是测不出来的** —— 两个判据都落在平面的
# 涨落带里。真要防这一档,得靠更低的噪声或更多的点。
CIRCLE_MAX_RESIDUAL_SIGMA = 4.5     # 平面实测最大 3.56
CIRCLE_MAX_JUMP_SIGMA = 7.0         # 平面实测最大 5.18
#: 残差 RMS 只作诊断展示,不参与否决(分辨力太低)。
CIRCLE_RESIDUAL_MAX_RATIO = 4.0

#: 至少要几个点才拟合(3 个参数 A/B/C,冗余度太低的拟合没有残差可言)。
CIRCLE_MIN_POINTS = 8


def circle_tilt_resolution_deg(noise_floor_m: float, radius_m: float,
                               n_points: int) -> float:
    """圆法能分辨的最小倾斜角(度,1σ)= σ_z·√(2/n) / r。

    这是个**硬的测量下限**,与算法无关:要测更小的倾斜,只能加大半径、加密取点
    或降低噪声。实测(15.4 pm 噪声、20 nm 半径、24 点)1σ ≈ 0.013°,200 次里的
    最大偏差 0.068°。

    调平例程的验收阈必须留在这个分辨率之上 —— 否则「残余倾斜没达标」只是在追噪声。
    """
    if not (radius_m > 0 and n_points > 0 and noise_floor_m > 0):
        return 0.0
    return math.degrees(
        float(noise_floor_m) * math.sqrt(2.0 / float(n_points)) / float(radius_m))


def fit_circle_tilt(
    angles_rad: "list[float] | np.ndarray",
    z_values: "list[float] | np.ndarray",
    radius_m: float,
    *,
    times_s: "list[float] | np.ndarray | None" = None,
    noise_floor_m: float = 0.0,
) -> CircleTilt:
    """把恒流内接圆上的 Z(θ) 拟合成倾斜。

    这是 Nanonis SmarTilt 做法的自研版:反馈开着让针尖沿一个圆走一圈,恒流下 Z
    跟随表面,于是

        Z(θ) = slope_x · r·cos θ + slope_y · r·sin θ + C

    对 (A, B, C) 是**线性**的,一次最小二乘就解出来,不需要迭代。

    **为什么用圆而不是拟合整帧**:整圈在几秒内跑完,两个方向是在同一个时间尺度
    上测的;而一帧图的慢扫轴跨越整帧时长(几十分钟),那段时间的热漂移与真实倾斜
    无法区分(见 :func:`estimate_tilt` 的 ``slow_axis_trusted``)。

    ``times_s`` 给出每个点的采样时刻时,顺带拟合一个**线性漂移项**并扣除 ——
    圆虽然跑得快,但几秒里仍可能漂几十皮米,而漂移在圆上表现为一个与 θ 无关、
    与时间线性相关的分量,正好可以从倾斜里分离出来。这就是「圆闭合差」的严格版。
    """
    theta = np.asarray(angles_rad, dtype=np.float64)
    z = np.asarray(z_values, dtype=np.float64)
    if theta.shape != z.shape or theta.ndim != 1:
        return CircleTilt(valid=False, invalid_reason="shape_mismatch")

    finite = np.isfinite(theta) & np.isfinite(z)
    theta, z = theta[finite], z[finite]
    n = int(theta.size)
    if n < CIRCLE_MIN_POINTS:
        return CircleTilt(valid=False, invalid_reason="too_few_points",
                          n_points=n)
    if not (radius_m > 0):
        return CircleTilt(valid=False, invalid_reason="bad_radius", n_points=n)

    cols = [np.cos(theta), np.sin(theta), np.ones(n)]
    drift_col = None
    if times_s is not None:
        t = np.asarray(times_s, dtype=np.float64)[finite]
        if t.size == n and np.isfinite(t).all() and float(t.max() - t.min()) > 0:
            drift_col = t - float(t.mean())
            cols.append(drift_col)

    design = np.column_stack(cols)
    try:
        coeff, *_ = np.linalg.lstsq(design, z, rcond=None)
    except np.linalg.LinAlgError:
        return CircleTilt(valid=False, invalid_reason="fit_failed", n_points=n)
    if not np.all(np.isfinite(coeff)):
        return CircleTilt(valid=False, invalid_reason="fit_failed", n_points=n)

    a, b = float(coeff[0]), float(coeff[1])
    drift_rate = float(coeff[3]) if drift_col is not None else 0.0

    resid = z - design @ coeff
    resid_rms = float(np.sqrt(np.mean(resid ** 2)))
    max_resid = float(np.max(np.abs(resid)))
    # 环形的相邻差分(最后一点接回第一点):台阶是一个**不连续**,而正弦拟合
    # 无论如何都消不掉不连续 —— 这是分离带最好的那个信号。
    order = np.argsort(theta)
    ring = resid[order]
    max_jump = float(np.max(np.abs(np.diff(np.r_[ring, ring[:1]]))))

    slope_x = a / float(radius_m)
    slope_y = b / float(radius_m)
    mag = math.hypot(slope_x, slope_y)

    ratio = resid_rms / noise_floor_m if noise_floor_m > 0 else 0.0
    jump_ratio = max_jump / noise_floor_m if noise_floor_m > 0 else 0.0
    peak_ratio = max_resid / noise_floor_m if noise_floor_m > 0 else 0.0

    base = dict(
        residual_rms_m=resid_rms, residual_ratio=ratio,
        max_residual_ratio=peak_ratio, max_jump_ratio=jump_ratio,
        drift_rate_m_s=drift_rate, n_points=n,
    )

    # 圆上有台阶 / 污染 / 针尖事件时的否决。**不能用残差 RMS 当主判据** ——
    # 台阶的基频会被正弦拟合吸收成一个假倾斜,RMS 里只剩高次谐波(见模块常量处
    # 的实测表)。真正有分辨力的是「残差里的最大不连续」。
    if noise_floor_m > 0 and (peak_ratio > CIRCLE_MAX_RESIDUAL_SIGMA
                              or jump_ratio > CIRCLE_MAX_JUMP_SIGMA):
        return CircleTilt(valid=False, invalid_reason="residual_too_large",
                          **base)

    return CircleTilt(
        valid=True,
        tilt_x_deg=math.degrees(math.atan(slope_x)),
        tilt_y_deg=math.degrees(math.atan(slope_y)),
        slope_mag_deg=math.degrees(math.atan(mag)),
        downhill_deg=math.degrees(math.atan2(-slope_y, -slope_x)) % 360.0,
        **base,
    )


def z_span_for_frame(slope_mag_deg: float, frame_diagonal_m: float) -> float:
    """给定倾斜角与帧对角线,算这一帧会吃掉多少 Z 量程(米)。

    这是触发判据用的**统一物理量**。用它而不是「粗扫一个角度阈值、精扫另一个」
    的好处:同样 0.3° 在 1 µm 帧上吃掉 5.2 nm 的 Z,在 10 nm 帧上只吃 52 pm ——
    「粗扫敏感、精扫宽容」自动成立,少一个自由度,而且防 Z 打满这个物理动机直接
    可见。
    """
    return float(frame_diagonal_m) * math.tan(math.radians(float(slope_mag_deg)))


__all__ = [
    "STRUCTURE_RATIO_THRESHOLD",
    "DOMINANCE_MIN_TERRACE_PX",
    "DOMINANCE_TILES",
    "MIN_FRAME_PX",
    "CIRCLE_MIN_POINTS",
    "CIRCLE_RESIDUAL_MAX_RATIO",
    "CIRCLE_MAX_RESIDUAL_SIGMA",
    "CIRCLE_MAX_JUMP_SIGMA",
    "circle_tilt_resolution_deg",
    "StepVerdict",
    "TiltEstimate",
    "CircleTilt",
    "fit_circle_tilt",
    "noise_floor",
    "fit_plane_robust",
    "plane_subtract",
    "detrend_quadratic",
    "structure_dominance",
    "step_dominance_multiscale",
    "assess_steps",
    "estimate_tilt",
    "z_span_for_frame",
]
