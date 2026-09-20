"""团簇「圆不圆」的**单一真源**:从质心量边界的 r(θ),看它的相对离散。

旧判据 ``circularity = 4πA/P²`` 对像素栅格有系统性的方向与尺寸依赖。

---

## 一、旧判据错在哪 —— 不是阈值定歪了,是它量的根本不是「圆」

`P` 数的是像素化边界暴露在背景的**边**数(crack perimeter)。这个口径下:

    形状              4πA/P²
    轴对齐正方形      **0.7854 = π/4,与边长无关(精确)**
    数字化圆盘        0.465(R=3) → 0.617(R→∞)  上确界 4π²/64 = 0.6169
    45° 菱形          0.393

**正方形的分比圆盘高。** 因为阶梯边界对轴对齐的直边是**精确**的,对斜边则把
长度高估 √2 倍、对圆高估 4/π 倍。所以这个量量的是「边界有多贴合像素栅格」,
不是「有多圆」—— 同一个团簇转 45° 分数就变。

于是阈值 **0.65 落在圆盘的上确界 0.617 和正方形的 0.785 之间**:

> **任何圆盘,不论多大多完美,都过不了 `circularity ≥ 0.65`;而一个轴对齐的
> 方块永远过得了。** 这道闸门的方向是反的。

而且它连排序都不对:合成 A≈202 的椭圆,b/a = 1.00 / 0.95 / 0.90 / 0.80 读到
0.535 / 0.623 / 0.617 / 0.623 —— **完美的圆得分最低**。40 次抽样(亚像素定心 ×
旋转 × 4% 高度噪声)测它区分 b/a=0.85 与正圆的能力,d′ = **0.1**(A=200)。
**它认不出椭圆。** 它剩下的一点分辨力全在「有没有瓣」上(d′ 1.4–3.1),
而那一档有比它好一个数量级的做法。

⚠️ **Kulpa 修正(P × π/4)不是修法。** 它是常数缩放 ⇒ 排序一字不变、d′ 一字不变,
只是把圆盘从 0.55 抬到 1.0 —— 于是 b/a 从 0.95 到 0.70 全被夹成 1.000,
**顶部反而更糊**。实测在册,别再提这条路。

---

## 二、新判据:r(θ) 的相对离散,减掉「同面积完美圆盘」的那一份

    radial_dispersion       = std(r) / mean(r)     r = 边界像素到质心的距离
    radial_dispersion_floor = 同一个量,量在一个**同面积的合成完美圆盘**上
    radial_dispersion_excess= sqrt(max(0, rd² − floor²))
    equivalent_axis_ratio   = 把 excess 反解成「同样离散的椭圆的短轴/长轴」

### 为什么必须减 floor

不减就是**换个地方重建同一个系统性偏差**。完美圆盘的 rd 本身随面积变:

    A(px)      12     20     40     60    100    200    400    900   2000
    floor   0.132  0.121  0.075  0.067  0.047  0.033  0.023  0.016  0.011

**6 倍的量程差,全是像素化,与形状无关。** 一个固定阈值会系统性地判小团簇
「不圆」—— 正是我们在修的那个毛病。减掉之后(合成完美圆盘,四档噪声):

    A         0% 噪声      2%       4%       8%
    60       0.010     0.027    0.063    0.115
    200      0.006     0.028    0.056    0.105
    2000     0.001     0.026    0.048    0.098

**减完之后残差不再随面积变**,只随噪声变 —— 也就是说残差是**测量噪声**,
而一个阈值可以对所有尺寸通用。

`floor` 是**推导出来的,不是拟合的**:画一个同面积的圆盘,用同一段代码量它,
在八个亚像素中心上平均。没有可调常数,没有对用户那几帧的记忆。

### 阈值有精确的物理含义

`excess` 与椭圆一一对应(小形变下 r(θ)=R(1+ε cos2θ) ⇒ std/mean = ε/√2):

    长短轴差   5%(b/a=0.95) → excess 0.018
              10%(b/a=0.90) → 0.037
              15%(b/a=0.85) → 0.057
              20%(b/a=0.80) → 0.079
              25%(b/a=0.75) → 0.102
              30%(b/a=0.70) → 0.126
              50%(b/a=0.50) → 0.247

所以我们**不报一个没有单位的分数**,直接报 `equivalent_axis_ratio` ——
「这个团簇的不规则程度,相当于一个短轴/长轴 = q 的椭圆」。这是用户能直接
读的数,也是阈值该表达的语言。合成验证:b/a=0.80 的椭圆在 A=60…900 上
excess 读 0.107/0.103/0.100/0.097/0.094,解析值 0.079,差的部分是 4% 噪声
按平方相加(√(0.079²+0.056²)=0.097)—— **量纲、数值、尺寸无关性三项都对上。**

---

## 三、为什么是这个,不是别的(全部实测,d′ = |Δmean|/pooled sd,各 40 抽样)

`aspect`(二阶矩长短轴比,已有)和 `radial_dispersion` **互补,而且各自在
自己那一档是最强的**,所以两个都留,合取使用:

    d′            轻椭圆 b/a=0.85        三瓣(aspect 看不见)
    A=          60   200   900        60    200    900
    circ_old   0.0   0.1   0.2       2.3    1.8    1.4    ← 全面垫底
    aspect     2.4   6.9  10.7       0.2    0.1    0.2    ← 拉长最强,瓣全瞎
    rd(本件)  1.3   3.2   5.4      10.4   17.8   25.5    ← 瓣/缺口最强
    IoU(等面积圆) 0.9 2.7 4.6       7.4   14.5   30.3
    Fourier |c2|/|c0| 2.3 6.4 —      0.3    0.1    —
    solidity   0.2   0.1   0.1       2.3    3.8    5.7

被否掉的方案,以及**为什么**(这几条已钉成测试,别再重新发明):

* **亚像素边界(marching squares)** —— **实测不需要**:
  同一批抽样上 rd_sub 的 d′ = 1.2/3.0/5.8,与像素边界的 1.3/3.2/5.4 **无差别**,
  却要多一个 skimage 依赖和一个 level 参数。
* **r(θ) 按角度重采样**(而不是照单收下边界点):d′ 略高,但它要求区域对质心
  **星形**,真实带噪边界上 A≥100 时有 >10% 的抽样直接算不出来,A≥400 基本全废。
  **偶尔更准 + 经常没有 = 不能用。**
* **等面积圆 IoU**:全程略低于 rd,且在 A<200 时量化台阶是 1/A,
  合成完美圆盘 A=60/100/150 读数**全是 1.000**(没有分辨力)。
* **Fourier 一阶项 |c2|/|c0|**:就是一个更差的 `aspect`(对瓣 d′≈0)。
* **面积多极矩 Q_m = |Σz^m|/Σ|z|^m**:Q2 与 aspect 数值等价(d′ 6.8 vs 6.9),
  Q3/Q4 各自很强但只认自己那一重对称(Q3 对四瓣 d′=0.2),合起来 Qirr 全面输给 rd。
* **按角扇区的面积反解 r(θ)**(想用上全部像素而不只是边界):d′ 只有 rd 的
  1/5 —— sqrt(计数) 估计量自己的噪声吃掉了多用像素的好处。

---

## 四、这个量**判不了**的时候要说出来

* **A < 20 px**:合成完美圆盘在 1% 噪声下,excess 的 95 分位对应轴比 0.773
  (A=20)、0.642(A=12)—— 也就是**一个完美的圆有 5% 的概率被读成「差 36%」**。
  这条线以下返回 ``None`` 加理由,不返回一个数。
* **边界像素 < 6**:算不出 std。
* **残差 ≈ 帧噪声时**:那意味着「和这一帧能分辨的完美圆盘没有区别」,
  此时应报告可分辨的上界，不把与零假设无法区分的残差当作精确形变。

⚠️ 与 ``vision.barker_quality.circularity_score`` 的关系:那一个也是 std(r)/mean(r),
但它是**整帧针尖质量**(多个 feature、多个高度层、取中位数、不减 floor),
量的是另一个问题,数值不可与这里互比。两边都不是对方的替身,
所以这里没有去改它 —— 但**如果哪天要合并,合的是这一份的 floor 逻辑**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

#: 边界点少于这个数就算不出 std。
_MIN_BOUNDARY_PX = 6

#: 面积下界。**实测**:合成完美圆盘 + 1% 噪声,excess 的 95 分位 ——
#: A=12 → 0.157(相当于轴比 0.642),A=20 → 0.091(0.773),A=40 → 0.069(0.823)。
#: 20 是「完美圆盘被误读成差 25% 以上的概率 < 5%」这句话的那个面积。
#: 它**没有对用户的帧标定过**,是对合成零假设标定的。
MIN_AREA_PX = 20

#: 反解 excess → 椭圆轴比时的采样格。
_Q_GRID = np.linspace(0.02, 1.0, 981)


def _ellipse_dispersion(q: float) -> float:
    """轴比 q 的椭圆,r(θ) 的 std/mean —— 解析形状,**推导不是拟合**。"""
    t = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    r = q / np.hypot(q * np.cos(t), np.sin(t))
    return float(r.std() / r.mean())


_Q_DISPERSION = np.array([_ellipse_dispersion(q) for q in _Q_GRID])


def axis_ratio_from_dispersion(d: float) -> float:
    """把相对离散翻译成「同样离散的椭圆的短轴/长轴」。

    这是阈值该说的语言:``equivalent_axis_ratio >= 0.75`` 就是
    「不比一个长短轴差 25% 的椭圆更不规则」。
    """
    if not (d == d) or d <= _Q_DISPERSION[-1]:
        return 1.0
    if d >= _Q_DISPERSION[0]:
        return float(_Q_GRID[0])
    # _Q_DISPERSION 随 q 单调**递减**,np.interp 要求 xp 递增 ⇒ 两边取负。
    return float(np.interp(-d, -_Q_DISPERSION, _Q_GRID))


def dispersion_of_mask(mask: npt.NDArray) -> float | None:
    """std(r)/mean(r),r = 边界像素到**面积质心**的距离。None = 边界点不够。"""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    edge = mask & ~ndi.binary_erosion(mask)
    ey, ex = np.where(edge)
    if ey.size < _MIN_BOUNDARY_PX:
        return None
    ys, xs = np.where(mask)
    r = np.hypot(ey - ys.mean(), ex - xs.mean())
    m = float(r.mean())
    return float(r.std() / m) if m > 0 else None


_FLOOR_CACHE: dict[int, float] = {}

#: 量 floor 时用的八个亚像素中心。定心相位会让读数摆动(完美圆盘 A≈202 上
#: 峰谷 0.016),取平均是为了让 floor 不去追某一个特定的相位。
_SUBPIXEL_OFFSETS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
                     (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.13, 0.37))


def dispersion_floor(area_px: int) -> float:
    """同面积**完美圆盘**的 radial dispersion —— 这个量的零点。

    像素化本身就会制造离散,而且随面积变(A=20 → 0.121,A=2000 → 0.011)。
    不减掉它,一个固定阈值就会系统性地判小团簇「不圆」—— 那正是这次要修的毛病,
    只是换了个判据重犯一次。

    **推导,不是拟合**:画一个同面积的圆盘,用同一段代码量,在八个亚像素中心上
    平均。没有可调常数。
    """
    a = int(area_px)
    if a < 1:
        return 0.0
    if a in _FLOOR_CACHE:
        return _FLOOR_CACHE[a]
    radius = math.sqrt(a / math.pi)
    n = int(2 * radius + 9)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    c = (n - 1) / 2.0
    vals = []
    for ox, oy in _SUBPIXEL_OFFSETS:
        v = dispersion_of_mask(np.hypot(xx - c - ox, yy - c - oy) <= radius)
        if v is not None:
            vals.append(v)
    out = float(np.mean(vals)) if vals else 0.0
    _FLOOR_CACHE[a] = out
    return out


@dataclass(frozen=True)
class Roundness:
    """一个团簇的圆度读数。``ok=False`` 时只有 ``reason`` 有意义。"""

    ok: bool
    area_px: int
    dispersion: float | None = None
    floor: float | None = None
    excess: float | None = None
    axis_ratio: float | None = None
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "radial_dispersion": self.dispersion,
            "radial_dispersion_floor": self.floor,
            "radial_dispersion_excess": self.excess,
            "equivalent_axis_ratio": self.axis_ratio,
            "roundness_undecidable": None if self.ok else (self.reason or "判不了"),
        }


def assess_mask(mask: npt.NDArray, *, min_area_px: int = MIN_AREA_PX) -> Roundness:
    """量一个连通域圆不圆。**判不了就说判不了**,不返回一个凑出来的数。"""
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area < min_area_px:
        return Roundness(
            ok=False, area_px=area,
            reason=(f"只有 {area} 像素,低于 {min_area_px} —— 这个尺度上像素化本身"
                    f"就能让一个**完美的圆**读出 {axis_ratio_from_dispersion(dispersion_floor(max(area, 1))):.2f} "
                    "的轴比,给出的数会比没有数更糟。面积/峰高照常可用。"))
    d = dispersion_of_mask(mask)
    if d is None:
        return Roundness(ok=False, area_px=area,
                         reason=f"边界像素不足 {_MIN_BOUNDARY_PX} 个,算不出离散")
    f = dispersion_floor(area)
    ex = math.sqrt(max(0.0, d * d - f * f))
    return Roundness(ok=True, area_px=area, dispersion=d, floor=f, excess=ex,
                     axis_ratio=axis_ratio_from_dispersion(ex))


#: 高度加权二阶矩至少要多少个有效像素才肯说话。比 ``MIN_AREA_PX`` 松,
#: 因为它不依赖**边界**(边界像素少是 ``dispersion_of_mask`` 的死穴),
#: 但仍需要足够多的点把协方差矩阵撑起来。
MIN_WEIGHTED_PX = 12


def weighted_axis_ratio(height: npt.NDArray, mask: npt.NDArray,
                        base: float) -> "float | None":
    """使用每一点的高度作权重估计轴比，避免二值边界过度依赖阈值。

    对 clip(height − base, 0, None) 加权坐标求协方差，轴比为 √(λ_min/λ_max)。
    接近背景的点权重较小，因此掩膜边界的小幅变化对结果影响较弱。

    :param height: 去斜后的高度图，单位米。
    :param mask: 选取测量范围的布尔掩膜。
    :param base: 背景高度，须由背景众数估计，不能用被团簇拉高的全图均值。
    :returns: 短轴/长轴 ∈ (0, 1]；无法估计时返回 None。
    """
    h = np.asarray(height, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if h.shape != m.shape or not m.any():
        return None
    ys, xs = np.nonzero(m)
    w = np.clip(h[ys, xs] - float(base), 0.0, None)
    w = np.where(np.isfinite(w), w, 0.0)
    good = w > 0
    if int(good.sum()) < MIN_WEIGHTED_PX:
        return None
    ys, xs, w = ys[good], xs[good], w[good]
    tot = float(w.sum())
    if not (tot > 0):
        return None
    cy = float((w * ys).sum() / tot)
    cx = float((w * xs).sum() / tot)
    dy, dx = ys - cy, xs - cx
    cyy = float((w * dy * dy).sum() / tot)
    cxx = float((w * dx * dx).sum() / tot)
    cxy = float((w * dy * dx).sum() / tot)
    ev = np.linalg.eigvalsh(np.array([[cyy, cxy], [cxy, cxx]]))
    lo, hi = float(ev[0]), float(ev[1])
    if not (hi > 0) or lo < 0:
        return None
    return float(math.sqrt(lo / hi))


def background_level(height: npt.NDArray, bins: int = 96) -> "float | None":
    """背景高度 = 直方图**众数**,不是 mean。

    mean 被团簇本身拉走 —— 簇越大拉得越多,于是「背景在哪」这个问题的答案
    取决于要测的东西有多大。众数不受这个影响(只要背景仍占多数像素)。
    """
    h = np.asarray(height, dtype=np.float64)
    fl = h[np.isfinite(h)].ravel()
    if fl.size < 16:
        return None
    counts, edges = np.histogram(fl, bins=bins)
    k = int(counts.argmax())
    return float((edges[k] + edges[k + 1]) / 2.0)


__all__ = ["Roundness", "assess_mask", "axis_ratio_from_dispersion",
           "background_level", "dispersion_floor", "dispersion_of_mask",
           "weighted_axis_ratio", "MIN_AREA_PX", "MIN_WEIGHTED_PX"]
