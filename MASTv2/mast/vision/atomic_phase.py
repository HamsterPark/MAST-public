"""这一帧上有没有原子相 —— 原子分辨针尖的验收判据。

常用做法是做两下就停下来,看扫出来的图里有没有原子相 ——
判据可能是 FFT。确实是 FFT,但**单靠"FFT 里有个峰"会被针尖抖动骗过去**,所以这
里是三条判据取与,外加一道尺度门。

## 合成实测:两个看起来合理的判据被推翻了

写这个模块时先按常规做法选了判据,然后按本仓纪律拿物理合成数据实测(合成参数:
Au(111) 原子行间距 0.2494 nm、起伏 10 pm、噪声 1-10 pm、5 nm/256 px)。两个判据
当场出局:

* **径向自相关环平均**(``seg_scale_adaptive.radial_ac_ratio``)对六角晶格
  **不适用**。它适用于近各向同性的阵列，但六角晶格的自相关在各晶格方向
  上是强正峰、方向之间是负的，环上一平均就互相
  抵消 —— 实测真晶格只有 +0.07..+0.18,远在 0.35 阈值之下。照搬会把每一帧真的
  原子分辨判成「无序」。
* **峰强度类判据**(带内 SNR、FFT 锐度)分不开**准周期抖动**。用带通白噪声模拟
  针尖抖动产生的条纹:30 个种子 30 个都在原子带里产生了合格的谱峰(SNR 15-38),
  自相关角向最大值 0.13-0.28,与 6 pm 噪声下真晶格的 0.23 完全重叠。

## 真正分得开的那一条:角向集中度

晶格与抖动的本质区别不在「有没有峰」,而在峰的**形状**:晶格是倒空间里的**离散
布拉格点**(六角面 6 个,60° 等距),抖动是一个**弥散环**。把功率谱在 |k|≈1/T 的
环上按角度分 bin,取 ``最大 bin / 中位 bin``:

    真晶格   97 .. 7645   (最差的 97 是起伏与噪声 1:1 时)
    带通抖动  1.8 .. 3.3

30 倍余量,阈值取 20。这条判据同时对慢轴漂移免疫(剪切把点拉成短弧,集中度仍有
1800+)。

自相关的角向最大值仍然算出来放在 ``order_ratio`` 里 —— 它是有用的诊断(平移一个
周期后图像与自己有多像),但**不作硬判据**,因为上面那组实测说明它分不开抖动。

## 尺度门:判不了要说判不了,不能说"没有"

``_fft_sharpness`` 的 ``has_lattice``(>8)只描述谱峰，不能独自证明原子分辨。
这里另以 **nm/px** 限制适用尺度:低于 0.02 满权重,
0.02-0.05 过渡带,高于 0.05 时晶格物理上不可分辨 —— 这一档直接拒判
(``reasons=("scale_gate",)``)而不是报「没有原子相」。

区别很实在:「没有原子相」会让配方接着扰动针尖;「这一帧判不了」应该让它换个更小
的视野再看。5 nm / 256 px = 0.0195 nm/px 刚好落在满权重档,这也是配方选这组帧
参数的原因。

## 晶格常数只报快扫方向

帧法的老问题(见 ``mast.vision.tilt`` 的模块注释):**只有快扫方向是准的**。一行
在几毫秒内扫完,漂移可忽略;慢轴方向相邻两行隔着一整个行时间,漂移累积成剪切,把
晶格拉变形。所以:

* 判「有没有原子相」用二维谱 —— 剪切把布拉格点拉成短弧,判据仍成立;
* 报**晶格常数**只用逐行 1D 谱(``period_fast_axis_nm``),它对慢轴漂移完全免疫
  (实测:剪切 0.30 px/行下快轴周期变化 < 1%,而径向周期偏了 10%);
* 与 ``expected_a_nm`` 比对时**下界严、上界松**:快轴测到的是晶格周期在快扫方向
  上的投影 ``a/|cos θ|`` —— 投影只会让周期变大,不会变小。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: 原子周期带(纳米)。与 ``seg_scale_adaptive.DEFAULTS["atomic_band_nm"]`` 同源。
ATOMIC_BAND_NM: tuple[float, float] = (0.18, 0.80)

#: 尺度门(纳米/像素)。与 ``tip_metrics`` 的 Bragg 尺度门同源 —— 那里是
#: 2026-07-27 用物理真值标出来的(rho +0.37 → +0.05)。
SCALE_FULL_NMPP = 0.02
SCALE_OFF_NMPP = 0.05

# 绝对起伏下限不能通用于不同衬度、噪声与成像条件。
# 应结合相对噪声量和结构证据，避免把低幅值晶格误判为空白。
#
# 弥散对角谱脊也会集中在少数方向，因而可能越过角向集中度门。
# 这里把高起伏与一阶峰半径不一致合取，用于限制这一失效模式。
# 只看半径散布会受小帧 FFT 分箱误差影响；只看起伏会误伤含台阶的晶格。
# 帧内中途针尖变化由两半统计等单独处理，不由此处替代。
# 执行阈值须针对目标成像条件验证。
_STREAK_RMS_MIN_PM = 40.0

# 一阶峰半径的相对散布上限。
# 同阶晶格峰应具有相容的半径；谱脊上的极大则可能沿线分布在不同半径。
# 此统计补充只关注方向集中度的判据，并须结合起伏和尺度条件使用。
_PEAK_RADIUS_CV_MAX = 0.20

#: 判得了的最小**周期数**。低于它,晶格的谱线落在 ``seg_scale_adaptive`` 那道
#: ``min(H, W) / 4`` 的搜索上限之外 —— 这时报「没有晶格」等于**没往那里看**。
#:
#: 2026-08-23 二分实测(合成完美晶格):能判的下界是 **4.70–4.75 个周期**,在
#: 128/256/384 三种像素数、0.235/0.288/0.400 三种晶格常数、噪声 0→2x 信号幅度下
#: **全都一样** —— 这是纯几何限制,不是统计阈值,所以不需要留"空隙"。取 5.0:
#: 比实测下界高一点点,4.75..5.0 这一窄条本来判得了、现在归进判不了,是刻意的保守。
_MIN_PERIODS_IN_FRAME = 5.0

#: 角向集中度阈值。合成实测:真晶格 97..7645,带通抖动 1.8..3.3(见模块注释)。
DEFAULT_CONCENTRATION_MIN = 20.0


#: :func:`assess_atomic_phase` 可能放进 ``reasons`` 的**全部**出局词(闭集)。
#:
#: 存在的理由是下游要对它做**穷举**映射:调用方把每个词翻译成「没有原子相」还是
#: 「判不了」时,漏掉一个词就会静默落进调用方的兜底档 —— 而那个兜底档多半是
#: 「没有」(最坏的那一档:接着去扰动针尖)。有了这个常量,新增出局词而不更新映射
#: 表的调用方会在测试里当场变红,而不是在真机上安静地判错。
#:
#: ⚠️ 加出局词时**必须同时加进这里**。``tests/v2/unit/vision/test_atomic_scale_plan.py``
#: 直接扫本函数源码里的字面量与本元组对账,漏一个就红。
ALL_REASONS: tuple[str, ...] = (
    # 判据性出局词 —— 这一帧上「没有」原子分辨
    "no_lattice_peak",
    "not_a_lattice",
    "fft_not_sharp",
    "fast_axis_no_peak",
    "period_below_lattice",
    "period_far_above_lattice",
    "peaks_not_one_lattice",
    "peaks_are_ridges",
    "radial_fast_axis_disagree",
    # 判不了 —— 「没有根据下结论」,与「没有」是两件事
    "too_few_periods",
    "scale_gate",
    "scale_reduced",
    "unknown_pixel_size",
    "insufficient_data",
    "dead_flat",
    "dependency_unavailable",
)


@dataclass(frozen=True)
class AtomicPhaseResult:
    """一帧扫描图上原子相的判定。

    ``passed`` 只在判据全过、且尺度门允许判定时为真。

    ``period_nm`` 是二维谱给的径向周期(会被慢轴漂移拉偏),
    ``period_fast_axis_nm`` 是逐行 1D 谱测的快扫方向周期 ——
    **要引用晶格常数时用后者**。
    """

    passed: bool
    scale: str | None = None                 # full / reduced / off
    nm_per_px: float | None = None
    period_nm: float | None = None
    period_fast_axis_nm: float | None = None
    snr: float = 0.0
    angular_concentration: float = 0.0       # 离散布拉格点 vs 弥散环(核心判据)
    order_ratio: float = 0.0                 # 自相关角向最大值(诊断,不作判据)
    fft_sharpness: float = 0.0
    expected_a_nm: float | None = None
    slow_axis_trusted: bool = False          # 帧法恒为 False（见模块注释）

    # 按扫描顺序报告前后半帧的角向集中度。None 表示没有进行两半检查。
    # 整帧混合值可能掩盖帧内状态变化，不能单独代表采集结束时的针尖状态。
    half_concentrations: tuple[float, float] | None = None
    #: 前后两半**各自的 ``passed``**。
    #:
    #: 2026-08-24 补。这两个判决**本来就算出来了** —— 上面那两行
    #: ``first = assess_atomic_phase(h[:mid])`` / ``second = …`` 拿到的是完整
    #: 结果，而此前只把 ``angular_concentration`` 取出来当 ``half_concentrations``,
    #: **判决被折叠成了一个数字**，于是下游想问「这半张到底算不算晶格」时
    #: 只能拿数字去猜一个阈值 —— 而半帧的 conc 天然比整帧低 0.38×,
    #: 猜出来的阈值必然是错的。
    #:
    #: 它能回答一个 ``half_concentrations`` 回答不了的问题:
    #: **「整帧好」与「半帧好」**。实测(正反扫结论一致)::
    #:
    #:     0445 摧毁前   前半 156.7 ✓ / 后半  91.0 ✓   两半都过
    #:     0447 摧毁前   前半  27.9 ✓ / 后半  74.9 ✓   两半都过
    #:     0475 摧毁后   前半  12.6 ✗ / 后半  98.2 ✓   半张
    #:     0480 摧毁后   前半  60.9 ✓ / 后半   0.0 ✗   半张
    #:
    #: 整帧 conc 分别是 293.4 / 148.7 / 89.1 / 82.6 —— **整帧那个数分不开后两行**,
    #: 而这正掩盖了「有几行原子分辨,但质量显然不够」这种情形。
    #:
    #: ⚠️ ``None`` 表示**没算**(帧太窄/被调用方关掉),不是「没过」。
    half_passed: tuple[bool, bool] | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def scale_gate(nm_per_px: float | None) -> str | None:
    """这一帧的像素尺度够不够分辨原子。``None`` = 不知道像素尺度。"""
    if nm_per_px is None or not np.isfinite(nm_per_px) or nm_per_px <= 0:
        return None
    if nm_per_px < SCALE_FULL_NMPP:
        return "full"
    if nm_per_px <= SCALE_OFF_NMPP:
        return "reduced"
    return "off"


def _size_nm(size_m: float) -> float:
    """米 → 纳米,并抹掉换算本身产生的浮点噪声。返回 0.0 = 这不是一个视野。

    没有这一步会出一个非常难查的错:用户输 12.8 nm,界面转成米再转回来得到
    ``12.800000000000002`` nm,除以 256 px 得 0.05000000000000001 —— 比过渡带的
    闭上界**大一个最低位**,于是「证据不足」被判成「物理上不可分辨」,而用户
    看到的两个数字一模一样。

    抹到 9 位小数(纳米)= 1e-18 m。这个量级在物理上毫无意义,但比双精度往返误差
    大好几个数量级,足够吸收 nm↔m 的来回换算。``core.scan_policy._BOUND_REL_TOL``
    为档位边界解的是同一个问题,理由逐字相同。

    ⚠️ 这是**入口去噪**,不是第二道闸:阈值比较仍然只有 :func:`scale_gate` 一处。
    """
    try:
        nm = float(size_m) * 1e9
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(nm) and nm > 0):
        return 0.0
    return round(nm, 9)


def min_pixels_for_scale(size_m: float, *,
                         target: float = SCALE_FULL_NMPP) -> int:
    """这个视野要进 ``target`` 档,至少得取多少像素。参数不成立时返回 0。

    尺度门的满权重档是**严格小于**(见 :func:`scale_gate`),所以要的是「最小的、
    使 ``size/N < target`` 成立的整数 N」= ``floor(x) + 1``,**不是** ``ceil(x)``。
    两者只在 x 恰好是整数时不同,而那恰恰是最常撞上的一档:5.12 nm 要 **257** px,
    256 px 算出来正好 0.02 —— 等于门槛,过不去。

    ``floor`` 之后还要拿门再验一次:``size_nm / target`` 这个除法本身有浮点噪声
    (真值 250 可能算成 249.999…),照着它取整会得到一个**过不了自己要过的那道门**
    的像素数。不信除法,信门。
    """
    try:
        tgt = float(target)
    except (TypeError, ValueError):
        return 0
    size_nm = _size_nm(size_m)
    if size_nm <= 0.0:
        return 0
    if not (math.isfinite(tgt) and tgt > 0):
        return 0
    n = int(size_nm / tgt) + 1
    for _ in range(3):                      # 浮点噪声最多差 1 个 ulp
        if size_nm / n < tgt:
            break
        n += 1
    return int(n)


def plan_scale(size_m: float,
               pixels: int) -> tuple[float | None, str | None, str]:
    """**下发扫描之前**回答「这组帧参数判不判得出原子相」。

    返回 ``(nm_per_px, scale, problem)``:

    * ``scale`` 是 :func:`scale_gate` 的三态;``None`` = 这组参数根本算不出 nm/px
      (视野或像素数非正/非有限),判据到时候只会回 ``unknown_pixel_size``;
    * ``problem`` 是给人看的一句话,``""`` = 没问题(满权重档)。

    存在的理由(``special_tip_workflow`` 里写过一遍,这里是它的归属地):扫完一张
    判不了的图再说,白花一帧的时间,而且流程会把「判不了」误读成「还没弄出原子
    相」接着去扰动针尖。

    住在本模块而不是 ``core.scan_policy``,是因为**阈值住在这里**
    (:data:`SCALE_FULL_NMPP` / :data:`SCALE_OFF_NMPP`)。闸放在别处就会出现第二份
    0.02/0.05,而两份阈值迟早会漂。

    可调的旋钮只有两个:``nm/px = size/pixels`` 与 ``line_time`` 无关 —— 要么缩视野
    要么加像素。⚠️ 加像素时必须同比加 ``line_time``:否则每像素驻留砍半,
    ``nm/px`` 好看了而每个采样点携带的信息反而更少,闸门是被骗过去的
    (见 ``core.scan_policy`` 的 ``atomic_verify`` 档注释)。
    """
    try:
        px = int(pixels)
    except (TypeError, ValueError):
        return None, None, "视野或像素数不是数,算不出 nm/px。"
    size_nm = _size_nm(size_m)
    if size_nm <= 0.0 or px <= 0:
        return None, None, (
            f"视野 {size_m!r} / 像素 {pixels!r} 不是一组有效的帧参数 —— "
            f"算不出 nm/px,判据只会说「不知道像素尺度」。")

    nmpp = size_nm / px
    scale = scale_gate(nmpp)
    if scale == "off":
        need = min_pixels_for_scale(size_m)
        return nmpp, scale, (
            f"{size_nm:g} nm / {px} px = {nmpp:.4f} nm/px,超过 "
            f"{SCALE_OFF_NMPP} nm/px —— 这个尺度上晶格物理上不可分辨,判据只会说"
            f"「判不了」。请缩小视野或加大像素数(这个视野要 {need} px 以上)。")
    if scale == "reduced":
        need = min_pixels_for_scale(size_m)
        return nmpp, scale, (
            f"{size_nm:g} nm / {px} px = {nmpp:.4f} nm/px 落在过渡带 "
            f"[{SCALE_FULL_NMPP}, {SCALE_OFF_NMPP}] —— 判据会给出结论但证据强度"
            f"不足以当验收依据。建议 {size_nm:g} nm 用 {need} px 以上。")
    return nmpp, scale, ""


def _ring_mask(shape: tuple[int, int], radius_px: float, rel: float = 0.20):
    """``(选中掩码, 相对中心的角度)``,环宽 ±``rel``。"""
    cy, cx = shape[0] // 2, shape[1] // 2
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dy, dx = yy - cy, xx - cx
    rr = np.hypot(dy, dx)
    sel = (rr > (1.0 - rel) * radius_px) & (rr < (1.0 + rel) * radius_px)
    return sel, np.arctan2(dy, dx)


def _angular_bins(values, angles, n_bins: int):
    idx = ((angles + math.pi) / (2.0 * math.pi) * n_bins).astype(int) % n_bins
    out = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = float(values[m].mean())
    return out


def angular_concentration(image: npt.ArrayLike, period_px: float,
                          n_bins: int = 72) -> float:
    """功率谱在 |k|≈1/T 环上的角向集中度 = ``最大 bin / 中位 bin``。

    这是分开「真晶格」与「针尖抖动造出的准周期条纹」的那一条判据:晶格在倒空间
    是离散的布拉格点(角向极不均匀),抖动是弥散环(角向均匀)。实测分离度见模块
    注释。
    """
    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 16 or not (period_px and period_px >= 3.0):
        return 0.0
    x = h - h.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
    # 频率半径:周期 T_px 对应的谱半径是 N/T。
    f0 = float(P.shape[0]) / float(period_px)
    sel, ang = _ring_mask(P.shape, f0)
    if int(sel.sum()) < n_bins:
        return 0.0
    bins = _angular_bins(P[sel], ang[sel], n_bins)
    med = float(np.median(bins))
    if med <= 0.0:
        return 0.0
    return float(bins.max() / med)


def order_ratio(image: npt.ArrayLike, period_px: float,
                n_bins: int = 36) -> float:
    """自相关在半径 T 的环上、按角度分 bin 后的最大值 ÷ ``r(0)``。

    读作「沿最好的那个方向平移一个周期后,图像与自己有多像」。诊断用 —— 实测它
    分不开准周期抖动(见模块注释),所以不作硬判据。

    刻意**不用** ``seg_scale_adaptive.radial_ac_ratio``:那个取环**平均**,对六角
    晶格会把正负抵消掉。
    """
    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 16 or not (period_px and period_px >= 3.0):
        return 0.0
    x = h - h.mean()
    F = np.fft.fft2(x)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2))) / x.size
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    centre = float(ac[cy, cx])
    if not np.isfinite(centre) or centre <= 0.0:
        return 0.0
    sel, ang = _ring_mask(ac.shape, float(period_px), rel=0.15)
    if not sel.any():
        return 0.0
    bins = _angular_bins(ac[sel] / centre, ang[sel], n_bins)
    return float(bins.max())


def fast_axis_period_nm(
    image: npt.ArrayLike,
    nm_per_px: float,
    *,
    band_nm: tuple[float, float] = ATOMIC_BAND_NM,
    rel_prom: float = 0.35,
) -> tuple[float | None, float]:
    """逐行 1D 功率谱里的原子周期,以及它的信噪比。返回 ``(周期nm | None, snr)``。

    每一行独立做 FFT 再把功率谱平均 —— 慢轴漂移(行与行之间的错位)完全进不来。
    代价是只能测到晶格周期在快扫方向上的**投影**,见模块注释。

    强峰有多个时取**周期最小**的那个,不取功率最大的。理由是投影的方向性:六角
    晶格的三组波矢在快扫方向上给出 ``a``、``a/cos60° = 2a``、``2a`` 三个周期,后
    两个还叠在一起,功率是第一个的两倍 —— 按功率取会稳定地报出两倍晶格常数。而
    投影只会**放大**周期,所以最小的那个才最接近真实晶格常数。
    (``seg_scale_adaptive._band_peak`` 出于同一条理由也取最小周期,那里防的是锁到
    moiré 超结构上。)
    """
    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] < 8 or not (nm_per_px and nm_per_px > 0):
        return None, 0.0
    # 逐行去均值去斜(一行里的背景就是一条直线,不必上二阶)。
    x = np.arange(h.shape[1], dtype=np.float64)
    xc = x - x.mean()
    denom = float(np.sum(xc * xc)) or 1.0
    rows = h - h.mean(axis=1, keepdims=True)
    slope = (rows @ xc) / denom
    rows = rows - slope[:, None] * xc[None, :]
    win = np.hanning(h.shape[1])
    P = np.abs(np.fft.rfft(rows * win[None, :], axis=1)) ** 2
    spec = P.mean(axis=0)
    freq = np.fft.rfftfreq(h.shape[1])           # 周期/像素
    with np.errstate(divide="ignore"):
        period_px = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12), np.inf)
    period_nm = period_px * float(nm_per_px)
    sel = ((period_nm >= band_nm[0]) & (period_nm <= band_nm[1])
           & (period_px >= 3.0))               # < 3 px 的周期是采样噪声
    if not sel.any():
        return None, 0.0
    base = float(np.median(spec[freq > 0])) + 1e-30
    band = spec[sel]
    band_periods = period_nm[sel]
    peak_snr = float(band.max() / base)
    # 峰要在带内显著,否则报「带里最大的那个 bin」等于报噪声。
    if peak_snr < 1.0 + rel_prom:
        return None, peak_snr
    # 所有「强峰」= 功率达到带内最强峰的 rel_prom 倍;其中取周期最小的那个。
    strong = band >= float(rel_prom) * float(band.max())
    if not strong.any():                       # pragma: no cover — max 自己必在内
        return None, peak_snr
    j = int(np.argmin(band_periods[strong]))
    return float(band_periods[strong][j]), float(band[strong][j] / base)


def assess_atomic_phase(
    image: npt.ArrayLike,
    *,
    nm_per_px: float | None,
    expected_a_nm: float | None = None,
    snr_min: float = 4.0,
    concentration_min: float = DEFAULT_CONCENTRATION_MIN,
    sharpness_min: float = 8.0,
    band_nm: tuple[float, float] = ATOMIC_BAND_NM,
    tolerance_frac: float = 0.15,
    max_projection_ratio: float = 2.0,
    allow_reduced_scale: bool = False,
    _check_halves: bool = True,
) -> AtomicPhaseResult:
    """这一帧上有没有原子分辨。判据取与,外加一道尺度门。

    纯函数:输入一个二维高度数组与显式阈值,输出 frozen 结果。不读配置、不碰硬件、
    不抛异常。

    判据:

    1. 原子带里有显著谱峰(``snr_min``)—— 排除没有周期结构的帧;
    2. **角向集中度** ≥ ``concentration_min`` —— 离散布拉格点而不是弥散环。这是
       分开真晶格与针尖抖动的那一条(见模块注释的实测);
    3. FFT 峰锐度 ≥ ``sharpness_min`` —— 与仓里既有的针尖判据保持一致;
    4. 逐行 1D 谱也看得到这个周期 —— 否则那是慢轴方向的行噪声,不是晶格。

    ``allow_reduced_scale`` 为假(默认)时,只有 ``nm/px < 0.02`` 的帧会给出正面
    结论;过渡带(0.02..0.05)通过判据也仍以 ``scale_reduced`` 落选 —— 在那个尺度
    上「有原子相」这句话的证据强度撑不住一次针尖验收。
    """
    reasons: list[str] = []
    warns: list[str] = []
    scale = scale_gate(nm_per_px)
    nmpp = float(nm_per_px) if scale is not None else None

    def _fail(*why: str, **kw) -> AtomicPhaseResult:
        return AtomicPhaseResult(
            passed=False, scale=scale, nm_per_px=nmpp,
            expected_a_nm=(float(expected_a_nm) if expected_a_nm else None),
            reasons=tuple(why), warnings=tuple(warns), **kw)

    if scale is None:
        # 像素尺度未知 —— 周期换算不出纳米,任何「原子相」结论都没有根据。
        return _fail("unknown_pixel_size")
    if scale == "off":
        return _fail("scale_gate")

    h = np.asarray(image, dtype=np.float64)
    if h.ndim == 3 and h.shape[0] in (1, 2):
        h = h[0]
    if h.ndim != 2 or min(h.shape) < 16:
        return _fail("insufficient_data")

    # 排除未采集行必须早于填补 NaN。
    # 原始缓冲中的未写入全零区域会在 FFT 中引入人为结构并改变集中度。
    # 使用 frame_validity.acquired_row_mask 的共用口径，仅在已采集数据上判读。
    from mast.vision.frame_validity import acquired_row_mask

    acquired = acquired_row_mask(h)
    n_dropped = int(h.shape[0] - int(acquired.sum()))
    if n_dropped:
        h = h[acquired]
        if h.ndim != 2 or min(h.shape) < 16:
            # 扫出来的太少 —— **说「判不了」，不说「没有」**。
            return _fail("insufficient_data")

    # 视野必须容纳足够周期，只有像素数充足并不保证目标周期落在搜索窗内。
    # 周期数不足属于未知，不能当成没有晶格而触发继续修针。
    # expected_a_nm 缺席时使用原子带下限，仅拒绝连最小候选周期都装不下的视野。
    _period_nm = float(expected_a_nm) if expected_a_nm else float(band_nm[0])
    if _period_nm > 0 and min(h.shape) * nmpp < _MIN_PERIODS_IN_FRAME * _period_nm:
        return _fail("too_few_periods")

    if not np.isfinite(h).all():
        med = np.nanmedian(h) if np.isfinite(h).any() else 0.0
        h = np.nan_to_num(h, nan=float(med) if np.isfinite(med) else 0.0)

    try:
        from mast.vision.seg_scale_adaptive import (
            DEFAULTS,
            detect_texture,
            flatten_robust,
        )
        from mast.vision.tip_metrics import _detrend, _fft_sharpness
    except Exception as exc:  # noqa: BLE001 — 依赖缺席就是「判不了」
        logger.debug("原子相判据依赖缺席: %s", exc)
        return _fail("dependency_unavailable")

    flat = flatten_robust(h)
    std = float(flat.std())
    if not np.isfinite(std) or std <= 0.0:
        return _fail("dead_flat")

    # ── 判据 1:原子带里有没有显著谱峰 ──
    params = dict(DEFAULTS)
    params["atomic_band_nm"] = tuple(band_nm)
    params["lat_snr"] = float(snr_min)
    tex = detect_texture(flat, float(nmpp), params)
    atomic = tex.get("atomic")
    period_nm: float | None = None
    snr = 0.0
    t_px = 0.0
    if atomic:
        t_px, snr = float(atomic[0]), float(atomic[1])
        period_nm = t_px * float(nmpp)
    else:
        reasons.append("no_lattice_peak")

    # ── 判据 2:离散布拉格点,不是弥散环(核心) ──
    conc = angular_concentration(flat, t_px) if t_px >= 3.0 else 0.0
    if conc < float(concentration_min):
        reasons.append("not_a_lattice")

    # ── 判据 3:布拉格峰锐度 ──
    sharp, _res_nm, _has_lat = _fft_sharpness(_detrend(h) / std, float(nmpp))
    sharp = float(sharp)
    if sharp < float(sharpness_min):
        reasons.append("fft_not_sharp")

    # ── 判据 3b:这些峰是**同一个晶格**的吗 ──
    #
    # 补角向集中度的盲区(见 _PEAK_RADIUS_CV_MAX 的自述):一条穿过原点的弥散条纹
    # 在角度上也是集中的,照样能拿到 60 分。但条纹上的「峰」半径各不相同,
    # 而真晶格的一阶峰同半径 —— 这一条问的是另一件事,所以补得上。
    #
    # 取不到峰(帧太小/尺度不对)就**不判**:那是「读不到」,上面已经有专门的出局词。
    try:
        from mast.vision.lattice_calibration import find_lattice_peaks

        _lat = find_lattice_peaks(h, float(nmpp))

        # 候选全部被识别为谱脊，是否定晶格假设的证据，不是数据读取失败。
        # 不能在剔除全部脊峰后跳过判据，让只有方向集中而无局部峰的纹理得到通过。
        if (not _lat.ok) and _lat.reason == "too_few_peaks" and _lat.n_ridge >= 2:
            reasons.append("peaks_are_ridges")

        # 半径散布必须使用所有局部极大，包括剔除的 ridge_peaks。
        # 剔脊用于避免污染定标；结构判定仍需保留这些证据，否则一个过滤步骤可能
        # 删除另一道判据的全部输入。
        _all = tuple(_lat.peaks) + tuple(_lat.ridge_peaks)
        if len(_all) >= 3:
            _r = np.array([float(np.hypot(pk.kx, pk.ky)) for pk in _all])
            _m = float(_r.mean())
            _cv = (float(_r.std()) / _m) if _m > 0 else 0.0
            # **两条并且** —— 单用任何一条都会误伤，见 _STREAK_RMS_MIN_PM 的自述。
            _rms_pm = float(np.nanstd(flat)) * 1e12
            if _cv > _PEAK_RADIUS_CV_MAX and _rms_pm >= _STREAK_RMS_MIN_PM:
                reasons.append("peaks_not_one_lattice")
    except Exception as exc:  # noqa: BLE001 — 本函数从不抛
        logger.debug("一阶峰半径散布算不了(按不否决处理): %s", exc)

    # ── 判据 4:逐行谱也要看到它(可信的那个方向) ──
    fast_nm, fast_snr = fast_axis_period_nm(flat, float(nmpp), band_nm=band_nm)
    if fast_nm is None:
        # 二维谱说有、逐行说没有 —— 那是慢轴方向的周期性(行噪声/干扰),不是晶格。
        reasons.append("fast_axis_no_peak")
    elif period_nm is not None and abs(fast_nm - period_nm) > 0.5 * max(
            fast_nm, period_nm):
        # **判否,不是告警。** 这一条 2026-08-24 从 ``warns`` 挪过来。
        #
        # 判据 4 的整个用意是「二维谱会被慢轴假象骗,逐行谱才是可信的那个方向」。
        # 上面那支(逐行谱**什么也没看到**)判否;这一支是逐行谱看到了、但看到的是
        # **另一个周期**。两支说的是同一件事:**可信方向没有确认二维谱那个峰**。
        # 一支判否一支只记一笔,是漏接,不是设计。
        #
        # 抓到它的形状(合成,2.4 nm / 128 px):**波浪形横带** —— 周期在慢轴上,
        # 相位沿快轴游走。二维谱说 0.241 nm、逐行谱说 0.486 nm,而它拿到角向
        # 集中度 **2931** 一路通关。有意思的是**完美**横带反而被正确判否
        # (conc 11.7):加上相位游走之后它才越过闸门 —— 越乱越像晶格。
        #
        # 周期不一致属于否决条件；阈值有效性需独立验证，不能依赖精选帧调参。
        reasons.append("radial_fast_axis_disagree")

    # 诊断量:平移一个周期后图像与自己有多像。不作判据(实测分不开抖动)。
    order = order_ratio(flat, t_px) if t_px >= 3.0 else 0.0

    # ── 与已知晶格常数比对(下界严、上界松) ──
    if expected_a_nm and fast_nm is not None:
        a = float(expected_a_nm)
        if fast_nm < a * (1.0 - float(tolerance_frac)):
            # 投影只会让周期变大 —— 明显更小的周期不是这个晶格。
            reasons.append("period_below_lattice")
        elif fast_nm > a * float(max_projection_ratio):
            reasons.append("period_far_above_lattice")
        elif fast_nm > a * (1.0 + float(tolerance_frac)):
            # 合理的取向投影,不是问题,但要说出来。
            warns.append("period_consistent_with_oblique_lattice")

    if scale == "reduced":
        warns.append("scale_reduced")
        if not allow_reduced_scale:
            reasons.append("scale_reduced")

    # 按扫描顺序检查前后半帧，避免整帧混合读数掩盖采集中途的针尖状态变化。
    # 只在前半独立通过、后半独立判不成晶格时确认该类变化；
    # 单看读数比值会把衬度渐变误当成突变，不能取代各半帧的独立判定。
    halves = None
    half_ok = None

    # 两半检查按物理尺寸和可容纳的晶格周期数守卫，不按行数守卫。
    # 任一半视野不足时，集中度零值只表示无法测量，不应被比值计算解释成帧内突变。
    _half_nm = (h.shape[0] / 2.0) * (nmpp or 0.0)
    if _check_halves and not reasons and _half_nm >= 2.0:
        mid = h.shape[0] // 2
        kw = dict(nm_per_px=nmpp, expected_a_nm=expected_a_nm, snr_min=snr_min,
                  concentration_min=concentration_min, sharpness_min=sharpness_min,
                  band_nm=band_nm, tolerance_frac=tolerance_frac,
                  max_projection_ratio=max_projection_ratio,
                  allow_reduced_scale=allow_reduced_scale, _check_halves=False)
        first = assess_atomic_phase(h[:mid], **kw)      # 先扫的那一半
        second = assess_atomic_phase(h[mid:], **kw)     # 后扫的那一半
        halves = (float(first.angular_concentration),
                  float(second.angular_concentration))
        half_ok = (bool(first.passed), bool(second.passed))

        # 此处只报告两半统计，不独立判定认证结果。
        # 调用方阈值可变，且半帧的谱峰宽度与整帧不同，不能套用整帧绝对集中度下限。
        # 发证方应结合相同尺寸两半的相对变化与其他证据作判断。
    return AtomicPhaseResult(
        passed=not reasons,
        scale=scale,
        nm_per_px=nmpp,
        period_nm=period_nm,
        period_fast_axis_nm=fast_nm,
        snr=float(max(snr, fast_snr if fast_nm is not None else 0.0)),
        angular_concentration=conc,
        order_ratio=order,
        fft_sharpness=sharp,
        expected_a_nm=(float(expected_a_nm) if expected_a_nm else None),
        slow_axis_trusted=False,
        half_concentrations=halves, half_passed=half_ok,
        reasons=tuple(reasons),
        warnings=tuple(warns),
    )


__all__ = [
    "ALL_REASONS",
    "ATOMIC_BAND_NM",
    "DEFAULT_CONCENTRATION_MIN",
    "SCALE_FULL_NMPP",
    "SCALE_OFF_NMPP",
    "AtomicPhaseResult",
    "angular_concentration",
    "assess_atomic_phase",
    "fast_axis_period_nm",
    "min_pixels_for_scale",
    "order_ratio",
    "plan_scale",
    "scale_gate",
]
