"""这一帧上的原子相是**哪一种** —— 多畴/孪晶/多相晶体的畴指纹。

``atomic_phase`` 回答「这一帧上**有没有**原子相」;本模块回答「这是**哪一种**」。
两个不同的问题,所以是两个模块 —— 塞进同一个函数会让 ``passed`` 的语义分叉
(「过」到底是「有晶格」还是「认出畴了」)。形状照 ``herringbone.py``:复用
``atomic_phase`` 的环掩码与角向分 bin、自己定尺度门、自己出三态 verdict、把被否
掉的方案钉成测试。见「S3 畴搜索设计」D1/D2。

**通用能力,零样品知识**:对称性(六角 60°/矩形 90°)、原型峰表、容差、权重全部
来自运行时标定的 :class:`~mast.vision.domain_reference.DomainReference`。
**参照系建立之前,本模块只输出指纹 + ``undetermined(no_reference)``** ——
代码不能自己发明「这是 A 相」(与 ``scan_prep_thresholds`` 的「没有数据就造一个
profile 等于伪造标定」同源)。

## 指纹是「方向-周期」峰表,不是标量

一个标量(比如「主方向角」)既表达不了周期集合,在六角对称下也不稳定:同一个畴
的连续两帧,argmax 落在 6 个布拉格点的哪一个由**针尖衬度各向异性**决定(三个晶格
方向被画出来的深浅不一样,而且随针尖状态帧帧在变)⇒ 方向可以差 60°。
合成实测(20 个种子,各向异性 ±15%):**单 argmax 方向在同一个畴内跳了 60.1°,
而峰集合指纹的同畴距离只有 0.0013** —— 差了四十多倍。所以指纹是有序峰表,比对在
``symmetry_deg`` 的商空间里做。测试 ``test_single_argmax_direction_flips_by_60deg_on_hex``
把这条钉住。

(纯正弦的功率谱与相位无关,所以只加噪声/换相位**造不出**这个翻转 —— 拿那种语料
去「验证」这条会得到一个全绿而毫无内容的测试。)

## 距离的两项**都是相对量**,这是刻意的

    d(F₁,F₂) = w_θ·d_θ + w_T·d_T   (两项各自归一到 [0,1],所以 w 默认 1/1 = 等权)
    d_θ:  方向之差,mod symmetry_deg 的圆周距离 ÷ (symmetry_deg/2)
    d_T:  配对峰的周期**相对**差 |T₁-T₂| / max(T₁,T₂)

差与比对压电标定的**各向同性**倍率误差天然免疫:一个统一的 ``s`` 倍尺度误差把所有
周期同乘 ``s``(比值不变)、不改任何角度。未经过独立外部标尺校准时，
``period_nm`` 的绝对值不宜直接与文献晶格常数比较；相对距离则不受两帧
共同的各向同性倍率误差影响。
测试 ``test_isotropic_calibration_bias_does_not_change_the_distance`` 把它变成可证伪
的:两帧的 ``nm_per_px`` 同乘 0.875 后距离差 **5.6e-17**(浮点噪声),而 ``d_T``
改回绝对差 nm 那条变异必红。

⚠️ 免疫的是**固定**的倍率误差(仪器属性,两边同乘)。标定在两帧之间**变了**
(只有一边偏)时距离跳 **0.062** —— 比同畴分母还大。参照系里存的是纳米,所以
**重新标定压电之后参照系要重建**。「免疫」不等于「怎么标定都无所谓」。

⚠️ **免疫的是各向同性误差,不是各向异性误差**。x 与 y 是两根独立的压电,
``s_x ≠ s_y`` 会改角度、会改不同取向峰之间的周期比,而且**随扫描框角度变化** ⇒
同一个畴在两个帧角下报成两个畴。这条只能由真机(不同帧角各扫一张)判,合成语料
造不出它 —— 见设计 §6 R4。

## 角度必须归到样品系,帧角读不到就拒判

``stripe_peak`` 报的角是**从快扫轴量起**的。扫描框一转,同一个畴的指纹整体旋转::

    k_angle_sample_deg = (k_angle_frame_deg + scan_angle_deg) mod 180

⚠️ **符号(``+`` 还是 ``-``)靠真机实测钉住,不能推**:``stripe_peak`` 的角是
「+列 → 行号增大」,而仪器 ``SCAN_ANGLE`` 的正方向未经核对。合成语料按 ``+`` 的
约定生成,所以 ``test_frame_rotation_does_not_change_the_fingerprint`` 证明的是
**归一化确实被执行了**,不是**约定是对的** —— 后者只有真机能答
(``test_angle_sign_convention_matches_the_instrument``,已 skip,名字里写明在等哪个数)。

帧角**读不到** ⇒ ``unknown_frame_angle``,``k_angle_sample_deg`` 留 ``None``,
**不按 0° 处理**。把 ``None`` 折叠成 ``0.0`` 会让同一个畴在两种帧角下报成两个畴,
而且零报错。

## 只吃**原生采集帧** —— 尺度门会被零信息注入骗过去

``scale_gate`` 键在**标称** ``nm/px`` 上,不是实测分辨率:双线性升采样 ×2 把
``nm_per_px`` 减半、**不带任何新信息**,却能把一帧从 ``off`` 抬进 ``full``。
对真实采集的帧这道门是诚实的(``nm_per_px`` 来自仪器头);只要输入可能被重采样过,
它就等于没有。⇒ :func:`extract_fingerprint` 的入参约定是**原生采集帧**,调用方
重采样过就必须传 ``native_sampling=False``(直接 ``resampled_input`` 拒判),并且
在能查的维度上查一道:高频功率被插值核抹掉时挂 ``possible_resampled_input`` 警告
(**警告不是拒判** —— 一帧本来就很平滑的真帧也会低高频,拿它当硬门会误伤)。

指纹本身对升采样是不变的(见下面的四检验 ①)——**会被骗的是门,不是指纹**。

## 慢轴盲区:标准平场会**吃掉**一个方向,而且零报错

``flatten_robust = level_iterative(align_rows_mediandiff(h))``,而
``align_rows_mediandiff`` 减掉逐行中值差的累积和 —— 它**正是**用来杀行偏移条纹的,
所以 **k 沿慢轴的那个晶格方向在判据看到它之前就被减没了**。合成实测:一个六角
晶格转到「某个方向恰好落在慢轴上」时,三个峰只报得出两个;落在缺口**边上**的峰
活得下来但被拉偏(真值 89.0° 报成 87.2°)。

后果分两种,必须分开说:

* **六角**(``symmetry_deg=60``)丢一个峰**不影响比对** —— 三个方向 mod 60 本来
  就等价(同畴两个帧角下 d = 0.0002);
* **非六角**(矩形 90° 等)丢的那个峰带着一个独立周期,比对**会偏**:实测同一个畴
  在帧角 0° 与 30° 下 d = 0.029,而漂移受控时的同畴分母只有 0.040 —— 已经是同一
  量级。

所以缺口里有东西而平场后没有时挂 ``slow_axis_blind_spot`` 警告(**说明判据没查
什么**,不是判据失败),下一步是把扫描框转 ~30° 重扫。**不把对照里的峰塞进指纹**:
那个位置上真实晶格与扫描线伪影在一帧图里分不开(``herringbone`` 的既有结论)。

## 剪切:``fast_axis_period_nm`` 那道复核**抓不到它**(实测)

设计里原本指望「快轴周期 vs 二维谱峰的投影」能复核剪切。**实测证伪**:剪切沿快扫
轴平移每一行,``k_x`` 分量不变 ⇒ 快轴周期不变,而峰投影 ``T/|cos φ| = 1/|k_x|``
**同样不变**。剪切 0→0.5 px/行,两者的分歧一直是 0.041(那 0.041 是逐行 1D 谱的
频率量化,不是信号)。**两个量对同一件事都免疫,所以它们对不上时说的不是剪切。**

那道复核仍然留着(它抓的是径向峰锁到了谐波/moiré/带沿上,码名沿用设计的
``shear_suspect``,下一步对两种成因都成立),但真正看得见剪切的是
``notes["period_spread"]``(峰周期的 max/min):剪切 0/0.1/0.2/0.3/0.5 px/行 →
1.001/1.085/1.182/1.292/1.540,单调。**它刻意不设阈值**:不知道样品对称性就分不开
「被剪切的六角」与「本来就斜方」——那是定理不是实现缺陷。要判它得拿
``DomainReference.symmetry_deg`` 当外部事实,那是下一轮的事。

## 判据有效性四检验:可重复 ≠ 有效

(本仓 2026-08-14 结案:一个判据可以稳定、可重复、看起来合理,同时**量的根本不是
你要的那个东西**。所以分离度之前先跑这四条,全部在合成数据上跑。)

===  ==============================================================  ===================
 ①   零信息注入:双线性升采样 ×2 后重跑                              角度/周期逐项不变
 ②   信息剥夺:512 → 256 → 128 → 64 px                              找到指纹开始变的那档
 ③   内部常数扫描:``_CONC_BINS`` / 环宽 / 质心窗 / 排他半径 …        指纹不正比于任何一个
 ④   物理常数裁判:外部标尺(晶格常数、已知孪晶角)                  判据自己的一致性不算证据
===  ==============================================================  ===================

合成实测(晶格周期 0.2494 nm、起伏 10 pm、噪声 1-10 pm、5 nm / 256 px = 0.0195 nm/px):

    ① 升采样 ×2       角度 Δ ≤ **0.002°**、周期相对 Δ ≤ **3e-5**、rel_power Δ ≤ 0.001
                      —— 指纹对零信息注入是不变的。**被骗的是尺度门**:同一帧
                      0.06 nm/px 判 ``off``,升采样后 0.03 nm/px 就成了 ``reduced``
    ② 信息剥夺        固定 2.5 nm 视野,512 → 256 → 128 → 64 px:指纹几乎不动
                      (角度 Δ ≤ 0.031°,d ≤ 0.0003);32 px(3.2 px/周期)撞尺度门
                      拒判。⇒ **指纹要的不是像素数,是 nm/px**;满权重档以上再加像素
                      对指纹没有增量
    ③ 内部常数        质心窗 3×3→5×5→7×7 角度 Δ ≤ 0.065°;环排他半径 3/5/8 bin
                      指纹**完全不变**(2 bin 时同一个峰被数两次 —— 那正是默认取 3
                      的推导);功率下限 0.02→0.2 不变
    ③ 角向 bin 数     ``_CONC_BINS`` 36/72/144 指纹**逐项不变**(方向来自二维谱峰的
                      亚像素质心,不来自角向 bin);同一条测试里跑的**替身**(用 bin
                      中心报方向)则随 bin 数跳 ≥ 360/144 —— 这条理由由扫描证明,
                      不是转述
    ④ 外部标尺        合成语料的真值就是判据外面的尺子:真周期报数误差 ≤ **0.11%**,
                      真取向误差 ≤ **0.06°**,已知取向差 25° 测出 **25.01°**。
                      真机上换成晶格常数与(若有)已知孪晶角

## 分离度(D6):分母比分子重要

    separation = 组间最小距离 / 同畴帧对最大距离

**一个判据在正例上有信号不算数,要证明它在反例上没有**(本仓两次同款教训:
带通抖动对照、纯白噪声对照)。四组对照的合成实测:

    ① 同畴不同帧(噪声 1-10 pm、剪切 0-0.3 px/行、行偏移、随机相位)
                              d ≤ **0.1167**   ← **这就是分母**
                              剪切压到 ≤0.1 px/行 时 d ≤ **0.0401**
    ② 纯取向差 Δθ=2/5/10/20/30°  d = 0.033 / 0.084 / 0.167 / 0.333 / 0.499(单调)
    ③ 排布不同               周期比 1.15 → 0.065;六角 vs 矩形 → 0.150
    ④ 反例(纯白噪声 ×10、带通抖动 ×10、死平帧、尺度门 off)
                              **22 帧 22 个 undetermined,一个都没有落成新畴**

**分母几乎全部来自慢轴剪切,不是判据**:同一组里只变噪声 d ≤ 0.007、只变行偏移
0.0005、只换随机相位 0.0005,而只变剪切 0→0.3 px/行 就是 0.117。所以

===============  ==========  ================================================
 漂移条件          分母        分得开什么
===============  ==========  ================================================
 剪切 ≤0.3 px/行   0.1167     Δθ ≥ 10°、六角 vs 矩形。**周期比 1.15 分不开**
 剪切 ≤0.1 px/行   0.0401     Δθ ≥ 5°、周期比 1.15、六角 vs 矩形
===============  ==========  ================================================

    ⇒ 分离度(Δθ=5° / 同畴分母)= **0.72**(高剪切)/ **2.09**(漂移受控)
       分离度(Δθ=10°)        = **1.43**(高剪切)/ **4.17**(漂移受控)

**要提高取向分辨率,拧的是扫描速度/漂移,不是判据里的常数** —— 这也正是
``shear`` 那一族 ``undetermined`` 给出「降扫速/等漂移稳」的原因。

真机标定接在离线之后,且**真机分离度不达标不算判据成立**。

## 三态,不是两态

``verdict ∈ {label_i…} ∪ {"mixed", "undetermined"}``。``undetermined`` **必须细分**
(见 :data:`UNDETERMINED_REASONS`)——「判不了」和「没有」是两句话,前者写成后者会
让上层接着在同一个位置扎针/换点。

``mixed`` 与 ``ambiguous_match`` 是**两个判据不是一个阈值**:``mixed`` 要求
「A 的峰**和** B 的峰**都在**」(各自覆盖率 ≥ ``mixed_coverage_min``),
``ambiguous_match`` 是「哪个的峰都不全在,且次近-最近 < ``ambiguity_margin``」。
一帧跨在畴界上、和一帧判不了,长得一样但要说不同的话 —— 而 ``mixed`` 是找畴界时
**最有价值的信号**(畴界就在这一帧里)。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from mast.vision.atomic_phase import (
    ATOMIC_BAND_NM,
    DEFAULT_CONCENTRATION_MIN,
    scale_gate,
)

if TYPE_CHECKING:  # pragma: no cover — 只为类型,运行时不导入(避免环)
    from mast.vision.domain_reference import DomainReference

logger = logging.getLogger(__name__)

#: 慢轴缺口(度)。k 落在慢轴 ±这个角度内 ⇒ 与扫描线伪影不可分,要跑对照。
#: 沿用 ``herringbone.SLOW_AXIS_NOTCH_DEG`` 的推导(角向 bin 宽 5° + 峰的角向半宽)。
SLOW_AXIS_NOTCH_DEG = 10.0

#: 谱峰互相排他的半径(FFT bin)。Hann 窗主瓣半宽是 2 bin,取 3 让同一个峰不会被
#: 数两次;而六角晶格相邻布拉格点在环上隔着 ``2·r·sin(30°) = r`` 个 bin
#: (原子尺度帧上 r ≈ 20),所以 3 远不至于把真峰吃掉。四检验 ③ 扫 2..5。
PEAK_EXCLUSION_BINS = 3.0

#: 峰表的功率下限(÷ 最强峰)。低于它的峰在噪声里,进指纹只会让距离变吵。
MIN_REL_POWER = 0.05

#: 指纹里最多几个峰。六角晶格 mod 180 后是 3 个方向,矩形 2 个 —— 6 留了余量。
MAX_PEAKS = 6

#: 快轴周期与二维谱峰投影的最大相对分歧。超过它挂 ``shear_suspect``。
#: 由慢轴剪切的已知量级定:0.30 px/行时快轴周期变化 < 1%,而径向周期偏 ~10%
#: (``atomic_phase`` 模块注释的实测)⇒ 门要放在 10% 之上、明显剪切之下。
SHEAR_TOL_FRAC = 0.25

#: ``undetermined`` 的细分码(闭集)。每一条都要有**可执行的下一步** ——
#: 「判不了」不给下一步等于死路。
UNDETERMINED_REASONS: tuple[str, ...] = (
    "unknown_pixel_size",     # 不知道 nm/px          → 补 header/几何来源
    "scale_gate",             # 像素太粗,分辨不了晶格 → 换更小视野/更多像素
    "scale_reduced",          # 过渡带,测量本身降级   → 同上,或显式 allow_reduced_scale
    "too_few_periods",        # 帧里周期太少,核心判据静默返回 0 → 换**更大**视野
    "insufficient_data",      # 帧太小/不是二维       → 换一帧
    "resampled_input",        # 调用方声明重采样过     → 拿原生采集帧来
    "unknown_frame_angle",    # 帧角读不到            → 补角度来源(**不按 0° 处理**)
    "no_atomic_phase",        # 这一帧根本没有原子分辨 → 换点/修针(**不是新畴**)
    "no_peaks",               # 带里没有可用谱峰      → 同上
    "slow_axis_degenerate",   # 主峰落慢轴缺口且对照对不上 → **转 ~30° 重扫**
    "shear_suspect",          # 快轴周期与径向不一致   → 降扫速/等漂移稳
    "no_reference",           # 还没有标定过的参照系   → 走人机协作建参照系
    "reference_unusable",     # 参照系里没有可比的原型 → 修参照系文件
    "ambiguous_match",        # 到每个原型都差不多,且哪个的峰都不全在 → 重采/换点
    "no_match",               # 像谁都不像(可能是第三簇)→ **升级问人**,别自动加簇
)


@dataclass(frozen=True)
class DomainPeak:
    """指纹里的一个「方向-周期」峰。

    ``k_angle_sample_deg`` 是**比对唯一用的那个角**;帧角读不到时它是 ``None``
    (不是 0.0 —— 见模块注释)。``on_slow_axis`` 是**帧相关**的元数据,不参与
    距离:它随扫描框角度变,而指纹按定义不该随扫描框变。
    """

    k_angle_frame_deg: float
    k_angle_sample_deg: float | None
    period_nm: float
    rel_power: float
    on_slow_axis: bool = False

    def as_triple(self) -> tuple[float, float, float]:
        """``(样品系角度, 周期nm, 相对功率)`` —— 落库/参照系用的紧凑形式。"""
        if self.k_angle_sample_deg is None:
            raise ValueError("帧角未知的峰没有样品系角度,不可用于比对")
        return (float(self.k_angle_sample_deg), float(self.period_nm),
                float(self.rel_power))


@dataclass(frozen=True)
class DomainFingerprint:
    """一帧上的畴指纹。``peaks`` 按功率降序。

    **拒判时与帧内容无关的量照报**(尺度、几何、集中度)——「测出来是零」和
    「没测」必须是两句话。
    """

    peaks: tuple[DomainPeak, ...] = ()
    n_peaks: int = 0
    scan_angle_deg: float | None = None      # None ⇒ 整个指纹不可用于比对
    nm_per_px: float | None = None
    scale: str | None = None                 # full / reduced / off / None
    periods_in_frame: float | None = None
    period_fast_axis_nm: float | None = None  # atomic_phase 的独立复核
    period_fast_axis_predicted_nm: float | None = None  # 峰投影出来的预期值
    angular_concentration: float = 0.0
    atomic_passed: bool = False              # assess_atomic_phase 的结论,转发不重算
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: dict = field(default_factory=dict)

    @property
    def comparable(self) -> bool:
        """能不能拿去比对。任何一条 reason 都让它变成不可比。"""
        return bool(self.peaks) and self.scan_angle_deg is not None and not self.reasons

    @property
    def blocking_reason(self) -> str:
        """不可比的**第一个**原因(闭集里的码);可比时是空串。"""
        if self.reasons:
            return self.reasons[0]
        if self.scan_angle_deg is None:
            return "unknown_frame_angle"
        if not self.peaks:
            return "no_peaks"
        return ""

    def triples(self) -> tuple[tuple[float, float, float], ...]:
        """样品系峰表。不可比时返回空 —— 半个指纹不许流出去。"""
        if not self.comparable:
            return ()
        return tuple(p.as_triple() for p in self.peaks)


@dataclass(frozen=True)
class DomainVerdict:
    """一帧的畴判定。``verdict`` 是产品,``label`` 只在认出来时非空。"""

    label: str | None
    verdict: str                             # label / "mixed" / "undetermined"
    reason: str                              # undetermined 时是闭集里的细分码
    distances: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    margin: float = float("inf")             # 次近 - 最近(只有一个原型时是 inf)
    reference_version: str | None = None
    fingerprint: DomainFingerprint = field(default_factory=DomainFingerprint)


# ── 谱峰:方向与周期的唯一来源 ──────────────────────────────────────────────

def _power_spectrum(image: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Hann 窗 + 二维功率谱(fftshift 过)。

    ⚠️ 与 ``herringbone.stripe_peak`` 的前半段**逐字相同**,是刻意保持同步的一份
    拷贝(那个函数只给一个峰,拿不到谱,没法在它上面取多峰)。同步性由
    ``test_strongest_ring_peak_matches_stripe_peak`` 钉着:本模块最强的那个峰必须
    与 ``stripe_peak`` 逐位一致。
    """
    x = image - image.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    return np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2


def ring_peaks(
    image: npt.ArrayLike,
    band_px: tuple[float, float],
    *,
    max_peaks: int = MAX_PEAKS,
    min_rel_power: float = MIN_REL_POWER,
    exclusion_bins: float = PEAK_EXCLUSION_BINS,
    centroid_half_window: int = 1,
) -> tuple[tuple[float, float, float], ...]:
    """径向窗内最强的那几个谱峰 → ``((周期px, 方向deg, 相对功率), ...)``,按功率降序。

    与 ``herringbone.stripe_peak`` 同一套配方(窗、谱、径向窗、``centroid_half_window``
    邻域的功率加权质心亚像素细化),区别只在**取多个峰**:每取一个就把它与它的镜像
    ``-k`` 各挖掉一个半径 ``exclusion_bins`` 的圆盘(实图的功率谱严格中心对称,
    ±k 是同一个方向 mod 180,不挖会把同一个方向数两次)。

    相对功率用邻域**求和**而不是单点:Hann 窗的扇贝损失(峰不在整数 bin 上时功率
    被分给邻居)对单点读数有 ~1.4 dB 的抖动,而峰表要在不同帧之间比。

    角度从**快扫轴(+列)**量到**行号增大**方向,mod 180 —— 与 ``stripe_peak`` 同。
    """
    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 16:
        return ()
    lo, hi = float(band_px[0]), float(band_px[1])
    if not (lo > 0 and hi > lo):
        return ()
    n_max = max(0, int(max_peaks))
    if n_max == 0:
        return ()

    H, W = h.shape
    P = _power_spectrum(h)
    cy, cx = H // 2, W // 2
    # 频率按各自的轴长归一 —— 非方帧上 dy/H 与 dx/W 的标度不同。
    fy = (np.arange(H) - cy)[:, None] / float(H)
    fx = (np.arange(W) - cx)[None, :] / float(W)
    fr = np.hypot(fy, fx)
    sel = (fr >= 1.0 / hi) & (fr <= 1.0 / lo)
    if not sel.any():
        return ()

    yy_i, xx_i = np.mgrid[0:H, 0:W]
    avail = sel.copy()
    half = max(0, int(centroid_half_window))
    found: list[tuple[float, float, float]] = []
    powers: list[float] = []

    for _ in range(n_max):
        masked = np.where(avail, P, -np.inf)
        if not np.isfinite(masked).any():
            break
        iy, ix = np.unravel_index(int(np.argmax(masked)), P.shape)
        y0, y1 = max(0, iy - half), min(H, iy + half + 1)
        x0, x1 = max(0, ix - half), min(W, ix + half + 1)
        w = P[y0:y1, x0:x1]
        tot = float(w.sum())
        if tot > 0:
            yv = np.arange(y0, y1, dtype=np.float64)[:, None]
            xv = np.arange(x0, x1, dtype=np.float64)[None, :]
            fy_r = (float((w * yv).sum() / tot) - cy) / float(H)
            fx_r = (float((w * xv).sum() / tot) - cx) / float(W)
        else:  # pragma: no cover — argmax 自己在窗里,权重和不会是 0
            fy_r, fx_r = float(fy[iy, 0]), float(fx[0, ix])
        f = math.hypot(fy_r, fx_r)
        # 挖掉这个峰与它的镜像(先挖再判,免得死循环)。
        for py, px in ((iy, ix), (2 * cy - iy, 2 * cx - ix)):
            avail &= np.hypot(yy_i - py, xx_i - px) > float(exclusion_bins)
        if f <= 0:  # pragma: no cover — 窗把 DC 排除在外
            continue
        found.append((1.0 / f, float(math.degrees(math.atan2(fy_r, fx_r)) % 180.0),
                      0.0))
        powers.append(tot)

    if not found:
        return ()
    top = max(powers) or 1.0
    out = [(t, a, p / top) for (t, a, _), p in zip(found, powers)
           if p / top >= float(min_rel_power)]
    return tuple(out)


def _predicted_fast_axis_period_nm(
    peaks: tuple[tuple[float, float, float], ...], nm_per_px: float
) -> float | None:
    """峰表投影到快扫轴上的**最小**周期(纳米),用来和逐行 1D 谱对账。

    一个方向 φ(从快扫轴量起)、周期 T 的波,在快扫方向上的周期是 ``T/|cos φ|``;
    ``fast_axis_period_nm`` 取的正是「强峰里周期最小的那个」,所以这里也取 min。
    """
    best: float | None = None
    for period_px, angle_deg, _rel in peaks:
        c = abs(math.cos(math.radians(angle_deg)))
        if c < 1e-3:                      # k 几乎沿慢轴 —— 快轴上看不到它
            continue
        t = float(period_px) * float(nm_per_px) / c
        if best is None or t < best:
            best = t
    return best


def _high_frequency_starved(image: npt.NDArray[np.float64]) -> tuple[bool, float]:
    """``(像不像被插值升采样过, 高频/中频功率比)``。

    双线性升采样把原 Nyquist 以上的功率几乎抹平。这是**能查的那个维度** ——
    但**只挂警告不拒判**:一帧本来就很平滑(针尖钝、表面平)的真帧也会低高频,
    拿它当硬门会误伤真数据。声明式的 ``native_sampling=False`` 才是硬门。
    """
    P = _power_spectrum(image)
    H, W = P.shape
    cy, cx = H // 2, W // 2
    fy = (np.arange(H) - cy)[:, None] / float(H)
    fx = (np.arange(W) - cx)[None, :] / float(W)
    fr = np.hypot(fy, fx)
    mid = (fr >= 0.10) & (fr <= 0.20)
    outer = fr >= 0.35
    if not mid.any() or not outer.any():  # pragma: no cover — 16 px 以上必有
        return False, float("nan")
    m = float(np.median(P[mid]))
    o = float(np.median(P[outer]))
    if not (m > 0):
        return False, float("nan")
    ratio = o / m
    return bool(ratio < 1e-3), float(ratio)


def extract_fingerprint(
    image: npt.ArrayLike,
    *,
    nm_per_px: float | None,
    scan_angle_deg: float | None,
    max_peaks: int = MAX_PEAKS,
    band_nm: tuple[float, float] = ATOMIC_BAND_NM,
    concentration_min: float = DEFAULT_CONCENTRATION_MIN,
    slow_axis_notch_deg: float = SLOW_AXIS_NOTCH_DEG,
    min_rel_power: float = MIN_REL_POWER,
    exclusion_bins: float = PEAK_EXCLUSION_BINS,
    centroid_half_window: int = 1,
    shear_tol_frac: float = SHEAR_TOL_FRAC,
    allow_reduced_scale: bool = False,
    native_sampling: bool = True,
) -> DomainFingerprint:
    """一帧 → 畴指纹。纯函数:不读配置、不碰硬件、不抛异常。

    **入参约定:``image`` 是原生采集帧**(``nm_per_px`` 来自仪器头)。裁剪过可以,
    重采样/插值过不行 —— 调用方重采样过就传 ``native_sampling=False``,那会直接
    ``resampled_input`` 拒判。理由见模块注释「只吃原生采集帧」:``scale_gate`` 键在
    标称 ``nm/px`` 上,升采样能把一帧从 ``off`` 抬进 ``full`` 而不带任何新信息。

    ``scan_angle_deg`` 是扫描框角度(度)。**读不到就传 ``None``,不要传 0.0** ——
    见模块注释。

    产出:``peaks`` 按功率降序;``reasons`` 非空 ⇒ 这个指纹**不可比**
    (:attr:`DomainFingerprint.comparable`),但与帧内容无关的读数照报。
    """
    reasons: list[str] = []
    warns: list[str] = []
    notes: dict = {}

    h = np.asarray(image, dtype=np.float64)
    if h.ndim == 3 and h.shape[0] in (1, 2):
        h = h[0]

    scale = scale_gate(nm_per_px)
    nmpp = float(nm_per_px) if scale is not None else None
    angle = (float(scan_angle_deg) % 180.0) if scan_angle_deg is not None else None

    def _out(**kw) -> DomainFingerprint:
        return DomainFingerprint(
            scan_angle_deg=angle, nm_per_px=nmpp, scale=scale,
            reasons=tuple(reasons), warnings=tuple(warns), notes=notes, **kw)

    if not native_sampling:
        # 声明式的硬门。放在最前面:重采样过的帧连尺度门都不该信。
        reasons.append("resampled_input")
        return _out()
    if scale is None:
        reasons.append("unknown_pixel_size")
        return _out()
    if scale == "off":
        reasons.append("scale_gate")
        return _out()
    if h.ndim != 2 or min(h.shape) < 16:
        reasons.append("insufficient_data")
        return _out()
    if not np.isfinite(h).all():
        med = np.nanmedian(h) if np.isfinite(h).any() else 0.0
        h = np.nan_to_num(h, nan=float(med) if np.isfinite(med) else 0.0)

    # 帧里装得下几个周期。低于这个下限,``angular_concentration`` 的守卫会让核心
    # 判据**整条静默返回 0** —— 那不是「没有畴」,是这一帧问不了这个问题。
    # 下限直接引 herringbone 解出来的那个数(单一真源,抄错了那边的测试会红)。
    from mast.vision.herringbone import PERIODS_IN_FRAME_OFF

    band = (float(band_nm[0]), float(band_nm[1]))
    periods_in_frame = float(min(h.shape)) * nmpp / band[0] if band[0] > 0 else None
    # 先用带的**下沿**(最短周期)算 —— 这是「最多能有几个周期」的乐观估计:
    # 连它都够不着下限,这一帧对整条带都问不了,可以立刻退。真正卡住的那一刀在
    # 下面,用**实测**周期再来一次(拿上沿当预检会把好帧误拒:带宽 0.18-0.80 nm
    # 而晶格是 0.25 nm 时,上沿算出来的周期数只有实际的三分之一)。
    if periods_in_frame is not None and periods_in_frame < PERIODS_IN_FRAME_OFF:
        reasons.append("too_few_periods")
        return _out(periods_in_frame=periods_in_frame)

    if angle is None:
        # 先记着,但**继续算** —— 帧角读不到时指纹本身仍是有用的诊断,只是不可比。
        reasons.append("unknown_frame_angle")

    starved, hf_ratio = _high_frequency_starved(h)
    notes["hf_power_ratio"] = hf_ratio
    if starved:
        warns.append("possible_resampled_input")

    try:
        from mast.vision.atomic_phase import assess_atomic_phase
        from mast.vision.seg_scale_adaptive import flatten_robust, level_iterative
    except Exception as exc:  # noqa: BLE001 — 依赖缺席就是「判不了」
        logger.debug("畴指纹依赖缺席: %s", exc)
        reasons.append("insufficient_data")
        notes["dependency_error"] = f"{type(exc).__name__}: {exc}"
        return _out(periods_in_frame=periods_in_frame)

    # ── 「这一帧有没有原子相」直接**转发**,不重算一套 ──
    atomic = assess_atomic_phase(
        h, nm_per_px=nmpp, concentration_min=float(concentration_min),
        band_nm=band, allow_reduced_scale=True)
    if not atomic.passed:
        reasons.append("no_atomic_phase")
        notes["atomic_reasons"] = list(atomic.reasons)

    if scale == "reduced":
        warns.append("scale_reduced")
        if not allow_reduced_scale:
            reasons.append("scale_reduced")

    flat = flatten_robust(h)
    band_px = (max(3.0, band[0] / nmpp), band[1] / nmpp)
    raw_peaks = ring_peaks(
        flat, band_px, max_peaks=int(max_peaks), min_rel_power=float(min_rel_power),
        exclusion_bins=float(exclusion_bins),
        centroid_half_window=int(centroid_half_window))
    if not raw_peaks:
        reasons.append("no_peaks")
        return _out(periods_in_frame=periods_in_frame,
                    period_fast_axis_nm=atomic.period_fast_axis_nm,
                    angular_concentration=atomic.angular_concentration,
                    atomic_passed=atomic.passed)

    # 实测周期定的那个下限 —— 环的半径(像素)就是帧里的周期数。低于它,
    # ``angular_concentration`` 整条**静默返回 0**,于是这一帧无论有什么都会以
    # 「没有原子相」落选。它排在 ``no_atomic_phase`` **前面**是有意的:两条码的
    # 下一步相反(修针/换点 vs 换**更大**的帧),而根因是这一条。
    periods_in_frame = float(min(h.shape)) / float(raw_peaks[0][0])
    if periods_in_frame < PERIODS_IN_FRAME_OFF:
        reasons.insert(0, "too_few_periods")

    # ── 慢轴退化对照:标准平场自己会吃掉一个方向,还会凭空造一个 ──
    #
    # ``flatten_robust = level_iterative(align_rows_mediandiff(h))``,而
    # ``align_rows_mediandiff`` 减掉逐行中值差的**累积和**:真实行偏移为零时那串
    # 中值差是噪声,累积和是一条**随机游走**,全落在 kx=0 上而且相干 —— 它能在纯
    # 白噪声上造出一个带内合格的慢轴峰(SNR 门与角向集中度门两道都拦不住)。
    # 分开真假的唯一办法是「另一条路(只跑 level_iterative、不做行对齐)对不对得上」。
    # 对照**每一帧都跑**,因为它同时回答两个问题(见下面两段)。
    ctrl = ring_peaks(
        level_iterative(h), band_px, max_peaks=int(max_peaks),
        min_rel_power=float(min_rel_power), exclusion_bins=float(exclusion_bins),
        centroid_half_window=int(centroid_half_window))
    notes["slow_axis_control_peaks"] = len(ctrl)

    def _in_notch(a: float) -> bool:
        return abs((float(a) % 180.0) - 90.0) <= float(slow_axis_notch_deg)

    dom_t, dom_a, _dom_p = raw_peaks[0]
    if _in_notch(dom_a):
        corroborated = any(
            _in_notch(a) and abs(t - dom_t) <= 0.2 * max(t, dom_t) for t, a, _ in ctrl)
        if not corroborated and atomic.passed:
            # ⚠️ 只在**原子相判据已经过了**的帧上说这句话。纯噪声帧上标准平场同样
            # 造得出这个峰(实测 20 个种子里 19 个),但那时正确答案是「这儿什么都
            # 没有」,不是「转个角度再扫一张」—— herringbone 的
            # ``test_pure_noise_is_absent_not_undetermined`` 就是被这一条钉住的。
            reasons.append("slow_axis_degenerate")
            notes["slow_axis"] = (
                f"最强峰指着慢轴({dom_a:.0f}°),但不做行对齐时它不存在 —— "
                f"它是 align_rows_mediandiff 的累积中值差(随机游走)造出来的,"
                f"不是表面结构。把扫描框转 ~30° 重扫一张再判。")

    # 慢轴盲区:``align_rows_mediandiff`` 会把 k 沿慢轴的分量**在判据看到它之前就
    # 减掉**。实测:一个六角晶格转到某个方向恰好落在慢轴上时,它的三个峰只报得出
    # 两个 —— 而且零报错。这是**说明判据没查什么**,不是判据失败:
    #
    # * 六角(symmetry 60°)下丢一个峰不影响比对 —— 三个方向 mod 60 本来就等价;
    # * **非六角**(矩形 90° 等)下丢的那个峰带着一个独立的周期,比对会偏。
    #
    # 所以这里报一条警告 + 把对照在盲区里看到的东西记进 notes,让上层能据此决定
    # 要不要转 ~30° 重扫。**不把对照里的峰塞进指纹**:那个位置上真实晶格与扫描线
    # 伪影在一帧图里分不开(herringbone 的既有结论),塞进去等于把伪影当结构。
    if not any(_in_notch(a) for _t, a, _p in raw_peaks):
        blind = [(t, a) for t, a, _p in ctrl if _in_notch(a)]
        if blind:
            warns.append("slow_axis_blind_spot")
            notes["slow_axis_blind_spot"] = (
                f"标准平场后慢轴缺口(90°±{slow_axis_notch_deg:g}°)里没有峰,但不做"
                f"行对齐时那里有 {len(blind)} 个(周期 "
                f"{', '.join(f'{t * nmpp:.3f}' for t, _ in blind)} nm)。那个方向上真实"
                f"晶格与扫描线伪影一帧图里分不开,所以它没有进指纹 —— 样品的对称性"
                f"要求那里有一个峰的话,把扫描框转 ~30° 重扫。")

    # 峰周期的散布 —— 剪切**能被看见**的那个量(下面那道快轴复核看不见它)。
    # 合成实测(剪切 px/行 → max/min):0.00→1.001, 0.10→1.085, 0.20→1.182,
    # 0.30→1.292, 0.50→1.540 —— 单调。
    # **刻意不设阈值**:一个真正斜方的晶格本来就有不同的周期,不知道样品对称性
    # 就分不开「被剪切的六角」与「本来就斜方」(这是定理不是实现缺陷)。报数,
    # 让读的人自己看;要判它得拿 ``DomainReference.symmetry_deg`` 当外部事实。
    _pers = [t * nmpp for t, _a, _p in raw_peaks]
    if len(_pers) >= 2 and min(_pers) > 0:
        notes["period_spread"] = float(max(_pers) / min(_pers))

    # ── 周期复核:快轴周期(对慢轴漂移免疫)与峰投影对不对得上 ──
    #
    # **不是从两个数里选一个**:``fast_axis_period_nm`` 只做复核,指纹的周期一律
    # 来自二维谱峰(它与方向配对;快轴那个数按定义只有一个,配不了方向)。
    #
    # ⚠️ **这道复核抓不到剪切,实测过**:剪切沿快扫轴平移每一行,``k_x`` 分量不变
    # ⇒ 快轴周期不变,而峰投影 ``T/|cos φ| = 1/|k_x|`` **同样不变**。合成实测:
    # 剪切 0→0.5 px/行,分歧一直是 0.041(那 0.041 是逐行 1D 谱的频率量化,不是
    # 信号)。**两个量对同一件事都免疫,所以它们对不上时说的不是剪切** —— 它抓的
    # 是径向峰锁到了别的周期上(谐波、moiré、带沿)。码名沿用设计的
    # ``shear_suspect``(下一步「降扫速/等漂移稳」对这两种成因都成立),但别把它
    # 读成「剪切检测器」。剪切要看上面的 ``period_spread`` + 样品对称性。
    pred = _predicted_fast_axis_period_nm(raw_peaks, nmpp)
    meas = atomic.period_fast_axis_nm
    if pred is not None and meas is not None and band[0] <= pred <= band[1]:
        disagree = abs(pred - meas) / max(pred, meas)
        notes["fast_axis_disagreement"] = float(disagree)
        if disagree > float(shear_tol_frac):
            reasons.append("shear_suspect")

    peaks = tuple(
        DomainPeak(
            k_angle_frame_deg=float(a),
            k_angle_sample_deg=(None if angle is None else float((a + angle) % 180.0)),
            period_nm=float(t) * nmpp,
            rel_power=float(p),
            on_slow_axis=bool(abs((a % 180.0) - 90.0) <= float(slow_axis_notch_deg)),
        )
        for t, a, p in raw_peaks
    )
    if any(p.on_slow_axis for p in peaks) and not _in_notch(dom_a):
        # 次强峰落在缺口里:**保留并标记**,不丢弃。丢弃会让指纹随扫描框角度变
        # (缺口是帧的属性,不是样品的),那正好毁掉样品系归一。
        warns.append("slow_axis_peak_present")

    return _out(
        peaks=peaks, n_peaks=len(peaks), periods_in_frame=periods_in_frame,
        period_fast_axis_nm=meas, period_fast_axis_predicted_nm=pred,
        angular_concentration=atomic.angular_concentration,
        atomic_passed=atomic.passed)


# ── 距离:商空间 + 两个相对量 ────────────────────────────────────────────────

def _circular_deg(delta: float, symmetry_deg: float) -> float:
    """``delta`` 在周期 ``symmetry_deg`` 的圆上到 0 的距离。"""
    s = float(symmetry_deg)
    d = abs(float(delta)) % s
    return min(d, s - d)


def _pair_cost(a: tuple[float, float, float], b: tuple[float, float, float], *,
               symmetry_deg: float, w_angle: float, w_period: float) -> float:
    """两个峰之间的代价 ∈ [0, 1]。两项都归一,总和再除以权重和。

    ``d_θ`` 是**差**、``d_T`` 是**比** —— 各向同性标定误差对两者都无效
    (见模块注释)。把 ``d_T`` 改成绝对差 nm 会让这条免疫性消失,变异测试钉住它。
    """
    d_theta = _circular_deg(a[0] - b[0], symmetry_deg) / (float(symmetry_deg) / 2.0)
    ta, tb = abs(float(a[1])), abs(float(b[1]))
    d_period = abs(ta - tb) / max(ta, tb) if max(ta, tb) > 0 else 1.0
    wsum = float(w_angle) + float(w_period)
    return (float(w_angle) * d_theta + float(w_period) * d_period) / wsum


def _directed(a: tuple, b: tuple, **kw) -> float:
    """``a`` 的每个峰到 ``b`` 里最近的那个,按 ``rel_power`` 加权平均。"""
    num = den = 0.0
    for pa in a:
        best = min(_pair_cost(pa, pb, **kw) for pb in b)
        w = max(float(pa[2]), 0.0)
        num += w * best
        den += w
    return num / den if den > 0 else 1.0


def peak_distance(
    peaks_a: tuple[tuple[float, float, float], ...],
    peaks_b: tuple[tuple[float, float, float], ...],
    *,
    symmetry_deg: float,
    w_angle: float = 1.0,
    w_period: float = 1.0,
) -> float | None:
    """两个样品系峰表之间的距离 ∈ [0, 1]。``None`` = 算不了(不是 0)。

    对称化(两个方向各算一次取平均),所以峰数不等时不会偏袒峰少的那一方。
    """
    if not peaks_a or not peaks_b:
        return None
    if not (symmetry_deg and float(symmetry_deg) > 0):
        return None
    if float(w_angle) < 0 or float(w_period) < 0 or (float(w_angle) + float(w_period)) <= 0:
        return None
    kw = dict(symmetry_deg=float(symmetry_deg), w_angle=float(w_angle),
              w_period=float(w_period))
    return 0.5 * (_directed(peaks_a, peaks_b, **kw) + _directed(peaks_b, peaks_a, **kw))


def peak_coverage(
    prototype_peaks: tuple[tuple[float, float, float], ...],
    peaks: tuple[tuple[float, float, float], ...],
    *,
    symmetry_deg: float,
    w_angle: float = 1.0,
    w_period: float = 1.0,
    tol: float,
) -> float:
    """原型的峰有多大比例(按 ``rel_power`` 加权)在这一帧里找得到,∈ [0, 1]。

    这是 ``mixed`` 的判据:``mixed`` 要求**两个原型的峰都在**,而
    ``ambiguous_match`` 是**哪个的峰都不全在**。一个距离阈值分不开这两件事。
    """
    if not prototype_peaks or not peaks or not (symmetry_deg and symmetry_deg > 0):
        return 0.0
    kw = dict(symmetry_deg=float(symmetry_deg), w_angle=float(w_angle),
              w_period=float(w_period))
    num = den = 0.0
    for pa in prototype_peaks:
        w = max(float(pa[2]), 0.0)
        den += w
        if min(_pair_cost(pa, pb, **kw) for pb in peaks) <= float(tol):
            num += w
    return num / den if den > 0 else 0.0


def fingerprint_distance(
    a: DomainFingerprint,
    b: DomainFingerprint,
    *,
    symmetry_deg: float,
    w_angle: float = 1.0,
    w_period: float = 1.0,
) -> float | None:
    """两个指纹之间的距离。``None`` = 至少一方不可比(**不是 0,也不是 inf**)。"""
    if not a.comparable or not b.comparable:
        return None
    return peak_distance(a.triples(), b.triples(), symmetry_deg=symmetry_deg,
                         w_angle=w_angle, w_period=w_period)


# ── 判定 ────────────────────────────────────────────────────────────────────

def classify(fp: DomainFingerprint, ref: "DomainReference | None") -> DomainVerdict:
    """指纹 + 参照系 → 判定。**没有参照系永远不出 label**。

    顺序是有讲究的:

    1. **指纹不可比先说** ⇒ 原样转发它自己的细分码(``scale_gate`` 绝不能被读成
       「这里没有畴」或「这里是新畴」)。它排在 ``no_reference`` **前面**是因为
       整个普查阶段每一帧都没有参照系 —— 那时 ``no_reference`` 是已知的、
       不可行动的;而「这一帧太粗/没有原子相/帧角读不到」才是当场能改的那件事。
       两条都不出 label,所以先后不影响「没标定就不许出 A/B」这条硬规矩。
    2. 没有参照系 / 参照系未标定 ⇒ ``undetermined(no_reference)``。
       ``match.*`` 全 0 表示**未标定**,不许当「零容差」用。
    3. **``mixed`` 先判**(两个原型的峰都在)—— 它是找畴界时最有价值的信号,
       而且它与 ``ambiguous`` 长得像,晚判会被距离阈值吃掉。
    4. 最近的原型在容差内、且与次近拉开 ``ambiguity_margin`` ⇒ 出 label。
    5. 在容差内但拉不开 ⇒ ``ambiguous_match``;**谁都不在容差内 ⇒ ``no_match``**
       (可能是第三簇 —— 这是「证据矛盾」,要升级问人,不许自动加一簇)。
    """
    empty: dict[str, float] = {}
    calibrated = ref is not None and getattr(ref, "calibrated", False)
    version = getattr(ref, "version", None) if calibrated else None
    if not fp.comparable:
        return DomainVerdict(label=None, verdict="undetermined",
                             reason=fp.blocking_reason or "no_peaks",
                             distances=empty, coverage=empty,
                             reference_version=version, fingerprint=fp)
    if not calibrated:
        return DomainVerdict(label=None, verdict="undetermined",
                             reason="no_reference", distances=empty, coverage=empty,
                             reference_version=None, fingerprint=fp)

    mine = fp.triples()
    kw = dict(symmetry_deg=ref.symmetry_deg, w_angle=ref.w_angle,
              w_period=ref.w_period)
    distances: dict[str, float] = {}
    coverage: dict[str, float] = {}
    for label in ref.labels:
        proto = ref.peaks_of(label)
        d = peak_distance(mine, proto, **kw)
        if d is None:
            continue
        distances[label] = float(d)
        coverage[label] = peak_coverage(proto, mine, tol=ref.match_tol, **kw)
    if not distances:
        return DomainVerdict(label=None, verdict="undetermined",
                             reason="reference_unusable", distances=empty,
                             coverage=empty, reference_version=ref.version,
                             fingerprint=fp)

    ordered = sorted(distances.items(), key=lambda kv: kv[1])
    best, best_d = ordered[0]
    margin = (ordered[1][1] - best_d) if len(ordered) > 1 else float("inf")

    covered = [lab for lab, c in coverage.items() if c >= ref.mixed_coverage_min]
    if len(covered) >= 2:
        # ⚠️ 「两家的峰都在」只有在**两家本来分得开**的时候才说明畴界在这一帧里。
        # 两个原型自己就相距 ≤ match_tol 时,任何一帧单畴图都会同时覆盖它们 ——
        # 那不是 mixed,是这个参照系分不开这两个畴。报成 mixed 会让二分以为
        # 「畴界就在脚下」并当场收敛到一个根本不存在的边界。
        separated = any(
            (pd := peak_distance(ref.peaks_of(x), ref.peaks_of(y), **kw)) is not None
            and pd > ref.match_tol
            for i, x in enumerate(covered) for y in covered[i + 1:])
        if separated:
            return DomainVerdict(label=None, verdict="mixed", reason="",
                                 distances=distances, coverage=coverage,
                                 margin=margin, reference_version=ref.version,
                                 fingerprint=fp)
        return DomainVerdict(label=None, verdict="undetermined",
                             reason="ambiguous_match", distances=distances,
                             coverage=coverage, margin=margin,
                             reference_version=ref.version, fingerprint=fp)
    if best_d > ref.match_tol:
        return DomainVerdict(label=None, verdict="undetermined", reason="no_match",
                             distances=distances, coverage=coverage, margin=margin,
                             reference_version=ref.version, fingerprint=fp)
    if margin < ref.ambiguity_margin:
        return DomainVerdict(label=None, verdict="undetermined",
                             reason="ambiguous_match", distances=distances,
                             coverage=coverage, margin=margin,
                             reference_version=ref.version, fingerprint=fp)
    return DomainVerdict(label=best, verdict=best, reason="", distances=distances,
                         coverage=coverage, margin=margin,
                         reference_version=ref.version, fingerprint=fp)


__all__ = [
    "MAX_PEAKS",
    "MIN_REL_POWER",
    "PEAK_EXCLUSION_BINS",
    "SHEAR_TOL_FRAC",
    "SLOW_AXIS_NOTCH_DEG",
    "UNDETERMINED_REASONS",
    "DomainFingerprint",
    "DomainPeak",
    "DomainVerdict",
    "classify",
    "extract_fingerprint",
    "fingerprint_distance",
    "peak_coverage",
    "peak_distance",
    "ring_peaks",
]
