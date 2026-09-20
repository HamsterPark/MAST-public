"""这一帧上有没有 Au(111) 的 22×√3 herringbone —— 大平台上的针尖验收判据。

## 为什么要有它

大平台图可能不含台阶，因而不能使用台阶边缘锐度作验收输入。
在具有相应重构的表面上，可改为检验 herringbone 周期性；表面先验与
成像条件必须由调用方确认，不能仅凭缺少台阶推断存在重构。

顺带补上一个空档:``AssessAtomicPhase`` 在 **0.1953 nm/px**(50 nm 帧 / 256 px)
上以 ``scale_gate`` 拒判 —— 而那正是 herringbone 的尺度。这个模块填的就是它拒判
的那个带。

## 判的是什么、不判什么

**判**:soliton 条纹的周期性 —— 22×√3 重构把表面沿 ⟨1-10⟩ 压缩 4.2%,多出来的
原子挤成一对 discommensuration 线(soliton),线对沿条纹法向以 ~6.3 nm 重复。这是
FFT 上唯一稳的观测量。

**不判**:zigzag(chevron)本身。全周期 ~30 nm 的折线是条纹的**相位调制**,在谱
上是 Bragg 峰两侧 Δk=1/L 的卫星峰,不是 |k|=1/L 处的独立峰。在 50 nm 帧上
Δk = 50/30 = 1.7 px —— **落在峰自己的宽度里面,物理上分不开**。所以这里报的是
「~6 nm 单轴条纹调制,与 Au(111) herringbone 的先验一致」,不是「看见 zigzag 了」。
帧小于 ``6 × chevron_prior``(≈180 nm)时会挂一条 ``chevron_not_resolvable``
警告 —— 这是**说明判据没查什么**,不是判据失败。

## 先验不是阈值

知识库里的 6.3 nm / 30 nm / 20 pm(``clean_metal.CONSTANTS["Au(111)"]`` 与
``MATERIALS["Au(111)"]["herringbone_imaging"]``)在这里**只当搜索窗的中心**,不
当判定门槛。周期、方向、起伏一律**报数**;要不要判「够好」由调用方给阈值。本仓
纪律:知识库的值是近似,随仪器/样品变(``knowledge_not_ground_truth``)。

## 尺度门:两个条件,而且它们在这个尺度上分家了

原子相判据一个 ``nm/px`` 就够,因为 0.02 nm/px 的 256 px 帧自动装得下 20 个晶格
周期。herringbone 不行:6.3 nm 的周期下,「像素够不够细」与「帧里装不装得下几个
周期」是**两个独立的条件**，必须分别检查。

(A) **每周期的像素数** ``px_per_period = T_prior / nm_per_px``

    * ``3``(``off``):复用的机器自己的硬底 —— ``seg_scale_adaptive._band_peak``
      与 ``detect_texture`` 都要求 ``T >= 3 px``,``atomic_phase.angular_concentration``
      对 ``period_px < 3`` 直接返回 0。低于它,**没有任何东西会被算出来**,而
      「算不出来」被写成「没有 herringbone」就是撒谎。
    * ``6``(``full``):像素积分对正弦的衰减因子是 ``sinc(π/T_px)`` ——
      T=3 时 0.827(起伏被低报 17%),T=6 时 0.955(低报 4.5%),T=8 时 0.975。
      要让 ``stripe_corrugation_pm`` 这个数**能当数用**,取 5% 偏差处 ⇒ 6 px。
      (STM 像素不是严格盒平均,所以这是偏差的**上界**。)

(B) **帧里装得下几个周期** ``periods_in_frame = min(H, W) × nm_per_px / T_prior``

    * ``5.35``(``off``):**核心判据自己的守卫解出来的**。
      ``angular_concentration`` 把功率谱在 ``|k| ≈ 1/T`` 的环上分 bin,开头一句
      ``if ring_px < n_bins: return 0.0``。环的半径(像素)恰好就是
      ``N / T_px = frame_nm / T_nm = periods_in_frame``,环面积 ``0.8π f0²``,于是
      ``0.8π · periods² ≥ 72 ⟺ periods ≥ 5.35``。低于它**判据整条静默返回 0**,
      每一帧都落进 ``not_a_stripe_pattern`` —— 不论图上有什么都报「没有条纹」。
    * ``6``(``full``):有限帧上条纹峰自己的角向半宽 ≈ ``atan(1/N_periods)``;
      N=6 时 9.5°,正好等于下面那道慢轴陷阱的 10° 缺口。少于 6 个周期,峰宽超过
      缺口,方向判定与缺口本身都不可信。

  ⚠️ 这里原本写的是 ``4``(``detect_texture`` 把周期带上沿夹在 ``min(H,W)/4``
  的那个硬底)。**那个数不够**:``_band_peak`` 单独什么都判不了 —— 实测它在纯白
  噪声上给出 SNR 9.5..27.8。真正在判的是角向集中度,所以门必须定在**集中度**失效
  的地方,不是定在找峰失效的地方。写成 4 时,5.08 个周期的帧配上
  ``allow_reduced_scale=True`` 会**稳定地**报 ``absent``,不论图上画的是什么。

**两个下限不是随手取的,它们正好是让搜索窗不被机器内部夹住的那两个数**:
``detect_texture`` 内部会做 ``a_lo = max(2.5, band_lo/nmpp)`` 与
``a_hi = min(min(H,W)/4, band_hi/nmpp)``。代入下面的默认窗 (0.7, 1.4)×先验:

    下沿不被夹 ⟺ 0.7 × px_per_period ≥ 2.5  ⟺ px_per_period ≥ 3.6   (< 6 ✓)
    上沿不被夹 ⟺ 1.4 × T₀/nmpp ≤ min(H,W)/4 ⟺ periods_in_frame ≥ 5.6 (< 6 ✓)

即:**满权重档 = 「我们要搜的那个窗完整地存在」**。这不是巧合,是把门定在这儿的
理由。

像素采样足够细仍不保证帧内周期数足够；视野过小时应拒判，不能报告结构不存在。

## 搜索窗:上沿必须 ≤ 2 × 下沿

``_band_peak`` 在「强峰」里取**周期最小**的那个(它的注释写明是为了不锁到 moiré
上)。herringbone 的剖面不是正弦 —— fcc 区 ~2/3、hcp 区 ~1/3,二次谐波很强。所以
只要窗里同时装得下 ``T`` 和 ``T/2``,报出来的就会**稳定地是半个周期**。

约束因此是精确的:窗 ``(αT₀, βT₀)`` 里任何真周期 ``T`` 的二次谐波 ``T/2`` 都必须
落在窗外 ⟺ ``βT₀/2 ≤ αT₀`` ⟺ **``β ≤ 2α``**。默认 (0.7, 1.4) 取等号的安全侧。
调用方把窗开得更宽时**拒判**(``band_admits_harmonic``),不是照报一个减半的周期。

峰落在窗沿 5% 以内时挂 ``period_at_band_edge`` —— 搜索被窗截断这件事要看得见。

## 周期这个**数**不由 ``_band_peak`` 给 —— 它会锁到 chevron 卫星上

检测(「窗里有没有显著峰」)复用 ``detect_texture`` 一字不改。但**报出来的周期**
另走一条:``_band_peak`` 在「强峰」(prominence ≥ 0.35 × 最大)里取**周期最小**的
那个,而 zigzag 的折臂在 ``|k| = √((k₀cosα)² + (k₀sinα + 1/L)²)`` 处留了一个真实的
卫星峰 —— 它比基频**更靠外**,于是被稳定地选中。

合成实测(折臂模型,α=16°、L=30 nm、T=6.3 nm、50 nm/256 px):

    ``_band_peak`` 报的周期    5.84 .. 6.23 nm   偏低 1.1% .. 7.2%(随取向变)
    卫星的预期位置             |k| = 0.0334 (基频 0.0310)  ⇒ 恰好是那 −7%

所以 ``period_nm`` 改由 :func:`stripe_peak` 给:二维功率谱在窗内的**全局最强峰**
(卫星比基频弱,argmax 不会选它),再对 3×3 邻域做功率加权质心做亚像素细化。
代价是二维格点的径向量化 ——8 个周期的帧上峰半径只有 ~8 px,不细化的话 ΔT/T 有
12.7%;细化后合成实测残差见测试。

**方向的角分辨率是物理上限,不是实现缺陷**:有限帧上条纹峰的角向半宽 ≈
``atan(1/N_periods)``,8 个周期时 7.1°。报出来的 ``k_angle_deg`` 就是这个精度,
而且 herringbone 本来就有**两条臂**(±α ≈ ±16°),argmax 落在其中一条上 ——
所以它是「某一条臂的法向」,不是「平均取向」。

## 慢轴那一条:条纹平行于快扫方向时,判据自己把信号吃掉了

``flatten_robust = level_iterative(align_rows_mediandiff(h))``。
``align_rows_mediandiff`` 减掉逐行中值差的累积和 —— 它**正是**用来杀行偏移条纹的。
后果:**沿快扫方向的条纹(k 沿慢轴)会在判据看到它之前就被减没**。

同一个几何上还叠着第二重退化:每行随机偏移的扫描线噪声(1/f、蠕变、热漂移)产生
的就是 k 沿慢轴的条纹。**这两件事在一帧图里没法分开**。

所以这个几何一律判 ``undetermined``(``stripes_along_fast_axis``),并给出可执行的
下一步:**把扫描框转 ~30° 重扫**。判成 ``absent``(「这里没有 herringbone」)会让
forge 接着去扎一根其实没问题的针。

检测靠一次**不做行对齐**的平场(``level_iterative`` 单独跑)上的同一套判据:
标准平场看不见的东西,只有绕开标准平场才看得见。

### 而同一个操作还会**凭空造出**一个慢轴条纹峰

``align_rows_mediandiff`` 减掉的是逐行中值差的**累积和**。真实行偏移为零时,那串
中值差就是噪声,它的累积和是一条**随机游走** —— 全部落在 ``kx=0`` 上,而且相干。
合成实测(40 个种子的纯高斯白噪声,50 nm/256 px):

    标准平场后          带内 SNR **9.5 .. 27.8**(门是 4)、方向 89..91°、集中度 0..74
    只 ``level_iterative``   **一个峰都没有**(8/8)

**这个尺度上 SNR 门与角向集中度门两道都拦不住纯白噪声**。拦得住的只有「另一条路
对不对得上」。所以两件事用同一个对照跑出来的答案分开:

===========================  ==========================  ==================
慢轴上有峰,而且              另一条路(不做行对齐)          结论
===========================  ==========================  ==================
对得上(raw 也有,且集中)      raw/flat 集中度比 13 .. 5e13  ``undetermined``
                                                          真信号,但退化 → 转 30° 重扫
对不上(raw 根本没有这个峰)   比值 = 0(8/8)                ``absent``
                                                          行对齐自己造的,不是证据
===========================  ==========================  ==================

对照组:真有慢轴分量的**行偏移条纹**比值 26..5e13(判 ``undetermined``,对);
被吃掉的真 herringbone(θ=88)比值 13(判 ``undetermined``,对);
纯白噪声比值 0(判 ``absent``,对 —— 正确答案是「这儿什么都没有」,不是
「转个角度再扫一张」)。

**一个曾经在这儿的设计被否掉并钉成了测试**:早先的版本只看标准平场后的 k 指不
指着慢轴,不看另一条路对不对得上。那一版把 30 个纯噪声帧里的 21 个判成
``undetermined``,给用户的话是「把扫描框转 ~30° 重扫」——
在一张什么都没有的图上。见 ``test_pure_noise_is_absent_not_undetermined``。

## 双针尖:周期结构上给不出无阈值的判定,这是定理不是实现缺陷

双针尖是卷积:``I = I_true ∗ (δ + a·δ_d)``,频域 ``Î(k)·(1 + a·e^{-ik·d})``。
**对纯单 k 的周期信号,这只是把那一个 Bragg 系数乘上一个复数** —— 周期还是那个
周期,谱上不产生任何新峰。换句话说:一个「双针尖看到的正弦条纹」与一个「单针尖
看到的、振幅相位不同的正弦条纹」**逐点相同**,任何只吃周期分量的统计量都分不开
它们。(测试 ``test_double_tip_on_a_pure_sinusoid_is_unidentifiable`` 把这条钉住。)

真正带信息的是**非周期内容**:elbow、缺陷、吸附物、台阶。仓里已有的
``vision.double_tip.detect_double_tip`` 正是这么做的 —— 它先减掉主导周期分量,再在
残差的自相关里找离心复制峰(clean feature-bearing 帧上 AUROC 1.0)。这里**直接调
它,不另写一套**,并且把它能不能说话的那个前提也报出来:

    ``aperiodic_fraction`` = 减掉周期分量后残差的方差占比。

它小的时候,``double_tip_detected=False`` 的含义是**「没有证据」**而不是
「针尖没问题」。这两句话必须分开 —— 本仓为「判不了被当成没问题」栽过不止一次。
所以这里**不给** ``aperiodic_fraction`` 设阈值:报数,让读的人自己看。

``second_harmonic_ratio`` 同理只作诊断:双针尖会改它(d≈T/2 时基频被打掉、条纹
看起来「一分为二」),但**针尖变尖也会改它**(分辨出 soliton 线对本身就是二次谐波),
而且 fcc/hcp 宽度天生不等 ⇒ 剖面本来就不对称。三条退化叠在一起,它不能当判据。

## 三态,不是两态

``verdict`` ∈ ``{"herringbone", "absent", "undetermined"}``。
``undetermined`` = **这一帧答不了这个问题**(尺度门、过渡带、死平帧、慢轴退化、
窗设错);``absent`` = 门都过了、确实没找到条纹。
把前者写成后者会让 forge 接着扰动针尖 —— 而该做的是换个视野/转个角度重扫。

过渡带(``scale == "reduced"``)归 ``undetermined`` 而不是 ``absent``:那一档的
**测量本身**是降级的(起伏被低报 5-17%,或峰的角向宽度超过慢轴缺口),它既撑不住
「有」也撑不住「没有」。要正面结论得显式 ``allow_reduced_scale=True``。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: soliton 线对重复周期的先验(纳米)。**只当搜索窗的中心,不当判定门槛。**
#: 与 ``knowledge.clean_metal.CONSTANTS["Au(111)"]["herringbone_period_nm"]`` 同源;
#: skill 层会经 ``core.sample_facts`` 按当前衬底取,取不到才回落到这个数。
DEFAULT_PERIOD_PRIOR_NM = 6.3

#: zigzag 全周期的先验(纳米)。**这里不测它** —— 只用来算「要多大的帧才看得见
#: 它」,好在警告里说清判据没查什么。见模块注释。
DEFAULT_CHEVRON_PRIOR_NM = 30.0

#: 搜索窗 = ``(PERIOD_BAND_LO, PERIOD_BAND_HI) × 先验``。
#: **上沿必须 ≤ 2 × 下沿**,否则窗里同时装得下 T 和 T/2,``_band_peak`` 的
#: 「取最小周期」会稳定地报出半个周期。见模块注释。
PERIOD_BAND_LO = 0.7
PERIOD_BAND_HI = 1.4

#: 尺度门(见模块注释的推导)。``*_OFF`` 是复用的机器自己的硬底,
#: ``*_FULL`` 是「搜索窗完整存在 + 起伏偏差 ≤5%」的那两个数。
PX_PER_PERIOD_OFF = 3.0
PX_PER_PERIOD_FULL = 6.0
PERIODS_IN_FRAME_FULL = 6.0

#: ``atomic_phase._ring_mask`` / ``angular_concentration`` 的默认环宽与 bin 数。
#: 抄在这里只为把下面那个下限**算**出来 —— 抄错了
#: ``test_periods_floor_is_where_angular_concentration_stops_working`` 会红。
_RING_REL = 0.20
_CONC_BINS = 72

#: 帧里最少要有几个条纹周期。
#:
#: **这个数不是拍的,是从复用的那个判据自己的守卫里解出来的。**
#: ``angular_concentration`` 把功率谱在 ``|k| ≈ 1/T`` 的环上按角度分 bin,而它开头
#: 有一句 ``if ring_px < n_bins: return 0.0``。环的半径(像素)恰好就是
#: ``N / T_px = frame_nm / T_nm = periods_in_frame`` —— 帧里有几个周期,环就有几个
#: 像素半径。环的面积是 ``π f0² ((1+rel)² − (1−rel)²) = 0.8π f0²``,于是
#:
#:     0.8π · periods_in_frame² ≥ 72  ⟺  periods_in_frame ≥ 5.35
#:
#: 低于它,**核心判据整条静默返回 0**,于是每一帧都落进 ``not_a_stripe_pattern``
#: —— 也就是不论图上有什么都报「没有条纹」。这正是本仓反复栽的那种「守卫把一切
#: 都拒了,而判据只朝一侧失败,所以看不出来」。
#:
#: 早先这里写的是 4.0(``detect_texture`` 把周期带上沿夹在 ``min(H,W)/4`` 的那个
#: 硬底)。那个数**不够**:``_band_peak`` 单独什么都判不了(实测它在纯白噪声上
#: 给出 SNR 9.5..27.8),真正在判的是角向集中度,所以门必须定在**集中度**失效的
#: 地方,不是定在找峰失效的地方。
PERIODS_IN_FRAME_OFF = math.sqrt(
    _CONC_BINS / (math.pi * ((1.0 + _RING_REL) ** 2 - (1.0 - _RING_REL) ** 2)))

#: 慢轴缺口(度)。k 落在慢轴 ±这个角度内 ⇒ 与扫描线伪影不可分 ⇒ 拒判。
#: 由角向 bin 宽(360/72 = 5°)与 6 个周期时峰的角向半宽 ``atan(1/6) = 9.5°``
#: 共同定出来。
SLOW_AXIS_NOTCH_DEG = 10.0

# 四象限起伏散布上限，用于提示局部台阶或大特征污染条纹幅值。
# 这是诊断警告阈值，不参与 verdict；应用到新成像条件前须验证。
CORRUGATION_SPREAD_MAX = 2.0

#: 角向集中度阈值:**离散条纹峰 vs 弥散环**。这条直接沿用
#: ``atomic_phase.DEFAULT_CONCENTRATION_MIN``(合成实测:真晶格 97..7645、带通抖动
#: 1.8..3.3),并在本模块自己的合成对照组上重新量过分隔度 —— 数字写在
#: ``tests/v2/unit/vision/test_herringbone.py`` 的那条分隔度测试里,它红了就说明
#: 判据退回到「FFT 里有峰就算数」。
#:
#: 单轴条纹只有 ±k 两瓣(六角晶格有 6 瓣),中位 bin 更低 ⇒ 比值只会更大,所以
#: 沿用晶格那侧的阈值是**保守**方向。


@dataclass(frozen=True)
class HerringboneResult:
    """一帧扫描图上 Au(111) herringbone 的判定 + 针尖质量读数。

    ``verdict`` 是产品:``"herringbone"`` / ``"absent"`` / ``"undetermined"``。
    ``passed`` 只是 ``verdict == "herringbone"`` 的别名,给 composite 步骤用。

    **拒判时与帧内容无关的量照报**(尺度、几何、起伏 RMS)——「测出来是零」和
    「没测」必须是两句话。
    """

    verdict: str                                   # herringbone / absent / undetermined
    passed: bool = False

    # ── 尺度与几何 ──
    scale: str | None = None                       # full / reduced / off / None=不知道
    nm_per_px: float | None = None
    px_per_period: float | None = None
    periods_in_frame: float | None = None
    frame_nm: float | None = None

    # ── 条纹本身 ──
    period_nm: float | None = None                 # 实测 soliton 线对周期
    period_prior_nm: float | None = None           # 搜索窗中心(先验,**未标定**)
    period_band_nm: tuple[float, float] | None = None
    snr: float = 0.0
    angular_concentration: float = 0.0             # 离散条纹 vs 弥散环(核心判据)
    k_angle_deg: float | None = None               # 调制方向(条纹法向),0°=快扫轴
    stripe_angle_deg: float | None = None          # 条纹走向 = k_angle + 90°
    fft_sharpness: float = 0.0

    # ── 针尖质量读数(报数,不判) ──
    stripe_corrugation_pm: float | None = None     # 带通分量的**峰峰值**
    #: 四象限起伏的 ``最大/最小``。**读 ``stripe_corrugation_pm`` 之前先看它** ——
    #: 超过 ``CORRUGATION_SPREAD_MAX`` 说明那个数里混了台阶之类的局部大特征。
    corrugation_quadrant_spread: float | None = None
    corrugation_rms_m: float = 0.0                 # 整帧去趋势后的起伏 RMS
    second_harmonic_ratio: float = 0.0             # 诊断:退化量,见模块注释

    # ── 双针尖(条件成立才有意义,见模块注释) ──
    double_tip_detected: bool | None = None
    double_tip_score: float | None = None
    double_tip_threshold: float | None = None
    double_tip_separation_nm: float | None = None
    aperiodic_fraction: float | None = None        # 双针尖那个数能不能说话的前提

    # 未经行对齐的窗内峰 SNR 不携带方向信息，既可能来自划痕也可能来自目标纹理。
    # 需要判断慢轴结构时，应结合后续具有方向约束的功率统计。
    row_free_band_snr: float = 0.0
    #: 慢轴那一侧的功率占比 ∈ [0, 1]:不做行对齐的帧上,环内「离慢轴 ≤ 缺口」的
    #: 角向 bin 最大值 ÷ 全环最大值。**这个才带方向。**
    #: 1.0 = 最强的条纹就在慢轴上(与扫描线伪影不可分,判据会拒判);
    #: 小 = 慢轴上没有能跟主条纹相比的东西。
    slow_axis_power_ratio: float | None = None

    # ── zigzag:**没查**,只说要多大的帧才查得动 ──
    chevron_prior_nm: float | None = None
    chevron_min_frame_nm: float | None = None

    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: dict = field(default_factory=dict)


# ── 尺度门 ───────────────────────────────────────────────────────────────────

def scale_gate(nm_per_px: float | None, shape, period_prior_nm: float
               ) -> tuple[str | None, float | None, float | None]:
    """``(档位, px_per_period, periods_in_frame)``。``档位 is None`` = 不知道像素尺度。

    两个条件都要过才是 ``full``;任一掉到硬底就是 ``off``。推导见模块注释。
    """
    if (nm_per_px is None or not np.isfinite(nm_per_px) or nm_per_px <= 0
            or not period_prior_nm or period_prior_nm <= 0):
        return None, None, None
    try:
        n_min = int(min(shape[0], shape[1]))
    except (TypeError, IndexError):
        return None, None, None
    if n_min <= 0:
        return None, None, None
    px_per_period = float(period_prior_nm) / float(nm_per_px)
    periods_in_frame = n_min / px_per_period
    if px_per_period < PX_PER_PERIOD_OFF or periods_in_frame < PERIODS_IN_FRAME_OFF:
        return "off", px_per_period, periods_in_frame
    if (px_per_period >= PX_PER_PERIOD_FULL
            and periods_in_frame >= PERIODS_IN_FRAME_FULL):
        return "full", px_per_period, periods_in_frame
    return "reduced", px_per_period, periods_in_frame


# ── 方向:角向 bin 的峰在哪 ─────────────────────────────────────────────────

def stripe_peak(image: npt.ArrayLike, band_px: tuple[float, float]
                ) -> tuple[float, float] | None:
    """窗内最强的那个谱峰 → ``(周期 px, 调制方向 deg)``，找不到返回 ``None``。

    **周期这个数的唯一来源**（``_band_peak`` 会锁到 chevron 卫星上，见模块注释）。
    取二维功率谱在径向窗内的全局 argmax，再对 3×3 邻域做功率加权质心做亚像素细化。

    角度从**快扫轴(+列)**量到**行号增大**的方向，mod 180：
    0°/180° = k 沿快扫轴（条纹垂直于快扫方向，安全）；
    90° = k 沿慢轴（条纹平行于快扫方向，与扫描线伪影不可分）。
    """
    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 16:
        return None
    lo, hi = float(band_px[0]), float(band_px[1])
    if not (lo > 0 and hi > lo):
        return None
    H, W = h.shape
    x = h - h.mean()
    win = np.outer(np.hanning(H), np.hanning(W))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
    cy, cx = H // 2, W // 2
    # 频率按各自的轴长归一 —— 非方帧上 dy/H 与 dx/W 的标度不同。
    fy = (np.arange(H) - cy)[:, None] / float(H)
    fx = (np.arange(W) - cx)[None, :] / float(W)
    fr = np.hypot(fy, fx)
    sel = (fr >= 1.0 / hi) & (fr <= 1.0 / lo)
    if not sel.any():
        return None
    masked = np.where(sel, P, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(masked)), P.shape)
    # 3×3 功率加权质心 —— 不细化的话径向量化误差是 1/r（8 个周期时 12.7%）。
    y0, y1 = max(0, iy - 1), min(H, iy + 2)
    x0, x1 = max(0, ix - 1), min(W, ix + 2)
    w = P[y0:y1, x0:x1]
    tot = float(w.sum())
    if tot > 0:
        yy = np.arange(y0, y1, dtype=np.float64)[:, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, :]
        fy_r = (float((w * yy).sum() / tot) - cy) / float(H)
        fx_r = (float((w * xx).sum() / tot) - cx) / float(W)
    else:  # pragma: no cover — argmax 自己就在窗里，权重和不会是 0
        fy_r, fx_r = float(fy[iy, 0]), float(fx[0, ix])
    f = math.hypot(fy_r, fx_r)
    if f <= 0:  # pragma: no cover — 窗把 DC 排除在外
        return None
    return 1.0 / f, float(math.degrees(math.atan2(fy_r, fx_r)) % 180.0)


def slow_axis_offset_deg(k_angle_deg: float | None) -> float | None:
    """k 离**慢轴**(90°)还有多少度。缺口判定用它,不用绝对角。"""
    if k_angle_deg is None:
        return None
    return float(abs((float(k_angle_deg) % 180.0) - 90.0))


def slow_axis_power_ratio(image: npt.ArrayLike, period_px: float, *,
                          notch_deg: float = SLOW_AXIS_NOTCH_DEG,
                          n_bins: int = 72) -> float | None:
    """慢轴方向功率占比 ∈ [0, 1]：慢轴附近角向 bin 的最大值 / 全环最大值。

    与不带方向的 row_free_band_snr 不同，它描述慢轴条纹相对主条纹的强度。
    须在未做行对齐的图上计算，以免 align_rows_mediandiff 先移除待测分量。

    它只探测慢轴方向这一失效模式，不是通用质量分；其他方向的划痕不在其覆盖范围。
    """
    from mast.vision.atomic_phase import _angular_bins, _ring_mask

    h = np.asarray(image, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 16 or not (period_px and period_px >= 3.0):
        return None
    x = h - h.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
    f0 = float(P.shape[0]) / float(period_px)
    sel, ang = _ring_mask(P.shape, f0)
    if int(sel.sum()) < n_bins:
        return None
    bins = _angular_bins(P[sel], ang[sel], n_bins)
    peak = float(bins.max())
    if peak <= 0.0:
        return None
    centres = -180.0 + (np.arange(n_bins) + 0.5) * 360.0 / n_bins
    near = np.abs((np.abs(centres) % 180.0) - 90.0) <= float(notch_deg)
    if not near.any():  # pragma: no cover — 缺口至少覆盖一个 bin
        return None
    return float(bins[near].max() / peak)


# ── 带通:条纹分量的振幅 ────────────────────────────────────────────────────

def bandpass(image: npt.ArrayLike, period_px: float) -> npt.NDArray[np.float64]:
    """只留 ``f0 ± 35%`` 的那一层。

    ⚠️ 掩膜与 ``seg_scale_adaptive.bandpass_envelope`` **逐字相同**,是刻意保持
    同步的一份拷贝:那个文件的头注明写着「code kept line-identical apart from
    packaging so the validated numbers keep applying」,所以不在那里加辅助函数。
    同步性由 ``test_bandpass_matches_seg_scale_adaptive`` 钉着 —— 那条测试把本函数
    的输出补上同样的平滑,要求与 ``bandpass_envelope`` 逐点相等。
    """
    h = np.asarray(image, dtype=np.float64)
    H, W = h.shape
    F = np.fft.fft2(h - h.mean())
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    fr = np.hypot(fy, fx)
    f0 = 1.0 / float(period_px)
    keep = (fr > 0.65 * f0) & (fr < 1.35 * f0)
    return np.real(np.fft.ifft2(F * keep))


def stripe_corrugation_pm(image_m: npt.ArrayLike, period_px: float) -> float | None:
    """条纹分量的**峰峰值起伏**,皮米。输入必须是米。

    正弦的峰峰值 = ``2√2 × rms``。选峰峰值而不是振幅,是因为文献与知识库里
    「herringbone 起伏 ~20 pm」这句话说的就是峰峰值。
    """
    h = np.asarray(image_m, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 16 or not (period_px and period_px >= 3.0):
        return None
    bp = bandpass(h, period_px)
    rms = float(np.sqrt(np.mean(bp * bp)))
    if not np.isfinite(rms):
        return None
    return float(2.0 * math.sqrt(2.0) * rms * 1e12)


def corrugation_quadrant_spread(image_m: npt.ArrayLike, period_px: float
                                ) -> float | None:
    """四象限分别估计起伏，再取最大/最小，诊断整体条纹幅值是否均匀。

    台阶或局部大特征的宽带功率可能进入条纹带通，抬高其所在象限的幅值。
    象限散布随结果字段报告；CORRUGATION_SPREAD_MAX 仅控制警告，不参与 verdict。
    阈值应用到新成像条件前须验证。
    """
    h = np.asarray(image_m, dtype=np.float64)
    if h.ndim != 2 or min(h.shape) < 32 or not (period_px and period_px >= 3.0):
        return None
    from mast.vision.seg_scale_adaptive import flatten_robust

    ny, nx = h.shape[0] // 2, h.shape[1] // 2
    vals = []
    for ys, xs in ((slice(0, ny), slice(0, nx)), (slice(0, ny), slice(nx, None)),
                   (slice(ny, None), slice(0, nx)),
                   (slice(ny, None), slice(nx, None))):
        sub = h[ys, xs]
        if min(sub.shape) < 16:
            return None
        v = stripe_corrugation_pm(flatten_robust(sub), period_px)
        if v is None or not np.isfinite(v) or v <= 0:
            return None
        vals.append(float(v))
    lo = min(vals)
    return float(max(vals) / lo) if lo > 0 else None


def second_harmonic_ratio(image: npt.ArrayLike, period_px: float,
                          k_angle_deg: float | None) -> float:
    """``P(2k₀) / P(k₀)``,沿实测的 k 方向取。**诊断量,不作判据**(见模块注释)。"""
    h = np.asarray(image, dtype=np.float64)
    if (h.ndim != 2 or min(h.shape) < 16 or k_angle_deg is None
            or not (period_px and period_px >= 6.0)):
        return 0.0
    x = h - h.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
    cy, cx = P.shape[0] // 2, P.shape[1] // 2
    f0 = float(P.shape[0]) / float(period_px)
    th = math.radians(float(k_angle_deg))

    def _local_max(radius: float) -> float:
        dy, dx = radius * math.sin(th), radius * math.cos(th)
        best = 0.0
        for sy in (1.0, -1.0):                 # ±k 两瓣都看,取大的
            yy = int(round(cy + sy * dy))
            xx = int(round(cx + sy * dx))
            y0, y1 = max(0, yy - 2), min(P.shape[0], yy + 3)
            x0, x1 = max(0, xx - 2), min(P.shape[1], xx + 3)
            if y1 > y0 and x1 > x0:
                best = max(best, float(P[y0:y1, x0:x1].max()))
        return best

    p1 = _local_max(f0)
    if p1 <= 0.0 or 2.0 * f0 >= min(cy, cx):   # 二次谐波超出谱范围 → 说不了
        return 0.0
    return float(_local_max(2.0 * f0) / p1)


# ── 主判据 ───────────────────────────────────────────────────────────────────

def assess_herringbone(
    image: npt.ArrayLike,
    *,
    nm_per_px: float | None,
    period_prior_nm: float = DEFAULT_PERIOD_PRIOR_NM,
    chevron_prior_nm: float = DEFAULT_CHEVRON_PRIOR_NM,
    period_band_nm: tuple[float, float] | None = None,
    snr_min: float = 4.0,
    concentration_min: float | None = None,
    slow_axis_notch_deg: float = SLOW_AXIS_NOTCH_DEG,
    allow_reduced_scale: bool = False,
    z_unit_is_m: bool = True,
) -> HerringboneResult:
    """这一帧上有没有 herringbone,以及针尖把它画得多好。

    纯函数:输入一个二维高度数组(米)与显式阈值,输出 frozen 结果。不读配置、
    不碰硬件、不抛异常。

    判定(三条取与,外加两道门):

    0. **帧能不能判** —— ``vision.frame_validity.judge_frame``,与
       ``AssessTipSharpness`` / ``AssessClusterRoundness`` / ``FindFlatRegion`` /
       ``CheckLineQuality`` 共用的**同一份**判据,不复制第二套。
    1. **尺度门** —— ``px_per_period`` 与 ``periods_in_frame`` 都要够(见模块注释);
    2. 搜索窗里有显著谱峰(``snr_min``);
    3. **角向集中度** ≥ ``concentration_min`` —— 离散条纹峰而不是弥散环;
    4. k **不在慢轴缺口里** —— 否则与扫描线伪影不可分,拒判(不是「没有」)。

    ``concentration_min`` 留 ``None`` 时取 ``atomic_phase.DEFAULT_CONCENTRATION_MIN``
    (单一真源;单轴条纹只有两瓣,沿用晶格那侧的阈值是保守方向)。

    ``z_unit_is_m=False`` 时 ``stripe_corrugation_pm`` 不给 —— 起伏是有量纲的,
    z 不是米就报不出皮米,**报一个未知单位的数比不报更坏**。
    """
    reasons: list[str] = []
    warns: list[str] = []
    notes: dict = {}

    if concentration_min is None:
        from mast.vision.atomic_phase import DEFAULT_CONCENTRATION_MIN
        concentration_min = float(DEFAULT_CONCENTRATION_MIN)

    prior = float(period_prior_nm) if period_prior_nm else 0.0
    chev = float(chevron_prior_nm) if chevron_prior_nm else 0.0
    chev_min_frame = (chev * PERIODS_IN_FRAME_FULL) if chev > 0 else None

    h_raw = np.asarray(image, dtype=np.float64)
    if h_raw.ndim == 3 and h_raw.shape[0] in (1, 2):
        h_raw = h_raw[0]

    scale, px_per_period, periods = scale_gate(nm_per_px, h_raw.shape, prior)
    nmpp = float(nm_per_px) if scale is not None else None
    frame_nm = (float(min(h_raw.shape)) * nmpp
                if (nmpp and h_raw.ndim == 2) else None)

    def _out(verdict: str, *why: str, **kw) -> HerringboneResult:
        return HerringboneResult(
            verdict=verdict, passed=(verdict == "herringbone"),
            scale=scale, nm_per_px=nmpp, px_per_period=px_per_period,
            periods_in_frame=periods, frame_nm=frame_nm,
            period_prior_nm=(prior or None),
            chevron_prior_nm=(chev or None), chevron_min_frame_nm=chev_min_frame,
            reasons=tuple(why), warnings=tuple(warns), notes=notes, **kw)

    if prior <= 0:
        return _out("undetermined", "no_period_prior")

    # ── 搜索窗:上沿 ≤ 2 × 下沿,否则窗里装得下 T/2(见模块注释) ──
    if period_band_nm is None:
        band = (PERIOD_BAND_LO * prior, PERIOD_BAND_HI * prior)
    else:
        band = (float(period_band_nm[0]), float(period_band_nm[1]))
    if not (band[0] > 0 and band[1] > band[0]):
        return _out("undetermined", "bad_period_band")
    if band[1] > 2.0 * band[0] * (1.0 + 1e-9):
        # 照报会稳定地给出半个周期 —— 那比拒判坏得多。
        notes["band_admits_harmonic"] = (
            f"窗 ({band[0]:.2f}, {band[1]:.2f}) nm 的上沿超过下沿的 2 倍，"
            f"窗里同时装得下 T 与 T/2，_band_peak 的「取最小周期」会报出半个周期。")
        return _out("undetermined", "band_admits_harmonic")

    if h_raw.ndim != 2 or min(h_raw.shape) < 16:
        return _out("undetermined", "insufficient_data")
    if not np.isfinite(h_raw).all():
        med = np.nanmedian(h_raw) if np.isfinite(h_raw).any() else 0.0
        h_raw = np.nan_to_num(h_raw, nan=float(med) if np.isfinite(med) else 0.0)

    try:
        from mast.vision.atomic_phase import angular_concentration
        from mast.vision.double_tip import _residual, detect_double_tip
        from mast.vision.frame_validity import judge_frame
        from mast.vision.seg_scale_adaptive import (
            DEFAULTS,
            detect_texture,
            flatten_robust,
            level_iterative,
        )
        from mast.vision.tip_metrics import _detrend, _fft_sharpness
    except Exception as exc:  # noqa: BLE001 — 依赖缺席就是「判不了」
        logger.debug("herringbone 判据依赖缺席: %s", exc)
        return _out("undetermined", "dependency_unavailable")

    # ── 0. 这一帧能不能判(与另外四个判据同一份判据) ──
    frame = judge_frame(h_raw)
    if not frame.usable:
        notes["frame_unusable"] = frame.reason
        return _out("undetermined", "frame_unusable",
                    corrugation_rms_m=float(frame.corrugation_rms_m))
    rms_m = float(frame.corrugation_rms_m)

    if scale is None:
        return _out("undetermined", "unknown_pixel_size", corrugation_rms_m=rms_m)
    if scale == "off":
        return _out("undetermined", "scale_gate", corrugation_rms_m=rms_m)
    if scale == "reduced":
        warns.append("scale_reduced")
    if chev_min_frame and frame_nm and frame_nm < chev_min_frame:
        # **说明判据没查什么**,不是判据失败。见模块注释「判的是什么、不判什么」。
        warns.append("chevron_not_resolvable")

    flat = flatten_robust(h_raw)

    params = dict(DEFAULTS)
    params["atomic_band_nm"] = band
    params["lat_snr"] = float(snr_min)

    # 机器内部的两道夹子有没有咬到我们的窗 —— 满权重档下按推导不该咬到。
    lo_px = max(2.5, band[0] / nmpp)
    hi_px = min(min(flat.shape) / 4.0, band[1] / nmpp)
    if hi_px < band[1] / nmpp * (1.0 - 1e-9) or lo_px > band[0] / nmpp * (1.0 + 1e-9):
        warns.append("band_clipped_by_frame")

    band_px = (band[0] / nmpp, band[1] / nmpp)

    # 检测门:``detect_texture`` 一字不改地复用。它给的是「窗里有没有显著峰」
    # 与那个峰的 SNR —— **不用它给的周期**(会锁到 chevron 卫星上,见模块注释)。
    tex = detect_texture(flat, float(nmpp), params)
    peak = tex.get("atomic")
    snr = float(peak[1]) if peak else 0.0

    # 周期与方向:窗内最强峰 + 亚像素细化。
    refined = stripe_peak(flat, band_px) if peak else None
    period_nm: float | None = None
    t_px = 0.0
    k_deg: float | None = None
    if refined:
        t_px, k_deg = refined
        period_nm = t_px * float(nmpp)
        if (period_nm <= band[0] * 1.05) or (period_nm >= band[1] * 0.95):
            warns.append("period_at_band_edge")

    conc = angular_concentration(flat, t_px) if t_px >= 3.0 else 0.0
    stripe_deg = ((k_deg + 90.0) % 180.0) if k_deg is not None else None
    off_deg = slow_axis_offset_deg(k_deg)

    # ── 慢轴那一侧的两件事,判据必须分开 ──
    # 只有绕开 align_rows_mediandiff 才看得见它们,所以这一段**必须在报 absent
    # 之前**跑 —— 它要挡的正是「标准平场把信号减没了,于是判成没有」。
    slow_snr = 0.0
    slow_ratio: float | None = None
    raw_lvl = level_iterative(h_raw)
    tex_raw = detect_texture(raw_lvl, float(nmpp), params)
    peak_raw = tex_raw.get("atomic")
    refined_raw = stripe_peak(raw_lvl, band_px) if peak_raw else None
    raw_corroborates = False
    if refined_raw:
        slow_snr = float(peak_raw[1])
        t_raw, ang_raw = refined_raw
        off_raw = slow_axis_offset_deg(ang_raw)
        slow_conc = angular_concentration(raw_lvl, t_raw) if t_raw >= 3.0 else 0.0
        slow_ratio = slow_axis_power_ratio(raw_lvl, t_raw,
                                           notch_deg=float(slow_axis_notch_deg))
        raw_corroborates = (off_raw <= float(slow_axis_notch_deg)
                            and slow_conc >= float(concentration_min))
        if raw_corroborates:
            # (1) 真的有慢轴条纹 —— 可能是 herringbone，也可能是扫描线伪影，
            #     一帧图里分不开。而且标准平场本来就会把它减掉。
            notes["slow_axis"] = (
                f"不做行对齐时，窗里出现 k 沿慢轴的条纹（周期 "
                f"{t_raw * nmpp:.2f} nm，方向 {ang_raw:.0f}°，SNR {slow_snr:.1f}，"
                f"集中度 {slow_conc:.0f}）。这个几何下 herringbone 与扫描线伪影"
                f"不可分，而且标准平场（align_rows_mediandiff）本来就会把它减掉。"
                f"把扫描框转 ~30° 重扫一张再判。")
            return _out("undetermined", "stripes_along_fast_axis",
                        corrugation_rms_m=rms_m, snr=snr,
                        period_nm=period_nm, period_band_nm=band,
                        row_free_band_snr=slow_snr,
                        slow_axis_power_ratio=slow_ratio)

    if not peak:
        reasons.append("no_stripe_peak")
    elif off_deg is not None and off_deg <= float(slow_axis_notch_deg):
        # (2) 标准平场里指着慢轴、而**不做行对齐时根本没有这个峰** ——
        #     这个峰是 ``align_rows_mediandiff`` 自己造的,不是证据。
        #
        # 它减掉的是逐行中值差的**累积和**;真实行偏移为零时,那串中值差是噪声,
        # 累积和就是一条**随机游走** —— 全部落在 kx=0 上,而且相干。合成实测
        # (40 个种子的纯高斯白噪声,50 nm/256 px):
        #
        #     标准平场后   带内 SNR 9.5..27.8(门是 4)、方向 89..91°、集中度 0..74
        #     只 level_iterative  **一个峰都没有**(8/8)
        #
        # 也就是说这个尺度上 SNR 门与集中度门**两道都拦不住它**,拦得住的只有
        # 「另一条路对不对得上」。对照组(真的有慢轴分量的行偏移条纹)两条路都
        # 看得见,raw/flat 集中度比 26..5e13;被吃掉的真 herringbone(θ=88)比值 13。
        #
        # 这里刻意**不**用 ``k_along_slow_axis`` 拒判:那样纯噪声会得到
        # 「转个角度再扫一张」,而正确答案是「这儿什么都没有」。
        notes["row_alignment_artifact"] = (
            f"窗里那个峰指着慢轴（{k_deg:.0f}°），但不做行对齐时它不存在 —— "
            f"它是 align_rows_mediandiff 的累积中值差（随机游走）造出来的，"
            f"不是表面结构。")
        reasons.append("row_alignment_artifact")
    if conc < float(concentration_min):
        reasons.append("not_a_stripe_pattern")

    sharp, _res_nm, _has_lat = _fft_sharpness(_detrend(h_raw) / (rms_m or 1.0),
                                              float(nmpp))

    corr_pm = (stripe_corrugation_pm(flat, t_px)
               if (t_px >= 3.0 and z_unit_is_m) else None)
    # 起伏读数需同时报告空间离散度；局部台阶等大特征可能抬高整帧统计。
    spread = (corrugation_quadrant_spread(h_raw, t_px)
              if (corr_pm is not None) else None)
    if spread is not None and spread > CORRUGATION_SPREAD_MAX:
        warns.append("corrugation_inhomogeneous")
    if corr_pm is not None and scale == "reduced":
        # 像素积分对正弦的衰减 sinc(π/T_px):这一档最多低报 17%(见模块注释)。
        warns.append("corrugation_underreported_at_reduced_scale")
    h2 = second_harmonic_ratio(flat, t_px, k_deg) if t_px >= 3.0 else 0.0
    if h2 > 1.0:
        # h2>1 表示二次谐波功率超过窗内基频。
        # band_hi<=2*band_lo 时，该谐波周期位于搜索窗外；必须提示窗口可能漏掉
        # 主导周期，不能把窗内报告的次要分量冒充整帧主成分。
        warns.append("dominant_period_outside_band")

    # ── 双针尖:直接调仓里那个,并把「它能不能说话」的前提一起报出来 ──
    #
    # ⚠️ **必须先归一化**。``detect_double_tip`` 开头有一道 ``std < 1e-9`` 的早退,
    # 单位是米 —— 而 herringbone 的起伏是 20 pm,整帧去趋势后 std 就在 1e-11 左右,
    # 正好落在那道守卫下面。实测:直接喂米量级数组,单针尖与双针尖**同样**返回
    # ``score=0.0 / is_double=False``,判据看起来「在跑」其实一次都没跑。
    # 这与 ``tip_sharpness.py`` 注释里 ``assess_tip_classical`` 那道 1e-9 守卫
    # 是同一类,同一个原因(Au(111) 单原子台阶才 236 pm)。不去改那个函数 ——
    # 它同时喂着 VIGIL 判据链;这里把量纲交代清楚再调。
    dt_detected = dt_score = dt_sep = dt_thr = None
    aper = None
    try:
        f0 = flat - float(flat.mean())
        fstd = float(f0.std())
        if fstd > 0:
            fn = (f0 / fstd).astype(np.float32)        # 无量纲,std == 1
            aper = float(np.var(_residual(fn)))        # var(fn) == 1 ⇒ 直接是占比
            dt = detect_double_tip(fn, nm_per_px=float(nmpp))
            dt_detected = bool(dt.is_double)
            dt_score = float(dt.score)
            dt_thr = float(dt.threshold)
            dt_sep = (float(dt.separation_nm)
                      if dt.separation_nm is not None else None)
    except Exception as exc:  # noqa: BLE001 — 附加读数坏了不该让判据失败
        logger.debug("双针尖读数算不出来: %s", exc)
        notes["double_tip_error"] = f"{type(exc).__name__}: {exc}"

    common = dict(
        corrugation_rms_m=rms_m, snr=snr, period_nm=period_nm,
        period_band_nm=band, angular_concentration=conc,
        k_angle_deg=k_deg, stripe_angle_deg=stripe_deg,
        fft_sharpness=float(sharp), stripe_corrugation_pm=corr_pm,
        corrugation_quadrant_spread=spread,
        second_harmonic_ratio=float(h2),
        double_tip_detected=dt_detected, double_tip_score=dt_score,
        double_tip_threshold=dt_thr, double_tip_separation_nm=dt_sep,
        aperiodic_fraction=aper, row_free_band_snr=slow_snr,
        slow_axis_power_ratio=slow_ratio,
    )
    if scale == "reduced" and not allow_reduced_scale:
        # 过渡带落 ``undetermined`` 而**不是** ``absent``:这一档的测量本身是降级
        # 的(每周期 3-6 px 时像素积分把起伏低报 5-17%,或者帧里只有 4-6 个周期
        # 时峰的角向宽度超过慢轴缺口),所以它既撑不住「有」,也撑不住「没有」。
        # 判成 absent 会让 forge 据此接着扰动针尖 —— 而该做的是换个视野重扫。
        return _out("undetermined", "scale_reduced", *reasons, **common)
    if reasons:
        return _out("absent", *reasons, **common)
    return _out("herringbone", **common)


__all__ = [
    "DEFAULT_CHEVRON_PRIOR_NM",
    "DEFAULT_PERIOD_PRIOR_NM",
    "PERIODS_IN_FRAME_FULL",
    "PERIODS_IN_FRAME_OFF",
    "PERIOD_BAND_HI",
    "PERIOD_BAND_LO",
    "PX_PER_PERIOD_FULL",
    "PX_PER_PERIOD_OFF",
    "SLOW_AXIS_NOTCH_DEG",
    "HerringboneResult",
    "assess_herringbone",
    "bandpass",
    "scale_gate",
    "second_harmonic_ratio",
    "slow_axis_offset_deg",
    "slow_axis_power_ratio",
    "corrugation_quadrant_spread",
    "stripe_corrugation_pm",
    "stripe_peak",
]
