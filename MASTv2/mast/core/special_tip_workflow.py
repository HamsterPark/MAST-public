"""特异化针尖的流程表 —— 从「好针尖基础态」再往前走的两条路。

``noble_tip_workflow`` 把针尖修到基础态(正反扫描线重合、簇单峰且圆)。之后要什么
样的针尖取决于要做什么实验,要求的头两种是:

* **做 STS 的金属性针尖**(:data:`SPECTROSCOPY_TIP`)—— 浅扎出小而圆的簇,打开
  lock-in 测一条谱,看贵金属 (111) 面的肖克利表面态在不在该在的位置。
  判据:金属性针尖不一定是最尖的针尖,往往不是最尖的针尖。
* **原子分辨针尖**(:data:`ATOMIC_TIP`)—— 在无台阶的小平区快扫,同时在 ±20 mV
  内随机跳偏压扰动针尖,停下来看图里有没有原子相。这一套参数是现场经验配方,
  没有独立的理论依据。

## 分层(照 noble_tip_workflow 的分工，一个字都不改)

  * :mod:`mast.core.tip_conditioning_policy` 答「**这根针**能承受多大脉冲/多深
    下压」—— 按材料 × 制法 × 形态查表,带安全包络,超上限**拒绝不夹紧**;
  * 本模块答「**这套流程**每一步的判据是什么」—— 成功阈值、预算、扫描尺寸、步进;
  * **包络永远赢**。流程遇到拒绝要如实报告并停下,不许自己降参数重试。

## 精修那一段不重写

两个配方都要「先把针尖扎好」,而那一整套(D1 深扎修形状 → D2 从 100 pm 起一级级
加到 Z 恰好跳变 → 在临界深度反复扎)已经是 ``_tip_phases.poke_phase`` 的实现。
:meth:`poke_workflow` 把本表里的精修档翻译成一个 ``NobleTipWorkflow`` 交给它 ——
判据只有一份,阈值不会两边漂。

## 谱窗口跟着衬底走,不写死

STS 的扫描窗口按**相对 onset** 定义(``sweep_below_onset_v`` / ``sweep_above_v``)
而不是写死 −0.8..0.3 V。Au(111) 的 onset 在 −0.49 V,那对窗口是合理的;换到
Ag(111)(−0.065 V)同样一组数字会把 90% 的采样点花在一段没有信息的能量上,还把
onset 挤到窗口边缘 —— 而判据要求 onset 两侧都有数据。窗口跟着衬底走,这两件事
自动都对。

## 帧参数不是随便挑的

原子相判据有一道**尺度门**:nm/px ≥ 0.05 时晶格物理上不可分辨,判据会拒判
(``vision.atomic_phase`` 的模块注释)。出厂的 5 nm / 256 px = 0.0195 nm/px 刚好
落在满权重档。:meth:`eval_pixel_size_nm` 与 :meth:`scale_problem` 让流程在下发扫
描之前就能发现「这组帧参数根本判不出原子相」,而不是扫完一张图才知道。

数值一律标**待真机标定**:出厂值是起点不是真值,随仪器、样品、温度变。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any

from mast.core.noble_tip_workflow import NobleTipWorkflow
from mast.core.noble_tip_workflow import resolve as _noble_resolve

logger = logging.getLogger(__name__)


# ── 配方 1:做 STS 的金属性针尖 ──────────────────────────────────────────────

@dataclass(frozen=True)
class SpectroscopyTipWorkflow:
    """浅扎 → 测谱 → 看肖克利表面态,不达标就再扎再测。

    单位写在字段名里(``_v`` / ``_nm`` / ``_a``)—— 这是本仓流程表的纪律,
    「0.5」在米和纳米上差九个数量级。
    """

    # ── 精修档(交给 poke_phase) ──
    #: 首扎深度。比修针基础流程浅 —— 金属性针尖要的是一个小而圆的簇,不是把针尖
    #: 重塑一遍。要求:用小深度扎针尖。待真机标定。
    #:
    #: 2026-08-10:基础流程从 1.5 nm 降到 **500 pm**(用户范本,见
    #: ``noble_tip_workflow.poke_depth_nm``),这里跟着降到 **300 pm** 以保住那条
    #: 「比基础流程浅」的关系 —— 0.6 nm 现在**比基础流程还深**,关系会反过来。
    #: ⚠️ 300 pm 是**从基础流程推出来的**,不是用户标定的:范本只覆盖 Au(111)
    #: 修针那条线,没有谈金属性针尖。真机标定前它就是个保住相对关系的占位值。
    poke_depth_nm: float = 0.3
    #: 在临界深度反复扎几次。基础流程是 5 次(要把簇做到最小最圆);这里 2 次就够,
    #: 多扎反而容易把已经合格的金属性针尖扎成一根更尖但更不稳的针。待真机标定。
    critical_repeat_n: int = 2
    poke_budget: int = 10
    poke_dwell_s: float = 0.5
    cluster_scan_nm: float = 10.0
    #: 圆度达标线 = **等效轴比**(`AssessClusterRoundness.min_axis_ratio`)。
    #: 「不比一个长短轴差 25% 的椭圆更不规则」。
    #:
    #: ⚠️ 2026-08-11 由 `round_threshold=0.65` 改名而来,**两个数不可换算**:
    #: 旧的比的是 `0.6*circularity + 0.4*aspect`,而那个 circularity = 4πA/P² 在
    #: 像素化边界上的**上确界只有 0.617**(轴对齐正方形却是 0.785)—— 阈值 0.65
    #: 卡在两者之间,**圆的一律不合格、方的一律合格**。改名是为了让漏改的地方
    #: 当场报错:两个阈值方向相同,静默沿用旧值会**照跑不误**并把闸门放宽。
    #:
    #: 合取的另一半 `min_aspect` 留在技能默认值(0.6),这里不重复暴露。
    #: **未标定**:0.75 是从「离散↔椭圆轴比」的物理映射上取的整数点,
    #: 不是对哪一批帧拟合出来的。
    min_axis_ratio: float = 0.75

    # ── STS 采集条件 ──
    #: 稳定点条件(设谱之前先把针尖稳在这里)。知识库 Au(111) sts_params:
    #: −700 mV / 0.5 nA。待真机标定。
    stab_bias_v: float = -0.7
    stab_setpoint_a: float = 0.5e-9
    #: 谱窗口 = [onset − below, +above]。见模块注释:跟着衬底走,不写死。
    sweep_below_onset_v: float = 0.31
    sweep_above_v: float = 0.30
    sts_points: int = 400
    #: lock-in 调制。知识库 Au(111):713 Hz、4 K 下 2-5 mV。
    #: **幅度是 rms 还是峰值,真机验收单里有一条专门核对**(ConfigureLockIn 的
    #: amplitude_v 语义要在硬件上确认过才敢当 rms 用)。待真机标定。
    mod_freq_hz: float = 713.0
    mod_amp_v: float = 0.005
    settle_s: float = 1.0

    # ── 判据 ──
    #: onset 位置容差。知识库判据是「sharp step within ±20 mV of expected」。
    onset_tol_v: float = 0.020
    #: 样品温度 —— 只用于算展宽下限(3.5 kT/e),不驱动任何硬件。
    temperature_k: float = 4.2
    #: 台阶宽度上限(10-90)。超过它的「台阶」是一段缓慢抬升,不是 onset。
    onset_width_max_v: float = 0.040

    # ── 回路 ──
    #: 扎针 ↔ 测谱最多来回几次。预算是行动上限,不是重试次数 —— 打不动就如实
    #: 报告并停下。
    max_rounds: int = 3
    #: 测谱点与刚扎出的簇之间至少隔多远(在簇上测谱,测的是簇不是表面态)。
    sts_spot_separation_nm: float = 3.0
    scan_timeout_s: float = 300.0

    def sweep_window_v(self, onset_v: float) -> tuple[float, float]:
        """给定衬底的 onset,返回 ``(start_v, end_v)``。

        窗口必须**跨过费米面**并把 onset 留在里面且两侧都有数据 —— 判据要靠
        onset 两边的背景才能把台阶与窗口边缘的斜率区分开。
        """
        start = float(onset_v) - abs(float(self.sweep_below_onset_v))
        end = abs(float(self.sweep_above_v))
        return (min(start, end), max(start, end))

    def poke_workflow(self) -> NobleTipWorkflow:
        """本配方的精修档,翻译成 ``poke_phase`` 认识的流程表。"""
        return _noble_resolve({
            "poke_depth_nm": self.poke_depth_nm,
            "critical_repeat_n": self.critical_repeat_n,
            "poke_budget": self.poke_budget,
            "poke_dwell_s": self.poke_dwell_s,
            "cluster_scan_nm": self.cluster_scan_nm,
            "min_axis_ratio": self.min_axis_ratio,
            "scan_timeout_s": self.scan_timeout_s,
        })


# ── 配方 2:原子分辨针尖 ────────────────────────────────────────────────────

@dataclass(frozen=True)
class AtomicTipWorkflow:
    """快扫小平区 + 偏压随机扰动,直到图里出现原子相。

    编排上的一个决定:**扰动打在牺牲帧里,评估用紧接着的一张干净帧**。扰动期间
    帧内对比度逐行突变,FFT 证据会被自己污染 —— 同帧评估等于拿被污染的证据判针
    尖。这也忠实于现场的手法:做两下就停下来看扫出来的图。
    """

    # ── 成像条件(用户配方值) ──
    #: 「偏压调到 20 mV 电流调到 500 pA」。
    eval_bias_v: float = 0.020
    eval_setpoint_a: float = 500e-12
    #: 「在一个 10nm 或者 5nm 的没台阶的地方快速扫图」。5 nm / 256 px 过尺度门。
    eval_frame_nm: float = 5.0
    eval_pixels: int = 256
    #: 「很快地扫图」—— 一帧 256 线 × 0.15 s ≈ 38 s,慢轴漂移还不至于毁掉判据。
    eval_line_time_s: float = 0.15
    #: 牺牲帧:与评估帧同尺寸但偏开一点,把扰动的损伤散布在别处。
    sacrificial_offset_nm: float = 8.0

    # ── 偏压扰动包络(交给 BiasWiggle) ──
    #: 「随机在正负 20mV 之内高频率地切换偏压」。下限不是 0:恒流反馈下 |V| 越小
    #: 针尖被推得越近,在零附近逗留会撞针(见 bias_settle 的死区论证)。
    wiggle_lower_v: float = 0.004
    wiggle_upper_v: float = 0.020
    #: 「不能太快太高频」—— 每档停留 50-150 ms,约 7-20 Hz,接近手拉滑条的节奏。
    wiggle_dwell_min_s: float = 0.05
    wiggle_dwell_max_s: float = 0.15
    #: 「偏压斜率变化不要太大,怕仪器受不了」。
    wiggle_slew_v_per_s: float = 1.0
    #: 「做两下」= 一次突发。
    wiggle_burst_s: float = 5.0
    #: 扰动期间电流超过它立即恢复偏压并中止。
    wiggle_abort_current_a: float = 5e-9

    # ── 判据 ──
    snr_min: float = 4.0
    #: 角向集中度 —— 分开真晶格与准周期抖动的那一条(vision.atomic_phase 有实测)。
    concentration_min: float = 20.0
    sharpness_min: float = 8.0

    # ── 回路 ──
    max_cycles: int = 6
    #: 「实在搞不出来就退回扎针尖法里面去扎两下,回来再搞」。
    cycles_per_fallback: int = 2
    fallback_budget: int = 2
    #: 回退扎针用的档(比配方 1 更轻:只是把针尖重新弄活,不是重塑)。
    fallback_poke_depth_nm: float = 0.5
    fallback_repeat_n: int = 2
    fallback_poke_budget: int = 6
    cluster_scan_nm: float = 10.0
    #: 圆度达标线 = **等效轴比**(`AssessClusterRoundness.min_axis_ratio`)。
    #: 「不比一个长短轴差 25% 的椭圆更不规则」。
    #:
    #: ⚠️ 2026-08-11 由 `round_threshold=0.65` 改名而来,**两个数不可换算**:
    #: 旧的比的是 `0.6*circularity + 0.4*aspect`,而那个 circularity = 4πA/P² 在
    #: 像素化边界上的**上确界只有 0.617**(轴对齐正方形却是 0.785)—— 阈值 0.65
    #: 卡在两者之间,**圆的一律不合格、方的一律合格**。改名是为了让漏改的地方
    #: 当场报错:两个阈值方向相同,静默沿用旧值会**照跑不误**并把闸门放宽。
    #:
    #: 合取的另一半 `min_aspect` 留在技能默认值(0.6),这里不重复暴露。
    #: **未标定**:0.75 是从「离散↔椭圆轴比」的物理映射上取的整数点,
    #: 不是对哪一批帧拟合出来的。
    min_axis_ratio: float = 0.75
    poke_dwell_s: float = 0.5
    #: 找平区时要求的无台阶窗口(比评估帧大一圈,免得台阶正好压在帧边)。
    flat_margin_nm: float = 2.0
    scan_timeout_s: float = 300.0

    def _scale_plan(self) -> "tuple[float | None, str | None, str]":
        """这一组帧参数在尺度门下的判定 —— 全部来自 ``vision.atomic_phase``。

        本配方**不再自己比阈值**:0.02 / 0.05 与「过渡带要多少像素」的算术都住在
        判据模块里(那里是它们的归属地),这里只消费判定、保留自己的措辞。
        """
        from mast.vision.atomic_phase import plan_scale

        # 像素数 0 / 负数不是「无穷细」而是一组坏参数,但历史行为是按 1 算再报
        # 「太粗」——保持不变:换成「参数非法」是另一件事,不混在这次归一里。
        return plan_scale(float(self.eval_frame_nm) * 1e-9,
                          max(int(self.eval_pixels), 1))

    def eval_pixel_size_nm(self) -> float:
        """评估帧的 nm/px —— 原子相判据的尺度门就看它。

        与 :meth:`scale_problem` 判定用的是**同一个数**(同一次 ``plan_scale``
        的算法),不是各算一遍:报出来的 nm/px 与拿去判的 nm/px 一旦能不一样,
        trace 里就会出现「0.0195 却说超标」这种没法查的事。
        """
        nmpp, _scale, _problem = self._scale_plan()
        return float(nmpp or 0.0)

    def scale_problem(self) -> str:
        """这组帧参数判不判得出原子相;没问题返回 ``""``。

        在**下发扫描之前**就能回答 —— 扫完一张判不了的图再说,白花一帧的时间,
        而且流程会把「判不了」误读成「还没弄出原子相」接着去扰动针尖。

        判定归 ``atomic_phase.plan_scale``,措辞留在这里(「评估帧」是本配方的话)。
        """
        from mast.vision.atomic_phase import (
            SCALE_FULL_NMPP,
            SCALE_OFF_NMPP,
            min_pixels_for_scale,
        )

        nmpp, scale, _problem = self._scale_plan()
        if scale == "off":
            return (f"评估帧 {self.eval_frame_nm:g} nm / {self.eval_pixels} px "
                    f"= {nmpp:.4f} nm/px，超过 {SCALE_OFF_NMPP} nm/px —— 这个尺度上"
                    f"晶格物理上不可分辨，判据只会说「判不了」。请缩小视野或加大"
                    f"像素数。")
        if scale == "reduced":
            return (f"评估帧 {nmpp:.4f} nm/px 落在过渡带 "
                    f"[{SCALE_FULL_NMPP}, {SCALE_OFF_NMPP}] —— 判据会给出结论但"
                    f"证据强度不足以当验收依据。建议 {self.eval_frame_nm:g} nm 用 "
                    f"{min_pixels_for_scale(float(self.eval_frame_nm) * 1e-9)} px 以上。")
        return ""

    def wiggle_params(self) -> dict[str, Any]:
        """交给 ``BiasWiggle`` 的一组参数。"""
        return {
            "base_bias_v": float(self.eval_bias_v),
            "wiggle_lower_v": float(self.wiggle_lower_v),
            "wiggle_upper_v": float(self.wiggle_upper_v),
            "dwell_min_s": float(self.wiggle_dwell_min_s),
            "dwell_max_s": float(self.wiggle_dwell_max_s),
            "slew_rate_v_per_s": float(self.wiggle_slew_v_per_s),
            "burst_s": float(self.wiggle_burst_s),
            "abort_current_a": float(self.wiggle_abort_current_a),
        }

    def poke_workflow(self) -> NobleTipWorkflow:
        """回退扎针用的流程表。"""
        return _noble_resolve({
            "poke_depth_nm": self.fallback_poke_depth_nm,
            "critical_repeat_n": self.fallback_repeat_n,
            "poke_budget": self.fallback_poke_budget,
            "poke_dwell_s": self.poke_dwell_s,
            "cluster_scan_nm": self.cluster_scan_nm,
            "min_axis_ratio": self.min_axis_ratio,
            "scan_timeout_s": self.scan_timeout_s,
        })


#: 出厂基线。两者都**待真机标定**。
SPECTROSCOPY_TIP = SpectroscopyTipWorkflow()
ATOMIC_TIP = AtomicTipWorkflow()


#: 字段边界。**越界拒绝,不夹紧** —— 与 ``noble_tip_workflow._BOUNDS`` 同一条
#: 论证:这里每个数字都会变成真实的硬件动作,夹紧会让调用方以为自己设的是 X 而
#: 实际跑的是 Y。
_BOUNDS: dict[str, tuple[float, float]] = {
    # 精修档(与 noble_tip_workflow 的同名字段保持同一区间)
    "poke_depth_nm": (0.01, 100.0),
    "critical_repeat_n": (1, 50),
    "poke_budget": (1, 200),
    "poke_dwell_s": (0.0, 10.0),
    "cluster_scan_nm": (0.5, 500.0),
    "min_axis_ratio": (0.0, 1.0),
    "scan_timeout_s": (1.0, 3600.0),
    # STS
    "stab_bias_v": (-10.0, 10.0),
    "stab_setpoint_a": (1e-12, 1e-7),
    "sweep_below_onset_v": (0.01, 5.0),
    "sweep_above_v": (0.0, 5.0),
    "sts_points": (16, 10000),
    "mod_freq_hz": (1.0, 100000.0),
    # 与 ConfigureLockIn 的 amplitude_v 上限一致(0..1 V)。
    "mod_amp_v": (1e-5, 1.0),
    "settle_s": (0.0, 60.0),
    "onset_tol_v": (0.001, 0.5),
    "temperature_k": (0.01, 400.0),
    "onset_width_max_v": (0.001, 1.0),
    "max_rounds": (1, 20),
    "sts_spot_separation_nm": (0.0, 500.0),
    # 原子分辨
    "eval_bias_v": (-10.0, 10.0),
    "eval_setpoint_a": (1e-12, 1e-7),
    "eval_frame_nm": (0.5, 500.0),
    "eval_pixels": (16, 4096),
    "eval_line_time_s": (1e-3, 60.0),
    "sacrificial_offset_nm": (0.0, 500.0),
    # 扰动包络。上限刻意宽于 BiasWiggle 自己的硬帽 —— 流程表不是安全层,真正的
    # 拒绝发生在技能里(那里 |V| ≤ 0.1 V、突发 ≤ 10 s、斜率 ≤ 2 V/s 写死在代码里)。
    "wiggle_lower_v": (0.0005, 0.1),
    "wiggle_upper_v": (0.001, 0.1),
    "wiggle_dwell_min_s": (0.01, 5.0),
    "wiggle_dwell_max_s": (0.01, 10.0),
    "wiggle_slew_v_per_s": (0.005, 2.0),
    "wiggle_burst_s": (0.1, 10.0),
    "wiggle_abort_current_a": (1e-11, 1e-6),
    "snr_min": (1.0, 1e6),
    "concentration_min": (1.0, 1e9),
    "sharpness_min": (1.0, 1e6),
    "max_cycles": (1, 50),
    "cycles_per_fallback": (1, 50),
    "fallback_budget": (0, 20),
    "fallback_poke_depth_nm": (0.01, 100.0),
    "fallback_repeat_n": (1, 50),
    "fallback_poke_budget": (1, 200),
    "flat_margin_nm": (0.0, 100.0),
}

_INT_FIELDS = frozenset({
    "critical_repeat_n", "poke_budget", "sts_points", "max_rounds",
    "eval_pixels", "max_cycles", "cycles_per_fallback", "fallback_budget",
    "fallback_repeat_n", "fallback_poke_budget",
})


def _resolve_into(base, overrides: "dict[str, Any] | None"):
    """出厂基线 + 显式给的值;``None`` 当没给,越界**丢弃并写日志**。"""
    if not overrides:
        return base
    known = {f.name for f in fields(base)}
    clean: dict[str, Any] = {}
    for key, val in overrides.items():
        if key not in known or val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            logger.warning("特异化针尖流程: %s=%r 不是数字,忽略", key, val)
            continue
        lo, hi = _BOUNDS.get(key, (float("-inf"), float("inf")))
        if not (lo <= num <= hi):
            logger.warning("特异化针尖流程: %s=%g 超出 [%g, %g],忽略(不夹紧)",
                           key, num, lo, hi)
            continue
        clean[key] = int(round(num)) if key in _INT_FIELDS else num
    return replace(base, **clean) if clean else base


def resolve_spectroscopy(overrides: "dict[str, Any] | None" = None,
                         base: SpectroscopyTipWorkflow = SPECTROSCOPY_TIP
                         ) -> SpectroscopyTipWorkflow:
    """配方 1 的流程表:出厂基线 + 用户显式给的值。"""
    return _resolve_into(base, overrides)


def resolve_atomic(overrides: "dict[str, Any] | None" = None,
                   base: AtomicTipWorkflow = ATOMIC_TIP) -> AtomicTipWorkflow:
    """配方 2 的流程表:出厂基线 + 用户显式给的值。"""
    return _resolve_into(base, overrides)


__all__ = [
    "SpectroscopyTipWorkflow",
    "AtomicTipWorkflow",
    "SPECTROSCOPY_TIP",
    "ATOMIC_TIP",
    "resolve_spectroscopy",
    "resolve_atomic",
]
