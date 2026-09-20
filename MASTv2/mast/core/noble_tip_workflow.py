"""贵金属表面的针尖修整流程参数与派生计算。

流程表定义进针、建结、电脉冲、扫图评估、机械修整、预算与步进。
tip_conditioning_policy 定义当前针尖的安全包络；执行仍由该包络约束。

参数是可配置的算法基线，不是任何目标仪器的标定报告。显式指定值与
未指定默认值分开处理：违法请求不被静默夹紧，默认值的必要调整会报告。
扫描速度由视野与线时派生，超时预算须覆盖最终实际下发的扫描参数。
针尖身份与污染避让半径分别由对应的共享模块提供，避免多份状态分叉。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NobleTipWorkflow:
    """一次针尖修整流程的全部判据与预算。单位写在字段名里。"""


    approach_bias_v: float = 4.0


    junction_bias_v: float = 0.05
    junction_setpoint_a: float = 1.0e-9

    # ── A 大修(电脉冲)─────────────────────────────────────────────────
    #: 起始脉冲电压。**会过针尖方案表的包络**,不合适的针尖上会被拒绝。
    pulse_v: float = 10.0
    pulse_width_s: float = 0.5

    #: down 方向沿用的阈值未标定；目标仪器需独立核验。
    pulse_success_dz_nm: float = 5.0

    pulses_per_polarity: int = 4

    pulse_same_spot_budget: int = 4
    #: 大修阶段的总发数上限。行动预算,不是重试次数 —— 打不动就该报告。
    pulse_budget: int = 20

    descend_pulse_v: tuple[float, ...] = (7.0, 5.0, 3.0)


    verify_bias_v: float = 1.0
    verify_setpoint_a: float = 1.0e-10

    verify_scan_nm: float = 50.0

    verify_pixels: "int | None" = 128

    verify_line_time_s: "float | None" = None
    #: 快扫方向上 z-forward 与 z-backward 的相似度下限。允许整体 shift ——
    #: 扫得快时两个方向本来就差一个滞后,看的是对齐之后还重不重合。
    fwdbwd_threshold: float = 0.80
    #: 验证↔大修之间来回几次就认输并如实报告。
    max_verify_rounds: int = 3


    step_scan_nm: float = 200.0

    step_fallback_nm: "float | None" = None

    flat_region_nm: float = 50.0

    step_pixels: "int | None" = None
    step_line_time_s: "float | None" = None

    accept_sharp_edge_nm: "float | None" = None


    poke_depth_nm: float = 0.5

    poke_bias_qplus_v: float = 0.020

    poke_bias_v: "float | None" = None

    poke_bias_via_shaper: bool = False

    poke_bias_slew_v_per_s: float = 0.2
    #: 改完偏压之后、下压之前的静置时间。同上,**未标定**。
    poke_bias_settle_s: float = 1.0


    poke_amp_baseline_s: float = 0.3
    #: 扎之后盯振幅的时长(≈ 几个 τ)。
    poke_amp_watch_s: float = 1.0
    #: 采样率(振幅走 ``Signals_ValGet`` 通用路,实际上限约 50 Hz)。
    poke_amp_poll_hz: float = 50.0
    #: 等 ring-down 回到基线的超时。**超时不判失败**,只如实记「没等到」。
    poke_ringdown_timeout_s: float = 5.0

    poke_ringdown_settle_ratio: float = 1.5

    poke_same_depth_retries: int = 3

    poke_dwell_s: float = 0.5
    #: 没扎上就加深这么多比例再来。
    #:
    #: ⚠️ 这条只服务「**没碰到**」。碰到了但簇不好是另一回事,那时候要
    #: **变浅**(见 `poke_shallow_frac`)—— 两种失败长得像,处置正相反。
    poke_deepen_frac: float = 0.20

    poke_shallow_frac: float = 0.40

    poke_depth_min_nm: float = 0.2

    poke_depth_max_nm: float = 2.0
    #: 看 cluster 的小图视野。
    cluster_scan_nm: float = 10.0

    poke_flat_window_nm: float = 35.0

    poke_flat_dry_refills: int = 20

    poke_site_separation_nm: float = 30.0

    poke_site_scan_nm: float = 200.0
    poke_site_pixels: int = 128
    #: **1.172 s = 200 / 170.65**,由 ``forge_v_tip_nm_s`` 派生,不是挑的数 ——
    #: 视野翻倍而线时不变就等于把针尖速度也翻倍(341 nm/s)。
    #:
    #: 不靠 ``forge_line_time_s()`` 自动限速来兜:那条路会**每次跑都吐一条
    #: 「已放慢」的说明**,而说明是给「出乎意料」用的。存着的值和实际跑的值
    #: 不一致本身就是下一个人对不上账的起点(89 张分割图那次的教训)。
    poke_site_line_time_s: float = 1.172
    #: 簇小图的分辨率与每线时间。语义同 ``step_pixels`` / ``step_line_time_s``:
    #: ``None`` = 走档位表。**与台阶图分开是有意的** —— 10 nm 的簇图和 100 nm 的
    #: 台阶图不该共用一个数,哪怕今天它们的值恰好相同(只测过一个工作点)。
    cluster_pixels: "int | None" = None
    cluster_line_time_s: "float | None" = None

    min_axis_ratio: float = 0.75

    critical_start_pm: float = 300.0
    critical_step_pm: float = 50.0

    critical_repeat_n: int = 3

    poke_unround_streak_to_pulse: int = 8
    #: 「回退打脉冲」在一站里最多做几次。用完就换区 —— 同一片表面上反复
    #: 「打一轮脉冲再扎一轮」不收敛的话,问题多半不在这一片表面上。
    poke_pulse_rescues: int = 2
    poke_budget: int = 30


    forge_scan_nm: float = 100.0

    forge_step_scan_nm: float = 100.0
    #: 100 nm 上找不到台阶时的回退视野。
    #:
    #: 只用来**找台阶**,不重判正反扫、也不当验收图。找不到就找不到 ——
    #: 「不要把小起伏硬算作台阶」。
    forge_step_fallback_nm: float = 200.0
    #: 看簇的小图。深扎坑尺度远小于验证帧,单独给一档。
    forge_cluster_scan_nm: float = 10.0


    forge_step_pixels: int = 256
    forge_step_line_time_s: float = 0.15

    forge_verify_pixels: int = 256
    forge_verify_line_time_s: float = 0.586
    forge_cluster_pixels: int = 256
    forge_cluster_line_time_s: float = 0.15


    forge_v_tip_nm_s: float = 170.65


    forge_pixels: "int | None" = None
    forge_line_time_s: "float | None" = None

    forge_scan_timeout_s: float = 1300.0

    # ── 通用 ────────────────────────────────────────────────────────────
    #: 换点之后等一下再动手 —— 一次横向移动会重新激起压电蠕变。
    move_settle_s: float = 0.3
    #: 单次扫描的等待上限。
    scan_timeout_s: float = 300.0


#: 出厂基线:Au / Ag / Cu 单晶,或 mica 上这三者的膜。
NOBLE_METAL_BASELINE = NobleTipWorkflow()


#: 每个字段的可接受范围。**越界拒绝,不夹紧** —— 夹了调用方会以为自己设的是 X
#: 而实际跑的是 Y,而这里每个数字都会变成真实的硬件动作。
_BOUNDS: dict[str, tuple[float, float]] = {
    "approach_bias_v": (-10.0, 10.0),
    "junction_bias_v": (-10.0, 10.0),
    "junction_setpoint_a": (1e-12, 1e-7),
    "pulse_v": (-10.0, 10.0),
    "pulse_width_s": (1e-3, 2.0),
    "pulse_success_dz_nm": (0.1, 1000.0),
    "pulses_per_polarity": (1, 50),
    "pulse_same_spot_budget": (1, 50),
    "pulse_budget": (1, 200),
    "verify_bias_v": (-10.0, 10.0),
    "verify_setpoint_a": (1e-12, 1e-7),
    "verify_scan_nm": (1.0, 5000.0),
    # 与 SetScanBuffer 的 ParameterSpec 对齐 —— 表里存一个下游必然拒绝的值
    # 没有意义(同 scan_policy._TIER_FIELD_SPEC 那条自述)。
    "verify_pixels": (16, 4096),
    "fwdbwd_threshold": (0.0, 1.0),
    "max_verify_rounds": (1, 20),
    "step_scan_nm": (1.0, 5000.0),
    "flat_region_nm": (1.0, 1000.0),
    # 与 SetScanBuffer / ConfigureScan 的 ParameterSpec 对齐 —— 表里存一个下游
    # 必然拒绝的值没有意义(同 scan_policy._TIER_FIELD_SPEC 那条自述)。
    "step_pixels": (16, 4096),
    "step_line_time_s": (1e-4, 600.0),
    "cluster_pixels": (16, 4096),
    "cluster_line_time_s": (1e-4, 600.0),
    "forge_step_pixels": (16, 4096),
    "forge_step_line_time_s": (1e-4, 600.0),
    "forge_cluster_pixels": (16, 4096),
    "forge_cluster_line_time_s": (1e-4, 600.0),
    "forge_pixels": (16, 4096),
    "forge_line_time_s": (1e-4, 600.0),

    "forge_v_tip_nm_s": (1.0, 488.0),
    "poke_depth_nm": (0.01, 100.0),
    # 扎针偏压。上限 1 V 是刻意的:范本里这个数是 **20 mV**,而它存在的理由是
    # 「别把音叉激起来」—— 一个几伏的"扎针偏压"不是调参,是把这条物理反过来用。
    # 真要在高偏压下动表面,那是 pulse 的活,走 pulse 的闸门。
    "poke_bias_qplus_v": (-1.0, 1.0),
    "poke_bias_v": (-1.0, 1.0),
    "poke_same_depth_retries": (0, 20),
    "poke_bias_slew_v_per_s": (0.01, 100.0),
    "poke_bias_settle_s": (0.0, 30.0),
    "poke_amp_baseline_s": (0.05, 10.0),
    "poke_amp_watch_s": (0.05, 30.0),
    "poke_amp_poll_hz": (1.0, 2000.0),
    "poke_ringdown_timeout_s": (0.0, 60.0),
    "poke_ringdown_settle_ratio": (1.0, 100.0),
    "poke_dwell_s": (0.0, 10.0),
    "poke_deepen_frac": (0.01, 5.0),
    # 上界 1.0 **开区间**(见 `_EXCLUSIVE_UPPER`)—— 填 1.0 就是「不变浅」、
    # 填 >1 就是偷偷变回加深。这条分支的**方向**是它存在的全部理由,
    # 而 `depth * frac` 这行代码本身认不出方向,所以闸门必须认。
    "poke_shallow_frac": (0.01, 1.0),
    "poke_depth_min_nm": (0.01, 100.0),
    # 平区窗上界取找台面那张图的视野:窗比图还大,``FindFlatRegion`` 永远
    # 找不到,而它会如实报「这里没有」⇒ 流程会一直重扫到认输。所以这里拦住。
    "poke_flat_window_nm": (1.0, 5000.0),
    "poke_flat_dry_refills": (1, 100),
    "poke_depth_max_nm": (0.05, 100.0),
    "cluster_scan_nm": (0.5, 500.0),
    "min_axis_ratio": (0.0, 1.0),
    "critical_start_pm": (1.0, 10000.0),
    "critical_step_pm": (1.0, 5000.0),
    "critical_repeat_n": (1, 50),
    "poke_unround_streak_to_pulse": (1, 100),
    "poke_pulse_rescues": (0, 10),
    "poke_budget": (1, 200),
    "forge_scan_nm": (1.0, 5000.0),
    "forge_step_scan_nm": (1.0, 5000.0),
    "forge_cluster_scan_nm": (0.5, 500.0),
    "forge_scan_timeout_s": (1.0, 3600.0),
    "move_settle_s": (0.0, 10.0),
    "scan_timeout_s": (1.0, 3600.0),
}


_EXCLUSIVE_UPPER = frozenset({"forge_v_tip_nm_s", "poke_shallow_frac"})

_INT_FIELDS = frozenset({"pulses_per_polarity", "pulse_same_spot_budget",
                        "poke_unround_streak_to_pulse", "poke_pulse_rescues",
                        "pulse_budget", "max_verify_rounds",
                         "critical_repeat_n", "poke_budget",
                         "poke_same_depth_retries", "poke_flat_dry_refills",
                         "step_pixels", "cluster_pixels",
                         "forge_step_pixels", "forge_cluster_pixels",
                         "forge_pixels", "verify_pixels"})


def resolve(overrides: "dict[str, Any] | None" = None,
            base: NobleTipWorkflow = NOBLE_METAL_BASELINE) -> NobleTipWorkflow:
    """出厂基线 + 用户显式给的值。

    ``None`` 当作没给 —— 技能的可选参数不填时就是 None,那正是「按流程表来」的
    意思,不是「设成 0」。越界的值**丢弃并写日志**,不夹紧。
    """
    if not overrides:
        return base
    known = {f.name for f in fields(base)}
    clean: dict[str, Any] = {}
    for key, val in overrides.items():
        if key not in known or val is None:
            continue
        if key == "descend_pulse_v":
            try:
                raw = tuple(val)
            except TypeError:
                logger.warning("修针流程: descend_pulse_v=%r 不是序列,忽略", val)
                continue
            seq = tuple(float(v) for v in raw
                        if isinstance(v, (int, float)) and -10.0 <= float(v) <= 10.0)
            if len(seq) != len(raw):
                logger.warning("修针流程: descend_pulse_v 有超出 ±10 V 的值,已丢弃")
            clean[key] = seq
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            logger.warning("修针流程: %s=%r 不是数字,忽略", key, val)
            continue
        lo, hi = _BOUNDS.get(key, (float("-inf"), float("inf")))
        hi_open = key in _EXCLUSIVE_UPPER
        if not (lo <= num <= hi) or (hi_open and num >= hi):
            logger.warning("修针流程: %s=%g 超出 [%g, %g%s,忽略(不夹紧)",
                           key, num, lo, hi, ")" if hi_open else "]")
            continue
        clean[key] = int(round(num)) if key in _INT_FIELDS else num
    return replace(base, **clean) if clean else base



def forge_line_time_s(wf: NobleTipWorkflow, size_nm: float,
                      line_time_s: "float | None"
                      ) -> "tuple[float | None, str | None]":
    """返回评估图的每线时间及必要说明。
    
    None 保持 None，表示由 ScanAt 档位表解析；无效数值交由下游验证。
    已经低于流程速度上限的线时保持不变。未显式覆写时，超速线时按
    视野除以上限派生；显式覆写保持原值并报告超限，避免静默改变请求。
    这是流程参数派生层，不替代执行层的硬件安全包络。
    """
    if line_time_s is None:
        return None, None
    lt = float(line_time_s)
    fov = float(size_nm)
    ceiling = float(wf.forge_v_tip_nm_s)
    if lt <= 0 or fov <= 0 or ceiling <= 0:
        # 这几个值不合法是上游出错的信号,不是「一个很快的扫描」。这一层不替
        # 上游发明数字,原样放行 —— 下游 ``validate_params`` / ``scan_resolver``
        # 有各自的拒绝口,它们比这里更知道该说什么。
        return lt, None
    speed = fov / lt
    if speed <= ceiling:
        return lt, None
    if wf.forge_line_time_s is not None:
        return lt, (
            f"{fov:.0f} nm 图按你钉的 {lt:g} s/线跑 ⇒ 针尖 {speed:.0f} nm/s,"
            f"超过 forge 速度上限 {ceiling:g} nm/s(**未夹紧**:你逐字说过的数照跑)。"
            "请核验显式设置是否适用于目标仪器与当前针尖。")
    capped = fov / ceiling
    return capped, (
        f"{fov:.0f} nm 图的每线时间由 {lt:g} s 派生为 {capped:.4g} s —— "
        f"照流程表的数会跑到针尖 {speed:.0f} nm/s,超过 forge 速度上限 "
        f"{ceiling:g} nm/s。帧时因此变长,**视野不在帧时的式子里**。")


def forge_fallback_line_time_s(wf: NobleTipWorkflow
                               ) -> "tuple[float | None, bool]":
    """派生更大回退视野的线时，返回线时及是否保留显式覆写。
    
    默认按视野比例缩放线时以保持针尖横向速度；显式覆写保持原值。
    报告与执行必须复用此函数，避免展示的速度与实际参数不一致。
    """
    lt = wf.step_line_time_s
    if lt is None or not wf.step_scan_nm or not wf.step_fallback_nm:
        return lt, False
    if wf.forge_line_time_s is not None:
        return float(lt), True
    return (float(lt) * float(wf.step_fallback_nm) / float(wf.step_scan_nm),
            False)


def forge_speed_notes(wf: NobleTipWorkflow) -> list[str]:
    """计算每种评估图的针尖横向速度，并生成与执行参数一致的说明。
    
    速度由视野除以线时得到；缺线时表示走档位表，不编造具体速度。
    回退图复用统一的线时派生函数。
    """
    frames = (
        ("验证帧", wf.verify_scan_nm, wf.verify_line_time_s),
        ("台阶/验收图", wf.step_scan_nm, wf.step_line_time_s),

        ("台阶回退图", wf.step_fallback_nm, forge_fallback_line_time_s(wf)[0]),
        ("找台面图", wf.poke_site_scan_nm, wf.poke_site_line_time_s),
        ("簇图", wf.cluster_scan_nm, wf.cluster_line_time_s),
    )
    notes: list[str] = []
    for label, fov, raw_lt in frames:
        lt, _why = forge_line_time_s(wf, float(fov or 0.0), raw_lt)
        if not lt or not fov:
            # 「走档位表」和「跑多快」是两件事,而这一层看不见档位表的解析结果。
            # 报成一个具体的 nm/s 就是把「读不到」折叠成一个值 —— 本仓一天记过
            # 四次的那个形状。如实说不知道。
            notes.append(f"{label} {float(fov or 0):.0f} nm:每线时间走档位表,"
                         "针尖速度在这一层算不出来")
            continue
        notes.append(f"{label} {float(fov):.0f} nm / {lt:.4g} s/线 ⇒ 针尖 "
                     f"{float(fov) / lt:.0f} nm/s")
    return notes



_ENVELOPE_REL_TOL = 1e-9


def within_envelope(value_in_policy_units: float, limit: float) -> bool:
    """``|value| <= limit``,吸收单位换算的浮点末位。见 :data:`_ENVELOPE_REL_TOL`。

    判据只有一份:运行时的对账与结构闸门共用它。两份「差不多相等」的实现迟早
    在最低位上各自漂移,而那正是这条容差要挡的东西。
    """
    return abs(float(value_in_policy_units)) <= float(limit) * (1.0 + _ENVELOPE_REL_TOL)


#: 工作流字段 → (方案表的包络键, 方案表的推荐值键, 本表单位→方案表单位的换算)
_ENVELOPE_PAIRS: tuple[tuple[str, str, "str | None", float], ...] = (
    # 脉冲电压:两边都是伏特。
    ("pulse_v", "max_abs_pulse_v", "pulse_v", 1.0),
    # 深扎首深:本表 nm,方案表米。
    ("poke_depth_nm", "max_poke_depth_m", "poke_deep_depth_m", 1e-9),
    # 深扎的加深上限。方案表没有对应的「上限推荐值」,所以违法时取包络本身 ——
    # 那不是夹紧一个请求,是把一个够不着的天花板降到实际够得着的高度。
    ("poke_depth_max_nm", "max_poke_depth_m", None, 1e-9),
)


def _tip_policy_now() -> "dict[str, Any] | None":
    """当前针尖的方案表(含包络)。整条链坏掉才返回 None。

    ⚠️ **「未登记针尖」不是「没有包络」**(2026-08-10 更正)。这个函数的第一版在
    ``current_tip_facts()`` 返回空时直接 return None、不对账,理由写的是
    「系统不知道装的是什么针,没资格替用户改他的默认值」——**那条理由用错了地方**:

    未登记时 ``resolve_policy(None)`` 给的是**通用保守档**,而那一档有自己的包络
    (``max_abs_pulse_v = 6.0``),``_check_envelope`` 照样对着它拒绝。
    于是「不对账」在这里不是 fail-open,是**保证失败**:出厂 ``pulse_v = 10 V``
    > 6 V ⇒ **未登记针尖上一发脉冲都打不出去**,而未登记是本系统最常见的状态。

    真正 fail-open 的是另一道门:``_tip_policy.qplus_gate``(qPlus **策略**门)在
    读不到针尖时放行。两道门、两条哲学,第一版把其中一道的性质安到了另一道头上
    ——``tip_conditioning_resolver`` 的模块注释犯的是同一个错,已一并改掉。

    只有**整条链不可用**(import 失败、解析器抛异常)才 return None:那时确实没有
    包络可对,而一个坏掉的诊断链不该让修针失败。
    """
    try:
        from mast.core.tip_conditioning_policy import resolve_policy
        from mast.core.tip_state import current_tip_facts

        try:
            facts = current_tip_facts()
        except Exception as exc:  # noqa: BLE001 — 读不到登记 = 按未登记处理
            logger.debug("修针流程: 读针尖登记失败(按未登记的通用档对账): %s", exc)
            facts = None
        # facts 为空 → 通用保守档(**不是**「没有包络」)。
        return resolve_policy(facts or None)
    except Exception as exc:  # noqa: BLE001 — 对账不可用绝不能让修针失败
        logger.debug("修针流程: 读针尖方案表失败(不对账): %s", exc)
        return None


def reconcile_with_tip_envelope(
    workflow: NobleTipWorkflow,
    *,
    specified: "frozenset[str] | set[str] | tuple[str, ...]" = (),
    policy: "dict[str, Any] | None" = None,
) -> "tuple[NobleTipWorkflow, list[str]]":
    """把**违反当前针尖包络的默认值**换成这根针自己那张表推荐的值。

    ``specified`` 是用户逐字给过的字段名 —— 这些**一个都不动**,它们照旧送去过
    包络门,超了就被拒绝(那正是「拒绝不夹紧」要保住的东西)。

    返回 ``(工作流, 说明列表)``。说明列表为空 = 什么都没换。
    """
    pol = _tip_policy_now() if policy is None else policy
    if not pol:
        return workflow, []
    given = set(specified or ())
    changes: dict[str, float] = {}
    notes: list[str] = []
    for field_name, limit_key, rec_key, scale in _ENVELOPE_PAIRS:
        if field_name in given:
            continue
        limit = _num_or_none(pol.get(limit_key))
        if limit is None or limit <= 0:
            continue
        current = _num_or_none(getattr(workflow, field_name, None))
        if current is None:
            continue
        if within_envelope(current * scale, limit):
            continue                      # 合法,不碰
        # 优先用方案表自己的推荐值;它按构造落在包络内,而且是**针对这根针**的。
        replacement = None
        if rec_key:
            rec = _num_or_none(pol.get(rec_key))
            if rec is not None and within_envelope(rec, limit):
                replacement = abs(rec) / scale
        if replacement is None:
            replacement = limit / scale
        lo, hi = _BOUNDS.get(field_name, (float("-inf"), float("inf")))
        if not (lo <= replacement <= hi):
            continue                      # 换出去也会被本表自己的边界拒,那就别换
        changes[field_name] = replacement
        notes.append(
            f"{field_name}: 出厂默认 {current:g} 超出当前针尖的 {limit_key}="
            f"{limit:g}（超 {abs(current) * scale / limit:.2g}×），"
            f"已改用这根针方案表里的 {replacement:g}。"
            "（用户逐字指定的值不会被这样替换——那种情况仍然是拒绝，不夹紧。）"
        )
    if not changes:
        return workflow, []
    for note in notes:
        logger.warning("修针流程与针尖包络对账: %s", note)
    return replace(workflow, **changes), notes


def _num_or_none(value: Any) -> "float | None":
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return None if num != num else num


def descend_sequence(workflow: NobleTipWorkflow) -> tuple[float, ...]:
    """扫描验证失败后要补的递减脉冲 —— qPlus 上是空的。

    针尖类型的唯一真源是针尖登记(``tip_state.is_qplus``),不是另设一个设置项。
    读不到就当不是 qPlus:未登记针尖时系统不知道装的是什么,没资格替用户否决
    —— 与 ``_tip_policy.qplus_gate`` 同一条论证(而真正会戳到音叉的下压类动作,
    那道软门仍然默认拒绝)。
    """
    try:
        from mast.core.tip_state import is_qplus
        if is_qplus():
            return ()
    except Exception as exc:  # noqa: BLE001
        logger.debug("读针尖类型失败(按非 qPlus 处理): %s", exc)
    return tuple(workflow.descend_pulse_v)


__all__ = [
    "NobleTipWorkflow",
    "NOBLE_METAL_BASELINE",
    "resolve",
    "descend_sequence",
    "reconcile_with_tip_envelope",
    "within_envelope",
]
