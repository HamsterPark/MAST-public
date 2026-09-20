"""辅助通道：Z、qPlus 振幅、频率偏移与 lock-in 的低速只读采样。

采样通过 Signals_ValsGet 在电流泵的空闲预算内完成，复用监控 worker，
不重配示波器、不另起常驻线程。角色锁与急停通道保持独立。
采样间隔是节流上限；实际节奏由可用机会决定，必须从时间戳统计。
锁等待不超过调用方预算，读数失败应降级，maybe_sample 不向外抛错。

共振子的振幅弛豫时间为 Q/(π f0)，信息带宽约 f0/(2Q)；采样率应结合
目标仪器的读数与实际时间间隔解释。Z 高频噪声由独立 burst 通道处理。

Z 的量程与方向均来自目标仪器配置或读回。振幅只在确认进针、持续归零、
未处于动作屏蔽期且确认激励有效时参与告警；未驱动解调器的噪声底
不能当作自由振荡基线。缺失状态必须保留为未知。Δf 与 lock-in 只记录
不判级，原始值、统计量、判据与降级原因分别保存。公开快照不含现场数据。
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

#: 与电流泵同一个角色。6502(monitor) 已经满了（含撞针看门狗），6501(main) 上的
#: 阻塞读会抽干 API 线程池 —— 6503 是唯一有余量的，而且它本来就是「采数据」那一条。
DATA_ROLE = "data"

#: 角色锁最多等这么久。**绝不排队** —— 见模块 docstring 的硬约束 1。
_LOCK_TIMEOUT_S = 0.25

#: 锁等待的下限。搭车采样（``service._pump_idle``）会把 budget 一路传到
#: :meth:`AuxSampler._call`，锁等待取 ``min(_LOCK_TIMEOUT_S, budget)`` —— 但不能
#: 一路压到 0：0 秒的 acquire 在别人刚好持锁的那一瞬间必然失败，于是「扫描期间
#: 一次都采不到」会变成常态，而不是偶尔跳过一拍。
_LOCK_TIMEOUT_MIN_S = 0.01


_CADENCE_WINDOW_N = 21

#: 采样间隔的硬下限，压过设置值。
#:
#: 与 ``thresholds.FIELD_BOUNDS["cm_aux_interval_s"]`` 的下界**同一个数**，
#: 有测试钉住 —— 两处各写一个数就是「设置里能填 0.05，代码里其实卡在 0.1」
#: 这种只在实机上才现形的分歧。
_MIN_INTERVAL_S = 0.1

#: 探测失败之后至少隔这么久再探一次。用户在 Nanonis 里改信号配置不应该变成一场
#: 探测风暴（与 pump._RECONFIG_MIN_INTERVAL_S 同一理由）。
_REDISCOVER_MIN_INTERVAL_S = 60.0

#: 「音叉在不在被驱动」这一问的重问间隔。
#:
#: 它是**惰性**问句（与 ``AutoApproach_OnOffGet`` 同款）：只在振幅确实归零、
#: 也就是 ``amp_zero_on_approach`` 眼看要响的时候才发。正常运行期间一次都不发，
#: 有测试钉住 —— 5 Hz 下每拍两个额外往返会把「一次调用取回全部三路」这条设计
#: 直接作废。
#:
#: 30 s 而不是更长：激励是**用户随时会开关**的东西（切 AFM/STM 模式、
#: 调 PLL）。一个太长的缓存会让判据在他刚打开激励之后仍然沉默半天，
#: 而那半天正好是他最想让它工作的时候。
_EXCITATION_PROBE_MIN_INTERVAL_S = 30.0

#: 缓存的激励状态过了这么久就不再采信 —— 回到「不知道」，而不是继续用旧答案。
#:
#: 比重问间隔长，但**有限**。无限缓存的失败模式是：一次读到「在振」之后，
#: 用户关掉激励，而判据拿着一个几小时前的答案继续判 —— 那正是这条前置要挡的
#: 情形，只是晚了几个小时。
_EXCITATION_CACHE_TTL_S = 120.0

#: 窗口里至少要有这么多点，窗口统计量才有意义。少于此一律返回 None（判不了 ≠ 没有）。
_MIN_WINDOW_N = 12

#: 相邻两点的间隔超过标称值的这个倍数就不当作「相邻」——
#: 采样是机会式的（角色忙就跳过），一个跨了 5 秒的差分里装着 5 秒的漂移，
#: 会被读成一个台阶。同 ``features.py``「跨 run 的一阶差分会凭空造出台阶」。
_STEP_DT_TOLERANCE = 1.8

# 窗口容量覆盖合法配置的最长窗口与最短采样间隔，测试从 FIELD_BOUNDS 推导最坏情况以检测配置漂移。
_MAX_WINDOW_POINTS = 40000

#: 连续这么多份读不懂的回包就重新探测一次（同 pump._BAD_REPLY_LIMIT）。
#: 不设上限的话，一台回包结构不同的机器会每段白试一次，永远试下去。
_BAD_REPLY_LIMIT = 10


_DECAY_RATIO = 0.8

#: 读 Z 行程上下限最多试这么多次。见 ``AuxSampler._maybe_read_z_limits``：
#: 一次也不重试 = 一次瞬时忙就永久废掉 ``z_rail``；无限重试 = 在不支持这条命令的
#: 机器上每分钟一次白跑的往返。
_Z_LIMIT_MAX_TRIES = 5


_AMP_ZERO_HOLD_S = 5.0


_AMP_RING_FRAC = 2.0

#: 进针证据的保持时长（闩）。进针的证据是断续的：Nanonis 自动逼近在「压电伸出找电流」
#: 与「缩回、马达走一步」之间来回切，1 Hz 的 InstrumentState 快照看到的是一串闪烁。
#: 判据要求「最近 60 秒内出现过进针证据」而不是「此刻正在进针」，闪烁就不会把判据
#: 打成筛子。60 s 也覆盖「撞上之后用户刚停下自动逼近」的那一小段 —— 那恰恰是最想
#: 收到这条告警的时刻。
_APPROACH_LATCH_S = 60.0

#: |I| 低于设定点的这个比例 = **确证无结**。
#: ``approach.py::_engagement_bar`` 用的是「≥50% 才算隧穿已建立」。两个数
#: **故意不相等，方向也相反**：那条是「敢不敢说已经进针成功」，这条是「敢不敢说
#: 肯定还没有结」。中间 20%-50% 是刻意留的死区 —— 两句话都不说，判据闸门保持关闭。
_NO_JUNCTION_FRAC = 0.2

#: 惰性硬件问句的最小间隔。见 ``_ApproachGate.hardware_probe`` —— 只有在振幅已经
#: 归零、而技能与物理证据都没能确认进针时才会发出，正常运行期间一次都不发。
_APPROACH_PROBE_MIN_INTERVAL_S = 5.0

# 进针技能名提供动作上下文，但不能覆盖手动操作。另结合 Z 反馈、结状态与硬件查询；粗动技能名没有方向信息，不能单独视作进针。
APPROACH_SKILL_PATTERNS: frozenset[str] = frozenset({
    "autoapproach", "approachtip", "tryengage",
})

#: 名字里含这些子串的技能 = **正在故意扰动针尖**（扎、脉冲、成形）。
#:
#: 与 ``service.SUPPRESS_SKILL_PATTERNS`` **刻意不是同一张表**：那张表里有进针
#: （``autoapproach`` / ``approachtip`` / ``tryengage``），而进针正是这条判据唯一
#: 要判的时刻。照抄那张表等于让这条规则**在它唯一能发挥作用的场合永久静默** ——
#: 一道看着在防护、其实把自己关掉了的闸门。
DISTURB_SKILL_PATTERNS: frozenset[str] = frozenset({
    "tipshape", "tippulse", "conditiontip", "shapetip", "biaspulse",
    "preparenobletip", "resolutiontip", "biaswiggle", "poke", "plunge",
    "pulseprobebias",
})

# ── 通道目录 ────────────────────────────────────────────────────────────────

#: 振幅通道的名字线索。**与 ``skills/builtins/qplus_amplitude.py`` 的
#: ``_AMP_HINTS`` / ``_OSC_HINTS`` 逐字相同**，各写一份是为了不让 monitoring 依赖
#: skills 层（那会把技能注册表拖进这个必须能独立 import 的包）。
#: ``test_aux.py::test_amp_hints_match_qplus_skill`` 钉住两者相等 ——
#: 与 ``commission._SAT_SHIPPED_DEFAULT`` 同一手法。
_AMP_HINTS: tuple[str, ...] = ("amplitude", "amp")
_OSC_HINTS: tuple[str, ...] = ("oc ", "ocd", "osc", "pll", "excitation")


@dataclass(frozen=True)
class AuxChannelSpec:
    """一路辅助通道的静态描述。

    ``judged=False`` 的通道**只记录，不判级** —— 记录永远安全，而编造一个没有基线的
    阈值不是。
    """

    kind: str
    label_zh: str
    unit: str
    judged: bool
    hints: tuple[str, ...]
    #: 额外的「还必须含其中之一」词组。振幅要靠它把 "OC D1 Amplitude" 和
    #: 任何恰好含 "amp" 的名字（"Sample …"）分开。
    co_hints: tuple[str, ...] = ()
    #: instrument_profile 里的索引覆写键（用户可以指定名字匹配不到的通道）。
    override_key: str = ""


CHANNEL_SPECS: tuple[AuxChannelSpec, ...] = (
    AuxChannelSpec(
        kind="z", label_zh="Z 位置", unit="m", judged=True,
        hints=("z (m)", "z(m)", "z position", "z_pos"),
    ),
    AuxChannelSpec(
        kind="amplitude", label_zh="qPlus 振幅", unit="m", judged=True,
        hints=_AMP_HINTS, co_hints=_OSC_HINTS,
        override_key="qplus_amplitude_signal_index",
    ),
    AuxChannelSpec(
        kind="df", label_zh="频率偏移", unit="Hz", judged=False,
        hints=("freq. shift", "freq shift", "frequency shift"),
    ),

    AuxChannelSpec(
        kind="lockin", label_zh="dI/dV (lock-in)", unit="A", judged=False,
        hints=("li demod", "lockin", "lock-in", "demod"),
        override_key="lockin_signal_index",
    ),
    # 偏压与辅助通道同次读取并共享时间戳及 segment_id，保留变化过程。偏压为用户设置量，仅记录，不用告警阈值评价实验选择。
    AuxChannelSpec(
        kind="bias", label_zh="偏压", unit="V", judged=False,
        hints=("bias (v)", "bias(v)", "bias"),
        override_key="bias_signal_index",
    ),
)

#: 判据名。刻意**不并进** ``alerts.WARN_RULES`` —— 那张表是电流的，
#: 两套判据各自独立是本次的硬性要求。告警行落进同一张 ``alerts`` 表（rule 是字符串），
#: 所以既有的告警历史 UI 免费显示它们。
#:
#: ``amp_zero_on_approach`` 取代了原来的 ``amp_collapse``：**名字换掉是故意的**。
#: 语义变了（全程判塌陷 → 只在进针期间判归零），而告警历史里旧的 ``amp_collapse``
#: 行是另一套逻辑产出的。沿用同一个名字会让两批含义不同的行看起来是同一件事。
#: ``amp_unstable`` 整条删除，理由见模块 docstring。
AUX_WARN_RULES: tuple[str, ...] = (
    "z_drift_high", "z_step", "z_rail", "amp_zero_on_approach",
)

#: 调用方传 ``suppressed=True`` 时**仍然要报**的规则。
#:
#: ``service.SUPPRESS_SKILL_PATTERNS`` 里含 ``autoapproach`` / ``approachtip`` /
#: ``tryengage`` —— 而进针正是 ``amp_zero_on_approach`` **唯一**要判的时刻。
#: 让它跟着一起被抑制，等于让这条判据在它唯一能发挥作用的场合永久静默：
#: 一道看着在防护、其实把自己关掉了的闸门。
#:
#: 这不是「绕过抑制」，是「它有自己更贴切的闸门」：进针闸只在进针期间开，
#: 扰动屏蔽专管扎针/脉冲那一族 —— 后者才是那张技能表在这条通道上真正想挡的东西。
SUPPRESSION_EXEMPT_RULES: frozenset[str] = frozenset({"amp_zero_on_approach"})

#: 落进 ``aux_samples`` 表的窗口特征列（顺序即写入顺序）。
#:
#: 振幅那一组现在**把闸门本身也记下来**（``amp_gate_open`` / ``amp_blanked`` /
#: ``amp_ring_age_s``）。不记的话，事后看到一行「振幅 0.4 pm 但没告警」根本分不出
#: 是「没在进针」「刚打过脉冲」还是「判据坏了」—— 而这三者要采取的行动完全不同。
AUX_METRIC_COLUMNS: tuple[str, ...] = (
    "z_m", "z_mean_m", "z_drift_m_per_s", "z_span_m",
    "z_step_m", "z_step_count", "z_step_robust", "z_headroom_frac",
    # 分别记录向进针端和退针端的剩余行程，保留米数与比例。符号未知时有向量保持 NULL，不猜方向。
    "z_headroom_retract_m", "z_headroom_extend_m",
    "z_headroom_retract_frac", "z_headroom_extend_frac",
    "z_drift_half1_m_per_s", "z_drift_half2_m_per_s", "z_drift_decaying",
    "junction_age_s",
    "amp_m", "amp_mean_m", "amp_rel_sd",
    "amp_frac_of_baseline", "amp_min_frac_of_baseline", "amp_drop_frac",
    "amp_zero", "amp_zero_hold_s", "amp_ring_age_s", "amp_off",
    "amp_gate_open", "amp_blanked",
    # 音叉**在不在被驱动**（缺陷⑰ 的另一半）。三态：1 = 在振、0 = 没被驱动、
    # NULL = **没问过或没读到**。与 ``lockin_mod_on`` 同一条理由，也同一个坑：
    # 塌成两态会把「不知道」写成「关着」，而那是一句关于仪器状态的假陈述。
    # 这里塌错的后果更直接 —— 0 会被读成「已确认没被驱动」，于是一行本该
    # 「判不了」的样本读起来像「已经判过了，没事」。
    "amp_excited",
    "df_hz", "df_drift_hz_per_s", "df_span_hz",
    # lock-in：只有「读数 / 均值 / 展布」三个，**刻意不加漂移率**。斜率对一个会
    # 穿零的带符号投影没有物理含义 —— 相位转 180° 会给出一个漂亮的、完全虚假的
    # 线性趋势。后缀必须是 ``_a``：commission._fmt 先判 ``_a``（→ pA），
    # 一个 ``lockin_drift_a_per_s`` 会落进「没人认领」的分支按裸数字回显。
    "lockin_a", "lockin_mean_a", "lockin_span_a",
    # 调制**开没开**。需求是：辅助通道 didv 应该以某种方式标出 lockin
    # 开没开 —— 而这不是个显示问题，是**这条曲线是什么**的问题：调制关着的时候
    # 解调器输出的是噪声与串扰底，不是 dI/dV。同一条曲线、同一个量纲、同一个数量级，
    # 而两种读法之间隔着「这是测量」和「这不是测量」。
    #
    # 与 ``amp_gate_open`` / ``amp_blanked`` 同一条理由：**闸门本身也要落库**。
    # 事后看到一行 lockin_a 却分不出它是不是测量值，这一行就等于没有 —— 而分不出
    # 的那一刻，最省事的读法永远是「它是 dI/dV」。
    #
    # ⚠️ 三态，不是两态：1=确认开、0=确认关、**NULL=没读到**。第三态不能塌进 0，
    # 见 ``core/state.py`` 那句「we did not read it 必须不能变成 modulation is off」。
    "lockin_mod_on",
    # 偏压。``bias_changed`` 是**三态**：1 = 这个窗口里改过、0 = 没改过、
    # NULL = 样本不够判不了 —— 与 ``amp_excited`` / ``lockin_mod_on`` 同一条
    # 理由：塌成两态会把「不知道」写成「没改过」，而演示回放时那正好是最要命的
    # 一句假话（找不到用户动手的那一刻）。
    "bias_v", "bias_mean_v", "bias_span_v", "bias_step_v", "bias_changed",
)


# ── 小工具 ──────────────────────────────────────────────────────────────────


def _decoded(rv: Any) -> list:
    """拆开 nanonis_spm 的 ``(header, raw, fields)``。同 pump._decoded。"""
    if isinstance(rv, tuple) and len(rv) >= 3 and isinstance(rv[2], list):
        return rv[2]
    if isinstance(rv, list):
        return rv
    return []


def _scalar(v) -> Optional[float]:
    """解码出的标量偶尔裹在 1-tuple 里。

    转发到共享原语 :func:`mast.io.nanonis_files.scalar_float`(v6.1.3 合并)——
    这份实现与它**语义完全一致**(拿不到就 ``None``,不编 0.0),所以只是去重。
    ⚠️ ``pump.py`` / ``zburst.py`` 里那两份**不一致**(空序列回 ``0.0``),
    没有一起并过来:那是一次行为改变,单独判,见 KNOWN_ISSUES。
    """
    from mast.io.nanonis_files import scalar_float
    return scalar_float(v)


def _values_array(d: list) -> list[float]:
    """从 ``Signals_ValsGet`` 的解码体里取出那串浮点。

    回包规格是 ``["i", "*f"]`` —— 先一个长度，再一个数组。库里的例子
    (``Example_LockInSweep``) 显示数组元素还可能各自裹一层，所以逐个 ``_scalar``。
    """
    for field_val in d:
        if isinstance(field_val, (list, tuple)) and field_val:
            out = [_scalar(x) for x in field_val]
            if all(v is not None for v in out):
                return [float(v) for v in out]  # type: ignore[arg-type]
    return []


def _match_channel(names: Sequence[str], spec: AuxChannelSpec) -> tuple[int, str]:
    """按名字线索找一路信号。找不到返回 ``(-1, "")``（不是故障，是「这台机器没有」）。"""
    for hint in spec.hints:
        for i, nm in enumerate(names):
            low = " ".join(str(nm).lower().split())
            if hint not in low:
                continue
            if spec.co_hints and not any(c in low for c in spec.co_hints):
                continue
            return i, str(nm)
    return -1, ""


def _profile_number(key: str) -> Optional[float]:
    """读 instrument_profile 的一个数；读不到返回 None。永不抛。

    **不新开真源。** 振幅基线就是 ``ReadTipOscillationAmplitude(set_baseline=True)``
    写进 profile 的那一个；f₀/Q 就是 ``AcquirePLLFreqSweep`` 写的那两个。
    """
    try:
        from mast.core.instrument_profile import get_config
        v = get_config(key, None)
        if isinstance(v, bool) or v is None:
            return None
        return float(v)
    except Exception:  # noqa: BLE001 — profile 缺失只影响判据，绝不影响采样
        logger.debug("instrument_profile[%s] unreadable", key, exc_info=True)
        return None


def _retract_sign() -> Optional[int]:
    """Z 读数增大是否表示远离样品：+1 是、-1 否、None 未知。
    
    退针方向取显式配置的伸长方向之反号。必须使用
    z_extend_sign_or_none，不能把出厂默认当作已声明的接线事实。
    未配置时有向余量也保持 None，避免把退针与进针互换。
    """
    try:
        from mast.core.instrument_profile import z_extend_sign_or_none
        s = z_extend_sign_or_none()
    except Exception:  # noqa: BLE001 — 档案缺失只影响这两个量
        logger.debug("z_extend_sign unreadable", exc_info=True)
        return None
    return -s if s in (1, -1) else None


def amplitude_tau_s() -> Optional[float]:
    """qPlus 振幅的 1/e 弛豫时间 ``τ = Q / (π f₀)``，算不出返回 None。

    这是「1 Hz 够不够采振幅」这个问题的**可计算答案**，而不是一句断言：
    读的是 ``AcquirePLLFreqSweep`` 实测写回的 f₀/Q。没扫过就没有 τ，
    那就如实报 None，不拿标称值冒充实测。
    """
    f0 = _profile_number("qplus_f0_measured_hz")
    q = _profile_number("qplus_q_measured")
    if not f0 or not q or f0 <= 0 or q <= 0:
        return None
    return float(q / (math.pi * f0))


# ── 窗口特征（纯函数，无硬件、无 store、无配置） ──────────────────────────


def _finite_pairs(ts: Iterable[float], vals: Iterable[float]):
    import numpy as np
    t = np.asarray(list(ts), dtype=np.float64)
    y = np.asarray(list(vals), dtype=np.float64)
    n = min(t.size, y.size)
    t, y = t[:n], y[:n]
    m = np.isfinite(t) & np.isfinite(y)
    return t[m], y[m]


def _ols_slope(t, y) -> Optional[float]:
    """对**真实时间戳**做最小二乘，不是对样本序号。

    采样是机会式的（角色忙就跳过），拿序号当时间会把漂移率算错一个与跳过率相关的
    倍数 —— 而且是静默地错。
    """
    import numpy as np
    if t.size < 3:
        return None
    span = float(t[-1] - t[0])
    if span <= 0:
        return None
    slope, _ = np.polyfit(t - t[0], y, 1)
    return float(slope)


def _nominal_dt(t) -> float:
    """观测到的相邻间隔中位数。自标定，比配置里的标称值更贴近现实。"""
    import numpy as np
    if t.size < 2:
        return 0.0
    d = np.diff(t)
    d = d[d > 0]
    return float(np.median(d)) if d.size else 0.0


def z_window_features(ts, vals, *, jump_k: float = 8.0,
                      limits: tuple[float, float] | None = None,
                      retract_sign: int | None = None) -> dict:
    """Z 的窗口统计量。

    **实测的工作区间**（合成语料，300 s 窗口 @ 1 Hz，400 次/档，出厂阈值
    50 pm/s 与 500 pm，k=8）::

        零假设                              z_drift_high   z_step
        ─────────────────────────────────────────────────────────
        纯高斯白噪声 σ = 1…200 pm                0%          0%
        随机游走 wander ≤ 1 nm/300 s             0%          0%
        随机游走 wander = 5 nm/300 s          0.50%          0%
        随机游走 wander = 10 nm/300 s           16%          0%     ← 见下
        线性漂移 49 pm/s + 噪声                  0%          0%
        ─────────────────────────────────────────────────────────
        正例：500 pm 阶跃(叠随机游走)             —          51%    ← 恰在阈值上
        正例：1 nm 阶跃                          —         100%
        正例：60 pm/s 持续漂移                 100%           —

    「随机游走 10 nm/300 s 有 16% 会响」**不是误报** —— 那是 2 nm/min 的真实漂移，
    用户确实想知道。但它说明了一件事：这条阈值的正确值完全取决于这台机器的热态，
    所以 ``cm_aux_alerts_enabled`` 出厂是关的，而标定报告存在的意义就是把它打开。

    ``z_span_m`` **只上报，绝不做判据**：随机游走（压电蠕变/热漂的正确零假设）的
    极差按 √N 增长，任何固定阈值在足够长的窗口上都会被纯漂移触发。

    ``z_step_m`` 走**一阶差分**，这对随机游走是正确的估计量 —— 随机游走的一阶差分
    是 iid 高斯，``median(|Δ|) + k·1.4826·MAD(|Δ|)`` 正好是它的稳健阈值。
    跨采样间隙的差分先被屏蔽掉（见 ``_STEP_DT_TOLERANCE``）。

    **σ 退化的三条分支**（每一条都是实测逼出来的）：

    * 差分**全为 0** —— 信号在窗口里逐位不变（Z 反馈关着、Z 停在原地）。
      没有台阶，报 0。
    * MAD 为 0 但有非零差分 —— 量化比噪声粗（Z 的读数台阶大于它的抖动），
      或者一条干净得没有噪声的阶跃。此时**没有噪声背景可比**，稳健阈值塌成 0，
      再拿 k 去乘只会把一半差分判成台阶。**退回绝对判据**：报出最大的那一次位移，
      由用户的 ``cm_z_step_warn_m``（绝对米数）去卡。
      ``z_step_robust=0`` 如实标出这一次用的不是统计阈值。
      —— 早先这一支报 ``None``，结果是**一条干净的 2 nm 阶跃被判成「判不了」**，
      也就是把想抓的正例整个漏掉；实测发现（probe E）。
    * 正常 —— 统计阈值，``z_step_robust=1``。

    ── 余量：一个双侧量 + 两个**有向**量（2026-08-10） ──────────────────────

    ``limits`` 必须是 :func:`mast.core.envelope_reconcile.resolve_z_travel` 解析出来
    的行程区间，**不是** ``ZCtrl_LimitsGet`` 的原始回包。软限值在
    ``z_limits_enabled = 0`` 时不起任何作用，对着它算出来的百分比是一对惰性数字的
    百分比 —— 而它和一个算对的百分比长得一模一样。

    ``z_headroom_frac`` 是 ``min(到上轨, 到下轨)``，**双侧**。它能说「快到头了」，
    说不出「往哪边」——而这两件事要的处置正好相反：

        样品变近 ⇒ Z 压向退针端 ⇒ 粗动**退**
        样品变远 ⇒ Z 压向进针端 ⇒ 粗动**进**

    所以另给两个有向量。``retract_sign`` = +1 表示「Z 变大 = 远离样品」
    （即 ``-get_z_extend_sign()``）。``None`` = 不知道这台机器的 Z 符号约定 ⇒
    两个有向量都是 ``None``，**不猜**：猜错的代价是把「该退」说成「该进」。

    双侧量**保留不动** —— 它是历史数据的列，删掉会让旧行不可比，而且
    ``retract_sign`` 读不到时它仍然算得出来。两者不是同一个量的两种写法。
    """
    import numpy as np
    out: dict[str, Optional[float]] = {
        "z_m": None, "z_mean_m": None, "z_drift_m_per_s": None,
        "z_span_m": None, "z_step_m": None, "z_step_count": None,
        "z_step_robust": None, "z_headroom_frac": None,
        "z_headroom_retract_m": None, "z_headroom_extend_m": None,
        "z_headroom_retract_frac": None, "z_headroom_extend_frac": None,
        "z_drift_half1_m_per_s": None, "z_drift_half2_m_per_s": None,
        "z_drift_decaying": None,
    }
    t, y = _finite_pairs(ts, vals)
    if y.size == 0:
        return out
    out["z_m"] = float(y[-1])

    if limits is not None:
        lo, hi = float(min(limits)), float(max(limits))
        travel = hi - lo
        if travel > 0:
            z = float(y[-1])
            to_hi, to_lo = max(0.0, hi - z), max(0.0, z - lo)
            out["z_headroom_frac"] = float(min(to_hi, to_lo) / travel)
            if retract_sign is not None and retract_sign != 0:
                # +1: 退针端 = 上轨；-1: 退针端 = 下轨。
                retract_m, extend_m = ((to_hi, to_lo) if retract_sign > 0
                                       else (to_lo, to_hi))
                out["z_headroom_retract_m"] = float(retract_m)
                out["z_headroom_extend_m"] = float(extend_m)
                out["z_headroom_retract_frac"] = float(retract_m / travel)
                out["z_headroom_extend_frac"] = float(extend_m / travel)

    if y.size < _MIN_WINDOW_N:
        return out

    out["z_mean_m"] = float(y.mean())
    out["z_drift_m_per_s"] = _ols_slope(t, y)
    out["z_span_m"] = float(np.percentile(y, 95) - np.percentile(y, 5))
    out.update(_drift_decay(t, y))

    nominal = _nominal_dt(t)
    dt = np.diff(t)
    dy = np.abs(np.diff(y))
    ok = (dt > 0) & (dt <= (nominal * _STEP_DT_TOLERANCE if nominal > 0 else np.inf))
    d = dy[ok]
    if d.size >= _MIN_WINDOW_N:
        if not np.any(d > 0):
            # 信号在窗口里逐位不变（Z 反馈关着、Z 停在原地）。没有台阶，就是 0。
            out["z_step_m"], out["z_step_count"] = 0.0, 0
            out["z_step_robust"] = 1.0
        else:
            med = float(np.median(d))
            sigma = 1.4826 * float(np.median(np.abs(d - med)))
            if sigma > 0:
                over = d[d > med + jump_k * sigma]
                out["z_step_count"] = int(over.size)
                out["z_step_m"] = float(over.max()) if over.size else 0.0
                out["z_step_robust"] = 1.0
            else:
                # 尺度退化：量化比噪声粗，或者阶跃干净到没有背景。统计阈值在这里
                # 塌成 0，所以退回**绝对判据** —— 报出最大的一次位移，由用户的
                # 绝对阈值去卡。见 docstring 的三条分支。
                top = float(d.max())
                out["z_step_m"] = top
                out["z_step_count"] = int(np.count_nonzero(d >= top))
                out["z_step_robust"] = 0.0
    return out


def _drift_decay(t, y) -> dict:
    """把窗口分成两半，分别拟合 Z 斜率并判断漂移是否衰减。
    
    衰减瞬态与持续漂移需要不同解释；使用半窗斜率比判断趋势，
    不从缺失时段外推整定时间。结果只是趋势特征，不是物理归因。
    """
    n = t.size
    if n < 2 * _MIN_WINDOW_N:
        return {}
    mid = n // 2
    s1 = _ols_slope(t[:mid], y[:mid])
    s2 = _ols_slope(t[mid:], y[mid:])
    if s1 is None or s2 is None:
        return {}
    out = {"z_drift_half1_m_per_s": s1, "z_drift_half2_m_per_s": s2}
    a1, a2 = abs(s1), abs(s2)
    if a1 <= 0:
        # 前半没在漂，后半再怎么样也不叫「衰减」。
        out["z_drift_decaying"] = 0.0
    else:
        out["z_drift_decaying"] = 1.0 if a2 < _DECAY_RATIO * a1 else 0.0
    return out


def amp_window_features(ts, vals, *, baseline: float | None = None,
                        zero_frac: float = 0.30,
                        hold_s: float = _AMP_ZERO_HOLD_S,
                        ring_frac: float = _AMP_RING_FRAC) -> dict:
    """qPlus 振幅的纯窗口统计；这里只判断持续低振幅，不直接判撞针。
    
    amp_zero 要求末端连续 hold_s 秒低于 zero_frac × 基线；孤立高点
    按实现规则处理，避免短暂扰动不断重置连续段。持续时间按时间戳计算，
    不能用样本数代替。缺基线或有效点不足须明确返回未知。
    进针状态、动作屏蔽和激励状态由有状态闸门提供，最后在 evaluate_aux 汇合。
    """
    import numpy as np
    out: dict[str, Optional[float]] = {
        "amp_m": None, "amp_mean_m": None, "amp_rel_sd": None,
        "amp_frac_of_baseline": None, "amp_min_frac_of_baseline": None,
        "amp_drop_frac": None, "amp_zero": None, "amp_zero_hold_s": None,
        "amp_ring_age_s": None, "amp_off": None,
    }
    t, y = _finite_pairs(ts, vals)
    if y.size == 0:
        return out
    last = float(y[-1])
    out["amp_m"] = last

    base = float(baseline) if (baseline is not None and baseline > 0) else None
    if base is not None:
        out["amp_frac_of_baseline"] = last / base

    if y.size < _MIN_WINDOW_N:
        return out

    mean = float(y.mean())
    out["amp_mean_m"] = mean
    if abs(mean) > 0:

        out["amp_rel_sd"] = float(y.std() / abs(mean))

    med = float(np.median(y))
    if med > 0:
        out["amp_drop_frac"] = float(max(0.0, 1.0 - last / med))

    if base is None:
        return out

    floor = float(zero_frac) * base
    out["amp_min_frac_of_baseline"] = float(y.min()) / base
    # 窗口里从来没有非零读数。最可能是通道关着/没接 —— 但也可能是针尖整个窗口都
    # 还扎着。两者分不出来，所以这是一句「判不了」，不是「没撞」。
    out["amp_off"] = 0.0 if float(y.max()) >= floor else 1.0

    below = y < floor
    if not bool(below[-1]):
        out["amp_zero"], out["amp_zero_hold_s"] = 0.0, 0.0
    else:
        # 从最后一个样本往回数，找到这一段连续「在零上」的起点。
        #
        # **跨采样间隙的地方要断开。** 采样是机会式的（角色忙就跳过），两个相隔
        # 60 s 的样本之间我们**什么都没看到** —— 把那 60 s 算进「已经持续多久」
        # 就是拿没有观测的时间去凑一个「持续」的结论。同 z_step 屏蔽跨间隙差分，
        # 同 features.py「跨 run 的一阶差分会凭空造出台阶」。
        # 断开的代价只是晚一拍报（间隙之后重新攒 hold_s），方向是对的。
        nominal = _nominal_dt(t)
        max_dt = nominal * _STEP_DT_TOLERANCE if nominal > 0 else float("inf")
        i = int(y.size) - 1
        while i > 0:
            if (t[i] - t[i - 1]) > max_dt:
                break                       # 采样间隙：那段时间我们什么都没看到
            if bool(below[i - 1]):
                i -= 1
                continue

            if (i - 2 >= 0 and bool(below[i - 2])
                    and (t[i - 1] - t[i - 2]) <= max_dt):
                i -= 2
                continue

            break
        held = float(t[-1] - t[i])
        out["amp_zero_hold_s"] = held
        # 起点在窗口第一个样本上时，我们只知道「至少这么久」——如果它凑不满 hold_s
        # 就还是判不满足，宁可晚一拍。
        out["amp_zero"] = 1.0 if held >= float(hold_s) else 0.0

    ring = y >= (float(ring_frac) * base)
    if bool(ring.any()):
        out["amp_ring_age_s"] = float(t[-1] - t[np.nonzero(ring)[0][-1]])
    return out


def lockin_window_features(ts, vals) -> dict:
    """dI/dV lock-in 只记录，不判级。
    
    X/Y 是带符号投影，相位转动可能使其穿零，不能直接套用振幅塌缩判据。
    lockin_span_a 采用 5–95 百分位差，避免孤立尖峰主导整个窗口的范围。
    """
    out: dict[str, Optional[float]] = {
        "lockin_a": None, "lockin_mean_a": None, "lockin_span_a": None}
    import numpy as np
    t, y = _finite_pairs(ts, vals)
    if y.size == 0:
        return out
    out["lockin_a"] = float(y[-1])
    if y.size >= _MIN_WINDOW_N:
        out["lockin_mean_a"] = float(np.mean(y))
        out["lockin_span_a"] = float(np.percentile(y, 95) - np.percentile(y, 5))
    return out


def df_window_features(ts, vals) -> dict:
    """频率偏移 —— **只记录，不判级**。

    没有这台机器的 Δf 基线，就不编 Δf 的阈值。这与 envhistory 对 Z 噪声谱的处理
    是同一条纪律：把没验过误报率的判据放上去，赌输的形式是半夜一条假告警。
    """
    out: dict[str, Optional[float]] = {
        "df_hz": None, "df_drift_hz_per_s": None, "df_span_hz": None}
    import numpy as np
    t, y = _finite_pairs(ts, vals)
    if y.size == 0:
        return out
    out["df_hz"] = float(y[-1])
    if y.size >= _MIN_WINDOW_N:
        out["df_drift_hz_per_s"] = _ols_slope(t, y)
        out["df_span_hz"] = float(np.percentile(y, 95) - np.percentile(y, 5))
    return out



_BIAS_CHANGE_REL = 5e-4
#: 相对判据在 0 V 附近失效,所以再加一个绝对下限(1 mV)。两者取或。
_BIAS_CHANGE_ABS_V = 1e-3


def bias_window_features(ts, vals) -> dict:
    """偏压 —— **只记录，不判级**。

    偏压是用户设定的量，不是仪器状态的读数：它取什么值是实验意图，不是故障。
    所以这里一个阈值都不编，只回答三个问题 ——
    **现在是多少 / 这个窗口里改过没有 / 改了多大**。

    ``bias_step_v`` 取窗口内**最大的一次相邻跳变**（带符号），不是首尾之差：
    演示里常见「改上去再改回来」，首尾之差会把那种操作读成「什么也没发生」。
    """
    out: dict[str, Optional[float]] = {
        "bias_v": None, "bias_mean_v": None, "bias_span_v": None,
        "bias_step_v": None, "bias_changed": None}
    import numpy as np
    t, y = _finite_pairs(ts, vals)
    if y.size == 0:
        return out
    out["bias_v"] = float(y[-1])
    if y.size < 2:
        return out                      # 一个点看不出改没改 —— 留 NULL，不写 0
    out["bias_mean_v"] = float(y.mean())
    out["bias_span_v"] = float(y.max() - y.min())
    d = np.diff(y)
    k = int(np.argmax(np.abs(d)))
    step = float(d[k])
    out["bias_step_v"] = step
    scale = max(abs(float(y[k])), abs(float(y[k + 1])))
    out["bias_changed"] = 1.0 if (abs(step) > _BIAS_CHANGE_ABS_V
                                  or abs(step) > _BIAS_CHANGE_REL * scale) else 0.0
    return out


# ── 进针闸门 / 扰动屏蔽 ─────────────────────────────────────────────────────


def _num(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _skill_matches(ctx: dict, patterns: frozenset[str]) -> bool:
    skill = str(ctx.get("ctx_skill") or "").lower()
    return bool(skill) and any(p in skill for p in patterns)


def no_junction(ctx: dict) -> bool:
    """**确证**当前没有隧道结（不是「不知道有没有」）。

    只有在电流与设定点都读得到、快照不陈旧、而且 |I| < 20% 设定点时才说 True。
    读不到就说 False —— 这句话是用来**打开**一道闸门的，说不出口就不说。

    与 ``approach.py::_engagement_bar`` 的 50% 故意不相等、方向也相反：
    那条是「敢不敢说已经进针成功」，这条是「敢不敢说肯定还没有结」。
    20%-50% 之间两句话都不说 —— 那段死区是刻意的。
    """
    if ctx.get("ctx_stale") is True:
        return False
    i = _num(ctx.get("ctx_current_a"))
    sp = _num(ctx.get("ctx_setpoint_a"))
    if i is None or sp is None or abs(sp) <= 0:
        return False
    return abs(i) < _NO_JUNCTION_FRAC * abs(sp)


class _ApproachGate:
    """有状态的进针证据与动作屏蔽闸门。
    
    进针证据包括：匹配的技能令牌、Z 反馈开启且确认无结、必要时惰性查询
    AutoApproach_OnOffGet。任一证据可更新进针闩；硬件查询有最小间隔，
    只在振幅特征需要解释而其他证据不足时触发。
    最近的扰动动作决定屏蔽期；起振事件留作观测记录，不单独触发屏蔽。
    """

    def __init__(self) -> None:
        self._last_approach: Optional[float] = None
        self._last_disturb: Optional[float] = None
        self._last_probe = -math.inf

    def reset(self) -> None:
        self._last_approach = None
        self._last_disturb = None
        self._last_probe = -math.inf

    def observe(self, ctx: dict, metrics: dict, now: float, th,
                probe: Callable[[], Optional[bool]] | None = None) -> dict:
        """吃一拍上下文 + 窗口特征，吐出两个闸门位。``probe`` 可以为 None。"""
        if _skill_matches(ctx, APPROACH_SKILL_PATTERNS):
            self._last_approach = now
        if _skill_matches(ctx, DISTURB_SKILL_PATTERNS):
            self._last_disturb = now
        if ctx.get("ctx_zctrl_on") is True and no_junction(ctx):
            self._last_approach = now

        blank_s = max(0.0, float(getattr(th, "cm_amp_blank_s", 0.0)))

        blanked = bool(self._last_disturb is not None
                       and now - self._last_disturb < blank_s)

        if (not blanked and not self._latched(now)
                and probe is not None
                and (_num(metrics.get("amp_zero")) or 0.0) >= 1
                and now - self._last_probe >= _APPROACH_PROBE_MIN_INTERVAL_S):
            # 惰性硬件问句。到这里说明：振幅确实在零上、没在屏蔽期、而技能与物理
            # 证据都没能确认进针 —— 这正是「问一句值不值得」的答案为「值得」的时刻。
            self._last_probe = now
            if probe() is True:
                self._last_approach = now

        return {"amp_gate_open": 1.0 if self._latched(now) else 0.0,
                "amp_blanked": 1.0 if blanked else 0.0}

    def _latched(self, now: float) -> bool:
        return (self._last_approach is not None
                and now - self._last_approach < _APPROACH_LATCH_S)


# ── 判据 ────────────────────────────────────────────────────────────────────


def evaluate_aux(metrics: dict, ctx: dict, th) -> tuple[list[str], dict]:
    """判一次辅助通道。纯函数 —— 无状态、无 I/O，可以直接测。

    **上下文闸门与电流那边故意不同。** ``AlertEngine.evaluate`` 对未知上下文取宽松
    （「没有 InstrumentState 的装机上不能把监控整个废掉」）；Z 的两条位移规则反过来
    取严格，因为**这两个数的物理含义本身就取决于上下文**：

    * 扫描时 Z 就是在跟着形貌走 —— 一条有台阶的表面每行都跳 250 pm。
      未知即放行 = 扫描期间连续误报。
    * 退针态的定义就是「Z 停在最大值」。不给 ``z_rail`` 加 Z 反馈闸门的话，
      每一次退针都报一次「Z 顶到量程」。

    Fail-closed：``None``（算不出来）永远不触发任何规则。
    """
    rules: list[str] = []
    detail: dict = {}

    scanning = ctx.get("ctx_scanning")
    zctrl = ctx.get("ctx_zctrl_on")
    #: 位移规则要求「明确不在扫描」。None（没有快照）不放行 —— 见 docstring。
    z_motion_ok = scanning is False

    drift = _num(metrics.get("z_drift_m_per_s"))

    decaying = _num(metrics.get("z_drift_decaying")) or 0.0
    if (z_motion_ok and drift is not None and decaying < 1
            and abs(drift) > th.cm_z_drift_warn_m_per_s):
        rules.append("z_drift_high")
        detail["z_drift_m_per_s"] = drift
        detail["z_drift_half1_m_per_s"] = _num(metrics.get("z_drift_half1_m_per_s"))
        detail["z_drift_half2_m_per_s"] = _num(metrics.get("z_drift_half2_m_per_s"))

    step = _num(metrics.get("z_step_m"))
    if z_motion_ok and step is not None and step > th.cm_z_step_warn_m:
        rules.append("z_step")
        detail["z_step_m"] = step
        detail["z_step_count"] = _num(metrics.get("z_step_count"))

    # 双侧余量决定触发时机，有向余量决定报告哪一端接近极限。符号未知时报告方向未知，不填默认端。
    head = _num(metrics.get("z_headroom_frac"))
    if zctrl is True and head is not None and head < th.cm_z_headroom_warn_frac:
        rules.append("z_rail")
        detail["z_headroom_frac"] = head
        ret = _num(metrics.get("z_headroom_retract_frac"))
        ext = _num(metrics.get("z_headroom_extend_frac"))
        detail["z_headroom_retract_frac"] = ret
        detail["z_headroom_extend_frac"] = ext
        if ret is not None and ext is not None:
            detail["z_rail_side"] = "retract" if ret <= ext else "extend"


    zero = _num(metrics.get("amp_zero"))
    gate = _num(metrics.get("amp_gate_open"))
    blanked = _num(metrics.get("amp_blanked"))
    amp_off = _num(metrics.get("amp_off"))
    excited = _num(metrics.get("amp_excited"))
    if (zero is not None and zero >= 1
            and gate is not None and gate >= 1
            and (blanked is None or blanked < 1)
            and (amp_off is None or amp_off < 1)
            and excited is not None and excited >= 1):
        rules.append("amp_zero_on_approach")
        detail["amp_frac_of_baseline"] = _num(metrics.get("amp_frac_of_baseline"))
        detail["amp_m"] = _num(metrics.get("amp_m"))
        detail["amp_zero_hold_s"] = _num(metrics.get("amp_zero_hold_s"))

    return rules, detail


def summarize_aux_zh(rule: str, metrics: dict, detail: dict | None = None) -> str:
    """一句中文：测到了什么，它意味着什么。"""
    d = detail or {}

    def g(key: str) -> float:
        return _num(d.get(key)) or _num(metrics.get(key)) or 0.0

    if rule == "z_drift_high":
        v = g("z_drift_m_per_s")
        return (f"Z 位置持续漂移({v * 1e12:.1f} pm/s,约 {v * 1e9 * 60:.2f} nm/min)"
                "**且没有在衰减**——刚落针的热弛豫会自己衰减下去,这个不会。"
                "检查温度稳定性、样品/针尖是否夹紧;长时间扫描会拉花。")
    if rule == "z_step":
        v = g("z_step_m")
        n = int(g("z_step_count"))
        return (f"Z 出现突跳(最大 {v * 1e12:.0f} pm,窗口内 {n} 次)"
                "——针尖状态突变、原子台阶或反馈回路被扰动。")
    if rule == "z_rail":
        # **方向不走 g()**：``g`` 取不到就给 0.0,那对一个「往哪边」的问题
        # 是个能读、看着正常、而且指向某一端的假答案。
        v = g("z_headroom_frac")
        side = d.get("z_rail_side") or metrics.get("z_rail_side")
        head = f"Z 已逼近压电量程边缘(余量仅 {v * 100:.1f}%)"
        if side == "retract":
            return (f"{head},正压向**退针端**——样品在变近,"
                    "继续下去反馈顶死。处置:粗动**退**(远离样品)。")
        if side == "extend":
            return (f"{head},正压向**进针端**——样品在变远,"
                    "继续下去反馈顶死。处置:粗动**进**(趋向样品)。")
        # 三态的第三态。**一条方向盲的告警比没有告警更坏**:它读起来像
        # 「快顶死了,粗动一下」,而两个方向的处置正好相反,猜错就是往样品里送。
        # 所以这里不给处置,只说清楚缺什么、去哪儿补(缺口就在感觉到它的地方说)。
        return (f"{head}——继续漂下去反馈就顶死了。**但往哪一端未知**:"
                "本机没有声明「压电伸长(趋向样品)对应 Z 读数符号」"
                "(设置 → 退针 → `z_extend_sign`)。而「该退」与「该进」"
                "的处置正好相反,**填上之前不要凭猜粗动**。")
    if rule == "amp_zero_on_approach":
        frac = g("amp_frac_of_baseline")
        held = g("amp_zero_hold_s")
        return (f"进针期间 qPlus 振幅归零(只有未接触本底的 {frac:.1%},已持续 "
                f"{held:.0f} s,且已确认 PLL 激励开着)——**针尖很可能已经扎进表面**。"
                "这条判据存在的理由就是它:针尖不导电时,扎进去也不会有电流信号,"
                "电流类判据全瞎,只有振幅会掉下去。"
                "立即停止进针并退针;退针后振幅应恢复,若不恢复那不是撞针而是"
                "振荡回路本身的问题。")
    return f"辅助通道规则 {rule} 触发。"


class _WarnDebounce:
    """同一条规则在冷却期内不重复告警。

    刻意**不复用** ``AlertEngine`` 的实例：连击状态混在一起的话，电流的饱和连击
    会被 Z 的漂移打断。共用的只有 ``cm_alert_cooldown_s`` 这个数字 ——
    它是「UI 别刷屏」的旋钮，不是物理阈值。
    """

    def __init__(self, thresholds_getter: Callable[[], Any] | None = None):
        if thresholds_getter is None:
            from mast.monitoring.thresholds import get_monitor_thresholds
            thresholds_getter = get_monitor_thresholds
        self._th = thresholds_getter
        self._last: dict[str, float] = {}

    def allow(self, rule: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        last = self._last.get(rule)
        if last is not None and now - last < self._th().cm_alert_cooldown_s:
            return False
        self._last[rule] = now
        return True

    def reset(self) -> None:
        self._last.clear()


# ── 采样器 ──────────────────────────────────────────────────────────────────


@dataclass
class _Resolved:
    spec: AuxChannelSpec
    index: int
    name: str


@dataclass
class AuxSample:
    """一次辅助通道采样的全部产出。"""

    ts: float
    segment_id: Optional[int] = None
    values: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    verdict: str = "ok"                    # ok | warn | suppressed
    rules: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


class AuxSampler:
    """Z / 振幅 / Δf 的低速旁路。**永不写示波器，永不阻塞采集。**"""

    def __init__(self, pool_getter: Callable[[], Any],
                 thresholds_getter: Callable[[], Any] | None = None):
        self._pool_getter = pool_getter
        if thresholds_getter is None:
            from mast.monitoring.thresholds import get_monitor_thresholds
            thresholds_getter = get_monitor_thresholds
        self._th = thresholds_getter
        self._channels: list[_Resolved] = []
        self._configured = False
        # -inf, not 0.0: these are compared against a caller-supplied clock, and
        # 0.0 silently means "sampled at the epoch" — which throttles away the
        # very first sample under any time base that starts near zero
        # (a monotonic clock, or a test passing explicit timestamps).
        self._last_probe = -math.inf
        self._last_sample = -math.inf
        self._bad_replies = 0

        self._deltas: deque = deque(maxlen=_CADENCE_WINDOW_N)
        #: 本次调用允许的锁等待上限。由 :meth:`maybe_sample` 的 ``budget_s`` 设，
        #: 调用一结束就复位 —— 它是**一拍的参数**，不是采样器的配置。
        self._lock_budget_s: float | None = None
        #: 「音叉在不在被驱动」的缓存（缺陷⑰）。``None`` = 还不知道 —— 与 False
        #: **不是**一回事，见 _excitation_metrics。``_excited_probe`` 是上次问的时刻，
        #: 用来防止读不到时变成问句风暴。
        self._excited: bool | None = None
        self._excited_at = -math.inf
        self._excited_probe = -math.inf

        self._prev_zctrl: bool | None = None
        self._junction_since: float | None = None
        #: ``ZCtrl_LimitsGet`` 的原始回包（软限值）与它启没启用。**留着是为了如实
        #: 上报**：快照里两者分开出去，看的人才分得出「余量按压电半程算的」和
        #: 「按软限值算的」。判据用的是下面那个解析结果，不是这一对。
        self._z_limits: tuple[float, float] | None = None
        self._z_limits_enabled: bool | None = None
        #: 解析出来的行程区间（``core.envelope_reconcile.ZTravel``）。判据的分母。
        self._z_travel = None
        #: 三条读（压电全程 / 软限值 / 软限值启没启用）齐了没有。见
        #: ``_maybe_read_z_limits``：重试的条件是这个，不是「有没有拿到区间」。
        self._z_reads_complete = False
        #: 行程读取的重试预算。见 _maybe_read_z_limits。
        self._limit_tries = 0
        self._last_limit_try = -math.inf
        self._series: dict[str, deque] = {}
        self._debounce = _WarnDebounce(thresholds_getter)
        #: 「在不在进针 / 刚才有没有人扰动过针尖」。振幅那条判据的全部闸门都在它里面。
        self._gate = _ApproachGate()
        self.detail = "尚未探测"
        self.stats = {"sampled": 0, "busy": 0, "bad_reply": 0, "probe_fail": 0}
        self._last: AuxSample | None = None

    # ── 探测 ────────────────────────────────────────────────────────────

    def _pool(self):
        pool = self._pool_getter()
        if pool is None:
            raise _Skip("尚未连接 Nanonis")
        return pool

    @property
    def _lock_timeout(self) -> float:
        """这一次调用的锁等待上限。

        搭车采样时泵只借给我们几十毫秒（见 ``pump._sleep_with_idle``），
        而默认的 0.25 s 足够让泵错过五次缓冲刷新。等待因此**必须**压进 budget ——
        这就是 ``on_idle(budget_s)`` 那个参数存在的全部理由，不是装饰。
        压不进去的那一拍按「忙」处理，跳过，与角色锁被扫描占着时同一条路径。
        """
        b = self._lock_budget_s
        if b is None:
            return _LOCK_TIMEOUT_S
        return max(_LOCK_TIMEOUT_MIN_S, min(_LOCK_TIMEOUT_S, float(b)))

    def _call(self, verb: str, *args):
        """一次只读调用。

        动词全部是**字面量**：仓库的中止策略检查 / 安全审计 / API 覆盖率普查
        都靠 grep ``safe_call("…")``，藏进变量的动词对三者都是隐形的。

        ``count_health=False``：熔断器是四个 role 共用的一个实例，
        ``record_success`` 无条件清空失败连击 —— 一个高频后台轮询器在闲置 role 上
        不断成功，会让「连续三次失败」永远攒不满，等于把全局熔断器废掉。
        """
        pool = self._pool()
        lock_to = self._lock_timeout
        if verb == "Signals_NamesGet":
            rec = pool.safe_call("Signals_NamesGet", role=DATA_ROLE,
                                 lock_timeout_s=lock_to, count_health=False)
        elif verb == "Signals_ValsGet":
            rec = pool.safe_call("Signals_ValsGet", list(args[0]), 0, role=DATA_ROLE,
                                 lock_timeout_s=lock_to, count_health=False)
        elif verb == "ZCtrl_LimitsGet":
            rec = pool.safe_call("ZCtrl_LimitsGet", role=DATA_ROLE,
                                 lock_timeout_s=lock_to, count_health=False)
        elif verb == "ZCtrl_LimitsEnabledGet":
            # 没有它，上面那对数字**不知道算不算数** —— 手册原话是未启用时
            # "has no effect"。以前不问，于是余量对着一对可能惰性的数字算。
            rec = pool.safe_call("ZCtrl_LimitsEnabledGet", role=DATA_ROLE,
                                 lock_timeout_s=lock_to, count_health=False)
        elif verb == "Piezo_RangeGet":
            rec = pool.safe_call("Piezo_RangeGet", role=DATA_ROLE,
                                 lock_timeout_s=lock_to, count_health=False)
        elif verb == "AutoApproach_OnOffGet":
            # 惰性问句，正常运行期间一次都不发 —— 见 _ApproachGate 与
            # test_the_hardware_question_is_never_asked_in_the_quiet_case。
            rec = pool.safe_call("AutoApproach_OnOffGet", role=DATA_ROLE,
                                 lock_timeout_s=lock_to, count_health=False)
        elif verb == "PLL_OutOnOffGet":
            # 同款惰性问句（缺陷⑰）：只在振幅确实归零、判据眼看要响时才发。
            rec = pool.safe_call("PLL_OutOnOffGet", int(args[0]), role=DATA_ROLE,
                                 lock_timeout_s=lock_to, count_health=False)
        elif verb == "PLL_ExcitationGet":
            rec = pool.safe_call("PLL_ExcitationGet", int(args[0]), role=DATA_ROLE,
                                 lock_timeout_s=lock_to, count_health=False)
        else:  # pragma: no cover — 防止有人加了动词却忘了分支
            raise _Skip(f"未知动词 {verb}")
        err = getattr(rec, "error", "") or ""
        if err:
            from mast.core.connection import is_lock_busy
            if is_lock_busy(rec):
                raise _Busy(err)
            if "comms_circuit_open" in err:
                raise _Skip("TCP 已熔断")
            raise _Skip(f"{verb}: {err}")
        return rec

    def _discover(self) -> None:
        """把三路信号在 128 名表里的索引找出来。**只做名字匹配，不写任何东西。**"""
        rec = self._call("Signals_NamesGet")
        names: list[str] = []
        for field_val in _decoded(rec.return_value):
            if isinstance(field_val, (list, tuple)) and field_val and isinstance(
                    field_val[0], (str, bytes)):
                names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in field_val]
                break
        if not names:
            raise _Skip("读不到信号名表")

        found: list[_Resolved] = []
        for spec in CHANNEL_SPECS:
            idx, nm = -1, ""
            if spec.override_key:
                # 用户登记过的索引优先 —— 那个键存在的目的就是覆写名字匹配。
                ov = _profile_number(spec.override_key)
                if ov is not None and 0 <= int(ov) < len(names):
                    idx = int(ov)
                    nm = names[idx]
            if idx < 0:
                idx, nm = _match_channel(names, spec)
            if idx >= 0:
                found.append(_Resolved(spec=spec, index=idx, name=nm))
        if not found:
            wanted = " / ".join(s.label_zh for s in CHANNEL_SPECS)
            raise _Skip(f"{len(names)} 路信号里没有找到 {wanted}")

        self._channels = found
        self._series = {r.spec.kind: deque(maxlen=_MAX_WINDOW_POINTS) for r in found}
        self._z_travel, self._z_reads_complete = self._read_z_travel()
        self._limit_tries = 1
        self._configured = True
        self.detail = "、".join(f"{r.spec.label_zh}=#{r.index}" for r in found)
        logger.info("辅助通道: %s（低速旁路 1 Hz，不动示波器）", self.detail)

    def _maybe_read_z_limits(self, now: float) -> None:
        """行程：探测时读一次，读不到就再试几次，然后放弃。

        不能「读一次就算」：那一次恰好撞上角色被占（扫描在抓帧）的话，
        ``z_headroom_frac`` 会**在这条连接的余生里**永远是 None，``z_rail`` 永远静默 ——
        一次瞬时忙换来一条判据的永久失效，而且没有任何东西会说这件事。

        也不能无限重试：这台机器可能根本不支持这些命令，那样就是每分钟一次白跑的
        往返。所以重试有次数上限，用完就在 ``detail`` 里说清楚。
        """
        # 重试的条件是「三条读**齐了没有**」，不是「有没有拿到一个区间」。
        # 只要拿到区间就停手的话，一次瞬时忙让 ``ZCtrl_LimitsGet`` 失败、而压电全程
        # 读到了，就会永久落在压电半程这个**更宽**的分母上 —— 软限值明明启用着且
        # 更紧，而余量算得偏大、判据偏钝。失败方向是静默，正是这一族要防的。
        if self._z_reads_complete or self._limit_tries >= _Z_LIMIT_MAX_TRIES:
            return
        if now - self._last_limit_try < _REDISCOVER_MIN_INTERVAL_S:
            return
        self._last_limit_try = now
        self._limit_tries += 1
        self._z_travel, self._z_reads_complete = self._read_z_travel()
        if self._z_travel is None and self._limit_tries >= _Z_LIMIT_MAX_TRIES:
            logger.info("辅助通道: 读不到 Z 行程（试了 %d 次），"
                        "「Z 逼近量程」这条判据保持静默", self._limit_tries)

    def _read_z_travel(self) -> "tuple[Any, bool]":
        """读取压电行程、软限值及其启用状态，返回行程区间与读取完整性。
        
        统一交给 core.envelope_reconcile.resolve_z_travel 解析。未启用的软限
        不作为边界；部分读取失败时按可用信息降级，全失败时返回 None，
        余量判据保持未知，不用猜测的量程触发告警。
        """
        from mast.core.envelope_reconcile import resolve_z_travel

        piezo_z = self._read_piezo_z_full()
        zlim = self._read_z_limits()
        enabled = self._read_z_limits_enabled()
        self._z_limits = zlim
        self._z_limits_enabled = enabled
        travel = resolve_z_travel(piezo_z_full_m=piezo_z, z_limits_m=zlim,
                                  z_limits_enabled=enabled)
        complete = (piezo_z is not None and zlim is not None
                    and enabled is not None)
        return travel, complete

    def _read_one(self, verb: str):
        """一条只读调用，任何失败都换成 None（读不到就是读不到）。"""
        try:
            return self._call(verb)
        except (_Busy, _Skip):
            return None
        except Exception:  # noqa: BLE001
            logger.debug("%s failed", verb, exc_info=True)
            return None

    def _read_piezo_z_full(self) -> Optional[float]:
        """``Piezo_RangeGet`` 的 Z 分量 —— **全程**（半程 = 全程/2 由解析器做）。"""
        rec = self._read_one("Piezo_RangeGet")
        if rec is None:
            return None
        d = _decoded(rec.return_value)
        if len(d) < 3:
            return None
        z = _scalar(d[2])
        return None if z is None or z == 0 else abs(float(z))

    def _read_z_limits(self) -> Optional[tuple[float, float]]:
        """Nanonis 的 Z 软限值 ``(lo, hi)``。**它只有启用时才算数** —— 启没启用由
        :meth:`_read_z_limits_enabled` 单独回答，这里不替它猜。"""
        rec = self._read_one("ZCtrl_LimitsGet")
        if rec is None:
            return None
        d = _decoded(rec.return_value)
        if len(d) < 2:
            return None
        hi, lo = _scalar(d[0]), _scalar(d[1])
        if hi is None or lo is None or hi == lo:
            return None
        return (float(lo), float(hi))

    def _read_z_limits_enabled(self) -> Optional[bool]:
        """软限值启没启用。**三态**：True / False / None（没问到或读不到）。

        ``None`` 与 ``False`` 在解析器里走同一条路（都不把软限值算进边界），但它们
        不是同一句话：一句是「已确认没启用」，一句是「不知道」。塌成两态之后，
        「读不到」会被记成一条关于仪器状态的假陈述。
        """
        rec = self._read_one("ZCtrl_LimitsEnabledGet")
        if rec is None:
            return None
        d = _decoded(rec.return_value)
        if not d:
            return None
        v = _scalar(d[0])
        return None if v is None else bool(v)

    # ── 采样 ────────────────────────────────────────────────────────────

    def observed_interval_s(self) -> Optional[float]:
        """实测的采样节奏（秒），点数不够就 ``None``。

        中位数，理由见 :data:`_CADENCE_WINDOW_N`。**「测不出来」和「测出来很慢」
        必须是两句话** —— 所以点数不够时返回 None，不返回设置值充数。
        """
        d = sorted(self._deltas)
        if len(d) < 3:
            return None
        m = len(d) >> 1
        return float(d[m] if len(d) % 2 else (d[m - 1] + d[m]) / 2.0)

    def maybe_sample(self, ctx: dict | None = None,
                     segment_id: int | None = None,
                     suppressed: bool = False,
                     now: float | None = None,
                     budget_s: float | None = None) -> Optional[AuxSample]:
        """机会式采一次。返回 ``None`` = 这一次跳过（没到点/忙/没通道）。**永不抛。**

        ``budget_s`` —— 调用方承诺给这一拍的时间上限（秒）。搭车在泵的空闲里采样
        时必须给：锁等待会压到它以内（见 :attr:`_lock_timeout`），压不进去就当忙、
        跳过这一拍。``None`` = 调用方有的是时间（段边界那条路），用默认 0.25 s。

        ``suppressed`` 由调用方给（service 已经算过一次）：修针 / 进针 / 谱学期间
        Z 与振幅**本来就**该剧烈变化。抑制的段照样记录 —— 那正是最有价值的语料 ——
        但不出告警，并且判级如实写成 ``suppressed`` 而不是悄悄写成 ``ok``。

        ⚠️ **一个例外**，见 :data:`SUPPRESSION_EXEMPT_RULES`：
        ``service.SUPPRESS_SKILL_PATTERNS`` 里含 ``autoapproach`` / ``approachtip`` /
        ``tryengage``，而进针正是 ``amp_zero_on_approach`` 唯一要判的时刻 ——
        让它跟着一起被抑制，等于让这条判据永远不响。
        """
        now = time.time() if now is None else now
        prev_probe = self._last_probe
        self._lock_budget_s = budget_s
        try:
            th = self._th()
            if not th.aux_enabled:
                return None
            if now - self._last_sample < max(_MIN_INTERVAL_S,
                                             float(th.cm_aux_interval_s)):
                return None
            if not self._configured:
                if now - self._last_probe < _REDISCOVER_MIN_INTERVAL_S:
                    return None
                # 先记账再探测，任何一条出口都不可能变成紧循环。
                self._last_probe = now
                self._discover()
            return self._sample(th, ctx or {}, segment_id, bool(suppressed), now)
        except _Busy:
            # 角色忙不是错误，是「扫描正拿着 data 口」这个正常事件。
            self.stats["busy"] += 1
            # 忙也算用掉了这一拍。不这样的话，把间隔调成 10 s 的用户在扫描期间
            # 会得到每次机会一次的重试 —— 每次一个锁等待，而机会现在有 20 Hz。
            self._last_sample = now
            # 但**探测不能因为一次瞬时占用被罚一分钟**：滚回去，下一拍就能再试。
            self._last_probe = prev_probe
            return None
        except _Skip as exc:
            self.stats["probe_fail"] += 1
            self._configured = False
            self.detail = str(exc)
            logger.debug("aux sample skipped: %s", exc)
            return None
        except Exception:  # noqa: BLE001 — 辅助通道绝不反噬电流采集
            self.stats["bad_reply"] += 1
            logger.debug("aux sample failed (swallowed)", exc_info=True)
            return None
        finally:
            # 一拍的参数，用完就还回去。留着的话，段边界那条本来有 0.31 s 空闲的
            # 路径会继承上一次搭车的几十毫秒预算 —— 一个只在「先搭过车」时才发生的
            # 状态泄漏，正是最难查的那一类。
            self._lock_budget_s = None

    def _sample(self, th, ctx: dict, segment_id: int | None,
                suppressed: bool, now: float) -> Optional[AuxSample]:
        rec = self._call("Signals_ValsGet", [r.index for r in self._channels])
        vals = _values_array(_decoded(rec.return_value))
        if len(vals) < len(self._channels):
            self.stats["bad_reply"] += 1
            self._bad_replies += 1
            if self._bad_replies >= _BAD_REPLY_LIMIT:
                # 一直读不懂回包，多半是这条连接上的信号表或回包结构和我们以为的
                # 不一样。重新探测一次（受 60 s 节流），而不是每段白试到天荒地老。
                self._bad_replies = 0
                self._configured = False
                self.detail = "回包结构读不懂，将重新探测"
            return None
        self._bad_replies = 0
        if math.isfinite(self._last_sample) and now > self._last_sample:
            self._deltas.append(now - self._last_sample)
        self._last_sample = now
        self.stats["sampled"] += 1
        self._maybe_read_z_limits(now)

        values: dict[str, float] = {}
        window = max(10.0, float(th.cm_aux_window_s))
        for r, v in zip(self._channels, vals):
            values[r.spec.kind] = float(v)
            dq = self._series[r.spec.kind]
            dq.append((now, float(v)))
            while dq and now - dq[0][0] > window:
                dq.popleft()

        self._track_junction(ctx, now)
        metrics = self._window_metrics(th)
        if self._junction_since is not None:
            metrics["junction_age_s"] = float(now - self._junction_since)
        # 闸门要看窗口特征（起振自检读的是 amp_ring_age_s），所以在窗口统计量之后。
        # 没有振幅通道的机器上**不算**闸门：那两列留 NULL 读作「不适用」，
        # 写成 0 会读成「没在进针」—— 而那是一句关于这台机器状态的假陈述。
        if any(r.spec.kind == "amplitude" for r in self._channels):
            metrics.update(self._gate.observe(ctx, metrics, now, th,
                                              probe=self._probe_auto_approach))
            # 「音叉在不在被驱动」（缺陷⑰ 的另一半）。放在闸门之后，因为它读
            # ``amp_zero`` 决定值不值得问；放在 evaluate_aux 之前，因为判据要用它。
            metrics.update(self._excitation_metrics(metrics, now))
        # 调制开没开 —— **零额外往返**：``core/state.py`` 每秒已经在读
        # ``LockIn_ModOnOffGet``，``service._context_labels`` 已经把它放进 ctx
        # （``ctx_lockin_on``，电流那边的 jump_burst 判据在用）。这里只是把它
        # 跟着这一拍的读数一起记下来。
        #
        # **读不到就不写这个键**（不是写 0）。列留 NULL 读作「没读到」，
        # 写 0 会读成「确认调制是关的」—— 而那是一句关于仪器状态的假陈述，
        # 会让人把一条真 dI/dV 当成噪声底扔掉。
        if any(r.spec.kind == "lockin" for r in self._channels):
            mod_on = ctx.get("ctx_lockin_on")
            if mod_on is not None:
                metrics["lockin_mod_on"] = 1.0 if mod_on else 0.0
        rules, detail = evaluate_aux(metrics, ctx, th)
        kept = [r for r in rules
                if not suppressed or r in SUPPRESSION_EXEMPT_RULES]
        verdict = "warn" if kept else ("suppressed" if suppressed else "ok")
        sample = AuxSample(ts=now, segment_id=segment_id, values=values,
                           metrics=metrics, verdict=verdict,
                           rules=kept, detail=detail)
        self._last = sample
        return sample

    def _probe_auto_approach(self) -> Optional[bool]:
        """Nanonis 的自动逼近模块在跑吗？读不到就 ``None``（判不了 ≠ 没在跑）。

        **惰性**：只有 :class:`_ApproachGate` 认为值得问的时候才会被调用 ——
        振幅已经归零、不在屏蔽期、而技能与物理证据都没能确认进针。正常运行期间
        一次都不发，有测试钉住。角色锁照样只等 0.25 s，忙就当作读不到。
        """
        try:
            rec = self._call("AutoApproach_OnOffGet")
        except (_Busy, _Skip):
            return None
        except Exception:  # noqa: BLE001 — 问不到就是问不到，绝不反噬采样
            logger.debug("AutoApproach_OnOffGet failed", exc_info=True)
            return None
        d = _decoded(rec.return_value)
        v = _scalar(d[0]) if d else None
        return None if v is None else bool(v)

    def _excitation_metrics(self, metrics: dict, now: float) -> dict:
        """按需确认音叉是否被驱动，返回 amp_excited 的三态观测。
        
        只有振幅持续归零、判据需要解释时才查询，以控制额外通信开销。
        未知或读取失败时不写这个键；0 表示已确认关闭，不能用 0 冒充未知。
        未驱动时解调器噪声底不代表自由振荡振幅。缓存须服从有效期。
        """
        # 缓存过期就当没有 —— 一次「在振」之后用户关掉激励，而判据拿着几小时前
        # 的答案继续判，正是这条前置要挡的情形，只是晚了几个小时。
        if (self._excited is not None
                and now - self._excited_at < _EXCITATION_CACHE_TTL_S):
            return {"amp_excited": 1.0 if self._excited else 0.0}
        # 什么时候值得问：**这条判据现在是活的**。两种情形，任一即可 ——
        #
        #   amp_zero      振幅确实掉下去了，判据眼看要响；
        #   amp_gate_open 正在进针，也就是这条判据唯一会工作的那段时间。
        #
        # 只挂 ``amp_zero`` 是不够的：那样在一台**健康**的机器上，
        # 整个进针过程都因为「没问过」而判不了，面板永远给不出「正常」——
        # 而进针恰恰是用户最想看到这一路说话的时候。挂上 gate 之后，代价是一次
        # 进针里多问一两轮（受 30 s 节流），收益是那段时间里的判级是有依据的。
        #
        # 平时（不在进针、振幅也没掉）仍然**一句都不问** —— 有测试钉住。
        live = ((_num(metrics.get("amp_zero")) or 0.0) >= 1
                or (_num(metrics.get("amp_gate_open")) or 0.0) >= 1)
        if not live:
            return {}                      # 还不到值得问的时候
        if now - self._excited_probe < _EXCITATION_PROBE_MIN_INTERVAL_S:
            return {}                      # 刚问过且没问出来，别变成问句风暴
        self._excited_probe = now
        on = self._probe_excitation()
        if on is None:
            return {}                      # 读不到 = 不知道，不是「关着」
        self._excited, self._excited_at = on, now
        return {"amp_excited": 1.0 if on else 0.0}

    def _probe_excitation(self) -> Optional[bool]:
        """PLL 在不在驱动音叉？读不到就 ``None``。

        **两条证据都要**：输出开关开着（``PLL_OutOnOffGet``）**并且**激励幅度大于 0
        （``PLL_ExcitationGet``）。开关开着而幅度是 0，音叉照样没被驱动 ——
        口径与技能侧 ``_excitation_state`` 保持一致。

        任一条读不到就 ``None``：**不知道音叉有没有被驱动的时候，振幅这个数不代表
        任何东西**。代价是「PLL 读不回来的机器上这条判据不可用」，收益是它不再在
        一台未驱动音叉的机器上持续误报。
        """
        on: Optional[bool] = None
        exc_v: Optional[float] = None
        try:
            rec = self._call("PLL_OutOnOffGet", 1)
            d = _decoded(rec.return_value)
            raw = _scalar(d[0]) if d else None
            if raw is not None:
                on = bool(int(raw))
        except (_Busy, _Skip):
            return None
        except Exception:  # noqa: BLE001 — 问不到就是问不到，绝不反噬采样
            logger.debug("PLL_OutOnOffGet failed", exc_info=True)
            return None
        try:
            rec_v = self._call("PLL_ExcitationGet", 1)
            d_v = _decoded(rec_v.return_value)
            exc_v = _scalar(d_v[0]) if d_v else None
        except (_Busy, _Skip):
            return None
        except Exception:  # noqa: BLE001
            logger.debug("PLL_ExcitationGet failed", exc_info=True)
            return None
        if on is None or exc_v is None:
            return None
        return bool(on) and float(exc_v) > 0.0

    def _track_junction(self, ctx: dict, now: float) -> None:
        """以 Z 反馈 False→True 记录最近落针时刻，因此手动操作和技能操作都可覆盖。未观察到跃迁时保持 None，不能假设刚刚落针。"""
        z = ctx.get("ctx_zctrl_on")
        cur = None if z is None else bool(z)
        if cur is True and self._prev_zctrl is False:
            self._junction_since = now
        self._prev_zctrl = cur

    def _window_metrics(self, th) -> dict:
        out: dict = {}
        kinds = {r.spec.kind for r in self._channels}
        if "z" in kinds:
            ts, ys = _split(self._series["z"])
            tv = self._z_travel
            out.update(z_window_features(
                ts, ys, jump_k=float(th.cm_z_jump_k),
                limits=None if tv is None else (tv.lo_m, tv.hi_m),
                retract_sign=_retract_sign()))
        if "amplitude" in kinds:
            ts, ys = _split(self._series["amplitude"])
            out.update(amp_window_features(
                ts, ys, baseline=_profile_number("qplus_amplitude_baseline"),
                zero_frac=float(th.cm_amp_zero_frac)))
        if "df" in kinds:
            ts, ys = _split(self._series["df"])
            out.update(df_window_features(ts, ys))
        if "lockin" in kinds:
            ts, ys = _split(self._series["lockin"])
            out.update(lockin_window_features(ts, ys))
        if "bias" in kinds:
            ts, ys = _split(self._series["bias"])
            out.update(bias_window_features(ts, ys))
        return {k: v for k, v in out.items() if v is not None}

    def should_emit(self, rule: str, now: float | None = None) -> bool:
        """告警去抖。调用方（service）决定要不要落一条 alerts 行。"""
        return self._debounce.allow(rule, now)

    def reset(self) -> None:
        """断链/停机后忘掉一切 —— 索引描述的是**那一条**连接。

        与 ``CurrentMonitorService.stop()`` 清空 ``_pump`` 同一个理由：重连之后
        信号表可能已经不一样了，把旧连接的索引带进新连接就是在读别的信号。
        """
        self._channels = []
        self._series = {}
        self._configured = False
        self._bad_replies = 0
        self._prev_zctrl = None
        self._junction_since = None
        self._z_limits = None
        self._z_limits_enabled = None
        self._z_travel = None
        self._z_reads_complete = False
        self._limit_tries = 0
        self._last_limit_try = -math.inf
        self._last = None
        self._last_probe = -math.inf
        self._last_sample = -math.inf
        # 激励状态也要忘掉，同「进针闩」的理由：它是关于**那一条连接上那台仪器**的
        # 事实。Nanonis 重启 / 换机器之后拿旧答案给新读数开闸，正是这条前置要挡的
        # 情形（缺陷⑰）。忘掉之后回到「不知道」= 不判，方向是安全的。
        self._excited = None
        self._excited_at = -math.inf
        self._excited_probe = -math.inf
        self._deltas.clear()
        self._debounce.reset()
        # 进针闩也要忘掉：它记的是「那一条连接上什么时候有过进针证据」。
        # 带进新连接就是拿旧会话的证据给这一次的振幅读数开闸。
        self._gate.reset()
        self.detail = "尚未探测"

    # ── 对外快照（零 TCP） ──────────────────────────────────────────────

    def snapshot(self) -> dict:
        th = self._th()
        base = _profile_number("qplus_amplitude_baseline")
        tau = amplitude_tau_s()
        last = self._last
        metrics = dict(last.metrics) if last else {}
        by_kind = {r.spec.kind: r for r in self._channels}

        channels: list[dict] = []
        for spec in CHANNEL_SPECS:
            r = by_kind.get(spec.kind)
            value = (last.values.get(spec.kind) if last else None)
            note = ""
            if r is None and not self._configured:
                # **「还没看过」不是「这台机器没有」。** 探测成功之前就说
                # 「本机没有 qPlus」，是在没有证据的时候下一个结论 —— 而这条结论
                # 恰好会让人放弃排查。所以分开说。
                verdict = "unknown"
                note = f"信号表还没读到（{self.detail}）——尚未判定这台机器有没有这一路。"
            elif r is None:
                verdict = "unavailable"
                note = _UNAVAILABLE_NOTE.get(
                    spec.kind, "信号表里没有找到这一路。")
            elif spec.kind == "lockin":
                # lock-in 的第一句话不是「判不判」，是**「这条曲线是不是测量值」**。
                # 调制关着的时候它是噪声与串扰底，判级与否根本轮不上。
                verdict = "unjudged"
                note = _lockin_note(metrics)
            elif not spec.judged:
                verdict = "unjudged"
                # 「不判」有两种完全不同的理由，混成一句话会误导：Δf 是「还没标定」
                # （将来可以判），lock-in 是「这个量的形状不支持那类判据」（将来也
                # 不该判）。看到前者的人会去标定，看到后者的人不该去。
                note = _UNJUDGED_NOTE.get(
                    spec.kind, "只记录不判级——没有本机基线就不编阈值。")
            elif not th.aux_alerts_enabled:
                verdict = "unjudged"
                note = "告警未启用：阈值尚未在本机标定，先跑 commission 报告。"
            elif spec.kind == "amplitude" and base is None:
                verdict = "unjudged"
                note = ("没有未接触本底，判不了「振幅归零」——没有本底就不知道"
                        "「不为零」长什么样。针尖确认未接触时调用 "
                        "ReadTipOscillationAmplitude(set_baseline=True)。")
            elif spec.kind == "amplitude":
                verdict, note = _amplitude_verdict(last, metrics)
            else:
                fired = [x for x in (last.rules if last else [])
                         if x.startswith("z_" if spec.kind == "z" else "amp_")]
                verdict = "suppressed" if (last and last.verdict == "suppressed") \
                    else ("warn" if fired else "ok")
            channels.append({
                "kind": spec.kind, "label_zh": spec.label_zh, "unit": spec.unit,
                "available": r is not None,
                "signal_index": r.index if r else -1,
                "signal_name": r.name if r else "",
                "judged": bool(spec.judged),
                "value": value, "ts": last.ts if last else None,
                "verdict": verdict, "note": note,
                "metrics": {k: float(v) for k, v in metrics.items()
                            if k.startswith(_PREFIX[spec.kind])},
            })

        observed = self.observed_interval_s()
        return {
            "enabled": bool(th.aux_enabled),
            "alerts_enabled": bool(th.aux_alerts_enabled),
            "interval_s": float(th.cm_aux_interval_s),

            "observed_interval_s": observed,
            "window_s": float(th.cm_aux_window_s),
            "sampled": int(self.stats["sampled"]),
            "skipped_busy": int(self.stats["busy"]),
            "last_ts": last.ts if last else None,
            "baseline_amp_m": base,
            "amp_tau_s": tau,

            "amp_oversampled": (None if tau is None else bool(
                (observed if observed is not None
                 else float(th.cm_aux_interval_s)) <= tau)),
            "detail": self.detail,
            # ``z_limits_m`` 仍然是 ``ZCtrl_LimitsGet`` 的**原始**软限值 —— 语义没变。
            # 但它不再是余量的分母：软限值未启用时不起任何作用。下面两个说的是
            # **真正被当成分母的那对数字，以及是哪一条读答出来的**。一个算错分母的
            # 百分比和一个算对的长得一模一样，来源就是唯一分得开它们的东西。
            "z_limits_m": list(self._z_limits) if self._z_limits else [],
            "z_limits_enabled": self._z_limits_enabled,
            "z_travel_m": ([self._z_travel.lo_m, self._z_travel.hi_m]
                           if self._z_travel is not None else []),
            "z_travel_source": (self._z_travel.source
                                if self._z_travel is not None else ""),
            "channels": channels,
        }


# 每个通道 kind 必须有指标列名前缀，与 CHANNEL_SPECS 同步。
_PREFIX = {"z": "z_", "amplitude": "amp_", "df": "df_", "lockin": "lockin_",
           "bias": "bias_"}

#: 「这台机器没有这一路」要说清是**哪一路**、以及那正不正常。
#: 一句通用的「没找到」会让用户去查一个根本不存在的故障。
_UNAVAILABLE_NOTE: dict[str, str] = {
    "z": "信号表里没有找到 Z 通道。",
    "amplitude": "信号表里没有这一路——这台机器很可能没有 qPlus 传感器，不是故障。",
    "df": "信号表里没有这一路——这台机器很可能没有 qPlus 传感器，不是故障。",
    "lockin": ("信号表里没有找到 lock-in 解调通道——没有 lock-in 是完全正常的配置。"
               "本机有而没认出来的话，在「设置 → 仪器标定」填 "
               "lockin_signal_index。"),
}

#: 「记录但不判级」的两种理由，见调用点的注释。lock-in 走 :func:`_lockin_note`
#: （它还要先说调制开没开），所以这张表现在只剩兜底那一支。
_UNJUDGED_NOTE: dict[str, str] = {}

#: lock-in 「不打算判」的理由。**与调制状态无关**，所以单独放着：调制开着的时候
#: 这句话仍然成立（X 分量穿零），它回答的是「将来会不会加判据」，不是「现在这条
#: 曲线是不是测量值」。两个问题混成一段的话，看到调制关闭的人会以为只是没标定。
_LOCKIN_UNJUDGED_WHY = ("不判级也不打算判：本机读的是解调器 X 分量"
                        "（带符号，会穿零），比例类判据会在相位转动时报假警。")


def _lockin_note(metrics: dict) -> str:
    """dI/dV 通道的一句话。**先说调制开没开。**

    需求是辅助通道 didv 应该以某种方式标出 lockin 开没开。这不是显示偏好：
    调制关着的时候解调器输出的是噪声与串扰底，量纲、数量级、曲线形状都和真 dI/dV
    一样，只有「它是不是测量值」这一件事不一样 —— 而那件事在数字里看不出来。

    三态各说各的话。**「没读到」绝不能说成「关闭」**：那会让人把一条真 dI/dV
    当成噪声底扔掉，而这一路本来就是拿来做谱学的。
    """
    mod = _num(metrics.get("lockin_mod_on"))
    if mod is None:
        # 措辞里刻意不出现「dI/dV」三个字：这一支要说的正是「说不出它是不是」，
        # 而任何含着那个词的句子读起来都像在给一个答案。
        return ("调制状态**没读到**（不等于关闭）——现在判断不了这条曲线是测量值"
                "还是噪声底。" + _LOCKIN_UNJUDGED_WHY)
    if mod >= 1:
        return "调制**开着**，这条曲线是 dI/dV 信号。" + _LOCKIN_UNJUDGED_WHY
    return ("调制**关闭**——这条曲线**不是 dI/dV**，是解调器的噪声与串扰底。"
            "数值照记，但别按 dI/dV 读。")


def _amplitude_verdict(last: "AuxSample | None", metrics: dict) -> tuple[str, str]:
    """振幅通道的判级 + 一句为什么。

    **「不在进针所以没判」绝不能显示成「正常」。** 这条判据一天里绝大部分时间是
    不判的（只在进针期间判），如果那些时候给绿色，面板就会在 99% 的时间里对一个
    根本没在做的判断打包票 —— 这正是本仓反复付学费的那个形状：
    一个读数在最需要它说话的时候，长得像「没问题」。
    """
    if last is None:
        return "unknown", "还没有采到样本。"
    if any(r.startswith("amp_") for r in last.rules):
        return "warn", ""
    if (_num(metrics.get("amp_blanked")) or 0.0) >= 1:
        # 只列**技能**这一个原因 —— 起振那条触发已经删掉了（见
        # _ApproachGate.observe）。把已经不存在的触发继续写在文案里，读者会据此
        # 得出「刚才起振了所以不判」，而实际上起振根本不再让它进这个分支。
        # 那正是本仓「状态字段说的话和实际不符、而下游全都信它」的形状，
        # 只不过这一次的下游是人。
        return ("unjudged",
                "有扎针 / 电脉冲类技能正在占用仪器，振幅在这期间不判 —— "
                "那时的接触是**故意的**，不该报成事故。")
    # 音叉没被驱动 / 不知道有没有被驱动（缺陷⑰）。排在 amp_off 之前：这是**更根本**
    # 的一条 —— 通道读得到数、数也不为零，只是那个数不代表任何东西。
    #
    # ⚠️ **判级用 ``unjudged`` 而不是 ``unavailable``**，尽管技能侧那条前置用的是后者。
    # 这个面板上 ``unavailable`` 已经有一个**不同而且会误导**的含义：「本机没有这一路
    # 信号」（没装 qPlus 的 STM，正常配置不是故障）。激励关着的机器**有**这一路，
    # 只是没被驱动 —— 说成「本机没有」会让用户就此停止排查，而前端那份词表的注释
    # 里恰好写着这句话：「the second reading is the one that makes someone stop
    # investigating」。语义要求（第三态、既不是 ok 也不是 warn、永不显示成绿色）
    # 由 ``unjudged`` 一字不差地满足；换个新词只会让同一个面板上出现两个说法。
    excited = _num(metrics.get("amp_excited"))
    if excited is not None and excited < 1:
        return ("unjudged",
                "PLL 激励关着，音叉没有被驱动——**这一路现在没有判据能力**。"
                "读到的是未驱动解调器的噪声底，拿它跟自由振荡"
                "基线比永远比出「塌了」。要让撞针判据工作，先把 PLL 输出与激励打开。")
    if excited is None and ((_num(metrics.get("amp_zero")) or 0.0) >= 1
                            or (_num(metrics.get("amp_gate_open")) or 0.0) >= 1):
        # 只在**问过而没问出来**的时候说这句（问的条件见 _excitation_metrics）。
        # 平时不问，那时候沉默是对的 —— 一条「读不到 PLL」的提示天天挂着，
        # 一周之内就会被无视，而那正好毁掉它在真的读不到时的作用。
        return ("unjudged",
                "**读不到 PLL 激励状态**，所以判不了是不是撞针——不知道音叉有没有"
                "被驱动的时候，振幅这个数不代表任何东西。"
                "检查 PLL 模块是否加载、TCP 是否正常。")
    if (_num(metrics.get("amp_off")) or 0.0) >= 1:
        return ("unjudged",
                "窗口内从来没有读到过非零振幅——最可能是这条通道关着/没接。"
                "**这不是「没撞针」**：针尖整个窗口都还扎着也长这样。")
    if (_num(metrics.get("amp_gate_open")) or 0.0) >= 1:
        # 只有**确认过音叉在振**才敢说这句。``amp_excited`` 缺席时我们没问过
        # （振幅没归零，不值得问）—— 那时说「针尖还能自由振动」就是替一个没做过的
        # 观测打包票，而这一路在 STM 模式下平时读的本来就是噪声底。
        if excited is not None and excited >= 1:
            return ("ok", "正在进针，音叉在被驱动，且振幅没有归零——针尖还能自由振动。")
        return ("unjudged",
                "正在进针，振幅也没有归零。但**没有确认音叉在不在被驱动**"
                "(只有振幅真的掉下去时才会去问 PLL)，所以这不构成「针尖还能自由"
                "振动」的证据——STM 模式下这一路平时读的就是噪声底。")
    return ("unjudged",
            "只在**进针期间**判「振幅归零」，现在没有进针证据，所以不判——"
            "这不是「振幅正常」。STM 模式下这一路平时是噪声：隧穿不阻尼它"
            "；扎针与电脉冲可能使振幅发生瞬态变化。"
            "它唯一不可替代的用处是：针尖不导电时扎进去也没有电流信号，"
            "那时只有振幅会掉下去。")


class _Skip(Exception):
    """本轮跳过 —— 不是错误（模块没开 / 熔断 / 这台机器没有这路信号）。"""


class _Busy(Exception):
    """角色锁被占 —— 正常事件，直接跳过这一次。"""


def _split(dq) -> tuple[list[float], list[float]]:
    if not dq:
        return [], []
    ts, ys = zip(*dq)
    return list(ts), list(ys)


__all__ = [
    "AuxSampler", "AuxSample", "AuxChannelSpec", "CHANNEL_SPECS",
    "AUX_WARN_RULES", "AUX_METRIC_COLUMNS", "SUPPRESSION_EXEMPT_RULES",
    "APPROACH_SKILL_PATTERNS", "DISTURB_SKILL_PATTERNS",
    "z_window_features", "amp_window_features", "df_window_features",
    "bias_window_features",
    "lockin_window_features",
    "evaluate_aux", "summarize_aux_zh", "amplitude_tau_s", "no_junction",
]
