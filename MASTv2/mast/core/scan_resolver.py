"""Scan resolver — 把「意图」确定性地解析成一套完整的扫描参数。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

这是「LLM 不填数字」这条设计原则的落点。分工:

  * **意图层**(LLM / 用户):扫哪(center)、多大(size)、什么目的(purpose)、
    以及**用户逐字点名过的**物理参数(bias / setpoint / angle)。
  * **策略层**(本模块 + :mod:`mast.core.scan_policy`):速度、像素、PI 增益、
    setpoint 缺省 —— 从尺度推导的执行细节。

分界判据:**出现在用户自然语言里的量 = 意图**(「在这里扫一张 200 nm 的图」);
**从尺度推导出来的 = 策略**(「200 nm 的图该用多少像素、多慢」)。

为什么不能靠提示词让模型别填数字:2026 年的三模式 assembler 就是那么做的
(靠改变喂给模型的知识量间接改变行为),最后被 D-discard。活下来的是
``mode_mw`` 那条路 —— 同一套工具、用信念文本 + 硬门控改变**可执行的动作集**。
所以这里的做法是**把发明数字的诱因移除**:``ScanAt`` 的可选参数「不填」是默认
路径,策略层接得住;而不是在提示词里恳求模型别填。

本模块是**纯函数**,零 I/O、零硬件、零 LLM。输出仍然全量过 SafetyGate ——
这里的 clamp 是「参数卫生」(把坏偏好修成可用值并声明),SafetyGate 是「安全
边界」(只拒不改)。两层语义不同,都保留,互不豁免。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mast.core import scan_policy

logger = logging.getLogger(__name__)


# ── 参数卫生边界(与既有 skill 的 ParameterSpec 对齐) ────────────────────────
# 对齐来源:
#   size/center  → ConfigureScan.width_m/height_m 的 1e-10..1e-5
#   line_time_s  → ConfigureScan.line_time_s 的 1e-4..600
#   pixels       → SetScanBuffer(本设计新增)的 16..4096
#   setpoint_a   → SetSetpoint 的 1e-12..1e-7,这里放宽下界到 1e-15 只是卫生,
#                  真正的拒绝由 SafetyGate 做
#   bias_v       → SafetyGate 的全局 ±10 V
_CLAMP: dict[str, tuple[float, float]] = {
    "size_m":         (1e-10, 1e-5),
    "line_time_s":    (1e-4, 600.0),
    "pixels":         (16, 4096),
    # 与 SetSetpoint 的 ParameterSpec 对齐(1 pA..100 nA)。刻意不放宽:产出一个
    # 必然被下游 validate_params 拒掉的值,等于把错误推迟到硬件路径上才发现。
    "setpoint_a":     (1e-12, 1e-7),
    "bias_v":         (-10.0, 10.0),
    "angle_deg":      (-180.0, 180.0),
    # 与 SetZCtrlGain 的 ParameterSpec 对齐(2026-08-03 起它才有上界)。这两行原本
    # 是 1e6 / 1e3 —— 比技能自己的上界还宽 6 个数量级,于是档位表里一个手滑的
    # 增益会被"修剪"成一个仍然必被 validate_params 拒掉的值,正好违反本表开头
    # 那条自述(不产出必然被下游拒绝的值)。
    "p_gain":         (0.0, 1e-6),
    "time_constant_s": (0.0, 10.0),
}

#: 针尖横向速度的兜底上限(m/s)。真值应由 instrument_profile 提供;这个默认
#: 值(2 µm/s)只是「明显过快」的护栏 —— 在 1 µm 帧上对应 0.5 s/线,已经是
#: 大多数机器的快扫极限。
_DEFAULT_V_TIP_MAX = 2e-6

#: ``purpose`` 的保留值:按尺寸自动定档。其余合法值是档名。
PURPOSE_AUTO = "auto"

#: 显式点名这些档 = 一句「我要看原子」的物理意图，足以决定工作点。
_ATOMIC_TIERS = frozenset({"atomic", "atomic_verify"})

#: trace 里 source 的全部取值。UI 与测试都按这个枚举断言。
SOURCE_EXPLICIT = "explicit"              # 用户逐字点名(经 LLM 转述)
SOURCE_TIER_OPERATOR = "tier-operator"    # 用户的档位表
SOURCE_TIER_FACTORY = "tier-factory"      # 出厂档位表
SOURCE_PREFS = "prefs"                    # 实验默认参数偏好
SOURCE_PREFS_DERIVED = "prefs-derived"    # 由偏好推导(如 速度 → 每线时间)
SOURCE_DEFAULT = "default"                # 模块内建默认(如 channels)
SOURCE_KEEP = "keep-current"              # 不下发,保持硬件现值
SOURCE_INTENT = "intent"                  # 由 purpose 表达的物理意图（见下）


@dataclass(frozen=True)
class ScanIntent:
    """LLM / 用户唯一能表达的东西。

    ``explicit`` 只装**用户逐字点名过的**值。它与「模型自己觉得合适的值」在
    结构上无法区分 —— 这一点老实承认,所以每个 explicit 值都会进 trace 并显示
    给用户看:一个标着『用户显式』、而用户并没有给过的 line_time,是一眼能看出来的。
    """

    center_x_m: float
    center_y_m: float
    size_m: float
    purpose: str = PURPOSE_AUTO
    explicit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedScan:
    """解析结果 —— 一套可以直接喂给 skill 的完整参数,外加来源痕迹。

    ``set_setpoint`` / ``set_bias`` / ``set_zctrl_gain`` 为 ``None`` 表示
    **不下发那个硬件写**(空即 no-op),不是「下发 0」。
    """

    tier_name: str
    configure_scan: dict[str, Any]
    set_scan_buffer: dict[str, Any]
    set_setpoint: "dict[str, Any] | None"
    set_bias: "dict[str, Any] | None"
    set_zctrl_gain: "dict[str, Any] | None"
    trace: dict[str, dict[str, Any]]
    warnings: list[str]
    estimated_scan_s: float

    def summary_lines(self) -> list[str]:
        """供用户核对参数来源的可读表，SI 值与人类单位并列显示。"""
        label = {
            SOURCE_EXPLICIT: "用户显式指定",
            SOURCE_TIER_OPERATOR: "档位表(用户)",
            SOURCE_TIER_FACTORY: "档位表(出厂)",
            SOURCE_PREFS: "实验默认偏好",
            SOURCE_PREFS_DERIVED: "实验默认偏好(推导)",
            SOURCE_DEFAULT: "内建默认",
            SOURCE_KEEP: "保持硬件现值",
        }
        out: list[str] = []
        for name, rec in self.trace.items():
            src = label.get(str(rec.get("source")), str(rec.get("source")))
            val = rec.get("value")
            human = rec.get("human")
            val_s = f"{val}" if human is None else f"{val}  ({human})"
            tier = rec.get("tier")
            tier_s = f" @{tier}" if tier else ""
            out.append(f"- {name} = {val_s} ← {src}{tier_s}")
        return out


def _num(value: Any, kind: type = float) -> "float | int | None":
    """强制成数值;NaN / inf / 非数值 → None。"""
    try:
        val = kind(value)
    except (TypeError, ValueError):
        return None
    if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
        return None
    return val


def _clamp(name: str, value: Any, warnings: list[str], kind: type = float) -> Any:
    """按 :data:`_CLAMP` 修剪一个值,越界时往 warnings 里记一条。

    修剪而不是拒绝,是因为这一层是参数卫生:偏好表里一个手滑的数字不该让整次
    扫描失败。真正该拒绝的越界由 SafetyGate 负责(它只拒不改)。
    """
    val = _num(value, kind)
    if val is None:
        return None
    bounds = _CLAMP.get(name)
    if bounds is None:
        return val
    lo, hi = bounds
    if val < lo or val > hi:
        clamped = max(lo, min(hi, val))
        warnings.append(
            f"{name}={val:.6g} 超出可用范围 [{lo:.6g}, {hi:.6g}],已修剪为 {clamped:.6g}"
        )
        val = clamped
    return int(val) if kind is int else float(val)


def _read_prefs() -> dict[str, Any]:
    """读实验默认参数偏好。

    **刻意用函数内的延迟 import**:``experiment_prefs`` 住在
    ``mast.agents._shared``,而 core → agents 是反向依赖(既有方向是
    agents/_shared → core,见 instrument_profile_mw)。在函数体内 import 不会
    在模块图上造成这条边,读不到就当没设 —— 偏好本来就是可选的。
    """
    try:
        from mast.agents._shared.experiment_prefs import get_prefs
        prefs = get_prefs()
        return prefs if isinstance(prefs, dict) else {}
    except Exception as exc:  # noqa: BLE001 - 偏好读不到绝不能让扫描失败
        logger.debug("scan_resolver: prefs 读取失败(按未设处理): %s", exc)
        return {}


def _read_v_tip_max() -> float:
    """针尖横向速度上限(m/s),来自 instrument_profile;没设就用兜底值。"""
    try:
        from mast.core import instrument_profile
        val = instrument_profile.get_config("v_tip_max_m_s", None)
        num = _num(val)
        if num is not None and num > 0:
            return float(num)
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan_resolver: v_tip_max 读取失败(用兜底值): %s", exc)
    return _DEFAULT_V_TIP_MAX


def _tier_source(tier: dict[str, Any], key: str) -> str:
    """这一档的某个字段到底是用户填的还是出厂回填的。"""
    if str(tier.get("source")) != "operator":
        return SOURCE_TIER_FACTORY
    if key in (tier.get("_factory_filled") or ()):
        return SOURCE_TIER_FACTORY
    return SOURCE_TIER_OPERATOR


def resolve_scan(
    intent: ScanIntent,
    *,
    tiers_lookup: Any = None,
    prefs: "dict[str, Any] | None" = None,
    v_tip_max_m_s: "float | None" = None,
) -> ResolvedScan:
    """把一个 :class:`ScanIntent` 解析成完整参数集 + 来源痕迹。

    优先级链(**逐字段独立**走,不是整组切换):

    1. ``explicit`` —— 用户单次显式指定(「XD 定的」,最高优先)
    2. 档位表里用户覆写过的值
    3. ``experiment_prefs`` 对应字段
    4. 档位表出厂默认
    5. keep-current —— 不下发该硬件写

    ``bias`` 不走档位表(它不是尺度的函数),链是 explicit > prefs > keep-current。
    """
    warnings: list[str] = []
    trace: dict[str, dict[str, Any]] = {}
    explicit = dict(intent.explicit or {})
    prefs = _read_prefs() if prefs is None else dict(prefs)
    v_tip_max = _read_v_tip_max() if v_tip_max_m_s is None else float(v_tip_max_m_s)

    def note(name: str, value: Any, source: str,
             tier_name: "str | None" = None, human: "str | None" = None) -> None:
        trace[name] = {"value": value, "source": source,
                       "tier": tier_name, "human": human}

    # ── 尺寸(意图,必填) ───────────────────────────────────────────────────
    # 两种非法要分开对待:
    #   * **非数值 / 非正数**(None、NaN、0、负数)不是「太小的尺寸」,而是上游
    #     出错的信号(变量没初始化、解析失败)。把 0 静默 clamp 成 0.1 nm 会扫出
    #     一张荒谬的图却什么都不说 —— 那正是本项目视为致命的 fail-silent。抛。
    #   * **正数但超出硬件范围**是「一个人真的打出来的数字」,属参数卫生,修剪
    #     并声明。
    raw_size = _num(intent.size_m)
    if raw_size is None or raw_size <= 0:
        raise ValueError(
            f"size_m 无效: {intent.size_m!r}(扫描边长必须是正数;"
            "不替用户发明尺寸)"
        )
    size_m = _clamp("size_m", raw_size, warnings)
    if size_m is None:                      # pragma: no cover - 上面已挡住
        raise ValueError(f"size_m 无效: {intent.size_m!r}")
    note("size_m", size_m, SOURCE_EXPLICIT, human=f"{size_m * 1e9:.4g} nm")

    center_x = _num(intent.center_x_m)
    center_y = _num(intent.center_y_m)
    if center_x is None or center_y is None:
        raise ValueError(
            f"扫描中心无效: ({intent.center_x_m!r}, {intent.center_y_m!r})"
        )
    note("center_x_m", center_x, SOURCE_EXPLICIT, human=f"{center_x * 1e9:.4g} nm")
    note("center_y_m", center_y, SOURCE_EXPLICIT, human=f"{center_y * 1e9:.4g} nm")

    # ── 定档 ───────────────────────────────────────────────────────────────
    purpose = str(intent.purpose or PURPOSE_AUTO).strip().lower()
    lookup = tiers_lookup or scan_policy
    if purpose and purpose != PURPOSE_AUTO:
        tier = lookup.get_tier_by_name(purpose)
        if tier is None:
            warnings.append(
                f"purpose='{intent.purpose}' 不是已知档名"
                f"(可用: {', '.join(lookup.tier_names())}),按尺寸自动定档"
            )
            tier = lookup.get_tier_for_size(size_m)
        else:
            auto_tier = lookup.get_tier_for_size(size_m)
            if auto_tier.get("name") != tier.get("name"):
                warnings.append(
                    f"purpose='{tier.get('name')}' 强制换档:"
                    f"{size_m * 1e9:.4g} nm 按尺寸本应属 '{auto_tier.get('name')}' 档"
                )
    else:
        tier = lookup.get_tier_for_size(size_m)
    tier_name = str(tier.get("name", "?"))

    # 「档位是被**显式点名**的」与「按尺寸碰巧定到这一档」是两件事。
    # 前者是一句物理意图（"我要看原子"），后者只是尺度巧合 —— 只有前者
    # 才有资格决定工作点（见下面 bias/setpoint 那两段）。
    purpose_named = bool(purpose and purpose != PURPOSE_AUTO
                         and tier.get("name") == purpose)
    intent_wp: "dict[str, float] | None" = None
    if purpose_named and tier_name in _ATOMIC_TIERS:
        from mast.vision.imaging_window import atomic_working_point
        intent_wp = atomic_working_point(None)

    # ── pixels ─────────────────────────────────────────────────────────────
    if explicit.get("pixels") is not None:
        pixels = _clamp("pixels", explicit["pixels"], warnings, int)
        note("pixels", pixels, SOURCE_EXPLICIT)
    else:
        tier_src = _tier_source(tier, "pixels")
        # 偏好只有在「档位表这个字段其实是出厂值」时才插得进来 —— 用户按尺度
        # 设过的值比一个全局标量更精确,不该被后者盖掉。
        if tier_src == SOURCE_TIER_FACTORY and prefs.get("scan_lines") is not None:
            pixels = _clamp("pixels", prefs["scan_lines"], warnings, int)
            note("pixels", pixels, SOURCE_PREFS)
        else:
            pixels = _clamp("pixels", tier.get("pixels"), warnings, int)
            note("pixels", pixels, tier_src, tier_name)
    if pixels is None:                       # pragma: no cover - 档位表保证非空
        pixels = 256
        note("pixels", pixels, SOURCE_DEFAULT)

    # ── line_time_s ────────────────────────────────────────────────────────
    if explicit.get("line_time_s") is not None:
        line_time = _clamp("line_time_s", explicit["line_time_s"], warnings)
        note("line_time_s", line_time, SOURCE_EXPLICIT, human=f"{line_time:.4g} s/线")
    else:
        tier_src = _tier_source(tier, "line_time_s")
        pref_lt = prefs.get("line_time_s")
        pref_speed = prefs.get("scan_speed_nm_s")
        if tier_src == SOURCE_TIER_FACTORY and pref_lt is not None:
            line_time = _clamp("line_time_s", pref_lt, warnings)
            note("line_time_s", line_time, SOURCE_PREFS, human=f"{line_time:.4g} s/线")
        elif tier_src == SOURCE_TIER_FACTORY and pref_speed:
            # 用户设的是「扫描速度」而不是「每线时间」—— 两者由帧宽换算。
            # 表里只存 line_time(单一真源),所以这里把速度折算过去。
            speed_m_s = _num(pref_speed)
            derived = size_m / (speed_m_s * 1e-9) if speed_m_s else None
            line_time = _clamp("line_time_s", derived, warnings)
            note("line_time_s", line_time, SOURCE_PREFS_DERIVED,
                 human=f"由 {speed_m_s:.4g} nm/s 与 {size_m * 1e9:.4g} nm 帧宽推导")
        else:
            line_time = _clamp("line_time_s", tier.get("line_time_s"), warnings)
            note("line_time_s", line_time, tier_src, tier_name,
                 human=f"{line_time:.4g} s/线")
    if line_time is None:                    # pragma: no cover
        line_time = 0.5
        note("line_time_s", line_time, SOURCE_DEFAULT)

    # ── 组合约束:针尖横向速度 ─────────────────────────────────────────────
    # 这是模型必然漏掉的那一类约束 —— 像素、每线时间、帧宽单独看都合法,乘起来
    # 才知道针尖要以多快的速度扫过表面。太快会拖坏针尖,而且反馈跟不上。
    speed = size_m / line_time if line_time > 0 else float("inf")
    if speed > v_tip_max:
        needed = size_m / v_tip_max
        warnings.append(
            f"针尖横向速度 {speed * 1e9:.4g} nm/s 超过上限 "
            f"{v_tip_max * 1e9:.4g} nm/s,每线时间由 {line_time:.4g} s "
            f"放慢到 {needed:.4g} s"
        )
        line_time = _clamp("line_time_s", needed, warnings) or needed
        rec = trace.get("line_time_s", {})
        rec["value"] = line_time
        rec["human"] = f"{line_time:.4g} s/线(受针尖速度上限限制)"
        trace["line_time_s"] = rec

    # ── angle_deg ──────────────────────────────────────────────────────────
    # 不传给 ConfigureScan 时它会**保持硬件当前角度**(2026-07-03 修的坑:
    # 传 0 会让每次 recenter 都把画面转回 0°)。所以 keep-current 的实现是
    # 「不放进 configure_scan 字典」,而不是「放一个 0」。
    angle = None
    if explicit.get("angle_deg") is not None:
        angle = _clamp("angle_deg", explicit["angle_deg"], warnings)
        note("angle_deg", angle, SOURCE_EXPLICIT, human=f"{angle:.4g}°")
    elif prefs.get("scan_angle_deg") is not None:
        angle = _clamp("angle_deg", prefs["scan_angle_deg"], warnings)
        note("angle_deg", angle, SOURCE_PREFS, human=f"{angle:.4g}°")
    else:
        note("angle_deg", None, SOURCE_KEEP)

    # ── channels ───────────────────────────────────────────────────────────
    channels = explicit.get("channels")
    if isinstance(channels, str) and channels.strip():
        note("channels", channels.strip(), SOURCE_EXPLICIT)
        channels = channels.strip()
    else:
        channels = "Z,Current"
        note("channels", channels, SOURCE_DEFAULT)

    # ── setpoint_a ─────────────────────────────────────────────────────────
    setpoint = None
    if explicit.get("setpoint_a") is not None:
        setpoint = _clamp("setpoint_a", explicit["setpoint_a"], warnings)
        note("setpoint_a", setpoint, SOURCE_EXPLICIT,
             human=f"{setpoint * 1e12:.4g} pA" if setpoint else None)
    elif tier.get("setpoint_a") is not None:
        setpoint = _clamp("setpoint_a", tier["setpoint_a"], warnings)
        note("setpoint_a", setpoint, _tier_source(tier, "setpoint_a"), tier_name,
             human=f"{setpoint * 1e12:.4g} pA" if setpoint else None)
    elif prefs.get("setpoint_pa") is not None:
        sp_pa = _num(prefs["setpoint_pa"])
        setpoint = _clamp("setpoint_a", (sp_pa or 0.0) * 1e-12, warnings)
        note("setpoint_a", setpoint, SOURCE_PREFS,
             human=f"{sp_pa:.4g} pA" if sp_pa else None)
    else:
        if intent_wp is not None and intent_wp.get("setpoint_a") is not None:
            setpoint = _clamp("setpoint_a", intent_wp["setpoint_a"], warnings)
            note("setpoint_a", setpoint, SOURCE_INTENT,
                 human=f"{setpoint * 1e12:.4g} pA（purpose='{tier_name}'）")
        else:
            note("setpoint_a", None, SOURCE_KEEP)

    # ── bias_v ─────────────────────────────────────────────────────────────
    # **bias 不进档位表**:它决定探测的电子态与成像对比,是物理意图参数,不是
    # 尺度的函数。知识库里 L1 那个 "0.5-2 V" 只是巡查建议,不是「1 µm 的图就该
    # 用 1 V」。resolver 永远不为 bias 发明数值。
    bias = None
    if explicit.get("bias_v") is not None:
        bias = _clamp("bias_v", explicit["bias_v"], warnings)
        note("bias_v", bias, SOURCE_EXPLICIT, human=f"{bias:.4g} V")
    elif prefs.get("bias_v") is not None:
        bias = _clamp("bias_v", prefs["bias_v"], warnings)
        note("bias_v", bias, SOURCE_PREFS, human=f"{bias:.4g} V")
    elif intent_wp is not None and intent_wp.get("bias_v") is not None:
        # 扫描尺寸本身不能决定偏压，但 purpose=atomic 明确给出了用途。
        # 此时从 MakeAtomicResolutionTip.eval_bias_v 取得工作点，避免沿用
        # 上一个技能留下的偏压；数值来源仍是流程表而不是 resolver 的猜测。
        bias = _clamp("bias_v", intent_wp["bias_v"], warnings)
        note("bias_v", bias, SOURCE_INTENT,
             human=f"{bias:.4g} V（purpose='{tier_name}' 的意图默认）")
    else:
        note("bias_v", None, SOURCE_KEEP)

    # ── PI 增益 ────────────────────────────────────────────────────────────
    # 刻意**不**在 ScanAt 的显式覆盖集里:增益不是用户在自然语言里会说的量
    # (「用 P=1e-11 扫」不是人话)。要调就去档位表里调,或直接用 SetZCtrlGain。
    p_gain = _clamp("p_gain", tier.get("p_gain"), warnings)
    t_const = _clamp("time_constant_s", tier.get("time_constant_s"), warnings)
    if p_gain is not None or t_const is not None:
        note("p_gain", p_gain, _tier_source(tier, "p_gain"), tier_name)
        note("time_constant_s", t_const, _tier_source(tier, "time_constant_s"), tier_name)
    else:
        note("p_gain", None, SOURCE_KEEP)
        note("time_constant_s", None, SOURCE_KEEP)

    # ── 装配 ───────────────────────────────────────────────────────────────
    configure: dict[str, Any] = {
        "center_x_m": center_x,
        "center_y_m": center_y,
        "width_m": size_m,
        "height_m": size_m,
        "channels": channels,
        "set_scan_speed": True,
        "line_time_s": line_time,
    }
    if angle is not None:
        configure["angle_deg"] = angle

    # SetZCtrlGain 三个参数全必填,且 I = P/T(积分增益由 P 与时间常数导出,
    # 不是独立自由度)。所以只填了一半的 PI 配置是**不可执行**的 —— 与其送一个
    # 必然被 validate_params 拒掉的调用,不如在这里说清楚缺什么。
    set_zctrl: "dict[str, Any] | None" = None
    if p_gain is not None and t_const is not None:
        if t_const > 0:
            set_zctrl = {
                "p_gain": p_gain,
                "time_constant_s": t_const,
                "i_gain": p_gain / t_const,
            }
        else:
            warnings.append(
                f"档位 '{tier_name}' 的时间常数为 0,无法导出积分增益(I = P/T),"
                "本次不下发 PI 设置"
            )
    elif p_gain is not None or t_const is not None:
        missing = "time_constant_s" if p_gain is not None else "p_gain"
        warnings.append(
            f"档位 '{tier_name}' 的 PI 配置不完整(缺 {missing}),"
            "Z 反馈需要 P 与时间常数成对设置,本次不下发 PI 设置"
        )

    return ResolvedScan(
        tier_name=tier_name,
        configure_scan=configure,
        set_scan_buffer={"pixels": int(pixels), "lines": int(pixels)},
        set_setpoint=({"setpoint_a": setpoint} if setpoint is not None else None),
        set_bias=({"bias_v": bias} if bias is not None else None),
        set_zctrl_gain=set_zctrl,
        trace=trace,
        warnings=warnings,
        estimated_scan_s=scan_policy.estimate_scan_seconds(pixels, line_time),
    )


def preview(size_m: float, purpose: str = PURPOSE_AUTO,
            **explicit: Any) -> dict[str, Any]:
    """给设置界面用的零硬件成本预览:「200 nm 的图会用什么参数」。

    用户改档位表时立刻看到效果,不必真的扫一张。中心固定 (0,0) —— 预览关心
    的是参数不是位置。
    """
    intent = ScanIntent(
        center_x_m=0.0, center_y_m=0.0, size_m=size_m,
        purpose=purpose,
        explicit={k: v for k, v in explicit.items() if v is not None},
    )
    res = resolve_scan(intent)
    return {
        "tier_name": res.tier_name,
        "pixels": res.set_scan_buffer["pixels"],
        "line_time_s": res.configure_scan["line_time_s"],
        "estimated_scan_s": res.estimated_scan_s,
        "setpoint_a": (res.set_setpoint or {}).get("setpoint_a"),
        "bias_v": (res.set_bias or {}).get("bias_v"),
        "p_gain": (res.set_zctrl_gain or {}).get("p_gain"),
        "time_constant_s": (res.set_zctrl_gain or {}).get("time_constant_s"),
        "trace": res.trace,
        "warnings": res.warnings,
        "summary": res.summary_lines(),
    }


__all__ = [
    "PURPOSE_AUTO",
    "SOURCE_EXPLICIT",
    "SOURCE_TIER_OPERATOR",
    "SOURCE_TIER_FACTORY",
    "SOURCE_PREFS",
    "SOURCE_PREFS_DERIVED",
    "SOURCE_DEFAULT",
    "SOURCE_KEEP",
    "ScanIntent",
    "ResolvedScan",
    "resolve_scan",
    "preview",
]
