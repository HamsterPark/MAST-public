"""修针参数解析 —— 一个数字从哪来,必须查得到。

形状照抄 :mod:`mast.core.scan_resolver`(逐字段独立走链 + 来源 trace + 依赖可
注入),解决的是同一类问题:模型不该替物理发明数字,但也不能一刀切地拒绝它给的
值。

优先级链(**逐字段独立**,不是整组切换):

1. ``explicit`` —— 调用方显式给的值(模型或用户在 tool call 里写的)
2. 用户覆写方案表(``SettingsStore["tip_conditioning_overrides"]``)
3. 出厂方案表按当前针尖的「材料 × 制备 × 形态」查出来的档
4. 通用保守默认(针尖未登记时就是这一档)

## clamp 还是拒绝

这一层**不 clamp**。``scan_resolver`` 里 clamp 是对的 —— 偏好表里一个手滑的
数字不该让整次扫描失败,那是参数卫生。这里不一样:超出针尖安全包络的处理会
**不可逆地毁掉硬件**(qPlus 石英音叉尤其:损坏要拆机重装 + 重新标定 f₀/Q)。
把 8 V 悄悄夹成 3 V,用户会以为自己打了 8 V 而结果不对;把针戳穿了则什么都
救不回来。所以超包络一律**拒绝并说明**,与粗动电压四重锁同一条哲学。

## 未登记针尖:这里**照样拒绝**(2026-08-10 更正)

这段原本写着「未登记针尖时不拒绝任何东西(fail-open,与 sample_gate 同款)」。
**那句是假的。** 未登记时用的是通用保守档 ``_DEFAULT``,而那一档**有自己的包络**
(``max_abs_pulse_v = 6.0``、``max_poke_depth_m = 3.0e-9``),``_check_envelope``
照样对着它判、照样拒绝。fail-open 的是**另一件事**:
``_tip_policy.qplus_gate``(qPlus 那道**策略**门)在读不到针尖时放行 ——
两道门,两条哲学,而这段注释把其中一道的性质安到了另一道头上。

代价是具体的:``NobleTipWorkflow.pulse_v`` 出厂 10 V > 通用档的 6 V,
⇒ **没有登记针尖时,ForgeAuTip 的大修相一发脉冲都打不出去**,
而「未登记」是本系统最常见的状态。这是「出厂默认与包络从没对过账」的第三个面
(前两个:qPlus 档的 poke 与 pulse)。对账见
``mast.core.noble_tip_workflow.reconcile_with_tip_envelope``,
覆盖每一档**包括通用档**的结构闸门见
``tests/v2/unit/core/test_workflow_defaults_fit_every_tip_envelope.py``。

**要推翻「未登记也拒绝」需要回答**:不知道装的是什么针时,按最保守档拒绝,
和放行让用户自己负责,哪个更可能毁掉硬件?现在的实现选前者;
真正错的从来不是这个选择,而是**注释在替它说一句不成立的话**,
以及**出厂默认落在自己包络之外**。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mast.core.tip_conditioning_policy import resolve_policy

logger = logging.getLogger(__name__)

SOURCE_EXPLICIT = "explicit"
SOURCE_OVERRIDE = "operator_override"
SOURCE_POLICY = "policy_table"
SOURCE_DEFAULT = "factory_default"


@dataclass
class ResolvedConditioning:
    """解析结果:参数 + 每个值的来源 + 拒绝原因(如果有)。"""

    params: dict[str, Any] = field(default_factory=dict)
    #: 字段 → 来源(explicit / operator_override / policy_table / factory_default)
    trace: dict[str, str] = field(default_factory=dict)
    #: 非空 = 被拒绝,调用方必须**不执行**并把这些话回给用户。
    refusals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: 当前针尖的一句话说明(注入结果里给模型看)。
    notes: list[str] = field(default_factory=list)
    #: 解析时当前针尖的事实(None = 未登记)。
    tip: "dict[str, Any] | None" = None

    @property
    def ok(self) -> bool:
        return not self.refusals

    def human_trace(self) -> str:
        """「pulse_v=5.0（方案表）」这样的一行,放进技能结果里。"""
        label = {SOURCE_EXPLICIT: "调用方指定", SOURCE_OVERRIDE: "用户覆写",
                 SOURCE_POLICY: "针尖方案表", SOURCE_DEFAULT: "通用默认"}
        bits = [f"{k}={self.params[k]!r}（{label.get(v, v)}）"
                for k, v in sorted(self.trace.items()) if k in self.params]
        return "；".join(bits)


def _read_tip_facts() -> "dict[str, Any] | None":
    """当前针尖事实。读不到就当未登记 —— 绝不猜。"""
    try:
        from mast.core.tip_state import current_tip_facts
        return current_tip_facts()
    except Exception as exc:  # noqa: BLE001
        logger.debug("tip_conditioning_resolver: 读针尖失败(按未登记处理): %s", exc)
        return None


def _read_overrides() -> dict[str, Any]:
    """用户在设置里填的方案覆写。

    刻意用函数内延迟 import:``webui.settings_store`` 不该出现在 core 的模块图
    上(core 是被依赖的一层)。读不到就当没设。

    ``settings_store_for_runtime()`` 而不是 ``SettingsStore()``:后者的
    ``config_dir`` 是**必填位置参数**,无参调用抛 TypeError、被下面的 except 吞掉,
    于是这个函数从上线起恒返回 ``{}`` —— 用户的覆写一次都没被读到过(2026-08-10
    实测)。而返回 ``{}`` 正好等于「没设覆写」,所以整条链看上去完全正常。
    """
    try:
        from mast.webui.settings_store import settings_store_for_runtime
        val = settings_store_for_runtime().get("tip_conditioning_overrides")
        return dict(val) if isinstance(val, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("tip_conditioning_resolver: 覆写读取失败(按未设处理): %s", exc)
        return {}


def _num(value: Any) -> "float | None":
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def resolve_conditioning(
    fields: "list[str] | tuple[str, ...]",
    explicit: "dict[str, Any] | None" = None,
    *,
    facts: "dict[str, Any] | None" = None,
    overrides: "dict[str, Any] | None" = None,
    skip_facts_lookup: bool = False,
) -> ResolvedConditioning:
    """把请求的 *fields* 解析成一整组参数 + 来源痕迹。

    *explicit* 里值为 None / 缺席的字段走方案表;给了值的字段被采纳,但仍要过
    安全包络检查(**超了就拒绝,不夹紧**)。

    *facts* / *overrides* 可注入,方便测试;默认从 holder 与 SettingsStore 读。
    """
    if facts is None and not skip_facts_lookup:
        facts = _read_tip_facts()
    if overrides is None:
        overrides = _read_overrides()

    policy = resolve_policy(facts, overrides=overrides)
    pol_sources = policy.get("_sources") or {}

    res = ResolvedConditioning(tip=facts,
                               notes=[str(n) for n in (policy.get("_notes") or [])])
    explicit = dict(explicit or {})

    for name in fields:
        given = explicit.get(name)
        if given is not None and str(given).strip() != "":
            res.params[name] = given
            res.trace[name] = SOURCE_EXPLICIT
            continue
        if name in policy:
            res.params[name] = policy[name]
            res.trace[name] = pol_sources.get(name, SOURCE_POLICY)
        # 表里没有这个字段就不填 —— 由技能自己的 ParameterSpec 默认兜底。

    _check_envelope(res, policy)
    return res


def _check_envelope(res: ResolvedConditioning, policy: dict[str, Any]) -> None:
    """安全包络:超上限**拒绝**,不夹紧(见模块 docstring)。"""
    tip = res.tip or {}
    is_qplus = str(tip.get("form") or "") == "qplus"
    what = f"当前针尖（{tip.get('name') or '未命名'}）" if tip else "未登记针尖的通用档"

    max_pulse = _num(policy.get("max_abs_pulse_v"))
    for key in ("pulse_v", "shaper_bias_v", "shaper_lift_v"):
        val = _num(res.params.get(key))
        if val is None or max_pulse is None:
            continue
        if abs(val) > max_pulse:
            res.refusals.append(
                f"{key}={val:g} V 超出{what}的安全上限 ±{max_pulse:g} V。"
                + ("qPlus 石英音叉损坏不可逆（需拆机重装并重新标定 f₀/Q），"
                   "所以这里拒绝而不是替你夹到上限。" if is_qplus else
                   "拒绝而不是夹到上限——夹了你会以为自己用的是原来那个值。")
                + "确需更大幅度请先确认针尖类型登记正确，或在设置里调整该针尖的方案上限。")

    max_depth = _num(policy.get("max_poke_depth_m"))
    for key in ("shaper_depth_m", "poke_shallow_depth_m", "poke_deep_depth_m"):
        val = _num(res.params.get(key))
        if val is None or max_depth is None:
            continue
        # 深度是负数(向表面下压),比较绝对值。
        if abs(val) > max_depth:
            res.refusals.append(
                f"{key}={val:.3e} m 的下压深度超出{what}的上限 {max_depth:.3e} m。"
                + ("qPlus 传感器被戳坏不可逆。" if is_qplus else "")
                + "拒绝执行。")

    max_count = _num(policy.get("max_pulse_count"))
    cnt = _num(res.params.get("pulse_count"))
    if cnt is not None and max_count is not None and cnt > max_count:
        res.refusals.append(
            f"pulse_count={int(cnt)} 超出{what}的上限 {int(max_count)} 发。拒绝执行。")


def envelope_of(facts: "dict[str, Any] | None" = None) -> dict[str, Any]:
    """当前针尖的安全包络(给 UI / 技能描述用)。"""
    if facts is None:
        facts = _read_tip_facts()
    pol = resolve_policy(facts, overrides=_read_overrides())
    from mast.core.tip_conditioning_policy import LIMIT_FIELDS
    return {k: pol[k] for k in LIMIT_FIELDS if k in pol}


__all__ = [
    "ResolvedConditioning", "resolve_conditioning", "envelope_of",
    "SOURCE_EXPLICIT", "SOURCE_OVERRIDE", "SOURCE_POLICY", "SOURCE_DEFAULT",
]
