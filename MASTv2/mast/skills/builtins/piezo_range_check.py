"""压电范围对账：比较软件配置与仪器当前读数。

软件安全范围可能与温度、扫描器或标定变化后的硬件范围不一致，
导致选点器提出不可到达的目标。配置值不能替代实际范围核验。
本技能只报告，不更改任何设置；修改范围需要独立的明确操作。

ok 表示在 tolerance_frac 内一致；mismatch 表示两侧均可读但不一致；
unknown 表示仪器侧不可读。未知不能当作一致。
"""

from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 相对差超过它就算不一致。0.02 = 2% —— 比读数抖动大得多,
#: 比这次事故的 23% 小得多。
DEFAULT_TOLERANCE_FRAC = 0.02


class CheckPiezoRange(BaseSkill):
    """Compare the configured XY half-range against what the scanner reports."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CheckPiezoRange",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读扫描器真实的 XY/Z 压电量程，并与配置里的安全限值（SafetyLimits.xy_max_m）"
                "比对。报 ok / mismatch / unknown —— 从不改变任何东西。换过制冷剂、"
                "换过扫描器、或做过任何一次压电重标定之后都跑一下：配置里的量程比真实量程大，会让选点器提出针尖根本到不了的目标，"
                "接着那次移动会以一个重试也修不好的超时告终。"
            ),
            parameters=[
                ParameterSpec(
                    name="tolerance_frac",
                    type="float",
                    description=(
                        "相对差超过此值时报告 mismatch。默认 0.02（2%）为工程比较容差，使用前应按当前读数精度与配置要求验证。"
                    ),
                    required=False,
                    default=DEFAULT_TOLERANCE_FRAC,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            tags=["piezo", "range", "safety", "read", "selfcheck"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        tol = float(params.get("tolerance_frac", DEFAULT_TOLERANCE_FRAC))
        calls: list = []

        # ── 仪器那边 ────────────────────────────────────────────────────
        rec = context.safe_call("Piezo_RangeGet")
        calls.append(rec)
        inst_half_x = inst_half_y = inst_half_z = None
        if not rec.error:
            # 复用解码器读取协议信封中 parsed 列表的数值。
            # 仅扫描顶层会漏掉范围数据，使自检无法回答问题。
            from mast.skills.builtins.readback import _decode_nanonis

            decoded = _decode_nanonis(getattr(rec, "return_value", None))
            vals = [float(v) for v in (decoded or ())
                    if isinstance(v, (int, float))] if isinstance(
                        decoded, (list, tuple)) else []
            # Piezo_RangeGet 回 (x_range, y_range, z_range) —— **全程**,不是半程。
            if len(vals) >= 2:
                inst_half_x = abs(float(vals[0])) / 2.0
                inst_half_y = abs(float(vals[1])) / 2.0
            if len(vals) >= 3:
                inst_half_z = abs(float(vals[2])) / 2.0

        # ── 配置那边 ────────────────────────────────────────────────────
        cfg_half = None
        cfg_err = ""
        # 配置侧通过 core.safety._get_effective_limits 获取配置、管理员覆盖与硬件收紧后的有效值。
        # 直接读取类默认值会绕过覆盖，使核对对象偏离实际生效值。
        try:
            from mast.config import SafetyLimits
            from mast.core.safety import _get_effective_limits

            # 与 ``api/safety_view.py:80`` / ``core/runtime.py:61`` **同一个用法**:
            # 类默认值 → 合并管理员覆写 → 按仪器事实收紧。
            # 直接读 ``SafetyLimits().xy_max_m`` 会**绕过覆写**,报出来的数就不是
            # 实际生效的那个 —— 一个对账工具报错数字比不对账更坏。
            lim = _get_effective_limits(SafetyLimits())
            v = getattr(lim, "xy_max_m", None)
            if v:
                cfg_half = abs(float(v))
            else:
                cfg_err = "生效的 SafetyLimits 里没有 xy_max_m"
        except Exception as exc:  # noqa: BLE001
            cfg_err = f"{type(exc).__name__}: {exc}"
            logger.debug("CheckPiezoRange: 读不到生效的 SafetyLimits: %s", exc)

        data = {
            "instrument_half_x_m": inst_half_x,
            "instrument_half_y_m": inst_half_y,
            "instrument_half_z_m": inst_half_z,
            "configured_half_xy_m": cfg_half,
            "tolerance_frac": tol,
        }

        # ── 三态 ────────────────────────────────────────────────────────
        #
        # 仪器读不到 ⇒ **判不了**。折成「一致」会让这个自检变成一句永远为真的
        # 安慰话 —— 那正是它要防的东西。
        if inst_half_x is None or inst_half_y is None:
            return SkillResult(
                skill_name="CheckPiezoRange", success=True,
                data={**data, "verdict": "unknown",
                      "undecidable": ("读不到扫描器的压电范围"
                                      f"(Piezo_RangeGet: {rec.error or '返回值里没有数字'})"
                                      " —— **判不了,不是「一致」**。")},
                summary="压电范围对账:**判不了** —— 仪器那边读不到。",
                nanonis_calls=calls)
        if cfg_half is None:
            return SkillResult(
                skill_name="CheckPiezoRange", success=True,
                data={**data, "verdict": "unknown",
                      "undecidable": ("读不到 SafetyLimits.xy_max_m"
                                      + (f"({cfg_err})" if cfg_err else "")
                                      + " —— **判不了**。")},
                summary="压电范围对账:**判不了** —— 读不到配置里的 xy_max_m。",
                nanonis_calls=calls)

        inst_half = min(inst_half_x, inst_half_y)   # 两轴取小的,包络要保守
        rel = abs(cfg_half - inst_half) / max(inst_half, 1e-15)
        data["min_instrument_half_m"] = inst_half
        data["relative_difference"] = rel
        data["configured_exceeds_instrument"] = bool(cfg_half > inst_half)

        if rel <= tol:
            return SkillResult(
                skill_name="CheckPiezoRange", success=True,
                data={**data, "verdict": "ok"},
                summary=(f"压电范围对账 **一致**:仪器半程 {inst_half * 1e9:.1f} nm,"
                         f"配置 {cfg_half * 1e9:.1f} nm(差 {rel * 100:.1f}%)。"),
                nanonis_calls=calls)

        worse = cfg_half > inst_half
        return SkillResult(
            skill_name="CheckPiezoRange", success=True,
            data={**data, "verdict": "mismatch"},
            summary=(
                f"压电范围 **对不上**:仪器半程 {inst_half * 1e9:.1f} nm"
                f"(X {inst_half_x * 1e9:.1f} / Y {inst_half_y * 1e9:.1f}),"
                f"而配置 xy_max_m = {cfg_half * 1e9:.1f} nm —— 差 {rel * 100:.1f}%。"
                + (" **配置比仪器大**:选点器会提议针尖到不了的目标,"
                   "移动会以「超时」失败,而重试没有用 —— 换温度/换扫描器/改压电"
                   "标定之后最常见的就是这一种。请把 xy_max_m 改成 "
                   f"≤ {inst_half * 1e9:.0f} nm,或在 Nanonis 里把压电范围调回去。"
                   if worse else
                   " 配置比仪器小 —— 不会撞限位,但会白白浪费可用面积。")),
            nanonis_calls=calls)
