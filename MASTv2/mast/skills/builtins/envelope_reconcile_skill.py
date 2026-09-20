"""核对并收紧软件安全范围，使其与当前硬件量程一致。

配置值可能与温度、扫描器或标定变化后的硬件读数不同。
此技能只调整配置，不写仪器；保留边界符号，仅允许收紧，拒绝扩大范围。
apply=False 时只报告拟议变更。读不到硬件范围时报告未知，不能当作一致。
"""

from __future__ import annotations

import logging
import math

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 覆写文件名 —— 与 ``api/routes/instrument_init.py`` 的 ``_OVERRIDE_STORES`` 同一份。
OVERRIDE_FILE = "safety_limits.json"


class ReconcileSafetyEnvelope(BaseSkill):
    """Reconcile the configured safety envelope against measured travel."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ReconcileSafetyEnvelope",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # CONFIRM 而不是 DANGEROUS：它**只能收紧**，而收紧是让包络更接近硬件
            # 真值。放宽被硬拒，所以最坏结果是「更保守」，不是「没有限制」。
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把配置里的安全包络（SafetyLimits）与**实测**压电行程对账，并"
                "**只收紧地**更新它。默认 apply=False 只报告不写。配置比硬件宽 ⇒ "
                "收紧到实测值（那条上限本来什么都没拦住）；配置比硬件窄 ⇒ "
                "**拒绝并报出来**，放宽是人的决定，不是自动化的。换制冷剂、换扫描器、"
                "重标定压电之后都该跑一次。它不碰仪器的任何设置，只改 MAST 这一侧。"
            ),
            parameters=[
                ParameterSpec(
                    name="apply", type="bool", required=False, default=False,
                    description=(
                        "True 才真的写。默认 False = 只报告 —— 先看清楚要改什么，"
                        "再决定改不改。"),
                ),
                ParameterSpec(
                    name="fields", type="str", required=False, default="",
                    description=(
                        "只处理这几个字段，逗号分隔（如 xy_max_m,xy_min_m）。"
                        "留空 = 全部对得上账的字段。"),
                ),
            ],
            preconditions=[],
            estimated_duration_s=5.0,
            composition_level=0,
            tags=["safety", "envelope", "piezo", "range", "calibration", "selfcheck"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        do_apply = bool(params.get("apply", False))
        raw_fields = str(params.get("fields") or "")
        only = {s.strip() for s in raw_fields.split(",") if s.strip()}

        # ── 对账（复用共用件，不写第二份比对逻辑）──────────────────────────
        try:
            from mast.config import SafetyLimits
            from mast.core.envelope_reconcile import WIDER, reconcile_envelope
            from mast.core.safety import _get_effective_limits
        except Exception as exc:  # noqa: BLE001
            return SkillResult(
                skill_name="ReconcileSafetyEnvelope", success=False,
                error="依赖缺席: %s: %s" % (type(exc).__name__, exc),
                data={"verdict": "unknown"})

        try:
            limits = _get_effective_limits(SafetyLimits())
            rec = reconcile_envelope(context.safe_call, limits)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(
                skill_name="ReconcileSafetyEnvelope", success=False,
                error="对账失败: %s: %s" % (type(exc).__name__, exc),
                data={"verdict": "unknown"})

        findings = [f for f in rec.findings if not only or f.field_name in only]
        wider = [f for f in findings if f.direction == WIDER]
        narrower = [f for f in findings if f.direction != WIDER]

        # ⚠️ **拒绝，不夹紧。** 放宽有正当理由，但那是人的决定；这里悄悄夹一下，
        # 就等于「我要求 A，系统给了我 B」。
        refused = [
            {"field": f.field_name, "configured": f.configured,
             "measured": f.measured, "ratio": f.ratio,
             "reason": ("配置比实测**窄** —— 改它是**放宽**包络，本技能只收紧。"
                        "要放宽请人去初始化页确认。")}
            for f in narrower
        ]

        plan: dict = {}
        for f in wider:
            # 按**幅度**收紧并保留符号 —— 下界是负的（xy_min_m / z_min_m），
            # 直接写 measured 会把 −1.5 µm 变成 +1.2 µm。
            # ``_compare`` 的自述写明它按幅度判宽窄，方向不由符号决定。
            sign_from = f.configured if f.configured else 1.0
            plan[f.field_name] = math.copysign(abs(f.measured), sign_from)

        data = {
            "verdict": ("ok" if not findings and not rec.unreadable
                        else "mismatch" if findings else "unknown"),
            "applied": False,
            "summary_cn": rec.summary(),
            "findings": [f.describe() for f in findings],
            "unreadable": list(rec.unreadable),
            "would_tighten": dict(plan),
            "refused_widen": refused,
            "measured": dict(rec.measured),
        }

        if not do_apply:
            note = "只报告未写（apply=False）。" if plan else "没有需要收紧的项。"
            return SkillResult(
                skill_name="ReconcileSafetyEnvelope", success=True,
                data=data, summary=note + rec.summary())

        if not plan:
            return SkillResult(
                skill_name="ReconcileSafetyEnvelope", success=True, data=data,
                summary="没有需要收紧的项，什么都没写。" + rec.summary())

        # ── 写入：整份合并再存（覆写文件是整份替换）────────────────────────
        try:
            from mast.admin.override_store import ConfigOverrideRegistry

            reg = ConfigOverrideRegistry.get()
            merged = dict(reg.get_raw(OVERRIDE_FILE) or {})
            merged.update(plan)
            reg.save_and_reload(OVERRIDE_FILE, merged)
        except Exception as exc:  # noqa: BLE001
            data["error"] = "%s: %s" % (type(exc).__name__, exc)
            return SkillResult(
                skill_name="ReconcileSafetyEnvelope", success=False,
                error="写入失败: %s: %s" % (type(exc).__name__, exc), data=data)

        # ── 回读比对：说「写进去了」之前先看一眼 ───────────────────────────
        readback: dict = {}
        try:
            fresh = _get_effective_limits(SafetyLimits())
            for k, want in plan.items():
                got = getattr(fresh, k, None)
                ok = (got is not None
                      and abs(float(got) - float(want)) <= abs(float(want)) * 1e-6)
                readback[k] = {"requested": want, "stored": got, "match": bool(ok)}
        except Exception as exc:  # noqa: BLE001
            readback["_error"] = "%s: %s" % (type(exc).__name__, exc)

        bad = [k for k, v in readback.items()
               if isinstance(v, dict) and not v.get("match")]
        data["applied"] = True
        data["readback"] = readback
        return SkillResult(
            skill_name="ReconcileSafetyEnvelope",
            success=not bad,
            error=("回读对不上: " + ", ".join(bad)) if bad else "",
            data=data,
            summary=("收紧了 %d 项%s；拒绝放宽 %d 项。"
                     % (len(plan),
                        "（回读全部一致）" if not bad else "（**回读对不上**）",
                        len(refused))),
        )
