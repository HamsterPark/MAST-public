"""统一判断哪些操作需要执行留痕与通知。

该判据供 auto_approval_mw、skill_adapter、executor 和 execution_context 共用，
避免通知、记录和执行策略各自漂移。当前策略以执行、留痕和通知处理这些操作，
记录中的审批语义为已执行并通知。

拒绝型防护独立执行：粗动逼近硬闸、驱动电压限制、保护禁用与标定变更限制、
未保护横移、SafetyGate、针尖包络和 bias_nonzero 等前置条件不由本模块放行。
改变交互批准策略时，应评估具体风险、现有硬限制覆盖和误报代价。"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: 诊断台账里这条改动的落点。与 ``core.diagnostics.Kind`` 里的字面量同名 ——
#: 「本来会拦我几次」问的就是这个 kind。
DIAG_KIND = "notice_only"

#: ``approvals.approver_kind`` 的取值。schema 的 CHECK 只认三个值
#: (``human_operator`` / ``human_pi`` / ``automated_policy``),而这里既没有人点过
#: 按钮、也不该假装有人点过 —— 是**一条用户立下的常规策略**在放行。
AUTO_APPROVER_KIND = "automated_policy"

#: ``approval_method``(自由文本列)。写成一句能被 grep 到的话,而不是复用
#: ``policy_rule_v1`` —— 审计的人要能一眼分出「按策略自动放行并通知」和
#: 「有人点了批准」。
AUTO_APPROVAL_METHOD = "auto_executed_notified"

#: ``approver_id``。这台机器上没有 per-user 认证,写 ``operator`` 是如实,
#: 编一个 user id 不是。
AUTO_APPROVER_ID = "operator"


def _live_mode() -> Any:
    """当前操作模式;读不到返回 ``None``(= 不做模式相关的判断)。"""
    try:
        from mast.core.operating_mode import current_operating_mode

        return current_operating_mode()
    except Exception:  # noqa: BLE001 — 判据在热路径上,绝不抛
        return None


def would_have_asked(
    meta: Any,
    *,
    tool_name: str = "",
    args: "dict | None" = None,
    mode: Any = None,
) -> str | None:
    """这一步在 ⑰ 之前会不会停下来等人批准?会 → 返回一句中文理由;不会 → ``None``。

    Parameters
    ----------
    meta:
        ``SkillMetadata``(或任何有 ``safety_level`` / ``capabilities`` 的对象)。
        ``None`` ⇒ 不是技能工具,一律 ``None``。
    tool_name / args:
        用于电脉冲判据(``is_electrical_pulse`` 要看参数:带电压的 TipShape 也算)。
        缺省时退回 ``meta.name`` / 空参数。
    mode:
        操作模式。``None`` ⇒ 读实时全局值,这样中间件(有 ``get_mode``)和
        skill_adapter(没有)得到的是**同一个答案**。

    永不抛:判据坏掉的失败模式是「没有通知」,不是「技能跑不了」。
    """
    if meta is None:
        return None
    try:
        from mast.core.types import SafetyLevel

        if getattr(meta, "safety_level", None) == SafetyLevel.DANGEROUS:
            # 旧路径:``instrument_control/graph._derive_hitl_map`` 把每个
            # safety_level=DANGEROUS 的技能挂进 HumanInTheLoopMiddleware。
            return f"DANGEROUS 技能({getattr(meta, 'name', '') or tool_name})"
    except Exception:  # noqa: BLE001
        logger.debug("auto_approval: 读 safety_level 失败", exc_info=True)

    try:
        from mast.core.safety import is_electrical_pulse
        from mast.core.types import OperatingMode

        eff_mode = mode if mode is not None else _live_mode()
        if eff_mode is not None and OperatingMode.coerce(eff_mode) is OperatingMode.SEMI:
            name = tool_name or str(getattr(meta, "name", "") or "")
            caps = getattr(meta, "capabilities", None) or frozenset()
            if is_electrical_pulse(name, args or {}, caps):
                # 旧路径:``ModeGatedPulseHITLMiddleware``(SEMI 专用)。
                return f"半自动模式下的电脉冲({name})"
    except Exception:  # noqa: BLE001
        logger.debug("auto_approval: 电脉冲判据不可用", exc_info=True)
    return None


def notify(subject: str, reason: str, **fields: Any) -> None:
    """把「本来会在这里等人」写进诊断台账 + 日志。永不抛、永不阻塞。

    这是**通知**那一半的落点。记录在 ``artifacts/diagnostics/refusals.jsonl``
    与诊断面板里,可以按 ``kind=notice_only`` 查 —— 验收这条改动时该问的
    「本来会拦我几次」就是这个查询。
    """
    logger.warning("auto_approval 放行并通知:%s —— %s", subject, reason)
    try:
        from mast.core.diagnostics import record as _diag

        _diag(DIAG_KIND, subject, reason, **fields)
    except Exception:  # noqa: BLE001 — 台账写不进去绝不能反噬技能执行
        logger.debug("auto_approval: 诊断台账写入失败", exc_info=True)


def record_auto_approval(repos: Any, action_id: str, *, skill: str,
                         reason: str, params: "dict | None" = None) -> str | None:
    """给一条**已经执行了**的动作补一行 ``approvals``,语义 = 已执行并通知。

    为什么还要写这张表:审批链路割掉之后,``approvals`` 会变成一张只有历史行的
    死表 —— 而事后查「这个 DANGEROUS 动作是谁准的」时,**空表和「没人准过」长得
    一模一样**。2026-07-27 的取证正好栽在这个形状上(11 个 CONFIRM 级动作跑过,
    approvals 0 行,无从判断有没有人批过)。所以照写,只是 ``approver_kind`` 如实
    写成 ``automated_policy``、``approval_method`` 写成 ``auto_executed_notified``。

    Best-effort:审计写失败绝不能反噬已经发生的动作,失败记 warning(不是 debug)
    —— 悄无声息的审计缺口正是上面那次取证拖了那么久的原因。
    """
    if repos is None or not action_id:
        return None
    try:
        svc = getattr(repos, "approvals", None)
        if svc is None:
            return None
        import json as _json

        evidence = _json.dumps(
            {
                "verdict": "auto_executed",
                "skill": skill,
                "params": params or {},
                "why": reason,
                "policy": ("确认框/审批整条链路改为"
                           "只提醒不阻碍(见 core/auto_approval.py)"),
            },
            ensure_ascii=False, default=str,
        )[:4000]
        return svc.issue(
            action_id=action_id,
            approver_id=AUTO_APPROVER_ID,
            approver_kind=AUTO_APPROVER_KIND,
            approval_method=AUTO_APPROVAL_METHOD,
            approval_evidence=evidence,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("auto_approval: approvals 审计写入失败 action=%s: %s",
                       action_id, exc)
        return None


__all__ = [
    "AUTO_APPROVAL_METHOD",
    "AUTO_APPROVER_ID",
    "AUTO_APPROVER_KIND",
    "DIAG_KIND",
    "notify",
    "record_auto_approval",
    "would_have_asked",
]
