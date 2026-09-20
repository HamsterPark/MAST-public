"""「哪些技能会在执行时发一条『本来会在这里等你批准』的通知」。

为什么它从 ``instrument_control/graph.py`` 搬了出来（2026-08-27 抽出）
---------------------------------------------------------------------
这个判定**一行 langgraph 都没有**：它走一遍技能注册表，问每个技能的
``metadata().safety_level`` 是不是 DANGEROUS。它住在 IC 的图文件里，只是因为那张图
是第一个需要它的人。代价在退出 langgraph 时现形：新运行时的 ``ic_assembly.py``
为了拿这个判定，得 import 一个即将被删的图模块。

抄第二遍在这里比别处更危险 —— 判据抄漏一个技能的症状是「本来该通知的没通知」，
而那种漏法在日志里长得和「今天没有 DANGEROUS 技能」一模一样。

单一真源 = 技能自己的 ``safety_level``
--------------------------------------
SINGLE SOURCE OF TRUTH = each skill's ``metadata().safety_level``. Every skill
whose safety_level is DANGEROUS is gated; the allowed operator decisions come
from :data:`HITL_DECISION_OVERRIDES` (narrowed for specific skills) or default to
approve/edit/reject. This replaces the old hardcoded ``_DANGEROUS_HITL_MAP``,
which could silently drift from the metadata (a skill demoted to CONFIRM in its
module while the map still gated it, or a skill promoted to DANGEROUS that the
map never picked up). Deriving means a metadata flip is the ONLY edit needed to
gate/ungate a skill, and a parity test pins the two together.

⑰（2026-08-08）审批框割掉之后这个函数换了工作
---------------------------------------------
之前这是 ``HumanInTheLoopMiddleware`` 的 ``interrupt_on`` 表 —— 表里的技能会弹审批
框。现在没有框了。**判定「哪些技能算 DANGEROUS」的判据一个字没改**，只是下游从
「拦住等人」变成了「照跑并通知」。真正在运行时判的是
``mast.core.auto_approval.would_have_asked``（同一个 ``safety_level`` 判据）；这个
函数现在只用于**启动时把名单打进日志** —— 「有几个技能进了这条通道」是验收
那次改动的第一个问题，而一条启动日志比翻代码便宜。
"""
from __future__ import annotations

import logging

from mast.core.types import SafetyLevel

logger = logging.getLogger(__name__)


# ⑰(2026-08-08)：**这张表已经没有消费者了，但不要删。**
#
# 它曾经窄化某个 DANGEROUS 技能在审批框里允许的操作（批准/编辑/拒绝）。审批框割掉
# 之后，「允许哪些裁决」这个问题不再存在。留着是因为它是本模块的形状说明，而
# ``derive_hitl_map`` 现在回答的是另一个问题：**哪些技能在执行时要发通知**。
#
# 下面这段历史保留原样 —— 它记着一次「注释比代码更权威」的事故。
#
# Per-skill allowed_decisions OVERRIDES for the HITL gate. Skill metadata's
# safety_level (== DANGEROUS) is the SOURCE OF TRUTH for which skills are
# HITL-gated — derive_hitl_map() walks the registry and gates every DANGEROUS
# skill. This table only narrows the operator decisions for specific skills.
#
# 2026-06-11 safety re-scoping: the ONLY physically-dangerous action is an
# open-loop coarse Z step TOWARD the sample (pan-type stepper — no feedback
# stop). That is NOT a static-safety_level skill but a PARAMETER condition
# (MotorMove direction='z-approach'), gated by SafetyGateMiddleware's
# coarse-approach layer (fail-closed block on the autonomous agent path) and by
# SkillExecutor (human approval on the manual path) — see
# mast.core.safety.is_coarse_sample_approach. The classic hardware verbs
# (BiasPulse / MotorMove / EmergencyRetract / TipShape) were re-scoped to
# AUTO/CONFIRM in 2026-06-11 because Nanonis bounds them.
#
# CORRECTION (2026-07-28): the sentence that used to stand here — "NO builtin
# skill carries safety_level=DANGEROUS anymore: derive_hitl_map() returns an
# empty map and the HITL middleware is a no-op" — has been FALSE for a while,
# and the same claim propagated into logging/v2/policy.py and the dispatch
# walkthrough. Auto-discovery over builtins+composite currently yields TEN
# DANGEROUS skills (LockNanonisUI, QuitNanonis, LoadNanonisScript,
# LoadMultiPassConfig, MoveProbeXY, SetLaserOnOff, SetPiControllerOnOff,
# StartRfGenerator, RunRfFrequencySweep, CreateZCtrlPreset), so the map is
# non-empty and HumanInTheLoopMiddleware IS mounted in production. Say so,
# because "a gate nobody knows is armed" is how an interrupt path goes untested.
#
# CreateZCtrlPreset (2026-08-03) is the newest, and it is DANGEROUS for a reason
# worth stating: it touches no hardware at all. What it writes is a value every
# future ApplyZCtrlPreset of that group will trust, so the blast radius is every
# later use, not this call. The approval card is also a build-then-verify step —
# it renders the arguments AS PARSED, so a number
# that arrived corrupted is visible there and can be edited or rejected before it
# is ever stored.
#
# ⑰(2026-08-08)**上面这段最后一句现在是历史，不是现状，而且它记的是一次真实的
# 能力削弱** —— 那张卡没了，所以「存进去之前先看一眼」也没了。替代是事后的：
# AutoApprovalNoticeMiddleware 把同一份解析后参数写进诊断台账（notice_only）。
# 看得见，但在存进去之后。
#
# 要把「存之前先看一眼」拿回来，**正确的做法不是把审批框接回来**，而是给这个值配
# 一条拒绝型的量级判据 —— 与丢指数事故的其它五层防线同款（参数组 + SI 前缀 +
# SafetyGate 量级提示…）。那样它不弹框也能拦，而且拦得比人眼稳。
#
# This table stays as the documented hook for narrowing the decisions offered
# for a particular DANGEROUS skill.
HITL_DECISION_OVERRIDES: dict[str, list[str]] = {}

#: Default decisions for a DANGEROUS skill with no explicit override.
DEFAULT_HITL_DECISIONS = ["approve", "edit", "reject"]


def derive_hitl_map(registry, *, owner: str = "instrument_control") -> dict[str, dict]:
    """走一遍注册表，挑出 ``safety_level == DANGEROUS`` 的技能。

    ``owner`` 只进日志行 —— 两条驱动链（旧图 / 新运行时的 ``ic_assembly``）会各调
    一次，而「同一份名单被谁问过」在排查时是有用的。判定本身与调用方无关。

    注册表读不出来时**返回空表并 warning**，不抛：这个函数只喂一条启动日志，让它
    把整台机器的装配拦下来是不成比例的。真正在运行时判 DANGEROUS 的是
    ``mast.core.auto_approval.would_have_asked``，它不经过这里。
    """
    out: dict[str, dict] = {}
    try:
        skills = registry.list_skills()
    except Exception as exc:  # pragma: no cover — registry is built by us
        logger.warning("derive_hitl_map: registry.list_skills() failed: %s", exc)
        return out
    for meta in skills:
        if getattr(meta, "safety_level", None) == SafetyLevel.DANGEROUS:
            decisions = HITL_DECISION_OVERRIDES.get(meta.name, DEFAULT_HITL_DECISIONS)
            out[meta.name] = {"allowed_decisions": list(decisions)}
    logger.info("%s: derived HITL gate for %d DANGEROUS skill(s): %s",
                owner, len(out), sorted(out.keys()))
    return out


__all__ = ["derive_hitl_map", "HITL_DECISION_OVERRIDES", "DEFAULT_HITL_DECISIONS"]
