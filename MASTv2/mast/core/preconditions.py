"""Single source of truth for hardware-state precondition vocabulary.

审查 [#86]: the precondition word-lists were duplicated and had
drifted between the two layers that interpret ``SkillMetadata.preconditions``:

  * ``mast.skills.base.BaseSkill.check_preconditions`` used an EXACT-key dict
    (``_PRECONDITION_CHECKS``) mapping a precondition string to a
    ``(HardwareState attr, expected value)`` pair. It knew ``z_controller_off``
    / ``scan_not_running`` but NOT ``bias_nonzero``, and an unknown string was
    reported as "Cannot verify".
  * ``mast.core.safety.SafetyGuard`` / ``mast.agents._shared.safety_mw.SafetyGate``
    used SUBSTRING matching (``"z_controller" in pc and "on" in pc`` …) and knew
    ``bias_nonzero`` but produced different, hand-rolled message strings.

Because the two layers disagreed on which strings are recognised, a skill could
declare a precondition that one layer enforced and the other silently passed
(e.g. ``scan_stopped`` was a real key in base.py but matched NOTHING in the
SafetyGate substring chain, so the GLOBAL agent-path guard never enforced it).

This module is now the ONE place that defines the vocabulary. Both layers import
from here:

  * ``PRECONDITION_CHECKS`` — the exact-name mapping consumed by ``BaseSkill``.
  * ``check_state_preconditions`` — the substring-matching enforcement used by
    ``SafetyGuard`` / ``SafetyGate`` (preserves the existing message strings).

Adding a new precondition = adding ONE entry below; both layers pick it up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle at module load (types imports nothing here)
    from mast.core.types import HardwareState


# ─────────────────────────────────────────────────────────────────────────
# Exact-name vocabulary — consumed by BaseSkill.check_preconditions.
#
# precondition string → (HardwareState attribute, expected value)
# ─────────────────────────────────────────────────────────────────────────
PRECONDITION_CHECKS: dict[str, tuple[str, object]] = {
    "z_controller_on": ("z_controller_on", True),
    "z_controller_off": ("z_controller_on", False),
    "scan_running": ("scan_running", True),
    "scan_stopped": ("scan_running", False),
    "scan_not_running": ("scan_running", False),
}


# 报告布尔前置条件失败时，应附带 PRECONDITION_CONTEXT_FIELDS 中的状态。
# z_controller_on 由状态码是否为 2 得出，False 不能区分 Off、Hold、
# SwitchingOff、SafeTip 与 Withdrawing；z_controller_status 保留这种区别。
PRECONDITION_CONTEXT_FIELDS: dict[str, str] = {
    "z_controller_on": "z_controller_status",
}

#: 布尔为 False 时,各个模块状态各自意味着什么(只用于措辞,不参与判定)。
_ZCTRL_STATUS_HINT: dict[str, str] = {
    "Off": "确实被关掉了",
    "Hold": "被**挂起**(hold)——不是被关掉;挂起它的那一步没有把它放回来",
    "SwitchingOff": "正在关闭中(switch-off delay 未走完)——再等一下可能就好了",
    "SafeTip": "Nanonis 自己进了 **SafeTip 保护态**——没有人关它,是仪器躲开了",
    "Withdrawing": "正在退针",
}


def context_suffix(attr: str, state: "HardwareState") -> str:
    """违反消息的补充说明,例如 ``（Z-Controller 模块状态: SafeTip …）``。

    读不到伴随字段就返回空串 —— 「没读到」不该被写成一个编出来的状态名。
    """
    companion = PRECONDITION_CONTEXT_FIELDS.get(attr)
    if not companion:
        return ""
    value = getattr(state, companion, None)
    if value is None:
        return ""
    text = str(value)
    hint = _ZCTRL_STATUS_HINT.get(text) if attr == "z_controller_on" else None
    if hint:
        return f"（{companion}={text}：{hint}）"
    return f"（{companion}={text}）"


# ─────────────────────────────────────────────────────────────────────────
# Substring-match enforcement — consumed by SafetyGuard / SafetyGate.
#
# Each rule: (name_substrings, attr, bad_value, message). When EVERY substring
# in ``name_substrings`` is present in the lower-cased precondition string, the
# rule applies: if ``getattr(state, attr) == bad_value`` the message is emitted.
# Order matters — first matching rule wins (mirrors the original if/elif chain).
#
# ``bad_value`` is the state value that VIOLATES the precondition.
#
# ⚠️ **这张表只对 :data:`PRECONDITION_CHECKS` 里没有的名字生效**(2026-08-10)。
#
# 为什么必须这样:``"on"`` 是 ``"z_controller_off"`` 的子串 —— c-**on**-troller。
# 于是第一条规则(要求控制器 ON)命中了 ``z_controller_off``,``break`` 掉,
# 第二条规则**从来没有被执行过,是不可达的死代码**。净效果是
# ``z_controller_off`` 被**完全反过来**判:
#
#     硬件 OFF(正是它要求的) → 拒绝,理由还写着「Z controller is OFF」
#     硬件 ON (正是它禁止的) → **放行**
#
# 放行那一侧才是要命的:``MotorMove`` / ``MotorMoveClosedLoop`` 声明的就是
# ``z_controller_off``,而「反馈环还闭着的时候跑开环粗进针马达」正是这条前置存在
# 的全部理由(``zcontrol.py`` 里 TryEngageController 那段逐字写过:那是撞针)。
#
# 精确前置条件应优先于子串兜底，避免状态快照正确而规则反向解释。
#
# 与 ``core/safety.py`` 的 ``_GLOBAL_CHECKS`` 是同一族教训(那次是「避免宽泛的
# 'x'/'z',用 'x_m'/'bias_v'」)。宽泛子串在**否定形式**上尤其危险:
# 一个词的否定词往往包含它自己。
# ─────────────────────────────────────────────────────────────────────────
_SUBSTRING_RULES: list[tuple[tuple[str, ...], str, object, str]] = [
    # 否定形式排在前面:即使有人把精确名单去掉,先匹配 "off" 也比先匹配 "on"
    # 少错一次(深度防御,不是主要防线 —— 主要防线是上面那条「精确名单优先」)。
    (("z_controller", "off"), "z_controller_on", True,
     "{precondition}' — Z controller is ON"),
    (("z_controller", "on"), "z_controller_on", False,
     "{precondition}' — Z controller is OFF"),
    (("scan", "not_running"), "scan_running", True,
     "{precondition}' — scan is running"),
    # NOTE: "scan_stopped" also means "scan must not be running"; it is handled
    # by the not_running rule via the alias check below so the substring chain
    # and the exact-name dict agree.
    (("scan", "stopped"), "scan_running", True,
     "{precondition}' — scan is running"),
    (("scan", "running"), "scan_running", False,
     "{precondition}' — scan is not running"),
    (("bias", "nonzero"), "bias_v", 0.0,
     "{precondition}' — bias is zero"),
    # Tip clearance before a coarse motor move. Added to core/safety.py in the
    # 2026-07-03 review and — despite this module's docstring — never mirrored
    # here, so the two layers that DO read this file (BaseSkill.check_preconditions
    # and, since 2026-07-31, SafetyGuard/SafetyGate) silently passed it (found
    # 2026-07-31).
    #
    # Note what this DOES NOT mean: `withdrawn` is the fine-Z piezo at its high
    # limit, one or two microns. A lateral coarse move needs tens of microns,
    # which only a coarse-Z retract provides — that is RelocateCoarseXY's job,
    # not this precondition's.
    (("withdrawn",), "withdrawn", False,
     "{precondition}' — tip is not withdrawn (retract the tip before a coarse "
     "motor move)"),
    (("tip", "clear"), "withdrawn", False,
     "{precondition}' — tip is not withdrawn (retract the tip before a coarse "
     "motor move)"),
]


# ─────────────────────────────────────────────────────────────────────────
# COMPUTED preconditions — answered by a live subsystem, not a state field.
#
# Everything above compares a HardwareState attribute. Some conditions have no
# such attribute: chamber pressure is polled by the environment monitor, not by
# the Nanonis state snapshot, and its verdict also depends on an operator
# attestation and on per-rig thresholds.
#
# Note the INVERTED failure philosophy. Every substring rule above fails OPEN on
# unknown state (`actual is bad_value` — None never matches), which is right when
# the worst case is a wasted scan. `vacuum_ok_for_coarse` is the opposite: an
# unknown pressure is precisely the condition it exists to refuse, because a
# gauge that cannot see is a gauge in the discharge band or at atmosphere, with
# nothing to tell them apart. Driving a coarse piezo there arcs across the
# stack's insulation, and there is no retry.
# ─────────────────────────────────────────────────────────────────────────

def _vacuum_ok_for_coarse() -> tuple[bool, str]:
    try:
        from mast.core.vacuum_interlock import check as _vac_check

        verdict = _vac_check()
        return bool(verdict.allow), verdict.reason
    except Exception as exc:  # noqa: BLE001 — a broken interlock is a refusal
        return False, f"真空互锁本身不可用({exc}),按 fail-closed 拒绝粗动"


#: precondition name (exact, lower-cased) → () -> (ok, reason)
COMPUTED_CHECKS: dict[str, object] = {
    "vacuum_ok_for_coarse": _vacuum_ok_for_coarse,
}


#: ``(attr, expected)`` → 人话。措辞与旧的子串模板逐字相同,好让既有断言
#: (``"Z controller is OFF" in v[0]``)继续通过 —— 这次改的是**判定**,不是措辞。
_EXACT_MESSAGES: dict[tuple, str] = {
    ("z_controller_on", True): "Z controller is OFF",
    ("z_controller_on", False): "Z controller is ON",
    ("scan_running", True): "scan is not running",
    ("scan_running", False): "scan is running",
}


def check_state_preconditions(
    preconditions: list[str], state: "HardwareState"
) -> list[str]:
    """Enforcement shared by SafetyGuard and SafetyGate.

    Returns a list of violation messages (empty = all preconditions met). The
    message strings keep the pre-refactor SafetyGuard wording (so existing
    assertions like ``"Z controller is OFF" in v[0]`` keep passing), plus the
    actual/expected pair and any companion field.

    **精确名单优先,子串规则只兜底**(2026-08-10)。在此之前这一层**只有**子串规则,
    而 ``"on"`` 是 ``"z_controller_off"`` 的子串,于是这条前置被完全反过来判 ——
    见 :data:`_SUBSTRING_RULES` 上面那段。两层现在用**同一张精确表**做判定,
    子串只负责本模块 docstring 说的那件事:接住不在精确表里、但仍然认得出来的名字。
    """
    violations: list[str] = []
    for precondition in preconditions:
        pc = precondition.lower()
        computed = COMPUTED_CHECKS.get(pc)
        if computed is not None:
            ok, reason = computed()  # type: ignore[operator]
            if not ok:
                violations.append(f"Precondition failed: '{precondition}' — {reason}")
            continue
        exact = PRECONDITION_CHECKS.get(pc) or PRECONDITION_CHECKS.get(precondition)
        if exact is not None:
            attr, expected = exact
            actual = getattr(state, attr, None)
            # ``None`` = 读不到 ⇒ 放行(与子串层历来的 fail-open 一致:
            # 未知状态下拦住可能让用户在针要撞上去时按不动按钮)。
            if actual is not None and actual != expected:
                violations.append(
                    f"Precondition failed: '{precondition}' — "
                    + _EXACT_MESSAGES.get(
                        (attr, bool(expected) if isinstance(expected, bool) else expected),
                        f"{attr} is {actual!r}, expected {expected!r}")
                    + f"（读到 {attr}={actual!r}，要求 {expected!r}）"
                    + context_suffix(attr, state)
                )
            continue
        for substrings, attr, bad_value, template in _SUBSTRING_RULES:
            if all(s in pc for s in substrings):
                actual = getattr(state, attr, None)
                if attr == "bias_v":
                    # bias_nonzero: fails only on an explicit 0.0, never on
                    # unknown (None) — mirrors the original `is not None and == 0`.
                    violated = actual is not None and actual == bad_value
                else:
                    # boolean state attrs: identity match, exactly like the
                    # original `state.x is True/False` (so None/unknown passes).
                    violated = actual is bad_value
                if violated:
                    # 伴随字段必须两层都印 —— 本模块的开头写着两层曾经各写一份而
                    # 悄悄漂移。只在其中一层加,下次仍然会有一半现场看到裸布尔。
                    violations.append(
                        "Precondition failed: '"
                        + template.format(precondition=precondition)
                        + context_suffix(attr, state)
                    )
                break
    return violations


def precondition_recognized(name: str) -> bool:
    """True when *name* is a precondition this module can evaluate — either an
    exact-name key (``PRECONDITION_CHECKS``) or a substring-rule match
    (``_SUBSTRING_RULES``).

    Lets ``BaseSkill.check_preconditions`` tell a genuinely-unknown precondition
    (→ "Cannot verify") apart from a recognised one that is simply satisfied
    (→ pass). Without this, ``bias_nonzero`` — which lives only in the substring
    rules, NOT the exact-name dict — was reported as "Cannot verify" on every
    approach even when the bias was non-zero (2026-07-06 cross-machine test)."""
    if name in PRECONDITION_CHECKS:
        return True
    pc = name.lower()
    if pc in COMPUTED_CHECKS:
        return True
    return any(all(s in pc for s in subs) for subs, *_ in _SUBSTRING_RULES)


__all__ = [
    "PRECONDITION_CHECKS",
    "PRECONDITION_CONTEXT_FIELDS",
    "context_suffix",
    "check_state_preconditions",
    "precondition_recognized",
]
