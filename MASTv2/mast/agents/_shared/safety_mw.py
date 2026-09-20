"""SafetyGateMiddleware — global hardware-bounds layer on top of per-skill validation.

Port of v1 mast/core/safety.py:SafetyGuard wrapped as a LangChain
AgentMiddleware. Three checks (compass §4.4, "ports the existing three-layer
SafetyGuard"):

  1. **Global parameter bounds** — match parameter (name, unit) against
     `_GLOBAL_CHECKS`, then validate against SafetyLimits. E.g., any param
     named like 'bias_v' with unit 'V' must be in [bias_min_v, bias_max_v]
     regardless of which skill is being invoked.

  2. **State preconditions** — match `meta.preconditions` strings against the
     latest HardwareState (z_controller_on, scan_running, etc.). If state
     provider is None, this layer is skipped.

  3. **Admin override merge** — at construction time, merge JSON overrides
     from ConfigOverrideRegistry on top of code defaults. Reload via
     `mw.reload_overrides()`. R15 risk mitigation: admin GUI changes must
     stay effective in v2.

The per-skill bounds (ParameterSpec.min_value/max_value) and per-skill
preconditions are still enforced inside `wrap_skill` (calling
BaseSkill.validate_params + BaseSkill.check_preconditions). This middleware
adds the GLOBAL layer on top, plus admin override.

This middleware does NOT block on `safety_level` alone — it never has. A
DANGEROUS skill used to be routed to a human approval dialog by langchain's
`HumanInTheLoopMiddleware`; since ⑰ (2026-08-08) that dialog is gone and the
same skills are announced instead, by
`agents/_shared/auto_approval_mw.AutoApprovalNoticeMiddleware`. Compose them:
`[SafetyGateMiddleware(...), AutoApprovalNoticeMiddleware(...)]` — the notice
sits INSIDE the gate, because "not waiting for approval, running now" must not
be said about a call this gate is about to refuse.

**Everything in this file is refusal-type and none of it changed in ⑰**: it
does not pop a dialog and does not wait for anybody; it says no at the boundary
and explains why. Over the two field nights that produced ⑰ this layer was the
one that actually stopped something dangerous, and it did so with zero false
positives.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

from mast.admin.override_store import ConfigOverrideRegistry
from mast.config import SafetyLimits
from mast.core.si_quantity import SIParseError, parse_quantity
from mast.core.types import HardwareState, OperatingMode, SafetyLevel, SkillMetadata

logger = logging.getLogger(__name__)

#: the same relative slack core.safety applies: '100n' is 1.0000000000000001e-07 and must
#: clear a 1e-07 ceiling (2026-09-11, STM-Bench trials)
_FLOAT_SLACK = 1e-9


# ─────────────────────────────────────────────────────────────────────
# _GLOBAL_CHECKS — SINGLE SOURCE OF TRUTH.
# Previously this was a verbatim copy of v1 mast/core/safety.py:13-24, but the
# copy drifted: core/safety.py later added the scan-extent caps
# (width_m / height_m → scan_size_max_m, 10 µm) after a "width_m=2 (2 metres!)"
# incident, and this copy was never resynced. Since the v2 AGENT path uses THIS
# list (SafetyGateMiddleware), the cap silently vanished on the live path while
# the admin GUI still displayed it as active.
# Importing the canonical object makes divergence impossible; a parity test pins
# safety_mw._GLOBAL_CHECKS is core.safety._GLOBAL_CHECKS.
# ─────────────────────────────────────────────────────────────────────

from mast.core.safety_escalation import (  # noqa: E402
    active_approach_refusal,
    is_approach_escalation,
)
from mast.core.safety import (  # noqa: E402
    _GLOBAL_CHECKS,
    _valid_check_item,
    check_state_preconditions as _core_check_state_preconditions,
    is_calibration_change,
    is_coarse_drive_change,
    is_coarse_sample_approach,
    is_protection_disable,
    is_unguarded_lateral_coarse_move,
    mode_refusal,
    physically_absurd_violations,
)


# ─────────────────────────────────────────────────────────────────────
# Pure SafetyGuard logic (port of v1 SafetyGuard, decoupled from middleware)
# ─────────────────────────────────────────────────────────────────────

class SafetyGate:
    """3-layer safety guard with admin override. Pure logic, no LangChain dep.

    Mirrors v1 mast/core/safety.py:SafetyGuard one-to-one. Tested standalone in
    tests/v2/unit/test_safety_mw.py.
    """

    def __init__(
        self,
        limits: SafetyLimits,
        registry: ConfigOverrideRegistry | None = None,
    ):
        self._raw_limits = limits
        self._registry = registry
        self._reload()

    def _reload(self) -> None:
        """Recompute effective limits + checks from raw + admin overrides."""
        self._limits = self._effective_limits(self._raw_limits)
        self._checks = self._effective_checks()
        # Pre-resolve limit values for each check to avoid repeated getattr at run
        # time. Defence-in-depth (mirrors core.safety.SafetyGuard.__init__): a
        # single check whose limit attr is missing on the limits object is dropped
        # with a warning rather than aborting the whole SafetyGate construction —
        # which on the LIVE agent path would crash instrument_control's build()
        # and silently take the executor offline (security 审查, #1).
        #
        # The SafetyLimits field NAMES are carried alongside the resolved values
        # (6-tuple, not v1's 4-tuple) so a rejection can tell the operator WHICH
        # knob to turn — "widen xy_max_m" is actionable, "the envelope is too
        # small" is not ().
        self._resolved_checks: list[tuple[str, str, float, float, str, str]] = []
        for pat, unit_pat, min_attr, max_attr in self._checks:
            try:
                gmin = getattr(self._limits, min_attr)
                gmax = getattr(self._limits, max_attr)
            except AttributeError:
                logger.warning(
                    "Skipping safety check %r: limit attr missing "
                    "(min_attr=%r, max_attr=%r)", pat, min_attr, max_attr,
                )
                continue
            self._resolved_checks.append(
                (pat, unit_pat, gmin, gmax, min_attr, max_attr))

    def reload(self) -> None:
        """Public hot-reload API — call after admin GUI saves."""
        if self._registry is not None:
            self._registry.reload()
        self._reload()

    # ── Effective config (merge raw + admin override) ────────────────

    def _effective_limits(self, limits: SafetyLimits) -> SafetyLimits:
        """Vendored from v1 _get_effective_limits, plus the instrument clamp.

        两段，顺序要紧：

        1. **管理员覆写**（只在注入了 registry 时；没注入就是「不合并」，这是既有
           契约，测试依赖它）。
        2. **按已登记的仪器事实收紧** —— 与 registry 无关，所以**无论有没有 registry
           都跑**。加这一段是因为 agent 侧原来完全没有它：手动路径经
           ``core.safety._get_effective_limits`` 收紧，agent 路径走的是本函数，
           两条路对「前放只有 ±10 nA」这件事的认知不一致（KNOWN_ISSUES §2.16）。
           收紧只有一个方向，所以它永远不会放宽任何一条线。
        """
        if self._registry is not None:
            try:
                ovr = self._registry.get_safety_limits()
                if ovr:
                    limits = limits.model_copy(update=ovr)
            except Exception as e:
                logger.warning("admin SafetyLimits override merge failed: %s", e)
        try:
            from mast.core.safety import clamp_to_instrument_facts

            limits, _notes = clamp_to_instrument_facts(limits)
        except Exception as e:  # noqa: BLE001 — 收紧失败绝不能让闸门建不起来
            logger.warning("instrument-fact clamp failed: %s", e)
        return limits

    def _effective_checks(self) -> list[tuple[str, str, str, str]]:
        """Vendored from v1 _get_effective_checks."""
        if self._registry is None:
            return list(_GLOBAL_CHECKS)
        try:
            ovr = self._registry.get_safety_checks()
        except Exception:
            return list(_GLOBAL_CHECKS)
        if not ovr:
            return list(_GLOBAL_CHECKS)
        checks = list(_GLOBAL_CHECKS)
        # Removals (drop entries by pattern name)
        removals = set(ovr.get("removals", []))
        checks = [c for c in checks if c[0] not in removals]
        # Replacements — a malformed override (typo'd min_attr/max_attr, missing
        # keys, non-str fields) is dropped with a warning and the built-in check
        # left intact, rather than crashing SafetyGate construction. Validated via
        # the same single-source helper used by core.safety (security ).
        for item in ovr.get("overrides", []) or []:
            valid = _valid_check_item(item)
            if valid is None:
                continue
            checks = [valid if c[0] == valid[0] else c for c in checks]
        # Additions (validated)
        for item in ovr.get("additions", []) or []:
            valid = _valid_check_item(item)
            if valid is not None:
                checks.append(valid)
        return checks

    # ── Layer 1: Global parameter bounds ─────────────────────────────

    def check_global_bounds(
        self, meta: SkillMetadata, params: dict
    ) -> list[str]:
        """Run only the GLOBAL bounds check (per-skill bounds are wrap_skill's job).

        Vendored from v1 SafetyGuard.check_parameter_bounds, but only the
        global limits portion (lines 110-123 of v1 file). Per-skill bounds in
        ParameterSpec.min_value/max_value are validated by BaseSkill in v2's
        wrap_skill.
        """
        # #118 deepening: catch physically-impossible magnitudes (e.g. a 1.5 A
        # tunnelling setpoint) FIRST — a unit/exponent slip, not an envelope
        # violation. Reported alone so the model fixes the magnitude instead of
        # reading a "raise the limit" message. Independent of admin SafetyLimits.
        absurd = physically_absurd_violations(meta, params)
        if absurd:
            return absurd

        violations: list[str] = []
        param_specs = {p.name: p for p in meta.parameters}

        for name, value in params.items():
            spec = param_specs.get(name)
            if spec is None:
                continue
            if not isinstance(value, (int, float)):
                # A string-typed numeric would otherwise skip the global bound
                # entirely and reach hardware unchecked (2026-06-10 review).
                #
                # Since 2026-08-04 this is the NORMAL case, not an oddity: every
                # dimensioned parameter is declared a string in the tool schema,
                # because a number-typed argument came back corrupted 12 times
                # out of 12 on this provider. So the coercion here has to
                # understand the same forms the adapter accepts — including the
                # SI prefix ones ('3p') — or the gate would silently stop
                # checking the very parameters that most need it.
                try:
                    value = parse_quantity(value, strict=False, what=name)
                except (SIParseError, TypeError, ValueError):
                    continue
            name_lower = name.lower()
            unit_lower = spec.unit.lower() if spec.unit else ""
            for (pattern, unit_pat, global_min, global_max,
                 min_attr, max_attr) in self._resolved_checks:
                if pattern in name_lower and unit_pat in unit_lower:
                    global_min = global_min - _FLOAT_SLACK * abs(global_min)
                    global_max = global_max + _FLOAT_SLACK * abs(global_max)
                    if value < global_min:
                        violations.append(
                            f"Parameter '{name}' = {value} below global safety minimum "
                            f"{global_min}"
                            + self._diagnose(name, value, global_min, unit_lower,
                                             min_attr)
                        )
                    if value > global_max:
                        violations.append(
                            f"Parameter '{name}' = {value} above global safety maximum "
                            f"{global_max}"
                            + self._diagnose(name, value, global_max, unit_lower,
                                             max_attr)
                        )
                    break
        return violations

    # ── Out-of-envelope diagnosis ────────────────────────────────────
    #
    # A value outside the envelope has two causes that need OPPOSITE advice:
    #
    #   * a dropped exponent — the model wrote `1.5` meaning `1.5e-8` (15 nm).
    #     Off by many orders of magnitude; the fix is a new NUMBER, and the
    #     teaching text below puts that correction straight into the retry loop.
    #   * a correctly-written value that simply sits outside the CONFIGURED
    #     envelope. The fix is a human decision about the envelope — NOT a new
    #     number.
    #
    # Do not describe an ordinary envelope violation as a dropped exponent.
    # That would invite the model to replace a correctly supplied coordinate
    # with an invented one. Use the magnitude ratio to distinguish the cases.
    #
    # An exponent slip is ≥ 1000× by construction — one SI-prefix step is 1e3,
    # the common µm-as-m slip is 1e6, nm-as-m is 1e9. Anything under that is an
    # envelope question, so the two diagnoses cannot overlap. The test is
    # direction-independent: a value 1e-3× the FLOOR is as much a slip as one
    # 1e3× the cap (the old code hinted only on the above-max side, so a
    # too-small setpoint got no diagnosis at all).
    _EXPONENT_SLIP_FACTOR = 1e3

    @classmethod
    def _diagnose(
        cls, name: str, value: float, limit: float, unit_lower: str, limit_attr: str
    ) -> str:
        """Diagnostic suffix for an out-of-envelope value ('' = no diagnosis)."""
        if limit == 0 or value == 0:
            return ""
        ratio = abs(value) / abs(limit)
        if 1.0 / cls._EXPONENT_SLIP_FACTOR < ratio < cls._EXPONENT_SLIP_FACTOR:
            # In-scale overshoot → an ENVELOPE question, not a number question.
            # State the EVIDENCE and forbid the one dangerous move (silently
            # substituting a made-up value); do not assert that the value is
            # correct — the gate cannot know that either.
            return (
                f". NOTE: {value} is {ratio:.3g}× the {limit:g} limit, i.e. the same "
                f"order of magnitude — so this is NOT the classic dropped-exponent "
                f"mistake (those land 1e3× or further out). Treat it as an ENVELOPE "
                f"question: the value may be exactly what was intended and simply "
                f"outside the CONFIGURED safety envelope ('{limit_attr} = {limit:g}', "
                f"which the operator can adjust in 高级管理 → 全局安全限制 / Admin → "
                f"Safety Limits). Do NOT silently substitute a different number to get "
                f"past this gate. If this value is what you meant, say so and report "
                f"the limit to the operator; if you meant a different magnitude, state "
                f"the correction explicitly."
            )
        if unit_lower == "m" and ratio > 1.0:
            return (
                f". NOTE: '{name}' is a METRE quantity (SI). {value} m is "
                f"{ratio:.0e}× the {limit:g} m limit — you almost "
                f"certainly meant nanometres and lost the magnitude. "
                f"Re-emit as a STRING WITH AN SI PREFIX (15 nm → '15n', "
                f"100 nm → '100n', 1.5 µm → '1.5u'). Do NOT resend the same "
                f"value, and do NOT switch to '1.5e-8' — several of these "
                f"parameters reject exponent form outright."
            )
        if unit_lower == "a" and ratio > 1.0:
            # Same teaching pattern for a tunnelling-current setpoint: the model
            # wrote e.g. 1.5 (= 1.5 A) meaning a pico/nanoamp setpoint and hit
            # the retry loop 3×.
            return (
                f". NOTE: '{name}' is a tunnelling CURRENT, an AMPERE "
                f"quantity (SI). {value} A is {ratio:.0e}× the "
                f"{limit:g} A limit — STM setpoints are tiny "
                f"(1 pA … 100 nA). You likely meant pico/nanoamps: "
                f"re-emit as a STRING WITH AN SI PREFIX (100 pA → '100p', "
                f"1 nA → '1n', 50 pA → '50p'). Do NOT resend the same "
                f"value, and do NOT switch to '1e-10' — this parameter "
                f"rejects exponent form outright."
            )
        return (
            f". NOTE: '{name}' = {value} is {ratio:.0e}× the {limit:g} limit — an "
            f"order-of-magnitude gap, so this is far more likely a lost magnitude "
            f"than a real target. Re-emit as a STRING, using an SI prefix for a "
            f"small quantity ('100p', '15n', '1.5u'). Do NOT resend the same value."
        )

    # ── Layer 2: State preconditions ─────────────────────────────────

    def check_state_preconditions(
        self, meta: SkillMetadata, state: HardwareState
    ) -> list[str]:
        """Delegates to ``mast.core.safety.check_state_preconditions``.

        This used to be a hand-vendored copy of that parser, and it had ALREADY
        DRIFTED: the core version grew a ``withdrawn`` branch in the 2026-07-03
        review ("retract before a coarse motor move") and this copy never did, so
        for anything the AGENT ran, that precondition silently evaluated to
        "fine" — on the one path where nobody is watching. ``_GLOBAL_CHECKS``
        above is shared for exactly this reason after exactly this kind of drift;
        the precondition parser simply had not been given the same treatment yet
        (2026-07-31)."""
        return _core_check_state_preconditions(meta, state)

    # ── Layer 3: Approval requirement (info only — HITL middleware acts) ──

    @staticmethod
    def requires_approval(meta: SkillMetadata) -> str:
        """Returns 'none' / 'llm' / 'human'.

        Decoupled from execution: HumanInTheLoopMiddleware uses tool.metadata
        ['danger_level'] separately. This method exists for symmetry with v1
        and for any caller that wants to know.
        """
        if meta.safety_level == SafetyLevel.AUTO:
            return "none"
        elif meta.safety_level == SafetyLevel.CONFIRM:
            return "llm"
        return "human"


# ─────────────────────────────────────────────────────────────────────
# Middleware integration
# ─────────────────────────────────────────────────────────────────────

class SafetyGateMiddleware(AgentMiddleware):
    """Wrap each tool call: run SafetyGate.check_global_bounds + state precond.

    On violation, intercepts the tool call and returns a ToolMessage with
    status="error" so the agent's LLM sees the failure and can retry with
    different parameters or hand off.
    """

    def __init__(
        self,
        limits: SafetyLimits | None = None,
        get_state: Callable[[], HardwareState] | None = None,
        registry: ConfigOverrideRegistry | None = None,
        recorder: "Callable[[dict], None] | None" = None,
        get_mode: "Callable[[], Any] | None" = None,
    ):
        super().__init__()
        self.gate = SafetyGate(
            limits=limits or SafetyLimits(),
            registry=registry,
        )
        self._get_state = get_state
        # Global operating-mode gate (SAFE hard-block of tip processing / SEMI
        # shallow-only). Read live so a UI mode switch takes effect without
        # rebuilding the agent. None → no mode gating (AUTO-equivalent), which
        # preserves the prior always-allow behaviour for callers that don't wire
        # a mode source.
        self._get_mode = get_mode
        # Training-log recorder (RFC P2): observes each gate verdict (allow/block)
        # for safety positive/negative samples. Injected by the GUI; NEVER affects
        # the safety decision (read-only, fail-safe).
        self._recorder = recorder

    @property
    def name(self) -> str:
        return "SafetyGateMiddleware"

    def reload_overrides(self) -> None:
        """Hot-reload admin overrides without rebuilding the agent."""
        self.gate.reload()

    def _mode_block(
        self, tool_name: str, meta_obj: SkillMetadata, args: dict, call_id: str
    ) -> "ToolMessage | None":
        """Operating-mode tip-processing gate. Returns a blocking ToolMessage or None.

        The DECISION moved to :func:`mast.core.safety.mode_refusal` on
        2026-08-27; this method now only reads the mode and renders the verdict.
        The reason for the move is not tidiness: the same decision was written
        out a second time in ``skill_forge_tools`` and was **missing entirely**
        from ``ExecutionContext.run``, so a composite / conduct step / direct
        ``POST /api/skills/{name}/execute`` walked straight past it. Three
        renderers, one judgement.

        A failed mode read still fails OPEN to prior (AUTO) behaviour — the mode
        gate is the "do not repair the tip" behaviour contract, and the physical
        protections do not route through it.
        """
        try:
            mode = OperatingMode.coerce(self._get_mode())
        except Exception as e:  # a bad mode read must not wedge the executor
            logger.warning("get_mode failed in safety gate: %s", e)
            return None  # fail-open to prior (AUTO) behaviour
        refusal = mode_refusal(tool_name, meta_obj, args, mode)
        if refusal is None:
            return None
        logger.info("SafetyGate[%s] blocked: %s", mode.value, tool_name)
        return ToolMessage(
            content=f"[safety_gate] {refusal}",
            tool_call_id=call_id,
            name=tool_name,
            status="error",
        )

    def _safety_block(self, request: ToolCallRequest) -> "ToolMessage | None":
        """Run both safety layers; return a blocking ToolMessage, or None to allow.

        Shared by the sync ``wrap_tool_call`` (GUI ``graph.stream()``) and the
        async ``awrap_tool_call`` (CLI ``await graph.ainvoke()``) so the SAME
        safety logic guards both invocation paths. LangChain 1.2's base async hook
        raises NotImplementedError without an async twin — so an async run would
        otherwise crash at the first DANGEROUS-skill tool call instead of being
        gated.
        """
        tool = request.tool
        if tool is None:
            return None
        meta_obj: Any = (tool.metadata or {}).get("skill_metadata")
        if meta_obj is None or not isinstance(meta_obj, SkillMetadata):
            # Not a skill tool (e.g., handoff tool, buffer tool) — pass through.
            return None

        tool_call = request.tool_call
        # tool_call may be a dict or ToolCall object; LangChain commonly uses dict
        args = (
            tool_call.get("args", {})
            if isinstance(tool_call, dict)
            else getattr(tool_call, "args", {})
        )
        call_id = (
            tool_call.get("id", "")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "id", "")
        )

        # Layer 0: the ONE physically-dangerous action — an open-loop coarse Z
        # step TOWARD the sample (pan-type stepper, no feedback stop) — must NOT
        # run autonomously. Block it fail-closed and tell the model to hand off
        # to a human operator (who runs it via the GUI manual/approval path).
        # is_coarse_sample_approach keys on MotorMove direction='z-approach'
        # only; everything else is bounded by Nanonis (2026-06-11 re-scoping).
        if is_coarse_sample_approach(tool.name, args or {}):
            logger.warning(
                "SafetyGate blocked autonomous coarse sample approach: %s", tool.name
            )
            return ToolMessage(
                content=(
                    "[safety_gate] coarse_sample_approach_blocked: an open-loop coarse "
                    "Z step toward the sample can crash the tip and must be performed by "
                    "a human operator via the GUI. Do NOT retry — hand off to the user."
                ),
                tool_call_id=call_id,
                name=tool.name,
                status="error",
            )

        # Layer 0b: disabling a hardware protection (SafeTip / Z soft-limits)
        # removes a safety net and must not happen autonomously — hand off to a
        # human. Enabling them stays AUTO (2026-07-03 review).
        if is_protection_disable(tool.name, args or {}):
            logger.warning(
                "SafetyGate blocked autonomous protection-disable: %s", tool.name
            )
            return ToolMessage(
                content=(
                    "[safety_gate] protection_disable_blocked: turning OFF a hardware "
                    "protection (SafeTip / Z soft-limits) removes a tip-crash safety net "
                    "and must be done by a human operator via the GUI. Do NOT retry — "
                    "hand off to the user."
                ),
                tool_call_id=call_id,
                name=tool.name,
                status="error",
            )

        # Layer 0c: rewriting the global bias/current calibration scale defeats
        # every downstream voltage/current bound — never autonomous (2026-07-03).
        if is_calibration_change(tool.name, args or {}):
            logger.warning(
                "SafetyGate blocked autonomous calibration change: %s", tool.name
            )
            return ToolMessage(
                content=(
                    "[safety_gate] calibration_change_blocked: rewriting the global "
                    "bias/current calibration scale silently defeats every voltage/current "
                    "safety bound and must be done by a human operator via the GUI. Do NOT "
                    "retry — hand off to the user."
                ),
                tool_call_id=call_id,
                name=tool.name,
                status="error",
            )

        # Layer 0e: the coarse stepper's DRIVE voltage. Some Nanonis controllers
        # output 400 V; some piezo stacks fail at 300. Nothing reads back which
        # rig this is, and getting it wrong destroys the stack — there is no
        # retry. The per-rig ceiling is knowledge only the operator has (it lives
        # in mast.core.coarse_drive, declared behind the admin PIN), so the
        # operator is who consults it. SetMotorFreqAmp is CONFIRM, and CONFIRM on
        # this path means the model approves itself, so the level cannot express
        # this on its own (2026-07-31).
        if is_coarse_drive_change(tool.name, args or {}):
            logger.warning(
                "SafetyGate blocked autonomous coarse-drive change: %s", tool.name
            )
            return ToolMessage(
                content=(
                    "[safety_gate] coarse_drive_change_blocked: 粗动马达的驱动电压/频率"
                    "只能由用户设置。控制器支持的电压与这台机器的压电叠堆能承受的电压"
                    "是两个不同的数(有的控制器给到 400 V,而有的叠堆 300 V 就烧了),"
                    "没有任何读数能告诉你是哪一种,设错了不可恢复。"
                    "Do NOT retry — 需要改就交给用户在【高级】页设置。"
                ),
                tool_call_id=call_id,
                name=tool.name,
                status="error",
            )

        # Layer 0f: a RAW lateral coarse step. NOT a ban on relocating — the
        # guarded composite RelocateCoarseXY stays autonomous, and it is the one
        # that steps the coarse Z motor back for clearance first, checks the
        # chamber pressure, verifies the drive voltage, watches the current
        # between chunks and records the move on the coarse map. MotorMove does
        # none of that: its only precondition is the fine-Z piezo being at its
        # high limit (~1 µm), and that check passes when the state is unknown.
        # Leaving both autonomous would simply mean the model keeps calling the
        # cheaper one, so the incentive has to point the other way (2026-07-31).
        if is_unguarded_lateral_coarse_move(tool.name, args or {}):
            logger.warning(
                "SafetyGate blocked bare lateral coarse move: %s", tool.name
            )
            return ToolMessage(
                content=(
                    "[safety_gate] unguarded_lateral_coarse_move_blocked: "
                    "请改用 **RelocateCoarseXY** 换区,不要直接调 MotorMove 横向移动。"
                    "MotorMove 只检查压电是否收到顶(约 1 µm 余量,而且读不到状态时会放行),"
                    "不会先用粗动马达退针清障、不看真空度、不核对驱动电压、"
                    "移动过程中不看电流、也不会记进粗动大地图。"
                    "RelocateCoarseXY 会做这些,并且可以自主执行。"
                    "先用 get_coarse_map 看该往哪走。"
                    "确实需要裸移动时,请交给用户人工执行。"
                ),
                tool_call_id=call_id,
                name=tool.name,
                status="error",
            )

        # Layer 0g: an active ApproachTip refusal must also block a direct
        # AutoApproach tool call. This is a sequence guard, not a change to the
        # skill safety level. ApproachTip retains its own verified escalation
        # route through ExecutionContext; ordinary unblocked calls are unchanged.
        if is_approach_escalation(tool.name):
            refusal = active_approach_refusal()
            if refusal is not None:
                logger.warning(
                    "SafetyGate blocked %s: %s refused the escalation %.0fs ago",
                    tool.name, refusal.source, refusal.age_s(),
                )
                return ToolMessage(
                    content=(
                        "[safety_gate] approach_escalation_refused: "
                        f"{refusal.source} 在 {refusal.age_s():.0f} 秒前拒绝了粗进针"
                        f"升级，理由：{refusal.reason}。AutoApproach 就是它拒绝的那个"
                        "粗进针，直接调用等于绕过刚做出的安全判断。Do NOT retry "
                        "AutoApproach. 出路有三条：(1) 重新调用 ApproachTip —— 它会"
                        "重新核验 Z 反馈状态，若确实需要粗进针会自己升级，本拦截随即"
                        "解除；(2) 先修好拒绝理由里说的那个状态（如 Z 反馈开关读不出"
                        "/ 关不掉）再走 ApproachTip；(3) 交给用户在 GUI 手动进针。"
                    ),
                    tool_call_id=call_id,
                    name=tool.name,
                    status="error",
                )

        # Layer 0e: global operating-mode tip-processing gate (2026-07-07).
        # SAFE hard-blocks electrical pulses + mechanical tip shaping; SEMI blocks
        # a too-deep mechanical plunge (electrical pulses in SEMI are NOT blocked
        # anywhere any more — ⑰ replaced their confirmation dialog with a
        # notice); AUTO / no get_mode is a no-op.
        if self._get_mode is not None:
            mode_block = self._mode_block(tool.name, meta_obj, args or {}, call_id)
            if mode_block is not None:
                return mode_block

        # Layer 1: Global bounds
        violations = self.gate.check_global_bounds(meta_obj, args or {})
        if violations:
            logger.warning("SafetyGate blocked %s: %s", tool.name, violations)
            return ToolMessage(
                content=f"[safety_gate] global_bounds_violation: {'; '.join(violations)}",
                tool_call_id=call_id,
                name=tool.name,
                status="error",
            )

        # Layer 2: State preconditions (only if get_state callable provided)
        if self._get_state is not None and meta_obj.preconditions:
            try:
                hw_state = self._get_state()
            except Exception as e:
                logger.warning("get_state failed in safety check: %s", e)
                hw_state = None
            if hw_state is not None:
                state_violations = self.gate.check_state_preconditions(meta_obj, hw_state)
                if state_violations:
                    logger.warning("SafetyGate blocked %s on state: %s", tool.name, state_violations)
                    return ToolMessage(
                        content=f"[safety_gate] precondition_violation: {'; '.join(state_violations)}",
                        tool_call_id=call_id,
                        name=tool.name,
                        status="error",
                    )
        return None  # all checks passed → allow

    def _record_safety(self, request: ToolCallRequest, blocked) -> None:
        """Observe a gate verdict. Read-only + fail-safe — a recorder failure must
        NEVER affect the safety decision.

        TWO sinks, deliberately:

        * ``self._recorder`` — the TRAINING log. Optional, and it records BOTH
          verdicts (an allow is a positive sample). Absent outside a recorded
          trajectory.
        * ``core.diagnostics`` — the REFUSAL ledger. Always on, and it records
          only blocks. This exists because the first sink is optional: with no
          trajectory active, a block used to leave **no trace anywhere**. The
          operator got 「进针功能调用失败」  and neither they nor we could say
          which layer had refused it, on what state, with what args. A refusal
          that isn't written down did not happen, as far as the next person is
          concerned.
        """
        tool = request.tool
        if tool is None:
            return
        try:
            meta_obj = (tool.metadata or {}).get("skill_metadata")
            if not isinstance(meta_obj, SkillMetadata):
                return  # only record skill tools (skip handoff/buffer tools)
            tc = request.tool_call
            args = (tc.get("args", {}) if isinstance(tc, dict)
                    else getattr(tc, "args", {}))
            call_id = (tc.get("id", "") if isinstance(tc, dict)
                       else getattr(tc, "id", ""))
            reason = (str(getattr(blocked, "content", "") or "")
                      if blocked is not None else "")

            if blocked is not None:
                try:
                    from mast.core.diagnostics import record

                    mode = ""
                    if self._get_mode is not None:
                        try:
                            mode = str(OperatingMode.coerce(self._get_mode()))
                        except Exception:  # noqa: BLE001
                            mode = "?"
                    record("safety_block", tool.name, reason,
                           args=args or {},
                           safety_level=str(getattr(meta_obj, "safety_level", "")),
                           mode=mode)
                except Exception:  # noqa: BLE001
                    pass

            rec = self._recorder
            if rec is not None:
                rec({
                    "skill": tool.name,
                    "args": args or {},
                    "tool_call_id": call_id,
                    "verdict": "block" if blocked is not None else "allow",
                    "reason": reason,
                })
        except Exception:
            logger.debug("safety recorder failed (swallowed)", exc_info=True)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        blocked = self._safety_block(request)
        self._record_safety(request, blocked)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        blocked = self._safety_block(request)
        self._record_safety(request, blocked)
        if blocked is not None:
            return blocked
        return await handler(request)


__all__ = [
    "SafetyGate",
    "SafetyGateMiddleware",
    "_GLOBAL_CHECKS",
]
