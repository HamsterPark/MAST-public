"""HITL map derivation + safety-level parity pins (2026-06-11 re-scoping).

Safety model after the 2026-06-11 re-scoping: the physically-dangerous action
class is a coarse Z motion that can crash the tip — the OPEN-LOOP MotorMove
z-approach (pan-type piezo stepper, no current-feedback stop) AND, conservatively,
any Z-bearing MotorMoveClosedLoop move (closed-loop is positional feedback, which
also does not stop on tip contact). It is gated as a PARAMETER condition by
SafetyGateMiddleware (fail-closed block on the autonomous agent path) and by
SkillExecutor (human approval on the manual path) — see
mast.core.safety.is_coarse_sample_approach. Everything else is bounded by
Nanonis (bias/current/fine-Z ranges, the current-feedback AutoApproach module),
so NO builtin skill carries safety_level=DANGEROUS and the derived HITL map is
empty.

Pins:
1. _derive_hitl_map keys == exactly the DANGEROUS skills (parity with metadata).
2. NO builtin skill is DANGEROUS (the re-scoping). A future DANGEROUS flip is a
   deliberate decision that must consciously update this pin.
3. The demoted skills stay demoted (BiasPulse / EmergencyRetract / TipShape /
   TipShapeWithReadback = AUTO; MotorMove = CONFIRM).
4. is_coarse_sample_approach flags coarse Z motions that can crash the tip: the
   open-loop MotorMove z-approach AND any Z-bearing MotorMoveClosedLoop move
   (closed-loop is positional feedback — no tip-contact stop). MotorMove exposes
   the z-approach / z-retract directions; a pure-XY closed-loop move is NOT flagged.
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found above " + str(Path(__file__).resolve()))


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.agents.instrument_control.graph import _derive_hitl_map
from mast.agents.instrument_control.tools import discover_instrument_skills
from mast.core.safety import is_coarse_sample_approach
from mast.core.types import SafetyLevel

# Skills demoted in the 2026-06-11 re-scoping (Nanonis bounds them — NOT DANGEROUS).
_DEMOTED_TO_AUTO = {"BiasPulse", "EmergencyRetract", "TipShape", "TipShapeWithReadback"}
_DEMOTED_TO_CONFIRM = {"MotorMove"}


@pytest.fixture(scope="module")
def registry():
    return discover_instrument_skills()


@pytest.fixture(scope="module")
def meta_by_name(registry):
    return {m.name: m for m in registry.list_skills()}


class TestDeriveHitlMap:
    def test_map_keys_equal_all_dangerous_skills(self, registry, meta_by_name):
        # Parity invariant: the derived map keys are EXACTLY the DANGEROUS skills.
        hitl = set(_derive_hitl_map(registry).keys())
        all_dangerous = {
            n for n, m in meta_by_name.items()
            if m.safety_level == SafetyLevel.DANGEROUS
        }
        assert hitl == all_dangerous, (
            f"HITL map ({sorted(hitl)}) must equal all DANGEROUS skills "
            f"({sorted(all_dangerous)})"
        )

    # The exact DANGEROUS set. Updated deliberately on 2026-07-13 — this pin's own
    # instruction was "a future DANGEROUS flip is a deliberate decision that must
    # consciously update this pin", and the optional-hardware modules forced it.
    #
    # The 2026-06-11 re-scoping emptied this set on one premise: **everything is
    # bounded by Nanonis' own limits** (bias ±10 V, the piezo ranges, the
    # current-feedback AutoApproach). That premise held while every skill drove the
    # core STM — it is why TipShape, which deliberately crashes the tip into the
    # surface, is merely AUTO. It does NOT survive contact with a laser and an RF
    # amplifier. The five below are exactly the actions whose hazard Nanonis' limits
    # do not bound:
    #
    #   SetLaserOnOff        a human eye hazard. Nanonis bounds nothing here, and MAST
    #                        cannot see a shutter or whether anyone is at the scope.
    #   StartRfGenerator     dBm into a tunnel junction is energy. The module's max
    #   RunRfFrequencySweep  power is far above what a junction tolerates, so "within
    #                        range" and "safe" are unrelated.
    #   MoveProbeXY          probe-probe collision. Nanonis bounds each scanner's own
    #                        range and knows NOTHING about where the other probes are.
    #   SetPiControllerOnOff closes a feedback loop onto an ARBITRARY output — it can
    #                        drive a piezo with no tip-contact stop, which is precisely
    #                        the "coarse motion, no feedback stop" class the 2026-06-11
    #                        re-scoping named as the one real danger.
    #
    # Everything else in the optional set is CONFIRM: a bias sweep is bounded like
    # SetBias, a per-probe Z loop is bounded like the core one. Note in particular
    # that PulseProbeBias is CONFIRM, not DANGEROUS — its core twin BiasPulse is AUTO,
    # and gating the multi-probe copy harder than the original would be incoherent.
    # Ground 1 — the HARDWARE hazards Nanonis' own limits do not bound. Every one is
    # an optional-hardware skill: the modules ship OFF, so they are gated twice over
    # (not in the tool list at all; and if you switch the module on, HITL still stops
    # the agent to ask).
    DANGEROUS_UNBOUNDED_HARDWARE = [
        "MoveProbeXY",
        "RunRfFrequencySweep",
        "SetLaserOnOff",
        "SetPiControllerOnOff",
        "StartRfGenerator",
    ]

    # Ground 2 — a DIFFERENT kind of hazard, and the reason the invariant below had to
    # be restated rather than merely extended (2026-07-13).
    #
    # LockNanonisUI puts a MODAL WINDOW over the Nanonis software and prevents the
    # operator from interacting with the instrument. It endangers no hardware. What it
    # endangers is the ability to INTERVENE — and every other safety mechanism in this
    # system ultimately falls back on exactly that: a human can walk to the microscope
    # and take over. The safety gate, the mode gate, the abort button, the approval
    # prompts, all of them assume that floor is there. An agent that can remove it has
    # removed the floor under all of them.
    #
    # It was CONFIRM, which derives no HITL gate on the agent path — so an autonomous
    # run could lock the operator out without anyone being asked. UnlockNanonisUI is
    # the opposite direction and is deliberately AUTO: restoring a human's control is
    # never the dangerous way to go. Same asymmetry as Laser_OnOffSet (abort-safe only
    # in its OFF form) and AutoApproach_OnOffSet (only in its stop form).
    DANGEROUS_REMOVES_HUMAN_OVERSIGHT = [
        "LockNanonisUI",
    ]

    # Ground 3 — steps AROUND a protection (2026-07-13). These are the 高级能力: powers
    # the operator can lend the agent from 高级 → 能力开关, behind the admin PIN, all OFF
    # by default. They are DANGEROUS for a reason distinct from both grounds above:
    # they do not damage hardware and they do not lock the human out. They make one of
    # this system's own barriers stop meaning what it says.
    #
    #   LoadNanonisScript   The script allow-list approves *a script, in a slot* — and a
    #                       script runs on the real-time controller, where SafetyGate,
    #                       the mode gate, the abort gate and HITL are ALL blind. Putting
    #                       a different file into a vetted slot would leave the approval
    #                       in the file and gone in fact. (Which is why the skill also
    #                       REFUSES allow-listed slots outright — the HITL gate is the
    #                       second fence, not the first.)
    #   QuitNanonis         Ends the session. Everything MAST believes about the
    #                       instrument stops being true at that instant.
    #   LoadMultiPassConfig Loads a scan configuration from a file MAST cannot read. The
    #                       file decides each pass's bias and Z offset.
    #
    # SaveNanonisScript / SaveNanonisScriptLut / SaveMultiPassConfig are CONFIRM, not
    # DANGEROUS: they write a file and change nothing about what the instrument does.
    # WaitForScanEndBlocking is CONFIRM too — it costs the main connection for the
    # duration, which is a cost, not a hazard.
    DANGEROUS_STEPS_AROUND_A_PROTECTION = [
        "LoadMultiPassConfig",
        "LoadNanonisScript",
        "QuitNanonis",
    ]

    # ── 理由四：写下一个系统将来会默认信任的数值 ────────────────────────────
    #
    # Added 2026-08-03 with the Z-parameter presets. This ground is about WHEN the
    # harm lands, not about what the call touches.
    #
    #   CreateZCtrlPreset   Touches no hardware whatsoever — it writes a row to a
    #                       settings file. That is precisely why it needs the card:
    #                       what it stores is the value every later
    #                       ApplyZCtrlPreset('<name>') will send straight into the
    #                       Z loop without asking anyone again. The blast radius is
    #                       every future use, not this call.
    #
    #                       The card is also the review step the mechanism was
    #                       designed around — "agent 可以新建默认参数组，但是建完他
    #                       被要求检查". hitl_bridge mirrors the arguments AS PARSED,
    #                       so a gain that lost its exponent shows up on the card as
    #                       the wrong number and can be edited or rejected before it
    #                       is ever stored.
    #
    #                       Why the check has to be a HUMAN one: on 2026-08-03 an
    #                       agent sent 3 for 3e-12, then read the stored 3.0 back and
    #                       reported "p_gain = 3.0（= 3e-12）… 数值通道工作正常".
    #                       A check performed by the party that made the error is not
    #                       a check.
    DANGEROUS_WRITES_A_TRUSTED_DEFAULT = [
        "CreateZCtrlPreset",
    ]

    EXPECTED_DANGEROUS = sorted(
        DANGEROUS_UNBOUNDED_HARDWARE
        + DANGEROUS_REMOVES_HUMAN_OVERSIGHT
        + DANGEROUS_STEPS_AROUND_A_PROTECTION
        + DANGEROUS_WRITES_A_TRUSTED_DEFAULT
    )

    def test_dangerous_set_is_exactly_the_three_grounds(self, meta_by_name):
        dangerous = sorted(
            n for n, m in meta_by_name.items()
            if m.safety_level == SafetyLevel.DANGEROUS
        )
        assert dangerous == self.EXPECTED_DANGEROUS, (
            f"DANGEROUS 集合变了：期望 {self.EXPECTED_DANGEROUS}，实际 {dangerous}。\n"
            "DANGEROUS = 自动进 HITL 人工审批闸门。加进来意味着「自治运行时必须停下来问人」，"
            "拿掉意味着「无人值守时可以自己做」。两者都要有意为之。\n"
            "目前只有三条理由够格：①Nanonis 限值管不住的硬件危险（射频功率、激光、探针互撞、"
            "环闭到任意输出）；②拿掉人的介入能力（锁 Nanonis 界面）；③绕过本系统自己的某道"
            "保护（覆盖已审脚本槽位、退出 Nanonis、载入 MAST 看不见的扫描配置）。"
        )

    def test_no_CORE_skill_is_dangerous_for_HARDWARE_reasons(self, meta_by_name):
        """The 2026-06-11 re-scoping, restated precisely and still standing.

        Its claim was about HARDWARE: the core STM skills are all bounded by Nanonis'
        own limits, which is why TipShape — a skill that deliberately drives the tip
        into the surface — is merely AUTO. That claim is untouched. What it never said
        anything about is a skill that endangers the human's ability to intervene, or
        one that makes a barrier stop meaning what it says. Those are grounds 2 and 3;
        they are not counter-examples to the re-scoping, they are outside its scope.
        """
        from mast.skills.hardware_modules import SKILL_OWNER
        exempt = set(self.DANGEROUS_REMOVES_HUMAN_OVERSIGHT) | \
            set(self.DANGEROUS_STEPS_AROUND_A_PROTECTION) | \
            set(self.DANGEROUS_WRITES_A_TRUSTED_DEFAULT)
        core_dangerous = sorted(
            n for n, m in meta_by_name.items()
            if m.safety_level == SafetyLevel.DANGEROUS
            and n not in SKILL_OWNER
            and n not in exempt
        )
        assert core_dangerous == [], (
            f"核心 skill 里出现了因**硬件**危险而 DANGEROUS 的：{core_dangerous}。"
            "核心 STM 动作由 Nanonis 限值 + is_coarse_sample_approach 参数级门控管，"
            "不靠 safety_level 静态人工门控。"
        )

    def test_every_ground_3_skill_is_behind_the_capability_gate_AND_the_pin(self):
        """Gated three times over, and each fence catches something the others do not:

        1. the capability ships OFF ⇒ the skill is not in the agent's tool list at all;
        2. switching it on needs the admin PIN (a human hand, not a mis-click);
        3. and even then HITL stops the agent to ask before each call.

        Plus, for the script Load specifically, a fourth: it refuses vetted slots
        outright, because a human approval that can be silently invalidated is not an
        approval.
        """
        from mast.api.admin_pin import GUARDED_KEYS
        from mast.skills.advanced_capabilities import (
            CAPABILITY_BY_ID,
            SETTINGS_KEY,
            SKILL_OWNER as ADV_OWNER,
        )
        assert SETTINGS_KEY in GUARDED_KEYS
        for name in self.DANGEROUS_STEPS_AROUND_A_PROTECTION:
            cap = ADV_OWNER.get(name)
            assert cap, f"{name} 是 DANGEROUS 却不属于任何可关闭的高级能力"
            assert CAPABILITY_BY_ID[cap].default_on is False, f"{cap} 默认开着"

    def test_hitl_map_gates_exactly_the_dangerous_skills(self, registry):
        """The map is DERIVED from safety_level — this pins that it really is."""
        assert sorted(_derive_hitl_map(registry)) == self.EXPECTED_DANGEROUS

    def test_hardware_dangerous_skills_all_belong_to_an_optional_module(self, meta_by_name):
        """Gated twice over: the module ships OFF (so the skill is not even in the
        agent's tool list), and switching the module on still leaves HITL in the way."""
        from mast.skills.hardware_modules import SKILL_OWNER
        for name in self.DANGEROUS_UNBOUNDED_HARDWARE:
            assert name in SKILL_OWNER, f"{name} 是 DANGEROUS 却不属于任何可关闭的硬件模块"

    def test_the_de_escalating_direction_is_never_gated(self, meta_by_name):
        """Unlocking the UI gives the operator their instrument back. Gating THAT would
        be perverse — and a run that left the UI locked is a real thing that happens.
        The rule across this whole system: the direction that reduces capability is
        free; the direction that increases it needs a human."""
        m = meta_by_name.get("UnlockNanonisUI")
        assert m is not None, "UnlockNanonisUI 不见了——锁上了却解不开"
        assert m.safety_level == SafetyLevel.AUTO, (
            f"UnlockNanonisUI 是 {m.safety_level.name}。恢复人的控制权永远不是危险的方向。"
        )

    def test_each_entry_has_allowed_decisions(self, registry):
        # Structural invariant — now non-vacuous.
        m = _derive_hitl_map(registry)
        assert m, "HITL map 空了——DANGEROUS skill 的人工审批闸门没接上"
        for name, cfg in m.items():
            assert isinstance(cfg, dict) and cfg.get("allowed_decisions"), (
                f"{name} HITL entry must carry a non-empty allowed_decisions list"
            )
            assert "approve" in cfg["allowed_decisions"]


class TestSafetyLevelRescopingPinned:
    """2026-06-11 re-scoping pins — the demoted skills stay demoted.

    Guards the inverse of the old v0.3.22 regression: now a SILENT RE-PROMOTION
    to DANGEROUS (which would re-enable a human gate the re-scoping removed on
    purpose) fails the suite.
    """

    @pytest.mark.parametrize("skill_name", sorted(_DEMOTED_TO_AUTO))
    def test_demoted_to_auto(self, skill_name, meta_by_name):
        assert skill_name in meta_by_name, (
            f"{skill_name} not discovered in the instrument registry"
        )
        assert meta_by_name[skill_name].safety_level == SafetyLevel.AUTO, (
            f"{skill_name} is bounded by Nanonis and was demoted to AUTO "
            "(2026-06-11 re-scoping); re-promoting it is a deliberate decision."
        )

    @pytest.mark.parametrize("skill_name", sorted(_DEMOTED_TO_CONFIRM))
    def test_demoted_to_confirm(self, skill_name, meta_by_name):
        assert meta_by_name[skill_name].safety_level == SafetyLevel.CONFIRM, (
            f"{skill_name} baseline is CONFIRM; the z-approach danger is gated "
            "separately by is_coarse_sample_approach."
        )


class TestCoarseApproachGate:
    """The ONE physically-dangerous action: open-loop coarse Z toward the sample."""

    def test_motormove_z_approach_is_flagged(self):
        assert is_coarse_sample_approach("MotorMove", {"direction": "z-approach"}) is True

    def test_lateral_and_retract_not_flagged(self):
        for d in ("x+", "x-", "y+", "y-", "z-retract"):
            assert is_coarse_sample_approach("MotorMove", {"direction": d}) is False, (
                f"direction {d!r} is not toward the sample — must not be gated"
            )

    def test_other_skills_not_flagged(self):
        assert is_coarse_sample_approach("BiasPulse", {}) is False
        # AutoApproach has Nanonis current-feedback protection → not the danger.
        assert is_coarse_sample_approach("AutoApproach", {}) is False
        # A pure-XY closed-loop move (no Z component) is NOT a coarse sample
        # approach. (A Z-bearing closed-loop move IS now flagged — see
        # test_closed_loop_z_move_is_flagged below.)
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"target_x_m": 1e-6, "target_y_m": 0.0}) is False
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"target_z_m": 0.0}) is False
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"target_z_m": None}) is False

    def test_closed_loop_z_move_is_flagged(self):
        # Closed-loop coarse moves use POSITIONAL feedback (no tip-contact stop)
        # and the Z direction can't be proven safe statically, so ANY Z-bearing
        # closed-loop move is conservatively gated like an open-loop z-approach.
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"target_z_m": -1e-7}) is True
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"target_z_m": 1e-7}) is True
        # Absolute move with a Z target present is gated even at 0.0 (an absolute
        # Z target's safe side can't be proven statically).
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"target_z_m": 0.0, "absolute": True}) is True
        # No Z / zero relative Z / pure-XY → NOT flagged.
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"target_z_m": 0.0}) is False
        assert is_coarse_sample_approach(
            "MotorMoveClosedLoop", {"target_x_m": 1e-6, "target_y_m": 1e-6}) is False

    def test_motormove_exposes_z_directions(self, meta_by_name):
        spec = next(
            p for p in meta_by_name["MotorMove"].parameters if p.name == "direction"
        )
        assert "z-approach" in spec.allowed_values
        assert "z-retract" in spec.allowed_values


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:randomly"])
