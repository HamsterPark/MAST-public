"""Context-injection / tool-design fixes.

Pins three separate fixes that all share one theme — a good framework should not
make the LLM flail:

  * #133 — the orchestrator drives the research loop autonomously (literature /
    plan / report / parallel) instead of waiting for the operator to remind it.
  * #118 — the current setpoint is in AMPERES and is tiny; both the SetSetpoint
    tool description AND the safety-gate rejection teach the unit + range so the
    model stops re-sending `1.5` (= 1.5 A) three times.
  * #142 — the IC agent knows leveling (调平) exists and when to use it, and the
    real SetPiezoTilt / GetPiezoTilt skills are named.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_context_injection_fixes.py -x -v
"""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
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

from mast.agents._shared.safety_mw import SafetyGate
from mast.agents.instrument_control.prompts import SYSTEM_PROMPT
from mast.agents.orchestrator.graph import _ROUTER_PROMPT
from mast.config import SafetyLimits
from mast.skills.builtins.zcontrol import SetSetpoint


# ── #133: orchestrator autonomous-workflow context ───────────────────────

class TestOrchestratorAutonomousWorkflow:
    def test_prompt_has_autonomous_section(self):
        assert "autonomously" in _ROUTER_PROMPT.lower()

    def test_prompt_tells_it_to_lead_the_pipeline_stages(self):
        """每个阶段都要它主动发起，不等用户催。

        2026-08-24：路由提示词整体中文化之后，这里钉的从**英文词**换成**中文
        实质**。之前钉 `"first" in p` 这类词，是拿一个措辞当「那条规则还在」的
        代理 —— 措辞一换就红，而规则其实没动；反过来，规则被删了而那个词恰好
        还在别处出现，它又不会红。两个方向都不可靠。
        """
        p = _ROUTER_PROMPT
        # agent 名是标识符，中英文版本里都一样
        for agent in ("literature", "experiment_design",
                      "paper_writing", "paper_review"):
            assert agent in p, f"路由提示词里没有 {agent}"
        # 「新问题先派文献」这条规则
        assert "先派 literature" in p or "先 **literature**" in p, (
            "「新研究问题先收集先验」那条规则不见了")
        # 「不必等人说写方案」
        assert "不必等人说" in p or "不要等人催" in p, (
            "「不要等用户一个阶段一个阶段地催」那条规则不见了")
        # 并行指挥
        assert "并行" in p and "Parallel dispatch" in p, (
            "并行派单那一节不见了（英文小节名保留是给检索用的）")

    def test_prompt_says_lull_is_not_completion(self):
        """具体的失效形状：它在两个阶段之间派了 __end__ 然后干等。"""
        p = _ROUTER_PROMPT
        assert "空档不等于任务完成" in p, (
            "「阶段之间的空档不等于任务完成」那条不见了 —— 少了它，"
            "编排器会在每个阶段之间停下来等人催")


# ── #118: setpoint units — tool description + safety-gate teaching ────────

class TestSetpointUnitContext:
    def test_setsetpoint_description_names_unit_and_range(self):
        meta = SetSetpoint().metadata()
        sp = next(p for p in meta.parameters if p.name == "setpoint_a")
        d = sp.description
        # 2026-08-24 技能描述中文化：`AMPERE` 全大写是英文里的强调手段，中文用
        # 「安培」把同一件事说清楚。这条断言要的是**单位被教到了**，不是那个词。
        assert "AMPERE" in d or "安培" in d, "没有把 setpoint 的单位讲清楚"
        # 2026-08-04：实质不变（单位 + 一个能照抄的具体值 + 量程 + 那个错误模式），
        # 变的是**正确形式** —— setpoint_a 整个量程远小于 1，所以它强制要求 SI 前缀，
        # 而 '1e-10' 现在会被 parse_si 直接拒绝。钉住新形式而不是放宽断言。
        assert "'100p'" in d                          # a concrete copyable value
        assert "1 pA" in d and "100 nA" in d          # the typical range bounds
        # Teaches the specific error mode (bare 1.5 = 1.5 A).
        assert "1.5" in d

    def test_ic_prompt_has_setpoint_amperes_note(self):
        assert "AMPERES" in SYSTEM_PROMPT
        # The setpoint note (distinct from the meters/scan-size note).
        assert "SetSetpoint" in SYSTEM_PROMPT

    def test_safety_gate_amperes_hint_on_exponent_slip_setpoint(self):
        # A setpoint that is an actual EXPONENT SLIP — 5e-4 A written for 500 pA,
        # 5000× the 100 nA cap, still below the 1 mA physical-impossibility floor
        # so it reaches the tunable-envelope check — must get the AMPERES teaching
        # hint so the model fixes the magnitude instead of re-sending it.
        gate = SafetyGate(SafetyLimits())
        meta = SetSetpoint().metadata()
        v = gate.check_global_bounds(meta, {"setpoint_a": 5e-4})  # 5000× the cap
        assert len(v) == 1
        msg = v[0]
        assert "above" in msg.lower()
        assert "AMPERE" in msg                          # names the unit
        # 2026-08-04：仍然「给出一个能照抄的正确写法」，只是那个写法从 1e-10 变成
        # '100p' —— 教旧形式会把模型送进第二次拒绝，而拒绝语本身正是要终结循环的。
        assert "'100p'" in msg or "'1n'" in msg
        assert "do not resend" in msg.lower()           # stop the dead loop

    def test_safety_gate_does_not_cry_units_on_a_plausible_over_cap_setpoint(self):
        # amperes axis. 500 nA is above the configured 100 nA
        # cap but only 5× above it and a physically real setpoint. The old gate
        # gave it the same "you likely meant pico/nanoamps, do NOT resend the same
        # value" lecture as a 1e7× slip — telling the model a correctly written
        # value is malformed, which invites it to invent a different one. An
        # in-scale overshoot is an ENVELOPE question and must be reported as one.
        gate = SafetyGate(SafetyLimits())
        meta = SetSetpoint().metadata()
        v = gate.check_global_bounds(meta, {"setpoint_a": 5e-7})  # 500 nA > 100 nA cap
        assert len(v) == 1
        msg = v[0]
        assert "above" in msg.lower()
        assert "likely meant pico/nanoamps" not in msg    # no misdiagnosis
        assert "NOT the classic dropped-exponent" in msg  # says so explicitly
        assert "setpoint_max_a" in msg                    # names the knob
        assert "全局安全限制" in msg                        # and where to turn it

    def test_safety_gate_absurd_setpoint_is_caught_with_magnitude_teaching(self):
        # The exact field-mistake: setpoint_a=1.5 (= 1.5 A). This is
        # order-of-magnitude impossible, so the physical-absurdity guard fires
        # FIRST (rig-independent, holds even if the admin loosened the cap) and
        # still teaches the unit + scientific notation so the loop breaks.
        gate = SafetyGate(SafetyLimits())
        meta = SetSetpoint().metadata()
        v = gate.check_global_bounds(meta, {"setpoint_a": 1.5})
        assert len(v) == 1
        msg = v[0]
        # 2026-08-04：教的形式从「科学计数法」换成「带 SI 前缀的字符串」——
        # 对这个参数，前者已经是会被拒的写法了。
        assert "SI 前缀" in msg or "SI prefix" in msg
        assert "'100p'" in msg or "'1n'" in msg         # a copyable correct form
        # Teaches "don't retry the same value" (zh or en).
        assert "切勿重试" in msg or "do not resend" in msg.lower()

    def test_safety_gate_valid_setpoint_passes(self):
        gate = SafetyGate(SafetyLimits())
        meta = SetSetpoint().metadata()
        assert gate.check_global_bounds(meta, {"setpoint_a": 1e-10}) == []


# ── #142: leveling (调平) capability + context ────────────────────────────

class TestLevelingContext:
    def test_ic_prompt_mentions_leveling(self):
        # The agent must KNOW leveling exists and roughly when to use it.
        assert "调平" in SYSTEM_PROMPT
        assert "tilt" in SYSTEM_PROMPT.lower()

    def test_ic_prompt_names_the_real_tilt_skills(self):
        assert "SetPiezoTilt" in SYSTEM_PROMPT
        assert "GetPiezoTilt" in SYSTEM_PROMPT

    def test_tilt_skills_exist_and_are_discoverable(self):
        # The skills the prompt points at must actually be in the IC tool set.
        from mast.agents.instrument_control.tools import discover_instrument_skills
        names = {m.name for m in discover_instrument_skills().list_skills()}
        assert "SetPiezoTilt" in names
        assert "GetPiezoTilt" in names


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
