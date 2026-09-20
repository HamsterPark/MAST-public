"""Phase 3 finish — SafetyGate + SafetyGateMiddleware tests.

Validates:
  - SafetyGate.check_global_bounds catches bias_v out of [-10, 10]
  - SafetyGate.check_state_preconditions catches z_controller_on=False when
    skill demands Z on
  - Admin override (ConfigOverrideRegistry) tightens / loosens / removes
    global checks
  - SafetyGateMiddleware.wrap_tool_call returns ToolMessage(status="error")
    on violation, calls handler on safe inputs
"""
from __future__ import annotations

import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from typing import Any
from unittest.mock import MagicMock

import pytest

from mast.admin.override_store import ConfigOverrideRegistry
from mast.agents._shared.safety_mw import (
    SafetyGate,
    SafetyGateMiddleware,
    _GLOBAL_CHECKS,
)
from mast.agents._shared.skill_adapter import wrap_skill
from mast.config import SafetyLimits
from mast.core.types import (
    HardwareState,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
)
from mast.skills.base import BaseSkill
from mast.skills.builtins.bias import GetBias, SetBias


# ─────────────────────────────────────────────────────────────────────
# SafetyGate (pure logic, no middleware)
# ─────────────────────────────────────────────────────────────────────

class TestSafetyGateBounds:
    def setup_method(self):
        self.gate = SafetyGate(SafetyLimits())  # defaults: bias ±10 V

    def test_bias_within_bounds_passes(self):
        meta = SetBias().metadata()
        v = self.gate.check_global_bounds(meta, {"bias_v": 5.0})
        assert v == []

    def test_bias_above_max_fails(self):
        meta = SetBias().metadata()
        v = self.gate.check_global_bounds(meta, {"bias_v": 12.0})
        assert len(v) == 1
        assert "above" in v[0].lower()

    def test_bias_below_min_fails(self):
        meta = SetBias().metadata()
        v = self.gate.check_global_bounds(meta, {"bias_v": -50.0})
        assert len(v) == 1
        assert "below" in v[0].lower()

    def test_scan_size_above_max_gives_teaching_unit_hint(self):
        # The IC operator hit a model that emitted `width_m=1.5` (metres) for a
        # 15 nm scan and looped. The rejection must TEACH: name the unit, show
        # scientific notation, and say "don't resend the same value" — so the
        # correction lands in the retry loop instead of an 8-turn dead loop.
        meta = SkillMetadata(
            name="ConfigureScan",
            parameters=[ParameterSpec(name="width_m", type="float", unit="m")],
        )
        v = self.gate.check_global_bounds(meta, {"width_m": 1.5})
        assert len(v) == 1
        msg = v[0]
        assert "above" in msg.lower()
        # 2026-08-04：形式从科学计数法换成 SI 前缀字符串 —— 对 width_m 这类整个
        # 量程远小于 1 的参数，'1.5e-8' 本身已经是会被拒的写法，教它等于把模型
        # 从一个死循环换到另一个。四件实质都还在。
        assert "METRE" in msg or "METER" in msg        # names the unit
        assert "'15n'" in msg                          # shows the right value
        assert "SI PREFIX" in msg or "SI 前缀" in msg   # tells it how to fix
        assert "do not resend" in msg.lower()          # stop the dead loop

    def test_non_numeric_param_skipped(self):
        meta = SkillMetadata(
            name="X",
            parameters=[ParameterSpec(name="mode", type="str", unit="")],
        )
        v = self.gate.check_global_bounds(meta, {"mode": "constant_current"})
        assert v == []

    def test_param_without_matching_unit_skipped(self):
        # If param has no matching (name+unit) in _GLOBAL_CHECKS → no global check
        meta = SkillMetadata(
            name="X",
            parameters=[ParameterSpec(name="bias_v", type="float", unit="other")],
        )
        v = self.gate.check_global_bounds(meta, {"bias_v": 9999.0})
        assert v == []  # unit didn't match "v"


class TestSafetyGatePreconditions:
    def setup_method(self):
        self.gate = SafetyGate(SafetyLimits())

    def test_z_controller_on_required_passes(self):
        meta = SkillMetadata(name="X", preconditions=["z_controller_on"])
        state = HardwareState(z_controller_on=True)
        v = self.gate.check_state_preconditions(meta, state)
        assert v == []

    def test_z_controller_on_required_fails(self):
        meta = SkillMetadata(name="X", preconditions=["z_controller_on"])
        state = HardwareState(z_controller_on=False)
        v = self.gate.check_state_preconditions(meta, state)
        assert len(v) == 1
        assert "Z controller is OFF" in v[0]

    def test_scan_not_running_fails(self):
        meta = SkillMetadata(name="X", preconditions=["scan_not_running"])
        state = HardwareState(scan_running=True)
        v = self.gate.check_state_preconditions(meta, state)
        assert len(v) == 1


class TestSafetyGateAdminOverride:
    """ConfigOverrideRegistry hot-merges JSON edits into limits + checks."""

    def setup_method(self):
        ConfigOverrideRegistry.reset()

    def test_no_registry_means_default_checks(self):
        gate = SafetyGate(SafetyLimits(), registry=None)
        assert len(gate._checks) == len(_GLOBAL_CHECKS)

    def test_admin_tightens_bias_max(self, tmp_path):
        # Write override JSON: bias_max_v dropped from 10 to 3
        ovr_dir = tmp_path / "config" / "overrides"
        ovr_dir.mkdir(parents=True)
        (ovr_dir / "safety_limits.json").write_text(
            '{"bias_max_v": 3.0}',
            encoding="utf-8",
        )
        ConfigOverrideRegistry.reset()
        registry = ConfigOverrideRegistry(overrides_dir=ovr_dir)
        gate = SafetyGate(SafetyLimits(), registry=registry)
        meta = SetBias().metadata()
        # 5 V was OK with default 10 V max, now violates 3 V cap
        violations = gate.check_global_bounds(meta, {"bias_v": 5.0})
        assert any("above" in v for v in violations)

    def test_admin_removes_bias_check(self, tmp_path):
        ovr_dir = tmp_path / "config" / "overrides"
        ovr_dir.mkdir(parents=True)
        (ovr_dir / "safety_checks.json").write_text(
            '{"removals": ["bias_v"]}', encoding="utf-8",
        )
        ConfigOverrideRegistry.reset()
        registry = ConfigOverrideRegistry(overrides_dir=ovr_dir)
        gate = SafetyGate(SafetyLimits(), registry=registry)
        meta = SetBias().metadata()
        # No global bias check — value of 1000 V passes the GLOBAL layer
        violations = gate.check_global_bounds(meta, {"bias_v": 1000.0})
        assert violations == []

    def test_malformed_override_does_not_crash_and_keeps_builtins(self, tmp_path):
        """A typo'd admin safety_checks.json (unknown min_attr) must NOT abort
        SafetyGate construction — that would take the live IC agent offline
        (security 审查, #1). The bad entry is dropped; built-in
        bias check still fires."""
        ovr_dir = tmp_path / "config" / "overrides"
        ovr_dir.mkdir(parents=True)
        (ovr_dir / "safety_checks.json").write_text(
            '{"additions": [{"pattern": "foo_v", "unit": "v", '
            '"min_attr": "NOT_A_REAL_FIELD", "max_attr": "also_bogus"}]}',
            encoding="utf-8",
        )
        ConfigOverrideRegistry.reset()
        registry = ConfigOverrideRegistry(overrides_dir=ovr_dir)
        gate = SafetyGate(SafetyLimits(), registry=registry)  # must not raise
        meta = SetBias().metadata()
        # Built-in bias cap survives the bad addition.
        violations = gate.check_global_bounds(meta, {"bias_v": 50.0})
        assert any("above" in v for v in violations)
        # The bogus check did not resolve into the active set.
        assert all("foo_v" != pat for pat, *_ in gate._resolved_checks)

    def test_malformed_override_replacement_leaves_builtin_intact(self, tmp_path):
        """A bad 'overrides' entry is skipped, keeping the built-in untouched
        (rather than crashing or silently disabling the bias cap)."""
        ovr_dir = tmp_path / "config" / "overrides"
        ovr_dir.mkdir(parents=True)
        (ovr_dir / "safety_checks.json").write_text(
            '{"overrides": [{"pattern": "bias_v", "unit": "v", '
            '"min_attr": "bogus_min"}]}',  # missing max_attr + bad min_attr
            encoding="utf-8",
        )
        ConfigOverrideRegistry.reset()
        registry = ConfigOverrideRegistry(overrides_dir=ovr_dir)
        gate = SafetyGate(SafetyLimits(), registry=registry)  # must not raise
        meta = SetBias().metadata()
        violations = gate.check_global_bounds(meta, {"bias_v": 50.0})
        assert any("above" in v for v in violations)  # default ±10 V cap intact


# ─────────────────────────────────────────────────────────────────────
# SafetyGateMiddleware (LangChain integration)
# ─────────────────────────────────────────────────────────────────────

def _make_request(tool, args: dict, call_id: str = "test-1") -> Any:
    """Build a ToolCallRequest-like object (LangChain accepts dict tool_call)."""
    req = MagicMock()
    req.tool = tool
    req.tool_call = {"name": tool.name, "args": args, "id": call_id, "type": "tool_call"}
    req.state = {}
    req.runtime = MagicMock()
    return req


class TestSafetyGateMiddleware:
    def setup_method(self):
        self.mw = SafetyGateMiddleware(SafetyLimits())

    def test_skill_within_bounds_calls_handler(self):
        tool = wrap_skill(SetBias, lambda: MagicMock(safe_call=lambda *a, **k: None))
        req = _make_request(tool, {"bias_v": 1.0})
        handler = MagicMock(return_value="ok")
        result = self.mw.wrap_tool_call(req, handler)
        handler.assert_called_once()
        assert result == "ok"

    def test_skill_out_of_bounds_blocked(self):
        tool = wrap_skill(SetBias, lambda: MagicMock(safe_call=lambda *a, **k: None))
        req = _make_request(tool, {"bias_v": 100.0})
        handler = MagicMock()
        result = self.mw.wrap_tool_call(req, handler)
        handler.assert_not_called()
        assert hasattr(result, "content")
        assert "global_bounds_violation" in result.content
        assert result.status == "error"

    def test_coarse_z_approach_blocked_on_agent_path(self):
        """2026-06-11: an open-loop coarse Z step toward the sample is the one
        physically-dangerous action; the autonomous agent path blocks it
        fail-closed (operator runs it manually via the GUI instead)."""
        from mast.skills.builtins.motor import MotorMove
        tool = wrap_skill(MotorMove, lambda: MagicMock(safe_call=lambda *a, **k: None))
        req = _make_request(tool, {"direction": "z-approach", "steps": 10})
        handler = MagicMock()
        result = self.mw.wrap_tool_call(req, handler)
        handler.assert_not_called()
        assert "coarse_sample_approach_blocked" in result.content
        assert result.status == "error"

    def test_motormove_lateral_is_not_the_autonomous_path(self):
        """2026-07-31: a BARE lateral coarse step is blocked — but relocating is not.

        This reverses the earlier expectation on purpose. A lateral move is not
        dangerous because it heads toward the sample (it does not); it is
        dangerous because ``MotorMove`` performs it with the tip hanging over the
        surface on ~1 µm of fine-Z clearance — a check that also passes when the
        state is unknown — with no coarse-Z retract, no chamber-pressure check,
        no drive-voltage readback, no current watch between chunks and no record
        of where the stage has already been.

        ``RelocateCoarseXY`` does all of that and stays CONFIRM, i.e. autonomous.
        Gating the raw primitive is what makes that meaningful: with both
        available the model would keep reaching for the cheaper one."""
        from mast.skills.builtins.motor import MotorMove
        tool = wrap_skill(MotorMove, lambda: MagicMock(safe_call=lambda *a, **k: None))
        req = _make_request(tool, {"direction": "x+", "steps": 10})
        handler = MagicMock(return_value="ok")
        result = self.mw.wrap_tool_call(req, handler)
        handler.assert_not_called()
        assert "unguarded_lateral_coarse_move_blocked" in result.content
        assert "RelocateCoarseXY" in result.content, (
            "a refusal that does not name the path that DOES work just stalls the run"
        )
        assert result.status == "error"

    def test_coarse_drive_change_is_blocked_on_the_autonomous_path(self):
        """The drive amplitude is the operator's parameter, not the agent's.

        Some controllers output 400 V; some stacks fail at 300; nothing reads
        back which rig this is; a mistake is unrecoverable. SetMotorFreqAmp is
        CONFIRM, and CONFIRM here means the model approves itself — so the level
        cannot express this and a Layer-0 gate must."""
        from mast.skills.builtins.motor import SetMotorFreqAmp
        tool = wrap_skill(SetMotorFreqAmp,
                          lambda: MagicMock(safe_call=lambda *a, **k: None))
        req = _make_request(tool, {"frequency_hz": 1000.0, "amplitude_v": 100.0,
                                   "axis": "all"})
        handler = MagicMock(return_value="ok")
        result = self.mw.wrap_tool_call(req, handler)
        handler.assert_not_called()
        assert "coarse_drive_change_blocked" in result.content
        assert result.status == "error"

    def test_motormove_z_retract_not_coarse_blocked(self):
        # z-retract steps AWAY from the sample → safe, not blocked.
        from mast.skills.builtins.motor import MotorMove
        tool = wrap_skill(MotorMove, lambda: MagicMock(safe_call=lambda *a, **k: None))
        req = _make_request(tool, {"direction": "z-retract", "steps": 10})
        handler = MagicMock(return_value="ok")
        result = self.mw.wrap_tool_call(req, handler)
        handler.assert_called_once()
        assert result == "ok"

    def test_handoff_tool_passes_through(self):
        """Tools without skill_metadata (e.g., handoff) bypass the gate."""
        from mast.agents._shared.handoff import make_handoff
        tool = make_handoff("data_processing", "test")
        req = _make_request(tool, {"reason": "done"})
        handler = MagicMock(return_value="handed_off")
        result = self.mw.wrap_tool_call(req, handler)
        handler.assert_called_once()
        assert result == "handed_off"

    def test_state_precondition_blocks(self):
        # Build a skill that requires z_controller_on
        class NeedsZ(BaseSkill):
            def metadata(self):
                return SkillMetadata(
                    name="NeedsZ",
                    safety_level=SafetyLevel.CONFIRM,
                    preconditions=["z_controller_on"],
                )

            def execute(self, ctx, params):
                return None

        tool = wrap_skill(NeedsZ, lambda: MagicMock())
        # State has z_controller_on=False → precondition should fail
        bad_state = HardwareState(z_controller_on=False)
        mw = SafetyGateMiddleware(SafetyLimits(), get_state=lambda: bad_state)
        req = _make_request(tool, {})
        handler = MagicMock()
        result = mw.wrap_tool_call(req, handler)
        handler.assert_not_called()
        assert "precondition_violation" in result.content


class TestSafetyGateRecorder:
    """RFC P2: the gate observes each verdict (allow/block) for training samples,
    read-only + fail-safe — the recorder must never alter the safety decision."""

    def test_recorder_captures_allow(self):
        rec = []
        mw = SafetyGateMiddleware(SafetyLimits(), recorder=rec.append)
        tool = wrap_skill(SetBias, lambda: MagicMock(safe_call=lambda *a, **k: None))
        req = _make_request(tool, {"bias_v": 1.0})
        mw.wrap_tool_call(req, MagicMock(return_value="ok"))
        assert len(rec) == 1
        assert rec[0]["verdict"] == "allow"
        assert rec[0]["skill"] == "SetBias"
        assert rec[0]["args"] == {"bias_v": 1.0}

    def test_recorder_captures_block(self):
        rec = []
        mw = SafetyGateMiddleware(SafetyLimits(), recorder=rec.append)
        tool = wrap_skill(SetBias, lambda: MagicMock(safe_call=lambda *a, **k: None))
        req = _make_request(tool, {"bias_v": 100.0})
        mw.wrap_tool_call(req, MagicMock())
        assert len(rec) == 1
        assert rec[0]["verdict"] == "block"
        assert "global_bounds_violation" in rec[0]["reason"]

    def test_recorder_failure_never_breaks_gate(self):
        def boom(_):
            raise ValueError("recorder broke")
        mw = SafetyGateMiddleware(SafetyLimits(), recorder=boom)
        tool = wrap_skill(SetBias, lambda: MagicMock(safe_call=lambda *a, **k: None))
        # in-bounds → handler must still run despite the recorder raising
        handler = MagicMock(return_value="ok")
        assert mw.wrap_tool_call(_make_request(tool, {"bias_v": 1.0}), handler) == "ok"
        # out-of-bounds → still blocked
        res = mw.wrap_tool_call(_make_request(tool, {"bias_v": 100.0}), MagicMock())
        assert getattr(res, "status", None) == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
