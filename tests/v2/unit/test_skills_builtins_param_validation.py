"""Group `skills_builtins` — 审查 findings [#31] and [#10].

[#31] HIGH — Optional skill params without an explicit `default` become None
      and fail validation, making core write skills (incl. SetBias) silently
      unusable through the agent path.
      FIX: BaseSkill.validate_params now tolerates a None value for an OPTIONAL
      (required=False) param — it means "not provided", and execute() bodies
      already fall back via params.get(name, <default>). Type/range/allowed
      checks still fire for provided values; required params are still enforced.

[#10] RESOLVED (2026-06-11 safety re-scoping) — no builtin skill is DANGEROUS.
      The safety model was re-scoped to the single action that can physically
      damage the instrument: an OPEN-LOOP coarse Z step TOWARD the sample
      (MotorMove direction='z-approach', a pan-type piezo stepper with no
      current feedback to self-stop). Everything else has a Nanonis backstop
      (bias/current/fine-Z ranges, AutoApproach current feedback) and cannot
      wreck the rig, so nothing needs the DANGEROUS HITL pane.
      The dangerous case is NOT gated by safety_level — it is detected by
      mast.core.safety.is_coarse_sample_approach(skill_name, params): the agent
      path BLOCKS it fail-closed (SafetyGateMiddleware), the executor path
      forces human approval. As a result the re-classified safety levels are:
        BiasPulse / TipShape / TipShapeWithReadback / AutoApproach → AUTO
        MotorMove (baseline) / MotorMoveClosedLoop → CONFIRM
        SetBiasCalibration / SetCurrentCalibration → CONFIRM (metadata write)
      _derive_hitl_map() now returns {} and the HITL middleware is a no-op (no
      static DANGEROUS skill remains). The tests below pin this re-scoped
      contract so any regression toward the old DANGEROUS model is caught.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/test_skills_builtins_param_validation.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (MASTv2 must shadow v1 mast/) ──
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

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.builtins.bias import SetBias, SetBiasCalibration
from mast.skills.builtins.current import SetCurrentCalibration
from mast.skills.builtins.motor import MotorMove, MotorMoveClosedLoop
from mast.skills.builtins.bias_pulse import BiasPulse
from mast.skills.builtins.tip_shaper import TipShape


# ──────────────────────────────────────────────────────────────────────────
# Fake ExecutionContext — records safe_call, returns canned NanonisCallRecord.
# Tests the REAL validate_params / execute, never mocks the skill itself.
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        entry = self.canned.get(method, {})
        return NanonisCallRecord(
            method=method,
            args=args,
            return_value=entry.get("return_value"),
            error=entry.get("error", ""),
        )


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# A synthetic skill exercising every optional/required × default combination.
class _OptParamSkill(BaseSkill):
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="OptParamSkill",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="",
            parameters=[
                ParameterSpec(name="req_f", type="float", required=True),
                # optional, NO explicit default → default is None
                ParameterSpec(
                    name="opt_f", type="float", required=False,
                    min_value=0.01, max_value=100.0,
                ),
                ParameterSpec(name="opt_i", type="int", required=False),
                ParameterSpec(
                    name="opt_choice", type="str", required=False,
                    allowed_values=["a", "b"],
                ),
            ],
        )

    def execute(self, context, params: dict) -> SkillResult:
        return SkillResult(skill_name="OptParamSkill", success=True, data=dict(params))


# ──────────────────────────────────────────────────────────────────────────
# [#31] validate_params: optional None tolerated, everything else still checked
# ──────────────────────────────────────────────────────────────────────────

class TestOptionalNoneValidation:
    def test_setbias_optional_none_no_longer_errors(self):
        """The exact reproduction from finding #31: SetBias + slew=None."""
        errs = SetBias().validate_params(
            {"bias_v": 0.5, "slew_rate_v_per_s": None}
        )
        assert errs == []

    def test_setbias_optional_provided_value_still_validated(self):
        # out of range (max 100) must still be rejected
        errs = SetBias().validate_params(
            {"bias_v": 0.5, "slew_rate_v_per_s": 999.0}
        )
        assert any("above maximum" in e for e in errs)

    def test_setbias_optional_wrong_type_still_rejected(self):
        errs = SetBias().validate_params(
            {"bias_v": 0.5, "slew_rate_v_per_s": "fast"}
        )
        assert any("must be float" in e for e in errs)

    def test_required_missing_still_rejected(self):
        errs = SetBias().validate_params({"slew_rate_v_per_s": None})
        assert any("Missing required" in e for e in errs)

    def test_required_explicit_none_still_rejected(self):
        # A required param explicitly None is NOT a valid value.
        errs = SetBias().validate_params({"bias_v": None})
        assert any("must be float" in e for e in errs)

    def test_multiple_optional_none_all_skipped(self):
        s = _OptParamSkill()
        errs = s.validate_params(
            {"req_f": 1.0, "opt_f": None, "opt_i": None, "opt_choice": None}
        )
        assert errs == []

    def test_optional_none_does_not_bypass_required_check(self):
        # opt_* all None but req_f missing → still flagged.
        s = _OptParamSkill()
        errs = s.validate_params({"opt_f": None, "opt_i": None})
        assert any("Missing required" in e and "req_f" in e for e in errs)

    def test_optional_choice_provided_invalid_still_rejected(self):
        s = _OptParamSkill()
        errs = s.validate_params({"req_f": 1.0, "opt_choice": "zzz"})
        assert any("not in" in e for e in errs)


# ──────────────────────────────────────────────────────────────────────────
# [#31] End-to-end through the agent tool path (wrap_skill → _run →
# validate_params → execute). This is the path that was BROKEN: pydantic
# materialises the omitted optional param as None and forwards it.
# ──────────────────────────────────────────────────────────────────────────

class TestSetBiasAgentPathEndToEnd:
    def test_setbias_omitting_slew_executes(self):
        canned = {"Bias_Set": {"return_value": ("", b"", [])}}
        tool = wrap_skill(SetBias, make_provider(canned))
        # pydantic injects slew_rate_v_per_s=None when the LLM omits it
        res = _invoke(tool, bias_v=0.5, slew_rate_v_per_s=None)
        msg = res.update["messages"][0]
        assert msg.status == "success"
        assert "precondition_failed" not in str(res)
        assert res.update["executed_skills"] == ["SetBias"]

    def test_setbias_actually_calls_bias_set(self):
        canned = {"Bias_Set": {"return_value": ("", b"", [])}}
        instances: list[FakeCtx] = []

        def capturing():
            ctx = FakeCtx(canned=canned)
            instances.append(ctx)
            return ctx

        tool = wrap_skill(SetBias, capturing)
        _invoke(tool, bias_v=0.5, slew_rate_v_per_s=None)
        calls = [c for c in instances[-1].calls if c[0] == "Bias_Set"]
        assert calls and calls[0][1] == (0.5,)

    def test_setbias_with_slew_still_ramps(self):
        # Sanity: when slew IS provided, the ramp path still works.
        canned = {
            "Bias_Get": {"return_value": ("", b"", [0.0])},
            "Bias_Set": {"return_value": ("", b"", [])},
        }
        instances: list[FakeCtx] = []

        def capturing():
            ctx = FakeCtx(canned=canned)
            instances.append(ctx)
            return ctx

        tool = wrap_skill(SetBias, capturing)
        # Large step + fast slew → at least one ramped Bias_Set
        res = _invoke(tool, bias_v=5.0, slew_rate_v_per_s=100.0)
        assert res.update["messages"][0].status == "success"
        bias_sets = [c for c in instances[-1].calls if c[0] == "Bias_Set"]
        assert len(bias_sets) >= 1


# ──────────────────────────────────────────────────────────────────────────
# [#10/#84] RESOLVED (2026-06-11 safety re-scoping) — no builtin is DANGEROUS.
# The danger model now targets ONE physical hazard (open-loop coarse Z step
# toward the sample) via is_coarse_sample_approach(), not via safety_level. The
# write skills that were briefly DANGEROUS are re-classified: BiasPulse/TipShape
# (and TipShapeWithReadback/AutoApproach) → AUTO, since Nanonis range/feedback
# backstops prevent instrument damage; MotorMove (baseline) + MotorMoveClosedLoop
# stay CONFIRM; the calibration skills stay CONFIRM (metadata write). With no
# static DANGEROUS skill, _derive_hitl_map() returns {} and the HITL middleware
# is a no-op — test_hitl_derivation.py + test_safety_mw.py guard that and the
# coarse-approach gate. This test pins the re-scoped safety_level contract.
# ──────────────────────────────────────────────────────────────────────────

class TestDangerLevelContract:
    # (skill, name, expected level) — post 2026-06-11 re-scoping: NO builtin is
    # DANGEROUS. BiasPulse/TipShape are AUTO (Nanonis backstops); MotorMove +
    # MotorMoveClosedLoop are CONFIRM (the z-approach hazard is param-gated, not
    # level-gated); the calibration skills stay CONFIRM (metadata write).
    @pytest.mark.parametrize("skill_cls,name,level", [
        (MotorMove, "MotorMove", SafetyLevel.CONFIRM),
        (BiasPulse, "BiasPulse", SafetyLevel.AUTO),
        (TipShape, "TipShape", SafetyLevel.AUTO),
        (MotorMoveClosedLoop, "MotorMoveClosedLoop", SafetyLevel.CONFIRM),
        (SetBiasCalibration, "SetBiasCalibration", SafetyLevel.CONFIRM),
        (SetCurrentCalibration, "SetCurrentCalibration", SafetyLevel.CONFIRM),
    ])
    def test_post_hitl_repair_safety_contract(self, skill_cls, name, level):
        """Re-scoped contract (2026-06-11): no builtin skill is DANGEROUS.
        BiasPulse/TipShape are AUTO; MotorMove/MotorMoveClosedLoop/calibration
        stay CONFIRM. The genuine hazard (open-loop coarse Z-approach) is gated
        by is_coarse_sample_approach(params), not by safety_level."""
        meta = skill_cls().metadata()
        assert meta.name == name
        assert meta.safety_level == level
        # No builtin may be DANGEROUS under the re-scoped model.
        assert meta.safety_level != SafetyLevel.DANGEROUS

    @pytest.mark.parametrize("skill_cls", [
        MotorMove, MotorMoveClosedLoop, BiasPulse, TipShape,
        SetBiasCalibration, SetCurrentCalibration,
    ])
    def test_danger_level_metadata_matches_safety_level(self, skill_cls):
        """tool.metadata['danger_level'] must mirror safety_level.name — proves
        the danger_level stamp is not silently diverging from the source."""
        tool = wrap_skill(skill_cls, make_provider())
        meta = skill_cls().metadata()
        assert tool.metadata["danger_level"] == meta.safety_level.name


# ──────────────────────────────────────────────────────────────────────────
# Safety fix #3: calibration/scale-factor setters now carry sane numeric bounds
# so an agent can't set an extreme multiplier/offset that silently defeats the
# ±10 V bias / setpoint bounds. The multiplier is bound 1e-3 … 1e3 (rejects 0,
# negatives, and absurd factors); offsets carry a generous-but-finite ceiling.
# ──────────────────────────────────────────────────────────────────────────


def _spec(skill_cls, name):
    return next(p for p in skill_cls().metadata().parameters if p.name == name)


class TestCalibrationBounds:
    def test_bias_calibration_multiplier_bounds(self):
        spec = _spec(SetBiasCalibration, "calibration")
        assert spec.min_value == 1e-3
        assert spec.max_value == 1e3
        # 0, negative, and absurd factors are rejected.
        assert any("calibration" in e for e in
                   SetBiasCalibration().validate_params({"calibration": 0.0, "offset": 0.0}))
        assert any("calibration" in e for e in
                   SetBiasCalibration().validate_params({"calibration": -1.0, "offset": 0.0}))
        assert any("calibration" in e and "maximum" in e for e in
                   SetBiasCalibration().validate_params({"calibration": 1e6, "offset": 0.0}))
        # A normal calibration (~1.0) with a sane offset passes.
        assert SetBiasCalibration().validate_params({"calibration": 1.0, "offset": 0.0}) == []

    def test_bias_calibration_offset_bounds(self):
        spec = _spec(SetBiasCalibration, "offset")
        assert spec.min_value == -100.0
        assert spec.max_value == 100.0
        assert any("offset" in e and "maximum" in e for e in
                   SetBiasCalibration().validate_params({"calibration": 1.0, "offset": 1e6}))

    def test_current_calibration_multiplier_bounds(self):
        spec = _spec(SetCurrentCalibration, "calibration")
        assert spec.min_value == 1e-3
        assert spec.max_value == 1e3
        assert any("calibration" in e for e in
                   SetCurrentCalibration().validate_params({"calibration": 0.0, "offset": 0.0}))
        assert any("calibration" in e and "maximum" in e for e in
                   SetCurrentCalibration().validate_params({"calibration": 1e9, "offset": 0.0}))
        assert SetCurrentCalibration().validate_params({"calibration": 1.0, "offset": 0.0}) == []

    def test_current_calibration_offset_bounds(self):
        spec = _spec(SetCurrentCalibration, "offset")
        assert spec.min_value == -1.0
        assert spec.max_value == 1.0
        assert any("offset" in e and "maximum" in e for e in
                   SetCurrentCalibration().validate_params({"calibration": 1.0, "offset": 5.0}))


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
