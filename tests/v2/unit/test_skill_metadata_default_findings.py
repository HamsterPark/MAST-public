"""审查 findings #138 / #139 / #144 / #86 / #145.

#138  builtins/sweep.py — execute() fallback for an omitted optional param
      diverged from the ParameterSpec default. On the composite path defaults are
      NOT back-filled, so the (different) execute fallback silently changed
      behaviour. FIX: align the .get fallback with the spec default.

#139  builtins/imaging.py SetScanSpeed — execute() read params.get("speed_ratio")
      but the param was never declared (dead read, permanently 1). FIX: declare a
      ParameterSpec(speed_ratio, default 1.0) so it is actually settable.

#144  core/registry.py register() silently overwrote a same name+version skill.
      FIX: warn on a genuine (different-class) collision; stay quiet on idempotent
      re-registration of the same class.

#86   skills/base.py precondition word-list had no single source of truth shared
      with the SafetyGate layer. FIX: centralise into mast.core.preconditions;
      base.py imports PRECONDITION_CHECKS from there.

#145  skills/base.py validate_params mutated the CALLER's dict (int→float). FIX:
      coerce only a local copy; never write back into the caller's dict.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/test_skill_metadata_default_findings.py -q -p no:randomly
"""
from __future__ import annotations

# ── path bootstrap BEFORE any mast.* import (MASTv2 must shadow v1 mast/) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.core.registry import SkillRegistry
from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.builtins.imaging import SetScanSpeed
from mast.skills.builtins.sweep import ConfigureBiasSweep, ConfigureLockInSweep


# ──────────────────────────────────────────────────────────────────────────
# Fake ExecutionContext — records safe_call, returns canned records.
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


def _arg_for(ctx: FakeCtx, method: str, index: int):
    """Return positional arg `index` from the first call to `method`."""
    for m, args in ctx.calls:
        if m == method:
            return args[index]
    raise AssertionError(f"{method} was never called; calls={ctx.calls}")


# ──────────────────────────────────────────────────────────────────────────
# #138 — sweep fallbacks equal the declared ParameterSpec defaults.
# ──────────────────────────────────────────────────────────────────────────
class TestSweepDefaultParity:
    def _spec_default(self, meta: SkillMetadata, name: str):
        return next(p.default for p in meta.parameters if p.name == name)

    def test_bias_sweep_period_fallback_matches_spec_default(self):
        skill = ConfigureBiasSweep()
        spec_default = self._spec_default(skill.metadata(), "period_ms")
        assert spec_default == 4.0  # the single source of truth
        # Composite path: period_ms omitted → execute() must use the SAME default.
        ctx = FakeCtx()
        res = skill.execute(ctx, {"lower_v": -1.0, "upper_v": 1.0, "num_steps": 100})
        assert res.success
        # GenSwp_PropsSet(settle, slew, num_steps, period_ms, ...) — period is arg 3
        sent_period = _arg_for(ctx, "GenSwp_PropsSet", 3)
        assert sent_period == spec_default, (
            f"execute used {sent_period}, spec default is {spec_default}"
        )
        assert res.data["period_ms"] == spec_default

    def test_bias_sweep_explicit_period_still_honoured(self):
        skill = ConfigureBiasSweep()
        ctx = FakeCtx()
        skill.execute(ctx, {"lower_v": -1.0, "upper_v": 1.0,
                            "num_steps": 100, "period_ms": 12.5})
        assert _arg_for(ctx, "GenSwp_PropsSet", 3) == 12.5

    def test_lockin_sweep_period_fallbacks_match_spec_defaults(self):
        skill = ConfigureLockInSweep()
        meta = skill.metadata()
        int_default = self._spec_default(meta, "integration_periods")
        set_default = self._spec_default(meta, "settling_periods")
        assert (int_default, set_default) == (1, 1)
        ctx = FakeCtx()
        res = skill.execute(ctx, {"lower_hz": 1.0, "upper_hz": 100.0, "num_steps": 50})
        assert res.success
        # LockInFreqSwp_PropsSet(num_steps, int_periods, _, set_periods, ...)
        assert _arg_for(ctx, "LockInFreqSwp_PropsSet", 1) == int_default
        assert _arg_for(ctx, "LockInFreqSwp_PropsSet", 3) == set_default


# ──────────────────────────────────────────────────────────────────────────
# #139 — SetScanSpeed.speed_ratio is a real declared & settable parameter.
# ──────────────────────────────────────────────────────────────────────────
class TestSetScanSpeedDeclaresSpeedRatio:
    def test_speed_ratio_is_declared(self):
        meta = SetScanSpeed().metadata()
        names = {p.name for p in meta.parameters}
        assert "speed_ratio" in names, "speed_ratio must be a declared ParameterSpec"
        spec = next(p for p in meta.parameters if p.name == "speed_ratio")
        assert spec.required is False
        assert spec.default == 1.0
        assert spec.type == "float"

    def test_speed_ratio_default_sent_to_nanonis(self):
        ctx = FakeCtx()
        res = SetScanSpeed().execute(ctx, {
            "fwd_speed": 1e-7, "bwd_speed": 1e-7,
            "fwd_line_time": 0.1, "bwd_line_time": 0.1,
        })
        assert res.success
        # Scan_SpeedSet(fwd, bwd, fwd_t, bwd_t, keep_const, speed_ratio) — arg 5
        assert _arg_for(ctx, "Scan_SpeedSet", 5) == 1.0
        assert res.data["speed_ratio"] == 1.0

    def test_speed_ratio_now_settable(self):
        """The whole point of #139: a caller-supplied speed_ratio reaches HW."""
        ctx = FakeCtx()
        SetScanSpeed().execute(ctx, {
            "fwd_speed": 1e-7, "bwd_speed": 2e-7,
            "fwd_line_time": 0.1, "bwd_line_time": 0.2,
            "speed_ratio": 2.0,
        })
        assert _arg_for(ctx, "Scan_SpeedSet", 5) == 2.0

    def test_speed_ratio_validates_through_base(self):
        # Declared param → validate_params now knows it; a bad type is rejected.
        errs = SetScanSpeed().validate_params({
            "fwd_speed": 1e-7, "bwd_speed": 1e-7,
            "fwd_line_time": 0.1, "bwd_line_time": 0.1,
            "speed_ratio": "fast",
        })
        assert any("speed_ratio" in e and "must be float" in e for e in errs)


# ──────────────────────────────────────────────────────────────────────────
# #144 — registry warns on a genuine name+version collision, not on a re-run.
# ──────────────────────────────────────────────────────────────────────────
def _mk_skill(cls_name: str, skill_name: str, version: str = "1.0.0"):
    def metadata(self):
        return SkillMetadata(
            name=skill_name, version=version, category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO, description="",
        )

    def execute(self, ctx, params):
        return SkillResult(skill_name=skill_name, success=True)

    return type(cls_name, (BaseSkill,), {"metadata": metadata, "execute": execute})


class TestRegistryCollisionWarning:
    def test_distinct_class_same_name_version_warns(self, caplog):
        reg = SkillRegistry()
        A = _mk_skill("A_impl", "DupSkill", "1.0.0")
        B = _mk_skill("B_impl", "DupSkill", "1.0.0")
        reg.register(A)
        with caplog.at_level(logging.WARNING, logger="mast.core.registry"):
            reg.register(B)
        assert any("collision" in r.message.lower() for r in caplog.records), \
            "a genuine name+version collision must be logged"
        # Last-writer-wins behaviour is preserved (no crash).
        assert reg.get("DupSkill") is B

    def test_reregistering_same_class_is_quiet(self, caplog):
        reg = SkillRegistry()
        A = _mk_skill("A_again", "IdemSkill", "1.0.0")
        reg.register(A)
        with caplog.at_level(logging.WARNING, logger="mast.core.registry"):
            reg.register(A)  # idempotent discover() re-run
        assert not any("collision" in r.message.lower() for r in caplog.records), \
            "re-registering the SAME class must not warn"

    def test_real_discovery_has_no_collisions(self, caplog):
        """The shipped builtins must not collide among themselves."""
        reg = SkillRegistry()
        with caplog.at_level(logging.WARNING, logger="mast.core.registry"):
            reg.discover("mast.skills.builtins")
        collisions = [r for r in caplog.records if "collision" in r.message.lower()]
        assert collisions == [], f"unexpected builtin collisions: {[c.message for c in collisions]}"


# ──────────────────────────────────────────────────────────────────────────
# #86 — single source of truth for the precondition vocabulary.
# ──────────────────────────────────────────────────────────────────────────
class TestPreconditionSingleSource:
    def test_base_imports_from_shared_module(self):
        from mast.core.preconditions import PRECONDITION_CHECKS
        from mast.skills.base import _PRECONDITION_CHECKS
        # Same object → genuinely a single source of truth, not a copy.
        assert _PRECONDITION_CHECKS is PRECONDITION_CHECKS

    def test_canonical_keys_present(self):
        from mast.core.preconditions import PRECONDITION_CHECKS
        assert PRECONDITION_CHECKS["z_controller_off"] == ("z_controller_on", False)
        assert PRECONDITION_CHECKS["scan_not_running"] == ("scan_running", False)
        assert PRECONDITION_CHECKS["scan_stopped"] == ("scan_running", False)

    def test_shared_substring_enforcement_matches_legacy_strings(self):
        from mast.core.preconditions import check_state_preconditions
        from mast.core.types import HardwareState
        # z_controller_on with controller OFF → exact legacy message.
        v = check_state_preconditions(["z_controller_on"],
                                      HardwareState(z_controller_on=False))
        assert v and "Z controller is OFF" in v[0]
        # met → no violation
        assert check_state_preconditions(["z_controller_on"],
                                         HardwareState(z_controller_on=True)) == []
        # bias_nonzero only fires on explicit 0.0, never on unknown.
        assert check_state_preconditions(["bias_nonzero"],
                                         HardwareState(bias_v=0.0))
        assert check_state_preconditions(["bias_nonzero"],
                                         HardwareState(bias_v=None)) == []

    def test_scan_stopped_now_enforced_in_substring_layer(self):
        """Previously 'scan_stopped' matched NOTHING in the safety substring chain
        (the gap #86 is about). Now it is enforced consistently with base.py."""
        from mast.core.preconditions import check_state_preconditions
        from mast.core.types import HardwareState
        v = check_state_preconditions(["scan_stopped"],
                                      HardwareState(scan_running=True))
        assert v and "scan is running" in v[0]
        assert check_state_preconditions(["scan_stopped"],
                                         HardwareState(scan_running=False)) == []


# ──────────────────────────────────────────────────────────────────────────
# #145 — validate_params must NOT mutate the caller's dict.
# ──────────────────────────────────────────────────────────────────────────
class _FloatParamSkill(BaseSkill):
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="FloatParamSkill", version="1.0.0", category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM, description="",
            parameters=[
                ParameterSpec(name="x", type="float", required=True,
                              min_value=0.0, max_value=10.0),
            ],
        )

    def execute(self, ctx, params) -> SkillResult:
        return SkillResult(skill_name="FloatParamSkill", success=True, data=dict(params))


class TestValidateParamsNoMutation:
    def test_int_for_float_does_not_mutate_caller_dict(self):
        s = _FloatParamSkill()
        params = {"x": 3}            # int passed for a float param
        errs = s.validate_params(params)
        assert errs == []           # int-for-float is accepted
        # The caller's dict is UNTOUCHED — still the original int.
        assert params == {"x": 3}
        assert type(params["x"]) is int

    def test_range_check_still_uses_coerced_local_value(self):
        s = _FloatParamSkill()
        # 11 (int) for a float param with max 10 → still rejected via local coercion.
        errs = s.validate_params({"x": 11})
        assert any("above maximum" in e for e in errs)

    def test_real_setbias_int_bias_not_mutated(self):
        from mast.skills.builtins.bias import SetBias
        params = {"bias_v": 1}      # int for the float bias_v param
        errs = SetBias().validate_params(params)
        assert errs == []
        assert params == {"bias_v": 1}
        assert type(params["bias_v"]) is int


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
