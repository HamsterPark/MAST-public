"""ApproachTip — deterministic, safe '进针' dispatcher (2026-07-01).

进针 is ambiguous + carries tip-crash risk, so the escalation decision must be made
IN CODE, not left to the LLM re-reading a flag. These tests pin that logic:
  - already / near tunnelling → engage only, NEVER a coarse approach
  - tip too far (needs_auto_approach) → escalate to the feedback-protected AutoApproach
  - engage failure / ambiguous outcome → STOP, never approach on a maybe
  - the open-loop coarse stepper (MotorMove z-approach) is NEVER invoked here
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.core.types import SkillResult  # noqa: E402
from mast.skills.builtins.approach import ApproachTip  # noqa: E402


class _FakeRunCtx:
    """Context whose .run(name, params) returns canned SkillResults + records calls."""

    def __init__(self, canned: dict):
        self.canned = canned
        self.runs: list = []

    def run(self, skill_name, params):
        self.runs.append((skill_name, params))
        r = self.canned.get(skill_name)
        if r is None:
            return SkillResult(skill_name=skill_name, success=False, error="unmocked")
        return r


def _eng(engaged: bool, needs_auto: bool) -> SkillResult:
    return SkillResult(skill_name="TryEngageController", success=True,
                       data={"engaged": engaged, "needs_auto_approach": needs_auto,
                             "peak_current_a": 1e-9, "setpoint_a": 1e-9})


def _reads(current_a: float = 5e-10, setpoint_a: float = 5e-10) -> dict:
    """Canned GetCurrent/GetSetpoint for the post-approach verification."""
    return {
        "GetCurrent": SkillResult(skill_name="GetCurrent", success=True,
                                  data={"current_a": current_a}),
        "GetSetpoint": SkillResult(skill_name="GetSetpoint", success=True,
                                   data={"setpoint_a": setpoint_a}),
    }


def _names(ctx) -> list[str]:
    return [n for n, _ in ctx.runs]


def test_already_tunnelling_engages_no_coarse_approach():
    ctx = _FakeRunCtx({"TryEngageController": _eng(True, False)})
    res = ApproachTip().execute(ctx, {})
    assert res.success
    assert res.data["method"] == "engage_controller"
    assert res.data["auto_approach_used"] is False
    # CRITICAL: AutoApproach must NOT run once feedback already engaged.
    assert _names(ctx) == ["TryEngageController"]


def test_tip_far_escalates_to_auto_approach():
    ctx = _FakeRunCtx({
        "TryEngageController": _eng(False, True),
        "AutoApproach": SkillResult(skill_name="AutoApproach", success=True, data={"ok": True}),
        **_reads(),  # tunnelling verified: |I| = setpoint
    })
    res = ApproachTip().execute(ctx, {})
    assert res.success
    assert res.data["method"] == "auto_approach"
    assert res.data["auto_approach_used"] is True
    assert res.data["measured_current_a"] == 5e-10
    # Two read ROUNDS, not one: the verdict now needs readings that agree with
    # each other (2026-08-05). A single GetCurrent/GetSetpoint pair here would
    # mean the settle judgement had been reverted to the racy instantaneous read.
    assert _names(ctx) == ["TryEngageController", "AutoApproach",
                           "GetCurrent", "GetSetpoint",
                           "GetCurrent", "GetSetpoint"]
    assert res.data["engagement"]["agreed_n"] >= 2


def test_approach_success_claim_rejected_without_current():
    """A success flag from AutoApproach must be rejected when independent synthetic readbacks remain below engagement thresholds."""
    ctx = _FakeRunCtx({
        "TryEngageController": _eng(False, True),
        "AutoApproach": SkillResult(skill_name="AutoApproach", success=True, data={}),
        **_reads(current_a=-3e-13, setpoint_a=8e-10),  # independent synthetic readbacks
    })
    res = ApproachTip().execute(ctx, {})
    assert not res.success
    assert res.data["engaged"] is False
    # Wording changed 2026-08-05; the contract did not. "稳定地" is the load-
    # bearing word — it says the reading was CONFIRMED low, not sampled once.
    assert "稳定地" in (res.error or "") and "未进入隧穿" in (res.error or "")
    assert res.data["measured_current_a"] == -3e-13
    assert res.data["setpoint_a"] == 8e-10


def test_approach_verification_defeats_lowered_setpoint():
    """An artificially lowered setpoint cannot turn synthetic noise above the relative threshold into engagement; the absolute floor still applies."""
    ctx = _FakeRunCtx({
        "TryEngageController": _eng(False, True),
        "AutoApproach": SkillResult(skill_name="AutoApproach", success=True, data={}),
        **_reads(current_a=-3e-13, setpoint_a=4e-13),
    })
    res = ApproachTip().execute(ctx, {})
    assert not res.success
    assert "稳定地" in (res.error or "") and "未进入隧穿" in (res.error or "")
    # The bar quoted is the 1 pA noise floor, not 50% of the synthetic 0.4 pA.
    assert "1.00 pA" in (res.error or "")


def test_engage_failure_stops_no_approach():
    ctx = _FakeRunCtx({"TryEngageController":
                       SkillResult(skill_name="TryEngageController", success=False, error="boom")})
    res = ApproachTip().execute(ctx, {})
    assert not res.success
    assert _names(ctx) == ["TryEngageController"]  # never approached after a failed engage


def test_ambiguous_engage_does_not_approach_on_a_maybe():
    # not engaged AND not needs_auto_approach → conservative stop (no motor).
    ctx = _FakeRunCtx({"TryEngageController": _eng(False, False)})
    res = ApproachTip().execute(ctx, {})
    assert not res.success
    assert "AutoApproach" not in _names(ctx)


def test_never_calls_open_loop_coarse_stepper():
    ctx = _FakeRunCtx({
        "TryEngageController": _eng(False, True),
        "AutoApproach": SkillResult(skill_name="AutoApproach", success=True, data={}),
        **_reads(),
    })
    ApproachTip().execute(ctx, {})
    # The one crash-risky action — open-loop MotorMove z-approach — is never used.
    assert "MotorMove" not in _names(ctx)


def test_auto_approach_failure_surfaced():
    ctx = _FakeRunCtx({
        "TryEngageController": _eng(False, True),
        "AutoApproach": SkillResult(skill_name="AutoApproach", success=False, error="tcp lost"),
    })
    res = ApproachTip().execute(ctx, {})
    assert not res.success
    assert "auto-approach phase failed" in (res.error or "")


def test_metadata_is_auto_and_tagged_for_jinzhen():
    m = ApproachTip().metadata()
    assert m.safety_level.value == "auto"      # both phases feedback-protected
    assert m.name == "ApproachTip"
    assert "进针" in m.tags and "approach" in m.tags


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
