"""Regression pins for the 2026-07-03 review — approach / Z / motor safety.

Covers:
  * AutoApproach gains a bias_nonzero precondition (no coarse approach at 0 V,
    where the hardware current-feedback stop can't fire).
  * StopAutoApproach skill exists and is AUTO.
  * TryEngageController fails safe (needs_auto_approach=False) when the current
    chain is dead — it must NOT recommend a blind coarse approach.
  * MotorMove refuses an unknown direction instead of silently moving X+, and
    refuses a lateral move when the tip is known not-withdrawn.
  * is_protection_disable gates turning OFF SafeTip / Z-limits.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/builtins/test_review_2026_07_03_approach.py -q
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.core.safety import is_protection_disable
from mast.core.types import HardwareState, NanonisCallRecord
from mast.skills.builtins.approach import AutoApproach, StopAutoApproach
from mast.skills.builtins.zcontrol import TryEngageController
from mast.skills.builtins.motor import MotorMove


@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list = field(default_factory=list)
    state_obj: Any = None

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        entry = self.canned.get(method)
        if entry is None:
            return NanonisCallRecord(method=method, args=args, error=f"unmocked:{method}")
        return NanonisCallRecord(method=method, args=args,
                                 return_value=entry.get("return_value"),
                                 error=entry.get("error", ""))

    @property
    def state(self):
        return self

    def snapshot(self):
        return self.state_obj


# ── AutoApproach precondition ────────────────────────────────────────────────
def test_auto_approach_requires_nonzero_bias():
    meta = AutoApproach().metadata()
    assert "bias_nonzero" in meta.preconditions


# ── StopAutoApproach exists + AUTO ───────────────────────────────────────────
def test_stop_auto_approach_skill():
    meta = StopAutoApproach().metadata()
    assert meta.name == "StopAutoApproach"
    assert meta.safety_level.value in ("auto", "AUTO") or str(meta.safety_level).endswith("AUTO")
    ctx = FakeCtx(canned={"AutoApproach_OnOffSet": {"return_value": ("", b"", [])}})
    res = StopAutoApproach().execute(ctx, {})
    assert res.success
    assert ("AutoApproach_OnOffSet", (0,)) in ctx.calls


# ── TryEngageController fail-safe on dead current chain ───────────────────────
def test_try_engage_failsafe_when_current_chain_dead():
    # setpoint reads fine, but EVERY Current_Get errors → must NOT recommend a
    # coarse approach (needs_auto_approach=False) and must report failure.
    ctx = FakeCtx(canned={
        "ZCtrl_SetpntGet": {"return_value": ("", b"", [1e-9])},
        "ZCtrl_OnOffSet": {"return_value": ("", b"", [])},
        # Current_Get unmocked → error every poll.
    })
    res = TryEngageController().execute(ctx, {"settle_s": 0.02, "poll_hz": 2})
    assert res.success is False
    assert res.data.get("needs_auto_approach") is False


def test_try_engage_recommends_approach_when_measurable_but_far():
    # Valid current reads, but current stays far below setpoint → legitimately
    # recommends a coarse approach.
    #
    # 2026-07-13: it must ALSO confirm the Z loop is really open before doing so —
    # ZCtrl_OnOffGet asks the real-time controller, which Nanonis' manual says is the
    # only way to know (the Z-Controller module can still read "Off" while the RT
    # controller has not caught up). A rig where that read is missing or says ON no
    # longer gets a coarse-approach recommendation; see test_try_engage.py for those.
    ctx = FakeCtx(canned={
        "ZCtrl_SetpntGet": {"return_value": ("", b"", [1e-9])},
        "ZCtrl_OnOffSet": {"return_value": ("", b"", [])},
        "ZCtrl_OnOffGet": {"return_value": ("", b"", [0])},   # RT confirms: loop OPEN
        "Current_Get": {"return_value": ("", b"", [1e-12])},  # 1000x below setpoint
    })
    res = TryEngageController().execute(ctx, {"settle_s": 0.02, "poll_hz": 2})
    assert res.success is True
    assert res.data.get("needs_auto_approach") is True
    assert res.data.get("z_controller_verified") is True


# ── MotorMove direction + withdrawn ──────────────────────────────────────────
def test_motor_move_unknown_direction_fails():
    ctx = FakeCtx(canned={"Motor_StartMove": {"return_value": ("", b"", [])}})
    res = MotorMove().execute(ctx, {"direction": "sideways", "steps": 5})
    assert not res.success
    assert "unknown motor direction" in (res.error or "")
    # Must NOT have issued a move.
    assert not any(c[0] == "Motor_StartMove" for c in ctx.calls)


def test_motor_move_lateral_blocked_when_not_withdrawn():
    st = HardwareState()
    st.withdrawn = False
    ctx = FakeCtx(canned={"Motor_StartMove": {"return_value": ("", b"", [])}}, state_obj=st)
    res = MotorMove().execute(ctx, {"direction": "x+", "steps": 5})
    assert not res.success
    assert "withdrawn" in (res.error or "")


def test_motor_move_lateral_ok_when_withdrawn():
    st = HardwareState()
    st.withdrawn = True
    ctx = FakeCtx(canned={"Motor_StartMove": {"return_value": ("", b"", [])}}, state_obj=st)
    res = MotorMove().execute(ctx, {"direction": "x+", "steps": 5})
    assert res.success


def test_motor_move_zapproach_not_blocked_by_withdrawn_check():
    # z-approach legitimately steps a non-withdrawn tip (human-gated elsewhere).
    st = HardwareState()
    st.withdrawn = False
    ctx = FakeCtx(canned={"Motor_StartMove": {"return_value": ("", b"", [])}}, state_obj=st)
    res = MotorMove().execute(ctx, {"direction": "z-approach", "steps": 5})
    assert res.success


# ── Protection-disable gate ──────────────────────────────────────────────────
def test_is_protection_disable():
    assert is_protection_disable("EnableSafeTip", {"enable": False}) is True
    assert is_protection_disable("EnableSafeTip", {"enable": "false"}) is True
    assert is_protection_disable("EnableSafeTip", {"enable": True}) is False
    assert is_protection_disable("EnableSafeTip", {}) is False  # default enable
    assert is_protection_disable("SetZLimitsEnabled", {"enabled": False}) is True
    assert is_protection_disable("SetZLimitsEnabled", {"enabled": True}) is False
    assert is_protection_disable("SetBias", {"bias_v": 1.0}) is False
