"""TryEngageController — smart '进针/engage' skill (idea #2).

Try Z-controller ON; if tunneling reaches ~setpoint → engaged (feedback left ON);
else turn feedback back OFF and flag needs_auto_approach. No coarse motor move.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.skills.builtins.zcontrol import TryEngageController


class _Rec:
    def __init__(self, error="", return_value=None):
        self.error = error
        self.return_value = return_value


class FakeCtx:
    """A fake that MODELS THE HARDWARE rather than just echoing calls.

    2026-07-13: it now tracks the Z-controller's actual state and answers
    ``ZCtrl_OnOffGet`` — the real-time controller's view — with it. Until today
    nothing asked, so the fake did not have to know; TryEngageController simply
    *claimed* the loop was off, and that claim is what authorises the open-loop
    coarse approach. ``rt_stuck_on`` and ``fail_onoff_get`` let a test pretend the
    write did not land or the readback failed, which is the case the old code could
    not even see. See mast.skills.verify.
    """

    def __init__(self, setpoint=1e-9, currents=None, fail_setpoint=False,
                 rt_stuck_on=False, fail_onoff_get=False):
        self.setpoint = setpoint
        self.currents = currents if currents is not None else [0.0]
        self.fail_setpoint = fail_setpoint
        # rt_stuck_on: the OFF write returns OK over TCP and the real-time controller
        # keeps the loop CLOSED. This is the case the old code could not see at all —
        # and the one where it would have told the agent to run a coarse approach.
        self.rt_stuck_on = rt_stuck_on
        self.fail_onoff_get = fail_onoff_get  # the readback itself fails
        self.calls = []
        self._ci = 0
        self._z_on = 0                        # the hardware's ACTUAL state

    def safe_call(self, method, *args):
        self.calls.append((method, args))
        if method == "ZCtrl_SetpntGet":
            if self.fail_setpoint:
                return _Rec(error="no setpoint")
            return _Rec(return_value=("", None, [self.setpoint]))
        if method == "ZCtrl_OnOffSet":
            want = int(args[0])
            if want == 1 or not self.rt_stuck_on:
                self._z_on = want
            # want == 0 and rt_stuck_on → the write is ACKed and simply not applied
            return _Rec(return_value=("", None, [args[0]]))
        if method == "ZCtrl_OnOffGet":
            if self.fail_onoff_get:
                return _Rec(error="TCP timeout")
            return _Rec(return_value=("", None, [self._z_on]))
        if method == "Current_Get":
            v = self.currents[min(self._ci, len(self.currents) - 1)]
            self._ci += 1
            return _Rec(return_value=("", None, [v]))
        return _Rec()


def _onoff_calls(ctx):
    return [a[0] for (m, a) in ctx.calls if m == "ZCtrl_OnOffSet"]


def test_engaged_leaves_feedback_on():
    ctx = FakeCtx(setpoint=1e-9, currents=[0.8e-9] * 20)
    res = TryEngageController().execute(ctx, {"settle_s": 0.05, "poll_hz": 10})
    assert res.success
    assert res.data["engaged"] is True
    assert res.data["z_controller_on"] is True
    assert res.data["needs_auto_approach"] is False
    assert _onoff_calls(ctx) == [1]  # turned ON, never OFF


def test_not_engaged_turns_feedback_back_off():
    ctx = FakeCtx(setpoint=1e-9, currents=[1e-12] * 20)  # nowhere near setpoint
    res = TryEngageController().execute(ctx, {"settle_s": 0.05, "poll_hz": 10})
    assert res.success  # the skill succeeded at *checking* — not a failure
    assert res.data["engaged"] is False
    assert res.data["z_controller_on"] is False
    assert res.data["needs_auto_approach"] is True
    assert _onoff_calls(ctx) == [1, 0]  # ON to try, then OFF to clean up
    assert "AutoApproach" in res.data["message"]
    # 2026-07-13: and that OFF is now a READING, not a claim.
    assert res.data["z_controller_verified"] is True
    assert any(m == "ZCtrl_OnOffGet" for m, _ in ctx.calls), (
        "没有向实时控制器确认 Z 反馈已断开就建议粗进针"
    )


def test_coarse_approach_refused_when_the_off_write_did_not_land():
    """The OFF is ACKed over TCP and the real-time controller keeps the loop closed.

    Old behaviour: the skill reported z_controller_on=False (its own request, echoed)
    and needs_auto_approach=True — sending the open-loop coarse stepper toward the
    surface with the feedback still driving Z. Nanonis' manual warns explicitly that
    the module and the RT controller disagree during the communication delay, and
    that ZCtrl_OnOffGet is how you find out.
    """
    ctx = FakeCtx(setpoint=1e-9, currents=[1e-12] * 20, rt_stuck_on=True)
    res = TryEngageController().execute(ctx, {"settle_s": 0.05, "poll_hz": 10})
    assert res.data["needs_auto_approach"] is False, "Z 反馈其实还闭着，却仍建议粗进针"
    assert res.success is False
    assert res.data["z_controller_on"] is True   # the truth


def test_coarse_approach_refused_when_the_state_cannot_be_read():
    """Fail closed. The FAILURE of the check must never be what authorises the
    dangerous action — the same rule that already guards the broken-current path."""
    ctx = FakeCtx(setpoint=1e-9, currents=[1e-12] * 20, fail_onoff_get=True)
    res = TryEngageController().execute(ctx, {"settle_s": 0.05, "poll_hz": 10})
    assert res.data["needs_auto_approach"] is False
    assert res.success is False
    assert res.data["z_controller_on"] is None   # unknown — NOT False


def test_setpoint_read_error_fails_without_touching_feedback():
    ctx = FakeCtx(fail_setpoint=True)
    res = TryEngageController().execute(ctx, {"settle_s": 0.05})
    assert res.success is False
    assert "setpoint" in res.error
    assert _onoff_calls(ctx) == []  # never toggled feedback


def test_metadata_is_confirm_write():
    md = TryEngageController().metadata()
    assert md.name == "TryEngageController"
    assert md.category.value == "write" or str(md.category).endswith("WRITE")
    assert str(md.safety_level).endswith("CONFIRM") or md.safety_level.value == "confirm"


def test_registry_discovers_it():
    from mast.core.registry import SkillRegistry
    r = SkillRegistry()
    r.discover("mast.skills.builtins")
    assert r.has("TryEngageController")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
