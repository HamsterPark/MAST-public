"""Confirm-don't-assume: the Z-loop readback, and the fail-closed rule around it.

The bug this guards against is not hypothetical. Before 2026-07-13:

    rec_off = context.safe_call("ZCtrl_OnOffSet", 0)     # error never checked
    calls.append(rec_off)
    return SkillResult(..., data={"z_controller_on": False,
                                  "needs_auto_approach": True})

``needs_auto_approach: True`` tells the agent to run the OPEN-LOOP COARSE STEPPER —
the one motion in this system with no current-feedback stop, and the one that
reliably destroys a tip. It was authorised by ``z_controller_on: False``, which was
not a reading of anything. It was the skill repeating its own request back to itself.

Nanonis' own manual is explicit that the two are different questions:

    ZCtrl.OnOffGet — "returns the status FROM THE REAL-TIME CONTROLLER (i.e. not
    from the Z-Controller module). This function is useful to make sure that the
    Z-controller is REALLY off before starting an experiment. Due to the
    communication delay …"
"""

from __future__ import annotations

import pytest

from mast.skills.builtins.zcontrol import TryEngageController, ZControllerOnOff
from mast.skills.verify import verify_z_controller, z_off_or_reason


class _Rec:
    def __init__(self, value=None, error=None):
        self.return_value = value
        self.error = error


class FakeCtx:
    """Answers safe_call from a scripted table. Records the call order."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls: list[tuple] = []

    def safe_call(self, verb, *args, **kw):
        self.calls.append((verb, args))
        a = self.answers.get(verb, _Rec(None, None))
        return a(self) if callable(a) else a

    def check_abort(self):
        return False

    def verbs(self):
        return [v for v, _ in self.calls]


# Nanonis nests its returns: (header, error, body).
def _ok(*body):
    return _Rec([0, 0, list(body)], None)


def _err(msg="TCP timeout"):
    return _Rec(None, msg)


# ─────────────────────────────────────────────────────────────────────────────
# verify_z_controller — the primitive
# ─────────────────────────────────────────────────────────────────────────────

def test_reads_the_real_time_controller_not_the_module():
    ctx = FakeCtx({"ZCtrl_OnOffGet": _ok(0)})
    v = verify_z_controller(ctx, expect=False)
    assert "ZCtrl_OnOffGet" in ctx.verbs(), "必须问实时控制器，不是问模块(StatusGet)"
    # StatusGet is only read as DIAGNOSTICS on a mismatch — a matching read must
    # not cost an extra round-trip, and must never be the thing we conclude from.
    assert "ZCtrl_StatusGet" not in ctx.verbs()
    assert v["on"] is False
    assert v["verified"] is True
    assert v["matches"] is True
    assert v["error"] is None
    # 2026-07-27: settle fields added; a first read that already agrees must not
    # poll at all (no switch-off-delay lookup, no wait).
    assert v["switch_off_delay_s"] is None
    assert "ZCtrl_SwitchOffDelayGet" not in ctx.verbs()


def test_mismatch_is_reported():
    ctx = FakeCtx({"ZCtrl_OnOffGet": _ok(1)})       # hardware says ON
    v = verify_z_controller(ctx, expect=False)      # we asked for OFF
    assert v["on"] is True and v["matches"] is False


def test_unreadable_gives_None_not_False():
    """`on=None` (unknown) must never collapse into `on=False` (safe). That collapse
    is the whole bug: the FAILURE of the check becomes the AUTHORISATION."""
    for answer in (_err(), _Rec([0, 0, []], None), _Rec(None, None)):
        ctx = FakeCtx({"ZCtrl_OnOffGet": answer})
        v = verify_z_controller(ctx, expect=False)
        assert v["on"] is None, f"读不到时返回了 {v['on']!r}，必须是 None"
        assert v["verified"] is False
        assert v["matches"] is None


# ─────────────────────────────────────────────────────────────────────────────
# z_off_or_reason — the fail-closed form
# ─────────────────────────────────────────────────────────────────────────────

def test_z_off_only_when_actually_confirmed_off():
    off, why, _ = z_off_or_reason(FakeCtx({"ZCtrl_OnOffGet": _ok(0)}))
    assert off is True and why is None


@pytest.mark.parametrize("answer,label", [
    (_ok(1), "环还闭着"),
    (_err(), "读取失败"),
    (_Rec([0, 0, []], None), "返回无法解析"),
])
def test_anything_other_than_a_confirmed_off_is_not_off(answer, label):
    off, why, _ = z_off_or_reason(FakeCtx({"ZCtrl_OnOffGet": answer}))
    assert off is False, f"{label} 时却报告 Z 反馈已断开"
    assert why, "必须给出人能读懂的原因"


# ─────────────────────────────────────────────────────────────────────────────
# ZControllerOnOff — stop echoing the request back as the state
# ─────────────────────────────────────────────────────────────────────────────

def test_onoff_reports_the_verified_state_not_the_request():
    ctx = FakeCtx({"ZCtrl_OnOffSet": _ok(), "ZCtrl_OnOffGet": _ok(1)})
    r = ZControllerOnOff().execute(ctx, {"enable": True})
    assert r.success and r.data["z_controller_on"] is True
    assert r.data["verified"] is True
    assert "ZCtrl_OnOffGet" in ctx.verbs(), "写完没读回——那 z_controller_on 就只是复述请求"


def test_onoff_fails_loudly_when_the_write_did_not_take():
    """The write returns OK over TCP and the hardware did not change. This is the case
    the old code could not even see."""
    ctx = FakeCtx({"ZCtrl_OnOffSet": _ok(), "ZCtrl_OnOffGet": _ok(1)})   # asked OFF, got ON
    r = ZControllerOnOff().execute(ctx, {"enable": False})
    assert r.success is False
    assert r.data["z_controller_on"] is True     # the TRUTH, not the request
    assert r.data["requested"] is False


def test_onoff_says_unknown_rather_than_assuming_it_worked():
    ctx = FakeCtx({"ZCtrl_OnOffSet": _ok(), "ZCtrl_OnOffGet": _err()})
    r = ZControllerOnOff().execute(ctx, {"enable": False})
    assert r.data["z_controller_on"] is None, "读不回来时不许把 unknown 说成 as-requested"
    assert r.data["verified"] is False
    assert "无法读回" in (r.summary or "")   # SkillResult has `summary`, not `message`


# ─────────────────────────────────────────────────────────────────────────────
# TryEngageController — the one that authorises a coarse approach
# ─────────────────────────────────────────────────────────────────────────────

def _engage_ctx(onoff_get, current=0.0, setpoint=1e-10):
    """Not tunnelling (current far below setpoint) → the skill will want to switch the
    loop back off and recommend a coarse approach."""
    return FakeCtx({
        "ZCtrl_SetpntGet": _ok(setpoint),
        "ZCtrl_OnOffSet": _ok(),
        "ZCtrl_OnOffGet": onoff_get,
        "Current_Get": _ok(current),
    })


def test_coarse_approach_is_authorised_only_by_a_confirmed_off():
    ctx = _engage_ctx(_ok(0))          # RT controller confirms OFF
    r = TryEngageController().execute(ctx, {"settle_s": 0.01, "poll_hz": 100.0})
    assert r.success and r.data["needs_auto_approach"] is True
    assert r.data["z_controller_on"] is False
    assert r.data["z_controller_verified"] is True


@pytest.mark.parametrize("answer,label", [
    (_ok(1), "Z 反馈其实还闭着"),
    (_err(), "读不到 Z 反馈状态"),
])
def test_coarse_approach_is_REFUSED_when_off_cannot_be_confirmed(answer, label):
    """THE test. Recommending a coarse approach with the feedback possibly still closed
    is a tip crash. The failure of the check must never be what authorises the
    dangerous action — the same fail-closed rule that already guards the broken
    current-measurement path."""
    ctx = _engage_ctx(answer)
    r = TryEngageController().execute(ctx, {"settle_s": 0.01, "poll_hz": 100.0})
    assert r.data["needs_auto_approach"] is False, (
        f"{label}，却仍然建议跑粗进针（AutoApproach 是开环马达，没有电流反馈停止）"
    )
    assert r.success is False
    assert r.data["z_controller_on"] is None or r.data["z_controller_on"] is True


def test_broken_current_chain_still_fails_closed_and_still_verifies():
    """The pre-existing fail-closed path (no valid current reads) must keep working —
    and must now also report a VERIFIED z_controller_on rather than a claimed one."""
    ctx = FakeCtx({
        "ZCtrl_SetpntGet": _ok(1e-10),
        "ZCtrl_OnOffSet": _ok(),
        "ZCtrl_OnOffGet": _ok(0),
        "Current_Get": _err("preamp offline"),
    })
    r = TryEngageController().execute(ctx, {"settle_s": 0.01, "poll_hz": 100.0})
    assert r.success is False
    assert r.data["needs_auto_approach"] is False
    assert r.data["z_controller_on"] is False       # read, not claimed
    assert r.data["z_controller_verified"] is True
