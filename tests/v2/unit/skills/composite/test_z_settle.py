"""Synthetic Z-loop trajectories verify convergence, initial conditions, missing readback and the meaning of a settled position.

Composite consequences are tested separately in relocation and sample-change flows."""
from __future__ import annotations

import random

import pytest

from mast.core import instrument_profile as ip
from mast.skills.composite._z_settle import (
    settle_and_read_z,
    settle_tolerance_m,
)


class _Rec:
    def __init__(self, error: str = "", return_value=None):
        self.error = error
        self.return_value = return_value


class _Loop:
    """A Z loop you can shape: how it ramps, whether it ever arrives, and
    whether there is a junction to hold once it does.

    ``z_at(n)`` is the piezo position at the n-th poll — the whole point being
    that it is a FUNCTION OF THE POLL, not a constant. A stand-in that answers
    with a constant is a rig on which "sleep, then read" is always correct, and
    that is exactly the rig the broken check was tested against.
    """

    def __init__(self, z_at, current_a=1e-10, z_error="", on=1):
        self.z_at = z_at
        self.current_a = current_a
        self.z_error = z_error
        self.on = on
        self.n = 0
        self.calls: list[str] = []

    def safe_call(self, verb, *args, **kw):
        self.calls.append(verb)
        if verb == "ZCtrl_ZPosGet":
            self.n += 1
            if self.z_error:
                return _Rec(self.z_error, None)
            return _Rec("", ("h", "b", [self.z_at(self.n)]))
        if verb == "Current_Get":
            return _Rec("", ("h", "b", [self.current_a]))
        if verb == "ZCtrl_SetpntGet":
            return _Rec("", ("h", "b", [1e-10]))
        if verb == "ZCtrl_OnOffGet":
            return _Rec("", ("h", "b", [self.on]))
        return _Rec("", None)

    def check_abort(self):
        return False


def _settle(loop, **kw):
    kw.setdefault("timeout_s", 3.0)
    kw.setdefault("poll_interval_s", 0.0)
    return settle_and_read_z(loop, **kw)


@pytest.fixture(autouse=True)
def _profile():
    before = ip.get_profile()
    ip.set_persist_sink(None)
    ip.set_profile({"z_recede_min_nm": 1.0})     # → 0.5 nm convergence band
    yield
    ip.set_profile(before)


# ── the band is derived, not invented ───────────────────────────────────────

def test_the_convergence_band_tracks_the_decision_threshold():
    """One number, one place.

    The band has to be tighter than the threshold it feeds, or a reading can be
    called settled while still drifting enough to flip the verdict. Deriving it
    means an operator who loosens the threshold for a noisy rig gets a matching
    band; a second config key would quietly stay at the old value."""
    ip.set_profile({"z_recede_min_nm": 1.0})
    assert settle_tolerance_m() == pytest.approx(0.5e-9)
    ip.set_profile({"z_recede_min_nm": 8.0})
    assert settle_tolerance_m() == pytest.approx(4.0e-9)


# ── what must NOT be called converged ───────────────────────────────────────

def test_a_ramp_is_not_converged_no_matter_how_long_it_has_been_running():
    """An unbounded synthetic ramp must time out as moving, never as a settled reading."""
    res = _settle(_Loop(lambda n: -750e-9 + n * 110e-9), timeout_s=0.4,
                  poll_interval_s=0.01)
    assert res.settled is False
    assert res.usable is False
    assert res.state == "moving"
    assert "超时" in res.why()


def test_a_loop_that_has_not_started_moving_is_not_converged():
    """"Not moving" is also true before the loop begins.

    Without this clause the very first window — taken while the piezo is still
    sitting where the withdraw left it — reads as settled, and the fix
    reintroduces the bug it was written to remove."""
    parked_then_ramps = _Loop(
        lambda n: -750e-9 if n <= 8 else -750e-9 + (n - 8) * 110e-9,
        current_a=1e-14)          # no junction: stillness alone must not count
    res = _settle(parked_then_ramps, timeout_s=0.3, poll_interval_s=0.01)
    assert res.settled is False, (
        "a piezo that has not moved yet was mistaken for one that has arrived"
    )


def test_a_stationary_reading_with_a_junction_is_accepted_immediately():
    """The counterpart: if the loop is holding current, it HAS done its job.

    Requiring observed travel unconditionally would hang on the legitimate case
    where feedback was already engaged and had nothing to do."""
    res = _settle(_Loop(lambda n: -300e-9, current_a=1e-10))
    assert res.settled is True and res.state == "tracking"


# ── noise must not read as motion ───────────────────────────────────────────

@pytest.mark.parametrize("seed", range(10))
def test_a_noisy_but_stationary_z_still_converges(seed):
    """Net drift across the window, not peak-to-peak.

    Peak-to-peak noise here (0.8 nm) is LARGER than the 0.5 nm band, so a
    quietness test would never converge on the instrument and every relocation would
    time out. Random noise does not accumulate into a net displacement; a ramp
    does. So the criterion detects MOTION rather than silence."""
    rng = random.Random(seed)

    def z(n):
        base = -300e-9 if n > 6 else -300e-9 + (7 - n) * 150e-9
        return base + rng.uniform(-0.4e-9, 0.4e-9)

    res = _settle(_Loop(z, current_a=1e-10))
    assert res.settled is True and res.state == "tracking"


# ── what a settled Z MEANS ──────────────────────────────────────────────────

def test_settled_with_no_current_is_the_rail_not_a_gap():
    """The piezo ran out of range with nothing to find.

    This is the NORMAL outcome of a clearance ladder past the first rung or two,
    so it is a usable answer — but Z is a bound, and the caller has to be able to
    tell that from a distance."""
    res = _settle(_Loop(lambda n: min(-750e-9 + n * 300e-9, 750e-9),
                        current_a=1e-14))
    assert res.settled is True
    assert res.state == "out_of_range"
    assert res.at_rail is True
    assert res.usable is True, "a rail reading is comparable, just not a distance"


def test_settled_while_tunnelling_is_a_real_gap_reading():
    res = _settle(_Loop(lambda n: min(-750e-9 + n * 300e-9, -100e-9),
                        current_a=1e-10))
    assert res.state == "tracking" and res.at_rail is False and res.usable is True


# ── failures have to be distinguishable from each other ─────────────────────

def test_an_unreadable_z_gives_up_early_and_says_it_is_not_a_timing_problem():
    """"Cannot read Z" and "this rig is slow" need different repairs.

    Waiting out a 20 s budget cannot turn a dead readback into a reading, and a
    message that suggests raising the budget would send the operator the wrong
    way."""
    res = _settle(_Loop(lambda n: 0.0, z_error="timeout"), timeout_s=20.0,
                  poll_interval_s=0.001)
    assert res.state == "unreadable" and res.usable is False
    assert res.samples < 10, "it waited out a budget it could not possibly use"
    assert "加大预算没有用" in res.why()


def test_the_timeout_message_carries_the_numbers_that_diagnose_it():
    """Budget, elapsed, drift rate, and whether the loop was even closed.

    Mirrors ZControllerOnOff: without them nobody can tell "we judged too early"
    from "the feedback never engaged"."""
    res = _settle(_Loop(lambda n: n * 110e-9, current_a=1e-14, on=0),
                  timeout_s=0.2, poll_interval_s=0.01)
    why = res.why()
    assert res.loop_confirmed_on is False
    assert "预算" in why and "nm/窗口" in why
    assert "实时控制器在开始时回报 Z 反馈是断开的" in why, (
        "a loop that was never closed must not be reported as a slow rig"
    )


def test_an_abort_during_the_settle_is_not_a_reading():
    class _Aborting(_Loop):
        def check_abort(self):
            return self.n >= 3

    res = _settle(_Aborting(lambda n: -750e-9 + n * 110e-9))
    assert res.state == "aborted"
    assert res.usable is False, "an aborted settle must not hand back a Z to judge"


# ── the initial condition ───────────────────────────────────────────────────

def test_every_settle_starts_from_the_withdraw_position():
    """Every settle must begin with withdrawal so separately converged readings share an initial condition."""
    loop = _Loop(lambda n: -300e-9)
    _settle(loop)
    assert loop.calls[0] == "ZCtrl_Withdraw"
    assert loop.calls.index("ZCtrl_Withdraw") < loop.calls.index("ZCtrl_OnOffSet")


def test_the_poll_reads_are_not_logged_one_by_one():
    """A 5 s settle at 10 Hz is ~100 round-trips per rung.

    Logging each one buries the result — and the checkpoint — under its own
    diagnostics. The summary rides on the ZSettle instead."""
    log: list = []
    loop = _Loop(lambda n: min(-750e-9 + n * 60e-9, -300e-9), current_a=1e-10)
    res = _settle(loop, log=log)
    assert res.samples > 5
    assert len(log) <= 6, f"logged {len(log)} records for one settle"
