"""Three layers read ``SkillMetadata.preconditions``. They must read the same list.

``mast.core.preconditions`` has declared itself the single source of the
precondition vocabulary since . It was not one. Three hand-maintained
chains existed and all three had drifted apart:

  * ``core.safety.SafetyGuard`` — the only one that knew ``withdrawn``;
  * ``agents._shared.safety_mw.SafetyGate`` — a copy that never grew it, so the
    "is the tip actually clear" check silently passed for everything the AGENT
    ran, i.e. on the one path with nobody watching;
  * ``core.preconditions`` — what ``BaseSkill.check_preconditions`` reads.

The third asymmetry is the sharp one, and it is worth stating plainly because it
is not what you would guess: a rule added ONLY to ``core.safety`` does not merely
go unenforced at the wrap_skill layer — ``precondition_recognized`` returns False
there, so the skill fails outright with "Cannot verify precondition". A new
precondition is therefore not a partial improvement, it is a broken skill.

That is exactly what happened with ``vacuum_ok_for_coarse``: every
``RelocateCoarseXY`` call would have failed on the real agent path while passing
every direct-``run_composite`` unit test.
"""
from __future__ import annotations

import pytest

from mast.config import SafetyLimits
from mast.core.types import HardwareState, SkillMetadata

ALL_PRECONDITIONS = [
    "z_controller_on", "z_controller_off",
    "scan_running", "scan_not_running", "scan_stopped",
    "bias_nonzero", "tip_withdrawn", "vacuum_ok_for_coarse",
]

#: State that violates each precondition, so every rule has something to fire on.
VIOLATING_STATE = HardwareState(
    z_controller_on=True, scan_running=True, bias_v=0.0, withdrawn=False)


def _core(names, state):
    from mast.core.safety import check_state_preconditions
    return check_state_preconditions(SkillMetadata(name="X", preconditions=names), state)


def _gate(names, state):
    from mast.agents._shared.safety_mw import SafetyGate
    return SafetyGate(SafetyLimits()).check_state_preconditions(
        SkillMetadata(name="X", preconditions=names), state)


def _shared(names, state):
    from mast.core.preconditions import check_state_preconditions
    return check_state_preconditions(list(names), state)


@pytest.mark.parametrize("name", ALL_PRECONDITIONS)
def test_all_three_layers_agree_on_every_precondition(name):
    a = _core([name], VIOLATING_STATE)
    b = _gate([name], VIOLATING_STATE)
    c = _shared([name], VIOLATING_STATE)
    assert a == b == c, (
        f"precondition {name!r} is interpreted differently by the three layers:\n"
        f"  core.safety : {a}\n  SafetyGate  : {b}\n  preconditions: {c}"
    )


@pytest.mark.parametrize("name", ALL_PRECONDITIONS)
def test_every_precondition_is_recognised_by_baseskill(name):
    """Otherwise the skill declaring it fails with "Cannot verify precondition".

    This is the asymmetry that makes a partially-registered precondition WORSE
    than an unenforced one: it does not degrade, it refuses."""
    from mast.core.preconditions import precondition_recognized

    assert precondition_recognized(name), (
        f"{name!r} is unknown to core.preconditions, so any skill declaring it "
        f"fails at the wrap_skill layer rather than being checked"
    )


def test_withdrawn_is_enforced_on_the_agent_path_too():
    """The drift that was live: the agent-path copy never grew this branch."""
    v = _gate(["tip_withdrawn"], HardwareState(withdrawn=False))
    assert v and "not withdrawn" in v[0]


def test_withdrawn_still_fails_open_on_unknown_state():
    """Unchanged on purpose. Most preconditions here fail OPEN when the state is
    unknown, which is right when the worst case is a wasted scan."""
    assert _core(["tip_withdrawn"], HardwareState(withdrawn=None)) == []
    assert _gate(["tip_withdrawn"], HardwareState(withdrawn=None)) == []


def test_vacuum_is_the_one_that_fails_CLOSED_on_unknown():
    """And it is the opposite, deliberately.

    An unknown pressure is precisely the condition this exists to refuse: a gauge
    that cannot see is a gauge in the discharge band or at atmosphere, with
    nothing to tell them apart, and driving a coarse piezo there arcs across the
    stack. There is no retry for that."""
    from mast.core import vacuum_interlock as vac

    vac.set_pressure_source(None)
    vac.revoke_attestation()
    try:
        for layer in (_core, _gate, _shared):
            v = layer(["vacuum_ok_for_coarse"], HardwareState())
            assert v, f"{layer.__name__} let an unknown pressure through"
    finally:
        vac.set_pressure_source(None)


def test_relocate_passes_its_own_preconditions_when_the_rig_is_healthy():
    """The end-to-end shape of the bug: it passed run_composite and failed here."""
    import datetime as dt

    from mast.core import vacuum_interlock as vac
    from mast.skills.composite.relocate_coarse_xy import RelocateCoarseXY

    vac.set_pressure_source(lambda: vac.PressureSample(
        value=1e-8, unit="Pa", status="ok",
        timestamp=dt.datetime.now().isoformat(),
        sensor_name="vacuum", sensor_class="DL7VacuumSensor"))
    try:
        unmet = RelocateCoarseXY().check_preconditions(
            HardwareState(scan_running=False))
        assert unmet == [], unmet
    finally:
        vac.set_pressure_source(None)


def test_relocate_is_refused_by_its_own_preconditions_with_no_gauge():
    from mast.core import vacuum_interlock as vac
    from mast.skills.composite.relocate_coarse_xy import RelocateCoarseXY

    vac.set_pressure_source(None)
    vac.revoke_attestation()
    unmet = RelocateCoarseXY().check_preconditions(HardwareState(scan_running=False))
    assert any("vacuum_ok_for_coarse" in u for u in unmet)
    assert not any("Cannot verify" in u for u in unmet), (
        "it must be REFUSED with a reason, not reported as unrecognised"
    )
