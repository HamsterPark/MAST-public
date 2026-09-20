"""``MoveAtomTo``: the order of operations that decides whether an atom is moved or lost.

Lateral manipulation is not one command. The tip has to be brought to the atom at an imaging
resistance, taken *down* to a manipulation resistance only once it is there, dragged, and then
taken back up **before** it moves away again. Get that order wrong and the tip either never
picks the atom up or drags it back off the target on the way out — and either way the frame
afterwards looks like a normal frame.

So what is pinned here is the sequence, not an outcome: nothing in this file needs a
microscope.
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

from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite.graph_executor import CompositeProgress  # noqa: E402
from mast.skills.composite.move_atom_to import MoveAtomTo  # noqa: E402

ATOM = {"atom_x_m": 0.0, "atom_y_m": 0.0, "target_x_m": 4e-9, "target_y_m": 0.0}
#: what the instrument answers when asked. The imaging range is 10 nA, which is below the
#: manipulation setpoint — that is the normal case, and the reason the range has to be widened.
RIG = {"GetBias": {"bias_v": 0.05},
       "GetSetpoint": {"setpoint_a": 50e-12},
       "GetTipSpeed": {"speed_m_s": 293e-9, "custom_speed": True},
       "GetCurrentGains": {"gains": ["1E6", "1E7", "1E8", "1E9", "1E10", "1E11"],
                           "gain_index": 3, "full_scale_a": 10e-9},
       "ScanAt": {"scan_path": "verify.sxm"}}


@dataclass
class FakeCtx:
    """Every sub-skill succeeds, with per-skill canned data."""

    data_for: dict = field(default_factory=dict)
    fail: set = field(default_factory=set)
    run_log: list = field(default_factory=list)
    emitted: list = field(default_factory=list)
    flushes: int = 0

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        ok = skill_name not in self.fail
        return SkillResult(skill_name=skill_name, success=ok,
                           data=dict(self.data_for.get(skill_name, {})),
                           error=None if ok else "canned failure")

    def emit_progress(self, progress: CompositeProgress) -> None:
        self.emitted.append(CompositeProgress.from_dict(progress.to_dict()))

    def get_progress(self, name: str) -> dict | None:
        return None

    def checkpoint_flush(self) -> None:
        self.flushes += 1


def _ids(skill: MoveAtomTo, params: dict) -> list[str]:
    """The step ids of one pass through the plan, without running anything."""
    from mast.skills.composite.graph_executor import GraphExecutor

    ex = GraphExecutor(composite_name="MoveAtomTo", context=FakeCtx(data_for=dict(RIG)))
    out = []
    for step in skill.plan_dynamic({**params}, ex):
        out.append(step.step_id)
        if len(out) > 400:
            break
    return out


def _order(log: list, *names: str) -> list[int]:
    """Where each named skill first appears in the run log."""
    return [next((i for i, (n, _) in enumerate(log) if n == name), -1) for name in names]


def test_the_junction_is_lowered_only_after_the_tip_is_at_the_atom():
    """Dropping to 57 nA before arriving drags whatever is on the way."""
    ctx = FakeCtx(data_for=dict(RIG))
    MoveAtomTo().run_composite(ctx, dict(ATOM, verify=False))
    move, gain, bias, setpoint = _order(ctx.run_log, "MoveToXY", "SetCurrentGain",
                                        "SetBias", "SetSetpoint")
    assert move >= 0 and setpoint > move, ctx.run_log
    assert bias > move and gain > move


def test_the_preamp_range_is_read_before_it_can_be_raised():
    """It cannot be widened without knowing what it is, and reading it off the wrong readback
    step is silent: the only symptom is that the range is never changed."""
    ctx = FakeCtx(data_for=dict(RIG))
    MoveAtomTo().run_composite(ctx, dict(ATOM, verify=False))
    names = [n for n, _ in ctx.run_log]
    assert "GetCurrentGains" in names
    assert names.index("GetCurrentGains") < names.index("SetCurrentGain")


def test_the_preamp_range_is_raised_before_the_setpoint():
    """57 nA demanded on a 10 nA range is a crash, not a manipulation."""
    ctx = FakeCtx(data_for=dict(RIG))
    MoveAtomTo().run_composite(ctx, dict(ATOM, verify=False))
    gain, setpoint = _order(ctx.run_log, "SetCurrentGain", "SetSetpoint")
    assert 0 <= gain < setpoint, ctx.run_log


def test_the_current_is_taken_back_up_before_the_tip_leaves():
    """The release order is the whole trick: setpoint first, then bias. Moving away at the
    manipulation resistance takes the atom along."""
    ctx = FakeCtx(data_for=dict(RIG))
    MoveAtomTo().run_composite(ctx, dict(ATOM, verify=False))
    names = [n for n, _ in ctx.run_log]
    last_drag = max(i for i, n in enumerate(names) if n == "MoveToXY")
    restores = [i for i, n in enumerate(names) if n in ("SetSetpoint", "SetBias")]
    after = [i for i in restores if i > last_drag]
    assert len(after) >= 2, ctx.run_log
    assert names[after[0]] == "SetSetpoint" and names[after[1]] == "SetBias"


def test_the_drag_is_broken_into_short_hops():
    """FolMe moves the tip in a straight line; the atom follows one lattice site at a time,
    so the waypoints have to be finer than a site or the physics is skipped over."""
    ids = _ids(MoveAtomTo(), dict(ATOM, verify=False))
    drags = [i for i in ids if ":drag_" in i]
    assert len(drags) >= 10, ids


def test_verify_false_means_the_answer_is_unknown_not_success():
    ctx = FakeCtx(data_for=dict(RIG))
    res = MoveAtomTo().run_composite(ctx, dict(ATOM, verify=False))
    assert res.data["moved"] is None
    assert not any(n == "VerifyAdatomAt" for n, _ in ctx.run_log)


def test_a_confirmed_atom_at_the_target_reads_as_moved():
    ctx = FakeCtx(data_for={**RIG, "VerifyAdatomAt": {"verdict": "at_target",
                                               "found_x_m": 4e-9, "found_y_m": 0.0,
                                               "residual_m": 5e-11}})
    res = MoveAtomTo().run_composite(ctx, dict(ATOM, verify=True))
    assert res.data["moved"] is True
    assert res.data["verify_verdict"] == "at_target"
    assert res.data["residual_nm"] == pytest.approx(0.05, rel=1e-6)


def test_an_atom_left_behind_is_retried_at_a_lower_resistance():
    """The pull threshold is not knowable in advance, so one retry lowers the resistance. It
    is a second attempt, not a second composite: the instrument is still restored once."""
    ctx = FakeCtx(data_for={**RIG, "VerifyAdatomAt": {"verdict": "displaced",
                                               "found_x_m": 1e-9, "found_y_m": 0.0}})
    res = MoveAtomTo().run_composite(ctx, dict(ATOM, verify=True, max_attempts=2))
    assert res.data["attempts"] == 2, res.data["steps"]
    assert any(s.startswith("a2:") for s in res.data["steps"])
    setpoints = [p.get("setpoint_a") for n, p in ctx.run_log
                 if n == "SetSetpoint" and p.get("setpoint_a")]
    assert max(setpoints) > min(s for s in setpoints if s) , setpoints


def test_an_atom_that_is_not_there_is_not_retried():
    """Retrying at a lower resistance when the atom was never found just rakes the surface."""
    ctx = FakeCtx(data_for={**RIG, "VerifyAdatomAt": {"verdict": "not_found"}})
    res = MoveAtomTo().run_composite(ctx, dict(ATOM, verify=True, max_attempts=2))
    assert res.data["attempts"] == 1
    assert not any(s.startswith("a2:") for s in res.data["steps"])
    assert res.data["moved"] is False


def test_the_instrument_is_restored_even_when_the_move_fails():
    """Leaving the junction at 57 nA and 10 mV after an error is how the next command crashes
    the tip. A failed move must still hand the instrument back."""
    ctx = FakeCtx(data_for=dict(RIG), fail={"VerifyAdatomAt"})
    res = MoveAtomTo().run_composite(ctx, dict(ATOM, verify=True))
    assert res.data["instrument_restored"] is True
    assert res.data["restored"]["setpoint"] and res.data["restored"]["bias"]


def test_failing_to_restore_the_junction_fails_the_skill_whatever_else_happened():
    ctx = FakeCtx(fail={"SetSetpoint"},
                  data_for={**RIG, "VerifyAdatomAt": {"verdict": "at_target", "found_x_m": 4e-9,
                                               "found_y_m": 0.0, "residual_m": 1e-11}})
    res = MoveAtomTo().run_composite(ctx, dict(ATOM, verify=True))
    assert res.success is False
    assert res.data["instrument_restored"] is False
    assert "setpoint" in (res.error or "").lower() or "设定点" in (res.error or "")


def test_the_manipulation_bias_keeps_the_sign_of_the_imaging_bias():
    """Flipping the sign mid-manipulation is a different experiment: the atom sees the field
    the other way round."""
    ctx = FakeCtx(data_for={**RIG, "GetBias": {"bias_v": -0.05}})
    MoveAtomTo().run_composite(ctx, dict(ATOM, verify=False, manip_bias_v=0.01))
    biases = [p.get("bias_v") for n, p in ctx.run_log if n == "SetBias" and "bias_v" in p]
    assert biases and biases[0] < 0, biases


def test_the_coordinate_epoch_is_passed_through_to_every_move():
    """Coordinates that came out of storage are only valid against the epoch they were read
    in; dropping the epoch on the way to FolMe is how a stale position gets used."""
    ids_params = FakeCtx(data_for=dict(RIG))
    MoveAtomTo().run_composite(ids_params, dict(ATOM, verify=False, coord_epoch=7))
    moves = [p for n, p in ids_params.run_log if n == "MoveToXY"]
    assert moves and all(p.get("coord_epoch") == 7 for p in moves)
