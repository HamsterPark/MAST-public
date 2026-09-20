"""Recorder branches that put damage and coordinate boundaries on the scan map.

Three classes of event were invisible to the map: a lateral coarse move (which
invalidates every coordinate recorded before it), a crash (whose location is
exactly what must never be scanned again), and an approach dimple. Two of them
cannot go through the generic skill→kind table because their coordinate semantics
differ from its precedence, so they are explicit branches — and an explicit branch
that sits after the table's early-return would never run, which is what these
tests pin down.

The classifier and the payload helpers are pure, so they are tested directly
rather than through a live CoreRuntime.
"""
from __future__ import annotations

import pytest

from mast.core.runtime import (
    _refine_marker_from_saved_file,
    crash_point,
    lateral_coarse_move_info,
)
from mast.io.exp_map import classify_skill, marker_from_skill


# ── lateral coarse move ────────────────────────────────────────────────────

def _motor(direction, success=True, steps=50):
    return {"skill": "MotorMove", "success": success,
            "params": {"direction": direction, "steps": steps},
            "data": {"direction": direction, "steps": steps}}


@pytest.mark.parametrize("direction", ["x+", "x-", "y+", "y-"])
def test_lateral_coarse_move_is_detected(direction):
    info = lateral_coarse_move_info(_motor(direction))
    assert info == {"direction": direction, "steps": 50, "partial": False}


@pytest.mark.parametrize("direction", ["z-approach", "z-retract"])
def test_z_coarse_move_is_not_a_coordinate_boundary(direction):
    """Approaching or retracting does not change where on the surface the tip
    is, so it must not age a single marker."""
    assert lateral_coarse_move_info(_motor(direction)) is None


def test_failed_coarse_move_is_not_a_boundary():
    """A failed MotorMove did not move the stage, so coordinates still hold.

    Note what this payload contains: a full ``data`` with ``steps: 50``. That is
    deliberately MORE than the real skill produces (``MotorMove`` writes no data
    on any of its failure branches), and it still must not register — because
    the failure path admits exactly ONE key, ``lateral_steps_taken``, and
    ``MotorMove`` cannot and does not write it."""
    assert lateral_coarse_move_info(_motor("x+", success=False)) is None


def test_coarse_move_reads_direction_from_the_result_not_only_params():
    payload = {"skill": "MotorMove", "success": True, "params": {},
               "data": {"direction": "y-", "steps": 10}}
    assert lateral_coarse_move_info(payload)["direction"] == "y-"


def test_coarse_move_tolerates_a_missing_step_count():
    payload = {"skill": "MotorMove", "success": True,
               "params": {"direction": "x+"}, "data": {}}
    assert lateral_coarse_move_info(payload) == {"direction": "x+", "steps": None,
                                                 "partial": False}


# ── the move happened; the composite failed afterwards ─────────────────────
#
# A later re-approach failure must not erase lateral movement already completed.

def _relocate(*, success, direction="x+", steps=240, taken=None, dry_run=False,
              requested=240):
    data = {"direction": direction, "steps": steps, "dry_run": dry_run}
    if taken is not None:
        data["lateral_steps_taken"] = taken
    return {"skill": "RelocateCoarseXY", "success": success, "data": data,
            "params": {"axis": direction[0], "direction": direction[1],
                       "steps": requested, "dry_run": dry_run}}


def test_a_move_that_happened_is_recorded_even_when_the_composite_failed():
    info = lateral_coarse_move_info(_relocate(success=False, taken=240))
    assert info == {"direction": "x+", "steps": 240, "partial": True}


def test_a_partial_move_is_recorded_by_what_it_took_not_what_it_asked_for():
    """Watchdog abort halfway: 90 of 240 steps physically happened."""
    info = lateral_coarse_move_info(
        _relocate(success=False, steps=90, taken=90, requested=240))
    assert info["steps"] == 90, "odometer fed the request instead of the outcome"


def test_a_refusal_before_moving_still_reaches_nothing():
    """§2.28: motor commanded, stage stuck, composite refuses. Zero steps taken.

    The params still carry the full 240 — which is exactly why the failure path
    must never look at them."""
    assert lateral_coarse_move_info(
        _relocate(success=False, steps=0, taken=0, requested=240)) is None


def test_a_failed_result_without_the_explicit_key_is_ignored():
    """``steps`` alone must NOT open the failure path.

    The simpler design — "on failure, read data['steps']" — looks equivalent and
    is not: ``steps`` means different things at different points in a composite's
    life, and no reader of a dict can tell which one they are holding. The
    admission ticket has to be a key whose NAME is the promise."""
    assert lateral_coarse_move_info(
        _relocate(success=False, steps=240, taken=None)) is None


def test_dry_run_is_never_a_relocation():
    """A dry run reports intended steps but must never update the position ledger."""
    assert lateral_coarse_move_info(
        _relocate(success=True, steps=120, taken=0, dry_run=True)) is None
    assert lateral_coarse_move_info(
        _relocate(success=False, steps=120, taken=0, dry_run=True)) is None


def test_other_skills_are_never_coarse_moves():
    assert lateral_coarse_move_info(
        {"skill": "MotorGetPos", "success": True, "data": {"direction": "x+"}}) is None


def test_motormove_is_not_classified_by_the_generic_table():
    """It has to be an explicit branch: the generic precedence would place the
    boundary using scan-frame/tip rules that do not apply to it."""
    assert classify_skill("MotorMove") is None


# ── crash point ────────────────────────────────────────────────────────────

def test_crash_point_comes_from_the_failed_scan_centre():
    payload = {"skill": "FullScan", "success": False,
               "params": {"center_x_m": 1e-7, "center_y_m": 2e-7},
               "data": {"crash_indicator": True,
                        "center_x_m": 1.5e-7, "center_y_m": 2.5e-7}}
    # Result data wins over params: it is where the scan actually ran.
    assert crash_point(payload) == (1.5e-7, 2.5e-7)


def test_crash_point_falls_back_to_params():
    payload = {"skill": "FullScan", "success": False,
               "params": {"center_x_m": 1e-7, "center_y_m": 2e-7},
               "data": {"crash_indicator": True}}
    assert crash_point(payload) == (1e-7, 2e-7)


def test_no_crash_point_without_the_indicator():
    """A scan can fail for many reasons; only a crash marks the surface."""
    payload = {"skill": "FullScan", "success": False,
               "params": {"center_x_m": 1e-7, "center_y_m": 2e-7},
               "data": {"timed_out": True}}
    assert crash_point(payload) is None


def test_no_crash_point_on_a_successful_scan():
    payload = {"skill": "FullScan", "success": True,
               "params": {"center_x_m": 1e-7, "center_y_m": 2e-7},
               "data": {"crash_indicator": False}}
    assert crash_point(payload) is None


def test_crash_without_a_location_is_not_placed():
    """Better no marker than a marker in the wrong place."""
    payload = {"skill": "FullScan", "success": False, "params": {},
               "data": {"crash_indicator": True}}
    assert crash_point(payload) is None


# ── approach ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("skill", ["ApproachTip", "AutoApproach"])
def test_approach_skills_are_positioned(skill):
    assert classify_skill(skill) == "approach"


@pytest.mark.parametrize("skill", ["StopAutoApproach", "GetAutoApproachStatus"])
def test_approach_stop_and_status_are_not_positioned(skill):
    """Both contain "autoapproach" as a substring, and neither drives the tip
    into the surface — one halts a running approach, the other reads a flag."""
    assert classify_skill(skill) is None


# ── the skill's own position readback ──────────────────────────────────────

class _State:
    x_pos_m = 9e-7          # deliberately far from the readback below
    y_pos_m = 9e-7
    scan_center_x_m = None
    scan_center_y_m = None
    stale = False


def test_skill_reported_position_beats_the_cached_snapshot():
    """Tip forming leaves a permanent crater and takes no position parameter, so
    it reads its own position as it fires. That reading must win over a snapshot
    that may be a refresh interval old."""
    m = marker_from_skill("TipShape", {}, _State(),
                          data={"x_m": 1e-7, "y_m": 2e-7})
    assert (m.x_m, m.y_m) == (1e-7, 2e-7)
    assert m.kind == "tip_shape"


def test_explicit_parameters_still_win_over_a_readback():
    m = marker_from_skill("BiasPulse", {"x_m": 3e-7, "y_m": 4e-7}, _State(),
                          data={"x_m": 1e-7, "y_m": 2e-7})
    assert (m.x_m, m.y_m) == (3e-7, 4e-7)


def test_snapshot_is_used_when_the_skill_could_not_read_its_position():
    """An unreadable position is reported as absent, never guessed — so the
    marker degrades to the snapshot rather than disappearing."""
    m = marker_from_skill("TipShape", {}, _State(), data={})
    assert (m.x_m, m.y_m) == (9e-7, 9e-7)


def test_half_a_readback_is_discarded_rather_than_mixed():
    """A position is a pair. Taking x from the readback and y from the snapshot
    would synthesise a coordinate where nothing ever happened — and it would be
    unlabellable, since it came from neither source."""
    m = marker_from_skill("TipShape", {}, _State(),
                          data={"x_m": float("nan"), "y_m": 1e-7})
    assert (m.x_m, m.y_m) == (9e-7, 9e-7)


def test_readback_pair_is_taken_together():
    m = marker_from_skill("TipShape", {}, _State(),
                          data={"x_m": 1e-7, "y_m": 2e-7})
    assert (m.x_m, m.y_m) == (1e-7, 2e-7)


# ── spectroscopy position from the saved file's own header ─────────────────
#
# This lives in the RECORDER, not in AcquireSTS: parsing a saved file is a record
# concern, the skill's job is the measurement, and ``_attach_saved_dat`` has
# already resolved the path into the result. Anyone looking for it in
# spectroscopy.py will not find it — hence these tests.

def _dat(tmp_path, x="1.234e-7", y="5.678e-7", name="sts_001.dat"):
    p = tmp_path / name
    p.write_text(
        "Experiment\tbias spectroscopy\n"
        f"X (m)\t{x}\nY (m)\t{y}\n"
        "Date\t02.01.2000\n\n[DATA]\nBias (V)\tCurrent (A)\n0.0\t1e-12\n",
        encoding="utf-8")
    return str(p)


def test_spectrum_marker_takes_the_exact_xy_from_the_dat_header(tmp_path):
    """AcquireSTS carries no position at all — it measures wherever the tip is —
    so its marker would otherwise be placed from the cached snapshot. Nanonis
    wrote the true stage coordinate into the file it just saved."""
    path = _dat(tmp_path)
    m = marker_from_skill("AcquireSTS", {"save_basename": ""}, _State(),
                          data={"path": path})
    assert (m.x_m, m.y_m) == (9e-7, 9e-7)          # snapshot, before refinement
    refined, meta = _refine_marker_from_saved_file(m, {"path": path})
    assert (refined.x_m, refined.y_m) == (1.234e-7, 5.678e-7)
    assert meta["pos_src"] == "dat_header"
    assert meta["file"] == path


def test_refinement_only_touches_spectroscopy_markers(tmp_path):
    """A scan's footprint comes from the frame it rastered, not from a file
    header that happens to be in its result."""
    path = _dat(tmp_path)
    m = marker_from_skill("FullScan", {"center_x_m": 1e-7, "center_y_m": 1e-7},
                          _State(), data={"path": path})
    refined, meta = _refine_marker_from_saved_file(m, {"path": path})
    assert refined is m and meta == {}


def test_unparseable_header_leaves_the_readback_position(tmp_path):
    """Degrade to the snapshot rather than dropping the marker."""
    bad = tmp_path / "broken.dat"
    bad.write_text("no position here\n", encoding="utf-8")
    m = marker_from_skill("AcquireSTS", {}, _State(), data={"path": str(bad)})
    refined, meta = _refine_marker_from_saved_file(m, {"path": str(bad)})
    assert (refined.x_m, refined.y_m) == (9e-7, 9e-7)
    assert meta == {}


def test_absurd_header_coordinate_is_rejected(tmp_path):
    """A mis-parsed column must not put a marker a kilometre off the sample and
    blow up every extent calculation downstream (±1 mm plausibility guard)."""
    path = _dat(tmp_path, x="1.5", y="2.5")     # metres
    m = marker_from_skill("AcquireSTS", {}, _State(), data={"path": path})
    refined, _ = _refine_marker_from_saved_file(m, {"path": path})
    assert (refined.x_m, refined.y_m) == (9e-7, 9e-7)


def test_non_dat_artifact_path_is_ignored(tmp_path):
    m = marker_from_skill("AcquireSTS", {}, _State(),
                          data={"path": str(tmp_path / "scan.sxm")})
    refined, meta = _refine_marker_from_saved_file(m, {"path": str(tmp_path / "scan.sxm")})
    assert refined is m and meta == {}
