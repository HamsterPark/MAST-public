"""A truncated row window must not be allowed to invent a coordinate generation.

``ExperimentStorage.get_markers()`` returns at most ``limit`` (2000) rows, newest
first. Both consumers of the scan-map analysis used to count ``coarse_move`` rows
*inside that window* to decide which generation is live. On a long-lived sample
the count comes up short — and the failure is not a graceful one:
``filter_epoch`` trusts the stored ``coord_epoch`` column, so a short count does
not select "no rows", it selects a REAL, OLDER generation and analyses a patch of
surface the tip left long ago. Coverage, keep-out zones and the next scan
position all come back confidently wrong.

The authoritative answer, ``storage.current_epoch()``, is a COUNT over the whole
scope through ``idx_map_markers_scope`` — microseconds. There was never a reason
to re-derive it.

This mattered little while coarse moves were rare. The coarse-motion work makes
relocation routine, which turns this from a corner case into the normal one.
"""
from __future__ import annotations

import os

import pytest

from mast.io.map_analysis import AnalysisConfig, analyze_map, current_epoch_of


def _scan_row(idx: int, epoch: int, x_nm: float, y_nm: float) -> dict:
    return {
        "id": idx, "timestamp": f"2026-07-31T00:{idx % 60:02d}:00", "kind": "scan",
        "skill_name": "FullScan", "x_m": x_nm * 1e-9, "y_m": y_nm * 1e-9,
        "w_m": 100e-9, "h_m": 100e-9, "angle_deg": 0.0, "label": "",
        "status": "done", "source": "skill", "meta": {}, "coord_epoch": epoch,
    }


def _coarse_row(idx: int, epoch: int) -> dict:
    return {
        "id": idx, "timestamp": f"2026-07-31T00:{idx % 60:02d}:30",
        "kind": "coarse_move", "skill_name": "RelocateCoarseXY",
        "x_m": None, "y_m": None, "w_m": None, "h_m": None, "angle_deg": 0.0,
        "label": "粗动换区 x+", "status": "done", "source": "skill",
        "meta": {"direction": "x+", "steps": 200}, "coord_epoch": epoch,
    }


def _window_missing_the_first_move() -> list[dict]:
    """Rows as a truncated window would deliver them.

    Reality: 2 coarse moves have happened, so the live generation is 2. The
    window starts after the first one, so only ONE coarse_move row is visible —
    deriving from the window yields 1, and generation 1 has real rows in it.
    """
    rows: list[dict] = []
    n = 0
    # Generation 1 (the previous patch of surface, already worked over).
    for i in range(4):
        n += 1
        rows.append(_scan_row(n, 1, -400 + 100 * i, -400))
    n += 1
    rows.append(_coarse_row(n, 1))          # boundary: gen 1 → gen 2
    # Generation 2 (where the tip actually is now — one scan so far).
    n += 1
    rows.append(_scan_row(n, 2, 0, 0))
    return rows


def test_derived_epoch_is_wrong_when_the_window_is_truncated():
    """Establishes the premise: the derived count is short by the dropped moves."""
    rows = _window_missing_the_first_move()
    assert current_epoch_of(rows) == 1, "premise: the window sees only one boundary"


def test_explicit_epoch_selects_the_live_generation():
    cfg = AnalysisConfig()
    rows = _window_missing_the_first_move()

    truthful = analyze_map(rows, cfg, current_epoch=2)
    assert truthful.current_epoch == 2
    assert truthful.markers_current_epoch == 1, (
        "generation 2 holds exactly one scan in this fixture"
    )


def test_without_the_explicit_epoch_the_analysis_describes_the_wrong_surface():
    """The regression itself: a silent, confident answer about dead surface."""
    cfg = AnalysisConfig()
    rows = _window_missing_the_first_move()

    derived = analyze_map(rows, cfg)                      # old behaviour
    truthful = analyze_map(rows, cfg, current_epoch=2)    # fixed behaviour

    assert derived.current_epoch == 1 and truthful.current_epoch == 2
    # 4 scans + the coarse_move row itself: the boundary row is stamped with the
    # OLD generation on purpose (it is drawn in the pre-move frame).
    assert derived.markers_current_epoch == 5, "picks up the abandoned region"
    assert derived.markers_current_epoch != truthful.markers_current_epoch, (
        "if these ever agree this fixture stopped exercising the bug"
    )
    assert derived.coverage_frac > truthful.coverage_frac, (
        "the stale generation reports surface as covered that has never been "
        "imaged in the live frame — the one error that makes a survey skip real "
        "surface"
    )


def test_current_epoch_query_is_right_even_past_the_row_limit(tmp_path, monkeypatch):
    """``storage.current_epoch()`` counts the scope, not a window."""
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "markers.db"))
    from mast.logging.storage import ExperimentStorage

    st = ExperimentStorage(str(tmp_path / "markers.db"))
    for i in range(5):
        st.log_marker(kind="coarse_move", x_m=None, y_m=None,
                      experiment_id="e1", sample_id="s1",
                      meta={"direction": "x+", "steps": 100})
        for j in range(3):
            st.log_marker(kind="scan", x_m=j * 1e-9, y_m=0.0, w_m=1e-8, h_m=1e-8,
                          experiment_id="e1", sample_id="s1")

    assert st.current_epoch("e1", "s1") == 5

    narrow = st.get_markers("e1", "s1", limit=4)
    assert len(narrow) == 4
    assert current_epoch_of(narrow) < 5, (
        "premise: a narrow window under-counts — that is exactly why the query "
        "must be the authority"
    )


def test_last_coarse_move_timestamp_marks_the_boundary(tmp_path, monkeypatch):
    """The wall-clock boundary saved .sxm files are sorted against.

    A saved scan carries no ``coord_epoch``; its mtime versus this instant is the
    only handle on whether its header coordinates still mean anything."""
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "b.db"))
    from mast.logging.storage import ExperimentStorage

    st = ExperimentStorage(str(tmp_path / "b.db"))
    assert st.last_coarse_move_timestamp("e1", "s1") is None, (
        "no move yet ⇒ no boundary ⇒ nothing is stale"
    )

    st.log_marker(kind="scan", x_m=0.0, y_m=0.0, experiment_id="e1", sample_id="s1")
    assert st.last_coarse_move_timestamp("e1", "s1") is None, "scans are not boundaries"

    st.log_marker(kind="coarse_move", x_m=None, y_m=None,
                  experiment_id="e1", sample_id="s1")
    first = st.last_coarse_move_timestamp("e1", "s1")
    assert first is not None

    st.log_marker(kind="coarse_move", x_m=None, y_m=None,
                  experiment_id="e1", sample_id="s1")
    second = st.last_coarse_move_timestamp("e1", "s1")
    assert second is not None and second >= first, "must track the LATEST move"

    # Scoping: another sample's move is not this sample's boundary.
    assert st.last_coarse_move_timestamp("e1", "other") is None


@pytest.mark.skipif(os.name == "nt" and False, reason="placeholder guard")
def test_coarse_move_effects_are_best_effort_and_never_raise():
    """The invalidation hook runs AFTER the marker is committed.

    A failure there must not turn a completed relocation into a failed one, so it
    reports outcomes instead of raising — including when the things it clears are
    not importable at all."""
    from mast.core.coarse_move_effects import on_coarse_move_recorded

    out = on_coarse_move_recorded(source="test")
    assert set(out) == {"plan_overlay", "tip_crash_tracker"}
    assert all(isinstance(v, str) for v in out.values())
