"""map_markers.coord_epoch — the coordinate-system generation.

A lateral coarse-motor move slides the sample stage under the tip. The piezo XY
readout keeps reporting the same numbers, but they now address a different patch
of surface, so every marker recorded before that moment is stranded in a dead
coordinate system. ``coord_epoch`` is what lets the map draw them faded and lets
the analysis ignore them instead of treating ruined surface as ruined and fresh
surface as already-scanned.

Isolation: every test redirects the experiment DB with ``MAST_EXPERIMENT_DB`` into
tmp_path (tests writing into the operator's real records has happened four times
in this repo, always because something redirected one env var while storage read
another) — here the path is passed to the constructor explicitly, so there is
nothing to get out of sync.
"""
from __future__ import annotations

import sqlite3

import pytest

from mast.logging.storage import ExperimentStorage


@pytest.fixture()
def store(tmp_path):
    return ExperimentStorage(str(tmp_path / "exp.db"))


def _marker(store, kind, **kw):
    return store.log_marker(kind=kind, x_m=kw.pop("x_m", 0.0),
                            y_m=kw.pop("y_m", 0.0), **kw)


# ── stamping ───────────────────────────────────────────────────────────────

def test_epoch_advances_only_after_a_coarse_move(store):
    """A coarse_move row stamps the OLD generation: it marks the boundary and is
    itself drawn in the pre-move frame."""
    _marker(store, "scan")
    _marker(store, "coarse_move")
    _marker(store, "scan")
    _marker(store, "sts")
    assert [r["coord_epoch"] for r in store.get_markers()] == [0, 0, 1, 1]


def test_current_epoch_counts_coarse_moves(store):
    assert store.current_epoch() == 0
    _marker(store, "coarse_move")
    assert store.current_epoch() == 1
    _marker(store, "coarse_move")
    assert store.current_epoch() == 2


def test_z_moves_do_not_advance_the_generation(store):
    """Approaching or retracting does not change WHERE on the surface the tip
    is, so it invalidates no coordinate. Only the lateral move does, and the
    recorder is what decides that — nothing else may write a coarse_move."""
    _marker(store, "approach")
    _marker(store, "scan")
    assert store.current_epoch() == 0


def test_generation_is_scoped_per_experiment_and_sample(store):
    """换样品 → 新画布: a coarse move on one sample must not age another's
    markers."""
    _marker(store, "coarse_move", experiment_id="e1", sample_id="s1")
    assert store.current_epoch("e1", "s1") == 1
    assert store.current_epoch("e1", "s2") == 0
    assert store.current_epoch("e2", None) == 0
    _marker(store, "scan", experiment_id="e1", sample_id="s2")
    assert store.get_markers("e1", "s2")[0]["coord_epoch"] == 0


def test_marker_without_coordinates_still_bounds_a_generation(store):
    """A coarse move whose tip position could not be read is still a boundary —
    the count comes from the row existing, not from its xy."""
    store.log_marker(kind="coarse_move", x_m=None, y_m=None)
    assert store.current_epoch() == 1


# ── reading one generation ─────────────────────────────────────────────────

def test_get_markers_filters_to_one_generation(store):
    _marker(store, "scan", label="old")
    _marker(store, "coarse_move", label="boundary")
    _marker(store, "scan", label="new")
    # The boundary row belongs to the generation it ENDED, so it stays visible
    # with the markers it was drawn alongside.
    assert [r["label"] for r in store.get_markers(coord_epoch=0)] == ["old", "boundary"]
    assert [r["label"] for r in store.get_markers(coord_epoch=1)] == ["new"]


def test_generation_filter_composes_with_scope(store):
    _marker(store, "scan", experiment_id="e1", sample_id="s1", label="keep")
    _marker(store, "scan", experiment_id="e1", sample_id="s2", label="other")
    rows = store.get_markers("e1", "s1", coord_epoch=0)
    assert [r["label"] for r in rows] == ["keep"]


# ── legacy databases ───────────────────────────────────────────────────────

def _legacy_db(path) -> None:
    """A map_markers table exactly as it existed before the column."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE map_markers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            experiment_id TEXT,
            sample_id     TEXT,
            kind          TEXT NOT NULL DEFAULT 'move',
            skill_name    TEXT NOT NULL DEFAULT '',
            x_m           REAL,
            y_m           REAL,
            w_m           REAL,
            h_m           REAL,
            angle_deg     REAL NOT NULL DEFAULT 0.0,
            label         TEXT NOT NULL DEFAULT '',
            status        TEXT NOT NULL DEFAULT 'done',
            source        TEXT NOT NULL DEFAULT 'skill',
            meta          TEXT NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute("INSERT INTO map_markers (timestamp, kind, label) "
                 "VALUES ('2026-01-01T00:00:00', 'scan', 'legacy')")
    conn.commit()
    conn.close()


def test_opening_a_legacy_database_adds_the_column(tmp_path):
    db = tmp_path / "old.db"
    _legacy_db(db)
    ExperimentStorage(str(db))          # migration runs on open
    conn = sqlite3.connect(str(db))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(map_markers)")}
    conn.close()
    assert "coord_epoch" in cols


def test_legacy_rows_read_as_generation_zero(tmp_path):
    """NULL is not "unknown": a database with no coarse_move rows has only ever
    had one coordinate system, so every legacy row genuinely IS generation 0.
    That is why the migration adds the column without backfilling it."""
    db = tmp_path / "old.db"
    _legacy_db(db)
    store = ExperimentStorage(str(db))
    assert store.current_epoch() == 0
    # The legacy row's column is untouched…
    assert store.get_markers()[0]["coord_epoch"] is None
    # …yet asking for generation 0 still returns it.
    assert [r["label"] for r in store.get_markers(coord_epoch=0)] == ["legacy"]


def test_legacy_row_excluded_from_a_later_generation(tmp_path):
    db = tmp_path / "old.db"
    _legacy_db(db)
    store = ExperimentStorage(str(db))
    _marker(store, "coarse_move")
    _marker(store, "scan", label="fresh")
    assert [r["label"] for r in store.get_markers(coord_epoch=1)] == ["fresh"]
