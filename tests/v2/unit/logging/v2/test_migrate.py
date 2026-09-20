"""End-to-end migration test from a synthetic v1 db to v2."""
import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
while _MASTV2_ROOT in sys.path:
    sys.path.remove(_MASTV2_ROOT)
sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import json
import sqlite3
import uuid
from datetime import datetime

import pytest

from mast.logging.v2.migrate import migrate
from mast.logging.v2.storage import ExperimentStoreV2


def _make_v1_db(path):
    """Create a small v1 db matching the vendored schema."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE experiments (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, goal_text TEXT DEFAULT '',
          start_time TEXT NOT NULL, end_time TEXT,
          status TEXT DEFAULT 'running', notes TEXT DEFAULT '');
        CREATE TABLE samples (
          id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
          name TEXT NOT NULL, description TEXT DEFAULT '',
          start_time TEXT NOT NULL, end_time TEXT,
          status TEXT DEFAULT 'active',
          sample_type TEXT DEFAULT '', sample_subtype TEXT DEFAULT '');
        CREATE TABLE actions (
          id TEXT PRIMARY KEY, experiment_id TEXT, sample_id TEXT,
          timestamp TEXT NOT NULL, skill_name TEXT NOT NULL,
          skill_version TEXT DEFAULT '', parameters TEXT DEFAULT '{}',
          result TEXT DEFAULT '', state_before TEXT DEFAULT '',
          state_after TEXT DEFAULT '', nanonis_calls TEXT DEFAULT '[]',
          context TEXT DEFAULT '', duration_s REAL DEFAULT 0,
          approval_source TEXT DEFAULT 'auto');
        CREATE TABLE environment_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
          sensor_name TEXT NOT NULL, value REAL NOT NULL,
          unit TEXT DEFAULT '', status TEXT DEFAULT 'ok');
        CREATE TABLE plans (
          plan_id TEXT PRIMARY KEY, experiment_id TEXT, sample_id TEXT,
          name TEXT, goal TEXT, definition TEXT,
          status TEXT, current_phase INTEGER, current_step INTEGER,
          notes TEXT, created_at TEXT, updated_at TEXT);
        """
    )
    eid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    aid1 = str(uuid.uuid4())
    aid2 = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO experiments (id, name, goal_text, start_time, status, notes) "
        "VALUES (?,?,?,?,?,?)",
        (eid, "Si(111) test", "Migrate me", now, "completed", "all good"),
    )
    conn.execute(
        "UPDATE experiments SET end_time = ? WHERE id = ?",
        (now, eid),
    )
    conn.execute(
        "INSERT INTO samples (id, experiment_id, name, description, start_time, "
        "sample_type, sample_subtype) VALUES (?,?,?,?,?,?,?)",
        (sid, eid, "sample-1", "", now, "clean_metal", "Si(111)"),
    )
    result = {"success": True, "elapsed_s": 0.05, "data": {}}
    state_after = {"bias_v": -2.0, "current_a": 1e-10, "z_pos_m": 1e-7}
    conn.execute(
        "INSERT INTO actions (id, experiment_id, sample_id, timestamp, skill_name, "
        "parameters, result, state_after, approval_source) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (aid1, eid, sid, now, "SetBias", json.dumps({"bias_v": -2.0}),
         json.dumps(result), json.dumps(state_after), "llm"),
    )
    fail_result = {"success": False, "error": "tip crashed", "elapsed_s": 1.0}
    conn.execute(
        "INSERT INTO actions (id, experiment_id, sample_id, timestamp, skill_name, "
        "parameters, result, approval_source) VALUES (?,?,?,?,?,?,?,?)",
        (aid2, eid, sid, now, "AutoApproach", "{}",
         json.dumps(fail_result), "llm"),
    )
    conn.commit()
    conn.close()


def test_full_migration(tmp_path):
    v1 = tmp_path / "v1.db"
    v2 = tmp_path / "v2.db"
    _make_v1_db(v1)

    report = migrate(v1, v2)
    assert report.campaigns_added == 1
    assert report.samples_added >= 1
    assert report.experiments_added == 1
    assert report.actions_added == 2
    # 4 state_after fields × 1 action with state_after = up to 4 observations
    assert report.observations_added >= 3
    assert report.warnings == []

    store = ExperimentStoreV2(v2)
    counts = store.table_counts()
    assert counts["experiments"] == 1
    assert counts["actions"] == 2
    assert counts["observations"] >= 3
    assert counts["campaigns"] == 1


def test_migration_idempotent(tmp_path):
    v1 = tmp_path / "v1.db"
    v2 = tmp_path / "v2.db"
    _make_v1_db(v1)
    r1 = migrate(v1, v2)
    r2 = migrate(v1, v2)
    assert r2.actions_added == 0
    assert r2.skipped >= r1.actions_added


def test_failed_action_marked_failed(tmp_path):
    v1 = tmp_path / "v1.db"
    v2 = tmp_path / "v2.db"
    _make_v1_db(v1)
    migrate(v1, v2)
    store = ExperimentStoreV2(v2)
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT action_type, status FROM actions ORDER BY action_type"
        ).fetchall()
    by_name = {r["action_type"]: r["status"] for r in rows}
    assert by_name["SetBias"] == "succeeded"
    assert by_name["AutoApproach"] == "failed"
