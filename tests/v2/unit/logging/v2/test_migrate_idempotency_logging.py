"""Logging-group regression tests for review finding [30].

[30] migrate.migrate() claimed to be idempotent + transactional but was
neither: the id-mapping table was persisted only once at the very end, so a
crash mid-migration left an empty mapping and a re-run duplicated every row.

These tests prove the real fix: mappings are now persisted INCREMENTALLY
(``_save_one`` after each entity, with the action mapping committed only after
its observations), so a re-run after a simulated crash does NOT duplicate any
already-committed entity.

Run from the MASTv2 dir:
    ../.venv-v2-py313/Scripts/python.exe -m pytest \
        ../tests/v2/unit/logging/v2/test_migrate_idempotency_logging.py -v
(sys.path / mast-resolution handled by the sibling conftest.py.)
"""
from __future__ import annotations

# ── Force MASTv2/ to the head of sys.path BEFORE importing mast.* ─────
# (the sibling conftest does this too, but the inline block guarantees it even
# when pytest resolves `mast` to the v1 tree first during collection — same
# pattern as test_migrate.py.)
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


# ── v1 db fixture builder (mirrors the vendored v1 schema) ──────────

def _make_v1_db(path, *, n_actions: int = 5):
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
        CREATE TABLE plans (
          plan_id TEXT PRIMARY KEY, experiment_id TEXT, sample_id TEXT,
          name TEXT, goal TEXT, definition TEXT,
          status TEXT, current_phase INTEGER, current_step INTEGER,
          notes TEXT, created_at TEXT, updated_at TEXT);
        """
    )
    eid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO experiments (id, name, goal_text, start_time, end_time, status, notes) "
        "VALUES (?,?,?,?,?,?,?)",
        (eid, "Si(111) test", "Migrate me", now, now, "completed", "all good"),
    )
    conn.execute(
        "INSERT INTO samples (id, experiment_id, name, description, start_time, "
        "sample_type, sample_subtype) VALUES (?,?,?,?,?,?,?)",
        (sid, eid, "sample-1", "", now, "clean_metal", "Si(111)"),
    )
    conn.execute(
        "INSERT INTO plans (plan_id, experiment_id, name, goal, definition, status, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), eid, "plan-1", "do science", json.dumps({"steps": []}),
         "active", now),
    )
    state_after = {"bias_v": -2.0, "current_a": 1e-10, "z_pos_m": 1e-7}
    result = {"success": True, "elapsed_s": 0.05, "data": {}}
    action_ids = []
    for i in range(n_actions):
        aid = str(uuid.uuid4())
        action_ids.append(aid)
        conn.execute(
            "INSERT INTO actions (id, experiment_id, sample_id, timestamp, skill_name, "
            "parameters, result, state_after, approval_source) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, eid, sid, now, f"SetBias{i}", json.dumps({"bias_v": -2.0}),
             json.dumps(result), json.dumps(state_after), "llm"),
        )
    conn.commit()
    conn.close()
    return {"experiment": eid, "sample": sid, "actions": action_ids}


def _counts(v2_path) -> dict:
    store = ExperimentStoreV2(v2_path)
    return store.table_counts()


def _map_rows(v2_path) -> dict[tuple[str, str], str]:
    conn = sqlite3.connect(str(v2_path))
    conn.row_factory = sqlite3.Row
    try:
        return {
            (r["kind"], r["v1_id"]): r["v2_id"]
            for r in conn.execute("SELECT * FROM _v1_to_v2_map")
        }
    finally:
        conn.close()


# ── Tests ─────────────────────────────────────────────────────────────

def test_incremental_mapping_is_persisted_per_entity(tmp_path):
    """After a successful migration every migrated entity has its OWN
    committed mapping row — the proof that idempotency no longer depends on a
    single final flush."""
    v1 = tmp_path / "v1.db"
    v2 = tmp_path / "v2.db"
    ids = _make_v1_db(v1, n_actions=5)

    migrate(v1, v2)
    m = _map_rows(v2)

    # default campaign, the sample, the experiment, the plan, all 5 actions
    assert ("campaign", "__default__") in m
    assert ("sample", ids["sample"]) in m
    assert ("experiment", ids["experiment"]) in m
    for aid in ids["actions"]:
        assert ("action", aid) in m, f"action {aid} mapping not persisted"
    # 1 campaign + 1 sample + 1 experiment + 1 plan + 5 actions
    assert len(m) == 9


def test_rerun_after_crash_does_not_duplicate_rows(tmp_path):
    """THE core [30] regression: simulate a crash partway through the action
    loop, then re-run a full migration. The re-run must NOT duplicate the
    actions/observations the first (partial) run already committed."""
    v1 = tmp_path / "v1.db"
    v2 = tmp_path / "v2.db"
    _make_v1_db(v1, n_actions=5)

    # ---- Run 1: crash after the 3rd action's observations are distilled ----
    # We patch ObservationRepo.record_scalar so that the 3rd action raises
    # AFTER its observations are written but mapping logic continues; simpler
    # and more faithful: blow up inside actions.begin on the 4th action so the
    # 4th/5th actions never get committed at all.
    # NB: the action loop wraps each action in `except Exception`, so a process
    # crash must be modelled with a BaseException (KeyboardInterrupt) that the
    # bare `except Exception` does NOT catch — this is exactly what a real
    # SIGINT / power loss looks like to the migrator.
    import mast.logging.v2.migrate as migrate_mod

    real_atomic = migrate_mod._migrate_one_action_atomic
    call_box = {"n": 0}

    def exploding_atomic(*args, **kwargs):
        call_box["n"] += 1
        if call_box["n"] == 4:
            raise KeyboardInterrupt("simulated crash mid-migration (4th action)")
        return real_atomic(*args, **kwargs)

    migrate_mod._migrate_one_action_atomic = exploding_atomic
    try:
        with pytest.raises(KeyboardInterrupt):
            migrate(v1, v2)
    finally:
        migrate_mod._migrate_one_action_atomic = real_atomic

    after_crash = _counts(v2)
    map_after_crash = _map_rows(v2)
    # 3 actions committed (their mappings persisted incrementally), the 4th
    # raised before any begin() row, so exactly 3 action mappings exist.
    assert after_crash["actions"] == 3, after_crash
    assert sum(1 for k in map_after_crash if k[0] == "action") == 3
    # observations: 3 actions × 3 scalar fields (bias_v, current_a, z_pos_m) = 9
    assert after_crash["observations"] == 9, after_crash
    # the experiment / sample / campaign mappings survived the crash too
    assert after_crash["experiments"] == 1
    assert after_crash["samples"] == 1
    assert after_crash["campaigns"] == 1

    # ---- Run 2: full re-run, no crash ----
    report = migrate(v1, v2)
    final = _counts(v2)

    # NO duplication: still exactly 5 actions, 1 experiment, 1 sample, 1 campaign.
    assert final["actions"] == 5, f"actions duplicated: {final}"
    assert final["experiments"] == 1, f"experiments duplicated: {final}"
    assert final["samples"] == 1, f"samples duplicated: {final}"
    assert final["campaigns"] == 1, f"campaigns duplicated: {final}"
    assert final["plans"] == 1, f"plans duplicated: {final}"
    # 5 actions × 3 observable scalars = 15
    assert final["observations"] == 15, f"observations duplicated: {final}"

    # The re-run added only the 2 missing actions, skipped the already-migrated.
    assert report.actions_added == 2
    assert report.skipped >= 3  # 3 actions + experiment + sample skipped


def test_double_full_run_is_pure_noop(tmp_path):
    """A clean migrate() followed by a second clean migrate() adds nothing."""
    v1 = tmp_path / "v1.db"
    v2 = tmp_path / "v2.db"
    _make_v1_db(v1, n_actions=4)

    migrate(v1, v2)
    counts1 = _counts(v2)
    r2 = migrate(v1, v2)
    counts2 = _counts(v2)

    # Every DATA table is byte-for-byte unchanged on the second run. (The
    # audit_log intentionally grows by exactly one row per invocation — each
    # migration run is auditable — so it is excluded from the no-op assertion.)
    data_tables1 = {k: v for k, v in counts1.items() if k != "audit_log"}
    data_tables2 = {k: v for k, v in counts2.items() if k != "audit_log"}
    assert data_tables1 == data_tables2, "second run mutated a data table"
    assert counts2["audit_log"] == counts1["audit_log"] + 1, "audit not appended once"

    assert r2.actions_added == 0
    assert r2.experiments_added == 0
    assert r2.samples_added == 0
    assert r2.campaigns_added == 0
    assert r2.observations_added == 0
    assert r2.plans_added == 0


def test_action_is_fully_atomic_on_mid_transaction_crash(tmp_path):
    """A crash hitting BETWEEN the action INSERT and its observation INSERTs must
    roll the WHOLE action unit back — leaving NO orphan action row at all, so the
    re-run re-creates it once (never duplicating) and it ends with exactly its
    observations.

    We inject the crash inside _migrate_one_action_atomic's transaction by
    making ``ulid_now`` raise on the 2nd id it mints for the first action (the
    1st id is the action id, the 2nd is the first observation's id — i.e. AFTER
    the action row was INSERTed but BEFORE commit)."""
    v1 = tmp_path / "v1.db"
    v2 = tmp_path / "v2.db"
    _make_v1_db(v1, n_actions=2)

    import mast.logging.v2.migrate as migrate_mod
    real_ulid = migrate_mod.ulid_now
    box = {"n": 0}

    def exploding_ulid():
        box["n"] += 1
        # action.id is the 1st mint; the 1st observation.id is the 2nd mint —
        # by then the action row is already INSERTed inside the open txn.
        if box["n"] == 2:
            raise KeyboardInterrupt("simulated crash mid-action-transaction")
        return real_ulid()

    migrate_mod.ulid_now = exploding_ulid
    try:
        with pytest.raises(KeyboardInterrupt):
            migrate(v1, v2)
    finally:
        migrate_mod.ulid_now = real_ulid

    # Full rollback: NO action row, NO observation, NO action mapping survived
    # the crashed transaction.
    after = _counts(v2)
    assert after["actions"] == 0, f"action row not rolled back: {after}"
    assert after["observations"] == 0, f"observation not rolled back: {after}"
    map_after = _map_rows(v2)
    assert sum(1 for k in map_after if k[0] == "action") == 0

    # Re-run completes everything cleanly: exactly 2 actions (no duplicate),
    # each with its 3 scalar observations.
    migrate(v1, v2)
    final = _counts(v2)
    assert final["actions"] == 2, final
    assert final["observations"] == 6, final

    # No action ends up with the wrong number of observations.
    conn = sqlite3.connect(str(v2))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT a.id, COUNT(o.id) AS n FROM actions a "
            "LEFT JOIN observations o ON o.action_id = a.id "
            "GROUP BY a.id"
        ).fetchall()
    finally:
        conn.close()
    for r in rows:
        assert r["n"] == 3, f"action {r['id']} has {r['n']} observations (expected 3)"
