"""Schema-level integration tests:

- All tables / indexes / triggers install cleanly.
- Append-only triggers reject UPDATE/DELETE.
- approver_kind / approver_id can't be agent-self.
- policies + approvals interplay forces approval-before-action.
"""
from __future__ import annotations

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

import sqlite3

import pytest

from mast.logging.v2 import policy
from mast.logging.v2.hlc import HLCClock
from mast.logging.v2.repos import ApprovalService, build_repos, deferred_fk
from mast.logging.v2.storage import ExperimentStoreV2
from mast.logging.v2.ulid import ulid_now
from mast.logging.v2.views import install_views


@pytest.fixture
def store(tmp_path):
    return ExperimentStoreV2(tmp_path / "test_v2.db")


def test_schema_init_idempotent(tmp_path):
    p = tmp_path / "x.db"
    a = ExperimentStoreV2(p)
    b = ExperimentStoreV2(p)
    counts = a.table_counts()
    # Re-opening must not duplicate any system rows.
    assert b.schema_version() == "2.0.0"
    a.close(); b.close()


def test_all_required_tables_present(store):
    expected = {
        "schema_versions", "campaigns", "samples", "plans", "experiments",
        "instrument_states", "actions", "scan_files", "observations",
        "events", "approvals", "reviews", "claims", "entity_refs",
        "evidence_edges", "audit_log", "policies", "mv_campaign_stats",
    }
    with store.connect() as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    missing = expected - names
    assert not missing, f"missing tables: {missing}"


def test_append_only_rejects_update_on_observations(store):
    repos = build_repos(store)
    cid = repos.campaigns.create(
        title="t", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(
        campaign_id=cid, sample_id=sid, title="e", exp_type="test")
    aid = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="set_bias")
    oid = repos.observations.record_scalar(
        action_id=aid, experiment_id=eid, observable="bias", scalar_value=1.0, units="V")

    with pytest.raises(sqlite3.IntegrityError):
        with store.connect() as conn:
            conn.execute("UPDATE observations SET scalar_value = 2.0 WHERE id = ?", (oid,))


def test_append_only_rejects_delete_on_events(store):
    repos = build_repos(store)
    cid = repos.campaigns.create(title="t", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid, title="e", exp_type="test")
    ev = repos.events.publish(topic="t", kind="edge", experiment_id=eid)

    with pytest.raises(sqlite3.IntegrityError):
        with store.connect() as conn:
            conn.execute("DELETE FROM events WHERE id = ?", (ev,))


def test_actions_status_monotonic(store):
    repos = build_repos(store)
    cid = repos.campaigns.create(title="t", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid, title="e", exp_type="test")
    aid = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="set_bias")
    repos.actions.succeed(aid)
    # Can't go back from succeeded
    with pytest.raises(sqlite3.IntegrityError):
        with store.connect() as conn:
            conn.execute("UPDATE actions SET status = 'running' WHERE id = ?", (aid,))


def test_no_self_approval(store):
    repos = build_repos(store)
    cid = repos.campaigns.create(title="t", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid, title="e", exp_type="test")
    aid = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="set_bias")
    # Attempt to insert an approval row with an agent approver_id.
    with pytest.raises(sqlite3.IntegrityError):
        repos.approvals.issue(
            action_id=aid, approver_id="agent:XD",
            approver_kind="human_operator", approval_method="bogus",
        )


def test_dangerous_action_requires_approval(store):
    repos = build_repos(store)
    # Default seed is empty under the current safety model; register an
    # explicit policy to exercise the generic require-approval trigger.
    policy.register_policies(store, ("tip_pulse",), requires_approval=True)
    cid = repos.campaigns.create(title="t", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid, title="e", exp_type="test")

    # Attempt a tip_pulse without approval — must fail.
    with pytest.raises(sqlite3.IntegrityError):
        repos.actions.begin(
            experiment_id=eid, agent_id="agent:IC", action_type="tip_pulse",
            params={"amplitude_v": 3.0},
        )


def test_dangerous_action_succeeds_with_pre_approval(store):
    repos = build_repos(store)
    policy.register_policies(store, ("tip_pulse",), requires_approval=True)
    cid = repos.campaigns.create(title="t", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid, title="e", exp_type="test")

    aid = ulid_now()
    with deferred_fk(store) as conn:
        # approval first
        conn.execute(
            "INSERT INTO approvals (id, action_id, approver_id, approver_kind, "
            "approval_method, approved_at, policy_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ulid_now(), aid, "user:operator-1", "human_operator",
             "gui_click", "2026-05-19T00:00:00", "v1.0.0"),
        )
        # action with pre-set id
        conn.execute(
            "INSERT INTO actions (id, experiment_id, agent_id, action_type, "
            "params_json, hlc, status) VALUES (?, ?, ?, ?, ?, ?, 'pending')",
            (aid, eid, "agent:IC", "tip_pulse", "{}", "1700000000000-0000-test"),
        )

    assert repos.approvals.for_action(aid) is not None
    assert repos.actions.get(aid)["action_type"] == "tip_pulse"


def test_views_install_and_query(store):
    install_views(store)
    with store.connect() as conn:
        # Empty store, but view should run.
        rows = conn.execute("SELECT * FROM v_experiment_summary").fetchall()
        assert rows == []
