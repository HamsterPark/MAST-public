"""Findings #79 (atomic approval+action) and #131 (tz-aware timestamps).

#79  — ApprovalService.issue_and_begin must insert the approval row and the
        dangerous action row inside ONE transaction with deferred FK checks, so
        there is no window where the approval FK or the require-approval trigger
        sees a half-written pair. Verifies success, atomic rollback on failure,
        and that the trigger still blocks the un-approved path.

#131 — every timestamp written by the v2 logging layer (schema_versions,
        policies, vector_search embedding_meta) must be timezone-aware UTC, not
        a naive datetime.utcnow() string.

Offline: pure SQLite + HashEmbedder; no network, no LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import sqlite3
from datetime import datetime

import pytest

from mast.logging.v2 import policy
from mast.logging.v2.repos import build_repos
from mast.logging.v2.storage import ExperimentStoreV2


@pytest.fixture
def store(tmp_path):
    return ExperimentStoreV2(tmp_path / "f.db")


@pytest.fixture
def repos(store):
    return build_repos(store)


def _scaffold(repos):
    cid = repos.campaigns.create(
        title="t", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(
        campaign_id=cid, sample_id=sid, title="e", exp_type="test")
    return eid


def _gate(store, action_type: str = "tip_pulse") -> None:
    """Register an explicit approval policy for the trigger-mechanism tests.

    The default seed (DEFAULT_DANGEROUS_ACTION_TYPES) is intentionally empty
    under the current safety model, so these tests register a representative
    gated action_type directly to exercise the generic
    trg_action_requires_approval mechanism.
    """
    policy.register_policies(store, (action_type,), requires_approval=True)


# ── ───────────────────────────────────────────────────────

def test_issue_and_begin_atomically_creates_approved_dangerous_action(store, repos):
    _gate(store)
    eid = _scaffold(repos)

    action_id, approval_id = repos.approvals.issue_and_begin(
        experiment_id=eid,
        agent_id="agent:IC",
        action_type="tip_pulse",            # DANGEROUS — trigger-guarded
        params={"amplitude_v": 3.0},
        approver_id="user:operator-1",
        approver_kind="human_operator",
        approval_method="gui_click",
        policy_version="v1.0.0",
    )

    act = repos.actions.get(action_id)
    assert act is not None
    assert act["action_type"] == "tip_pulse"
    assert act["status"] == "pending"

    appr = repos.approvals.for_action(action_id)
    assert appr is not None
    assert appr["id"] == approval_id
    assert appr["approver_id"] == "user:operator-1"


def test_dangerous_action_without_issue_and_begin_still_blocked(store, repos):
    """The plain begin() path must remain trigger-guarded (no regression)."""
    _gate(store)
    eid = _scaffold(repos)
    with pytest.raises(sqlite3.IntegrityError):
        repos.actions.begin(
            experiment_id=eid, agent_id="agent:IC", action_type="tip_pulse",
            params={"amplitude_v": 3.0},
        )


def test_issue_and_begin_rolls_back_both_rows_on_failure(store, repos):
    """If the action insert violates a constraint, the approval must not persist.

    A self-approval (approver_id starting with 'agent:' + non-policy kind) trips
    trg_no_self_approval on the approval insert, so neither row may survive.
    """
    _gate(store)
    eid = _scaffold(repos)

    with pytest.raises(sqlite3.IntegrityError):
        repos.approvals.issue_and_begin(
            experiment_id=eid,
            agent_id="agent:IC",
            action_type="tip_pulse",
            params={"amplitude_v": 3.0},
            approver_id="agent:XD",            # illegal self-approval
            approver_kind="human_operator",
        )

    # Nothing leaked: no tip_pulse action, no approval rows.
    with store.connect() as conn:
        n_actions = conn.execute(
            "SELECT COUNT(*) AS c FROM actions WHERE action_type = 'tip_pulse'"
        ).fetchone()["c"]
        n_appr = conn.execute(
            "SELECT COUNT(*) AS c FROM approvals"
        ).fetchone()["c"]
    assert n_actions == 0
    assert n_appr == 0


def test_issue_and_begin_rolls_back_when_action_insert_fails(store, repos):
    """Force the action insert (second statement) to fail and confirm the
    already-inserted approval is rolled back too (true atomicity, not just
    'approval first wins')."""
    _gate(store)
    eid = _scaffold(repos)

    # Bad experiment_id FK on the action → action insert fails after the
    # approval insert has run inside the same tx.
    with pytest.raises(sqlite3.IntegrityError):
        repos.approvals.issue_and_begin(
            experiment_id="nonexistent-experiment",
            agent_id="agent:IC",
            action_type="tip_pulse",
            params={"amplitude_v": 3.0},
            approver_id="user:operator-1",
            approver_kind="human_operator",
        )

    with store.connect() as conn:
        n_appr = conn.execute("SELECT COUNT(*) AS c FROM approvals").fetchone()["c"]
    assert n_appr == 0


def test_issue_and_begin_accepts_caller_supplied_action_id(store, repos):
    _gate(store)
    eid = _scaffold(repos)
    from mast.logging.v2.ulid import ulid_now
    pre = ulid_now()
    action_id, _ = repos.approvals.issue_and_begin(
        experiment_id=eid,
        agent_id="agent:IC",
        action_type="tip_pulse",
        approver_id="user:op",
        action_id=pre,
    )
    assert action_id == pre
    assert repos.actions.get(pre)["id"] == pre


# ── ──────────────────────────────────────────────────────

def _assert_tz_aware(ts: str):
    dt = datetime.fromisoformat(ts)
    assert dt.tzinfo is not None, f"timestamp not tz-aware: {ts!r}"
    assert dt.utcoffset() is not None


def test_schema_version_timestamp_is_tz_aware(store):
    with store.connect() as conn:
        row = conn.execute(
            "SELECT applied_at FROM schema_versions WHERE version = '2.0.0'"
        ).fetchone()
    _assert_tz_aware(row["applied_at"])


def test_policy_created_at_is_tz_aware(store):
    # Default seed is empty under the current safety model; register one
    # explicit policy so there is a row whose created_at we can assert on.
    _gate(store)
    rows = policy.list_active_policies(store)
    assert rows
    for r in rows:
        _assert_tz_aware(r["created_at"])


def test_vector_search_embedded_at_is_tz_aware(tmp_path):
    from mast.logging.v2.vector_search import NumpyFallbackSearch, HashEmbedder
    backend = NumpyFallbackSearch(
        tmp_path / "emb.db", HashEmbedder(dim=128), dim=128)
    backend.index(entity_kind="observation", entity_id="o1", text="hello world")
    with sqlite3.connect(str(tmp_path / "emb.db")) as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT embedded_at FROM embeddings").fetchone()
    backend.close()
    _assert_tz_aware(row["embedded_at"])
