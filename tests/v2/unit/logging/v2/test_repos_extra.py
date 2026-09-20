"""Tests for the repos methods added for the redesign UI:
ScanFileRepo.list_dedup_groups / list_with_fixity / observation_count,
ClaimGraphRepo.compose (3-table transaction), and raw_query (SELECT-only).
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

import pytest

from mast.logging.v2.repos import build_repos, raw_query
from mast.logging.v2.storage import ExperimentStoreV2


@pytest.fixture
def repos(tmp_path):
    return build_repos(ExperimentStoreV2(tmp_path / "x.db"))


def _seed_experiment(repos):
    cid = repos.campaigns.create(title="c", hypothesis="h", goal={},
                                  hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid,
                                   title="e", exp_type="test")
    return cid, sid, eid


def test_register_dedups_by_sha256(repos):
    """CAS invariant: re-registering the same sha256 returns the same row id —
    the UNIQUE constraint on scan_files.sha256 means content is stored once."""
    _, _, eid = _seed_experiment(repos)
    a1 = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="scan")
    a2 = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="scan")
    f1 = repos.scan_files.register(produced_by_action_id=a1, sha256="a" * 64,
                                   size_bytes=1, current_path="/p1", format_kind="sxm",
                                   parser_spec="t")
    f2 = repos.scan_files.register(produced_by_action_id=a2, sha256="a" * 64,
                                   size_bytes=1, current_path="/p1-dup", format_kind="sxm",
                                   parser_spec="t")
    assert f1 == f2  # same content → same CAS row


def test_list_dedup_groups_empty_under_cas_invariant(repos):
    """Under the sha256 UNIQUE constraint there is one row per content, so
    list_dedup_groups is structurally always []. The method exists for the
    API contract / defensive use against legacy imports."""
    _, _, eid = _seed_experiment(repos)
    a = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="scan")
    repos.scan_files.register(produced_by_action_id=a, sha256="a" * 64, size_bytes=1,
                              current_path="/p1", format_kind="sxm", parser_spec="t")
    repos.scan_files.register(produced_by_action_id=a, sha256="b" * 64, size_bytes=1,
                              current_path="/p2", format_kind="sxm", parser_spec="t")
    assert repos.scan_files.list_dedup_groups() == []


def test_list_with_fixity_filters_format(repos):
    _, _, eid = _seed_experiment(repos)
    a = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="scan")
    repos.scan_files.register(produced_by_action_id=a, sha256="c" * 64, size_bytes=1,
                              current_path="/x.sxm", format_kind="sxm", parser_spec="t")
    repos.scan_files.register(produced_by_action_id=a, sha256="d" * 64, size_bytes=1,
                              current_path="/x.dat", format_kind="dat", parser_spec="t")
    assert len(repos.scan_files.list_with_fixity()) == 2
    assert len(repos.scan_files.list_with_fixity(format_kind="sxm")) == 1


def test_compose_writes_three_tables(repos):
    cid, _, eid = _seed_experiment(repos)
    a = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="scan")
    repos.actions.succeed(a)
    o = repos.observations.record_scalar(action_id=a, experiment_id=eid,
                                          observable="bias", scalar_value=1.0, units="V")
    claim_id = repos.claims.compose(
        statement="bias scan supports clean surface",
        confidence=0.8, status="proposed",
        experiment_id=eid, campaign_id=cid,
        created_by="operator:alice",
        edges=[
            {"kind": "observation", "id": o, "edge_type": "prov:wasDerivedFrom", "weight": 0.7},
            {"kind": "action", "id": a, "edge_type": "prov:wasGeneratedBy", "weight": 1.0},
        ],
    )
    assert repos.claims.get_claim(claim_id)["statement"].startswith("bias scan")
    edges = repos.claims.edges_for(claim_id)
    assert len(edges) == 2
    kinds = {e["entity_kind"] for e in edges}
    assert kinds == {"observation", "action"}


def test_compose_rejects_agent_attribution(repos):
    cid, _, eid = _seed_experiment(repos)
    with pytest.raises(ValueError):
        repos.claims.compose(statement="x", edges=[], created_by="agent:paper_writing")


def test_raw_query_allows_select(repos):
    cid, _, eid = _seed_experiment(repos)
    rows = raw_query(repos.store, "SELECT id, title FROM experiments")
    assert len(rows) == 1
    assert rows[0]["id"] == eid


def test_raw_query_rejects_write(repos):
    for bad in (
        "DELETE FROM actions",
        "UPDATE actions SET status='x'",
        "INSERT INTO actions VALUES (1)",
        "DROP TABLE actions",
        "SELECT 1; DELETE FROM actions",
        "PRAGMA table_info(actions)",
    ):
        with pytest.raises(ValueError):
            raw_query(repos.store, bad)


def test_raw_query_with_cte(repos):
    _, _, eid = _seed_experiment(repos)
    rows = raw_query(
        repos.store,
        "WITH x AS (SELECT id FROM experiments) SELECT COUNT(*) AS n FROM x",
    )
    assert rows[0]["n"] == 1
