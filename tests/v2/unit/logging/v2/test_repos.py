"""End-to-end test of the repository layer:

A miniature experiment is recorded (campaign → sample → experiment →
actions → observations → scan file → events → claim graph → review) and
each step's repository contract is asserted.
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

from mast.logging.v2.repos import build_repos
from mast.logging.v2.storage import ExperimentStoreV2


@pytest.fixture
def repos(tmp_path):
    store = ExperimentStoreV2(tmp_path / "test.db")
    return build_repos(store)


def test_full_lifecycle(repos):
    # Campaign
    cid = repos.campaigns.create(
        title="Si(111) auto-survey",
        hypothesis="LLM can drive a clean Si scan with no manual interventions.",
        hypothesis_kind="confirmatory",
        goal={"target_observable": "topography", "success_criteria": ["rms_noise<5e-12"]},
        created_by="user:test",
    )
    assert repos.campaigns.get(cid)["title"].startswith("Si(111)")
    repos.campaigns.set_status(cid, "running")
    assert repos.campaigns.get(cid)["status"] == "running"

    # Sample
    sid = repos.samples.create(
        label="Si(111)-7x7 #3", material="Si",
        prep_method="sputter+anneal",
        prep_log={"cycles": 5, "max_temp_K": 1473},
    )

    # Plan
    pid = repos.plans.create(
        plan_kind="pre_experiment",
        title="topo-then-sts",
        definition={"steps": ["coarse_scan", "fine_scan", "sts_grid"]},
        campaign_id=cid,
        hypothesis="hypothesis",
        success_criteria={"min_observations": 5},
    )
    repos.plans.activate(pid)
    assert repos.plans.get(pid)["status"] == "active"

    # Experiment
    eid = repos.experiments.start(
        campaign_id=cid, sample_id=sid, title="topo #1", exp_type="topo_scan", plan_id=pid,
    )
    assert repos.experiments.get(eid)["title"] == "topo #1"

    # Snapshot
    snap = repos.instrument_states.snapshot(
        experiment_id=eid,
        state={"bias_v": -2.0, "current_a": 1e-10, "z_pos_m": 1e-7},
        reason="periodic",
    )
    assert repos.instrument_states.latest(eid)["id"] == snap

    # Actions
    a1 = repos.actions.begin(
        experiment_id=eid, agent_id="agent:IC", action_type="set_bias",
        params={"bias_v": -2.0},
    )
    repos.actions.succeed(a1, duration_ms=42)
    a2 = repos.actions.begin(
        experiment_id=eid, agent_id="agent:IC", action_type="scan",
        params={"width_m": 5e-9}, parent_action_id=a1,
    )
    repos.actions.succeed(a2, duration_ms=15000)

    chain = repos.actions.ancestry(a2)
    assert [r["id"] for r in chain] == [a1, a2]
    descendants = repos.actions.descendants(a1)
    assert any(r["id"] == a2 for r in descendants)

    # Scan file
    sfid = repos.scan_files.register(
        produced_by_action_id=a2,
        sha256="a" * 64,
        size_bytes=1024,
        current_path="/tmp/scan001.sxm",
        format_kind="sxm",
        parser_spec="nanonis-sxm-v3",
        meta={"bias_v": -2.0, "width_m": 5e-9},
    )
    # Idempotency on sha256 collision
    sfid2 = repos.scan_files.register(
        produced_by_action_id=a2, sha256="a" * 64, size_bytes=1024,
        current_path="/tmp/scan001-dup.sxm", format_kind="sxm",
        parser_spec="nanonis-sxm-v3",
    )
    assert sfid == sfid2

    # Observation
    o1 = repos.observations.record_scan(
        action_id=a2, experiment_id=eid, observable="topography",
        scan_file_id=sfid, channel="Z (m)",
    )
    o2 = repos.observations.record_scalar(
        action_id=a2, experiment_id=eid, observable="rms_noise",
        scalar_value=3.4e-12, units="m",
    )
    assert len(repos.observations.for_action(a2)) == 2

    # Event with dedup
    repos.events.publish(
        topic="instrument.tip_status", kind="snapshot",
        experiment_id=eid, payload={"state": "stable"}, dedup_key="stable",
    )
    none_again = repos.events.publish(
        topic="instrument.tip_status", kind="snapshot",
        experiment_id=eid, payload={"state": "stable"}, dedup_key="stable",
    )
    assert none_again is None
    assert len(repos.events.for_experiment(eid)) == 1

    # Result with too-large summary should be rejected at the repo guard.
    with pytest.raises(ValueError):
        repos.observations.record_summary(
            action_id=a2, experiment_id=eid, observable="huge",
            result_summary={"data": "x" * 10_000},
        )

    # Claim + edges
    claim_id = repos.claims.create_claim(
        statement="herringbone reconstruction visible",
        experiment_id=eid, confidence=0.7, created_by="agent:PR",
    )
    repos.claims.add_edge(
        claim_id=claim_id, target_kind="observation", target_id=o1,
        edge_type="mast:supports", weight=0.8,
    )
    edges = repos.claims.edges_for(claim_id)
    assert any(e["entity_id"] == o1 for e in edges)

    # Review
    rid = repos.reviews.record(
        target_kind="claim", target_id=claim_id,
        reviewer_id="user:pi", reviewer_kind="human_pi",
        verdict="accept", comments={"note": "OK"},
    )
    assert any(r["id"] == rid for r in repos.reviews.for_target("claim", claim_id))

    # Audit
    repos.audit.record(
        actor_id="user:test", actor_kind="user", event="manual_test",
        payload={"x": 1},
    )
    assert len(repos.audit.by_event("manual_test")) == 1

    # End experiment
    repos.experiments.end(eid, exit_status="success", conclusion="ok",
                          evidence_ids=[o1, o2])
    e = repos.experiments.get(eid)
    assert e["exit_status"] == "success"

    # Stats roll-up via trigger MV
    stats = repos.campaigns.with_stats(limit=10)
    row = next(s for s in stats if s["id"] == cid)
    assert row["experiment_count"] == 1
    assert row["action_count"] == 2
    assert row["observation_count"] == 2
    assert row["scan_file_count"] == 1


def test_action_retract_appends_compensating(repos):
    cid = repos.campaigns.create(title="t", hypothesis="h", goal={}, hypothesis_kind="exploratory")
    sid = repos.samples.create(label="s", material="m")
    eid = repos.experiments.start(campaign_id=cid, sample_id=sid, title="e", exp_type="test")
    a1 = repos.actions.begin(experiment_id=eid, agent_id="agent:t", action_type="set_bias")
    repos.actions.succeed(a1)
    retracted = repos.actions.retract(
        a1, retracted_by="user:operator", rationale="wrong target", experiment_id=eid,
    )
    assert repos.actions.get(a1)["status"] == "retracted"
    assert repos.actions.get(retracted)["action_type"] == "retract"
    assert repos.actions.get(retracted)["parent_action_id"] == a1
