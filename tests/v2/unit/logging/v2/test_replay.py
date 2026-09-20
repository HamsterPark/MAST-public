"""Evaluation replay (RFC P3) — fixture loading, decision signatures, and the
gold-vs-candidate step diff (route consistency / tool agreement / param drift /
out-of-bounds safety blocks / HITL triggers)."""
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
from mast.logging.v2 import replay as rp


@pytest.fixture
def repos(tmp_path):
    return build_repos(ExperimentStoreV2(tmp_path / "replay.db"))


def _seed_run(repos, *, thread, intent, hw_state, route, tools, safety=(), hitl=(),
              exit_status="success"):
    tid = repos.trajectories.begin(
        thread_id=thread,
        operator_intent={"raw_text": intent},
        context_snapshot={"hardware_state": hw_state, "model_id": "kimi"},
    )
    for frm, to in route:
        repos.steps.record(trajectory_id=tid, step_type="route_decision",
                           agent_id="SUP", output={"from": frm, "to": to})
    for agent, skill, params in tools:
        repos.steps.record(trajectory_id=tid, step_type="tool_call", agent_id=agent,
                           input={"params": params}, output={"skill": skill, "success": True})
    for skill, verdict in safety:
        repos.steps.record(trajectory_id=tid, step_type="safety_gate", agent_id="IC",
                           input={"skill": skill}, output={"verdict": verdict})
    for skill, verdict in hitl:
        repos.steps.record(trajectory_id=tid, step_type="hitl_resolution",
                           output={"skill": skill, "verdict": verdict})
    repos.trajectories.end(tid, exit_status=exit_status)
    return tid


def test_reconstruct_initial_state():
    snap = {"hardware_state": {"bias_v": 1.0, "z_controller_on": True}, "model_id": "x"}
    assert rp.reconstruct_initial_state(snap) == {"bias_v": 1.0, "z_controller_on": True}
    assert rp.reconstruct_initial_state({"no_state": 1}) is None
    assert rp.reconstruct_initial_state(None) is None


def test_load_fixture(repos):
    tid = _seed_run(repos, thread="t1", intent="scan HOPG",
                    hw_state={"bias_v": 0.5},
                    route=[("SUP", "IC")],
                    tools=[("IC", "SetBias", {"bias_v": 0.5})])
    fx = rp.load_fixture(repos, tid)
    assert fx is not None
    assert fx.intent == {"raw_text": "scan HOPG"}
    assert fx.initial_state == {"bias_v": 0.5}
    assert fx.thread_id == "t1"
    assert rp.load_fixture(repos, "nonexistent") is None


def test_signatures(repos):
    tid = _seed_run(repos, thread="t2", intent="x", hw_state={},
                    route=[("SUP", "IC"), ("IC", "SUP")],
                    tools=[("IC", "SetBias", {"bias_v": 1.0})],
                    safety=[("SetBias", "allow")],
                    hitl=[("MotorMove", "approve")])
    rec = rp.load_fixture(repos, tid).gold
    assert rp.route_signature(rec) == [("SUP", "IC"), ("IC", "SUP")]
    assert rp.tool_signature(rec) == [("IC", "SetBias", ("bias_v",))]
    assert rp.safety_signature(rec) == [("SetBias", "allow")]
    assert rp.hitl_signature(rec) == [("MotorMove", "approve")]


def test_diff_identical_run(repos):
    common = dict(hw_state={}, route=[("SUP", "IC")],
                  tools=[("IC", "SetBias", {"bias_v": 1.0})])
    g = rp.load_fixture(repos, _seed_run(repos, thread="g", intent="x", **common)).gold
    c = rp.load_fixture(repos, _seed_run(repos, thread="c", intent="x", **common)).gold
    d = rp.diff(g, c)
    assert d.route_identical is True
    assert d.tool_skill_jaccard == 1.0
    assert d.param_drift == {}
    assert d.hitl_trigger_match is True


def test_diff_route_divergence(repos):
    g = rp.load_fixture(repos, _seed_run(
        repos, thread="g2", intent="x", hw_state={},
        route=[("SUP", "IC")], tools=[])).gold
    c = rp.load_fixture(repos, _seed_run(
        repos, thread="c2", intent="x", hw_state={},
        route=[("SUP", "data_processing")], tools=[])).gold
    d = rp.diff(g, c)
    assert d.route_identical is False
    assert d.route_gold == [("SUP", "IC")]
    assert d.route_candidate == [("SUP", "data_processing")]


def test_diff_param_drift_and_out_of_bounds(repos):
    g = rp.load_fixture(repos, _seed_run(
        repos, thread="g3", intent="x", hw_state={},
        route=[("SUP", "IC")],
        tools=[("IC", "SetBias", {"bias_v": 1.0})],
        safety=[("SetBias", "allow")])).gold
    # candidate proposes an out-of-bounds bias → safety blocks it ('落界')
    c = rp.load_fixture(repos, _seed_run(
        repos, thread="c3", intent="x", hw_state={},
        route=[("SUP", "IC")],
        tools=[("IC", "SetBias", {"bias_v": 99.0})],
        safety=[("SetBias", "block")])).gold
    d = rp.diff(g, c)
    assert "SetBias" in d.param_drift
    assert d.param_drift["SetBias"]["gold"] == {"bias_v": 1.0}
    assert d.param_drift["SetBias"]["candidate"] == {"bias_v": 99.0}
    assert d.safety_blocks_gold == 0
    assert d.safety_blocks_candidate == 1


def test_diff_hitl_trigger_mismatch(repos):
    g = rp.load_fixture(repos, _seed_run(
        repos, thread="g4", intent="x", hw_state={}, route=[], tools=[],
        hitl=[("MotorMove", "approve")])).gold
    c = rp.load_fixture(repos, _seed_run(
        repos, thread="c4", intent="x", hw_state={}, route=[], tools=[],
        hitl=[])).gold
    d = rp.diff(g, c)
    assert d.hitl_trigger_match is False
    assert d.hitl_triggers_gold == ["MotorMove"]
    assert d.hitl_triggers_candidate == []


def test_replay_with_injectable_runner(repos):
    gold_id = _seed_run(repos, thread="gold", intent="scan Si", hw_state={"bias_v": 0.5},
                        route=[("SUP", "IC")],
                        tools=[("IC", "SetBias", {"bias_v": 0.5})])

    # fake runner: "re-runs" by seeding an identical candidate trajectory, asserting
    # it received the fixture's fixed input + reconstructed initial state.
    seen = {}

    def fake_run(fx: rp.ReplayFixture) -> str:
        seen["intent"] = fx.intent
        seen["initial_state"] = fx.initial_state
        return _seed_run(repos, thread="cand", intent="scan Si", hw_state={"bias_v": 0.5},
                         route=[("SUP", "IC")],
                         tools=[("IC", "SetBias", {"bias_v": 0.5})])

    result = rp.replay_with(repos, gold_id, fake_run)
    assert seen["intent"] == {"raw_text": "scan Si"}
    assert seen["initial_state"] == {"bias_v": 0.5}
    assert result["candidate_id"] is not None
    assert result["diff"].route_identical is True
    assert result["diff"].tool_skill_jaccard == 1.0


def test_replay_with_missing_gold(repos):
    assert rp.replay_with(repos, "nope", lambda fx: "x") is None


def test_replay_with_failed_run(repos):
    gid = _seed_run(repos, thread="g5", intent="x", hw_state={}, route=[], tools=[])
    result = rp.replay_with(repos, gid, lambda fx: None)
    assert result["candidate_id"] is None
    assert result["diff"] is None
