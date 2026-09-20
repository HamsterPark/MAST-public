"""Agent training/usage trajectory layer (RFC agent_training_log_rfc.md, P1).

Covers TrajectoryRepo / StepRepo contracts, the append-only + set-once triggers,
the no-tensor JSON sanitizer, and the fire-and-forget QueuedTraceSink end-to-end.
"""
from __future__ import annotations

import sys
import sqlite3
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

from mast.logging.v2.repos import build_repos, _json_safe
from mast.logging.v2.storage import ExperimentStoreV2
from mast.logging.v2.trace_sink import QueuedTraceSink, NullTraceSink


@pytest.fixture
def repos(tmp_path):
    return build_repos(ExperimentStoreV2(tmp_path / "test.db"))


def test_trajectory_lifecycle(repos):
    tid = repos.trajectories.begin(
        thread_id="th-1",
        operator_intent={"raw_text": "scan Si(111)", "language": "en", "mode": "normal"},
        context_snapshot={"model_id": "kimi", "thinking_mode": "fixed"},
    )
    s1 = repos.steps.record(trajectory_id=tid, step_type="route_decision",
                            hop_idx=0, agent_id="SUP",
                            output={"to": "IC", "reason": "needs instrument"})
    s2 = repos.steps.record(trajectory_id=tid, step_type="tool_call",
                            parent_step_id=s1, agent_id="IC",
                            tool_call_id="tc-1",
                            output={"skill": "SetBias", "params": {"bias_v": 0.5}})
    repos.trajectories.end(tid, exit_status="success",
                           final_outcome={"answer_text": "done"})
    repos.trajectories.set_quality(tid, {"operator_rating": 5, "reward": 1.0})

    traj = repos.trajectories.get(tid)
    assert traj["thread_id"] == "th-1"
    assert traj["exit_status"] == "success"
    assert '"operator_rating": 5' in traj["quality_json"]
    steps = repos.steps.by_trajectory(tid)
    assert [s["id"] for s in steps] == [s1, s2]          # hlc-ordered
    assert steps[1]["parent_step_id"] == s1               # causal chain
    assert steps[1]["tool_call_id"] == "tc-1"


def test_end_is_set_once_via_coalesce(repos):
    tid = repos.trajectories.begin(thread_id="th-2")
    repos.trajectories.end(tid, exit_status="success")
    # Re-ending with a different status is a silent no-op (COALESCE keeps first).
    repos.trajectories.end(tid, exit_status="failed")
    assert repos.trajectories.get(tid)["exit_status"] == "success"


def test_quality_is_backfillable_multiple_times(repos):
    tid = repos.trajectories.begin(thread_id="th-3")
    repos.trajectories.set_quality(tid, {"reward": 0.2})
    repos.trajectories.set_quality(tid, {"reward": 0.9})       # revised
    assert '"reward": 0.9' in repos.trajectories.get(tid)["quality_json"]


def test_steps_are_append_only(repos):
    tid = repos.trajectories.begin(thread_id="th-4")
    sid = repos.steps.record(trajectory_id=tid, step_type="observation")
    with repos.store.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE trajectory_steps SET hlc='x' WHERE id=?", (sid,))
    with repos.store.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM trajectory_steps WHERE id=?", (sid,))


def test_trajectory_immutable_thread_id(repos):
    tid = repos.trajectories.begin(thread_id="th-5")
    with repos.store.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE trajectories SET thread_id='other' WHERE id=?", (tid,))


def test_json_safe_strips_non_primitive():
    out = _json_safe({"ok": 1, "txt": "x", "bad": {1, 2, 3},
                      "nested": {"arr": [1, object()]}})
    assert out["ok"] == 1 and out["txt"] == "x"
    assert out["bad"].startswith("<non-primitive:")
    assert out["nested"]["arr"][1].startswith("<non-primitive:")
    # huge string capped
    assert len(_json_safe("z" * 9000)) <= 4001


def test_agent_turn_reasoning_keeps_wider_cap(repos):
    """The agent_turn step (CoT) gets a wider per-string cap (16000) so multi-KB
    reasoning survives, while other step types stay at the conservative 4000."""
    tid = repos.trajectories.begin(thread_id="th-rc")
    big = "r" * 9000
    s_turn = repos.steps.record(trajectory_id=tid, step_type="agent_turn",
                                agent_id="IC", output={"reasoning": big})
    s_tool = repos.steps.record(trajectory_id=tid, step_type="tool_call",
                                agent_id="IC", output={"blob": big})
    import json as _j
    turn_reason = _j.loads(repos.steps.by_trajectory(tid)[0]["output_json"])["reasoning"]
    tool_blob = _j.loads(repos.steps.by_trajectory(tid)[1]["output_json"])["blob"]
    assert len(turn_reason) == 9000               # full reasoning kept (< 16000)
    assert len(tool_blob) <= 4001                  # non-turn still capped at 4000
    assert s_turn != s_tool


def test_queued_trace_sink_end_to_end(repos):
    sink = QueuedTraceSink(repos)
    try:
        tid = sink.begin_trajectory(thread_id="th-q",
                                    operator_intent={"raw_text": "go"})
        # ids returned synchronously (non-blocking)
        assert isinstance(tid, str) and tid
        sids = [sink.record_step(trajectory_id=tid, step_type="agent_turn",
                                 hop_idx=i, output={"i": i}) for i in range(5)]
        sink.end_trajectory(tid, exit_status="success")
        sink.set_quality(tid, {"reward": 1})
        sink.flush()                       # drain the background queue
        assert repos.trajectories.get(tid)["exit_status"] == "success"
        got = repos.steps.by_trajectory(tid)
        assert [s["id"] for s in got] == sids   # FIFO worker → trajectory before steps, in order
        assert sink.dropped == 0
    finally:
        sink.close()


def test_null_sink_mints_ids_but_writes_nothing(repos):
    sink = NullTraceSink()
    tid = sink.begin_trajectory(thread_id="th-n")
    assert isinstance(tid, str) and tid
    sink.record_step(trajectory_id=tid, step_type="error")
    sink.end_trajectory(tid, exit_status="failed")
    # nothing persisted
    assert repos.trajectories.get(tid) is None


# ── export (RFC P3) ───────────────────────────────────────────────────

import json
from mast.logging.v2 import trajectory_export as tex


def _seed(repos, *, thread, intent, steps, exit_status="success"):
    tid = repos.trajectories.begin(thread_id=thread,
                                   operator_intent={"raw_text": intent})
    for st in steps:
        repos.steps.record(trajectory_id=tid, **st)
    repos.trajectories.end(tid, exit_status=exit_status,
                           final_outcome={"answer_text": "done"})
    return tid


def test_export_jsonl_and_sft(repos, tmp_path):
    _seed(repos, thread="t1", intent="scan Si",
          steps=[{"step_type": "tool_call", "agent_id": "IC", "tool_call_id": "c1",
                  "output": {"skill": "SetBias", "success": True}}])
    _seed(repos, thread="t2", intent="condition tip",
          steps=[{"step_type": "tool_call", "agent_id": "IC",
                  "output": {"skill": "TipPulse", "success": True}}])
    out = tmp_path / "traj.jsonl"
    n = tex.export_jsonl(repos, str(out))
    assert n == 2
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["steps"][0]["output"]["skill"] in ("SetBias", "TipPulse")
    sft = list(tex.to_sft_samples(repos))
    assert len(sft) == 2
    assert all("intent" in s and "trajectory" in s for s in sft)
    assert sft[0]["intent"]["raw_text"] in ("scan Si", "condition tip")


def test_preference_pairs_from_hitl_edit(repos):
    tid = repos.trajectories.begin(thread_id="tp", operator_intent={"raw_text": "x"})
    repos.steps.record(trajectory_id=tid, step_type="hitl_resolution",
                       output={"verdict": "edit", "skill": "SetBias",
                               "model_params": {"bias_v": 2.0},
                               "operator_params": {"bias_v": 0.5},
                               "reason": "too high"})
    repos.trajectories.end(tid, exit_status="success")
    pairs = list(tex.to_preference_pairs(repos))
    assert len(pairs) == 1
    assert pairs[0]["rejected"] == {"bias_v": 2.0}
    assert pairs[0]["chosen"] == {"bias_v": 0.5}
    assert pairs[0]["skill"] == "SetBias"


def test_failure_view(repos):
    _seed(repos, thread="ok", intent="good", steps=[], exit_status="success")
    _seed(repos, thread="bad", intent="oops", steps=[], exit_status="failed")
    _seed(repos, thread="rb", intent="rolled",
          steps=[{"step_type": "rollback", "agent_id": "IC",
                  "output": {"rolled_back": True, "error": "boom"}}],
          exit_status="success")
    failed = list(tex.failure_view(repos))
    intents = {f["operator_intent"]["raw_text"] for f in failed}
    assert "oops" in intents and "rolled" in intents and "good" not in intents
