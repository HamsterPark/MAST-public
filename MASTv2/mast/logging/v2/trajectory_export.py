"""Export agent training/usage trajectories to training datasets (RFC P3).

Read-side only — joins ``trajectories`` + ``trajectory_steps`` (and the parsed
JSON fields) into:

  * full JSONL dump (one self-contained trajectory per line),
  * SFT samples (intent + context → ordered step sequence),
  * DPO/preference pairs (from ``hitl_resolution`` steps where the operator
    edited the model's params — model original = rejected, human = chosen),
  * a failure view (exit_status failed/aborted, or any rolled_back step).

Nothing here writes to the store, so it can never break a run.
"""
from __future__ import annotations

import json
from typing import Any, Iterator


def _loads(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


def assemble(repos: Any, trajectory_id: str) -> dict | None:
    """Full trajectory record: header + parsed steps (RFC §1 shape)."""
    traj = repos.trajectories.get(trajectory_id)
    if not traj:
        return None
    steps = repos.steps.by_trajectory(trajectory_id)
    return {
        "id": traj["id"],
        "thread_id": traj["thread_id"],
        "experiment_id": traj.get("experiment_id"),
        "created_at_hlc": traj["created_at_hlc"],
        "ended_at_hlc": traj.get("ended_at_hlc"),
        "operator_intent": _loads(traj["operator_intent_json"]),
        "context_snapshot": _loads(traj["context_snapshot_json"]),
        "exit_status": traj.get("exit_status"),
        "final_outcome": _loads(traj.get("final_outcome_json")),
        "quality": _loads(traj.get("quality_json")),
        "steps": [
            {
                "id": s["id"],
                "type": s["step_type"],
                "hop_idx": s.get("hop_idx"),
                "parent_step_id": s.get("parent_step_id"),
                "agent_id": s.get("agent_id"),
                "actor_kind": s.get("actor_kind"),
                "model_id": s.get("model_id"),
                "tool_call_id": s.get("tool_call_id"),
                "action_id": s.get("action_id"),
                "input": _loads(s.get("input_json")),
                "output": _loads(s.get("output_json")),
                "duration_ms": s.get("duration_ms"),
            }
            for s in steps
        ],
    }


def iter_full(repos: Any, *, limit: int = 10000,
              thread_id: str | None = None) -> Iterator[dict]:
    for t in repos.trajectories.list_recent(limit=limit):
        if thread_id is not None and t["thread_id"] != thread_id:
            continue
        rec = assemble(repos, t["id"])
        if rec is not None:
            yield rec


def export_jsonl(repos: Any, out_path: str, *, limit: int = 10000,
                 thread_id: str | None = None) -> int:
    """Write one self-contained trajectory per line. Returns the count."""
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in iter_full(repos, limit=limit, thread_id=thread_id):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def to_sft_samples(repos: Any, *, limit: int = 10000) -> Iterator[dict]:
    """One SFT sample per trajectory: the operator intent + initial context as
    the input, the ordered (type, agent, input, output) step sequence as the
    target trajectory, plus the outcome/quality for filtering."""
    for rec in iter_full(repos, limit=limit):
        yield {
            "intent": rec["operator_intent"],
            "context": rec["context_snapshot"],
            "trajectory": [
                {"type": s["type"], "agent": s["agent_id"],
                 "input": s["input"], "output": s["output"]}
                for s in rec["steps"]
            ],
            "exit_status": rec["exit_status"],
            "final_outcome": rec["final_outcome"],
            "quality": rec["quality"],
        }


def to_preference_pairs(repos: Any, *, limit: int = 10000) -> Iterator[dict]:
    """DPO/preference pairs from ``hitl_resolution`` steps where the operator
    EDITED the model's proposed params: model original = rejected, operator
    correction = chosen. (Populated once P2 wires HITL-resolution capture.)"""
    for rec in iter_full(repos, limit=limit):
        for s in rec["steps"]:
            if s["type"] != "hitl_resolution":
                continue
            out = s["output"] or {}
            if (out.get("verdict") == "edit"
                    and out.get("model_params") is not None
                    and out.get("operator_params") is not None):
                yield {
                    "intent": rec["operator_intent"],
                    "skill": out.get("skill"),
                    "rejected": out["model_params"],     # model's original
                    "chosen": out["operator_params"],    # human correction
                    "reason": out.get("reason"),
                    "trajectory_id": rec["id"],
                }


def failure_view(repos: Any, *, limit: int = 10000) -> Iterator[dict]:
    """Trajectories that failed/aborted, or contain a rolled-back step — for
    failure mining."""
    for rec in iter_full(repos, limit=limit):
        if rec["exit_status"] in ("failed", "aborted", "timeout"):
            yield rec
            continue
        if any((s["output"] or {}).get("rolled_back") for s in rec["steps"]):
            yield rec
