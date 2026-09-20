"""Evaluation replay for agent trajectories (RFC P3).

Re-run a recorded trajectory's FIXED operator input against a (possibly new)
model and diff the resulting decisions step-by-step — without real hardware. The
original run's ``context_snapshot.hardware_state`` reconstructs the simulator's
initial state, so a replay can be driven entirely off the logged fixture.

This module is READ-side + orchestration glue only:
  * ``load_fixture`` / ``reconstruct_initial_state`` pull the fixed input and the
    initial state out of a recorded trajectory.
  * the ``*_signature`` helpers reduce an assembled trajectory to comparable
    decision sequences (route / tool / safety / hitl).
  * ``diff`` compares a GOLD trajectory against a CANDIDATE and reports route
    consistency, tool-skill agreement, parameter drift, out-of-bounds (safety
    block) events, and HITL-trigger agreement.
  * ``replay_with`` is injectable: it hands the fixture to a caller-supplied
    ``run_fn`` (which actually executes a graph against a fresh recorder sink and
    returns the new trajectory id), then assembles + diffs. The live-graph wiring
    lives in the GUI/CLI so this module imports no agent — keeping it testable
    with a fake ``run_fn`` and free of the no-cross-boundary hazard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mast.logging.v2 import trajectory_export as _tex


@dataclass
class ReplayFixture:
    """The fixed input + reconstructed initial state of a recorded run."""
    trajectory_id: str
    thread_id: str
    intent: dict | None
    context_snapshot: dict | None
    initial_state: dict | None
    gold: dict  # the assembled gold trajectory (header + steps)


def reconstruct_initial_state(context_snapshot: dict | None) -> dict | None:
    """Pull the hardware-state dict the run started from, to seed a simulator's
    ``get_state`` (no real instrument needed). Returns None when the snapshot
    carried no hardware state."""
    if not isinstance(context_snapshot, dict):
        return None
    for key in ("hardware_state", "hardwareState", "state", "instrument_state"):
        v = context_snapshot.get(key)
        if isinstance(v, dict):
            return v
    return None


def load_fixture(repos: Any, trajectory_id: str) -> ReplayFixture | None:
    """Assemble the gold trajectory and extract its replayable fixture."""
    gold = _tex.assemble(repos, trajectory_id)
    if gold is None:
        return None
    snap = gold.get("context_snapshot")
    return ReplayFixture(
        trajectory_id=trajectory_id,
        thread_id=gold.get("thread_id"),
        intent=gold.get("operator_intent"),
        context_snapshot=snap,
        initial_state=reconstruct_initial_state(snap),
        gold=gold,
    )


# ── decision signatures (order-preserving, hashable) ──────────────────

def route_signature(rec: dict) -> list[tuple]:
    """Ordered (from, to) route decisions."""
    out = []
    for s in rec.get("steps", []):
        if s.get("type") == "route_decision":
            o = s.get("output") or {}
            out.append((o.get("from"), o.get("to")))
    return out


def tool_signature(rec: dict) -> list[tuple]:
    """Ordered (agent_id, skill, sorted-param-keys) tool calls."""
    out = []
    for s in rec.get("steps", []):
        if s.get("type") != "tool_call":
            continue
        o = s.get("output") or {}
        params = (s.get("input") or {}).get("params") or {}
        keys = tuple(sorted(params.keys())) if isinstance(params, dict) else ()
        out.append((s.get("agent_id"), o.get("skill"), keys))
    return out


def safety_signature(rec: dict) -> list[tuple]:
    """Ordered (skill, verdict) safety-gate decisions. A verdict=='block' is an
    out-of-bounds ('落界') event."""
    out = []
    for s in rec.get("steps", []):
        if s.get("type") != "safety_gate":
            continue
        o = s.get("output") or {}
        i = s.get("input") or {}
        out.append((i.get("skill"), o.get("verdict")))
    return out


def hitl_signature(rec: dict) -> list[tuple]:
    """Ordered (skill, verdict) HITL resolutions."""
    out = []
    for s in rec.get("steps", []):
        if s.get("type") != "hitl_resolution":
            continue
        o = s.get("output") or {}
        out.append((o.get("skill"), o.get("verdict")))
    return out


def _tool_params(rec: dict) -> dict[str, dict]:
    """skill -> the first call's params (for drift comparison)."""
    out: dict[str, dict] = {}
    for s in rec.get("steps", []):
        if s.get("type") != "tool_call":
            continue
        skill = (s.get("output") or {}).get("skill")
        params = (s.get("input") or {}).get("params")
        if skill and skill not in out and isinstance(params, dict):
            out[skill] = params
    return out


@dataclass
class ReplayDiff:
    route_identical: bool
    route_gold: list = field(default_factory=list)
    route_candidate: list = field(default_factory=list)
    tool_skills_gold: list = field(default_factory=list)
    tool_skills_candidate: list = field(default_factory=list)
    tool_skill_jaccard: float = 0.0
    param_drift: dict = field(default_factory=dict)   # skill -> {gold, candidate}
    safety_blocks_gold: int = 0
    safety_blocks_candidate: int = 0
    hitl_triggers_gold: list = field(default_factory=list)
    hitl_triggers_candidate: list = field(default_factory=list)
    hitl_trigger_match: bool = True

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def diff(gold: dict, candidate: dict) -> ReplayDiff:
    """Step-by-step eval diff of a CANDIDATE replay against the GOLD trajectory:
    route consistency, tool-skill agreement (Jaccard), per-skill parameter drift,
    out-of-bounds (safety block) counts, and HITL-trigger agreement."""
    rg, rc = route_signature(gold), route_signature(candidate)
    tg, tc = tool_signature(gold), tool_signature(candidate)
    sg, sc = safety_signature(gold), safety_signature(candidate)
    hg, hc = hitl_signature(gold), hitl_signature(candidate)

    skills_g = {t[1] for t in tg if t[1]}
    skills_c = {t[1] for t in tc if t[1]}

    pg, pc = _tool_params(gold), _tool_params(candidate)
    drift: dict[str, dict] = {}
    for skill in skills_g & skills_c:
        if pg.get(skill) != pc.get(skill):
            drift[skill] = {"gold": pg.get(skill), "candidate": pc.get(skill)}

    return ReplayDiff(
        route_identical=(rg == rc),
        route_gold=rg, route_candidate=rc,
        tool_skills_gold=sorted(skills_g), tool_skills_candidate=sorted(skills_c),
        tool_skill_jaccard=_jaccard(skills_g, skills_c),
        param_drift=drift,
        safety_blocks_gold=sum(1 for _, v in sg if v == "block"),
        safety_blocks_candidate=sum(1 for _, v in sc if v == "block"),
        hitl_triggers_gold=[h[0] for h in hg],
        hitl_triggers_candidate=[h[0] for h in hc],
        hitl_trigger_match=({h[0] for h in hg} == {h[0] for h in hc}),
    )


def replay_with(
    repos: Any,
    trajectory_id: str,
    run_fn: Callable[[ReplayFixture], str | None],
) -> dict | None:
    """Replay a recorded trajectory and diff it against the original.

    ``run_fn(fixture)`` is the injectable runner: it executes a graph against the
    fixture's fixed ``intent`` (seeding the simulator from ``initial_state``) on a
    FRESH recorder sink, and returns the NEW trajectory id (or None on failure).
    The live-graph wiring lives in the caller so this module imports no agent.

    Returns ``{fixture, candidate_id, diff}`` or None if the gold trajectory or
    the replay run is missing."""
    fixture = load_fixture(repos, trajectory_id)
    if fixture is None:
        return None
    candidate_id = run_fn(fixture)
    if not candidate_id:
        return {"fixture": fixture, "candidate_id": None, "diff": None}
    candidate = _tex.assemble(repos, candidate_id)
    if candidate is None:
        return {"fixture": fixture, "candidate_id": candidate_id, "diff": None}
    return {
        "fixture": fixture,
        "candidate_id": candidate_id,
        "diff": diff(fixture.gold, candidate),
    }


__all__ = [
    "ReplayFixture", "ReplayDiff",
    "reconstruct_initial_state", "load_fixture",
    "route_signature", "tool_signature", "safety_signature", "hitl_signature",
    "diff", "replay_with",
]
