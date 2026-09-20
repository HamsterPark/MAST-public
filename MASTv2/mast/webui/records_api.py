"""Adapter: v2 logging store → the JSON payload the Records redesign UI consumes.

The React prototype (``static/redesign/jsx/06-records*.jsx``) reads its data
from ``window.__MAST_RECORDS__``. This module shapes the real ``ExperimentStoreV2``
content into exactly the field names the prototype's mock constants used, so
the UI renders live data with zero JSX edits beyond the ``_MR.x ||`` fallback.

If the v2 db is missing or empty, ``build_records_payload`` returns ``{}`` so
the UI falls back to the prototype's bundled mock data (a populated demo).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HLC_RE = re.compile(r"^(\d{13})-(\d{4})-(.+)$")


def hlc_display(raw: str | None) -> str:
    """Convert a stored HLC ('1742658128000-0009-main') to the UI form
    ('2026-03-22T15:42:08Z-0009-main'). Pass through anything else."""
    if not raw:
        return ""
    m = _HLC_RE.match(raw)
    if not m:
        return raw
    pt_ms, counter, node = m.groups()
    dt = datetime.fromtimestamp(int(pt_ms) / 1000, tz=timezone.utc)
    return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}Z-{counter}-{node}"


def _loads(raw: Any, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


# ── builders ──────────────────────────────────────────────────────────

def build_records_payload(db_path: str | Path | None = None) -> dict:
    """Return the full ``window.__MAST_RECORDS__`` payload, or ``{}`` for mock."""
    from mast.logging.v2.repos import build_repos
    from mast.logging.v2.storage import ExperimentStoreV2, open_store

    try:
        store = ExperimentStoreV2(db_path) if db_path else open_store()
    except Exception as exc:
        logger.warning("records payload: cannot open v2 store: %s", exc)
        return {}

    repos = build_repos(store)
    experiments = repos.experiments.with_counts(limit=500)
    if not experiments:
        # Empty store → let the UI show its bundled demo mock.
        return {}

    campaigns = _campaigns(repos)
    samples = _samples(repos)
    plans = _plans(repos)
    exps = _experiments(repos, experiments)
    focus_exp_id = _pick_focus(experiments)
    actions = _actions(repos, focus_exp_id)
    observations = _observations(repos, focus_exp_id)
    scan_files = _scan_files(repos)
    approvals = _approvals(repos, actions)
    claims = _claims(repos)
    evidence = _evidence(repos, claims)
    events = _events(repos, focus_exp_id)
    audit = _audit(repos)
    policies = _policies(repos)

    return {
        "campaigns": campaigns,
        "samples": samples,
        "plans": plans,
        "experiments": exps,
        "focus_exp_id": focus_exp_id,
        "actions": actions,
        "observations": observations,
        "scan_files": scan_files,
        "approvals": approvals,
        "claims": claims,
        "evidence": evidence,
        "paper_refs": [],
        "events": events,
        "audit": audit,
        "policies": policies,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "_source": str(store.db_path),
    }


def _goal_progress_of(row: dict) -> dict:
    """一行纲领的目标判据现状。**永不抛。**

    实现借 ``campaign_tools._goal_progress`` —— 那是 agent 侧读到的同一份答案。
    两处各算一遍的话，面板上写着「1/2 满足」而 RD 手里是「判不了」，谁也不会
    报错，但用户与模型从此活在两个世界里。
    """
    try:
        from mast.agents._shared.campaign_tools import _goal_progress

        return _goal_progress(row)
    except Exception:  # noqa: BLE001
        return {"verdict": "unknown", "reason": "判据模块不可用"}


def _campaigns(repos) -> list[dict]:
    out = []
    for c in repos.campaigns.with_stats(limit=200):
        out.append({
            "id": c["id"],
            "title": c["title"],
            "hypothesis": c["hypothesis"],
            "hypothesis_kind": c["hypothesis_kind"],
            "goal_json": _loads(c.get("goal_json"), {}),
            "status": c["status"],
            "created_at": hlc_display(c.get("created_at")),
            "created_by": c.get("created_by", ""),
            "parent_campaign_id": c.get("parent_campaign_id"),
            # 目标判据（2026-08-28）。读即求值、永不抛 —— 一份判不了的纲领不该
            # 让整个记录页 500。
            "goal_progress": _goal_progress_of(c),
            "stats": {
                "experiments": c.get("experiment_count", 0),
                "actions": c.get("action_count", 0),
                "observations": c.get("observation_count", 0),
                "scans": c.get("scan_file_count", 0),
            },
        })
    return out


def _samples(repos) -> list[dict]:
    return [
        {
            "id": s["id"],
            "label": s["label"],
            "material": s["material"],
            "prep_method": s.get("prep_method") or "",
            "created_at": hlc_display(s.get("created_at")),
        }
        for s in repos.samples.list_active(limit=200)
    ]


def _plans(repos) -> list[dict]:
    with repos.store.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM plans ORDER BY created_at DESC LIMIT 200"
        ).fetchall()]
    return [
        {
            "id": p["id"],
            "campaign_id": p.get("campaign_id"),
            "plan_kind": p["plan_kind"],
            "title": p["title"],
            "hypothesis": p.get("hypothesis"),
            "status": p["status"],
        }
        for p in rows
    ]


def _experiments(repos, experiments: list[dict]) -> list[dict]:
    # Scan counts for every experiment in one GROUP BY pass instead of a
    # per-experiment connect()+query (that was an N+1 opening up
    # to 500 SQLite connections — one per experiment row).
    scan_counts: dict[str, int] = {}
    with repos.store.connect() as conn:
        for row in conn.execute(
            "SELECT a.experiment_id AS eid, COUNT(DISTINCT sf.id) AS c "
            "FROM scan_files sf "
            "JOIN actions a ON sf.produced_by_action_id = a.id "
            "GROUP BY a.experiment_id"
        ).fetchall():
            scan_counts[row["eid"]] = row["c"]

    out = []
    for e in experiments:
        scan_count = scan_counts.get(e["id"], 0)
        out.append({
            "id": e["id"],
            "campaign_id": e["campaign_id"],
            "sample_id": e["sample_id"],
            "plan_id": e.get("plan_id"),
            "title": e["title"],
            "exp_type": e["exp_type"],
            "started_at": hlc_display(e.get("started_at")),
            "ended_at": hlc_display(e.get("ended_at")) or None,
            "exit_status": e.get("exit_status"),
            "conclusion": e.get("conclusion"),
            "conclusion_evidence_ids": _loads(e.get("conclusion_evidence_ids"), None),
            "action_count": e.get("action_count", 0),
            "obs_count": e.get("observation_count", 0),
            "scan_count": scan_count,
        })
    return out


def _pick_focus(experiments: list[dict]) -> str:
    """Default-focus experiment for Timeline/Action: the one with most actions."""
    ranked = sorted(experiments, key=lambda e: e.get("action_count", 0), reverse=True)
    return ranked[0]["id"] if ranked else ""


def _actions(repos, experiment_id: str) -> list[dict]:
    if not experiment_id:
        return []
    out = []
    for a in repos.actions.for_experiment(experiment_id, limit=2000):
        out.append({
            "id": a["id"],
            "experiment_id": a["experiment_id"],
            "parent_action_id": a.get("parent_action_id"),
            "agent_id": a["agent_id"],
            "action_type": a["action_type"],
            "params": _loads(a.get("params_json"), {}),
            "hlc": hlc_display(a.get("hlc")),
            "status": a["status"],
            "duration_ms": a.get("duration_ms"),
            "state_delta": _loads(a.get("state_delta_json"), None),
            "error": a.get("error"),
        })
    return out


def _observations(repos, experiment_id: str) -> list[dict]:
    if not experiment_id:
        return []
    out = []
    for o in repos.observations.for_experiment(experiment_id, limit=2000):
        summary = ""
        rs = _loads(o.get("result_summary_json"), None)
        if isinstance(rs, dict):
            summary = rs.get("summary") or json.dumps(rs, ensure_ascii=False)[:160]
        out.append({
            "id": o["id"],
            "action_id": o["action_id"],
            "experiment_id": o["experiment_id"],
            "observable": o["observable"],
            "channel": o.get("channel"),
            "scan_file_id": o.get("scan_file_id"),
            "hlc": hlc_display(o.get("hlc")),
            "scalar_value": o.get("scalar_value"),
            "units": o.get("units"),
            "summary": summary,
        })
    return out


def _scan_files(repos) -> list[dict]:
    dedup = {}
    for grp in repos.scan_files.list_dedup_groups():
        ids = grp["file_ids"]
        for fid in ids[1:]:  # all but the first are dedup-of the first
            dedup[fid] = ids[0]
    out = []
    for sf in repos.scan_files.list_with_fixity(limit=500):
        out.append({
            "id": sf["id"],
            "sha256": sf["sha256"],
            "size": sf["size_bytes"],
            "format": sf["format_kind"],
            "current_path": sf["current_path"],
            "produced_by_action_id": sf["produced_by_action_id"],
            "fixity_ok": bool(sf["fixity_ok"]),
            "ocfl_object_id": sf.get("ocfl_object_id"),
            "dedup_of": dedup.get(sf["id"]),
            "meta": _loads(sf.get("meta_json"), {}),
        })
    return out


def _approvals(repos, actions: list[dict]) -> list[dict]:
    # Fetch every approval for the focus experiment's actions in ONE query
    # instead of repos.approvals.for_action() per action (_fetchone opens a
    # fresh SQLite connection per call, so up to 2000 actions meant up to 2000
    # connect()/close() round-trips on a Records refresh).
    action_ids = [a["id"] for a in actions]
    if not action_ids:
        return []
    rows: list[dict] = []
    with repos.store.connect() as conn:
        # SQLite caps host params (default 999); chunk to stay well under it.
        for i in range(0, len(action_ids), 500):
            chunk = action_ids[i:i + 500]
            placeholders = ",".join("?" * len(chunk))
            rows.extend(
                dict(r) for r in conn.execute(
                    f"SELECT * FROM approvals WHERE action_id IN ({placeholders})",
                    chunk,
                ).fetchall()
            )
    out = []
    seen = set()
    for ap in rows:
        if ap["id"] in seen:
            continue
        seen.add(ap["id"])
        out.append({
            "id": ap["id"],
            "action_id": ap["action_id"],
            "approver_id": ap["approver_id"],
            "approver_kind": ap["approver_kind"],
            "approval_method": ap["approval_method"],
            "approval_evidence": ap.get("approval_evidence"),
            "approved_at": hlc_display(ap.get("approved_at")),
            "expires_at": hlc_display(ap.get("expires_at")) or None,
            "policy_version": ap.get("policy_version"),
        })
    return out


def _claims(repos) -> list[dict]:
    return [
        {
            "id": c["id"],
            "experiment_id": c.get("experiment_id"),
            "campaign_id": c.get("campaign_id"),
            "statement": c["statement"],
            "confidence": c.get("confidence"),
            "status": c["status"],
            "created_at": hlc_display(c.get("created_at")),
            "created_by": c.get("created_by", ""),
        }
        for c in repos.claims.list_claims(limit=200)
    ]


def _evidence(repos, claims: list[dict]) -> list[dict]:
    out = []
    for c in claims:
        for e in repos.claims.edges_for(c["id"]):
            out.append({
                "id": e["id"],
                "claim_id": e["claim_id"],
                "edge_type": e["edge_type"],
                "target": {"kind": e["entity_kind"], "id": e["entity_id"]},
                "weight": e.get("weight"),
                "rationale": e.get("rationale"),
            })
    return out


def _events(repos, experiment_id: str) -> list[dict]:
    if not experiment_id:
        return []
    return [
        {
            "id": ev["id"],
            "topic": ev["topic"],
            "kind": ev["kind"],
            "severity": ev["severity"],
            "hlc": hlc_display(ev.get("hlc")),
            "experiment_id": ev.get("experiment_id"),
            "action_id": ev.get("action_id"),
            "producer": ev.get("producer"),
            "payload": _loads(ev.get("payload_json"), {}),
        }
        for ev in repos.events.for_experiment(experiment_id, limit=500)
    ]


def _audit(repos) -> list[dict]:
    with repos.store.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM audit_log ORDER BY hlc DESC LIMIT 100"
        ).fetchall()]
    return [
        {
            "id": r["id"],
            "actor_id": r["actor_id"],
            "actor_kind": r["actor_kind"],
            "event": r["event"],
            "hlc": hlc_display(r.get("hlc")),
            "payload": _loads(r.get("payload_json"), {}),
        }
        for r in rows
    ]


def _policies(repos) -> list[dict]:
    from mast.logging.v2 import policy
    return [
        {
            "id": p["id"],
            "action_type": p["action_type"],
            "requires_approval": bool(p["requires_approval"]),
            "required_kind": p.get("required_kind"),
            "reason": p["reason"],
            "version": p["version"],
            "active": bool(p["active"]),
        }
        for p in policy.list_active_policies(repos.store)
    ]


# ── persistence helper used by app.py at GUI build time ───────────────

def write_records_data(out_path: str | Path, db_path: str | Path | None = None) -> dict:
    """Build the payload and write it to *out_path* as JSON. Returns the payload."""
    payload = build_records_payload(db_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload
