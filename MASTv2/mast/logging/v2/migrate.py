"""One-shot migration of the vendored v1 logging schema → v2.

Reads ``mast_experiments.db`` (the v1 5-table file) and writes the contents
into a fresh ``mast_experiments_v2.db`` (the 13-entity v2 schema), filling
in missing fields with the safest reasonable defaults. Idempotent:
re-running on the same v2 file is a no-op for already-migrated rows
(detected by stable id-mapping table).

Strategy
========
* v1 ``experiments`` → v2 ``experiments`` (under a single default campaign).
* v1 ``samples``     → v2 ``samples``.
* v1 ``actions``     → v2 ``actions`` + (best-effort) ``observations``
                       distilled from ``state_after`` and ``result.data``.
* v1 ``plans``       → v2 ``plans``.

Idempotency / crash-safety
==========================
The repo layer commits each write on its own connection (one autocommit per
statement), so the migration is **not** a single big transaction — that would
require routing every repo write through one shared connection, a change that
belongs in the repo layer, not here. Instead idempotency is provided by an
``_v1_to_v2_map`` table that is persisted **incrementally**: the mapping row
for each migrated entity is committed *immediately after* that entity (and, for
actions, after its distilled observations) is written. Because entities are
created in FK-dependency order, a crash leaves a consistent already-migrated
prefix, and re-running ``migrate()`` is a true no-op for every row whose
mapping was committed — it never duplicates them. The mapping row is the
commit marker: an entity is only considered "migrated" once its mapping is
saved, so a crash *before* the mapping write simply re-creates that one entity
on the next run (the half-written entity is harmless — it has no mapping and is
not referenced by any committed downstream row).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from mast.logging.v2.hlc import HLCClock
from mast.logging.v2.repos import build_repos
from mast.logging.v2.storage import ExperimentStoreV2
from mast.logging.v2.ulid import ulid_now

logger = logging.getLogger(__name__)


@dataclass
class MigrationReport:
    campaigns_added: int = 0
    samples_added: int = 0
    experiments_added: int = 0
    actions_added: int = 0
    observations_added: int = 0
    plans_added: int = 0
    skipped: int = 0
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def summary(self) -> str:
        lines = [
            "── v1 → v2 migration report ──",
            f"  campaigns added       : {self.campaigns_added}",
            f"  samples added         : {self.samples_added}",
            f"  experiments added     : {self.experiments_added}",
            f"  actions added         : {self.actions_added}",
            f"  observations distilled: {self.observations_added}",
            f"  plans added           : {self.plans_added}",
            f"  rows skipped          : {self.skipped}",
        ]
        if self.warnings:
            lines.append(f"  warnings              : {len(self.warnings)}")
            lines.extend(f"    - {w}" for w in self.warnings[:10])
            if len(self.warnings) > 10:
                lines.append(f"    ... and {len(self.warnings) - 10} more")
        return "\n".join(lines)


def migrate(
    v1_db_path: str | Path,
    v2_db_path: str | Path,
    *,
    default_campaign_title: str = "Migrated v1 experiments",
    default_campaign_hypothesis: str = "Bulk import of pre-2026-05-19 v1 records",
    node_id: str = "migrator",
) -> MigrationReport:
    v1_db_path = Path(v1_db_path)
    v2_db_path = Path(v2_db_path)
    if not v1_db_path.exists():
        raise FileNotFoundError(v1_db_path)

    store = ExperimentStoreV2(v2_db_path)
    clock = HLCClock(node_id=node_id)
    repos = build_repos(store, clock=clock)
    report = MigrationReport()

    # ── id mapping ────────────────────────────────────────────────────
    _ensure_mapping_table(store)
    v1_to_v2: dict[tuple[str, str], str] = _load_mapping(store)

    v1_conn = sqlite3.connect(str(v1_db_path))
    v1_conn.row_factory = sqlite3.Row

    # ── 1. Default campaign ──────────────────────────────────────────
    campaign_id = _get_or_create_default_campaign(
        repos, v1_to_v2, default_campaign_title, default_campaign_hypothesis,
    )
    if campaign_id and ("campaign", "__default__") not in v1_to_v2:
        v1_to_v2[("campaign", "__default__")] = campaign_id
        _save_one(store, "campaign", "__default__", campaign_id)
        report.campaigns_added += 1

    # ── 2. Samples ────────────────────────────────────────────────────
    v1_samples = v1_conn.execute("SELECT * FROM samples").fetchall()
    for row in v1_samples:
        row = dict(row)
        key = ("sample", row["id"])
        if key in v1_to_v2:
            report.skipped += 1
            continue
        try:
            label = row["name"] or f"sample-{row['id'][:8]}"
            material = (row.get("sample_type") or "") + (
                f"/{row['sample_subtype']}" if row.get("sample_subtype") else ""
            )
            sid = repos.samples.create(
                label=label,
                material=material or "unknown",
                prep_method=None,
                prep_log={"v1_description": row.get("description") or ""},
            )
            v1_to_v2[key] = sid
            _save_one(store, "sample", row["id"], sid)
            report.samples_added += 1
        except Exception as exc:
            report.warnings.append(f"sample {row['id']}: {exc}")

    # ── 3. Experiments ────────────────────────────────────────────────
    v1_exps = v1_conn.execute("SELECT * FROM experiments").fetchall()
    for row in v1_exps:
        row = dict(row)
        key = ("experiment", row["id"])
        if key in v1_to_v2:
            report.skipped += 1
            continue
        try:
            # Find a sample for this experiment if any v1 action mentions one.
            sid = _find_v1_sample_for_experiment(v1_conn, row["id"])
            sample_v2 = v1_to_v2.get(("sample", sid)) if sid else None
            if sample_v2 is None:
                # Fall back to a placeholder sample per experiment.
                sample_v2 = repos.samples.create(
                    label=f"unknown-{row['id'][:8]}",
                    material="unknown",
                    prep_log={"reason": "v1 had no sample link"},
                )
                report.samples_added += 1
            eid = repos.experiments.start(
                campaign_id=campaign_id,
                sample_id=sample_v2,
                title=row["name"] or "unnamed",
                exp_type="legacy_v1",
            )
            v1_to_v2[key] = eid
            _save_one(store, "experiment", row["id"], eid)
            # Close it if v1 closed it.
            if row.get("end_time"):
                exit_status = "success" if row["status"] == "completed" else "aborted"
                repos.experiments.end(
                    eid, exit_status=exit_status,
                    conclusion=row.get("notes") or row.get("goal_text") or None,
                )
            report.experiments_added += 1
        except Exception as exc:
            report.warnings.append(f"experiment {row['id']}: {exc}")

    # ── 4. Plans ──────────────────────────────────────────────────────
    try:
        v1_plans = v1_conn.execute("SELECT * FROM plans").fetchall()
    except sqlite3.OperationalError:
        v1_plans = []
    for row in v1_plans:
        row = dict(row)
        key = ("plan", row["plan_id"])
        if key in v1_to_v2:
            report.skipped += 1
            continue
        try:
            v2_exp = v1_to_v2.get(("experiment", row.get("experiment_id")))
            definition = json.loads(row["definition"]) if row.get("definition") else {}
            pid = repos.plans.create(
                plan_kind="pre_experiment",
                title=row.get("name") or "unnamed plan",
                definition=definition,
                campaign_id=campaign_id,
                experiment_id=v2_exp,
                hypothesis=row.get("goal"),
                created_by="legacy_v1",
            )
            v1_to_v2[key] = pid
            _save_one(store, "plan", row["plan_id"], pid)
            report.plans_added += 1
        except Exception as exc:
            report.warnings.append(f"plan {row.get('plan_id')}: {exc}")

    # ── 5. Actions + Observations ────────────────────────────────────
    v1_actions = v1_conn.execute(
        "SELECT * FROM actions ORDER BY timestamp"
    ).fetchall()
    for row in v1_actions:
        row = dict(row)
        key = ("action", row["id"])
        if key in v1_to_v2:
            report.skipped += 1
            continue
        try:
            v2_exp = v1_to_v2.get(("experiment", row.get("experiment_id")))
            if v2_exp is None:
                report.warnings.append(f"action {row['id']}: experiment not migrated")
                report.skipped += 1
                continue
            # The action, its distilled observations, and its id-mapping row are
            # written in ONE transaction (all-or-nothing). This makes the per-
            # action migration genuinely atomic: a crash/interrupt at any point
            # rolls the whole unit back, so there is never a committed action
            # row without its mapping (which would otherwise be re-created — and
            # thus duplicated — on the next run).
            aid, n_obs = _migrate_one_action_atomic(
                store, row, v2_exp, clock,
            )
            v1_to_v2[key] = aid
            report.actions_added += 1
            report.observations_added += n_obs
        except Exception as exc:
            report.warnings.append(f"action {row['id']}: {exc}")

    # ── 6. Final mapping sweep (defensive) + audit trail ──────────────
    # Every mapping row was already committed incrementally via _save_one as
    # each entity was migrated (this is what makes re-runs idempotent). This
    # final INSERT OR IGNORE sweep is a no-op for already-saved rows and only
    # exists to catch any mapping that an above branch added to the in-memory
    # dict without an explicit _save_one (belt-and-suspenders).
    _save_mapping(store, v1_to_v2)
    repos.audit.record(
        actor_id="migrator",
        actor_kind="system",
        event="v1_to_v2_bulk_migration",
        payload={
            "v1_db": str(v1_db_path),
            "added": {
                "campaigns": report.campaigns_added,
                "samples": report.samples_added,
                "experiments": report.experiments_added,
                "actions": report.actions_added,
                "observations": report.observations_added,
                "plans": report.plans_added,
            },
            "skipped": report.skipped,
            "warnings": len(report.warnings),
        },
    )
    v1_conn.close()
    return report


# ── Helpers ───────────────────────────────────────────────────────────

def _ensure_mapping_table(store: ExperimentStoreV2) -> None:
    with store.connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _v1_to_v2_map ("
            "  kind TEXT NOT NULL, v1_id TEXT NOT NULL, v2_id TEXT NOT NULL, "
            "  PRIMARY KEY (kind, v1_id)"
            ")"
        )


def _load_mapping(store: ExperimentStoreV2) -> dict[tuple[str, str], str]:
    with store.connect() as conn:
        return {
            (r["kind"], r["v1_id"]): r["v2_id"]
            for r in conn.execute("SELECT * FROM _v1_to_v2_map")
        }


def _save_mapping(store: ExperimentStoreV2, m: dict[tuple[str, str], str]) -> None:
    with store.connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO _v1_to_v2_map(kind, v1_id, v2_id) VALUES (?, ?, ?)",
            [(k, v1_id, v2_id) for (k, v1_id), v2_id in m.items()],
        )


def _save_one(store: ExperimentStoreV2, kind: str, v1_id: str, v2_id: str) -> None:
    """Persist a single id-mapping row immediately so re-runs are idempotent.

    Uses INSERT OR IGNORE keyed on (kind, v1_id): if a mapping for this v1 row
    already exists (e.g. a previous partial run committed it), the new row is
    ignored and the original v2_id stands — re-runs never duplicate or rewrite
    an already-migrated entity. This row is the per-entity commit marker.
    """
    with store.connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO _v1_to_v2_map(kind, v1_id, v2_id) VALUES (?, ?, ?)",
            (kind, v1_id, v2_id),
        )


def _get_or_create_default_campaign(
    repos,
    mapping: dict,
    title: str,
    hypothesis: str,
) -> str:
    existing = mapping.get(("campaign", "__default__"))
    if existing:
        return existing
    return repos.campaigns.create(
        title=title, hypothesis=hypothesis, hypothesis_kind="exploratory",
        goal={"target_observable": "n/a", "success_criteria": [], "budget_hours": None},
        created_by="migrator",
    )


def _find_v1_sample_for_experiment(v1_conn: sqlite3.Connection, exp_id: str) -> str | None:
    """v1 samples have experiment_id FK. Pick the earliest one or None."""
    row = v1_conn.execute(
        "SELECT id FROM samples WHERE experiment_id = ? ORDER BY start_time LIMIT 1",
        (exp_id,),
    ).fetchone()
    if row:
        return row["id"]
    # Fall back: maybe v1 actions reference a sample_id directly.
    row = v1_conn.execute(
        "SELECT DISTINCT sample_id FROM actions "
        "WHERE experiment_id = ? AND sample_id != '' LIMIT 1",
        (exp_id,),
    ).fetchone()
    return row["sample_id"] if row else None


def _extract_observation_scalars(row: dict) -> list[tuple[str, float, str]]:
    """Pure extractor: pull (observable, scalar_value, units) tuples out of a v1
    action's ``state_after`` JSON. No DB access — the caller writes them inside
    the per-action transaction so action + observations commit atomically."""
    out: list[tuple[str, float, str]] = []
    state_after = row.get("state_after") if isinstance(row, dict) else row["state_after"]
    if not state_after:
        return out
    try:
        state = json.loads(state_after)
    except Exception:
        return out
    if not isinstance(state, dict):
        return out
    for attr, observable, units in (
        ("bias_v", "bias", "V"),
        ("current_a", "current", "A"),
        ("z_pos_m", "z_position", "m"),
        ("setpoint_a", "setpoint", "A"),
    ):
        val = state.get(attr)
        if val is None:
            continue
        try:
            out.append((observable, float(val), units))
        except (TypeError, ValueError):
            continue
    return out


def _migrate_one_action_atomic(
    store: ExperimentStoreV2,
    row: dict,
    v2_exp: str,
    clock: HLCClock,
) -> tuple[str, int]:
    """Write one v1 action + its observations + its id-mapping row in a SINGLE
    transaction (all-or-nothing).

    The action is inserted with its TERMINAL status directly (succeeded/failed
    derived from the v1 ``result``), so no later UPDATE is needed — the
    append-only status triggers only police UPDATEs, not the initial INSERT, so
    a direct terminal-status insert is valid. The ``_v1_to_v2_map`` row is
    inserted in the same transaction, so on commit the action is atomically both
    persisted AND marked migrated. A crash before commit rolls everything back,
    leaving no orphan action for the next run to duplicate.

    Returns ``(v2_action_id, n_observations)``.
    """
    aid = ulid_now()
    params = json.loads(row.get("parameters") or "{}")
    agent_id = row.get("approval_source") or "unknown"
    if agent_id == "llm":
        agent_id = "agent:legacy_v1"
    action_type = row.get("skill_name") or "unknown"
    prompt_id = row.get("context") or None

    # Derive terminal status + duration from the v1 result blob.
    result = json.loads(row.get("result") or "null") if row.get("result") else None
    status = "succeeded"
    error = None
    duration_ms = None
    if isinstance(result, dict):
        if result.get("success") is False:
            status = "failed"
            error = result.get("error") or "v1 failure"
        else:
            duration_ms = int(float(result.get("elapsed_s") or 0) * 1000)

    obs = _extract_observation_scalars(row)

    # One transaction for the whole unit.
    conn = sqlite3.connect(str(store.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        action_hlc = clock.now().encode()
        conn.execute(
            "INSERT INTO actions (id, experiment_id, agent_id, action_type, prompt_id, "
            "params_json, hlc, duration_ms, status, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (aid, v2_exp, agent_id, action_type, prompt_id,
             json.dumps(params, default=str, ensure_ascii=False),
             action_hlc, duration_ms, status, error),
        )
        for observable, value, units in obs:
            conn.execute(
                "INSERT INTO observations (id, action_id, experiment_id, observable, "
                "hlc, scalar_value, units) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ulid_now(), aid, v2_exp, observable,
                 clock.now().encode(), value, units),
            )
        conn.execute(
            "INSERT OR IGNORE INTO _v1_to_v2_map(kind, v1_id, v2_id) VALUES (?, ?, ?)",
            ("action", row["id"], aid),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    return aid, len(obs)
