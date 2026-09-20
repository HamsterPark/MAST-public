"""Repository layer over the v2 logging schema.

One class per logical entity. All write methods are idempotent at the call
site (caller supplies the ULID) and never UPDATE fact tables — completion
fields like ``actions.status`` are moved through whitelisted column-level
triggers.

All HLC timestamps are accepted as strings (so callers can pre-generate
HLCs via a single shared clock); helpers default to "now" via the supplied
``HLCClock`` instance.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from mast.logging.v2.hlc import HLC, HLCClock
from mast.logging.v2.storage import ExperimentStoreV2
from mast.logging.v2.ulid import ulid_now

logger = logging.getLogger(__name__)


# ── Small helpers ─────────────────────────────────────────────────────

def _utc_now() -> str:
    """ISO-8601 UTC 时间戳（秒精度）。file_locations 用它做 ingested_at/verified_at。"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jdump(obj: Any) -> str:
    if obj is None:
        return "{}"
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, default=str, ensure_ascii=False)


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def _hlc_str(hlc: HLC | str | None, clock: HLCClock | None) -> str:
    if hlc is None:
        if clock is None:
            raise ValueError("Either hlc or clock must be supplied")
        return clock.now().encode()
    if isinstance(hlc, HLC):
        return hlc.encode()
    return hlc


# ── Base class with shared connection plumbing ────────────────────────

class _BaseRepo:
    def __init__(self, store: ExperimentStoreV2, clock: HLCClock | None = None):
        self.store = store
        self.clock = clock

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self.store.connect() as conn:
            yield conn

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._conn() as conn:
            conn.execute(sql, params)

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._conn() as conn:
            return _row_to_dict(conn.execute(sql, params).fetchone())

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ── Campaign ──────────────────────────────────────────────────────────

#: The two closed sets the ``campaigns`` DDL enforces with CHECK constraints.
#: Spelled here as well so a CALLER can reject an off-enum value with a sentence
#: instead of letting SQLite raise ``IntegrityError`` — the row is refused either
#: way (**拒绝，不夹紧**), the difference is only whether the refusal says why.
#: The DDL stays the authority: these tuples must mirror ``schema.DDL_CAMPAIGNS``.
HYPOTHESIS_KINDS: tuple[str, ...] = (
    "exploratory", "confirmatory", "calibration", "methodology")
CAMPAIGN_STATUSES: tuple[str, ...] = (
    "draft", "running", "paused", "completed", "aborted")


class CampaignRepo(_BaseRepo):
    def create(
        self,
        *,
        title: str,
        hypothesis: str,
        hypothesis_kind: str = "exploratory",
        goal: dict | str | None = None,
        created_by: str = "system",
        parent_campaign_id: str | None = None,
    ) -> str:
        cid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO campaigns (id, title, hypothesis, hypothesis_kind, goal_json, "
            "status, created_at, created_by, parent_campaign_id) "
            "VALUES (?, ?, ?, ?, ?, 'draft', ?, ?, ?)",
            (cid, title, hypothesis, hypothesis_kind, _jdump(goal or {}),
             now, created_by, parent_campaign_id),
        )
        return cid

    def set_status(self, campaign_id: str, status: str) -> None:
        self._exec("UPDATE campaigns SET status = ? WHERE id = ?", (status, campaign_id))

    def update(
        self,
        campaign_id: str,
        *,
        title: str | None = None,
        hypothesis: str | None = None,
        hypothesis_kind: str | None = None,
        goal: dict | str | None = None,
    ) -> bool:
        """Revise a campaign in place. ``None`` = leave that column alone.

        Returns True when a row was actually changed, False when the id names no
        campaign — the caller must be able to tell "updated" from "there was
        nothing there", because a silent no-op is how a revision goes missing.

        Why a campaign is MUTABLE while the fact tables below are not: a campaign
        is a **hypothesis**, and a research programme that cannot revise its own
        hypothesis is not a research programme. Actions / observations / scans
        record what HAPPENED and are append-only (see the trigger packs in
        ``schema.py``); this row records what we are currently trying to find out.
        Its history is not lost either — every change goes through an agent turn
        or an operator action, both of which are recorded elsewhere.

        No status knob on purpose: ``set_status`` already owns that transition and
        two writers for one column is how the two disagree about the state machine.
        """
        sets: list[str] = []
        params: list[Any] = []
        if title is not None:
            sets.append("title = ?")
            params.append(title)
        if hypothesis is not None:
            sets.append("hypothesis = ?")
            params.append(hypothesis)
        if hypothesis_kind is not None:
            # Mirrors the DDL CHECK. Refused here so the caller gets a name it can
            # act on rather than a bare IntegrityError from three layers down.
            if hypothesis_kind not in HYPOTHESIS_KINDS:
                raise ValueError(
                    f"hypothesis_kind must be one of {HYPOTHESIS_KINDS}, "
                    f"got {hypothesis_kind!r}")
            sets.append("hypothesis_kind = ?")
            params.append(hypothesis_kind)
        if goal is not None:
            sets.append("goal_json = ?")
            params.append(_jdump(goal))
        if not sets:
            return False
        params.append(campaign_id)
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE campaigns SET {', '.join(sets)} WHERE id = ?", tuple(params))
            return cur.rowcount > 0

    def get(self, campaign_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM campaigns WHERE id = ?", (campaign_id,))

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict]:
        if status:
            return self._fetchall(
                "SELECT * FROM campaigns WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        return self._fetchall(
            "SELECT * FROM campaigns ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    def with_stats(self, limit: int = 100) -> list[dict]:
        return self._fetchall(
            "SELECT c.*, "
            "  COALESCE(m.experiment_count, 0) AS experiment_count, "
            "  COALESCE(m.action_count, 0) AS action_count, "
            "  COALESCE(m.observation_count, 0) AS observation_count, "
            "  COALESCE(m.scan_file_count, 0) AS scan_file_count, "
            "  m.last_activity_hlc "
            "FROM campaigns c LEFT JOIN mv_campaign_stats m ON c.id = m.campaign_id "
            "ORDER BY c.created_at DESC LIMIT ?",
            (limit,),
        )


# ── Sample ────────────────────────────────────────────────────────────

class SampleRepo(_BaseRepo):
    def create(
        self,
        *,
        label: str,
        material: str,
        prep_method: str | None = None,
        prep_log: dict | None = None,
    ) -> str:
        sid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO samples (id, label, material, prep_method, prep_log_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, label, material, prep_method,
             _jdump(prep_log) if prep_log else None, now),
        )
        return sid

    def retire(self, sample_id: str) -> None:
        now = _hlc_str(None, self.clock)
        self._exec("UPDATE samples SET retired_at = ? WHERE id = ?", (now, sample_id))

    def get(self, sample_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM samples WHERE id = ?", (sample_id,))

    def list_active(self, limit: int = 200) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM samples WHERE retired_at IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )


# ── Plan ──────────────────────────────────────────────────────────────

class PlanRepo(_BaseRepo):
    def create(
        self,
        *,
        plan_kind: str,
        title: str,
        definition: dict,
        campaign_id: str | None = None,
        experiment_id: str | None = None,
        hypothesis: str | None = None,
        success_criteria: dict | None = None,
        created_by: str = "system",
    ) -> str:
        pid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO plans (id, campaign_id, experiment_id, plan_kind, title, "
            "definition_json, hypothesis, success_criteria_json, status, created_at, created_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)",
            (pid, campaign_id, experiment_id, plan_kind, title,
             _jdump(definition), hypothesis,
             _jdump(success_criteria) if success_criteria else None,
             now, created_by),
        )
        return pid

    def activate(self, plan_id: str) -> None:
        self._exec("UPDATE plans SET status = 'active' WHERE id = ?", (plan_id,))

    def complete(self, plan_id: str) -> None:
        self._exec("UPDATE plans SET status = 'completed' WHERE id = ?", (plan_id,))

    def get(self, plan_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM plans WHERE id = ?", (plan_id,))

    def for_experiment(self, experiment_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM plans WHERE experiment_id = ? ORDER BY created_at",
            (experiment_id,),
        )


# ── Experiment ────────────────────────────────────────────────────────

class ExperimentRepo(_BaseRepo):
    def start(
        self,
        *,
        campaign_id: str,
        sample_id: str,
        title: str,
        exp_type: str,
        plan_id: str | None = None,
        instrument_state_id: str | None = None,
    ) -> str:
        eid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO experiments (id, campaign_id, sample_id, plan_id, title, "
            "exp_type, instrument_state_id, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (eid, campaign_id, sample_id, plan_id, title, exp_type,
             instrument_state_id, now),
        )
        return eid

    def end(
        self,
        experiment_id: str,
        *,
        exit_status: str = "success",
        conclusion: str | None = None,
        evidence_ids: list[str] | None = None,
    ) -> None:
        now = _hlc_str(None, self.clock)
        self._exec(
            "UPDATE experiments SET ended_at = ?, exit_status = ?, conclusion = ?, "
            "conclusion_evidence_ids = ? WHERE id = ?",
            (now, exit_status, conclusion,
             _jdump(evidence_ids) if evidence_ids else None,
             experiment_id),
        )

    def get(self, experiment_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM experiments WHERE id = ?", (experiment_id,))

    def list(self, *, campaign_id: str | None = None, limit: int = 200) -> list[dict]:
        if campaign_id:
            return self._fetchall(
                "SELECT * FROM experiments WHERE campaign_id = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (campaign_id, limit),
            )
        return self._fetchall(
            "SELECT * FROM experiments ORDER BY started_at DESC LIMIT ?", (limit,)
        )

    def list_open(self) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM experiments WHERE ended_at IS NULL ORDER BY started_at"
        )

    def with_counts(self, *, campaign_id: str | None = None, limit: int = 200) -> list[dict]:
        params: tuple = (limit,)
        where = ""
        if campaign_id:
            where = "WHERE e.campaign_id = ?"
            params = (campaign_id, limit)
        return self._fetchall(
            f"SELECT e.*, "
            f"  (SELECT COUNT(*) FROM actions a WHERE a.experiment_id = e.id) AS action_count, "
            f"  (SELECT COUNT(*) FROM observations o WHERE o.experiment_id = e.id) AS observation_count, "
            f"  (SELECT s.label FROM samples s WHERE s.id = e.sample_id) AS sample_label "
            f"FROM experiments e {where} "
            f"ORDER BY e.started_at DESC LIMIT ?",
            params,
        )


# ── InstrumentState ───────────────────────────────────────────────────

class InstrumentStateRepo(_BaseRepo):
    def snapshot(
        self,
        *,
        experiment_id: str | None,
        state: dict,
        reason: str = "periodic",
    ) -> str:
        sid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO instrument_states (id, experiment_id, hlc, state_json, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, experiment_id, now, _jdump(state), reason),
        )
        return sid

    def latest(self, experiment_id: str) -> dict | None:
        return self._fetchone(
            "SELECT * FROM instrument_states WHERE experiment_id = ? "
            "ORDER BY hlc DESC LIMIT 1",
            (experiment_id,),
        )


# ── Action ────────────────────────────────────────────────────────────

class ActionRepo(_BaseRepo):
    def begin(
        self,
        *,
        experiment_id: str,
        agent_id: str,
        action_type: str,
        params: dict | None = None,
        parent_action_id: str | None = None,
        caused_by_event_id: str | None = None,
        prompt_id: str | None = None,
        thread_id: str | None = None,
        tool_call_id: str | None = None,
        instrument_state_id: str | None = None,
        state_delta: dict | None = None,
        hlc: HLC | str | None = None,
        action_id: str | None = None,
    ) -> str:
        """Insert an action row in 'pending' state. Returns the action_id.

        ``action_id`` may be supplied so dangerous actions can pre-issue an
        approval row (see ApprovalService.issue) before the action is inserted.
        """
        aid = action_id or ulid_now()
        now = _hlc_str(hlc, self.clock)
        self._exec(
            "INSERT INTO actions (id, experiment_id, parent_action_id, caused_by_event_id, "
            "agent_id, action_type, prompt_id, thread_id, tool_call_id, params_json, "
            "instrument_state_id, state_delta_json, hlc, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (aid, experiment_id, parent_action_id, caused_by_event_id,
             agent_id, action_type, prompt_id, thread_id, tool_call_id,
             _jdump(params or {}), instrument_state_id,
             _jdump(state_delta) if state_delta else None, now),
        )
        return aid

    def mark_running(self, action_id: str) -> None:
        self._exec("UPDATE actions SET status = 'running' WHERE id = ? "
                   "AND status = 'pending'", (action_id,))

    def succeed(self, action_id: str, *, duration_ms: int | None = None) -> None:
        self._exec(
            "UPDATE actions SET status = 'succeeded', duration_ms = COALESCE(?, duration_ms) "
            "WHERE id = ? AND status IN ('pending','running')",
            (duration_ms, action_id),
        )

    def fail(self, action_id: str, error: str, *, duration_ms: int | None = None) -> None:
        self._exec(
            "UPDATE actions SET status = 'failed', error = ?, "
            "duration_ms = COALESCE(?, duration_ms) "
            "WHERE id = ? AND status IN ('pending','running')",
            (error, duration_ms, action_id),
        )

    def rollback(self, action_id: str) -> None:
        self._exec(
            "UPDATE actions SET status = 'rolled_back' "
            "WHERE id = ? AND status IN ('pending','running')",
            (action_id,),
        )

    def retract(
        self,
        action_id: str,
        *,
        retracted_by: str,
        rationale: str,
        experiment_id: str,
    ) -> str:
        """Add a compensating 'retract' action that points to the original.

        Per compass §3.1 — append-only. Original action keeps its status
        chain; UI hides it via the retracted_by_action_id relation.
        """
        # Mark the original retracted.
        self._exec(
            "UPDATE actions SET status = 'retracted' "
            "WHERE id = ? AND status NOT IN ('rolled_back','retracted')",
            (action_id,),
        )
        # Append the compensating action.
        return self.begin(
            experiment_id=experiment_id,
            agent_id=retracted_by,
            action_type="retract",
            params={"target_action_id": action_id, "rationale": rationale},
            parent_action_id=action_id,
        )

    def get(self, action_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM actions WHERE id = ?", (action_id,))

    def for_experiment(
        self,
        experiment_id: str,
        *,
        agent_id: str | None = None,
        action_type: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        clauses = ["experiment_id = ?"]
        params: list = [experiment_id]
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if action_type:
            clauses.append("action_type = ?")
            params.append(action_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        params.append(limit)
        return self._fetchall(
            f"SELECT * FROM actions WHERE {' AND '.join(clauses)} "
            f"ORDER BY hlc LIMIT ?",
            tuple(params),
        )

    def ancestry(self, action_id: str, *, max_depth: int = 50) -> list[dict]:
        """Walk parent_action_id backwards up to *max_depth*. Root first."""
        return self._fetchall(
            "WITH RECURSIVE chain(id, parent_action_id, depth) AS ("
            "  SELECT id, parent_action_id, 0 FROM actions WHERE id = ?"
            "  UNION ALL"
            "  SELECT a.id, a.parent_action_id, c.depth + 1 "
            "  FROM actions a JOIN chain c ON a.id = c.parent_action_id "
            "  WHERE c.depth < ?"
            ") "
            "SELECT a.* FROM chain c JOIN actions a USING (id) "
            "ORDER BY c.depth DESC",
            (action_id, max_depth),
        )

    def descendants(self, action_id: str, *, max_depth: int = 50) -> list[dict]:
        return self._fetchall(
            "WITH RECURSIVE chain(id, parent_action_id, depth) AS ("
            "  SELECT id, parent_action_id, 0 FROM actions WHERE id = ?"
            "  UNION ALL"
            "  SELECT a.id, a.parent_action_id, c.depth + 1 "
            "  FROM actions a JOIN chain c ON a.parent_action_id = c.id "
            "  WHERE c.depth < ?"
            ") "
            "SELECT a.* FROM chain c JOIN actions a USING (id) "
            "ORDER BY a.hlc",
            (action_id, max_depth),
        )


# ── ScanFile ──────────────────────────────────────────────────────────

class ScanFileRepo(_BaseRepo):
    def register(
        self,
        *,
        produced_by_action_id: str,
        sha256: str,
        size_bytes: int,
        current_path: str,
        format_kind: str,
        parser_spec: str,
        mime_type: str | None = None,
        parser_kwargs: dict | None = None,
        meta: dict | None = None,
        hlc: HLC | str | None = None,
    ) -> str:
        sfid = ulid_now()
        now = _hlc_str(hlc, self.clock)
        mime = mime_type or _mime_for_format(format_kind)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM scan_files WHERE sha256 = ?", (sha256,)
            ).fetchone()
            if existing:
                return existing["id"]
            conn.execute(
                "INSERT INTO scan_files (id, sha256, size_bytes, mime_type, format_kind, "
                "current_path, parser_spec, parser_kwargs_json, produced_by_action_id, "
                "meta_json, hlc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sfid, sha256, size_bytes, mime, format_kind, current_path,
                 parser_spec, _jdump(parser_kwargs) if parser_kwargs else None,
                 produced_by_action_id, _jdump(meta or {}), now),
            )
        return sfid

    def for_action(self, action_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM scan_files WHERE produced_by_action_id = ? ORDER BY hlc",
            (action_id,),
        )

    def by_sha(self, sha256: str) -> dict | None:
        return self._fetchone("SELECT * FROM scan_files WHERE sha256 = ?", (sha256,))

    def get(self, scan_file_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM scan_files WHERE id = ?", (scan_file_id,))


    def list_with_fixity(self, *, format_kind: str | None = None, limit: int = 500) -> list[dict]:
        """All scan files, optionally filtered by format, newest first."""
        if format_kind:
            return self._fetchall(
                "SELECT * FROM scan_files WHERE format_kind = ? ORDER BY hlc DESC LIMIT ?",
                (format_kind, limit),
            )
        return self._fetchall(
            "SELECT * FROM scan_files ORDER BY hlc DESC LIMIT ?", (limit,)
        )

    def list_dedup_groups(self) -> list[dict]:
        """Return sha256 groups with > 1 row (CAS dedup, compass §2.6).

        Each group: {sha256, file_ids: [...], count}.
        """
        rows = self._fetchall(
            "SELECT sha256, COUNT(*) AS n, GROUP_CONCAT(id) AS ids "
            "FROM scan_files GROUP BY sha256 HAVING n > 1 ORDER BY n DESC"
        )
        return [
            {"sha256": r["sha256"], "count": r["n"],
             "file_ids": (r["ids"] or "").split(",")}
            for r in rows
        ]

    def observation_count(self, scan_file_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) AS c FROM observations WHERE scan_file_id = ?",
                (scan_file_id,),
            ).fetchone()["c"]


def _mime_for_format(fmt: str) -> str:
    return {
        "sxm": "application/x-nanonis-sxm",
        "dat": "application/x-nanonis-dat",
        "3ds": "application/x-nanonis-3ds",
        "h5": "application/x-hdf5",
        "png": "image/png",
        "npy": "application/x-numpy",
        "parquet": "application/x-parquet",
        "tiff": "image/tiff",
    }.get(fmt, "application/octet-stream")


# ── Observation ───────────────────────────────────────────────────────

class FileLocationRepo(_BaseRepo):
    """"这份字节现在躺在哪" —— 唯一一张可变的文件事实表。

    与 :class:`ScanFileRepo` 分工：``scan_files`` 是 append-only 的全局内容索引
    （一份内容一行，由哪个 action 产生，摘要多少）；``file_locations`` 记的是
    位置，而位置是多对一且会变的（复制失败后重试成功要能改回 ok，同一份字节
    可以同时在实验文件夹和隔离区）。

    设计文档：docs/v2/design/experiment_folder_persistence.md §6
    """

    def record(
        self, *,
        sha256: str,
        experiment_id: str,
        rel_path: str,
        source: str,
        sample_id: str | None = None,
        root_kind: str = "experiment_folder",
        origin_path: str | None = None,
        action_id: str | None = None,
        size_bytes: int = 0,
        status: str = "ok",
    ) -> None:
        """登记一处位置。同 (sha, experiment, rel_path) 重复调用是幂等的 upsert。

        ``action_id`` 若指向一条不存在的 action 会触发外键错误 —— 调用方
        （ingest sink）已经把整个调用包在 try 里，DB 拒绝不影响文件已经落好。
        """
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO file_locations (sha256, experiment_id, sample_id, rel_path, "
                "root_kind, origin_path, source, action_id, size_bytes, ingested_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(sha256, experiment_id, rel_path) DO UPDATE SET "
                "  status = excluded.status, "
                "  root_kind = excluded.root_kind, "
                "  size_bytes = MAX(file_locations.size_bytes, excluded.size_bytes), "
                "  origin_path = COALESCE(excluded.origin_path, file_locations.origin_path), "
                "  action_id = COALESCE(file_locations.action_id, excluded.action_id)",
                (sha256, experiment_id, sample_id, rel_path, root_kind, origin_path,
                 source, action_id, int(size_bytes or 0), now, status),
            )

    def mark(self, sha256: str, experiment_id: str, rel_path: str, *,
             status: str, verified: bool = False) -> None:
        """改状态（ok / missing / copy_failed / quarantined）。"""
        with self._conn() as conn:
            if verified:
                conn.execute(
                    "UPDATE file_locations SET status = ?, verified_at = ? "
                    "WHERE sha256 = ? AND experiment_id = ? AND rel_path = ?",
                    (status, _utc_now(), sha256, experiment_id, rel_path),
                )
            else:
                conn.execute(
                    "UPDATE file_locations SET status = ? "
                    "WHERE sha256 = ? AND experiment_id = ? AND rel_path = ?",
                    (status, sha256, experiment_id, rel_path),
                )

    def by_sha(self, sha256: str, experiment_id: str | None = None) -> list[dict]:
        if experiment_id:
            return self._fetchall(
                "SELECT * FROM file_locations WHERE sha256 = ? AND experiment_id = ? "
                "ORDER BY ingested_at",
                (sha256, experiment_id),
            )
        return self._fetchall(
            "SELECT * FROM file_locations WHERE sha256 = ? ORDER BY ingested_at", (sha256,))

    def for_experiment(self, experiment_id: str, *, limit: int = 500) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM file_locations WHERE experiment_id = ? "
            "ORDER BY ingested_at DESC LIMIT ?",
            (experiment_id, int(limit)),
        )

    def for_sample(self, sample_id: str, *, limit: int = 500) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM file_locations WHERE sample_id = ? "
            "ORDER BY ingested_at DESC LIMIT ?",
            (sample_id, int(limit)),
        )

    def counts(self, experiment_id: str) -> dict:
        row = self._fetchone(
            "SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes), 0) AS bytes "
            "FROM file_locations WHERE experiment_id = ? AND status = 'ok'",
            (experiment_id,),
        )
        return {"files": int((row or {}).get("n") or 0),
                "bytes": int((row or {}).get("bytes") or 0)}

    def fixity_failed(self) -> list[dict]:
        return self._fetchall("SELECT * FROM scan_files WHERE fixity_ok = 0 ORDER BY hlc DESC")

class ObservationRepo(_BaseRepo):
    def record_scalar(
        self,
        *,
        action_id: str,
        experiment_id: str,
        observable: str,
        scalar_value: float,
        units: str | None = None,
        channel: str | None = None,
        hlc: HLC | str | None = None,
    ) -> str:
        oid = ulid_now()
        now = _hlc_str(hlc, self.clock)
        self._exec(
            "INSERT INTO observations (id, action_id, experiment_id, observable, channel, "
            "hlc, scalar_value, units) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (oid, action_id, experiment_id, observable, channel, now,
             float(scalar_value), units),
        )
        return oid

    def record_scan(
        self,
        *,
        action_id: str,
        experiment_id: str,
        observable: str,
        scan_file_id: str,
        channel: str | None = None,
        result_summary: dict | None = None,
        hlc: HLC | str | None = None,
    ) -> str:
        oid = ulid_now()
        now = _hlc_str(hlc, self.clock)
        summary = _jdump(result_summary) if result_summary else None
        if summary and len(summary) > 4096:
            raise ValueError("result_summary_json exceeds 4 KiB; store large data in scan_files")
        self._exec(
            "INSERT INTO observations (id, action_id, experiment_id, observable, channel, "
            "hlc, scan_file_id, result_summary_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (oid, action_id, experiment_id, observable, channel, now,
             scan_file_id, summary),
        )
        return oid

    def record_summary(
        self,
        *,
        action_id: str,
        experiment_id: str,
        observable: str,
        result_summary: dict,
        channel: str | None = None,
        hlc: HLC | str | None = None,
    ) -> str:
        oid = ulid_now()
        now = _hlc_str(hlc, self.clock)
        summary = _jdump(result_summary)
        if len(summary) > 4096:
            raise ValueError("result_summary_json exceeds 4 KiB")
        self._exec(
            "INSERT INTO observations (id, action_id, experiment_id, observable, channel, "
            "hlc, result_summary_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (oid, action_id, experiment_id, observable, channel, now, summary),
        )
        return oid

    def for_action(self, action_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM observations WHERE action_id = ? ORDER BY hlc",
            (action_id,),
        )

    def for_experiment(
        self,
        experiment_id: str,
        *,
        observable: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        if observable:
            return self._fetchall(
                "SELECT * FROM observations WHERE experiment_id = ? AND observable = ? "
                "ORDER BY hlc LIMIT ?",
                (experiment_id, observable, limit),
            )
        return self._fetchall(
            "SELECT * FROM observations WHERE experiment_id = ? ORDER BY hlc LIMIT ?",
            (experiment_id, limit),
        )


# ── Event ─────────────────────────────────────────────────────────────

class EventRepo(_BaseRepo):
    def publish(
        self,
        *,
        topic: str,
        kind: str,
        payload: dict | None = None,
        severity: str = "info",
        experiment_id: str | None = None,
        action_id: str | None = None,
        producer: str | None = None,
        dedup_key: str | None = None,
        hlc: HLC | str | None = None,
    ) -> str | None:
        """Insert an event. For ``kind='snapshot'`` with a ``dedup_key`` already
        present in the last write on the same topic, the event is dropped
        (returns ``None``) — this implements the "equal-value merge" rule from
        compass §2.2.7.
        """
        now = _hlc_str(hlc, self.clock)
        if kind == "snapshot" and dedup_key is not None:
            last = self._fetchone(
                "SELECT dedup_key FROM events WHERE topic = ? "
                "ORDER BY hlc DESC LIMIT 1",
                (topic,),
            )
            if last and last.get("dedup_key") == dedup_key:
                return None
        eid = ulid_now()
        self._exec(
            "INSERT INTO events (id, topic, kind, severity, hlc, experiment_id, "
            "action_id, producer, payload_json, dedup_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (eid, topic, kind, severity, now, experiment_id, action_id,
             producer, _jdump(payload or {}), dedup_key),
        )
        return eid

    def for_experiment(self, experiment_id: str, limit: int = 1000) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM events WHERE experiment_id = ? ORDER BY hlc LIMIT ?",
            (experiment_id, limit),
        )

    def by_topic(self, topic: str, *, limit: int = 1000) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM events WHERE topic = ? ORDER BY hlc DESC LIMIT ?",
            (topic, limit),
        )


# ── Approval ──────────────────────────────────────────────────────────

class ApprovalService(_BaseRepo):
    def issue(
        self,
        *,
        action_id: str,
        approver_id: str,
        approver_kind: str = "human_operator",
        approval_method: str = "gui_click",
        approval_evidence: str | None = None,
        policy_version: str | None = None,
        expires_at: str | None = None,
        hlc: HLC | str | None = None,
    ) -> str:
        """Issue an approval row BEFORE the action insert. The caller must wrap
        the (approval insert + action insert) pair in a transaction with
        deferred FK checks (see ``deferred_fk`` context manager below).
        """
        aid = ulid_now()
        now = _hlc_str(hlc, self.clock)
        self._exec(
            "INSERT INTO approvals (id, action_id, approver_id, approver_kind, "
            "approval_method, approval_evidence, approved_at, expires_at, policy_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (aid, action_id, approver_id, approver_kind, approval_method,
             approval_evidence, now, expires_at, policy_version),
        )
        return aid

    def issue_and_begin(
        self,
        *,
        # — action fields (mirror ActionRepo.begin) —
        experiment_id: str,
        agent_id: str,
        action_type: str,
        params: dict | None = None,
        parent_action_id: str | None = None,
        caused_by_event_id: str | None = None,
        prompt_id: str | None = None,
        thread_id: str | None = None,
        tool_call_id: str | None = None,
        instrument_state_id: str | None = None,
        state_delta: dict | None = None,
        action_hlc: HLC | str | None = None,
        action_id: str | None = None,
        # — approval fields (mirror ApprovalService.issue) —
        approver_id: str,
        approver_kind: str = "human_operator",
        approval_method: str = "gui_click",
        approval_evidence: str | None = None,
        policy_version: str | None = None,
        expires_at: str | None = None,
        approval_hlc: HLC | str | None = None,
    ) -> tuple[str, str]:
        """Atomically issue an approval row and insert its dangerous action.

        Closes the FK race described in compass §3.6 / finding #79: the
        ``approvals.action_id`` FK requires the action to exist, while
        ``trg_action_requires_approval`` requires the approval row to exist
        before the dangerous action is inserted. Doing the two inserts via
        two independent connections (``ApprovalService.issue`` then
        ``ActionRepo.begin``) leaves a window where a concurrent reader sees
        an approval pointing at a not-yet-committed action, or fails the
        trigger entirely. This method runs BOTH inserts inside one
        ``deferred_fk`` transaction so they commit together or not at all.

        Returns ``(action_id, approval_id)``.
        """
        aid = action_id or ulid_now()
        approval_id = ulid_now()
        a_now = _hlc_str(action_hlc, self.clock)
        approval_now = _hlc_str(approval_hlc, self.clock)
        with deferred_fk(self.store) as conn:
            # Approval first so the trigger's NOT EXISTS check passes.
            conn.execute(
                "INSERT INTO approvals (id, action_id, approver_id, approver_kind, "
                "approval_method, approval_evidence, approved_at, expires_at, policy_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (approval_id, aid, approver_id, approver_kind, approval_method,
                 approval_evidence, approval_now, expires_at, policy_version),
            )
            # Then the action, with deferred FK so it may reference the
            # approval's not-yet-resolved action_id within the same tx.
            conn.execute(
                "INSERT INTO actions (id, experiment_id, parent_action_id, "
                "caused_by_event_id, agent_id, action_type, prompt_id, thread_id, "
                "tool_call_id, params_json, instrument_state_id, state_delta_json, "
                "hlc, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                (aid, experiment_id, parent_action_id, caused_by_event_id,
                 agent_id, action_type, prompt_id, thread_id, tool_call_id,
                 _jdump(params or {}), instrument_state_id,
                 _jdump(state_delta) if state_delta else None, a_now),
            )
        return aid, approval_id

    def for_action(self, action_id: str) -> dict | None:
        return self._fetchone(
            "SELECT * FROM approvals WHERE action_id = ?", (action_id,)
        )

    def by_approver(self, approver_id: str, *, limit: int = 100) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM approvals WHERE approver_id = ? ORDER BY approved_at DESC LIMIT ?",
            (approver_id, limit),
        )


@contextmanager
def deferred_fk(store: ExperimentStoreV2) -> Iterator[sqlite3.Connection]:
    """Run a (approval, action) write pair in one transaction with deferred FKs.

    Usage:
        with deferred_fk(store) as conn:
            conn.execute("INSERT INTO approvals ...", (...))
            conn.execute("INSERT INTO actions   ...", (...))
    """
    conn = sqlite3.connect(str(store.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN")
        conn.execute("PRAGMA defer_foreign_keys = 1")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Claim graph ───────────────────────────────────────────────────────

class ClaimGraphRepo(_BaseRepo):
    def create_claim(
        self,
        *,
        statement: str,
        experiment_id: str | None = None,
        campaign_id: str | None = None,
        confidence: float | None = None,
        created_by: str = "system",
    ) -> str:
        cid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO claims (id, experiment_id, campaign_id, statement, confidence, "
            "status, created_at, created_by) VALUES (?, ?, ?, ?, ?, 'proposed', ?, ?)",
            (cid, experiment_id, campaign_id, statement, confidence, now, created_by),
        )
        return cid

    def set_status(self, claim_id: str, status: str) -> None:
        self._exec("UPDATE claims SET status = ? WHERE id = ?", (status, claim_id))

    def add_edge(
        self,
        *,
        claim_id: str,
        target_kind: str,
        target_id: str,
        edge_type: str = "mast:supports",
        weight: float | None = None,
        rationale: str | None = None,
        created_by: str = "system",
    ) -> str:
        ref_id = self._upsert_ref(target_kind, target_id)
        eid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO evidence_edges (id, claim_id, edge_type, target_ref_id, weight, "
            "rationale, hlc, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (eid, claim_id, edge_type, ref_id, weight, rationale, now, created_by),
        )
        return eid

    def _upsert_ref(
        self,
        target_kind: str,
        target_id: str,
        external_url: str | None = None,
    ) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM entity_refs WHERE entity_kind = ? AND entity_id = ? "
                "AND COALESCE(external_url,'') = COALESCE(?, '')",
                (target_kind, target_id, external_url),
            ).fetchone()
            if row:
                return row["id"]
            rid = ulid_now()
            conn.execute(
                "INSERT INTO entity_refs (id, entity_kind, entity_id, external_url) "
                "VALUES (?, ?, ?, ?)",
                (rid, target_kind, target_id, external_url),
            )
            return rid

    def get_claim(self, claim_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM claims WHERE id = ?", (claim_id,))

    def edges_for(self, claim_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT e.*, r.entity_kind, r.entity_id, r.external_url "
            "FROM evidence_edges e JOIN entity_refs r ON e.target_ref_id = r.id "
            "WHERE e.claim_id = ? ORDER BY e.hlc",
            (claim_id,),
        )

    def evidence_subgraph(self, claim_id: str, *, max_depth: int = 5) -> list[dict]:
        """Recursive walk through claim → claim edges, returning all reachable
        non-claim evidence (observations, scan_files, actions, papers).
        """
        return self._fetchall(
            "WITH RECURSIVE walk(claim_id, target_ref_id, depth) AS ("
            "  SELECT claim_id, target_ref_id, 0 FROM evidence_edges WHERE claim_id = ?"
            "  UNION ALL"
            "  SELECT ee.claim_id, ee.target_ref_id, w.depth + 1 "
            "  FROM evidence_edges ee JOIN entity_refs r ON ee.target_ref_id = r.id "
            "  JOIN walk w ON r.entity_id = ee.claim_id "
            "  WHERE r.entity_kind = 'claim' AND w.depth < ?"
            ") "
            "SELECT DISTINCT r.entity_kind, r.entity_id, r.external_url "
            "FROM walk w JOIN entity_refs r ON w.target_ref_id = r.id "
            "WHERE r.entity_kind != 'claim'",
            (claim_id, max_depth),
        )

    def list_claims(self, *, experiment_id: str | None = None,
                    campaign_id: str | None = None, limit: int = 200) -> list[dict]:
        if experiment_id:
            return self._fetchall(
                "SELECT * FROM claims WHERE experiment_id = ? ORDER BY created_at DESC LIMIT ?",
                (experiment_id, limit),
            )
        if campaign_id:
            return self._fetchall(
                "SELECT * FROM claims WHERE campaign_id = ? ORDER BY created_at DESC LIMIT ?",
                (campaign_id, limit),
            )
        return self._fetchall(
            "SELECT * FROM claims ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    def compose(
        self,
        *,
        statement: str,
        edges: list[dict],
        confidence: float | None = None,
        status: str = "proposed",
        experiment_id: str | None = None,
        campaign_id: str | None = None,
        created_by: str = "operator",
    ) -> str:
        """Atomically write a claim + its entity_refs + evidence_edges (compass §2.10).

        ``edges`` is a list of dicts: ``{kind, id, edge_type, weight?, rationale?}``.
        The whole 3-table write is one transaction — all or nothing.
        Returns the new claim_id.
        """
        if created_by.startswith("agent:"):
            raise ValueError("claims composed by an operator cannot be attributed to agent:*")
        claim_id = ulid_now()
        now = _hlc_str(None, self.clock)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO claims (id, experiment_id, campaign_id, statement, confidence, "
                "status, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (claim_id, experiment_id, campaign_id, statement, confidence,
                 status, now, created_by),
            )
            for edge in edges:
                kind = edge["kind"]
                target_id = edge["id"]
                # upsert entity_ref
                row = conn.execute(
                    "SELECT id FROM entity_refs WHERE entity_kind = ? AND entity_id = ? "
                    "AND COALESCE(external_url,'') = ''",
                    (kind, target_id),
                ).fetchone()
                if row:
                    ref_id = row["id"]
                else:
                    ref_id = ulid_now()
                    conn.execute(
                        "INSERT INTO entity_refs (id, entity_kind, entity_id) VALUES (?, ?, ?)",
                        (ref_id, kind, target_id),
                    )
                conn.execute(
                    "INSERT INTO evidence_edges (id, claim_id, edge_type, target_ref_id, "
                    "weight, rationale, hlc, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ulid_now(), claim_id, edge.get("edge_type", "mast:supports"),
                     ref_id, edge.get("weight"), edge.get("rationale"), now, created_by),
                )
        return claim_id


# ── Audit log ─────────────────────────────────────────────────────────

class AuditLogRepo(_BaseRepo):
    def record(
        self,
        *,
        actor_id: str,
        event: str,
        actor_kind: str = "system",
        payload: dict | None = None,
    ) -> str:
        aid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO audit_log (id, actor_id, actor_kind, event, hlc, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (aid, actor_id, actor_kind, event, now, _jdump(payload or {})),
        )
        return aid

    def by_actor(self, actor_id: str, limit: int = 200) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM audit_log WHERE actor_id = ? ORDER BY hlc DESC LIMIT ?",
            (actor_id, limit),
        )

    def by_event(self, event: str, limit: int = 200) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM audit_log WHERE event = ? ORDER BY hlc DESC LIMIT ?",
            (event, limit),
        )


# ── Review ────────────────────────────────────────────────────────────

class ReviewRepo(_BaseRepo):
    def record(
        self,
        *,
        target_kind: str,
        target_id: str,
        reviewer_id: str,
        reviewer_kind: str,
        verdict: str,
        comments: dict | None = None,
    ) -> str:
        rid = ulid_now()
        now = _hlc_str(None, self.clock)
        self._exec(
            "INSERT INTO reviews (id, target_kind, target_id, reviewer_id, reviewer_kind, "
            "verdict, comments_json, hlc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, target_kind, target_id, reviewer_id, reviewer_kind, verdict,
             _jdump(comments or {}), now),
        )
        return rid

    def for_target(self, target_kind: str, target_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM reviews WHERE target_kind = ? AND target_id = ? "
            "ORDER BY hlc DESC",
            (target_kind, target_id),
        )


# ── Agent training trajectories (RFC docs/v2/design/agent_training_log_rfc.md) ──

# Default per-string cap for sanitized JSON (no-tensor / no-big-blob invariant).
# Reasoning traces (agent_turn steps) are the most valuable training signal and
# routinely run several KB, so the StepRepo widens the cap for that one step type
# (see _AGENT_TURN_STR_CAP) while everything else stays at the conservative bound
# that keeps state snapshots from inlining base64 images / arrays.
_DEFAULT_STR_CAP = 4000
_AGENT_TURN_STR_CAP = 16000


def _json_safe(obj, _depth: int = 0, *, max_str: int = _DEFAULT_STR_CAP):
    """Recursively keep only JSON primitives; replace anything else (ndarray,
    bytes, file handles, arbitrary objects) with a short type marker, and cap
    container/string sizes. Enforces the 'no tensor / big object inlined into
    the store' invariant for trajectory_steps.input/output — these come from
    agent state snapshots that may carry images/arrays.

    ``max_str`` caps each string; callers logging reasoning traces pass a higher
    bound (the default keeps state snapshots compact)."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + "…"
    if _depth >= 6:
        return "<deep>"
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, _depth + 1, max_str=max_str)
                for k, v in list(obj.items())[:200]}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, _depth + 1, max_str=max_str) for v in list(obj)[:200]]
    return f"<non-primitive:{type(obj).__name__}>"


def _jdump_safe(obj, *, max_str: int = _DEFAULT_STR_CAP) -> str | None:
    return None if obj is None else _jdump(_json_safe(obj, max_str=max_str))


class TrajectoryRepo(_BaseRepo):
    """One orchestrator run = one trajectory (RFC §1). ``begin`` at run start,
    ``end`` set-once at run end, ``set_quality`` backfilled from operator
    feedback. All FIRE-AND-FORGET callers must wrap calls so a logging failure
    never propagates into the agent graph."""

    def begin(self, *, thread_id: str, operator_intent: dict | None = None,
              context_snapshot: dict | None = None, experiment_id: str | None = None,
              campaign_id: str | None = None, sample_id: str | None = None,
              hlc: HLC | str | None = None, trajectory_id: str | None = None) -> str:
        tid = trajectory_id or ulid_now()
        now = _hlc_str(hlc, self.clock)
        self._exec(
            "INSERT INTO trajectories (id, thread_id, experiment_id, campaign_id, "
            "sample_id, created_at_hlc, operator_intent_json, context_snapshot_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (tid, thread_id, experiment_id, campaign_id, sample_id, now,
             _jdump(_json_safe(operator_intent or {})),
             _jdump(_json_safe(context_snapshot or {}))),
        )
        return tid

    def end(self, trajectory_id: str, *, exit_status: str | None = None,
            final_outcome: dict | None = None, hlc: HLC | str | None = None) -> None:
        """Set-once via COALESCE (matches the immutable-cols trigger; re-calling
        with a different value is a silent no-op, not an ABORT)."""
        now = _hlc_str(hlc, self.clock)
        self._exec(
            "UPDATE trajectories SET "
            "ended_at_hlc = COALESCE(ended_at_hlc, ?), "
            "exit_status = COALESCE(exit_status, ?), "
            "final_outcome_json = COALESCE(final_outcome_json, ?) "
            "WHERE id = ?",
            (now, exit_status,
             _jdump_safe(final_outcome) if final_outcome is not None else None,
             trajectory_id),
        )

    def set_quality(self, trajectory_id: str, quality: dict) -> None:
        """Backfill operator rating / reward (freely updatable)."""
        self._exec("UPDATE trajectories SET quality_json = ? WHERE id = ?",
                   (_jdump(_json_safe(quality)), trajectory_id))

    def get(self, trajectory_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM trajectories WHERE id = ?",
                              (trajectory_id,))

    def by_thread(self, thread_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM trajectories WHERE thread_id = ? ORDER BY created_at_hlc",
            (thread_id,))

    def list_recent(self, *, limit: int = 10000) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM trajectories ORDER BY created_at_hlc DESC LIMIT ?",
            (int(limit),))


class StepRepo(_BaseRepo):
    """Ordered causal steps of a trajectory (RFC §1). Append-only; input/output
    are sanitized to JSON primitives (no tensors). ``action_id`` /
    ``observation_id`` / ``approval_id`` point back at the fact tables — write
    the fact row first, then the step that references it."""

    def record(self, *, trajectory_id: str, step_type: str,
               parent_step_id: str | None = None, hop_idx: int | None = None,
               actor_kind: str | None = None, agent_id: str | None = None,
               model_id: str | None = None, tool_call_id: str | None = None,
               action_id: str | None = None, observation_id: str | None = None,
               approval_id: str | None = None, input: dict | None = None,
               output: dict | None = None, duration_ms: int | None = None,
               hlc: HLC | str | None = None, step_id: str | None = None) -> str:
        sid = step_id or ulid_now()
        now = _hlc_str(hlc, self.clock)
        # agent_turn carries the reasoning trace (CoT) — the prime SFT signal —
        # which routinely runs multiple KB, so give its output a wider per-string
        # cap. Everything else (state snapshots etc.) stays at the conservative
        # default that keeps base64/array blobs from inlining.
        out_cap = _AGENT_TURN_STR_CAP if step_type == "agent_turn" else _DEFAULT_STR_CAP
        self._exec(
            "INSERT INTO trajectory_steps (id, trajectory_id, parent_step_id, hlc, "
            "hop_idx, step_type, actor_kind, agent_id, model_id, tool_call_id, "
            "action_id, observation_id, approval_id, input_json, output_json, "
            "duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, trajectory_id, parent_step_id, now, hop_idx, step_type,
             actor_kind, agent_id, model_id, tool_call_id, action_id,
             observation_id, approval_id, _jdump_safe(input),
             _jdump_safe(output, max_str=out_cap), duration_ms),
        )
        return sid

    def by_trajectory(self, trajectory_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM trajectory_steps WHERE trajectory_id = ? ORDER BY hlc",
            (trajectory_id,))


# ── Top-level facade ──────────────────────────────────────────────────

@dataclass
class V2Repos:
    """One-stop bag of repositories — convenient for higher-level code."""
    store: ExperimentStoreV2
    clock: HLCClock
    campaigns: CampaignRepo
    samples: SampleRepo
    plans: PlanRepo
    experiments: ExperimentRepo
    instrument_states: InstrumentStateRepo
    actions: ActionRepo
    scan_files: ScanFileRepo
    file_locations: FileLocationRepo
    observations: ObservationRepo
    events: EventRepo
    approvals: ApprovalService
    claims: ClaimGraphRepo
    audit: AuditLogRepo
    reviews: ReviewRepo
    trajectories: TrajectoryRepo
    steps: StepRepo


_FORBIDDEN_SQL = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "begin", "commit",
)


def raw_query(
    store: ExperimentStoreV2,
    sql: str,
    params: tuple = (),
    *,
    max_rows: int = 1000,
) -> list[dict]:
    """Run a read-only SELECT against the v2 store (compass §4 / §6.5).

    Rejects any statement that is not a single SELECT — the Slice-view SQL
    workbench is the consumer. Multiple statements and write keywords are
    refused. Caller must still treat field names as a trusted whitelist.
    """
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        raise ValueError("only a single statement is allowed")
    low = stripped.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("only SELECT / WITH queries are allowed")
    # crude keyword guard — defence in depth on top of the SELECT-only check
    import re
    tokens = set(re.findall(r"[a-z_]+", low))
    bad = tokens & set(_FORBIDDEN_SQL)
    if bad:
        raise ValueError(f"forbidden SQL keyword(s): {sorted(bad)}")
    with store.connect() as conn:
        rows = conn.execute(stripped, params).fetchmany(max_rows)
        return [dict(r) for r in rows]


def build_repos(store: ExperimentStoreV2, clock: HLCClock | None = None) -> V2Repos:
    """Factory that wires every repo against a single store + clock."""
    if clock is None:
        clock = HLCClock(node_id="main")
    return V2Repos(
        store=store,
        clock=clock,
        campaigns=CampaignRepo(store, clock),
        samples=SampleRepo(store, clock),
        plans=PlanRepo(store, clock),
        experiments=ExperimentRepo(store, clock),
        instrument_states=InstrumentStateRepo(store, clock),
        actions=ActionRepo(store, clock),
        scan_files=ScanFileRepo(store, clock),
        file_locations=FileLocationRepo(store, clock),
        observations=ObservationRepo(store, clock),
        events=EventRepo(store, clock),
        approvals=ApprovalService(store, clock),
        claims=ClaimGraphRepo(store, clock),
        audit=AuditLogRepo(store, clock),
        reviews=ReviewRepo(store, clock),
        trajectories=TrajectoryRepo(store, clock),
        steps=StepRepo(store, clock),
    )
