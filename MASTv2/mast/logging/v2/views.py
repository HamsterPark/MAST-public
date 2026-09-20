"""Read-only SQL views over the v2 schema.

These are convenience aggregations the UI / agents query frequently. They
are plain VIEWs (zero cost when not touched) and they always read from the
canonical fact tables — never cached.

For materialised rollups see ``schema.DDL_MV_CAMPAIGN_STATS`` and the
trigger pack ``TRIGGERS_MV_CAMPAIGN``.
"""
from __future__ import annotations

VIEW_EXPERIMENT_SUMMARY = """
CREATE VIEW IF NOT EXISTS v_experiment_summary AS
SELECT
  e.id                       AS experiment_id,
  e.campaign_id,
  c.title                    AS campaign_title,
  e.sample_id,
  s.label                    AS sample_label,
  s.material                 AS sample_material,
  e.title                    AS title,
  e.exp_type,
  e.started_at,
  e.ended_at,
  e.exit_status,
  e.conclusion,
  (SELECT count(*) FROM actions a WHERE a.experiment_id = e.id)            AS action_count,
  (SELECT count(*) FROM actions a WHERE a.experiment_id = e.id
                                  AND a.status = 'succeeded')              AS action_ok,
  (SELECT count(*) FROM actions a WHERE a.experiment_id = e.id
                                  AND a.status = 'failed')                 AS action_err,
  (SELECT count(*) FROM observations o WHERE o.experiment_id = e.id)       AS observation_count,
  (SELECT count(DISTINCT sf.id) FROM scan_files sf
     JOIN actions a ON sf.produced_by_action_id = a.id
     WHERE a.experiment_id = e.id)                                         AS scan_file_count,
  (SELECT count(*) FROM events ev WHERE ev.experiment_id = e.id)           AS event_count,
  (SELECT count(*) FROM claims cl WHERE cl.experiment_id = e.id)           AS claim_count
FROM experiments e
LEFT JOIN campaigns c ON e.campaign_id = c.id
LEFT JOIN samples s ON e.sample_id = s.id
"""

VIEW_SKILL_USAGE = """
CREATE VIEW IF NOT EXISTS v_skill_usage AS
SELECT
  action_type,
  agent_id,
  count(*)                                          AS n,
  sum(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS n_ok,
  sum(CASE WHEN status='failed'    THEN 1 ELSE 0 END) AS n_err,
  sum(CASE WHEN status='retracted' THEN 1 ELSE 0 END) AS n_retracted,
  avg(duration_ms)                                  AS avg_duration_ms,
  max(hlc)                                          AS last_seen
FROM actions
GROUP BY action_type, agent_id
"""

VIEW_TIMELINE = """
CREATE VIEW IF NOT EXISTS v_timeline AS
SELECT 'action' AS kind, id, experiment_id, hlc,
       action_type AS label, status, NULL AS scalar_value, NULL AS observable
FROM actions
UNION ALL
SELECT 'observation', id, experiment_id, hlc,
       observable, NULL, scalar_value, observable
FROM observations
UNION ALL
SELECT 'event', id, experiment_id, hlc,
       topic, kind, NULL, NULL
FROM events WHERE experiment_id IS NOT NULL
"""

VIEW_PENDING_APPROVALS = """
CREATE VIEW IF NOT EXISTS v_pending_approvals AS
SELECT a.* FROM actions a
LEFT JOIN approvals ap ON a.id = ap.action_id
WHERE EXISTS (
        SELECT 1 FROM policies p
        WHERE p.active = 1 AND p.requires_approval = 1 AND p.action_type = a.action_type
      )
  AND ap.id IS NULL
"""

VIEW_OPEN_EXPERIMENTS = """
CREATE VIEW IF NOT EXISTS v_open_experiments AS
SELECT e.*, c.title AS campaign_title, s.label AS sample_label
FROM experiments e
LEFT JOIN campaigns c ON e.campaign_id = c.id
LEFT JOIN samples s   ON e.sample_id   = s.id
WHERE e.ended_at IS NULL
"""

VIEW_RECENT_SCAN_FILES = """
CREATE VIEW IF NOT EXISTS v_recent_scan_files AS
SELECT sf.*, a.experiment_id, a.agent_id, a.action_type
FROM scan_files sf
JOIN actions a ON sf.produced_by_action_id = a.id
"""

ALL_VIEWS: list[str] = [
    VIEW_EXPERIMENT_SUMMARY,
    VIEW_SKILL_USAGE,
    VIEW_TIMELINE,
    VIEW_PENDING_APPROVALS,
    VIEW_OPEN_EXPERIMENTS,
    VIEW_RECENT_SCAN_FILES,
]


def install_views(store) -> None:
    """Create (or replace) all read-only views on the given store."""
    with store.connect() as conn:
        for v in ALL_VIEWS:
            conn.executescript(v)
