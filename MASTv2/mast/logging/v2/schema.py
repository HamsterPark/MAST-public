"""DDL for the v2 experiment record system.

Per compass terminal plan (docs/v2/compass_artifact_wf-84adfed0...). All
fact tables are append-only, enforced by BEFORE UPDATE / BEFORE DELETE
triggers. Causal links are stored as ``parent_action_id`` (PROV-O
``wasInformedBy``). Big data is externalised to ``scan_files`` (CAS by
sha256), referenced from ``observations``.

Tables (16 in total — 13 logical entities, with ``claim_graph`` expanded
to three tables ``claims`` + ``evidence_edges`` + ``entity_refs``):

  schema_versions    migration log
  campaigns          long-running scientific projects
  samples            physical samples
  plans              experiment / strategy plans (XD agent output)
  experiments        single experimental session
  instrument_states  full snapshots of Nanonis state (delta-source)
  actions            command-side: "what we did" (append-only)
  scan_files         CAS index of .sxm/.dat/.h5 files (append-only)
  observations       query-side: "what we measured" (append-only)
  events             EventBus persistence (append-only, dedup'd)
  approvals          HITL approval records (append-only)
  reviews            PR-agent + human review verdicts
  claims             scientific claims (hypotheses / conclusions)
  evidence_edges     PROV-O edges between claims / observations / actions
  entity_refs        polymorphic targets for evidence edges
  audit_log          security / governance trail (append-only, locked)
  policies           which action_types need approval (mutable cfg)
"""
from __future__ import annotations

# ── Pragmas applied once on every connection ──────────────────────────
REQUIRED_PRAGMAS: list[str] = [
    "PRAGMA foreign_keys = ON",
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA mmap_size = 268435456",  # 256 MB
]

# ── Table DDL, ordered so FKs resolve cleanly ─────────────────────────
DDL_SCHEMA_VERSIONS = """
CREATE TABLE IF NOT EXISTS schema_versions (
  version              TEXT PRIMARY KEY,        -- semver, e.g. '2.0.0'
  applied_at           TEXT NOT NULL,           -- HLC string or ISO datetime
  description          TEXT NOT NULL,
  up_script_sha256     TEXT,
  down_script_sha256   TEXT,
  applied_by           TEXT NOT NULL DEFAULT 'system'
) STRICT;
"""

DDL_CAMPAIGNS = """
CREATE TABLE IF NOT EXISTS campaigns (
  id                   TEXT PRIMARY KEY,        -- ULID
  title                TEXT NOT NULL,
  hypothesis           TEXT NOT NULL,
  hypothesis_kind      TEXT NOT NULL CHECK (hypothesis_kind IN
                        ('exploratory','confirmatory','calibration','methodology')),
  goal_json            TEXT NOT NULL CHECK (json_valid(goal_json)),
  status               TEXT NOT NULL DEFAULT 'draft' CHECK (status IN
                        ('draft','running','paused','completed','aborted')),
  created_at           TEXT NOT NULL,           -- HLC
  created_by           TEXT NOT NULL,
  parent_campaign_id   TEXT REFERENCES campaigns(id),
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_campaigns_status_created ON campaigns(status, created_at);
CREATE INDEX IF NOT EXISTS idx_campaigns_parent ON campaigns(parent_campaign_id)
  WHERE parent_campaign_id IS NOT NULL;
"""

DDL_SAMPLES = """
CREATE TABLE IF NOT EXISTS samples (
  id                   TEXT PRIMARY KEY,
  label                TEXT NOT NULL,
  material             TEXT NOT NULL,
  prep_method          TEXT,
  prep_log_json        TEXT CHECK (prep_log_json IS NULL OR json_valid(prep_log_json)),
  created_at           TEXT NOT NULL,
  retired_at           TEXT,                    -- soft delete; NULL = active
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);
CREATE INDEX IF NOT EXISTS idx_samples_material ON samples(material);
CREATE INDEX IF NOT EXISTS idx_samples_active ON samples(retired_at) WHERE retired_at IS NULL;
"""

DDL_PLANS = """
CREATE TABLE IF NOT EXISTS plans (
  id                   TEXT PRIMARY KEY,
  campaign_id          TEXT REFERENCES campaigns(id),
  experiment_id        TEXT,                    -- FK resolved post-experiments via trigger if needed
  plan_kind            TEXT NOT NULL CHECK (plan_kind IN
                        ('pre_experiment','tip_conditioning_strategy',
                         'parameter_sweep','safety_policy','data_processing')),
  title                TEXT NOT NULL,
  definition_json      TEXT NOT NULL CHECK (json_valid(definition_json)),
  hypothesis           TEXT,
  success_criteria_json TEXT CHECK (success_criteria_json IS NULL
                        OR json_valid(success_criteria_json)),
  status               TEXT NOT NULL DEFAULT 'draft' CHECK (status IN
                        ('draft','active','completed','abandoned')),
  created_at           TEXT NOT NULL,
  created_by           TEXT NOT NULL,
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_plans_campaign ON plans(campaign_id);
CREATE INDEX IF NOT EXISTS idx_plans_experiment ON plans(experiment_id)
  WHERE experiment_id IS NOT NULL;
"""

DDL_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS experiments (
  id                   TEXT PRIMARY KEY,
  campaign_id          TEXT NOT NULL REFERENCES campaigns(id),
  sample_id            TEXT NOT NULL REFERENCES samples(id),
  plan_id              TEXT REFERENCES plans(id),
  title                TEXT NOT NULL,
  exp_type             TEXT NOT NULL,           -- 'topo_scan','sts_grid','tip_prep',...
  instrument_state_id  TEXT,                    -- references instrument_states (created lazily)
  started_at           TEXT NOT NULL,
  ended_at             TEXT,                    -- NULL = in progress
  exit_status          TEXT CHECK (exit_status IS NULL OR exit_status IN
                        ('success','aborted','failed','timeout')),
  conclusion           TEXT,
  conclusion_evidence_ids TEXT CHECK (conclusion_evidence_ids IS NULL
                        OR json_valid(conclusion_evidence_ids)),
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_experiments_campaign_start ON experiments(campaign_id, started_at);
CREATE INDEX IF NOT EXISTS idx_experiments_sample ON experiments(sample_id);
CREATE INDEX IF NOT EXISTS idx_experiments_open ON experiments(ended_at) WHERE ended_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_experiments_exp_type ON experiments(exp_type);
"""

DDL_INSTRUMENT_STATES = """
CREATE TABLE IF NOT EXISTS instrument_states (
  id                   TEXT PRIMARY KEY,
  experiment_id        TEXT REFERENCES experiments(id),
  hlc                  TEXT NOT NULL,
  state_json           TEXT NOT NULL CHECK (json_valid(state_json)),
  reason               TEXT NOT NULL,           -- 'periodic','reboot','tip_change','manual',...
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_states_exp_hlc ON instrument_states(experiment_id, hlc);
"""

DDL_ACTIONS = """
CREATE TABLE IF NOT EXISTS actions (
  id                   TEXT PRIMARY KEY,
  experiment_id        TEXT NOT NULL REFERENCES experiments(id),
  parent_action_id     TEXT REFERENCES actions(id),
  caused_by_event_id   TEXT,                    -- for actions resuming after a break
  agent_id             TEXT NOT NULL,           -- 'XD','DP','IC','operator','policy:auto'
  action_type          TEXT NOT NULL,           -- 'set_bias','scan','sts','tip_pulse',...
  prompt_id            TEXT,                    -- which user prompt / orchestrator turn
  thread_id            TEXT,                    -- LangGraph thread_id
  tool_call_id         TEXT,                    -- LangGraph tool_call uid
  params_json          TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(params_json)),
  instrument_state_id  TEXT REFERENCES instrument_states(id),
  state_delta_json     TEXT CHECK (state_delta_json IS NULL OR json_valid(state_delta_json)),
  hlc                  TEXT NOT NULL,
  duration_ms          INTEGER,
  status               TEXT NOT NULL DEFAULT 'pending' CHECK (status IN
                        ('pending','running','succeeded','failed','rolled_back','retracted')),
  error                TEXT,
  param_bias_v         REAL GENERATED ALWAYS AS
                        (CAST(json_extract(params_json, '$.bias_v') AS REAL)) VIRTUAL,
  param_setpoint_a     REAL GENERATED ALWAYS AS
                        (CAST(json_extract(params_json, '$.setpoint_a') AS REAL)) VIRTUAL,
  param_x_m            REAL GENERATED ALWAYS AS
                        (CAST(json_extract(params_json, '$.x_m') AS REAL)) VIRTUAL,
  param_y_m            REAL GENERATED ALWAYS AS
                        (CAST(json_extract(params_json, '$.y_m') AS REAL)) VIRTUAL,
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_actions_exp_hlc ON actions(experiment_id, hlc);
CREATE INDEX IF NOT EXISTS idx_actions_parent ON actions(parent_action_id)
  WHERE parent_action_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_actions_agent_type ON actions(agent_id, action_type, hlc);
CREATE INDEX IF NOT EXISTS idx_actions_thread ON actions(thread_id) WHERE thread_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_actions_bias ON actions(param_bias_v)
  WHERE param_bias_v IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);
"""

DDL_SCAN_FILES = """
CREATE TABLE IF NOT EXISTS scan_files (
  id                   TEXT PRIMARY KEY,
  sha256               TEXT NOT NULL UNIQUE,
  size_bytes           INTEGER NOT NULL CHECK (size_bytes >= 0),
  mime_type            TEXT NOT NULL,
  format_kind          TEXT NOT NULL CHECK (format_kind IN
                        ('sxm','dat','3ds','h5','png','npy','parquet','tiff','other')),
  current_path         TEXT NOT NULL,
  parser_spec          TEXT NOT NULL,           -- e.g. 'nanonis-sxm-v3','pyspm-0.4'
  parser_kwargs_json   TEXT CHECK (parser_kwargs_json IS NULL
                        OR json_valid(parser_kwargs_json)),
  produced_by_action_id TEXT NOT NULL REFERENCES actions(id),
  meta_json            TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(meta_json)),
  fixity_ok            INTEGER NOT NULL DEFAULT 1 CHECK (fixity_ok IN (0,1)),
  fixity_checked_at    TEXT,
  ocfl_object_id       TEXT,
  hlc                  TEXT NOT NULL,
  meta_bias_v          REAL GENERATED ALWAYS AS
                        (CAST(json_extract(meta_json, '$.bias_v') AS REAL)) VIRTUAL,
  meta_setpoint_a      REAL GENERATED ALWAYS AS
                        (CAST(json_extract(meta_json, '$.setpoint_a') AS REAL)) VIRTUAL,
  meta_width_m         REAL GENERATED ALWAYS AS
                        (CAST(json_extract(meta_json, '$.width_m') AS REAL)) VIRTUAL,
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_scanfiles_action ON scan_files(produced_by_action_id);
CREATE INDEX IF NOT EXISTS idx_scanfiles_format ON scan_files(format_kind);
CREATE INDEX IF NOT EXISTS idx_scanfiles_fixity ON scan_files(fixity_ok) WHERE fixity_ok = 0;
CREATE INDEX IF NOT EXISTS idx_scanfiles_hlc ON scan_files(hlc);
"""


# ── file_locations —— 唯一一张【可变】的文件事实表 ────────────────────
#
# 为什么不塞进 scan_files：scan_files 在 _APPEND_ONLY_FACT_TABLES 里，BEFORE
# UPDATE 触发器直接 ABORT，`current_path` 永远无法 UPDATE。而"文件在哪"是一个
# **多对一且可变**的事实：同一份字节可以同时躺在实验文件夹、隔离区和用户自己
# 的 Nanonis 目录里；一次 copy_failed 之后重试成功还要能把 status 改回 ok。
# 把它塞进 append-only 表就只能靠 events 打补丁，很脏。
#
# scan_files 保持它原本的语义（全局内容索引：一份内容一行，由哪个 action 产生，
# 摘要是什么），两张表各司其职。
#
# rel_path 存的是**相对实验目录的 POSIX 路径** —— 这样整个实验文件夹被移动或
# 改名时，一行都不用改。
#
# 设计文档：docs/v2/design/experiment_folder_persistence.md §6
DDL_FILE_LOCATIONS = """
CREATE TABLE IF NOT EXISTS file_locations (
  sha256        TEXT NOT NULL,
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  sample_id     TEXT,
  rel_path      TEXT NOT NULL,
  root_kind     TEXT NOT NULL DEFAULT 'experiment_folder'
                  CHECK (root_kind IN ('experiment_folder','origin','quarantine')),
  origin_path   TEXT,
  source        TEXT NOT NULL CHECK (source IN
                  ('skill','manual','migrated','import','inplace','stray')),
  action_id     TEXT REFERENCES actions(id),
  size_bytes    INTEGER NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
  ingested_at   TEXT NOT NULL,
  verified_at   TEXT,
  status        TEXT NOT NULL DEFAULT 'ok'
                  CHECK (status IN ('ok','missing','copy_failed','quarantined')),
  PRIMARY KEY (sha256, experiment_id, rel_path)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_floc_exp ON file_locations(experiment_id, ingested_at);
CREATE INDEX IF NOT EXISTS idx_floc_sample ON file_locations(sample_id)
  WHERE sample_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_floc_status ON file_locations(status) WHERE status != 'ok';
CREATE INDEX IF NOT EXISTS idx_floc_action ON file_locations(action_id)
  WHERE action_id IS NOT NULL;
"""

DDL_OBSERVATIONS = """
CREATE TABLE IF NOT EXISTS observations (
  id                   TEXT PRIMARY KEY,
  action_id            TEXT NOT NULL REFERENCES actions(id),
  experiment_id        TEXT NOT NULL REFERENCES experiments(id),
  observable           TEXT NOT NULL,           -- 'topography','didv','i_t','rms_noise',...
  channel              TEXT,                    -- e.g. 'Z (m)','Current (A)'
  hlc                  TEXT NOT NULL,
  scalar_value         REAL,
  units                TEXT,
  result_summary_json  TEXT CHECK (result_summary_json IS NULL
                        OR (json_valid(result_summary_json)
                            AND length(result_summary_json) <= 4096)),
  scan_file_id         TEXT REFERENCES scan_files(id),
  schema_version       INTEGER NOT NULL DEFAULT 2,
  CHECK (scalar_value IS NOT NULL
         OR scan_file_id IS NOT NULL
         OR result_summary_json IS NOT NULL)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_obs_exp_observable ON observations(experiment_id, observable, hlc);
CREATE INDEX IF NOT EXISTS idx_obs_action ON observations(action_id);
CREATE INDEX IF NOT EXISTS idx_obs_scan_file ON observations(scan_file_id)
  WHERE scan_file_id IS NOT NULL;
"""

DDL_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
  id                   TEXT PRIMARY KEY,
  topic                TEXT NOT NULL,           -- 'instrument.scan_started','agent.XD.msg',...
  kind                 TEXT NOT NULL CHECK (kind IN
                        ('edge','snapshot','heartbeat','external')),
  severity             TEXT NOT NULL DEFAULT 'info' CHECK (severity IN
                        ('debug','info','warning','error','critical')),
  hlc                  TEXT NOT NULL,
  experiment_id        TEXT REFERENCES experiments(id),
  action_id            TEXT REFERENCES actions(id),
  producer             TEXT,                    -- 'planner','vision','buffer_summarizer','safety'
  payload_json         TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(payload_json)),
  dedup_key            TEXT,
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_events_topic_hlc ON events(topic, hlc);
CREATE INDEX IF NOT EXISTS idx_events_exp_hlc ON events(experiment_id, hlc)
  WHERE experiment_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity, hlc);
CREATE INDEX IF NOT EXISTS idx_events_dedup ON events(topic, dedup_key)
  WHERE dedup_key IS NOT NULL;
"""

DDL_APPROVALS = """
CREATE TABLE IF NOT EXISTS approvals (
  id                   TEXT PRIMARY KEY,
  action_id            TEXT NOT NULL UNIQUE REFERENCES actions(id),
  approver_id          TEXT NOT NULL,
  approver_kind        TEXT NOT NULL CHECK (approver_kind IN
                        ('human_operator','human_pi','automated_policy')),
  approval_method      TEXT NOT NULL,           -- 'gui_click','signed_token','policy_rule_v1'
  approval_evidence    TEXT,
  approved_at          TEXT NOT NULL,
  expires_at           TEXT,
  policy_version       TEXT,
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_approvals_approver ON approvals(approver_id, approved_at);
"""

DDL_REVIEWS = """
CREATE TABLE IF NOT EXISTS reviews (
  id                   TEXT PRIMARY KEY,
  target_kind          TEXT NOT NULL CHECK (target_kind IN
                        ('experiment','claim','draft_paper','observation')),
  target_id            TEXT NOT NULL,
  reviewer_id          TEXT NOT NULL,
  reviewer_kind        TEXT NOT NULL CHECK (reviewer_kind IN
                        ('pr_agent','human_pi','human_peer','external_referee')),
  verdict              TEXT NOT NULL CHECK (verdict IN
                        ('accept','reject','request_changes','tentative')),
  comments_json        TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(comments_json)),
  hlc                  TEXT NOT NULL,
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_reviews_target ON reviews(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_reviews_reviewer ON reviews(reviewer_id, hlc);
"""

DDL_CLAIMS = """
CREATE TABLE IF NOT EXISTS claims (
  id                   TEXT PRIMARY KEY,
  experiment_id        TEXT REFERENCES experiments(id),
  campaign_id          TEXT REFERENCES campaigns(id),
  statement            TEXT NOT NULL,
  confidence           REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  status               TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN
                        ('proposed','supported','contested','retracted','verified')),
  created_at           TEXT NOT NULL,
  created_by           TEXT NOT NULL,
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_claims_experiment ON claims(experiment_id);
CREATE INDEX IF NOT EXISTS idx_claims_campaign ON claims(campaign_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
"""

DDL_ENTITY_REFS = """
CREATE TABLE IF NOT EXISTS entity_refs (
  id                   TEXT PRIMARY KEY,
  entity_kind          TEXT NOT NULL CHECK (entity_kind IN
                        ('action','observation','scan_file','claim','event',
                         'experiment','sample','plan','paper_ref','external_url')),
  entity_id            TEXT NOT NULL,
  external_url         TEXT,
  UNIQUE (entity_kind, entity_id, external_url)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_entity_refs_entity ON entity_refs(entity_kind, entity_id);
"""

DDL_EVIDENCE_EDGES = """
CREATE TABLE IF NOT EXISTS evidence_edges (
  id                   TEXT PRIMARY KEY,
  claim_id             TEXT NOT NULL REFERENCES claims(id),
  edge_type            TEXT NOT NULL CHECK (edge_type IN
                        ('prov:wasDerivedFrom','prov:wasAttributedTo','prov:used',
                         'prov:wasGeneratedBy','prov:wasInformedBy','prov:hadPrimarySource',
                         'mast:supports','mast:contradicts','mast:replicates','mast:cites')),
  target_ref_id        TEXT NOT NULL REFERENCES entity_refs(id),
  weight               REAL CHECK (weight IS NULL OR (weight >= -1 AND weight <= 1)),
  rationale            TEXT,
  hlc                  TEXT NOT NULL,
  created_by           TEXT NOT NULL,
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_edges_claim ON evidence_edges(claim_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_target ON evidence_edges(target_ref_id);
"""

DDL_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS audit_log (
  id                   TEXT PRIMARY KEY,
  actor_id             TEXT NOT NULL,
  actor_kind           TEXT NOT NULL CHECK (actor_kind IN
                        ('user','agent','system','admin')),
  event                TEXT NOT NULL,           -- 'schema_migration','approval_bypass','file_delete',...
  hlc                  TEXT NOT NULL,
  payload_json         TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(payload_json)),
  schema_version       INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_audit_actor_hlc ON audit_log(actor_id, hlc);
CREATE INDEX IF NOT EXISTS idx_audit_event_hlc ON audit_log(event, hlc);
"""

DDL_POLICIES = """
CREATE TABLE IF NOT EXISTS policies (
  id                   TEXT PRIMARY KEY,
  action_type          TEXT NOT NULL,
  requires_approval    INTEGER NOT NULL DEFAULT 0 CHECK (requires_approval IN (0,1)),
  required_kind        TEXT CHECK (required_kind IS NULL OR required_kind IN
                        ('human_operator','human_pi','automated_policy','any_human')),
  reason               TEXT NOT NULL,
  version              TEXT NOT NULL,
  active               INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
  created_at           TEXT NOT NULL,
  UNIQUE (action_type, version)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_policies_action ON policies(action_type, active);
"""

# ── Agent training / usage trajectories (RFC: agent_training_log_rfc.md) ──
# One orchestrator run = one trajectory; its ordered causal steps串成
# operator_intent → route → agent_turn → tool_call → safety → hitl → observation.
# The fact data (params/results/state) lives in the existing tables (actions /
# observations / approvals); trajectory_steps only holds pointers + the decision
# snapshot. NO tensors inline — only scan_file sha256 pointers (enforced in StepRepo).
DDL_TRAJECTORIES = """
CREATE TABLE IF NOT EXISTS trajectories (
  id                    TEXT PRIMARY KEY,                 -- ULID
  thread_id             TEXT NOT NULL,                    -- LangGraph thread_id (join anchor)
  experiment_id         TEXT REFERENCES experiments(id),
  campaign_id           TEXT,
  sample_id             TEXT,
  created_at_hlc        TEXT NOT NULL,
  ended_at_hlc          TEXT,                             -- set once (NULL → value)
  operator_intent_json  TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(operator_intent_json)),
  context_snapshot_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(context_snapshot_json)),
  exit_status           TEXT CHECK (exit_status IS NULL OR
                          exit_status IN ('success','aborted','failed','timeout')),
  final_outcome_json    TEXT CHECK (final_outcome_json IS NULL OR json_valid(final_outcome_json)),
  quality_json          TEXT CHECK (quality_json IS NULL OR json_valid(quality_json)),  -- backfillable
  schema_version        INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_traj_thread ON trajectories(thread_id);
CREATE INDEX IF NOT EXISTS idx_traj_exp ON trajectories(experiment_id);
CREATE INDEX IF NOT EXISTS idx_traj_open ON trajectories(ended_at_hlc) WHERE ended_at_hlc IS NULL;
"""

DDL_TRAJECTORY_STEPS = """
CREATE TABLE IF NOT EXISTS trajectory_steps (
  id              TEXT PRIMARY KEY,                        -- ULID
  trajectory_id   TEXT NOT NULL REFERENCES trajectories(id),
  parent_step_id  TEXT REFERENCES trajectory_steps(id),    -- causal chain
  hlc             TEXT NOT NULL,                            -- monotonic order
  hop_idx         INTEGER,                                 -- supervisor hop sequence
  step_type       TEXT NOT NULL CHECK (step_type IN
                    ('route_decision','agent_turn','tool_call','safety_gate',
                     'hitl_request','hitl_resolution','observation','rollback','error')),
  actor_kind      TEXT,
  agent_id        TEXT,
  model_id        TEXT,
  tool_call_id    TEXT,                                    -- join key across the run
  action_id       TEXT REFERENCES actions(id),             -- tool_call step → fact table
  observation_id  TEXT REFERENCES observations(id),
  approval_id     TEXT REFERENCES approvals(id),
  input_json      TEXT CHECK (input_json IS NULL OR json_valid(input_json)),   -- decision-time snapshot
  output_json     TEXT CHECK (output_json IS NULL OR json_valid(output_json)), -- verdict/reason/params/reasoning
  duration_ms     INTEGER,
  schema_version  INTEGER NOT NULL DEFAULT 2
) STRICT;
CREATE INDEX IF NOT EXISTS idx_tstep_traj_hlc ON trajectory_steps(trajectory_id, hlc);
CREATE INDEX IF NOT EXISTS idx_tstep_tool_call ON trajectory_steps(tool_call_id) WHERE tool_call_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tstep_parent ON trajectory_steps(parent_step_id) WHERE parent_step_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tstep_type ON trajectory_steps(step_type, hlc);
CREATE INDEX IF NOT EXISTS idx_tstep_action ON trajectory_steps(action_id) WHERE action_id IS NOT NULL;
"""

# ── Append-only triggers ──────────────────────────────────────────────
# Reject UPDATE and DELETE on every fact table. Approved exceptions are
# handled via column-level whitelist triggers further below.

_APPEND_ONLY_FACT_TABLES = (
    "scan_files",
    "observations",
    "events",
    "approvals",
    "audit_log",
    "trajectory_steps",      # steps are immutable once written (append-only)
)

TRIGGERS_APPEND_ONLY = "\n".join(
    f"""
CREATE TRIGGER IF NOT EXISTS trg_{t}_no_update BEFORE UPDATE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_{t}_no_delete BEFORE DELETE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} are append-only'); END;
"""
    for t in _APPEND_ONLY_FACT_TABLES
)

# Actions are append-only EXCEPT three columns that move from pending → terminal once:
#   status   : pending|running -> succeeded|failed|rolled_back|retracted
#   duration_ms : NULL -> int
#   error       : NULL -> string (only on failure)
TRIGGERS_ACTIONS_LIMITED = """
CREATE TRIGGER IF NOT EXISTS trg_actions_no_delete BEFORE DELETE ON actions
BEGIN SELECT RAISE(ABORT, 'actions are append-only (use action_type=retracted to compensate)'); END;

CREATE TRIGGER IF NOT EXISTS trg_actions_immutable_cols BEFORE UPDATE ON actions
WHEN OLD.id != NEW.id
  OR OLD.experiment_id != NEW.experiment_id
  OR COALESCE(OLD.parent_action_id,'') != COALESCE(NEW.parent_action_id,'')
  OR OLD.agent_id != NEW.agent_id
  OR OLD.action_type != NEW.action_type
  OR OLD.params_json != NEW.params_json
  OR OLD.hlc != NEW.hlc
  OR COALESCE(OLD.thread_id,'') != COALESCE(NEW.thread_id,'')
  OR COALESCE(OLD.tool_call_id,'') != COALESCE(NEW.tool_call_id,'')
  OR COALESCE(OLD.prompt_id,'') != COALESCE(NEW.prompt_id,'')
BEGIN
  SELECT RAISE(ABORT, 'actions immutable columns: id/experiment_id/agent_id/action_type/params/hlc');
END;

CREATE TRIGGER IF NOT EXISTS trg_actions_status_monotonic BEFORE UPDATE OF status ON actions
WHEN NEW.status NOT IN ('pending','running','succeeded','failed','rolled_back','retracted')
  OR (OLD.status = 'succeeded' AND NEW.status NOT IN ('succeeded','retracted'))
  OR (OLD.status = 'failed' AND NEW.status NOT IN ('failed','retracted'))
  OR (OLD.status = 'rolled_back' AND NEW.status NOT IN ('rolled_back','retracted'))
  OR (OLD.status = 'retracted' AND NEW.status != 'retracted')
BEGIN
  SELECT RAISE(ABORT, 'invalid status transition; terminal states immutable except retraction');
END;
"""

# Experiments can have ended_at / exit_status / conclusion / conclusion_evidence_ids
# set exactly once (NULL → set). Everything else is immutable.
TRIGGERS_EXPERIMENTS_LIMITED = """
CREATE TRIGGER IF NOT EXISTS trg_experiments_no_delete BEFORE DELETE ON experiments
BEGIN SELECT RAISE(ABORT, 'experiments cannot be deleted'); END;

CREATE TRIGGER IF NOT EXISTS trg_experiments_immutable_cols BEFORE UPDATE ON experiments
WHEN OLD.id != NEW.id
  OR OLD.campaign_id != NEW.campaign_id
  OR OLD.sample_id != NEW.sample_id
  OR OLD.title != NEW.title
  OR OLD.exp_type != NEW.exp_type
  OR OLD.started_at != NEW.started_at
  OR (OLD.ended_at IS NOT NULL AND OLD.ended_at != COALESCE(NEW.ended_at, OLD.ended_at))
  OR (OLD.exit_status IS NOT NULL AND OLD.exit_status != COALESCE(NEW.exit_status, OLD.exit_status))
BEGIN
  SELECT RAISE(ABORT, 'experiments immutable columns or terminal field reassigned');
END;
"""

# Trajectories: ended_at_hlc / exit_status / final_outcome_json set exactly once
# (NULL → value). quality_json is freely backfillable (operator rating/reward may
# arrive post-hoc and be revised). Everything else immutable; no delete.
TRIGGERS_TRAJECTORIES_LIMITED = """
CREATE TRIGGER IF NOT EXISTS trg_trajectories_no_delete BEFORE DELETE ON trajectories
BEGIN SELECT RAISE(ABORT, 'trajectories cannot be deleted'); END;

CREATE TRIGGER IF NOT EXISTS trg_trajectories_immutable_cols BEFORE UPDATE ON trajectories
WHEN OLD.id != NEW.id
  OR OLD.thread_id != NEW.thread_id
  OR OLD.created_at_hlc != NEW.created_at_hlc
  OR OLD.operator_intent_json != NEW.operator_intent_json
  OR OLD.context_snapshot_json != NEW.context_snapshot_json
  OR (OLD.ended_at_hlc IS NOT NULL AND OLD.ended_at_hlc != COALESCE(NEW.ended_at_hlc, OLD.ended_at_hlc))
  OR (OLD.exit_status IS NOT NULL AND OLD.exit_status != COALESCE(NEW.exit_status, OLD.exit_status))
  OR (OLD.final_outcome_json IS NOT NULL AND
      OLD.final_outcome_json != COALESCE(NEW.final_outcome_json, OLD.final_outcome_json))
BEGIN
  SELECT RAISE(ABORT, 'trajectories immutable columns or terminal field reassigned');
END;
"""

# No LLM self-approval allowed (compass §2.2.8). approver_kind must be human or
# automated_policy. approver_id starting with 'agent:' is rejected.
TRIGGERS_APPROVALS = """
CREATE TRIGGER IF NOT EXISTS trg_no_self_approval BEFORE INSERT ON approvals
WHEN NEW.approver_kind NOT IN ('human_operator','human_pi','automated_policy')
  OR (NEW.approver_kind != 'automated_policy' AND NEW.approver_id LIKE 'agent:%')
BEGIN
  SELECT RAISE(ABORT, 'LLM/agent cannot approve actions; need human_* or automated_policy');
END;
"""

# Block insertion of an action that requires approval when no approval row exists yet.
# Predicates the dangerous action_types via the policies table (lazily registered).
TRIGGERS_REQUIRE_APPROVAL = """
CREATE TRIGGER IF NOT EXISTS trg_action_requires_approval BEFORE INSERT ON actions
WHEN EXISTS (
       SELECT 1 FROM policies
       WHERE active = 1
         AND requires_approval = 1
         AND action_type = NEW.action_type
     )
  AND NOT EXISTS (
       SELECT 1 FROM approvals WHERE action_id = NEW.id
     )
BEGIN
  SELECT RAISE(ABORT, 'this action_type requires explicit approval — insert approval row first');
END;
"""

# ── Materialized rollup tables (kept by AFTER INSERT triggers) ────────
DDL_MV_CAMPAIGN_STATS = """
CREATE TABLE IF NOT EXISTS mv_campaign_stats (
  campaign_id          TEXT PRIMARY KEY REFERENCES campaigns(id),
  experiment_count     INTEGER NOT NULL DEFAULT 0,
  action_count         INTEGER NOT NULL DEFAULT 0,
  observation_count    INTEGER NOT NULL DEFAULT 0,
  scan_file_count      INTEGER NOT NULL DEFAULT 0,
  last_activity_hlc    TEXT
) STRICT;
"""

TRIGGERS_MV_CAMPAIGN = """
CREATE TRIGGER IF NOT EXISTS trg_mv_camp_exp_ins AFTER INSERT ON experiments
BEGIN
  INSERT INTO mv_campaign_stats(campaign_id, experiment_count, last_activity_hlc)
    VALUES (NEW.campaign_id, 1, NEW.started_at)
    ON CONFLICT(campaign_id) DO UPDATE SET
      experiment_count = experiment_count + 1,
      last_activity_hlc = NEW.started_at;
END;

CREATE TRIGGER IF NOT EXISTS trg_mv_camp_act_ins AFTER INSERT ON actions
BEGIN
  INSERT INTO mv_campaign_stats(campaign_id, action_count, last_activity_hlc)
    VALUES (
      (SELECT campaign_id FROM experiments WHERE id = NEW.experiment_id),
      1, NEW.hlc
    )
    ON CONFLICT(campaign_id) DO UPDATE SET
      action_count = action_count + 1,
      last_activity_hlc = NEW.hlc;
END;

CREATE TRIGGER IF NOT EXISTS trg_mv_camp_obs_ins AFTER INSERT ON observations
BEGIN
  INSERT INTO mv_campaign_stats(campaign_id, observation_count, last_activity_hlc)
    VALUES (
      (SELECT campaign_id FROM experiments WHERE id = NEW.experiment_id),
      1, NEW.hlc
    )
    ON CONFLICT(campaign_id) DO UPDATE SET
      observation_count = observation_count + 1,
      last_activity_hlc = NEW.hlc;
END;

CREATE TRIGGER IF NOT EXISTS trg_mv_camp_scan_ins AFTER INSERT ON scan_files
BEGIN
  INSERT INTO mv_campaign_stats(campaign_id, scan_file_count, last_activity_hlc)
    VALUES (
      (SELECT campaign_id FROM experiments
        WHERE id = (SELECT experiment_id FROM actions WHERE id = NEW.produced_by_action_id)),
      1, NEW.hlc
    )
    ON CONFLICT(campaign_id) DO UPDATE SET
      scan_file_count = scan_file_count + 1,
      last_activity_hlc = NEW.hlc;
END;
"""

# ── Master DDL list, applied in order ─────────────────────────────────
ALL_DDL: list[str] = [
    DDL_SCHEMA_VERSIONS,
    DDL_CAMPAIGNS,
    DDL_SAMPLES,
    DDL_PLANS,
    DDL_EXPERIMENTS,
    DDL_INSTRUMENT_STATES,
    DDL_ACTIONS,
    DDL_SCAN_FILES,
    DDL_FILE_LOCATIONS,
    DDL_OBSERVATIONS,
    DDL_EVENTS,
    DDL_APPROVALS,
    DDL_REVIEWS,
    DDL_CLAIMS,
    DDL_ENTITY_REFS,
    DDL_EVIDENCE_EDGES,
    DDL_AUDIT_LOG,
    DDL_POLICIES,
    DDL_TRAJECTORIES,
    DDL_TRAJECTORY_STEPS,
    DDL_MV_CAMPAIGN_STATS,
    TRIGGERS_APPEND_ONLY,
    TRIGGERS_ACTIONS_LIMITED,
    TRIGGERS_EXPERIMENTS_LIMITED,
    TRIGGERS_TRAJECTORIES_LIMITED,
    TRIGGERS_APPROVALS,
    TRIGGERS_REQUIRE_APPROVAL,
    TRIGGERS_MV_CAMPAIGN,
]
