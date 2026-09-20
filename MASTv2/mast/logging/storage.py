"""SQLite backend for experiment logging."""

from __future__ import annotations

import json
import logging          # stdlib — absolute import, not this package (mast.logging)
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

from mast.core.types import ActionRecord, SampleRecord
from mast.logging.action_record import action_to_dict, dict_to_action

logger = logging.getLogger(__name__)


class ExperimentStorage:
    """SQLite backend for experiment logging."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        # WAL lets readers run alongside a writer; writers still exclude each
        # other, so a second writer needs a busy timeout to wait its turn rather
        # than raise "database is locked" at once.
        #
        # CORRECTION (2026-07-27): this line was added believing sqlite's default
        # busy timeout is 0 and that operator feedback was being lost to lock
        # collisions. **That was wrong.** Python's ``sqlite3.connect()`` takes a
        # ``timeout`` argument that IS the busy timeout and defaults to 5.0 s;
        # measured on this build::
        #
        #     sqlite3.connect(p)                → PRAGMA busy_timeout = 5000
        #     sqlite3.connect(p, timeout=0)     → PRAGMA busy_timeout = 0
        #
        # The call above passes no ``timeout``, so 5 s was already in force and
        # this PRAGMA changes nothing. It is kept only to make the value explicit
        # and to pin it against someone later passing ``timeout=0``.
        #
        # Consequence worth carrying: if operator feedback ever really was lost,
        # lock contention was NOT the cause and the real one is still unfound.
        # (Independently reached in docs/v2/fixes/2026-07-27-recfix.md §8, which
        # vetoed the same change on the v2 store for the same reason.)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist, and migrate existing ones."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    goal_text   TEXT NOT NULL DEFAULT '',
                    start_time  TEXT NOT NULL,
                    end_time    TEXT,
                    status      TEXT NOT NULL DEFAULT 'running',
                    notes       TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id              TEXT PRIMARY KEY,
                    experiment_id   TEXT NOT NULL,
                    name            TEXT NOT NULL,
                    description     TEXT NOT NULL DEFAULT '',
                    start_time      TEXT NOT NULL,
                    end_time        TEXT,
                    status          TEXT NOT NULL DEFAULT 'active',
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id              TEXT PRIMARY KEY,
                    experiment_id   TEXT,
                    sample_id       TEXT,
                    timestamp       TEXT NOT NULL,
                    skill_name      TEXT NOT NULL,
                    skill_version   TEXT NOT NULL DEFAULT '',
                    parameters      TEXT DEFAULT '{}',
                    result          TEXT DEFAULT '',
                    state_before    TEXT DEFAULT '',
                    state_after     TEXT DEFAULT '',
                    nanonis_calls   TEXT DEFAULT '[]',
                    context         TEXT DEFAULT '',
                    duration_s      REAL DEFAULT 0.0,
                    approval_source TEXT DEFAULT 'auto',
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS environment_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    sensor_name TEXT NOT NULL,
                    value       REAL NOT NULL,
                    unit        TEXT NOT NULL DEFAULT '',
                    status      TEXT NOT NULL DEFAULT 'ok'
                )
            """)
            # v2 addition: chat / agent conversation tied to the active experiment
            # (requirement: 对话应当记录在实验记录中). role ∈ user|assistant|
            # operator|system|tool; agent names the producing model/agent.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT NOT NULL,
                    experiment_id TEXT,
                    sample_id     TEXT,
                    role          TEXT NOT NULL,
                    agent         TEXT NOT NULL DEFAULT '',
                    content       TEXT NOT NULL,
                    meta          TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_exp "
                "ON conversation_log(experiment_id, id)"
            )
            # v2 addition: operator feedback on the agent conversation — quick
            # rating taps and free-text comments, recorded into the experiment
            # record (requirement: 对话反馈打分与实验记录存在一起). kind ∈
            # rating|comment; rating holds the chosen option value (rating rows)
            # and comment holds free text (comment rows).
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    experiment_id   TEXT,
                    sample_id       TEXT,
                    conversation_id TEXT,
                    kind            TEXT NOT NULL DEFAULT 'rating',
                    rating          TEXT NOT NULL DEFAULT '',
                    comment         TEXT NOT NULL DEFAULT '',
                    agent           TEXT NOT NULL DEFAULT '',
                    meta            TEXT NOT NULL DEFAULT '{}',
                    -- 处理状态。NULL = 未处理；有值 = 处理时刻（ISO）。
                    -- 刻意**不加**一个独立的 resolved 布尔：两个字段可以互相
                    -- 矛盾，而「什么时候处理的」是判断一条反馈还算不算数所必需
                    -- 的信息（同一个症状可以在两个版本里各报一次）。
                    resolved_at      TEXT,
                    resolved_version TEXT,
                    resolved_note    TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_exp "
                "ON feedback(experiment_id, id)"
            )
            # v2 addition: spatial map markers — every position-tagged operation
            # (scan footprint / STS / bias pulse / tip-shaping / move) and manual
            # Nanonis activity, recorded so the live scan-map IS the experiment
            # record's map (requirement: 此地图就是实验记录的地图). Coordinates are
            # Nanonis stage frame, METRES. kind ∈ scan|sts|pulse|tip_shape|move|
            # manual|coarse_move|approach|crash; source ∈ skill|manual|plan|import;
            # status ∈ done|failed|active.
            #
            # ``coord_epoch`` is the COORDINATE-SYSTEM GENERATION (2026-07-30). A
            # lateral coarse-motor move slides the sample stage: the piezo XY
            # readout keeps reporting the same numbers, but they now point at a
            # different patch of surface, so every older marker's coordinate is
            # dead. Its definition is "how many ``coarse_move`` rows precede this
            # one in this scope" — stamped at INSERT time (see ``log_marker``).
            # Older databases have NULL here, which reads as 0: they contain no
            # coarse_move rows at all (the kind is new), so by that definition
            # every legacy row genuinely IS generation 0. No backfill needed.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS map_markers (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT NOT NULL,
                    experiment_id TEXT,
                    sample_id     TEXT,
                    kind          TEXT NOT NULL DEFAULT 'move',
                    skill_name    TEXT NOT NULL DEFAULT '',
                    x_m           REAL,
                    y_m           REAL,
                    w_m           REAL,
                    h_m           REAL,
                    angle_deg     REAL NOT NULL DEFAULT 0.0,
                    label         TEXT NOT NULL DEFAULT '',
                    status        TEXT NOT NULL DEFAULT 'done',
                    source        TEXT NOT NULL DEFAULT 'skill',
                    meta          TEXT NOT NULL DEFAULT '{}',
                    coord_epoch   INTEGER
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_map_markers_scope "
                "ON map_markers(experiment_id, sample_id, id)"
            )
            # v2 addition (2026-07-28): the ONE pointer that says which experiment
            # and sample are current. Experiments and samples are permanent rows
            # with no lifecycle state — an experiment may sit idle for years and
            # still resume — so "which one is current" cannot live in ``experiments.status``.
            # Switching = moving this pointer; it never touches an experiment row.
            #
            # It lives in THIS db (not ui_settings.json) because the pointer names
            # rows in this db: put it in another file and a data-dir switch or a db
            # restore turns it into a dangling reference. Here the two live and die
            # together.
            #
            # CHECK (id = 1) + INSERT OR REPLACE = race-free single-row upsert.
            # Design: docs/v2/design/experiment_folder_persistence.md §10
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_scope (
                    id            INTEGER PRIMARY KEY CHECK (id = 1),
                    experiment_id TEXT,
                    sample_id     TEXT,
                    updated_at    TEXT NOT NULL,
                    updated_by    TEXT NOT NULL DEFAULT '',
                    note          TEXT NOT NULL DEFAULT ''
                )
            """)
            # v2 addition (2026-07-29): 文档索引 —— 文献报告 / 实验计划 /
            # 实验报告 / 论文草稿 / 评审报告。
            # 设计：docs/v2/design/document_and_library_management.md §3.5
            #
            # 这是**索引，不是记录**：权威在实验文件夹里的 doc.json +
            # versions.jsonl，这三张表随时可由 reindex 从文件夹重建。写入是
            # best-effort —— 写失败只告警，磁盘上已经是权威了。
            #
            # 为什么放 v1 库而不是 v2 schema.py（三条硬理由）：
            #   1. 文档键在 **v1** experiment id 上（文件夹、active_scope、
            #      conversations、plans 全是 v1 id）。v2 是另一套 id 空间 ——
            #      前车之鉴是 file_locations.experiment_id FK 指 v2，逼得
            #      reindex 只能把外键失败当「预期内」吞掉。
            #   2. v2 的 STRICT + append-only 触发器与「title 可改」的可变索引
            #      相性差。
            #   3. plans / conversation_log / feedback / map_markers 全是同库
            #      同模式的先例。
            #
            # 没有 status / finalized_at / archived 列 —— 文档和实验一样没有
            # 终态，永远可以再来一版（INCREMENTAL-ONLY）。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id          TEXT PRIMARY KEY,
                    experiment_id   TEXT,
                    sample_id       TEXT,
                    kind            TEXT NOT NULL DEFAULT 'experiment_report',
                    title           TEXT NOT NULL DEFAULT '',
                    dir_name        TEXT NOT NULL DEFAULT '',
                    root_kind       TEXT NOT NULL DEFAULT 'experiment',
                    created_at      TEXT NOT NULL DEFAULT '',
                    created_by      TEXT NOT NULL DEFAULT '',
                    conversation_id TEXT,
                    run_id          TEXT,
                    target_doc_id   TEXT,
                    target_version  INTEGER,
                    latest_version  INTEGER NOT NULL DEFAULT 0,
                    updated_at      TEXT NOT NULL DEFAULT '',
                    legacy_stem     TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_scope "
                "ON documents(experiment_id, kind, updated_at DESC)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    doc_id          TEXT NOT NULL,
                    version         INTEGER NOT NULL,
                    rel_path        TEXT NOT NULL DEFAULT '',
                    sha256          TEXT NOT NULL DEFAULT '',
                    words           INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT NOT NULL DEFAULT '',
                    created_by      TEXT NOT NULL DEFAULT '',
                    conversation_id TEXT,
                    run_id          TEXT,
                    note            TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (doc_id, version)
                )
            """)
            # 关联实验（主归属**不**进这张表 —— 两处真源会立刻分叉）。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_links (
                    doc_id        TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    PRIMARY KEY (doc_id, experiment_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_links_exp "
                "ON document_links(experiment_id)"
            )
            # ── 针尖登记 (2026-07-31) ──────────────────────────────────
            #
            # 仪器级：换实验、换样品都未必换针尖，所以这张表**不挂
            # experiment_id**（对比 samples）。它记的是这台仪器的换针史。
            #
            # 「服役期」模型 —— 每行是**一次装入**，不是一根物理针。同一根针
            # 拔下来再装回去记新行：qPlus 传感器重装后 Q 值就变了，粘的位置、
            # 引线应力都不同，当新针对待才符合物理。
            #
            # **没有 active_tip 指针表**，这是刻意的。active_scope 需要指针是
            # 因为实验/样品永久存在且来回切，「当前」无法从行状态导出；而针尖
            # 服役期是线性时间的（永远不存在切回旧行），于是
            # 「当前针尖 = 唯一 removed_at IS NULL 的行」就是完备编码。
            # 冗余状态才会悬空 —— experiment_log.restore_scope 的整条自愈链都
            # 在处理指针悬空，不建指针这类 bug 结构性消失。
            #
            # 词表（material / fabrication / form）用软约束:TEXT 列 + 工具层
            # 归一，不加 SQL CHECK —— samples.sample_type 就是这么做的，
            # 收紧词表时不必迁移老库。
            #
            # ``retire_snapshot`` 存这根针退役时被清掉的学习标定（dI/dV、qPlus
            # 振幅基线等）。换针使那些量失效必须清，但清掉不等于要丢 —— 存进
            # 退役行是归档语义，日后想查「上根针的 dI/dV 标定是多少」还在。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tips (
                    id                 TEXT PRIMARY KEY,
                    tip_index          INTEGER,
                    name               TEXT NOT NULL DEFAULT '',
                    material           TEXT NOT NULL DEFAULT '',
                    material_detail    TEXT NOT NULL DEFAULT '',
                    fabrication        TEXT NOT NULL DEFAULT 'unknown',
                    form               TEXT NOT NULL DEFAULT 'stm_wire',
                    wire_diameter_mm   REAL,
                    qplus_sensor_model TEXT NOT NULL DEFAULT '',
                    qplus_f0_hz        REAL,
                    qplus_q            REAL,
                    qplus_k_n_per_m    REAL,
                    installed_at       TEXT,
                    removed_at         TEXT,
                    installed_by       TEXT NOT NULL DEFAULT '',
                    note               TEXT NOT NULL DEFAULT '',
                    retire_snapshot    TEXT NOT NULL DEFAULT '{}',
                    created_at         TEXT NOT NULL
                )
            """)
            # 当前针尖的查询走这条索引（removed_at IS NULL 的那一行）。
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tips_current "
                "ON tips(removed_at, created_at DESC)"
            )
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns/tables that may be missing in older databases."""
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "notes" not in cols:
            conn.execute("ALTER TABLE experiments ADD COLUMN notes TEXT NOT NULL DEFAULT ''")

        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(actions)").fetchall()
        }
        if "sample_id" not in cols:
            conn.execute("ALTER TABLE actions ADD COLUMN sample_id TEXT")

        sample_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        if "sample_type" not in sample_cols:
            conn.execute("ALTER TABLE samples ADD COLUMN sample_type TEXT NOT NULL DEFAULT ''")
        if "sample_subtype" not in sample_cols:
            conn.execute("ALTER TABLE samples ADD COLUMN sample_subtype TEXT NOT NULL DEFAULT ''")

        # ── 2026-07-28: 实验文件夹持久化 ────────────────────────────
        #
        # ``last_active_at`` 是本模型里唯一有意义的"活跃度"信号 —— 它**不是状态，
        # 是描述**：记录"上次动过这个实验/样品"。切换器按它倒序排，这正是
        # 「做了一个月这个又回去做那个」需要的排序（按上次干它的时间，而不是
        # 创建时间）。历史行为 NULL → 调用方退化到 start_time，行为可预测。
        #
        # ``dir_name`` 是实验文件夹的目录名，**创建时冻结**：scan_files 是
        # append-only，current_path 永远无法 UPDATE，目录一改名所有历史行就指向
        # 不存在的路径。改名只改 name 列，dir_name 不动。
        #
        # ``v2_campaign_id`` / ``v2_sample_id`` 把 v1 行钉到 v2 库的对应实体上
        # （v1 实验 ↔ v2 campaign，v1 样品 ↔ v2 sample）。
        exp_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "last_active_at" not in exp_cols:
            conn.execute("ALTER TABLE experiments ADD COLUMN last_active_at TEXT")
        if "dir_name" not in exp_cols:
            conn.execute("ALTER TABLE experiments ADD COLUMN dir_name TEXT")
        if "v2_campaign_id" not in exp_cols:
            conn.execute("ALTER TABLE experiments ADD COLUMN v2_campaign_id TEXT")

        sample_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        if "last_active_at" not in sample_cols:
            conn.execute("ALTER TABLE samples ADD COLUMN last_active_at TEXT")
        if "dir_name" not in sample_cols:
            conn.execute("ALTER TABLE samples ADD COLUMN dir_name TEXT")
        if "sample_index" not in sample_cols:
            conn.execute("ALTER TABLE samples ADD COLUMN sample_index INTEGER")
        if "v2_sample_id" not in sample_cols:
            conn.execute("ALTER TABLE samples ADD COLUMN v2_sample_id TEXT")

        # 环境记录终于能归属到实验/样品。这张表至今写了很多行、
        # 读取函数 get_environment_history 却一个调用方都没有 —— 归属列
        # 加上之后它才有意义（GET /api/experiments/{id}/environment）。
        env_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(environment_log)").fetchall()
        }
        if "experiment_id" not in env_cols:
            conn.execute("ALTER TABLE environment_log ADD COLUMN experiment_id TEXT")
        if "sample_id" not in env_cols:
            conn.execute("ALTER TABLE environment_log ADD COLUMN sample_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_env_scope "
            "ON environment_log(experiment_id, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_active "
            "ON experiments(last_active_at DESC)"
        )

        # ── 2026-07-30: 扫描地图坐标代次 ───────────────────────────────
        #
        # Nullable on purpose: a legacy row's NULL is not "unknown", it is
        # generation 0 (see the table comment). Adding the column is enough —
        # backfilling it with 0 would write the same meaning twice.
        # feedback：处理状态三列。**必须在这里补**，因为上面的建表是
        # CREATE TABLE IF NOT EXISTS —— 对已经存在的表一个字都不改，而真机上的
        # 库早就存在了。只改 DDL 的结果是：新列在开发机上有、在用户那台机器上
        # 没有，UPDATE 报「没有这一列」，而处理状态永远标不上。
        fb_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()
        }
        for col in ("resolved_at", "resolved_version", "resolved_note"):
            if col not in fb_cols:
                conn.execute(f"ALTER TABLE feedback ADD COLUMN {col} TEXT")

        marker_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(map_markers)").fetchall()
        }
        if "coord_epoch" not in marker_cols:
            conn.execute("ALTER TABLE map_markers ADD COLUMN coord_epoch INTEGER")

    # ── Experiments ──────────────────────────────────────────────────

    def create_experiment(self, name: str, goal_text: str = "") -> str:
        """Create new experiment, return experiment_id (UUID)."""
        experiment_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiments (id, name, goal_text, start_time, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (experiment_id, name, goal_text, datetime.now().isoformat(), "running"),
            )
        return experiment_id

    def end_experiment(self, experiment_id: str, status: str = "completed") -> None:
        """Mark experiment as ended."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET end_time = ?, status = ? WHERE id = ?",
                (datetime.now().isoformat(), status, experiment_id),
            )

    def rename_experiment(self, experiment_id: str, name: str) -> bool:
        """Rename an experiment (MAST records name, NOT a Nanonis field).
        Returns True if a row was updated."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE experiments SET name = ? WHERE id = ?",
                (name, experiment_id),
            )
            return cur.rowcount > 0

    def get_experiment(self, experiment_id: str) -> dict | None:
        """Retrieve experiment metadata."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_experiments(self, limit: int = 50) -> list[dict]:
        """List recent experiments, newest first, each with its CURRENT sample.

        ``sample_name`` / ``sample_id`` come from the newest ``active`` sample,
        falling back to the newest sample of any status. The panel that shows
        the running experiment had only the experiment UUID to display and
        rendered ``441ebe7c-2a7`` — the operator's response was 「这里的id不是
        用户友好的。这里应该显示样品名。这个id是样品id还是对话id？」. An id
        that cannot even be identified is not a readout.

        One query, no N+1: the correlated subqueries run per experiment row
        inside SQLite, and the list is capped at *limit*.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.*, "
                "  (SELECT s.name FROM samples s WHERE s.experiment_id = e.id "
                "   ORDER BY (s.last_active_at IS NULL), s.last_active_at DESC, "
                "            s.start_time DESC "
                "   LIMIT 1) AS sample_name, "
                "  (SELECT s.id FROM samples s WHERE s.experiment_id = e.id "
                "   ORDER BY (s.last_active_at IS NULL), s.last_active_at DESC, "
                "            s.start_time DESC "
                "   LIMIT 1) AS sample_id "
                "FROM experiments e ORDER BY e.start_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_running_experiment_by_name(self, name: str) -> dict | None:
        """The most recent still-``running`` experiment whose name matches *name*
        (case-insensitive, whitespace-trimmed, Unicode-correct), or None.

        Used for idempotent start_experiment: a re-issued start of the same named
        run (after a resume / a re-dispatch) attaches to the open row instead of
        minting a duplicate. Matching is done in Python (``str.casefold``) rather
        than SQL ``lower()`` because names carry CJK (e.g. 'NiI2质量表征') that
        SQLite's ASCII-only ``lower()`` would not fold."""
        target = (name or "").strip().casefold()
        if not target:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE status = 'running' "
                "ORDER BY start_time DESC"
            ).fetchall()
        for r in rows:
            if (r["name"] or "").strip().casefold() == target:
                return dict(r)
        return None

    def find_experiment_by_name(self, name: str) -> dict | None:
        """任意年龄的同名实验里最近活动过的那一条，或 None。

        与 :meth:`find_running_experiment_by_name` 的差别只有一点：**不过滤
        status**。实验没有终态，一条 2026-06 的实验和一条今天的实验一样可以
        被继续做 —— 这正是「做了一个月这个又回去做那个」。

        匹配仍在 Python 里做（``str.casefold``）而不是 SQL ``lower()``：名字带
        CJK（如 'NiI2质量表征'），SQLite 只折 ASCII 的 ``lower()`` 折不动它。
        """
        target = (name or "").strip().casefold()
        if not target:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments "
                "ORDER BY (last_active_at IS NULL), last_active_at DESC, start_time DESC"
            ).fetchall()
        for r in rows:
            if (r["name"] or "").strip().casefold() == target:
                return dict(r)
        return None

    def list_experiments_recent(self, limit: int = 30, query: str = "") -> list[dict]:
        """切换器的数据源：按**上次活动时间**倒序。

        ``list_experiments`` 按 ``start_time DESC`` 排，对"回到上个月那个实验"
        是错的排序 —— 那个实验是上个月**建**的，但可能是昨天才动过。
        """
        q = (query or "").strip().casefold()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.*, "
                "  (SELECT COUNT(*) FROM samples s WHERE s.experiment_id = e.id) "
                "    AS sample_count, "
                "  (SELECT COUNT(*) FROM actions a WHERE a.experiment_id = e.id) "
                "    AS action_count "
                "FROM experiments e "
                "ORDER BY (e.last_active_at IS NULL), e.last_active_at DESC, "
                "         e.start_time DESC "
                "LIMIT ?",
                (max(1, int(limit)) if not q else 500,),
            ).fetchall()
        out = [dict(r) for r in rows]
        if q:
            out = [r for r in out if q in (r.get("name") or "").casefold()
                   or q in (r.get("goal_text") or "").casefold()][:max(1, int(limit))]
        return out

    # ── Active scope pointer ─────────────────────────────────────────

    def get_active_scope(self) -> dict | None:
        """当前实验/样品指针。没设过返回 None。"""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM active_scope WHERE id = 1"
                ).fetchone()
            return dict(row) if row else None
        except sqlite3.Error:
            return None

    def set_active_scope(self, experiment_id: str | None, sample_id: str | None,
                         *, updated_by: str = "", note: str = "") -> None:
        """原子地移动指针，并给指向的行打 ``last_active_at`` 戳。

        单事务。**不写 status，不写 end_time** —— 切换不改变任何实验/样品行的
        内容，它只是换了个指向。
        """
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO active_scope "
                "(id, experiment_id, sample_id, updated_at, updated_by, note) "
                "VALUES (1, ?, ?, ?, ?, ?)",
                (experiment_id, sample_id, now, updated_by, note),
            )
            if experiment_id:
                conn.execute(
                    "UPDATE experiments SET last_active_at = ? WHERE id = ?",
                    (now, experiment_id),
                )
            if sample_id:
                conn.execute(
                    "UPDATE samples SET last_active_at = ? WHERE id = ?",
                    (now, sample_id),
                )

    def set_experiment_dir_name(self, experiment_id: str, dir_name: str) -> None:
        """记下实验文件夹的目录名（创建时冻结，改名不动它）。"""
        with self._connect() as conn:
            conn.execute("UPDATE experiments SET dir_name = ? WHERE id = ?",
                         (dir_name, experiment_id))

    def set_sample_dir_name(self, sample_id: str, dir_name: str, index: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE samples SET dir_name = ?, sample_index = ? WHERE id = ?",
                (dir_name, int(index), sample_id))

    def next_sample_index(self, experiment_id: str) -> int:
        """该实验**下一个尚未创建**的样品的序号（1-based，单调递增不回收）。

        不回收是刻意的：回收会让 S02 先后指向两个不同的样品，而目录名一旦
        创建就冻结，那两个东西就永远分不清了。

        给**已存在**的样品分配序号请用 :meth:`sample_ordinal` —— 那时行已经在
        库里，用本方法会多算它自己一个（第一个样品会变成 S02）。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(COALESCE(sample_index, 0)) AS mx, COUNT(*) AS n "
                "FROM samples WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        mx = int((row["mx"] if row else 0) or 0)
        n = int((row["n"] if row else 0) or 0)
        return max(mx, n) + 1

    def sample_ordinal(self, sample_id: str) -> int:
        """一个**已存在**样品在它所属实验内的创建次序（1-based）。

        按 ``start_time`` 数它自己及更早的样品，所以第一个建的样品就是 1，
        与资源管理器里的字母序一致。已经分配过 ``sample_index`` 的行直接复用
        （目录名冻结后序号不能再变）。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT experiment_id, start_time, sample_index FROM samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
            if not row:
                return 1
            if row["sample_index"]:
                return int(row["sample_index"])
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM samples "
                "WHERE experiment_id = ? AND start_time <= ?",
                (row["experiment_id"], row["start_time"]),
            ).fetchone()
        return max(1, int((n["n"] if n else 1) or 1))

    def list_experiments_with_counts(self, limit: int = 50) -> list[dict]:
        """List recent experiments with sample counts (single query, no N+1)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.*, COUNT(s.id) AS sample_count "
                "FROM experiments e "
                "LEFT JOIN samples s ON e.id = s.experiment_id "
                "GROUP BY e.id "
                "ORDER BY e.start_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Samples ────────────────────────────────────────────────────

    def create_sample(
        self,
        experiment_id: str,
        name: str,
        description: str = "",
        sample_type: str = "",
        sample_subtype: str = "",
    ) -> str:
        """Create a new sample under an experiment. Returns sample_id (UUID)."""
        sample_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO samples "
                "(id, experiment_id, name, description, start_time, status, sample_type, sample_subtype) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sample_id, experiment_id, name, description,
                 datetime.now().isoformat(), "active", sample_type, sample_subtype),
            )
        return sample_id

    def end_sample(self, sample_id: str, status: str = "completed") -> None:
        """Mark a sample as ended."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE samples SET end_time = ?, status = ? WHERE id = ?",
                (datetime.now().isoformat(), status, sample_id),
            )

    def rename_sample(self, sample_id: str, name: str) -> bool:
        """Rename a sample (MAST records name). Returns True if a row was updated."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE samples SET name = ? WHERE id = ?",
                (name, sample_id),
            )
            return cur.rowcount > 0

    def get_sample(self, sample_id: str) -> dict | None:
        """Retrieve sample metadata."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM samples WHERE id = ?", (sample_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_samples(self, experiment_id: str) -> list[dict]:
        """List all samples for an experiment, ordered by start_time."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM samples WHERE experiment_id = ? ORDER BY start_time",
                (experiment_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_active_sample(self, experiment_id: str) -> dict | None:
        """Get the currently active sample for an experiment (if any).

        DEPRECATED (2026-07-28): 用 :meth:`get_last_active_sample`。样品没有
        终态，``status='active'`` 只是历史遗留的展示列，用它挑"当前样品"会
        漏掉所有被旧代码 end 成 'completed' 的样品 —— 而那些样品完全可以被
        切回来继续用。保留本方法只为不破坏既有调用方。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM samples WHERE experiment_id = ? AND status = 'active' "
                "ORDER BY start_time DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_last_active_sample(self, experiment_id: str) -> dict | None:
        """该实验里**上次用过**的样品（无 status 过滤）。

        切回一个旧实验时应该落到"上次在这个实验里用的那块样品"上 —— 这是
        用户的直觉。历史行没有 ``last_active_at`` 就退到 ``start_time``。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM samples WHERE experiment_id = ? "
                "ORDER BY (last_active_at IS NULL), last_active_at DESC, start_time DESC "
                "LIMIT 1",
                (experiment_id,),
            ).fetchone()
        return dict(row) if row else None

    def find_sample_by_name(self, experiment_id: str, name: str) -> dict | None:
        """实验内按名字找样品（casefold，CJK 安全）。"""
        target = (name or "").strip().casefold()
        if not target:
            return None
        for s in self.get_samples(experiment_id):
            if (s.get("name") or "").strip().casefold() == target:
                return s
        return None

    # ── Tips（针尖登记，仪器级）─────────────────────────────────────
    #
    # 「当前针尖」不存指针，就是唯一 ``removed_at IS NULL`` 的行（见建表注释）。

    #: ``update_tip`` 允许改的列。装入/退役时刻与序号不在其中 —— 那是事件事实，
    #: 改它们等于改历史；``id`` / ``created_at`` 同理。
    _TIP_UPDATABLE: tuple[str, ...] = (
        "name", "material", "material_detail", "fabrication", "form",
        "wire_diameter_mm", "qplus_sensor_model", "qplus_f0_hz", "qplus_q",
        "qplus_k_n_per_m", "note", "installed_at",
    )

    def create_tip(self, fields: dict, *, retire_snapshot: dict | None = None) -> str:
        """登记一次装入：退役当前针尖 + 插入新行。返回新 tip_id。

        **单个事务**做两件事,因为半途失败的两种结果都很糟:只退役了旧针 =
        仪器凭空「没有针」;只插了新针 = 库里两行 open,「当前针尖」不再唯一。

        退役的是**所有** open 行而不只是最新一行 —— 正常情况下 open 行本来就
        只有一行,但如果手改库或早期 bug 留下了第二行,这里顺手收敛掉。事务
        本身就是自愈,不需要单独的一致性检查流程。
        """
        tip_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        snap = json.dumps(retire_snapshot or {}, ensure_ascii=False)
        row = {k: fields.get(k) for k in self._TIP_UPDATABLE}
        with self._connect() as conn:
            conn.execute(
                "UPDATE tips SET removed_at = ?, retire_snapshot = ? "
                "WHERE removed_at IS NULL",
                (now, snap),
            )
            mx = conn.execute(
                "SELECT MAX(COALESCE(tip_index, 0)) AS mx, COUNT(*) AS n FROM tips"
            ).fetchone()
            nxt = max(int((mx["mx"] if mx else 0) or 0),
                      int((mx["n"] if mx else 0) or 0)) + 1
            conn.execute(
                "INSERT INTO tips (id, tip_index, name, material, material_detail, "
                "fabrication, form, wire_diameter_mm, qplus_sensor_model, "
                "qplus_f0_hz, qplus_q, qplus_k_n_per_m, installed_at, "
                "installed_by, note, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tip_id, nxt,
                 str(row.get("name") or ""),
                 str(row.get("material") or ""),
                 str(row.get("material_detail") or ""),
                 str(row.get("fabrication") or "unknown"),
                 str(row.get("form") or "stm_wire"),
                 row.get("wire_diameter_mm"),
                 str(row.get("qplus_sensor_model") or ""),
                 row.get("qplus_f0_hz"), row.get("qplus_q"),
                 row.get("qplus_k_n_per_m"),
                 str(row.get("installed_at") or now),
                 str(fields.get("installed_by") or ""),
                 str(row.get("note") or ""),
                 now),
            )
        return tip_id

    def get_current_tip(self) -> dict | None:
        """当前装在仪器里的针尖（``removed_at IS NULL``），没有则 None。

        查到多行只可能来自手改库；取最新一行并告警,而不是抛 —— 读不到当前
        针尖绝不该让调用方失败（注入块、样品快照都在读它）。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tips WHERE removed_at IS NULL "
                "ORDER BY created_at DESC, tip_index DESC"
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            logger.warning(
                "tips 表里有 %d 行 removed_at IS NULL（应当只有一行）；"
                "取最新一行。下次登记针尖会自动收敛。", len(rows))
        return dict(rows[0])

    def get_tip(self, tip_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tips WHERE id = ?", (tip_id,)).fetchone()
        return dict(row) if row else None

    def list_tips(self, limit: int = 100) -> list[dict]:
        """换针史,最近装入的在前。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tips ORDER BY created_at DESC, tip_index DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def tips_in_service_during(self, experiment_id: str) -> list[dict]:
        """一次实验**期间在役**的针尖，装入时刻早的在前。

        「针尖记录等是不是也应该在实验记录中?」

        这是**展示层聚合**，不是第二个真源：``tips`` 仍然是针尖的唯一真源，
        这个方法一个字都不写，只是把「这段时间里装的是哪根针」这个本来就能从
        两张表答出来的问题**答出来**。两张表恰好在同一个 SQLite 文件里
        （见 ``tips`` 的建表注释），所以纯 SQL 就够，不必把区间交集算在前端。

        区间交集，两端都可能开口：

        * 实验 ``[start_time, end_time)``；``end_time`` 为 NULL = **还没结束**
          （大多数行就是 NULL —— 见 experiment_folder_persistence 那条纪律：
          实验没有结束，更没有归档）。开口右端 ⇒ 一切之后的针尖都算重叠。
        * 针尖 ``[installed_at, removed_at)``；``removed_at`` 为 NULL = **在役**。

        NULL 的处理必须显式。``WHERE a.x <= b.y`` 形式的条件碰上 NULL 一律求值成
        NULL、被 WHERE 丢掉 —— 于是「实验没结束」和「针尖还在役」这两种**最常见**
        的情况会双双静默消失，而症状是一个空区块，读起来像「这次实验没换过针」。

        两个已知的口径问题，选的都是**宁可多列一根**而不是漏掉一根：

        1. ``installed_at`` 允许人工回填成纯日期 ``"2026-07-28"``（UI 明确鼓励，
           见 ``api/routes/tips.py`` 的字段说明）。字符串比较下它等价于当天
           00:00，也就是**偏早**。偏早只会把一根边界上的针多算进来。
        2. ``installed_at`` 可以为空。退回 ``created_at`` —— 那一列全长、必非空，
           且是这行**写进库**的时刻，即服役开始的一个下界。同样偏早。

        列表里带上 ``overlap_exact``：两条口径都没用上时为 True。让「这根针确实在
        役」和「这根针大概在役」看得出区别，而不是把后者印成前者。
        """
        with self._connect() as conn:
            exp = conn.execute(
                "SELECT start_time, end_time FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
            if exp is None:
                return []
            rows = conn.execute(
                # tip_start = COALESCE(installed_at, created_at) —— created_at 必非空，
                # 所以 tip_start 永不为 NULL，左端不需要再兜底。
                # 右端两个 NULL 各自展开成显式的 OR，不能靠 COALESCE 塞一个
                # '9999' 之类的哨兵：那种哨兵会跟着排序和导出跑出去，变成一个
                # 谁都不认识、又比较得动的假时间戳。
                "SELECT * FROM tips "
                "WHERE (? IS NULL OR ? = '' "
                "       OR COALESCE(installed_at, created_at) < ?) "
                "  AND (removed_at IS NULL OR removed_at > ?) "
                "ORDER BY COALESCE(installed_at, created_at) ASC, tip_index ASC",
                (exp["end_time"], exp["end_time"], exp["end_time"],
                 exp["start_time"]),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            installed = d.get("installed_at")
            d["overlap_exact"] = bool(installed) and len(str(installed)) > 10
            out.append(d)
        return out

    def find_tip_by_name(self, name: str) -> dict | None:
        """按名字找针尖（casefold，CJK 安全）。模型只有名字,没有 UUID。"""
        target = (name or "").strip().casefold()
        if not target:
            return None
        for t in self.list_tips(limit=500):
            if (t.get("name") or "").strip().casefold() == target:
                return t
        return None

    def update_tip(self, tip_id: str, fields: dict) -> bool:
        """补记针尖属性（线径、型号、备注…）。返回是否改到了行。

        只认 :data:`_TIP_UPDATABLE` 白名单里的列 —— 未知键静默忽略,与
        instrument_profile.sanitize 同样的形状。
        """
        sets, vals = [], []
        for key in self._TIP_UPDATABLE:
            if key in fields:
                sets.append(f"{key} = ?")
                vals.append(fields[key])
        if not sets:
            return False
        vals.append(tip_id)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE tips SET {', '.join(sets)} WHERE id = ?", vals)
            return cur.rowcount > 0

    def retire_current_tip(self, *, retire_snapshot: dict | None = None) -> bool:
        """物理取出针尖但还没装新的。返回是否退役了行。"""
        snap = json.dumps(retire_snapshot or {}, ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tips SET removed_at = ?, retire_snapshot = ? "
                "WHERE removed_at IS NULL",
                (datetime.now().isoformat(), snap),
            )
            return cur.rowcount > 0

    # ── Actions ──────────────────────────────────────────────────────

    def log_action(self, record: ActionRecord) -> None:
        """Insert an ActionRecord into the actions table."""
        d = action_to_dict(record)
        columns = ", ".join(d.keys())
        placeholders = ", ".join("?" for _ in d)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO actions ({columns}) VALUES ({placeholders})",
                tuple(d.values()),
            )

    def get_actions(
        self, experiment_id: str, sample_id: str | None = None
    ) -> list[ActionRecord]:
        """Retrieve actions for an experiment, ordered by timestamp.

        If *sample_id* is given, returns only actions for that sample.
        """
        if sample_id is not None:
            query = (
                "SELECT * FROM actions WHERE experiment_id = ? AND sample_id = ? "
                "ORDER BY timestamp"
            )
            params: tuple = (experiment_id, sample_id)
        else:
            query = "SELECT * FROM actions WHERE experiment_id = ? ORDER BY timestamp"
            params = (experiment_id,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict_to_action(dict(r)) for r in rows]

    def recent_actions(self, experiment_id: str, limit: int = 20) -> list[dict]:
        """某实验**最近**的动作（新的在前），给简报用的轻量视图。

        :meth:`get_actions` 会把整个实验的动作连同完整结果与 TCP 调用全量反序列化
        —— 一个跑了几天的实验有上千行。简报只要最近十几行的「谁、做了什么、成没成」，
        所以只取这几列，结果 JSON 里也只摘 ``success`` / ``error`` / ``summary``。
        ``context`` 是调用方身份（外部 agent 写 ``ext:<名字>``；agent 路径为空）。
        """
        import json as _json

        n = max(1, min(int(limit or 20), 500))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, sample_id, skill_name, context, "
                "approval_source, duration_s, result FROM actions "
                "WHERE experiment_id = ? ORDER BY timestamp DESC LIMIT ?",
                (experiment_id, n),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            raw = d.pop("result", "") or ""
            try:
                res = _json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                res = {}
            if not isinstance(res, dict):
                res = {}
            d["success"] = res.get("success")
            d["error"] = str(res.get("error") or "")[:300]
            d["summary"] = str(res.get("summary") or "")[:300]
            out.append(d)
        return out

    def get_last_invocation_times(self) -> dict[str, str]:
        """Return {skill_name: latest_timestamp} for all invoked skills."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT skill_name, MAX(timestamp) as last_ts "
                "FROM actions GROUP BY skill_name"
            ).fetchall()
        return {row["skill_name"]: row["last_ts"] for row in rows}

    def get_skill_execution_counts(self) -> dict[str, int]:
        """Return {skill_name: total_count} for all skills in the actions table."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT skill_name, COUNT(*) as cnt FROM actions GROUP BY skill_name"
            ).fetchall()
        return {row["skill_name"]: row["cnt"] for row in rows}

    # ── Environment ──────────────────────────────────────────────────

    def log_environment(
        self, sensor_name: str, value: float, unit: str, status: str = "ok",
        *, experiment_id: str | None = None, sample_id: str | None = None,
    ) -> None:
        """Log a single environment sensor reading.

        ``experiment_id`` / ``sample_id`` 是 2026-07-28 加的：在此之前这张表
        写了很多行却没有任何归属，也就没有任何读取方 —— 采了数据没人能用。
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO environment_log "
                "(timestamp, sensor_name, value, unit, status, experiment_id, sample_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), sensor_name, value, unit, status,
                 experiment_id, sample_id),
            )

    def get_environment_history(
        self,
        sensor_name: str,
        start_time: str | None = None,
        end_time: str | None = None,
        *,
        experiment_id: str | None = None,
        sample_id: str | None = None,
        limit: int = 0,
    ) -> list[dict]:
        """Query environment log for a sensor within time range / scope."""
        query = "SELECT * FROM environment_log WHERE sensor_name = ?"
        params: list = [sensor_name]
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if sample_id:
            query += " AND sample_id = ?"
            params.append(sample_id)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        query += " ORDER BY timestamp"
        if limit and int(limit) > 0:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def prune_environment_log(
        self, cutoff: str, *,
        keep_statuses: tuple[str, ...] = ("alarm", "error"),
        batch_rows: int = 20000,
        max_batches: int = 200,
        pause_s: float = 0.05,
    ) -> int:
        """删掉 *cutoff* 之前的原始环境读数。返回删除行数。

        这是这张表的**第一个删除者**。在此之前它以 2 秒一条的速度只增不减
        （43,200 行/天/传感器），而它的读取函数一个调用方都没有。现在
        :mod:`mast.envhistory` 会把同样的数据聚合成永久的统计桶，原始行因此
        变成"最近两周可以看细节"的缓存，而不是必须永久保存的记录。

        三条纪律：

        * **告警行豁免。** ``alarm`` / ``error`` 的读数是
          ``_on_env_alarm`` 停机决策与事后复盘的证据，永久保留。它们本来就
          稀少，留着不花什么。
        * **分批 + 批间让路。** 一次 ``DELETE`` 掉几十万行会把写锁按住好几秒,
          而同一个库上还有实验记录、动作记录和用户反馈在写。每批一个小事务,
          批间松手，让别人插得进来。
        * ``cutoff`` 必须与 :meth:`log_environment` 写入的时间戳**同构**
          （``datetime.now().isoformat()``，本地时间无时区）。两者都是可按
          字典序比较的定长前缀格式；换成 UTC 或带时区的字符串会让比较悄悄失效
          —— 表现是"清扫跑了但一行没删"，或者更糟，删掉不该删的。
        """
        placeholders = ",".join("?" * len(keep_statuses)) or "''"
        sql = (
            "DELETE FROM environment_log WHERE id IN ("
            " SELECT id FROM environment_log"
            f" WHERE timestamp < ? AND status NOT IN ({placeholders})"
            " LIMIT ?)"
        )
        params = [str(cutoff), *keep_statuses, int(max(1, batch_rows))]
        deleted = 0
        try:
            for _ in range(max(1, int(max_batches))):
                with self._connect() as conn:
                    cur = conn.execute(sql, params)
                    n = int(cur.rowcount or 0)
                deleted += n
                if n <= 0:
                    break
                if pause_s > 0:
                    time.sleep(float(pause_s))
        except sqlite3.Error as exc:
            # 清扫失败不是事故：下一轮会再来。已经删掉的那些批次是已提交的。
            logger.warning("environment_log 清扫中断（已删 %d 行）：%s", deleted, exc)
        return deleted

    def list_environment_sensors(self, experiment_id: str | None = None) -> list[dict]:
        """本实验（或全库）出现过的传感器 + 采样数 + 时间范围。"""
        q = ("SELECT sensor_name, unit, COUNT(*) AS n, "
             "MIN(timestamp) AS first_at, MAX(timestamp) AS last_at "
             "FROM environment_log")
        params: list = []
        if experiment_id:
            q += " WHERE experiment_id = ?"
            params.append(experiment_id)
        q += " GROUP BY sensor_name, unit ORDER BY sensor_name"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    # ── Conversation (chat / agent dialogue) ──────────────────────────

    def log_conversation(
        self,
        role: str,
        content: str,
        *,
        experiment_id: str | None = None,
        sample_id: str | None = None,
        agent: str = "",
        meta: dict | None = None,
    ) -> int:
        """Record one chat / agent message into the experiment record.

        role: user | assistant | operator | system | tool. Ties to the active
        experiment when *experiment_id* is given (the GUI passes the running
        experiment so the conversation is part of that record). Returns the row
        id. Best-effort serialisation of *meta* (never raises on odd types)."""
        try:
            meta_json = json.dumps(meta or {}, ensure_ascii=False, default=str)
        except Exception:
            meta_json = "{}"
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversation_log "
                "(timestamp, experiment_id, sample_id, role, agent, content, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), experiment_id, sample_id,
                 role, agent, content, meta_json),
            )
            return int(cur.lastrowid)

    def get_conversation(
        self,
        experiment_id: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Return conversation messages (chronological).

        When *experiment_id* is given, only that experiment's messages are
        returned; otherwise the most recent *limit* messages across all."""
        if experiment_id is not None:
            query = ("SELECT * FROM conversation_log WHERE experiment_id = ? "
                     "ORDER BY id DESC LIMIT ?")
            params: list = [experiment_id, limit]
        else:
            query = "SELECT * FROM conversation_log ORDER BY id DESC LIMIT ?"
            params = [limit]
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out = [dict(r) for r in rows]
        out.reverse()  # chronological (oldest first)
        for r in out:
            try:
                r["meta"] = json.loads(r.get("meta") or "{}")
            except Exception:
                r["meta"] = {}
        return out

    # ── Feedback (operator rating / comment on the agent conversation) ─────

    def log_feedback(
        self,
        *,
        rating: str = "",
        comment: str = "",
        experiment_id: str | None = None,
        sample_id: str | None = None,
        conversation_id: str | None = None,
        agent: str = "",
        meta: dict | None = None,
    ) -> int:
        """Record one operator feedback row into the experiment record.

        A *rating* tap and a free *comment* are independent: pass ``rating`` for
        a quick-rating row (``kind='rating'``) or ``comment`` for a comment row
        (``kind='comment'``). Ties to the active experiment / conversation when
        given. Best-effort meta serialisation; never raises on odd types."""
        kind = "rating" if rating else "comment"
        try:
            meta_json = json.dumps(meta or {}, ensure_ascii=False, default=str)
        except Exception:
            meta_json = "{}"
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO feedback "
                "(timestamp, experiment_id, sample_id, conversation_id, kind, "
                "rating, comment, agent, meta) VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), experiment_id, sample_id,
                 conversation_id, kind, rating, comment, agent, meta_json),
            )
            return int(cur.lastrowid)

    def set_feedback_resolved(
        self,
        feedback_id: int,
        *,
        resolved: bool = True,
        version: str = "",
        note: str = "",
    ) -> bool:
        """Mark one feedback row processed (or un-mark it). ``True`` if a row changed.

        Exists because the feedback list had no created-at and no processed flag,
        so every new batch began with an archaeology pass over git log and
        KNOWN_ISSUES to work out which entries were already closed — and one
        batch got that wrong in both directions (an item re-reported after being
        fixed, and an item assumed fixed that was only half-fixed).

        Un-marking is supported on purpose: a symptom that comes back is NOT the
        same as a mistake in the marking, and the operator needs to be able to
        say「这条又回来了」without losing the record that it was once addressed.
        """
        ts = datetime.now().isoformat() if resolved else None
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE feedback SET resolved_at = ?, resolved_version = ?,"
                " resolved_note = ? WHERE id = ?",
                (ts, version if resolved else "", note if resolved else "",
                 int(feedback_id)),
            )
            return cur.rowcount > 0

    def get_feedback(
        self,
        experiment_id: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Return feedback rows (chronological).

        Filtered to one experiment when *experiment_id* is given; otherwise the
        most recent *limit* rows across all."""
        if experiment_id is not None:
            query = ("SELECT * FROM feedback WHERE experiment_id = ? "
                     "ORDER BY id DESC LIMIT ?")
            params: list = [experiment_id, limit]
        else:
            query = "SELECT * FROM feedback ORDER BY id DESC LIMIT ?"
            params = [limit]
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out = [dict(r) for r in rows]
        out.reverse()
        for r in out:
            try:
                r["meta"] = json.loads(r.get("meta") or "{}")
            except Exception:
                r["meta"] = {}
        return out

    # ── Map markers (spatial record of every positioned operation) ─────────

    def log_marker(
        self,
        *,
        kind: str,
        x_m: float | None,
        y_m: float | None,
        w_m: float | None = None,
        h_m: float | None = None,
        angle_deg: float = 0.0,
        label: str = "",
        skill_name: str = "",
        status: str = "done",
        source: str = "skill",
        experiment_id: str | None = None,
        sample_id: str | None = None,
        meta: dict | None = None,
    ) -> int:
        """Record one spatial map marker into the experiment record.

        Coordinates are Nanonis stage frame, METRES. ``w_m``/``h_m`` are the
        footprint size for scan/frame markers (point operations leave them
        None). Ties to the active experiment / sample when given so the map is
        scoped per sample (换样品 → 新画布). Best-effort meta serialisation;
        never raises on odd types. Returns the new row id.

        Stamps ``coord_epoch`` inside the same write transaction as the INSERT.
        This is the ONLY ``INSERT INTO map_markers`` in the codebase, which is
        what makes a single stamping point possible: the four writers (skill
        recorder / manual state-diff watcher / manual spectrum tick / import
        route) must not each track "the current generation" themselves — that
        state would have to survive restarts, and four copies of it would drift.

        The stamp is a CACHE of a derivable fact, not an independent truth: the
        table is append-only (no UPDATE/DELETE path exists), and the generation
        is a monotone function of the row prefix, so a value computed at INSERT
        time can never go stale. A ``coarse_move`` row stamps the OLD generation
        — it marks the boundary and is itself drawn in the pre-move frame."""
        try:
            meta_json = json.dumps(meta or {}, ensure_ascii=False, default=str)
        except Exception:
            meta_json = "{}"
        with self._connect() as conn:
            epoch = self._count_coarse_moves(conn, experiment_id, sample_id)
            cur = conn.execute(
                "INSERT INTO map_markers "
                "(timestamp, experiment_id, sample_id, kind, skill_name, "
                "x_m, y_m, w_m, h_m, angle_deg, label, status, source, meta, "
                "coord_epoch) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(), experiment_id, sample_id, kind,
                 skill_name, x_m, y_m, w_m, h_m, float(angle_deg or 0.0),
                 label, status, source, meta_json, epoch),
            )
            return int(cur.lastrowid)

    @staticmethod
    def _scope_clauses(
        experiment_id: str | None, sample_id: str | None
    ) -> tuple[list[str], list]:
        """Scope filter shared by every map-marker query, so the generation
        count can never be scoped differently from the rows it counts."""
        clauses: list[str] = []
        params: list = []
        if experiment_id is not None:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if sample_id is not None:
            clauses.append("sample_id = ?")
            params.append(sample_id)
        return clauses, params

    def _count_coarse_moves(
        self, conn: sqlite3.Connection,
        experiment_id: str | None, sample_id: str | None,
    ) -> int:
        clauses, params = self._scope_clauses(experiment_id, sample_id)
        clauses.append("kind = 'coarse_move'")
        where = " WHERE " + " AND ".join(clauses)
        row = conn.execute(
            f"SELECT COUNT(*) FROM map_markers{where}", params
        ).fetchone()
        return int(row[0]) if row else 0

    def current_epoch(
        self,
        experiment_id: str | None = None,
        sample_id: str | None = None,
    ) -> int:
        """Current coordinate generation for a scope = number of lateral coarse
        moves recorded in it. Cheap (COUNT over the scope index)."""
        with self._connect() as conn:
            return self._count_coarse_moves(conn, experiment_id, sample_id)

    def last_coarse_move_timestamp(
        self,
        experiment_id: str | None = None,
        sample_id: str | None = None,
    ) -> str | None:
        """ISO timestamp of the most recent lateral coarse move, or None.

        The boundary between coordinate generations expressed in wall time, so
        artefacts that are NOT map markers — saved ``.sxm`` files, whose only
        link to a generation is when they were written — can be sorted into
        "before the stage moved" and "after". Queried rather than derived from a
        row window: ``get_markers`` truncates, and a boundary that silently
        disappears would mark every stale scan as current."""
        with self._connect() as conn:
            clauses, params = self._scope_clauses(experiment_id, sample_id)
            clauses.append("kind = 'coarse_move'")
            where = " WHERE " + " AND ".join(clauses)
            row = conn.execute(
                f"SELECT timestamp FROM map_markers{where} ORDER BY id DESC LIMIT 1",
                params,
            ).fetchone()
            return str(row[0]) if row and row[0] else None

    def get_markers(
        self,
        experiment_id: str | None = None,
        sample_id: str | None = None,
        limit: int = 2000,
        coord_epoch: int | None = None,
    ) -> list[dict]:
        """Return map markers (chronological, oldest first).

        Scoped to one experiment (and optionally one sample) when given;
        otherwise the most recent *limit* markers across all. meta is parsed
        back to a dict.

        ``coord_epoch`` filters to one coordinate generation. Asking for
        generation 0 also returns legacy rows whose column is NULL — a legacy
        row predates the column, and a database with no ``coarse_move`` rows has
        only ever had one coordinate system."""
        clauses, params = self._scope_clauses(experiment_id, sample_id)
        if coord_epoch is not None:
            if int(coord_epoch) == 0:
                clauses.append("(coord_epoch = 0 OR coord_epoch IS NULL)")
            else:
                clauses.append("coord_epoch = ?")
                params.append(int(coord_epoch))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM map_markers{where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        out = [dict(r) for r in rows]
        out.reverse()  # chronological (oldest first)
        for r in out:
            try:
                r["meta"] = json.loads(r.get("meta") or "{}")
            except Exception:
                r["meta"] = {}
        return out

    # ── Documents（索引；权威在实验文件夹） ──────────────────────────

    def upsert_document(self, meta: dict, related: list[str] | None = None) -> None:
        """写/更新一行文档索引 + 它的关联实验集合。

        ``INSERT OR REPLACE`` 在这里是对的（而在 ``plans`` 那里是 bug）：这张表
        是**索引**，一行就是 doc.json 当前状态的投影，没有历史可丢 —— 版本历史
        在 ``document_versions`` 和 ``versions.jsonl`` 里。
        """
        doc_id = str(meta.get("doc_id") or "").strip()
        if not doc_id:
            return
        tv = meta.get("target_version")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(doc_id, experiment_id, sample_id, kind, title, dir_name, root_kind, "
                " created_at, created_by, conversation_id, run_id, target_doc_id, "
                " target_version, latest_version, updated_at, legacy_stem) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id, meta.get("experiment_id"), meta.get("sample_id"),
                    str(meta.get("kind") or "experiment_report"),
                    str(meta.get("title") or ""), str(meta.get("dir_name") or ""),
                    str(meta.get("root_kind") or "experiment"),
                    str(meta.get("created_at") or ""), str(meta.get("created_by") or ""),
                    meta.get("conversation_id"), meta.get("run_id"),
                    meta.get("target_doc_id"),
                    int(tv) if tv not in (None, "") else None,
                    int(meta.get("latest_version") or 0),
                    str(meta.get("updated_at") or ""), meta.get("legacy_stem"),
                ),
            )
            if related is not None:
                conn.execute("DELETE FROM document_links WHERE doc_id = ?", (doc_id,))
                for eid in related:
                    e = str(eid or "").strip()
                    if e:
                        conn.execute(
                            "INSERT OR IGNORE INTO document_links (doc_id, experiment_id) "
                            "VALUES (?, ?)", (doc_id, e))

    def insert_document_version(self, doc_id: str, row: dict) -> None:
        """登记一个版本。``INSERT OR IGNORE`` —— 版本行不可变，重复登记是幂等的
        （reindex 会把已有的行再喂一遍）。"""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO document_versions "
                "(doc_id, version, rel_path, sha256, words, created_at, created_by, "
                " conversation_id, run_id, note) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id, int(row.get("version") or 0), str(row.get("rel_path") or ""),
                    str(row.get("sha256") or ""), int(row.get("words") or 0),
                    str(row.get("created_at") or ""), str(row.get("created_by") or ""),
                    row.get("conversation_id"), row.get("run_id"),
                    str(row.get("note") or ""),
                ),
            )

    def get_document(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?",
                               (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, *, experiment_id: str | None = None,
                       kind: str | None = None, include_unfiled: bool = True,
                       include_related: bool = True, limit: int = 500) -> list[dict]:
        """文档索引行，最近更新在前。

        ``experiment_id`` 给定时同时收主归属与（可选）关联的行，并给每行标注
        ``relation`` ∈ ``primary|related``。
        """
        clauses: list[str] = []
        params: list = []
        if experiment_id:
            if include_related:
                clauses.append(
                    "(experiment_id = ? OR doc_id IN "
                    "(SELECT doc_id FROM document_links WHERE experiment_id = ?))")
                params += [experiment_id, experiment_id]
            else:
                clauses.append("experiment_id = ?")
                params.append(experiment_id)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if not include_unfiled:
            clauses.append("root_kind != 'unfiled'")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents{where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        out = [dict(r) for r in rows]
        if experiment_id:
            for r in out:
                r["relation"] = "primary" if r.get("experiment_id") == experiment_id else "related"
        return out

    def get_document_versions(self, doc_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM document_versions WHERE doc_id = ? ORDER BY version",
                (doc_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def document_links(self, doc_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT experiment_id FROM document_links WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
        return [str(r["experiment_id"]) for r in rows]

    def count_documents(self, experiment_id: str) -> int:
        """该实验的文档数（含关联）—— 给实验列表的角标用，不必拉全表。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE experiment_id = ? "
                "OR doc_id IN (SELECT doc_id FROM document_links WHERE experiment_id = ?)",
                (experiment_id, experiment_id),
            ).fetchone()
        return int((row["n"] if row else 0) or 0)
