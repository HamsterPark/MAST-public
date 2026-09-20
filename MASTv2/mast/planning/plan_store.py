"""Plan persistence: SQLite metadata + Markdown human-readable documents.

Expert mode plans are stored in two forms:
  1. SQLite (in experiments DB): structured tracking (status, progress)
  2. Markdown: 实验文件夹里的一个**文档**（``plans/<doc-dir>/``），见下

Resume flow:
  1. SQLite: query last active plan
  2. Read Markdown: inject into system prompt as context
  3. LLM continues from current_phase/current_step

定义 vs 进度（2026-07-29，设计 §3.7）
------------------------------------

旧实现只有一个 ``plan_<id>.md``，``save()`` 和 ``update_progress()`` 都往它裸
``write_text``：**推进一个阶段就吃掉上一份计划**，而且写到一半崩就是半个文件。

现在拆成两件事：

* **定义**（name / goal / phases 结构）→ ``plans/<doc-dir>/vNNN.md``，永不覆盖。
  只有定义**内容真的变了**才发新版本（``_sync_definition`` 会先比一次正文）。
* **进度**（当前阶段/步骤/状态）→ ``progress.jsonl`` 原子追加 + ``progress.md``
  原子替换。进度是事件和视图，不是版本 —— 一个 10 阶段的过夜计划推完约 30 次
  进度写，每次发版本就是版本爆炸。

计划没有实验归属（或指向不存在的实验 id）时退回本 store 自己目录下的
``plan_<id>.md``，同样改成原子替换。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PlanStatus(str, Enum):
    DRAFT = "draft"           # LLM generated, pending user approval
    APPROVED = "approved"     # User approved, ready to execute
    RUNNING = "running"       # Currently executing
    PAUSED = "paused"         # Paused (can resume)
    COMPLETED = "completed"   # All phases done
    ABORTED = "aborted"       # User aborted


@dataclass
class PlanPhase:
    id: str                           # "surface_prep"
    name: str                         # "表面制备"
    steps: list[dict] = field(default_factory=list)  # [{"skill": ..., "params": ...}]
    success_criteria: str = ""
    on_fail: str = "retry"            # "retry" / "skip" / "abort"
    status: str = "pending"           # "pending" / "running" / "done" / "failed" / "skipped"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "steps": self.steps,
            "success_criteria": self.success_criteria,
            "on_fail": self.on_fail,
            "status": self.status,
        }

    @staticmethod
    def from_dict(d: dict) -> "PlanPhase":
        return PlanPhase(
            id=d.get("id", ""),
            name=d.get("name", ""),
            steps=d.get("steps", []),
            success_criteria=d.get("success_criteria", ""),
            on_fail=d.get("on_fail", "retry"),
            status=d.get("status", "pending"),
        )


@dataclass
class ExperimentPlan:
    plan_id: str
    experiment_id: str = ""
    sample_id: str = ""
    name: str = ""
    goal: str = ""
    phases: list[PlanPhase] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    current_phase_idx: int = 0
    current_step_idx: int = 0
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "experiment_id": self.experiment_id,
            "sample_id": self.sample_id,
            "name": self.name,
            "goal": self.goal,
            "phases": [p.to_dict() for p in self.phases],
            "status": self.status.value,
            "current_phase_idx": self.current_phase_idx,
            "current_step_idx": self.current_step_idx,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExperimentPlan":
        return ExperimentPlan(
            plan_id=d.get("plan_id", ""),
            experiment_id=d.get("experiment_id", ""),
            sample_id=d.get("sample_id", ""),
            name=d.get("name", ""),
            goal=d.get("goal", ""),
            phases=[PlanPhase.from_dict(p) for p in d.get("phases", [])],
            status=PlanStatus(d.get("status", "draft")),
            current_phase_idx=d.get("current_phase_idx", 0),
            current_step_idx=d.get("current_step_idx", 0),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            notes=d.get("notes", ""),
        )

    @property
    def total_steps(self) -> int:
        return sum(len(p.steps) for p in self.phases)

    @property
    def completed_steps(self) -> int:
        count = 0
        for i, phase in enumerate(self.phases):
            if phase.status == "done":
                count += len(phase.steps)
            elif phase.status == "running" and i == self.current_phase_idx:
                count += self.current_step_idx
        return count


class PlanStore:
    """Manage experiment plans in SQLite + Markdown."""

    def __init__(self, db_path: str | Path, plans_dir: str | Path | None = None):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._plans_dir = Path(plans_dir) if plans_dir else self._db_path.parent / "plans"
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _connect(self):
        import sqlite3
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id         TEXT PRIMARY KEY,
                    experiment_id   TEXT,
                    sample_id       TEXT,
                    name            TEXT NOT NULL,
                    goal            TEXT DEFAULT '',
                    definition      TEXT NOT NULL,
                    status          TEXT DEFAULT 'draft',
                    current_phase   INTEGER DEFAULT 0,
                    current_step    INTEGER DEFAULT 0,
                    notes           TEXT DEFAULT '',
                    created_at      TEXT,
                    updated_at      TEXT
                )
            """)
            # 2026-07-29：计划的 markdown 变成实验文件夹里的一个**文档**
            # （kind=experiment_plan）。这一列把 plan 行钉到那个文档上。
            # 设计：docs/v2/design/document_and_library_management.md §3.7
            cols = {r[1] for r in conn.execute("PRAGMA table_info(plans)").fetchall()}
            if "doc_id" not in cols:
                conn.execute("ALTER TABLE plans ADD COLUMN doc_id TEXT")

    def save(self, plan: ExperimentPlan) -> str:
        """Save plan to SQLite + generate Markdown. Returns plan_id."""
        if not plan.plan_id:
            plan.plan_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        if not plan.created_at:
            plan.created_at = now
        plan.updated_at = now

        definition = json.dumps(
            [p.to_dict() for p in plan.phases], ensure_ascii=False,
        )

        # doc_id 必须显式带上：``INSERT OR REPLACE`` 是**删旧行 + 插新行**，
        # 列清单里没写的列一律回落到默认值（这里就是 NULL）。漏了它的后果不是
        # 报错而是静默失忆 —— 每次 save 都以为这个计划还没有文档，于是**另立一份
        # 新文档**，同一个计划在 reports 里越攒越多。
        existing_doc_id = self.doc_id_for(plan.plan_id)

        with closing(self._connect()) as conn, conn:
            conn.execute("""
                INSERT OR REPLACE INTO plans
                (plan_id, experiment_id, sample_id, name, goal, definition,
                 status, current_phase, current_step, notes, created_at, updated_at,
                 doc_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan.plan_id, plan.experiment_id, plan.sample_id,
                plan.name, plan.goal, definition,
                plan.status.value, plan.current_phase_idx, plan.current_step_idx,
                plan.notes, plan.created_at, plan.updated_at,
                existing_doc_id,
            ))

        self._sync_definition(plan)
        self._write_progress_view(plan)
        return plan.plan_id

    def load(self, plan_id: str) -> ExperimentPlan | None:
        """Load plan from SQLite."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM plans WHERE plan_id = ?", (plan_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_plan(dict(row))

    def get_active(self) -> ExperimentPlan | None:
        """Get the most recent running/paused plan."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM plans WHERE status IN ('running', 'paused') "
                "ORDER BY updated_at DESC LIMIT 1",
            ).fetchone()
        if not row:
            return None
        return self._row_to_plan(dict(row))

    def update_progress(
        self, plan_id: str, phase_idx: int, step_idx: int,
        phase_status: str = "running",
    ) -> None:
        """Update plan progress in SQLite + Markdown.

        Persists current_phase/current_step AND the per-phase status into the
        ``definition`` JSON column so a later ``load()`` round-trips the phase
        status (previously the phase_status was only reflected in Markdown and
        was silently lost on reload).
        """
        now = datetime.now().isoformat()
        # Load first so we can mutate the phase status inside `definition`.
        plan = self.load(plan_id)
        if plan is None:
            return
        if 0 <= phase_idx < len(plan.phases):
            plan.phases[phase_idx].status = phase_status
        plan.current_phase_idx = phase_idx
        plan.current_step_idx = step_idx
        plan.updated_at = now

        definition = json.dumps(
            [p.to_dict() for p in plan.phases], ensure_ascii=False,
        )
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE plans SET definition=?, current_phase=?, current_step=?, "
                "updated_at=? WHERE plan_id=?",
                (definition, phase_idx, step_idx, now, plan_id),
            )

        # 进度**不发版本**：它是事件 + 视图。一个 10 阶段的过夜计划推完约 30 次
        # 进度写（advance_plan 每阶段 update_progress ×2 + update_status ×1），
        # 每次都发版本就是版本爆炸。见 §3.7 的裁决。
        self._append_progress_event(plan, {
            "op": "progress", "at": now, "phase_idx": phase_idx,
            "step_idx": step_idx, "phase_status": phase_status,
            "plan_status": plan.status.value,
        })
        self._write_progress_view(plan)

    def update_status(
        self, plan_id: str, status: PlanStatus, notes: str | None = None,
    ) -> None:
        """Update plan status (and optionally notes).

        When ``notes`` is provided it is persisted to the ``notes`` column.
        abort_plan / pause_plan set an explanatory note; without persisting it
        here the note was silently dropped.
        """
        now = datetime.now().isoformat()
        with closing(self._connect()) as conn, conn:
            if notes is not None:
                conn.execute(
                    "UPDATE plans SET status=?, notes=?, updated_at=? WHERE plan_id=?",
                    (status.value, notes, now, plan_id),
                )
            else:
                conn.execute(
                    "UPDATE plans SET status=?, updated_at=? WHERE plan_id=?",
                    (status.value, now, plan_id),
                )
        plan = self.load(plan_id)
        if plan is not None:
            self._append_progress_event(plan, {
                "op": "status", "at": now, "plan_status": status.value,
                "notes": notes or "",
            })
            self._write_progress_view(plan)

    def list_plans(self, experiment_id: str | None = None, limit: int = 20) -> list[dict]:
        """List plan summaries."""
        with closing(self._connect()) as conn:
            if experiment_id:
                rows = conn.execute(
                    "SELECT plan_id, name, goal, status, current_phase, current_step, "
                    "created_at, updated_at FROM plans "
                    "WHERE experiment_id=? ORDER BY updated_at DESC LIMIT ?",
                    (experiment_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT plan_id, name, goal, status, current_phase, current_step, "
                    "created_at, updated_at FROM plans "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def doc_id_for(self, plan_id: str) -> str | None:
        """这个计划对应的文档 id（``kind=experiment_plan``）。没同步过返回 None。"""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT doc_id FROM plans WHERE plan_id = ?",
                               (plan_id,)).fetchone()
        return (row["doc_id"] or None) if row else None

    def markdown_path(self, plan_id: str) -> Path:
        """计划正文的路径。

        同步过文档的计划返回**定义版本**文件（``plans/<doc-dir>/vNNN.md``）；没有
        实验归属、因此没有文档的计划返回本 store 自己目录下的旧路径（兼容）。
        """
        doc_id = self.doc_id_for(plan_id)
        if doc_id:
            try:
                from mast.documents import store as _docstore
                entry = _docstore().get(doc_id)
                if entry is not None:
                    p = entry.version_path()
                    if p is not None:
                        return p
            except Exception:  # noqa: BLE001 — 退回旧路径，不让取路径这种事抛
                pass
        return self._plans_dir / f"plan_{plan_id}.md"

    # ── Internal helpers ──────────────────────────────────────────────

    def _row_to_plan(self, row: dict) -> ExperimentPlan:
        phases_data = json.loads(row.get("definition", "[]"))
        phases = [PlanPhase.from_dict(p) for p in phases_data]
        return ExperimentPlan(
            plan_id=row["plan_id"],
            experiment_id=row.get("experiment_id", ""),
            sample_id=row.get("sample_id", ""),
            name=row.get("name", ""),
            goal=row.get("goal", ""),
            phases=phases,
            status=PlanStatus(row.get("status", "draft")),
            current_phase_idx=row.get("current_phase", 0),
            current_step_idx=row.get("current_step", 0),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
            notes=row.get("notes", ""),
        )

    # ── 文档同步：定义是版本，进度是事件 + 视图 ──────────────────────
    #
    # 设计文档 §3.7 的裁决。旧实现只有一个 ``plan_<id>.md``，``save()`` 和
    # ``update_progress()`` 都往它 ``write_text``（非原子，崩在中途就是半个文件），
    # 于是**推进一个阶段就吃掉上一份计划**。DB 侧也一样：``INSERT OR REPLACE``
    # 不留历史行。
    #
    # 拆开之后：
    #   定义（name/goal/phases 结构）→ ``plans/<doc-dir>/vNNN.md``，永不覆盖
    #   进度（当前阶段/步骤/状态）  → ``progress.jsonl`` 追加 + ``progress.md``
    #                                 原子替换（视图，丢了随时能重生成）
    # 先例：chat 导出的「jsonl 是增量载体，md 是它的视图」。

    def _doc_target(self, plan: ExperimentPlan):
        """能不能把这个计划同步成文档。返回 store 或 None。

        条件是**实验行真的存在** —— 计划没有实验归属（或指向一个不存在的实验 id，
        测试里很常见）时不碰实验文件夹，退回本 store 自己的目录。否则一个用假 id
        构造的 PlanStore 会往真实数据根里写东西。
        """
        if not (plan.experiment_id or "").strip():
            return None
        try:
            from mast.documents.paths import exp_dir_for
            if exp_dir_for(plan.experiment_id, create=False) is None:
                return None
            from mast.documents import store as _docstore
            return _docstore()
        except Exception as exc:  # noqa: BLE001
            logger.debug("plan doc sync unavailable: %r", exc)
            return None

    def _sync_definition(self, plan: ExperimentPlan) -> None:
        """把计划定义同步成文档版本。**内容没变就不发新版本。**"""
        s = self._doc_target(plan)
        if s is None:
            self._write_legacy_markdown(plan)
            return
        text = self._render_definition(plan)
        doc_id = self.doc_id_for(plan.plan_id) or ""
        if doc_id:
            entry = s.get(doc_id)
            if entry is not None and (entry.read_text() or "") == text:
                return  # 定义没动 —— 这是「进度更新不发版本」的最后一道闸
        res = s.save(text=text, kind="experiment_plan", title=plan.name or "实验计划",
                     doc_id=doc_id, experiment_id=plan.experiment_id or None,
                     sample_id=plan.sample_id or None,
                     created_by="agent:experiment_design",
                     note="计划定义" if not doc_id else "计划定义修订")
        if res.ok and res.doc_id and res.doc_id != doc_id:
            with closing(self._connect()) as conn, conn:
                conn.execute("UPDATE plans SET doc_id=? WHERE plan_id=?",
                             (res.doc_id, plan.plan_id))
        if not res.ok:
            logger.warning("plan definition doc save failed (%s): %s",
                           plan.plan_id, res.error)
            self._write_legacy_markdown(plan)

    def _append_progress_event(self, plan: ExperimentPlan, event: dict) -> None:
        s = self._doc_target(plan)
        doc_id = self.doc_id_for(plan.plan_id)
        if s is None or not doc_id:
            return
        s.append_event(doc_id, "progress.jsonl", event)

    def _write_progress_view(self, plan: ExperimentPlan) -> None:
        s = self._doc_target(plan)
        doc_id = self.doc_id_for(plan.plan_id)
        if s is None or not doc_id:
            self._write_legacy_markdown(plan)
            return
        s.write_view(doc_id, "progress.md", self._render_progress(plan))

    def _write_legacy_markdown(self, plan: ExperimentPlan) -> None:
        """没有实验归属时的退路：写本 store 目录下的 ``plan_<id>.md``。

        原子替换（``.part`` + ``os.replace``），不再裸 ``write_text`` —— 旧实现崩在
        写中途留下的是半个文件。
        """
        path = self._plans_dir / f"plan_{plan.plan_id}.md"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".md.part")
            tmp.write_text(self._render_progress(plan), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("plan markdown write failed (%s): %r", path, exc)

    def _render_definition(self, plan: ExperimentPlan) -> str:
        """**只有定义**：目标、阶段、步骤、成功标准。

        刻意不含任何进度痕迹（没有勾选框、没有状态 emoji、没有「当前步骤」箭头、
        没有 updated_at 页脚）—— 只要进度会改变这段文本，每次推进就会生成一个新
        版本，而那正是要避免的版本爆炸。
        """
        lines = [f"# {plan.name}", ""]
        if plan.goal:
            lines += [f"**目标**：{plan.goal}", ""]
        for i, phase in enumerate(plan.phases):
            lines.append(f"## Phase {i + 1}: {phase.name}")
            for step in phase.steps:
                skill = step.get("skill", "")
                params = step.get("params", {})
                notes = step.get("notes", "")
                if skill:
                    ps = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
                    step_text = f"`{skill}({ps})`"
                else:
                    step_text = f"[{step.get('action', 'manual')}]"
                lines.append(f"- {step_text}" + (f" — {notes}" if notes else ""))
            if phase.success_criteria:
                lines.append(f"- **成功标准**：{phase.success_criteria}")
            lines.append("")
        lines += [
            "---",
            "",
            f"计划 ID：`{plan.plan_id}`　创建：{plan.created_at}",
            "",
            "> 这是计划的**定义**。执行进度在同目录的 `progress.md`（视图）与",
            "> `progress.jsonl`（事件日志）里 —— 定义被修订才会产生新版本。",
        ]
        return "\n".join(lines)

    def _render_progress(self, plan: ExperimentPlan) -> str:
        """进度视图。会被反复原子替换 —— 它是事件日志的渲染结果，不是版本。"""
        lines: list[str] = []
        lines.append(f"# {plan.name}")
        lines.append(f"目标: {plan.goal}")
        lines.append("")

        for i, phase in enumerate(plan.phases):
            # Status emoji
            if phase.status == "done":
                emoji = "\u2705"  # ✅
            elif phase.status == "running" or i == plan.current_phase_idx and plan.status == PlanStatus.RUNNING:
                emoji = "\U0001f504"  # 🔄
            elif phase.status in ("failed", "skipped"):
                emoji = "\u274c"  # ❌
            else:
                emoji = "\u23f3"  # ⏳

            lines.append(f"## Phase {i + 1}: {phase.name} {emoji}")

            for j, step in enumerate(phase.steps):
                skill = step.get("skill", "")
                params = step.get("params", {})
                notes = step.get("notes", "")

                # Determine if step is done
                is_done = (
                    phase.status == "done"
                    or (i == plan.current_phase_idx and j < plan.current_step_idx)
                )
                is_current = (i == plan.current_phase_idx and j == plan.current_step_idx
                              and plan.status == PlanStatus.RUNNING)

                checkbox = "[x]" if is_done else "[ ]"
                if skill:
                    ps = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
                    step_text = f"{skill}({ps})"
                else:
                    step_text = f"[{step.get('action', 'manual')}]"

                marker = " \u2190 当前步骤" if is_current else ""
                note_text = f" \u2014 {notes}" if notes else ""
                lines.append(f"- {checkbox} {step_text}{note_text}{marker}")

            if phase.success_criteria:
                lines.append(f"- 成功标准: {phase.success_criteria}")

            lines.append("")

        # Footer
        total = plan.total_steps
        done = plan.completed_steps
        lines.append(f"状态: {plan.status.value} | "
                      f"Phase {plan.current_phase_idx + 1}/{len(plan.phases)} | "
                      f"Step {done}/{total}")
        lines.append(f"更新: {plan.updated_at}")
        return "\n".join(lines)
