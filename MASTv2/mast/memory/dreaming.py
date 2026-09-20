"""Dreaming — async background consolidation of past experiments into memory.

Design: docs/v2/design/agentic-cognition.md §3.

Like sleep-time memory consolidation: a low-priority background pass reads the
experiment record (experiments, actions, conversation-phase summaries) plus the
existing memory, and distils RECURRING patterns / protocols / hypotheses into
new ``kind="dream"|"insight"|"hypothesis"`` memory entries.

Honesty (hard rule): every dream output is **visibly labelled "AI 做梦推断(非
实测)"** and lands ONLY in the memory store — it never writes the experiment
record or the knowledge base. With no LLM key the consolidator falls back to a
dependency-free rule-based pass so dreaming still produces (modest) value
offline; either way it is dedup'd so a dream isn't re-written every cycle.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

DREAM_TAG = "🌙 AI 做梦推断(非实测,仅供参考)"

# consolidator(context: dict) -> list[{path, title, content, kind}]
Consolidator = Callable[[dict], list]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _field(obj, key, default=None):
    """Read ``key`` from either a dict or a dataclass/record object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def gather_context(db_path, *, n_experiments: int = 8) -> dict:
    """Read a sample of recent experiment records for consolidation."""
    out: dict = {"experiments": [], "skill_counts": {}, "statuses": {},
                 "phase_summaries": []}
    try:
        from mast.logging.storage import ExperimentStorage
        st = ExperimentStorage(str(db_path))
    except Exception as exc:  # pragma: no cover
        logger.debug("dreaming: storage open failed: %s", exc)
        return out
    try:
        exps = st.list_experiments_with_counts(limit=n_experiments)
    except Exception:
        try:
            exps = st.list_experiments(limit=n_experiments)
        except Exception:
            exps = []
    for e in exps:
        eid = _field(e, "id")
        status = _field(e, "status", "?") or "?"
        out["experiments"].append({
            "id": eid, "name": _field(e, "name", ""),
            "status": status,
            "goal": _field(e, "goal_text", "") or _field(e, "goal", ""),
        })
        out["statuses"][status] = out["statuses"].get(status, 0) + 1
        try:
            actions = st.get_actions(eid) if eid else []
        except Exception:
            actions = []
        for a in actions:
            sk = _field(a, "skill_name", "?") or "?"
            out["skill_counts"][sk] = out["skill_counts"].get(sk, 0) + 1
    # conversation phase summaries (if the sharding table exists)
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT title, summary FROM conversation_phase WHERE summary != '' "
            "ORDER BY id DESC LIMIT 12").fetchall()
        conn.close()
        out["phase_summaries"] = [{"title": r[0], "summary": r[1]} for r in rows]
    except Exception:
        pass
    return out


def rule_based_consolidate(context: dict) -> list:
    """Offline consolidation: surface the most-used skills, status mix, and a
    recap of recent phase summaries. Modest but real (no fabrication)."""
    exps = context.get("experiments", [])
    if not exps:
        return []
    skills = sorted(context.get("skill_counts", {}).items(),
                    key=lambda kv: kv[1], reverse=True)[:8]
    statuses = context.get("statuses", {})
    lines = [DREAM_TAG, "",
             f"## 跨 {len(exps)} 个近期实验的固化复盘", ""]
    if statuses:
        lines.append("**实验状态分布**: " + ", ".join(f"{k}×{v}" for k, v in statuses.items()))
    if skills:
        lines.append("**高频技能**: " + ", ".join(f"{s}×{n}" for s, n in skills))
        top = skills[0][0]
        lines.append(f"\n**模式观察(推断)**: `{top}` 是最常用的操作,"
                     f"值得固化其成功参数为协议;失败状态实验({statuses.get('failed', 0)} 个)"
                     f"可回顾共因。")
    ph = context.get("phase_summaries", [])
    if ph:
        lines.append("\n**近期对话阶段摘要**:")
        for p in ph[:5]:
            lines.append(f"- {p.get('title') or '阶段'}: {(p.get('summary') or '')[:120]}")
    return [{"path": "dreams/consolidation.md",
             "title": "做梦复盘(规则式)", "kind": "dream",
             "content": "\n".join(lines)}]


class DreamingService:
    """Background consolidation pass. Best-effort, low-priority, cancelable."""

    def __init__(self, db_path, memory_store, *, consolidator: Consolidator | None = None,
                 interval_s: float = 1800.0, namespace: str = "global"):
        self._db_path = db_path
        self._memory = memory_store
        self._consolidate = consolidator or rule_based_consolidate
        self._interval = max(60.0, float(interval_s))
        self._namespace = namespace
        self._running = False
        self._thread: threading.Thread | None = None
        self._should_dream: Callable[[], bool] = lambda: True

    def set_should_dream(self, predicate: Callable[[], bool]) -> None:
        """Inject an idle predicate — the app passes 'no active task' so dreaming
        only runs when the system is quiet."""
        self._should_dream = predicate

    def dream_once(self) -> list[dict]:
        """Run one consolidation cycle now. Returns the memory entries written."""
        written: list[dict] = []
        try:
            ctx = gather_context(self._db_path)
            entries = self._consolidate(ctx) or []
        except Exception as exc:
            logger.debug("dream cycle failed: %s", exc)
            return written
        for e in entries:
            content = e.get("content", "")
            if not content.strip():
                continue
            # honesty: ensure the dream tag is present
            if DREAM_TAG not in content:
                content = f"{DREAM_TAG}\n\n{content}"
            path = e.get("path") or f"dreams/{_content_hash(content)}.md"
            # dedup: skip if an identical dream already exists at this path
            existing = None
            try:
                existing = self._memory.read(self._namespace, path)
            except Exception:
                existing = None
            if existing and _content_hash(existing.get("content", "")) == _content_hash(content):
                continue
            try:
                self._memory.write(self._namespace, path, content,
                                   title=e.get("title", "做梦"),
                                   kind=e.get("kind", "dream"), author="dream",
                                   tags=["dream"])
                written.append({"path": path, "title": e.get("title", "")})
            except Exception as exc:
                logger.debug("dream write failed: %s", exc)
        if written:
            logger.info("Dreaming: consolidated %d memory entr(ies)", len(written))
        return written

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="DreamingService")
        self._thread.start()
        logger.info("DreamingService started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        # initial settle before the first dream
        slept = 0.0
        while self._running:
            time.sleep(1.0)
            slept += 1.0
            if slept >= self._interval:
                slept = 0.0
                try:
                    if self._should_dream():
                        self.dream_once()
                except Exception as exc:  # pragma: no cover
                    logger.debug("dream loop error: %s", exc)


__all__ = ["DreamingService", "gather_context", "rule_based_consolidate", "DREAM_TAG"]
