"""Experiment-Design agent tool list.

Phase 5 real implementation. XD does NOT wrap instrument skills as tools — it
reads the skill catalog via describe_skills() for planning purposes, then
hands the plan to IC which executes the skills.

Tools:
  1. describe_skills(category, safety_level, tag) — Markdown catalog from registry
  2. lookup_sample(query)                          — v2 knowledge-base workflow lookup
  3. query_past_experiments(sample_type, max_n)    — real SQLite experiment-log query
  4. buffer tools (if buf supplied)                — read live tip status for context
  5. handoff_to_supervisor                         — return control to orchestrator
  6. handoff_to_instrument_control                 — send plan for execution

Note: XD reads the registry but does NOT call wrap_skill(). Only IC does that.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from mast.agents._shared.buffer_tools import make_buffer_tools
from mast.agents._shared.handoff import make_handoff
from mast.core.registry import SkillRegistry

if TYPE_CHECKING:
    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# 技能目录（``describe_skills`` 要的那份）
#
# 2026-08-27 从 ``graph.py`` 搬到这里。它构建一个 ``SkillRegistry`` —— **一行
# langgraph 都没有**，住在图文件里只是因为那张图是第一个需要它的人。代价在退出
# langgraph 时现形：``agentruntime/assembly.py`` 为了拿这份目录，得
# ``importlib.import_module("mast.agents.<id>.graph")``，也就是**新的通用装配器
# 依赖着它正要替代的那些图**。
#
# 搬到 ``tools.py`` 还让两个 agent 对称起来：IC 的同类函数
# ``discover_instrument_skills`` 本来就住在它的 ``tools.py`` 里。
# ─────────────────────────────────────────────────────────────────────

#: XD 浏览技能目录时读的包。XD **不**把它们包成工具 —— 那是 IC 的活。
_CATALOG_PACKAGES = (
    "mast.skills.builtins",
    "mast.skills.composite",
)


def discover_xd_catalog(
        packages: tuple[str, ...] = _CATALOG_PACKAGES) -> SkillRegistry:
    """Build a SkillRegistry so XD can browse available skills.

    This is lighter than IC's discover because we only need metadata (for
    ``describe_skills()``), not instances (for execution).
    """
    registry = SkillRegistry()
    n = registry.discover(*packages)
    logger.info("experiment_design: discovered %d skills for catalog", n)
    return registry


#: 旧名字，保留 —— 既有测试断言这两个是**同一个对象**。
_discover_catalog = discover_xd_catalog


# ─────────────────────────────────────────────────────────────────────
# Experiment-log path helpers
#
# These mirror the resolution logic used by the Paper-Writing agent's
# query_experiment_records tool. The implementation is COPIED here (not
# imported) to honour the v2 agent-boundary invariant: experiment_design
# must not import from mast.agents.paper_writing. Only mast.logging.storage
# (a non-agent module) is shared — that import is permitted.
# ─────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "mast").is_dir() and (p / "MASTv2").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[4]


def _experiment_db_path() -> Path:
    """Resolve the SQLite experiment-log DB path — the LIVE store the app writes to.

    Order: MAST_EXPERIMENT_DB env → the ACTIVE ExperimentLog's real DB → the config
    store path → the legacy <repo_root>/data/experiments.db fallback. The old code
    ONLY had the last fallback, which never matches the real store
    (experiments/mast_experiments.db), so query_past_experiments always failed with
    'experiment log not found' in dev AND in the packaged app (2026-07-01 fix).
    """
    env = os.environ.get("MAST_EXPERIMENT_DB", "").strip()
    if env:
        return Path(env).expanduser()
    try:
        from mast.logging.experiment_log import get_active_log
        db = getattr(getattr(get_active_log(), "_storage", None), "_db_path", None)
        if db:
            return Path(db)
    except Exception:  # noqa: BLE001
        pass
    try:
        from mast.config import MASTConfig
        return Path(MASTConfig().db_path)
    except Exception:  # noqa: BLE001
        pass
    return _repo_root() / "data" / "experiments.db"


def _truncate(s: str, n: int = 1500) -> str:
    return s if len(s) <= n else s[:n] + "...[truncated]"


# ─────────────────────────────────────────────────────────────────────
# Tool factories
# ─────────────────────────────────────────────────────────────────────

def describe_skills_tool(registry: SkillRegistry):
    """Return a @tool that calls registry.describe_skills() and returns Markdown.

    The tool accepts optional filter arguments mirroring describe_skills()
    signature so the LLM can narrow the catalog (e.g. by category or safety_level).
    """

    @tool("describe_skills")
    def describe_skills(
        category: str | None = None,
        safety_level: str | None = None,
        tag: str | None = None,
    ) -> str:
        """Browse the instrument skill catalog.

        Returns a Markdown listing of available skills. Optionally filter by:
          category     — e.g. "scan", "spectroscopy", "motion", "tip_management"
          safety_level — "AUTO", "CONFIRM", or "DANGEROUS"
          tag          — any skill tag string

        Use this to discover skill names and parameters before writing an
        ExperimentPlan. Skill names in the plan must match exactly.
        """
        from mast.core.types import SafetyLevel  # avoid top-level v1 import

        sl_enum = None
        if safety_level is not None:
            try:
                sl_enum = SafetyLevel[safety_level.upper()]
            except (KeyError, AttributeError):
                logger.debug("Unknown safety_level filter %r; ignoring", safety_level)

        md = registry.describe_skills(
            category=category,
            safety_level=sl_enum,
            tag=tag,
        )
        logger.debug("describe_skills called (cat=%r sl=%r tag=%r) → %d chars",
                     category, safety_level, tag, len(md))
        return md

    return describe_skills


def lookup_sample_tool():
    """Return a @tool that looks up sample domain knowledge from the v2 KB.

    Real implementation backed by ``mast.knowledge.match_material`` (fuzzy
    sample name → (category, material) resolver) and
    ``mast.knowledge.format_workflow_for_llm`` (compact workflow + parameters
    summary). The KB lives at MASTv2/mast/knowledge/<10 categories>.py and
    covers ~60 materials including Au(111), Si(111)-7x7, MoS2, NbSe2, etc.
    """

    @tool("lookup_sample")
    def lookup_sample(query: str) -> str:
        """Look up domain knowledge about a sample material.

        Args:
            query: Free-form text describing what to look up, e.g.
                   "Au(111) typical tunnelling parameters", "WSe2 defects",
                   "Si(111)-7x7 features", "kondo molecule", "金 (Chinese)".

        Returns the matched category + material name, completeness flag, and
        a compact workflow summary (~200-300 tokens) covering recommended
        bias / setpoint ranges, expected features, and typical phases.
        Returns a "no match" note if the query cannot be resolved.
        """
        try:
            from mast.knowledge import (
                match_material,
                get_completeness,
                format_workflow_for_llm,
            )
        except Exception as e:
            return f"lookup_sample: knowledge KB import failed ({type(e).__name__}: {e})"

        matched = match_material(query)
        if matched is None:
            return (
                f"No match found for '{query}'. "
                "Try a known material name (e.g. Au(111), Si(111)-7x7, WSe2) "
                "or category (e.g. clean_metal, semiconductor, 2d_material, "
                "topological, superconductor, magnetic_spm, molecular_adsorbate, "
                "thin_film, oxide_surface, on_surface_synthesis)."
            )
        type_id, material_name = matched
        completeness = get_completeness(type_id) or "?"
        try:
            workflow = format_workflow_for_llm(type_id, material_name)
        except Exception as e:  # pragma: no cover — defensive
            workflow = f"(format_workflow_for_llm failed: {type(e).__name__}: {e})"

        header = (
            f"Match: category={type_id}, material={material_name or '(category-level)'}, "
            f"completeness={completeness}\n"
        )
        return header + workflow

    return lookup_sample


def query_past_experiments_tool():
    """Return a @tool that queries prior experiment runs for a sample type.

    Real implementation backed by ``mast.logging.storage.ExperimentStorage``
    (the same SQLite experiment log read by the Paper-Writing agent). The DB
    path is taken from ``MAST_EXPERIMENT_DB`` env or defaults to
    ``<repo_root>/data/experiments.db``.

    The query logic is copied here (rather than imported from
    paper_writing) to satisfy the v2 agent-boundary invariant: only the
    non-agent ``mast.logging.storage`` module is shared.
    """

    @tool("query_past_experiments")
    def query_past_experiments(
        sample_type: str | None = None,
        max_n: int = 5,
    ) -> str:
        """Query past experiments for a given sample type.

        Args:
            sample_type: Material / sample descriptor to filter by, e.g. "HOPG",
                         "Au(111)", "MoS2". Matched (case-insensitively) against
                         the experiment name, goal text, and each run's sample
                         type / name. None returns the most recent N runs.
            max_n:       Maximum number of past runs to return (default 5).

        Returns a Markdown bulleted list of prior runs (id / name / status /
        sample count / start time) to inform the current plan and avoid
        repeating known-bad parameters. Returns a "no DB" note if the SQLite
        experiment log does not exist yet, and an honest "no records matched"
        note when the filter yields nothing.
        """
        limit = max(1, min(int(max_n), 200))

        db_path = _experiment_db_path()
        if not db_path.is_file():
            return (
                f"query_past_experiments: SQLite experiment log not found at "
                f"{db_path}. Set MAST_EXPERIMENT_DB or run an experiment first "
                "to create it. For now, rely on the operator's knowledge and "
                "lookup_sample() for context."
            )

        try:
            from mast.logging.storage import ExperimentStorage
        except Exception as e:
            return (
                "query_past_experiments: storage import failed "
                f"({type(e).__name__}: {e})"
            )

        try:
            store = ExperimentStorage(db_path)
            # Over-fetch when filtering so the post-filter can still return up
            # to `limit` matches; cap to keep the scan bounded.
            fetch_n = limit if not sample_type else min(200, max(limit, 50))
            rows = store.list_experiments_with_counts(limit=fetch_n)
        except Exception as e:
            return (
                "query_past_experiments: query failed "
                f"({type(e).__name__}: {e})"
            )

        if sample_type:
            needle = sample_type.strip().lower()

            def _matches(r: dict) -> bool:
                # Cheap fields first.
                if needle in (r.get("name", "") or "").lower():
                    return True
                if needle in (r.get("goal_text", "") or "").lower():
                    return True
                # Fall back to per-run sample rows (sample_type / name).
                try:
                    samples = store.get_samples(r.get("id", ""))
                except Exception:
                    samples = []
                for s in samples:
                    if needle in (s.get("sample_type", "") or "").lower():
                        return True
                    if needle in (s.get("sample_subtype", "") or "").lower():
                        return True
                    if needle in (s.get("name", "") or "").lower():
                        return True
                return False

            rows = [r for r in rows if _matches(r)]

        rows = rows[:limit]

        if not rows:
            return (
                "query_past_experiments: no records matched "
                f"(sample_type={sample_type!r}, max_n={max_n})."
            )

        header_filter = f" matching {sample_type!r}" if sample_type else ""
        lines = [
            f"Past experiments ({len(rows)}{header_filter}) from {db_path.name}:"
        ]
        for r in rows:
            sc = r.get("sample_count", 0)
            eid = (r.get("id", "") or "")[:8]
            lines.append(
                f"  - {eid}…  '{r.get('name', '')}'  "
                f"[{r.get('status', '?')}, {sc} samples, "
                f"{(r.get('start_time', '') or '')[:19]}]"
            )
        return _truncate("\n".join(lines))

    return query_past_experiments


# ─────────────────────────────────────────────────────────────────────
# Top-level tool list assembly
# ─────────────────────────────────────────────────────────────────────

def build_tools(
    buf: "BufferService | None",
    registry: SkillRegistry,
) -> list:
    """Assemble the full tool list for the XD agent.

    Order: domain tools first (most called during planning), buffer reads second
    (used for live tip context), handoffs last (terminal actions).

    Args:
        buf:      BufferService for live tip/scan context, or None in offline tests.
        registry: SkillRegistry populated with available instrument skills.
    """
    tools: list = [
        describe_skills_tool(registry),
        lookup_sample_tool(),
        query_past_experiments_tool(),
    ]
    if buf is not None:
        tools = tools + make_buffer_tools(buf)
    tools = tools + [
        make_handoff(
            "supervisor",
            (
                "Return control to the orchestrator/supervisor. Use when planning is "
                "blocked (ambiguous request, operator clarification needed) or when "
                "the plan has been rejected."
            ),
        ),
        make_handoff(
            "instrument_control",
            "Hand the completed ExperimentPlan to Instrument-Control for execution.",
        ),
    ]
    logger.info(
        "experiment_design: built %d tools (buf=%s)",
        len(tools),
        buf is not None,
    )
    return tools


# Standalone-safe tools surfaced into the skill menu by the auto-bridge
# (tool_skills.WORKFLOW_TOOL_EXPORTS). XD's tools are FACTORY closures, so we
# materialise the no-arg, runtime-independent ones here as module-level objects.
# describe_skills_tool is omitted: it needs the live registry AND is meta over the
# skill catalogue (it would describe the very menu it appears in).
WORKFLOW_EXPORT_TOOLS = [lookup_sample_tool(), query_past_experiments_tool()]

__all__ = [
    "describe_skills_tool",
    "lookup_sample_tool",
    "query_past_experiments_tool",
    "WORKFLOW_EXPORT_TOOLS",
    "build_tools",
]
