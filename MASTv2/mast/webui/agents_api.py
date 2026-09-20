"""Adapter: MAST multi-agent runtime → the Agents redesign UI payload.

The React prototype (``static/redesign/jsx/05-agents.jsx``) reads its data
from ``window.__MAST_AGENTS__``. Unlike the Records UI, the live multi-agent
data sources the Agents UI wants — a LangGraph event stream, per-agent
threads, hand-off interception, operator interjection — are listed in the
design handoff as backend capabilities that do not exist
yet. Until they do, ``build_agents_payload`` returns only the pieces that
are real and shape-safe; everything else is omitted so the prototype's
bundled mock data renders the faithful design.

Currently real:
  - ``models``   : per-agent model alias from the shared ``AGENT_MODEL`` registry
  - ``thinking`` : per-agent *effective* thinking level (what the model
                   actually runs at — reasoning models pinned "high (固定)")

Everything else (agents registry monograms, live state, hand-offs,
artifacts, threads) falls back to the prototype mock
for the wiring plan once the stream endpoint lands.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_AGENT_IDS = (
    # "orchestrator" FIRST — it is the supervisor, and it was the one id this
    # tuple left out.
    #
    # The orchestrator row used to render 「—  —」 for model/thinking, looking
    # like a UI bug. It was not: AGENT_MODEL has carried `orchestrator: kimi-k3`
    # all along, and `_resolve_agent_models` iterates THIS tuple, so the
    # supervisor was skipped before the registry was ever consulted. The data
    # existed; nothing asked for it.
    #
    # The UI keys the supervisor row as `_supervisor` (registry.tsx SUP_ID), so
    # the payload is emitted under BOTH ids — see the alias below. Renaming one
    # side instead would have been a two-file change with a silent failure mode
    # (a row that matches nothing renders blank, exactly like the bug).
    #
    # 同一个漏法的第二次：漏掉一个 id，那一行就显示「— —」，读起来像 UI 坏了，
    # 其实是这份名单没提问。research_director 2026-08-21 加进来。
    "orchestrator",
    "research_director",
    "literature", "experiment_design", "instrument_control",
    "data_processing", "paper_writing", "paper_review", "buffer_summarizer",
)

#: Frontend id → registry id, for agents the UI names differently.
#: `_supervisor` is what components/agents/registry.tsx uses for the SUP row.
_ID_ALIASES = {"_supervisor": "orchestrator"}


def build_agents_payload() -> dict:
    """Return the ``window.__MAST_AGENTS__`` payload.

    Returns ``{}`` (full mock) unless per-agent model overrides resolve —
    in which case only ``models`` / ``thinking`` are supplied and the rest
    of the UI still uses the prototype mock (shape-safe partial override).
    """
    models, thinking = _resolve_agent_models()
    if not models:
        return {}
    return {"models": models, "thinking": thinking}


def _resolve_agent_models() -> tuple[dict, dict]:
    """Best-effort read of per-agent model + thinking level.

    Reads the v2 agents shared model registry (``AGENT_MODEL``) for the
    per-agent model alias, and derives the *honest* effective thinking
    strength each model will actually run at via ``effective_thinking``
    (reasoning models are pinned "high (固定)"; Claude honours the level,
    defaulting to "off"). Returns ({}, {}) if the registry is unavailable.
    """
    models: dict[str, str] = {}
    thinking: dict[str, str] = {}
    try:
        from mast.agents._shared import models as agent_models
        registry = getattr(agent_models, "AGENT_MODEL", None)
        effective_thinking = getattr(agent_models, "effective_thinking", None)
        if isinstance(registry, dict):
            for aid in _AGENT_IDS:
                if aid not in registry:
                    continue
                model_id = str(registry[aid])
                models[aid] = model_id
                # Per-agent thinking is not separately configurable yet, so the
                # effective level is whatever the model intrinsically runs at
                # (no requested override → None). This is the real value, not a
                # hardcoded blank.
                if callable(effective_thinking):
                    thinking[aid] = effective_thinking(model_id, None)
            # Emit the supervisor under the id the UI actually looks up too, so
            # neither side has to know the other's naming .
            for ui_id, reg_id in _ID_ALIASES.items():
                if reg_id in models:
                    models[ui_id] = models[reg_id]
                    if reg_id in thinking:
                        thinking[ui_id] = thinking[reg_id]
    except Exception as exc:
        logger.debug("agents_api: model registry unavailable: %s", exc)
        return {}, {}
    return models, thinking


def write_agents_data(out_path: str | Path) -> dict:
    """Build the payload and write it to *out_path* as JSON. Returns the payload."""
    payload = build_agents_payload()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


# ─────────────────────────────────────────────────────────────────────
# REAL per-agent tool catalog — replaces the inspector's hardcoded JSX mock
# (AG_TOOLS) so the Agents UI shows the TRUE tool list (IC ≈ 224, not 11).
#
# Shipped in the /agents/snapshot payload (gui/app.py _snapshot) under "tools".
# The static catalog is computed ONCE and cached; warm it off the event loop
# (warm_agent_tools on a daemon thread) so the first snapshot never blocks.
# ─────────────────────────────────────────────────────────────────────

# 3 read-only buffer tools every graph agent gets (mast.agents._shared.buffer_tools).
_BUFFER_TOOL_NAMES = ("read_latest_tip_status", "get_scan_progress",
                      "get_tip_history_since")

# Per-agent handoff targets → tool name is f"handoff_to_{target}"
# (mast.agents._shared.handoff.make_handoff).
_HANDOFF_TARGETS = {
    "instrument_control": ("supervisor", "data_processing"),
    "literature": ("experiment_design", "supervisor"),
    "experiment_design": ("instrument_control", "supervisor"),
    "data_processing": ("supervisor", "paper_writing"),
    "paper_writing": ("paper_review", "supervisor"),
    "paper_review": ("paper_writing", "supervisor"),
}

# Fixed-@tool agents: module + the list attribute(s) holding their domain tools.
_FIXED_TOOL_LISTS = {
    "literature": ("mast.agents.literature.tools", ("AGENT_TOOLS", "LIBRARY_TOOLS")),
    "data_processing": ("mast.agents.data_processing.tools", ("AGENT_TOOLS",)),
    "paper_writing": ("mast.agents.paper_writing.tools", ("AGENT_TOOLS",)),
    "paper_review": ("mast.agents.paper_review.tools", ("AGENT_TOOLS",)),
}

_AGENT_TOOLS_CACHE: dict | None = None
_AGENT_TOOLS_LOCK = threading.Lock()

# The LIVE SkillRegistry (set by bootstrap, same pattern as builder_api /
# composite_panel). Before 2026-08-19 this module built its OWN registry and
# called discover() on it — so /agents/tools reported a catalogue that no agent
# ever had: no declarative composites, no custom skills, no bridged agent
# @tools, and (a plain bug) no mast.skills.paper at all. Adding a cache
# invalidation without this would have been a fake fix: the recompute would
# still have produced the same wrong list.
_live_registry = None
_LAST_COMPUTE_DEGRADED: bool = False


def set_live_registry(registry) -> None:
    """Point this module at the process's real SkillRegistry."""
    global _live_registry
    _live_registry = registry


def invalidate_agent_tools() -> None:
    """Drop the cache — next warm rebuilds. Call after any hot-(un)register."""
    global _AGENT_TOOLS_CACHE
    with _AGENT_TOOLS_LOCK:
        _AGENT_TOOLS_CACHE = None


def agent_tools_degraded() -> bool:
    """True if the last compute fell back to a private registry.

    Degraded means the listing is missing composites / custom / bridged tools
    and is NOT what the agents actually hold. Reported, never silent — the
    same discipline as reload_wiring.agent_refresh_pending().
    """
    return _LAST_COMPUTE_DEGRADED


def _tool(name: str, safety: str = "AUTO", level: str | None = None) -> dict:
    d = {"name": name, "safety": safety}
    if level:
        d["level"] = level
    return d


def _domain_names(mod_path: str, attrs: tuple[str, ...]) -> list[str]:
    import importlib
    mod = importlib.import_module(mod_path)
    out: list[str] = []
    for a in attrs:
        for t in (getattr(mod, a, None) or []):
            n = getattr(t, "name", None) or getattr(t, "__name__", None)
            if n:
                out.append(str(n))
    return out


def _handoff_tools(agent_id: str) -> list[dict]:
    return [_tool(f"handoff_to_{t}") for t in _HANDOFF_TARGETS.get(agent_id, ())]


def _buffer_tools() -> list[dict]:
    return [_tool(n) for n in _BUFFER_TOOL_NAMES]


def _compute_agent_tools() -> dict[str, list[dict]]:
    """Build {agent_id: [{name, safety, level?}]} from the REAL tool sources.

    Best-effort: each agent is wrapped in its own try/except so one failure can
    never drop the rest. Metadata-only — no torch / LLM clients / pools.
    """
    out: dict[str, list[dict]] = {}

    # instrument_control — the ONLY skill-registry agent; REAL name+safety+level.
    global _LAST_COMPUTE_DEGRADED
    try:
        reg = _live_registry
        if reg is None:
            # Fallback only. Missing: declarative composites, custom skills,
            # bridged agent @tools, and mast.skills.paper. Say so.
            from mast.core.registry import SkillRegistry
            reg = SkillRegistry()
            reg.discover("mast.skills.builtins", "mast.skills.composite",
                         "mast.skills.paper")
            _LAST_COMPUTE_DEGRADED = True
            logger.warning(
                "agent-tools: no live registry wired — /agents/tools will list "
                "a PRIVATE registry (no composites / custom / bridged tools). "
                "This is not what the agents hold.")
        else:
            _LAST_COMPUTE_DEGRADED = False
        # Apply the SAME gates build_instrument_skill_tools applies: a skill the
        # agent is not allowed to see must not appear here either, or this listing
        # becomes a different way of lying about the same thing.
        #
        # 并集的唯一定义处是 mast.skills.tool_face（硬件模块 + 高级能力 + 订阅）。
        # 在它存在之前这段是手抄的第二份 —— 于是第三道门（订阅）加进去的时候，
        # 这里会静默地保持两道门，界面列出的工具表就不再是模型手上那份。
        # tool_face.compute 自己 fail-open 并记下读不出来的门名，不会抛。
        try:
            from mast.skills import tool_face
            _all = frozenset(m.name for m in reg.list_skills())
            _face = tool_face.compute(_all)
            _skip = _face.names
            if _face.unreadable:
                logger.warning("agent-tools: 门读不出来（%s）—— 这部分未过滤",
                               "/".join(_face.unreadable))
        except Exception as exc:  # noqa: BLE001 — a gate we cannot read is not a gate
            logger.warning("agent-tools: skill gates unreadable (%s) — "
                           "listing UNFILTERED", exc)
            _skip = frozenset()
        ic: list[dict] = []
        for m in reg.list_skills():
            if m.name in _skip:
                continue
            sl = getattr(m, "safety_level", None)
            sev = getattr(sl, "name", None) or str(getattr(sl, "value", "auto")).upper()
            lvl = getattr(m, "composition_level", 0) or 0
            ic.append(_tool(m.name, sev, f"L{lvl}"))
        ic += _buffer_tools() + _handoff_tools("instrument_control")
        out["instrument_control"] = ic
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent-tools: instrument_control enum failed: %s", exc)

    # Fixed-@tool agents — read their module-level tool lists (cheap import).
    for aid, (mod_path, attrs) in _FIXED_TOOL_LISTS.items():
        try:
            names = _domain_names(mod_path, attrs)
            out[aid] = ([_tool(n) for n in names]
                        + _buffer_tools() + _handoff_tools(aid))
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent-tools: %s enum failed: %s", aid, exc)

    # experiment_design — tools come from factories; names are stable.
    out["experiment_design"] = (
        [_tool(n) for n in ("describe_skills", "lookup_sample",
                            "query_past_experiments")]
        + _buffer_tools() + _handoff_tools("experiment_design")
    )

    # buffer_summarizer — plain summarizer functions (no buffer/handoff tools).
    out["buffer_summarizer"] = [
        _tool(n) for n in ("summarize_tip_status", "summarize_segmentation",
                           "summarize_partial", "summarize_tip_fine")
    ]

    # supervisor — a router node with NO @tool tools; surface its REAL control
    # capabilities honestly instead of the mock's fictional 6.
    out["_supervisor"] = [_tool("route"), _tool("loop_guard"), _tool("budget_gate")]

    return out


def warm_agent_tools() -> dict:
    """Compute + cache the tool catalog. Call off the event loop.

    Locked since 2026-08-19: invalidate_agent_tools() can now fire from a hot
    reload on another thread while a request is warming.
    """
    global _AGENT_TOOLS_CACHE
    with _AGENT_TOOLS_LOCK:
        if _AGENT_TOOLS_CACHE is not None:
            return _AGENT_TOOLS_CACHE
        try:
            _AGENT_TOOLS_CACHE = _compute_agent_tools()
        except Exception as exc:  # noqa: BLE001
            logger.warning("warm_agent_tools failed: %s", exc)
            _AGENT_TOOLS_CACHE = {}
        return _AGENT_TOOLS_CACHE


def get_agent_tools() -> dict:
    """Return the cached catalog, or {} if not warmed yet. NEVER computes/blocks."""
    return _AGENT_TOOLS_CACHE or {}
