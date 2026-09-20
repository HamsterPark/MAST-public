"""Runtime access to the Nanonis Mimea SOFTWARE manual (GUI online help).

The manual is extracted offline from the installed Nanonis help CHM by
``tools/extract_nanonis_manual.py`` into
``<project_root>/artifacts/nanonis_manual/nanonis_manual.json`` (gitignored,
machine-local — see that script's LICENSING NOTE). This module loads it
best-effort and exposes two integration points (the operator-chosen hybrid):

  A. ``hint_for_skill(name)`` — a SHORT (≤2 line) operational hint derived from
     the skill's domain → manual module, baked into the LangChain tool
     description so the agent always has it when choosing/using the skill.
  B. ``search(query)`` / ``topic(key)`` — full-section lookup for the on-demand
     ``nanonis_manual`` tool, so the agent can pull depth (parameter ranges,
     procedures, gotchas) WITHOUT the 132-topic manual bloating base context.

Everything degrades gracefully: if the JSON is absent (manual not yet extracted
on this machine) all functions return empty / None and the agent simply has no
Nanonis manual — never an error.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

def _artifacts_dir() -> Path:
    try:
        from mast._runtime_paths import project_root
        return project_root() / "artifacts" / "nanonis_manual"
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[3] / "artifacts" / "nanonis_manual"


def _manual_path() -> Path:
    return _artifacts_dir() / "nanonis_manual.json"


def _skill_map_path() -> Path:
    return _artifacts_dir() / "skill_manual_map.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    """Load the manual JSON once. Returns {} if absent/unreadable (graceful)."""
    p = _manual_path()
    if not p.exists():
        logger.info("Nanonis manual not found at %s — manual injection disabled "
                    "(run tools/extract_nanonis_manual.py to enable).", p)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        logger.info("Nanonis manual loaded: %d topics from %s", len(data), p)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nanonis manual load failed (%s): %s", p, exc)
        return {}


@lru_cache(maxsize=1)
def _load_skill_map() -> dict[str, str]:
    """Load the baked {skill_name: topic_key} map (gui-free at runtime).

    Built offline by ``tools/extract_nanonis_manual.py`` (which may import
    ``mast.webui.encyclopedia`` — fine in a dev tool, but NOT at agent runtime
    where pulling in gradio/app would be heavy and risk an import cycle).
    Returns {} if absent (→ no hints; graceful)."""
    p = _skill_map_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nanonis skill map load failed (%s): %s", p, exc)
        return {}


def available() -> bool:
    """True if the manual store is present and non-empty."""
    return bool(_load())


def list_topics() -> list[dict]:
    """All topics as ``[{key, module, title}]`` (for the lookup tool's index)."""
    return [
        {"key": k, "module": v.get("module", ""), "title": v.get("title", k)}
        for k, v in sorted(_load().items())
    ]


def topic(key: str) -> dict | None:
    """Return one topic dict ``{module, topic, title, text, ...}`` or None."""
    return _load().get(key)


def _first_sentences(text: str, *, max_chars: int = 240) -> str:
    """A short hint: the first substantive sentence(s) of a topic body."""
    # Drop the leading title-ish lines; take the first real paragraph.
    body = "\n".join(
        ln for ln in text.splitlines() if len(ln.strip()) > 30
    ) or text
    body = body.strip()
    if len(body) <= max_chars:
        return body
    cut = body[:max_chars]
    # Prefer to end on a sentence boundary.
    m = re.search(r"^(.*[.!?])\s", cut, re.S)
    return (m.group(1) if m else cut).strip() + " …"


def hint_for_skill(skill_name: str) -> str | None:
    """Integration A: a SHORT manual-derived hint for a skill's tool description.

    Resolves skill → primary manual topic via the offline-baked
    ``skill_manual_map.json`` (NO gui import at runtime) → first sentence(s).
    Returns None when the skill isn't mapped or the manual isn't available. The
    hint is prefixed so it's visually distinct in the tool description.
    """
    man = _load()
    if not man:
        return None
    key = _load_skill_map().get(skill_name)
    t = man.get(key or "")
    if not t:
        return None
    # Keep it tight (~1 sentence): many skills share a module, so the same hint
    # repeats across their tool descriptions — a short blurb bounds that cost.
    hint = _first_sentences(t.get("text", ""), max_chars=150)
    if not hint:
        return None
    return (f"[Nanonis 手册·{t.get('title', key)}] {hint} "
            f"（细节用 nanonis_manual 工具按需查询）")


def modules_index() -> str:
    """Integration A (deduped): a one-time Nanonis-module index for the IC system
    prompt. Each DISTINCT manual topic referenced by the skill→topic map appears
    ONCE here (not per-skill), so the IC agent gets module orientation at ~470
    tokens instead of ~9k duplicated across 176 tool descriptions (review COST-1).
    Per-skill depth is still fetched on demand via the ``nanonis_manual`` tool.
    Returns "" when the manual/map isn't available (graceful)."""
    man = _load()
    smap = _load_skill_map()
    if not man or not smap:
        return ""
    keys = sorted(set(smap.values()))
    lines = ["Nanonis 软件模块速查（你的仪器技能对应这些 GUI 模块；"
             "需要参数范围/操作流程/注意事项时用 nanonis_manual 工具按需查全文）："]
    for key in keys:
        t = man.get(key)
        if not t:
            continue
        hint = _first_sentences(t.get("text", ""), max_chars=160).replace("\n", " ")
        if hint:
            lines.append(f"- {t.get('title', key)}: {hint}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _strip_manual_nav(text: str) -> str:
    """Drop leftover Nanonis-manual page navigation captured during extraction —
    a trailing 'Back Home' footer sits at the end of every topic. Removing it
    keeps both the agent's on-demand lookup and the inspector display clean."""
    text = re.sub(r"(?im)^\s*Back\s+Home\s*$", "", text)
    return text.strip()


def search(query: str, *, limit: int = 1, max_chars: int = 50000) -> str:
    """Integration B: look up the manual for the on-demand ``nanonis_manual`` tool.

    Matches *query* against topic key / module / title (case-insensitive,
    keyword overlap). Returns the best topic's FULL text. ``max_chars`` is a
    generous backstop (50 000 > the largest real topic ~32 700), so in practice
    nothing truncates — the agent explicitly requested this lookup and needs the
    complete section to set parameters correctly (a half section → guessing). The
    result goes into the tool-message CONTEXT (input), not the output budget, and
    only on demand, so the token cost is paid by the agent's own decision.
    On no match, returns a compact index of available modules so the agent can
    retry with a valid module/topic name. Never raises.
    """
    man = _load()
    if not man:
        return ("（Nanonis 软件手册未在本机提取；运行 "
                "tools/extract_nanonis_manual.py 后可用。）")
    q = (query or "").strip().lower()
    if not q:
        return _index_text(man)
    q_tokens = set(re.findall(r"[a-z0-9]+", q))

    def score(key: str, v: dict) -> int:
        hay = f"{key} {v.get('module','')} {v.get('title','')}".lower()
        s = 0
        if q in hay:
            s += 100
        s += sum(5 for tok in q_tokens if tok in hay)
        # light body match as a tiebreaker
        if q in v.get("text", "").lower():
            s += 10
        return s

    ranked = sorted(man.items(), key=lambda kv: score(*kv), reverse=True)
    best_key, best = ranked[0]
    if score(best_key, best) <= 0:
        return ("未找到匹配的手册条目。可用模块/主题：\n" + _index_text(man))

    out = []
    for key, v in ranked[:max(1, limit)]:
        if score(key, v) <= 0:
            break
        body = _strip_manual_nav(v.get("text", ""))
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n…（条目过长已截断；可指定更具体的主题）"
        out.append(f"# {v.get('title', key)}  (模块: {v.get('module','')} · 键: {key})\n\n{body}")
    return "\n\n———\n\n".join(out)


def _index_text(man: dict) -> str:
    """Compact module → topics index (for empty/failed queries)."""
    by_mod: dict[str, list[str]] = {}
    for k, v in man.items():
        by_mod.setdefault(v.get("module", "?"), []).append(v.get("title", k))
    lines = [f"- {mod}: {', '.join(sorted(set(ts)))}" for mod, ts in sorted(by_mod.items())]
    return "\n".join(lines)
