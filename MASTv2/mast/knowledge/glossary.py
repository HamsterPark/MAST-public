"""STM glossary lookup — load stm_glossary.yaml and expose three runtime hooks.

Three operations:

  * ``normalize_asr(text)`` — replace common ASR transcription errors with the
    canonical English / standard Chinese form. Used after `voice.transcribe()`.

  * ``read_as_for_tts(text)`` — substitute terms with their TTS-friendly
    Chinese reading. Used before `voice.synthesize()`.

  * ``lookup(query, k=30)`` — return up to k glossary entries relevant to the
    user's chat message. Used to inject domain context into LLM system prompts.

The glossary is loaded once on first call and cached. Re-load with
``reload_glossary()`` if the yaml file changes at runtime.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)

# ── Path resolution ──────────────────────────────────────────────────
# This file lives in {repo}/mast/knowledge/ in v1 or
# {repo}/MASTv2/mast/knowledge/ in v2; locate the sibling stm_glossary.yaml.
_HERE = Path(__file__).resolve().parent
_GLOSSARY_PATH = _HERE / "stm_glossary.yaml"


# ── Cache ─────────────────────────────────────────────────────────────
_lock = threading.Lock()
_cache: dict[str, object] = {
    "loaded": False,
    "entries": [],            # list of dicts as parsed from yaml
    "asr_map": {},            # alias_lower → canonical_replacement
    "tts_map": {},            # canonical_en_lower / abbrev_lower → tts_read_as_zh
    "lookup_index": [],       # list of (search_keys_lower, entry) for retrieval
}


def _load() -> None:
    """Lazy-load the glossary yaml on first use; rebuild lookup tables."""
    if _cache["loaded"]:
        return
    with _lock:
        if _cache["loaded"]:
            return
        path = _GLOSSARY_PATH
        if not path.exists():
            logger.info("stm_glossary.yaml not found at %s — hooks will be no-ops", path)
            _cache["entries"] = []
            _cache["asr_map"] = {}
            _cache["tts_map"] = {}
            _cache["lookup_index"] = []
            _cache["loaded"] = True
            return
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        except yaml.YAMLError as e:
            logger.error("stm_glossary.yaml parse failed: %s", e)
            _cache["entries"] = []
            _cache["asr_map"] = {}
            _cache["tts_map"] = {}
            _cache["lookup_index"] = []
            _cache["loaded"] = True
            return
        if not isinstance(data, list):
            data = []

        entries: list[dict] = []
        asr_map: dict[str, str] = {}
        tts_map: dict[str, str] = {}
        lookup_index: list[tuple[set[str], dict]] = []

        for entry in data:
            if not isinstance(entry, dict):
                continue
            cen = (entry.get("canonical_en") or "").strip()
            if not cen:
                continue
            entries.append(entry)
            abbrev = (entry.get("abbrev") or "").strip()
            zh_list = entry.get("zh") or []
            asr_norm = entry.get("asr_normalize") or []
            tts_zh = (entry.get("tts_read_as_zh") or "").strip()

            # Canonical Chinese: use zh[0] when available
            zh_canonical = zh_list[0] if zh_list else ""

            # Build ASR replacement map: each asr_normalize alias → canonical
            # representation. Use zh_canonical first (Chinese context) then
            # canonical_en. Keep abbrev as a separate option.
            target = zh_canonical or cen
            for alias in asr_norm:
                a = (str(alias) or "").strip()
                if a and a.lower() != target.lower():
                    asr_map.setdefault(a.lower(), target)
            # Also map common alternate Chinese spellings → zh_canonical
            for alt in zh_list[1:]:
                a = (str(alt) or "").strip()
                if a and zh_canonical and a != zh_canonical:
                    asr_map.setdefault(a.lower(), zh_canonical)

            # TTS replacement: canonical_en (and abbrev) → tts_read_as_zh
            if tts_zh:
                tts_map.setdefault(cen.lower(), tts_zh)
                if abbrev:
                    tts_map.setdefault(abbrev.lower(), tts_zh)

            # Lookup index: collect every searchable surface form
            keys: set[str] = set()
            keys.add(cen.lower())
            if abbrev:
                keys.add(abbrev.lower())
            for z in zh_list:
                if z:
                    keys.add(str(z).lower())
            for a in asr_norm:
                if a:
                    keys.add(str(a).lower())
            lookup_index.append((keys, entry))

        _cache["entries"] = entries
        _cache["asr_map"] = asr_map
        _cache["tts_map"] = tts_map
        _cache["lookup_index"] = lookup_index
        # Pre-compile a single alternation regex per map. The previous code
        # re.compile()'d every alias on EVERY call; with 4.7k/2.1k aliases that
        # blew past the stdlib re-cache (512) and cost ~16 s / ~7 s per call on
        # the (synchronous) voice path. Built once here; reused on every call.
        _cache["asr_replacer"] = _compile_replacer(asr_map)
        _cache["tts_replacer"] = _compile_replacer(tts_map)
        _cache["loaded"] = True
        logger.info(
            "Loaded glossary: %d entries, %d asr aliases, %d tts mappings",
            len(entries), len(asr_map), len(tts_map),
        )


def reload_glossary() -> None:
    """Force reload — call after editing stm_glossary.yaml at runtime."""
    with _lock:
        _cache["loaded"] = False
    _load()


def entries() -> list[dict]:
    _load()
    return list(_cache["entries"])  # type: ignore[arg-type]


# ── Public hooks ──────────────────────────────────────────────────────

def _compile_replacer(mapping: dict) -> tuple | None:
    """Build ONE alternation regex + lower-cased target map for whole-token
    replacement, sorted longest-alias-first so multi-word aliases win.

    Replaces the old per-call re.compile() of every alias. Returns None when the
    map is empty. CJK has no \\b, so we guard with non-word/non-CJK lookarounds.
    """
    aliases = sorted((a for a in mapping if a), key=len, reverse=True)
    if not aliases:
        return None
    lower_map = {str(a).lower(): mapping[a] for a in aliases}
    pattern = re.compile(
        r"(?<![\w一-鿿])(" + "|".join(re.escape(a) for a in aliases) + r")(?![\w一-鿿])",
        re.IGNORECASE,
    )
    return (pattern, lower_map)


def normalize_asr(text: str) -> str:
    """Apply ASR error → canonical replacements. Whole-word, case-insensitive.

    Returns the input unchanged if the glossary is empty or no aliases match.
    """
    _load()
    replacer = _cache.get("asr_replacer")
    if not text or not replacer:
        return text
    pattern, lower_map = replacer
    return pattern.sub(lambda m: lower_map.get(m.group(1).lower(), m.group(0)), text)


def read_as_for_tts(text: str) -> str:
    """Replace term tokens with their TTS-friendly Chinese reading.

    Sample: "STM tip on Au(111)" → "扫描隧道显微镜 tip on Au 一一一" if entries
    for STM and Au(111) carry tts_read_as_zh values.
    """
    _load()
    replacer = _cache.get("tts_replacer")
    if not text or not replacer:
        return text
    pattern, lower_map = replacer
    return pattern.sub(lambda m: lower_map.get(m.group(1).lower(), m.group(0)), text)


# Top-level OpenAlex concept names that are too generic for LLM prompt injection
# but are still useful for ASR/TTS normalization (so we keep them in the yaml,
# only filter at lookup time).
_LOOKUP_SKIP_CANONICAL: frozenset[str] = frozenset({
    "materials science", "composite material", "crystallography",
    "nanotechnology", "optoelectronics", "optics", "mathematics",
    "physical chemistry", "organic chemistry", "condensed matter physics",
    "quantum mechanics", "engineering", "mechanical engineering",
    "computer science", "biology", "chemistry", "physics",
    "molecule", "scanning", "tunneling", "tunnelling", "microscopy",
    "spectroscopy", "surface", "structure", "general",
})


# ── Lookup (for LLM context injection) ───────────────────────────────

# Pure-ASCII keys this short (1-2 chars) are almost always element symbols or
# abbreviations (au, si, he, fe, co, w, ...). Matched as a raw substring they
# fire spuriously inside ordinary words — "he" in "the", "si" in "consider",
# "co" in "control", "as" in "case" — flooding the result with irrelevant
# element entries. For these we demand a whole-token (word-boundary) match
# instead. CJK keys have no whitespace boundaries (Chinese is unspaced), so the
# substring path is kept for them and for longer ASCII keys.
_SHORT_ASCII_RE = re.compile(r"^[a-z0-9]{1,2}$")


def _substring_hit(k_alias: str, q: str, q_tokens: set[str]) -> bool:
    """Whether *k_alias* counts as a full match against query *q*.

    Short pure-ASCII keys must appear as a standalone token (word boundary);
    everything else uses the original cheap substring test.
    """
    if _SHORT_ASCII_RE.match(k_alias):
        return k_alias in q_tokens
    return k_alias in q


def lookup(query: str, k: int = 30) -> list[dict]:
    """Return up to *k* glossary entries relevant to *query*.

    Simple substring + token-overlap scoring. Good enough for LLM-prompt
    augmentation; not a replacement for the literature embedding index.

    Generic top-level concepts (e.g. "Materials science", "Physics") are
    excluded from the result even if their alias appears in the query —
    they add noise without disambiguation value.
    """
    _load()
    if not query or not _cache["lookup_index"]:
        return []
    q = query.lower()
    q_tokens = set(re.findall(r"[a-z0-9]+|[一-鿿]+", q))

    scored: list[tuple[float, dict]] = []
    for keys, entry in _cache["lookup_index"]:  # type: ignore[arg-type]
        cen = (entry.get("canonical_en") or "").lower().strip()
        if cen in _LOOKUP_SKIP_CANONICAL:
            continue
        score = 0.0
        for k_alias in keys:
            if not k_alias:
                continue
            if _substring_hit(k_alias, q, q_tokens):
                score += 5.0 * len(k_alias)  # full-substring match
                continue
            # Token overlap
            kw = set(re.findall(r"[a-z0-9]+|[一-鿿]+", k_alias))
            common = q_tokens & kw
            if common:
                score += sum(len(t) for t in common)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:k]]


def format_for_prompt(matches: Iterable[dict]) -> str:
    """Render lookup() result as a short text block to drop into a system prompt."""
    lines: list[str] = []
    for m in matches:
        cen = m.get("canonical_en", "")
        abbrev = m.get("abbrev") or ""
        zh = m.get("zh") or []
        zh0 = zh[0] if zh else ""
        domain = m.get("domain", "")
        head = cen if not abbrev else f"{cen} ({abbrev})"
        if zh0:
            lines.append(f"- {head} = {zh0}  [{domain}]")
        else:
            lines.append(f"- {head}  [{domain}]")
    return "\n".join(lines)


__all__ = [
    "entries",
    "reload_glossary",
    "normalize_asr",
    "read_as_for_tts",
    "lookup",
    "format_for_prompt",
]
