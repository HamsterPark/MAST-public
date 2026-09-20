"""Editable price book + one cost formula for every paid API MAST calls.

Prices are in each provider's **native currency** (CNY for the Chinese providers,
USD for Anthropic) — we never fake an FX conversion into the ledger; the 「用量·
花销」page subtotals per currency and offers an optional labelled 折算.

⚠️ The DEFAULT numbers below are APPROXIMATE and WILL drift — provider pricing
changes and depends on tier/context/cache. They exist so the ledger shows a
believable number out of the box; the operator who pays the bills should correct
any model via the override file (:func:`save_pricing_override`) — a JSON merged
over these defaults at load. Every record stores whether its price was KNOWN, so
the UI can mark estimated-or-missing prices with a ≈ / 未定价 badge rather than
pretending precision.

One unified cost formula (:func:`compute_cost`) prices whatever dimensions a call
has — input/output tokens (LLM · OCR-as-VLM · embeddings), characters (TTS), or
audio minutes (ASR) — by summing only the rates that entry sets.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── provider inference (self-contained; billing must not depend on the agents layer) ──

def provider_for(model_id: str) -> str:
    """Infer the billing provider from a model id. Unknown → ``"unknown"``."""
    m = (model_id or "").lower()
    if m.startswith("minimax"):
        return "minimax"
    if m.startswith("glm"):
        return "zhipu"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("kimi", "moonshot")):
        return "moonshot"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith(("qwen", "text-embedding", "cosyvoice", "sambert", "paraformer")):
        return "dashscope"
    return "unknown"


# ── price entry + formula ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PriceEntry:
    """A unit price. Only the rates relevant to a call's *kind* are set.

    Rates:
      * ``input_per_m`` / ``output_per_m`` — currency per 1e6 tokens (LLM, OCR
        via a VLM, embeddings use ``input_per_m`` only).
      * ``char_per_m`` — currency per 1e6 characters (TTS synthesis).
      * ``per_minute`` — currency per minute of audio (ASR).
    """
    currency: str = "CNY"
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    char_per_m: float = 0.0
    per_minute: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "input_per_m": self.input_per_m,
            "output_per_m": self.output_per_m,
            "char_per_m": self.char_per_m,
            "per_minute": self.per_minute,
            "note": self.note,
        }


def compute_cost(
    entry: PriceEntry,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    chars: int = 0,
    seconds: float = 0.0,
) -> float:
    """Total cost in ``entry.currency`` — sum of only the dimensions priced."""
    return (
        (input_tokens / 1e6) * entry.input_per_m
        + (output_tokens / 1e6) * entry.output_per_m
        + (chars / 1e6) * entry.char_per_m
        + (seconds / 60.0) * entry.per_minute
    )


# ── default price book (APPROXIMATE — edit via the override file) ────────────
# Keyed by a normalized model id (see _normalize). Values are the operator's to
# correct. USD for Anthropic, CNY for everyone else.

_USD = "USD"
_CNY = "CNY"

DEFAULT_PRICING: dict[str, PriceEntry] = {
    # ── Anthropic Claude (USD / 1M tokens) ──
    "claude-opus-4-8":   PriceEntry(_USD, 15.0, 75.0, note="approx"),
    "claude-opus-4-7":   PriceEntry(_USD, 15.0, 75.0, note="approx"),
    "claude-opus-4-6":   PriceEntry(_USD, 15.0, 75.0, note="approx"),
    "claude-opus-4":     PriceEntry(_USD, 15.0, 75.0, note="approx"),
    "claude-sonnet-4-6": PriceEntry(_USD, 3.0, 15.0, note="approx"),
    "claude-sonnet-4":   PriceEntry(_USD, 3.0, 15.0, note="approx"),
    "claude-haiku-4-5":  PriceEntry(_USD, 0.8, 4.0, note="approx"),
    # ── Moonshot Kimi (CNY / 1M tokens) ──
    "kimi-k3":           PriceEntry(_CNY, 4.0, 16.0, note="approx"),
    "kimi-k2.7-code":    PriceEntry(_CNY, 4.0, 16.0, note="approx"),
    "kimi-k2.6":         PriceEntry(_CNY, 4.0, 16.0, note="approx"),
    "moonshot-v1-128k":  PriceEntry(_CNY, 60.0, 60.0, note="approx"),
    # ── DeepSeek (CNY / 1M tokens) ──
    "deepseek-v4-pro":   PriceEntry(_CNY, 2.0, 8.0, note="approx"),
    # ── Qwen / DashScope chat (CNY / 1M tokens) ──
    "qwen3.7-max":       PriceEntry(_CNY, 20.0, 60.0, note="approx"),
    # ── Zhipu GLM (CNY / 1M tokens) ──
    "glm-5.2":           PriceEntry(_CNY, 5.0, 15.0, note="approx"),
    "glm-5.1":           PriceEntry(_CNY, 5.0, 15.0, note="approx"),
    # ── MiniMax (CNY / 1M tokens) ──
    "minimax-m3":        PriceEntry(_CNY, 8.0, 24.0, note="approx"),
    # ── DashScope OCR (VLM, billed as tokens; CNY / 1M) ──
    "qwen3.5-ocr":       PriceEntry(_CNY, 5.0, 5.0, note="approx OCR-as-VLM"),
    "qwen-vl-ocr":       PriceEntry(_CNY, 5.0, 5.0, note="approx OCR-as-VLM"),
    # ── DashScope TTS (CNY / 1M characters) ──
    "qwen3-tts-flash":          PriceEntry(_CNY, char_per_m=200.0, note="approx TTS ~¥2/万字"),
    "qwen3-tts-flash-realtime": PriceEntry(_CNY, char_per_m=200.0, note="approx TTS ~¥2/万字"),
    # ── DashScope ASR (CNY / audio minute) ──
    "qwen3-asr-flash":          PriceEntry(_CNY, per_minute=0.24, note="approx ASR ~¥0.24/min"),
    "qwen3-asr-flash-realtime": PriceEntry(_CNY, per_minute=0.24, note="approx ASR ~¥0.24/min"),
    # ── DashScope embeddings (CNY / 1M tokens) ──
    "text-embedding-v4": PriceEntry(_CNY, 0.5, 0.0, note="approx embedding"),
}

# Per-provider fallback for an unknown model of a KNOWN provider (so a new Kimi
# snapshot still bills roughly right instead of 未定价). currency + a mid rate.
_PROVIDER_FALLBACK: dict[str, PriceEntry] = {
    "anthropic": PriceEntry(_USD, 3.0, 15.0, note="provider fallback"),
    "moonshot":  PriceEntry(_CNY, 4.0, 16.0, note="provider fallback"),
    "deepseek":  PriceEntry(_CNY, 2.0, 8.0, note="provider fallback"),
    "dashscope": PriceEntry(_CNY, 20.0, 60.0, note="provider fallback"),
    "zhipu":     PriceEntry(_CNY, 5.0, 15.0, note="provider fallback"),
    "minimax":   PriceEntry(_CNY, 8.0, 24.0, note="provider fallback"),
}


def _normalize(model_id: str) -> str:
    return (model_id or "").strip().lower()


# ── override file (operator-editable) ────────────────────────────────────────

def _override_path() -> Path:
    from mast._runtime_paths import project_root
    return Path(project_root()) / "experiments" / "usage_pricing.json"


_CACHE: dict[str, PriceEntry] | None = None
_CACHE_MTIME: float | None = None


def load_pricing(force: bool = False) -> dict[str, PriceEntry]:
    """Defaults merged with the override JSON (override wins per-model).

    Cached on the override file's mtime so edits are picked up without a restart
    but repeated calls don't re-read the disk.
    """
    global _CACHE, _CACHE_MTIME
    path = _override_path()
    try:
        mtime = path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        mtime = 0.0
    if not force and _CACHE is not None and _CACHE_MTIME == mtime:
        return _CACHE

    merged = dict(DEFAULT_PRICING)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for mid, spec in (raw.get("models") or {}).items():
                if isinstance(spec, dict):
                    base = merged.get(_normalize(mid), PriceEntry())
                    merged[_normalize(mid)] = replace(
                        base,
                        currency=str(spec.get("currency", base.currency)),
                        input_per_m=float(spec.get("input_per_m", base.input_per_m)),
                        output_per_m=float(spec.get("output_per_m", base.output_per_m)),
                        char_per_m=float(spec.get("char_per_m", base.char_per_m)),
                        per_minute=float(spec.get("per_minute", base.per_minute)),
                        note=str(spec.get("note", base.note or "custom")),
                    )
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("usage_pricing.json ignored (%s)", exc)

    _CACHE, _CACHE_MTIME = merged, mtime
    return merged


def save_pricing_override(models: dict[str, dict], *, usd_to_cny: Optional[float] = None) -> Path:
    """Persist operator price edits (merged over defaults on next load)."""
    path = _override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"models": models}
    if usd_to_cny is not None:
        payload["usd_to_cny"] = float(usd_to_cny)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    load_pricing(force=True)
    return path


def usd_to_cny_rate() -> float:
    """Optional FX for the labelled 折算 total. Override file wins, else ~7.2."""
    path = _override_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            r = raw.get("usd_to_cny")
            if isinstance(r, (int, float)) and r > 0:
                return float(r)
        except (OSError, ValueError):
            pass
    return 7.2


# ── lookup ───────────────────────────────────────────────────────────────────

def lookup_price(model_id: str, *, provider: Optional[str] = None) -> tuple[PriceEntry, bool]:
    """(entry, known). ``known`` is False when we fell back to a provider default
    or a zero-price placeholder — the UI marks those ≈ / 未定价."""
    book = load_pricing()
    norm = _normalize(model_id)

    # 1. exact match
    if norm in book:
        return book[norm], True
    # 2. prefix match (a dated snapshot like claude-opus-4-8-2026… → claude-opus-4-8)
    for key, entry in book.items():
        if norm.startswith(key) or key.startswith(norm):
            if len(key) >= 6:  # avoid matching on a too-short stem
                return entry, True
    # 3. provider fallback
    prov = provider or provider_for(model_id)
    if prov in _PROVIDER_FALLBACK:
        return _PROVIDER_FALLBACK[prov], False
    # 4. truly unknown
    return PriceEntry(currency="CNY", note="未定价"), False


__all__ = [
    "PriceEntry",
    "compute_cost",
    "provider_for",
    "DEFAULT_PRICING",
    "load_pricing",
    "save_pricing_override",
    "usd_to_cny_rate",
    "lookup_price",
]
