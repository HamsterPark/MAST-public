"""接口花销记录 — MAST API cost/usage ledger.

A self-contained subsystem that records the cost of every paid API call MAST
makes — LLM turns (all agents + main chat/QuickAsk), voice TTS/ASR, document
OCR, embeddings — and aggregates the spend for the 「用量·花销」page.

Design:
  * :mod:`mast.billing.pricing` — an EDITABLE price book (native currency per
    provider) + a single cost formula. Defaults are approximate and clearly
    flagged; the operator who pays the bills can correct any model's price.
  * :mod:`mast.billing.ledger` — a standalone SQLite ledger (append + aggregate),
    process-global singleton, thread-safe, never raises on the hot path.
  * :mod:`mast.billing.capture` — the capture hooks: a LangChain callback for the
    model factory + thin ``record_*`` helpers for the non-LLM call sites.

Never on any hot path may a billing error affect the actual API call — every
public entry point is fail-safe (records best-effort, swallows + logs).
"""
from __future__ import annotations

from mast.billing.capture import (
    LlmUsageCallback,
    record_asr,
    record_llm,
    record_ocr,
    record_tts,
)
from mast.billing.ledger import UsageLedger, UsageRecord, get_ledger
from mast.billing.pricing import PriceEntry, compute_cost, load_pricing, lookup_price

__all__ = [
    "LlmUsageCallback",
    "record_llm",
    "record_tts",
    "record_asr",
    "record_ocr",
    "UsageLedger",
    "UsageRecord",
    "get_ledger",
    "PriceEntry",
    "compute_cost",
    "load_pricing",
    "lookup_price",
]
