"""Capture hooks that feed the ledger.

* :class:`LlmUsageCallback` — a LangChain callback attached by the model factory
  to every agent/chat model; its ``on_llm_end`` reads the turn's usage_metadata
  and books the cost. Bound with (source, model_id, provider) at construction so
  it labels the row even when the provider omits the model name.
* :func:`record_llm` / :func:`record_tts` / :func:`record_asr` / :func:`record_ocr`
  — thin helpers for call sites that already hold the usage numbers directly
  (ClaudeClient, the voice + OCR paths).

Every entry point is fail-safe: a billing error is swallowed + logged, never
propagated to the API call that triggered it.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from mast.billing.ledger import UsageRecord, get_ledger
from mast.billing.pricing import compute_cost, lookup_price, provider_for

logger = logging.getLogger(__name__)


# ── record helpers ───────────────────────────────────────────────────────────

def record_llm(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    source: str = "",
    provider: Optional[str] = None,
    kind: str = "llm",
    meta: Optional[dict] = None,
) -> None:
    """Price + book one token-based call (LLM / OCR-as-VLM / embedding)."""
    try:
        if not input_tokens and not output_tokens:
            return  # no usage reported → nothing to bill (don't write a 0 row)
        prov = provider or provider_for(model)
        entry, known = lookup_price(model, provider=prov)
        cost = compute_cost(entry, input_tokens=input_tokens, output_tokens=output_tokens)
        get_ledger().record(UsageRecord(
            kind=kind, provider=prov, model=model or "unknown", source=source,
            input_tokens=int(input_tokens or 0), output_tokens=int(output_tokens or 0),
            cost=cost, currency=entry.currency, cost_known=known, meta=meta or {},
        ))
    except Exception:  # noqa: BLE001
        logger.debug("record_llm failed (swallowed)", exc_info=True)


def record_tts(*, model: str, chars: int, source: str = "voice") -> None:
    try:
        if not chars:
            return
        entry, known = lookup_price(model)
        cost = compute_cost(entry, chars=chars)
        get_ledger().record(UsageRecord(
            kind="tts", provider=provider_for(model), model=model, source=source,
            chars=int(chars), cost=cost, currency=entry.currency, cost_known=known,
        ))
    except Exception:  # noqa: BLE001
        logger.debug("record_tts failed (swallowed)", exc_info=True)


def record_asr(*, model: str, seconds: float, source: str = "voice") -> None:
    try:
        if not seconds:
            return
        entry, known = lookup_price(model)
        cost = compute_cost(entry, seconds=seconds)
        get_ledger().record(UsageRecord(
            kind="asr", provider=provider_for(model), model=model, source=source,
            seconds=float(seconds), cost=cost, currency=entry.currency, cost_known=known,
        ))
    except Exception:  # noqa: BLE001
        logger.debug("record_asr failed (swallowed)", exc_info=True)


def record_ocr(
    *, model: str, input_tokens: int = 0, output_tokens: int = 0,
    source: str = "ocr", pages: Optional[int] = None,
) -> None:
    """OCR via a VLM bills as tokens; pages tucked into meta for context."""
    record_llm(
        model=model, input_tokens=input_tokens, output_tokens=output_tokens,
        source=source, kind="ocr", meta=({"pages": pages} if pages else None),
    )


# ── LangChain callback ───────────────────────────────────────────────────────

def _extract_usage(response: Any) -> tuple[int, int, str]:
    """(input_tokens, output_tokens, model_name) from an LLMResult, best-effort."""
    in_tok = out_tok = 0
    model = ""
    for batch in (getattr(response, "generations", None) or []):
        for gen in (batch or []):
            msg = getattr(gen, "message", None)
            if msg is None:
                continue
            um = getattr(msg, "usage_metadata", None)
            if isinstance(um, dict):
                in_tok = int(um.get("input_tokens") or in_tok)
                out_tok = int(um.get("output_tokens") or out_tok)
            rm = getattr(msg, "response_metadata", None) or {}
            model = rm.get("model_name") or rm.get("model") or model
    lo = getattr(response, "llm_output", None) or {}
    if not in_tok and not out_tok:
        tu = lo.get("token_usage") or lo.get("usage") or {}
        in_tok = int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0)
        out_tok = int(tu.get("completion_tokens") or tu.get("output_tokens") or 0)
    model = model or lo.get("model_name") or ""
    return in_tok, out_tok, model


class LlmUsageCallback(BaseCallbackHandler):
    """Books every model call it is attached to. Bound with call context."""

    def __init__(self, *, source: str = "", model_id: str = "", provider: str = ""):
        super().__init__()
        self._source = source
        self._model_id = model_id
        self._provider = provider or (provider_for(model_id) if model_id else "")

    def on_llm_end(self, response: Any, **_kwargs: Any) -> None:
        try:
            in_tok, out_tok, model = _extract_usage(response)
            record_llm(
                model=(model or self._model_id),
                input_tokens=in_tok, output_tokens=out_tok,
                source=self._source, provider=(self._provider or None),
            )
        except Exception:  # noqa: BLE001 — callbacks must never break a run
            logger.debug("LlmUsageCallback.on_llm_end failed (swallowed)", exc_info=True)


__all__ = [
    "LlmUsageCallback",
    "record_llm",
    "record_tts",
    "record_asr",
    "record_ocr",
]
