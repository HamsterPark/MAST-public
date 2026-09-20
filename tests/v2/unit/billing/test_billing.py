"""接口花销记录 — mast.billing (pricing + ledger + capture).

Fully offline: an isolated tmp ledger + tmp pricing override per test, LLM
results faked. See docs/v2/ (用量·花销 / API cost recorder).
"""
from __future__ import annotations

import time

import pytest

from mast.billing import ledger as L
from mast.billing import pricing as P
from mast.billing.capture import (
    LlmUsageCallback,
    record_asr,
    record_llm,
    record_ocr,
    record_tts,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    led = L.UsageLedger(tmp_path / "u.sqlite")
    L.set_ledger_for_test(led)
    P._CACHE = None
    P._CACHE_MTIME = None
    monkeypatch.setattr(P, "_override_path", lambda: tmp_path / "usage_pricing.json")
    yield
    led.close()
    L.set_ledger_for_test(None)
    P._CACHE = None
    P._CACHE_MTIME = None


# ── pricing ──────────────────────────────────────────────────────────────────

def test_provider_for():
    assert P.provider_for("kimi-k3") == "moonshot"
    assert P.provider_for("claude-opus-4-8") == "anthropic"
    assert P.provider_for("glm-5.2") == "zhipu"
    assert P.provider_for("MiniMax-M3") == "minimax"
    assert P.provider_for("qwen3.7-max") == "dashscope"
    assert P.provider_for("deepseek-v4-pro") == "deepseek"
    assert P.provider_for("something-weird") == "unknown"


def test_compute_cost_formula():
    e = P.PriceEntry("CNY", input_per_m=4.0, output_per_m=16.0)
    # 1M in @4 + 0.5M out @16 = 4 + 8 = 12
    assert P.compute_cost(e, input_tokens=1_000_000, output_tokens=500_000) == pytest.approx(12.0)


def test_lookup_exact_and_prefix():
    e, known = P.lookup_price("kimi-k3")
    assert known and e.currency == "CNY" and e.input_per_m > 0
    # dated snapshot resolves via prefix
    e2, known2 = P.lookup_price("claude-opus-4-8-20260101")
    assert known2 and e2.currency == "USD"


def test_lookup_provider_fallback_marks_unknown():
    e, known = P.lookup_price("kimi-k9-brand-new")   # unknown Kimi snapshot
    assert e.currency == "CNY"
    assert known is False                            # provider fallback → not authoritative


def test_lookup_truly_unknown():
    e, known = P.lookup_price("totally-made-up-xyz")
    assert known is False and e.note == "未定价"


def test_pricing_override_merges(tmp_path, monkeypatch):
    P.save_pricing_override({"kimi-k3": {"currency": "CNY", "input_per_m": 1.0, "output_per_m": 2.0}})
    e, known = P.lookup_price("kimi-k3")
    assert known and e.input_per_m == 1.0 and e.output_per_m == 2.0


# ── ledger ───────────────────────────────────────────────────────────────────

def test_ledger_record_and_summary():
    led = L.get_ledger()
    led.record(L.UsageRecord(kind="llm", provider="moonshot", model="kimi-k3",
                             source="orchestrator", input_tokens=1000, output_tokens=500,
                             cost=0.012, currency="CNY"))
    led.record(L.UsageRecord(kind="llm", provider="anthropic", model="claude-opus-4-8",
                             source="chat", input_tokens=2000, output_tokens=1000,
                             cost=0.105, currency="USD"))
    s = led.summary()
    assert s["count"] == 2
    assert s["by_currency"]["CNY"]["cost"] == pytest.approx(0.012)
    assert s["by_currency"]["USD"]["cost"] == pytest.approx(0.105)
    provs = {r["key"] for r in s["by_provider"]}
    assert provs == {"moonshot", "anthropic"}
    assert "combined_cny" in s and s["combined_cny"] > 0.105   # USD folded in via FX


def test_ledger_range_filter():
    led = L.get_ledger()
    now = time.time()
    led.record(L.UsageRecord(kind="llm", provider="p", model="m", cost=1.0, ts=now - 10000))
    led.record(L.UsageRecord(kind="llm", provider="p", model="m", cost=2.0, ts=now))
    s = led.summary(since=now - 100)
    assert s["count"] == 1
    assert s["by_currency"]["CNY"]["cost"] == pytest.approx(2.0)


def test_ledger_recent_and_reset():
    led = L.get_ledger()
    for i in range(3):
        led.record(L.UsageRecord(kind="llm", provider="p", model=f"m{i}", cost=1.0))
    r = led.recent(limit=2)
    assert len(r) == 2 and r[0]["model"] == "m2"     # newest first
    assert led.reset() == 3
    assert led.summary()["count"] == 0


# ── capture ──────────────────────────────────────────────────────────────────

def test_record_llm_prices_and_books():
    record_llm(model="kimi-k3", input_tokens=1_000_000, output_tokens=500_000,
               source="orchestrator")
    s = L.get_ledger().summary()
    assert s["count"] == 1
    row = L.get_ledger().recent()[0]
    assert row["provider"] == "moonshot" and row["cost"] == pytest.approx(12.0)
    assert row["cost_known"] is True


def test_record_llm_skips_zero_usage():
    record_llm(model="kimi-k3", input_tokens=0, output_tokens=0, source="x")
    assert L.get_ledger().summary()["count"] == 0


def test_record_llm_unknown_model_marked_estimated():
    record_llm(model="kimi-k99-secret", input_tokens=1_000_000, output_tokens=0, source="x")
    row = L.get_ledger().recent()[0]
    assert row["cost_known"] is False                # provider fallback → estimated


def test_record_tts_asr_ocr():
    record_tts(model="qwen3-tts-flash", chars=10_000)
    record_asr(model="qwen3-asr-flash", seconds=120)
    record_ocr(model="qwen3.5-ocr", input_tokens=5000, output_tokens=1000, pages=3)
    kinds = {r["key"] for r in L.get_ledger().summary()["by_kind"]}
    assert {"tts", "asr", "ocr"} <= kinds


def test_llm_usage_callback_books_from_result():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 1_000_000, "output_tokens": 500_000, "total_tokens": 1_500_000},
        response_metadata={"model_name": "kimi-k3"},
    )
    res = LLMResult(generations=[[ChatGeneration(message=msg)]])
    LlmUsageCallback(source="orchestrator", model_id="kimi-k3").on_llm_end(res)

    row = L.get_ledger().recent()[0]
    assert row["source"] == "orchestrator" and row["input_tokens"] == 1_000_000
    assert row["cost"] == pytest.approx(12.0)


def test_capture_is_fail_safe(monkeypatch):
    # A broken ledger must not propagate out of a record_* call.
    class _Boom:
        def record(self, *_a, **_k):
            raise RuntimeError("db down")
    L.set_ledger_for_test(_Boom())  # type: ignore[arg-type]
    record_llm(model="kimi-k3", input_tokens=100, output_tokens=100, source="x")  # no raise
    record_tts(model="qwen3-tts-flash", chars=100)                                # no raise


def test_callback_swallows_bad_result():
    LlmUsageCallback(source="x").on_llm_end(object())  # garbage → no raise
    assert L.get_ledger().summary()["count"] == 0
