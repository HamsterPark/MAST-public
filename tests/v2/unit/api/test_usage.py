"""Contract tests for the 用量·花销 endpoints (GET/POST /api/usage/*)."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.usage import router
from mast.billing import ledger as L
from mast.billing import pricing as P


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    L.set_ledger_for_test(L.UsageLedger(tmp_path / "u.sqlite"))
    P._CACHE = None
    P._CACHE_MTIME = None
    monkeypatch.setattr(P, "_override_path", lambda: tmp_path / "usage_pricing.json")
    yield
    L.set_ledger_for_test(None)
    P._CACHE = None
    P._CACHE_MTIME = None


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_summary_empty_is_zeroed():
    b = _client().get("/api/usage/summary?range=7d").json()
    assert b["range"] == "7d" and b["count"] == 0
    assert b["by_provider"] == []


def test_summary_after_records():
    from mast.billing.capture import record_llm
    record_llm(model="kimi-k3", input_tokens=1_000_000, output_tokens=500_000, source="orchestrator")
    record_llm(model="claude-opus-4-8", input_tokens=1_000_000, output_tokens=0, source="chat")
    b = _client().get("/api/usage/summary?range=all").json()
    assert b["count"] == 2
    assert b["by_currency"]["CNY"]["cost"] == pytest.approx(12.0)
    assert b["by_currency"]["USD"]["cost"] == pytest.approx(15.0)
    provs = {r["key"] for r in b["by_provider"]}
    assert provs == {"moonshot", "anthropic"}
    assert b["combined_cny"] > 15.0                     # USD folded via FX


def test_recent_and_reset():
    from mast.billing.capture import record_llm
    record_llm(model="kimi-k3", input_tokens=1000, output_tokens=500, source="x")
    c = _client()
    ev = c.get("/api/usage/recent?limit=10").json()["events"]
    assert len(ev) == 1 and ev[0]["provider"] == "moonshot"
    assert c.post("/api/usage/reset").json()["deleted"] == 1
    assert c.get("/api/usage/summary?range=all").json()["count"] == 0


def test_pricing_view_lists_defaults():
    b = _client().get("/api/usage/pricing").json()
    models = {m["model"] for m in b["models"]}
    assert "kimi-k3" in models and "claude-opus-4-8" in models
    assert b["usd_to_cny"] > 0


def test_pricing_edit_persists_and_affects_cost():
    c = _client()
    c.post("/api/usage/pricing", json={
        "models": [{"model": "kimi-k3", "currency": "CNY",
                    "input_per_m": 1.0, "output_per_m": 2.0}],
    })
    # new price → 1M in @1 + 1M out @2 = 3
    from mast.billing.capture import record_llm
    record_llm(model="kimi-k3", input_tokens=1_000_000, output_tokens=1_000_000, source="x")
    b = c.get("/api/usage/summary?range=all").json()
    assert b["by_currency"]["CNY"]["cost"] == pytest.approx(3.0)
