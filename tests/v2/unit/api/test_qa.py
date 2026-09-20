"""QA (查询助手) contract tests.

Per the house test rule the router under test is NOT yet included in
``mast.api.app`` (integration wires that); we mount it on a throwaway
FastAPI app with a fresh AppContext. We assert:

  * the endpoint returns 200 + the schema-shaped body;
  * standalone (no QuickAsk backend wired) ⇒ degrades with an explanatory answer,
    never a 500;
  * an empty question is a benign no-op (not degraded);
  * wiring a FAKE QuickAskAgent drives the live relay path — the question + scope
    are forwarded verbatim to ``one_shot`` and the answer is surfaced;
  * a backend that raises degrades (degraded=True) instead of 500-ing;
  * the resolved ``qa_model`` follows the same priority as the old GUI handler.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.qa import router


# ── throwaway apps ─────────────────────────────────────────────────────


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ── fakes mirroring the kept core's shape ──────────────────────────────


class _FakeConfig:
    """Mimics ClaudeClient._config (model_alias attribute)."""

    def __init__(self, alias: str) -> None:
        self.model_alias = alias


class _FakeClient:
    """Mimics ClaudeClient (model_alias + _config.model_alias)."""

    def __init__(self, alias: str) -> None:
        self.model_alias = alias
        self._config = _FakeConfig(alias)


class _FakeQuickAsk:
    """Minimal QuickAskAgent stand-in: records (question, scope), returns canned."""

    def __init__(self, alias: str = "glm-5.2", *, raises: bool = False) -> None:
        self._client = _FakeClient(alias)
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    def one_shot(self, query: str, *, max_steps: int = 8, scope: str = "all") -> str:
        self.calls.append((query, scope))
        if self._raises:
            raise RuntimeError("boom")
        return f"answer to {query!r} [scope={scope}]"


def _wired_client(qa) -> tuple[TestClient, AppContext]:
    ctx = AppContext()
    # The leader wires a live QuickAskAgent onto the context; AppContext is a
    # plain object so we attach it the same way for the test.
    ctx.quickask = qa  # type: ignore[attr-defined]
    return _client(ctx), ctx


# ── standalone (no backend wired) degrades, never 500 ──────────────────


def test_qa_standalone_degrades_not_broken() -> None:
    r = _client().post("/api/qa", json={"question": "what is the bias?"})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert isinstance(body["answer"], str) and body["answer"]
    assert body["scope"] == "all"
    assert isinstance(body["model"], str) and body["model"]  # falls back to default


def test_qa_empty_question_is_benign_noop() -> None:
    r = _client().post("/api/qa", json={"question": "   "})
    assert r.status_code == 200
    body = r.json()
    # Empty input is a no-op, NOT a backend failure.
    assert body["degraded"] is False
    assert isinstance(body["answer"], str) and body["answer"]


def test_qa_default_scope_is_all() -> None:
    r = _client().post("/api/qa", json={"question": "hi"})
    assert r.status_code == 200
    assert r.json()["scope"] == "all"


# ── wired backend drives the live relay path ───────────────────────────


def test_qa_wired_forwards_question_and_scope() -> None:
    qa = _FakeQuickAsk(alias="glm-5.2")
    client, _ = _wired_client(qa)
    r = client.post(
        "/api/qa", json={"question": "推荐 Au(111) 偏压?", "scope": "literature"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["scope"] == "literature"
    assert body["model"] == "glm-5.2"
    assert "推荐 Au(111) 偏压?" in body["answer"]
    assert "scope=literature" in body["answer"]
    # The question + scope reached one_shot verbatim.
    assert qa.calls == [("推荐 Au(111) 偏压?", "literature")]


def test_qa_wired_default_scope_forwarded_as_all() -> None:
    qa = _FakeQuickAsk()
    client, _ = _wired_client(qa)
    r = client.post("/api/qa", json={"question": "现在偏压多少?"})
    assert r.status_code == 200
    assert r.json()["scope"] == "all"
    assert qa.calls[-1][1] == "all"


def test_qa_wired_backend_raises_degrades_not_500() -> None:
    qa = _FakeQuickAsk(raises=True)
    client, _ = _wired_client(qa)
    r = client.post("/api/qa", json={"question": "boom?"})
    assert r.status_code == 200  # never 500
    body = r.json()
    assert body["degraded"] is True
    assert "查询失败" in body["answer"]
    assert "RuntimeError" in body["answer"]


def test_qa_wired_empty_question_does_not_call_backend() -> None:
    qa = _FakeQuickAsk()
    client, _ = _wired_client(qa)
    r = client.post("/api/qa", json={"question": ""})
    assert r.status_code == 200
    assert r.json()["degraded"] is False
    assert qa.calls == []  # benign no-op never reaches one_shot


# ── qa_model resolution priority ───────────────────────────────────────


def test_qa_model_prefers_live_client_alias() -> None:
    qa = _FakeQuickAsk(alias="minimax-m3")
    client, _ = _wired_client(qa)
    r = client.post("/api/qa", json={"question": "hi"})
    assert r.json()["model"] == "minimax-m3"


def test_qa_model_falls_back_to_persisted_setting(tmp_path) -> None:
    # No live client → resolve from the persisted qa_model setting.
    from mast.webui.settings_store import SettingsStore

    store = SettingsStore(str(tmp_path))
    store.update(qa_model="deepseek-v4-pro")

    ctx = AppContext()
    ctx.wire(settings_store=store)
    r = _client(ctx).post("/api/qa", json={"question": "hi"})
    body = r.json()
    # No backend wired → degraded, but the model still resolves off settings.
    assert body["degraded"] is True
    assert body["model"] == "deepseek-v4-pro"
