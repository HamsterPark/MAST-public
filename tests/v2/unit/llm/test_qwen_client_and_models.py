"""Regression tests for the Qwen/DashScope LLM wiring.

Covers four findings (all reproduced from real code, fixed in this change):

  #78  llm/client.py — qwen thinking kwargs carry tool_choice="auto"; when a
       call has NO tools, _chat_openai_compat must NOT send tool_choice
       (DashScope 400s with "tool_choice is only allowed when tools are
       specified"). It must still send tool_choice when tools ARE present.

  #130 llm/client.py — single_turn() collected only `text` blocks; a
       reasoning-only reply (a `thinking` block, no `text`) was dropped and
       single_turn returned "". It must fall back to the thinking text.

  #99  agents/_shared/models.py — the `dashscope` key entries were dead config
       (provider_for never returned "dashscope"), so the qwen path had no key /
       no base_url. qwen is now fully wired (key file, env, base_url, fallback).

  #128 agents/_shared/models.py — provider_for() raised ValueError on any
       qwen* model id, crashing agent/orchestrator registration. It now
       resolves to "qwen".

All tests run WITHOUT network or real API keys: the OpenAI-compat HTTP client
is replaced with a fake that records the request body and returns canned JSON.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/llm/test_qwen_client_and_models.py -q -p no:randomly
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see conftest note) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest

from mast.config import LLMConfig
from mast.llm.client import ClaudeClient
from mast.agents._shared import models as M


# ── Fake OpenAI-compatible HTTP client ───────────────────────────────

class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeHttp:
    """Records the last posted body; returns a configurable canned response."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.last_body: dict | None = None
        self.headers: dict[str, str] = {}

    def post(self, _url: str, json: dict):  # noqa: A002 — mirrors httpx signature
        self.last_body = json
        return _FakeResponse(self.payload)


def _make_qwen_client(payload: dict) -> tuple[ClaudeClient, _FakeHttp]:
    """Build a ClaudeClient pinned to qwen with a fake HTTP backend (no network)."""
    cfg = LLMConfig()
    cfg.use("qwen3.7-max")
    assert cfg.provider == "qwen"
    client = ClaudeClient.__new__(ClaudeClient)  # bypass __init__/_build_backend
    client._config = cfg
    client._provider = cfg.provider
    client._model = cfg.model
    client._max_tokens = cfg.max_tokens
    client._anth = None
    fake = _FakeHttp(payload)
    client._http = fake
    return client, fake


def _qwen_payload(*, text: str = "", reasoning: str = "", finish: str = "stop") -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-fake",
        "model": "qwen3.7-max",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 11},
    }


# ── : tool_choice must not leak into tool-free requests ───

def test_no_tools_request_omits_tool_choice():
    """A qwen chat() with no tools must NOT put tool_choice in the body."""
    client, fake = _make_qwen_client(_qwen_payload(text="ok"))
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert fake.last_body is not None
    assert "tool_choice" not in fake.last_body, (
        "tool_choice leaked into a tool-free request — DashScope would 400"
    )
    assert "tools" not in fake.last_body
    # The other thinking kwargs SHOULD still be present.
    assert fake.last_body.get("enable_thinking") is True
    assert fake.last_body.get("parallel_tool_calls") is False


def test_with_tools_request_includes_tool_choice():
    """When tools ARE supplied, tool_choice must be re-added (value 'auto')."""
    client, fake = _make_qwen_client(_qwen_payload(text="ok"))
    tools = [{
        "name": "get_bias",
        "description": "read bias",
        "input_schema": {"type": "object", "properties": {}},
    }]
    client.chat(messages=[{"role": "user", "content": "read bias"}], tools=tools)
    assert fake.last_body is not None
    assert fake.last_body.get("tool_choice") == "auto"
    assert isinstance(fake.last_body.get("tools"), list)
    assert fake.last_body["tools"][0]["function"]["name"] == "get_bias"


def test_tool_choice_strip_does_not_mutate_thinking_kwargs_source():
    """Stripping tool_choice from one call must not corrupt the next call."""
    client, fake = _make_qwen_client(_qwen_payload(text="ok"))
    # First a tool-free call (strips tool_choice) ...
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert "tool_choice" not in fake.last_body
    # ... then a call WITH tools must still get tool_choice back.
    tools = [{"name": "t", "description": "", "input_schema": {"type": "object"}}]
    client.chat(messages=[{"role": "user", "content": "x"}], tools=tools)
    assert fake.last_body.get("tool_choice") == "auto"


# ── : single_turn must surface reasoning-only replies ───

def test_single_turn_returns_text_when_present():
    client, _ = _make_qwen_client(_qwen_payload(text="42", reasoning="some cot"))
    out = client.single_turn("what is 6*7?")
    assert out == "42"  # prefers the visible text over the thinking trace


def test_single_turn_falls_back_to_reasoning_when_no_text():
    """Reasoning-only reply (thinking block, empty text) must not vanish."""
    client, _ = _make_qwen_client(_qwen_payload(text="", reasoning="the answer is 42"))
    out = client.single_turn("what is 6*7?")
    assert out == "the answer is 42", (
        "single_turn dropped a reasoning-only reply and returned empty"
    )


def test_single_turn_empty_when_truly_empty():
    client, _ = _make_qwen_client(_qwen_payload(text="", reasoning=""))
    assert client.single_turn("hi") == ""


# ── : provider_for resolves qwen ────────────────────────

@pytest.mark.parametrize("mid", ["qwen3.7-max", "qwen3.7-max-2026-05-20", "qwen-anything"])
def test_provider_for_resolves_qwen(mid):
    assert M.provider_for(mid) == "qwen"


def test_provider_for_still_rejects_unknown():
    with pytest.raises(ValueError):
        M.provider_for("totally-unknown-model")


# ── : qwen is fully wired, not dead config ───────────────

def test_qwen_provider_plumbing_present():
    assert M.PROVIDER_BASE_URL.get("qwen") == \
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert M._PROVIDER_KEY_FILE["qwen"].name == "dashscope.env"
    assert "DASHSCOPE_API_KEY" in M._PROVIDER_KEY_ENV["qwen"]
    assert M._FALLBACK_MODEL_BY_PROVIDER["qwen"] == M.QWEN3_7_MAX


def test_qwen_is_reasoning_pinned_high():
    # effective_thinking is the honest-UI surface; qwen should be pinned high.
    assert M.effective_thinking(M.QWEN3_7_MAX, "low") == "high (固定)"


def test_make_chat_model_qwen_resolves_without_crashing(monkeypatch):
    """make_chat_model(model_id='qwen3.7-max') must reach construction.

    Previously provider_for() raised ValueError before any key lookup, taking
    agent registration down. We stub the key load + the langchain subclass so
    the test stays offline and dependency-free, and assert the qwen base_url /
    model id flow through.
    """
    monkeypatch.setattr(M, "load_provider_key", lambda provider: "fake-key-123")

    captured: dict = {}

    class _FakeChatClass:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        M, "make_reasoning_preserving_chat_openai_class",
        lambda: _FakeChatClass, raising=False,
    )
    # The function imports the reasoning subclass factory lazily from the
    # reasoning_chat_model module; patch that symbol too.
    import mast.agents._shared.reasoning_chat_model as rcm
    monkeypatch.setattr(
        rcm, "make_reasoning_preserving_chat_openai_class",
        lambda: _FakeChatClass,
    )

    obj = M.make_chat_model(model_id="qwen3.7-max", max_tokens=256, allow_fallback=False)
    assert isinstance(obj, _FakeChatClass)
    assert captured["model"] == "qwen3.7-max"
    assert captured["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert captured["api_key"] == "fake-key-123"
