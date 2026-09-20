"""MiniMax (Anthropic-compatible provider) wiring + the per-model thinking-param
gating fix.

Covers the operator-reported bugs:
  • suspicion 3 — moonshot-v1-128k (chat-only) must NOT be sent thinking params;
    "fixed" reasoning models (kimi/deepseek/qwen) still get them.
  • MiniMax routing — provider inference, base_url, thinking_enabled (tunable),
    Anthropic SDK path (NOT the OpenAI-compat path), and make_chat_model.

No network: the OpenAI-compat body is captured via a fake httpx client.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py note) ──
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

import pytest  # noqa: E402

from mast.config import (  # noqa: E402
    ANTHROPIC_COMPAT_BASE_URL,
    LLMConfig,
    MODEL_PRESETS,
    _provider_for_model,
    model_thinking_mode,
)


# ── config: provider inference + capability table ───────────────────────

def test_minimax_provider_inference():
    assert _provider_for_model("MiniMax-M3") == "minimax"
    # prefix inference still routes any minimax* id, even retired ones
    assert _provider_for_model("minimax-anything") == "minimax"


def test_minimax_presets_registered():
    # Only M3 is exposed now (M2.x retired 2026-06-15).
    assert "minimax-m3" in MODEL_PRESETS
    mid, prov, _desc, _mt = MODEL_PRESETS["minimax-m3"]
    assert prov == "minimax"
    assert mid.startswith("MiniMax")
    for retired in ("minimax-m2.7", "minimax-m2.5", "minimax-m2.1", "minimax-m2"):
        assert retired not in MODEL_PRESETS, retired


@pytest.mark.parametrize("model_id,expected", [
    ("MiniMax-M3", "tunable"),
    ("claude-opus-4-7", "tunable"),
    ("kimi-k2.6", "fixed"),
    ("deepseek-v4-pro", "fixed"),
    ("qwen3.7-max", "fixed"),
    ("moonshot-v1-128k", "none"),   # the bug: must NOT receive thinking params
])
def test_model_thinking_mode(model_id, expected):
    assert model_thinking_mode(model_id) == expected


def test_minimax_llmconfig_use():
    c = LLMConfig()
    c.use("minimax-m3")
    assert c.provider == "minimax"
    assert c.model == "MiniMax-M3"
    assert c.base_url == ANTHROPIC_COMPAT_BASE_URL["minimax"]
    assert c.base_url.startswith("https://api.minimaxi.com")
    assert c.thinking_enabled is True   # tunable budget applies to MiniMax
    # Switching back to a fixed-reasoning model drops the tunable budget gate.
    c.use("kimi-k2.6")
    assert c.thinking_enabled is False


# ── client: thinking-param gating in the OpenAI-compat body ─────────────

class _FakeResp:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "x", "model": "m", "choices": [
                {"message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class _FakeHttp:
    def __init__(self):
        self.last_body = None
        self.headers = {}

    def post(self, _path, json=None):
        self.last_body = json
        return _FakeResp()

    def close(self):
        pass


def _capture_body(model_alias: str) -> dict:
    from mast.llm.client import ClaudeClient
    cfg = LLMConfig()
    cfg.use(model_alias)
    cfg.api_key = "test-key"           # avoid empty-key paths
    client = ClaudeClient(cfg)
    fake = _FakeHttp()
    client._http = fake                # bypass the real OpenAI-compat backend
    client.chat(messages=[{"role": "user", "content": "hi"}])
    return fake.last_body


# NOTE: test_moonshot_128k_gets_no_thinking_params removed 2026-06-23 — the
# moonshot-128k (chat-only "none") preset was dropped per operator request. The
# "none model never gets thinking" classification is still guarded by the
# model_thinking_mode parametrization above (("moonshot-v1-128k", "none")).


def test_fixed_reasoning_models_still_get_thinking():
    """Kimi / DeepSeek (fixed reasoning) keep their provider thinking kwargs."""
    kimi = _capture_body("kimi-k2.6")
    assert kimi.get("thinking") == {"type": "enabled", "keep": "all"}
    ds = _capture_body("deepseek-v4-pro")
    assert ds.get("thinking") == {"type": "enabled"}
    assert ds.get("reasoning_effort") == "max"


# ── agents path: make_chat_model routes MiniMax through ChatAnthropic ────

def test_make_chat_model_minimax_uses_anthropic_with_base_url(monkeypatch):
    import mast.agents._shared.models as M
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    m = M.make_chat_model(model_id="MiniMax-M3", max_tokens=128,
                          temperature=0.3, allow_fallback=False)
    assert type(m).__name__ == "ChatAnthropic"
    assert str(getattr(m, "anthropic_api_url", "")).startswith("https://api.minimaxi.com")
    # MiniMax thinking is honoured (NOT pinned 固定 like Kimi/DeepSeek/Qwen).
    assert M.effective_thinking("MiniMax-M3", "medium") == "medium"


# ── Zhipu GLM-5.2 (OpenAI-compatible reasoning provider) ────────────────

def test_glm_provider_and_preset():
    # glm-5.2 is the ACTIVE/DEFAULT GLM preset (access opened 2026-06-18). glm-5.1
    # was retired from the presets/pickers in favour of 5.2, but a stored "glm-5.1"
    # still resolves its provider via _provider_for_model (use()'s else-branch), so
    # old configs / agent overrides don't break.
    assert _provider_for_model("glm-5.1") == "zhipu"
    assert _provider_for_model("glm-5.2") == "zhipu"
    assert _provider_for_model("glm-4.6") == "zhipu"
    assert "glm-5.2" in MODEL_PRESETS
    assert "glm-5.1" not in MODEL_PRESETS          # retired in favour of 5.2
    mid, prov, _d, _mt = MODEL_PRESETS["glm-5.2"]
    assert (mid, prov) == ("glm-5.2", "zhipu")
    # GLM reasons by default → "fixed" (thinking always sent, not user-off) — for
    # both ids (model_thinking_mode keys on provider=zhipu, not on the presets).
    assert model_thinking_mode("glm-5.2") == "fixed"
    assert model_thinking_mode("glm-5.1") == "fixed"


def test_glm_llmconfig_use():
    c = LLMConfig()
    c.use("glm-5.1")
    assert c.provider == "zhipu"
    assert c.base_url == "https://open.bigmodel.cn/api/paas/v4"


def test_glm_sends_thinking_enabled():
    body = _capture_body("glm-5.1")
    assert body.get("thinking") == {"type": "enabled"}


def test_glm_agents_path_is_reasoning_with_floor(monkeypatch):
    import mast.agents._shared.models as M
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    # both GLM ids are reasoning models with the 16000 floor
    for gid in ("glm-5.1", "glm-5.2"):
        m = M.make_chat_model(model_id=gid, max_tokens=4096, allow_fallback=False)
        assert type(m).__name__ == "ReasoningPreservingChatOpenAI"
        assert m.max_tokens == 16000          # floored to avoid CoT truncation
        assert M.effective_thinking(gid, "medium") == "high (固定)"


def test_kimi_k2_7_code_is_reasoning(monkeypatch):
    """The new K2.7 code model is a thinking model like the rest of Kimi:
    fixed thinking mode + temperature forced to 1 + 16000 max_tokens floor."""
    import mast.agents._shared.models as M
    assert _provider_for_model("kimi-k2.7-code") == "moonshot"
    assert "kimi-k2.7-code" in MODEL_PRESETS
    assert "kimi-k2.5" not in MODEL_PRESETS          # retired 2026-06-15
    assert model_thinking_mode("kimi-k2.7-code") == "fixed"
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    m = M.make_chat_model(model_id="kimi-k2.7-code", max_tokens=4096,
                          temperature=0.2, allow_fallback=False)
    assert type(m).__name__ == "ReasoningPreservingChatOpenAI"
    assert m.max_tokens == 16000          # floored to avoid CoT truncation
    assert m.temperature == 1.0            # Kimi reasoning models force temp=1


# ── Anthropic adaptive thinking (Opus 4.6/4.7/4.8 + Sonnet 4.6) ──────────

class _FakeUsage:
    input_tokens = 1
    output_tokens = 1


class _FakeAnthResp:
    id = "x"
    role = "assistant"
    model = "m"
    stop_reason = "end_turn"
    content: list = []
    usage = _FakeUsage()


class _FakeAnth:
    def __init__(self):
        self.last = None
        outer = self

        class _Msgs:
            def create(self, **kw):
                outer.last = kw
                return _FakeAnthResp()

            def stream(self, **kw):
                outer.last = kw
                raise AssertionError("unexpected stream path in test")

        self.messages = _Msgs()


def _capture_anthropic_kwargs(model_alias: str, thinking_level: str | None = None) -> dict:
    from mast.llm.client import ClaudeClient
    cfg = LLMConfig()
    cfg.use(model_alias)
    cfg.api_key = "test-key"
    if thinking_level is not None:
        cfg.set_thinking(thinking_level)
    client = ClaudeClient(cfg)
    fake = _FakeAnth()
    client._anth = fake
    client.chat(messages=[{"role": "user", "content": "hi"}])
    return fake.last


def test_opus_uses_adaptive_thinking_not_budget():
    """Opus 4.7 REJECTS manual budget_tokens — must use thinking={type:adaptive}."""
    kw = _capture_anthropic_kwargs("opus")  # claude-opus-4-7, default max
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kw["output_config"] == {"effort": "max"}
    assert "budget_tokens" not in kw.get("thinking", {})


def test_haiku_uses_manual_budget_tokens():
    """Haiku does NOT support adaptive → manual budget_tokens path."""
    kw = _capture_anthropic_kwargs("haiku", thinking_level="low")
    assert kw["thinking"]["type"] == "enabled"
    assert "budget_tokens" in kw["thinking"]


# ── fix 1: Haiku max-thinking must clamp to the model output ceiling (64K) ──

class _FakeStreamCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return _FakeAnthResp()


class _FakeAnthStreamable(_FakeAnth):
    """Anthropic fake whose .stream() captures kwargs (max_tokens>16384 streams)."""

    def __init__(self):
        self.last = None
        outer = self

        class _Msgs:
            def create(self, **kw):
                outer.last = kw
                return _FakeAnthResp()

            def stream(self, **kw):
                outer.last = kw
                return _FakeStreamCtx()

        self.messages = _Msgs()


def _capture_anthropic_kwargs_streamable(model_alias, thinking_level):
    from mast.llm.client import ClaudeClient
    cfg = LLMConfig()
    cfg.use(model_alias)
    cfg.api_key = "test-key"
    cfg.set_thinking(thinking_level)
    client = ClaudeClient(cfg)
    fake = _FakeAnthStreamable()
    client._anth = fake
    client.chat(messages=[{"role": "user", "content": "hi"}])
    return fake.last


def test_haiku_max_thinking_clamped_to_model_output_limit():
    """thinking=max on Haiku must NOT push max_tokens to 128000 (400s — Haiku caps 64K).

    Regression for the hardcoded ``max_output = 128000`` in the manual-budget path:
    Haiku 4.5's output ceiling is 64000, so the bump must clamp there and budget_tokens
    must stay strictly below it.
    """
    kw = _capture_anthropic_kwargs_streamable("haiku", "max")
    assert kw["max_tokens"] == 64000, kw["max_tokens"]
    assert kw["thinking"]["type"] == "enabled"
    assert kw["thinking"]["budget_tokens"] < kw["max_tokens"]


def test_model_output_limit_table():
    from mast.config import model_output_limit
    assert model_output_limit("claude-haiku-4-5-20251001") == 64000
    assert model_output_limit("claude-opus-4-7") == 128000
    assert model_output_limit("MiniMax-M3") == 64000
    # Unknown model → conservative 64000 fallback (never over a real ceiling).
    assert model_output_limit("some-future-model") == 64000


# ── fix 5: OpenAI-compat path retries 429/5xx/network, never a 4xx ──

def _mk_http_response(status: int):
    import httpx
    req = httpx.Request("POST", "http://x/chat/completions")
    body = {
        "id": "x", "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    return httpx.Response(status, json=body, request=req)


class _SeqHttp:
    """Returns responses with the given status sequence; counts calls."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0
        self.headers = {}

    def post(self, _path, json=None):
        self.calls += 1
        return _mk_http_response(self.statuses.pop(0))

    def close(self):
        pass


def _fast_retry_client(alias="kimi-k2.6"):
    from mast.llm.client import ClaudeClient
    cfg = LLMConfig()
    cfg.use(alias)
    cfg.api_key = "test-key"
    c = ClaudeClient(cfg)
    c._OPENAI_RETRY_BASE_DELAY = 0.0   # no real sleeping in tests
    c._OPENAI_RETRY_MAX_DELAY = 0.0
    return c


def test_openai_compat_retries_5xx_then_succeeds():
    c = _fast_retry_client()
    c._http = _SeqHttp([503, 503, 200])
    out = c.chat(messages=[{"role": "user", "content": "hi"}])
    assert c._http.calls == 3
    assert out["content"][0]["text"] == "ok"


def test_openai_compat_retries_429_then_succeeds():
    c = _fast_retry_client()
    c._http = _SeqHttp([429, 200])
    c.chat(messages=[{"role": "user", "content": "hi"}])
    assert c._http.calls == 2


def test_openai_compat_does_not_retry_4xx():
    import httpx
    c = _fast_retry_client()
    c._http = _SeqHttp([400, 400, 400, 400])
    with pytest.raises(httpx.HTTPStatusError):
        c.chat(messages=[{"role": "user", "content": "hi"}])
    assert c._http.calls == 1   # a deterministic client error is never retried


def test_openai_compat_retries_transient_network_error():
    import httpx

    class _NetFlaky:
        def __init__(self):
            self.calls = 0
            self.headers = {}

        def post(self, _path, json=None):
            self.calls += 1
            if self.calls < 3:
                raise httpx.ConnectError("boom")
            return _mk_http_response(200)

        def close(self):
            pass

    c = _fast_retry_client()
    c._http = _NetFlaky()
    c.chat(messages=[{"role": "user", "content": "hi"}])
    assert c._http.calls == 3


def test_openai_compat_gives_up_after_max_retries():
    import httpx
    c = _fast_retry_client()
    # Always 500 → 1 initial + 3 retries = 4 calls, then raise.
    c._http = _SeqHttp([500, 500, 500, 500])
    with pytest.raises(httpx.HTTPStatusError):
        c.chat(messages=[{"role": "user", "content": "hi"}])
    assert c._http.calls == c._OPENAI_MAX_RETRIES + 1


# ── fix 2: a ClaudeClient owns a PRIVATE config copy (no cross-client bleed) ──

def test_client_does_not_mutate_shared_config():
    from mast.llm.client import ClaudeClient
    shared = LLMConfig()                 # default now kimi-k3 (moonshot)
    shared.api_key = "kimi-key"
    main = ClaudeClient(shared)
    qa = ClaudeClient(shared)            # QA shares the SAME config object
    qa.use("deepseek-v4-pro")            # switch QA model at runtime
    # The shared config object the GUI keeps for the main chat is untouched...
    assert shared.provider == "moonshot"
    assert shared.model == MODEL_PRESETS["kimi-k3"][0]
    # ...and the two clients hold independent providers.
    assert main.provider == "moonshot"
    assert qa.provider == "deepseek"


# ── fix 3: switching to a key-less provider must not leak the old key ──

def test_use_clears_key_when_new_provider_has_none(monkeypatch, tmp_path):
    # Point the project root at an empty dir so NO provider key file exists, and
    # clear every provider env var, so deepseek genuinely has no key.
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY",
                "MOONSHOT_API_KEY", "KIMI_API_KEY", "DASHSCOPE_API_KEY",
                "ALIYUN_BAILIAN_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    import importlib
    import mast.config as C
    importlib.reload(C)
    cfg = C.LLMConfig()
    cfg.api_key = "leaky-kimi-key"
    cfg.use("deepseek-v4-pro")
    assert cfg.api_key == "", repr(cfg.api_key)   # NOT the stale kimi key
    importlib.reload(C)   # restore module state for any later test


# ── fix 4: DeepSeek presets honour the ≥16000 truncation floor ──

def test_deepseek_presets_meet_16000_floor():
    # deepseek-v4-flash retired 2026-06-15 — only Pro remains.
    assert "deepseek-v4-flash" not in MODEL_PRESETS
    _mid, prov, _desc, max_tokens = MODEL_PRESETS["deepseek-v4-pro"]
    assert prov == "deepseek"
    assert max_tokens >= 16000


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
