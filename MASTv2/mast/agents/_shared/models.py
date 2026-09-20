"""Centralised chat-model registry — one place to update on model churn.

Three providers are wired:
  * `moonshot`   — Kimi K2.x via Moonshot's OpenAI-compatible /v1 endpoint.
  * `deepseek`   — DeepSeek V4 via DeepSeek's OpenAI-compatible endpoint.
  * `anthropic`  — Claude via the official `langchain_anthropic.ChatAnthropic`.

Default for every agent is **Kimi K3** (per operator request 2026-07-18).
Override per agent with the AGENT_MODEL dict below or per-call via
`make_chat_model("orchestrator", model_id="deepseek-v4-pro")`.

Key files (shared with v1) live under `<repo>/api key/`:
    api key.env   — Anthropic key
    kimi.env      — Moonshot / Kimi key
    deepseek.env  — DeepSeek key

VCR cassettes in tests/behavior_pins/ are pinned to the (model, prompt) pair
that produced them — refresh them whenever you bump the AGENT_MODEL defaults.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Model-id constants ──────────────────────────────────────────────
KIMI_K2_6 = "kimi-k2.6"
# K2.7 code-specialised — a thinking/reasoning model like the rest of Kimi
# (intrinsic high reasoning, temperature forced to 1, reasoning_content round-trip).
KIMI_K2_7_CODE = "kimi-k2.7-code"
# K3 (released 2026-07-16): 2.8T open-weight, 1M context, native vision. Reasoning
# is ALWAYS-ON ("thinking mode") — same OpenAI-compat contract as the K2 line
# (thinking enabled by default, NO reasoning_effort knob on /v1, temperature pinned
# to 1.0, reasoning_content round-tripped between turns). Now the global default
# (AGENT_MODEL below + config.DEFAULT_MODEL_ALIAS). Verify the exact id + any new
# param constraints against Moonshot's live model list before a production run.
KIMI_K3 = "kimi-k3"
MOONSHOT_128K = "moonshot-v1-128k"

DEEPSEEK_V4_PRO = "deepseek-v4-pro"

OPUS_4_7 = "claude-opus-4-7"
SONNET_4_6 = "claude-sonnet-4-6"
HAIKU_4_5 = "claude-haiku-4-5-20251001"

# Qwen3.7-Max via DashScope (OpenAI-compatible "compatible-mode" endpoint).
QWEN3_7_MAX = "qwen3.7-max"

# MiniMax via its Anthropic-SDK-compatible endpoint (api.minimaxi.com/anthropic).
# M3 = text+vision+tools+thinking. Thinking is a TUNABLE Anthropic-style budget
# (NOT pinned high like Kimi/DeepSeek/Qwen). Only M3 is exposed now (M2.x retired).
MINIMAX_M3 = "MiniMax-M3"

# Zhipu GLM (OpenAI-compatible; reasons by default, returns reasoning_content).
# GLM_5_2 access OPENED 2026-06-18 (was 403「您无权访问glm-5.2」through 2026-06-15);
# it is now the global DEFAULT model (config.DEFAULT_MODEL_ALIAS) and the offered
# GLM in the pickers. GLM_5_1 is retired from the pickers but the constant stays so
# any stored "glm-5.1" agent override still normalises. Both are thinking models
# (fixed high reasoning).
GLM_5_1 = "glm-5.1"
GLM_5_2 = "glm-5.2"

# Claude models that use ADAPTIVE thinking (mirror config.ADAPTIVE_THINKING_MODELS).
# Opus 4.7/4.8 REJECT manual budget_tokens (400) — must use thinking={type:adaptive}.
_ADAPTIVE_THINKING = frozenset({
    "claude-opus-4-8", OPUS_4_7, "claude-opus-4-6", SONNET_4_6,
})

# Legacy aliases (retiring 2026-06-15) — only referenced by rollback path.
OPUS_4 = "claude-opus-4"
SONNET_4 = "claude-sonnet-4"


# ── Per-agent default model ─────────────────────────────────────────
# All agents default to KIMI-K3 per operator request 2026-07-18 (was GLM-5.2):
# Kimi K3 is the strongest open-weight model available (2.8T, released 2026-07-16).
# Switch to another model selectively at call sites or in the UI picker.
# NOTE: this is only the CODE fallback; the running system is governed by the
# persisted per-agent overrides in config/overrides/agent_overrides.json — keep
# both in sync (see core/runtime._resolve_override_models).
AGENT_MODEL: dict[str, str] = {
    "orchestrator":       KIMI_K3,
    "research_director":  KIMI_K3,
    "literature":         KIMI_K3,
    "experiment_design":  KIMI_K3,
    "instrument_control": KIMI_K3,
    "data_processing":    KIMI_K3,
    "paper_writing":      KIMI_K3,
    "paper_review":       KIMI_K3,
}


# ── Provider plumbing ───────────────────────────────────────────────

PROVIDER_BASE_URL: dict[str, str] = {
    "moonshot": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu":    "https://open.bigmodel.cn/api/paas/v4",
}

# Anthropic-SDK-compatible base URLs (consumed via ChatAnthropic anthropic_api_url,
# NOT the OpenAI /v1 path). Verified live: api.minimaxi.com (NOT api.minimax.io).
ANTHROPIC_COMPAT_BASE_URL: dict[str, str] = {
    "minimax": "https://api.minimaxi.com/anthropic",
}

# API key dir resolution — must honour MAST2_PROJECT_ROOT / data_dir.txt so
# that the bundled binary finds keys at the user-data path (e.g. <data-root>)
# instead of the empty C:\MAST2\api key\ inside the install dir. Previously
# this used a hardcoded parents[4] which silently broke orchestrator builds
# whenever the user data dir was relocated (logged 2026-05-28: launcher saw
# "Orchestrator build failed: No API key for provider 'moonshot'" while
# QuickAsk's ClaudeClient — which went through mast.config — found the same
# kimi.env without trouble).
from mast._runtime_paths import project_root as _project_root  # noqa: E402

_API_KEY_DIR = _project_root() / "api key"

_PROVIDER_KEY_FILE: dict[str, Path] = {
    "anthropic": _API_KEY_DIR / "api key.env",
    "moonshot":  _API_KEY_DIR / "kimi.env",
    "deepseek":  _API_KEY_DIR / "deepseek.env",
    # `qwen` is the provider id that provider_for() returns for qwen* chat
    # models; `dashscope` is the voice/embedding alias. Both share one key
    # file (mirrors mast.config._API_KEY_FILES). The `dashscope` entry was
    # previously dead config — provider_for() never returned "dashscope", so
    # load_provider_key("qwen") found nothing and the qwen path got no key.
    "qwen":      _API_KEY_DIR / "dashscope.env",
    "dashscope": _API_KEY_DIR / "dashscope.env",
    "minimax":   _API_KEY_DIR / "minimax.env",
    "zhipu":     _API_KEY_DIR / "glm.env",
}

_PROVIDER_KEY_ENV: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "moonshot":  ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "deepseek":  ("DEEPSEEK_API_KEY",),
    "qwen":      ("DASHSCOPE_API_KEY", "ALIYUN_BAILIAN_API_KEY"),
    "dashscope": ("DASHSCOPE_API_KEY", "ALIYUN_BAILIAN_API_KEY"),
    "minimax":   ("MINIMAX_API_KEY",),
    "zhipu":     ("ZHIPU_API_KEY", "GLM_API_KEY"),
}


def provider_for(model_id: str) -> str:
    """Infer provider from a full model id."""
    if model_id.lower().startswith("minimax"):
        return "minimax"
    if model_id.lower().startswith("glm"):
        return "zhipu"
    if model_id.startswith("claude"):
        return "anthropic"
    if model_id.startswith("kimi") or model_id.startswith("moonshot"):
        return "moonshot"
    if model_id.startswith("deepseek"):
        return "deepseek"
    if model_id.startswith("qwen"):
        # Qwen3.7-Max chat models go through DashScope's OpenAI-compatible
        # ("compatible-mode") endpoint, keyed by dashscope.env. Without this
        # branch make_chat_model(model_id="qwen3.7-max") raised ValueError and
        # took the whole agent registration down (the orchestrator build
        # propagated it as "Unknown model id: 'qwen3.7-max'").
        return "qwen"
    raise ValueError(f"Unknown model id: {model_id!r}")


def _read_key_file(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def load_provider_key(provider: str) -> str:
    """Load provider key: env var first, then provider's key file."""
    for var in _PROVIDER_KEY_ENV.get(provider, ()):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return _read_key_file(_PROVIDER_KEY_FILE.get(provider, Path()))


# ── Public factory ──────────────────────────────────────────────────

#: UI model choice id → the full model id ``make_chat_model`` understands.
#: Single source of truth: ``core/runtime._resolve_override_models`` (which builds
#: the actual chat models) and :func:`resolve_effective_model_id` (which sizes
#: context windows) MUST agree, or compaction is sized for a model the agent is
#: not running.
UI_MODEL_ALIASES: dict[str, str] = {
    "kimi-k2.6": "kimi-k2.6",
    "kimi-k2.7-code": "kimi-k2.7-code",
    "kimi-k3": KIMI_K3,
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-r1.5": "deepseek-v4-pro",
    "sonnet-4.6": SONNET_4_6,
    "haiku-4.5": HAIKU_4_5,
    "minimax-m3": MINIMAX_M3,
    "glm-5.1": GLM_5_1,
    "glm-5.2": GLM_5_2,
}


def get_model_id(agent: str) -> str:
    """The CODE-DEFAULT model id for an agent — not necessarily the running one.

    Use :func:`resolve_effective_model_id` when the answer has to match what the
    agent actually runs (window sizing, compaction thresholds). This function
    cannot see the persisted per-agent overrides, and callers that assumed it
    could were sizing compaction for a 250k window while the agent ran on a 120k
    model — i.e. compaction could only ever fire AFTER the provider had already
    rejected the request.
    """
    return AGENT_MODEL[agent]


def resolve_effective_model_id(agent: str) -> str:
    """The model id ``agent`` will ACTUALLY run with, overrides included.

    Reads the persisted per-agent override (the layer that really governs the
    running system) and normalises the UI's choice id to a full model id; falls
    back to the code default when there is no override, the registry is
    unavailable, or the stored value is unusable. Never raises — a lookup failure
    must degrade to the default, not break the turn.
    """
    default = AGENT_MODEL.get(agent) or AGENT_MODEL["orchestrator"]
    try:
        from mast.admin.override_store import ConfigOverrideRegistry
        entry = (ConfigOverrideRegistry.get().get_agent_overrides() or {}).get(agent)
    except Exception:  # noqa: BLE001 — registry optional / not yet initialised
        return default
    if not isinstance(entry, dict):
        return default
    ui_model = (entry.get("model") or "").strip()
    if not ui_model:
        return default
    return UI_MODEL_ALIASES.get(ui_model, ui_model)


# Reasoning-only models that reject custom temperatures (server forces 1.0).
# Verified empirically: moonshot returns
#   "invalid temperature: only 1 is allowed for this model"
# when calling kimi-k2.x with temperature != 1.
_FORCED_TEMPERATURE_1: frozenset[str] = frozenset({KIMI_K2_6, KIMI_K2_7_CODE, KIMI_K3})


# ── Thinking / reasoning strength ───────────────────────────────────
# Maps a UI "thinking level" to an Anthropic extended-thinking budget (tokens).
# Only Claude models accept a *tunable* budget here. Moonshot Kimi K2.x and
# DeepSeek V4 Pro are reasoning models that always think server-side at full
# strength ("high"); there is no honoured per-call effort knob for them (the
# Moonshot/DeepSeek OpenAI-compatible endpoints reject unknown params, so we do
# NOT fabricate one — that would be a no-op-pretending-to-work). For those the
# level is informational and effectively pinned to "high".
# MUST include every level the GUI per-agent dropdown offers (it uses
# config.THINKING_PRESETS = off/low/medium/high/max). If "max" is missing here,
# normalize_thinking_level("max") → None → a Claude/MiniMax agent set to "max"
# gets thinking SILENTLY DISABLED (the "params don't take effect" bug).
THINKING_LEVELS: tuple[str, ...] = ("off", "low", "medium", "high", "max")
_THINKING_BUDGET_TOKENS: dict[str, int] = {
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "max": 32768,   # agents' max tunable budget (Claude / MiniMax)
}
# Models whose reasoning is intrinsic + always-on (cannot be dialled down).
_ALWAYS_HIGH_REASONING: frozenset[str] = frozenset({KIMI_K2_6, KIMI_K2_7_CODE, KIMI_K3, DEEPSEEK_V4_PRO, QWEN3_7_MAX, GLM_5_1, GLM_5_2})

#: Never stream a call that carries tools — stream the rest normally.
#
# Streamed tool calls are reassembled from deltas, and the final parse is
# ``langchain_core.utils.json.parse_partial_json``, whose recovery loop pops
# trailing characters until what is left parses::
#
#     while new_chars:
#         try: return json.loads("".join(new_chars + stack), strict=strict)
#         except json.JSONDecodeError: new_chars.pop()
#
# A fragment cut mid-number therefore does not raise, it SUCCEEDS on a shorter
# number: ``{"p_gain": 3e-12`` truncated in the exponent parses as ``3``, and any
# cut inside ``0.0000000000030`` parses as ``0``. Silently — no log, no error.
# A streamed tool call can therefore turn incomplete numeric text into a valid
# but incorrect magnitude. Tool-call responses must be parsed as complete data.
#
# ``"tool_calling"`` rather than ``True``: langchain checks it in
# ``BaseChatModel._should_stream`` and it wins over every affirmative trigger,
# including the streaming handler langgraph attaches for messages-mode. Only
# calls that actually pass ``tools`` are affected, so summarisation, compaction
# and the other tool-less helpers keep streaming.
#
# The one visible cost is voice (``chat/engine.stream_events`` → the same
# tool-carrying IC graph): token-level events stop arriving, so a reply is spoken
# once it is complete instead of sentence by sentence. ``voice/session`` already
# falls back to the final message when no tokens streamed, so this degrades the
# cadence, not the feature. Accepted deliberately — a scan size or a bias that
# loses its exponent in transit costs more than a few seconds of latency.
_DISABLE_STREAMING_FOR_TOOLS = "tool_calling"


def normalize_thinking_level(level: str | None) -> str | None:
    """Coerce an arbitrary thinking-level string to a known level or None."""
    if level is None:
        return None
    lv = str(level).strip().lower()
    return lv if lv in THINKING_LEVELS else None


def effective_thinking(model_id: str, level: str | None) -> str:
    """What thinking strength a model will *actually* run at (for honest UI).

    Reasoning models are pinned high; Claude honours the requested level
    (defaulting to off when unspecified)."""
    if model_id in _ALWAYS_HIGH_REASONING:
        return "high (固定)"
    lv = normalize_thinking_level(level)
    return lv or "off"


# Vision capability. RE-EXPORT, not a second table: the truth lives beside
# ``model_thinking_mode`` in mast.config (``_VISION_MODELS`` +
# ``model_supports_vision``), because that is where the per-model capability
# family already is and a capability answered in two files is a capability that
# will eventually disagree with itself. Imported here so the agent-side callers
# reach it next to the other model helpers they already use.
#
# The old ``config.MINIMAX_VISION_MODELS`` stub — zero consumers, wired to
# nothing — was absorbed into that table rather than left standing next to it.
from mast.config import model_supports_vision  # noqa: E402,F401


def rejects_forced_tool_choice(model_id: str | None) -> bool:
    """Whether forced tool choice is incompatible with this reasoning model.

    Function-calling structured output pins tool_choice to a schema tool. Some
    always-on reasoning endpoints reject that combination. Skip the unsupported
    tier instead of paying for a request known to fail before text-JSON fallback.
    """
    return bool(model_id) and model_id in _ALWAYS_HIGH_REASONING


def model_id_of(model: object) -> str | None:
    """Best-effort model id from a LangChain chat model (None if unknown)."""
    for attr in ("model_name", "model", "model_id"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


_FALLBACK_MODEL_BY_PROVIDER: dict[str, str] = {
    "moonshot":  KIMI_K3,
    "deepseek":  DEEPSEEK_V4_PRO,
    "anthropic": SONNET_4_6,
    "qwen":      QWEN3_7_MAX,
    "minimax":   MINIMAX_M3,
    "zhipu":     GLM_5_1,   # the accessible GLM (5.2 is 403 until access opens)
}

# When the agent's configured provider has no key, try these providers in order.
# Picked to match the typical user setup: most have moonshot + deepseek; a
# subset have anthropic. Falling through silently avoids the orchestrator
# failing to build just because the configured Kimi key was relocated.
_PROVIDER_FALLBACK_ORDER: tuple[str, ...] = ("moonshot", "deepseek", "anthropic")


def make_chat_model(
    agent: str | None = None,
    *,
    model_id: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    thinking_level: str | None = None,
    allow_fallback: bool = True,
    request_timeout: float | None = None,
    usage_source: str | None = None,
) -> Any:
    """Construct the appropriate chat model for *agent* / *model_id*.

    Precedence:
      1. explicit `model_id` arg
      2. AGENT_MODEL[agent]
      3. raises if neither given.

    ``request_timeout`` (seconds) bounds a single HTTP call. Default None keeps
    the provider SDK's own behaviour, which is what the agent graphs want — they
    are driven by a human who can hit stop. Callers that fan work out across
    threads under a deadline (``literature.deep_read``) must set it: without it a
    stalled provider connection leaves a worker thread parked indefinitely, and
    the caller's own timeout can only stop *waiting* for that thread, not end it.

    When the resolved provider has no API key AND ``allow_fallback`` is True
    (default), this walks the provider fallback list (moonshot → deepseek →
    anthropic) and uses the first one with a configured key. This keeps the
    multi-agent orchestrator building on user installs that only have one or
    two LLM providers configured — instead of bailing entirely and pushing
    everything down the MissionPlanner single-agent path.

    Returns a langchain BaseChatModel-compatible instance:
      * `ChatAnthropic`  — for Claude models (provider="anthropic")
      * `ChatOpenAI`     — for moonshot/deepseek via OpenAI-compatible /v1.

    For Kimi reasoning models the *temperature* arg is silently coerced to 1.0
    because the Moonshot server rejects any other value.

    Raises:
        RuntimeError if no fallback provider has an API key configured.
    """
    if model_id is None:
        if agent is None:
            raise ValueError("make_chat_model: pass either agent= or model_id=")
        model_id = AGENT_MODEL[agent]

    provider = provider_for(model_id)
    api_key = load_provider_key(provider)
    if not api_key and allow_fallback:
        for cand_provider in _PROVIDER_FALLBACK_ORDER:
            if cand_provider == provider:
                continue
            if load_provider_key(cand_provider):
                cand_model = _FALLBACK_MODEL_BY_PROVIDER.get(cand_provider)
                if not cand_model:
                    continue
                logger.warning(
                    "make_chat_model: provider %r has no key — falling back to %r (%s).",
                    provider, cand_provider, cand_model,
                )
                model_id = cand_model
                provider = cand_provider
                api_key = load_provider_key(provider)
                break
    if not api_key:
        raise RuntimeError(
            f"No API key for provider {provider!r}. "
            f"Set env var or write the key to api key/{Path(_PROVIDER_KEY_FILE[provider]).name}."
        )

    # Reasoning models pin temperature to 1.0 server-side.
    if model_id in _FORCED_TEMPERATURE_1 and temperature != 1.0:
        logger.debug(
            "Coercing temperature %.2f → 1.0 for reasoning model %s",
            temperature, model_id,
        )
        temperature = 1.0

    level = normalize_thinking_level(thinking_level)

    # Cost recorder — attach to the model so every call books its token usage into
    # the 用量·花销 ledger. Fail-safe: a billing import/build error never blocks
    # model construction (the LLM call matters, the accounting is best-effort).
    #
    # ``usage_source`` overrides the ledger's ``source`` dimension WITHOUT touching
    # which model gets built. Needed because ``source`` is the ONLY per-call
    # attribution the ledger has, and it is derived from ``agent`` — so a caller
    # that legitimately runs on another agent's model profile (the campaign gate
    # seat runs on "orchestrator") books its spend under that agent's name and
    # becomes impossible to separate afterwards. Default None = unchanged.
    #
    # Prompt capture below deliberately keeps ``agent``: it answers "what did this
    # agent actually receive", which is a different question from "who spent this".
    _bill_cb: list = []
    try:
        from mast.billing.capture import LlmUsageCallback
        _bill_cb.append(LlmUsageCallback(
            source=(usage_source or agent or "adhoc"),
            model_id=model_id, provider=provider))
    except Exception:  # noqa: BLE001
        pass

    # Prompt capture — keeps the last few REAL message lists so 高级管理 → 上下文注入
    # can show what an agent actually received, instead of a dry-run guess that
    # may not match. Bounded, in-memory, never persisted. Same fail-safe rule.
    try:
        from mast.prompts.capture import make_callback as _prompt_capture_cb
        _bill_cb.append(_prompt_capture_cb(source=(agent or "adhoc"),
                                           model_id=model_id, provider=provider))
    except Exception:  # noqa: BLE001
        pass

    if provider in ("anthropic", "minimax"):
        from langchain_anthropic import ChatAnthropic
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "api_key": api_key,
            "disable_streaming": _DISABLE_STREAMING_FOR_TOOLS,
        }
        if _bill_cb:
            kwargs["callbacks"] = _bill_cb
        if request_timeout is not None:
            kwargs["default_request_timeout"] = float(request_timeout)
        # MiniMax speaks the Anthropic messages API at a custom base URL; route
        # ChatAnthropic there. Thinking is tunable for MiniMax too (it is NOT in
        # _ALWAYS_HIGH_REASONING), so the budget block below applies as for Claude.
        if provider == "minimax":
            kwargs["anthropic_api_url"] = ANTHROPIC_COMPAT_BASE_URL["minimax"]
        # Thinking requires temperature == 1. Opus 4.6/4.7/4.8 + Sonnet 4.6 use
        # ADAPTIVE thinking (manual budget_tokens is REJECTED on Opus 4.7/4.8);
        # everything else (Haiku, older Claude, MiniMax) uses manual budget_tokens.
        if level and level != "off":
            kwargs["temperature"] = 1.0
            if model_id in _ADAPTIVE_THINKING:
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {"effort": level}
                if max_tokens < 16000:
                    kwargs["max_tokens"] = 16000
                logger.debug("Anthropic %s adaptive-thinking effort=%s (max_tokens=%d)",
                             model_id, level, kwargs["max_tokens"])
            else:
                budget = _THINKING_BUDGET_TOKENS[level]
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                if max_tokens <= budget:
                    kwargs["max_tokens"] = budget + 4096
                logger.debug("Anthropic %s extended-thinking %s (budget=%d, max_tokens=%d)",
                             model_id, level, budget, kwargs["max_tokens"])
        return ChatAnthropic(**kwargs)

    # OpenAI-compatible providers (moonshot / deepseek) — go via ChatOpenAI
    # with a custom base_url. Both endpoints implement /v1/chat/completions.
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401  used inside subclass
    except ImportError as e:  # pragma: no cover — install path
        raise RuntimeError(
            "langchain-openai is required for provider={!r}. "
            "Install it into the v2 venv: "
            ".venv-v2-py313/Scripts/pip install langchain-openai".format(provider)
        ) from e
    base_url = PROVIDER_BASE_URL[provider]

    # Reasoning/thinking models (Kimi K2.x, DeepSeek V4 Pro) require that
    # `reasoning_content` is round-tripped between turns. LangChain's
    # ChatOpenAI 1.2.x does NOT preserve reasoning_content (both
    # _convert_dict_to_message and _convert_message_to_dict drop it),
    # so any multi-turn tool_call flow fails on turn 2 with::
    #     400: thinking is enabled but reasoning_content is missing
    #     in assistant tool call message at index N
    # Setting enable_thinking=False alone doesn't fix it (the server
    # still expects the round-trip). Our subclass preserves it.
    REASONING_MODELS = {KIMI_K2_6, KIMI_K2_7_CODE, KIMI_K3, DEEPSEEK_V4_PRO, QWEN3_7_MAX, GLM_5_1, GLM_5_2}
    if level and model_id in REASONING_MODELS and level != "high":
        # Honest: these reasoning models always think at full strength; we don't
        # silently inject an effort knob the endpoint would reject. Surfaced as
        # "high (固定)" by effective_thinking() so the UI doesn't claim otherwise.
        logger.debug("%s is a reasoning model — thinking pinned high (requested %s)",
                     model_id, level)
    # Reasoning models stream a chain-of-thought BEFORE the visible answer, so a
    # small max_tokens (agents pass 2048–4096) truncates the CoT — Kimi K2.6 docs
    # require ≥16000; DeepSeek's CoT counts against max_tokens (default 32K). Floor
    # it so the reasoning trace + answer fit. max_tokens is a CAP, not a forced
    # length, so this only PREVENTS truncation; it does not inflate normal output.
    if model_id in REASONING_MODELS and max_tokens < 16000:
        logger.debug("Flooring max_tokens %d → 16000 for reasoning model %s "
                     "(avoid CoT truncation)", max_tokens, model_id)
        max_tokens = 16000

    if model_id in REASONING_MODELS:
        from mast.agents._shared.reasoning_chat_model import (
            make_reasoning_preserving_chat_openai_class,
        )
        ChatClass = make_reasoning_preserving_chat_openai_class()
    else:
        from langchain_openai import ChatOpenAI as ChatClass

    open_kwargs: dict[str, Any] = {
        "model": model_id,
        "api_key": api_key,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "callbacks": _bill_cb,
        "disable_streaming": _DISABLE_STREAMING_FOR_TOOLS,
    }
    if request_timeout is not None:
        open_kwargs["timeout"] = float(request_timeout)
    return ChatClass(**open_kwargs)


__all__ = [
    "KIMI_K2_6", "KIMI_K2_7_CODE", "KIMI_K3", "MOONSHOT_128K",
    "DEEPSEEK_V4_PRO",
    "OPUS_4_7", "SONNET_4_6", "HAIKU_4_5",
    "QWEN3_7_MAX",
    "MINIMAX_M3",
    "GLM_5_1", "GLM_5_2",
    "OPUS_4", "SONNET_4",
    "AGENT_MODEL", "PROVIDER_BASE_URL", "ANTHROPIC_COMPAT_BASE_URL",
    "provider_for", "load_provider_key",
    "get_model_id", "make_chat_model",
    "THINKING_LEVELS", "normalize_thinking_level", "effective_thinking",
    "model_supports_vision",
]

