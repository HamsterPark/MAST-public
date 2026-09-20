"""Pydantic configuration: host, ports, safety limits, API keys."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field

from mast._runtime_paths import project_root

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Lazily resolve the project root on EACH access.

    : ``_PROJECT_ROOT = project_root()`` froze the root at *import*
    time. Every other call site (``_runtime_paths.project_root()``, the GUI,
    the launcher) resolves it lazily, honouring ``MAST2_PROJECT_ROOT`` /
    ``sys.frozen`` whenever they're consulted. A config instance built after the
    env var changed (tests set it, the frozen launcher sets it before importing
    mast) therefore disagreed with everyone else. Routing the model defaults
    through a ``default_factory`` that calls ``project_root()`` per-construction
    keeps config paths consistent with the lazy resolver.
    """
    return project_root()


# api key/ dir is resolved per-call too so it tracks the lazy project root.
def _api_key_dir() -> Path:
    return _project_root() / "api key"


# Module-level helpers below still need the api-key directory at import time to
# build the per-provider file map. They use the lazy resolver so a late env-var
# override is honoured the first time a key is actually read (the map stores
# the resolved paths; if you need a different root, set the env var before
# importing mast.config — same contract as every other path consumer).
_PROJECT_ROOT = _project_root()
_API_KEY_DIR = _api_key_dir()

# Per-provider key files. Each holds one key on the first non-comment line.
_API_KEY_FILES: dict[str, Path] = {
    "anthropic": _API_KEY_DIR / "api key.env",  # legacy filename — keeps v1 setups working
    "moonshot":  _API_KEY_DIR / "kimi.env",
    "deepseek":  _API_KEY_DIR / "deepseek.env",
    "dashscope": _API_KEY_DIR / "dashscope.env",
    # Qwen3.7-Max via DashScope shares the same key file as the voice
    # models (qwen3-tts-flash / qwen3-asr-flash), so users don't need to
    # provision a second key. Kept as an alias entry rather than a
    # symlink so the loader logic stays trivial.
    "qwen":      _API_KEY_DIR / "dashscope.env",
    # MiniMax exposes an Anthropic-SDK-compatible endpoint (see
    # ANTHROPIC_COMPAT_BASE_URL); provisioned via its own key file / env var.
    "minimax":   _API_KEY_DIR / "minimax.env",
    # Zhipu GLM (OpenAI-compatible endpoint open.bigmodel.cn/api/paas/v4).
    "zhipu":     _API_KEY_DIR / "glm.env",
}

# Per-provider env var precedence over the file.
_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "moonshot":  ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "deepseek":  ("DEEPSEEK_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY", "ALIYUN_BAILIAN_API_KEY"),
    "qwen":      ("DASHSCOPE_API_KEY", "ALIYUN_BAILIAN_API_KEY"),
    "minimax":   ("MINIMAX_API_KEY",),
    "zhipu":     ("ZHIPU_API_KEY", "GLM_API_KEY"),
}

# OpenAI-compatible base URLs for non-Anthropic providers.
PROVIDER_BASE_URL: dict[str, str] = {
    "moonshot": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu":    "https://open.bigmodel.cn/api/paas/v4",
}

# Anthropic-SDK-COMPATIBLE base URLs. These providers speak the Anthropic
# /v1/messages wire format, so they reuse the Anthropic code path in
# ClaudeClient / make_chat_model with a custom base_url (NOT the OpenAI-compat
# httpx path). Plain "anthropic" resolves to "" → the SDK default api.anthropic.com.
ANTHROPIC_COMPAT_BASE_URL: dict[str, str] = {
    # Verified live: api.minimaxi.com (NOT api.minimax.io, which 401s) accepts
    # the Anthropic SDK's x-api-key against /anthropic/v1/messages.
    "minimax": "https://api.minimaxi.com/anthropic",
}

# Providers that speak the Anthropic message format (tunable thinking budget,
# top_k/stop_sequences omitted from the body, ChatAnthropic on the agents path).
ANTHROPIC_COMPAT_PROVIDERS: frozenset[str] = frozenset({"anthropic", "minimax"})


def _read_key_file(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def _load_provider_key(provider: str) -> str:
    """Load the API key for *provider*: env var first, then provider's key file."""
    for var in _API_KEY_ENV.get(provider, ()):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return _read_key_file(_API_KEY_FILES.get(provider, Path()))


def _load_api_key() -> str:
    """Backwards-compatible loader (Anthropic key)."""
    return _load_provider_key("anthropic")


def _port_env(var: str, default: int) -> int:
    """端口的环境变量覆盖 —— **0 = 显式停用这个角色**。

    为什么需要:这四个端口从前只有代码里的默认值,没有任何外部入口。运维可能需要
    腾出一个 Nanonis 端口给外部直连(例如给应急设备留出手动急停通道),而唯一的
    办法竟然是改代码重新打包 —— 一个纯运维决定不该要一次发版。

    非法值一律**忽略并保留默认**,不夹紧:一个写错的端口号静静变成另一个端口,
    比启动失败危险得多(会连上错的东西)。
    """
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        logger.warning("%s=%r 不是整数,忽略,仍用 %d", var, raw, default)
        return default
    if val == 0:
        logger.warning("%s=0 —— 该 Nanonis 角色**显式停用**", var)
        return 0
    if not (1 <= val <= 65535):
        logger.warning("%s=%d 不是合法端口,忽略,仍用 %d", var, val, default)
        return default
    return val


class NanonisConfig(BaseModel):
    """Nanonis TCP connection settings.

    四个端口都可用环境变量覆盖(``MAST_NANONIS_PORT_MAIN`` /
    ``_MONITOR`` / ``_DATA`` / ``_EMERGENCY``),**设成 0 即停用该角色**。
    停用的代价见 ``ConnectionPool.connect_all`` 里那段注释。
    """
    host: str = "127.0.0.1"
    port_main: int = Field(default_factory=lambda: _port_env("MAST_NANONIS_PORT_MAIN", 6501))
    port_monitor: int = Field(default_factory=lambda: _port_env("MAST_NANONIS_PORT_MONITOR", 6502))
    port_data: int = Field(default_factory=lambda: _port_env("MAST_NANONIS_PORT_DATA", 6503))
    port_emergency: int = Field(default_factory=lambda: _port_env("MAST_NANONIS_PORT_EMERGENCY", 6504))
    timeout_s: float = 5.0
    # Separate, much shorter timeout JUST for the socket.connect() call.
    # Windows ConnectionRefused/dropped paths can block up to ~2 s per port
    # under SYN-retry semantics; with 4 ports that was 8 s of GUI main-thread
    # stall on startup + on every reconnect tick. recv operations (Nanonis
    # API calls) still get the full `timeout_s`.
    connect_timeout_s: float = 0.3

    @property
    def all_ports(self) -> list[int]:
        return [self.port_main, self.port_monitor, self.port_data, self.port_emergency]


class SafetyLimits(BaseModel):
    """Hardware safety boundaries."""
    bias_min_v: float = -10.0
    bias_max_v: float = 10.0
    z_min_m: float = 0.0
    z_max_m: float = 1.5e-6
    xy_min_m: float = -1.5e-6
    xy_max_m: float = 1.5e-6
    setpoint_min_a: float = 1e-12   # 1 pA
    setpoint_max_a: float = 100e-9  # 100 nA
    # Scan size — caps width_m / height_m so the LLM can't request
    # 2-meter scans (we saw width_m=2 succeed without these in v0.3.5
    # because xy_*_m only cover the *center*, not the *extent*).
    scan_size_min_m: float = 1e-10  # 0.1 nm
    scan_size_max_m: float = 1e-5   # 10 µm (way bigger than any STM scan)
    # Relative Z offset (STS retract distance, Z-spectroscopy start/end). This is
    # a *signed relative* displacement of the fine-Z piezo, NOT an absolute Z
    # position — a legitimate STS retract is NEGATIVE. It must therefore NOT be
    # clamped against the absolute z_min_m/z_max_m floor (which starts at 0.0 and
    # would reject every negative retract). ±1.6 µm brackets the full fine-Z range.
    z_offset_min_m: float = -1.6e-6
    z_offset_max_m: float = 1.6e-6
    # Tip-shaper plunge/lift excursion (tip_lift_m / lift_height_m / deep_depth_m).
    # Typical values are ±2 nm; the per-skill ParameterSpec historically allowed
    # ±1 µm (500×), with NO global backstop. Cap at ±100 nm so a hallucinated
    # half-micron plunge is rejected before it reaches the piezo, while still
    # allowing an aggressive (tens-of-nm) intentional poke.
    tip_lift_min_m: float = -1e-7
    tip_lift_max_m: float = 1e-7


# Model presets: short alias → (model_id, provider, description, default_max_tokens)
# Provider is one of "anthropic" | "moonshot" | "deepseek".
MODEL_PRESETS: dict[str, tuple[str, str, str, int]] = {
    # ── Moonshot / Kimi (default provider) ──
    # max_tokens ≥ 16k per Kimi K2.6 docs so full reasoning_content + answer
    # aren't truncated (8192 risked cutting off the chain-of-thought).
    "kimi-k2.6":       ("kimi-k2.6",                "moonshot",  "Kimi K2.6 — 256k context, vision + reasoning", 16384),
    "kimi-k2.7-code":  ("kimi-k2.7-code",           "moonshot",  "Kimi K2.7 Code — 代码/工具推理 (思考模型)",      16384),
    # K3 (2026-07-16): 2.8T 开源最强, 1M ctx, 原生视觉, reasoning 常开. 全局默认模型。
    "kimi-k3":         ("kimi-k3",                   "moonshot",  "Kimi K3 — 2.8T 开源最强, 1M ctx, 视觉+思维 (默认模型)", 16384),
    # moonshot-128k (moonshot-v1-128k, chat-only no-reasoning) removed 2026-06-23
    # per operator request — the thinking-capable Kimi presets above supersede it.
    # ── DeepSeek ──
    # max_tokens ≥ 16k so the reasoning_content + answer aren't truncated. DeepSeek V4
    # thinking mode folds the chain-of-thought INTO the output budget (reasoner doc:
    # default 32K, max 64K, CoT counted), so the old 8192/4096 presets risked cutting
    # off the answer mid-reasoning — the same 16000-floor truncation defence applied to
    # Kimi K2.6. max_tokens is a ceiling, not a forced spend: raising it only prevents
    # truncation, it does not lengthen a normal reply.
    "deepseek-v4-pro":   ("deepseek-v4-pro",        "deepseek",  "DeepSeek V4 Pro — strong reasoning",          16384),
    # ── Anthropic Claude ──
    "opus":            ("claude-opus-4-7",          "anthropic", "Anthropic Opus 4.7 — top-tier reasoning",     16384),
    "sonnet":          ("claude-sonnet-4-6",        "anthropic", "Anthropic Sonnet 4.6 — balanced",              8192),
    "haiku":           ("claude-haiku-4-5-20251001","anthropic", "Anthropic Haiku 4.5 — fastest",                4096),
    # ── Qwen via DashScope (compatible-mode OpenAI API) ──
    # 1 M context, 65 536 max output. Shares dashscope.env with the
    # voice models. ``enable_thinking`` defaults to True per Anthropic-
    # style reasoning gate set by ``ClaudeClient._thinking_body_kwargs``.
    "qwen3.7-max":         ("qwen3.7-max",                 "qwen", "通义千问 3.7 Max — 1 M ctx, 满血思维链",        65536),
    "qwen3.7-max-preview": ("qwen3.7-max-2026-05-20",      "qwen", "通义千问 3.7 Max 预览 — 快照版,可复现",         65536),
    # ── MiniMax (Anthropic-SDK-compatible endpoint) ──
    # Only M3 is exposed now (text+image+video+tools+thinking; see
    # _VISION_MODELS below). Tunable Anthropic-style thinking budget; top_k /
    # stop_sequences are ignored by the endpoint (already absent from our body).
    "minimax-m3":   ("MiniMax-M3",   "minimax", "MiniMax M3 — Anthropic 兼容, 文本+视觉+工具+思维", 8192),
    # ── Zhipu GLM (OpenAI-compatible; reasons by default, returns reasoning_content) ──
    # glm-5.2 access OPENED on this account (probed HTTP 200 on 2026-06-18; it had
    # returned 403「您无权访问」through 2026-06-15). It is now the DEFAULT model
    # (DEFAULT_MODEL_ALIAS) and the offered GLM model. glm-5.1 was RETIRED from the
    # presets/pickers in favour of 5.2 — a stored "glm-5.1" alias still resolves
    # (use()'s else-branch → _provider_for_model → zhipu), so old configs don't break.
    "glm-5.2":      ("glm-5.2",      "zhipu",   "智谱 GLM-5.2 — 长程任务/推理",                    16384),
}

# ── Vision (image input) capability — SINGLE SOURCE OF TRUTH ──────────
#
# Same family as ``model_thinking_mode()`` below, and for the same reason: one
# table both the request builder and the UI read, so we never send a content
# block to a model that cannot take one.
#
# This table REPLACES the old ``MINIMAX_VISION_MODELS`` frozenset that used to
# live here. That set had ZERO consumers — a stub for a feature that was never
# wired — and leaving it next to a real table would have made two places to
# answer one question. Its single fact (MiniMax-M3 has vision) is preserved
# below.
#
# ⚠️ MEMBERSHIP IS PER MODEL ID, NEVER PER PREFIX. Moonshot is the proof:
# ``moonshot-v1-128k-vision-preview`` accepts images and ``moonshot-v1-128k``
# does NOT — same prefix, opposite answer. A ``startswith("moonshot")`` rule
# would be wrong for one of them no matter which way it went.
#
# Sources (official vendor docs; see docs/api_providers/):
#   * Moonshot / Kimi — platform.kimi.ai/docs/guide/use-kimi-vision-model,
#     fetched 2026-08-11. Model list quoted verbatim in moonshot_kimi_vision.md.
#   * Anthropic / MiniMax / others — see docs/api_providers/vision_support.md.
#
# DEFAULT IS FALSE. A model id that is not listed here is treated as text-only,
# which degrades to exactly today's behaviour (a text ToolMessage). Guessing
# "probably supports it" is the failure mode this table exists to prevent: the
# cost of a wrong False is a missing picture, the cost of a wrong True is a 400
# on every scan the agent looks at.
_VISION_MODELS: frozenset[str] = frozenset({
    # ── Moonshot / Kimi ── official list, quoted in moonshot_kimi_vision.md §1
    "kimi-k3",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "kimi-k2.7-code-highspeed",
    "moonshot-v1-8k-vision-preview",
    "moonshot-v1-32k-vision-preview",
    "moonshot-v1-128k-vision-preview",
    # NOTE: "moonshot-v1-128k" is deliberately ABSENT — it is a chat-only model
    # and is NOT in Moonshot's vision list. It is also the sole entry in
    # _MODEL_THINKING_OVERRIDE below, for the matching reason.
    # ── Anthropic Claude ── image input on every current model; the native block
    # is {"type":"image","source":{...}}, NOT image_url. We emit the OpenAI
    # image_url shape and langchain_anthropic converts it (pinned by a test in
    # tests/v2/agents/_shared/test_vision_channel.py — it is a third-party
    # behaviour we do not own).
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    # ── MiniMax (Anthropic-SDK-compatible endpoint) ──
    # M3 only. M2.7/M2.5/M2.1/M2 are text+tools with NO image/video — already
    # audited in docs/api_providers/minimax.md, which is where that split is
    # documented; this line is the machine-readable half of it.
    "MiniMax-M3",
    # ── DELIBERATELY ABSENT (checked, not forgotten) — see vision_support.md ──
    #   deepseek-v4-pro : DeepSeek's API documents no image request format.
    #   glm-5.1/glm-5.2 : docs.bigmodel.cn declares 输入模态 = 文本 (text only).
    #   qwen3.7-max     : the exact id is not on DashScope's current model list
    #                     at all, so its modalities cannot be confirmed. Newer
    #                     siblings (qwen3.8-max) DO appear under 图像与视频·理解
    #                     — that is a different id and does not transfer.
})


def model_supports_vision(model_id: str) -> bool:
    """Whether *model_id* accepts image content blocks in a request.

    The single source of truth for image routing, mirroring
    :func:`model_thinking_mode`'s role for thinking params. Unknown ⇒ False, so
    an unrecognised or newly-added model silently stays text-only instead of
    failing a live request.
    """
    return model_id in _VISION_MODELS

# Per-model thinking capability OVERRIDES (exceptions only — the rest is derived
# from the provider in model_thinking_mode()). Values: "none" | "fixed" | "tunable".
_MODEL_THINKING_OVERRIDE: dict[str, str] = {
    "moonshot-v1-128k": "none",   # chat-only, no reasoning — must NOT be sent thinking
}

# Default model alias used by LLMConfig and the GUI dropdown.
# 2026-07-18: switched from glm-5.2 to kimi-k3 (Moonshot Kimi K3, the strongest
# open-weight model, released 2026-07-16) per operator request. LLMConfig.api_key's
# default_factory loads THIS alias's provider key (moonshot / kimi.env), so the
# default model + key stay consistent.
DEFAULT_MODEL_ALIAS = "kimi-k3"

# Extended thinking budget presets (Anthropic-only feature; ignored by other providers).
THINKING_PRESETS: dict[str, int] = {
    "off":    0,
    "low":    4096,
    "medium": 10000,
    "high":   20000,
    "max":    128000,
}

# Claude models that use ADAPTIVE thinking (thinking={type:"adaptive"} +
# output_config={effort}). Opus 4.7/4.8 REQUIRE it — manual
# thinking={type:"enabled", budget_tokens} is rejected with a 400 there. Opus 4.6
# / Sonnet 4.6 recommend it (budget_tokens deprecated). Haiku + older Claude +
# MiniMax do NOT support adaptive → they keep the manual budget_tokens path.
ADAPTIVE_THINKING_MODELS: frozenset[str] = frozenset({
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
})

# Thinking preset name → adaptive `effort` level (None ⇒ no thinking).
THINKING_EFFORT: dict[str, str | None] = {
    "off": None, "low": "low", "medium": "medium", "high": "high", "max": "max",
}

# Maximum OUTPUT tokens each model will accept, by model id. Used to clamp the
# manual-budget thinking path (Haiku / older Claude / MiniMax) so a large thinking
# budget never pushes max_tokens past the model's real ceiling — exceeding it is a
# hard 400 ("max_tokens: N > NNNNN, the maximum allowed for this model"). Haiku 4.5
# caps at 64 000 (NOT 128 000 — that is Opus-tier only); MiniMax M-series at 64 000.
# Models absent here fall back to MODEL_DEFAULT_OUTPUT_LIMIT.
MODEL_OUTPUT_LIMIT: dict[str, int] = {
    # Anthropic Claude
    "claude-haiku-4-5-20251001": 64000,   # Haiku 4.5 hard output ceiling
    "claude-haiku-4-5":          64000,
    "claude-sonnet-4-6":         64000,    # Sonnet 4.6 (adaptive path normally, but cap anyway)
    "claude-opus-4-6":           128000,
    "claude-opus-4-7":           128000,
    "claude-opus-4-8":           128000,
    # MiniMax (Anthropic-compatible) — M3 manual-budget path
    "MiniMax-M3":   64000,
}

# Conservative fallback output ceiling for any model not in MODEL_OUTPUT_LIMIT.
# 64 000 is the smallest current Claude output cap (Haiku/Sonnet 4.6), so it is a
# safe lower bound that won't 400 on an unknown manual-budget model.
MODEL_DEFAULT_OUTPUT_LIMIT: int = 64000


def model_output_limit(model_id: str) -> int:
    """Max output tokens the given model accepts (for clamping the thinking path).

    Single source of truth so the client never bumps ``max_tokens`` above a model's
    real ceiling. Returns the per-model override when known, else a conservative
    64 000 fallback.
    """
    return MODEL_OUTPUT_LIMIT.get(model_id, MODEL_DEFAULT_OUTPUT_LIMIT)


# Maximum INPUT (context-window) tokens each model accepts, by model id. Used by the
# context-compaction middleware to decide WHEN to summarise older turns so a long
# conversation never overflows the window (a hard provider 400
# "context_length_exceeded" / "prompt is too long"). These are deliberately
# CONSERVATIVE under-estimates of each model's real window — a low guess only
# triggers compaction slightly early; it never 400s. Distinct from
# MODEL_OUTPUT_LIMIT (that is the OUTPUT ceiling). Models absent here fall back to
# MODEL_DEFAULT_INPUT_CONTEXT.
MODEL_INPUT_CONTEXT: dict[str, int] = {
    # Moonshot / Kimi — K2.x ~256k real; K3 ~1M real (capped well under for cost/latency)
    "kimi-k2.6":       240000,
    "kimi-k2.7-code":  240000,
    "kimi-k3":         250000,
    "moonshot-v1-128k": 120000,
    # DeepSeek V4 — ~128k real
    "deepseek-v4-pro": 120000,
    # Anthropic Claude — 200k real
    "claude-opus-4-6":           190000,
    "claude-opus-4-7":           190000,
    "claude-opus-4-8":           190000,
    "claude-sonnet-4-6":         190000,
    "claude-haiku-4-5-20251001": 190000,
    "claude-haiku-4-5":          190000,
    # Qwen 3.7 Max — ~1M real, capped well under for cost/latency
    "qwen3.7-max":            250000,
    "qwen3.7-max-2026-05-20": 250000,
    # MiniMax M3 — large window; conservative cap (re-check docs/api_providers/)
    "MiniMax-M3":   180000,
    # Zhipu GLM — ~128k/200k depending on variant; conservative
    "glm-5.1":      120000,
    "glm-5.2":      120000,
}

# Conservative fallback input window for any model not in MODEL_INPUT_CONTEXT.
# 120 000 is below every supported provider's real window, so an unknown model
# compacts a bit early rather than risking an overflow 400.
MODEL_DEFAULT_INPUT_CONTEXT: int = 120000


def model_input_context(model_id: str) -> int:
    """Max INPUT (context-window) tokens the given model accepts.

    Single source of truth for the compaction middleware's token budget. Returns
    the per-model conservative value when known, else MODEL_DEFAULT_INPUT_CONTEXT.
    """
    return MODEL_INPUT_CONTEXT.get(model_id, MODEL_DEFAULT_INPUT_CONTEXT)


def _provider_for_model(model_id: str) -> str:
    """Infer provider from a full model id. Defaults to 'anthropic'."""
    for _, (mid, prov, _, _) in MODEL_PRESETS.items():
        if mid == model_id:
            return prov
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
        return "qwen"
    return "anthropic"


def model_thinking_mode(model_id: str) -> str:
    """How a model exposes extended thinking — the single source of truth used by
    BOTH the chat client and the Settings UI so unsupported params are never sent
    and the UI only offers knobs that actually do something.

    Returns one of:
      "none"    — the model has NO thinking; thinking params must NEVER be sent
                  (e.g. moonshot-v1-128k → sending it 400s / is silently wrong).
      "fixed"   — reasoning is intrinsic and always-on; the provider's thinking
                  kwargs are sent unconditionally and the UI shows it as 固定
                  (Kimi / DeepSeek / Qwen — turning it "off" is not a real mode).
      "tunable" — Anthropic-style budget; the operator's off/low/.../max actually
                  changes behaviour (Anthropic Claude + MiniMax).
    """
    if model_id in _MODEL_THINKING_OVERRIDE:
        return _MODEL_THINKING_OVERRIDE[model_id]
    prov = _provider_for_model(model_id)
    if prov in ANTHROPIC_COMPAT_PROVIDERS:   # anthropic + minimax
        return "tunable"
    if prov in ("moonshot", "deepseek", "qwen", "zhipu"):
        return "fixed"
    return "none"


class LLMConfig(BaseModel):
    """Multi-provider chat-model settings.

    Default provider follows DEFAULT_MODEL_ALIAS (Moonshot / Kimi K3 as of
    2026-07-18). Switch providers by calling `use(alias_or_model_id)` —
    `provider` and `api_key` are refreshed automatically. Anthropic-style
    extended-thinking only takes effect when `provider == "anthropic"`; other
    providers ignore the budget silently.
    """
    # Load the key for the DEFAULT model's provider (via default_factory, NOT a
    # hardcoded provider) so the default config always gets its OWN provider's key:
    # the default Kimi K3 config gets the moonshot key; a GLM-5.2 config would get
    # the zhipu key. use() refreshes it on any later switch.
    api_key: str = Field(
        default_factory=lambda: _load_provider_key(MODEL_PRESETS[DEFAULT_MODEL_ALIAS][1])
    )
    model: str = MODEL_PRESETS[DEFAULT_MODEL_ALIAS][0]
    provider: str = MODEL_PRESETS[DEFAULT_MODEL_ALIAS][1]
    max_tokens: int = MODEL_PRESETS[DEFAULT_MODEL_ALIAS][3]
    # Default to MAX thinking budget (128k tokens) — only takes effect on
    # Anthropic. For Kimi/DeepSeek the provider-specific thinking kwargs
    # are sent unconditionally by ClaudeClient._thinking_body_kwargs().
    thinking_budget: int = THINKING_PRESETS["max"]

    @property
    def base_url(self) -> str:
        """Endpoint base URL for the current provider.

        - Anthropic-compatible providers (minimax) → their /anthropic base; plain
          "anthropic" → "" (the SDK default api.anthropic.com).
        - OpenAI-compatible providers (moonshot/deepseek/qwen) → their /v1 base.
        """
        if self.provider in ANTHROPIC_COMPAT_PROVIDERS:
            return ANTHROPIC_COMPAT_BASE_URL.get(self.provider, "")
        return PROVIDER_BASE_URL.get(self.provider, "")

    def use(self, model_name: str) -> None:
        """Switch model by preset alias or full model ID. Updates provider + key."""
        if model_name in MODEL_PRESETS:
            model_id, provider, _, default_tokens = MODEL_PRESETS[model_name]
            self.model = model_id
            self.provider = provider
            self.max_tokens = default_tokens
        else:
            self.model = model_name
            self.provider = _provider_for_model(model_name)
        # Refresh key for the new provider (env var > provider key file). Always
        # REPLACE the key with the new provider's own — even when the lookup comes
        # back empty. The old `if new_key:` guard kept the previous provider's key
        # whenever the new provider had none, which then went out as an Authorization
        # Bearer / x-api-key header to a DIFFERENT vendor's endpoint (leaking a Kimi
        # key to DeepSeek, etc., and 401-ing in a confusing way). A blank key now
        # fails cleanly at the new endpoint instead of sending the wrong credential.
        self.api_key = _load_provider_key(self.provider)

    def set_thinking(self, level: str | int) -> None:
        """Set extended thinking budget (Anthropic-only).

        Args:
            level: Preset name ('off','low','medium','high','max') or int budget_tokens.
        """
        if isinstance(level, str):
            self.thinking_budget = THINKING_PRESETS.get(level, 0)
        else:
            self.thinking_budget = max(0, int(level))

    @property
    def thinking_enabled(self) -> bool:
        # Tunable Anthropic-style budget applies to Anthropic AND MiniMax
        # (both speak the messages API with budget_tokens). Other providers'
        # thinking is handled separately by ClaudeClient._thinking_body_kwargs.
        return self.thinking_budget > 0 and self.provider in ANTHROPIC_COMPAT_PROVIDERS

    @property
    def thinking_level(self) -> str:
        """Return preset name for current thinking budget."""
        for name, budget in THINKING_PRESETS.items():
            if self.thinking_budget == budget:
                return name
        return f"{self.thinking_budget} tokens"

    @property
    def model_alias(self) -> str:
        """Return short alias for the current model, or the full ID if no alias."""
        for alias, (model_id, _, _, _) in MODEL_PRESETS.items():
            if self.model == model_id:
                return alias
        return self.model

    @property
    def model_description(self) -> str:
        """Human-readable description of the current model."""
        for _, (model_id, _, desc, _) in MODEL_PRESETS.items():
            if self.model == model_id:
                return desc
        return self.model


class VoiceConfig(BaseModel):
    """Voice interaction settings (streaming ASR + TTS via Aliyun DashScope).

    Two provider tiers: STREAMING realtime (``*-realtime`` over the /ws voice
    channel — the primary path) with graceful fallback to the BATCH client
    (``qwen3-*-flash`` REST) wherever realtime is unavailable (e.g. an account
    without realtime entitlement — see docs/api_providers/dashscope_realtime_voice.md)."""
    enabled: bool = True
    api_key: str = Field(default_factory=lambda: _load_provider_key("dashscope"))
    # batch tier (fallback + legacy /api/voice REST)
    tts_model: str = "qwen3-tts-flash"
    asr_model: str = "qwen3-asr-flash"
    # streaming (realtime) tier — the /ws/voice primary path
    streaming: bool = True
    tts_model_rt: str = "qwen3-tts-flash-realtime"
    asr_model_rt: str = "qwen3-asr-flash-realtime"
    default_voice: str = "Cherry"      # Cherry / Ethan / Chelsie / Serena / Dylan
    default_mode: str = "ptt"          # ptt | wake | duplex
    narrate_execution: bool = True     # speak agent tool executions ("扫描完成")
    autoplay: bool = True              # auto-play assistant TTS in chat
    cache_dir_name: str = "tts_cache"  # under experiments_dir


class ServerConfig(BaseModel):
    """GUI server / remote access settings."""
    server_name: str = "127.0.0.1"       # "0.0.0.0" for LAN access
    server_port: int = 7860
    share: bool = False                   # Gradio public tunnel
    auth_username: str = ""               # empty = no auth
    auth_password: str = ""
    ssl_certfile: str = ""                # path to cert PEM (optional)
    ssl_keyfile: str = ""                 # path to key PEM (optional)
    auth_message: str = "MAST - Modular Autonomous SPM Toolkit"
    api_key: str = ""                     # API key for FastAPI REST server


class KnowledgeConfig(BaseModel):
    """Knowledge retrieval settings.

    2026-08-24：``default_mode`` / ``auto_detect_mode`` / ``knowledge_budget`` /
    ``skill_budget`` 四个字段删除。它们是 v1 三档**注入式**知识（simple /
    normal / expert，`knowledge/assembler.py`）的参数，而 v2 的裁决是知识走
    **拉取式**：agent 用 ``query_knowledge(query, detail_level)`` 按需取。
    assembler 在 v2 是 D-discarded，四个字段全仓零读者。

    留下的三个权重仍然活着 —— 它们是混合检索的打分权重，`knowledge/` 的检索
    路径真的在读。
    """
    tfidf_weight: float = 0.50         # TF-IDF cosine weight in hybrid score
    tag_weight: float = 0.25           # tag overlap weight
    alias_weight: float = 0.25         # alias match weight


class UpdateConfig(BaseModel):
    """Intranet push-update settings.

    The launcher writes these via the «自动更新» / «推送服务器» panels.
    Persistence path: env vars override at process start; the *_token.env
    files under api key/ store the shared secret.
    """
    # Client side: poll a remote push server
    enabled: bool = False                # client poll on/off
    server_url: str = ""                 # e.g. https://<push-server>:8766
    check_interval_min: int = 30
    # Server side: bind config for `MAST.exe --push-server-mode`
    server_host: str = "0.0.0.0"
    # MAST uses 8766 to coexist with v1 on the same admin host (v1 → 8765)
    server_port: int = 8766


class ModelConfig(BaseModel):
    """Trained ML model paths and deployment settings."""
    models_dir: Path = Path("models")
    tip_classifier_path: str = ""       # Path to metadata.json for tip classifier
    segmenter_path: str = ""            # Path to metadata.json for segmenter
    atom_detector_path: str = ""        # Path to metadata.json for atom detector
    dqn_policy_path: str = ""           # Path to metadata.json for DQN policy
    auto_discover: bool = True          # Auto-discover models in models_dir
    confidence_threshold: float = 0.7   # Below this, upgrade safety to CONFIRM


class BufferConfig(BaseModel):
    """BufferService settings (Phase 2, v2 agent layer)."""
    # default_factory → resolved lazily at construction, not import .
    wal_path: Path = Field(
        default_factory=lambda: _project_root() / "MASTv2" / "artifacts" / "buffer.wal.sqlite")
    history_size: int = 100
    queue_max_size: int = 1024
    soak_seconds: float = 30.0


class Paths(BaseModel):
    """Project-shared paths (v2 agent layer)."""
    # All derived from the lazily-resolved project root  so a config
    # built after MAST2_PROJECT_ROOT changes uses the current root.
    project_root: Path = Field(default_factory=_project_root)
    artifacts_dir: Path = Field(
        default_factory=lambda: _project_root() / "MASTv2" / "artifacts")
    legacy_artifacts: Path = Field(
        default_factory=lambda: _project_root() / "artifacts" / "legacy")
    overrides_dir: Path = Field(
        default_factory=lambda: _project_root() / "config" / "overrides")
    experiments_db: Path = Field(
        default_factory=lambda: _project_root() / "experiments" / "mast_experiments.db")


class MASTConfig(BaseModel):
    """Top-level MAST configuration."""
    nanonis: NanonisConfig = Field(default_factory=NanonisConfig)
    safety: SafetyLimits = Field(default_factory=SafetyLimits)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    update: UpdateConfig = Field(default_factory=UpdateConfig)
    # FROZEN-SAFE: these were bare relative paths, so in the installed app they
    # resolved against the process CWD — and the --service-mode instance runs
    # with CWD = C:\Windows\System32, where "experiments/…" is not writable →
    # ExperimentStorage raised → the agent got "实验记录不可用" and the records /
    # vision panels were empty (2026-06-29). _project_root() is the exe dir when
    # frozen (consistent regardless of CWD) and the repo root in dev (unchanged).
    experiments_dir: Path = Field(default_factory=lambda: _project_root() / "experiments")
    session_dir: Path = Path("")
    db_path: Path = Field(
        default_factory=lambda: _project_root() / "experiments" / "mast_experiments.db")
