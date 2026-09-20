"""ClaudeClient — multi-provider chat-model wrapper for MAST.

Despite the legacy name, this client speaks to six providers:
  * `anthropic`  — Claude Opus / Sonnet / Haiku via the official SDK.
  * `minimax`    — MiniMax M-series via its Anthropic-compatible endpoint.
  * `moonshot`   — Kimi K2.x via Moonshot's OpenAI-compatible /v1 endpoint.
  * `deepseek`   — DeepSeek V4 via DeepSeek's OpenAI-compatible endpoint.
  * `qwen`       — Qwen via DashScope's OpenAI-compatible endpoint.
  * `zhipu`      — GLM via its OpenAI-compatible endpoint.

The public `chat(messages, system, tools)` signature and the dict it returns
follow Anthropic's shape so all v1 callers (planner.py, interpreter.py,
skill_author.py) stay unchanged. When `provider != "anthropic"` the client
translates Anthropic-style messages / tools to OpenAI format on the way out
and translates the OpenAI response back to Anthropic shape on the way in.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

import httpx
from anthropic import Anthropic

from mast.config import (
    ADAPTIVE_THINKING_MODELS,
    ANTHROPIC_COMPAT_PROVIDERS,
    LLMConfig,
    PROVIDER_BASE_URL,
    THINKING_EFFORT,
    model_output_limit,
    model_thinking_mode,
)

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Multi-provider chat client with an Anthropic-shaped public API."""

    # OpenAI-compat retry policy. A /chat/completions POST is idempotent (the
    # server runs a fresh completion each time and we never send an idempotency
    # key that would dedup), so retrying a transient failure is safe. We retry
    # ONLY on 429 (rate limit), 5xx (server / overloaded), and transient network
    # errors (connect/read timeouts, dropped connections) — never on a 4xx like
    # 400/401/404, which is a deterministic client error that a retry can't fix.
    _OPENAI_MAX_RETRIES = 3        # total attempts = 1 + 3 retries
    _OPENAI_RETRY_BASE_DELAY = 1.0  # seconds; exponential: 1, 2, 4 (+ jitter)
    _OPENAI_RETRY_MAX_DELAY = 20.0  # cap a single backoff sleep

    def __init__(self, config: LLMConfig):
        # Own a PRIVATE copy of the config. `use()` / `set_thinking()` mutate the
        # config in place; if two clients (e.g. the main-chat planner and the QA
        # assistant) shared the SAME LLMConfig instance, switching the QA model would
        # rewrite the shared object's model/provider/api_key/thinking — silently
        # repointing main chat and, because `thinking_enabled` is gated on provider,
        # turning main-chat extended thinking OFF whenever QA picked a non-Anthropic
        # model. A deep copy gives each client an independent config so a runtime
        # switch on one never bleeds into the other. (LLMConfig is a pydantic model;
        # model_copy(deep=True) is the supported clone, with a plain-copy fallback.)
        self._config = _clone_config(config)
        self._model = self._config.model
        self._max_tokens = self._config.max_tokens
        self._provider = self._config.provider
        # Label for the 用量·花销 ledger (the direct-client path — QuickAsk /
        # summarizer / standalone — as opposed to the agents' factory models,
        # which the make_chat_model callback books). Callers may override.
        self._bill_source: str = "chat"
        self._anth: Anthropic | None = None
        self._http: httpx.Client | None = None
        self._build_backend()

    # ── Backend lifecycle ────────────────────────────────────────────

    def _build_backend(self) -> None:
        """(Re)create whichever client matches the current provider."""
        if self._provider in ANTHROPIC_COMPAT_PROVIDERS:
            # Anthropic SDK path. base_url "" → SDK default (api.anthropic.com)
            # for plain "anthropic"; MiniMax → its /anthropic-compatible base.
            self._anth = Anthropic(api_key=self._config.api_key,
                                   base_url=self._config.base_url or None)
            if self._http is not None:
                try:
                    self._http.close()
                except Exception:
                    pass
                self._http = None
        else:
            base = PROVIDER_BASE_URL.get(self._provider, "")
            if not base:
                raise ValueError(f"Unknown provider: {self._provider!r}")
            self._http = httpx.Client(
                base_url=base,
                # Read timeout = 300s (5 min). Kimi K2.6 / DeepSeek with max
                # thinking + 238 tools normally answer in <120 s; 5 min is
                # generous and still finite. UNBOUNDED timeout (`None`) was
                # the v0.3.14 default and made the chat permanently hang
                # whenever Moonshot accepted a request but didn't reply (a
                # real failure mode we saw in service-20260512.log step 2:
                # POST sent at 11:03:32, no response, log frozen until the
                # user force-restarted the service). With a finite timeout
                # we get a clean ReadTimeout that surfaces to the chat,
                # the user can retry, and the Stop button works between
                # round-trips. Connect timeout stays small to fail fast on
                # DNS / wrong URL.
                timeout=httpx.Timeout(300.0, connect=15.0),
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
            )
            self._anth = None

    # ── Runtime switches ─────────────────────────────────────────────

    def use(self, model_name: str) -> None:
        """Switch model at runtime. Accepts alias or full ID."""
        old_provider = self._provider
        self._config.use(model_name)
        self._model = self._config.model
        self._max_tokens = self._config.max_tokens
        self._provider = self._config.provider
        if self._provider != old_provider:
            self._build_backend()
        elif self._provider in ANTHROPIC_COMPAT_PROVIDERS:
            self._anth = Anthropic(api_key=self._config.api_key,
                                   base_url=self._config.base_url or None)
        else:
            assert self._http is not None
            self._http.headers["Authorization"] = f"Bearer {self._config.api_key}"
        logger.info(
            "Switched to model: %s (%s, provider=%s)",
            self._config.model_alias, self._model, self._provider,
        )

    def set_thinking(self, level: str | int) -> None:
        """Set extended thinking budget (Anthropic-only; silently ignored elsewhere)."""
        self._config.set_thinking(level)
        logger.info("Thinking set to: %s", self._config.thinking_level)

    def _thinking_body_kwargs(self) -> dict:
        """Provider-specific kwargs for max-effort thinking mode.

        Each provider has its own way of expressing "give me your best
        chain-of-thought reasoning":

        - DeepSeek V4 Pro / Flash:
          ``thinking={"type":"enabled"}, reasoning_effort="max"``
          docs: https://api-docs.deepseek.com/guides/thinking_mode
          (note: SDK puts ``thinking`` in extra_body; over raw HTTP it goes
          to the top of the body — same wire format)

        - Moonshot Kimi K2.6 / K2.5 / K2-thinking:
          ``thinking={"type":"enabled","keep":"all"}``
          docs: https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model.md
          (Kimi does NOT accept reasoning_effort — sending it 400s.)

        - Anthropic:
          uses its own ``thinking={budget_tokens=N}`` kwarg in messages.create,
          handled separately in _chat_anthropic.

        We always send these kwargs by default (rather than relying on the
        provider's "thinking on by default" behavior) so the wire intent is
        explicit and visible in service logs.
        """
        if self._provider == "deepseek":
            return {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            }
        if self._provider == "moonshot":
            return {
                "thinking": {"type": "enabled", "keep": "all"},
            }
        if self._provider == "qwen":
            # Qwen3.7-Max via DashScope's compatible-mode endpoint:
            #   - ``enable_thinking=True``  ON (the documented switch — without it
            #     reasoning_content is empty; suppresses sleepwalking, improves
            #     tool-selection accuracy). NOT sending ``thinking_budget`` =
            #     unlimited thinking tokens = MAX thinking.
            #   - ``parallel_tool_calls=False`` because most STM steps are
            #     sequential (approach → scan → analyze).
            #   - ``tool_choice="auto"`` (forced when thinking is on — Qwen
            #     rejects a fixed function name in this mode).
            # NOTE: ``preserve_thinking`` was REMOVED — it is NOT a real DashScope
            # parameter (absent from the deep-thinking docs); CoT round-trip is
            # handled by re-sending reasoning_content in the message history
            # (_anthropic_messages_to_openai), not by a request flag.
            return {
                "enable_thinking": True,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
            }
        if self._provider == "zhipu":
            # GLM-5.1 (OpenAI-compat): enable deep thinking. It reasons by
            # default; we send it explicitly for MAX thinking. Returns CoT in
            # `reasoning_content` (round-tripped via the message history).
            return {"thinking": {"type": "enabled"}}
        return {}

    @property
    def current_model(self) -> str:
        return self._config.model_alias

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def thinking_level(self) -> str:
        return self._config.thinking_level

    # ── Public chat API (Anthropic-shaped) ───────────────────────────

    def chat(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
    ) -> dict:
        """Send Anthropic-shaped *messages* to the configured provider.

        Returns a dict with the same shape as Anthropic responses:
            {"id", "role", "content"=[blocks], "model", "stop_reason", "usage"}
        regardless of the underlying provider.
        """
        if self._provider in ANTHROPIC_COMPAT_PROVIDERS:
            result = self._chat_anthropic(messages, system, tools)
        else:
            result = self._chat_openai_compat(messages, system, tools)
        self._record_usage(result)
        return result

    def _record_usage(self, result: dict) -> None:
        """Book this call's token cost into the 用量·花销 ledger (fail-safe)."""
        try:
            usage = (result or {}).get("usage") or {}
            in_tok = int(usage.get("input_tokens") or 0)
            out_tok = int(usage.get("output_tokens") or 0)
            if not in_tok and not out_tok:
                return
            from mast.billing.capture import record_llm
            record_llm(model=(result.get("model") or self._model),
                       input_tokens=in_tok, output_tokens=out_tok,
                       source=self._bill_source, provider=self._provider)
        except Exception:  # noqa: BLE001 — billing must never break chat
            logger.debug("ClaudeClient usage record failed (swallowed)", exc_info=True)

    def single_turn(self, prompt: str, system: str = "") -> str:
        """Simple single-turn text response.

        Returns the assistant's visible text. If the model replies with ONLY
        reasoning (a `thinking` block and no `text` block) — which Qwen and
        Kimi/DeepSeek thinking models do for short prompts where the final
        answer is short enough to live inside the chain-of-thought — we fall
        back to the thinking text instead of returning an empty string. The
        previous version silently dropped the entire reply in that case.
        """
        response = self.chat(
            messages=[{"role": "user", "content": prompt}],
            system=system,
        )
        parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response["content"]:
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking_parts.append(block.get("thinking", "") or "")
        if not parts and thinking_parts:
            return "\n".join(p for p in thinking_parts if p)
        return "\n".join(parts)

    # ── Anthropic backend ────────────────────────────────────────────

    def _chat_anthropic(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None,
    ) -> dict:
        assert self._anth is not None
        thinking_on = self._config.thinking_enabled

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }

        if thinking_on and self._model in ADAPTIVE_THINKING_MODELS:
            # Opus 4.6/4.7/4.8 + Sonnet 4.6: ADAPTIVE thinking. Manual
            # budget_tokens is rejected (Opus 4.7/4.8) / deprecated (4.6/Sonnet
            # 4.6). effort = the chosen level (max → "always think, no limit");
            # display:"summarized" so we still receive readable thinking text
            # (Opus 4.7/4.8 default to "omitted"). Adaptive also auto-enables
            # interleaved thinking (better for tool loops). max_tokens is the
            # hard cap on thinking+text — give it room.
            effort = THINKING_EFFORT.get(self._config.thinking_level) or "high"
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            kwargs["output_config"] = {"effort": effort}
            if kwargs["max_tokens"] < 16000:
                kwargs["max_tokens"] = 16000
        elif thinking_on:
            # Haiku / older Claude / MiniMax: manual budget_tokens (no adaptive).
            # max_output is the MODEL'S real output ceiling — NOT a hardcoded 128000.
            # Haiku 4.5 caps at 64 000 (128 000 is Opus-tier only); MiniMax M-series at
            # 64 000. The old `max_output = 128000` meant a max-effort budget (128k)
            # bumped max_tokens to 128 000 and Haiku 400'd ("max_tokens > 64000")
            # every time — thinking=max was simply unusable on Haiku. Clamp to the
            # per-model limit so the bump stays inside what the model accepts.
            budget = self._config.thinking_budget
            max_output = model_output_limit(self._model)
            if budget >= self._max_tokens:
                kwargs["max_tokens"] = min(budget + 8192, max_output)
            # Never let max_tokens exceed the model ceiling, even if the configured
            # self._max_tokens already did (defensive — preset tokens are < ceiling).
            if kwargs.get("max_tokens", self._max_tokens) > max_output:
                kwargs["max_tokens"] = max_output
            effective_max = kwargs.get("max_tokens", self._max_tokens)
            # budget_tokens MUST be strictly < max_tokens (Anthropic contract).
            if budget >= effective_max:
                budget = effective_max - 1024
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        try:
            if kwargs.get("max_tokens", 0) > 16384:
                with self._anth.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
            else:
                response = self._anth.messages.create(**kwargs)
        except Exception as e:
            logger.error("Anthropic API error: %s", e)
            raise

        content_blocks = [_anthropic_block_to_dict(b) for b in response.content]
        # F3 (2026-06-08): MiniMax intermittently leaks <think>/<mm:think> CoT
        # into visible TEXT blocks instead of a proper thinking block. Strip the
        # tag dialect at the source so the raw reasoning + literal tags never
        # reach the chat renderer. MiniMax-only; no-op for clean text / real Claude.
        if self._provider == "minimax":
            for b in content_blocks:
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                    b["text"] = _strip_think_tags(b["text"])
        return {
            "id": response.id,
            "role": response.role,
            "content": content_blocks,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    # ── OpenAI-compatible backend (Moonshot / DeepSeek) ──────────────

    def _chat_openai_compat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None,
    ) -> dict:
        assert self._http is not None
        oai_messages = _anthropic_messages_to_openai(messages, system)
        body: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": self._max_tokens,
        }
        # Thinking kwargs are gated on the MODEL's capability — NOT sent blindly
        # per-provider. A "none" model (e.g. moonshot-v1-128k, chat-only) must
        # never receive thinking params (they 400 / are silently wrong); "fixed"
        # reasoning models (Kimi/DeepSeek/Qwen) get them unconditionally as
        # before. This closes the "wrong params uploaded to some models" bug.
        # (DeepSeek forbids temperature/top_p/frequency_penalty with thinking on,
        # so we omit those entirely — they are never added to the OpenAI body.)
        tool_choice = None
        if model_thinking_mode(self._model) != "none":
            thinking_kwargs = self._thinking_body_kwargs()
            # `tool_choice` is a tools-only parameter. Qwen's thinking kwargs
            # carry tool_choice="auto" (forced when enable_thinking is on), but
            # DashScope rejects tool_choice with HTTP 400 when no `tools` array
            # is present. Strip it here and re-add below only when we send tools.
            tool_choice = thinking_kwargs.pop("tool_choice", None)
            body.update(thinking_kwargs)
        if tools:
            body["tools"] = [_anthropic_tool_to_openai(t) for t in tools]
            body["tool_choice"] = tool_choice or "auto"

        resp = self._post_chat_completions_with_retry(body)
        data = resp.json()
        return _openai_response_to_anthropic(data, self._model)

    def _post_chat_completions_with_retry(self, body: dict) -> httpx.Response:
        """POST /chat/completions with bounded exponential-backoff retry.

        Retries 429 / 5xx / transient network errors (idempotent request); raises
        immediately on any other 4xx and re-raises the last error after exhausting
        retries. Honours a ``Retry-After`` header when the server sends one.
        """
        assert self._http is not None
        last_exc: Exception | None = None
        for attempt in range(self._OPENAI_MAX_RETRIES + 1):
            try:
                resp = self._http.post("/chat/completions", json=body)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as e:
                last_exc = e
                status = e.response.status_code if e.response is not None else 0
                detail = e.response.text if e.response is not None else ""
                retryable = status == 429 or status >= 500
                if not retryable or attempt >= self._OPENAI_MAX_RETRIES:
                    # Surface the API's actual error body — invaluable for debugging
                    # 400s where the status code alone is useless. Logged at ERROR so
                    # it lands in service-YYYYMMDD.log next to the rest of the trace.
                    logger.error(
                        "%s API HTTP %s on %s: %s",
                        self._provider, status, self._model, detail[:1500],
                    )
                    raise
                delay = self._retry_delay(attempt, e.response)
                logger.warning(
                    "%s API HTTP %s on %s (attempt %d/%d) — retrying in %.1fs: %s",
                    self._provider, status, self._model, attempt + 1,
                    self._OPENAI_MAX_RETRIES, delay, detail[:300],
                )
                time.sleep(delay)
            except (httpx.TransportError, httpx.HTTPError) as e:
                # Transient network errors: connect/read timeouts, dropped
                # connections, protocol errors. httpx.TransportError covers
                # ConnectError/ReadTimeout/etc.; HTTPError is the catch-all base
                # for anything else httpx raises (still safe to retry the
                # idempotent POST). Non-httpx exceptions fall through and raise.
                last_exc = e
                if attempt >= self._OPENAI_MAX_RETRIES:
                    logger.error("%s API network error on %s: %s",
                                 self._provider, self._model, e)
                    raise
                delay = self._retry_delay(attempt, None)
                logger.warning(
                    "%s API network error on %s (attempt %d/%d) — retrying in "
                    "%.1fs: %s",
                    self._provider, self._model, attempt + 1,
                    self._OPENAI_MAX_RETRIES, delay, e,
                )
                time.sleep(delay)
        # Unreachable in practice (the loop either returns or raises), but keep a
        # definite raise so the type checker / a logic slip never returns None.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("retry loop exited without a response")

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Backoff seconds for *attempt* (0-indexed). Honours Retry-After when set."""
        if response is not None:
            ra = response.headers.get("retry-after")
            if ra:
                try:
                    # Retry-After is whole seconds here (providers send a number);
                    # cap it so a hostile/huge value can't wedge the chat.
                    return min(float(ra), self._OPENAI_RETRY_MAX_DELAY)
                except (TypeError, ValueError):
                    pass
        backoff = self._OPENAI_RETRY_BASE_DELAY * (2 ** attempt)
        return min(backoff, self._OPENAI_RETRY_MAX_DELAY) + random.uniform(0, 0.5)


def _clone_config(config: LLMConfig) -> LLMConfig:
    """Return an independent copy of *config* so a client's runtime `use()` /
    `set_thinking()` never mutates a config object shared with another client.

    Uses pydantic's deep `model_copy` when available; falls back to a plain
    re-construction from the model's field values for any non-pydantic shim.
    """
    try:
        return config.model_copy(deep=True)
    except Exception:
        try:
            return type(config)(**config.model_dump())
        except Exception:
            # Last resort: hand back the original (preserves old behaviour rather
            # than crashing). Should never hit for a real pydantic LLMConfig.
            return config


# ── Translation helpers ──────────────────────────────────────────────

# Matches a PAIRED think block only: <think …>…</think> or <mm:think>…</mm:think>
# (the \b after `think` avoids eating <thinker>/<thinking-foo> custom elements).
_THINK_BLOCK_RE = re.compile(
    r"<(?:mm:)?think\b[^>]*>.*?</(?:mm:)?think\s*>", re.DOTALL | re.IGNORECASE
)


def _strip_think_tags(text: str) -> str:
    """Strip leaked PAIRED ``<think>…</think>`` / ``<mm:think>…</mm:think>``
    chain-of-thought blocks from a user-visible text block (审查 F3,
    hardened after adversarial review).

    Some MiniMax responses intermittently put reasoning in a *text* block wrapped
    in think tags instead of a proper ``thinking`` block; the raw CoT then leaks
    into the chat answer. This removes ONLY complete paired blocks, and ONLY when
    such a block actually exists — so:
      * legitimate prose / code containing a literal ``<think>`` token, a custom
        ``<thinker>`` element, or an unpaired tag is LEFT UNTOUCHED (no corruption);
      * a block that empties the text (the whole answer was a leaked block)
        correctly returns "" — the caller drops empty text (it does NOT fall back
        to the raw text, which would re-leak the very CoT this strips).
    Never edits the XSS-escape path. No-op for clean text (real Claude never emits
    these tags) and a fast bail-out when no think tag is present.
    """
    if not text or "think" not in text.lower():
        return text
    if not _THINK_BLOCK_RE.search(text):
        # No COMPLETE paired block → don't touch it (orphan tags / literal
        # mentions are almost always legitimate answer text, not a CoT leak).
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


def _anthropic_block_to_dict(block: Any) -> dict:
    if hasattr(block, "type"):
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "thinking":
            return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
        if block.type == "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, dict):
        return block
    return {"type": "unknown", "raw": str(block)}


def _anthropic_tool_to_openai(tool: dict) -> dict:
    """Convert Anthropic tool definition to OpenAI function-tool definition."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _anthropic_content_to_openai_text(content: Any) -> str:
    """Flatten Anthropic content (str or block list) into a plain text string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            t = block.get("type")
            if t == "text":
                parts.append(block.get("text", ""))
            elif t == "tool_use":
                pass  # handled separately as tool_calls
            elif t == "tool_result":
                inner = block.get("content")
                parts.append(_anthropic_content_to_openai_text(inner) if inner is not None else "")
            elif t == "thinking":
                # Reasoning content is round-tripped via the OpenAI message's
                # `reasoning_content` field (set in _anthropic_messages_to_openai
                # for assistant blocks). Do NOT include str(block) in plain
                # text — that polluted Kimi/DeepSeek replies with literal
                # "{'type': 'thinking', 'thinking': '...', 'signature': ''}"
                # text in v0.3.5 and earlier.
                pass
            else:
                # Unknown block type — fall back to repr but warn loudly so
                # we catch new types early instead of leaking them silently.
                parts.append(str(block))
        else:
            parts.append(str(block))
    return "".join(parts)


def _anthropic_messages_to_openai(messages: list[dict], system: str) -> list[dict]:
    """Translate Anthropic-shaped message list (with system arg) into OpenAI messages."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            if isinstance(content, list):
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if tool_results:
                    for tr in tool_results:
                        text = tr.get("content")
                        text_str = _anthropic_content_to_openai_text(text) if not isinstance(text, str) else text
                        out.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": text_str or "",
                        })
                    # Append any non-tool_result text as a follow-up user msg.
                    text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    if text_blocks:
                        out.append({
                            "role": "user",
                            "content": "\n".join(b.get("text", "") for b in text_blocks),
                        })
                    continue
            out.append({"role": "user", "content": _anthropic_content_to_openai_text(content)})
        elif role == "assistant":
            text_str = _anthropic_content_to_openai_text(content)
            tool_calls: list[dict] = []
            reasoning_text = ""
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}), default=str),
                            },
                        })
                    elif btype == "thinking":
                        # Captured by _openai_response_to_anthropic from
                        # Kimi/DeepSeek's reasoning_content. Echo it back so
                        # turn N+1 doesn't trip the API's
                        # "thinking is enabled but reasoning_content is
                        # missing in assistant tool call message" guard.
                        reasoning_text = block.get("thinking", "") or ""
            asst_msg: dict[str, Any] = {"role": "assistant", "content": text_str}
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
                # OpenAI spec: content can be empty string (or null) when tool_calls present
                if not text_str:
                    asst_msg["content"] = ""
            if reasoning_text:
                asst_msg["reasoning_content"] = reasoning_text
            out.append(asst_msg)
        elif role == "system":
            out.append({"role": "system", "content": _anthropic_content_to_openai_text(content)})
        else:
            out.append({"role": role, "content": _anthropic_content_to_openai_text(content)})
    return out


def _openai_response_to_anthropic(data: dict, model: str) -> dict:
    """Translate one OpenAI-style chat.completion response into Anthropic shape."""
    choices = data.get("choices") or []
    if not choices:
        return {
            "id": data.get("id", ""), "role": "assistant", "content": [],
            "model": model, "stop_reason": "end_turn",
            "usage": _openai_usage_to_anthropic(data.get("usage")),
        }
    msg = choices[0].get("message", {}) or {}
    finish = choices[0].get("finish_reason", "stop")

    blocks: list[dict] = []
    # Kimi K2.x and DeepSeek V4 (when thinking is enabled) return their CoT
    # in `reasoning_content`. We MUST round-trip it back to the API on the
    # next turn alongside any tool_calls — otherwise the next request fails:
    #   "thinking is enabled but reasoning_content is missing in assistant
    #    tool call message at index N"
    # Capture as an Anthropic-shape "thinking" block so the planner's
    # message-list serialization preserves it.
    reasoning = msg.get("reasoning_content") or ""
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    text = msg.get("content") or ""
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments", "{}") or "{}"
        try:
            parsed_args = json.loads(raw_args)
        except (TypeError, json.JSONDecodeError):
            parsed_args = {"_raw": raw_args}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "input": parsed_args,
        })

    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",
    }

    return {
        "id": data.get("id", ""),
        "role": "assistant",
        "content": blocks,
        "model": data.get("model", model),
        "stop_reason": stop_reason_map.get(finish, "end_turn"),
        "usage": _openai_usage_to_anthropic(data.get("usage")),
    }


def _openai_usage_to_anthropic(usage: dict | None) -> dict:
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


# ── Backwards-compatible alias used by status_panel/html_builders ────
LLMClient = ClaudeClient


