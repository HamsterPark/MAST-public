# Moonshot / Kimi K2.6 — Thinking Model (OpenAI-compatible /v1/chat/completions)

Source: https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model (fetched 2026-06-01)

> 2026-06-15: added `kimi-k2.7-code` (K2.7 code-specialised) — treated as a Kimi
> thinking model (same as K2.6: `_FORCED_TEMPERATURE_1`, `_ALWAYS_HIGH_REASONING`,
> `REASONING_MODELS`, max_tokens floored to 16000). `kimi-k2.5` was removed. Verify
> the exact id `kimi-k2.7-code` against the Moonshot platform model list before
> relying on it in production (an `engine_overloaded` reply is a capacity state,
> NOT model-not-found).

## Enable thinking
```json
{ "thinking": { "type": "enabled", "keep": "all" } }
```
- Thinking is **enabled by default**; disable with `{"type":"disabled"}`.
- `keep:"all"` only affects whether HISTORICAL-turn `reasoning_content` is preserved;
  it does NOT change current-turn thinking generation.
- NO `reasoning_effort`, NO `chat_template_kwargs` on the OpenAI-compat endpoint.
- MAST sends `thinking={type:enabled, keep:all}`. ✓

## Temperature / max_tokens
- **Temperature is FIXED at 1.0** for kimi-k2.6 (set temperature=1.0). MAST coerces kimi
  to 1.0 via _FORCED_TEMPERATURE_1. ✓
- **Recommended `max_tokens ≥ 16000`** so full reasoning_content + content aren't truncated.
  ⚠️ MAST preset `kimi-k2.6` max_tokens = **8192** → TOO LOW; risks truncating reasoning.

## Response field
- Reasoning is returned in **`reasoning_content`** (access via getattr; SDK type doesn't expose it).

## MULTI-TURN
- With `keep:"all"`: keep `reasoning_content` from EVERY historical assistant message in
  `messages` as-is. Simplest: append the returned assistant message back into `messages`.
- MAST's reasoning-preserving ChatOpenAI subclass round-trips reasoning_content. ✓
