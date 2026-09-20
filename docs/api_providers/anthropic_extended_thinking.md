# Anthropic Claude — Extended Thinking (Messages API)

Source: https://platform.claude.com/docs/en/build-with-claude/extended-thinking (fetched 2026-06-01)

## Request param
```json
{ "thinking": { "type": "enabled", "budget_tokens": N } }
```
- `budget_tokens` **must be < `max_tokens`** (except interleaved thinking, where it's
  the total across thinking blocks in one turn and may exceed max_tokens).
- No explicit minimum stated (commonly 1024). Above ~32k Claude may not use the full budget.
- Cannot combine with `max_tokens: 0` (cache pre-warm).

## Sampling params
- This page does NOT state temperature/top_p/top_k constraints. (Anthropic's API
  contract elsewhere: when thinking is enabled, temperature must be 1 / unset, and
  top_p/top_k are restricted. MAST's _chat_anthropic sends NO temperature → SDK default.)

## Adaptive thinking (IMPORTANT)
- On Opus 4.6 / Sonnet 4.6, **`type:"adaptive"` adaptive thinking is RECOMMENDED**, and
  **manual `budget_tokens` is DEPRECATED**. MAST currently sends `type:"enabled" + budget_tokens`.

## Multi-turn
- Thinking blocks returned by the model carry a `signature`; when continuing a turn with
  tool_use you must pass the thinking blocks back unmodified (signature preserved). The
  Anthropic SDK handles this when you append the returned content blocks verbatim.

## MAST mapping
- `client._chat_anthropic`: `thinking={type:enabled, budget_tokens}` only when
  `thinking_enabled` (budget>0 & provider in anthropic/minimax); bumps max_tokens if budget>=max.
  Streams when max_tokens>16384. No temperature sent. ✓ mostly correct; adaptive is an upgrade option.
