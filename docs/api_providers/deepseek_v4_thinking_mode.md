# DeepSeek V4 — Thinking Mode (hybrid)  [our model: deepseek-v4-pro]

Source: https://api-docs.deepseek.com/guides/thinking_mode (fetched 2026-06-01)

## Enable thinking
- OpenAI format: `{"thinking": {"type": "enabled" | "disabled"}}`  ← MAST sends this
- Anthropic format: `{"output_config": {"effort": "high" | "max"}}`
- Thinking toggle **defaults to enabled**.

## Reasoning effort
- Accepted values: **`high` or `max` ONLY**. `low`/`medium` → mapped to `high`; `xhigh` → `max`.
- Default effort = `high` for normal requests; auto `max` for agent requests (Claude Code/OpenCode).
- MAST sends `reasoning_effort="max"` → valid (max effort). ✓

## Sampling params
- Thinking mode does **NOT** support `temperature`, `top_p`, `presence_penalty`,
  `frequency_penalty` — set but **no effect**, no error. (`logprobs`/`top_logprobs` → 400 on
  deepseek-reasoner; see deepseek_reasoning_model.md.)
- MAST omits temperature in _chat_openai_compat for deepseek. ✓ (Agents path passes
  temperature=0.2 to ChatOpenAI → ignored by endpoint, harmless.)

## MULTI-TURN (critical, differs by tool use)
- WITHOUT tool calls: intermediate assistant `reasoning_content` **need not** be re-sent.
- WITH tool calls: intermediate assistant `reasoning_content` **MUST** be kept and
  **passed back** in all subsequent turns. (Else 400 "reasoning_content missing".)
- MAST's reasoning-preserving ChatOpenAI subclass keeps reasoning_content across turns. ✓

## max_tokens
- Not specified on this page (reasoner doc: default 32K, max 64K, includes CoT).
