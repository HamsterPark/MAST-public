# DeepSeek Reasoning Model (deepseek-reasoner) — API params

Source: https://api-docs.deepseek.com/guides/reasoning_model (fetched 2026-06-01)

## Supported
- `max_tokens`: max output length INCLUDING the CoT. Default 32K, max 64K.

## NOT supported
- `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` — accepted but
  **NO effect** (silently ignored, no error).
- `logprobs`, `top_logprobs` — **trigger a 400 error**.

## Enabling reasoning
- NO explicit parameter. `deepseek-reasoner` reasons automatically. No
  `reasoning_effort` / `thinking` field described here.

## Output
- `reasoning_content` (CoT, sibling of `content`) + `content` (final answer).

## Multi-turn (CRITICAL — OPPOSITE of v4 thinking_mode)
- If `reasoning_content` is included in the INPUT messages → **400 error**.
  Must strip `reasoning_content` before the next request.

> NOTE for MAST audit: this is `deepseek-reasoner`. Our model is `deepseek-v4-pro`
> for which the code sends `thinking={"type":"enabled"}, reasoning_effort="max"`
> and PRESERVES reasoning_content across turns (reasoning_chat_model). Those are
> the requirements of the V4 *thinking_mode* (hybrid) endpoint, which is a
> DIFFERENT contract from deepseek-reasoner above. Confirm against thinking_mode doc.
