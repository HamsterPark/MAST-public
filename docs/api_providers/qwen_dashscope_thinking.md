# Qwen3.7-Max via DashScope — compatible-mode (OpenAI) thinking

Sources: https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
         https://www.alibabacloud.com/help/en/model-studio/deep-thinking  (fetched 2026-06-01)

## Endpoint
- China: https://dashscope.aliyuncs.com/compatible-mode/v1  (MAST uses this)
- Intl/US variants exist (dashscope-intl / dashscope-us).

## Enable thinking
- **`enable_thinking: true`** required to activate reasoning (else reasoning_content empty/absent).
  Via OpenAI SDK: `extra_body={"enable_thinking": True}`.
- qwen3-max hybrid thinking is **disabled by default**.
- MAST sends `enable_thinking=True`. ✓
- ⚠️ MAST also sends `preserve_thinking=True`, `parallel_tool_calls=False`, `tool_choice="auto"`.
  TODO verify `preserve_thinking` is a real DashScope param (may be silently ignored/non-standard).

## Response field
- Reasoning in `reasoning_content`; answer in `content`.

## MULTI-TURN
- Compatible with OpenAI/Anthropic specs; multi-turn tool calls supported. Confirm whether
  reasoning_content must be round-tripped (Qwen generally does NOT require re-sending CoT, but
  verify for tool-call turns). MAST's reasoning-preserving subclass keeps it (safe).

## Temperature
- Not fixed (unlike Kimi). MAST passes temperature through for qwen.
