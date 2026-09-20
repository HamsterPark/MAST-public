# Zhipu GLM-5.2 — OpenAI-compatible endpoint

Sources: https://docs.bigmodel.cn/cn/guide/models/text/glm-5.2 ,
         https://open.bigmodel.cn/api/paas/v4  (verified live 2026-06-01)

> 2026-06-15: GLM-5.2 supersedes GLM-5.1 in MAST (`glm-5.1` removed from the
> registry). Same OpenAI-compatible endpoint + reasoning behaviour; only the
> model id changed. (Live availability depends on the bigmodel.cn account having
> balance — a 1113 "余额不足" reply is a billing state, NOT model-not-found.)

## Endpoint
- Base: **https://open.bigmodel.cn/api/paas/v4** (OpenAI-compatible; `/chat/completions`).
- Auth: `Authorization: Bearer <key>`. Key format `xxxx.yyyy`.
- Request/response/streaming/tool-calling format = OpenAI chat completions.

## Model
- `glm-5.2` — long-horizon tasks + reasoning (current MAST GLM model).

## Thinking
- `thinking={"type":"enabled"}` enables deep thinking (also reasons by DEFAULT — a
  request with NO thinking param still produced reasoning_content in our live test).
- Returns the chain-of-thought in **`reasoning_content`** (sibling of `content`).
- ⚠️ Like Kimi, a small `max_tokens` truncates: with max_tokens=64 the reasoning
  consumed all tokens and `content` came back EMPTY (finish_reason="length"). MAST
  presets glm-5.2 at 16384 and the agents path floors reasoning models to 16000.

## MULTI-TURN
- reasoning_content round-trip handled the same as Kimi/DeepSeek: MAST captures it
  (`_openai_response_to_anthropic` → thinking block) and re-sends it
  (`_anthropic_messages_to_openai` → reasoning_content) / agents use
  `ReasoningPreservingChatOpenAI`. Treated as a REASONING_MODEL + _ALWAYS_HIGH_REASONING.

## MAST mapping
- provider `zhipu`; classified `model_thinking_mode="fixed"`; `_thinking_body_kwargs`
  sends `{"thinking":{"type":"enabled"}}`. Key: env `ZHIPU_API_KEY`/`GLM_API_KEY` or
  `api key/glm.env` (gitignored). Selectable in main chat / QA / all agents.
