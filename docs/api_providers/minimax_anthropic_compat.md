# MiniMax M3 / M2.x — Anthropic-SDK-compatible endpoint

Source: MiniMax official param table (from the vendor's parameter table) +
verified live against https://api.minimaxi.com/anthropic/v1/messages (x-api-key, 200).

## Endpoint
- Base: **https://api.minimaxi.com/anthropic**  (SDK appends /v1/messages).
  NOTE: api.minimax.io 401s — use api.minimaxi.com.
- Auth: Anthropic SDK `x-api-key` works.

## Models
- `MiniMax-M3` (text + image + video + tool_use + tool_result + thinking)
- `MiniMax-M2.7 / M2.5 / M2.1 / M2` (text + tools + thinking; NO image/video)
  (+ `-highspeed` variants of M2.7/M2.5/M2.1 exist)

> 2026-06-15: MAST exposes **only `MiniMax-M3`** in the registry/UI; the M2.x
> family above is the vendor's full menu (kept here for reference) but was
> removed from MODEL_PRESETS.

## Supported params (Anthropic SDK)
| param | status |
|---|---|
| model, messages, max_tokens, stream, system | 完全支持 |
| temperature | [0,2], 建议 1 |
| top_p | [0,1]; **M3 默认 0.95**, M2.x 默认 0.9 |
| tools, tool_choice | 完全支持 |
| thinking | **完全支持**(可调推理内容) |
| metadata | 完全支持 |
| **top_k** | **忽略** |
| **stop_sequences** | **忽略** |
| mcp_servers, context_management, container | 忽略 |

## Message content blocks
- text / tool_use / tool_result / thinking — 完全支持
- image / video — **仅 M3**

## MULTI-TURN (important)
- `type="thinking"` 推理内容:**多轮 thinking 对话中需要原样回带**(pass thinking
  blocks back unchanged). The Anthropic SDK does this when the returned content
  blocks are appended verbatim to the next request's messages.
- ⚠️ AUDIT: confirm MAST's chat history for the minimax/anthropic path appends the
  full returned content (incl. thinking blocks with signature), not just text.

## Limits
- URL/base64 video ≤ 50 MB, image ≤ 10 MB, request body ≤ 64 MB (M3). Larger video via Files API.
- count_tokens supported: POST /anthropic/v1/messages/count_tokens.
