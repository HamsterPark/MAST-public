"""Render a LangGraph agent message channel into chat-markdown for the frontend.

Converts ``list[AnyMessage]`` (an ``AgentState['messages']`` read back from the
checkpointer, or accumulated live during a stream) into the chat UI's
``[{role, content}]`` shape. Self-contained (no ``mast.webui`` dependency) so the
engine and REST endpoints can render without importing the GUI.

Tool calls + their results are folded INTO the assistant turn as collapsible
blocks; reasoning/thinking (Anthropic ``thinking`` blocks or OpenAI-compat
``reasoning_content``) renders as a default-collapsed ``<details>`` like the
legacy chat did, so long traces don't dominate the view.
"""

from __future__ import annotations

import html
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _esc(text: Any) -> str:
    """Neutralise raw HTML in model/tool free text (Chatbot is sanitize_html=False).

    Escapes only ``< > &`` so markdown still renders but an echoed corpus abstract
    containing ``<img onerror=…>`` shows as literal text.
    """
    return html.escape(str(text if text is not None else ""), quote=False)


def _thinking_details(thinking: str, summary: str = "🤔 思考过程") -> str:
    if not thinking or not thinking.strip():
        return ""
    flat = " ".join(thinking.split())
    preview = _esc(flat[:60]) + ("…" if len(flat) > 60 else "")
    body = _esc(thinking)
    return (f"<details><summary>{summary} · {len(thinking)} 字 · {preview}</summary>\n\n"
            f"_{body.strip()}_\n\n</details>")


def _tool_result_block(name: str, content: str) -> str:
    flat = " ".join(str(content or "").split())
    preview = _esc(flat[:50]) + ("…" if len(flat) > 50 else "")
    body = _esc(content)
    return (f"<details><summary>🔧 {_esc(name)} → {preview}</summary>\n\n"
            f"```\n{body.strip()}\n```\n\n</details>")


def extract_ai_parts(msg) -> tuple[str, str, list[dict]]:
    """(thinking, text, tool_calls) from an AIMessage across all 6 providers.

    Anthropic carries content as a list of ``{type: text|thinking|tool_use}``
    blocks; OpenAI-compat carries a plain string + ``additional_kwargs.
    reasoning_content`` + ``.tool_calls``.
    """
    thinking, text = "", ""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts, think_parts = [], []
        for blk in content:
            if not isinstance(blk, dict):
                text_parts.append(str(blk)); continue
            t = blk.get("type")
            if t == "text":
                text_parts.append(blk.get("text", ""))
            elif t == "thinking":
                think_parts.append(blk.get("thinking", ""))
            elif t == "reasoning_content":
                think_parts.append(blk.get("reasoning_content", ""))
        text = "".join(text_parts)
        thinking = "\n".join(p for p in think_parts if p)
    if not thinking:
        ak = getattr(msg, "additional_kwargs", None) or {}
        rc = ak.get("reasoning_content") or ak.get("reasoning")
        if rc:
            thinking = str(rc)
    tool_calls = list(getattr(msg, "tool_calls", None) or [])
    return thinking, text, tool_calls


def _fmt_tool_calls(tool_calls: list[dict]) -> str:
    lines = []
    for tc in tool_calls:
        name = tc.get("name", "tool")
        args = tc.get("args", {})
        try:
            arg_s = json.dumps(args, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            arg_s = str(args)
        if len(arg_s) > 200:
            arg_s = arg_s[:200] + "…"
        lines.append(f"→ `{_esc(name)}`({_esc(arg_s)})")
    return "\n".join(lines)


def _msg_t(msg) -> "float | None":
    """这条消息**发生的时刻**,没盖过就是 ``None``。

    ``None`` = 不知道,**不是 0**：前端对无效值不渲染时间,而一个 0 会被画成
    1970 年 —— 一个假时间比没有时间坏,因为它会被当成真的去推理。
    进程重启前就存在的消息一律没有戳(见 ``message_clock_mw`` 的自述)。
    """
    try:
        from mast.agents._shared.message_clock_mw import message_time
        return message_time(msg)
    except Exception:  # noqa: BLE001 — 少一个时间戳,不许弄坏一次渲染
        return None


def _with_t(entry: dict, msg) -> dict:
    """给渲染出来的那一条带上时刻。**没有就不带这个键**,不带 0。"""
    t = _msg_t(msg)
    if t is not None:
        entry["t"] = t
    return entry


def render_history(messages: list) -> list[dict]:
    """Convert an AgentState message list → Chatbot ``[{role, content[, t]}]``.

    Tool results are folded into the assistant turn that called them.

    ``t`` = 这条消息发生的 epoch 秒,由 ``MessageClockMiddleware`` 盖上。
    **可能缺席** —— 那表示「不知道它是什么时候说的」(重启前的历史),
    不表示「零时刻」。要求：agent 的发言也带上时间;
    而 ``frontend/src/lib/narration.ts`` 里那句「要让它变精确,得给
    render_history 的每条消息加稳定时间戳」说的正是这个键。
    """
    out: list[dict] = []
    # map tool_call_id → tool name, harvested from AIMessage.tool_calls
    call_names: dict[str, str] = {}
    try:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    except Exception:  # noqa: BLE001
        return out

    for msg in messages or []:
        if isinstance(msg, HumanMessage):
            txt = msg.content if isinstance(msg.content, str) else _join_text(msg.content)
            # skip the synthetic tool-result-bearing human turns (none here) /
            # empty seeds
            if str(txt).strip():
                out.append(_with_t({"role": "user", "content": _esc(txt)}, msg))
        elif isinstance(msg, AIMessage):
            thinking, text, tool_calls = extract_ai_parts(msg)
            for tc in tool_calls:
                if tc.get("id"):
                    call_names[tc["id"]] = tc.get("name", "tool")
            parts = []
            if thinking:
                parts.append(_thinking_details(thinking))
            if text and text.strip():
                parts.append(_esc(text.strip()))
            if tool_calls:
                parts.append(_fmt_tool_calls(tool_calls))
            content = "\n\n".join(p for p in parts if p)
            out.append(_with_t(
                {"role": "assistant", "content": content or "_(无内容)_"}, msg))
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or call_names.get(
                getattr(msg, "tool_call_id", ""), "tool")
            body = msg.content if isinstance(msg.content, str) else _join_text(msg.content)
            block = _tool_result_block(name, body)
            # attach to the last assistant turn so the result reads under its call
            if out and out[-1]["role"] == "assistant":
                out[-1]["content"] = out[-1]["content"] + "\n\n" + block
            else:
                # 没有可挂靠的 assistant turn ⇒ 自成一条,用**工具消息自己**的时刻。
                # 折叠进上一条时刻意不改那条的 t：那是**发起调用**的时刻,
                # 比工具返回的时刻更早、也更贴近「这一步是什么时候开始的」。
                out.append(_with_t({"role": "assistant", "content": block}, msg))
    return out


def _join_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", "") or b.get("content", ""))
            else:
                parts.append(str(b))
        return "".join(parts)
    return str(content or "")


__all__ = ["render_history", "extract_ai_parts"]
