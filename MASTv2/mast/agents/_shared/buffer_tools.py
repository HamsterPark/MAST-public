"""Buffer-read tools — agents query latest vision/scan state without blocking.

These are the ONLY way an agent (LLM-side) reads the s-scale BufferService.
Agents never write to the buffer; only the VisionProducer thread does.

Usage in any agent's tool list:
    from mast.agents._shared.buffer_tools import make_buffer_tools
    tools = [
        ...domain_tools,
        *make_buffer_tools(buf),       # adds 3 read-only buffer tools
    ]
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

if TYPE_CHECKING:
    from mast.buffer.service import BufferService


def make_buffer_tools(buf: "BufferService") -> list:
    """Return three read-only LangChain tools bound to the given BufferService."""

    @tool("read_latest_tip_status")
    def read_latest_tip_status() -> dict:
        """Read the most recent tip assessment from the vision buffer.

        Returns {seqno, tip} where tip is None if no assessment yet, otherwise
        a dict with quality / confidence / scan_id / frame_idx / t_mono_ns.
        Non-blocking — returns immediately even if vision thread is mid-frame.
        """
        ts, seq = buf.get_latest_tip_status()
        return {"seqno": seq, "tip": ts.model_dump() if ts is not None else None}

    # Previous (line_idx, t_mono_ns) this tool reported, so a caller can be told
    # whether the scan ADVANCED between two reads. See the tool docstring.
    _last_seen: dict = {}

    @tool("get_scan_progress")
    def get_scan_progress() -> dict:
        """当前扫描进度（第几行 / 共几行 / 预计剩余秒数），以及**它是否还在推进**。

        返回 {seqno, progress, age_s, advancing, note}：
          * progress —— 最后一次发布的进度；**没有新扫描时它不会消失**，会一直停在
            上一次扫描结束时的值。
          * age_s    —— 这条进度是多久以前发布的（秒）。几十秒不动 = 扫描没在推进。
          * advancing —— 与**你上一次调用本工具**相比，line_idx 是否前进了。
            None 表示这是第一次调用，无从比较。

        **不要用 seqno 判断扫描是否在跑。** seqno 是整个视觉缓冲区的全局序号，任何
        事件都会让它增长；扫描早已停止时它照样在涨。例如：
        agent 看到「新 seqno + 第 2/512 行」，断定「硬件扫描继续，中止只打断了等待
        调用」，于是又等了两轮，而扫描其实已经停了。
        判据用 advancing / age_s；要确认硬件状态请读 GetScanFrame / 扫描状态。
        """
        import time as _t

        p, seq = buf.get_latest_progress()
        if p is None:
            return {"seqno": seq, "progress": None, "age_s": None,
                    "advancing": None, "note": "从未发布过扫描进度。"}
        d = p.model_dump()
        age_s = None
        t_ns = d.get("t_mono_ns")
        if isinstance(t_ns, (int, float)) and t_ns:
            age_s = round(max(0.0, (_t.monotonic_ns() - float(t_ns)) / 1e9), 1)
        line = d.get("line_idx")
        prev_line = _last_seen.get("line_idx")
        advancing = None if prev_line is None else (
            isinstance(line, (int, float)) and isinstance(prev_line, (int, float))
            and line > prev_line)
        _last_seen["line_idx"] = line
        if advancing is False:
            note = (f"自你上次查询以来行号没有变化（仍在第 {line} 行）——"
                    "扫描很可能已经停止，不要再等。")
        elif age_s is not None and age_s > 60:
            note = f"这条进度是 {age_s:.0f} 秒前发布的——扫描很可能已经停止。"
        else:
            note = ""
        return {"seqno": seq, "progress": d, "age_s": age_s,
                "advancing": advancing, "note": note}

    @tool("get_tip_history_since")
    def get_tip_history_since(since_seq: int) -> list[dict]:
        """Get tip-status entries newer than `since_seq`.

        Use this when an agent woke up via subscribe() and wants to catch up on
        what happened. Returns list of TipStatus dicts (may be empty).
        """
        history = buf.get_tip_history(since_seq)
        return [t.model_dump() for t in history]

    return [read_latest_tip_status, get_scan_progress, get_tip_history_since]


__all__ = ["make_buffer_tools"]
