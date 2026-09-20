"""RecorderMiddleware — cross-provider agent_turn capture (RFC §3, #8 P2).

Covers the reasoning_content portability matrix (risk #7): OpenAI-compatible
providers (Kimi/DeepSeek/Qwen/GLM) surface reasoning on
``additional_kwargs.reasoning_content``; Anthropic-compatible providers (Claude
adaptive thinking / MiniMax) surface it as ``type=="thinking"`` content blocks.
A provider with no reasoning records None, never crashes, and the capture is
read-only + fail-safe (must never break the agent run)."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mast.agents._shared.recorder_mw import (
    RecorderMiddleware,
    build_turn_payload,
)


# ── reasoning extraction across providers ────────────────────────────

def test_openai_compat_reasoning_from_additional_kwargs():
    """Kimi/DeepSeek/Qwen/GLM: reasoning_content lands on additional_kwargs."""
    msg = AIMessage(
        content="The bias should be 1.0 V.",
        additional_kwargs={"reasoning_content": "Let me reason: lower bias for HOPG..."},
    )
    p = build_turn_payload("instrument_control", msg)
    assert p["reasoning"] == "Let me reason: lower bias for HOPG..."
    assert p["content"] == "The bias should be 1.0 V."
    assert p["agent_id"] == "instrument_control"


def test_anthropic_compat_reasoning_from_thinking_blocks():
    """Claude/MiniMax: extended thinking arrives as type==thinking content blocks."""
    msg = AIMessage(content=[
        {"type": "thinking", "thinking": "First I consider the tip state...",
         "signature": "abc"},
        {"type": "text", "text": "I'll set the bias to 1.0 V."},
    ])
    p = build_turn_payload("instrument_control", msg)
    assert p["reasoning"] == "First I consider the tip state..."
    assert p["content"] == "I'll set the bias to 1.0 V."


def test_anthropic_multiple_thinking_blocks_joined():
    msg = AIMessage(content=[
        {"type": "thinking", "thinking": "part A"},
        {"type": "thinking", "thinking": "part B"},
        {"type": "text", "text": "answer"},
    ])
    p = build_turn_payload("literature", msg)
    assert p["reasoning"] == "part A\npart B"


def test_redacted_thinking_block_skipped():
    """Encrypted redacted_thinking carries no readable text → no reasoning."""
    msg = AIMessage(content=[
        {"type": "redacted_thinking", "data": "encrypted-blob"},
        {"type": "text", "text": "done"},
    ])
    p = build_turn_payload("paper_review", msg)
    assert p["reasoning"] is None
    assert p["content"] == "done"


def test_no_reasoning_records_none_not_crash():
    """A non-reasoning provider/turn must record None, never raise."""
    msg = AIMessage(content="plain answer")
    p = build_turn_payload("data_processing", msg)
    assert p["reasoning"] is None
    assert p["content"] == "plain answer"


def test_empty_reasoning_string_is_none():
    msg = AIMessage(content="x", additional_kwargs={"reasoning_content": "   "})
    assert build_turn_payload("x", msg)["reasoning"] is None


# ── tool_calls / usage / meta ────────────────────────────────────────

def test_tool_calls_extracted():
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "SetBias", "args": {"bias_v": 1.0}, "id": "call_1"},
            {"name": "Scan", "args": {}, "id": "call_2"},
        ],
    )
    p = build_turn_payload("instrument_control", msg)
    assert [t["name"] for t in p["tool_calls"]] == ["SetBias", "Scan"]
    assert p["tool_calls"][0]["args"] == {"bias_v": 1.0}
    assert p["tool_calls"][0]["id"] == "call_1"


def test_usage_and_meta_extracted():
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        response_metadata={"model_name": "kimi-k2.6", "finish_reason": "stop"},
    )
    p = build_turn_payload("literature", msg)
    assert p["usage"] == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    assert p["model_id"] == "kimi-k2.6"
    assert p["finish_reason"] == "stop"


def test_anthropic_stop_reason_meta():
    msg = AIMessage(content="hi", response_metadata={"model": "claude-opus-4-8",
                                                     "stop_reason": "end_turn"})
    p = build_turn_payload("x", msg)
    assert p["model_id"] == "claude-opus-4-8"
    assert p["finish_reason"] == "end_turn"


# ── after_model hook: fires recorder, fail-safe, read-only ───────────

def _state(*msgs):
    return {"messages": list(msgs)}


def test_after_model_fires_recorder():
    rec = []
    mw = RecorderMiddleware(agent_id="literature", recorder=rec.append)
    out = mw.after_model(_state(HumanMessage("go"),
                                AIMessage(content="done",
                                          additional_kwargs={"reasoning_content": "think"})),
                         runtime=None)
    assert out is None  # read-only: no state delta
    assert len(rec) == 1
    assert rec[0]["reasoning"] == "think"
    assert rec[0]["agent_id"] == "literature"


def test_after_model_ignores_non_aimessage_last():
    rec = []
    mw = RecorderMiddleware(agent_id="x", recorder=rec.append)
    mw.after_model(_state(AIMessage(content="a"), ToolMessage("result", tool_call_id="t1")),
                   runtime=None)
    assert rec == []


def test_none_recorder_is_noop():
    mw = RecorderMiddleware(agent_id="x", recorder=None)
    # must not raise
    assert mw.after_model(_state(AIMessage(content="a")), runtime=None) is None


def test_recorder_failure_never_breaks_run():
    def boom(_):
        raise ValueError("sink down")
    mw = RecorderMiddleware(agent_id="x", recorder=boom)
    # the swallowed error must not propagate out of after_model
    assert mw.after_model(_state(AIMessage(content="a")), runtime=None) is None


def test_empty_messages_no_crash():
    rec = []
    mw = RecorderMiddleware(agent_id="x", recorder=rec.append)
    assert mw.after_model(_state(), runtime=None) is None
    assert rec == []


@pytest.mark.asyncio
async def test_aafter_model_fires_recorder():
    rec = []
    mw = RecorderMiddleware(agent_id="paper_writing", recorder=rec.append)
    out = await mw.aafter_model(_state(AIMessage(content="z")), runtime=None)
    assert out is None
    assert len(rec) == 1
    assert rec[0]["agent_id"] == "paper_writing"
