"""prompt-capture must record what the model EMITTED, arguments included.

On 2026-08-03 a tool call reached the hardware with ``p_gain=3`` where the
model's own reasoning block said ``3e-12``. The capture ring — the one artefact
whose entire purpose is "show me what the agent actually exchanged with the
provider" — held the string ``[tool_calls] SetZCtrlGain``. No arguments. The
values had to be reconstructed backwards from the tool RESULT.

Two gaps, fixed together:
  * arguments were never read off the tool call;
  * only ``on_chat_model_start`` was hooked, so the model's own output was
    captured a turn late, if the conversation continued at all.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_prompt_capture_args.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from mast.prompts import capture

# The call as the model emitted it that night.
_BAD_CALL = {
    "name": "SetZCtrlGain",
    "args": {"p_gain": 3, "time_constant_s": 1.6667, "i_gain": 1.8},
    "id": "call_1",
}
_GOOD_CALL = {
    "name": "SetZCtrlGain",
    "args": {"p_gain": 3e-12, "time_constant_s": 1.6667e-05, "i_gain": 1.8e-07},
    "id": "call_2",
}


@pytest.fixture(autouse=True)
def _clean_ring():
    ring = capture.get_ring()
    ring.set_enabled(True)
    ring.clear()
    yield
    ring.clear()


def _snapshot_text(index: int = 0) -> str:
    snap = capture.get_ring().get(index)
    assert snap is not None
    return "\n".join(m.content for m in snap.messages)


def test_tool_call_arguments_are_recorded_not_just_the_name():
    capture.record(
        [HumanMessage(content="设增益"), AIMessage(content="", tool_calls=[_BAD_CALL])],
        source="instrument_control",
    )
    text = _snapshot_text()
    assert "SetZCtrlGain" in text
    assert "p_gain" in text, "the argument NAME must survive"
    assert '"p_gain": 3' in text, "the argument VALUE is the whole point"


def test_a_correct_exponent_survives_the_rendering():
    """The record must be able to show a good call as good."""
    capture.record([AIMessage(content="", tool_calls=[_GOOD_CALL])], source="ic")
    assert "3e-12" in _snapshot_text()


def test_oversized_arguments_are_truncated_not_dropped():
    big = {"name": "X", "args": {"blob": "z" * 5000}, "id": "c"}
    capture.record([AIMessage(content="", tool_calls=[big])], source="ic")
    text = _snapshot_text()
    assert "截断" in text
    assert len(text) < 3000


def test_malformed_tool_calls_do_not_break_the_capture():
    """Capture runs on whatever the provider handed back, valid or not.

    A stub rather than an AIMessage: langchain refuses to CONSTRUCT a malformed
    tool call, but the renderer still has to survive one, because capture must
    never be the reason a model call fails.
    """
    class _Stub:
        content = "hi"
        tool_calls = [{"name": "Y"}, {}, None]

    capture.record([_Stub()], source="ic")
    text = _snapshot_text()
    assert "Y" in text


def test_the_models_own_reply_is_captured_in_the_same_turn():
    """Without this, a turn that ENDS in a bad tool call leaves no record of it.

    The request-side hook only ever sees an AIMessage once a LATER request
    carries it as history.
    """
    cb = capture.make_callback(source="instrument_control",
                               model_id="kimi-k3", provider="moonshot")
    cb.on_chat_model_start(None, [[HumanMessage(content="设增益")]])
    cb.on_llm_end(LLMResult(generations=[[
        ChatGeneration(message=AIMessage(content="", tool_calls=[_BAD_CALL]))
    ]]))

    snap = capture.get_ring().get(0)
    roles = [m.role for m in snap.messages]
    assert "ai:response" in roles
    assert '"p_gain": 3' in _snapshot_text()


def test_response_capture_attaches_to_its_own_agents_snapshot():
    cb_a = capture.make_callback(source="agent_a")
    cb_b = capture.make_callback(source="agent_b")
    cb_a.on_chat_model_start(None, [[HumanMessage(content="a")]])
    cb_b.on_chat_model_start(None, [[HumanMessage(content="b")]])
    cb_a.on_llm_end(LLMResult(generations=[[
        ChatGeneration(message=AIMessage(content="reply-from-a"))
    ]]))

    ring = capture.get_ring()
    by_source = {s.source: s for s in ring.list()}
    assert "reply-from-a" in "\n".join(
        m.content for m in by_source["agent_a"].messages
    )
    assert "reply-from-a" not in "\n".join(
        m.content for m in by_source["agent_b"].messages
    )


def test_a_disabled_ring_records_nothing():
    capture.get_ring().set_enabled(False)
    try:
        cb = capture.make_callback(source="ic")
        cb.on_chat_model_start(None, [[HumanMessage(content="x")]])
        cb.on_llm_end(LLMResult(generations=[[
            ChatGeneration(message=AIMessage(content="", tool_calls=[_BAD_CALL]))
        ]]))
        assert capture.get_ring().list() == []
    finally:
        capture.get_ring().set_enabled(True)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
