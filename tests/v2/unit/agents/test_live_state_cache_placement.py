"""The live-state block must not sit where the prompt-cache breakpoint goes.

Dispatch audit 2026-07-28 次级「prompt cache 断点落在实时读数之后」: every field
in the live block is ``:.4e``-formatted and re-read per call, so it changes on
essentially every turn. It used to be appended to the END of the system message,
and ``AnthropicPromptCachingMiddleware`` puts its breakpoint at exactly that end.
LiveState runs outermost, so the breakpoint landed AFTER the volatile text — the
tools block (224 tools, the big one) cached while the system message and the
whole history missed every turn, at 1 h-TTL write prices.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_live_state_cache_placement.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2 = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from mast.agents._shared.live_state_mw import LiveStateMiddleware  # noqa: E402
from mast.core.types import HardwareState  # noqa: E402


class _Req:
    def __init__(self, system_message, messages):
        self.system_message = system_message
        self.messages = list(messages)


def _state(bias=1.0):
    return HardwareState(bias_v=bias, current_a=1e-10, scan_width_m=1e-7)


def _mw(bias=1.0):
    return LiveStateMiddleware(lambda: _state(bias))


SYS = SystemMessage(content="你是 instrument_control。")


def test_the_system_message_is_left_byte_identical():
    mw = _mw()
    req = _Req(SYS, [HumanMessage(content="扫一张图")])
    mw._apply(req)
    assert req.system_message.content == SYS.content, (
        "the volatile block is back in the system message — the cache "
        "breakpoint sits at its end, so system + all history miss every turn")


def test_the_block_rides_on_the_last_human_message():
    mw = _mw()
    req = _Req(SYS, [HumanMessage(content="扫一张图")])
    mw._apply(req)
    assert "MAGNITUDE CHECK" in req.messages[-1].content
    assert "扫一张图" in req.messages[-1].content


def test_the_original_message_object_is_not_mutated():
    """The list in the request is the checkpointed one; a per-call injection
    that mutated it would write the readings into durable history."""
    mw = _mw()
    original = HumanMessage(content="扫一张图")
    req = _Req(SYS, [original])
    mw._apply(req)
    assert original.content == "扫一张图"
    assert req.messages[0] is not original


def test_the_system_message_is_stable_across_turns_with_changing_readings():
    """The property the cache actually needs."""
    a, b = _Req(SYS, [HumanMessage(content="q")]), _Req(SYS, [HumanMessage(content="q")])
    _mw(1.0)._apply(a)
    _mw(2.5)._apply(b)
    assert a.system_message.content == b.system_message.content
    assert a.messages[-1].content != b.messages[-1].content, "the readings did change"


def test_it_picks_the_LAST_human_turn_not_the_first():
    mw = _mw()
    req = _Req(SYS, [
        HumanMessage(content="第一轮"),
        AIMessage(content="好"),
        HumanMessage(content="第二轮"),
    ])
    mw._apply(req)
    assert "MAGNITUDE CHECK" not in req.messages[0].content
    assert "MAGNITUDE CHECK" in req.messages[2].content


def test_multimodal_content_keeps_its_blocks():
    mw = _mw()
    req = _Req(SYS, [HumanMessage(content=[{"type": "text", "text": "看这张图"}])])
    mw._apply(req)
    blocks = req.messages[-1].content
    assert isinstance(blocks, list) and len(blocks) == 2
    assert blocks[0]["text"] == "看这张图"
    assert "MAGNITUDE CHECK" in blocks[1]["text"]


def test_falls_back_to_the_system_message_when_there_is_no_human_turn():
    """Correct numbers outrank a cache hit."""
    mw = _mw()
    req = _Req(SYS, [ToolMessage(content="ok", tool_call_id="t1")])
    mw._apply(req)
    assert "MAGNITUDE CHECK" in req.system_message.content


def test_no_state_is_still_a_no_op():
    mw = LiveStateMiddleware(lambda: None)
    req = _Req(SYS, [HumanMessage(content="q")])
    mw._apply(req)
    assert req.system_message is SYS
    assert req.messages[-1].content == "q"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
