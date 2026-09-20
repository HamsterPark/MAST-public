"""Regression tests for context-engineering Layer 2 (offload) + Layer 3
(ToolRefinementMiddleware, before_model). See docs/v2/design/context_engineering.md.

before_model is exercised directly (it fires before every model call in both
private and group ReAct loops); we do not drive a full create_agent loop here
because GenericFakeChatModel cannot bind/emit tool_calls.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mast.agents._shared.tool_refine_mw import ToolRefinementMiddleware, _MARKER
from mast.agents._shared.skill_adapter import _offload_long_summary
from mast.webui.settings_store import KNOWN_KEYS


class _R:
    def __init__(self, c):
        self.content = c


class _FakeSumm:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return _R("[refined] bias=1.5V x=2.0nm ok")


_LONG = "verbose narrative padding describing the result at length " * 20 + " bias=1.5V"


def _state_two_tools():
    return {"messages": [
        HumanMessage("a"),
        AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
        ToolMessage(content=_LONG, tool_call_id="c1", name="t", status="success", id="tm1"),
        AIMessage("ok"),
        AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c2"}]),
        ToolMessage(content=_LONG + " current=30pA", tool_call_id="c2", name="t",
                    status="success", id="tm2"),
        AIMessage("ok2"),
    ]}


def test_reuse_id_inplace_and_preserve_fields():
    mw = ToolRefinementMiddleware(summarizer_model=_FakeSumm(), min_chars=600, keep_recent=1)
    out = mw.before_model(_state_two_tools(), None)
    assert len(out["messages"]) == 1
    r = out["messages"][0]
    assert r.id == "tm1"
    assert (r.tool_call_id, r.name, r.status) == ("c1", "t", "success")
    assert r.additional_kwargs.get(_MARKER) is True
    assert r.content.startswith("[refined]")


def test_keep_recent_untouched_and_idempotent():
    mw = ToolRefinementMiddleware(summarizer_model=_FakeSumm(), min_chars=600, keep_recent=1)
    st = _state_two_tools()
    st["messages"][2].additional_kwargs[_MARKER] = True
    assert mw.before_model(st, None) is None


def test_failsafe_keeps_original():
    class Boom:
        def invoke(self, p):
            raise RuntimeError("down")
    mw = ToolRefinementMiddleware(summarizer_model=Boom(), min_chars=600, keep_recent=1)
    assert mw.before_model(_state_two_tools(), None) is None


def test_never_inflate():
    class Big:
        def invoke(self, p):
            return _R("X" * 999999)
    mw = ToolRefinementMiddleware(summarizer_model=Big(), min_chars=600, keep_recent=1)
    assert mw.before_model(_state_two_tools(), None) is None


def test_short_and_nontext_skipped():
    mw = ToolRefinementMiddleware(summarizer_model=_FakeSumm(), min_chars=600, keep_recent=1)
    st = {"messages": [
        AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
        ToolMessage(content="short", tool_call_id="c1", name="t", status="success", id="s1"),
        AIMessage("x"),
        AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c2"}]),
        ToolMessage(content="also short", tool_call_id="c2", name="t", status="success", id="s2"),
    ]}
    assert mw.before_model(st, None) is None


def test_settings_keys_registered():
    for k in ("compaction_model", "tool_refine_enabled", "tool_refine_min_chars",
              "chat_model_calls_per_run", "chat_tool_calls_per_thread"):
        assert k in KNOWN_KEYS, k


def test_offload_short_passthrough():
    assert _offload_long_summary("sk", "abc") == "abc"


def test_offload_long_truncates_and_refs():
    out = _offload_long_summary("MySkill", "y" * 3000)
    assert len(out) < 3000
    assert "truncated 3000 chars" in out and "saved:" in out
