"""Tool-carrying model calls must not be streamed.

Partial JSON recovery can turn a truncated exponent into a shorter number or
a truncated decimal into zero without raising. Tests verify nonstreamed tool
calls so these recoverable fragments cannot become instrument parameters."""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
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

from mast.agents._shared.models import make_chat_model

_TOOLS = [{"type": "function", "function": {"name": "SetZCtrlGain"}}]


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    for var in ("MOONSHOT_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(var, "test-key-not-used")


@pytest.mark.parametrize("model_id", ["kimi-k3", "claude-opus-4-7"])
def test_both_provider_branches_disable_streaming_for_tool_calls(model_id):
    """OpenAI-compatible and Anthropic branches build different classes."""
    model = make_chat_model(model_id=model_id)
    assert model.disable_streaming == "tool_calling"


def test_a_streaming_handler_does_not_re_enable_streaming_for_tool_calls():
    """The setting has to WIN over an attached streaming handler.

    langgraph attaches ``StreamMessagesHandler`` — a ``_StreamingCallbackHandler``
    — whenever ``stream_mode`` includes "messages", and its presence is otherwise
    enough to turn streaming on. If ``disable_streaming`` did not take precedence,
    the voice path would keep streaming tool calls and this fix would be cosmetic.
    """
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import _StreamingCallbackHandler

    class _Streaming(_StreamingCallbackHandler):
        def tap_output_aiter(self, run_id, output):  # pragma: no cover
            return output

        def tap_output_iter(self, run_id, output):  # pragma: no cover
            return output

    model = make_chat_model(model_id="kimi-k3")
    import uuid

    mgr = CallbackManagerForLLMRun(
        run_id=uuid.uuid4(), handlers=[_Streaming()], inheritable_handlers=[]
    )
    assert model._should_stream(async_api=False, run_manager=mgr) is True, (
        "sanity: a streaming handler alone should enable streaming"
    )
    assert model._should_stream(
        async_api=False, run_manager=mgr, tools=_TOOLS
    ) is False, "a tool-carrying call must not stream"


def test_tool_less_helpers_keep_streaming():
    """Summarisation/compaction and friends are unaffected — no tools, no change."""
    model = make_chat_model(model_id="kimi-k3")
    assert model.disable_streaming != True  # noqa: E712 — must not be blanket-True


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
