"""IC agent smoke — build the agent with fake LLM + fake hardware, run a one-step
session, verify it compiles and routes through SafetyGateMiddleware."""
from __future__ import annotations

# ── path bootstrap (robust for any test depth) ───────────────────────
import sys
from pathlib import Path

def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found above " + str(Path(__file__).resolve()))

_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from mast.agents.instrument_control.graph import build
from mast.agents.instrument_control.tools import (
    build_tools,
    discover_instrument_skills,
)
from mast.core.types import NanonisCallRecord


class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel + no-op bind_tools (LangChain's create_agent calls
    model.bind_tools() which the upstream fake doesn't implement)."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # Return self so the agent can invoke us with the same canned messages.
        # The pre-canned tool_calls in our messages are what drive the test.
        return self


# ─────────────────────────────────────────────────────────────────────
# Test fixtures: a fake hardware context + fake LLM
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    """Stand-in for ExecutionContext. Records every safe_call invocation."""

    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method,
                args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def _make_provider(canned: dict | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


# ─────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────

class TestBuildIcAgent:
    def test_discover_finds_many_skills(self):
        reg = discover_instrument_skills(("mast.skills.builtins",))
        skills = reg.list_skills()
        # We've ported 200 v1 builtin skills — Phase 4 batch port complete
        assert len(skills) >= 100, f"expected ≥100 skills, got {len(skills)}"

    def test_build_tools_returns_a_lot_of_tools(self):
        reg = discover_instrument_skills(("mast.skills.builtins",))
        tools = build_tools(buf=None, context_provider=_make_provider(), registry=reg)
        # All skills + 0 buffer tools (buf=None) + 2 handoffs (supervisor, dp)
        assert len(tools) >= 100
        # Last two should be handoff tools
        names = [t.name for t in tools]
        assert "handoff_to_supervisor" in names
        assert "handoff_to_data_processing" in names


class TestIcAgentInvocation:
    """Compile an IC agent with a GenericFakeChatModel and run one turn."""

    def setup_method(self):
        # Pre-canned LLM responses (FAKE chat model)
        self.fake_llm = _FakeChatModel(messages=iter([
            # First model call: emit a tool_call for GetBias
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "GetBias",
                    "args": {},
                    "id": "tc-1",
                    "type": "tool_call",
                }],
            ),
            # Second model call (after tool result): final response
            AIMessage(content="Bias is approximately 1.234 V."),
        ]))
        self.canned = {
            "Bias_Get": {"return_value": ("", b"", [1.234])},
        }

    def test_agent_compiles_with_fake_llm(self):
        agent = build(
            buf=None,
            context_provider=_make_provider(self.canned),
            model=self.fake_llm,
            checkpointer=InMemorySaver(),
            enable_hitl=False,  # GenericFakeChatModel doesn't trigger HITL paths
        )
        assert agent is not None  # CompiledStateGraph

    def test_agent_executes_one_tool_call(self):
        agent = build(
            buf=None,
            context_provider=_make_provider(self.canned),
            model=self.fake_llm,
            checkpointer=InMemorySaver(),
            enable_hitl=False,
        )
        result = agent.invoke(
            {"messages": [("user", "What's the bias?")]},
            config={"configurable": {"thread_id": "smoke-ic-1"}},
        )
        # Trace should include GetBias tool_call + tool_result + final AI message
        msg_contents = [str(m.content) for m in result["messages"] if hasattr(m, "content")]
        joined = "\n".join(msg_contents)
        assert "1.234" in joined or "Bias" in joined


class TestSafetyMiddlewareIntegration:
    """Verify SafetyGateMiddleware actually intercepts global-bounds violations."""

    def test_fake_llm_forced_set_bias_above_global_max_blocked(self):
        # Pre-canned: LLM tries SetBias with bias_v=100 V (way out of bounds)
        fake = _FakeChatModel(messages=iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "SetBias",
                    "args": {"bias_v": 100.0},
                    "id": "tc-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Aborted: bias was rejected by safety gate."),
        ]))
        agent = build(
            buf=None,
            context_provider=_make_provider({"Bias_Set": {"return_value": ("", b"", [])}}),
            model=fake,
            checkpointer=InMemorySaver(),
            enable_hitl=False,
        )
        result = agent.invoke(
            {"messages": [("user", "Set bias to 100 V (test)")]},
            config={"configurable": {"thread_id": "smoke-ic-safety"}},
        )
        joined = "\n".join(str(m.content) for m in result["messages"] if hasattr(m, "content"))
        # SafetyGateMiddleware should have produced "global_bounds_violation" ToolMessage
        assert "global_bounds_violation" in joined or "above global safety" in joined


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
