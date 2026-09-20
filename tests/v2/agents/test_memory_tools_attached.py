"""Cross-session memory tools attach to every agent + the orchestrator.

End-to-end check that the shared persistent-memory tools (from
``mast.agents._shared.memory_tools``) reach each agent's tool set via the
``extra_tools`` build() param, and that the orchestrator threads them down to
its sub-graphs via its ``memory_tools`` param.

The tools come from the SHARED module and are passed in as parameters — no
agent imports a sibling, so the cross-agent-import invariant holds.
"""
from __future__ import annotations

# ── path bootstrap (robust walk-up, mirrors the agent smoke tests) ────
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
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
# Evict stale mast.* modules that may have resolved to v1 outside MASTv2.
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from mast.agents._shared.memory_tools import make_memory_tools  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel with no-op bind_tools / structured-output so
    create_agent() (and the orchestrator supervisor) won't crash."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return self


def _fake_model() -> _FakeChatModel:
    return _FakeChatModel(messages=iter([AIMessage(content="ok")]))


@tool("zzz_probe_tool")
def zzz_probe_tool(x: str) -> str:
    """A throwaway probe tool used to assert extra_tools wiring."""
    return x


def _compiled_tool_names(agent) -> set[str]:
    """Tool names registered on a compiled create_agent() graph.

    create_agent wires its ToolNode under the ``tools`` node; the ToolNode's
    ``tools_by_name`` is the authoritative set of tools the agent can call.
    """
    node = agent.nodes.get("tools")
    bound = getattr(node, "bound", node)
    by_name = getattr(bound, "tools_by_name", None)
    assert by_name is not None, (
        "compiled agent has no tools node with tools_by_name; "
        "graph shape changed?"
    )
    return set(by_name.keys())


def _build_agent(name: str, *, extra_tools):
    """Build one of the 6 agents by name with a fake model + extra_tools."""
    model = _fake_model()
    if name == "literature":
        from mast.agents.literature.graph import build
        return build(buf=None, model=model, extra_tools=extra_tools)
    if name == "experiment_design":
        from mast.agents.experiment_design.graph import build
        return build(buf=None, model=model, extra_tools=extra_tools)
    if name == "data_processing":
        from mast.agents.data_processing.graph import build
        return build(buf=None, model=model, extra_tools=extra_tools)
    if name == "paper_writing":
        from mast.agents.paper_writing.graph import build
        return build(buf=None, model=model, extra_tools=extra_tools)
    if name == "paper_review":
        from mast.agents.paper_review.graph import build
        return build(buf=None, model=model, extra_tools=extra_tools)
    if name == "instrument_control":
        from mast.agents.instrument_control.graph import build

        class _FakeCtx:
            pass

        return build(
            buf=None,
            context_provider=lambda: _FakeCtx(),
            model=model,
            enable_hitl=False,  # no checkpointer in this test
            extra_tools=extra_tools,
        )
    raise ValueError(name)


_ALL_AGENTS = (
    "literature",
    "experiment_design",
    "data_processing",
    "paper_writing",
    "paper_review",
    "instrument_control",
)


# ─────────────────────────────────────────────────────────────────────
# Test 1 — every agent accepts extra_tools and surfaces them
# ─────────────────────────────────────────────────────────────────────

class TestExtraToolsAttachToEachAgent:
    @pytest.mark.parametrize("agent_name", _ALL_AGENTS)
    def test_probe_tool_in_compiled_agent(self, agent_name):
        """A single @tool passed via extra_tools must appear on the agent."""
        baseline = _compiled_tool_names(_build_agent(agent_name, extra_tools=None))
        with_extra = _compiled_tool_names(
            _build_agent(agent_name, extra_tools=[zzz_probe_tool])
        )
        assert "zzz_probe_tool" in with_extra, (
            f"{agent_name}: extra_tools probe not attached; "
            f"tools={sorted(with_extra)}"
        )
        # Exactly one tool added (no accidental duplication / drop).
        assert with_extra == baseline | {"zzz_probe_tool"}, (
            f"{agent_name}: expected baseline + 1 tool, got delta "
            f"{with_extra ^ baseline}"
        )

    @pytest.mark.parametrize("agent_name", _ALL_AGENTS)
    def test_none_extra_tools_is_noop(self, agent_name):
        """extra_tools=None must leave the tool set unchanged (backwards-compat)."""
        a = _compiled_tool_names(_build_agent(agent_name, extra_tools=None))
        b = _compiled_tool_names(_build_agent(agent_name, extra_tools=[]))
        assert a == b, f"{agent_name}: [] vs None differ — {a ^ b}"


# ─────────────────────────────────────────────────────────────────────
# Test 2 — the real make_memory_tools(provider) 4 tools attach
# ─────────────────────────────────────────────────────────────────────

class TestRealMemoryToolsAttach:
    def _provider(self):
        # store=None is fine: tools degrade gracefully ("memory store
        # unavailable"); we only assert they are WIRED, not exercised here.
        return lambda: {"store": None, "namespace": "global",
                        "experiment_id": None, "author": "agent"}

    def test_four_memory_tools_built(self):
        mem = make_memory_tools(self._provider())
        names = {t.name for t in mem}
        assert names == {"memory_write", "memory_read",
                         "memory_list", "memory_search"}

    @pytest.mark.parametrize("agent_name", _ALL_AGENTS)
    def test_memory_tools_on_agent(self, agent_name):
        mem = make_memory_tools(self._provider())
        names = _compiled_tool_names(_build_agent(agent_name, extra_tools=mem))
        for n in ("memory_write", "memory_read", "memory_list", "memory_search"):
            assert n in names, f"{agent_name}: missing memory tool {n}"


# ─────────────────────────────────────────────────────────────────────
# Test 3 — orchestrator threads memory_tools down to its sub-graphs
# ─────────────────────────────────────────────────────────────────────

class TestOrchestratorThreadsMemoryTools:
    def test_orchestrator_builds_with_memory_tools(self):
        """orchestrator.build(memory_tools=[...]) compiles and the literature
        sub-graph contains the passed tool."""
        from mast.agents.orchestrator.graph import build as build_orch

        orch = build_orch(
            buf=None,
            supervisor_model=None,
            agent_model_overrides={"literature": _fake_model()},
            include_agents=("literature",),
            memory_tools=[zzz_probe_tool],
        )
        # The literature sub-graph is a compiled node on the orchestrator.
        lit = orch.nodes.get("literature")
        sub = getattr(lit, "bound", lit)
        names = _compiled_tool_names(sub)
        assert "zzz_probe_tool" in names, (
            f"orchestrator did not thread memory_tools into literature: "
            f"{sorted(names)}"
        )

    def test_orchestrator_memory_tools_default_none(self):
        """Omitting memory_tools must still build (backwards-compat)."""
        from mast.agents.orchestrator.graph import build as build_orch

        orch = build_orch(
            buf=None,
            supervisor_model=None,
            agent_model_overrides={"literature": _fake_model()},
            include_agents=("literature",),
        )
        assert orch is not None
        lit = orch.nodes.get("literature")
        sub = getattr(lit, "bound", lit)
        assert "zzz_probe_tool" not in _compiled_tool_names(sub)

    def test_real_memory_tools_through_orchestrator(self):
        """The real 4 memory tools reach a sub-graph via the orchestrator."""
        from mast.agents.orchestrator.graph import build as build_orch

        mem = make_memory_tools(
            lambda: {"store": None, "namespace": "global",
                     "experiment_id": None, "author": "agent"}
        )
        orch = build_orch(
            buf=None,
            supervisor_model=None,
            agent_model_overrides={"paper_review": _fake_model()},
            include_agents=("paper_review",),
            memory_tools=mem,
        )
        pr = orch.nodes.get("paper_review")
        sub = getattr(pr, "bound", pr)
        names = _compiled_tool_names(sub)
        for n in ("memory_write", "memory_read", "memory_list", "memory_search"):
            assert n in names, f"orchestrator paper_review missing {n}"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
