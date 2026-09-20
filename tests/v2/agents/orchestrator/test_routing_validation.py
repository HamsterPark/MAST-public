"""Orchestrator routing-validation + honesty fixes.

Covers the self-review findings the orchestrator owner was assigned:

  #53  supervisor must validate routing decisions against the WIRED agent set
       (build()'s include_agents), NOT the full 6-name roster. Routing to a
       valid-but-unwired agent crashed LangGraph with a missing-node error.
  #54  the structured-output result must be parsed INSIDE the try with type
       normalisation — a non-dict return (pydantic obj / str / None) used to
       hit .get() outside the try and crash the node with AttributeError.
  #97/#101  recursion_limit=50 is now a REAL build-time default bound via
       compiled.with_config(recursion_limit=50) — the docstring used to claim
       "graph compile-time" while nothing set it.
  #102 max_int reducer was exported-but-unused dead code → removed from state.

All tests run with NO LLM / NO network (supervisor_model=None or a tiny fake
structured-output stub). Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/orchestrator/test_routing_validation.py -q -p no:randomly
"""
from __future__ import annotations

# ── path bootstrap (robust walk-up; matches sibling orchestrator tests) ──
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

from mast.agents import state as state_mod
from mast.agents.orchestrator.graph import (
    _AGENT_NAMES,
    Route,
    _supervisor_node_factory,
    build,
)


def _base_state(**kw) -> dict:
    s = {
        "visit_count": {}, "messages": [], "executed_skills": [],
        "scan_paths": [], "scan_metadata": {}, "error_log": [],
        "pending_approvals": {},
    }
    s.update(kw)
    return s


# ════════════════════════════════════════════════════════════════════
# #54 — structured-output parsing is robust to non-dict returns
# ════════════════════════════════════════════════════════════════════

class _StructuredStub:
    """Minimal stand-in for ``model.with_structured_output(Route)``.

    ``.invoke()`` returns whatever object the test seeded — exercising the
    pydantic-obj / plain-str / None / dict shapes the parser must tolerate.
    """

    def __init__(self, to_return):
        self._to_return = to_return

    def with_structured_output(self, schema, **kw):
        return self

    def invoke(self, messages):
        return self._to_return


class _RoutePydantic:
    """A pydantic-v2-like object (has .model_dump) returning a Route dict."""

    def __init__(self, next_agent, reason="ok"):
        self._d = {"next_agent": next_agent, "reason": reason}

    def model_dump(self):
        return dict(self._d)


class TestStructuredOutputRobustness:
    def test_dict_route_dispatches_normally(self):
        model = _StructuredStub({"next_agent": "literature", "reason": "search"})
        node = _supervisor_node_factory(model)
        cmd = node(_base_state())
        assert cmd.goto == "literature"
        assert cmd.update["active_agent"] == "literature"

    def test_pydantic_route_object_is_normalised(self):
        # #54: a pydantic-style object (no .get) must be model_dump()'d, not
        # crash the node with AttributeError on .get().
        model = _StructuredStub(_RoutePydantic("data_processing"))
        node = _supervisor_node_factory(model)
        cmd = node(_base_state())
        assert cmd.goto == "data_processing"

    def test_string_naming_agent_is_recovered_not_crash(self):
        # #54 + F5 (2026-06-08): a bare string return is NOT a dict. It must NOT
        # raise AttributeError. Behaviour was upgraded by the provider-portable
        # router (_route_decision): a loosely-formatted reply that NAMES a valid
        # agent (e.g. GLM emitting `literature` / `Route(literature, …)` instead
        # of clean JSON) is now RECOVERED and dispatched, rather than discarded to
        # END. This is the whole point of F5 — don't throw away a correct route
        # just because the JSON wrapper was wrong.
        model = _StructuredStub("literature")  # a str, not a Route dict
        node = _supervisor_node_factory(model)
        cmd = node(_base_state())
        assert cmd.goto == "literature"
        assert cmd.update["active_agent"] == "literature"

    def test_unparseable_string_ends_gracefully(self):
        # The graceful-END path is preserved for a string with NO recoverable
        # agent name: the node must end via the routing-error branch, not crash.
        model = _StructuredStub("I have no idea what to do here")
        node = _supervisor_node_factory(model)
        cmd = node(_base_state())
        assert cmd.goto == END
        assert cmd.update["active_agent"] == "__end__"
        assert "Routing error" in cmd.update["messages"][-1].content

    def test_none_return_ends_gracefully(self):
        model = _StructuredStub(None)
        node = _supervisor_node_factory(model)
        cmd = node(_base_state())
        assert cmd.goto == END
        assert "Routing error" in cmd.update["messages"][-1].content

    def test_invoke_raising_still_ends(self):
        class _Boom:
            def with_structured_output(self, schema, **kw):
                return self
            def invoke(self, messages):
                raise RuntimeError("model down")
        node = _supervisor_node_factory(_Boom())
        cmd = node(_base_state())
        assert cmd.goto == END
        assert "Routing error" in cmd.update["messages"][-1].content


# ════════════════════════════════════════════════════════════════════
# #53 — routing is validated against the WIRED agent set
# ════════════════════════════════════════════════════════════════════

class TestWiredAgentValidation:
    def test_router_target_not_wired_routes_to_end(self):
        # Only literature is wired; router asks for data_processing (a valid NAME
        # but NOT a wired node). Must END, not Command(goto="data_processing")
        # which would crash LangGraph at runtime.
        model = _StructuredStub({"next_agent": "data_processing", "reason": "x"})
        node = _supervisor_node_factory(model, wired_agents=("literature",))
        cmd = node(_base_state())
        assert cmd.goto == END
        assert cmd.update["active_agent"] == "__end__"

    def test_router_target_wired_dispatches(self):
        model = _StructuredStub({"next_agent": "literature", "reason": "x"})
        node = _supervisor_node_factory(model, wired_agents=("literature",))
        cmd = node(_base_state())
        assert cmd.goto == "literature"
        # visit_count delta only counts the wired target.
        assert cmd.update["visit_count"] == {"supervisor": 1, "literature": 1}

    def test_routing_hint_to_unwired_agent_falls_through(self):
        # An agent's routing hint toward an unwired agent must NOT be dispatched
        # (would crash). With no model it falls through to the no-model END path.
        node = _supervisor_node_factory(None, wired_agents=("literature",))
        cmd = node(_base_state(routing_hints=["paper_writing"]))
        assert cmd.goto == END
        assert cmd.update["active_agent"] == "__end__"
        # stale hint cleared on the way out (None CLEARS the plural channel)
        assert cmd.update["routing_hints"] is None

    def test_routing_hint_to_wired_agent_dispatches(self):
        node = _supervisor_node_factory(None, wired_agents=("paper_writing",))
        cmd = node(_base_state(routing_hints=["paper_writing"]))
        assert cmd.goto == "paper_writing"
        assert cmd.update["routing_hints"] is None

    def test_empty_wired_set_always_ends(self):
        # Degenerate: no agents wired → every route ENDs (never a missing node).
        model = _StructuredStub({"next_agent": "literature", "reason": "x"})
        node = _supervisor_node_factory(model, wired_agents=())
        cmd = node(_base_state())
        assert cmd.goto == END

    def test_default_wired_set_is_all_six(self):
        # Back-compat: omitting wired_agents keeps the full roster behaviour.
        model = _StructuredStub({"next_agent": "paper_review", "reason": "x"})
        node = _supervisor_node_factory(model)  # no wired_agents arg
        cmd = node(_base_state())
        assert cmd.goto == "paper_review"


class TestBuildPassesWiredAgents:
    def test_build_supervisor_only_routes_to_included(self):
        # Build with a single wired agent + a router that insists on a DIFFERENT
        # (unwired) agent. The compiled graph must END instead of raising a
        # missing-node error at invoke time (end-to-end).
        model = _StructuredStub({"next_agent": "data_processing", "reason": "x"})
        from langchain_core.language_models.fake_chat_models import (
            GenericFakeChatModel,
        )

        class _FakeAgentLLM(GenericFakeChatModel):
            def bind_tools(self, tools, *, tool_choice=None, **kwargs):
                return self

        graph = build(
            buf=None,
            supervisor_model=model,
            agent_model_overrides={
                "paper_review": _FakeAgentLLM(
                    messages=iter([AIMessage(content="done")])
                )
            },
            checkpointer=InMemorySaver(),
            include_agents=("paper_review",),
        )
        result = graph.invoke(
            _base_state(messages=[("user", "hi")]),
            config={"configurable": {"thread_id": "wired-1"}},
        )
        # Router wanted data_processing (unwired) → validated away → END.
        assert result["active_agent"] == "__end__"


# ════════════════════════════════════════════════════════════════════
# #97/#101 — recursion_limit=50 is a real build-time default
# ════════════════════════════════════════════════════════════════════

class TestRecursionLimitDefault:
    def test_compiled_graph_carries_recursion_limit_50(self):
        graph = build(
            buf=None,
            supervisor_model=None,
            checkpointer=InMemorySaver(),
            include_agents=(),  # supervisor-only is fine for config inspection
        )
        # with_config binds it into the runnable's default config.
        cfg = getattr(graph, "config", None) or {}
        assert cfg.get("recursion_limit") == 50, (
            f"recursion_limit default must be 50, got {cfg!r}"
        )

    def test_explicit_invoke_recursion_limit_still_overrides(self):
        # The bound default must NOT prevent a caller from raising/lowering it.
        #
        # ``__supervisor_no_model__`` is what actually disables LLM routing;
        # ``supervisor_model=None`` alone lets build() construct a real model
        # from the configured keys, which made this recursion-limit test depend
        # on a live routing answer it never meant to exercise.
        graph = build(
            buf=None,
            supervisor_model=None,
            agent_model_overrides={"__supervisor_no_model__": True},
            checkpointer=InMemorySaver(),
            include_agents=(),
        )
        # Invoke with an explicit override; it should run (no model → END) and
        # not be clobbered by the bound default. We just assert it runs cleanly.
        result = graph.invoke(
            _base_state(messages=[("user", "hi")]),
            config={"configurable": {"thread_id": "rl-override-1"},
                    "recursion_limit": 75},
        )
        assert result["active_agent"] == "__end__"


# ════════════════════════════════════════════════════════════════════
# #102 — max_int reducer removed (was exported-but-unused dead code)
# ════════════════════════════════════════════════════════════════════

class TestMaxIntRemoved:
    def test_max_int_not_in_state_module(self):
        assert not hasattr(state_mod, "max_int")

    def test_max_int_not_exported(self):
        assert "max_int" not in state_mod.__all__

    def test_surviving_reducers_still_present(self):
        # Guard against an over-eager deletion: the reducers actually wired to
        # channels must remain.
        for name in ("dedupe_append", "dedupe_event_refs", "merge_dicts",
                     "sum_int_dicts", "merge_libraries"):
            assert hasattr(state_mod, name), name
            assert name in state_mod.__all__, name


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-p", "no:randomly"])
