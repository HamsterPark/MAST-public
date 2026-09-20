"""Orchestrator smoke — top-level graph wiring + loop guard + END routing."""
from __future__ import annotations

# ── path bootstrap (robust walk-up) ──
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
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from mast.agents.orchestrator.graph import build, Route, _supervisor_node_factory


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self
    def with_structured_output(self, schema, **kwargs):
        # Used by supervisor_node; return a stub that returns the next AIMessage parsed as Route
        return self


# ─────────────────────────────────────────────────────────────────────
# Direct supervisor_node tests (bypass full graph)
# ─────────────────────────────────────────────────────────────────────

class TestSupervisorLoopGuard:
    def test_total_visits_too_high_ends(self):
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node({"visit_count": {"a": 41}, "messages": [], "executed_skills": [],
                     "scan_paths": [], "scan_metadata": {}, "error_log": [],
                     "pending_approvals": {}})
        assert "Loop guard tripped" in cmd.update["messages"][0].content

    def test_total_visits_too_high_ends(self):
        # Per-agent cap removed 2026-06-29 → the total-hop guard (>40) is the
        # loop backstop. A single agent at 41 makes total 41 > 40 → guard trips.
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node({"visit_count": {"a": 41}, "messages": [], "executed_skills": [],
                     "scan_paths": [], "scan_metadata": {}, "error_log": [],
                     "pending_approvals": {}})
        assert "Loop guard tripped" in cmd.update["messages"][0].content

    def test_budget_exhausted_ends(self):
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node({"budget_remaining_usd": 0.0, "visit_count": {}, "messages": [],
                     "executed_skills": [], "scan_paths": [], "scan_metadata": {},
                     "error_log": [], "pending_approvals": {}})
        assert "Budget exhausted" in cmd.update["messages"][0].content


class TestSupervisorNoModel:
    def test_ends_when_no_model_configured(self):
        node = _supervisor_node_factory(supervisor_model=None)
        cmd = node({"visit_count": {}, "messages": [], "executed_skills": [],
                     "scan_paths": [], "scan_metadata": {}, "error_log": [],
                     "pending_approvals": {}})
        assert cmd.update["active_agent"] == "__end__"
        # visit_count incremented
        assert cmd.update["visit_count"]["supervisor"] == 1


# ─────────────────────────────────────────────────────────────────────
# Full orchestrator graph build (subset of agents)
# ─────────────────────────────────────────────────────────────────────

class TestOrchestratorBuild:
    def test_minimal_orchestrator_with_no_agents(self):
        # include_agents=() means only supervisor node is wired (we still need at
        # least one node + edge) so include just one cheap agent (paper_review
        # with fake LLM).
        #
        # ``__supervisor_no_model__`` is what actually disables LLM routing.
        # ``supervisor_model=None`` alone does NOT: build() then constructs a
        # real orchestrator model from the configured provider keys, so on any
        # machine that has keys this smoke test was quietly making a live routing
        # call and asserting on whatever that model happened to answer.
        fake = _FakeChatModel(messages=iter([AIMessage(content="reviewed")]))
        graph = build(
            buf=None,
            supervisor_model=None,
            agent_model_overrides={"paper_review": fake,
                                   "__supervisor_no_model__": True},
            checkpointer=InMemorySaver(),
            include_agents=("paper_review",),
        )
        # First invocation: supervisor → END (no model)
        result = graph.invoke(
            {"messages": [("user", "hello")], "visit_count": {},
             "executed_skills": [], "scan_paths": [], "scan_metadata": {},
             "error_log": [], "pending_approvals": {}},
            config={"configurable": {"thread_id": "smoke-orch-1"}},
        )
        assert "active_agent" in result
        assert result["active_agent"] == "__end__"

    def test_instrument_control_requires_context_provider(self):
        with pytest.raises(ValueError, match="context_provider"):
            build(
                buf=None,
                supervisor_model=None,
                checkpointer=InMemorySaver(),
                include_agents=("instrument_control",),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
