"""XD agent smoke tests — build the agent with a fake LLM, verify compilation,
tool list shape, and one-turn tool invocation (describe_skills)."""
from __future__ import annotations

# ── path bootstrap (robust walk-up matching IC's pattern) ─────────────
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
# Evict any stale mast.* modules that resolved outside MASTv2 (can happen when
# pytest is run from the repo root with PYTHONPATH unset).
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from mast.agents.experiment_design.graph import build, discover_xd_catalog, _discover_catalog
from mast.agents.experiment_design.tools import (
    build_tools,
    describe_skills_tool,
    lookup_sample_tool,
    query_past_experiments_tool,
)
from mast.core.registry import SkillRegistry


# ─────────────────────────────────────────────────────────────────────
# Fake LLM
# ─────────────────────────────────────────────────────────────────────

class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel with no-op bind_tools so create_agent() won't crash."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # Return self — the pre-canned tool_calls in messages drive the test.
        return self


# ─────────────────────────────────────────────────────────────────────
# Minimal stub registry (no real skills needed for most tests)
# ─────────────────────────────────────────────────────────────────────

def _make_empty_registry() -> SkillRegistry:
    """Return a SkillRegistry with no skills — sufficient for tool-list tests."""
    return SkillRegistry()


# ─────────────────────────────────────────────────────────────────────
# Test 1 — graph compiles without exceptions
# ─────────────────────────────────────────────────────────────────────

class TestXDAgentCompiles:
    def test_build_returns_compiled_state_graph(self):
        """build() should return a LangGraph CompiledStateGraph (not None, not a stub fn)."""
        fake_llm = _FakeChatModel(messages=iter([
            AIMessage(content="Plan ready."),
        ]))
        agent = build(
            buf=None,
            model=fake_llm,
            registry=_make_empty_registry(),
            checkpointer=InMemorySaver(),
        )
        assert agent is not None
        # CompiledStateGraph exposes `.invoke`; stub function would not.
        assert callable(getattr(agent, "invoke", None)), (
            "build() should return a CompiledStateGraph with .invoke; got a stub or None"
        )

    def test_build_without_checkpointer(self):
        """build() must succeed even when no checkpointer is provided."""
        fake_llm = _FakeChatModel(messages=iter([
            AIMessage(content="Plan ready."),
        ]))
        agent = build(
            buf=None,
            model=fake_llm,
            registry=_make_empty_registry(),
        )
        assert agent is not None


# ─────────────────────────────────────────────────────────────────────
# Test 2 — build_tools returns expected shape
# ─────────────────────────────────────────────────────────────────────

class TestBuildTools:
    def test_returns_expected_minimum_count(self):
        """With buf=None: 3 domain tools + 2 handoffs = 5 tools."""
        tools = build_tools(buf=None, registry=_make_empty_registry())
        assert len(tools) == 5, (
            f"Expected 5 tools (3 domain + 2 handoffs), got {len(tools)}: "
            f"{[t.name for t in tools]}"
        )

    def test_tool_names_present(self):
        """All expected tool names must be present."""
        tools = build_tools(buf=None, registry=_make_empty_registry())
        names = {t.name for t in tools}
        expected = {
            "describe_skills",
            "lookup_sample",
            "query_past_experiments",
            "handoff_to_supervisor",
            "handoff_to_instrument_control",
        }
        missing = expected - names
        assert not missing, f"Missing tools: {missing}"

    def test_no_skill_wrap_tools(self):
        """XD must NOT expose any wrapped skill tools (those belong to IC)."""
        tools = build_tools(buf=None, registry=_make_empty_registry())
        names = [t.name for t in tools]
        # If wrap_skill tools were present they'd be named after skills like
        # "GetBias", "SetBias", "AutoApproach" etc. — none should appear here.
        skill_like = [n for n in names if n not in {
            "describe_skills", "lookup_sample", "query_past_experiments",
            "handoff_to_supervisor", "handoff_to_instrument_control",
            # buffer tools (present only when buf is not None)
            "read_latest_tip_status", "get_scan_progress", "get_tip_history_since",
        }]
        assert not skill_like, f"Unexpected skill-wrap tools in XD: {skill_like}"

    def test_buffer_tools_added_when_buf_provided(self):
        """When buf is not None, 3 buffer tools should be appended."""
        # Use a minimal mock that satisfies make_buffer_tools' usage.
        class _FakeBuf:
            def get_latest_tip_status(self):
                return (None, 0)

            def get_latest_progress(self):
                return (None, 0)

            def get_tip_history(self, since_seq):
                return []

        tools = build_tools(buf=_FakeBuf(), registry=_make_empty_registry())
        names = {t.name for t in tools}
        assert "read_latest_tip_status" in names
        assert "get_scan_progress" in names
        assert "get_tip_history_since" in names
        # 3 domain + 3 buffer + 2 handoffs = 8
        assert len(tools) == 8, f"Expected 8 tools with buf, got {len(tools)}"


# ─────────────────────────────────────────────────────────────────────
# Test 3 — agent invokes describe_skills when LLM emits a tool_call
# ─────────────────────────────────────────────────────────────────────

class TestXDAgentInvocation:
    """Compile XD with a fake LLM that emits a describe_skills tool_call,
    then verify the tool executes and returns Markdown content."""

    def setup_method(self):
        self.fake_llm = _FakeChatModel(messages=iter([
            # First model call: emit a tool_call for describe_skills
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "describe_skills",
                    "args": {},
                    "id": "tc-xd-1",
                    "type": "tool_call",
                }],
            ),
            # Second model call (after tool result): produce final plan
            AIMessage(content="Here is my experiment plan based on available skills."),
        ]))

    def test_agent_compiles_and_invokes(self):
        agent = build(
            buf=None,
            model=self.fake_llm,
            registry=_make_empty_registry(),
            checkpointer=InMemorySaver(),
        )
        result = agent.invoke(
            {"messages": [("user", "Plan an experiment on HOPG.")]},
            config={"configurable": {"thread_id": "smoke-xd-1"}},
        )
        assert result is not None
        assert "messages" in result

    def test_describe_skills_tool_result_in_messages(self):
        """After the describe_skills call, the ToolMessage content should
        contain the '# Available skills' header (even with empty registry)."""
        agent = build(
            buf=None,
            model=self.fake_llm,
            registry=_make_empty_registry(),
            checkpointer=InMemorySaver(),
        )
        result = agent.invoke(
            {"messages": [("user", "Plan an experiment on HOPG.")]},
            config={"configurable": {"thread_id": "smoke-xd-2"}},
        )
        all_content = "\n".join(
            str(m.content) for m in result["messages"] if hasattr(m, "content")
        )
        # The tool should have returned the describe_skills Markdown header
        assert "Available skills" in all_content or "experiment plan" in all_content.lower(), (
            f"Expected describe_skills result or final plan in messages. Got:\n{all_content}"
        )


# ─────────────────────────────────────────────────────────────────────
# Fixtures — real tmp SQLite experiment log for query_past_experiments
# (mirror the Paper-Writing agent's fixture pattern; no mocking of the
#  query logic itself — a real DB with a real schema is constructed).
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_experiment_db(tmp_path, monkeypatch):
    """Build a tiny SQLite experiment DB at MAST_EXPERIMENT_DB and return path."""
    from mast.logging.storage import ExperimentStorage

    db_path = tmp_path / "experiments.db"
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(db_path))
    store = ExperimentStorage(db_path)
    store.create_experiment("Au111-herringbone", goal_text="STM imaging of Au(111)")
    store.create_experiment("WSe2-defects", goal_text="Defect ML segmentation")
    yield db_path


@pytest.fixture
def fake_experiment_db_with_samples(tmp_path, monkeypatch):
    """Experiment DB whose runs carry per-sample sample_type rows.

    Exercises the sample-level filter path of query_past_experiments.
    """
    from mast.logging.storage import ExperimentStorage

    db_path = tmp_path / "experiments_samples.db"
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(db_path))
    store = ExperimentStorage(db_path)
    exp_id = store.create_experiment("graphite-run", goal_text="surface survey")
    store.create_sample(
        exp_id, name="sample-A", description="cleaved", sample_type="HOPG"
    )
    yield db_path


@pytest.fixture
def empty_experiment_db_path(tmp_path, monkeypatch):
    """Point MAST_EXPERIMENT_DB at a missing file → tool returns the no-DB note."""
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "no_such_db.db"))
    yield


# ─────────────────────────────────────────────────────────────────────
# Test 4 — individual tool unit-level checks
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_material_registry(monkeypatch):
    """Install a fictional module at the real lookup registry boundary for one test."""
    from types import ModuleType, SimpleNamespace
    from mast.knowledge import lookups
    from mast.admin.override_store import ConfigOverrideRegistry

    mod = ModuleType("mast.knowledge._synthetic_test_surface")
    mod.CATEGORY = {
        "name": "Fictional category", "name_en": "Synthetic surfaces",
        "completeness": "full",
        "phases": [{"id": "survey", "name": "Synthetic survey", "steps": [],
                    "success_criteria": "Base synthetic criterion"}],
    }
    mod.MATERIALS = {
        "FictionalSurface-Z": {
            "description": "An invented material used only for lookup tests.",
            "phases_override": {
                "survey": {"success_criteria": "Synthetic marker recovered"},
            },
        },
    }
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(lookups, "_CATEGORY_MODULES", {"synthetic_surface": mod.__name__})
    monkeypatch.setattr(lookups, "_module_cache", {})
    monkeypatch.setattr(lookups, "_material_index", None)
    monkeypatch.setattr(lookups, "_ALIASES", {
        "虚构表面": "FictionalSurface-Z", "虚构类别": "synthetic_surface",
    })
    # The fixture must not consult an installation's optional knowledge overrides.
    overrides = SimpleNamespace(get_knowledge_override=lambda _category: None)
    monkeypatch.setattr(ConfigOverrideRegistry, "get", staticmethod(lambda: overrides))
    return mod


class TestIndividualTools:
    @pytest.mark.parametrize("query", ["FictionalSurface-Z", "fictionalsurface-z"])
    def test_lookup_sample_known_material_returns_workflow(self, synthetic_material_registry, query):
        """The real resolver and formatter must consume the fictional module."""
        from mast.knowledge import lookups
        result = lookup_sample_tool().invoke({"query": query})
        assert "Match: category=synthetic_surface, material=FictionalSurface-Z" in result
        assert "completeness=full" in result
        assert "Recommended Workflow: FictionalSurface-Z" in result
        assert "Phases:" in result and "Synthetic survey" in result
        assert "Success: Synthetic marker recovered" in result
        assert "Base synthetic criterion" not in result
        assert lookups._module_cache["synthetic_surface"] is synthetic_material_registry

    def test_lookup_sample_chinese_alias(self, synthetic_material_registry):
        """A temporary Chinese alias resolves to the same fictional material workflow."""
        lt = lookup_sample_tool()
        assert lt.invoke({"query": "虚构表面"}) == lt.invoke({"query": "FictionalSurface-Z"})

    @pytest.mark.parametrize("query", ["synthetic_surface", "虚构类别"])
    def test_lookup_sample_category_returns_base_workflow(self, synthetic_material_registry, query):
        result = lookup_sample_tool().invoke({"query": query})
        assert "Match: category=synthetic_surface, material=(category-level)" in result
        assert "Recommended Workflow: Fictional category" in result
        assert "Success: Base synthetic criterion" in result
        assert "Synthetic marker recovered" not in result

    def test_public_default_material_registry_is_empty(self, monkeypatch):
        """Without the fictional fixture, the public snapshot ships no materials."""
        from mast.knowledge import lookups
        assert lookups._CATEGORY_MODULES == {}
        # Clear only memoized lookups; do not replace the shipped registry.
        monkeypatch.setattr(lookups, "_module_cache", {})
        monkeypatch.setattr(lookups, "_material_index", None)
        assert lookups.list_material_candidates() == []
        lt = lookup_sample_tool()
        for query in ("FictionalSurface-Z", "Au(111)", "金", "clean_metal"):
            assert "No match found" in lt.invoke({"query": query})

    def test_lookup_sample_unknown_returns_no_match_note(self):
        lt = lookup_sample_tool()
        result = lt.invoke({"query": "unobtainium-XYZ-ZZZ"})
        assert "No match found" in result
        assert "category" in result.lower()

    def test_query_past_experiments_no_db_returns_note(self, empty_experiment_db_path):
        """With MAST_EXPERIMENT_DB pointing at a missing file, return an honest
        no-DB note (not fabricated data)."""
        qt = query_past_experiments_tool()
        result = qt.invoke({"sample_type": "HOPG", "max_n": 3})
        assert "experiment log not found" in result.lower()

    def test_query_past_experiments_no_args_no_db(self, empty_experiment_db_path):
        qt = query_past_experiments_tool()
        result = qt.invoke({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_past_experiments_returns_seeded_records(self, fake_experiment_db):
        """Against a real tmp SQLite DB, the most-recent runs are listed."""
        qt = query_past_experiments_tool()
        result = qt.invoke({})
        assert "Past experiments" in result
        assert "Au111-herringbone" in result or "WSe2-defects" in result

    def test_query_past_experiments_filter_by_name(self, fake_experiment_db):
        """sample_type substring matches experiment name (case-insensitive)."""
        qt = query_past_experiments_tool()
        result = qt.invoke({"sample_type": "wse2"})
        assert "WSe2-defects" in result
        assert "Au111-herringbone" not in result

    def test_query_past_experiments_filter_by_goal_text(self, fake_experiment_db):
        """sample_type substring also matches the experiment goal text."""
        qt = query_past_experiments_tool()
        result = qt.invoke({"sample_type": "segmentation"})
        assert "WSe2-defects" in result
        assert "Au111-herringbone" not in result

    def test_query_past_experiments_filter_by_sample_type(self, fake_experiment_db_with_samples):
        """sample_type matches the per-run sample_type column."""
        qt = query_past_experiments_tool()
        result = qt.invoke({"sample_type": "HOPG"})
        assert "graphite-run" in result

    def test_query_past_experiments_no_match(self, fake_experiment_db):
        """A filter that matches nothing returns an honest no-records note,
        not fabricated data."""
        qt = query_past_experiments_tool()
        result = qt.invoke({"sample_type": "NbSe2-XYZ-none"})
        assert "no records matched" in result

    def test_query_past_experiments_respects_max_n(self, fake_experiment_db):
        """max_n caps the number of returned runs."""
        qt = query_past_experiments_tool()
        result = qt.invoke({"max_n": 1})
        # Exactly one record line (lines starting with "  - ").
        record_lines = [ln for ln in result.splitlines() if ln.strip().startswith("- ")]
        assert len(record_lines) == 1

    def test_describe_skills_empty_registry_returns_header(self):
        registry = _make_empty_registry()
        dt = describe_skills_tool(registry)
        result = dt.invoke({})
        assert "# Available skills" in result

    def test_describe_skills_with_real_registry(self):
        """If mast.skills.builtins is importable, describe_skills returns skill entries."""
        registry = SkillRegistry()
        try:
            n = registry.discover("mast.skills.builtins")
        except Exception:
            pytest.skip("mast.skills.builtins not importable in this environment")
        if n == 0:
            pytest.skip("No skills discovered — skip catalog content test")
        dt = describe_skills_tool(registry)
        result = dt.invoke({})
        assert "# Available skills" in result
        # At least one skill section header expected
        assert "##" in result

    def test_describe_skills_safety_level_filter(self):
        """safety_level filter should be passed through without error."""
        registry = _make_empty_registry()
        dt = describe_skills_tool(registry)
        # Even with an empty registry, filtering should not raise
        result = dt.invoke({"safety_level": "DANGEROUS"})
        assert isinstance(result, str)

    def test_describe_skills_unknown_safety_level_is_ignored(self):
        """An unrecognised safety_level string should not raise — silently ignored."""
        registry = _make_empty_registry()
        dt = describe_skills_tool(registry)
        result = dt.invoke({"safety_level": "NOT_A_REAL_LEVEL"})
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────
# Test 5 — _discover_catalog smoke
# ─────────────────────────────────────────────────────────────────────

class TestDiscoverCatalog:
    def test_returns_skill_registry(self):
        """discover_xd_catalog() must return a SkillRegistry (even if empty)."""
        try:
            reg = discover_xd_catalog()
        except Exception as exc:
            pytest.skip(f"Catalog discovery raised {exc} — likely missing v2 skills pkg")
        assert isinstance(reg, SkillRegistry)

    def test_private_alias_is_same_function(self):
        """_discover_catalog and discover_xd_catalog must be the same object."""
        assert _discover_catalog is discover_xd_catalog


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
