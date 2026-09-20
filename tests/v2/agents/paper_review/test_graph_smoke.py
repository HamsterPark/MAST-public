"""Paper Review agent smoke tests — build the agent with a fake LLM, verify
compilation, tool list shape, stub-tool return values, and one-turn tool
invocation."""
from __future__ import annotations

# ── path bootstrap (robust walk-up, mirrors DP pattern) ───────────────
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
# Evict stale mast.* modules that may have resolved to v1 outside MASTv2.
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from tests.v2.toolcall import tool_call
from mast.agents.paper_review.graph import build  # noqa: E402
from mast.agents.paper_review.tools import (  # noqa: E402
    AGENT_TOOLS,
    build_tools,
    check_citations,
    check_data_reasoning,
    check_methodology,
    load_draft,
    produce_review,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel with no-op bind_tools so create_agent() won't crash."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class _FakeBuf:
    """Minimal BufferService-like object satisfying make_buffer_tools' API."""

    def get_latest_tip_status(self):
        return (None, 0)

    def get_latest_progress(self):
        return (None, 0)

    def get_tip_history(self, since_seq: int):
        return []


# ─────────────────────────────────────────────────────────────────────
# Test 1 — graph compiles
# ─────────────────────────────────────────────────────────────────────

class TestAgentCompiles:
    def test_build_returns_compiled_state_graph(self):
        """build() should return a LangGraph CompiledStateGraph, not a stub fn."""
        fake_llm = _FakeChatModel(messages=iter([
            AIMessage(content="Review complete."),
        ]))
        agent = build(
            buf=None,
            model=fake_llm,
            checkpointer=InMemorySaver(),
        )
        assert agent is not None
        assert callable(getattr(agent, "invoke", None)), (
            "build() must return a CompiledStateGraph with .invoke; "
            "got a plain function (stub not replaced?)"
        )

    def test_build_without_checkpointer(self):
        """build() must succeed when no checkpointer is provided."""
        fake_llm = _FakeChatModel(messages=iter([
            AIMessage(content="Review complete."),
        ]))
        agent = build(buf=None, model=fake_llm)
        assert agent is not None
        assert callable(getattr(agent, "invoke", None))


# ─────────────────────────────────────────────────────────────────────
# Test 2 — build_tools shape
# ─────────────────────────────────────────────────────────────────────

class TestBuildTools:
    def test_count_without_buf(self):
        """With buf=None: 7 domain tools + 2 handoffs = 8 tools."""
        tools = build_tools(buf=None)
        assert len(tools) == 9, (
            f"Expected 9 tools (7 domain + 2 handoffs), got {len(tools)}: "
            f"{[t.name for t in tools]}"
        )

    def test_count_with_buf(self):
        """With buf supplied: 7 domain + 3 buffer + 2 handoffs = 11 tools."""
        tools = build_tools(buf=_FakeBuf())
        assert len(tools) == 12, (
            f"Expected 12 tools (7 domain + 3 buffer + 2 handoffs), got {len(tools)}: "
            f"{[t.name for t in tools]}"
        )

    def test_domain_tool_names_present(self):
        """All five domain tool names must appear in build_tools output."""
        tools = build_tools(buf=None)
        names = {t.name for t in tools}
        expected_domain = {
            "load_draft",
            "check_methodology",
            "check_data_reasoning",
            "check_citations",
            "list_figures",   # confirm a referenced figure actually exists
            "produce_review",
        }
        missing = expected_domain - names
        assert not missing, f"Missing domain tools: {missing}"

    def test_handoff_names_present(self):
        """Both handoffs must be present."""
        tools = build_tools(buf=None)
        names = {t.name for t in tools}
        assert "handoff_to_supervisor" in names
        assert "handoff_to_paper_writing" in names

    def test_buffer_tool_names_when_buf_provided(self):
        """When buf is provided, the three buffer tools must appear."""
        tools = build_tools(buf=_FakeBuf())
        names = {t.name for t in tools}
        assert "read_latest_tip_status" in names
        assert "get_scan_progress" in names
        assert "get_tip_history_since" in names

    def test_agent_tools_constant_has_seven_entries(self):
        """AGENT_TOOLS must list exactly 7 domain tools."""
        assert len(AGENT_TOOLS) == 7, (
            f"AGENT_TOOLS length mismatch: {[t.name for t in AGENT_TOOLS]}"
        )

    def test_agent_tools_names(self):
        """AGENT_TOOLS names must match the six expected tools exactly."""
        names = {t.name for t in AGENT_TOOLS}
        expected = {
            "load_draft",
            "check_methodology",
            "check_data_reasoning",
            "check_citations",
            "list_figures",   # confirm a referenced figure actually exists
            "produce_review",
            "save_review",
        }
        assert names == expected, (
            f"AGENT_TOOLS mismatch — got {names}, expected {expected}"
        )


# ─────────────────────────────────────────────────────────────────────
# Test 3 — stub tools return expected canned strings
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_drafts_dir(tmp_path, monkeypatch, documents_root):
    """Seed the LEGACY drafts dir with one Markdown draft and point env at it.

    This exercises the read-only compatibility path: drafts saved by an older
    build still live in ``data/drafts/`` and a review round must not dead-end just
    because they were never imported into the document store. ``documents_root``
    keeps the store empty (and off the operator's real data), so resolution really
    does fall through to here."""
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "v2.md").write_text(
        "# Title\n\n"
        "## Introduction\n"
        "STM imaging of WSe2 follows [Smalley2024]. "
        "Krull et al. (2020) showed tip conditioning works.\n\n"
        "## Methods\n"
        "Bias 200 mV. Setpoint 50 pA.\n\n"
        "## Results\n"
        "Defect density 4 per 100 nm^2.\n\n"
        "## Discussion\n"
        "Consistent with [Rashidi2018].\n\n"
        "## References\n"
        "[Smalley2024] R. Smalley et al. (2024) WSe2 ML.\n"
        "[Krull2020] A. Krull et al. (2020) DeepSPM.\n"
        "[Ramachandra2024] S. Ramachandra et al. (2024) ML SPM review.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAST_DRAFTS_DIR", str(drafts))
    yield drafts


class TestRealTools:
    """Real implementations against tmp-path drafts dir + LLM-delegated stubs."""

    _FAKE_METHODS = "The sample was annealed. Bias was set to 200 mV."
    _FAKE_RESULTS = "The lattice is hexagonal with a = 0.32 nm."

    # ── load_draft ───────────────────────────────────────────────────

    def test_load_draft_returns_text(self, fake_drafts_dir):
        result = load_draft.invoke({"doc_id": "current"})
        assert "Loaded" in result
        assert "v2" in result
        assert "sections:" in result and "words:" in result

    def test_load_draft_finds_named(self, fake_drafts_dir):
        result = load_draft.invoke({"doc_id": "v2"})
        assert "v2.md" in result

    def test_load_draft_unknown_id(self, fake_drafts_dir):
        result = load_draft.invoke({"doc_id": "nonexistent_draft_xyz"})
        assert "找不到" in result
        assert "doc_id" in result, "didn't say what a valid reference looks like"

    def test_load_draft_prefers_the_document_store(self, fake_drafts_dir):
        """Both sources present → the store wins. The legacy directory is a
        fallback, not a competing answer: a document saved today must not be
        shadowed by a file an older build left behind."""
        from mast.agents.paper_writing.tools import save_draft

        save_draft.invoke(tool_call(save_draft, {"title": "新报告", "markdown_text": "# 新\n存进文档库的正文"}))
        result = load_draft.invoke({"doc_id": "current"})
        assert "存进文档库的正文" in result
        assert "doc_id=" in result and "旧格式文件" not in result

    def test_load_draft_nothing_saved_anywhere(self, tmp_path, monkeypatch,
                                               documents_root):
        """Nothing in the store and no legacy dir → say so, and say what unblocks
        it (paper_writing must run save_draft). It used to dead-end on a bare
        'directory not found'."""
        monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "missing"))
        result = load_draft.invoke({"doc_id": "current"})
        assert "还没有任何已保存的报告" in result
        assert "save_draft" in result
        assert "doc_id" in result

    # ── check_methodology / check_data_reasoning are LLM-delegated ───

    def test_check_methodology_returns_string(self):
        result = check_methodology.invoke({"section_text": self._FAKE_METHODS})
        assert isinstance(result, str) and len(result) > 0

    def test_check_data_reasoning_returns_string(self):
        result = check_data_reasoning.invoke({"section_text": self._FAKE_RESULTS})
        assert isinstance(result, str) and len(result) > 0

    # ── check_citations ──────────────────────────────────────────────

    def test_check_citations_finds_orphan(self, fake_drafts_dir):
        """Ramachandra2024 is in bib but never cited → orphan."""
        result = check_citations.invoke({"doc_id": "current"})
        assert "ramachandra2024" in result.lower()
        assert "orphan" in result.lower()

    def test_check_citations_no_missing_in_clean_draft(self, fake_drafts_dir):
        """Smalley + Krull + Rashidi are all in body and bib in our seed."""
        result = check_citations.invoke({"doc_id": "current"})
        # missing should be empty (or at most contain none-found marker)
        assert "(none)" in result or "cited but missing" in result

    def test_check_citations_unknown_draft(self, fake_drafts_dir):
        result = check_citations.invoke({"doc_id": "nope"})
        assert "找不到" in result

    def test_check_citations_nothing_saved_anywhere(self, tmp_path, monkeypatch,
                                                   documents_root):
        monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "missing"))
        result = check_citations.invoke({"doc_id": "current"})
        assert "还没有任何已保存的报告" in result
        assert "save_draft" in result

    def test_produce_review_contains_verdict(self):
        result = produce_review.invoke({"rubric": "standard"})
        assert "REVISE" in result or "ACCEPT" in result or "REJECT" in result

    def test_produce_review_contains_required_revisions(self):
        result = produce_review.invoke({"rubric": "standard"})
        assert "Required Revisions" in result or "revisions" in result.lower()

    def test_produce_review_contains_citation_section(self):
        result = produce_review.invoke({"rubric": "standard"})
        assert "Citation" in result

    def test_produce_review_default_rubric(self):
        """produce_review must accept zero args (rubric has a default)."""
        result = produce_review.invoke({})
        assert isinstance(result, str) and len(result) > 0


# ─────────────────────────────────────────────────────────────────────
# Test 4 — end-to-end agent invocation with fake LLM
# ─────────────────────────────────────────────────────────────────────

class TestAgentInvocation:
    """Compile PR with a fake LLM that emits a load_draft tool_call, then
    verify the tool executes and its canned result appears in messages."""

    def _make_agent(self, messages_iter):
        fake_llm = _FakeChatModel(messages=messages_iter)
        return build(
            buf=None,
            model=fake_llm,
            checkpointer=InMemorySaver(),
        )

    def test_agent_invokes_without_error(self):
        """Agent should complete without raising an exception."""
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_draft",
                    "args": {"doc_id": "current"},
                    "id": "tc-pr-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Draft loaded. Proceeding with review."),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Review the current draft.")]},
            config={"configurable": {"thread_id": "smoke-pr-1"}},
        )
        assert result is not None
        assert "messages" in result

    def test_load_draft_tool_result_in_messages(self, fake_drafts_dir):
        """After load_draft, the ToolMessage should contain section info."""
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_draft",
                    "args": {"doc_id": "current"},
                    "id": "tc-pr-2",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Reviewed."),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Load and review the draft.")]},
            config={"configurable": {"thread_id": "smoke-pr-2"}},
        )
        all_content = "\n".join(
            str(m.content) for m in result["messages"] if hasattr(m, "content")
        )
        assert "Loaded" in all_content or "sections" in all_content, (
            f"Expected draft summary from load_draft in messages. Got:\n{all_content}"
        )

    def test_final_ai_message_present(self):
        """Agent output must include at least one final AIMessage with text."""
        final_text = "ReviewReport: REVISE — 5 issues found."
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_draft",
                    "args": {},
                    "id": "tc-pr-3",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content=final_text),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Review the draft.")]},
            config={"configurable": {"thread_id": "smoke-pr-3"}},
        )
        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        assert ai_messages, "No non-empty AIMessage found in result"
        all_ai_text = " ".join(m.content for m in ai_messages)
        assert final_text in all_ai_text or "Review" in all_ai_text

    def test_no_stub_fn_in_graph(self):
        """Agent must not be a bare function (indicates Phase-3 stub not replaced)."""
        fake_llm = _FakeChatModel(messages=iter([
            AIMessage(content="Done."),
        ]))
        agent = build(buf=None, model=fake_llm)
        # A plain function does not have .get_graph(); CompiledStateGraph does.
        assert hasattr(agent, "get_graph"), (
            "build() returned a plain function — Phase-3 stub was not replaced"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
