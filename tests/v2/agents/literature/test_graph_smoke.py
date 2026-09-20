"""Literature agent smoke tests — build the agent with a fake LLM, verify
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

import os  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from mast.agents.literature.graph import build  # noqa: E402
from mast.agents.literature.tools import (  # noqa: E402
    AGENT_TOOLS,
    build_tools,
    extract_protocol,
    read_paper_section,
    search_papers,
    web_search,
)

#: The agent's full tool surface. Kept as named constants so a change to the
#: inventory shows up as a diff of NAMES rather than a bare number nobody can
#: check: 6 domain + 15 library + 3 buffer (or no-op stand-ins) + 2 handoffs.
_EXPECTED_AGENT_TOOLS = {
    "search_papers",
    "read_paper_section",
    "extract_protocol",
    "web_search",
    "search_local_corpus",   # OpenAlex pipeline
    "fetch_fulltext_oa",     # self-service open-access retrieval
}
_EXPECTED_TOOL_COUNT = 26


@pytest.fixture
def empty_corpus(tmp_path, monkeypatch):
    """Force search_papers to look at an empty tmp dir (deterministic).

    Also re-points MAST_PAPERS_DIR: the corpus scan always includes the canonical
    ingest destination, so pinning only MAST_PAPER_CORPUS would leave whatever
    the autouse literature fixture set in the search path.
    """
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(tmp_path))
    monkeypatch.setenv("MAST_PAPERS_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture
def synthetic_corpus(tmp_path, monkeypatch):
    """Generate a tiny PyMuPDF PDF with searchable text and section headings."""
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "smalley2024.pdf"
    doc = fitz.open()
    page = doc.new_page()
    body = (
        "Abstract\n"
        "We characterise WSe2 surface defects by STM.\n"
        "\n"
        "1. Introduction\n"
        "Tip conditioning is a critical step in STM imaging.\n"
        "\n"
        "2. Methods\n"
        "Bias voltage was set to 200 mV. Setpoint current was 50 pA.\n"
        "Scan rate of 2 Hz was used. Samples were annealed at 600 °C for 1 h.\n"
        "\n"
        "3. Results\n"
        "Defect density was 4 per 100 nm^2.\n"
        "\n"
        "4. Discussion\n"
        "These findings suggest selective trap states.\n"
    )
    page.insert_text((50, 60), body, fontsize=10)
    doc.save(pdf_path)
    doc.close()
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(tmp_path))
    yield tmp_path


@pytest.fixture
def no_tavily_key(monkeypatch):
    """Strip the Tavily key so web_search returns the configuration note."""
    monkeypatch.setenv("TAVILY_API_KEY", "")
    # Also point key file lookup at an empty dir so it can't find tavily.env
    monkeypatch.setenv("MAST_PAPER_CORPUS", "")
    yield


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
            AIMessage(content="Literature review complete."),
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
            AIMessage(content="Literature review complete."),
        ]))
        agent = build(buf=None, model=fake_llm)
        assert agent is not None
        assert callable(getattr(agent, "invoke", None))


# ─────────────────────────────────────────────────────────────────────
# Test 2 — build_tools shape
# ─────────────────────────────────────────────────────────────────────

class TestBuildTools:
    def test_count_without_buf(self):
        """With buf=None: 6 domain + 13 library + 3 no-op buffer + 2 handoffs = 24.

        (4 PDF tools + search_local_corpus from the OpenAlex pipeline +
        fetch_fulltext_oa for self-service open-access retrieval; the 13 P1
        library-curation + read-only knowledge + fetch-request-board tools are
        always attached. The 3 buffer tools are bound as no-op stand-ins even
        when buf=None — same NAMES as make_buffer_tools — so the prompt never
        advertises a tool that isn't callable; total matches the with-buf count.)
        """
        tools = build_tools(buf=None)
        assert len(tools) == _EXPECTED_TOOL_COUNT, (
            f"Expected {_EXPECTED_TOOL_COUNT} tools (6 domain + 15 library + 3 "
            f"no-op buffer + 2 handoffs), got {len(tools)}: {[t.name for t in tools]}"
        )
        # the no-op stand-ins carry the same names as the real buffer tools
        names = {t.name for t in tools}
        assert {"read_latest_tip_status", "get_scan_progress",
                "get_tip_history_since"} <= names

    def test_count_with_buf(self):
        """With buf supplied: 6 domain + 13 library + 3 buffer + 2 handoffs = 24.

        13 library tools, not 11: lib_copy and save_literature_report arrived
        with experiment-scoped libraries (2026-07-29).
        """
        tools = build_tools(buf=_FakeBuf())
        assert len(tools) == _EXPECTED_TOOL_COUNT, (
            f"Expected {_EXPECTED_TOOL_COUNT} tools (6 domain + 15 library + 3 "
            f"buffer + 2 handoffs), got {len(tools)}: {[t.name for t in tools]}"
        )

    def test_domain_tool_names_present(self):
        """All four domain tool names must appear in build_tools output."""
        tools = build_tools(buf=None)
        names = {t.name for t in tools}
        expected_domain = {
            "search_papers",
            "read_paper_section",
            "extract_protocol",
            "web_search",
        }
        missing = expected_domain - names
        assert not missing, f"Missing domain tools: {missing}"

    def test_handoff_names_present(self):
        """Both handoffs must be present."""
        tools = build_tools(buf=None)
        names = {t.name for t in tools}
        assert "handoff_to_supervisor" in names
        assert "handoff_to_experiment_design" in names

    def test_buffer_tool_names_when_buf_provided(self):
        """When buf is provided, the three buffer tools must appear."""
        tools = build_tools(buf=_FakeBuf())
        names = {t.name for t in tools}
        assert "read_latest_tip_status" in names
        assert "get_scan_progress" in names
        assert "get_tip_history_since" in names

    def test_agent_tools_constant_matches_the_name_list(self):
        assert len(AGENT_TOOLS) == len(_EXPECTED_AGENT_TOOLS), (
            f"AGENT_TOOLS length mismatch: {[t.name for t in AGENT_TOOLS]}"
        )

    def test_agent_tools_names(self):
        """AGENT_TOOLS names must match exactly.

        Membership here is not cosmetic: build_tools() only swaps AGENT_TOOLS
        entries for "unavailable" placeholders, so a dependency-gated tool put in
        LIBRARY_TOOLS by mistake would fail once per call instead of announcing
        itself dead up front.
        """
        assert {t.name for t in AGENT_TOOLS} == _EXPECTED_AGENT_TOOLS


# ─────────────────────────────────────────────────────────────────────
# Test 3 — stub tools return expected canned strings
# ─────────────────────────────────────────────────────────────────────

class TestRealTools:
    """Invoke each real tool against a synthetic PyMuPDF corpus + check
    Tavily wiring (skip web_search live call if no key configured)."""

    # ── search_papers ────────────────────────────────────────────────

    def test_search_papers_no_corpus_returns_note(self, empty_corpus):
        result = search_papers.invoke({"query": "WSe2 defects"})
        assert "no PDFs" in result or "no matches" in result.lower()

    def test_search_papers_finds_synthetic_pdf(self, synthetic_corpus):
        result = search_papers.invoke({"query": "WSe2 defects"})
        assert "smalley2024" in result
        assert "score=" in result and "first match" in result.lower()

    def test_search_papers_no_match_in_corpus(self, synthetic_corpus):
        """Query terms that are absent from the synthetic PDF."""
        result = search_papers.invoke({"query": "kagome zzzzz qqqqq"})
        assert "no matches" in result.lower()

    def test_search_papers_default_max_results(self, synthetic_corpus):
        """search_papers must accept query alone (max_results has a default)."""
        result = search_papers.invoke({"query": "WSe2"})
        assert isinstance(result, str) and len(result) > 0

    def test_search_papers_explicit_max_results(self, synthetic_corpus):
        result = search_papers.invoke({"query": "STM", "max_results": 3})
        assert isinstance(result, str) and len(result) > 0

    # ── read_paper_section ───────────────────────────────────────────

    def test_read_paper_section_returns_methods_text(self, synthetic_corpus):
        result = read_paper_section.invoke(
            {"paper_id": "smalley2024", "section": "methods"}
        )
        assert "Methods" in result
        assert "200 mV" in result or "50 pA" in result

    def test_read_paper_section_returns_results_text(self, synthetic_corpus):
        result = read_paper_section.invoke(
            {"paper_id": "smalley2024", "section": "results"}
        )
        # Heading detection should at least include "Results"
        assert "Results" in result.title() or "results" in result.lower()

    def test_read_paper_section_unknown_paper(self, synthetic_corpus):
        result = read_paper_section.invoke(
            {"paper_id": "nonexistent_paper", "section": "methods"}
        )
        assert "not found" in result.lower()

    def test_read_paper_section_unknown_section(self, synthetic_corpus):
        result = read_paper_section.invoke(
            {"paper_id": "smalley2024", "section": "literature_review"}
        )
        assert "unknown section" in result.lower()

    # ── extract_protocol ─────────────────────────────────────────────

    def test_extract_protocol_finds_bias(self, synthetic_corpus):
        result = extract_protocol.invoke({"paper_id": "smalley2024"})
        assert "bias" in result.lower()
        assert "200" in result and ("mV" in result or "v" in result.lower())

    def test_extract_protocol_finds_setpoint(self, synthetic_corpus):
        result = extract_protocol.invoke({"paper_id": "smalley2024"})
        assert "setpoint" in result.lower() or "50" in result

    def test_extract_protocol_finds_scan_rate(self, synthetic_corpus):
        result = extract_protocol.invoke({"paper_id": "smalley2024"})
        assert "scan_rate" in result or "Hz" in result

    def test_extract_protocol_unknown_paper(self, synthetic_corpus):
        result = extract_protocol.invoke({"paper_id": "nonexistent_paper"})
        assert "not found" in result.lower()

    # ── web_search ───────────────────────────────────────────────────

    def test_web_search_returns_string(self):
        result = web_search.invoke({"query": "STM tip conditioning"})
        assert isinstance(result, str) and len(result) > 0

    def test_web_search_response_is_well_formed(self):
        """Either Tavily is configured (live result) or we return a config note."""
        result = web_search.invoke({"query": "WSe2 STM defect 2024"})
        is_live = "Tavily results" in result or "results for" in result
        is_unconfigured = "TAVILY_API_KEY not configured" in result
        is_no_results = "no Tavily results" in result
        assert is_live or is_unconfigured or is_no_results, (
            f"web_search returned unexpected shape: {result[:300]}"
        )


# ─────────────────────────────────────────────────────────────────────
# Test 4 — end-to-end agent invocation with fake LLM
# ─────────────────────────────────────────────────────────────────────

class TestAgentInvocation:
    """Compile Literature with a fake LLM that emits a search_papers tool_call,
    then verify the tool executes and its canned result appears in messages."""

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
                    "name": "search_papers",
                    "args": {"query": "WSe2 defects STM"},
                    "id": "tc-lit-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Found relevant papers. Proceeding with extraction."),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Find papers on WSe2 defects.")]},
            config={"configurable": {"thread_id": "smoke-lit-1"}},
        )
        assert result is not None
        assert "messages" in result

    def test_search_papers_tool_result_in_messages(self, synthetic_corpus):
        """After search_papers, the ToolMessage content should contain a paper id."""
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_papers",
                    "args": {"query": "tip conditioning"},
                    "id": "tc-lit-2",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Reviewed paper list."),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Search for tip conditioning papers.")]},
            config={"configurable": {"thread_id": "smoke-lit-2"}},
        )
        all_content = "\n".join(
            str(m.content) for m in result["messages"] if hasattr(m, "content")
        )
        assert "smalley2024" in all_content or "score=" in all_content, (
            f"Expected paper id from search_papers in messages. Got:\n{all_content}"
        )

    def test_final_ai_message_present(self):
        """Agent output must include at least one final AIMessage with text."""
        final_text = "Prior art summarised. Ready for experiment design."
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_papers",
                    "args": {"query": "WSe2"},
                    "id": "tc-lit-3",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content=final_text),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Review literature on WSe2.")]},
            config={"configurable": {"thread_id": "smoke-lit-3"}},
        )
        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        assert ai_messages, "No non-empty AIMessage found in result"
        all_ai_text = " ".join(m.content for m in ai_messages)
        assert final_text in all_ai_text or "Prior" in all_ai_text

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


# ─────────────────────────────────────────────────────────────────────
# Test 5 — turn_recorder wiring (RFC #8 P2): RecorderMiddleware captures
# each agent turn end-to-end when the GUI injects a recorder.
# ─────────────────────────────────────────────────────────────────────

class TestTurnRecorderWiring:
    def test_turn_recorder_captures_each_turn(self):
        """build(turn_recorder=...) must append RecorderMiddleware so each model
        call emits one agent_turn payload (reasoning + tool_calls + content)."""
        captured: list[dict] = []
        agent = build(
            buf=None,
            model=_FakeChatModel(messages=iter([
                AIMessage(
                    content="",
                    additional_kwargs={"reasoning_content": "I should search first."},
                    tool_calls=[{"name": "search_papers",
                                 "args": {"query": "WSe2"},
                                 "id": "tc-rec-1", "type": "tool_call"}],
                ),
                AIMessage(content="Summary done."),
            ])),
            checkpointer=InMemorySaver(),
            turn_recorder=captured.append,
        )
        agent.invoke(
            {"messages": [("user", "Review WSe2 literature.")]},
            config={"configurable": {"thread_id": "smoke-rec-1"}},
        )
        # Two model calls → two agent_turn captures.
        assert len(captured) == 2
        assert all(c["agent_id"] == "literature" for c in captured)
        assert captured[0]["reasoning"] == "I should search first."
        assert captured[0]["tool_calls"][0]["name"] == "search_papers"
        assert captured[1]["content"] == "Summary done."

    def test_no_turn_recorder_is_noop(self):
        """build() without turn_recorder leaves the agent untouched (no error)."""
        agent = build(
            buf=None,
            model=_FakeChatModel(messages=iter([AIMessage(content="done")])),
            checkpointer=InMemorySaver(),
        )
        result = agent.invoke(
            {"messages": [("user", "hi")]},
            config={"configurable": {"thread_id": "smoke-rec-2"}},
        )
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
