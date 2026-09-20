"""Paper Writing agent smoke tests — build the agent with a fake LLM, verify
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

from mast.agents.paper_writing.graph import build  # noqa: E402
from mast.agents.paper_writing.tools import (  # noqa: E402
    AGENT_TOOLS,
    build_tools,
    draft_section,
    embed_figure,
    lookup_citation,
    query_experiment_records,
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
            AIMessage(content="Draft complete."),
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
            AIMessage(content="Draft complete."),
        ]))
        agent = build(buf=None, model=fake_llm)
        assert agent is not None
        assert callable(getattr(agent, "invoke", None))


# ─────────────────────────────────────────────────────────────────────
# Test 2 — build_tools shape
# ─────────────────────────────────────────────────────────────────────

class TestBuildTools:
    def test_count_without_buf(self):
        """With buf=None: 9 domain tools + 2 handoffs = 11 tools."""
        tools = build_tools(buf=None)
        assert len(tools) == 11, (
            f"Expected 11 tools (9 domain + 2 handoffs), got {len(tools)}: "
            f"{[t.name for t in tools]}"
        )

    def test_count_with_buf(self):
        """With buf supplied: 9 domain + 3 buffer + 2 handoffs = 14 tools."""
        tools = build_tools(buf=_FakeBuf())
        assert len(tools) == 14, (
            f"Expected 14 tools (9 domain + 3 buffer + 2 handoffs), got {len(tools)}: "
            f"{[t.name for t in tools]}"
        )

    def test_domain_tool_names_present(self):
        """All six domain tool names must appear in build_tools output."""
        tools = build_tools(buf=None)
        names = {t.name for t in tools}
        expected_domain = {
            "query_experiment_records",
            "lookup_citation",
            "draft_section",
            # 2026-07-27: the read side of the figures artifact. Without it PW
            # could only embed a path it was handed in conversation.
            "list_figures",
            "embed_figure",
            "save_draft",
            # 2026-07-28: 单文件自包含 HTML 导出(图片 base64 内嵌)——
            # markdown 是工作稿,这个才是能发给别人的交付物。
            "export_report_html",
            "load_review",
        }
        missing = expected_domain - names
        assert not missing, f"Missing domain tools: {missing}"

    def test_handoff_names_present(self):
        """Both handoffs must be present."""
        tools = build_tools(buf=None)
        names = {t.name for t in tools}
        assert "handoff_to_supervisor" in names
        assert "handoff_to_paper_review" in names

    def test_buffer_tool_names_when_buf_provided(self):
        """When buf is provided, the three buffer tools must appear."""
        tools = build_tools(buf=_FakeBuf())
        names = {t.name for t in tools}
        assert "read_latest_tip_status" in names
        assert "get_scan_progress" in names
        assert "get_tip_history_since" in names

    def test_agent_tools_constant_has_nine_entries(self):
        """AGENT_TOOLS must list exactly 9 domain tools."""
        assert len(AGENT_TOOLS) == 9, (
            f"AGENT_TOOLS length mismatch: {[t.name for t in AGENT_TOOLS]}"
        )

    def test_agent_tools_names(self):
        """AGENT_TOOLS names must match the nine expected tools exactly."""
        names = {t.name for t in AGENT_TOOLS}
        expected = {
            "query_experiment_records",
            "lookup_citation",
            "draft_section",
            # 2026-07-27: the read side of the figures artifact. Without it PW
            # could only embed a path it was handed in conversation.
            "list_figures",
            "embed_figure",
            "save_draft",
            # 2026-07-28: 单文件自包含 HTML 导出(图片 base64 内嵌)——
            # markdown 是工作稿,这个才是能发给别人的交付物。
            "export_report_html",
            # 2026-07-29: docx 覆盖 HTML 覆盖不了的那半 —— 发给别人**改**
            # (导师/合作者走 Word 的修订与批注,期刊也不收 HTML 投稿)。
            # 全文挂 Word 内置样式,见 design §6.1。
            "export_report_docx",
            # load_review closes the revision loop: without a tool to OPEN the
            # reviewer's saved report, PW depended on the issue list surviving in
            # the handoff message text — which compaction can summarise away.
            "load_review",
        }
        assert names == expected, (
            f"AGENT_TOOLS mismatch — got {names}, expected {expected}"
        )


# ─────────────────────────────────────────────────────────────────────
# Test 3 — stub tools return expected canned strings
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_experiment_db(tmp_path, monkeypatch):
    """Build a tiny SQLite experiment DB at MAST_EXPERIMENT_DB and return the path."""
    import sys
    sys.path.insert(0, _MASTV2_ROOT)
    from mast.logging.storage import ExperimentStorage

    db_path = tmp_path / "experiments.db"
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(db_path))
    store = ExperimentStorage(db_path)
    store.create_experiment("Au111-herringbone", goal_text="STM imaging of Au(111)")
    store.create_experiment("WSe2-defects", goal_text="Defect ML segmentation")
    yield db_path


@pytest.fixture
def empty_experiment_db_path(tmp_path, monkeypatch):
    """Point to a missing DB so query_experiment_records returns the no-DB note."""
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "no_such_db.db"))
    yield


@pytest.fixture
def fake_figure_png(tmp_path, monkeypatch, documents_root):
    """Create a tiny PNG image and a figures dir to receive copies.

    ``documents_root`` 不是可选的（2026-07-29 架构审查在这里抓到一次**活的**泄漏）：
    ``MAST_FIGURES_DIR`` 改的是**源**目录，而 ``embed_figure`` 的**落点**从 2026-07-29
    起是实验内的 ``<exp>/reports/_assets/`` —— 它跟着 ``MAST_EXPERIMENT_ROOT`` 和 live
    ``active_scope`` 走。只设 FIGURES_DIR 的话，这条测试会把 ``topo.png`` 写进用户
    真实数据根里那个正好活跃着的实验（实测解析到
    ``D:\\MAST-Data\\experiments\\…__627c3dab\\reports\\_assets``）。

    这正是 conftest 记的那句「a store whose path comes from a DIFFERENT env var than
    the one the test set」，也正是 ⑨ 那条落点变更留下的尾巴：fixture 抄的前提
    「设了 FIGURES_DIR 就算隔离」在落点搬进实验文件夹的那一刻就失效了。
    """
    src = tmp_path / "src" / "topo.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    # Minimal 1x1 PNG (8-byte signature + IHDR + IDAT + IEND)
    src.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\xdacd\xf8\xcf\x00\x00\x00\x03\x00\x01\xb1\xc1\xa6\x91"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    figures_dir = tmp_path / "figures"
    monkeypatch.setenv("MAST_FIGURES_DIR", str(figures_dir))
    yield str(src)


class TestRealTools:
    """Real tool implementations against tmp_path SQLite + figure fixtures."""

    _FAKE_CAPTION = "STM image of Au(111) herringbone reconstruction at 77 K."

    # ── query_experiment_records ─────────────────────────────────────

    def test_query_returns_seeded_records(self, fake_experiment_db):
        result = query_experiment_records.invoke({})
        assert "Experiment records" in result
        assert "Au111-herringbone" in result or "WSe2-defects" in result

    def test_query_filter_by_id_substring(self, fake_experiment_db):
        result = query_experiment_records.invoke({"experiment_id": "WSe2"})
        assert "WSe2-defects" in result
        assert "Au111-herringbone" not in result

    def test_query_filter_no_match(self, fake_experiment_db):
        result = query_experiment_records.invoke({"experiment_id": "NbSe2"})
        assert "no records matched" in result

    def test_query_no_db_returns_note(self, empty_experiment_db_path):
        result = query_experiment_records.invoke({})
        assert "SQLite DB not found" in result

    def test_query_with_limit(self, fake_experiment_db):
        result = query_experiment_records.invoke({"limit": 1})
        # Header line + at most 1 record line
        assert "1/1" in result or "Experiment records (1" in result

    # ── lookup_citation ──────────────────────────────────────────────

    def test_lookup_citation_known_key_smalley(self):
        result = lookup_citation.invoke({"key": "smalley2024"})
        assert "Smalley" in result and "2024" in result

    def test_lookup_citation_known_key_krull(self):
        result = lookup_citation.invoke({"key": "krull2020"})
        assert "Krull" in result and "2020" in result

    def test_lookup_citation_year_present(self):
        result = lookup_citation.invoke({"key": "rashidi2018"})
        assert "2018" in result and "Rashidi" in result

    def test_lookup_citation_unknown_key(self):
        result = lookup_citation.invoke({"key": "made_up_key"})
        assert "not found" in result.lower()
        # Honest miss: lists the keys it DOES know and refuses to invent one.
        assert "curated" in result.lower()
        assert "smalley2024" in result
        assert "do not invent" in result.lower()

    # ── draft_section ────────────────────────────────────────────────

    def test_draft_section_methods_template(self):
        result = draft_section.invoke(
            {"section": "methods", "context": "Au(111) at 77 K, scan rate 2 Hz"}
        )
        assert "## Methods" in result
        assert "Au(111)" in result  # operator notes appended verbatim

    def test_draft_section_results_template(self):
        result = draft_section.invoke(
            {"section": "results", "context": "lattice constant 0.288 nm"}
        )
        assert "## Results" in result
        assert "lattice constant" in result

    def test_draft_section_unknown_section(self):
        result = draft_section.invoke(
            {"section": "appendix", "context": "x"}
        )
        assert "unknown section" in result.lower()

    def test_draft_section_all_five_sections_work(self):
        for section in ("introduction", "methods", "results", "discussion", "conclusion"):
            result = draft_section.invoke(
                {"section": section, "context": f"context for {section}"}
            )
            assert f"context for {section}" in result

    # ── embed_figure ─────────────────────────────────────────────────

    def test_embed_figure_copies_and_returns_markdown(self, fake_figure_png):
        result = embed_figure.invoke(
            {"scan_path": fake_figure_png, "caption": self._FAKE_CAPTION}
        )
        assert "![" in result and "](" in result  # markdown img syntax
        assert self._FAKE_CAPTION in result
        assert "topo.png" in result

    def test_embed_figure_missing_source(self):
        result = embed_figure.invoke(
            {"scan_path": "/nonexistent/scan_doesnt_exist.png", "caption": "x"}
        )
        assert "source file not found" in result.lower()


# ─────────────────────────────────────────────────────────────────────
# Test 4 — end-to-end agent invocation with fake LLM
# ─────────────────────────────────────────────────────────────────────

class TestAgentInvocation:
    """Compile PW with a fake LLM that emits a query_experiment_records tool_call,
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
                    "name": "query_experiment_records",
                    "args": {},
                    "id": "tc-pw-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Retrieved records. Starting draft."),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Write a paper from the latest experiments.")]},
            config={"configurable": {"thread_id": "smoke-pw-1"}},
        )
        assert result is not None
        assert "messages" in result

    def test_query_records_tool_result_in_messages(self, fake_experiment_db):
        """After query_experiment_records, the ToolMessage should contain exp names."""
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "query_experiment_records",
                    "args": {"limit": 5},
                    "id": "tc-pw-2",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="Records retrieved."),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Fetch experiment records.")]},
            config={"configurable": {"thread_id": "smoke-pw-2"}},
        )
        all_content = "\n".join(
            str(m.content) for m in result["messages"] if hasattr(m, "content")
        )
        assert (
            "Au111-herringbone" in all_content
            or "WSe2-defects" in all_content
            or "Experiment records" in all_content
        ), (
            f"Expected experiment names from query in messages. Got:\n{all_content}"
        )

    def test_final_ai_message_present(self):
        """Agent output must include at least one final AIMessage with text."""
        final_text = "Draft manuscript ready for review."
        agent = self._make_agent(iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "query_experiment_records",
                    "args": {},
                    "id": "tc-pw-3",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content=final_text),
        ]))
        result = agent.invoke(
            {"messages": [("user", "Write the paper.")]},
            config={"configurable": {"thread_id": "smoke-pw-3"}},
        )
        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        assert ai_messages, "No non-empty AIMessage found in result"
        all_ai_text = " ".join(m.content for m in ai_messages)
        assert final_text in all_ai_text or "Draft" in all_ai_text

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
