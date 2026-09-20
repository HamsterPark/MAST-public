"""Dead-tool detection for the Literature agent.

A tool whose hard dependency is missing is dead for the whole session. In the
2026-07-19/20 field runs web_search (no TAVILY_API_KEY) failed 10× and
search_papers (empty PDF corpus) 9×, because nothing told the agent the tool was
permanently unavailable — it kept trying, one wasted turn at a time.

The fix, exercised here:
  * probe dependency-gated tools ONCE at build time (`tool_availability`);
  * swap each dead tool for a same-named placeholder that fail-fasts with a
    single "known constraint — do not retry" line (name + count preserved so the
    prompt stays valid);
  * inject a one-time constraints note into the system prompt so the agent is
    told up front (`unavailable_tools_note`).

Run:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/agents/literature/test_dead_tools.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
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

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from mast.agents.literature import tools as littools  # noqa: E402
from mast.agents.literature.graph import build  # noqa: E402
from mast.agents.literature.tools import (  # noqa: E402
    AGENT_TOOLS,
    build_tools,
    tool_availability,
    unavailable_tools_note,
)


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.fixture
def all_dead(monkeypatch):
    """Force BOTH dependency-gated tools unavailable, deterministically."""
    monkeypatch.setattr(littools, "_read_tavily_key", lambda: None)
    monkeypatch.setattr(littools, "_all_pdfs", lambda: [])
    yield


@pytest.fixture
def all_live(monkeypatch):
    """Force BOTH dependency-gated tools available."""
    monkeypatch.setattr(littools, "_read_tavily_key", lambda: "dummy-key")
    monkeypatch.setattr(littools, "_all_pdfs", lambda: [Path("paper.pdf")])
    yield


# ── the probe ────────────────────────────────────────────────────────────────

class TestProbe:
    def test_reports_both_dead_when_deps_missing(self, all_dead):
        avail = tool_availability()
        assert set(avail) == {"web_search", "search_papers"}
        assert "TAVILY_API_KEY" in avail["web_search"]
        assert "PDF" in avail["search_papers"]

    def test_reports_nothing_when_deps_present(self, all_live):
        assert tool_availability() == {}

    def test_web_search_alone_when_only_key_missing(self, monkeypatch):
        monkeypatch.setattr(littools, "_read_tavily_key", lambda: None)
        monkeypatch.setattr(littools, "_all_pdfs", lambda: [Path("paper.pdf")])
        assert set(tool_availability()) == {"web_search"}


# ── the swap ───────────────────────────────────────────────────────────────

class TestBuildToolsSwap:
    def test_dead_tools_are_swapped_but_name_and_count_survive(self, all_dead):
        tools = build_tools(buf=None)
        # count unchanged (26) — placeholders keep the slot
        assert len(tools) == 26, [t.name for t in tools]
        names = {t.name for t in tools}
        assert {"web_search", "search_papers"} <= names

    def test_placeholder_fail_fasts_with_a_do_not_retry_line(self, all_dead):
        tools = {t.name: t for t in build_tools(buf=None)}
        out = tools["web_search"].invoke({"query": "NiI2 Au(111) STM"})
        assert "不可用" in out
        assert "已知约束" in out and ("请不要重复调用" in out or "不要反复调用" in out)
        # it must steer to a tool that actually works
        assert "search_local_corpus" in out or "lib_search" in out

    def test_placeholder_description_announces_unavailable(self, all_dead):
        tools = {t.name: t for t in build_tools(buf=None)}
        assert "不可用" in (tools["web_search"].description or "")

    def test_live_tools_are_the_real_ones(self, all_live):
        real = {t.name: t for t in AGENT_TOOLS}
        built = {t.name: t for t in build_tools(buf=None)}
        # same object identity → not swapped
        assert built["web_search"] is real["web_search"]
        assert built["search_papers"] is real["search_papers"]

    def test_offline_tools_are_never_gated(self, all_dead):
        names = {t.name for t in build_tools(buf=None)}
        # the corpus/lib tools have no external dependency and must stay callable
        assert {"search_local_corpus", "lib_search", "literature_priors"} <= names


# ── the one-time note ─────────────────────────────────────────────────────────

class TestConstraintsNote:
    def test_note_lists_the_dead_tools(self, all_dead):
        note = unavailable_tools_note()
        assert "web_search" in note and "search_papers" in note
        assert "请勿调用" in note or "不要反复调用" in note
        assert "search_local_corpus" in note  # points at the working alternative

    def test_note_is_empty_when_all_live(self, all_live):
        assert unavailable_tools_note() == ""

    def test_build_injects_the_note_into_the_system_prompt(self, all_dead, monkeypatch):
        """The agent must be TOLD up front — the note has to reach create_agent's
        system_prompt, not just exist as a helper. Capture the argument directly."""
        import mast.agents.literature.graph as litgraph

        captured: dict = {}
        real_create = litgraph.create_agent

        def _spy(*args, **kwargs):
            captured["system_prompt"] = kwargs.get("system_prompt", "")
            return real_create(*args, **kwargs)

        monkeypatch.setattr(litgraph, "create_agent", _spy)
        build(
            buf=None,
            model=_FakeChatModel(messages=iter([AIMessage(content="done")])),
            checkpointer=InMemorySaver(),
        )
        sp = captured["system_prompt"]
        assert "当前不可用的工具" in sp, "the constraints note never reached the prompt"
        assert "web_search" in sp and "search_papers" in sp


# ── build still compiles / runs with dead tools ──────────────────────────────

class TestBuildWithDeadTools:
    def test_build_compiles_when_tools_are_dead(self, all_dead):
        agent = build(
            buf=None,
            model=_FakeChatModel(messages=iter([AIMessage(content="ok")])),
            checkpointer=InMemorySaver(),
        )
        result = agent.invoke(
            {"messages": [("user", "查 NiI2 文献")]},
            config={"configurable": {"thread_id": "dead-1"}},
        )
        assert result is not None and "messages" in result

    def test_calling_a_dead_tool_returns_the_note_not_an_error(self, all_dead):
        """End-to-end: the model calls web_search, gets the fail-fast note back as
        a ToolMessage (not an exception, not an unknown-tool error)."""
        agent = build(
            buf=None,
            model=_FakeChatModel(messages=iter([
                AIMessage(content="", tool_calls=[{
                    "name": "web_search",
                    "args": {"query": "NiI2 Au(111)"},
                    "id": "tc-dead-1", "type": "tool_call"}]),
                AIMessage(content="改用本地库。"),
            ])),
            checkpointer=InMemorySaver(),
        )
        result = agent.invoke(
            {"messages": [("user", "上网查 NiI2")]},
            config={"configurable": {"thread_id": "dead-2"}},
        )
        blob = "\n".join(str(m.content) for m in result["messages"]
                         if hasattr(m, "content"))
        assert "不可用" in blob and "已知约束" in blob


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
