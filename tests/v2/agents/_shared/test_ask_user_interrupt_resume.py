"""End-to-end ask_user interrupt/resume on a REAL compiled agent graph.

This is the falsification point for the whole feature. The pure-function tests
next door can all pass while the tool is completely dead in a running graph —
that is not hypothetical: the artifact channel shipped in 2026-07 with tools
whose ``Command(update=…)`` was SILENTLY DISCARDED, and its tests were green
because they called ``tool.func()`` directly, a path that never touches ToolNode
or the graph. So everything here drives ``agent.invoke`` on a graph compiled with
a checkpointer, and asserts on what the AGENT ends up seeing.

What is pinned:
  * calling ask_user PAUSES the graph with ``kind == "ask_user"`` and the tool
    has NOT returned anything to the model yet;
  * the payload carries the structured question (options / multi_select /
    allow_custom / timeout_action) the choice card needs;
  * ``Command(resume=<answer dict>)`` — the SINGLE-VALUE shape, not the
    ``{"decisions": [...]}`` envelope the DANGEROUS gate uses — feeds the answer
    back and the tool result the model reads contains the operator's choice;
  * a continue-timeout resume tells the agent nobody answered instead of
    looking like a real answer;
  * a rejected payload (empty question) never interrupts at all.
"""
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

from typing import Any  # noqa: E402

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from mast.agents._shared.ask_tools import ASK_USER_TOOLS  # noqa: E402
from mast.agents.literature.graph import build as build_literature  # noqa: E402


class _FakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # Remember what the agent was actually given — the "is the tool even on
        # this agent" question is worth asserting directly.
        self._bound_tool_names = [getattr(t, "name", "") for t in (tools or [])]
        return self


def _ask_llm(args: dict[str, Any]):
    return _FakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "ask_user", "args": args, "id": "tc-ask-1", "type": "tool_call",
        }]),
        AIMessage(content="收到，按用户的选择继续。"),
    ]))


def _build(llm):
    """A real literature agent with only the shared ask tool attached.

    literature (not instrument_control) on purpose: it is the plainest of the
    six builds, so nothing here can pass because of an IC-specific middleware.
    ``ask_user`` reaches every agent through the same shared list.
    """
    return build_literature(
        buf=None, model=llm, checkpointer=InMemorySaver(),
        extra_tools=list(ASK_USER_TOOLS), standalone=True,
    )


_Q = {"question": "接下来先扫哪个区域？若无人回答我会默认扫 B 区。",
      "options": [{"label": "A 区", "description": "缺陷密集，针尖风险高"},
                  {"label": "B 区", "description": "平坦台面，适合 STS 基线"}],
      "header": "区域选择"}


def _tool_texts(result: dict) -> str:
    return "\n".join(
        str(getattr(m, "content", "")) for m in (result.get("messages") or [])
        if getattr(m, "type", "") == "tool")


class TestAskUserPauses:
    def test_the_tool_is_actually_on_the_agent(self):
        llm = _ask_llm(_Q)
        agent = _build(llm)
        agent.invoke({"messages": [("user", "开始")]},
                     config={"configurable": {"thread_id": "ask-wired"}})
        assert "ask_user" in getattr(llm, "_bound_tool_names", [])

    def test_graph_pauses_with_ask_user_payload(self):
        agent = _build(_ask_llm(_Q))
        cfg = {"configurable": {"thread_id": "ask-pause"}}
        result = agent.invoke({"messages": [("user", "开始")]}, config=cfg)

        assert agent.get_state(cfg).next, "graph should be paused awaiting an answer"
        assert result.get("__interrupt__"), "invoke should surface __interrupt__"
        val = result["__interrupt__"][0].value
        assert val["kind"] == "ask_user"
        assert val["question"].startswith("接下来先扫哪个区域？")
        assert [o["label"] for o in val["options"]] == ["A 区", "B 区"]
        assert val["options"][0]["description"]      # descriptions survive
        assert val["multi_select"] is False
        assert val["allow_custom"] is True
        assert val["timeout_action"] == "continue"
        assert val["header"] == "区域选择"
        # And the model has NOT been handed an answer yet.
        assert "[用户回答]" not in _tool_texts(result)

    def test_declared_halt_survives_into_the_payload(self):
        # The API layer reads timeout_action off this payload to decide whether a
        # 900 s silence continues or stops the run — if it did not survive the
        # trip, every question would silently become fail-open.
        agent = _build(_ask_llm({**_Q, "timeout_action": "halt"}))
        result = agent.invoke({"messages": [("user", "开始")]},
                              config={"configurable": {"thread_id": "ask-halt"}})
        assert result["__interrupt__"][0].value["timeout_action"] == "halt"

    def test_bad_arguments_never_reach_the_operator(self):
        # Empty question → the tool returns a correction; the graph must run to
        # completion with no interrupt at all.
        agent = _build(_ask_llm({"question": "  "}))
        cfg = {"configurable": {"thread_id": "ask-badargs"}}
        result = agent.invoke({"messages": [("user", "开始")]}, config=cfg)
        assert not result.get("__interrupt__")
        assert not agent.get_state(cfg).next
        assert "question" in _tool_texts(result)


class TestResumeCarriesTheAnswer:
    def test_single_choice_answer_reaches_the_model(self):
        agent = _build(_ask_llm(_Q))
        cfg = {"configurable": {"thread_id": "ask-answer"}}
        agent.invoke({"messages": [("user", "开始")]}, config=cfg)
        assert agent.get_state(cfg).next

        # THE SINGLE-VALUE SHAPE. ask_user resumes like workflow_human, not like
        # the DANGEROUS gate's {"decisions": [...]} envelope.
        result = agent.invoke(
            Command(resume={"selected": ["B 区"], "custom_text": "", "note": ""}),
            config=cfg)

        texts = _tool_texts(result)
        assert "[用户回答]" in texts
        assert "B 区" in texts
        assert not agent.get_state(cfg).next, "graph should have resumed to the end"

    def test_multi_choice_custom_and_note_all_survive(self):
        agent = _build(_ask_llm({**_Q, "multi_select": True}))
        cfg = {"configurable": {"thread_id": "ask-multi"}}
        agent.invoke({"messages": [("user", "开始")]}, config=cfg)
        result = agent.invoke(
            Command(resume={"selected": ["A 区", "B 区"],
                            "custom_text": "先 A 后 B",
                            "note": "别超过 30 分钟"}),
            config=cfg)
        texts = _tool_texts(result)
        assert "A 区、B 区" in texts
        assert "先 A 后 B" in texts
        assert "别超过 30 分钟" in texts

    def test_open_question_custom_only(self):
        agent = _build(_ask_llm({"question": "这批数据你想怎么处理？"}))
        cfg = {"configurable": {"thread_id": "ask-open"}}
        result0 = agent.invoke({"messages": [("user", "开始")]}, config=cfg)
        val = result0["__interrupt__"][0].value
        assert val["options"] == [] and val["allow_custom"] is True

        result = agent.invoke(
            Command(resume={"selected": [], "custom_text": "只留 3 K 那几张",
                            "note": ""}),
            config=cfg)
        assert "只留 3 K 那几张" in _tool_texts(result)

    def test_continue_timeout_reads_as_unanswered_not_as_a_choice(self):
        agent = _build(_ask_llm(_Q))
        cfg = {"configurable": {"thread_id": "ask-timeout"}}
        agent.invoke({"messages": [("user", "开始")]}, config=cfg)

        result = agent.invoke(
            Command(resume={"selected": [], "custom_text": "", "timeout": True,
                            "note": "提问超时(900s)未收到用户回答"}),
            config=cfg)
        texts = _tool_texts(result)
        assert "未应答" in texts
        assert "保守默认" in texts
        assert "选择：" not in texts, "a timeout must not look like a decision"
