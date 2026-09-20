"""The supervisor can ask the operator instead of guessing.

Two things are being pinned, and the second matters more than the feature.

1. When the router says "too ambiguous to route" it hands a question to the
   ``ask_operator`` node, which pauses the graph; the answer comes back as a
   HumanMessage so the next routing pass sees it like any other operator input.
   The node is separate from ``supervisor_node`` on purpose: a resume REPLAYS
   its node from the top, and inside the supervisor that replay would re-run the
   LLM route call and might never reach the interrupt again — silently dropping
   the operator's answer.

2. NOTHING CHANGES WHEN THE ROUTER DOESN'T USE IT. ``clarify_*`` are optional in
   a schema six providers have to agree on, and the tiered router exists
   precisely because they disagree (F5, 2026-06-08). Every provider that ignores
   an optional field, and tier 3 which cannot express one at all, must route
   byte-for-byte as before.
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────
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

import pytest  # noqa: E402

from mast.agents.orchestrator.graph import (  # noqa: E402
    ParallelRoute,
    _ask_operator_node,
    _clarify_fields,
)


class TestClarifyFieldsAreOptional:
    def test_absent_clarify_produces_nothing_at_all(self):
        # The caller spreads this into its result, so {} == byte-identical
        # routing output for every router that ignores the new keys.
        assert _clarify_fields({"next_agents": ["literature"], "reason": "r"}) == {}
        assert _clarify_fields({"clarify_question": "   "}) == {}
        assert _clarify_fields({"clarify_options": ["a", "b"]}) == {}, \
            "options without a question is not a question"

    def test_question_and_options_are_normalised(self):
        out = _clarify_fields({"clarify_question": " 先扫形貌还是先测谱？ ",
                               "clarify_options": ["形貌", "  ", "能谱"]})
        assert out == {"clarify_question": "先扫形貌还是先测谱？",
                       "clarify_options": ["形貌", "能谱"]}

    def test_a_bare_string_option_is_tolerated(self):
        out = _clarify_fields({"clarify_question": "q?", "clarify_options": "只有一个"})
        assert out["clarify_options"] == ["只有一个"]

    def test_question_without_options_is_still_a_question(self):
        # Open-ended is legitimate — the card renders a text box.
        out = _clarify_fields({"clarify_question": "你想达成什么？"})
        assert out == {"clarify_question": "你想达成什么？", "clarify_options": []}

    def test_schema_keeps_the_pair_optional(self):
        # If these ever became required, every provider that omits them would
        # start failing structured output — the tier-1 path for all six.
        required = getattr(ParallelRoute, "__required_keys__", frozenset())
        assert "clarify_question" not in required
        assert "clarify_options" not in required
        assert {"next_agents", "reason"} <= set(required)


class TestAskOperatorNode:
    def test_empty_question_hands_control_straight_back(self):
        # A cleared / resumed-old-checkpoint state must not pause on a blank card.
        cmd = _ask_operator_node({"pending_user_question": None})
        assert cmd.goto == "supervisor"
        assert cmd.update["pending_user_question"] is None
        assert not cmd.update.get("messages")

    def test_interrupt_payload_is_an_ask_user_question(self, monkeypatch):
        seen = {}

        def _fake_interrupt(payload):
            seen["payload"] = payload
            return {"selected": ["先扫形貌"], "custom_text": "", "note": ""}

        import langgraph.types as lgt
        monkeypatch.setattr(lgt, "interrupt", _fake_interrupt)

        cmd = _ask_operator_node({"pending_user_question": {
            "question": "先扫形貌还是先测谱？",
            "options": ["先扫形貌", "先测 dI/dV"],
        }})

        p = seen["payload"]
        assert p["kind"] == "ask_user"
        assert p["question"] == "先扫形貌还是先测谱？"
        assert [o["label"] for o in p["options"]] == ["先扫形貌", "先测 dI/dV"]
        # The supervisor could not route in the first place, so it has no
        # fallback to continue with — silence must stop the run, not re-enter
        # the coin flip this node exists to avoid.
        assert p["timeout_action"] == "halt"
        # Self-reported owner: a parent-graph node's namespace is empty, and the
        # publisher would otherwise file this under instrument_control.
        assert p["agent_id"] == "_supervisor"

    def test_answer_returns_to_the_router_as_operator_input(self, monkeypatch):
        import langgraph.types as lgt
        monkeypatch.setattr(lgt, "interrupt", lambda _p: {
            "selected": ["先扫形貌"], "custom_text": "别超过 30 分钟", "note": "",
        })

        cmd = _ask_operator_node({"pending_user_question": {
            "question": "先扫形貌还是先测谱？", "options": ["先扫形貌", "先测 dI/dV"]}})

        assert cmd.goto == "supervisor"
        assert cmd.update["pending_user_question"] is None, "the question is consumed"
        msgs = cmd.update["messages"]
        assert len(msgs) == 1
        text = msgs[0].content
        assert "先扫形貌" in text and "别超过 30 分钟" in text
        # A HumanMessage, so _trim_for_routing carries it into the next routing
        # pass with no new channel to maintain.
        assert msgs[0].type == "human"

    def test_unavailable_channel_is_reported_not_faked(self, monkeypatch):
        import langgraph.types as lgt

        def _boom(_p):
            raise RuntimeError("no graph runtime")

        monkeypatch.setattr(lgt, "interrupt", _boom)
        cmd = _ask_operator_node({"pending_user_question": {"question": "q?", "options": []}})
        assert cmd.goto == "supervisor"
        assert "未获答复" in cmd.update["messages"][0].content

    def test_graph_interrupt_propagates(self, monkeypatch):
        # It is control flow. Swallowing it here would resume the graph with a
        # fabricated answer instead of pausing.
        import langgraph.types as lgt
        from langgraph.errors import GraphInterrupt

        def _raise(_p):
            raise GraphInterrupt(())

        monkeypatch.setattr(lgt, "interrupt", _raise)
        with pytest.raises(GraphInterrupt):
            _ask_operator_node({"pending_user_question": {"question": "q?", "options": []}})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestGoalQuestions:
    """目标闸门的两张卡：答案要落成**结构化结果**，不是指望模型第二次读对。

    普通澄清问题只需把答案放回消息流。目标闸门问的是结构化的东西（继续/结束、
    够了/还不够），所以答案也要结构化：``goal_hold`` 选中的「继续：派 X」直接
    变成一条 ``routing_hints`` —— 走既有的交接提示分发路径（已经过全部熔断），
    **不再花一次路由调用**。
    """

    def _ans(self, monkeypatch, selected):
        import langgraph.types as lgt

        monkeypatch.setattr(lgt, "interrupt",
                            lambda _p: {"selected": list(selected)})

    def test_choosing_continue_becomes_a_routing_hint(self, monkeypatch):
        self._ans(monkeypatch, ["继续：派 data_processing（产出分析结果）"])
        cmd = _ask_operator_node({"pending_user_question": {
            "question": "还差分析。继续还是结束？",
            "options": ["继续：派 data_processing（产出分析结果）", "就此结束（记录为未达成）"],
            "kind": "goal_hold",
            "routes": {"继续：派 data_processing（产出分析结果）": "data_processing"},
        }})
        assert cmd.update["routing_hints"] == ["data_processing"]
        assert cmd.goto == "supervisor"

    def test_choosing_end_adds_no_hint(self, monkeypatch):
        """选「就此结束」不加提示 —— 本 run 的 goal.asked 已置位，
        下一跳模型再说 __end__ 时 Gate 2 会大声放行。"""
        self._ans(monkeypatch, ["就此结束（记录为未达成）"])
        cmd = _ask_operator_node({"pending_user_question": {
            "question": "q", "options": [], "kind": "goal_hold",
            "routes": {"继续：派 data_processing（产出分析结果）": "data_processing"},
        }})
        assert "routing_hints" not in cmd.update

    def test_a_plain_clarify_answer_is_untouched(self, monkeypatch):
        """零回归：没有 kind 的老卡片，行为一个字都不变。"""
        self._ans(monkeypatch, ["随便"])
        cmd = _ask_operator_node({"pending_user_question": {
            "question": "q", "options": ["随便"]}})
        assert set(cmd.update) == {"messages", "pending_user_question",
                                   "active_agent"}

    def test_confirm_yes_is_recorded_structurally(self, monkeypatch):
        self._ans(monkeypatch, ["够了，就到这里"])
        cmd = _ask_operator_node({"pending_user_question": {
            "question": "够了吗？", "options": ["够了，就到这里", "还不够，继续"],
            "kind": "goal_confirm", "routes": {}}, "goal": {"text": "t"}})
        assert cmd.update["goal"]["operator_confirmed"]["answer"] == "yes"
        assert cmd.update["goal"]["text"] == "t", "把 goal 的其余字段冲掉了"

    def test_confirm_anything_other_than_yes_is_no(self, monkeypatch):
        """**读不到 / 没选 / 自定义文本一律记 no。**

        这是一个安全计数器方向的字段：把「没答」读成「答应了」会让一次沉默
        变成一次结束。
        """
        for selected in ([], ["还不够，继续"], ["我再想想"]):
            self._ans(monkeypatch, selected)
            cmd = _ask_operator_node({"pending_user_question": {
                "question": "够了吗？", "options": [], "kind": "goal_confirm",
                "routes": {}}})
            assert cmd.update["goal"]["operator_confirmed"]["answer"] == "no", \
                f"{selected} 被读成了「够了」"
