"""The questions asked of an agent — "关键是要设计好发送给他们的问题".

``docs/v2/design/wakeup_scheduling.md`` §3.3. The operator's brief was explicit that
the prompt text IS the design here, so these tests assert on the question's content,
not just on the plumbing. Five rules, each with a failure behind it:

1. facts only, rendered by the SAME renderer the agent will really receive (two
   wordings of the same world = decide from A, work from B);
2. name what is missing AND who produces it ("缺少一些信息" earns an equally vague
   answer back);
3. the ASYMMETRY OF FAILURE stated in the question — biasing toward "wait" yields a
   deadlock nobody sees, biasing toward "start" yields one visible imperfect
   artifact, so the prompt says starting is the default and every failure path in
   the module returns ``start``;
4. ``waiting_for`` is a CLOSED SET, enforced on OUR side so even a degraded parse
   cannot produce an unmatchable value (free text = a wait that never ends);
5. tell it the elapsed time, the decline count and the deadline — an agent that does
   not know it has declined three times cannot decide better than it did the first.
"""
from __future__ import annotations

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

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from mast.agents._shared import activation as act  # noqa: E402
from mast.agents._shared.artifact_channel import (  # noqa: E402
    WAITABLE_FIELDS,
    doc_ref,
    render_upstream_block,
)


# ── stubs ──────────────────────────────────────────────────────────────────

class _Answers:
    """A model that answers the activation question with a fixed dict."""

    def __init__(self, payload, *, fail_structured=False, text=""):
        self.payload = payload
        self.fail_structured = fail_structured
        self.text = text
        self.seen: list = []

    def with_structured_output(self, schema, method=None):
        outer = self

        class _S:
            def invoke(self, msgs):
                outer.seen.append(msgs)
                if outer.fail_structured:
                    raise RuntimeError("provider rejects function_calling")
                return dict(outer.payload)

        return _S()

    def invoke(self, msgs):
        self.seen.append(msgs)
        return AIMessage(content=self.text)


def _question_text(model) -> str:
    """The prompt actually sent, flattened."""
    assert model.seen, "no question was sent"
    return "\n".join(str(m.get("content", "")) for m in model.seen[0])


# ════════════════════════════════════════════════════════════════════
# Rule 1 — the same renderer, so the decision and the work see one world
# ════════════════════════════════════════════════════════════════════

class TestSameRendererAsTheRealBlock:
    def test_the_have_block_is_the_agents_real_context_block(self):
        state = {"draft": doc_ref(doc_id="draft__abc", version=2, title="稿件")}
        q = act.build_start_question(
            agent="paper_review", instruction="评审", state=state,
            missing_soft=["analysis"], available_tools={"load_document"})
        real = render_upstream_block(state, "paper_review",
                                     available_tools={"load_document"})
        assert real, "precondition: the renderer produced nothing to compare"
        # every line of the real block appears verbatim in the question
        for line in real.splitlines():
            if line.strip():
                assert line in q, f"the question paraphrases the context block: {line!r}"

    def test_an_empty_environment_is_stated_as_a_confirmed_empty(self):
        q = act.build_start_question(agent="paper_writing", instruction="写",
                                     state={}, missing_soft=["analysis"])
        assert "确认过的空" in q, \
            "an empty environment must be distinguishable from a failed lookup"
        # and no invented examples
        for fake in ("rpt__", "draft__", "示例", "例如"):
            assert fake not in q


# ════════════════════════════════════════════════════════════════════
# Rule 2 — name the gap and its producer
# ════════════════════════════════════════════════════════════════════

class TestNamesWhatIsMissingAndWho:
    def test_missing_fields_are_named_with_their_producer(self):
        q = act.build_start_question(agent="paper_writing", instruction="写报告",
                                     state={}, missing_soft=["analysis"])
        assert "analysis" in q
        assert "data_processing" in q, "the producer of the gap must be named"

    def test_unknown_is_presented_separately_from_missing(self):
        """An unreadable store is not an absent product. Folding them together would
        have the agent decide on a false premise."""
        q = act.build_start_question(agent="paper_writing", instruction="写",
                                     state={}, missing_soft=["analysis"],
                                     unknown=["literature_report"])
        assert "查不到" in q and "不等于" in q
        assert q.index("还没有的") < q.index("查不到"), \
            "the two sections must be distinct, in a fixed order"


# ════════════════════════════════════════════════════════════════════
# Rule 3 — the asymmetry of failure is IN the question
# ════════════════════════════════════════════════════════════════════

class TestAsymmetryOfFailureIsStated:
    def test_the_question_says_starting_is_the_default(self):
        """The dividing line between a scheduler and a silent death channel."""
        q = act.build_start_question(agent="paper_writing", instruction="写",
                                     state={}, missing_soft=["analysis"])
        assert "默认是现在就开工" in q

    def test_the_question_says_when_waiting_is_justified(self):
        q = act.build_start_question(agent="paper_writing", instruction="写",
                                     state={}, missing_soft=["analysis"])
        assert "变成猜测" in q, \
            "without a criterion, 'wait if you need to' is an invitation to wait"

    def test_no_model_defaults_to_start(self):
        d = act.ask_should_start(None, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis"])
        assert d["action"] == "start"

    def test_a_dead_provider_defaults_to_start(self):
        class _Dead:
            def with_structured_output(self, *a, **k):
                raise RuntimeError("down")

            def invoke(self, *a, **k):
                raise RuntimeError("down")

        d = act.ask_should_start(_Dead(), agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis"])
        assert d["action"] == "start"
        assert "defaulted to start" in d["reason"]

    def test_an_unparseable_answer_defaults_to_start(self):
        m = _Answers({}, fail_structured=True, text="hmm, hard to say either way")
        d = act.ask_should_start(m, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis"])
        assert d["action"] == "start"

    def test_an_off_enum_action_defaults_to_start(self):
        m = _Answers({"action": "postpone", "waiting_for": [], "reason": "x"},
                     text="not json")
        d = act.ask_should_start(m, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis"])
        assert d["action"] == "start"

    def test_nothing_missing_costs_no_model_call(self):
        m = _Answers({"action": "wait", "waiting_for": ["analysis"], "reason": "x"})
        d = act.ask_should_start(m, agent="literature", instruction="查",
                                 state={}, missing_soft=[], unknown=[])
        assert d["action"] == "start" and d["parse_path"] == "skipped"
        assert not m.seen, "a question was asked when there was nothing to decide"


# ════════════════════════════════════════════════════════════════════
# Rule 4 — the closed set, enforced on our side
# ════════════════════════════════════════════════════════════════════

class TestWaitingForIsAClosedSet:
    def test_the_closed_set_is_listed_in_the_question(self):
        q = act.build_start_question(agent="paper_writing", instruction="写",
                                     state={}, missing_soft=["analysis"])
        for f in WAITABLE_FIELDS:
            assert f in q, f"{f} is waitable but never offered to the model"

    def test_the_question_explains_why_free_text_is_refused(self):
        q = act.build_start_question(agent="paper_writing", instruction="写",
                                     state={}, missing_soft=["analysis"])
        assert "永远不会被叫醒" in q, \
            "the model must be told WHY the set is closed, or it will improvise"

    def test_an_out_of_set_answer_is_discarded(self):
        """Enforced HERE rather than trusted from the model: an unmatchable
        waiting_for is a park that can never wake — the worst outcome available."""
        m = _Answers({"action": "wait", "waiting_for": ["更多的好数据"],
                      "reason": "需要更多数据"})
        d = act.ask_should_start(m, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis"])
        assert d["action"] == "wait"
        assert d["waiting_for"] == ["analysis"], \
            "free text survived into waiting_for, or the fallback was not applied"

    def test_a_partly_valid_answer_keeps_only_the_valid_names(self):
        m = _Answers({"action": "wait", "waiting_for": ["analysis", "银弹"],
                      "reason": "x"})
        d = act.ask_should_start(m, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis", "last_scan"])
        assert d["waiting_for"] == ["analysis"]

    def test_an_empty_answer_falls_back_to_what_it_was_told_was_missing(self):
        """"Wait for nothing in particular" is not a wait that can end."""
        m = _Answers({"action": "wait", "waiting_for": [], "reason": "x"})
        d = act.ask_should_start(m, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis", "last_scan"])
        assert d["waiting_for"] == ["analysis", "last_scan"]

    def test_a_string_instead_of_a_list_is_accepted(self):
        m = _Answers({"action": "wait", "waiting_for": "analysis", "reason": "x"})
        d = act.ask_should_start(m, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis"])
        assert d["waiting_for"] == ["analysis"]

    def test_starting_never_carries_a_waiting_for(self):
        m = _Answers({"action": "start", "waiting_for": ["analysis"], "reason": "x"})
        d = act.ask_should_start(m, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis"])
        assert d["waiting_for"] == []

    def test_the_json_hint_lists_only_closed_set_names(self):
        """The degraded text tier is the one most likely to improvise, so its hint
        has to carry the set too."""
        for f in WAITABLE_FIELDS:
            assert f in act._JSON_HINT
        assert '"action"' in act._JSON_HINT


# ════════════════════════════════════════════════════════════════════
# Rule 5 — the agent is told its own history
# ════════════════════════════════════════════════════════════════════

class TestWakeQuestionTellsItItsHistory:
    def _q(self, **kw):
        base = dict(agent="paper_review", waiting_for=["draft"], state={},
                    arrived=["draft"], waited_human="3 小时", declines=2,
                    deadline_human="还有 5 小时")
        base.update(kw)
        return act.build_wake_question(**base)

    def test_it_states_how_long_it_has_waited(self):
        assert "3 小时" in self._q()

    def test_it_states_how_many_times_it_declined(self):
        q = self._q()
        assert "2 次" in q

    def test_it_states_the_deadline_and_what_happens_at_it(self):
        q = self._q()
        assert "还有 5 小时" in q
        assert "浮到用户面前" in q
        assert "不会自动继续" in q, \
            "the model must know the timeout escalates rather than silently resuming"

    def test_it_names_what_just_arrived(self):
        q = self._q(arrived=["analysis"])
        assert "analysis" in q and "data_processing" in q

    def test_it_leans_toward_waking_when_the_wait_is_satisfied(self):
        q = self._q()
        assert "已经到位了" in q and "应当醒" in q

    def test_it_offers_the_same_closed_set(self):
        q = self._q()
        for f in WAITABLE_FIELDS:
            assert f in q

    def test_ask_should_wake_returns_the_same_shape(self):
        m = _Answers({"action": "start", "waiting_for": [], "reason": "醒"})
        d = act.ask_should_wake(m, agent="paper_review", waiting_for=["draft"],
                                state={}, arrived=["draft"], waited_human="1 小时",
                                declines=0, deadline_human="还有 23 小时")
        assert set(d) == {"action", "waiting_for", "reason", "parse_path"}
        assert d["action"] == "start"

    def test_declining_again_re_reports_a_closed_set_target(self):
        m = _Answers({"action": "wait", "waiting_for": ["随便什么"], "reason": "再等"})
        d = act.ask_should_wake(m, agent="paper_review", waiting_for=["draft"],
                                state={}, arrived=["draft"], waited_human="1 小时",
                                declines=1, deadline_human="还有 23 小时")
        assert d["waiting_for"] == ["draft"]


class TestHumanizeAge:
    @pytest.mark.parametrize("secs,needle", [
        (5, "不到 1 分钟"), (600, "分钟"), (7200, "小时"), (200000, "天"),
    ])
    def test_readable_durations(self, secs, needle):
        assert needle in act.humanize_age(secs)

    def test_negative_and_none_do_not_crash(self):
        assert act.humanize_age(-5)
        assert act.humanize_age(None)


class TestEngineChoice:
    def test_it_uses_the_shared_hardened_router_not_structured_output_directly(self):
        """``with_structured_output`` is NOT provider-portable across MAST's six
        providers — ``cognition_llm.py`` says so and ``orchestrator/graph.py`` still
        records the six distinct failure modes that taught it. The hardened
        two-tier decider must be the only path."""
        # AST, not a substring scan: the module docstring MENTIONS the name to
        # record why it is avoided, and that prose is wanted. Only a real call is a
        # defect. (Same technique, same reason, as the dead-constant check in
        # tests/v2/agents/orchestrator/test_artifact_channel.py.)
        import ast
        import inspect

        src = inspect.getsource(act)
        assert "route_decision" in src
        calls = [n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Attribute)
                 and n.attr == "with_structured_output"]
        assert not calls, (
            "activation.py references with_structured_output as code at line(s) "
            f"{[n.lineno for n in calls]} — it is not provider-portable across "
            "MAST's 6 providers; go through llm_route.route_decision")

    def test_the_text_tier_can_still_decide(self):
        """When function_calling is rejected the tolerant text parse must still
        produce a decision, otherwise five of six providers silently default to
        start and the whole mechanism is inert for them."""
        m = _Answers({}, fail_structured=True,
                     text='{"action": "wait", "waiting_for": ["analysis"], '
                          '"reason": "缺分析"}')
        d = act.ask_should_start(m, agent="paper_writing", instruction="写",
                                 state={}, missing_soft=["analysis"])
        assert d["action"] == "wait"
        assert d["waiting_for"] == ["analysis"]
        assert d["parse_path"] in ("text_json", "call_re", "bare_name")
