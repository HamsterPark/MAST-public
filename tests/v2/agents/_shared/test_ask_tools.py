"""Pure-function contract for the ``ask_user`` structured-question tool.

The payload builder is the ONLY thing standing between a weak model's arguments
and a card the operator sees, so its tolerance (str options, over-long lists,
empty option sets) and its refusals (empty question) are pinned here. The answer
formatter is pinned separately because that string is the agent's entire view of
what the operator said — an unparseable resume must not read like a real answer.

The interrupt itself is exercised on a REAL compiled graph in
``test_ask_user_interrupt_resume.py``; this file deliberately stays pure.
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

from mast.agents._shared.ask_tools import (  # noqa: E402
    ASK_USER_TOOLS,
    _MAX_OPTIONS,
    _build_payload,
    _format_answer,
    _normalize_options,
    ask_user,
)


def _payload(**kw):
    base = dict(question="先扫哪个区域？", options=[{"label": "A"}, {"label": "B"}],
                multi_select=False, allow_custom=True, header="", timeout_action="continue")
    base.update(kw)
    return _build_payload(base["question"], base["options"], base["multi_select"],
                          base["allow_custom"], base["header"], base["timeout_action"])


class TestNormalizeOptions:
    def test_plain_strings_become_labelled_options(self):
        # Weak models send ["A","B"] at least as often as the documented shape.
        assert _normalize_options(["A 区", "B 区"]) == [
            {"label": "A 区", "description": ""},
            {"label": "B 区", "description": ""},
        ]

    def test_dicts_keep_description_and_accept_value_alias(self):
        out = _normalize_options([
            {"label": "A", "description": "针尖风险高"},
            {"value": "B", "detail": "平坦"},
        ])
        assert out == [{"label": "A", "description": "针尖风险高"},
                       {"label": "B", "description": "平坦"}]

    def test_junk_and_duplicates_are_dropped_not_raised(self):
        out = _normalize_options(["A", "A", "", None, 42, {"description": "no label"}])
        assert out == [{"label": "A", "description": ""}]

    def test_none_is_an_empty_list(self):
        assert _normalize_options(None) == []


class TestBuildPayload:
    def test_empty_question_is_rejected_before_any_interrupt(self):
        # A string return is handed back to the model as the tool result, so the
        # call is corrected without the operator ever seeing a blank card.
        out = _payload(question="   ")
        assert isinstance(out, str)
        assert "question" in out

    def test_happy_path_shape(self):
        p = _payload(header="区域选择")
        assert p["kind"] == "ask_user"
        assert p["question"] == "先扫哪个区域？"
        assert p["header"] == "区域选择"
        assert [o["label"] for o in p["options"]] == ["A", "B"]
        assert p["multi_select"] is False
        assert p["allow_custom"] is True
        assert p["timeout_action"] == "continue"
        assert p["agent_id"] == ""

    def test_too_many_options_are_truncated_AND_said_so(self):
        # Silent truncation would read to the operator as "these are all the
        # choices" — the drop has to be visible in the question itself.
        p = _payload(options=[f"opt{i}" for i in range(_MAX_OPTIONS + 3)])
        assert len(p["options"]) == _MAX_OPTIONS
        assert "未列出" in p["question"]
        assert "3" in p["question"]

    def test_empty_options_force_allow_custom(self):
        # An open question with no text box is unanswerable.
        p = _payload(options=[], allow_custom=False)
        assert p["options"] == []
        assert p["allow_custom"] is True

    def test_multi_select_needs_options(self):
        p = _payload(options=[], multi_select=True)
        assert p["multi_select"] is False

    def test_timeout_action_normalises_unknown_to_continue(self):
        assert _payload(timeout_action="halt")["timeout_action"] == "halt"
        assert _payload(timeout_action="HALT")["timeout_action"] == "halt"
        assert _payload(timeout_action="explode")["timeout_action"] == "continue"
        assert _payload(timeout_action="")["timeout_action"] == "continue"

    def test_header_is_bounded(self):
        p = _payload(header="x" * 100)
        assert len(p["header"]) <= 24


class TestFormatAnswer:
    def test_single_choice(self):
        p = _payload()
        out = _format_answer(p, {"selected": ["B"], "custom_text": "", "note": ""})
        assert "[用户回答]" in out
        assert "先扫哪个区域？" in out
        assert "选择：B" in out

    def test_multi_choice_and_custom_and_note(self):
        p = _payload(multi_select=True)
        out = _format_answer(p, {"selected": ["A", "B"], "custom_text": "先 A 后 B",
                                 "note": "别超过 30 分钟"})
        assert "A、B" in out
        assert "先 A 后 B" in out
        assert "别超过 30 分钟" in out

    def test_open_question_answer(self):
        p = _payload(options=[])
        out = _format_answer(p, {"selected": [], "custom_text": "扫左上角",
                                 "note": ""})
        assert "回答：扫左上角" in out

    def test_timeout_tells_the_agent_to_use_its_stated_default(self):
        p = _payload()
        out = _format_answer(p, {"selected": [], "custom_text": "", "timeout": True,
                                 "note": "提问超时(900s)未收到用户回答"})
        assert "未应答" in out
        assert "保守默认" in out
        # And it must NOT read like the operator chose something.
        assert "选择：" not in out

    def test_malformed_resume_is_not_dressed_up_as_an_answer(self):
        p = _payload()
        out = _format_answer(p, "approve")
        assert "无法解析" in out
        assert "未获答复" in out


class TestToolBody:
    def test_bad_args_return_the_correction_without_interrupting(self):
        # .func bypasses the graph entirely; an empty question must be refused
        # here rather than reaching interrupt().
        out = ask_user.func(question="")
        assert isinstance(out, str) and "question" in out

    def test_outside_a_graph_runtime_it_degrades_honestly(self):
        # No LangGraph runtime → interrupt() raises. The tool must say the
        # channel is unavailable and point at the async alternative, NOT raise
        # and NOT pretend it got an answer.
        out = ask_user.func(question="先扫哪个区域？", options=["A", "B"])
        assert "request_user_action" in out
        assert "[用户回答]" not in out

    def test_export_list(self):
        assert ASK_USER_TOOLS == [ask_user]
        assert ask_user.name == "ask_user"
