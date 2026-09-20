"""HITL verdict translation — the logic that was missing for ~5 weeks.

``_build_decision`` was lost when the Gradio
layer was deleted (7aa1996) and never migrated to ``CoreRuntime``, so every
approval degraded and no human could approve anything — DANGEROUS skill gates,
composite workflow human nodes, or ``buffer_hitl`` for CRITICAL hardware events
alike. These tests pin the restored behaviour, with emphasis on the two
safety-carrying parts that are easy to drop silently:

* ``coerce_arg_value`` — without it, an edited ``bias_v="50"`` stays a STRING and
  ``check_global_bounds`` (which only inspects int/float) skips it entirely. The
  post-edit safety re-check would be silently disabled.
* ``enforce_allowed`` — without it, an operator can send ``edit`` to an
  interrupt that only advertises ``approve`` (e.g. EmergencyRetract) and rewrite
  its parameters. A real bypass, not a theoretical one.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> Path:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return p / "MASTv2"
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != str(_ROOT):
    while str(_ROOT) in sys.path:
        sys.path.remove(str(_ROOT))
    sys.path.insert(0, str(_ROOT))
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.core.hitl_decision import (  # noqa: E402
    build_ask_answer,
    build_decision,
    build_workflow_route,
    canonical_verdict,
    coerce_arg_value,
    enforce_allowed,
)


# ── verdict normalisation ────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expect", [
    ("approve", "approve"), ("accept", "approve"), ("approved", "approve"),
    ("reject", "reject"), ("deny", "reject"), ("rejected", "reject"),
    ("edit", "edit"), ("edit_approve", "edit"), ("edited", "edit"),
    ("  APPROVE  ", "approve"), ("", ""), (None, ""),
])
def test_canonical_verdict(raw, expect):
    assert canonical_verdict(raw) == expect


# ── basic translation ────────────────────────────────────────────────────────
def test_approve():
    assert build_decision("approve", "SetBias", {}, None, "") == {"type": "approve"}


def test_reject_carries_operator_reason():
    d = build_decision("reject", "SetBias", {}, None, "tip looks bad")
    assert d == {"type": "reject", "message": "tip looks bad"}


def test_reject_without_reason_still_has_a_message():
    d = build_decision("reject", "SetBias", {}, None, "")
    assert d["type"] == "reject" and d["message"]


def test_unknown_verdict_fails_safe_to_reject():
    """A typo, or a future UI sending a new verb, must never auto-run
    something dangerous."""
    d = build_decision("banana", "TipPulse", {"pulse_v": 5.0}, None, "")
    assert d["type"] == "reject"
    assert "banana" in d["message"]


# ── edit: the arg-type restoration that keeps SafetyGate effective ───────────
def test_edit_merges_wholesale_not_a_patch():
    """HITL replaces ToolCall.args WHOLESALE — an edit touching one field must
    still carry the untouched ones, or they silently revert to skill defaults."""
    d = build_decision("edit", "ConfigureScan",
                       {"width_m": 1e-7, "height_m": 1e-7, "angle_deg": 0.0},
                       {"width_m": "2e-7"}, "")
    args = d["edited_action"]["args"]
    assert args == {"width_m": 2e-7, "height_m": 1e-7, "angle_deg": 0.0}
    assert d["edited_action"]["name"] == "ConfigureScan"


@pytest.mark.parametrize("original,raw,expect", [
    (1.0, "50", 50.0),          # float stays float — SafetyGate sees a number
    (1.0, "1e-7", 1e-7),        # scientific notation survives
    (5, "12", 12),              # int stays int
    (True, "false", False),     # bool from checkbox text
    (True, "yes", True),
    (None, "42", 42),           # no original to mirror → numeric parse
    (None, "abc", "abc"),       # unparseable → keep raw
    (1.0, 7.5, 7.5),            # already numeric → untouched
    (1.0, None, None),
])
def test_coerce_arg_value(original, raw, expect):
    assert coerce_arg_value(original, raw) == expect


def test_edited_numeric_is_a_real_number_not_a_string():
    """THE safety-critical one: check_global_bounds only inspects int/float, so
    a string would sail past the post-edit bounds re-check untouched."""
    d = build_decision("edit", "SetBias", {"bias_v": 1.0}, {"bias_v": "50"}, "")
    v = d["edited_action"]["args"]["bias_v"]
    assert isinstance(v, float) and v == 50.0
    assert not isinstance(v, str)


# ── allow-list enforcement ───────────────────────────────────────────────────
def test_allowed_decisions_blocks_edit_on_an_approve_only_interrupt():
    """EmergencyRetract-style gates advertise approve-only. Letting an operator
    'edit' one would let them rewrite its parameters — a real bypass."""
    d = enforce_allowed("edit", ["approve"], "EmergencyRetract",
                        {"z_m": 1e-6}, {"z_m": "9"}, "")
    assert d["type"] == "reject"
    assert "not permitted" in d["message"]
    # the operator's params must NOT have made it through in any form
    assert "edited_action" not in d


def test_allowed_decisions_permits_what_it_advertises():
    d = enforce_allowed("edit", ["approve", "reject", "edit"], "SetBias",
                        {"bias_v": 1.0}, {"bias_v": "2.5"}, "")
    assert d["type"] == "edit"
    assert d["edited_action"]["args"]["bias_v"] == 2.5


def test_empty_allow_list_means_unrestricted():
    """Absent metadata must not accidentally lock everything out — the blocked
    worker still needs a well-formed decision."""
    assert enforce_allowed("approve", [], "X", {}, None, "")["type"] == "approve"
    assert enforce_allowed("approve", None, "X", {}, None, "")["type"] == "approve"


def test_disallowed_verdict_collapses_rather_than_raising():
    """It must degrade to a decision, never an exception — an exception here
    leaves the worker blocked forever."""
    d = enforce_allowed("banana", ["approve"], "X", {}, None, "")
    assert isinstance(d, dict) and d["type"] == "reject"


# ── workflow_human route resolution ──────────────────────────────────────────
def test_workflow_route_matches_case_insensitively():
    d, err = build_workflow_route("RETRY", ["retry", "skip"], "note text")
    assert err is None
    assert d == {"route": "retry", "note": "note text"}


def test_workflow_route_rejects_an_option_not_offered():
    """interpreter.py raises RuntimeError on an out-of-set route, so this must
    be caught at the API edge and NOT written into the resolved store."""
    d, err = build_workflow_route("banana", ["retry", "skip"], "")
    assert d is None
    assert err and "banana" in err


def test_workflow_route_without_declared_options_passes_through():
    d, err = build_workflow_route("anything", [], "")
    assert err is None and d["route"] == "anything"


# ── ask_user answer resolution ───────────────────────────────────────────────
# What the operator is permitted to answer is defined by the question the agent
# asked, so the checks live here (pure, authoritative) rather than in the UI.
# An error means the caller answers `answer_invalid` and writes NOTHING into the
# resolved store, leaving the blocked worker free to receive a corrected answer.
_ASK = {
    "question": "先扫哪个区域？",
    "options": [{"label": "A 区", "description": ""},
                {"label": "B 区", "description": ""}],
    "multi_select": False,
    "allow_custom": True,
}


def test_ask_answer_accepts_an_offered_option():
    d, err = build_ask_answer(["B 区"], "", "看起来更干净", _ASK)
    assert err is None
    assert d == {"selected": ["B 区"], "custom_text": "", "note": "看起来更干净"}


def test_ask_answer_rejects_an_option_never_offered():
    d, err = build_ask_answer(["C 区"], "", "", _ASK)
    assert d is None
    assert err and "C 区" in err


def test_ask_answer_rejects_multiple_picks_on_a_single_select():
    d, err = build_ask_answer(["A 区", "B 区"], "", "", _ASK)
    assert d is None
    assert err and "单选" in err


def test_ask_answer_allows_multiple_picks_when_the_question_said_so():
    d, err = build_ask_answer(["A 区", "B 区"], "", "", dict(_ASK, multi_select=True))
    assert err is None
    assert d["selected"] == ["A 区", "B 区"]


def test_ask_answer_rejects_custom_text_when_the_question_forbade_it():
    d, err = build_ask_answer(["A 区"], "其实我想扫 C", "",
                              dict(_ASK, allow_custom=False))
    assert d is None
    assert err and "自定义" in err


def test_ask_answer_rejects_an_empty_answer():
    """Neither a pick nor text is not an answer — resuming the tool with it
    would make the agent believe the operator said something."""
    d, err = build_ask_answer([], "   ", "", _ASK)
    assert d is None
    assert err and "空" in err


def test_ask_answer_open_question_takes_custom_text_only():
    d, err = build_ask_answer([], "扫左上角", "",
                              {"question": "怎么扫？", "options": [],
                               "multi_select": False, "allow_custom": True})
    assert err is None
    assert d == {"selected": [], "custom_text": "扫左上角", "note": ""}


def test_ask_answer_tolerates_a_missing_ask_block():
    """A pending row whose ``ask`` was lost (old checkpoint, odd client) must
    still be answerable with free text rather than becoming unanswerable."""
    d, err = build_ask_answer([], "随便", "", None)
    assert err is None
    assert d["custom_text"] == "随便"
