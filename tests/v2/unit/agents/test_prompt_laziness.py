"""Agent prompts must define completion and stopping conditions. These tests verify the declared instructions, while model compliance requires separate evaluation."""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
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
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import importlib  # noqa: E402

import pytest  # noqa: E402

_AGENTS = ["literature", "experiment_design", "instrument_control",
           "data_processing", "paper_writing", "paper_review"]


def _prompt(agent: str) -> str:
    return importlib.import_module(f"mast.agents.{agent}.prompts").SYSTEM_PROMPT


def _router() -> str:
    from mast.agents.orchestrator.graph import _ROUTER_PROMPT
    return _ROUTER_PROMPT


# ════════════════════════════════════════════════════════════════════════════
# Every agent is told when to stop
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("agent", _AGENTS)
def test_agent_has_a_scope_boundary_section(agent):
    """A section that says what NOT to do, near the top where it is read."""
    p = _prompt(agent)
    markers = ("工作边界", "计划规模", "报告规模", "执行边界", "你在审什么")
    assert any(m in p for m in markers), (
        f"{agent} has no scope-boundary section — it will keep going until a "
        "numeric cap stops it"
    )


@pytest.mark.parametrize("agent", _AGENTS)
def test_agent_is_told_what_to_do_when_unsure(agent):
    """The tie-break must resolve toward doing LESS. Without it, "should I also
    …?" resolves toward more every time."""
    p = _prompt(agent)
    assert "拿不准" in p or "不确定" in p, (
        f"{agent} never says which way to lean when it is unsure")


# ════════════════════════════════════════════════════════════════════════════
# Scope and completion contracts
# ════════════════════════════════════════════════════════════════════════════

def test_literature_does_not_read_full_texts_by_default():
    """Full-text reading is not the default retrieval step."""
    p = _prompt("literature")
    assert "默认不读全文" in p
    assert "一次检索就够" in p


def test_review_defaults_to_accept():
    p = _prompt("paper_review")
    assert "默认判决是 ACCEPT" in p, "review still has no default verdict"


def test_review_separates_writing_defects_from_experiment_scope():
    """The core of the loop: paper_writing cannot add a control experiment by
    rewriting. Demanding it produces either fabricated data or endless apology."""
    p = _prompt("paper_review")
    assert "实验范围" in p and "无法通过改写补上" in p
    for phrase in ("对照", "误差棒"):
        assert phrase in p, f"the rubric item that cannot be fixed by writing "
        f"({phrase}) is not called out as a limitation rather than a revision"


def test_review_knows_it_is_not_reviewing_a_journal_submission():
    p = _prompt("paper_review")
    assert "内部实验报告" in p
    assert "rigorous peer reviewer" not in p, (
        "the framing that produced journal-grade demands is still there")


def test_design_ties_plan_size_to_the_request():
    """Plan scope must follow the requested measurement scope."""
    p = _prompt("experiment_design")
    assert "用户说多少就是多少" in p
    assert "没人要求的测量" in p


def test_instrument_control_does_not_add_measurements():
    """允许选择执行手段和必要验证，但不得主动扩大测量目标。"""
    p = _prompt("instrument_control")
    assert "不要**主动加新的测量目标" in p
    assert "计划里有没有" in p, "no criterion for whether an action belongs"


def test_instrument_control_may_verify_but_not_expand():
    """放开的那一半也要钉住 —— 而且要连**边界**一起钉。

    「可以追加验证性测量」如果没有边界,它就是「想多测什么都行」的另一种说法。
    边界有三条:服务于已有目标、每步至多一次、报告里标出来。三条缺一条,这条
    放开就退化成上一条测试正在防的东西。
    """
    p = _prompt("instrument_control")
    assert "验证性" in p
    assert "服务于计划里已经有的那个测量目标" in p, "没有把验证与新目标区分开"
    assert "至多一次" in p, "验证性测量没有次数上限 = 没有边界"


def test_data_processing_does_not_analyse_unasked():
    p = _prompt("data_processing")
    assert "分析是按需的" in p or "ONLY ON REQUEST" in p


def test_paper_writing_does_not_pad_the_report():
    p = _prompt("paper_writing")
    assert "内部实验报告" in p
    assert "不要编数据补上" in p, (
        "nothing stops it from inventing the control experiment review asked for")


# ════════════════════════════════════════════════════════════════════════════
# The orchestrator decides how many rounds everything runs
# ════════════════════════════════════════════════════════════════════════════

def test_router_defines_what_done_means():
    r = _router()
    assert "每个阶段只走一遍" in r
    assert "一遍过是正常结果" in r


def test_router_caps_the_write_review_ping_pong():
    """The loop the operator named. One round, then hand the residue to a human —
    the final standard for a report is set by a person, not by two agents
    persuading each other.

    Tightened 2026-07-28 after a measured end-to-end run: the flow did
    PW → PR → PW → **PR**, and that second review — re-reading a draft that had
    just addressed the first one — cost about as many super-steps as the entire
    scan + analysis before it, and changed nothing. "One round" now explicitly
    means the revision is NOT re-reviewed."""
    r = _router()
    assert "最多一个来回" in r
    assert "改完不再复审" in r
    assert "遗留问题" in r, "还得说清剩下的意见去哪儿了"


def test_router_makes_review_opt_in():
    """审稿是可选的。A request for 「写一份实验报告」 got a review
    round it never asked for — the router's own rule already said 「用户没要求的
    阶段不要自己加」 while two other lines told it to route PW → PR by default."""
    r = _router()
    assert "审稿是可选的" in r
    # And the contradicting instructions must be gone: nothing may state the
    # PW → PR hop as unconditional.
    assert "paper_writing → paper_review" not in r
    assert "**paper_writing** → **paper_review**" not in r


def test_the_default_shape_diagram_does_not_contradict_the_rule():
    """The prompt's own pipeline diagram is the most directive line in its
    section, and it USED TO END 「… → 报告 → 审稿 → __end__」 while the prose two
    lines below said review was optional.

    Measured, not assumed: a HITL-on end-to-end run on the real LLM dispatched
    paper_review for a goal that asked only for a report — after the prose had
    already been changed. The model followed the diagram. A rule that contradicts
    the picture next to it is not a rule.
    """
    r = _router()
    shape = next((ln for ln in r.splitlines()
                  if "文献" in ln and "__end__" in ln), "")
    assert shape, "默认形状那一行不见了"
    assert "审稿" not in shape, f"默认形状里仍然写着审稿: {shape.strip()}"
    # The optional branch must still be reachable — opt-in, not removed.
    assert "仅当用户要求" in r


def test_router_treats_the_visit_ceiling_as_a_breaker_not_a_budget():
    r = _router()
    assert "硬熔断" in r
    assert "不是可以用满的配额" in r


def test_the_router_does_not_claim_a_breaker_that_was_deleted():
    """The router should describe only the implemented total-hop limit."""
    r = _router()
    assert "总跳数" in r, "the one real breaker is not named"
    assert "40" in r
    for phantom in ("最多 6 次", "6 次是硬熔断"):
        assert phantom not in r, (
            f"router prompt still advertises the deleted per-agent breaker: {phantom}")


def test_router_still_drives_the_pipeline():
    """The autonomy must survive. The operator wants one sentence to run the
    whole flow — the fix is a terminal condition, not passivity."""
    r = _router()
    # 2026-08-24 中文化：钉中文实质而不是英文原句。
    assert "不要等人催" in r, "「不要等用户催」那条不见了"
    assert "空档不等于任务完成" in r, (
        "「阶段之间的空档不等于任务完成」不见了 —— 少了它，编排器会在每个阶段"
        "之间停下来等人")
    assert "literature" in r and "paper_review" in r


def test_router_does_not_add_unrequested_stages():
    r = _router()
    assert "用户没要求的阶段" in r


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
