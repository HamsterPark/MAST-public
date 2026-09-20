"""IC must finish the requested measurements before handing results off.

Completion, partial failure and stop conditions must be explicit in the IC
prompt and handoff tool. Deterministic supervisor dispatch cannot compensate
for an early handoff already selected by the instrument-control agent.

Tests distinguish completing the requested scope from adding unrequested
measurements, and preserve stop criteria for an unreachable target.
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


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

import pytest  # noqa: E402

from mast.agents.instrument_control.prompts import SYSTEM_PROMPT as IC  # noqa: E402
from mast.agents.instrument_control.tools import _handoff_description  # noqa: E402

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of  # noqa: E402


def _flat(s: str) -> str:
    """Collapse whitespace so an assertion survives re-wrapping of the prose."""
    return " ".join(s.split())


# ════════════════════════════════════════════════════════════════════════════
# The completion gate
# ════════════════════════════════════════════════════════════════════════════

def test_the_prompt_states_a_completion_gate_before_handoff():
    """The rule that was missing entirely: all requested points first."""
    assert "交接时机" in IC, "没有一节专门讲什么时候才能交接"
    flat = _flat(IC)
    assert "全部" in flat and "才交接" in flat, (
        "没说清「全部做完才交接」——「做完这一步就交接」在五点请求里第一点后就成立")


def test_the_five_point_case_is_spelled_out_not_left_to_inference():
    """A rule stated only in the abstract ("finish your slice") is what we had."""
    assert "5 个点" in IC
    assert "5 条谱都采完" in IC or "5 条谱" in IC


def test_an_alternative_path_is_named():
    """ad7f07d's lesson: a rule that says only "don't hand off" gets ignored under
    the pull of the checklist. It has to say what to do INSTEAD."""
    assert "替代路径" in IC
    assert "接着调下一个仪器工具" in _flat(IC), (
        "只说了「不交接」，没说不交接的时候该干嘛")


def test_the_report_rule_is_no_longer_scoped_to_a_single_scan():
    """The exact wording that fired after point 1 of 5."""
    flat = _flat(IC)
    assert "THEN either handoff_to_data_processing (if the image needs analysis)" \
        not in flat, "单次采集就交接的原句还在"
    assert "报一句进度 ≠ 交接" in flat or "报一句进度 ≠ 交接" in IC, (
        "必须把「每点都要报」和「只在最后交接」分开，否则报告规则又变成交接触发器")


def test_downstream_cannot_do_the_work_instead():
    """The operator's actual complaint. data_processing has no hardware — an
    un-acquired spectrum is not deferred work, it is lost work."""
    assert "碰不到硬件" in IC


# ════════════════════════════════════════════════════════════════════════════
# 执行边界 must not read as permission to stop early
# ════════════════════════════════════════════════════════════════════════════

def test_doing_less_is_named_as_a_deviation_too():
    """执行边界 exists to stop IC adding measurements (test_prompt_laziness). Without
    an explicit carve-out, 「拿不准…不做」 covers points that were actually asked
    for."""
    assert "少做和多做一样是偏离计划" in IC
    assert "不算「多做一步」" in IC, (
        "没把「请求之内没做完」和「请求之外多做」区分开")


def test_the_anti_overreach_rules_survived():
    """The fix must not undo the 2026-07-27 laziness fix — pinned there too, but
    these two prompts are edited by the same hands and pull opposite ways.

    2026-08-25:执行边界改成 what / how 两侧之后,收紧的那一侧仍然完整 —— 判据句
    (「计划里有没有」)、不加新测量目标、拿不准就不做,三条都在。变的是**手段**
    那一侧不再被这几条管住(编排、组合技能、验证性测量),而这条测试从来守的是
    目标那一侧。
    """
    assert "不要**主动加新的测量目标" in IC
    assert "计划里有没有" in IC
    assert "拿不准" in IC


def test_the_how_side_is_opened_without_opening_the_what_side():
    """放开与收紧必须同时可见 —— 只钉一半的话,另一半被悄悄拆掉时没人会知道。"""
    assert "怎么做到" in IC and "不设审批" in IC, "手段那一侧没有被明确放开"
    assert "组合**不给你任何新的 what**" in IC, (
        "没说清「组合出来的技能不能带来新的测量目标」—— 那是这次放开的承重墙")


# ════════════════════════════════════════════════════════════════════════════
# Stop criteria — the opposite failure
# ════════════════════════════════════════════════════════════════════════════

def test_a_repeatedly_failing_point_is_abandoned_not_retried_forever():
    assert "连续失败 2 次" in IC
    assert "继续做剩下的点" in IC, (
        "说了别重试，但没说接着做剩下的点 —— 会变成整批放弃")


def test_unmet_hardware_preconditions_stop_the_sequence():
    assert "硬件预条件不满足" in IC
    assert "handoff_to_supervisor 求助" in _flat(IC)


def test_a_mid_sequence_handoff_that_needs_analysis_first_is_allowed():
    """Not an absolute ban — the legitimate case must stay reachable, and must be
    stated out loud so the operator can tell it from an abandoned run."""
    assert "下一步必须先看分析结果" in IC
    assert "剩余" in IC and "别让人以为你做完了" in IC


def test_tip_repair_returns_to_the_unfinished_points():
    """A CRITICAL tip_change already tells IC to abort + ConditionTip. Without
    this, "repair then hand off" is the natural reading."""
    assert "回到没做完的点位接着做" in IC


# ════════════════════════════════════════════════════════════════════════════
# The tool descriptions — read every turn, same rule
# ════════════════════════════════════════════════════════════════════════════

def test_the_data_processing_handoff_tool_states_the_gate():
    d = _handoff_description("data_processing")
    assert "Hand newly-acquired scan/STS results" not in d, (
        "无条件的工具描述还在 —— 模型每一轮都读它，清单管不住它")
    assert "ONLY once every point/parameter the operator asked for" in d
    assert "cannot touch the instrument" in d


def test_the_supervisor_handoff_tool_is_not_a_mid_sequence_pause():
    d = _handoff_description("supervisor")
    assert "EVERY measurement" in d
    assert "keep acquiring instead" in d


def test_prefer_the_composite_that_does_the_whole_set():
    """GridSTS acquires the whole grid in one call (nx*ny points). Splitting a
    5-point request into single AcquireSTS calls is what creates the chance to
    hand off after point 1."""
    assert "GridSTS" in IC
    assert "nx=5" in IC


# ════════════════════════════════════════════════════════════════════════════
# Where the rule lives — and why it is here
# ════════════════════════════════════════════════════════════════════════════

def test_the_file_explains_why_this_is_not_the_orchestrators_decision():
    """The runtime rule stays in the prompt; maintenance guidance stays in comments."""
    from pathlib import Path
    import mast.agents.instrument_control.prompts as _p

    assert "确定性派发" in IC
    assert "编排器" in IC
    assert "路由提示词在这一跳上根本不会被查询" in IC
    source = Path(_p.__file__).read_text(encoding="utf-8")
    comments = "\n".join(line for line in source.splitlines() if line.lstrip().startswith("#"))
    assert "handoff 在 supervisor_node 里是确定性派发" in comments
    assert "给维护者" in comments
    assert "要改的是**这个文件**" in comments
    assert "给维护者" not in IC
    assert "要改的是**这个文件**" not in IC


def test_the_orchestrator_states_the_same_rule_and_does_not_contradict_it():
    """The router does not decide this hop, but it DOES dispatch data_processing
    on its own — the two prompts must agree."""
    from mast.agents.orchestrator.graph import _ROUTER_PROMPT

    flat = _flat(_ROUTER_PROMPT)
    # 2026-08-24 路由提示词中文化：钉的从英文原句换成中文实质。
    # 要守的是三件事：①「无条件交给 DP」那句不在了；②有「派回 IC」这条出路；
    # ③说清了为什么必须派回（DP 碰不到仪器）。
    assert "After instrument_control completes scans/STS, hand to data_processing." \
        not in flat, "编排器那句无条件的「IC 完了就给 DP」还在，和 IC 提示词打架"
    assert "派回 instrument_control" in flat, (
        "没有「有测量没做完就派回 IC」这条出路")
    assert "碰不到仪器" in flat and "补不了" in flat, (
        "没说清为什么必须派回 —— 少了理由，模型会把它当成一条可商量的偏好")


def test_the_handoff_hint_really_is_dispatched_deterministically():
    """The premise of the whole fix. If the LLM router ran on every hop, the
    orchestrator prompt WOULD be the right place — pin the mechanism, not just
    the claim about it."""
    import inspect

    from mast.agents.orchestrator import graph as og

    src = source_of(og.build_orchestrator_graph) \
        if hasattr(og, "build_orchestrator_graph") else inspect.getsource(og)
    assert "agent_hints" in src
    # The hint branch must short-circuit before the LLM router is consulted.
    assert "routing_hints" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
