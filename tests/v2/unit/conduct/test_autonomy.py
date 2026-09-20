"""自主度：放开的是「谁点头」，不是「什么可做」。

这一组测试的重心不在三个分支各走一遍（那是最容易写也最不值钱的部分），
而在两条**方向性**的断言上：

1. **认不出来 ⇒ 最严**。一个拼错的档位名如果被当成 autonomous，仪器就会在
   没人看着的时候按一个谁也没批准过的策略动。本仓在「读不到被折成一个具体
   的值」上栽过一整天，这里是同一个形状的另一个入口。
2. **包络不随自主度动**。这一条钉的是**别的模块**——autonomy 自己不做数值
   判断，所以要测的是「它没有那个能力」，而不是「它这次没那么做」。
"""

from __future__ import annotations

import inspect

import pytest

from mast.conduct import autonomy as A


# ── 闭集与收口 ────────────────────────────────────────────────────

def test_the_levels_are_ordered_from_strict_to_loose():
    assert A.AUTONOMY_LEVELS == ("attended", "supervised", "autonomous")
    assert A.rank("attended") < A.rank("supervised") < A.rank("autonomous")


def test_the_default_is_the_one_that_changes_nothing():
    assert A.DEFAULT_LEVEL == "attended"


@pytest.mark.parametrize("bad", ["", None, "auto", "自主", "2", "unsupervised"])
def test_a_name_we_do_not_recognise_becomes_the_strictest(bad):
    """含一个陷阱：``"2"`` 是**字符串**，不是档位码。

    normalise 不该去猜「他大概是想写 autonomous」——那种体贴正是
    「读不到被折成一个具体的值」的开头。
    """
    assert A.normalise(bad) == "attended"


@pytest.mark.parametrize("messy", ["AUTONOMOUS ", " Supervised", "Attended"])
def test_case_and_whitespace_are_forgiven_but_meaning_is_never_guessed(messy):
    """容错与猜测是两件事。

    大小写和首尾空格来自人手打字与 JSON 往返，认它们不损失任何信息；
    而把 "auto" 认成 "autonomous" 是在替人做决定 —— 上一条测的正是后者。
    """
    assert A.normalise(messy) == messy.strip().lower()


@pytest.mark.parametrize("code,expect", [
    (0, "attended"), (1, "supervised"), (2, "autonomous"),
    (0.0, "attended"), (2.0, "autonomous"),
    (3, "attended"), (-1, "attended"), (None, "attended"), ("x", "attended"),
])
def test_the_settings_code_round_trips_and_fails_strict(code, expect):
    assert A.from_code(code) == expect


def test_code_and_name_are_inverse():
    for lv in A.AUTONOMY_LEVELS:
        assert A.from_code(A.to_code(lv)) == lv


def test_the_stricter_of_two_wins():
    assert A.stricter_of("autonomous", "attended") == "attended"
    assert A.stricter_of("attended", "autonomous") == "attended"
    assert A.stricter_of("autonomous", "supervised") == "supervised"
    assert A.stricter_of("autonomous", "autonomous") == "autonomous"
    # 认不出来的一侧把结果拉到最严 —— 而不是被忽略。
    assert A.stricter_of("autonomous", "typo") == "attended"


# ── 谁能点头 ──────────────────────────────────────────────────────

def test_a_human_may_approve_in_every_level():
    for lv in A.AUTONOMY_LEVELS:
        v = A.who_may_approve(lv, by="用户")
        assert v.allowed and v.ignition_delay_s == 0.0


def test_an_agent_is_refused_in_attended_and_told_what_to_change():
    v = A.who_may_approve("attended", by="agent:research_director")
    assert not v.allowed
    # 拒绝必须给出路：一个只被告知「不行」的 agent 会换个说法再试一次。
    assert "supervised" in v.reason or "autonomous" in v.reason
    # 而且要说清放开的是什么、不放开的是什么。
    assert "包络" in v.reason


def test_supervised_lets_an_agent_approve_but_defers_ignition():
    v = A.who_may_approve("supervised", by="agent:experiment_design")
    assert v.allowed and v.deferred
    assert v.ignition_delay_s == A.DEFAULT_IGNITION_DELAY_S
    assert "撤回" in v.reason or "abort" in v.reason


def test_supervised_honours_an_explicit_window_including_zero():
    assert A.who_may_approve("supervised", by="agent:x",
                             ignition_delay_s=0).ignition_delay_s == 0.0
    assert A.who_may_approve("supervised", by="agent:x",
                             ignition_delay_s=30).ignition_delay_s == 30.0
    # 负数不该变成「立刻」以外的任何东西
    assert A.who_may_approve("supervised", by="agent:x",
                             ignition_delay_s=-5).ignition_delay_s == 0.0


def test_autonomous_fires_immediately():
    v = A.who_may_approve("autonomous", by="agent:research_director")
    assert v.allowed and not v.deferred


def test_an_unknown_level_refuses_the_agent():
    """档位读不出来时，agent 批准不算数 —— 这是最严档的行为。"""
    v = A.who_may_approve("typo", by="agent:x")
    assert not v.allowed


# ── 方向性：这一层不许碰包络 ────────────────────────────────────────

def test_this_module_cannot_relax_any_envelope():
    """结构断言：autonomy 不 import 任何做数值判断的东西。

    「autonomous 档下把上限放宽一点」是这套设计里最容易被顺手写下的一行，
    也是唯一会让三档变成三套安全标准的一行。让它在结构上写不出来：这个模块
    看不见 validator、看不见 SafetyGate、看不见任何参数表。
    """
    import ast

    tree = ast.parse(inspect.getsource(A))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    src = " ".join(sorted(imported))
    for forbidden in ("validator", "safety", "FIELD_BOUNDS", "check_params",
                      "ParamSpec", "skills"):
        assert forbidden not in src, (
            f"autonomy 里出现了 {forbidden!r}。这一层只回答「谁点头」；"
            f"任何对「什么可做」的改动都属于包络，要去它自己的模块里改，"
            f"并留下依据。")


def test_the_docstring_still_states_the_split():
    doc = A.__doc__ or ""
    assert "谁能点这个头" in doc and "什么可以做" in doc, (
        "模块 docstring 不再说明这条分界。它是这一层存在的全部理由 —— "
        "理由没了，下一个人会很自然地把包络也做成可配的。")
