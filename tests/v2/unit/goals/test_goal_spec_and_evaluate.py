"""目标判据：闭集、三态、以及「结论只从那一个核出来」。

这个文件钉三件事，顺序就是它们的重要性：

1. **done 当且仅当 :func:`mast.conduct.rules.evaluate` 回 TRUE。**
   有一条变异测试把那个核打成恒 UNDECIDABLE，届时**所有**结论必须变 unknown。
   它证明的不是「求值对不对」，而是「没有第二条通往 done 的路」——本仓刚为
   「差点建成第二真源」付过一次账。
2. **读不到 ≠ 通过。** 收集器缺席 / 抛异常 / 少给字段，一律 unknown，不是 done
   也不是 not_done。
3. **闭集整体拒绝。** 非法谓词不许被悄悄丢掉：丢掉 ``all`` 里一个合取项会让
   目标被削弱，于是更早「达成」，于是错误地抑制唤醒。
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
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.conduct import rules as _rules  # noqa: E402
from mast.goals import (  # noqa: E402
    CATALOG,
    DONE,
    NOT_DONE,
    UNKNOWN,
    describe_done_when,
    evaluate_done_when,
    fingerprint,
    normalise_done_when,
)
from mast.goals.spec import compile_rule, iter_predicates  # noqa: E402

_FIELDS = ("literature_report", "experiment_plan", "analysis", "draft",
           "review", "last_scan", "research_campaign")


def _n(raw):
    return normalise_done_when(raw, artifact_fields=_FIELDS)


def _ok(raw):
    spec, errs = _n(raw)
    assert errs == [], errs
    assert spec is not None
    return spec


# ── 1. 闭集 ────────────────────────────────────────────────────────────

def test_absent_done_when_is_not_an_error():
    """没有判据是合法的，而且就是今天所有调用方的样子。"""
    for empty in (None, "", {}, []):
        spec, errs = _n(empty)
        assert (spec, errs) == (None, [])


def test_an_unknown_kind_is_rejected_by_name_and_position():
    spec, errs = _n([{"kind": "artifact_present", "field": "analysis"},
                     {"kind": "vibes"}])
    assert spec is None, "有一条不合法却还是给出了 spec —— 那份 spec 是被削弱过的"
    assert len(errs) == 1
    assert "[done_when[1]]" in errs[0] and "vibes" in errs[0]
    # 目录要出现在报错里：模型在它写错的地方拿到闭集，而不是被劝说去查文档。
    assert "conduct_completed" in errs[0]


def test_a_field_outside_the_waitable_set_is_rejected():
    """``scan_id`` 是随 ``last_scan`` 走的标识符，从不独立「到达」。"""
    spec, errs = _n([{"kind": "artifact_present", "field": "scan_id"}])
    assert spec is None and "scan_id" in errs[0]


def test_missing_and_extra_args_are_both_named():
    _, e1 = _n([{"kind": "conduct_completed"}])
    assert "恰好一个" in e1[0]
    _, e2 = _n([{"kind": "conduct_completed", "spec_id": "a", "conduct_id": "b"}])
    assert "恰好一个" in e2[0]
    _, e3 = _n([{"kind": "artifact_present", "field": "analysis", "typo": 1}])
    assert "typo" in e3[0]


def test_a_node_cannot_be_both_a_combo_and_a_predicate():
    _, errs = _n({"all": [{"kind": "artifact_present", "field": "analysis"}],
                  "kind": "artifact_present"})
    assert errs and "既像组合又像谓词" in errs[0]


def test_a_bare_list_means_all_not_any():
    """列表 = ``all``。``any`` 会让一条便宜谓词短路整个目标 —— 那是「太早停」。"""
    spec = _ok([{"kind": "artifact_present", "field": "analysis"},
                {"kind": "artifact_present", "field": "draft"}])
    assert spec.op == "all"


def test_fingerprint_changes_when_the_criteria_change():
    a = _ok([{"kind": "artifact_present", "field": "analysis"}])
    b = _ok([{"kind": "artifact_present", "field": "draft"}])
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(a) == fingerprint(
        _ok([{"kind": "artifact_present", "field": "analysis"}]))


def test_every_catalog_kind_has_a_collector():
    """目录里加一条谓词却忘了接收集器 ⇒ 它永远判不了，而且看起来像「还没到」。"""
    from mast.goals.sources import make_collector

    collect = make_collector(versions={}, askable=True)
    for kind, spec in CATALOG.items():
        got = collect(kind, {})
        assert got is None or "没有为" not in str(got.get("_why", "")), (
            f"{kind} 在目录里，但 make_collector 没为它接上收集器")


# ── 2. 三态 ────────────────────────────────────────────────────────────

def _spec2():
    return _ok([{"kind": "artifact_present", "field": "analysis"},
                {"kind": "conduct_completed", "spec_id": "x"}])


def test_all_satisfied_is_done():
    v = evaluate_done_when(_spec2(), lambda k, a: (
        {"new_since_baseline": True} if k == "artifact_present"
        else {"completed_count": 1}))
    assert v.verdict == DONE and v.is_done and v.satisfied == 2 and v.total == 2


def test_one_unreadable_is_unknown_not_done():
    v = evaluate_done_when(_spec2(), lambda k, a: (
        {"new_since_baseline": True} if k == "artifact_present" else None))
    assert v.verdict == UNKNOWN and not v.is_done
    assert len(v.unknowns) == 1 and "读不到" in v.unknowns[0].reason


def test_a_definite_false_beats_an_unreadable_one():
    """``all`` 先看假再看判不了 —— 一条确定为假足以否决，不必让人来看。"""
    v = evaluate_done_when(_spec2(), lambda k, a: (
        {"new_since_baseline": False} if k == "artifact_present" else None))
    assert v.verdict == NOT_DONE and len(v.unmet) == 1


def test_a_collector_that_raises_is_unknown_not_done():
    def boom(kind, args):
        raise RuntimeError("库打不开")

    v = evaluate_done_when(_spec2(), boom)
    assert v.verdict == UNKNOWN
    assert all(i.state == UNKNOWN for i in v.per_predicate)


def test_a_collector_that_omits_the_field_is_unknown():
    """字段不在证据包里 ⇒ 判不了。这是「读不到不写值」那条纪律的另一面。"""
    v = evaluate_done_when(
        _ok([{"kind": "artifact_present", "field": "analysis"}]),
        lambda k, a: {})
    assert v.verdict == UNKNOWN


def test_no_done_when_is_unknown_not_done():
    """「没人写过什么算答完」不是「答完了」。"""
    v = evaluate_done_when(None, lambda k, a: {})
    assert v.verdict == UNKNOWN and not v.is_done


def test_any_takes_one_true():
    spec = _ok({"any": [{"kind": "artifact_present", "field": "analysis"},
                        {"kind": "artifact_present", "field": "draft"}]})
    v = evaluate_done_when(spec, lambda k, a: {
        "new_since_baseline": a.get("field") == "draft"})
    assert v.verdict == DONE


def test_two_predicates_of_the_same_kind_do_not_share_evidence():
    """两条同类谓词的证据字段同名 —— 不分命名空间的话第二条会读到第一条的答案。"""
    spec = _ok([{"kind": "artifact_present", "field": "analysis"},
                {"kind": "artifact_present", "field": "draft"}])
    v = evaluate_done_when(spec, lambda k, a: {
        "new_since_baseline": a.get("field") == "analysis"})
    assert v.verdict == NOT_DONE, "两条同类谓词串了台"
    states = [i.state for i in v.per_predicate]
    assert states == [DONE, NOT_DONE], states
    fields = {leaf.field for leaf in _leaves(compile_rule(spec))}
    assert fields == {"p0.new_since_baseline", "p1.new_since_baseline"}


def _leaves(rule):
    from mast.conduct.spec import RuleLeaf, RuleTree

    if isinstance(rule, RuleLeaf):
        return [rule]
    if isinstance(rule, RuleTree):
        return [x for c in rule.children for x in _leaves(c)]
    return []


# ── 3. 变异：结论只从那一个核出来 ────────────────────────────────────

def test_mutation_patching_the_one_core_turns_everything_unknown(monkeypatch):
    """把 ``conduct.rules.evaluate`` 打成恒 UNDECIDABLE ⇒ done 消失。

    这条不是在测求值对不对，而是在测**没有第二条通往 done 的路**。如果本包
    哪天自己写了一段比较逻辑，这条会绿着 —— 所以它同时断言 not_done 也消失了
    （一个只在 done 上短路的旁路同样会被逮住）。
    """
    def _always_undecidable(rule, evidence):
        return _rules.UNDECIDABLE

    monkeypatch.setattr(_rules, "evaluate", _always_undecidable)
    for collector in (lambda k, a: {"new_since_baseline": True,
                                    "completed_count": 9},
                      lambda k, a: {"new_since_baseline": False,
                                    "completed_count": 0}):
        v = evaluate_done_when(_spec2(), collector)
        assert v.verdict == UNKNOWN, (
            "换掉唯一的求值核之后还能得出结论 —— 说明本包里长出了第二份判断")


def test_the_core_is_conducts_and_not_a_copy():
    """结构：本包不许自己实现三态比较。"""
    import mast.goals.verdict as vm

    src = Path(_MASTV2_ROOT, "mast", "goals", "verdict.py").read_text(
        encoding="utf-8")
    assert "_rules.evaluate(" in src
    assert vm._FROM_TRISTATE == {
        _rules.TRUE: DONE, _rules.FALSE: NOT_DONE, _rules.UNDECIDABLE: UNKNOWN}


def test_goals_package_does_not_import_agents_or_core_at_module_level():
    """依赖方向：顶层只许 stdlib + mast.conduct。

    这条挡的是「顺手在顶上 import 一下 artifact_channel」——那会把 agents 栈
    拉进 runtime 的唤醒线程里，而那条路的整个意义是不依赖 agent 上下文。
    """
    import ast

    root = Path(_MASTV2_ROOT, "mast", "goals")
    for f in sorted(root.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in tree.body:            # 只看顶层，函数体内的惰性 import 允许
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.startswith("mast."):
                    assert m.startswith("mast.conduct") or m.startswith("mast.goals"), (
                        f"{f.name} 顶层 import 了 {m} —— 只许 mast.conduct / mast.goals")


# ── 渲染 ────────────────────────────────────────────────────────────────

def test_describe_is_human_readable():
    spec = _spec2()
    text = describe_done_when(spec)
    assert "analysis" in text and "x" in text and "且" in text


def test_render_block_says_do_not_end_when_unmet():
    from mast.goals import render_goal_block

    v = evaluate_done_when(_spec2(), lambda k, a: {"new_since_baseline": False,
                                                   "completed_count": 0})
    block = render_goal_block(v, goal_text="测一测")
    assert "✗" in block and "不要 __end__" in block
    assert "0/2" in block


def test_render_block_is_empty_without_a_goal():
    from mast.goals import GoalVerdict, render_goal_block

    assert render_goal_block(GoalVerdict()) == ""


@pytest.mark.parametrize("kind", sorted(CATALOG))
def test_every_catalog_entry_is_self_consistent(kind):
    """目录自洽：``evidence`` 声明的字段名必须就是 ``leaf`` 要读的那个。

    写岔了的后果是**永远 UNDECIDABLE** —— 一个永远判不了的判据看起来像
    「还没到」，不像 bug，可以安静地活很久。
    """
    spec = CATALOG[kind]
    dummy = _dummy_predicate(kind)
    leaf = spec.leaf(dummy)
    assert leaf.field in spec.evidence, (
        f"{kind}: leaf 读 {leaf.field!r}，而 evidence 声明的是 {spec.evidence}")
    assert spec.describe(dummy)
    assert len(iter_predicates(dummy)) == 1


def _dummy_predicate(kind: str):
    from mast.goals.spec import Predicate

    args = {"field": "analysis", "n": 2, "min_count": 1, "tag": "t",
            "spec_id": "s", "dry_limit": None, "allow_preexisting": False}
    spec = CATALOG[kind]
    use = {k: args[k] for k in spec.arg_names() if k in args}
    if kind == "conduct_completed":
        use.pop("conduct_id", None)
    return Predicate(kind=kind, args=tuple(sorted(use.items())))


# ── 4. 基线：**没有基线 ≠ 当时是空的** ──────────────────────────────

class TestBaselineIsNotOptional:
    """campaign 路径上「没有基线」会一路走到求值 —— 那里它必须是 unknown。

    第一版 ``_cmp_versions`` 在 ``base is None`` 时回 ``True``（注释自己写着
    「没有基线 = 目标设定时那一类是空的**或没记**」）。把「没记」折叠成「当时是
    空的」的后果，在真实路径上是这样的：

    RD 写下 ``done_when: [artifact_present(analysis)]``（工具文档把它排在闭集
    第一位），而实验文件夹里已有任何一份别的纲领留下的 analysis ⇒
    ``campaign_get`` 立刻显示「已达成 1/1」，唤醒调度器把这条纲领下**每一份**
    park 都 ``mark_done_by_goal`` 关掉，那些 agent 一次都不醒。
    """

    @staticmethod
    def _one(baseline, versions):
        from mast.goals.sources import collect_artifact_present

        return collect_artifact_present({"field": "analysis"},
                                        baseline=baseline, versions=versions)

    def test_no_baseline_is_undecidable_not_true(self):
        got = self._one(None, {"analysis": (3, 123.0)})
        assert "new_since_baseline" not in got, (
            f"没有基线却给出了结论：{got} —— 任何历史产物都会让判据假满足")
        assert "基线" in got.get("_why", "") or "版本" in got.get("_why", "")

    def test_a_real_baseline_makes_it_answerable(self):
        assert self._one({"analysis": (3, 123.0)},
                         {"analysis": (3, 123.0)}) == {"new_since_baseline": False}
        assert self._one({"analysis": (3, 123.0)},
                         {"analysis": (4, 999.0)}) == {"new_since_baseline": True}

    def test_a_confirmed_empty_class_is_still_a_readable_no(self):
        """「磁盘上确实一份都没有」是**读得到**的否定，不是判不了。"""
        assert self._one(None, {"analysis": (0, 0.0)}) == {
            "new_since_baseline": False}

    def test_count_predicate_refuses_to_guess_a_zero_baseline(self):
        from mast.goals.sources import collect_artifact_count

        got = collect_artifact_count({"field": "analysis", "n": 2},
                                     baseline=None,
                                     versions={"analysis": (5, 9.0)})
        assert "delta_count" not in got, (
            f"没有基线却数出了增量：{got} —— 历史产物全被算成这次的")

    def test_snapshot_returns_none_when_it_cannot_read(self):
        """抓不到就说抓不到。写一个 ``{}`` = 「那一刻世界是空的」。"""
        from mast.goals.sources import snapshot_baseline

        assert snapshot_baseline(versions={}) is None
        assert snapshot_baseline(versions={"analysis": (0, 0.0)}) is not None

    def test_the_baseline_remembers_what_was_already_in_state(self):
        """基线要同时记下**当时 state 里已有哪些产物**。

        否则续接会话时，上一个任务留在 ``last_wins`` 通道里的产物会被 state
        快路读成「这一 run 刚产出的」。
        """
        from mast.goals.sources import (
            collect_artifact_present,
            snapshot_baseline,
        )

        base = snapshot_baseline(versions={"analysis": (1, 5.0)},
                                 state_present=lambda f: f == "analysis")
        assert base["_state_present"] == ("analysis",)
        got = collect_artifact_present({"field": "analysis"}, baseline=base,
                                       versions={"analysis": (1, 5.0)},
                                       state_present=lambda f: True)
        assert got == {"new_since_baseline": False}, (
            f"基线时就在 state 里的产物被当成了新的：{got}")
