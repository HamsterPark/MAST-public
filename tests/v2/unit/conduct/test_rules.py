"""闸门求值 —— 三态判据树 + 路由映射。

设计:``campaign_director_design.md`` §4.4、§10-1(「读不到」不是一个值)、
§10-5(fail-open 方向:默认保守)。

这一层全是纯函数,所以测得起穷举。核心命题只有一句:**「判不了」必须是第三种
结果**,而且它的去向由 spec 声明,不由求值器替它决定。
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.conduct import rules
from mast.conduct.rules import FALSE, TRUE, UNDECIDABLE
from mast.conduct.spec import (
    EvidenceSpec,
    GateOutcome,
    GateSpec,
    RuleLeaf,
    RuleTree,
)

E = {"n_keep": 3, "verdict": "resolved", "nothing": None}


# ── 叶子 ─────────────────────────────────────────────────────────────────

def test_a_present_field_compares_normally():
    assert rules.evaluate(RuleLeaf("n_keep", ">=", 1), E) == TRUE
    assert rules.evaluate(RuleLeaf("n_keep", "<=", 1), E) == FALSE
    assert rules.evaluate(RuleLeaf("verdict", "==", "resolved"), E) == TRUE
    assert rules.evaluate(RuleLeaf("verdict", "in", ("resolved", "absent")), E) == TRUE


def test_a_missing_field_is_undecidable_not_false():
    """把「没测出来」说成「测出来是零」—— 本仓一天出现五次的那个形状。"""
    assert rules.evaluate(RuleLeaf("n_missing", ">=", 1), E) == UNDECIDABLE


def test_a_present_but_null_field_is_not_the_same_as_absent():
    """``None`` 是一个合法读数(「这一项没有值」),和「这一项不在证据包里」
    是两件要做不同事的事。"""
    assert rules.evaluate(RuleLeaf("nothing", "exists", None), E) == TRUE
    assert rules.evaluate(RuleLeaf("absent_key", "exists", None), E) == FALSE


def test_exists_is_the_one_operator_that_can_decide_on_absence():
    assert rules.evaluate(RuleLeaf("n_keep", "exists"), E) == TRUE
    assert rules.evaluate(RuleLeaf("nope", "exists"), E) == FALSE


def test_a_type_mismatch_is_undecidable_not_false():
    """拿字符串和数比大小不是「不成立」,是判不了 —— 折叠成 False 会让一条
    坏掉的判据看起来像一次正常的否决。"""
    assert rules.evaluate(RuleLeaf("verdict", ">=", 1), E) == UNDECIDABLE


def test_dotted_paths_reach_into_nested_evidence():
    ev = {"frame": {"metrics": {"snr": 12.0}}}
    assert rules.evaluate(RuleLeaf("frame.metrics.snr", ">=", 10.0), ev) == TRUE
    assert rules.evaluate(RuleLeaf("frame.metrics.nope", ">=", 1), ev) == UNDECIDABLE


# ── 组合 ─────────────────────────────────────────────────────────────────

def test_all_lets_a_definite_false_win_over_an_undecidable():
    """一条确定为假的子判据足以否决整棵树。

    反过来报成判不了,会把一次本该干脆的否决变成一次要人来看的悬案 ——
    保守不等于什么都推给人。
    """
    tree = RuleTree("all", (RuleLeaf("n_keep", "<=", 0),      # false
                            RuleLeaf("missing", ">=", 1)))     # undecidable
    assert rules.evaluate(tree, E) == FALSE


def test_all_is_undecidable_when_nothing_is_definitely_false():
    tree = RuleTree("all", (RuleLeaf("n_keep", ">=", 1),      # true
                            RuleLeaf("missing", ">=", 1)))     # undecidable
    assert rules.evaluate(tree, E) == UNDECIDABLE


def test_any_lets_a_definite_true_win():
    tree = RuleTree("any", (RuleLeaf("n_keep", ">=", 1),
                            RuleLeaf("missing", ">=", 1)))
    assert rules.evaluate(tree, E) == TRUE


def test_any_is_undecidable_when_nothing_is_definitely_true():
    tree = RuleTree("any", (RuleLeaf("n_keep", "<=", 0),
                            RuleLeaf("missing", ">=", 1)))
    assert rules.evaluate(tree, E) == UNDECIDABLE


def test_not_of_undecidable_stays_undecidable():
    """取反一个「不知道」还是「不知道」。"""
    assert rules.evaluate(RuleTree("not", (RuleLeaf("missing", ">=", 1),)),
                          E) == UNDECIDABLE
    assert rules.evaluate(RuleTree("not", (RuleLeaf("n_keep", ">=", 1),)), E) == FALSE


# ── 闸门:证据缺席 ───────────────────────────────────────────────────────

def _gate(**kw) -> GateSpec:
    kw.setdefault("gate_id", "g")
    kw.setdefault("kind", "rule")
    kw.setdefault("rule", RuleLeaf("n_keep", ">=", 1))
    kw.setdefault("routes", {"pass": GateOutcome("pass"),
                             "fail": GateOutcome("wait_operator")})
    kw.setdefault("evidence", (EvidenceSpec(source="step_data", selector="s"),))
    return GateSpec(**kw)


def test_missing_evidence_takes_the_declared_route_not_the_rule():
    """闸门根本看不到那些证据,所以不可能拿它们判 —— 结构过滤的意义在这里。"""
    r = rules.evaluate_gate(_gate(), {}, attended=True, missing=("step_data:s",))
    assert r.verdict == "wait_operator"
    assert r.evidence_missing is True
    assert "证据缺席" in r.reason


def test_missing_evidence_can_be_declared_as_fail():
    r = rules.evaluate_gate(_gate(evidence_missing="fail"), {}, attended=True,
                            missing=("x",))
    assert r.verdict == "fail"


# ── 闸门:rule 型 ────────────────────────────────────────────────────────

def test_a_true_rule_passes_and_a_false_rule_takes_the_fail_route():
    assert rules.evaluate_gate(_gate(), E, attended=True).verdict == "pass"
    assert rules.evaluate_gate(_gate(rule=RuleLeaf("n_keep", "<=", 0)),
                               E, attended=True).verdict == "wait_operator"


def test_an_undecidable_rule_asks_the_operator_when_attended():
    r = rules.evaluate_gate(_gate(rule=RuleLeaf("missing", ">=", 1)),
                            E, attended=True)
    assert r.verdict == "wait_operator"
    assert r.escaped is True
    assert r.rule_state == UNDECIDABLE


def test_an_undecidable_rule_takes_the_declared_escape_when_unattended():
    """无人值守时问人必超时 —— 直接走 spec 声明的保守去向。"""
    g = _gate(rule=RuleLeaf("missing", ">=", 1), unattended_escape="fail")
    assert rules.evaluate_gate(g, E, attended=False).verdict == "fail"


# ── 闸门:llm 型 ─────────────────────────────────────────────────────────

def _llm_gate(**kw) -> GateSpec:
    node = {"id": "n", "routes": {"go": "继续", "hold": "停一下"}, "escape": "hold"}
    kw.setdefault("gate_id", "lg")
    kw.setdefault("kind", "llm")
    kw.setdefault("llm_node", node)
    kw.setdefault("rule", None)
    kw.setdefault("routes", {"go": GateOutcome("pass"),
                             "hold": GateOutcome("wait_operator")})
    kw.setdefault("evidence", (EvidenceSpec(source="step_data", selector="s"),))
    return GateSpec(**kw)


def test_an_llm_route_maps_to_its_declared_verdict():
    r = rules.evaluate_gate(_llm_gate(), E, attended=True,
                            decide_route=lambda node, inp: {"route": "go",
                                                            "reason": "看着行"})
    assert r.verdict == "pass" and r.llm_used is True


def test_an_escaped_llm_decision_is_forced_conservative_when_unattended():
    """无人值守时 uncertain 不问人,直接保守分支。"""
    g = _llm_gate(unattended_escape="fail")
    r = rules.evaluate_gate(g, E, attended=False,
                            decide_route=lambda n, i: {"route": "hold",
                                                       "escaped": True})
    assert r.verdict == "fail"


def test_a_missing_decider_is_undecidable_not_a_pass():
    """判决器没接上 ⇒ 判不了。**不是通过** —— 这正是「没接线」看起来像
    「一切正常」的那族缺陷。"""
    r = rules.evaluate_gate(_llm_gate(), E, attended=True, decide_route=None)
    assert r.verdict == "wait_operator" and r.escaped is True


def test_a_crashing_decider_never_takes_the_workflow_down():
    def boom(node, inputs):
        raise RuntimeError("provider 挂了")

    r = rules.evaluate_gate(_llm_gate(), E, attended=True, decide_route=boom)
    assert r.verdict == "wait_operator"
    assert "判不了" in r.reason


def test_a_route_with_no_verdict_mapping_is_undecidable_not_a_pass():
    """spec 保证每条路由都有裁决;真出现没映射的,说明保证被人改开了。"""
    r = rules.evaluate_gate(_llm_gate(), E, attended=True,
                            decide_route=lambda n, i: {"route": "沒見過"})
    assert r.verdict == "wait_operator" and r.escaped is True


def test_mutation_folding_undecidable_into_false_makes_a_bad_rule_look_decisive(
        monkeypatch):
    """变异验证:把三态压成两态,判不了就会长得像一次正常否决。

    ① 先证明变异生效(判不了变成了 false);② 再看被守卫的行为:一条**取不到
    证据**的判据,从「请人来看」变成了「判定不通过」,于是走 fail 路由 ——
    这里把 fail 接成 ``detour``,后果就是拿一次「没读到」触发一整轮修针绕道。
    """
    gate = _gate(rule=RuleLeaf("missing", ">=", 1),
                 routes={"pass": GateOutcome("pass"),
                         "fail": GateOutcome("detour", "判定针尖坏了")})
    good = rules.evaluate_gate(gate, E, attended=True)
    assert good.verdict == "wait_operator" and good.escaped is True

    real = rules.evaluate

    def two_state(rule, evidence):
        got = real(rule, evidence)
        return FALSE if got == UNDECIDABLE else got

    monkeypatch.setattr(rules, "evaluate", two_state)
    # ① 变异已应用
    assert rules.evaluate(RuleLeaf("missing", ">=", 1), E) == FALSE
    # ② 守卫失效:证据没读到,却判出了「针尖坏了,去绕道」
    bad = rules.evaluate_gate(gate, E, attended=True)
    assert bad.verdict == "detour" and bad.escaped is False


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
