"""ConductSpec —— 「这个值能不能存在」这一道,在 import 的那一刻。

设计:``docs/v2/design/campaign_director_design.md`` §4.1-4.6。

这一层测的全是**闭集与搭配**:一个拼错的 kind、一个接到 ``pass`` 的 escape、
一个没有 ``stale_after_s`` 的温度条件 —— 它们的共同点是,放过去之后要等到凌晨
三点、针在表面上、人在睡觉的时候才会现形。所以判定越早越好,越便宜越好。

结构规则(bindings 命中上游、等人前必已退针、DANGEROUS 落在声明了 capability
的阶段)不在这里 —— 那些需要整份 spec 才判得了,住在 ``test_validator.py``。
"""
from __future__ import annotations

# ── path bootstrap ──
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

import dataclasses

import pytest

from mast.conduct.spec import (
    CONSERVATIVE_VERDICTS,
    EVIDENCE_EPOCHS,
    ConductBudget,
    ConductSpec,
    ConditionSpec,
    DetourPolicy,
    EvidenceSpec,
    GateOutcome,
    GateSpec,
    ParamSpec,
    RuleLeaf,
    RuleTree,
    StageSpec,
    StepSpec,
    WaitSpec,
)


def _step(step_id="X.01", **kw) -> StepSpec:
    kw.setdefault("kind", "skill")
    kw.setdefault("skill", "SafeRetract")
    return StepSpec(step_id=step_id, **kw)


def _gate(**kw) -> GateSpec:
    kw.setdefault("gate_id", "g")
    kw.setdefault("kind", "rule")
    kw.setdefault("rule", RuleLeaf("x", "exists"))
    kw.setdefault("routes", {"pass": GateOutcome("pass")})
    return GateSpec(**kw)


# ── 闭集 ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [
    {"kind": "scan"},                       # 不存在的 kind
    {"kind": "skill", "skill": ""},         # skill 步没有技能名
    {"kind": "analysis", "analysis_fn": ""},
    {"kind": "wait"},                       # wait 步没有 WaitSpec
])
def test_a_malformed_step_dies_at_import(kwargs):
    with pytest.raises((ValueError, TypeError)):
        StepSpec(step_id="X", **kwargs)


def test_a_step_cannot_be_two_kinds_at_once():
    """kind 决定该带哪些字段。带错的字段不是「多余」,是两种解释同时成立。"""
    with pytest.raises(ValueError):
        StepSpec(step_id="X", kind="skill", skill="ScanAt",
                 analysis_fn="something")


def test_a_parameter_cannot_be_written_and_bound_at_once():
    """params 写死 + bindings 绑定同一个名字 = 两个真源。

    运行时谁赢取决于实现细节,而实现细节会变 —— 到那天,一份「一直好用」的
    模板会开始用另一个数跑。
    """
    with pytest.raises(ValueError) as e:
        StepSpec(step_id="X", kind="skill", skill="ScanAt",
                 params={"bias_v": 1.0}, bindings={"bias_v": "params.b"})
    assert "两个真源" in str(e.value)


def test_ids_are_not_strings_of_characters():
    """``steps="S1"`` 静默变成五个字符是经典坑,这里当场炸。"""
    with pytest.raises(TypeError):
        StageSpec(stage_id="S", title="t", steps="S1")


def test_duplicate_step_ids_are_refused():
    """事件、产出、绑定全按 step_id 分键 —— 重复就是把两步的账记到一起。"""
    with pytest.raises(ValueError) as e:
        StageSpec(stage_id="S", title="t",
                  steps=(_step("a"), _step("a")))
    assert "重复" in str(e.value)


def test_an_entry_action_shares_the_step_id_namespace():
    with pytest.raises(ValueError):
        StageSpec(stage_id="S", title="t", entry_actions=(_step("a"),),
                  steps=(_step("a"),))


# ── 闸门:判不了不许变成通过 ─────────────────────────────────────────────

def test_an_llm_gate_must_map_every_route_it_can_return():
    """``decide_route`` 保证返回值必是 node 的某条路由。

    少映射一条 = 一个静默的洞:模型选中了它,Director 拿不到 verdict,然后呢?
    """
    node = {"id": "n", "routes": {"go": "继续", "stop": "停"}}
    with pytest.raises(ValueError) as e:
        _gate(kind="llm", rule=None, llm_node=node,
              routes={"go": GateOutcome("pass")})
    assert "stop" in str(e.value)


def test_an_escape_route_may_not_map_to_pass():
    """escape 是「判不了」的出口。接到 pass 就是让判不了变成通过。"""
    node = {"id": "n", "routes": {"go": "继续", "uncertain_out": "说不好"},
            "escape": "uncertain_out"}
    with pytest.raises(ValueError) as e:
        _gate(kind="llm", rule=None, llm_node=node,
              routes={"go": GateOutcome("pass"),
                      "uncertain_out": GateOutcome("pass")})
    assert "判不了" in str(e.value)


def test_the_default_escape_is_the_last_route_like_decide_route_says():
    """escape 没写时,``decide_route`` 取最后一条路由 —— 这里必须同一条规则。

    两处对「escape 是谁」有两种理解,就会出现「校验通过、运行时逃向另一条路」。
    """
    node = {"id": "n", "routes": {"go": "继续", "hold": "停一下"}}
    with pytest.raises(ValueError):
        _gate(kind="llm", rule=None, llm_node=node,
              routes={"go": GateOutcome("pass"), "hold": GateOutcome("pass")})
    # 最后一条改成保守裁决就立得住
    g = _gate(kind="llm", rule=None, llm_node=node,
              routes={"go": GateOutcome("pass"),
                      "hold": GateOutcome("wait_operator")})
    assert g.routes["hold"].verdict in CONSERVATIVE_VERDICTS


def test_unattended_escape_can_never_be_pass():
    """无人值守时的 uncertain 去向 —— 没人在场,更不能替它说「过」。"""
    with pytest.raises(ValueError):
        _gate(unattended_escape="pass")


def test_evidence_missing_has_no_default_that_means_pass():
    """证据缺席的去向是闭集 {fail, wait_operator} —— 没有「继续」这一项。"""
    with pytest.raises(ValueError):
        _gate(evidence_missing="pass")


def test_a_rule_gate_needs_a_rule_and_an_llm_gate_needs_a_node():
    with pytest.raises(ValueError):
        _gate(kind="rule", rule=None)
    with pytest.raises(ValueError):
        _gate(kind="llm", rule=None, llm_node=None)


def test_cross_epoch_evidence_is_not_expressible():
    """跨代次证据在本仓一律作废 —— 连「写出来」这一步都不给。

    ``min_epoch`` 是个只有一个合法值的闭集。想放宽的人得先答 S3 设计的开放
    问题 3(畴指纹是样品事实,但它的 verdict 要在新 epoch 下重判),
    而不是顺手多写一个字符串。
    """
    assert EVIDENCE_EPOCHS == ("current",)
    with pytest.raises(ValueError):
        EvidenceSpec(source="frame_metrics", min_epoch="any")


# ── 判据树 ───────────────────────────────────────────────────────────────

def test_not_takes_exactly_one_child():
    with pytest.raises(ValueError):
        RuleTree("not", (RuleLeaf("a", "exists"), RuleLeaf("b", "exists")))


def test_an_empty_combinator_is_refused():
    """``all()`` 在 python 里是 True —— 一个空的 all 就是一个恒过的闸门。"""
    with pytest.raises(ValueError):
        RuleTree("all", ())


def test_in_needs_a_set_not_a_bare_value():
    with pytest.raises(ValueError):
        RuleLeaf("verdict", "in", "resolved")


# ── 等待:两个闸各答各的问题 ─────────────────────────────────────────────

def test_a_condition_wait_must_declare_when_the_reading_goes_stale():
    """缺少温度更新时应通过 stale_after_s 超时降级，避免在不可用输入上无限等待。"""
    with pytest.raises(ValueError) as e:
        ConditionSpec(signal="temperature_k", op="<=", value=5.0,
                      stale_after_s=0)
    assert "读不到" in str(e.value)


def test_a_both_wait_keeps_the_human_gate():
    """双闸 = 人的 ack **AND** 物理条件。关掉人闸就不是双闸了。

    人确认了不等于降到温,降到温不等于样品换好 —— 两个证据回答两个问题,
    互不替代。
    """
    cond = ConditionSpec(signal="temperature_k", op="<=", value=5.0,
                         stale_after_s=600.0)
    with pytest.raises(ValueError):
        WaitSpec(kind="both", message="换样品", condition=cond,
                 ack_required=False)
    w = WaitSpec(kind="both", message="换样品", condition=cond)
    assert w.ack_required is True


def test_a_condition_wait_defaults_to_no_human_gate():
    cond = ConditionSpec(signal="temperature_k", op="<=", value=5.0,
                         stale_after_s=600.0)
    assert WaitSpec(kind="condition", message="等降温",
                    condition=cond).ack_required is False


def test_an_operator_wait_may_not_smuggle_a_condition():
    cond = ConditionSpec(signal="temperature_k", op="<=", value=5.0,
                         stale_after_s=600.0)
    with pytest.raises(ValueError):
        WaitSpec(kind="operator", message="等人", condition=cond)


# ── 绕道 ─────────────────────────────────────────────────────────────────

def test_a_detour_with_triggers_but_nowhere_to_go_is_refused():
    """触发得了、无处可去 —— 又一个「能挂不能解」。"""
    with pytest.raises(ValueError) as e:
        DetourPolicy(target_stage="", triggers=frozenset({"gate_verdict"}))
    assert "无处可去" in str(e.value)


def test_a_conduct_without_a_repair_stage_is_expressible():
    """没有修针段是一种合法形态(早期模板),但必须显式说出来。"""
    d = DetourPolicy(target_stage="", triggers=frozenset())
    assert d.target_stage == "" and not d.triggers


# ── 预算与节奏 ───────────────────────────────────────────────────────────

def test_the_tick_interval_stays_in_the_designed_band():
    with pytest.raises(ValueError):
        ConductBudget(tick_interval_s=120.0)


def test_waiting_never_ticks_faster_than_running():
    with pytest.raises(ValueError):
        ConductBudget(tick_interval_s=10.0, wait_tick_interval_s=5.0)


# ── 不可变与别名 ─────────────────────────────────────────────────────────

def test_a_spec_is_frozen():
    s = _step()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.skill = "SomethingElse"


def test_params_are_copied_not_aliased():
    """模板里的 dict 被下游改写,同一份 spec 的下一次运行就悄悄换了参数。"""
    src = {"bias_v": 1.0}
    s = StepSpec(step_id="X", kind="skill", skill="ScanAt", params=src)
    src["bias_v"] = 99.0
    assert s.params["bias_v"] == 1.0


def test_sequences_become_tuples():
    s = StepSpec(step_id="X", kind="skill", skill="ScanAt",
                 produces=["a", "b"])
    assert isinstance(s.produces, tuple)


# ── 顶层 ─────────────────────────────────────────────────────────────────

def _spec(**kw) -> ConductSpec:
    kw.setdefault("spec_id", "t_v1")
    kw.setdefault("spec_version", 1)
    kw.setdefault("title", "t")
    kw.setdefault("stages", (StageSpec(stage_id="S", title="s",
                                       steps=(_step(),)),))
    kw.setdefault("detour", DetourPolicy(target_stage="", triggers=frozenset()))
    return ConductSpec(**kw)


def test_spec_version_starts_at_one_and_is_an_integer():
    """恢复自检 A5 拿它对账。``True`` 当版本号会一路畅通然后等于 1。"""
    with pytest.raises(ValueError):
        _spec(spec_version=0)
    with pytest.raises(ValueError):
        _spec(spec_version=True)


def test_duplicate_stage_ids_are_refused():
    st = StageSpec(stage_id="S", title="s", steps=(_step(),))
    with pytest.raises(ValueError):
        _spec(stages=(st, st))


def test_duplicate_param_names_are_refused():
    with pytest.raises(ValueError):
        _spec(params_schema=(ParamSpec(name="a", type="float"),
                             ParamSpec(name="a", type="int")))


def test_a_param_envelope_with_swapped_bounds_is_refused():
    with pytest.raises(ValueError):
        ParamSpec(name="a", type="float", min_value=10.0, max_value=1.0)


def test_all_steps_puts_entry_actions_first():
    """入口重申设置跑在主体之前 —— 顺序是承重的(校验器按它判上下游)。"""
    stage = StageSpec(stage_id="S", title="s",
                      entry_actions=(_step("e1"),), steps=(_step("s1"),))
    assert [s.step_id for s in stage.all_steps] == ["e1", "s1"]


def test_analysis_and_wait_steps_do_not_touch_hardware():
    """它们不取仪器令牌 —— 这条被 Director 用来决定要不要 hold_for_skill。"""
    cond = ConditionSpec(signal="temperature_k", op="<=", value=5.0,
                         stale_after_s=600.0)
    assert _step(kind="skill", skill="ScanAt").touches_hardware is True
    assert StepSpec(step_id="a", kind="analysis",
                    analysis_fn="f").touches_hardware is False
    assert StepSpec(step_id="w", kind="wait",
                    wait=WaitSpec(kind="condition", message="m",
                                  condition=cond)).touches_hardware is False


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
