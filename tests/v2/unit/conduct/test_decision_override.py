"""裁决停下之后那条**留痕的解锁路**。

## 它修的是什么

`waiting_operator` 有两个来源(`_dispatch` 里那段注释):一个 `wait` 步(等 ack +
物理条件,可轮询),与一次**裁决转人**(闸门判 wait_operator / 恢复自检没过 /
续跑点接不上)。前者一直有出口;**后者一条都没有** ——

* `_apply_wait_op` 要 `active_wait` 且 `wait_id` 匹配,而裁决转人没有 `active_wait`;
* `resume` 只在 `paused` 有效。

于是用户看过之后只剩 `takeover` 与 `abort`:凌晨闸门停下,早上人看了觉得没问题,
唯一的选择是**放弃这份 conduct** 或**永久接管**。本仓已经为这个形状付过一次账
(2026-08-13:端口抖 21 s → 判成硬故障 → 退针挂闩 → 而解闩的函数是死代码,
针和仪器完好、机器锁死到重启)。

## 五条硬判据(逐条对应)

1. **留痕 + 持续显示** —— `decision_overridden` 事件 + `status_reason`,不是闪一次;
2. **只解这一次判定,不是这道闸** —— `decision_id` 每判一次换一个,下次重新判;
3. **闸门的裁决与人的决定分开记** —— `gate_evaluated` 原样留着;
4. **与 detour 的关系** —— 「继续」只回答闸门那个问题,不提供「去修针」那一档;
5. **无人值守不多一条出口** —— 意图只能由人写进 `conduct_ops`,Director 自己
   永远不生成它。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from _harness import (
    FakeExecutor,
    FakeReading,
    FakeTemperature,
    build,
    outcome,
    retract_step,
    rule_gate,
    spec,
    stage,
    step,
    tip_check,
    wait_step,
)

from mast.conduct.director import OP_VALID_STATUSES
from mast.conduct.store import OPS, OPS_REASON_REQUIRED


def _failing_gate_spec(fail_verdict="wait_operator", which="exit"):
    """一个必然判 fail 的闸门(上游没产出它要的字段 ⇒ 证据缺席 ⇒ 转人)。"""
    gate = rule_gate("g1", selector="S.01", fail_verdict=fail_verdict)
    if which == "exit":
        return spec([stage("S", steps=(step("S.01"), step("S.02")), exit_gate=gate)])
    if which == "entry":
        return spec([stage("S", steps=(step("S.01"),), entry_gate=gate)])
    return spec([stage("S", steps=(step("S.01", gate=gate), step("S.02")))])


def _stop_at_gate(tmp_path, which="exit", data=None, **kw):
    """跑到闸门判 wait_operator 为止。"""
    rig = build(tmp_path, _failing_gate_spec(which=which),
                executor=FakeExecutor({"ScanAt": [outcome(True, data=data or {})]}),
                **kw)
    rig.run_until("waiting_operator")
    assert rig.row()["status"] == "waiting_operator", "前提没成立:没停在闸门上"
    return rig


def _pending(rig):
    return rig.director.pending_decision(rig.conduct_id)


class _Rep:
    """极简报告对象 —— 只消费意图,不接着推进。"""

    def __init__(self):
        self.actions: list[str] = []
        self.not_checked: list[str] = []

    def did(self, what):
        self.actions.append(what)
        return self


def _override(rig, decision_id=None, reason="我看过了", *, advance=False):
    """放行。

    默认**只消费这一条意图、不接着推进** —— 一个完整的 tick 在放行之后会继续
    跑下一步,于是「放行把位置放到哪儿」会被下一步的推进盖掉,断言看到的是两件事
    叠加的结果。想看后续就显式 ``advance=True``。
    """
    pend = _pending(rig)
    rig.store.enqueue_op(
        rig.conduct_id, "override_decision",
        args={"decision_id": str(decision_id if decision_id is not None
                                 else (pend or {}).get("decision_id")),
              "reason": reason},
        requested_by="operator")
    if advance:
        return rig.tick()
    rig.director._consume_ops(rig.conduct_id, _Rep())
    return None


def _rejections(rig):
    return [(e["payload"] or {}) for e in rig.events(kind="op_rejected")]


# ── 0. 先钉住「以前出不去」这件事本身 ──────────────────────────────────

def test_ack_and_resume_cannot_release_a_verdict_stop(tmp_path):
    """**这两条路是坏的,而且必须一直是坏的。**

    `ack` 要 `active_wait`(裁决转人没有),`resume` 只在 `paused` 有效。
    钉住它们,是因为「给 resume 放宽状态」看起来像一个更小的修法 —— 而那样
    做会让暂停期的自检(resume 的语义)被一次闸门放行悄悄触发。
    """
    rig = _stop_at_gate(tmp_path)
    assert rig.row()["active_wait"] is None
    rig.store.enqueue_op(rig.conduct_id, "ack", args={"wait_id": "whatever"})
    rig.store.enqueue_op(rig.conduct_id, "resume")
    rig.tick()
    ops = {r.get("op") for r in _rejections(rig)}
    assert {"ack", "resume"} <= ops, _rejections(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "resume" not in OP_VALID_STATUSES or \
        "waiting_operator" not in OP_VALID_STATUSES["resume"]


# ── 1. 留痕 + 持续显示 ─────────────────────────────────────────────────

def test_the_override_is_recorded_and_carries_who_and_why(tmp_path):
    rig = _stop_at_gate(tmp_path)
    _override(rig, reason="谱存到了另一个目录,人工核过 12 条都可用")
    ev = rig.events(kind="decision_overridden")
    assert len(ev) == 1
    p = ev[0]["payload"]
    assert p["by"] == "operator"
    assert "人工核过 12 条" in p["reason"]
    assert p["kind"] == "gate"
    # 放行之后不再卡着 —— 这里的 spec 只有一个阶段,出口闸放行即 completed。
    assert rig.row()["status"] != "waiting_operator"
    # **持续显示**:停在那里的那句话被清掉了,而放行这件事留在审计流里。
    assert rig.row()["status_reason"] == ""


def test_a_reason_is_structurally_required(tmp_path):
    """理由缺席就等于事后没人说得清为什么 —— 在**入队那一层**就拦住。"""
    assert "override_decision" in OPS_REASON_REQUIRED
    rig = _stop_at_gate(tmp_path)
    with pytest.raises(ValueError, match="reason"):
        rig.store.enqueue_op(rig.conduct_id, "override_decision",
                             args={"decision_id": "1"})


def test_a_decision_id_is_structurally_required(tmp_path):
    """没有 decision_id 就没法说清放行的是**哪一次** —— 同 ack 的 wait_id。"""
    rig = _stop_at_gate(tmp_path)
    with pytest.raises(ValueError, match="decision_id"):
        rig.store.enqueue_op(rig.conduct_id, "override_decision",
                             args={"reason": "r"})


# ── 2. 只解这一次判定,不是这道闸 ──────────────────────────────────────

def test_the_same_gate_is_judged_again_next_time(tmp_path):
    """**它不是跳过闸门的万能钥匙。** 放行之后再走到同一道闸,照样重新判。

    放行不在闸门上留任何状态 —— 所以「再判一次」的证据是:**同一道闸再评一次,
    仍然判 wait_operator,而且换了一个新的 decision_id**。
    (线性 spec 里同一道闸只会走到一次,所以这里直接把它再评一遍 —— 那正是
    「下一次走到它」在引擎里的样子。)
    """
    rig = _stop_at_gate(tmp_path, which="entry")
    first = _pending(rig)["decision_id"]
    _override(rig)
    assert rig.row()["status"] == "running"

    spec_obj = rig.spec
    stage_obj = spec_obj.stages[0]
    rig.director._run_gate(rig.conduct_id, rig.row(), spec_obj, stage_obj,
                           stage_obj.entry_gate, "entry", _Rep())
    assert rig.row()["status"] == "waiting_operator", (
        "放行之后这道闸不再判了 —— 那就是跳过了这道闸,不是放行了一次判定")
    again = _pending(rig)
    assert again and again["decision_id"] != first, "第二次判定沿用了旧编号"


def test_an_override_aimed_at_a_stale_decision_is_refused(tmp_path):
    """对着一个已经翻篇的判定说「继续」 —— 与对旧等待点 ack 同一个形状。"""
    rig = _stop_at_gate(tmp_path)
    real = _pending(rig)["decision_id"]
    _override(rig, decision_id=real + 999)
    assert rig.row()["status"] == "waiting_operator", "放行了一个不是当前的判定"
    why = [r["why"] for r in _rejections(rig) if r.get("op") == "override_decision"]
    assert why and "不匹配" in why[0]


# ── 3. 闸门的裁决与人的决定分开记 ─────────────────────────────────────

def test_the_gate_verdict_is_never_rewritten_into_a_pass(tmp_path):
    """**事后对账不能看到一道从不判 fail 的闸。**

    人推翻的是那个裁决,所以那个裁决必须原样留着,而且推翻它的那条记录要说得出
    被推翻的是什么。
    """
    rig = _stop_at_gate(tmp_path)
    _override(rig, reason="核过了")
    gates = rig.events(kind="gate_evaluated")
    assert all(g["payload"]["verdict"] == "wait_operator" for g in gates), (
        "闸门的判定被改写成 pass 了")
    p = rig.events(kind="decision_overridden")[0]["payload"]
    assert p["overridden_verdict"] == "wait_operator"
    assert p["overridden_reason"]


# ── 4. 位置走对了(三个闸位各一条)────────────────────────────────────

@pytest.mark.parametrize("which,expect_stage,expect_step", [
    ("entry", 0, 0),        # 入口闸放行 ⇒ 进第一步
    ("step", 0, 1),         # 步后闸放行 ⇒ 位置在 _step_ok 里已经推进过
])
def test_the_position_after_an_override_matches_a_real_pass(
        tmp_path, which, expect_stage, expect_step):
    """放行走的是**闸门自己判 pass 时那一段代码**(``_gate_passed``)。

    两条路各写一份的话,迟早会在「entry 闸放行之后 step_idx 是 0 还是 -1」
    这种地方分岔,而分岔的那一侧没有测试。
    """
    rig = _stop_at_gate(tmp_path, which=which)
    _override(rig)
    row = rig.row()
    assert (row["stage_idx"], row["step_idx"]) == (expect_stage, expect_step)


def test_an_exit_gate_override_advances_to_the_next_stage(tmp_path):
    gate = rule_gate("g1", selector="S.01")
    s = spec([stage("A", steps=(step("A.01"),), exit_gate=gate),
              stage("B", steps=(step("B.01"),))])
    rig = build(tmp_path, s, executor=FakeExecutor())
    rig.run_until("waiting_operator")
    _override(rig)
    assert rig.row()["stage_idx"] == 1


# ── 5. 不适用的两种停法 ───────────────────────────────────────────────

def test_a_wait_step_stop_has_no_pending_decision(tmp_path):
    """等待步用 ack / waive —— 那是双闸,不是一次裁决。两张卡不会同时出现。"""
    s = spec([stage("S", steps=(retract_step("S.00"), wait_step("S.01"),
                                step("S.02")))])
    rig = build(tmp_path, s)
    rig.run_until("waiting_operator")
    assert rig.row()["active_wait"] is not None
    assert _pending(rig) is None


def test_an_override_with_nothing_pending_is_refused_with_a_trace(tmp_path):
    rig = build(tmp_path, spec([stage("S", steps=(step("S.01"),))]))
    rig.tick()
    rig.store.record(rig.conduct_id, "status_change", changes={
        "status": "waiting_operator", "status_reason": "别的原因停的"})
    rig.store.enqueue_op(rig.conduct_id, "override_decision",
                         args={"decision_id": "1", "reason": "r"})
    rig.tick()
    why = [r["why"] for r in _rejections(rig) if r.get("op") == "override_decision"]
    assert why and "没有停在一个可放行的" in why[0]


# ── 6. 无人值守不多出一条出口 ─────────────────────────────────────────

def test_unattended_never_produces_an_override_by_itself(tmp_path):
    """**没有人在,自然没有人能解。**

    放行是一条**意图**:只有人能把它写进 `conduct_ops`。Director 自己转多少
    个 tick 都不会生成它 —— 这条钉的正是「无人值守路径不会因为它多一条出口」。
    """
    rig = _stop_at_gate(tmp_path, attended=False)
    for _ in range(10):
        rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert rig.events(kind="decision_overridden") == []
    assert _pending(rig) is not None, "无人值守时那条路应当仍然**存在**,只是没人按"


# ── 7. 恢复自检那一档(A3 判针坏那条路)────────────────────────────────

def _recovery_rig(tmp_path, **kw):
    s = spec([stage("S", steps=(step("S.01"), step("S.02")))],
             auto_resume=True, recovery=tip_check())
    return build(tmp_path, s,
                 executor=FakeExecutor({"PreScanCheck": [
                     outcome(True, data={"tip_ready": False})]}),
                 link_probe=lambda: {"main": True},
                 temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)),
                 **kw)


def test_a_recovery_check_that_failed_can_be_released_by_a_person(tmp_path):
    """A3 判针坏之后同样出不去 —— 而那条路是作者接的。

    `takeover → resume` 会再跑一遍同一项、再停在同一个地方,所以它不是出口。
    """
    rig = _recovery_rig(tmp_path)
    rig.tick()
    rig.director.reconcile_after_restart()
    for _ in range(20):
        rig.tick()
        if rig.row()["status"] == "waiting_operator":
            break
    assert "针坏" in rig.row()["status_reason"]
    pend = _pending(rig)
    assert pend and pend["kind"] == "recovery" and pend["item"] == "A3_tip"

    _override(rig, reason="复验扫的是刚打过脉冲的那片,换个点手工看过,针是好的")
    # 回到**自检**继续走,而不是直接冲进 running —— 后面还有别的检查项要跑完。
    assert rig.row()["status"] == "recovery_pending"
    items = [(e["payload"] or {}) for e in rig.events(kind="recovery_item")]
    overridden = [i for i in items if i.get("verdict") == "overridden"]
    assert overridden and overridden[0]["item"] == "A3_tip"
    # **不是记成 pass**:「探针说过了」与「人说算了」是两句话。
    assert not any(i.get("item") == "A3_tip" and i.get("verdict") == "pass"
                   for i in items)


def test_a_released_recovery_item_is_not_run_again(tmp_path):
    """放行之后那一项不再重跑 —— 否则它会立刻再停在同一个地方。"""
    rig = _recovery_rig(tmp_path)
    rig.tick()
    rig.director.reconcile_after_restart()
    for _ in range(20):
        rig.tick()
        if rig.row()["status"] == "waiting_operator":
            break
    before = rig.executor.skills_called().count("PreScanCheck")
    _override(rig, reason="手工看过")
    for _ in range(10):
        rig.tick()
        if rig.row()["status"] == "running":
            break
    assert rig.row()["status"] == "running"
    assert rig.executor.skills_called().count("PreScanCheck") == before


def test_a_blocked_resume_point_can_also_be_released(tmp_path):
    """「续跑点接不上」同样要有出路 —— 而放行的语义是**就地续跑,不回退**。"""
    from test_recovery_m4a import FakeMeta, meta_of

    s = spec([stage("S", steps=(step("S.01", skill="TipPulse"), step("S.02"),
                                step("S.03")))],
             auto_resume=True, recovery=tip_check())
    rig = build(tmp_path, s, executor=FakeExecutor({
        "PreScanCheck": [outcome(True, data={"tip_ready": True})]}),
        link_probe=lambda: {"main": True},
        skill_meta=meta_of({"TipPulse": FakeMeta(caps={"bias_pulse"}),
                            "ScanAt": FakeMeta(), "PreScanCheck": FakeMeta()}),
        temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.tick(3)
    at = rig.row()["step_idx"]
    rig.director.reconcile_after_restart()
    for _ in range(20):
        rig.tick()
        if rig.row()["status"] == "waiting_operator":
            break
    assert "续跑点接不上" in rig.row()["status_reason"]
    _override(rig, reason="脉冲那一步我确认过不用重放")
    # 放行本身**不动位置**;之后的推进是正常执行,所以只能断言「没往回退」。
    assert rig.row()["step_idx"] >= at, "放行把位置往回退了 —— 那正是被否掉的动作"
    for _ in range(10):
        rig.tick()
        if rig.row()["status"] in ("running", "completed"):
            break
    assert rig.row()["status"] in ("running", "completed")
    assert not any((e["payload"] or {}).get("rewound")
                   for e in rig.events(kind="recovery_item")), "放行之后仍然回退了"


def test_a_release_does_not_survive_into_the_next_epoch(tmp_path):
    """一次放行不是永久豁免:重启或绕道 bump 代次之后,世界是新的。"""
    from test_recovery_m4a import FakeMeta, meta_of

    s = spec([stage("S", steps=(step("S.01", skill="TipPulse"), step("S.02"),
                                step("S.03")))],
             auto_resume=True, recovery=tip_check())
    rig = build(tmp_path, s, executor=FakeExecutor({
        "PreScanCheck": [outcome(True, data={"tip_ready": True})]}),
        link_probe=lambda: {"main": True},
        skill_meta=meta_of({"TipPulse": FakeMeta(caps={"bias_pulse"}),
                            "ScanAt": FakeMeta(), "PreScanCheck": FakeMeta()}),
        temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.tick(3)
    rig.director.reconcile_after_restart()
    for _ in range(20):
        rig.tick()
        if rig.row()["status"] == "waiting_operator":
            break
    _override(rig, reason="这一代我确认过")
    assert rig.director._resume_point_overridden(rig.conduct_id) is True
    # 又一次重启 ⇒ A4 bump 代次 ⇒ 上一代那句「就地续跑吧」不再算数。
    rig.store.record(rig.conduct_id, "status_change", changes={
        "evidence_epoch": int(rig.row()["evidence_epoch"]) + 1})
    assert rig.director._resume_point_overridden(rig.conduct_id) is False


# ── 8. 闭集三边对齐 ───────────────────────────────────────────────────

def test_the_new_op_is_declared_everywhere_it_has_to_be(tmp_path):
    """加一个可编辑的键永远是**多边动作**(本仓已经四次)。

    这里钉后端三处;前端那一份由 `frontend/test/conduct.test.ts` 的 parity
    测试对着 `store.OPS` / `director.OP_VALID_STATUSES` 抽取比对。
    """
    from mast.api.schemas_conduct import OpName
    import typing

    assert "override_decision" in OPS
    assert "override_decision" in OP_VALID_STATUSES
    assert "override_decision" in typing.get_args(OpName)
    assert OP_VALID_STATUSES["override_decision"] == frozenset({"waiting_operator"})


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# ── 9. EvidenceSpec 字段投影(同一批的第二件)───────────────────────────

def _gate_with_projection(fields, selector="S.01"):
    from mast.conduct.spec import EvidenceSpec, GateOutcome, GateSpec, RuleLeaf

    return GateSpec(
        gate_id="p", kind="rule",
        evidence=(EvidenceSpec(source="step_data", selector=selector,
                               max_age_s=3600.0, min_epoch="current",
                               fields=tuple(fields)),),
        rule=RuleLeaf("n", ">=", 1),
        routes={"pass": GateOutcome("pass"),
                "fail": GateOutcome("wait_operator", "不通过")},
        unattended_escape="wait_operator", evidence_missing="wait_operator")


def _projected(rig, gate):
    """跑到闸门,把它实际收到的那一包证据取出来。"""
    return rig.director._collect_evidence(rig.conduct_id, rig.row(), gate)


def test_a_projection_keeps_only_the_named_fields(tmp_path):
    """一道闸门问的是一个具体问题,喂给它的应该正好是回答那个问题要的东西。

    整包进 rule 闸门没关系(判据只读它点名的字段),整包进 **llm 闸门的提示词**
    就不是没关系:路由是闭集、模型填不了数,所以它不危险,**但多余字段会把判断
    拉偏**。
    """
    s = spec([stage("S", steps=(step("S.01", produces=("n", "_progress")),
                                step("S.02")))])
    rig = build(tmp_path, s, executor=FakeExecutor({"ScanAt": [
        outcome(True, data={"n": 3, "_progress": {"partial_data": {
            "center_x_m": 1e-9, "bias_v": 1.2, "attempts": [1, 2, 3]}}})]}))
    rig.tick(2)
    values, missing = _projected(rig, _gate_with_projection(["n"]))
    assert values == {"n": 3}
    assert missing == ()
    # 没投影的那一份仍然是整包 —— 默认行为一个字没变。
    whole, _ = _projected(rig, _gate_with_projection([]))
    assert "_progress" in whole


def test_a_projection_that_names_a_missing_field_blames_the_spec(tmp_path):
    """⚠️ **「投影落空」与「这一步没产出」是两句话。**

    前者是 spec 点错了字段,后者是仪器那一侧的事实。运行时两者都走 missing,
    但措辞必须分开 —— 用户照着去查的地方完全不一样。
    """
    s = spec([stage("S", steps=(step("S.01", produces=("n",)), step("S.02")))])
    rig = build(tmp_path, s,
                executor=FakeExecutor({"ScanAt": [outcome(True, data={"n": 3})]}))
    rig.tick(2)
    _values, missing = _projected(rig, _gate_with_projection(["sharpness"]))
    assert missing and "spec 点错了字段" in missing[0]
    assert "没产出" in missing[0]

    # 对照:真的没有这一步的产出时,说的是**另一句**话。
    _v2, m2 = _projected(rig, _gate_with_projection(["n"], selector="S.99"))
    assert m2 and "没有这一步的产出" in m2[0]
    assert "spec 点错了字段" not in m2[0]


def test_the_validator_catches_a_projection_typo_at_approve_time(tmp_path):
    """运行时那句话指向仪器,而根因在模板里 —— 所以要在 approve 时拦。"""
    from mast.conduct import analyses
    from mast.conduct.validator import validate_spec

    s = spec([stage("S", steps=(step("S.01", produces=("n",)),
                                step("S.02", gate=_gate_with_projection(
                                    ["sharpness"]))))])
    rep = validate_spec(s, analyses=analyses.known_names())
    assert "evidence_field_not_produced" in {f.code for f in rep.findings}

    ok = spec([stage("S", steps=(step("S.01", produces=("n",)),
                                 step("S.02", gate=_gate_with_projection(["n"]))))])
    rep2 = validate_spec(ok, analyses=analyses.known_names())
    assert "evidence_field_not_produced" not in {f.code for f in rep2.findings}
