"""Director × 推进 —— tick 顺序、步执行、绑定、闸门、绕道、急停。

设计:``campaign_director_design.md`` §5(转移表)、§6(tick 顺序)、§10。

这一层的命题不是「状态机会转」,而是**每一次转都带着它的理由和留痕**:
heartbeat 与步推进是两件事、绑定取不到就失败不兜底、闸门只看得见当代次的证据、
急停闩挂着就停在原地不清闩。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from _harness import (
    FakeExecutor,
    build,
    outcome,
    retract_step,
    rule_gate,
    spec,
    stage,
    step,
)

from mast.conduct.director import BUSY_YIELD_AFTER_S
from mast.conduct.spec import (
    DetourPolicy,
    EvidenceSpec,
    GateOutcome,
    GateSpec,
    ParamSpec,
    RuleLeaf,
    StageFailPolicy,
    StepSpec,
)
from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata


def _one_step():
    return spec([stage("S", steps=(step("S.01"),))])


# ── tick 顺序 ────────────────────────────────────────────────────────────

def test_every_tick_writes_a_heartbeat(tmp_path):
    """heartbeat 只证明**决策循环活着**,不证明步在推进 —— 两件事,两个指标。"""
    rig = build(tmp_path, _one_step())
    assert rig.row()["heartbeat_at"] is None
    rig.clock.advance(7.0)
    rig.tick()
    assert rig.row()["heartbeat_at"] == rig.clock.t


def test_an_approved_conduct_is_adopted_into_running(tmp_path):
    rig = build(tmp_path, _one_step())
    rig.tick()
    assert rig.row()["status"] == "running"
    assert "adopted" in rig.event_kinds()


def test_a_draft_is_never_adopted_on_its_own(tmp_path):
    """approve 只接受 UI 来源。Director 不替人批。"""
    rig = build(tmp_path, _one_step(), approve=False)
    rig.tick()
    assert rig.row()["status"] == "draft"


def test_no_active_conduct_is_not_an_error(tmp_path):
    rig = build(tmp_path, _one_step())
    rig.store.record(rig.conduct_id, "completed", changes={"status": "completed"})
    rep = rig.tick()
    assert "no_active_conduct" in rep.actions


# ── 步执行 ───────────────────────────────────────────────────────────────

def test_a_step_reaches_the_executor_with_its_resolved_parameters(tmp_path):
    s = spec([stage("S", steps=(
        step("S.01", skill="ScanAt", params={"size_m": 5e-9},
             bindings={"setpoint_a": "params.setpoint_a"}),))],
        params=(ParamSpec(name="setpoint_a", type="float"),))
    rig = build(tmp_path, s, params={"setpoint_a": 5e-11})
    rig.tick(); rig.tick()
    skill, params, run_id = rig.executor.calls[0]
    assert skill == "ScanAt"
    assert params == {"size_m": 5e-9, "setpoint_a": 5e-11}
    assert run_id


def test_a_binding_that_cannot_be_resolved_fails_the_step_with_no_fallback(tmp_path):
    """绑定取不到 = 步失败,**没有默认值兜底**。

    「取不到就用 0」会把一次读失败变成一次看起来完全正常的运行。
    """
    s = spec([stage("S", steps=(
        step("S.01", bindings={"x": "steps.NOBODY.x"}),))])
    rig = build(tmp_path, s)
    rig.tick(); rig.tick()
    # 断言的是「**那一步**没被发下去」，不是「一次调用都没有」——
    # 2026-08-21 引擎补了「进 waiting_operator 之前做确认式退针」，那个
    # SafeRetract 不是步，它恰恰是这条路正确的一半。原来的写法把两件事
    # 混在一个断言里，于是一次安全动作看起来像一次违规派发。
    dispatched = [c for c in rig.executor.calls if c[0] != "SafeRetract"]
    assert dispatched == [], "绑定都没解出来就把步发下去了"
    assert rig.row()["status"] == "waiting_operator"
    assert "没有默认值兜底" in rig.row()["status_reason"]


def test_an_upstream_produce_flows_into_a_downstream_binding(tmp_path):
    s = spec([stage("S", steps=(
        step("S.01", produces=("path",)),
        step("S.02", bindings={"p": "steps.S.01.path"})))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"path": "/scans/1.sxm"})]})
    rig = build(tmp_path, s, executor=ex)
    for _ in range(4):
        rig.tick()
    assert rig.executor.calls[1][1]["p"] == "/scans/1.sxm"


def test_produces_survive_a_restart_because_they_live_in_the_audit_stream(tmp_path):
    """产出不另开一张表:``step_finished`` 本来就是 append-only 的真相,
    而且天然跨重启存活。"""
    s = spec([stage("S", steps=(step("S.01", produces=("path",)),
                                step("S.02", bindings={"p": "steps.S.01.path"})))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"path": "/a.sxm"})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick(); rig.tick()
    flat = rig.director._produced_flat(rig.conduct_id)
    assert flat["steps.S.01.path"] == "/a.sxm"


# ── analysis 步 ──────────────────────────────────────────────────────────

def test_an_analysis_step_runs_in_process_without_the_executor(tmp_path):
    """analysis 步不取仪器令牌 —— 它根本不碰仪器。"""
    s = spec([stage("S", steps=(
        StepSpec(step_id="S.01", kind="analysis", analysis_fn="double",
                 params={"n": 3}, produces=("m",)),))])
    rig = build(tmp_path, s,
                analyses_get=lambda name: (lambda p: {"m": p["n"] * 2}))
    rig.tick(); rig.tick()
    assert rig.executor.calls == []
    assert rig.director._produced_flat(rig.conduct_id)["steps.S.01.m"] == 6


def test_an_analysis_that_does_not_produce_what_it_declared_fails_the_step(tmp_path):
    """声明了产出却没给 —— 下游那条 binding 会在运行时取不到。"""
    s = spec([stage("S", steps=(
        StepSpec(step_id="S.01", kind="analysis", analysis_fn="f",
                 produces=("m",)),))])
    rig = build(tmp_path, s, analyses_get=lambda name: (lambda p: {"other": 1}))
    rig.tick(); rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert "没有产出它声明的" in rig.row()["status_reason"]


def test_a_crashing_analysis_is_a_step_failure_not_a_director_crash(tmp_path):
    s = spec([stage("S", steps=(
        StepSpec(step_id="S.01", kind="analysis", analysis_fn="f"),))])

    def boom(name):
        def _f(p):
            raise ValueError("参数不对")
        return _f

    rig = build(tmp_path, s, analyses_get=boom)
    rig.tick(); rig.tick()
    assert rig.row()["status"] == "waiting_operator"


# ── 重试与阶段失败 ───────────────────────────────────────────────────────

def test_a_step_retries_up_to_its_budget_then_the_stage_policy_decides(tmp_path):
    s = spec([stage("S", steps=(step("S.01", retries=2),),
                    on_fail=StageFailPolicy(then="wait_operator"))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=False, error="质量不合格")]})
    rig = build(tmp_path, s, executor=ex)
    for _ in range(6):
        rig.tick()
    assert ex.skills_called().count("ScanAt") == 3, "重试预算没按 retries 走"
    assert rig.row()["status"] == "waiting_operator"


def test_the_retry_budget_is_counted_from_the_audit_stream_not_memory(tmp_path):
    """重启之后内存计数会归零,于是一个反复失败的步会获得一整套新的重试预算。"""
    s = spec([stage("S", steps=(step("S.01", retries=1),))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=False, error="坏")]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick(); rig.tick()
    assert rig.director._attempts(rig.conduct_id, "S.01") == 1
    rig.tick()
    assert rig.director._attempts(rig.conduct_id, "S.01") == 2


def test_an_optional_step_failure_does_not_take_the_stage_down(tmp_path):
    """谱已经采到了,判不了就是判不了 —— 不该拖垮整段。"""
    s = spec([stage("S", steps=(step("S.01", optional=True), step("S.02")))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=False, error="判不了"),
                                  outcome(ok=True)]})
    rig = build(tmp_path, s, executor=ex)
    for _ in range(5):
        rig.tick()
    assert rig.row()["status"] in ("running", "completed")
    assert ex.skills_called().count("ScanAt") >= 2


def test_a_tip_event_is_recorded_but_the_director_does_not_grab_the_rescue(tmp_path):
    """针尖事件必然挂闩 ⇒ 下一 tick 的 pre-flight 会拦。这里只如实记账,
    **不抢救济**(watchdog 的处置序列是独立的)。"""
    s = spec([stage("S", steps=(step("S.01"),))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=False, tip_event=True)]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick(); rig.tick()
    assert "针尖事件" in rig.row()["status_reason"]
    assert "SafeRetract" not in ex.skills_called()


# ── per-run abort Event 生命周期 ────────────────────────────────────────

def test_the_abort_event_is_registered_during_the_step_and_cleared_after(tmp_path):
    """注册=步启动前;注销=返回后的 finally(成败都清)。

    留着的话,下一次 abort 会置一个早就结束的 run 的 Event,而真正在跑的那个
    没人管。
    """
    seen = {}

    class Spy(FakeExecutor):
        def run(self, skill, params, *, run_id):
            seen["live"] = self.director.aborts.live_run_ids
            seen["active_run_id"] = self.rig.row()["active_run_id"]
            return super().run(skill, params, run_id=run_id)

    ex = Spy()
    rig = build(tmp_path, _one_step(), executor=ex)
    ex.director = rig.director
    ex.rig = rig
    rig.tick(); rig.tick()
    assert len(seen["live"]) == 1, "步跑着的时候没有注册 abort Event"
    assert seen["active_run_id"], "步跑着的时候 active_run_id 是空的"
    assert rig.director.aborts.live_run_ids == (), "步结束了 Event 还留着"
    assert rig.row()["active_run_id"] == ""


def test_the_event_is_cleared_even_when_the_executor_raises(tmp_path):
    from _harness import HangingExecutor
    rig = build(tmp_path, _one_step(), executor=HangingExecutor())
    rig.tick(); rig.tick()
    assert rig.director.aborts.live_run_ids == ()
    assert rig.row()["active_run_id"] == ""


# ── busy / yielding ─────────────────────────────────────────────────────

def test_busy_is_not_a_step_failure(tmp_path):
    """仪器被别人占着不是这一步失败 —— 折叠会消耗掉这一步的重试预算。"""
    s = spec([stage("S", steps=(step("S.01", retries=0),))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=False, busy=True)]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick(); rig.tick(); rig.tick()
    assert rig.row()["status"] == "running", "busy 被当成步失败了"


def test_persistent_busy_yields_the_instrument(tmp_path):
    """连续 busy 超过阈值 ⇒ 让路 + 通知(拒绝不排队的纪律不动)。"""
    s = spec([stage("S", steps=(step("S.01"),))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=False, busy=True)]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick(); rig.tick()
    rig.clock.advance(BUSY_YIELD_AFTER_S + 1)
    rig.tick()
    assert rig.row()["status"] == "yielding"
    assert any(n.kind == "yielding" for n in rig.notifier.sent)


def test_yielding_returns_to_running_when_the_lock_frees_up(tmp_path):
    s = spec([stage("S", steps=(step("S.01"),))])
    free = {"v": False}
    rig = build(tmp_path, s, lock_probe=lambda: free["v"],
                executor=FakeExecutor({"ScanAt": [outcome(ok=False, busy=True)]}))
    rig.tick(); rig.tick()
    rig.clock.advance(BUSY_YIELD_AFTER_S + 1)
    rig.tick()
    assert rig.row()["status"] == "yielding"
    rig.tick()
    assert rig.row()["status"] == "yielding", "锁还占着就不该回去"
    free["v"] = True
    rig.tick()
    assert rig.row()["status"] == "running"


# ── 闸门 ─────────────────────────────────────────────────────────────────

def test_a_passing_exit_gate_advances_the_stage(tmp_path):
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),),
                    exit_gate=rule_gate(selector="S.01")),
              stage("T", steps=(step("T.01"),))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 3})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick()          # adopt
    rig.tick()          # S.01
    rig.tick()          # exit gate
    assert rig.row()["stage_idx"] == 1


def test_a_gate_never_sees_evidence_from_an_older_epoch(tmp_path):
    """结构过滤:闸门收不到跨代次的证据。

    「在自己刚炸出来的坑上判针尖」因此**做不到**,而不是靠闸门作者记得过滤。
    """
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),),
                    exit_gate=rule_gate(selector="S.01"))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 3})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick(); rig.tick()
    # 代次一变,刚才那份产出就不再是「当前代次的证据」
    rig.store.record(rig.conduct_id, "status_change",
                     changes={"evidence_epoch": 1})
    rig.tick()
    row = rig.row()
    assert row["status"] == "waiting_operator"
    ev = rig.events(kind="gate_evaluated")[-1]["payload"]
    assert ev["evidence_missing"] is True
    assert any("代证据" in m for m in ev["missing"])


def test_a_gate_never_sees_stale_evidence(tmp_path):
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),),
                    exit_gate=rule_gate(selector="S.01", max_age_s=60.0))])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 3})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick(); rig.tick()
    rig.clock.advance(120.0)
    rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert rig.events(kind="gate_evaluated")[-1]["payload"]["evidence_missing"]


def test_an_unwired_evidence_source_is_reported_missing_not_ignored(tmp_path):
    """本期没接的证据源(verify_verdict / frame_metrics / monitor_events)
    报成缺席,闸门于是走保守去向 —— 而不是当没这回事直接判。"""
    gate = GateSpec(gate_id="g", kind="rule",
                    evidence=(EvidenceSpec(source="monitor_events"),),
                    rule=RuleLeaf("anything", "exists"),
                    routes={"pass": GateOutcome("pass"),
                            "fail": GateOutcome("wait_operator")})
    s = spec([stage("S", steps=(step("S.01"),), exit_gate=gate)])
    rig = build(tmp_path, s)
    for _ in range(4):
        rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert any("未接入" in m
               for m in rig.events(kind="gate_evaluated")[-1]["payload"]["missing"])


def test_the_llm_wake_budget_is_spent_at_most_once_per_stage(tmp_path):
    node = {"id": "n", "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}
    gate = GateSpec(gate_id="lg", kind="llm", llm_node=node,
                    evidence=(EvidenceSpec(source="step_data", selector="S.01"),),
                    routes={"go": GateOutcome("pass"),
                            "hold": GateOutcome("wait_operator")})
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),), exit_gate=gate)])
    calls = {"n": 0}

    def decide(node_, inputs):
        calls["n"] += 1
        return {"route": "hold", "reason": "再看看"}

    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 1})]})
    rig = build(tmp_path, s, executor=ex, decide_route=decide)
    for _ in range(4):
        rig.tick()
    assert calls["n"] == 1
    assert rig.row()["llm_wakes"] == {"S": 1}
    # 预算用完之后再评一次:**判不了**,而不是再唤一次,也不是就当过了
    rig.store.record(rig.conduct_id, "status_change",
                     changes={"status": "running", "status_reason": ""})
    rig.tick()
    assert calls["n"] == 1
    assert rig.row()["status"] == "waiting_operator"


def test_the_spent_budget_says_so_instead_of_blaming_the_wiring(tmp_path):
    """预算用完与「判决器没接上」去向完全一样,**但说的话不一样**。

    以前两条都印「llm 判决器未接入 ⇒ 判不了」。那句话印在一台明明接好了判决器
    的机器的面板上,会把用户送去查一根没有断的线 —— 而真正该看的是「这一段
    已经问过一次了」。兜底值合理得让人看不出兜底发生过,这是同一族。
    """
    node = {"id": "n", "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}
    gate = GateSpec(gate_id="lg", kind="llm", llm_node=node,
                    evidence=(EvidenceSpec(source="step_data", selector="S.01"),),
                    routes={"go": GateOutcome("pass"),
                            "hold": GateOutcome("wait_operator")})
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),), exit_gate=gate)])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 1})]})
    rig = build(tmp_path, s, executor=ex,
                decide_route=lambda n, i: {"route": "hold", "reason": "再看看"})
    for _ in range(4):
        rig.tick()
    rig.store.record(rig.conduct_id, "status_change",
                    changes={"status": "running", "status_reason": ""})
    rig.tick()
    last = rig.events(kind="gate_evaluated")[-1]["payload"]
    assert "预算已用完" in last["reason"] and "1/1" in last["reason"]
    assert "未接入" not in last["reason"]
    assert rig.row()["status"] == "waiting_operator"


def test_an_unwired_seat_still_says_it_is_unwired(tmp_path):
    """反面:真的没接判决器时,那句话仍然要是「未接入」。

    上一条测试若用一个 catch-all 的新文案替掉两者,这条会红。
    """
    node = {"id": "n", "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}
    gate = GateSpec(gate_id="lg", kind="llm", llm_node=node,
                    evidence=(EvidenceSpec(source="step_data", selector="S.01"),),
                    routes={"go": GateOutcome("pass"),
                            "hold": GateOutcome("wait_operator")})
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),), exit_gate=gate)])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 1})]})
    rig = build(tmp_path, s, executor=ex)      # decide_route 不注入
    for _ in range(4):
        rig.tick()
    last = rig.events(kind="gate_evaluated")[-1]["payload"]
    assert "未接入" in last["reason"]
    assert rig.row()["status"] == "waiting_operator"
    assert last["llm"]["wakes_used"] == 0, "没唤成也不该记一次唤醒"
    assert rig.row()["llm_wakes"] == {}


def test_a_llm_verdict_carries_who_answered_it_into_the_audit_stream(tmp_path):
    """**判决要留痕到人眼前。**

    少了这一块,一条 LLM 判决在面板闸门史上与一条 rule 判定长得一模一样 ——
    而这两者事后要做的核对完全不同(一个查判据,一个查那次判决本身)。
    """
    node = {"id": "n", "responsibility": "判这一段还值不值得接着测",
            "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}
    gate = GateSpec(gate_id="lg", kind="llm", llm_node=node,
                    evidence=(EvidenceSpec(source="step_data", selector="S.01"),),
                    routes={"go": GateOutcome("pass"),
                            "hold": GateOutcome("wait_operator")})
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),), exit_gate=gate)])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 1})]})
    rig = build(tmp_path, s, executor=ex, decide_route=lambda n, i: {
        "route": "go", "reason": "两个偏压都有分辨", "escaped": False,
        "parse_path": "structured", "model": "kimi-k3", "duration_ms": 812})
    for _ in range(4):
        rig.tick()
    p = rig.events(kind="gate_evaluated")[-1]["payload"]
    assert p["kind"] == "llm"
    assert p["llm"]["model"] == "kimi-k3"
    assert p["llm"]["parse_path"] == "structured"
    assert p["llm"]["wakes_used"] == 1 and p["llm"]["wakes_max"] == 1
    assert p["llm"]["responsibility"] == "判这一段还值不值得接着测"


def test_a_seat_failure_reaches_the_panel_as_did_not_judge_not_as_a_verdict(tmp_path):
    """席位抛 ``SeatUnavailable``(超时 / 建不出模型 / 返回值不在闭集里)⇒
    **判不了**,而且面板上说得出是哪一种。

    这一条把 ``llm_seat`` 与 ``director`` 串起来测:两边各自绿、中间那一截断了,
    是本仓反复出现的形状。这里走的是真的 ``make_decide_route``,只把
    ``decide`` 与 ``log`` 换成替身。
    """
    from mast.conduct.llm_seat import make_decide_route

    node = {"id": "n", "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}
    gate = GateSpec(gate_id="lg", kind="llm", llm_node=node,
                    evidence=(EvidenceSpec(source="step_data", selector="S.01"),),
                    routes={"go": GateOutcome("pass"),
                            # escape 路由通向 detour —— 一次 provider 500
                            # **不许**发出「半夜换样品」。
                            "hold": GateOutcome("detour")})
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),), exit_gate=gate),
              stage("R", steps=(retract_step("R.00"),))],
             detour=DetourPolicy(target_stage="R"))
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 1})]})
    seat = make_decide_route(
        model_factory=lambda node_: object(),
        log=lambda rec: None,
        decide=lambda node_, inputs, model=None: {
            "route": "hold", "escaped": True,
            "escape_reason": "error: 502 Bad Gateway"})
    rig = build(tmp_path, s, executor=ex, decide_route=seat)
    for _ in range(4):
        rig.tick()
    p = rig.events(kind="gate_evaluated")[-1]["payload"]
    assert p["verdict"] == "wait_operator", "provider 故障不该发出 detour"
    assert "502" in p["llm"]["unavailable"]
    assert rig.row()["status"] == "waiting_operator"
    assert not rig.row().get("detour"), "一次 provider 500 把机器送进了修针段"


def test_a_rule_gate_leaves_no_llm_block_at_all(tmp_path):
    """rule 闸门不带 ``llm`` 块 —— 给它一个空 dict 兜底,面板就分不出
    「这条不是 LLM 判的」与「是 LLM 判的但没记到模型名」。"""
    s = spec([stage("S", steps=(step("S.01", produces=("n",)),),
                    exit_gate=rule_gate())])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 1})]})
    rig = build(tmp_path, s, executor=ex)
    for _ in range(4):
        rig.tick()
    p = rig.events(kind="gate_evaluated")[-1]["payload"]
    assert p["kind"] == "rule"
    assert "llm" not in p


# ── 急停闩 ───────────────────────────────────────────────────────────────

def test_a_latched_estop_stops_the_conduct_where_it_stands(tmp_path):
    rig = build(tmp_path, _one_step())
    rig.tick()
    rig.latch.latched = True
    rig.latch.why = "环境监控判故障"
    rep = rig.tick()
    row = rig.row()
    assert row["status"] == "halted_estop"
    assert row["status_reason"] == "环境监控判故障"
    assert "halted_estop" in rep.actions


def test_the_director_never_clears_the_latch_itself(tmp_path):
    """急停闩是**消费者**关系:查闩、挂着就停、**不清闩**。

    清闩是人的动作(safety 路由),Director 自己清等于自己给自己放行。
    """
    rig = build(tmp_path, _one_step())
    rig.tick()
    rig.latch.latched = True
    rig.tick(); rig.tick()
    assert rig.latch.latched is True
    assert rig.row()["status"] == "halted_estop"


def test_a_cleared_latch_goes_through_recovery_not_straight_back_to_running(tmp_path):
    """闩清 ≠ 针没事。"""
    rig = build(tmp_path, _one_step())
    rig.tick()
    rig.latch.latched = True
    rig.tick()
    rig.latch.latched = False
    rig.tick()
    assert rig.row()["status"] == "recovery_pending"
    assert "闩清不等于针没事" in rig.row()["status_reason"]


def test_an_estop_without_a_reason_still_records_that_it_had_none(tmp_path):
    """一个没有「为什么」的停,人只能靠猜或者重启进程 —— 至少要说清「闩没给」。"""
    rig = build(tmp_path, _one_step())
    rig.tick()
    rig.latch.latched = True
    rig.latch.why = ""
    rig.tick()
    assert "闩没给原因" in rig.row()["status_reason"]


# ── SAFE 模式对账 ───────────────────────────────────────────────────────

def _meta(name, level=SafetyLevel.DANGEROUS, caps=("bias_pulse",)):
    return SkillMetadata(name=name, category=SkillCategory.WRITE,
                         safety_level=level, capabilities=frozenset(caps))


def test_a_dangerous_step_in_an_undeclared_stage_stops_before_running(tmp_path):
    """**不静默跳过** —— 停下来问人。"""
    s = spec([stage("S", steps=(step("S.01", skill="TipPulse"),))])
    rig = build(tmp_path, s, skill_meta=lambda n: _meta(n))
    rig.tick(); rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert rig.executor.calls == [], "危险步在没声明 capability 的阶段跑掉了"


def test_a_declared_capability_lets_the_dangerous_step_through(tmp_path):
    s = spec([stage("S", steps=(step("S.01", skill="TipPulse"),),
                    capabilities=frozenset({"bias_pulse"}))])
    rig = build(tmp_path, s, skill_meta=lambda n: _meta(n))
    rig.tick(); rig.tick()
    assert rig.executor.skills_called() == ["TipPulse"]


def test_an_unavailable_reconciliation_is_reported_not_assumed_ok(tmp_path):
    """没注入 skill_meta ⇒ 这条检查**没跑**,如实进报告 —— 「没检查」不许
    长得像「检查通过」。"""
    rig = build(tmp_path, _one_step())
    rep = rig.tick()
    assert any("SAFE 模式对账" in s for s in rep.not_checked)


# ── 绕道 ─────────────────────────────────────────────────────────────────

def _detour_spec(max_detours=3):
    fail_gate = rule_gate("g", selector="M.01", fail_verdict="detour")
    return spec([
        stage("R", steps=(retract_step("R.00"), step("R.01"))),
        stage("M", steps=(step("M.01", produces=("n",)),), exit_gate=fail_gate),
    ], detour=DetourPolicy(target_stage="R",
                           max_detours_per_conduct=max_detours))


def test_a_detour_bumps_the_evidence_epoch_and_jumps_to_the_repair_stage(tmp_path):
    """进 detour 即 bump epoch:坏针之前采的证据全部作废。"""
    s = _detour_spec()
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick()          # adopt 会把位置重置到 0,所以摆位要排在它之后
    rig.store.record(rig.conduct_id, "status_change", changes={"stage_idx": 1})
    for _ in range(6):          # 一进绕道就停下来看落点
        rig.tick()
        if rig.row()["detour"]:
            break
    row = rig.row()
    assert row["evidence_epoch"] == 1
    assert row["stage_idx"] == 0 and row["step_idx"] == 0, "没跳到修针段的第 0 步"
    assert row["detour"]["return_stage_idx"] == 1


def test_a_detour_lands_on_a_confirmed_retract(tmp_path):
    """绕道的触发条件就是「针可能坏了」,第一个动作必须是把针拿开。"""
    s = _detour_spec()
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick()          # adopt 会把位置重置到 0,所以摆位要排在它之后
    rig.store.record(rig.conduct_id, "status_change", changes={"stage_idx": 1})
    for _ in range(6):
        rig.tick()
    assert "SafeRetract" in ex.skills_called()


def test_the_detour_circuit_breaker_stops_the_ping_pong(tmp_path):
    """修针 ping-pong 会烧一整夜机时和一根针。"""
    s = _detour_spec(max_detours=0)
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick()          # adopt 会把位置重置到 0,所以摆位要排在它之后
    rig.store.record(rig.conduct_id, "status_change", changes={"stage_idx": 1})
    for _ in range(4):
        rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert "熔断" in rig.row()["status_reason"]


def test_a_detour_verdict_without_a_repair_stage_asks_for_a_human(tmp_path):
    """没有可去之处的绕道就是死路 —— 转成「请人来修针」。"""
    gate = rule_gate("g", selector="M.01", fail_verdict="detour")
    s = spec([stage("M", steps=(step("M.01", produces=("n",)),), exit_gate=gate)])
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})]})
    rig = build(tmp_path, s, executor=ex)
    for _ in range(4):
        rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert "没有修针段" in rig.row()["status_reason"]


def test_the_conduct_returns_from_a_detour_and_finishes(tmp_path):
    s = _detour_spec()
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0}),
                                  outcome(ok=True, data={"n": 5})]})
    rig = build(tmp_path, s, executor=ex)
    rig.tick()          # adopt 会把位置重置到 0,所以摆位要排在它之后
    rig.store.record(rig.conduct_id, "status_change", changes={"stage_idx": 1})
    for _ in range(14):
        rig.tick()
    kinds = rig.event_kinds()
    assert "detour_entered" in kinds and "detour_returned" in kinds
    assert rig.row()["detour"] is None


# ── 完成 ─────────────────────────────────────────────────────────────────

def test_a_conduct_that_runs_out_of_stages_completes_and_frees_the_slot(tmp_path):
    rig = build(tmp_path, _one_step())
    for _ in range(5):
        rig.tick()
    row = rig.row()
    assert row["status"] == "completed"
    assert row["active_slot"] is None


# ── 预算 ─────────────────────────────────────────────────────────────────

def test_an_exceeded_budget_asks_for_a_human_instead_of_hard_stopping(tmp_path):
    """一个跑了六小时的实验因为差几毛钱被砍掉,比超支更贵。"""
    rig = build(tmp_path, _one_step(), cost_reader=lambda cid: 999.0)
    rig.tick(); rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert "不硬停" in rig.row()["status_reason"]


def test_an_unreadable_spend_is_not_treated_as_zero(tmp_path):
    """读不到花销 ≠ 花了 0。如实进「没检查」,不假装在管预算。"""
    rig = build(tmp_path, _one_step(), cost_reader=lambda cid: None)
    rep = rig.tick()
    rep2 = rig.tick()
    assert any("读不到花销" in s for s in rep2.not_checked)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
