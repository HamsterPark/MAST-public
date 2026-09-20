"""M4-a:启动清算、A1/A2/A3 真路、以及「从上一个已通过的闸门续跑」。

设计:``campaign_director_design.md`` §0(一句话里那半句「重启经显式恢复自检从
上一个已通过闸门续跑」)、§5(RECOVERY_PENDING 各行)、§8(两档清单 + 预授权位
语义表)。M1-b 已经钉住的部分在 ``test_director_recovery.py``,这里只钉新的。

四条命题:

1. **清算按状态枚举,不按活跃位问一句** —— 活跃位对不上账的那一刻正是清算存在
   的理由,而那时 ``active()`` 回 ``None``,「读不到」会被读成「没有」;
2. **A1/A2/A3 三个真探针,每个都有第三种答案** —— 通/不通/**探不出来**,
   够新/超窗/**读不到**,好/坏/**判不了**;
3. **A3 判针坏 → 停下来问人,不进修针段**(裁决,不是没做完);
4. **续跑点退到本阶段最近一次闸门放行之后**,而这条回退**不许跨过**等人步或
   会改变表面的动作;退不回去就说出来,不就地续跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from _harness import (
    FakeReading,
    FakeTemperature,
    build,
    rule_gate,
    spec,
    stage,
    step,
    tip_check,
    wait_step,
)

from mast.conduct import recovery
from mast.conduct.spec import (
    AT_ENTRY_GATE,
    ConditionSpec,
    RecoveryPolicy,
    RuleLeaf,
    StepSpec,
    WaitSpec,
)


class FakeMeta:
    """技能元数据替身 —— 只需要 ``capabilities`` 与 ``safety_level`` 两项。"""

    class _Level:
        def __init__(self, value):
            self.value = value

    def __init__(self, caps=(), level="auto"):
        self.capabilities = frozenset(caps)
        self.safety_level = self._Level(level)


def meta_of(table: dict):
    return lambda name: table.get(name)


def _recover(rig, limit: int = 30):
    for _ in range(limit):
        rig.tick()
        if rig.row()["status"] != "recovery_pending":
            return
    raise AssertionError("自检没在 %d 个 tick 内走完" % limit)


def _items(rig) -> list[str]:
    return [(e["payload"] or {}).get("item")
            for e in rig.events(kind="recovery_item")]


def _detail(rig, item: str) -> str:
    for ev in rig.events(kind="recovery_item"):
        p = ev["payload"] or {}
        if p.get("item") == item:
            return str(p.get("detail") or "")
    return ""


# ── 1. 启动清算:按状态枚举 ─────────────────────────────────────────────

def test_the_sweep_enumerates_by_status_not_by_who_holds_the_slot(tmp_path):
    """**活跃位对不上账的那一刻,正是清算存在的理由。**

    ``active()`` 问的是「谁占着活跃位」。一份非终态却没占位的孤儿(崩在中间的
    一次写、外部改库、旧版本写下的行)会让它回 ``None`` —— 而 ``None`` 会被读成
    「没有要清算的」。一份跑了两天的 conduct 就那样安静地停在那里。
    """
    rig = build(tmp_path, spec([stage("S", steps=(step("S.01"),))]),
                recovery_probe=lambda item: "pass")
    rig.tick()
    assert rig.row()["status"] == "running"
    # 手工制造孤儿:非终态,活跃位空。
    import sqlite3
    with sqlite3.connect(str(tmp_path / "conduct.db")) as conn:
        conn.execute("UPDATE conducts SET active_slot=NULL WHERE conduct_id=?",
                     (rig.conduct_id,))
    assert rig.store.active() is None, "前提没成立:孤儿没造出来"

    rep = rig.director.reconcile_after_restart()
    assert rig.row()["status"] == "recovery_pending", (
        "按活跃位问的那一版会在这里静默地什么都不做")
    assert "restart_to_recovery" in rep.actions


def test_an_orphan_that_a_person_paused_is_reported_not_silently_fixed(tmp_path):
    """PAUSED 的孤儿:**说出来,但不动它**。

    尊重人的暂停是一条规则,单活跃不变式对不上账是另一件事。两条都成立时该做的
    是把第二件报出来 —— 而不是替它改一列(状态改动只有 ``record`` 一扇门)。
    """
    rig = build(tmp_path, spec([stage("S", steps=(step("S.01"),))]))
    rig.tick()
    rig.store.record(rig.conduct_id, "status_change",
                     changes={"status": "paused", "status_reason": "人按了暂停"})
    import sqlite3
    with sqlite3.connect(str(tmp_path / "conduct.db")) as conn:
        conn.execute("UPDATE conducts SET active_slot=NULL WHERE conduct_id=?",
                     (rig.conduct_id,))
    rep = rig.director.reconcile_after_restart()
    assert rig.row()["status"] == "paused"
    assert any("活跃位" in n for n in rep.not_checked), rep.not_checked


def test_a_conduct_that_cannot_be_moved_says_so_loudly(tmp_path):
    """搬不动就**大声说**,不吞。

    吞掉的话屏幕上会有一份永远停在旧状态、而且没有任何解释的 conduct ——
    「能挂不能解」的又一种写法。
    """
    rig = build(tmp_path, spec([stage("S", steps=(step("S.01"),))]))
    rig.tick()

    def boom(*a, **kw):
        raise RuntimeError("另一个 conduct 占着活跃位")

    rig.store.record = boom
    rep = rig.director.reconcile_after_restart()
    assert any("搬不进" in n for n in rep.not_checked), rep.not_checked
    assert any(n.kind == "recovery_blocked" for n in rig.notifier.sent)


# ── 2. A1:连接探活的三态 ───────────────────────────────────────────────

def _link_rig(tmp_path, probe, **kw):
    return build(tmp_path, spec([stage("S", steps=(step("S.01"),))],
                                auto_resume=True, recovery=tip_check()),
                 link_probe=probe, **kw)


def test_a_role_that_is_down_points_at_nanonis_and_the_service_locator(tmp_path):
    rig = _link_rig(tmp_path, lambda: {"main": True, "scan": False})
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "NI Service Locator" in rig.row()["status_reason"]
    assert "netstat 会误导" in rig.row()["status_reason"]


def test_a_role_that_cannot_be_probed_is_not_a_role_that_is_down(tmp_path):
    """**探不出来 ≠ 不通。** 两者要人做的事不一样:一个去看仪器,一个去看接线。"""
    rig = _link_rig(tmp_path, lambda: {"main": True, "scan": None})
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "探不出来" in _detail(rig, "A1_link")
    assert "NI Service Locator" not in _detail(rig, "A1_link")


def test_an_empty_probe_result_is_unreadable_not_all_clear(tmp_path):
    """一个 role 都没报 ⇒ **读不到**,不是「全都好」。"""
    rig = _link_rig(tmp_path, lambda: {})
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "读不到" in _detail(rig, "A1_link")


def test_all_roles_responding_passes(tmp_path):
    rig = _link_rig(tmp_path, lambda: {"main": True, "scan": True},
                    temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert "2 个 role 全部响应" in _detail(rig, "A1_link")


# ── 3. A2:温度的两问 ───────────────────────────────────────────────────

def _temp_spec(ceiling_ref="", with_wait=True):
    steps = [step("S.01")]
    if with_wait:
        steps = [
            StepSpec(step_id="S.00_retract", kind="skill", skill="SafeRetract"),
            StepSpec(step_id="S.00b_wait", kind="wait", wait=WaitSpec(
                kind="condition", message="等降温",
                condition=ConditionSpec(signal="temperature_k", op="<=",
                                        value=300.0, stale_after_s=600.0,
                                        value_ref="params.work_k"))),
            step("S.01"),
        ]
    return spec([stage("S", steps=tuple(steps))], auto_resume=True,
                recovery=RecoveryPolicy(temp_ceiling_ref=ceiling_ref))


def test_a_stale_reading_is_unreadable_not_a_temperature_that_is_wrong(tmp_path):
    """**读不到既不是到了也不是没到。**"""
    rig = build(tmp_path, _temp_spec(with_wait=False),
                link_probe=lambda: {"main": True},
                temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=99999.0)))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "另一个串口程序" in _detail(rig, "A2_temp")


def test_a_reading_with_no_value_never_becomes_zero_kelvin(tmp_path):
    """没有值 ⇒ 读不到。**不是 0 K** —— 0 K 会让 ``<= 5 K`` 立刻成立。"""
    rig = build(tmp_path, _temp_spec(with_wait=False),
                link_probe=lambda: {"main": True},
                temperature=FakeTemperature(FakeReading(reason="no_sensor")))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert "读不到" in _detail(rig, "A2_temp")


def test_a_window_nobody_declared_is_reported_as_not_checked(tmp_path):
    """**没问过 ≠ 问过了。** 没有声明工作点就说没检查,而不是印一个「通过」。"""
    rig = build(tmp_path, _temp_spec(with_wait=False),
                link_probe=lambda: {"main": True},
                temperature=FakeTemperature(FakeReading(value_k=290.0, age_s=1.0)))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    detail = _detail(rig, "A2_temp")
    assert "没检查" in detail, detail
    assert "没有声明过工作温度上限" in detail


def test_the_declared_working_point_comes_from_the_operators_own_number(tmp_path):
    """窗从 spec 自己声明的等待条件推得,而那个数是**用户填的**。

    不是发明一个数:那道等待条件的语义逐字是「换完样品之后等温度回到工作点」,
    回答的正是 A2 要问的问题。
    """
    rig = build(tmp_path, _temp_spec(), params={"work_k": 5.2},
                link_probe=lambda: {"main": True},
                temperature=FakeTemperature(FakeReading(value_k=77.0, age_s=1.0)))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    detail = _detail(rig, "A2_temp")
    assert "超出声明的工作点" in detail and "5.2" in detail


def test_two_different_declared_ceilings_refuse_to_pick_one(tmp_path):
    """声明了两个不同的上限 ⇒ **不猜**,报没检查。

    猜一个(取最小?取当前阶段的?)会让 A2 稳定地在量一个不是目标的东西。
    """
    s = spec([stage("A", steps=(
        StepSpec(step_id="A.00", kind="skill", skill="SafeRetract"),
        StepSpec(step_id="A.01", kind="wait", wait=WaitSpec(
            kind="condition", message="w",
            condition=ConditionSpec(signal="temperature_k", op="<=", value=5.0,
                                    stale_after_s=600.0))))),
        stage("B", steps=(
            StepSpec(step_id="B.00", kind="skill", skill="SafeRetract"),
            StepSpec(step_id="B.01", kind="wait", wait=WaitSpec(
                kind="condition", message="w",
                condition=ConditionSpec(signal="temperature_k", op="<=", value=77.0,
                                        stale_after_s=600.0)))))])
    window = recovery.temperature_window(s, {})
    assert not window.declared
    assert "不止一个" in window.source


def test_a_named_ceiling_that_the_params_do_not_carry_is_unreadable(tmp_path):
    """模板指名了一个参数而 conduct 参数里没有它 —— 「填了但没进去」那族。"""
    s = _temp_spec(ceiling_ref="params.nope", with_wait=False)
    window = recovery.temperature_window(s, {})
    assert not window.declared
    assert "取不到" in window.source


# ── 4. A3:三态,以及那个裁决 ───────────────────────────────────────────

def _a3_rig(tmp_path, outcomes, **kw):
    from _harness import FakeExecutor, outcome

    return build(tmp_path,
                 spec([stage("S", steps=(step("S.01"),))], auto_resume=True,
                      recovery=tip_check()),
                 executor=FakeExecutor({"PreScanCheck": outcomes}),
                 link_probe=lambda: {"main": True},
                 temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)),
                 **kw)


def test_a_declared_tip_check_actually_runs_through_the_executor(tmp_path):
    """A3 是**步**,不是回调:它走 ``executor.run``(完整安全管道 + 取令牌)。"""
    from _harness import outcome

    rig = _a3_rig(tmp_path, [outcome(True, data={"tip_ready": True})])
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert "PreScanCheck" in rig.executor.skills_called()
    assert rig.row()["status"] == "running"
    assert "针尖复验通过" in _detail(rig, "A3_tip")


def test_a_spec_that_never_declared_a_tip_check_is_unreadable_not_a_pass(tmp_path):
    """**不填 = 不知道。** 一个「没声明就当过了」的缺省会让任何一份没接线的模板
    在重启后自动续跑。"""
    rig = build(tmp_path, spec([stage("S", steps=(step("S.01"),))],
                               auto_resume=True),
                link_probe=lambda: {"main": True},
                temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "没有声明恢复期针尖复验" in _detail(rig, "A3_tip")


def test_an_inconclusive_reading_is_not_a_broken_tip(tmp_path):
    """``tip_ready=None`` 是**判不了**,不是「针坏了」。

    折叠成 False 的后果不是判错一次 —— 是把「这一帧没看清」说成「针坏了」,
    而那句话会把用户送去换样品。
    """
    from _harness import outcome

    rig = _a3_rig(tmp_path, [outcome(True, data={"tip_ready": None})])
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    detail = _detail(rig, "A3_tip")
    assert "判不了" in detail
    assert "针坏" not in detail


def test_an_inconclusive_reading_is_retried_once_before_giving_up(tmp_path):
    """判不了先整串重跑一次(§8「换位重试 1 次」)—— 换位是复验步自己的事。"""
    from _harness import outcome

    rig = _a3_rig(tmp_path, [outcome(True, data={"tip_ready": None}),
                             outcome(True, data={"tip_ready": True})])
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.executor.skills_called().count("PreScanCheck") == 2
    assert rig.row()["status"] == "running"


def test_a_tip_check_that_crashes_is_undecidable_not_a_broken_tip(tmp_path):
    """复验步自己跑挂了 ⇒ 判不了。一次 executor 失败与一帧看不清是两件事。"""
    from _harness import outcome

    rig = _a3_rig(tmp_path, [outcome(False, error="扫描没起来"),
                             outcome(False, error="扫描没起来")])
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "没跑成" in _detail(rig, "A3_tip")
    assert "针坏" not in _detail(rig, "A3_tip")


def test_an_instrument_busy_tip_check_never_spends_the_retry_budget(tmp_path):
    """仪器被别的链路占着 **不是** 一次判决 —— 下一 tick 再试,不吃重试预算。"""
    from _harness import outcome

    rig = _a3_rig(tmp_path, [outcome(False, busy=True),
                             outcome(True, data={"tip_ready": True})])
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "running"
    # busy 那次不该产生 A3_tip_retry 记录。
    assert "A3_tip_retry" not in _items(rig)


def test_a_tip_judged_bad_after_a_restart_stops_instead_of_detouring(tmp_path):
    """⚠️ **裁决:A3 判针坏 → 停下来问人,不进修针段。**(M4-a)

    这是 ``test_recovery_tip_fail_still_has_no_producer`` 那条报警的正向那一半。
    完整理由在 ``director._tip_bad_reason`` 的 docstring 里;这里钉三件事:

    * 去向是 WAITING_OPERATOR,**不是** detour;
    * 一条 ``detour_entered`` 事件都不许有;
    * 那句话里说得出**为什么**(修针段第三步就是没有上限的等人换样品)。
    """
    from _harness import FakeExecutor, outcome
    from mast.conduct.spec import DetourPolicy

    detour_target = stage("FIX", steps=(step("FIX.00", skill="SafeRetract"),),
                          entered_only_by_detour=True)
    s = spec([detour_target, stage("S", steps=(step("S.01"),))],
             detour=DetourPolicy(target_stage="FIX",
                                 triggers=frozenset({"recovery_tip_fail"})),
             auto_resume=True, recovery=tip_check())
    rig = build(tmp_path, s,
                executor=FakeExecutor({"PreScanCheck": [
                    outcome(True, data={"tip_ready": False})]}),
                link_probe=lambda: {"main": True},
                temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert rig.events(kind="detour_entered") == [], "恢复路径自己绕进了修针段"
    reason = rig.row()["status_reason"]
    assert "不自动进修针段" in reason
    assert "等人换样品" in reason


def test_a_bad_verdict_on_an_unvetted_spot_says_it_might_be_the_surface(tmp_path):
    """选点时地图读不到 ⇒ 这个「坏」也可能是表面。**说出来。**"""
    from _harness import FakeExecutor, outcome

    steps = (StepSpec(step_id="R.00_spot", kind="skill", skill="FindCleanSpot",
                      produces=("x_m", "map_known")),
             StepSpec(step_id="R.01_scan", kind="composite", skill="PreScanCheck",
                      bindings={"center_x_m": "steps.R.00_spot.x_m"},
                      produces=("tip_ready",)))
    s = spec([stage("S", steps=(step("S.01"),))], auto_resume=True,
             recovery=tip_check(steps=steps))
    rig = build(tmp_path, s, executor=FakeExecutor({
        "FindCleanSpot": [outcome(True, data={"x_m": 1e-9, "map_known": False})],
        "PreScanCheck": [outcome(True, data={"tip_ready": False})]}),
        link_probe=lambda: {"main": True},
        temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert "map_known=false" in rig.row()["status_reason"]


def test_a_sibling_binding_inside_the_tip_check_resolves_by_its_plain_name(tmp_path):
    """复验步之间的绑定写的是**本名** —— 执行体加的前缀不该漏到模板里。"""
    from _harness import FakeExecutor, outcome

    steps = (StepSpec(step_id="R.00_spot", kind="skill", skill="FindCleanSpot",
                      produces=("x_m",)),
             StepSpec(step_id="R.01_scan", kind="composite", skill="PreScanCheck",
                      bindings={"center_x_m": "steps.R.00_spot.x_m"},
                      produces=("tip_ready",)))
    s = spec([stage("S", steps=(step("S.01"),))], auto_resume=True,
             recovery=tip_check(steps=steps))
    rig = build(tmp_path, s, executor=FakeExecutor({
        "FindCleanSpot": [outcome(True, data={"x_m": 4.2e-9})],
        "PreScanCheck": [outcome(True, data={"tip_ready": True})]}),
        link_probe=lambda: {"main": True},
        temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    scan = [c for c in rig.executor.calls if c[0] == "PreScanCheck"][0]
    assert scan[1]["center_x_m"] == pytest.approx(4.2e-9)


def test_a_waiting_conduct_still_never_touches_the_tip_check(tmp_path):
    """等待态重启只做无接触档 —— A3 声明了也不跑。"""
    from _harness import FakeExecutor, retract_step

    s = spec([stage("S", steps=(retract_step("S.00"), wait_step("S.01"),
                                step("S.02")))],
             recovery=tip_check())
    rig = build(tmp_path, s, executor=FakeExecutor(),
                link_probe=lambda: {"main": True},
                temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.run_until("waiting_operator")
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "PreScanCheck" not in rig.executor.skills_called()
    assert rig.row()["active_wait"] is not None


# ── 5. 续跑点:从上一个已通过的闸门 ────────────────────────────────────

def _plan(s, stage_idx, step_idx, gates=(), skill_meta=None, params=None):
    row = {"stage_idx": stage_idx, "step_idx": step_idx,
           "params": dict(params or {})}
    return recovery.resume_plan(s, row, gates, skill_meta=skill_meta)


def _gate_ev(stage_id, gate_id, which, verdict="pass"):
    return {"stage_id": stage_id,
            "payload": {"gate_id": gate_id, "which": which, "verdict": verdict}}


def test_with_no_gate_passed_the_resume_point_is_the_start_of_the_stage(tmp_path):
    """本阶段一道闸都没判过 ⇒ 退到**本阶段起点**。

    起点算数是因为 ``entry_actions`` 按定义就是「阶段入口重申设置(idempotent)」
    —— 那正是一个可以无条件重放的检查点。
    """
    s = spec([stage("S", steps=(step("S.01"), step("S.02"), step("S.03")))])
    plan = _plan(s, 0, 2, skill_meta=meta_of({"ScanAt": FakeMeta()}))
    assert plan.rewound and plan.step_idx == 0
    assert not plan.blocked


def test_the_resume_point_is_just_after_the_last_gate_that_passed(tmp_path):
    s = spec([stage("S", steps=(step("S.01", gate=rule_gate("g1")),
                                step("S.02"), step("S.03")))])
    plan = _plan(s, 0, 2, gates=[_gate_ev("S", "g1", "step")],
                 skill_meta=meta_of({"ScanAt": FakeMeta()}))
    assert plan.rewound and plan.step_idx == 1


def test_a_gate_that_did_not_pass_is_not_a_checkpoint(tmp_path):
    """判 fail 的闸门不是检查点 —— 它没有祝福过任何位置。"""
    s = spec([stage("S", steps=(step("S.01", gate=rule_gate("g1")),
                                step("S.02"), step("S.03")))])
    plan = _plan(s, 0, 2,
                 gates=[_gate_ev("S", "g1", "step", verdict="fail")],
                 skill_meta=meta_of({"ScanAt": FakeMeta()}))
    assert plan.step_idx == 0


def test_the_rewind_never_crosses_a_wait_for_a_person(tmp_path):
    """回退**不许**跨过等人步:那个 ack 是一个人半夜起来换了样品。"""
    s = spec([stage("S", steps=(StepSpec(step_id="S.00", kind="skill",
                                         skill="SafeRetract"),
                                wait_step("S.01"), step("S.02"), step("S.03")))])
    plan = _plan(s, 0, 3, skill_meta=meta_of({"ScanAt": FakeMeta(),
                                              "SafeRetract": FakeMeta()}))
    assert plan.blocked
    assert "等人步" in plan.blocked
    assert plan.step_idx == 3, "退不回去就该原地不动,由人来定"


def test_the_rewind_never_crosses_something_that_changes_the_surface(tmp_path):
    """回退**不许**跨过脉冲/扎针/除层 —— 重放它就是又打一发。"""
    s = spec([stage("S", steps=(step("S.01", skill="TipPulse"), step("S.02"),
                                step("S.03")))])
    plan = _plan(s, 0, 2, skill_meta=meta_of({
        "TipPulse": FakeMeta(caps={"bias_pulse"}), "ScanAt": FakeMeta()}))
    assert plan.blocked and "bias_pulse" in plan.blocked


def test_a_dangerous_step_blocks_the_rewind_too(tmp_path):
    s = spec([stage("S", steps=(step("S.01", skill="Forge"), step("S.02"),
                                step("S.03")))])
    plan = _plan(s, 0, 2, skill_meta=meta_of({
        "Forge": FakeMeta(level="dangerous"), "ScanAt": FakeMeta()}))
    assert plan.blocked and "DANGEROUS" in plan.blocked


def test_a_skill_nobody_can_look_up_blocks_the_rewind(tmp_path):
    """**「查不到危不危险」不是「不危险」。**"""
    s = spec([stage("S", steps=(step("S.01", skill="Mystery"), step("S.02"),
                                step("S.03")))])
    plan = _plan(s, 0, 2, skill_meta=meta_of({}))
    assert plan.blocked and "查不到" in plan.blocked


def test_no_skill_metadata_at_all_blocks_the_rewind(tmp_path):
    s = spec([stage("S", steps=(step("S.01"), step("S.02"), step("S.03")))])
    plan = _plan(s, 0, 2, skill_meta=None)
    assert plan.blocked and "没有技能元数据" in plan.blocked


def test_the_rewind_stays_inside_the_current_stage(tmp_path):
    """只在本阶段内回退(§8 逐字:「该阶段最近 epoch-safe 检查点」)。

    上一阶段的出口闸门放行过,但跨阶段回退要重放一整段测量 —— 那是人的决定。
    """
    s = spec([stage("A", steps=(step("A.01"),)),
              stage("B", steps=(step("B.01"), step("B.02")))])
    plan = _plan(s, 1, 1, gates=[_gate_ev("A", "gA", "exit")],
                 skill_meta=meta_of({"ScanAt": FakeMeta()}))
    assert plan.stage_idx == 1 and plan.step_idx == 0


def test_a_binding_that_would_eat_pre_restart_output_blocks_the_resume(tmp_path):
    """**这一条是 A4 的直接后果,而且今天在别处没有人问。**

    ``_collect_evidence`` 按 ``min_epoch`` 把跨代次证据挡在闸门外,而
    ``_resolve_params`` 走的是另一条路 —— 它只问「上游产出过没有」,不问「哪一代
    产出的」。于是一串重启前算出来的**坐标**可以原样喂进下一个硬件技能。
    这里在选续跑点的时候就把这种点判成不自洽。
    """
    s = spec([stage("S", steps=(
        step("S.01", produces=("positions",), gate=rule_gate("g1")),
        step("S.02", bindings={"positions": "steps.S.01.positions"}),
        step("S.03")))])
    # 闸门在 S.01 之后放行过 ⇒ 续跑点是第 1 步,而第 1 步绑着第 0 步的产出。
    plan = _plan(s, 0, 2, gates=[_gate_ev("S", "g1", "step")],
                 skill_meta=meta_of({"ScanAt": FakeMeta()}))
    assert plan.blocked
    assert "重启之前的产出" in plan.blocked


def test_a_binding_whose_producer_will_rerun_is_fine(tmp_path):
    s = spec([stage("S", steps=(
        step("S.01", produces=("positions",)),
        step("S.02", bindings={"positions": "steps.S.01.positions"}),
        step("S.03")))])
    plan = _plan(s, 0, 2, skill_meta=meta_of({"ScanAt": FakeMeta()}))
    assert plan.rewound and plan.step_idx == 0 and not plan.blocked


def test_standing_on_an_entry_gate_has_nothing_to_rewind(tmp_path):
    s = spec([stage("S", steps=(step("S.01"),), entry_gate=rule_gate("ge"))])
    plan = _plan(s, 0, AT_ENTRY_GATE, skill_meta=meta_of({"ScanAt": FakeMeta()}))
    assert not plan.rewound and not plan.blocked


def test_the_resume_point_is_recorded_and_the_pointer_really_moves(tmp_path):
    """端到端:自检全过 → 位置真的退回去了,而且这件事留了痕。"""
    from _harness import FakeExecutor, outcome

    s = spec([stage("S", steps=(step("S.01"), step("S.02"), step("S.03")))],
             auto_resume=True, recovery=tip_check())
    rig = build(tmp_path, s, executor=FakeExecutor({
        "PreScanCheck": [outcome(True, data={"tip_ready": True})]}),
        link_probe=lambda: {"main": True},
        skill_meta=meta_of({"ScanAt": FakeMeta(), "PreScanCheck": FakeMeta()}),
        temperature=FakeTemperature(FakeReading(value_k=4.9, age_s=1.0)))
    rig.tick(3)                       # adopt + 跑掉 S.01、S.02
    assert rig.row()["step_idx"] == 2
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "running"
    assert rig.row()["step_idx"] == 0, "位置没退回去"
    assert "resume_point" in _items(rig)


def test_a_blocked_resume_point_overrides_preauthorisation(tmp_path):
    """**预授权只授权「通过后不打扰」。** 算不出自洽的续跑点根本还没到「通过」。"""
    from _harness import FakeExecutor, outcome

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
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "续跑点接不上" in rig.row()["status_reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])


# ── 6. 结构闸:复验步绕开了阶段那套对账,这几条是它唯一的口 ────────────

def _validate(s, skills=None):
    from mast.conduct import analyses
    from mast.conduct.validator import validate_spec

    return validate_spec(s, skills=skills, analyses=analyses.known_names())


def _codes(rep) -> set:
    return {f.code for f in rep.findings}


def test_a_dangerous_recovery_step_is_refused(tmp_path):
    """A3 跑在进程刚死过一次之后、没有人在场 —— **复验只许只读**。

    而且复验步不在任何阶段里:pre-flight 的 SAFE 模式对账查的是「下一个流程步
    落在哪个阶段、阶段声明了哪些 capability」,它查不到复验步。这道闸是唯一的
    对账口 —— 少了它,SAFE 模式在恢复路径上形同虚设。
    """
    s = spec([stage("S", steps=(step("S.01"),))],
             recovery=tip_check(skill="TipPulse"))
    rep = _validate(s, skills={"TipPulse": FakeMeta(caps={"bias_pulse"}),
                               "ScanAt": FakeMeta()})
    assert "recovery_step_writes" in _codes(rep)


def test_a_recovery_step_binding_to_a_flow_step_is_refused(tmp_path):
    """复验步绑流程步 = 拿**重启之前**采的数当证据,而那正是自检要作废的东西。"""
    steps = (StepSpec(step_id="R.00", kind="composite", skill="PreScanCheck",
                      bindings={"center_x_m": "steps.S.01.x_m"},
                      produces=("tip_ready",)),)
    s = spec([stage("S", steps=(step("S.01", produces=("x_m",)),))],
             recovery=tip_check(steps=steps))
    rep = _validate(s, skills={"PreScanCheck": FakeMeta(), "ScanAt": FakeMeta()})
    assert "recovery_binding_not_upstream" in _codes(rep)


def test_a_rule_reading_a_field_nobody_produces_is_refused(tmp_path):
    """判据读一个没人产出的字段 ⇒ 它会**永远判不了**,而那在面板上长得像
    「一直没跑」。"""
    s = spec([stage("S", steps=(step("S.01"),))],
             recovery=tip_check(rule=RuleLeaf("sharpness", ">=", 1)))
    rep = _validate(s, skills={"PreScanCheck": FakeMeta(), "ScanAt": FakeMeta()})
    assert "recovery_rule_field_not_produced" in _codes(rep)


def test_without_a_registry_the_recovery_check_is_reported_as_skipped(tmp_path):
    """**没检查 ≠ 检查通过。**"""
    s = spec([stage("S", steps=(step("S.01"),))], recovery=tip_check())
    rep = _validate(s, skills=None)
    assert any("恢复自检 A3" in k for k in rep.checks_skipped), rep.checks_skipped


def test_a_recovery_step_may_not_reuse_a_flow_step_id():
    """同名会让一次自检的读数盖掉一个流程步的产出(它们走同一条审计流)。"""
    steps = (StepSpec(step_id="S.01", kind="composite", skill="PreScanCheck",
                      produces=("tip_ready",)),)
    with pytest.raises(ValueError, match="同名"):
        spec([stage("S", steps=(step("S.01"),))], recovery=tip_check(steps=steps))


def test_a_tip_check_without_a_rule_is_refused():
    """采了证据没有判据,结论就只剩「没报错」—— 那正是「假成功」的形状。"""
    from mast.conduct.spec import RecoveryPolicy as RP

    with pytest.raises(ValueError, match="没有判据"):
        RP(tip_check=(StepSpec(step_id="R.00", kind="skill", skill="X",
                               produces=("a",)),))


def test_a_wait_step_may_not_hide_inside_the_tip_check():
    """要人来看是自检的**结论**,不是自检的一步。"""
    from mast.conduct.spec import RecoveryPolicy as RP

    with pytest.raises(ValueError, match="不能是 wait 步"):
        RP(tip_check=(wait_step("R.00"),), tip_rule=RuleLeaf("a", "==", 1))


# ── 7. 人读副本:那一行要说得出「过了没有」────────────────────────────

def test_the_progress_line_says_whether_a_check_passed(tmp_path):
    """``progress.jsonl`` 的用法是逐行走查(§9 M4 晨检)。一行只说「恢复自检的
    一项」等于让人逐行展开 JSON。"""
    from mast.conduct.journal import _summary_for

    line = _summary_for("recovery_item", {"item": "A3_tip", "verdict": "unreadable"})
    assert "A3_tip" in line and "读不到" in line and "≠通过" in line
    assert _summary_for("recovery_item", {"item": "A1_link", "verdict": "pass"}) \
        != _summary_for("recovery_item", {"item": "A1_link", "verdict": "fail"})


def test_the_spec_snapshot_shows_the_steps_that_will_touch_the_tip(tmp_path):
    """一份 spec 快照描述了所有会动针的步骤 —— **重启之后那两步也算**。

    它们不在任何阶段里,按 stage 走的枚举一个字都印不出来(§10c 的同一个形状:
    引擎多认一个位置,所有枚举都要走一遍)。
    """
    from mast.conduct.journal import render_spec_markdown

    with_check = render_spec_markdown(
        spec([stage("S", steps=(step("S.01"),))], recovery=tip_check()), {})
    assert "PreScanCheck" in with_check and "会动仪器" in with_check
    without = render_spec_markdown(spec([stage("S", steps=(step("S.01"),))]), {})
    assert "没有声明" in without and "读不到 ≠ 通过" in without
