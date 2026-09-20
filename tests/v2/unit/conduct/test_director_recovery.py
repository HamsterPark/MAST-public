"""Director × 重启恢复 —— 清算、自检、以及「读不到不等于通过」。

设计:``campaign_director_design.md`` §5(restart 行 + RECOVERY_PENDING 各行)、
§8(自检清单与预授权位语义表)。

三条命题:

1. **重启不抹掉人的决定**:PAUSED 还是 PAUSED,等待还是等待;
2. **死在步里的那一步产出不可信、不重放**;
3. **预授权只授权「通过后不打扰」,从不授权「失败也继续」** ——
   而「读不到」和「失败」走同一条出口。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from _harness import (
    FakeRecoveryProbe,
    build,
    retract_step,
    spec,
    stage,
    step,
    wait_step,
)

from mast.conduct.director import CONTACT_CHECKS, NO_CONTACT_CHECKS


def _simple(auto_resume=False, version=1):
    return spec([stage("S", steps=(step("S.01"), step("S.02")))],
                auto_resume=auto_resume, version=version)


def _probe(**kw):
    return FakeRecoveryProbe(**kw)


def _recover(rig, limit: int = 12):
    for _ in range(limit):
        rig.tick()
        if rig.row()["status"] != "recovery_pending":
            return
    return


def _items(rig) -> list[str]:
    return [(e["payload"] or {}).get("item")
            for e in rig.events(kind="recovery_item")]


# ── 重启的入口 ───────────────────────────────────────────────────────────

def test_a_restart_sends_a_running_conduct_through_recovery(tmp_path):
    rig = build(tmp_path, _simple())
    rig.tick()
    rep = rig.director.reconcile_after_restart()
    assert rig.row()["status"] == "recovery_pending"
    assert "restart_to_recovery" in rep.actions
    assert "重启前是 running" in rig.row()["status_reason"]


@pytest.mark.parametrize("status", ["paused", "draft", "approved"])
def test_a_restart_respects_states_that_mean_a_person_decided(tmp_path, status):
    """PAUSED 保持 PAUSED(尊重人的暂停);DRAFT/APPROVED 还没被采纳,不动。"""
    rig = build(tmp_path, _simple(), approve=(status != "draft"))
    if status == "paused":
        rig.tick()
        rig.store.record(rig.conduct_id, "status_change", changes={
            "status": "paused", "status_reason": "人按了暂停"})
    rig.director.reconcile_after_restart()
    assert rig.row()["status"] == status


def test_a_restart_keeps_a_wait_alive(tmp_path):
    """「等换样品」不该被一次重启变成「重新开始」—— active_wait 留在库里。"""
    s = spec([stage("S", steps=(retract_step("S.00"), wait_step("S.01"),
                                step("S.02")))])
    rig = build(tmp_path, s)
    rig.run_until("waiting_operator")
    wait_id = rig.row()["active_wait"]["wait_id"]
    rig.director.reconcile_after_restart()
    assert rig.row()["status"] == "recovery_pending"
    assert rig.row()["active_wait"]["wait_id"] == wait_id


# ── A6:死在步里的清算 ──────────────────────────────────────────────────

def test_an_interrupted_run_is_marked_and_its_output_is_never_replayed(tmp_path):
    """进程死在步中 ⇒ 那一步的产出**不可信、不重放**。

    composite 的 sidecar 按 (name, run_id) 分键,所以旧 run 的中间产物天然作废;
    这里做的是把这件事**记下来**并把 active_run_id 清掉,否则下一次 abort 会去
    置一个早就不存在的 run 的 Event。
    """
    rig = build(tmp_path, _simple(), recovery_probe=_probe())
    rig.tick()
    rig.store.record(rig.conduct_id, "status_change",
                     changes={"active_run_id": "run-dead"})
    rig.director.reconcile_after_restart()
    rig.tick()
    ev = rig.events(kind="step_interrupted")
    assert ev and ev[0]["run_id"] == "run-dead"
    assert "不重放" in ev[0]["payload"]["why"]
    assert rig.row()["active_run_id"] == ""


def test_no_leftover_run_is_also_a_pass(tmp_path):
    rig = build(tmp_path, _simple(), recovery_probe=_probe())
    rig.tick()
    rig.director.reconcile_after_restart()
    rig.tick()
    assert _items(rig)[0] == "A6_leftovers"
    assert rig.events(kind="step_interrupted") == []


# ── A5:模板版本对账 ────────────────────────────────────────────────────

def test_a_spec_version_mismatch_stops_and_never_migrates_itself(tmp_path):
    """跑到一半的 conduct 遇上换了定义的模板,续跑意味着前半段和后半段属于
    两个不同的实验。**不自动迁移**。"""
    rig = build(tmp_path, _simple(version=1), recovery_probe=_probe())
    rig.tick()
    rig.store.record(rig.conduct_id, "status_change", changes={"spec_version": 7})
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "不自动迁移" in rig.row()["status_reason"]


# ── A4:代次记账 ────────────────────────────────────────────────────────

def test_a_restart_invalidates_evidence_gathered_before_it(tmp_path):
    """重启一律 bump 证据代次 —— 重启前采的证据不再参与闸门判定。"""
    rig = build(tmp_path, _simple(), recovery_probe=_probe())
    rig.tick()
    assert rig.row()["evidence_epoch"] == 0
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["evidence_epoch"] == 1


# ── A1/A2/A3:没探针 = 读不到 ──────────────────────────────────────────

def test_without_probes_the_checklist_stops_at_unreadable(tmp_path):
    """**读不到不等于通过。** 没接探针就停下来问人,而不是「反正也没报错」。"""
    rig = build(tmp_path, _simple(auto_resume=True))     # 即使开了预授权
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    row = rig.row()
    assert row["status"] == "waiting_operator"
    assert "读不到不等于通过" in row["status_reason"]


def test_a_failed_check_stops_even_with_auto_resume(tmp_path):
    """预授权只授权「通过后不打扰」,从不授权「失败也继续」。"""
    rig = build(tmp_path, _simple(auto_resume=True),
                recovery_probe=_probe(verdicts={"A2_temp": "fail"}))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "A2_temp" in rig.row()["status_reason"]


def test_a_probe_that_answers_nonsense_is_treated_as_unreadable(tmp_path):
    rig = build(tmp_path, _simple(auto_resume=True),
                recovery_probe=_probe(default="probably fine"))
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"


def test_a_crashing_probe_is_unreadable_not_a_pass(tmp_path):
    def boom(item):
        raise RuntimeError("串口没了")

    rig = build(tmp_path, _simple(auto_resume=True), recovery_probe=boom)
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"


# ── 全过之后 ─────────────────────────────────────────────────────────────

def test_all_checks_passing_with_preauthorisation_resumes_without_asking(tmp_path):
    rig = build(tmp_path, _simple(auto_resume=True), recovery_probe=_probe())
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "running"
    # 清单之外还多一条 ``resume_point``:自检全过之后「从哪儿接着跑」也是一个
    # 要留痕的判断(M4-a)。用 ``<=`` 而不是 ``==`` 是刻意的 —— 这条测试钉的是
    # 「六项一项不少地跑过」,不是「一共只写了六条记录」。
    assert set(NO_CONTACT_CHECKS) | set(CONTACT_CHECKS) <= set(_items(rig))
    assert "resume_point" in _items(rig)


def test_all_checks_passing_without_preauthorisation_waits_for_a_word(tmp_path):
    rig = build(tmp_path, _simple(auto_resume=False), recovery_probe=_probe())
    rig.tick()
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert "等一句「继续」" in rig.row()["status_reason"]


def test_a_waiting_conduct_skips_the_contact_check_and_returns_to_waiting(tmp_path):
    """A3 要进针,会把「等换样品」直接打破 —— 针尖重验推迟到等待解除之后。"""
    s = spec([stage("S", steps=(retract_step("S.00"), wait_step("S.01"),
                                step("S.02")))])
    probe = _probe()
    rig = build(tmp_path, s, recovery_probe=probe)
    rig.run_until("waiting_operator")
    rig.director.reconcile_after_restart()
    _recover(rig)
    assert rig.row()["status"] == "waiting_operator"
    done = set(_items(rig))
    assert set(NO_CONTACT_CHECKS) <= done
    assert not (set(CONTACT_CHECKS) & done), "等待态里做了接触档自检"
    assert "A3_tip" not in probe.asked
    assert rig.row()["active_wait"] is not None, "自检把等待弄丢了"


def test_each_check_is_one_tick_of_work(tmp_path):
    """自检项也是步,跨 tick —— 一个 tick 里跑完全部会把长检查变成一次阻塞。"""
    rig = build(tmp_path, _simple(), recovery_probe=_probe())
    rig.tick()
    rig.director.reconcile_after_restart()
    rig.tick()
    assert len(_items(rig)) == 1
    rig.tick()
    assert len(_items(rig)) == 2


def test_a_resume_op_runs_the_same_checklist(tmp_path):
    """暂停期间世界未知 —— resume 与重启走同一条自检,快则秒过。"""
    rig = build(tmp_path, _simple(auto_resume=True), recovery_probe=_probe())
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "pause")
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "resume")
    rig.tick()
    assert rig.row()["status"] == "recovery_pending"
    _recover(rig)
    assert rig.row()["status"] == "running"
    assert set(_items(rig)) >= set(NO_CONTACT_CHECKS)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
