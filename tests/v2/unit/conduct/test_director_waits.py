"""Director × 两类等待 —— 双闸、hold、stale、重播、waive。

设计:``campaign_director_design.md`` §3-5、§5(等待相关各行)、§10-1。

一句话:**人的 ack 和物理条件回答两个不同的问题,互不替代**。人确认了不等于
降到温,降到温不等于样品换好。所以两个闸各记各的、顺序无关、缺哪个说哪个。

而第三种情况才是最要命的:**读数过期 = 读不到 ≠ 没到**。把它当成「条件还没
满足」就会安安静静地等到天亮 —— 今天的真机正是这个条件(温度口被别的程序占着,
一条序列都没有)。
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
    retract_step,
    spec,
    stage,
    step,
    wait_step,
)

from mast.conduct.spec import ConditionSpec

COND = ConditionSpec(signal="temperature_k", op="<=", value=5.0,
                     stale_after_s=600.0, hold_s=1800.0, desc="≤ 5 K")


def _wait_spec(kind="both", condition=COND, **kw):
    return spec([stage("S", steps=(
        retract_step("S.00"),
        wait_step("S.01", kind=kind, condition=condition, **kw),
        step("S.02")))])


def _rig(tmp_path, *, kind="both", condition=COND, value_k=None, age_s=None,
         reason=None, **kw):
    temp = FakeTemperature(FakeReading(value_k=value_k, age_s=age_s,
                                       reason=reason))
    rig = build(tmp_path, _wait_spec(kind=kind, condition=condition, **kw),
                temperature=temp)
    rig.run_until("waiting_operator" if kind != "condition" else "waiting_condition")
    return rig


def _wait_id(rig) -> str:
    return rig.row()["active_wait"]["wait_id"]


def _ack(rig, wait_id=None, by="用户"):
    rig.store.enqueue_op(rig.conduct_id, "ack",
                         args={"wait_id": wait_id or _wait_id(rig)},
                         requested_by=by)
    rig.tick()


# ── 进入等待 ─────────────────────────────────────────────────────────────

def test_a_wait_step_enters_the_right_state_and_notifies(tmp_path):
    rig = _rig(tmp_path)
    row = rig.row()
    assert row["status"] == "waiting_operator"
    assert row["active_wait"]["ack_required"] is True
    assert row["status_reason"], "等待没有把「在等什么」写进状态理由"
    assert any(n.kind == "wait" for n in rig.notifier.sent)


def test_a_condition_only_wait_needs_no_human(tmp_path):
    rig = _rig(tmp_path, kind="condition", value_k=300.0, age_s=1.0)
    assert rig.row()["status"] == "waiting_condition"
    assert rig.row()["active_wait"]["ack_required"] is False


def test_each_wait_gets_its_own_id(tmp_path):
    """wait_id 每次等待唯一 —— 这是「对旧等待点 ack」能被认出来的前提。"""
    rig = _rig(tmp_path)
    assert len(_wait_id(rig)) >= 6


# ── 双闸:顺序无关 ───────────────────────────────────────────────────────

def test_the_human_gate_alone_does_not_release_a_double_gate(tmp_path):
    """人确认了不等于降到温。"""
    rig = _rig(tmp_path, value_k=300.0, age_s=1.0)
    _ack(rig)
    assert rig.row()["status"] == "waiting_operator"
    assert rig.row()["active_wait"]["ack_at"] is not None


def test_the_condition_alone_does_not_release_a_double_gate(tmp_path):
    """降到温不等于样品换好。"""
    rig = _rig(tmp_path, value_k=4.0, age_s=1.0)
    rig.tick()
    rig.clock.advance(COND.hold_s + 1)
    rig.tick()
    assert rig.row()["status"] == "waiting_operator"


def test_ack_first_then_condition_releases(tmp_path):
    rig = _rig(tmp_path, value_k=300.0, age_s=1.0)
    _ack(rig)
    rig.temperature.reading = FakeReading(value_k=4.0, age_s=1.0)
    rig.tick()                       # 第一次读到达标 → 记 met_since
    rig.clock.advance(COND.hold_s + 1)
    rig.tick()
    assert rig.row()["status"] == "running"


def test_condition_first_then_ack_releases(tmp_path):
    """顺序无关:两个证据各记各的,每 tick 判合取。"""
    rig = _rig(tmp_path, value_k=4.0, age_s=1.0)
    rig.tick()
    rig.clock.advance(COND.hold_s + 1)
    rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    _ack(rig)
    assert rig.row()["status"] == "running"


def test_a_released_wait_moves_on_to_the_next_step(tmp_path):
    rig = _rig(tmp_path, kind="operator", condition=None)
    _ack(rig)
    row = rig.row()
    assert row["status"] == "running"
    assert row["active_wait"] is None
    assert "wait_released" in rig.event_kinds()
    rig.tick()
    assert "ScanAt" in rig.executor.skills_called(), "等待解除后没接着往下走"


# ── hold_s:防回弹 ───────────────────────────────────────────────────────

def test_hold_s_is_not_satisfied_the_instant_the_reading_crosses(tmp_path):
    rig = _rig(tmp_path, kind="condition", value_k=4.0, age_s=1.0)
    rig.tick()
    assert rig.row()["status"] == "waiting_condition", "刚到就放行,没等 hold"
    rig.clock.advance(COND.hold_s + 1)
    rig.tick()
    assert rig.row()["status"] == "running"


def test_a_reading_that_bounces_back_restarts_the_hold(tmp_path):
    """降到位又升回去不算到位。"""
    rig = _rig(tmp_path, kind="condition", value_k=4.0, age_s=1.0)
    rig.tick()
    assert rig.row()["active_wait"]["condition_met_since"] is not None
    rig.temperature.reading = FakeReading(value_k=9.0, age_s=1.0)   # 回升
    rig.tick()
    assert rig.row()["active_wait"]["condition_met_since"] is None
    rig.temperature.reading = FakeReading(value_k=4.0, age_s=1.0)
    rig.tick()
    rig.clock.advance(COND.hold_s - 10)      # 从**重新**到达那一刻起算
    rig.tick()
    assert rig.row()["status"] == "waiting_condition"


# ── stale:读不到 ≠ 没到 ────────────────────────────────────────────────

def test_a_stale_reading_downgrades_a_condition_wait_to_asking_a_human(tmp_path):
    """读数太旧 ⇒ 读不到 ⇒ **要人来看**,不是安安静静接着等。"""
    rig = _rig(tmp_path, kind="condition", value_k=4.0,
               age_s=COND.stale_after_s + 1)
    rig.tick()
    row = rig.row()
    assert row["status"] == "waiting_operator"
    assert "读不到不等于没到" in row["status_reason"]
    assert any(n.kind == "stale" for n in rig.notifier.sent)


def test_an_unreadable_sensor_is_stale_not_a_satisfied_condition(tmp_path):
    """没有值 + 没有年龄 = 「不知道」。绝不能因为 ``None <= 5`` 报错就当成没到,
    更不能当成到了。"""
    rig = _rig(tmp_path, kind="condition", reason="no_sensor")
    rig.tick()
    assert rig.row()["status"] == "waiting_operator"


def test_a_stale_reading_never_releases_even_with_an_ack(tmp_path):
    """人的 ack 只答「样品换好了」,答不了「降到温了」。"""
    rig = _rig(tmp_path, value_k=4.0, age_s=1.0)
    _ack(rig)
    rig.temperature.reading = FakeReading(value_k=4.0,
                                          age_s=COND.stale_after_s + 1)
    rig.clock.advance(COND.hold_s + 1)
    rig.tick()
    assert rig.row()["status"] != "running"


# ── waive:能停必须能解 ─────────────────────────────────────────────────

def test_a_waived_condition_releases_and_leaves_a_permanent_mark(tmp_path):
    """传感器读不到时 condition 闸会永远 stale —— 人必须有一条**显式、留痕**
    的解锁路。这不是默默放行:标记进事件、进 active_wait,面板持续显示。
    """
    rig = _rig(tmp_path, reason="no_sensor")
    _ack(rig)
    rig.store.enqueue_op(rig.conduct_id, "waive_condition",
                         args={"wait_id": _wait_id(rig), "reason": "温度计坏了"},
                         requested_by="用户")
    rig.tick()
    assert "wait_waived" in rig.event_kinds()
    rig.tick()
    row = rig.row()
    assert row["status"] == "running"
    released = [e for e in rig.events(kind="wait_released")][-1]
    assert released["payload"]["waived_by"] == "用户"


def test_a_waive_without_a_reason_is_refused_at_the_queue(tmp_path):
    rig = _rig(tmp_path)
    with pytest.raises(ValueError):
        rig.store.enqueue_op(rig.conduct_id, "waive_condition",
                             args={"wait_id": _wait_id(rig)})


# ── 重播与超时 ───────────────────────────────────────────────────────────

def test_the_request_is_replayed_on_schedule_not_every_tick(tmp_path):
    """每 4 h 重播一次。每 tick 都发就成了噪声,而噪声等于没有通知。"""
    rig = _rig(tmp_path, kind="operator", condition=None,
               renotify_every_s=14400.0)
    first = len(rig.notifier.sent)
    for _ in range(5):
        rig.tick()
    assert len(rig.notifier.sent) == first, "还没到点就重播了"
    rig.clock.advance(14401)
    rig.tick()
    assert len(rig.notifier.sent) == first + 1
    assert rig.notifier.sent[-1].kind == "wait_renotify"


def test_the_replay_says_which_gate_is_still_missing(tmp_path):
    rig = _rig(tmp_path, value_k=300.0, age_s=1.0, renotify_every_s=100.0)
    rig.clock.advance(101)
    rig.tick()
    msg = rig.notifier.sent[-1].message
    assert "人的确认" in msg and "物理条件" in msg


def test_exceeding_max_wait_escalates_but_never_gives_up(tmp_path):
    """等人**没有 fail-closed**。HITL 那条 900 s 上限正是 conduct 不能用它的
    原因 —— 换样品可能是一整夜。"""
    rig = _rig(tmp_path, kind="operator", condition=None, max_wait_s=3600.0)
    rig.clock.advance(3601)
    rig.tick()
    assert rig.row()["status"] == "waiting_operator", "超时把等待放弃了"
    assert any(n.kind == "wait_overdue" for n in rig.notifier.sent)
    # 只升级一次,不每 tick 刷屏
    n = len([x for x in rig.notifier.sent if x.kind == "wait_overdue"])
    rig.clock.advance(3600)
    rig.tick()
    assert len([x for x in rig.notifier.sent if x.kind == "wait_overdue"]) == n


def test_the_last_reading_is_kept_for_the_panel(tmp_path):
    """面板要显示「现在读到多少、多旧」—— 缺哪个闸就显示哪个的证据。"""
    rig = _rig(tmp_path, value_k=7.5, age_s=3.0)
    rig.tick()
    aw = rig.row()["active_wait"]
    assert aw["last_reading"] == 7.5
    assert aw["last_reading_age_s"] == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
