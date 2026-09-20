"""supervised 的撤销窗 —— 「有生产方没有消费方」的那一半，终于接上了。

## 它之前是什么状态

``who_may_approve`` 在 ``supervised`` 档下会算出一个 ``ignition_delay_s``，
``ApprovalVerdict.deferred`` 也定义了；而**全仓没有一个消费方**：两条 approve
路径（HTTP 与 agent 工具）都只读 ``verdict.allowed``，Director 的 ``_dispatch``
在 ``approved`` 分支无条件 ``record("adopted", status="running")``。

也就是说 ``supervised`` 在行为上**等于** ``autonomous`` —— 而拒绝 attended 批准
时的那句文案还在推荐它：「把自主度调到 supervised（批后有撤销窗）」。撤销窗是这
一档存在的全部理由，它却不存在。

## 为什么撤回本身不用新写

``abort`` 在 ``approved`` 态本来就是合法操作（``OP_VALID_STATUSES``），走
``_abort`` 进终态。缺的只是「采纳时尊重 ``ignite_at``」这一处。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _harness import FakeExecutor, build, outcome, spec, stage, step  # noqa: E402

from mast.conduct.autonomy import ignition_payload, who_may_approve  # noqa: E402


def _spec():
    return spec([stage("S1", [step("S1.00", "Noop")])])


def _rig(tmp_path, delay_s: float | None, *, by: str = "agent:xd",
         level: str = "supervised"):
    """建一份**已批准**的 conduct，批准事件带上撤销窗。

    走 ``ignition_payload``（生产用的那一个），不在测试里手写 payload —— 手写
    的话，生产侧改了字段名这里也不会红，而这条测试的全部意义就是钉住那个字段。
    """
    ex = FakeExecutor(script={"Noop": [outcome(ok=True)]})
    # **每个用例一个库**。第一版用了一个固定路径，于是所有用例共用一份
    # conduct 库，第二个就撞上「单活跃 conduct」不变式 —— 那条不变式替我抓到了
    # 这次测试隔离失误。
    rig = build(tmp_path, _spec(), executor=ex, approve=False)
    verdict = who_may_approve(level, by=by, ignition_delay_s=delay_s)
    rig.store.record(rig.conduct_id, "approved",
                     changes={"status": "approved", "approved_by": by},
                     payload={"by": by,
                              **ignition_payload(verdict, rig.clock())})
    return rig


def _status(rig) -> str:
    return rig.store.get(rig.conduct_id)["status"]


def _kinds(rig) -> list[str]:
    return [e["kind"] for e in rig.store.events(rig.conduct_id, limit=200)]


# ── 窗内不点火 ──────────────────────────────────────────────────────

def test_inside_the_window_nothing_ignites(tmp_path):
    rig = _rig(tmp_path, 600.0)
    rig.director.step_tick()
    assert _status(rig) == "approved", "撤销窗开着却已经点火了"
    assert "ignition_held" in _kinds(rig)
    assert "adopted" not in _kinds(rig)
    assert rig.executor.calls == [], "窗内竟然已经在跑步骤了"

    rig.clock.advance(599.0)
    rig.director.step_tick()
    assert _status(rig) == "approved", "还差 1 秒就点火了"

    rig.clock.advance(2.0)
    rig.director.step_tick()
    assert _status(rig) == "running", "窗口过了却不点火 —— 这一档从此永远不动"


def test_the_hold_is_recorded_and_announced_exactly_once(tmp_path):
    """每 tick 一条审计行是没人读的审计行；每分钟一次通知是会被静音的通知。

    而被静音的通知等于没有通知 —— 这条纪律在唤醒调度器的熔断上已经写过一次。
    """
    rig = _rig(tmp_path, 600.0)
    for _ in range(5):
        rig.director.step_tick()
        rig.clock.advance(10.0)
    assert _kinds(rig).count("ignition_held") == 1
    notes = [n for n in rig.notifier.sent if n.kind == "ignition_window"]
    assert len(notes) == 1, f"通知发了 {len(notes)} 条"
    assert "abort" in notes[0].message and "600" in notes[0].message


def test_abort_inside_the_window_really_cancels_it(tmp_path):
    """撤销窗的全部意义：**来得及后悔**。

    窗内 abort 之后，无论过多久、tick 多少次，这份 conduct 都不许再点火。
    """
    rig = _rig(tmp_path, 600.0)
    rig.director.step_tick()
    rig.store.enqueue_op(rig.conduct_id, "abort", args={"reason": "想想还是算了"})
    rig.director.step_tick()
    assert _status(rig) == "aborted"

    rig.clock.advance(10_000.0)
    for _ in range(3):
        rig.director.step_tick()
    assert _status(rig) == "aborted", "窗口过了之后它自己又点火了"
    # 断言的是**模板的步**没跑。中止序列自己会调 SafeRetract（确认式退针）——
    # 把它一起断言成 0 次调用，等于顺手要求中止时不许退针，那是另一条纪律的反面。
    ran = [c[0] for c in rig.executor.calls]
    assert "Noop" not in ran, f"这份 conduct 的步骤还是跑了：{ran}"
    assert "adopted" not in _kinds(rig)


# ── 窗外照常 ────────────────────────────────────────────────────────

def test_a_human_approval_ignites_at_once(tmp_path):
    """人批准在任何一档都不留窗 —— 人已经在场了，撤销窗防的正是「人不在场」。"""
    rig = _rig(tmp_path, 600.0, by="operator")
    rig.director.step_tick()
    assert _status(rig) == "running"
    assert "ignition_held" not in _kinds(rig)


def test_zero_delay_ignites_at_once(tmp_path):
    rig = _rig(tmp_path, 0.0)
    rig.director.step_tick()
    assert _status(rig) == "running"
    assert "ignition_held" not in _kinds(rig)


def test_autonomous_ignites_at_once(tmp_path):
    rig = _rig(tmp_path, None, level="autonomous")
    rig.director.step_tick()
    assert _status(rig) == "running"


def test_an_approval_event_without_ignite_at_ignites_at_once(tmp_path):
    """旧格式的事件（这条改动之前批的那些）按「立刻」处理。

    **不是**按「读不到所以不许点火」：撤销窗保护的是「人来得及后悔」，不是一条
    安全判据；一份升级前批准的 conduct 卡死在这里，才是真的坏。
    """
    ex = FakeExecutor(script={"Noop": [outcome(ok=True)]})
    rig = build(tmp_path, _spec(), executor=ex, approve=True)   # harness 不写 payload
    rig.director.step_tick()
    assert _status(rig) == "running"


def test_an_unreadable_ignite_at_does_not_wedge_it(tmp_path):
    ex = FakeExecutor(script={"Noop": [outcome(ok=True)]})
    rig = build(tmp_path, _spec(), executor=ex, approve=False)
    rig.store.record(rig.conduct_id, "approved",
                     changes={"status": "approved", "approved_by": "agent:x"},
                     payload={"by": "agent:x", "ignite_at": "明天"})
    rig.director.step_tick()
    assert _status(rig) == "running"


# ── 绝对时刻，不是剩余秒数 ──────────────────────────────────────────

def test_the_window_does_not_restart_after_a_process_restart(tmp_path):
    """窗口是**绝对时刻**。

    存剩余秒数的话，进程在窗口中间重启会从头再数 —— 而一份 conduct 的时间尺度
    是 hour~day，重启是常态，于是一个十分钟的窗可以被无限延长。
    """
    rig = _rig(tmp_path, 600.0)
    rig.director.step_tick()
    rig.clock.advance(700.0)

    # 「重启」：同一个库、同一个时钟，换一个 Director。
    from mast.conduct.director import ConductDirector

    fresh = ConductDirector(rig.store, spec_provider=lambda sid: rig.spec,
                            executor=rig.executor, latch=rig.latch,
                            temperature=rig.temperature, notifier=rig.notifier,
                            clock=rig.clock)
    fresh.step_tick()
    assert _status(rig) == "running", "重启把撤销窗从头数了一遍"


# ── 变异 ────────────────────────────────────────────────────────────

def test_mutation_without_the_consumer_it_ignites_immediately(tmp_path, monkeypatch):
    """把 ``_ignition_hold`` 打成恒 None（= 消费方不存在）⇒ 窗内直接点火。

    这条与第一条是一对，钉的正是这次改动之前的真实行为：生产方在算窗口，
    没有人读它。
    """
    rig = _rig(tmp_path, 600.0)
    monkeypatch.setattr(type(rig.director), "_ignition_hold",
                        lambda self, cid, row, rep: None)
    rig.director.step_tick()
    assert _status(rig) == "running", (
        "拆掉消费方之后它也没点火 —— 那说明拦住它的是别的东西")


def test_a_second_window_after_pause_is_announced_again(tmp_path):
    """pause → 再 approve 的**第二个**窗口也要通知一次。

    第一版查的是「这份 conduct 有史以来有没有过 ignition_held」，于是第二个窗口
    既不记事件也不通知 —— 窗口仍然生效，但没人被告知可以撤回，而那条通知正是
    这一档的交付物。
    """
    rig = _rig(tmp_path, 600.0)
    rig.director.step_tick()
    assert len([n for n in rig.notifier.sent if n.kind == "ignition_window"]) == 1

    # 点火、暂停、再批一次（第二个窗口）
    rig.clock.advance(601.0)
    rig.director.step_tick()
    assert _status(rig) == "running"
    rig.store.record(rig.conduct_id, "status_change",
                     changes={"status": "approved"})
    v = who_may_approve("supervised", by="agent:xd", ignition_delay_s=300.0)
    rig.store.record(rig.conduct_id, "approved",
                     changes={"status": "approved", "approved_by": "agent:xd"},
                     payload={"by": "agent:xd",
                              **ignition_payload(v, rig.clock())})
    rig.director.step_tick()

    assert _status(rig) == "approved", "第二个撤销窗没生效"
    notes = [n for n in rig.notifier.sent if n.kind == "ignition_window"]
    assert len(notes) == 2, f"第二个窗口没通知：{len(notes)} 条"


def test_the_latest_approval_wins_not_the_earliest(tmp_path):
    """``store.events`` 是升序 + LIMIT，取尾拿到的是**最早**那些。

    读错的后果：一份重批过的 conduct 会拿一个早已过去的 ``ignite_at``，
    撤销窗静默失效。
    """
    rig = _rig(tmp_path, 1.0)                   # 第一次批：窗口 1 秒
    rig.clock.advance(5.0)
    v = who_may_approve("supervised", by="agent:xd", ignition_delay_s=900.0)
    rig.store.record(rig.conduct_id, "approved",
                     changes={"status": "approved", "approved_by": "agent:xd"},
                     payload={"by": "agent:xd",
                              **ignition_payload(v, rig.clock())})
    rig.director.step_tick()
    assert _status(rig) == "approved", (
        "读的是第一次批准的窗口（早就过了）—— 撤销窗静默失效")
