"""Director × 意图队列 —— 优先级、幂等、以及**「不可能」的转移显式拒绝**。

设计:``campaign_director_design.md`` §6-2、§5 末(「不可能」转移 ⇒ 显式拒绝 +
op_rejected 留痕,**不静默 no-op**,每条钉进穷举测试)。

为什么这条这么重要:静默 no-op 的样子是「用户按了按钮,什么都没发生,而且
事后查不到按过」。那和按钮坏了没有区别,但它看起来是好的。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

# 本目录是一个包(有 __init__.py),裸 import 同目录模块要先把它放进 sys.path。
# ``_harness`` 自己会做 MASTv2 的 bootstrap。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from _harness import build, retract_step, spec, stage, step, wait_step

from mast.conduct.director import OP_VALID_STATUSES
from mast.conduct.store import OPS, STATUSES, TERMINAL_STATUSES


def _simple():
    return spec([stage("S", steps=(step("S.01"), step("S.02")))])


@pytest.fixture
def rig(tmp_path):
    return build(tmp_path, _simple())


def _kinds(rig):
    return [e["kind"] for e in rig.events()]


def _rejections(rig):
    return [e["payload"] for e in rig.events(kind="op_rejected")]


# ── 优先级与吞并 ─────────────────────────────────────────────────────────

def test_abort_is_consumed_before_everything_else(rig):
    rig.tick()                                   # adopt → running
    rig.store.enqueue_op(rig.conduct_id, "set_attended", args={"attended": False})
    rig.store.enqueue_op(rig.conduct_id, "pause")
    rig.store.enqueue_op(rig.conduct_id, "abort", args={"reason": "样品掉了"})
    rig.tick()
    assert rig.row()["status"] == "aborted"


def test_abort_swallows_the_rest_of_the_batch_but_leaves_a_trace(rig):
    """一个已经决定中止的 conduct,再执行「继续」只是给日志添乱 —— 但要留痕。"""
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "abort", args={"reason": "停"})
    rig.store.enqueue_op(rig.conduct_id, "pause")
    rig.tick()
    swallowed = [p for p in _rejections(rig) if "作废" in p.get("why", "")]
    assert swallowed and swallowed[0]["op"] == "pause"


def test_every_op_is_consumed_exactly_once(rig):
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "pause")
    rig.tick()
    rig.tick()
    assert rig.store.pending_ops(rig.conduct_id) == []
    assert len([e for e in rig.events(kind="op_consumed")]) == 1


# ── 「不可能」的转移:穷举 ───────────────────────────────────────────────

@pytest.mark.parametrize("op", sorted(OPS))
def test_every_op_declares_which_statuses_accept_it(op):
    """闭集要有主人。一个没在表里的意图,消费时会走「拒绝」——那没问题,
    但它必须是**声明过的**拒绝,而不是查表查空。"""
    assert op in OP_VALID_STATUSES, f"{op} 没有声明适用状态"
    assert OP_VALID_STATUSES[op] <= set(STATUSES)
    assert not (OP_VALID_STATUSES[op] & set(TERMINAL_STATUSES)), \
        f"{op} 声明了终态 —— 终态不接受任何意图"


@pytest.mark.parametrize("op,status", sorted(
    (op, st) for op, st in itertools.product(sorted(OPS), sorted(STATUSES))
    if st not in OP_VALID_STATUSES[op] and st not in TERMINAL_STATUSES))
def test_an_impossible_op_is_refused_with_a_trace(tmp_path, op, status):
    """状态 × 意图的全矩阵:每一格「不可能」都必须留下 op_rejected。

    ``COMPLETED + ack``、``DRAFT + pause`` 这些正是设计点名要钉的格子。
    """
    rig = build(tmp_path, _simple(), approve=False)
    if status != "draft":
        rig.store.record(rig.conduct_id, "status_change", changes={
            "status": status, "status_reason": "测试摆位"})
    # 每个 op 各自的必填项(``store.enqueue_op`` 会在缺项时**抛**,而这条测试要
    # 测的是「状态不对时被显式拒绝」,不是「参数不全时入不了队」——后者另有测试)。
    args = ({"abort": {"reason": "r"},
             "waive_condition": {"reason": "r", "wait_id": "w"},
             "ack": {"wait_id": "w"},
             "override_decision": {"reason": "r", "decision_id": "1"}}
            .get(op, {}))
    rig.store.enqueue_op(rig.conduct_id, op, args=args)
    rig.director._consume_ops(rig.conduct_id, _Rep())
    rejected = [p["op"] for p in _rejections(rig)]
    assert op in rejected, f"{op} 在 {status} 下被静默吃掉了"
    assert rig.store.pending_ops(rig.conduct_id) == [], "拒绝了也要消费掉"


class _Rep:
    """给 ``_consume_ops`` 用的极简报告对象。"""

    def __init__(self):
        self.actions: list[str] = []

    def did(self, what):
        self.actions.append(what)
        return self


# ── 具体几条 ─────────────────────────────────────────────────────────────

def test_pause_does_not_touch_the_tip(rig):
    """pause ≠ abort:**不动针**。用户按暂停常常正是为了手动干预。"""
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "pause", requested_by="用户")
    rig.tick()
    assert rig.row()["status"] == "paused"
    assert "SafeRetract" not in rig.executor.skills_called()
    assert "暂停" in rig.row()["status_reason"]


def test_resume_goes_through_a_self_check_not_straight_back_to_running(rig):
    """暂停期间世界未知(人可能动过仪器)—— 统一走自检,快则秒过。"""
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "pause")
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "resume")
    rig.tick()
    assert rig.row()["status"] == "recovery_pending"


def test_takeover_pauses_and_says_it_was_a_takeover(rig):
    """手动接管必须显式 —— 用户在 GUI 上动仪器不持令牌,隐式检测必误判。"""
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "takeover", requested_by="用户")
    rig.tick()
    assert rig.row()["status"] == "paused"
    ev = [e for e in rig.events(kind="status_change")
          if (e["payload"] or {}).get("takeover")]
    assert ev, "接管没留下标记,事后分不清是暂停还是接管"


def test_abort_from_a_running_step_confirms_the_retract(rig):
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "abort", args={"reason": "样品掉了"},
                         requested_by="用户")
    rig.tick()
    row = rig.row()
    assert row["status"] == "aborted"
    assert "SafeRetract" in rig.executor.skills_called()
    assert "已确认" in row["status_reason"]
    assert row["active_slot"] is None, "终态必须让出活跃位"


def test_an_unconfirmed_retract_during_abort_is_never_reported_as_confirmed(
        tmp_path):
    """退针没确认就说没确认。**绝不假装退成功了** —— watchdog 兜底,人来看。"""
    from _harness import FakeExecutor, outcome
    ex = FakeExecutor({"SafeRetract": [outcome(ok=False, error="已下发未确认")]})
    rig = build(tmp_path, _simple(), executor=ex)
    rig.tick()
    rig.store.enqueue_op(rig.conduct_id, "abort", args={"reason": "停"})
    rig.tick()
    row = rig.row()
    assert row["status"] == "aborted"
    assert "**未确认**" in row["status_reason"]
    crit = [n for n in rig.notifier.sent if n.severity == "crit"]
    assert crit, "退针没确认却没有一条高优通知"
    assert ex.skills_called().count("SafeRetract") == 2, "失败了要重试一次"


def test_abort_in_a_waiting_state_does_not_retract_again(tmp_path):
    """等待态里针已经退了(校验器规则①的结构保证),直达 ABORTED。"""
    s = spec([stage("S", steps=(retract_step("S.00"), wait_step("S.01"),
                                step("S.02")))])
    rig = build(tmp_path, s)
    rig.run_until("waiting_operator")
    assert rig.row()["status"] == "waiting_operator"
    before = rig.executor.skills_called().count("SafeRetract")
    rig.store.enqueue_op(rig.conduct_id, "abort", args={"reason": "不做了"})
    rig.tick()
    assert rig.row()["status"] == "aborted"
    assert rig.executor.skills_called().count("SafeRetract") == before
    assert "针已退" in rig.row()["status_reason"]


def test_an_ack_for_the_wrong_wait_is_refused(tmp_path):
    """wait_id 每次等待唯一 —— 对旧等待点的 ack 必须被认出来(API 侧 409)。"""
    s = spec([stage("S", steps=(retract_step("S.00"), wait_step("S.01")))])
    rig = build(tmp_path, s)
    rig.run_until("waiting_operator")
    rig.store.enqueue_op(rig.conduct_id, "ack", args={"wait_id": "不是这个"})
    rig.tick()
    assert rig.row()["status"] == "waiting_operator"
    assert any("不匹配" in p.get("why", "") for p in _rejections(rig))


def test_set_attended_changes_the_stored_flag(rig):
    rig.tick()
    assert rig.row()["attended"] is True
    rig.store.enqueue_op(rig.conduct_id, "set_attended", args={"attended": False})
    rig.tick()
    assert rig.row()["attended"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
