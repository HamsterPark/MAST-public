"""ConductStore —— 状态真源的三张表。

设计:``docs/v2/design/campaign_director_design.md`` §4.7 / §6。

这一层测的是**结构纪律**,不是 CRUD:

* 状态改动只有一扇门,而且事件与状态同一个事务(崩在中间整体回滚);
* 单活跃不变式由数据库执行,不靠调用方检查;
* 异常态必须带「为什么」;
* 意图队列按优先级消费且**幂等**(重放一次 abort 不是小事)。

数据隔离:全部 ``tmp_path``,构造函数**没有默认路径**。测试污染真实数据在本仓
已经发生过五次,每一次的入口都是一个善意的默认值。
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

import pytest

from mast.conduct.store import (
    OP_PRIORITY,
    REASON_REQUIRED_STATUSES,
    ActiveConductExists,
    ConductStore,
    ConductStoreError,
    UnknownConduct,
)


class FakeClock:
    """手驱时钟。拨钟测时间语义,不靠 sleep。"""

    def __init__(self, t0: float = 1_700_000_000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock) -> ConductStore:
    return ConductStore(tmp_path / "conduct.db", clock=clock)


def _make(store, **kw) -> str:
    kw.setdefault("experiment_id", "exp-1")
    kw.setdefault("spec_id", "t_v1")
    kw.setdefault("spec_version", 1)
    return store.create(**kw)


# ── 数据隔离 ─────────────────────────────────────────────────────────────

def test_there_is_no_default_path(tmp_path):
    """没有「不传就用全局实验库」这条逃生门 —— 那是污染真实数据的入口。"""
    with pytest.raises(TypeError):
        ConductStore()          # type: ignore[call-arg]
    with pytest.raises(ValueError):
        ConductStore("")


def test_memory_databases_are_refused(tmp_path):
    """``:memory:`` 会给每条连接一个独立空库:写了读不到,而且一个错都不报。"""
    with pytest.raises(ValueError) as e:
        ConductStore(":memory:")
    assert "读不到" in str(e.value)


# ── 建 ───────────────────────────────────────────────────────────────────

def test_create_writes_the_row_and_the_event(store):
    cid = _make(store)
    row = store.get(cid)
    assert row["status"] == "draft"
    assert row["experiment_id"] == "exp-1"
    assert row["active_slot"] == 1
    kinds = [e["kind"] for e in store.events(cid)]
    assert kinds == ["created"]


def test_a_conduct_must_belong_to_an_experiment(store):
    """没有实验归属的 conduct,产物没有地方落。"""
    with pytest.raises(ValueError):
        store.create(experiment_id="", spec_id="t", spec_version=1)


def test_only_one_unfinished_conduct_at_a_time(store):
    """单活跃不变式由 UNIQUE 执行 —— 第二个 INSERT 直接失败,不靠调用方检查。"""
    _make(store)
    with pytest.raises(ActiveConductExists):
        _make(store)


def test_finishing_one_frees_the_slot(store):
    a = _make(store)
    store.record(a, "aborted", changes={"status": "aborted",
                                        "status_reason": "用户中止"})
    assert store.get(a)["active_slot"] is None
    b = _make(store)                       # 现在可以建下一个
    assert store.active()["conduct_id"] == b


def test_active_returns_the_unfinished_one(store):
    cid = _make(store)
    assert store.active()["conduct_id"] == cid
    store.record(cid, "completed", changes={"status": "completed"})
    assert store.active() is None


# ── 改:唯一一扇门 ───────────────────────────────────────────────────────

def test_an_unknown_event_kind_is_refused(store):
    cid = _make(store)
    with pytest.raises(ValueError) as e:
        store.record(cid, "something_happened")
    assert "EVENT_KINDS" in str(e.value)


def test_an_unknown_column_is_refused_not_silently_dropped(store):
    """写错列名要当场炸。静默丢一次改动 = 一次状态转移凭空消失。"""
    cid = _make(store)
    with pytest.raises(ValueError) as e:
        store.record(cid, "status_change", changes={"stauts": "running"})
    assert "MUTABLE_COLUMNS" in str(e.value)


def test_an_unknown_conduct_raises_instead_of_returning_none(store):
    with pytest.raises(UnknownConduct):
        store.record("nope", "status_change", changes={"status": "running"})
    with pytest.raises(UnknownConduct):
        store.touch_heartbeat("nope")


@pytest.mark.parametrize("status", REASON_REQUIRED_STATUSES)
def test_an_abnormal_status_must_say_why(store, status):
    """一个没有「为什么」的停,人只能靠猜或者重启进程。

    2026-08-13 那次锁死的一半正是这个:闩挂上了,而「因为什么」在通知通道里
    被静默丢掉了。
    """
    cid = _make(store)
    with pytest.raises(ValueError) as e:
        store.record(cid, "status_change", changes={"status": status})
    assert "status_reason" in str(e.value)
    store.record(cid, "status_change",
                 changes={"status": status, "status_reason": "有原因"})
    assert store.get(cid)["status_reason"] == "有原因"


def test_a_status_change_records_the_transition_not_just_the_value(store):
    """审计要的是转移。只记终值的话,「它是怎么变成这样的」永远答不上来。"""
    cid = _make(store)
    store.record(cid, "adopted", changes={"status": "running"})
    ev = store.events(cid, kind="adopted")[0]
    assert ev["payload"]["status_from"] == "draft"
    assert ev["payload"]["status_to"] == "running"


def test_a_terminal_conduct_cannot_be_resurrected(store):
    """让一个 ABORTED 复活,等于让它的中止序列变成一段没有结论的历史。"""
    cid = _make(store)
    store.record(cid, "aborted", changes={"status": "aborted",
                                          "status_reason": "人喊停"})
    with pytest.raises(ConductStoreError) as e:
        store.record(cid, "status_change", changes={"status": "running"})
    assert "终态" in str(e.value)


def test_json_columns_round_trip_as_objects(store):
    cid = _make(store, params={"setpoint_a": 5e-11})
    store.record(cid, "detour_entered",
                 changes={"detour": {"return_stage_idx": 1, "reason": "坏针"},
                          "evidence_epoch": 3})
    row = store.get(cid)
    assert row["params"]["setpoint_a"] == 5e-11
    assert row["detour"]["reason"] == "坏针"
    assert row["evidence_epoch"] == 3
    store.record(cid, "detour_returned", changes={"detour": None})
    assert store.get(cid)["detour"] is None


def test_updating_one_field_never_forgets_the_others(store):
    """改一个列不许把没提到的列冲掉。

    ``plan_store.py`` 记过这个坑的另一头:``INSERT OR REPLACE`` 是**删旧行+插
    新行**,列清单里没写的一律回落默认值 —— 后果不是报错而是**静默失忆**
    (每次 save 都以为这个计划还没有文档,于是另立一份,同一个计划越攒越多)。
    这里用「显式列 INSERT + 白名单 UPDATE」结构上避开它,这条测行为。
    """
    cid = _make(store, params={"setpoint_a": 5e-11})
    store.record(cid, "adopted", changes={"status": "running"})
    store.record(cid, "step_started", changes={"stage_idx": 1, "step_idx": 2,
                                               "active_run_id": "run-7"})
    store.record(cid, "budget_tick", changes={"budget_spent_usd": 1.25})
    row = store.get(cid)
    assert row["params"] == {"setpoint_a": 5e-11}     # create 时写的还在
    assert row["status"] == "running"                  # 上一次改的还在
    assert (row["stage_idx"], row["step_idx"]) == (1, 2)
    assert row["active_run_id"] == "run-7"
    assert row["budget_spent_usd"] == 1.25
    assert row["experiment_id"] == "exp-1"
    assert row["spec_version"] == 1


def test_the_store_never_uses_insert_or_replace():
    """钉住被否掉的写法。

    ``INSERT OR REPLACE`` 省事,代价是漏一列就静默丢一列 —— 而漏列这件事,
    代码审查看不出来,测试也只在恰好断言了那一列时才抓得到。所以这里直接禁掉
    这个写法:要改行就 UPDATE 具名列。
    """
    src = (Path(_MASTV2_ROOT) / "mast" / "conduct" / "store.py").read_text(
        encoding="utf-8")
    assert "INSERT OR REPLACE" not in src.upper()


def test_the_event_and_the_state_share_one_transaction(store, monkeypatch):
    """**变异验证**:让写事件那一半失败,状态那一半必须一起回滚。

    ① 先证明变异生效(record 抛了);② 再看被守卫的东西:状态没变。
    没有这条,一次崩溃会留下「事件说进了等待态,状态行还在 RUNNING」——
    而恢复清算读的正是状态行。
    """
    cid = _make(store)
    store.record(cid, "adopted", changes={"status": "running"})

    def boom(*a, **k):
        raise RuntimeError("写事件失败")

    monkeypatch.setattr(ConductStore, "_insert_event", boom)
    # ① 变异已应用
    with pytest.raises(RuntimeError):
        store.record(cid, "status_change",
                     changes={"status": "paused", "status_reason": "人按了暂停"})
    # ② 状态整体回滚
    assert store.get(cid)["status"] == "running"
    assert store.get(cid)["status_reason"] == ""


# ── 心跳 ─────────────────────────────────────────────────────────────────

def test_heartbeat_is_not_a_state_change(store, clock):
    """心跳只证明 tick 循环在转,**不证明步在推进**。

    所以它不写事件、不动 updated_at —— 停滞告警要靠 heartbeat 年龄 +
    active_run_id + 步的 started_at 三者区分「正常长步」与「线程死了」。
    把这两件事混成一个数,就再也分不出来了。
    """
    cid = _make(store)
    before = store.get(cid)["updated_at"]
    clock.advance(30.0)
    ts = store.touch_heartbeat(cid)
    row = store.get(cid)
    assert row["heartbeat_at"] == ts == clock.t
    assert row["updated_at"] == before
    assert store.events(cid, kind="heartbeat_stall") == []


def test_timestamps_come_from_the_injected_clock(tmp_path):
    """一个 store 里只有一个时间源 —— 两个时间源会让 hold_s / stale_after /
    renotify 的测试在真机上对不上账。"""
    clk = FakeClock(t0=1_000_000.0)
    st = ConductStore(tmp_path / "c.db", clock=clk)
    assert st.now_epoch() == 1_000_000.0
    clk.advance(5.0)
    assert st.now_epoch() == 1_000_005.0


# ── 意图队列 ─────────────────────────────────────────────────────────────

def test_an_unknown_op_is_refused(store):
    cid = _make(store)
    with pytest.raises(ValueError):
        store.enqueue_op(cid, "self_destruct")


def test_abort_and_waive_must_carry_a_reason(store):
    """两条都是「人做了一个会被追问的决定」。理由缺席 = 事后没人说得清。"""
    cid = _make(store)
    with pytest.raises(ValueError):
        store.enqueue_op(cid, "abort", requested_by="op")
    with pytest.raises(ValueError):
        store.enqueue_op(cid, "waive_condition", args={"wait_id": "w1"},
                         requested_by="op")
    store.enqueue_op(cid, "abort", args={"reason": "样品掉了"},
                     requested_by="op")


def test_ack_must_name_which_wait_it_answers(store):
    """wait_id 每次等待唯一 —— 对旧等待点的 ack 必须能被认出来。"""
    cid = _make(store)
    with pytest.raises(ValueError):
        store.enqueue_op(cid, "ack", requested_by="op")


def test_ops_come_out_by_priority_then_arrival(store):
    """abort > takeover > pause > resume > ack > waive > set_attended。"""
    cid = _make(store)
    store.enqueue_op(cid, "set_attended", args={"attended": False})
    store.enqueue_op(cid, "pause")
    store.enqueue_op(cid, "abort", args={"reason": "停"})
    got = [o["op"] for o in store.pending_ops(cid)]
    assert got == ["abort", "pause", "set_attended"]
    assert OP_PRIORITY["abort"] == min(OP_PRIORITY.values())


def test_same_priority_keeps_arrival_order(store):
    cid = _make(store)
    a = store.enqueue_op(cid, "ack", args={"wait_id": "w1"})
    b = store.enqueue_op(cid, "ack", args={"wait_id": "w2"})
    assert [o["op_id"] for o in store.pending_ops(cid)] == [a, b]


def test_consuming_an_op_is_idempotent(store):
    """重放一次 pause 只是多余;重放一次 abort 是在别人重启之后再中止一次。"""
    cid = _make(store)
    op = store.enqueue_op(cid, "pause")
    assert store.consume_op(op) is True
    assert store.consume_op(op) is False
    assert store.pending_ops(cid) == []


def test_ops_for_an_unknown_conduct_are_refused(store):
    with pytest.raises(UnknownConduct):
        store.enqueue_op("nope", "pause")


# ── 事件流 ───────────────────────────────────────────────────────────────

def test_events_are_append_only_and_ordered(store):
    cid = _make(store)
    store.record(cid, "adopted", changes={"status": "running"})
    store.record(cid, "step_started", stage_id="S2", step_id="S2.01",
                 run_id="r1")
    store.record(cid, "step_finished", stage_id="S2", step_id="S2.01",
                 run_id="r1", payload={"ok": True})
    evs = store.events(cid)
    assert [e["kind"] for e in evs] == ["created", "adopted", "step_started",
                                        "step_finished"]
    assert [e["event_id"] for e in evs] == sorted(e["event_id"] for e in evs)
    assert evs[-1]["payload"]["ok"] is True
    assert evs[-1]["run_id"] == "r1"


def test_events_can_be_tailed_from_a_cursor(store):
    cid = _make(store)
    first = store.events(cid)[-1]["event_id"]
    store.record(cid, "adopted", changes={"status": "running"})
    tail = store.events(cid, after_event_id=first)
    assert [e["kind"] for e in tail] == ["adopted"]


def test_events_filter_by_kind(store):
    cid = _make(store)
    store.record(cid, "gate_evaluated", stage_id="S2",
                 payload={"verdict": "pass"})
    store.record(cid, "gate_evaluated", stage_id="S4",
                 payload={"verdict": "wait_operator"})
    assert len(store.events(cid, kind="gate_evaluated")) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
