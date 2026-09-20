"""真探针 —— M1-b 的诚实替身接到真东西上之后,四个口分别要保住什么。

这个文件钉的是**映射**,不是实现:每一条都对应一件「折叠了就会出事」的区分。

1. **busy ≠ 失败**:被仪器令牌拒了不该吃掉这一步的重试预算。而且这条要**走真正
   会跑的那条代码** —— 只测适配器自己的 if,等于测了一个我们自己写的常量。
2. **闩读不到 ≠ 闩没挂**:接不上闩要抛(director 会记成一次读失败),不能回
   一个漂亮的「没挂」。
3. **温度没有 ≠ 0 K**:``value_k=None`` + reason,永远不给一个会让 ``<=5 K``
   立刻成立的数。
4. **重播 ≠ 再发一条**:心愿单的幂等靠一个真的请求 id,不靠比对文案。
"""
from __future__ import annotations

import functools
import sys
import threading
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.conduct.adapters import (  # noqa: E402
    INSTRUMENT_BUSY_KEY,
    OWNER_PREFIX,
    ConductNotifier,
    RuntimeExecutor,
    RuntimeLatch,
    RuntimeTemperature,
)
from mast.conduct.ports import Notification  # noqa: E402
from mast.core.execution_context import ExecutionContext  # noqa: E402
from mast.core.registry import SkillRegistry  # noqa: E402
from mast.core.types import (  # noqa: E402
    ParameterSpec, SafetyLevel, SkillCategory, SkillMetadata, SkillResult,
)
from mast.skills.base import BaseSkill  # noqa: E402


# ── 1. busy ≠ 失败 ─────────────────────────────────────────────────

class _FakeWrite(BaseSkill):
    """一个需要令牌的写技能(category=WRITE、没有 retract/emergency tag)。"""

    def metadata(self):
        return SkillMetadata(
            name="FakeConductWrite", version="1.0.0",
            category=SkillCategory.WRITE, safety_level=SafetyLevel.CONFIRM,
            description="", parameters=[ParameterSpec(name="x", type="float",
                                                      required=False)])

    def execute(self, ctx, params):
        return SkillResult(skill_name="FakeConductWrite", success=True,
                           data=dict(params))


def _registry():
    reg = SkillRegistry()
    reg.register(_FakeWrite)
    return reg


def test_a_real_instrument_busy_carries_the_structural_marker(monkeypatch):
    """**走真正会跑的那条代码。**

    别人真的握着令牌 → ``hold_for_skill`` 抛 ``InstrumentBusy`` →
    ``ExecutionContext.run`` 把它吞成一个失败的 SkillResult。这里要的是:
    那个 SkillResult 上带着**结构化标记**,而不是只有一句给人读的错误文案。

    只把等待时间从 5 s 缩到 50 ms(其余全是真的:真锁、真异常、真的 except 分支)
    —— 否则这条测试要跑 5 秒,而跑得慢的测试迟早会被跳过。
    """
    from mast.core import instrument_lock as il

    real = il.hold_for_skill
    monkeypatch.setattr(il, "hold_for_skill",
                        functools.partial(real, timeout_s=0.05))

    holder_in = threading.Event()
    release = threading.Event()

    def _hold_it():
        with il.instrument_lock().hold(owner="别的入口", skill="LongScan",
                                       timeout_s=1.0):
            holder_in.set()
            release.wait(timeout=5.0)

    t = threading.Thread(target=_hold_it, daemon=True)
    t.start()
    assert holder_in.wait(timeout=5.0), "替身没拿到令牌,这条测试没测到东西"
    try:
        ctx = ExecutionContext(pool=None, state=None, registry=_registry(),
                               owner=f"{OWNER_PREFIX}c1")
        result = ctx.run("FakeConductWrite", {})
    finally:
        release.set()
        t.join(timeout=5.0)

    assert result.success is False
    assert result.data.get(INSTRUMENT_BUSY_KEY) is True, (
        "被令牌拒了的 SkillResult 上没有结构化标记 —— "
        "Director 只能去比对错误文案,而文案改一个字它就静默失灵")
    assert result.data.get("holder", {}).get("owner") == "别的入口"

    out = RuntimeExecutor.to_outcome(result, run_id="r1")
    assert out.busy is True and out.ok is False, (
        "busy 被折叠成了普通失败 —— 一次正常的并发仲裁会吃掉这一步的重试预算")


def test_a_plain_failure_is_not_busy():
    res = SkillResult(skill_name="X", success=False, error="设备说不")
    out = RuntimeExecutor.to_outcome(res, run_id="r2")
    assert out.busy is False and out.ok is False and out.error == "设备说不"


def test_success_carries_data_through_for_produces():
    res = SkillResult(skill_name="X", success=True, data={"x_m": 1.5e-9})
    out = RuntimeExecutor.to_outcome(res, run_id="r3")
    assert out.ok is True and out.data["x_m"] == 1.5e-9


def test_waveforms_are_trimmed_before_they_reach_the_audit_stream():
    """技能 data 会进 ``conduct_events.payload_json`` —— 整条谱塞进去会撑爆库。"""
    res = SkillResult(skill_name="X", success=True,
                      data={"trace": list(range(5000)), "blob": b"\x00" * 4096})
    out = RuntimeExecutor.to_outcome(res, run_id="r4")
    assert out.data["trace"]["_truncated"] is True
    assert out.data["trace"]["len"] == 5000
    assert "bytes len=4096" in out.data["blob"]


def test_executor_reports_what_is_missing_not_a_skill_failure():
    """缺 pool/state/registry ⇒ 说**缺什么**,别报成「技能失败了」。"""

    class _Bare:
        pass

    ex = RuntimeExecutor(_Bare(), aborts=_Aborts(), conduct_id_getter=lambda: "c")
    out = ex.run("Whatever", {}, run_id="r5")
    assert out.ok is False and "缺少" in out.error
    assert "connection_pool" in out.error


class _Aborts:
    def __init__(self):
        self.events = {}

    def get(self, run_id):
        return self.events.get(run_id)


# ── 2. 闩:读不到 ≠ 没挂 ──────────────────────────────────────────

class _Rt:
    def __init__(self, latch=None, temp=None):
        self._latch = latch
        self._temp = temp

    def emergency_latch_state(self):
        if self._latch is None:
            raise RuntimeError("no latch")
        return self._latch

    def latest_temperature(self, channel=None):
        return self._temp


def test_latch_maps_all_three_fields():
    st = RuntimeLatch(_Rt(latch={"latched": True, "abort_set": True,
                                 "why": "环境告警:温度越窗"})).state()
    assert st.latched and st.abort_set
    assert st.why == "环境告警:温度越窗", "「为什么」丢了,解闩的人只能自己编一个"


def test_latch_that_cannot_be_read_raises_rather_than_reporting_clear():
    """runtime 上根本没有这个方法 ⇒ **抛**。

    回一个 ``latched=False`` 会让「读不到闩」的部署长得和「闩没挂」一模一样,
    而这两件事一个要修接线、一个可以继续跑。
    """
    class _NoLatch:
        pass

    with pytest.raises(RuntimeError):
        RuntimeLatch(_NoLatch()).state()


# ── 3. 温度:没有 ≠ 0 K ───────────────────────────────────────────

def test_temperature_without_a_source_is_none_not_zero_kelvin():
    class _NoTemp:
        pass

    r = RuntimeTemperature(_NoTemp()).read()
    assert r.value_k is None, "0 K 会让 `<= 5 K` 这类条件立刻成立"
    assert r.reason


def test_temperature_read_that_explodes_degrades_to_no_source():
    class _Boom:
        def latest_temperature(self, channel=None):
            raise OSError("COM13 被别人占着")

    r = RuntimeTemperature(_Boom()).read()
    assert r.value_k is None and r.reason


def test_temperature_delegates_to_the_public_port():
    from mast.core.temperature import TempReading

    reading = TempReading(channel="SPM", value_k=4.6, age_s=12.0, source="lakeshore")
    assert RuntimeTemperature(_Rt(temp=reading)).read().value_k == 4.6


# ── 4. 通知 / 心愿单 ──────────────────────────────────────────────

class _Bus:
    def __init__(self):
        self.frames = []

    def publish_conduct_alert(self, conduct_id, **kw):
        self.frames.append({"conduct_id": conduct_id, **kw})


class _Board:
    def __init__(self):
        self.posts = []
        self.rows = {}
        self._n = 0

    def post_agent_request(self, agent_id, message, **kw):
        self._n += 1
        rec = {"id": f"r-{self._n}", "agent_id": agent_id, "message": message,
               "status": "pending", **kw}
        self.posts.append(rec)
        self.rows[rec["id"]] = rec
        return rec

    def get_request(self, rid):
        return self.rows.get(rid)


def _notifier():
    bus, board = _Bus(), _Board()
    return ConductNotifier(experiment_id_getter=lambda: "exp-1",
                            board=board, bus=bus), bus, board


def test_a_wait_goes_to_the_wishlist_and_carries_the_conduct_id():
    n, bus, board = _notifier()
    rid = n.request_operator_action(Notification(
        kind="wait", conduct_id="c1", message="请换样品并降温",
        payload={"wait_id": "w1"}))
    assert rid == "r-1"
    assert board.posts[0]["agent_id"] == "conduct:c1"
    assert board.posts[0]["experiment_id"] == "exp-1"
    assert bus.frames and bus.frames[0]["code"] == "wait"


def test_renotify_does_not_pile_up_a_second_open_request():
    """重播 = 再打一帧,**不是**再发一条心愿单。

    幂等靠 ``request_id`` 这个真的 id;文案里带着「还缺什么」,每次都不一样,
    所以靠比对文案的去重在这里必然失效。
    """
    n, bus, board = _notifier()
    rid = n.request_operator_action(Notification(
        kind="wait", conduct_id="c1", message="请换样品并降温",
        payload={"wait_id": "w1"}))
    again = n.request_operator_action(Notification(
        kind="wait_renotify", conduct_id="c1",
        message="请换样品并降温(还缺:物理条件)",
        payload={"wait_id": "w1", "request_id": rid}))
    assert again == rid
    assert len(board.posts) == 1, "同一个等待点堆了第二条心愿单"
    assert len(bus.frames) == 2, "重播那一帧没打出去"


def test_a_resolved_request_is_re_posted_on_the_next_renotify():
    """人把上一条标成 done 之后又没做完 ⇒ 重播要真的再发一条。

    「还开着就不重发」的反面必须成立,否则这条等待会永远没有活着的请求。
    """
    n, bus, board = _notifier()
    rid = n.request_operator_action(Notification(
        kind="wait", conduct_id="c1", message="请换样品", payload={"wait_id": "w1"}))
    board.rows[rid]["status"] = "done"
    again = n.request_operator_action(Notification(
        kind="wait_renotify", conduct_id="c1", message="请换样品(还缺:人的确认)",
        payload={"wait_id": "w1", "request_id": rid}))
    assert again != rid and len(board.posts) == 2


def test_plain_alerts_do_not_open_wishlist_rows():
    """告警(stale / yielding / budget)只播报,不往心愿单堆待办事项。"""
    n, bus, board = _notifier()
    assert n.request_operator_action(Notification(
        kind="stale", conduct_id="c1", message="温度读不到", severity="warn")) == ""
    assert board.posts == []
    assert bus.frames[0]["severity"] == "warn"


def test_a_broken_notification_channel_never_changes_the_state_machine():
    """通知发不出去只记日志 —— 状态机照旧。

    「通知通道自拒 ⇒ 锁死却一声不吭」是本仓真出过的事故形状,但它的解法是
    **让通知失败可见**,不是让它去改状态。
    """
    class _DeadBus:
        def publish_conduct_alert(self, *a, **kw):
            raise RuntimeError("总线没了")

    class _DeadBoard:
        def post_agent_request(self, *a, **kw):
            raise RuntimeError("板子没了")

        def get_request(self, rid):
            raise RuntimeError("板子没了")

    n = ConductNotifier(board=_DeadBoard(), bus=_DeadBus())
    assert n.request_operator_action(Notification(
        kind="wait", conduct_id="c1", message="请来一下")) == ""
    assert len(n.sent) == 1, "发失败了也要留下「试过」的记录"


# ── 5. SAFE 也管得住 conduct 这一路（2026-08-27） ────────────────────

class _FakePulse(BaseSkill):
    """一个声明了电脉冲能力的假技能。

    conduct 的每一步都经 ``RuntimeExecutor`` → ``ExecutionContext.run``；
    在模式闸补进 ``run()`` 之前，那条路**完全看不到** SAFE —— 模板里含
    TipPulse 的一步在安全模式下真的会把脉冲打出去。
    """

    def metadata(self):
        return SkillMetadata(
            name="FakeConductPulse", version="1.0.0",
            category=SkillCategory.WRITE, safety_level=SafetyLevel.AUTO,
            capabilities=frozenset({"bias_pulse"}),
            description="", parameters=[])

    def execute(self, ctx, params):
        raise AssertionError("SAFE 下不该跑到 execute —— 脉冲已经出去了")


def test_safe_mode_stops_a_pulse_on_the_conduct_path():
    from mast.core.operating_mode import bind_mode_source

    reg = SkillRegistry()
    reg.register(_FakePulse)
    bind_mode_source(lambda: "safe")
    try:
        ctx = ExecutionContext(pool=None, state=None, registry=reg,
                               owner=f"{OWNER_PREFIX}c-safe")
        res = ctx.run("FakeConductPulse", {})
    finally:
        bind_mode_source(None)   # 泄漏一个绑定会翻掉后面每个测试的针尖判定

    assert res.success is False
    assert "safe_mode_tip_processing_blocked" in (res.error or "")

    out = RuntimeExecutor.to_outcome(res, run_id="r-safe")
    assert out.ok is False and out.busy is False, (
        "模式拒绝被折叠成 busy —— Director 会当成并发仲裁去重试，"
        "而这一步在 SAFE 下永远不该成功")


def test_auto_mode_lets_the_same_conduct_step_run():
    """反例：闸只在 SAFE/SEMI 起作用，AUTO 下这一步照跑。

    没有这一条，上面那条可能绿在「这个假技能本来就跑不起来」上。
    """
    from mast.core.operating_mode import bind_mode_source

    class _Ok(_FakePulse):
        def execute(self, ctx, params):
            return SkillResult(skill_name="FakeConductPulse", success=True)

    reg = SkillRegistry()
    reg.register(_Ok)
    bind_mode_source(lambda: "auto")
    try:
        ctx = ExecutionContext(pool=None, state=None, registry=reg,
                               owner=f"{OWNER_PREFIX}c-auto")
        res = ctx.run("FakeConductPulse", {})
    finally:
        bind_mode_source(None)
    assert res.success is True, res.error
