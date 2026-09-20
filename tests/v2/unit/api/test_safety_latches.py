"""闩族要能**一次看全**:挂没挂 / 为什么 / 谁能解 —— 且读不到就说读不到。

## 为什么需要这一页

2026-08-13 的伤害不是「机器停了」,是**人被送去查一个根本不存在的原因**:
每一次仪器调用都被拒,拒绝语说是他自己中止的,而系统里没有任何一页能回答
「现在到底是什么在拦我」。逐闩各有各的端点也答不了这个问题 —— 读的人得先
知道该问哪一个,而那正是他不知道的。

交接文档 §2 点名五处同形状「能挂不能解」。补完之后各自缺的仍然不一样:
急停闩齐了;连接重连口有了但连接健康度**全仓零读者**;空转记账读不到也解不掉;
看门狗的释放口只是「开始一个新任务」的副作用。把全家摆到同一页上,缺的那一格
会自己显形 —— `without_release` 非空就是一份 bug 报告。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/api/test_safety_latches.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from mast.agents._shared import stall_guard_mw as sg  # noqa: E402
from mast.agents._shared.stall_guard_mw import StallGuardMiddleware  # noqa: E402
from mast.api.context import AppContext  # noqa: E402
from mast.api.routes.safety import router  # noqa: E402


# ── 替身 ────────────────────────────────────────────────────────────────

class _Watchdog:
    def __init__(self, triggered: bool = False) -> None:
        self.is_anomaly_triggered = triggered
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self.is_anomaly_triggered = False


class _Executor:
    def __init__(self, wd) -> None:
        self._watchdog = wd


class _Pool:
    def __init__(self, snap: dict | None = None) -> None:
        self._snap = snap

    def comms_snapshot(self) -> dict:
        if self._snap is None:
            raise RuntimeError("pool is gone")
        return dict(self._snap)


class _App:
    """最小 app 替身。任何一格传 None = 那一格读不到(不是「没挂」)。"""

    def __init__(self, *, latched=False, why="", pool=None, watchdog=None) -> None:
        self._latched = latched
        self._why = why
        self._pool = pool
        self._executor = _Executor(watchdog) if watchdog is not None else None
        self._wd = watchdog

    def emergency_latch_state(self) -> dict:
        return {"latched": self._latched, "abort_set": self._latched,
                "why": self._why}

    def clear_emergency_latch(self, why: str = "") -> bool:
        was, self._latched = self._latched, False
        return was

    def reset_watchdog(self) -> bool:
        if self._wd is None:
            return False
        self._wd.reset()
        return True


def _client(app_obj=None) -> TestClient:
    api = FastAPI()
    ctx = AppContext()
    if app_obj is not None:
        ctx.live_app = app_obj
    api.state.ctx = ctx
    api.include_router(router, prefix="/api")
    return TestClient(api)


def _healthy_pool() -> _Pool:
    return _Pool({"state": "CLOSED", "streak": 0, "fail_threshold": 4,
                  "cooldown_remaining_s": 0.0, "tripped_total": 0,
                  "last_reason": ""})


def _by_id(body: dict) -> dict:
    return {row["id"]: row for row in body["latches"]}


@pytest.fixture(autouse=True)
def _clean_ladder(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    sg.release(why="test setup")
    yield
    sg.release(why="test teardown")


# ── 闭集 ────────────────────────────────────────────────────────────────

def test_lists_every_known_stop_source():
    """闭集枚举。一个闩因为没人注册而从列表里消失,和它不存在症状一模一样。"""
    body = _client(_App(pool=_healthy_pool(), watchdog=_Watchdog())).get(
        "/api/safety/latches").json()
    assert set(_by_id(body)) == {
        "emergency", "stall_guard", "comms_breaker", "watchdog_anomaly"}


def test_every_latch_names_a_release_path():
    """**这条就是族规本身**:`release` 为空 = 一份 bug 报告,不是正常状态。"""
    body = _client(_App(pool=_healthy_pool(), watchdog=_Watchdog())).get(
        "/api/safety/latches").json()
    assert body["without_release"] == [], (
        f"这些闩挂得上、解不掉:{body['without_release']}")
    for row in body["latches"]:
        assert row["release_actor"], f"{row['id']} 没说谁能解"
        assert row["effect"], f"{row['id']} 没说挂着的时候会发生什么"


# ── 三态:读不到 ≠ 没挂 ──────────────────────────────────────────────────

def test_unreadable_is_null_not_false():
    """没有活的 app 时,**靠 app 取数的那几格**是「读不到」,不许折叠成「没挂」。

    把一次读取失败写成一个看起来完全合理的具体答案,是 2026-08-13 那一族事故的
    共同形状(五次,全在不同子系统)。

    空转记账是例外,而且是**正确**的例外:它的真源是进程内的中间件注册表,
    不经过 app,所以「没有活的 app」时它给出的 False 是一个真答案,不是折叠。
    这个区别本身要钉住 —— 否则下一个人会「顺手」把它也改成 None,把一个读得到
    的事实变成读不到。
    """
    body = _client().get("/api/safety/latches").json()
    rows = _by_id(body)
    for _id in ("emergency", "comms_breaker", "watchdog_anomaly"):
        assert rows[_id]["latched"] is None, (
            f"{_id} 把读不到折叠成了 {rows[_id]['latched']}")
        assert rows[_id]["unreadable_reason"], f"{_id} 说读不到却不说为什么"
    assert rows["stall_guard"]["latched"] is False
    assert rows["stall_guard"]["unreadable_reason"] == ""
    assert body["any_latched"] is None, "有三格读不到,总结却敢说「没事」"
    assert set(body["unreadable"]) == {"emergency", "comms_breaker",
                                       "watchdog_anomaly"}
    assert body["degraded"] is True


def test_any_latched_is_null_when_something_is_unreadable():
    """读得到的都没挂、但有一格读不到 ⇒ 诚实的答案是「不知道」。"""
    body = _client(_App(pool=None, watchdog=_Watchdog())).get(
        "/api/safety/latches").json()
    rows = _by_id(body)
    assert rows["emergency"]["latched"] is False
    assert rows["comms_breaker"]["latched"] is None
    assert body["any_latched"] is None
    assert "comms_breaker" in body["unreadable"]


def test_all_clear_reads_as_false_not_null():
    body = _client(_App(pool=_healthy_pool(), watchdog=_Watchdog())).get(
        "/api/safety/latches").json()
    assert body["any_latched"] is False and body["unreadable"] == []


def test_a_reader_that_raises_is_reported_not_swallowed():
    body = _client(_App(pool=_Pool(None), watchdog=_Watchdog())).get(
        "/api/safety/latches").json()
    row = _by_id(body)["comms_breaker"]
    assert row["latched"] is None
    assert "pool is gone" in row["unreadable_reason"]


# ── 每一格都要说得出「为什么」 ──────────────────────────────────────────

def test_emergency_latch_carries_its_why():
    body = _client(_App(latched=True, why="环境告警:pressure 越限",
                        pool=_healthy_pool(), watchdog=_Watchdog())).get(
        "/api/safety/latches").json()
    row = _by_id(body)["emergency"]
    assert row["latched"] is True and "pressure" in row["why"]
    assert body["any_latched"] is True


def test_comms_breaker_is_read_here_for_the_first_time():
    """`comms_snapshot()` 一直存在、全仓零读者(生产方接好、消费方不存在的那一族)。"""
    pool = _Pool({"state": "OPEN", "streak": 5, "fail_threshold": 4,
                  "cooldown_remaining_s": 12.5, "tripped_total": 2,
                  "last_reason": "WinError 10054"})
    body = _client(_App(pool=pool, watchdog=_Watchdog())).get(
        "/api/safety/latches").json()
    row = _by_id(body)["comms_breaker"]
    assert row["latched"] is True
    assert "10054" in row["why"] and "5" in row["why"]
    assert row["detail"]["cooldown_remaining_s"] == 12.5


def test_watchdog_latch_says_it_is_the_inverse_kind():
    """看门狗的闩挂着**不拒绝任何东西** —— 挂着代表网撤了。
    和「拒绝一切」的闩并成一个 bool 会让人读反。"""
    body = _client(_App(pool=_healthy_pool(), watchdog=_Watchdog(True))).get(
        "/api/safety/latches").json()
    row = _by_id(body)["watchdog_anomaly"]
    assert row["latched"] is True and row["why"]
    assert "不再拒绝任何动作" in row["effect"]


# ── 空转记账:挂上 → 看得见为什么 → 解 → 下游放行 ────────────────────────

class _Req:
    def __init__(self, messages):
        self.messages = list(messages)


def _arm_the_stall_ladder() -> StallGuardMiddleware:
    g = StallGuardMiddleware(agent_name="instrument_control")
    for _ in range(2):
        msgs = [HumanMessage(content="做点什么")] + [
            ToolMessage(content="[GetBias] failed: TimeoutError: timed out after 3.02s",
                        tool_call_id="tc", name="GetBias", status="error")
            for _ in range(3)]
        g.wrap_model_call(_Req(msgs), lambda r: AIMessage(content="<model ran>"))
    return g


def test_stall_ladder_shows_up_with_why_and_can_be_released_end_to_end():
    g = _arm_the_stall_ladder()
    c = _client(_App(pool=_healthy_pool(), watchdog=_Watchdog()))

    row = _by_id(c.get("/api/safety/latches").json())["stall_guard"]
    assert row["latched"] is True, "上膛了却在闩族页面上看不见"
    assert "GetBias" in row["why"] and "instrument_control" in row["why"]
    assert row["since"] and row["since"] > 0, "「何时」必须和「挂没挂」一起给"
    assert row["detail"]["armed"] == 1

    r = c.post("/api/safety/clear-stall-guard", json={"reason": "确认是偶发超时"})
    assert r.status_code == 200
    assert r.json()["released"] == 1, f"路由没有传到中间件:{r.json()}"
    assert "GetBias" in r.json()["signatures"][0]

    after = _by_id(c.get("/api/safety/latches").json())["stall_guard"]
    assert after["latched"] is False and after["detail"]["tracked"] == 0
    del g


def test_clear_stall_guard_can_be_scoped():
    ic = _arm_the_stall_ladder()
    c = _client(_App(pool=_healthy_pool(), watchdog=_Watchdog()))
    body = c.post("/api/safety/clear-stall-guard",
                  json={"agent": "data_processing"}).json()
    assert body["released"] == 0, "指名解别的 agent,却把这个 agent 的解了"
    assert _by_id(c.get("/api/safety/latches").json())["stall_guard"]["latched"] is True
    del ic


def test_armed_rows_are_never_truncated_away():
    """还在爬阶梯的可以截断,**上膛的一条都不能省** —— 那正是要回答的问题。
    而且截断了要说出来:悄悄少给几行的诊断口,读的人会拿它当全集。"""
    g = _arm_the_stall_ladder()
    for k in range(40):                       # 灌一堆只到第一级的噪声签名
        msgs = [HumanMessage(content="x")] + [
            ToolMessage(content=f"[Noise{k}] failed: boom", tool_call_id="t",
                        name=f"Noise{k}", status="error") for _ in range(3)]
        g.wrap_model_call(_Req(msgs), lambda r: AIMessage(content="<model ran>"))

    detail = _by_id(_client(_App()).get("/api/safety/latches").json())[
        "stall_guard"]["detail"]
    assert detail["armed"] == 1 and detail["truncated"] is True
    assert any(r["armed"] and r["tool"] == "GetBias" for r in detail["rows"])
    assert detail["tracked"] == 41, "总数要如实报,截断的是展示不是计数"
    del g


def test_clearing_an_empty_ladder_is_honest():
    body = _client(_App()).post("/api/safety/clear-stall-guard", json={}).json()
    assert body["ok"] is True and body["released"] == 0


# ── 看门狗重新布防 ──────────────────────────────────────────────────────

def test_watchdog_can_be_rearmed_without_starting_a_task():
    """唯一的重新布防口原本挂在「开始一个新任务」的副作用上 —— 想恢复保护
    就得先启动一个任务,离「没有释放口」只差一步。"""
    wd = _Watchdog(True)
    c = _client(_App(pool=_healthy_pool(), watchdog=wd))
    body = c.post("/api/safety/rearm-watchdog").json()
    assert body["ok"] is True and body["rearmed"] is True
    assert body["was_latched"] is True, "要说出重新布防之前它挂着没有"
    assert wd.resets == 1 and wd.is_anomaly_triggered is False
    assert _by_id(c.get("/api/safety/latches").json())[
        "watchdog_anomaly"]["latched"] is False


def test_rearm_without_a_live_app_degrades_instead_of_500():
    r = _client().post("/api/safety/rearm-watchdog")
    assert r.status_code == 200
    assert r.json()["degraded"] is True and r.json()["ok"] is False


# ── 这一族端点是最后的逃生门:自己坏掉也不许 500 ─────────────────────────

def test_a_broken_app_never_500s_the_latch_page():
    class _Broken(_App):
        def emergency_latch_state(self):
            raise RuntimeError("boom")

    r = _client(_Broken(pool=_healthy_pool(), watchdog=_Watchdog())).get(
        "/api/safety/latches")
    assert r.status_code == 200
    row = _by_id(r.json())["emergency"]
    assert row["latched"] is None and "boom" in row["unreadable_reason"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
