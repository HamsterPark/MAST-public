"""急停闩必须**能被解开**,而且状态要说得出「为什么」。

## 为什么有这两个端点

2026-08-13 之前:``POST /api/safety/emergency-stop`` 挂闩,而**解**它的东西在
整个系统里不存在 —— ``CoreRuntime.clear_emergency_latch()`` 有实现,但没有任何
路由、没有任何按钮调它,是死代码。闩另有两个**无人参与**的来源(任意 E_STOP
事件、环境告警),所以它可以在没人按过任何按钮的情况下挂上。

真的发生过:``main`` 端口抖了 21 秒,环境监控把「这一次读不到」判成硬
故障 ⇒ 退针 + 挂闩;通知用的 E_STOP 又因为 reason 不在白名单里被静默丢掉。
用户看到的是「什么都没发生」,而此后每一次仪器调用都回他一句「用户已中止
本次运行」。连接 21 秒后自愈,针和仪器全程完好,机器却锁死到进程重启为止。

**能停不能解的开关不是安全措施,是死锁。**

## 这里钉的两件事

1. 解除口存在、接上了、真的调到 app 那个方法(方法能用 ≠ 路由接上了);
2. 状态端点把「挂没挂」和「**为什么**」一起给 —— 只回 True 的接口,读的人
   只能自己编原因,而那正是这次事故的伤害所在。
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

from mast.api.context import AppContext  # noqa: E402
from mast.api.routes.safety import router  # noqa: E402


class _App:
    """最小的 app 替身:只有闩,以及解闩会留下的痕迹。"""

    def __init__(self, latched: bool = False, why: str = "") -> None:
        self._latched = latched
        self._why = why
        self.cleared_with: list[str] = []

    def emergency_latch_state(self) -> dict:
        return {"latched": self._latched, "abort_set": self._latched,
                "why": self._why}

    def clear_emergency_latch(self, why: str = "") -> bool:
        self.cleared_with.append(why)
        was, self._latched, self._why = self._latched, False, ""
        return was


def _client(app_obj=None) -> TestClient:
    api = FastAPI()
    ctx = AppContext()
    if app_obj is not None:
        ctx.live_app = app_obj
    api.state.ctx = ctx
    api.include_router(router, prefix="/api")
    return TestClient(api)


# ── 状态:挂没挂 + 为什么 ────────────────────────────────────────────────

def test_state_reports_latched_and_why():
    c = _client(_App(latched=True, why="环境告警:pressure = 0.001 mbar 越限"))
    r = c.get("/api/safety/emergency-latch")
    assert r.status_code == 200
    body = r.json()
    assert body["latched"] is True
    assert body["abort_set"] is True
    assert "pressure" in body["why"], (
        "状态只说了「闩着」没说为什么 —— 读的人只能自己编一个原因,"
        "而 2026-08-13 他编出来的是「大概是作者停的」。")


def test_no_latch_reads_clean():
    r = _client(_App()).get("/api/safety/emergency-latch")
    assert r.status_code == 200
    assert r.json()["latched"] is False


def test_no_live_app_degrades_instead_of_500():
    """没有活的 app 时要给一个有类型的降级答案,不许 500。"""
    r = _client().get("/api/safety/emergency-latch")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ── 解除 ────────────────────────────────────────────────────────────────

def test_clear_actually_calls_through_to_the_app():
    """路由要真的调到 app 的方法 —— 方法能用 ≠ 路由接上了。

    这条测的正是缺失的那一环:``clear_emergency_latch`` 一直存在、一直能用,
    只是**没有任何东西调它**。
    """
    app_obj = _App(latched=True, why="E_STOP 事件(reason=environment)")
    c = _client(app_obj)
    r = c.post("/api/safety/clear-emergency", json={"reason": "确认过针和样品"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["was_latched"] is True
    assert app_obj.cleared_with == ["确认过针和样品"], (
        f"路由没把解除请求传到 app:{app_obj.cleared_with}")
    assert app_obj._latched is False


def test_clear_reports_what_it_cleared():
    """解掉之后要说出被解掉的那个闩当初**为什么**挂 —— 原因不会因为解了就消失。"""
    c = _client(_App(latched=True, why="环境告警:pressure 越限"))
    body = c.post("/api/safety/clear-emergency", json={}).json()
    assert "pressure" in body["cleared_why"]


def test_clearing_when_nothing_is_latched_is_honest():
    """没锁的时候解,要如实说「本来就没锁」,不许假报成功解除。"""
    body = _client(_App()).post("/api/safety/clear-emergency", json={}).json()
    assert body["ok"] is True
    assert body["was_latched"] is False


def test_clear_without_live_app_degrades():
    r = _client().post("/api/safety/clear-emergency", json={})
    assert r.status_code == 200
    assert r.json()["degraded"] is True
    assert r.json()["ok"] is False


def test_a_broken_app_does_not_500_the_estop_family():
    """app 的钩子自己抛,端点也不许 500 —— 这一族端点是最后的逃生门。"""

    class _Broken(_App):
        def clear_emergency_latch(self, why: str = "") -> bool:
            raise RuntimeError("boom")

    r = _client(_Broken(latched=True)).post("/api/safety/clear-emergency", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert r.json()["errors"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ── 卡住的连接要能手工重连 ──────────────────────────────────────────────

class _Pool:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list = []

    def reconnect_role(self, role: str, why: str = "") -> bool:
        self.calls.append((role, why))
        return self.ok


class _AppWithPool:
    def __init__(self, pool) -> None:
        self._pool = pool


def test_a_wedged_role_can_be_reconnected():
    """2026-08-13:``main`` 半开之后,**全系统没有任何地方能让人手工重连**。

    `ConnectionPool.break_role` 存在但没有调用点(和 `clear_emergency_latch`
    当初一样是够不到的逃生门),唯一出路是重启整个 MAST —— 而那要有人走到
    机器跟前双击。
    """
    pool = _Pool()
    c = _client(_AppWithPool(pool))
    r = c.post("/api/safety/reconnect-role", json={"role": "main", "reason": "半开"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["reconnected"] is True
    assert pool.calls == [("main", "半开")], f"没有传到连接池:{pool.calls}"


def test_a_failed_reconnect_is_reported_honestly():
    """重连没成功就说没成功 —— 不许回一个「ok」让人以为好了。"""
    c = _client(_AppWithPool(_Pool(ok=False)))
    body = c.post("/api/safety/reconnect-role", json={"role": "main"}).json()
    assert body["ok"] is True          # 端点本身工作正常
    assert body["reconnected"] is False  # 但重连没成


def test_reconnect_without_a_live_pool_degrades():
    r = _client().post("/api/safety/reconnect-role", json={"role": "main"})
    assert r.status_code == 200
    assert r.json()["degraded"] is True
    assert r.json()["ok"] is False
