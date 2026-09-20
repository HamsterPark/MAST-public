"""``POST /api/agents/run-task`` 的 ``done_when``：整体拒绝，或原样播种。

## 为什么是「整体拒绝」而不是「夹紧」

``_clamp_waiting_for`` 夹的是**模型**的输出 —— 拒绝了它无处可去，夹到闭集里
至少还能醒过来。这里的来源是**用户 / UI**：静默丢掉一条谓词等于把目标悄悄
改小，而「更容易达成的目标」正是这道闸要防的东西。拒绝无害：没有 ``done_when``
就是今天的行为。

## 为什么「不给就一个键都不多」

一个新键如果无条件出现在 ``initial_state`` 里，「这次没给目标」和「给了个空
目标」就变成了同一件事 —— 而前者必须逐字节保持旧行为。
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.api.context import AppContext  # noqa: E402
from mast.api.routes.orchestrator import router as orch_router  # noqa: E402


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(orch_router, prefix="/api")
    return TestClient(app)


# ── 拒绝 ────────────────────────────────────────────────────────────────

def test_an_unknown_predicate_is_422_and_names_every_problem():
    r = _client().post("/api/agents/run-task", json={
        "task": "扫一张图",
        "done_when": [{"kind": "artifact_present", "field": "analysis"},
                      {"kind": "vibes"},
                      {"kind": "conduct_completed"}],
    })
    assert r.status_code == 422
    d = r.json()["detail"]
    assert len(d["problems"]) == 2, d["problems"]
    assert any("vibes" in p for p in d["problems"])
    assert any("conduct_completed" in p for p in d["problems"])
    # 目录与示例要跟着报文回去 —— 让写错的人在出错的地方拿到闭集，
    # 而不是被一句「请查文档」打发走。
    kinds = {e["kind"] for e in d["done_when_catalog"]}
    assert "artifact_present" in kinds and "best_frame_settled" in kinds
    assert d["example"]


def test_a_field_outside_the_waitable_set_is_422():
    r = _client().post("/api/agents/run-task", json={
        "task": "扫一张图",
        "done_when": [{"kind": "artifact_present", "field": "scan_id"}]})
    assert r.status_code == 422
    assert "scan_id" in str(r.json()["detail"]["problems"])


def test_nothing_starts_when_the_criteria_are_rejected(monkeypatch):
    """拒绝就是**没有开始运行** —— 不是先跑起来再报错。"""
    started: list = []
    import mast.api.routes.orchestrator as mod

    monkeypatch.setattr(mod, "_run_task_stream",
                        lambda *a, **k: started.append(1) or iter(()))
    r = _client().post("/api/agents/run-task",
                       json={"task": "x", "done_when": [{"kind": "nope"}]})
    assert r.status_code == 422 and started == []


# ── 播种 ────────────────────────────────────────────────────────────────

def _seeded(monkeypatch, body) -> dict:
    """跑一次，把 ``initial_state`` 抓出来。"""
    import mast.api.routes.orchestrator as mod

    seen: dict = {}

    class _Orch:
        # 签名照**生产调用点**写：``orchestrator.stream(stream_input, config=cfg,
        # stream_mode=..., subgraphs=True)``。写成位置参数 ``cfg`` 会 TypeError，
        # 而那个异常被 gen() 吞成一条 error 帧 —— 测试于是红在一个假的理由上
        # （或者更糟：断言恰好还成立，绿在一个假的理由上）。
        def stream(self, state, *, config=None, **kw):
            seen["state"] = state
            return iter(())

        def get_state(self, config=None):
            raise RuntimeError("no checkpointer in this test")

    class _App:
        _orchestrator = _Orch()
        _orch_abort = None
        _orch_interrupts = None

    monkeypatch.setattr(mod, "_live_app", lambda ctx: _App())
    c = _client()
    with c.stream("POST", "/api/agents/run-task", json=body) as resp:
        for _ in resp.iter_lines():
            pass
    return seen.get("state") or {}


def test_a_valid_done_when_is_seeded_into_state(monkeypatch):
    st = _seeded(monkeypatch, {
        "task": "扫一张图并写分析",
        "done_when": [{"kind": "artifact_present", "field": "analysis"}]})
    assert "goal" in st, "判据没进 initial_state —— supervisor 那道闸永远不会被触发"
    assert st["goal"]["done_when"] == {
        "all": [{"kind": "artifact_present", "field": "analysis",
                 "allow_preexisting": False}]}
    assert st["goal"]["text"] == "扫一张图并写分析"


def test_goal_text_overrides_the_task_for_display(monkeypatch):
    st = _seeded(monkeypatch, {"task": "很长很长的一段话……",
                               "goal_text": "拿到一张能印的图",
                               "done_when": [{"kind": "best_frame_settled",
                                              "tag": "hunt-1"}]})
    assert st["goal"]["text"] == "拿到一张能印的图"


def test_without_done_when_the_goal_channel_is_explicitly_cleared(monkeypatch):
    """没给判据 ⇒ ``goal`` 被**显式清成 None**，不是「这个键不出现」。

    「不出现」看起来更保守，实际上是一个高危缺陷：``conversation_id`` 复用同一
    个 ``thread_id``，checkpoint 里上一个任务的 goal 没有任何人清 —— 于是第二个
    任务继承它的 done_when 与 baseline，第一跳就「目标已达成」直接结束。

    与紧邻的 ``visit_count: None`` 是同一件事、同一个理由。
    """
    st = _seeded(monkeypatch, {"task": "扫一张图"})
    assert "goal" in st and st["goal"] is None, (
        f"goal 没有被显式清除：{st.get('goal')!r} —— "
        "续接会话时它会继承上一个任务的目标")


def test_a_resumed_thread_does_not_inherit_the_previous_goal(monkeypatch):
    """同一个 conversation 上的第二个任务，通道里写下去的是 None。"""
    st1 = _seeded(monkeypatch, {
        "task": "第一个任务", "conversation_id": "conv-A",
        "done_when": [{"kind": "artifact_present", "field": "analysis"}]})
    assert st1["goal"] and st1["goal"]["done_when"]
    st2 = _seeded(monkeypatch, {"task": "第二个任务",
                                "conversation_id": "conv-A"})
    assert st2["goal"] is None


@pytest.mark.parametrize("shape", [
    {"kind": "artifact_present", "field": "analysis"},                 # 单条
    [{"kind": "artifact_present", "field": "analysis"}],               # 列表 = all
    {"all": [{"kind": "artifact_present", "field": "analysis"}]},      # 显式
    {"any": [{"kind": "artifact_present", "field": "analysis"},
             {"kind": "artifact_present", "field": "draft"}]},
])
def test_the_three_accepted_shapes(monkeypatch, shape):
    st = _seeded(monkeypatch, {"task": "x", "done_when": shape})
    assert st["goal"]["done_when"]
