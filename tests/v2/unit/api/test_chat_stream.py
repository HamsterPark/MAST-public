"""Chat streaming SSE endpoint (degraded path).

With no ConversationEngine wired, POST /api/agents/{id}/chat must stream a single
error frame (degraded) then done — and the stream must TERMINATE (no hang)."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import pytest

from mast.api.context import AppContext
from mast.api.routes import chat_stream
from mast.api.routes.chat_stream import chat_abort_hook, router


@pytest.fixture(autouse=True)
def _clean_turn_registry():
    """进程级登记表：上一条用例留下的活跃轮次会让「没有正在跑的轮次」凭空变绿。"""
    chat_stream._reset_turn_registry_for_tests()
    yield
    chat_stream._reset_turn_registry_for_tests()


def _client() -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_chat_degraded_streams_error_then_done() -> None:
    c = _client()
    frames = []
    with c.stream(
        "POST", "/api/agents/instrument_control/chat",
        json={"conversation_id": "c1", "user_text": "hi"},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line.startswith("data:"):
                frames.append(json.loads(line[len("data:"):].strip()))
    kinds = [f["kind"] for f in frames]
    assert "error" in kinds
    assert frames[0]["degraded"] is True
    assert kinds[-1] == "done"


def test_abort_hook_no_active_turn() -> None:
    """没有正在跑的回合 ⇒ ``signalled == 0``，并且**说出为什么**。

    契约在 2026-08-11 收紧过：这个 hook 原来返回 bool，且那个 bool 的含义是
    「这个 agent_id 名下有没有一个 Event 对象」—— 而那张表只增不减，所以进程发生过
    第一次对话之后它**永远**为真。现在它返回事实：停到了几轮、哪几轮、没停的话为什么。
    """
    out = chat_abort_hook("nonexistent_agent")
    assert out["signalled"] == 0
    assert out["conversation_ids"] == []
    assert out["reason"], "拒绝要说人话，不能只给一个 0"


def test_abort_hook_stops_lying_once_the_turn_is_over() -> None:
    """**这是原来那个 bug 的核心**：回合结束之后，端点不许再说停到了东西。

    旧实现可能在没有活跃回合时仍返回成功，使调用方误以为停止信号已经送达。
    """
    key, ev = chat_stream._begin_turn("instrument_control", "conv-1")
    live = chat_abort_hook("instrument_control")
    assert live["signalled"] == 1 and live["conversation_ids"] == ["conv-1"]
    assert ev.is_set(), "登记在案的回合必须真的收到信号，不是只被数了一下"

    chat_stream._end_turn(key)
    after = chat_abort_hook("instrument_control")
    assert after["signalled"] == 0, "回合已经结束，却还报停到了东西"
    assert after["reason"]


def test_abort_can_name_one_conversation() -> None:
    """私聊的多个会话都挂在同一个 agent 下 —— 不点名会连带停掉并发的另一条。"""
    k1, ev1 = chat_stream._begin_turn("instrument_control", "conv-1")
    k2, ev2 = chat_stream._begin_turn("instrument_control", "conv-2")
    try:
        out = chat_abort_hook("instrument_control", "conv-2")
        assert out["conversation_ids"] == ["conv-2"]
        assert ev2.is_set() and not ev1.is_set()
    finally:
        chat_stream._end_turn(k1)
        chat_stream._end_turn(k2)
