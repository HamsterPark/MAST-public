"""Contract tests for the agents orchestrator-control routes (HITL resolve /
hold / release / interject).

The router is not yet mounted in mast.api.app (integration wires that), so
each test builds a throwaway FastAPI app and includes the router under /api.

Guarantees asserted:
  * STANDALONE (no live app wired): every endpoint returns 200 with a typed
    degraded body (``degraded=True``, ``ok`` falsey) — NEVER a 500.
  * LIVE (a fake app exposing ``_orch_interrupts`` / ``_agents_api_state`` /
    ``_build_decision``): the relay reaches the live state — resolve hands the
    decision to the blocked worker and sets its Event; hold/release mutate the
    holds dict; interject appends under the lock.
"""

from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.agents_control import router


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx if ctx is not None else AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture()
def client() -> TestClient:
    return _client()


# ── a minimal fake live app exercising the live relay paths ──────────────────
class _FakeLiveApp:
    """Mimics the live MASTApp's operator-control surface (the bits the relay
    touches), using the same dict shapes as gui/app.py:_mount_agents_api."""

    def __init__(self, with_pending: bool = True) -> None:
        self._agents_api_state = {
            "lock": threading.Lock(),
            "holds": {},
            "interjects": [],
            "interrupts": {},
        }
        self._orch_interrupts = {
            "lock": threading.Lock(),
            "pending": {},
            "resolved": {},
            "events": {},
        }
        if with_pending:
            self._orch_interrupts["pending"]["evt1"] = {
                "skill": "SetBias",
                "params": {"bias_v": 1.0},
                "allowed_decisions": ["approve", "reject", "edit"],
            }
            self._orch_interrupts["events"]["evt1"] = threading.Event()

    # NOTE (2026-07-27): this fake deliberately does NOT define `_build_decision`.
    #
    # It used to — "mirrors the static MASTApp._build_decision" — and that is
    # exactly why this suite stayed green while HITL was dead in the field for
    # ~5 weeks. `MASTApp` was deleted with the Gradio layer (7aa1996) and the
    # method was never migrated, so the route's `getattr(app, "_build_decision")`
    # returned None on every real call and every approval answered "live decision
    # builder unavailable". The fake asserted a contract that no real object
    # satisfied — a test that could only ever pass.
    #
    # Translation now lives in mast.core.hitl_decision (pure functions), and
    # tests/v2/unit/api/test_api_live_contract.py statically forbids the API
    # layer from gating behaviour on attributes CoreRuntime lacks.


def _live_ctx(app: _FakeLiveApp) -> AppContext:
    ctx = AppContext()
    ctx.live_app = app  # type: ignore[attr-defined]
    return ctx


# ── STANDALONE degradation (no live app) ─────────────────────────────────────
def test_resolve_degrades_standalone(client: TestClient) -> None:
    r = client.post(
        "/api/agents/instrument_control/interrupts/evt1/resolve",
        json={"decision": "approve"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["degraded"] is True
    assert b["ok"] is False
    assert b["applied"] is False
    assert b["interrupt_id"] == "evt1"


def test_hold_release_degrade_standalone(client: TestClient) -> None:
    for verb in ("hold", "release"):
        r = client.post(f"/api/agents/instrument_control/{verb}")
        assert r.status_code == 200
        b = r.json()
        assert b["degraded"] is True
        assert b["ok"] is False


def test_interject_degrades_standalone(client: TestClient) -> None:
    r = client.post("/api/agents/_supervisor/interject", json={"text": "hello"})
    assert r.status_code == 200
    b = r.json()
    assert b["degraded"] is True
    assert b["ok"] is False


def test_hold_unknown_agent_degrades(client: TestClient) -> None:
    r = client.post("/api/agents/not_an_agent/hold")
    assert r.status_code == 200
    b = r.json()
    assert b["degraded"] is True
    assert "unknown agent" in (b["detail"] or "")


def test_interject_empty_text_degrades(client: TestClient) -> None:
    r = client.post("/api/agents/_supervisor/interject", json={"text": "   "})
    assert r.status_code == 200
    b = r.json()
    assert b["degraded"] is True
    assert b["ok"] is False


# ── LIVE relay paths ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("decision", ["approve", "reject", "edit"])
def test_resolve_live_applies(decision: str) -> None:
    app = _FakeLiveApp(with_pending=True)
    client = _client(_live_ctx(app))
    payload = {"decision": decision, "comment": "ok"}
    if decision == "edit":
        payload["edited_args"] = {"bias_v": 0.5}
    r = client.post(
        f"/api/agents/instrument_control/interrupts/evt1/resolve", json=payload
    )
    assert r.status_code == 200
    b = r.json()
    assert b["ok"] is True
    assert b["applied"] is True
    assert b["status"] == "applied"
    assert b["degraded"] is False
    # decision handed to the blocked worker + Event set.
    assert "evt1" in app._orch_interrupts["resolved"]
    assert app._orch_interrupts["events"]["evt1"].is_set()
    dec = app._orch_interrupts["resolved"]["evt1"]
    expected_type = "approve" if decision == "approve" else decision
    assert dec["type"] == expected_type
    if decision == "edit":
        assert dec["edited_action"]["args"]["bias_v"] == 0.5
    # audit recorded on operator-control state.
    assert app._agents_api_state["interrupts"]["evt1"]["applied"] is True


def test_resolve_live_unknown_id_no_pending() -> None:
    app = _FakeLiveApp(with_pending=False)
    client = _client(_live_ctx(app))
    r = client.post(
        "/api/agents/instrument_control/interrupts/ghost/resolve",
        json={"decision": "approve"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["ok"] is False
    assert b["status"] == "no_pending_interrupt"
    assert b["degraded"] is True
    # audit still recorded with applied=False.
    assert app._agents_api_state["interrupts"]["ghost"]["applied"] is False


def test_hold_then_release_live() -> None:
    app = _FakeLiveApp()
    client = _client(_live_ctx(app))
    r = client.post("/api/agents/data_processing/hold")
    assert r.json()["ok"] is True
    assert app._agents_api_state["holds"].get("data_processing") is True
    r = client.post("/api/agents/data_processing/release")
    assert r.json()["ok"] is True
    assert "data_processing" not in app._agents_api_state["holds"]


def test_hold_all_broadcast_live() -> None:
    app = _FakeLiveApp()
    client = _client(_live_ctx(app))
    r = client.post("/api/agents/__all__/hold")
    assert r.json()["ok"] is True
    assert app._agents_api_state["holds"].get("__all__") is True


def test_interject_live_appends() -> None:
    app = _FakeLiveApp()
    # interjections only queue while a run is ACTIVE (there is a consumer)
    app._agents_api_state["task"] = {"active": True, "conversation_id": ""}
    client = _client(_live_ctx(app))
    r = client.post(
        "/api/agents/instrument_control/interject", json={"text": "stop scanning"}
    )
    b = r.json()
    assert b["ok"] is True
    assert b["degraded"] is False
    assert b["system_addendum_id"]
    queued = app._agents_api_state["interjects"]
    assert len(queued) == 1
    assert queued[0]["text"] == "stop scanning"
    assert queued[0]["agent_id"] == "instrument_control"


def test_interject_without_active_run_is_rejected_with_guidance() -> None:
    """Feedback 2026-07-10 #90: an interjection sent after the run silently died
    used to be queued into a black hole (no consumer) while the operator
    believed it was delivered. It must degrade with an actionable message and
    queue NOTHING."""
    app = _FakeLiveApp()  # no task slot → no active run
    client = _client(_live_ctx(app))
    r = client.post(
        "/api/agents/instrument_control/interject", json={"text": "继续"}
    )
    b = r.json()
    assert b["degraded"] is True
    assert not b.get("ok")
    assert "没有正在运行的任务" in b["detail"]
    assert app._agents_api_state["interjects"] == []


# ── ask_user: the operator ANSWERS a question (not a verdict) ────────────────
def _ask_app() -> _FakeLiveApp:
    app = _FakeLiveApp(with_pending=False)
    app._orch_interrupts["pending"]["ask1"] = {
        "skill": "向用户提问",
        "params": {},
        "allowed_decisions": ["answer"],
        "kind": "ask_user",
        "ask": {
            "question": "先扫哪个区域？",
            "options": [{"label": "A 区", "description": ""},
                        {"label": "B 区", "description": ""}],
            "multi_select": False,
            "allow_custom": True,
            "timeout_action": "continue",
        },
    }
    app._orch_interrupts["events"]["ask1"] = threading.Event()
    return app


def test_ask_user_answer_reaches_the_blocked_worker() -> None:
    """The answer resumes the tool VERBATIM — it must NOT be run through the
    approve/reject/edit translation, which would read "answer" as an unknown
    verb and collapse it to a reject Decision the tool cannot interpret."""
    app = _ask_app()
    client = _client(_live_ctx(app))
    r = client.post(
        "/api/agents/literature/interrupts/ask1/resolve",
        json={"decision": "answer", "selected": ["B 区"], "comment": "更干净"},
    )
    b = r.json()
    assert b["ok"] is True and b["applied"] is True
    assert b["status"] == "applied"
    assert app._orch_interrupts["resolved"]["ask1"] == {
        "selected": ["B 区"], "custom_text": "", "note": "更干净",
    }
    assert app._orch_interrupts["events"]["ask1"].is_set()


def test_ask_user_open_answer_uses_custom_text() -> None:
    app = _ask_app()
    app._orch_interrupts["pending"]["ask1"]["ask"]["options"] = []
    client = _client(_live_ctx(app))
    r = client.post(
        "/api/agents/literature/interrupts/ask1/resolve",
        json={"decision": "answer", "custom_text": "扫左上角"},
    )
    assert r.json()["applied"] is True
    assert app._orch_interrupts["resolved"]["ask1"]["custom_text"] == "扫左上角"


def test_ask_user_invalid_answer_leaves_the_worker_waiting() -> None:
    """An option that was never offered is a correctable mistake, not a reason
    to kill a run. Same contract as ``route_not_allowed``: nothing is written
    into ``resolved`` and the Event is NOT set, so the operator can resubmit."""
    app = _ask_app()
    client = _client(_live_ctx(app))
    r = client.post(
        "/api/agents/literature/interrupts/ask1/resolve",
        json={"decision": "answer", "selected": ["C 区"]},
    )
    b = r.json()
    assert b["status"] == "answer_invalid"
    assert b["applied"] is False
    assert "C 区" in b["detail"]
    assert "ask1" not in app._orch_interrupts["resolved"]
    assert not app._orch_interrupts["events"]["ask1"].is_set()
    # …and the pending row survives so the retry has something to answer.
    assert "ask1" in app._orch_interrupts["pending"]


def test_ask_user_empty_answer_is_refused() -> None:
    app = _ask_app()
    client = _client(_live_ctx(app))
    r = client.post(
        "/api/agents/literature/interrupts/ask1/resolve",
        json={"decision": "answer", "selected": [], "custom_text": "  "},
    )
    assert r.json()["status"] == "answer_invalid"
    assert not app._orch_interrupts["events"]["ask1"].is_set()


# ── ⑰(2026-08-08):闸门没了,端点留着,而且必须**如实说没有** ────────────────
#
# 这两个端点是 2026-08-04 那次事故的带外出口:私聊里按一次「拒绝」把仪器卡到进程
# 重启,而树里唯一的重开入口在一条主聊天永远走不到的路上。
#
# ⑰ 把整条打断链割掉之后,再也没有闸门会关上 —— 那次事故在结构上不可能重现。
# 但端点**不能删**:前端在读、``skills/builtins/hardware_events`` 在读、OpenAPI
# schema 里有它。它们现在的职责是**如实回答「没有闸门」**,而不是 404 或异常。
#
# 下面两条测试因此从「闸门关得上、清得掉」改成「闸门关不上,而端点仍然正常应答」。
# 原来那个 ``held_gate`` fixture 靠 ``interrupt_fn=lambda: None`` 复刻「审批通道
# 死掉」的现场来把闸门做成关闭态;注入口和闸门一起删了,fixture 也就没有了对象。


@pytest.fixture()
def live_gate(tmp_path):
    """一个活着的 BufferHITLMiddleware —— 它记录了事件,但闸门开着。

    仍然要 close():注册表是进程级的,漏一个实例会渗进别的测试。"""
    import asyncio

    from mast.agents._shared.buffer_hitl import make_buffer_hitl_middleware
    from mast.buffer.schemas import make_e_stop
    from mast.buffer.service import BufferService

    async def _build():
        buf = BufferService(wal_path=tmp_path / "buf.sqlite", wal_enabled=False)
        await buf.start()
        mw = make_buffer_hitl_middleware(buffer=buf)
        buf.emit_event(make_e_stop(reason="user", detail="x", seqno=buf.next_seq()))
        await asyncio.sleep(0)
        mw.before_model(state={}, runtime=None)   # 抽干队列,记录事件
        return buf, mw

    buf, mw = asyncio.run(_build())
    try:
        yield mw
    finally:
        mw.close()
        asyncio.run(buf.stop())


def test_gate_endpoint_paths_match_the_string_the_agent_is_told() -> None:
    """Anti-drift pin. 端点路径与 ``buffer_hitl`` 里那三个常量必须一致。

    ⑰ 之后 agent 的拒绝语里不再出现这个路径(没有拒绝语了),但常量与路由仍然是
    两处必须相等的字面量,而前端按常量拼 URL。"""
    from mast.agents._shared.buffer_hitl import (
        GATE_API_PREFIX,
        GATE_RESOLVE_PATH,
        GATE_STATE_PATH,
        gate_resolve_url,
    )

    paths = {r.path for r in _client().app.routes}
    assert GATE_API_PREFIX + GATE_STATE_PATH in paths
    assert GATE_API_PREFIX + GATE_RESOLVE_PATH in paths
    assert gate_resolve_url() == GATE_API_PREFIX + GATE_RESOLVE_PATH


def test_gate_endpoints_work_without_a_live_app(client: TestClient) -> None:
    """NOT a live-app relay: the gate lives on middleware instances in a
    process-wide registry. It must answer in standalone too."""
    r = client.get("/api/agents/hitl-gates")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    r = client.post("/api/agents/hitl-gates/resolve", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True          # 0 reopened is a no-op, not an error


def test_gate_state_endpoint_reports_no_closed_gate(client: TestClient,
                                                    live_gate) -> None:
    """**语义反转的正主。** 原名 ``test_gate_state_endpoint_reports_a_closed_gate``。

    同样的输入(一个 E_STOP 进了缓冲区),断言反过来:``closed == 0``。
    而端点仍然**看得见这条中间件**,并且报出它记了几条 —— 「没有闸门」与
    「没有中间件在跑」必须是两个回答。"""
    body = client.get("/api/agents/hitl-gates").json()
    assert body["ok"] is True
    assert body["closed"] == 0, "又有闸门关上了"
    mine = [g for g in body["gates"]
            if g.get("recorded_not_escalated", 0) >= 1]
    assert mine, "端点看不见这条记录了事件的中间件"
    assert mine[0]["closed"] is False
    assert mine[0]["reask_armed"] is False


def test_gate_resolve_endpoint_is_an_honest_no_op(client: TestClient,
                                                  live_gate) -> None:
    """原名 ``test_gate_resolve_endpoint_reopens_a_wedged_gate``。

    2026-08-04 那次事故要的是「只有 HTTP 的用户能把仪器解卡」。现在没有卡这回
    事,所以这个端点回答的是 ``reopened == 0`` —— **成功地什么都没做**,而不是报错。
    钉住这一点,是因为一个开始 500 的兼容端点会让前端以为后端坏了。"""
    r = client.post("/api/agents/hitl-gates/resolve",
                    json={"note": "阈值未标定造成的误报"})
    body = r.json()
    assert body["ok"] is True
    assert body["reopened"] == 0
    assert live_gate.gate_state()["closed"] is False
    assert client.get("/api/agents/hitl-gates").json()["closed"] == 0
