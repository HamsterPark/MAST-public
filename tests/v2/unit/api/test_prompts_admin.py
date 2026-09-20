"""Contract tests for 高级管理 → 上下文注入 (``/api/admin/prompts*``).

Covers the two halves of the feature and, above all, the honesty rule: a block
that needs live hardware or per-request state must come back EMPTY with a
reason, never filled with a plausible sample. A fabricated block would be
debugged as if it were real — which is precisely the class of failure that
produced the 2026-07-27 coordinate incident in the first place.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.admin.override_store import ConfigOverrideRegistry
from mast.api.context import AppContext
from mast.api.routes.admin import router


@pytest.fixture
def overrides_dir(tmp_path: Path) -> Path:
    """A throwaway ConfigOverrideRegistry rooted in tmp_path (singleton reset)."""
    d = tmp_path / "overrides"
    ConfigOverrideRegistry.reset()
    ConfigOverrideRegistry.get(d)
    yield d
    ConfigOverrideRegistry.reset()


@pytest.fixture
def client(overrides_dir: Path) -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_capture():
    from mast.prompts import capture as cap

    cap.get_ring().clear()
    yield
    cap.get_ring().clear()


# ── inventory ────────────────────────────────────────────────────────────────

def test_list_returns_the_whole_inventory(client: TestClient):
    body = client.get("/api/admin/prompts").json()
    assert body["degraded"] is False
    assert body["count"] == len(body["items"]) >= 20
    ids = {i["id"] for i in body["items"]}
    # the eight agent-facing statics + the router + the middleware blocks
    assert "agent.instrument_control.system" in ids
    assert "orchestrator.router.system" in ids
    assert "mw.live_state" in ids


def test_agent_system_prompts_carry_real_text(client: TestClient):
    items = {i["id"]: i for i in client.get("/api/admin/prompts").json()["items"]}
    ic = items["agent.instrument_control.system"]
    assert ic["availability"] == "static"
    assert ic["overridable"] is True
    assert ic["default_chars"] > 1000       # the real prompt, not a stub
    assert ic["preview"].strip()


def test_unrenderable_blocks_are_empty_with_a_reason_not_a_fake(client: TestClient):
    """The rule this whole feature stands on."""
    items = {i["id"]: i for i in client.get("/api/admin/prompts").json()["items"]}
    for pid in ("mw.live_state", "mw.memory_recall", "mw.safety_gate.block",
                "mw.resume_context.experiment", "mw.stall_guard.nudge"):
        row = items[pid]
        assert row["availability"] in ("needs_hardware", "needs_request"), pid
        assert row["default_chars"] == 0, f"{pid} produced text it cannot know"
        assert row["preview"] == "", f"{pid} previewed invented text"
        assert row["unavailable_reason"].strip(), f"{pid} gave no reason"
        assert row["overridable"] is False


def test_live_state_reason_points_at_the_capture_not_a_sample(client: TestClient):
    row = client.get("/api/admin/prompts/mw.live_state").json()
    assert row["default_text"] == ""
    assert row["effective_text"] == ""
    assert "无法离线渲染" in row["unavailable_reason"]


def test_detail_returns_full_bodies(client: TestClient):
    body = client.get("/api/admin/prompts/mw.mode_belief.safe").json()
    assert "安全模式" in body["default_text"]
    assert body["effective_text"] == body["default_text"]
    assert body["override_text"] is None
    assert body["overridden"] is False


def test_unknown_id_is_404(client: TestClient):
    r = client.get("/api/admin/prompts/does.not.exist")
    assert r.status_code == 404
    assert r.json()["unavailable_reason"]


# ── override write / clear ───────────────────────────────────────────────────

def test_write_override_changes_the_effective_text(client: TestClient):
    r = client.post("/api/admin/prompts/mw.mode_belief.safe",
                    json={"text": "自定义安全信念"})
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["overridden"] is True

    body = client.get("/api/admin/prompts/mw.mode_belief.safe").json()
    assert body["effective_text"] == "自定义安全信念"
    assert body["override_text"] == "自定义安全信念"
    assert "安全模式" in body["default_text"]      # default still visible for diffing
    assert body["overridden"] is True


def test_override_is_written_to_disk(client: TestClient, overrides_dir: Path):
    client.post("/api/admin/prompts/sub.tool_refine", json={"text": "精炼提示 v2"})
    stored = overrides_dir / "prompt_overrides.json"
    assert stored.exists()
    assert "精炼提示 v2" in stored.read_text(encoding="utf-8")


def test_override_survives_a_restart(client: TestClient, overrides_dir: Path):
    """The actual "改了以后重启还在" claim, exercised the only honest way:
    throw the process-level singleton away and rebuild it from disk."""
    client.post("/api/admin/prompts/mw.prefill_guard.continue",
                json={"text": "CONTINUE-AFTER-RESTART"})

    ConfigOverrideRegistry.reset()                 # ← simulated restart
    ConfigOverrideRegistry.get(overrides_dir)

    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    fresh = TestClient(app)
    body = fresh.get("/api/admin/prompts/mw.prefill_guard.continue").json()
    assert body["effective_text"] == "CONTINUE-AFTER-RESTART"
    assert body["overridden"] is True


def test_delete_restores_the_default(client: TestClient):
    client.post("/api/admin/prompts/mw.mode_belief.semi", json={"text": "临时"})
    r = client.delete("/api/admin/prompts/mw.mode_belief.semi")
    assert r.status_code == 200 and r.json()["overridden"] is False
    body = client.get("/api/admin/prompts/mw.mode_belief.semi").json()
    assert body["effective_text"] == body["default_text"]
    assert body["override_text"] is None


def test_empty_text_clears_rather_than_injecting_nothing(client: TestClient):
    client.post("/api/admin/prompts/mw.mode_belief.semi", json={"text": "临时"})
    client.post("/api/admin/prompts/mw.mode_belief.semi", json={"text": "   "})
    body = client.get("/api/admin/prompts/mw.mode_belief.semi").json()
    assert body["effective_text"] == body["default_text"]
    assert body["effective_text"].strip()          # never a blank system prompt


def test_computed_block_rejects_a_text_override(client: TestClient):
    r = client.post("/api/admin/prompts/mw.live_state", json={"text": "假的实时状态"})
    assert r.status_code == 400
    assert r.json()["ok"] is False
    assert "运行时状态计算" in r.json()["message"]


def test_oversized_override_is_rejected(client: TestClient):
    from mast.prompts.overrides import MAX_OVERRIDE_CHARS

    r = client.post("/api/admin/prompts/sub.tool_refine",
                    json={"text": "x" * (MAX_OVERRIDE_CHARS + 1)})
    assert r.status_code == 400
    assert "过长" in r.json()["message"]


def test_overridden_count_tracks_writes(client: TestClient):
    assert client.get("/api/admin/prompts").json()["overridden_count"] == 0
    client.post("/api/admin/prompts/sub.tool_refine", json={"text": "a"})
    client.post("/api/admin/prompts/mw.mode_belief.safe", json={"text": "b"})
    assert client.get("/api/admin/prompts").json()["overridden_count"] == 2


def test_agent_prompt_write_says_when_it_takes_effect(client: TestClient):
    """An agent prompt is frozen at graph-build time — the UI must not imply
    the edit is live on the next message."""
    msg = client.post("/api/admin/prompts/agent.paper_review.system",
                      json={"text": "新的审稿人设定"}).json()["message"]
    assert "重建" in msg or "重启" in msg


# ── capture ("agent 实际收到了什么") ─────────────────────────────────────────

def test_capture_is_empty_and_says_so_honestly(client: TestClient):
    body = client.get("/api/admin/prompt-capture").json()
    assert body["count"] == 0 and body["items"] == []
    assert "不做离线模拟渲染" in body["note"]
    assert body["capacity"] > 0


def test_capture_records_a_real_request(client: TestClient):
    from langchain_core.messages import HumanMessage, SystemMessage

    from mast.prompts import capture as cap

    cap.record([SystemMessage(content="SYS + 注入块"), HumanMessage(content="扫图")],
               source="instrument_control", model_id="kimi-k3", provider="moonshot")

    lst = client.get("/api/admin/prompt-capture").json()
    assert lst["count"] == 1
    row = lst["items"][0]
    assert row["source"] == "instrument_control"
    assert row["model_id"] == "kimi-k3"
    assert row["message_count"] == 2
    assert row["system_chars"] == len("SYS + 注入块")
    assert row["age_s"] >= 0

    detail = client.get("/api/admin/prompt-capture/0").json()
    assert detail["found"] is True
    assert [m["role"] for m in detail["messages"]] == ["system", "human"]
    assert detail["messages"][0]["content"] == "SYS + 注入块"


def test_capture_newest_first(client: TestClient):
    from mast.prompts import capture as cap

    for i in range(3):
        cap.record([], source=f"agent{i}")
    items = client.get("/api/admin/prompt-capture").json()["items"]
    assert [i["source"] for i in items] == ["agent2", "agent1", "agent0"]


def test_capture_ring_is_bounded(client: TestClient):
    from mast.prompts import capture as cap

    # total_seen is a process-lifetime counter (clear() empties the ring but
    # deliberately does NOT rewrite history), so compare against a baseline.
    before = client.get("/api/admin/prompt-capture").json()["total_seen"]
    for i in range(cap.MAX_SNAPSHOTS + 5):
        cap.record([], source=f"a{i}")
    body = client.get("/api/admin/prompt-capture").json()
    assert body["count"] == cap.MAX_SNAPSHOTS
    assert body["total_seen"] - before == cap.MAX_SNAPSHOTS + 5


def test_capture_truncation_reports_the_original_length(client: TestClient):
    """A clipped body must never read as a short prompt."""
    from langchain_core.messages import SystemMessage

    from mast.prompts import capture as cap

    big = "x" * (cap._MAX_MESSAGE_CHARS + 500)
    cap.record([SystemMessage(content=big)], source="ic")
    detail = client.get("/api/admin/prompt-capture/0").json()
    msg = detail["messages"][0]
    assert msg["truncated"] is True
    assert msg["chars"] == len(big)                 # ORIGINAL length reported
    assert len(msg["content"]) < len(big)


def test_capture_missing_index_is_404(client: TestClient):
    assert client.get("/api/admin/prompt-capture/7").status_code == 404


# ── seq:index 会挪位，seq 不会(2026-08-18,聊天页内联查看用它) ──────────────


def test_index_renumbers_on_every_call_but_seq_does_not(client: TestClient):
    """**这就是 by-seq 存在的理由。**

    ``index`` 是「从新往旧数第几个」。列表拿到 index=0,中间落了一次模型调用,
    再按 index=0 取详情 —— 拿到的是**另一个请求**,而回包里没有任何字段说这件事。
    人点管理页时看不见(两次点击之间不会有请求);聊天页边跑边看时它是常态。
    """
    from langchain_core.messages import SystemMessage

    from mast.prompts import capture as cap

    cap.record([SystemMessage(content="第一次")], source="ic")
    first = client.get("/api/admin/prompt-capture").json()["items"][0]
    assert first["index"] == 0

    cap.record([SystemMessage(content="第二次")], source="ic")

    # 按 index 取 —— 拿到的是新的那次(悄悄换了人)。
    by_index = client.get("/api/admin/prompt-capture/0").json()
    assert by_index["messages"][0]["content"] == "第二次"

    # 按 seq 取 —— 还是当初那一次。
    by_seq = client.get(
        f"/api/admin/prompt-capture/by-seq/{first['seq']}").json()
    assert by_seq["found"] is True
    assert by_seq["seq"] == first["seq"]
    assert by_seq["messages"][0]["content"] == "第一次"


def test_seq_is_monotonic_and_unique(client: TestClient):
    from mast.prompts import capture as cap

    for _ in range(5):
        cap.record([], source="ic")
    seqs = [i["seq"] for i in
            client.get("/api/admin/prompt-capture").json()["items"]]
    assert len(set(seqs)) == len(seqs), f"seq 撞号了:{seqs}"
    assert seqs == sorted(seqs, reverse=True), "items 是新的在前,seq 也该是"


def test_an_evicted_seq_is_404_not_a_different_snapshot(client: TestClient):
    """挤出去的那一份要**明说没有**,不许顺手给一个别的。

    「这一份已经被挤掉了」是用户看得懂的答案;悄悄给一份别人的注入
    会被当成这一轮的来读,而它看起来完全正常。
    """
    from mast.prompts import capture as cap

    cap.record([], source="ic")
    gone = client.get("/api/admin/prompt-capture").json()["items"][0]["seq"]
    for _ in range(cap.MAX_SNAPSHOTS + 1):
        cap.record([], source="ic")

    r = client.get(f"/api/admin/prompt-capture/by-seq/{gone}")
    assert r.status_code == 404
    assert r.json()["found"] is False


def test_by_seq_is_not_swallowed_by_the_index_route(client: TestClient):
    """路由顺序:``/by-seq/{seq}`` 不能被 ``/{index}`` 吃掉。

    两条路由长得像,而「被吃掉」的症状是 422/404 —— 和「这一份没了」一模一样,
    于是整个功能会以「总是说没记录」的形式坏掉。
    """
    from mast.prompts import capture as cap

    cap.record([], source="ic")
    seq = client.get("/api/admin/prompt-capture").json()["items"][0]["seq"]
    r = client.get(f"/api/admin/prompt-capture/by-seq/{seq}")
    assert r.status_code == 200, r.text
    assert r.json()["found"] is True


def test_the_ring_holds_more_than_one_chat_turn(client: TestClient):
    """一轮带工具调用就是好几次模型请求 —— 环太浅时**上一轮必定已经没了**。

    这条钉的不是某个具体数字,是那个下限:8 条的环装不下一轮多工具的对话,
    聊天页那个折叠块会在几乎每条消息上说「没有留底」。
    """
    from mast.prompts import capture as cap

    assert cap.MAX_SNAPSHOTS >= 16, (
        f"环只有 {cap.MAX_SNAPSHOTS} 条:一轮多工具调用就能把上一轮挤光")


def test_capture_can_be_cleared(client: TestClient):
    from mast.prompts import capture as cap

    cap.record([], source="ic")
    r = client.delete("/api/admin/prompt-capture")
    assert r.status_code == 200 and r.json()["cleared"] == 1
    assert client.get("/api/admin/prompt-capture").json()["count"] == 0


def test_capture_kill_switch_stops_recording_and_is_reported(client: TestClient):
    from mast.prompts import capture as cap

    ring = cap.get_ring()
    try:
        ring.set_enabled(False)
        cap.record([], source="ic")
        body = client.get("/api/admin/prompt-capture").json()
        assert body["enabled"] is False
        assert body["count"] == 0        # nothing recorded while off
        # "off" must not read as "no model call has run yet"
        assert "已关闭" in body["note"]
    finally:
        ring.set_enabled(True)
    assert client.get("/api/admin/prompt-capture").json()["enabled"] is True
