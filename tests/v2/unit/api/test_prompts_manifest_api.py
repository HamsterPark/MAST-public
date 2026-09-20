"""``/api/prompts/*`` —— 面向用户的**只读**注入视图。

它不在 PIN 门后面，所以「只读」不是风格约定而是安全前提：一旦这里能写，
高级管理那道 PIN 就形同虚设。``test_the_read_side_has_no_write_routes`` 钉着它。

Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/api/test_prompts_manifest_api.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_REPO = Path(__file__).resolve().parents[4]
_MASTV2 = str(_REPO / "MASTv2")
if sys.path and sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)

from mast.api.routes import prompts as prompts_route  # noqa: E402
from mast.prompts import builds, capture, tool_surface  # noqa: E402


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(prompts_route.router, prefix="/api")
    app.state.ctx = SimpleNamespace()
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean():
    capture.get_ring().reset_for_tests()
    yield
    capture.get_ring().reset_for_tests()


# ── 矩阵 ────────────────────────────────────────────────────────────────

def test_matrix_lists_every_agent_and_every_block(client):
    r = client.get("/api/prompts/manifest")
    assert r.status_code == 200
    d = r.json()
    ids = {b["id"] for b in d["blocks"]}
    assert len(d["agents"]) >= 9
    assert {"agent.instrument_control.system", "mw.tip_context",
            "mw.live_state"} <= ids
    for b in d["blocks"]:
        assert set(d["cells"][b["id"]]) == {a["id"] for a in d["agents"]}


def test_matrix_shows_targeting_not_just_a_wall_of_yes(client):
    """这张表存在的**唯一目的**是回答「有没有针对性」。

    如果每一格都是 true，它就没有回答任何问题 —— 而 2026-08-24 之前登记表
    渲染出来差不多就是那样（四条定向块被标成了「全局」）。
    """
    d = client.get("/api/prompts/manifest").json()
    cells = d["cells"]
    exclusive = [b["id"] for b in d["blocks"] if b["exclusive"]]
    assert len(exclusive) >= 8, f"只有 {len(exclusive)} 条专属块，针对性可疑"

    tip = cells["mw.tip_context"]
    assert tip["instrument_control"] and tip["data_processing"]
    assert not tip["paper_review"] and not tip["paper_writing"], (
        "针尖块又发给论文 agent 了 —— 它们不碰针尖也不读谱")

    live = cells["mw.live_state"]
    assert live["instrument_control"]
    assert not any(v for k, v in live.items() if k != "instrument_control")


def test_matrix_says_when_and_where_each_block_lands(client):
    d = client.get("/api/prompts/manifest").json()
    by_id = {b["id"]: b for b in d["blocks"]}
    assert by_id["mw.live_state"]["position"] == "last_human"
    assert by_id["mw.memory_recall"]["position"] == "last_human"
    assert by_id["mw.tip_context"]["position"] == "system"
    assert by_id["mw.experiment_prefs"]["when"] == "when_set"
    assert by_id["mw.mode_belief.safe"]["when"] == "on_mode"
    assert by_id["agent.instrument_control.system"]["when"] == "build_time"


def test_non_renderable_blocks_carry_a_reason_and_no_sample_text(client):
    """诚实性铁律：拿不到就返回空 + 写明原因，**绝不填示例文本**。"""
    d = client.get("/api/prompts/manifest").json()
    for b in d["blocks"]:
        if b["availability"] in ("needs_hardware", "needs_request"):
            assert b["preview"] == "", f"{b['id']} 填了示例文本"
            assert b["unavailable_reason"], f"{b['id']} 空着却没说为什么"


# ── 单个 agent ──────────────────────────────────────────────────────────

def test_agent_manifest_puts_its_own_system_prompt_first(client):
    d = client.get("/api/prompts/manifest/paper_review").json()
    assert d["blocks"][0]["id"] == "agent.paper_review.system"
    assert d["order_source"] in ("build", "declared")


def test_agent_manifest_404s_on_an_unknown_agent(client):
    r = client.get("/api/prompts/manifest/no_such_agent")
    assert r.status_code == 404
    assert "没有名为" in r.json()["tool_surface_note"]


def test_tool_surface_is_null_with_a_reason_before_any_build(client):
    """没量过就说没量过 —— 这里不做静态估算，估出来的和实测长得一样。"""
    # 两个注册表是**独立**的：`manifest_for` 在没有 BuildRecord 时会退回
    # `tool_surface.get(agent)`（那是为了「进程建过图但记录被清了」这种情况）。
    # 只清一个的话，这条测试单跑绿、跟别的一起跑红 —— 而跨测试污染出来的是
    # 一个**看起来完全正常**的数字。
    builds.reset()
    tool_surface.reset()
    d = client.get("/api/prompts/manifest/literature").json()
    assert d["tool_surface"] is None
    assert "还没有建过" in d["tool_surface_note"]


# ── 快照 ────────────────────────────────────────────────────────────────

def test_latest_capture_distinguishes_the_four_empty_reasons(client):
    ring = capture.get_ring()

    r = client.get("/api/prompts/capture/latest/paper_review")
    assert r.status_code == 404
    assert r.json()["reason_code"] == "no_calls"

    capture.record([], source="instrument_control", metadata={"langgraph_node": "model"})
    r = client.get("/api/prompts/capture/latest/paper_review")
    assert r.json()["reason_code"] == "no_calls_for_agent"

    ring.set_enabled(False)
    try:
        r = client.get("/api/prompts/capture/latest/paper_review")
        assert r.json()["reason_code"] == "disabled"
    finally:
        ring.set_enabled(True)

    r = client.get("/api/prompts/capture/latest/nope")
    assert r.status_code == 404
    assert r.json()["reason_code"] == "unknown_agent"


def test_latest_capture_returns_blocks_and_provider_tokens(client):
    from langchain_core.messages import HumanMessage, SystemMessage

    from mast.agents._shared import inject

    req = SimpleNamespace(system_message=SystemMessage(content="BASE"),
                          messages=[HumanMessage(content="hi")])
    req = inject.append_system_block(req, "mw.tip_context", "TIP BLOCK")
    capture.record([req.system_message], source="data_processing",
                   metadata={"langgraph_node": "model"},
                   invocation_params={"tools": [{"function": {"name": "T", "x": "y" * 300}}]})

    d = client.get("/api/prompts/capture/latest/data_processing").json()
    assert d["found"] is True
    assert d["blocks_source"] == "ledger"
    ids = [b["id"] for b in d["messages"][0]["blocks"]]
    assert ids == ["system.base", "mw.tip_context"]
    # 工具面**不在 messages 里**，但必须报出来
    assert d["tool_count"] == 1 and d["tools_chars"] > 300
    # provider 没报 token → null，不折算
    assert d["input_tokens"] is None and d["tokens_source"] == "unavailable"


def test_latest_capture_ignores_middleware_sub_llm_calls(client):
    capture.record([], source="data_processing", metadata={"langgraph_node": "model"})
    capture.record([], source="data_processing",
                   metadata={"langgraph_node": "data_processing.before_model"})
    d = client.get("/api/prompts/capture/latest/data_processing").json()
    assert d["node"] == "model", "返回了一条中间件内部的摘要请求"


# ── 只读 ────────────────────────────────────────────────────────────────

def test_the_read_side_has_no_write_routes():
    """这一组不在 PIN 门后面 —— 能写的话 PIN 就形同虚设。"""
    methods = set()
    for route in prompts_route.router.routes:
        methods |= set(getattr(route, "methods", ()) or ())
    assert methods <= {"GET", "HEAD", "OPTIONS"}, (
        f"/api/prompts 出现了写方法：{methods - {'GET', 'HEAD', 'OPTIONS'}}")


def test_the_module_never_500s_when_the_backend_is_missing(client, monkeypatch):
    """单独起的 API 进程要能降级，不能启动失败也不能 500。"""
    import builtins

    real_import = builtins.__import__

    def boom(name, *a, **kw):
        if name.startswith("mast.prompts"):
            raise ImportError("simulated")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    r = client.get("/api/prompts/manifest")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
