"""简报 / 状态 / 作用域 / 急停 / 铁律 / 健康。

简报的三条纪律（见 ``mast.api.ext.overview`` 模块说明）：逐段独立降级、零硬件 I/O、
有界。急停的一条纪律：闩上的原因如实写成触发它的外部 agent —— 不替别人编原因。
"""
from __future__ import annotations

import threading

import pytest

from _ext_world import H, wait_terminal


def test_the_briefing_carries_every_section_and_a_markdown_digest(world):
    b = world.client.get("/briefing", headers=H()).json()
    from mast.api.ext.overview import SECTIONS

    assert set(b["sections"]) == set(SECTIONS)
    for name in ("status", "scope", "resume", "tip", "instrument", "prefs", "safety",
                 "recent_actions", "recording", "jobs", "operator_requests"):
        assert b["sections"][name]["ok"] is True, (name, b["sections"][name])
    assert "## scope" in b["text"] and "外部网关测试" in b["text"]
    assert b["actor"] == "ext:tester"


def test_one_broken_section_does_not_take_the_briefing_down(world, monkeypatch):
    import mast.api.ext.overview as ov

    def _boom(request, caller):
        raise RuntimeError("tip store exploded")

    monkeypatch.setitem(ov.SECTIONS, "tip", _boom)
    b = world.client.get("/briefing", headers=H()).json()
    assert b["sections"]["tip"]["ok"] is False and "exploded" in b["sections"]["tip"]["error"]
    assert {"section": "tip", "reason": "RuntimeError: tip store exploded"} in b["degraded"]
    assert b["sections"]["scope"]["ok"] is True, "一段坏了拖垮了别的段"
    assert "（读不到：" in b["text"], "读不到必须在文字里说出来，不能静默消失"


def test_the_briefing_does_no_hardware_io(world):
    world.client.get("/briefing", headers=H())
    world.client.get("/status", headers=H())
    assert world.state.refreshed == 0, "简报调了 state.refresh()（真硬件 I/O）"
    assert world.pool.calls == [], f"简报发了 Nanonis 命令：{world.pool.calls}"
    assert world.rt.session_dir_asked == 0, "简报去问了 Nanonis 会话目录（一条 TCP 命令）"


def test_unknown_briefing_section_is_a_422_that_lists_the_real_ones(world):
    r = world.client.get("/briefing", params={"sections": "tip,nope"}, headers=H())
    assert r.status_code == 422 and "tip" in r.json()["available"]


def test_status_reports_mode_as_unknown_when_nobody_bound_it(world):
    s = world.client.get("/status", headers=H()).json()
    assert s["mode"] == "unknown", "未绑定的运行模式被说成了一个具体的值"
    assert s["lock"] == {"held": False}
    assert s["connection"]["roles"]["main"] is True and s["connection"]["roles"]["data"] is False
    assert s["scope"]["experiment"]["name"] == "外部网关测试"


def test_status_says_why_the_abort_is_set(world):
    from mast.core.execution_context import mark_abort

    mark_abort(world.rt._orch_abort, "群聊 run 被停止")
    s = world.client.get("/status", headers=H()).json()
    assert s["abort"]["set"] is True
    b = world.client.get("/briefing", params={"sections": "status"}, headers=H()).json()
    assert "中止事件已置位" in b["text"]
    world.rt._orch_abort.clear()


def test_scope_switch_create_and_busy_refusal(world):
    r = world.client.post("/scope", json={"experiment": {"name": "第二个实验", "goal": "g"},
                                          "sample": {"name": "样品B"}}, headers=H()).json()
    assert r["experiment"]["name"] == "第二个实验" and r["sample"]["name"] == "样品B"
    r = world.client.post("/scope", json={"experiment": {"id": world.eid}}, headers=H()).json()
    assert r["experiment"]["id"] == world.eid
    assert r["sample"]["id"] == world.sid, "切回实验时应恢复它上次用过的样品"

    from mast.core.instrument_lock import instrument_lock

    held, release = threading.Event(), threading.Event()

    def _holder():
        with instrument_lock().hold(owner="群聊任务", skill="LongScan"):
            held.set()
            release.wait(30)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        r = world.client.post("/scope", json={"sample": {"name": "样品C"}}, headers=H())
        assert r.status_code == 409 and r.json()["error"] == "instrument_busy"
        r = world.client.post("/scope", json={"sample": {"name": "样品C"}, "force": True},
                              headers=H())
        assert r.status_code == 200 and r.json()["warnings"]
    finally:
        release.set()
        t.join(10)


def test_scope_errors_are_specific(world):
    r = world.client.post("/scope", json={"experiment": {"id": "no-such"}}, headers=H())
    assert r.status_code == 404 and r.json()["error"] == "unknown_experiment"
    r = world.client.post("/scope", json={}, headers=H())
    assert r.status_code == 422


def test_estop_records_who_pressed_it_and_cancels_external_jobs(world):
    jid = world.client.post("/jobs", json={"skill": "ExtSlow"}, headers=H()).json()["job_id"]
    r = world.client.post("/estop", json={"reason": "电流失控"}, headers=H()).json()
    assert r["ok"] and r["retracted"] is True
    assert "外部 agent ext:tester 触发急停" in r["why"] and "电流失控" in r["why"]
    assert world.rt.estops == [r["why"]], "闩上的原因没有如实写成外部 agent"
    assert jid in r["cancelled_jobs"]
    v = wait_terminal(world.client, jid)
    assert v["state"] in ("cancelled", "failed") and v["terminal"]
    st = world.client.get("/status", headers=H()).json()
    assert st["abort"]["emergency"] is True


def test_guide_and_health(world):
    g = world.client.get("/guide", params={"lang": "zh"}).json()
    assert g["lang"] == "zh" and len(g["rules"]) >= 10
    assert {r["id"] for r in g["rules"]} >= {"briefing-first", "jobs-not-timeouts",
                                             "busy-means-wait", "operating-mode"}
    h = world.client.get("/health").json()
    assert h["ok"] and h["api_version"] == "1" and h["missing"] == []


def test_unknown_paths_are_json_404_not_the_spa_shell(world):
    r = world.client.get("/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
    assert r.json()["error"] == "not_found"
    assert r.headers["x-mast-ext-version"] == "1"


def test_every_kind_of_error_carries_the_version_header(world, monkeypatch):
    """客户端靠版本头区分「外部面里的 404」与「这个 MAST 没挂外部面」—— 业务错误、
    请求校验错误、未处理异常的 500 都要带。"""
    from fastapi.testclient import TestClient

    import mast.api.ext.overview as ov

    r = world.client.get("/jobs/j_000000000000", headers=H())
    assert r.status_code == 404 and r.json()["error"] == "unknown_job"
    assert r.headers["x-mast-ext-version"] == "1"
    r = world.client.post("/jobs", json={"params": {}}, headers=H())
    assert r.status_code == 422 and r.headers["x-mast-ext-version"] == "1"

    def _boom(request):
        raise RuntimeError("status exploded")

    monkeypatch.setattr(ov, "_status", _boom)
    r = TestClient(world.app, raise_server_exceptions=False).get("/status", headers=H())
    assert r.status_code == 500 and r.json()["error"] == "internal"
    assert r.headers["x-mast-ext-version"] == "1"


@pytest.mark.parametrize("hdr, expect", [
    ({}, "ext:anonymous"),
    ({"X-MAST-Actor": "  Claude Code / Night  "}, "ext:claude-code-night"),
    ({"X-MAST-Actor": "../../etc"}, "ext:etc"),
])
def test_the_actor_header_is_only_ever_cleaned_never_a_reason_to_refuse(world, hdr, expect):
    b = world.client.get("/briefing", params={"sections": "recording"}, headers=hdr)
    assert b.status_code == 200
    assert b.json()["actor"] == expect
