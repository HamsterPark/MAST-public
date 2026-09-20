"""浏览器发起的请求、写请求的 Content-Type、急停 / 取消的必填请求体、作业参数上限、journal 压实。

外部面只给程序化客户端（MCP server、脚本、curl）用。安全审计（2026-09-19）指出：``/estop``
的请求体曾是可选的 ⇒ 操作员浏览器里任意一个网页发一个不带请求体的 ``no-cors`` POST，
就能在无认证的回环部署上触发急停、挂闩、取消所有外部作业。
"""
from __future__ import annotations

import json

from _ext_world import RAN, H, wait_terminal
from mast.api.ext.jobs import JobManager

_EVIL = {"Origin": "http://evil.example", "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}
_LOOP = {"Origin": "http://127.0.0.1:7862", "Host": "127.0.0.1:7862", "Sec-Fetch-Site": "same-origin"}


def test_a_cross_site_page_cannot_press_the_emergency_stop(world):
    # no-cors fetch：不带请求体、不带 Content-Type
    r = world.client.post("/estop", headers=_EVIL)
    assert r.status_code in (403, 415), r.text
    # 看起来像 JSON 的 text/plain（简单请求，浏览器不做预检）
    r = world.client.post("/estop", content=b'{"reason": "x"}',
                          headers={**_EVIL, "Content-Type": "text/plain"})
    assert r.status_code in (403, 415), r.text
    # application/json 的跨站请求（浏览器会先预检；服务端这边同样拒绝）。Host 是回环地址、
    # 没有 Sec-Fetch-*：只剩 Origin 这一道判据能拦它。
    r = world.client.post("/estop", json={"reason": "x"},
                          headers={"Origin": "http://evil.example", "Host": "127.0.0.1:7862"})
    assert r.status_code == 403 and r.json()["error"] == "cross_origin_refused"
    assert world.rt.estops == [], "一个网页按下了急停"
    assert r.headers["x-mast-ext-version"] == "1"


def test_a_cross_site_page_cannot_submit_a_job(world):
    r = world.client.post("/jobs", json={"skill": "ExtWrite"}, headers=_EVIL)
    assert r.status_code == 403
    r = world.client.post("/jobs", content=b'{"skill": "ExtWrite"}', headers=_EVIL)
    assert r.status_code in (403, 415)
    assert RAN == []
    # 跨站的 GET（例如 <img src=…>）不带 Origin，只有 Sec-Fetch-Site 说出它的来历
    r = world.client.get("/briefing", headers={"Host": "127.0.0.1:7862",
                                               "Sec-Fetch-Site": "cross-site",
                                               "Sec-Fetch-Mode": "no-cors"})
    assert r.status_code == 403 and r.json()["error"] == "cross_origin_refused"


def test_a_write_without_a_json_body_is_refused_before_it_runs(world):
    """不依赖某个版本 FastAPI 的 strict_content_type 默认值：旧版会把没有 Content-Type 的
    请求体当 JSON 解析 —— 那正是绕开 CORS 预检的办法。"""
    r = world.client.post("/estop")
    assert r.status_code == 415 and r.json()["error"] == "unsupported_media_type"
    r = world.client.post("/jobs", content=b'{"skill": "ExtRead"}')   # Blob 式：无 Content-Type
    assert r.status_code == 415
    assert RAN == [] and world.rt.estops == []


def test_estop_and_cancel_need_a_body(world):
    r = world.client.post("/estop", content=b"", headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert world.rt.estops == []
    jid = world.client.post("/jobs", json={"skill": "ExtSlow"}, headers=H()).json()["job_id"]
    r = world.client.post(f"/jobs/{jid}/cancel", content=b"",
                          headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    r = world.client.post(f"/jobs/{jid}/cancel", json={}, headers=H())
    assert r.status_code == 200 and r.json()["cancel_requested"] is True
    assert wait_terminal(world.client, jid)["state"] == "cancelled"


def test_the_dns_rebinding_shape_is_refused(world):
    """攻击者的域名重新解析到 127.0.0.1：浏览器眼里是同源，Host 是那个陌生主机名。"""
    rebound = {"Host": "attacker.example:7862", "Sec-Fetch-Site": "same-origin",
               "Sec-Fetch-Mode": "cors"}
    r = world.client.get("/data/files", headers=rebound)
    assert r.status_code == 403
    r = world.client.post("/notes", json={"title": "t", "content": "c"},
                          headers={**rebound, "Origin": "http://attacker.example:7862"})
    assert r.status_code == 403


def test_a_same_origin_browser_on_loopback_is_allowed(world):
    """交互文档（/docs）在 127.0.0.1 上「Try it out」—— 同源、回环，放行。"""
    assert world.client.get("/health", headers=_LOOP).status_code == 200
    r = world.client.post("/notes", json={"title": "t", "content": "c"}, headers={**_LOOP, **H()})
    assert r.status_code == 200, r.text


def test_programmatic_clients_are_unaffected_by_the_host_name(world):
    """MCP server / curl 不带 Origin 与 Sec-Fetch-*；经局域网主机名访问照常。"""
    assert world.client.get("/health", headers={"Host": "rig.lab:7862"}).status_code == 200


def test_oversized_params_are_refused(world):
    r = world.client.post("/jobs", json={"skill": "ExtRead", "params": {"blob": "x" * 70_000}},
                          headers=H())
    assert r.status_code == 422, r.text
    assert RAN == []


def test_the_journal_is_compacted_while_the_process_runs(world, monkeypatch):
    import mast.api.ext.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "JOURNAL_COMPACT_BYTES", 4000)
    ids = []
    for _ in range(8):
        jid = world.client.post("/jobs", json={"skill": "ExtRead"}, headers=H()).json()["job_id"]
        wait_terminal(world.client, jid)
        assert world.jm.get(jid).done.wait(5)
        ids.append(jid)
    lines = [ln for ln in world.jm.journal_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) < 8 * 3, f"没压实：{len(lines)} 行（每个作业三次状态变化）"
    # 压实后重载：一个不少，都是终态
    again = JobManager(world.jm.journal_path.parent)
    assert all(again.get(j) is not None and again.get(j).terminal for j in ids)
    assert {json.loads(ln)["job"]["job_id"] for ln in lines} >= set(ids[-1:])
