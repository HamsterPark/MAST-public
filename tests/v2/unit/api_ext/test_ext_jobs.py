"""外部面的作业：生命周期、长轮询、取消、撞锁不排队、幂等、并发上限、崩溃、重启、SAFE。

每一条都对应一次真实付过的代价（设计稿 docs/v2/design/external_agent_gateway.md）：
客户端超时后服务端还在跑、之后全撞仪器锁；重试打出第二发；进程重启后不知道哪一步
做了没有；SAFE 的契约从某个没人想到的入口漏出去。
"""
from __future__ import annotations

import threading
import time

from _ext_world import GATES, RAN, H, wait_terminal


def test_a_job_runs_and_lands_in_the_record_with_the_callers_name(world):
    r = world.client.post("/jobs", json={"skill": "ExtRead", "params": {"x_m": "5n"}},
                          headers=H())
    assert r.status_code == 202, r.text
    job = wait_terminal(world.client, r.json()["job_id"])
    assert job["state"] == "succeeded", job
    assert job["result"]["data"] == {"value": 42}
    assert job["params_used"]["x_m"] == 5e-9, "SI 字符串没有在门口还原"
    rec = job["recorded"]
    assert rec["v1"] is True and rec["v2_action_id"], rec
    v1 = world.st.recent_actions(world.eid)
    assert v1[0]["context"] == "ext:tester/s1" and v1[0]["approval_source"] == "llm"
    v2 = world.repos.actions.for_experiment(world.v2eid)
    assert v2[-1]["agent_id"] == "ext:tester" and v2[-1]["tool_call_id"] == job["job_id"]


def test_long_poll_returns_at_the_deadline_without_blocking_the_job(world):
    r = world.client.post("/jobs", json={"skill": "ExtSlow"}, headers=H())
    jid = r.json()["job_id"]
    t0 = time.monotonic()
    v = world.client.get(f"/jobs/{jid}", params={"wait_s": 1}, headers=H()).json()
    dt = time.monotonic() - t0
    assert v["state"] == "running" and not v["terminal"]
    assert 0.8 <= dt < 5, f"长轮询没按时返回：{dt:.2f}s"
    GATES["slow"].set()
    assert wait_terminal(world.client, jid)["state"] == "succeeded"


def test_cancel_stops_a_running_job_and_says_who_stopped_it(world):
    jid = world.client.post("/jobs", json={"skill": "ExtSlow"}, headers=H()).json()["job_id"]
    time.sleep(0.3)
    v = world.client.post(f"/jobs/{jid}/cancel", json={"reason": "换方案"}, headers=H()).json()
    assert v["cancel_requested"] is True
    v = wait_terminal(world.client, jid)
    assert v["state"] == "cancelled", v
    assert "外部 agent 取消" in v["result"]["error"] and "换方案" in v["result"]["error"], (
        "中止原因没传到技能那里 —— 被停的一方不知道是谁停的")


def test_the_process_wide_abort_also_stops_an_external_job(world):
    from mast.core.execution_context import mark_abort

    jid = world.client.post("/jobs", json={"skill": "ExtSlow"}, headers=H()).json()["job_id"]
    time.sleep(0.3)
    mark_abort(world.rt._orch_abort, "测试：进程级中止")
    v = wait_terminal(world.client, jid)
    assert v["state"] == "failed" and "进程级中止" in v["result"]["error"]
    assert v["abort"] and v["abort"]["set"] is True
    world.rt._orch_abort.clear()


def test_a_busy_instrument_is_refused_with_the_holder_not_queued(world):
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
        jid = world.client.post("/jobs", json={"skill": "ExtWrite"}, headers=H()).json()["job_id"]
        v = wait_terminal(world.client, jid, total_s=30)
    finally:
        release.set()
        t.join(10)
    assert v["state"] == "refused_busy", v
    assert v["busy_holder"]["owner"] == "群聊任务"
    assert ("ExtWrite", {}) not in RAN


def test_request_id_makes_a_resend_idempotent_and_a_different_body_a_conflict(world):
    body = {"skill": "ExtRead", "params": {}, "request_id": "r-1"}
    a = world.client.post("/jobs", json=body, headers=H())
    b = world.client.post("/jobs", json=body, headers=H())
    assert a.status_code == 202 and b.status_code == 200
    assert a.json()["job_id"] == b.json()["job_id"] and b.json()["idempotent_replay"] is True
    wait_terminal(world.client, a.json()["job_id"])
    assert [n for n, _ in RAN].count("ExtRead") == 1, "重发打出了第二发"
    c = world.client.post("/jobs", json={**body, "params": {"x_m": 1e-9}}, headers=H())
    assert c.status_code == 409 and c.json()["error"] == "request_id_conflict"
    # 另一个调用方用同一个 id 不冲突 —— 幂等键是 (actor, request_id)
    d = world.client.post("/jobs", json=body, headers=H("other"))
    assert d.status_code == 202


def test_the_concurrency_cap_answers_429(world):
    world.jm.max_concurrent = 2
    ids = [world.client.post("/jobs", json={"skill": "ExtSlow"}, headers=H()).json()["job_id"]
           for _ in range(2)]
    r = world.client.post("/jobs", json={"skill": "ExtSlow"}, headers=H())
    assert r.status_code == 429 and r.json()["error"] == "too_many_jobs"
    # 作业堆满的时刻往往正是最需要停的时刻：停止类补救技能不受上限约束
    r = world.client.post("/jobs", json={"skill": "StopScan"}, headers=H())
    assert r.status_code == 202, r.text
    assert wait_terminal(world.client, r.json()["job_id"])["state"] == "succeeded"
    assert ("StopScan", {}) in RAN
    GATES["slow"].set()
    for jid in ids:
        wait_terminal(world.client, jid)


def test_a_skill_that_raises_ends_the_job_it_does_not_hang_it(world):
    jid = world.client.post("/jobs", json={"skill": "ExtBoom"}, headers=H()).json()["job_id"]
    v = wait_terminal(world.client, jid)
    assert v["terminal"] and v["state"] == "failed" and "probe exploded" in v["result"]["error"]


def test_an_unknown_skill_is_404_with_suggestions(world):
    r = world.client.post("/jobs", json={"skill": "ExtRea"}, headers=H())
    assert r.status_code == 404 and r.json()["error"] == "unknown_skill"
    assert "ExtRead" in r.json()["did_you_mean"]


def test_a_disabled_skill_is_refused_at_the_door(world, monkeypatch):
    import mast.skills.tool_face as tf

    monkeypatch.setattr(tf, "compute", lambda all_names=None: tf.FaceSkip(
        hardware=frozenset({"ExtWrite"})))
    r = world.client.post("/jobs", json={"skill": "ExtWrite"}, headers=H())
    assert r.status_code == 422 and r.json()["error"] == "skill_disabled"
    assert ("ExtWrite", {}) not in RAN


def test_safe_mode_refuses_a_pulse_submitted_as_a_job(world):
    """SAFE 的契约（「针尖视为良好，不修针」）不许从外部入口漏出去。"""
    from mast.core.operating_mode import bind_mode_source

    bind_mode_source(lambda: "safe")
    jid = world.client.post("/jobs", json={"skill": "ExtPulse", "params": {"pulse_v": 3.0}},
                            headers=H()).json()["job_id"]
    v = wait_terminal(world.client, jid)
    assert v["state"] == "failed", v
    assert "safe_mode" in v["result"]["error"]
    assert not any(n == "ExtPulse" for n, _ in RAN), "SAFE 下脉冲探针被执行了"


def test_list_jobs_is_scoped_to_the_caller_unless_asked(world):
    world.client.post("/jobs", json={"skill": "ExtRead"}, headers=H("alice"))
    world.client.post("/jobs", json={"skill": "ExtRead"}, headers=H("bob"))
    mine = world.client.get("/jobs", headers=H("alice")).json()["jobs"]
    assert {j["actor"] for j in mine} == {"alice"}
    both = world.client.get("/jobs", params={"all": True}, headers=H("alice")).json()["jobs"]
    assert {j["actor"] for j in both} >= {"alice", "bob"}


def test_a_restart_marks_unfinished_jobs_lost_and_never_replays_them(world):
    from mast.api.ext.jobs import JobManager

    jid = world.client.post("/jobs", json={"skill": "ExtSlow", "request_id": "r-restart"},
                            headers=H()).json()["job_id"]
    time.sleep(0.3)
    # 「进程重启」：新管理器读同一份 journal，旧线程还在跑（真重启时它已经不存在了）
    fresh = JobManager(world.tmp / "journal")
    lost = fresh.get(jid)
    assert lost is not None and lost.state == "lost_on_restart"
    n_before = len(RAN)
    job, replay = fresh.submit(world.ctx, _caller(), "ExtSlow", {}, request_id="r-restart")
    assert replay is True and job.job_id == jid and job.state == "lost_on_restart", (
        "重启后用同一个 request_id 重发，拿到的必须是那条 lost_on_restart，而不是再跑一次")
    assert len(RAN) == n_before
    GATES["slow"].set()
    wait_terminal(world.client, jid)


def test_the_shutdown_hook_is_registered_and_stops_jobs_before_the_pool_closes(world):
    names = [n for n, _ in world.rt._shutdown_hooks]
    assert "ext-gateway jobs" in names
    jid = world.client.post("/jobs", json={"skill": "ExtSlow"}, headers=H()).json()["job_id"]
    time.sleep(0.3)
    world.jm.shutdown(timeout_s=5)
    v = world.jm.get(jid)
    assert v.state == "cancelled", v.state
    r = world.client.post("/jobs", json={"skill": "ExtRead"}, headers=H())
    assert r.status_code == 503 and r.json()["error"] == "shutting_down"


def _caller():
    from mast.api.ext.common import Caller

    return Caller(actor="tester", session="s1")
