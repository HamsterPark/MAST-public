"""笔记（进 MAST 记忆库，内部 agent 会召回）、向操作员发问、交接报告。"""
from __future__ import annotations

from _ext_world import H, wait_terminal


def test_a_note_lands_in_the_experiment_namespace_signed_by_the_agent(world):
    r = world.client.post("/notes", json={"title": "针尖状态", "content": "条纹 3 pm，可以扫原子",
                                          "kind": "insight", "tags": ["tip"]}, headers=H()).json()
    assert r["ok"] and r["namespace"] == f"experiment:{world.eid}"
    assert r["path"].startswith("ext/tester/")
    row = world.cog.store.read(r["namespace"], r["path"])
    assert row["author"] == "ext:tester" and row["kind"] == "insight"


def test_the_same_note_twice_is_one_row(world):
    body = {"title": "t", "content": "same"}
    a = world.client.post("/notes", json=body, headers=H()).json()
    b = world.client.post("/notes", json=body, headers=H()).json()
    assert a["path"] == b["path"]
    rows = world.cog.store.list(a["namespace"], limit=50)
    assert sum(1 for r in rows if r["path"] == a["path"]) == 1


def test_internal_agents_can_recall_what_the_external_agent_wrote(world):
    """双向：外部写的，内部 agent 的召回（MemoryRecallMiddleware 用的同一个 recall）找得到。"""
    world.client.post("/notes", json={"title": "偏压选择", "content": "负偏压下晶格衬度更好"},
                      headers=H())
    hits = world.cog.recall("晶格衬度", experiment_id=world.eid, k=5)
    assert any("晶格衬度" in h["content"] for h in hits)


def test_the_external_agent_can_search_notes_written_by_internal_agents(world):
    world.cog.remember(f"experiment:{world.eid}", "agent/ic/1", "Z 漂移 0.2 nm/h",
                       title="漂移", author="agent:instrument_control")
    r = world.client.get("/notes", params={"q": "漂移"}, headers=H()).json()
    assert any(n["author"] == "agent:instrument_control" for n in r["notes"])


def test_a_global_note_when_there_is_no_experiment_says_so(world):
    world.log.end_experiment()
    r = world.client.post("/notes", json={"title": "x", "content": "y"}, headers=H()).json()
    assert r["namespace"] == "global" and r["warnings"]


def test_asking_the_operator_and_reading_the_answer(world):
    from mast.wishlist import resolve_agent_request

    other = world.client.post("/requests", json={"message": "别人的问题"}, headers=H("other"))
    assert other.status_code == 200
    r = world.client.post("/requests", json={"message": "Au(111) 那张图在哪个目录？"},
                          headers=H()).json()
    rid = r["request"]["id"]
    assert r["request"]["agent_id"] == "ext:tester"
    mine = world.client.get("/requests", headers=H()).json()["requests"]
    assert [x["id"] for x in mine] == [rid]
    resolve_agent_request(rid, "done", note="在这里", path="D:/data/au111")
    got = world.client.get(f"/requests/{rid}", headers=H()).json()["request"]
    assert got["status"] == "done" and got["path"] == "D:/data/au111"
    b = world.client.get("/briefing", params={"sections": "operator_requests"},
                         headers=H()).json()
    assert "D:/data/au111" in b["text"]


def test_request_status_filter_uses_the_names_an_agent_would_guess(world):
    """心愿单里「未答」叫 pending；客户端传 open 曾经永远拿到空列表。"""
    rid = world.client.post("/requests", json={"message": "q"}, headers=H()).json()["request"]["id"]
    for status in ("open", "pending", "OPEN"):
        rows = world.client.get("/requests", params={"status": status}, headers=H()).json()
        assert [x["id"] for x in rows["requests"]] == [rid], status
    assert world.client.get("/requests", params={"status": "done"},
                            headers=H()).json()["requests"] == []
    r = world.client.get("/requests", params={"status": "answered"}, headers=H())
    assert r.status_code == 422 and "open" in r.json()["available"]


def test_handover_saves_a_report_with_jobs_actions_and_notes(world):
    jid = world.client.post("/jobs", json={"skill": "ExtRead"}, headers=H()).json()["job_id"]
    wait_terminal(world.client, jid)
    world.client.post("/notes", json={"title": "结论", "content": "针尖可用"}, headers=H())
    r = world.client.post("/handover", json={"summary": "扫了一帧", "next_steps": "换区域"},
                          headers=H())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["jobs"] == 1 and body["actions"] >= 1 and body["notes"] >= 1
    from mast.documents.store import store

    entry = store().get(body["doc_id"])
    text = entry.read_text()
    assert "扫了一帧" in text and "换区域" in text and "ExtRead" in text and "针尖可用" in text
    assert entry.versions[-1].created_by == "ext:tester"


def test_handover_counts_only_this_actors_actions_not_a_name_sharing_prefix(world):
    """``ext:tester`` 是 ``ext:tester2`` 的前缀 —— 交接报告不许把别人的动作算成自己的。"""
    for actor in ("tester2", "tester"):
        jid = world.client.post("/jobs", json={"skill": "ExtRead"},
                                headers=H(actor)).json()["job_id"]
        assert wait_terminal(world.client, jid, actor=actor)["state"] == "succeeded"
    body = world.client.post("/handover", json={"summary": "s"}, headers=H()).json()
    assert body["actions"] == 1 and body["jobs"] == 1, body
