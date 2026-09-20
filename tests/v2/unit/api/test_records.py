"""Domain D contract tests — experiment detail, records drill-down, feedback.

Per the house test rule, the router under test is NOT yet included in
mast.api.app (integration wires that); we mount it on a throwaway FastAPI
app with a fresh AppContext. We assert:

  * every endpoint returns its defined status + the schema-shaped body;
  * standalone (no live core wired) paths degrade — empty, never broken, never 500;
  * write endpoints return a typed degraded result with no core;
  * wiring a REAL ExperimentStorage (temp SQLite) drives the v1-backed endpoints
    (detail / create / end / add-sample / feedback) through their live paths.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.records import router


# ── throwaway apps ─────────────────────────────────────────────────────


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _wired_client(tmp_path) -> tuple[TestClient, object]:
    """A client backed by a REAL v1 ExperimentStorage over a temp SQLite db."""
    from mast.logging.storage import ExperimentStorage

    storage = ExperimentStorage(str(tmp_path / "rec.db"))
    ctx = AppContext()
    ctx.wire(experiment_storage=storage)
    return _client(ctx), storage


# ── standalone (degraded) paths ────────────────────────────────────────


def test_experiment_detail_degrades_unwired() -> None:
    r = _client().get("/api/experiments/abc123")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "abc123"
    assert body["degraded"] is True
    assert body["found"] is False
    assert body["samples"] == [] and body["actions"] == []
    assert body["map_markers"] == [] and body["feedback"] == []


def test_create_experiment_degrades_unwired() -> None:
    r = _client().post("/api/experiments", json={"name": "exp", "goal": "g"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["id"] is None


def test_end_experiment_degrades_unwired() -> None:
    r = _client().post("/api/experiments/abc/end", json={"status": "aborted"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["id"] == "abc"


def test_create_sample_degrades_unwired() -> None:
    r = _client().post("/api/experiments/abc/samples", json={"name": "s1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["experiment_id"] == "abc"


def test_campaigns_contract() -> None:
    # The v2 records adapter (gui.records_api) reaches the on-disk ExperimentStoreV2
    # independent of ctx. On a clean/standalone machine the store is absent/empty
    # → degraded empty page; on a dev box with a real store it returns rows. Either
    # way the contract holds and the page echoes the request.
    r = _client().get("/api/records/campaigns")
    assert r.status_code == 200
    body = r.json()
    assert body["page"] == 1 and body["page_size"] == 50
    assert isinstance(body["campaigns"], list)
    if body["degraded"]:
        assert body["campaigns"] == [] and body["count"] == 0
    else:
        # populated rows must be schema-shaped
        assert body["count"] >= len(body["campaigns"])
        for c in body["campaigns"]:
            assert isinstance(c["id"], str) and c["id"]
            assert set(c["stats"]) == {"experiments", "actions", "observations", "scans"}


def test_campaigns_echoes_pagination_params() -> None:
    r = _client().get(
        "/api/records/campaigns",
        params={"status": "active", "from": "2026-01-01", "agent": "ic", "page": 3, "page_size": 10},
    )
    assert r.status_code == 200
    body = r.json()
    # filters are accepted; the page window is always echoed back
    assert body["page"] == 3 and body["page_size"] == 10
    # filtered to a likely-empty intersection → empty window regardless of store
    assert body["campaigns"] == []


def test_action_detail_unknown_id_is_empty_not_broken() -> None:
    # An id that cannot exist: degraded (no store) OR found=False (store wired but
    # id absent) — never a 500, never a populated body.
    r = _client().get("/api/records/actions/__no_such_action__")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "__no_such_action__"
    assert body["found"] is False
    assert body["params"] == {}
    assert body["degraded"] in (True, False)


def test_feedback_degrades_unwired() -> None:
    r = _client().post("/api/feedback", json={"rating": "up", "experiment_id": "e1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["id"] is None


# ── validation ─────────────────────────────────────────────────────────


def test_create_experiment_requires_name() -> None:
    r = _client().post("/api/experiments", json={"goal": "no name"})
    assert r.status_code == 422  # name is required by the schema


def test_campaigns_rejects_bad_page() -> None:
    r = _client().get("/api/records/campaigns", params={"page": 0})
    assert r.status_code == 422  # ge=1


# ── wired (live v1 storage) paths ──────────────────────────────────────


def test_create_then_detail_roundtrip(tmp_path) -> None:
    client, storage = _wired_client(tmp_path)

    # create
    r = client.post("/api/experiments", json={"name": "Au111 run", "goal": "map terraces"})
    assert r.status_code == 200
    created = r.json()
    assert created["ok"] is True and created["degraded"] is False
    exp_id = created["id"]
    assert exp_id

    # detail — found, real metadata, empty children
    r = client.get(f"/api/experiments/{exp_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["found"] is True and detail["degraded"] is False
    assert detail["name"] == "Au111 run"
    assert detail["goal_text"] == "map terraces"
    assert detail["status"] == "running"
    assert detail["samples"] == [] and detail["actions"] == []


def test_detail_not_found_is_empty_not_broken(tmp_path) -> None:
    client, _ = _wired_client(tmp_path)
    r = client.get("/api/experiments/does-not-exist")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is False and body["degraded"] is False
    assert body["samples"] == []


def test_add_sample_and_detail(tmp_path) -> None:
    client, storage = _wired_client(tmp_path)
    exp_id = storage.create_experiment("exp", "goal")

    r = client.post(
        f"/api/experiments/{exp_id}/samples",
        json={"name": "sample-A", "description": "fresh tip", "sample_type": "clean_metal"},
    )
    assert r.status_code == 200
    sres = r.json()
    assert sres["ok"] is True and sres["degraded"] is False
    assert sres["experiment_id"] == exp_id and sres["id"]

    r = client.get(f"/api/experiments/{exp_id}")
    detail = r.json()
    assert len(detail["samples"]) == 1
    s = detail["samples"][0]
    assert s["name"] == "sample-A"
    assert s["sample_type"] == "clean_metal"
    assert s["status"] == "active"
    assert s["action_count"] == 0


def test_end_experiment_wired(tmp_path) -> None:
    client, storage = _wired_client(tmp_path)
    exp_id = storage.create_experiment("exp", "goal")

    r = client.post(f"/api/experiments/{exp_id}/end", json={"status": "completed"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["id"] == exp_id and body["status"] == "completed"

    # storage reflects the end
    exp = storage.get_experiment(exp_id)
    assert exp["status"] == "completed"
    assert exp["end_time"]


def test_feedback_wired_then_surfaced_in_detail(tmp_path) -> None:
    client, storage = _wired_client(tmp_path)
    exp_id = storage.create_experiment("exp", "goal")

    # rating row
    r = client.post(
        "/api/feedback",
        json={"rating": "thumbs_up", "experiment_id": exp_id, "agent": "ic"},
    )
    assert r.status_code == 200
    fb = r.json()
    assert fb["ok"] is True and fb["degraded"] is False
    assert fb["kind"] == "rating" and isinstance(fb["id"], int)

    # comment row
    r = client.post(
        "/api/feedback",
        json={"comment": "tip looked unstable", "experiment_id": exp_id},
    )
    assert r.json()["kind"] == "comment"

    # both surface in the experiment detail's feedback block
    detail = client.get(f"/api/experiments/{exp_id}").json()
    kinds = sorted(f["kind"] for f in detail["feedback"])
    assert kinds == ["comment", "rating"]
    rating_row = next(f for f in detail["feedback"] if f["kind"] == "rating")
    assert rating_row["rating"] == "thumbs_up"
    assert rating_row["agent"] == "ic"


def test_feedback_listing_keeps_the_page_it_came_from(tmp_path) -> None:
    """"对话反馈知道用户是在哪个界面反馈的吗？" 

    The widget sends ``meta.page`` and the DB stores it — but the GLOBAL listing
    built its rows without ``meta=``, so the one endpoint that shows ALL feedback
    was the one that could not say where any of it came from. (In the 2026-07-10
    export, every one of the 108 rows carried an empty page for exactly this kind
    of reason — the trail back to the screen was simply not kept.)
    """
    client, storage = _wired_client(tmp_path)
    exp_id = storage.create_experiment("exp", "goal")

    client.post("/api/feedback", json={
        "comment": "扫描地图不能放大缩小", "experiment_id": exp_id,
        "agent": "page_records", "meta": {"page": "/records"},
    })

    listing = client.get("/api/feedback").json()
    row = next(f for f in listing["items"] if f["comment"].startswith("扫描地图"))
    assert row["meta"].get("page") == "/records", (
        f"the listing dropped the page the operator was on: {row}")

    # and the per-experiment view agrees — one truth, two surfaces
    detail = client.get(f"/api/experiments/{exp_id}").json()
    drow = next(f for f in detail["feedback"] if f["comment"].startswith("扫描地图"))
    assert drow["meta"].get("page") == "/records"


def _seed_v2_store(db_path: str) -> dict[str, str]:
    """Seed a real ExperimentStoreV2 with TWO experiments so one becomes the
    Records "focus" (most actions) and the other does not. Returns the ids of a
    focus-experiment action and a non-focus-experiment action."""
    from mast.logging.v2.repos import build_repos
    from mast.logging.v2.storage import ExperimentStoreV2

    store = ExperimentStoreV2(db_path)
    repos = build_repos(store)
    cid = repos.campaigns.create(title="camp", hypothesis="h")
    sid = repos.samples.create(label="s", material="Au(111)")

    # Focus experiment: 3 actions → it wins _pick_focus (most actions).
    eid_focus = repos.experiments.start(
        campaign_id=cid, sample_id=sid, title="focus", exp_type="scan"
    )
    focus_action = ""
    for i in range(3):
        focus_action = repos.actions.begin(
            experiment_id=eid_focus, agent_id="ic", action_type=f"focus_{i}"
        )

    # Other experiment: a single action — NOT in the focus payload's subset.
    eid_other = repos.experiments.start(
        campaign_id=cid, sample_id=sid, title="other", exp_type="scan"
    )
    other_action = repos.actions.begin(
        experiment_id=eid_other,
        agent_id="dp",
        action_type="line_profile",
        params={"k": "v"},
    )
    repos.actions.succeed(other_action, duration_ms=42)
    return {"focus_action": focus_action, "other_action": other_action}


def test_action_detail_resolves_across_experiments(tmp_path, monkeypatch) -> None:
    # open_store() reads $MAST_DATA_DIR/experiments/<db>; point it at a temp dir
    # so build_records_payload() + the cross-experiment fallback see our seed.
    from mast.logging.v2.storage import DEFAULT_DB_NAME

    data_dir = tmp_path / "data"
    db_path = data_dir / "experiments" / DEFAULT_DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ids = _seed_v2_store(str(db_path))
    monkeypatch.setenv("MAST_DATA_DIR", str(data_dir))

    client = _client()

    # The action from the NON-focus experiment is absent from the focus payload's
    # actions subset; before the fix this 404'd as "未找到动作". It must now resolve
    # via the direct store lookup.
    r = client.get(f"/api/records/actions/{ids['other_action']}")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["degraded"] is False
    assert body["id"] == ids["other_action"]
    assert body["action_type"] == "line_profile"
    assert body["agent_id"] == "dp"
    assert body["status"] == "succeeded"
    assert body["duration_ms"] == 42
    assert body["params"] == {"k": "v"}

    # The focus-experiment action still resolves via the focus payload path.
    r = client.get(f"/api/records/actions/{ids['focus_action']}")
    assert r.status_code == 200
    assert r.json()["found"] is True

    # A genuinely unknown id is still empty-not-broken (store reachable → found=False).
    r = client.get("/api/records/actions/__no_such_action__")
    assert r.status_code == 200
    assert r.json()["found"] is False


def test_map_markers_surface_in_detail(tmp_path) -> None:
    client, storage = _wired_client(tmp_path)
    exp_id = storage.create_experiment("exp", "goal")
    storage.log_marker(
        kind="scan",
        x_m=1e-7,
        y_m=2e-7,
        w_m=5e-8,
        h_m=5e-8,
        label="frame-1",
        skill_name="StartScan",
        experiment_id=exp_id,
    )

    detail = client.get(f"/api/experiments/{exp_id}").json()
    assert len(detail["map_markers"]) == 1
    m = detail["map_markers"][0]
    assert m["kind"] == "scan"
    assert m["x_m"] == 1e-7 and m["w_m"] == 5e-8
    assert m["label"] == "frame-1"
    assert m["source"] == "skill"


# ── 反馈处理状态（请求：「反馈表没有 created_at 和已处理标记，新旧混在一起」）


def test_feedback_starts_unresolved_and_can_be_marked(tmp_path):
    c, _ = _wired_client(tmp_path)
    fid = c.post("/api/feedback", json={"comment": "扫描地图太小"}).json()["id"]

    before = [f for f in c.get("/api/feedback").json()["items"] if f["id"] == fid][0]
    assert before["resolved_at"] is None
    assert before["resolved_version"] == ""

    r = c.post(f"/api/feedback/{fid}/resolved",
               json={"resolved": True, "version": "v6.2.1"}).json()
    assert r["ok"] is True

    after = [f for f in c.get("/api/feedback").json()["items"] if f["id"] == fid][0]
    assert after["resolved_at"], "处理时刻没落下来"
    assert after["resolved_version"] == "v6.2.1"


def test_a_symptom_that_came_back_can_be_un_marked(tmp_path):
    """「又回来了」与「当初标错了」是两回事。

    用户要能说前者。有三条反馈就是修好、没上机目视、
    然后原样回来的 —— 如果标记只能单向，重报的那一条会被标记本身说成已处理。
    """
    c, _ = _wired_client(tmp_path)
    fid = c.post("/api/feedback", json={"comment": "浅色下看不清"}).json()["id"]
    c.post(f"/api/feedback/{fid}/resolved", json={"resolved": True, "version": "v6.1.2"})
    c.post(f"/api/feedback/{fid}/resolved", json={"resolved": False})
    row = [f for f in c.get("/api/feedback").json()["items"] if f["id"] == fid][0]
    assert row["resolved_at"] is None
    assert row["resolved_version"] == ""


def test_marking_a_row_that_does_not_exist_reports_failure(tmp_path):
    """「标过了」和「那一行不在」必须分得开。

    返回一个高高兴兴的 ok=True 空操作，等于让这个标记继承它本来要消除的那个歧义。
    """
    c, _ = _wired_client(tmp_path)
    r = c.post("/api/feedback/999999/resolved",
               json={"resolved": True, "version": "v6.2.1"}).json()
    assert r["ok"] is False


def test_resolving_never_500s_without_storage():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from mast.api.context import AppContext
    from mast.api.routes.records import router as _r

    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(_r, prefix="/api")
    r = TestClient(app).post("/api/feedback/1/resolved", json={"resolved": True})
    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_an_old_db_without_the_resolved_columns_gets_them(tmp_path):
    """真机上的库早就存在，而建表语句是 CREATE TABLE IF NOT EXISTS。

    只改 DDL 的结果是：新列在开发机上有、在用户那台机器上没有，
    UPDATE 报「没有这一列」，处理状态永远标不上 —— 而这个功能存在的全部意义
    就是标得上。
    """
    import sqlite3

    from mast.logging.storage import ExperimentStorage

    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(str(db))
    con.executescript(
        "CREATE TABLE feedback ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,"
        " experiment_id TEXT, sample_id TEXT, conversation_id TEXT,"
        " kind TEXT NOT NULL DEFAULT 'rating', rating TEXT NOT NULL DEFAULT '',"
        " comment TEXT NOT NULL DEFAULT '', agent TEXT NOT NULL DEFAULT '',"
        " meta TEXT NOT NULL DEFAULT '{}');"
        "INSERT INTO feedback (timestamp, comment) VALUES ('2026-08-01', '旧的一条');"
    )
    con.commit()
    con.close()

    def cols():
        c2 = sqlite3.connect(str(db))
        try:
            return {r[1] for r in c2.execute("PRAGMA table_info(feedback)")}
        finally:
            c2.close()

    assert "resolved_at" not in cols(), "前提没成立，这条测试什么也没证明"

    st = ExperimentStorage(db)
    assert {"resolved_at", "resolved_version", "resolved_note"} <= cols()
    # 老行还在，而且标得上 —— 补了列却写不进等于没补。
    rows = st.get_feedback()
    assert len(rows) == 1 and rows[0]["comment"] == "旧的一条"
    assert st.set_feedback_resolved(rows[0]["id"], version="v6.2.1") is True
    assert st.get_feedback()[0]["resolved_version"] == "v6.2.1"


# ── 在役针尖聚合 ────────────────────────────────────────
#
# 「针尖记录等是不是也应该在实验记录中?」
#
# 区间交集的口径钉在 tests/v2/unit/logging/test_tips_in_service_during.py。
# 这一组钉的是**接线**:那个方法算得再对,没人在详情端点里调它,用户照样什么
# 都看不到。两件事各有各的失败方式,所以各有各的测试。

def test_experiment_detail_lists_the_tips_in_service(tmp_path) -> None:
    client, storage = _wired_client(tmp_path)
    r = client.post("/api/experiments", json={"name": "Au111 run", "goal": "g"})
    exp_id = r.json()["id"]
    storage.create_tip({"name": "W-01", "material": "W", "fabrication": "etched",
                        "form": "stm_wire"})

    detail = client.get(f"/api/experiments/{exp_id}").json()
    assert [t["name"] for t in detail["tips"]] == ["W-01"]
    tip = detail["tips"][0]
    assert tip["in_service_now"] is True       # removed_at 为空 = 还装着
    assert tip["id"], "没有 id 就链不回针尖卡片"


def test_the_detail_never_invents_tips_when_there_are_none(tmp_path) -> None:
    """空是空。「查不到」不能长成「有一根不知道是什么的针」。"""
    client, _ = _wired_client(tmp_path)
    exp_id = client.post("/api/experiments",
                         json={"name": "e", "goal": "g"}).json()["id"]
    assert client.get(f"/api/experiments/{exp_id}").json()["tips"] == []


def test_a_tip_registered_after_the_experiment_ended_is_not_listed(tmp_path) -> None:
    """结束之后才装的针不算这次实验的。

    这一条同时守住「实验的 end_time 真的被用上了」—— 只查「所有针尖」的实现
    在上面那条测试里看起来完全正常。
    """
    client, storage = _wired_client(tmp_path)
    exp_id = client.post("/api/experiments",
                         json={"name": "e", "goal": "g"}).json()["id"]
    ended = client.post(f"/api/experiments/{exp_id}/end", json={"status": "completed"})
    # 断言它真的结束了。少了这一句,一个 422(比如漏传 body)会让 end_time 保持
    # NULL,于是「实验还没结束 ⇒ 之后的针尖也算」这条**正确**行为把测试判红,
    # 而红的原因和它要测的东西毫无关系。写这条测试时就正好踩了一次。
    assert ended.status_code == 200 and ended.json()["ok"] is True
    storage.create_tip({"name": "以后才装的", "material": "W"})

    detail = client.get(f"/api/experiments/{exp_id}").json()
    assert detail["end_time"], "实验没结束的话这条测试测不到它要测的东西"
    assert [t["name"] for t in detail["tips"]] == []


def test_a_broken_tips_lookup_does_not_degrade_the_whole_detail(tmp_path) -> None:
    """针尖查不到 ≠ 这份实验记录坏了。

    整块 try 的实现会把一次针尖查询失败变成 degraded=True,于是用户看不到
    动作、样品、反馈 —— 为了一个新加的附属区块，赔上整页。
    """
    client, storage = _wired_client(tmp_path)
    exp_id = client.post("/api/experiments",
                         json={"name": "e", "goal": "g"}).json()["id"]

    def boom(_eid):
        raise RuntimeError("tips table is gone")

    storage.tips_in_service_during = boom
    detail = client.get(f"/api/experiments/{exp_id}").json()
    assert detail["found"] is True and detail["degraded"] is False
    assert detail["tips"] == []
