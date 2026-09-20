"""技能市场 API：读、写、以及那次让写变成**真的**的重建。

这个文件防的是与 ``test_hardware_modules_api.py`` 同一个事故形态：一个 **PERSIST
了但从来没 LIVE** 的开关。agent 的工具表在建图时冻结，所以改订阅会写进 JSON、更新
holder、把界面变绿 —— 而正在跑的 agent 手上一个工具都没变，直到重启。

外加一条这个功能特有的：**路由遮蔽**。``/api/skills/{name}`` 会把
``/api/skills/market`` 捕获成 ``name="market"`` 并返回 200 加一个错的 handler
（实测确认过），所以市场用了独立前缀 ``/skill-market``。那条断言在这里钉着。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mast.admin import reload_wiring
from mast.agents.instrument_control.tools import discover_instrument_skills
from mast.api.app import create_app
from mast.skills import subscription as sub
from mast.webui import builder_api


@pytest.fixture(scope="module")
def registry():
    return discover_instrument_skills()


class _FakeRuntime:
    """替身 CoreRuntime。记下重建被请求了几次，可注入排队/失败。"""

    def __init__(self, code=reload_wiring.ORCH_REBUILDING, boom=False):
        self.rebuild_calls = 0
        self._code = code
        self._boom = boom
        self._registry = None

    def request_agent_rebuild(self, reason=""):
        self.rebuild_calls += 1
        if self._boom:
            raise RuntimeError("orchestrator is wedged")
        return self._code, ""

    # refresh_after_skill_change 会摸这两个；给出「没有」的诚实答案。
    _safety = None
    _conv_engine = None

    def pending_agent_rebuild(self):
        return self._code == reload_wiring.ORCH_PENDING_QUEUED


@pytest.fixture
def client(subscription_store, registry, monkeypatch):
    builder_api.set_live_registry(registry)
    builder_api.invalidate_catalog()
    app = create_app()

    class _Ctx:
        skill_registry = registry
        live_app = None

    app.state.ctx = _Ctx()
    yield TestClient(app)
    builder_api.set_live_registry(None)
    builder_api.invalidate_catalog()


@pytest.fixture
def wired(client):
    """把一个假 runtime 接上去（reload_wiring 是模块级弱引用状态）。"""
    def _wire(rt):
        client.app.state.ctx.live_app = rt
        reload_wiring.wire_override_reload(rt)
        return rt
    yield _wire


def _names(registry) -> set[str]:
    return {m.name for m in registry.list_skills()}


# ─────────────────────────────────────────────────────────────────────────────
# 路由
# ─────────────────────────────────────────────────────────────────────────────

def test_market_routes_have_their_own_prefix(client):
    paths = set(client.app.openapi()["paths"])
    market = {p for p in paths if "skill-market" in p}
    assert market, "市场端点一个都没注册"
    assert not any(p.startswith("/api/skills/market") for p in paths), (
        "市场端点落在 /api/skills/ 下 —— 会被 /skills/{name} 捕获成 name='market'，"
        "返回 200 加一个错的 handler（本仓踩过两次）")


def test_the_shadowing_hazard_is_real_not_theoretical(client):
    """自证：如果当初把前缀写成 /api/skills/market，它**真的**会被吃掉。

    没有这一条，上面那条断言只是一句传说 —— 而传说会被下一个人以「多此一举」
    为由删掉。
    """
    r = client.get("/api/skills/market")
    assert r.status_code == 200 and r.json().get("name") == "market", (
        "遮蔽危险不再存在了？那就重新评估前缀选择，别直接删断言")


# ─────────────────────────────────────────────────────────────────────────────
# 读
# ─────────────────────────────────────────────────────────────────────────────

def test_status_starts_uncustomised_and_full(client, registry):
    body = client.get("/api/skill-market/status").json()
    assert body["customised"] is False
    assert body["market_total"] == len(_names(registry))
    assert body["subscribed_count"] == body["market_total"], "出厂就该是全订阅"
    assert body["unreadable"] == ""
    assert set(body["mandatory"]) == set(sub.MANDATORY_SKILLS)


def test_catalog_marks_subscription_and_mandatory(client, registry):
    body = client.get("/api/skill-market/catalog", params={"page_size": 50000}).json()
    assert body["total"] == len(_names(registry))
    rows = {r["name"]: r for r in body["skills"]}
    assert all(r["subscribed"] for r in rows.values()), "未定制时每一行都该是已订阅"
    for name in sub.MANDATORY_SKILLS:
        if name in rows:
            assert rows[name]["mandatory"] is True


def test_catalog_can_filter_by_subscription(client, registry):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    off = client.get("/api/skill-market/catalog",
                     params={"subscribed": "0", "page_size": 50000}).json()
    assert [r["name"] for r in off["skills"]] == ["SetBias"]
    on = client.get("/api/skill-market/catalog",
                    params={"subscribed": "1", "page_size": 50000}).json()
    assert "SetBias" not in {r["name"] for r in on["skills"]}
    assert on["total"] + off["total"] == len(_names(registry))


def test_market_shows_unsubscribed_skills_too(client):
    """市场是**全集**。看不见未订阅的技能，市场就没有意义了（也没法推荐）。"""
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    body = client.get("/api/skill-market/catalog", params={"page_size": 50000}).json()
    row = next(r for r in body["skills"] if r["name"] == "SetBias")
    assert row["subscribed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 写：落三处 + 请求重建
# ─────────────────────────────────────────────────────────────────────────────

def test_write_lands_in_all_three_places(client, subscription_store, registry):
    import json

    r = client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.json()["materialised"] is True, "第一次定制该 materialise"

    assert sub.is_subscribed("SetBias") is False                      # holder
    doc = json.loads(subscription_store.read_text(encoding="utf-8"))  # 盘
    assert "SetBias" not in doc["entries"] and doc["customised"] is True
    body = client.get("/api/skill-market/status").json()              # 回读
    assert body["customised"] is True
    assert body["subscribed_count"] == len(_names(registry)) - 1


def test_write_requests_a_tool_list_rebuild(client, wired):
    """THE test。没有它，退订在下次重启前是化妆品。"""
    rt = wired(_FakeRuntime())
    r = client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    assert rt.rebuild_calls == 1, (
        "改订阅后没有请求重建 agent 工具表。JSON 写了、holder 更新了、界面变绿了 —— "
        "而运行中的 agent 的工具表一个都没变。")
    assert r.json()["rebuild_note"], "重建结果必须有一句人话，不能只有 ok"


def test_write_invalidates_the_ui_mirror_cache(client, wired):
    """装载面变了 ⇒ ``/agents/tools`` 的缓存必须作废，否则界面继续显示旧表。"""
    from mast.webui import agents_api

    wired(_FakeRuntime())
    agents_api.set_live_registry(client.app.state.ctx.skill_registry)
    agents_api.warm_agent_tools()
    assert agents_api._AGENT_TOOLS_CACHE is not None, "前提没立住"
    try:
        client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
        assert agents_api._AGENT_TOOLS_CACHE is None, (
            "订阅变了但 /agents/tools 的缓存还在 —— 界面会继续显示退订前那张表")
    finally:
        agents_api.set_live_registry(None)
        agents_api.invalidate_agent_tools()


def test_queued_rebuild_says_so_instead_of_going_green(client, wired):
    """任务运行中重建会排队。存下来了 ≠ 生效了，必须说。"""
    wired(_FakeRuntime(code=reload_wiring.ORCH_PENDING_QUEUED))
    body = client.post("/api/skill-market/subscription",
                       json={"unsubscribe": ["SetBias"]}).json()
    assert body["ok"] is True
    assert body["agent_path_pending"] is True, "排队了却报 pending=False —— 替未来打包票"
    assert "任务" in body["rebuild_note"] or "排队" in body["rebuild_note"]


def test_rebuild_failure_is_reported_not_a_500(client, wired):
    wired(_FakeRuntime(boom=True))
    r = client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    assert r.status_code == 200
    assert r.json()["agent_path_pending"] is True


def test_headless_says_next_boot(client):
    """没有活的 runtime（独立 API 模式）—— 存下来了，下次启动生效。"""
    body = client.post("/api/skill-market/subscription",
                       json={"unsubscribe": ["SetBias"]}).json()
    assert body["agent_path_pending"] is True
    assert "启动" in body["rebuild_note"]


def test_fingerprint_matches_is_tri_state(client, registry, wired):
    """None ≠ False。进程没建过工具表时是「判断不了」，不是「没跟上」。"""
    from mast.agents.instrument_control import tools as ic_tools

    before = dict(ic_tools.LAST_WRAP)
    try:
        ic_tools.LAST_WRAP.update({"fingerprint": "", "at": 0.0, "n": 0, "overlaid": []})
        body = client.get("/api/skill-market/status").json()
        assert body["fingerprint_matches"] is None, "没建过表就该说判断不了"

        ic_tools.build_instrument_skill_tools(registry, lambda: None)
        body = client.get("/api/skill-market/status").json()
        assert body["fingerprint_matches"] is True

        wired(_FakeRuntime(code=reload_wiring.ORCH_PENDING_QUEUED))
        client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
        body = client.get("/api/skill-market/status").json()
        assert body["fingerprint_matches"] is False, (
            "工具表还是退订前那份，探针却说跟上了")
    finally:
        ic_tools.LAST_WRAP.update(before)


# ─────────────────────────────────────────────────────────────────────────────
# 写：拒绝与诚实
# ─────────────────────────────────────────────────────────────────────────────

def test_unsubscribing_a_mandatory_skill_is_refused_with_a_reason(client):
    body = client.post("/api/skill-market/subscription",
                       json={"unsubscribe": ["WithdrawTip"]}).json()
    assert body["skipped_mandatory"] == ["WithdrawTip"]
    assert "WithdrawTip" in body["rebuild_note"], "拒绝了却不说是哪一个"
    assert sub.is_subscribed("WithdrawTip") is True


def test_unknown_names_are_reported_not_silently_dropped(client):
    body = client.post("/api/skill-market/subscription",
                       json={"subscribe": ["NoSuchSkill"]}).json()
    assert body["unknown"] == ["NoSuchSkill"]


def test_the_write_schema_has_no_whole_replace_field(client):
    """被否掉的设计钉成测试：整体替换会在无关编辑时抹平几百条订阅。"""
    from mast.api.schemas_skill_market import SubscriptionWriteRequest

    assert set(SubscriptionWriteRequest.model_fields) == {"subscribe", "unsubscribe"}
    r = client.post("/api/skill-market/subscription", json={"entries": ["A"]})
    assert r.status_code == 422, "写请求收下了名单外的字段（extra 没 forbid）"


def test_every_write_field_is_handled_by_the_route():
    """第三处白名单闸门：pydantic 收下的每个字段，路由里都得有人管它。"""
    import inspect

    from mast.api.routes import skill_market
    from mast.api.schemas_skill_market import SubscriptionWriteRequest

    src = inspect.getsource(skill_market.write_subscription)
    unhandled = [f for f in SubscriptionWriteRequest.model_fields
                 if f"body.{f}" not in src]
    assert not unhandled, (
        f"写 schema 里这些字段没有任何人处理：{unhandled} —— "
        "pydantic 收下、路由不管 = 界面上填了、什么都没发生")


def test_unsubscribing_everything_leaves_exactly_the_mandatory_ones(client, registry):
    """退订一切之后，工具面上剩下的**恰好**是必装项。"""
    market = _names(registry)
    body = client.post("/api/skill-market/subscription",
                       json={"unsubscribe": sorted(market)}).json()
    assert body["ok"] is True
    expected = len(sub.MANDATORY_SKILLS & market)
    assert body["subscribed_count"] == expected, (
        f"退订了一切，界面却说还有 {body['subscribed_count']} 个订阅着")
    assert client.get("/api/skill-market/status").json()["subscribed_count"] == expected
    assert len(sub.unloaded_skill_names(market)) == len(market) - expected


def test_an_empty_subscription_counts_as_zero_not_as_everything():
    """空集是一个答案，不是「没有答案」。

    ``len(subscribed_names() or market)`` 在空集上会滑到 market 去 —— 于是「一个
    技能都没订阅」显示成「全部 459 个都订阅着」。

    **为什么这条是单元测试而不是端到端**：走 API 退订一切之后 entries 里还留着
    必装项（非空），所以那条路根本触碰不到 falsy 分支 —— 我先写的端到端版本在
    退回旧公式之后**仍然是绿的**。这个空集只有在必装项与市场不相交时才会出现
    （降级/局部注册表），而规则本身住在 ``_count`` 里，就在这里钉它。
    """
    from mast.api.routes.skill_market import _count

    market = {"A", "B", "C"}
    assert _count(frozenset(), market) == 0, "空订阅被当成了全订阅"
    assert _count(None, market) == 3, "未定制（None）才是全订阅"
    assert _count(frozenset({"A"}), market) == 1
    # 必装项无条件计入（即使 entries 里没有它）
    mand = next(iter(sub.MANDATORY_SKILLS))
    assert _count(frozenset(), market | {mand}) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 审计流：写了，而且**有人读**
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_endpoint_shows_who_changed_what(client):
    """第一版的 audit 是一条只写不读的日志 —— 唯一的消费方是一条测试。

    [[producer_wired_consumer_absent]]：「已经记下来了」是生产方的话，要问谁读它。
    """
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    rows = client.get("/api/skill-market/audit").json()["entries"]
    assert rows, "改了订阅，审计流里却什么都没有"
    actions = [(r["action"], r["via"], tuple(r["skills"])) for r in rows]
    assert ("materialise", "ui", ()) in actions, "第一次定制没留下 materialise"
    assert ("unsubscribe", "ui", ("SetBias",)) in actions


def test_audit_survives_a_reset(client):
    """reset 清掉 entries，**不清 audit** —— 「他什么时候把定制推倒重来的」
    正是事后最想知道的一件事。"""
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    client.post("/api/skill-market/subscription/reset")
    actions = [r["action"] for r in client.get("/api/skill-market/audit").json()["entries"]]
    assert "reset" in actions
    assert "unsubscribe" in actions, "reset 把它之前的历史一起抹了"


def test_audit_records_the_recommendation_that_caused_it(client):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    rec = sub.add_recommendation("SetBias", by_agent="ic")["recommendation"]
    client.post(f"/api/skill-market/recommendations/{rec['id']}/resolve",
                json={"accept": True})
    vias = [r["via"] for r in client.get("/api/skill-market/audit").json()["entries"]]
    assert any(v == f"recommendation:{rec['id']}" for v in vias), (
        "接受推荐改了订阅面，审计流却说不出是哪条推荐干的")


def test_audit_is_capped_and_newest_last(client, registry):
    for name in sorted(_names(registry))[:5]:
        client.post("/api/skill-market/subscription", json={"unsubscribe": [name]})
    rows = client.get("/api/skill-market/audit", params={"limit": 2}).json()["entries"]
    assert len(rows) == 2, "limit 没生效"
    assert rows[-1]["at"] >= rows[0]["at"], "顺序反了 —— 最新的该在最后"


# ─────────────────────────────────────────────────────────────────────────────
# 出厂态：新技能**立刻**在工具面上，这是有意的
# ─────────────────────────────────────────────────────────────────────────────

def test_in_factory_state_a_new_skill_is_immediately_on_the_face(client, registry):
    """**被否掉的设计钉成测试。**

    一度想让「agent 新造的技能不自动进工具面」在出厂态也成立 —— 查下来那是错的：

    * 技能工坊（``agents/_shared/skill_forge_tools.py``）**明确不指望订阅门当人闸**。
      它的论证是治理在执行层：每个子步走 ``ExecutionContext.run``（abort 闸 / 样品门 /
      五条 Layer-0 硬闸 / 参数包络 / DANGEROUS 留痕），安全级与能力标签由
      ``interpreter`` 从子步**继承**，声明只能收紧不能放松。
    * 在订阅层加一道「新技能要人点头」的闸，等于把工具面重新当成安全边界 ——
      而 2026-08-20 conduct 那次拍板正是把它拆掉（「把关在服务端，不靠不给工具」）。
    * 出厂态加闸还会**破坏零回归**：一台从没定制过订阅的机器，升级当天工具面就会变。

    **要推翻这条需要回答**：有哪一个危险动作，是「技能在工具面上」能做到、而
    「agent 用 run_composite 直接跑同一个 spec」做不到的？现在的答案是一个都没有。

    真正该补的不是闸，是**可见性** —— 见前端 `locallyAuthoredUnsubscribed`。
    """
    assert sub.is_customised() is False
    market = _names(registry)
    assert sub.unloaded_skill_names(market | {"ForgedByAgent"}) == frozenset(), (
        "出厂态下新技能被挡在工具面外了 —— 这会让升级当天的工具面变化")


def test_after_customising_a_new_skill_waits_for_a_nod(client, registry):
    """定制之后就反过来了：新技能只进市场。这是 materialise 的副产品。"""
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    market = _names(registry)
    assert "ForgedByAgent" in sub.unloaded_skill_names(market | {"ForgedByAgent"})


# ─────────────────────────────────────────────────────────────────────────────
# builder palette 的 join
# ─────────────────────────────────────────────────────────────────────────────

def test_builder_palette_carries_subscription_columns(client):
    body = client.get("/api/builder/catalog", params={"page_size": 50000}).json()
    rows = {r["name"]: r for r in body["skills"]}
    assert rows, "palette 是空的，下面的断言证明不了什么"
    assert all(r["subscribed"] for r in rows.values()), "未定制时每一行都该是已订阅"
    for name in sub.MANDATORY_SKILLS:
        if name in rows:
            assert rows[name]["mandatory"] is True


def test_builder_palette_reflects_unsubscription_in_the_column(client):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    rows = {r["name"]: r for r in
            client.get("/api/builder/catalog", params={"page_size": 50000}).json()["skills"]}
    assert rows["SetBias"]["subscribed"] is False


def test_builder_palette_does_not_filter_by_default(client, registry):
    """默认不过滤 —— BuilderPage 还没有「只看订阅/全市场」的开关。

    这条钉的是一个**刻意的半成品状态**：这时候把默认改成只显示订阅项，用户会
    看到 palette 无缘无故少了一半，而界面上没有任何东西解释为什么。开关和默认
    一起进（二期）。要改默认，先加开关。
    """
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    body = client.get("/api/builder/catalog", params={"page_size": 50000}).json()
    assert body["total"] == len(_names(registry)), "默认就把 palette 过滤掉了"


def test_builder_palette_can_filter_on_request(client, registry):
    off = client.get("/api/builder/catalog",
                     params={"subscribed": "0", "page_size": 50000}).json()
    assert off["total"] == 0, "未定制时不该有「未订阅」的行"
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    off = client.get("/api/builder/catalog",
                     params={"subscribed": "0", "page_size": 50000}).json()
    assert [r["name"] for r in off["skills"]] == ["SetBias"]
    on = client.get("/api/builder/catalog",
                    params={"subscribed": "1", "page_size": 50000}).json()
    assert on["total"] == len(_names(registry)) - 1


def test_reset_goes_back_to_full(client, registry):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    body = client.post("/api/skill-market/subscription/reset").json()
    assert body["ok"] is True and body["customised"] is False
    assert sub.unloaded_skill_names(_names(registry)) == frozenset()


def test_subscription_is_not_behind_the_admin_pin():
    """钉住裁定：订阅不授能，所以不进 GUARDED_KEYS。

    要推翻它需要回答：订阅列表能让哪一个此前做不到的危险动作变得能做？
    """
    from mast.api.admin_pin import GUARDED_KEYS

    assert not any("subscription" in k or "market" in k for k in GUARDED_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# 推荐：agent 只能写 pending
# ─────────────────────────────────────────────────────────────────────────────

def test_pending_lifecycle_full_loop(client, wired):
    rt = wired(_FakeRuntime())
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    calls_before = rt.rebuild_calls

    rec = sub.add_recommendation("SetBias", by_agent="instrument_control",
                                 reason="要调偏压")["recommendation"]
    listed = client.get("/api/skill-market/recommendations").json()
    assert [p["skill"] for p in listed["pending"]] == ["SetBias"]
    assert sub.is_subscribed("SetBias") is False, "光是列出来就把订阅改了？"

    body = client.post(f"/api/skill-market/recommendations/{rec['id']}/resolve",
                       json={"accept": True}).json()
    assert body["ok"] is True and body["changed"] == ["SetBias"]
    assert sub.is_subscribed("SetBias") is True
    assert rt.rebuild_calls > calls_before, "接受推荐改了工具面，却没请求重建"
    assert client.get("/api/skill-market/recommendations").json()["pending"] == []


def test_rejecting_changes_nothing_but_leaves_a_trace(client):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    rec = sub.add_recommendation("SetBias")["recommendation"]
    client.post(f"/api/skill-market/recommendations/{rec['id']}/resolve",
                json={"accept": False})
    assert sub.is_subscribed("SetBias") is False
    resolved = client.get("/api/skill-market/recommendations").json()["resolved"]
    assert [r["status"] for r in resolved] == ["rejected"]


def test_resolving_an_unknown_recommendation_is_refused(client):
    r = client.post("/api/skill-market/recommendations/rec-nope/resolve",
                    json={"accept": True})
    assert r.status_code == 200 and r.json()["ok"] is False


def test_catalog_surfaces_the_pending_recommendation_id(client):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    rec = sub.add_recommendation("SetBias")["recommendation"]
    rows = client.get("/api/skill-market/catalog",
                      params={"page_size": 50000}).json()["skills"]
    row = next(r for r in rows if r["name"] == "SetBias")
    assert row["pending_rec_id"] == rec["id"], (
        "推荐写下了，但目录里看不见它 —— 生产方接了，消费方不存在")
