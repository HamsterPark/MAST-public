"""订阅列表的分享：导出 → 别人的机器 → 导入。

两条红线在这里钉着：

1. **代码不随单走。** manifest 里只有名字（以及 ``user_composite`` 的纯数据 spec）。
   一份能自动带别人 .py 过来的分享格式就是一条 RCE 通道 —— 这与 push server
   ``/skills/upload`` 拒收 .py 是同一条线，那边也有一条同形的测试钉着。
2. **缺失的技能要点名，不能静默丢。** 对方机器上没有的技能，导入方必须看见它是
   哪一个、来自哪种来源、下一步该干什么。「导入成功」而清单少了三条，是本仓
   [[unknown_is_not_an_answer]] 那一族里最难发现的一种。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mast.agents.instrument_control.tools import discover_instrument_skills
from mast.api.app import create_app
from mast.skills import subscription as sub
from mast.webui import builder_api

_SPEC = {
    "name": "MarketImportProbe",
    "description": "manifest 导入用的探针组合",
    "safety_level": "confirm",
    "params": [],
    "nodes": [],
    "tags": [],
}


@pytest.fixture(scope="module")
def registry():
    return discover_instrument_skills()


@pytest.fixture
def client(subscription_store, seeded_composite_store, registry):
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


def _export(client) -> dict:
    r = client.get("/api/skill-market/export")
    assert r.status_code == 200
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# 导出
# ─────────────────────────────────────────────────────────────────────────────

def test_export_of_an_uncustomised_machine_is_the_whole_market(client, registry):
    man = _export(client)
    assert man["kind"] == "mast-skill-subscription"
    assert man["customised"] is False
    assert {e["name"] for e in man["entries"]} == {m.name for m in registry.list_skills()}


def test_export_carries_the_origin_of_each_entry(client):
    man = _export(client)
    sources = {e["source"] for e in man["entries"]}
    assert "builtin" in sources
    assert sources <= {"builtin", "composite", "paper", "user_composite", "overlay",
                       "custom", "agent_tool", "other", "absent"}, (
        f"来源词汇跑出了 classify_origin 的闭集：{sources}")


def test_no_python_source_travels_in_the_manifest(client):
    """红线：代码不随订阅单走。"""
    import json

    blob = json.dumps(_export(client), ensure_ascii=False)
    for smell in ("def execute", "import os", "class ", "__import__", "lambda "):
        assert smell not in blob, f"manifest 里出现了代码痕迹：{smell!r}"
    for e in _export(client)["entries"]:
        if e["source"] != "user_composite":
            assert e["spec"] is None, (
                f"{e['name']}（{e['source']}）内嵌了 spec —— 只有用户组合可以，"
                "因为只有它是纯数据")


def test_export_reflects_the_subscription_not_the_market(client):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    names = {e["name"] for e in _export(client)["entries"]}
    assert "SetBias" not in names
    assert _export(client)["customised"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 导入
# ─────────────────────────────────────────────────────────────────────────────

def test_roundtrip_reproduces_the_subscription(client, registry):
    client.post("/api/skill-market/subscription",
                json={"unsubscribe": ["SetBias", "StartScan"]})
    man = _export(client)

    client.post("/api/skill-market/subscription/reset")
    assert sub.is_customised() is False

    r = client.post("/api/skill-market/import", json={"manifest": man, "mode": "replace"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert sub.is_subscribed("SetBias") is False
    assert sub.is_subscribed("StartScan") is False
    assert sub.unloaded_skill_names({m.name for m in registry.list_skills()}) == {
        "SetBias", "StartScan"}


def test_missing_skills_are_named_not_dropped(client):
    man = _export(client)
    man["entries"].append({"name": "GhostSkill", "source": "overlay", "version": ""})
    body = client.post("/api/skill-market/import",
                       json={"manifest": man, "mode": "replace"}).json()
    missing = {m["name"]: m for m in body["report"]["missing"]}
    assert "GhostSkill" in missing
    assert missing["GhostSkill"]["hint"], "只说「没有」，不说下一步该干什么"
    assert sub.is_subscribed("GhostSkill") is False, (
        "本机没有的技能被写进了活订阅 —— 一个幽灵名字从此躺在清单里")


def test_dry_run_reports_without_changing_anything(client):
    man = _export(client)
    man["entries"] = [e for e in man["entries"] if e["name"] != "SetBias"]
    before = sub.is_customised()

    body = client.post("/api/skill-market/import",
                       json={"manifest": man, "mode": "replace", "dry_run": True}).json()
    assert body["dry_run"] is True and body["ok"] is True
    assert body["report"]["matched"], "dry_run 该给出预览，不是空报告"
    assert sub.is_customised() is before, "dry_run 改了状态"
    assert sub.is_subscribed("SetBias") is True


def test_mandatory_is_forced_in_on_import(client):
    """别人手改过的 manifest 也砍不掉必装项。"""
    man = _export(client)
    man["entries"] = [e for e in man["entries"]
                      if e["name"] not in sub.MANDATORY_SKILLS]
    body = client.post("/api/skill-market/import",
                       json={"manifest": man, "mode": "replace"}).json()
    assert set(body["report"]["mandatory_added"]) >= (
        sub.MANDATORY_SKILLS & {"SafeRetract", "WithdrawTip", "StopScan"})
    for name in sub.MANDATORY_SKILLS:
        assert sub.is_subscribed(name) is True


def test_merge_keeps_what_was_already_there(client):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    man = _export(client)
    man["entries"] = [{"name": "SetBias", "source": "builtin", "version": ""}]

    client.post("/api/skill-market/import", json={"manifest": man, "mode": "merge"})
    assert sub.is_subscribed("SetBias") is True, "merge 该把对方的加进来"
    assert sub.is_subscribed("StartScan") is True, "merge 却把本机原有的丢了"


def test_replace_really_replaces(client):
    man = {"kind": "mast-skill-subscription", "schema_version": 1,
           "entries": [{"name": "SetBias", "source": "builtin", "version": ""}]}
    client.post("/api/skill-market/import", json={"manifest": man, "mode": "replace"})
    assert sub.is_subscribed("SetBias") is True
    assert sub.is_subscribed("StartScan") is False


def test_embedded_composite_goes_through_the_existing_validator(client):
    """内嵌 spec 走既有 builder 保存路径 —— 校验一行都不在市场路由里重实现。

    先证正例（一份合法 spec 真的落地了），再证负例，否则「非法被拒」可能只是因为
    这条路根本没通。
    """
    good = {"kind": "mast-skill-subscription", "schema_version": 1,
            "entries": [{"name": "MarketImportProbe", "source": "user_composite",
                         "version": "1", "spec": dict(_SPEC)}]}
    body = client.post("/api/skill-market/import",
                       json={"manifest": good, "mode": "merge"}).json()
    assert body["report"]["composites_saved"] == ["MarketImportProbe"], (
        f"合法 spec 没能落地：{body['report']['composites_failed']}")

    bad_spec = {**_SPEC, "name": "MarketImportBad", "nodes": "不是列表"}
    bad = {"kind": "mast-skill-subscription", "schema_version": 1,
           "entries": [{"name": "MarketImportBad", "source": "user_composite",
                        "version": "1", "spec": bad_spec}]}
    body = client.post("/api/skill-market/import",
                       json={"manifest": bad, "mode": "merge"}).json()
    assert body["report"]["composites_saved"] == []
    assert [f["name"] for f in body["report"]["composites_failed"]] == ["MarketImportBad"]


# ─────────────────────────────────────────────────────────────────────────────
# 拒绝的形状
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("man,why", [
    ({}, "kind"),
    ({"kind": "something-else", "entries": []}, "kind"),
    ({"kind": "mast-skill-subscription"}, "entries"),
    ({"kind": "mast-skill-subscription", "entries": "nope"}, "entries"),
])
def test_a_file_that_is_not_a_subscription_list_is_refused(client, man, why):
    body = client.post("/api/skill-market/import",
                       json={"manifest": man, "mode": "replace"}).json()
    assert body["ok"] is False
    assert why in body["reason"] or "订阅" in body["reason"]


def test_unknown_mode_is_refused(client):
    man = _export(client)
    body = client.post("/api/skill-market/import",
                       json={"manifest": man, "mode": "obliterate"}).json()
    assert body["ok"] is False and "mode" in body["reason"]


def test_import_request_forbids_extra_fields(client):
    r = client.post("/api/skill-market/import",
                    json={"manifest": {}, "mode": "replace", "force": True})
    assert r.status_code == 422
