"""技能检索（按动作 / Nanonis 命令）、技能卡、组合技能起草与保存、Python 技能提议。"""
from __future__ import annotations

import pytest

from _ext_world import H, wait_terminal


@pytest.fixture(scope="module")
def real_registry():
    """真的内置技能全集 —— 动词倒排要在真源码上证明自己不是空转。"""
    from mast.core.registry import SkillRegistry

    reg = SkillRegistry()
    reg.discover()
    return reg


def test_the_verb_index_is_not_vacuous_on_the_real_tree(real_registry):
    from mast.api.ext.skills import SkillIndex

    rows = SkillIndex().rows(real_registry)
    with_verbs = [n for n, r in rows.items() if r["verbs"]]
    assert len(with_verbs) >= 100, f"只有 {len(with_verbs)} 个技能读出了命令 —— 扫描器空转？"
    # 几个真实的对应（技能 → 它一定会发的命令）
    assert "Bias_Get" in rows["GetBias"]["verbs"]
    assert "Bias_Set" in rows["SetBias"]["verbs"]
    assert any(v.startswith("Scan_") for v in rows["StartScan"]["verbs"])


def test_the_card_and_the_contribution_checker_agree(real_registry):
    """技能卡的足迹与 ``scripts/skill_check.py`` 出自同一份分析 —— 同一个技能不许两种结论。"""
    from mast.api.ext.skills import SkillIndex, footprint
    from mast.skills.compliance import skill_footprint

    rows = SkillIndex().rows(real_registry)
    for name in ("GetBias", "SetBias", "StartScan", "FullScan", "StopScan"):
        fp = skill_footprint(real_registry, name)
        assert footprint(rows[name]) == fp.effective, name
        assert rows[name]["verbs"] == list(fp.verbs), name
    assert footprint(rows["GetBias"]) == "hardware-read-only"
    assert footprint(rows["SetBias"]) == "hardware-write"


def test_the_footprint_reads_source_in_a_frozen_build(real_registry, monkeypatch):
    """打包版里 ``inspect.getsourcefile`` 指向的 .py 不存在；分析必须退到随包的源码副本，
    否则仪器上的每个技能都会读成 unknown。"""
    import inspect

    import mast.skills.compliance as comp

    monkeypatch.setattr(inspect, "getsourcefile", lambda obj: r"Z:\frozen\PYZ\missing.py")
    comp._parse_file.cache_clear()
    for name, verb in (("SetBias", "Bias_Set"), ("GetBias", "Bias_Get")):
        fp = comp.skill_footprint(real_registry, name)
        assert verb in fp.verbs, fp
        assert not any("源码读不到" in r for r in fp.reasons), fp
    assert comp.skill_footprint(real_registry, "GetBias").footprint == "hardware-read-only"


def test_search_finds_skills_by_the_nanonis_command_they_send(world, real_registry):
    world.ctx.skill_registry = real_registry
    r = world.client.get("/skills/search", params={"q": "Bias_Set"}, headers=H()).json()
    names = [x["name"] for x in r["results"]]
    assert "SetBias" in names, names[:10]
    hit = next(x for x in r["results"] if x["name"] == "SetBias")
    assert any(m.startswith("verb:") for m in hit["matched_on"])
    assert hit["footprint"] == "hardware-write"
    assert hit["origin_code"] == "builtin" and hit["official"] is True


def test_search_by_intent_word_reaches_the_official_skill(world, real_registry):
    world.ctx.skill_registry = real_registry
    r = world.client.get("/skills/search", params={"q": "扫描"}, headers=H()).json()
    top = [x["name"] for x in r["results"][:15]]
    assert "FullScan" in top, top


def test_the_skill_card_has_what_an_agent_needs_before_running(world):
    card = world.client.get("/skills/ExtRead", headers=H()).json()
    assert card["safety_level"] == "auto" and card["category"] == "read"
    assert card["origin_code"] != "builtin" and card["official"] is False, (
        "测试里临时注册的探针被说成了官方技能")
    p = {x["name"]: x for x in card["parameters"]}
    assert p["x_m"]["unit"] == "m" and p["x_m"]["max"] == 1e-6
    assert card["si_params"] == {"x_m": "prefix_required"}
    assert card["takes_instrument_token"] is False and card["requires_sample"] is False
    assert card["tool_face"]["disabled"] is False
    assert card["duration"]["measured"] is None and card["duration"]["note"]


def test_the_card_reports_measured_duration_from_this_instruments_records(world):
    for _ in range(3):
        jid = world.client.post("/jobs", json={"skill": "ExtRead"}, headers=H()).json()["job_id"]
        assert wait_terminal(world.client, jid)["state"] == "succeeded"
    card = world.client.get("/skills/ExtRead", headers=H()).json()
    m = card["duration"]["measured"]
    assert m and m["n"] == 3 and 0 <= m["p50_s"] <= m["p95_s"] <= m["max_s"], m


def test_unknown_skill_card_is_404_with_suggestions(world):
    r = world.client.get("/skills/ExtRea", headers=H())
    assert r.status_code == 404 and "ExtRead" in r.json()["did_you_mean"]


def test_composite_draft_returns_the_syntax_on_request(world):
    r = world.client.post("/composites/draft", json={"spec": "?"}, headers=H()).json()
    assert r["ok"] is False and "CompositeSpec" in r["syntax"]


def test_composite_save_signs_as_the_external_agent_and_registers(world):
    spec = {"name": "ExtTwoReads", "description": "读两次",
            "safety_level": "auto", "params": [],
            "nodes": [{"type": "step", "id": "a", "skill": "ExtRead", "params": {}},
                      {"type": "step", "id": "b", "skill": "ExtRead", "params": {}}]}
    d = world.client.post("/composites/draft", json={"spec": spec}, headers=H()).json()
    assert d["ok"] is True, d
    s = world.client.post("/composites", json={"spec": spec}, headers=H()).json()
    assert s["ok"] is True and s["hot_registered"] is True, s
    assert world.reg.has("ExtTwoReads")
    import mast.webui.composite_panel as cp

    meta = cp._store.load_meta("ExtTwoReads")
    assert meta["_author"] == "ext:tester" and meta["_origin"] == "ext_gateway"
    # 同名覆盖要带 base_version（乐观锁）
    s2 = world.client.post("/composites", json={"spec": spec}, headers=H()).json()
    assert s2["ok"] is False and s2["error"] == "base_version_required"


def test_the_card_follows_a_new_version_of_the_same_composite(world):
    """同名、同数量的更新也要让技能卡换成新版本（缓存键只看名字会一直给旧的子步）。"""
    spec = {"name": "ExtCombo", "description": "v1", "safety_level": "auto", "params": [],
            "nodes": [{"type": "step", "id": "a", "skill": "ExtRead", "params": {}},
                      {"type": "step", "id": "b", "skill": "ExtRead", "params": {}}]}
    s1 = world.client.post("/composites", json={"spec": spec}, headers=H()).json()
    assert s1["ok"] is True, s1
    assert world.client.get("/skills/ExtCombo", headers=H()).json()["sub_skills"] == ["ExtRead"]
    spec2 = {**spec, "description": "v2",
             "nodes": [{"type": "step", "id": "a", "skill": "ExtRead", "params": {}},
                       {"type": "step", "id": "b", "skill": "ExtWrite", "params": {}}]}
    s2 = world.client.post("/composites", json={"spec": spec2, "base_version": s1["version"]},
                           headers=H()).json()
    assert s2["ok"] is True, s2
    card = world.client.get("/skills/ExtCombo", headers=H()).json()
    assert card["sub_skills"] == ["ExtRead", "ExtWrite"], card["sub_skills"]


def test_a_pure_alias_composite_is_refused_like_on_the_agent_track(world):
    spec = {"name": "JustRead", "description": "套壳", "safety_level": "auto", "params": [],
            "nodes": [{"type": "step", "id": "a", "skill": "ExtRead", "params": {}}]}
    s = world.client.post("/composites", json={"spec": spec}, headers=H()).json()
    assert s["ok"] is False


_GOOD = '''
from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata, SkillResult
from mast.skills.base import BaseSkill


class ExtProposed(BaseSkill):
    def metadata(self):
        return SkillMetadata(name="ExtProposed", version="0.1.0",
                             category=SkillCategory.ANALYSIS, safety_level=SafetyLevel.AUTO,
                             description="提议的分析技能")

    def execute(self, context, params):
        return SkillResult(skill_name="ExtProposed", success=True)
'''


def test_a_proposal_is_written_for_review_never_registered_or_run(world):
    r = world.client.post("/skills/proposals", json={
        "name": "ExtProposed", "code": _GOOD, "rationale": "组合表达不了"}, headers=H()).json()
    assert r["ok"] is True and r["enabled"] is False, r
    p = world.tmp / "custom_skills" / "ExtProposed.py"
    assert p.is_file() and "ext:tester" in p.read_text(encoding="utf-8")
    assert not world.reg.has("ExtProposed"), "提议被注册了"
    comp = r["compliance"]
    assert "unavailable" not in comp, f"合规报告没算出来（签名对不上？）：{comp}"
    assert comp["skill_name"] == "ExtProposed" and isinstance(comp["ok"], bool), comp


def test_a_proposal_with_forbidden_code_is_rejected(world):
    bad = _GOOD.replace("from mast.core.types", "import os\nfrom mast.core.types")
    r = world.client.post("/skills/proposals", json={
        "name": "ExtBad", "code": bad, "rationale": "x"}, headers=H()).json()
    assert r["ok"] is False and r["error"] == "rejected"
    assert not (world.tmp / "custom_skills" / "ExtBad.py").exists()
