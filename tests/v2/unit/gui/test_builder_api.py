"""P1 builder backend API tests — catalog / composite CRUD / validate / favorites.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/gui/test_builder_api.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from mast.core.registry import SkillRegistry
from mast.core.types import (
    ParameterSpec, SafetyLevel, SkillCategory, SkillMetadata, SkillResult,
)
from mast.webui import builder_api, composite_panel
from mast.skills.base import BaseSkill
from mast.skills.builtins.motor import MotorMove
from mast.skills.composite.version_store import CompositeVersionStore


class FakeBias(BaseSkill):
    def metadata(self):
        return SkillMetadata(
            name="FakeBias", version="1.2.0", category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM, description="set bias",
            parameters=[ParameterSpec(name="bias_v", type="float", unit="V",
                                      required=True, min_value=-10, max_value=10)],
            tags=["bias", "write"], composition_level=0)
    def execute(self, ctx, params):
        return SkillResult(skill_name="FakeBias", success=True, data=params)


class FakeRead(BaseSkill):
    def metadata(self):
        return SkillMetadata(
            name="FakeRead", version="1.0.0", category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO, description="read something",
            parameters=[], tags=["read"], composition_level=0)
    def execute(self, ctx, params):
        return SkillResult(skill_name="FakeRead", success=True, data={"v": 1})


SPEC = {
    "name": "WF1", "description": "demo", "safety_level": "confirm",
    "params": [{"name": "n", "type": "int", "default": 2}],
    "nodes": [
        {"type": "step", "id": "a", "skill": "FakeRead", "params": {}},
        {"type": "step", "id": "b", "skill": "FakeBias",
         "params": {"bias_v": 1.5}},
    ],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    reg = SkillRegistry()
    for cls in (FakeBias, FakeRead, MotorMove):
        reg.register(cls)
    store = CompositeVersionStore(root=tmp_path / "cs")
    # wire module globals (restored by monkeypatch teardown)
    monkeypatch.setattr(builder_api, "_registry", reg)
    monkeypatch.setattr(builder_api, "_catalog_cache", None)
    monkeypatch.setattr(builder_api, "_favorites_path",
                        lambda: tmp_path / "fav.json")
    monkeypatch.setattr(composite_panel, "_store", store)
    monkeypatch.setattr(composite_panel, "_live_registry", reg)
    monkeypatch.setattr(composite_panel, "_agent_refresh", None)
    app = Starlette(routes=builder_api.build_routes())
    app.auth = None  # auth disabled — guard passes (修复项 tested separately)
    app.auth_dependency = None
    return TestClient(app), reg, store


# ── catalog ──────────────────────────────────────────────────────────────────

class TestCatalog:
    def test_index_fields(self, client):
        c, reg, _ = client
        r = c.get("/skills/catalog")
        assert r.status_code == 200
        skills = {e["name"]: e for e in r.json()["skills"]}
        assert r.json()["total"] == 3
        fb = skills["FakeBias"]
        assert fb["category"] == "WRITE" and fb["safety"] == "confirm"
        assert fb["source"] == "other"          # test class, not in mast.skills
        assert skills["MotorMove"]["source"] == "builtin"
        assert skills["MotorMove"]["domain"] == "导航"  # DOMAINS 反查
        assert fb["domain"] == "其他"            # 兜底桶 — 名单外技能必可见

    def test_full_card(self, client):
        c, *_ = client
        card = c.get("/skills/catalog/FakeBias").json()
        assert card["version"] == "1.2.0"
        p = card["parameters"][0]
        assert p["name"] == "bias_v" and p["min"] == -10 and p["unit"] == "V"
        assert c.get("/skills/catalog/Nope").status_code == 404

    def test_query_filters(self, client):
        c, *_ = client
        assert c.get("/skills/catalog?category=READ").json()["total"] == 1
        assert c.get("/skills/catalog?q=bias").json()["total"] == 1
        assert c.get("/skills/catalog?source=builtin").json()["total"] == 1
        assert c.get("/skills/catalog?safety=auto").json()["total"] == 1

    def test_cache_invalidation_on_change(self, client):
        c, reg, _ = client
        assert c.get("/skills/catalog").json()["total"] == 3
        reg.unregister("FakeRead")
        builder_api.invalidate_catalog()
        assert c.get("/skills/catalog").json()["total"] == 2


# ── composite CRUD ───────────────────────────────────────────────────────────

class TestCompositeCrud:
    def test_save_load_roundtrip_and_hot_register(self, client):
        c, reg, store = client
        r = c.post("/composites/WF1", json={"spec": SPEC, "base_version": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["version"] == 1 and body["hot_registered"] is True
        assert reg.has("WF1")                       # 热注册生效
        got = c.get("/composites/WF1").json()
        assert got["spec"]["name"] == "WF1"
        assert got["versions"][0]["version"] == 1
        assert c.get("/composites").json()["composites"][0]["name"] == "WF1"

    def test_save_cas_conflict_409(self, client):
        c, *_ = client
        c.post("/composites/WF1", json={"spec": SPEC, "base_version": 0})
        c.post("/composites/WF1", json={"spec": SPEC, "base_version": 1})
        r = c.post("/composites/WF1", json={"spec": SPEC, "base_version": 1})
        assert r.status_code == 409
        assert r.json()["error"] == "version_conflict"
        assert r.json()["stored_version"] == 2

    def test_save_invalid_spec_422(self, client):
        c, *_ = client
        bad = dict(SPEC, nodes=[{"type": "step", "id": "x"}])  # no skill
        r = c.post("/composites/WF1", json={"spec": bad})
        assert r.status_code == 422

    def test_name_mismatch_400(self, client):
        c, *_ = client
        assert c.post("/composites/Other", json={"spec": SPEC}).status_code == 400

    def test_sync_ready_record_fields(self, client, tmp_path):
        c, *_ = client
        c.post("/composites/WF1", json={"spec": SPEC})
        import json as _json
        raw = _json.loads((tmp_path / "cs" / "WF1.json").read_text("utf-8"))
        assert raw["_content_sha256"]
        assert raw["_resolved_skills"] == {"FakeRead": "1.0.0",
                                           "FakeBias": "1.2.0"}
        assert "_machine" in raw and "_author" in raw

    def test_restore_and_delete(self, client):
        c, reg, _ = client
        c.post("/composites/WF1", json={"spec": SPEC})
        c.post("/composites/WF1", json={"spec": dict(SPEC, description="v2")})
        r = c.post("/composites/WF1/restore", json={"version": 1})
        assert r.status_code == 200 and r.json()["version"] == 3
        assert c.get("/composites/WF1").json()["spec"]["description"] == "demo"
        d = c.request("DELETE", "/composites/WF1")
        assert d.status_code == 200
        assert not reg.has("WF1")                   # 幽灵清除
        assert c.get("/composites/WF1").status_code == 404

    def test_delete_traversal_name_is_404_not_500(self, client):
        """review F2：穿越形名字不可成为文件存在性预言机（不 500、不泄露）。"""
        c, *_ = client
        r = c.request("DELETE", "/composites/..%5C..%5Cfoo")
        assert r.status_code in (400, 404)
        r2 = c.request("DELETE", "/composites/" + "%2e%2e%5cconfig")
        assert r2.status_code in (400, 404)

    def test_save_as_existing_name_409(self, client):
        """review F3：「另存为」语义 = base_version 0，重名必须 409 而非叠版本。"""
        c, *_ = client
        c.post("/composites/WF1", json={"spec": SPEC})
        r = c.post("/composites/WF1", json={"spec": SPEC, "base_version": 0})
        assert r.status_code == 409

    def test_validate_dangling_ref_warning(self, client):
        """review F4：$expr 引用未知名字 → 警告（不拦保存）。"""
        c, *_ = client
        bad = dict(SPEC, nodes=[
            {"type": "step", "id": "a", "skill": "FakeBias",
             "params": {"bias_v": {"$expr": "ghost_node * 2"}}},
        ])
        rep = c.post("/composites/validate", json={"spec": bad}).json()
        assert rep["ok"] is True                     # 警告级
        assert any("ghost_node" in w for w in rep["steps"][0]["warnings"])

    def test_validate_depth_bomb_rejected(self, client):
        """review F6：深嵌套炸弹被显式深度上限拒绝，不打穿递归。"""
        c, *_ = client
        node = {"type": "step", "id": "leaf", "skill": "FakeRead", "params": {}}
        for i in range(80):
            node = {"type": "if", "id": f"if{i}", "cond": "True",
                    "then": [node], "else": []}
        rep = c.post("/composites/validate",
                     json={"spec": dict(SPEC, nodes=[node])}).json()
        assert rep["ok"] is False
        assert any("嵌套" in p for p in rep["problems"])

    def test_diff(self, client):
        c, *_ = client
        c.post("/composites/WF1", json={"spec": SPEC})
        spec2 = dict(SPEC, nodes=SPEC["nodes"] + [
            {"type": "step", "id": "c", "skill": "FakeRead", "params": {}}])
        c.post("/composites/WF1", json={"spec": spec2})
        diff = c.get("/composites/WF1/diff?v1=1&v2=2").json()
        assert diff["added"] == ["c"]


# ── validate ─────────────────────────────────────────────────────────────────

class TestValidate:
    def _post(self, c, spec):
        return c.post("/composites/validate", json={"spec": spec}).json()

    def test_clean_spec_ok(self, client):
        c, *_ = client
        rep = self._post(c, SPEC)
        assert rep["ok"] is True
        assert all(not s["errors"] for s in rep["steps"])

    def test_unknown_skill_and_param(self, client):
        c, *_ = client
        bad = dict(SPEC, nodes=[
            {"type": "step", "id": "a", "skill": "NoSuch", "params": {}},
            {"type": "step", "id": "b", "skill": "FakeBias",
             "params": {"bias_v": 1.0, "oops": 2}},
        ])
        rep = self._post(c, bad)
        by_id = {s["id"]: s for s in rep["steps"]}
        assert any("不存在" in e for e in by_id["a"]["errors"])
        assert any("未知参数" in e for e in by_id["b"]["errors"])
        assert rep["ok"] is False

    def test_out_of_bounds_literal(self, client):
        c, *_ = client
        bad = dict(SPEC, nodes=[{"type": "step", "id": "a", "skill": "FakeBias",
                                 "params": {"bias_v": 99.0}}])
        rep = self._post(c, bad)
        assert rep["ok"] is False
        assert any("99" in e for e in rep["steps"][0]["errors"])

    def test_coarse_z_approach_design_time_error(self, client):
        c, *_ = client
        bad = dict(SPEC, nodes=[{"type": "step", "id": "a", "skill": "MotorMove",
                                 "params": {"direction": "z-approach",
                                            "steps": 5}}])
        rep = self._post(c, bad)
        assert rep["ok"] is False
        assert any("粗逼近" in e for e in rep["steps"][0]["errors"])

    def test_dynamic_direction_rejected(self, client):
        c, *_ = client
        bad = dict(SPEC, nodes=[{"type": "step", "id": "a", "skill": "MotorMove",
                                 "params": {"direction": {"$expr": "d"},
                                            "steps": 5}}])
        rep = self._post(c, bad)
        assert any("不可动态" in e for e in rep["steps"][0]["errors"])

    def test_missing_required_with_expr_is_ok(self, client):
        c, *_ = client
        ok = dict(SPEC, nodes=[{"type": "step", "id": "a", "skill": "FakeBias",
                                "params": {"bias_v": {"$expr": "n*0.1"}}}])
        rep = self._post(c, ok)
        assert rep["steps"][0]["errors"] == []

    def test_nested_steps_walked(self, client):
        c, *_ = client
        nested = dict(SPEC, nodes=[
            {"type": "loop", "id": "l", "mode": "repeat", "count": 2,
             "body": [{"type": "if", "id": "i", "cond": "n > 1",
                       "then": [{"type": "step", "id": "deep",
                                 "skill": "FakeBias",
                                 "params": {"bias_v": 99.0}}],
                       "else": []}]},
        ])
        rep = self._post(c, nested)
        assert any(s["id"] == "deep" and s["errors"] for s in rep["steps"])


# ── favorites ────────────────────────────────────────────────────────────────

class TestGenerate:
    """P2-E builder agent：NL→spec，生成→校验→回喂修复环。"""

    class _FakeGen:
        def __init__(self, replies):
            self.replies = list(replies)
            self.calls = 0
        def invoke(self, msgs):
            self.calls += 1
            class _R: pass
            r = _R(); r.content = self.replies.pop(0)
            return r

    GOOD = ('{"name": "AI生成", "description": "d", "safety_level": "confirm",'
            '"params": [], "nodes": [{"type": "step", "id": "a",'
            '"skill": "FakeRead", "params": {}}]}')
    BAD = GOOD.replace("FakeRead", "NoSuchSkill")

    def test_one_shot_ok(self, client, monkeypatch):
        c, *_ = client
        from mast.webui import builder_api as ba
        fake = self._FakeGen([self.GOOD])
        monkeypatch.setattr(ba, "_gen_model_factory", lambda: fake)
        r = c.post("/builder/generate", json={"prompt": "读一下"})
        assert r.status_code == 200
        body = r.json()
        assert body["spec"]["nodes"][0]["skill"] == "FakeRead"
        assert body["report"]["ok"] is True and body["attempts"] == 1

    def test_repair_loop_fixes_bad_spec(self, client, monkeypatch):
        c, *_ = client
        from mast.webui import builder_api as ba
        fake = self._FakeGen([self.BAD, self.GOOD])   # 第一轮坏 → 回喂修复
        monkeypatch.setattr(ba, "_gen_model_factory", lambda: fake)
        r = c.post("/builder/generate", json={"prompt": "读一下"})
        assert r.status_code == 200
        assert r.json()["attempts"] == 2 and fake.calls == 2

    def test_unfixable_422_with_report(self, client, monkeypatch):
        c, *_ = client
        from mast.webui import builder_api as ba
        fake = self._FakeGen([self.BAD, self.BAD])
        monkeypatch.setattr(ba, "_gen_model_factory", lambda: fake)
        r = c.post("/builder/generate", json={"prompt": "读一下"})
        assert r.status_code == 422
        assert "report" in r.json()

    def test_missing_prompt_400(self, client):
        c, *_ = client
        assert c.post("/builder/generate", json={}).status_code == 400


class TestPersonasEndpoint:
    def test_personas_listed(self, client):
        c, *_ = client
        d = c.get("/builder/personas").json()
        assert any(p["id"] == "instrument_control" for p in d["personas"])


class TestFavorites:
    def test_roundtrip(self, client):
        c, *_ = client
        assert c.get("/builder/favorites").json()["favorites"] == []
        r = c.post("/builder/favorites",
                   json={"favorites": ["FakeBias", "MotorMove"]})
        assert r.json()["favorites"] == ["FakeBias", "MotorMove"]
        assert c.get("/builder/favorites").json()["favorites"] == [
            "FakeBias", "MotorMove"]

    def test_bad_payload_400(self, client):
        c, *_ = client
        assert c.post("/builder/favorites", json={}).status_code == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
