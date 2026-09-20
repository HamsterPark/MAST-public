"""P1 验收链路测试：真实全量 registry → 目录 → 画布产物保存 → 热注册 → agent 工具。

对应 RFC 验收口径的服务端可测部分：浏览器画出的「循环修针直到合格」形态的
spec（set/loop-while/if/step 全节点类型）经 POST 保存后，必须：
  (a) 通过设计期校验；(b) 落盘带 sync-ready 字段；(c) 热注册进 LIVE registry；
  (d) 能被 wrap_skill 包成 agent 工具（IC 工具表的机械前提）；
  (e) 出现在技能目录（palette 立即可见，source=user_composite）。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/gui/test_builder_acceptance.py -x -v
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

import json

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from mast.core.registry import SkillRegistry
from mast.webui import builder_api, composite_panel
from mast.skills.composite.version_store import CompositeVersionStore

# 浏览器画布会产出的「循环修针直到合格」形态（全部 4 种节点类型 + $expr +
# 工作流参数），技能用全量 registry 里真实存在的无参/简单技能。
ACCEPT_SPEC = {
    "name": "验收_循环修针",
    "description": "acceptance: loop until quality ok",
    "safety_level": "confirm",
    "params": [{"name": "max_tries", "type": "int", "default": 3,
                "description": "", "required": False}],
    "nodes": [
        {"type": "set", "id": "init", "var": "attempts", "value": "0"},
        {"type": "loop", "id": "repair", "mode": "while",
         "cond": "attempts < max_tries", "max_iter": 10, "body": [
            {"type": "step", "id": "pulse", "skill": "TipPulse",
             "params": {"pulse_v": {"$expr": "min(3.0, 1.0 + attempts)"}}},
            {"type": "set", "id": "inc", "var": "attempts",
             "value": "attempts + 1"},
         ]},
        {"type": "if", "id": "verdict", "cond": "attempts >= 1",
         "then": [{"type": "step", "id": "readback", "skill": "GetBias",
                   "params": {}}],
         "else": []},
    ],
}


@pytest.fixture(scope="module")
def full_registry():
    reg = SkillRegistry()
    n = reg.discover()
    assert n > 150, f"full discover too small: {n}"
    return reg


@pytest.fixture()
def client(tmp_path, monkeypatch, full_registry):
    store = CompositeVersionStore(root=tmp_path / "cs")
    monkeypatch.setattr(builder_api, "_registry", full_registry)
    monkeypatch.setattr(builder_api, "_catalog_cache", None)
    monkeypatch.setattr(builder_api, "_favorites_path",
                        lambda: tmp_path / "fav.json")
    monkeypatch.setattr(composite_panel, "_store", store)
    monkeypatch.setattr(composite_panel, "_live_registry", full_registry)
    monkeypatch.setattr(composite_panel, "_agent_refresh", None)
    app = Starlette(routes=builder_api.build_routes())
    app.auth = None
    app.auth_dependency = None
    before = {m.name for m in full_registry.list_skills()}
    yield TestClient(app), full_registry, store
    # 模块级 registry 复用 — 清掉本测试期间新注册的一切（防跨测污染）
    for m in list(full_registry.list_skills()):
        if m.name not in before:
            full_registry.unregister(m.name)


def test_catalog_serves_full_registry(client):
    c, reg, _ = client
    d = c.get("/skills/catalog").json()
    assert d["total"] >= 200                       # 211 builtins + composites…
    names = {e["name"] for e in d["skills"]}
    assert {"SetBias", "MotorMove", "TipPulse", "FullScan"} <= names
    # 域映射 + 兜底桶：每个条目都有 domain，不存在不可见技能
    assert all(e["domain"] for e in d["skills"])
    card = c.get("/skills/catalog/MotorMove").json()
    dirp = next(p for p in card["parameters"] if p["name"] == "direction")
    assert "z-approach" in dirp["allowed_values"]


def test_acceptance_workflow_roundtrip(client):
    c, reg, store = client
    # 1) 校验通过
    rep = c.post("/composites/validate", json={"spec": ACCEPT_SPEC}).json()
    assert rep["ok"] is True, rep
    # 2) 保存 → 热注册
    r = c.post(f"/composites/{ACCEPT_SPEC['name']}",
               json={"spec": ACCEPT_SPEC, "base_version": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hot_registered"] is True and body["version"] == 1
    # 3) sync-ready 落盘
    raw = json.loads((store._root / f"{ACCEPT_SPEC['name']}.json")
                     .read_text("utf-8"))
    assert raw["_content_sha256"] and "TipPulse" in raw["_resolved_skills"]
    # 4) LIVE registry 可解析为可实例化技能
    assert reg.has(ACCEPT_SPEC["name"])
    cls = reg.get(ACCEPT_SPEC["name"])
    meta = cls().metadata()
    assert meta.name == ACCEPT_SPEC["name"]
    assert meta.parameters[0].name == "max_tries"
    # 5) wrap_skill → agent 工具（IC 工具表的机械前提）
    from mast.agents._shared.skill_adapter import wrap_skill
    tool = wrap_skill(cls, lambda: None)
    assert tool.name == ACCEPT_SPEC["name"]
    assert tool.metadata["skill_metadata"].name == ACCEPT_SPEC["name"]
    # 6) palette 立即可见（保存已失效目录缓存）
    d = c.get("/skills/catalog?source=user_composite").json()
    assert ACCEPT_SPEC["name"] in {e["name"] for e in d["skills"]}


def test_acceptance_spec_rejects_zapproach_smuggling(client):
    """画布若被手改塞入 z-approach 步骤，保存必须 422（设计期+运行期双拒）。"""
    c, *_ = client
    bad = json.loads(json.dumps(ACCEPT_SPEC))
    bad["name"] = "验收_走私粗逼近"
    bad["nodes"].append({"type": "step", "id": "smuggle", "skill": "MotorMove",
                         "params": {"direction": "z-approach", "steps": 3}})
    r = c.post(f"/composites/{bad['name']}", json={"spec": bad})
    assert r.status_code == 422
    assert any("粗逼近" in e for s in r.json()["steps"] for e in s["errors"])


# NOTE: test_builder_static_assets_present + test_built_js_in_sync_with_jsx were
# REMOVED — they guarded the old Gradio builder's vendored static JS
# (mast/gui/static/builder/builder-ui.{js,jsx} + reactflow vendor), which the TS
# rewrite deleted. The builder is now the React BuilderPage (frontend/src/pages/
# BuilderPage.tsx) backed by /api/composites + /api/builder/* (routes/builder.py).
# The backend-acceptance tests above (catalog/roundtrip/validation) still apply.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
