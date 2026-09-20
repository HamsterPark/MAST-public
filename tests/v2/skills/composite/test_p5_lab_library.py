"""P5 实验室库：manifest-only 上传端点 / 客户端分享 / custom 显式白名单加载。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_p5_lab_library.py -x -v
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

MANIFEST = {"name": "巡查环", "version": 3, "description": "demo",
            "safety_level": "confirm", "params": [], "nodes": [],
            "_author": "yh", "_machine": "lab-pc", "_content_sha256": "ab" * 32}


# ── push server 端（FastAPI TestClient） ─────────────────────────────────────

@pytest.fixture()
def server(tmp_path):
    from fastapi.testclient import TestClient

    from mast.update.server import build_app
    # server token（server 从 data_root/"api key"/update_server_token.env 读）
    (tmp_path / "api key").mkdir()
    (tmp_path / "api key" / "update_server_token.env").write_text(
        "tok123\n", encoding="utf-8")
    app = build_app(tmp_path)
    return TestClient(app), tmp_path


def _auth():
    return {"Authorization": "Bearer tok123"}


class TestUploadEndpoint:
    def test_upload_and_index(self, server):
        c, root = server
        r = c.post("/skills/upload", headers=_auth(),
                   json={"manifest": MANIFEST, "client_version": "2.6.2"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] and body["status"] == "pending_review"
        # inbox 落盘 + 索引
        inbox = root / "push_distribution" / "skill_inbox"
        assert (inbox / f"{body['id']}.json").exists()
        idx = c.get("/skills/index", headers=_auth()).json()
        assert idx["skills"][0]["name"] == "巡查环"
        assert idx["skills"][0]["content_sha256"] == "ab" * 32
        assert idx["skills"][0]["author"] == "yh"

    def test_code_shaped_payload_rejected(self, server):
        """红线：nodes 必须是 list——任何代码形态的载荷 400。"""
        c, _ = server
        bad = dict(MANIFEST)
        bad["nodes"] = "import os; os.system('rm -rf /')"
        r = c.post("/skills/upload", headers=_auth(), json={"manifest": bad})
        assert r.status_code == 400
        assert "NOT accepted" in r.json()["detail"]

    def test_auth_required(self, server):
        c, _ = server
        assert c.post("/skills/upload",
                      json={"manifest": MANIFEST}).status_code == 401

    def test_traversal_name_rejected(self, server):
        c, _ = server
        bad = dict(MANIFEST); bad["name"] = "../evil"
        assert c.post("/skills/upload", headers=_auth(),
                      json={"manifest": bad}).status_code == 400

    def test_oversize_rejected(self, server):
        c, _ = server
        big = dict(MANIFEST); big["notes"] = "x" * 600_000
        assert c.post("/skills/upload", headers=_auth(),
                      json={"manifest": big}).status_code == 413


# ── 客户端分享 ───────────────────────────────────────────────────────────────

class TestShareClient:
    def test_share_requires_config(self, tmp_path, monkeypatch):
        from mast.webui import builder_api, composite_panel
        from mast.skills.composite.spec import CompositeSpec
        from mast.skills.composite.version_store import CompositeVersionStore
        store = CompositeVersionStore(root=tmp_path / "cs")
        store.save(CompositeSpec(name="WF", safety_level="confirm", nodes=[]))
        monkeypatch.setattr(composite_panel, "_store", store)
        import mast.update.client as uc
        monkeypatch.setattr(uc, "read_server_url", lambda root: "")
        monkeypatch.setattr(uc, "_read_token", lambda root: "")
        out = builder_api.share_to_lab_sync("WF")
        assert "未配置" in out["error"]

    def test_share_posts_raw_manifest(self, tmp_path, monkeypatch):
        from mast.webui import builder_api, composite_panel
        from mast.skills.composite.spec import CompositeSpec
        from mast.skills.composite.version_store import CompositeVersionStore
        store = CompositeVersionStore(root=tmp_path / "cs")
        store.save(CompositeSpec(name="WF", safety_level="confirm", nodes=[]),
                   extra_meta={"_resolved_skills": {}})
        monkeypatch.setattr(composite_panel, "_store", store)
        import mast.update.client as uc
        monkeypatch.setattr(uc, "read_server_url",
                            lambda root: "http://<lan-host>:8766")
        monkeypatch.setattr(uc, "_read_token", lambda root: "tok")
        monkeypatch.setattr(uc, "_require_https", lambda url: "")
        sent = {}

        class _Resp:
            status_code = 200
            def json(self):
                return {"ok": True, "id": "sk-1", "status": "pending_review"}
        import httpx
        def fake_post(url, headers=None, json=None, timeout=None):
            sent.update(url=url, headers=headers, body=json)
            return _Resp()
        monkeypatch.setattr(httpx, "post", fake_post)
        out = builder_api.share_to_lab_sync("WF")
        assert out["ok"] and out["id"] == "sk-1"
        assert sent["url"].endswith("/skills/upload")
        m = sent["body"]["manifest"]
        assert m["name"] == "WF" and "_content_sha256" in m  # sync-ready 字段随行
        assert "nodes" in m and isinstance(m["nodes"], list)

    def test_unknown_workflow(self, tmp_path, monkeypatch):
        from mast.webui import builder_api, composite_panel
        from mast.skills.composite.version_store import CompositeVersionStore
        monkeypatch.setattr(composite_panel, "_store",
                            CompositeVersionStore(root=tmp_path / "cs"))
        assert "error" in builder_api.share_to_lab_sync("Nope")


# ── custom 显式白名单加载 ────────────────────────────────────────────────────

GOOD_SKILL = '''
from mast.core.types import SkillCategory, SkillMetadata, SkillResult
from mast.skills.base import BaseSkill

class MyCustom(BaseSkill):
    def metadata(self):
        return SkillMetadata(name="MyCustom", version="1.0.0",
                             category=SkillCategory.ANALYSIS, description="d")
    def execute(self, ctx, params):
        return SkillResult(skill_name="MyCustom", success=True, data={"v": 7})
'''

EVIL_SKILL = '''
import os
from mast.skills.base import BaseSkill
'''


class TestCustomLoader:
    def _wire(self, tmp_path, monkeypatch):
        from mast.skills import custom_loader
        d = tmp_path / "custom_skills"
        d.mkdir()
        monkeypatch.setattr(custom_loader, "custom_skills_dir", lambda: d)
        return custom_loader, d

    def test_enabled_skill_loads(self, tmp_path, monkeypatch):
        from mast.core.registry import SkillRegistry
        cl, d = self._wire(tmp_path, monkeypatch)
        (d / "MyCustom.py").write_text(GOOD_SKILL, encoding="utf-8")
        (d / "enabled.json").write_text(json.dumps({"enabled": ["MyCustom"]}),
                                        encoding="utf-8")
        reg = SkillRegistry()
        assert cl.load_custom_skills(reg) == ["MyCustom"]
        assert reg.has("MyCustom")
        res = reg.get("MyCustom")().execute(None, {})
        assert res.data["v"] == 7

    def test_dropped_file_without_enable_never_executes(self, tmp_path,
                                                        monkeypatch):
        """拷文件 ≠ 启用：enabled.json 不列 → 永不 import。"""
        from mast.core.registry import SkillRegistry
        cl, d = self._wire(tmp_path, monkeypatch)
        (d / "MyCustom.py").write_text(GOOD_SKILL, encoding="utf-8")
        reg = SkillRegistry()
        assert cl.load_custom_skills(reg) == []
        assert not reg.has("MyCustom")

    def test_ast_violation_blocked(self, tmp_path, monkeypatch):
        from mast.core.registry import SkillRegistry
        cl, d = self._wire(tmp_path, monkeypatch)
        (d / "Evil.py").write_text(EVIL_SKILL, encoding="utf-8")
        (d / "enabled.json").write_text(json.dumps({"enabled": ["Evil"]}),
                                        encoding="utf-8")
        reg = SkillRegistry()
        assert cl.load_custom_skills(reg) == []   # import os → deny-list 拦

    def test_skill_author_writes_to_data_root(self, monkeypatch, tmp_path):
        """P5-A 迁移守卫：skill_author 目标在数据根而非冻结包内。"""
        import mast.llm.skill_author as sa
        assert "config" in str(sa._CUSTOM_SKILLS_DIR)
        assert "custom_skills" in str(sa._CUSTOM_SKILLS_DIR)
        pkg = Path(_MASTV2_ROOT) / "mast"
        assert not str(sa._CUSTOM_SKILLS_DIR).startswith(str(pkg))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
