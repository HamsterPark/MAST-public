"""P4 技能化闭环：outputs 签名 / step 版本钉死 / effective safety 硬化 / 决策日志端点。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_p4_skillization.py -x -v
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

from mast.core.execution_context import ExecutionContext
from mast.core.registry import SkillRegistry
from mast.core.types import (
    ParameterSpec, SafetyLevel, SkillCategory, SkillMetadata, SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.composite.spec import CompositeSpec


def _mk_skill(name, version, marker, safety=SafetyLevel.AUTO):
    class _S(BaseSkill):
        def metadata(self):
            return SkillMetadata(name=name, version=version,
                                 category=SkillCategory.READ,
                                 safety_level=safety, description="",
                                 parameters=[])
        def execute(self, ctx, params):
            return SkillResult(skill_name=name, success=True,
                               data={"marker": marker})
    _S.__name__ = f"{name}_v{version.replace('.', '_')}"
    return _S


def _run(spec, registry, params=None):
    from mast.skills.composite.interpreter import SpecComposite
    ectx = ExecutionContext(pool=None, state=None, registry=registry)
    return SpecComposite(spec).execute(ectx, params or {})


# ── outputs 签名 ─────────────────────────────────────────────────────────────

class TestOutputs:
    def test_outputs_evaluated_into_result(self):
        reg = SkillRegistry()
        reg.register(_mk_skill("Probe", "1.0.0", 42))
        spec = CompositeSpec(
            name="OutDemo", safety_level="auto",
            params=[],
            nodes=[{"type": "step", "id": "p", "skill": "Probe", "params": {}},
                   {"type": "set", "id": "s", "var": "doubled",
                    "value": "p['marker'] * 2"}],
            outputs=[{"name": "marker", "expr": "p['marker']"},
                     {"name": "doubled", "expr": "doubled"}],
        )
        assert spec.validate() == []
        res = _run(spec, reg)
        assert res.success, res.error
        assert res.data["outputs"] == {"marker": 42, "doubled": 84}

    def test_bad_output_expr_recorded_not_fatal(self):
        reg = SkillRegistry()
        reg.register(_mk_skill("Probe", "1.0.0", 1))
        spec = CompositeSpec(
            name="OutDemo2", safety_level="auto",
            nodes=[{"type": "step", "id": "p", "skill": "Probe", "params": {}}],
            outputs=[{"name": "ghost", "expr": "nonexistent_var"}],
        )
        res = _run(spec, reg)
        assert res.success
        assert res.data["outputs"]["ghost"] is None
        assert "ghost" in res.data["output_errors"]

    def test_outputs_validation(self):
        bad = CompositeSpec(name="X", nodes=[],
                            outputs=[{"name": "bad name!", "expr": "1"},
                                     {"name": "a", "expr": "(("}])
        probs = bad.validate()
        assert any("invalid output name" in p for p in probs)
        assert any("syntax error" in p for p in probs)


# ── step 版本钉死 ────────────────────────────────────────────────────────────

class TestVersionPin:
    def _reg(self):
        reg = SkillRegistry()
        reg.register(_mk_skill("Dual", "1.0.0", "old"))
        reg.register(_mk_skill("Dual", "2.0.0", "new"))
        return reg

    def test_unpinned_takes_latest(self):
        spec = CompositeSpec(
            name="PinDemo", safety_level="auto",
            nodes=[{"type": "step", "id": "d", "skill": "Dual", "params": {}}],
            outputs=[{"name": "m", "expr": "d['marker']"}])
        res = _run(spec, self._reg())
        assert res.data["outputs"]["m"] == "new"

    def test_pinned_takes_exact_version(self):
        spec = CompositeSpec(
            name="PinDemo2", safety_level="auto",
            nodes=[{"type": "step", "id": "d", "skill": "Dual",
                    "params": {}, "skill_version": "1.0.0"}],
            outputs=[{"name": "m", "expr": "d['marker']"}])
        res = _run(spec, self._reg())
        assert res.success, res.error
        assert res.data["outputs"]["m"] == "old"     # 升级不再漂移

    def test_pinned_missing_version_fails_closed(self):
        spec = CompositeSpec(
            name="PinDemo3", safety_level="auto",
            nodes=[{"type": "step", "id": "d", "skill": "Dual",
                    "params": {}, "skill_version": "9.9.9"}])
        res = _run(spec, self._reg())
        assert res.success is False
        assert "9.9.9" in res.error


# ── builder 校验硬化 ─────────────────────────────────────────────────────────

class TestBuilderValidation:
    def _wire(self, monkeypatch, reg):
        from mast.webui import builder_api
        monkeypatch.setattr(builder_api, "_registry", reg)
        return builder_api

    def test_effective_safety_hard_error(self, monkeypatch):
        reg = SkillRegistry()
        reg.register(_mk_skill("Danger", "1.0.0", 0,
                               safety=SafetyLevel.DANGEROUS))
        ba = self._wire(monkeypatch, reg)
        rep = ba.validate_spec_payload({
            "name": "W", "safety_level": "confirm", "params": [],
            "nodes": [{"type": "step", "id": "d", "skill": "Danger",
                       "params": {}}]})
        assert rep["ok"] is False                    # 不再是警告
        assert any("effective level" in p for p in rep["problems"])

    def test_pinned_version_lint(self, monkeypatch):
        reg = SkillRegistry()
        reg.register(_mk_skill("Solo", "1.0.0", 0))
        ba = self._wire(monkeypatch, reg)
        rep = ba.validate_spec_payload({
            "name": "W", "safety_level": "confirm", "params": [],
            "nodes": [{"type": "step", "id": "s", "skill": "Solo",
                       "params": {}, "skill_version": "3.0.0"}]})
        assert any("钉住的版本" in e
                   for s in rep["steps"] for e in s["errors"])


# ── 决策日志端点 ─────────────────────────────────────────────────────────────

def test_decisions_endpoint(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from mast.webui import builder_api
    from mast.skills.composite import llm_node
    log = tmp_path / "dlog.jsonl"
    recs = [
        {"workflow": "A", "mechanism": "llm", "route": "ok"},
        {"workflow": "B", "mechanism": "agent", "ok": True},
        {"workflow": "A", "mechanism": "llm", "route": "escalate"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    monkeypatch.setattr(llm_node, "decision_log_path", lambda: log)
    monkeypatch.setattr(builder_api, "_registry", SkillRegistry())
    app = Starlette(routes=builder_api.build_routes())
    app.auth = None; app.auth_dependency = None
    c = TestClient(app)
    d = c.get("/builder/decisions").json()
    assert d["total_returned"] == 3
    assert d["decisions"][0]["route"] == "escalate"   # 倒序最近
    d2 = c.get("/builder/decisions?workflow=A").json()
    assert d2["total_returned"] == 2
    d3 = c.get("/builder/decisions?mechanism=agent").json()
    assert d3["total_returned"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
