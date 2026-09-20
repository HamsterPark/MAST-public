"""P2-G：human 节点 + 带外步进度 sidecar + GraphInterrupt 控制流豁免。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_human_node_sidecar.py -x -v
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
from langgraph.errors import GraphInterrupt

from mast.skills.composite import graph_executor as ge_mod
from mast.skills.composite import interpreter as interp_mod
from mast.skills.composite.graph_executor import GraphExecutor
from mast.skills.composite.spec import CompositeSpec


class _Res:
    def __init__(self, success=True, data=None, error=""):
        self.success = success
        self.data = data or {}
        self.error = error
        self.nanonis_calls = []


class FakeCtx:
    def __init__(self, results=None):
        self.calls = []
        self._results = results or {}
    def run(self, skill_name, params):
        self.calls.append((skill_name, dict(params)))
        r = self._results.get(skill_name)
        if callable(r):
            return r(params)
        return r or _Res()


@pytest.fixture(autouse=True)
def _tmp_sidecar_dir(tmp_path, monkeypatch):
    d = tmp_path / "composite_progress"
    d.mkdir()
    import re as _re
    # run_id was added to the key (sidecars are scoped to ONE run, 2026-07-11);
    # these specs run with no run_id, i.e. the legacy name-only file.
    def _path(name, run_id=""):
        safe = _re.sub(r"[^\w一-鿿-]", "_", str(name))[:80] or "composite"
        if run_id:
            return d / f"{safe}__{run_id}.json"
        return d / f"{safe}.json"
    monkeypatch.setattr(ge_mod, "_sidecar_path", _path)
    monkeypatch.setattr(ge_mod, "_sidecar_dir", lambda: d)
    return d


HUMAN_SPEC_NODES = [
    {"type": "step", "id": "scan", "skill": "Scan", "params": {}},
    {"type": "human", "id": "ask", "message": "质量 {q}，继续吗？",
     "inputs": {"q": {"$expr": "scan['quality']"}},
     "routes": {"go": [{"type": "step", "id": "save", "skill": "Save",
                        "params": {}}],
                "stop": [{"type": "step", "id": "park", "skill": "Park",
                          "params": {}}]}},
]


def _spec(nodes=None, name="HumanDemo"):
    return CompositeSpec(name=name, safety_level="confirm", params=[],
                         nodes=nodes or json.loads(json.dumps(HUMAN_SPEC_NODES)))


def _run(spec, ctx):
    from mast.skills.composite.interpreter import SpecComposite
    return SpecComposite(spec).execute(ctx, {})


# ── 校验 ─────────────────────────────────────────────────────────────────────

def test_human_validation():
    assert _spec().validate() == []
    bad = _spec()
    bad.nodes[1].pop("message")
    assert any("message" in p for p in bad.validate())
    bad2 = _spec()
    bad2.nodes[1]["routes"] = {}
    assert any("routes" in p for p in bad2.validate())


# ── human 通道 ───────────────────────────────────────────────────────────────

def test_operator_decision_walks_chosen_route(monkeypatch):
    seen = {}
    def fake_channel(payload):
        seen.update(payload)
        return {"route": "go", "note": "看着不错"}
    monkeypatch.setattr(interp_mod, "human_channel", fake_channel)
    ctx = FakeCtx({"Scan": _Res(data={"quality": 0.8})})
    res = _run(_spec(), ctx)
    assert res.success, res.error
    called = [c[0] for c in ctx.calls]
    assert "Save" in called and "Park" not in called
    assert seen["routes"] == ["go", "stop"]
    assert "0.8" in seen["message"]          # 模板插值

def test_no_channel_fails_loudly(monkeypatch):
    monkeypatch.setattr(interp_mod, "human_channel", lambda payload: None)
    ctx = FakeCtx({"Scan": _Res(data={"quality": 0.8})})
    with pytest.raises(RuntimeError, match="HITL"):
        _run(_spec(), ctx)

def test_off_route_decision_rejected(monkeypatch):
    monkeypatch.setattr(interp_mod, "human_channel",
                        lambda payload: {"route": "yolo"})
    ctx = FakeCtx({"Scan": _Res(data={"quality": 0.8})})
    with pytest.raises(RuntimeError, match="选项"):
        _run(_spec(), ctx)

def test_cached_decision_skips_channel(monkeypatch):
    """重放/恢复时绝不重复打扰用户。"""
    calls = {"n": 0}
    def fake_channel(payload):
        calls["n"] += 1
        return "go"
    monkeypatch.setattr(interp_mod, "human_channel", fake_channel)
    from mast.skills.composite.interpreter import SpecComposite
    spec = _spec()
    ctx = FakeCtx({"Scan": _Res(data={"quality": 0.8})})
    ex = GraphExecutor(spec.name, ctx)
    ex.progress.partial_data["_human"] = {"ask": {"route": "stop", "note": ""}}
    ex.run_plan(SpecComposite(spec).plan_dynamic({}, ex))
    assert calls["n"] == 0
    assert "Park" in [c[0] for c in ctx.calls]


# ── sidecar ──────────────────────────────────────────────────────────────────

def test_sidecar_written_per_step_and_cleared_on_success(_tmp_sidecar_dir,
                                                         monkeypatch):
    monkeypatch.setattr(interp_mod, "human_channel", lambda p: "go")
    side = _tmp_sidecar_dir / "HumanDemo.json"
    flushed = {}
    real_flush = GraphExecutor.flush_sidecar
    def spy(self):
        real_flush(self)
        if side.exists():
            flushed["content"] = json.loads(side.read_text("utf-8"))
    monkeypatch.setattr(GraphExecutor, "flush_sidecar", spy)
    ctx = FakeCtx({"Scan": _Res(data={"quality": 0.8})})
    res = _run(_spec(), ctx)
    assert res.success
    # 运行期间确实写过（含已完成步骤），成功后清掉
    assert flushed["content"]["completed_steps"]
    assert not side.exists()

def test_sidecar_resume_beats_stale_state(_tmp_sidecar_dir, monkeypatch):
    """interrupt 重放场景：state 里没有进度，sidecar 有 → 跳过已完成步骤
    并复用 llm/human 决策缓存（绝不重复执行仪器动作/重新问 LLM）。"""
    monkeypatch.setattr(interp_mod, "human_channel",
                        lambda p: pytest.fail("不应再问人"))
    side = _tmp_sidecar_dir / "HumanDemo.json"
    side.write_text(json.dumps({
        "composite_name": "HumanDemo",
        "completed_steps": ["scan"],
        "partial_data": {"_human": {"ask": {"route": "go", "note": ""}}},
    }), encoding="utf-8")
    ctx = FakeCtx({"Scan": _Res(data={"quality": 0.8})})
    res = _run(_spec(), ctx)
    assert res.success
    called = [c[0] for c in ctx.calls]
    assert "Scan" not in called          # 已完成步骤不重放到硬件
    assert "Save" in called              # 缓存决议直接走 go 分支
    assert not side.exists()             # 本次成功后清除


# ── GraphInterrupt 控制流豁免 ────────────────────────────────────────────────

def test_graph_interrupt_bubbles_through_ctx_run():
    from mast.core.registry import SkillRegistry
    from mast.core.execution_context import ExecutionContext
    from mast.core.types import SkillCategory, SkillMetadata
    from mast.skills.base import BaseSkill

    class Interrupting(BaseSkill):
        def metadata(self):
            return SkillMetadata(name="Interrupting", version="1.0.0",
                                 category=SkillCategory.READ, description="")
        def execute(self, ctx, params):
            raise GraphInterrupt(())
    reg = SkillRegistry(); reg.register(Interrupting)
    ectx = ExecutionContext(pool=None, state=None, registry=reg)
    with pytest.raises(GraphInterrupt):
        ectx.run("Interrupting", {})

def test_graph_interrupt_bubbles_through_run_plan():
    def boom(params):
        raise GraphInterrupt(())
    ctx = FakeCtx({"Boom": boom})
    ex = GraphExecutor("X", ctx)
    from mast.skills.composite.graph_executor import CompositeStep
    with pytest.raises(GraphInterrupt):
        ex.run_plan(iter([CompositeStep(step_id="b", skill_name="Boom",
                                        params={})]))

def test_graph_interrupt_bubbles_through_wrap_skill_without_rollback():
    from mast.agents._shared.skill_adapter import wrap_skill
    from mast.core.types import SkillCategory, SkillMetadata
    from mast.skills.base import BaseSkill

    rolled = {"n": 0}

    class Interrupting(BaseSkill):
        def metadata(self):
            return SkillMetadata(name="Interrupting", version="1.0.0",
                                 category=SkillCategory.READ, description="")
        def execute(self, ctx, params):
            raise GraphInterrupt(())
        def rollback(self, ctx, params):
            rolled["n"] += 1
    tool = wrap_skill(Interrupting, lambda: FakeCtx())
    with pytest.raises(GraphInterrupt):
        tool.func(tool_call_id="t1", state={})
    assert rolled["n"] == 0     # 暂停绝不触发硬件回滚


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
