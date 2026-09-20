"""P3-B：agent 节点（受预算约束的域 agent 委托）。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_agent_node.py -x -v
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

from mast.skills.composite import agent_node
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
        return r(params) if callable(r) else (r or _Res())


class FakeGraph:
    def __init__(self, reply="检索完成：3 条要点", raise_exc=None,
                 state_messages=None):
        self.reply = reply
        self.raise_exc = raise_exc
        self.state_messages = state_messages
        self.invoked_with = None
    def invoke(self, payload, cfg):
        self.invoked_with = (payload, cfg)
        if self.raise_exc:
            raise self.raise_exc
        return {"messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": self.reply}]}
    def get_state(self, cfg):
        class _S: pass
        s = _S()
        s.values = {"messages": self.state_messages or []}
        return s


AGENT_NODE = {
    "type": "agent", "id": "lit", "agent": "literature",
    "task": "检索 {material} 的扫描参数",
    "inputs": {"material": {"$expr": "params['material']"}},
    "max_model_calls": 4,
    "on_error": [{"type": "step", "id": "fallback", "skill": "CallHuman",
                  "params": {}}],
}


def _spec(node=None):
    return CompositeSpec(
        name="AgentDemo", safety_level="confirm",
        params=[{"name": "material", "type": "str", "default": "Si(111)",
                 "description": "", "required": False}] and [],
        nodes=[json.loads(json.dumps(node or AGENT_NODE)),
               {"type": "step", "id": "use", "skill": "Use",
                "params": {"summary": {"$expr": "lit['text']"}}}])


def _run(spec, ctx, params=None):
    from mast.skills.composite.interpreter import SpecComposite
    return SpecComposite(spec).execute(ctx, params or {})


@pytest.fixture(autouse=True)
def _tmp_decision_log(tmp_path, monkeypatch):
    from mast.skills.composite import llm_node
    monkeypatch.setattr(llm_node, "decision_log_path",
                        lambda: tmp_path / "dlog.jsonl")
    return tmp_path / "dlog.jsonl"


class TestValidation:
    def test_good(self):
        assert _spec().validate() == []

    def test_instrument_control_banned(self):
        n = json.loads(json.dumps(AGENT_NODE))
        n["agent"] = "instrument_control"
        probs = _spec(n).validate()
        assert any("最小能动性" in p for p in probs)

    def test_unknown_agent_rejected(self):
        n = json.loads(json.dumps(AGENT_NODE))
        n["agent"] = "skynet"
        assert any("invalid agent" in p for p in _spec(n).validate())

    def test_missing_task(self):
        n = json.loads(json.dumps(AGENT_NODE))
        n["task"] = " "
        assert any("task" in p for p in _spec(n).validate())


class TestExecution:
    def test_delegation_binds_text_downstream(self, monkeypatch):
        g = FakeGraph(reply="要点A；要点B")
        monkeypatch.setattr(agent_node, "_build_agent_graph",
                            lambda aid, **kw: g)
        ctx = FakeCtx()
        res = _run(_spec(), ctx, {"material": "Au(111)"})
        assert res.success, res.error
        use = [c for c in ctx.calls if c[0] == "Use"]
        assert use and use[0][1]["summary"] == "要点A；要点B"
        # task 模板插值 + no-handoff 约束注入
        sent = g.invoked_with[0]["messages"][0]["content"]
        assert "Au(111)" in sent and "不要调用任何 handoff" in sent
        assert "CallHuman" not in [c[0] for c in ctx.calls]

    def test_budget_caps_applied(self, monkeypatch):
        seen = {}
        def fake_build(aid, *, max_model_calls, max_tool_calls):
            seen.update(aid=aid, mmc=max_model_calls, mtc=max_tool_calls)
            return FakeGraph()
        monkeypatch.setattr(agent_node, "_build_agent_graph", fake_build)
        n = json.loads(json.dumps(AGENT_NODE))
        n["max_model_calls"] = 999     # 超过硬上限 → 钳到 12
        _run(_spec(n), FakeCtx(), {"material": "x"})
        assert seen["mmc"] == agent_node.MAX_MODEL_CALLS_CAP

    def test_failure_walks_on_error(self, monkeypatch):
        monkeypatch.setattr(agent_node, "_build_agent_graph",
                            lambda aid, **kw: FakeGraph(
                                raise_exc=RuntimeError("budget exceeded")))
        ctx = FakeCtx()
        res = _run(_spec(), ctx, {"material": "x"})
        assert res.success                      # 工作流不崩
        assert "CallHuman" in [c[0] for c in ctx.calls]

    def test_handoff_crash_salvages_state_text(self, monkeypatch):
        g = FakeGraph(raise_exc=RuntimeError("ParentCommand"),
                      state_messages=[
                          {"role": "assistant", "content": "其实答案是 42"}])
        monkeypatch.setattr(agent_node, "_build_agent_graph",
                            lambda aid, **kw: g)
        ctx = FakeCtx()
        _run(_spec(), ctx, {"material": "x"})
        use = [c for c in ctx.calls if c[0] == "Use"]
        assert use and use[0][1]["summary"] == "其实答案是 42"

    def test_cached_delegation_not_rerun(self, monkeypatch):
        calls = {"n": 0}
        def fake_build(aid, **kw):
            calls["n"] += 1
            return FakeGraph()
        monkeypatch.setattr(agent_node, "_build_agent_graph", fake_build)
        from mast.skills.composite.graph_executor import GraphExecutor
        from mast.skills.composite.interpreter import SpecComposite
        spec = _spec()
        ctx = FakeCtx()
        ex = GraphExecutor(spec.name, ctx)
        ex.progress.partial_data["_agents"] = {
            "lit": {"ok": True, "text": "缓存的答案", "note": ""}}
        ex.run_plan(SpecComposite(spec).plan_dynamic({"material": "x"}, ex))
        assert calls["n"] == 0                  # 委托绝不重跑
        use = [c for c in ctx.calls if c[0] == "Use"]
        assert use and use[0][1]["summary"] == "缓存的答案"

    def test_decision_logged(self, monkeypatch, _tmp_decision_log):
        monkeypatch.setattr(agent_node, "_build_agent_graph",
                            lambda aid, **kw: FakeGraph())
        _run(_spec(), FakeCtx(), {"material": "x"})
        rec = json.loads(
            _tmp_decision_log.read_text("utf-8").strip().splitlines()[-1])
        assert rec["mechanism"] == "agent" and rec["agent"] == "literature"
        assert rec["ok"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
