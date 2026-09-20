"""Workflow `llm` node (P2-B): validation / route / data / escape / resume / log.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_llm_node.py -x -v
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

from mast.skills.composite import llm_node
from mast.skills.composite.spec import CompositeSpec


# ── fakes ────────────────────────────────────────────────────────────────────

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


class FakeStructuredModel:
    """with_structured_output(...).invoke(...) → canned dict; counts calls."""
    model_name = "fake/structured"
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0
    def with_structured_output(self, schema, method=None):
        outer = self
        class _Runner:
            def invoke(self, msgs):
                outer.calls += 1
                outer.last_messages = msgs
                return dict(outer.decision)
        return _Runner()
    def invoke(self, msgs):  # text tier — should not be reached in these tests
        raise AssertionError("text tier should not run when structured works")


class FakeTextModel:
    """structured tier fails; text tier returns canned content."""
    model_name = "fake/text"
    def __init__(self, text):
        self.text = text
    def with_structured_output(self, schema, method=None):
        class _Boom:
            def invoke(self, msgs):
                raise RuntimeError("response_format unavailable")
        return _Boom()
    def invoke(self, msgs):
        class _R:  # AIMessage-ish
            pass
        r = _R(); r.content = self.text
        return r


class ExplodingModel:
    model_name = "fake/exploding"
    def with_structured_output(self, schema, method=None):
        raise RuntimeError("network down")
    def invoke(self, msgs):
        raise RuntimeError("network down")


ROUTE_NODE = {
    "type": "llm", "id": "router", "mode": "route",
    "responsibility": "判断针尖是否需要修复",
    "inputs": {"quality": {"$expr": "q['quality']"}},
    "routes": {
        "ok": [{"type": "step", "id": "save", "skill": "SaveIt", "params": {}}],
        "fix": [{"type": "step", "id": "pulse", "skill": "PulseIt", "params": {}}],
        "escalate": [{"type": "step", "id": "call", "skill": "CallHuman",
                      "params": {}}],
    },
    "escape": "escalate",
}


def _spec(extra_nodes=None):
    return CompositeSpec(
        name="LlmDemo", safety_level="confirm", params=[],
        nodes=[{"type": "step", "id": "q", "skill": "Assess", "params": {}}]
              + (extra_nodes if extra_nodes is not None
                 else [json.loads(json.dumps(ROUTE_NODE))]),
    )


def _run(spec, ctx, params=None):
    from mast.skills.composite.interpreter import SpecComposite
    return SpecComposite(spec).execute(ctx, params or {})


@pytest.fixture(autouse=True)
def _tmp_decision_log(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_node, "decision_log_path",
                        lambda: tmp_path / "decision_log.jsonl")
    yield tmp_path / "decision_log.jsonl"


# ── spec validation ──────────────────────────────────────────────────────────

class TestValidation:
    def test_good_llm_node_passes(self):
        assert _spec().validate() == []

    def test_missing_responsibility(self):
        n = json.loads(json.dumps(ROUTE_NODE)); n.pop("responsibility")
        assert any("responsibility" in p for p in _spec([n]).validate())

    def test_escape_must_be_a_route(self):
        n = json.loads(json.dumps(ROUTE_NODE)); n["escape"] = "nope"
        assert any("escape" in p for p in _spec([n]).validate())

    def test_uncertain_route_name_reserved(self):
        n = json.loads(json.dumps(ROUTE_NODE))
        n["routes"]["uncertain"] = []
        assert any("reserved" in p for p in _spec([n]).validate())

    def test_data_mode_schema_types(self):
        n = {"type": "llm", "id": "d", "mode": "data", "responsibility": "x",
             "output_schema": {"a": "complex"}}
        assert any("output_schema" in p for p in _spec([n]).validate())

    def test_duplicate_id_inside_route_caught(self):
        n = json.loads(json.dumps(ROUTE_NODE))
        n["routes"]["ok"][0]["id"] = "q"          # collides with outer step
        assert any("duplicate" in p for p in _spec([n]).validate())


# ── route mode（经解释器全链路）───────────────────────────────────────────────

class TestRouteMode:
    def test_chosen_branch_runs(self, monkeypatch):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: FakeStructuredModel(
                                {"route": "fix", "reason": "blunt"}))
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.2})})
        res = _run(_spec(), ctx)
        assert res.success, res.error
        called = [c[0] for c in ctx.calls]
        assert "PulseIt" in called and "SaveIt" not in called

    def test_inputs_resolved_into_prompt(self, monkeypatch):
        model = FakeStructuredModel({"route": "ok", "reason": ""})
        monkeypatch.setattr(llm_node, "_model_factory", lambda node: model)
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.93})})
        _run(_spec(), ctx)
        joined = json.dumps(model.last_messages, ensure_ascii=False)
        assert "0.93" in joined and "判断针尖" in joined

    def test_uncertain_goes_to_escape(self, monkeypatch):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: FakeStructuredModel(
                                {"route": "uncertain", "reason": "?"}))
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.5})})
        _run(_spec(), ctx)
        assert "CallHuman" in [c[0] for c in ctx.calls]

    def test_model_exception_goes_to_escape(self, monkeypatch):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: ExplodingModel())
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.5})})
        res = _run(_spec(), ctx)
        assert res.success, res.error            # 工作流不崩，走 escape
        assert "CallHuman" in [c[0] for c in ctx.calls]

    def test_text_tier_fallback(self, monkeypatch):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: FakeTextModel(
                                '前略 {"route": "ok", "reason": "good"} 后略'))
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.9})})
        _run(_spec(), ctx)
        assert "SaveIt" in [c[0] for c in ctx.calls]

    def test_route_binding_visible_downstream(self, monkeypatch):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: FakeStructuredModel(
                                {"route": "ok", "reason": "fine"}))
        tail = [json.loads(json.dumps(ROUTE_NODE)),
                {"type": "step", "id": "after", "skill": "After",
                 "params": {"chosen": {"$expr": "router['route']"}}}]
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.9})})
        _run(_spec(tail), ctx)
        after = [c for c in ctx.calls if c[0] == "After"]
        assert after and after[0][1]["chosen"] == "ok"

    def test_decision_logged(self, monkeypatch, _tmp_decision_log):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: FakeStructuredModel(
                                {"route": "ok", "reason": "good"}))
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.9})})
        _run(_spec(), ctx)
        lines = _tmp_decision_log.read_text("utf-8").strip().splitlines()
        rec = json.loads(lines[-1])
        assert rec["workflow"] == "LlmDemo" and rec["node_id"] == "router"
        assert rec["route"] == "ok" and rec["parse_path"] == "structured"
        assert rec["model"] == "fake/structured"
        assert rec["inputs"]["quality"] == 0.9

    def test_resume_replays_cached_decision_without_llm(self, monkeypatch):
        """决策已缓存进 progress.partial_data 时绝不重新问 LLM。"""
        calls = {"n": 0}
        def _factory(node):
            calls["n"] += 1
            return FakeStructuredModel({"route": "fix", "reason": ""})
        monkeypatch.setattr(llm_node, "_model_factory", _factory)
        from mast.skills.composite.graph_executor import GraphExecutor
        from mast.skills.composite.interpreter import SpecComposite
        spec = _spec()
        skill = SpecComposite(spec)
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.2})})
        ex = GraphExecutor(spec.name, ctx)
        # 预置缓存 = 上一次 run 已选 ok（与本次模型想选的 fix 相反）
        ex.progress.partial_data["_decisions"] = {
            "router": {"route": "ok", "reason": "cached", "escaped": False}}
        ex.run_plan(skill.plan_dynamic({}, ex))
        assert calls["n"] == 0                  # LLM 从未被问
        assert "SaveIt" in [c[0] for c in ctx.calls]   # 走缓存的 ok 分支


# ── data mode ────────────────────────────────────────────────────────────────

DATA_NODE = {
    "type": "llm", "id": "judge", "mode": "data",
    "responsibility": "给出修针策略",
    "inputs": {"quality": {"$expr": "q['quality']"}},
    "output_schema": {"strategy": "str", "confident": "bool"},
    "on_error": [{"type": "step", "id": "fallback", "skill": "CallHuman",
                  "params": {}}],
}


class TestDataMode:
    def test_data_bound_downstream(self, monkeypatch):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: FakeStructuredModel(
                                {"strategy": "gentle", "confident": True}))
        nodes = [json.loads(json.dumps(DATA_NODE)),
                 {"type": "step", "id": "use", "skill": "Use",
                  "params": {"s": {"$expr": "judge['strategy']"}}}]
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.4})})
        _run(_spec(nodes), ctx)
        use = [c for c in ctx.calls if c[0] == "Use"]
        assert use and use[0][1]["s"] == "gentle"

    def test_parse_failure_walks_on_error(self, monkeypatch):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: FakeTextModel("我也说不好啊"))
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.4})})
        _run(_spec([json.loads(json.dumps(DATA_NODE))]), ctx)
        assert "CallHuman" in [c[0] for c in ctx.calls]

    def test_text_tier_json_parsed_and_coerced(self, monkeypatch):
        monkeypatch.setattr(llm_node, "_model_factory",
                            lambda node: FakeTextModel(
                                '{"strategy": "poke", "confident": "true"}'))
        nodes = [json.loads(json.dumps(DATA_NODE)),
                 {"type": "step", "id": "use", "skill": "Use",
                  "params": {"c": {"$expr": "judge['confident']"}}}]
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.4})})
        _run(_spec(nodes), ctx)
        use = [c for c in ctx.calls if c[0] == "Use"]
        assert use and use[0][1]["c"] is True   # "true" → bool 强转


# ── persona（P2-C）────────────────────────────────────────────────────────────

class TestPersona:
    def test_list_and_render(self):
        from mast.skills.composite.persona import list_personas, render_sections
        cat = {p["id"]: p for p in list_personas()}
        assert "instrument_control" in cat
        assert any(s["id"] == "tip_conditioning"
                   for s in cat["instrument_control"]["sections"])
        txt = render_sections("instrument_control", ["tip_conditioning"])
        assert "针尖" in txt
        all_txt = render_sections("instrument_control")
        assert len(all_txt) > len(txt)

    def test_version_suffix_tolerated(self):
        from mast.skills.composite.persona import render_sections
        assert "针尖" in render_sections("instrument_control@999",
                                          ["tip_conditioning"])

    def test_unknown_persona_degrades_in_llm_node(self, monkeypatch):
        """未知 persona 不阻断决策——llm_node 降级为无 persona 注入。"""
        model = FakeStructuredModel({"route": "ok", "reason": ""})
        monkeypatch.setattr(llm_node, "_model_factory", lambda node: model)
        n = json.loads(json.dumps(ROUTE_NODE))
        n["persona"] = "no_such_agent"
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.9})})
        res = _run(_spec([n]), ctx)
        assert res.success
        assert "SaveIt" in [c[0] for c in ctx.calls]

    def test_persona_text_injected_into_prompt(self, monkeypatch):
        model = FakeStructuredModel({"route": "ok", "reason": ""})
        monkeypatch.setattr(llm_node, "_model_factory", lambda node: model)
        n = json.loads(json.dumps(ROUTE_NODE))
        n["persona"] = "instrument_control"
        n["context_sections"] = ["safety"]
        ctx = FakeCtx({"Assess": _Res(data={"quality": 0.9})})
        _run(_spec([n]), ctx)
        joined = json.dumps(model.last_messages, ensure_ascii=False)
        assert "粗逼近" in joined           # safety 切片确实进了 prompt


# ── builder lint 对 llm 节点的支持 ───────────────────────────────────────────

def test_builder_validate_descends_llm_routes(monkeypatch):
    from mast.core.registry import SkillRegistry
    from mast.webui import builder_api
    reg = SkillRegistry()
    monkeypatch.setattr(builder_api, "_registry", reg)
    spec = {"name": "X", "safety_level": "confirm", "params": [],
            "nodes": [json.loads(json.dumps(ROUTE_NODE))]}
    rep = builder_api.validate_spec_payload(spec)
    # 路由槽位里的 step 技能不存在 → 逐节点错误（证明走查下探了 routes）
    ids = {s["id"] for s in rep["steps"] if s["errors"]}
    assert {"save", "pulse", "call"} <= ids


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
