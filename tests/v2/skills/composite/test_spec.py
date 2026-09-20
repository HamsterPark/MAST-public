"""Declarative composite spec: safe evaluator + interpreter end-to-end.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_spec.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
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

from mast.skills.composite.spec import (
    CompositeSpec, ExprError, ParamSpec, resolve_params, safe_eval,
)


# ── Fake execution context (what GraphExecutor needs) ───────────────────
class _Res:
    def __init__(self, success=True, data=None, error=""):
        self.success = success
        self.data = data or {}
        self.error = error
        self.nanonis_calls = []


class FakeCtx:
    def __init__(self, results=None):
        self.calls = []                  # [(skill, params), ...]
        self._results = results or {}    # skill_name -> _Res or callable
    def run(self, skill_name, params):
        self.calls.append((skill_name, dict(params)))
        r = self._results.get(skill_name)
        if callable(r):
            return r(params)
        return r or _Res(success=True, data={})


def _run(spec, params, ctx):
    from mast.skills.composite.interpreter import SpecComposite
    skill = SpecComposite(spec)
    return skill.execute(ctx, params)


# ── Safe evaluator ──────────────────────────────────────────────────────
class TestSafeEval:
    @pytest.mark.parametrize("expr,ctx,expect", [
        ("1 + 2 * 3", {}, 7),
        ("(a + b) / 2", {"a": 4, "b": 6}, 5.0),
        ("n > 1 and n < 10", {"n": 5}, True),
        ("x if x > 0 else -x", {"x": -3}, 3),
        ("d['k']", {"d": {"k": 42}}, 42),
        ("len(items)", {"items": [1, 2, 3]}, 3),
        ("max(a, b)", {"a": 2, "b": 9}, 9),
        ("v not in [1, 2, 3]", {"v": 5}, True),
        ("list(range(3))", {}, [0, 1, 2]),
    ])
    def test_ok(self, expr, ctx, expect):
        assert safe_eval(expr, ctx) == expect

    @pytest.mark.parametrize("expr", [
        "__import__('os')",          # import
        "(1).__class__",             # attribute / dunder
        "open('x')",                 # non-whitelisted call
        "[x for x in range(3)]",     # comprehension
        "lambda: 1",                 # lambda
        "a.b",                       # attribute access
        "().__class__.__bases__",    # dunder chain
        "globals()",                 # non-whitelisted call
        "2 ** 99999",                # exponent guard
    ])
    def test_rejects_unsafe(self, expr):
        with pytest.raises(ExprError):
            safe_eval(expr, {"a": 1})

    def test_unknown_var(self):
        with pytest.raises(ExprError):
            safe_eval("missing + 1", {})

    def test_range_cap(self):
        with pytest.raises(ExprError):
            safe_eval("range(10000000)", {})


# ── resolve_params ──────────────────────────────────────────────────────
def test_resolve_params():
    ctx = {"v": 5, "name": "tip"}
    out = resolve_params({"a": 1, "b": {"$expr": "v * 2"}, "c": "literal",
                          "nested": {"x": {"$expr": "v + 1"}}}, ctx)
    assert out == {"a": 1, "b": 10, "c": "literal", "nested": {"x": 6}}


# ── Spec validation + clone ─────────────────────────────────────────────
class TestSpec:
    def _good_spec(self):
        return CompositeSpec(
            name="Demo", safety_level="confirm",
            params=[ParamSpec(name="n", type="int", default=2)],
            nodes=[
                {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
                {"type": "if", "id": "c", "cond": "n > 1",
                 "then": [{"type": "step", "id": "b", "skill": "BiasPulse", "params": {}}],
                 "else": []},
            ],
        )

    def test_validate_ok(self):
        assert self._good_spec().validate() == []

    def test_validate_catches_problems(self):
        spec = CompositeSpec(name="", safety_level="bogus", nodes=[
            {"type": "step", "id": "a"},                      # missing skill
            {"type": "step", "id": "a", "skill": "X", "params": {}},  # dup id
            {"type": "if", "cond": "a b c"},                  # bad expr
            {"type": "loop", "mode": "spin", "body": []},     # bad mode
        ])
        probs = spec.validate()
        assert any("name is empty" in p for p in probs)
        assert any("safety_level" in p for p in probs)
        assert any("missing 'skill'" in p for p in probs)
        assert any("duplicate node id" in p for p in probs)
        assert any("syntax error" in p for p in probs)
        assert any("invalid loop mode" in p for p in probs)

    def test_clone(self):
        s = self._good_spec()
        s.version = 5
        c = s.clone("DemoCopy", author="me")
        assert c.name == "DemoCopy" and c.version == 1 and c.author == "me"
        assert "Cloned from Demo v5" in c.notes
        assert c.nodes == s.nodes and c.nodes is not s.nodes  # deep copy

    def test_roundtrip(self):
        s = self._good_spec()
        assert CompositeSpec.from_dict(s.to_dict()).to_dict() == s.to_dict()


# ── Interpreter end-to-end ──────────────────────────────────────────────
class TestInterpreter:
    def test_if_loop_set(self):
        spec = CompositeSpec(name="Demo", params=[ParamSpec("n", "int", 3)], nodes=[
            {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
            {"type": "set", "var": "thr", "value": "0.5"},
            {"type": "if", "id": "c", "cond": "n > 1", "then": [
                {"type": "loop", "id": "lp", "mode": "repeat", "count": "n", "var": "i",
                 "body": [{"type": "step", "id": "p", "skill": "BiasPulse",
                           "params": {"i": {"$expr": "i"}, "v": {"$expr": "n * 2"}}}]},
            ], "else": [{"type": "step", "id": "s", "skill": "Single", "params": {}}]},
        ])
        ctx = FakeCtx()
        res = _run(spec, {"n": 3}, ctx)
        assert res.success
        skills = [c[0] for c in ctx.calls]
        assert skills == ["GetBias", "BiasPulse", "BiasPulse", "BiasPulse"]
        # params resolved per iteration
        assert [c[1] for c in ctx.calls if c[0] == "BiasPulse"] == [
            {"i": 0, "v": 6}, {"i": 1, "v": 6}, {"i": 2, "v": 6}]

    def test_else_branch(self):
        spec = CompositeSpec(name="Demo", params=[ParamSpec("n", "int", 1)], nodes=[
            {"type": "if", "id": "c", "cond": "n > 1",
             "then": [{"type": "step", "id": "p", "skill": "Many", "params": {}}],
             "else": [{"type": "step", "id": "s", "skill": "Single", "params": {}}]},
        ])
        ctx = FakeCtx()
        _run(spec, {"n": 1}, ctx)
        assert [c[0] for c in ctx.calls] == ["Single"]

    def test_foreach(self):
        spec = CompositeSpec(name="Demo", nodes=[
            {"type": "loop", "id": "lp", "mode": "foreach", "var": "pt",
             "iterable": "[10, 20, 30]",
             "body": [{"type": "step", "id": "m", "skill": "Move",
                       "params": {"x": {"$expr": "pt"}}}]},
        ])
        ctx = FakeCtx()
        _run(spec, {}, ctx)
        assert [c[1]["x"] for c in ctx.calls] == [10, 20, 30]

    def test_result_binding_condition(self):
        # a later 'if' branches on an earlier step's result.data
        spec = CompositeSpec(name="Demo", nodes=[
            {"type": "step", "id": "meas", "skill": "Measure", "params": {}},
            {"type": "if", "id": "q", "cond": "meas['quality'] >= 7",
             "then": [{"type": "step", "id": "ok", "skill": "Accept", "params": {}}],
             "else": [{"type": "step", "id": "bad", "skill": "Retry", "params": {}}]},
        ])
        ctx = FakeCtx({"Measure": _Res(data={"quality": 8})})
        _run(spec, {}, ctx)
        assert [c[0] for c in ctx.calls] == ["Measure", "Accept"]

        ctx2 = FakeCtx({"Measure": _Res(data={"quality": 3})})
        _run(spec, {}, ctx2)
        assert [c[0] for c in ctx2.calls] == ["Measure", "Retry"]

    def test_while_with_max_iter(self):
        spec = CompositeSpec(name="Demo", nodes=[
            {"type": "set", "var": "k", "value": "0"},
            {"type": "loop", "id": "w", "mode": "while", "cond": "k < 3", "var": "i",
             "max_iter": 100, "body": [
                {"type": "step", "id": "tick", "skill": "Tick", "params": {}},
                {"type": "set", "var": "k", "value": "k + 1"},
             ]},
        ])
        ctx = FakeCtx()
        _run(spec, {}, ctx)
        assert [c[0] for c in ctx.calls] == ["Tick", "Tick", "Tick"]

    def test_mandatory_failure_aborts(self):
        spec = CompositeSpec(name="Demo", nodes=[
            {"type": "step", "id": "a", "skill": "WillFail", "params": {}},
            {"type": "step", "id": "b", "skill": "NeverReached", "params": {}},
        ])
        ctx = FakeCtx({"WillFail": _Res(success=False, error="boom")})
        res = _run(spec, {}, ctx)
        assert res.success is False
        assert [c[0] for c in ctx.calls] == ["WillFail"]  # aborted before b


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
