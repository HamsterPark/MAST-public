"""P5 control-flow extension: try/finally, break/continue, succeed/fail,
success_when, the `_failed` marker, and the safe_eval sequence-mul guard.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_spec_controlflow.py -x -v
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
    CompositeSpec, ExprError, ParamSpec, safe_eval,
)


# ── Fake execution context (mirrors test_spec.py) ───────────────────────
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
        return r or _Res(success=True, data={})


def _run(spec, params, ctx):
    from mast.skills.composite.interpreter import SpecComposite
    return SpecComposite(spec).execute(ctx, params)


def _skills(ctx):
    return [c[0] for c in ctx.calls]


# ── try / finally ───────────────────────────────────────────────────────
class TestTryFinally:
    def test_finally_runs_on_normal_completion(self):
        spec = CompositeSpec(name="T", nodes=[
            {"type": "try",
             "body": [{"type": "step", "id": "a", "skill": "A", "params": {}}],
             "finally": [{"type": "step", "id": "cl", "skill": "Cleanup", "params": {}}]},
        ])
        ctx = FakeCtx()
        res = _run(spec, {}, ctx)
        assert res.success
        assert _skills(ctx) == ["A", "Cleanup"]

    def test_body_step_failure_does_not_abort_and_finally_runs(self):
        # A mandatory-looking step inside try is forced optional → the body
        # keeps going AND the finally region runs.
        spec = CompositeSpec(name="T", nodes=[
            {"type": "try", "body": [
                {"type": "step", "id": "a", "skill": "WillFail", "params": {}},
                {"type": "step", "id": "b", "skill": "B", "params": {}},
            ], "finally": [{"type": "step", "id": "cl", "skill": "Cleanup", "params": {}}]},
        ])
        ctx = FakeCtx({"WillFail": _Res(success=False, error="boom")})
        res = _run(spec, {}, ctx)
        assert _skills(ctx) == ["WillFail", "B", "Cleanup"]
        assert res.success  # no abort, no verdict → all_good

    def test_finally_runs_on_early_fail(self):
        spec = CompositeSpec(name="T", nodes=[
            {"type": "try", "body": [
                {"type": "step", "id": "a", "skill": "A", "params": {}},
                {"type": "fail", "reason": "'gave up'"},
                {"type": "step", "id": "never", "skill": "Never", "params": {}},
            ], "finally": [{"type": "step", "id": "cl", "skill": "Cleanup", "params": {}}]},
        ])
        ctx = FakeCtx()
        res = _run(spec, {}, ctx)
        assert res.success is False
        assert res.error == "gave up"
        assert _skills(ctx) == ["A", "Cleanup"]  # 'never' skipped, cleanup ran

    def test_break_from_try_runs_finally_then_breaks_loop(self):
        spec = CompositeSpec(name="T", nodes=[
            {"type": "loop", "id": "lp", "mode": "repeat", "count": "3", "var": "i", "body": [
                {"type": "try", "body": [
                    {"type": "step", "id": "a", "skill": "A", "params": {}},
                    {"type": "if", "id": "c", "cond": "i == 0",
                     "then": [{"type": "break"}], "else": []},
                ], "finally": [{"type": "step", "id": "cl", "skill": "CL", "params": {}}]},
            ]},
        ])
        ctx = FakeCtx()
        _run(spec, {}, ctx)
        # i=0: A, break → finally CL runs, then break exits the loop
        assert _skills(ctx) == ["A", "CL"]


# ── break / continue ────────────────────────────────────────────────────
class TestBreakContinue:
    def test_break_repeat(self):
        spec = CompositeSpec(name="T", params=[ParamSpec("n", "int", 5)], nodes=[
            {"type": "loop", "id": "lp", "mode": "repeat", "count": "n", "var": "i", "body": [
                {"type": "step", "id": "m", "skill": "M", "params": {"i": {"$expr": "i"}}},
                {"type": "if", "id": "c", "cond": "i >= 2",
                 "then": [{"type": "break"}], "else": []},
            ]},
        ])
        ctx = FakeCtx()
        _run(spec, {"n": 5}, ctx)
        assert [c[1]["i"] for c in ctx.calls] == [0, 1, 2]

    def test_continue_while(self):
        spec = CompositeSpec(name="T", nodes=[
            {"type": "set", "var": "k", "value": "0"},
            {"type": "loop", "id": "w", "mode": "while", "cond": "k < 4", "var": "i", "body": [
                {"type": "set", "var": "k", "value": "k + 1"},
                {"type": "if", "id": "odd", "cond": "k % 2 == 1",
                 "then": [{"type": "continue"}], "else": []},
                {"type": "step", "id": "even", "skill": "Even", "params": {"k": {"$expr": "k"}}},
            ]},
        ])
        ctx = FakeCtx()
        _run(spec, {}, ctx)
        # k: 1(skip) 2(Even) 3(skip) 4(Even)
        assert [c[1]["k"] for c in ctx.calls] == [2, 4]


# ── succeed / fail verdict nodes ────────────────────────────────────────
class TestVerdictNodes:
    def test_succeed_ends_early_with_success(self):
        spec = CompositeSpec(name="T", nodes=[
            {"type": "step", "id": "a", "skill": "A", "params": {}},
            {"type": "succeed", "reason": "'done early'"},
            {"type": "step", "id": "never", "skill": "Never", "params": {}},
        ])
        ctx = FakeCtx()
        res = _run(spec, {}, ctx)
        assert res.success
        assert _skills(ctx) == ["A"]

    def test_fail_node_overrides_all_good(self):
        # Every step succeeds, but a `fail` node makes the composite fail.
        spec = CompositeSpec(name="T", nodes=[
            {"type": "step", "id": "a", "skill": "A", "params": {}},
            {"type": "fail", "reason": "'policy says no'"},
        ])
        ctx = FakeCtx()
        res = _run(spec, {}, ctx)
        assert res.success is False
        assert res.error == "policy says no"


# ── success_when / fail_message ─────────────────────────────────────────
class TestSuccessWhen:
    def test_success_when_true(self):
        spec = CompositeSpec(name="T", success_when="succeeded >= 1", nodes=[
            {"type": "set", "var": "succeeded", "value": "0"},
            {"type": "loop", "id": "lp", "mode": "repeat", "count": "3", "var": "i", "body": [
                {"type": "step", "id": "m", "skill": "M", "params": {}, "optional": True},
                {"type": "set", "var": "succeeded", "value": "succeeded + 1"},
            ]},
        ])
        ctx = FakeCtx()
        res = _run(spec, {}, ctx)
        assert res.success

    def test_success_when_false_uses_fail_message(self):
        spec = CompositeSpec(name="T", success_when="succeeded >= 1",
                             fail_message="'no points succeeded'", nodes=[
            {"type": "set", "var": "succeeded", "value": "0"},
        ])
        ctx = FakeCtx()
        res = _run(spec, {}, ctx)
        assert res.success is False
        assert res.error == "no points succeeded"


# ── `_failed` marker on optional/try-body step failure ──────────────────
class TestFailedMarker:
    def _spec(self):
        return CompositeSpec(name="T", nodes=[
            {"type": "try", "body": [
                {"type": "step", "id": "move", "skill": "Move", "params": {}},
                {"type": "if", "id": "c", "cond": "'_failed' in move",
                 "then": [{"type": "step", "id": "rec", "skill": "Recover", "params": {}}],
                 "else": [{"type": "step", "id": "go", "skill": "Go", "params": {}}]},
            ], "finally": []},
        ])

    def test_failed_branch(self):
        ctx = FakeCtx({"Move": _Res(success=False, error="x")})
        _run(self._spec(), {}, ctx)
        assert _skills(ctx) == ["Move", "Recover"]

    def test_ok_branch(self):
        ctx = FakeCtx({"Move": _Res(success=True, data={"x": 1})})
        _run(self._spec(), {}, ctx)
        assert _skills(ctx) == ["Move", "Go"]


# ── safe_eval sequence-multiplication guard ─────────────────────────────
class TestSeqMulGuard:
    def test_rejects_string_bomb(self):
        with pytest.raises(ExprError):
            safe_eval("'a' * 10000000", {})

    def test_rejects_list_bomb(self):
        with pytest.raises(ExprError):
            safe_eval("[0] * 10000000", {})

    def test_allows_small(self):
        assert safe_eval("'ab' * 3", {}) == "ababab"


# ── validation of the new nodes ─────────────────────────────────────────
class TestValidation:
    def test_break_outside_loop_rejected(self):
        probs = CompositeSpec(name="T", nodes=[{"type": "break"}]).validate()
        assert any("only valid inside a loop" in p for p in probs)

    def test_continue_inside_loop_ok(self):
        spec = CompositeSpec(name="T", nodes=[
            {"type": "loop", "id": "lp", "mode": "repeat", "count": "3", "body": [
                {"type": "continue"}]},
        ])
        assert spec.validate() == []

    def test_try_succeed_fail_validate_ok(self):
        spec = CompositeSpec(name="T", success_when="x > 0",
                             params=[ParamSpec("x", "number", 1)], nodes=[
            {"type": "try", "body": [
                {"type": "step", "id": "a", "skill": "A", "params": {}},
                {"type": "fail", "reason": "'bad'"},
            ], "finally": [{"type": "step", "id": "c", "skill": "C", "params": {}}]},
            {"type": "succeed"},
        ])
        assert spec.validate() == []

    def test_bad_success_when_syntax(self):
        probs = CompositeSpec(name="T", success_when="a b c", nodes=[]).validate()
        assert any("success_when" in p and "syntax" in p for p in probs)

    def test_roundtrip_preserves_verdict_fields(self):
        s = CompositeSpec(name="T", success_when="x>0", fail_message="'no'",
                          nodes=[{"type": "step", "id": "a", "skill": "A", "params": {}}])
        s2 = CompositeSpec.from_dict(s.to_dict())
        assert s2.success_when == "x>0" and s2.fail_message == "'no'"
        assert s2.to_dict() == s.to_dict()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
