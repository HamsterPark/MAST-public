"""v2 unit tests for mast.skills.builtins.tip_shaper.

Skills covered: TipShape (AUTO) — 1 skill total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_tip_shaper.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.tip_shaper import TipShape


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_tip_shape_shape():
    tool = wrap_skill(TipShape, make_provider())
    assert tool.name == "TipShape"
    # AUTO: tip shaping ramps fine-Z + pulses bias to reshape the apex; both are
    # Nanonis-bounded (no instrument damage), so autonomous agents may run it ungated.
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    # All parameters are optional
    assert all(not v.is_required() for v in fields.values())
    assert "switch_off_delay_s" in fields
    assert "bias_v" in fields
    assert "timeout_ms" in fields
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["switch_off_delay_s"].annotation is str
    assert fields["timeout_ms"].annotation is int


def test_skill_source_points_to_tip_shaper_module():
    tool = wrap_skill(TipShape, make_provider())
    assert tool.metadata["skill_source"].endswith(".tip_shaper")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_tip_shape_executes_with_defaults():
    canned = {
        "TipShaper_PropsSet": {"return_value": ("", b"", [])},
        "TipShaper_Start": {"return_value": ("", b"", [])},
        # 2026-08-11: TipShape 不再有一个写死的 bias 默认值 —— 没显式给就去读
        # **此刻的成像偏压**,读不到就拒绝执行。所以这些测试必须让它读得到。
        "Bias_Get": {"return_value": ("", b"", [0.05])},
    }
    tool = wrap_skill(TipShape, make_provider(canned))
    result = _invoke(tool)  # all defaults
    update = result.update
    assert update["executed_skills"] == ["TipShape"]
    # ⚠️ ``executed_skills`` 上面那一行**一个字都不检查结果** —— 技能拒绝执行时
    # 它同样是 ["TipShape"]。2026-08-11 实测:一个纯默认的调用因为读不到偏压
    # 而失败,这条测试照旧全绿。所以再钉一条「真的跑成了」。
    # (这正是本轮那组缺陷能活下来的形状之一:断言瞄的不是本该出问题的对象。)
    assert "error" not in str(result).lower(), str(result)
    assert not update.get("error_log"), update.get("error_log")


def test_tip_shape_calls_both_methods():
    canned = {
        "TipShaper_PropsSet": {"return_value": ("", b"", [])},
        "TipShaper_Start": {"return_value": ("", b"", [])},
        # 2026-08-11: TipShape 不再有一个写死的 bias 默认值 —— 没显式给就去读
        # **此刻的成像偏压**,读不到就拒绝执行。所以这些测试必须让它读得到。
        "Bias_Get": {"return_value": ("", b"", [0.05])},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(TipShape, capturing_provider)
    _invoke(tool, bias_v=2.5)
    last_ctx = instances[-1]
    methods = [c[0] for c in last_ctx.calls]
    assert "TipShaper_PropsSet" in methods
    assert "TipShaper_Start" in methods


def test_tip_shape_change_bias_encoding():
    """change_bias=True → encoded as 1 in TipShaper_PropsSet arg position 1."""
    canned = {
        "TipShaper_PropsSet": {"return_value": ("", b"", [])},
        "TipShaper_Start": {"return_value": ("", b"", [])},
        # 2026-08-11: TipShape 不再有一个写死的 bias 默认值 —— 没显式给就去读
        # **此刻的成像偏压**,读不到就拒绝执行。所以这些测试必须让它读得到。
        "Bias_Get": {"return_value": ("", b"", [0.05])},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(TipShape, capturing_provider)
    _invoke(tool, change_bias=True, bias_v=3.0)
    last_ctx = instances[-1]
    props_calls = [c for c in last_ctx.calls if c[0] == "TipShaper_PropsSet"]
    assert len(props_calls) == 1
    # args: (switch_off_delay_s, change_bias_int, bias_v, ...)
    # change_bias=True → 1
    assert props_calls[0][1][1] == 1  # change_bias encoded as 1


def test_tip_shape_restore_feedback_encoding():
    """restore_feedback=False → encoded as 2 in TipShaper_PropsSet last arg."""
    canned = {
        "TipShaper_PropsSet": {"return_value": ("", b"", [])},
        "TipShaper_Start": {"return_value": ("", b"", [])},
        # 2026-08-11: TipShape 不再有一个写死的 bias 默认值 —— 没显式给就去读
        # **此刻的成像偏压**,读不到就拒绝执行。所以这些测试必须让它读得到。
        "Bias_Get": {"return_value": ("", b"", [0.05])},
    }
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(TipShape, capturing_provider)
    _invoke(tool, restore_feedback=False)
    last_ctx = instances[-1]
    props_calls = [c for c in last_ctx.calls if c[0] == "TipShaper_PropsSet"]
    assert len(props_calls) == 1
    # restore_feedback=False → 2 (last arg)
    assert props_calls[0][1][-1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
