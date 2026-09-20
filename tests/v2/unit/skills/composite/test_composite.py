"""v2 unit tests for mast.skills.composite.

SHAPE-ONLY tests: wrap_skill produces a tool with the right name,
danger_level, and args_schema. End-to-end execution with sub-skill
chaining requires hardware + Phase 4 Session 4 runtime wiring.

Skills covered:
  FullScan (CONFIRM)
  AssessImageQuality (AUTO)
  ConditionTip (DANGEROUS)
  TipPulse (DANGEROUS)
  GridSTS (CONFIRM)
  PreScanCheck (CONFIRM)
  TrackDrift_ReferenceScan (CONFIRM)

Run:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/composite/test_composite.py -x -v
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
from mast.core.types import NanonisCallRecord, SkillResult

# Import all composite skill classes
from mast.skills.composite.full_scan import FullScan
from mast.skills.composite.assess_quality import AssessImageQuality
from mast.skills.composite.condition_tip import ConditionTip
from mast.skills.composite.tip_pulse import TipPulse
from mast.skills.composite.grid_sts import GridSTS
from mast.skills.composite.prescan_check import PreScanCheck
from mast.skills.composite.drift_track import TrackDrift_ReferenceScan


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)
    _abort: bool = False

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, return_value=None, error="")

    def check_abort(self) -> bool:
        return self._abort

    def run(self, skill_name: str, params: dict) -> SkillResult:
        """Fake context.run() for composite sub-skill calls."""
        return SkillResult(
            skill_name=skill_name,
            success=True,
            data={},
            nanonis_calls=[],
        )


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_full_scan_shape():
    tool = wrap_skill(FullScan, make_provider())
    assert tool.name == "FullScan"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "center_x_m" in fields
    assert "center_y_m" in fields
    assert "width_m" in fields
    assert "height_m" in fields
    assert fields["center_x_m"].is_required()
    assert fields["center_y_m"].is_required()


def test_full_scan_source():
    tool = wrap_skill(FullScan, make_provider())
    assert "full_scan" in tool.metadata["skill_source"]


def test_assess_image_quality_shape():
    tool = wrap_skill(AssessImageQuality, make_provider())
    assert tool.name == "AssessImageQuality"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "scan_path" in fields
    assert not fields["scan_path"].is_required()


def test_condition_tip_shape():
    tool = wrap_skill(ConditionTip, make_provider())
    assert tool.name == "ConditionTip"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "pulse_v" in fields
    # 2026-07-31 起 pulse_v **不再必填**。定案:「（当前水平的）LLM 可不懂
    # STM 实验，让他自己想参数就麻烦了」—— 留空则按当前针尖的「材料 × 制备 ×
    # 形态」查方案表(mast.core.tip_conditioning_policy)。必填等于逼模型每次现编
    # 一个电压,那正是要移除的诱因。
    assert not fields["pulse_v"].is_required()
    assert "max_attempts" in fields
    assert "target_quality" in fields


def test_tip_pulse_shape():
    tool = wrap_skill(TipPulse, make_provider())
    assert tool.name == "TipPulse"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "pulse_v" in fields
    assert not fields["pulse_v"].is_required()   # 同上:留空 → 针尖方案表
    assert "duration_s" in fields
    assert "count" in fields


def test_grid_sts_shape():
    tool = wrap_skill(GridSTS, make_provider())
    assert tool.name == "GridSTS"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "center_x_m" in fields
    assert "center_y_m" in fields
    assert "spacing_m" in fields
    assert fields["spacing_m"].is_required()


def test_prescan_check_shape():
    tool = wrap_skill(PreScanCheck, make_provider())
    assert tool.name == "PreScanCheck"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "center_x_m" in fields
    assert "width_m" in fields
    assert fields["width_m"].is_required()


def test_track_drift_shape():
    tool = wrap_skill(TrackDrift_ReferenceScan, make_provider())
    assert tool.name == "TrackDrift_ReferenceScan"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "ref_x_m" in fields
    assert "ref_y_m" in fields
    assert fields["ref_x_m"].is_required()


# ── Metadata tests ─────────────────────────────────────────────────────────────

def test_all_composites_have_version():
    for cls in [FullScan, AssessImageQuality, ConditionTip, TipPulse,
                GridSTS, PreScanCheck, TrackDrift_ReferenceScan]:
        tool = wrap_skill(cls, make_provider())
        assert tool.metadata["skill_version"], f"{cls.__name__} missing version"


def test_all_composites_have_description():
    for cls in [FullScan, AssessImageQuality, ConditionTip, TipPulse,
                GridSTS, PreScanCheck, TrackDrift_ReferenceScan]:
        tool = wrap_skill(cls, make_provider())
        assert len(tool.description) > 10, f"{cls.__name__} description too short"


# ── TipPulse execution test ───────────────────────────────────────────────────

def test_tip_pulse_executes_with_fake_ctx():
    """TipPulse doesn't call context.run (it's atomic), so we can test execution."""
    canned = {
        "Bias_Get": {"return_value": ("", b"", [-0.5])},
        "Bias_Set": {"return_value": ("", b"", [])},
    }
    tool = wrap_skill(TipPulse, make_provider(canned))
    result = _invoke(tool, pulse_v=3.0, duration_s=0.01, count=1)
    update = result.update
    assert update["executed_skills"] == ["TipPulse"]


def test_tip_pulse_missing_required():
    tool = wrap_skill(TipPulse, make_provider())
    result = _invoke(tool)  # missing pulse_v
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "pulse_v" in msg_content
        or "Missing" in msg_content
    )


# ── AssessImageQuality execution test (no hardware needed for missing-file path) ──

def test_assess_quality_no_file_returns_failure():
    """Without a scan file, AssessImageQuality should return success=False."""
    tool = wrap_skill(AssessImageQuality, make_provider())
    result = _invoke(tool, scan_path="")
    # Should complete (no exception); may succeed or fail depending on scan finder
    update = result.update
    assert update["executed_skills"] == ["AssessImageQuality"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
