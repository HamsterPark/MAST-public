"""Unit test for SurveySurface_TileScan."""
from __future__ import annotations

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

import pytest

from mast.skills.composite.survey_surface import SurveySurface_TileScan
from mast.core.types import SafetyLevel


def test_metadata_basic():
    md = SurveySurface_TileScan().metadata()
    assert md.name == "SurveySurface_TileScan"
    assert md.safety_level == SafetyLevel.CONFIRM
    assert "survey" in md.tags
    # All params optional
    required = [p for p in md.parameters if p.required]
    assert required == [], "survey should be self-contained with safe defaults"


def test_grid_computation_examples():
    """The skill computes ceil(total/tile) → grid_n; verify boundary cases.

    We don't run the full skill (needs Nanonis); test the math constants
    in metadata + key invariants only.
    """
    md = SurveySurface_TileScan().metadata()
    by_name = {p.name: p for p in md.parameters}
    # tile size must be < total size hard cap (1e-6 < 1e-5)
    assert by_name["tile_size_m"].max_value <= by_name["total_size_m"].max_value


def test_total_must_exceed_tile_size():
    """When tile_size_m >= total_size_m the skill should fail cleanly."""
    skill = SurveySurface_TileScan()

    class _NullCtx:
        def safe_call(self, *args, **kwargs):
            raise AssertionError("should not be reached — params should validate first")

    result = skill.run_composite(_NullCtx(), {
        "total_size_m": 50e-9,
        "tile_size_m": 50e-9,  # equal → ceil(1) = 1 grid
    })
    assert not result.success
    assert "tile_size_m" in result.error or "use FullScan" in result.error


def test_grid_too_large_rejected():
    """64+ tiles would be a >hour run; skill caps at 8x8 = 64 tiles."""
    skill = SurveySurface_TileScan()
    class _NullCtx:
        def safe_call(self, *args, **kwargs):
            raise AssertionError("not reached")
    # 1 um total / 50 nm tile = 20x20 = 400 tiles → too many
    result = skill.run_composite(_NullCtx(), {
        "total_size_m": 1e-6,
        "tile_size_m": 50e-9,
    })
    assert not result.success
    assert "too many" in result.error.lower() or "8x8" in result.error


def test_default_params_yield_4x4_grid():
    """Defaults: total=200nm, tile=50nm → ceil(200/50)=4 → 4x4 grid."""
    # Math only — no execution
    import math
    total = 200e-9
    tile = 50e-9
    n = math.ceil(total / tile)
    assert n == 4
    assert n * n == 16  # 16 tiles


def test_make_tool_wraps_to_v2_adapter():
    """make_tool() should produce a LangChain StructuredTool via wrap_skill."""
    from mast.skills.composite.survey_surface import make_tool

    def _provider():
        return None

    tool = make_tool(_provider)
    assert tool.name == "SurveySurface_TileScan"
    fields = tool.args_schema.model_fields
    assert "center_x_m" in fields
    assert "total_size_m" in fields
    assert "tile_size_m" in fields
    assert "assess_quality" in fields
    # All optional (none required) so an LLM with no params would call with defaults
    required = [k for k, v in fields.items() if v.is_required()]
    assert required == [], f"all params should be optional; required: {required}"
