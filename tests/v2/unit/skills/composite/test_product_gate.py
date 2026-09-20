"""Feedback ⑨: composite product-validity gate.

"跑完/存盘" is not "produced a usable product". 5305868e: a scan finished
(timed_out:False) and was saved, yet the frame was crashed / all-NaN and the run
still reported success. CompositeSkillGraph._graph_execute now runs a
positive-evidence-only product check after a drained plan, downgrading an
otherwise-"ok" composite to a degraded FAILURE when the terminal product is a
crash / all-NaN / dead-flat / unreadable scan. A composite with no recognizable
product is untouched.
"""
from __future__ import annotations

# ── path bootstrap ──
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

import uuid
from dataclasses import dataclass, field

import numpy as np

from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeStep


@dataclass
class _FakeResult:
    success: bool = True
    error: str = ""
    data: dict = field(default_factory=dict)
    nanonis_calls: list = field(default_factory=list)


class _FakeCtx:
    """Minimal ExecutionContext: every sub-skill 'runs' and succeeds."""

    def __init__(self):
        self.run_id = "gate-" + uuid.uuid4().hex[:8]

    def run(self, skill_name, params, **kw):
        return _FakeResult(success=True, data={})


class _ProbeComposite(CompositeSkillGraph):
    """A one-step composite whose aggregate data we control, to drive the gate."""

    def __init__(self, agg: dict):
        super().__init__()
        self._agg = agg

    def metadata(self):
        return SkillMetadata(
            name="ProbeComposite", version="1.0.0",
            category=SkillCategory.ANALYSIS, safety_level=SafetyLevel.AUTO,
            description="probe", estimated_duration_s=0.1, composition_level=1,
        )

    def plan(self, params):
        return [CompositeStep(step_id="s1", skill_name="Noop", params={})]

    def aggregate(self, sub_results, progress):
        return dict(self._agg)


def _run(agg: dict):
    return _ProbeComposite(agg).execute(_FakeCtx(), {})


def test_clean_composite_succeeds():
    res = _run({"foo": 1})
    assert res.success is True


def test_no_product_keys_is_untouched():
    # arbitrary non-path data must never trip the gate
    res = _run({"scanned": 3, "best_quality": 0.4, "note": "all good"})
    assert res.success is True


def test_crash_indicator_downgrades_to_degraded():
    res = _run({"crash_indicator": True, "crash_channel": "Z"})
    assert res.success is False
    assert res.error.startswith("degraded:")
    assert res.data.get("degraded") is True


def test_all_nan_product_downgrades(tmp_path):
    p = tmp_path / "nan_scan.npy"
    np.save(p, np.full((8, 8), np.nan))
    res = _run({"saved_path": str(p)})
    assert res.success is False and res.error.startswith("degraded:")


def test_dead_flat_product_downgrades(tmp_path):
    p = tmp_path / "flat_scan.npy"
    np.save(p, np.zeros((8, 8)))
    res = _run({"output_path": str(p)})
    assert res.success is False and res.error.startswith("degraded:")


def test_valid_product_passes(tmp_path):
    p = tmp_path / "good_scan.npy"
    np.save(p, np.random.default_rng(2).normal(size=(8, 8)))
    res = _run({"saved_path": str(p)})
    assert res.success is True


def test_unreadable_product_downgrades(tmp_path):
    # a product path that "was saved" but does not exist on disk
    res = _run({"scan_path": str(tmp_path / "vanished.sxm")})
    assert res.success is False and res.error.startswith("degraded:")


def test_non_scan_artifact_path_is_ignored(tmp_path):
    # a .png thumbnail is not a scan array — the gate must not try to judge it
    res = _run({"output_path": str(tmp_path / "thumb.png")})
    assert res.success is True
