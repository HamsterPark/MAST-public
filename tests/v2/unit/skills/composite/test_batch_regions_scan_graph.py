"""GraphExecutor + BatchRegionsScan tests (region batch-scan composite).

Mirrors test_survey_surface_graph.py: a FakeCtx records run() calls and returns
canned skill results, exercising region JSON parse/validation, plan() step
generation, full execution, per-region .sxm path capture, recommendation, and
partial-success on timeout.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_batch_regions_scan_graph.py -x -v
"""
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

import json
from dataclasses import dataclass, field

import pytest

from mast.core.types import SafetyLevel, SkillResult
from mast.skills.composite.graph_executor import CompositeProgress
from mast.skills.composite.batch_regions_scan import BatchRegionsScan


@dataclass
class FakeCtx:
    run_log: list = field(default_factory=list)
    prior_progress: dict | None = None
    emitted: list = field(default_factory=list)
    quality_cycle: list = field(default_factory=list)
    _q_idx: int = 0
    wait_timed_out: bool = False
    save_counter: int = 0
    #: 0-based index of the first WaitScanComplete call that should time out.
    #: -1 = never. Lets a test express a PARTIAL batch (some regions scanned,
    #: some not), which wait_timed_out (all-or-nothing) cannot.
    timeout_from_call: int = -1
    _wait_calls: int = 0
    #: v6.1.3 — same two knobs for "the scan stopped part-way". A region can be
    #: truncated without ever timing out, which is the whole distinction.
    wait_stopped_early: bool = False
    stopped_from_call: int = -1

    def run(self, skill_name, params):
        self.run_log.append((skill_name, dict(params)))
        if skill_name == "WaitScanComplete":
            timed_out = self.wait_timed_out
            if self.timeout_from_call >= 0:
                timed_out = self._wait_calls >= self.timeout_from_call
            stopped_early = self.wait_stopped_early
            if self.stopped_from_call >= 0:
                stopped_early = self._wait_calls >= self.stopped_from_call
            self._wait_calls += 1
            return SkillResult(skill_name=skill_name, success=True,
                               data={"timed_out": timed_out,
                                     "stopped_early": stopped_early,
                                     "outcome": ("timed_out" if timed_out
                                                 else "stopped_early" if stopped_early
                                                 else "completed"),
                                     "lines_done": 77 if stopped_early else 512,
                                     "lines_total": 512,
                                     "lines_verified": True})
        if skill_name == "SaveScan":
            self.save_counter += 1
            return SkillResult(
                skill_name=skill_name, success=True,
                data={"saved_path": f"/fake/scan_{self.save_counter}.sxm",
                      "timed_out": False})
        if skill_name == "AssessImageQuality":
            if self.quality_cycle:
                q = self.quality_cycle[self._q_idx % len(self.quality_cycle)]
                self._q_idx += 1
            else:
                q = 0.5
            return SkillResult(skill_name=skill_name, success=True,
                               data={"fft_quality": q, "label": "ok"})
        return SkillResult(skill_name=skill_name, success=True, data={})

    def emit_progress(self, progress):
        self.emitted.append(CompositeProgress.from_dict(progress.to_dict()))

    def get_progress(self, name):
        return self.prior_progress

    def checkpoint_flush(self):
        pass


_REGIONS = [
    {"center_x_m": 1e-7, "center_y_m": 0.0, "width_m": 5e-8, "height_m": 5e-8,
     "label": "A"},
    {"center_x_m": -2e-7, "center_y_m": 1e-7, "width_m": 1e-7, "height_m": 1e-7},
]


def _rj(regions=None):
    return json.dumps(regions if regions is not None else _REGIONS)


# ── parsing / validation ──────────────────────────────────────────────────

def test_parse_valid_json():
    regions, err = BatchRegionsScan()._parse_regions({"regions": _rj()})
    assert err == ""
    assert len(regions) == 2
    assert regions[0]["label"] == "A"
    assert regions[1]["label"] == "R2"        # auto-labelled
    assert regions[0]["angle_deg"] == 0.0


def test_parse_list_passthrough():
    regions, err = BatchRegionsScan()._parse_regions({"regions": _REGIONS})
    assert err == "" and len(regions) == 2


def test_parse_invalid_json():
    _, err = BatchRegionsScan()._parse_regions({"regions": "{not json"})
    assert "JSON" in err


def test_parse_empty_and_missing():
    _, err = BatchRegionsScan()._parse_regions({"regions": "[]"})
    assert "empty" in err.lower()
    _, err2 = BatchRegionsScan()._parse_regions({})
    assert "required" in err2.lower()


def test_parse_missing_fields():
    _, err = BatchRegionsScan()._parse_regions(
        {"regions": json.dumps([{"center_x_m": 0}])})
    assert "numeric" in err or "center" in err


def test_parse_unit_slip_rejected():
    bad = [{"center_x_m": 0, "center_y_m": 0, "width_m": 100, "height_m": 1e-8}]
    _, err = BatchRegionsScan()._parse_regions({"regions": json.dumps(bad)})
    assert "size out of range" in err


def test_parse_too_many():
    many = [{"center_x_m": 0, "center_y_m": 0, "width_m": 1e-8,
             "height_m": 1e-8}] * 100
    _, err = BatchRegionsScan()._parse_regions({"regions": json.dumps(many)})
    assert "too many" in err


# ── plan ────────────────────────────────────────────────────────────────────

def test_plan_step_count_default():
    # save_each=True (default), assess=False → 5 steps/region
    plan = BatchRegionsScan().plan({"regions": _rj()})
    assert len(plan) == 2 * 5
    assert plan[0].step_id == "region_0:configure"
    assert len({s.step_id for s in plan}) == len(plan)


def test_plan_no_save_no_assess():
    plan = BatchRegionsScan().plan({"regions": _rj(), "save_each": False})
    assert len(plan) == 2 * 4
    assert all("save" not in s.step_id for s in plan)


def test_plan_with_assess():
    plan = BatchRegionsScan().plan({"regions": _rj(), "assess_quality": True})
    assert len(plan) == 2 * 6


def test_plan_phase_order():
    plan = BatchRegionsScan().plan({"regions": _rj(), "assess_quality": True})
    phases = [s.step_id.partition(":")[2] for s in plan[:6]]
    assert phases == ["configure", "speed", "start", "wait", "save", "assess"]


def test_plan_configure_params():
    cfg = BatchRegionsScan().plan({"regions": _rj()})[0]
    assert cfg.skill_name == "ConfigureScan"
    assert cfg.params["center_x_m"] == 1e-7
    assert cfg.params["width_m"] == 5e-8


# ── full execution ──────────────────────────────────────────────────────────

def test_full_execution_walks_every_step():
    ctx = FakeCtx()
    result = BatchRegionsScan().execute(ctx, {"regions": _rj()})
    assert result.success
    names = [n for n, _ in ctx.run_log]
    assert names.count("ConfigureScan") == 2
    assert names.count("StartScan") == 2
    assert names.count("WaitScanComplete") == 2
    assert names.count("SaveScan") == 2
    assert result.data["region_count"] == 2
    assert result.data["success_count"] == 2
    assert result.data["fail_count"] == 0


def test_save_each_records_paths():
    ctx = FakeCtx()
    result = BatchRegionsScan().execute(ctx, {"regions": _rj()})
    paths = result.data["scanned_paths"]
    assert len(paths) == 2
    assert all(p.endswith(".sxm") for p in paths)
    assert all(r.get("sxm_path") for r in result.data["regions"])


def test_recommended_region_highest_quality():
    ctx = FakeCtx(quality_cycle=[0.2, 0.8])
    result = BatchRegionsScan().execute(
        ctx, {"regions": _rj(), "assess_quality": True})
    best = result.data["recommended_region"]
    assert best is not None
    assert best["quality"] == pytest.approx(0.8)


def test_invalid_regions_fail_no_subskill():
    ctx = FakeCtx()
    result = BatchRegionsScan().execute(ctx, {"regions": "[]"})
    assert not result.success
    assert ctx.run_log == []


def test_all_regions_failing_is_not_a_success():
    """2026-07-27 field forensics: a batch where EVERY region failed used to
    return success=True under "partial-success semantics", so the actions table
    recorded result.success = true and the operator was told the batch scanned.
    Nothing downstream ever read fail_count. Zero scanned = failure."""
    ctx = FakeCtx(wait_timed_out=True)
    result = BatchRegionsScan().execute(ctx, {"regions": _rj()})
    assert not result.success, "0/2 regions scanned was reported as success"
    assert result.data["fail_count"] == 2
    assert result.data["success_count"] == 0
    # The partial data must still be attached so the caller can see WHY.
    assert result.data["region_count"] == 2


def test_partial_success_stays_success_but_says_so():
    """A genuine partial keeps success=True — the scanned regions are real data
    and re-running the whole batch would waste them — but the summary must state
    the shortfall instead of leaving it inside data["fail_count"], which is
    exactly where nobody looked."""
    # Region 0 scans; region 1 times out.
    ctx = FakeCtx(timeout_from_call=1)
    result = BatchRegionsScan().execute(ctx, {"regions": _rj()})
    assert result.data["success_count"] >= 1, "fixture produced no success"
    assert result.success
    assert result.data["fail_count"] >= 1
    assert result.summary and "部分完成" in result.summary


# ── regions stopped part-way (v6.1.3, KNOWN_ISSUES §2.24) ───────────────
#
# Pinning that SOMEONE READS `stopped_early` — every assertion is on the
# batch's own region records, never on the wait skill's data.


def test_a_region_stopped_part_way_is_marked_failed():
    """Follows this call site's existing timeout policy rather than inventing
    one: the REGION fails, the batch carries on. Truncating one region says
    nothing about the next."""
    result = BatchRegionsScan().execute(
        FakeCtx(wait_stopped_early=True), {"regions": _rj()})
    assert result.data["fail_count"] == 2
    assert result.data["success_count"] == 0
    assert not result.success              # 0 scanned is still a failed batch


def test_a_truncated_region_says_stopped_not_timeout():
    """Raising the timeout fixes one and does nothing for the other."""
    result = BatchRegionsScan().execute(
        FakeCtx(wait_stopped_early=True), {"regions": _rj()})
    errors = [r.get("error", "") for r in result.data["regions"]]
    assert all("stopped early" in e for e in errors), errors
    assert not any("timeout" in e for e in errors), errors
    assert all("77/512" in e for e in errors), errors


def test_a_partial_batch_with_one_truncated_region_still_returns_its_data():
    """The regions that DID scan are real data — re-running the whole batch to
    punish one truncated region would waste them. Same shape the timeout path
    already has."""
    ctx = FakeCtx(stopped_from_call=1)     # region 0 scans, region 1 truncated
    result = BatchRegionsScan().execute(ctx, {"regions": _rj()})
    assert result.data["success_count"] >= 1
    assert result.data["fail_count"] >= 1
    assert result.success
    assert result.summary and "部分完成" in result.summary


def test_a_truncated_region_publishes_no_path():
    """A truncated frame can still be SAVED — and then the .sxm on disk looks
    complete to everything downstream. It must not reach scanned_paths."""
    result = BatchRegionsScan().execute(
        FakeCtx(wait_stopped_early=True), {"regions": _rj()})
    assert result.data["scanned_paths"] == []


def test_failed_regions_do_not_contribute_scanned_paths():
    """A rejected region could still carry an sxm_path from an earlier region's
    state; publishing it made a failed region look like it produced data."""
    ctx = FakeCtx(wait_timed_out=True)
    result = BatchRegionsScan().execute(ctx, {"regions": _rj()})
    assert result.data["scanned_paths"] == [], (
        f"failed regions published paths: {result.data['scanned_paths']}")


def test_region_geometry_preserved():
    ctx = FakeCtx()
    result = BatchRegionsScan().execute(ctx, {"regions": _rj()})
    regions = result.data["regions"]
    assert regions[0]["center_x_m"] == pytest.approx(1e-7)
    assert regions[1]["center_x_m"] == pytest.approx(-2e-7)


def test_metadata_confirm_gated():
    md = BatchRegionsScan().metadata()
    assert md.safety_level == SafetyLevel.CONFIRM
    assert md.name == "BatchRegionsScan"
    assert any(p.name == "regions" and p.required for p in md.parameters)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
