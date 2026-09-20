"""2026-07-27 field forensics — the recording layer ( / /).

One session, three ways the record lied about it:

  the v2 store logged 22 actions, ALL 'succeeded', for a session the v1
          store recorded 27 actions and 5 failures for. The only v2 writer was
          a post_hook, and wrap_skill calls post_hooks only on success. The row
          it dropped that mattered most:
              `AutoApproach failed: The tip is NOT engaged; do not scan`
  every one of the 27 v1 rows read
              {"data": {}, "nanonis_calls": [], "state_before": null,
               "summary": null, "elapsed_s": 0.0}
          — including GetCurrent / GetZPosition, whose entire value is the
          number they return — and `approval_source` held the skill's DANGER
          LEVEL instead of who authorised it.
  9 scans across two BatchRegionsScan batches left 2 markers on the map,
          and all 47 markers in the table said `done` — including a batch where
          the safety gate had rejected 3 of 4 regions outright.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_forensics_20260727_records.py -q
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
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.types import (  # noqa: E402
    HardwareState,
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillMetadata,
    SkillResult,
)
from mast.logging.storage import ExperimentStorage  # noqa: E402


def _rt(storage, eid, **extra):
    """Duck-typed self for the recorder methods (they only touch these attrs)."""
    el = SimpleNamespace(current_experiment_id=eid, current_sample_id=None)
    return SimpleNamespace(_storage=storage, _experiment_log=el,
                           _v2_repos=None, _v2_eid=None, _state=None, **extra)


# ─────────────────────────────────────────────────────────────────────
# — the v1 action row must carry the skill's actual result
# ─────────────────────────────────────────────────────────────────────

def test_v1_action_keeps_data_summary_artifact_state_and_tcp_calls(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("NiI2")
    before = HardwareState(bias_v=1.0, current_a=1.2e-10)
    after = HardwareState(bias_v=-2.0, current_a=3.4e-10)
    calls = [NanonisCallRecord(method="Bias.Set", args=(-2.0,), elapsed_s=0.01)]

    CoreRuntime._record_v1_action(_rt(st, eid), {
        "skill": "BatchRegionsScan", "skill_version": "1.2.0",
        "params": {"regions": "P1,P2"}, "success": True, "error": "",
        "duration_ms": 4200, "elapsed_s": 4.2,
        "summary": "2/2 regions scanned",
        "artifact_path": r"D:\MAST-data\P2.sxm",
        "data": {"region_count": 2, "success_count": 2,
                 "scanned_paths": [r"D:\MAST-data\P1.sxm", r"D:\MAST-data\P2.sxm"]},
        "state_before": before, "state_after": after,
        "nanonis_calls": calls,
        "approval_source": "llm",
    })

    a = st.get_actions(eid)[0]
    # The four fields the forensics found empty on all 27 rows:
    assert a.result.data, "result.data must not be an empty dict"
    assert a.result.data["region_count"] == 2
    assert a.result.state_before is not None and a.result.state_before.bias_v == 1.0
    assert a.result.state_after is not None and a.result.state_after.bias_v == -2.0
    assert a.result.nanonis_calls and a.result.nanonis_calls[0].method == "Bias.Set"
    assert a.result.data["artifact_path"].endswith("P2.sxm")
    # …plus the ones that were being thrown away alongside them:
    assert a.result.summary == "2/2 regions scanned"
    assert a.result.elapsed_s == pytest.approx(4.2)
    # ACTION-level copies too — report.py / records_export.py / citations read these
    assert a.state_before is not None and a.nanonis_calls
    # approval_source is WHO, not HOW DANGEROUS
    assert a.approval_source == "llm"


def test_v1_action_approval_source_is_not_the_danger_level(tmp_path):
    """`approval_source` used to be handed `danger_level`, so the column read
    'CONFIRM'/'AUTO' — outside the v1 'auto'/'llm'/'human' vocabulary."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_v1_action(_rt(st, eid), {
        "skill": "SetBiasRamp", "params": {}, "success": True,
        "danger_level": "CONFIRM", "approval_source": "llm",
    })
    a = st.get_actions(eid)[0]
    assert a.approval_source == "llm"
    assert a.approval_source not in ("CONFIRM", "AUTO", "DANGEROUS")


def test_v1_action_read_skill_keeps_its_return_value(tmp_path):
    """GetCurrent's ONLY value is the number it read back."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_v1_action(_rt(st, eid), {
        "skill": "GetCurrent", "params": {}, "success": True,
        "data": {"current_a": 1.35e-13}, "duration_ms": 8,
    })
    assert st.get_actions(eid)[0].result.data["current_a"] == pytest.approx(1.35e-13)


def test_v1_action_long_traces_are_bounded_with_the_loss_declared(tmp_path):
    """A skill that returns 50k samples must not put 50k samples in the row —
    but the clipping has to be VISIBLE, not silent."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_v1_action(_rt(st, eid), {
        "skill": "TipShapeWithReadback", "params": {}, "success": True,
        "data": {"verdict": "no_change",
                 "z_samples_m": [1e-9] * 50000,
                 "current_samples_a": [5e-11] * 50000},
    })
    d = st.get_actions(eid)[0].result.data
    assert d["verdict"] == "no_change"          # the scalar a human reads survives
    assert len(d["z_samples_m"]) == 201         # 200 samples + the marker
    assert d["z_samples_m"][-1] == "<+49800 more>"


def test_v1_action_oversized_payload_is_truncated_not_emptied(tmp_path):
    """Over the whole-row budget: scalars are kept, the loss is declared, and
    the result is NEVER silently reset to `{}` — an empty dict is
    indistinguishable from 'the skill returned nothing', which is exactly the
    reading that made 27 field actions look contentless."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_v1_action(_rt(st, eid), {
        "skill": "GridSTS", "params": {}, "success": True,
        "data": dict({"n_points": 5},
                     **{f"curve_{i}": ["x" * 1500] for i in range(150)}),
    })
    d = st.get_actions(eid)[0].result.data
    assert d["n_points"] == 5
    assert "_truncated" in d and "exceeded" in d["_truncated"]
    assert "curve_0" not in d


def test_v1_action_survives_unserialisable_result_data(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_v1_action(_rt(st, eid), {
        "skill": "X", "params": {}, "success": True,
        "data": {"handle": object(), "n": 3},
    })
    d = st.get_actions(eid)[0].result.data
    assert d["n"] == 3 and isinstance(d["handle"], str)


def test_citations_cite_nanonis_when_tcp_calls_were_made(tmp_path):
    """`if action.nanonis_calls:` in citations/manager.py was permanently false
    on the agent path, so MAST never cited Nanonis for a session that had run
    hundreds of TCP commands."""
    from mast.citations.database import NANONIS_SPM
    from mast.citations.manager import CitationManager

    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_v1_action(_rt(st, eid), {
        "skill": "SetBias", "params": {"bias_v": -2.0}, "success": True,
        "nanonis_calls": [NanonisCallRecord(method="Bias.Set", args=(-2.0,))],
    })
    keys = {c.key for c in CitationManager(st).for_experiment(eid)}
    assert NANONIS_SPM.key in keys


def test_citations_read_tcp_calls_from_either_place(tmp_path):
    """Skills put their calls in ``SkillResult.nanonis_calls``; the action-level
    list is a copy the records layer makes. Reading only the copy is what made
    the check permanently false — records_export.py already reads both, and the
    citation manager now agrees with it."""
    from mast.citations.database import NANONIS_SPM
    from mast.citations.manager import CitationManager
    from mast.core.types import ActionRecord

    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    st.log_action(ActionRecord(
        experiment_id=eid, skill_name="SetBias",
        result=SkillResult(
            skill_name="SetBias", success=True,
            nanonis_calls=[NanonisCallRecord(method="Bias.Set", args=(-2.0,))]),
        nanonis_calls=[]))          # action-level copy absent
    keys = {c.key for c in CitationManager(st).for_experiment(eid)}
    assert NANONIS_SPM.key in keys


# ─────────────────────────────────────────────────────────────────────
# — the v2 store must record failures, with real params
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def live_v2(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.logging.v2.live import open_live_v2
    repos, eid = open_live_v2()
    assert repos is not None and eid
    return repos, eid


def test_v2_action_records_a_failure(live_v2):
    repos, eid = live_v2
    rt = SimpleNamespace(_v2_repos=repos, _v2_eid=eid, _active_thread_id="agents-abc")

    ok_id = CoreRuntime._record_v2_action(rt, {
        "skill": "StartScan", "params": {"bias_v": -1.5}, "success": True,
        "duration_ms": 900, "tool_call_id": "call_1"})
    bad_id = CoreRuntime._record_v2_action(rt, {
        "skill": "AutoApproach", "params": {"setpoint_a": 5e-11}, "success": False,
        "error": ("_phase_wait_complete failed: AutoApproach module stopped but "
                  "NO tunnelling current is present. The tip is NOT engaged; "
                  "do not scan."),
        "duration_ms": 61000, "tool_call_id": "call_2"})
    assert ok_id and bad_id

    rows = {r["action_type"]: r for r in repos.actions.for_experiment(eid)}
    assert set(rows) == {"StartScan", "AutoApproach"}
    assert rows["StartScan"]["status"] == "succeeded"
    # The row a success-only ledger cannot hold:
    assert rows["AutoApproach"]["status"] == "failed"
    assert "tip is NOT engaged" in rows["AutoApproach"]["error"]
    # …and the columns that were NULL / '{}' on all 22 field rows:
    assert rows["AutoApproach"]["duration_ms"] == 61000
    assert rows["AutoApproach"]["tool_call_id"] == "call_2"
    assert rows["AutoApproach"]["thread_id"] == "agents-abc"
    assert rows["StartScan"]["params_json"] != "{}"
    # the schema's hot-filter generated columns finally resolve
    assert rows["StartScan"]["param_bias_v"] == pytest.approx(-1.5)
    assert rows["AutoApproach"]["param_setpoint_a"] == pytest.approx(5e-11)


def test_v2_action_registers_the_files_the_skill_really_wrote(live_v2, tmp_path):
    """`scan_files` held 0 rows for a session that produced 9 .sxm, so the only
    surviving copy of those paths was a chat transcript that was itself
    truncated."""
    repos, eid = live_v2
    rt = SimpleNamespace(_v2_repos=repos, _v2_eid=eid, _active_thread_id=None)
    real = tmp_path / "R1.sxm"
    real.write_bytes(b"NANONIS SXM PAYLOAD")
    ghost = tmp_path / "never_written.sxm"          # the agent named it; nobody wrote it

    aid = CoreRuntime._record_v2_action(rt, {
        "skill": "BatchRegionsScan", "params": {}, "success": True,
        "data": {"scanned_paths": [str(real), str(ghost)]}})

    files = repos.scan_files.for_action(aid)
    assert [f["current_path"] for f in files] == [str(real)]
    assert files[0]["format_kind"] == "sxm" and files[0]["size_bytes"] == 19
    assert len(files[0]["sha256"]) == 64


def test_v2_action_of_a_failed_skill_registers_nothing(live_v2, tmp_path):
    repos, eid = live_v2
    rt = SimpleNamespace(_v2_repos=repos, _v2_eid=eid, _active_thread_id=None)
    f = tmp_path / "partial.sxm"
    f.write_bytes(b"x")
    aid = CoreRuntime._record_v2_action(rt, {
        "skill": "StartScan", "params": {}, "success": False, "error": "aborted",
        "artifact_path": str(f)})
    assert repos.scan_files.for_action(aid) == []


def test_v2_action_no_repos_is_a_silent_noop():
    assert CoreRuntime._record_v2_action(
        SimpleNamespace(_v2_repos=None, _v2_eid=None), {"skill": "X"}) is None


# ── the seam itself: wrap_skill must record a FAILED skill ──

class _FailingSkill:
    """Minimal BaseSkill-shaped stub whose execute() reports a failure."""

    def metadata(self):
        return SkillMetadata(
            name="AutoApproach", version="1.0.0", safety_level=SafetyLevel.CONFIRM,
            description="stub", parameters=[ParameterSpec(
                name="setpoint_a", type="float", required=False, default=5e-11,
                description="")])

    def validate_params(self, params):
        return []

    def execute(self, ctx, params):
        return SkillResult(
            skill_name="AutoApproach", success=False,
            error="The tip is NOT engaged; do not scan.",
            data={"engaged": False},
            nanonis_calls=[NanonisCallRecord(method="AutoApproach.Open")],
        )


def test_wrap_skill_records_failures_and_hook_only_runs_on_success():
    """The records write used to live inside the post_hook, which wrap_skill
    calls ONLY on success — so a failed skill produced no record at all."""
    from mast.agents._shared.skill_adapter import wrap_skill

    seen: list[dict] = []
    hook_calls: list[str] = []

    tool = wrap_skill(
        _FailingSkill, lambda: SimpleNamespace(),
        post_hook=lambda name, data, ctx: hook_calls.append(name),
        recorder=lambda p: (seen.append(p), "action-id-1")[1],
    )
    tool.func(setpoint_a=5e-11, tool_call_id="tc1")

    assert len(seen) == 1, "a failed skill must still reach the recorder"
    p = seen[0]
    assert p["success"] is False
    assert "NOT engaged" in p["error"]
    assert p["data"] == {"engaged": False}
    assert p["nanonis_calls"] and p["nanonis_calls"][0].method == "AutoApproach.Open"
    assert p["approval_source"] == "llm"        # not the "CONFIRM" danger level
    assert hook_calls == [], "post_hook stays success-gated (it renders figures)"


def test_wrap_skill_hands_the_action_id_to_the_post_hook():
    """Recording happens BEFORE the hook so the hook attaches its artifacts to
    the real action row instead of opening a second one for the same call."""
    from mast.agents._shared.skill_adapter import wrap_skill

    class _OkSkill(_FailingSkill):
        def execute(self, ctx, params):
            return SkillResult(skill_name="AutoApproach", success=True,
                               data={"engaged": True})

    got: list = []
    tool = wrap_skill(
        _OkSkill, lambda: SimpleNamespace(),
        post_hook=lambda name, data, ctx: got.append(
            getattr(ctx, "records_action_id", None)),
        recorder=lambda p: "action-id-1",
    )
    tool.func(tool_call_id="tc1")
    assert got == ["action-id-1"]


# ─────────────────────────────────────────────────────────────────────
# — one marker per region, with the region's own status
# ─────────────────────────────────────────────────────────────────────

def _batch_payload(success_flags):
    """A BatchRegionsScan result shaped like batch_regions_scan.aggregate()."""
    regions = []
    for i, ok in enumerate(success_flags):
        r = {"index": i + 1, "label": f"R{i + 1}", "success": ok,
             "center_x_m": 1.0e-6 + i * 1e-7, "center_y_m": 1.4e-6,
             "width_m": 1e-7, "height_m": 1e-7}
        if not ok:
            r["error"] = ("configure: center_y_m = 1.7031e-06 violates global "
                          "safety maximum 1.5e-06")
        else:
            r["sxm_path"] = rf"D:\MAST-data\R{i + 1}.sxm"
        regions.append(r)
    return {
        "skill": "BatchRegionsScan", "success": True,
        "params": {"regions": "…"},
        "data": {"region_count": len(regions), "regions": regions,
                 "success_count": sum(1 for r in regions if r["success"])},
    }


def test_every_region_gets_its_own_marker_with_its_own_status(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    # The real batch #1: region D scanned, A/B/C rejected by the safety gate.
    CoreRuntime._record_map_marker(_rt(st, eid), _batch_payload([False, False, False, True]))

    ms = st.get_markers(experiment_id=eid)
    assert len(ms) == 4, "4 regions must leave 4 markers, not 1"
    assert sorted(m["status"] for m in ms) == ["done", "failed", "failed", "failed"]
    failed = [m for m in ms if m["status"] == "failed"]
    assert all(m["kind"] == "scan" and m["skill_name"] == "BatchRegionsScan"
               for m in ms)
    assert "violates global safety maximum" in failed[0]["meta"]["error"]
    # distinct positions — not four copies of the composite's final position
    assert len({(m["x_m"], m["y_m"]) for m in ms}) == 4


def test_single_position_skill_still_gets_exactly_one_marker(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_map_marker(_rt(st, eid), {
        "skill": "BiasSpectr", "success": True,
        "params": {"x_m": 1.0e-6, "y_m": 1.4e-6}, "data": {"points": 200}})
    ms = st.get_markers(experiment_id=eid)
    assert len(ms) == 1 and ms[0]["kind"] == "sts" and ms[0]["status"] == "done"


def test_failed_single_skill_marker_is_failed(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    CoreRuntime._record_map_marker(_rt(st, eid), {
        "skill": "BiasSpectr", "success": False,
        "params": {"x_m": 1.0e-6, "y_m": 1.4e-6}, "data": {}})
    assert st.get_markers(experiment_id=eid)[0]["status"] == "failed"


def test_region_without_a_position_is_skipped_not_misplaced(tmp_path):
    """A marker at the wrong place is worse than no marker."""
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    payload = _batch_payload([True, True])
    payload["data"]["regions"][0].pop("center_x_m")
    CoreRuntime._record_map_marker(_rt(st, eid), payload)
    ms = st.get_markers(experiment_id=eid)
    assert len(ms) == 1 and ms[0]["x_m"] == pytest.approx(1.1e-6)


def test_non_finite_region_coordinate_is_rejected(tmp_path):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    eid = st.create_experiment("E")
    payload = _batch_payload([True])
    payload["data"]["regions"][0]["center_x_m"] = float("nan")
    CoreRuntime._record_map_marker(_rt(st, eid), payload)
    assert st.get_markers(experiment_id=eid) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
