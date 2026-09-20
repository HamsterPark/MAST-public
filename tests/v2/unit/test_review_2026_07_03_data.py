"""Regression pins for the 2026-07-03 review — data / records cluster.

  * read_sxm honours the per-channel Direction from DATA_INFO (a single-direction
    channel used to desync the byte offset and shift every later channel).
  * ExperimentLog.start_experiment ends the prior running experiment (no orphaned
    'running' rows).
  * PlanStore drives advance/pause/resume via the meta-tools' DB layer.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]


# ── read_sxm direction ───────────────────────────────────────────────────────
def _make_sxm(tmp_path, nx, ny, channels):
    """channels: list of (name, direction, n_frames)."""
    lines = [":SCAN_PIXELS:", f"{nx} {ny}", ":DATA_INFO:",
             "Channel\tName\tUnit\tDirection\tCalibration\tOffset"]
    for i, (name, direction, _n) in enumerate(channels):
        lines.append(f"{i+1}\t{name}\tm\t{direction}\t1\t0")
    header = "\r\n".join(lines) + "\r\n"
    blob = header.encode("utf-8") + b"\x1a\x04"
    # append frames: each (name, direction) → n_frames of nx*ny big-endian f4
    val = 0.0
    for (name, direction, nframes) in channels:
        for _f in range(nframes):
            for _p in range(nx * ny):
                blob += struct.pack(">f", val)
                val += 1.0
    p = tmp_path / "test.sxm"
    p.write_bytes(blob)
    return str(p)


def test_read_sxm_honours_single_direction_channel(tmp_path):
    from mast.io.nanonis_files import read_sxm
    # Channel A is single-direction "fwd" (1 frame); channel B is "both" (2).
    nx, ny = 2, 2
    path = _make_sxm(tmp_path, nx, ny, [("A", "fwd", 1), ("B", "both", 2)])
    out = read_sxm(path)
    chans = out["channels"]
    assert set(chans) == {"A", "B"}
    # A has only forward; B has forward + backward — and B's forward must start
    # at the RIGHT offset (frame index 1), not be shifted by an assumed A-backward.
    assert "forward" in chans["A"] and "backward" not in chans["A"]
    assert "forward" in chans["B"] and "backward" in chans["B"]
    # A.forward = pixels 0..3; B.forward = pixels 4..7 (no desync).
    assert chans["A"]["forward"].ravel().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert chans["B"]["forward"].ravel().tolist() == [4.0, 5.0, 6.0, 7.0]
    assert chans["B"]["backward"].ravel().tolist() == [8.0, 9.0, 10.0, 11.0]


# ── experiment_log scope switch (was: close-prior) ───────────────────────────
def test_start_experiment_moves_pointer_without_ending_prior(tmp_path):
    """Starting a second experiment moves the scope pointer and leaves the first
    one完全不变。

    SUPERSEDES ``test_start_experiment_ends_prior`` (2026-07-28). The original
    review finding was that every prior experiment was orphaned as active, and
    the fix at the time was to supersede the previous row on每次 start. The
    operator has since rejected the whole idea of a terminal state — 「没必要做
    归档。有的实验可能过了十年重启。」 — so the finding is now addressed the
    other way round: "which one is current" lives in ONE pointer instead of
    being inferred from ``status``, and nothing gets orphaned because nothing
    is ever closed.
    """
    from mast.logging.storage import ExperimentStorage
    from mast.logging.experiment_log import ExperimentLog
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    log = ExperimentLog(st)
    e1 = log.start_experiment("first", "goal1")
    e2 = log.start_experiment("second", "goal2")
    assert e1 != e2
    # 指针在新的那个上，而且是唯一的真相来源。
    assert log.current_experiment_id == e2
    assert st.get_active_scope()["experiment_id"] == e2
    # 被切走的那个一个字段都没动，而且随时可以切回来。
    rows = {r["id"]: r for r in st.list_experiments()}
    assert rows[e1]["end_time"] is None
    assert log.switch_experiment(e1).ok
    assert log.current_experiment_id == e1


# ── plan advance / pause / resume ────────────────────────────────────────────
def test_plan_lifecycle(tmp_path):
    from mast.planning.plan_store import (
        PlanStore, ExperimentPlan, PlanPhase, PlanStatus,
    )
    ps = PlanStore(str(tmp_path / "plans.db"), plans_dir=str(tmp_path / "plans"))
    plan = ExperimentPlan(
        plan_id="", name="overnight", goal="g",
        phases=[PlanPhase(id="p1", name="approach"),
                PlanPhase(id="p2", name="scan"),
                PlanPhase(id="p3", name="sts")],
        status=PlanStatus.APPROVED)
    pid = ps.save(plan)

    # advance p1 → running p2
    ps.update_progress(pid, 0, 0, phase_status="done")
    ps.update_progress(pid, 1, 0, phase_status="running")
    ps.update_status(pid, PlanStatus.RUNNING)
    reloaded = ps.load(pid)
    assert reloaded.current_phase_idx == 1
    assert reloaded.phases[0].status == "done"

    # pause + resume via get_active
    ps.update_status(pid, PlanStatus.PAUSED)
    active = ps.get_active()
    assert active is not None and active.plan_id == pid
    assert active.status == PlanStatus.PAUSED
    assert active.current_phase_idx == 1  # resume point survived
