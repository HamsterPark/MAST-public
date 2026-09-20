"""The vision pulse must follow the SCAN, not a clock.

Observed failure: after reading the first quarter of a frame, the monitor kept
analysing that same partial image and never picked up newly-scanned data,
processing only one full frame at the very end of the scan.

The monitor scheduled every milestone off ``elapsed / total_time`` — a Nanonis
time ESTIMATE. Nanonis's estimate routinely runs short, and when it does that
ratio SATURATES: ``frac`` pins at 0.999, all seven partial milestones burn inside
the first slice of the real scan, and the vision pulse then goes silent for the
rest of it. The operator saw the model chewing on the same ~quarter-frame for the
whole scan, then one full frame at the end. That is not a display bug; the
monitor really was not looking.

(#94 — "扫描进度显示扫描刚完成（511/512行）" — is the same bug's fingerprint:
round(0.999 × 512) = 511. Publishing an authoritative 100 % on scan-complete made
that number right without making the SCHEDULE right.)

A clock cannot know how far a scan has got. The scan buffer can: Nanonis
NaN-fills unacquired rows, so the NaN front IS the scan front. These tests drive
the monitor against a fake instrument whose real scan takes 4× the estimate, and
demand the pulse track the real thing.
"""
from __future__ import annotations

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

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.vision.module import (  # noqa: E402
    SegmentationResult, TipCoarseResult, TipFineResult,
)
from mast.vision.scan_monitor import ScanVisionMonitor  # noqa: E402
from mast.vision.seg_utils import encode_rle  # noqa: E402

LINES = 32
PIXELS = 32


class _Rec:
    def __init__(self, vals, error=""):
        self.error = error
        self.return_value = ("", b"", vals)


class NanonisLikePool:
    """A scan that really takes ``running_polls`` polls, and NaN-fills the rows it
    has not acquired yet — which is what a real Nanonis scan buffer looks like."""

    def __init__(self, running_polls: int = 40):
        self.running_polls = running_polls
        self.status_calls = 0
        self.frame_grabs = 0

    @property
    def real_frac(self) -> float:
        return min(1.0, self.status_calls / self.running_polls)

    def safe_call(self, method, *args, role="main"):
        if method == "Scan_StatusGet":
            self.status_calls += 1
            return _Rec([1 if self.status_calls <= self.running_polls else 0])
        if method == "Scan_FrameDataGrab":
            self.frame_grabs += 1
            return _Rec([0, "", LINES, PIXELS, self._frame(), 0])
        return _Rec([])

    def _frame(self) -> np.ndarray:
        """A live Nanonis scan buffer: acquired rows carry data, the rest are NaN
        (the NaN front IS the scan front). Reply shape is the real one —
        ``[name_len, name, rows, cols, data_2D, direction]`` with the data as a
        2-D element, not a flat list."""
        acquired = int(round(self.real_frac * LINES))
        img = np.full((LINES, PIXELS), np.nan, dtype=float)
        if acquired:
            img[:acquired] = np.linspace(
                0.0, 30.0, acquired * PIXELS).reshape(acquired, PIXELS)
        return img


class RecordingVision:
    """Records how COMPLETE each frame handed to the model actually was."""

    def __init__(self):
        self.frame_fracs: list[float] = []

    def set_scan_size_nm(self, nm):
        pass

    def _note(self, image):
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[0]
        filled = int(np.sum(~np.all(arr == 0.0, axis=1)))
        self.frame_fracs.append(filled / arr.shape[0])

    def assess_tip_coarse(self, image):
        self._note(image)
        return TipCoarseResult(label="good", confidence=0.8, embedding_sha="x",
                               scan_size_nm=5.0, tip_radius_nm=0.3,
                               sharpness_log10=-1.2)

    def assess_tip_fine(self, image):
        return TipFineResult(label="M0", top2=[("M0", 0.7)], is_usable=True,
                             morph="M0", switching=False, drift=False,
                             perturbation=False, multi_tip=False, n_tips=1.0)

    def segment(self, image, classes=None, *, level=None):
        seg = np.zeros((LINES, PIXELS), dtype=np.uint8)
        return SegmentationResult(mask_rle=encode_rle(seg), shape=(LINES, PIXELS),
                                  class_counts={"TERRACE": LINES * PIXELS},
                                  level=1, classes=["TERRACE"])


class StubBuffer:
    def __init__(self):
        self.seq = 0
        self.tips, self.regions, self.progress, self.events = [], [], [], []

    def next_seq(self):
        self.seq += 1
        return self.seq

    def put_tip_status(self, ts): self.tips.append(ts)
    def put_region(self, rm): self.regions.append(rm)
    def put_progress(self, p): self.progress.append(p)
    def emit_event(self, ev): self.events.append(ev)


def _clock():
    state = {"n": 0}

    def fn():
        v = float(state["n"])
        state["n"] += 1
        return v
    return fn


@pytest.fixture()
def run(tmp_path, monkeypatch):
    """Drive one monitor to completion. ``estimate_s`` is what Nanonis THINKS the
    scan will take; the fake instrument takes 4× that."""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr("mast.vision.scan_monitor._PROBE_EVERY_S", 0.0)

    def _go(*, running_polls=40, estimate_s=10.0):
        pool = NanonisLikePool(running_polls=running_polls)
        vision = RecordingVision()
        buf = StubBuffer()
        mon = ScanVisionMonitor(
            pool, scan_id="s1", buffer=buf, vision_getter=lambda: vision,
            poll_interval_s=0.0, scan_size_nm=5.0, total_lines=LINES,
            pixels=PIXELS, total_time_s=estimate_s, time_fn=_clock(),
        )
        mon._run()
        return pool, vision, buf
    return _go


# ════════════════════════════════════════════════════════════════════════
# THE regression
# ════════════════════════════════════════════════════════════════════════

def test_pulse_keeps_looking_when_the_time_estimate_runs_short(run):
    """The estimate says 10 s; the scan really takes 40. The pulse must follow
    the SCAN.

    Before the fix, ``frac`` reached 0.999 a quarter of the way in, every partial
    milestone fired against a ~25 %-complete frame, and nothing more was ever
    looked at until the end — the operator's report exactly.
    """
    _pool, vision, _buf = run(running_polls=40, estimate_s=10.0)

    partials = vision.frame_fracs[:-1]        # the last one is the 100 % frame
    assert len(partials) >= 5, f"the pulse went quiet: {vision.frame_fracs}"

    # 1. the frames really did get more complete as the scan progressed
    assert partials == sorted(partials), (
        f"the model was re-analysing stale frames: {partials}")

    # 2. the pulse did NOT burn its whole budget in the first quarter — the
    #    single sentence of the .
    late = [f for f in partials if f > 0.5]
    assert late, (
        f"every partial milestone fired inside the first half of the scan and "
        f"the pulse then went silent — #76 verbatim. frames={partials}")

    # 3. it looked at something close to a finished frame before the end
    assert max(partials) >= 0.8, (
        f"the last thing the model saw before completion was only "
        f"{max(partials):.0%} of a frame: {partials}")


def test_milestones_land_near_their_real_fractions(run):
    """A milestone labelled 50 % must fire when the SCAN is ~50 % done, not when
    a short clock says so."""
    _pool, vision, _buf = run(running_polls=40, estimate_s=10.0)

    partials = vision.frame_fracs[:-1]
    # thresholds are 0.125 … 0.875; each fired frame should be within a poll or
    # two of its own threshold
    expected = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875][:len(partials)]
    for got, want in zip(partials, expected):
        assert abs(got - want) < 0.15, (
            f"milestone {want:.0%} fired against a {got:.0%} frame — "
            f"the schedule is not tracking the scan: {partials}")


def test_progress_never_pins_below_completion(run):
    """#94: the bar sat at 511/512 because frac was capped at 0.999 by a
    saturated estimate. Progress must reach the real end, and never go
    backwards on the way (the re-anchor must not rewind the bar)."""
    _pool, _vision, buf = run(running_polls=40, estimate_s=10.0)

    idx = [p.line_idx for p in buf.progress]
    assert idx == sorted(idx), f"the progress bar went backwards: {idx}"
    assert idx[-1] == LINES, f"progress stopped at {idx[-1]}/{LINES}"


def test_an_over_long_estimate_also_tracks_the_scan(run):
    """The estimate can also be too LONG — then a clock-driven pulse fires only
    once or twice before the scan ends. The measurement fixes both directions."""
    _pool, vision, _buf = run(running_polls=20, estimate_s=200.0)

    partials = vision.frame_fracs[:-1]
    assert len(partials) >= 5, (
        f"an over-long estimate starved the pulse: {vision.frame_fracs}")


def test_full_frame_only_at_the_real_end(run):
    """The 100 % milestone still fires ONLY on a confirmed scan-complete, and it
    is the only frame that is actually whole."""
    _pool, vision, buf = run(running_polls=40, estimate_s=10.0)

    assert vision.frame_fracs[-1] == pytest.approx(1.0), (
        "the final milestone did not analyse a complete frame")
    assert sum(1 for f in vision.frame_fracs if f >= 0.999) == 1, (
        "more than one frame was whole — a partial milestone got a full frame")


# ════════════════════════════════════════════════════════════════════════
# The fallback: an instrument that does NOT NaN-fill
# ════════════════════════════════════════════════════════════════════════

class StaleFillPool(NanonisLikePool):
    """Backfills unacquired rows with STALE data instead of NaN — the front
    cannot be measured, so the monitor must fall back to the clock. It must still
    not pin: a saturated estimate is what silenced the pulse."""

    def _frame(self) -> np.ndarray:
        return np.nan_to_num(super()._frame(), nan=7.0)   # stale, never NaN


def test_unmeasurable_instrument_still_fires_every_milestone(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr("mast.vision.scan_monitor._PROBE_EVERY_S", 0.0)
    pool = StaleFillPool(running_polls=40)
    vision = RecordingVision()
    mon = ScanVisionMonitor(
        pool, scan_id="s1", buffer=StubBuffer(), vision_getter=lambda: vision,
        poll_interval_s=0.0, scan_size_nm=5.0, total_lines=LINES, pixels=PIXELS,
        total_time_s=10.0, time_fn=_clock(),
    )
    mon._run()
    # 7 partials + 1 final — none may be skipped just because we cannot measure
    assert len(vision.frame_fracs) == 8, (
        f"the clock fallback dropped milestones: {vision.frame_fracs}")
