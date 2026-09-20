"""Adaptive AutoApproach / scan timeouts.

Two failures the operator hit:
  * AutoApproach's fixed wait timeout truncated a legitimately long coarse
    approach from far away (300–900 s was not enough).
  * FullScan estimated the scan-completion timeout from a HARDCODED 512 lines,
    so a higher-resolution or slower scan was cut off at ~88 %.

These tests pin the raised AutoApproach backstop and FullScan's estimation from
the REAL line count (read via Scan_BufferGet).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_scan_approach_timeouts.py -x -v
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

from dataclasses import dataclass, field

import pytest

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.builtins.approach import AutoApproach
from mast.skills.composite.full_scan import FullScan


# ── #123: AutoApproach wait-timeout backstop raised ──────────────────────

class TestAutoApproachTimeout:
    def test_default_wait_timeout_raised_to_30_min(self):
        # Was 300 s (and the operator found even 900 s too short). The wait loop
        # exits the instant the module reaches the setpoint, so a generous
        # backstop is nearly free and stops truncating far approaches.
        assert AutoApproach._DEFAULT_WAIT_TIMEOUT_S == 1800.0

    def test_param_default_matches_the_constant(self):
        meta = AutoApproach().metadata()
        sp = next(p for p in meta.parameters if p.name == "wait_timeout_s")
        assert sp.default == 1800.0
        # The description must warn against hand-lowering it.
        assert "backstop" in sp.description.lower()


# ── #148: FullScan estimates the timeout from the REAL line count ────────

class TestFullScanEstimator:
    def test_estimate_scales_with_lines_and_line_time(self):
        # A 512-line 1 s/line scan (fwd+bwd) needs ~1024 s of pure scan time; the
        # estimate must comfortably exceed that (headroom), never fall below.
        est = FullScan._estimate_wait_timeout_s(1.0, 512)
        assert est > 1024.0

    def test_estimate_grows_with_resolution(self):
        # The whole point: a 1024-line scan must get a LONGER timeout than 512.
        assert (FullScan._estimate_wait_timeout_s(1.0, 1024)
                > FullScan._estimate_wait_timeout_s(1.0, 512))

    def test_estimate_floored_at_300s(self):
        # A tiny fast scan still gets the 300 s floor (not a pathologically short
        # value that races the scan).
        assert FullScan._estimate_wait_timeout_s(0.1, 512) == 300.0

    def test_estimate_bad_line_count_falls_back_to_512(self):
        assert (FullScan._estimate_wait_timeout_s(1.0, None)
                == FullScan._estimate_wait_timeout_s(1.0, 512))
        assert (FullScan._estimate_wait_timeout_s(1.0, 0)
                == FullScan._estimate_wait_timeout_s(1.0, 512))

    def test_read_scan_lines_parses_buffer(self):
        @dataclass
        class Ctx:
            def safe_call(self, method, *a, role="main"):
                if method == "Scan_BufferGet":
                    # [num_channels, channel_indexes, pixels, lines]
                    return NanonisCallRecord(
                        method=method, args=a,
                        return_value=("", b"", [2, [0, 14], 256, 1024]))
                return NanonisCallRecord(method=method, args=a, return_value=None)
        assert FullScan._read_scan_lines(Ctx()) == 1024

    def test_read_scan_lines_none_on_failure(self):
        @dataclass
        class Ctx:
            def safe_call(self, method, *a, role="main"):
                return NanonisCallRecord(method=method, args=a, error="boom",
                                         return_value=None)
        assert FullScan._read_scan_lines(Ctx()) is None


# ── FullScan integration: line count actually drives the wait timeout ────

@dataclass
class _AdaptiveCtx:
    """Records the timeout_ms passed to WaitScanComplete; serves a canned
    Scan_BufferGet with ``lines`` and varying frame data (no crash)."""
    lines: int = 1024
    wait_timeout_ms: int | None = None
    run_log: list = field(default_factory=list)

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append(skill_name)
        if skill_name == "WaitScanComplete":
            self.wait_timeout_ms = params.get("timeout_ms")
            return SkillResult(skill_name=skill_name, success=True,
                               data={"timed_out": False, "polls": 1})
        return SkillResult(skill_name=skill_name, success=True, data={})

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        if method == "Scan_BufferGet":
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [2, [0, 14], 256, self.lines]))
        if method == "Scan_FrameDataGrab":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [0.0, 1.0, 2.0, 3.0]))
        return NanonisCallRecord(method=method, args=args, return_value=None)


def test_fullscan_wait_timeout_tracks_real_line_count():
    """No explicit wait_timeout_s → FullScan reads 1024 lines from the buffer
    and hands WaitScanComplete a timeout matching that, NOT the 512 assumption."""
    ctx = _AdaptiveCtx(lines=1024)
    res = FullScan().execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
        "line_time_s": 1.0,
    })
    assert res.success
    assert ctx.wait_timeout_ms is not None
    # Matches the estimator for 1024 lines (and exceeds the 512-line value).
    assert ctx.wait_timeout_ms == int(
        FullScan._estimate_wait_timeout_s(1.0, 1024) * 1000)
    assert ctx.wait_timeout_ms > int(
        FullScan._estimate_wait_timeout_s(1.0, 512) * 1000)
    # And it comfortably exceeds the bare 1024 s scan duration.
    assert ctx.wait_timeout_ms > 1024 * 1000


def test_fullscan_explicit_timeout_still_honoured():
    """An explicit wait_timeout_s wins over the estimate (no buffer read needed)."""
    ctx = _AdaptiveCtx(lines=1024)
    res = FullScan().execute(ctx, {
        "center_x_m": 0.0, "center_y_m": 0.0,
        "width_m": 50e-9, "height_m": 50e-9,
        "line_time_s": 1.0, "wait_timeout_s": 120.0,
    })
    assert res.success
    assert ctx.wait_timeout_ms == 120_000


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
