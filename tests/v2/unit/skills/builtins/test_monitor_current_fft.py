"""v2 unit tests for mast.skills.builtins.monitor_current_fft.

Skill covered: MonitorCurrentFFT (AUTO) — polls Current_Get at high rate
for a window, then runs numpy.fft.rfft on the buffer (software FFT).

Nanonis methods exercised: Current_Get.

Tests use short duration_s windows (0.1 s) so the internal poll loop
terminates fast — no slow tests. duration_s=0.1 @ poll_hz=2000 reliably
collects > 4 samples (the FFT minimum) even on a slow box.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_monitor_current_fft.py -x -v
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
from mast.skills.builtins.monitor_current_fft import MonitorCurrentFFT


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


# A Current_Get payload: _decoded() unwraps the (err, b"", [value]) tuple.
def _cur(x: float = 1.0e-9):
    return ("", b"", [x])


# Standard fast-but-sufficient capture window for FFT (>4 samples).
_FAST_FFT = dict(duration_s=0.1, poll_hz=2000.0)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_monitor_current_fft_shape():
    tool = wrap_skill(MonitorCurrentFFT, make_provider())
    assert tool.name == "MonitorCurrentFFT"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert all(not v.is_required() for v in fields.values())
    for name in ("duration_s", "poll_hz", "window", "detrend", "output"):
        assert name in fields
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["duration_s"].annotation is str
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter，2026-08-04）
    assert fields["poll_hz"].annotation is str
    assert fields["window"].annotation is str
    assert fields["detrend"].annotation is bool
    assert fields["output"].annotation is str


def test_monitor_current_fft_skill_source():
    tool = wrap_skill(MonitorCurrentFFT, make_provider())
    assert tool.metadata["skill_source"].endswith(".monitor_current_fft")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_monitor_current_fft_executes():
    canned = {"Current_Get": {"return_value": _cur(1.0e-9)}}
    tool = wrap_skill(MonitorCurrentFFT, make_provider(canned))
    result = _invoke(tool, **_FAST_FFT)
    update = result.update
    assert update["executed_skills"] == ["MonitorCurrentFFT"]
    assert "error_log" not in update


def test_monitor_current_fft_polls_current_get():
    canned = {"Current_Get": {"return_value": _cur(2.5e-9)}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(MonitorCurrentFFT, capturing_provider)
    _invoke(tool, **_FAST_FFT)
    methods = {c[0] for c in instances[-1].calls}
    assert methods == {"Current_Get"}
    # FFT needs >= 4 samples; the 0.1 s window must clear that bar
    assert len(instances[-1].calls) >= 4


def test_monitor_current_fft_power_output():
    """output='power' takes the PSD branch and still succeeds."""
    canned = {"Current_Get": {"return_value": _cur(1.0e-9)}}
    tool = wrap_skill(MonitorCurrentFFT, make_provider(canned))
    result = _invoke(tool, duration_s=0.1, poll_hz=2000.0, output="power")
    assert result.update["executed_skills"] == ["MonitorCurrentFFT"]
    assert "error_log" not in result.update


def test_monitor_current_fft_rect_no_detrend():
    """window='rect' + detrend=False exercises the no-window branch."""
    canned = {"Current_Get": {"return_value": _cur(1.0e-9)}}
    tool = wrap_skill(MonitorCurrentFFT, make_provider(canned))
    result = _invoke(
        tool, duration_s=0.1, poll_hz=2000.0, window="rect", detrend=False,
    )
    assert result.update["executed_skills"] == ["MonitorCurrentFFT"]
    assert "error_log" not in result.update


def test_monitor_current_fft_hamming_window():
    """window='hamming' exercises the np.hamming branch."""
    canned = {"Current_Get": {"return_value": _cur(1.0e-9)}}
    tool = wrap_skill(MonitorCurrentFFT, make_provider(canned))
    result = _invoke(tool, duration_s=0.1, poll_hz=2000.0, window="hamming")
    assert result.update["executed_skills"] == ["MonitorCurrentFFT"]
    assert "error_log" not in result.update


# ── Boundary / error paths ────────────────────────────────────────────────────

def test_monitor_current_fft_too_few_samples():
    """All TCP polls error → < 4 samples → graceful failure result."""
    canned = {"Current_Get": {"return_value": None, "error": "TCP timeout"}}
    tool = wrap_skill(MonitorCurrentFFT, make_provider(canned))
    result = _invoke(tool, **_FAST_FFT)
    update = result.update
    assert update["executed_skills"] == ["MonitorCurrentFFT"]
    assert "error_log" in update
    assert "Too few samples" in str(result)


def test_monitor_current_fft_abort_stops_loop():
    """context.check_abort() True → loop ends immediately → fast failure."""

    @dataclass
    class AbortingCtx(FakeCtx):
        def check_abort(self) -> bool:
            return True

    canned = {"Current_Get": {"return_value": _cur(1.0e-9)}}
    tool = wrap_skill(MonitorCurrentFFT, lambda: AbortingCtx(canned=canned))
    # 10 s window but immediate abort → returns fast with too-few-samples
    result = _invoke(tool, duration_s=10.0, poll_hz=2000.0)
    update = result.update
    assert update["executed_skills"] == ["MonitorCurrentFFT"]
    assert "error_log" in update


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
