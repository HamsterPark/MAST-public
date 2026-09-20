"""Regression tests for mast.skills.builtins.bias.SetBias slew path + Get index.

Focus: the slew-ramp start MUST be the real present bias. If ``Bias_Get``
fails (or returns an unparseable response), ``SetBias`` must REFUSE to ramp
rather than silently assume a 0.0V start — a 0V start would step straight to
the first ramp target, the exact large instantaneous jump the slew rate is
meant to prevent.

Also pins the Nanonis return-triple index used by the Get* skills:
``return_value == (error_string, raw_bytes, parsed_list)`` and the real datum
lives at ``parsed[2][i]`` (NOT ``parsed[0]`` / ``parsed[1]``).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_set_bias_slew.py -x -v
"""
from __future__ import annotations

# ── path bootstrap BEFORE any mast.* imports ──
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

from mast.core.types import NanonisCallRecord
from mast.skills.builtins.bias import (
    GetBias,
    GetBiasCalibration,
    GetCurrent,
    SetBias,
)


# ── FakeCtx ─────────────────────────────────────────────────────────────────


@dataclass
class FakeCtx:
    """Canned per-method responses keyed by Nanonis method name.

    Each entry is ``{"return_value": (...), "error": ...}``.
    """

    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method,
                args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


# ── Get-skill index correctness (return-triple [2][i]) ──────────────────────


def test_get_bias_reads_parsed_index_2_0():
    ctx = FakeCtx(canned={"Bias_Get": {"return_value": ("", b"\x00", [3.14])}})
    result = GetBias().execute(ctx, {})
    assert result.success
    assert result.data["bias_v"] == pytest.approx(3.14)


def test_get_current_reads_parsed_index_2_0():
    ctx = FakeCtx(canned={"Current_Get": {"return_value": ("", b"\x00", [1.5e-9])}})
    result = GetCurrent().execute(ctx, {})
    assert result.success
    assert result.data["current_a"] == pytest.approx(1.5e-9)


def test_get_bias_calibration_reads_both_floats():
    # Bias.CalibrGet ResponseTypes == ["f", "f"] → calibration, offset.
    ctx = FakeCtx(
        canned={"Bias_CalibrGet": {"return_value": ("", b"\x00", [2.0, -0.25])}}
    )
    result = GetBiasCalibration().execute(ctx, {})
    assert result.success
    assert result.data["calibration"] == pytest.approx(2.0)
    assert result.data["offset"] == pytest.approx(-0.25)


# ── SetBias slew: start must be the REAL bias, never an assumed 0.0V ─────────


def test_slew_ramps_from_real_current_bias(monkeypatch):
    """Bias_Get reports 5.0V → ramp 5.0V→5.4V must step monotonically up,
    NOT jump down to a 0V-anchored first step."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(
        canned={
            "Bias_Get": {"return_value": ("", b"", [5.0])},
            "Bias_Set": {"return_value": ("", b"", [])},
        }
    )
    result = SetBias().execute(
        ctx, {"bias_v": 5.4, "slew_rate_v_per_s": 1.0}
    )
    assert result.success
    set_targets = [c[1][0] for c in ctx.calls if c[0] == "Bias_Set"]
    # 0.4V / (1.0 * 0.1) = 4 steps → linspace(5.0, 5.4, 5)[1:]
    assert set_targets == pytest.approx([5.1, 5.2, 5.3, 5.4])
    # No step is anywhere near a 0V-anchored ramp.
    assert min(set_targets) >= 5.0


def test_slew_aborts_when_bias_get_errors(monkeypatch):
    """Bias_Get hard error → SetBias must NOT issue any Bias_Set (no ramp from
    an assumed 0V start)."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(
        canned={
            "Bias_Get": {"error": "TCP timeout"},
            "Bias_Set": {"return_value": ("", b"", [])},
        }
    )
    result = SetBias().execute(
        ctx, {"bias_v": 5.0, "slew_rate_v_per_s": 1.0}
    )
    assert not result.success
    assert "slew" in (result.error or "").lower()
    assert "TCP timeout" in (result.error or "")
    # Critically: NO Bias_Set was attempted.
    assert not any(c[0] == "Bias_Set" for c in ctx.calls)


def test_slew_aborts_when_bias_get_unparseable(monkeypatch):
    """Bias_Get success but malformed payload (missing parsed datum) → abort,
    do not silently fall back to a 0V ramp start."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    # parsed list at index 2 is empty → parsed[2][0] would IndexError.
    ctx = FakeCtx(
        canned={
            "Bias_Get": {"return_value": ("", b"", [])},
            "Bias_Set": {"return_value": ("", b"", [])},
        }
    )
    result = SetBias().execute(
        ctx, {"bias_v": 5.0, "slew_rate_v_per_s": 1.0}
    )
    assert not result.success
    assert "unparseable" in (result.error or "").lower()
    assert not any(c[0] == "Bias_Set" for c in ctx.calls)


def test_slew_aborts_when_return_value_too_short(monkeypatch):
    """return_value lacks the index-2 parsed slot entirely → abort."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(
        canned={
            "Bias_Get": {"return_value": ("", b"")},  # len == 2, no [2]
            "Bias_Set": {"return_value": ("", b"", [])},
        }
    )
    result = SetBias().execute(
        ctx, {"bias_v": 5.0, "slew_rate_v_per_s": 1.0}
    )
    assert not result.success
    assert not any(c[0] == "Bias_Set" for c in ctx.calls)


# ── SetBias non-slew path unchanged ─────────────────────────────────────────


def test_instant_set_no_slew():
    ctx = FakeCtx(canned={"Bias_Set": {"return_value": ("", b"", [])}})
    result = SetBias().execute(ctx, {"bias_v": 2.0})
    assert result.success
    assert result.data["bias_v"] == pytest.approx(2.0)
    # No Bias_Get for the instant path.
    assert not any(c[0] == "Bias_Get" for c in ctx.calls)
    set_calls = [c for c in ctx.calls if c[0] == "Bias_Set"]
    assert len(set_calls) == 1


def test_slew_small_change_falls_through_to_instant(monkeypatch):
    """|Δv| < 1mV with slew set → single instant Bias_Set after Bias_Get
    (still needs a valid Bias_Get to decide the change is small)."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    ctx = FakeCtx(
        canned={
            "Bias_Get": {"return_value": ("", b"", [1.0])},
            "Bias_Set": {"return_value": ("", b"", [])},
        }
    )
    result = SetBias().execute(
        ctx, {"bias_v": 1.0005, "slew_rate_v_per_s": 1.0}
    )
    assert result.success
    set_calls = [c for c in ctx.calls if c[0] == "Bias_Set"]
    assert len(set_calls) == 1
    assert set_calls[0][1][0] == pytest.approx(1.0005)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
