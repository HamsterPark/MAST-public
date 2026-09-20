"""Constant readout detection with independent synthetic sequences.

Bit-identical values or an extremely small relative spread should carry a
diagnostic note. A varying sequence is the negative control; too few samples
provide insufficient evidence. The raw statistics remain available to callers."""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.skills.builtins.capture_signal_buffer import (  # noqa: E402
    CaptureSignalBuffer,
)

#: Binary-exact synthetic baseline avoids roundoff in the zero-variance control.
_FROZEN = -(2.0 ** -24)


class _Rec:
    def __init__(self, value):
        self.return_value = value
        self.error = ""


class _Ctx:
    """Feeds a scripted sequence of readings, cycling if it runs out."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def safe_call(self, verb, *a, **kw):
        v = self._values[self._i % len(self._values)]
        self._i += 1
        return _Rec(("", b"\x00", [v]))

    def check_abort(self):
        return False


def _run(values, duration=0.12, poll_hz=400.0):
    return CaptureSignalBuffer().execute(
        _Ctx(values),
        {"channel": "z", "duration_s": duration, "poll_hz": poll_hz},
    )


# ════════════════════════════════════════════════════════════════════════════

def test_a_perfectly_constant_signal_is_flagged():
    res = _run([_FROZEN])
    assert res.success
    assert res.data["n_samples"] >= 8, "fixture collected too few samples"
    assert res.data["std"] == 0.0
    assert "anomaly" in res.data, (
        "The synthetic constant sequence must carry a diagnostic note."
    )
    note = res.data["anomaly"]
    assert "完全相同" in note
    assert "不要把这些数字当作测量结果使用" in note


def test_the_flag_names_plausible_causes():
    """A warning that does not say what to check is barely better than none."""
    note = _run([_FROZEN]).data["anomaly"]
    for cause in ("缓存", "未连接", "并未在采集"):
        assert cause in note, f"cause not mentioned: {cause}"


def test_a_varying_synthetic_signal_is_not_flagged():
    """An independently varying synthetic sequence must not produce a constant-readout alarm."""
    values = [_FROZEN * (1 + i * 1e-4) for i in range(64)]
    res = _run(values)
    assert res.data["std"] > 0
    assert "anomaly" not in res.data


def test_near_constant_is_also_flagged():
    """Exact equality is not the only frozen-readout shape; a relative spread
    below 1e-9 is equally impossible for a live piezo trace."""
    values = [_FROZEN * (1 + i * 1e-13) for i in range(64)]
    res = _run(values)
    assert "anomaly" in res.data
    assert "近乎恒定" in res.data["anomaly"]


def test_tiny_sample_counts_are_not_flagged():
    """With a handful of points, identical values carry no statistical weight —
    flagging them would be noise."""
    res = _run([_FROZEN], duration=0.01, poll_hz=200.0)
    if res.data["n_samples"] >= 8:
        pytest.skip("timing produced too many samples for this case")
    assert "anomaly" not in res.data


def test_the_numbers_themselves_are_still_reported():
    """The warning is an addition, not a replacement — a caller that knows the
    channel really is parked still gets its statistics."""
    res = _run([_FROZEN])
    assert res.data["mean"] == pytest.approx(_FROZEN)
    assert res.data["min"] == res.data["max"] == pytest.approx(_FROZEN)
    assert res.data["channel"] == "z" and res.data["unit"] == "m"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
