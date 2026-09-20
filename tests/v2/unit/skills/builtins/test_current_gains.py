"""``GetCurrentGains``: the preamp range, which nothing could read before.

``SetCurrentGain`` has always been there; the readback had not. Without it "widen the range
before dropping the setpoint" is not a step anybody can take — the range is unknown, so it is
either changed blind or not changed at all. Lateral manipulation needs tens of nanoamps while
imaging runs at ten, and a setpoint above the range reads back saturated: the Z loop never sees
it reach the target and extends until the tip hits the surface.
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

import pytest  # noqa: E402

from mast.skills.builtins.current import GetCurrentGains  # noqa: E402

GAINS = ["1E6", "1E7", "1E8", "1E9", "1E10", "1E11"]


class Ctx:
    def __init__(self, reply=None, error=""):
        self.reply, self.error = reply, error

    def safe_call(self, name, *args):
        outer = self

        class R:
            error = outer.error
            return_value = outer.reply
        return R()


def _reply(index: int):
    """What Current.GainsGet answers: gains and filters with their selected indices."""
    return ("", b"...", [44, 6, GAINS, index, 27, 3, ["none", "1 kHz"], 0])


@pytest.mark.parametrize("index,full_scale", [(3, 1e-8), (2, 1e-7), (5, 1e-10)])
def test_the_full_scale_follows_from_the_gain_name(index, full_scale):
    """The gain is a transimpedance in V/A and the DAC swings ±10 V, so the range is
    10 V / gain. A lower index is a WIDER range, which is the direction manipulation needs."""
    res = GetCurrentGains().execute(Ctx(_reply(index)), {})
    assert res.success
    assert res.data["gain_index"] == index
    assert res.data["gain"] == GAINS[index]
    assert res.data["full_scale_a"] == pytest.approx(full_scale, rel=1e-9)
    assert res.data["gains"] == GAINS


def test_a_wire_error_is_a_failure_not_a_guess():
    res = GetCurrentGains().execute(Ctx(None, error="timed out"), {})
    assert res.success is False and "timed out" in (res.error or "")


def test_a_reply_it_cannot_read_leaves_the_range_unknown():
    """No full scale is "we do not know", which a caller can act on. A made-up one is not."""
    res = GetCurrentGains().execute(Ctx(("", b"", [1, 2])), {})
    assert res.success
    assert res.data.get("full_scale_a") is None


def test_it_is_a_read_and_touches_nothing():
    meta = GetCurrentGains().metadata()
    assert meta.category.value.lower() == "read"
    assert meta.parameters == []
