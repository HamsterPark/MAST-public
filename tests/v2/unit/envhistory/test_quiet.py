"""The quiet gate: is the instrument idle enough for this reading to be
"environment" rather than "measurement"?

Every branch gets a case, because the two wrong answers fail in opposite and
equally quiet ways: judging a scan "quiet" poisons the tunnelling-current trend
with topography, and judging an idle instrument "active" silently stops
recording the very thing the operator asked for.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import types

from mast.envhistory.quiet import (
    ACTIVE, NO_TUNNEL, QUIET, UNKNOWN,
    admits, classify, context_from_snapshots,
)


def ctx(**kw):
    base = {"ctx_zctrl_on": True, "ctx_scanning": False, "ctx_skill": "",
            "ctx_stale": False}
    base.update(kw)
    return base


def test_idle_tunnelling_is_quiet():
    assert classify(ctx()) == QUIET
    assert admits(QUIET) is True


def test_a_skill_holding_the_token_is_active():
    assert classify(ctx(ctx_skill="TipShaping")) == ACTIVE
    # Any skill, not just the suppression list the alert engine uses: for
    # "is this environment?", anything driving the instrument disqualifies it.
    assert classify(ctx(ctx_skill="GetBias")) == ACTIVE


def test_a_blank_skill_string_is_not_a_holder():
    assert classify(ctx(ctx_skill="   ")) == QUIET


def test_scanning_is_active():
    assert classify(ctx(ctx_scanning=True)) == ACTIVE


def test_feedback_off_means_no_tunnel_junction():
    assert classify(ctx(ctx_zctrl_on=False)) == NO_TUNNEL


def test_stale_snapshot_is_unknown_even_if_it_looks_idle():
    assert classify(ctx(ctx_stale=True)) == UNKNOWN


def test_missing_feedback_state_is_unknown_not_a_guess():
    assert classify(ctx(ctx_zctrl_on=None)) == UNKNOWN
    assert classify({}) == UNKNOWN
    assert classify(None) == UNKNOWN


def test_only_quiet_admits_readings():
    assert admits(QUIET) is True
    for other in (ACTIVE, NO_TUNNEL, UNKNOWN):
        assert admits(other) is False


def test_stale_wins_over_everything():
    """A stale snapshot cannot report a tip crash as 'no tunnel junction'."""
    assert classify(ctx(ctx_stale=True, ctx_zctrl_on=False)) == UNKNOWN


class _Snap:
    scan_running = False
    bias_v = 0.5
    setpoint_a = 2e-11
    z_pos_m = 1e-9
    z_controller_on = True
    stale = False


class _State:
    def snapshot(self):
        return _Snap()


def test_context_from_snapshots_matches_the_monitoring_key_names():
    """The same classify() serves the environment tick and the current-monitor
    segment stream, so both must produce the same key names."""
    out = context_from_snapshots(lambda: _State(), lambda: {"skill": "ScanFrame"})
    assert out["ctx_zctrl_on"] is True
    assert out["ctx_scanning"] is False
    assert out["ctx_bias_v"] == 0.5
    assert out["ctx_skill"] == "ScanFrame"
    assert classify(out) == ACTIVE


def test_context_survives_a_state_that_raises():
    def boom():
        raise RuntimeError("no pool")
    out = context_from_snapshots(boom, lambda: {})
    assert out["ctx_skill"] == ""
    assert classify(out) == UNKNOWN     # nothing known → do not pretend


def test_context_survives_a_lock_snapshot_that_raises():
    def boom():
        raise RuntimeError("lock gone")
    out = context_from_snapshots(lambda: _State(), boom)
    assert out["ctx_zctrl_on"] is True
    assert out["ctx_skill"] == ""       # unknown holder, not a crash
    assert classify(out) == QUIET


def test_context_handles_a_state_with_no_snapshot_method():
    out = context_from_snapshots(lambda: types.SimpleNamespace())
    assert classify(out) == UNKNOWN
