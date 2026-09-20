"""Active Z-controller injection.

A rig can define MULTIPLE Z controllers (e.g. "Current", "log Current", "df")
with exactly one active. Before this fix the active one was reachable ONLY via
the on-demand GetZCtrlList skill, so the agent looped several turns on "Z
feedback still closed" during 进针 and only discovered — by manually querying —
that the active controller was "log Current". These tests pin that the active
controller now rides in HardwareState (read in InstrumentState.refresh via
ZCtrl_CtrlListGet) and in the live-state block on every LLM call.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_active_zcontroller_injection.py -x -v
"""
from __future__ import annotations

# ── path bootstrap (must come BEFORE any mast.* import) ──
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

from dataclasses import dataclass

import pytest

from mast.agents._shared.live_state_mw import format_live_state_block
from mast.core.state import InstrumentState
from mast.core.types import HardwareState, NanonisCallRecord


# A rig with three Z controllers, "log Current" (index 1) active.
_CTRL_NAMES = ["Current", "log Current", "df"]
_CTRL_ACTIVE = 1


@dataclass
class FakePool:
    """Minimal ConnectionPool stub. ``ctrl_list_ok=False`` simulates a failed
    ZCtrl_CtrlListGet read (to exercise the carry-forward path)."""
    ctrl_list_ok: bool = True

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        canned = {
            "Bias_Get": ("", b"", [0.5]),
            "ZCtrl_StatusGet": ("", b"", [1]),  # 1 = Off (feedback open)
            "ZCtrl_SetpntGet": ("", b"", [1e-10]),
            "Current_Get": ("", b"", [5e-12]),
            "ZCtrl_ZPosGet": ("", b"", [1.5e-7]),
            "FolMe_XYPosGet": ("", b"", [0.0, 0.0]),
            "ZCtrl_LimitsGet": ("", b"", [3e-7, 0.0]),
            "Scan_StatusGet": ("", b"", [0]),
            "Scan_FrameGet": ("", b"", [0.0, 0.0, 1e-7, 1e-7, 0.0]),
        }
        if method == "ZCtrl_CtrlListGet":
            if not self.ctrl_list_ok:
                # error string non-empty → refresh() skips populating it
                return NanonisCallRecord(
                    method=method, args=args, error="link down",
                    return_value=None)
            # Variables == [list_size, num_controllers, [names...], active_index]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=("", b"", [3, 3, list(_CTRL_NAMES), _CTRL_ACTIVE]),
            )
        ret = canned.get(method, ("", b"", None))
        return NanonisCallRecord(method=method, args=args, return_value=ret)


# ── _parse_ctrl_list (pure) ──────────────────────────────────────────────

def test_parse_ctrl_list_extracts_names_and_active_index():
    names, idx = InstrumentState._parse_ctrl_list(
        [3, 3, ["Current", "log Current", "df"], 1])
    assert names == ["Current", "log Current", "df"]
    assert idx == 1


def test_parse_ctrl_list_leading_ints_are_not_mistaken_for_active_index():
    # The two ints BEFORE the names list must be ignored; only an int AFTER the
    # names counts as the active index.
    names, idx = InstrumentState._parse_ctrl_list([5, 5, ["a", "b"], 0])
    assert names == ["a", "b"]
    assert idx == 0


def test_parse_ctrl_list_bad_shape_returns_empty():
    assert InstrumentState._parse_ctrl_list(None) == ([], 0)
    assert InstrumentState._parse_ctrl_list([]) == ([], 0)
    assert InstrumentState._parse_ctrl_list([1, 2, 3]) == ([], 0)


# ── refresh() populates the active controller ────────────────────────────

def test_refresh_populates_active_z_controller():
    state = InstrumentState(FakePool()).refresh()
    assert state.z_controller_names == _CTRL_NAMES
    assert state.z_controller_index == _CTRL_ACTIVE
    assert state.z_controller_name == "log Current"


def test_refresh_carries_forward_on_failed_ctrllist_read():
    """A momentary ZCtrl_CtrlListGet miss must NOT blank the active controller
    out of the cached state — it is carried forward from the last good read."""
    inst = InstrumentState(FakePool(ctrl_list_ok=True))
    inst.refresh()  # seeds the cache with "log Current"
    # Now the list read starts failing; other reads still succeed.
    inst._pool = FakePool(ctrl_list_ok=False)
    state = inst.refresh()
    assert state.z_controller_name == "log Current"
    assert state.z_controller_names == _CTRL_NAMES


# ── live-state block rendering ───────────────────────────────────────────

def test_live_state_block_shows_active_controller():
    state = HardwareState(
        z_controller_on=False,
        z_controller_name="log Current",
        z_controller_index=1,
        z_controller_names=list(_CTRL_NAMES),
    )
    block = format_live_state_block(state)
    assert "Active Z controller" in block
    assert "log Current" in block          # names the active one
    assert "index 1 of 3" in block          # shows which of how many
    assert "df" in block and "Current" in block  # lists the alternatives


def test_live_state_block_no_active_controller_line_when_absent():
    # No active-controller data → no such line (back-compat: the block is
    # otherwise unchanged for rigs with a single/unknown controller).
    state = HardwareState(bias_v=0.5, z_controller_on=True)
    block = format_live_state_block(state)
    assert "Active Z controller" not in block


def test_live_state_block_names_the_recovery_tools():
    # The guidance must point at the REAL skills so the agent can act on it.
    state = HardwareState(z_controller_name="log Current", z_controller_index=0,
                          z_controller_names=["log Current"])
    block = format_live_state_block(state)
    assert "GetZCtrlList" in block
    assert "SetActiveZController" in block


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
