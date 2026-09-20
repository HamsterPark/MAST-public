"""Pin the agent-layer SafetyGate global-bounds to core.safety (no drift).

A drifted copy had silently dropped the scan-extent caps (width_m/height_m) on
the live v2 agent path. This locks them together.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_safety_parity.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]


def test_agent_global_checks_are_core_global_checks():
    from mast.agents._shared import safety_mw
    from mast.core import safety
    # SAME object — divergence is structurally impossible.
    assert safety_mw._GLOBAL_CHECKS is safety._GLOBAL_CHECKS


def test_scan_extent_caps_present():
    from mast.agents._shared import safety_mw
    names = {c[0] for c in safety_mw._GLOBAL_CHECKS}
    # the caps that a "width_m=2 (2 metres!)" scan must hit
    assert "width_m" in names
    assert "height_m" in names
    for c in safety_mw._GLOBAL_CHECKS:
        if c[0] in ("width_m", "height_m"):
            assert c[2] == "scan_size_min_m" and c[3] == "scan_size_max_m"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
