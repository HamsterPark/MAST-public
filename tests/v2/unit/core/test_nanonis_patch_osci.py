"""v2 unit test for the Osci1T_TimebaseGet wrapper-bug fix in nanonis_patch.

nanonis_spm v1.0.9 ships ``Osci1T_TimebaseGet`` with a copy-paste bug: it
sends the command name "Osci1T.TimebaseSet" (the SETTER) instead of
"Osci1T.TimebaseGet", so the available-timebases list a caller needs to choose
a sample rate is never delivered. mast.core.nanonis_patch monkey-patches it to
send the correct command. This test pins the fix.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/core/test_nanonis_patch_osci.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
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

import pytest


class _FakeNano:
    """Records quickSend calls so we can assert the command string."""

    def __init__(self):
        self.sent: list[tuple] = []

    def quickSend(self, cmd, body, body_types, return_types):
        self.sent.append((cmd, body, body_types, return_types))
        # Mimic a 2-timebase controller: (current_index, n, [dt0, dt1]).
        return ["", b"", [0, 2, [5e-5, 1e-4]]]


def test_patched_timebase_get_sends_correct_command():
    from mast.core import nanonis_patch

    fake = _FakeNano()
    out = nanonis_patch._patched_Osci1T_TimebaseGet(fake)

    assert len(fake.sent) == 1
    cmd, body, body_types, return_types = fake.sent[0]
    # The whole point of the fix: GET, not SET.
    assert cmd == "Osci1T.TimebaseGet"
    assert cmd != "Osci1T.TimebaseSet"
    assert body == []           # GET takes no arguments
    assert return_types == ["i", "i", "*f"]
    # And the recorded response is forwarded through unchanged.
    assert out[2] == [0, 2, [5e-5, 1e-4]]


def test_patch_is_applied_to_class_on_import():
    from nanonis_spm import Nanonis

    from mast.core import nanonis_patch

    # apply() runs on import; the class method is the patched one.
    assert Nanonis.Osci1T_TimebaseGet is nanonis_patch._patched_Osci1T_TimebaseGet


def test_revert_restores_original():
    from nanonis_spm import Nanonis

    from mast.core import nanonis_patch

    nanonis_patch.revert()
    try:
        assert Nanonis.Osci1T_TimebaseGet is nanonis_patch._original_Osci1T_TimebaseGet
    finally:
        nanonis_patch.apply()  # leave the patch in place for other tests
    assert Nanonis.Osci1T_TimebaseGet is nanonis_patch._patched_Osci1T_TimebaseGet


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
