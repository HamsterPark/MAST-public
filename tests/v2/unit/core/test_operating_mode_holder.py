"""The process-level operating-mode holder (mast.core.operating_mode).

This holder is what lets the NON-agent layers (vision publisher thread, buffer
service, composite skills) honour SAFE mode's "the tip is fine" contract. Its
whole safety story rests on one property: **unknown means do not override**. An
unbound holder — every test process, the headless pipeline, offline tools — must
behave exactly as the code did before the holder existed.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.core.operating_mode import (  # noqa: E402
    bind_mode_source,
    current_operating_mode,
    safe_mode_active,
)
from mast.core.types import OperatingMode  # noqa: E402


@pytest.fixture(autouse=True)
def _unbind_after():
    """Every test in this repo shares one process — a leaked binding would flip
    tip verdicts in unrelated tests. Fail-closed on cleanup."""
    yield
    bind_mode_source(None)


def test_unbound_is_unknown_and_never_overrides():
    bind_mode_source(None)
    assert current_operating_mode() is None
    assert safe_mode_active() is False


def test_safe_binding_activates():
    bind_mode_source(lambda: "safe")
    assert current_operating_mode() is OperatingMode.SAFE
    assert safe_mode_active() is True


@pytest.mark.parametrize("mode", ["semi", "auto"])
def test_semi_and_auto_never_override(mode):
    """SEMI wants REAL verdicts — its contract is "shallow shaping allowed,
    pulses to HITL", which only makes sense against a truthful tip assessment."""
    bind_mode_source(lambda: mode)
    assert safe_mode_active() is False


def test_enum_source_accepted():
    bind_mode_source(lambda: OperatingMode.SAFE)
    assert safe_mode_active() is True


def test_unknown_string_does_not_override():
    """OperatingMode.coerce sends junk to AUTO — which must NOT be SAFE."""
    bind_mode_source(lambda: "definitely-not-a-mode")
    assert current_operating_mode() is OperatingMode.AUTO
    assert safe_mode_active() is False


def test_source_returning_none_is_unknown():
    bind_mode_source(lambda: None)
    assert current_operating_mode() is None
    assert safe_mode_active() is False


def test_raising_source_is_swallowed_as_unknown():
    """Read happens on the vision publisher thread; an exception there must not
    propagate, and must not be mistaken for SAFE."""
    def _boom():
        raise RuntimeError("settings store is gone")

    bind_mode_source(_boom)
    assert current_operating_mode() is None
    assert safe_mode_active() is False


def test_rebind_replaces_previous_source():
    bind_mode_source(lambda: "safe")
    assert safe_mode_active() is True
    bind_mode_source(lambda: "auto")
    assert safe_mode_active() is False


def test_unbind_restores_no_override():
    bind_mode_source(lambda: "safe")
    assert safe_mode_active() is True
    bind_mode_source(None)
    assert safe_mode_active() is False


class _FakeStore:
    """Stands in for SettingsStore — only `get` is on the runtime's read path."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


class _FakeRuntime:
    """Borrows the REAL `CoreRuntime._current_operating_mode` so this test pins
    the contract between the two halves of the wiring, not a re-implementation
    of it: whatever that method returns must be something the holder parses."""

    def __init__(self, data):
        self._settings = _FakeStore(data)

    @property
    def _current_operating_mode(self):
        from mast.core.runtime import CoreRuntime
        return lambda: CoreRuntime._current_operating_mode(self)


@pytest.mark.parametrize("stored,expected_safe", [
    ({"autonomy_mode": "safe"}, True),
    ({"autonomy_mode": "semi"}, False),
    ({"autonomy_mode": "auto"}, False),
    ({}, False),                       # key absent → runtime fails open to "auto"
    ({"autonomy_mode": ""}, False),    # empty string → same
])
def test_runtime_mode_method_feeds_the_holder(stored, expected_safe):
    rt = _FakeRuntime(stored)
    bind_mode_source(rt._current_operating_mode)
    assert safe_mode_active() is expected_safe


def test_runtime_without_settings_store_is_not_safe():
    """`_settings = None` happens when SettingsStore init failed; the runtime
    fails open to "auto" and the holder must not read that as SAFE."""
    from mast.core.runtime import CoreRuntime

    class _NoSettings:
        _settings = None

    bind_mode_source(lambda: CoreRuntime._current_operating_mode(_NoSettings()))
    assert safe_mode_active() is False


def test_reads_are_live_not_cached():
    """The operator switches modes from the top bar mid-experiment; the holder
    must reflect it on the very next verdict, with no rebuild."""
    box = {"mode": "auto"}
    bind_mode_source(lambda: box["mode"])
    assert safe_mode_active() is False
    box["mode"] = "safe"
    assert safe_mode_active() is True
    box["mode"] = "auto"
    assert safe_mode_active() is False
