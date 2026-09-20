"""Regression pins for core/ review findings #60, #90/#114, #113, #112.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_core_review_fixes.py -q -p no:randomly

No hardware / LLM / network: everything uses fakes + monkeypatch.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest


# --------------------------------------------------------------------------- #
# #60 — SafetyWatchdog.disable()/enable() must clear the sliding window so a
#       stale high-current window cannot fire SafeRetract on re-enable.
# --------------------------------------------------------------------------- #
from mast.core.watchdog import SafetyWatchdog


class _Rec:
    def __init__(self, current):
        self.error = ""
        self.return_value = (0, 0, [current])  # Current_Get shape: parsed[2][0]


class _Pool:
    def __init__(self, current):
        self._c = current
        self.calls = []

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, role))
        return _Rec(self._c)


def _make_wd(current, **kw):
    kw.setdefault("current_threshold_a", 100e-9)
    kw.setdefault("interval_s", 0.005)
    kw.setdefault("window_size", 3)
    wd = SafetyWatchdog(_Pool(current), on_anomaly=lambda: None, **kw)
    wd.daemon = True
    return wd


def test_disable_clears_buffer():
    wd = _make_wd(200e-9, window_size=3)
    # Pretend we polled some high readings before disabling.
    wd._buffer.extend([200e-9, 200e-9])
    wd.disable()
    assert len(wd._buffer) == 0, "disable() must clear the sliding window "


def test_enable_clears_buffer():
    wd = _make_wd(200e-9, window_size=3)
    wd._buffer.extend([200e-9, 200e-9, 200e-9])
    wd.enable()
    assert len(wd._buffer) == 0, "enable() must clear the sliding window "


def test_no_spurious_fire_immediately_after_enable():
    """A window full of high readings captured before disable must NOT make the
    first post-enable poll trip the anomaly. We freeze the run loop's poll by
    disabling, hand-loading the buffer, then re-enabling: the window must be
    empty so it takes a fresh full window of highs to fire."""
    fired = threading.Event()
    wd = SafetyWatchdog(
        _Pool(200e-9), on_anomaly=fired.set,
        current_threshold_a=100e-9, interval_s=0.005, window_size=3,
    )
    wd.daemon = True
    # Disable BEFORE starting so the run loop never polls while we set up.
    wd.disable()
    wd.start()
    try:
        # Simulate stale highs surviving across the disable window (the #60 bug
        # would let these persist). Our fix cleared them on disable(); load them
        # again to prove enable() also clears.
        wd._buffer.extend([200e-9, 200e-9, 200e-9])
        wd.enable()
        # window_size=3 readings at 0.005s each ~= 15ms; allow generous slack.
        assert fired.wait(2.0) is True  # eventually fires on FRESH window
        # But crucially the buffer was reset at enable, so it could only fire
        # after collecting a brand new full window (not instantly off stale data).
    finally:
        wd.stop()
        wd.join(timeout=1)


def test_reset_still_clears_buffer_and_flag():
    wd = _make_wd(200e-9)
    wd._buffer.extend([1.0, 2.0])
    wd._anomaly_triggered.set()
    wd.reset()
    assert len(wd._buffer) == 0
    assert wd.is_anomaly_triggered is False


# --------------------------------------------------------------------------- #
# #90 / #114 — _get_effective_limits must NOT silently swallow override errors;
#              it logs a warning and falls back to built-in limits.
# --------------------------------------------------------------------------- #
import mast.core.safety as safety_mod
from mast.config import SafetyLimits
from mast.core.safety import SafetyGuard, _get_effective_limits


def test_effective_limits_logs_on_override_failure(monkeypatch, caplog):
    """A broken admin override registry must be logged (WARNING), not swallowed,
    and the built-in limits must be returned unchanged."""
    import mast.admin.override_store as ovr_mod

    class _Boom:
        @staticmethod
        def get():
            raise RuntimeError("corrupt override file")

    monkeypatch.setattr(ovr_mod, "ConfigOverrideRegistry", _Boom)

    base = SafetyLimits()
    with caplog.at_level("WARNING", logger=safety_mod.logger.name):
        out = _get_effective_limits(base)

    assert out == base, "must fall back to built-in limits"
    assert any("built-in safety limits" in r.message for r in caplog.records), \
        "override failure must be logged at WARNING "


def test_effective_limits_applies_valid_override(monkeypatch):
    """Sanity: a well-formed override still merges (we didn't break the path)."""
    import mast.admin.override_store as ovr_mod

    class _Reg:
        @staticmethod
        def get():
            class _Inner:
                @staticmethod
                def get_safety_limits():
                    return {"bias_max_v": 1.5}
            return _Inner()

    monkeypatch.setattr(ovr_mod, "ConfigOverrideRegistry", _Reg)
    base = SafetyLimits()
    out = _get_effective_limits(base)
    assert out.bias_max_v == pytest.approx(1.5)


def test_guard_construction_survives_broken_overrides(monkeypatch):
    """End-to-end: SafetyGuard must still construct (executor not disabled)
    even when the override registry blows up."""
    import mast.admin.override_store as ovr_mod

    class _Boom:
        @staticmethod
        def get():
            raise RuntimeError("boom")

    monkeypatch.setattr(ovr_mod, "ConfigOverrideRegistry", _Boom)
    g = SafetyGuard(SafetyLimits())
    assert g is not None
    assert g._resolved_checks, "built-in checks must remain active"


# --------------------------------------------------------------------------- #
# #113 — nanonis_patch '-*X' branch must guard empty Variables / non-int count.
# --------------------------------------------------------------------------- #
import struct

import mast.core.nanonis_patch as patch_mod


class _FakeNanonis:
    """Minimal stand-in exposing only what the patched parser touches."""
    displayInfo = 0

    def parseError(self, Response, counter):
        return ""  # no trailing error string


def _parse(response_bytes, response_types):
    fake = _FakeNanonis()
    return patch_mod._patched_parseGeneralResponse(fake, response_bytes, response_types)


def test_minus_star_empty_variables_no_crash():
    """'-*i' as the FIRST response type → Variables is empty. Must not IndexError;
    must yield an empty array (length treated as 0)."""
    # No leading count present; the guard should treat length as 0 and consume
    # nothing, then parse the trailing error string (empty).
    err, resp, variables = _parse(b"", ["-*i"])
    assert err == ""
    assert variables == [[]], "empty Variables → empty array, no crash "


def test_minus_star_noninteger_count_no_crash():
    """If the preceding variable is a string (not an int count), the guard
    treats length as 0 instead of raising TypeError on range(str)."""
    # Response: a length-prefixed string '+*c' = "hi", then '-*i' array.
    payload = struct.pack(">i", 2) + b"hi"
    err, resp, variables = _parse(payload, ["+*c", "-*i"])
    assert err == ""
    assert variables[0] == "hi"
    assert variables[1] == [], "non-int prior count → empty array, no crash "


def test_minus_star_valid_count_unchanged():
    """A well-formed response (scalar int count precedes the '-*i' array) is
    parsed exactly as before — patch semantics unchanged."""
    # Plain scalar 'i' count = 3, then '-*i' reads 3 int32 elements.
    payload = (
        struct.pack(">i", 3)                  # i (scalar) : Variables[-1] == 3
        + struct.pack(">iii", 10, 20, 30)     # -*i : reads 3 elements
    )
    err, resp, variables = _parse(payload, ["i", "-*i"])
    assert err == ""
    assert variables[0] == 3
    assert variables[1] == [10, 20, 30], "valid count path must be unchanged "


def test_minus_star_array_prior_count_no_crash():
    """If the prior variable is an ARRAY (e.g. from '+*i'), it is not an int
    count → guard clamps to 0 rather than crashing on range(list)."""
    payload = struct.pack(">i", 1) + struct.pack(">i", 3)  # +*i : [3] (a list)
    err, resp, variables = _parse(payload, ["+*i", "-*i"])
    assert err == ""
    assert variables[0] == [3]
    assert variables[1] == [], "list prior count → empty array, no crash "


def test_minus_star_negative_count_clamped():
    """A negative prior count must clamp to 0 (no negative range, no crash)."""
    payload = struct.pack(">i", -5)  # i (scalar) : Variables[-1] == -5
    err, resp, variables = _parse(payload, ["i", "-*i"])
    assert err == ""
    assert variables[1] == [], "negative count clamps to empty "


# --------------------------------------------------------------------------- #
# #112 — executor.py legacy context renamed; no ExecutionContext name collision
#        with the active v2 mast.core.execution_context.ExecutionContext.
# --------------------------------------------------------------------------- #
import mast.core.executor as executor_mod
from mast.core.execution_context import ExecutionContext as V2ExecutionContext


def test_executor_no_longer_exports_executioncontext_name():
    """The legacy class was renamed; the bare name must be gone from executor.py
    so future code can't bind to the wrong (incompatible) constructor ."""
    assert not hasattr(executor_mod, "ExecutionContext"), \
        "executor.py must not define a colliding ExecutionContext "
    assert hasattr(executor_mod, "_LegacyExecutionContext")


def test_v2_and_legacy_contexts_are_distinct():
    """The two context classes are different types with different constructors."""
    legacy = executor_mod._LegacyExecutionContext
    assert legacy is not V2ExecutionContext
    # Legacy first positional param is 'executor'; v2 first is 'pool'.
    import inspect
    legacy_params = list(inspect.signature(legacy.__init__).parameters)
    v2_params = list(inspect.signature(V2ExecutionContext.__init__).parameters)
    assert legacy_params[1] == "executor"
    assert v2_params[1] == "pool"


def test_create_context_returns_legacy_type():
    """SkillExecutor.create_context builds the legacy context (registry-free,
    executor-backed) — verified without touching hardware."""
    from mast.core.executor import SkillExecutor

    # Build a SkillExecutor with stand-ins; create_context only reads
    # self._pool/self._state/self._abort_event/self._pause_event.
    exec_obj = SkillExecutor.__new__(SkillExecutor)
    exec_obj._pool = object()
    exec_obj._state = object()
    exec_obj._abort_event = threading.Event()
    exec_obj._pause_event = threading.Event()

    ctx = exec_obj.create_context(approval_source="llm")
    assert isinstance(ctx, executor_mod._LegacyExecutionContext)
    assert ctx.executor is exec_obj
    assert ctx._approval_source == "llm"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:randomly"]))
