"""InstrumentState.apply_patch — write a successful skill's returned state fields
back to the cached snapshot so the NEXT skill's precondition sees them.

Fixes the real-rig loop: StartScan failed its z_controller_on precondition right
after ZControllerOnOff(True) because the cached snapshot was still 'unknown' until
the ~1s monitor refresh caught up; the agent then retried until the recursion
limit. Patching the cache on success closes that gap.
"""

from __future__ import annotations

from mast.core.state import InstrumentState
from mast.core.types import HardwareState, NanonisCallRecord


def _state() -> InstrumentState:
    # apply_patch only touches self._cache; skip __init__ (it needs a live pool).
    st = InstrumentState.__new__(InstrumentState)
    st._cache = HardwareState()
    return st


class _ErrPool:
    """A sim with missing modules: every monitor read errors → refresh() can't
    determine any field (all None). Optionally answer ZCtrl_StatusGet with a
    definitive code to test that a real read still supersedes the cache."""

    def __init__(self, zctrl_status_code: int | None = None):
        self._code = zctrl_status_code

    def safe_call(self, method, *args, role="main"):
        if method == "ZCtrl_StatusGet" and self._code is not None:
            return NanonisCallRecord(method=method, args=args, error="",
                                     return_value=("", b"", [self._code]))
        return NanonisCallRecord(method=method, args=args,
                                 error=f"Could not access module for {method}",
                                 return_value=None)


def test_apply_patch_writes_known_fields() -> None:
    st = _state()
    assert st.snapshot().z_controller_on is None  # unknown initially
    st.apply_patch(z_controller_on=True, bias_v=2.0)
    assert st.snapshot().z_controller_on is True  # next precondition will see True
    assert st.snapshot().bias_v == 2.0


def test_apply_patch_applies_false_but_skips_none() -> None:
    st = _state()
    st._cache.scan_running = True
    st._cache.current_a = 1e-9
    # False IS a real value (StopScan→scan_running=False) and must be applied
    st.apply_patch(scan_running=False, current_a=None)
    assert st.snapshot().scan_running is False
    # None must NOT clobber a known value
    assert st.snapshot().current_a == 1e-9


def test_apply_patch_ignores_unknown_keys() -> None:
    st = _state()
    # skill result data carries non-state keys (path, sxm_path, …) — ignore them,
    # never raise, never attach them to the state object.
    st.apply_patch(path="C:/x.sxm", sxm_path="y", z_controller_on=True)
    assert st.snapshot().z_controller_on is True
    assert not hasattr(st.snapshot(), "path")


def test_refresh_does_not_downgrade_patched_value_to_unknown() -> None:
    # THE real-rig dead loop's true root cause: apply_patch wrote True, but the
    # 1s background refresh rebuilt a fresh all-None state (sim's ZCtrl_StatusGet
    # errored) and replaced the cache wholesale → True clobbered back to None →
    # StartScan looped on z_controller_on forever. refresh() must NOT downgrade a
    # known value to None on a failed read.
    st = InstrumentState(_ErrPool())  # all reads error
    st.apply_patch(z_controller_on=True)
    assert st.snapshot().z_controller_on is True
    st.refresh()  # every read fails → fresh state has z_controller_on=None
    assert st.snapshot().z_controller_on is True  # survives — NOT downgraded


def test_refresh_definitive_read_still_supersedes_cache() -> None:
    # Safety: carry-forward must not MASK a real change. A definitive Off read
    # (ZCtrl_StatusGet code=1) must still update a cached True → False.
    st = InstrumentState(_ErrPool())
    st.apply_patch(z_controller_on=True)
    st._pool = _ErrPool(zctrl_status_code=1)  # 1 = Off (definitive)
    st.refresh()
    assert st.snapshot().z_controller_on is False  # real Off wins over cache
