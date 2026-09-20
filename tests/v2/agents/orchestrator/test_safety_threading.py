"""Orchestrator → instrument_control safety-stack threading (review fix).

The multi-agent (orchestrator) path used to build the instrument_control agent
WITHOUT handing down get_state / safety_limits / the admin override registry, so
on that path:

  * SafetyGateMiddleware's Layer-2 state-precondition check (z_controller_off →
    withdraw-before-coarse-approach, scan_not_running) was a silent no-op (it had
    no live state to read), while the MANUAL executor path enforced it; and
  * admin-tightened SafetyLimits overrides (a smaller bias/current/Z/scan cap the
    operator saved in the admin GUI) were dropped — code defaults were used.

build() now accepts get_state / safety_limits / override_registry and threads the
EFFECTIVE limits (code defaults merged with admin overrides) plus get_state into
instrument_control.build(). These tests pin that wiring WITHOUT an LLM / network:
they monkeypatch instrument_control.graph.build to capture the kwargs the
orchestrator passes it.

Run from repo root::

    PYTHONPATH=MASTv2 .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/orchestrator/test_safety_threading.py -q -p no:cacheprovider
"""
from __future__ import annotations

# ── path bootstrap (robust walk-up; matches sibling orchestrator tests) ──
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
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.config import SafetyLimits
from mast.core.types import HardwareState
from mast.agents.orchestrator.graph import _effective_safety_limits, build


# ─────────────────────────────────────────────────────────────────────
# _effective_safety_limits — admin SafetyLimits override merge
# ─────────────────────────────────────────────────────────────────────

class _FakeRegistry:
    """Minimal ConfigOverrideRegistry stand-in: only get_safety_limits()."""

    def __init__(self, overrides: dict | None, *, raise_on_get: bool = False):
        self._overrides = overrides
        self._raise = raise_on_get

    def get_safety_limits(self) -> dict:
        if self._raise:
            raise RuntimeError("corrupt safety_limits.json")
        return self._overrides or {}


class TestEffectiveSafetyLimits:
    def test_no_registry_passes_limits_through(self):
        lim = SafetyLimits(bias_max_v=3.0)
        assert _effective_safety_limits(lim, None) is lim

    def test_no_registry_none_limits_stays_none(self):
        # None limits + no registry → None (IC build then uses its own default).
        assert _effective_safety_limits(None, None) is None

    def test_admin_override_tightens_limit(self):
        reg = _FakeRegistry({"bias_max_v": 2.0, "setpoint_max_a": 5e-9})
        eff = _effective_safety_limits(SafetyLimits(), reg)
        assert eff is not None
        assert eff.bias_max_v == 2.0
        assert eff.setpoint_max_a == 5e-9
        # untouched fields keep code defaults
        assert eff.bias_min_v == SafetyLimits().bias_min_v

    def test_admin_override_merges_onto_none_base(self):
        # No explicit safety_limits but a registry with overrides → merge onto a
        # fresh default SafetyLimits so the admin cap still lands.
        reg = _FakeRegistry({"bias_max_v": 1.5})
        eff = _effective_safety_limits(None, reg)
        assert eff is not None
        assert eff.bias_max_v == 1.5

    def test_empty_override_returns_base_defaults(self):
        reg = _FakeRegistry({})
        eff = _effective_safety_limits(None, reg)
        assert eff is not None
        assert eff.bias_max_v == SafetyLimits().bias_max_v

    def test_corrupt_override_fails_open_to_supplied_limits(self):
        # A throwing registry must NOT crash the build; the supplied limits are
        # passed through unchanged (fail-open mirrors core.safety).
        lim = SafetyLimits(bias_max_v=4.0)
        reg = _FakeRegistry(None, raise_on_get=True)
        assert _effective_safety_limits(lim, reg) is lim


# ─────────────────────────────────────────────────────────────────────
# build() threads get_state + effective safety_limits into build_ic
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def _captured_ic(monkeypatch):
    """Patch instrument_control.graph.build to capture kwargs; returns a dict."""
    captured: dict = {}

    import mast.agents.instrument_control.graph as ic_graph

    def _fake_build(buf, **kwargs):
        captured["buf"] = buf
        captured.update(kwargs)
        # Return any object that LangGraph can add as a node. A bare callable is
        # fine for StateGraph.add_node (it treats it as a node function); we never
        # invoke the graph in these tests, only build it.
        def _node(state):  # pragma: no cover — never invoked
            return {}
        return _node

    monkeypatch.setattr(ic_graph, "build", _fake_build)
    return captured


def _ctx_provider():
    # instrument_control's build only needs SOME callable; our fake build ignores it.
    return object()


class TestBuildThreadsSafetyKnobs:
    def test_get_state_and_effective_limits_reach_build_ic(self, _captured_ic):
        def _get_state() -> HardwareState:  # pragma: no cover — never called here
            return HardwareState(
                bias_v=0.1, current_a=1e-9, z_pos_m=5e-7,
                z_controller_on=True, scan_running=False, setpoint_a=10e-12,
            )

        reg = _FakeRegistry({"bias_max_v": 2.5})
        build(
            buf=None,
            supervisor_model=None,
            context_provider=_ctx_provider,
            include_agents=("instrument_control",),
            enable_hitl=False,
            get_state=_get_state,
            safety_limits=SafetyLimits(),
            override_registry=reg,
        )
        # get_state threaded straight through
        assert _captured_ic.get("get_state") is _get_state
        # safety_limits is the MERGED effective object (admin override applied)
        eff = _captured_ic.get("safety_limits")
        assert eff is not None
        assert eff.bias_max_v == 2.5

    def test_defaults_thread_none_get_state_and_no_merge(self, _captured_ic):
        # No knobs passed (offline tests / library users) → IC gets get_state=None
        # and safety_limits=None (its own default), no crash.
        build(
            buf=None,
            supervisor_model=None,
            context_provider=_ctx_provider,
            include_agents=("instrument_control",),
            enable_hitl=False,
        )
        assert _captured_ic.get("get_state") is None
        assert _captured_ic.get("safety_limits") is None

    def test_corrupt_registry_does_not_break_build(self, _captured_ic):
        reg = _FakeRegistry(None, raise_on_get=True)
        build(
            buf=None,
            supervisor_model=None,
            context_provider=_ctx_provider,
            include_agents=("instrument_control",),
            enable_hitl=False,
            safety_limits=SafetyLimits(bias_max_v=7.0),
            override_registry=reg,
        )
        # fail-open: the supplied limits reach build_ic unchanged
        eff = _captured_ic.get("safety_limits")
        assert eff is not None
        assert eff.bias_max_v == 7.0


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
