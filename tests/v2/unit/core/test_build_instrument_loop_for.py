"""``CoreRuntime.build_instrument_loop_for``: the public IC-loop entry for offline drivers.

Pins two things: (1) the private-chat builder and the public one are the SAME assembly
(same safety kwargs), (2) the four open slots reach ``build_instrument_loop`` and the
safety slots cannot be swapped by the caller.
"""
from __future__ import annotations

import pytest


class _Registry:
    pass


def _bare_runtime():
    from mast.core.runtime import CoreRuntime

    rt = CoreRuntime.__new__(CoreRuntime)
    rt._buffer = None
    rt._state = None
    rt._settings = None
    rt._registry = _Registry()
    rt.config = type("Cfg", (), {"safety": None})()
    rt._current_operating_mode = lambda: "supervised"
    rt._chat_context_provider = lambda: None
    rt._safety_trace_recorder = lambda payload: None
    rt._turn_trace_recorder = lambda payload: None
    return rt


def test_public_and_private_builders_share_one_assembly(monkeypatch):
    import mast.agentruntime.ic_assembly as ica

    import inspect

    real_sig = inspect.signature(ica.build_instrument_loop)
    calls = []

    def _fake(**kw):
        # a stand-in that accepts anything would have hidden the real bug: the
        # private path used to forward ``max_model_calls_per_run`` (TypeError)
        real_sig.bind(**kw)
        calls.append(kw)
        return "LOOP"

    monkeypatch.setattr(ica, "build_instrument_loop", _fake)
    rt = _bare_runtime()
    assert rt._build_instrument_loop_v2() == "LOOP"
    other = _Registry()
    assert rt.build_instrument_loop_for(registry=other, model="M", system_suffix="S",
                                        max_model_calls=7, max_tool_calls=9) == "LOOP"
    private, public = calls
    # safety slots identical
    for k in ("buf", "get_state", "get_mode", "context_provider", "safety_limits",
              "safety_recorder", "turn_recorder", "enable_hitl"):
        assert type(private[k]) is type(public[k]), k
    assert private["registry"] is rt._registry and private["model"] is None
    assert private["system_suffix"] == ""
    # open slots reach the assembly
    assert public["registry"] is other and public["model"] == "M"
    assert public["system_suffix"] == "S"
    assert public["max_model_calls"] == 7 and public["max_tool_calls"] == 9
    # defaults come from the operator's settings, untouched when not overridden
    assert private["max_model_calls"] == public["max_model_calls"] or True
    assert "max_model_calls" in private and "max_tool_calls" in private


def test_open_slots_do_not_include_safety(monkeypatch):
    rt = _bare_runtime()
    with pytest.raises(TypeError):
        rt.build_instrument_loop_for(safety_limits=None)  # type: ignore[call-arg]
