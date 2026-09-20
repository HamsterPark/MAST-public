"""InstrumentProfileMiddleware injection tests.

Unlike ExperimentPrefsMiddleware, this ALWAYS injects (mechanism knowledge is
always relevant to 进/退针), so there is no empty-holder no-op case.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/agents/_shared/test_instrument_profile_mw.py -x -v
"""
from __future__ import annotations

# ── path bootstrap ──
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

import pytest
from langchain_core.messages import SystemMessage

from mast.core import instrument_profile as ip
from mast.agents._shared.instrument_profile_mw import InstrumentProfileMiddleware


@pytest.fixture(autouse=True)
def _clean_holder():
    ip.set_persist_sink(None)
    ip.set_profile({})
    yield
    ip.set_persist_sink(None)
    ip.set_profile({})


def _fake_request(system_message):
    return types.SimpleNamespace(system_message=system_message)


def test_middleware_always_injects_even_when_empty():
    """Mechanism knowledge is always relevant — no empty-holder no-op."""
    mw = InstrumentProfileMiddleware()
    req = _fake_request(SystemMessage(content="BASE PROMPT"))
    out = mw._apply(req)
    assert out.system_message.content.startswith("BASE PROMPT")
    assert "退针" in out.system_message.content
    assert "dI/dV" in out.system_message.content


def test_middleware_creates_system_message_when_absent():
    mw = InstrumentProfileMiddleware()
    req = _fake_request(None)
    out = mw._apply(req)
    assert isinstance(out.system_message, SystemMessage)
    assert "instrument_profile" in out.system_message.content


def test_middleware_reflects_calibration():
    ip.set_calibration(1.5e-3, bias_v=0.5, setpoint_a=1e-10)
    mw = InstrumentProfileMiddleware()
    out = mw._apply(_fake_request(SystemMessage(content="B")))
    assert "1.500 mV" in out.system_message.content


def test_middleware_render_failure_is_noop():
    def boom():
        raise RuntimeError("holder read failed")

    mw = InstrumentProfileMiddleware(get_profile_fn=boom)
    req = _fake_request(SystemMessage(content="BASE"))
    out = mw._apply(req)
    assert out.system_message.content == "BASE"   # never crashes a run


def test_middleware_async_hook_injects():
    import asyncio

    mw = InstrumentProfileMiddleware()
    captured = {}

    async def handler(req):
        captured["sm"] = req.system_message
        return "RESP"

    async def drive():
        return await mw.awrap_model_call(
            _fake_request(SystemMessage(content="B")), handler)

    out = asyncio.run(drive())
    assert out == "RESP"
    assert "退针" in captured["sm"].content


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
