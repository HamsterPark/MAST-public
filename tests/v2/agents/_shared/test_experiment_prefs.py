"""Tests for the experiment default-parameter preferences .

Covers the live-read holder + pure render + middleware injection that feeds the
operator's preferred scan size / speed / setpoint / bias into the IC /
experiment_design agents' context (mast.agents._shared.experiment_prefs).

Synthetic only — no agent build, no model call. The middleware is exercised via
a tiny stand-in request object (it only touches ``request.system_message``).
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── path bootstrap (force MASTv2 mast to win over any v1 mast) ──────────
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

import types

import pytest
from langchain_core.messages import SystemMessage

from mast.agents._shared import experiment_prefs as ep


@pytest.fixture(autouse=True)
def _clear_holder():
    """Every test starts + ends with an empty holder (module global)."""
    ep.set_prefs(None)
    yield
    ep.set_prefs(None)


# ── sanitize ──────────────────────────────────────────────────────────
def test_sanitize_coerces_clamps_and_drops():
    out = ep.sanitize({
        "scan_size_nm": "50",          # str → float
        "scan_lines": 256.9,           # float → int
        "bias_v": 99,                  # clamp to 10
        "scan_speed_nm_s": -5,         # clamp to 0
        "setpoint_pa": 100,
        "notes": "  勿超过 1V  ",       # trimmed
        "junk_key": "x",               # dropped (unknown)
    })
    assert out["scan_size_nm"] == 50.0
    assert out["scan_lines"] == 256 and isinstance(out["scan_lines"], int)
    assert out["bias_v"] == 10.0
    assert out["scan_speed_nm_s"] == 0.0
    assert out["setpoint_pa"] == 100.0
    assert out["notes"] == "勿超过 1V"
    assert "junk_key" not in out


def test_sanitize_drops_blank_and_nonnumeric():
    out = ep.sanitize({"scan_size_nm": "", "setpoint_pa": "abc", "bias_v": None})
    assert out == {}


def test_sanitize_non_dict_is_empty():
    assert ep.sanitize(None) == {}
    assert ep.sanitize("nope") == {}
    assert ep.sanitize(42) == {}


def test_sanitize_drops_nan_inf():
    assert ep.sanitize({"bias_v": float("nan")}) == {}
    assert ep.sanitize({"scan_size_nm": float("inf")}) == {}


def test_sanitize_new_numeric_fields():
    out = ep.sanitize({"line_time_s": "0.5", "scan_angle_deg": 400})
    assert out["line_time_s"] == 0.5
    assert out["scan_angle_deg"] == 360.0  # clamped to +360


def test_sanitize_categorical_scan_direction():
    assert ep.sanitize({"scan_direction": "up"})["scan_direction"] == "up"
    assert ep.sanitize({"scan_direction": "down"})["scan_direction"] == "down"
    # invalid choice dropped
    assert "scan_direction" not in ep.sanitize({"scan_direction": "sideways"})
    # non-str dropped
    assert "scan_direction" not in ep.sanitize({"scan_direction": 5})


def test_editable_keys_include_new_fields():
    for k in ("line_time_s", "scan_angle_deg", "scan_direction"):
        assert k in ep.EDITABLE_KEYS


# ── holder round-trip ─────────────────────────────────────────────────
def test_holder_roundtrip_and_clear():
    ep.set_prefs({"scan_size_nm": 30, "setpoint_pa": 50})
    assert ep.get_prefs() == {"scan_size_nm": 30.0, "setpoint_pa": 50.0}
    # returns a COPY (mutating it does not corrupt the holder)
    got = ep.get_prefs()
    got["scan_size_nm"] = 999
    assert ep.get_prefs()["scan_size_nm"] == 30.0
    # None clears
    ep.set_prefs(None)
    assert ep.get_prefs() == {}


# ── format_prefs_block ─────────────────────────────────────────────────
def test_format_block_renders_labels_and_units():
    block = ep.format_prefs_block({"scan_size_nm": 50.0, "bias_v": 0.5, "notes": "常温"})
    assert "用户实验默认参数偏好" in block
    assert "扫描尺寸(边长): 50.0 nm" in block
    assert "偏压 bias: 0.5 V" in block
    assert "其他偏好: 常温" in block
    # honest framing: these are preferences, not bounds
    assert "SafetyLimits" in block


def test_format_block_renders_new_fields():
    block = ep.format_prefs_block(
        {"line_time_s": 0.5, "scan_angle_deg": 30.0, "scan_direction": "down"}
    )
    assert "每线时间: 0.5 s" in block
    assert "扫描旋转角: 30.0 °" in block
    # categorical rendered with the friendly display text, not the raw value
    assert "扫描方向: 从上往下 (down)" in block


def test_format_block_empty_is_blank():
    assert ep.format_prefs_block({}) == ""
    assert ep.format_prefs_block(None) == ""


# ── middleware injection ───────────────────────────────────────────────
def _fake_request(system_message):
    return types.SimpleNamespace(system_message=system_message)


def test_middleware_noop_when_holder_empty():
    mw = ep.ExperimentPrefsMiddleware()
    req = _fake_request(SystemMessage(content="BASE PROMPT"))
    out = mw._apply(req)
    # unchanged — no prefs set → nothing appended
    assert out.system_message.content == "BASE PROMPT"


def test_middleware_appends_block_to_existing_system_message():
    ep.set_prefs({"scan_size_nm": 40})
    mw = ep.ExperimentPrefsMiddleware()
    req = _fake_request(SystemMessage(content="BASE PROMPT"))
    out = mw._apply(req)
    assert out.system_message.content.startswith("BASE PROMPT")
    assert "扫描尺寸(边长): 40.0 nm" in out.system_message.content


def test_middleware_creates_system_message_when_absent():
    ep.set_prefs({"setpoint_pa": 120})
    mw = ep.ExperimentPrefsMiddleware()
    req = _fake_request(None)
    out = mw._apply(req)
    assert isinstance(out.system_message, SystemMessage)
    assert "电流设定点 setpoint: 120.0 pA" in out.system_message.content


def test_middleware_get_prefs_failure_is_noop():
    def boom():
        raise RuntimeError("holder read failed")

    mw = ep.ExperimentPrefsMiddleware(get_prefs_fn=boom)
    req = _fake_request(SystemMessage(content="BASE"))
    out = mw._apply(req)
    assert out.system_message.content == "BASE"  # never crashes a run


def test_middleware_async_hook_injects():
    """awrap_model_call is mandatory for the async ``python -m mast`` dispatch —
    a sync-only middleware raises NotImplementedError there. Driven via
    asyncio.run so the test needs no pytest-asyncio plugin config."""
    import asyncio

    ep.set_prefs({"bias_v": 1.0})
    mw = ep.ExperimentPrefsMiddleware()
    captured = {}

    async def handler(req):
        captured["sm"] = req.system_message
        return "RESP"

    async def drive():
        return await mw.awrap_model_call(
            _fake_request(SystemMessage(content="B")), handler
        )

    out = asyncio.run(drive())
    assert out == "RESP"
    assert "偏压 bias: 1.0 V" in captured["sm"].content
