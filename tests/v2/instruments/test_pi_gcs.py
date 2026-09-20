"""PI GCS driver (E-816 / E-861) — command/reply protocol behaviour.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/instruments/test_pi_gcs.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
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

import pytest

from mast.instruments.base import AxisConfig, InstrumentError
from mast.instruments.pi_gcs import PiGcsController, parse_gcs_value

from .conftest import FakeTransport


def _make(replies: list[bytes], *, model="E-861", channel="1",
          servo_on_connect=False) -> tuple[PiGcsController, FakeTransport]:
    ft = FakeTransport(replies=replies)
    ctrl = PiGcsController(
        transport=ft,
        model=model,
        servo_on_connect=servo_on_connect,
        axes=[AxisConfig(name="z", channel=channel, min_pos=0.0, max_pos=20.0, unit="mm")],
    )
    return ctrl, ft


class TestParseGcsValue:
    def test_axis_prefixed(self):
        assert parse_gcs_value("A=10.500000\n", "A") == "10.500000"

    def test_axis_digit(self):
        assert parse_gcs_value("1=0.003", "1") == "0.003"

    def test_bare_value_fallback(self):
        assert parse_gcs_value("42.0\n", "A") == "42.0"

    def test_missing_axis_raises(self):
        with pytest.raises(InstrumentError):
            parse_gcs_value("B=1.0\n", "A")

    def test_empty_reply_raises(self):
        with pytest.raises(InstrumentError):
            parse_gcs_value("   ", "A")


class TestPiGcs:
    def test_move_sends_mov_then_checks_err(self):
        replies = [
            b"",            # MOV (silent set command)
            b"0\n",         # ERR? after MOV
            b"1=5.000000\n",  # POS?
            b"1=1\n",       # ONT?
            b"1=1\n",       # FRF?
            b"1=5.000000\n",  # POS? (2nd status poll in wait loop)
            b"1=1\n",       # ONT?
            b"1=1\n",       # FRF?
        ]
        ctrl, ft = _make(replies)
        status = ctrl.axis("z").move_abs(5.0, wait=True, timeout=1.0)
        assert status.position == 5.0
        assert status.on_target is True
        assert ft.sent[0] == b"MOV 1 5.000000\n"
        assert ft.sent[1] == b"ERR?\n"

    def test_gcs_error_code_raises(self):
        replies = [b"", b"7\n"]  # ERR? → 7 (position out of limits, e.g.)
        ctrl, _ = _make(replies)
        with pytest.raises(InstrumentError, match="GCS error 7"):
            ctrl.axis("z").move_abs(5.0, wait=False)

    def test_stop_swallows_benign_error_10(self):
        replies = [b"", b"10\n"]  # STP → ERR? 10 "stopped by command"
        ctrl, _ = _make(replies)
        ctrl.axis("z").stop()  # must not raise

    def test_e816_has_no_homing(self):
        ctrl, _ = _make([], model="E-816", channel="A")
        with pytest.raises(InstrumentError, match="no homing"):
            ctrl.axis("z").home()

    def test_e861_home_sends_frf(self):
        replies = [
            b"", b"0\n",          # FRF + ERR?
            b"1=0.000000\n",      # POS?
            b"1=1\n",             # ONT?
            b"1=1\n",             # FRF? → referenced
        ]
        ctrl, ft = _make(replies)
        ctrl.axis("z").home(wait=True, timeout=1.0)
        assert ft.sent[0] == b"FRF 1\n"

    def test_e816_status_skips_frf_query(self):
        replies = [b"A=1.250000\n", b"A=1\n"]  # POS? + ONT? only
        ctrl, ft = _make(replies, model="E-816", channel="A")
        status = ctrl.axis("z").get_status()
        assert status.position == 1.25
        assert status.homed is None
        assert all(not s.startswith(b"FRF?") for s in ft.sent)

    def test_servo_on_connect_sends_svo(self):
        replies = [b"", b"0\n"]  # SVO + ERR?
        ft = FakeTransport(replies=replies)
        ctrl = PiGcsController(
            transport=ft, model="E-816", servo_on_connect=True,
            axes=[AxisConfig(name="p", channel="A", min_pos=0.0, max_pos=100.0)],
        )
        ctrl.connect()
        assert ft.sent[0] == b"SVO A 1\n"

    def test_no_reply_to_query_raises(self):
        replies = [b"", b""]  # MOV silent, then ERR? gives nothing
        ctrl, _ = _make(replies)
        with pytest.raises(InstrumentError, match="no reply"):
            ctrl.axis("z").move_abs(1.0, wait=False)

    def test_constructor_requires_port_or_transport(self):
        with pytest.raises(ValueError):
            PiGcsController()
