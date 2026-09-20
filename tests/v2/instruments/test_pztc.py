"""PZTC nm003 driver — Modbus RTU framing + real vendor register map.

Register map is the ground truth from pztc_nm001通信协议MODBUS_V1.0.xlsx:
holding stride 16 (setpoint_place=18, setpoint_speed=20), coil stride 16
(close_loop_move=8, enc_look_zero=3), discrete-input stride 2
(zero_status=0, place_achieve=1), input-reg stride 3 (place_counter=0).
Channels are 0-based.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/instruments/test_pztc.py -x -v
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

from mast.instruments.base import AxisConfig, InstrumentError, InstrumentUnavailable
from mast.instruments.pztc_nm003 import (
    ModbusExceptionError,
    PztcController,
    PztcRegisterMap,
    build_rtu_frame,
    crc16_modbus,
    parse_rtu_response,
)

from .conftest import FakeTransport


def _crc(body: bytes) -> bytes:
    crc = crc16_modbus(body)
    return body + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


class PztcFake:
    """Stateful SerialLike that answers by Modbus function code, so behaviour
    tests don't depend on the exact number of status polls.

    ``position`` (int32 pulses) is returned for 2-register input reads;
    ``place_achieve`` / ``zero_status`` bits for discrete-input reads (odd
    address = place_achieve, even = zero_status, matching stride-2 layout for
    ch0/ch1). Writes are echoed and recorded in ``writes``."""

    def __init__(self, *, slave: int = 1, position: int = 0,
                 place_achieve: bool = True, zero_status: bool = True,
                 word_big: bool = True):
        self.slave = slave
        self.position = position
        self.place_achieve = place_achieve
        self.zero_status = zero_status
        self.word_big = word_big
        self.sent: list[bytes] = []
        self.writes: list[tuple] = []  # (func, addr, payload_bytes)
        self.closed = False

    def _i32_regs(self) -> bytes:
        u = self.position & 0xFFFFFFFF
        hi, lo = (u >> 16) & 0xFFFF, u & 0xFFFF
        a, b = (hi, lo) if self.word_big else (lo, hi)
        return a.to_bytes(2, "big") + b.to_bytes(2, "big")

    def transact(self, payload, *, read_size=None, read_until=None, timeout=None):
        p = bytes(payload)
        self.sent.append(p)
        func = p[1]
        body = p[2:-2]
        addr = int.from_bytes(body[0:2], "big")
        if func in (0x03, 0x04):
            count = int.from_bytes(body[2:4], "big")
            data = self._i32_regs() if count == 2 else b"\x00\x00" * count
            frame = bytes([self.slave, func, 2 * count]) + data
        elif func == 0x02:
            bit = self.place_achieve if (addr % 2 == 1) else self.zero_status
            frame = bytes([self.slave, func, 1, 1 if bit else 0])
        elif func in (0x05, 0x06):
            self.writes.append((func, addr, body[2:4]))
            frame = bytes([self.slave, func]) + body
        elif func == 0x10:
            self.writes.append((func, addr, body[5:]))
            frame = bytes([self.slave, func]) + body[0:4]
        else:
            frame = bytes([self.slave, func]) + body
        return _crc(frame)

    def close(self):
        self.closed = True


def _ctrl(fake: PztcFake, *, channel: int = 0, counts_per_unit: float = 1.0,
          register_map: PztcRegisterMap | None = None) -> PztcController:
    return PztcController(
        transport=fake, slave=fake.slave,
        register_map=register_map,
        axes=[AxisConfig(name="delay", channel=channel, min_pos=-1e9,
                         max_pos=1e9, unit="um")],
        counts_per_unit=counts_per_unit, inter_frame_gap_s=0.0,
    )


def _writes(fake, func):
    return [w for w in fake.writes if w[0] == func]


# ── framing (unchanged core) ──────────────────────────────────────────────
class TestModbusFraming:
    def test_crc16_reference_vector(self):
        assert crc16_modbus(bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01])) == 0x0A84

    def test_build_frame_appends_crc_little_endian(self):
        f = build_rtu_frame(1, 3, bytes([0, 0, 0, 1]))
        assert f == bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x84, 0x0A])

    def test_parse_roundtrip(self):
        body = bytes([0x01, 0x03, 0x02, 0x12, 0x34])
        assert parse_rtu_response(_crc(body), slave=1, func=0x03) == bytes([0x02, 0x12, 0x34])

    def test_parse_rejects_bad_crc(self):
        frame = bytearray(_crc(bytes([0x01, 0x03, 0x02, 0x12, 0x34])))
        frame[-1] ^= 0xFF
        with pytest.raises(InstrumentError, match="CRC"):
            parse_rtu_response(bytes(frame), slave=1, func=0x03)

    def test_parse_rejects_wrong_slave(self):
        with pytest.raises(InstrumentError, match="slave"):
            parse_rtu_response(_crc(bytes([0x02, 0x03, 0x02, 0x12, 0x34])), slave=1, func=0x03)

    def test_parse_decodes_exception_frame(self):
        with pytest.raises(ModbusExceptionError, match="illegal data address"):
            parse_rtu_response(_crc(bytes([0x01, 0x83, 0x02])), slave=1, func=0x03)

    def test_parse_rejects_short_frame(self):
        with pytest.raises(InstrumentError, match="short"):
            parse_rtu_response(b"\x01\x03\x00", slave=1, func=0x03)


# ── register map (vendor ground truth, 0-based channels) ──────────────────
class TestRegisterMap:
    def test_default_addresses(self):
        m = PztcRegisterMap()
        assert m.setpoint_place == 18 and m.setpoint_speed == 20
        assert m.close_loop_move == 8 and m.enc_look_zero == 3
        assert m.place_counter == 0 and m.place_achieve == 1 and m.zero_status == 0

    def test_per_object_strides(self):
        m = PztcRegisterMap()
        # holding stride 16, coil stride 16, discrete stride 2, input stride 3
        assert m.holding_addr(m.setpoint_place, 0) == 18
        assert m.holding_addr(m.setpoint_place, 1) == 34
        assert m.holding_addr(m.setpoint_place, 2) == 50
        assert m.coil_addr(m.close_loop_move, 2) == 40
        assert m.discrete_addr(m.place_achieve, 2) == 5
        assert m.input_addr(m.place_counter, 2) == 6

    def test_from_dict_override(self):
        m = PztcRegisterMap.from_dict({"word_order_big": False, "bogus": 1})
        assert m.word_order_big is False
        assert m.setpoint_place == 18  # untouched default


# ── motion behaviour ──────────────────────────────────────────────────────
class TestPztcMotion:
    def test_move_abs_frame_sequence_ch0(self):
        fake = PztcFake(position=100, place_achieve=True)
        ctrl = _ctrl(fake)
        status = ctrl.axis("delay").move_abs(100.0, wait=True, timeout=1.0)
        assert status.position == 100.0
        assert status.on_target is True

        # setpoint written as int32 (2 regs) via fn 0x10 at holding addr 18
        wm = _writes(fake, 0x10)
        assert wm and wm[0][1] == 18
        assert wm[0][2] == (100).to_bytes(4, "big", signed=True)
        # closed_loop_move coil ON via fn 0x05 at coil addr 8
        wc = _writes(fake, 0x05)
        assert wc and wc[0][1] == 8 and wc[0][2] == b"\xff\x00"
        # place_counter read via fn 0x04 at input addr 0
        assert any(s[1] == 0x04 and int.from_bytes(s[2:4], "big") == 0 for s in fake.sent)
        # place_achieve read via fn 0x02 at discrete addr 1
        assert any(s[1] == 0x02 and int.from_bytes(s[2:4], "big") == 1 for s in fake.sent)

    def test_channel_1_strides(self):
        fake = PztcFake(position=0, place_achieve=True)
        ctrl = _ctrl(fake, channel=1)
        ctrl.axis("delay").move_abs(5.0, wait=False)
        wm = _writes(fake, 0x10)
        wc = _writes(fake, 0x05)
        assert wm[0][1] == 34   # setpoint_place ch1 = 18 + 16
        assert wc[0][1] == 24   # close_loop_move ch1 = 8 + 16

    def test_get_position_reads_input_int32(self):
        fake = PztcFake(position=-250)
        ctrl = _ctrl(fake)
        assert ctrl.axis("delay").get_position() == -250.0
        assert fake.sent[0][1] == 0x04  # function 04 input register

    def test_on_target_reflects_place_achieve(self):
        fake = PztcFake(position=10, place_achieve=False)
        ctrl = _ctrl(fake)
        ax = ctrl.axis("delay")
        ax.move_abs(10.0, wait=False)          # sets _last_target so status reads the bit
        st = ax.get_status()
        assert st.on_target is False and st.moving is True

    def test_wait_times_out_when_never_achieved(self):
        fake = PztcFake(position=0, place_achieve=False)
        ctrl = _ctrl(fake)
        ax = ctrl.axis("delay")
        ax.POLL_INTERVAL_S = 0.001
        from mast.instruments.base import MotionTimeout
        with pytest.raises(MotionTimeout):
            ax.move_abs(5.0, wait=True, timeout=0.02)

    def test_counts_per_unit_scaling(self):
        fake = PztcFake(position=500, place_achieve=True)
        ctrl = _ctrl(fake, counts_per_unit=10.0)   # 10 pulses/µm
        ctrl.axis("delay").move_abs(50.0, wait=False)
        assert _writes(fake, 0x10)[0][2] == (500).to_bytes(4, "big", signed=True)

    def test_word_order_little(self):
        # low word at lower address: value = hi<<16 | lo, regs = [lo, hi]
        fake = PztcFake(position=(1 << 16) | 2, word_big=False)
        ctrl = _ctrl(fake, register_map=PztcRegisterMap(word_order_big=False))
        assert ctrl.axis("delay").get_position() == float((1 << 16) | 2)

    def test_stop_disables_close_loop_move_coil(self):
        fake = PztcFake()
        ctrl = _ctrl(fake)
        ctrl.axis("delay").stop()
        wc = _writes(fake, 0x05)
        assert wc[0][1] == 8 and wc[0][2] == b"\x00\x00"   # coil 8 OFF = abort

    def test_home_triggers_look_zero_and_polls_zero_status(self):
        fake = PztcFake(zero_status=True)
        ctrl = _ctrl(fake)
        ctrl.axis("delay").home(wait=True, timeout=1.0)
        # enc_look_zero coil (addr 3) written ON
        wc = _writes(fake, 0x05)
        assert wc[0][1] == 3 and wc[0][2] == b"\xff\x00"
        # zero_status discrete input (addr 0) polled via fn 0x02
        assert any(s[1] == 0x02 and int.from_bytes(s[2:4], "big") == 0 for s in fake.sent)

    def test_read_discrete_input_parses_bit(self):
        fake = PztcFake(place_achieve=True, zero_status=False)
        ctrl = _ctrl(fake)
        # discrete addr 1 (odd) -> place_achieve True; addr 0 (even) -> zero_status False
        assert ctrl.read_discrete_input(1) is True
        assert ctrl.read_discrete_input(0) is False


# ── error handling ────────────────────────────────────────────────────────
class TestPztcErrors:
    def test_modbus_exception_surfaces(self):
        ft = FakeTransport(replies=[_crc(bytes([0x01, 0x84, 0x02]))])  # fn04|80
        ctrl = PztcController(transport=ft, axes=[
            AxisConfig(name="delay", channel=0, min_pos=0, max_pos=1e9)],
            inter_frame_gap_s=0.0)
        with pytest.raises(ModbusExceptionError):
            ctrl.axis("delay").get_position()

    def test_dead_bus_reports_unavailable(self):
        ft = FakeTransport(unavailable=True)
        ctrl = PztcController(transport=ft, axes=[
            AxisConfig(name="delay", channel=0, min_pos=0, max_pos=1e9)],
            inter_frame_gap_s=0.0)
        with pytest.raises(InstrumentUnavailable):
            ctrl.axis("delay").get_position()

    def test_empty_reply_reports_no_reply(self):
        ft = FakeTransport(replies=[b""])
        ctrl = PztcController(transport=ft, axes=[
            AxisConfig(name="delay", channel=0, min_pos=0, max_pos=1e9)],
            inter_frame_gap_s=0.0)
        with pytest.raises(InstrumentError, match="no reply"):
            ctrl.axis("delay").get_position()

    def test_constructor_requires_port_or_transport(self):
        with pytest.raises(ValueError):
            PztcController()

    def test_default_baudrate_is_460800(self):
        c = PztcController(port="COM17")
        assert c._baudrate == 460800
