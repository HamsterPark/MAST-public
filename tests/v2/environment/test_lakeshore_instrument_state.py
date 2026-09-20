"""Read-only instrument-state decoding, pinned to a REAL Lake Shore 335.

**Which replies here are captured, and which are constructed.** Keep the two
apart: in a test file they look exactly alike, and a constructed reply read as
a captured one is a made-up number wearing a measurement's clothes.

* ``REAL_335`` was captured off the actual instrument on the instrument (COM13,
  57600 7-O-1) on 2026-07-27 — **not invented**. That matters: the field order,
  the sign/exponent formatting ("+000.0", "+77.402"), the zero-padded RDGST
  ("000") and the CSV enum replies ("3,0,0") are exactly what the hardware
  sends, so a refactor that breaks the parser fails here instead of on the instrument.
  Tests that override one key of it (``dict(REAL_335, **{...})``) still stand on
  captured formatting for every other field.

* The ``RDGST?`` **tri-state** tests further down do not. Their replies — ``""``
  (timeout), ``"ERR"``, ``"0.0"``, ``"-1"``, and a raised ``SerialUnavailable``
  — are **constructed** (2026-08-15). They encode what the SOFTWARE must do with
  a non-answer; they are not evidence about what this instrument sends. What
  that leaves unverified is spelled out on :func:`_snapshot_with_rdgst`.

The safety invariant is asserted too: nothing in this module may put a Lakeshore
SETTER on the wire. The recorded transport captures every byte written, and the
test asserts they are all queries.
"""

from __future__ import annotations

import pytest

from mast.environment.lakeshore_temp import (
    LakeshoreError,
    LakeshoreTemperatureSensor,
    decode_reading_status,
    read_snapshot,
)
from mast.environment.serial_transport import SerialUnavailable

# Captured 2026-07-27 from LSCI,MODEL335,LSA2SHB/#######,2.1
REAL_335 = {
    "*IDN?": "LSCI,MODEL335,LSA2SHB/#######,2.1",
    "INNAME? A": "SPM",
    "INNAME? B": "Magnet",
    "INTYPE? A": "3,1,3,1,1",          # NTC RTD
    "INTYPE? B": "1,0,0,0,1",          # Diode
    "KRDG? A": "+77.402",
    "KRDG? B": "+77.275",
    "CRDG? A": "-195.75",
    "CRDG? B": "-195.88",
    "SRDG? A": "+264.61",              # ohms
    "SRDG? B": "+1.0277",              # volts
    "RDGST? A": "000",
    "RDGST? B": "000",
    "RANGE? 1": "0",
    "RANGE? 2": "0",
    "HTR? 1": "+000.0",
    "HTR? 2": "+000.0",
    "SETP? 1": "+500.00",
    "SETP? 2": "+40.000",
    "OUTMODE? 1": "3,0,0",             # open loop, no input, powerup off
    "OUTMODE? 2": "1,1,0",             # closed-loop PID on input A, powerup off
    "RAMP? 1": "0,+000.0",
    "RAMP? 2": "0,+000.0",
}


class RecordedTransport:
    """Replays the captured replies and records every byte written."""

    def __init__(self, table: dict[str, str] | None = None):
        self.table = dict(REAL_335 if table is None else table)
        self.written: list[str] = []

    def transact(self, payload: bytes, **kw) -> bytes:
        sent = payload.decode("ascii").strip()
        self.written.append(sent)
        reply = self.table.get(sent)
        if reply is None:
            raise AssertionError(f"instrument was asked something unrecorded: {sent!r}")
        return (reply + "\r\n").encode("ascii")

    def close(self) -> None:
        pass


def test_snapshot_decodes_the_real_instrument() -> None:
    t = RecordedTransport()
    snap = read_snapshot(t, port="COM13")

    assert snap.model == "MODEL335"
    assert snap.firmware == "2.1"
    assert snap.serial_number == "LSA2SHB/#######"
    assert snap.port == "COM13"

    a, b = snap.channels
    assert (a.channel, a.label, a.sensor_type) == ("A", "SPM", "NTC 热敏电阻")
    assert (b.channel, b.label, b.sensor_type) == ("B", "Magnet", "二极管")
    assert a.kelvin == pytest.approx(77.402)
    assert b.kelvin == pytest.approx(77.275)
    assert a.celsius == pytest.approx(-195.75)
    assert a.sensor_value == pytest.approx(264.61)   # ohms, NTC RTD
    assert b.sensor_value == pytest.approx(1.0277)   # volts, diode
    assert a.faults == [] and b.faults == []


def test_heater_state_is_read_as_off() -> None:
    """The whole point of the readout: show honestly whether a heater is on."""
    snap = read_snapshot(RecordedTransport(), port="COM13")
    o1, o2 = snap.outputs

    assert (o1.range_code, o1.range_label, o1.heater_on) == (0, "关闭", False)
    assert (o2.range_code, o2.range_label, o2.heater_on) == (0, "关闭", False)
    assert o1.heater_pct == 0.0 and o2.heater_pct == 0.0
    assert snap.any_heater_on is False

    assert o1.mode == "开环" and o1.control_input == "无"
    assert o2.mode == "闭环 PID" and o2.control_input == "A"
    assert o1.powerup_enabled is False and o2.powerup_enabled is False
    assert o1.ramping is False


def test_parked_setpoint_does_not_read_as_heating() -> None:
    """Output 2 is CONFIGURED for closed-loop PID and output 1 parks at 500 K.

    Both are inert because RANGE is 0. If heater_on were derived from mode or
    setpoint the panel would cry wolf on a perfectly cold instrument.
    """
    snap = read_snapshot(RecordedTransport(), port="COM13")
    assert snap.outputs[0].setpoint == pytest.approx(500.0)
    assert snap.outputs[1].setpoint == pytest.approx(40.0)
    assert snap.any_heater_on is False


def test_heater_on_is_detected_when_range_is_nonzero() -> None:
    table = dict(REAL_335, **{"RANGE? 1": "2", "HTR? 1": "+034.7"})
    snap = read_snapshot(RecordedTransport(table), port="COM13")

    assert snap.outputs[0].heater_on is True
    assert snap.outputs[0].range_label == "中"
    assert snap.outputs[0].heater_pct == pytest.approx(34.7)
    assert snap.any_heater_on is True


def test_ramp_rate_keeps_its_fraction() -> None:
    """A 0.1 K/min ramp must not decode as 0 K/min."""
    table = dict(REAL_335, **{"RAMP? 1": "1,+000.1"})
    snap = read_snapshot(RecordedTransport(table), port="COM13")
    assert snap.outputs[0].ramping is True
    assert snap.outputs[0].ramp_rate == pytest.approx(0.1)


def test_only_queries_are_ever_written() -> None:
    """The structural safety claim: no byte sequence sent is a setter."""
    t = RecordedTransport()
    read_snapshot(t, port="COM13")

    assert t.written, "nothing was sent — the assertion below would be vacuous"
    for sent in t.written:
        verb = sent.split(" ", 1)[0]
        assert verb.endswith("?"), f"non-query command reached the instrument: {sent!r}"
    # The commands that actually drive a heater, spelled out.
    for danger in ("SETP ", "RANGE ", "MOUT ", "OUTMODE ", "PID ", "*RST"):
        assert not any(s.upper().startswith(danger) for s in t.written)


def test_reading_status_flags_decode() -> None:
    assert decode_reading_status(0) == []
    assert decode_reading_status(1) == ["读数无效"]
    assert decode_reading_status(16) == ["温度低于量程"]
    assert decode_reading_status(32 + 128) == ["温度高于量程", "传感器超量程"]


def test_faulty_reading_is_surfaced_not_hidden() -> None:
    """A sensor that is over-range still returns a NUMBER; without the RDGST
    decode the panel would present it as a real temperature."""
    table = dict(REAL_335, **{"RDGST? A": "32"})
    snap = read_snapshot(RecordedTransport(table), port="COM13")
    assert snap.channels[0].faults == ["温度高于量程"]


# ── RDGST? 三态：问过干净 / 问过有故障 / 根本没问出来 ────────────────────────
#
# 超时的时候 pyserial 的 read_until 回 b"" 且**不抛**（verified against pyserial
# 3.5：`c = self.read(1)` 拿到 b"" 就 break，返回已收到的字节）。于是 transact
# 回 b""、_query 回 ""，而 `int("" or 0)` = 0 —— 一个完全合法的「干净」状态字。
# 一个开路 / 过量程但状态位没答上来的通道，就这样拿到了健康背书。


def _snapshot_with_rdgst(reply):
    """Build a snapshot where only ``RDGST? A`` behaves as *reply*.

    An ``Exception`` instance is raised instead of answered — that covers the
    port dropping mid-snapshot, which reaches the same field by another road.

    **NOT VERIFIED ON HARDWARE (2026-08-15).** Every *reply* passed in here is
    constructed. What these tests pin is the software contract — a non-answer
    must not become ``[]`` — and that holds whatever the wire does. What they do
    NOT establish, and what nobody has checked on the instrument:

      * whether a 335 with an open / over-range sensor really leaves ``RDGST?``
        unanswered, or answers it with a fault bitmask. If it always answers,
        the tri-state never fires in the field and the real defence is the
        fault-bit decode, not this;
      * whether a half-dead RS-232 link drops THIS query specifically, or drops
        the whole transaction — if ``KRDG?`` dies with it the channel has no
        kelvin and is filtered one step earlier, before health is consulted;
      * what a 218 / 224 / 336 does. ``RDGST?`` support is taken from the manual
        across the family, not observed.

    All three need the rig. Until then read this as a guard against a fold the
    code was PROVEN to perform (``int("" or 0)`` → 0 → "clean"), not as a claim
    about how often the instrument triggers it.
    """
    class T(RecordedTransport):
        def transact(self, payload: bytes, **kw) -> bytes:
            sent = payload.decode("ascii").strip()
            if sent == "RDGST? A":
                self.written.append(sent)
                if isinstance(reply, Exception):
                    raise reply
                return (reply + "\r\n").encode("ascii")
            return super().transact(payload, **kw)
    return read_snapshot(T(), port="COM13")


def test_a_status_query_that_never_answered_is_not_clean() -> None:
    """判据：调用方能不能把「问过了，干净」和「根本没问出来」分开。"""
    from mast.environment.lakeshore_temp import HEALTH_CLEAN, HEALTH_UNKNOWN

    silent = _snapshot_with_rdgst("").channels[0]
    assert silent.faults is None, "超时被折叠成了「干净」"
    assert silent.health == HEALTH_UNKNOWN

    # 反向对照：同一条路径，唯一差别是仪器答了 "000"。
    answered = _snapshot_with_rdgst("000").channels[0]
    assert answered.faults == []
    assert answered.health == HEALTH_CLEAN
    assert silent.health != answered.health


@pytest.mark.parametrize("reply", [
    "",                                   # 超时：read_until 回 b""
    "ERR",                                # 垃圾回复：int() 抛
    "0.0",                                # 浮点：int() 抛
    "-1",                                 # 负数：位掩码不可能为负
    SerialUnavailable("COM13 掉了"),        # 端口中途没了：_query 抛
])
def test_every_way_of_not_answering_lands_on_unknown(reply) -> None:
    """四条不同的失败路径以前**全部**折叠成 []。

    两个折叠点，不是一个：`int("" or 0)` 吃掉超时，而 read_snapshot 里那个
    `except Exception: pass` 让其余三条留着构造函数的默认值 —— 那个默认值当时
    也是 []。
    """
    assert _snapshot_with_rdgst(reply).channels[0].faults is None


def test_the_status_word_still_decodes_when_it_does_answer() -> None:
    """反向对照，逐位：三态没有把会答话的那条路径一起拒掉。"""
    for wire, expect in (("000", []), ("001", ["读数无效"]),
                         ("032", ["温度高于量程"]), ("128", ["传感器超量程"])):
        assert _snapshot_with_rdgst(wire).channels[0].faults == expect, wire


def test_parse_reading_status_is_the_tristate_boundary() -> None:
    from mast.environment.lakeshore_temp import parse_reading_status

    assert parse_reading_status("000") == []          # 问过，干净
    assert parse_reading_status("32") == ["温度高于量程"]
    assert parse_reading_status(b"000\r\n") == []      # 带终止符的原始字节
    for absent in ("", "   ", None, "ERR", "0.0", "-1"):
        assert parse_reading_status(absent) is None, absent


def test_no_sentinel_bitmask_is_ever_invented() -> None:
    """哨兵在位掩码上不成立 —— 这是「别用 -1」那条规矩的可执行形式。"""
    assert decode_reading_status(-1) == [
        "读数无效", "温度低于量程", "温度高于量程", "传感器读数为零", "传感器超量程"]


def test_unknown_status_survives_serialisation() -> None:
    """as_dict 不能把 None 收回成 [] —— 面板就是在这一层读它的。"""
    silent = _snapshot_with_rdgst("").channels[0].as_dict()
    clean = _snapshot_with_rdgst("000").channels[0].as_dict()
    assert silent["faults"] is None and silent["health"] == "unknown"
    assert clean["faults"] == [] and clean["health"] == "clean"


def test_unparseable_field_degrades_to_none_not_a_wrong_number() -> None:
    table = dict(REAL_335, **{"KRDG? A": "ERR 5", "SETP? 1": ""})
    snap = read_snapshot(RecordedTransport(table), port="COM13")
    assert snap.channels[0].kelvin is None      # NOT 5.0
    assert snap.outputs[0].setpoint is None
    assert snap.channels[1].kelvin == pytest.approx(77.275)   # siblings unaffected


def test_sensor_instrument_state_uses_the_shared_open_port() -> None:
    """instrument_state() must go through the sensor's existing transport, so a
    settings refresh never needs a second (impossible) handle on the port."""
    t = RecordedTransport()
    sensor = LakeshoreTemperatureSensor(name="SPM", port="COM13", channel="A",
                                        transport=t)
    snap = sensor.instrument_state()

    assert snap is not None
    assert snap.model == "MODEL335"
    assert snap.channels[0].label == "SPM"
    assert snap.port == "COM13"


def test_query_gate_rejects_setters() -> None:
    """The gate is structural: a verb without '?' can't be sent even if a future
    edit adds it to the whitelist by mistake."""
    from mast.environment.lakeshore_temp import _assert_query

    for good in ("KRDG?", "RANGE?", "OUTMODE?", "*IDN?"):
        assert _assert_query(good) == good
    for bad in ("RANGE", "SETP", "MOUT 1,50", "*RST", "PID"):
        with pytest.raises(LakeshoreError):
            _assert_query(bad)


def test_query_argument_is_validated_not_coerced() -> None:
    """A bad output number must raise, not silently become channel 'A' — that
    would report output 1's heater under another output's heading."""
    from mast.environment.lakeshore_temp import _query

    with pytest.raises(LakeshoreError):
        _query(RecordedTransport(), "RANGE?", "1\r\nRANGE 1,3")
