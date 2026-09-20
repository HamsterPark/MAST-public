"""ZhuoJu (卓聚科技) PZTC nm003 closed-loop piezo stage controller.

Protocol: **Modbus RTU over USB-RS422** (FTDI VCP). Replaces the LabVIEW
``Hard_sub/PZTC_*.vi`` wrappers used by ``Pump probe*.vi`` / ``2D stage.vi``.
1-3 channels per controller; multiple controllers daisy-chain with distinct
slave addresses (vendor readme).

Register map
============
The full register map is recovered from the vendor spec shipped in the dev
kit — ``PZTC_nm003开发包/PZTC_V1.0/pztc_nm001通信协议MODBUS_V1.0.xlsx`` — and
baked into :class:`PztcRegisterMap` as defaults (no on-rig calibration
needed). Four Modbus object spaces, each with its own per-channel stride
(channel is 0-based, matching the DLL ``Channel`` argument):

======================  ========  ======  ================  ================
object space            fn read   fn wr   stride (per ch)   key registers
======================  ========  ======  ================  ================
holding registers       0x03      06/10   16                setpoint_place(18),
                                                             setpoint_speed(20)
coils                   0x01      0x05    16                close_loop_move(8),
                                                             enc_look_zero(3)
discrete inputs         0x02      —       2                 zero_status(0),
                                                             place_achieve(1)
input registers         0x04      —       3                 place_counter(0),
                                                             speed_counter(2)
======================  ========  ======  ================  ================

Positions are int32 in encoder *pulses* (2 registers). One pulse of physical
travel is stage-specific — set per axis via ``counts_per_unit``; the host
software uses 1 pulse = 1 nm for the delay stage (counts_per_unit = 1000 for
µm). The 32-bit word order across the 2 registers (``word_order_big``) is the
one thing to confirm on the bench — a wrong order makes the read-back position
obviously garbage, which OpticalStageWiggle catches immediately.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, fields
from typing import Mapping

from mast.environment.serial_transport import (
    SerialLike,
    SerialSettings,
    SerialUnavailable,
    open_transport,
)
from mast.instruments.base import (
    AxisConfig,
    AxisStatus,
    InstrumentError,
    InstrumentUnavailable,
    MotionAxis,
    MotionController,
    MotionTimeout,
)

logger = logging.getLogger(__name__)

__all__ = [
    "crc16_modbus",
    "build_rtu_frame",
    "parse_rtu_response",
    "ModbusExceptionError",
    "PztcRegisterMap",
    "PztcAxis",
    "PztcController",
]


# ── Modbus RTU framing (standard, self-contained, no pymodbus dep) ────────


def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS (poly 0xA001 reflected, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def build_rtu_frame(slave: int, func: int, payload: bytes) -> bytes:
    """addr + func + payload + CRC16 (CRC little-endian on the wire)."""
    body = bytes([slave & 0xFF, func & 0xFF]) + payload
    crc = crc16_modbus(body)
    return body + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


class ModbusExceptionError(InstrumentError):
    """Slave replied with a Modbus exception frame."""

    _CODES = {
        0x01: "illegal function",
        0x02: "illegal data address",
        0x03: "illegal data value",
        0x04: "slave device failure",
        0x05: "acknowledge",
        0x06: "slave device busy",
    }

    def __init__(self, func: int, code: int):
        self.func = func
        self.code = code
        desc = self._CODES.get(code, "unknown exception")
        super().__init__(
            f"Modbus exception 0x{code:02X} ({desc}) for function 0x{func & 0x7F:02X}"
        )


def parse_rtu_response(frame: bytes, *, slave: int, func: int) -> bytes:
    """Validate CRC/addr/function of *frame*; return the data payload
    (after addr+func, before CRC). Raises on short frame, CRC mismatch,
    wrong slave, or a Modbus exception frame (func | 0x80)."""
    if len(frame) < 5:
        raise InstrumentError(
            f"Modbus response too short ({len(frame)} bytes): {frame.hex(' ')}"
        )
    body, crc_lo, crc_hi = frame[:-2], frame[-2], frame[-1]
    crc = crc16_modbus(body)
    if (crc & 0xFF, (crc >> 8) & 0xFF) != (crc_lo, crc_hi):
        raise InstrumentError(f"Modbus CRC mismatch on frame {frame.hex(' ')}")
    if body[0] != (slave & 0xFF):
        raise InstrumentError(
            f"Modbus reply from slave {body[0]}, expected {slave}"
        )
    rfunc = body[1]
    if rfunc == (func | 0x80):
        code = body[2] if len(body) > 2 else 0
        raise ModbusExceptionError(func, code)
    if rfunc != func:
        raise InstrumentError(
            f"Modbus reply function 0x{rfunc:02X}, expected 0x{func:02X}"
        )
    return body[2:]


# ── register map — ground truth from the vendor Modbus spec ───────────────


@dataclass(frozen=True)
class PztcRegisterMap:
    """PZTC nm001/nm003 Modbus register map. Channel is **0-based** (0/1/2).

    Per-channel address = ``base + channel * <space>_stride``. Every default
    below is the vendor ground truth (see module docstring); override a field
    only if a firmware revision moves it. Base addresses are for channel 0.
    """

    # holding registers — fn 0x03 read / 0x06,0x10 write, stride 16
    setpoint_place: int = 18       # close_setpoint_place, int32 (2 reg), pulses
    setpoint_speed: int = 20       # close_setpoint_speed, uint16, pulses
    pid0: int = 23                 # speed-loop PI (hi byte Ki, lo byte Kp)
    pid1: int = 24                 # fine-position-loop PI
    holding_stride: int = 16

    # coils — fn 0x01 read / 0x05 write, stride 16
    close_loop_move: int = 8       # 1 = closed loop enabled, 0 = abort
    enc_look_zero: int = 3         # 1 = start find-zero homing
    enc_dir: int = 13              # 0 = default, 1 = reversed encoder direction
    enc_clear: int = 14            # write 1 to zero the encoder counter
    place_pid_en: int = 5          # position loop enable (hw default on)
    coil_stride: int = 16

    # discrete inputs — fn 0x02 read only, stride 2
    zero_status: int = 0           # 1 = find-zero reached
    place_achieve: int = 1         # 1 = target reached (hw auto-clears on new setpoint)
    discrete_stride: int = 2

    # input registers — fn 0x04 read only, stride 3
    place_counter: int = 0         # int32 (2 reg), current position in pulses
    speed_counter: int = 2         # uint16, current speed in pulses
    input_stride: int = 3

    # int32 word order across its 2 registers (True = high word at lower addr)
    word_order_big: bool = True

    def holding_addr(self, base: int, channel: int) -> int:
        return int(base) + int(channel) * self.holding_stride

    def coil_addr(self, base: int, channel: int) -> int:
        return int(base) + int(channel) * self.coil_stride

    def discrete_addr(self, base: int, channel: int) -> int:
        return int(base) + int(channel) * self.discrete_stride

    def input_addr(self, base: int, channel: int) -> int:
        return int(base) + int(channel) * self.input_stride

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PztcRegisterMap":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        unknown = set(raw) - known
        if unknown:
            logger.warning("PztcRegisterMap: ignoring unknown keys %s", sorted(unknown))
        return cls(**kwargs)  # type: ignore[arg-type]


# ── axis ──────────────────────────────────────────────────────────────────


class PztcAxis(MotionAxis):
    """One PZTC channel. Native unit = encoder counts scaled by
    ``counts_per_unit`` (counts / AxisConfig.unit); positions exposed to
    callers are in the axis's configured unit."""

    def __init__(
        self,
        config: AxisConfig,
        controller: "PztcController",
        *,
        counts_per_unit: float = 1.0,
        position_tolerance: float = 5.0,   # counts
    ):
        super().__init__(config, controller)
        if counts_per_unit <= 0:
            raise ValueError("counts_per_unit must be > 0")
        self._pztc: PztcController = controller
        self._counts_per_unit = float(counts_per_unit)
        self._tolerance_counts = float(position_tolerance)
        self._last_target_counts: int | None = None
        self._last_pos_counts: int | None = None

    # -- unit helpers --------------------------------------------------------

    def _to_counts(self, pos: float) -> int:
        return int(round(pos * self._counts_per_unit))

    def _from_counts(self, counts: int) -> float:
        return counts / self._counts_per_unit

    # -- primitives (controller lock already held by base class) -------------

    def _move_abs_raw(self, target: float) -> None:
        ch = int(self.config.channel)
        counts = self._to_counts(target)
        m = self._pztc._map
        # write the int32 setpoint (2 holding regs), then enable closed-loop.
        self._pztc.write_holdings(
            m.holding_addr(m.setpoint_place, ch), self._pztc._split_i32(counts)
        )
        self._pztc.write_coil(m.coil_addr(m.close_loop_move, ch), True)
        self._last_target_counts = counts

    def _get_position_raw(self) -> float:
        ch = int(self.config.channel)
        m = self._pztc._map
        regs = self._pztc.read_input(m.input_addr(m.place_counter, ch), 2)
        return self._from_counts(self._pztc._join_i32(regs))

    def _get_status_raw(self) -> AxisStatus:
        ch = int(self.config.channel)
        m = self._pztc._map
        regs = self._pztc.read_input(m.input_addr(m.place_counter, ch), 2)
        counts = self._pztc._join_i32(regs)
        # on-target from the hardware 'place_achieve' discrete input, which the
        # controller auto-clears when a new setpoint is written — authoritative,
        # no position-delta heuristic. Unknown before the first move.
        on_target: bool | None = None
        if self._last_target_counts is not None:
            on_target = self._pztc.read_discrete_input(
                m.discrete_addr(m.place_achieve, ch)
            )
        return AxisStatus(
            position=self._from_counts(counts),
            moving=(on_target is False),
            on_target=on_target,
            raw={"counts": counts, "channel": ch},
        )

    def _stop_raw(self) -> None:
        ch = int(self.config.channel)
        m = self._pztc._map
        # abort by disabling the closed-loop-move coil (0 = 关闭).
        self._pztc.write_coil(m.coil_addr(m.close_loop_move, ch), False)
        self._last_target_counts = None

    def home(self, *, wait: bool = True, timeout: float | None = None) -> AxisStatus:
        """Find-zero via the ``enc_look_zero`` coil, polling the
        ``zero_status`` discrete input (not ``place_achieve``, which tracks
        setpoint moves) for completion."""
        ch = int(self.config.channel)
        m = self._pztc._map
        with self._controller._io_lock:
            self._pztc.write_coil(m.coil_addr(m.enc_look_zero, ch), True)
        self._last_target_counts = None
        if not wait:
            return self.get_status()
        deadline = time.monotonic() + (timeout or self.DEFAULT_TIMEOUT_S)
        while time.monotonic() < deadline:
            with self._controller._io_lock:
                done = self._pztc.read_discrete_input(
                    m.discrete_addr(m.zero_status, ch)
                )
            if done:
                return self.get_status()
            time.sleep(self.POLL_INTERVAL_S)
        raise MotionTimeout(
            f"axis {self.config.name!r}: find-zero not reached within "
            f"{timeout or self.DEFAULT_TIMEOUT_S:.1f}s"
        )


# ── controller ────────────────────────────────────────────────────────────


class PztcController(MotionController):
    """One PZTC nm003 box on an RS-422 bus.

    ``transport`` may be injected (tests: any ``SerialLike``); otherwise a
    real serial port is opened lazily from ``port``/``baudrate`` at
    :meth:`connect` time.
    """

    #: response deadline per Modbus transaction (seconds)
    IO_TIMEOUT_S = 0.5

    def __init__(
        self,
        *,
        port: str | None = None,
        baudrate: int = 460800,
        slave: int = 1,
        register_map: PztcRegisterMap | None = None,
        axes: list[AxisConfig] | None = None,
        counts_per_unit: float = 1.0,
        position_tolerance: float = 5.0,
        transport: SerialLike | None = None,
        inter_frame_gap_s: float = 0.004,
    ):
        super().__init__()
        if transport is None and not port:
            raise ValueError("PztcController needs either port or transport")
        self._port = port
        self._baudrate = int(baudrate)
        self._slave = int(slave)
        self._map = register_map or PztcRegisterMap()
        self._axis_configs = list(axes or [])
        self._counts_per_unit = float(counts_per_unit)
        self._position_tolerance = float(position_tolerance)
        self._transport: SerialLike | None = transport
        self._owns_transport = transport is None
        # RS-422 half-duplex style bus: keep a minimal silent gap between
        # frames (Modbus 3.5 char times; 4 ms is safe for 9600+ baud).
        self._gap_s = float(inter_frame_gap_s)
        self._last_io = 0.0

    # -- lifecycle ------------------------------------------------------------

    def _connect_raw(self) -> None:
        if self._transport is None:
            settings = SerialSettings(
                port=self._port or "",
                baudrate=self._baudrate,
                timeout=self.IO_TIMEOUT_S,
                write_timeout=self.IO_TIMEOUT_S,
            )
            try:
                t = open_transport(settings)
                t.open()
            except SerialUnavailable as exc:
                raise InstrumentUnavailable(f"PZTC on {self._port}: {exc}") from exc
            self._transport = t

    def _close_raw(self) -> None:
        t = self._transport
        if t is not None and self._owns_transport:
            try:
                t.close()
            except Exception:  # noqa: BLE001
                pass
            self._transport = None

    def _build_axes(self) -> dict[str, MotionAxis]:
        return {
            cfg.name: PztcAxis(
                cfg,
                self,
                counts_per_unit=self._counts_per_unit,
                position_tolerance=self._position_tolerance,
            )
            for cfg in self._axis_configs
        }

    # -- Modbus primitives -----------------------------------------------------

    def _transact(self, func: int, payload: bytes, *, resp_len: int) -> bytes:
        t = self._transport
        if t is None:
            raise InstrumentUnavailable("PZTC transport not connected")
        # enforce inter-frame silence
        gap = self._gap_s - (time.monotonic() - self._last_io)
        if gap > 0:
            time.sleep(gap)
        frame = build_rtu_frame(self._slave, func, payload)
        try:
            raw = t.transact(frame, read_size=resp_len, timeout=self.IO_TIMEOUT_S)
        except SerialUnavailable as exc:
            raise InstrumentUnavailable(f"PZTC I/O failed: {exc}") from exc
        finally:
            self._last_io = time.monotonic()
        if not raw:
            raise InstrumentError(
                f"PZTC slave {self._slave}: no reply to function 0x{func:02X}"
            )
        # An exception frame is 5 bytes — shorter than any normal reply we
        # request; trim to actual content before parsing.
        return parse_rtu_response(bytes(raw), slave=self._slave, func=func)

    def read_holding(self, address: int, count: int = 1) -> list[int]:
        payload = address.to_bytes(2, "big") + count.to_bytes(2, "big")
        data = self._transact(0x03, payload, resp_len=5 + 2 * count)
        return self._unpack_regs(data, count)

    def read_input(self, address: int, count: int = 1) -> list[int]:
        payload = address.to_bytes(2, "big") + count.to_bytes(2, "big")
        data = self._transact(0x04, payload, resp_len=5 + 2 * count)
        return self._unpack_regs(data, count)

    def read_discrete_input(self, address: int, count: int = 1) -> bool:
        """Read a discrete input (fn 0x02); returns the first bit as a bool."""
        payload = address.to_bytes(2, "big") + count.to_bytes(2, "big")
        data = self._transact(0x02, payload, resp_len=5 + (count + 7) // 8)
        # data = [byte_count][status bytes]; bit 0 of first status byte.
        if len(data) < 2:
            raise InstrumentError(
                f"discrete-input read at {address} returned {data.hex(' ')}"
            )
        return bool(data[1] & 0x01)

    def write_holding(self, address: int, value: int) -> None:
        payload = address.to_bytes(2, "big") + (value & 0xFFFF).to_bytes(2, "big")
        self._transact(0x06, payload, resp_len=8)

    def write_holdings(self, address: int, values: list[int]) -> None:
        count = len(values)
        payload = (
            address.to_bytes(2, "big")
            + count.to_bytes(2, "big")
            + bytes([2 * count])
            + b"".join((v & 0xFFFF).to_bytes(2, "big") for v in values)
        )
        self._transact(0x10, payload, resp_len=8)

    def write_coil(self, address: int, on: bool) -> None:
        payload = address.to_bytes(2, "big") + (b"\xff\x00" if on else b"\x00\x00")
        self._transact(0x05, payload, resp_len=8)

    @staticmethod
    def _unpack_regs(data: bytes, count: int) -> list[int]:
        if len(data) < 1 + 2 * count:
            raise InstrumentError(
                f"Modbus read returned {len(data)} bytes, expected {1 + 2 * count}"
            )
        byte_count = data[0]
        if byte_count != 2 * count:
            raise InstrumentError(
                f"Modbus byte count {byte_count}, expected {2 * count}"
            )
        return [
            int.from_bytes(data[1 + 2 * i: 3 + 2 * i], "big") for i in range(count)
        ]

    # -- int32-across-two-registers packing (honours word order) -----------------

    def _split_i32(self, value: int) -> list[int]:
        u32 = value & 0xFFFFFFFF
        hi, lo = (u32 >> 16) & 0xFFFF, u32 & 0xFFFF
        return [hi, lo] if self._map.word_order_big else [lo, hi]

    def _join_i32(self, regs: list[int]) -> int:
        hi, lo = (regs[0], regs[1]) if self._map.word_order_big else (regs[1], regs[0])
        u32 = (hi << 16) | lo
        return u32 - 0x100000000 if u32 & 0x80000000 else u32
