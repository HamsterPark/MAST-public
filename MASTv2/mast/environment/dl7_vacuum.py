"""DL-7 vacuum gauge driver (RS-485 / Modbus-RTU).

Protocol (from ``DL-7真空计RS485接口说明new.docx``):

* Line: 9600 bps, 8N1. RS-485 half-duplex, frames in hex.
* This is a standard **Modbus-RTU "read input registers" (function 0x04)**
  transaction. Master asks for 2 registers starting at 0::

      07 04 0000 0002 <CRC16-lo> <CRC16-hi>      (address 0x07)

  Gauge replies::

      07 04 04 <reg1:2> <reg2:2> <CRC16-lo> <CRC16-hi>
              └ byte count = 4

  ``reg1`` (A) is the mantissa ×10, ``reg2`` (B) is the magnitude of the
  negative exponent (B ∈ 2..8). Pressure = (A / 10) · 10⁻ᴮ Pa. The documented
  example ``001C 0002`` → A=28, B=2 → 2.8·10⁻² Pa.
* CRC16: init 0xFFFF, poly 0xA001, transmitted **low byte first** (Modbus).

The gauge measures 5·10⁻⁸ … 1·10⁻¹ Pa. Multiple gauges share an RS-485 bus and
are addressed by the strap on the rear terminal (``address`` argument), or sit
on separate USB-RS485 adapters (separate ``port``s) — both are supported by
instantiating one :class:`DL7VacuumSensor` per gauge with a distinct name.
"""

from __future__ import annotations

import logging

from mast.core.types import SensorReading
from mast.environment.alarm import AlarmSpec
from mast.environment.base import EnvironmentSensor
from mast.environment.serial_transport import (
    SerialLike,
    SerialSettings,
    SerialTransport,
    SerialUnavailable,
    open_transport,
)

logger = logging.getLogger(__name__)

DL7_BAUD = 9600
DL7_DEFAULT_ADDRESS = 7
DL7_FUNCTION = 0x04
DL7_RESPONSE_LEN = 9  # addr + fn + bytecount(=4) + 2 regs + CRC(2)


def crc16(data: bytes) -> int:
    """Modbus CRC16 (init 0xFFFF, poly 0xA001). Returns the 16-bit value."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def _crc_suffix(data: bytes) -> bytes:
    """CRC16 of *data* as 2 bytes, low byte first (Modbus wire order)."""
    crc = crc16(data)
    return bytes((crc & 0xFF, (crc >> 8) & 0xFF))


def build_request(address: int = DL7_DEFAULT_ADDRESS, *, start: int = 0,
                  count: int = 2) -> bytes:
    """Build the master read-input-registers request frame."""
    body = bytes((
        address & 0xFF, DL7_FUNCTION,
        (start >> 8) & 0xFF, start & 0xFF,
        (count >> 8) & 0xFF, count & 0xFF,
    ))
    return body + _crc_suffix(body)


class DL7FrameError(ValueError):
    """Malformed / mismatched DL-7 response frame."""


def parse_response(frame: bytes, address: int = DL7_DEFAULT_ADDRESS) -> float:
    """Parse a DL-7 response frame → pressure in **Pa**.

    Raises :class:`DL7FrameError` on length / address / function / byte-count /
    CRC mismatch so callers can distinguish a wrong/garbled reply from a dead
    port.
    """
    if len(frame) < DL7_RESPONSE_LEN:
        raise DL7FrameError(f"short frame: {len(frame)} bytes ({frame.hex()})")
    frame = frame[:DL7_RESPONSE_LEN]
    if frame[0] != (address & 0xFF):
        raise DL7FrameError(f"address mismatch: got {frame[0]:#04x}, want {address:#04x}")
    if frame[1] != DL7_FUNCTION:
        raise DL7FrameError(f"function mismatch: got {frame[1]:#04x}")
    if frame[2] != 0x04:
        raise DL7FrameError(f"byte-count mismatch: got {frame[2]:#04x}")
    if _crc_suffix(frame[:7]) != frame[7:9]:
        raise DL7FrameError(f"CRC mismatch on {frame.hex()}")

    mantissa_raw = (frame[3] << 8) | frame[4]   # A (×10)
    neg_exponent = (frame[5] << 8) | frame[6]   # B
    pressure_pa = (mantissa_raw / 10.0) * (10.0 ** (-neg_exponent))
    return pressure_pa


def probe_dl7(port: str, address: int = DL7_DEFAULT_ADDRESS,
              timeout: float = 0.4) -> bool:
    """Return True if a DL-7 gauge answers on *port* at *address*.

    Opens a transport, runs one transaction, parses the reply. Any failure
    (no pyserial, port busy, wrong/garbled device) → False. Never raises.
    """
    settings = SerialSettings(port=port, baudrate=DL7_BAUD, bytesize=8,
                              parity="N", stopbits=1, timeout=timeout)
    transport: SerialTransport | None = None
    try:
        transport = open_transport(settings)
        reply = transport.transact(build_request(address),
                                   read_size=DL7_RESPONSE_LEN, timeout=timeout)
        parse_response(reply, address)
        return True
    except Exception:  # SerialUnavailable / DL7FrameError / anything → not a DL-7
        return False
    finally:
        if transport is not None:
            transport.close()


class DL7VacuumSensor(EnvironmentSensor):
    """A single DL-7 vacuum gauge as an :class:`EnvironmentSensor`.

    Reports pressure in Pa. Construction never opens the port — the first
    :meth:`read` does, and any open/transact failure degrades to
    ``status="unavailable"`` (port gone) or ``"error"`` (garbled reply) so a
    missing gauge can't crash the monitor.
    """

    def __init__(
        self,
        *,
        name: str = "vacuum",
        port: str | None = None,
        address: int = DL7_DEFAULT_ADDRESS,
        unit: str = "Pa",
        alarm: AlarmSpec | None = None,
        timeout: float = 0.4,
        transport: SerialLike | None = None,
    ):
        self._name = name
        self._address = address
        self._unit = unit
        self._alarm = alarm or AlarmSpec()
        self._timeout = timeout
        self._port = port
        self._transport: SerialLike | None = transport
        # Settings are derived from `port` whenever there IS one, even when a
        # transport was injected. The old `transport is None and ...` guard left
        # every pooled gauge with `_settings = None`, so once anything dropped
        # its transport it raised "no port configured" forever — and pooling the
        # autodetected DL-7 is exactly what d17ff1b started doing. A sensor that
        # knows its port must always be able to say so.
        if port is not None:
            self._settings = SerialSettings(
                port=port, baudrate=DL7_BAUD, bytesize=8, parity="N",
                stopbits=1, timeout=timeout)
        else:
            self._settings = None
        self._request = build_request(address)

    def name(self) -> str:
        return self._name

    def _ensure_transport(self) -> SerialLike:
        if self._transport is None:
            if self._settings is None:
                raise SerialUnavailable("no port configured")
            self._transport = open_transport(self._settings)
        return self._transport

    def read(self) -> SensorReading:
        try:
            transport = self._ensure_transport()
        except SerialUnavailable:
            return SensorReading(value=0.0, unit=self._unit, status="unavailable")
        try:
            reply = transport.transact(self._request, read_size=DL7_RESPONSE_LEN,
                                       timeout=self._timeout)
        except SerialUnavailable:
            # Port dropped (unplugged adapter etc.) — drop the OS handle so a
            # later read can re-open, and report unavailable rather than error.
            # Close rather than discard: several gauges can share one RS-485 bus
            # (different Modbus addresses, one transport — see
            # autodetect._shared_transport), and replacing the object here would
            # give this gauge a second exclusive handle on a port its neighbour
            # already holds.
            close = getattr(self._transport, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best-effort
                    pass
            return SensorReading(value=0.0, unit=self._unit, status="unavailable")
        try:
            pressure = parse_response(reply, self._address)
        except DL7FrameError as exc:
            logger.debug("DL-7 %s parse error: %s", self._name, exc)
            return SensorReading(value=0.0, unit=self._unit, status="error")
        status = self._alarm.evaluate(pressure, base_status="ok")
        return SensorReading(value=pressure, unit=self._unit, status=status)

    def close(self) -> None:
        """Release the OS handle. KEEPS the transport object.

        Same reason as ``LakeshoreTemperatureSensor.close`` — several gauges can
        share one RS-485 bus, and dropping the reference gives this one a
        private handle on a port its neighbour holds. ``read()``'s failure path
        already keeps the object; these two must agree.
        """
        close = getattr(self._transport, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover
                pass


__all__ = [
    "crc16",
    "build_request",
    "parse_response",
    "probe_dl7",
    "DL7VacuumSensor",
    "DL7FrameError",
    "DL7_BAUD",
    "DL7_DEFAULT_ADDRESS",
]
