"""Serial transport for RS-485 / RS-232 environment sensors.

Thin wrapper over *pyserial* so the rest of ``mast.environment`` never imports
``serial`` directly. Two design constraints drive this module:

1. **No-hardware must not crash** (explicit user requirement: 没有真空计或
   温度计，要求程序也不崩溃). When pyserial is absent, when no COM port exists,
   or when opening a port fails, this layer degrades gracefully —
   :func:`list_serial_ports` returns ``[]`` and :class:`SerialTransport`
   raises :class:`SerialUnavailable`, which the sensor classes catch to report
   ``status="unavailable"`` / ``"error"`` instead of propagating.

2. **Testable without hardware.** Sensors talk to a transport through the
   tiny :class:`SerialLike` protocol (just ``transact`` / ``close``), so unit
   tests inject a fake transport with canned frames — no serial port, no
   pyserial, no DL-7 / Lakeshore on the bench.

Half-duplex RS-485 buses (DL-7) and request/response SCPI links (Lakeshore)
are both *transactional*: write a request, read the reply. :meth:`transact`
encapsulates that with an input-buffer flush, a bounded read, and short
timeouts so a silent/wrong device can never hang the monitor thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── pyserial availability (probed once, lazily importable) ───────────────
try:  # pragma: no cover - trivial import guard
    import serial as _pyserial  # type: ignore
    from serial.tools import list_ports as _list_ports  # type: ignore

    HAS_PYSERIAL = True
except Exception:  # pragma: no cover - exercised on machines without pyserial
    _pyserial = None  # type: ignore
    _list_ports = None  # type: ignore
    HAS_PYSERIAL = False


class SerialUnavailable(RuntimeError):
    """Raised when a serial port cannot be opened (no pyserial / no port)."""


@dataclass(frozen=True)
class SerialPortInfo:
    """One enumerated serial port."""

    device: str           # e.g. "COM15" / "/dev/ttyUSB0"
    description: str = ""  # human string from the OS, e.g. "USB-SERIAL CH340"
    hwid: str = ""         # VID:PID etc.


@dataclass(frozen=True)
class SerialSettings:
    """Serial line parameters. String parity ('N'/'E'/'O') keeps this importable
    without pyserial; it is mapped to the pyserial constant only at open()."""

    port: str
    baudrate: int = 9600
    bytesize: int = 8
    parity: str = "N"        # 'N' | 'E' | 'O' | 'M' | 'S'
    stopbits: float = 1.0    # 1 | 1.5 | 2
    timeout: float = 0.4     # read timeout (s) — short so a dead bus can't hang
    write_timeout: float = 0.4

    def label(self) -> str:
        return f"{self.port}@{self.baudrate} {self.bytesize}{self.parity}{int(self.stopbits)}"


@runtime_checkable
class SerialLike(Protocol):
    """Minimal interface the sensor drivers depend on (real or fake)."""

    def transact(
        self,
        payload: bytes,
        *,
        read_size: int | None = ...,
        read_until: bytes | None = ...,
        timeout: float | None = ...,
    ) -> bytes: ...

    def close(self) -> None: ...


def list_serial_ports() -> list[SerialPortInfo]:
    """Enumerate available serial ports. Returns ``[]`` if pyserial is absent
    or enumeration fails — never raises."""
    if not HAS_PYSERIAL:
        return []
    try:
        out: list[SerialPortInfo] = []
        for p in _list_ports.comports():
            out.append(
                SerialPortInfo(
                    device=getattr(p, "device", "") or "",
                    description=getattr(p, "description", "") or "",
                    hwid=getattr(p, "hwid", "") or "",
                )
            )
        return out
    except Exception as exc:  # pragma: no cover - OS quirk
        logger.debug("list_serial_ports failed: %s", exc)
        return []


_PARITY_MAP = {"N": "PARITY_NONE", "E": "PARITY_EVEN", "O": "PARITY_ODD",
               "M": "PARITY_MARK", "S": "PARITY_SPACE"}
_STOPBITS_MAP = {1.0: "STOPBITS_ONE", 1.5: "STOPBITS_ONE_POINT_FIVE",
                 2.0: "STOPBITS_TWO"}
_BYTESIZE_MAP = {5: "FIVEBITS", 6: "SIXBITS", 7: "SEVENBITS", 8: "EIGHTBITS"}


class SerialTransport:
    """Real pyserial-backed transport. Lazy — the port is opened on first use.

    Construction never raises; :meth:`open` / :meth:`transact` raise
    :class:`SerialUnavailable` so callers handle a missing port in one place.
    """

    def __init__(self, settings: SerialSettings):
        self.settings = settings
        self._ser = None  # serial.Serial | None
        # A request/response link is only coherent if one transaction completes
        # before the next begins. Two threads DO share a transport now: the
        # EnvironmentMonitor loop polls temperature every interval while the
        # settings panel reads instrument state (heater range/setpoint) on
        # demand — interleaved write/read would splice one reply onto the other
        # request. Guard the whole flush→write→read sequence, not just the write.
        self._io_lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        if self._ser is not None:
            return
        if not HAS_PYSERIAL:
            raise SerialUnavailable("pyserial not installed")
        s = self.settings
        try:
            self._ser = _pyserial.Serial(
                port=s.port,
                baudrate=s.baudrate,
                bytesize=getattr(_pyserial, _BYTESIZE_MAP.get(s.bytesize, "EIGHTBITS")),
                parity=getattr(_pyserial, _PARITY_MAP.get(s.parity.upper(), "PARITY_NONE")),
                stopbits=getattr(_pyserial, _STOPBITS_MAP.get(float(s.stopbits), "STOPBITS_ONE")),
                timeout=s.timeout,
                write_timeout=s.write_timeout,
            )
        except Exception as exc:
            self._ser = None
            raise SerialUnavailable(f"open {s.label()} failed: {exc}") from exc

    def close(self) -> None:
        # Deliberately does NOT take _io_lock. Closing the port is what CANCELS
        # a read that is blocked in the kernel — EnvironmentMonitor.stop() relies
        # on exactly that to unblock a wedged loop thread. Taking the lock here
        # would make close() wait on the very read it is meant to interrupt.
        ser = self._ser
        self._ser = None
        if ser is not None:
            try:
                ser.close()
            except Exception:  # pragma: no cover
                pass

    @property
    def is_open(self) -> bool:
        return self._ser is not None and bool(getattr(self._ser, "is_open", False))

    # -- transactional I/O -------------------------------------------------
    def transact(
        self,
        payload: bytes,
        *,
        read_size: int | None = None,
        read_until: bytes | None = None,
        timeout: float | None = None,
    ) -> bytes:
        """Write *payload*, then read the reply.

        Exactly one of ``read_size`` (Modbus: fixed-length frame) or
        ``read_until`` (SCPI: terminator-delimited line) should be given;
        if both are None a single short read is returned. Flushes stale input
        before writing so a previous partial reply can't corrupt this one.
        Raises :class:`SerialUnavailable` if the port isn't usable.
        """
        with self._io_lock:
            if self._ser is None:
                self.open()
            ser = self._ser
            if ser is None:  # pragma: no cover - open() would have raised
                raise SerialUnavailable("port not open")

            if timeout is not None:
                try:
                    ser.timeout = timeout
                except Exception:  # pragma: no cover
                    pass

            try:
                try:
                    ser.reset_input_buffer()
                except Exception:  # pragma: no cover
                    pass
                ser.write(payload)
                try:
                    ser.flush()
                except Exception:  # pragma: no cover
                    pass

                if read_until is not None:
                    data = ser.read_until(read_until)
                elif read_size is not None:
                    data = ser.read(read_size)
                else:
                    data = ser.read(256)
                return bytes(data or b"")
            except SerialUnavailable:
                raise
            except Exception as exc:
                raise SerialUnavailable(
                    f"transact on {self.settings.label()} failed: {exc}") from exc

    # context-manager sugar
    def __enter__(self) -> "SerialTransport":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_transport(settings: SerialSettings) -> SerialTransport:
    """Factory used by sensors; kept as a seam tests can monkeypatch."""
    return SerialTransport(settings)


__all__ = [
    "HAS_PYSERIAL",
    "SerialUnavailable",
    "SerialPortInfo",
    "SerialSettings",
    "SerialLike",
    "SerialTransport",
    "list_serial_ports",
    "open_transport",
]
