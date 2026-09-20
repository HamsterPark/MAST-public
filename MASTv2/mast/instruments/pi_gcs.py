"""Physik Instrumente GCS controllers: E-816 (piezo) and E-861 (NEXACT).

Speaks the text-based **GCS command set** directly over a serial link
(RS-232 / USB-VCP) — no PIPython dependency, no new pins in
requirements-v2.txt. Replaces the vendor ``GCS_LabVIEW`` VI stacks found
in the lab's LabVIEW project.

Command subset used (GCS 1/2 common ground, verified against the E-816
and E-861 vendor SDKs shipped with the LabVIEW project):

- ``*IDN?``            identify
- ``SVO <ax> <0|1>``   servo (closed-loop) off/on
- ``MOV <ax> <pos>``   closed-loop absolute move
- ``POS? <ax>``        current position       → ``<ax>=<float>``
- ``ONT? <ax>``        on-target flag         → ``<ax>=<0|1>``
- ``ERR?``             last error code        → ``<int>``
- ``STP``              smooth stop (sets error 10, which we swallow)
- ``FRF <ax>`` / ``FRF? <ax>``  reference move + referenced query (E-861)

Model differences handled here:

- **E-816**: axes are letters (``A``-``D``), no homing concept (piezo with
  absolute sensor) — ``home()`` raises. Servo must be ON for MOV.
- **E-861**: axis is usually ``"1"``; a closed-loop absolute MOV requires
  the axis to be referenced once (FRF). ``home()`` = FRF.
"""

from __future__ import annotations

import logging

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
)

logger = logging.getLogger(__name__)

__all__ = ["PiGcsAxis", "PiGcsController", "parse_gcs_value"]

#: GCS error codes we intentionally swallow: 10 = "controller was stopped
#: by command" — the documented, expected side-effect of STP.
_BENIGN_ERRORS = {10}


def parse_gcs_value(line: str, axis: str) -> str:
    """Extract the value from a GCS ``<axis>=<value>`` reply line.

    Accepts replies with or without the axis prefix (older E-816 firmware
    replies to single-axis queries with a bare value)."""
    text = line.strip()
    if not text:
        raise InstrumentError(f"empty GCS reply for axis {axis!r}")
    for part in text.splitlines():
        part = part.strip()
        if "=" in part:
            key, _, value = part.partition("=")
            if key.strip().lstrip("\x00").upper() == str(axis).upper():
                return value.strip()
    # bare-value fallback (single-axis query, prefix-less firmware)
    if "=" not in text:
        return text
    raise InstrumentError(f"GCS reply {line!r} has no value for axis {axis!r}")


class PiGcsAxis(MotionAxis):
    """One GCS axis (letter for E-816, digit for E-861)."""

    def __init__(self, config: AxisConfig, controller: "PiGcsController"):
        super().__init__(config, controller)
        self._pi: PiGcsController = controller
        self._ax = str(config.channel)

    # -- primitives (controller lock already held by the base class) --------

    def _move_abs_raw(self, target: float) -> None:
        self._pi._command(f"MOV {self._ax} {target:.6f}")

    def _get_position_raw(self) -> float:
        reply = self._pi._query(f"POS? {self._ax}")
        try:
            return float(parse_gcs_value(reply, self._ax))
        except ValueError as exc:
            raise InstrumentError(f"bad POS? reply {reply!r}") from exc

    def _get_status_raw(self) -> AxisStatus:
        pos = self._get_position_raw()
        on_target: bool | None = None
        try:
            ont = parse_gcs_value(self._pi._query(f"ONT? {self._ax}"), self._ax)
            on_target = ont.strip() in ("1", "True")
        except InstrumentError:
            pass  # older firmware without ONT? — fall back to not-moving
        homed: bool | None = None
        if self._pi.model_needs_reference:
            try:
                frf = parse_gcs_value(self._pi._query(f"FRF? {self._ax}"), self._ax)
                homed = frf.strip() == "1"
            except InstrumentError:
                homed = None
        return AxisStatus(
            position=pos,
            # GCS has no universal "moving" query at this subset level; the
            # on-target flag is authoritative for settledness.
            moving=(on_target is False),
            on_target=on_target,
            homed=homed,
        )

    def _stop_raw(self) -> None:
        self._pi._stop_all_raw()

    def _home_raw(self) -> None:
        if not self._pi.model_needs_reference:
            raise InstrumentError(
                f"PI {self._pi.model}: axis {self._ax} has an absolute sensor, "
                "no homing procedure exists"
            )
        self._pi._command(f"FRF {self._ax}")


class PiGcsController(MotionController):
    """One PI GCS controller on a serial link.

    ``model`` selects the small behavioural differences ("E-816" vs
    "E-861"); anything else defaults to E-861-like behaviour (referenced
    stepper). ``transport`` may be injected for tests.
    """

    IO_TIMEOUT_S = 0.5
    TERMINATOR = b"\n"

    def __init__(
        self,
        *,
        port: str | None = None,
        baudrate: int = 115200,
        model: str = "E-861",
        axes: list[AxisConfig] | None = None,
        servo_on_connect: bool = True,
        transport: SerialLike | None = None,
    ):
        super().__init__()
        if transport is None and not port:
            raise ValueError("PiGcsController needs either port or transport")
        self._port = port
        self._baudrate = int(baudrate)
        self.model = model.upper().strip()
        self._axis_configs = list(axes or [])
        self._servo_on_connect = bool(servo_on_connect)
        self._transport: SerialLike | None = transport
        self._owns_transport = transport is None

    @property
    def model_needs_reference(self) -> bool:
        """E-861 (and other steppers) must be referenced before MOV;
        the E-816 piezo has an absolute sensor."""
        return self.model != "E-816"

    # -- lifecycle -----------------------------------------------------------

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
                raise InstrumentUnavailable(
                    f"PI {self.model} on {self._port}: {exc}"
                ) from exc
            self._transport = t
        if self._servo_on_connect:
            for cfg in self._axis_configs:
                try:
                    self._command(f"SVO {cfg.channel} 1")
                except InstrumentError as exc:
                    logger.warning(
                        "PI %s: servo-on for axis %s failed: %s",
                        self.model, cfg.channel, exc,
                    )

    def _close_raw(self) -> None:
        t = self._transport
        if t is not None and self._owns_transport:
            try:
                t.close()
            except Exception:  # noqa: BLE001
                pass
            self._transport = None

    def _build_axes(self) -> dict[str, MotionAxis]:
        return {cfg.name: PiGcsAxis(cfg, self) for cfg in self._axis_configs}

    # -- GCS I/O ---------------------------------------------------------------

    def _raw_transact(self, line: str, *, expect_reply: bool) -> str:
        t = self._transport
        if t is None:
            raise InstrumentUnavailable(f"PI {self.model} transport not connected")
        payload = line.strip().encode("ascii") + self.TERMINATOR
        try:
            if expect_reply:
                raw = t.transact(
                    payload, read_until=self.TERMINATOR, timeout=self.IO_TIMEOUT_S
                )
                return raw.decode("ascii", errors="replace")
            # command with no reply — write, then drain nothing
            t.transact(payload, read_size=0, timeout=0.05)
            return ""
        except SerialUnavailable as exc:
            raise InstrumentUnavailable(f"PI {self.model} I/O failed: {exc}") from exc

    def _query(self, cmd: str) -> str:
        reply = self._raw_transact(cmd, expect_reply=True)
        if not reply.strip():
            raise InstrumentError(f"PI {self.model}: no reply to {cmd!r}")
        return reply

    def _command(self, cmd: str) -> None:
        """Set-command + mandatory ERR? check (GCS set commands are silent;
        the error queue is the only acknowledgement)."""
        self._raw_transact(cmd, expect_reply=False)
        self._check_error(context=cmd)

    def _check_error(self, *, context: str) -> None:
        reply = self._query("ERR?")
        try:
            code = int(reply.strip().split()[-1])
        except (ValueError, IndexError) as exc:
            raise InstrumentError(
                f"PI {self.model}: unparseable ERR? reply {reply!r} after {context!r}"
            ) from exc
        if code and code not in _BENIGN_ERRORS:
            raise InstrumentError(
                f"PI {self.model}: GCS error {code} after {context!r}"
            )

    def _stop_all_raw(self) -> None:
        """STP stops all axes and by design queues error 10 — swallow it."""
        self._raw_transact("STP", expect_reply=False)
        try:
            self._check_error(context="STP")
        except InstrumentError:
            # A stop must never raise on the panic path; the error queue is
            # cleared by the ERR? read itself.
            pass
