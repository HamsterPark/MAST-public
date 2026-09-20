"""Lakeshore temperature monitor / controller driver (SCPI over RS-232/485).

Lakeshore cryogenic monitors and controllers (Model 218 / 224 / 331 / 332 /
335 / 336 / 340 …) speak a common SCPI-like ASCII protocol:

* Identify: ``*IDN?`` → ``LSCI,MODEL336,<serial>,<firmware>`` (CRLF terminated).
  The leading ``LSCI`` token is how we auto-recognise the instrument.
* Read Kelvin: ``KRDG? <input>`` → e.g. ``+273.150E+0`` (CRLF). ``CRDG?`` gives
  Celsius, ``SRDG?`` raw sensor units.
* Inputs are letters ``A``..``D`` on the 33x/335/336 family and numbers
  ``1``..``8`` on the 218; both are supported via the ``channel`` argument.

Line settings differ across the range (the 218 defaults to 9600, the 336 to
57600), all **7 data bits, odd parity, 1 stop bit**. :func:`probe_lakeshore`
tries the common combinations so auto-detection works without the user knowing
the exact baud.

The bundled "另一个串口程序" (a LabVIEW logger) talks the same serial
protocol; this driver lets MAST read the instrument directly instead of
scraping the logger, while leaving the logger free to run if the user prefers.
"""

from __future__ import annotations

import logging
import re

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

LSCI_TERMINATOR = b"\r\n"

# (baudrate, bytesize, parity, stopbits) combos tried during auto-detection,
# most-common first. All Lakeshore serial links are 7-bit / odd-parity.
LSCI_SERIAL_CANDIDATES: tuple[tuple[int, int, str, float], ...] = (
    (57600, 7, "O", 1),   # 336 / 335 default
    (9600, 7, "O", 1),    # 218 / 33x default
    (115200, 7, "O", 1),  # 336 high-speed option
)

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# SECURITY: only read/query commands are ever sent. Whitelisting the command
# and restricting the channel to an alphanumeric token prevents a crafted
# config entry (reachable via the POST /environment/sensors endpoint) from
# injecting an arbitrary SCPI WRITE — e.g. channel="A\r\nRANGE 1,3" would
# otherwise drive a heater on a controller-type Lakeshore. MAST never writes
# to the instrument from this path; these are temperature reads only.
_SAFE_COMMANDS = frozenset({"KRDG?", "CRDG?", "SRDG?"})
_CHANNEL_RE = re.compile(r"^[A-Za-z0-9]{1,4}$")

# Additional READ-ONLY queries used by discovery and the instrument-state
# readout (heater range / output / setpoint / control mode). Every entry ends
# in '?', and :func:`_query` re-checks that at call time — so no code path in
# this module can emit a Lakeshore SETTER (SETP / RANGE / MOUT / OUTMODE / PID
# without the '?'), which is what would actually drive a heater. The state
# readout exists precisely so MAST can SHOW whether a heater is on without ever
# being able to turn one on.
_STATE_QUERIES = frozenset({
    "*IDN?",       # identity
    "INNAME?",     # user-assigned input label, e.g. "SPM" / "Magnet"
    "INTYPE?",     # sensor type / range / compensation
    "RDGST?",      # reading status bits (invalid / over- / under-range)
    "RANGE?",      # heater range: 0 = OFF
    "HTR?",        # heater output, %
    "SETP?",       # setpoint
    "OUTMODE?",    # control mode + control input + powerup-enable
    "RAMP?",       # setpoint ramp on/off + rate
}) | _SAFE_COMMANDS


def _sanitize_command(command: str) -> str:
    cmd = str(command).strip().upper()
    if cmd not in _SAFE_COMMANDS:
        logger.warning("Lakeshore: rejecting non-whitelisted command %r → KRDG?", command)
        return "KRDG?"
    return cmd


def _sanitize_channel(channel: str) -> str:
    # None / "" must fall back to 'A' — note str(None) == "None" happens to be a
    # 4-char alphanumeric token that would otherwise pass the regex and get sent
    # as a literal channel "None" (review 2.1.13 #23).
    if channel is None:
        return "A"
    ch = str(channel).strip()
    if ch and _CHANNEL_RE.match(ch):
        return ch
    logger.warning("Lakeshore: rejecting unsafe channel %r → A", channel)
    return "A"


# Reading-unit implied by the SCPI query, so a CRDG?/SRDG? reading isn't
# mislabelled as kelvin (review 2.1.13 #7).
_UNIT_BY_COMMAND = {"KRDG?": "K", "CRDG?": "°C", "SRDG?": ""}


class LakeshoreError(ValueError):
    """Unparseable Lakeshore reply."""


def parse_idn(raw: bytes | str) -> tuple[bool, str]:
    """Return ``(is_lakeshore, model)`` from a ``*IDN?`` reply.

    Lakeshore replies start with the manufacturer token ``LSCI`` (older units
    spell out ``LAKESHORE``). The second comma-field is the model, e.g.
    ``MODEL336``.
    """
    text = raw.decode("ascii", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    text = text.strip()
    parts = [p.strip() for p in text.split(",")]
    first = parts[0].upper() if parts else ""
    # Field-based, not substring: the manufacturer token is the FIRST CSV field
    # ("LSCI" exact, or "LAKESHORE..." on older units). A bare substring match
    # would mis-identify a device whose payload merely contains "LSCI", and an
    # empty model field must NOT be accepted as a Lakeshore (review 2.1.13 #6).
    is_lsci = first == "LSCI" or first.startswith("LAKESHORE")
    model = parts[1] if len(parts) > 1 else ""
    return (is_lsci and bool(model)), model


def parse_reading(raw: bytes | str) -> float:
    """Parse a ``KRDG?`` reply (e.g. ``+273.150E+0``) → float kelvin.

    Anchored match (not search): a malformed reply like ``ERR 5`` must raise
    rather than silently return 5 as a plausible-but-wrong temperature
    (review 2.1.13 #5)."""
    text = raw.decode("ascii", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    m = _NUM_RE.match(text.strip())
    if not m:
        raise LakeshoreError(f"reply is not a leading number: {text!r}")
    return float(m.group(0))


def default_channels(model: str) -> list[str]:
    """Best-effort default input list for a model string."""
    m = (model or "").upper()
    if "218" in m:
        return [str(i) for i in range(1, 9)]   # 8 inputs
    if "224" in m:
        return ["A", "B", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5"]
    if any(t in m for t in ("336", "350")):   # 4-input controllers
        return ["A", "B", "C", "D"]
    if any(t in m for t in ("335", "331", "332", "340")):  # 2-input
        return ["A", "B"]
    return ["A"]


class LakeshoreInfo:
    """Result of a successful probe."""

    __slots__ = ("model", "idn", "settings")

    def __init__(self, model: str, idn: str, settings: SerialSettings):
        self.model = model
        self.idn = idn
        self.settings = settings

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"LakeshoreInfo(model={self.model!r}, {self.settings.label()})"


def probe_lakeshore(port: str, *, candidates=LSCI_SERIAL_CANDIDATES,
                    timeout: float = 0.5) -> "LakeshoreInfo | None":
    """Try to identify a Lakeshore instrument on *port*.

    Iterates the serial-parameter candidates, sends ``*IDN?``, and returns a
    :class:`LakeshoreInfo` on the first reply containing ``LSCI``. Returns
    ``None`` (never raises) if nothing matches / no pyserial / port busy.
    """
    for baud, bits, parity, stop in candidates:
        settings = SerialSettings(port=port, baudrate=baud, bytesize=bits,
                                  parity=parity, stopbits=stop, timeout=timeout)
        transport: SerialTransport | None = None
        try:
            transport = open_transport(settings)
            reply = transport.transact(b"*IDN?" + LSCI_TERMINATOR,
                                       read_until=b"\n", timeout=timeout)
            ok, model = parse_idn(reply)
            if ok:
                return LakeshoreInfo(model=model, idn=reply.decode("ascii", "replace").strip(),
                                     settings=settings)
        except (SerialUnavailable, Exception):
            pass
        finally:
            if transport is not None:
                transport.close()
    return None


# ── read-only instrument state ──────────────────────────────────────────────
# Decode tables for the Model 33x/335/336 family. Unknown codes fall back to the
# raw number rather than a guess, so a model we haven't verified can't be shown
# a confident-but-wrong label.
_OUTMODE_MODES = {0: "关闭", 1: "闭环 PID", 2: "分段 (Zone)", 3: "开环",
                  4: "监视输出", 5: "升温电源"}
_OUTMODE_INPUTS = {0: "无", 1: "A", 2: "B", 3: "C", 4: "D"}
_RANGE_LABELS = {0: "关闭", 1: "低", 2: "中", 3: "高"}
_SENSOR_TYPES = {0: "未启用", 1: "二极管", 2: "铂电阻 RTD", 3: "NTC 热敏电阻",
                 4: "热电偶", 5: "电容"}
# RDGST? bit flags — a non-zero status means the kelvin number is NOT trustworthy.
_RDGST_FLAGS = ((1, "读数无效"), (16, "温度低于量程"), (32, "温度高于量程"),
                (64, "传感器读数为零"), (128, "传感器超量程"))

# The health of one input's reading, as three DISTINCT answers. The third is the
# whole point: "we asked and it is clean" and "we never got an answer" are
# different facts about the world, and only the first of them backs a number.
HEALTH_CLEAN = "clean"      # RDGST? answered; no fault bit set
HEALTH_FAULTY = "faulty"    # RDGST? answered; at least one fault bit set
HEALTH_UNKNOWN = "unknown"  # RDGST? was not asked / did not answer / unparseable


def decode_reading_status(code: int) -> list[str]:
    """Decode a ``RDGST?`` *bitmask* into human-readable faults.

    ``[]`` means "the instrument answered and every fault bit is clear". That is
    a MEASUREMENT, not a default — never manufacture it from a non-answer. Use
    :func:`parse_reading_status`, which keeps "did not answer" as ``None``.

    The argument is a bitmask, which leaves no room for an in-band sentinel: a
    ``-1`` meaning "unknown" decodes as ``-1 & 1 == 1`` → 读数无效 and then every
    other flag too, inventing a faulty sensor the instrument never reported.
    """
    return [label for bit, label in _RDGST_FLAGS if code & bit]


def parse_reading_status(reply: "str | bytes | None") -> "list[str] | None":
    """``RDGST?`` reply text → fault labels, or ``None`` when it never answered.

    This is where the tri-state has to be established, because the reply text is
    the LAST layer at which "no answer" is still distinguishable from "zero".
    Everything below turns a silent link into an empty string without raising —
    pyserial's ``read_until`` returns ``b""`` at timeout, ``transact`` passes it
    through, ``_query`` decodes it to ``""`` — and one line later the difference
    is gone for good: ``int("" or 0)`` is ``0``, and ``0`` is a perfectly valid
    *clean* status word.

    Returns ``None`` — not ``[]``, and not a sentinel bitmask — for an empty
    reply, a non-integer reply, and a negative one (the status word is 0..255; a
    negative cannot be a bitmask and decoding it would invent faults).
    """
    if reply is None:
        return None
    text = (reply.decode("ascii", "replace")
            if isinstance(reply, (bytes, bytearray)) else str(reply)).strip()
    if not text:
        return None
    try:
        code = int(text)
    except (TypeError, ValueError):
        logger.debug("Lakeshore: unparseable RDGST? reply %r — status unknown", text)
        return None
    if code < 0:
        logger.debug("Lakeshore: negative RDGST? reply %r — status unknown", text)
        return None
    return decode_reading_status(code)


def _assert_query(verb: str) -> str:
    """Gatekeeper: only whitelisted queries, and every one must end in '?'.

    Both halves matter. The whitelist stops an unexpected verb; the trailing-'?'
    check is the structural invariant — in the Lakeshore command set the setter
    and the getter differ only by that character (``RANGE 1,3`` drives a heater,
    ``RANGE? 1`` asks about one). A future edit that adds a verb without the '?'
    fails here instead of on the instrument.
    """
    v = str(verb).strip().upper()
    if not v.endswith("?") or v not in _STATE_QUERIES:
        raise LakeshoreError(f"refusing non-query command {verb!r}")
    return v


def _query(transport: SerialLike, verb: str, arg: str = "", *,
           timeout: float = 0.5) -> str:
    """Send one whitelisted query and return the trimmed reply text.

    The argument is validated strictly rather than coerced: :func:`_sanitize_channel`
    falls back to ``"A"`` on garbage, which is right for a user-supplied channel
    but wrong here — silently rewriting an output number would make the panel
    report output 1's heater state under output 3's heading. Bad argument →
    raise, so the caller degrades that one field to "unknown".
    """
    v = _assert_query(verb)
    payload = v
    if arg:
        token = str(arg).strip()
        if not _CHANNEL_RE.match(token):
            raise LakeshoreError(f"unsafe query argument {arg!r}")
        payload = f"{v} {token}"
    raw = transport.transact(payload.encode("ascii") + LSCI_TERMINATOR,
                             read_until=b"\n", timeout=timeout)
    return raw.decode("ascii", "replace").strip()


def _csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        try:
            out.append(int(float(part.strip())))
        except ValueError:
            out.append(-1)
    return out


class ChannelState:
    """Live state of one sensor input, as shown in the settings readout.

    ``faults`` is THREE-state, and the third state is load-bearing:

      * ``[]``     — ``RDGST?`` answered and no fault bit is set. The kelvin
        number carries the instrument's own backing.
      * ``[...]``  — ``RDGST?`` answered with fault bits. The number is not
        trustworthy (an over-range sensor still returns a plausible float).
      * ``None``   — ``RDGST?`` was never asked, or never answered. **Nothing is
        known** about this reading's validity.

    It used to be two-state (``self.faults = faults or []``), which made the
    third case indistinguishable from the first: an input whose status query
    timed out was registered as a healthy thermometer, and its temperature went
    on to satisfy a conduct's 换样品 wait gate. The gate's own ``stale_after_s``
    cannot catch that — the link is alive and the readings are fresh, so
    freshness answers "is this message old?", not "is this number backed?".

    Ask :attr:`health` rather than re-spelling the three-way test per caller.
    """

    __slots__ = ("channel", "label", "kelvin", "celsius", "sensor_value",
                 "sensor_type", "faults")

    def __init__(self, channel: str, label: str = "", kelvin: float | None = None,
                 celsius: float | None = None, sensor_value: float | None = None,
                 sensor_type: str = "", faults: "list[str] | None" = None):
        self.channel = channel
        self.label = label
        self.kelvin = kelvin
        self.celsius = celsius
        self.sensor_value = sensor_value
        self.sensor_type = sensor_type
        # NOT ``faults or []`` — that fold IS the defect. An omitted argument
        # means "nobody asked this channel", which is exactly what None says.
        self.faults: "list[str] | None" = faults

    @property
    def health(self) -> str:
        """:data:`HEALTH_CLEAN` / :data:`HEALTH_FAULTY` / :data:`HEALTH_UNKNOWN`.

        The single definition of the three-way rule. A distinction is only worth
        having if every consumer draws it the same way, and the cheapest way to
        guarantee that is to give it one name and one implementation.
        """
        if self.faults is None:
            return HEALTH_UNKNOWN
        return HEALTH_FAULTY if self.faults else HEALTH_CLEAN

    def as_dict(self) -> dict:
        return {"channel": self.channel, "label": self.label, "kelvin": self.kelvin,
                "celsius": self.celsius, "sensor_value": self.sensor_value,
                "sensor_type": self.sensor_type,
                # ``None`` survives onto the wire as JSON ``null``: null = never
                # asked, [] = asked and clean. Collapsing them here would undo
                # the distinction one layer short of the panel that shows it.
                "faults": None if self.faults is None else list(self.faults),
                # Sent alongside so a client never has to re-derive the verdict.
                # A client-side ``faults ?? []`` reads null as clean — which is
                # precisely the fold this whole change removes.
                "health": self.health}


class OutputState:
    """Live state of one control output — this is the heater readout.

    ``heater_on`` is derived from the RANGE code alone: on this instrument family
    range 0 means the output is off no matter what the setpoint or control mode
    say. A parked setpoint of 500 K with range 0 is harmless, and the panel must
    not read that as "heating".
    """

    __slots__ = ("output", "range_code", "range_label", "heater_pct", "setpoint",
                 "mode", "control_input", "powerup_enabled", "ramping", "ramp_rate")

    def __init__(self, output: int, range_code: int = -1, range_label: str = "",
                 heater_pct: float | None = None, setpoint: float | None = None,
                 mode: str = "", control_input: str = "",
                 powerup_enabled: bool = False, ramping: bool = False,
                 ramp_rate: float | None = None):
        self.output = output
        self.range_code = range_code
        self.range_label = range_label
        self.heater_pct = heater_pct
        self.setpoint = setpoint
        self.mode = mode
        self.control_input = control_input
        self.powerup_enabled = powerup_enabled
        self.ramping = ramping
        self.ramp_rate = ramp_rate

    @property
    def heater_on(self) -> bool:
        return self.range_code > 0

    def as_dict(self) -> dict:
        return {"output": self.output, "range_code": self.range_code,
                "range_label": self.range_label, "heater_on": self.heater_on,
                "heater_pct": self.heater_pct, "setpoint": self.setpoint,
                "mode": self.mode, "control_input": self.control_input,
                "powerup_enabled": self.powerup_enabled, "ramping": self.ramping,
                "ramp_rate": self.ramp_rate}


class LakeshoreSnapshot:
    """Everything MAST can read off the instrument without writing to it."""

    __slots__ = ("model", "idn", "firmware", "serial_number", "port",
                 "channels", "outputs", "settings")

    def __init__(self, model: str = "", idn: str = "", firmware: str = "",
                 serial_number: str = "", port: str = "",
                 channels: list[ChannelState] | None = None,
                 outputs: list[OutputState] | None = None,
                 settings: SerialSettings | None = None):
        self.model = model
        self.idn = idn
        self.firmware = firmware
        self.serial_number = serial_number
        self.port = port
        self.channels = channels or []
        self.outputs = outputs or []
        # Line parameters that actually answered, so "adopt" pins the exact baud
        # instead of making every boot re-run the auto-baud search.
        self.settings = settings

    @property
    def any_heater_on(self) -> bool:
        return any(o.heater_on for o in self.outputs)

    def as_dict(self) -> dict:
        return {"model": self.model, "idn": self.idn, "firmware": self.firmware,
                "serial_number": self.serial_number, "port": self.port,
                "any_heater_on": self.any_heater_on,
                "channels": [c.as_dict() for c in self.channels],
                "outputs": [o.as_dict() for o in self.outputs]}


def default_outputs(model: str) -> list[int]:
    """Control-output numbers for a model (monitors have none)."""
    m = (model or "").upper()
    if any(t in m for t in ("218", "224")):
        return []                       # monitor-only, no heater
    if any(t in m for t in ("336", "350")):
        return [1, 2, 3, 4]
    return [1, 2]                       # 335 / 33x / 340


def _float_or_none(text: str) -> float | None:
    try:
        return parse_reading(text)
    except LakeshoreError:
        return None


def read_snapshot(transport: SerialLike, *, model: str = "", idn: str = "",
                  port: str = "", channels: list[str] | None = None,
                  timeout: float = 0.5) -> LakeshoreSnapshot:
    """Read the full read-only instrument state over an OPEN transport.

    Every failure degrades to a missing field rather than raising: a partially
    answered instrument still yields a useful panel, and this is called from a
    settings page that must never 500. Reuses the caller's transport so it can
    run against the port the EnvironmentMonitor already owns.
    """
    if not idn:
        try:
            idn = _query(transport, "*IDN?", timeout=timeout)
        except Exception:  # noqa: BLE001 - degrade to an empty identity
            idn = ""
    parts = [p.strip() for p in idn.split(",")] if idn else []
    if not model and len(parts) > 1:
        model = parts[1]
    snap = LakeshoreSnapshot(
        model=model, idn=idn, port=port,
        serial_number=parts[2] if len(parts) > 2 else "",
        firmware=parts[3] if len(parts) > 3 else "",
    )

    for ch in (channels or default_channels(model)):
        state = ChannelState(channel=ch)
        for verb, setter in (
            ("INNAME?", lambda v, s=state: setattr(s, "label", v)),
            ("KRDG?", lambda v, s=state: setattr(s, "kelvin", _float_or_none(v))),
            ("CRDG?", lambda v, s=state: setattr(s, "celsius", _float_or_none(v))),
            ("SRDG?", lambda v, s=state: setattr(s, "sensor_value", _float_or_none(v))),
        ):
            try:
                setter(_query(transport, verb, ch, timeout=timeout))
            except Exception:  # noqa: BLE001 - leave the field unset
                pass
        try:
            codes = _csv_ints(_query(transport, "INTYPE?", ch, timeout=timeout))
            if codes:
                state.sensor_type = _SENSOR_TYPES.get(codes[0], str(codes[0]))
        except Exception:  # noqa: BLE001
            pass
        try:
            # parse_reading_status, NOT ``int(reply or 0)``: the ``or 0`` is what
            # turned a timed-out status query into the bitmask 0, i.e. "clean".
            state.faults = parse_reading_status(
                _query(transport, "RDGST?", ch, timeout=timeout))
        except Exception:  # noqa: BLE001
            # The query itself failed (port dropped mid-snapshot, argument
            # rejected, …). Say "unknown" out loud instead of leaning on the
            # constructor default: a premise this load-bearing should not be
            # owned by a default two hundred lines away.
            state.faults = None
        snap.channels.append(state)

    for out in default_outputs(model):
        o = OutputState(output=out)
        try:
            o.range_code = int(_query(transport, "RANGE?", str(out), timeout=timeout))
            o.range_label = _RANGE_LABELS.get(o.range_code, str(o.range_code))
        except Exception:  # noqa: BLE001 - unknown range → heater_on stays False
            pass
        try:
            o.heater_pct = _float_or_none(_query(transport, "HTR?", str(out), timeout=timeout))
        except Exception:  # noqa: BLE001
            pass
        try:
            o.setpoint = _float_or_none(_query(transport, "SETP?", str(out), timeout=timeout))
        except Exception:  # noqa: BLE001
            pass
        try:
            codes = _csv_ints(_query(transport, "OUTMODE?", str(out), timeout=timeout))
            if len(codes) > 0:
                o.mode = _OUTMODE_MODES.get(codes[0], str(codes[0]))
            if len(codes) > 1:
                o.control_input = _OUTMODE_INPUTS.get(codes[1], str(codes[1]))
            if len(codes) > 2:
                o.powerup_enabled = bool(codes[2])
        except Exception:  # noqa: BLE001
            pass
        try:
            # "<on/off>,<K/min>" — the rate is a FLOAT (0.1 K/min is a normal
            # ramp), so it must not go through the int decoder used for the
            # enum-valued replies.
            fields = _query(transport, "RAMP?", str(out), timeout=timeout).split(",")
            if fields:
                o.ramping = bool(int(float(fields[0].strip() or 0)))
            if len(fields) > 1:
                o.ramp_rate = _float_or_none(fields[1].strip())
        except Exception:  # noqa: BLE001
            pass
        snap.outputs.append(o)
    return snap


def discover_lakeshore(port: str, *, candidates=LSCI_SERIAL_CANDIDATES,
                       timeout: float = 0.5) -> "LakeshoreSnapshot | None":
    """Probe *port* and, if a Lakeshore answers, read its full state in one go.

    Used by the "扫描设备接口" flow: the confirmation dialog needs the model AND
    the live per-channel values, so the user can tell at a glance whether the
    identification is right before adopting it. Opens and closes its own
    transport; returns None (never raises) when nothing answers.
    """
    info = probe_lakeshore(port, candidates=candidates, timeout=timeout)
    if info is None:
        return None
    transport: SerialTransport | None = None
    try:
        transport = open_transport(info.settings)
        snap = read_snapshot(transport, model=info.model, idn=info.idn,
                             port=port, timeout=timeout)
        snap.settings = info.settings
        return snap
    except Exception as exc:  # noqa: BLE001 - identity is still worth returning
        logger.debug("discover_lakeshore(%s): state read failed: %s", port, exc)
        return LakeshoreSnapshot(model=info.model, idn=info.idn, port=port,
                                 settings=info.settings)
    finally:
        if transport is not None:
            transport.close()


class LakeshoreTemperatureSensor(EnvironmentSensor):
    """One Lakeshore input channel as an :class:`EnvironmentSensor` (kelvin).

    A multi-channel instrument is represented as several sensors with distinct
    names — that satisfies "可能有不止一个温度计 + 用户命名" without any special
    multiplexing in the monitor. Open/transact failures degrade to
    ``unavailable`` / ``error`` rather than crashing.
    """

    def __init__(
        self,
        *,
        name: str = "temperature",
        port: str | None = None,
        channel: str = "A",
        settings: SerialSettings | None = None,
        unit: str = "K",
        alarm: AlarmSpec | None = None,
        timeout: float = 0.5,
        transport: SerialLike | None = None,
        command: str = "KRDG?",
    ):
        self._name = name
        self._channel = _sanitize_channel(channel)
        self._alarm = alarm or AlarmSpec()
        self._timeout = timeout
        self._command = _sanitize_command(command)
        # If the caller left the default "K" but the command reads Celsius /
        # sensor units, label it correctly; an explicit non-K unit is honoured.
        if unit == "K" and self._command != "KRDG?":
            self._unit = _UNIT_BY_COMMAND.get(self._command, "K")
        else:
            self._unit = unit
        self._transport: SerialLike | None = transport
        if settings is not None:
            self._settings = settings
        elif port is not None:
            self._settings = SerialSettings(
                port=port, baudrate=57600, bytesize=7, parity="O",
                stopbits=1, timeout=timeout)
        else:
            self._settings = None
        self._query = f"{self._command} {self._channel}".encode("ascii") + LSCI_TERMINATOR

    def name(self) -> str:
        return self._name

    def _ensure_transport(self) -> SerialLike:
        if self._transport is None:
            if self._settings is None:
                raise SerialUnavailable("no port configured")
            self._transport = open_transport(self._settings)
        return self._transport

    def _release_transport(self) -> None:
        """Close the OS handle without discarding a possibly-shared transport."""
        close = getattr(self._transport, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best-effort
                pass

    def read(self) -> SensorReading:
        try:
            transport = self._ensure_transport()
        except SerialUnavailable:
            return SensorReading(value=0.0, unit=self._unit, status="unavailable")
        try:
            reply = transport.transact(self._query, read_until=b"\n",
                                       timeout=self._timeout)
        except SerialUnavailable:
            # Drop the OS handle so a later read reconnects, but KEEP the
            # transport object: it is shared with the instrument's other inputs
            # (autodetect._shared_transport). Replacing it with a fresh one here
            # would open a second exclusive handle on a port a sibling sensor is
            # already holding, and that input would never recover.
            self._release_transport()
            return SensorReading(value=0.0, unit=self._unit, status="unavailable")
        try:
            kelvin = parse_reading(reply)
        except LakeshoreError as exc:
            logger.debug("Lakeshore %s parse error: %s", self._name, exc)
            return SensorReading(value=0.0, unit=self._unit, status="error")
        status = self._alarm.evaluate(kelvin, base_status="ok")
        return SensorReading(value=kelvin, unit=self._unit, status=status)

    def instrument_state(self, *, channels: list[str] | None = None
                         ) -> "LakeshoreSnapshot | None":
        """Full read-only instrument state over THIS sensor's open port.

        Reuses the sensor's own transport instead of opening a second handle —
        Windows COM ports are exclusive, so a settings panel that opened its own
        connection could never read an instrument the monitor is already
        polling. Serialisation against the monitor's 2 s poll is handled by the
        transport's I/O lock. Returns None if the port isn't usable.
        """
        try:
            transport = self._ensure_transport()
        except SerialUnavailable:
            return None
        port = self._settings.port if self._settings is not None else ""
        try:
            return read_snapshot(transport, port=port, channels=channels,
                                 timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 - panel must never 500
            logger.debug("instrument_state on %s failed: %s", self._name, exc)
            return None

    def close(self) -> None:
        """Release the OS handle. KEEPS the transport object.

        Discarding it here (``self._transport = None``) is what reopened
        /#15, which an earlier fix had already closed. The transport is SHARED
        with this instrument's other inputs — one Lake Shore 335 on COM13 serves
        both ``SPM`` and ``Magnet``, and a Windows COM port admits exactly one
        handle. Dropping the reference means the next read calls
        ``_ensure_transport``, which opens a *private* handle from ``_settings``;
        the sibling input then can never get the port back, and the two channels
        resume the anti-phase alternation, and readings for whichever one loses the
        handle stop being recorded.

        The path this runs on is not exotic: ``EnvironmentMonitor.stop()`` →
        ``_close_sensors()``, reached every time the operator presses
        「扫描设备接口」（``/api/environment/discover`` releases the bus and then
        restarts the SAME sensor objects). ``rescan`` / ``adopt`` happened to
        survive only because they rebuild the sensor list with a fresh pool.

        ``read()``'s own failure path already got this right — see
        ``_release_transport``. The two must agree, and a test now pins that.

        Reopening still works: ``SerialTransport.close()`` drops the pyserial
        object and ``transact()`` reopens on next use.
        """
        self._release_transport()


__all__ = [
    "parse_idn",
    "parse_reading",
    "default_channels",
    "default_outputs",
    "decode_reading_status",
    "parse_reading_status",
    "HEALTH_CLEAN",
    "HEALTH_FAULTY",
    "HEALTH_UNKNOWN",
    "probe_lakeshore",
    "discover_lakeshore",
    "read_snapshot",
    "ChannelState",
    "OutputState",
    "LakeshoreSnapshot",
    "LakeshoreInfo",
    "LakeshoreTemperatureSensor",
    "LakeshoreError",
    "LSCI_SERIAL_CANDIDATES",
]
