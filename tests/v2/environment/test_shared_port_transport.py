"""One physical port → one transport, shared by every sensor on it.

A Lake Shore 335 with inputs A and B is modelled as TWO sensors, but it is one
device on one cable. Windows opens COM ports exclusively, so if each sensor
lazily opened its own handle the first would win and the second input would
report ``unavailable`` forever — which is exactly what a two-channel config
would have hit on the instrument (2026-07-27).

**These tests only cover the CONFIGURED path** (``build_sensors_from_config``).
The autodetect path built its sensors without a shared transport until
2026-08-05 and hit this exact failure on the instrument; it is pinned separately in
``tests/v2/unit/environment/test_lakeshore_multi_channel.py``. Passing here has
never implied the other path is safe.
"""

from __future__ import annotations

from mast.environment.autodetect import build_sensors_from_config


def _two_channel_cfg() -> dict:
    return {
        "autodetect": False,
        "sensors": [
            {"id": "ls_a", "name": "SPM", "type": "lakeshore_temp", "port": "COM13",
             "channel": "A", "baudrate": 57600, "bytesize": 7, "parity": "O", "stopbits": 1},
            {"id": "ls_b", "name": "Magnet", "type": "lakeshore_temp", "port": "COM13",
             "channel": "B", "baudrate": 57600, "bytesize": 7, "parity": "O", "stopbits": 1},
        ],
    }


def test_both_inputs_share_one_transport() -> None:
    sensors, used = build_sensors_from_config(_two_channel_cfg())

    assert len(sensors) == 2
    assert used == {"COM13"}
    a, b = sensors
    assert a._transport is not None
    assert a._transport is b._transport, "each input opened its own COM13 handle"


def test_different_ports_do_not_share() -> None:
    cfg = _two_channel_cfg()
    cfg["sensors"][1]["port"] = "COM17"
    sensors, used = build_sensors_from_config(cfg)

    assert used == {"COM13", "COM17"}
    assert sensors[0]._transport is not sensors[1]._transport


def test_building_never_opens_the_port() -> None:
    """Construction must stay lazy: building the sensor set happens at boot on
    machines where the instrument may not be plugged in at all."""
    sensors, _ = build_sensors_from_config(_two_channel_cfg())
    for s in sensors:
        assert s._transport.is_open is False


def test_conflicting_line_settings_do_not_create_a_second_handle(caplog) -> None:
    """A port has ONE line configuration. A second entry asking for another baud
    is a config mistake — it must not become a second exclusive open, and it
    must be loud, because the losing entry's declared baud is silently ignored."""
    import logging

    cfg = _two_channel_cfg()
    cfg["sensors"][1]["baudrate"] = 9600
    with caplog.at_level(logging.WARNING, logger="mast.environment.autodetect"):
        sensors, _ = build_sensors_from_config(cfg)

    assert sensors[0]._transport is sensors[1]._transport
    assert sensors[0]._transport.settings.baudrate == 57600   # first entry wins
    assert any("one port, one configuration" in r.getMessage() for r in caplog.records)


def test_failed_read_keeps_the_shared_transport(monkeypatch) -> None:
    """On a read failure the sensor must close the handle, NOT swap in a fresh
    transport — that would give it a second exclusive open on a port its
    sibling still holds, and that input would never recover."""
    from mast.environment.serial_transport import SerialUnavailable

    sensors, _ = build_sensors_from_config(_two_channel_cfg())
    a, b = sensors
    shared = a._transport
    closed: list[bool] = []

    def _boom(*args, **kw):
        raise SerialUnavailable("port dropped")

    monkeypatch.setattr(shared, "transact", _boom)
    monkeypatch.setattr(shared, "close", lambda: closed.append(True))

    reading = a.read()
    assert reading.status == "unavailable"
    assert closed == [True]                 # handle dropped
    assert a._transport is shared           # object kept
    assert b._transport is shared           # sibling still shares it
