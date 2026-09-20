"""Environment sensor drivers — DL-7 vacuum (Modbus) + Lakeshore (SCPI) + thinking.

Exercises the real protocol code with a fake serial transport (no hardware, no
pyserial port), the alarm/archiving path, the auto-detect/no-hardware degrade,
and the per-agent thinking-strength plumbing.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/environment/test_sensors.py -x -v
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

from mast.core.types import SensorReading
from mast.environment import dl7_vacuum as dl7
from mast.environment import lakeshore_temp as lsci
from mast.environment.alarm import AlarmSpec, worst_status
from mast.environment.serial_transport import SerialUnavailable


# ── Fake transport (implements SerialLike) ───────────────────────────────
class FakeTransport:
    def __init__(self, reply: bytes = b"", *, unavailable: bool = False,
                 replies: list[bytes] | None = None):
        self._reply = reply
        self._replies = list(replies) if replies else None
        self.unavailable = unavailable
        self.sent: list[bytes] = []
        self.closed = False

    def transact(self, payload, *, read_size=None, read_until=None, timeout=None):
        if self.unavailable:
            raise SerialUnavailable("fake: no port")
        self.sent.append(bytes(payload))
        if self._replies is not None:
            return self._replies.pop(0) if self._replies else b""
        return self._reply

    def close(self):
        self.closed = True


# ── DL-7 vacuum gauge (Modbus-RTU) ───────────────────────────────────────
class TestDL7:
    def test_crc16_documented(self):
        # CRC of the documented request body 07 04 00 00 00 02
        assert dl7.crc16(bytes.fromhex("070400000002")) == 0xAD71

    def test_build_request(self):
        req = dl7.build_request(7)
        assert req[:6] == bytes.fromhex("070400000002")
        # CRC low byte first
        assert req[6] == 0x71 and req[7] == 0xAD
        assert len(req) == 8

    def test_parse_documented_example(self):
        # 07 04 04 001C 0002 → A=28, B=2 → 2.8e-2 Pa
        body = bytes.fromhex("070404001C0002")
        frame = body + dl7._crc_suffix(body)
        assert dl7.parse_response(frame, address=7) == pytest.approx(2.8e-2)

    def test_parse_rejects_short(self):
        with pytest.raises(dl7.DL7FrameError):
            dl7.parse_response(b"\x07\x04\x04", address=7)

    def test_parse_rejects_bad_crc(self):
        body = bytes.fromhex("070404001C0002")
        bad = body + b"\x00\x00"
        with pytest.raises(dl7.DL7FrameError):
            dl7.parse_response(bad, address=7)

    def test_parse_rejects_addr_mismatch(self):
        body = bytes.fromhex("050404001C0002")
        frame = body + dl7._crc_suffix(body)
        with pytest.raises(dl7.DL7FrameError):
            dl7.parse_response(frame, address=7)

    def test_sensor_read_ok(self):
        body = bytes.fromhex("070404001C0002")
        frame = body + dl7._crc_suffix(body)
        s = dl7.DL7VacuumSensor(name="vac", transport=FakeTransport(frame))
        r = s.read()
        assert r.status == "ok"
        assert r.value == pytest.approx(2.8e-2)
        assert r.unit == "Pa"

    def test_sensor_unavailable_on_serial_error(self):
        s = dl7.DL7VacuumSensor(name="vac", transport=FakeTransport(unavailable=True))
        r = s.read()
        assert r.status == "unavailable"

    def test_sensor_error_on_bad_frame(self):
        s = dl7.DL7VacuumSensor(name="vac", transport=FakeTransport(b"\x00\x01\x02"))
        r = s.read()
        assert r.status == "error"

    def test_sensor_alarm_threshold(self):
        # pressure 2.8e-2 Pa with a 1e-3 ceiling -> alarm (vacuum degraded)
        body = bytes.fromhex("070404001C0002")
        frame = body + dl7._crc_suffix(body)
        s = dl7.DL7VacuumSensor(name="vac", transport=FakeTransport(frame),
                                alarm=AlarmSpec(max=1e-3, warn_max=1e-4))
        assert s.read().status == "alarm"

    def test_sensor_no_port_unavailable(self):
        # No transport and no port -> unavailable, not crash.
        s = dl7.DL7VacuumSensor(name="vac")
        assert s.read().status == "unavailable"

    def test_probe_true_false(self, monkeypatch):
        body = bytes.fromhex("070404001C0002")
        frame = body + dl7._crc_suffix(body)
        monkeypatch.setattr(dl7, "open_transport", lambda settings: FakeTransport(frame))
        assert dl7.probe_dl7("COM_FAKE") is True
        monkeypatch.setattr(dl7, "open_transport", lambda settings: FakeTransport(b"junk"))
        assert dl7.probe_dl7("COM_FAKE") is False


# ── Lakeshore temperature (SCPI) ─────────────────────────────────────────
class TestLakeshore:
    def test_parse_idn_lsci(self):
        ok, model = lsci.parse_idn(b"LSCI,MODEL336,1234567,1.0\r\n")
        assert ok and model == "MODEL336"

    def test_parse_idn_lakeshore_word(self):
        ok, model = lsci.parse_idn("LAKESHORE,218S,abc,1.6")
        assert ok and model == "218S"

    def test_parse_idn_non_lakeshore(self):
        ok, model = lsci.parse_idn(b"SOMECO,WIDGET,1,1\r\n")
        assert ok is False

    @pytest.mark.parametrize("raw,expect", [
        (b"+273.150E+0\r\n", 273.15),
        (b"295.12\r\n", 295.12),
        (b"  4.2 \r\n", 4.2),
        (b"-1.5E+1\r\n", -15.0),
    ])
    def test_parse_reading(self, raw, expect):
        assert lsci.parse_reading(raw) == pytest.approx(expect)

    def test_parse_reading_bad(self):
        with pytest.raises(lsci.LakeshoreError):
            lsci.parse_reading(b"\r\n")

    def test_default_channels(self):
        assert lsci.default_channels("MODEL336") == ["A", "B", "C", "D"]
        assert lsci.default_channels("218S") == [str(i) for i in range(1, 9)]
        assert lsci.default_channels("335") == ["A", "B"]
        assert lsci.default_channels("") == ["A"]

    def test_sensor_read_ok(self):
        s = lsci.LakeshoreTemperatureSensor(
            name="T", channel="A", transport=FakeTransport(b"+077.350E+0\r\n"))
        r = s.read()
        assert r.status == "ok"
        assert r.value == pytest.approx(77.35)
        assert r.unit == "K"

    def test_sensor_query_uses_channel(self):
        ft = FakeTransport(b"4.2\r\n")
        s = lsci.LakeshoreTemperatureSensor(name="T", channel="B", transport=ft)
        s.read()
        assert ft.sent and ft.sent[0] == b"KRDG? B\r\n"

    def test_sensor_unavailable(self):
        s = lsci.LakeshoreTemperatureSensor(
            name="T", transport=FakeTransport(unavailable=True))
        assert s.read().status == "unavailable"

    def test_sensor_error_on_garbage(self):
        s = lsci.LakeshoreTemperatureSensor(name="T", transport=FakeTransport(b"\r\n"))
        assert s.read().status == "error"

    def test_sensor_alarm_high(self):
        s = lsci.LakeshoreTemperatureSensor(
            name="T", transport=FakeTransport(b"310.0\r\n"),
            alarm=AlarmSpec(max=300.0, warn_max=290.0))
        assert s.read().status == "alarm"

    def test_rejects_injection_channel(self):
        # A crafted channel with CRLF must NOT inject a second SCPI command.
        ft = FakeTransport(b"4.2\r\n")
        s = lsci.LakeshoreTemperatureSensor(
            name="T", channel="A\r\nRANGE 1,3", transport=ft)
        s.read()
        assert ft.sent[0] == b"KRDG? A\r\n"

    def test_rejects_non_whitelisted_command(self):
        # Only KRDG?/CRDG?/SRDG? allowed — a write command is coerced to KRDG?.
        ft = FakeTransport(b"4.2\r\n")
        s = lsci.LakeshoreTemperatureSensor(
            name="T", channel="A", command="MOUT 1,100", transport=ft)
        s.read()
        assert ft.sent[0] == b"KRDG? A\r\n"

    def test_allows_celsius_command(self):
        ft = FakeTransport(b"25.0\r\n")
        s = lsci.LakeshoreTemperatureSensor(
            name="T", channel="A", command="CRDG?", transport=ft)
        s.read()
        assert ft.sent[0] == b"CRDG? A\r\n"

    def test_probe_identifies(self, monkeypatch):
        monkeypatch.setattr(lsci, "open_transport",
                            lambda settings: FakeTransport(b"LSCI,MODEL336,1,1.0\r\n"))
        info = lsci.probe_lakeshore("COM_FAKE")
        assert info is not None and info.model == "MODEL336"

    def test_probe_none_when_not_lakeshore(self, monkeypatch):
        monkeypatch.setattr(lsci, "open_transport",
                            lambda settings: FakeTransport(b"OTHER,X,1,1\r\n"))
        assert lsci.probe_lakeshore("COM_FAKE") is None


# ── Alarm ────────────────────────────────────────────────────────────────
class TestAlarm:
    def test_ceiling(self):
        a = AlarmSpec(max=1e-6, warn_max=1e-7)
        assert a.evaluate(1e-5) == "alarm"
        assert a.evaluate(5e-7) == "warning"
        assert a.evaluate(1e-8) == "ok"

    def test_band(self):
        a = AlarmSpec(max=300, min=4, warn_max=290, warn_min=10)
        assert a.evaluate(310) == "alarm"
        assert a.evaluate(3) == "alarm"
        assert a.evaluate(295) == "warning"
        assert a.evaluate(5) == "warning"
        assert a.evaluate(100) == "ok"

    def test_preserves_error(self):
        a = AlarmSpec(max=1.0)
        assert a.evaluate(99, base_status="error") == "error"
        assert a.evaluate(99, base_status="unavailable") == "unavailable"

    def test_empty(self):
        assert AlarmSpec().is_empty()
        assert AlarmSpec().evaluate(123.0) == "ok"

    def test_roundtrip_dict(self):
        d = {"max": 1e-6, "warn_max": 1e-7}
        assert AlarmSpec.from_dict(d).to_dict() == d

    def test_worst_status(self):
        assert worst_status("ok", "warning", "alarm") == "alarm"
        assert worst_status("ok", "unavailable") == "unavailable"
        assert worst_status("ok", "ok") == "ok"

    def test_worst_status_full_order(self):
        # review 2.1.13 #30: pin the full severity ordering
        assert worst_status("error", "alarm") == "alarm"        # alarm worst
        assert worst_status("warning", "unavailable") == "warning"
        assert worst_status("error", "warning") == "error"
        assert worst_status("unavailable", "ok") == "unavailable"


# ── Config + autodetect (no hardware) ────────────────────────────────────
class TestConfigAutodetect:
    def test_config_roundtrip(self, tmp_path):
        from mast.environment import config as c
        p = tmp_path / "env.json"
        c.update_sensor("vac_main", path=p, name="主腔真空", type="dl7_vacuum",
                        port="COM19", alarm={"max": 1e-6})
        cfg = c.load_config(p)
        assert cfg["sensors"][0]["name"] == "主腔真空"
        assert cfg["sensors"][0]["alarm"]["max"] == 1e-6
        assert c.remove_sensor("vac_main", path=p) is True
        assert c.load_config(p)["sensors"] == []

    def test_load_missing_is_default(self, tmp_path):
        from mast.environment import config as c
        cfg = c.load_config(tmp_path / "nope.json")
        assert cfg == {"autodetect": True, "sensors": []}

    def test_load_corrupt_is_default(self, tmp_path):
        from mast.environment import config as c
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        assert c.load_config(p) == {"autodetect": True, "sensors": []}

    def test_build_from_config(self, tmp_path):
        from mast.environment import config as c
        from mast.environment.autodetect import build_sensors_from_config
        cfg = {"autodetect": False, "sensors": [
            {"id": "v", "name": "真空", "type": "dl7_vacuum", "port": "COM13"},
            {"id": "t", "name": "温度", "type": "lakeshore_temp", "port": "COM14", "channel": "A"},
        ]}
        sensors, used = build_sensors_from_config(cfg)
        assert {s.name() for s in sensors} == {"真空", "温度"}
        assert used == {"COM13", "COM14"}

    def test_build_skips_unknown_type(self):
        from mast.environment.autodetect import build_sensors_from_config
        sensors, _ = build_sensors_from_config({"sensors": [{"id": "x", "type": "bogus"}]})
        assert sensors == []

    def test_build_environment_sensors_no_hardware(self, monkeypatch):
        # Force autodetect off + empty config -> placeholders only, never raises.
        from mast.environment import autodetect as ad
        monkeypatch.setattr(ad, "load_config", lambda: {"autodetect": False, "sensors": []})
        sensors = ad.build_environment_sensors()
        names = {s.name() for s in sensors}
        assert {"vacuum", "temperature", "helium_level", "noise_level"} <= names
        for s in sensors:
            assert s.read().status == "unavailable"

    def test_autodetect_no_ports(self, monkeypatch):
        from mast.environment import autodetect as ad
        monkeypatch.setattr(ad, "list_serial_ports", lambda: [])
        assert ad.autodetect_sensors() == []

    def test_autodetect_positive(self, monkeypatch):
        # review 2.1.13 #18: positive multi-device detection + DL-7-then-Lakeshore
        # probe order + skip_ports.
        from mast.environment import autodetect as ad
        from mast.environment.lakeshore_temp import LakeshoreInfo
        from mast.environment.serial_transport import SerialPortInfo, SerialSettings
        ports = [SerialPortInfo(device="COM13"), SerialPortInfo(device="COM14"),
                 SerialPortInfo(device="COM15")]
        monkeypatch.setattr(ad, "list_serial_ports", lambda: ports)
        # COM13 = DL-7; COM14 = Lakeshore; COM15 = nothing
        monkeypatch.setattr(ad, "probe_dl7", lambda dev: dev == "COM13")
        def _probe_lsci(dev):
            if dev == "COM14":
                return LakeshoreInfo("MODEL336", "LSCI,MODEL336,1,1",
                                     SerialSettings(port=dev, baudrate=57600,
                                                    bytesize=7, parity="O"))
            return None
        monkeypatch.setattr(ad, "probe_lakeshore", _probe_lsci)
        found = ad.autodetect_sensors(skip_ports={"COM15"})
        names = [s.name() for s in found]
        assert any("COM13" in n for n in names)   # DL-7 detected
        assert any("COM14" in n for n in names)   # Lakeshore detected
        assert len(found) == 2                    # COM15 skipped, nothing on it


# ── EnvironmentMonitor (alarm transitions + archiving) ───────────────────
class TestMonitor:
    def _make_sensor(self, values, alarm):
        class _S:
            def __init__(s):
                s.vals = list(values)
                s.i = 0
            def name(s):
                return "vac"
            def read(s):
                v = s.vals[min(s.i, len(s.vals) - 1)]
                s.i += 1
                return SensorReading(value=v, unit="Pa", status=alarm.evaluate(v))
        return _S()

    def test_alarm_fires_once_per_transition(self):
        from mast.environment.monitor import EnvironmentMonitor
        a = AlarmSpec(max=1e-6)
        fired = []
        m = EnvironmentMonitor([self._make_sensor([1e-8, 1e-5, 1e-5, 1e-8], a)],
                               on_alarm=lambda n, r, p: fired.append((n, r.status, p)))
        for _ in range(4):
            m.read_all()
        assert fired == [("vac", "alarm", "ok")]   # only the ok->alarm transition

    def test_alarms_and_overall(self):
        from mast.environment.monitor import EnvironmentMonitor
        a = AlarmSpec(max=1e-6)
        m = EnvironmentMonitor([self._make_sensor([1e-5], a)])
        m.read_all()
        assert "vac" in m.alarms()
        assert m.overall_status() == "alarm"

    def test_archiving_to_storage(self, tmp_path):
        from mast.environment.monitor import EnvironmentMonitor
        from mast.logging.storage import ExperimentStorage
        storage = ExperimentStorage(str(tmp_path / "exp.db"))
        a = AlarmSpec()
        m = EnvironmentMonitor([self._make_sensor([1.23], a)], storage=storage)
        readings = m.read_all()
        # archive like the loop does
        for nm, r in readings.items():
            storage.log_environment(sensor_name=nm, value=r.value, unit=r.unit, status=r.status)
        hist = storage.get_environment_history("vac")
        assert len(hist) >= 1

    def test_replace_sensors(self):
        from mast.environment.monitor import EnvironmentMonitor
        a = AlarmSpec()
        m = EnvironmentMonitor([self._make_sensor([1.0], a)])
        m.read_all()
        m.replace_sensors([self._make_sensor([2.0], a)])
        assert m.sensor_names() == ["vac"]
        assert m.get_latest() == {}  # cache cleared on replace

    def _closable(self, name="s"):
        class _C:
            def __init__(s): s.closed = False
            def name(s): return name
            def read(s): return SensorReading(value=1.0, unit="Pa", status="ok")
            def close(s): s.closed = True
        return _C()

    def test_replace_closes_displaced_handle(self):
        # review 2.1.13 #1/#8: a displaced sensor's serial handle must be closed.
        from mast.environment.monitor import EnvironmentMonitor
        old = self._closable("vac")
        m = EnvironmentMonitor([old])
        new = self._closable("vac")
        m.replace_sensors([new])
        assert old.closed is True   # freed
        assert new.closed is False  # kept

    def test_stop_closes_all_handles(self):
        from mast.environment.monitor import EnvironmentMonitor
        s = self._closable("vac")
        m = EnvironmentMonitor([s])
        m.stop()
        assert s.closed is True

    def test_alarm_rearm_and_escalation(self):
        # review 2.1.13 #16: warning→alarm escalation and alarm→ok→alarm re-arm
        # must each fire (not suppressed as 'same alert state').
        from mast.environment.monitor import EnvironmentMonitor
        a = AlarmSpec(max=1e-6, warn_max=1e-7)
        # 1e-8 ok → 5e-7 warning → 1e-5 alarm → 1e-8 ok → 1e-5 alarm
        seq = [1e-8, 5e-7, 1e-5, 1e-8, 1e-5]
        fired = []
        m = EnvironmentMonitor([self._make_sensor(seq, a)],
                               on_alarm=lambda n, r, p: fired.append((r.status, p)))
        for _ in seq:
            m.read_all()
        assert fired == [("warning", "ok"), ("alarm", "warning"),
                         ("alarm", "ok")]

    def test_loop_archives_real(self, tmp_path):
        # review 2.1.13 #17: exercise the real background _loop → storage path.
        import time as _t
        from mast.environment.monitor import EnvironmentMonitor
        from mast.logging.storage import ExperimentStorage
        storage = ExperimentStorage(str(tmp_path / "exp.db"))
        m = EnvironmentMonitor([self._make_sensor([1.23], AlarmSpec())],
                               storage=storage, interval_s=0.05)
        m.start()
        try:
            deadline = _t.time() + 3.0
            while _t.time() < deadline and not storage.get_environment_history("vac"):
                _t.sleep(0.1)
        finally:
            m.stop()
        assert storage.get_environment_history("vac")  # the real loop archived


# ── Thinking strength plumbing ───────────────────────────────────────────
class TestThinking:
    def test_normalize(self):
        from mast.agents._shared.models import normalize_thinking_level
        assert normalize_thinking_level("HIGH") == "high"
        assert normalize_thinking_level("bogus") is None
        assert normalize_thinking_level(None) is None

    def test_effective(self):
        from mast.agents._shared.models import effective_thinking, KIMI_K2_6, SONNET_4_6
        assert "固定" in effective_thinking(KIMI_K2_6, "low")   # reasoning model pinned high
        assert effective_thinking(SONNET_4_6, "high") == "high"
        assert effective_thinking(SONNET_4_6, None) == "off"

    def test_anthropic_thinking_budget(self, monkeypatch):
        from mast.agents._shared import models as M
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
        m = M.make_chat_model(model_id=M.SONNET_4_6, max_tokens=2048,
                              temperature=0.1, thinking_level="high",
                              allow_fallback=False)
        # Sonnet 4.6 uses ADAPTIVE thinking (manual budget_tokens is rejected on
        # Opus 4.7/4.8; mirrored across the adaptive set). No budget_tokens — the
        # level rides in output_config.effort and max_tokens is floored to 16000.
        assert getattr(m, "thinking", None) == {"type": "adaptive"}
        assert m.max_tokens >= 16000       # floored to give thinking+text room
        assert m.temperature == 1.0        # forced by Anthropic when thinking on

    def test_anthropic_thinking_off(self, monkeypatch):
        from mast.agents._shared import models as M
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
        m = M.make_chat_model(model_id=M.SONNET_4_6, thinking_level="off",
                              allow_fallback=False)
        assert getattr(m, "thinking", None) in (None, {})

    def test_anthropic_thinking_budgets(self, monkeypatch):
        # review 2.1.13 #31: pin low/medium budgets + the max_tokens bump rule.
        # Uses Haiku 4.5 — a non-adaptive Claude that still takes manual
        # budget_tokens (the adaptive set is covered in test_anthropic_thinking_budget).
        from mast.agents._shared import models as M
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
        for level, budget in (("low", 2048), ("medium", 8192)):
            m = M.make_chat_model(model_id=M.HAIKU_4_5, max_tokens=1024,
                                  thinking_level=level, allow_fallback=False)
            assert m.thinking == {"type": "enabled", "budget_tokens": budget}
            assert m.max_tokens >= budget + 1     # bumped over budget
            assert m.temperature == 1.0

    def test_reasoning_model_no_anthropic_thinking(self, monkeypatch):
        # review 2.1.13 #14: a reasoning model (Kimi) must NOT get an Anthropic
        # extended-thinking dict (the honesty 'no fabricated knob' guarantee);
        # passing a level is a safe no-op, not a crash.
        from mast.agents._shared import models as M
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-dummy")
        m = M.make_chat_model(model_id=M.KIMI_K2_6, thinking_level="low",
                              allow_fallback=False)
        assert getattr(m, "thinking", None) in (None, {})   # no Anthropic knob


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
