"""Phase-3 contract tests for Domain E — Settings write + admin overrides +
safety PIN + environment sensors + Nanonis hardware.

The router is NOT yet mounted in mast.api.app (integration wires that), so
each test builds a throwaway FastAPI app and includes the router under /api.

The guarantees asserted here:
  * every endpoint returns its declared status (200 / 202) and a body matching
    its response_model;
  * with no live core wired (standalone AppContext) every subsystem-backed
    endpoint DEGRADES — empty-but-valid, ``degraded: true``, never 500;
  * write endpoints that hit a pure-stdlib core (SettingsStore against an
    isolated tmp config dir) actually persist whitelisted keys + drop the rest.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.admin import router


def _client(user_root: str | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext(user_root=user_root)
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture()
def client() -> TestClient:
    return _client()


# NOTE: POST /api/settings tests moved to test_settings_admin_write.py (the real
# unified write now lives in routes/settings_admin_write.py, not admin.py).


# ── Admin overrides ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "category",
    ["safety_limits", "checks", "constraints", "skill", "knowledge",
     "guidance", "encyclopedia", "agent"],
)
def test_override_get_degrades_unwired(client: TestClient, category: str) -> None:
    r = client.get(f"/api/admin/overrides/{category}")
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == category
    assert body["degraded"] is True
    assert body["data"] == {}
    assert body["has_override"] is False


def test_override_unknown_category_degrades(client: TestClient) -> None:
    r = client.get("/api/admin/overrides/not_a_category")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_override_write_degrades_unwired(client: TestClient) -> None:
    r = client.post("/api/admin/overrides/safety_limits", json={"data": {"bias_max_v": 5.0}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True
    assert body["category"] == "safety_limits"
    assert body["reloaded"] is False


def test_override_history_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/admin/overrides/checks/history")
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "checks"
    assert body["degraded"] is True
    assert body["entries"] == [] and body["count"] == 0


def test_override_restore_degrades_unwired(client: TestClient) -> None:
    r = client.post("/api/admin/overrides/safety_limits/restore/20260101T000000")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True


def test_override_write_with_live_registry(tmp_path) -> None:
    """When a registry IS wired (here a real one pointed at a tmp dir), the write
    persists + hot-reloads and the read reflects it. Proves the passthrough."""
    from mast.admin.override_store import ConfigOverrideRegistry

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path / "overrides")
    app = FastAPI()
    ctx = AppContext()
    ctx.override_registry = reg  # leader-style wiring (attribute set on ctx)
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    c = TestClient(app)

    w = c.post("/api/admin/overrides/safety_limits", json={"data": {"bias_max_v": 5.0}})
    assert w.status_code == 200
    wb = w.json()
    assert wb["ok"] is True and wb["degraded"] is False
    assert wb["data"] == {"bias_max_v": 5.0}
    # `reloaded` reports whether a hot-reload SUBSCRIBER actually ran. Nothing is
    # subscribed here (nor anywhere in production, 2026-08-03), so it is False and
    # the operator is told a restart is needed. It used to be a hardcoded True —
    # which this very assertion happily accepted for as long as it was a literal.
    assert wb["reloaded"] is False
    assert wb["restart_required"] is True

    g = c.get("/api/admin/overrides/safety_limits")
    gb = g.json()
    assert gb["has_override"] is True
    assert gb["data"] == {"bias_max_v": 5.0}

    # empty payload ⇒ reset to defaults (file deleted) → read is empty again
    d = c.post("/api/admin/overrides/safety_limits", json={"data": {}})
    assert d.json()["ok"] is True
    assert c.get("/api/admin/overrides/safety_limits").json()["has_override"] is False


def test_reloaded_flips_true_when_a_hook_is_actually_subscribed(tmp_path) -> None:
    """`reloaded` must track a real subscriber, not the constant True it used to be.

    The whole point of the field is to distinguish "persisted" from "in effect".
    A test that only ever sees one value cannot tell a working report from a
    hardcoded one — so this asserts the OTHER value, with a hook subscribed.
    """
    from mast.admin.override_store import ConfigOverrideRegistry

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path / "overrides")
    seen: list[int] = []
    reg.register_reload_hook(lambda: seen.append(1))

    app = FastAPI()
    ctx = AppContext()
    ctx.override_registry = reg
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    c = TestClient(app)

    wb = c.post("/api/admin/overrides/safety_limits",
                json={"data": {"bias_max_v": 5.0}}).json()
    assert seen == [1]                      # the hook really ran
    assert wb["reloaded"] is True           # …and the response says so
    # No live SafetyGuard is wired here, so "did the process pick it up" is
    # genuinely unknown — and unknown must not be reported as False.
    assert wb["restart_required"] in (False, None)


def test_a_raising_hook_is_not_counted_as_a_reload(tmp_path) -> None:
    """A hook that blew up did not re-derive anything; it must not read as success."""
    from mast.admin.override_store import ConfigOverrideRegistry

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path / "overrides")

    def _boom() -> None:
        raise RuntimeError("derived cache rebuild failed")

    reg.register_reload_hook(_boom)

    app = FastAPI()
    ctx = AppContext()
    ctx.override_registry = reg
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    c = TestClient(app)

    wb = c.post("/api/admin/overrides/safety_limits",
                json={"data": {"bias_max_v": 5.0}}).json()
    assert wb["ok"] is True                 # the write itself still succeeded
    assert wb["reloaded"] is False          # but nothing re-derived
    assert wb["data"] == {"bias_max_v": 5.0}


# ── Admin PIN unlock ────────────────────────────────────────────────────────
def test_pin_unlock_no_pin_set(tmp_path, monkeypatch) -> None:
    # Isolate project_root to an empty tmp dir (the real repo may ship a
    # config/admin_pin.txt). With no PIN file present the gate stays locked with
    # reason 'no_pin_set' — a valid typed body, never a 500.
    import mast._runtime_paths as rp

    monkeypatch.setattr(rp, "project_root", lambda: tmp_path)
    c = _client()
    r = c.post("/api/admin/unlock-pin", json={"pin": "1234"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["reason"] == "no_pin_set"
    assert body["token"] is None
    assert body["degraded"] is False


def test_pin_unlock_match_mints_token(tmp_path, monkeypatch) -> None:
    """With a launcher-style admin_pin.txt present, a correct PIN mints a token."""
    import hashlib

    import mast._runtime_paths as rp

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    pin = "secret42"
    (cfg_dir / "admin_pin.txt").write_text(
        hashlib.sha256(pin.encode()).hexdigest(), encoding="utf-8"
    )
    monkeypatch.setattr(rp, "project_root", lambda: tmp_path)

    c = _client()
    ok = c.post("/api/admin/unlock-pin", json={"pin": pin})
    okb = ok.json()
    assert okb["ok"] is True
    assert okb["token"] and isinstance(okb["token"], str)
    assert okb["degraded"] is False

    bad = c.post("/api/admin/unlock-pin", json={"pin": "wrong"})
    assert bad.json()["ok"] is False
    assert bad.json()["reason"] == "wrong"

    empty = c.post("/api/admin/unlock-pin", json={"pin": ""})
    assert empty.json()["reason"] == "empty"


# ── Environment sensors ─────────────────────────────────────────────────────
def test_sensors_get_shape(tmp_path, monkeypatch) -> None:
    # Point the env-config at an isolated tmp file so the read is deterministic
    # and never touches the real config dir.
    import mast.environment.config as envcfg

    monkeypatch.setattr(envcfg, "config_path", lambda: tmp_path / "environment_sensors.json")
    c = _client()
    r = c.get("/api/environment/sensors")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["autodetect"] is True
    assert body["sensors"] == [] and body["count"] == 0


def test_sensors_add_update_delete_roundtrip(tmp_path, monkeypatch) -> None:
    import mast.environment.config as envcfg

    path = tmp_path / "environment_sensors.json"
    monkeypatch.setattr(envcfg, "config_path", lambda: path)
    c = _client()

    add = c.post(
        "/api/environment/sensors",
        json={
            "id": "vac_main",
            "name": "主腔真空",
            "type": "dl7_vacuum",
            "port": "COM15",
            "unit": "Pa",
            "extra": {"address": 7, "alarm": {"max": 1e-6}},
        },
    )
    assert add.status_code == 200
    ab = add.json()
    assert ab["ok"] is True and ab["degraded"] is False
    assert ab["sensor"]["id"] == "vac_main"
    assert ab["sensor"]["name"] == "主腔真空"
    # type-specific fields preserved in extra
    assert ab["sensor"]["extra"]["address"] == 7

    # reflected in the listing
    lst = c.get("/api/environment/sensors").json()
    assert lst["count"] == 1 and lst["sensors"][0]["id"] == "vac_main"

    # delete it
    d = c.delete("/api/environment/sensors/vac_main")
    assert d.status_code == 200
    db = d.json()
    assert db["ok"] is True and db["removed"] is True

    assert c.get("/api/environment/sensors").json()["count"] == 0


def test_sensor_delete_missing(tmp_path, monkeypatch) -> None:
    import mast.environment.config as envcfg

    monkeypatch.setattr(envcfg, "config_path", lambda: tmp_path / "environment_sensors.json")
    c = _client()
    d = c.delete("/api/environment/sensors/nope")
    assert d.status_code == 200
    body = d.json()
    assert body["ok"] is True
    assert body["removed"] is False


def test_sensors_rescan_degrades_unwired(client: TestClient, monkeypatch, tmp_path) -> None:
    import mast.environment.config as envcfg

    monkeypatch.setattr(envcfg, "config_path", lambda: tmp_path / "environment_sensors.json")
    r = client.get("/api/environment/sensors/rescan")
    assert r.status_code == 200
    body = r.json()
    # no live monitor wired → degraded, but a valid (empty) sensor list
    assert body["degraded"] is True
    assert body["ok"] is False
    assert body["sensors"] == []


# ── Rescan against a LIVE monitor ───────────────────────────────────────────
# A rescan must stop the monitor, rebuild and replace sensors, then restart it.


class _FakeMonitor:
    """Records the call ORDER, which is what both regressions are about."""

    def __init__(self, running: bool = False):
        self._running = running
        self.calls: list[str] = []
        self.sensors: list = []

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True
        self.calls.append("start")

    def stop(self) -> None:
        self._running = False
        self.calls.append("stop")

    def replace_sensors(self, sensors) -> None:
        self.sensors = list(sensors)
        self.calls.append("replace")

    def sensor_names(self) -> list[str]:
        return [s.name() for s in self.sensors]


def _client_with_monitor(monitor) -> TestClient:
    app = FastAPI()
    ctx = AppContext()
    ctx.environment_monitor = monitor
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _patch_build(monkeypatch, monitor, sensors, *, raises: bool = False):
    """Patch the sensor rebuild, recording when it runs relative to stop/start."""
    import mast.environment.autodetect as ad

    def _fake_build(*a, **kw):
        monitor.calls.append("build")
        if raises:
            raise RuntimeError("rebuild boom")
        return list(sensors)

    monkeypatch.setattr(ad, "build_environment_sensors", _fake_build)


def _real_lakeshore():
    from mast.environment.lakeshore_temp import LakeshoreTemperatureSensor

    # port=None → no settings, so constructing it never touches a serial port.
    return LakeshoreTemperatureSensor(name="Lake Shore 335 A")


def _placeholder():
    from mast.environment.placeholders import TemperatureSensor

    return TemperatureSensor()


def test_sensors_rescan_frees_bus_before_probing(monkeypatch) -> None:
    """stop() must precede the rebuild.

    COM ports are exclusive on Windows: if the live sensor still holds the port
    when autodetect probes it, the probe fails, the rescan 'finds' nothing, and
    a WORKING gauge gets replaced by a placeholder.
    """
    mon = _FakeMonitor(running=True)
    _patch_build(monkeypatch, mon, [_real_lakeshore()])
    r = _client_with_monitor(mon).get("/api/environment/sensors/rescan")

    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert mon.calls.index("stop") < mon.calls.index("build")


def test_sensors_rescan_starts_loop_when_hardware_found(monkeypatch) -> None:
    """Booted with no instrument (loop idle) → rescan finds one → loop runs.

    Without the start() the panel would show live values while nothing was ever
    archived and no over-limit could escalate to E_STOP.
    """
    mon = _FakeMonitor(running=False)
    _patch_build(monkeypatch, mon, [_real_lakeshore()])
    r = _client_with_monitor(mon).get("/api/environment/sensors/rescan")

    assert r.json()["ok"] is True
    assert mon.is_running is True
    assert mon.calls == ["stop", "build", "replace", "start"]


def test_sensors_rescan_leaves_loop_idle_for_placeholders(monkeypatch) -> None:
    """No real hardware → the loop stays idle (no unavailable-row spam)."""
    mon = _FakeMonitor(running=False)
    _patch_build(monkeypatch, mon, [_placeholder()])
    r = _client_with_monitor(mon).get("/api/environment/sensors/rescan")

    assert r.json()["ok"] is True
    assert mon.is_running is False
    assert "start" not in mon.calls


def test_sensors_rescan_restores_loop_if_rebuild_fails(monkeypatch) -> None:
    """A failed rebuild must not leave monitoring silently dead."""
    mon = _FakeMonitor(running=True)
    _patch_build(monkeypatch, mon, [], raises=True)
    r = _client_with_monitor(mon).get("/api/environment/sensors/rescan")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert mon.is_running is True          # loop put back the way we found it
    assert "replace" not in mon.calls      # sensor set left untouched


# ── 扫描设备接口: discover → confirm → adopt ─────────────────────────────────
_FAKE_FOUND = [
    {
        "port": "COM13", "description": "Lake Shore Model 335", "hwid": "USB VID:PID=1FB9:0300",
        "kind": "lakeshore_temp", "identified": True, "already_configured": False,
        "model": "MODEL335", "idn": "LSCI,MODEL335,LSA2SHB/#######,2.1",
        "firmware": "2.1", "serial_number": "LSA2SHB/#######", "any_heater_on": False,
        "channels": [{"channel": "A", "label": "SPM", "kelvin": 77.402},
                     {"channel": "B", "label": "Magnet", "kelvin": 77.275}],
        "outputs": [{"output": 1, "range_code": 0, "range_label": "关闭", "heater_on": False}],
        "suggested_sensors": [
            {"id": "lakeshore_com3_a", "name": "SPM", "type": "lakeshore_temp",
             "port": "COM13", "channel": "A", "unit": "K", "baudrate": 57600,
             "bytesize": 7, "parity": "O", "stopbits": 1},
            {"id": "lakeshore_com3_b", "name": "Magnet", "type": "lakeshore_temp",
             "port": "COM13", "channel": "B", "unit": "K", "baudrate": 57600,
             "bytesize": 7, "parity": "O", "stopbits": 1},
        ],
    },
    {"port": "COM1", "description": "通信端口", "kind": "", "identified": False,
     "already_configured": False, "suggested_sensors": []},
]


def test_discover_reports_identified_and_unidentified_ports(monkeypatch) -> None:
    import mast.environment.autodetect as ad

    mon = _FakeMonitor(running=True)
    monkeypatch.setattr(ad, "discover_devices", lambda *a, **k: [dict(d) for d in _FAKE_FOUND])
    r = _client_with_monitor(mon).get("/api/environment/discover")

    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["identified_count"] == 1
    ls = body["devices"][0]
    assert ls["model"] == "MODEL335" and ls["identified"] is True
    assert [c["label"] for c in ls["channels"]] == ["SPM", "Magnet"]
    assert ls["any_heater_on"] is False
    # The port that answered nothing is still reported, so the dialog can say
    # what it looked at rather than silently omitting it.
    assert body["devices"][1]["port"] == "COM1"
    assert body["devices"][1]["identified"] is False


def test_discover_frees_the_bus_and_puts_it_back(monkeypatch) -> None:
    import mast.environment.autodetect as ad

    mon = _FakeMonitor(running=True)

    def _probe(*a, **k):
        mon.calls.append("probe")
        return []

    monkeypatch.setattr(ad, "discover_devices", _probe)
    _client_with_monitor(mon).get("/api/environment/discover")

    assert mon.calls == ["stop", "probe", "start"]
    assert mon.is_running is True     # a scan must not leave monitoring off


def test_discover_restores_the_loop_even_when_the_scan_throws(monkeypatch) -> None:
    import mast.environment.autodetect as ad

    mon = _FakeMonitor(running=True)

    def _boom(*a, **k):
        raise RuntimeError("port enumeration exploded")

    monkeypatch.setattr(ad, "discover_devices", _boom)
    r = _client_with_monitor(mon).get("/api/environment/discover")

    assert r.status_code == 200          # never 500
    assert r.json()["devices"] == []
    assert mon.is_running is True


def test_discover_degrades_without_a_live_monitor(client: TestClient) -> None:
    """A standalone API process must not open ports the live app owns."""
    r = client.get("/api/environment/discover")
    assert r.status_code == 200
    assert r.json()["degraded"] is True
    assert r.json()["devices"] == []


def test_adopt_persists_and_survives_a_restart(monkeypatch, tmp_path) -> None:
    """Adoption is what makes the port stick: the entries must come back from
    the persisted config on the next boot, with no rescan."""
    import mast.environment.autodetect as ad
    import mast.environment.config as envcfg

    cfg = tmp_path / "environment_sensors.json"
    monkeypatch.setattr(envcfg, "config_path", lambda: cfg)
    mon = _FakeMonitor(running=False)
    monkeypatch.setattr(ad, "build_environment_sensors", lambda *a, **k: [_real_lakeshore()])

    r = _client_with_monitor(mon).post(
        "/api/environment/sensors/adopt",
        json={"sensors": _FAKE_FOUND[0]["suggested_sensors"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["adopted"] == 2 and body["rejected"] == []

    # Persisted, with the instrument's own labels and the pinned line settings.
    saved = envcfg.load_config()["sensors"]
    assert [s["name"] for s in saved] == ["SPM", "Magnet"]
    assert {s["port"] for s in saved} == {"COM13"}
    assert [s["channel"] for s in saved] == ["A", "B"]
    assert saved[0]["baudrate"] == 57600 and saved[0]["parity"] == "O"

    # Real hardware is now configured → the archive loop was started.
    assert mon.is_running is True


def test_adopt_rejects_junk_without_persisting_it(monkeypatch, tmp_path) -> None:
    import mast.environment.config as envcfg

    cfg = tmp_path / "environment_sensors.json"
    monkeypatch.setattr(envcfg, "config_path", lambda: cfg)

    r = _client_with_monitor(_FakeMonitor()).post(
        "/api/environment/sensors/adopt",
        json={"sensors": [
            {"id": "", "type": "lakeshore_temp"},                 # no id
            {"id": "x", "type": "definitely_not_a_sensor"},       # unknown type
        ]},
    )
    body = r.json()
    assert body["ok"] is False and body["adopted"] == 0
    assert len(body["rejected"]) == 2
    assert envcfg.load_config()["sensors"] == []


def test_instrument_state_reads_over_the_live_port(monkeypatch) -> None:
    """The settings readout must reuse the monitor's open handle, and must
    report a two-input instrument ONCE rather than once per input."""
    from mast.environment.lakeshore_temp import LakeshoreTemperatureSensor
    from tests.v2.environment.test_lakeshore_instrument_state import RecordedTransport

    shared = RecordedTransport()
    a = LakeshoreTemperatureSensor(name="SPM", port="COM13", channel="A", transport=shared)
    b = LakeshoreTemperatureSensor(name="Magnet", port="COM13", channel="B", transport=shared)
    mon = _FakeMonitor(running=True)
    mon._sensors = {"SPM": a, "Magnet": b}

    r = _client_with_monitor(mon).get("/api/environment/instrument-state")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert len(body["instruments"]) == 1          # one cable, one instrument
    inst = body["instruments"][0]
    assert inst["model"] == "MODEL335" and inst["port"] == "COM13"
    assert [c["label"] for c in inst["channels"]] == ["SPM", "Magnet"]
    assert inst["any_heater_on"] is False
    assert inst["outputs"][0]["range_label"] == "关闭"
    # Reading state must NOT have taken the bus away from the monitor.
    assert "stop" not in mon.calls


def test_instrument_state_carries_the_unknown_status_all_the_way_out() -> None:
    """「问过了，干净」和「根本没问出来」必须在 HTTP 响应里也分得开。

    两件事一起钉：
      * ``faults`` 是 ``null`` 而不是 ``[]`` —— 而且端点没有 500。schema 以前写的
        是 ``list[str]``（非空默认 ``[]``），真让后端回一个 ``None`` 就是
        ValidationError，而这行构造不在 try 里。
      * ``health`` 直接给出判词，前端不必自己拼 —— 它自己拼的那次写的是
        ``(c.faults ?? []).length > 0``，把 null 读成了「正常」。
    """
    from mast.environment.lakeshore_temp import LakeshoreTemperatureSensor
    from tests.v2.environment.test_lakeshore_instrument_state import RecordedTransport

    class _SilentStatusOnA(RecordedTransport):
        """A answers KRDG? but its RDGST? times out (read_until → b"")."""

        def transact(self, payload: bytes, **kw) -> bytes:
            if payload.decode("ascii").strip() == "RDGST? A":
                return b"\r\n"
            return super().transact(payload, **kw)

    shared = _SilentStatusOnA()
    mon = _FakeMonitor(running=True)
    mon._sensors = {
        "SPM": LakeshoreTemperatureSensor(name="SPM", port="COM13", channel="A",
                                          transport=shared),
    }

    r = _client_with_monitor(mon).get("/api/environment/instrument-state")
    assert r.status_code == 200, r.text
    chans = {c["channel"]: c for c in r.json()["instruments"][0]["channels"]}

    assert chans["A"]["faults"] is None, "超时被折叠成了「干净」"
    assert chans["A"]["health"] == "unknown"
    # 反向对照：同一次响应里，答了 "000" 的那个通道仍然是「问过，干净」。
    assert chans["B"]["faults"] == []
    assert chans["B"]["health"] == "clean"
    # 而读数本身照常回报 —— 三态不是把会答话的那条路径一起拒掉。
    assert chans["A"]["kelvin"] == pytest.approx(77.402)


def test_instrument_state_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/environment/instrument-state")
    assert r.status_code == 200
    assert r.json()["degraded"] is True
    assert r.json()["instruments"] == []


def test_has_real_sensors_agrees_with_runtime_gate() -> None:
    """The boot gate and the rescan gate must be the same predicate."""
    from mast.core.runtime import CoreRuntime
    from mast.environment.autodetect import has_real_sensors

    for sensors in ([], [_placeholder()], [_real_lakeshore()],
                    [_placeholder(), _real_lakeshore()]):
        assert has_real_sensors(sensors) is CoreRuntime._has_real_env_sensors(sensors)


# ── Nanonis hardware ────────────────────────────────────────────────────────
def test_nanonis_connect_accepted_degraded(client: TestClient) -> None:
    r = client.post("/api/nanonis/connect", json={})
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] is True
    assert body["status"] == "unavailable"
    assert body["degraded"] is True


def test_nanonis_connection_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/nanonis/connection")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["connected"] is False
    roles = {p["role"] for p in body["ports"]}
    assert roles == {"main", "monitor", "data", "emergency"}
    assert all(p["connected"] is False for p in body["ports"])


def test_nanonis_connection_prefills_default_ports_unwired(client: TestClient) -> None:
    """With no live config wired, the snapshot still pre-fills the canonical
    6501-6504 ports from mast.config defaults (UI pre-fill never null)."""
    r = client.get("/api/nanonis/connection")
    assert r.status_code == 200
    by_role = {p["role"]: p["port"] for p in r.json()["ports"]}
    assert by_role == {
        "main": 6501,
        "monitor": 6502,
        "data": 6503,
        "emergency": 6504,
    }


def test_nanonis_connection_prefills_from_live_config() -> None:
    """When a live config.nanonis is wired onto the ctx, its configured port
    values are echoed back per role (connected/degraded unaffected: no pool)."""
    from mast.config import NanonisConfig

    class _Ctx:
        class config:  # noqa: N801 - mimics live AppContext.config.nanonis
            nanonis = NanonisConfig(
                host="10.0.0.5",
                port_main=7001,
                port_monitor=7002,
                port_data=7003,
                port_emergency=7004,
            )

    app = FastAPI()
    app.state.ctx = _Ctx()
    app.include_router(router, prefix="/api")
    r = TestClient(app).get("/api/nanonis/connection")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True  # still no connection pool wired
    assert body["host"] == "10.0.0.5"
    by_role = {p["role"]: p["port"] for p in body["ports"]}
    assert by_role == {
        "main": 7001,
        "monitor": 7002,
        "data": 7003,
        "emergency": 7004,
    }


# ── System self-check ───────────────────────────────────────────────────────
def test_system_check_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/system/check")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["items"] == [] and body["count"] == 0
