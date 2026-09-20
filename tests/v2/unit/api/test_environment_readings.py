"""Contract tests for the live ENVIRONMENT readings relay
(routes/environment_readings.py).

Guarantees asserted:
  * with no live monitor wired (standalone AppContext) GET
    /api/environment/readings DEGRADES — every headline gauge ``value=None``,
    ``degraded=true``, status 200, never 500;
  * with a live monitor wired (a tiny fake exposing get_latest()/overall_status()/
    _sensors) the endpoint RELAYS real values into the four headline gauges and
    the full sensor list — by canonical name AND by driver type for
    auto-detected names;
  * an unavailable / error reading surfaces as ``value=None`` (N/A) with
    ``connected=false``;
  * a monitor whose get_latest() raises still degrades (never 500).

The router is mounted in mast.api.app, but mirroring test_admin.py these build a
throwaway app + include the router directly so the slice is tested in isolation.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.environment_readings import router
from mast.core.types import SensorReading


def _client(monitor=None) -> TestClient:
    app = FastAPI()
    ctx = AppContext()
    if monitor is not None:
        ctx.environment_monitor = monitor
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


class _FakeMonitor:
    """Minimal stand-in for EnvironmentMonitor: latest cache + sensor objects."""

    def __init__(self, latest, sensors=None, overall="ok", raise_latest=False):
        self._latest = dict(latest)
        self._sensors = dict(sensors or {})
        self._overall = overall
        self._raise = raise_latest

    def get_latest(self):
        if self._raise:
            raise RuntimeError("boom")
        return dict(self._latest)

    def overall_status(self):
        return self._overall


# Lightweight sensor stand-ins carrying the real driver class NAMES the route
# matches on (substring/class-name based, no heavy import needed).
class DL7VacuumSensor:  # noqa: N801 - mirrors the real driver class name
    pass


class LakeshoreTemperatureSensor:  # noqa: N801
    pass


# ── degraded (no monitor) ────────────────────────────────────────────────────
def test_readings_degrade_unwired() -> None:
    r = _client().get("/api/environment/readings")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    for k in ("vacuum", "temperature", "helium_level", "noise_level"):
        assert body[k]["value"] is None
        assert body[k]["connected"] is False
        assert body[k]["status"] == "unavailable"
    assert body["sensors"] == []
    assert body["overall_status"] == "unavailable"


# ── live relay by canonical name (placeholder names) ─────────────────────────
def test_readings_relay_by_canonical_name() -> None:
    latest = {
        "vacuum": SensorReading(value=2.8e-2, unit="Pa", status="ok"),
        "temperature": SensorReading(value=4.2, unit="K", status="ok"),
        "helium_level": SensorReading(value=0.0, unit="%", status="unavailable"),
        "noise_level": SensorReading(value=0.0, unit="pm", status="unavailable"),
    }
    r = _client(_FakeMonitor(latest)).get("/api/environment/readings")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["vacuum"]["value"] == pytest.approx(2.8e-2)
    assert body["vacuum"]["unit"] == "Pa"
    assert body["vacuum"]["connected"] is True
    assert body["temperature"]["value"] == pytest.approx(4.2)
    assert body["temperature"]["connected"] is True
    # unavailable placeholder → N/A
    assert body["helium_level"]["value"] is None
    assert body["helium_level"]["connected"] is False
    assert len(body["sensors"]) == 4
    assert body["overall_status"] == "ok"


# ── live relay by DRIVER TYPE (auto-detected non-canonical names) ────────────
def test_readings_relay_by_driver_type() -> None:
    latest = {
        "DL-7 真空计 (COM15)": SensorReading(value=5e-8, unit="Pa", status="ok"),
        "MODEL336 (simulated-C)": SensorReading(value=15.25, unit="K", status="warning"),
    }
    sensors = {
        "DL-7 真空计 (COM15)": DL7VacuumSensor(),
        "MODEL336 (simulated-C)": LakeshoreTemperatureSensor(),
    }
    r = _client(_FakeMonitor(latest, sensors)).get("/api/environment/readings")
    body = r.json()
    assert body["degraded"] is False
    # routed into headline slots by driver class name
    assert body["vacuum"]["value"] == pytest.approx(5e-8)
    assert body["temperature"]["value"] == pytest.approx(15.25)
    assert body["temperature"]["status"] == "warning"
    # sensor rows carry the driver type
    types = {s["name"]: s["type"] for s in body["sensors"]}
    assert types["DL-7 真空计 (COM15)"] == "DL7VacuumSensor"
    assert types["MODEL336 (simulated-C)"] == "LakeshoreTemperatureSensor"


# ── unavailable / error reading → N/A ────────────────────────────────────────
def test_readings_unavailable_is_na() -> None:
    latest = {
        "vacuum": SensorReading(value=0.0, unit="Pa", status="error"),
    }
    r = _client(_FakeMonitor(latest, overall="error")).get("/api/environment/readings")
    body = r.json()
    assert body["vacuum"]["value"] is None
    assert body["vacuum"]["connected"] is False
    assert body["vacuum"]["status"] == "error"
    row = body["sensors"][0]
    assert row["value"] is None and row["connected"] is False


# ── monitor that raises still degrades (never 500) ───────────────────────────
def test_readings_monitor_raises_degrades() -> None:
    r = _client(_FakeMonitor({}, raise_latest=True)).get("/api/environment/readings")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ── two synthetic sensors, one headline slot ──────────────────────────────
def test_a_connected_sensor_wins_the_headline_over_an_unavailable_one() -> None:
    """合成两个温度通道：不可用项在前时，可用项仍应占据表盘并保留各自状态。"""
    mon = _FakeMonitor(
        # 合成顺序：不可用通道在前。
        latest={
            "SPM (simulated-A)": SensorReading(value=0.0, unit="K", status="unavailable"),
            "Magnet (simulated-B)": SensorReading(value=12.75, unit="K", status="ok"),
        },
        sensors={"SPM (simulated-A)": LakeshoreTemperatureSensor(),
                 "Magnet (simulated-B)": LakeshoreTemperatureSensor()},
    )
    body = _client(mon).get("/api/environment/readings").json()
    assert body["temperature"]["value"] == pytest.approx(12.75)
    assert body["temperature"]["status"] == "ok"
    assert body["temperature"]["connected"] is True
    # 两个都还在完整列表里，各自的真实状态原样保留。
    by_name = {s["name"]: s for s in body["sensors"]}
    assert by_name["SPM (simulated-A)"]["connected"] is False
    assert by_name["Magnet (simulated-B)"]["connected"] is True


def test_the_handover_the_other_way_round_also_lands() -> None:
    """调换合成通道的可用状态，表盘应随之选择新的可用值。"""
    mon = _FakeMonitor(
        latest={
            "SPM (simulated-A)": SensorReading(value=13.5, unit="K", status="ok"),
            "Magnet (simulated-B)": SensorReading(value=0.0, unit="K", status="unavailable"),
        },
        sensors={"SPM (simulated-A)": LakeshoreTemperatureSensor(),
                 "Magnet (simulated-B)": LakeshoreTemperatureSensor()},
    )
    body = _client(mon).get("/api/environment/readings").json()
    assert body["temperature"]["value"] == pytest.approx(13.5)


def test_a_genuinely_dead_slot_still_reads_na_rather_than_vanishing() -> None:
    """两个都读不到时，槽位仍然要被占住 —— N/A 是一个答案，空缺不是。"""
    mon = _FakeMonitor(
        latest={
            "SPM (simulated-A)": SensorReading(value=0.0, unit="K", status="unavailable"),
            "Magnet (simulated-B)": SensorReading(value=0.0, unit="K", status="error"),
        },
        sensors={"SPM (simulated-A)": LakeshoreTemperatureSensor(),
                 "Magnet (simulated-B)": LakeshoreTemperatureSensor()},
    )
    body = _client(mon).get("/api/environment/readings").json()
    assert body["temperature"]["value"] is None
    assert body["temperature"]["connected"] is False
    assert body["temperature"]["status"] == "unavailable"


# ── 占位 vs 读不到 ─────────────────────────────────────────
#
# 这两件事今天在 `status` 上长得一模一样（都是 `unavailable`），而它们是不同的
# 事实：一台 COM 口拔掉的真空计插回去就好，`NoiseSensor` 插什么都没用 —— MAST
# 里根本没有那个量的驱动，也没有任何发现流程能把它填上。
#
# 面板据此分开画（读不到 = N/A 一行等着它回来；占位 = 归进「未接入」），所以
# 这个字段错一个方向就会说一句假话，两个方向都要钉。
def test_a_placeholder_says_it_is_a_placeholder() -> None:
    from mast.environment.placeholders import NoiseSensor, VacuumSensor

    latest = {
        "noise_level": SensorReading(value=0.0, unit="pm", status="unavailable"),
        "vacuum": SensorReading(value=0.0, unit="mbar", status="unavailable"),
    }
    sensors = {"noise_level": NoiseSensor(), "vacuum": VacuumSensor()}
    body = _client(_FakeMonitor(latest, sensors)).get("/api/environment/readings").json()
    flags = {s["name"]: s["placeholder"] for s in body["sensors"]}
    assert flags == {"noise_level": True, "vacuum": True}


def test_a_real_driver_that_cannot_read_is_not_a_placeholder() -> None:
    """方向的另一半：读不到的真表**不许**被说成「没接」。

    弄反了的症状是用户被告知这台机器压根没有真空计，于是不会去查那根线 ——
    而真空互锁正等着那个读数。
    """
    latest = {"DL-7 真空计 (COM15)": SensorReading(value=0.0, unit="Pa", status="error")}
    sensors = {"DL-7 真空计 (COM15)": DL7VacuumSensor()}
    body = _client(_FakeMonitor(latest, sensors)).get("/api/environment/readings").json()
    row = body["sensors"][0]
    assert row["placeholder"] is False
    assert row["status"] == "error"


def test_a_renamed_placeholder_is_still_a_placeholder() -> None:
    """autodetect 的 `_RenamedSensor` 包一层之后判据仍要成立。

    包装器是这一路上唯一会让 `isinstance` 失手的东西，而它失手的方向恰好是
    「占位又装回读数行里」—— 与这条要修的缺陷同一个形状。
    """
    from mast.environment.placeholders import NoiseSensor

    class _Renamed:
        def __init__(self, inner):
            self._inner = inner

    latest = {"噪声(重命名)": SensorReading(value=0.0, unit="pm", status="unavailable")}
    sensors = {"噪声(重命名)": _Renamed(NoiseSensor())}
    body = _client(_FakeMonitor(latest, sensors)).get("/api/environment/readings").json()
    assert body["sensors"][0]["placeholder"] is True


def test_an_unknown_sensor_object_is_not_called_a_placeholder() -> None:
    """拿不到传感器对象时按「不是占位」处理 —— 未知不该变成一句断言。"""
    latest = {"某个表": SensorReading(value=1.0, unit="x", status="ok")}
    body = _client(_FakeMonitor(latest)).get("/api/environment/readings").json()
    assert body["sensors"][0]["placeholder"] is False
