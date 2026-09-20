"""环境传感器热插拔后，归档与告警必须使用更新后的传感器集合。

启动时没有硬件、随后 rescan 或 adopt 的路径都应重新接好归档循环与 quiet gating。
仅在面板显示数值不能证明后台记录和门控已经生效。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.admin import router


class _FakeMonitor:
    """记录调用**顺序** —— 热插拔的两个回归都是关于顺序的。"""

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


class _FakeSink:
    def __init__(self) -> None:
        self.gated: list | None = None
        self.calls = 0

    def set_gated_sensors(self, names) -> None:
        self.calls += 1
        self.gated = None if names is None else list(names)


class _FakeRecorder:
    def __init__(self) -> None:
        self.sink = _FakeSink()


class _QuietGatedSensor:
    """一个自称「只在仪器安静时才记」的传感器(Nanonis 侧的电流镜像那一类)。"""

    quiet_gated = True

    def __init__(self, name: str = "tunnel_current") -> None:
        self._name = name

    def name(self) -> str:
        return self._name


def _client_with_monitor(monitor) -> TestClient:
    app = FastAPI()
    ctx = AppContext()
    ctx.environment_monitor = monitor
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _real_lakeshore(name: str = "SPM (COM17)"):
    from mast.environment.lakeshore_temp import LakeshoreTemperatureSensor

    # port=None → 没有 settings,构造时不碰任何串口。
    return LakeshoreTemperatureSensor(name=name)


@pytest.fixture()
def recorder() -> _FakeRecorder:
    """装一个假记录器进进程单例,用完还回去(它是全局状态)。"""
    from mast.envhistory import recorder as rec_mod

    previous = rec_mod.get_recorder()
    fake = _FakeRecorder()
    rec_mod.set_recorder(fake)
    try:
        yield fake
    finally:
        rec_mod.set_recorder(previous)


def _patch_build(monkeypatch, sensors):
    import mast.environment.autodetect as ad

    monkeypatch.setattr(ad, "build_environment_sensors",
                        lambda *a, **k: list(sensors))


# ════════════════════════════════════════════════════════════════════════════
def test_rescan_resyncs_the_archive_gate_to_the_new_sensor_set(
        monkeypatch, recorder) -> None:
    """**变异对照条**:删掉 rescan 里的 ``_resync_env_history_gates`` 调用,这条必红。

    热插拔进来的 quiet-gated 序列必须出现在门控名单里,否则它会在仪器不安静时
    被照记不误。
    """
    mon = _FakeMonitor(running=False)
    _patch_build(monkeypatch, [_real_lakeshore(), _QuietGatedSensor()])

    r = _client_with_monitor(mon).get("/api/environment/sensors/rescan")

    assert r.json()["ok"] is True
    assert mon.is_running is True                    # 归档循环起来了
    assert recorder.sink.gated == ["tunnel_current"]  # 门控也跟着换了


def test_adopt_resyncs_the_archive_gate_too(monkeypatch, tmp_path, recorder) -> None:
    """两个换传感器集合的入口都要补 —— 只修一个就是留下另一半的同一个洞。"""
    import mast.environment.config as envcfg

    monkeypatch.setattr(envcfg, "config_path",
                        lambda: tmp_path / "environment_sensors.json")
    mon = _FakeMonitor(running=False)
    _patch_build(monkeypatch, [_real_lakeshore(), _QuietGatedSensor()])

    r = _client_with_monitor(mon).post(
        "/api/environment/sensors/adopt",
        json={"sensors": [{"id": "lakeshore_com3_a", "name": "SPM",
                           "type": "lakeshore_temp", "port": "COM17",
                           "channel": "A", "unit": "K"}]},
    )

    assert r.json()["ok"] is True
    assert mon.is_running is True
    assert recorder.sink.gated == ["tunnel_current"]


def test_a_rig_with_nothing_to_gate_is_told_so_explicitly(
        monkeypatch, recorder) -> None:
    """空名单 = 「一个都不门控」,与 ``None`` = 「沿用默认那张名字表」刻意不同。

    传 ``None`` 会让门控退回一张写死的名字表,而这台机器上那些名字可能一个都不存在
    —— 那正是本仓库 KNOWN_KEYS 那类静默 no-op 的形状。
    """
    mon = _FakeMonitor(running=False)
    _patch_build(monkeypatch, [_real_lakeshore()])

    _client_with_monitor(mon).get("/api/environment/sensors/rescan")

    assert recorder.sink.calls == 1
    assert recorder.sink.gated == []
    assert recorder.sink.gated is not None


def test_the_gate_resync_never_fails_a_rescan(monkeypatch, recorder) -> None:
    """记账坏了不许把热插拔本身弄失败 —— 归档循环该起还是要起。"""

    def _boom(_names):
        raise RuntimeError("sink exploded")

    recorder.sink.set_gated_sensors = _boom          # type: ignore[assignment]
    mon = _FakeMonitor(running=False)
    _patch_build(monkeypatch, [_real_lakeshore()])

    r = _client_with_monitor(mon).get("/api/environment/sensors/rescan")

    assert r.json()["ok"] is True
    assert mon.is_running is True


def test_no_recorder_wired_is_not_an_error(monkeypatch) -> None:
    """独立 API 进程里没有记录器 —— 静默跳过,不 500。"""
    from mast.envhistory import recorder as rec_mod

    previous = rec_mod.get_recorder()
    rec_mod.set_recorder(None)
    try:
        mon = _FakeMonitor(running=False)
        _patch_build(monkeypatch, [_real_lakeshore()])
        r = _client_with_monitor(mon).get("/api/environment/sensors/rescan")
        assert r.status_code == 200 and r.json()["ok"] is True
    finally:
        rec_mod.set_recorder(previous)


def test_a_failed_rebuild_leaves_the_gate_untouched(monkeypatch, recorder) -> None:
    """重建失败时不该拿一份没建成的传感器集合去改门控。"""
    import mast.environment.autodetect as ad

    def _boom(*a, **k):
        raise RuntimeError("rebuild boom")

    monkeypatch.setattr(ad, "build_environment_sensors", _boom)
    mon = _FakeMonitor(running=True)

    r = _client_with_monitor(mon).get("/api/environment/sensors/rescan")

    assert r.json()["degraded"] is True
    assert recorder.sink.calls == 0
    assert mon.is_running is True                    # 循环被放回原样
