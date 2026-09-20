"""传感器存在性与当前读数可用性来自不同状态。

已配置但尚无缓存数据时，应返回 unavailable，不能报告 no_sensor。
测试覆盖启动后热插拔、共享连接和通道选择。"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.temperature import NO_SENSOR, UNAVAILABLE  # noqa: E402
from mast.core.types import SensorReading  # noqa: E402
from mast.environment.lakeshore_temp import LakeshoreTemperatureSensor  # noqa: E402
from mast.environment.placeholders import (  # noqa: E402
    NoiseSensor,
    TemperatureSensor,
    VacuumSensor,
)


class _Monitor:
    def __init__(self, latest: dict | None = None, sensors: dict | None = None):
        self._latest = dict(latest or {})
        self._sensors = dict(sensors or {})

    def get_latest(self) -> dict:
        return self._latest


def _rt(latest=None, sensors=None) -> CoreRuntime:
    rt = CoreRuntime.__new__(CoreRuntime)
    rt._monitor = _Monitor(latest, sensors)
    return rt


def _lakeshore(name: str, channel: str = "A") -> LakeshoreTemperatureSensor:
    # transport 显式给 None：这些用例从不真的去读串口，读数由 _Monitor 供。
    return LakeshoreTemperatureSensor(name=name, channel=channel, transport=None)


# ════════════════════════════════════════════════════════════════════════════
def test_a_configured_thermometer_with_no_cached_reading_is_unavailable() -> None:
    """归档循环还没跑过一拍 —— 「配了但读不到」，**不是**「没装」。

    折叠成 ``no_sensor`` 就是告诉调用方「等下去永远等不到」，而它其实只要等
    rescan/循环起来就有值了。
    """
    rt = _rt(latest={}, sensors={"SPM (COM17)": _lakeshore("SPM (COM17)")})

    chans = rt.temperature_channels()
    assert [c.name for c in chans] == ["SPM (COM17)"]
    assert chans[0].real is True

    r = rt.latest_temperature()
    assert r.value_k is None
    assert r.reason == UNAVAILABLE
    assert r.reason != NO_SENSOR
    assert r.channel == "SPM (COM17)"


def test_only_placeholders_is_no_sensor() -> None:
    """占位实现 = MAST 在这台机器上没有温度驱动，插什么都不会好。"""
    rt = _rt(latest={"temperature": SensorReading(value=0.0, unit="K",
                                                  status="unavailable")},
             sensors={"temperature": TemperatureSensor()})

    chans = rt.temperature_channels()
    assert [c.real for c in chans] == [False]
    assert rt.latest_temperature().reason == NO_SENSOR


def test_non_temperature_sensors_are_not_temperature_channels() -> None:
    rt = _rt(
        latest={
            "vacuum": SensorReading(value=1e-9, unit="mbar"),
            "noise_level": SensorReading(value=3.2, unit="pm"),
        },
        sensors={"vacuum": VacuumSensor(), "noise_level": NoiseSensor()},
    )
    assert rt.temperature_channels() == []
    assert rt.latest_temperature().reason == NO_SENSOR


def test_the_real_rig_pair_reports_the_live_channel_with_its_age() -> None:
    """两个合成温度通道共享连接时，一个不可用不能遮蔽另一个可用通道。"""
    rt = _rt(
        latest={
            "SPM (COM17)": SensorReading(value=None, unit="K",
                                        status="unavailable"),
            "Magnet (COM17)": SensorReading(value=80.0, unit="K", status="ok"),
        },
        sensors={"SPM (COM17)": _lakeshore("SPM (COM17)", "A"),
                 "Magnet (COM17)": _lakeshore("Magnet (COM17)", "B")},
    )
    r = rt.latest_temperature()
    assert r.value_k == pytest.approx(80.0)
    assert r.channel == "Magnet (COM17)"
    assert r.source == "LakeshoreTemperatureSensor"
    # SensorReading 的默认 timestamp 是「现在」——年龄必须是个数，不是 None。
    assert r.age_s is not None and r.age_s < 60.0


def test_a_named_channel_reaches_the_mute_one_and_says_why() -> None:
    rt = _rt(
        latest={
            "SPM (COM17)": SensorReading(value=None, unit="K",
                                        status="unavailable"),
            "Magnet (COM17)": SensorReading(value=80.0, unit="K", status="ok"),
        },
        sensors={"SPM (COM17)": _lakeshore("SPM (COM17)", "A"),
                 "Magnet (COM17)": _lakeshore("Magnet (COM17)", "B")},
    )
    r = rt.latest_temperature("SPM")
    assert r.value_k is None
    assert r.reason == UNAVAILABLE          # 哑的是它，不是「没装」
    assert r.channel == "SPM (COM17)"


def test_the_legacy_private_accessor_still_answers_the_same_float() -> None:
    """``_latest_temperature_k`` 保留并委托 —— 粗动里程表的注记不受影响。"""
    rt = _rt(
        latest={"Magnet (COM17)": SensorReading(value=80.0, unit="K"),
                "SPM (COM17)": SensorReading(value=4.2, unit="K")},
        sensors={"Magnet (COM17)": _lakeshore("Magnet (COM17)", "B"),
                 "SPM (COM17)": _lakeshore("SPM (COM17)", "A")},
    )
    assert rt._latest_temperature_k() == pytest.approx(4.2)   # 样品台优先
    assert rt.latest_temperature().value_k == pytest.approx(4.2)


def test_no_monitor_never_raises() -> None:
    rt = CoreRuntime.__new__(CoreRuntime)
    rt._monitor = None
    assert rt.temperature_channels() == []
    assert rt.latest_temperature().value_k is None
    assert rt._latest_temperature_k() is None


def test_a_raising_monitor_never_propagates() -> None:
    class _Boom:
        def get_latest(self):
            raise RuntimeError("serial handle gone")

    rt = CoreRuntime.__new__(CoreRuntime)
    rt._monitor = _Boom()
    assert rt.latest_temperature().value_k is None
    assert rt._latest_temperature_k() is None
