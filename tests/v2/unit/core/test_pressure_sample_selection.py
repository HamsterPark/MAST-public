"""真空读数按物理单位与连接状态选择，不按名称猜测。

名称匹配的断开占位符不能遮蔽有效压力传感器；传感器类别仍由互锁独立核验。"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.types import SensorReading  # noqa: E402


class _PlaceholderSensor:
    """没接规的机器上自动探测出来的那个 —— 名字恰好就叫 vacuum。"""


class _DL7:
    """真规。"""


class _Monitor:
    def __init__(self, latest: dict, sensors: dict | None = None) -> None:
        self._latest = latest
        self._sensors = sensors or {}

    def get_latest(self) -> dict:
        return self._latest


def _rt(latest: dict | None, sensors: dict | None = None):
    rt = CoreRuntime.__new__(CoreRuntime)
    rt._monitor = _Monitor(latest, sensors) if latest is not None else None
    return rt


def _r(value, unit, status="ok"):
    return SensorReading(value=value, unit=unit, status=status)


# ════════════════════════════════════════════════════════════════════════════
def test_a_live_gauge_wins_over_the_dead_placeholder_named_vacuum() -> None:
    """存在可用压力传感器时，应优先使用其读数，而不是同名或更早出现的占位符。"""
    rt = _rt(
        {
            "vacuum": _r(None, "mbar", status="unavailable"),   # 占位符，从没读到值
            "DL-7 (COM14)": _r(2.4e-8, "mbar"),                   # 真规
        },
        {"vacuum": _PlaceholderSensor(), "DL-7 (COM14)": _DL7()},
    )
    s = rt._latest_pressure_sample()
    assert s is not None
    assert s.sensor_name == "DL-7 (COM14)", "又选中了那个死的占位符"
    assert s.value == pytest.approx(2.4e-8)
    assert s.sensor_class == "_DL7"


def test_the_placeholder_is_still_returned_when_it_is_all_there_is() -> None:
    """都不活时**仍要返回一个**，不能返回 None。

    互锁需要 `sensor_class` 才能给出那句「真空计是占位实现，读数恒为 0，这不是
    『完美真空』而是『没有数据』」的拒绝理由。返回 None 会把拒绝的**原因**一起
    丢掉，用户只会看到一句「读不到」—— 而「为什么读不到」正是他要的信息。
    """
    rt = _rt(
        {"vacuum": _r(0.0, "mbar", status="unavailable")},
        {"vacuum": _PlaceholderSensor()},
    )
    s = rt._latest_pressure_sample()
    assert s is not None, "返回 None 会连拒绝理由一起丢掉"
    assert s.sensor_name == "vacuum"
    assert s.status == "unavailable"
    assert s.sensor_class == "_PlaceholderSensor", "互锁靠这个字段按身份先拒"


def test_selection_is_by_unit_not_by_name() -> None:
    """名字里没有 vacuum 的真规照样要被选中 —— 判别字段是 unit。"""
    rt = _rt({"Chamber (COM14)": _r(3.1e-9, "mbar")}, {"Chamber (COM14)": _DL7()})
    s = rt._latest_pressure_sample()
    assert s is not None and s.sensor_name == "Chamber (COM14)"


def test_a_non_pressure_unit_is_not_mistaken_for_a_gauge() -> None:
    """依据物理单位选择压力读数；温度、液位与噪声通道不能因名称接近而被误用。"""
    rt = _rt({
        "Magnet (COM17)": _r(80.0, "K"),
        "helium_level": _r(62.0, "%"),
        "noise_level": _r(3.2, "pm"),
    })
    assert rt._latest_pressure_sample() is None


def test_an_odd_unit_still_falls_back_to_the_name() -> None:
    """单位写法不在表里（如 Torr）时，名字兜底仍然生效 —— 只是排在 unit 之后。"""
    rt = _rt({"vacuum gauge": _r(1.8e-9, "Torr")}, {"vacuum gauge": _DL7()})
    s = rt._latest_pressure_sample()
    assert s is not None and s.sensor_name == "vacuum gauge"


def test_no_monitor_and_empty_monitor_both_give_no_sample() -> None:
    """「没有样本」是互锁认得的拒绝理由，不能变成异常。"""
    assert _rt(None)._latest_pressure_sample() is None
    assert _rt({})._latest_pressure_sample() is None


def test_a_raising_monitor_never_propagates() -> None:
    """取样口坏掉必须表现为「拒绝粗动」，不能把调用方带崩。"""

    class _Boom:
        def get_latest(self):
            raise RuntimeError("serial handle gone")

    rt = CoreRuntime.__new__(CoreRuntime)
    rt._monitor = _Boom()
    assert rt._latest_pressure_sample() is None
