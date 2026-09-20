"""温度按单位选择，不依赖名字是否含 temp。

用合成读数验证通道选择、样品台优先级、单位换算与缺失状态，
不复用任何实际仪器的端口或温度记录。
"""
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
from mast.core.types import SensorReading  # noqa: E402


class _Monitor:
    def __init__(self, latest: dict) -> None:
        self._latest = latest

    def get_latest(self) -> dict:
        return self._latest


def _rt(latest: dict | None):
    rt = CoreRuntime.__new__(CoreRuntime)
    rt._monitor = _Monitor(latest) if latest is not None else None
    return rt


def _r(value, unit, status="ok"):
    return SensorReading(value=value, unit=unit, status=status)


# ════════════════════════════════════════════════════════════════════════════
def test_sensor_names_without_temp_are_found() -> None:
    """合成名称都不含 temp；仍须识别其中状态正常且单位为 K 的读数。"""
    rt = _rt({
        "SPM (simulated-A)":    _r(None, "K", status="unavailable"),
        "Magnet (simulated-B)": _r(16.05, "K"),
        "helium_level":  _r(None, "%", status="unavailable"),
        "vacuum":        _r(None, "mbar", status="unavailable"),
        "noise_level":   _r(None, "pm", status="unavailable"),
    })
    assert rt._latest_temperature_k() == pytest.approx(16.05)


def test_the_stage_wins_over_other_temperature_sensors() -> None:
    """磁体杜瓦的温度不等于样品台的温度 —— 两个都可用时取样品台。"""
    rt = _rt({
        "Magnet (simulated-B)": _r(14.0, "K"),
        "SPM (simulated-A)":    _r(6.0, "K"),
    })
    assert rt._latest_temperature_k() == pytest.approx(6.0)


def test_a_non_temperature_unit_is_never_mistaken_for_kelvin() -> None:
    """噪声、电流和百分比的合成读数不能被解释为温度。"""
    rt = _rt({
        "noise_level":    _r(3.2, "pm"),
        "tunnel_current": _r(2e-13, "A"),
        "helium_level":   _r(62.0, "%"),
    })
    assert rt._latest_temperature_k() is None


def test_celsius_is_converted() -> None:
    rt = _rt({"Chiller": _r(20.0, "C")})
    assert rt._latest_temperature_k() == pytest.approx(293.15)


def test_an_unavailable_sensor_is_skipped_not_read_as_zero() -> None:
    """status != ok 的通道 value 可能是 None 或 0 —— 都不能当成 0 K。"""
    rt = _rt({
        "SPM (simulated-A)":    _r(None, "K", status="unavailable"),
        "Magnet (simulated-B)": _r(0.0, "K", status="error"),
    })
    assert rt._latest_temperature_k() is None


def test_no_monitor_and_empty_monitor_both_degrade_to_none() -> None:
    assert _rt(None)._latest_temperature_k() is None
    assert _rt({})._latest_temperature_k() is None


def test_a_raising_monitor_never_propagates() -> None:
    """温度只是注记 —— 它坏掉绝不能让粗动自检整个失败。"""

    class _Boom:
        def get_latest(self):
            raise RuntimeError("serial handle gone")

    rt = CoreRuntime.__new__(CoreRuntime)
    rt._monitor = _Boom()
    assert rt._latest_temperature_k() is None
