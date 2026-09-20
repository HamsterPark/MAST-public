"""温度公共只读口：三种「读不到」必须各自可辨，年龄不许被伪造。

## 为什么这条测试存在

在此之前温度只有一个私有口 ``CoreRuntime._latest_temperature_k``，返回
``float | None``。那个 ``None`` 同时是三句话：

    「这台机器没装温度计」          —— 等下去永远等不到，该拒绝
    「装了，但此刻读不到」          —— 关掉占着 COM17 的程序就好，该去看端口
    「有读数，但太旧」              —— 该再等一拍

对调用方要做的事完全相反。折叠之后，campaign 的「等降温」在
「先启动 MAST、后关 另一个串口程序」这个真实顺序下就会静默变成「永远等」。

## 变异验证（本文件自带）

``test_no_sensor_and_unavailable_are_not_the_same_answer`` 是这里的对照条：
把 ``read_temperature`` 里的 ``NO_SENSOR`` 与 ``UNAVAILABLE`` 折叠成同一个字符串，
它必须变红。同理 ``test_an_unreadable_age_is_never_reported_as_zero`` 对应
「把 ``age_s`` 的 ``None`` 兜底成 0.0」这一处变异。

从仓库根跑::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_temperature_port.py -q
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

from datetime import datetime, timedelta  # noqa: E402

import pytest  # noqa: E402

from mast.core.temperature import (  # noqa: E402
    AMBIGUOUS_CHANNEL,
    NO_SENSOR,
    NO_SOURCE,
    UNAVAILABLE,
    UNKNOWN_CHANNEL,
    TempChannel,
    TempReading,
    age_s,
    channels,
    latest_temperature,
    read_temperature,
    set_channels_source,
    set_source,
    to_kelvin,
)

#: 注入时钟的基准。所有带年龄的用例都从这一刻回看。
_NOW = datetime(2000, 1, 1, 12, 0, 0)


def _ts(seconds_ago: float) -> str:
    return (_NOW - timedelta(seconds=seconds_ago)).isoformat()


def _real(name: str, value, unit: str = "K", status: str = "ok",
          seconds_ago: float | None = 0.0) -> TempChannel:
    return TempChannel(
        name=name, value=value, unit=unit, status=status,
        timestamp=_ts(seconds_ago) if seconds_ago is not None else "",
        driver="LakeshoreTemperatureSensor", real=True,
    )


def _placeholder(name: str = "temperature") -> TempChannel:
    """占位实现：MAST 在这台机器上**没有**温度驱动。"""
    return TempChannel(name=name, value=0.0, unit="K", status="unavailable",
                       driver="TemperatureSensor", real=False)


# ════════════════════════════════════════════════════════════════════════════
# 三种「读不到」各自可辨
# ════════════════════════════════════════════════════════════════════════════
def test_no_sensor_and_unavailable_are_not_the_same_answer() -> None:
    """**变异对照条**：把这两个原因折叠成一个，这条必须红。

    两者的处置相反：``no_sensor`` 等下去永远等不到（该拒绝），``unavailable``
    关掉占端口的程序就好（该去看端口）。
    """
    no_thermometer = read_temperature([_placeholder()], now=_NOW)
    mute_thermometer = read_temperature(
        [_real("SPM (COM17)", None, status="unavailable")], now=_NOW)

    assert no_thermometer.value_k is None and mute_thermometer.value_k is None
    assert no_thermometer.reason == NO_SENSOR
    assert mute_thermometer.reason == UNAVAILABLE
    assert no_thermometer.reason != mute_thermometer.reason


def test_no_sensor_when_the_machine_has_no_temperature_channel_at_all() -> None:
    assert read_temperature([], now=_NOW).reason == NO_SENSOR
    assert read_temperature(None, now=_NOW).reason == NO_SENSOR


def test_an_error_reading_is_unavailable_not_zero_kelvin() -> None:
    """``read_all`` 读失败时写的是 ``value=0.0, status="error"``。

    把它当读数就是把 **0 K** 记进实验记录 —— 比没有温度糟得多。
    """
    r = read_temperature([_real("SPM", 0.0, status="error")], now=_NOW)
    assert r.value_k is None
    assert r.reason == UNAVAILABLE


def test_unavailable_still_names_which_thermometer_went_mute() -> None:
    """读不到时**仍然**回报通道名 —— 「哪个温度计哑了」才是能动手的信息。"""
    r = read_temperature([_real("SPM (COM17)", None, status="unavailable",
                                seconds_ago=8.0)], now=_NOW)
    assert r.reason == UNAVAILABLE
    assert r.channel == "SPM (COM17)"
    assert r.age_s == pytest.approx(8.0)


def test_stale_is_the_callers_call_and_the_age_always_travels() -> None:
    """本函数不判陈旧，但 ``age_s`` 必须随值返回，否则上层判不了。"""
    r = read_temperature([_real("SPM", 80.0, seconds_ago=900.0)], now=_NOW)
    assert r.value_k == pytest.approx(80.0)
    assert r.reason is None                      # 有值就没有 reason
    assert r.age_s == pytest.approx(900.0)
    assert r.freshness(60.0) == "stale"
    assert r.freshness(3600.0) == "fresh"


def test_freshness_is_three_state_not_a_bool() -> None:
    """``bool`` 会被 ``if not stale:`` 一句把「不知道」当成「新鲜」。"""
    unknown_age = read_temperature([_real("SPM", 80.0, seconds_ago=None)], now=_NOW)
    assert unknown_age.value_k == pytest.approx(80.0)
    assert unknown_age.age_s is None
    assert unknown_age.freshness(60.0) == "unknown"
    # 三个取值互不相同 —— 折叠成两态这条就红。
    assert {unknown_age.freshness(60.0),
            read_temperature([_real("SPM", 80.0, seconds_ago=1.0)],
                             now=_NOW).freshness(60.0),
            read_temperature([_real("SPM", 80.0, seconds_ago=999.0)],
                             now=_NOW).freshness(60.0)} == {
        "unknown", "fresh", "stale"}


def test_an_unreadable_age_is_never_reported_as_zero() -> None:
    """**变异对照条**：给 ``age_s`` 兜一个 0.0，这条必须红。

    0 秒的年龄会让任何 ``age_s <= limit`` 的判据无条件通过 —— 把「不知道多旧」
    变成「刚刚读的」。
    """
    assert age_s("", now=_NOW) is None
    assert age_s(None, now=_NOW) is None
    assert age_s("不是时间戳", now=_NOW) is None
    r = read_temperature([_real("SPM", 4.2, seconds_ago=None)], now=_NOW)
    assert r.age_s is None and r.age_s != 0.0


def test_no_source_is_not_unavailable() -> None:
    """MAST 自己没接上 ≠ 仪器不给数。前者重启才会变，后者关个程序就好。"""
    set_source(None)
    set_channels_source(None)
    r = latest_temperature()
    assert r.reason == NO_SOURCE
    assert r.reason != UNAVAILABLE
    assert channels() is None            # 「问不到」不是「一个都没有」


# ════════════════════════════════════════════════════════════════════════════
# 通道选择
# ════════════════════════════════════════════════════════════════════════════
def test_the_stage_wins_over_the_magnet_dewar() -> None:
    r = read_temperature(
        [_real("Magnet (COM17)", 80.0), _real("SPM (COM17)", 4.2)], now=_NOW)
    assert r.value_k == pytest.approx(4.2)
    assert r.channel == "SPM (COM17)"


def test_a_readable_channel_beats_a_preferred_but_mute_one() -> None:
    """两个共享连接的温度通道可交替可用；选择逻辑应使用当次可用读数。"""
    r = read_temperature(
        [_real("SPM (COM17)", None, status="unavailable"),
         _real("Magnet (COM17)", 80.0)], now=_NOW)
    assert r.value_k == pytest.approx(80.0)
    assert r.channel == "Magnet (COM17)"


def test_a_named_channel_is_honoured_even_when_another_is_readable() -> None:
    r = read_temperature(
        [_real("SPM", 4.2), _real("Magnet", 80.0)], channel="Magnet", now=_NOW)
    assert r.value_k == pytest.approx(80.0)
    assert r.channel == "Magnet"


def test_a_short_name_matches_the_com_suffixed_channel() -> None:
    """配置里写 ``SPM``，监控里那个通道叫 ``SPM (COM17)``。"""
    r = read_temperature([_real("SPM (COM17)", 4.2), _real("Magnet (COM17)", 80.0)],
                         channel="spm", now=_NOW)
    assert r.value_k == pytest.approx(4.2)


def test_an_unknown_channel_is_not_reported_as_no_thermometer() -> None:
    """折叠成 ``no_sensor`` 会让用户去找一个并不缺的传感器。"""
    r = read_temperature([_real("SPM", 4.2)], channel="Cryostat", now=_NOW)
    assert r.reason == UNKNOWN_CHANNEL
    assert r.reason != NO_SENSOR
    assert r.channel == "Cryostat"       # 回显你要的那个名字


def test_an_ambiguous_name_refuses_instead_of_picking_one() -> None:
    """挑错的那一半时间没人会发现。"""
    r = read_temperature([_real("SPM A", 4.2), _real("SPM B", 5.1)],
                         channel="SPM", now=_NOW)
    assert r.reason == AMBIGUOUS_CHANNEL
    assert r.value_k is None


def test_naming_a_placeholder_channel_says_no_sensor() -> None:
    r = read_temperature([_placeholder("temperature")],
                         channel="temperature", now=_NOW)
    assert r.reason == NO_SENSOR


def test_an_unknown_sensor_object_is_not_treated_as_a_placeholder() -> None:
    """``real is None`` = 不知道。当成占位会把一台正报着 77 K 的机器判成没温度计。"""
    unknown = TempChannel(name="Magnet (COM17)", value=80.0, unit="K",
                          status="ok", timestamp=_ts(1.0), real=None)
    r = read_temperature([unknown], now=_NOW)
    assert r.value_k == pytest.approx(80.0)


# ════════════════════════════════════════════════════════════════════════════
# 单位换算
# ════════════════════════════════════════════════════════════════════════════
def test_a_warning_band_reading_is_still_a_reading() -> None:
    """降温途中温度必然长时间落在 warn 带里 —— 那正是要看的那个数。

    旧的 ``status != "ok" -> None`` 会让一台正在从 300 K 降下来的机器对
    「现在几度」一路回答「读不到」。
    """
    assert to_kelvin(300.0, "K", "warning") == pytest.approx(300.0)
    assert to_kelvin(310.0, "K", "alarm") == pytest.approx(310.0)
    assert to_kelvin(0.0, "K", "error") is None
    assert to_kelvin(None, "K", "unavailable") is None


def test_a_non_temperature_unit_is_never_mistaken_for_kelvin() -> None:
    """2e-13 A 的电流读数被当成 2e-13 K，比没有温度更糟。"""
    r = read_temperature([
        TempChannel(name="tunnel_current", value=2e-13, unit="A", status="ok"),
        TempChannel(name="noise_level", value=3.2, unit="pm", status="ok"),
    ], now=_NOW)
    assert r.value_k is None


def test_celsius_is_converted() -> None:
    assert to_kelvin(20.0, "C") == pytest.approx(293.15)
    assert to_kelvin(20.0, "degC") == pytest.approx(293.15)


# ════════════════════════════════════════════════════════════════════════════
# 注入口
# ════════════════════════════════════════════════════════════════════════════
def test_a_raising_source_degrades_to_no_source_and_never_propagates() -> None:
    def _boom(_ch):
        raise RuntimeError("serial handle gone")

    set_source(_boom)
    try:
        assert latest_temperature().reason == NO_SOURCE
    finally:
        set_source(None)


def test_a_source_returning_junk_is_refused_not_relayed() -> None:
    set_source(lambda _ch: 80.0)     # 一个裸 float，不是 TempReading
    try:
        assert latest_temperature().reason == NO_SOURCE
    finally:
        set_source(None)


def test_the_injected_source_receives_the_requested_channel() -> None:
    seen: list = []

    def _src(ch):
        seen.append(ch)
        return TempReading(value_k=4.2, age_s=1.0, source="x", channel=ch or "")

    set_source(_src)
    try:
        assert latest_temperature("Magnet").value_k == pytest.approx(4.2)
        assert seen == ["Magnet"]
    finally:
        set_source(None)


def test_channels_distinguishes_none_from_empty() -> None:
    set_channels_source(lambda: [])
    try:
        assert channels() == []          # 确实一个都没有
    finally:
        set_channels_source(None)
    assert channels() is None            # 问不到
