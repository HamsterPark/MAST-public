"""``GetTemperature``:读得到就给数,读不到要说清是**哪一种**读不到。

驱动齐全、数据落库,却**没有任何技能能问一句「现在几度」**——
``mast/agents/**`` 与 ``mast/skills/**`` 对温度此前零引用。等降温、判「到温了没有」
这类条件只能靠人看面板,于是 campaign 的 WAITING_CONDITION 等不了物理量。

这条测试守的是那个补上的口子,以及它**不许**退化成一个 ``float | None``:

* ``no_sensor``   —— 等下去永远等不到 ⇒ 拒绝,别轮询;
* ``unavailable`` —— 关掉占着 COM 口的程序就好 ⇒ 可以等;
* ``no_source``   —— MAST 自己没接上 ⇒ 与仪器无关。

从仓库根跑::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_temperature_skill.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if _MASTV2_ROOT not in sys.path:
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core import temperature as temp  # noqa: E402
from mast.core.temperature import (  # noqa: E402
    NO_SENSOR,
    NO_SOURCE,
    UNAVAILABLE,
    TempChannel,
    TempReading,
)
from mast.core.types import SafetyLevel, SkillCategory  # noqa: E402
from mast.skills.builtins.temperature import GetTemperature  # noqa: E402


class _Ctx:
    """技能不碰 context —— 只读、不取仪器令牌、不发一条 Nanonis 命令。

    故意让任何属性访问都炸:一旦哪天有人在这个只读技能里伸手去摸硬件,
    这里立刻会红。
    """

    def __getattr__(self, name):  # pragma: no cover - 触发即是失败
        raise AssertionError(f"GetTemperature 不该碰 context.{name}")


@pytest.fixture(autouse=True)
def _isolated_source():
    """每条用例前后都把进程级注入口清干净 —— 它是全局状态。"""
    temp.set_source(None)
    temp.set_channels_source(None)
    yield
    temp.set_source(None)
    temp.set_channels_source(None)


def _run(params: dict | None = None):
    return GetTemperature().execute(_Ctx(), params or {})


# ════════════════════════════════════════════════════════════════════════════
def test_metadata_is_read_only_and_fully_declared() -> None:
    m = GetTemperature().metadata()
    assert m.category is SkillCategory.READ
    assert m.safety_level is SafetyLevel.AUTO      # safety_level 不可省
    names = {p.name for p in m.parameters}
    assert names == {"channel", "max_age_s"}
    # 两个都可选:不传 channel = 自动选样品台;不传 max_age_s = 不判陈旧。
    assert all(not p.required for p in m.parameters)


def test_a_live_reading_comes_through_with_its_age_and_source() -> None:
    temp.set_source(lambda ch: TempReading(
        value_k=4.21, age_s=2.5, source="LakeshoreTemperatureSensor",
        channel="SPM (COM13)"))
    res = _run()
    assert res.success is True
    assert res.data["value_k"] == pytest.approx(4.21)
    assert res.data["age_s"] == pytest.approx(2.5)
    assert res.data["source"] == "LakeshoreTemperatureSensor"
    assert res.data["channel"] == "SPM (COM13)"
    assert res.data["reason"] is None
    assert "what_to_do" not in res.data          # 有值就没有「怎么办」


def test_no_sensor_and_unavailable_reach_the_agent_as_different_answers() -> None:
    """**变异对照条**:把两个 reason 折叠成一个,这条必须红。"""
    temp.set_source(lambda ch: TempReading(reason=NO_SENSOR))
    no_sensor = _run().data
    temp.set_source(lambda ch: TempReading(reason=UNAVAILABLE,
                                           channel="SPM (COM13)"))
    unavailable = _run().data

    assert no_sensor["value_k"] is None and unavailable["value_k"] is None
    assert no_sensor["reason"] != unavailable["reason"]
    # 「怎么办」也必须不同 —— 一个是别等,一个是可以等。
    assert no_sensor["what_to_do"] != unavailable["what_to_do"]
    assert "永远等不到" in no_sensor["what_to_do"]
    assert "可能会好" in unavailable["what_to_do"]


def test_reading_nothing_is_still_a_successful_query() -> None:
    """「读不到」是一个**答案**,不是工具坏了。

    报成 error 会让一次寻常的「还没装温度计」看起来像故障,并把 reason 里
    那句可执行的信息一起丢掉。
    """
    temp.set_source(lambda ch: TempReading(reason=NO_SENSOR))
    res = _run()
    assert res.success is True
    assert res.error in (None, "")


def test_an_unwired_source_says_no_source_not_no_sensor() -> None:
    """MAST 这一侧没接上 ≠ 这台机器没温度计。"""
    res = _run()                                  # 夹具已把源清空
    assert res.data["reason"] == NO_SOURCE
    assert res.data["reason"] != NO_SENSOR
    assert res.data["available_channels"] is None  # 「问不到」不是「一个都没有」


def test_staleness_is_explicit_only() -> None:
    """不传 ``max_age_s`` 就**不给** freshness —— 不替调用方拍一个阈值。

    一个用默认阈值算出来的「够新」结论,看起来和真的一模一样。
    """
    temp.set_source(lambda ch: TempReading(value_k=77.3, age_s=900.0,
                                           source="X", channel="SPM"))
    assert "freshness" not in _run().data

    assert _run({"max_age_s": 60.0}).data["freshness"] == "stale"
    assert _run({"max_age_s": 3600.0}).data["freshness"] == "fresh"


def test_an_unknown_age_is_unknown_freshness_not_fresh() -> None:
    temp.set_source(lambda ch: TempReading(value_k=77.3, age_s=None,
                                           source="X", channel="SPM"))
    assert _run({"max_age_s": 60.0}).data["freshness"] == "unknown"


def test_the_channel_parameter_reaches_the_source() -> None:
    seen: list = []

    def _src(ch):
        seen.append(ch)
        return TempReading(value_k=77.3, age_s=1.0, source="X", channel=ch or "")

    temp.set_source(_src)
    _run({"channel": "  Magnet  "})               # 顺带验证首尾空白被吃掉
    assert seen == ["Magnet"]
    _run({"channel": ""})                          # 空串 = 没指名
    assert seen[-1] is None


def test_available_channels_shows_placeholders_as_not_real() -> None:
    """占位与真驱动今天都报 ``unavailable`` —— 只有 ``real`` 分得开。"""
    temp.set_source(lambda ch: TempReading(reason=NO_SENSOR))
    temp.set_channels_source(lambda: [
        TempChannel(name="temperature", value=0.0, unit="K",
                    status="unavailable", driver="TemperatureSensor", real=False),
        TempChannel(name="Magnet (COM13)", value=77.3, unit="K", status="ok",
                    driver="LakeshoreTemperatureSensor", real=True),
    ])
    chans = _run().data["available_channels"]
    assert [c["name"] for c in chans] == ["temperature", "Magnet (COM13)"]
    assert [c["real"] for c in chans] == [False, True]
    assert chans[1]["value_k"] == pytest.approx(77.3)


def test_a_raising_source_never_breaks_the_skill() -> None:
    def _boom(_ch):
        raise RuntimeError("serial handle gone")

    temp.set_source(_boom)
    res = _run()
    assert res.success is True
    assert res.data["reason"] == NO_SOURCE
