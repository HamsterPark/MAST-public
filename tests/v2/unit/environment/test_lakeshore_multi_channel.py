"""A two-input temperature controller must read out as two named sensors.

— the ENVIRONMENT rail showed::

    Vacuum        N/A
    Temperature   77.42 K
    Helium Level  N/A
    Noise Level   N/A
    MODEL335 (COM13)  77.42 K      ← the same reading, a second time
    vacuum           N/A          ← and the placeholders, a second time
    helium_level     N/A
    noise_level      N/A
    「应该显示为SPM和Magnet」

Two separate defects, both pinned here:

1. ``autodetect`` built a sensor for ``default_channels(model)[0]`` only, so a
   Lake Shore 335 (two inputs) produced ONE row. "SPM" and "Magnet" are not
   words to hard-code — they are the INPUT NAMES stored in the controller
   (``INNAME?``), which is the only place that knows which sensor is which.
2. The rail de-duplicated against ``SensorEntry.type``, which carries the driver
   CLASS name (``LakeshoreTemperatureSensor``), so the comparison against
   "temperature" never matched and every sensor was printed twice. The mapping
   existed server-side; it was simply never sent to the client (now ``kind``).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/environment/test_lakeshore_multi_channel.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


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
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import types  # noqa: E402

import pytest  # noqa: E402

import mast.environment.autodetect as A  # noqa: E402
from mast.environment.lakeshore_temp import (  # noqa: E402
    ChannelState,
    LakeshoreSnapshot,
    SerialSettings,
)

_LSCI = types.SimpleNamespace(model="MODEL335", idn="LSCI,MODEL335,1,2",
                              settings=SerialSettings(port="COM13"))


@pytest.fixture()
def with_channels(monkeypatch):
    """Patch discover_lakeshore to report *channels*.

    Note every ``ChannelState`` below passes ``faults=`` explicitly. That is not
    noise: ``faults`` is three-state (``[]`` = RDGST? answered clean, non-empty
    = answered with faults, ``None`` = never answered), and omitting it now
    means the THIRD thing. Before the tri-state, omitting it silently meant
    "clean" — which is exactly how a channel whose status query timed out got
    registered as a healthy thermometer. A fixture that keeps saying nothing
    would keep testing a case these tests do not mean.
    """
    def _apply(channels):
        monkeypatch.setattr(
            "mast.environment.lakeshore_temp.discover_lakeshore",
            lambda port, **_kw: LakeshoreSnapshot(model="MODEL335",
                                                  channels=channels))
    return _apply


def _names(port: str = "COM13") -> list[str]:
    return [s.name() for s in A._lakeshore_sensors(port, _LSCI)]


# ════════════════════════════════════════════════════════════════════════════
# Both inputs, named the way the operator named them ON THE INSTRUMENT
# ════════════════════════════════════════════════════════════════════════════

def test_both_inputs_become_sensors_under_their_own_names(with_channels):
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=[])])
    assert _names() == ["SPM (COM13)", "Magnet (COM13)"]


def test_unnamed_inputs_fall_back_to_the_channel_letter(with_channels):
    """Still better than the model name repeated N times."""
    with_channels([ChannelState("A", "", 77.42, faults=[]),
                   ChannelState("B", "", 4.2, faults=[])])
    assert _names() == ["MODEL335 A (COM13)", "MODEL335 B (COM13)"]


def test_an_unwired_input_does_not_get_a_permanent_zero_row(with_channels):
    """A 335 answers +000.000E+00 on an empty input. Zero kelvin is not a
    measurement — 'not None' would have added a dead 0.00 K row."""
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "", 0.0, faults=[])])
    assert _names() == ["SPM (COM13)"]


def test_a_faulted_input_is_skipped(with_channels):
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.2, faults=["S.OVER"])])
    assert _names() == ["SPM (COM13)"]


# ════════════════════════════════════════════════════════════════════════════
# 「问过了，干净」vs「根本没问出来」—— 判据就是这一对能不能分开
#
# RDGST? 那一问超时的时候，pyserial 的 read_until 回 b"" 且不抛，_query 回 ""，
# 而 `int("" or 0)` = 0 是一个**完全合法的「干净」状态字**。于是一个传感器开路 /
# 过量程、状态位没答上来的通道，会被登记成一路健康的温度计，它的读数照常喂进
# campaign 换样品那道等待闸。
#
# 那道闸自己的 stale_after_s 拦不住：链路是活的、读数是新鲜的，缺的是**背书**。
# 新鲜度回答「这条消息旧不旧」，回答不了「这个数是不是编的」——两个问题。
# ════════════════════════════════════════════════════════════════════════════

def test_an_unconfirmed_input_is_not_registered_as_healthy(with_channels):
    """faults=None（RDGST? 没答上来）≠ faults=[]（问过，干净）。"""
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=None)])
    assert _names() == ["SPM (COM13)"], (
        "状态位没问出来的通道被当成健康温度计登记了 —— "
        "它的读数会去满足换样品等待闸")


def test_a_confirmed_clean_input_IS_registered(with_channels):
    """反向对照：同一个通道，唯一的差别是 RDGST? 答了「干净」。

    没有这一条，上面那条测试对着「把 B 一律拒掉」的实现也会通过 ——
    那种实现和修好了长得一模一样。
    """
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=[])])
    assert _names() == ["SPM (COM13)", "Magnet (COM13)"]


def test_health_is_the_one_definition_of_the_three_way_rule():
    """三态判据只有一处实现，调用方不各拼一套。"""
    from mast.environment.lakeshore_temp import (
        HEALTH_CLEAN,
        HEALTH_FAULTY,
        HEALTH_UNKNOWN,
    )

    assert ChannelState("A", faults=[]).health == HEALTH_CLEAN
    assert ChannelState("A", faults=["S.OVER"]).health == HEALTH_FAULTY
    assert ChannelState("A", faults=None).health == HEALTH_UNKNOWN
    # 缺省即「没问过」—— 构造时不说，就不算问过。
    assert ChannelState("A").health == HEALTH_UNKNOWN
    assert HEALTH_CLEAN != HEALTH_UNKNOWN


def test_all_unconfirmed_falls_back_and_does_not_claim_they_were_silent(
        with_channels, caplog):
    """全都没问出来时退回单通道，而且**日志不能说谎**。

    退化的两个理由把用户指向两个不同的地方：「一个通道都没报读数」是去查接线
    / 端口占用，「报了但状态位没背书」是去查那条链路为什么半哑。印错一个，
    人就会去修一个没坏的东西。
    """
    import logging

    caplog.set_level(logging.INFO, logger="mast.environment.autodetect")
    with_channels([ChannelState("A", "SPM", 77.42, faults=None),
                   ChannelState("B", "Magnet", 4.21, faults=None)])
    assert _names() == ["MODEL335 (COM13)"]

    text = caplog.text
    assert "no channel confirmed a healthy reading" in text
    assert "no channel reported a reading" not in text, (
        "报了读数却被说成没报 —— 证据回答的不是被问的那个问题")
    # 而且掉队的通道必须被点名，否则「行不见了」就是一次静默退化。
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "没有一条 WARNING —— 通道悄悄消失了"
    assert "RDGST?" in warnings[0].getMessage()
    assert "SPM" in text or "A=" in text


def test_a_genuinely_silent_port_still_says_so(with_channels, caplog):
    """反向对照：真的一个读数都没有时，说的还是原来那句话。"""
    import logging

    caplog.set_level(logging.INFO, logger="mast.environment.autodetect")
    with_channels([ChannelState("A", "", None, faults=[]),
                   ChannelState("B", "", None, faults=[])])
    assert _names() == ["MODEL335 (COM13)"]
    assert "no channel reported a reading" in caplog.text
    assert "no channel confirmed a healthy reading" not in caplog.text


def test_no_readable_channel_degrades_to_the_old_single_sensor(with_channels):
    """Never worse than before: a busy/silent port still yields one sensor."""
    # faults=[] on purpose: this test is about NO READING, so the health of the
    # (absent) reading must not be what makes it fall back — otherwise it would
    # pass for the wrong reason once the tri-state landed.
    with_channels([ChannelState("A", "", None, faults=[]),
                   ChannelState("B", "", None, faults=[])])
    assert _names() == ["MODEL335 (COM13)"]


def test_a_raising_snapshot_still_degrades_not_crashes(monkeypatch):
    def _boom(port, **_kw):
        raise OSError("port busy")
    monkeypatch.setattr("mast.environment.lakeshore_temp.discover_lakeshore", _boom)
    assert _names() == ["MODEL335 (COM13)"]


def test_the_sensors_actually_target_distinct_channels(with_channels):
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=[])])
    chans = [s._channel for s in A._lakeshore_sensors("COM13", _LSCI)]
    assert chans == ["A", "B"], "两个 sensor 指向同一个通道 = 同一个读数显示两遍"


# ════════════════════════════════════════════════════════════════════════════
# The rail must be able to tell "MODEL335 (COM13)" IS the temperature row
# ════════════════════════════════════════════════════════════════════════════

def test_sensor_entry_reports_its_headline_kind():
    """`type` is the driver class (diagnostics); `kind` is what the panel needs.
    Comparing the class name to "temperature" is what caused the duplication."""
    from mast.api.schemas_environment_readings import SensorEntry

    e = SensorEntry(name="SPM (COM13)", type="LakeshoreTemperatureSensor",
                    kind="temperature", value=77.42, unit="K",
                    status="ok", connected=True)
    assert e.kind == "temperature"
    assert e.type != e.kind, "type 和 kind 混为一谈就会再来一次重复显示"


def test_readings_endpoint_fills_kind_for_a_lakeshore():
    import types as _t

    from mast.api.routes import environment_readings as ER

    slot = ER._TYPE_TO_HEADLINE.get("LakeshoreTemperatureSensor")
    assert slot == "temperature", "驱动类→表头槽的映射断了"
    # And an unknown driver maps to nothing rather than to a wrong slot.
    assert ER._TYPE_TO_HEADLINE.get(type(_t.SimpleNamespace()).__name__) is None


# ════════════════════════════════════════════════════════════════════════════
# ...and both of them must actually READ.
#
# Two named rows is only half the job. The 环境历史 → 趋势 page for
# SPM (COM13) showed an empty chart, "unavailable · 1521 条读数未计入":
# every one of those 1521 readings came back `status="unavailable"`, and a
# reading with no value cannot enter a statistics bucket.
#
# The cause was one missing argument. `_build_one` (the CONFIGURED path) has
# passed a per-port shared transport since the beginning — its docstring even
# spells out why: "Windows opens COM ports EXCLUSIVELY, so letting each sensor
# lazily open its own handle means the first one wins and every other input is
# stuck reporting unavailable forever." `_lakeshore_sensors` (the AUTODETECT
# path) passed `settings=` but never `transport=`, so a controller MAST found
# by itself was the one configuration that could not read both of its inputs.
#
# The exclusivity fake below is the point of these tests: without it a unit
# test opens two fakes happily and the defect is invisible.
# ════════════════════════════════════════════════════════════════════════════

class _FakePort:
    """A transport over a port that only ONE handle may hold — like Windows."""

    _open_ports: set[str] = set()

    def __init__(self, settings):
        self.settings = settings
        self._held = False

    def _acquire(self):
        key = str(self.settings.port).upper()
        if self._held:
            return
        if key in _FakePort._open_ports:
            from mast.environment.serial_transport import SerialUnavailable
            raise SerialUnavailable(f"{key} 已被另一个句柄独占")
        _FakePort._open_ports.add(key)
        self._held = True

    def transact(self, payload, **_kw):
        self._acquire()
        # Answer whichever channel was asked for; the value doesn't matter, only
        # that a reply arrives at all.
        return b"+077.353E+00\r\n"

    def close(self):
        if self._held:
            _FakePort._open_ports.discard(str(self.settings.port).upper())
            self._held = False


@pytest.fixture()
def exclusive_ports(monkeypatch):
    """Make `open_transport` hand out fakes over an exclusive port."""
    _FakePort._open_ports.clear()
    monkeypatch.setattr("mast.environment.autodetect.open_transport", _FakePort)
    monkeypatch.setattr("mast.environment.lakeshore_temp.open_transport", _FakePort)
    yield
    _FakePort._open_ports.clear()


def test_both_autodetected_inputs_share_one_handle(with_channels, exclusive_ports):
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=[])])
    sensors = A._lakeshore_sensors("COM13", _LSCI, {})
    assert len(sensors) == 2
    a, b = sensors
    assert a._transport is b._transport is not None, (
        "两个 input 各开各的句柄 —— 拿不到端口的那个会永远 unavailable")


def test_both_autodetected_inputs_actually_read(with_channels, exclusive_ports):
    """验收就是这一条：两个通道都读得出值，才有读数能进统计桶。"""
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=[])])
    readings = [s.read() for s in A._lakeshore_sensors("COM13", _LSCI, {})]
    assert [r.status for r in readings] == ["ok", "ok"], (
        f"有通道读不到：{[(r.status, r.value) for r in readings]}")
    assert all(r.value > 0 for r in readings)


def test_the_exclusivity_fake_can_actually_fail(exclusive_ports):
    """把探针本身钉住。

    这个 fake 若不真的排他，上面两条测试对着**修复前**的代码也会通过 ——
    那种测试和全部通过长得一模一样。
    """
    from mast.environment.serial_transport import SerialSettings, SerialUnavailable

    s = SerialSettings(port="COM13")
    first = _FakePort(s)
    first.transact(b"*IDN?")
    with pytest.raises(SerialUnavailable):
        _FakePort(s).transact(b"*IDN?")
    first.close()
    _FakePort(s).transact(b"*IDN?")   # 释放之后又能开了


def test_unshared_sensors_would_have_failed(exclusive_ports):
    """反证：不共享 transport 时，第二个 input 就是 unavailable。

    这是修复前的实际行为，写成测试是为了让「共享」这个参数承重 —— 下一个人
    会重新想到「settings 传了就够了」（它看起来显然对）。
    """
    from mast.environment.lakeshore_temp import (
        LakeshoreTemperatureSensor,
        SerialSettings,
    )

    s = SerialSettings(port="COM13")
    a = LakeshoreTemperatureSensor(name="SPM", settings=s, channel="A")
    b = LakeshoreTemperatureSensor(name="Magnet", settings=s, channel="B")
    assert a.read().status == "ok"
    assert b.read().status == "unavailable"


def test_fallback_single_sensor_also_joins_the_pool(monkeypatch, exclusive_ports):
    """探测失败的退化路径同样要进池子，否则退化就是新的泄漏点。"""
    def _boom(port, **_kw):
        raise OSError("port busy")
    monkeypatch.setattr("mast.environment.lakeshore_temp.discover_lakeshore", _boom)
    pool: dict = {}
    sensors = A._lakeshore_sensors("COM13", _LSCI, pool)
    assert len(sensors) == 1
    assert "COM13" in pool
    assert sensors[0]._transport is pool["COM13"]


def test_config_and_autodetect_share_the_same_pool(with_channels, exclusive_ports,
                                                   monkeypatch):
    """一根线一个句柄，无论那个 sensor 是配置来的还是扫出来的。"""
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=[])])
    monkeypatch.setattr(A, "load_config", lambda: {
        "autodetect": True,
        "sensors": [{"id": "spm", "name": "SPM", "type": "lakeshore_temp",
                     "port": "COM13", "channel": "A"}],
    })
    monkeypatch.setattr(A, "list_serial_ports", lambda: [])
    pool: dict = {}
    configured, used = A.build_sensors_from_config(A.load_config(), pool)
    autod = A.autodetect_sensors(skip_ports=used, transports=pool)
    assert used == {"COM13"} and autod == []
    assert configured[0]._transport is pool["COM13"]


# ══ 停/起一轮之后共享仍然成立（/#15 复发，2026-08-05） ══════════════
#
# d17ff1b 修的是**建**的时候：自动探测出来的两个 input 现在共用一个句柄。
# 它没管**停**的时候：`close()` 里那句 `self._transport = None` 把共享对象
# 整个丢掉，于是下一次读会从 `_settings` 各开各的私有句柄，反相交替原样回来。
#
# 触发路径一点也不罕见：用户按一次「扫描设备接口」——
# `/api/environment/discover` 会 `_release_bus`（→ stop → _close_sensors →
# close）然后用**同一批 sensor 对象** restart。rescan / adopt 之所以没事，
# 只是因为它们会带一个新池子重建 sensor 列表。


def test_sharing_survives_a_stop_start_cycle(with_channels, exclusive_ports):
    """停一轮再起来，两个 input 必须仍然共用同一个 transport 并且都读得出。"""
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=[])])
    a, b = A._lakeshore_sensors("COM13", _LSCI, {})
    assert [s.read().status for s in (a, b)] == ["ok", "ok"]

    for s in (a, b):          # EnvironmentMonitor._close_sensors 干的事
        s.close()

    assert a._transport is b._transport is not None, (
        "close() 丢掉了共享 transport —— 下一次读会各开各的私有句柄")
    assert [s.read().status for s in (a, b)] == ["ok", "ok"], (
        "停/起一轮之后有通道读不到了，反相交替就是这么回来的")


def test_close_releases_the_os_handle_it_was_holding(with_channels, exclusive_ports):
    """反证：close() 仍然必须**真的**放掉端口。

    只留着对象而不关句柄，就变成另一个方向的错 —— 别的进程
    （另一个串口程序、探测代码）永远拿不到 COM13。
    """
    from mast.environment.serial_transport import SerialSettings

    with_channels([ChannelState("A", "SPM", 77.42, faults=[])])
    a = A._lakeshore_sensors("COM13", _LSCI, {})[0]
    assert a.read().status == "ok"
    assert "COM13" in _FakePort._open_ports
    a.close()
    assert "COM13" not in _FakePort._open_ports, "端口没被放掉"
    # 放掉之后别人能开，而且我们自己还能再开回来。
    other = _FakePort(SerialSettings(port="COM13"))
    other.transact(b"*IDN?")
    other.close()
    assert a.read().status == "ok"


def test_close_and_the_read_failure_path_agree(with_channels, exclusive_ports):
    """两条路径对「共享对象要不要留」必须给同一个答案。

    `read()` 的失败分支早就写对了（`_release_transport`，还带一段注释解释
    为什么不能丢）。`close()` 写的是相反的做法，而没有任何东西让这两者对账 ——
    于是同一个决定在同一个类里有两种实现，只有一种是对的。
    """
    with_channels([ChannelState("A", "SPM", 77.42, faults=[]),
                   ChannelState("B", "Magnet", 4.21, faults=[])])
    a, b = A._lakeshore_sensors("COM13", _LSCI, {})
    shared = a._transport
    a._release_transport()
    assert a._transport is shared
    a.close()
    assert a._transport is shared
    assert b._transport is shared


def test_a_pooled_dl7_can_still_name_its_port(exclusive_ports):
    """d17ff1b 顺手引入的第二个洞：池化的 DL-7 拿不回自己的端口。

    `transport is None and port is not None` 这个条件让每一个池化的真空计
    `_settings = None`，于是一旦 transport 被丢掉就永远
    「no port configured」。而 d17ff1b 做的正是把自动探测到的 DL-7 池化。
    """
    from mast.environment.dl7_vacuum import DL7VacuumSensor
    from mast.environment.serial_transport import SerialSettings

    s = SerialSettings(port="COM17")
    g = DL7VacuumSensor(name="真空计", port="COM17", transport=_FakePort(s))
    assert g._settings is not None, "池化的 gauge 说不出自己在哪个口上"
    assert g._settings.port == "COM17"
    g.close()
    # 关了之后仍然要能重新开 —— 无论是靠留下来的对象还是靠 settings。
    assert g._transport is not None or g._settings is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
