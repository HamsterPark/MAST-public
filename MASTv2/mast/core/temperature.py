"""温度公共只读口 —— 一个读数，外加「读不到」的**几种不同答案**。

在此之前，温度在 v2 里只有一个私有取数口（``CoreRuntime._latest_temperature_k``），
它的返回类型是 ``float | None``，而那个 ``None`` 同时表示：

  * 这台机器根本没装温度计（插什么都不会好）；
  * 已安装，但此刻读不到，例如端口被其他程序占用；
  * 有读数，但已经很旧了（还在降温的实验按它判「到温了」就会判错）。

这三件事对**调用方要做什么**的指示完全相反：第一种要立刻拒绝（等下去永远等不到），
第二种要去看端口占用，第三种要再等一拍。把它们折叠成同一个 ``None``，等待条件就会
悄悄变成「永远等」——「读不到 ≠ 出故障 ≠ 零 ≠ 干净」在本仓库已经反复咬人。

所以这里给出的是 :class:`TempReading`：**值 + 年龄 + 出处 + 通道 + 为什么没有值**。

## 年龄由调用方判陈旧，不由这里判

``age_s`` 随值一起返回，:meth:`TempReading.freshness` 把「多旧算旧」这个**只有调用方
知道**的阈值留给调用方。它返回三态字符串而不是 ``bool``：``bool | None`` 的返回类型
会被 ``if not r.is_stale(60)`` 一句悄悄把「不知道」当成「新鲜」——那正是要防的错。

``age_s`` 读不出来时是 ``None``，**绝不是 0.0**。伪造一个 0 秒的年龄会让任何
``age_s <= limit`` 的判据无条件通过，把「不知道多旧」变成「刚刚读的」。

## 分层

本模块**不 import** ``mast.environment``：分类（哪个通道是温度、哪个是占位实现）
由持有传感器对象的 ``CoreRuntime`` 做完，以 :class:`TempChannel` 记录传进来。
技能层则通过下面的进程级注入口取数（形状与 ``core.vacuum_interlock`` 的压强源、
``core.coarse_map_provider`` 的温度源一致），因此 ``mast/skills/**`` 不必也不能
去 import 活的 app 对象。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# 「没有值」的几种原因 —— 刻意不折叠
# ─────────────────────────────────────────────────────────────────────
#: 这台机器上没有真的温度传感器（只有占位实现，或一个温度通道都没有）。
#: **等下去永远等不到**：调用方应当拒绝，而不是重试或等待。
NO_SENSOR = "no_sensor"

#: 有真的温度传感器，但此刻拿不到读数（串口被别的程序占着、读失败、
#: 或归档循环还没跑过一拍所以缓存是空的）。**可能会好**：值得再等/去看端口。
UNAVAILABLE = "unavailable"

#: 有读数，但太旧了。**本模块永不产出它** —— 多旧算旧只有调用方知道
#: （见 :meth:`TempReading.freshness`）。列在这里是为了让「陈旧」这个概念
#: 有且只有一个名字，而不是每个消费方各拼一个字符串。
STALE = "stale"

#: 指名要的那个通道这台机器上没有。**不是**「没有温度计」——
#: 折叠成 ``no_sensor`` 会让用户去找一个并不缺的传感器。
UNKNOWN_CHANNEL = "unknown_channel"

#: 指名要的名字同时对上了不止一个通道。同样不是「没有」，是「你得说清是哪个」。
AMBIGUOUS_CHANNEL = "ambiguous_channel"

#: MAST **自己这一侧**没接上：进程里没有注入温度源（独立 API 进程、
#: 环境监控没建起来、或注入的源抛了）。与仪器无关，重启/接线才会变。
NO_SOURCE = "no_source"

REASONS = (NO_SENSOR, UNAVAILABLE, STALE, UNKNOWN_CHANNEL,
           AMBIGUOUS_CHANNEL, NO_SOURCE)

# ── 一个**刻意不在**这个闭集里的态：「读数的故障位没问出来」（2026-08-15）──
#
# Lake Shore 的 ``RDGST?`` 回答的是「这个数可不可信」。那一问超时的时候，
# pyserial 的 ``read_until`` 回 ``b""`` 且**不抛**，``_query`` 回 ``""``，而
# ``int("" or 0)`` = 0 是一个完全合法的「干净」状态字 —— 于是一个传感器开路 /
# 过量程、状态位没答上来的通道，会被登记成一路健康的温度计，读数照常流到这里，
# 再去满足 conduct 换样品那道等待闸。那道闸的 ``stale_after_s`` 拦不住它：链路
# 是活的、读数是新鲜的，缺的是**背书**。新鲜度回答「这条消息旧不旧」，回答不了
# 「这个数是不是编的」——两个问题。
#
# 修法在上游：``environment.lakeshore_temp.ChannelState.faults`` 现在是三态
# （``[]`` 问过且干净 / ``[...]`` 问过有故障 / ``None`` 没问出来），而
# ``environment.autodetect._lakeshore_sensors`` 只把 ``HEALTH_CLEAN`` 的通道登记
# 成传感器。
#
# **这里不加对应的 reason，因为这一层拿不到那个信息**：``TempChannel`` 由
# ``core.runtime.temperature_channels()`` 从 ``SensorReading``（值/单位/状态/
# 时间戳）加 ``is_real_sensor`` 拼出来，而轮询循环走的是
# ``LakeshoreTemperatureSensor.read()`` —— 它只发 ``KRDG?``，从不发 ``RDGST?``。
# 加一个没有任何生产方的返回态，等于给下一个人留一条永远不会亮的分支。
#
# 真要让它到得了这一层，得同时动三处（少一处就是「消费方接好了、生产方没有」）：
#   1. ``LakeshoreTemperatureSensor.read()`` 连带查 ``RDGST?``，把结论挂在读数上；
#   2. ``TempChannel`` 加一个承载它的字段；
#   3. ``core.runtime.temperature_channels()`` 填这个字段。
# 三处齐了，再来把常量加进 ``REASONS``。
#
# 还剩一个已知未修的口子：**配置来的**通道不过 autodetect 那道筛子
# （``environment.autodetect._build_one`` 压根不看 faults），所以在
# ``environment_sensors.json`` 里钉死的 Lake Shore 通道，状态位如何都会照常喂进来。

#: 多个温度通道都可读时**优先样品台**：磁体杜瓦的温度不等于样品台的温度。
#: 与 _latest_temperature_k 使用相同偏好，不依赖特定串口或设备实例名称。
_STAGE_HINTS = ("spm", "stage", "sample", "tip", "cryo")

_KELVIN_UNITS = ("k", "kelvin")
_CELSIUS_UNITS = ("c", "°c", "degc", "celsius")
_TEMPERATURE_UNITS = _KELVIN_UNITS + _CELSIUS_UNITS

#: 哪些 status 算「这一拍真的读到了一个数」。
#:
#: ``warning`` / ``alarm`` **在内**，这是一处修正：它们表示值越过了报警带，
#: 不表示读不到。原来的 ``status != "ok" -> None`` 会让一台正在从 300 K 降下来的
#: 机器（降温途中必然长时间落在 warn 带里）对「现在几度」一路回答「读不到」——
#: 而这恰恰是等降温的实验唯一要看的那个数。
#:
#: ``error`` / ``unavailable`` 在外，且必须在外：``EnvironmentMonitor.read_all``
#: 在读失败时写的是 ``SensorReading(value=0.0, unit="", status="error")``——
#: 把它当成读数就是把 **0 K** 记进实验记录。
_LIVE_STATUSES = ("ok", "warning", "alarm")


def to_kelvin(value, unit, status: str = "ok") -> "float | None":
    """按值、单位与状态换算开尔文；无有效温度时返回 None。
    
    识别依据单位而非名称，不能把电流等其他物理量记成温度。"""
    if str(status or "").strip().lower() not in _LIVE_STATUSES:
        return None
    u = str(unit or "").strip().lower()
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if u in _KELVIN_UNITS:
        return v
    if u in _CELSIUS_UNITS:
        return v + 273.15
    return None


def is_temperature_unit(unit) -> bool:
    """这个单位是温度单位吗（K / °C 的各种写法）。"""
    return str(unit or "").strip().lower() in _TEMPERATURE_UNITS


# ─────────────────────────────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TempChannel:
    """一个**温度型**通道，按持有传感器对象的那一层分类完之后的样子。

    ``real`` 是三态，这很重要：

      * ``True``  —— 真驱动（Lake Shore 之类）。
      * ``False`` —— 占位实现（``environment/placeholders.py``）：MAST 在这台机器上
        **没有**这个量的驱动，插什么都不会好 ⇒ ``no_sensor``。
      * ``None``  —— **不知道**（拿不到传感器对象，比如只有一份读数快照）。
        不知道**不等于**占位：把它当占位会让一台正常报着 77 K 的机器被判成
        「没装温度计」。
    """

    name: str
    value: "float | None" = None
    unit: str = ""
    status: str = ""
    timestamp: str = ""
    #: 驱动类名，读数的出处。占位与真驱动今天都报 ``unavailable``，只有出处分得开。
    driver: str = ""
    real: "bool | None" = None

    def kelvin(self) -> "float | None":
        return to_kelvin(self.value, self.unit, self.status)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "value": self.value, "unit": self.unit,
            "status": self.status, "driver": self.driver, "real": self.real,
            "value_k": self.kelvin(),
        }


@dataclass(frozen=True)
class TempReading:
    """一次温度查询的完整答案。

    ``value_k is None`` 时 ``reason`` 一定不是 ``None``，反之亦然 —— 不存在
    「没有值也没有原因」的返回。
    """

    #: 开尔文。``None`` = 没有值，看 ``reason``。
    value_k: "float | None" = None
    #: 这个读数有多旧（秒）。``None`` = **不知道多旧**，不是 0。
    age_s: "float | None" = None
    #: 出处（驱动类名）。没有值时是 ``""`` —— 没人给出过读数。
    source: str = ""
    #: 落到哪个通道上。没指名且一个通道都没有时是 ``""``；
    #: 指名了但对不上时，回显的是**你要的那个名字**。
    channel: str = ""
    #: 为什么没有值，取自本模块的常量。有值时是 ``None``。
    reason: "str | None" = None

    def freshness(self, max_age_s: float) -> str:
        """``"fresh"`` / ``"stale"`` / ``"unknown"`` —— **三态，不是 bool**。

        ``bool`` 会被 ``if not stale:`` 一句把「不知道多旧」变成「新鲜」，而
        「不知道」恰恰是最该停下来的那种答案。年龄不明或压根没有值时是
        ``"unknown"``。
        """
        if self.value_k is None or self.age_s is None:
            return "unknown"
        try:
            return "stale" if float(self.age_s) > float(max_age_s) else "fresh"
        except (TypeError, ValueError):
            return "unknown"

    def as_dict(self) -> dict:
        return {
            "value_k": self.value_k,
            "age_s": self.age_s,
            "source": self.source,
            "channel": self.channel,
            "reason": self.reason,
        }


# ─────────────────────────────────────────────────────────────────────
# 选择：通道集合 → 一个答案（纯函数，可注入时钟）
# ─────────────────────────────────────────────────────────────────────
def age_s(timestamp, now: "datetime | None" = None) -> "float | None":
    """ISO 时间戳 → 距 ``now`` 多少秒。解析不出来就 ``None``（**不是 0**）。

    ``now`` 是注入口：测试给一个固定时刻，就不必让断言去追真实时钟。
    """
    ts = str(timestamp or "").strip()
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    ref = now if isinstance(now, datetime) else datetime.now()
    # 一边带时区一边不带就没法相减 —— 统一到「本地朴素时间」。
    if (parsed.tzinfo is None) != (ref.tzinfo is None):
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        else:
            ref = ref.astimezone().replace(tzinfo=None)
    try:
        return (ref - parsed).total_seconds()
    except (TypeError, ValueError, OverflowError):  # pragma: no cover - defensive
        return None


def _prefer_stage(cands: "list[TempChannel]") -> TempChannel:
    for c in cands:
        if any(h in c.name.lower() for h in _STAGE_HINTS):
            return c
    return cands[0]


def _match(chans: "list[TempChannel]", requested: str):
    """返回命中与落空原因：名称先精确匹配，再忽略大小写，最后尝试唯一子串。
    
    子串匹配用于名称带额外描述的通道；多个命中必须要求明确选择，不能静默挑一个。"""
    req = requested.strip()
    for pred in (lambda c: c.name == req,
                 lambda c: c.name.casefold() == req.casefold()):
        hits = [c for c in chans if pred(c)]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, AMBIGUOUS_CHANNEL
    fold = req.casefold()
    hits = [c for c in chans if fold in c.name.casefold()]
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        return None, AMBIGUOUS_CHANNEL
    return None, UNKNOWN_CHANNEL


def _read_one(c: TempChannel, now: "datetime | None") -> TempReading:
    k = c.kelvin()
    age = age_s(c.timestamp, now)
    if k is None:
        return TempReading(
            value_k=None, age_s=age, source="", channel=c.name,
            reason=NO_SENSOR if c.real is False else UNAVAILABLE,
        )
    return TempReading(value_k=k, age_s=age, source=c.driver,
                       channel=c.name, reason=None)


def read_temperature(
    channels: "Sequence[TempChannel] | None",
    *,
    channel: "str | None" = None,
    now: "datetime | None" = None,
) -> TempReading:
    """从一组温度通道里给出一个答案。纯函数、永不抛。

    没指名 ``channel`` 时：占位通道不参选 → 可读的优先 → 可读的里面样品台优先。
    一个都读不到时**仍然回报那个本该用的通道名和它的年龄**，因为「哪个温度计哑了」
    才是用户能动手的那条信息。
    """
    chans = [c for c in (channels or []) if isinstance(c, TempChannel)]
    req = str(channel or "").strip()

    if not chans:
        # 一个温度通道都没有。此时即使指名了也该说「没有温度计」，
        # 而不是「没有这个通道」—— 后者会让人去找一个不存在的配置项。
        return TempReading(channel=req, reason=NO_SENSOR)

    if req:
        hit, why = _match(chans, req)
        if hit is None:
            return TempReading(channel=req, reason=why)
        return _read_one(hit, now)

    candidates = [c for c in chans if c.real is not False]
    if not candidates:
        # 全是占位实现：这台机器没有温度计驱动，等下去永远等不到。
        return TempReading(channel="", reason=NO_SENSOR)

    readable = [c for c in candidates if c.kelvin() is not None]
    return _read_one(_prefer_stage(readable or candidates), now)


# ─────────────────────────────────────────────────────────────────────
# 进程级注入口（技能层从这里取数，不 import 活的 app 对象）
# ─────────────────────────────────────────────────────────────────────
_lock = threading.RLock()
_source: "Callable[[str | None], TempReading] | None" = None
_channels_source: "Callable[[], Sequence[TempChannel]] | None" = None


def set_source(source: "Callable[[str | None], TempReading] | None") -> None:
    """装上 ``(channel) -> TempReading``（运行时启动时注入）。"""
    global _source
    with _lock:
        _source = source


def set_channels_source(
        source: "Callable[[], Sequence[TempChannel]] | None") -> None:
    """装上 ``() -> [TempChannel]``：这台机器上有哪些温度通道。"""
    global _channels_source
    with _lock:
        _channels_source = source


def latest_temperature(channel: "str | None" = None) -> TempReading:
    """当前温度。**永不抛** —— 没接上/源出错都回 ``no_source``。

    ``no_source`` 与 ``unavailable`` 分开：前者说的是 MAST 自己没接上（重启、
    换进程才会变），后者说的是仪器此刻不给数（关掉占端口的程序就好）。
    """
    src = _source
    if src is None:
        return TempReading(channel=str(channel or ""), reason=NO_SOURCE)
    try:
        out = src(channel)
    except Exception as exc:  # noqa: BLE001 — 问温度绝不该把调用方搞崩
        logger.debug("temperature source failed: %r", exc)
        return TempReading(channel=str(channel or ""), reason=NO_SOURCE)
    if not isinstance(out, TempReading):
        logger.debug("temperature source returned %r, not a TempReading", type(out))
        return TempReading(channel=str(channel or ""), reason=NO_SOURCE)
    return out


def channels() -> "list[TempChannel] | None":
    """这台机器上的温度通道；``None`` = **问不到**（没接上/源出错）。

    ``None`` 和 ``[]`` 刻意不同：``[]`` 是「确实一个都没有」，
    ``None`` 是「不知道有没有」。
    """
    src = _channels_source
    if src is None:
        return None
    try:
        out = src()
    except Exception as exc:  # noqa: BLE001
        logger.debug("temperature channels source failed: %r", exc)
        return None
    if out is None:
        return None
    return [c for c in out if isinstance(c, TempChannel)]


__all__ = [
    "TempChannel",
    "TempReading",
    "NO_SENSOR",
    "UNAVAILABLE",
    "STALE",
    "UNKNOWN_CHANNEL",
    "AMBIGUOUS_CHANNEL",
    "NO_SOURCE",
    "REASONS",
    "to_kelvin",
    "is_temperature_unit",
    "age_s",
    "read_temperature",
    "set_source",
    "set_channels_source",
    "latest_temperature",
    "channels",
]
