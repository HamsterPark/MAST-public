"""把 Nanonis 侧的量包成环境传感器 —— 让它们走同一条归档链路。

有两种，用途完全不同：

:class:`InstrumentStateSensor`
    **零 TCP。** 镜像 :class:`mast.core.state.InstrumentState` 那个 1 Hz 后台
    刷新已经拿到的值（电流、偏压、Z）。要记录的正是"针尖长时间恒流隧穿停留下的
    current" —— 而那个值**每秒都已经在被读了**，再开一条自己的轮询只会
    往一个已经满载的 TCP 角色上再加流量。

:class:`NanonisSignalSensor`
    走 128 路信号里的一路（``Signals.ValGet``）。给那些"表接在 Nanonis 的模拟
    输入上"的量用 —— 磁场、液氦液面如果是这种接法，这里就是它们的入口，
    不需要写任何新驱动。

两者都实现 :class:`~mast.environment.base.EnvironmentSensor`，所以
:class:`~mast.environment.monitor.EnvironmentMonitor` 一视同仁：告警、
``environment_log``、CSV 双写、历史统计桶全部自动获得。

两个属性约定
------------

``counts_as_real``
    :func:`mast.environment.autodetect.has_real_sensors` 认这个属性。没有它,
    一台只有 Nanonis 源、没有任何串口表的机器会被判成"没有真传感器",
    后台监控循环根本不启动 —— 于是连温度以外的一切都记不到。

``quiet_gated``
    这条序列只在仪器安静时才有意义（见 :mod:`mast.envhistory.quiet`）。
    扫描进行中的隧道电流是形貌信号，混进"隧道电流趋势"里会毁掉整条曲线。
    历史 sink 读这个属性决定要不要给它上安静门；实时读数与告警**不受影响**,
    那些路径要的就是"此刻的电流是多少"。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from mast.core.types import SensorReading
from mast.environment.base import EnvironmentSensor

logger = logging.getLogger(__name__)

#: 与电流监控同一个角色。``monitor``(6502) 已被撞针看门狗 + InstrumentState +
#: 扫描监控占到 ~20 事务/秒（预算写在 frontend/src/lib/pollRates.ts），不能再加。
DATA_ROLE = "data"


def _first_value(record) -> float:
    """从一次 Nanonis 调用记录里取出第一个解析值。

    nanonis_spm 返回 ``(error_string, raw_bytes, parsed_list)``；真正的读数在
    ``parsed_list[0]``。老代码读的是 ``return_value[0]`` —— 那是**空的错误串**,
    于是 ``float("")`` 每次读都抛 ValueError。
    """
    rv = record.return_value
    if isinstance(rv, (list, tuple)) and len(rv) > 2:
        parsed = rv[2]
        if isinstance(parsed, (list, tuple)) and parsed:
            v = parsed[0]
            if isinstance(v, (list, tuple)):
                return float(v[0]) if v else 0.0
            return float(v)
        return 0.0
    if isinstance(rv, (int, float)):
        return float(rv)
    return 0.0


class InstrumentStateSensor(EnvironmentSensor):
    """把 ``InstrumentState`` 缓存里的一个字段当成环境传感器。**零 TCP。**

    ``stale`` 的快照报 ``unavailable`` 而不是把上一个已知值再报一遍：一条
    "针尖停在 20 pA"的曲线在 TCP 断掉之后继续平直地画下去，比曲线断掉危险得多。
    """

    #: 只有 Nanonis 源的机器也要能启动后台监控循环。
    counts_as_real = True

    def __init__(self, state_getter: Callable[[], Any], *,
                 name: str = "tunnel_current", attr: str = "current_a",
                 unit: str = "A", quiet_gated: bool = True) -> None:
        self._state_getter = state_getter
        self._name = str(name)
        self._attr = str(attr)
        self._unit = str(unit)
        self.quiet_gated = bool(quiet_gated)

    def name(self) -> str:
        return self._name

    def read(self) -> SensorReading:
        try:
            state = self._state_getter() if callable(self._state_getter) else self._state_getter
            snap = state.snapshot() if state is not None else None
        except Exception:  # noqa: BLE001 — 读缓存都失败就是没有读数
            snap = None
        if snap is None:
            return SensorReading(value=0.0, unit=self._unit, status="unavailable")
        if getattr(snap, "stale", False):
            return SensorReading(value=0.0, unit=self._unit, status="unavailable")
        val = getattr(snap, self._attr, None)
        if val is None:
            return SensorReading(value=0.0, unit=self._unit, status="unavailable")
        try:
            return SensorReading(value=float(val), unit=self._unit, status="ok")
        except (TypeError, ValueError):
            # 保留转换失败的原值供诊断，不能只留下占位的 0.0。
            # 数值数组可能包含单元素元组，float((x,)) 会抛 TypeError；
            # 错误记录需要足以区分响应形状错误与仪器故障。
            logger.warning(
                "环境传感器 %s 的值转不成数,报 error(**不是**环境故障): "
                "value=%r type=%s —— 值本身留在这里,便于判断是不是元组包装。",
                self._name, val, type(val).__name__,
            )
            return SensorReading(value=0.0, unit=self._unit, status="error")

    def is_healthy(self) -> bool:
        return self.read().status in ("ok", "warning")


class NanonisSignalSensor(EnvironmentSensor):
    """读 128 路信号里的一路。给磁场 / 液氦液面这类"接在模拟输入上"的表用。

    与仓库里所有高频只读轮询同一套纪律：

    * ``role="data"`` —— ``monitor`` 口的 20 事务/秒预算里有撞针看门狗，不能挤；
    * ``count_health=False`` —— 熔断器是四个 role 共用的一个实例，它的
      ``record_success`` 无条件清空失败连击。一个稳定成功的后台轮询器会让
      "连续三次失败"永远攒不满，等于把全局熔断器废掉；
    * 熔断开路时**零 TCP** 直接报 unavailable —— 往开着的熔断器里重试正是它
      存在的意义所在；
    * ``RoleBusy`` 是**正常事件**不是错误：报 unavailable，下一拍再来，不重试。
    """

    counts_as_real = True

    def __init__(self, pool_getter: Callable[[], Any], *, name: str,
                 signal_name: str | None = None, signal_index: int | None = None,
                 unit: str = "", quiet_gated: bool = False) -> None:
        if signal_name is None and signal_index is None:
            raise ValueError("NanonisSignalSensor 需要 signal_name 或 signal_index")
        self._pool_getter = pool_getter
        self._name = str(name)
        self._signal_name = (signal_name or "").strip().lower() or None
        self._index: int | None = int(signal_index) if signal_index is not None else None
        self._unit = str(unit)
        self.quiet_gated = bool(quiet_gated)
        self._last_status = "unavailable"
        self._resolve_failed_at = 0.0

    def name(self) -> str:
        return self._name

    def read(self) -> SensorReading:
        pool = self._pool()
        if pool is None:
            return self._out(0.0, "unavailable")
        try:
            if not pool.comms_healthy():
                return self._out(0.0, "unavailable")
        except Exception:  # noqa: BLE001
            return self._out(0.0, "unavailable")

        idx = self._resolve_index(pool)
        if idx is None or idx < 0:
            return self._out(0.0, "unavailable")
        try:
            rec = pool.safe_call("Signals_ValGet", int(idx), 1,
                                 role=DATA_ROLE, count_health=False)
        except Exception:  # noqa: BLE001
            return self._out(0.0, "error")
        if getattr(rec, "error", ""):
            from mast.core.connection import is_lock_busy
            # 角色锁被占：有人正在用 data 口（扫描抓帧 / 电流监控）。这不是故障。
            return self._out(0.0, "unavailable" if is_lock_busy(rec) else "error")
        try:
            return self._out(_first_value(rec), "ok")
        except (TypeError, ValueError):
            return self._out(0.0, "error")

    def is_healthy(self) -> bool:
        """报上一次读到的状态，**不额外发一次 TCP。**

        健康检查是给面板用的，它不该在一个共享的 TCP 角色上凭空多出一轮流量。
        """
        return self._last_status in ("ok", "warning")

    # ── 内部 ──────────────────────────────────────────────────────────

    def _resolve_index(self, pool) -> int | None:
        """按名字在 128 路信号里找索引，解析一次就缓存。

        解析失败后 60 秒内不再重试：信号名列表不会自己变，而每次失败都是一次
        白花的 ``Signals_NamesGet``（那是个 128 条字符串的回包，不便宜）。
        """
        if self._index is not None:
            return self._index
        if (time.monotonic() - self._resolve_failed_at) < 60.0:
            return None
        try:
            rec = pool.safe_call("Signals_NamesGet", role=DATA_ROLE, count_health=False)
        except Exception:  # noqa: BLE001
            self._resolve_failed_at = time.monotonic()
            return None
        if getattr(rec, "error", ""):
            self._resolve_failed_at = time.monotonic()
            return None
        names: list[str] = []
        rv = rec.return_value
        parsed = rv[2] if isinstance(rv, (list, tuple)) and len(rv) > 2 else []
        for field_val in parsed or []:
            if isinstance(field_val, (list, tuple)) and field_val and isinstance(
                    field_val[0], (str, bytes)):
                names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in field_val]
                break
        want = self._signal_name or ""
        for i, nm in enumerate(names):
            if want and want in nm.lower():
                self._index = i
                logger.info("环境传感器 %s 绑定到 Nanonis 信号 #%d (%s)",
                            self._name, i, nm)
                return i
        self._resolve_failed_at = time.monotonic()
        logger.warning("环境传感器 %s：128 路信号里找不到 %r", self._name, want)
        return None

    def _pool(self):
        try:
            return self._pool_getter() if callable(self._pool_getter) else self._pool_getter
        except Exception:  # noqa: BLE001
            return None

    def _out(self, value: float, status: str) -> SensorReading:
        self._last_status = status
        return SensorReading(value=float(value), unit=self._unit, status=status)


__all__ = ["InstrumentStateSensor", "NanonisSignalSensor", "DATA_ROLE"]
