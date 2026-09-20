"""Director 与外界之间的**注入口**(以及它们的诚实替身)。

设计:``campaign_director_design.md`` §3-1/2/7、§9(单测策略:假 executor、
时钟注入、替身回包要模拟真实形状)。

## 为什么全部注入

Director 是一条常驻线程,它要动仪器、读急停闩、读温度、发通知。这四样在测试里
都不能真做,而在生产里又必须是**那一个**实现(不是复制一份)。所以这里只定义
形状,实现由 M1-c 在 runtime 挂点上接进来。

## 替身的诚实

本模块提供的替身都是**「什么都没有」而不是「一切正常」**:

* :class:`NullLatch` 报「闩没挂」——这是真的,因为它压根不知道有没有闩;
* :class:`NullTemperature` 报 ``value_k=None`` + 一个 reason,**不报 0 K**;
* :class:`LoggingNotifier` 把通知写进日志并**留下记录**,不是丢掉。

一个「一切正常」的替身会让没接线的部署看起来运行良好 —— 那正是本仓
「producer wired, consumer absent」那族缺陷的温床。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── 执行一步的结果 ──────────────────────────────────────────────────

@dataclass(frozen=True)
class StepOutcome:
    """一步跑完之后 Director 需要知道的全部。

    ``busy`` 与 ``ok=False`` **分开**:仪器被别人占着不是这一步失败 ——
    前者下一 tick 重试,后者进重试/失败策略。把两者折叠会让一次正常的并发
    仲裁消耗掉这一步的重试预算。
    """

    ok: bool
    #: 被仪器令牌拒了(``InstrumentBusy``)。拒绝不排队是既有纪律。
    busy: bool = False
    #: 针尖事件 / 撞针 —— 由 E_STOP 闩接管,这里只是把它带出来记账。
    tip_event: bool = False
    crash: bool = False
    error: str = ""
    #: 技能返回的 data(用来登记 produces)。
    data: dict = field(default_factory=dict)
    #: 这一步的 run_id(Director 生成后传进来,原样带回便于对账)。
    run_id: str = ""


class ExecutorPort:
    """跑一个技能。生产实现包 ``core.executor.SkillExecutor.run``。

    **Director 永远没有裸 TCP 权**(§3-1):每一步都走 ``executor.run`` 的完整
    安全管道(registry → 状态刷新 → SafetyGuard → 快照)。watchdog 才是那条
    有权走裸 ``urgent_call`` 的紧急救济通道 —— 两者的区别写在这里,是因为
    「反正都是代码驱动仪器」这句话会把它们混成一个。
    """

    def run(self, skill: str, params: dict, *, run_id: str) -> StepOutcome:
        raise NotImplementedError


# ── 急停闩 ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LatchState:
    """闩挂没挂 + **为什么挂的**。

    两者必须一起给。一个只说 True 的接口,读的人只能自己编原因 —— 而那正是
    曾经发生过的锁死的伤害所在(闩挂上了,「因为什么」在通知通道里被静默
    丢掉,于是没人知道要解什么)。
    """

    latched: bool = False
    abort_set: bool = False
    why: str = ""


class NullLatch:
    """没接闩时的替身:如实报「我不知道有没有闩,按没挂处理」。

    为什么不报「挂着」:那会让任何没接线的部署一启动就停在 HALTED_ESTOP,而且
    **解不开**(没有闩可以解)。选择放行 + 由 M1-c 负责接线,是两害相权。
    """

    def state(self) -> LatchState:
        return LatchState()


# ── 温度(修复项 的公共只读口)────────────────────────────────────────

class NullTemperature:
    """没接温度口时的替身。

    返回 ``value_k=None`` + reason —— **不返回 0 K**。0 K 是一个会让
    ``<= 5 K`` 这类条件立刻成立的数,而我们其实什么都没读到。
    """

    def read(self, channel: str = ""):
        from mast.core.temperature import NO_SOURCE, TempReading
        return TempReading(channel=channel, reason=NO_SOURCE)


# ── 通知 / 心愿单 ───────────────────────────────────────────────────

@dataclass
class Notification:
    kind: str
    conduct_id: str
    message: str
    severity: str = "info"
    payload: dict = field(default_factory=dict)


class LoggingNotifier:
    """M1-b 的通知替身:写日志 **并留下记录**。

    真接线(心愿单 ``post_agent_request`` + EventBus)是 M2。留记录是为了让
    「通知到底发出去没有」在测试和排障里可问 —— 一个只 ``pass`` 的替身会让
    「通知链路没接」和「通知发了没人看」长得一模一样。
    """

    def __init__(self) -> None:
        self.sent: list[Notification] = []

    def notify(self, note: Notification) -> None:
        self.sent.append(note)
        logger.info("[conduct %s] %s: %s", note.conduct_id, note.kind,
                    note.message)

    def request_operator_action(self, note: Notification) -> str:
        """发一条要人办的事(生产实现进心愿单)。返回外部请求 id,没有就空串。"""
        self.notify(note)
        return ""


# ── per-run abort Event ─────────────────────────────────────────────

class AbortRegistry:
    """per-run abort Event 的注册表。

    生命周期(§6):注册=步启动前;注销=``executor.run`` 返回后的 ``finally``
    (成败都清)。生产实现挂到 ``runtime`` 的 per-run Event 字典上,好让 API
    线程的 abort **不经 tick** 就能置位 —— Director 卡在一次长步里时 tick 不会
    来,而 abort 按钮必须还能用。
    """

    def __init__(self) -> None:
        self._events: dict[str, Any] = {}

    def register(self, run_id: str):
        import threading
        ev = threading.Event()
        self._events[run_id] = ev
        return ev

    def unregister(self, run_id: str) -> None:
        self._events.pop(run_id, None)

    def get(self, run_id: str):
        """这一步的 Event(没有就 ``None``)。

        生产的执行体要把它**并进** ``ExecutionContext`` 的 abort 并集里 ——
        少了它,面板上的 abort 按钮就只能等下一个 tick,而 Director 卡在一次
        长步里时 tick 根本不会来。
        """
        return self._events.get(run_id)

    def is_set(self, run_id: str) -> bool:
        ev = self._events.get(run_id)
        return bool(ev is not None and ev.is_set())

    def signal(self, run_id: str) -> bool:
        """API 线程用:立刻置位。返回是否真有这么一个 run。"""
        ev = self._events.get(run_id)
        if ev is None:
            return False
        ev.set()
        return True

    @property
    def live_run_ids(self) -> tuple:
        return tuple(self._events)


__all__ = ["StepOutcome", "ExecutorPort", "LatchState", "NullLatch",
           "NullTemperature", "Notification", "LoggingNotifier",
           "AbortRegistry"]
