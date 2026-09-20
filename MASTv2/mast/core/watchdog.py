"""轮询隧道电流，连续超阈值时请求紧急退针。

窗口内每个读数都超阈值才调用 on_anomaly；执行侧负责停止进针及马达、
撤回 Z、设置 abort 并发出 E_STOP。未确认退针时不上闩，冷却后重试。

阈值通过 threshold_getter 每 tick 读取监控设置 cm_sat_current_a，与饱和
判据共享配置来源。无法读取时不静默使用另一套硬编码限值。

蓄意针尖操作期间由 core.tip_intent.active_tip_work 活谓词抑制。它读取受锁
保护的进程级仪器令牌状态，因此看门狗线程可见，不依赖线程局部状态或容易
失配的 disable/enable 配对。抑制时清空窗口，恢复后重新积累完整确认窗口。

显式 disable(for_s=...) 保留给调用方，但必须有期限，不能永久关闭保护。
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable

from mast.core.connection import ConnectionPool

logger = logging.getLogger(__name__)

#: :meth:`SafetyWatchdog.disable` 的默认时长,以及它的硬上界。
#:
#: **不是「够长所以不会碍事」的数**,是照它保护的动作算出来的:一次深扎的
#: ``poke_dwell_s`` 取值上界是 10 s(``core/noble_tip_workflow``),电脉冲是毫秒级;
#: 扎完之后电流还要沉降,``monitoring`` 对同一件事的出厂估计是 ``cm_afterglow_s`` = 5 s。
#: 10 + 5 = **15 s**。
#:
#: 上界取默认的 4 倍(60 s),给「一次扎针后面紧跟着另一次」留余量,同时保证
#: **任何**忘记恢复的路径最多让安全网静默一分钟 —— 而不是静默到进程重启。
DEFAULT_DISABLE_S: float = 15.0
MAX_DISABLE_S: float = 60.0

#: 令牌放手之后,还算「MAST 刚刚还在操作」多久(秒)。
#:
#: 它盖的是**技能与技能之间**那个空档(agent 在想下一步),此刻把针放在接触位置的
#: 仍然是 MAST。锚点用**本机制自己的确认窗口**(``interval_s × window_size``,出厂
#: 4 s):桥只需要比一次确认窗口长一点点就够,而它是这套判定自己的时间尺度。
#:
#: **刻意不用 ``cm_afterglow_s``**(monitoring 对同类问题的 5 s):那个数自己标着
#: 「出厂 5 s 是占位值,不是实测」。**派生自一个占位值,就是把未标定状态传染出去,
#: 而且传染之后更难发现** —— 下一个人看到这里有个依据,不会再去查它的依据是假的。
_BRIDGE_FROM_WINDOW = True     # 见 SafetyWatchdog._mast_driving()

#: 判定为「人在操作」之后,**抑制自己过期**的时限(秒)。超过它就不管判定如何,退针。
#:
#: 人工操作的识别只是推断，不能无限期抑制安全动作。
#: 本时限是独立的经验默认，不应随确认窗口自动缩放。
#: 在目标仪器上应复核人工操作期间连续贴轨的时长分布，并确认该上限适用。
HUMAN_RAIL_OVERRIDE_S: float = 20.0


def _shipped_saturation_threshold_a() -> float:
    """兜底阈值 —— **仍然是 ``cm_sat_current_a``**,只是取它的出厂默认而非活值。

    刻意不在这里写一个数字字面量:写一个,就又造出了第二个「握着执行器的数」,
    因此这里读取同一个 dataclass 的同一个字段。

    ``MonitorThresholds`` 只依赖 ``threading`` / ``dataclasses``(没有 numpy),
    所以这个 import 实际上不会失败;真失败了由调用方大声记一笔并跳过本 tick。
    """
    from mast.monitoring.thresholds import MonitorThresholds

    return float(MonitorThresholds().cm_sat_current_a)


class SafetyWatchdog(threading.Thread):
    """Daemon thread that polls tunneling current via monitor port.

    Uses sliding window: if all readings in the window exceed the
    threshold, triggers the anomaly callback (typically SafeRetract
    via emergency port).

    Reference: Scanbot (safeCurrentCheck) + Nanonis_AutoSTM (SafeTipthreading).
    """

    daemon = True

    def __init__(
        self,
        pool: ConnectionPool,
        on_anomaly: Callable[[], object],
        current_threshold_a: float | None = None,
        interval_s: float = 0.5,
        window_size: int = 8,
        retrigger_cooldown_s: float = 30.0,
        *,
        threshold_getter: Callable[[], float] | None = None,
        suppress_getter: Callable[[], str] | None = None,
        idle_getter: Callable[[], "float | None"] | None = None,
        human_rail_override_s: float = HUMAN_RAIL_OVERRIDE_S,
    ):
        """
        Parameters
        ----------
        current_threshold_a:
            固定阈值。**只给测试和显式覆盖用。** 生产路径传 ``threshold_getter``
            以便与监控饱和阈值共享配置来源。
            ``None``(默认)= 没有固定值,每 tick 去 getter 那里问。
        threshold_getter:
            每 tick 调一次,返回当前阈值(安培)。生产上由 ``executor.start_watchdog``
            接到 ``monitoring.thresholds.cm_sat_current_a``。
        suppress_getter:
            每 tick 调一次,返回**正在进行的蓄意针尖动作**的名字,没有就是 ``""``。
            生产上接 ``core.tip_intent.active_tip_work``。返回真值时这一 tick
            不判、并清空窗口。永不被信任会抛(调用点自己兜住)。
        """
        super().__init__(name="SafetyWatchdog")
        self._pool = pool
        self._on_anomaly = on_anomaly
        self._fixed_threshold = (float(current_threshold_a)
                                 if current_threshold_a is not None else None)
        self._threshold_getter = threshold_getter
        self._suppress_getter = suppress_getter
        #: 仪器令牌空闲了多久;``None`` = 正被持有(= MAST 正在驱动)。
        #: 生产上接 ``instrument_lock().idle_s``。``None`` getter ⇒ 判据缺席 ⇒
        #: **一律认为 MAST 在驱动**(照常武装)—— 缺席不该悄悄关掉这张网。
        self._idle_getter = idle_getter
        self._human_rail_override_s = float(human_rail_override_s)
        #: 当前这一段连续贴轨是什么时候开始的(``monotonic``);没在贴轨就是 ``None``。
        #: 用来让「判定为人工」的抑制**自己会过期**。
        self._rail_since: float | None = None
        #: 「因判定为人工而没动手」只在边沿说一次,不刷屏。
        self._said_human = False
        #: 最近一次成功解析出来的阈值。getter 临时读不到时先用它 —— 一个刚刚还
        #: 正确的值比出厂默认更接近这台机器的真实前放量程。
        self._last_good_threshold: float | None = self._fixed_threshold
        #: 上一 tick 是不是处在抑制态 —— 只用来让日志在**边沿**各说一次,
        #: 而不是每 0.5 s 刷一行。
        self._was_suppressed = False
        #: 阈值降级过没有(用来让那句 WARNING 也只在边沿说)。
        self._threshold_degraded = False
        #: :meth:`disable` 的到期时刻(``time.monotonic()``)。``None`` = 没在计时。
        self._disabled_until: float | None = None
        self._interval = interval_s
        self._window_size = window_size
        # After a retract that could NOT be confirmed (callback returned False),
        # the latch is NOT set — otherwise a single failed SafeRetract would
        # permanently disarm the safety net for the whole session (review
        # 2026-07-03). Instead we cool down this long before allowing a re-fire,
        # so we keep retrying the retract without spamming it every tick.
        self._retrigger_cooldown_s = retrigger_cooldown_s
        self._last_trigger_monotonic: float | None = None
        self._stop_event = threading.Event()
        self._buffer: collections.deque[float] = collections.deque(maxlen=window_size)
        self._anomaly_triggered = threading.Event()
        self._enabled = threading.Event()
        self._enabled.set()  # Enabled by default

    def run(self) -> None:
        logger.info(
            "SafetyWatchdog started: threshold=%s, interval=%.1fs, window=%d "
            "(= 连续 %.1f s 全部超阈值才退针);修针期间由 tip_intent 活谓词抑制:%s",
            (f"{self._fixed_threshold:.2e} A(固定)" if self._fixed_threshold is not None
             else "跟随 cm_sat_current_a"),
            self._interval, self._window_size,
            self._interval * self._window_size,
            "是" if self._suppress_getter is not None else "**否(没接谓词)**",
        )
        while not self._stop_event.is_set():
            if not self._enabled.is_set():
                # 先看 disable 到没到期 —— 到期自己活过来,不等任何人调 enable()。
                if self._expire_disable_if_due():
                    time.sleep(self._interval)
                    continue

            # 蓄意修针期间不判:扎针本来就该让电流贴轨(poke_dwell_s 上界 10 s,
            # 而本网只要连续 4 s 就开火)。清空窗口 ⇒ 恢复后要重新攒满才可能开火,
            # 这就是余波期,不需要另一个旋钮。
            who = self._suppressing_skill()
            if who:
                if not self._was_suppressed:
                    logger.info(
                        "SafetyWatchdog: %s 正在做蓄意针尖动作 —— 暂停判定并清空窗口;"
                        "它一还令牌就自动恢复(活谓词,不靠配对调用)。", who)
                    self._was_suppressed = True
                self._buffer.clear()
                time.sleep(self._interval)
                continue
            if self._was_suppressed:
                logger.info("SafetyWatchdog: 蓄意针尖动作结束 —— 恢复判定"
                            "(需重新攒满 %d 个读数,约 %.1f s)",
                            self._window_size, self._interval * self._window_size)
                self._was_suppressed = False

            threshold = self._resolve_threshold()
            if threshold is None:
                time.sleep(self._interval)
                continue

            try:
                record = self._pool.safe_call("Current_Get", role="monitor")
                if not record.error and record.return_value is not None:
                    current = self._extract_current(record.return_value)
                    if current is None:
                        # Parse failure: do NOT inject a placeholder (0.0) into the
                        # sliding window. A spurious 0.0 always reads as below
                        # threshold, so it would reset/dilute the all-readings-high
                        # check and delay or mask a genuine tip-crash overcurrent
                        # (review: parse-failure must never hide a real anomaly).
                        # Skip this tick — the window keeps only valid readings.
                        logger.debug(
                            "SafetyWatchdog: unparseable Current_Get reply %r; "
                            "skipping tick (window preserved)",
                            record.return_value,
                        )
                        time.sleep(self._interval)
                        continue
                    self._buffer.append(abs(current))

                    # Check sliding window
                    railed = (len(self._buffer) >= self._window_size
                              and all(v > threshold for v in self._buffer))
                    if not railed:
                        self._rail_since = None
                        self._said_human = False
                    else:
                        held_s = self._rail_held_for(time.monotonic())
                        # ── 现在动仪器的是人吗? ──────────────────────────
                        #
                        # 用户用 Nanonis GUI 手动修针时 MAST 一条命令都不发,
                        # 令牌当然也没人持 —— 而他扎针打出来的贴轨和撞针长得一样。
                        # 因此不能仅凭令牌空闲，就把手动操作期间的瞬态判成撞针。
                        #
                        # ⚠️ 但这个判定是**推断**:「MAST 安静」不等于「有人在操作」。
                        # 抑制必须自行过期；持续异常超过时限后不再依赖人工操作推断。
                        # 见 :data:`HUMAN_RAIL_OVERRIDE_S`。
                        if not self._mast_driving() and held_s < self._human_rail_override_s:
                            if not self._said_human:
                                logger.warning(
                                    "SafetyWatchdog: 电流持续贴轨(%.1fs),但 MAST "
                                    "没在驱动仪器 —— 判定为**人工操作**,"
                                    "**记录但不退针**。若贴轨持续超过 %.0fs 仍会退针"
                                    "(人工扎针的贴轨上界是 4 s)。",
                                    held_s, self._human_rail_override_s)
                                self._said_human = True
                            time.sleep(self._interval)
                            continue
                        if self._said_human:
                            logger.critical(
                                "SafetyWatchdog: 贴轨已持续 %.1fs,超过人工操作的合理"
                                "上界 —— **不再按人工处理**,执行退针。", held_s)
                            self._said_human = False
                        if self._should_fire():
                            logger.critical(
                                "SafetyWatchdog: anomaly detected! All %d readings > %.2e A",
                                self._window_size, threshold,
                            )
                            self._last_trigger_monotonic = time.monotonic()
                            # Push real-time anomaly event to UI
                            try:
                                from mast.core.events import EventBus
                                EventBus.get().publish_anomaly(
                                    threshold_a=threshold,
                                    readings=list(self._buffer),
                                )
                            except Exception:
                                pass
                            # The callback performs the emergency retract and
                            # returns whether it was CONFIRMED. Only latch (stop
                            # monitoring) on a confirmed retract; on failure leave
                            # the latch clear so the cooldown lets us retry rather
                            # than disarming the net forever. A None return (other
                            # callers) is treated as success for backward compat.
                            try:
                                confirmed = self._on_anomaly()
                            except Exception as cb_exc:  # never let the poll die
                                logger.critical(
                                    "SafetyWatchdog on_anomaly raised: %s", cb_exc)
                                confirmed = False
                            if confirmed is not False:
                                self._anomaly_triggered.set()
                            else:
                                logger.critical(
                                    "SafetyWatchdog: retract NOT confirmed — will "
                                    "retry after %.0fs cooldown (net stays armed)",
                                    self._retrigger_cooldown_s,
                                )
            except Exception as exc:
                logger.debug("SafetyWatchdog poll error: %s", exc)

            time.sleep(self._interval)

        logger.info("SafetyWatchdog stopped")

    # ── 阈值 ─────────────────────────────────────────────────────────────

    def _resolve_threshold(self) -> float | None:
        """本 tick 生效的阈值(安培),解析不出来时 ``None``。

        **绝不静默回退。** ``lookup(name) || DEFAULT`` 那个形状 —— name 写错就得到
        一个能跑的错版本 —— 正是这次事故的本体,修的时候不能把它再种一次。
        所以每一次降级都在 WARNING 以上说话(只在**边沿**说,不刷屏),并且说清楚
        用的是哪一档:

        1. 显式固定值(测试/覆盖)——直接用,不说话;
        2. ``threshold_getter`` 的活值 —— 正常路径,恢复正常时说一句;
        3. 上一次拿到过的好值 —— WARNING,「在用上次的值」;
        4. ``cm_sat_current_a`` 的**出厂默认** —— WARNING,「在用兜底值」;
        5. 连出厂默认都读不到 —— CRITICAL,返回 ``None``,本 tick 不判。
           这不是「静默失效」:每个冷却周期都会再喊一次,而下一 tick 还会重试。
        """
        if self._fixed_threshold is not None:
            return self._fixed_threshold

        if self._threshold_getter is not None:
            try:
                value = float(self._threshold_getter())
            except Exception as exc:  # noqa: BLE001
                value = float("nan")
                logger.debug("SafetyWatchdog: 阈值 getter 抛了: %s", exc)
            if value == value and value > 0.0:      # 非 NaN 且为正
                if self._threshold_degraded:
                    logger.warning(
                        "SafetyWatchdog: 阈值恢复正常 —— 重新跟随 cm_sat_current_a "
                        "(%.3e A)", value)
                    self._threshold_degraded = False
                self._last_good_threshold = value
                return value

        # ── 到这里就是降级了 ──
        if self._last_good_threshold is not None:
            if not self._threshold_degraded:
                logger.warning(
                    "SafetyWatchdog: 读不到 cm_sat_current_a —— **改用上次的好值** "
                    "%.3e A 继续武装(不是静默回退,这一行就是告知)。",
                    self._last_good_threshold)
                self._threshold_degraded = True
            return self._last_good_threshold

        try:
            fallback = _shipped_saturation_threshold_a()
        except Exception:  # pragma: no cover — MonitorThresholds 无重依赖,不该失败
            if not self._threshold_degraded:
                logger.critical(
                    "SafetyWatchdog: 既读不到 cm_sat_current_a 也读不到它的出厂默认 "
                    "—— **本轮不判**(每 tick 重试)。此刻针尖保护只剩拒绝型防护。",
                    exc_info=True)
                self._threshold_degraded = True
            return None
        if not self._threshold_degraded:
            logger.warning(
                "SafetyWatchdog: 读不到 cm_sat_current_a 的活值,且本进程从未读到过 "
                "—— **改用它的出厂默认** %.3e A 继续武装。这个数没有在这台机器上标定过,"
                "请去设置里确认前置放大器满量程。", fallback)
            self._threshold_degraded = True
        self._last_good_threshold = fallback
        return fallback

    # ── 抑制 ─────────────────────────────────────────────────────────────

    def _suppressing_skill(self) -> str:
        """现在有没有蓄意的针尖动作在进行;有就返回它的名字。永不抛。

        读不到一律返回 ``""``(= **不抑制**)。抑制的失败方向必须是「保护照常生效」,
        与 ``tip_intent.exempt_during_tip_work`` 的同一条纪律。
        """
        if self._suppress_getter is None:
            return ""
        try:
            return str(self._suppress_getter() or "")
        except Exception:  # noqa: BLE001 — 看门狗线程,绝不能因此死掉
            logger.debug("SafetyWatchdog: 读不到修针意图(不抑制)", exc_info=True)
            return ""

    # ── 现在动仪器的是我们,还是人? ──────────────────────────────────────

    def _mast_driving(self) -> bool:
        """MAST 此刻是不是在驱动仪器。永不抛。

        ``令牌被持有`` **或** ``令牌刚放手不到一个确认窗口``。

        判据缺席(没接 getter / 读不到)⇒ **True**(照常武装)。
        这个失败方向是刻意的:抑制是**新加的**,而它关掉的是这棵树上唯一一条会自己
        动手保护针尖的路。**读不清就武装**,不是「读不清就闭嘴」。
        """
        if self._idle_getter is None:
            return True
        try:
            idle = self._idle_getter()
        except Exception:  # noqa: BLE001
            logger.debug("SafetyWatchdog: 读不到仪器令牌状态(按 MAST 在驱动处理)",
                         exc_info=True)
            return True
        if idle is None:              # 正被持有
            return True
        try:
            return float(idle) < (self._interval * self._window_size)
        except Exception:  # noqa: BLE001
            return True

    def _rail_held_for(self, now: float) -> float:
        """当前这一段连续贴轨已经持续了多久(秒)。调用前 buffer 已判定为全超阈。"""
        if self._rail_since is None:
            self._rail_since = now
        return max(0.0, now - self._rail_since)

    def _expire_disable_if_due(self) -> bool:
        """到期就自动重新武装。返回「现在是不是仍然被 disable 着」。

        **这是 :meth:`disable` 能安全存在的全部理由**:恢复不依赖任何人记得调
        ``enable()``。
        """
        until = self._disabled_until
        if until is None:
            return not self._enabled.is_set()
        if time.monotonic() >= until:
            self._disabled_until = None
            self._buffer.clear()
            self._enabled.set()
            logger.warning(
                "SafetyWatchdog: disable 到期,**自动重新武装**"
                "(没有人调 enable(),这正是设计)。")
            return False
        return True

    def _should_fire(self) -> bool:
        """Decide whether to (re)fire the anomaly callback this tick.

        Fire when the latch is clear AND either we have never fired, or the
        previous fire was an UNCONFIRMED retract and the cooldown has elapsed.
        Once a retract is confirmed the latch is set and this stays False until
        an explicit reset() (recovery).
        """
        if self._anomaly_triggered.is_set():
            return False
        last = self._last_trigger_monotonic
        if last is None:
            return True
        return (time.monotonic() - last) >= self._retrigger_cooldown_s

    @staticmethod
    def _extract_current(parsed: object) -> float | None:
        """Pull the current value out of a Current_Get reply.

        Current_Get shape is ``(header, body, [current])`` so the value lives at
        ``parsed[2][0]``. Returns ``None`` (not 0.0) if the reply does not match
        that shape or the value isn't a finite float — the caller treats ``None``
        as "skip this tick" so a malformed reply never pollutes the sliding
        window used for tip-crash detection.
        """
        if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
            return None
        payload = parsed[2]
        if not isinstance(payload, (list, tuple)) or len(payload) == 0:
            return None
        try:
            current = float(payload[0])
        except (TypeError, ValueError):
            return None
        # NaN / inf are not real readings — drop them too.
        if current != current or current in (float("inf"), float("-inf")):
            return None
        return current

    def stop(self) -> None:
        """Signal the watchdog to stop."""
        self._stop_event.set()

    def enable(self) -> None:
        """Enable current monitoring.

        Clears the sliding window first: any readings still buffered from
        before the disable window are stale. If monitoring was disabled during
        an intentionally high-current operation (e.g. tip conditioning), those
        leftover readings could make the very first post-enable poll satisfy
        ``all(v > threshold)`` and falsely fire SafeRetract ().
        """
        self._buffer.clear()
        self._disabled_until = None
        self._enabled.set()
        logger.info("SafetyWatchdog enabled")

    def disable(self, for_s: float = DEFAULT_DISABLE_S) -> float:
        """临时停判,**并且到点自己活过来**。返回实际生效的时长(秒)。

        ⚠️ **时长不可省,也不可无限。** 2026-08-10 之前这个方法没有时限、并且
        **零调用方** —— 一个「靠配对 ``enable()`` 才能恢复」的 disable 就是本次要修
        的那个病的另一种形态:某条路径抛异常、提前 return、或者作者忘了写 finally,
        安全网就**静默地永久关闭**,而从外面看和武装着一模一样。

        所以恢复**不依赖任何人记得调 ``enable()``**::

            wd.disable(for_s=12)   # 12 秒后自动重新武装,不需要配对调用

        :meth:`enable` 仍然可用,但它只是**提前恢复**,不是恢复的唯一途径。

        时长被夹在 ``(0, MAX_DISABLE_S]``;两个数的理由见
        :data:`DEFAULT_DISABLE_S` / :data:`MAX_DISABLE_S`(照 ``poke_dwell_s`` 的
        取值上界 + 电流沉降算出来的,不是「够长所以不碍事」)。

        自动抑制**不走这条路**:蓄意修针由 ``suppress_getter`` 活谓词处理
        (见模块 docstring),那条路连时限都不需要 —— 令牌一还就恢复。
        这里是给不持令牌的显式调用方留的逃生门。

        清空滑动窗口:停判之前刚采到的读数不能跨过这段窗口活下来,否则恢复后第一
        次轮询就可能凑满 ``all(v > threshold)`` 而误退针()。
        """
        # ⚠️ ``for_s <= 0`` **不是**「用默认值」。``_pos()`` 那个「v<=0 → 出厂默认」
        # 语义不适用于安全网的关闭时长：「关 0 秒」若被读成正的默认时长，
        # 是把保护关掉了一段没人要求的时间。
        #
        # 所以 ≤0 / NaN / 读不懂 ⇒ **什么都不关**,并且说一句。失败方向是「保护照常生效」。
        try:
            requested = float(for_s)
        except (TypeError, ValueError):
            requested = float("nan")
        if requested != requested or requested <= 0.0:
            logger.warning(
                "SafetyWatchdog: disable(for_s=%r) 不是一个正的时长 —— **没有关闭任何"
                "东西**,安全网照常武装。要关就给一个正数秒。", for_s)
            return 0.0
        span = min(MAX_DISABLE_S, requested)
        self._disabled_until = time.monotonic() + span
        self._enabled.clear()
        self._buffer.clear()
        logger.info("SafetyWatchdog disabled for %.1fs(到点自动重新武装)", span)
        return span

    def reset(self) -> None:
        """Clear anomaly state and buffer after recovery (re-arm the net)."""
        self._buffer.clear()
        self._anomaly_triggered.clear()
        self._last_trigger_monotonic = None
        logger.info("SafetyWatchdog reset")

    @property
    def is_anomaly_triggered(self) -> bool:
        return self._anomaly_triggered.is_set()

    def set_threshold(self, threshold_a: float) -> None:
        """钉死一个固定阈值,**从此不再跟随 ``cm_sat_current_a``**。

        这是覆盖,不是调参 —— 调参请去改 ``cm_sat_current_a``,那是这个数的真源,
        改完下一 tick 就生效。这里存在是为了测试和显式覆盖。
        """
        self._fixed_threshold = float(threshold_a)
        self._last_good_threshold = self._fixed_threshold
        logger.warning(
            "SafetyWatchdog: 阈值被钉死在 %.2e A —— **不再跟随 cm_sat_current_a**。",
            threshold_a)

    @property
    def effective_threshold_a(self) -> float | None:
        """此刻会用的阈值(只读,给测试与遥测)。解析不出来时 ``None``。"""
        return self._resolve_threshold()

    @property
    def is_suppressed_now(self) -> bool:
        """此刻是不是因为蓄意针尖动作而停判(只读)。"""
        return bool(self._suppressing_skill())
