"""EnvironmentMonitor: continuous background monitoring of all sensors."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from mast.core.types import SensorReading
from mast.environment.alarm import worst_status
from mast.environment.base import EnvironmentSensor
from mast.logging.storage import ExperimentStorage

logger = logging.getLogger(__name__)

# Statuses that constitute an "alarm condition" worth surfacing/escalating.
_ALERT_STATUSES = ("warning", "alarm", "error")

AlarmCallback = Callable[[str, SensorReading, str], None]
# (sensor_name, reading, previous_status) -> None


class EnvironmentMonitor:
    """Continuously monitors all environment sensors, logs to storage.

    Over-limit handling (requirement: 如果有真空计/温度计，支持数据存档和超标报警):
      * Every reading is archived to ``environment_log`` (the SQLite store).
      * When a sensor's status crosses *into* an alert state (ok → warning /
        alarm / error), :pyattr:`on_alarm` is invoked once for that transition
        and a WARNING is logged. ``on_alarm`` is where the GUI wires a buffer
        event / banner. Re-entry into the same state is not re-fired so a
        sustained over-limit doesn't spam.
      * :meth:`alarms` returns the sensors currently in an alert state.
    """

    def __init__(
        self,
        sensors: list[EnvironmentSensor],
        storage: ExperimentStorage | None = None,
        interval_s: float = 1.0,
        on_alarm: AlarmCallback | None = None,
        sinks: list | None = None,
        scope_provider=None,
    ):
        self._sensors: dict[str, EnvironmentSensor] = {s.name(): s for s in sensors}
        self._storage = storage
        self._interval = interval_s
        self._on_alarm = on_alarm
        # 额外的读数消费者（如 EnvironmentCsvSink）。注入式：monitor 不 import
        # core / runtime，environment 层因此不必知道实验文件夹的存在。
        self._sinks = list(sinks or [])
        # () -> (experiment_id, sample_id) —— 让读数能归属到实验/样品。
        # environment_log 此前没有这两列，写了很多行却零读取方。
        self._scope_provider = scope_provider
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest: dict[str, SensorReading] = {}
        self._prev_status: dict[str, str] = {}
        self._lock = threading.Lock()
        # Set by stop() to wake an in-progress interval sleep immediately AND to
        # tell _loop it must not start another read cycle once it has been
        # cleared. Cleared at start() so a restarted monitor isn't pre-tripped.
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background monitoring thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("EnvironmentMonitor started (interval=%.1fs, %d sensor(s))",
                    self._interval, len(self._sensors))

    @property
    def is_running(self) -> bool:
        """True while the background archive/alarm loop is active.

        Public because a caller that must stop() the monitor to free the serial
        bus (the live rescan) has to know whether to start() it again.
        """
        return self._running

    def stop(self) -> None:
        """Stop monitoring and CLOSE every sensor's serial handle.

        Closing matters: DL-7/Lakeshore sensors open a pyserial port on first
        read and cache it; Windows opens COM ports exclusively. If we drop the
        sensors without closing, the port stays held for the process lifetime,
        a later rescan can't re-open it, and the gauge silently goes
        "unavailable". stop() is the single teardown that frees them.

        Ordering is load-bearing: the background ``_loop`` calls ``read_all()``
        which makes each sensor lazily (re)open its COM port. If we closed the
        handles while ``_loop`` were still alive, a subsequent read cycle would
        reopen — and re-cache — the very port we just freed, leaking it for the
        process lifetime. So we must guarantee the thread has TRULY stopped
        before closing handles:

          1. signal stop (``_running``/``_stop_event``) so the interval sleep
             wakes immediately and ``_loop`` won't begin another read cycle;
          2. join — bounded, because a sensor may be blocked in a kernel serial
             read longer than ``interval``;
          3. if the thread is still alive after the bounded join (stuck in a
             blocking read), close the handles to UNBLOCK that read, then join
             again so the now-unblocked loop observes the cleared flag and exits
             before it can touch the sensors. Closing again after the final join
             is harmless (close() is idempotent / best-effort).
        """
        self._running = False
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Generous bound: a slow serial read may exceed one interval. Cap at
            # a few seconds so stop() can't hang the GUI indefinitely.
            join_timeout = max(self._interval * 2, 3.0)
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                # Thread is wedged in a blocking read. Closing the transport is
                # what cancels that read; do it now, then give the loop a moment
                # to notice _running is False and return before it can reopen.
                with self._lock:
                    self._close_sensors(list(self._sensors.values()))
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    logger.warning(
                        "EnvironmentMonitor._loop did not stop within %.1fs; "
                        "sensor handles closed but thread still alive (daemon — "
                        "will not block process exit)",
                        join_timeout * 2,
                    )
        self._thread = None
        # Thread is stopped (or, in the wedged case, has already had its handles
        # closed above). Close once more under the lock — idempotent — to cover
        # the normal path where the thread exited cleanly without us closing.
        with self._lock:
            sensors = list(self._sensors.values())
        self._close_sensors(sensors)
        logger.info("EnvironmentMonitor stopped (%d sensor handle(s) closed)", len(sensors))

    @staticmethod
    def _close_sensors(sensors) -> None:
        """Close each sensor's transport if it exposes close() (unwraps
        _RenamedSensor). Best-effort — never raises."""
        for s in sensors:
            inner = getattr(s, "_inner", s)  # unwrap autodetect._RenamedSensor
            close = getattr(inner, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best-effort
                    pass

    def read_all(self) -> dict[str, SensorReading]:
        """Read all sensors once (synchronous). Returns name -> reading.

        Also detects alert-state transitions and fires ``on_alarm`` for each.
        Thread-safety: the sensor set is snapshotted under the lock (so a
        concurrent replace_sensors can't mutate the dict mid-iteration), the
        slow serial reads run WITHOUT the lock, and _prev_status / _latest are
        updated under the lock. on_alarm callbacks fire AFTER releasing the lock
        so a callback that re-enters the monitor can't deadlock.
        """
        with self._lock:
            sensors = dict(self._sensors)
        readings: dict[str, SensorReading] = {}
        for name, sensor in sensors.items():
            try:
                readings[name] = sensor.read()
            except Exception as exc:
                logger.warning("Sensor '%s' read failed: %s", name, exc)
                readings[name] = SensorReading(value=0.0, unit="", status="error")
        transitions: list[tuple[str, SensorReading, str]] = []
        with self._lock:
            for name, reading in readings.items():
                prev = self._prev_status.get(name, "ok")
                cur = reading.status
                if cur in _ALERT_STATUSES and cur != prev:
                    transitions.append((name, reading, prev))
                self._prev_status[name] = cur
            self._latest = dict(readings)
        for name, reading, prev in transitions:
            logger.warning(
                "Environment alarm: %s %s%s → %s",
                name, reading.value,
                (" " + reading.unit) if reading.unit else "", reading.status,
            )
            if self._on_alarm is not None:
                try:
                    self._on_alarm(name, reading, prev)
                except Exception as exc:  # pragma: no cover - callback is best-effort
                    logger.debug("on_alarm callback failed for %s: %s", name, exc)
        return readings

    def get_latest(self) -> dict[str, SensorReading]:
        """Get most recent cached readings."""
        with self._lock:
            return dict(self._latest)

    def alarms(self) -> dict[str, SensorReading]:
        """Sensors currently in an alert state (warning / alarm / error)."""
        with self._lock:
            return {n: r for n, r in self._latest.items()
                    if r.status in _ALERT_STATUSES}

    def overall_status(self) -> str:
        """Worst status across all sensors (alarm worst, ok best)."""
        with self._lock:
            return worst_status(*(r.status for r in self._latest.values())) \
                if self._latest else "ok"

    def check_health(self) -> dict[str, bool]:
        """Check health of all sensors."""
        return {name: sensor.is_healthy() for name, sensor in self._sensors.items()}

    def add_sensor(self, sensor: EnvironmentSensor) -> None:
        """Add a sensor at runtime."""
        self._sensors[sensor.name()] = sensor
        logger.info("Added sensor: %s", sensor.name())

    def replace_sensors(self, sensors: list[EnvironmentSensor]) -> None:
        """Swap the whole sensor set (used after a config change / rescan).

        Closes the displaced sensors (those not carried over by identity) so
        their serial ports are freed rather than leaked. NB: when the caller
        re-probes COM ports to BUILD ``sensors``, it must free the old ports
        FIRST (see MASTApp._rebuild_environment_monitor, which stop()s the old
        monitor before building) — closing here only covers the in-place swap.
        """
        new = {s.name(): s for s in sensors}
        new_ids = {id(s) for s in new.values()}
        with self._lock:
            old = list(self._sensors.values())
            self._sensors = new
            self._latest = {}
            self._prev_status = {}
        self._close_sensors([s for s in old if id(s) not in new_ids])
        logger.info("EnvironmentMonitor sensors replaced (%d sensor(s))", len(new))

    def sensor_names(self) -> list[str]:
        return list(self._sensors.keys())

    def _loop(self) -> None:
        """Background loop: read all sensors, log to storage.

        Re-checks the stop flag *before* each read cycle so that once stop()
        has signalled, the loop never starts a fresh read_all() (which would
        reopen a COM port that stop() is about to close). The inter-cycle wait
        uses the stop Event so stop() interrupts it immediately rather than
        blocking up to a full interval.
        """
        while self._running and not self._stop_event.is_set():
            readings = self.read_all()
            eid = sid = None
            if self._scope_provider is not None:
                try:
                    eid, sid = self._scope_provider()
                except Exception:  # noqa: BLE001 — scope 读不到就不归属，照常记录
                    eid = sid = None
            if self._storage is not None:
                for name, reading in readings.items():
                    try:
                        self._storage.log_environment(
                            sensor_name=name,
                            value=reading.value,
                            unit=reading.unit,
                            status=reading.status,
                            experiment_id=eid,
                            sample_id=sid,
                        )
                    except Exception as exc:
                        logger.warning("Failed to log sensor '%s': %s", name, exc)
            # 额外 sink（CSV 落盘）。每个 sink 自己吞异常并自我禁用；这里再包一层,
            # 因为这个循环的首要职责是发现真空/温度异常并告警 —— 写文件失败绝不
            # 能让它停摆。
            for sink in self._sinks:
                for name, reading in readings.items():
                    try:
                        sink.write(name, reading.value, reading.unit, reading.status)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("env sink write failed for %s: %r", name, exc)
            # Interruptible sleep: returns True immediately when stop() sets the
            # event, so teardown isn't delayed by a long interval.
            if self._stop_event.wait(self._interval):
                break
