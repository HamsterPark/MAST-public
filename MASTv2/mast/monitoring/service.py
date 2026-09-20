"""The current-monitor daemon: lifecycle, context labelling, suppression, telemetry.

One background thread pulls segments from the pump, labels each with what the
instrument was doing, extracts features, stores them, and publishes a scalar
summary. It is deliberately the only place in this package that knows about the
rest of MAST.

Three behaviours worth stating up front, because each fixes a failure this repo
has already had:

**It stops when the pool does.** ``reconnect()`` tears down the connection pool;
a daemon still polling against the dead pool spins on errors and races the
rebuild for the port. ``InstrumentState`` is explicitly stopped there for that
reason — and ``ExperimentalMonitor``, added later, was not, which is a live bug.
This service registers on setup, shutdown AND reconnect.

**It never feeds an open breaker.** When comms are unhealthy the loop parks with
zero TCP until they recover. Retrying into an open circuit breaker is the exact
behaviour the breaker exists to prevent.

**It goes quiet when it cannot work.** The bundled simulator does not load
Osci1T at all, so "unavailable" is a normal state, not an error: log once, emit
one status event, re-probe every minute, and never spam.

Suppression: while a tip-shaping, pulsing or approach skill is running, the
current is *supposed* to look violent. Those segments are recorded and pinned —
they are the most valuable training data there is — but they never raise an
alert, and their verdict is stored as ``suppressed`` rather than silently as
``ok``, so a later reader can tell "we chose not to judge this" from "we judged
it fine".
"""
from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Substrings (case-insensitive) of skill names during which current excursions
#: are expected. Matched against whoever holds the instrument token.
SUPPRESS_SKILL_PATTERNS: frozenset[str] = frozenset({
    # tip conditioning / pulsing — the current is meant to look violent
    "tipshape", "tippulse", "conditiontip", "shapetip", "biaspulse",
    # PrepareNobleTip contains none of the substrings above, and it is the one
    # that runs longest (pulses AND plunges for tens of minutes): without this
    # entry every shot inside it reads as a CRITICAL current excursion and the
    # halt it triggers stops the conditioning run partway through.
    "preparenobletip",
    # Special-tip operations intentionally vary bias and may generate current excursions; include their outer skill names in the suppression context.
    "resolutiontip", "biaswiggle",
    # The instrument token names the outermost composite, so the complete tip-forging workflow needs its own suppression entry even when every inner step is already covered.
    "forgeau",
    # approach / retract / coarse motion — not a tunnelling junction throughout
    "autoapproach", "approachtip", "tryengage", "withdraw", "retract",
    "motormove", "coarse",
    # spectroscopy — an I-V or I-z sweep drives the preamp into its rail BY
    # DESIGN. Without these, a grid of STS points reads as sustained saturation
    # and three segments later halts the very composite skill running it.
    "biasspectr", "spectroscopy", "spectrum", "sts", "sweep", "iv_curve",
    "setbias", "lockin",
})

#: States reported to the UI. Ordered roughly by how much attention they deserve.
STATE_DISABLED = "disabled"
STATE_NO_POOL = "no_pool"
STATE_COMMS_DOWN = "comms_down"
STATE_PROBING = "probing"
STATE_RUNNING = "running"
STATE_UNAVAILABLE = "unavailable"
STATE_PAUSED = "paused"

_DISABLED_POLL_S = 2.0          # cheap enough to make the settings switch feel live
_PAUSED_POLL_S = 5.0
_UNAVAILABLE_RETRY_S = 60.0
#: Short bounded join when starting while a previous worker is retiring. Its
#: blocking points are all bounded (safe_call ≤35 s, _stop.wait ≤60 s), and on
#: the reconnect path it has usually already left — so a brief wait converts the
#: common case back into a successful start instead of a refusal.
_RETIRING_JOIN_S = 2.0
#: How long the background watcher keeps waiting before giving up and saying so.
_RETIRING_MAX_WAIT_S = 120.0
_RETENTION_INTERVAL_S = 600.0


class CurrentMonitorService:
    """Background acquisition + feature + alert loop for the tunnelling current."""

    def __init__(self, *,
                 pool_getter: Callable[[], Any],
                 state_getter: Callable[[], Any] | None = None,
                 scan_id_getter: Callable[[], str] | None = None,
                 store_getter: Callable[[], Any] | None = None,
                 thresholds_getter: Callable[[], Any] | None = None):
        self._pool_getter = pool_getter
        self._state_getter = state_getter or (lambda: None)
        self._scan_id_getter = scan_id_getter or (lambda: "")
        if store_getter is None:
            from mast.monitoring.store import get_store
            store_getter = get_store
        self._store_getter = store_getter
        if thresholds_getter is None:
            from mast.monitoring.thresholds import get_monitor_thresholds
            thresholds_getter = get_monitor_thresholds
        self._th = thresholds_getter

        self._thread: threading.Thread | None = None
        #: A worker that outlived its join — see start()/stop().
        self._retiring: threading.Thread | None = None
        #: Background waiter that restarts once _retiring finally exits.
        self._restart_watcher: threading.Thread | None = None
        #: Bumped by every stop(). The restart watcher captures it and gives up
        #: if it changed — otherwise a watcher armed before a shutdown would
        #: wake up afterwards and start a worker during teardown.
        self._generation = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pump = None
        self._engine = None
        #: 辅助通道采样器（Z 位置 / qPlus 振幅 / Δf）。零示波器写入，寄生在这条
        #: worker 线程上 —— 见 monitoring/aux_channels.py 的模块 docstring。
        self._aux = None
        #: 最近一段**已经落库**的段落 id。搭车采样（``_pump_idle``）拿它给 aux 行
        #: 标记「我旁边是哪一段电流」—— 那时下一段还没成形，更没有 id。
        #: ``None`` = 本次运行还没攒出过一段（刚起服务的头一秒），照样采，
        #: 只是那几行没有段落可指。
        self._last_seg_id = None
        #: 活跃噪声基线的 sigma 模型，缓存在这里。判据每段都要用它，而
        #: ``AlertEngine.evaluate`` 按契约不做 I/O —— 每段查一次库还会把锁压到
        #: 采集线程的路径上。刷新是**定时**的（见 ``_BASELINE_REFRESH_S``），不是
        #: 每段一次：基线是人发起的动作产生的，秒级的新鲜度没有意义。
        #: ``None`` = 没有活跃基线 ⇒ 判据回到固定阈值，与本功能上线前逐字节相同。
        self._baseline_model = None
        self._baseline_id = None
        self._baseline_checked_at = 0.0
        self._warmed = False
        self._event_handle = None

        self._state = STATE_DISABLED
        self._detail = ""
        self._retry_in_s = 0.0
        self._segments_done = 0
        self._gaps_total_s = 0.0
        self._last_segment_ts: float | None = None
        self._last_sweep = 0.0
        self._pin_until = 0.0
        #: 抑制的「余波期」——最近一个可抑制技能结束的时刻与名字(缺陷⑨)。
        self._afterglow_until = 0.0
        self._afterglow_name = ""

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Idempotent. Safe to call from setup and again after a reconnect.

        Never runs two workers at once. ``stop()`` can return with its thread
        still alive — a poll can sit in ``safe_call`` for up to 35 s (5 s recv +
        30 s role lock) — and a second one started then gives two pumps writing
        segment files concurrently. Their names are unique only under a single
        writer ("take the first free name"), so two in the same millisecond both
        see the same name free and one overwrites the other; worse, each writes
        a ``segments`` row for the same wall-clock second, both claiming to be a
        contiguous recording.

        Waiting is preferred to refusing. On the reconnect path a live previous
        worker is the COMMON case, not an edge one: ``stop_service()`` joins for
        5 s while the worker may be inside a 30 s role lock, and only ~1-3 s of
        pool teardown/rebuild separates it from ``start_service()``. A short
        bounded join here turns most of those back into a successful start; the
        genuinely wedged remainder gets a background watcher instead of silence,
        because the moment a monitor is most needed is right after the link was
        replaced.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            retiring = self._retiring
        # Join OUTSIDE the lock: status() takes it, and blocking that would make
        # the API stall on exactly the fault we are trying to report.
        if retiring is not None and retiring.is_alive():
            retiring.join(timeout=_RETIRING_JOIN_S)

        blocked = False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._retiring is not None and self._retiring.is_alive():
                blocked = True
            else:
                self._retiring = None
                self._stop = threading.Event()
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name="CurrentMonitorService")
                self._thread.start()
        if blocked:
            self._handle_blocked_start()
            return
        atexit.register(self._stop.set)
        self._subscribe_skill_steps()

    def _handle_blocked_start(self) -> None:
        """Say so, and arrange to try again — a refused start must not be silent.

        Without this the daemon simply stops: ``running`` goes false while
        ``state`` still holds whatever the old worker last wrote (only the worker
        calls ``_set_state``), so ``/status`` reports ``running=false`` next to
        ``state="running"`` and the operator sees a panel that merely stopped
        updating.
        """
        logger.warning("current monitor: previous worker still winding down; "
                       "waiting for it before starting a new one")
        self._set_state(STATE_PAUSED, "上一个采集线程仍在退出,监控暂未启动",
                        retry_in_s=_RETIRING_MAX_WAIT_S)
        with self._lock:
            watcher = self._restart_watcher
            if watcher is not None and watcher.is_alive():
                return
            generation = self._generation
            self._restart_watcher = threading.Thread(
                target=self._await_retirement_then_start, args=(generation,),
                daemon=True, name="CurrentMonitorRestart")
            self._restart_watcher.start()

    def _await_retirement_then_start(self, generation: int) -> None:
        try:
            t = self._retiring
            if t is not None:
                t.join(timeout=_RETIRING_MAX_WAIT_S)
                if t.is_alive():
                    # Say what is actually true. A manual start would take this
                    # same refusal path (2 s join → still alive → blocked), so
                    # promising "start it by hand" would hand the operator a
                    # button that cannot work — the same class of mistake as the
                    # "never issues a hardware command" wording fixed earlier.
                    logger.error(
                        "current monitor: previous worker still stuck after "
                        "%.0fs — monitoring stays off; it resumes by itself "
                        "once that thread exits", _RETIRING_MAX_WAIT_S)
                    self._set_state(
                        STATE_PAUSED,
                        "上一个采集线程长时间未退出;它退出后会自动恢复,"
                        "若持续卡住需重启服务")
                    return
            with self._lock:
                # A stop() since we were armed means teardown (or an explicit
                # disable) happened. Starting now would put a worker back up
                # during shutdown.
                if generation != self._generation:
                    return
                self._retiring = None
            self.start()
        except Exception:  # noqa: BLE001 — a watcher must never take the process
            logger.debug("restart watcher failed (swallowed)", exc_info=True)

    def stop(self, join_timeout: float = 5.0) -> None:
        """Stop the worker and forget the acquisition it was running.

        ``_pump`` is cleared as well as the thread. It caches the scope's
        channel index and timebase from ``configure()``, and after a reconnect
        those describe the OLD connection — which now matters more than it used
        to, because the channel-verification guard compares against exactly that
        cached index. A restarted service must re-probe, not carry the previous
        link's configuration into the new one.

        ``_aux`` is dropped for the same reason: it caches SIGNAL INDICES from
        ``Signals_NamesGet``, and the signal table can differ across a reconnect
        (the operator reconfigures Nanonis signals). Carrying old indices into a
        new link means silently recording a different signal under the name "Z".
        """
        with self._lock:
            self._generation += 1
            self._stop.set()
            t, self._thread = self._thread, None
            # 先接住 pump 再置空：停机时要用它把 Osci1T 还给用户（见下）。
            pump, self._pump = self._pump, None
            self._engine = None
            self._aux = None
            self._warmed = False
        self._unsubscribe_skill_steps()
        stopped_cleanly = t is None or not t.is_alive()
        if t is not None and t.is_alive():
            t.join(timeout=join_timeout)
            stopped_cleanly = not t.is_alive()
            if t.is_alive():
                # Remember it: start() must not add a second pump alongside a
                # worker that is still inside a blocking call.
                with self._lock:
                    self._retiring = t
                logger.warning("current monitor thread did not stop within %.1fs "
                               "(it is a daemon; a restart will wait for it)",
                               join_timeout)

        # Osci1T 是单实例共享模块，configure() 改过它的通道与时基。还给用户 ——
        # 不还的话，一次 MAST 启动就永久改掉了他示波器上看的信号，而他不会收到
        # 任何提示。
        #
        # **只在工作线程确实已经停下之后才还原**：worker 还活着就意味着它随时可能
        # 再 _maybe_reconfigure() 一次，那样我们只是跟自己的采集线程抢写同一个模块，
        # 还原完立刻被盖掉。宁可不还，也不要制造一场写竞争。
        #
        # 两个调用点（重连、优雅停机）都在 pool.close_all() **之前**，所以这里连接
        # 还活着。restore() 自身全程尽力而为、绝不抛。
        if stopped_cleanly and pump is not None:
            try:
                pump.restore()
            except Exception:  # noqa: BLE001 — 停机路径绝不因为「礼貌」而失败
                logger.debug("Osci1T restore failed (swallowed)", exc_info=True)

    def status(self) -> dict:
        """JSON-safe snapshot for the REST endpoint. Zero TCP."""
        cfg = self._pump.config if self._pump is not None else {}
        th = self._th()
        pool = self._pool_getter()
        return {
            "running": bool(self._thread is not None and self._thread.is_alive()),
            "enabled_in_settings": bool(th.enabled),
            "alerts_enabled": bool(th.alerts_enabled),
            "state": self._state,
            "detail": self._detail,
            "retry_in_s": round(self._retry_in_s, 1),
            # 策略从**泵实例**上取,不再是写死的字面量 "osci1t"。在此之前这个字段
            # 看起来是动态的,而任何时候都只会是那一个值 —— 于是「切到 2T 了吗」
            # 这个问题在 API 上根本问不出来。降级(要 2T、模块没开、退回 1T)在
            # ``detail`` 里说,一次静默降级和一次故障一样难查。
            "strategy": self._pump_strategy(cfg),
            "hr_available": getattr(self._pump, "hr_available", None),
            "fs_hz": float(cfg.get("fs_hz") or 0.0),
            "channel_name": str(cfg.get("channel_name") or ""),
            "n_buffer": int(cfg.get("n_buffer") or 0),
            # Commissioning facts that can ONLY be learnt on the instrument.
            # Both are already known from the acquisition path, so exposing them
            # costs no extra TCP — which matters, because the moment someone
            # wants them is while an experiment is running.
            "rt_freq_hz": float(cfg.get("rt_freq_hz") or 0.0),
            "timebases_s": [float(v) for v in (cfg.get("timebases_s") or [])],
            "timebase_index": int(cfg.get("timebase_index", -1)),
            # 时基表的自校验结论。仪器自报的档数与数组长度对不上时,采样率与
            # 「这台机器还有没有更快的档」这两个结论都要先怀疑它。
            "timebase_check": str(cfg.get("timebase_check") or ""),
            "pump_stats": dict(getattr(self._pump, "stats", {}) or {}),
            "segment_seconds": float(th.cm_segment_s),
            "connected": bool(pool is not None and _comms_healthy(pool)),
            "segments_done": self._segments_done,
            "gaps_total_s": round(self._gaps_total_s, 3),
            "last_segment_ts": self._last_segment_ts,
            "retention_hours": float(th.cm_keep_hours),
            "retention_gb": float(th.cm_keep_gb),
            # 辅助通道（Z / qPlus 振幅）。零 TCP —— 采样器只交回它上一次采到的东西。
            "aux": (self._aux.snapshot() if self._aux is not None else None),
        }

    def _pump_strategy(self, cfg: dict) -> Optional[str]:
        """哪个示波器方言正在跑。``None`` = 泵还没配起来（没配起来就没有事实）。"""
        if not cfg:
            return None
        return str(getattr(self._pump, "STRATEGY", "osci1t"))

    # ── main loop ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                th = self._th()
                if not th.enabled:
                    self._set_state(STATE_DISABLED, "监控已在设置中关闭")
                    self._stop.wait(_DISABLED_POLL_S)
                    continue

                pool = self._pool_getter()
                if pool is None:
                    self._set_state(STATE_NO_POOL, "尚未连接 Nanonis")
                    self._stop.wait(_PAUSED_POLL_S)
                    continue
                if not _comms_healthy(pool):
                    # Breaker open: park with zero TCP until it closes.
                    self._set_state(STATE_COMMS_DOWN, "TCP 通信已熔断,等待恢复")
                    self._stop.wait(_PAUSED_POLL_S)
                    continue

                self._pump_until_interrupted(th)
            except Exception:  # noqa: BLE001 — the loop must outlive any single fault
                logger.debug("current monitor loop error (retrying)", exc_info=True)
                self._set_state(STATE_PAUSED, "内部错误,稍后重试")
                self._stop.wait(_PAUSED_POLL_S)

    def _warm_up(self) -> None:
        """Trigger the lazy imports before the first segment arrives.

        scipy, the FFT path and the store's first write cost ~800 ms between
        them the first time they are touched. Paying that while a segment is
        already in hand costs sixteen oscilloscope buffers; paying it here costs
        nothing, because the pump has not started yet.
        """
        if getattr(self, "_warmed", False):
            return
        self._warmed = True
        try:
            import numpy as np

            from mast.monitoring import features as F
            probe = np.zeros(512, dtype=np.float64)
            F.compute_segment_features([probe], 20000.0)
            self._store_getter().storage_stats()
        except Exception:  # noqa: BLE001 — warm-up is an optimisation only
            logger.debug("current monitor warm-up skipped", exc_info=True)

    def _pump_until_interrupted(self, th) -> None:
        from mast.monitoring.pump import PumpPaused, PumpUnavailable, make_pump

        if self._pump is None:
            self._set_state(STATE_PROBING, "正在探测示波器模块")
            self._warm_up()
            self._pump = make_pump(self._pool_getter,
                                   segment_s=th.cm_segment_s,
                                   target_fs_hz=th.cm_target_fs_hz,
                                   strategy=th.acquisition_strategy,
                                   window_target_s=th.cm_2t_window_s)
        if self._engine is None:
            from mast.monitoring.alerts import AlertEngine
            self._engine = AlertEngine(self._th)
        if self._aux is None:
            from mast.monitoring.aux_channels import AuxSampler
            self._aux = AuxSampler(self._pool_getter, self._th)

        try:
            # 降级说明跟着 RUNNING 一起报出去:要的是 Osci2T、模块没开、退回了
            # Osci1T —— 采集照常,但「为什么往返还是那么多」只能在这里读到。
            # 一次静默降级和一次故障一样难查。
            note = str(getattr(self._pump, "fallback_note", "") or "")
            for segment in self._pump.pump_segments(self._stop,
                                                    on_idle=self._pump_idle):
                if self._state != STATE_RUNNING:
                    self._set_state(STATE_RUNNING, note)
                self._on_segment(segment)
        except PumpUnavailable as exc:
            # Expected on the bundled simulator, which loads neither scope.
            module = str(getattr(self._pump, "MODULE_NAME", "Osci1T"))
            self._pump = None
            self._set_state(
                STATE_UNAVAILABLE,
                f"示波器模块({module})不可用——仿真器上属正常现象",
                retry_in_s=_UNAVAILABLE_RETRY_S)
            logger.info("current monitor: %s unavailable (%s); re-probing in %.0fs",
                        module, exc, _UNAVAILABLE_RETRY_S)
            self._stop.wait(_UNAVAILABLE_RETRY_S)
        except PumpPaused as exc:
            self._set_state(STATE_PAUSED, str(exc), retry_in_s=_PAUSED_POLL_S)
            self._stop.wait(_PAUSED_POLL_S)

    # ── 搭车采样（辅助通道） ─────────────────────────────────────────────────

    def _pump_idle(self, budget_s: float) -> None:
        """泵在等缓冲刷新，把这段空闲让给辅助通道。**全程吞异常。**

        为什么辅助通道搬到这里来 —— 它原来只在 :meth:`_on_segment` 里有机会，
        而段边界一秒只来 0.75 次（1.024 s 采集 + 0.31 s 特征提取）。也就是说
        **把 ``cm_aux_interval_s`` 调到 0.2 秒并不会得到 5 Hz**：那个设置是节流的
        上限，不是驱动采样的时钟，调到比段周期还短只等于「每段都采」。用户
        要的提速，必须先有更多的采样机会才谈得上。

        搬到这里之后，三条让 1 Hz 当初站得住的性质**一条都没变**：

        * **零新增常驻线程**。这仍然是监控服务自己的 worker 线程，
          start/stop/reconnect 生命周期照旧（孤儿线程撞正在拆的 Nanonis 端口，
          这个仓库付过学费）。
        * **与监控主环零争锁**。同一条线程 ——「泵和它抢 data role」在物理上
          不成立，一条线程不会阻塞在自己身上。
        * **与急停看门狗零争锁**。看门狗跑在 ``role="monitor"``（0.5 s 一次
          ``Current_Get``），E-STOP 走 ``role="emergency"``。锁是**按 role 分的**，
          我们从头到尾只碰 ``data``。

        变的是一条，如实记下来：``segment_id`` 现在指**上一段已落库的**段落，
        而不是紧接着要落库的那一段。方向反了，含义没变 —— 每个 aux 样本仍然
        贴着它旁边的那段电流，「电流变噪的那一刻 Z 在做什么」照样问得出来。
        """
        th = self._th()
        if not th.aux_enabled or self._aux is None:
            return
        try:
            ctx = self._context_labels()
            self._handle_aux(self._store_getter(), th, ctx, self._last_seg_id,
                             self._is_suppressed(ctx), budget_s=budget_s)
        except Exception:  # noqa: BLE001 — 辅助通道绝不反噬电流采集
            logger.debug("idle aux sample failed (swallowed)", exc_info=True)

    # ── per segment ─────────────────────────────────────────────────────────

    #: 活跃基线的重查间隔。基线由人发起的表征产生，切换一份基线是一个刻意的
    #: 动作，30 s 的滞后没有代价；而每段查一次库会把 SQLite 的锁放到采集线程
    #: 的关键路径上，那是有代价的。
    _BASELINE_REFRESH_S = 30.0

    def _refresh_baseline(self, store):
        """当前活跃基线的 sigma 模型（带缓存）。读不到就是 None。

        **读不到与「没有基线」在这里合流是刻意的**：两种情况下判据都该回到固定
        阈值，而不是停止判断。区别记在日志里，不记在返回值里——返回一个「有基线
        但用不了」的对象只会让每个调用方各自再判一次它能不能用。
        """
        now = time.time()
        if now - self._baseline_checked_at < self._BASELINE_REFRESH_S:
            return self._baseline_model
        self._baseline_checked_at = now
        try:
            row = store.active_baseline() if store is not None else None
        except Exception:  # noqa: BLE001 — 一个观察者不该因为查不到基线而中断采集
            logger.debug("active_baseline lookup failed (swallowed)", exc_info=True)
            return self._baseline_model
        if not row:
            if self._baseline_id is not None:
                logger.info("current monitor: 基线已停用，rms_high 回到固定阈值")
            self._baseline_model, self._baseline_id = None, None
            return None
        from mast.monitoring.baseline import SigmaModel
        model = SigmaModel.from_dict(row.get("sigma_model"))
        if model is None and row.get("id") != self._baseline_id:
            logger.warning("current monitor: 基线 %s 没有可用的 sigma 模型，"
                           "rms_high 走固定阈值", row.get("id"))
        if row.get("id") != self._baseline_id:
            logger.info("current monitor: rms_high 改用基线 %s (%s)",
                        row.get("id"), row.get("label") or "")
        self._baseline_model = model
        self._baseline_id = row.get("id")
        return model

    def baseline_status(self) -> dict:
        """``/status`` 里的基线块 —— 用户看「现在比基线差多少」的地方。"""
        out: dict = {"active_id": self._baseline_id, "available": False}
        model = self._baseline_model
        if model is None:
            out["detail"] = ("没有活跃基线，噪声判据用的是固定阈值 cm_rms_warn_a。"
                             "跑一次 CharacteriseCurrentNoise 建立基线。")
            return out
        out.update(available=True, sigma_model=model.to_dict())
        try:
            store = self._store_getter()
            row = store.active_baseline() if store else None
            if row:
                out["label"] = row.get("label")
                out["ts"] = row.get("ts")
                out["conditions"] = row.get("conditions")
                out["white_model"] = row.get("white_model")
                # 极性检查结果必须穿过 store、service 和 schema。
                out["polarity"] = row.get("polarity")
                out["bias_magnitude"] = row.get("bias_magnitude")
        except Exception:  # noqa: BLE001
            logger.debug("baseline_status row lookup failed", exc_info=True)
        return out

    def baseline_check(self) -> dict:
        """当前最新一段对活跃基线的比值。判不了就说为什么，不给绿色。"""
        from mast.monitoring.baseline import compare, condition_diff
        store = self._store_getter()
        model = self._refresh_baseline(store)
        latest = None
        try:
            rows, _total, _t = store.features_query(limit=1, offset=0) if store else ([], 0, False)
            # features_query 按时间升序，要最新一行得反着取
            rows2, _t2, _th2 = store.features_query(
                since=time.time() - 120.0, limit=200) if store else ([], 0, False)
            latest = (rows2 or rows or [None])[-1]
        except Exception:  # noqa: BLE001
            logger.debug("baseline_check features lookup failed", exc_info=True)
        if not latest:
            return {"judged": False, "reason": "最近两分钟没有已落库的段"}
        row = None
        try:
            row = store.active_baseline() if store else None
        except Exception:  # noqa: BLE001
            pass
        diff = condition_diff((row or {}).get("conditions"), self._context_conditions())
        v = compare(latest.get("rms_detrended_a"), latest.get("mean_a"), model,
                    baseline_id=(row or {}).get("id"),
                    conditions_match=(not diff) if row else None,
                    condition_diff=diff)
        out = v.to_dict()
        out["seg_id"] = latest.get("segment_id")
        out["ts"] = latest.get("t_start")
        out["scanning"] = latest.get("ctx_scanning")
        return out

    def _context_conditions(self) -> dict:
        """当前的条件快照，与基线存的那份同一套键（baseline.CONDITION_KEYS）。

        读不到的键留 ``None`` —— :func:`baseline.condition_diff` 把「任一侧未知」
        报成未知而不是相同，所以这里宁可缺，不可猜。
        """
        out: dict = {}
        try:
            from mast.core import instrument_profile as ip
            out["tip_id"] = ip.get_config("active_tip_id", None)
        except Exception:  # noqa: BLE001
            out["tip_id"] = None
        try:
            pump = self._pump
            out["fs_hz"] = float(getattr(pump, "fs_hz", 0.0) or 0.0) or None
        except Exception:  # noqa: BLE001
            out["fs_hz"] = None
        return out

    def _on_segment(self, segment) -> None:
        import numpy as np

        from mast.monitoring import features as F

        th = self._th()
        store = self._store_getter()
        ctx = self._context_labels()

        params = F.FeatureParams(sat_current_a=th.cm_sat_current_a)
        feats = F.compute_segment_features(segment.runs, segment.fs_hz, params)
        if not feats:
            return
        feats["t_start"] = segment.t_start

        suppressed = self._is_suppressed(ctx)
        if suppressed:
            # Also pin the surrounding window: this is exactly the data a future
            # tip classifier needs, and it is about to be overwritten by more
            # boring segments.
            self._pin_event_window(store, th, segment.t_start,
                                   f"skill:{ctx.get('ctx_skill') or 'unknown'}")

        # A segment that is mostly hole is not a measurement. RoleBusy back-off
        # can book seconds of gap into a one-second segment (a scan grabbing the
        # data role), leaving a handful of real samples; three of those must not
        # be able to confirm a CRITICAL between them.
        span = max(1e-9, float(segment.t_end - segment.t_start))
        mostly_gap = float(segment.gap_s or 0.0) > 0.5 * span

        # 仪器自己会告诉我们前放满量程是多少 —— 只要它贴过一次轨。
        self._cross_check_rail(feats, th)

        verdict = self._engine.evaluate(feats, ctx,
                                        baseline=self._refresh_baseline(store))
        actionable = (th.alerts_enabled and not suppressed and not mostly_gap)
        if actionable:
            confirmed = self._engine.confirm(verdict)
        else:
            # Break the streak rather than skipping the machine entirely. Simply
            # not calling confirm() would let a saturation before tip-shaping and
            # one after it count as consecutive; passing the real verdict through
            # would be worse, because confirm() has side effects — it would arm
            # the cool-down on an alert we deliberately did not raise, and the
            # next genuine occurrence would be silently swallowed.
            from mast.monitoring.alerts import Verdict
            confirmed = self._engine.confirm(Verdict("ok"))

        # `verdict.level` is already "suppressed" when the only rules that fired
        # were context-suppressed ones (scanning — see alerts.SCAN_SUPPRESSED_RULES);
        # the skill-based `suppressed` flag overrides everything, as before.
        level = "suppressed" if suppressed else verdict.level
        if mostly_gap and level != "suppressed":
            level = "unknown"       # too little data to stand behind a verdict
        if level == "critical_candidate":
            level = "warn"          # not yet confirmed; stored as advisory

        npy_path, npy_bytes = self._persist_samples(store, segment)
        env = F.envelope(segment.samples, segment.fs_hz, params.env_buckets_per_s)
        seg_id = store.add_segment(
            {"t_start": segment.t_start, "t_end": segment.t_end,
             "osci_t0": segment.osci_t0, "fs_hz": segment.fs_hz,
             "n_samples": segment.n_samples, "n_runs": len(segment.runs),
             "gap_s": segment.gap_s, "discontinuity": segment.discontinuity,
             "npy_path": npy_path, "npy_bytes": npy_bytes,
             "channel_name": segment.channel_name, "source": segment.source,
             "pinned": bool(suppressed or time.time() < self._pin_until)},
            env.tobytes(), 1.0 / max(1, params.env_buckets_per_s),
        )

        fired_rule = None
        if actionable:
            fired_rule = self._handle_alerts(store, th, segment, feats, verdict,
                                             seg_id, confirmed)
            if fired_rule:
                level = "critical"

        # "照记不报": a rule we declined to judge is still written down, with the
        # reason. Without it a stored `suppressed` row cannot answer "suppressed
        # for WHAT", and the next person calibrating this rig would have to guess.
        extra = self._suppression_extra(verdict, suppressed, ctx)
        store.add_features(seg_id or 0, feats, ctx, level, extra=extra)

        self._segments_done += 1
        self._gaps_total_s += float(segment.gap_s or 0.0)
        self._last_segment_ts = segment.t_end
        self._publish_segment(seg_id, segment, feats, level, ctx, extra)
        # 辅助通道在段落**已经落库之后**才轮到 —— 一次慢/失败的 Signals_ValsGet
        # 绝不能让一段电流丢掉。它自己吞异常，见 _handle_aux。
        self._handle_aux(store, th, ctx, seg_id, suppressed)
        self._last_seg_id = seg_id
        self._maybe_sweep(store, th)
        self._offer_to_env_history(segment, feats, ctx)

    @staticmethod
    def _suppression_extra(verdict, skill_suppressed: bool,
                           ctx: dict) -> Optional[dict]:
        """``extra_json`` for one features row, or ``None`` when nothing was
        suppressed. Scalars and short strings only — this column is read back by
        the calibration tool, not by a renderer."""
        parked = list(getattr(verdict, "suppressed_rules", ()) or ())
        if not parked and not skill_suppressed:
            return None
        out: dict = {}
        if parked:
            out["suppressed_rules"] = parked
            out["suppressed_by"] = "scanning"
        if skill_suppressed:
            # The skill gate wins on the row's level, so name it last — whoever
            # reads this row sees the reason that actually decided the verdict.
            out["suppressed_by"] = "skill"
            held = str(ctx.get("ctx_skill") or "")
            out["suppressed_skill"] = held or str(ctx.get("ctx_afterglow_skill") or "")
            if not held:
                # A WEAKER claim than "a skill was running", and it has to look
                # different in the record: nobody held the token, we are only
                # inside the wake of one that just finished. If afterglow ever
                # starts hiding real events, this is the field that shows it.
                out["suppressed_by"] = "skill_afterglow"
                out["suppressed_afterglow"] = True
        return out

    def _handle_aux(self, store, th, ctx: dict, seg_id, suppressed: bool,
                    *, budget_s: float | None = None) -> None:
        """采一次 Z / qPlus 振幅 / Δf，落库、判级、发布。**全程吞异常。**

        两个调用点，同一条实现：

        * :meth:`_on_segment` —— 段落刚落库，泵正停在 0.31 s 的特征提取间隙里。
          不给 budget：这里的空闲远大于一次采样。
        * :meth:`_pump_idle` —— 泵在等下一次缓冲刷新（约 47 ms）。给 budget，
          采样器据此把**锁等待**压进去；压不进去就跳过这一拍。

        为什么不另起一条线程：见 ``monitoring/aux_channels.py`` 的模块 docstring。
        一句话 —— 零新增常驻线程（这个仓库为孤儿后台线程撞正在拆的 Nanonis 端口
        付过学费），而且和泵同线程意味着两者**不可能**互相抢 data role。

        采样器自己保证不阻塞（角色锁最多等 0.25 s，且不超过 budget，忙就跳过），
        因此锁就绪后还需一次 TCP 往返；具体耗时取决于目标仪器与连接状态。
        """
        aux = self._aux
        if aux is None:
            return
        try:
            sample = aux.maybe_sample(ctx, segment_id=seg_id, suppressed=suppressed,
                                      budget_s=budget_s)
            if sample is None:
                return
            store.add_aux_sample(sample.ts, sample.metrics, ctx,
                                 segment_id=seg_id, verdict=sample.verdict,
                                 rules=sample.rules)
            if th.aux_alerts_enabled and sample.rules:
                self._emit_aux_warns(store, aux, sample, seg_id)
            self._publish_aux(sample)
        except Exception:  # noqa: BLE001 — 辅助通道绝不反噬电流采集
            logger.debug("aux sample handling failed (swallowed)", exc_info=True)

    #: 交叉对账的判据。见 :meth:`_cross_check_rail`。
    #: `railed_frac` 是**最长一段逐位相同**的占比 —— 真实噪声下相邻采样几乎不可能
    #: 逐位相同,所以 0.9 已经是「这条信号被钉住了」而不是「很安静」。
    _RAIL_PINNED_FRAC = 0.9
    #: 钉住值低于配置满量程的这个比例,才算「配置明显偏高」。留 10% 给标定误差。
    _RAIL_MISMATCH_RATIO = 0.9

    def _cross_check_rail(self, feats: dict, th) -> None:
        """核对观测到的贴轨值与 cm_sat_current_a。railed_frac 不依赖配置的饱和阈值，因此可发现满量程配置偏高的情况；差异仅在边沿报告，不替代仪器标定。"""
        try:
            railed = feats.get("railed_frac")
            from mast.monitoring.alerts import _num

            pinned = max(abs(_num(feats.get("max_a")) or 0.0),
                         abs(_num(feats.get("min_a")) or 0.0))
            configured = float(th.cm_sat_current_a)
            if railed is None or float(railed) < self._RAIL_PINNED_FRAC:
                self._rail_mismatch_said = False
                return
            if pinned <= 0.0 or configured <= 0.0:
                return
            if pinned >= configured * self._RAIL_MISMATCH_RATIO:
                self._rail_mismatch_said = False   # 对得上 —— 边沿复位
                return
            if getattr(self, "_rail_mismatch_said", False):
                return
            self._rail_mismatch_said = True
            logger.warning(
                "电流信号被钉在 %.6e A(段内最长 %.0f%% 逐位相同),而配置的前放满量程 "
                "cm_sat_current_a = %.6e A —— **相差 %.1f 倍**。\n"
                "仪器实际的满量程很可能是前者:贴轨时读数逐位不变,那个值就是轨本身。\n"
                "后果:`saturation` 判据用的是配置值,所以它**可能永远不会响**,"
                "而看门狗的退针阈值也跟着它，可能漏掉实际饱和。\n"
                "请到设置里核对 cm_sat_current_a 是否等于本机前置放大器满量程。",
                pinned, float(railed) * 100.0, configured,
                (configured / pinned) if pinned else float("inf"))
        except Exception:  # noqa: BLE001 — 对账绝不能反噬采集
            logger.debug("rail cross-check failed (swallowed)", exc_info=True)

    def _emit_aux_warns(self, store, aux, sample, seg_id) -> None:
        """把 aux 的 WARN 落进**同一张** alerts 表（rule 名不同，历史 UI 免费显示）。

        故意全是 WARN，一条 CRITICAL 都没有：能 halt 正在跑的复合技能的规则必须是
        「关于仪器的陈述、且能从单段自证」，而这两路在本机一个阈值都还没标定过。
        想主动问「现在撞针了没有」，``CheckTipCrashByAmplitude`` 一直都在。
        """
        from mast.monitoring.aux_channels import summarize_aux_zh

        for rule in sample.rules:
            if not aux.should_emit(rule):
                continue
            summary = summarize_aux_zh(rule, sample.metrics, sample.detail)
            alert_id = store.add_alert(ts=sample.ts, level="warn", rule=rule,
                                       summary_zh=summary, segment_id=seg_id,
                                       features=sample.detail)
            self._publish_alert(alert_id, "warn", rule, summary, None, seg_id)

    def _publish_aux(self, sample) -> None:
        """One scalar summary per aux sample (5 Hz at the factory interval).

        5 Hz 而不是抽稀后再发：前端对 ``kind == "aux"`` 只做 ``setQueryData``
        就地打补丁，**不发任何请求**（只有 ``kind == "segment"`` 会触发一次节流
        过的 invalidate）—— 所以提速在这条路上不会变成请求风暴。而这些事件正是
        「进针时振幅归零」这类要靠密采样才看得见的现象的载体，抽稀等于把提速
        的成果在最后一步扔掉。

        Its own ``kind`` rather than extra fields on the ``segment`` event: the
        frontend's ``patchStatusFromWsEvent`` rebuilds ``latest`` wholesale from
        a ``segment`` payload, and widening that payload would put aux numbers
        into the current-feature map.
        """
        m = sample.metrics
        self._publish({
            "kind": "aux",
            "ts": float(sample.ts),
            "seg_id": int(seg) if (seg := sample.segment_id) else 0,
            "z_m": _opt(m.get("z_m")),
            "z_drift_m_per_s": _opt(m.get("z_drift_m_per_s")),
            "amp_m": _opt(m.get("amp_m")),
            "amp_frac_of_baseline": _opt(m.get("amp_frac_of_baseline")),
            "df_hz": _opt(m.get("df_hz")),
            "level": sample.verdict,
            "rules": list(sample.rules),
        })

    def _offer_to_env_history(self, segment, feats: dict, ctx: dict) -> None:
        """Hand the segment to the environment-history recorder, if one exists.

        Deliberately LAST and deliberately swallowed. The segment is already in
        the store by the time we get here, so a recorder that throws, hangs on
        an import or does not exist at all costs this subsystem nothing. The
        recorder itself sub-samples (one FFT per ~10 s by default) — it does not
        get to spend this thread's time on every segment.
        """
        try:
            from mast.envhistory.recorder import get_recorder
            rec = get_recorder()
            if rec is not None:
                rec.on_current_segment(segment, feats, ctx)
        except Exception:  # noqa: BLE001 — environment history never bites back
            logger.debug("env-history hook failed (swallowed)", exc_info=True)

    def _persist_samples(self, store, segment) -> tuple[Optional[str], int]:
        """Write the raw segment to .npy. Failure loses samples, not the row."""
        import numpy as np

        from mast.monitoring.store import segment_npy_path
        try:
            path = segment_npy_path(store.data_dir, segment.t_start, segment.fs_hz)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, segment.samples.astype(np.float32))
            return str(path), int(path.stat().st_size)
        except Exception:  # noqa: BLE001 — features still get stored
            logger.debug("could not persist segment samples", exc_info=True)
            return None, 0

    def _handle_alerts(self, store, th, segment, feats, verdict, seg_id,
                       rule: Optional[str]) -> Optional[str]:
        """Act on an already-confirmed verdict.

        ``confirm()`` is called by the caller for every segment so the streak
        stays honest; this only decides what to do with the outcome.
        """
        from mast.monitoring.alerts import emit_critical, summarize_zh

        if rule:
            summary = summarize_zh(rule, feats, verdict.detail)
            emitted = emit_critical(rule, summary, feats, None, seg_id,
                                    self._scan_id_getter() or "")
            alert_id = store.add_alert(ts=time.time(), level="critical", rule=rule,
                                       summary_zh=summary, segment_id=seg_id,
                                       evidence_png=None, features=verdict.detail,
                                       emitted_buffer=emitted)
            self._pin_event_window(store, th, segment.t_start, f"alert:{rule}")
            self._publish_alert(alert_id, "critical", rule, summary, None, seg_id)
            logger.warning("current monitor CRITICAL [%s]: %s", rule, summary)
            if alert_id and self._may_render_evidence(th):
                self._render_evidence_async(store, segment, feats, rule, alert_id)
            return rule

        if verdict.level == "warn":
            for warn_rule in verdict.rules:
                if not self._engine.should_emit_warn(warn_rule):
                    continue
                summary = summarize_zh(warn_rule, feats, verdict.detail)
                alert_id = store.add_alert(ts=time.time(), level="warn",
                                           rule=warn_rule, summary_zh=summary,
                                           segment_id=seg_id,
                                           features=verdict.detail)
                self._publish_alert(alert_id, "warn", warn_rule, summary, None, seg_id)
        return None

    def _render_evidence_async(self, store, segment, feats: dict, rule: str,
                               alert_id: int) -> None:
        """Draw the trace+spectrum PNG off the acquisition thread.

        Measured at up to ~1 s (matplotlib's first render dominates), which is
        twenty oscilloscope buffers — stalling the pump for that long right when
        something is going wrong is the worst possible moment to stop recording.
        The alert is already published and stored; the picture is attached when
        it is ready. Rate-limited by the caller, so at most one of these threads
        exists at a time.
        """
        from mast.monitoring.alerts import render_evidence_png

        def _work() -> None:
            try:
                path = render_evidence_png(segment, feats, store.evidence_dir, rule)
                if path:
                    store.set_alert_evidence(alert_id, path)
            except Exception:  # noqa: BLE001 — evidence is a nicety
                logger.debug("async evidence render failed", exc_info=True)

        threading.Thread(target=_work, daemon=True,
                         name=f"cm-evidence-{alert_id}").start()

    def _may_render_evidence(self, th) -> bool:
        now = time.time()
        last = getattr(self, "_last_evidence_ts", 0.0)
        if now - last < th.cm_evidence_min_interval_s:
            return False
        self._last_evidence_ts = now
        return True

    def _pin_event_window(self, store, th, ts: float, reason: str) -> None:
        try:
            store.pin_range(ts - th.cm_pin_preroll_s, ts + th.cm_pin_postroll_s, reason)
            self._pin_until = max(self._pin_until, time.time() + th.cm_pin_postroll_s)
        except Exception:  # noqa: BLE001
            logger.debug("pin_range failed (swallowed)", exc_info=True)

    def _maybe_sweep(self, store, th) -> None:
        now = time.time()
        if now - self._last_sweep < _RETENTION_INTERVAL_S:
            return
        self._last_sweep = now
        try:
            store.retention_sweep(th.cm_keep_hours, th.cm_keep_gb)
        except Exception:  # noqa: BLE001
            logger.debug("retention sweep failed (swallowed)", exc_info=True)
        try:
            # aux 行按年龄单独滚删。刻意不并进上面的字节预算 —— 一行 aux 是二十个数，
            # 一年也才几十 MB，而段落是 288 MB/**小时**。合在一起会让磁盘满的时候
            # 删掉便宜的那个而不是贵的那个。
            store.sweep_aux(th.cm_aux_keep_hours)
        except Exception:  # noqa: BLE001
            logger.debug("aux sweep failed (swallowed)", exc_info=True)

    # ── context ─────────────────────────────────────────────────────────────

    def _context_labels(self) -> dict:
        """What the instrument was doing during this segment. Zero TCP.

        Read from the 1 Hz InstrumentState cache and the instrument-token
        snapshot — both are plain in-memory reads. Issuing our own hardware
        queries here would put the monitor onto the command path, which is
        exactly what a read-only observer must not do.
        """
        out: dict = {"ctx_skill": ""}
        try:
            state = self._state_getter()
            snap = state.snapshot() if state is not None else None
            if snap is not None:
                out.update({
                    "ctx_scanning": getattr(snap, "scan_running", None),
                    "ctx_bias_v": getattr(snap, "bias_v", None),
                    "ctx_setpoint_a": getattr(snap, "setpoint_a", None),
                    # The aux path's approach gate needs |I| vs setpoint to tell
                    # "the feedback is hunting for a junction" (= an approach,
                    # including one the operator started from the Nanonis panel)
                    # from "we are sitting in a junction". Nothing else reads it;
                    # it is not stored on the aux row, which keeps the current
                    # value on the segment row where it already lives.
                    "ctx_current_a": getattr(snap, "current_a", None),
                    "ctx_z_m": getattr(snap, "z_pos_m", None),
                    "ctx_zctrl_on": getattr(snap, "z_controller_on", None),
                    # 调制产生的设计内纹波须纳入上下文，避免 jump_burst 将其解释为接触不稳。
                    "ctx_lockin_on": getattr(snap, "lockin_mod_on", None),
                    "ctx_stale": getattr(snap, "stale", None),
                })
        except Exception:  # noqa: BLE001 — labels are optional, samples are not
            logger.debug("context snapshot failed", exc_info=True)
        try:
            from mast.core.instrument_lock import instrument_lock
            holder = instrument_lock().snapshot() or {}
            out["ctx_skill"] = str(holder.get("skill") or "")
        except Exception:  # noqa: BLE001
            logger.debug("instrument lock snapshot failed", exc_info=True)
        # Include the configured afterglow window; see _is_suppressed.
        out["ctx_afterglow_skill"] = self._afterglow_skill()
        return out

    # Transient effects can outlast a tool call, leaving no token holder during the settling tail. Remember the latest completed suppressible skill for cm_afterglow_s. Increasing the consecutive-segment count would also delay unrelated dangerous alerts; extending the shared instrument token would alter other consumers.

    def _note_skill_done(self, name: str) -> None:
        """Remember that a suppressible skill just finished. Cheap, no lock."""
        self._afterglow_until = time.time() + self._afterglow_window_s()
        self._afterglow_name = name

    def _afterglow_window_s(self) -> float:
        """Return the configured afterglow window. The default is an uncalibrated placeholder. Estimate a suitable value from the distribution of elapsed time between completed approach or withdrawal actions and the last subsequent spike-bearing segment."""
        try:
            return max(0.0, float(getattr(self._th(), "cm_afterglow_s", 5.0)))
        except Exception:  # noqa: BLE001
            return 5.0

    def _afterglow_skill(self) -> str:
        """The skill whose wake we are still in, or ``""``."""
        if time.time() < getattr(self, "_afterglow_until", 0.0):
            return str(getattr(self, "_afterglow_name", "") or "")
        return ""

    @staticmethod
    def _is_suppressed(ctx: dict) -> bool:
        """Is this segment inside a skill that is SUPPOSED to look violent?

        Two ways to qualify, and they are reported apart (``suppressed_skill``
        vs ``suppressed_afterglow``) so a later reader can tell "a skill was
        holding the token" from "a skill had just let go" — the second is a
        weaker claim and should not be able to hide behind the first.
        """
        skill = str(ctx.get("ctx_skill") or "").lower()
        if skill and any(p in skill for p in SUPPRESS_SKILL_PATTERNS):
            return True
        after = str(ctx.get("ctx_afterglow_skill") or "").lower()
        return bool(after) and any(p in after for p in SUPPRESS_SKILL_PATTERNS)

    # ── event bus ───────────────────────────────────────────────────────────

    def _publish(self, payload: dict) -> None:
        try:
            from mast.core.events import EventBus
            EventBus.get().publish_current_monitor(**payload)
        except Exception:  # noqa: BLE001 — telemetry is never load-bearing
            logger.debug("publish_current_monitor failed (swallowed)", exc_info=True)

    def _publish_segment(self, seg_id, segment, feats: dict, level: str,
                         ctx: dict, extra: dict | None = None) -> None:
        """One scalar summary per segment (~1 Hz). Arrays never ride the bus."""
        # `suppressed` used to mean exactly one thing (a tip-shaping skill is
        # running), so the UI hard-codes that sentence. It now has a second
        # cause, and a banner that says "修针 / 电脉冲窗口内" over a scanning
        # segment is a false statement about the instrument. Carry the reason so
        # the banner can stop guessing — see docs/v2/KNOWN_ISSUES.md §2.20.
        extra = extra or {}
        self._publish({
            "kind": "segment",
            "seg_id": int(seg_id) if seg_id else 0,
            "ts": float(segment.t_end),
            "fs_hz": float(segment.fs_hz),
            "rms_pa": _scale(feats.get("rms_detrended_a"), 1e12),
            "mean_na": _scale(feats.get("mean_a"), 1e9),
            "spike_sigma": _plain(feats.get("spike_max_sigma")),
            "rtn_score": _plain(feats.get("rtn_score")),
            "sat_frac": _plain(feats.get("sat_frac")),
            "level": level,
            "gap_s": float(segment.gap_s or 0.0),
            "ctx_scanning": bool(ctx.get("ctx_scanning") or False),
            "ctx_lockin_on": bool(ctx.get("ctx_lockin_on") or False),
            "ctx_skill": str(ctx.get("ctx_skill") or ""),
            # Comma-joined, NOT a list: the segment event is scalars-only (a
            # pinned invariant — the bus feeds a bounded replay buffer, and one
            # array per second evicts everything else). Same convention the
            # store already uses for the aux `rules` column.
            "suppressed_rules": ",".join(
                str(r) for r in (extra.get("suppressed_rules") or [])),
            "suppressed_by": str(extra.get("suppressed_by") or ""),
        })

    def _publish_alert(self, alert_id, level: str, rule: str, summary: str,
                       png: str | None, seg_id) -> None:
        self._publish({
            "kind": "alert",
            "alert_id": int(alert_id) if alert_id else 0,
            "level": level, "rule": rule, "summary_zh": summary,
            "evidence_png": png or "",
            "seg_id": int(seg_id) if seg_id else 0,
        })

    def _set_state(self, state: str, detail: str = "",
                   retry_in_s: float = 0.0) -> None:
        """Publish only on transition — a state event per poll would swamp the
        100-event replay buffer and push out everything else."""
        if state == self._state and detail == self._detail:
            return
        self._state, self._detail, self._retry_in_s = state, detail, retry_in_s
        cfg = self._pump.config if self._pump is not None else {}
        self._publish({
            "kind": "status", "state": state, "detail": detail,
            "retry_in_s": float(retry_in_s),
            "strategy": self._pump_strategy(cfg),
            "hr_available": getattr(self._pump, "hr_available", None),
            "fs_hz": float(cfg.get("fs_hz") or 0.0),
            "channel_name": str(cfg.get("channel_name") or ""),
        })

    # ── skill-step subscription (pins the tail of an event window) ──────────

    def _subscribe_skill_steps(self) -> None:
        """SKILL_STEP fires when a skill FINISHES, so it cannot suppress in real
        time — the token snapshot does that. What it adds is the trailing pin:
        by the time a tip-shaping skill completes, the interesting segments are
        already written, and this makes sure they survive the retention sweep."""
        try:
            from mast.core.events import EventBus
            self._event_handle = EventBus.get().subscribe_with_id(self._on_bus_event)
        except Exception:  # noqa: BLE001
            logger.debug("could not subscribe to the event bus", exc_info=True)

    def _unsubscribe_skill_steps(self) -> None:
        handle, self._event_handle = self._event_handle, None
        if handle is None:
            return
        try:
            from mast.core.events import EventBus
            EventBus.get().unsubscribe(handle)
        except Exception:  # noqa: BLE001
            logger.debug("could not unsubscribe from the event bus", exc_info=True)

    def _on_bus_event(self, seq, event) -> None:
        try:
            from mast.core.events import EventType
            if getattr(event, "type", None) is not EventType.SKILL_STEP:
                return
            data = getattr(event, "data", None) or {}
            name = str(data.get("skill") or data.get("skill_name") or "").lower()
            if not name or not any(p in name for p in SUPPRESS_SKILL_PATTERNS):
                return
            # Same event, two jobs. Pinning keeps the segment as training data;
            # the afterglow keeps the NEXT few seconds from popping a
            # confirmation box at the operator. Reusing the subscription that
            # already exists is the whole reason this fix needs no new coupling.
            self._note_skill_done(name)
            th = self._th()
            self._pin_event_window(self._store_getter(), th, time.time(),
                                   f"skill_done:{name}")
        except Exception:  # noqa: BLE001 — a bus callback must never raise
            logger.debug("skill-step pin failed (swallowed)", exc_info=True)


def _comms_healthy(pool) -> bool:
    try:
        fn = getattr(pool, "comms_healthy", None)
        return bool(fn()) if callable(fn) else True
    except Exception:  # noqa: BLE001
        return True


def _plain(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


def _scale(v, factor: float) -> float:
    return _plain(v) * factor


def _opt(v) -> Optional[float]:
    """None-preserving float.

    Deliberately NOT ``_plain`` (which maps missing → 0.0). On the aux channels
    zero is a real, meaningful value — a collapsed qPlus amplitude IS ~0 — so
    coercing "not measured" to 0.0 would publish a crash that never happened.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


# ── process-global handle ────────────────────────────────────────────────────

_SERVICE: CurrentMonitorService | None = None
_SERVICE_LOCK = threading.Lock()


def get_service() -> CurrentMonitorService | None:
    return _SERVICE


def start_service(app) -> CurrentMonitorService | None:
    """Create (if needed) and start the daemon for a CoreRuntime. Never raises.

    ``app`` is the CoreRuntime; the getters are closures over it so a reconnect
    that swaps the pool is picked up without re-wiring.
    """
    global _SERVICE
    try:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = CurrentMonitorService(
                    pool_getter=lambda: getattr(app, "_pool", None),
                    state_getter=lambda: getattr(app, "_state", None),
                    scan_id_getter=lambda: str(getattr(app, "_current_scan_id", "") or ""),
                )
            svc = _SERVICE
        svc.start()
        return svc
    except Exception:  # noqa: BLE001 — monitoring must never break boot
        logger.warning("current monitor did not start", exc_info=True)
        return None


def stop_service(join_timeout: float = 5.0) -> None:
    """Stop the daemon if running. Never raises."""
    try:
        svc = _SERVICE
        if svc is not None:
            svc.stop(join_timeout=join_timeout)
    except Exception:  # noqa: BLE001
        logger.debug("current monitor stop failed (swallowed)", exc_info=True)


def set_service_for_test(svc: CurrentMonitorService | None) -> None:
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = svc


__all__ = ["CurrentMonitorService", "get_service", "start_service",
           "stop_service", "set_service_for_test", "SUPPRESS_SKILL_PATTERNS"]
