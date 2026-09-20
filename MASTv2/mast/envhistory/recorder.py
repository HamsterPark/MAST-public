"""环境历史记录器的协调器 —— 一个进程单例，把四条链路缝在一起。

它自己**没有常驻线程**。所有工作寄生在既有心跳上：

* 标量桶  ← :class:`mast.envhistory.sink.EnvHistorySink`，跑在 EnvironmentMonitor
  的 2 秒线程上；
* I 噪声谱 ← :meth:`on_current_segment`，跑在 CurrentMonitorService 的段流上；
* 原始行清扫 ← 到期时起一个**有界单飞**线程，干完就退；
* Z 噪声谱  ← 到期时起一个**有界单飞** burst 线程，干完就退。

后两者为什么是"起了就退"而不是常驻：这个仓库已经为常驻后台线程付过学费 ——
``ExperimentalMonitor`` 至今没挂 reconnect，重连时它会拿着一个已经关掉的
ConnectionPool 空转并撞正在重建的 Nanonis 端口。不养常驻线程，这类 bug 在结构
上就不存在；代价只是每次到期多花几十微秒起一个线程。
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

from mast.envhistory.spectra import SpectrumAccumulator

logger = logging.getLogger(__name__)

#: 清扫/burst 线程的 join 上限。停机路径上的无界 join 会把"线程暂时卡住"升级
#: 成"用户去强杀进程",而强杀 mid-TCP 会永久损坏 Nanonis 端口。
_JOIN_S = 5.0

#: 段里缺口占比超过这个数就不拿它算谱。RoleBusy 退避能把几秒的空洞记进一个
#: 一秒的段,剩下几个真样本 —— 那不是一次测量。
_MAX_GAP_FRAC = 0.2


class _AnyEvent:
    """把两个 Event 当成一个用：任一被 set 就算 set，wait 取较早醒的那个。

    :func:`~mast.envhistory.zburst.run_z_burst` 只接受一个 stop 对象，而 burst
    有两个理由要停（进程关机 / 重连要拆 pool）。用一个薄壳而不是把两个 Event
    合并成一个，是因为它们的生命周期不同：关机是终态，重连不是。
    """

    def __init__(self, *events) -> None:
        self._events = tuple(events)

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)

    def wait(self, timeout: float | None = None) -> bool:
        # 单个 Event 时直接转发（常见情形，零开销）。多个时退化成短轮询：
        # burst 的等待都是毫秒级的，多睡最多 50 ms 不影响相位锁定。
        if len(self._events) == 1:
            return self._events[0].wait(timeout)
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            if self.is_set():
                return True
            if deadline is None:
                self._events[0].wait(0.05)
                continue
            remain = deadline - time.monotonic()
            if remain <= 0:
                return self.is_set()
            self._events[0].wait(min(0.05, remain))


class EnvHistoryRecorder:
    """协调器。所有公开方法都吞异常：记录器坏掉的正确表现是"没有历史"。"""

    def __init__(self, *, store_getter: Callable | None = None,
                 thresholds_getter: Callable | None = None,
                 storage_getter: Callable | None = None,
                 pool_getter: Callable | None = None,
                 state_getter: Callable | None = None,
                 scope_provider: Callable | None = None) -> None:
        self._store_getter = store_getter
        self._th_getter = thresholds_getter
        self._storage_getter = storage_getter
        self._pool_getter = pool_getter
        self._state_getter = state_getter
        self._scope = scope_provider

        from mast.envhistory.sink import EnvHistorySink
        self.sink = EnvHistorySink(
            store_getter=store_getter,
            thresholds_getter=thresholds_getter,
            scope_provider=scope_provider,
            state_getter=state_getter,
            on_tick=self.on_tick,
        )

        self._spec_current = SpectrumAccumulator("current", unit="A^2/Hz")
        self._spec_z = SpectrumAccumulator("z", unit="m^2/Hz")
        self._spec_lock = threading.Lock()

        self._lock = threading.Lock()
        self._sweep_thread: threading.Thread | None = None
        self._burst_thread: threading.Thread | None = None
        self._stopping = threading.Event()
        # Separate from _stopping on purpose: a reconnect must abort an in-flight
        # burst (it is holding a pool that is about to be torn down) WITHOUT
        # retiring the recorder — the environment monitor keeps running across a
        # reconnect and its buckets must keep accumulating.
        self._burst_abort = threading.Event()

        #: 「上一次真的跑过」的墙钟时刻；**0.0 = 从没跑过**。
        #:
        #: 初值 0.0 让周期闸门（``now - _last_*_at < interval``）在第一次必然放行
        #: ——「**第一次 tick 就跑，之后才按间隔**」是刻意的设计，不是漏算：
        #: ``test_tick_runs_the_sweep_once_per_interval`` 用 1e9 秒的间隔断言第一次
        #: 仍然会跑。别把它改成 ``max(上次, 启动时刻)`` —— 2026-08-02 试过，四个
        #: 测试当场变红，而那四个测试才是对的。
        self._last_sweep_at: float = 0.0
        self._last_sweep_result: dict = {}
        self._last_z_at: float = 0.0
        self._last_z_result: dict = {}
        self._last_spectrum_at: dict[str, float] = {}
        self._insufficient_quiet = False
        self._spectra_written = 0

    # ── 段流钩子（跑在 CurrentMonitorService 线程上） ─────────────────

    def on_current_segment(self, segment, feats: dict | None = None,
                           ctx: dict | None = None) -> None:
        """收一段电流。**永不抛** —— 调用方是刚验收的采集路径。

        注意这里只做两件很便宜的事：按子采样间隔算一次 PSD（~ms，每 10 s 一
        次），以及到点时把攒好的 median 写一行。绝不在这条线程上做绘图、
        文件 IO 之外的任何慢事。
        """
        try:
            th = self._th()
            if th is None or not th.spectra_enabled:
                return
            from mast.envhistory.quiet import admits, classify
            now = time.time()
            quietness = classify(ctx)
            with self._spec_lock:
                acc = self._spec_current
                if admits(quietness) and acc.should_accum(
                        now, th.eh_spectrum_accum_every_s):
                    self._accumulate_segment(acc, segment, now, th, ctx)
                snap = acc.maybe_emit(
                    now, interval_s=th.eh_spectrum_interval_s,
                    min_segments=th.spectrum_min_segments)
                # "攒不够安静段" 只能由**当前这个窗口**开着多久来判断,不能拿
                # "距上一条快照多久" —— 开机后还没有上一条，那个差值是 now 本身,
                # 于是第一段数据刚进来就会报一次不存在的异常。
                window_s = acc.stats(now).get("window_s") or 0.0
                self._insufficient_quiet = bool(
                    snap is None and acc.n_accum > 0
                    and window_s > 2.0 * float(th.eh_spectrum_interval_s)
                )
            if snap is not None:
                self._persist_spectrum(snap)
        except Exception:  # noqa: BLE001 — 环境记录绝不反噬采集
            logger.debug("on_current_segment failed (swallowed)", exc_info=True)

    def _accumulate_segment(self, acc: SpectrumAccumulator, segment,
                            now: float, th, ctx: dict | None) -> None:
        span = max(1e-9, float(getattr(segment, "t_end", 0.0))
                   - float(getattr(segment, "t_start", 0.0)))
        if float(getattr(segment, "gap_s", 0.0) or 0.0) > _MAX_GAP_FRAC * span:
            return
        if getattr(segment, "discontinuity", False):
            # 有人在底下改了示波器配置：这段的采样率标签未必还对得上样本。
            return
        runs = list(getattr(segment, "runs", None) or [])
        fs = float(getattr(segment, "fs_hz", 0.0) or 0.0)
        if not runs or fs <= 0:
            return
        from mast.monitoring.features import psd_of_runs
        freqs, psd = psd_of_runs(runs, fs)
        if len(freqs) < 4:
            return
        acc.add(freqs, psd, fs_hz=fs, ts=now, bins=th.spectrum_bins, ctx=ctx)

    # ── 心跳钩子（跑在 EnvironmentMonitor 线程上） ────────────────────

    def on_tick(self, now: float, ctx: dict | None = None,
                quietness: str = "unknown") -> None:
        """每个环境心跳查一次到期。全部是整数比较，真正的活在各自线程里。"""
        if self._stopping.is_set():
            return
        th = self._th()
        if th is None or not th.enabled:
            return
        self._maybe_sweep(now, th)
        self._maybe_z_burst(now, th, ctx, quietness)

    # ── 原始行清扫 ────────────────────────────────────────────────────

    def _maybe_sweep(self, now: float, th) -> None:
        if (now - self._last_sweep_at) < float(th.eh_sweep_interval_s):
            return
        with self._lock:
            if self._sweep_thread is not None and self._sweep_thread.is_alive():
                return          # 单飞：上一轮还没跑完就跳过这一轮
            self._last_sweep_at = now
            t = threading.Thread(target=self._sweep_body, args=(float(th.eh_raw_keep_days),),
                                 name="envhistory-sweep", daemon=True)
            self._sweep_thread = t
        t.start()

    def _sweep_body(self, keep_days: float) -> None:
        try:
            storage = self._storage()
            if storage is None or not hasattr(storage, "prune_environment_log"):
                return
            cutoff = (datetime.now() - timedelta(days=max(1.0, keep_days))).isoformat()
            deleted = storage.prune_environment_log(cutoff)
            self._last_sweep_result = {
                "at": time.time(), "cutoff": cutoff, "rows_pruned": int(deleted or 0),
            }
            if deleted:
                logger.info("环境原始读数清扫：删除 %d 行（早于 %s，告警行保留）",
                            int(deleted), cutoff[:19])
        except Exception:  # noqa: BLE001 — 清扫失败下一轮再来
            self._last_sweep_result = {"at": time.time(), "error": True}
            logger.debug("environment_log sweep failed", exc_info=True)

    # ── Z 噪声谱 burst ────────────────────────────────────────────────

    def _maybe_z_burst(self, now: float, th, ctx: dict | None,
                       quietness: str) -> None:
        if not th.z_enabled:
            return
        if (now - self._last_z_at) < float(th.eh_z_interval_s):
            return
        from mast.envhistory.quiet import admits
        if not admits(quietness):
            self._last_z_result = {"at": now, "skipped": "仪器不空闲"}
            return
        pool = self._pool()
        if pool is None:
            self._last_z_result = {"at": now, "skipped": "未连接 Nanonis"}
            return
        try:
            if not pool.comms_healthy():
                # 熔断开路时任何新调用都是喂毒 —— 零 TCP 跳过这一轮。
                self._last_z_result = {"at": now, "skipped": "TCP 已熔断"}
                return
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            if self._burst_thread is not None and self._burst_thread.is_alive():
                return
            self._last_z_at = now
            self._burst_abort.clear()
            t = threading.Thread(target=self._z_burst_body,
                                 args=(float(th.eh_z_burst_s), int(th.spectrum_bins),
                                       dict(ctx or {})),
                                 name="envhistory-zburst", daemon=True)
            self._burst_thread = t
        t.start()

    def abort_hardware_work(self) -> None:
        """Ask an in-flight Z burst to stop touching the instrument, now.

        Called on the reconnect path, BEFORE ``pool.close_all()``. The pool's
        own ``_closed`` flag would already stop the burst from reaching a
        socket, but leaving it to discover that through ten failed calls means
        ten calls issued against a pool being rebuilt. Asking is cheaper and
        does not retire the recorder the way :meth:`stop` does.
        """
        self._burst_abort.set()

    def _z_burst_body(self, burst_s: float, bins: int, ctx: dict) -> None:
        try:
            from mast.envhistory.zburst import run_z_burst
            snap = run_z_burst(self._pool_getter, stop=_AnyEvent(self._stopping,
                                                                self._burst_abort),
                               burst_s=burst_s, bins=bins, ctx=ctx)
            if snap is None:
                self._last_z_result = {"at": time.time(), "skipped": "本轮未取到数据"}
                return
            self._persist_spectrum(snap)
            self._last_z_result = {"at": time.time(), "n_segments": snap.n_segments,
                                   "fs_hz": snap.fs_hz}
        except Exception as exc:  # noqa: BLE001
            self._last_z_result = {"at": time.time(), "skipped": repr(exc)}
            logger.debug("z burst failed", exc_info=True)

    # ── 落盘 ──────────────────────────────────────────────────────────

    def _persist_spectrum(self, snap) -> None:
        store = self._store()
        if store is None:
            return
        eid, sid = self._scope_ids()
        rid = store.add_spectrum(
            ts=snap.ts, channel=snap.channel, span_s=snap.span_s,
            n_segments=snap.n_segments, fs_hz=snap.fs_hz,
            freqs=snap.freqs, psd=snap.psd, unit=snap.unit,
            quietness=snap.quietness, ctx=snap.ctx,
            ctx_stable=getattr(snap, "ctx_stable", None),
            experiment_id=eid, sample_id=sid)
        if rid:
            self._spectra_written += 1
            self._last_spectrum_at[snap.channel] = snap.ts

    # ── 生命周期 ──────────────────────────────────────────────────────

    def stop(self) -> None:
        """落定未满的桶，有界等待两个后台线程。永不抛。"""
        self._stopping.set()
        try:
            self.sink.flush()
        except Exception:  # noqa: BLE001
            pass
        for attr in ("_sweep_thread", "_burst_thread"):
            t = getattr(self, attr, None)
            if t is not None and t.is_alive():
                t.join(timeout=_JOIN_S)
                if t.is_alive():
                    logger.warning("envhistory: %s 未在 %.0fs 内退出（daemon，不阻塞进程退出）",
                                   t.name, _JOIN_S)

    # ── 自述 ──────────────────────────────────────────────────────────

    def status(self) -> dict:
        th = self._th()
        now = time.time()
        with self._spec_lock:
            spec = {
                "current": self._spec_current.stats(now),
                "z": self._spec_z.stats(now),
            }
        spec["current"]["last_ts"] = self._last_spectrum_at.get("current")
        spec["current"]["insufficient_quiet"] = bool(self._insufficient_quiet)
        spec["z"]["last_ts"] = self._last_spectrum_at.get("z")
        spec["z"]["last_result"] = dict(self._last_z_result)
        out: dict = {
            "enabled": bool(th.enabled) if th else False,
            "spectra_enabled": bool(th.spectra_enabled) if th else False,
            "z_enabled": bool(th.z_enabled) if th else False,
            "bucket_s": float(th.bucket_s) if th else 0.0,
            "sink": self.sink.stats(),
            "spectra": spec,
            "spectra_written": self._spectra_written,
            "sweep": {"last_at": self._last_sweep_at or None,
                      **dict(self._last_sweep_result)},
        }
        store = self._store()
        out["store"] = store.storage_stats() if store is not None else {}
        return out

    # ── 依赖解析（全部懒 + 吞异常） ───────────────────────────────────

    def _th(self):
        try:
            if self._th_getter is not None:
                return self._th_getter()
            from mast.envhistory.thresholds import get_env_history_thresholds
            return get_env_history_thresholds()
        except Exception:  # noqa: BLE001
            return None

    def _store(self):
        try:
            if self._store_getter is not None:
                return self._store_getter()
            from mast.envhistory.store import get_store
            return get_store()
        except Exception:  # noqa: BLE001
            return None

    def _storage(self):
        try:
            return self._storage_getter() if self._storage_getter else None
        except Exception:  # noqa: BLE001
            return None

    def _pool(self):
        try:
            return self._pool_getter() if self._pool_getter else None
        except Exception:  # noqa: BLE001
            return None

    def _scope_ids(self) -> tuple[str | None, str | None]:
        if self._scope is None:
            return None, None
        try:
            scope = self._scope()
            if scope and len(scope) >= 4:
                return (scope[2] or None), (scope[3] or None)
        except Exception:  # noqa: BLE001
            pass
        return None, None


# ── 进程级单例 ───────────────────────────────────────────────────────

_RECORDER: EnvHistoryRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def get_recorder() -> EnvHistoryRecorder | None:
    """当前记录器，**可能是 None**。

    刻意不自动创建：记录器需要 runtime 注入 storage / pool / state / scope 四个
    回调才有意义，凭空造一个只会得到一个什么都读不到的空壳。段流钩子按 None
    静默跳过。
    """
    return _RECORDER


def set_recorder(rec: EnvHistoryRecorder | None) -> None:
    global _RECORDER
    with _RECORDER_LOCK:
        _RECORDER = rec


__all__ = ["EnvHistoryRecorder", "get_recorder", "set_recorder"]
