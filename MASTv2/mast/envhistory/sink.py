"""EnvironmentMonitor 的历史 sink —— 把 2 秒读数流聚合成统计桶写进库。

注入式，与 :class:`mast.environment.csv_sink.EnvironmentCsvSink` 并列挂在
``EnvironmentMonitor(sinks=[...])`` 上。协议就是一个方法::

    sink.write(sensor_name, value, unit, status)

**这个方法永不抛、永不阻塞。** 环境监控线程的首要职责是发现真空/温度异常并
告警；写历史失败绝不能让它停摆。任何一次失败都让这个 sink 自禁 60 秒并只告警
一次，监控循环照常跑。

它顺带做两件事，理由都是"这里已经有一个 2 秒的心跳，不值得为它们另起线程"：

* 合成序列 ``instrument_quiet``（安静=1/否则=0，桶的 mean 即安静占空比）。它让
  任何一张历史图都能叠一条"这段时间仪器在不在被人动"的底色,也是事后核对
  ``tunnel_current`` 为什么缺了一段的依据。
* 每个心跳喊一次 ``on_tick()``,让 recorder 去检查"清扫到期了吗""Z 谱该采了
  吗"。到期检查本身是 µs 级的整数比较,真正的工作在它自己的有界线程里。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable

from mast.envhistory.buckets import BucketAccumulator

logger = logging.getLogger(__name__)

#: 写失败后静默多久再重试。与 csv_sink 取同一个数,理由相同。
_DISABLE_S = 60.0

#: 上下文快照的缓存寿命。比监控周期(2 s)短,所以每个心跳的第一个传感器会
#: 刷新它;比单次 read_all 的耗时长,所以同一轮里的七个传感器共用一份快照 ——
#: 否则同一秒内的读数可能被判成不同的安静态。
_CTX_TTL_S = 1.0

#: 默认只对隧道电流开安静门。温度/真空/液氦/磁场无论仪器在干什么都是同一个
#: 物理量,给它们上门只会白白丢数据。
_DEFAULT_GATED: frozenset[str] = frozenset({"tunnel_current"})

#: 合成序列名。写在这里而不是散在各处,因为前端和文档都引用它。
QUIET_SERIES = "instrument_quiet"


class EnvHistorySink:
    """把环境读数聚合成桶。线程内串行（只有监控线程调用 :meth:`write`）。"""

    def __init__(self, *, store_getter: Callable | None = None,
                 thresholds_getter: Callable | None = None,
                 scope_provider: Callable | None = None,
                 state_getter: Callable | None = None,
                 on_tick: Callable | None = None) -> None:
        self._store_getter = store_getter
        self._thresholds_getter = thresholds_getter
        self._scope = scope_provider
        self._state_getter = state_getter
        self._on_tick = on_tick
        self._lock = threading.RLock()
        self._accs: dict[str, BucketAccumulator] = {}
        self._gated: frozenset[str] = _DEFAULT_GATED
        self._ctx: dict = {}
        self._ctx_quiet: str = "unknown"
        self._ctx_at: float = 0.0
        self._disabled_until = 0.0
        self._warned = False
        self.written = 0
        self.failed = 0
        self.buckets_written = 0

    # ── 配置 ──────────────────────────────────────────────────────────

    def set_gated_sensors(self, names: Iterable[str] | None) -> None:
        """哪些序列只在仪器安静时才记。运行时接线（传感器集合变了）会调它。

        ``None`` = 沿用默认；**空序列 = 一个都不门控**。两者刻意不同：这台机器
        可能一个 Nanonis 传感器都没有（没连上），那时正确答案是"没有需要门控的
        序列"，而不是"回退到默认那张名字表"。把空当成"没设置"正是本仓库
        KNOWN_KEYS 那类静默 no-op 的形状。
        """
        with self._lock:
            self._gated = (_DEFAULT_GATED if names is None
                           else frozenset(str(n) for n in names))

    @property
    def gated_sensors(self) -> frozenset[str]:
        return self._gated

    # ── sink 协议 ─────────────────────────────────────────────────────

    def write(self, sensor: str, value: float, unit: str, status: str) -> None:
        """收一条读数。**永不抛。**

        自禁期间连 :meth:`_refresh_ctx` 都不做，所以 ``on_tick`` 也不响 —— 那 60 秒
        里清扫与 Z burst 一并暂停。这是刻意的：写入正在失败的时候不是开始删东西
        的好时机。下一拍自然恢复。
        """
        now = time.time()
        if time.monotonic() < self._disabled_until:
            return
        th = self._thresholds()
        if th is None or not th.enabled:
            return
        try:
            quietness = self._refresh_ctx(now)
            eid, sid = self._scope_ids()
            admit = True
            if sensor in self._gated:
                from mast.envhistory.quiet import admits
                admit = admits(quietness)
            self._feed(sensor, now, value, unit, status, th.bucket_s,
                       admit=admit, eid=eid, sid=sid)
            self.written += 1
        except Exception as exc:  # noqa: BLE001 — 监控线程绝不因记录而停摆
            self._fail(exc)

    def _feed(self, sensor: str, ts: float, value, unit, status,
              bucket_s: float, *, admit: bool,
              eid: str | None, sid: str | None) -> None:
        with self._lock:
            acc = self._accs.get(sensor)
            if acc is None:
                acc = self._accs[sensor] = BucketAccumulator(sensor)
            done = acc.add(ts, value, unit, status, dt_s=bucket_s,
                           admit=admit, experiment_id=eid, sample_id=sid)
        if done is not None:
            self._persist([done])

    # ── 心跳搭车的两件事 ──────────────────────────────────────────────

    def _refresh_ctx(self, now: float) -> str:
        """按 TTL 刷新上下文快照；刷新的那一拍顺带喂合成序列并喊 on_tick。"""
        if (now - self._ctx_at) < _CTX_TTL_S:
            return self._ctx_quiet
        from mast.envhistory.quiet import QUIET, classify, context_from_snapshots
        ctx = context_from_snapshots(self._state_getter)
        quietness = classify(ctx)
        self._ctx, self._ctx_quiet, self._ctx_at = ctx, quietness, now

        th = self._thresholds()
        if th is not None:
            eid, sid = self._scope_ids()
            try:
                self._feed(QUIET_SERIES, now, 1.0 if quietness == QUIET else 0.0,
                           "", "ok", th.bucket_s, admit=True, eid=eid, sid=sid)
            except Exception:  # noqa: BLE001 — 合成序列不比真读数重要
                logger.debug("quiet series feed failed", exc_info=True)
        if self._on_tick is not None:
            try:
                self._on_tick(now, ctx, quietness)
            except Exception:  # noqa: BLE001 — 到期检查失败不影响本次读数
                logger.debug("envhistory tick failed", exc_info=True)
        return quietness

    @property
    def last_quietness(self) -> str:
        return self._ctx_quiet

    @property
    def last_context(self) -> dict:
        return dict(self._ctx)

    # ── 落盘 ──────────────────────────────────────────────────────────

    def _persist(self, buckets: list) -> None:
        """一个桶一次 upsert。

        刻意不攒批：所有传感器在同一个分钟边界翻桶，攒批确实能把七次写并成
        一次，但代价是崩溃时丢掉这一分钟里已经算好的统计 —— 而这些行是永久
        语料。一分钟七行 INSERT 在 WAL 上是微秒级的，换崩溃安全很划算。
        """
        store = self._store()
        if store is None:
            return
        n = store.upsert_buckets(buckets)
        if n:
            self.buckets_written += n
        elif buckets:
            self._fail(RuntimeError("store.upsert_buckets 未写入任何行"))

    def flush(self) -> int:
        """落定所有未满的桶（关机 / 换实验 / 换样品）。返回写入行数。

        **走与翻桶同一条 :meth:`_persist`**，因此写失败同样被计数、同样触发自禁。
        早先的实现在这里直接调 store 并忽略返回值：flush 恰好发生在关机、换实验、
        换样品这三个时刻，那正是丢一批桶最不容易被发现的地方 —— 界面上只是那一
        分钟没有点，而它看起来和"这一分钟仪器没读数"一模一样。
        """
        try:
            with self._lock:
                done = [b for acc in self._accs.values()
                        if (b := acc.flush()) is not None]
            if not done:
                return 0
            before = self.buckets_written
            self._persist(done)
            return self.buckets_written - before
        except Exception:  # noqa: BLE001
            logger.debug("envhistory flush failed", exc_info=True)
            return 0

    def rotate(self) -> None:
        """换样品/换实验：落定当前桶，让新读数从干净的桶开始归属新 scope。"""
        self.flush()

    def close(self) -> None:
        self.flush()

    # ── 杂项 ──────────────────────────────────────────────────────────

    def _fail(self, exc: BaseException) -> None:
        self.failed += 1
        self._disabled_until = time.monotonic() + _DISABLE_S
        if not self._warned:
            self._warned = True
            logger.warning("环境历史写入失败，暂停 %ds（实时监控与 CSV 不受影响）：%r",
                           int(_DISABLE_S), exc)

    def _store(self):
        try:
            if self._store_getter is not None:
                return self._store_getter()
            from mast.envhistory.store import get_store
            return get_store()
        except Exception:  # noqa: BLE001
            logger.debug("env history store unavailable", exc_info=True)
            return None

    def _thresholds(self):
        try:
            if self._thresholds_getter is not None:
                return self._thresholds_getter()
            from mast.envhistory.thresholds import get_env_history_thresholds
            return get_env_history_thresholds()
        except Exception:  # noqa: BLE001
            return None

    def _scope_ids(self) -> tuple[str | None, str | None]:
        """(experiment_id, sample_id)。scope_provider 与 runtime 既有的那个
        同形（返回四元组），读不到就不归属 —— 归属缺失不是不记录的理由。"""
        if self._scope is None:
            return None, None
        try:
            scope = self._scope()
        except Exception:  # noqa: BLE001
            return None, None
        if not scope:
            return None, None
        try:
            if len(scope) >= 4:
                return (scope[2] or None), (scope[3] or None)
            if len(scope) == 2:
                return (scope[0] or None), (scope[1] or None)
        except TypeError:
            pass
        return None, None

    def stats(self) -> dict:
        with self._lock:
            pending = {name: acc.pending for name, acc in self._accs.items()}
        return {
            "written": self.written,
            "failed": self.failed,
            "buckets_written": self.buckets_written,
            "disabled": time.monotonic() < self._disabled_until,
            "sensors": sorted(pending),
            "pending": pending,
            "gated": sorted(self._gated),
            "quietness": self._ctx_quiet,
        }


__all__ = ["EnvHistorySink", "QUIET_SERIES"]
