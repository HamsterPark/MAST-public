"""环境读数写 CSV —— 温度-时间文件，用户用 Excel/Origin 直接打开。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §8

为什么是 CSV 而不是 JSONL / parquet
-----------------------------------

* 这个文件唯一的高频用途就是**画一条温度曲线**。Excel / Origin 打开 CSV 是零
  摩擦的，JSONL 不是。
* 单行追加天然崩溃安全：最坏丢最后一个采样点，剩下的照样能画。
* 2 秒采样 = 43,200 行/天/传感器。JSONL 体积是 CSV 的 3–4 倍。
* parquet 不能便宜地追加，还要在 0.5 Hz 的监控线程上拉 pyarrow。

双写：实验级连续 + 样品级切片
-----------------------------

温度是**连续**的物理量，不因换样品而中断 —— 所以实验级按 UTC 日轮转记全量。
但分析某个样品时又需要"这段时间的温度"，所以同时往**当前活跃样品**的
``env/temperature.csv`` 写一份。

这是 INCREMENTAL-ONLY 的要求：不能设计成"样品结束时从实验级切一段出来"，因为
根本没有"结束"这个时机（实验永不归档）。多写 ~2.5 MB/天/传感器换来的是**断电
时样品目录已经完整**。

绝不阻塞监控线程
----------------

所有写入包在 try/except 里；一旦出错就**禁用这个 sink 60 秒**并告警一次，
监控循环照常跑。环境监控的首要职责是发现真空/温度异常并告警，写 CSV 失败不能
让它停摆。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: 表头。``epoch_s`` 是给绘图工具的：4 万行 ISO 字符串解析很慢，而且与 action
#: 时间戳做精确 join 时不用管时区。scope 两列写在行里，这样文件被拼接之后仍然
#: 可解读。
HEADER = "timestamp_iso,epoch_s,sensor,value,unit,status,experiment_id,sample_id\n"

#: 写失败后静默多久再重试。
_DISABLE_S = 60.0
#: fsync 间隔。0.5 Hz 的数据每次都 fsync 没必要。
_FSYNC_S = 30.0


def _csv_escape(v: Any) -> str:
    s = "" if v is None else str(v)
    if any(c in s for c in (",", '"', "\n", "\r")):
        return '"' + s.replace('"', '""') + '"'
    return s


class EnvironmentCsvSink:
    """把环境读数写进实验文件夹。注入给 :class:`EnvironmentMonitor`。

    ``scope_provider()`` 返回 ``(exp_dir, sample_dir_name, experiment_id,
    sample_id)``；返回 None 表示当前没有实验，本次读数只进 DB 不落文件。
    让 monitor 通过回调拿 scope，而不是自己 import runtime —— environment 层
    因此不必知道 core 的存在。
    """

    def __init__(self, scope_provider: Callable[[], Any]) -> None:
        self._scope = scope_provider
        self._lock = threading.RLock()
        self._handles: dict[Path, Any] = {}
        self._last_fsync: dict[Path, float] = {}
        self._disabled_until = 0.0
        self._warned = False
        self.written = 0
        self.failed = 0

    # ── 写入 ──────────────────────────────────────────────────────────

    def write(self, sensor: str, value: float, unit: str, status: str) -> None:
        """写一条读数。**永不抛，永不阻塞。**"""
        if time.monotonic() < self._disabled_until:
            return
        try:
            scope = self._scope()
        except Exception:  # noqa: BLE001
            return
        if not scope:
            return
        exp_dir, sample_dir, eid, sid = scope
        if exp_dir is None:
            return

        now = datetime.now(timezone.utc)
        row = ",".join((
            now.isoformat(timespec="milliseconds"),
            f"{now.timestamp():.3f}",
            _csv_escape(sensor),
            _csv_escape(value),
            _csv_escape(unit),
            _csv_escape(status),
            _csv_escape(eid or ""),
            _csv_escape(sid or ""),
        )) + "\n"

        # 实验级：跨样品连续，按 UTC 日轮转。单文件 ~2.5 MB，崩溃最多丢一天的尾巴。
        targets = [Path(exp_dir) / "env" / f"{_stem(sensor)}_{now:%Y-%m-%d}.csv"]
        # 样品级：该样品时段的切片，增量双写（不是"结束时切"—— 没有结束这个时机）。
        if sample_dir:
            targets.append(Path(exp_dir) / "samples" / sample_dir / "env"
                           / f"{_stem(sensor)}.csv")

        try:
            for t in targets:
                self._append(t, row)
            self.written += 1
        except OSError as exc:
            self.failed += 1
            self._disabled_until = time.monotonic() + _DISABLE_S
            self._close_all()
            if not self._warned:
                self._warned = True
                logger.warning("环境 CSV 写入失败，暂停 %ds（DB 记录不受影响）：%r",
                               int(_DISABLE_S), exc)

    def _append(self, path: Path, row: str) -> None:
        with self._lock:
            f = self._handles.get(path)
            if f is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                fresh = not path.exists() or path.stat().st_size == 0
                f = open(path, "a", encoding="utf-8", newline="")
                if fresh:
                    f.write(HEADER)
                self._handles[path] = f
                self._last_fsync[path] = time.monotonic()
            f.write(row)
            f.flush()
            now = time.monotonic()
            if now - self._last_fsync.get(path, 0.0) >= _FSYNC_S:
                self._last_fsync[path] = now
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

    # ── 生命周期 ──────────────────────────────────────────────────────

    def rotate(self) -> None:
        """关掉所有句柄（换样品/换实验时调用，下次写入自然开新文件）。"""
        self._close_all()

    def close(self) -> None:
        self._close_all()

    def _close_all(self) -> None:
        with self._lock:
            for f in self._handles.values():
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except (OSError, ValueError):
                    pass
                try:
                    f.close()
                except (OSError, ValueError):
                    pass
            self._handles.clear()
            self._last_fsync.clear()

    def stats(self) -> dict:
        return {"written": self.written, "failed": self.failed,
                "open_files": len(self._handles),
                "disabled": time.monotonic() < self._disabled_until}


def _stem(sensor: str) -> str:
    """传感器名 → 文件名词干。"""
    from mast.core.experiment_paths import slug
    return slug(sensor, "sensor", max_chars=32)


def slice_env_window(exp_dir: Path, sensor_stem: str,
                     started_at: str, ended_at: str | None = None) -> list[str]:
    """按时间窗从实验级 CSV 里切一段出来（**按需工具**，不在任何自动路径上）。

    双写已经保证样品目录在任何时刻都是完整的；这个函数是给"补历史"和"事后
    修正时间窗"用的。
    """
    out: list[str] = []
    lo = str(started_at or "")
    hi = str(ended_at or "9999")
    try:
        for p in sorted((Path(exp_dir) / "env").glob(f"{sensor_stem}_*.csv")):
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    ts = line.split(",", 1)[0]
                    if lo <= ts <= hi:
                        out.append(line)
    except OSError:
        pass
    return out
