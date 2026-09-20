"""把 2 秒读数流增量聚合成统计桶 —— 纯对象，无 IO、无线程、无锁。

每个传感器一个 :class:`BucketAccumulator`。读数一条条喂进来，跨过桶边界时
吐出一个完成的 :class:`Bucket`；调用方负责把它写进库。

为什么是增量（Welford）而不是攒一个 list 再算
---------------------------------------------

一个桶最多几十个点，攒 list 也不会爆。用 Welford 是因为**桶宽可以被用户调到
一小时** —— 那就是 1800 个点 × N 传感器常驻内存，且断电时全丢。增量统计的内存
是常数，而且任何时刻 :meth:`flush` 都能拿到一个当下就正确的桶。

哪些读数进统计、哪些只计数
--------------------------

``EnvironmentMonitor.read_all()`` 在传感器抛异常时写的是
``SensorReading(value=0.0, unit="", status="error")`` —— **那个 0.0 不是读数**。
把它算进 min 会让一条 77 K 的温度曲线出现一个 0，而那个 0 从来没有发生过。
所以：

* ``ok`` / ``warning`` 的读数进统计；
* ``error`` / ``unavailable`` 只计进 ``n_excluded``，值被丢弃；
* 被安静门拒掉的读数（扫描中的隧道电流）同样只计 ``n_excluded``；
* 但**所有**读数（含被排除的）都参与 ``worst_status`` —— 桶要诚实地说
  "这一分钟里最糟的情况是什么"，哪怕那一分钟一个有效值都没有。

时钟回跳
--------

分钟边界由 ``floor(ts / dt) * dt`` 算出。只要算出来的边界与当前桶不同就翻桶,
**两个方向都翻**。这个项目已经被时钟回跳咬过一次（扫描地图图层静默冻结）:
只往前翻的实现会在系统时间往回跳之后把所有后续读数判成"属于过去的桶"而永久
丢弃，表现是记录静静地停住。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: 值进入统计的读数状态。其余状态只计数。
_ADMITTED_STATUSES = frozenset({"ok", "warning"})


@dataclass
class Bucket:
    """一个完成的统计桶。字段名 = DDL 列名（存储契约）。"""

    sensor: str
    bucket_ts: float
    dt_s: float
    n: int
    mean: float | None
    min: float | None
    max: float | None
    std: float | None
    last: float | None
    unit: str
    worst_status: str
    n_excluded: int
    experiment_id: str | None
    sample_id: str | None

    def to_row(self) -> tuple:
        """按 DDL 列序展开，给 store 的 executemany 绑定。"""
        return (self.sensor, self.bucket_ts, self.dt_s, self.n,
                self.mean, self.min, self.max, self.std, self.last,
                self.unit, self.worst_status, self.n_excluded,
                self.experiment_id, self.sample_id)


def bucket_start(ts: float, dt_s: float) -> float:
    """``ts`` 落在哪个桶里。dt 非正时退化成"每条读数一个桶"。"""
    if dt_s <= 0:
        return float(ts)
    return math.floor(float(ts) / float(dt_s)) * float(dt_s)


class BucketAccumulator:
    """单个传感器的增量聚合器。非线程安全 —— 调用方（sink）自己串行化。"""

    def __init__(self, sensor: str) -> None:
        self.sensor = str(sensor)
        self._dt: float = 0.0
        self._start: float | None = None
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min: float | None = None
        self._max: float | None = None
        self._last: float | None = None
        self._unit = ""
        self._worst = "ok"
        self._excluded = 0
        self._scopes: dict[tuple[str | None, str | None], int] = {}

    # ── 喂数据 ────────────────────────────────────────────────────────

    def add(self, ts: float, value: float, unit: str, status: str,
            *, dt_s: float, admit: bool = True,
            experiment_id: str | None = None,
            sample_id: str | None = None) -> Bucket | None:
        """喂一条读数。跨桶（或桶宽变了）时返回上一个完成的桶。

        ``admit=False`` 表示调用方的语义门拒绝了这条读数（例如非安静时段的隧道
        电流）：它仍然计入 ``n_excluded`` 与 ``worst_status``，但不进统计。
        """
        ts = float(ts)
        dt_s = float(dt_s)
        start = bucket_start(ts, dt_s)
        done: Bucket | None = None
        if self._start is None:
            self._start, self._dt = start, dt_s
        elif start != self._start or dt_s != self._dt:
            # 边界（任一方向）或桶宽变化 —— 先落定旧桶再开新的。
            done = self._emit()
            self._reset()
            self._start, self._dt = start, dt_s

        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        st = str(status or "ok")
        self._worst = _worst(self._worst, st)
        key = (experiment_id or None, sample_id or None)
        self._scopes[key] = self._scopes.get(key, 0) + 1
        if unit:
            self._unit = str(unit)

        usable = admit and st in _ADMITTED_STATUSES and math.isfinite(v)
        if not usable:
            self._excluded += 1
            return done

        self._n += 1
        delta = v - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (v - self._mean)
        self._min = v if self._min is None else min(self._min, v)
        self._max = v if self._max is None else max(self._max, v)
        self._last = v
        return done

    def flush(self) -> Bucket | None:
        """强制落定当前桶（关机 / 换实验 / 测试）。空桶返回 None。"""
        if self._start is None:
            return None
        out = self._emit()
        self._reset()
        self._start = None
        return out

    @property
    def pending(self) -> int:
        """当前未落定的读数条数（含被排除的）—— 给 status 用。"""
        return self._n + self._excluded

    # ── 内部 ──────────────────────────────────────────────────────────

    def _emit(self) -> Bucket | None:
        if self._n == 0 and self._excluded == 0:
            return None
        # 总体标准差（ddof=0），与 numpy.std 的默认一致 —— 测试可以直接对拍。
        std = math.sqrt(self._m2 / self._n) if self._n > 0 else None
        eid, sid = _majority(self._scopes)
        return Bucket(
            sensor=self.sensor,
            bucket_ts=float(self._start or 0.0),
            dt_s=float(self._dt),
            n=self._n,
            mean=self._mean if self._n else None,
            min=self._min,
            max=self._max,
            std=std,
            last=self._last,
            unit=self._unit,
            worst_status=self._worst,
            n_excluded=self._excluded,
            experiment_id=eid,
            sample_id=sid,
        )

    def _reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = None
        self._max = None
        self._last = None
        self._worst = "ok"
        self._excluded = 0
        self._scopes = {}


def _worst(a: str, b: str) -> str:
    """两个状态里更糟的那个。复用环境层既有的严重度序，不另立一套。"""
    try:
        from mast.environment.alarm import worst_status
        return worst_status(a, b)
    except Exception:  # noqa: BLE001 — 纯函数层不因一个 import 失败而失灵
        return b if a == "ok" else a


def _majority(scopes: dict) -> tuple[str | None, str | None]:
    """桶内出现最多的 (experiment_id, sample_id)。

    换样品发生在桶中间时，整桶归给出现更多的那一边 —— 归给"最后一条"会让一个
    59 秒都属于 S01 的桶因为最后一秒切到 S02 而整体记到 S02 名下。
    """
    if not scopes:
        return None, None
    return max(scopes.items(), key=lambda kv: kv[1])[0]


__all__ = ["Bucket", "BucketAccumulator", "bucket_start"]
