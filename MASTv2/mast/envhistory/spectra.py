"""噪声谱快照的攒谱与压缩 —— 纯对象，无 IO、无线程、不依赖 numpy。

一条谱快照回答的是"**这台机器这半小时的本底噪声长什么样**"，不是"这一秒的
电流长什么样"。所以它不是把某一段的 PSD 存下来，而是：

1. 每隔 ``eh_spectrum_accum_every_s`` 从段流里收**一段安静的**谱进来；
2. 立刻对数分箱压到几百个点（原始 10000 点 × 每半小时 180 段是存不起的）；
3. 攒够 ``eh_spectrum_min_segments`` 且到了间隔，逐点取 **median** 发一条。

为什么每步都是 median 而不是 mean
---------------------------------

分箱内取 median：周期图的每个 bin 服从指数分布，尾巴很重，单个 bin 的涨落约
100%。取中位数把这份方差压掉，与 :mod:`mast.monitoring.features` 里 1/f 斜率
拟合采用 median 分箱的理由相同。

跨段取 median：半小时里总会有几段撞上开门、走路、一次意外的针尖跳变。mean 会
把它们摊进本底，median 直接把它们扔掉 —— 而"本底"恰恰是这条曲线要表达的东西。

分箱在低频端是近乎无损的（实测，20 kHz / 1 s 段 / 240 箱）
----------------------------------------------------------

一个对数箱里装得下多少个原始 bin，随频率变化很大::

      5 Hz →   1 个      500 Hz →  21 个
     50 Hz →   2 个     5000 Hz → 209 个

所以"箱内 median 能压掉尖锐的谱线"这句话**只在几百赫兹以上成立**：实测把一根
1e4 倍的谱线分别放在 50 / 500 / 5000 Hz，压缩后该箱的读数分别是 4950× / 1.0× /
1.0×。

这正是想要的行为，不是缺陷。**归档谱的用途就是看见工频线、看见泵的振动峰**;
一条把 50 Hz 抹平的曲线会让用户以为接地是好的。高频端那些孤立的单 bin 涨落
则是周期图的统计噪声，压掉它们才让本底可读。真要判"工频污染有多重"，
:mod:`mast.monitoring.features` 的 ``line_ratio`` 是专门做那件事的标量,
它比较的是两个频带的平均功率密度，不受本模块的分箱影响。

分辨率的诚实边界
----------------

* 谱**只从全速率样本算**，绝不从抽稀视图算（抽稀会把新 Nyquist 以上的东西混叠
  进读者要看的频段）。本模块只接收调用方算好的 (freqs, psd)，那一步在
  :func:`mast.monitoring.features._psd_of_runs` 里，它按 run 分别算再平均,
  从不跨缺口拼接。
* ``fs_hz`` 变了（有人重选了时基）就**清空重攒**：两个不同频率网格上的谱逐点
  取 median 是没有意义的。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

#: 分箱下界的兜底。低于 0.5 Hz 时一个 1 s 段根本没有对应的 bin。
_F_LO_FLOOR_HZ = 0.5

#: 攒谱环的硬上限：防一个被调得很长的间隔把内存吃掉。240 点 × 128 条 ≈ 250 KB。
_MAX_ACCUM = 128


#: 决定「这条谱是在什么条件下测的」的那几个量。
#:
#: 只有这三个进 :attr:`SpectrumSnapshot.ctx_stable` 的比较：它们变了，谱就不再是
#: 同一个工作点上的测量，两条谱也就不可比。``ctx_skill`` / ``ctx_scanning``
#: 刻意不在内 —— 那些一个窗口里本来就会来回变，拿它们判会让 ``ctx_stable``
#: 永远是 False，于是这个字段什么也不再区分。
_CTX_STATE_KEYS: tuple[str, ...] = ("ctx_bias_v", "ctx_setpoint_a", "ctx_zctrl_on")


def _state_of(ctx: dict | None) -> tuple:
    return tuple((ctx or {}).get(k) for k in _CTX_STATE_KEYS)


@dataclass
class SpectrumSnapshot:
    """一条待落库的谱快照。字段名与 store 的参数名对齐。"""

    ts: float
    channel: str
    span_s: float
    n_segments: int
    fs_hz: float
    freqs: list[float]
    psd: list[float]
    unit: str = "A^2/Hz"
    quietness: str = "quiet"
    ctx: dict = field(default_factory=dict)
    #: 攒谱期间 bias / setpoint / Z 反馈**有没有保持不变**。
    #:
    #: 一条谱可以横跨半小时。原来只留最后一次非空 ctx，于是一条「偏压 -1.2 V」
    #: 的谱可能有一大半是在 +0.5 V 下攒的 —— 数字看着精确，而它描述的是窗口的
    #: 末尾，不是这条谱。这个标志让「当时的状态」这句话可以是假的时候说出来。
    ctx_stable: bool = True


def log_bin(freqs: Sequence[float], psd: Sequence[float], n_bins: int,
            f_lo: float | None = None,
            f_hi: float | None = None) -> tuple[list[float], list[float]]:
    """把一条线性频率网格的 PSD 压成对数分箱的 (中心频率, median 功率)。

    空箱被**丢弃**而不是插值：低频端的箱可能比频率分辨率还窄，插出来的点会让
    读者以为那里有数据。返回的两个数组因此可能短于 ``n_bins``。
    """
    n = min(len(freqs), len(psd))
    if n < 2 or n_bins < 2:
        return [], []
    lo = f_lo if f_lo is not None else 0.0
    if lo <= 0:
        # 跳过 DC；第一个正频率就是频率分辨率 df。
        lo = next((float(f) for f in freqs if float(f) > 0.0), 0.0)
    lo = max(float(lo), _F_LO_FLOOR_HZ)
    hi = float(f_hi) if f_hi is not None else float(freqs[n - 1])
    if not (hi > lo):
        return [], []

    log_lo, log_hi = math.log(lo), math.log(hi)
    scale = n_bins / (log_hi - log_lo)
    acc: dict[int, list[tuple[float, float]]] = {}
    for i in range(n):
        f = float(freqs[i])
        if f < lo or f > hi:
            continue
        p = float(psd[i])
        if not math.isfinite(p):
            continue
        idx = int((math.log(f) - log_lo) * scale)
        if idx >= n_bins:
            idx = n_bins - 1
        acc.setdefault(idx, []).append((f, p))

    out_f: list[float] = []
    out_p: list[float] = []
    for idx in sorted(acc):
        pairs = acc[idx]
        # 中心频率取箱内实际频点的中位数 —— 报箱的几何中心会在稀疏的低频端
        # 谎报一个根本没有采到的频率。
        out_f.append(_median([f for f, _ in pairs]))
        out_p.append(_median([p for _, p in pairs]))
    return out_f, out_p


class SpectrumAccumulator:
    """一个通道的攒谱器。非线程安全 —— 调用方（recorder）自己串行化。"""

    def __init__(self, channel: str, unit: str = "A^2/Hz") -> None:
        self.channel = str(channel)
        self.unit = str(unit)
        self._fs: float | None = None
        self._grid: list[float] = []
        self._rows: list[list[float]] = []
        self._t_first: float | None = None
        self._t_last: float | None = None
        self._last_accum_ts: float = 0.0
        self._ctx: dict = {}
        self._ctx_stable: bool = True

    # ── 攒 ────────────────────────────────────────────────────────────

    def should_accum(self, now: float, every_s: float) -> bool:
        """距上次收段够久了吗 —— 子采样闸门，限住 FFT 的 CPU 开销。"""
        return (now - self._last_accum_ts) >= max(0.0, float(every_s))

    def add(self, freqs: Sequence[float], psd: Sequence[float], *,
            fs_hz: float, ts: float, bins: int, ctx: dict | None = None) -> bool:
        """收一段的谱。返回是否真的收下了。"""
        fs = float(fs_hz)
        if fs <= 0:
            return False
        if self._fs is not None and abs(fs - self._fs) > 1e-6:
            # 时基被改了：旧网格上的谱与新的不可通约，扔掉重来。
            self.reset()
        f, p = log_bin(freqs, psd, int(bins))
        if len(f) < 2:
            return False
        if self._grid and len(f) != len(self._grid):
            # 同一个 fs 下网格长度仍然变了（段长变化导致 df 变化）。逐点 median
            # 要求逐点对齐，对不齐就重来 —— 强行对齐等于把不同频率的功率平均。
            self.reset()
        self._fs = fs
        self._grid = f
        self._rows.append(p)
        if len(self._rows) > _MAX_ACCUM:
            del self._rows[0]
        self._t_first = ts if self._t_first is None else self._t_first
        self._t_last = float(ts)
        self._last_accum_ts = float(ts)
        if ctx:
            # 保留**第一条**：它和 span_s 说的是同一个起点，两个字段因此描述同一
            # 件事。留最后一条的话，「偏压」与「这条谱覆盖的时间」会指向窗口的
            # 两端，而没有任何字段说得出这件事。
            if self._ctx:
                if _state_of(ctx) != _state_of(self._ctx):
                    self._ctx_stable = False
            else:
                self._ctx = dict(ctx)
        return True

    # ── 发 ────────────────────────────────────────────────────────────

    def maybe_emit(self, now: float, *, interval_s: float,
                   min_segments: int, quietness: str = "quiet",
                   ) -> SpectrumSnapshot | None:
        """到点且攒够了就发一条并清空；否则返回 None（继续攒，窗口自然变长）。"""
        if not self._rows or self._t_first is None:
            return None
        if (now - self._t_first) < max(1.0, float(interval_s)):
            return None
        if len(self._rows) < max(1, int(min_segments)):
            # 到点但安静段不够 —— 不发半条。窗口继续延长,status 里会说明。
            return None
        med = [_median([row[i] for row in self._rows])
               for i in range(len(self._grid))]
        snap = SpectrumSnapshot(
            ts=float(now),
            channel=self.channel,
            span_s=float((self._t_last or now) - self._t_first),
            n_segments=len(self._rows),
            fs_hz=float(self._fs or 0.0),
            freqs=list(self._grid),
            psd=med,
            unit=self.unit,
            quietness=quietness,
            ctx=dict(self._ctx),
            ctx_stable=bool(self._ctx_stable),
        )
        self.reset()
        return snap

    def reset(self) -> None:
        self._fs = None
        self._grid = []
        self._rows = []
        self._t_first = None
        self._t_last = None
        self._ctx = {}
        self._ctx_stable = True

    # ── 自述 ──────────────────────────────────────────────────────────

    @property
    def n_accum(self) -> int:
        return len(self._rows)

    def stats(self, now: float | None = None) -> dict:
        elapsed = None
        if self._t_first is not None and now is not None:
            elapsed = float(now) - self._t_first
        return {"channel": self.channel, "n_accum": len(self._rows),
                "fs_hz": self._fs, "window_s": elapsed,
                "n_points": len(self._grid)}


def _median(vals: Sequence[float]) -> float:
    """中位数。刻意不用 statistics.median —— 它对非有限值会给出误导性结果,
    而周期图里出现 inf/nan 是真实会发生的（一段全零的 run）。"""
    clean = sorted(v for v in vals if math.isfinite(v))
    n = len(clean)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


__all__ = ["SpectrumSnapshot", "SpectrumAccumulator", "log_bin"]
