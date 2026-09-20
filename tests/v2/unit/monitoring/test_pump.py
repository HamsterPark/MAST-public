"""Osci1T pump: decoding, t0 stitching, gap accounting and degradation.

The test harness feeds wire-shaped tuples through a fake pool, because the repo
mocks ``nanonis_spm`` with a bare MagicMock — there is no protocol simulator to
lean on, so the decode path has to be exercised against hand-built replies in
exactly the shape the library produces (scalars and array elements arrive
wrapped in 1-tuples).
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import threading
import time

import numpy as np
import pytest

from mast.core.types import NanonisCallRecord
from mast.monitoring.pump import (
    Osci1TPump, PumpPaused, PumpUnavailable, Segment,
)

DT = 1.0 / 20000.0
NPTS = 1024


def _wire(*fields) -> tuple:
    """The (header, raw, fields) shape nanonis_spm hands back."""
    return ("", b"", list(fields))


#: 解码出来的数值有**两种**真实形态,替身两种都要发得出来。
#:
#: * ``tuple``  —— `decodeArray` 的原始产物(每个元素裹一层 1-元组);
#: * ``bare``   —— `core/nanonis_patch` 在解码层拆掉之后的形态。
#:
#: ⚠️ 这个文件原来**只发 tuple**,而 `nanonis_patch.apply()` 是 import 时自动执行的
#: (`core/connection.py:28` import-for-side-effect),所以**生产里早就是 bare 了**。
#: 也就是说:这些测试覆盖的是生产**已经不再产生**的那个形态,而生产真正跑的那个
#: 从来没被覆盖过。因此需要覆盖裸数值和元组两种回包形态。
SHAPES = ("tuple", "bare")


def _scalar_as(v, shape: str):
    return (v,) if shape == "tuple" else v


def _seq_as(vals, shape: str):
    return [(v,) for v in vals] if shape == "tuple" else list(vals)


def _data_reply(t0: float, dt: float = DT, n: int = NPTS,
                value: float = 100e-12, shape: str = "tuple") -> tuple:
    """A DataGet reply: t0, dt, n, then n samples in the requested element shape."""
    return _wire(t0, _scalar_as(dt, shape), _scalar_as(n, shape),
                 _seq_as([value + i * 1e-15 for i in range(n)], shape))


def _names_reply(names: list[str], shape: str = "tuple") -> tuple:
    return _wire(_scalar_as(len(names), shape), _scalar_as(len(names), shape),
                 tuple(names))


def _timebase_reply(values: list[float], shape: str = "tuple") -> tuple:
    return _wire(_scalar_as(0, shape), _scalar_as(len(values), shape),
                 _seq_as(values, shape))


class FakePool:
    """Scripted pool. ``handlers`` maps method name -> callable(args) -> record."""

    def __init__(self, handlers: dict, default_error: str = ""):
        self.handlers = handlers
        self.default_error = default_error
        self.calls: list[tuple] = []

    def safe_call(self, method, *args, role="main", **kw):
        self.calls.append((method, args, role))
        h = self.handlers.get(method)
        if h is None:
            return NanonisCallRecord(method=method, args=args,
                                     error=self.default_error)
        out = h(args) if callable(h) else h
        if isinstance(out, NanonisCallRecord):
            return out
        return NanonisCallRecord(method=method, args=args, return_value=out)


def _base_handlers(timebases=(DT, 2 * DT, 4 * DT), shape: str = "tuple") -> dict:
    return {
        "Osci1T_Run": lambda a: NanonisCallRecord(method="Osci1T_Run"),
        "Signals_NamesGet": lambda a: _names_reply(
            ["Z (m)", "Current (A)", "Bias (V)"], shape),
        "Osci1T_TimebaseGet": lambda a: _timebase_reply(list(timebases), shape),
        "Osci1T_TimebaseSet": lambda a: NanonisCallRecord(method="Osci1T_TimebaseSet"),
        "Osci1T_ChSet": lambda a: NanonisCallRecord(method="Osci1T_ChSet"),
        "Osci1T_TrigSet": lambda a: NanonisCallRecord(method="Osci1T_TrigSet"),
        "Util_RTFreqGet": lambda a: _wire(_scalar_as(20000.0, shape)),
    }


def _pump(handlers, **kw) -> tuple[Osci1TPump, FakePool]:
    pool = FakePool(handlers)
    return Osci1TPump(lambda: pool, **kw), pool


# ── configure ────────────────────────────────────────────────────────────────

def test_configure_finds_the_current_channel_by_name():
    pump, pool = _pump(_base_handlers())
    cfg = pump.configure()
    assert cfg["channel_index"] == 1                  # "Current (A)"
    assert cfg["channel_name"] == "Current (A)"
    assert ("Osci1T_ChSet", (1,), "data") in pool.calls


def test_configure_reads_on_the_data_role_only():
    """monitor (6502) already carries the tip-crash watchdog; main can stall the
    API thread pool. Everything here must go to data (6503)."""
    pump, pool = _pump(_base_handlers())
    pump.configure()
    assert {role for _, _, role in pool.calls} == {"data"}


def test_configure_picks_the_fastest_timebase_by_default():
    pump, pool = _pump(_base_handlers(timebases=(4 * DT, DT, 2 * DT)))
    cfg = pump.configure()
    # 时基表描述整屏时长。dt/fs 须等首份回包；之前以 0 表示未知。
    assert cfg["window_s"] == pytest.approx(DT)
    assert cfg["dt_s"] == 0.0 and cfg["fs_hz"] == 0.0
    assert ("Osci1T_TimebaseSet", (1,), "data") in pool.calls   # index of min


def test_configure_honours_a_target_sample_rate():
    pump, _ = _pump(_base_handlers(timebases=(DT, 2 * DT, 4 * DT)),
                    target_fs_hz=5000.0)
    # ⚠️ 已知局限：目标采样率是拿 1/fs 去跟**时基表**比的，而表里是整屏时长，
    # 真要命中一个采样率得除以 n —— 而 n 只有第一份回包才知道。默认值 0（取最快
    # 一档）不受影响，那也是实机在用的路径。这里钉住现有行为，别当它是"按 fs 选档"。
    assert pump.configure()["window_s"] == pytest.approx(4 * DT)


def test_configure_forces_immediate_trigger():
    """Left on Level with nothing crossing it, the scope stops re-arming and
    every poll returns the same stale buffer."""
    pump, pool = _pump(_base_handlers())
    pump.configure()
    trig = [c for c in pool.calls if c[0] == "Osci1T_TrigSet"]
    assert trig and trig[0][1][0] == 0


def test_configure_never_calls_the_broken_trigget_binding():
    pump, pool = _pump(_base_handlers())
    pump.configure()
    assert not any(c[0] == "Osci1T_TrigGet" for c in pool.calls)


def test_missing_module_raises_pump_unavailable():
    h = _base_handlers()
    h["Osci1T_Run"] = lambda a: NanonisCallRecord(
        method="Osci1T_Run", error="NeedModule: Osci1T not loaded")
    pump, _ = _pump(h)
    with pytest.raises(PumpUnavailable):
        pump.configure()


def test_role_busy_raises_pump_paused():
    h = _base_handlers()
    h["Osci1T_Run"] = lambda a: NanonisCallRecord(
        method="Osci1T_Run", error="RoleBusy: data held for 30.0s")
    pump, _ = _pump(h)
    with pytest.raises(PumpPaused):
        pump.configure()


def test_open_circuit_breaker_raises_pump_paused():
    """Retrying into an open breaker is exactly what it exists to prevent."""
    h = _base_handlers()
    h["Osci1T_Run"] = lambda a: NanonisCallRecord(
        method="Osci1T_Run", error="comms_circuit_open: Nanonis TCP 通信熔断")
    pump, _ = _pump(h)
    with pytest.raises(PumpPaused):
        pump.configure()


def test_configure_survives_a_missing_signal_list():
    h = _base_handlers()
    h["Signals_NamesGet"] = lambda a: NanonisCallRecord(
        method="Signals_NamesGet", error="boom")
    pump, _ = _pump(h)
    cfg = pump.configure()
    assert cfg["channel_index"] == -1        # leaves the scope channel alone
    assert cfg["window_s"] == pytest.approx(DT)


# ── poll / decode ────────────────────────────────────────────────────────────

def test_poll_once_decodes_wrapped_scalars_and_arrays():
    h = _base_handlers()
    h["Osci1T_DataGet"] = lambda a: _data_reply(5.0)
    pump, _ = _pump(h)
    pump.configure()
    chunk = pump.poll_once()
    assert chunk.t0 == 5.0
    assert chunk.dt == pytest.approx(DT)
    assert chunk.n == NPTS
    assert isinstance(chunk.y, np.ndarray) and chunk.y.size == NPTS
    assert chunk.y[0] == pytest.approx(100e-12)


def test_poll_once_rejects_an_unfilled_buffer():
    h = _base_handlers()
    h["Osci1T_DataGet"] = lambda a: _wire(0.0, (0.0,), (0,), [])
    pump, _ = _pump(h)
    pump.configure()
    assert pump.poll_once() is None


def test_poll_once_rejects_a_short_reply():
    h = _base_handlers()
    h["Osci1T_DataGet"] = lambda a: _wire(1.0, (DT,))
    pump, _ = _pump(h)
    pump.configure()
    assert pump.poll_once() is None


def test_poll_learns_the_buffer_depth_the_protocol_does_not_expose():
    h = _base_handlers()
    h["Osci1T_DataGet"] = lambda a: _data_reply(1.0, n=512)
    pump, _ = _pump(h)
    pump.configure()
    assert pump.config["n_buffer"] == 0
    pump.poll_once()
    assert pump.config["n_buffer"] == 512


# ── stitching ────────────────────────────────────────────────────────────────

def _run_pump(pump, max_segments=1, max_iters=400) -> list[Segment]:
    stop = threading.Event()
    out: list[Segment] = []
    for seg in pump.pump_segments(stop):
        out.append(seg)
        if len(out) >= max_segments:
            stop.set()
            break
    return out


def _sequence_handlers(t0_sequence, dt=DT, n=NPTS):
    """DataGet replies walking a scripted t0 list, then extrapolating.

    Past the end of the script the scope keeps running at a steady cadence —
    repeating the final t0 instead would look like an endless duplicate and the
    pump would spin without ever closing a segment.
    """
    seq = list(t0_sequence)
    state = {"i": 0}

    def data(_args):
        i = state["i"]
        state["i"] += 1
        if i < len(seq):
            t0 = seq[i]
        else:
            t0 = seq[-1] + (i - len(seq) + 1) * n * dt
        return _data_reply(t0, dt=dt, n=n)

    h = _base_handlers()
    h["Osci1T_DataGet"] = data
    return h


def test_back_to_back_traces_form_one_contiguous_run():
    step = NPTS * DT
    t0s = [i * step for i in range(30)]
    pump, _ = _pump(_sequence_handlers(t0s), segment_s=0.2)
    segs = _run_pump(pump)
    assert len(segs) == 1
    assert len(segs[0].runs) == 1                 # no splice points
    assert segs[0].gap_s == 0.0
    assert segs[0].discontinuity is False
    assert segs[0].n_samples >= int(0.2 / DT)


def test_duplicate_t0_is_dropped_not_counted_twice():
    """Polling faster than the scope re-arms is the expected steady state."""
    step = NPTS * DT
    t0s = []
    for i in range(30):
        t0s += [i * step, i * step]               # every trace seen twice
    pump, _ = _pump(_sequence_handlers(t0s), segment_s=0.2)
    segs = _run_pump(pump)
    n_expected = int(0.2 / DT)
    assert segs[0].n_samples < n_expected + NPTS  # not doubled
    assert len(segs[0].runs) == 1


def test_a_dropped_frame_is_recorded_as_a_gap_and_splits_the_run():
    step = NPTS * DT
    t0s = [0.0, step, 3 * step, 4 * step, 5 * step, 6 * step, 7 * step]
    pump, _ = _pump(_sequence_handlers(t0s), segment_s=0.2)
    segs = _run_pump(pump)
    seg = segs[0]
    assert seg.gap_s == pytest.approx(step, rel=1e-6)
    assert len(seg.runs) >= 2                     # never spliced across the hole
    assert seg.t_end - seg.t_start > seg.n_samples * DT   # span includes the gap


def test_a_backward_clock_marks_discontinuity_without_losing_samples():
    step = NPTS * DT
    t0s = [10 * step, 11 * step, 0.0, step, 2 * step, 3 * step, 4 * step]
    pump, _ = _pump(_sequence_handlers(t0s), segment_s=0.2)
    seg = _run_pump(pump)[0]
    assert seg.discontinuity is True
    assert seg.n_samples > 0


def test_a_timebase_change_marks_discontinuity():
    state = {"i": 0}
    step = NPTS * DT

    def data(_args):
        i = state["i"]
        state["i"] += 1
        if i < 3:
            return _data_reply(i * step, dt=DT)
        return _data_reply(1000.0 + i * step * 2, dt=2 * DT)

    h = _base_handlers()
    h["Osci1T_DataGet"] = data
    pump, _ = _pump(h, segment_s=0.2)
    seg = _run_pump(pump)[0]
    assert seg.discontinuity is True


def test_reconfigure_is_rate_limited(monkeypatch):
    """An operator fiddling with the front panel must not cause a config storm."""
    calls = {"n": 0}
    real = Osci1TPump.configure

    def counting(self):
        calls["n"] += 1
        return real(self)

    monkeypatch.setattr(Osci1TPump, "configure", counting)
    state = {"i": 0}

    def data(_args):
        i = state["i"]
        state["i"] += 1
        return _data_reply(i * NPTS * DT, dt=DT if i % 2 else 2 * DT)

    h = _base_handlers()
    h["Osci1T_DataGet"] = data
    pump, _ = _pump(h, segment_s=0.2)
    _run_pump(pump)
    assert calls["n"] <= 2          # initial + at most one throttled retry


def test_segment_boundary_respects_the_requested_length():
    step = NPTS * DT
    pump, _ = _pump(_sequence_handlers([i * step for i in range(200)]),
                    segment_s=0.5)
    seg = _run_pump(pump)[0]
    assert seg.n_samples * DT >= 0.5 - NPTS * DT
    assert seg.fs_hz == pytest.approx(20000.0)
    assert seg.channel_name == "Current (A)"


def test_samples_property_concatenates_runs():
    seg = Segment(t_start=0, t_end=1, osci_t0=0, fs_hz=20000.0,
                  runs=[np.ones(3), np.zeros(2)], n_samples=5)
    assert seg.samples.size == 5


def test_busy_role_backs_off_and_counts_the_wait_as_a_gap():
    """Scan frame grabs share the data role; queueing behind one is normal."""
    state = {"i": 0}
    step = NPTS * DT

    def data(_args):
        i = state["i"]
        state["i"] += 1
        if 3 <= i < 6:
            return NanonisCallRecord(method="Osci1T_DataGet",
                                     error="RoleBusy: data held")
        return _data_reply(i * step)

    h = _base_handlers()
    h["Osci1T_DataGet"] = data
    pump, _ = _pump(h, segment_s=0.2)
    seg = _run_pump(pump)[0]
    assert seg.gap_s > 0
    assert seg.n_samples > 0            # kept pumping after the busy stretch


def test_pump_stops_promptly_when_asked():
    step = NPTS * DT
    pump, _ = _pump(_sequence_handlers([i * step for i in range(500)]),
                    segment_s=100.0)          # a segment that will never close
    stop = threading.Event()
    stop.set()
    assert list(pump.pump_segments(stop)) == []


# ── 把 Osci1T 还给用户 ────────────────────────────────────────────────────
#
# Osci1T 是**单实例共享模块**。2026-08-02 之前 configure() 改完通道和时基就再也不
# 还了 —— 用户本来在示波器上看着别的信号，MAST 一启动就被顶掉，而他不会收到任何
# 提示。这几条守着「动过就要还」。

def _prior_handlers(prior_ch: int = 0, prior_tb: int = 2,
                    timebases=(DT, 2 * DT, 4 * DT)) -> dict:
    """基础 handlers + 一台「用户已经调过」的示波器。

    prior_ch=0 是 "Z (m)"，而 pump 会切到 "Current (A)"(1)；prior_tb=2 是最慢的一
    档，而 pump 默认取最快(0)。两个都刻意与 pump 将要写入的值不同，否则测试无法
    区分「还原了」和「本来就是这个值」。
    """
    h = _base_handlers(timebases)
    h["Osci1T_ChGet"] = lambda a: _wire((prior_ch,))
    h["Osci1T_TimebaseGet"] = lambda a: _wire(
        (prior_tb,), (len(timebases),), [(v,) for v in timebases])
    return h


def _writes(pool, verb: str) -> list:
    return [c for c in pool.calls if c[0] == verb]


def test_configure_snapshots_the_scope_before_touching_it():
    pump, pool = _pump(_prior_handlers(prior_ch=0, prior_tb=2))
    pump.configure()
    assert pump._prior == {"channel_index": 0, "timebase_index": 2}
    # 快照必须发生在写入**之前**，否则记下的是我们自己刚写进去的值
    ch_get = next(i for i, c in enumerate(pool.calls) if c[0] == "Osci1T_ChGet")
    ch_set = next(i for i, c in enumerate(pool.calls) if c[0] == "Osci1T_ChSet")
    assert ch_get < ch_set


def test_restore_puts_the_operators_channel_and_timebase_back():
    pump, pool = _pump(_prior_handlers(prior_ch=0, prior_tb=2))
    pump.configure()
    assert _writes(pool, "Osci1T_ChSet")[-1][1] == (1,)        # 我们切到了 Current
    assert _writes(pool, "Osci1T_TimebaseSet")[-1][1] == (0,)  # 和最快时基

    done = pump.restore()

    assert done == {"channel_index": 0, "timebase_index": 2}
    assert _writes(pool, "Osci1T_ChSet")[-1][1] == (0,)        # 还回 Z (m)
    assert _writes(pool, "Osci1T_TimebaseSet")[-1][1] == (2,)  # 还回最慢档


def test_reconfigure_does_not_overwrite_the_original_snapshot(monkeypatch):
    """重配不能把还原目标改成「我们自己上一轮写进去的值」—— 那等于没还原。"""
    import mast.monitoring.pump as pump_mod
    monkeypatch.setattr(pump_mod, "_RECONFIG_MIN_INTERVAL_S", 0.0)
    pump, pool = _pump(_prior_handlers(prior_ch=0, prior_tb=2))
    pump.configure()

    # 第二轮 ChGet 现在会读回**我们**设的 1（模拟通道已被本次配置更改）
    pool.handlers["Osci1T_ChGet"] = lambda a: _wire((1,))
    pump._maybe_reconfigure("test")

    assert pump._prior == {"channel_index": 0, "timebase_index": 2}
    assert pump.restore()["channel_index"] == 0


def test_restore_is_a_noop_when_nothing_was_ever_captured():
    pump, pool = _pump(_base_handlers())      # 没有 Osci1T_ChGet handler
    before = len(pool.calls)
    assert pump.restore() == {"channel_index": None, "timebase_index": None}
    assert len(pool.calls) == before          # 没动过就一个字节都不发


def test_prior_snapshot_never_breaks_configure():
    """快照是「礼貌」，采集是「正事」。读不到原值只能放弃还原，不能让采集起不来。"""
    h = _base_handlers()
    h["Osci1T_ChGet"] = lambda a: NanonisCallRecord(
        method="Osci1T_ChGet", error="lock busy: role 'data' held by monitor")
    pump, _ = _pump(h)
    cfg = pump.configure()                    # 不抛
    assert cfg["channel_name"] == "Current (A)"
    assert pump._prior.get("channel_index") is None


def test_restore_survives_a_dead_link():
    """停机路径上抛异常，只会把优雅停机变成不优雅停机。"""
    pump, pool = _pump(_prior_handlers())
    pump.configure()
    pool.handlers["Osci1T_ChSet"] = lambda a: (_ for _ in ()).throw(OSError("link down"))
    pool.handlers["Osci1T_TimebaseSet"] = lambda a: (_ for _ in ()).throw(OSError("link down"))
    assert pump.restore() == {"channel_index": None, "timebase_index": None}


# ── 合成相对触发时间戳：t0 不变但波形改变 ────────────────────────────────
# 判新不能只看时间戳；这里由可重复的生成器检验内容指纹路径。

def _static_t0_handlers(timebases=(DT, 2 * DT, 4 * DT)):
    """合成 DataGet：时间戳固定，但每次内容都不同。"""
    h = _base_handlers(timebases)
    box = {"i": 0}

    def _data(a):
        box["i"] += 1
        return _data_reply(0.0, dt=DT, n=NPTS, value=100e-12 + box["i"] * 1e-13)

    h["Osci1T_DataGet"] = _data
    h["Osci1T_ChGet"] = lambda a: _wire((1,))     # 通道校验：确实是我们那一路
    return h


def test_segments_still_flow_when_t0_never_advances():
    """核心回归钉：t0 冻住而内容在变，必须照样产出段落。"""
    pump, _ = _pump(_static_t0_handlers(), segment_s=0.2)
    stop = threading.Event()
    segs = []
    for seg in pump.pump_segments(stop):
        segs.append(seg)
        if len(segs) >= 2:
            stop.set()
    assert len(segs) >= 2, "t0 不推进时一个段落都没产出 —— 根因回来了"
    assert pump.stats["fresh"] >= 4
    assert pump.stats["duplicate"] == 0, "内容每次都不同，不该有任何重复"
    assert segs[0].n_samples > 0


def test_a_truly_repeated_buffer_is_still_a_duplicate():
    """反向：t0 和内容都不动 = 真重复，不能因为放宽判据就把它当新数据。"""
    h = _base_handlers()
    h["Osci1T_DataGet"] = lambda a: _data_reply(0.0, dt=DT, n=NPTS, value=1e-12)
    pump, _ = _pump(h, segment_s=0.2)
    stop = threading.Event()

    def _run():
        for _ in pump.pump_segments(stop):
            break

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=2.0)
    stop.set()
    t.join(timeout=2.0)
    assert pump.stats["duplicate"] > 0, "逐位相同的缓冲被当成了新数据"
    assert pump.stats["fresh"] <= 1, "只有第一份算新，其余都是同一屏"


def test_dt_and_fs_come_from_the_reply_not_the_timebase_table():
    """时基表给出整屏时长，每点间隔以回包为准；混淆两者会把采样率误差放大为点数倍。"""
    pump, _ = _pump(_static_t0_handlers(), segment_s=0.2)
    pump.configure()
    assert pump.config["dt_s"] == 0.0            # configure() 之后仍是未知
    pump.poll_once()
    assert pump.config["dt_s"] == pytest.approx(DT)
    assert pump.config["fs_hz"] == pytest.approx(1.0 / DT)
    assert pump.config["window_s"] == pytest.approx(DT)   # 表里那档，语义是整屏

# 回包数值与单元素元组需要同时覆盖，防止替身只代表一种形态。

@pytest.mark.parametrize("shape", SHAPES)
def test_configure_works_for_both_element_shapes(shape):
    pump, _ = _pump(_base_handlers(shape=shape))
    cfg = pump.configure()
    assert cfg["channel_index"] == 1            # "Current (A)"
    assert cfg["channel_name"] == "Current (A)"
    # 刻意不断言 fs_hz:它要等第一份数据回包才知道(时基只给整屏时长),
    # configure() 之后本来就是 0 —— 两种形态都一样,与元素形态无关。


@pytest.mark.parametrize("shape", SHAPES)
def test_scalar_decodes_for_both_element_shapes(shape):
    """`_scalar` / `_array` 是形态真正影响到的地方 —— 纯函数,直接喂两种形态。
    (刻意**不**去驱动 `pump_segments` 那个循环:它是墙钟 + 事件驱动的,
    拿它测解码等于又写一条依赖时序的测试 —— 今晚已经为此付过两次学费。)"""
    from mast.monitoring.pump import _scalar
    assert _scalar(_scalar_as(1.5, shape)) == pytest.approx(1.5)


@pytest.mark.parametrize("shape", SHAPES)
def test_array_decodes_to_1d_for_both_element_shapes(shape):
    from mast.monitoring.pump import _array
    out = _array(_seq_as([1.0, 2.0, 3.0], shape))
    assert out.ndim == 1, "元组形态静默变成了 (N,1)"
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_both_shapes_decode_to_the_same_array():
    """同解。两种形态解出不同的数,比崩掉更难发现。"""
    from mast.monitoring.pump import _array
    vals = [100e-12 + i * 1e-15 for i in range(64)]
    tup = _array(_seq_as(vals, "tuple"))
    bare = _array(_seq_as(vals, "bare"))
    assert tup.shape == bare.shape
    assert np.allclose(tup, bare)


# ── 时基表自校验（2026-08-09，调研报告 §3 / §9.2）──────────────────────────
#
# `Osci1T/2T.TimebaseGet` 的回包是 (当前档位索引, 档数, 时基表)，而此前这里只取
# 第三项 —— 仪器自己报的「我有几档」解出来就丢了，**数组长度与它从不对账**。
# 于是「这张表完整吗」只能靠人在外部推导（报告 §3 拿六个值去除以最小值，对上
# 手册的 50/20/10/5/2/1 平均阶梯）。人能推一次，机器该每次都验。

def _tb(vals):
    return np.asarray(vals, dtype=np.float64)


def test_timebase_check_passes_on_a_consistent_synthetic_table():
    """合成时基的计数、索引及数值自洽时应通过。"""
    from mast.monitoring.pump import check_timebase_table
    values = _tb([0.05 * (2 ** i) for i in range(6)])
    assert check_timebase_table(4, 6, values) == "ok"


def test_timebase_check_catches_a_short_table():
    """报告 §9.2 点名的那一条:仪器说 6 档,数组里只有 3 个值。"""
    from mast.monitoring.pump import check_timebase_table
    out = check_timebase_table(0, 6, _tb([1.0, 2.0, 3.0]))
    assert out.startswith("mismatch") and "6" in out and "3" in out


def test_timebase_check_catches_an_index_outside_the_table():
    """越界索引说明回包没对齐 —— 而它会在停机时被写回仪器。"""
    from mast.monitoring.pump import check_timebase_table
    assert check_timebase_table(9, 3, _tb([1.0, 2.0, 3.0])).startswith("mismatch")


def test_timebase_check_rejects_values_that_are_not_timebases():
    from mast.monitoring.pump import check_timebase_table
    assert check_timebase_table(0, 3, _tb([1.0, 0.0, 3.0])).startswith("mismatch")
    assert check_timebase_table(0, 3, _tb([1.0, float("nan"), 3.0])).startswith("mismatch")


def test_unreadable_count_is_unverified_not_ok_and_not_a_fault():
    """「没法对账」既不是通过也不是故障 —— 必须是第三句话。

    合成一个通过、把「没测出来」印成 ok，是这个仓库反复付学费的那一类；
    印成 mismatch 则会让一台健康的机器天天报警。
    """
    from mast.monitoring.pump import check_timebase_table
    out = check_timebase_table(0, None, _tb([1.0, 2.0]))
    assert out.startswith("unverified")
    assert not out.startswith("mismatch") and out != "ok"


def test_configure_records_the_selfcheck_verdict():
    pump, _ = _pump(_base_handlers())
    cfg = pump.configure()
    assert cfg["timebase_check"] == "ok"


def test_a_disagreeing_table_is_reported_but_still_used():
    """可选档位字段缺失不能否定已经完整解析的数组；校验应保留字段可信度边界。"""
    h = _base_handlers()
    # 仪器自报 9 档，实际给 3 个值
    h["Osci1T_TimebaseGet"] = lambda a: _wire((0,), (9,),
                                              [(DT,), (2 * DT,), (4 * DT,)])
    pump, pool = _pump(h)
    cfg = pump.configure()
    assert cfg["timebase_check"].startswith("mismatch")
    assert cfg["window_s"] == pytest.approx(DT)          # 照样选了最快那一档
    assert ("Osci1T_TimebaseSet", (0,), "data") in pool.calls


def test_an_out_of_range_index_is_never_written_back_on_stop():
    """越界的「原值」不能进 restore() —— 把一个刚判定为不可信的索引写回仪器,
    比不还原更糟。这条让自校验承重,而不只是打印一行字。"""
    h = _base_handlers()
    h["Osci1T_ChGet"] = lambda a: _wire((0,))
    h["Osci1T_TimebaseGet"] = lambda a: _wire((77,), (3,),
                                              [(DT,), (2 * DT,), (4 * DT,)])
    pump, pool = _pump(h)
    pump.configure()
    assert pump._prior.get("timebase_index") is None
    assert pump.restore()["timebase_index"] is None
    # 通道那一半照旧还原 —— 一项不可信不该连累另一项
    assert pump._prior.get("channel_index") == 0


# ── on_idle：辅助通道搭车（→ #30，2026-08-06） ─────────────────
#
# 提速的前提不是把 cm_aux_interval_s 调小 —— 那个设置是节流的**上限**。真正的
# 约束是**机会**：采样原来只在段边界上有机会，一秒 0.75 次。这一组钉的是
# 「机会来自哪里、以什么代价、以及泵在什么时候不肯让」。

def test_on_idle_is_offered_many_times_per_segment():
    """一段电流里搭车的机会得远多于一次——否则 5 Hz 无从谈起。

    段长 0.2 s，一屏 1024 点 / 20 kHz = 51.2 ms，所以一段里有 ~4 次刷新等待。
    段边界那条老路一段只给一次机会；这里要看到明显更多。
    """
    ticks = {"n": 0}
    pump, _ = _pump(_sequence_handlers([i * NPTS * DT for i in range(200)]),
                    segment_s=0.2)
    stop = threading.Event()
    for _seg in pump.pump_segments(stop, on_idle=lambda _b: ticks.__setitem__(
            "n", ticks["n"] + 1)):
        stop.set()
        break
    assert ticks["n"] >= 3, f"一段里只让出了 {ticks['n']} 次空闲"


def test_on_idle_receives_the_budget_it_may_spend():
    """budget 不是装饰：搭车者要拿它压自己的锁等待（见 aux_channels._lock_timeout）。

    给出去的每一个数都必须是正的、有限的、且不大于泵这一觉的长度上限 ——
    一个 0 或负数的 budget 会让搭车者把锁等待压到底，于是「扫描期间一次都采不到」。
    """
    seen: list[float] = []
    pump, _ = _pump(_sequence_handlers([i * NPTS * DT for i in range(200)]),
                    segment_s=0.2)
    stop = threading.Event()
    for _seg in pump.pump_segments(stop, on_idle=seen.append):
        stop.set()
        break
    assert seen, "一次都没让"
    from mast.monitoring.pump import _IDLE_MIN_SLACK_S, _SLEEP_MAX_S
    for b in seen:
        assert _IDLE_MIN_SLACK_S <= b <= _SLEEP_MAX_S


def test_on_idle_is_not_offered_when_the_data_role_is_busy():
    """role 忙**正说明别人拿着这把锁**，那时候搭车只会排在那次占用后面。

    这是整个设计要避开的那件事，所以退避里一次都不能让 —— 而退避恰恰是最长的
    那种「空闲」，最容易被顺手接上去。
    """
    state = {"i": 0}
    offers_during_busy = {"n": 0}
    busy = {"on": False}

    def data(_args):
        i = state["i"]
        state["i"] += 1
        busy["on"] = 3 <= i < 6
        if busy["on"]:
            return NanonisCallRecord(method="Osci1T_DataGet",
                                     error="RoleBusy: data held")
        return _data_reply(i * NPTS * DT)

    def on_idle(_b):
        if busy["on"]:
            offers_during_busy["n"] += 1

    h = _base_handlers()
    h["Osci1T_DataGet"] = data
    pump, _ = _pump(h, segment_s=0.2)
    stop = threading.Event()
    for _seg in pump.pump_segments(stop, on_idle=on_idle):
        stop.set()
        break
    assert offers_during_busy["n"] == 0


def test_a_slow_hitch_hiker_does_not_push_the_pump_off_phase():
    """搭车花掉的时间从**这一觉里扣**，不是加在后面。

    加在后面就是把泵的相位往后推，一次推过一个刷新窗口就丢一屏采样。这条是
    「提速不许削弱电流那一路」的可执行形式：同一段数据，让与不让，泵发出去的
    轮询次数不该有量级差别。
    """
    import time as _t

    def run(on_idle):
        pump, pool = _pump(_sequence_handlers([i * NPTS * DT for i in range(400)]),
                           segment_s=0.3)
        stop = threading.Event()
        t0 = _t.monotonic()
        for _seg in pump.pump_segments(stop, on_idle=on_idle):
            stop.set()
            break
        return _t.monotonic() - t0, sum(1 for c in pool.calls
                                        if c[0] == "Osci1T_DataGet")

    plain_s, plain_polls = run(None)
    # 每次让都花掉 5 ms —— 远小于一屏 51 ms，应当被这一觉吸收掉。
    slow_s, slow_polls = run(lambda _b: _t.sleep(0.005))
    assert slow_polls == pytest.approx(plain_polls, rel=0.35), (
        f"搭车改变了轮询次数：{plain_polls} → {slow_polls}")
    assert slow_s < plain_s + 0.5


def test_a_throwing_hitch_hiker_never_breaks_the_pump():
    """辅助通道绝不反噬电流采集——包括它自己炸了的时候。"""
    def boom(_b):
        raise RuntimeError("aux exploded")

    pump, _ = _pump(_sequence_handlers([i * NPTS * DT for i in range(200)]),
                    segment_s=0.2)
    stop = threading.Event()
    out = []
    for seg in pump.pump_segments(stop, on_idle=boom):
        out.append(seg)
        stop.set()
        break
    assert out and out[0].n_samples > 0


def test_pump_segments_without_on_idle_is_unchanged():
    """不给回调时行为与从前逐字一致——这是老调用方（和老测试）的保证。"""
    pump, _ = _pump(_sequence_handlers([i * NPTS * DT for i in range(200)]),
                    segment_s=0.2)
    seg = _run_pump(pump)[0]
    assert seg.n_samples > 0 and seg.fs_hz == pytest.approx(20000.0)


# ══ Osci2T（2026-08-09，调研报告 §10 首选建议）═══════════════════════════════
#
# **不是提速。** 采样率由 Signals Period（全局设置）定，1T 与 2T 完全一样（本机
# 2 kHz），所以全部电流阈值一个都不用重标 —— 这正是选 2T 而不是「把 Signals
# Oversampling 调小」的理由（后者会把一个此刻从不触发的 rms 告警变成每 14 秒响
# 一次，报告 §8.2）。换来的是一次 TCP 往返顶从前 50 次。
#
# 2T 与 1T 的结构差别只有一条，但它渗到三个地方：**时基改的是点数，不是速率。**

T2_DT = 1e-3                 # 合成 1 kHz 时序，便于计算期望值
#: 【手册】Oscilloscope 2T：点数只能是这六个值之一。
T2_POINTS = (256, 512, 1280, 2560, 5120, 12800)
T2_WINDOWS = tuple(n * T2_DT for n in T2_POINTS)      # 0.256 … 12.8 s


def _t2_data_reply(t0: float, dt: float = T2_DT, n: int = 1280,
                   value_a: float = 100e-12, value_b: float = 5e-9,
                   shape: str = "tuple") -> tuple:
    """``[t0, dt, sizeA, dataA, sizeB, dataB]`` —— 一个 dt 管两路。

    B 路刻意给一个**量级完全不同**的值（nA 级的 Z 电压 vs pA 级的电流）：
    解码若错拿了 B，任何一个幅度断言都会当场炸掉，而不是安静地对上。
    """
    return _wire(t0, _scalar_as(dt, shape), _scalar_as(n, shape),
                 _seq_as([value_a + i * 1e-15 for i in range(n)], shape),
                 _scalar_as(n, shape),
                 _seq_as([value_b + i * 1e-12 for i in range(n)], shape))


def _t2_handlers(*, prior_a: int = 0, prior_b: int = 7,
                 windows=T2_WINDOWS, n_points: int = 1280,
                 shape: str = "tuple") -> dict:
    """一台**有状态**的假 Osci2T：``ChsSet`` 写进去的，``ChsGet`` 就读得回来。

    ⚠️ 静态的 ChsGet（第一版就是）会永远回报动手**之前**的通道，于是
    ``channel_is_ours()`` 每一屏都判 False、每一屏都当成「被别人改走了」整屏丢弃，
    泵一段都不产出 —— 而在 ``for seg in pump.pump_segments(…)`` 的写法下这不是
    一次失败，是**挂死**（600 s 超时才发现）。替身必须能表达「我们写进去了」，
    否则它测的是一台谁也没配置过的示波器。

    ``t0`` 按 ``n·dt`` 连续推进：每屏跳一个不相干的常数会让每一屏都被判成
    「时钟倒退」，于是 discontinuity 恒为真 —— 那样点数变化那条测试会**因为一个
    与它无关的原因**而通过。
    """
    state = {"ch": (int(prior_a), int(prior_b)), "i": 0, "t": 0.0}

    def _chs_set(args):
        state["ch"] = (int(args[0]), int(args[1]))
        return NanonisCallRecord(method="Osci2T_ChsSet")

    def _data(a):
        state["i"] += 1
        t0 = state["t"]
        state["t"] = t0 + n_points * T2_DT
        return _t2_data_reply(t0, n=n_points, shape=shape,
                              value_a=100e-12 + state["i"] * 1e-13)

    return {
        "Osci2T_Run": lambda a: NanonisCallRecord(method="Osci2T_Run"),
        "Signals_NamesGet": lambda a: _names_reply(
            ["Z (m)", "Current (A)", "Bias (V)"], shape),
        "Osci2T_ChsGet": lambda a: _wire(_scalar_as(state["ch"][0], shape),
                                         _scalar_as(state["ch"][1], shape)),
        "Osci2T_ChsSet": _chs_set,
        "Osci2T_TimebaseGet": lambda a: _wire(
            _scalar_as(0, shape), _scalar_as(len(windows), shape),
            _seq_as(list(windows), shape)),
        "Osci2T_TimebaseSet": lambda a: NanonisCallRecord(method="Osci2T_TimebaseSet"),
        "Osci2T_TrigSet": lambda a: NanonisCallRecord(method="Osci2T_TrigSet"),
        "Osci2T_DataGet": _data,
        "Util_RTFreqGet": lambda a: _wire(_scalar_as(20000.0, shape)),
    }


def _pump2(handlers, **kw):
    from mast.monitoring.pump import Osci2TPump
    pool = FakePool(handlers)
    return Osci2TPump(lambda: pool, **kw), pool


def _collect(pump, n_segments: int, on_idle=None) -> list[Segment]:
    stop = threading.Event()
    out: list[Segment] = []
    for seg in pump.pump_segments(stop, on_idle=on_idle):
        out.append(seg)
        if len(out) >= n_segments:
            stop.set()
            break
    return out


def test_osci2t_never_touches_the_other_scope():
    """一个字节都不许发到 Osci1T 上。

    这不是洁癖:``nanonis_patch`` 的注释写着 v1.0.9 把大半个 Osci2T 家族接到了
    **Osci1T 的命令名**上,而后果「不是一次干净的失败 —— 一个自以为在配 Osci2T
    的调用方,静默地重配了另一台示波器」。现在那台示波器上跑的可能正是回退后的
    电流监控自己。
    """
    pump, pool = _pump2(_t2_handlers())
    pump.configure()
    assert not [c for c in pool.calls if c[0].startswith("Osci1T")]
    assert {role for _, _, role in pool.calls} == {"data"}


def test_osci2t_picks_the_longest_screen_within_the_target():
    """方向与 1T **相反**,而且这不是口味问题。

    1T 取 argmin(最快);2T 每一档速率相同,短的那档除了「一次往返拿回来的数据
    更少」之外没有任何好处 —— 取 argmin 等于选了最差的一档,而且**测不出来**:
    采样率、阈值、判据全都一模一样,只有往返次数悄悄多了 50 倍。
    """
    pump, pool = _pump2(_t2_handlers(), window_target_s=2.6)
    cfg = pump.configure()
    assert cfg["window_s"] == pytest.approx(2.56)          # ≤2.6 的最长一档
    assert ("Osci2T_TimebaseSet", (T2_WINDOWS.index(2.56),), "data") in pool.calls
    # 反证：取 argmin（1T 的方向）会选到 0.256，两者必须真的分得开
    assert cfg["window_s"] != pytest.approx(min(T2_WINDOWS))


def test_osci2t_falls_back_to_the_shortest_when_every_screen_is_too_long():
    pump, _ = _pump2(_t2_handlers(), window_target_s=0.01)
    assert pump.configure()["window_s"] == pytest.approx(min(T2_WINDOWS))


def test_osci2t_decodes_channel_a_and_ignores_channel_b():
    """六字段回包里 A 是电流。拿错 B 不会报错 —— 它只会静默地把 Z 存成电流。"""
    h = _t2_handlers()
    h["Osci2T_DataGet"] = lambda a: _t2_data_reply(1.0, value_a=100e-12,
                                                   value_b=5e-9)
    pump, _ = _pump2(h)
    pump.configure()
    chunk = pump.poll_once()
    assert chunk.n == 1280
    assert chunk.dt == pytest.approx(T2_DT)
    assert chunk.y[0] == pytest.approx(100e-12)            # A 路，不是 5e-9
    assert float(chunk.y.max()) < 1e-9


@pytest.mark.parametrize("shape", SHAPES)
def test_osci2t_decodes_both_element_shapes(shape):
    """1-元组与裸值两种形态都要走通 —— 见文件上方 SHAPES 的说明。"""
    h = _t2_handlers(shape=shape)
    h["Osci2T_DataGet"] = lambda a: _t2_data_reply(1.0, shape=shape)
    pump, _ = _pump2(h)
    pump.configure()
    chunk = pump.poll_once()
    assert chunk.y.ndim == 1 and chunk.y.size == 1280
    assert chunk.y[0] == pytest.approx(100e-12)


def test_osci2t_leaves_the_operators_second_channel_alone():
    """B 路写回用户原来那一路,不是我们想看的那一路。

    setter 必须同时给两路,所以「B 给什么」躲不掉。写自己的东西有两个代价:
    长期占走别人的双通道示波器,以及和 zburst(每 30 min 把 B 设成 Z)打起来。
    """
    pump, pool = _pump2(_t2_handlers(prior_a=0, prior_b=7))
    pump.configure()
    sets = [c for c in pool.calls if c[0] == "Osci2T_ChsSet"]
    assert sets and sets[-1][1] == (1, 7)      # A=Current(1)，B=用户的 7


def test_osci2t_restore_puts_both_channels_back():
    pump, pool = _pump2(_t2_handlers(prior_a=0, prior_b=7))
    pump.configure()
    done = pump.restore()
    assert done["channel_index"] == 0
    assert [c for c in pool.calls if c[0] == "Osci2T_ChsSet"][-1][1] == (0, 7)


# ── channel_is_ours：第二通道的加入不能让它误判 ─────────────────────────────

def test_channel_check_is_by_position_not_membership():
    """**这条是本次切换最容易写错的那一行。**

    ``ChsGet`` 回来的是两个索引。写成「我们那一路在不在这两个里面」看起来更宽容,
    却恰好漏掉最坏的情形:别人把电流挪到了 B、A 换成了 Bias —— 那时「在里面」为真,
    而 ``DataGet`` 的 **dataA** 已经是 Bias 了。Bias(~1 V)在电流通道上每个样本都
    过 90 nA 饱和阈,三段之后就是一条能停掉正在跑的复合技能的 CRITICAL。
    """
    h = _t2_handlers(prior_a=0, prior_b=7)
    pump, pool = _pump2(h)
    pump.configure()
    assert pump.channel_is_ours() is True
    # 电流(1)被挪到了 B 位,A 位换成了别人 —— 「在里面」为真,但位置错了
    pool.handlers["Osci2T_ChsGet"] = lambda a: _wire((2,), (1,))
    assert pump.channel_is_ours() is False, "按「在不在里面」判的话这里会漏"


def test_channel_check_ignores_the_second_position():
    """位置 1 刻意不判 —— 它不是电流,而它会被 zburst / ConfigureDualScope
    合法地改走。判它 = 每半小时丢一次完全好的电流数据。"""
    pump, pool = _pump2(_t2_handlers(prior_a=0, prior_b=7))
    pump.configure()
    pool.handlers["Osci2T_ChsGet"] = lambda a: _wire((1,), (30,))   # B 被改走
    assert pump.channel_is_ours() is True


def test_a_half_read_channel_reply_is_no_opinion_not_a_mismatch():
    """只读回一路的 2T 回包 = 没对齐。位置是这里唯一的信息,位置不明就别下结论。"""
    pump, pool = _pump2(_t2_handlers())
    pump.configure()
    pool.handlers["Osci2T_ChsGet"] = lambda a: _wire((1,))
    assert pump.channel_is_ours() is None


# ── 一屏 6.4 s vs 段 1.0 s ───────────────────────────────────────────────────

def test_one_long_screen_is_split_into_segments_of_the_configured_length():
    """段长仍由 cm_segment_s 决定,不由一屏多长决定。

    ``cm_crit_consecutive`` 数的是「连续多少**段**」。让段长跟着一屏走,那三条
    能停实验的 CRITICAL 规则的门槛就跟着变了 —— 而没有任何一个阈值被重标过。
    """
    pump, _ = _pump2(_t2_handlers(), segment_s=0.2)     # 一屏 1.28 s = 6.4 段
    segs = _collect(pump, 6)
    assert len(segs) == 6
    for seg in segs:
        assert seg.n_samples == 200                     # 正好 0.2 s @1 kHz
        assert seg.fs_hz == pytest.approx(1.0 / T2_DT)
        assert seg.gap_s == 0.0 and len(seg.runs) == 1
    # 段落时间戳必须跟着切,不能六段共用一个起点
    starts = [s.t_start for s in segs]
    assert starts == sorted(starts)
    # 容差 1 µs 而不是 1 ns：t_start 是**绝对 unix 秒**，float64 在 1.7e9 附近的
    # ulp 就有 0.24 µs —— 比一个采样间隔（本测试为 1 ms）细三个数量级，
    # 够用，但比 1 ns 粗。
    assert starts[1] - starts[0] == pytest.approx(0.2, abs=1e-6)


def test_splitting_loses_no_samples_across_screens():
    """切开的余数接到下一段的头上 —— 一个样本都不许丢。

    一屏 1280 点、段 200 点 ⇒ 每屏余 80 点。第 7 段必须由「上屏的 80 + 本屏的
    120」组成,而不是从本屏重新开始。
    """
    pump, _ = _pump2(_t2_handlers(), segment_s=0.2)
    segs = _collect(pump, 8)
    assert all(s.n_samples == 200 for s in segs)
    total = sum(s.n_samples for s in segs)
    assert total == 1600                                 # 1280 + 320，无重叠无丢失


def test_the_channel_is_verified_once_per_screen_not_once_per_segment():
    """覆盖没变松（整屏一起丢弃），往返从 6 次降到 1 次。"""
    pump, pool = _pump2(_t2_handlers(), segment_s=0.2)
    _collect(pump, 6)                                    # 六段全来自同一屏
    assert len([c for c in pool.calls if c[0] == "Osci2T_ChsGet"]) <= 2


def test_a_poisoned_screen_discards_every_segment_in_it():
    """通道被改走时整屏作废 —— 不「抢救」已经切出来的那几段。"""
    h = _t2_handlers(prior_a=0, prior_b=7)
    pump, pool = _pump2(h, segment_s=0.2)
    pump.configure()
    pool.handlers["Osci2T_ChsGet"] = lambda a: _wire((2,), (7,))   # A 被改走
    # 走线程 + 超时,不是 `for … break`:通过的时候这个生成器**永远不 yield**,
    # 而 `for` 循环里的 break 只有在拿到一段之后才会执行 —— 用 for 写就是
    # 「测试通过 = 挂死」。(第一版正是这么写的,600 s 才发现。)
    stop = threading.Event()
    got: list = []

    def _run():
        for seg in pump.pump_segments(stop):
            got.append(seg)
            break

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=2.0)
    stop.set()
    t.join(timeout=2.0)
    assert got == [], "通道已被改走，却还是落了段"
    assert pump.stats["fresh"] > 0, "根本没取到数据，这条测试什么都没证明"


def test_a_point_count_change_is_a_discontinuity_even_though_dt_never_moves():
    """Osci2T 改时基可能改变点数而不改变采样间隔；检测配置变化不能只比较 dt。"""
    box = {"i": 0, "t": 0.0}

    def _data(a):
        box["i"] += 1
        n = 1280 if box["i"] <= 1 else 256
        t0 = box["t"]
        box["t"] = t0 + n * T2_DT            # 时间轴连续：不许靠「时钟倒退」蒙混
        return _t2_data_reply(t0, dt=T2_DT, n=n,
                              value_a=100e-12 + box["i"] * 1e-13)

    h = _t2_handlers()
    h["Osci2T_DataGet"] = _data
    pump, _ = _pump2(h, segment_s=0.2)
    segs = _collect(pump, 7)
    assert segs[-1].discontinuity is True, "点数变了而 dt 没变 —— 没被发现"
    assert pump.stats["discontinuity"] >= 1
    # 反证：前六段（都来自第一屏，点数还没变）不许被误报成不连续，
    # 否则这条测试只是在证明「什么都会被判成不连续」。
    assert not any(s.discontinuity for s in segs[:6])


def test_segments_carry_the_strategy_that_produced_them():
    """``Segment.source`` 从此是活字段。

    在此之前它有默认值 ``"osci1t"``、**全仓零处赋值**,却一路写进了 segments 表
    并被 export.py 读走 —— 换采集源时它不会跟着变,也没有任何测试会红。
    """
    pump2, _ = _pump2(_t2_handlers(), segment_s=0.2)
    assert _collect(pump2, 1)[0].source == "osci2t"
    pump1, _ = _pump(_sequence_handlers([i * NPTS * DT for i in range(200)]),
                     segment_s=0.2)
    assert _run_pump(pump1)[0].source == "osci1t"


# ── 一屏变长不许把辅助通道的采样机会压没 ─────────────────────────────────────

def test_the_sleep_cap_follows_the_screen_length():
    """固定 0.5 s 的上限会把「一屏轮一次」变成「一屏轮十三次」。"""
    from mast.monitoring.pump import _SLEEP_MAX_S, _sleep_cap
    assert _sleep_cap(0.128) == _SLEEP_MAX_S          # 1T：一点没变
    assert _sleep_cap(6.4) >= 6.4                     # 2T：至少睡得完一屏


def test_a_long_sleep_still_offers_many_idle_slots():
    """**辅助通道的 5 Hz 不许被一屏 6.4 s 打成 0.17 Hz。**

    从前一觉最多 0.5 s，「睡前让一次」就等于每屏一次；一觉变成 5.9 s 之后，
    只让一次会把 #30 刚修好的提速整个还回去。所以一觉要切片。
    """
    from mast.monitoring.pump import _IDLE_SLICE_S, _sleep_with_idle
    seen: list[float] = []
    stop = threading.Event()
    _sleep_with_idle(stop, 0.6, seen.append)
    assert len(seen) >= 8, f"0.6 s 的一觉只让了 {len(seen)} 次"
    assert all(0 < b <= _IDLE_SLICE_S + 1e-9 for b in seen)


def test_a_stopped_pump_leaves_the_sliced_sleep_immediately():
    """切片不能让 stop 变钝 —— 每一片之后都要看一眼。"""
    from mast.monitoring.pump import _sleep_with_idle
    stop = threading.Event()

    def _idle(_b):
        stop.set()

    t0 = time.monotonic()
    assert _sleep_with_idle(stop, 5.0, _idle) is True
    assert time.monotonic() - t0 < 1.0


# ── 策略选择与回退 ───────────────────────────────────────────────────────────

def test_make_pump_uses_osci2t_when_the_module_answers():
    from mast.monitoring.pump import Osci2TPump, make_pump
    pool = FakePool(_t2_handlers())
    pump = make_pump(lambda: pool, strategy="osci2t")
    assert isinstance(pump, Osci2TPump)
    assert pump.STRATEGY == "osci2t" and pump.fallback_note == ""


def test_make_pump_falls_back_when_osci2t_is_not_loaded():
    """**授权 ≠ 已加载。** 许可证里有 Oscilloscope 2T,但模块要前面板开着才响应。

    不回退的话,2T 成了默认之后电流监控会整个停掉(服务把 PumpUnavailable 当
    「60 秒后再探」,永远地)。回退还得说出来 —— 静默降级和故障一样难查。
    """
    from mast.monitoring.pump import Osci1TPump, make_pump
    h = dict(_t2_handlers())
    h["Osci2T_TimebaseGet"] = lambda a: NanonisCallRecord(
        method="Osci2T_TimebaseGet", error="NeedModule: Osci2T not loaded")
    pool = FakePool(h)
    pump = make_pump(lambda: pool, strategy="osci2t")
    assert isinstance(pump, Osci1TPump)
    assert "osci2t" in pump.fallback_note and "osci1t" in pump.fallback_note


def test_make_pump_probes_with_a_read_that_writes_nothing():
    """探测只能用不带参数的 getter：名字错了或模块没开，代价都只是一次被拒绝的
    读，绝不会误配到别的示波器上去。"""
    from mast.monitoring.pump import make_pump
    pool = FakePool(_t2_handlers())
    make_pump(lambda: pool, strategy="osci2t")
    probes = [c for c in pool.calls if c[0].startswith("Osci2T")]
    assert probes and all(c[0].endswith("Get") for c in probes)


def test_osci1t_remains_reachable_as_an_explicit_choice():
    """回退路径要有测试,这就是它 —— 配置项设回 1T 时,连探都不探 2T。"""
    from mast.monitoring.pump import Osci1TPump, make_pump
    pool = FakePool(_base_handlers())
    pump = make_pump(lambda: pool, strategy="osci1t")
    assert isinstance(pump, Osci1TPump) and pump.fallback_note == ""
    assert not [c for c in pool.calls if c[0].startswith("Osci2T")]
