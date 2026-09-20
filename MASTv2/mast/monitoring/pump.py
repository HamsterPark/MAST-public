"""Segment pump: oscilloscope buffers to tunnelling-current segments.

Osci1T returns the displayed buffer in one round trip. Its timebase describes
whole-buffer duration; sample spacing and rate come from the returned ``dt``.
Use non-trigger-waiting reads so an absent trigger cannot hold the shared data
role until the socket times out.

Freshness uses both timestamp and content: only an unchanged timestamp AND
unchanged samples are duplicates. Relative trigger timestamps may remain
constant, while synthetic constant signals can retain identical samples with
advancing timestamps. Both cases must work.

Gap accounting uses instrument timestamps when they advance and host time
otherwise. The tolerance reflects the active clock's resolution. Gaps remain
explicit; concatenating across them would introduce artificial spectral steps.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Role the pump reads on. 6503 is the only quiet one of the four TCP roles —
#: ``monitor`` already carries ~20 transactions/s including the tip-crash
#: watchdog, and a blocking read on ``main`` can drain the API thread pool.
DATA_ROLE = "data"

#: Polling cadence, phase-locked to the buffer refill.
#:
#: The floor is physics: the scope overwrites its buffer every ``n·dt`` (51 ms at
#: 20 kHz / 1024 pts), so one successful fetch per refill — about 20/s — is the
#: least we can do without dropping samples. The cost of getting the cadence
#: wrong is real: a naive "sleep a small fixed slice after a duplicate" loop was
#: measured at 45 calls/s (≈45% duty on the data role) because it probed four
#: times into every refill window.
#:
#: So instead of polling blindly, aim: after a fresh frame, sleep most of a
#: trace; after a duplicate, sleep the REMAINDER of the current refill window,
#: computed from how long ago the last fresh frame arrived. Steady state
#: converges to roughly one probe per refill, and a missed prediction self-
#: corrects on the next iteration because the remainder is then near zero.
_SLEEP_FRESH_FRAC = 0.92        # just short of a full refill, so we rarely overshoot
_SLEEP_GUARD_S = 0.002          # nudge past the boundary rather than landing on it
_SLEEP_MIN_S = 0.005
_SLEEP_MAX_S = 0.5

#: 把「本来就要睡掉的那段空闲」借给一个搭车的读数（``on_idle``）之前，至少要有
#: 这么多空闲。
#:
#: 这条闸门存在的理由 —— 以及 ``on_idle`` 为什么必须收到 budget：
#: 泵与搭车者在**同一条线程、同一个 data role** 上。同线程意味着两者不可能互相
#: 抢锁（一条线程不会阻塞在自己持有的锁上），这正是搭车比另起一条线程安全的
#: 全部理由。但**别人**可以持有 data role（扫描抓帧就在这个 role 上），于是搭车
#: 者的那次调用可能卡在锁上 —— 而它卡多久，决定了泵会不会错过下一次缓冲刷新。
#: 所以 budget 不是装饰：搭车者必须把自己的锁等待压到 budget 以内。
#:
#: 睡眠本身是**绝对时刻**对齐的（见循环尾部的 ``target``），所以只要搭车读数
#: 落在 budget 里，泵的相位一个采样点都不差；万一超了，``_clamp(..., 0.0, …)``
#: 让它立刻去轮询，重复帧那条路径会在下一拍把相位拉回来。
_IDLE_MIN_SLACK_S = 0.02

#: 一觉里每隔这么久让搭车者上一次车。见 :func:`_sleep_with_idle` ——
#: 采样机会必须与「一屏多长」解耦,否则 Osci2T 的 6.4 s 一屏会把辅助通道的
#: 5 Hz 打成 0.17 Hz。20 Hz 与 Osci1T 上的既有机会率同量级。
_IDLE_SLICE_S = 0.05

#: Back-off when the data role is busy (a scan frame grab holds it briefly).
_BUSY_BACKOFF_START_S = 0.5
_BUSY_BACKOFF_MAX_S = 8.0

#: Consecutive malformed replies before we re-probe the module configuration.
_BAD_REPLY_LIMIT = 10

#: Consecutive duplicate buffers before we suspect the scope stopped re-arming
#: (trigger changed, module stopped, front panel closed) and re-configure.
_DUP_STREAK_LIMIT = 40

#: The channel is verified at every SEGMENT BOUNDARY rather than on a timer:
#: the segment is the unit that reaches storage, so checking there guarantees
#: nothing is stored unverified. Osci1T is a SINGLE shared module —
#: AcquireOsciTrace and the optional-scope skills point it wherever they like,
#: on a different TCP role and without taking an instrument token (READ-category
#: skills never take one), so nothing serialises us against them.

#: Never re-configure more than this often — an operator fiddling with the
#: scope front panel must not turn into a configuration storm.
_RECONFIG_MIN_INTERVAL_S = 30.0

#: Candidate substrings for the tunnelling-current signal, best first.
_CURRENT_HINTS = ("current (a)", "current")


class PumpUnavailable(Exception):
    """The oscilloscope module is not available (not loaded / not licensed).

    Not an error condition — the bundled simulator does not load Osci1T at all.
    The service reports it and retries quietly.
    """


class PumpPaused(Exception):
    """Transient: link down or role busy. Back off, keep the configuration."""


@dataclass
class TraceChunk:
    """One decoded ``Osci1T_DataGet`` reply."""

    t0: float                    # oscilloscope clock, first sample
    dt: float
    n: int
    y: np.ndarray                # float64
    host_ts: float               # wall clock when the reply landed


@dataclass
class Segment:
    """~1 s of current handed to the service.

    ``runs`` holds contiguous stretches. More than one means samples were lost
    in between; ``gap_s`` says how many seconds. Spectra are computed per run,
    amplitude statistics over everything.
    """

    t_start: float               # host epoch of the first sample
    t_end: float
    osci_t0: float               # oscilloscope clock of the first sample
    fs_hz: float
    runs: list[np.ndarray] = field(default_factory=list)
    n_samples: int = 0
    gap_s: float = 0.0
    discontinuity: bool = False
    channel_name: str = ""
    source: str = "osci1t"

    @property
    def samples(self) -> np.ndarray:
        if not self.runs:
            return np.empty(0)
        return self.runs[0] if len(self.runs) == 1 else np.concatenate(self.runs)


def _decoded(rv: Any) -> list:
    """Unwrap the ``(header, raw, fields)`` tuple nanonis_spm returns."""
    if isinstance(rv, tuple) and len(rv) >= 3 and isinstance(rv[2], list):
        return rv[2]
    if isinstance(rv, list):
        return rv
    return []


def _scalar(v) -> float:
    """Decoded scalars sometimes arrive wrapped in a 1-tuple."""
    if isinstance(v, (tuple, list)):
        return float(v[0]) if v else 0.0
    return float(v)


def _array(v) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return np.asarray(v, dtype=np.float64).reshape(-1)
    if not isinstance(v, (list, tuple)):
        return np.empty(0)
    if v and isinstance(v[0], (tuple, list)):
        return np.asarray([x[0] if x else 0.0 for x in v], dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def _int_or_none(v) -> Optional[int]:
    """整数,读不出来就是 ``None``。

    刻意不是 ``int(_scalar(v))``:后者把「读不出来」变成一次异常或一个 0,而这里
    的两个调用方(档数、当前档位索引)都需要把「没读到」与「读到 0」分开 ——
    索引 0 是合法的第一档,档数 0 是一张空表。
    """
    try:
        f = _scalar(v)
    except (TypeError, ValueError, IndexError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return int(f)


def check_timebase_table(current_index: Optional[int],
                         declared_count: Optional[int],
                         values: np.ndarray) -> str:
    """核对仪器自报的时基表，返回 ok、unverified 或 mismatch 说明。
    
    检查声明档数与数组长度、当前索引是否在表内、时基是否为有限正数。
    越界索引可能影响恢复仪器设置，因此诊断必须清楚区分未知与不一致。
    
    这些检查是诊断性的，当前实现不会据此否决整张表。若要改成阻断策略，
    需先验证目标协议的字段布局与兼容性；不能把未确认的字段解释当成已标定事实。"""
    problems: list[str] = []
    unknown: list[str] = []
    n = int(values.size)

    if declared_count is None:
        unknown.append("仪器没报档数(回包第 2 个字段读不出来)")
    elif int(declared_count) != n:
        problems.append(f"仪器自报 {int(declared_count)} 档,数组里有 {n} 个值")

    if current_index is None:
        unknown.append("当前档位索引读不出来")
    elif not (0 <= int(current_index) < n):
        problems.append(f"当前档位索引 {int(current_index)} 落在 [0,{n}) 之外")

    bad = int(np.count_nonzero(~np.isfinite(values) | (values <= 0)))
    if bad:
        problems.append(f"{bad}/{n} 个时基不是有限正数")

    if problems:
        return "mismatch: " + ";".join(problems)
    if unknown:
        # 「没测出来」与「测出来对不上」是两句话。前者不是仪器的毛病,也不该
        # 长得像一次通过。
        return "unverified: " + ";".join(unknown)
    return "ok"


class _ScopePump:
    """轮询示波器并把回包拼成 Segment；通用循环不依赖协议方言。
    
    Osci1TPump 与 Osci2TPump 只提供调用方法和解码器，共用判新、拼接、缺口计量、
    分段、相位锁定等待、通道校验及重配节流，避免修复只覆盖某条路径。
    硬件调用动词保留在子类 safe_call 的字符串字面量中，供中止、安全与覆盖审计检查。"""

    #: 上报给 ``/api/monitoring/status`` 的策略名。**由实例派生,不是字面量** ——
    #: 在此之前 ``service.status()`` 无条件写死 ``"osci1t"``,于是这个字段看起来
    #: 是动态的、实际上任何时候都只会是那一个值。
    STRATEGY: str = "?"

    #: Nanonis 里那个模块**本来的名字**,给人看的(日志、状态明细、标定报告)。
    #: 与 :data:`STRATEGY` 分开:后者是 API 上的取值(小写 token),前者要能让用户
    #: 在 Nanonis 的 Graphs 菜单里照着找。
    MODULE_NAME: str = "?"

    #: 这个方言用几路通道。位置 0 永远是**电流** —— 全部阈值、全部判据说的都是它。
    N_CHANNELS: int = 1

    def __init__(self, pool_getter: Callable[[], Any],
                 *, segment_s: float = 1.0, target_fs_hz: float = 0.0,
                 signal_hint: str = "current", window_target_s: float = 0.0):
        self._pool_getter = pool_getter
        self._segment_s = max(0.2, float(segment_s))
        self._target_fs_hz = max(0.0, float(target_fs_hz))
        self._signal_hint = signal_hint.lower()
        #: 想要多长的一屏(秒)。只有 Osci2T 用得上 —— 它的时基改的是**点数**,
        #: 不是速率,所以「一屏多长」是个可选项而不是副产品。0 = 用方言默认。
        self._window_target_s = max(0.0, float(window_target_s))
        self._cfg: dict = {}
        self._last_reconfig = 0.0
        self._bad_replies = 0
        self.last_busy = False
        self._timebases: list[float] = []
        self._timebase_index = -1
        #: 时基表自校验的结论(见 :func:`check_timebase_table`)。``""`` = 还没读过表。
        #: 上报出去而不是只写日志:冻结版的 stdout 是被吞掉的(``console=False``),
        #: 一条只存在于日志里的自校验,和没有自校验的区别在真机上等于零。
        self._timebase_check = ""
        #: Osci1T 是**单实例共享模块**：用户可能本来就在上面看别的信号。这里记下
        #: 我们动手**之前**的通道与时基，停止时还回去 —— 只在第一次 configure() 时
        #: 记录，后续的 _maybe_reconfigure() 不覆盖它（否则还原目标会变成我们自己
        #: 上一轮写进去的值，等于没还原）。
        #:
        #: 还不回去的两样，如实记在这里：``*_Run``（协议没有对应的
        #: run-state 读取）与触发模式（``Osci1T_TrigGet`` 在 nanonis_spm 1.0.9 里
        #: 响应规格是空的、必然误解析，读不到原值就无从还原）。
        self._prior: dict[str, int] = {}
        #: 我们**写进去**的通道索引,按位置。位置 0 是电流。
        #: ``()`` = 还没配过,或者一路都没解析出来(那时通道校验没有意见)。
        self._channels: tuple[int, ...] = ()
        #: 动手前示波器上的通道,按位置。Osci2T 的 setter 必须同时给两路,
        #: 所以「第二路给什么」这个问题躲不掉 —— 答案是**给回用户原来那一路**。
        self._prior_channels: tuple[int, ...] = ()
        #: 这台机器有没有 Oscilloscope High Resolution。``None`` = 还没探过
        #: （直接 new 出来的泵不会探；``make_pump()`` 会）。见 make_pump 的 docstring：
        #: 探到了也仍然走 Osci1T —— OsciHRPump 还不存在，这个字段是**事实上报**，
        #: 不是策略开关。
        self.hr_available: bool | None = None
        #: ``make_pump()`` 要的策略与实际给出的不一致时的说明。``""`` = 没有降级。
        #: 一次静默的降级和一次故障一样难查,所以它跟着 status 一起出去。
        self.fallback_note: str = ""
        #: Continuity tally, for telling "we are keeping up" apart from "the
        #: scope is not refilling" without adding a single TCP call.
        self.stats = {"fresh": 0, "duplicate": 0, "gap": 0, "discontinuity": 0,
                      "busy": 0, "bad_reply": 0}

    # ── configuration ───────────────────────────────────────────────────────

    @property
    def config(self) -> dict:
        return dict(self._cfg)

    def _pool(self):
        """The live pool, or a pause. Fetched per call so a reconnect is picked
        up without re-wiring the pump."""
        pool = self._pool_getter()
        if pool is None:
            raise PumpPaused("no connection pool")
        return pool

    def _guard(self, rec):
        """Classify a call record's error and raise the matching control-flow
        exception. Takes the RECORD rather than issuing the call, so every
        ``safe_call`` in this module keeps its verb as a string literal —
        the repo's abort-policy checks, security audit and API coverage census
        all find Nanonis calls by grepping for ``safe_call("…")``, and a verb
        hidden in a variable is invisible to all three.
        """
        err = getattr(rec, "error", "") or ""
        if err:
            # is_lock_busy takes the RECORD, not the message.
            from mast.core.connection import is_lock_busy
            if is_lock_busy(rec):
                self.last_busy = True
                raise PumpPaused(f"{getattr(rec, 'method', '?')}: {err}")
            if "comms_circuit_open" in err:
                # Breaker is open: the call never reached the socket. Backing
                # off is the only correct response — retrying is what the
                # breaker exists to prevent.
                raise PumpPaused("comms circuit open")
            if "NeedModule" in err:
                raise PumpUnavailable(err)
        self.last_busy = False
        return rec

    # ── 方言接缝（子类实现，动词一律字面量） ─────────────────────────────────
    #
    # 每个都返回 ``NanonisCallRecord``，分类交给 :meth:`_guard`。基类从不知道
    # 动词长什么样。

    def _call_run(self):
        raise NotImplementedError

    def _call_ch_get(self):
        raise NotImplementedError

    def _call_ch_set(self, channels: tuple[int, ...]):
        raise NotImplementedError

    def _call_timebase_get(self):
        raise NotImplementedError

    def _call_timebase_set(self, index: int):
        raise NotImplementedError

    def _call_trig_immediate(self):
        raise NotImplementedError

    def _call_data_get(self):
        raise NotImplementedError

    def _decode_chunk(self, d: list) -> Optional[tuple[float, float, np.ndarray]]:
        """把解码后的字段列表变成 ``(t0, dt, y)``；形状不对就返回 None。"""
        raise NotImplementedError

    def _pick_timebase_index(self, values: np.ndarray) -> int:
        """从时基表里挑一档。两种方言的「更好」不是同一个方向 —— 见各自实现。"""
        raise NotImplementedError

    def _channels_to_write(self, current_index: int) -> tuple[int, ...]:
        """要写进示波器的通道组合。位置 0 必须是 ``current_index``。"""
        raise NotImplementedError

    # ── configuration ───────────────────────────────────────────────────────

    def configure(self) -> dict:
        """Point the scope at the current channel and pick a timebase.

        Trigger is forced to Immediate. If it were left on Level and nothing
        crossed the level, the scope would stop re-arming and every poll would
        return the same stale buffer for as long as that lasted.
        (``Osci1T_TrigGet`` is deliberately never called — that binding in
        nanonis_spm 1.0.9 declares an empty response spec and misparses.)
        """
        rec = self._guard(self._call_run())
        if getattr(rec, "error", ""):
            raise PumpUnavailable(f"{self.STRATEGY} run: {rec.error}")

        channel_index, channel_name = self._resolve_channel()
        if channel_index >= 0:
            prior = self._read_channels()
            if prior:
                # **只记第一次**，和 _remember_prior 一样，理由也一样：
                # _maybe_reconfigure() 每 30 s 就可能再走一遍这里，而那时读回来的
                # 是**上一轮自己写进去的值**。覆盖 = 还原目标变成设的那个
                # = 等于没还原。tests::test_reconfigure_does_not_overwrite_the_
                # original_snapshot 钉着这条 —— 这个错误在重构时曾经再次出现过，
                # 那条测试当场就红了。
                if not self._prior_channels:
                    self._prior_channels = prior
                self._remember_prior("channel_index", int(prior[0]))
            want = self._channels_to_write(int(channel_index))
            self._guard(self._call_ch_set(want))
            self._channels = want

        window_s = self._select_timebase()
        try:
            self._guard(self._call_trig_immediate())
        except PumpUnavailable:
            raise
        except Exception:  # noqa: BLE001 — trigger config is best-effort
            logger.debug("%s trigger config failed (continuing)", self.STRATEGY,
                         exc_info=True)

        rt_freq = 0.0
        try:
            rt = self._guard(self._pool().safe_call(
                "Util_RTFreqGet", role=DATA_ROLE, count_health=False))
            d = _decoded(rt.return_value)
            if d:
                rt_freq = _scalar(d[0])
        except Exception:  # noqa: BLE001 — cross-check only, never fatal
            logger.debug("Util_RTFreqGet unavailable", exc_info=True)

        # 时基描述整屏时长，不是每点间隔。dt、fs 和样本数等首份回包再填；
        # 拿整屏时长当 dt 会把采样率报小 n 倍。未知时先报 0。
        self._cfg = {
            "channel_index": channel_index,
            "channel_name": channel_name,
            "window_s": window_s,
            "dt_s": 0.0,             # 见上：等回包
            "fs_hz": 0.0,
            "rt_freq_hz": rt_freq,
            "n_buffer": 0,           # protocol does not expose it; learnt on first reply
            "timebases_s": list(self._timebases),
            "timebase_index": self._timebase_index,
            "timebase_check": self._timebase_check,
        }
        self._cfg["channels"] = list(self._channels)
        self._last_reconfig = time.monotonic()
        self._bad_replies = 0
        logger.info("current monitor: %s on %s, window=%.3g s, RT=%.0f Hz "
                    "(dt/fs 等第一份回包)", self.STRATEGY,
                    channel_name or f"#{channel_index}", window_s, rt_freq)
        return dict(self._cfg)

    def _remember_prior(self, key: str, value: Optional[int]) -> None:
        """记下我们动手**之前**的一个设置，只记第一次。

        重复调用是无害的 no-op —— ``_maybe_reconfigure()`` 每次重配都会走同一条
        路径，若这里覆盖，还原目标就会变成我们自己上一轮写进去的值。
        """
        if value is None or key in self._prior:
            return
        self._prior[key] = int(value)

    def _decode_channels(self, rec) -> tuple[int, ...]:
        """一次 ChGet / ChsGet 的回包 → 按位置的索引元组；形状不对返回 ``()``。

        位数不足就整个作废,不补零也不截断:一个「读回来只有一路」的 Osci2T 回包
        说明这份回包没对齐,而**位置**正是这里唯一的信息 —— 拿一个位置不明的
        索引去跟位置 0 比,是把「没读到」当成了「读到了别的」。
        """
        if getattr(rec, "error", "") or not (d := _decoded(rec.return_value)):
            return ()
        if len(d) < self.N_CHANNELS:
            return ()
        out: list[int] = []
        for v in d[:self.N_CHANNELS]:
            idx = _int_or_none(v)
            if idx is None:
                return ()
            out.append(idx)
        return tuple(out)

    def _read_channels(self) -> tuple[int, ...]:
        """示波器当前指着哪几路信号，按位置；读不到返回 ``()``。

        纯尽力而为：这一次读取只为「停止时能还原」服务，**绝不能让 configure()
        失败** —— 采集本身比礼貌重要，所以这里连 PumpPaused 都吞掉。
        """
        try:
            return self._decode_channels(self._call_ch_get())
        except Exception:  # noqa: BLE001 — 见 docstring：这一步不许影响采集
            logger.debug("%s channel read failed", self.STRATEGY, exc_info=True)
            return ()

    def restore(self) -> dict:
        """把示波器还给用户：通道与时基写回我们动手之前的值。

        Osci1T / Osci2T 都是单实例共享模块。不还原的话，一次 MAST 启动就会永久
        改掉用户示波器上看的信号，而他没有收到任何提示。

        **还不回去的两样**（如实说明，不要让调用方以为示波器完全复原了）：
        ``*_Run`` —— 协议没有对应的 run-state 读取，我们不知道启动前它停没停；
        触发模式 —— ``Osci1T_TrigGet`` 在 nanonis_spm 1.0.9 里响应规格是空的、
        必然误解析（见 ``configure()`` 的 docstring），读不到原值就无从还原。

        全程尽力而为：停机路径上抛异常只会把优雅停机变成不优雅停机。
        """
        done: dict = {"channel_index": None, "timebase_index": None}
        if self._prior_channels:
            try:
                rec = self._call_ch_set(self._prior_channels)
                if not getattr(rec, "error", ""):
                    done["channel_index"] = int(self._prior_channels[0])
            except Exception:  # noqa: BLE001 — 停机路径，绝不抛
                logger.debug("%s channel restore failed", self.STRATEGY,
                             exc_info=True)
        want_tb = self._prior.get("timebase_index")
        if want_tb is not None:
            try:
                rec = self._call_timebase_set(int(want_tb))
                if not getattr(rec, "error", ""):
                    done["timebase_index"] = int(want_tb)
            except Exception:  # noqa: BLE001 — 停机路径，绝不抛
                logger.debug("%s timebase restore failed", self.STRATEGY,
                             exc_info=True)
        if any(v is not None for v in done.values()):
            logger.info("current monitor: %s restored (channel=%s, timebase=%s) "
                        "— run-state and trigger mode are not restorable",
                        self.STRATEGY, done["channel_index"],
                        done["timebase_index"])
        return done

    def _resolve_channel(self) -> tuple[int, str]:
        """Find the current channel among the 128 signal slots by name.

        ``Signals_NamesGet`` is the only way — ``Signals_InSlotsGet`` appears in
        twenty docstrings of nanonis_spm 1.0.9 but does not exist in the library
        or in the V5e protocol.
        """
        try:
            rec = self._guard(self._pool().safe_call(
                "Signals_NamesGet", role=DATA_ROLE, count_health=False))
            d = _decoded(rec.return_value)
            names: list[str] = []
            for field_val in d:
                if isinstance(field_val, (list, tuple)) and field_val and isinstance(
                        field_val[0], (str, bytes)):
                    names = [x.decode() if isinstance(x, bytes) else str(x)
                             for x in field_val]
                    break
            for hint in _CURRENT_HINTS:
                for i, nm in enumerate(names):
                    if hint in nm.lower():
                        return i, nm
            if names:
                logger.warning("current monitor: no current-like signal among %d "
                               "names; leaving the scope channel as configured",
                               len(names))
        except (PumpPaused, PumpUnavailable):
            raise
        except Exception:  # noqa: BLE001
            logger.debug("Signals_NamesGet failed", exc_info=True)
        return -1, ""

    def _select_timebase(self) -> float:
        """Choose a timebase index; return the resulting dt in seconds.

        The available timebases depend on the RT frequency and oversampling, so
        the list must be read at runtime. (``Osci1T_TimebaseGet`` only works
        because ``core.nanonis_patch`` fixes the upstream binding, which sent
        the *setter* command and could therefore never return the list.)
        """
        try:
            rec = self._guard(self._call_timebase_get())
            d = _decoded(rec.return_value)
            values = _array(d[2]) if len(d) >= 3 else np.empty(0)
            if values.size == 0:
                self._timebase_check = "unverified: 时基表没读到,无从对账"
                return 0.0
            # 回包序是 (当前时基索引, 时基个数, 时基表)。趁这一次读取把「我们动手前
            # 用的是哪一档」留下 —— 不需要额外的 TCP 往返。
            cur_idx = _int_or_none(d[0]) if len(d) >= 1 else None
            declared = _int_or_none(d[1]) if len(d) >= 2 else None
            self._timebase_check = check_timebase_table(cur_idx, declared, values)
            if self._timebase_check.startswith("mismatch"):
                logger.warning(
                    "current monitor: 时基表自校验不通过 —— %s。照常使用这张表"
                    "(见 check_timebase_table 的 docstring:否决它需要先在真机上"
                    "确认档数字段),但采样率与时基相关的结论都要先怀疑这一条。",
                    self._timebase_check)
            elif self._timebase_check.startswith("unverified"):
                logger.info("current monitor: 时基表只对上了一部分 —— %s",
                            self._timebase_check)
            # 索引越界时**不记**:这个值停机时会被写回仪器(restore()),
            # 而写回一个我们刚判定为不可信的索引,比不还原更糟。
            if cur_idx is not None and 0 <= cur_idx < values.size:
                self._remember_prior("timebase_index", cur_idx)
            idx = self._pick_timebase_index(values)
            self._guard(self._call_timebase_set(idx))
            # Keep the table. It depends on the RT frequency and oversampling,
            # so it can only be known on the instrument — and it is the first
            # thing to check when the achieved rate is not what was expected.
            # Recording it here costs nothing: the call already happened.
            self._timebases = [float(v) for v in values]
            self._timebase_index = idx
            return float(values[idx])
        except (PumpPaused, PumpUnavailable):
            raise
        except Exception:  # noqa: BLE001 — dt is learnt from the reply anyway
            logger.debug("timebase selection failed (using scope default)",
                         exc_info=True)
            return 0.0

    def _maybe_reconfigure(self, why: str) -> None:
        now = time.monotonic()
        if now - self._last_reconfig < _RECONFIG_MIN_INTERVAL_S:
            return
        logger.info("current monitor: re-configuring %s (%s)", self.STRATEGY, why)
        try:
            self.configure()
        except (PumpPaused, PumpUnavailable):
            raise
        except Exception:  # noqa: BLE001
            logger.debug("re-configure failed", exc_info=True)
            self._last_reconfig = now

    # ── polling ─────────────────────────────────────────────────────────────

    def channel_is_ours(self) -> Optional[bool]:
        """Is the scope still pointed at the signal we configured?

        The scope is a SINGLE module, shared with ``AcquireOsciTrace``, the
        optional-scope skills and (on Osci2T) ``envhistory.zburst``. Those run on
        a different TCP role and take no instrument token (READ-category skills
        never do), so nothing serialises them against this pump — and if one
        re-points the scope, every subsequent ``DataGet`` returns THEIR signal
        while we keep storing it under "Current (A)". With Bias (~1 V) on the
        channel every sample clears the 90 nA saturation threshold, which is
        three segments from a CRITICAL that halts a running composite skill and
        interrupts the planner.

        A ``dt`` check cannot catch this: switching channel leaves the timebase
        alone. Returns None when no channel was ever resolved, or when the check
        itself could not be completed — callers treat None as "no opinion".

        **只判位置 0,而且是按位置判,不是「在不在里面」。** 两条都要紧:

        * Osci2T 的 ``ChsGet`` 回来的是**两个**索引。写成「我们那一路在不在这两个
          里面」就会漏掉最坏的那个情形 —— 别人把电流挪到了 B、A 换成了 Bias:
          那时「在里面」为真,而 ``DataGet`` 的 dataA 已经是别人的信号了。
          这正是任务里那句「第二通道的加入不能让它误判」。
        * 位置 1 **刻意不判**。它不是电流,没有任何阈值说的是它;而它会被
          ``zburst``(每 30 min)与 ``ConfigureDualScope`` 合法地改走。判它 =
          每半小时一次「通道被人改了,丢弃 + 重配」,而丢掉的是完全好的电流数据。
        """
        want = self._channels
        if not want:
            return None
        try:
            rec = self._guard(self._call_ch_get())
        except (PumpPaused, PumpUnavailable):
            raise
        except Exception:  # noqa: BLE001
            return None
        cur = self._decode_channels(rec)
        if not cur:
            return None
        return int(cur[0]) == int(want[0])

    def poll_once(self) -> Optional[TraceChunk]:
        """One ``DataGet(0)``. Returns None on an unusable reply."""
        rec = self._guard(self._call_data_get())
        if getattr(rec, "error", ""):
            self._bad_replies += 1
            return None
        d = _decoded(rec.return_value)
        decoded = self._decode_chunk(d)
        if decoded is None:
            self._bad_replies += 1
            return None
        t0, dt, y = decoded
        if dt <= 0 or y.size == 0:
            # The scope was just started and the buffer has not filled yet.
            self._bad_replies += 1
            return None
        self._bad_replies = 0
        if not self._cfg.get("n_buffer"):
            self._cfg["n_buffer"] = int(y.size)
        # dt / fs 的**唯一权威来源**就是这里：回包自己说的。configure() 里那张时基表
        # 给的是整屏时长，换算不出每点间隔（要除以 n，而 n 只有回包才知道）。
        if dt > 0 and self._cfg.get("dt_s") != dt:
            self._cfg["dt_s"] = dt
            self._cfg["fs_hz"] = 1.0 / dt
        return TraceChunk(t0=t0, dt=dt, n=int(y.size), y=y, host_ts=time.time())

    def pump_segments(self, stop: threading.Event,
                      on_idle: Optional[Callable[[float], Any]] = None,
                      ) -> Iterator[Segment]:
        """Yield segments until stop is set. PumpUnavailable reports a missing module; PumpPaused reports a dropped link. Invoke on_idle only while waiting for buffer refresh, with an explicit budget covering lock wait and the auxiliary read; do not add reads during role-busy backoff."""
        if not self._cfg:
            self.configure()

        runs: list[np.ndarray] = []
        cur: list[np.ndarray] = []
        n_total = 0
        gap_s = 0.0
        discontinuity = False
        seg_t0: float | None = None
        seg_host: float = 0.0
        last_t0: float | None = None
        last_fresh_mono = time.monotonic()
        dup_streak = 0
        last_dt: float = float(self._cfg.get("dt_s") or 0.0)
        last_n = 0
        busy_backoff = _BUSY_BACKOFF_START_S
        #: 上一份缓冲的内容指纹与主机时刻。
        #: 相对触发时间戳可能保持不变，因此必须同时检查内容变化。
        last_key: bytes | None = None
        last_host: float = 0.0
        warned_static_t0 = False

        while not stop.is_set():
            try:
                chunk = self.poll_once()
                busy_backoff = _BUSY_BACKOFF_START_S
            except PumpPaused:
                # Scan frame grabs share this role; queueing behind one is
                # normal. Count the wait as a gap rather than pretending the
                # samples were contiguous.
                self.stats["busy"] += 1
                if seg_t0 is not None:
                    gap_s += busy_backoff
                if _sleep_interruptible(stop, busy_backoff):
                    break
                busy_backoff = min(_BUSY_BACKOFF_MAX_S, busy_backoff * 2)
                continue

            if chunk is None:
                if self._bad_replies >= _BAD_REPLY_LIMIT:
                    self._bad_replies = 0
                    self._maybe_reconfigure("repeated malformed replies")
                if _sleep_interruptible(stop, 0.1):
                    break
                continue

            trace_s = chunk.n * chunk.dt

            # Is this still OUR signal? Another skill can re-point the shared
            # scope at any moment, and nothing serialises us against it. Drop
            # the frame rather than storing somebody else's channel as current.
            #
            # **点数也要看,不只是 dt。** Osci1T 的时基改的是采样率,所以 dt 变了就
            # 抓得到;**Osci2T 的时基改的是点数**(256/512/1280/2560/5120/12800,
            # 全部按同一个 Signals period 采),dt 一动不动。只看 dt 的话,
            # `envhistory.zburst`(每 30 min 跑一次,而且它选完时基**从不还原**)
            # 和 `ConfigureDualScope` 技能会在我们脚下把一屏从 6.4 s 换成 0.128 s,
            # 而这里一无所知 —— 拼接照旧、缺口计量用着旧的 last_n,全错。
            if (last_dt > 0 and abs(chunk.dt - last_dt) > 1e-12) or (
                    last_n > 0 and chunk.n != last_n):
                # Timebase changed under us (operator, or another skill using
                # the scope). The old and new samples are not one signal.
                discontinuity = True
                self.stats["discontinuity"] += 1
                if cur:
                    runs.append(np.concatenate(cur) if len(cur) > 1 else cur[0])
                    cur = []
                last_t0, last_key = None, None
                self._maybe_reconfigure("timebase changed")

            key = chunk.y.tobytes()
            t0_moved = last_t0 is None or chunk.t0 != last_t0
            content_moved = last_key is None or key != last_key

            if last_t0 is not None:
                # 判新用「**t0 变了 或 内容变了**」，两条都不动才算重复：
                #   * t0 会走的仪器（合成语料、旧假设）：t0 变 → 新，与原逻辑一致，
                #     即使波形恰好逐位相同（常数信号的测试就是这样）也不会误判成重复。
                #   * t0 不走的仪器：内容变 → 新。
                #   * 真正的重复：时间戳和内容都没变。
                if not t0_moved and not content_moved:
                    self.stats["duplicate"] += 1
                    # We arrived before the scope refilled. Wait out the rest of
                    # the current window rather than probing again immediately —
                    # blind retries are what turn a 20/s job into a 45/s one.
                    # Floor the wait. If the scope stops re-arming at all — an
                    # operator switching it to Level trigger that never fires,
                    # the module being stopped, the front panel closed — then t0
                    # freezes, `remain` goes permanently negative, and a zero
                    # floor turns this into a back-to-back DataGet loop at TCP
                    # speed, holding the data role lock ~100% of the time
                    # against a port this project treats as fragile by rule.
                    remain = trace_s - (time.monotonic() - last_fresh_mono)
                    dup_streak += 1
                    if dup_streak >= _DUP_STREAK_LIMIT:
                        dup_streak = 0
                        self._maybe_reconfigure("buffer stopped advancing")
                    if _sleep_with_idle(stop, _clamp(
                            remain + _SLEEP_GUARD_S, _SLEEP_MIN_S,
                            _sleep_cap(trace_s)), on_idle):
                        break
                    continue
                dup_streak = 0
                if t0_moved:
                    delta = chunk.t0 - (last_t0 + last_n * last_dt)
                else:
                    # t0 冻在原地而内容在变 → 它是相对触发的常量，拿它做时间轴算术
                    # 只会让每一份缓冲都判成「时钟倒退」。改用主机时钟推断这两份
                    # 缓冲之间丢了多少时间。**精度差**（受轮询抖动影响），但方向
                    # 是对的，而 t0 那条路在这台仪器上连方向都没有。
                    if not warned_static_t0:
                        warned_static_t0 = True
                        logger.warning(
                            "current monitor: Osci1T 的 t0 不推进（恒为 %.6g），"
                            "改用内容判新 + 主机时钟计时。段落时间戳的精度取决于"
                            "轮询抖动，不再是仪器时基。", chunk.t0)
                    delta = (chunk.host_ts - last_host) - last_n * last_dt
                # 容差必须跟着时间来源走。仪器 t0 的分辨率是一个采样间隔，所以半个
                # dt 是对的；**主机时钟不是** —— 轮询抖动是毫秒级，而 dt 只有
                # 0.5 ms，拿 0.5*dt 去卡会把几乎每一份缓冲都判成有间隙，run 被切成
                # 碎片、gap_s 被垃圾累加。主机时钟模式下能分辨的最小单位就是一次
                # 轮询，所以门槛取半个缓冲长度：真丢了半屏才算间隙。
                tol = (0.5 * chunk.dt) if t0_moved else (0.5 * trace_s)
                if delta <= -tol:
                    # Clock went backwards: the module was restarted. Keep the
                    # samples we have, start a fresh run.
                    discontinuity = True
                    if cur:
                        runs.append(np.concatenate(cur) if len(cur) > 1 else cur[0])
                        cur = []
                elif delta >= tol:
                    self.stats["gap"] += 1
                    gap_s += float(delta)
                    if cur:
                        runs.append(np.concatenate(cur) if len(cur) > 1 else cur[0])
                        cur = []

            self.stats["fresh"] += 1
            last_t0, last_dt, last_n = chunk.t0, chunk.dt, chunk.n
            last_key, last_host = key, chunk.host_ts
            last_fresh_mono = time.monotonic()

            # ── 把这一屏喂进段落，需要几段就切几段 ─────────────────────────────
            #
            # **为什么切,而不是让段长跟着一屏走。** Osci1T 一屏 0.128 s,永远塞不满
            # 一个 1 s 的段落,所以在 2026-08-09 之前这里是「整屏追加,然后看够没够」。
            # Osci2T 一屏可以是 **6.4 s** —— 比段落长六倍。整屏追加会让段长变成一屏,
            # 而 ``cm_segment_s`` 是**全部电流阈值的计价单位**:``cm_crit_consecutive``
            # 数的是「连续多少**段**」,段一变长,那三条能停掉正在跑的复合技能的
            # CRITICAL 规则的门槛就跟着变了 —— 而没有任何一个阈值被重新标定过。
            # 切开则一个阈值都不用动:特征层只看「一串样本 + fs」,不看它是从哪一屏
            # 来的(``features.compute_segment_features(runs, fs_hz)``)。
            #
            # 顺带一个**行为变化**,如实记下来:段落现在**正好**是 cm_segment_s,
            # 而不是「至少 cm_segment_s、最多再多一屏」。Osci1T 上是 1.000 s 而不是
            # 从前的 1.000–1.128 s;多出来的那点样本不再算进上一段,而是接着下一段
            # 的头 —— 一个样本都没丢。
            ours: Optional[bool] = None      # 本屏的通道校验，只做一次（见下）
            poisoned = False
            offset = 0
            while offset < chunk.n:
                if seg_t0 is None:
                    seg_t0 = chunk.t0 + offset * chunk.dt
                    seg_host = chunk.host_ts - trace_s + offset * chunk.dt
                room_s = self._segment_s - (n_total * chunk.dt + gap_s)
                if room_s <= 0:
                    # 段落已经被缺口撑满了（长时间 RoleBusy）。整屏收下再收尾，
                    # 与切开之前的行为一致 —— 切出一个只有一两个样本的段落，
                    # 除了让 mostly_gap 多判一次之外没有任何用处。
                    take = chunk.n - offset
                else:
                    take = min(chunk.n - offset,
                               max(1, int(np.ceil(room_s / chunk.dt))))
                cur.append(chunk.y[offset:offset + take])
                n_total += take
                offset += take

                span_s = n_total * chunk.dt + gap_s
                if span_s < self._segment_s:
                    break

                # Verify the channel at every segment boundary — the segment is
                # the unit that reaches storage, so this guarantees nothing is
                # ever stored without having been checked. A periodic timer
                # cannot promise that: with a check interval longer than a
                # segment, a whole poisoned segment lands before the next check.
                #
                # 一屏只查一次:这一屏是**在同一套通道配置下一次取回来的**,对它问
                # 六遍会得到六个相同的答案,却要花六次 TCP 往返。Osci1T 上一段横跨
                # 约八屏,所以这仍然是「每段一次」,与从前逐字相同;Osci2T 上一屏
                # 横跨六段,于是变成「每屏一次」—— 覆盖没有变松(整屏一起丢弃),
                # 往返却从 6 次降到 1 次。
                if ours is None:
                    ours = self.channel_is_ours()
                if ours is False:
                    # Discard, do not salvage. The check only says the channel is
                    # wrong NOW; the swap could have happened anywhere in this
                    # segment, so every sample in it is suspect. Losing a second
                    # of current is cheap — storing somebody else's signal as
                    # tunnelling current poisons the corpus this subsystem exists
                    # to build, and Bias on the channel reads as 100% saturation,
                    # which is a scan-halting alert away.
                    logger.warning(
                        "current monitor: the oscilloscope channel was changed by "
                        "another caller — discarding %d samples and re-configuring",
                        n_total)
                    poisoned = True
                    break
                if cur:
                    runs.append(np.concatenate(cur) if len(cur) > 1 else cur[0])
                    cur = []
                yield Segment(
                    t_start=seg_host,
                    t_end=seg_host + span_s,
                    osci_t0=float(seg_t0),
                    fs_hz=(1.0 / chunk.dt) if chunk.dt > 0 else 0.0,
                    runs=runs,
                    n_samples=n_total,
                    gap_s=gap_s,
                    discontinuity=discontinuity,
                    channel_name=str(self._cfg.get("channel_name") or ""),
                    source=self.STRATEGY,
                )
                runs, n_total, gap_s, discontinuity, seg_t0 = [], 0, 0.0, False, None

            if poisoned:
                runs, cur = [], []
                n_total, gap_s, discontinuity, seg_t0 = 0, 0.0, False, None
                last_t0, last_key = None, None
                self._maybe_reconfigure("channel was re-pointed by someone else")
                if _sleep_interruptible(stop, _clamp(
                        trace_s, _SLEEP_MIN_S, _sleep_cap(trace_s))):
                    break
                continue

            # Schedule against an ABSOLUTE target rather than sleeping a fixed
            # slice: relative sleeps accumulate every round trip and every
            # feature-extraction pause into a phase drift, and once the drift
            # exceeds one refill a whole buffer is lost. Aiming at a wall-clock
            # instant makes a slow iteration shorten the next sleep instead.
            target = last_fresh_mono + _SLEEP_FRESH_FRAC * trace_s
            if _sleep_with_idle(stop, _clamp(
                    target - time.monotonic(), 0.0, _sleep_cap(trace_s)),
                    on_idle):
                break


class Osci1TPump(_ScopePump):
    """单通道示波器方言。一屏固定 **256 点**，时基改的是**采样率**。

    根据 Nanonis 手册对 Oscilloscope 模块的说明，图上显示的数据点数固定为 256，
    采样率等于 256 除以时基。所以时基表里最小的那个值 = 最快的采样率，
    :meth:`_pick_timebase_index` 取 ``argmin``。
    """

    STRATEGY = "osci1t"
    MODULE_NAME = "Osci1T"
    N_CHANNELS = 1

    def _call_run(self):
        return self._pool().safe_call("Osci1T_Run", role=DATA_ROLE,
                                      count_health=False)

    def _call_ch_get(self):
        return self._pool().safe_call("Osci1T_ChGet", role=DATA_ROLE,
                                      count_health=False)

    def _call_ch_set(self, channels: tuple[int, ...]):
        return self._pool().safe_call("Osci1T_ChSet", int(channels[0]),
                                      role=DATA_ROLE, count_health=False)

    def _call_timebase_get(self):
        return self._pool().safe_call("Osci1T_TimebaseGet", role=DATA_ROLE,
                                      count_health=False)

    def _call_timebase_set(self, index: int):
        return self._pool().safe_call("Osci1T_TimebaseSet", int(index),
                                      role=DATA_ROLE, count_health=False)

    def _call_trig_immediate(self):
        return self._pool().safe_call("Osci1T_TrigSet", 0, 1, 0.0, 0.0,
                                      role=DATA_ROLE, count_health=False)

    def _call_data_get(self):
        return self._pool().safe_call("Osci1T_DataGet", 0, role=DATA_ROLE,
                                      count_health=False)

    def _decode_chunk(self, d: list) -> Optional[tuple[float, float, np.ndarray]]:
        """回包是 ``[t0, dt, size, data]``。"""
        if len(d) < 4:
            return None
        try:
            t0 = _scalar(d[0])
            dt = _scalar(d[1])
            n = int(_scalar(d[2]))
            y = _array(d[3])
        except (TypeError, ValueError, IndexError):
            return None
        if n <= 0:
            return None
        if y.size > n:
            y = y[:n]
        return t0, dt, y

    def _pick_timebase_index(self, values: np.ndarray) -> int:
        if self._target_fs_hz > 0:
            # ⚠️ 已知局限（切换到基类时逐字保留）：目标采样率是拿 1/fs 去跟**时基表**
            # 比的，而表里是整屏时长，真要命中一个采样率得再除以 n —— 而 n 只有第一份
            # 回包才知道。默认值 0（取最快一档）不受此目标采样率推算限制影响。
            return int(np.argmin(np.abs(values - 1.0 / self._target_fs_hz)))
        return int(np.argmin(values))          # fastest available

    def _channels_to_write(self, current_index: int) -> tuple[int, ...]:
        return (int(current_index),)


class Osci2TPump(_ScopePump):
    """双通道示波器方言。时基改的是**点数**，采样率一动不动。

    【手册】``Reference/Graphs/Oscilloscope 2T.html``：「**Unlike the normal
    Oscilloscope where the number of points is fixed to 256**, here the Time base
    sets the number of points to **256, 512, 1280, 2560, 5120, and 12800**. They
    are acquired at the **Signals period** rate」。

    结果是两条与 1T 相反的性质，两条都在代码里承重：

    * **换时基不会改 dt。** 所以「时基被别人改了」这件事,只看 dt 是**看不见**的
      —— 见 :meth:`pump_segments` 里那条点数比较。本机上真的有人会改它:
      ``envhistory.zburst`` 每 30 min 跑一次 Z 噪声谱,它选完时基**从不还原**。
    * **「最好的一档」方向相反。** 1T 取 ``argmin``(最快);2T 每一档速率相同,
      长的那档只是一次往返拿回更多数据,所以取「不超过 ``window_target_s`` 的
      最长一档」。

    ⚠️ **这个方言换来的不是速度。** 采样率仍然是 Signals Period 定的(本机
    2 kHz),与 1T 一模一样 —— 所以全部电流阈值不用重标。换来的是:一次 TCP 往返
    顶从前 50 次(12800 vs 256 点),``data`` 角色的占用直线下降,而扫描抓帧和
    辅助通道都在抢这个角色。
    """

    STRATEGY = "osci2t"
    MODULE_NAME = "Osci2T"
    N_CHANNELS = 2

    #: 一屏想要多长(秒)。0 = 取表里最长的一档。
    DEFAULT_WINDOW_S = 6.4

    def _call_run(self):
        return self._pool().safe_call("Osci2T_Run", role=DATA_ROLE,
                                      count_health=False)

    def _call_ch_get(self):
        return self._pool().safe_call("Osci2T_ChsGet", role=DATA_ROLE,
                                      count_health=False)

    def _call_ch_set(self, channels: tuple[int, ...]):
        # setter 必须**同时**给两路 —— 「第二路给什么」这个问题躲不掉。
        b = int(channels[1]) if len(channels) > 1 else int(channels[0])
        return self._pool().safe_call("Osci2T_ChsSet", int(channels[0]), b,
                                      role=DATA_ROLE, count_health=False)

    def _call_timebase_get(self):
        return self._pool().safe_call("Osci2T_TimebaseGet", role=DATA_ROLE,
                                      count_health=False)

    def _call_timebase_set(self, index: int):
        return self._pool().safe_call("Osci2T_TimebaseSet", int(index),
                                      role=DATA_ROLE, count_health=False)

    def _call_trig_immediate(self):
        # (trigger_on, mode=Immediate, …) —— 与 zburst.py 的那次调用同签名。
        return self._pool().safe_call("Osci2T_TrigSet", 0, 0, 1, 0.0, 0.0, 0.0,
                                      role=DATA_ROLE, count_health=False)

    def _call_data_get(self):
        return self._pool().safe_call("Osci2T_DataGet", 0, role=DATA_ROLE,
                                      count_health=False)

    def _decode_chunk(self, d: list) -> Optional[tuple[float, float, np.ndarray]]:
        """回包是 ``[t0, dt, sizeA, dataA, sizeB, dataB]`` —— **一个 dt 管两路**。

        【库】``Osci2T_DataGet`` 的返回规格 ``["d","d","i","*d","i","*d"]``。
        两路等长、同 dt,这正是「双通道不会把每通道速率减半」那条结论的协议侧证据
        (若减半,协议里会出现第二个 dt)。

        **只取 A 路。** B 路是用户原来在看的那一路(见 :meth:`_channels_to_write`),
        本次刻意不采信也不存 —— 存一路 2 kHz 的第二通道要一个新列、一份新的保留
        预算和一个消费者,三样都不存在,而一个「取回来就扔」的字段正是这个仓库
        反复种下的那种死字段。要推翻这个决定,需要先回答:哪个问题**必须**由
        2 kHz 的第二通道来答,而 5 Hz 的 aux 采样答不了?
        """
        if len(d) < 4:
            return None
        try:
            t0 = _scalar(d[0])
            dt = _scalar(d[1])
            n = int(_scalar(d[2]))
            y = _array(d[3])
        except (TypeError, ValueError, IndexError):
            return None
        if n <= 0:
            return None
        if y.size > n:
            y = y[:n]
        return t0, dt, y

    def _pick_timebase_index(self, values: np.ndarray) -> int:
        """不超过 ``window_target_s`` 的**最长**一档；全都超了就取最短的。

        与 1T 的 ``argmin`` 方向相反,而且这个方向不是口味问题:2T 的每一档速率
        相同,短的那档除了「一次往返拿回来的数据更少」之外没有任何好处。
        """
        want = self._window_target_s or self.DEFAULT_WINDOW_S
        fits = np.flatnonzero(values <= want + 1e-12)
        if fits.size:
            return int(fits[int(np.argmax(values[fits]))])
        return int(np.argmin(values))

    def _channels_to_write(self, current_index: int) -> tuple[int, ...]:
        """A = 电流。**B = 用户原来那一路**，我们不往上写自己的东西。

        setter 要求同时给两路,所以「B 给什么」躲不掉,只有两个候选:
        我们自己想看的(比如 Z),或者用户本来就在看的。选后者 ——

        * 这台示波器是**借来的**:2T 是单实例共享模块,而这条泵是 7×24 常驻的,
          等于把用户的双通道示波器长期占走。占走 A 路是采集的代价,占走 B 路
          不是,那纯粹是我们多拿的。
        * 而且写自己的东西会**打起来**:``zburst`` 每 30 min 把 B 设成 Z,
          ``ConfigureDualScope`` 技能随时可能改它。我们若每次重配都把 B 抢回来,
          就是每半小时一场配置拉锯。所以既不抢 B,:meth:`channel_is_ours` 也
          **不判** B(见那里的说明)。

        读不到原值时退回 A —— 双路显示同一个信号是无害的,而胡乱猜一个索引写进
        别人的示波器不是。
        """
        if len(self._prior_channels) > 1:
            return (int(current_index), int(self._prior_channels[1]))
        # 读取用户 B 通道失败时回退到电流通道，必须记录原因，避免覆盖界面选择却没有可追溯信息。
        logger.warning(
            "%s: 读不到示波器原来的通道配置，B 路将被写成电流（#%d）—— "
            "**用户在 B 路上设的信号会被覆盖**。这不是有意抢占：`_read_channels` "
            "返回空时，「保留用户那一路」这条就没有可保留的值了。"
            "要保住 B 路，先查 Osci2T_ChGet 为什么读不到。",
            self.STRATEGY, int(current_index))
        return (int(current_index), int(current_index))


class OsciHRProbe:
    """Detection hook for the High-Resolution scope.
    
    Probe the target instrument and expose ``hr_available``. There is no HR pump
    implementation here; availability alone must not imply that acquisition uses HR.
    """

    @staticmethod
    def available(pool: Any) -> bool:
        if pool is None:
            return False
        try:
            rec = pool.safe_call("OsciHR_SamplesGet", role=DATA_ROLE,
                                 count_health=False)
        except Exception:  # noqa: BLE001
            return False
        return not (getattr(rec, "error", "") or "")


class Osci2TProbe:
    """探测 Osci2T 是否可用。
    
    授权与模块是否已加载是两回事；前面板未打开时命令可能被拒绝。
    用不写参数的 ``TimebaseGet`` 探测，失败时回退 1T 并说明原因。
    """

    @staticmethod
    def available(pool: Any) -> tuple[bool, str]:
        """返回示波器能力是否可用及原因；探测异常按不可用处理。
        
        必须区分仪器拒绝命令、模块未启用与解码规格不符。
        传输层 salvage 已区分错误状态和有效但布局不符的回包，此处如实解释，
        不能让代码解码问题变成要求用户操作仪器的建议。"""
        if pool is None:
            return False, "尚未连接 Nanonis"
        try:
            rec = pool.safe_call("Osci2T_TimebaseGet", role=DATA_ROLE,
                                 count_health=False)
        except Exception as exc:  # noqa: BLE001
            return False, f"探测失败:{exc}"
        err = getattr(rec, "error", "") or ""
        if not err:
            return True, ""
        if "NeedModule" in err:
            return False, "Osci2T 模块未加载(许可证有,但前面板没开)"
        from mast.core.nanonis_patch import LAYOUT_MISMATCH_PREFIX
        if LAYOUT_MISMATCH_PREFIX in err:
            # 仪器**正常回答了**,是我们读不懂 —— 这是代码缺陷,不是现场状态,
            # 所以措辞里不能出现「模块没开」那种让人去动仪器的暗示。
            return False, f"Osci2T 回包解不开(解码规格与实际布局不符):{err}"
        if "unpack requires" in err or "struct.error" in err:
            # 兜底:万一还有哪条路绕过了 salvage,至少不要把它说成模块没加载。
            return False, f"Osci2T 回包解析失败(原因未分诊):{err}"
        return False, err


def make_pump(pool_getter: Callable[[], Any], *, segment_s: float = 1.0,
              target_fs_hz: float = 0.0, strategy: str = "osci1t",
              window_target_s: float = 0.0) -> _ScopePump:
    """选择采集策略，并探测目标仪器的 HR 可用性。
    
    ``strategy`` 是期望值：请求 2T 时先检查模块，不可用则回退 1T 并记录原因。
    状态中的策略取自最终泵实例。``hr_available`` 只是能力探测结果；此处尚无
    HR 采集实现，不会因探测成功而宣称正在使用 HR。
    """
    cls: type[_ScopePump] = Osci1TPump
    note = ""
    if str(strategy or "").lower() == "osci2t":
        try:
            ok, why = Osci2TProbe.available(pool_getter())
        except Exception as exc:  # noqa: BLE001 — 探测失败绝不能挡住采集
            logger.debug("Osci2T probe failed", exc_info=True)
            ok, why = False, f"探测异常:{exc}"
        if ok:
            cls = Osci2TPump
        else:
            note = f"(要的是 osci2t,退回 osci1t:{why})"
            logger.warning("current monitor: 请求的 Osci2T 用不上 —— %s。"
                           "退回 Osci1T,采集照常,只是每次往返拿回来的数据少 50 倍。",
                           why)
    pump = cls(pool_getter, segment_s=segment_s, target_fs_hz=target_fs_hz,
               window_target_s=window_target_s)
    try:
        hr = OsciHRProbe.available(pool_getter())
    except Exception:  # noqa: BLE001 — 探测失败只影响诊断，绝不能挡住采集
        logger.debug("OsciHR probe failed (treating as unavailable)", exc_info=True)
        hr = False
    pump.hr_available = bool(hr)
    pump.fallback_note = note
    logger.info("current monitor: 采集策略 = %s%s（OsciHR %s）",
                pump.STRATEGY, note,
                "可用 —— 值得为它实现专用泵" if hr else "不可用")
    return pump


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sleep_interruptible(stop: threading.Event, seconds: float) -> bool:
    """Sleep in slices so stop() is honoured promptly. True if we should stop."""
    return stop.wait(max(0.0, seconds))


def _sleep_cap(trace_s: float) -> float:
    """一次睡眠的上限。

    :data:`_SLEEP_MAX_S` 那个 0.5 s 是照着 Osci1T 定的:一屏 0.128 s,0.5 s 已经是
    四屏,足够当「预测错了也不至于睡过头」的兜底。**Osci2T 的一屏可以是 6.4 s**,
    固定 0.5 s 上限会把「一屏轮一次」变成「一屏轮十三次」—— 十二次重复帧,而
    「一次往返顶从前五十次」正是这次切换的全部收益。所以上限跟着一屏走。

    这个 clamp 不是为了让 stop 及时 —— 睡眠本身是 ``stop.wait()``,置位即返回。
    """
    return max(_SLEEP_MAX_S, 1.25 * max(0.0, float(trace_s)))


def _sleep_with_idle(stop: threading.Event, seconds: float,
                     on_idle: Optional[Callable[[float], Any]]) -> bool:
    """Sleep ``seconds``, handing slices of it to ``on_idle``. True if we
    should stop.

    实际花掉的时间从这一觉里**扣掉**，不是加在后面 —— 加在后面就是把泵的相位
    往后推，一次推过一个刷新窗口就丢一屏采样。

    **切片,不是「睡前让一次」。** 从前一觉最多 0.5 s(Osci1T 实际约 0.118 s),
    所以「睡前给搭车者一次机会」等于每屏一次、约 8-20 Hz,辅助通道的 5 Hz 就是
    这么来的。Osci2T 一觉可以是 5.9 s —— 只让一次就把辅助采样打到 **0.17 Hz**,
    把 #30 刚修好的提速整个还回去。所以把这一觉切成 :data:`_IDLE_SLICE_S` 的片,
    每片让一次:采样机会与一屏多长**无关**,只与切片长度有关。

    给出去的 budget 是**这一片**的剩余量,不是整觉 —— 搭车者拿它压自己的锁等待,
    承诺必须是能兑现的那个数。

    ``on_idle`` 抛出的任何东西都在这里吞掉：搭车的读数绝不可以反噬电流采集。
    """
    if on_idle is None:
        return _sleep_interruptible(stop, seconds)
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        remain = deadline - time.monotonic()
        if remain <= 0:
            return stop.is_set()
        if remain < _IDLE_MIN_SLACK_S:
            # 剩的太少,不够让一次(见 _IDLE_MIN_SLACK_S):睡完收工。
            return _sleep_interruptible(stop, remain)
        budget = min(remain, _IDLE_SLICE_S)
        t0 = time.monotonic()
        try:
            on_idle(budget)
        except Exception:  # noqa: BLE001 — a hitch-hiker never breaks the pump
            logger.debug("pump on_idle callback failed (swallowed)", exc_info=True)
        rest = budget - (time.monotonic() - t0)
        if rest > 0 and _sleep_interruptible(stop, rest):
            return True
        if stop.is_set():
            return True


__all__ = ["Osci1TPump", "Osci2TPump", "OsciHRProbe", "Osci2TProbe",
           "Segment", "TraceChunk", "check_timebase_table",
           "PumpUnavailable", "PumpPaused", "make_pump", "DATA_ROLE"]
