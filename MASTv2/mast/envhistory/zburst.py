"""Z 噪音谱 —— Osci2T 双通道的一次性 burst 采集。

为什么是 burst 而不是第二个常驻泵
---------------------------------

Z 的诉求只有"这台机器的 Z 本底噪声谱长什么样"，那是小时~天尺度的量。为它常年
占住 ``data`` 角色，去和电流监控每秒抢二十次锁，换来的是一条半小时才画一个点的
曲线 —— 不划算，而且电流监控会把抢锁的等待诚实地记成缺口，等于用针尖监控的清晰
度去换 Z 的采样密度。

所以：每 ``eh_z_interval_s`` 醒一次，只在仪器安静时采 ``eh_z_burst_s`` 秒，算完
一条谱就退出线程。默认 30 min / 30 s，即 data 角色的额外占用约 1.7% 时间。

不存 Z 的段、不算 Z 的特征、不进 monitoring 的段表
--------------------------------------------------

只产出一条 ``env_spectra`` 行（``channel='z'``）。给 Z 建段表意味着又一套保留
策略和又一份 288 MB/h；算 Z 的 44 列特征意味着把针尖健康的语义硬套到位移信号
上，而且**没有任何消费者**。

尤其：**Z 只记录，不判级。** 电流那边的四条噪声判据全部是先对纯高斯白噪声验过
误报率才敢上的（其中四条被实测推翻重写）。把它们照搬到 Z 上是在赌，赌输的形式
是半夜一条假 CRITICAL 停掉正在跑的实验。

对共享模块的礼貌
----------------

Osci2T 是单实例共享模块，``skills/builtins/optional_scopes.py`` 里的技能随时可能
重配它，而 READ 类技能不取仪器令牌 —— 没有任何东西把我们和它们串行化。所以：

* 采集**前后**都校验通道，对不上就整轮丢弃，绝不"抢救"半条数据；
* 只在 Z 不在画面上时才改通道，并在结束时**改回去**；
* 熔断开路、RoleBusy、模块未加载一律是"跳过这一轮"，不是错误。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

#: 与电流监控同一个角色。6502(monitor) 已经满了,6501(main) 上的阻塞读会抽干
#: API 线程池 —— 6503 是唯一有余量的,而且它本来就是"采数据"的那一条。
DATA_ROLE = "data"

#: 信号名里认 Z 的线索,按优先级。Nanonis 的信号名形如 "Z (m)"。
_Z_HINTS: tuple[str, ...] = ("z (m)", "z(m)", "z position", "z_pos", "z ")
#: 认电流的线索 —— A 通道保持电流,这样一次 burst 顺带拿到同时刻的 I。
_I_HINTS: tuple[str, ...] = ("current (a)", "current(a)", "current")

#: B 路可以采什么。键是给调用方用的短名，值是在 128 路信号名里认它的线索。
#:
#: 为什么按**名字**认而不是写死索引：信号槽位是每台机器自己配的，写死 #24 在
#: 另一台机上会安静地采到别的东西 —— 而"安静地采到别的东西"是这类代码最坏的
#: 失败方式（数据看起来完全正常）。名字对不上就 `_Skip`，不猜。
_CHANNEL_HINTS: dict[str, tuple[str, ...]] = {
    "z": _Z_HINTS,
    "current": _I_HINTS,
    # 偏压。演示「改 bias 拿原子分辨」时要看阶跃后多久稳定，aux 的 4 Hz 看不见。
    "bias": ("bias (v)", "bias(v)", "bias"),
    "amplitude": ("amplitude (m)", "amplitude(m)", "amplitude"),
    # 识别供应商使用的频移和锁相通道名称及其常见别名。
    # 不要按通道序号或猜测名称识别；X/Y 为投影分量，不能含混合并为幅值。
    "df": ("freq. shift", "freq shift", "frequency shift"),

    # dI/dV 的 X 与 Y 使用不同键；哪路能作信号依据实际相位与配置判断。
    # 不提供含混的 didv 别名，避免调用方把某个正交分量误当成解调幅值 R。
    "didv_x": ("li demod 1 x",),
    "didv_y": ("li demod 1 y",),
}

#: 连续这么多次坏回包/忙就放弃本轮。与 pump 的量级一致。
_MAX_BAD = 10
_MAX_BUSY = 20

#: 相位锁定的两个常数,与 monitoring.pump 同源:拿到新帧后睡到"这一次填充窗口
#: 的 92%"这个绝对时刻,拿到重复帧就睡当前窗口的剩余量。相对 sleep 会把每次
#: 往返的耗时累积成相位漂移,漂过一个 refill 就整丢一个缓冲。
_SLEEP_FRESH_FRAC = 0.92
_SLEEP_MIN_S = 0.002
_SLEEP_GUARD_S = 0.005
_SLEEP_MAX_S = 0.5

#: 单次 burst 的墙钟硬上限系数。慢调用不该把一个 30 s 的 burst 拖成十分钟。
_WALL_CAP_FACTOR = 2.0
_WALL_CAP_EXTRA_S = 5.0


class _Skip(Exception):
    """本轮跳过 —— 不是错误，是正常事件（忙 / 熔断 / 模块没开）。"""


def run_z_burst(pool_getter: Callable[[], Any], *, stop=None,
                burst_s: float = 30.0, bins: int = 240,
                ctx: dict | None = None):
    """采一次 Z 噪声谱。成功返回 :class:`~mast.envhistory.spectra.SpectrumSnapshot`,
    任何一步不顺就返回 None（调用方把原因记进 status）。**永不抛。**"""
    from mast.envhistory.spectra import SpectrumSnapshot, log_bin

    restore: tuple[int, int] | None = None
    pool = None
    try:
        pool = pool_getter() if callable(pool_getter) else pool_getter
        if pool is None:
            raise _Skip("未连接 Nanonis")

        _call(pool, "Osci2T_Run")
        z_idx, z_name, i_idx = _resolve_channels(pool)
        if z_idx < 0:
            raise _Skip("128 路信号里没有找到 Z")

        want_a, want_b, restore = _ensure_channels(pool, z_idx, i_idx)
        _set_immediate_trigger(pool)
        dt = _select_timebase(pool)

        runs, _runs_i, fs_hz, n_fresh = _pump(pool, stop, burst_s, dt)
        if not runs:
            raise _Skip("burst 期间没有取到新数据")

        # 采完再校验一次：中途被别的技能改走通道的话，前面那些样本是别人的信号。
        if not _channels_still_ours(pool, want_a, want_b):
            raise _Skip("采集期间 Osci2T 通道被改动，本轮丢弃")

        from mast.monitoring.features import psd_of_runs
        freqs, psd = psd_of_runs(runs, fs_hz)
        f, p = log_bin(freqs, psd, int(bins))
        if len(f) < 2:
            raise _Skip("谱点太少")
        now = time.time()
        logger.info("Z 噪声谱：%d 段 @ %.0f Hz，%d 个频点（%s）",
                    n_fresh, fs_hz, len(f), z_name or f"#{z_idx}")
        return SpectrumSnapshot(
            ts=now, channel="z", span_s=float(burst_s), n_segments=n_fresh,
            fs_hz=fs_hz, freqs=f, psd=p, unit="m^2/Hz",
            quietness="quiet", ctx=dict(ctx or {}),
        )
    except _Skip as exc:
        logger.debug("z burst skipped: %s", exc)
        return None
    except Exception:  # noqa: BLE001 — 记录器绝不把异常抛回调度线程
        logger.debug("z burst failed", exc_info=True)
        return None
    finally:
        if restore is not None and pool is not None:
            _restore_channels(pool, restore)


def run_dual_burst(pool_getter: Callable[[], Any], *, stop=None,
                   burst_s: float = 30.0, set_timebase: bool = False,
                   channel: str = "z"):
    """采集与电流同步的 B 路 burst，channel 选择 B 路，默认 Z。
    
    返回原始样本，不计算谱也不入库。bias 通道使用示波器当前采样率，
    适用于观察偏压改变后的稳定过程；不假定某台仪器的采样率。
    
    set_timebase=False 沿用共享示波器时基。改变时基可能触发监控泵重配并覆盖
    通道，导致所有权验证失败；采样率和 dt 必须从实际回包取得。
    采前采后均核对通道，任一步不一致就丢弃整批；结束时恢复原通道。
    
    返回 dict 或 None，异常不向外抛。结果包括 z、i 分段数组，fs_hz、n_fresh、
    z_name、z_index、i_index。对应两路来自同一回包、同一时基；长度不一致时
    整对丢弃，不能返回部分对齐的数据。"""
    restore: tuple[int, int] | None = None
    pool = None
    try:
        pool = pool_getter() if callable(pool_getter) else pool_getter
        if pool is None:
            raise _Skip("未连接 Nanonis")

        _call(pool, "Osci2T_Run")
        key = str(channel or "z").strip().lower()
        hints = _CHANNEL_HINTS.get(key)
        if hints is None:
            raise _Skip("不认识的通道 %r（可选：%s）"
                        % (channel, ", ".join(sorted(_CHANNEL_HINTS))))
        z_idx, z_name, i_idx = _resolve_channels(pool, hints)
        if z_idx < 0:
            # 名字对不上就停手。**不退回到别的通道** —— 采到别的东西而不声张，
            # 比什么都没采到糟得多：数据会一路走到分析、图、结论。
            raise _Skip("128 路信号里没有找到 %s（线索 %s）" % (key, hints))
        if i_idx < 0:
            raise _Skip("128 路信号里没有找到电流 —— 双通道 burst 需要两路都在")

        want_a, want_b, restore = _ensure_channels(pool, z_idx, i_idx)
        _set_immediate_trigger(pool)
        # 见 docstring：改时基会把电流监控泵踢进重配，重配会把通道改回去，
        # 于是这一轮必然被 _channels_still_ours 判掉。
        dt = _select_timebase(pool) if set_timebase else 0.0

        runs_z, runs_i, fs_hz, n_fresh = _pump(pool, stop, burst_s, dt)
        if not runs_z:
            raise _Skip("burst 期间没有取到新数据")
        if not _channels_still_ours(pool, want_a, want_b):
            raise _Skip("采集期间 Osci2T 通道被改动，本轮丢弃")

        pairs = [(z, i) for z, i in zip(runs_z, runs_i) if i is not None]
        if not pairs:
            raise _Skip("没有一帧的两路是等长的")
        logger.info("双通道 burst：%d/%d 帧同步 @ %.0f Hz（Z=%s #%d，I #%d）",
                    len(pairs), n_fresh, fs_hz, z_name or "?", z_idx, i_idx)
        return {"z": [z for z, _ in pairs], "i": [i for _, i in pairs],
                "fs_hz": float(fs_hz), "n_fresh": int(n_fresh),
                # ``z``/``z_name`` 这两个键名是历史包袱（这条路原来只采 Z）。
                # ``channel`` 才是权威的「B 路采的是什么」—— 读的人必须能一眼
                # 看出手里这批数据是 Z 还是 bias，键名叫 z 而内容是 bias 是
                # 事故的标准配方。
                "channel": key, "z_name": z_name,
                "z_index": int(z_idx), "i_index": int(i_idx)}
    except _Skip as exc:
        logger.debug("dual burst skipped: %s", exc)
        return None
    except Exception:  # noqa: BLE001 — 与 run_z_burst 同一条：绝不抛回调用线程
        logger.debug("dual burst failed", exc_info=True)
        return None
    finally:
        if restore is not None and pool is not None:
            _restore_channels(pool, restore)


# ── Nanonis 调用（动词全部字面量：仓库的中止策略检查 / 安全审计 / API 覆盖率
#    普查都靠 grep `safe_call("…")`，藏进变量的动词对三者都是隐形的） ────────

def _call(pool, verb: str, *args):
    """一次只读调用 + 统一的错误分类。

    ``count_health=False``：熔断器是四个 role 共用的一个实例，``record_success``
    无条件清空失败连击。一个高频后台轮询器在闲置 role 上不断成功，会让"连续三次
    失败"这个条件永远攒不满，等于把全局熔断器废掉。
    """
    if verb == "Osci2T_Run":
        rec = pool.safe_call("Osci2T_Run", role=DATA_ROLE, count_health=False)
    elif verb == "Signals_NamesGet":
        rec = pool.safe_call("Signals_NamesGet", role=DATA_ROLE, count_health=False)
    elif verb == "Osci2T_ChGet":
        rec = pool.safe_call("Osci2T_ChGet", role=DATA_ROLE, count_health=False)
    elif verb == "Osci2T_ChSet":
        rec = pool.safe_call("Osci2T_ChSet", args[0], args[1],
                             role=DATA_ROLE, count_health=False)
    elif verb == "Osci2T_TimebaseGet":
        rec = pool.safe_call("Osci2T_TimebaseGet", role=DATA_ROLE, count_health=False)
    elif verb == "Osci2T_TimebaseSet":
        rec = pool.safe_call("Osci2T_TimebaseSet", args[0],
                             role=DATA_ROLE, count_health=False)
    elif verb == "Osci2T_TrigSet":
        rec = pool.safe_call("Osci2T_TrigSet", 0, 0, 1, 0.0, 0.0, 0.0,
                             role=DATA_ROLE, count_health=False)
    elif verb == "Osci2T_DataGet":
        rec = pool.safe_call("Osci2T_DataGet", 0, role=DATA_ROLE, count_health=False)
    else:  # pragma: no cover — 防止有人加了动词却忘了分支
        raise _Skip(f"未知动词 {verb}")
    return _guard(rec)


def _guard(rec):
    err = getattr(rec, "error", "") or ""
    if err:
        from mast.core.connection import is_lock_busy
        if is_lock_busy(rec):
            raise _Busy(err)
        if "comms_circuit_open" in err:
            raise _Skip("TCP 已熔断")
        if "NeedModule" in err:
            raise _Skip("Osci2T 模块未加载（仿真器上属正常）")
    return rec


class _Busy(Exception):
    """角色锁被占 —— 正常事件，退避后重试。"""


def _decoded(rv) -> list:
    if isinstance(rv, (list, tuple)) and len(rv) > 2 and isinstance(rv[2], (list, tuple)):
        return list(rv[2])
    return []


def _scalar(v) -> float:
    if isinstance(v, (list, tuple)):
        return float(v[0]) if v else 0.0
    return float(v)


# ── 配置 ─────────────────────────────────────────────────────────────

def _resolve_channels(pool, hints: "Sequence[str] | None" = None) -> tuple[int, str, int]:
    """从 128 路信号名里找 **B 路目标** 与电流的索引。

    ``hints`` 不给就找 Z（历史默认）。A 路恒为电流 —— 那样一次 burst 顺带拿到
    同时刻的 I，而 I 是把任何别的通道解释成物理量时的参照。

    ``Signals_NamesGet`` 是唯一的办法 —— ``Signals_InSlotsGet`` 在 nanonis_spm
    1.0.9 的二十个 docstring 里出现过，但库里和 V5e 协议里都没有这个东西。
    """
    rec = _call(pool, "Signals_NamesGet")
    names: list[str] = []
    for field_val in _decoded(rec.return_value):
        if isinstance(field_val, (list, tuple)) and field_val and isinstance(
                field_val[0], (str, bytes)):
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in field_val]
            break
    b_idx, b_name = _match(names, hints or _Z_HINTS)
    i_idx, _ = _match(names, _I_HINTS)
    return b_idx, b_name, i_idx


def _match(names: Sequence[str], hints: Sequence[str]) -> tuple[int, str]:
    for hint in hints:
        for i, nm in enumerate(names):
            if hint in nm.lower():
                return i, nm
    return -1, ""


def _ensure_channels(pool, z_idx: int, i_idx: int):
    """让 Z 出现在画面上。返回 (期望A, 期望B, 需要恢复的旧配置或 None)。

    已经在画面上就**一个字节都不写** —— 这是与用户共享一台示波器时最重要的
    一条礼貌，也顺带省掉一次写入失败的可能。
    """
    cur = _read_channels(pool)
    if cur and z_idx in cur:
        return cur[0], (cur[1] if len(cur) > 1 else cur[0]), None
    want_a = i_idx if i_idx >= 0 else (cur[0] if cur else z_idx)
    want_b = z_idx
    _call(pool, "Osci2T_ChSet", int(want_a), int(want_b))
    restore = (cur[0], cur[1]) if len(cur) >= 2 else None
    return want_a, want_b, restore


def _read_channels(pool) -> list[int]:
    try:
        rec = _call(pool, "Osci2T_ChGet")
    except (_Busy, _Skip):
        raise
    except Exception:  # noqa: BLE001
        return []
    if getattr(rec, "error", ""):
        return []
    out: list[int] = []
    for v in _decoded(rec.return_value):
        try:
            out.append(int(_scalar(v)))
        except (TypeError, ValueError):
            continue
    return out


def _channels_still_ours(pool, want_a: int, want_b: int) -> bool:
    cur = _read_channels(pool)
    if not cur:
        return True     # 读不到就不否定 —— 与 pump 的 "None 表示没有意见" 同义
    return want_b in cur


def _restore_channels(pool, restore: tuple[int, int]) -> None:
    """best-effort 把通道改回用户原来的样子。失败只记 debug：这一轮的谱已经
    拿到了，为了恢复显示而抛异常没有意义。"""
    try:
        _call(pool, "Osci2T_ChSet", int(restore[0]), int(restore[1]))
    except Exception:  # noqa: BLE001
        logger.debug("Osci2T 通道恢复失败（显示层，不影响数据）", exc_info=True)


def _set_immediate_trigger(pool) -> None:
    """强制 Immediate 触发。

    留在 Level 上而又没有信号越过电平时，示波器会停止重装，之后每次 DataGet 都
    返回同一个陈旧缓冲 —— 表现是"采了 30 秒，全是同一段"。
    """
    try:
        _call(pool, "Osci2T_TrigSet")
    except (_Busy, _Skip):
        raise
    except Exception:  # noqa: BLE001 — 触发配置尽力而为
        logger.debug("Osci2T_TrigSet failed (continuing)", exc_info=True)


def _select_timebase(pool) -> float:
    """选最快的时基，返回 dt（秒）。读不到就返回 0，dt 会从回包里学到。"""
    try:
        rec = _call(pool, "Osci2T_TimebaseGet")
        d = _decoded(rec.return_value)
        values = [float(x) for x in d[2]] if len(d) >= 3 and isinstance(
            d[2], (list, tuple)) else []
        if not values:
            return 0.0
        idx = min(range(len(values)), key=lambda i: values[i])
        _call(pool, "Osci2T_TimebaseSet", int(idx))
        return float(values[idx])
    except (_Busy, _Skip):
        raise
    except Exception:  # noqa: BLE001
        logger.debug("Osci2T 时基选择失败（用示波器当前设置）", exc_info=True)
        return 0.0


# ── 轮询 ─────────────────────────────────────────────────────────────

def _pump(pool, stop, burst_s: float, dt_hint: float):
    """轮 ``Osci2T_DataGet(0)`` 直到 burst 结束。

    返回 ``(runs_b, runs_a, fs_hz, 新帧数)`` —— B 是 Z，A 是电流。

    **两路都留着**，虽然 :func:`run_z_burst` 只用 B。它们出自同一个回包、同一个
    时基，因此是**逐样本同步**的，而那正是算相干性所需要的东西：电流噪声里有多少
    是 Z 真的在动、有多少不是，单看任何一路都答不出来。丢掉 A 再想要就得重采一次，
    而重采回来的两段不同步，相干性就无从谈起。

    恒用 ``DataGet(0)``（取当前显示的数据）而不是等触发：等触发会在没有越限
    时一路阻塞到 5 s 的 recv 超时，而且它会把 data 角色锁按住整条 trace;
    模式 0 只占一个往返。去重靠回包自带的 t0，免费。
    """
    import numpy as np

    runs: list[Any] = []
    runs_a: list[Any] = []
    last_t0: float | None = None
    last_fresh = time.monotonic()
    bad = busy = 0
    n_fresh = 0
    fs_hz = (1.0 / dt_hint) if dt_hint > 0 else 0.0
    start = time.monotonic()
    deadline = start + max(1.0, float(burst_s))
    wall_cap = start + max(1.0, float(burst_s)) * _WALL_CAP_FACTOR + _WALL_CAP_EXTRA_S

    while time.monotonic() < deadline:
        if stop is not None and getattr(stop, "is_set", lambda: False)():
            break
        if time.monotonic() > wall_cap:
            logger.debug("z burst 触到墙钟上限，提前收工")
            break
        try:
            rec = _call(pool, "Osci2T_DataGet")
        except _Busy:
            busy += 1
            if busy > _MAX_BUSY:
                # from None：RoleBusy 不是错误，是"扫描正拿着 data 口"这个正常
                # 事件。把它挂成 _Skip 的 __cause__ 会让日志读起来像故障。
                raise _Skip("data 角色持续被占") from None
            _sleep(stop, min(_SLEEP_MAX_S, 0.05 * busy))
            continue
        if getattr(rec, "error", ""):
            bad += 1
            if bad > _MAX_BAD:
                raise _Skip("连续坏回包")
            _sleep(stop, 0.05)
            continue
        d = _decoded(rec.return_value)
        if len(d) < 6:
            bad += 1
            if bad > _MAX_BAD:
                raise _Skip("回包结构不符（缺第二通道）")
            _sleep(stop, 0.05)
            continue
        try:
            t0 = float(_scalar(d[0]))
            dt = float(_scalar(d[1]))
            # 返回谱是 [t0, dt, sizeA, dataA, sizeB, dataB] —— B 通道是 Z。
            #
            # ⚠️ `.reshape(-1)` 不是装饰。`Osci2T_DataGet` 的 `*d` 数组走
            # `decodeArray`,而那个函数**曾经**把每个元素裹成 1-元组
            # (`nanonis_patch` 现在在解码层拆掉了,但这里不该依赖那个补丁还在)。
            # 元素是元组时 `np.asarray` **不报错**,它给出 **(N, 1)** ——
            # 一次静默的形状改变,而外面那个 `except (TypeError, ValueError,
            # IndexError)` 根本接不住;`y.size` 仍读作 N,所以下面那个空检查也照过。
            # 这是本轮元组分诊里**唯一**一处没有元素级守卫的数组读取。
            y = np.asarray(d[5], dtype=np.float64).reshape(-1)
            # A 通道（电流）同理 —— 同一个 reshape 守卫，同一个理由。
            ya = np.asarray(d[3], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, IndexError):
            bad += 1
            _sleep(stop, 0.05)
            continue
        if y.size == 0 or dt <= 0:
            bad += 1
            _sleep(stop, 0.05)
            continue

        trace_s = dt * y.size
        if last_t0 is not None and t0 == last_t0:
            # 示波器还没重装：睡到当前填充窗口的剩余量。下限是硬要求 —— 缓冲
            # 停止刷新时 t0 会冻结，剩余量永久为负，零下限会变成 TCP 极限速率
            # 的忙循环。
            remain = trace_s - (time.monotonic() - last_fresh)
            _sleep(stop, _clamp(remain + _SLEEP_GUARD_S, _SLEEP_MIN_S, _SLEEP_MAX_S))
            continue

        bad = busy = 0
        last_t0 = t0
        last_fresh = time.monotonic()
        n_fresh += 1
        runs.append(y)
        # 长度对不上就不收 A：一条与 Z 不等长的电流不能与它逐样本配对，
        # 而配不上的两路算出来的相干性是个看起来很正常的假数。
        runs_a.append(ya if ya.size == y.size else None)
        fs_hz = 1.0 / dt
        # 瞄准一个绝对时刻,而不是"睡固定时长":慢的那一轮会自动缩短下一次睡眠,
        # 相位不漂。
        target = last_fresh + _SLEEP_FRESH_FRAC * trace_s
        _sleep(stop, _clamp(target - time.monotonic(), 0.0, _SLEEP_MAX_S))

    return runs, runs_a, fs_hz, n_fresh


def _sleep(stop, seconds: float) -> None:
    """可被 stop 打断的睡眠。关机时不必等完一个窗口。"""
    s = max(0.0, float(seconds))
    if s <= 0:
        return
    if stop is not None and hasattr(stop, "wait"):
        stop.wait(s)
    else:  # pragma: no cover — 只有测试会不传 stop
        time.sleep(s)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


__all__ = ["run_z_burst", "run_dual_burst", "DATA_ROLE"]
