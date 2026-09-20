"""异步硬件动作 + 同连接双通道读回 —— 采集循环的单一实现。

TipShaper 和 Bias_Pulse 都有一个 "Wait_until_done" 参数:传 0 时调用立即返回,
动作在 Nanonis 控制器上自己跑完,于是 Python 侧可以在**同一条** TCP 连接上边跑
边轮询电流与 Z —— 拿到的是过程中的曲线,不是事后的前后对比。

两个技能的循环本来就是同一个,且其中四个细节都容易写错:

  * **绝对时刻调度**(``next_poll += period``,不是 sleep(period))。相对睡眠会把
    每次 RTT 和处理时间累积成相位漂移。
  * **pre-roll 基线**要在开火之前采够,否则判据没有 z1 可比。
  * **开火那一次调用阻塞了多久**要计时:某些固件忽略 wait=0,真的把整个过程跑完
    才返回。那时采到的主要是事后状态,必须如实标注而不是假装拿到了过程曲线。
  * **abort 早退**每帧都要查。

写两份的下场是其中一份的修复到不了另一份,所以这里只写一份。

本模块还负责**把采到的原始曲线落盘**(``save_trace``)—— 见那个函数的 docstring。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

# ⚠️ ``time`` 这个模块属性在测试里会被换成假时钟(见 tests 的 ``readback_clock``
# 夹具:``monkeypatch.setattr(_readback_stream, "time", FakeClock())``),而那个
# 替身**只有** ``perf_counter`` / ``sleep`` / ``advance``。所以采集循环之外的任何
# 地方都不许再用 ``time.`` —— 落盘要时间戳就用上面这个 ``datetime``,要唯一名字
# 就用 ``uuid``。写成 ``time.time()`` 会在夹具下抛 AttributeError,而且是在
# 「曲线落不下来」这条最不该静默的路上。


def scalar(rv) -> float | None:
    """Nanonis ``(err_str, raw_bytes, [values])`` 回包里的第一个标量。"""
    d = rv
    if isinstance(rv, tuple) and len(rv) >= 3 and isinstance(rv[2], list):
        d = rv[2]
    if not isinstance(d, list) or not d:
        return None
    v = d[0]
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    return float(v) if isinstance(v, (int, float)) else None


def stats(xs: list[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {"n": n, "min": min(xs), "max": max(xs), "mean": mean, "std": var ** 0.5}


@dataclass
class ReadbackCapture:
    """一次「开火 + 读回」采集的产物,时间轴以采集开始为 0。"""

    current_s: list[float] = field(default_factory=list)
    current_t: list[float] = field(default_factory=list)
    z_s: list[float] = field(default_factory=list)
    z_t: list[float] = field(default_factory=list)
    #: 动作是否真的发出去了(pre-roll 期间被 abort 掉就没有)。
    fired: bool = False
    #: 开火那次 ``safe_call`` 的 record,调用方要查它的 error。
    fire_record: Any = None
    #: 开火时刻(采集时钟,秒)—— 判据用它切前后窗口。
    fire_t_s: float | None = None
    #: 开火那次调用自身阻塞了多久。见模块文档的诚实守卫。
    fire_blocked_s: float = 0.0
    capture_s: float = 0.0
    aborted: bool = False

    @property
    def empty(self) -> bool:
        return not self.current_s and not self.z_s


def stream_with_action(
    context,
    *,
    fire: Callable[[], Any],
    total_capture_s: float,
    pre_roll_s: float,
    poll_hz: float,
    calls: list,
) -> ReadbackCapture:
    """采 ``pre_roll_s`` 秒基线 → 调 ``fire()`` → 继续采到 ``total_capture_s``。

    ``fire`` 应当发起一次**异步**硬件动作并返回它的 NanonisCallRecord;本函数负责
    计时、把 record 收进 ``calls``、以及把它放进结果里让调用方检查错误。每帧采
    一次电流再采一次 Z(相隔一个 RTT,视作同时)。
    """
    cap = ReadbackCapture()
    period = 1.0 / max(float(poll_hz), 1e-6)
    t0 = time.perf_counter()
    next_poll = t0

    while True:
        now = time.perf_counter()
        elapsed = now - t0
        if elapsed >= total_capture_s:
            break
        if hasattr(context, "check_abort") and context.check_abort():
            cap.aborted = True
            break

        if not cap.fired and elapsed >= pre_roll_s:
            tc = time.perf_counter()
            rec = fire()
            cap.fire_blocked_s = time.perf_counter() - tc
            calls.append(rec)
            cap.fired = True
            cap.fire_record = rec
            cap.fire_t_s = tc - t0
            if getattr(rec, "error", ""):
                break

        if now < next_poll:
            time.sleep(min(next_poll - now, 0.001))
            continue

        rec_i = context.safe_call("Current_Get")
        calls.append(rec_i)
        if not rec_i.error:
            vi = scalar(rec_i.return_value)
            if vi is not None:
                cap.current_s.append(vi)
                cap.current_t.append(time.perf_counter() - t0)
        rec_z = context.safe_call("ZCtrl_ZPosGet")
        calls.append(rec_z)
        if not rec_z.error:
            vz = scalar(rec_z.return_value)
            if vz is not None:
                cap.z_s.append(vz)
                cap.z_t.append(time.perf_counter() - t0)
        next_poll += period

    cap.capture_s = max(cap.current_t[-1] if cap.current_t else 0.0,
                        cap.z_t[-1] if cap.z_t else 0.0)
    return cap


def channel_block(samples: list[float], times: list[float], unit: str) -> dict:
    """一个通道的 ``{samples_<u>, t_s, n, min_<u>, …}`` 结果块。"""
    return {
        f"samples_{unit}": samples, "t_s": times,
        **{f"{k}_{unit}" if k in ("min", "max", "mean", "std") else k: v
           for k, v in stats(samples).items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 落盘 —— 让这条曲线**取得到**
# ═══════════════════════════════════════════════════════════════════════════
#
# ## 它以前去哪了
#
# ``channel_block`` 存的是**完整样本**(``samples_m`` / ``t_s`` 两条全量列表),
# 不是摘要。但它到不了任何人手上:
#
#   * 工具边界只把 ``SkillResult.summary`` 送过去(``skill_adapter``),而这两个
#     技能都认真写了一行人话摘要 ⇒ agent 看到的就只有那一行;
#   * ``data`` 进记录库,而记录层会把长列表**截到 200 个点 + 一个
#     ``"<+N more>"`` 标记**(``test_v1_action_long_traces_are_bounded_...``,
#     那是对的:一行数据库记录不该塞五万个浮点);
#   * 而记录库本身没有「按实验/技能列动作」的 HTTP 端点,单条详情端点
#     (``/api/records/actions/{id}``)返回的 ``ActionDetail`` **根本不含
#     result data** —— 就算知道 id 也取不到曲线。
#
# ⇒ 一条**专门为了看清扎针过程而采集**的曲线,采完之后没有任何人能看到。
#
# ## 为什么是 JSON,不是 ``.npy`` / ``.npz``
#
# 本仓既有的做法是 ``GrabScanFrameData``:落 ``.npy``、回包只带路径。这里**复用
# 那条路**(落盘 + 回指针),但换了容器,理由是读的人不同:
#
#   * 帧是二维图像、消费者是 numpy 代码;这条曲线的第一消费者是**用户本人**,
#     而真机上没有独立的 python(``C:\\MAST\\_internal\\python.exe`` 不存在,
#     PyInstaller onedir),``.npz`` 在那台机器上打不开;
#   * 两个通道的点数**可以不一样**(某一帧的 ``Current_Get`` 出错、``ZCtrl_ZPosGet``
#     成功,循环就只记后者),还要带阶段边界与整形参数 —— 这是一个带结构的
#     记录,不是一个矩形数组;
#   * 一次采集 ~2 kHz × ~1 s ≈ 数千个浮点,JSON 也就几百 KB。
#
# 一个自描述的文本文件,记事本能打开、``json.load`` 能吃、跨机拷贝不掉信息。

#: 产物结构的版本号。**读的人先看这个字段再决定怎么解**;改了结构就必须改它。
TRACE_SCHEMA = "mast.readback_trace/1"


#: 落盘目录的显式覆写。与 ``MAST_DRAFTS_DIR`` / ``MAST_FIGURES_DIR`` 同一族
#: (``agents/_shared/data_paths``):产物目录本来就该能挪到数据盘上。
#:
#: 它同时是**测试隔离的抓手**。这条曲线是从 17 个测试文件都够得着的技能里写出来
#: 的(扎针的四个 composite、修针工作流、operating_mode…),指望每个文件各自记得
#: 重定向 = 「每页各自记得」,人肉找不齐;所以套件在 ``tests/v2/conftest.py`` 里
#: 用一条 autouse 夹具统一设它,另有一条闸门在事后核对真实目录没被写过。
TRACE_DIR_ENV = "MAST_TRACES_DIR"


def trace_dir():
    """原始读回曲线的落盘目录:``$MAST_TRACES_DIR`` > ``<project_root>/experiments/traces/``。

    **每次调用重新解析**(而不是模块级常量):路径固化在 import 时会让重定向失效
    —— 那正是本仓「测试污染真实数据」踩过五次的形状,其中两次的具体样子就是
    「测试重定向了 env A,而代码读的是 env B / 一段私有的 ``parents[N]`` 走法」。
    """
    import os
    from pathlib import Path

    from mast._runtime_paths import project_root

    raw = os.environ.get(TRACE_DIR_ENV, "").strip()
    d = Path(raw).expanduser() if raw else project_root() / "experiments" / "traces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_trace(capture: "ReadbackCapture", *, skill: str, meta: dict,
               stages: list | None = None, out_dir=None) -> dict:
    """把一次采集的**两条原始曲线 + 元数据**写成一个 JSON 产物。

    返回 ``{"trace_path": "<abs path>"}``,失败时返回 ``{"trace_error": "..."}``
    —— 两者都直接 splat 进 ``SkillResult.data``,于是「曲线在哪」和「曲线为什么
    没落成」都在回包里说得出口。**不返回空 dict**:落盘失败若无声无息,下一个人
    看到的又会是一条采完就消失的曲线,而这正是本函数存在的理由。

    落盘失败**不会**让技能失败:硬件动作已经做完了,一次写盘错误不该把一次成功的
    扎针变成一次报错(与 ``post_hook`` 同一条纪律)。

    产物结构(``schema = mast.readback_trace/1``)::

        {"schema", "skill", "created_utc",
         "event_t_s",           # 开火时刻(采集时钟,秒)= 阶段边界的锚点
         "capture_s", "fired", "aborted",
         "stages": [...],       # 有就带(TipShaper 的 switch_off/plunge/…)
         "meta": {...},         # 扎针参数、坐标、判定 —— 调用方给什么带什么
         "channels": {"z":       {"unit": "m", "n", "t_s": [...], "samples": [...]},
                      "current": {"unit": "A", "n", "t_s": [...], "samples": [...]}}}

    **两个通道各自带自己的时间戳**,因为它们各自独立:采集循环每帧先读电流再读
    Z(相隔一个 RTT),而任一次读失败就只有另一条记了点。把它们当同一根时间轴
    对齐是错的,所以这里不对齐,如实各存各的。
    """
    try:
        import json
        from pathlib import Path

        d = Path(out_dir) if out_dir else trace_dir()
        d.mkdir(parents=True, exist_ok=True)
        slug = "".join(ch if ch.isalnum() else "_"
                       for ch in (skill or "readback"))[:40]
        # 时间戳负责**按发生顺序排、肉眼对得上实验记录**;唯一性靠 uuid 与「取第一个
        # 没被占用的名字」,不靠时钟。单靠毫秒戳会撞 —— ``scan_frame`` 那边用两次
        # 真机事故买过这个教训(一次读到上一轮的残留、一次直接覆盖掉前一帧)。
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = d / f"{slug}_{stamp}_{uuid.uuid4().hex[:8]}"
        out = base.with_suffix(".json")
        n = 1
        while out.exists():
            out = base.with_name(f"{base.name}_{n:02d}").with_suffix(".json")
            n += 1

        payload = {
            "schema": TRACE_SCHEMA,
            "skill": skill,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "event_t_s": capture.fire_t_s,
            "capture_s": capture.capture_s,
            "fired": capture.fired,
            "aborted": capture.aborted,
            "stages": list(stages or []),
            "meta": dict(meta or {}),
            "channels": {
                "z": {"unit": "m", "n": len(capture.z_s),
                      "t_s": list(capture.z_t), "samples": list(capture.z_s)},
                "current": {"unit": "A", "n": len(capture.current_s),
                            "t_s": list(capture.current_t),
                            "samples": list(capture.current_s)},
            },
        }
        out.write_text(json.dumps(payload, ensure_ascii=False),
                       encoding="utf-8")
        return {"trace_path": str(out)}
    except Exception as exc:  # noqa: BLE001 — 落盘失败绝不带走一次成功的硬件动作
        return {"trace_error": f"{type(exc).__name__}: {exc}"}


def trace_ref(saved: dict) -> str:
    """回包摘要末尾那一句**指针**(或那一句「没落成」)。

    两个技能共用同一句措辞,而且**失败也出声** —— 只在成功时才附一句的写法,
    会让「曲线没了」和「本来就没采」在摘要里长得一模一样。
    """
    p = (saved or {}).get("trace_path")
    if p:
        return f" | 原始曲线: {p}"
    err = (saved or {}).get("trace_error")
    return f" | ⚠️ 原始曲线未落盘: {err}" if err else ""
