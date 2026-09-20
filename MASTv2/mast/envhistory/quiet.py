"""仪器此刻算不算"安静" —— 纯函数，零 TCP。

为什么需要这个判定
------------------

要记录的是**环境**：针尖长时间恒流隧穿停留下的电流、电流噪声谱、Z 噪声谱。
这三样只有在仪器没人动的时候才是环境量 —— 扫描进行中的电流是**形貌信号**，
把它混进"隧道电流趋势"里会毁掉整条曲线；修针窗口里的噪声谱是那次修针的记录,
不是这台机器的本底。

所以这些序列在入桶/入谱之前先过一道安静门。温度、真空、液氦、磁场**不过**这道
门 —— 它们无论仪器在干什么都是同一个物理量。

输入从哪来
----------

调用方给的 ctx dict 与 :meth:`mast.monitoring.service.CurrentMonitorService._context_labels`
产出的形状完全一致（同样的键名），来源是两个纯内存快照：``InstrumentState``
的 1 Hz 缓存和 ``instrument_lock().snapshot()``。两者都不发 TCP —— 一个只读
观察者绝不能把自己放到命令路径上。

已知盲区（不修，只声明）
------------------------

用户在 Nanonis 前面板上手动改 bias / setpoint 既不取仪器令牌也不置 scan
标志，会被判成 ``quiet``。谱那一侧用的是 median-of-K 聚合，这类瞬态被天然稀释;
标量那一侧混进来的是几个点。代价可接受，靠加 TCP 轮询去堵它不值。
"""
from __future__ import annotations

#: 隧穿保持中，没人在动仪器 —— 谱与恒流统计唯一有意义的窗口。
QUIET = "quiet"
#: 有技能持令牌，或扫描在跑 —— 电流/Z 是形貌信号，不是环境噪声。
ACTIVE = "active"
#: Z 反馈断开 / 已退针 —— 根本没有隧道结。
NO_TUNNEL = "no_tunnel"
#: 快照缺失或已陈旧 —— 诚实地不判，而不是猜一个。
UNKNOWN = "unknown"

#: 判定结果里"这条读数能进安静序列吗"的唯一真源。
_ADMITS = frozenset({QUIET})


def classify(ctx: dict | None) -> str:
    """把上下文快照判成四态之一。

    顺序是刻意的：先查快照可信度（stale 的快照说什么都不算数），再查有没有
    隧道结（没有结的时候"安静"是无意义的——针根本没在测量），最后才查有没有
    人在动仪器。
    """
    if not ctx:
        return UNKNOWN
    if ctx.get("ctx_stale"):
        return UNKNOWN
    zctrl = ctx.get("ctx_zctrl_on")
    if zctrl is None:
        # 连 Z 反馈状态都读不到，说明 InstrumentState 还没跑起来或全失败。
        return UNKNOWN
    if not zctrl:
        return NO_TUNNEL
    # 任何持有仪器令牌的技能都算"不安静"。这比电流监控的 SUPPRESS_SKILL_PATTERNS
    # 子集更严：对告警抑制而言只有修针/进针值得特殊对待，对"这是环境吗"而言
    # 任何驱动仪器的动作都让读数不再是环境。更严也更简单。
    if str(ctx.get("ctx_skill") or "").strip():
        return ACTIVE
    if ctx.get("ctx_scanning"):
        return ACTIVE
    return QUIET


def admits(quietness: str) -> bool:
    """这个判定结果允许读数进入"仅安静"序列吗。

    单独一个函数而不是 ``== QUIET``：将来若要放宽（例如允许 ``unknown``）,
    改一处即可，不必去追散落各处的比较。
    """
    return quietness in _ADMITS


def context_from_snapshots(state_getter=None, lock_snapshot=None) -> dict:
    """从两个零 TCP 快照拼出 ctx —— 给没有段流的调用方（sink 侧）用。

    与 ``CurrentMonitorService._context_labels`` 的键名逐字一致，这样同一个
    :func:`classify` 服务两条链路。任何一步失败都不抛：读不到就留 None,
    :func:`classify` 会把它判成 ``unknown``。
    """
    out: dict = {"ctx_skill": ""}
    try:
        state = state_getter() if callable(state_getter) else state_getter
        snap = state.snapshot() if state is not None else None
        if snap is not None:
            out.update({
                "ctx_scanning": getattr(snap, "scan_running", None),
                "ctx_bias_v": getattr(snap, "bias_v", None),
                "ctx_setpoint_a": getattr(snap, "setpoint_a", None),
                "ctx_z_m": getattr(snap, "z_pos_m", None),
                "ctx_zctrl_on": getattr(snap, "z_controller_on", None),
                "ctx_stale": getattr(snap, "stale", None),
            })
    except Exception:  # noqa: BLE001 — 上下文是可选的，记录不是
        pass
    try:
        if lock_snapshot is None:
            from mast.core.instrument_lock import instrument_lock
            holder = instrument_lock().snapshot() or {}
        else:
            holder = (lock_snapshot() if callable(lock_snapshot) else lock_snapshot) or {}
        out["ctx_skill"] = str(holder.get("skill") or "")
    except Exception:  # noqa: BLE001
        pass
    return out


__all__ = ["QUIET", "ACTIVE", "NO_TUNNEL", "UNKNOWN",
           "classify", "admits", "context_from_snapshots"]
