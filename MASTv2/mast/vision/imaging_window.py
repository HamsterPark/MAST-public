"""检查原子分辨判读的成像条件窗口。

修针流程可能改变偏压或电流，后续采集前必须回读确认；不合适的工作点可能
降低晶格对比，不能仅凭此把针尖判坏并继续扰动。

窗口不满足时输出 undetermined，而非 absent。内置窗口只作粗略的适用性
检查，不是推荐扫描条件；其偏压、电流边界须针对目标样品和仪器独立验证。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 偏压绝对值上限（V）。超出即认为这一帧回答不了原子分辨。
ATOMIC_BIAS_MAX_V = 0.15

#: 电流设定下限（A）。低于此认为针尖太远。
ATOMIC_SETPOINT_MIN_A = 50e-12

#: 一些衬底的窄窗口（留给以后按材料细化；缺省走上面的通用值）。
_PER_SURFACE: dict[str, tuple[float, float]] = {
    # surface: (|bias| 上限 V, setpoint 下限 A)
    "Au(111)": (0.15, 50e-12),
    "Ag(111)": (0.15, 50e-12),
    "Cu(111)": (0.15, 50e-12),
}


@dataclass(frozen=True)
class WindowVerdict:
    """``ok`` 为假时，``reason`` 是稳定的机读码，``detail_zh`` 给人看。"""

    ok: bool
    reason: str = ""
    detail_zh: str = ""
    bias_v: float | None = None
    setpoint_a: float | None = None
    bias_max_v: float = ATOMIC_BIAS_MAX_V
    setpoint_min_a: float = ATOMIC_SETPOINT_MIN_A


#: 「我要看原子」这个意图对应的标准工作点。来源是本仓流程表：
#: ``MakeAtomicResolutionTip.eval_bias_v`` = 0.02 V、``eval_setpoint_a`` = 500 pA。
#:
#: 为什么要有这个常量：``scan_policy`` 的档位表**刻意不存 bias**（「bias 是物理
#: 意图参数，不是尺度的函数」——那句话是对的），于是工作点一路「不设 = 沿用」。
#: 但 bias 虽然不是**尺度**的函数，却是**意图**的函数：说「我要原子分辨」就等于
#: 说了要什么偏压。「**每一个 skill 在一开始会制定自己的
#: 工作点，而不是依赖上一个 skill**。」
ATOMIC_WORKING_POINT: dict[str, float] = {"bias_v": 0.02, "setpoint_a": 500e-12}


def atomic_working_point(surface: str | None = None) -> dict[str, float]:
    """「出原子分辨」这个意图的标准工作点。当前不随衬底变，留参数是为了以后细化。"""
    return dict(ATOMIC_WORKING_POINT)


def window_for(surface: str | None) -> tuple[float, float]:
    """该衬底的 (|bias| 上限 V, setpoint 下限 A)。未知衬底走通用值。"""
    if surface:
        hit = _PER_SURFACE.get(str(surface).strip())
        if hit:
            return hit
    return ATOMIC_BIAS_MAX_V, ATOMIC_SETPOINT_MIN_A


def check_atomic_window(bias_v: float | None,
                        setpoint_a: float | None,
                        surface: str | None = None) -> WindowVerdict:
    """这一帧的成像条件支不支持「有没有原子分辨」这个问题。

    **读不到条件时放行**（返回 ok）—— 缺字段是「不知道」，不是「不合格」，
    把它当不合格会让所有缺头信息的旧帧凭空变成判不了。
    """
    bmax, smin = window_for(surface)
    if bias_v is None and setpoint_a is None:
        return WindowVerdict(True, bias_v=bias_v, setpoint_a=setpoint_a,
                             bias_max_v=bmax, setpoint_min_a=smin)

    if bias_v is not None and abs(float(bias_v)) > bmax:
        return WindowVerdict(
            False, "bias_out_of_atomic_window",
            "偏压 %.3f V 超出原子分辨窗口（|V| ≤ %.2f V）—— 这一帧**回答不了**"
            "有没有原子分辨，**不要读成针尖不好**。多半是修针流程改了工作点"
            "没改回来（PulseConditionTip 会设成 %.2f V）。"
            % (float(bias_v), bmax, 0.05),
            bias_v=bias_v, setpoint_a=setpoint_a,
            bias_max_v=bmax, setpoint_min_a=smin)

    if setpoint_a is not None and float(setpoint_a) < smin:
        return WindowVerdict(
            False, "setpoint_below_atomic_window",
            "电流设定 %.1f pA 低于 %.0f pA —— 针尖太远，原子起伏按指数衰减，"
            "这一帧**回答不了**，**不要读成针尖不好**。"
            % (float(setpoint_a) * 1e12, smin * 1e12),
            bias_v=bias_v, setpoint_a=setpoint_a,
            bias_max_v=bmax, setpoint_min_a=smin)

    return WindowVerdict(True, bias_v=bias_v, setpoint_a=setpoint_a,
                         bias_max_v=bmax, setpoint_min_a=smin)


__all__ = [
    "ATOMIC_BIAS_MAX_V",
    "ATOMIC_WORKING_POINT",
    "atomic_working_point",
    "ATOMIC_SETPOINT_MIN_A",
    "WindowVerdict",
    "check_atomic_window",
    "window_for",
]
