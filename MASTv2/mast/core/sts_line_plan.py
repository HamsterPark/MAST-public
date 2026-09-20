"""跨畴界线谱的**几何** —— 纯函数、零 IO、可离线全测。

一条穿过畴界的线谱,全部的价值就在于**跨过那道界**。所以几何必须能在**跑之前**
画出来给人看(conduct 面板 / dry-run),这要求它一行硬件代码都不碰。本模块只回答
三个问题:点位在哪、按什么顺序采、这一晚要跑多久。**采集本身不在这里。**

## 轴叫 `axis`,不叫 `normal`(S4 STS 设计 D22 / 陷阱 23)

上游(畴界搜索)产出的是一个 **bracket**:一对判定不同的点 ``p_lo`` / ``p_hi``,
二分到两者相距不超过一个帧宽为止。可用的方向只有 **bracket 轴** ``p_hi − p_lo``
—— 那是**搜索线**的方向,不是畴界的法向。二分沿着种子网格上相邻两点连成的线走,
畴界可以与那条线成任意夹角 θ,两者只在 θ=0 时重合。

⇒ 本模块所有对外字段一律叫 ``axis_*``。**名字必须说实话**:叫它法向,下一个人
就会拿它去算「畴界宽度」,而那个数会系统性偏大 1/cos θ,并且**不会有任何东西
报警**。有一条测试专门断言这些字段名 —— 一条看着多余的测试,挡的是一次静默的系统
误差。

## 精细窗不得窄于定位不确定度(D22 / 陷阱 24 —— 最贵的那种失败)

畴界在 bracket 内的**位置是未知的**,只知道它落在 ``|p_hi − p_lo|`` 之内。精细窗
比这个区间窄,密采的点就可能**整片落在畴界的同一侧**:跑满一晚上、数据齐全、每
一条谱都合格,而结论是空的 —— **没有任何单点判据会对此报警**。

所以 ``fine_half_width_m = max(spec.fine_half_width_nm × 1e-9, uncertainty_m)``,
这个 ``max`` 写在本模块里,由测试钉住并做变异验证。它**不能**靠调用方记得。

## 采集顺序 = ``|s|`` 升序(D23)

一条线可能要跑半小时以上(见 :func:`estimate_line_duration`),压电漂移会让命令
坐标与真实畴界位置逐渐错开。按 ``|s|`` 从小到大采,让漂移损伤的是**物理上最不
重要**的远端点。顺序写进 :attr:`LinePoint.order`,事后才能把「第 k 个点」与漂移
量对上。

代价是真实的,而且本模块**如实报出来**:这个顺序在畴界两侧来回横跳,走的路程远
大于按空间顺序走一遍(见 ``move_total_s`` 与 ``move_total_s_if_spatial``)。

## 三种输入形态是闭集,跨站点一律拒绝(D25 / 陷阱 25)

``bracket`` / ``mixed_frame`` / ``explicit``。跨站点的定位结果只有**粗动步数**,
没有米坐标 —— 上游明令不许把「±N 步」伪装成米坐标(与禁止 ``est_disp_m`` 参与
计算同一条理由)。**没有米坐标就无从布点**:拒绝并说清楚要先把它收进单个站点内,
而不是就近编一个坐标。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ── 拒绝 ────────────────────────────────────────────────────────────────────

#: :class:`LinePlanRefused` 的 ``code`` 闭集。拒绝的**理由**要能被程序分辨,
#: 否则调用方只能拿字符串做判断,而文案是会改的。
REFUSAL_CODES: tuple[str, ...] = (
    "zero_length_axis",        # p_hi == p_lo:方向无从定义,**不是**除零
    "axis_not_finite",
    "origin_not_finite",
    "cross_site_no_metres",    # 只有步数没有米坐标
    "unknown_input_form",
    "missing_axis",            # mixed 帧没有 bracket 也没给显式角度 —— 不猜
    "missing_uncertainty",     # 显式几何没说定位不确定度 —— 「不知道」不许当 0
    "bad_spec",
    "fine_zone_over_budget",   # 光精细区就超过点数上限
)


class LinePlanRefused(ValueError):
    """几何算不出来 ⇒ **拒绝,不去编一个**。

    ``code`` 取自 :data:`REFUSAL_CODES`;``args[0]`` 是给人看的一句话,拒绝时
    必须说得出**下一步该做什么**(照本仓「能停就要能解」的纪律)。
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


# ── 流程表 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class STSLineSpec:
    """一条跨畴界线谱的布点参数。**LLM 一个都不填**(D22 末段)。

    理由:畴界的物理尺度是几个晶格常数。均匀间距要么在墙上采不够、要么在畴内
    浪费大半个晚上 —— 这不是模型能从上下文里推出来的数,它必须来自流程表或
    conduct spec(与「按组名下发、零数值参数」同一条纪律)。

    ⚠️ 出厂值是**几何占位,不是标定值**:它们只保证「近墙密、远墙疏」这个形状
    成立,不代表在任何具体样品上合适。真正把这条线做对的数字要由用户/模板层
    给。唯一**不**由 spec 说了算的是精细窗 —— 它的下界由上游的定位不确定度
    决定(见 :func:`resolve_fine_half_width_m`)。
    """

    #: 精细区(近墙)点距。比典型晶格常数大、比一个原子帧小的保守占位。
    fine_spacing_nm: float = 1.0
    #: 粗区(远墙)点距。必须 ≥ ``fine_spacing_nm``,否则「近密远疏」是句假话。
    coarse_spacing_nm: float = 5.0
    #: 精细区半宽的**下限**。实际生效值 = max(本值, 定位不确定度)。出厂值取与
    #: 上游典型原子帧同量级(``AtomicTipWorkflow.eval_frame_nm`` = 5 nm)。
    fine_half_width_nm: float = 5.0
    #: 线的半长。至少要伸到精细窗之外,否则两端可能都还在不确定度带里。
    line_half_length_nm: float = 25.0
    #: 点数上限(预算,不是物理)。与 ``BatchRegionsScan`` 的区域上限对齐 ——
    #: 真正的约束是 checkpoint 载荷,在拿到真实载荷之前**不先拍一个大数**。
    max_points: int = 64


#: 出厂基线。见 :class:`STSLineSpec` 的警示:这是几何占位,不是标定。
LINE_STS = STSLineSpec()


#: 字段边界。**越界拒绝,不夹紧** —— 与 ``noble_tip_workflow._BOUNDS`` /
#: ``special_tip_workflow._BOUNDS`` 同一条论证:夹紧会让调用方以为自己设的是 X
#: 而实际跑的是 Y。
_BOUNDS: dict[str, tuple[float, float]] = {
    "fine_spacing_nm": (0.01, 100.0),
    "coarse_spacing_nm": (0.01, 1000.0),
    "fine_half_width_nm": (0.0, 1000.0),
    "line_half_length_nm": (0.01, 5000.0),
    "max_points": (1, 400),
}

_INT_FIELDS = frozenset({"max_points"})


def resolve_line_spec(overrides: "dict[str, Any] | None" = None,
                      base: STSLineSpec = LINE_STS) -> STSLineSpec:
    """出厂基线 + 显式给的值;``None`` 当没给,越界**丢弃并写日志**(不夹紧)。"""
    if not overrides:
        return base
    known = {f.name for f in fields(base)}
    clean: dict[str, Any] = {}
    for key, val in overrides.items():
        if key not in known or val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            logger.warning("线谱几何: %s=%r 不是数字,忽略", key, val)
            continue
        if not math.isfinite(num):
            logger.warning("线谱几何: %s=%r 不是有限数,忽略", key, val)
            continue
        lo, hi = _BOUNDS.get(key, (float("-inf"), float("inf")))
        if not (lo <= num <= hi):
            logger.warning("线谱几何: %s=%g 超出 [%g, %g],忽略(不夹紧)",
                           key, num, lo, hi)
            continue
        clean[key] = int(round(num)) if key in _INT_FIELDS else num
    return replace(base, **clean) if clean else base


# ── 输入形态(闭集)───────────────────────────────────────────────────────


#: S4 接受的输入形态。**闭集** —— 见模块注释与 D25。
INPUT_FORMS: tuple[str, ...] = ("bracket", "mixed_frame", "explicit")

#: 上游对「跨站点畴界」的形态名。它**没有米坐标**,本模块一律拒绝。
CROSS_SITE_FORM = "cross_site"

#: 跨站点拒绝的原话。测试按这个子串断言 —— 拒绝必须说得出下一步。
CROSS_SITE_HINT = "需要先把它收进单个站点内"

#: marker meta 里一旦出现这些键,说明定位结果是**粗动步数**而不是米坐标。
_STEPS_ONLY_KEYS = ("uncertainty_steps", "gap_steps", "gap_moves", "steps",
                    "site_index_lo", "site_index_hi")


@dataclass(frozen=True)
class LineGeometry:
    """布点所需的最小几何:原点、**bracket 轴**、定位不确定度。

    ``axis_x`` / ``axis_y`` 是**未归一化**的轴向量(通常就是 ``p_hi − p_lo``);
    归一化在 :func:`plan_line_detail` 里做一次。``uncertainty_m`` 是畴界位置的
    已知上界 —— 它决定精细窗的下限,是本设计里最贵的那条防线的输入。
    """

    form: str
    origin_x_m: float
    origin_y_m: float
    axis_x: float
    axis_y: float
    uncertainty_m: float
    #: 这组几何是怎么来的(哪个 bracket / 哪一帧 / 谁给的),写进产物供事后重判。
    source_note: str = ""


def looks_like_cross_site(meta: "dict | None", *,
                          x_m: "float | None" = None,
                          y_m: "float | None" = None) -> bool:
    """这条记录是不是「只有步数、没有米坐标」的跨站点结果。

    **保守判定**:出现步数类键,或者干脆没有米坐标,都算。宁可多问一句,也不要
    拿一个不存在的坐标去布点 —— 后者会安静地跑完一整晚。
    """
    m = meta or {}
    if str(m.get("form") or m.get("role") or "") == CROSS_SITE_FORM:
        return True
    if any(m.get(k) is not None for k in _STEPS_ONLY_KEYS):
        return True
    return x_m is None or y_m is None


def _finite(v: Any) -> "float | None":
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pair(xy: Any, what: str, code: str) -> tuple[float, float]:
    try:
        a, b = xy  # type: ignore[misc]
    except Exception:  # noqa: BLE001
        raise LinePlanRefused(code, f"{what} 不是一对坐标: {xy!r}") from None
    fa, fb = _finite(a), _finite(b)
    if fa is None or fb is None:
        raise LinePlanRefused(code, f"{what} 里有非有限数: {xy!r}")
    return fa, fb


def resolve_line_geometry(
    form: str,
    *,
    lo_xy: "Sequence[float] | None" = None,
    hi_xy: "Sequence[float] | None" = None,
    center_xy: "Sequence[float] | None" = None,
    frame_size_m: "float | None" = None,
    origin_xy: "Sequence[float] | None" = None,
    axis_xy: "Sequence[float] | None" = None,
    axis_deg: "float | None" = None,
    uncertainty_m: "float | None" = None,
    source_note: str = "",
) -> LineGeometry:
    """把三种输入形态之一收敛成 :class:`LineGeometry`。**纯函数。**

    * ``bracket`` —— 给 ``lo_xy`` / ``hi_xy``。原点 = 中点,轴 = ``hi − lo``,
      不确定度 = ``|hi − lo|``(二分只保证畴界落在这段里,**不保证在中点**)。
    * ``mixed_frame`` —— 给 ``center_xy`` + ``frame_size_m``。原点 = 帧中心,
      不确定度 = 帧尺寸(畴界就在这一帧里,位置未知)。轴向要么由 bracket 给
      (``lo_xy``/``hi_xy`` 或 ``axis_xy``),要么**显式给** ``axis_deg``;两者都
      没有 ⇒ 拒绝。**不拿帧角当轴角**:帧角读不到时是 ``None`` 而不是 0,拿它
      去当轴向就是又一次「读不到被当成答了」。
    * ``explicit`` —— 给 ``origin_xy`` + (``axis_xy`` 或 ``axis_deg``) +
      ``uncertainty_m``。不确定度**必填**:留空默认成 0 会让精细窗退回 spec 的
      值,正好踩中陷阱 24。真的确知位置就显式写 0,那是一句话,不是一个默认。

    ``axis_deg`` 是与 xy **同一坐标系**的方位角(+x 为 0°,逆时针为正),
    本模块**不做任何帧角换算**。
    """
    f = str(form or "").strip()
    if f == CROSS_SITE_FORM:
        raise LinePlanRefused(
            "cross_site_no_metres",
            "这条畴界只定位到粗动步数(站点 k 与 k+1 之间 ±N 步),没有米坐标,"
            f"线谱无从布点。{CROSS_SITE_HINT}:在两站之间做一次半步长粗动把新站点"
            "插进去,再在单个站点内重新定位。")
    if f not in INPUT_FORMS:
        raise LinePlanRefused(
            "unknown_input_form",
            f"不认识的输入形态 {form!r};只接受 {list(INPUT_FORMS)}"
            f"(跨站点结果请见 {CROSS_SITE_HINT} 的说明)。")

    axis: "tuple[float, float] | None" = None
    if axis_xy is not None:
        axis = _pair(axis_xy, "轴向量", "axis_not_finite")
    elif axis_deg is not None:
        ang = _finite(axis_deg)
        if ang is None:
            raise LinePlanRefused("axis_not_finite", f"axis_deg 不是有限数: {axis_deg!r}")
        axis = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))

    if f == "bracket":
        lo = _pair(lo_xy, "bracket 的 lo 点", "origin_not_finite")
        hi = _pair(hi_xy, "bracket 的 hi 点", "origin_not_finite")
        ax = (hi[0] - lo[0], hi[1] - lo[1])
        span = math.hypot(*ax)
        # 不确定度就是 bracket 的跨度本身:二分只保证畴界在这段里。显式给的值
        # 只允许**放大**它 —— 缩小等于声称比上游更确定,而没人做过那次测量。
        unc = span
        given = _finite(uncertainty_m)
        if given is not None and given > span:
            unc = given
        return LineGeometry(form=f, origin_x_m=(lo[0] + hi[0]) / 2.0,
                            origin_y_m=(lo[1] + hi[1]) / 2.0,
                            axis_x=ax[0], axis_y=ax[1], uncertainty_m=unc,
                            source_note=source_note)

    if f == "mixed_frame":
        ctr = _pair(center_xy, "mixed 帧中心", "origin_not_finite")
        if axis is None and lo_xy is not None and hi_xy is not None:
            lo = _pair(lo_xy, "bracket 的 lo 点", "axis_not_finite")
            hi = _pair(hi_xy, "bracket 的 hi 点", "axis_not_finite")
            axis = (hi[0] - lo[0], hi[1] - lo[1])
        if axis is None:
            raise LinePlanRefused(
                "missing_axis",
                "这一帧被判成 mixed(畴界就在帧内),但没有 bracket 轴、也没有显式"
                "给出轴向角度。帧角不能拿来当轴向(它读不到时是 null 而不是 0)"
                "⇒ 请给 axis_deg,或者先跑一次二分拿到 bracket。")
        size = _finite(frame_size_m)
        if size is None or size <= 0:
            raise LinePlanRefused(
                "missing_uncertainty",
                "mixed 帧要用帧尺寸当定位不确定度(畴界在帧内,位置未知),"
                f"但 frame_size_m 读不到: {frame_size_m!r}。")
        unc = size
        given = _finite(uncertainty_m)
        if given is not None and given > unc:
            unc = given
        return LineGeometry(form=f, origin_x_m=ctr[0], origin_y_m=ctr[1],
                            axis_x=axis[0], axis_y=axis[1], uncertainty_m=unc,
                            source_note=source_note)

    # explicit
    org = _pair(origin_xy, "显式原点", "origin_not_finite")
    if axis is None:
        raise LinePlanRefused(
            "missing_axis", "显式几何必须给出轴向(axis_xy 或 axis_deg)。")
    unc = _finite(uncertainty_m)
    if unc is None:
        raise LinePlanRefused(
            "missing_uncertainty",
            "显式几何必须同时给出定位不确定度 uncertainty_m(畴界在轴上位置的已知"
            "上界)。留空会让精细窗退回流程表的值,而密采点可能整片落在畴界同一侧"
            "—— 那是一种「跑完了、数据也齐、结论是空的」失败。确知位置就显式写 0。")
    if unc < 0:
        raise LinePlanRefused("missing_uncertainty",
                              f"uncertainty_m 不能为负: {unc!r}")
    return LineGeometry(form="explicit", origin_x_m=org[0], origin_y_m=org[1],
                        axis_x=axis[0], axis_y=axis[1], uncertainty_m=unc,
                        source_note=source_note)


# ── 点位 ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LinePoint:
    """线上的一个点。

    ``axis_s_m`` 是沿 **bracket 轴**的带符号位移 —— **不是**沿畴界法向的距离。
    字段名带 ``axis_`` 前缀是刻意的,见模块注释。
    """

    #: 空间序号,0 = 最负端(仅用于画图与对账,**不是**采集顺序)。
    index: int
    #: 采集顺序,0 = 最先采。``|axis_s_m|`` 升序(D23)。
    order: int
    #: 沿 bracket 轴的带符号位移(m)。
    axis_s_m: float
    x_m: float
    y_m: float
    #: ``fine`` / ``coarse`` —— 这个点属于近墙密采区还是远墙疏采区。
    zone: str
    label: str = ""

    def as_dict(self) -> dict:
        """喂给逐点引擎 / 记录层的形态。键名与本类字段一致。"""
        return {
            "index": self.index,
            "order": self.order,
            "axis_s_m": self.axis_s_m,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "zone": self.zone,
            "label": self.label,
        }


#: :attr:`LinePlan.warnings` 的闭集。结论仍然成立,但你该知道。
PLAN_WARNINGS: tuple[str, ...] = (
    "fine_window_widened_by_uncertainty",   # spec 的精细窗比定位不确定度还窄
    "line_does_not_extend_past_uncertainty",  # 两端仍可能在不确定度带里
    "points_truncated_by_budget",
)


@dataclass(frozen=True)
class LinePlan:
    """一条线的完整几何记录。字段名一律 ``axis_*``,**不是** ``normal_*``。"""

    points: tuple[LinePoint, ...]
    #: bracket 轴的单位向量。
    axis_unit_x: float
    axis_unit_y: float
    #: bracket 轴在 xy 坐标系里的方位角(deg)。**这不是畴界法向的角度。**
    axis_angle_deg: float
    #: 归一化之前的轴长(bracket 形态下 = ``|p_hi − p_lo|``)。
    axis_length_m: float
    #: 上游给的定位不确定度 —— 精细窗下限的来源。
    uncertainty_m: float
    #: **实际生效**的精细区半宽 = max(spec 值, uncertainty_m)。
    fine_half_width_m: float
    #: 上面那个 max 是谁赢的:``spec`` 或 ``uncertainty``。
    fine_half_width_source: str
    #: 实际生效的线半长。
    half_length_m: float
    origin_x_m: float
    origin_y_m: float
    form: str
    warnings: tuple[str, ...] = ()
    dropped_point_count: int = 0
    source_note: str = ""

    @property
    def point_count(self) -> int:
        return len(self.points)

    def positions(self) -> list[dict]:
        """交给逐点取谱引擎的位置表,**已按采集顺序排好**。"""
        return [p.as_dict() for p in sorted(self.points, key=lambda q: q.order)]

    def summary_dict(self) -> dict:
        """写进 SkillResult.data 的几何摘要(不含逐点表)。"""
        return {
            "form": self.form,
            "point_count": self.point_count,
            "origin_x_m": self.origin_x_m,
            "origin_y_m": self.origin_y_m,
            "axis_unit_x": self.axis_unit_x,
            "axis_unit_y": self.axis_unit_y,
            "axis_angle_deg": self.axis_angle_deg,
            "axis_length_m": self.axis_length_m,
            "uncertainty_m": self.uncertainty_m,
            "fine_half_width_m": self.fine_half_width_m,
            "fine_half_width_source": self.fine_half_width_source,
            "half_length_m": self.half_length_m,
            "warnings": list(self.warnings),
            "dropped_point_count": self.dropped_point_count,
            "source_note": self.source_note,
            # 名字说实话:这是搜索线的方向,与畴界真法向差一个未知夹角。
            "axis_is_not_the_wall_normal": True,
        }


def resolve_fine_half_width_m(spec: STSLineSpec, uncertainty_m: float) -> float:
    """精细区半宽 = ``max(spec.fine_half_width_nm × 1e-9, uncertainty_m)``。

    **这个 max 是本模块存在的头号理由**(D22 / 陷阱 24)。畴界在 bracket 内的位置
    未知,只知道它在 ``uncertainty_m`` 之内;精细窗比它窄,密采点就可能整片落在
    畴界同一侧 —— 跑满一晚、每条谱都合格、结论是空的,而且没有任何单点判据会
    报警。防线只有这一行和它的变异测试。
    """
    from_spec = float(spec.fine_half_width_nm) * 1e-9
    unc = float(uncertainty_m)
    if not math.isfinite(unc) or unc < 0:
        unc = 0.0
    return max(from_spec, unc)


def _validate_spec(spec: STSLineSpec) -> None:
    fine = float(spec.fine_spacing_nm)
    coarse = float(spec.coarse_spacing_nm)
    if not (fine > 0):
        raise LinePlanRefused("bad_spec", f"fine_spacing_nm 必须为正: {fine!r}")
    if not (coarse > 0):
        raise LinePlanRefused("bad_spec", f"coarse_spacing_nm 必须为正: {coarse!r}")
    if coarse < fine:
        # 「近墙密、远墙疏」是这条线的全部设计。反过来时不去悄悄纠正 ——
        # 那会让调用方以为自己配的是 X 而实际跑的是 Y。
        raise LinePlanRefused(
            "bad_spec",
            f"coarse_spacing_nm({coarse:g}) 小于 fine_spacing_nm({fine:g}),"
            "「近墙密远墙疏」就不成立了。")
    if int(spec.max_points) < 1:
        raise LinePlanRefused("bad_spec", f"max_points 必须 ≥ 1: {spec.max_points!r}")
    if not (float(spec.line_half_length_nm) > 0):
        raise LinePlanRefused(
            "bad_spec", f"line_half_length_nm 必须为正: {spec.line_half_length_nm!r}")


#: 去重用的量化尺度(m)。1 pm —— 远小于任何真实点距,远大于浮点误差。
_DEDUP_M = 1e-12


def plan_line_detail(origin_xy: "Sequence[float]",
                     axis_xy: "Sequence[float]",
                     uncertainty_m: float,
                     spec: "STSLineSpec | None" = None) -> LinePlan:
    """算出整条线的几何。**纯函数,零 IO。**

    Args:
        origin_xy: 线的中心(m)。bracket 形态下是 lo/hi 的中点。
        axis_xy: **bracket 轴**向量(不必归一化)。零长度 ⇒ 拒绝,不除零。
        uncertainty_m: 畴界位置的已知上界(bracket 形态 = ``|p_hi − p_lo|``)。
        spec: 布点参数;留空取出厂基线。

    Raises:
        LinePlanRefused: 零长度轴 / 非有限数 / spec 不自洽 / 精细区就超预算。
    """
    sp = spec or LINE_STS
    _validate_spec(sp)

    ox, oy = _pair(origin_xy, "线的原点", "origin_not_finite")
    ax, ay = _pair(axis_xy, "bracket 轴", "axis_not_finite")

    axis_len = math.hypot(ax, ay)
    if axis_len <= 0.0:
        # p_hi == p_lo。方向根本没有定义 —— 归一化会除零,而随便挑一个方向去跑
        # 一晚上是更坏的结局。
        raise LinePlanRefused(
            "zero_length_axis",
            "bracket 轴长度为 0(p_hi 与 p_lo 是同一个点),方向无从定义。"
            "请先把二分跑到两端确实不同,或者显式给出轴向角度。")
    ux, uy = ax / axis_len, ay / axis_len

    unc = _finite(uncertainty_m)
    if unc is None or unc < 0:
        raise LinePlanRefused(
            "missing_uncertainty",
            f"uncertainty_m 必须是 ≥ 0 的有限数: {uncertainty_m!r}")

    fine_half = resolve_fine_half_width_m(sp, unc)
    spec_fine_half = float(sp.fine_half_width_nm) * 1e-9
    widened = fine_half > spec_fine_half
    fine_src = "uncertainty" if widened else "spec"

    half_len = float(sp.line_half_length_nm) * 1e-9
    if half_len < fine_half:
        # 线比精细窗还短 ⇒ 整条线都在不确定度带里。撑到精细窗,并在下面挂警告:
        # 这时**一个粗区点都没有**,两端不保证到达任一侧畴内。
        half_len = fine_half

    fine_step = float(sp.fine_spacing_nm) * 1e-9
    coarse_step = float(sp.coarse_spacing_nm) * 1e-9
    eps = fine_step * 1e-9        # 只吃浮点残差,不改变格点

    max_points = int(sp.max_points)

    # 精细区:0, ±fine_step, ±2·fine_step … 直到 |s| ≤ fine_half。
    # 先**算**出格点数再铺,不靠循环去发现自己爆了 —— 一个 0.01 nm 的点距配一个
    # 被撑到 1 µm 的精细窗会铺十万个点,那是一次可以避免的空转。
    k_max = int(math.floor((fine_half + eps) / fine_step))
    n_fine = 2 * k_max + 1
    if n_fine > max_points:
        raise LinePlanRefused(
            "fine_zone_over_budget",
            f"光是精细区就要 {n_fine} 点,超过上限 {max_points}。精细窗被撑到 "
            f"{fine_half * 1e9:.1f} nm(定位不确定度 {unc * 1e9:.1f} nm)"
            f"、点距 {sp.fine_spacing_nm:g} nm。要么放宽 fine_spacing_nm,"
            "要么提高 max_points —— 但**不要**缩小精细窗,那正是这条线的意义所在。")
    fine_s: list[float] = [0.0]
    for k in range(1, k_max + 1):
        fine_s.extend((-k * fine_step, k * fine_step))

    # 粗区:锚在精细窗**边界**上(不是最后一个精细点上),这样精细窗边缘到第一个
    # 粗点之间至少隔一个粗间距,不会在窗边挤出两个几乎重合的点。
    # ``j_max`` 是**几何上应有**的对数,``j_gen`` 是预算允许铺出来的对数 ——
    # 两者的差就是被预算砍掉的点数,如实记在 ``dropped_point_count`` 里。铺一堆
    # 马上要丢的点没有意义,但少报「丢了多少」有意义,所以两个数分开算。
    j_max = max(0, int(math.floor((half_len - fine_half + eps) / coarse_step)))
    j_gen = min(j_max, max(0, (max_points - n_fine + 1) // 2))
    coarse_s: list[float] = []
    for j in range(1, j_gen + 1):
        s = fine_half + j * coarse_step
        coarse_s.extend((-s, s))

    zone_of: dict[int, str] = {}
    for s in fine_s:
        zone_of[int(round(s / _DEDUP_M))] = "fine"
    for s in coarse_s:
        zone_of.setdefault(int(round(s / _DEDUP_M)), "coarse")

    warnings: list[str] = []
    if widened:
        warnings.append("fine_window_widened_by_uncertainty")
    if j_max <= 0:
        # 线没有伸出精细窗 ⇒ 两端仍可能落在不确定度带内,也就可能都没进到畴内。
        warnings.append("line_does_not_extend_past_uncertainty")

    # 预算不够时丢的是**最远**的粗点:漂移损伤最不重要的就是它们(D23 同一条
    # 理由)。精细点一个都不丢 —— 精细区本身超预算在上面已经拒绝过了。
    keys = sorted(zone_of, key=lambda q: (abs(q), q))
    dropped = 2 * (j_max - j_gen)
    if len(keys) > max_points:
        dropped += len(keys) - max_points
        keys = keys[:max_points]
    if dropped:
        warnings.append("points_truncated_by_budget")

    # 采集顺序:|s| 升序;同一距离的两侧用带符号值定序(⇒ 0, −d, +d, −2d, +2d …),
    # 两侧交替,漂移不会全砸在一侧。
    ordered = sorted(keys, key=lambda q: (abs(q), q))
    # 空间序号:从最负端到最正端,只用于画图与对账。
    spatial = sorted(keys)
    index_of = {q: i for i, q in enumerate(spatial)}

    points: list[LinePoint] = []
    for order, q in enumerate(ordered):
        s = q * _DEDUP_M
        points.append(LinePoint(
            index=index_of[q],
            order=order,
            axis_s_m=s,
            x_m=ox + ux * s,
            y_m=oy + uy * s,
            zone=zone_of[q],
            label=f"线谱 s={s * 1e9:+.2f} nm",
        ))
    points.sort(key=lambda p: p.index)

    return LinePlan(
        points=tuple(points),
        axis_unit_x=ux,
        axis_unit_y=uy,
        axis_angle_deg=math.degrees(math.atan2(uy, ux)) % 360.0,
        axis_length_m=axis_len,
        uncertainty_m=unc,
        fine_half_width_m=fine_half,
        fine_half_width_source=fine_src,
        half_length_m=half_len,
        origin_x_m=ox,
        origin_y_m=oy,
        form="",
        warnings=tuple(warnings),
        dropped_point_count=dropped,
    )


def plan_line_across_wall(origin_xy: "Sequence[float]",
                          axis_xy: "Sequence[float]",
                          uncertainty_m: float,
                          spec: "STSLineSpec | None" = None) -> list[LinePoint]:
    """S4 STS 设计 D21 的签名:点位表,按**空间**顺序排。

    采集顺序在 :attr:`LinePoint.order` 里(``|s|`` 升序);要按采集顺序拿表用
    :meth:`LinePlan.positions`。轴向、生效的精细窗等元信息见 :func:`plan_line_detail`。
    """
    return list(plan_line_detail(origin_xy, axis_xy, uncertainty_m, spec).points)


def plan_from_geometry(geom: LineGeometry,
                       spec: "STSLineSpec | None" = None) -> LinePlan:
    """:func:`resolve_line_geometry` 的产物 → :class:`LinePlan`(带上形态与出处)。"""
    plan = plan_line_detail((geom.origin_x_m, geom.origin_y_m),
                            (geom.axis_x, geom.axis_y),
                            geom.uncertainty_m, spec)
    return replace(plan, form=geom.form, source_note=geom.source_note)


# ── 时间账(D24 / 陷阱 16)──────────────────────────────────────────────────


#: FolMe 时间预算的兼容默认(m/s)，不是目标仪器的校准凭据。
#: 移动开销与采谱开销均须计入；调用方应传入已核验的实际速度。
MEASURED_FOLME_SPEED_M_S = 5e-9

#: 每次 ``MoveToXY`` 的固定开销(s)。与 ``navigation.py`` 的等待预算
#: ``距离/速度 × 1.5 + 10`` 里那个常数项同源。
MOVE_OVERHEAD_S = 10.0


def _path_length(points: Sequence[LinePoint], key) -> float:
    ordered = sorted(points, key=key)
    total = 0.0
    for a, b in zip(ordered, ordered[1:]):
        total += math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)
    return total


def estimate_line_duration(points: Sequence[LinePoint],
                           *,
                           settle_s_per_point: float = 0.0,
                           acquire_s_per_point: "float | None" = None,
                           speed_m_s: "float | None" = None,
                           start_xy: "Sequence[float] | None" = None) -> dict:
    """开跑前把这一晚要花多久算出来(D24)。**纯函数。**

    移动开销与采谱开销同量级 —— 一条 51 点的线里,光是 ``MoveToXY`` 的固定开销
    就是 500 s,而 D23 的「近墙优先」顺序在畴界两侧来回横跳,路程比按空间顺序
    走一遍**大得多**。两个数都报出来,让用户看见这个顺序的代价。

    ``acquire_s_per_point`` 留空 ⇒ ``total_s`` 是 ``None``,并在 ``unknown`` 里
    点名。每条谱的真实耗时(Nanonis 的每点开销远大于 ``整定+积分`` 之和)本仓
    没有标定过,**「不知道」不许伪装成 0** —— 那会让预计时长少报一大截,而这个
    数存在的全部意义就是别让人第二天早上才发现只跑完三分之一。
    """
    pts = list(points)
    n = len(pts)
    speed = _finite(speed_m_s)
    if speed is not None and speed > 0:
        speed_used, speed_src = speed, "caller"
    else:
        speed_used, speed_src = MEASURED_FOLME_SPEED_M_S, "nominal_unconfigured"

    travel = _path_length(pts, key=lambda p: p.order)
    travel_spatial = _path_length(pts, key=lambda p: p.index)
    n_moves = max(0, n - 1)
    if start_xy is not None and pts:
        sx, sy = _pair(start_xy, "起点", "origin_not_finite")
        first = min(pts, key=lambda p: p.order)
        travel += math.hypot(first.x_m - sx, first.y_m - sy)
        travel_spatial += math.hypot(first.x_m - sx, first.y_m - sy)
        n_moves = n

    move_total = travel / speed_used + n_moves * MOVE_OVERHEAD_S
    move_spatial = travel_spatial / speed_used + n_moves * MOVE_OVERHEAD_S
    settle_total = max(0.0, float(settle_s_per_point or 0.0)) * n

    acq = _finite(acquire_s_per_point)
    unknown: list[str] = []
    if acq is None or acq < 0:
        acq_total = None
        unknown.append("acquire_s_per_point")
    else:
        acq_total = acq * n

    total = None if acq_total is None else move_total + settle_total + acq_total
    return {
        "n_points": n,
        "n_moves": n_moves,
        "travel_m": travel,
        "travel_m_if_spatial_order": travel_spatial,
        "speed_m_s": speed_used,
        "speed_source": speed_src,
        "move_total_s": move_total,
        # D23 的顺序买的是抗漂移,卖的是这段差值。把它摆出来,别让人以为免费。
        "move_total_s_if_spatial_order": move_spatial,
        "settle_total_s": settle_total,
        "acquire_total_s": acq_total,
        "total_s": total,
        #: 已知项之和 —— 采谱时间未知时,它是**下界**而不是估计值。
        "known_lower_bound_s": move_total + settle_total + (acq_total or 0.0),
        "unknown": unknown,
        "start_leg_counted": start_xy is not None,
    }


def format_duration_note(budget: dict) -> str:
    """显示时间预算，并区分名义估算与调用方提供的速度。"""
    n = int(budget.get("n_points") or 0)
    move_min = float(budget.get("move_total_s") or 0.0) / 60.0
    total = budget.get("total_s")
    nominal = budget.get("speed_source") == "nominal_unconfigured"
    suffix = ("移动速度采用未标定名义值；时长为估算，须按目标仪器核验。"
              if nominal else "")
    if total is None:
        partial_label = "已估算部分约 " if nominal else "已知部分 ≥ "
        return (f"{n} 点;移动开销约 {move_min:.0f} 分钟(FolMe "
                f"{float(budget.get('speed_m_s') or 0.0) * 1e9:.1f} nm/s),"
                f"每点采谱耗时未标定 ⇒ **总时长未知**,{partial_label}"
                f"{float(budget.get('known_lower_bound_s') or 0.0) / 60.0:.0f} 分钟。"
                f"{suffix}")
    return (f"{n} 点;预计约 {float(total) / 60.0:.0f} 分钟"
            f"(其中移动 {move_min:.0f} 分钟)。{suffix}")


__all__ = [
    "CROSS_SITE_FORM",
    "CROSS_SITE_HINT",
    "INPUT_FORMS",
    "LINE_STS",
    "LineGeometry",
    "LinePlan",
    "LinePlanRefused",
    "LinePoint",
    "MEASURED_FOLME_SPEED_M_S",
    "MOVE_OVERHEAD_S",
    "PLAN_WARNINGS",
    "REFUSAL_CODES",
    "STSLineSpec",
    "estimate_line_duration",
    "format_duration_note",
    "looks_like_cross_site",
    "plan_from_geometry",
    "plan_line_across_wall",
    "plan_line_detail",
    "resolve_fine_half_width_m",
    "resolve_line_geometry",
    "resolve_line_spec",
]
