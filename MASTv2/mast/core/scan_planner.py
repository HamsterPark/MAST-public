"""多图规划器 —— 「一下子要扫若干张图」时,谁来决定扫哪几张、按什么参数、什么顺序。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

判据:

> 「如果 llm 一下子要扫若干张图片呢?这时候就要有规划脚本 skill 了:怎样在当前的
> 扫描区域选择合适的几个区域作为目标,用户是否要求了变偏压、变速度、变像素密度,
> 是否要 zoomin 以及在哪里 zoomin……这些要预设好脚本,让 llm 去决定这些东西,
> 他没有这个知识的,不能让他去决定的。」

本模块是那个脚本的**纯函数核心**:意图 → 逐帧计划。零 I/O、零硬件、零 LLM,所以
可以完整地单测。硬件侧的执行与守卫在
:mod:`mast.skills.composite.plan_scan_batch`。

## 分工

  * **LLM 说的**:要几张、什么意图(巡查 / 放大 / 变偏压系列 / 重复)、目标特征
    (闭集词表)、以及用户逐字点名过的数值;
  * **本模块决定的**:每张扫哪儿(避开已毁区、避开彼此)、每张用什么参数(查档位表)、
    按什么顺序(减少大跳变)、以及算不出来时**怎么诚实地拒绝**。

## 为什么这些不能交给模型

  * 选点要读覆盖率栅格与避让圆 —— 那是图形学,模型看不到数组;
  * 参数按尺度查表 —— 那是用户的偏好,不是模型的知识;
  * 偏压序列穿零要走安全通道 —— 那是会撞针的领域常识;
  * 同一个请求两次要给同样的计划 —— 模型不保证这一点。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: 意图类别(闭集)。模型只能从这里挑,不能自由造词。
INTENT_KINDS = (
    "survey",            # 在未覆盖的表面上铺 N 张
    "zoomin",            # 在当前帧里挑特征放大
    "bias_series",       # 同一位置,逐张变偏压
    "param_series",      # 同一位置,逐张变某个参数
    "repeat",            # 同一位置同一参数,重复 N 张(时序演化)
    "regions_explicit",  # 调用方直接给定区域
)

#: 目标特征(闭集)= 四类分割词表 + 人工标记。
#: **刻意不含任何具体物理缺陷类型** —— 分割给的是几何/异常语义,
#: DEFECT = 「异常凸凹」,不等于任何特定缺陷。见 :data:`REJECT_CODES`。
FEATURE_KINDS = ("terrace", "step_edge", "defect", "contamination", "user_marked")

#: series 可以扫的参数(闭集)。
SERIES_PARAMS = ("bias_v", "setpoint_a", "size_m", "pixels", "line_time_s")

#: 结构化拒绝码。规划器**绝不把「不确定」吞成静默妥协** —— 要么给完整计划,
#: 要么给带 code 的拒绝 + 可行替代。
REJECT_CODES = (
    "needs_semantic_vision",   # 超出四类分割能表达的语义
    "series_without_values",   # 要变某个参数却没说变成什么 —— 不发明序列
    "no_uncovered_area",       # 这片表面在当前策略下扫不动了
    "candidates_exhausted",    # 过滤后候选少于请求数(可部分执行)
    "out_of_safe_range",       # 目标出压电安全范围
    "budget_exceeded",         # 时间预算装不下
    "bad_intent",              # 意图本身不合法(未知 kind / 缺必填字段)
)

#: 一次批次最多几帧。上限来自 BatchRegionsScan 的既有约定。
MAX_FRAMES = 64

#: 「大跳变」判据:移动距离超过这个倍数的帧宽,就要在扫之前插入稳定等待
#: (压电蠕变在大跳变后要一段时间才安定)。
BIG_MOVE_FRAMES = 5.0


@dataclass
class PlannedFrame:
    """计划里的一帧。参数已经解析完毕,可以直接下发。"""

    index: int
    center_x_m: float
    center_y_m: float
    size_m: float
    label: str = ""
    #: 该帧的完整解析结果(mast.core.scan_resolver.ResolvedScan),含 trace。
    resolved: Any = None
    #: 这一帧相对上一帧移动了多远(米);第一帧是相对起点。
    move_distance_m: float = 0.0
    #: 移动超过 BIG_MOVE_FRAMES 个帧宽 → 需要稳定等待。
    needs_settle: bool = False
    #: 相对上一帧变化了的参数名(下发时只动这些)。
    changed_params: tuple[str, ...] = ()
    reason: str = ""

    def footprint(self) -> dict[str, Any]:
        """给 show_plan_on_map 用的 marker 形状。"""
        return {
            "kind": "scan",
            "x_m": self.center_x_m,
            "y_m": self.center_y_m,
            "w_m": self.size_m,
            "h_m": self.size_m,
            "label": self.label or f"#{self.index + 1}",
        }


@dataclass
class PlanReject:
    """结构化拒绝。``code`` 在 :data:`REJECT_CODES` 里,``alternatives`` 给可行替代。"""

    code: str
    detail: str
    alternatives: list[str] = field(default_factory=list)
    #: 部分可行时(candidates_exhausted)已经排出来的帧。
    partial: list[PlannedFrame] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "alternatives": list(self.alternatives),
            "partial_frames": len(self.partial),
        }


@dataclass
class ScanPlan:
    """一份可执行的多图计划。"""

    kind: str
    frames: list[PlannedFrame]
    total_estimated_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_frames": len(self.frames),
            "total_estimated_s": self.total_estimated_s,
            "total_estimated_min": self.total_estimated_s / 60.0,
            "notes": list(self.notes),
            "frames": [
                {
                    "index": f.index,
                    "center_x_m": f.center_x_m,
                    "center_y_m": f.center_y_m,
                    "size_m": f.size_m,
                    "label": f.label,
                    "tier": getattr(f.resolved, "tier_name", None),
                    "line_time_s": (getattr(f.resolved, "configure_scan", {}) or {})
                        .get("line_time_s"),
                    "pixels": (getattr(f.resolved, "set_scan_buffer", {}) or {})
                        .get("pixels"),
                    "bias_v": ((getattr(f.resolved, "set_bias", None) or {})
                               .get("bias_v")),
                    "estimated_s": getattr(f.resolved, "estimated_scan_s", 0.0),
                    "move_distance_m": f.move_distance_m,
                    "needs_settle": f.needs_settle,
                    "changed_params": list(f.changed_params),
                    "reason": f.reason,
                }
                for f in self.frames
            ],
        }


def expand_series(spec: Any) -> "list[float] | None":
    """把 series 规格展开成数值列表。展不开返回 None(**绝不发明序列**)。

    接受 ``{"values": [...]}`` 或 ``{"start", "stop", "n", "spacing"}``。
    """
    if not isinstance(spec, dict):
        return None
    values = spec.get("values")
    if isinstance(values, (list, tuple)) and values:
        out = []
        for v in values:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            if fv != fv:                       # NaN
                return None
            out.append(fv)
        return out

    start, stop, n = spec.get("start"), spec.get("stop"), spec.get("n")
    if start is None or stop is None or n is None:
        return None
    try:
        start, stop, n = float(start), float(stop), int(n)
    except (TypeError, ValueError):
        return None
    if n < 1 or n > MAX_FRAMES:
        return None
    if n == 1:
        return [start]
    spacing = str(spec.get("spacing", "linear")).lower()
    if spacing == "log":
        if start <= 0 or stop <= 0:
            return None
        ratio = (stop / start) ** (1.0 / (n - 1))
        return [start * ratio ** i for i in range(n)]
    step = (stop - start) / (n - 1)
    return [start + step * i for i in range(n)]


def order_by_travel(points: list[tuple[float, float]],
                    start: tuple[float, float]) -> list[int]:
    """贪心最近邻排序,返回下标顺序。

    n ≤ 64 不值得上 TSP;排序的目的也不主要是省时间,而是**减少大跳变** ——
    每次大位移都会重新激起压电蠕变,让接下来的一两帧带上漂移。
    """
    remaining = list(range(len(points)))
    order: list[int] = []
    cur = start
    while remaining:
        best = min(remaining,
                   key=lambda i: math.hypot(points[i][0] - cur[0],
                                            points[i][1] - cur[1]))
        order.append(best)
        cur = points[best]
        remaining.remove(best)
    return order


def order_series_monotonic(values: list[float],
                           current: "float | None") -> list[int]:
    """把系列值排成从「离当前值最近的那一端」开始的单调序。

    为什么不按给定顺序:偏压来回跳会反复激起针尖-样品结的回滞与充放电瞬态。
    单调走一遍,每一步的跳变都最小。
    """
    idx = sorted(range(len(values)), key=lambda i: values[i])
    if current is None or not idx:
        return idx
    lo, hi = values[idx[0]], values[idx[-1]]
    return idx if abs(current - lo) <= abs(current - hi) else list(reversed(idx))


def _resolve_frame(center: tuple[float, float], size_m: float,
                   explicit: dict, purpose: str, resolver):
    from mast.core.scan_resolver import ScanIntent
    return resolver(ScanIntent(
        center_x_m=center[0], center_y_m=center[1], size_m=size_m,
        purpose=purpose,
        explicit={k: v for k, v in explicit.items() if v is not None},
    ))


def _annotate_moves(frames: list[PlannedFrame],
                    start: tuple[float, float]) -> None:
    """填 move_distance_m / needs_settle / changed_params。"""
    prev_xy = start
    prev_key: dict[str, Any] = {}
    for f in frames:
        f.move_distance_m = math.hypot(f.center_x_m - prev_xy[0],
                                       f.center_y_m - prev_xy[1])
        f.needs_settle = f.move_distance_m > BIG_MOVE_FRAMES * f.size_m
        cfg = getattr(f.resolved, "configure_scan", {}) or {}
        key = {
            "line_time_s": cfg.get("line_time_s"),
            "pixels": (getattr(f.resolved, "set_scan_buffer", {}) or {}).get("pixels"),
            "bias_v": (getattr(f.resolved, "set_bias", None) or {}).get("bias_v"),
            "setpoint_a": (getattr(f.resolved, "set_setpoint", None) or {})
                .get("setpoint_a"),
            "size_m": cfg.get("width_m"),
        }
        f.changed_params = tuple(
            k for k, v in key.items()
            if v is not None and prev_key.get(k) != v
        )
        prev_key = key
        prev_xy = (f.center_x_m, f.center_y_m)


def plan_batch(
    intent: dict,
    *,
    tip_xy: tuple[float, float] = (0.0, 0.0),
    survey_positions: "list[tuple[float, float]] | None" = None,
    zoom_candidates: "list[dict] | None" = None,
    resolver=None,
) -> "ScanPlan | PlanReject":
    """把一个批量扫图意图展开成逐帧计划,或给出结构化拒绝。

    位置来源由调用方注入(``survey_positions`` 来自 map_analysis 的候选序列,
    ``zoom_candidates`` 来自当前帧的分割结果),这样本函数保持纯粹、可完整单测。
    """
    if resolver is None:
        from mast.core.scan_resolver import resolve_scan as resolver

    kind = str(intent.get("kind", "")).strip().lower()
    if kind not in INTENT_KINDS:
        return PlanReject(
            "bad_intent",
            f"未知的意图类别 {intent.get('kind')!r};可用:{', '.join(INTENT_KINDS)}")

    overrides = dict(intent.get("overrides") or {})
    purpose = str(intent.get("purpose") or "auto")
    constraints = dict(intent.get("constraints") or {})
    notes: list[str] = []

    n_req = intent.get("n_images")
    try:
        n_req = int(n_req) if n_req is not None else None
    except (TypeError, ValueError):
        return PlanReject("bad_intent", f"n_images 不是整数: {n_req!r}")
    if n_req is not None and (n_req < 1 or n_req > MAX_FRAMES):
        return PlanReject(
            "bad_intent",
            f"n_images={n_req} 超出 1..{MAX_FRAMES}")

    # ── 特征词表检查:超出四类分割能表达的语义 → 诚实拒绝 ────────────────
    target = dict(intent.get("target") or {})
    feature = target.get("feature")
    if feature is not None and str(feature).lower() not in FEATURE_KINDS:
        return PlanReject(
            "needs_semantic_vision",
            (f"「{feature}」超出了分割能表达的语义。分割只有 "
             f"{', '.join(FEATURE_KINDS[:4])} 四类几何/异常类别 —— "
             "DEFECT 是「异常凸凹」,不等于任何特定的物理缺陷类型。"),
            alternatives=[
                "改用四类之一(例如 defect = 任意异常凸凹)",
                "在扫描地图上人工标点,然后用 feature='user_marked'",
            ])

    frames: list[PlannedFrame] = []

    # ── survey:在未覆盖表面上铺 N 张 ────────────────────────────────────
    if kind == "survey":
        size = float(overrides.get("size_m") or intent.get("size_m") or 0.0)
        if size <= 0:
            return PlanReject("bad_intent", "survey 需要 size_m(每张图多大)")
        want = n_req or 4
        positions = list(survey_positions or [])
        if not positions:
            return PlanReject(
                "no_uncovered_area",
                "当前策略下这片表面已经没有可用的未覆盖位置了。",
                alternatives=["粗动换到新区域", "放宽重扫阈值以复用已扫区域"])
        exhausted = len(positions) < want
        positions = positions[:want]
        order = order_by_travel(positions, tip_xy)
        for slot, i in enumerate(order):
            x, y = positions[i]
            frames.append(PlannedFrame(
                index=slot, center_x_m=x, center_y_m=y, size_m=size,
                label=f"survey#{slot + 1}",
                resolved=_resolve_frame((x, y), size, overrides, purpose, resolver),
                reason="未覆盖区域"))
        if exhausted:
            _annotate_moves(frames, tip_xy)
            return PlanReject(
                "candidates_exhausted",
                (f"只找到 {len(frames)} 个可用位置,少于请求的 {want} 张。"
                 "可以先扫这几张。"),
                alternatives=[f"按 {len(frames)} 张执行", "粗动换区后再扫剩下的"],
                partial=frames)

    # ── zoomin:在当前帧里挑特征放大 ─────────────────────────────────────
    elif kind == "zoomin":
        final_size = float(target.get("final_size_m") or overrides.get("size_m") or 0.0)
        if final_size <= 0:
            return PlanReject("bad_intent", "zoomin 需要 final_size_m(放大到多大)")
        want = n_req or 1
        cands = list(zoom_candidates or [])
        if not cands:
            return PlanReject(
                "candidates_exhausted",
                "当前帧里没有符合条件的目标(尺寸、离边距离、离台阶距离、避让区过滤后为空)。",
                alternatives=["先扫一张更大的图再挑", "换一个 feature 类别"])
        exhausted = len(cands) < want
        cands = cands[:want]
        pts = [(float(c["x_m"]), float(c["y_m"])) for c in cands]
        order = order_by_travel(pts, tip_xy)
        for slot, i in enumerate(order):
            c = cands[i]
            frames.append(PlannedFrame(
                index=slot, center_x_m=pts[i][0], center_y_m=pts[i][1],
                size_m=final_size,
                label=f"zoom#{slot + 1}:{c.get('feature', '')}",
                resolved=_resolve_frame(pts[i], final_size, overrides, purpose,
                                        resolver),
                reason=str(c.get("reason", "分割候选"))))
        if exhausted:
            _annotate_moves(frames, tip_xy)
            return PlanReject(
                "candidates_exhausted",
                f"只找到 {len(frames)} 个候选,少于请求的 {want} 张。",
                alternatives=[f"按 {len(frames)} 张执行"],
                partial=frames)

    # ── series / repeat:同一位置,逐张变一个参数 ────────────────────────
    elif kind in ("bias_series", "param_series", "repeat"):
        center = target.get("coords") or {}
        cx = float(center.get("x_m", tip_xy[0]))
        cy = float(center.get("y_m", tip_xy[1]))
        size = float(overrides.get("size_m") or intent.get("size_m") or 0.0)
        if size <= 0:
            return PlanReject("bad_intent", f"{kind} 需要 size_m")

        if kind == "repeat":
            want = n_req or 2
            for i in range(want):
                frames.append(PlannedFrame(
                    index=i, center_x_m=cx, center_y_m=cy, size_m=size,
                    label=f"repeat#{i + 1}",
                    resolved=_resolve_frame((cx, cy), size, overrides, purpose,
                                            resolver),
                    reason="时序重复"))
        else:
            series = dict(intent.get("series") or {})
            param = str(series.get("param")
                        or ("bias_v" if kind == "bias_series" else "")).lower()
            if param not in SERIES_PARAMS:
                return PlanReject(
                    "bad_intent",
                    f"series.param 必须是 {', '.join(SERIES_PARAMS)} 之一"
                    f"(收到 {param!r})")
            values = expand_series(series)
            if not values:
                return PlanReject(
                    "series_without_values",
                    (f"要变 {param} 却没说变成哪些值。规划器**不发明序列** —— "
                     "请给 values 列表,或 start/stop/n。"),
                    alternatives=["给出 values,例如 [-1.0, -0.5, 0.5, 1.0]",
                                  "给出 start / stop / n"])
            if len(values) > MAX_FRAMES:
                return PlanReject(
                    "bad_intent",
                    f"序列有 {len(values)} 个值,超过单批上限 {MAX_FRAMES}")

            # 单调序:来回跳会反复激起回滞与充放电瞬态
            current = overrides.get(param)
            order = order_series_monotonic(values, current)
            for slot, i in enumerate(order):
                val = values[i]
                per_frame = dict(overrides)
                per_frame[param] = val
                frame_size = float(val) if param == "size_m" else size
                frames.append(PlannedFrame(
                    index=slot, center_x_m=cx, center_y_m=cy, size_m=frame_size,
                    label=f"{param}={val:.4g}",
                    resolved=_resolve_frame((cx, cy), frame_size, per_frame,
                                            purpose, resolver),
                    reason=f"{param} 系列(单调序)"))

    # ── regions_explicit:调用方直接给定区域 ─────────────────────────────
    elif kind == "regions_explicit":
        regions = intent.get("regions") or []
        if not isinstance(regions, (list, tuple)) or not regions:
            return PlanReject("bad_intent", "regions_explicit 需要非空 regions 列表")
        if len(regions) > MAX_FRAMES:
            return PlanReject(
                "bad_intent",
                f"regions 有 {len(regions)} 个,超过单批上限 {MAX_FRAMES}")
        pts = []
        sizes = []
        for r in regions:
            try:
                pts.append((float(r["center_x_m"]), float(r["center_y_m"])))
                sizes.append(float(r.get("size_m")
                                   or r.get("width_m") or 0.0))
            except (KeyError, TypeError, ValueError):
                return PlanReject("bad_intent", f"区域格式非法: {r!r}")
        if any(s <= 0 for s in sizes):
            return PlanReject("bad_intent", "每个区域都需要 size_m / width_m")
        order = order_by_travel(pts, tip_xy)
        for slot, i in enumerate(order):
            frames.append(PlannedFrame(
                index=slot, center_x_m=pts[i][0], center_y_m=pts[i][1],
                size_m=sizes[i],
                label=str(regions[i].get("label") or f"region#{slot + 1}"),
                resolved=_resolve_frame(pts[i], sizes[i], overrides, purpose,
                                        resolver),
                reason="调用方指定"))

    if not frames:
        return PlanReject("bad_intent", "展开后没有任何帧")

    _annotate_moves(frames, tip_xy)
    total = sum(getattr(f.resolved, "estimated_scan_s", 0.0) for f in frames)

    budget_min = constraints.get("max_total_minutes")
    if budget_min:
        try:
            budget_s = float(budget_min) * 60.0
        except (TypeError, ValueError):
            budget_s = None
        if budget_s and total > budget_s:
            return PlanReject(
                "budget_exceeded",
                (f"这份计划估计需要 {total / 60.0:.1f} 分钟,超过预算 "
                 f"{float(budget_min):.1f} 分钟。"),
                alternatives=[
                    f"减到 {max(1, int(len(frames) * budget_s / total))} 张",
                    "降低分辨率或提高扫描速度(在档位表里改,或本次显式指定)",
                ],
                partial=frames)

    n_settle = sum(1 for f in frames if f.needs_settle)
    if n_settle:
        notes.append(f"{n_settle} 帧之前有大跳变,会先等压电稳定")

    return ScanPlan(kind=kind, frames=frames, total_estimated_s=total, notes=notes)


__all__ = [
    "INTENT_KINDS",
    "FEATURE_KINDS",
    "SERIES_PARAMS",
    "REJECT_CODES",
    "MAX_FRAMES",
    "BIG_MOVE_FRAMES",
    "PlannedFrame",
    "PlanReject",
    "ScanPlan",
    "expand_series",
    "order_by_travel",
    "order_series_monotonic",
    "plan_batch",
]
