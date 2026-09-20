"""``kind="analysis"`` 步能调的**纯函数**注册表。

设计:``campaign_director_design.md`` §4.3(``analysis_fn`` 指向本注册表)、
§6-4(analysis 步在进程内执行,**不取仪器令牌**)。

## 三条硬约束

**一、纯函数。** 不碰仪器、不写库、不读全局可变状态。它们在 Director 线程里
同步跑,一次阻塞就是整条决策循环的停滞;而且 conduct 重跑时它们必须给同样的答案。

**二、数值全部来自参数。** 这里一个物理量都不许拍。要点位就让调用方把点位给
进来 —— ``io/map_analysis.py`` 开头那句「never by asking a language model to
look at a picture and guess」是同一条规矩的另一头:**不发明坐标**。所以
:func:`plan_sts_points` 宁可拒绝,也不替谁挑几个点。

**三、声明什么就要产出什么。** 返回的 dict 必须覆盖那一步 ``produces`` 里声明
的每个字段(由 Director 核对)。少一个,下游那条 binding 就会在运行时取不到 ——
而 binding **没有默认值兜底**,那是设计的一部分。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: name → 纯函数。函数签名一律 ``fn(params: dict) -> dict``。
_REGISTRY: "dict[str, Callable[[dict], dict]]" = {}


class AnalysisError(RuntimeError):
    """分析步自己拒绝。**说清为什么** —— 一个没有理由的失败没法被人处置。"""


def register(name: str, fn: "Callable[[dict], dict]") -> None:
    if name in _REGISTRY and _REGISTRY[name] is not fn:
        # 同名覆盖在 registry 那边有过教训:静默覆盖会让两个不同的东西共用一个
        # 名字,而调用方拿到哪一个取决于 import 顺序。
        raise ValueError(f"分析函数 {name!r} 已经注册过一个不同的实现")
    _REGISTRY[name] = fn


def get(name: str) -> "Callable[[dict], dict]":
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"没有分析函数 {name!r};已注册 {sorted(_REGISTRY)}") from None


def known_names() -> "frozenset[str]":
    """给 ``validate_spec(analyses=…)`` 用。"""
    return frozenset(_REGISTRY)


# ── 参数读取的小工具(拒绝比猜好)────────────────────────────────────

def _need(params: dict, name: str) -> Any:
    if name not in params or params[name] is None:
        raise AnalysisError(
            f"缺少参数 {name!r} —— 分析步不猜数,请在 spec 的 params 或 bindings "
            f"里给它。已收到 {sorted(params)}")
    return params[name]


def _floats(text: str, what: str) -> list[float]:
    out: list[float] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(float(chunk))
        except ValueError:
            raise AnalysisError(f"{what} 里的 {chunk!r} 不是数值") from None
    if not out:
        raise AnalysisError(f"{what} 是空的 —— 序列不由这里发明")
    return out


#: 压电可达范围(米)。与 ``MoveToXY`` 的 ParameterSpec 同界:一个超出压电行程的
#: 坐标不是「远一点」,是根本到不了。
_PIEZO_ABS_MAX_M = 1.5e-6


# ── 注册的分析函数 ──────────────────────────────────────────────────

def plan_bias_series(params: dict) -> dict:
    """逐偏压成像计划 —— 交给 ``ExecuteScanPlan`` 原样执行。

    排帧用的是 ``core.scan_planner.plan_batch`` **那一个**规划器,不在这里另写
    一份:同一个动作的第 N 份实现里,通常只有一份是对的。

    产出的计划顶层盖 ``coord_epoch`` 章(修复项 的生产侧要求:「conduct L0 步
    下发坐标时一律带章」)。查不到代次就**不盖**,不盖 0 —— 0 是「还没粗动过」
    这个真实答案。消费侧 ``ExecuteScanPlan`` 认这个章,粗动之后整批拒绝。
    """
    from mast.core.coord_epoch import read_current_epoch
    from mast.core.scan_planner import PlanReject, plan_batch

    biases = _floats(_need(params, "biases_v"), "biases_v")
    size_m = float(_need(params, "size_m"))
    intent = {"kind": "bias_series", "size_m": size_m,
              "series": {"param": "bias_v", "values": biases}}
    plan = plan_batch(intent, tip_xy=(0.0, 0.0))
    if isinstance(plan, PlanReject):
        raise AnalysisError(
            f"规划器拒绝({plan.code}): {plan.detail};"
            f"可行替代: {plan.alternatives}")
    plan_dict = plan.as_dict()
    epoch = read_current_epoch()
    if epoch is not None:
        plan_dict["coord_epoch"] = epoch
    return {"plan_json": json.dumps(plan_dict, ensure_ascii=False),
            "n_frames": len(plan.frames),
            "coord_epoch": epoch}


def plan_sts_points(params: dict) -> dict:
    """取谱点位 —— **只整理调用方给的点,不发明点**。

    ``positions_m`` 形如 ``"1e-8,2e-8; -3e-8,0"``(分号分点、逗号分 xy,单位米)。
    没给就拒绝:显式几何参数保留给 conduct spec 与用户,让模型或占位实现去
    挑几个点,是「发明坐标」那条老账的入口。

    ``n_points`` 是**上限**,不是目标数:多了截断并说出来,少了不补。

    ## 盖 ``coord_epoch`` 章(2026-08-15 补)

    与 :func:`plan_bias_series` **同一条规矩**、同一个理由:修复项 的生产侧要求
    「conduct L0 步下发坐标时一律带章」。这里原来没盖 —— 两个并排的规划器,
    同一条要求,只有一个做了;于是这一串点位的代次保护全靠用户在
    ``params.coord_epoch`` 里**手填**一个数,而手填的数与这批点位的真实出身
    之间没有任何东西对得上账。章由**实时查得**,不由人填。

    查不到代次就**不盖**,不盖 0 —— 0 是「这个作用域还没粗动过」这个真实答案。
    消费侧(``director._check_coord_frame`` 与 ``SpectroscopyAtPositions``)
    把「没盖章」与「对不上」分成两件事。
    """
    from mast.core.coord_epoch import read_current_epoch

    raw = str(_need(params, "positions_m"))
    cap = int(_need(params, "n_points"))
    if cap <= 0:
        raise AnalysisError("n_points 必须为正")

    points: list[tuple[float, float]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xy = _floats(chunk, "positions_m")
        if len(xy) != 2:
            raise AnalysisError(f"点 {chunk!r} 不是一对 x,y")
        x, y = xy
        if abs(x) > _PIEZO_ABS_MAX_M or abs(y) > _PIEZO_ABS_MAX_M:
            # 超出压电行程不是「远一点」,是根本到不了 —— 拒绝,不夹紧。
            raise AnalysisError(
                f"点 ({x:g}, {y:g}) m 超出压电可达范围 "
                f"±{_PIEZO_ABS_MAX_M:g} m —— 拒绝,不夹紧。"
                f"单位是**米**:10 nm 写 1e-8")
        points.append((x, y))
    if not points:
        raise AnalysisError("positions_m 里一个点都没有 —— 点位不由这里发明")

    truncated = len(points) > cap
    kept = points[:cap]
    epoch = read_current_epoch()
    out = {
        "positions_json": json.dumps([{"x_m": x, "y_m": y} for x, y in kept]),
        "n_points": len(kept),
        "truncated": truncated,
    }
    if epoch is not None:
        out["coord_epoch"] = epoch
    return out


register("plan_bias_series", plan_bias_series)
register("plan_sts_points", plan_sts_points)

# 判据③(跨点聚合)的实现在 mast.conduct.cross_check —— 同样是纯函数,只是它
# 自带一族数据类型,放这里会把本文件变成两件事。**注册在这里,实现在那里。**
from mast.conduct.cross_check import cross_check_analysis  # noqa: E402

register("aggregate_cross_points", cross_check_analysis)


__all__ = ["AnalysisError", "register", "get", "known_names",
           "cross_check_analysis", "plan_bias_series", "plan_sts_points"]
