"""修针流程的四个阶段 —— 判据与循环写在这里,三个入口技能只是薄壳。

用户描述这套流程时是模块化的:「我们实验物理学家会灵活地、模块化地运行上述步
骤,每种步骤当然可能做不止一次」。所以阶段是四个生成器,可以单独跑(``PulseCondi
tionTip`` / ``PokeConditionTip``),也可以串起来跑完整流程(``PrepareNobleTip``)。

**为什么是生成器而不是嵌套 composite**:composite 调 composite 在本仓没有先例,
断点续跑(``CompositeProgress`` + step_id)与 abort 的交互没人验证过。共用生成器
让三个入口都跑在同一套单层机制上,step_id 加个阶段前缀就够区分,而判据只有一份。

每个阶段返回一个 dict(``yield from`` 会把它交给调用方),里面既有结论也有过程 ——
打了几发、每发怎么判的、为什么停。报告里那些数字是用户复盘的唯一依据。

## 三条贯穿全流程的纪律

* **每次动手前先换地方**。原地再来一次读到的是上一次留下的坑的行为,判据就废了。
  换到哪里由扫描地图说了算(``FindCleanSpot``),不由这里猜。
* **预算是行动上限,不是重试次数**。打不动就如实报告并停下,不是无限打下去。
* **abort 在每个 yield 之间被检查**(GraphExecutor 每步之前查 abort 与 halt),所以
  阶段只要老实地一步一 yield,长流程就是可中断的。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Iterator

from mast.core.noble_tip_workflow import (
    NobleTipWorkflow,
    forge_fallback_line_time_s,
    forge_line_time_s,
)
from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor

logger = logging.getLogger(__name__)


# ── 读子步骤结果的小工具 ────────────────────────────────────────────────────

def _data(executor: GraphExecutor, step_id: str) -> dict:
    """某一步的 ``result.data``;没跑成功就是空 dict。"""
    res = executor.sub_results.get(step_id)
    if res is None:
        return {}
    return dict(getattr(res, "data", None) or {})


def _ok(executor: GraphExecutor, step_id: str) -> bool:
    """这一步成功了吗。``sub_results`` 只收成功的步骤。"""
    return step_id in executor.sub_results


def _err(executor: GraphExecutor, step_id: str) -> str:
    """这一步**为什么**失败(原文)。没记到就是空串。

    ``sub_results`` 只收成功的步骤,所以失败原因不在那里；它在
    ``progress.failed_reasons``(2026-08-12 才加的 —— 在那之前失败的原文
    只进日志/诊断/旁白,一个 ``optional=True`` 的步骤等于把失败交给了一个
    **看不见原因**的调用方)。

    **报失败时永远优先用这个,而不是「我们打算做什么」。**
    """
    try:
        return str(executor.progress.failed_reasons.get(step_id, "") or "")
    except Exception:  # noqa: BLE001 — 取诊断信息绝不许再抛
        return ""


def _spots_str(spots: list[tuple[float, float]]) -> str:
    return ";".join(f"{x},{y}" for x, y in spots)


#: ``AutoTilt`` 的四个结局。真源是 ``composite/auto_tilt.py`` 里 ``report()``
#: 的第一个实参 —— **不是** 一个叫 ``skipped`` 的布尔字段(那个字段不存在)。
_TILT_DID_ACT = ("applied",)
_TILT_NO_ACT = ("no_action_needed",)
_TILT_SKIPPED = ("skipped",)


def _tilt_readout(executor: GraphExecutor, step_id: str) -> dict:
    """``AutoTilt`` 那一步到底怎么了 —— **一份读法,两个调用方共用**。

    ## 为什么要有这个函数

    2026-08-18 查出来的:C 相和 D 相各自写了一份读法,而**两份读的键名在
    ``AutoTilt`` 的回包里全都不存在**::

        tilt_d.get("skipped")          # 没有这个键。真源是 data["outcome"]
        tilt_d.get("z_span_before_m")  # 没有。真源是 data["before"]["z_span_m"]
        tilt_d.get("z_span_after_m")   # 没有。真源是 data["after"]["z_span_m"]
        tilt_d.get("action")           # 没有。

    后果不是报错,是**每一个读数都恒为 None**:``not None`` 恒真 ⇒ 「有没有跳过」
    这一问从来没有被真正回答过,而旁白里那句「落差 A → B」一次也没出现过 ——
    它每次都走 `(落差没记下来)` 那一支,看起来像仪器没给数,其实是我们没去取。
    这是本仓 `silent_fallback_wrong_name` 那一型的又一次:**兜底值合理得让人
    看不出兜底发生了**。

    ## 三态不是两态

    ``AutoTilt`` 有四个结局,而「成功/失败」两态装不下它们:

    ==================  =========================================  =========
    outcome             含义                                        success
    ==================  =========================================  =========
    ``applied``         真的动了压电,而且复测达标                    True
    ``no_action_needed``**本来就够平,一个字没改**                    True
    ``skipped``         没标定 / 测不到 —— **没做,不是没做成**       False
    ``failed``          做了没做成(发散、硬件拒绝、迭代用尽)         False
    ==================  =========================================  =========

    把 ``no_action_needed`` 念成「调平做了」是在报告一件没发生的事;
    把 ``skipped`` 念成「调平没做成」是把一个前置条件缺失说成了一次失败 ——
    前者该去跑 ``TiltCalibrate``,后者该去查标定为什么失效,下一步完全不同。

    读不到那一步的 data 时(步骤失败 ⇒ ``sub_results`` 里根本没有它)退到
    ``_err`` 的原文,**绝不编一个原因**。
    """
    td = _data(executor, step_id) or {}
    outcome = str(td.get("outcome") or "")
    # `data` 为空有两种:步骤失败(sub_results 不收失败的步骤)、或者技能没回包。
    # 两种都不许说成「跳过」——「读不到」不是一个结局。
    if not outcome:
        return {"outcome": "", "done": False, "skipped": False,
                "reason": "", "detail": _err(executor, step_id),
                "before_m": None, "after_m": None, "hint": ""}
    before = td.get("before") if isinstance(td.get("before"), dict) else {}
    after = td.get("after") if isinstance(td.get("after"), dict) else {}
    return {
        "outcome": outcome,
        "done": outcome in _TILT_DID_ACT,
        "no_action": outcome in _TILT_NO_ACT,
        "skipped": outcome in _TILT_SKIPPED,
        "reason": str(td.get("reason") or ""),
        # ``detail`` 是给人看的那句话,``reason`` 是机器可读的短码。
        # 旁白要念的是 detail —— 念 `calibration_missing` 等于没说。
        "detail": str(td.get("detail") or ""),
        "before_m": (before or {}).get("z_span_m"),
        "after_m": (after or {}).get("z_span_m"),
        "hint": str(td.get("next_action_hint") or ""),
    }


#: 一张图的坐标**是哪来的**。每一次扫图都必须表态,由 ``scan_at_params`` 的
#: 必填参数在 Python 层面强制(忘了传直接 TypeError,不用等测试)。
#:
#: ## 为什么不是「自动找干净地方」
#:
#: 查下来**不自动** —— 6 处 ``_relocate``
#: 和 5 处扫图靠人肉配对。但也**不能改成自动找**:一半的扫图是故意指定坐标的
#: (回退图要同一片、簇图要看刚扎的那一点、验收图要回到台阶上),自动找会把它们
#: 全毁掉。
#:
#: 所以强制的不是「去找」,是**「说清楚这个坐标哪来的」**。忘了找 → 没有合法的
#: origin 可填;故意指定 → 填对应的那一个,理由留在代码里。
#:
#: 新增取值要同时补一条真实理由 —— 这张表是**受控词表**,不是自由文本。
#: 连续这么多次「重扫了但这一片没有可用平区」就认输换区(2026-08-14)。
#:
#: 一次空手不算数 —— 可能只是碰上了台阶密集的一小片。而连续几次空手说的是
#: 「这一整片台阶太密」,那是**位置的结论,不是针尖的结论**,措辞里要分开。
_MAX_DRY_REFILLS: int = 3

SCAN_ORIGINS: dict[str, str] = {
    "clean_spot": "刚由 FindCleanSpot 在地图的可用区里找来的(默认路径)",
    "same_frame": "**故意**沿用上一步的同一中心 —— 换个视野重扫同一片",
    "analysis": "**故意**用图像分析给出的坐标(台阶位置 / 平区中心)",
    "just_poked": "**故意**回到刚动过手的那一点(看簇、看坑)",
}


def scan_at_params(wf: NobleTipWorkflow, x: float, y: float, *,
                   size_nm: float, pixels: "int | None",
                   line_time_s: "float | None", origin: str) -> dict:
    """统一组装修针流程每次 ScanAt 调用的参数。
    
    origin 必须属于 SCAN_ORIGINS，仅用于审计，不传给 ScanAt。
    所有调用点经本函数传递分辨率和每线时间，防止流程工作点在调用链中丢失。
    None 表示不下发该键，由扫描解析器处理；不能把 None 作为参数值传给验证层。
    """
    # 词表校验:值写错是开发期错误,当场抛比静默放行好 —— 一个拼错的 origin
    # 和一个没想过的 origin 长得一模一样,而后者正是这道闸门要拦的东西。
    if origin not in SCAN_ORIGINS:
        raise ValueError(
            f"scan_at_params: origin={origin!r} 不在受控词表里。"
            f"合法值:{sorted(SCAN_ORIGINS)}。"
            "要加新值就同时在 SCAN_ORIGINS 里写清楚它是什么情况 —— "
            "这道闸门问的是「这个坐标哪来的」,不接受自由文本。")
    logger.debug("scan_at_params: %.1f nm @ (%.3g, %.3g) origin=%s (%s)",
                 size_nm, x, y, origin, SCAN_ORIGINS[origin])

    # ── 每线时间由针尖速度上限派生──────────────────────
    #
    # 在它之前这里原样下发流程表里那个数,而那个数是**按阶段各自拍下来的常数**:
    # 台阶/验收图 0.15 s 在 100 nm 上就是 **667 nm/s**,比已知会刮伤针尖的
    # 488 nm/s 还快 37%。验收图正是判「针尖够不够锐」
    # 的那一张 —— 在自己刚刮出来的痕迹上判针尖。
    #
    # 规矩全在 ``forge_line_time_s`` 里(只在超速时动 / None 进 None 出 /
    # 用户钉过的不夹紧只出声),这里只负责**每一次 ScanAt 都过它**。
    # 四个调用点全走本函数,由 ``test_forge_scan_working_point`` 的结构闸门钉着,
    # 所以接在这里等于接在全部四处 —— 这正是本函数存在的理由。
    line_time_s, speed_note = forge_line_time_s(wf, float(size_nm), line_time_s)
    if speed_note:
        logger.info("修针评估图限速:%s", speed_note)

    params: dict[str, Any] = {
        "center_x_m": float(x),
        "center_y_m": float(y),
        "size_m": float(size_nm) * 1e-9,
    }
    if pixels is not None:
        params["pixels"] = int(pixels)
    if line_time_s is not None:
        params["line_time_s"] = float(line_time_s)

    # 等待预算由像素数和每线时间推导，不能与扫描工作点脱节。
    # 信息齐全时取配置预算与 est×1.3+30 的较大者，与 ScanAt 的估算规则保持一致。
    # 工作点由下游档位表解析时，由 ScanAt 根据最终几何进一步确定预算。
    budget = float(wf.scan_timeout_s)
    if pixels is not None and line_time_s is not None:
        from mast.core.scan_policy import estimate_scan_seconds

        est = estimate_scan_seconds(pixels, line_time_s)
        if est > 0:
            derived = est * 1.3 + 30.0
            if derived > budget:
                logger.info(
                    "修针评估图:%s px × %s s/线 ⇒ 估计 %.0f s/帧,超过流程表的等待"
                    "上限 %.0f s,本次按几何派生为 %.0f s(限额的计价单位由别处决定"
                    "就必须派生)。", pixels, line_time_s, est, budget, derived)
                budget = derived
    params["wait_timeout_s"] = budget
    return params


def _scan_path(executor: GraphExecutor, save_id: str, latest_id: str) -> str:
    """刚扫的那张图的文件路径 —— 先看 SaveScan,再看 GetLatestScanFile。"""
    for step_id in (save_id, latest_id):
        d = _data(executor, step_id)
        p = d.get("path") or d.get("file_path") or d.get("scan_path")
        if p:
            return str(p)
    return ""

# 处理过的落点在本次运行共享清单中累计，同时记入地图以供后续调用查询。
_DIRTY_KEY = "dirty_spots"


def _dirty(executor: GraphExecutor) -> list[tuple[float, float]]:
    """这一跑已经弄脏的落点(pulse + poke,跨阶段共享)。"""
    pd = executor.progress.partial_data
    return [tuple(p) for p in (pd.get(_DIRTY_KEY) or [])]


def _spot_list(*groups: "Iterable[tuple[float, float]] | None") -> str:
    """落点 →  ``FindFlatRegion.exclude_used_spots`` 那个串(米,分号分隔)。

    几组合并、**去重**、保持顺序。坐标一律写成米的科学计数法:那个参数走的是
    ``si_quantity.parse_quantity(strict=False)``,``-4.79e-07`` 与 ``-479n``
    都收,而科学计数法不需要在这里挑前缀 —— 少一个能挑错的地方。

    ⚠️ **解析不了的块技能会如实报出来**(``_parse_excluded`` 的自述:静默跳过
    等于「一个都不排除」然后高高兴兴把针尖送回坏点),所以这里宁可格式笨一点。

    空表返回 ``""`` —— 与「不传这个键」同义,那正是这个参数的缺省。
    """
    seen: set[tuple[float, float]] = set()
    out: list[str] = []
    for g in groups:
        for p in (g or ()):
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            if x != x or y != y:            # NaN
                continue
            key = (round(x, 15), round(y, 15))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{x:.6e},{y:.6e}")
    return ";".join(out)


def _mark_dirty(executor: GraphExecutor, x: float, y: float,
                *, kind: str = "pulse", label: str = "") -> None:
    """将处理位置同时写入本次共享清单和持久地图，避免其他阶段或后续调用重访。"""
    pd = executor.progress.partial_data
    cur = [tuple(p) for p in (pd.get(_DIRTY_KEY) or [])]
    spot = (float(x), float(y))
    if spot in cur:
        # 同一个点重复标记 —— 地图那边也不必再写一条。
        return
    cur.append(spot)
    pd[_DIRTY_KEY] = [list(p) for p in cur]
    _log_damage_marker(executor, x, y, kind=kind, label=label)


def _log_damage_marker(executor: GraphExecutor, x: float, y: float,
                       *, kind: str, label: str = "") -> None:
    """往地图写一条损伤标记;写不成就记一笔,让报文说得出来。

    真正落库的那几行在 ``map_scope.record_damage_marker`` —— **和
    ``load_markers`` 同一个文件**,取 storage 的方式逐行相同。写进 A 而从 B 读,
    几何上等于没写,而症状是「它怎么又在同一个地方动手」,没有一处会报错。
    """
    from mast.core.map_scope import record_damage_marker

    # 出处取执行器自己的名字,**不写死字面量**。两个理由:
    #   · 这段代码 ForgeAuTip 与 PrepareNobleTip 都在用,写死会让一半标记说谎;
    #   · ``skill_name="X"`` 这个形状被 test_required_skills_covers_what_the_
    #     workflow_actually_calls 当成「流程调用了技能 X」来扫 —— 一个出处标签
    #     混进技能清单,会让那道自检去找一个根本不存在的依赖。
    row_id = record_damage_marker(
        x, y, kind=kind,
        label=label or ("脉冲坑" if kind == "pulse" else "修针坑"),
        skill_name=str(getattr(executor, "_composite_name", "") or ""))
    if row_id is None:
        _bump_marker_failure(executor, f"{kind}")


def _bump_marker_failure(executor: GraphExecutor, why: str) -> None:
    """地图写入失败必须报告，后续选点不能声称该位置已被排除。"""
    pd = executor.progress.partial_data
    pd["map_marker_failures"] = int(pd.get("map_marker_failures") or 0) + 1
    if not pd.get("map_marker_failure_why"):
        pd["map_marker_failure_why"] = why


#: 「换小一档就有」最多听几次建议。听不动就换地方 —— 无限缩窗会让落点越来越
#: 小,最后在一个比针尖还小的窗里「找到」一块平区,那不是平区,是噪声的一个局部。
_FLAT_SHRINK_TRIES = 3
#: 窗口不许缩到这个以下。10 nm 的簇图要落在窗里,窗比它还小就没有意义了。
_FLAT_WINDOW_FLOOR_M = 20e-9


def _finite_num(v) -> "float | None":
    """能拿来算/报的有限数,否则 None。``bool`` 排除(True 不是 1)。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _say(kind: str, /, **data) -> None:
    """发送旁白且不向流程抛异常；kind 为位置参数，以免与旁白负载中的同名字段冲突。"""
    try:
        from mast.chat.narration import narrate

        narrate(kind, **data)
    except Exception:  # noqa: BLE001
        pass


def _decide(entry: dict, kind: str, **facts) -> None:
    """扎针决策的报文与旁白在同一处生成，展示下一步动作及其原因。"""
    entry["decision"] = kind
    _say("poke_decision", kind=kind, **facts)


#: 已经搜过、确认没有可用平区的那些帧心(跨 refill 累计)。
_SEARCHED_KEY = "searched_frames"


def _step_split_look(executor: GraphExecutor, *, scan_path: str,
                     flat: "dict | None", frame_nm: "float | None" = None) -> dict:
    """在这张大图上问:台阶被劈开了吗。**顺带把分析图发进旁白。**

    判据在 ``vision.double_tip.step_splitting`` —— 去斜 → 高度直方图 → 台面能级
    → 相邻台面差是不是单原子台阶的整数倍。尺子是 Au(111) 的台阶高度,
    一个**有物理零点**的量(本机读数偏约 15%,所以走 ``instrument_profile``
    的 ``au_step_pm``,不是写死 235.5)。

    永不抛:判据坏掉只丢一条旁白,绝不弄坏正在跑的实验。
    """
    out: dict = {"verdict": "undecidable"}
    try:
        import numpy as np

        from mast.io.nanonis_files import read_sxm
        from mast.vision.double_tip import step_splitting

        ch = (read_sxm(str(scan_path)).get("channels") or {}).get("Z")
        if not ch:
            return out
        out = step_splitting(np.asarray(ch.get("forward"), dtype=float),
                             frame_m=(float(frame_nm) * 1e-9
                                      if frame_nm else None))
    except Exception as exc:  # noqa: BLE001
        logger.debug("台阶劈裂判据没跑成: %s", exc)
        return out

    # 三态判读都提供旁白，区分未评估、无法判定和已评估。
    is_split = str(out.get("verdict") or "") == "split"

    # 配图只给劈裂那一态画 —— 画一张台面分析图要重读 .sxm 再渲染,
    # 而「干净」那一张图上没有任何东西需要看。这是成本,不是信息量的取舍。
    image = None
    if is_split:
        try:
            from mast.vision.terrace_panel import render_terrace_panel

            png = render_terrace_panel(scan_path, out, flat)
            if png:
                image = {"src": png, "origin": "milestone_png"}
        except Exception as exc:  # noqa: BLE001
            logger.debug("台面分析图没画成: %s", exc)

    try:
        from mast.chat.narration import narrate

        narrate("step_split", image=image, result=out)
    except Exception:  # noqa: BLE001
        pass
    return out


def _remember_searched(executor: GraphExecutor, x: float, y: float) -> None:
    """记下「这一整片我搜过了,没有台面」。

    与 ``_mark_dirty`` 不是一回事:那记的是**弄脏了**(要避让,而且要进地图),
    这记的是**搜过了**(表面还干净,只是这儿没有够大的平台)。
    两者混在一起会让干净的表面被当成脏的。
    """
    pd = executor.progress.partial_data
    cur = [tuple(t) for t in (pd.get(_SEARCHED_KEY) or [])]
    pt = (float(x), float(y))
    if pt not in cur:
        cur.append(pt)
        pd[_SEARCHED_KEY] = [list(t) for t in cur]


def _move_to(executor: GraphExecutor, *, step_prefix: str,
             x: float, y: float) -> Iterator[CompositeStep]:
    """将针尖移动到指定坐标，保持真实位置与后续处理中心一致。"""
    yield CompositeStep(
        step_id=f"{step_prefix}:pre_move_feedback", skill_name="ZControllerOnOff",
        params={"enable": True},
        optional=True, checkpoint_after=False, tags=("poke", "move"))
    yield CompositeStep(
        step_id=f"{step_prefix}:move", skill_name="MoveToXY",
        params={"x_m": float(x), "y_m": float(y), "wait": True},
        optional=False, checkpoint_after=False, tags=("poke", "move"))


def _relocate(executor: GraphExecutor, *, step_prefix: str, purpose: str,
              used: list[tuple[float, float]],
              avoid: "list[tuple[float, float, float]] | None" = None,
              ) -> Iterator[CompositeStep]:
    """查找最近干净区域并避开累计处理位置；返回坐标及地图可用性，或 None。"""
    yield CompositeStep(
        step_id=f"{step_prefix}:find_spot",
        skill_name="FindCleanSpot",
        # 要避开整片搜过的区域时多要几个候选 —— 只要 4 个很可能全落在里面。
        params={"purpose": purpose, "exclude_spots": _spots_str(used),
                "count": 12 if avoid else 4},
        # optional=True 是有意的:「这片表面没有干净的地方了」不是一个错误,是流程
        # 必须自己处理的**信息** —— 要报告出去、要把反馈开回来、要建议粗动换区。
        # 标成必需的话 executor 会在这一步直接中止整个计划,而生成器再也拿不回控
        # 制权:收尾不跑,报告里连「为什么停」都没有。
        optional=True, checkpoint_after=False,
        tags=("relocate", purpose),
    )
    if not _ok(executor, f"{step_prefix}:find_spot"):
        return None
    spot = _data(executor, f"{step_prefix}:find_spot")
    x, y = spot.get("x_m"), spot.get("y_m")
    if x is None or y is None:
        return None

    # ── 搜过的整片区域:从候选里挑第一个不在里面的 ─────────────────────
    if avoid:
        def _clear(px, py):
            return all((px - ax) ** 2 + (py - ay) ** 2 >= ar ** 2
                       for ax, ay, ar in avoid)

        if not _clear(float(x), float(y)):
            picked = None
            for c in (spot.get("candidates") or []):
                cx_, cy_ = c.get("x_m"), c.get("y_m")
                if cx_ is None or cy_ is None:
                    continue
                if _clear(float(cx_), float(cy_)):
                    picked = (float(cx_), float(cy_))
                    break
            if picked is not None:
                x, y = picked
            # 一个都不剩时**沿用首选并说出来** —— 停下来是更坏的处置:
            # 「这片都搜过了」不等于「不能在这儿动手」,而无限换区更贵。
            # 但它必须留痕,否则下一个人会以为 avoid 生效了。
            else:
                pd = executor.progress.partial_data
                pd["searched_area_exhausted"] = (
                    int(pd.get("searched_area_exhausted") or 0) + 1)

    if float(spot.get("distance_m") or 0.0) > 0.0:
        # 移动入口负责建立 z_controller_on 前置。
        # 尝试开启反馈为 optional，失败需报告，再由 MoveToXY 给出准确的前置错误。
        # ZCtrl_StatusGet 与 ZCtrl_OnOffGet 表达不同状态；开启成功不能替代状态前置检查。
        # 不要仅由相邻动作推断控制器状态，需在相关操作后读取对应寄存器。
        yield CompositeStep(
            step_id=f"{step_prefix}:pre_move_feedback",
            skill_name="ZControllerOnOff",
            params={"enable": True},
            optional=True, checkpoint_after=False,
            tags=("relocate", "feedback"),
        )
        # 尝试开启反馈失败时必须报告原因，再由后续移动步骤检查准确的前置条件。
        if not _ok(executor, f"{step_prefix}:pre_move_feedback"):
            note = (f"{step_prefix}: 移动前尝试打开 Z 反馈(ZControllerOnOff enable=True)"
                    "**这一步本身失败了** —— 接下来 MoveToXY 若报 z_controller_on 前置"
                    "不满足,原因在这里,不是「谁把它关了」。")
            logger.warning(note)
            executor.set_partial(f"{step_prefix}:feedback_restore_failed", note)
        yield CompositeStep(
            step_id=f"{step_prefix}:move",
            skill_name="MoveToXY",
            params={"x_m": float(x), "y_m": float(y), "wait": True},
            optional=False, checkpoint_after=False,
            tags=("relocate", purpose),
        )
        if not _ok(executor, f"{step_prefix}:move"):
            return None
    return float(x), float(y), bool(spot.get("map_known", False))


# ── A 阶段:电脉冲大修 ──────────────────────────────────────────────────────

def pulse_phase(executor: GraphExecutor, wf: NobleTipWorkflow, *,
                prefix: str = "A", pulse_v: float | None = None,
                voltage_sequence: "tuple[float, ...] | None" = None,
                set_junction: bool = True) -> Iterator[dict]:
    """打脉冲直到 Z 向上跳出几十 nm,或者预算用完。

    极性策略（定案）:先用一种极性;同一极性连打 ``pulses_per_polarity`` 发都
    没效果就翻过来再试。极性与连败计数存进 ``partial_data``,所以中途被打断再续
    跑不会从头数。

    ``voltage_sequence`` 给定时(扫描验证失败后的 7/5/3 V 递减),按序换电压而不是
    换极性 —— 那是用户在大修与精修之间的过渡手法。
    """
    pd = executor.progress.partial_data
    log: list[dict[str, Any]] = list(pd.get(f"{prefix}:log", []))
    used: list[tuple[float, float]] = [tuple(p) for p in pd.get(f"{prefix}:used", [])]
    polarity = int(pd.get(f"{prefix}:polarity", 1))
    streak = int(pd.get(f"{prefix}:streak", 0))
    fired = int(pd.get(f"{prefix}:fired", 0))
    _ls = pd.get(f"{prefix}:last_spot")
    last_spot: "tuple[float, float] | None" = (
        (float(_ls[0]), float(_ls[1])) if isinstance(_ls, (list, tuple)) and len(_ls) == 2
        else None)

    base_v = abs(float(pulse_v if pulse_v is not None else wf.pulse_v))
    seq = tuple(voltage_sequence or ())
    out: dict[str, Any] = {"phase": "pulse", "satisfied": False, "fired": fired,
                           "log": log, "polarity_flips": 0, "map_known": True}

    if set_junction:
        # 低偏压 + 大电流 = 针尖离表面很近,脉冲才作用得上(用户:20~100 mV,到 1 nA)。
        yield CompositeStep(
            step_id=f"{prefix}:junction_bias", skill_name="SetBias",
            params={"bias_v": wf.junction_bias_v},
            optional=False, checkpoint_after=False, tags=("junction",))
        yield CompositeStep(
            step_id=f"{prefix}:junction_setpoint", skill_name="SetSetpoint",
            params={"setpoint_a": wf.junction_setpoint_a},
            optional=False, checkpoint_after=False, tags=("junction",))
        if not (_ok(executor, f"{prefix}:junction_bias")
                and _ok(executor, f"{prefix}:junction_setpoint")):
            out["reason"] = "无法建立修针所需的结条件(低偏压/大电流)"
            return out

    budget = int(wf.pulse_budget)
    while fired < budget:
        shot = fired + 1
        sp = f"{prefix}:shot{shot}"

        # 脉冲改变局部表面，即使未达到修针目标也要记录处理位置并换点。
        same_spot = int(pd.get(f"{prefix}:same_spot", 0))
        keep_spot = (last_spot is not None
                     and same_spot < int(wf.pulse_same_spot_budget))
        if keep_spot:
            x, y = last_spot
            map_known = True
            spot = (x, y, True)
        else:
            # 排除表使用全流程累计的 _dirty()；每轮重置的 used 无法防止重访旧处理点。
            spot = yield from _relocate(executor, step_prefix=sp, purpose="pulse",
                                        used=_dirty(executor))
        if spot is None:
            out["reason"] = ("这片表面已经没有可用的落点了 —— 该粗动换区"
                             "(RelocateCoarseXY),或者换一块样品区域。")
            out["surface_spent"] = True
            break
        x, y, map_known = spot
        # 这一发**还没打**,但落点已经定了 —— 先记进脏点表 + 地图。
        # 记在打之前而不是打之后:脉冲若失败,表面照样被弄脏了
        #(「这一步没成」不等于「这里还干净」),而记在后面会漏掉那一种。
        #
        # ``kind="pulse"`` 决定避让半径(500 nm)。写死而不是从 purpose 推:
        # 传错 kind 不会报错,只会让避让圈小一个量级,而那种错发现不了。
        _mark_dirty(executor, x, y, kind="pulse")
        out["map_known"] = out["map_known"] and map_known

        volts = (seq[min(fired, len(seq) - 1)] if seq else base_v)
        bias = polarity * abs(volts)
        yield CompositeStep(
            step_id=f"{sp}:pulse",
            skill_name="BiasPulseWithReadback",
            params={
                "bias_v": bias,
                "width_s": wf.pulse_width_s,
                # 移动之后的基线采集同时充当 settle:一次横向移动会重新激起蠕变,
                # 而 pre-roll 本来就要采一段稳定的 z1。
                "pre_roll_s": max(0.1, float(wf.move_settle_s)),
                "post_roll_s": 0.3,
                "step_tol_nm": 0.5,
            },
            optional=False, checkpoint_after=True,
            tags=("pulse", f"shot={shot}"),
        )
        fired += 1
        used.append((x, y))
        # 记住这一发落在哪、在这个点上已经打了几发 —— 豁免靠它。
        same_spot = (same_spot + 1) if keep_spot else 1
        last_spot = (x, y)
        pd[f"{prefix}:same_spot"] = same_spot
        pd[f"{prefix}:last_spot"] = [x, y]

        if not _ok(executor, f"{sp}:pulse"):
            entry = {"shot": shot, "x_m": x, "y_m": y, "bias_v": bias,
                     "outcome": "failed"}
            log.append(entry)
            out["reason"] = "脉冲执行失败 —— 停下来让人看,不要继续打。"
            _stash(executor, prefix, log, used, polarity, streak, fired)
            break

        step = _data(executor, f"{sp}:pulse").get("step", {}) or {}
        # ⭐ ``direction`` 是**四态**,产生方(``bias_pulse_readback.py:263``)分得很清楚:
        #     up / down          真的测到了 Z 跳变,方向已知
        #     none               真的测到了,是零 —— 脉冲没打动针尖
        #     insufficient_data  **判不了**(采样不够/读不到)
        #
        # 2026-08-14 之前这里是 ``direction == "up"``,于是 **down / none /
        # insufficient_data 三个全落进 `no_effect`** —— 上游分清楚的四态,
        # 被下游一个 `==` 压成布尔。和当年白名单把 `"measured"` 判成不合格
        # (见 ``forge_au_tip._SHARPNESS_VERDICT_KIND``)是同一个形状。
        #
        # ``delta_m`` 也不再 ``or 0.0``:读不到时它是 None,折成 0.0 就等于说
        # 「测到了,是零」。**「读不到」不是一个值**(2026-08-13/14 一天出现五次的
        # 那族缺陷),所以这里保留 None,由下面的分支自己表态。
        direction = str(step.get("direction") or "insufficient_data")
        raw_dz = step.get("delta_m")
        dz_nm = (float(raw_dz) * 1e9) if raw_dz is not None else None
        entry = {"shot": shot, "x_m": x, "y_m": y, "bias_v": bias,
                 "direction": direction, "dz_nm": dz_nm,
                 # 米也留一份 —— 旁白那一侧全线用 SI 基本单位(``before_m`` /
                 # ``x_m`` / ``delta_m``),而 ``dz_nm`` 这个名字说的是**输出**
                 # 单位不是输入单位。2026-08-18 曾按名字喂入纳米值,
                 # 一发 12.4 nm 的跳变被念成 12400000000.00 nm。
                 "dz_m": (float(raw_dz) if raw_dz is not None else None)}
        log.append(entry)

        # 「A 的判据可以是 down 也行。」
        # 判的是**针尖被改变了没有**,不是改变的方向 —— 向上向下都算脉冲起了作用。
        # ⚠️ 阈值 ``pulse_success_dz_nm`` 原本是照「向上几十 nm」定的下沿,
        # down 方向沿用同一个数**没有数据支持**(up/down 的幅度分布可能不同)。
        if (direction in ("up", "down") and dz_nm is not None
                and abs(dz_nm) >= wf.pulse_success_dz_nm):
            entry["outcome"] = "satisfied"
            out["satisfied"] = True
            out["last_dz_nm"] = dz_nm
            out["last_direction"] = direction
            _stash(executor, prefix, log, used, polarity, streak, fired)
            break

        # 「读不到」单独记账 —— 控制流和 no_effect 相同(都是再打一发),但**报告里
        # 必须分得开**:20 发全是 `no_effect` 说的是「脉冲打不动这根针」,
        # 20 发全是 `unreadable` 说的是「Z 读回来有问题,去查信号链」。
        # 两句话指向完全不同的下一步,混在一起就都问不出来了。
        if direction == "insufficient_data" or dz_nm is None:
            entry["outcome"] = "unreadable"
            out["unreadable"] = int(out.get("unreadable") or 0) + 1
        else:
            entry["outcome"] = "no_effect"
        # 攒满同点预算 ⇒ 下一发必须换地方(清掉 last_spot,下一轮走 _relocate)。
        if same_spot >= int(wf.pulse_same_spot_budget):
            last_spot = None
            pd[f"{prefix}:same_spot"] = 0
            pd[f"{prefix}:last_spot"] = None
        streak += 1
        # 递减序列跑完就停 —— 那是一串定好的电压,不是「打到满意为止」。
        if seq and fired >= len(seq):
            out["reason"] = f"递减脉冲序列 {seq} 已打完"
            _stash(executor, prefix, log, used, polarity, streak, fired)
            break
        if not seq and streak >= int(wf.pulses_per_polarity):
            polarity = -polarity
            streak = 0
            out["polarity_flips"] = int(out["polarity_flips"]) + 1
            entry["polarity_flipped_to"] = polarity
        _stash(executor, prefix, log, used, polarity, streak, fired)

    if not out["satisfied"] and "reason" not in out:
        # 「读不到」的发数要说出来 —— 见上面那段注释:20 发 no_effect 和
        # 20 发 unreadable 指向完全不同的下一步。
        unread = int(out.get("unreadable") or 0)
        tail = (f";其中 **{unread} 发读不到 Z 跳变**(判不了,不等于没效果 —— "
                "先查 Z 读回和信号链,别急着加大脉冲)" if unread else "")
        out["reason"] = (f"打满 {fired} 发(预算 {budget})仍未出现 "
                         f"±{wf.pulse_success_dz_nm:.0f} nm 的 Z 跳变{tail}")
    # 没出现过也要有这个键:缺字段和 0 在报告里读起来一样,但一个是「没统计」
    # 一个是「统计了,是零」。
    out["unreadable"] = int(out.get("unreadable") or 0)
    out["fired"] = fired
    out["log"] = log
    out["used_spots"] = used
    # 这一相的**结论**从来没进过旁白 —— 每一发的 Z 跳变有 ``pulse_readback``,
    # 但「这一批到底修没修动针尖」只在报文里。而那才是用户盯着屏幕要的那句。
    _say("pulse_result", satisfied=bool(out.get("satisfied")), fired=fired,
         dz_m=_finite_num((log[-1] or {}).get("dz_m")) if log else None,
         reason=str(out.get("reason") or ""))
    return out


def _stash(executor: GraphExecutor, prefix: str, log, used, polarity, streak,
           fired) -> None:
    """把极性状态机存进 partial_data —— 断点续跑不从头数。"""
    executor.set_partial(f"{prefix}:log", list(log))
    executor.set_partial(f"{prefix}:used", [list(p) for p in used])
    executor.set_partial(f"{prefix}:polarity", int(polarity))
    executor.set_partial(f"{prefix}:streak", int(streak))
    executor.set_partial(f"{prefix}:fired", int(fired))


# ── B 阶段:扫图验证正反扫描线 ──────────────────────────────────────────────

def verify_phase(executor: GraphExecutor, wf: NobleTipWorkflow, *,
                 prefix: str = "B") -> Iterator[dict]:
    """回正常成像条件,扫一张图看 z-forward 与 z-backward 重不重合。

    三态,不是两态:相似度读不出来时是 ``inconclusive``,既不能当通过也不能当不通过
    —— 当成不通过会让流程接着去打脉冲,而根本原因可能只是没读到线数据。

    ## 2026-08-18 补上的两件事

    **一、这一相从来不说话。** 它是修针环的第一道判据(要求:正反扫描重合
    是基本判据,团簇圆度排在其后),而屏幕上只看得见「扫了一张图」,
    然后针尖要么被放行、要么挨一批脉冲 —— 中间那个理由一个字都没有。
    现在每一个出口都发一条 ``fwd_bwd_result``。

    **二、三态只在最后一个出口成立,前面三个出口全是两态。**
    「无法回到正常成像条件」/「找不到干净区域」/「扫描验证没能完成」这三条路
    以前都返回 ``passed=False, inconclusive=False`` —— 而下游只认这两个字段:
    ``passed`` 假、``inconclusive`` 假 **就是「针尖不好」,直接打脉冲**。
    这三种情况的共同点恰恰是**针尖一次都没被测过**。
    (本仓 `unknown_is_not_an_answer` 的又一次:「读不到」被当成了「出故障」。)
    现在它们都走 ``inconclusive`` —— 那条路已经存在,而且外环对它的处理
    (换个站点重新量、连续几站没测到就报 ``spinning`` 停手)正是为这种情况写的。
    """
    out: dict[str, Any] = {"phase": "verify", "passed": False,
                           "inconclusive": False}

    def _unmeasured(reason: str) -> dict:
        """针尖**一次都没被测到**就退出:这是「判不了」,不是「不合格」。"""
        out["inconclusive"] = True
        out["reason"] = reason
        _say("fwd_bwd_result", inconclusive=True, reason=reason)
        return out

    yield CompositeStep(
        step_id=f"{prefix}:bias", skill_name="SetBias",
        params={"bias_v": wf.verify_bias_v},
        optional=False, checkpoint_after=False, tags=("verify",))
    yield CompositeStep(
        step_id=f"{prefix}:setpoint", skill_name="SetSetpoint",
        params={"setpoint_a": wf.verify_setpoint_a},
        optional=False, checkpoint_after=False, tags=("verify",))
    if not (_ok(executor, f"{prefix}:bias") and _ok(executor, f"{prefix}:setpoint")):
        return _unmeasured("回不到正常成像条件,这一帧没扫成 —— 针尖没被测到")

    # 找一块干净地方扫 —— 在刚打过脉冲的坑上验证针尖,验的是坑不是针尖。
    # 本轮已弄脏的落点全部排除 —— 传空表就是「在坑上判针尖」的那半个原因。
    spot = yield from _relocate(executor, step_prefix=prefix, purpose="pulse",
                                used=_dirty(executor))
    if spot is None:
        # 「这片表面没有干净落点了」在本仓有一个既定的名字:``surface_spent``。
        # ``pulse_phase`` / ``poke_phase`` / 救援相 —— 四处对 ``_relocate`` 返回
        # None 都是这么归类的,只有 verify 这一处没有,于是同一件事在这里叫
        # 「针尖不好」(passed=False 且 inconclusive=False)。**同一个动作的第 N 份
        # 实现里,往往只有带事故注释的那几份是对的。**
        #
        # 两个位一起打:``surface_spent`` 说**为什么**停(该换区,不是该打脉冲),
        # ``inconclusive`` 说**针尖没被测过**(所以后面任何关于针尖的话都不成立)。
        out["surface_spent"] = True
        return _unmeasured("这片表面没有干净落点了,验证图没地方扫 —— 针尖没被测到")
    x, y, map_known = spot
    out["map_known"] = map_known

    # 验证帧走 PreScanCheck 而不是 ScanAt,所以它**够不到 ``scan_at_params``**
    # 那道限速 —— 五张评估图里唯一的例外。今天它正好等于上限(0.586 s 就是上限
    # 的来源),所以这一行现在一个数都不改;接它是为了别的两种情况:用户把
    # ``forge_scan_nm`` 调大(视野变大而线时不变 ⇒ 速度跟着涨,回退图就是这么栽
    # 的),或者哪天有人改动 ``forge_verify_line_time_s``。
    # **「今天不会触发」不是不接的理由** —— 一道只在别人不犯错时才正确的闸门,
    # 与没有这道闸门是同一件事。
    # 见下:扫完之后要在同一张图上问一句「台阶被劈开了吗」。
    verify_line_time_s, verify_speed_note = forge_line_time_s(
        wf, float(wf.verify_scan_nm), wf.verify_line_time_s)
    if verify_speed_note:
        logger.info("修针验证帧限速:%s", verify_speed_note)
        out["speed_note"] = verify_speed_note

    yield CompositeStep(
        step_id=f"{prefix}:prescan",
        skill_name="PreScanCheck",
        params={"center_x_m": x, "center_y_m": y,
                "width_m": wf.verify_scan_nm * 1e-9,
                "quality_threshold": wf.fwdbwd_threshold,
                # 验证帧自己的行数 —— **唯一能缩短一帧 verify 的旋钮**
                # (帧时 = 行数 × 每线时间 × 2,视野不在式子里)。
                # None ⇒ 不下发这个键 ⇒ PreScanCheck 继承现场(历史行为)。
                **({"pixels": int(wf.verify_pixels)} if wf.verify_pixels else {}),
                # 省略每线时间时不传键，使 PreScanCheck 能按尺度解析；schema 不得替省略值填入显式默认。
                **({"line_time_s": float(verify_line_time_s)}
                   if verify_line_time_s else {}),
                # PreScanCheck **从不设线数**(它的 ConfigureScan 走
                # ``Scan_BufferSet(ch, 0, 0)``,后两位恒 0 = 保持现值),所以这一步
                # 扫几行由**上一次扫描**决定 —— 而这条流程自己刚把它设成了 256。
                #
                # ⚠️ 这里传的是**下限**,不再是预算本身(2026-08-12)。
                # 原注释写着「PreScanCheck 的出厂预算是 15 s,设分辨率的人要为它
                # 负责」—— 那句话 2026-08-10 起就是假的:那天预算改成了按真实行数
                # 派生。而这个显式值当时会把派生**整个关掉**(旧实现是
                # ``if wait_timeout_s is None``),于是每线时间从
                # 0.1 s 改到档位表的 1.0 s(针尖速度 488 nm/s → 50 nm/s,
                # 488 nm/s 会刮伤针尖)之后,帧时 52 s → 512 s,这个 300 s 会让
                # **每一次 verify 都超时**。
                #
                # 修法做在 PreScanCheck 那一侧(``max(显式, 派生)``),不是在这里
                # 把 300 改大 —— 后者只是把同一个耦合挪个地方,而且下一个把线时
                # 再调慢的人会再踩一次。留着这个值是因为它仍有意义:用户愿意
                # 等更久时,``scan_timeout_s`` 照样起作用。
                "wait_timeout_s": wf.scan_timeout_s},
        optional=False, checkpoint_after=True,
        tags=("verify", "fwd_bwd"),
    )
    if not _ok(executor, f"{prefix}:prescan"):
        # 失败原文比「扫描验证没能完成」有用得多 —— 后者对用户没有信息量。
        why = _err(executor, f"{prefix}:prescan")
        return _unmeasured("扫描验证没能完成" + (f":{why}" if why else "")
                           + " —— 针尖没被测到")

    # ── 把这一帧留下来给 level 相用(2026-08-14 合并)────────────────────
    #
    # 要求:两张图合并成一张。在此之前 verify 和 level 在同一片区域各扫一张
    # (20 nm + 100 nm),而两张都只是「拍一张看看」。现在 verify 这张就是 100 nm,
    # 存下来交给 level 找台阶、挑平区 —— 一圈省一张图、省 5 分钟。
    #
    # 两步都是 ``optional=True``:存不下来只是**没得复用**(level 会自己再扫一张,
    # 回到合并前的行为),不该让一次正常的 verify 失败。
    yield CompositeStep(
        step_id=f"{prefix}:save", skill_name="SaveScan", params={},
        optional=True, checkpoint_after=False, tags=("verify",))
    yield CompositeStep(
        step_id=f"{prefix}:latest", skill_name="GetLatestScanFile", params={},
        optional=True, checkpoint_after=False, tags=("verify",))
    out["scan_path"] = _scan_path(executor, f"{prefix}:save", f"{prefix}:latest")

    # 这张验证图就是 100 nm(合并之后),够格问一句「台阶被劈开了吗」——
    # 而且这是整条流程里**最早**能问到它的地方。C 相复用这一帧时不会再问一次
    # (那边的 split look 在「自己扫」那一支里),所以这里问不会说两遍。
    if out["scan_path"]:
        sp = _step_split_look(executor, scan_path=str(out["scan_path"]),
                              flat=None, frame_nm=float(wf.verify_scan_nm))
        out["split_tip"] = (sp.get("verdict") == "split")
    else:
        # 存不下来 ⇒ 这一问**根本没被问过**,而下面那道一票否决因此不生效。
        #
        # 沉默在这里最危险:``step_split`` 那张卡片不会出现,用户看到的是
        # 「重合度过了」然后放行 —— 和「问了,是单尖」长得一模一样。
        # 说出来,而且说清是**没问**不是「没有」。
        out["split_tip_unasked"] = True
        _say("step_split", result={
            "verdict": "undecidable",
            # 模板自己会说「多针尖这一项判不了——」,这里只给**原因**,别重复主语。
            "reason": "验证图未存盘、取不到文件,本轮无法判读"})

    d = _data(executor, f"{prefix}:prescan")
    sim = d.get("similarity")
    tip_ready = d.get("tip_ready")
    out["similarity"] = sim
    out["scan_center"] = [x, y]
    # ⚠️ 判**三态**,而且判的是 ``tip_ready`` 本身,不是从 ``sim`` 反推。
    #
    # 原来这里只看 ``sim is None``,然后在下面写 ``bool(d.get("tip_ready", False))``。
    # 那条**今天**是对的 —— 但只因为「``tip_ready`` 为 None 当且仅当 ``sim`` 为
    # None」这条**巧合耦合**,不是构造保证。任何一次让 PreScanCheck 在有 ``sim``
    # 的情况下也返回 ``tip_ready=None`` 的改动,都会让 ``bool(None) → False`` 悄悄
    # 可达 —— 而 ``False`` 在这条流程里是**打脉冲**,``None`` 才是换地方重测。
    # 一个「判不了」被静默翻译成「针尖坏」,后果是拿脉冲去打一根可能好好的针。
    # (2026-08-10:起伏弃权门上线,弃权路径从此是常走的路,不再是边角。)
    if sim is None or tip_ready is None:
        # fail-open 在这里是错的方向:判不了不等于针尖好。
        out["inconclusive"] = True
        # 「为什么判不了」有好几种,指向的下一步不同 —— 把 PreScanCheck 给的那句
        # 原样带出来,别再统一说成「读不到」(起伏不足时数据读得好好的)。
        out["reason"] = (
            d.get("abstain_reason")
            or d.get("read_failure")
            or d.get("unusable_reason")
            or "读不到正反扫描线数据,判不了针尖好坏 —— 这不是「针尖没问题」。")
        _say("fwd_bwd_result", inconclusive=True, reason=out["reason"],
             threshold=float(wf.fwdbwd_threshold))
        return out

    # 正反扫重合与多针尖形貌是不同判据。
    # 只有明确 split 才触发此分支，undecidable 不作否定结论；阈值需在目标仪器验证。
    fwd_bwd_ok = bool(tip_ready)
    split = bool(out.get("split_tip"))
    out["passed"] = fwd_bwd_ok and not split
    out["vetoed_by_split_tip"] = fwd_bwd_ok and split
    out["reason"] = (f"正反扫描线相似度 {float(sim):.3f}"
                     f"（阈值 {wf.fwdbwd_threshold:.2f}）"
                     + ("；但大图上台阶被画了两遍（多针尖）—— 一票否决"
                        if out["vetoed_by_split_tip"] else ""))
    _say("fwd_bwd_result", passed=out["passed"], inconclusive=False,
         fwd_bwd_ok=fwd_bwd_ok, split_tip=split,
         similarity=float(sim), threshold=float(wf.fwdbwd_threshold))
    return out


# ── C 阶段:找台阶 + 挑平区调平 ─────────────────────────────────────────────

def level_phase(executor: GraphExecutor, wf: NobleTipWorkflow, *,
                prefix: str = "C", reuse: "tuple | None" = None) -> Iterator[dict]:
    """在一张大图上找台阶,并在其中挑一块无台阶的平区调平。

    台阶位置记进结果:验收时要回到台阶上看边缘够不够陡(近阶跃),而调平必须避开
    台阶 —— 台阶主导时倾斜拟合量的是台阶包络不是表面。

    ``reuse`` = verify 相刚扫的那张图 ``(path, x, y)``。给了就**不再自己扫**
    (2026-08-14 按要求合并)。verify 那张现在就是 100 nm,
    与本相原本要扫的是同一种图。存不下来时 verify 传 ``None``,这里退回自己扫
    —— 合并前的行为,不会因为省图而丢功能。
    """
    out: dict[str, Any] = {"phase": "level", "leveled": False}

    if reuse and reuse[0]:
        path = str(reuse[0])
        x, y = float(reuse[1]), float(reuse[2])
        out["reused_verify_scan"] = True
    else:
        out["reused_verify_scan"] = False
        # 本轮已弄脏的落点全部排除 —— 传空表就是「在坑上判针尖」的那半个原因。
        spot = yield from _relocate(executor, step_prefix=prefix,
                                    purpose="pulse", used=_dirty(executor))
        if spot is None:
            out["reason"] = "找不到干净区域找台阶"
            return out
        x, y, _known = spot

        yield CompositeStep(
            step_id=f"{prefix}:scan", skill_name="ScanAt",
            params=scan_at_params(wf, x, y, size_nm=wf.step_scan_nm,
                                  pixels=wf.step_pixels,
                                  line_time_s=wf.step_line_time_s,
                                  # 上面几行刚 _relocate 过 —— 这是默认路径。
                                  origin="clean_spot"),
            optional=False, checkpoint_after=False, tags=("level", "wide"))
        if not _ok(executor, f"{prefix}:scan"):
            out["reason"] = "台阶扫描没能完成"
            return out
        yield CompositeStep(
            step_id=f"{prefix}:save", skill_name="SaveScan", params={},
            optional=True, checkpoint_after=False, tags=("level",))
        yield CompositeStep(
            step_id=f"{prefix}:latest", skill_name="GetLatestScanFile", params={},
            optional=True, checkpoint_after=True, tags=("level",))
        path = _scan_path(executor, f"{prefix}:save", f"{prefix}:latest")
        # 每一张大图都问一句「台阶被劈开了吗」(# 「所有的图都应该有自动检测」)。只在**劈裂**时出声,见 _step_split_look。
        if path:
            sp = _step_split_look(executor, scan_path=path, flat=None,
                                  frame_nm=float(wf.step_scan_nm))
            if sp.get("verdict") == "split":
                out["split_tip"] = True
                out["needs_pulse"] = True

    out["wide_scan_path"] = path
    out["wide_scan_center"] = [x, y]
    if not path:
        out["reason"] = "扫完了但拿不到文件路径,后面的分析无从做起"
        return out

    # 台阶是否主导 —— 同时给出 AutoTilt 需要的 surface_rms_m。
    yield CompositeStep(
        step_id=f"{prefix}:steps", skill_name="AnalyzeFrameTilt",
        params={"scan_path": path, "check_steps": True},
        optional=True, checkpoint_after=False, tags=("level", "steps"))
    steps_d = _data(executor, f"{prefix}:steps")

    # ── 找不到台阶 → 退到更大的视野再找一次 ─────────────────────────────
    #
    # 「如果找不到台阶,那就换 200 nm 接着找,
    # 不要把小起伏硬算作台阶。」
    #
    # 只退**一次**,而且回退图**只用来找台阶**:正反扫在 100 nm 那张上已经判过
    # (回退图不重判),验收图仍固定 ``step_scan_nm``(尺度浮动会让锐度阈值失去
    # 意义 —— 同一根针尖 100 nm 上 2.6 nm、200 nm 上 5.5 nm)。
    #
    # ⚠️ **已知缺口**:回退在**同一个中心**扫更大的视野,而 ``AnalyzeFrameTilt``
    # 只回答「这一帧台阶主不主导」,**不给台阶的位置**。所以在 200 nm 上找到台阶
    # 之后,验收图仍在同一中心扫 100 nm —— 那个 100 nm 里未必有台阶。
    # 后果是验收报 ``no_step`` ⇒ ``undecidable`` ⇒ **不拦流程**(安全,只是白扫
    # 一张)。要真正修好,需要一个能给出台阶**坐标**的判据。
    # ⚠️ 直接读属性,**不用 getattr** —— ``test_every_workflow_field_has_a_consumer``
    # 是靠静态扫描找消费方的,一个 getattr 就把这个字段变成了它眼里的「死配置」。
    # 那道闸门存在的理由正是「读起来像可调项,而调它没有任何效果」,
    # 一行防御性写法就能把这道闸门绕过去。
    fallback_nm = wf.step_fallback_nm
    if (fallback_nm and not steps_d.get("step_dominated")
            and float(fallback_nm) > float(wf.step_scan_nm)):
        out["step_fallback_tried"] = float(fallback_nm)
        # ⚠️ **线时必须按视野放大,否则针尖速度跟着视野一起翻倍。**
        #
        # 针尖速度 = 视野 ÷ 每线时间。回退图的视野是 100 → 200 nm,若沿用
        # ``step_line_time_s``(0.15 s),速度就从 667 nm/s 变成 **1333 nm/s** ——
        # 比已知会刮伤针尖的 488 nm/s 快 2.7 倍。
        # (这个洞是写时间账那一节时发现的:代码本身读不出问题,是把
        #  「视野 ÷ 线时」算出来才看见。)
        #
        # 按比例放大 ⇒ 针尖速度与 100 nm 那张**完全相同**,代价只是帧时翻倍。
        # 帧时 = 行数 × 线时 × 2,视野不在式子里 —— 所以这里买到的确实是安全,
        # 不是别的东西。
        #
        # ⚠️ **但用户逐字说过的数不缩放** —— 规矩连同公式一起住在
        # ``forge_fallback_line_time_s``(2026-08-15 抽出去:报告那一侧也要说
        # 「回退图跑多快」,而抄一份就会把这条分支抄丢,报告里的数与真正下发的数
        # 从此不是一个)。这里只负责把结果说出来。
        fb_line_time, operator_pinned = forge_fallback_line_time_s(wf)
        if fb_line_time and not operator_pinned:
            out["step_fallback_line_time_s"] = round(fb_line_time, 4)
        elif operator_pinned:
            # 说出来 —— 静默地不缩放和静默地缩放一样坏。
            out["step_fallback_line_time_s"] = float(fb_line_time or 0.0)
            out["step_fallback_speed_note"] = (
                f"回退图按你钉的 {fb_line_time} s/线跑,未按视野缩放 ⇒ "
                f"针尖 {float(fallback_nm) / float(fb_line_time):.0f} nm/s"
                if fb_line_time else "")
        yield CompositeStep(
            step_id=f"{prefix}:fb_scan", skill_name="ScanAt",
            params=scan_at_params(wf, x, y, size_nm=float(fallback_nm),
                                  pixels=wf.step_pixels,
                                  line_time_s=fb_line_time,
                                  # 同一个中心,换大视野重扫 —— 找的就是「这一片
                                  # 有没有台阶」,换地方就换了问题。
                                  origin="same_frame"),
            optional=True, checkpoint_after=False, tags=("level", "wide", "fallback"))
        if _ok(executor, f"{prefix}:fb_scan"):
            yield CompositeStep(
                step_id=f"{prefix}:fb_save", skill_name="SaveScan", params={},
                optional=True, checkpoint_after=False, tags=("level", "fallback"))
            yield CompositeStep(
                step_id=f"{prefix}:fb_latest", skill_name="GetLatestScanFile",
                params={}, optional=True, checkpoint_after=True,
                tags=("level", "fallback"))
            fb_path = _scan_path(executor, f"{prefix}:fb_save", f"{prefix}:fb_latest")
            if fb_path:
                yield CompositeStep(
                    step_id=f"{prefix}:fb_steps", skill_name="AnalyzeFrameTilt",
                    params={"scan_path": fb_path, "check_steps": True},
                    optional=True, checkpoint_after=False,
                    tags=("level", "steps", "fallback"))
                fb_d = _data(executor, f"{prefix}:fb_steps")
                # 只有**真的**在回退图上找到台阶才改用它;没找到就保留原来那张,
                # 免得拿一张同样没台阶、但更粗的图去做后面的平区/调平。
                if fb_d.get("step_dominated"):
                    path, steps_d = fb_path, fb_d
                    out["wide_scan_path"] = fb_path
                    out["step_found_at_nm"] = float(fallback_nm)
                else:
                    out["step_fallback_result"] = (
                        f"{fallback_nm:.0f} nm 上也没有台阶主导 —— "
                        "这一片确实没有可用台阶(不是判据的问题)。")

    out["step_analysis"] = {k: steps_d.get(k) for k in
                            ("step_dominated", "step_ratio", "surface_rms_m",
                             "tilt_valid", "tilt_deg") if k in steps_d}

    # 调平应选择同一台面内的平区，避免把台阶包络作为表面倾斜。
    # same_terrace 受分层质量限制，不代表已经证明窗口无台阶。
    # FindFlatRegion 失败时由本阶段调整位置或尺寸，因此此步骤必须为 optional，
    # 不能把缺少合适平区升级为整个外环的全局故障。
    yield CompositeStep(
        step_id=f"{prefix}:flat", skill_name="FindFlatRegion",
        params={"scan_path": path, "min_window_m": wf.flat_region_nm * 1e-9,
                "same_terrace": True},
        optional=True, checkpoint_after=True, tags=("level", "flat"))
    flat = _data(executor, f"{prefix}:flat")
    fx, fy = flat.get("center_x_m"), flat.get("center_y_m")

    # FindFlatRegion 失败时若给出更小窗口的建议，只按建议重试一次，避免无限缩小评估尺度。
    _say("find_terrace_begin", frame_nm=float(wf.forge_scan_nm),
         window_nm=float(wf.flat_region_nm))

    if fx is None or fy is None:
        # 同时处理较小窗口建议与跨台阶导致的失败原因，不只接其中一个返回分支。
        line = float(flat.get("usable_rms_m") or 0.0)
        cands = [s for s in (flat.get("smaller_windows") or [])
                 if isinstance(s, dict) and s.get("best_rms_m") is not None
                 and line > 0 and float(s["best_rms_m"]) <= line]
        cross_cands = [s for s in (flat.get("smaller_windows_same_terrace") or [])
                       if isinstance(s, dict) and s.get("side_m")]
        side = None
        if cands:
            side = float(cands[0]["side_m"])
            why_small = "小一档就够平"
        elif cross_cands:
            side = float(cross_cands[0]["side_m"])
            why_small = "小一档就装得进单个台面"
        if side is not None:
            out["flat_window_downsized_m"] = side
            _say("terrace_window_shrink", from_nm=float(wf.flat_region_nm),
                 to_nm=side * 1e9)
            yield CompositeStep(
                step_id=f"{prefix}:flat2", skill_name="FindFlatRegion",
                params={"scan_path": path, "min_window_m": side,
                        "same_terrace": True},
                optional=True, checkpoint_after=True, tags=("level", "flat"))
            flat2 = _data(executor, f"{prefix}:flat2")
            if flat2.get("center_x_m") is not None:
                out["flat_window_downsize_reason"] = why_small
                flat = flat2
                fx, fy = flat.get("center_x_m"), flat.get("center_y_m")

    if fx is None or fy is None:
        # 说清是**为什么**找不到 —— 「跨台阶被否掉了 N 个」和「图太小/太脏」是
        # 两种完全不同的处境,给的下一步也不同(换更小的窗 vs 换地方)。
        n_cross = flat.get("windows_cross_terrace")
        detail = (f"(跨台阶否决 {n_cross} 个窗口,台面比 "
                  f"{wf.flat_region_nm:.0f} nm 窄)" if n_cross else "")
        out["reason"] = f"这张图里找不到够大的**单台面**平区来调平{detail}"
        _say("find_terrace_result", found=0, why=out["reason"],
             windows_checked=flat.get("windows_checked"),
             windows_cross_terrace=n_cross,
             usable_rms_pm=(_finite_num(flat.get("usable_rms_m")) or 0.0) * 1e12)
        return out
    out["flat_center"] = [fx, fy]
    out["flat_rms_m"] = flat.get("rms_m")
    # 多针尖那一问在**同一张图**上顺带答了,读数一起报(「既然分析了就都说出来」)。
    sp_c = _step_split_look(executor, scan_path=path, flat=flat,
                            frame_nm=float(wf.forge_scan_nm))
    _say("find_terrace_result", found=1,
         window_nm=(_finite_num(flat.get("window_side_m")) or 0.0) * 1e9,
         rms_pm=(_finite_num(flat.get("rms_m")) or 0.0) * 1e12,
         usable_rms_pm=(_finite_num(flat.get("usable_rms_m")) or 0.0) * 1e12,
         windows_checked=flat.get("windows_checked"),
         windows_cross_terrace=flat.get("windows_cross_terrace"),
         split_score=sp_c.get("score"), split_threshold=sp_c.get("score_threshold"),
         levels_pm=sp_c.get("levels_pm"),
         x_m=fx, y_m=fy)

    # `next_frame_m` 要是**实际选到的那块窗**的边长,不是出厂请求值 —— 上面可能
    # 已经缩过一档。AutoTilt 用它算「这点倾斜在下一帧里值不值得动手」,
    # 喂错尺寸会让预算判断跟着错(尺寸小一档,同样的斜率造成的 Z 跨度就小一截)。
    tilt_params: dict[str, Any] = {
        "next_frame_m": float(flat.get("window_side_m")
                              or wf.flat_region_nm * 1e-9)}
    rms = out["step_analysis"].get("surface_rms_m")
    if isinstance(rms, (int, float)) and rms > 0:
        tilt_params["surface_rms_m"] = float(rms)
    # ⚠️ **C 相的调平此前一条旁白都没有**(2026-08-23 修)。
    #
    # D 相(``_level_on_terrace``)08-17 就补上了 level_begin / level_result,
    # 而 C 相 —— 这一相的名字就叫 level_phase —— 从头到尾只有 GraphExecutor
    # 自动发的那两条通用句(auto_tilt / auto_tilt_result),结论只进
    # ``out["tilt"]`` 这份报文。「调平做没做、效果如何,
    # 旁白里看不到。」两相各改一半,正是本仓「一侧改了,另一侧没跟上」的同形。
    #
    # 这里用的窗与残差是**真正下发给 AutoTilt 的那一份**(tilt_params 与 flat),
    # 不是另取一份 —— 句子里的数与仪器收到的数中间没有人有机会改写。
    _say("level_begin",
         window_nm=float(tilt_params["next_frame_m"]) * 1e9,
         rms_pm=(_finite_num(flat.get("rms_m")) or 0.0) * 1e12)
    yield CompositeStep(
        step_id=f"{prefix}:tilt", skill_name="AutoTilt", params=tilt_params,
        optional=True, checkpoint_after=True, tags=("level", "tilt"))
    # 读法走 ``_tilt_readout`` —— 这里原来抄了一份自己的,而那份读的六个键名
    # (action / skipped / z_span_before_m / z_span_after_m …) 在 AutoTilt 的
    # 回包里**一个都不存在**,于是 ``out["tilt"]`` 恒等于 ``{"reason": …}``。
    t = _tilt_readout(executor, f"{prefix}:tilt")
    _say("level_result", done=t["done"], no_action=t.get("no_action"),
         skipped=t["skipped"], reason=t["detail"] or t["reason"],
         code=t["reason"], hint=t["hint"],
         before_m=t["before_m"], after_m=t["after_m"])
    out["tilt"] = {k: t[k] for k in
                   ("outcome", "done", "skipped", "reason", "detail",
                    "before_m", "after_m", "hint") if t.get(k) not in (None, "")}
    # AutoTilt 没有标定时会 skip 而不动硬件 —— 那是正确行为,不是失败。
    # ``no_action_needed``(本来就够平)也算调平到位:下游问的是「这块地方现在
    # 平不平」,不是「我们有没有动过压电」。
    out["leveled"] = bool(t["done"] or t.get("no_action"))
    if not out["leveled"]:
        out["reason"] = (t["detail"] or t["reason"]
                         or "调平未执行(多半是本机还没做过倾斜标定)")
    return out


# ── D 阶段:扎针尖精修 ──────────────────────────────────────────────────────

def _amp_signal_index() -> "int | None":
    """qPlus 振幅通道的 Nanonis 信号槽号;没配过返回 None。

    读的是仪器档案里那个已有的键(``qplus_amplitude_signal_index``,由
    ``ReadTipOscillationAmplitude`` 的标定流程写入)—— **不新建第二个来源**。
    """
    try:
        from mast.core.instrument_profile import get_config

        idx = get_config("qplus_amplitude_signal_index", None)
        idx = int(idx) if idx is not None else -1
        return idx if 0 <= idx <= 127 else None
    except Exception:  # noqa: BLE001
        return None


def _amp_capture(executor: GraphExecutor, wf: NobleTipWorkflow, *,
                 step_id: str, duration_s: float) -> Iterator[None]:
    """采一段振幅。没配振幅通道就什么都不做(这台机器上测不了起跳)。"""
    idx = _amp_signal_index()
    if idx is None or duration_s <= 0:
        return
    yield CompositeStep(
        step_id=step_id, skill_name="CaptureSignalBuffer",
        params={"channel": str(idx), "duration_s": float(duration_s),
                "poll_hz": float(wf.poke_amp_poll_hz), "include_samples": False},
        # optional:这是**诊断**,不是闸门。采不到振幅不该让扎针失败 ——
        # 起跳检测是额外加的一道观察,不是唯一一道。
        optional=True, checkpoint_after=False, tags=("poke", "ringdown"))


def _amp_rms(executor: GraphExecutor, step_id: str) -> "float | None":
    """一段采集的 RMS 幅度(用 std,不用 mean):振铃是**交流**,均值会把它抵消掉。"""
    if not _ok(executor, step_id):
        return None
    d = _data(executor, step_id)
    for key in ("std", "rms", "std_dev"):
        v = d.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return abs(float(v))
    lo, hi = d.get("min"), d.get("max")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return abs(float(hi) - float(lo)) / 2.0
    return None


def ringdown_report(base_rms: "float | None", after_rms: "float | None",
                    wf: NobleTipWorkflow) -> dict:
    """把两段振幅变成一句**可报的观察**,而不是一个判决。

    这里回答的是「怎么发现起跳?怎么应对?」。发现的机制在这里;**判决不在**:

        ``ring_up_detected`` 只有 None(判不了)与 True(比值远超阈值)两种取值,
        **永远不会是 False**。

    为什么不给 False:阈值(``poke_ringdown_settle_ratio``)是个**占位数**,还没有
    任何真机数据标定过。一个未标定的阈值给出「没起跳」,就是把「我不知道」说成
    「没问题」—— 本仓反复栽的正是这一跤。等用户采回四样数据(扎针前后的振幅
    序列、故意高偏压的正例、20 mV 的负例、f₀ 与 Q)再让它能说 False。

    ``ratio`` 与两段 RMS 一律如实带出:即使判不了,数字本身就是要采的那批数据。
    """
    out: dict[str, Any] = {"baseline_rms": base_rms, "after_rms": after_rms,
                           "ratio": None, "ring_up_detected": None,
                           "threshold_calibrated": False}
    if base_rms is None or after_rms is None:
        out["reason"] = "读不到振幅(没配 qplus_amplitude_signal_index,或采集失败)"
        return out
    if base_rms <= 0:
        out["reason"] = "基线 RMS 为 0 —— 比值无意义"
        return out
    ratio = after_rms / base_rms
    out["ratio"] = ratio
    if ratio >= float(wf.poke_ringdown_settle_ratio):
        out["ring_up_detected"] = True
        out["reason"] = (f"扎针后振幅 RMS 是基线的 {ratio:.1f} 倍"
                         f"(≥{wf.poke_ringdown_settle_ratio:g})—— 疑似音叉起跳。"
                         "**阈值未标定**,这是观察不是判决。")
    else:
        # 刻意不写 False:见 docstring。
        out["reason"] = (f"扎针后振幅 RMS 是基线的 {ratio:.1f} 倍,未达占位阈值 "
                         f"{wf.poke_ringdown_settle_ratio:g} —— **阈值未标定,"
                         "因此不下「没起跳」的结论**。")
    return out


def poke_bias_v(wf: NobleTipWorkflow) -> "float | None":
    """返回扎针偏压；None 表示沿用当前值。qPlus 的振动风险与 Q 等因素有关，不能由单次偏压结果推断。
    使用登记表 form 判定针尖类型，不能按名字子串识别；未登记时沿用通用配置。"""
    try:
        from mast.core.tip_state import is_qplus

        if is_qplus():
            return float(wf.poke_bias_qplus_v)
    except Exception:  # noqa: BLE001 — 读不到针尖不该让扎针失败
        logger.debug("poke_bias_v: 读针尖登记失败,按非 qPlus 处理", exc_info=True)
    return None if wf.poke_bias_v is None else float(wf.poke_bias_v)


def _narrate_indent(poke_data: dict, ind: dict, depth_m: float) -> None:
    """把这一针的 Z 曲线判读念出来，并把曲线本身画成图挂上去。

    扎针的检测应该写进旁白，包括读取到的 z 曲线也应该画出来。在此之前屏幕上
    只看得见「扎了一针」，判定连同它依据的整条曲线都只活在报文里。

    ⚠️ 旁白与配图**都不许弄坏实验** —— 与 ``narrate()`` 同一条纪律，全程吞异常。
    """
    image = None
    try:
        from mast.vision.poke_trace_panel import render_poke_trace_panel

        ch = (poke_data.get("channels") or {})
        z = (ch.get("z") or {})
        c = (ch.get("current") or {})
        png = render_poke_trace_panel(
            z.get("samples_m") or z.get("samples") or [],
            z.get("t_s") or [],
            c.get("samples_a") or c.get("samples") or [],
            c.get("t_s") or [],
            event_t=float(poke_data.get("event_t_s") or 0.0),
            indent=ind,
            meta={"tip_lift_m": -abs(float(depth_m))},
            label=str(poke_data.get("trace_path") or "poke").split("\\")[-1][:40])
        if png:
            image = {"src": png, "origin": "milestone_png"}
    except Exception as exc:  # noqa: BLE001
        logger.debug("扎针 Z 曲线图没画成: %s", exc)
    try:
        from mast.chat.narration import narrate

        # 旁白字段直接平铺，与接收方 schema 对齐；不得包进无人读取的 result 子层。
        narrate("poke_indent", image=image,
                verdict=str(ind.get("verdict") or "insufficient_data"),
                dz_pm=float(ind.get("delta_m") or 0.0) * 1e12,
                tol_pm=float(ind.get("tol_m") or 0.0) * 1e12,
                feedback_segment_source=str(
                    ind.get("feedback_segment_source") or ""),
                feedback_segment_s=ind.get("feedback_segment_s"),
                depth_pm=abs(float(depth_m)) * 1e12)
    except Exception:  # noqa: BLE001
        pass

# shaper 结束后的采集窗口用于观察反馈恢复后的稳定高度。
# 窗口必须覆盖电流回到 setpoint 以及 Z 稳定的过程；时长需在目标仪器验证。
# 参数上限由技能元数据控制，公开版本不包含站点时序标定。
_POKE_POST_ROLL_S = 1.5


def _poke_step(executor: GraphExecutor, wf: NobleTipWorkflow, *,
               step_prefix: str, depth_m: float,
               bias_v: "float | None" = None,
               restore_bias_v: "float | None" = None) -> Iterator[dict]:
    """在当前位置扎一下,返回判定。

    断反馈下压 → 驻留 → 恢复反馈,由硬件 TipShaper 完成。``restore_feedback=True``
    让固件自己把反馈收回去 —— 软件侧另有 finalize 兜底。

    ``lift_height_m`` 取 ``tip_lift_m`` 的相反数:压下去多少就抬回来多少
    (用户参数组:``lift height`` = ``tip lift`` 的相反数)。

    ## 20 mV 走哪条路(C2 的歧义,这里选了第一条)

    有两条要求:参数组里**change bias 关**、bias value 跟随现在的 bias,
    以及对 qPlus 针尖**换 20 mV bias** 来扎针。两条只有在一种读法下
    同时成立:**先把仪器的偏压设成 20 mV,再用 change-bias 关着扎**。若是让 shaper
    自己 change-bias 打到 20 mV,「change bias 关」这条要求就被推翻了。

    所以默认走这条(``wf.poke_bias_via_shaper=False``):
    ``SetBias(20 mV)`` → 扎(change_bias 关)→ ``SetBias(扎之前那个值)``。
    另一条也实现了(``poke_bias_via_shaper=True``):shaper 的 ``change_bias=True``
    + ``bias_v=20 mV``。两条都能跑,默认是前者。

    ⚠️ **这是从两句话里推出来的,不是用户逐字说的**。要推翻它只需要用户回答
    一句:扎的时候 Nanonis 的 bias 读数是 20 mV,还是仍是成像偏压?
    (2026-08-10 已把这个问题记录在案。)

    扎完**一定把偏压设回去**:不设回去的话后面每一张评估图都跑在 20 mV 上,而
    那不是用户看团簇用的条件 —— 也正是本仓「改了不说、也不改回来」那一类。
    """
    # ── shaper 那条路已停用 ────────────────────────────────────────────
    # 要求：要考虑 bias change 的时间,另外考虑针尖 z
    # 改变的时间,**要先慢慢改了 bias,z 的扎入要控制在其之后**。
    #
    # 顺序那一半 shaper 做得到:``Bias Settling Time`` 的文档明说它「**也是施加
    # Bias(V) 之后、第一次 Z 斜坡之前要等的时间**」(nanonis_spm
    # NanonisClass.py:3961)。但**缓变那一半做不到** —— shaper 的 change-bias 是
    # 把值**一次写上去**,没有任何斜率参数。而如果激励真的是静电力 ∝ V²(猜测),
    # 那一次阶跃本身就是一记冲量,正是要避开的东西。
    #
    # 半个保证不是保证,所以这条路**拒绝执行**而不是悄悄降级 —— 一个留着但会毁
    # 针的可选项比没有更坏,而一个被静默忽略的开关是本仓记过的「死配置」。
    if wf.poke_bias_via_shaper and bias_v is not None:
        return {"ok": False, "verdict": "refused",
                "reason": ("poke_bias_via_shaper=True 已停用:Nanonis TipShaper 的 "
                           "change-bias 只能**阶跃**改偏压,做不到要求的"
                           "「先慢慢改 bias」。请走默认路径(先 SetBias 缓变、"
                           "change_bias 关着扎)。")}

    set_bias_first = bias_v is not None
    if set_bias_first:
        # **缓变**,不是阶跃。SetBias 的 slew_rate_v_per_s 就是为此存在的。
        yield CompositeStep(
            step_id=f"{step_prefix}:poke_bias", skill_name="SetBias",
            params={"bias_v": float(bias_v),
                    "slew_rate_v_per_s": float(wf.poke_bias_slew_v_per_s)},
            optional=False, checkpoint_after=False,
            tags=("poke", "qplus_bias"))
        # 静置 + 采基线是**同一个窗口**:等偏压稳下来的那段时间,正是测「安静时
        # 的噪声底」最该测的时候。所以这里不需要一个单独的 Wait 步骤(全仓也没有
        # 通用的 Wait 技能),一次采集同时办两件事,而且它是**读**,中止路径上照样
        # 放行。窗口取两者的较大值。
        yield from _amp_capture(
            executor, wf, step_id=f"{step_prefix}:amp_base",
            duration_s=max(float(wf.poke_bias_settle_s),
                           float(wf.poke_amp_baseline_s)))

    # ``change_bias`` 恒为**关**:偏压在上面已经缓变到位了,shaper 不该再动它
    # (用户的参数组原文就是「change bias 关、bias value 跟随现在的 bias」)。
    shaper: dict[str, Any] = {
        "change_bias": False,
        "tip_lift_m": -abs(depth_m),      # 向下压
        "lift_height_m": abs(depth_m),    # 再抬回来(tip lift 的相反数)
        "bias_settling_s": wf.poke_dwell_s,
        "restore_feedback": True,
        "pre_roll_s": max(0.1, float(wf.move_settle_s)),
        # 采集窗口过短可能只覆盖恢复前的平台，使高度差趋近零。
        # 应在电流恢复后保留足够样本用于稳定高度的统计。
        "post_roll_s": _POKE_POST_ROLL_S,
    }
    yield CompositeStep(
        step_id=f"{step_prefix}:poke",
        skill_name="TipShapeWithReadback",
        params=shaper,
        optional=False, checkpoint_after=True,
        tags=("poke", f"depth_nm={abs(depth_m) * 1e9:.2f}"),
    )
    poke_ok = _ok(executor, f"{step_prefix}:poke")

    ring: dict[str, Any] = {}
    if set_bias_first:
        # 扎针后先观察振幅并等待衰减，再恢复偏压，保持收尾动作的顺序。
        yield from _amp_capture(executor, wf,
                                step_id=f"{step_prefix}:amp_after",
                                duration_s=wf.poke_amp_watch_s)
        base_rms = _amp_rms(executor, f"{step_prefix}:amp_base")
        after_rms = _amp_rms(executor, f"{step_prefix}:amp_after")
        ring = ringdown_report(base_rms, after_rms, wf)

        # 疑似起跳 ⇒ 再等一段(带超时),让它响完。等不到就**如实说等不到**,
        # 不假装它停了 —— 但也不因此判失败:这一层只观察。
        if ring.get("ring_up_detected") and wf.poke_ringdown_timeout_s > 0:
            yield from _amp_capture(
                executor, wf, step_id=f"{step_prefix}:amp_settle",
                duration_s=min(float(wf.poke_ringdown_timeout_s),
                               float(wf.poke_amp_watch_s) * 3.0))
            settle_rms = _amp_rms(executor, f"{step_prefix}:amp_settle")
            ring["settle_rms"] = settle_rms
            ring["settled"] = (
                None if (settle_rms is None or not base_rms) else
                bool(settle_rms / base_rms < float(wf.poke_ringdown_settle_ratio)))

    if set_bias_first and restore_bias_v is not None:
        # 无条件恢复:扎针失败时更要把偏压放回去(下一步是人来看,不该看到一个
        # 没人说过的 20 mV)。**缓变**,同样的理由 —— 升回去也是一次冲量。
        # optional=True:恢复失败不该盖掉扎针本身的结论。
        yield CompositeStep(
            step_id=f"{step_prefix}:poke_bias_restore", skill_name="SetBias",
            params={"bias_v": float(restore_bias_v),
                    "slew_rate_v_per_s": float(wf.poke_bias_slew_v_per_s)},
            optional=True, checkpoint_after=False,
            tags=("poke", "qplus_bias"))
    if not poke_ok:
        return {"ok": False, "verdict": "failed", "ringdown": ring}
    _poke_data = _data(executor, f"{step_prefix}:poke")
    ind = _poke_data.get("indent", {}) or {}
    _narrate_indent(_poke_data, ind, depth_m)
    return {"ok": True,
            "verdict": str(ind.get("verdict") or "insufficient_data"),
            "dz_nm": float(ind.get("delta_m") or 0.0) * 1e9,
            "depth_nm": abs(depth_m) * 1e9,
            # 起跳观察一路带出来 —— 它现在的主要用途是**采数据**给用户标定阈值。
            "ringdown": ring}


def _cluster_look(executor: GraphExecutor, wf: NobleTipWorkflow, *,
                  step_prefix: str, x: float, y: float) -> Iterator[dict]:
    """扫一张小图看刚扎出来的簇:几个峰、圆不圆。

    峰数用 ``n_components``:第一判据是「有几个峰?表明是否有双针尖」。
    """
    yield CompositeStep(
        step_id=f"{step_prefix}:cluster_scan", skill_name="ScanAt",
        params=scan_at_params(wf, x, y, size_nm=wf.cluster_scan_nm,
                              pixels=wf.cluster_pixels,
                              line_time_s=wf.cluster_line_time_s,
                              # 要看的就是**刚扎出来的那个簇**,换地方等于换对象。
                              # (`select="center"` 那条判据同理:画面里最大的常常
                              #  是两轮之前的坑,拿它评针尖是在评错东西。)
                              origin="just_poked"),
        optional=False, checkpoint_after=False, tags=("cluster",))
    if not _ok(executor, f"{step_prefix}:cluster_scan"):
        return {"ok": False, "reason": "簇扫描没能完成"}
    yield CompositeStep(
        step_id=f"{step_prefix}:cluster_save", skill_name="SaveScan", params={},
        optional=True, checkpoint_after=False, tags=("cluster",))
    yield CompositeStep(
        step_id=f"{step_prefix}:cluster_latest", skill_name="GetLatestScanFile",
        params={}, optional=True, checkpoint_after=False, tags=("cluster",))
    path = _scan_path(executor, f"{step_prefix}:cluster_save",
                      f"{step_prefix}:cluster_latest")
    if not path:
        return {"ok": False, "reason": "簇扫描拿不到文件路径"}

    yield CompositeStep(
        step_id=f"{step_prefix}:roundness", skill_name="AssessClusterRoundness",
        # 评估以本次处理点为中心的凸起，避免选中其他轮次的形貌。
        # 显式使用 bright 极性，防止大团簇占据画面时把背景识别成目标。
        # 使用 physical 阈值与 boundary 形状；weighted 指标可附带报告，不能替代对肩部的识别。
        # 指标稳定性与识别目标不同，不能只优化代理指标。
        params={"scan_path": path, "min_axis_ratio": wf.min_axis_ratio,
                "select": "center", "polarity": "bright",
                "threshold_mode": "physical", "shape_mode": "boundary"},
        optional=True, checkpoint_after=True, tags=("cluster",))
    if not _ok(executor, f"{step_prefix}:roundness"):
        return {"ok": False, "reason": "簇分析没能完成", "scan_path": path}
    d = _data(executor, f"{step_prefix}:roundness")
    n_peaks = int(d.get("n_components") or 0)
    return {
        "ok": True,
        "scan_path": path,
        # 等效轴比：「相当于一个短轴/长轴 = q 的椭圆」。连续量,
        # 因为用户的用法是**比较相继几次哪次更圆**,不是只看合格与否。
        "axis_ratio": d.get("equivalent_axis_ratio"),
        # 三态透出去：``is_round`` 可以是 None = 团簇太小**判不了**。
        # 下面的收工判据用 ``is True``,所以 None 与 False 都不收工 —— 那是对的
        # (不能把没验证过的针尖当整好了),但**理由不同**,得让上层说得出区别。
        "is_round": d.get("is_round"),
        "roundness_undecidable": d.get("roundness_undecidable"),
        "n_peaks": n_peaks,
        # 连通域数量不能直接区分多针尖与表面多个结构。
        # 每次处理后针尖可能改变，跨帧一致性也不自动适用。
        # 保留三态：None 表示无法判定，不能当作多针尖或不合格。
        "double_tip": d.get("multi_tip"),
        "multi_tip_undecidable": d.get("multi_tip_undecidable"),
        # 新算法的两个读数一并透出(判据由 shape_mode 决定,这里只是让上层看得见)。
        "weighted_axis_ratio": d.get("weighted_axis_ratio"),
        "boundary_axis_ratio": d.get("boundary_axis_ratio"),
        "area_px": d.get("area_px"),
    }


def flat_poke_sites(executor: GraphExecutor, wf: NobleTipWorkflow, *,
                    step_prefix: str, used: list, want: int) -> Iterator[Any]:
    """找一块台面,在上面调平,交出一批**已知平坦**的扎针落点。

    返回 ``("ok", sites)`` 或 ``("spent", None)`` —— **只有两态**。

    ═══════════════════════════════════════════════════════════════════════
    2026-08-17 重写。旧版有第三态 ``("undecidable", None)``,而调用方对它的
    处置是「退回按几何找落点,继续扎」—— 也就是**闭着眼睛乱扎**。
    ═══════════════════════════════════════════════════════════════════════

    要求,每一条都对应下面的一段:

      · 扎针没有落在调平后的台面上,而是直接问地图、在一个没标记的位置就扎 ——
        那么找台面这一步就失去了意义,反复扎在台阶边缘也就不奇怪。
      · 一处找不到台面之后就随机扎,扎两下才又回头去找台面。
      · 已经找过的位置必须记住并避开,不能在同一片区域反复找。
      · 台面上的调平没有出现在流程里。

    ── 这一版的逻辑,一句话一步

      ① 问地图要一个干净落点(**避开所有搜过的整片区域**)→ 移过去 → 扫一张 200 nm
      ② ``FindFlatRegion`` 找 35 nm 单台面窗
         · 找到      → ③
         · 「换小一档就有」→ **按它给的窗重试**(技能自己把答案递过来了)
         · 「这里没有」  → 记住这一整片 → 回 ①
      ③ **就在这块台面上调平**(AutoTilt)—— 这才是「台面上的调平」
      ④ 交出落点

    **没有「那就随便扎」这条路。** 找不到就换一块;换不动了就报表面用完,
    由外环粗动换区。要求:必须找到台面;找不到就扩大范围继续找。
    """
    if wf.poke_flat_window_nm >= wf.poke_site_scan_nm:
        raise ValueError(
            f"平区窗 {wf.poke_flat_window_nm:g} nm ≥ 找台面图视野 "
            f"{wf.poke_site_scan_nm:g} nm —— 这样永远找不到落点。"
            f"要么把 poke_flat_window_nm 调小,要么把 poke_site_scan_nm 调大。")

    pd = executor.progress.partial_data
    searched = [tuple(t) for t in (pd.get(_SEARCHED_KEY) or [])]
    frame_r = float(wf.poke_site_scan_nm) * 1e-9 / 2.0

    # ① 换一块地方(避开每一片搜过的 200 nm 区域)
    spot = yield from _relocate(
        executor, step_prefix=step_prefix, purpose="tip_shape", used=used,
        avoid=[(sx, sy, frame_r) for sx, sy in searched])
    if spot is None:
        return "spent", None
    cx, cy, _known = spot
    _say("find_terrace_begin", x_m=cx, y_m=cy,
         frame_nm=float(wf.poke_site_scan_nm),
         window_nm=float(wf.poke_flat_window_nm),
         searched_frames=len(searched))

    yield CompositeStep(
        step_id=f"{step_prefix}:scan", skill_name="ScanAt",
        params=scan_at_params(wf, cx, cy, size_nm=wf.poke_site_scan_nm,
                              pixels=wf.poke_site_pixels,
                              line_time_s=wf.poke_site_line_time_s,
                              origin="clean_spot"),
        optional=True, checkpoint_after=False, tags=("poke", "sites"))
    if not _ok(executor, f"{step_prefix}:scan"):
        # 扫不成 —— **不是**「这里没台面」,但处置一样:换一块。
        # 绝不退回「随便扎」:那一步的代价是一根针,而换一块只花几十秒。
        _say("find_terrace_result", found=0, why="这一片没扫成 —— 换一块再扫")
        _remember_searched(executor, cx, cy)
        return "ok", []

    yield CompositeStep(
        step_id=f"{step_prefix}:save", skill_name="SaveScan", params={},
        optional=True, checkpoint_after=False, tags=("poke", "sites"))
    yield CompositeStep(
        step_id=f"{step_prefix}:latest", skill_name="GetLatestScanFile", params={},
        optional=True, checkpoint_after=False, tags=("poke", "sites"))
    path = _scan_path(executor, f"{step_prefix}:save", f"{step_prefix}:latest")
    if not path:
        _say("find_terrace_result", found=0, why="拿不到刚扫那张图的路径 —— 换一块再扫")
        _remember_searched(executor, cx, cy)
        return "ok", []

    # ② 找台面 —— **技能说换小一档就换小一档**
    window_m = float(wf.poke_flat_window_nm) * 1e-9
    for shrink in range(_FLAT_SHRINK_TRIES):
        sid = f"{step_prefix}:flat" if shrink == 0 else f"{step_prefix}:flat{shrink}"
        yield CompositeStep(
            step_id=sid, skill_name="FindFlatRegion",
            params={"scan_path": path,
                    "min_window_m": window_m,
                    "min_separation_m": wf.poke_site_separation_nm * 1e-9,
                    "count": max(1, int(want)),
                    # 将累计处理位置传入 exclude_used_spots，防止新一批候选重新选中旧点。
                    # 排除半径复用 min_separation_m，不只限制当前批次点之间的距离。
                    "exclude_used_spots": _spot_list(_dirty(executor), used),
                    # 落点必须在**单台面内** —— 一个「两半各自平坦但中间有台阶」
                    # 的窗对扎针毫无用处,而最小 RMS 恰恰会选到它。
                    "same_terrace": True},
            optional=True, checkpoint_after=False, tags=("poke", "sites", "flat"))
        d = _data(executor, sid) or {}
        if _ok(executor, sid):
            break

        # 消费 FindFlatRegion 的 smaller_windows 建议，让重试参数由返回证据决定。
        usable = _finite_num(d.get("usable_rms_m"))
        smaller = [w for w in (d.get("smaller_windows") or [])
                   if isinstance(w, dict)]
        advice = None
        for w in smaller:
            side, rms = _finite_num(w.get("side_m")), _finite_num(w.get("best_rms_m"))
            if side is None or rms is None or usable is None:
                continue
            if rms <= usable and side >= _FLAT_WINDOW_FLOOR_M and side < window_m:
                advice = side
                break
        if advice is None:
            break
        _say("terrace_window_shrink",
             from_nm=window_m * 1e9, to_nm=advice * 1e9)
        window_m = advice
    else:
        d = _data(executor, f"{step_prefix}:flat{_FLAT_SHRINK_TRIES - 1}") or {}

    if not _ok(executor, sid):
        # 找过了,这一片没有(或者缩到底也没有)⇒ **记住这一整片**,换一块再扫。
        # 记的是帧心、半径按帧的一半算 —— 只排除 30 nm 的话下一轮会扫回同一片
        # (要求:找过的位置要记住并避开)。
        best = _finite_num(d.get("best_rms_m"))

        # 缺少合适台面时同时检查台阶劈裂，区分局部表面不足与针尖形貌问题。
        split = _step_split_look(executor, scan_path=path, flat=None,
                                 frame_nm=float(wf.poke_site_scan_nm))
        if split.get("verdict") == "split":
            return "split_tip", None

        _say("find_terrace_result", found=0,
             why=(f"这一片最平的一块残差 {best * 1e12:.0f} pm,超过可用线"
                  if best is not None else "这一片没有可用台面")
                 + " —— 台阶未见重影(非多针尖),是这一片确实不平;记下来,换一块再扫",
             usable_rms_pm=(_finite_num(d.get("usable_rms_m")) or 0.0) * 1e12,
             windows_checked=d.get("windows_checked"),
             windows_skipped=d.get("windows_skipped"),
             windows_cross_terrace=d.get("windows_cross_terrace"),
             split_score=split.get("score"),
             split_threshold=split.get("score_threshold"),
             levels_pm=split.get("levels_pm"))
        _remember_searched(executor, cx, cy)
        return "ok", []

    sites: list = []
    for site in (d.get("sites") or []):
        sx, sy = site.get("center_x_m"), site.get("center_y_m")
        if sx is not None and sy is not None:
            sites.append((float(sx), float(sy)))
    if not sites:
        # 技能说成功了却没给 sites(老版本不支持 count)—— 退回它给的那一个中心。
        one_x, one_y = d.get("center_x_m"), d.get("center_y_m")
        if one_x is not None and one_y is not None:
            sites = [(float(one_x), float(one_y))]
    if not sites:
        _say("find_terrace_result", found=0, why="技能说找到了却没给落点 —— 换一块再扫")
        _remember_searched(executor, cx, cy)
        return "ok", []

    # 先把多针尖那一问答了 —— 它和找台面用的是**同一张图**,
    # 而且它的读数要一起报出去(要求:分析过的量都要报出来)。
    sp = _step_split_look(executor, scan_path=path, flat=d,
                          frame_nm=float(wf.poke_site_scan_nm))
    _say("find_terrace_result", found=len(sites),
         window_nm=window_m * 1e9,
         rms_pm=(_finite_num(d.get("rms_m")) or 0.0) * 1e12,
         usable_rms_pm=(_finite_num(d.get("usable_rms_m")) or 0.0) * 1e12,
         windows_checked=d.get("windows_checked"),
         windows_skipped=d.get("windows_skipped"),
         windows_cross_terrace=d.get("windows_cross_terrace"),
         split_score=sp.get("score"), split_threshold=sp.get("score_threshold"),
         levels_pm=sp.get("levels_pm"),
         x_m=sites[0][0], y_m=sites[0][1])

    # 找到台面仍需消费 split 判决；台面存在不能推翻独立的多针尖判据。
    # split 时丢弃本批落点并返回对应分支；undecidable 保持不同含义。
    if str(sp.get("verdict") or "") == "split":
        return "split_tip", None

    # 每个调平阶段都使用当前图的测量结果，不能依赖上一阶段另一幅图的倾斜。
    yield from _level_on_terrace(executor, wf, step_prefix=step_prefix, d=d)
    return "ok", sites


def _level_on_terrace(executor: GraphExecutor, wf: NobleTipWorkflow, *,
                      step_prefix: str, d: dict) -> Iterator[Any]:
    """在刚找到的那块台面上调平。**说出来做没做。**

    ## 2026-08-18:这一步从上线那天起一次都没跑成过

    参数里带了一个 ``scan_path``,而 ``AutoTilt`` **没有声明过这个参数**——
    ``BaseSkill.validate_params`` 对未声明的键直接返回
    ``Unknown parameter: 'scan_path'``,于是这一步在 ``AutoTilt.execute``
    被调用**之前**就已经失败了。步骤是 ``optional=True``,所以整条流程若无其事
    地继续,只在旁白里留下一句没有原因的「调平没做成」。

    「台面上的调平」是 08-17 专门补上的一环(在那之前调平只在 C 相做,而 C 相
    看的是另一张图)。它补上去了,但**一次也没有真的执行过**。

    另外两处同时修掉:

    * ``next_frame_m`` **必须传**。AutoTilt 的触发判据是「这点斜坡在**下一帧**
      里吃掉多少 Z 量程」,不传就退到 ``Scan_FrameGet`` —— 那时现场是刚扫完的
      200 nm 落点图,而我们要调平的是其中那块 35 nm 的台面。差了近六倍的尺度,
      同一个斜率算出来的 Z 跨度差六倍,判决自然跟着错。C 相一直是传的。
    * ``surface_rms_m`` 不再显式传 ``None``。校验对「可选参数收到 None」是放行的
      (等价于没传),但那要靠读 ``base.py`` 里一段注释才知道 —— 让「没有值」
      直接表现为「不下发这个键」,不依赖下游的宽容。
    """
    rms = _finite_num(d.get("rms_m"))
    win = _finite_num(d.get("window_side_m"))
    _say("level_begin", window_nm=(win or 0.0) * 1e9, rms_pm=(rms or 0.0) * 1e12)
    tilt_params: dict[str, Any] = {}
    if win and win > 0:
        tilt_params["next_frame_m"] = float(win)
    if rms is not None and rms > 0:
        tilt_params["surface_rms_m"] = float(rms)
    yield CompositeStep(
        step_id=f"{step_prefix}:tilt", skill_name="AutoTilt", params=tilt_params,
        optional=True, checkpoint_after=False, tags=("poke", "sites", "level"))
    t = _tilt_readout(executor, f"{step_prefix}:tilt")
    _say("level_result", done=t["done"], no_action=t.get("no_action"),
         skipped=t["skipped"], reason=t["detail"] or t["reason"],
         code=t["reason"], hint=t["hint"],
         before_m=t["before_m"], after_m=t["after_m"])


def poke_phase(executor: GraphExecutor, wf: NobleTipWorkflow, *,
               prefix: str = "D", should_stop=None) -> Iterator[dict]:
    """先深后浅:深扎修形状,再在恰好跳变的临界深度反复浅扎把簇做小。

    D1 深扎 —— 刚大修完的针尖先来一下深的(1~2 nm)。没扎上就加深;扎出坑说明针
    尖很尖(用户:少见,是好信号);扎出簇就扫图看形状,有双峰或不圆就继续深扎。

    D2 临界浅扎 —— 从 100 pm 起步一级级加,直到 Z 恰好开始跳,然后在那个深度反复
    扎(每次换地方)。簇会越来越小、越来越圆,这是用户把针尖做尖的收尾手法。

    收尾一定发生:无论中途怎么退出,都把反馈开回来。
    """
    pd = executor.progress.partial_data
    log: list[dict[str, Any]] = list(pd.get(f"{prefix}:log", []))
    used: list[tuple[float, float]] = [tuple(p) for p in pd.get(f"{prefix}:used", [])]
    pokes = int(pd.get(f"{prefix}:pokes", 0))
    out: dict[str, Any] = {"phase": "poke", "stage": "deep", "refined": False,
                           "log": log, "pokes": pokes}

    budget = int(wf.poke_budget)
    # 平坦落点的手牌:空了就重扫一张图补一批(2026-08-14)。
    # **刻意不进 partial_data**:这些坐标只在扫出它们的那一帧里成立,
    # 而续跑时表面已经变了(中间那些扎针把它弄脏了)——照着旧坐标扎,
    # 是在扎一个已经不存在的平台。续跑重扫一张,几分钟的事。
    flat_sites: list = []
    refills = 0
    dry_refills = 0
    depth_nm = float(wf.poke_depth_nm)
    stage = str(pd.get(f"{prefix}:stage", "deep"))
    critical_nm: float | None = pd.get(f"{prefix}:critical_nm")
    repeats = int(pd.get(f"{prefix}:repeats", 0))
    #: 连续几针没做出「单峰且圆」。满了就退回去打脉冲(2026-08-17)。
    unround = int(pd.get(f"{prefix}:unround", 0))
    last_cluster: dict[str, Any] = {}
    #: 临界搜索是否已经起过步。续跑时 partial_data 里存着当时的深度,
    #: 不能再从 100 pm 重来一遍。
    _in_critical_search = bool(pd.get(f"{prefix}:in_critical_search", False))
    if _in_critical_search:
        depth_nm = float(pd.get(f"{prefix}:depth_nm", depth_nm))
    #: 同一深度已经换地方重试了几次(C11)。
    same_depth = int(pd.get(f"{prefix}:same_depth", 0))

    # ── 扎针偏压(C2)。qPlus 上是物理,不是偏好 —— 见 `poke_bias_v`。
    #
    # 扎之前那个偏压要**读一次**再存着,好在每次扎完放回去。读不到就不改偏压:
    # 「不知道原来是多少」时把它设成 20 mV 再也回不去,比不设更坏。
    pbias = poke_bias_v(wf)
    restore_bias: "float | None" = pd.get(f"{prefix}:bias_before")
    if pbias is not None and restore_bias is None:
        yield CompositeStep(
            step_id=f"{prefix}:bias_before", skill_name="GetBias", params={},
            optional=True, checkpoint_after=False, tags=("poke", "qplus_bias"))
        if _ok(executor, f"{prefix}:bias_before"):
            raw = _data(executor, f"{prefix}:bias_before").get("bias_v")
            restore_bias = None if raw is None else float(raw)
            executor.set_partial(f"{prefix}:bias_before", restore_bias)
        if restore_bias is None:
            logger.warning("扎针:读不到当前偏压,%s 的 %g V 不下发(改了就回不去)",
                           prefix, pbias)
            pbias = None
    out["poke_bias_v"] = pbias
    out["bias_before_v"] = restore_bias

    # 外层只跑一趟(末尾无条件 break):内层任何一处 break 都要落到同一段收尾。
    while True:
        while pokes < budget:
            # ⏱📊 **这个循环原来既看不到表也听不到告警。**
            #
            # 它最多跑 ``poke_budget`` = 30 次扎针,每次带一张簇图 ≈ **42 分钟**,
            # 而外面的时间预算只在内环(A⇄B)开头查、电流监控的水位线只在
            # 「精修开始前」查一次。于是两个洞开在同一个位置,而且正是**最危险的
            # 那一段**:这是整条流程里唯一会反复把针压进表面的地方。
            #
            # ``should_stop`` 是调用方给的一个回调:返回空 = 继续,返回一个
            # 理由字符串 = 立刻收工。**判据留在调用方**(它知道预算和告警),
            # 这里只负责问和听 —— 本相不需要知道「时间」或「告警」是什么。
            if should_stop is not None:
                try:
                    why = should_stop()
                except Exception:  # noqa: BLE001 — 问不出来不该让精修崩掉
                    why = ""
                if why:
                    out["stopped_early"] = str(why)
                    out["reason"] = (
                        f"精修中止({why}) —— **这不是关于针尖的结论**,"
                        f"已扎 {pokes} 次,判据没走完。")
                    break
            # ⚠️ **补落点必须在 ``pokes += 1`` 之前。** 重扫一张图找不到平区时
            # 这里会 ``continue``,而计数已经加过的话就等于**白吃一次扎针预算**
            # (而且 ``sp`` 那个编号也被占掉,产物里留下一个没有对应扎针的步骤号)。
            # 第一版正是插在计数之后 —— 靠把两行的顺序读出来才发现。
            # ── 落点必须落在**已知平坦**的地方(2026-08-14)──────────────
            #
            # 改动前这里直接 ``_relocate(purpose="tip_shape")`` —— 那只保证
            # 「这一点没被用过」,不保证「这一点是平的」⇒ 可能扎在台阶边缘。
            # 现在:一批平坦落点用完了就**就地重扫一张图**再要一批
            # (扎完四个之后再扫一张 100 nm 图找台面)。
            if not flat_sites:
                # ``flat_poke_sites`` 现在只有两态(2026-08-17 重写):
                #   ("spent", None) —— 换不动了,该粗动换区
                #   ("ok", sites)   —— 空表 = 这一片没有,换一块再扫
                # **没有第三态**。旧版的 "undecidable" 会退回按几何选点继续扎,
                # 那就是「闭着眼睛乱扎」的老问题——
                # 一次次扎在台阶上是正常的,因为本来就是盲扎,不看具体位置。
                status, got = yield from flat_poke_sites(
                    executor, wf, step_prefix=f"{prefix}:fs{refills}",
                    used=used, want=max(1, budget - pokes))
                refills += 1
                if status == "spent":
                    out["reason"] = "没有干净的地方可以继续扎了 —— 该换区或换样品位置。"
                    out["surface_spent"] = True
                    break
                if status == "split_tip":
                    # 多针尖造成的台阶分裂不能靠反复更换落点解决，需返回相应的处理分支。
                    out["needs_pulse"] = True
                    out["split_tip"] = True
                    out["reason"] = ("大图上台阶出现重影(多针尖)—— 找不到台面的原因在针尖,"
                                     "不在表面;换位置无效。回去打脉冲。")
                    break
                if not got:
                    # 这一片没有可用台面 —— 换一块再扫。**不是**表面用完。
                    # 那一片已经被 flat_poke_sites 记进 searched_frames,
                    # 所以下一次 _relocate 会避开它(要求:找过的位置要
                    # 记住并避开)。
                    dry_refills += 1
                    if dry_refills >= int(wf.poke_flat_dry_refills):
                        out["reason"] = (
                            f"连续 {dry_refills} 次重扫都找不到可用平区 —— "
                            "这一片台阶太密,该换区。**这不是针尖的结论。**")
                        out["surface_spent"] = True
                        break
                    continue
                dry_refills = 0
                flat_sites = got
            x, y = flat_sites.pop(0)
            pokes += 1
            sp = f"{prefix}:p{pokes}"
            used.append((x, y))
            # 扎针同样弄脏表面(避让半径 30 nm,比脉冲的 150 nm 小但不是零)。
            # 记进全流程共享表 + 地图,后面的 verify / level 才躲得开。
            # ``kind="tip_shape"`` ⇒ 30 nm 避让圈(扎针的坑比脉冲小得多)。
            _mark_dirty(executor, x, y, kind="tip_shape")
            _say("poke_begin", n=pokes, stage=stage, x_m=x, y_m=y,
                 depth_pm=float(depth_nm) * 1000.0)
            # ⭐ **去那个落点。** 少了这一步,针就扎在上一次停下的地方,
            # 而簇图扫在这个落点上 —— 扎的和看的不是同一处(2026-08-17 修)。
            yield from _move_to(executor, step_prefix=sp, x=x, y=y)

            if stage == "critical":
                # 临界期第一次进来才落到起步深度;之后深度由上一轮算好带过来,
                # 每轮都重置回起步值的话「100 → 150 → 200」这级台阶永远走不起来。
                if critical_nm is not None:
                    depth_nm = float(critical_nm)
                elif not _in_critical_search:
                    depth_nm = wf.critical_start_pm / 1000.0
                    _in_critical_search = True
            res = yield from _poke_step(executor, wf, step_prefix=sp,
                                        depth_m=depth_nm * 1e-9,
                                        bias_v=pbias,
                                        restore_bias_v=restore_bias)
            entry: dict[str, Any] = {"poke": pokes, "stage": stage,
                                     "x_m": x, "y_m": y, **res}
            log.append(entry)

            if not res.get("ok"):
                out["reason"] = "扎针尖执行失败 —— 停下来让人看。"
                break

            verdict = res["verdict"]

            # ── D1 深扎 ────────────────────────────────────────────────
            if stage == "deep":
                # 每次扎针之后都采图评估，Z 变化保留为辅助读数，不能单独阻止图像采集。
                cl = yield from _cluster_look(executor, wf, step_prefix=sp,
                                              x=x, y=y)
                entry["z_verdict"] = verdict      # 留档:不再用它分支
                if cl.get("ok") and cl.get("is_round") is None:
                    # is_round is None = 判据说「小到判不了」(低于 20 像素
                    # 下限)。在**这个**分支上,它才是最接近「没扎上」的读数 ——
                    # 而不是 Z 说的那个。
                    # ⚠️ 三态照旧:这不等于「表面上什么都没有」,只等于
                    # 「没有大到能判的东西」。所以下一步是**加深再试**,
                    # 不是「这根针不行」。
                    entry["note"] = (f"扫图上没有大到能判的簇(面积 {cl.get('area_px')} px,"
                                     f"Z 判定 {verdict})—— 加深再来")
                    prev_nm = depth_nm
                    depth_nm = min(depth_nm * (1.0 + wf.poke_deepen_frac),
                                   wf.poke_depth_max_nm)
                    entry["next_depth_nm"] = depth_nm
                    _decide(entry, "deeper", depth_pm=prev_nm * 1e3,
                            next_depth_pm=depth_nm * 1e3)
                    if depth_nm >= wf.poke_depth_max_nm:
                        entry["note"] = "已到深扎上限"
                        out["reason"] = (f"扎到 {wf.poke_depth_max_nm:.1f} nm 仍然"
                                         f"扫不出簇 —— 结条件或 Z 标定可能有问题。")
                        break
                    _stash_poke(executor, prefix, log, used, pokes, stage,
                                critical_nm, repeats)
                    continue
                if stage == "deep" and verdict == "tip_changed_or_pit":
                    # 负向 Z 变化触发阶段转换前必须已有图像支持；cluster 本身不代表可以切换到浅扎。
                    entry["note"] = f"扫图确认表面有变化 —— 转临界深度搜索"
                    _decide(entry, "stage_switch",
                            next_depth_pm=float(wf.critical_start_pm))
                    stage = "critical"
                    critical_nm = None
                entry["cluster"] = cl
                last_cluster = cl
                if not cl.get("ok"):
                    _stash_poke(executor, prefix, log, used, pokes, stage,
                                critical_nm, repeats)
                    continue
                # ``double_tip`` 现在是**三态**(2026-08-15)。写 `is True` 而不是
                # 直接取真值:None = 判不了,**判不了不该把针尖推去换地方**。
                # None 恰好也是 falsy,所以不写 `is True` 行为也一样 —— 但那是
                # 碰巧对,而碰巧对的东西下一个人改起来不会知道自己动了什么。
                if cl.get("double_tip") is True or not cl.get("is_round"):
                    # 扎上了、但不够圆 —— **换个地方,同一深度再来**(C11)。
                    #
                    # 用户的序列是 500 → 500 → 500 → 200 pm:不圆是**位置**的
                    # 问题(那块表面下面是什么样没人知道),不是深度不够。
                    # 从前这里直接加深,于是「不圆」会把针尖越扎越深 —— 与范本
                    # 方向相反,而且加深是不可逆的那个方向。
                    #
                    # 下一轮循环开头的 `_relocate` 本来就会换地方,所以这里只要
                    # **不动深度**。同一深度试满 `poke_same_depth_retries` 次仍不圆,
                    # 才动深度 —— 而且是**往浅里走**,见下。
                    #
                    # ⚠️ 2026-08-15 一百针标定纠正的方向错误。
                    #
                    # 从前这里加深,理由是「同一深度都不行 ⇒ 深度不够」。但**能走到
                    # 这一行,就说明已经接触上了**(没接触是上面 no_change 那条分支)。
                    # 接触上了还不圆,不是碰得不够深,是**搬多了**:
                    #     深度 → 面积 ρ=+0.53 → 轴比 ρ=−0.60,
                    #     深度 → 轴比 ρ=−0.46(p=0.005, n=35,只算真接触上的);
                    #     同一检验放到没接触那组 p=0.19 ⇒ 不是地形伪迹,是簇被扎大了。
                    # 各档接触时的轴比中位 0.740 / 0.641 / 0.441 / 0.527 / 0.353,
                    # 到 2 nm 判据几乎不可能过 —— 旧逻辑等于朝着「永远达不到」走,
                    # 而且加深是**不可逆**的那个方向。
                    #
                    # 范本序列 500 → 500 → 500 → **200** pm 最后一步也是变浅。
                    same_depth += 1
                    entry["same_depth_try"] = same_depth
                    if same_depth >= int(wf.poke_same_depth_retries):
                        prev_nm = depth_nm
                        depth_nm = max(depth_nm * wf.poke_shallow_frac,
                                       wf.poke_depth_min_nm)
                        entry["next_depth_nm"] = depth_nm
                        if depth_nm >= prev_nm:
                            # 已经贴着地板了,再浅没有数据支持 —— **说出来**,
                            # 别让「变浅了」这句话在日志里变成一句假话。
                            entry["note"] = (
                                f"同一深度换了 {same_depth} 个地方都不圆,"
                                f"但已在最浅档 {depth_nm * 1000:.0f} pm —— 不再变浅,"
                                f"继续换地方")
                            _decide(entry, "floor", tries=same_depth,
                                    depth_pm=depth_nm * 1e3)
                        else:
                            entry["note"] = (
                                f"同一深度换了 {same_depth} 个地方都不圆 —— "
                                f"接触上了但搬多了,变浅到 {depth_nm * 1000:.0f} pm"
                                f"(原 {prev_nm * 1000:.0f} pm)")
                            _decide(entry, "shallower", tries=same_depth,
                                    depth_pm=prev_nm * 1e3,
                                    next_depth_pm=depth_nm * 1e3)
                        same_depth = 0
                    else:
                        entry["note"] = (f"扎上了但不够圆 —— 换个地方,仍用 "
                                         f"{depth_nm * 1000:.0f} pm")
                        _decide(entry, "same_depth", depth_pm=depth_nm * 1e3)
                    executor.set_partial(f"{prefix}:same_depth", same_depth)
                    unround += 1
                    if unround >= int(wf.poke_unround_streak_to_pulse):
                        out["needs_pulse"] = True
                        out["reason"] = (
                            f"连续 {unround} 针都做不出单峰圆簇 —— "
                            f"退回去打脉冲重修一次再来扎。"
                            f"(一直扎下去是拿位置的问题当形状的问题解。)")
                        _say("back_to_pulse", tries=unround)
                        _stash_poke(executor, prefix, log, used, pokes, stage,
                                    critical_nm, repeats)
                        break
                    _stash_poke(executor, prefix, log, used, pokes, stage,
                                critical_nm, repeats)
                    continue
                # 单峰且圆 → 换思路,从浅及深找临界。
                unround = 0          # 连续不圆的账清零
                entry["note"] = "单峰且圆 — 转入临界浅扎"
                _decide(entry, "stage_switch",
                        next_depth_pm=float(wf.critical_start_pm))
                stage = "critical"
                critical_nm = None
                _stash_poke(executor, prefix, log, used, pokes, stage,
                            critical_nm, repeats)
                continue

            # D2 每针均采图，成功后将深度恢复起步值重新搜索。
            # 连续成功次数统计单峰圆簇，不能用同一深度的重复次数代替质量验证。
            cl = yield from _cluster_look(executor, wf, step_prefix=sp, x=x, y=y)
            entry["cluster"] = cl
            entry["z_verdict"] = verdict          # 留档,不再分支
            if cl.get("ok"):
                last_cluster = cl

            if cl.get("ok") and cl.get("is_round") is None:
                # 图上没有大到能判的东西 ⇒ 这一针没扎上,加一级再来。
                # ⚠️ 三态:这不等于「表面什么都没有」,只等于「没有大到能判的」。
                nxt = (depth_nm * 1000.0 + wf.critical_step_pm) / 1000.0
                if nxt > wf.poke_depth_max_nm:
                    out["reason"] = (f"浅扎从 {wf.critical_start_pm:.0f} pm 一路加到"
                                     f"上限都没扎出能判的簇")
                    break
                prev_nm = depth_nm
                depth_nm = nxt
                entry["next_depth_nm"] = nxt
                entry["note"] = (f"扫图上没有大到能判的簇(面积 {cl.get('area_px')} px)"
                                 f" —— 加到 {nxt * 1000:.0f} pm")
                _decide(entry, "deeper", depth_pm=prev_nm * 1e3,
                        next_depth_pm=nxt * 1e3)
                _stash_poke(executor, prefix, log, used, pokes, stage,
                            critical_nm, repeats, depth_nm=depth_nm,
                            in_critical=_in_critical_search)
                continue

            if not cl.get("ok"):
                # 扫不成 / 分析不成 —— **不是**「没扎上」,也不该动深度。
                entry["note"] = f"簇图判不了({cl.get('reason')})—— 换个地方再扎"
                _stash_poke(executor, prefix, log, used, pokes, stage,
                            critical_nm, repeats, depth_nm=depth_nm,
                            in_critical=_in_critical_search)
                continue

            # 扎上了,而且图判得动。记下这一针用的深度(留档,不锁)。
            critical_nm = depth_nm
            good = (cl.get("is_round") is True
                    and cl.get("double_tip") is not True)
            if good:
                repeats += 1
                unround = 0
                entry["note"] = (f"{depth_nm * 1000:.0f} pm 扎出单峰圆簇"
                                 f"(连续第 {repeats} 次)")
                # ``need`` 必须是**收工判据真正用的那个字段** —— 下面
                # ``repeats >= int(wf.critical_repeat_n)`` 就是它。写成别的名字,
                # 旁白会一直报一个不决定任何事情的数(而且没人会去核)。
                _decide(entry, "round", depth_pm=depth_nm * 1e3,
                        streak=repeats, need=int(wf.critical_repeat_n))
            else:
                # 连续性断了 —— 这一针不算数,计数归零。
                # 「判不了」也归零:收工要的是**连续 N 次确实圆**,
                # 而不是「连续 N 次没被否掉」。
                repeats = 0
                why = (cl.get("roundness_undecidable")
                       or f"等效轴比 {cl.get('axis_ratio')}")
                entry["note"] = f"{depth_nm * 1000:.0f} pm 扎上了但不达标({why})"
                unround += 1
                if unround >= int(wf.poke_unround_streak_to_pulse):
                    out["needs_pulse"] = True
                    out["reason"] = (
                        f"连续 {unround} 针都做不出单峰圆簇 —— "
                        f"退回去打脉冲重修一次再来扎。")
                    _say("back_to_pulse", tries=unround)
                    _stash_poke(executor, prefix, log, used, pokes, stage,
                                critical_nm, repeats, depth_nm=depth_nm,
                                in_critical=_in_critical_search)
                    break

            # **每次都从头找。** 针尖已经变了,上一次的临界深度对它不作数。
            depth_nm = wf.critical_start_pm / 1000.0
            entry["next_depth_nm"] = depth_nm

            _stash_poke(executor, prefix, log, used, pokes, stage, critical_nm,
                        repeats, depth_nm=depth_nm,
                        in_critical=_in_critical_search)

            if good and repeats >= int(wf.critical_repeat_n):
                # ``double_tip is not True`` —— **判不了不拦收工**,与这条流程
                # 对锐度的既定做法一致(「锐度判不了不拦,只有明确量到 blunt 才拦」)。
                # 但拦不拦是一回事,**说不说**是另一回事。
                out["refined"] = True
                mt = cl.get("double_tip")
                single = ("单峰" if mt is False
                          else "多针尖这一项**判不了**(不拦收工)")
                out["reason"] = (f"连续 {repeats} 次扎出{single}且圆的簇"
                                 f"(最后一次深度 {float(critical_nm) * 1000:.0f} pm;"
                                 f"每次都从 {wf.critical_start_pm:.0f} pm 重新找 ——"
                                 f"针尖每扎一次就变一次,那个深度不是常数)")
                if mt is None and cl.get("multi_tip_undecidable"):
                    out["multi_tip_undecidable"] = cl["multi_tip_undecidable"]
                break
            continue

        if pokes >= budget and "reason" not in out:
            out["reason"] = f"扎针次数用满预算 {budget}"
        break

    # 收尾:把反馈开回来。
    #
    # **不能写成 try/finally**:executor 中止时是从遍历这个生成器的 for 循环里
    # ``return`` 的,生成器随后被 close(),而在 finally 里 yield 会让 Python 抛
    # ``RuntimeError: generator ignored GeneratorExit``。ShapeTipOnSurface 同款
    # 结构(收尾步骤在循环之后)也是这个原因。
    #
    # 这一步是**兜底而不是唯一保障**:每一次扎针都带 ``restore_feedback=True``,
    # 固件在每次 shaper 流程末尾自己就把反馈收回去了;abort 路径另有
    # ``_ABORT_SAFE_WRITES`` 白名单与 E_STOP 的 ZCtrl_Withdraw。
    #
    # 参数名是 ``enable`` —— 曾写成 ``on``,validate 直接判失败,而 optional=True
    # 把失败吞掉,于是反馈没恢复且无人知道。
    yield CompositeStep(
        step_id=f"{prefix}:restore_feedback", skill_name="ZControllerOnOff",
        params={"enable": True},
        optional=True, checkpoint_after=False, tags=("finalize",))

    executor.set_partial(f"{prefix}:unround", unround)
    out.update({"pokes": pokes, "log": log, "used_spots": used,
                "stage": stage, "critical_depth_nm": critical_nm,
                "repeats": repeats, "unround_streak": unround,
                "cluster": last_cluster})
    # 这一相的结论 —— 达标没有、扎了多少针。此前只在报文里。
    _say("poke_result", refined=bool(out.get("refined")), pokes=pokes,
         reason=str(out.get("reason") or ""))
    return out


def _stash_poke(executor: GraphExecutor, prefix: str, log, used, pokes, stage,
                critical_nm, repeats, *, depth_nm=None, in_critical=None) -> None:
    executor.set_partial(f"{prefix}:log", list(log))
    executor.set_partial(f"{prefix}:used", [list(p) for p in used])
    executor.set_partial(f"{prefix}:pokes", int(pokes))
    executor.set_partial(f"{prefix}:stage", str(stage))
    executor.set_partial(f"{prefix}:critical_nm", critical_nm)
    executor.set_partial(f"{prefix}:repeats", int(repeats))
    # 续跑时要接着上次的深度往下走,不能从起步值重来一遍。
    if depth_nm is not None:
        executor.set_partial(f"{prefix}:depth_nm", float(depth_nm))
    if in_critical is not None:
        executor.set_partial(f"{prefix}:in_critical_search", bool(in_critical))


__all__ = ["pulse_phase", "verify_phase", "level_phase", "poke_phase",
           "scan_at_params"]
