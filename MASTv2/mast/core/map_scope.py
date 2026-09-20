"""扫描地图的访问入口 —— 让 skill 层也读得到「这片表面已经发生过什么」。

在此之前地图只有 agent 工具层(``meta_tools``)读得到:storage 由 runtime 注入进
agent 的 context dict,而 skill 拿到的 ``ExecutionContext`` 里只有连接池、仪器状
态和技能注册表。后果是任何**在 composite 内部**决定「下一个动作落在哪」的流程,
都只能靠自己在内存里记这一轮用过的点 —— 冷启动看不见上一轮的坑,第二次调用会
把针尖扎回同一个地方(``ShapeTipOnSurface`` 的 ``excluded_spots`` 正是如此)。

修针流程每打一发脉冲就要换一个地方,而「哪里还干净」全写在地图里,所以这一层
必须存在。

两个函数各自 fail-soft:读不到实验记录返回空表而不是抛异常。**空表的含义是
「不知道这片表面发生过什么」,不是「这片表面是干净的」** —— 调用方据此决定要不
要动手时必须把这个区别说出来。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def analysis_config(state: Any = None, *, safety: Any = None,
                    frame_size_m: float | None = None):
    """本机的 ``AnalysisConfig`` —— 由仪器 profile + 安全上限 + 实时扫描框拼出。

    避让半径、进针是否伤表面、选点策略全部来自 ``instrument_profile``:同一个物理
    量只有一个来源,否则 agent 看到的地图和用户看到的会各自漂移。

    永不抛异常:任何一步失败,调用方仍拿到一份用 spec 默认值拼出的可用配置。
    """
    from mast.core import instrument_profile as ip
    from mast.io.map_analysis import AnalysisConfig

    def _nm(key: str, fallback: float) -> float:
        v = ip.get_config(key, None)
        try:
            return float(v) * 1e-9 if v is not None else fallback
        except (TypeError, ValueError):
            return fallback

    def _nm_unless_set(key: str, fallback: float) -> float:
        """用户**显式设过**就用他的;否则用 *fallback*,**不要** spec 出厂值。

        ⚠️ 为什么不能直接用 ``_nm``:``get_config`` 的回退顺序是
        「档案值 → ``_CONFIG_SPEC`` 的出厂默认 → 调用方给的默认」。
        也就是说 spec 里那个静态出厂值会**把调用方的默认整个遮蔽掉** ——
        它永远轮不到。

        静态出厂默认可能遮蔽按硬件能力计算的条件默认，且不会报错。
        必须检查档案是否显式登记该键，不能凭配置解析结果判断用户是否设过。

        这类「静态表里的默认压过条件默认」只能靠**问档案里到底设没设过**来解:
        ``get_profile()`` 只含用户真的写过的键。
        """
        try:
            explicit = ip.get_profile() or {}
        except Exception:  # noqa: BLE001
            explicit = {}
        if key not in explicit:
            return fallback
        try:
            v = explicit[key]
            return float(v) * 1e-9 if v is not None else fallback
        except (TypeError, ValueError):
            return fallback

    half = 1.5e-6
    try:
        xy_max = getattr(safety, "xy_max_m", None) if safety is not None else None
        if xy_max:
            half = abs(float(xy_max))
    except Exception:  # noqa: BLE001
        pass

    # 候选路线的间距要贴着用户实际在用的扫描框 —— 按 100 nm 帧排的路线,
    # 放到 1 µm 的巡览上是错的。
    frame = 100e-9
    if frame_size_m and frame_size_m > 0:
        frame = float(frame_size_m)
    else:
        try:
            snap = state.snapshot() if state is not None else None
            w = getattr(snap, "scan_width_m", None) if snap is not None else None
            if w and 1e-10 < float(w) < half * 2:
                frame = float(w)
        except Exception:  # noqa: BLE001
            pass

    def _spacing() -> float:
        try:
            v = float(ip.get_config("scan_spacing_factor", 1.2))
        except (TypeError, ValueError):
            return 1.2
        # 低于 1 会让相邻帧重叠，那不是「更分散」的反面，是把同一块地方扫两遍。
        return v if v >= 1.0 else 1.2

    return AnalysisConfig(
        piezo_half_range_m=half,
        frame_size_m=frame,
        tip_shape_r_m=_nm("avoid_radius_tip_shape_nm", 30e-9),
        # 脉冲避让半径与中心区大小共同决定可用落点数量。
        # 半径过大可能使任何障碍都覆盖整个候选区域，造成反复换区而无可用点。
        # 修改配置时应同时核对硬件行程、格距和障碍占用，不能只增加避让半径。
        pulse_r_m=_nm_unless_set("avoid_radius_pulse_nm", 200e-9),
        crash_r_m=_nm("avoid_radius_crash_nm", 150e-9),
        approach_r_m=_nm("avoid_radius_approach_nm", 200e-9),
        approach_damages=ip.approach_damages_surface(),
        strategy=ip.get_scan_path_strategy(),
        # 间距因子过去从没被传过 —— AnalysisConfig 的默认 1.2 就是「尽可能挨着
        # 排」，而用户要的是更分散。环宽跟着走：只拉开环上的点距而不拉
        # 开环间距，只会把「挨着排」从一个方向换到另一个方向。
        point_spacing_factor=_spacing(),
        ring_width_factor=_spacing(),
        has_xy_coarse_motion=ip.has_xy_coarse_motion(),
        # 有 XY 粗动 ⇒ 只用压电范围最中间的 **1200 nm**(±600),2026-08-17 起。
        #
        # 它与上面的脉冲避让半径仍然是**一对**,只是配对的判据变了:
        #
        #   旧:两者相等(500/500)⇒「一发就盖满」⇒ 被迫换区。
        #      听起来对,实际是区里只剩一个落点 ⇒ 一个盘就归零 ⇒ 死锁。
        #   新:**中心区要放得下好几个落点**。±600 配格距 400 ⇒ 3×3 = 9 个。
        #      「只在中间」仍然成立(±600 只占 ±1219 的一半),
        #      而挡掉一个还剩八个 —— 换区照旧会发生,只是发生在**用完之后**,
        #      不是发生在**第一发之前**。
        #
        # ⇒ 改这两个数中的任何一个,都要回答:**改完之后中心区里还剩几个落点?**
        #   答案是 1 或 0 的话,那不是一条约束,是一个死锁。
        #   ``nearest_clean_from`` 里有一道兜底(区装不下一个落点就退回压电范围),
        #   但那是保命的,不该被当成正常工作点。
        #
        # 没有 XY 粗动 ⇒ None(不设中心区,整片表面都要用上)。
        center_zone_side_m=(_nm_unless_set("center_zone_side_nm", 1200e-9)
                            if ip.has_xy_coarse_motion() else None),
    )


def marker_rows() -> "tuple[list, bool]":
    """当前实验/样品的**原始**地图行 —— ``(rows, available)``，**跨所有代次**。

    ``load_markers`` 与粗动大地图共用这一份取数：前者按代次过滤再转成 marker，
    后者要的是原始行（站点是跨代次累加的开环里程表）。**两份取数迟早只有一份对**，
    所以抽在这里。

    ``available`` 为 False = **读不到记录**（没有活动实验 / 存储不可用），
    不是「这里什么都没发生过」。这两件事在几何上无法区分，必须一路传给调用方。
    """
    try:
        from mast.logging.experiment_log import get_active_log
        log = get_active_log()
        storage = getattr(log, "_storage", None) if log is not None else None
        if storage is None:
            return [], False
        rows = storage.get_markers(
            getattr(log, "current_experiment_id", None),
            getattr(log, "current_sample_id", None)) or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("读扫描地图失败(按「不知道」处理): %s", exc)
        return [], False
    return list(rows), True


def load_markers(*, all_epochs: bool = False) -> "tuple[list, int, bool]":
    """当前坐标代次的地图标记 —— ``(markers, epoch, available)``。

    ``available`` 为 False 表示**读不到记录**(没有活动实验、存储不可用),此时
    ``markers`` 是空表。空表和「这片表面确实干净」在几何上无法区分,所以这个布尔
    值必须一路传到调用方:在读不到历史的情况下往表面打 10 V 脉冲,和在确认干净的
    地方打,是两件不同的事。

    只给当前代次:横向粗动之后旧坐标指的是另一片表面,混代次会把新鲜表面标成
    已经用过的(见 ``map_analysis`` 的 coord_epoch)。
    """
    from mast.io.exp_map import markers_from_rows
    from mast.io.map_analysis import current_epoch_of, filter_epoch

    rows, ok = marker_rows()
    if not ok:
        return [], 0, False

    epoch = current_epoch_of(rows)
    live = rows if all_epochs else filter_epoch(rows, epoch)
    return list(markers_from_rows(live)), epoch, True


def record_damage_marker(x_m: float, y_m: float, *, kind: str,
                         label: str = "", skill_name: str = "") -> "int | None":
    """写入受影响落点；成功返回行 id，失败返回 None。

    与 load_markers 共用 storage 和 experiment_id / sample_id 作用域，
    保证选点器能够读取这里写入的标记。coord_epoch 由 storage.log_marker
    在写事务中统一确定，不在此处另算。

    kind 必须属于 DAMAGE_KINDS，否则避让几何无法取得半径；拒绝无效 kind
    并返回 None，让调用方记录失败。
    """
    from mast.io.map_analysis import DAMAGE_KINDS

    if kind not in DAMAGE_KINDS:
        logger.warning("record_damage_marker: 未知 kind %r —— 那条标记不会避让任何东西", kind)
        return None
    try:
        from mast.logging.experiment_log import get_active_log

        log = get_active_log()
        storage = getattr(log, "_storage", None) if log is not None else None
        if storage is None:
            return None
        return int(storage.log_marker(
            kind=kind, x_m=float(x_m), y_m=float(y_m),
            label=label or f"{kind} 损伤",
            skill_name=skill_name or "", source="skill",
            experiment_id=getattr(log, "current_experiment_id", None),
            sample_id=getattr(log, "current_sample_id", None),
            meta={"pos_src": "chosen_spot"}))
    except Exception as exc:  # noqa: BLE001 — 落库失败不许弄坏正在跑的实验
        logger.debug("写损伤标记失败(%s): %s", kind, exc)
        return None


def crash_memory_markers() -> "tuple[list, int]":
    """本进程记着的撞针点,借用地图标记的形状 —— ``(markers, unlocated)``。

    ## 为什么需要第二个来源

    撞针有**两份**记忆,而它们的失效方式正好互补:

    * **地图标记**(``kind="crash"``,由 ``runtime._record_map_marker`` 的显式分支
      落库)—— 跨重启、带代次,是权威的那一份;
    * **``tip_crash_tracker``**(进程内,30 min TTL)—— 撞针当场就有。

    落库那一步是 fire-and-forget 且整段包在 ``except`` 里(``runtime`` 里的
    ``_log_swallowed``),没有活动实验时 ``storage is None`` 会直接 return。
    也就是说存在一段时间、以及一整类情形,**撞针已经发生、tracker 已经记下、
    地图上却什么都没有**。这时选点器只问地图,就会把针尖送回自己刚炸出来的坑,
    而且零报错 —— 地图如实回答了「我这儿没有记录」,只是没人问另一个知道的人。

    这个函数就是去问那一个。它**只读**:不落库、不补写标记。补写会让 tracker 变成
    第二个生产者,和 recorder 抢同一张表 —— 本仓记过「同一个动作 N 份实现,往往只
    有一份对」。

    ## 代次

    ``coord_epoch=None``(从未持久化过,与实时框/针尖/计划覆盖层同款)。这不是漏填:
    横向粗动会把 tracker 整个清空(``coarse_move_effects`` 第 2 条),所以里面剩下的
    东西**必然**属于当前代次,不存在需要标注的旧代次坐标。

    ``unlocated`` 是记到了、但坐标不知道的撞针次数 —— 它避不开,调用方必须说出来。
    """
    from mast.io.exp_map import MapMarker

    try:
        from mast.core.tip_crash_tracker import get_tip_crash_tracker

        located, unlocated = get_tip_crash_tracker().crash_points()
    except Exception as exc:  # noqa: BLE001
        # 读不到这一路来源 ≠ 没撞过。调用方按「少了一个来源」处理,不是按「干净」。
        logger.debug("读撞针记忆失败(按「这一路来源没答上」处理): %s", exc)
        return [], 0

    markers = [
        MapMarker(
            kind="crash",          # 与落库那条显式分支同一个 kind —— 避让半径
            x_m=float(x),          # 因此走的是同一个 DAMAGE_KINDS["crash"]。
            y_m=float(y),
            label="撞针(本进程记忆)",
            status="failed",
            source="live",         # 未持久化,和实时框/针尖一样
            meta={"crash_count": int(n), "from": "tip_crash_tracker"},
        )
        for (x, y, n) in located
    ]
    return markers, int(unlocated)


__all__ = ["analysis_config", "crash_memory_markers", "load_markers",
           "record_damage_marker"]
