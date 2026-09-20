"""就近找一块还没被弄脏的表面 —— 「打一次换一个地方」的那个「地方」。

修针时每打一发脉冲、每扎一次针都要换位置:原地再来一次,读到的是上一次留下的坑
的行为,判据就废了。但换到哪里不是随便挑 —— 脉冲会把材料溅到周围(默认 150 nm
半径),扎针尖留下的簇小一些(30 nm),这些半径连同历史上每一次脉冲/撞针/进针的坐
标都记在扫描地图里。

与 ``get_next_scan_position`` 是两个问题:那个走的是从原点出发的固定巡览路线,
回答「第 N 张图该放哪」;这个回答「我刚在这儿动过手,最近的还能动手的地方在哪」。
一次大跨度移动会重新激起压电蠕变,而修针一轮要打十几次 —— 省下的不是时间,是后
面几帧图的质量。

读不到实验记录时**照样给点,但如实说明**:``map_known=false`` 意味着"不知道这片
表面发生过什么",不等于"这片表面是干净的"。这个区别必须一路传到报告里。

撞针的历史有**两个**来源,这里两个都问(见 ``map_scope.crash_memory_markers``):
落库的地图标记是权威的那一份,而它落库那一步会断(没有活动实验、异常被吞),断的
时候进程内的 ``tip_crash_tracker`` 往往是唯一还记得刚才撞过的人。两个来源合成
**同一种** ``kind="crash"``,因此共用同一个避让半径,不产生第二套约定。
"""
from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.builtins._tip_xy import read_tip_xy

logger = logging.getLogger(__name__)

#: 用途 → 该用哪个避让半径。与扫描地图的 DAMAGE_KINDS 一一对应,所以「打脉冲要
#: 避开多远」在地图上画的圈和这里选点用的圈永远是同一个数。
_PURPOSE_RADIUS = {
    "pulse": "pulse_r_m",
    "tip_shape": "tip_shape_r_m",
}

# 配置与仪器量程需要核对。仪器端读不到时必须保留 unknown，
# 不能把缺少读数当成配置已通过验证，否则选点器可能提出不可到达的目标。
PIEZO_SRC_CONFIG = "config"
PIEZO_SRC_INSTRUMENT_TIGHTER = "instrument(比配置小,已收紧)"
PIEZO_SRC_INSTRUMENT_LOOSER = "instrument(配置更保守,沿用配置)"


def parse_spots(raw: str) -> list[tuple[float, float]]:
    """``"x1,y1;x2,y2"`` → 坐标表。与 FindFlatRegion 的 exclude_used_spots 同款
    格式 —— 两个技能对「已经用过的点」用不同的写法只会让调用方写错。"""
    out: list[tuple[float, float]] = []
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return out


class FindCleanSpot(BaseSkill):
    """离当前针尖位置最近、且历史上没被破坏过的落点。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="FindCleanSpot",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "离针尖最近、而且**没有**被破坏过的那个落点，供下一次脉冲或扎针用。"
                "它读的是已记录的地图 marker，**外加**本进程自己的撞针记忆 —— "
                "绝不从一张图上猜。没有实验记录可查时返回 map_known=false，"
                "那句话的意思是「不知道」，不是「干净」；crash_memory_unlocated>0 "
                "表示**确实**知道发生过一次撞针、但那个位置谁也读不出来，"
                "于是它躲不开，返回的落点带着这份风险。"
            ),
            parameters=[
                ParameterSpec(
                    name="purpose", type="str",
                    description="这个落点拿来做什么：'pulse'（偏压脉冲，"
                                "污染半径更大）或 'tip_shape'"
                                "（扎针，半径小一些）。",
                    required=False, default="pulse",
                    allowed_values=["pulse", "tip_shape"]),
                ParameterSpec(
                    name="exclude_spots", type="str",
                    description="**这一轮**已经用掉的落点，'x1,y1;x2,y2'，"
                                "单位米。刚刚写下的 marker 可能还没落到库里，"
                                "所以这份清单由调用方自己拿着。",
                    required=False, default=""),
                ParameterSpec(
                    name="from_x_m", type="float", unit="m",
                    description="搜索的起点。不给就取针尖的实时位置。",
                    required=False, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(
                    name="from_y_m", type="float", unit="m",
                    description="搜索的起点。不给就取针尖的实时位置。",
                    required=False, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(
                    name="max_distance_m", type="float", unit="m",
                    description="离起点比这个还远的落点一律忽略。",
                    required=False, min_value=1e-9, max_value=3e-6),
                ParameterSpec(
                    name="count", type="int",
                    description="返回几个候选（由近到远排）。",
                    required=False, default=8, min_value=1, max_value=64),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["map", "position", "tip", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from mast.core.map_scope import (
            analysis_config,
            crash_memory_markers,
            load_markers,
        )
        from mast.io.map_analysis import nearest_clean_from

        calls: list = []
        purpose = str(params.get("purpose") or "pulse")
        count = int(params.get("count") or 8)
        exclude = parse_spots(str(params.get("exclude_spots") or ""))

        x0 = params.get("from_x_m")
        y0 = params.get("from_y_m")
        origin_src = "explicit"
        if x0 is None or y0 is None:
            pos = read_tip_xy(context)
            if pos is None:
                return SkillResult(
                    skill_name="FindCleanSpot", success=False,
                    error="Cannot read the tip position (FolMe_XYPosGet), and no "
                          "from_x_m/from_y_m was given — there is no origin to "
                          "search from.",
                    nanonis_calls=calls)
            x0, y0 = pos
            origin_src = "live_tip"
        x0, y0 = float(x0), float(y0)

        cfg = analysis_config(getattr(context, "state", None))

        # 压电量程会随温度、扫描器与标定改变，应读取当前值。
        # 可读时以配置与硬件范围的较保守边界约束选点；缺少硬件读数时明确报告不确定性。
        piezo_src = PIEZO_SRC_CONFIG
        rec_range = context.safe_call("Piezo_RangeGet")
        calls.append(rec_range)
        if not rec_range.error:
            # 复用协议解码器读取 (error, raw, parsed) 中 parsed 内的数值。
            # 只扫描顶层会漏掉数据，并使范围检查错误地退回配置。
            from mast.skills.builtins.readback import _decode_nanonis

            decoded = _decode_nanonis(getattr(rec_range, "return_value", None))
            vals = [float(v) for v in (decoded or ())
                    if isinstance(v, (int, float))] if isinstance(
                        decoded, (list, tuple)) else []
            if len(vals) >= 2:
                # Piezo_RangeGet 回**全程**,不是半程。
                inst_half = min(abs(float(vals[0])), abs(float(vals[1]))) / 2.0
                if inst_half > 0:
                    from dataclasses import replace as _replace

                    if inst_half < float(cfg.piezo_half_range_m):
                        cfg = _replace(cfg, piezo_half_range_m=inst_half)
                        piezo_src = PIEZO_SRC_INSTRUMENT_TIGHTER
                    else:
                        piezo_src = PIEZO_SRC_INSTRUMENT_LOOSER
        spot_r = float(getattr(cfg, _PURPOSE_RADIUS.get(purpose, "pulse_r_m")))
        markers, epoch, map_known = load_markers()
        # 撞针的第二个来源。地图那条链会断(没有活动实验 / 落库这一步整段被吞),
        # 断的时候 tracker 往往是唯一还记得的人 —— 见 ``crash_memory_markers``。
        # 合进来的是**同一个 kind="crash"**,所以避让半径依然由
        # ``DAMAGE_KINDS["crash"] → crash_r_m`` 给,这里不产生第二套半径。
        crash_mem, crash_unlocated = crash_memory_markers()
        # 「谁答上了」的名单。空表 = **没有任何一个来源知道这片表面的历史**,
        # 那与「两个来源都说干净」在几何上完全一样,只能靠这里说出来。
        sources = (["map"] if map_known else []) + (
            ["crash_memory"] if crash_mem else [])

        max_d = (float(params["max_distance_m"])
                 if params.get("max_distance_m") else None)
        # 打脉冲的落点上**不扫图**,所以不减帧边距(``nearest_clean_from`` 的
        # 注释一直这么写着,代码却一律减)。在 ±250 nm 的中心区里,那 50 nm
        # 白白吃掉五分之一的可用区。
        frame_m = 0.0 if purpose == "pulse" else None
        spots = nearest_clean_from(
            markers + crash_mem, cfg, x0, y0, spot_r_m=spot_r, count=count,
            exclude=exclude, max_distance_m=max_d, frame_m=frame_m)

        # ── 针尖站在可用区**外**:从区中心重搜,而不是报「表面用完」──────
        #
        # 候选必须落在 effective_half_range_m 的可用区内。针尖位于区外时，
        # 以它为中心搜索可能全部越界，不能由此推断表面已经用完。
        # 从区中心重搜；调用方随后按返回落点执行 MoveToXY。
        recentred_from = None
        if not spots:
            reach = float(cfg.effective_half_range_m)
            if abs(x0) > reach or abs(y0) > reach:
                recentred_from = (x0, y0)
                spots = nearest_clean_from(
                    markers + crash_mem, cfg, 0.0, 0.0, spot_r_m=spot_r,
                    count=count, exclude=exclude, max_distance_m=None,
                    frame_m=frame_m)

        # 「谁答上了」的账,成功和失败两条路都要带 —— 一次「没有干净点」同样要
        # 说得出是地图说的还是撞针记忆说的。
        provenance = {
            "map_known": map_known,
            "coord_epoch": epoch,
            "spot_radius_m": spot_r,
            "markers_seen": len(markers),
            # 本进程记着、且**已经参与避让**的撞针点数。
            "crash_memory_points": len(crash_mem),
            # 记到了却没有坐标的撞针 —— 几何上避不开,必须明说。
            "crash_memory_unlocated": crash_unlocated,
            "avoidance_sources": sources,
            # 换过原点就必须说 —— 悄悄把「从针尖处找」换成「从区中心找」,
            # 调用方会以为返回的落点就在手边,而它可能在一微米之外。
            "recentred": recentred_from is not None,
            "recentred_from_x_m": recentred_from[0] if recentred_from else None,
            "recentred_from_y_m": recentred_from[1] if recentred_from else None,
            "effective_half_range_m": float(cfg.effective_half_range_m),
            # 这次用的压电半程是谁说的 —— 「读不到」不许伪装成「已核对」。
            "piezo_half_range_m": float(cfg.piezo_half_range_m),
            "piezo_range_source": piezo_src,
        }

        if not spots:
            # ── 说出**是谁挡的** ────────────────────────────────────────
            #
            # 报错中呈现 provenance，区分已用区域与几何约束造成的阻挡，
            # 以免把不同原因统一解释为表面用完并反复换区。
            from mast.io.map_analysis import build_avoid_circles

            blockers = ""
            try:
                circles = build_avoid_circles(markers + crash_mem, cfg)
                r_eff = float(cfg.effective_half_range_m)
                # 只报**够得着区心的**那些 —— 远处的圈不是这次的原因。
                near = sorted(
                    ((c, (c.x_m ** 2 + c.y_m ** 2) ** 0.5) for c in circles),
                    key=lambda t: t[1])[:3]
                if near:
                    bits = [
                        f"{c.kind} 在 ({c.x_m * 1e9:.0f}, {c.y_m * 1e9:.0f}) nm"
                        f"(盘半径 {c.radius_m * 1e9:.0f} nm,距区心 {d * 1e9:.0f} nm,"
                        f"要 {(c.radius_m + spot_r) * 1e9:.0f} nm 才让得开)"
                        for c, d in near]
                    blockers = ";挡路的:" + "、".join(bits)
                elif r_eff < spot_r:
                    blockers = (f";⚠️ 可用区半程 {r_eff * 1e9:.0f} nm **比落点净空 "
                                f"{spot_r * 1e9:.0f} nm 还小** —— 这个区放不下一个点,"
                                f"粗动换区解决不了")
                else:
                    blockers = ";地图和撞针记忆里都没有挡路的东西 —— 看几何/配置"
            except Exception:  # noqa: BLE001 — 报错的修饰不许把报错本身弄坏
                pass

            return SkillResult(
                skill_name="FindCleanSpot", success=False,
                error=(f"No undamaged spot within reach for '{purpose}' "
                       f"(avoidance radius {spot_r * 1e9:.0f} nm)"
                       + (f",连从可用区中心重搜也没有"
                          f"(针尖当时在 ({recentred_from[0] * 1e9:.0f}, "
                          f"{recentred_from[1] * 1e9:.0f}) nm,已在可用区 "
                          f"±{cfg.effective_half_range_m * 1e9:.0f} nm 之外)"
                          if recentred_from else "")
                       + blockers
                       + ". This patch of surface is spent — relocate with "
                         "RelocateCoarseXY, or work on a fresh sample area."),
                data={**provenance, "candidates": 0},
                nanonis_calls=calls)

        first = spots[0]
        note = (f"距当前位置 {first.distance_m * 1e9:.0f} nm"
                if origin_src == "live_tip" else
                f"距给定原点 {first.distance_m * 1e9:.0f} nm")
        if not map_known:
            # 读不到地图**不是**「干净」。区别在于:此刻还有没有别人知道。
            note += ("；**读不到实验记录**" + (
                f",但本进程记着的 {len(crash_mem)} 个撞针点已经避开"
                if crash_mem else
                ",本进程也没有撞针记忆 —— **无法确认此处是否干净**"))
        if crash_unlocated:
            # 记到了、坐标不知道 ⇒ 画不出圈 ⇒ 返回的这个点有可能就在上面。
            note += (f"；⚠️ 另有 {crash_unlocated} 次撞针**读不到坐标**,"
                     f"避让圈画不出来,这个落点无法保证不在其上")
        return SkillResult(
            skill_name="FindCleanSpot", success=True,
            data={
                "x_m": first.x_m,
                "y_m": first.y_m,
                "distance_m": first.distance_m,
                "candidates": [{"x_m": s.x_m, "y_m": s.y_m,
                                "distance_m": s.distance_m} for s in spots],
                "purpose": purpose,
                "origin_x_m": x0,
                "origin_y_m": y0,
                "origin_source": origin_src,
                # 「不知道」和「干净」不是一回事 —— 调用方必须能分辨。
                **provenance,
                "reason": note,
            },
            nanonis_calls=calls,
            summary=f"下一个落点 ({first.x_m * 1e9:.1f}, {first.y_m * 1e9:.1f}) nm — {note}")


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(FindCleanSpot, context_provider)
