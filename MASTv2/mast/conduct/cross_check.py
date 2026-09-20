"""判据③「换位置复测仍差」—— 跨点聚合,闭集三值。

设计文档:``docs/v2/design/`` 的「S1 修针循环 conduct 化」设计,D4 / §3.4 / §5.2
(文件名按「S1 修针循环」检索;引擎层注释里**一个样品名都不许出现**)。
本模块是**引擎层**:纯函数、零 IO、零样品名(见 :mod:`mast.conduct` 的分层铁律)。

## 它回答的是判据②答不了的那半句

跨点判据要求「表面起伏极大**且**换位置扫依旧」。前半句是单帧观察
(:func:`mast.vision.corrugation_gate.judge_corrugation`),**它分不开「针尖团簇」
与「表面台阶簇」** —— 分得开的是换个位置再测一次。所以:

* 单帧只出 ``high`` / ``normal`` / ``low`` / ``undecidable``,**没有 `bad_tip`**;
* ``bad_tip`` 只可能从这里出来,而且要 **judged 全票 + judged ≥ 2**。

## 规则:全票,不设投票阈值

======================================  =====================
``judged ≥ min_judged`` 且全部判坏        ``bad_tip``
``judged ≥ min_judged`` 且不是全部判坏    ``surface_feature``
``judged < min_judged``                 ``undecidable``
各点 ``coord_epoch`` 不一致或读不到       ``undecidable``(整批作废)
======================================  =====================

**为什么不设 M/N 投票阈值**:那会引入第三个需要标定的数,而我们连第一个
(起伏上限)都还没标。「judged 全票」是唯一不需要额外常数、又能同时表达
「一致」与「分歧」的规则。

judged 数量足够且全部判好时也返回 surface_feature；数量不足则返回 undecidable。
两者都不能返回 bad_tip：此处只回答跨点证据是否一致指向针尖问题。

## 一个点算判坏 / 判好 / 不计

* **判坏** = ``tip_ready is False`` **或** ``corrugation_verdict == "high"``
  **或** ``spectrum_hysteresis == "fired"``;
* **判好** = ``tip_ready is True`` **且** ``corrugation_verdict != "high"``。
  ⚠️ ``spectrum_hysteresis`` **不参与判好** —— 它 ``not_fired`` 是**零信息**,
  写进合取项就等于让「没看见」给「好」投票(设计 §4 陷阱 12)。
* 两者都不是 ⇒ 该点**不计入 judged**。``tip_ready is None``(判据①判不了)
  在这里既不投坏票也不投好票 —— 「读不到」不是一个值。

## coord_epoch:读不到就整批作废

粗动会 bump ``coord_epoch``,而粗动之后「同一个坐标」指的是另一块表面 ——
一批点里混进不同代次,这批证据就不再是「换位置复测同一片区域」。
**epoch 读不到(``None``)与不一致同样作废**:两个未知不构成「一致」,
而这里要证的恰恰是「中途没发生粗动」。要拿到章走
``mast.core.coord_epoch.read_current_epoch()`` 的权威查询,不要用
``map_scope.load_markers`` 那个从截断窗口推导出来的值(设计 §4 陷阱 15)。

## 选点的两条筛不在这里

「两两距离 ≥ 帧宽 × 1.5」与「``crash_count == 0``」是**采点时**的筛
(设计 D5)——它们要读地图和撞针记忆,是 IO。本模块只吃已经采好的点,
但把 ``crash_count`` 一路记下来,这样事后能查「这个结论是不是在自己刚炸出来的
坑上判的」。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Sequence

#: 闭集,**只有这三个值**。
CROSS_VERDICTS: tuple[str, ...] = ("bad_tip", "surface_feature", "undecidable")

#: 谱正反扫迟滞的四态(判据④,设计 D14)。``not_fired`` **不是「这个点好」**;
#: ``unrated`` = 阈值未标定(S4 R2 之前恒为此);``skipped`` = 前置筛关着。
SPECTRUM_STATES: tuple[str, ...] = ("fired", "not_fired", "unrated", "skipped")


@dataclass(frozen=True)
class PointVerdict:
    """一个复测点上拿到的全部证据。字段语义见设计 §3.4。"""

    #: 点位(米)。只用于在 ``reason`` 里点名,不参与判决。
    x_m: float | None = None
    y_m: float | None = None
    #: 坐标代次。``None`` = 读不到 —— **不是**「没变过」。
    coord_epoch: int | None = None
    #: ``judge_frame().usable``。``None`` = 没记。
    frame_usable: bool | None = None
    #: ``PreScanCheck.data["similarity"]``,旁证。
    similarity: float | None = None
    #: ``PreScanCheck.data["tip_ready"]`` —— **三态,``None`` 不折叠**。
    tip_ready: bool | None = None
    #: ``AssessFrameCorrugation.data["verdict"]``,四态;不认的值一律不计 judged。
    corrugation_verdict: str = "undecidable"
    #: 数**和它的口径**一起记 —— 事后换口径重标时要认得出哪些点是老口径量的。
    corrugation_value_pm: float | None = None
    corrugation_detrend: str = ""
    corrugation_statistic: str = ""
    #: 采点时已筛为 0;记下来是为了事后能查。
    crash_count: int = 0
    #: 判据④(:data:`SPECTRUM_STATES`)。只进「判坏」的析取,绝不进「判好」的合取。
    spectrum_hysteresis: str = "skipped"
    #: 该点是否因谱迟滞触发而省掉了扫帧。为 True 时 similarity / corrugation_*
    #: 全是 ``None``,**不是**「测了、正常」。
    frame_skipped_by_prescreen: bool = False
    #: 判不了时**为什么**(四类各指不同的下一步)。
    abstain_reason: str = ""
    #: ``FindCleanSpot`` 那一侧的地图可读性。``False`` = 读不到记录,几何上与
    #: 「表面干净」不可区分,必须一路传到报告。
    map_known: bool = True
    #: 人话点名用(例如 "P2");空则用序号。
    label: str = ""

    def display(self, index: int) -> str:
        name = (self.label or "").strip() or f"点{index + 1}"
        if self.x_m is not None and self.y_m is not None:
            try:
                return f"{name}({self.x_m * 1e9:+.0f},{self.y_m * 1e9:+.0f} nm)"
            except (TypeError, ValueError):  # pragma: no cover — 防御
                pass
        return name


@dataclass(frozen=True)
class CrossCheckVerdict:
    """跨点聚合的结论。``verdict`` 取值只在 :data:`CROSS_VERDICTS` 内。"""

    verdict: str
    n_points: int
    n_judged: int
    n_bad: int
    n_good: int
    #: 各点一致时的那个值;不一致或读不到时 ``None`` + verdict 强制 undecidable。
    coord_epoch: int | None
    #: 任一点地图读不到 ⇒ False,一路传进报告。
    map_known: bool
    #: 人话一句,**点名是哪些点、各自判了什么**。
    reason: str
    #: 原始证据,进 conduct_events 的 payload。
    points: tuple[PointVerdict, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        d = dict(asdict(self))
        d["points"] = [dict(asdict(p)) for p in self.points]
        return d


def _is_bad(p: PointVerdict) -> bool:
    """判坏 = 三条**析取**。任何一条成立就够了。"""
    return (p.tip_ready is False
            or p.corrugation_verdict == "high"
            or p.spectrum_hysteresis == "fired")


def _is_good(p: PointVerdict) -> bool:
    """判好 = 两条**合取**。``spectrum_hysteresis`` 刻意不在里面(陷阱 12)。"""
    from mast.vision.corrugation_gate import VERDICTS

    if p.tip_ready is not True:
        return False
    if p.corrugation_verdict not in VERDICTS:
        # 不认的起伏 verdict = 读不到,不是「不是 high 所以算好」。
        return False
    return p.corrugation_verdict != "high"


def _why(p: PointVerdict) -> str:
    bits: list[str] = []
    if p.frame_skipped_by_prescreen:
        bits.append("谱前置筛触发,省了扫帧")
    if p.tip_ready is False:
        bits.append("判据①不合格")
    elif p.tip_ready is True:
        bits.append("判据①合格")
    else:
        bits.append("判据①判不了")
    if p.corrugation_verdict != "undecidable":
        bits.append(f"起伏 {p.corrugation_verdict}")
    if p.spectrum_hysteresis == "fired":
        bits.append("谱正反扫迟滞触发")
    if p.abstain_reason:
        bits.append(str(p.abstain_reason))
    return "、".join(bits)


def aggregate_cross_points(points: "Sequence[PointVerdict]", *,
                           min_judged: int = 2) -> CrossCheckVerdict:
    """把 N 个复测点聚合成闭集三值。**纯函数**。

    ``min_judged`` 是「几个点才算一次换位置复测」。**小于 2 会被拒绝**:
    一个点不构成「换位置」,那时 ``bad_tip`` 就退化成单帧结论 —— 而单帧分不开
    针尖团簇与表面台阶簇,这正是本判据存在的理由。
    """
    try:
        floor = int(min_judged)
    except (TypeError, ValueError):
        raise ValueError(f"min_judged={min_judged!r} 不是整数") from None
    if floor < 2:
        raise ValueError(
            f"min_judged={floor} 小于 2 —— 一个点不构成「换位置复测」,"
            f"那样 bad_tip 就退化成单帧结论,而单帧分不开针尖与表面。"
            f"要更严可以调大,不能调小。")

    pts = tuple(points or ())
    n = len(pts)
    map_known = all(bool(p.map_known) for p in pts) if pts else True

    def out(kind: str, reason: str, *, epoch: "int | None",
            n_judged: int = 0, n_bad: int = 0, n_good: int = 0) -> CrossCheckVerdict:
        return CrossCheckVerdict(
            verdict=kind, n_points=n, n_judged=n_judged, n_bad=n_bad,
            n_good=n_good, coord_epoch=epoch, map_known=map_known,
            reason=reason, points=pts)

    if not pts:
        return out("undecidable", "一个复测点都没有 —— 判不了。", epoch=None)

    # ── 1. 坐标代次:读不到与不一致同样作废 ────────────────────────────
    unknown = [p.display(i) for i, p in enumerate(pts) if p.coord_epoch is None]
    if unknown:
        return out("undecidable",
                   f"整批作废:{'、'.join(unknown)} 的坐标代次读不到 —— "
                   f"读不到不等于「中途没粗动过」,而这批点要证的正是这件事。"
                   f"代次走 storage 的权威查询(read_current_epoch),"
                   f"不要用地图窗口推导出来的值。", epoch=None)
    epochs = {int(p.coord_epoch) for p in pts}          # type: ignore[arg-type]
    if len(epochs) > 1:
        named = "、".join(f"{p.display(i)}=代次 {p.coord_epoch}"
                          for i, p in enumerate(pts))
        return out("undecidable",
                   f"整批作废:各点坐标代次不一致({named})—— "
                   f"中途发生过粗动,粗动之后同一个坐标指的是另一块表面。",
                   epoch=None)
    epoch = next(iter(epochs))

    # ── 2. 逐点归类 ────────────────────────────────────────────────────
    bad: list[str] = []
    good: list[str] = []
    abstained: list[str] = []
    for i, p in enumerate(pts):
        who = f"{p.display(i)}({_why(p)})"
        if _is_bad(p):
            bad.append(who)
        elif _is_good(p):
            good.append(who)
        else:
            abstained.append(who)
    n_bad, n_good = len(bad), len(good)
    n_judged = n_bad + n_good
    tally = (f"{n} 个点:判坏 {n_bad}、判好 {n_good}、判不了 {len(abstained)}"
             f"(judged={n_judged},门槛 {floor})")
    detail = ";".join(
        part for part in (
            f"判坏 {'、'.join(bad)}" if bad else "",
            f"判好 {'、'.join(good)}" if good else "",
            f"判不了 {'、'.join(abstained)}" if abstained else "",
        ) if part)

    # ── 3. 闭集三值 ────────────────────────────────────────────────────
    if n_judged < floor:
        return out("undecidable",
                   f"{tally}。{detail}。judged 不够 —— 一个点不构成「换位置复测」,"
                   f"判不了不等于判好,也不等于判坏。",
                   epoch=epoch, n_judged=n_judged, n_bad=n_bad, n_good=n_good)
    if n_bad == n_judged:
        return out("bad_tip",
                   f"{tally}。{detail}。judged 全票判坏 ⇒ 换了位置仍然差,"
                   f"指向针尖而不是这一片表面。",
                   epoch=epoch, n_judged=n_judged, n_bad=n_bad, n_good=n_good)
    return out("surface_feature",
               f"{tally}。{detail}。judged 不是全票判坏 ⇒ 换个位置就不一样了,"
               f"这指向表面而不是针尖;不要据此去修针尖或换样品。",
               epoch=epoch, n_judged=n_judged, n_bad=n_bad, n_good=n_good)


# ── conduct ``kind="analysis"`` 步的适配层 ──────────────────────────────

_POINT_FIELDS = {f.name for f in fields(PointVerdict)}


def _as_point(raw: Any, index: int) -> PointVerdict:
    """一条点记录 → :class:`PointVerdict`。未知键忽略,缺失键用字段默认值。

    **不做类型强转以外的补齐**:``tip_ready`` 缺席就是 ``None``(判不了),
    绝不当成 True/False。
    """
    from mast.conduct.analyses import AnalysisError

    if isinstance(raw, PointVerdict):
        return raw
    if not isinstance(raw, dict):
        raise AnalysisError(f"第 {index + 1} 个点不是一条记录:{raw!r}")
    kw = {k: v for k, v in raw.items() if k in _POINT_FIELDS}
    for key in ("x_m", "y_m", "similarity", "corrugation_value_pm"):
        if kw.get(key) is not None:
            try:
                val = float(kw[key])
            except (TypeError, ValueError):
                raise AnalysisError(
                    f"第 {index + 1} 个点的 {key}={kw[key]!r} 不是数值") from None
            kw[key] = val if math.isfinite(val) else None
    if kw.get("coord_epoch") is not None:
        try:
            kw["coord_epoch"] = int(kw["coord_epoch"])
        except (TypeError, ValueError):
            raise AnalysisError(
                f"第 {index + 1} 个点的 coord_epoch={kw['coord_epoch']!r} "
                f"不是整数代次") from None
    if kw.get("crash_count") is not None:
        try:
            kw["crash_count"] = int(kw["crash_count"])
        except (TypeError, ValueError):
            kw["crash_count"] = 0
    for key in ("corrugation_verdict", "corrugation_detrend",
                "corrugation_statistic", "spectrum_hysteresis",
                "abstain_reason", "label"):
        if key in kw and kw[key] is not None:
            kw[key] = str(kw[key])

    # 三态/四态字段在这里**拒绝**不认的值。纯函数那一侧对不认的值是保守的
    # (既不判坏也不判好),但保守加沉默 = 一个接错的线永远查不出来:一个
    # ``"false"`` 字符串会让一票「判坏」悄悄变成「判不了」。边界要吵。
    for key in ("tip_ready", "frame_usable"):
        if kw.get(key) is not None and not isinstance(kw[key], bool):
            raise AnalysisError(
                f"第 {index + 1} 个点的 {key}={kw[key]!r} 不是真/假/缺席三态之一 ——"
                f"「读不到」要写成缺席或 null,不是一个字符串")
    for key in ("map_known", "frame_skipped_by_prescreen"):
        if key in kw and not isinstance(kw[key], bool):
            raise AnalysisError(
                f"第 {index + 1} 个点的 {key}={kw[key]!r} 不是布尔值")

    from mast.vision.corrugation_gate import VERDICTS

    cv = kw.get("corrugation_verdict")
    if cv is not None and cv not in VERDICTS:
        raise AnalysisError(
            f"第 {index + 1} 个点的 corrugation_verdict={cv!r} 不在词表 "
            f"{list(VERDICTS)} 内(注意判据②**不产出** 'bad_tip')")
    sh = kw.get("spectrum_hysteresis")
    if sh is not None and sh not in SPECTRUM_STATES:
        raise AnalysisError(
            f"第 {index + 1} 个点的 spectrum_hysteresis={sh!r} 不在词表 "
            f"{list(SPECTRUM_STATES)} 内")
    return PointVerdict(**kw)


def cross_check_analysis(params: dict) -> dict:
    """``kind="analysis"`` 步的入口 —— 注册名 ``aggregate_cross_points``。

    ``points`` 可以是一串 dict(binding 直接传上一步的 ``points``)或它的
    JSON 文本。``min_judged`` 可选,默认 2。
    """
    from mast.conduct.analyses import AnalysisError

    raw = params.get("points")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise AnalysisError(f"points 不是合法 JSON:{exc}") from None
    if raw is None:
        raise AnalysisError(
            "缺少参数 'points' —— 复测点由采点那一步给,分析步不发明点位。")
    if isinstance(raw, dict):
        raise AnalysisError("points 应该是一串点,不是单个点")
    try:
        seq = list(raw)
    except TypeError:
        raise AnalysisError(f"points 不是一串点:{raw!r}") from None

    kwargs = {}
    if params.get("min_judged") is not None:
        try:
            kwargs["min_judged"] = int(params["min_judged"])
        except (TypeError, ValueError):
            raise AnalysisError(
                f"min_judged={params['min_judged']!r} 不是整数") from None
    try:
        res = aggregate_cross_points(
            [_as_point(r, i) for i, r in enumerate(seq)], **kwargs)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from None

    out = res.as_dict()
    out["cross_verdict"] = res.verdict          # 闸门 selector 用的名字
    return out


__all__ = ["CROSS_VERDICTS", "SPECTRUM_STATES", "CrossCheckVerdict",
           "PointVerdict", "aggregate_cross_points", "cross_check_analysis"]
