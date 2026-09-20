"""判据②「表面起伏极大」的门 —— **只出观察,不出结论**。

设计文档:``docs/v2/design/`` 的「S1 修针循环 conduct 化」设计,D1-D3 / D10 / §3.1
(文件名按「S1 修针循环」检索;通用层注释不写样品名前缀 ——
``stm_capability_vs_sample_layer.md`` 拍板④,由
``tests/v2/unit/test_no_sample_names_in_generic_layer.py`` 强制)。

## 它回答的问题,以及它**不**回答的那个

跨点判据要求「表面起伏极大**且**换位置扫依旧」—— 这是一个**合取**。
本模块只做前半句:这一帧的起伏比阈值大不大。后半句(是针还是表面)由
:func:`mast.conduct.cross_check.aggregate_cross_points` 的跨点聚合回答。

所以 :class:`CorrugationVerdict` 的词表里**没有 `bad_tip`**(:data:`VERDICTS`),
而且这不是疏漏 —— 一片台阶簇会给出与坏针团簇同样高的 RMS,单帧分不开这两者。
把「起伏大」直接翻译成「针坏了」是本设计明确否掉的方案,由
``tests/v2/unit/vision/test_corrugation_gate.py::test_corrugation_verdict_vocabulary``
钉住。

## 零第二真源:数不是这里算的

``value_pm`` 来自 :func:`mast.vision.frame_validity.judge_frame` 已经算好的
``corrugation_rms_m``(``_detrend`` 逐行中值 + OLS 平面 + float32,再取 ``std``)。
本模块**不自己去趋势、不自己取统计量** —— 同一个物理量的第二份实现迟早各自漂移,
而且「同名不同预处理 = 不同的量」,阈值跨口径不可搬。

当前使用 OLS + std 口径；其他去趋势或统计方法不能沿用未重新验证的阈值。
:data:`DETREND` / :data:`STATISTIC` 随每个数字共同报告。将来选型跑完要换口径,换的是**阈值和口径一起换** ——
一个 pm 数字离开 ``(detrend, statistic, ref_scan_nm)`` 三个声明就没有意义。

## 判定顺序不能换

    帧不可用 → 尺度不匹配 → 无阈值 → low → high → normal

四个 ``undecidable`` 的成因必须能从 ``reason`` 里分辨(它们指向不同的下一步:
换地方重扫 / 换视野重扫 / 去标定阈值)。顺序本身由测试钉住:例如「没有阈值」
排在 ``low`` 之前,意味着阈值没标定时连「起伏太小所以弃权」都不说 ——
这是有意的,``low`` 是判据①的弃权门,它不该在判据②未配置时替判据①发言。

## `low` 是判据①的弃权门,不是本判据的下限

``low`` 用的是 :data:`mast.skills.paper.line_check._DEFAULT_MIN_CORRUGATION_M`
**这个对象**(见 :func:`low_gate_m`)。两道门应共享同一个下限，
避免复制常量后各自漂移。

## 这个判据对什么瞎

见 :data:`BLIND_TO`:``_detrend`` 第一步减掉**逐行中值**,所以**行偏置型划痕**
在进入任何下游判据之前就被删干净了。合取项(``herringbone.slow_axis_power_ratio``
/ ``stripe_snr``)本设计**不接**：多个未标定阈值同时引入时，
无法区分判据失效的来源。所以每一份返回值都带着这句话 ——
让读者知道这个证据回答的是哪个问题。

## pm 阈值都带着一个系统偏差

见 :data:`Z_CAL_NOTE`。**z 重标定之后,所有 pm 单位的阈值作废**,而且机械地按比例
缩放**不是**替代品:分离度不一定跟着缩放存活,必须重跑选型程序(设计 §4 陷阱 7)。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

#: 闭集词表。**不含 `bad_tip`** —— 见模块注释第一节。
VERDICTS: tuple[str, ...] = ("high", "normal", "low", "undecidable")

#: 本切片唯一的口径。数字必须**随口径一起**出现,否则那个数没有意义。
#: 换口径 = 换判据 ⇒ 阈值必须同时重标(设计 D10、§4 陷阱 6)。
DETREND: str = "ols"
STATISTIC: str = "std"

#: 本判据测不到什么。**恒随返回值给出** —— D3:合取项本次不接,但缺口要说出来。
BLIND_TO: str = (
    "对行偏置型划痕不敏感:去趋势(_detrend)第一步就减掉逐行中值,"
    "行与行之间的整体高度差在进入本判据之前已被删干净。"
    "要问「有没有慢轴划痕」得另配合取项(herringbone.slow_axis_power_ratio / "
    "stripe_snr),本设计没有接 —— 那是另一个缺口,不是本判据的答案。"
)

#: 所有 pm 单位阈值共同的口径声明；公开快照不携带任何仪器的实测标定偏差。
Z_CAL_NOTE: str = (
    "压电 Z 标定会影响所有 pm 单位阈值；使用前须在目标仪器上核验。"
    "Z 重标之后旧阈值必须重新验证，不能仅按比例换算；应重跑选型与标定。"
)


def low_gate_m() -> float:
    """``low`` 档用的那个下限(米)——**判据①弃权门的那一个对象**。

    刻意做成一个取值函数而不是本模块的一个常量:这里一旦写下 ``15e-12``,
    仓里就有了第二个下限,而两个下限迟早不一样。测试用 ``is`` 钉这件事
    (float 是对象,同一个模块级常量 import 过来仍是同一个对象)。
    """
    # 公开版：CheckLineQuality（skills/paper）不随仓发布，常量就地保留。
    return 15e-12


@dataclass(frozen=True)
class CorrugationVerdict:
    """一帧的起伏观察。**这不是针尖结论** —— 见模块注释。"""

    #: 四态之一(:data:`VERDICTS`)。
    verdict: str
    #: 本口径下测出来的那个数(pm)。``None`` **只在帧不可用时**。
    value_pm: float | None
    #: 用了哪个口径。必须随数字一起出现,否则这个数没有意义。
    detrend: str
    statistic: str
    #: ``judge_frame`` 给的那个米数,**恒上报**(历史可比 + 与判据①的弃权门同源)。
    #: 拒判时它也有值 —— 「测出来是零」和「没测」是两句话。
    corrugation_rms_m: float | None
    #: 用了哪个阈值。``None`` ⇒ verdict 必为 ``undecidable``。
    threshold_pm: float | None
    #: 尺度对账的两个数。
    ref_scan_nm: float | None
    this_scan_nm: float | None
    #: 尺度相对容差(用户手打 100 nm 得到 99.98 nm 这类误差)。
    rel_tol: float
    #: pm 阈值共同的口径声明(:data:`Z_CAL_NOTE`)。
    z_cal_note: str
    #: 人话一句。``undecidable`` 时说清是**哪一种**。
    reason: str
    #: 本口径测不到什么(:data:`BLIND_TO`)。
    blind_to: str
    #: 阈值出自哪个 profile,以及那套数是在什么数据上标的。
    profile_name: str
    provenance: str

    def as_dict(self) -> dict:
        """``SkillResult.data`` / conduct 事件用的纯 JSON 字典。"""
        return dict(asdict(self))


def _finite(x: Any) -> float | None:
    """能转成有限 float 就转,否则 ``None``(「读不到」不是 0)。"""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def judge_corrugation(verdict: Any, *,
                      threshold_pm: "float | None",
                      ref_scan_nm: "float | None",
                      this_scan_nm: "float | None",
                      rel_tol: float = 0.05,
                      profile_name: str = "",
                      provenance: str = "") -> CorrugationVerdict:
    """这一帧的起伏算不算「极大」?

    ``verdict`` 是 :func:`mast.vision.frame_validity.judge_frame` 的产出 ——
    **本函数不碰图像**,只读它已经算好的 ``usable`` 与 ``corrugation_rms_m``。

    ``threshold_pm`` / ``ref_scan_nm`` 是一组:阈值离开它标定时的视野就没有意义。
    任何一个缺 ⇒ ``undecidable``(判不了),**不是**「没有上限所以放行」。

    判定顺序见模块注释,**不能换**。
    """
    usable = bool(getattr(verdict, "usable", False))
    rms_m = _finite(getattr(verdict, "corrugation_rms_m", None))
    thr = _finite(threshold_pm)
    ref = _finite(ref_scan_nm)
    this = _finite(this_scan_nm)
    # ``None`` 当作没给(签名默认就是 0.05,没有信息丢失)。**负数照收** ——
    # 那会让尺度门永远不通过(全部 undecidable),而那是安全的一侧,而且
    # reason 里会原样印出「容差 ±-5%」,一眼看得出是谁传错了。
    tol = _finite(rel_tol)
    tol = 0.05 if tol is None else tol

    def out(kind: str, reason: str, *, value_pm: "float | None") -> CorrugationVerdict:
        return CorrugationVerdict(
            verdict=kind, value_pm=value_pm, detrend=DETREND, statistic=STATISTIC,
            corrugation_rms_m=rms_m, threshold_pm=thr,
            ref_scan_nm=ref, this_scan_nm=this, rel_tol=tol,
            z_cal_note=Z_CAL_NOTE, reason=reason, blind_to=BLIND_TO,
            profile_name=str(profile_name or ""), provenance=str(provenance or ""))

    # ── 1. 帧不可用 ────────────────────────────────────────────────────
    # 「判不了」与「判出来不好」是两句话:下游该换一块地方重扫,不是去修针尖。
    if not usable or rms_m is None:
        why = str(getattr(verdict, "reason", "") or "这一帧不能拿来做针尖判定")
        return out("undecidable", f"帧不可用,起伏判不了 —— {why}", value_pm=None)

    value_pm = rms_m * 1e12

    # 起伏统计依赖视野；阈值与参考视野必须匹配，不进行未经标定的归一化或跨尺度外推。
    if ref is None or ref <= 0:
        if thr is None:
            # 两个都没填 = 这套阈值**整组没标定**。这句话指向的下一步是「去标定」,
            # 而不是「去补一个视野声明」—— 判不了的每一种成因各指不同的路。
            return out("undecidable",
                       "判不了:起伏门没有标定 —— 上限(corrugation_high_pm)"
                       "与它的标定视野(corrugation_ref_scan_nm)都没填。"
                       "两个是一组,缺一个就判不了;这不是「没有上限所以都算正常」。",
                       value_pm=value_pm)
        return out("undecidable",
                   "判不了:没有声明这个阈值是在哪个视野上标的"
                   "(corrugation_ref_scan_nm 未填)。一个 pm 阈值离开它的视野就"
                   "没有意义,本判据不换算也不外推。", value_pm=value_pm)
    if this is None or this <= 0:
        return out("undecidable",
                   f"判不了:这一帧的视野读不出来,对不了账"
                   f"(阈值是在 {ref:g} nm 上标的)。读不到不等于对得上。",
                   value_pm=value_pm)
    if abs(this / ref - 1.0) > tol:
        return out("undecidable",
                   f"判不了:视野对不上 —— 这一帧 {this:g} nm,而阈值是在 "
                   f"{ref:g} nm 上标的(容差 ±{tol:.0%})。换一张同视野的再判;"
                   f"本判据不做跨视野换算。", value_pm=value_pm)

    # ── 3. 没有阈值 ────────────────────────────────────────────────────
    # None = **判不了**,不是「没有上限」。排在 low 之前:阈值没标定时,判据②
    # 不该借判据①的弃权门发言。
    if thr is None:
        return out("undecidable",
                   "判不了:没有起伏上限阈值(profile 的 corrugation_high_pm 未填)。"
                   "None = 判不了,不是「没有上限所以都算正常」。"
                   "要标它得先跑选型与标定程序。", value_pm=value_pm)

    # ── 4. low:判据①的弃权门,恒用 OLS+std 口径的那个米数 ────────────
    gate_m = low_gate_m()
    if rms_m < gate_m:
        return out("low",
                   f"起伏 {value_pm:.1f} pm 低于弃权门 {gate_m * 1e12:.0f} pm —— "
                   f"没有形貌就没有可相关的信号,这一档**弃权**,"
                   f"既不说针好也不说针坏(它是判据①的那道门,不是本判据的下限)。",
                   value_pm=value_pm)

    # ── 5/6. high / normal ────────────────────────────────────────────
    if value_pm > thr:
        return out("high",
                   f"起伏 {value_pm:.1f} pm 高于上限 {thr:g} pm"
                   f"(视野 {this:g} nm,口径 {DETREND}+{STATISTIC})。"
                   f"⚠️ 这只是一个**观察**:一片台阶簇会给出同样高的起伏,"
                   f"「是针还是表面」要靠换位置复测的跨点聚合来回答。",
                   value_pm=value_pm)
    return out("normal",
               f"起伏 {value_pm:.1f} pm 未超上限 {thr:g} pm"
               f"(视野 {this:g} nm,口径 {DETREND}+{STATISTIC})。",
               value_pm=value_pm)


__all__ = ["BLIND_TO", "DETREND", "STATISTIC", "VERDICTS", "Z_CAL_NOTE",
           "CorrugationVerdict", "judge_corrugation", "low_gate_m"]
