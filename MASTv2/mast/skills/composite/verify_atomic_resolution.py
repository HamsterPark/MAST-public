"""对已保存 .sxm 执行帧准入检查、AssessAtomicPhase 裁决和三态映射。

AnalyzeScanImage 提供准入及交叉核对，裁决由接受显式阈值和尺度参数的 AssessAtomicPhase 完成。
atomic_resolved、atomic_absent、undecidable 分开返回；尺度不足及未知原因不能解释为没有原子相。
峰强度类诊断不参与裁决。实时缓冲未确认同源时不接受为输入。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)

logger = logging.getLogger(__name__)


# ── 闭集 ──────────────────────────────────────────────────────────────────────

#: 三态。刻意不复用 ``passed`` 这个二值词 —— 「判不了」需要一个自己的名字,
#: 否则它只能挤进「没有」里,而那两件事的下一步动作完全相反。
VERDICT_RESOLVED = "atomic_resolved"
VERDICT_ABSENT = "atomic_absent"
VERDICT_UNDECIDABLE = "undecidable"
ALL_VERDICTS: tuple[str, ...] = (
    VERDICT_RESOLVED, VERDICT_ABSENT, VERDICT_UNDECIDABLE)

#: ``undecidable`` 的补救动作(闭集)。每一条都必须是**能做**的一件事 ——
#: 一个说不出下一步的 "判不了" 与 "没有" 在流程上没有区别。
REMEDY_SHRINK_FIELD = "shrink_field"          # 缩视野 / 加像素(同比加线时)
REMEDY_WIDEN_FIELD = "widen_field"            # 扩视野(周期数不够,缩了只会更糟)
REMEDY_FIX_PIXEL_SCALE = "fix_pixel_scale"    # 查 .sxm 头解析与几何
REMEDY_RESCAN_FRAME = "rescan_frame"          # 这一帧本身坏了,重扫
REMEDY_MOVE_SITE = "move_site"                # 这个点是死平区,换点
REMEDY_FIX_ENVIRONMENT = "fix_environment"    # 依赖缺席,修环境
REMEDY_ASK_OPERATOR = "ask_operator"          # 只在兜底档出现(见下)
ALL_REMEDIES: tuple[str, ...] = (
    REMEDY_SHRINK_FIELD, REMEDY_WIDEN_FIELD, REMEDY_FIX_PIXEL_SCALE,
    REMEDY_RESCAN_FRAME, REMEDY_MOVE_SITE, REMEDY_FIX_ENVIRONMENT,
    REMEDY_ASK_OPERATOR)

#: 出局词 → (态, 补救)。**对 ``atomic_phase.ALL_REASONS`` 穷举**,测试断言。
#:
#: 上半段是**判据性**出局词:判据在一个判得了的尺度上看过了,结论是「没有」。
#: 下半段是**判不了**:证据不足以支持任何结论。两者的下一步动作相反 ——
#: 前者计入该偏压的失败预算,后者按 remedy 补救且**不计入**失败预算。
REASON_VERDICT: dict[str, tuple[str, str | None]] = {
    # ── 判据性:这一帧上没有原子分辨 ──
    "no_lattice_peak": (VERDICT_ABSENT, None),
    "not_a_lattice": (VERDICT_ABSENT, None),
    "fft_not_sharp": (VERDICT_ABSENT, None),
    "fast_axis_no_peak": (VERDICT_ABSENT, None),
    "period_below_lattice": (VERDICT_ABSENT, None),
    "period_far_above_lattice": (VERDICT_ABSENT, None),
    # 一堆峰但不是同一个晶格（半径各不相同）—— 这是关于「不是晶格」的**证据**，
    # 不是「判不了」。典型来源:钝针尖在倾斜面上成的像,谱上是一条穿过原点的
    # 弥散条纹,而条纹在角度上也集中 ⇒ 骗得过角向集中度。
    "peaks_not_one_lattice": (VERDICT_ABSENT, None),
    # 候选峰**全是脊上的点**,剔完一个一阶峰都不剩。归 absent 而不是 undecidable:
    # 判据看过了,而且看到的东西本身就是「这帧上是几条脊,不是一个二维晶格」——
    # 那是关于「没有」的证据。0569 那张噪声帧(剖面 ±1 pm、几条平滑横带)正是
    # 这一档;它此前因为「剔光 => ok=False => 整条判据跳过」而一路通关。
    "peaks_are_ridges": (VERDICT_ABSENT, None),
    # 二维谱给的周期与逐行谱给的周期差了一倍以上 ⇒ 可信的那个方向**没有确认**
    # 二维谱那个峰。与 ``fast_axis_no_peak`` 同族(那支是逐行谱什么也没看到),
    # 所以同样归 absent。典型来源:波浪形横带 —— 周期在慢轴、相位沿快轴游走。
    "radial_fast_axis_disagree": (VERDICT_ABSENT, None),
    # ── 判不了:证据不足,不是「没有」──
    # scale_gate 与 unknown_pixel_size 是**两种**判不了,补救不同:
    # 前者去缩视野/加像素,后者去查 .sxm 头解析与几何。
    # ``too_few_periods`` 与 ``scale_gate`` 的补救**方向相反**:前者视野太小
    # 装不下周期(要扩),后者像素太粗看不清周期(要缩)。写反了会让流程
    # 一路缩到更判不了。
    "too_few_periods": (VERDICT_UNDECIDABLE, REMEDY_WIDEN_FIELD),
    "scale_gate": (VERDICT_UNDECIDABLE, REMEDY_SHRINK_FIELD),
    "unknown_pixel_size": (VERDICT_UNDECIDABLE, REMEDY_FIX_PIXEL_SCALE),
    # ⚠️ 本表最容易被写反的一条。scale_reduced 让 passed=False,长得和「没过」
    # 一模一样,但判据模块写的是「证据强度撑不住一次针尖验收」。归成 absent
    # 会让 5.12-10 nm 整段稳定报假话。
    "scale_reduced": (VERDICT_UNDECIDABLE, REMEDY_SHRINK_FIELD),
    "insufficient_data": (VERDICT_UNDECIDABLE, REMEDY_RESCAN_FRAME),
    "dead_flat": (VERDICT_UNDECIDABLE, REMEDY_MOVE_SITE),
    "dependency_unavailable": (VERDICT_UNDECIDABLE, REMEDY_FIX_ENVIRONMENT),
}


#: 出局词 → **给人看的中文**。对 ``atomic_phase.ALL_REASONS`` 穷举(测试断言)。
#:
#: 2026-08-24 补。在此之前 :meth:`VerifyAtomicResolution._why` 是这么拼的::
#:
#:     "这一帧上没有原子分辨: " + "、".join(cls["absent_reasons"])
#:
#: 于是用户在旁白和技能回包里读到的是**英文 snake_case**:
#: 「这一帧上没有原子分辨: peaks_not_one_lattice、fast_axis_no_peak」。
#: 16 个词全是这样 —— 这不是某一个词漏了措辞,是这条路上从来没有过措辞。
#:
#: ⚠️ 措辞要说**观察到了什么**,不要说**下一步做什么**:下一步是 ``remedy`` 的活,
#: 两者混在一句里会让「判不了」读起来像「已经判了」。
REASON_ZH: dict[str, str] = {
    # ── 判据性:这一帧上没有原子分辨 ──
    "no_lattice_peak": "原子带里没有显著谱峰",
    "not_a_lattice": "谱是弥散的,不是离散的布拉格点",
    "fft_not_sharp": "布拉格峰不够锐",
    "fast_axis_no_peak": "逐行谱看不到这个周期(只有慢轴有 ⇒ 是行噪声不是晶格)",
    "period_below_lattice": "周期比这个晶格的还小",
    "period_far_above_lattice": "周期远大于这个晶格的",
    "peaks_not_one_lattice": "峰不在同一个半径上(不是同一个晶格)",
    "peaks_are_ridges": "候选峰全是脊上的点,剔完一个一阶峰都不剩",
    "radial_fast_axis_disagree": "二维谱与逐行谱给的周期对不上",
    # ── 判不了:证据不足,不是「没有」──
    "too_few_periods": "视野里装不下足够多的周期",
    "scale_gate": "像素太粗,这个尺度上判不了",
    "scale_reduced": "证据强度撑不住一次针尖验收",
    "unknown_pixel_size": "读不到像素尺度",
    "insufficient_data": "扫到的行太少",
    "dead_flat": "这一帧是平的,没有起伏",
    "dependency_unavailable": "判据的依赖缺席",
}


def reason_zh(word: str) -> str:
    """出局词的中文。**查不到就原样返回那个词** —— 绝不编一句。

    编一句的下场比露出英文词坏得多:用户会照着那句话去处置,
    而那句话与判据实际看到的东西无关。露出英文词至少是**可查的**。
    """
    return REASON_ZH.get(str(word), str(word))

#: 同时出现多个「判不了」时按这个顺序取补救(确定性,不看 dict 顺序)。
#: 排在前面的是**更靠近根因**的那一个:读不出像素尺度 / 依赖缺席 / 帧本身坏了,
#: 都比「尺度不够」更根本 —— 尺度不够是唯一会与判据性出局词共存的那一个
#: (``scale_reduced`` 与 ``not_a_lattice`` 会一起出现)。
_REMEDY_PRIORITY: tuple[str, ...] = (
    "unknown_pixel_size", "dependency_unavailable", "insufficient_data",
    "dead_flat", "too_few_periods", "scale_gate", "scale_reduced",
)


def classify_reasons(passed: bool,
                     reasons: "list[str] | tuple[str, ...]") -> dict[str, Any]:
    """出局词 → 三态。纯函数,不碰硬件、不读配置、不抛异常。

    返回 ``{verdict, remedy, undecidable_reasons, absent_reasons, unmapped_reasons}``。

    规则(顺序承重):

    1. ``passed`` 为真 ⇒ ``atomic_resolved``(判据全过且尺度门允许判定);
    2. 任何一个出局词属于「判不了」⇒ ``undecidable`` —— **判不了压倒没有**。
       某一条判据判不了时,「没有原子分辨」这句话就没有说的资格;
    3. 出局词全是判据性的 ⇒ ``atomic_absent``;
    4. 认不出来的出局词 ⇒ ``undecidable`` + ``ask_operator``。**兜底档不是
       atomic_absent**:一个没人认识的词最不该做的事,就是拿它去下「没有」的结论
       然后接着扰动针尖。(映射表对 ``ALL_REASONS`` 穷举,所以这一档在测试通过时
       不可达 —— 它防的是判据新增出局词而映射表没跟上的那一天。)
    """
    words = [str(r) for r in (reasons or ())]
    absent: list[str] = []
    undecidable: list[str] = []
    unmapped: list[str] = []
    for w in words:
        mapped = REASON_VERDICT.get(w)
        if mapped is None:
            unmapped.append(w)
        elif mapped[0] == VERDICT_UNDECIDABLE:
            undecidable.append(w)
        else:
            absent.append(w)

    out: dict[str, Any] = {
        "undecidable_reasons": undecidable,
        "absent_reasons": absent,
        "unmapped_reasons": unmapped,
    }
    if bool(passed) and not words:
        out.update(verdict=VERDICT_RESOLVED, remedy=None)
        return out
    if bool(passed):
        # 判据说过了却带着出局词 —— 自相矛盾,不替它圆场。
        out.update(verdict=VERDICT_UNDECIDABLE, remedy=REMEDY_ASK_OPERATOR)
        return out
    if unmapped:
        out.update(verdict=VERDICT_UNDECIDABLE, remedy=REMEDY_ASK_OPERATOR)
        return out
    if undecidable:
        for candidate in _REMEDY_PRIORITY:
            if candidate in undecidable:
                out.update(verdict=VERDICT_UNDECIDABLE,
                           remedy=REASON_VERDICT[candidate][1])
                return out
        out.update(verdict=VERDICT_UNDECIDABLE, remedy=REMEDY_ASK_OPERATOR)
        return out
    if absent:
        out.update(verdict=VERDICT_ABSENT, remedy=None)
        return out
    # passed=False 而一个出局词都没有 —— 判据不该产生这种结果。
    out.update(verdict=VERDICT_UNDECIDABLE, remedy=REMEDY_ASK_OPERATOR)
    return out


def atomic_scale_reject(size_m: float, pixels: int) -> "dict[str, Any] | None":
    """下发扫描**之前**的尺度拒绝。过得了门返回 ``None``。

    形状照 ``core.scan_planner.PlanReject.as_dict()``:``{code, detail, alternatives}``。

    **绝不静默缩帧。** 帧宽是用户的意图,偷偷改掉等于换了被测对象 —— 所以这里
    是一条**拒绝**,``size_m`` 原样回给调用方,由人(或上层 spec)决定走哪条路。
    ``alternatives`` 给两条**算得出来的具体**路,不是「建议调整参数」这种废话:
    这个视野要多少像素,以及这个像素数下视野要多小。

    ⚠️ 加像素的那条路必须**同比加 line_time**:提像素不提线时会让 ``nm/px``
    变好看而每像素驻留砍半 —— 尺度门是被骗过去的,不是真的过了。
    """
    from mast.vision.atomic_phase import (
        SCALE_FULL_NMPP,
        min_pixels_for_scale,
        plan_scale,
    )

    nmpp, scale, problem = plan_scale(size_m, pixels)
    if scale == "full":
        return None

    alternatives: list[str] = []
    need_px = min_pixels_for_scale(size_m)
    if need_px:
        alternatives.append(
            f"这个视野({(size_m or 0.0) * 1e9:g} nm)要 {need_px} px 以上"
            f"(线时必须同比加大,守住每像素驻留)")
    try:
        px = int(pixels)
    except (TypeError, ValueError):
        px = 0
    if px > 0:
        max_nm = px * SCALE_FULL_NMPP
        alternatives.append(f"保持 {px} px 的话,视野要小于 {max_nm:g} nm")
    return {
        "code": "atomic_scale_unreachable",
        "detail": problem or "这组帧参数进不了原子判据的满权重档",
        "alternatives": alternatives,
        "nm_per_px": nmpp,
        "scale": scale,
        # 原样回传 —— 证明这是拒绝,不是一次悄悄的改写。
        "size_m": size_m,
        "pixels": pixels,
    }


# ── 帧准入 ────────────────────────────────────────────────────────────────────

def _num(value: Any) -> "float | None":
    """能当数用就返回 float,``None`` / NaN / inf 一律 ``None``(= 读不到)。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def evaluate_frame_admission(metrics: "dict | None",
                             thresholds: Any) -> dict[str, Any]:
    """帧准入:这一帧**够不够格**被拿去判原子相。纯函数。

    闸在裁决**之前**。理由:一张残帧 / 噪声帧上的「没有原子分辨」,是在自己刚制造
    的坑上做判定 —— 那个结论说的是这一帧坏了,不是这个偏压上没有原子对比度。

    三项(设计 D1 逐条点名):NaN 占比、行相关中位数、正反扫不稳定度。

    **``bad_row_frac`` 报数但不设闸**:它的阈值叫 ``bad_row_frac_annotate``,
    出厂值是 **0.0** —— 那是一条「有一行坏就标注一下」的**标注**线,不是准入线。
    拿它当硬闸会让几乎每一张真实帧都过不了准入。字段标签会说谎,这里按它的实际
    用途读它,并把它列进 ``ungated_criteria`` 让 verdict 说得出自己的依据。

    「读不到」不算通过:任何一项该评而评不出来,准入就不通过,补救是重扫。
    唯一的例外是正反扫不稳定度 —— 单向帧本来就没有反扫可比,那是**不适用**
    而不是**读不到**,标 ``applicable=False`` 并如实说出来。
    """
    gated: list[dict[str, Any]] = []
    unreadable: list[str] = []

    if not isinstance(metrics, dict):
        return {
            "passed": False,
            "measured": False,
            "gated_criteria": [],
            "unreadable": ["frame_metrics"],
            "rowcorr_median": None, "nan_frac": None,
            "fb_instability": None, "bad_row_frac": None,
            "why": "帧准入指标读不到 —— 读不到不是通过",
        }

    artifacts = metrics.get("artifacts")
    bad_row = _num((artifacts or {}).get("bad_row_frac")
                   if isinstance(artifacts, dict) else None)

    def _check(name: str, value: "float | None", limit: "float | None",
               direction: str, *, applicable: bool = True) -> None:
        entry: dict[str, Any] = {
            "name": name, "value": value, "threshold": limit,
            "direction": direction, "applicable": applicable,
        }
        if not applicable:
            entry["passed"] = None
        elif value is None or limit is None:
            entry["passed"] = None
            unreadable.append(name)
        elif direction == "max":
            entry["passed"] = value <= limit
        else:
            entry["passed"] = value >= limit
        gated.append(entry)

    nan_frac = _num(metrics.get("nan_frac"))
    rowcorr = _num(metrics.get("rowcorr_median"))
    fb = _num(metrics.get("fb_instability"))
    fb_applicable = metrics.get("fb_instability") is not None

    _check("nan_frac", nan_frac, _num(getattr(thresholds, "nan_annotate", None)),
           "max")
    _check("rowcorr_median", rowcorr,
           _num(getattr(thresholds, "rowcorr_poor", None)), "min")
    _check("fb_instability", fb,
           _num(getattr(thresholds, "fb_instability_max", None)), "max",
           applicable=fb_applicable)

    failed = [e["name"] for e in gated if e["passed"] is False]
    passed = not failed and not unreadable
    if passed:
        why = "帧准入通过"
    elif failed:
        why = "帧质量不合格: " + "、".join(failed)
    else:
        why = "帧准入指标读不到(" + "、".join(unreadable) + ") —— 读不到不是通过"
    return {
        "passed": passed,
        "measured": True,
        "gated_criteria": gated,
        "unreadable": unreadable,
        "rowcorr_median": rowcorr,
        "nan_frac": nan_frac,
        "fb_instability": fb,
        "bad_row_frac": bad_row,
        "why": why,
    }


def _sample_pointer_agreement(profile_name: str) -> dict[str, Any]:
    """两个「现在是什么样品」的指针对不对得上。**提示,不是闸。**

    ``scan_prep_thresholds`` 的激活 profile(手动设)与实验记录里的当前样品
    (走实验日志)可以不一致,而今天没有任何代码在对账。这里对一次账,
    读不到就说读不到 —— ``None`` 不是 ``False``。
    """
    out: dict[str, Any] = {"profile_name": profile_name or None,
                           "sample_name": None, "agrees": None}
    try:
        from mast.core.sample_facts import current_sample_facts

        facts = current_sample_facts()
    except Exception as exc:  # noqa: BLE001 — 对账失败不该让分析失败
        logger.debug("样品指针对账失败(按「不知道」处理): %s", exc)
        return out
    if not facts.get("available"):
        return out
    name = facts.get("name") or facts.get("sample_subtype") or ""
    out["sample_name"] = name or None
    if not profile_name or not name:
        return out

    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    a, b = _norm(profile_name), _norm(name)
    if not a or not b:
        return out
    out["agrees"] = (a in b) or (b in a)
    return out


# ── 技能 ──────────────────────────────────────────────────────────────────────

class VerifyAtomicResolution(CompositeSkillGraph):
    """一帧上有没有原子分辨 —— 三态裁决 + 依据。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="VerifyAtomicResolution",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "对**一帧**已存盘的 .sxm 判断它有没有原子分辨 —— 并且说清是"
                "**三个答案里的哪一个**：'atomic_resolved'、'atomic_absent'、"
                "'undecidable'。"
                "**第三个才是重点**：一帧粗到分不开晶格的图、一帧读不出像素"
                "尺寸的图、一帧死平的或只采了一半的图，**撑不起「这里没有原子」这句话** —— 在那种帧上报「没有」是一句假话，它会把流程送去扰动一根"
                "本来好好的针尖。每一个 'undecidable' 都带一条补救动作"
                "（shrink_field / widen_field / fix_pixel_scale / rescan_frame / "
                "move_site / fix_environment）。"
                "只读，不碰硬件。输入**必须（MUST）**是已存盘的 .sxm 路径 —— 不接受"
                "实时扫描缓冲区，因为无法保证与保存帧同源。"
                "帧质量在晶格判据**之前**检查，这样一帧坏图就不会因为它没造成的"
                "损伤而被判。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path", type="str",
                    description=("**已存盘**的 .sxm 帧路径。实时缓冲区 / 数组"
                                 "一律不接受。"),
                    required=True),
                ParameterSpec(
                    name="channel", type="str",
                    description="形貌通道（标准是 'Z'）。",
                    required=False, default="Z"),
                ParameterSpec(
                    name="threshold_profile", type="str",
                    description=("帧准入阈值档（它们是在**哪个样品体系**上"
                                 "标定的）。留空 = 当前生效的那一档。"),
                    required=False, default=""),
                ParameterSpec(
                    name="expected_a_nm", type="float", unit="nm",
                    description=(
                        "用来比对的**原子行间距**期望值。留空则这项比对**关闭**"
                        "（默认）。这道检查的**下界是硬的、而且不对称**：填小了"
                        "会把真的原子分辨判掉，上界则很松。**不要拿文献上的猜测"
                        "去填它。**"),
                    required=False, min_value=0.0, max_value=10.0),
                ParameterSpec(
                    name="snr_min", type="float",
                    description="带内谱峰的最低信噪比。",
                    required=False, default=4.0, min_value=1.0, max_value=1e6),
                ParameterSpec(
                    name="concentration_min", type="float",
                    description=("最低角向集中度（分立的 Bragg 斑点 vs 弥散的"
                                 "环）—— **唯一**能把真晶格与针尖振铃分开的"
                                 "那条判据。"),
                    required=False, default=20.0, min_value=1.0, max_value=1e9),
                ParameterSpec(
                    name="sharpness_min", type="float",
                    description="FFT 谱峰的最低突出度。",
                    required=False, default=8.0, min_value=1.0, max_value=1e6),
                ParameterSpec(
                    name="allow_reduced_scale", type="bool",
                    description=("允许在 0.02-0.05 nm/px 的过渡带里给出肯定"
                                 "判定。默认 false —— 那一段的证据强度撑不住"
                                 "一次验收决定。打开它这件事会被记录。"),
                    required=False, default=False),
                ParameterSpec(
                    name="require_frame_admission", type="bool",
                    description=("质量闸门没过的帧，拒绝对它下判定"
                                 "（默认 true）。"),
                    required=False, default=True),
            ],
            estimated_duration_s=6.0,
            composition_level=3,
            tags=["atomic", "lattice", "fft", "analysis", "verdict", "read"],
        )

    # ── 计划 ──────────────────────────────────────────────────────────────

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        scan_path = str(params.get("scan_path") or "").strip()
        channel = str(params.get("channel") or "Z")
        profile = str(params.get("threshold_profile") or "").strip()
        require_admission = bool(params.get("require_frame_admission", True))

        self._verdict_out: dict[str, Any] = {}
        set_partial = executor.set_partial
        set_partial("frame_path", scan_path)
        set_partial("channel", channel)

        if not scan_path or not scan_path.lower().endswith(".sxm"):
            # 输入使用已保存的 .sxm；缓冲来源不确定时不得作同源数据判断。
            self._finish(executor, VERDICT_UNDECIDABLE, REMEDY_RESCAN_FRAME,
                         why=(f"要一个已保存的 .sxm 路径,收到 {scan_path!r}。"
                              f"实时缓冲不接受(它装的可能不是这一帧)。"))
            return

        # ── 1) 帧准入 + 对照(只准入不裁决)──
        yield CompositeStep(
            step_id="admit", skill_name="AnalyzeScanImage",
            params={"scan_path": scan_path, "channel": channel,
                    "threshold_profile": profile, "save_png": False},
            optional=True, checkpoint_after=False, tags=("admit", "read"))

        admit_result = executor.sub_results.get("admit")
        admit_data = dict(getattr(admit_result, "data", None) or {})
        metrics = admit_data.get("metrics")
        plan_block = admit_data.get("plan") or {}
        profile_name = str(plan_block.get("threshold_profile") or profile or "")
        profile_provenance = str(plan_block.get("threshold_provenance") or "")

        thresholds = self._resolve_thresholds(profile_name or profile)
        admission = evaluate_frame_admission(
            metrics if isinstance(metrics, dict) else None, thresholds)
        admission["required"] = require_admission
        set_partial("frame_admission", admission)
        set_partial("profile_name", profile_name or None)
        set_partial("profile_provenance", profile_provenance or None)
        set_partial("sample_pointer", _sample_pointer_agreement(profile_name))

        # 与 (B) 路的对照:同一帧、同一判据函数、另一条预处理。
        cross = self._cross_check(metrics)
        set_partial("cross_check", cross)

        if require_admission and not admission["passed"]:
            # 残帧上的「没有原子分辨」是在自己刚制造的坑上做判定 —— 不跑判据。
            self._finish(executor, VERDICT_UNDECIDABLE, REMEDY_RESCAN_FRAME,
                         why=admission["why"])
            return

        # ── 2) 唯一裁决 ──
        yield CompositeStep(
            step_id="assess", skill_name="AssessAtomicPhase",
            params=self._assess_params(params, scan_path, channel),
            optional=True, checkpoint_after=True, tags=("verdict", "read"))

        assess_result = executor.sub_results.get("assess")
        if assess_result is None or not getattr(assess_result, "success", False):
            err = getattr(assess_result, "error", None) or "判据没有返回结果"
            self._finish(executor, VERDICT_UNDECIDABLE, REMEDY_RESCAN_FRAME,
                         why=f"判据跑不起来: {err}")
            return

        # ── 3) 三态映射 ──
        self._finish_from_criterion(executor,
                                    dict(getattr(assess_result, "data", None) or {}))

    # ── 裁决 ──────────────────────────────────────────────────────────────

    def _assess_params(self, params: dict, scan_path: str,
                       channel: str) -> dict[str, Any]:
        """交给唯一裁决口的参数。

        ``expected_a_nm`` 缺省时**显式传 0**(= 关闭比对),而不是不传。不传会让
        ``AssessAtomicPhase`` 去从当前样品**推断**一个晶格常数,而那个值一进判据
        就是一道**下界严**的硬闸 —— 填错方向会把真原子分辨判成不合格。
        「默认不传 = 比对关闭」这句话,只有显式传 0 才成立。
        """
        out: dict[str, Any] = {
            "scan_path": scan_path,
            "channel": channel,
            "snr_min": float(params.get("snr_min", 4.0)),
            "concentration_min": float(params.get("concentration_min", 20.0)),
            "sharpness_min": float(params.get("sharpness_min", 8.0)),
            "allow_reduced_scale": bool(params.get("allow_reduced_scale", False)),
        }
        raw = params.get("expected_a_nm")
        try:
            expected = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            expected = 0.0
        out["expected_a_nm"] = expected if expected > 0 else 0.0
        return out

    def _finish_from_criterion(self, executor: GraphExecutor,
                               data: dict[str, Any]) -> None:
        reasons = list(data.get("reasons") or ())
        cls = classify_reasons(bool(data.get("passed")), reasons)
        gated, ungated = self._criteria_ledger(data)
        executor.set_partial("criterion", data)
        executor.set_partial("gated_criteria", gated)
        executor.set_partial("ungated_criteria", ungated)
        self._finish(executor, cls["verdict"], cls["remedy"],
                     why=self._why(cls, data), classification=cls,
                     criterion=data)

    def _criteria_ledger(
            self, data: dict[str, Any]) -> tuple[list[dict], list[dict]]:
        """哪些量**参与了**这次裁决,哪些只是诊断。

        没有这一栏,verdict 就是一句没依据的话:读的人无从知道那个 "absent" 是三条
        判据都看过了,还是尺度门根本没让判据开口。
        """
        params = getattr(self, "_assess_sent", {}) or {}
        gated = [
            {"name": "band_peak_snr", "value": data.get("snr"),
             "threshold": params.get("snr_min"),
             "out_word": "no_lattice_peak"},
            {"name": "angular_concentration",
             "value": data.get("angular_concentration"),
             "threshold": params.get("concentration_min"),
             "out_word": "not_a_lattice"},
            {"name": "fft_sharpness", "value": data.get("fft_sharpness"),
             "threshold": params.get("sharpness_min"),
             "out_word": "fft_not_sharp"},
            {"name": "fast_axis_period_nm",
             "value": data.get("period_fast_axis_nm"),
             "threshold": None, "out_word": "fast_axis_no_peak"},
            {"name": "scale_gate", "value": data.get("nm_per_px"),
             "threshold": data.get("scale"), "out_word": "scale_gate"},
        ]
        expected = data.get("expected_a_nm")
        if expected:
            gated.append({"name": "expected_a_nm_comparison",
                          "value": data.get("period_fast_axis_nm"),
                          "threshold": expected,
                          "out_word": "period_below_lattice"})
        ungated = [
            {"name": "order_ratio", "value": data.get("order_ratio"),
             "why_not_gated": ("自相关角向最大值 —— 合成实测分不开准周期抖动"
                               "(与 6 pm 噪声下的真晶格完全重叠),只作诊断")},
            {"name": "period_radial_nm", "value": data.get("period_nm"),
             "why_not_gated": "二维谱的径向周期会被慢轴漂移拉偏,引用晶格常数只能用快轴"},
            {"name": "slow_axis_trusted", "value": data.get("slow_axis_trusted"),
             "why_not_gated": "帧法下恒为假 —— 慢轴方向的数不可信"},
            {"name": "bad_row_frac", "value": None,
             "why_not_gated": ("它的阈值是**标注**线(出厂 0.0),不是准入线;"
                               "拿它当硬闸会让几乎每张真实帧都过不了准入")},
        ]
        if not expected:
            ungated.append({
                "name": "expected_a_nm_comparison", "value": None,
                "why_not_gated": ("比对默认关闭 —— 下界是硬闸且代价不对称,"
                                  "宁可不填也不用一个猜测值当闸"),
            })
        return gated, ungated

    @staticmethod
    def _why(cls: dict[str, Any], data: dict[str, Any]) -> str:
        verdict = cls["verdict"]
        if verdict == VERDICT_RESOLVED:
            per = data.get("period_fast_axis_nm")
            conc = data.get("angular_concentration")
            out = "有原子分辨"
            if per:
                out += f"(快扫方向周期 {float(per):.3f} nm"
                out += (f",角向集中度 {float(conc):.0f})" if conc else ")")
            return out
        if verdict == VERDICT_ABSENT:
            return ("这一帧上没有原子分辨: "
                    + "、".join(reason_zh(w) for w in cls["absent_reasons"]))
        # ``unmapped_reasons`` 按定义查不到中文 —— ``reason_zh`` 会原样返回，
        # 而那正是我们想要的:一个没见过的词露出来,比被翻译成一句像样的话好。
        words = cls["undecidable_reasons"] or cls["unmapped_reasons"]
        nmpp = data.get("nm_per_px")
        detail = ("、".join(reason_zh(w) for w in words) if words
                  else "判据没有给出可用结论")
        if nmpp:
            detail += f"(这一帧 {float(nmpp):.4f} nm/px)"
        return f"判不了(不等于没有): {detail}"

    def _finish(self, executor: GraphExecutor, verdict: str,
                remedy: "str | None", *, why: str,
                classification: "dict | None" = None,
                criterion: "dict | None" = None) -> None:
        cls = classification or {"undecidable_reasons": [], "absent_reasons": [],
                                 "unmapped_reasons": []}
        crit = criterion or {}
        self._verdict_out = {
            "verdict": verdict,
            "remedy": remedy,
            "why": why,
            "reasons": list(crit.get("reasons") or ()),
            "warnings": list(crit.get("warnings") or ()),
            "unmapped_reasons": list(cls.get("unmapped_reasons") or ()),
        }
        executor.set_partial("verdict", verdict)
        executor.set_partial("remedy", remedy)
        executor.set_partial("why", why)

    @staticmethod
    def _cross_check(metrics: "dict | None") -> dict[str, Any]:
        """与 ``AnalyzeScanImage`` 那条预处理的对照。**不阻断、不改裁决。**

        两个数在这一步都已经算出来了,比对是零成本的自检信号。让第二个观察者说话,
        不是让它抢方向盘。``None`` = 那一路没算出来,**不是**「不一致」。
        """
        atomic = (metrics or {}).get("atomic") if isinstance(metrics, dict) else None
        if not isinstance(atomic, dict):
            return {"analyze_scan_image_passed": None, "agrees": None,
                    "note": "对照路没有给出原子相结论(不等于不一致)"}
        return {
            "analyze_scan_image_passed": bool(atomic.get("passed")),
            "analyze_scan_image_reasons": list(atomic.get("reasons") or ()),
            "agrees": None,      # 在 aggregate 里与裁决口比对后填
        }

    @staticmethod
    def _resolve_thresholds(profile: str):
        try:
            from mast.vision.scan_prep_thresholds import resolve
            return resolve(profile or None)
        except Exception as exc:  # noqa: BLE001 — 读不到阈值 ⇒ 准入判不了
            logger.debug("帧准入阈值读不到: %s", exc)
            return None

    # ── 汇总 ──────────────────────────────────────────────────────────────

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        partial = dict(progress.partial_data)
        crit = partial.get("criterion") or {}
        out = dict(self._verdict_out or {})
        out.setdefault("verdict", VERDICT_UNDECIDABLE)
        out.setdefault("remedy", REMEDY_RESCAN_FRAME)
        out.setdefault("why", "没有走到裁决")
        out.setdefault("reasons", [])
        out.setdefault("warnings", [])

        cross = dict(partial.get("cross_check") or {})
        other = cross.get("analyze_scan_image_passed")
        if other is None:
            cross["agrees"] = None
        else:
            cross["agrees"] = bool(other) is (out["verdict"] == VERDICT_RESOLVED)
        pointer = dict(partial.get("sample_pointer") or {})

        out.update({
            "frame_path": partial.get("frame_path"),
            "channel": partial.get("channel"),
            "nm_per_px": crit.get("nm_per_px"),
            "scale": crit.get("scale"),
            "period_fast_axis_nm": crit.get("period_fast_axis_nm"),
            "period_radial_nm": crit.get("period_nm"),
            "angular_concentration": crit.get("angular_concentration"),
            # 帧内两半 —— 判据算好了放在 criterion 里，但一直没透传出去，
            # 于是调用方读到的永远是 None（2026-08-23 由旁白那条测试查出来：
            # 发出方读六个键，这里只回其中三个，句子会**若无其事地少说三样**）。
            "half_concentrations": crit.get("half_concentrations"),
            "fft_sharpness": crit.get("fft_sharpness"),
            "snr": crit.get("snr"),
            "order_ratio": crit.get("order_ratio"),
            "slow_axis_trusted": crit.get("slow_axis_trusted", False),
            "expected_a_nm": crit.get("expected_a_nm"),
            "frame_admission": partial.get("frame_admission") or {},
            "gated_criteria": partial.get("gated_criteria") or [],
            "ungated_criteria": partial.get("ungated_criteria") or [],
            "profile_name": partial.get("profile_name"),
            "profile_provenance": partial.get("profile_provenance"),
            "cross_check": cross,
            "sample_pointer": pointer,
            "sample_pointer_agrees": pointer.get("agrees"),
        })
        return out

    def _validate_products(self, data: dict) -> tuple[bool, str]:
        """本技能的产物是一个**判断**,不是一张图 —— 跳过通用产物闸。

        通用闸会把 ``frame_path`` 指向的帧再判一次(死平/全 NaN ⇒ degraded)。
        而「这一帧死平」正是本技能已经用 ``undecidable / move_site`` 说出来的话:
        让通用闸再判一遍,等于对同一张帧给出第二个来源不同的结论,然后用它盖掉
        第一个。同一个动作两份实现,只会留下一个说不清是谁答的答案。
        """
        return True, ""

    def run_composite(self, context, params: dict) -> SkillResult:
        self._verdict_out = {}
        self._assess_sent = self._assess_params(
            params, str(params.get("scan_path") or ""),
            str(params.get("channel") or "Z"))
        result = self._graph_execute(context, params)
        if not result.success:
            return result
        data = dict(result.data or {})
        return SkillResult(
            skill_name=self._skill_name(), success=True, data=data,
            summary=self._summary(data), nanonis_calls=list(result.nanonis_calls))

    @staticmethod
    def _summary(data: dict) -> str:
        verdict = data.get("verdict")
        head = {
            VERDICT_RESOLVED: "✔ 有原子分辨",
            VERDICT_ABSENT: "✘ 这一帧上没有原子分辨",
            VERDICT_UNDECIDABLE: "? 判不了（**不等于**没有）",
        }.get(verdict, f"? {verdict}")
        lines = [f"{head} — {data.get('why') or ''}",
                 f"帧: {Path(str(data.get('frame_path') or '')).name or '(无)'}"]
        if data.get("remedy"):
            lines.append(f"下一步: {data['remedy']}")
        adm = data.get("frame_admission") or {}
        if adm:
            lines.append(f"帧准入: {adm.get('why')}")
        cross = data.get("cross_check") or {}
        if cross.get("agrees") is False:
            lines.append(
                "⚠️ 两条预处理给出不同结论(裁决以 AssessAtomicPhase 为准;"
                "这只是一条自检信号,不改判)")
        elif cross.get("agrees") is None:
            lines.append("对照路没有结论 —— 无从对账(不等于一致)")
        if data.get("sample_pointer_agrees") is False:
            lines.append(
                f"⚠️ 阈值 profile `{data.get('profile_name')}` 与当前登记样品 "
                f"`{(data.get('sample_pointer') or {}).get('sample_name')}` 对不上"
                f"(提示,不阻断)")
        prov = data.get("profile_provenance")
        lines.append(f"阈值 profile `{data.get('profile_name')}` — {prov or '未标定'}")
        if data.get("warnings"):
            lines.append("注意: " + "、".join(data["warnings"]))
        per = data.get("period_fast_axis_nm")
        if per:
            lines.append(
                f"快扫方向周期 {float(per):.3f} nm（引用晶格常数只能用这个数;"
                f"慢轴不可信,且它直接吃 xy 标定误差）")
        return "\n".join(lines)


def make_tool(context_provider):
    return wrap_skill(VerifyAtomicResolution, context_provider)


__all__ = [
    "ALL_REMEDIES",
    "ALL_VERDICTS",
    "REASON_VERDICT",
    "VERDICT_ABSENT",
    "VERDICT_RESOLVED",
    "VERDICT_UNDECIDABLE",
    "VerifyAtomicResolution",
    "atomic_scale_reject",
    "classify_reasons",
    "evaluate_frame_admission",
]
