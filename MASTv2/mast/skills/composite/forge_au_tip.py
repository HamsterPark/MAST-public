"""ForgeAuTip：在 Au(111) 上组织脉冲、扫图验证、调平和扎针阶段。

外环受站点数、每站轮数、时间及粗动行程预算约束。耗尽预算时如实报告未达标。
level 阶段提供台阶，精修后报告锐度；缺少验收阈值不能声称锐度已经通过。
显式扫描工作点经 scan_at_params 统一传递，省略项交给扫描解析器；等待预算随几何推导。
工作点与判据必须由使用者按目标仪器确认，公开版本不携带站点标定。
修针过程中的形貌告警可按上下文处理，但持续异常、硬件互锁与粗动保护仍然有效。
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import replace
from typing import Any, Iterator

from mast.core.noble_tip_workflow import (
    NOBLE_METAL_BASELINE,
    NobleTipWorkflow,
    descend_sequence,
    forge_speed_notes,
    reconcile_with_tip_envelope,
    resolve,
)
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite._tip_phases import (
    _data,
    _err,
    _ok,
    # 旁白发射器。**借用而不是复制** —— 它是「永不抛」的那一份(一句话绝不许
    # 弄坏正在跑的实验),而一个复制品迟早会漏掉那个 try。
    # 2026-08-18 之前**这整个文件一句旁白都没有**:站点为什么结束、为什么换区、
    # 验收判了什么,全在沉默里,只有跑完那份报告里才有。
    _say,
    level_phase,
    poke_phase,
    pulse_phase,
    scan_at_params,
    verify_phase,
)
from mast.skills.composite.graph_executor import CompositeProgress, CompositeStep
# 同包内的两道前置闸,从 PrepareNobleTip 借来而不是复制:Tip Shaper 模块没起来就
# 早退、qPlus 传感器上的下压类动作默认拒绝(音叉损坏不可逆)。复制一份 = 两个
# 定义,而这两条的判据将来只会变得更严,不会更松。
from mast.skills.composite.prepare_noble_tip import (
    _qplus_blocked,
    _tip_shaper_preflight,
)

logger = logging.getLogger(__name__)

_WF = NOBLE_METAL_BASELINE

#: 外环失败原文里,表示「扫描是被停下的,不是自己没跑完」的标记。
#:
#: 这是对另一个模块(``ScanAt`` / ``PreScanCheck`` 的 ``_decide_outcome``)所写句子
#: 的子串匹配,而子串匹配平时是要避开的。这里可以接受,原因是它**只影响措辞**:
#: 用户按停时扫描步骤已经失败、外环已经中止,这条判据决定的仅仅是报告里说
#: 「用户停了」还是说一句泛泛的失败。真要做得更好,应该在产出方给一个机器可读的
#: 标记;在那之前,``test_forge_au_tip`` 里有一条钉子直接调那两个产出方生成真句子
#: 再喂给这里 —— 上游改措辞会在测试里炸,不会在真机上炸。
_OPERATOR_STOP_MARKERS: tuple[str, ...] = ("中途停止",)

#: 出厂时间预算(小时):总时间可调,默认为 12 小时。
#:
#: 时间预算控制是否启动下一轮；耗尽时报告预算状态，不能据此判定针尖失败。
#: 硬性轮数上限另用于防止无进展循环。
_DEFAULT_BUDGET_H: float = 12.0

#: 失控保险 —— **不是预算**。命中它本身就是 bug 的信号,所以设得远高于任何真实
#: 需求。存在的理由是 ``_stash`` 每站深拷整个 sites 列表进 partial_data,
#: 而 partial_data 要进 checkpoint:一旦出现每站几秒就失败的空转,12 小时能转
#: 上万圈,先撑爆 checkpoint,而流程看起来一直在「努力工作」。
_HARD_SITE_CAP: int = 200
_HARD_ROUND_CAP: int = 100

#: 连续这么多站**什么都没测到**就停,报 ``spinning``。
#:
#: 这不是「没修好」,是「没在修」。典型成因是与站点无关的故障(读不到线数据 /
#: 找不到干净落点 / 链路坏了)—— 换地方对它毫无帮助。没有这道闸的话,流程会一直
#: 换到时间耗尽,最后报「针尖未达标」;而那是一句**关于针尖的话**,这一跑却根本
#: 没测过针尖(同 ``_summary`` 里 ``measured_anything`` 那一段的纪律)。
_IDLE_SITE_LIMIT: int = 3


def _site_measured_something(site: dict) -> bool:
    """这一站有没有**真的产出过判据**。

    判的是「测到了东西」而不是「跑完了步骤」:一发脉冲打出去、或者一张验证图
    给出了相似度,都算;而一串立刻失败的步骤不算。空转判据靠它。
    """
    for ph in (site or {}).get("phases") or []:
        if not isinstance(ph, dict):
            continue
        if int(ph.get("fired") or 0) > 0:            # 脉冲真的打出去了
            return True
        if ph.get("similarity") is not None:         # 验证图真的给了相似度
            return True
        if int(ph.get("pokes") or 0) > 0:            # 扎针真的扎了
            return True
    return False

# 锐度按三态处理：量到但无阈值与明确不合格不同，不能把所有非通过状态用于继续干预。
_SHARPNESS_VERDICT_KIND: dict[str, str] = {
    "no_step": "undecidable",     # 图里没台阶
    "measured": "undecidable",    # 没有阈值可比
    "unresolved": "undecidable",  # 阈值细过采样极限
    "sharp": "pass",
    "blunt": "fail",
}

#: 每个「判不了」的态,给的下一步都不一样 —— 所以理由不能只有一句通用话。
_UNDECIDABLE_REASON: dict[str, str] = {
    "no_step": "这张图里没有台阶,边缘锐度无从测量(不是不合格)。",
    # 边缘宽度受像素栅格和平滑核影响，设置验收阈值前需独立验证采样分辨率与判据适用性。
    "measured": ("量到了台阶边缘宽度,但没有判定阈值,因此不能判定合格。"
                 "像素栅格和算法平滑会影响边缘宽度；必须先确认分辨率与判据的适用性,"
                 "再依据目标仪器的独立标定设置阈值。"),
    "unresolved": "验收图分辨率不足,阈值落在采样极限以下,无法判读(不是不合格)。",
}


def sharpness_verdict_kind(verdict: object) -> str:
    """``AssessTipSharpness`` 的 verdict → ``"pass"`` / ``"fail"`` / ``"undecidable"``。

    **认不出的一律当 ``"undecidable"``,不当 ``"fail"``。** 这是本函数存在的全部理由:
    上一版把「没想到的状态」默默算作不合格,而一个没想到的状态恰恰是最不该拿来
    下否定结论的东西。词汇表由 :data:`mast.skills.builtins.tip_sharpness.
    SHARPNESS_VERDICTS` 给出,覆盖性由 ``test_sharpness_verdict_coverage`` 钉住。
    """
    return _SHARPNESS_VERDICT_KIND.get(str(verdict or "").lower(), "undecidable")


def _looks_like_operator_stop(reason: str) -> bool:
    """失败原因是不是「扫描被停下了」。见 :data:`_OPERATOR_STOP_MARKERS`。"""
    text = str(reason or "")
    return any(m in text for m in _OPERATOR_STOP_MARKERS)


#: ``_forge_wf`` 会从 params 里读的键。**每一个都必须是 ForgeAuTip 声明过的参数。**
#:
#: 缺陷⑮(2026-08-06 首演):这张单子原来有 22 个键,而 ForgeAuTip 只声明了其中 6 个。
#: 剩下 16 个是**死读**:工具 schema 里没有它们,模型传了会被静默丢弃(pydantic 默认
#: ``extra=ignore``),于是值一路回落到流程表 —— 而门控照着流程表的数字拒绝。用户
#: 连试 `poke_depth_nm` 0.5 / 0.4,得到的都是同一句「超出 0.5 nm 包络」,里面那个
#: 1.5 从来不是他传的值。**从调用方和从技能内部都看不出问题在哪**:一边以为传了,
#: 一边以为没传。
#:
#: 死读比不读更坏:它让代码读起来像个可调项,而调它不会有任何效果。
#: 这张单子由测试对着 ``metadata()`` 逐条核 —— 加读一个键就必须同时声明它。
_WF_PARAM_KEYS: tuple = (
    "pulse_v", "pulse_budget", "pulse_same_spot_budget",
    "poke_budget", "fwdbwd_threshold",
    "poke_depth_nm", "poke_dwell_s",
    "forge_scan_nm", "forge_step_scan_nm", "forge_scan_timeout_s",
    "forge_pixels", "forge_line_time_s",
)


def _forge_wf_notes(params: dict) -> "tuple[NobleTipWorkflow, list[str]]":
    """从基础流程生成 forge 工作点，未覆写项保持原配置；实际参数由统一解析链和安全包络约束。"""
    wf = resolve({k: params.get(k) for k in _WF_PARAM_KEYS})
    # 默认工作点需与当前针尖的安全包络对账，不能因未显式传参绕过包络。
    wf, envelope_notes = reconcile_with_tip_envelope(
        wf, specified={k for k in _WF_PARAM_KEYS if params.get(k) is not None})
    # 未传入覆写值时使用流程配置；配置不代表目标仪器已标定。
    pixels = wf.forge_pixels
    line_time_s = wf.forge_line_time_s
    forged = replace(
        wf,
        verify_scan_nm=wf.forge_scan_nm,
        # 验证帧分辨率作为独立配置传递，不能无意覆盖当前扫描状态。
        verify_pixels=(pixels if pixels is not None else wf.forge_verify_pixels),
        verify_line_time_s=(line_time_s if line_time_s is not None
                            else wf.forge_verify_line_time_s),
        step_scan_nm=wf.forge_step_scan_nm,
        step_fallback_nm=wf.forge_step_fallback_nm,
        # ⚠️ 找落点那张图也必须听「一律」。2026-08-14 新加 ``poke_site_*`` 时
        # 这两行漏接了,于是用户传 forge_line_time_s=0.4 时它仍走 0.586 ——
        # **「一律」这个词第三次成为假话**(前两次:verify 收不到覆写、回退图按
        # 视野缩放覆盖了他钉的数)。加一张图就要回来看这里,这是第三个必改点。
        poke_site_pixels=(pixels if pixels is not None else wf.poke_site_pixels),
        poke_site_line_time_s=(line_time_s if line_time_s is not None
                               else wf.poke_site_line_time_s),
        cluster_scan_nm=wf.forge_cluster_scan_nm,
        scan_timeout_s=wf.forge_scan_timeout_s,
        step_pixels=(pixels if pixels is not None else wf.forge_step_pixels),
        step_line_time_s=(line_time_s if line_time_s is not None
                          else wf.forge_step_line_time_s),
        cluster_pixels=(pixels if pixels is not None else wf.forge_cluster_pixels),
        cluster_line_time_s=(line_time_s if line_time_s is not None
                             else wf.forge_cluster_line_time_s),
    )
    return forged, envelope_notes


def _forge_wf(params: dict) -> NobleTipWorkflow:
    """只要工作流(不关心对账说明)。见 :func:`_forge_wf_notes`。"""
    return _forge_wf_notes(params)[0]


class ForgeAuTip(CompositeSkillGraph):
    """Au(111) 上的完整修针外环:修 → 验 → 不行就换位置 → 再修,直到达标或预算耗尽。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ForgeAuTip",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            # 全量声明。SAFE 模式的门控靠的就是这两个 tag:不填的声明式流程会让
            # SAFE 形同虚设(safe_mode_tip_override 的教训),而这条外环里既打脉冲
            # 又扎针,两样都要声明。
            capabilities=frozenset({"bias_pulse", "tip_shaping"}),
            # CONFIRM:一次批准跑完整条(子步骤不重新过门),这正是用户要的粒度
            # —— 一次修针要打几十发脉冲、扎几十次、换好几个位置。真正危险的动作
            # 各有各的闸:脉冲过针尖包络、下压在 qPlus 上默认拒绝、粗动驱动电压有
            # Layer 0e 硬闸、SAFE 模式整类硬拦。
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "Au(111) 全流程修针：在当前站点执行电脉冲、扫图验证、调平和扎针精修。"
                "本站预算耗尽后可粗动换站并重新进针；总预算耗尽时如实报告未达标。"
                "收尾回读偏压、电流设定及 Z 反馈；读不到就报告缺失。工作点不自动恢复为入场条件。"
                "本流程用于针尖顶端已极其糟糕、明确需要重新锻造的情况，不能仅因常规修针失败而升级。应再给常规修针一次机会；代价高不等于成功率高。"
                "表面区域问题交给 RelocateCoarseXY；常规修针可使用 MakeAtomicResolutionTip。"
                "单次处理可分别使用 PrepareNobleTip、PulseConditionTip 或 PokeConditionTip。"
                "不确定处理档位时，使用 AchieveAtomicResolution 根据当前读数选择。"
            ),
            parameters=[
                # ⚠️ 这三个**都不给 default** —— 有了 default,「没传」就再也不是
                # 「没传」了(``prescan_check.py:96-102`` 记着那个坑的原形)。
                # 这里「没传」的语义分别是「用 12 h」和「没有计数上限」。
                ParameterSpec(
                    name="time_budget_h", type="float", unit="h",
                    description=(
                        f"整条流程的时间预算(小时)。留空取 {_DEFAULT_BUDGET_H} h。"
                        "**这是唯一的常规上限** —— 在这台仪器上换站(粗动 XY + 重新"
                        "进针)只花约一分钟且不撞针,所以预算不按站点数计。"
                        "跑满的语义是**请人来看看是怎么回事**,不是失败;"
                        "报告刻意不建议加大预算重跑。断点续跑时**累加**,不重新起算。"),
                    required=False, min_value=0.1, max_value=168.0),
                ParameterSpec(
                    name="max_sites", type="int",
                    description=(
                        "额外的站点数硬顶。**留空 = 没有上限**(默认路径,由 "
                        "time_budget_h 管着)。2026-08-14 之前它出厂是 3,而那个 3 "
                        "不对应任何物理量 —— 换一站的真实成本是约一分钟的重新进针,"
                        "不是一段不可再生的样品表面。只有你想明确限制站点数时才填。"),
                    required=False, min_value=1, max_value=200),
                ParameterSpec(
                    name="max_rounds_per_site", type="int",
                    description=(
                        "每站「大修 ⇄ 验证」的轮数硬顶。**留空 = 没有上限**。"
                        "留空时这一站什么时候结束由判据决定(验证通过 / 表面用完 / "
                        "判不了 / 时间到),不由一个计数决定。"),
                    required=False, min_value=1, max_value=100),
                ParameterSpec(
                    name="force_repair", type="bool",
                    description=(
                        "**跳过「到站先验」,先打一轮脉冲再说。**\n"
                        "默认 false:到一个新站点先扫一张验证图,针尖已达标就一发"
                        "不打,直接去找台阶 + 扎针精修(2026-08-17 加 —— 针尖"
                        "本来就是好的,不需要额外的 pulse)。\n"
                        "什么时候要设 true:**你知道针尖不好,而判据说它好**。"
                        "判据看的是正反扫描线重合度(阈值 0.80),它量不到的"
                        "针尖毛病(比如轻微双针尖)会让这一步放行。"),
                    required=False, default=False),
                ParameterSpec(
                    name="forge_scan_nm", type="float", unit="nm",
                    description=(f"外环内验证扫图的视野(快档)。留空取 "
                                 f"{_WF.forge_scan_nm} nm。修针评估要的是快,不是好看。"),
                    required=False, min_value=1.0, max_value=5000.0),
                ParameterSpec(
                    name="forge_step_scan_nm", type="float", unit="nm",
                    description=(f"找台阶的视野(快档)。留空取 "
                                 f"{_WF.forge_step_scan_nm} nm。"),
                    required=False, min_value=1.0, max_value=5000.0),
                ParameterSpec(
                    name="forge_scan_timeout_s", type="float", unit="s",
                    description=(f"外环内单帧扫描的等待上限。留空取 "
                                 f"{_WF.forge_scan_timeout_s} s。"),
                    required=False, min_value=1.0, max_value=3600.0),
                # 2026-08-10:没有这两个参数时,agent 为了保住扫描参数控制权会
                # **放弃这个技能**、改用原语手工拼外环,一个原语一次模型调用,
                # 30 次到顶被砍,仪器停在中途。没有参数不等于模型不会填数字 ——
                # 它会绕过整个技能去填。契约照 ScanAt 的 line_time_s / pixels:
                # 开出来,但只有用户逐字说过才传。
                ParameterSpec(
                    name="forge_pixels", type="int", unit="px",
                    description=(
                        f"外环内每张评估图的线数。留空使用流程表："
                        f"台阶/验收图 {_WF.forge_step_pixels}，簇图 {_WF.forge_cluster_pixels}。"
                        "只有用户明确指定时才填；默认工作点需要在目标仪器上验证。"),
                    required=False, min_value=16, max_value=4096),
                ParameterSpec(
                    name="forge_line_time_s", type="float", unit="s",
                    description=(
                        f"外环内**每一张**评估图的每线时间。留空按流程表"
                        f"({_WF.forge_step_line_time_s} s，需在目标仪器验证)。"
                        "**只有用户逐字说过这个数才填**;留空是默认路径。"
                        "调大 = 扫得更稳更慢,调小 = 更快但反馈可能跟不上。"),
                    required=False, min_value=1e-4, max_value=600.0),
                ParameterSpec(
                    name="relocate_steps", type="int",
                    description=("换位时的粗动步数。**留空是默认路径**:由粗动大地图"
                                 "按已访问站点算出方向与步数(它同时管着单轴行程预算)。"
                                 "只有用户逐字说过步数时才填。"),
                    required=False, min_value=1, max_value=100_000),
                ParameterSpec(
                    name="allow_on_qplus", type="bool",
                    description=("这个开关**默认就是放行的**,不必设 —— nm 尺度的下压是常规手段。"
                                 "只有把护栏开回来(MAST_QPLUS_POKE_GUARD=1)时它才有作用。"
                                 "护音叉的是深度包络(超了拒绝不夹紧)与扎针前把偏压降到 20 mV,"
                                 "不是这个开关。"),
                    required=False, default=False),
                # 内环判据的透传口:不填一律走贵金属流程表。
                ParameterSpec(
                    name="pulse_v", type="float", unit="V",
                    description=(f"脉冲电压幅值。留空按流程表({_WF.pulse_v} V),"
                                 "并且**仍会过当前针尖的安全包络**(超上限拒绝不夹紧)。"),
                    required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="pulse_budget", type="int",
                    description=f"单轮大修最多打几发。留空按流程表({_WF.pulse_budget})。",
                    required=False, min_value=1, max_value=200),
                ParameterSpec(
                    name="pulse_same_spot_budget", type="int",
                    description=(
                        "**没打动针尖**时同一落点上最多连打几发;打成功的那一发"
                        f"永不豁免。留空按流程表({_WF.pulse_same_spot_budget})。"),
                    required=False, min_value=1, max_value=50),
                ParameterSpec(
                    name="poke_budget", type="int",
                    description=f"单站精修最多扎几次。留空按流程表({_WF.poke_budget})。",
                    required=False, min_value=1, max_value=200),
                # ⑮(2026-08-06 首演):这两个**本来就在被读**(`_forge_wf` 的键单里),
                # 只是从没声明过 ⇒ 工具 schema 里没有它们 ⇒ 模型传的值被**静默丢弃**,
                # 于是深度一路回落到流程表的 1.5 nm,门控照着 1.5 拒绝。用户连试
                # 0.5 / 0.4 都被同一句「超出 0.5 nm 包络」挡回来,而那句话里的数字
                # 从来不是他传的那个 —— 从两端都看不出问题在哪。
                ParameterSpec(
                    name="poke_depth_nm", type="float", unit="nm",
                    description=(f"精修浅扎深度。留空按流程表({_WF.poke_depth_nm} nm),"
                                 "并且**仍会过当前针尖的安全包络**(超上限拒绝不夹紧)"
                                 " —— qPlus 上的包络比表值小得多,这个参数正是为那种"
                                 "情况准备的。"),
                    required=False, min_value=0.01, max_value=100.0),
                ParameterSpec(
                    name="poke_dwell_s", type="float", unit="s",
                    description=f"浅扎驻留时间。留空按流程表({_WF.poke_dwell_s} s)。",
                    required=False, min_value=0.0, max_value=60.0),
                ParameterSpec(
                    name="fwdbwd_threshold", type="float",
                    description=(f"正反扫描线相似度下限(验证判据)。留空按流程表"
                                 f"({_WF.fwdbwd_threshold})。"),
                    required=False, min_value=0.0, max_value=1.0),
            ],
            preconditions=["bias_nonzero"],
            estimated_duration_s=5400.0,
            composition_level=4,
            tags=["tip", "conditioning", "forge", "au111", "coarse", "composite"],
        )

    def validate_params(self, params: dict) -> list[str]:
        """标准校验 + 下压深度的针尖包络(超上限拒绝,不夹紧)。

        与 ``PokeConditionTip`` 同款:精修阶段真的会下压,而包络永远赢。在这里拒绝
        比跑到第 40 分钟才拒绝便宜得多。"""
        errors = super().validate_params(params)
        depth_nm = params.get("poke_depth_nm")
        if depth_nm is None:
            depth_nm = _WF.poke_depth_nm
        try:
            from mast.skills.builtins._tip_policy import apply_tip_policy

            _, plan = apply_tip_policy(
                {"poke_deep_depth_m": -abs(float(depth_nm)) * 1e-9},
                ("poke_deep_depth_m",))
            if plan is not None and not plan.ok:
                errors.extend(plan.refusals)
        except Exception:  # noqa: BLE001 — 包络读不到不是拒绝的理由,别把流程卡死
            logger.debug("ForgeAuTip: 针尖包络检查不可用", exc_info=True)
        return errors

    # ── 外环 ────────────────────────────────────────────────────────────

    def plan_dynamic(self, params: dict, executor) -> Iterator[CompositeStep]:
        wf, envelope_notes = _forge_wf_notes(params)
        # ── 2026-08-14:计数预算 → 时间预算 ──────────────────────────────
        #
        # 换位与重新进针的成本模型:
        #   · 有 xy 移动的仪器上,xy 移动的成本几乎为 0 —— 只用中间蠕变小的
        #     区域,用完就换地方。
        #   · 重新进针同样没有成本:本机进针不撞,耗时约一分钟。因此站点数与
        #     轮数都不必设上限;唯一的上限是总时间,过长则需要人来看一眼。
        #     总时间可调,默认 12 小时。
        #
        # 于是 ``max_sites`` × ``max_rounds_per_site`` 这个**记在错单位上的预算**
        # 退成可选:不传 = 没有计数上限。W4 那次跑完 116 分钟报 sites_exhausted,
        # 停手的理由不是「没地方去了」,是**数到 3 了** —— 而那个 3 不对应任何物理量。
        budget_h = params.get("time_budget_h")
        budget_s = float(_DEFAULT_BUDGET_H if budget_h is None
                         else budget_h) * 3600.0
        max_sites = int(params["max_sites"]) if params.get("max_sites") else None
        max_rounds = (int(params["max_rounds_per_site"])
                      if params.get("max_rounds_per_site") else None)
        force_repair = bool(params.get("force_repair"))

        # 起算时刻**只设一次** —— 续跑时累加,不重置。
        # 每次 resume 都从零起算的话,一个反复被打断的跑可以无限进行下去,
        # 预算就等于没有。``set_partial_default`` 正是「已有就别覆盖」。
        executor.set_partial_default("started_at", _time.time())
        try:
            started = float(executor.progress.partial_data.get("started_at"))
        except (TypeError, ValueError):          # 读不到就以现在起算,并说出来
            started = _time.time()
            executor.set_partial("started_at_unreadable", True)

        def _left_s() -> float:
            """还剩多少秒。**读不到时刻不猜** —— 上面已经兜住了。"""
            return budget_s - (_time.time() - started)

        # 战绩**增量落账**:每完成一步就写进 partial_data。外环随时可能被
        # abort(用户停止 / 物理硬线 / 某个必需子步骤失败)—— 那时生成器直接被
        # close(),后面一行都不会跑。报告攒到最后再写,等于「一被打断就什么都没
        # 有」,而被打断恰恰是最需要知道刚才做了什么的时候。
        sites: list[dict[str, Any]] = []
        self._stash(executor, sites, outcome="incomplete")
        # ⚠️ **这里不做回滚,而且这一层做不了**:executor 中止时是从遍历这个生成器的
        # for 循环里 return 的,生成器随后被 close(),后面一行都不会跑 —— 与
        # ``poke_phase`` 末尾那段「不能写成 try/finally」是同一个结构事实。要真的回滚
        # 只能落在 executor 的 abort 路径(``_ABORT_SAFE_WRITES``)上,那是另一个模块
        # 的决定。见 docs/v2/design/forge_scan_working_point.md §九。
        if envelope_notes:
            executor.set_partial("envelope_notes", list(envelope_notes))
        # 每张评估图的针尖横向速度**逐帧写进报告**。
        #
        # 这不是装饰:667 nm/s 在流程表里躺了五天不是因为没人读代码,是因为**代码
        # 里看不出来** —— 线时和视野分别住在两个字段里,要把商算出来才看得见。
        # 所以这一账由代码每一跑算一次交给用户,而不是留给下一个人心算。
        #
        # ⚠️ **单独一个键,不搭 ``envelope_notes`` 的便车。** 第一版把它们拼进同一
        # 个列表,而 ``_summary`` 给那个列表里的每一条都加前缀「参数按当前针尖的
        # 安全包络调整过」—— 于是五条速度里有四条**根本没被调整过**的会被印成
        # 「调整过」。那正是本仓记过的「字段标签会说谎」:标签说了代码没做的事,
        # 而用户会照着那句话去调一个没在生效的数。
        executor.set_partial("scan_speed_notes", forge_speed_notes(wf))
        # 计划值只表示预期工作点；最终仪器状态必须来自回读。
        executor.set_partial("intended_junction", {
            "bias_v": wf.junction_bias_v,
            "setpoint_a": wf.junction_setpoint_a,
            "what": "修针结条件(低偏压 + 大电流,针尖离表面很近)",
            "is_intent_not_reading": True,
        })

        if _tip_shaper_preflight(executor):
            return
        if _qplus_blocked(executor, "ForgeAuTip", params):
            return

        # 电流监控自检的水位线,**整条外环一个**,不是每站一个。
        #
        # 它必须在第一站开工**之前**就起算,而且跨站不重建 —— 否则两段最危险的时间
        # 会掉进缝里:①第一站的第 1 轮(窗口宽度 ≈0);②站与站之间的**换位 + 重新
        # 进针**(`_relocate`)。撞针最容易发生在重新进针那一下,而那正好落在两站之间。
        #
        # (第一版把它建在 `_work_one_site` 里、循环体前一行 —— 那等于没修:
        # 构造完下一行就查,窗口仍然 ≈0。是变异验证 F6「仍然绿」把这件事顶出来的:
        # 当时那条测试直接构造 `_CriticalWatch` 自己测,**没有测调用点** ——
        # 正是本仓反复出事的「测了原语,没测可达性」。)
        mark = _CriticalWatch(_time.time())

        outcome = "incomplete"
        site_no = 0
        idle_streak = 0
        while True:
            # ── 时间预算:唯一的常规上限 ──────────────────────────────
            if _left_s() <= 0:
                outcome = "time_budget_exhausted"
                break
            # ⚠️ **失控保险,不是预算。** 去掉计数上限之后,一旦出现「每站几秒就
            # 失败」的空转,12 小时能转上万圈,而 ``_stash`` 每站都把整个 sites
            # 列表深拷进 partial_data、partial_data 又要进 checkpoint ——
            # 会先把 checkpoint 撑爆,而流程看起来一直在「努力工作」。
            # 这个数远大于任何真实需求,命中它本身就是一个 bug 的信号。
            if site_no >= _HARD_SITE_CAP:
                outcome = "hard_cap"
                break
            site_no += 1
            site: dict[str, Any] = {"site": site_no, "rounds": 0,
                                    "outcome": "incomplete", "phases": []}
            sites.append(site)
            self._stash(executor, sites, outcome=outcome)
            # 一站几十分钟,这一条是后面所有旁白的坐标系:用户看到「第 3 站」
            # 才知道刚才那批脉冲和这一批不是同一片表面上的事。
            _say("site_begin", site_no=site_no,
                 first_site=(site_no == 1),
                 entry_check=(site_no > 1 and not force_repair))

            verdict = yield from self._work_one_site(executor, wf, params,
                                                     site=site, site_no=site_no,
                                                     max_rounds=max_rounds,
                                                     force_repair=force_repair,
                                                     sites=sites, mark=mark,
                                                     left_s=_left_s)
            # 站点结局:**一个出口**说一次。``_work_one_site`` 里有十几处
            # ``site["outcome"] = ...; return``,在每一处各发一条就等于让下一个
            # 加出口的人记得也加一条旁白 —— 本仓为「每处各自记得」付过四次学费。
            _say("site_result", site_no=site_no,
                 outcome=str(site.get("outcome") or verdict),
                 rounds=site.get("rounds"))
            if verdict == "ready":
                outcome = "ready"
                break

            # 电流监控的 CRITICAL 是**整条流程**的停止理由,不是「这一站不行」。
            # 换到下一站再扎一次,是在一根可能已经压进表面的针上继续动手 ——
            # 而换位本身(RelocateCoarseXY)也要用这根针。这里必须 break 掉外环,
            # 不能落进下面那条「果断换位」的路。
            if verdict == "critical_alert":
                outcome = "critical_alert"
                break
            if verdict == "time_budget_exhausted":
                outcome = "time_budget_exhausted"
                break

            # ── 空转判据:连续几站**什么都没测到**就停 ────────────────
            #
            # 这不是「没修好」,是「没在修」。典型成因是一个与站点无关的故障
            # (读不到线数据 / 找不到干净落点 / 通信坏了)—— 换地方对它毫无帮助,
            # 而流程会一直换下去直到时间耗尽,最后报一句「未达标」,
            # 那是一句**关于针尖的话**,而这一跑根本没测过针尖。
            if _site_measured_something(site):
                idle_streak = 0
            else:
                idle_streak += 1
                if idle_streak >= _IDLE_SITE_LIMIT:
                    outcome = "spinning"
                    break

            # ⏱ **换位之前再看一次表。** 换位 = 粗动 + 重新进针,约一分钟,而且
            # 会把针尖挪到一个陌生位置。时间检查原来只在循环开头,于是可能出现
            # 「换完位、重新进针、回到循环开头、发现超时、撒手」—— 那一分钟纯属
            # 浪费,而且**停手时针尖停在一个刚到的、还没测过的地方**。
            # 宁可在原地停:至少这一站的战绩是完整的。
            if _left_s() <= 0:
                outcome = "time_budget_exhausted"
                break
            # ⚠️ 计数上限**必须在这里查,不能在循环开头**(2026-08-14 改成 while
            # 时挪错了位置,测试当场逮到:`RelocateCoarseXY` 调了 2 次而不是 1 次)。
            # 原来的 for 循环把它放在 relocate 之前正是这个道理:最后一站不该再换位
            # —— 换了也不会再用,白花一分钟粗动+重新进针,还把针尖停在一个
            # **没测过的陌生位置**上。宁可停在刚测完的地方。
            if max_sites is not None and site_no >= max_sites:
                outcome = "sites_exhausted"
                break

            # 果断:到这里就是「这一站不行」。没有「再试试」的询问环节 ——
            # 判据已经在上面命中了(表面用完 / 轮数耗尽 / 验收没过)。
            moved = yield from self._relocate(executor, params,
                                              step_prefix=f"S{site_no}:move",
                                              sites=sites)
            # 换区**必定重新进针**,是这条流程里最重的一个动作(约一分钟,而且
            # 把针尖挪到一个陌生位置)。它此前一个字都没有 ——「在原地反复打脉冲」
            # 正是这一类:动作换了、旁白没有。
            _say("relocate_result", moved=(moved is True),
                 outcome=("" if moved is True else str(moved)))
            if moved is not True:
                outcome = str(moved)      # coarse_budget_exhausted / relocate_failed
                break

        executor.set_partial("elapsed_s", round(_time.time() - started, 1))
        executor.set_partial("time_budget_s", budget_s)
        self._stash(executor, sites, outcome=outcome)
        _say("run_result", outcome=str(outcome), sites=len(sites))

    # ── 一个站点上的完整内环 ─────────────────────────────────────────────

    def _work_one_site(self, executor, wf: NobleTipWorkflow, params: dict, *,
                       site: dict, site_no: int, max_rounds: "int | None",
                       force_repair: bool = False,
                       sites: list, mark: "_CriticalWatch",
                       left_s) -> Iterator[str]:
        """返回 ``"ready"``(验收通过)或换位理由。"""
        pfx = f"S{site_no}"
        descend = descend_sequence(wf)

        # ── 大修 ⇄ 验证 ────────────────────────────────────────────────
        # 每轮开工前看一眼电流监控(2026-08-10)。
        #
        # 为什么外环要自己看:2026-08-08 一条 CRITICAL saturation 报出来之后,
        # agent 又扫了 13 分钟 —— 那一轮的对照实验结论与下一轮的锐度数字全部作废。
        # 这条流程比那更脆弱:它**一轮一轮地扎、验、扫**,针尖中途崩了没人告诉它,
        # 它会把剩下的轮次全部跑在废数据上,而且每一轮还要再扎一次。
        #
        # 水位线由**外环**持有并贯穿整条流程(见 ``plan_dynamic``):只问「上次看过
        # 之后有没有新的」。用绝对回看窗口会让流程刚开工就被十分钟前、早已处置过的
        # 一条 CRITICAL 卡住;而每站重建一个水位线会把**换位与重新进针那一段**整个
        # 漏掉 —— 那恰恰是最容易撞针的一段。
        rnd = 0
        while True:
            # 时间预算优先于一切计数 —— 它是 2026-08-14 起唯一的常规上限。
            if left_s() <= 0:
                site["outcome"] = "time_budget_exhausted"
                return "time_budget_exhausted"
            # 显式给了才生效(默认没有轮数上限)。
            if max_rounds is not None and rnd >= max_rounds:
                site["outcome"] = "verify_exhausted"
                return "verify_exhausted"
            # 失控保险,同 ``_HARD_SITE_CAP``:一轮 = 一批脉冲 + 一张验证图,
            # 正常至少几分钟;真跑到这个数说明有东西在空转。
            if rnd >= _HARD_ROUND_CAP:
                # ⚠️ **不能报 verify_exhausted** —— 那句话是「反复大修后正反扫描线
                # 仍不重合」,一句**关于针尖的结论**。而这里发生的是失控保险触发,
                # 没有任何判据说过针尖不好。同 `_summary` 里那条纪律
                # (没测过就不能说针尖如何),这一处 2026-08-14 犯过一次。
                site["outcome"] = "round_hard_cap"
                return "round_hard_cap"
            rnd += 1
            if (hit := mark.check(site, f"第 {rnd} 轮开工前", round_no=rnd)):
                return hit
            site["rounds"] = rnd

            # ═══════════════════════════════════════════════════════════════
            # 到一个新站点:**先看针尖好不好,别一上来就打脉冲**(2026-08-17)
            # ═══════════════════════════════════════════════════════════════
            #
            # 粗动换区不应自动带一个 pulse —— 针尖本来是好的,
            # 粗动只是为了找个干净地方扎针,不需要额外的 pulse。
            #
            # 指的就是这里。原来的循环是
            #
            #     while True:
            #         pulse_phase(...)      ← 无条件
            #         verify_phase(...)     ← 判针尖好不好的那一步在后面
            #         if passed: break
            #
            # 于是**每到一个新站点必定先挨一批脉冲**,然后才第一次去看针尖。
            # 粗动换区之后就是新站点,所以「换区必带 pulse」。
            # 而换区的常见理由恰恰是「针尖是好的,只是这片表面用完了」——
            # 那一批脉冲既毁了一根好针,又白花几分钟。
            #
            # ⇒ 本站第一轮先验。验过就直接进 C/D(找台阶 + 精修),一发不打。
            #
            # 代价:针尖确实坏时多扫一张验证图 —— 而那张图本来第一轮结束也要扫,
            # 只是提前了。**换不到第二张图的代价,却换掉了「打一根好针」的风险。**
            #
            # ⚠️ 三态照旧:``inconclusive``(读不到正反扫描线)**不是**「针尖不好」,
            # 所以它不许触发脉冲 —— 走下面既有的那条 verify_inconclusive 出口。
            #
            # ``force_repair=true`` 整个跳过这一步:**你知道针尖不好,而判据
            # 说它好**。判据看的是正反扫描线重合度(阈值 0.80),它量不到的毛病
            # (轻微双针尖之类)会让这一步放行 —— 那时逃生门比判据管用。
            # ⚠️ **第一站不许跳过** —— 第一个脉冲一定是强制的。
            #
            # 到站先验是为了解决「粗动换区都会自动带一个 pulse」那件事,而那件事
            # 的前提是**针尖状态已经在上一站量过了**。整条流程刚开工时没有这个
            # 前提:用户发起修针,就是因为他认为针尖需要修 —— 让一个阈值 0.80
            # 的重合度判据把他的判断否掉,是「未标定的阈值否决人」的又一次。
            #
            # ⇒ ``site_no == 1`` 一律先打;从第二站起(必定是粗动换区来的)才先验。
            entry_check = (site_no > 1) and not force_repair
            skipped_pulse = False
            if rnd == 1 and entry_check:
                verify = yield from verify_phase(executor, wf,
                                                 prefix=f"{pfx}B0")
                site["phases"].append(verify)
                if verify.get("passed"):
                    # 使用产生方实际返回的字段，缺失值不得静默解释为动作已完成。
                    site["pulse_skipped_reason"] = (
                        "到站先验:针尖已达标(" + str(verify.get("reason") or "")[:60]
                        + "),本站一发脉冲都没打")
                    skipped_pulse = True
                elif verify.get("surface_spent"):
                    # 「这片表面没干净落点了」和「针尖判不了」是两句都成立的话,
                    # 而**要报出去的是前者** —— 它才说得清下一步(换区),
                    # 后者只说得清「别打脉冲」。verify 也会置 inconclusive,
                    # 所以这一支必须排在它前面,否则永远轮不到。
                    site["outcome"] = "surface_spent"
                    self._stash(executor, sites)
                    return "surface_spent"
                elif verify.get("inconclusive"):
                    site["outcome"] = "verify_inconclusive"
                    self._stash(executor, sites)
                    return "verify_inconclusive"
                self._stash(executor, sites)
            elif rnd == 1:
                site["pulse_skipped_reason"] = None
                site["forced_repair"] = (
                    "force_repair=true:跳过到站先验,直接大修" if force_repair
                    else "第一站:修针的第一发脉冲是强制的,不做到站先验")
            if skipped_pulse:
                break

            seq = descend if (rnd > 1 and descend) else None
            pulse = yield from pulse_phase(executor, wf, prefix=f"{pfx}A{rnd}",
                                           voltage_sequence=seq)
            site["phases"].append(pulse)
            self._stash(executor, sites)
            if pulse.get("surface_spent"):
                site["outcome"] = "surface_spent"
                return "surface_spent"

            verify = yield from verify_phase(executor, wf, prefix=f"{pfx}B{rnd}")
            site["phases"].append(verify)
            self._stash(executor, sites)
            if verify.get("passed"):
                break
            if verify.get("surface_spent"):
                # 见上面那一处:``surface_spent`` 必须排在 ``inconclusive`` 前面。
                site["outcome"] = "surface_spent"
                return "surface_spent"
            if verify.get("inconclusive"):
                # 「读不到正反扫描线」不是「针尖不好」。当成不好会让流程接着去打
                # 脉冲,而根本原因可能只是没读到线数据 —— 那样打下去是拿一根可能
                # 好好的针尖去撞一个测量问题。换个地方重新量,是这里唯一诚实的动作。
                site["outcome"] = "verify_inconclusive"
                return "verify_inconclusive"
            # (原来这里是 ``for ... else``:轮数用完 → verify_exhausted。
            #  2026-08-14 轮数上限退成可选之后,那两条出口挪到了循环开头 ——
            #  和时间预算、失控保险排在一起,所有「为什么停」都在同一个地方看得见。)

        # ── 找台阶 + 调平(验收判据要有台阶才量得出来)──────────────────
        #
        # 合并:verify 那张图现在就是 100 nm(``forge_scan_nm``),
        # 和本相原本要扫的是同一种图 —— 直接复用,一圈省一张图、省 5 分钟,
        # 两张图因此合并成了一张。
        #
        # ``reuse`` 传不出去时(SaveScan / GetLatestScanFile 是 optional,存不下来
        # 就没有路径)``level_phase`` 会自己扫一张 —— 退回合并前的行为,
        # **不会因为省图而丢功能**。
        vcenter = verify.get("scan_center") or []
        reuse = ((verify.get("scan_path"), vcenter[0], vcenter[1])
                 if verify.get("scan_path") and len(vcenter) == 2 else None)
        level = yield from level_phase(executor, wf, prefix=f"{pfx}C", reuse=reuse)
        site["phases"].append(level)
        if level.get("split_tip"):
            # 找台阶那张大图上台阶被劈开 ⇒ **针尖的问题**,就地打一轮脉冲。
            # 多针尖就直接 pulse。
            #
            # 不 return(那会让外环粗动换区,而换区解决的是「表面用完」);
            # 也不 continue(这里不在轮次循环里)。就地补一轮 A,然后照常进 D ——
            # D 自己还有「连续不圆就回退打脉冲」那条兜底。
            site["split_tip"] = True
            rescue = yield from pulse_phase(executor, wf, prefix=f"{pfx}RS")
            site["phases"].append(rescue)
            self._stash(executor, sites)
            if rescue.get("surface_spent"):
                site["outcome"] = "surface_spent"
                return "surface_spent"
        self._stash(executor, sites)
        step_center = level.get("wide_scan_center")

        # ── 精修 ───────────────────────────────────────────────────────
        # 精修之前**也要查一次**:大修⇄验证那一段可能刚把针压坏,而精修是这条流程
        # 里唯一会反复深扎的一段 —— 带着一根已经贴住表面的针进去,是这里最贵的动作。
        # (原来这道查只在 ``for rnd`` 里,精修与验收完全不经过它。)
        if (hit := mark.check(site, "精修开始前")):
            return hit

        # ⏱📊 **精修途中的停手判据** —— 两个洞一次补上。
        #
        # 在这之前:时间预算只在内环(A⇄B)开头查,电流监控的水位线只在「精修开始
        # 前」查一次。而 D 相最多 30 次扎针 ≈ 42 分钟,是整条流程里唯一反复把针
        # 压进表面的一段。于是超时最多能超 45 分钟,而途中崩了的针会被继续扎完。
        #
        # 判据留在这里(它知道预算和告警),``poke_phase`` 只负责每次扎针前问一句。
        def _poke_stop() -> str:
            if left_s() <= 0:
                return "time_budget_exhausted"
            if mark.check(site, "精修途中"):
                return "critical_alert"
            return ""

        # ── 精修 ⇄ 回退打脉冲 ──────────────────────────────────────────
        #
        # 老是不出圆团簇要回退到 pulse。
        #
        # 扎针改的是**形状**;一根被打钝/打歪的针,再换多少地方扎也出不来圆簇 ——
        # 一直扎下去是拿位置的问题当形状的问题解,而且每一针都在消耗表面。
        # 所以连续 ``poke_unround_streak_to_pulse`` 针不圆 ⇒ 回去打一轮脉冲,
        # 再回来扎。**不是换区**:换区解决的是「表面用完」,这里是「针不行」。
        # 直接读字段,不用 getattr —— ``test_every_workflow_field_has_a_consumer``
        # 走 AST 找消费方,``getattr(wf, "x")`` 它看不见,于是这个字段会被
        # 判成「读起来像在生效的死配置」。那道闸门是对的。
        rescues = int(wf.poke_pulse_rescues or 0)
        for _try in range(rescues + 1):
            poke = yield from poke_phase(executor, wf, prefix=f"{pfx}D{_try}",
                                         should_stop=_poke_stop)
            site["phases"].append(poke)
            self._stash(executor, sites)
            # 中途停手的两种理由都是**整条流程**的事,不是「这一站不行」——
            # 尤其 critical_alert:带着一根可能已经压进表面的针换站再扎,是最贵的动作。
            stopped = str(poke.get("stopped_early") or "")
            if stopped in ("time_budget_exhausted", "critical_alert"):
                site["outcome"] = stopped
                return stopped
            if poke.get("surface_spent"):
                site["outcome"] = "surface_spent"
                return "surface_spent"
            if not poke.get("needs_pulse"):
                break
            if _try >= rescues:
                # 回退额度用完 —— 说清楚是**为什么**停,别报成「针尖不好」。
                site["outcome"] = "pulse_rescue_exhausted"
                site["poke_unround_streak"] = poke.get("unround_streak")
                return "pulse_rescue_exhausted"
            if left_s() <= 0:
                site["outcome"] = "time_budget_exhausted"
                return "time_budget_exhausted"
            if (hit := mark.check(site, f"回退打脉冲前(第 {_try + 1} 次)")):
                return hit
            rescue = yield from pulse_phase(executor, wf, prefix=f"{pfx}R{_try}")
            site["phases"].append(rescue)
            site["pulse_rescues"] = _try + 1
            self._stash(executor, sites)
            if rescue.get("surface_spent"):
                site["outcome"] = "surface_spent"
                return "surface_spent"

        # ── 验收:回到有台阶的地方看边缘够不够陡 ────────────────────────
        accept = yield from self._accept(executor, wf, prefix=f"{pfx}E",
                                         center=step_center)
        site["phases"].append(accept)
        site["accept"] = accept
        self._stash(executor, sites)
        # 验收的三态此前**一个字都没进旁白**:用户看到流程扫完一张图然后结束,
        # 不知道锐度到底判了什么、还是压根没量出来。
        #
        # 在**这里**说而不是在 ``_accept`` 的六个 return 各说一次:后者等于要求
        # 下一个加出口的人记得也加一句旁白,而本仓为「每处各自记得」付过四次学费。
        _say("accept_result",
             **{k: accept.get(k) for k in
                ("kind", "verdict", "edge_resolution_nm", "sharp_edge_nm",
                 "sampling_floor_nm", "fwd_bwd_instability", "reason")})

        # 成功需要验证阶段明确通过，并完成精修阶段要求。
        # 台阶锐度只报告测量结果；缺少阈值时不能写成验收通过。
        # 明确得到 blunt 时阻止成功，缺失答案与否定测量分别处理。
        if poke.get("refined") and accept.get("kind") != "fail":
            site["outcome"] = "ready"
            return "ready"
        site["outcome"] = ("refine_incomplete" if not poke.get("refined")
                           else "sharpness_not_met")
        return site["outcome"]

    def _accept(self, executor, wf: NobleTipWorkflow, *, prefix: str,
                center) -> Iterator[dict]:
        """台阶锐度验收。**三态**:通过 / 未通过 / 量不出来。

        「量不出来」(没有台阶、拿不到文件、分析没跑成)绝不算通过 —— 那正是本仓
        反复栽的那一跤:把「判不了」记成「没问题」。它也不算「针尖不好」,所以在
        报告里它是自己一句话。"""
        # ``kind`` 是给外环看的三态:"pass" / "fail" / "undecidable"。
        # 默认 undecidable —— 下面每一条提前 return 的路都是「量不出来」,
        # 而量不出来**不是不合格**(见本方法 docstring)。只有真的拿到 verdict
        # 才会被改写。外环据此决定「拦不拦」,不再看那个会把三态压成布尔的 passed。
        out: dict[str, Any] = {"phase": "accept", "passed": False,
                               "measurable": False, "kind": "undecidable"}
        if not center:
            out["reason"] = ("没有找到台阶,量不出边缘锐度 —— 这不是「针尖不合格」,"
                             "是**这一站没法验收**。")
            return out

        yield CompositeStep(
            step_id=f"{prefix}:scan", skill_name="ScanAt",
            params=scan_at_params(wf, center[0], center[1],
                                  size_nm=wf.step_scan_nm,
                                  pixels=wf.step_pixels,
                                  line_time_s=wf.step_line_time_s,
                                  # 回到 C 相找到的台阶处 —— 验收判的就是台阶
                                  # 边缘,换个干净地方就没有台阶可量了。
                                  origin="analysis"),
            optional=True, checkpoint_after=False, tags=("accept",))
        yield CompositeStep(
            step_id=f"{prefix}:save", skill_name="SaveScan", params={},
            optional=True, checkpoint_after=False, tags=("accept",))
        yield CompositeStep(
            step_id=f"{prefix}:latest", skill_name="GetLatestScanFile", params={},
            optional=True, checkpoint_after=False, tags=("accept",))
        path = ""
        for sid in (f"{prefix}:save", f"{prefix}:latest"):
            d = _data(executor, sid)
            path = path or str(d.get("path") or d.get("file_path") or "")
        out["scan_path"] = path
        if not path:
            out["reason"] = "验收图存不下来 / 拿不到路径,锐度无从量起。"
            return out

        sharp_params: dict[str, Any] = {"scan_path": path}
        if wf.accept_sharp_edge_nm is not None:
            sharp_params["sharp_edge_nm"] = float(wf.accept_sharp_edge_nm)
        yield CompositeStep(
            step_id=f"{prefix}:sharpness", skill_name="AssessTipSharpness",
            params=sharp_params,
            optional=True, checkpoint_after=True, tags=("accept",))
        if not _ok(executor, f"{prefix}:sharpness"):
            out["reason"] = "锐度分析未能完成 —— 无法测量,不是不合格。"
            return out
        qd = _data(executor, f"{prefix}:sharpness")
        out.update({k: qd.get(k) for k in
                    ("edge_resolution_nm", "verdict", "has_step",
                     "sharp_edge_nm", "sampling_floor_nm",
                     "fwd_bwd_instability") if k in qd})
        if not qd.get("has_step"):
            out["reason"] = "这张图里没有台阶,边缘锐度无从测量(不是不合格)。"
            return out

        # no_step、measured 和 unresolved 均为无法完成锐度判定；缺少阈值不等于不合格。
        kind = sharpness_verdict_kind(qd.get("verdict"))
        out["kind"] = kind
        if kind == "undecidable":
            out["reason"] = _UNDECIDABLE_REASON.get(
                str(qd.get("verdict") or "").lower(),
                "边缘锐度无法判读(不是不合格)。") + str(qd.get("summary") or "")
            return out

        out["measurable"] = True
        out["passed"] = kind == "pass"
        out["reason"] = (f"台阶边缘锐度判定 {qd.get('verdict')!r}"
                         f"(edge_resolution {qd.get('edge_resolution_nm')} nm)")
        return out

    # ── 换位 ────────────────────────────────────────────────────────────

    def _relocate(self, executor, params: dict, *, step_prefix: str,
                  sites: list) -> Iterator[Any]:
        """果断换到下一个站点。返回 ``True`` 或一个失败 outcome 字符串。

        方向与步数**问粗动大地图要**,不在这里自己算:那张图同时管着单轴行程预算
        与站点间距,而「往哪走还有多少额度」有两个来源的话,迟早各自漂移。
        它说没有可去的方向,就是粗动预算到头了 —— 如实停,不硬走。
        """
        plan = self._relocation_plan(params)
        if plan is None:
            sites.append({"site": None, "outcome": "coarse_budget_exhausted",
                          "reason": self._last_map_note})
            self._stash(executor, sites)
            return "coarse_budget_exhausted"

        yield CompositeStep(
            step_id=f"{step_prefix}:relocate", skill_name="RelocateCoarseXY",
            params={"axis": plan["axis"], "direction": plan["direction"][-1],
                    "steps": int(plan["steps"]), "reapproach": True},
            # optional=True 与 _tip_phases._relocate 同一个理由:换位失败是外环
            # 必须自己处理的**信息**(要落进战绩、要如实报告),标成必需的话
            # executor 直接中止,生成器再也拿不回控制权,报告里连「为什么停」
            # 都没有。失败与否下一行自己查。
            optional=True, checkpoint_after=True, tags=("relocate", "coarse"))
        if not _ok(executor, f"{step_prefix}:relocate"):
            # 失败原因从执行结果传递到报告，计划描述不能替代实际失败原因。
            err = str(_err(executor, f"{step_prefix}:relocate") or "").strip()
            sites.append({"site": None, "outcome": "relocate_failed",
                          "reason": (f"粗动换位失败:{err}" if err
                                     else "粗动换位失败(技能没给失败原因)")
                                    + f"(计划:{plan})",
                          "relocate_error": err, "relocate_plan": plan})
            self._stash(executor, sites)
            return "relocate_failed"
        sites.append({"site": None, "outcome": "relocated", "plan": plan})
        self._stash(executor, sites)
        return True

    _last_map_note: str = ""

    def _relocation_plan(self, params: dict) -> "dict | None":
        """粗动大地图给的下一步。读不到地图 = 不换位(而不是瞎走一步)。

        「不知道去哪」与「去哪都行」是两回事。这台机器没有横向位置反馈,一次
        没有依据的粗动是不可撤销的。"""
        try:
            from mast.core import coarse_map_provider
            from mast.io.coarse_map import CoarseMapConfig, build_coarse_map

            rows, cfg = coarse_map_provider.markers_and_config()
            if rows is None:
                self._last_map_note = ("读不到粗动记录,不知道哪些位置已经用过 —— "
                                       "**不猜着走**。请用户看一眼大地图再决定。")
                return None
            steps = params.get("relocate_steps")
            cmap = build_coarse_map(rows, cfg or CoarseMapConfig(),
                                    steps=int(steps) if steps else None)
            self._last_map_note = cmap.note
            if cmap.suggestion is None:
                return None
            return cmap.suggestion.as_dict()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ForgeAuTip: 取不到换位建议: %s", exc)
            self._last_map_note = f"换位建议不可用({exc}) —— 不猜着走。"
            return None

    # ── 记账与报告 ──────────────────────────────────────────────────────

    def _stash(self, executor, sites: list, *, outcome: "str | None" = None) -> None:
        executor.set_partial("sites", [dict(s) for s in sites])
        if outcome is not None:
            executor.set_partial("outcome", outcome)

    def on_step_failed(self, step: CompositeStep, msg: str) -> bool:
        """记下失败原文,再交回默认策略(必需步骤照旧中止整条计划)。

        默认行为一个字没改 —— 这里只是把原文留下来,好让最后那句报告能分清
        「用户按了停止」和「预算耗尽」。"""
        try:
            self._executor.set_partial("last_failure", str(msg or ""))
        except Exception:  # noqa: BLE001 — 记账不许影响中止决策
            pass
        return super().on_step_failed(step, msg)

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        data = dict(progress.partial_data)
        sites = list(data.get("sites") or [])
        outcome = str(data.get("outcome") or "incomplete")
        if progress.aborted:
            outcome = ("stopped_early"
                       if _looks_like_operator_stop(progress.aborted_reason
                                                    or data.get("last_failure") or "")
                       else "aborted")
            data["outcome"] = outcome
        # 收尾时**去问一次仪器**,报文只说读到的东西。见 `_read_junction_now`。
        measured = self._read_junction_now()
        data["left_at_measured"] = measured
        data["summary_cn"] = _summary(outcome, sites, progress,
                                      measured=measured,
                                      envelope_notes=data.get("envelope_notes"),
                                      scan_speed_notes=data.get("scan_speed_notes"))
        data["sites_worked"] = len([s for s in sites if s.get("site")])
        return data

    def _read_junction_now(self) -> "dict | None":
        """收尾那一刻仪器的实测状态。读不到返回 ``None``(报文会说读不到)。

        为什么放在 ``aggregate`` 而不是别处:它是**唯一**在所有退出路径上都会跑的
        地方 —— 正常跑完、预算耗尽、必要步骤失败、用户中止、入口闸门早退,全都
        经过这里(``_base._graph_execute`` 在 ``run_plan`` 之后无条件调用它)。
        生成器在中止路径上已经被 close() 了,所以 ``plan_dynamic`` 里做不到这件事;
        这曾经是「只能报意图」的理由,而它只对生成器成立,对这里不成立。

        中止之后照样能读:``ExecutionContext.safe_call`` 的中止闸只拦写入,读取一律
        放行(它自己的注释:"READS always pass … a stop sequence needs to read
        status")。E_STOP 把 TCP 打死的情况下每个读会各自报错,于是报文说读不到 ——
        那也是一句真话,而「偏压 0.05 V」不是。

        永不抛:回读失败绝不能把一次已经跑完的修针变成异常。"""
        try:
            ctx = self._executor.context
        except Exception:  # noqa: BLE001 — 没有 executor(单测直接调 aggregate)
            return None
        if ctx is None:
            return None
        try:
            from mast.skills.verify import read_junction_state

            return read_junction_state(ctx)
        except Exception:  # noqa: BLE001
            logger.debug("ForgeAuTip: 收尾回读失败", exc_info=True)
            return None

    def _decide_outcome(self, all_good: bool, progress: CompositeProgress,
                        data: dict) -> tuple[bool, str]:
        """只有验收真的通过才算成功。

        预算耗尽是**失败**,而且要说清楚每站发生了什么。把「跑完了预算」讲成成功
        是这条流程最容易犯、也最贵的一个错:下一步会拿一根没修好的针去做实验,
        而报告说它是好的。"""
        summary = str(data.get("summary_cn") or "")
        if not all_good or progress.aborted:
            return False, (progress.aborted_reason or "ForgeAuTip aborted") + " || " + summary
        if str(data.get("outcome")) == "ready":
            return True, ""
        return False, summary


def _summary(outcome: str, sites: list, progress: CompositeProgress,
             *, measured: "dict | None" = None,
             envelope_notes: "list | None" = None,
             scan_speed_notes: "list | None" = None) -> str:
    """给人读的战绩。不达标时,第一句就说不达标。

    ``measured`` 是 :func:`mast.skills.verify.read_junction_state` 的实测回读。
    **这个函数不接受任何计划值当状态用** —— 参数从 ``left_at``(计划)改成
    ``measured``(读数)就是那次修复本身,不是改名。"""
    head = _OUTCOME_CN.get(outcome, f"外环结束({outcome})。")
    bits = [head]
    # 出厂默认被针尖包络改过就说出来 —— 用户看到「打了 20 发没跳变」时,
    # 必须能看见「打的是 3 V 不是表上的 10 V」,否则他会去调一个没在生效的数。
    for note in (envelope_notes or []):
        bits.append(f"⚠️ 参数按当前针尖的安全包络调整过:{note}")
    # 评估图的针尖横向速度 —— **陈述,不是警告**,所以前缀与上面那条不同:
    # 这几条多数描述的是「没被动过」的帧。给没调整过的东西盖一个「调整过」的戳,
    # 就是让用户去调一个没在生效的数(「字段标签会说谎」)。
    if scan_speed_notes:
        bits.append("本次评估图的针尖横向速度(参考:488 nm/s 曾致针尖损伤):"
                    + ";".join(str(n) for n in scan_speed_notes))
    # ``sites or []`` / ``(s or {})`` —— 结案句是**每一条退出路径上最后跑的东西**,
    # 包括那些出错的路径。它自己一抛,用户就什么报告都拿不到 ——
    # 一个只在顺利时才说得出话的总结,恰好在最需要它的时候沉默。
    worked = [s for s in (sites or []) if (s or {}).get("site")]
    if worked:
        bits.append(f"共走了 {len(worked)} 个站点:")
        for s in worked:
            line = (f"站点 #{s['site']}:{_OUTCOME_CN.get(str(s.get('outcome')), s.get('outcome'))}"
                    f"(大修⇄验证 {s.get('rounds', 0)} 轮)")
            acc = s.get("accept") or {}
            if acc.get("reason"):
                line += f";验收:{acc['reason']}"
            # 告警自己的话,不是「外环中止」四个字 —— 后者对用户没有信息量。
            ca = s.get("critical_alert") or {}
            if ca.get("summary_zh"):
                line += (f";⚠️ 电流监控 CRITICAL({ca.get('rule')}):"
                         f"{ca['summary_zh']}")
            bits.append(line)
    else:
        bits.append("一个站点都没走完。")
    if outcome == "critical_alert":
        # **刻意不建议提高预算**。默认那句「要继续请提高预算」在这里是错的建议:
        # 停手的理由不是预算不够,是针尖/信号链可能已经出事了。照着它调大轮数,
        # 等于让流程在一根压进表面的针上多扎几次。
        bits.append("**先查针尖和信号链,不要直接重跑。** 停手的理由不是预算不够 ——"
                    "电流监控判到了物理越界(贴轨 / 冻结 / 巨幅瞬变)。"
                    "确认针尖状态、必要时退针,处置完再决定要不要重跑;"
                    "监控面板上有这条告警的证据图。")
    elif outcome in ("time_budget_exhausted", "spinning",
                     "hard_cap", "round_hard_cap"):
        # 这三种停手同样**刻意不建议提高预算**,各自的理由已经逐条写在
        # ``_OUTCOME_CN`` 里(时间到 = 请人来看;空转 = 没在修,不是没修好;
        # 硬顶 = bug 信号)。分出这一支的唯一目的,是让下面那句默认的
        # 「要继续请提高预算」**不要落到它们头上** —— 对这三种情况,
        # 加大预算都是错的建议,而且是那种听起来很自然的错建议。
        pass
    elif outcome != "ready":
        # 终态由回读报告，预算耗尽文案只说明预算，不能替未执行的动作声明成功。
        measured_anything = any(
            (s or {}).get("phases") for s in (sites or []))
        if measured_anything:
            bits.append("**针尖未达标**,没有做任何「差不多算好了」的让步 —— 要继续请"
                        "提高时间预算(`time_budget_h`,出厂 12 h),或者换样品位置/换针。")
        else:
            bits.append("⚠️ **这一跑没有测到任何东西** —— 没有走完任何一个阶段,"
                        "没有脉冲、没有验证图、没有判据。**所以这里没有关于针尖好坏的"
                        "结论**:它既不是「未达标」,也不是「达标」,是**没测**。"
                        "先看下面 `last_failure` 那一句失败原文,那才是这一跑的真原因。")
    # 仪器现在是什么样:**一次实测回读**,每种退出路径上都跑。
    # 外环不会把工作点改回进场时的成像条件(结构上做不到,见 plan_dynamic 里那段
    # 注释),所以这句话是用户接着做任何事之前唯一的依据 —— 它必须是读数。
    bits.append(_measured_cn(measured))
    return " ".join(bits)


def _fmt_measured(measured: dict) -> list[str]:
    """实测量 → 一串「量:值 / 读不到的原因」。只说读到的那些。"""
    unread = measured.get("unreadable") or {}
    out: list[str] = []
    rows = (
        ("bias_v", "偏压", lambda v: f"{float(v):g} V"),
        ("setpoint_a", "电流设定", lambda v: f"{float(v) * 1e12:g} pA"),
        ("current_a", "实测电流", lambda v: f"{float(v) * 1e12:.3g} pA"),
        ("feedback_on", "Z 反馈", lambda v: "闭合(在调节到设定点)" if v else "**断开**"),
        ("z_controller_status", "Z 控制器模块状态", str),
        ("z_m", "Z 位置", lambda v: f"{float(v) * 1e9:.4g} nm"),
    )
    for key, label, fmt in rows:
        val = measured.get(key)
        if val is None:
            why = unread.get(key)
            # 读不到就说读不到,并且带上**为什么** —— 两种成因(链路报错 / 回包读不懂)
            # 指向完全不同的下一步,而「没有这一项」谁也查不下去。
            out.append(f"{label}:读不到({why})" if why else f"{label}:读不到")
            continue
        try:
            out.append(f"{label}:{fmt(val)}")
        except Exception:  # noqa: BLE001 — 一个格式化不了的值不该吞掉整句话
            out.append(f"{label}:{val!r}")
    return out


def _measured_cn(measured: "dict | None") -> str:
    """收尾那句状态陈述。**没读到就说没读到,绝不拿计划值顶替。**"""
    if not measured:
        return ("⚠️ **仪器现在的状态:没能回读**(收尾回读没跑成)。"
                "这条流程不会把工作点改回进场时的成像条件,所以请在做下一步之前"
                "自己确认一遍偏压 / 电流设定 / Z 反馈。")
    if not measured.get("any_read"):
        # 逐量渲染走的是**同一个** `_fmt_measured` —— 全读不到不是另一种句式。
        # 两套写法意味着「读不到」这件事有两种措辞,而下游(和测试)只会盯住其中
        # 一种;本仓已经栽过一次「判据落在措辞上,措辞一改就失效」。
        return ("⚠️ **仪器现在的状态:一个量都没读到** —— "
                + "；".join(_fmt_measured(measured))
                + "。请在做下一步之前自己确认一遍工作点。")
    # 状态时间必须是这次回读的时间，历史读数不能描述当前仪器。
    at = str(measured.get("read_at_iso") or "")
    head = (f"仪器状态(**截至 {at} 的实测回读**,不是计划值):" if at
            else "仪器状态(收尾实测回读,不是计划值):")
    tail = ""
    fb = measured.get("feedback_on")
    if fb is False:
        # 这正是那次说反了的地方:报文说「停在隧穿态」,硬件反馈是断开的。
        tail = " Z 反馈是断开的 —— **那一刻不是隧穿态**,接着成像前要先把工作点建起来。"
    elif fb is None:
        tail = " Z 反馈状态读不到,所以**不判断**它是不是还在隧穿 —— 请自己确认。"
    # 六个寄存器是依次读的。正常几毫秒,卡住时它们就不属于同一时刻了 —— 说出来,
    # 别把一组不同时的数并排摆成一张一致的快照。
    span = measured.get("span_s")
    if isinstance(span, (int, float)) and span > 1.0:
        tail += (f" ⚠️ 这几个量不是同时读的(跨度 {float(span):.1f} s),"
                 "彼此之间可能不一致。")
    tail += ("(这是那一刻的快照;外环结束后仪器仍可能自行变化 —— 中止后发生过"
             "针尖自动退到限位的情况。要动手前请重读一次。)")
    return head + "；".join(_fmt_measured(measured)) + "。" + tail


class _CriticalWatch:
    """一个**推进式**的水位线:每查一次就把线推到现在。

    存在的理由是覆盖范围,不是复用:原来的写法只在 ``for rnd`` 的循环体第一行查,
    于是(a)第 1 轮窗口宽度 ≈0,(b)**精修 ``poke_phase`` 与验收根本不经过它** ——
    而把针压进表面的恰恰是精修那一段。两条都是 forge-premortem 预演逮到的。

    :meth:`check` 命中时把证据写进 ``site`` 并返回 ``"critical_alert"``;
    没命中就推线并返回 ``""``,所以调用点是一行 ``if (hit := mark.check(...)): return hit``。
    """

    __slots__ = ("_at",)

    def __init__(self, at: float):
        self._at = float(at)

    def check(self, site: dict, where: str, *, round_no: int | None = None) -> str:
        hits = _critical_since(self._at)
        self._at = _time.time()
        if not hits:
            return ""
        # 如实报告原因,并且带上告警自己的话 —— 「外环中止」四个字对用户没有
        # 信息量,「隧道电流持续贴轨饱和…疑似撞针」有。
        top = hits[0]
        site["outcome"] = "critical_alert"
        site["critical_alert"] = {
            "rule": top.get("rule"), "ts": top.get("ts"),
            "summary_zh": top.get("summary_zh"),
            "caught_at": where,
            "round_would_have_been": round_no,
        }
        logger.warning("ForgeAuTip: %s 发现持续型 CRITICAL(%s),外环中止:%s",
                       where, top.get("rule"), top.get("summary_zh"))
        return "critical_alert"


def _halts_tip_work(rule: str) -> bool:
    """决定 CRITICAL 规则是否需要中止修针流程。
    
    修针会产生瞬变，只有持续类异常参与这里的中止决策。
    告警递送与流程中止是独立决策：未知信号仍可递送，但不在此推断为已知持续异常。
    信号名使用 alerts._CRIT_SIGNAL 的映射；硬件保护和 watchdog 不受这里的筛选影响。
    """
    try:
        from mast.core.tip_intent import SUSTAINED_PHYSICAL_SIGNALS
        from mast.monitoring.alerts import _CRIT_SIGNAL

        signal = _CRIT_SIGNAL.get(str(rule or ""))
        return bool(signal) and signal in SUSTAINED_PHYSICAL_SIGNALS
    except Exception:  # noqa: BLE001 — 判不了就别停修针(见上面的极性论证)
        logger.debug("ForgeAuTip: 判不了 %r 属于哪一类(不停手)", rule,
                     exc_info=True)
        return False


def _critical_since(watermark: float) -> list[dict]:
    """水位线之后出现的、**持续类**的 CRITICAL 电流监控告警。永不抛。

    「读不到就当没有」在这里是对的方向,而它在别处常常不是 —— 区别在于这条
    检查是**额外**加的一道网,不是唯一的一道:贴轨/冻结仍然照常落库、照常进面板、
    照常经 ``AlertDeliveryMiddleware`` 送到 agent 眼前。监控库没装(纯离线、
    numpy 缺席)时让整条修针流程拒绝开工,那是拿一个可选子系统去卡死主线。

    过滤在**这一层**做而不是在 SQL 里:``store.critical_alerts_since`` 是通用查询
    (``WHERE level='critical'``),把修针的领域判断塞进存储层会让下一个消费者
    继承一份它没要的语义。见 :func:`_halts_tip_work`。
    """
    try:
        # ``get_store_if_exists`` 而不是 ``get_store``:后者会**创建**库和目录。
        # 一条只读的自检不该有副作用,而且没跑过监控就意味着没有告警 ——
        # None 是诚实的答案。测试里它还避免写进用户真实的 experiments 目录。
        from mast.monitoring.store import get_store_if_exists

        store = get_store_if_exists()
        if store is None:
            return []
        rows = store.critical_alerts_since(watermark, limit=20)
        return [r for r in rows if _halts_tip_work(r.get("rule"))]
    except Exception:  # noqa: BLE001
        logger.debug("ForgeAuTip: 读不到电流监控告警(当作没有)", exc_info=True)
        return []


_OUTCOME_CN = {
    # 成功文案只描述已验证的步骤：正反扫一致性与团簇形貌。
    # 台阶锐度单独报数，是否通过由 sites[].accept 的实际结果说明。
    "ready": ("针尖已达标:正反扫描线重合(基本判据),簇单峰且圆。"
              "台阶边缘锐度只作记录 —— 具体见各站点的 `accept`(可能是「判不了」,"
              "那不是不合格)。"),
    "critical_alert": "电流监控报了 CRITICAL(贴轨/冻结/巨幅瞬变),外环当场停手 ——"
                      "**没有继续把剩下的轮次跑完**。",
    "sites_exhausted": "站点预算用尽,仍未修出合格针尖。",
    # 超时的语义是**叫人来看看**,不是失败。
    # 和 ``critical_alert`` 一样**刻意不建议提高预算**:12 小时没修好,
    # 再给 12 小时大概率也不行,该看的是「为什么修不好」。
    "time_budget_exhausted": ("跑满时间预算仍未达标 —— **请人来看看是怎么回事**。"
                              "不建议直接加大预算重跑:12 小时修不好,多半不是"
                              "时间不够。先看下面各站的战绩和最后一次实测回读。"),
    "spinning": ("连续几站**什么都没测到**就停手了 —— 这不是「没修好」,是「没在修」。"
                 "典型成因是与站点无关的故障(读不到线数据 / 找不到干净落点 / "
                 "链路故障),换位置对它没有帮助。**这里没有关于针尖好坏的结论。**"),
    "hard_cap": ("撞到失控保险(站点数硬顶)—— 正常流程不该走到这里,"
                 "这本身是一个 bug 的信号。请把这一跑的产物留给开发看。"),
    # ⚠️ 措辞不许提针尖。它和 verify_exhausted 的区别正是「判据说话」和
    # 「保险说话」的区别 —— 后者没有任何判据下过结论。
    "round_hard_cap": ("撞到失控保险(单站轮数硬顶)—— **这不是关于针尖的结论**,"
                       "没有任何判据说过针尖不好。正常流程不该走到这里,"
                       "请把这一跑的产物留给开发看。"),
    "coarse_budget_exhausted": "没有可去的新站点了(粗动行程预算/站点间距挡住了)。",
    "relocate_failed": "粗动换位失败,外环停在这里。",
    "stopped_early": "扫描被停下(用户按了 Stop / Nanonis 自停 / 安全停机)——"
                     "这是事实不是判断,外环照此中止。",
    "aborted": "外环被中止。",
    "incomplete": "外环没走完。",
    "surface_spent": "这片表面没有干净落点了",
    "verify_exhausted": "反复大修后正反扫描线仍不重合",
    # 2026-08-10:不再写死「读不到」——「判不了」现在有两种,而且指向不同的下一步:
    # ①读不到线数据(查通信);②这块地方太平、起伏低于下限(**换个有形貌的地方重量**)。
    # ②自起伏弃权门上线后是常走的路。具体哪一种在 verify 相的 ``reason`` 里。
    "verify_inconclusive": "判不了正反扫一致性(读不到数据,或这块地方太平)—— "
                           "这不等于针尖没问题,也不等于针尖有问题",
    "refine_incomplete": "精修没能把簇做到达标",
    "sharpness_not_met": "台阶锐度未达验收线",
    "relocated": "已换位",
    # 2026-08-18 补:这一条**代码里一直会赋,而这张表里没有** ⇒ 报告(以及站点
    # 战绩那一行)一直在印一个英文 slug `pulse_rescue_exhausted`。
    # 由新加的那道闸门逮到:
    # ``tests/v2/unit/chat/test_forge_outcomes_are_all_narratable.py``
    # —— 它不比对名单,它去源码里数 ``outcome = "..."``,所以两张表**一起**漏掉
    # 的那一个也躲不过去。
    "pulse_rescue_exhausted": ("扎针扎不圆、退回去打脉冲救的次数也用完了 ——"
                               "这一站不再救,换区。"),
}


def make_tools(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill

    return [wrap_skill(ForgeAuTip, context_provider)]


__all__ = ["ForgeAuTip"]
