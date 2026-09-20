"""贵金属表面上的针尖修整 —— 三个入口,共用一套阶段生成器。

* ``PulseConditionTip`` —— 只跑电脉冲大修(阶段 A)。
* ``PokeConditionTip``  —— 只跑扎针尖精修(阶段 D)。
* ``PrepareNobleTip``   —— 从头修到「好针尖基础态」(A → B → C → D → 验收)。

三个都是 CONFIRM:agent 路上一次批准就整条跑完(composite 的子步骤不重新过 CONFIRM
门控),这正是用户要的粒度 —— 一次修针要打十几发脉冲、扎十几次,每次都弹窗没法
做实验。**真正危险的动作各有各的闸**:脉冲幅度过针尖方案表的安全包络(超上限拒绝
不夹紧),下压类动作在 qPlus 传感器上默认拒绝,SAFE 模式整类硬拦,SEMI 模式每发脉
冲仍单独确认。

判据与循环都在 :mod:`mast.skills.composite._tip_phases`;这里只负责暴露参数、拼阶
段、把结果讲清楚。

与 ``ConditionTip`` 并存而不是取代它:那个用 FFT 质量判据做扫描中途的应急快修,是
针尖 CRITICAL 事件的处置手段;这里是要求的、从头到尾的修整流程。
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

from mast.core.si_quantity import format_si
from mast.core.noble_tip_workflow import (
    NOBLE_METAL_BASELINE,
    descend_sequence,
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
from mast.skills.composite._preflight import PROBE_TIP_SHAPER, preflight_modules
from mast.skills.composite._tip_phases import (
    level_phase,
    poke_phase,
    pulse_phase,
    scan_at_params,
    verify_phase,
)
from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor

logger = logging.getLogger(__name__)

_WF = NOBLE_METAL_BASELINE

#: 流程参数在 ParameterSpec 里的说明统一挂这句 —— 与 ScanAt 同款纪律:不填是
#: 默认路径,填了就会被当成用户指定并展示给他。
_FROM_TABLE = ("留空即按贵金属修针流程表取值；只有用户逐字说过的数才填进来。")


def _wf_from(params: dict):
    """将显式技能参数叠加到默认流程表，再与当前针尖包络核对。

    与 ForgeAuTip 共用核对函数。仅调整未显式指定的默认值；调用方
    提供的参数仍须通过包络检查，越界时拒绝而不夹紧。
    """
    wf = resolve({k: params.get(k) for k in _PARAM_KEYS})
    wf, _notes = reconcile_with_tip_envelope(
        wf, specified={k for k in _PARAM_KEYS if params.get(k) is not None})
    return wf


#: ``_wf_from`` 从 params 里读的键。与 ForgeAuTip 的 ``_WF_PARAM_KEYS`` 同一个用途
#: (⑮:读了就必须声明,否则模型传了会被静默丢弃)。
_PARAM_KEYS: tuple[str, ...] = (
    "pulse_v", "pulse_width_s", "pulse_success_dz_nm", "pulses_per_polarity",
    "pulse_same_spot_budget",
    "pulse_budget", "junction_bias_v", "junction_setpoint_a",
    "verify_bias_v", "verify_setpoint_a", "verify_scan_nm",
    "fwdbwd_threshold", "max_verify_rounds", "step_scan_nm",
    "flat_region_nm", "poke_depth_nm", "poke_dwell_s", "cluster_scan_nm",
    "min_axis_ratio", "critical_start_pm", "critical_step_pm",
    "critical_repeat_n", "poke_budget",
    # ⚠️ ``approach_bias_v`` 曾在这张单子里,而**全仓没有任何一处读
    # ``wf.approach_bias_v``**,PrepareNobleTip 也从没为它声明 ParameterSpec ——
    # 读了、没声明、也没人用,缺陷⑮ 的形状叠上一条死配置。2026-08-10 由
    # ``test_forge_scan_working_point`` 的结构闸门查出来。
    # **从读取单子里摘掉**(留在流程表里,由那道闸门的显式豁免名单管着):
    # 「进针前该不该设 4 V 偏压、设在哪一步」是用户的科学判断,不是顺手接一下
    # 就算数的事。接上它的那天,把这一行加回来 + 同时声明 ParameterSpec。
)


def _pulse_params() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="pulse_v", type="float", unit="V",
                      description=f"脉冲电压（幅值，极性由流程自己翻）。{_FROM_TABLE}"
                                  f"表值 {_WF.pulse_v} V。会过当前针尖的安全包络。",
                      required=False, min_value=-10.0, max_value=10.0),
        ParameterSpec(name="pulse_width_s", type="float", unit="s",
                      description=f"脉冲时长。{_FROM_TABLE}表值 {_WF.pulse_width_s} s。",
                      required=False, min_value=1e-3, max_value=2.0),
        ParameterSpec(name="pulse_success_dz_nm", type="float", unit="nm",
                      description=f"判定「这一发有效」的 Z 向上跳变量。{_FROM_TABLE}"
                                  f"表值 {_WF.pulse_success_dz_nm} nm。",
                      required=False, min_value=0.1, max_value=1000.0),
        ParameterSpec(name="pulses_per_polarity", type="int",
                      description=f"同一极性连打几发无效就翻极性。{_FROM_TABLE}"
                                  f"表值 {_WF.pulses_per_polarity}。",
                      required=False, min_value=1, max_value=50),
        ParameterSpec(name="pulse_same_spot_budget", type="int",
                      description=(
                          "**没打动针尖**时,同一个落点上最多连打几发。打成功的"
                          "那一发永不豁免(成功即换地方)。"
                          f"{_FROM_TABLE}表值 {_WF.pulse_same_spot_budget}。"),
                      required=False, min_value=1, max_value=50),
        ParameterSpec(name="pulse_budget", type="int",
                      description=f"这一阶段最多打几发（行动预算，不是重试次数）。"
                                  f"{_FROM_TABLE}表值 {_WF.pulse_budget}。",
                      required=False, min_value=1, max_value=200),
        ParameterSpec(name="junction_bias_v", type="float", unit="V",
                      description=f"修针期间的结偏压。{_FROM_TABLE}表值 {_WF.junction_bias_v} V。",
                      required=False, min_value=-10.0, max_value=10.0),
        ParameterSpec(name="junction_setpoint_a", type="float", unit="A",
                      description=f"修针期间的电流设定。{_FROM_TABLE}表值 {format_si(_WF.junction_setpoint_a)}A。",
                      required=False, min_value=1e-12, max_value=1e-7),
    ]


def _poke_params() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="poke_depth_nm", type="float", unit="nm",
                      description=f"深扎期的首扎深度。{_FROM_TABLE}表值 {_WF.poke_depth_nm} nm。"
                                  f"会过当前针尖的安全包络。",
                      required=False, min_value=0.01, max_value=100.0),
        ParameterSpec(name="poke_dwell_s", type="float", unit="s",
                      description=f"断反馈下压后的驻留时间。{_FROM_TABLE}表值 {_WF.poke_dwell_s} s。",
                      required=False, min_value=0.0, max_value=10.0),
        ParameterSpec(name="cluster_scan_nm", type="float", unit="nm",
                      description=f"看簇的小图视野。{_FROM_TABLE}表值 {_WF.cluster_scan_nm} nm。",
                      required=False, min_value=0.5, max_value=500.0),
        ParameterSpec(name="min_axis_ratio", type="float",
                      description=(f"簇的圆度达标线 = **等效轴比**"
                                   f"(0.75 =「不比长短轴差 25% 的椭圆更不规则」)。"
                                   f"{_FROM_TABLE}表值 {_WF.min_axis_ratio}。"
                                   f"⚠️ 2026-08-11 取代 round_threshold=0.65,"
                                   f"**两个数不可换算**。"),
                      required=False, min_value=0.0, max_value=1.0),
        ParameterSpec(name="critical_start_pm", type="float", unit="pm",
                      description=f"临界浅扎的起步深度。{_FROM_TABLE}表值 {_WF.critical_start_pm} pm。",
                      required=False, min_value=1.0, max_value=10000.0),
        ParameterSpec(name="critical_step_pm", type="float", unit="pm",
                      description=f"临界浅扎每级加深多少。{_FROM_TABLE}表值 {_WF.critical_step_pm} pm。",
                      required=False, min_value=1.0, max_value=5000.0),
        ParameterSpec(name="critical_repeat_n", type="int",
                      description=f"在临界深度反复扎几次。{_FROM_TABLE}表值 {_WF.critical_repeat_n}。",
                      required=False, min_value=1, max_value=50),
        ParameterSpec(name="poke_budget", type="int",
                      description=f"这一阶段最多扎几次。{_FROM_TABLE}表值 {_WF.poke_budget}。",
                      required=False, min_value=1, max_value=200),
    ]


def _verify_params() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="verify_bias_v", type="float", unit="V",
                      description=f"验证扫图的偏压。{_FROM_TABLE}表值 {_WF.verify_bias_v} V。",
                      required=False, min_value=-10.0, max_value=10.0),
        ParameterSpec(name="verify_setpoint_a", type="float", unit="A",
                      description=f"验证扫图的电流。{_FROM_TABLE}表值 {format_si(_WF.verify_setpoint_a)}A。",
                      required=False, min_value=1e-12, max_value=1e-7),
        ParameterSpec(name="verify_scan_nm", type="float", unit="nm",
                      description=f"验证扫图的视野。{_FROM_TABLE}表值 {_WF.verify_scan_nm} nm。",
                      required=False, min_value=1.0, max_value=5000.0),
        ParameterSpec(name="fwdbwd_threshold", type="float",
                      description=f"正反扫描线相似度下限。{_FROM_TABLE}表值 {_WF.fwdbwd_threshold}。",
                      required=False, min_value=0.0, max_value=1.0),
        ParameterSpec(name="max_verify_rounds", type="int",
                      description=f"验证↔大修最多来回几次。{_FROM_TABLE}表值 {_WF.max_verify_rounds}。",
                      required=False, min_value=1, max_value=20),
    ]


class _TipComposite(CompositeSkillGraph):
    """三个入口共用的收尾:结果里既有结论也有过程。"""

    def aggregate(self, sub_results: dict, progress) -> dict:
        data = dict(progress.partial_data)
        # 阶段结果是给人读的主体;partial_data 里那些 "A:log" 之类是给续跑用的
        # 内部状态,留着但不要盖过结论。
        # ``best_frame`` 在这里：一次「跑满轮数」的结论若把它漏掉，调用方就只能
        # 看到「没成」，而报告里明明躺着一张 passed 的确认帧 ——
        # **停下来的理由**与**有没有拿到东西**是两件事。
        for key in ("phases", "outcome", "summary_cn", "best_frame"):
            if key in getattr(self, "_phase_out", {}):
                data[key] = self._phase_out[key]
        return data

    _phase_out: dict[str, Any] = {}


class PulseConditionTip(_TipComposite):
    """只跑电脉冲大修:打到 Z 向上跳出几十 nm 为止。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PulseConditionTip",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            capabilities=frozenset({"bias_pulse"}),
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "贵金属表面上的偏压脉冲修针：打一发，盯着 Z 在打之前与打之后"
                "两个稳定值之间跳了多少，换一块新鲜表面，重复到 Z 向上跳出"
                "几十 nm 为止。连着打了几发都不见效之后会翻转极性。"
                "一次批准覆盖整轮；每支针尖各自的电压包络照样生效。"
            ),
            parameters=_pulse_params(),
            estimated_duration_s=180.0,
            composition_level=2,
            tags=["tip", "pulse", "conditioning", "composite"],
        )

    def plan_dynamic(self, params: dict, executor: GraphExecutor
                     ) -> Iterator[CompositeStep]:
        wf = _wf_from(params)
        out = yield from pulse_phase(executor, wf, prefix="A")
        self._phase_out = {
            "phases": [out],
            "outcome": "satisfied" if out.get("satisfied") else "not_satisfied",
            "summary_cn": _pulse_summary(out),
        }
        executor.set_partial("pulse_result", out)


class PokeConditionTip(_TipComposite):
    """只跑扎针尖精修:先深后浅,把簇做成单峰、圆、小。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PokeConditionTip",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            capabilities=frozenset({"tip_shaping"}),
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "平整贵金属区上的扎针精修：先深扎一针（定形），再把深度从 "
                "100 pm 起一档档往上加，直到 Z 刚好开始跳，然后在那个阈值上"
                "重复（定大小）。每咬一口就扫一次簇，查峰的个数与圆度。"
                "在 qPlus 上扎针是常规操作（守卫默认放行）；护住音叉的是深度包络与扎针前降到 20 mV，不是许可开关。"
            ),
            parameters=_poke_params() + [
                ParameterSpec(
                    name="allow_on_qplus", type="bool",
                    description=("默认关 —— 在 nm 尺度上扎针**本来就是**常规操作，不需要许可。"
                                "这个开关只有在守卫被重新打开时（MAST_QPLUS_POKE_GUARD=1）"
                                "才有意义。真正护住音叉的是深度包络（超上限一律**拒绝**，"
                                "绝不夹紧），以及扎针之前自动把偏压降到 20 mV —— "
                                "不是这个开关。"),
                    required=False, default=False),
            ],
            estimated_duration_s=600.0,
            composition_level=2,
            tags=["tip", "shaper", "conditioning", "cluster", "composite"],
        )

    def validate_params(self, params: dict) -> list[str]:
        """标准校验 + 下压深度的针尖包络（超上限拒绝，不夹紧）。"""
        errors = super().validate_params(params)
        depth_nm = params.get("poke_depth_nm")
        if depth_nm is None:
            depth_nm = _WF.poke_depth_nm
        from mast.skills.builtins._tip_policy import apply_tip_policy
        _, plan = apply_tip_policy(
            {"poke_deep_depth_m": -abs(float(depth_nm)) * 1e-9},
            ("poke_deep_depth_m",))
        if plan is not None and not plan.ok:
            errors.extend(plan.refusals)
        return errors

    def plan_dynamic(self, params: dict, executor: GraphExecutor
                     ) -> Iterator[CompositeStep]:
        stop = _tip_shaper_preflight(executor)
        if stop:
            return
        if _qplus_blocked(executor, "PokeConditionTip", params):
            return
        wf = _wf_from(params)
        out = yield from poke_phase(executor, wf, prefix="D")
        self._phase_out = {
            "phases": [out],
            "outcome": "refined" if out.get("refined") else "not_refined",
            "summary_cn": _poke_summary(out),
        }
        executor.set_partial("poke_result", out)


class PrepareNobleTip(_TipComposite):
    """把针尖修到「好针尖基础态」——大修 → 验证 → 调平 → 精修 → 验收。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="PrepareNobleTip",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            capabilities=frozenset({"bias_pulse", "tip_shaping"}),
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "Au/Ag/Cu（单晶或薄膜）上的完整修针流程：偏压脉冲一直打到 Z "
                "向上跳出几十 nm，验证正扫与反扫的扫描线是否重合，找台阶并调平"
                "一小块平区，然后从深扎一路扎到阈值附近的浅扎，直到那个簇变成"
                "单个圆凸起。最后以一次台阶边锐度检查收尾。"
                "**只在**用户要求整针、或者计划里明确要求时才用 —— "
                "这是一套又长又慎重的规程，不是「针尖好像还能更好一点」"
                "就随手跑的东西。"
            ),
            parameters=(_pulse_params() + _verify_params() + _poke_params() + [
                ParameterSpec(
                    name="step_scan_nm", type="float", unit="nm",
                    description=f"找台阶的视野。{_FROM_TABLE}表值 {_WF.step_scan_nm} nm。",
                    required=False, min_value=1.0, max_value=5000.0),
                ParameterSpec(
                    name="flat_region_nm", type="float", unit="nm",
                    description=f"调平用的平区大小。{_FROM_TABLE}表值 {_WF.flat_region_nm} nm。",
                    required=False, min_value=1.0, max_value=1000.0),
                ParameterSpec(
                    name="allow_on_qplus", type="bool",
                    description=("默认关 —— 在 nm 尺度上扎针**本来就是**常规操作，不需要许可。"
                                "这个开关只有在守卫被重新打开时（MAST_QPLUS_POKE_GUARD=1）"
                                "才有意义。真正护住音叉的是深度包络（超上限一律**拒绝**，"
                                "绝不夹紧），以及扎针之前自动把偏压降到 20 mV —— "
                                "不是这个开关。"),
                    required=False, default=False),
                ParameterSpec(
                    name="skip_refine", type="bool",
                    description="调平之后就停 —— 只做脉冲 + 验证，不扎针。"
                                "当这片表面不能再多出簇的时候用它。",
                    required=False, default=False),
            ]),
            estimated_duration_s=1800.0,
            composition_level=3,
            tags=["tip", "conditioning", "pulse", "shaper", "composite"],
        )

    def plan_dynamic(self, params: dict, executor: GraphExecutor
                     ) -> Iterator[CompositeStep]:
        wf = _wf_from(params)
        skip_refine = bool(params.get("skip_refine"))
        phases: list[dict] = []
        outcome = "incomplete"

        stop = _tip_shaper_preflight(executor, required=not skip_refine)
        if stop:
            return

        # ── A ⇄ B:大修与验证来回,直到正反扫描线重合 ──────────────────
        verify: dict[str, Any] = {}
        descend = descend_sequence(wf)
        for rnd in range(1, int(wf.max_verify_rounds) + 1):
            # 第 2 轮起用递减脉冲过渡(7/5/3 V);qPlus 上 descend 是空的,继续用
            # 原电压 —— 音叉上那几伏可能把叉臂也一起修了。
            seq = descend if (rnd > 1 and descend) else None
            pulse = yield from pulse_phase(
                executor, wf, prefix=f"A{rnd}", voltage_sequence=seq)
            phases.append(pulse)
            if pulse.get("surface_spent"):
                outcome = "surface_spent"
                break
            verify = yield from verify_phase(executor, wf, prefix=f"B{rnd}")
            phases.append(verify)
            if verify.get("surface_spent"):
                # 2026-08-18:verify 找不到干净落点时现在也归 ``surface_spent``
                # (与 ``pulse_phase`` 那一支同名同义)。它**必须排在
                # ``inconclusive`` 前面** —— verify 会把两个位一起置上,而
                # 「表面用完了」才说得清下一步(换区),「判不了」只说得清别打脉冲。
                outcome = "surface_spent"
                break
            if verify.get("inconclusive"):
                outcome = "verify_inconclusive"
                break
            if verify.get("passed"):
                outcome = "verified"
                break
        else:
            outcome = "verify_exhausted"

        if outcome != "verified":
            self._phase_out = _wrap(phases, outcome, wf)
            return

        # ── C:找台阶 + 调平 ────────────────────────────────────────────
        level = yield from level_phase(executor, wf, prefix="C")
        phases.append(level)
        step_center = level.get("wide_scan_center")

        if skip_refine:
            self._phase_out = _wrap(phases, "leveled_no_refine", wf)
            return

        # ── D:精修 ─────────────────────────────────────────────────────
        if _qplus_blocked(executor, "PrepareNobleTip", params):
            return
        poke = yield from poke_phase(executor, wf, prefix="D")
        phases.append(poke)

        # ── 验收:回到有台阶的地方看边缘够不够陡 ────────────────────────
        accept: dict[str, Any] = {"phase": "accept"}
        if step_center:
            # 走与 forge 同一个组装点。这条路径上 ``step_pixels`` /
            # ``step_line_time_s`` 出厂是 None ⇒ 两个键都不下发 ⇒ 分辨率与每线
            # 时间仍由用户的档位表决定,行为与从前逐字节相同。
            yield CompositeStep(
                step_id="E:scan", skill_name="ScanAt",
                params=scan_at_params(wf, step_center[0], step_center[1],
                                      size_nm=wf.step_scan_nm,
                                      pixels=wf.step_pixels,
                                      line_time_s=wf.step_line_time_s,
                                      # 同 ForgeAuTip._accept:回到 level 相找到的
                                      # 台阶处,验收判的就是那条边缘。
                                      origin="analysis"),
                optional=True, checkpoint_after=False, tags=("accept",))
            yield CompositeStep(
                step_id="E:save", skill_name="SaveScan", params={},
                optional=True, checkpoint_after=False, tags=("accept",))
            yield CompositeStep(
                step_id="E:latest", skill_name="GetLatestScanFile", params={},
                optional=True, checkpoint_after=False, tags=("accept",))
            path = ""
            for sid in ("E:save", "E:latest"):
                res = executor.sub_results.get(sid)
                d = dict(getattr(res, "data", None) or {}) if res else {}
                path = path or str(d.get("path") or d.get("file_path") or "")
            accept["scan_path"] = path
            if path:
                yield CompositeStep(
                    step_id="E:sharpness", skill_name="AssessTipSharpness",
                    params={"scan_path": path},
                    optional=True, checkpoint_after=True, tags=("accept",))
                res = executor.sub_results.get("E:sharpness")
                qd = dict(getattr(res, "data", None) or {}) if res else {}
                accept.update({k: qd.get(k) for k in
                               ("edge_resolution_nm", "verdict", "has_step",
                                "fwd_bwd_instability") if k in qd})
        phases.append(accept)

        good = bool(poke.get("refined"))
        self._phase_out = _wrap(phases, "ready" if good else "refine_incomplete", wf)


# ── 共用小工具 ──────────────────────────────────────────────────────────────

def _tip_shaper_preflight(executor: GraphExecutor, *, required: bool = True) -> bool:
    """Tip Shaper 模块没在 Nanonis 里跑起来就早退 —— 别跑了五步才发现。"""
    if not required:
        return False
    ctx = getattr(executor, "_context", None)
    if ctx is None:
        return False
    msg = preflight_modules(ctx, [PROBE_TIP_SHAPER])
    if msg:
        executor.progress.aborted = True
        executor.progress.aborted_reason = msg
        return True
    return False


def _qplus_blocked(executor: GraphExecutor, skill_name: str, params: dict) -> bool:
    """qPlus 传感器上的下压类操作默认拒绝(音叉损坏不可逆,要拆机重装)。"""
    from mast.skills.builtins._tip_policy import qplus_gate
    gate = qplus_gate(skill_name, params)
    if gate is None:
        return False
    executor.progress.aborted = True
    executor.progress.aborted_reason = gate.error
    return True


def _pulse_summary(out: dict) -> str:
    if out.get("satisfied"):
        return (f"打了 {out.get('fired')} 发，最后一发 Z 向上跳 "
                f"{float(out.get('last_dz_nm') or 0.0):.1f} nm —— 大修达标。")
    return f"打了 {out.get('fired')} 发未达标：{out.get('reason', '')}"


def _poke_summary(out: dict) -> str:
    cl = out.get("cluster") or {}
    if out.get("refined"):
        return (f"扎了 {out.get('pokes')} 次，临界深度 "
                f"{float(out.get('critical_depth_nm') or 0.0) * 1000:.0f} pm，"
                f"簇等效轴比 {cl.get('axis_ratio')} —— 精修达标。")
    return f"扎了 {out.get('pokes')} 次未达标：{out.get('reason', '')}"


_OUTCOME_CN = {
    "ready": "针尖已修到基础态（正反扫描线重合、簇单峰且圆）。",
    "verified": "正反扫描线已重合。",
    "verify_exhausted": "反复大修后正反扫描线仍不重合 —— 建议换针或换区。",
    "verify_inconclusive": "读不到正反扫描线数据，判不了针尖好坏（这不等于针尖没问题）。",
    "surface_spent": "这片表面已经没有干净的地方了 —— 需要粗动换区或换样品位置。",
    "leveled_no_refine": "已完成大修、验证与调平（按要求跳过了扎针精修）。",
    "refine_incomplete": "大修与调平完成，但精修没能把簇做到达标。",
    "not_satisfied": "未达标。",
    "not_refined": "精修未达标。",
    "incomplete": "流程未走完。",
}


def _wrap(phases: list[dict], outcome: str, wf) -> dict:
    bits = [_OUTCOME_CN.get(outcome, outcome)]
    for p in phases:
        if p.get("phase") == "pulse":
            bits.append(_pulse_summary(p))
        elif p.get("phase") == "verify" and p.get("reason"):
            bits.append(str(p["reason"]))
        elif p.get("phase") == "poke":
            bits.append(_poke_summary(p))
    # 读不到地图时,「没记录」不等于「表面干净」—— 这句必须出现在报告里。
    if any(p.get("map_known") is False for p in phases):
        bits.append("⚠ 期间读不到实验记录，无法确认落点是否干净。")
    return {"phases": phases, "outcome": outcome, "summary_cn": " ".join(bits)}


def make_tools(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return [wrap_skill(cls, context_provider)
            for cls in (PulseConditionTip, PokeConditionTip, PrepareNobleTip)]


__all__ = ["PulseConditionTip", "PokeConditionTip", "PrepareNobleTip"]
