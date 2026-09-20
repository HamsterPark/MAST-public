"""从「好针尖基础态」再往前走 —— 两种特异化针尖。

* ``MakeSpectroscopyTip``      —— 做 STS 的**金属性**针尖(扎 → 测谱 → 看表面态)。
* ``MakeAtomicResolutionTip``  —— **原子分辨**针尖(快扫 + 偏压扰动 → 看原子相)。

两个都是 CONFIRM:agent 路上一次批准整条跑完 —— 一次锻针尖要扎十几次、测好几条
谱,每步弹窗没法做实验。**真正危险的动作各有各的闸**:下压深度过针尖方案表的安全
包络(超上限拒绝不夹紧),qPlus 上默认拒绝,SAFE 模式整类硬拦,偏压扰动另有自己的
三道硬帽(``bias_wiggle``)。

## 不重写的部分

「把针尖扎好」那一整套(D1 深扎修形状 → D2 从 100 pm 起逐级加到 Z 恰好跳变 → 在
临界深度反复扎)已经是 :func:`mast.skills.composite._tip_phases.poke_phase`。两个
配方都 ``yield from`` 它,只是各自换一档参数(见 ``core.special_tip_workflow``)。
判据只有一份,阈值不会两边漂 —— 这是 ``_tip_phases`` 模块注释里那条「共用生成器
而不是嵌套 composite」的直接延续。

## 两条新判据

* 肖克利表面态台阶 —— :func:`mast.vision.spectroscopy.assess_shockley_onset`
* 原子相 —— :func:`mast.vision.atomic_phase.assess_atomic_phase`

都是纯函数、阈值全参数化,合成数据测得动(白噪声误报率、准周期抖动对照组见
``tests/v2/unit/vision/``)。技能外壳在 ``builtins.tip_spectro_assess``。

## 「没锻成」是 success=True

与 ``PrepareNobleTip`` 同一条:``outcome`` 与 ``SkillResult.success`` 是两件事。
计划跑完了就是 success,针尖够不够好写在 ``outcome`` 里。把「还不够好」表达成技能
失败,会让上层把一次正常的、有结论的尝试当成故障去重试。
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

from mast.core.si_quantity import format_si
from mast.core.special_tip_workflow import (
    ATOMIC_TIP,
    SPECTROSCOPY_TIP,
    resolve_atomic,
    resolve_spectroscopy,
)
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
)
from mast.skills.composite._preflight import PROBE_TIP_SHAPER, preflight_modules
# 同包内的共享件。下划线是「形式上私有」——判据与相位只有这一份，复制一份的下场
# 是两边阈值各自漂移（本仓已有 precondition 副本漂移的先例）。
from mast.skills.composite._tip_phases import (
    _data,
    flat_poke_sites,
    _dirty,
    _ok,
    _relocate,
    _scan_path,
    poke_phase,
)
from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor
from mast.skills.composite.prepare_noble_tip import _TipComposite
from mast.vision.atomic_lines import ADVISORY_STREAK

logger = logging.getLogger(__name__)

_FROM_TABLE = "留空即按流程表取值；只有用户逐字说过的数才填进来。"

_OUTCOME_CN = {
    # 配方 1
    "sts_tip_ready": "针尖已验证为金属性（肖克利表面态在该在的位置）。",
    "sts_rounds_exhausted": ("扎针与测谱来回做满预算，表面态仍未出现在正确位置 —— "
                             "建议回到完整修针流程（PrepareNobleTip），或换针。"),
    "sts_no_substrate": "不知道台面上是什么衬底，定不出表面态的期望位置。",
    "sts_no_surface_state": "这个衬底没有肖克利表面态，换个判据或换块衬底。",
    "sts_spectrum_unavailable": "谱采集完成但拿不到数据文件，判据无从做起。",
    # 配方 2
    "atomic_tip_ready": "针尖已达原子分辨。",
    "atomic_cycles_exhausted": ("扰动与回退扎针做满预算仍未出现原子相 —— "
                                "建议回到完整修针流程，或换一块更平整的区域。"),
    "atomic_time_budget_done": ("时间预算用完，交出这段时间里**最好的那一张**图 —— "
                                "产物是图不是针尖，图会累积。"),
    "atomic_time_budget_exhausted": ("时间预算用完，一张都没拿到。"
                                     "建议加预算重来，或先回完整修针流程。"),
    "atomic_no_flat_area": "找不到够大的无台阶平区来做原子分辨扫描。",
    "sts_split_tip": ("大图上台阶出现重影（多针尖）—— 找不到台面的原因在**针尖**不在表面，"
                      "换位置无效。这一路要打脉冲，不是接着浅扎。"),
    "sts_no_clean_terrace": ("找不到干净台面来测谱 —— 这是**表面**的问题不是针尖的，"
                             "流程主动停手：继续扎针只会把一根可能已经合格的针尖"
                             "扎废。建议换区或换样品位置后重来。"),
    "atomic_scale_misconfigured": "评估帧的像素尺度判不出原子相（见 reason）。",
    # 共用
    "surface_spent": "这片表面已经没有干净的地方了 —— 需要粗动换区或换样品位置。",
    "poke_failed": "扎针精修没能完成。",
    "incomplete": "流程未走完。",
}



def _budget_left(t_start: float, budget_s: float) -> float:
    """还剩多少秒。``budget_s <= 0`` = 不设预算（只受 max_cycles 约束）。"""
    import time as _t
    if budget_s <= 0:
        return float("inf")
    return budget_s - (_t.monotonic() - t_start)


def _wrap(phases: list[dict], outcome: str, extra: str = "") -> dict:
    bits = [_OUTCOME_CN.get(outcome, outcome)]
    if extra:
        bits.append(extra)
    for p in phases:
        reason = p.get("reason")
        if reason:
            bits.append(str(reason))
    # 读不到地图时,「没记录」不等于「表面干净」—— 这句必须出现在报告里
    # (与 PrepareNobleTip 同一条纪律)。
    if any(p.get("map_known") is False for p in phases):
        bits.append("⚠ 期间读不到实验记录，无法确认落点是否干净。")
    return {"phases": phases, "outcome": outcome, "summary_cn": " ".join(bits)}


def _qplus_blocked(executor: GraphExecutor, skill_name: str, params: dict) -> bool:
    """qPlus 传感器上的下压类操作默认拒绝(音叉损坏不可逆,要拆机重装)。"""
    from mast.skills.builtins._tip_policy import qplus_gate

    gate = qplus_gate(skill_name, params)
    if gate is None:
        return False
    executor.progress.aborted = True
    executor.progress.aborted_reason = gate.error
    return True


def _preflight(executor: GraphExecutor) -> bool:
    ctx = getattr(executor, "_context", None)
    if ctx is None:
        return False
    msg = preflight_modules(ctx, [PROBE_TIP_SHAPER])
    if msg:
        executor.progress.aborted = True
        executor.progress.aborted_reason = msg
        return True
    return False


# ── 配方 1:做 STS 的金属性针尖 ──────────────────────────────────────────────

class MakeSpectroscopyTip(_TipComposite):
    """浅扎出小而圆的簇 → 开 lock-in 测谱 → 看肖克利表面态,不达标就再来。"""

    def metadata(self) -> SkillMetadata:
        wf = SPECTROSCOPY_TIP
        return SkillMetadata(
            name="MakeSpectroscopyTip",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            capabilities=frozenset({"tip_shaping"}),
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把一根能用的针尖变成适合做 STS 的**金属性**针尖:**浅**扎出一个"
                "小而圆的簇,然后打开 lock-in 采一条 dI/dV 谱,检查贵金属 (111) 的"
                "肖克利表面态有没有出现在该在的能量上。不达标就反复「扎针—测谱」。"
                "金属性针尖**不是**最尖的针尖 —— 这正是它比 PrepareNobleTip 扎得"
                "更浅的原因。需要一个有表面态的衬底(Au/Ag/Cu(111));"
                "在 qPlus 传感器上默认拒绝,除非显式允许。"
            ),
            parameters=[
                ParameterSpec(
                    name="substrate", type="str",
                    description=("衬底,例如 'Au(111)'。"
                                 "留空则从已登记的样品里推断。"),
                    required=False, default=""),
                ParameterSpec(
                    name="expected_onset_v", type="float", unit="V",
                    description=("预期的肖克利 onset 位置。"
                                 "留空则取自衬底。"),
                    required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="onset_tol_v", type="float", unit="V",
                    description=f"onset 位置容差。{_FROM_TABLE}表值 {wf.onset_tol_v} V。",
                    required=False, min_value=0.001, max_value=0.5),
                ParameterSpec(
                    name="poke_depth_nm", type="float", unit="nm",
                    description=(f"浅扎深度。{_FROM_TABLE}表值 {wf.poke_depth_nm} nm。"
                                 f"会过当前针尖的安全包络。"),
                    required=False, min_value=0.01, max_value=100.0),
                ParameterSpec(
                    name="critical_repeat_n", type="int",
                    description=f"临界深度上反复扎几次。{_FROM_TABLE}表值 {wf.critical_repeat_n}。",
                    required=False, min_value=1, max_value=50),
                ParameterSpec(
                    name="mod_amp_v", type="float", unit="V",
                    description=(f"lock-in 调制幅度。{_FROM_TABLE}表值 {wf.mod_amp_v} V。"
                                 f"注意：rms 还是峰值取决于本机 lock-in 设置。"),
                    required=False, min_value=1e-5, max_value=1.0),
                ParameterSpec(
                    name="mod_freq_hz", type="float", unit="Hz",
                    description=f"lock-in 调制频率。{_FROM_TABLE}表值 {wf.mod_freq_hz} Hz。",
                    required=False, min_value=1.0, max_value=100000.0),
                ParameterSpec(
                    name="sts_points", type="int",
                    description=f"谱的采样点数。{_FROM_TABLE}表值 {wf.sts_points}。",
                    required=False, min_value=16, max_value=10000),
                ParameterSpec(
                    name="stab_bias_v", type="float", unit="V",
                    description=f"设谱前的稳定偏压。{_FROM_TABLE}表值 {wf.stab_bias_v} V。",
                    required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="stab_setpoint_a", type="float", unit="A",
                    description=f"设谱前的稳定电流。{_FROM_TABLE}表值 {format_si(wf.stab_setpoint_a)}A。",
                    required=False, min_value=1e-12, max_value=1e-7),
                ParameterSpec(
                    name="temperature_k", type="float", unit="K",
                    description=(f"样品温度（只用于算展宽下限，不驱动硬件）。"
                                 f"{_FROM_TABLE}表值 {wf.temperature_k} K。"),
                    required=False, min_value=0.01, max_value=400.0),
                ParameterSpec(
                    name="max_rounds", type="int",
                    description=f"扎针↔测谱最多来回几次。{_FROM_TABLE}表值 {wf.max_rounds}。",
                    required=False, min_value=1, max_value=20),
                ParameterSpec(
                    name="allow_on_qplus", type="bool",
                    description=("默认关 —— 在 nm 尺度上扎针**本来就是常规操作**,不需要额外"
                                "许可。只有当那道守卫被重新打开时(MAST_QPLUS_POKE_GUARD=1),"
                                "这个开关才有意义。真正保护音叉的是深度包络(超上限**拒绝**,"
                                "绝不夹紧),以及扎针前自动把偏压调到 20 mV —— 不是这个开关。"),
                    required=False, default=False),
            ],
            estimated_duration_s=900.0,
            composition_level=3,
            tags=["tip", "sts", "spectroscopy", "shockley", "composite"],
        )

    def validate_params(self, params: dict) -> list[str]:
        """标准校验 + 下压深度的针尖包络（超上限拒绝，不夹紧）。"""
        errors = super().validate_params(params)
        depth_nm = params.get("poke_depth_nm")
        if depth_nm is None:
            depth_nm = SPECTROSCOPY_TIP.poke_depth_nm
        from mast.skills.builtins._tip_policy import apply_tip_policy

        _, plan = apply_tip_policy(
            {"poke_deep_depth_m": -abs(float(depth_nm)) * 1e-9},
            ("poke_deep_depth_m",))
        if plan is not None and not plan.ok:
            errors.extend(plan.refusals)
        return errors

    def plan_dynamic(self, params: dict, executor: GraphExecutor
                     ) -> Iterator[CompositeStep]:
        wf = resolve_spectroscopy({k: params.get(k) for k in (
            "onset_tol_v", "poke_depth_nm", "critical_repeat_n", "mod_amp_v",
            "mod_freq_hz", "sts_points", "stab_bias_v", "stab_setpoint_a",
            "temperature_k", "max_rounds")})
        phases: list[dict] = []

        if _preflight(executor):
            return
        if _qplus_blocked(executor, "MakeSpectroscopyTip", params):
            return

        # ── 衬底:定不出期望值就不要开始扎针 ──
        # 先解析再动手,是因为「扎完十几次才发现判据无从建立」等于白白消耗表面
        # 与针尖。读不到 ≠ 可以按 Au(111) 处理。
        from mast.core.sample_facts import resolve_substrate

        facts = resolve_substrate(params.get("substrate") or None)
        expected = params.get("expected_onset_v")
        if expected is None:
            if not facts.available:
                self._phase_out = _wrap([], "sts_no_substrate", facts.reason)
                return
            if facts.surface_state_onset_v is None:
                self._phase_out = _wrap([], "sts_no_surface_state", facts.reason)
                return
            expected = float(facts.surface_state_onset_v)
        expected = float(expected)
        start_v, end_v = wf.sweep_window_v(expected)
        executor.set_partial("substrate", facts.material)
        executor.set_partial("expected_onset_v", expected)
        executor.set_partial("sweep_window_v", [start_v, end_v])

        poke_wf = wf.poke_workflow()
        outcome = "incomplete"
        passed_round: int | None = None

        for rnd in range(1, int(wf.max_rounds) + 1):
            # ── 1) 浅扎:金属性针尖要的是一个小而圆的簇,不是最尖的针 ──
            poke = yield from poke_phase(executor, poke_wf, prefix=f"P{rnd}")
            phases.append(poke)
            if poke.get("surface_spent"):
                outcome = "surface_spent"
                break

            # 验证图的扫描中心必须与实际移针位置一致。
            spot = yield from _relocate(executor, step_prefix=f"S{rnd}",
                                        purpose="pulse", used=_dirty(executor))
            if spot is None:
                outcome = "surface_spent"
                phases.append({"phase": "sts", "reason": "找不到干净区域测谱"})
                break
            sx, sy, map_known = spot

            # ── 2.5) 落点必须是**验过的干净台面**，不是「没在已用坐标表里」 ──
            #
            # 此前这一步不存在：``_relocate`` 只问地图要一个「没标记过」的坐标，
            # 挪过去就直接测谱。而同一个文件里的 MakeAtomicResolutionTip 在同一件
            # 事上是「扫一张 → FindFlatRegion(same_terrace=True) → 用验过的中心」。
            #
            # 载重的不是「测得准不准」，是**这笔账记给谁**：谱落在台阶边 / 吸附物
            # 上时肖克利台阶本来就不该出现，而循环会把它算成「针尖还不够金属性」
            # 接着扎 —— 一个位置问题被当成针尖问题，扎多少次都好不了，
            # 而每一次都在消耗针尖。
            #
            # 用 ``flat_poke_sites`` 而不是自己再搭一套 ScanAt+FindFlatRegion：
            # 它已经带着「避开每一片搜过的 200 nm 区域」「在台面上调平」，
            # 还分得出第三种情况 —— **台阶被劈开是针尖的问题，换地方没用**。
            # 自己搭的那版会把它误判成「表面脏」然后停手，而正确答案是打脉冲。
            status, sites = yield from flat_poke_sites(
                executor, poke_wf, step_prefix=f"S{rnd}T",
                used=_dirty(executor), want=1)
            if status == "split_tip":
                outcome = "sts_split_tip"
                phases.append({
                    "phase": "terrace", "round": rnd, "map_known": map_known,
                    "reason": ("大图上台阶出现重影（多针尖）—— 找不到台面的原因"
                               "**在针尖不在表面**，换位置无效。这一路要打脉冲，"
                               "不是接着浅扎。")})
                break
            if status == "spent" or not sites:
                # **不是针尖的问题**，所以不再扎 —— 这条是整段改动的要害。
                outcome = "sts_no_clean_terrace"
                phases.append({
                    "phase": "terrace", "round": rnd, "map_known": map_known,
                    "reason": ("找不到干净台面来测谱 —— 这是**表面**的问题，"
                               "不是针尖的。继续扎针只会把一根可能已经合格的"
                               "针尖扎废。建议粗动换区或换样品位置后重来。")})
                break
            sx, sy = sites[0]
            phases.append({"phase": "terrace", "round": rnd, "spot": [sx, sy],
                           "reason": "落点已验过：单台面窗口内无台阶，并已在台面上调平"})

            # ── 3) 稳定条件 + lock-in + 谱参数 ──
            yield CompositeStep(
                step_id=f"S{rnd}:stab_bias", skill_name="SetBias",
                params={"bias_v": wf.stab_bias_v},
                optional=False, checkpoint_after=False, tags=("sts",))
            yield CompositeStep(
                step_id=f"S{rnd}:stab_setpoint", skill_name="SetSetpoint",
                params={"setpoint_a": wf.stab_setpoint_a},
                optional=False, checkpoint_after=False, tags=("sts",))
            yield CompositeStep(
                step_id=f"S{rnd}:lockin_on", skill_name="ConfigureLockIn",
                params={"mod_on": True, "amplitude_v": wf.mod_amp_v,
                        "frequency_hz": wf.mod_freq_hz},
                optional=False, checkpoint_after=False, tags=("sts", "lockin"))
            yield CompositeStep(
                step_id=f"S{rnd}:sts_config", skill_name="ConfigureSTS",
                params={"start_v": start_v, "end_v": end_v,
                        "num_points": int(wf.sts_points)},
                optional=False, checkpoint_after=False, tags=("sts",))
            if not all(_ok(executor, f"S{rnd}:{s}") for s in
                       ("stab_bias", "stab_setpoint", "lockin_on", "sts_config")):
                phases.append({"phase": "sts", "reason": "谱采集条件没能建立",
                               "map_known": map_known})
                outcome = "sts_spectrum_unavailable"
                break

            # ── 4) 采谱 ──
            yield CompositeStep(
                step_id=f"S{rnd}:acquire", skill_name="AcquireSTS",
                params={"save_basename": f"spectip_r{rnd}"},
                optional=False, checkpoint_after=True, tags=("sts",))
            dat = _data(executor, f"S{rnd}:acquire")
            dat_path = str(dat.get("path") or dat.get("file_path") or "")
            if not dat_path:
                # 谱跑完了但没有落盘路径(autosave 关着?)—— 这一轮判不了,但不是
                # 针尖的问题,别把它算成一次失败的锻造尝试。
                phases.append({"phase": "sts", "round": rnd,
                               "reason": "谱采集完成但拿不到 .dat 路径（检查 Nanonis 的自动保存）",
                               "map_known": map_known})
                outcome = "sts_spectrum_unavailable"
                break

            # ── 5) 判据 ──
            yield CompositeStep(
                step_id=f"S{rnd}:assess", skill_name="AssessShockleyOnset",
                params={"dat_path": dat_path,
                        "expected_onset_v": expected,
                        "substrate": facts.material or "",
                        "tol_v": wf.onset_tol_v,
                        "lockin_mod_vrms": wf.mod_amp_v,
                        "temperature_k": wf.temperature_k,
                        "width_max_v": wf.onset_width_max_v},
                optional=True, checkpoint_after=True, tags=("sts", "assess"))
            verdict = _data(executor, f"S{rnd}:assess")
            entry = {
                "phase": "sts", "round": rnd, "spot": [sx, sy],
                "dat_path": dat_path, "map_known": map_known,
                "passed": bool(verdict.get("passed")),
                "onset_v": verdict.get("onset_v"),
                "reasons": verdict.get("reasons"),
                "warnings": verdict.get("warnings"),
            }
            if not _ok(executor, f"S{rnd}:assess"):
                entry["reason"] = "谱分析没能完成"
            elif entry["passed"]:
                entry["reason"] = (
                    f"onset {float(verdict.get('onset_v') or 0.0) * 1000:.0f} mV"
                    f"（期望 {expected * 1000:.0f}±{wf.onset_tol_v * 1000:.0f} mV）")
            else:
                entry["reason"] = (
                    f"第 {rnd} 轮未通过：{'、'.join(verdict.get('reasons') or [])}")
            phases.append(entry)

            if entry["passed"]:
                passed_round = rnd
                outcome = "sts_tip_ready"
                break
        else:
            outcome = "sts_rounds_exhausted"

        # ── 收尾:调制一定要关掉 ──
        # 把 lock-in 调制留在成像态上,后面每一张图都带着一个没人知道的抖动。
        # 不能写 try/finally —— executor 中止时生成器会被 close(),在 finally 里
        # yield 会抛 GeneratorExit(见 _tip_phases.poke_phase 的同款注释)。
        yield CompositeStep(
            step_id="finalize:lockin_off", skill_name="ConfigureLockIn",
            params={"mod_on": False},
            optional=True, checkpoint_after=False, tags=("finalize",))

        extra = (f"第 {passed_round} 轮通过。" if passed_round else "")
        self._phase_out = _wrap(phases, outcome, extra)
        executor.set_partial("outcome", outcome)


# ── 配方 2:原子分辨针尖 ────────────────────────────────────────────────────

#: 线级闸每轮打分用最近这么多条线 —— 整帧的中位会被扰动之前那些行拖住。
_LINE_WINDOW = 24
# 连续通过若干轮才启动确认帧，避免把单轮波动当作稳定改善。
_LINE_STREAK = ADVISORY_STREAK
#: 憋了这么多轮就**不管线级建议**照扫一张确认帧。线级阈值只在 2 正 2 负上标过,
#: 一个没标定好的阈值可以推迟判读,不可以永远挡住它。
_FORCE_EVAL_EVERY = 4

# 帧内突变使用后半与前半的角向集中度比值。
# 半帧与整帧的频域分辨率不同，不能直接比较绝对集中度。
# 阈值是可配置的启发式，公开版本不提供实验标定或成功率保证。
_MID_FRAME_CHANGE_RATIO = 0.04

class MakeAtomicResolutionTip(_TipComposite):
    """快扫小平区 + 偏压随机扰动,直到图里出现原子相。"""

    def metadata(self) -> SkillMetadata:
        wf = ATOMIC_TIP
        return SkillMetadata(
            name="MakeAtomicResolutionTip",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            capabilities=frozenset({"tip_shaping"}),
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把一根针尖哄出**原子**分辨:找一块无台阶的平区,在低偏压下"
                "(20 mV / 500 pA)快扫它,同时让偏压在 ±20 mV 之内随机跳变;"
                "然后扫一张干净帧,查它的 FFT 里有没有真正的晶格。扰动跑在一张"
                "**牺牲帧**里,判定来自紧随其后的那张干净帧 —— 一张在扰动期间采的"
                "帧,它自己的证据里就带着那份扰动。每隔几轮不成,就回退去做两次"
                "临界深度扎针。在 qPlus 传感器上默认拒绝,除非显式允许。"
                "**这是阶梯里的一档，不是入口** —— 用户只说「我想要原子分辨」"
                "时调 AchieveAtomicResolution，由它决定要不要走到这一档。"
            ),
            parameters=[
                ParameterSpec(
                    name="eval_frame_nm", type="float", unit="nm",
                    description=(f"评估帧视野。{_FROM_TABLE}表值 {wf.eval_frame_nm} nm。"
                                 f"与像素数一起决定判不判得出原子相。"),
                    required=False, min_value=0.5, max_value=500.0),
                ParameterSpec(
                    name="eval_pixels", type="int",
                    description=f"评估帧像素数。{_FROM_TABLE}表值 {wf.eval_pixels}。",
                    required=False, min_value=16, max_value=4096),
                ParameterSpec(
                    name="eval_bias_v", type="float", unit="V",
                    description=f"成像偏压。{_FROM_TABLE}表值 {wf.eval_bias_v} V。",
                    required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="eval_setpoint_a", type="float", unit="A",
                    description=f"成像电流。{_FROM_TABLE}表值 {format_si(wf.eval_setpoint_a)}A。",
                    required=False, min_value=1e-12, max_value=1e-7),
                ParameterSpec(
                    name="eval_line_time_s", type="float", unit="s",
                    description=f"每线时间（要快）。{_FROM_TABLE}表值 {wf.eval_line_time_s} s。",
                    required=False, min_value=1e-3, max_value=60.0),
                ParameterSpec(
                    name="wiggle_upper_v", type="float", unit="V",
                    description=(f"偏压扰动的幅值上限。{_FROM_TABLE}表值 "
                                 f"{wf.wiggle_upper_v} V。BiasWiggle 另有 ±0.1 V 硬帽。"),
                    required=False, min_value=0.001, max_value=0.1),
                ParameterSpec(
                    name="wiggle_burst_s", type="float", unit="s",
                    description=f"一次扰动突发多久。{_FROM_TABLE}表值 {wf.wiggle_burst_s} s。",
                    required=False, min_value=0.1, max_value=10.0),
                ParameterSpec(
                    name="time_budget_min", type="float", unit="min",
                    description=(
                        "总时间预算（分钟）。0 = 不设，只受 max_cycles 约束。"
                        "**通过之后默认不停**：只要还有时间预算就继续尝试，"
                        "确认帧都落成 .sxm，图会累积；针尖只有一个、没有备份，"
                        "针尖质量可能改善也可能退化，每轮均需重新验证。"),
                    required=False, default=0.0, min_value=0.0, max_value=1440.0),
                ParameterSpec(
                    name="stop_at_first_pass", type="bool",
                    description=(
                        "第一次通过就停。要的是**针尖处在好状态**（后面还要接着做别的）"
                        "而不是**一张图**时用它 —— 那两个目标的最优停机点不一样。"),
                    required=False, default=False),
                ParameterSpec(
                    name="max_cycles", type="int",
                    description=f"最多做几轮扰动+评估。{_FROM_TABLE}表值 {wf.max_cycles}。",
                    required=False, min_value=1, max_value=50),
                ParameterSpec(
                    name="cycles_per_fallback", type="int",
                    description=(f"每几轮不成就回退去扎两下。{_FROM_TABLE}表值 "
                                 f"{wf.cycles_per_fallback}。"),
                    required=False, min_value=1, max_value=50),
                ParameterSpec(
                    name="fallback_budget", type="int",
                    description=f"回退扎针最多几次。{_FROM_TABLE}表值 {wf.fallback_budget}。",
                    required=False, min_value=0, max_value=20),
                ParameterSpec(
                    name="expected_a_nm", type="float", unit="nm",
                    description=("预期的原子行间距。留空则取自衬底;"
                                 "填 0 就关掉这道检查。"),
                    required=False, min_value=0.0, max_value=10.0),
                ParameterSpec(
                    name="substrate", type="str",
                    description="衬底名;留空 = 从样品记录里推断。",
                    required=False, default=""),
                ParameterSpec(
                    name="allow_on_qplus", type="bool",
                    description=("默认关 —— 在 nm 尺度上扎针**本来就是常规操作**,不需要额外"
                                "许可。只有当那道守卫被重新打开时(MAST_QPLUS_POKE_GUARD=1),"
                                "这个开关才有意义。真正保护音叉的是深度包络(超上限**拒绝**,"
                                "绝不夹紧),以及扎针前自动把偏压调到 20 mV —— 不是这个开关。"),
                    required=False, default=False),
            ],
            estimated_duration_s=1200.0,
            composition_level=3,
            tags=["tip", "atomic", "lattice", "wiggle", "composite"],
        )

    def plan_dynamic(self, params: dict, executor: GraphExecutor
                     ) -> Iterator[CompositeStep]:
        wf = resolve_atomic({k: params.get(k) for k in (
            "eval_frame_nm", "eval_pixels", "eval_bias_v", "eval_setpoint_a",
            "eval_line_time_s", "wiggle_upper_v", "wiggle_burst_s",
            "max_cycles", "cycles_per_fallback", "fallback_budget")})
        phases: list[dict] = []

        # ── 帧参数先自查:判不出原子相的帧不值得扫 ──
        # 扫完一张判不了的图再说,白花一帧的时间,而且流程会把「判不了」误读成
        # 「还没弄出原子相」接着去扰动针尖。
        problem = wf.scale_problem()
        if problem:
            self._phase_out = _wrap([], "atomic_scale_misconfigured", problem)
            return
        if _preflight(executor):
            return
        if int(wf.fallback_budget) > 0 and _qplus_blocked(
                executor, "MakeAtomicResolutionTip", params):
            return

        expected_a = params.get("expected_a_nm")
        substrate = params.get("substrate") or ""
        executor.set_partial("eval_nm_per_px", wf.eval_pixel_size_nm())

        # ── 找一块无台阶的平区 ──
        # same_terrace 判据保证窗口内没有台阶 —— 台阶会在 FFT 里造出自己的低频
        # 结构,而且原子分辨本来就该在平台上做。
        spot = yield from _relocate(executor, step_prefix="F", purpose="tip_shape",
                                    used=_dirty(executor))
        if spot is None:
            self._phase_out = _wrap([{"phase": "setup", "map_known": False}],
                                    "surface_spent")
            return
        cx, cy, map_known = spot
        yield CompositeStep(
            step_id="F:wide", skill_name="ScanAt",
            params={"center_x_m": cx, "center_y_m": cy,
                    "size_m": max(wf.eval_frame_nm * 4.0, 20.0) * 1e-9,
                    "wait_timeout_s": wf.scan_timeout_s},
            optional=True, checkpoint_after=False, tags=("setup",))
        yield CompositeStep(
            step_id="F:save", skill_name="SaveScan", params={},
            optional=True, checkpoint_after=False, tags=("setup",))
        yield CompositeStep(
            step_id="F:latest", skill_name="GetLatestScanFile", params={},
            optional=True, checkpoint_after=False, tags=("setup",))
        wide_path = _scan_path(executor, "F:save", "F:latest")
        fx, fy = cx, cy
        if wide_path:
            yield CompositeStep(
                step_id="F:flat", skill_name="FindFlatRegion",
                params={"scan_path": wide_path,
                        "min_window_m": (wf.eval_frame_nm + wf.flat_margin_nm) * 1e-9,
                        "same_terrace": True},
                optional=True, checkpoint_after=True, tags=("setup", "flat"))
            flat = _data(executor, "F:flat")
            if flat.get("center_x_m") is not None:
                fx = float(flat["center_x_m"])
                fy = float(flat["center_y_m"])
        phases.append({"phase": "setup", "flat_center": [fx, fy],
                       "wide_scan_path": wide_path, "map_known": map_known,
                       "nm_per_px": wf.eval_pixel_size_nm()})

        poke_wf = wf.poke_workflow()
        fallbacks_used = int(executor.progress.partial_data.get("fallbacks", 0))
        line_streak = 0
        cycles_since_eval = 0
        import time as _time
        t_start = _time.monotonic()
        budget_s = float(params.get("time_budget_min") or 0.0) * 60.0
        stop_at_first_pass = bool(params.get("stop_at_first_pass", False))
        best_conc = -1.0
        best_frame: dict | None = None
        outcome = "incomplete"
        frame_m = wf.eval_frame_nm * 1e-9
        off_m = wf.sacrificial_offset_nm * 1e-9

        for cyc in range(1, int(wf.max_cycles) + 1):
            if _budget_left(t_start, budget_s) <= 0:
                outcome = ("atomic_time_budget_done" if best_frame
                           else "atomic_time_budget_exhausted")
                phases.append({"phase": "atomic", "cycle": cyc,
                               "reason": "时间预算用完了。"})
                break
            pfx = f"C{cyc}"

            # ── 1) 牺牲帧里打扰动 ──
            # 边扫边扰动:针尖在移动,扰动的损伤散布在一片而不是堆在一个点上。
            # 这一帧不保存也不判定 —— 它的对比度被扰动本身弄乱了。
            yield CompositeStep(
                step_id=f"{pfx}:sac_bias", skill_name="SetBias",
                params={"bias_v": wf.eval_bias_v},
                optional=False, checkpoint_after=False, tags=("wiggle",))
            yield CompositeStep(
                step_id=f"{pfx}:sac_setpoint", skill_name="SetSetpoint",
                params={"setpoint_a": wf.eval_setpoint_a},
                optional=False, checkpoint_after=False, tags=("wiggle",))
            yield CompositeStep(
                step_id=f"{pfx}:sac_configure", skill_name="ConfigureScan",
                params={"center_x_m": fx + off_m, "center_y_m": fy,
                        "width_m": frame_m, "height_m": frame_m,
                        "set_scan_speed": True,
                        "line_time_s": wf.eval_line_time_s},
                optional=False, checkpoint_after=False, tags=("wiggle",))
            yield CompositeStep(
                step_id=f"{pfx}:sac_start", skill_name="StartScan",
                params={"direction": "up"},
                optional=True, checkpoint_after=False, tags=("wiggle",))
            yield CompositeStep(
                step_id=f"{pfx}:wiggle", skill_name="BiasWiggle",
                params=wf.wiggle_params(),
                optional=True, checkpoint_after=True, tags=("wiggle",))
            wig = _data(executor, f"{pfx}:wiggle")

            # 线级建议用于决定是否暂缓昂贵的确认帧。
            # 线级筛查不能永久否决确认；达到规定轮数后仍强制采集确认帧。
            yield CompositeStep(
                step_id=f"{pfx}:lines", skill_name="AssessAtomicLines",
                params={"direction": 1, "n_recent_lines": _LINE_WINDOW},
                optional=True, checkpoint_after=False, tags=("wiggle", "lines"))
            adv = _data(executor, f"{pfx}:lines")
            line_med = adv.get("line_snr_median")
            line_streak = (line_streak + 1
                           if adv.get("worth_a_frame") else 0)
            cycles_since_eval += 1
            forced = cycles_since_eval >= _FORCE_EVAL_EVERY
            if line_med is not None and line_streak < _LINE_STREAK and not forced:
                # 单轮质量波动不等于针尖损伤；这里依据连续通过次数决定是否确认。
                phases.append({
                    "phase": "atomic", "cycle": cyc, "skipped_eval": True,
                    "line_snr_median": line_med,
                    "line_streak": line_streak,
                    "wiggle_flips": wig.get("flips_executed"),
                    "reason": adv.get("advisory") or "线级判读没给出建议",
                })
                continue

            # 扫描一定要停 —— 牺牲帧跑完整张没有意义,而且下一步要重设帧。
            yield CompositeStep(
                step_id=f"{pfx}:sac_stop", skill_name="StopScan", params={},
                optional=True, checkpoint_after=False, tags=("wiggle",))
            cycles_since_eval = 0
            line_streak = 0

            # ── 2) 干净帧评估 ──
            yield CompositeStep(
                step_id=f"{pfx}:eval_scan", skill_name="ScanAt",
                params={"center_x_m": fx, "center_y_m": fy, "size_m": frame_m,
                        "bias_v": wf.eval_bias_v,
                        "setpoint_a": wf.eval_setpoint_a,
                        "pixels": int(wf.eval_pixels),
                        "line_time_s": wf.eval_line_time_s,
                        "wait_timeout_s": wf.scan_timeout_s},
                optional=False, checkpoint_after=False, tags=("eval",))
            yield CompositeStep(
                step_id=f"{pfx}:eval_save", skill_name="SaveScan", params={},
                optional=True, checkpoint_after=False, tags=("eval",))
            yield CompositeStep(
                step_id=f"{pfx}:eval_latest", skill_name="GetLatestScanFile",
                params={}, optional=True, checkpoint_after=False, tags=("eval",))
            eval_path = _scan_path(executor, f"{pfx}:eval_save",
                                   f"{pfx}:eval_latest")
            entry: dict[str, Any] = {
                "phase": "atomic", "cycle": cyc,
                "wiggle_flips": wig.get("flips_executed"),
                "scan_path": eval_path,
            }
            if not eval_path:
                entry["reason"] = f"第 {cyc} 轮拿不到评估帧的文件路径"
                phases.append(entry)
                continue

            assess_params = {"scan_path": eval_path, "substrate": substrate,
                             "snr_min": wf.snr_min,
                             "concentration_min": wf.concentration_min,
                             "sharpness_min": wf.sharpness_min}
            if expected_a is not None:
                assess_params["expected_a_nm"] = float(expected_a)
            yield CompositeStep(
                step_id=f"{pfx}:assess", skill_name="AssessAtomicPhase",
                params=assess_params,
                optional=True, checkpoint_after=True, tags=("eval", "assess"))
            verdict = _data(executor, f"{pfx}:assess")
            entry.update({
                "passed": bool(verdict.get("passed")),
                "period_fast_axis_nm": verdict.get("period_fast_axis_nm"),
                "angular_concentration": verdict.get("angular_concentration"),
                "reasons": verdict.get("reasons"),
            })
            # 发出就绪结论前检查帧内两半的一致性，防止整帧平均值掩盖采集过程中的针尖变化。
            halves = verdict.get("half_concentrations")
            if entry["passed"] and halves and len(halves) == 2:
                first, second = float(halves[0] or 0.0), float(halves[1] or 0.0)
                if first > 0 and (second / first) < _MID_FRAME_CHANGE_RATIO:
                    entry["passed"] = False
                    entry["mid_frame_tip_change"] = {"first": first, "second": second,
                                                     "ratio": second / first}
                    entry["reason"] = (
                        f"第 {cyc} 轮整帧判过（conc "
                        f"{verdict.get('angular_concentration')}），但**针尖在这一帧"
                        f"中途变了**：前半 {first:.0f} / 后半 {second:.0f}"
                        f"（比值 {second / first:.4f} < {_MID_FRAME_CHANGE_RATIO}）。"
                        f"这一帧证明不了针尖**现在**好 —— 不发证，接着修。")

            if entry["passed"]:
                per = verdict.get("period_fast_axis_nm")
                conc = verdict.get("angular_concentration")
                entry["reason"] = (
                    f"第 {cyc} 轮出现原子相"
                    + (f"（快扫方向周期 {float(per):.3f} nm）" if per else ""))
                phases.append(entry)
                outcome = "atomic_tip_ready"
                # 确认帧独立存盘，预算内可继续尝试；后续质量退化不会改变已保存帧。
                if conc is not None and float(conc or 0) > float(best_conc):
                    best_conc = float(conc)
                    best_frame = {"cycle": cyc, "scan_path": eval_path,
                                  "angular_concentration": best_conc,
                                  "period_fast_axis_nm": per}
                if stop_at_first_pass:
                    break
                if _budget_left(t_start, budget_s) <= 0:
                    outcome = "atomic_time_budget_done"
                    break
                phases.append({
                    "phase": "atomic", "cycle": cyc, "keep_going": True,
                    "reason": ("已经拿到一张（角向集中度 %s）。**图已落袋**，"
                               "还有 %.0f 分钟预算 —— 接着试，最后交最好的那张。"
                               % (("%.1f" % conc) if conc else "?",
                                  _budget_left(t_start, budget_s) / 60.0))})
                line_streak = 0
                continue
            if "scale_gate" in (verdict.get("reasons") or []):
                # 判不了 ≠ 没有。接着扰动只会白白消耗针尖。
                entry["reason"] = (
                    f"评估帧判不出原子相（{verdict.get('nm_per_px')} nm/px 太粗）—— "
                    f"这不等于没有原子相。")
                phases.append(entry)
                outcome = "atomic_scale_misconfigured"
                break
            entry["reason"] = (
                f"第 {cyc} 轮无原子相：{'、'.join(verdict.get('reasons') or [])}")
            phases.append(entry)

            # ── 3) 每几轮不成,退回去扎两下 ──
            # 做不出来就退回扎针法扎两下,再回来继续。
            if (cyc % int(wf.cycles_per_fallback) == 0
                    and fallbacks_used < int(wf.fallback_budget)):
                fallbacks_used += 1
                executor.set_partial("fallbacks", fallbacks_used)
                poke = yield from poke_phase(executor, poke_wf,
                                             prefix=f"K{fallbacks_used}")
                phases.append(poke)
                if poke.get("surface_spent"):
                    outcome = "surface_spent"
                    break

                # 扎针后移动到新的平坦区域再评估。
                # 避免把处理位置的局部形貌变化误归因于针尖质量。
                spot2 = yield from _relocate(
                    executor, step_prefix=f"K{fallbacks_used}R",
                    purpose="tip_shape", used=_dirty(executor))
                if spot2 is None:
                    phases.append({"phase": "relocate", "after_poke": fallbacks_used,
                                   "reason": "扎完针找不到新的干净落点了"})
                    outcome = "surface_spent"
                    break
                fx, fy = spot2[0], spot2[1]
                phases.append({"phase": "relocate", "after_poke": fallbacks_used,
                               "flat_center": [fx, fy], "map_known": spot2[2],
                               "reason": "扎完针换到新落点继续 —— 不在刚扎出来的坑上判读"})
                line_streak = 0
                cycles_since_eval = 0
        else:
            outcome = "atomic_cycles_exhausted"

        # 拿到过图就不许报「没拿到」。改成「通过之后默认接着试」之后，正常成功的
        # 路径也会走到 for-else —— 那时候 outcome 会被写成 cycles_exhausted，
        # 而报告里明明躺着一张 passed 的确认帧。**停下来的理由**和
        # **有没有拿到东西**是两件事，别让前者覆盖后者。
        if best_frame is not None and outcome in (
                "atomic_cycles_exhausted", "atomic_time_budget_exhausted",
                "incomplete"):
            outcome = ("atomic_time_budget_done"
                       if outcome == "atomic_time_budget_exhausted"
                       else "atomic_tip_ready")

        if best_frame is not None:
            phases.append({"phase": "best", **best_frame,
                           "reason": ("**交出的是这一段里最好的那张**（角向集中度 "
                                      "%.1f，第 %d 轮）—— 图会累积，针尖不会。"
                                      % (best_frame["angular_concentration"],
                                         best_frame["cycle"]))})
        self._phase_out = _wrap(phases, outcome)
        self._phase_out["best_frame"] = best_frame
        executor.set_partial("outcome", outcome)


def make_tools(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill

    return [wrap_skill(cls, context_provider)
            for cls in (MakeSpectroscopyTip, MakeAtomicResolutionTip)]


__all__ = ["MakeSpectroscopyTip", "MakeAtomicResolutionTip"]
