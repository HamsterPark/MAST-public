"""两个针尖验收判据的技能外壳 —— 判据本体在 ``mast.vision`` 里。

* :class:`AssessShockleyOnset` —— 一条 dI/dV 谱里有没有肖克利表面态台阶,位置对
  不对。这是「这根针尖是不是金属性的」的判据。
* :class:`AssessAtomicPhase` —— 一帧扫描图上有没有原子相。这是「原子分辨针尖搞
  出来了没有」的判据。

**这一层只做 IO 与列名/像素尺度解析,一个阈值都不判**。判据是
:func:`mast.vision.spectroscopy.assess_shockley_onset` 与
:func:`mast.vision.atomic_phase.assess_atomic_phase` 两个纯函数,它们零 IO、阈值
全参数化,所以合成数据测得动(白噪声误报率、准周期抖动对照组都在
``tests/v2/unit/vision/`` 里)。

## 期望值从哪来

onset 位置与晶格常数都随**衬底**变。两个技能都可以不传这些数:留空时按
:func:`mast.core.sample_facts.resolve_substrate` 从当前样品记录推断,再去知识库
取常数。推断不出来就**拒绝执行并说清楚** —— 绝不默认按 Au(111) 处理,那会让在
Ag(111) 上的判据永远不通过,而流程会把这个误读成「针尖不行」,反复去修一根其实
没问题的针。

## 三态,不是两态

两个技能都**永远返回 success=True**(只要文件读得动):判据的结论在
``data.passed`` 与 ``data.reasons`` 里。技能失败保留给「这件事没做成」——文件不
存在、通道缺失、依赖缺席。把「判据没通过」表达成技能失败会让 composite 的
``optional=False`` 步骤直接中止整条流程,而「针尖还不够好」恰恰是流程要处理的
正常情况。
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

logger = logging.getLogger(__name__)

#: dI/dV 列名的候选,按优先级。Nanonis 的实际列名见 ``api/routes/signals.py``。
_DIDV_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("li", "demod", "x"),        # "LI Demod 1 X (A)"
    ("lix",),                    # "LIX 1 omega (A)"
    ("demod", "x"),
    ("didv",),
    ("di/dv",),
)


def _pick_column(columns: dict, patterns) -> "str | None":
    """按子串组合挑一列(全部子串都要出现,不区分大小写)。"""
    low = {name: str(name).lower() for name in columns}
    for pats in patterns:
        for name, lname in low.items():
            if all(p in lname for p in pats):
                return name
    return None


def _resolve_expected(explicit_value, substrate_param, *, want: str):
    """``(值, 衬底事实, 错误信息)``。三者只会有一个「有内容」的组合。

    *want* ∈ ``{"onset", "lattice"}``。
    """
    from mast.core.sample_facts import resolve_substrate

    facts = resolve_substrate(substrate_param or None)
    if explicit_value is not None:
        return float(explicit_value), facts, ""
    if not facts.available:
        return None, facts, (
            f"没给期望值，也推断不出衬底：{facts.reason}")
    if want == "onset":
        if facts.surface_state_onset_v is None:
            return None, facts, (
                f"{facts.material} 没有肖克利表面态 —— {facts.reason}"
                f"用别的判据（例如台阶锐度 AssessTipSharpness），"
                f"或换一块有表面态的衬底再验金属性。")
        return float(facts.surface_state_onset_v), facts, ""
    if facts.row_spacing_nm is None:
        return None, facts, f"知识库里没有 {facts.material} 的晶格常数。"
    return float(facts.row_spacing_nm), facts, ""


def _substrate_fields(facts) -> dict:
    """放进结果里的衬底来源痕迹 —— 一个数字从哪来,事后必须查得到。"""
    return {
        "substrate": facts.material,
        "substrate_source": facts.source,
        "substrate_available": facts.available,
    }


class AssessShockleyOnset(BaseSkill):
    """从一条 dI/dV 谱判断针尖是不是金属性的（看肖克利表面态在不在该在的位置）。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessShockleyOnset",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在一条已保存的 dI/dV 谱（.dat）里拟合 Shockley 表面态的 STEP，"
                "并把它的起始能量与本衬底的取值对照 —— 这是用户判定 METALLIC 针尖的检验。只读。"
                "expected_onset_v 留空则从已登记的样品取（Au(111) −0.49 V，Cu(111) −0.44 V，"
                "Ag(111) −0.065 V）。passed=false 表示台阶没有出现在它该在的位置；这和「针尖坏了」"
                "不是一回事 —— 要看 reasons/warnings（扫描窗口设错、或 lock-in phase 倒置，"
                "也会落到这里）。"
            ),
            parameters=[
                ParameterSpec(name="dat_path", type="str",
                              description="已保存的 .dat 谱文件路径。",
                              required=True),
                ParameterSpec(
                    name="expected_onset_v", type="float", unit="V",
                    description=("期望的起始能量。留空则从衬底取。"),
                    required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(
                    name="substrate", type="str",
                    description=("衬底名（例如 'Au(111)'）。留空则从当前样品记录推断。"),
                    required=False, default=""),
                ParameterSpec(
                    name="tol_v", type="float", unit="V",
                    description="起始位置的容差（知识库：±20 mV）。",
                    required=False, default=0.020, min_value=0.001, max_value=0.5),
                ParameterSpec(
                    name="lockin_mod_vrms", type="float", unit="V",
                    description=("这条谱所用的 lock-in 调制幅度（rms）—— 它定下展宽的下限。"),
                    required=False, default=0.005, min_value=0.0, max_value=1.0),
                ParameterSpec(
                    name="temperature_k", type="float", unit="K",
                    description="样品温度 —— 它定下热展宽。",
                    required=False, default=4.2, min_value=0.01, max_value=400.0),
                ParameterSpec(
                    name="width_max_v", type="float", unit="V",
                    description="仍然算作一个 onset 的最大 10-90 台阶宽度。",
                    required=False, default=0.040, min_value=0.001, max_value=1.0),
            ],
            estimated_duration_s=2.0,
            composition_level=1,
            tags=["tip", "sts", "spectroscopy", "shockley", "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        path = str(params["dat_path"])
        if not Path(path).exists():
            return SkillResult(skill_name="AssessShockleyOnset", success=False,
                               error=f"文件不存在: {path}")

        expected, facts, why = _resolve_expected(
            params.get("expected_onset_v"), params.get("substrate"),
            want="onset")
        if expected is None:
            return SkillResult(skill_name="AssessShockleyOnset", success=False,
                               error=why, data=_substrate_fields(facts))

        try:
            import numpy as np

            from mast.io.nanonis_files import read_dat
            from mast.vision.spectroscopy import assess_shockley_onset
        except ImportError as exc:
            return SkillResult(skill_name="AssessShockleyOnset", success=False,
                               error=f"缺依赖: {exc}")

        try:
            dat = read_dat(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name="AssessShockleyOnset", success=False,
                               error=f".dat 读取失败: {exc}")

        columns = dat.get("columns") or {}
        if not columns:
            return SkillResult(skill_name="AssessShockleyOnset", success=False,
                               error=f"{path} 里没有数据列（[DATA] 段是空的）")

        bias_col = _pick_column(columns, (("bias",), ("voltage",), ("v (v)",)))
        if bias_col is None:
            return SkillResult(
                skill_name="AssessShockleyOnset", success=False,
                error=f"谱里找不到偏压列。现有列: {sorted(columns)}")

        didv_col = _pick_column(columns, _DIDV_PATTERNS)
        didv_source = "lockin"
        bias = np.asarray(columns[bias_col], dtype=float)
        if didv_col is not None:
            didv = np.asarray(columns[didv_col], dtype=float)
        else:
            # 没有 lock-in 通道 → 用电流的数值微分。能用，但噪声大得多：
            # 判据的幅度门会跟着变严，所以要在结果里标出来这是降级路径。
            cur_col = _pick_column(columns, (("current",), ("i (a)",)))
            if cur_col is None:
                return SkillResult(
                    skill_name="AssessShockleyOnset", success=False,
                    error=(f"谱里既没有 lock-in 解调通道也没有电流列，"
                           f"算不出 dI/dV。现有列: {sorted(columns)}"))
            order = np.argsort(bias)
            didv = np.gradient(np.asarray(columns[cur_col], dtype=float)[order],
                               bias[order])
            bias = bias[order]
            didv_col = f"d({cur_col})/dV"
            didv_source = "numeric"

        res = assess_shockley_onset(
            bias, didv,
            expected_onset_v=float(expected),
            tol_v=float(params.get("tol_v", 0.020)),
            lockin_mod_vrms=float(params.get("lockin_mod_vrms", 0.005)),
            temperature_k=float(params.get("temperature_k", 4.2)),
            width_max_v=float(params.get("width_max_v", 0.040)),
        )

        data = {
            "dat_path": path,
            "passed": res.passed,
            "onset_v": res.onset_v,
            "onset_err_v": res.onset_err_v,
            "expected_onset_v": res.expected_onset_v,
            "tol_v": res.tol_v,
            "step_height": res.step_height,
            "step_sigma_ratio": res.step_sigma_ratio,
            "width_v": res.width_v,
            "width_floor_v": res.width_floor_v,
            "r2": res.r2,
            "delta_bic": res.delta_bic,
            "n_points_fit": res.n_points_fit,
            "reasons": list(res.reasons) + extra_reasons,
            "warnings": list(res.warnings) + extra_warnings,
            "bias_column": bias_col,
            "didv_column": didv_col,
            "didv_source": didv_source,
            **_substrate_fields(facts),
        }
        if res.passed:
            summary = (f"肖克利 onset {res.onset_v * 1000:.0f} mV"
                       f"（期望 {res.expected_onset_v * 1000:.0f}±"
                       f"{res.tol_v * 1000:.0f} mV）—— 金属性针尖合格。")
        else:
            summary = (f"未通过（{', '.join(res.reasons)}）"
                       + (f"；注意: {', '.join(res.warnings)}" if res.warnings
                          else ""))
        if didv_source == "numeric":
            summary += "（无 lock-in 通道，用电流数值微分，噪声偏大）"
        return SkillResult(skill_name="AssessShockleyOnset", success=True,
                           data=data, summary=summary)

# 覆盖率不足时返回无法判定。当前比例是通用软件启发式，非目标仪器标定；
# 不能因填补了未采集像素而把残帧当成完整证据。
_MIN_COVERAGE = 0.5


class AssessAtomicPhase(BaseSkill):
    """从一帧扫描图判断有没有原子分辨。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessAtomicPhase",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "判断一张已保存的 .sxm 帧有没有 ATOMIC 分辨。三条互相独立的判据必须一致："
                "原子周期带内有一个强峰、是离散的 Bragg "
                "斑点而不是弥散的环（把真实晶格与针尖振铃分开的就是这一条）、以及 FFT 峰的锐度。只读。"
                "比 0.05 nm/px 更粗的帧会 REFUSE 判读（reasons=['scale_gate']），而不是报「没有」"
                "—— 在那个尺度上晶格本来就分不开，所以「没有原子」会是一句假话。晶格常数只对 FAST "
                "扫描方向报（慢轴漂移让另一个方向不可信）。"
            ),
            parameters=[
                ParameterSpec(name="scan_path", type="str",
                              description=".sxm 帧的文件路径。",
                              required=True),
                ParameterSpec(name="channel", type="str",
                              description="形貌通道（标准是 'Z'）。",
                              required=False, default="Z"),
                ParameterSpec(
                    name="expected_a_nm", type="float", unit="nm",
                    description=("期望的原子行间距。留空则从衬底取；填 0 则关掉这项比较。"),
                    required=False, min_value=0.0, max_value=10.0),
                ParameterSpec(
                    name="substrate", type="str",
                    description=("衬底名。留空则从当前样品推断。"),
                    required=False, default=""),
                ParameterSpec(
                    name="snr_min", type="float",
                    description="带内峰信噪比的下限。",
                    required=False, default=4.0, min_value=1.0, max_value=1e6),
                ParameterSpec(
                    name="concentration_min", type="float",
                    description=("角向集中度的下限（离散 Bragg 斑点 vs 弥散环）。实测的分离度："
                                 "真实晶格 97-7645，针尖振铃 1.8-3.3。"),
                    required=False, default=20.0, min_value=1.0, max_value=1e9),
                ParameterSpec(
                    name="sharpness_min", type="float",
                    description="FFT 峰凸显度的下限。",
                    required=False, default=8.0, min_value=1.0, max_value=1e6),
                ParameterSpec(
                    name="allow_reduced_scale", type="bool",
                    description=("在 0.02-0.05 nm/px 的过渡带里也接受肯定结论。"),
                    required=False, default=False),
            ],
            estimated_duration_s=3.0,
            composition_level=1,
            tags=["tip", "atomic", "lattice", "fft", "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        path = str(params["scan_path"])
        channel_name = params.get("channel") or "Z"
        if not Path(path).exists():
            return SkillResult(skill_name="AssessAtomicPhase", success=False,
                               error=f"文件不存在: {path}")

        try:
            import numpy as np

            from mast.io.mosaic import parse_xy_meta
            from mast.io.nanonis_files import read_sxm
            from mast.vision.atomic_phase import assess_atomic_phase
        except ImportError as exc:
            return SkillResult(skill_name="AssessAtomicPhase", success=False,
                               error=f"缺依赖: {exc}")

        # 期望晶格常数是可选的:0 或推断不出来时只是不做这一项比对,不算失败。
        expected_raw = params.get("expected_a_nm")
        facts = None
        expected_a = None
        if expected_raw is not None and float(expected_raw) > 0:
            expected_a = float(expected_raw)
            from mast.core.sample_facts import resolve_substrate
            facts = resolve_substrate(params.get("substrate") or None)
        elif expected_raw is None:
            expected_a, facts, _why = _resolve_expected(
                None, params.get("substrate"), want="lattice")
        else:
            from mast.core.sample_facts import resolve_substrate
            facts = resolve_substrate(params.get("substrate") or None)

        try:
            scan = read_sxm(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name="AssessAtomicPhase", success=False,
                               error=f".sxm 读取失败: {exc}")

        channels = scan.get("channels", {}) or {}
        ch = channels.get(channel_name) or next(iter(channels.values()), None)
        if ch is None:
            return SkillResult(
                skill_name="AssessAtomicPhase", success=False,
                error=f"文件里没有可用通道(要的是 {channel_name!r})")
        arr = ch.get("forward")
        if arr is None:
            arr = ch.get("backward")
        if arr is None:
            return SkillResult(skill_name="AssessAtomicPhase", success=False,
                               error="通道里没有正扫/反扫数据")
        arr = np.asarray(arr, dtype=np.float64)

        nm_per_px = None
        try:
            meta = parse_xy_meta(scan.get("header", {}) or {})
            if meta and arr.ndim == 2 and arr.shape[1]:
                nm_per_px = (float(meta["w"]) / arr.shape[1]) * 1e9
        except Exception:  # noqa: BLE001
            nm_per_px = None

        # 未采集区域的 NaN 被替换为常数后可能产生与扫描无关的谱结构。
        # 技能层先检查有效覆盖率；证据不足时明确弃权，不把残帧的谱峰当成原子分辨证据。
        coverage = float(np.isfinite(arr).mean()) if arr.size else 0.0

        res = assess_atomic_phase(
            arr,
            nm_per_px=nm_per_px,
            expected_a_nm=expected_a,
            snr_min=float(params.get("snr_min", 4.0)),
            concentration_min=float(params.get("concentration_min", 20.0)),
            sharpness_min=float(params.get("sharpness_min", 8.0)),
            allow_reduced_scale=bool(params.get("allow_reduced_scale", False)),
        )

        incomplete = coverage < _MIN_COVERAGE
        passed = bool(res.passed) and not incomplete
        extra_reasons = ["incomplete_frame"] if incomplete else []
        extra_warnings = ([
            "只有 %.0f%% 的像素有数据 —— 这一帧**判不了**原子分辨。缺的部分在做谱"
            "之前会被当成常数，谱上因此多出与扫描无关的结构。"
            "**不要把这个结果读成「针尖不好」**，它说的是这一帧没扫完。"
            % (coverage * 100)] if incomplete else [])

        data = {
            "scan_path": path,
            "coverage": coverage,
            "incomplete_frame": incomplete,
            "passed": passed,
            "passed_before_coverage_gate": bool(res.passed),
            "scale": res.scale,
            "nm_per_px": res.nm_per_px,
            "period_nm": res.period_nm,
            "period_fast_axis_nm": res.period_fast_axis_nm,
            "snr": res.snr,
            "angular_concentration": res.angular_concentration,
            "order_ratio": res.order_ratio,
            "fft_sharpness": res.fft_sharpness,
            "expected_a_nm": res.expected_a_nm,
            "slow_axis_trusted": res.slow_axis_trusted,
            # 按扫描顺序分别上报前后半帧的角向集中度。
            # 针尖可能在帧内变化，整帧结果不能单独证明采集结束时仍处于同一状态。
            "half_concentrations": (list(res.half_concentrations)
                                    if res.half_concentrations else None),
            "reasons": list(res.reasons),
            "warnings": list(res.warnings),
        }
        if facts is not None:
            data.update(_substrate_fields(facts))

        if res.passed:
            per = res.period_fast_axis_nm
            summary = (f"有原子相：快扫方向周期 "
                       f"{per:.3f} nm" if per else "有原子相")
            summary += (f"，角向集中度 {res.angular_concentration:.0f}"
                        f"（阈值 {params.get('concentration_min', 20.0):g}）。")
        elif "scale_gate" in res.reasons:
            summary = (f"这一帧 {res.nm_per_px:.4f} nm/px 太粗，判不了原子相 —— "
                       f"这不等于「没有原子相」。换更小的视野或更多像素再看。")
        else:
            summary = f"没有原子相（{', '.join(res.reasons)}）"
        return SkillResult(skill_name="AssessAtomicPhase", success=True,
                           data=data, summary=summary)


def make_tools(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return [wrap_skill(cls, context_provider)
            for cls in (AssessShockleyOnset, AssessAtomicPhase)]


__all__ = ["AssessShockleyOnset", "AssessAtomicPhase"]
