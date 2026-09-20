"""谱**数据质量**闸的技能外壳 —— 判据本体在 ``mast.vision.spectroscopy`` 里。

:class:`AssessSpectrum` —— 一条存盘的 ``.dat`` 谱值不值得留。

**这一层只做 IO 与列名解析,一个阈值都不判**。判据是
:func:`mast.vision.spectroscopy.assess_spectrum_quality` 这个纯函数,零 IO、阈值
全参数化,所以合成数据测得动(白噪声误报率、两族对照、tip switch 分辨力矩阵都在
``tests/v2/unit/vision/test_spectrum_quality.py`` 里)。

## 为什么不放进 ``tip_spectro_assess.py``

那个文件是**针尖**验收(模块名与 docstring 都这么说)。谱数据质量不是针尖判据:
一条 ``discard`` 的谱可能只是窗口开错了、相位反了、表面不是那个面。两件事分开
放,免得下游把「这条谱不好」读成「这根针不行」而去反复修一根其实没问题的针。

## 只接 ``.dat`` 路径,不接内存数组

三条理由,任何一条单独成立(S4 STS 设计 D7):

1. agent 的参数适配器只认标量 int/float/str/bool —— 一条 400 点谱塞成逗号串会
   正面撞上「LLM 丢指数」那类事故;
2. checkpoint 不变式:数组不进 composite 的 partial_data;
3. ``.dat`` 头里带真实 xy,那是逐点位置核对的唯一依据。

**没有 .dat ⇒ 这一点是 ``unrated``,不是 ``discard``。** 「谱没落盘」是流程问题,
不是数据质量问题 —— 这两句话在闸门那头会走完全不同的分支。

## 三态,不是两态

``success=True`` 只要**文件读得动**;判决在 ``data.verdict`` 的四态闭集里
(``keep`` / ``keep_flagged`` / ``discard`` / ``unrated``)。技能失败保留给「这件事
没做成」—— 文件不存在、读不出来、``[DATA]`` 段是空的。把「这条谱不合格」表达成
技能失败,会让 composite 的 ``optional=False`` 步骤直接中止整条流程,而「这条谱
不好」恰恰是流程要处理的正常情况。

文件读得动、但里面没有电流列 ⇒ 仍然 ``success=True``,``verdict="unrated"``、
``reasons=["no_current_column"]``。那不是「没做成」,那是**如实回答了「判不了」**。

## ⚠️ 记录层的已知缺陷(不归本文件修)

``io/exp_map.classify_skill("AssessSpectrum")`` 今天返回 ``'sts'`` —— 一个**只读
文件、不碰仪器**的分析技能会在针尖当前位置落一个 ``kind='sts', status='done'``
的假谱学点,虚增谱学产量并污染「附近测过什么」的查询。结构性修法(ANALYSIS 类
一律不落标记 / ``_READONLY_PREFIXES`` 加 ``assess``)在记录层那边做,本文件只在
测试里把这条依赖标出来,见
``tests/v2/unit/skills/builtins/test_spectrum_assess.py``。
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

#: 反扫列的标记。Nanonis 真机列名是 ``Current [bwd] (A)``。
_BWD_MARKERS: tuple[str, ...] = ("[bwd]", "bwd", "backward")

_BIAS_PATTERNS: tuple[tuple[str, ...], ...] = (("bias",), ("voltage",), ("v (v)",))
_CURRENT_PATTERNS: tuple[tuple[str, ...], ...] = (("current",), ("i (a)",))
#: I(z) 的扫描轴。``Z (m)`` 在 I(V) 的 .dat 里通常只出现在**头**里(恒定值),
#: 出现在 ``[DATA]`` 段里才说明它在被扫。
_Z_PATTERNS: tuple[tuple[str, ...], ...] = (("z rel",), ("z (m)",), ("z spectr",))


def _is_backward(name: str) -> bool:
    low = str(name).lower()
    return any(m in low for m in _BWD_MARKERS)


def _pick_directional(columns, patterns, *, backward: bool) -> "str | None":
    """按子串组合挑一列,**显式**区分正/反扫。

    现行的通用取列helper返回**第一个**含子串的列,正反扫只靠字典顺序区分 ——
    真机列序恰好是 ``Current (A)`` 在 ``Current [bwd] (A)`` 之前,所以今天碰巧
    对。那是运气不是设计:换一台机器、换一个通道顺序就会静默取错方向,而两列都
    是电流、量级也一样,没有任何下游判据会报警。
    """
    for pats in patterns:
        for name in columns:
            low = str(name).lower()
            if all(p in low for p in pats) and _is_backward(name) is backward:
                return name
    return None


def _header_kind(header: dict) -> tuple[str, str]:
    """``(kind, 原文)`` —— 头里的 ``Experiment`` 字段**声称**这是什么谱。

    只作交叉检验:字段标签会说谎(它是仪器软件上一次的设置留下的),数据说了算。
    """
    raw = str((header or {}).get("Experiment", "") or "").strip()
    low = raw.lower()
    if "bias" in low:
        return "iv", raw
    if low.startswith("z") or "z spectr" in low:
        return "iz", raw
    return "", raw


class AssessSpectrum(BaseSkill):
    """判一条存盘的 .dat 谱值不值得留（数据质量，不是针尖判据）。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessSpectrum",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "判断一条已保存的单点谱学 .dat 值不值得留。只读，不碰硬件。verdict "
                "是一个封闭的四态集合：keep / keep_flagged / discard / unrated —— 「判不了」"
                "绝不（NEVER）被折叠进「失败」。哪些子判据有资格否决，取决于 spectral_family：饱和、"
                "谱 SNR、正/反扫迟滞永远否决（它们是仪器事实）；尖峰计数与反对称性只在 METALLIC "
                "衬底上否决 —— 在有能隙的衬底上，它们会把每一条好谱都判成坏的，"
                "而且对针尖跳变视而不见。spectral_family='' 表示 UNKNOWN，这和 metallic 不是一回事："
                "只有那三条与 family 无关的判据否决。每一个留空的阈值都是 UNCALIBRATED，"
                "只能给出 'unrated'，绝不给出坏结论 —— gated_criteria/ungated_criteria "
                "说明实际是哪几条做的判决。gap_ev 会报出来，但绝不（NEVER）参与否决，"
                "它也不是一次带隙测量。倒置的 dI/dV 会被标出来，绝不悄悄翻转。success=true "
                "只表示这个文件读得出来；判断在 data.verdict 里。"
            ),
            parameters=[
                ParameterSpec(name="dat_path", type="str",
                              description="已保存的 .dat 谱文件路径。",
                              required=True),
                ParameterSpec(
                    name="kind", type="str",
                    description=("auto / iv / iz。'auto' 从 DATA 本身判断（哪一列在被扫）；"
                                 "文件头只用来交叉核对。"),
                    required=False, default="auto",
                    allowed_values=["auto", "iv", "iz"]),
                ParameterSpec(
                    name="spectral_family", type="str",
                    description=("metallic / gapped / unknown。留空 = unknown。"
                                 "从已登记衬底推断这条线还没接上，所以留空是真的表示 unknown —— "
                                 "它绝不会被当成 metallic。"),
                    required=False, default="",
                    allowed_values=["", "metallic", "gapped", "unknown"]),
                ParameterSpec(
                    name="min_snr", type="float",
                    description=("谱 SNR 的下限（点间噪声上的 5-95 百分位跨度）。默认 UNCALIBRATED "
                                 "—— 留空则这条判据只报一个数。纯白噪声量出来是 3.0-5.3。"),
                    required=False, min_value=0.0, max_value=1e9),
                ParameterSpec(
                    name="max_saturation_frac", type="float",
                    description=("允许多大比例的点贴死在电流轨上。默认 UNCALIBRATED。"),
                    required=False, min_value=0.0, max_value=1.0),
                ParameterSpec(
                    name="max_hysteresis_outlier_frac", type="float",
                    description=("允许正扫与反扫之间有多大比例的点不一致 —— "
                                 "这是唯一对「针尖在扫的中途变了」有分辨力的判据。默认 "
                                 "UNCALIBRATED。"),
                    required=False, min_value=0.0, max_value=1.0),
                ParameterSpec(
                    name="min_smoothness", type="float",
                    description=("曲线平滑度下限。留空：这个数会随点数漂（同一条曲线，40 点时 0.79，"
                                 "1000 点时 0.99），所以一个固定阈值只对某一个 num_points 有效。"
                                 "归一化方案仍然悬着。"),
                    required=False, min_value=0.0, max_value=1.0),
                ParameterSpec(
                    name="require_backward", type="bool",
                    description=("没有反扫列的谱，拒绝给它评级。"),
                    required=False, default=False),
            ],
            estimated_duration_s=2.0,
            composition_level=2,
            tags=["sts", "spectroscopy", "quality", "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        name = "AssessSpectrum"
        path = str(params["dat_path"])
        if not Path(path).exists():
            return SkillResult(skill_name=name, success=False,
                               error=f"文件不存在: {path}")

        try:
            import numpy as np

            from mast.io.nanonis_files import read_dat
            from mast.vision.spectroscopy import (
                SpectrumQualityResult,
                assess_spectrum_quality,
                resolve_spectrum_kind,
            )
        except ImportError as exc:
            return SkillResult(skill_name=name, success=False,
                               error=f"缺依赖: {exc}")

        try:
            dat = read_dat(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=name, success=False,
                               error=f".dat 读取失败: {exc}")

        header = dat.get("header") or {}
        columns = dat.get("columns") or {}
        if not columns:
            return SkillResult(skill_name=name, success=False,
                               error=f"{path} 里没有数据列（[DATA] 段是空的）")

        bias_col = _pick_directional(columns, _BIAS_PATTERNS, backward=False)
        z_col = _pick_directional(columns, _Z_PATTERNS, backward=False)
        cur_col = _pick_directional(columns, _CURRENT_PATTERNS, backward=False)
        cur_bwd_col = _pick_directional(columns, _CURRENT_PATTERNS, backward=True)
        # dI/dV 列名的候选表是**针尖验收那边的单一真源**,这里引用而不是抄一份:
        # 抄一份意味着下一次仪器改了解调通道名,只有其中一份会被改到。
        from mast.skills.builtins.tip_spectro_assess import (
            _DIDV_PATTERNS,
            _pick_column,
        )
        didv_col = _pick_column(columns, _DIDV_PATTERNS)

        def _col(cname):
            return (None if cname is None
                    else np.asarray(columns[cname], dtype=float))

        base = {
            "dat_path": path,
            "bias_column": bias_col,
            "current_column": cur_col,
            "current_bwd_column": cur_bwd_col,
            "z_column": z_col,
            "didv_column": didv_col,
            "didv_source": ("lockin" if didv_col else None),
            "columns_present": list(columns),
        }
        # .dat 头里的真实 xy —— 共用 exp_map 的那一份解析(带「±1 mm 才算数」的
        # 合理性守卫),不在这里再写一遍。
        try:
            from mast.io.exp_map import extract_dat_position
            xy = extract_dat_position(path)
        except Exception:  # noqa: BLE001 — 位置拿不到不影响判据
            xy = None
        base["dat_x_m"], base["dat_y_m"] = (xy if xy else (None, None))

        if cur_col is None:
            # 文件读得动、但里面没有电流列 ⇒ 判不了,不是「没做成」。
            res = SpectrumQualityResult(verdict="unrated",
                                        reasons=("no_current_column",))
            return SkillResult(
                skill_name=name, success=True,
                data={**base, **_result_fields(res),
                      "kind_resolved": "", "kind_evidence": "没有电流列",
                      "spectral_family_source": "n/a"},
                summary=(f"判不了：{path} 里没有电流列。"
                         f"现有列: {sorted(columns)}"))

        # ── kind:内容判据,头只交叉检验 ────────────────────────────
        want = str(params.get("kind") or "auto").strip().lower()
        data_kind, evidence = resolve_spectrum_kind(
            bias_v=_col(bias_col), z_m=_col(z_col))
        hdr_kind, hdr_raw = _header_kind(header)
        extra_warnings: list[str] = []
        if want in ("iv", "iz"):
            kind = want
            evidence = f"调用方显式指定 {want}；数据侧：{evidence}"
        else:
            kind = data_kind
        if kind and hdr_kind and hdr_kind != kind:
            # 「用内容判据不用文件判据」——不一致时以数据为准并点名,而不是
            # 悄悄跟着头走。
            extra_warnings.append("kind_disagrees_with_header")
            evidence += f"；头里写的是 {hdr_raw!r}（不一致，以数据为准）"

        family_raw = str(params.get("spectral_family") or "").strip().lower()
        family = family_raw if family_raw else "unknown"
        family_source = "explicit" if family_raw else "default_unknown"

        bias = _col(bias_col)
        cur = _col(cur_col)
        z_nm = None if z_col is None else _col(z_col) * 1e9

        res = assess_spectrum_quality(
            kind=kind,
            current=cur,
            bias_v=bias,
            z_nm=z_nm,
            current_bwd=_col(cur_bwd_col),
            didv=_col(didv_col),
            spectral_family=family,
            min_snr=_opt_float(params.get("min_snr")),
            max_saturation_frac=_opt_float(params.get("max_saturation_frac")),
            max_hysteresis_outlier_frac=_opt_float(
                params.get("max_hysteresis_outlier_frac")),
            min_smoothness=_opt_float(params.get("min_smoothness")),
            require_backward=bool(params.get("require_backward", False)),
            extra_warnings=tuple(extra_warnings),
        )

        data = {**base, **_result_fields(res),
                "kind_resolved": res.kind,
                "kind_evidence": evidence,
                "spectral_family_source": family_source,
                "header_experiment": hdr_raw}
        return SkillResult(skill_name=name, success=True, data=data,
                           summary=_summarise(res, path))


def _opt_float(v):
    """空 = 未标定 ⇒ ``None``。**不能**在这里兜一个默认阈值:一个兜底的数字会让
    「没标定」看起来像「标定过了」,而那正是未标定阈值绝不产 bad 想挡的事。"""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _result_fields(res) -> dict:
    return {
        "verdict": res.verdict,
        "reasons": list(res.reasons),
        "warnings": list(res.warnings),
        "gated_criteria": list(res.gated_criteria),
        "ungated_criteria": list(res.ungated_criteria),
        "spectral_family": res.spectral_family,
        "n_points": res.n_points,
        "n_points_dropped": res.n_points_dropped,
        "spectrum_snr": res.spectrum_snr,
        "saturation_frac": res.saturation_frac,
        "hysteresis_median_frac": res.hysteresis_median_frac,
        "hysteresis_max_frac": res.hysteresis_max_frac,
        "hysteresis_outlier_points": res.hysteresis_outlier_points,
        "hysteresis_outlier_frac": res.hysteresis_outlier_frac,
        "iv_smoothness": res.iv_smoothness,
        "iv_symmetry": res.iv_symmetry,
        "iv_n_spikes": res.iv_n_spikes,
        "iv_is_stable": res.iv_is_stable,
        "iv_gap_ev": res.iv_gap_ev,
        "iz_fit_r2": res.iz_fit_r2,
        "iz_barrier_ev": res.iz_barrier_ev,
        "iz_decay_per_nm": res.iz_decay_per_nm,
        "iz_n_jumps": res.iz_n_jumps,
        "iz_is_clean_exponential": res.iz_is_clean_exponential,
    }


def _summarise(res, path: str) -> str:
    gated = "、".join(res.gated_criteria) if res.gated_criteria else "无"
    tail = f"（当闸的判据：{gated}）"
    if res.verdict == "unrated":
        why = "、".join(res.reasons) or "无可用判据"
        return (f"判不了（{why}）—— 这既不算合格也不算不合格。{tail}")
    if res.verdict == "discard":
        return f"不值得留（{'、'.join(res.reasons)}）。{tail}"
    if res.verdict == "keep_flagged":
        return (f"可用，但有事要看：{'、'.join(res.warnings)}。{tail}")
    return f"可用。{tail}"


def make_tools(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return [wrap_skill(AssessSpectrum, context_provider)]


__all__ = ["AssessSpectrum"]
