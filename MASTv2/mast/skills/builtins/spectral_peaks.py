"""``FindSpectralPeaks`` —— 一条 dI/dV 谱里的峰,以及它们的位置与宽度。

``AssessSpectrum`` 回答的是「这条谱能不能用」,是四态质量闸;本技能回答的是另一个问题:
**谱里有哪些峰**。量子围栏的受限态、吸附原子的能级、振动阈值,都是这一类读数。

判据本体在 :func:`mast.vision.spectral_peaks.find_peaks_1d`(纯函数、零 IO)。噪声尺度
取自谱自身的二阶差分,门限是 ``σ·√(2 ln n)`` —— 一条足够长的纯噪声里,总有某处会冒出
3σ,所以固定倍数的 σ 会在任何长谱里「找到」峰。
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

_NAME = "FindSpectralPeaks"
_DIDV_PATTERNS = (("li", "demod", "x"), ("lix",), ("demod", "x"), ("didv",), ("di/dv",))
_BIAS_PATTERNS = (("bias",), ("voltage",), ("v (v)",))
_CURRENT_PATTERNS = (("current",), ("i (a)",))


def _as_float(value):
    """A header value as a number, or None if it is not one."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _pick(columns: dict, patterns) -> str | None:
    for pat in patterns:
        for name in columns:
            low = name.lower()
            if all(tok in low for tok in pat):
                return name
    return None


class FindSpectralPeaks(BaseSkill):
    """找出一条谱里的峰,给出能量、宽度与显著度。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在一条已保存的 .dat 谱里找峰,报出每个峰的能量(eV 与 meV)、半高全宽和"
                "显著度。只读文件,不碰硬件。verdict 取 'peaks' / 'none' / 'undecidable'。"
                "噪声尺度取自这条谱自己;门限按谱长自动抬高,所以纯噪声不会被当成峰。"
                "没有 lock-in 通道时用电流的数值微分,并在 didv_source 里说明。"
            ),
            parameters=[
                ParameterSpec(name="dat_path", type="str",
                              description=".dat 谱文件路径。", required=True),
                ParameterSpec(
                    name="signal", type="str",
                    description="用哪一路:auto / didv(lock-in)/ current_derivative。",
                    required=False, default="auto",
                    allowed_values=["auto", "didv", "current_derivative"]),
                # 能量窗刻意没有 default:缺席就是「整条谱」。
                ParameterSpec(name="bias_min_v", type="float",
                              description="只在这个偏压以上找峰,单位**伏**(普通数字)。",
                              required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="bias_max_v", type="float",
                              description="只在这个偏压以下找峰,单位**伏**(普通数字)。",
                              required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="min_prominence_sigma", type="float",
                              description="显著度门限的下限,以噪声 σ 计。",
                              required=False, default=3.0, min_value=1.0, max_value=50.0),
                ParameterSpec(name="max_peaks", type="int",
                              description="最多报几个峰(按能量从低到高)。",
                              required=False, default=8, min_value=1, max_value=64),
                ParameterSpec(name="max_fwhm_v", type="float",
                              description="比这更宽的就是背景,不是能级,单位**伏**。",
                              required=False, default=0.30, min_value=0.001, max_value=10.0),
                ParameterSpec(name="polarity", type="str",
                              description="找极大(positive)还是极小(negative)。",
                              required=False, default="positive",
                              allowed_values=["positive", "negative"]),
                ParameterSpec(name="refine", type="str",
                              description="峰位细化方式。",
                              required=False, default="lorentzian",
                              allowed_values=["grid", "parabolic", "lorentzian"]),
            ],
            estimated_duration_s=4.0,
            composition_level=2,
            tags=["analysis", "spectroscopy", "sts", "didv", "peaks", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        path = str(params.get("dat_path") or "")
        if not path or not Path(path).exists():
            return SkillResult(skill_name=_NAME, success=False, error=f"文件不存在: {path}")
        try:
            import numpy as np

            from mast.io.nanonis_files import read_dat
            from mast.vision.spectral_peaks import find_peaks_1d
        except ImportError as exc:
            return SkillResult(skill_name=_NAME, success=False, error=f"缺依赖: {exc}")
        try:
            got = read_dat(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=_NAME, success=False, error=f".dat 读取失败: {exc}")
        cols = got.get("columns") or {}
        header = got.get("header") or {}
        bias_col = _pick(cols, _BIAS_PATTERNS)
        if bias_col is None:
            return SkillResult(skill_name=_NAME, success=False,
                               error=f"谱里没有偏压列(有的是 {sorted(cols)})")
        x = np.asarray(cols[bias_col], dtype=float)
        want = str(params.get("signal") or "auto")
        didv_col = None if want == "current_derivative" else _pick(cols, _DIDV_PATTERNS)
        if didv_col is not None:
            y = np.asarray(cols[didv_col], dtype=float)
            source = "lockin"
        else:
            cur_col = _pick(cols, _CURRENT_PATTERNS)
            if cur_col is None:
                return SkillResult(skill_name=_NAME, success=False,
                                   error="谱里既没有 lock-in 也没有电流列")
            y = np.gradient(np.asarray(cols[cur_col], dtype=float), x)
            didv_col, source = cur_col, "numeric"
        lo, hi = params.get("bias_min_v"), params.get("bias_max_v")
        window = np.ones_like(x, dtype=bool)
        if lo is not None:
            window &= x >= float(lo)
        if hi is not None:
            window &= x <= float(hi)
        if window.sum() < 15:
            return SkillResult(skill_name=_NAME, success=True,
                               data={"verdict": "undecidable", "reasons": ["window_too_small"],
                                     "n_points": int(window.sum()), "dat_path": path},
                               summary="能量窗里点太少,判不了")

        res = find_peaks_1d(
            x[window], y[window],
            prominence_sigma=float(params.get("min_prominence_sigma") or 3.0),
            max_peaks=int(params.get("max_peaks") or 8),
            max_fwhm_ev=float(params.get("max_fwhm_v") or 0.30),
            polarity=str(params.get("polarity") or "positive"),
            refine=str(params.get("refine") or "lorentzian"))

        peaks = [{"energy_ev": p.energy_ev, "energy_mev": p.energy_ev * 1e3,
                  "energy_err_ev": p.energy_err_ev, "fwhm_ev": p.fwhm_ev,
                  "amplitude": p.amplitude, "prominence_sigma": p.prominence_sigma,
                  "fit": p.fit, "fit_r2": p.fit_r2} for p in res.peaks]
        data = {"verdict": res.verdict, "n_peaks": len(peaks), "peaks": peaks,
                "energies_mev": [p["energy_mev"] for p in peaks],
                "noise_sigma": res.noise_sigma, "bias_range_v": res.energy_range_ev,
                "n_points": res.n_points, "didv_column": didv_col, "didv_source": source,
                "bias_column": bias_col, "reasons": list(res.reasons),
                "warnings": list(res.warnings), "dat_path": path,
                # numbers, not the raw header strings: a caller that does arithmetic with the
                # position (distance to a scatterer, say) would otherwise concatenate instead
                "dat_x_m": _as_float(header.get("X (m)")),
                "dat_y_m": _as_float(header.get("Y (m)"))}
        if peaks:
            listed = ", ".join(f"{p['energy_mev']:.0f} meV" for p in peaks[:4])
            summary = f"找到 {len(peaks)} 个峰:{listed}" + ("…" if len(peaks) > 4 else "")
        else:
            summary = f"没有超过门限的峰(σ = {res.noise_sigma:.2e})"
        return SkillResult(skill_name=_NAME, success=True, data=data, summary=summary)
