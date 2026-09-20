"""``InvertForceSaderJarvis`` —— 把一条 Δf(z) 变成力与势能。

qPlus 传感器记录的是频移,不是力。Sader 与 Jarvis(APL 84, 1801, 2004)给出的闭式反演把
两者连起来;Huber 等(Science 366, 235, 2019)就是用它看 CO 针尖与 Fe 原子成键的。

反演本体在 :mod:`mast.vision.force_inversion`(纯函数、零 IO)。这一层负责:找列、取
f0 / k / A、减背景曲线、把 F(z) 与 U(z) 落盘。

三个参数 **刻意没有 default**:``f0_hz`` / ``k_n_per_m`` / ``amplitude_m`` 缺席时要能触发
「从 .dat 头取 → 从仪器档案取 → 拒绝」这条回落链。弹性常数 k 不在任何 Nanonis 头里,它
来自针尖登记信息;拿不到就直接说,不猜。
"""

from __future__ import annotations

import json
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

_NAME = "InvertForceSaderJarvis"
#: 模拟器与真机语料用的两种写法都认
_DF_PATTERNS = (("freq", "shift"), ("df",), ("frequency", "shift"))
_Z_PATTERNS = (("z rel",), ("z (m)",), ("z spectr",))
_AMP_PATTERNS = (("amplitude",),)


def _pick(columns: dict, patterns) -> str | None:
    for pat in patterns:
        for name in columns:
            low = name.lower()
            if all(tok in low for tok in pat):
                return name
    return None


def _header_number(header: dict, *tokens: str) -> float | None:
    for key, val in (header or {}).items():
        low = str(key).lower()
        if all(t in low for t in tokens):
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


class InvertForceSaderJarvis(BaseSkill):
    """Δf(z) → F(z)、U(z),外加两条「这次反演能不能信」的读数。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "把一条 Δf(z) 的 .dat 用 Sader–Jarvis 方法反演成力 F(z) 与势能 U(z),"
                "报出力的极小值(pN)、它相对 Δf 极小值的位置、短程力衰减长度与结合能"
                "(meV)。给了干净表面的背景曲线就先相减。只读文件,不碰硬件。"
                "verdict 取 'well' / 'no_well' / 'undecidable'。另外报两条诊断:正向残差"
                "(反演出的力再算回 Δf 与实测差多少),以及振幅与力衰减长度之比 —— 后者"
                "的门限**未标定**,它提示换个振幅复测,不替你下结论。"
            ),
            parameters=[
                ParameterSpec(name="dat_path", type="str",
                              description="Δf(z) 曲线的 .dat 路径。", required=True),
                ParameterSpec(name="background_dat_path", type="str",
                              description="干净表面上同样扫程的 Δf(z),用来扣长程背景。",
                              required=False, default=""),
                # 传感器参数刻意没有 default:缺席要能触发回落链。
                ParameterSpec(name="f0_hz", type="float",
                              description=("传感器共振频率,单位**赫兹**(普通数字,例如 "
                                           "30000)。留空则从 .dat 头或仪器档案取。"),
                              required=False, min_value=1.0, max_value=1e7),
                ParameterSpec(name="k_n_per_m", type="float",
                              description=("传感器弹性常数,单位 **N/m**(普通数字,qPlus "
                                           "常见 1800)。任何 Nanonis 头里都没有这个数,"
                                           "留空则从仪器档案/针尖登记信息取。"),
                              required=False, min_value=1.0, max_value=1e6),
                ParameterSpec(name="amplitude_m", type="float", unit="m",
                              description=("振荡振幅,例如 '50p'(SI 前缀必须写)。留空则"
                                           "从 .dat 的振幅列或头里取。"),
                              required=False, min_value=1e-13, max_value=1e-8),
                ParameterSpec(name="direction", type="str",
                              description="用正扫、反扫、还是两者平均。",
                              required=False, default="auto",
                              allowed_values=["auto", "forward", "backward", "average"]),
                ParameterSpec(name="smooth_points", type="int",
                              description="反演前对 Δf 做几点平滑(0 = 不平滑)。",
                              required=False, default=0, min_value=0, max_value=99),
                ParameterSpec(name="df_column", type="str",
                              description="频移列的列名(留空自动找)。",
                              required=False, default=""),
                ParameterSpec(name="z_column", type="str",
                              description="z 列的列名(留空自动找)。",
                              required=False, default=""),
            ],
            estimated_duration_s=6.0,
            composition_level=2,
            tags=["analysis", "pll", "frequency", "force", "qplus", "afm", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        path = str(params.get("dat_path") or "")
        if not path or not Path(path).exists():
            return SkillResult(skill_name=_NAME, success=False, error=f"文件不存在: {path}")
        try:
            import numpy as np

            from mast.io.nanonis_files import read_dat
            from mast.vision.force_inversion import invert_force_curve
        except ImportError as exc:
            return SkillResult(skill_name=_NAME, success=False, error=f"缺依赖: {exc}")
        try:
            got = read_dat(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=_NAME, success=False, error=f".dat 读取失败: {exc}")

        cols = got.get("columns") or {}
        header = got.get("header") or {}
        z_col = params.get("z_column") or _pick(cols, _Z_PATTERNS)
        df_col = params.get("df_column") or _pick(cols, _DF_PATTERNS)
        if z_col not in cols or df_col not in cols:
            return SkillResult(skill_name=_NAME, success=False,
                               error=f"谱里没有 z 或频移列(有的是 {sorted(cols)})")
        z = np.asarray(cols[z_col], dtype=float)
        df = np.asarray(cols[df_col], dtype=float)

        f0, f0_src = self._sensor(params, "f0_hz", header, context,
                                  header_tokens=("center", "freq"),
                                  profile_keys=("qplus_f0_measured_hz", "qplus_f0_hz"))
        k, k_src = self._sensor(params, "k_n_per_m", header, context,
                                header_tokens=("spring", "constant"),
                                profile_keys=("qplus_k_n_per_m",))
        amp, amp_src = self._amplitude(params, header, cols)
        missing = [n for n, v in (("f0_hz", f0), ("k_n_per_m", k), ("amplitude_m", amp))
                   if v is None]
        if missing:
            return SkillResult(
                skill_name=_NAME, success=False,
                error=(f"缺传感器参数:{', '.join(missing)}。"
                       "f0 与振幅通常在 .dat 的 Oscillation Control 头里;"
                       "弹性常数 k 不在任何头里,要从针尖登记信息拿(qPlus 常见 1800 N/m)。"))

        bg = None
        bg_path = str(params.get("background_dat_path") or "")
        if bg_path:
            if not Path(bg_path).exists():
                return SkillResult(skill_name=_NAME, success=False,
                                   error=f"背景文件不存在: {bg_path}")
            try:
                bg_got = read_dat(bg_path)
                bg_cols = bg_got.get("columns") or {}
                bz = np.asarray(bg_cols[_pick(bg_cols, _Z_PATTERNS)], dtype=float)
                bdf = np.asarray(bg_cols[_pick(bg_cols, _DF_PATTERNS)], dtype=float)
                order = np.argsort(bz)
                bg = np.interp(np.sort(z), bz[order], bdf[order])
            except Exception as exc:  # noqa: BLE001
                return SkillResult(skill_name=_NAME, success=False,
                                   error=f"背景曲线读取失败: {exc}")

        res = invert_force_curve(z, df, f0_hz=f0, k_n_per_m=k, amplitude_m=amp,
                                 background_df_hz=bg,
                                 smooth_points=int(params.get("smooth_points") or 0))
        curve_path = self._save_curve(res, path)
        data = {"verdict": res.verdict,
                "f_min_pn": res.f_min_pn, "f_min_n": res.f_min_n,
                "z_f_min_m": res.z_f_min_m, "z_df_min_m": res.z_df_min_m,
                "z_offset_fmin_minus_dfmin_pm": (None if res.z_offset_fmin_minus_dfmin_m is None
                                                 else res.z_offset_fmin_minus_dfmin_m * 1e12),
                "e_bind_mev": res.e_bind_mev, "e_bind_ev": res.e_bind_ev,
                "decay_length_pm": res.decay_length_pm,
                "forward_residual": res.forward_residual,
                "amplitude_over_decay_length": res.amplitude_over_decay_length,
                "well_posedness": res.well_posedness,
                "background_used": res.background_used,
                "f0_hz": f0, "f0_source": f0_src, "k_n_per_m": k, "k_source": k_src,
                "amplitude_m": amp, "amplitude_source": amp_src,
                "n_points": res.n_points, "curve_path": curve_path,
                "df_column": df_col, "z_column": z_col,
                "reasons": list(res.reasons), "warnings": list(res.warnings),
                "dat_path": path, "background_dat_path": bg_path or None}
        if res.f_min_pn is not None:
            summary = (f"F_min = {res.f_min_pn:.1f} pN,E_b = {res.e_bind_mev:.0f} meV,"
                       f"衰减长度 {(res.decay_length_pm or 0):.0f} pm"
                       f"(正向残差 {res.forward_residual:.3f},{res.well_posedness})")
        else:
            summary = f"反演判定 {res.verdict}:{'、'.join(res.reasons) or '无'}"
        return SkillResult(skill_name=_NAME, success=True, data=data, summary=summary)

    # ── helpers ──
    @staticmethod
    def _sensor(params: dict, name: str, header: dict, context, *, header_tokens,
                profile_keys) -> tuple[float | None, str]:
        given = params.get(name)
        if given is not None:
            return float(given), "param"
        val = _header_number(header, *header_tokens)
        if val:
            return val, "dat_header"
        try:
            from mast.core.instrument_profile import get_config

            for key in profile_keys:
                v = get_config(key)
                if v:
                    return float(v), f"instrument_profile:{key}"
        except Exception:  # noqa: BLE001 — no profile is a reason to say so, not to raise
            pass
        return None, "none"

    @staticmethod
    def _amplitude(params: dict, header: dict, cols: dict) -> tuple[float | None, str]:
        import numpy as np

        given = params.get("amplitude_m")
        if given is not None:
            return float(given), "param"
        val = _header_number(header, "amplitude", "setpoint")
        if val:
            return val, "dat_header"
        col = _pick(cols, _AMP_PATTERNS)
        if col is not None:
            arr = np.asarray(cols[col], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size and 1e-13 < float(np.median(arr)) < 1e-8:
                return float(np.median(arr)), "dat_column"
        return None, "none"

    @staticmethod
    def _save_curve(res, dat_path: str) -> str | None:
        from pathlib import Path

        try:
            from mast.core._runtime_paths import project_root

            out = Path(project_root()) / "artifacts" / "force_inversion"
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{Path(dat_path).stem}_force.json"
            path.write_text(json.dumps(
                {"z_m": list(res.z_m), "force_n": list(res.force_n),
                 "energy_ev": list(res.energy_ev), "verdict": res.verdict,
                 "f_min_n": res.f_min_n, "e_bind_ev": res.e_bind_ev},
                ensure_ascii=False), encoding="utf-8")
            return str(path)
        except Exception:  # noqa: BLE001
            return None
