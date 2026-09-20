"""``FitDispersion`` —— 从一组驻波谱里读出表面态的带底与有效质量。

给一串离台阶(或吸附原子)不同距离的 dI/dV 谱,量出每个能量下的驻波波矢 k(E),再拟合
``E = E0 + ħ²k²/2m*``。这是 Hasegawa–Avouris(PRL 71, 1071)与 Crommie–Lutz–Eigler
(Nature 363, 524)1993 年做的那件事。

每条谱的位置从它自己的 ``.dat`` 头 ``X (m)`` / ``Y (m)`` 读,所以调用方只需要给散射体的
几何(台阶的一点加走向,或点散射体的坐标),不需要再抄一遍坐标。

判据本体在 :mod:`mast.vision.standing_wave`(纯函数、零 IO)。k(E) 全表落盘到
``artifacts/``,只把路径与前几行放进返回值——工具返回有 2000 字符上限。
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

_NAME = "FitDispersion"
_DIDV_PATTERNS = (("li", "demod", "x"), ("lix",), ("demod", "x"), ("didv",), ("di/dv",))
_BIAS_PATTERNS = (("bias",), ("voltage",), ("v (v)",))
_CURRENT_PATTERNS = (("current",), ("i (a)",))


def _pick(columns: dict, patterns) -> str | None:
    for pat in patterns:
        for name in columns:
            low = name.lower()
            if all(tok in low for tok in pat):
                return name
    return None


class FitDispersion(BaseSkill):
    """表面态色散:从驻波谱拟合 E0 与 m*。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从一组离台阶或吸附原子不同距离的 dI/dV 谱里,量出各能量的驻波波矢,"
                "再拟合抛物线色散,给出表面态带底 E0(eV / meV)与有效质量 m*(以电子"
                "质量计)。只读文件,不碰硬件。每条谱的位置从它的 .dat 头 X (m)/Y (m) 读。"
                "verdict 取 'dispersion' / 'no_standing_wave' / 'undecidable'。"
                "至少要四个能量能量出波矢,距离跨度至少三纳米,否则判不了。"
            ),
            parameters=[
                ParameterSpec(
                    name="dat_paths", type="str",
                    description=("谱文件路径列表,JSON 数组,例如 "
                                 '["a.dat", "b.dat"]。与 dat_dir 至少给一个。'),
                    required=False, default=""),
                ParameterSpec(name="dat_dir", type="str",
                              description="谱文件所在目录(配合 run_tag 过滤)。",
                              required=False, default=""),
                ParameterSpec(name="run_tag", type="str",
                              description="在 dat_dir 里只取 <run_tag>_p*.dat。",
                              required=False, default=""),
                ParameterSpec(name="scatterer_kind", type="str",
                              description="驻波是台阶(step)还是点散射体(point)造成的。",
                              required=True, allowed_values=["step", "point"]),
                ParameterSpec(name="edge_x_m", type="float", unit="m",
                              description="台阶上任一点的 x,例如 '12n'(SI 前缀必须写)。",
                              required=False, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(name="edge_y_m", type="float", unit="m",
                              description="台阶上任一点的 y,例如 '-3n'。",
                              required=False, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(name="edge_angle_deg", type="float",
                              description="台阶走向,单位**度**(普通数字),+x 方向为 0。",
                              required=False, min_value=-360.0, max_value=360.0),
                ParameterSpec(name="distances_nm", type="str",
                              description=(
                                  "每条谱到散射体的距离,JSON 数字数组,单位**纳米**,顺序与 "
                                  "dat_paths 一一对应。给了它就不再从散射体几何算距离 —— "
                                  "长时间序列里样品会漂,自己按批次跟踪几何比用一条固定的边准。"),
                              required=False, default=""),
                ParameterSpec(name="scatterer_x_m", type="float", unit="m",
                              description="点散射体的 x,例如 '5n'。",
                              required=False, min_value=-1.5e-6, max_value=1.5e-6),
                ParameterSpec(name="scatterer_y_m", type="float", unit="m",
                              description="点散射体的 y,例如 '5n'。",
                              required=False, min_value=-1.5e-6, max_value=1.5e-6),
                # 以下刻意没有 default:缺席要能触发衬底查表 / 默认值那一行。
                ParameterSpec(name="min_distance_nm", type="float",
                              description=("离散射体多近的谱不要,单位**纳米**(普通数字)。"
                                           "留空取 1。近场不服从远场模型。"),
                              required=False, min_value=0.1, max_value=50.0),
                ParameterSpec(name="energy_min_v", type="float",
                              description="只用这个偏压以上的能量,单位**伏**。",
                              required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="energy_max_v", type="float",
                              description="只用这个偏压以下的能量,单位**伏**。",
                              required=False, min_value=-10.0, max_value=10.0),
                ParameterSpec(name="normalize", type="bool",
                              description="先除以最远那条谱,去掉针尖与表面自身的谱形。",
                              required=False, default=True),
                ParameterSpec(name="min_r2", type="float",
                              description="k² 对 E 这条直线至少要这么好。",
                              required=False, default=0.80, min_value=0.0, max_value=1.0),
            ],
            estimated_duration_s=10.0,
            composition_level=2,
            tags=["analysis", "spectroscopy", "sts", "didv", "dispersion", "surface_state", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        try:
            import numpy as np

            from mast.io.nanonis_files import read_dat
            from mast.vision.standing_wave import (distance_to_line, distance_to_point,
                                                   fit_dispersion)
        except ImportError as exc:
            return SkillResult(skill_name=_NAME, success=False, error=f"缺依赖: {exc}")

        paths = self._paths(params)
        if not paths:
            return SkillResult(skill_name=_NAME, success=False,
                               error="没有谱文件:给 dat_paths(JSON 数组)或 dat_dir[+run_tag]")
        kind = str(params.get("scatterer_kind"))
        given = params.get("distances_nm")
        given_d = None
        if given:
            try:
                given_d = [float(v) for v in json.loads(given)]
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return SkillResult(skill_name=_NAME, success=False,
                                   error=f"distances_nm 不是一个数字数组: {exc}")
            if len(given_d) != len(paths):
                return SkillResult(
                    skill_name=_NAME, success=False,
                    error=(f"distances_nm 有 {len(given_d)} 个,dat_paths 有 {len(paths)} 个 —— "
                           "两者按位置一一对应,长度对不上就没法说哪条谱在哪儿"))
        if given_d is None:
            need = ("edge_x_m", "edge_y_m", "edge_angle_deg") if kind == "step"                 else ("scatterer_x_m", "scatterer_y_m")
            missing = [k for k in need if params.get(k) is None]
            if missing:
                return SkillResult(skill_name=_NAME, success=False,
                                   error=f"{kind} 需要 {', '.join(missing)}(或者直接给 distances_nm)")

        rows, warns, skipped = [], [], []
        for i, p in enumerate(paths):
            got = self._read_one(read_dat, p)
            if got is None:
                skipped.append(Path(p).name)
                continue
            if given_d is not None:
                got = {**got, "given_d_nm": given_d[i]}
            rows.append(got)
        if len(rows) < 4:
            return SkillResult(skill_name=_NAME, success=True,
                               data={"verdict": "undecidable", "n_spectra": len(rows),
                                     "reasons": ["too_few_readable_spectra"],
                                     "skipped": skipped, "dat_paths_used": []},
                               summary=f"只有 {len(rows)} 条谱可用,拟合不了色散")
        if skipped:
            warns.append("some_spectra_unreadable")

        xs = np.array([r["x_m"] for r in rows])
        ys = np.array([r["y_m"] for r in rows])
        if given_d is not None:
            # the caller tracked the geometry itself. That is the honest thing to do when the
            # sample drifts under a long series: the scatterer is somewhere else by the last
            # spectrum than it was at the first, and a single edge fitted once cannot know it.
            d = np.abs(np.array([r["given_d_nm"] for r in rows], dtype=float))
        elif kind == "step":
            d = distance_to_line(xs, ys, edge_x_m=float(params["edge_x_m"]),
                                 edge_y_m=float(params["edge_y_m"]),
                                 edge_angle_deg=float(params["edge_angle_deg"]))
            side = np.sign(d)
            majority = 1.0 if (side >= 0).sum() >= (side < 0).sum() else -1.0
            keep = side == majority
            if (~keep).any():
                warns.append("mixed_sides")
            rows = [r for r, k in zip(rows, keep) if k]
            d = np.abs(d[keep])
        else:
            d = distance_to_point(xs, ys, scatterer_x_m=float(params["scatterer_x_m"]),
                                  scatterer_y_m=float(params["scatterer_y_m"]))
        if len(rows) < 4:
            return SkillResult(skill_name=_NAME, success=True,
                               data={"verdict": "undecidable", "n_spectra": len(rows),
                                     "reasons": ["too_few_on_one_side"], "warnings": warns},
                               summary="同一侧的谱不够,拟合不了色散")

        grid = self._energy_grid(rows, params)
        if grid is None or grid.size < 4:
            return SkillResult(skill_name=_NAME, success=True,
                               data={"verdict": "undecidable", "reasons": ["no_common_energies"],
                                     "n_spectra": len(rows)},
                               summary="这些谱没有足够的公共能量区间")
        order = np.argsort(d)
        d = d[order]
        rows = [rows[i] for i in order]
        g = np.vstack([np.interp(grid, r["v"], r["didv"]) for r in rows]).T

        res = fit_dispersion(grid, d, g, scatterer_kind=kind,
                             min_distance_nm=float(params.get("min_distance_nm") or 1.0),
                             normalize=bool(params.get("normalize", True)),
                             min_r2=float(params.get("min_r2") or 0.80))

        table_path = self._save_table(context, res, rows)
        data = {"verdict": res.verdict, "e0_ev": res.e0_ev, "e0_mev": res.e0_mev,
                "e0_err_ev": res.e0_err_ev, "m_eff": res.m_eff, "m_eff_err": res.m_eff_err,
                "r2": res.r2, "n_spectra": res.n_spectra,
                "n_energies_used": res.n_energies_used,
                "distance_range_nm": res.distance_range_nm, "scatterer_kind": kind,
                "didv_source": rows[0]["source"], "k_table_path": table_path,
                "k_table_preview": [[round(a, 4), round(b, 4)] for a, b, _ in res.k_table[:5]],
                "reasons": list(res.reasons), "warnings": warns + list(res.warnings),
                "dat_paths_used": [r["path"] for r in rows], "skipped": skipped}
        if res.e0_mev is not None and res.m_eff is not None:
            summary = (f"E0 = {res.e0_mev:.0f} meV,m* = {res.m_eff:.3f} mₑ"
                       f"(r² = {res.r2:.3f},{res.n_energies_used} 个能量,"
                       f"{res.n_spectra} 条谱)")
        else:
            summary = f"色散判定 {res.verdict}:{'、'.join(res.reasons) or '无'}"
        return SkillResult(skill_name=_NAME, success=True, data=data, summary=summary)

    # ── helpers ──
    @staticmethod
    def _paths(params: dict) -> list[str]:
        from pathlib import Path

        raw = str(params.get("dat_paths") or "").strip()
        out: list[str] = []
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    out = [str(p) for p in parsed]
                elif isinstance(parsed, str):
                    out = [parsed]
            except (ValueError, TypeError):
                out = [p.strip() for p in raw.split(",") if p.strip()]
        d = str(params.get("dat_dir") or "").strip()
        if d and Path(d).is_dir():
            tag = str(params.get("run_tag") or "").strip()
            pattern = f"{tag}_p*.dat" if tag else "*.dat"
            out += [str(p) for p in sorted(Path(d).glob(pattern))]
        return [p for p in dict.fromkeys(out) if Path(p).exists()]

    @staticmethod
    def _read_one(read_dat, path: str) -> dict | None:
        import numpy as np

        try:
            got = read_dat(path)
        except Exception:  # noqa: BLE001 — an unreadable file is skipped and counted
            return None
        cols = got.get("columns") or {}
        header = got.get("header") or {}
        bias = _pick(cols, _BIAS_PATTERNS)
        if bias is None:
            return None
        didv = _pick(cols, _DIDV_PATTERNS)
        if didv is not None:
            y = np.asarray(cols[didv], dtype=float)
            source = "lockin"
        else:
            cur = _pick(cols, _CURRENT_PATTERNS)
            if cur is None:
                return None
            y = np.gradient(np.asarray(cols[cur], dtype=float),
                            np.asarray(cols[bias], dtype=float))
            source = "numeric"
        try:
            x_m = float(header.get("X (m)", header.get("x (m)")))
            y_m = float(header.get("Y (m)", header.get("y (m)")))
        except (TypeError, ValueError):
            return None
        if abs(x_m) > 1e-3 or abs(y_m) > 1e-3:
            return None
        v = np.asarray(cols[bias], dtype=float)
        order = np.argsort(v)
        return {"path": path, "v": v[order], "didv": y[order], "x_m": x_m, "y_m": y_m,
                "source": source}

    @staticmethod
    def _energy_grid(rows: list[dict], params: dict):
        import numpy as np

        lo = max(float(r["v"][0]) for r in rows)
        hi = min(float(r["v"][-1]) for r in rows)
        if params.get("energy_min_v") is not None:
            lo = max(lo, float(params["energy_min_v"]))
        if params.get("energy_max_v") is not None:
            hi = min(hi, float(params["energy_max_v"]))
        if not (hi > lo):
            return None
        n = int(np.clip(min(len(r["v"]) for r in rows) // 8, 8, 40))
        return np.linspace(lo, hi, n)

    @staticmethod
    def _save_table(context, res, rows) -> str | None:
        from pathlib import Path

        try:
            from mast.core._runtime_paths import project_root

            out = Path(project_root()) / "artifacts" / "dispersion"
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(rows[0]["path"]).stem
            path = out / f"{stem}_k_table.json"
            path.write_text(json.dumps(
                {"k_table": [list(t) for t in res.k_table],
                 "e0_ev": res.e0_ev, "m_eff": res.m_eff, "r2": res.r2,
                 "dat_paths": [r["path"] for r in rows]},
                ensure_ascii=False, indent=1), encoding="utf-8")
            return str(path)
        except Exception:  # noqa: BLE001 — the numbers are the product; the table is a courtesy
            return None
