"""Render a TipShapeWithReadback result as a z/current dual-curve figure.

Two stacked axes (Z on top, tunnelling current below) vs the capture clock, with
the TipShaper stage bands shaded (so the curves align to switch-off / plunge /
retract / end-wait), jump markers, the z1/z3 baselines, and a verdict box.

POST-skill rendering: the DANGEROUS TipShapeWithReadback skill must NOT import
matplotlib or touch disk mid-procedure, so this lives in mast/data/ (shared, not
per-agent) and is called from the post-skill hook with the skill's
``SkillResult.data``. Returns the saved PNG path. Pure read of JSON-safe data;
forces the Agg backend so it works headless.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # CJK font so Chinese labels render (not tofu boxes). Pick the first installed
    # one; on Windows YaHei/SimHei ship with the OS. Falls back silently.
    try:
        from matplotlib import font_manager
        avail = {fp.name for fp in font_manager.fontManager.ttflist}
        for f in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                  "Source Han Sans SC", "Microsoft JhengHei"):
            if f in avail:
                plt.rcParams["font.sans-serif"] = [f] + list(
                    plt.rcParams.get("font.sans-serif", []))
                break
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# TipShaper stage → (label, color)
_STAGE_STYLE = {
    "pre_roll": ("基线", "#94a3b8"),
    "switch_off": ("关反馈", "#64748b"),
    "z_ramp_1_plunge": ("下扎", "#dc2626"),
    "bias_settle": ("bias", "#a855f7"),
    "z_ramp_2_retract": ("回抬", "#2563eb"),
    "end_wait": ("等待", "#64748b"),
    "post_roll": ("恢复", "#16a34a"),
}

_VERDICT_CN = {
    "no_change": "没扎上",
    "cluster": "扎上了 (cluster)",
    "tip_changed_or_pit": "针尖改变/坑",
    "insufficient_data": "数据不足",
}


def _detect_plunge(zt, zs_nm, z1_nm, sigma_nm):
    """Detect the actual plunge window from the z trace (nm): plunge_start (z
    first drops well below baseline), the deepest point, retract_end (z returns).
    More reliable than the param-estimated times. None if no clear plunge."""
    if len(zs_nm) < 4 or not zt:
        return None
    zmin = min(zs_nm)
    drop = max(6.0 * sigma_nm, abs(zmin - z1_nm) * 0.2, 1e-4)
    below = [t for t, z in zip(zt, zs_nm) if z < z1_nm - drop]
    if not below:
        return None
    bt = [t for t, z in zip(zt, zs_nm) if z <= zmin + drop * 0.3]
    return {"plunge_start": min(below), "retract_end": max(below),
            "z_min_t": (sum(bt) / len(bt)) if bt else None}


def render_tip_shape_readback(data: dict[str, Any], save_path: str | Path) -> str:
    """Render *data* (a TipShapeWithReadback SkillResult.data) to *save_path* PNG.

    Returns the path as a str. Never raises on a slightly-malformed data dict —
    missing optional sections (stages/jumps/indent) are simply skipped.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt = _plt()
    z = data.get("z", {}) or {}
    cur = data.get("current", {}) or {}
    zt = list(z.get("t_s", []) or [])
    zs = [v * 1e9 for v in (z.get("samples_m", []) or [])]      # m → nm
    it = list(cur.get("t_s", []) or [])
    isamp = [v * 1e9 for v in (cur.get("samples_a", []) or [])]  # A → nA

    fig, (ax_z, ax_i) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 6.0))

    if zt:
        ax_z.plot(zt, zs, color="#0d9488", lw=1.2)
    ax_z.set_ylabel("Z (nm)")
    ax_z.grid(True, alpha=0.2)
    if it:
        ax_i.plot(it, isamp, color="#d97706", lw=1.2)
    ax_i.set_ylabel("电流 I (nA)")
    ax_i.set_xlabel("时间 t (s)")
    ax_i.grid(True, alpha=0.2)

    # full time extent for an open-ended final stage band
    t_end_all = max((zt[-1] if zt else 0.0), (it[-1] if it else 0.0))

    # The param-estimated stage times lag the firmware by a fixed overhead; anchor
    # them to the MEASURED plunge (z first drops below baseline) so the bands line
    # up with the actual curve.
    ind = data.get("indent", {}) or {}
    z1_nm = ind.get("z1_m", 0.0) * 1e9
    sigma_nm = ind.get("baseline_sigma_m", 0.0) * 1e9
    detected = _detect_plunge(zt, zs, z1_nm, sigma_nm)
    stages = data.get("stages", []) or []
    offset = 0.0
    if detected and stages:
        est_plunge = next((s.get("t_start") for s in stages
                           if s.get("stage") == "z_ramp_1_plunge"), None)
        if est_plunge is not None:
            offset = detected["plunge_start"] - est_plunge

    # ── stage bands (axvspan, offset-aligned) + labels on the Z axis ──
    for s in stages:
        t0 = s.get("t_start")
        t1 = s.get("t_end")
        if t1 is None:
            t1 = t_end_all
        if t0 is None:
            continue
        t0 += offset
        t1 += offset
        label, color = _STAGE_STYLE.get(s.get("stage", ""), (s.get("stage", ""), "#cbd5e1"))
        for ax in (ax_z, ax_i):
            ax.axvspan(t0, t1, color=color, alpha=0.10, lw=0)
        if t1 > t0:
            ax_z.text((t0 + t1) / 2.0, 0.98, label, transform=_blend(ax_z, plt),
                      ha="center", va="top", fontsize=7.5, color=color)

    # measured-plunge markers (solid) so estimated vs real is unambiguous
    if detected:
        for ax in (ax_z, ax_i):
            ax.axvline(detected["plunge_start"], color="#dc2626", lw=1.0, alpha=0.7)
            ax.axvline(detected["retract_end"], color="#2563eb", lw=1.0, alpha=0.7)

    # ── z1 / z3 baselines + jump markers ──
    ind = data.get("indent", {}) or {}
    if "z1_m" in ind:
        ax_z.axhline(ind["z1_m"] * 1e9, color="#475569", ls="--", lw=0.8, alpha=0.6)
    if "z3_m" in ind:
        ax_z.axhline(ind["z3_m"] * 1e9, color="#16a34a", ls="--", lw=0.8, alpha=0.6)
    for j in (data.get("jumps", {}).get("z", {}) or {}).get("events", []) or []:
        ax_z.axvline(j.get("t_s", 0.0), color="#dc2626", ls=":", lw=0.7, alpha=0.4)
    for j in (data.get("jumps", {}).get("current", {}) or {}).get("events", []) or []:
        ax_i.axvline(j.get("t_s", 0.0), color="#dc2626", ls=":", lw=0.7, alpha=0.4)

    # ── verdict box ──
    if ind:
        v = ind.get("verdict", "")
        dz_nm = ind.get("delta_m", 0.0) * 1e9
        txt = f"{_VERDICT_CN.get(v, v)}   Δz = {dz_nm:+.3f} nm\n{ind.get('advice', '')}"
        color = {"cluster": "#16a34a", "no_change": "#d97706",
                 "tip_changed_or_pit": "#dc2626"}.get(v, "#334155")
        ax_z.text(0.015, 0.04, txt, transform=ax_z.transAxes, ha="left", va="bottom",
                  fontsize=8.5, color=color,
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.9))

    tm = data.get("timing", {}) or {}
    fig.suptitle(
        f"针尖整形 + current/z 实时读出  (n_z={tm.get('n_z', len(zs))}, "
        f"fs≈{tm.get('fs_z_hz', 0):.0f}Hz)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _blend(ax, plt):
    """A blended transform: x in data coords, y in axes coords (for stage labels)."""
    import matplotlib.transforms as mtransforms
    return mtransforms.blended_transform_factory(ax.transData, ax.transAxes)


__all__ = ["render_tip_shape_readback"]
