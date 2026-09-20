"""Import an external noise-sweep manifest as a baseline. Each point names a directory containing summary.json and width.json, with optional psd.npz and hist.npz. The manifest supplies label, note, conditions and a points list with dir and sweep keys. All points use baseline.build_models, sharing the fitting implementation with CharacteriseCurrentNoise."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Line windows tracked on import — kept identical to the acquisition skill's
#: ``_LINES`` so an imported baseline and a measured one carry the same keys.
#: Divergence here would show up as a baseline whose lines cannot be compared
#: with the next one, which is the whole point of storing them.
_LINES: tuple[tuple[str, float, float], ...] = (
    ("vib_6hz", 4.0, 8.0), ("l_29hz", 28.0, 31.0), ("l_50hz", 48.5, 51.5),
    ("l_87hz", 85.0, 90.0), ("l_100hz", 98.5, 101.5),
    ("l_200hz", 198.0, 202.0), ("l_450hz", 445.0, 455.0),
    ("l_788hz", 780.0, 795.0),
)


def _load_point(root: str, entry: dict) -> Optional[dict]:
    """One point's metrics + curves, or None with a stated reason."""
    import numpy as np

    from mast.monitoring import baseline as B

    d = os.path.join(root, str(entry.get("dir") or ""))
    sp = os.path.join(d, "summary.json")
    wp = os.path.join(d, "width.json")
    if not os.path.exists(sp):
        logger.warning("跳过 %s：没有 summary.json", d)
        return None
    s = json.load(open(sp, encoding="utf-8"))
    w = json.load(open(wp, encoding="utf-8")) if os.path.exists(wp) else {}
    det = (w.get("detrended") or {})
    st = s.get("state_before") or {}

    freqs = psd = None
    pp = os.path.join(d, "psd.npz")
    if os.path.exists(pp):
        z = np.load(pp)
        freqs, psd = z["freqs_hz"], z["psd_a2_per_hz"]

    lines: dict = {}
    i_mean = s.get("I_mean_a")
    if freqs is not None:
        for nm, lo, hi in _LINES:
            m = B.peak_metrics(freqs, psd, lo, hi, i_mean)
            if m:
                lines[nm] = m
    else:
        # Fall back to whatever the sweep's own analysis recorded. Keys are
        # normalised to this module's names so an imported baseline stays
        # comparable with a measured one.
        for nm, m in (s.get("peaks") or {}).items():
            if m:
                lines[nm] = m

    hist = None
    hp = os.path.join(d, "hist.npz")
    if os.path.exists(hp):
        z = np.load(hp)
        if z["det_counts"].size:
            hist = (z["det_edges"], z["det_counts"])

    metrics = {
        "bias_v": st.get("bias_v"), "setpoint_a": st.get("setpoint_a"),
        "i_measured_a": i_mean, "z_m": st.get("z_pos_m"),
        "n_segments": s.get("n_segments") or 0,
        "sigma_a": det.get("sigma_a") or s.get("rms_detrended_a"),
        "sigma_iqr_a": det.get("sigma_from_iqr_a"),
        "sigma_mad_a": det.get("sigma_from_mad_a"),
        "fwhm_a": det.get("fwhm_a"), "fwhm_over_sigma": det.get("fwhm_over_sigma"),
        "iqr_a": det.get("iqr_a"), "p99_p1_a": det.get("p99_p1_a"),
        "ptp_a": det.get("ptp_a"),
        "kurtosis": det.get("kurtosis"), "skewness": det.get("skewness"),
        "mean_a": (w.get("raw") or {}).get("mean_a") or i_mean,
        "white_a2hz": s.get("white_floor_a2_per_hz"),
    }
    return {"tag": entry.get("sweep") or "", "sweep": entry.get("sweep") or "",
            "dir": entry.get("dir"), "metrics": metrics, "lines": lines,
            "psd": (freqs, psd) if freqs is not None else None, "hist": hist,
            "i_measured_a": i_mean, "sigma_a": metrics["sigma_a"],
            "white_a2hz": metrics["white_a2hz"], "bias_v": st.get("bias_v")}


def import_sweep(root: str, manifest: dict, *, activate: bool = False,
                 store=None) -> dict:
    """Build a baseline from an already-measured sweep. Returns a report dict."""
    from mast.monitoring import baseline as B
    from mast.monitoring.store import get_store

    store = store or get_store()
    if store is None:
        return {"ok": False, "error": "拿不到 monitoring store"}

    pts = [p for p in (_load_point(root, e) for e in manifest.get("points") or [])
           if p]
    if not pts:
        return {"ok": False, "error": "manifest 里没有一个点能读出来"}

    bid = store.create_baseline(
        label=str(manifest.get("label") or ""),
        note=str(manifest.get("note") or ""),
        conditions=dict(manifest.get("conditions") or {}),
        fs_hz=(manifest.get("conditions") or {}).get("fs_hz"))
    if not bid:
        return {"ok": False, "error": "基线行建不出来"}

    for i, p in enumerate(pts):
        store.add_baseline_point(bid, tag=p["tag"], ordinal=i,
                                 metrics=p["metrics"], psd=p["psd"],
                                 hist=p["hist"],
                                 extra={"lines": p["lines"], "dir": p["dir"],
                                        "imported": True})

    models = B.build_models(pts)
    sm, wm = models["sigma_model"], models["white_model"]
    status = "complete" if sm is not None else "aborted"
    store.finish_baseline(bid, status=status,
                          sigma_model=sm.to_dict() if sm else None,
                          white_model=wm.to_dict() if wm else None,
                          lines=models["lines"],
                          repeatability=models["repeatability"])
    activated = False
    if activate and status == "complete":
        activated = store.activate_baseline(bid)

    return {"ok": status == "complete", "baseline_id": bid, "status": status,
            "n_points": len(pts), "n_setpoint": models["n_setpoint"],
            "n_bias": models["n_bias"], "activated": activated,
            "sigma_model": sm.to_dict() if sm else None,
            "white_model": wm.to_dict() if wm else None,
            "lines": models["lines"], "repeatability": models["repeatability"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="含各工况点子目录的目录")
    ap.add_argument("--manifest", help="manifest JSON；缺省用 <root>/manifest.json")
    ap.add_argument("--activate", action="store_true",
                    help="导入后立刻启用为实时判据的基线")
    ap.add_argument("--json", action="store_true", help="只打印 JSON")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mpath = a.manifest or os.path.join(a.root, "manifest.json")
    if not os.path.exists(mpath):
        print("找不到 manifest: %s" % mpath, file=sys.stderr)
        return 2
    rep = import_sweep(a.root, json.load(open(mpath, encoding="utf-8")),
                       activate=a.activate)
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0 if rep.get("ok") else 1

    if not rep.get("ok"):
        print("导入未完成：%s" % (rep.get("error") or rep.get("status")))
        return 1
    sm, wm = rep["sigma_model"], rep["white_model"]
    print("基线 #%d 已建立（%d 个点：setpoint %d / bias %d）%s"
          % (rep["baseline_id"], rep["n_points"], rep["n_setpoint"],
             rep["n_bias"], "，已启用" if rep["activated"] else ""))
    if sm:
        print("  sigma 曲线  加性 %.3f pA   相对底 %.0f ppm   交叉 %.1f pA   R²=%.5f"
              % (sm["additive_sigma_a"] * 1e12, sm["relative_floor"] * 1e6,
                 (sm["crossover_a"] or 0) * 1e12, sm["r2"]))
    if wm:
        print("  白底分解    前放 %.2f fA/√Hz   乘性 %.1f ppm/√Hz   交叉 %.0f pA"
              % (wm["amp_a_per_rthz"] * 1e15, (wm["b"] ** 0.5) * 1e6,
                 (wm["crossover_a"] or 0) * 1e12))
    rp = rep.get("repeatability")
    if rp:
        print("  重复性      %d 次，相对散布 %.1f%%"
              % (rp["n"], (rp.get("relative_sd") or 0) * 100))
    lines = rep.get("lines") or {}
    if lines:
        print("  谱线：")
        for nm, l in sorted(lines.items(),
                            key=lambda kv: -(kv[1].get("a_per_rthz") or 0)):
            print("    %-10s %7.1f Hz  %8.1f fA/√Hz  a_I=%s a_V=%s  %s"
                  % (nm, l.get("f_peak_hz") or 0,
                     (l.get("a_per_rthz") or 0) * 1e15,
                     ("%+.2f" % l["a_i"]) if l.get("a_i") is not None else "  — ",
                     ("%+.2f" % l["a_v"]) if l.get("a_v") is not None else "  — ",
                     l.get("mechanism") or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
