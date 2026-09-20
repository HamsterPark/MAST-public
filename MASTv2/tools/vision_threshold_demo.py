"""VIGIL v2.5 tip-quality threshold demo — visualise what a threshold set accepts.

Runs the DEPLOYED MAST v2.5 model (``MASTv2/artifacts/mast_vision_v25.pt``) over a
uniform sample of the VIGIL sf09 real-STM corpus, caches each image's 6 raw head
scores (ONE inference per image), then re-derives the good/bad + usable decision
for several threshold sets WITHOUT re-inferring — the pure functions in
:mod:`mast.vision.thresholds` are the exact ones the live model uses, so the
galleries reflect precisely how MAST would judge these tips at each setting.

Outputs (→ ``MASTv2/experiments/vision_threshold_demo/``):
  * ``head_montage_<H>.png`` — sample images sorted by head H's raw score with
    the current threshold's pass/fail colouring (H ∈ N/Q/K/T/S); shows where the
    cut sits and which tips are near the boundary.
  * ``flip_gallery.png``     — the tips whose good/bad label FLIPS when you loosen
    the thresholds (default → lowered): "these would now be accepted".
  * ``sweep.png``            — accept-rate vs the good/bad cut and vs multi-apex
    tolerance (helps pick a value).
  * ``summary.json``         — counts + the exact threshold sets compared.

Run from the repo root (needs h5py in the venv + the deployed checkpoint):
    set PYTHONPATH=MASTv2
    set MAST_VISION_BACKEND=vigil
    .venv-v2-py313\\Scripts\\python.exe MASTv2\\tools\\vision_threshold_demo.py --n 800

All matplotlib text is ASCII on purpose (the default font has no CJK glyphs →
missing-glyph boxes). Console logs are plain text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_H5 = r"<vigil-repo>\output\sf09_hdf5\sf09_real_512.h5"
DEFAULT_OUT = "MASTv2/experiments/vision_threshold_demo"
IMG_STD_MIN = 6.0  # skip degenerate near-constant frames (display-quality filter only)

GREEN = "#2e7d32"
RED = "#c62828"


def _mastv2_root() -> Path:
    """Locate the MASTv2 dir (this file lives at MASTv2/tools/)."""
    p = Path(__file__).resolve()
    while p.parent != p:
        if p.name == "MASTv2" and (p / "mast").is_dir():
            return p
        if (p / "MASTv2" / "mast").is_dir():
            return p / "MASTv2"
        p = p.parent
    raise RuntimeError("MASTv2 root not found")


# ── head montage: images sorted by one head, coloured by its threshold ────────
def _head_montage(records, score_key, threshold, higher_is_good, title, path, *, plt, cells):
    recs = sorted(records, key=lambda r: r[score_key])
    if len(recs) > cells:
        pick = np.linspace(0, len(recs) - 1, cells).astype(int)
        recs = [recs[j] for j in pick]
    ncol = 8
    nrow = int(np.ceil(len(recs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 1.95))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for a, rec in zip(axes, recs):
        s = rec[score_key]
        ok = (s >= threshold) if higher_is_good else (s <= threshold)
        col = GREEN if ok else RED
        a.axis("on")
        a.imshow(rec["thumb"], cmap="afmhot", vmin=0, vmax=255)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2.6)
        a.set_title(f"{s:.2f}", fontsize=7, color=col, pad=1.5)
    rule = "higher = better" if higher_is_good else "higher = worse"
    fig.suptitle(f"{title}   (threshold={threshold:g}, {rule}; "
                 f"green=accept / red=reject)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6, w_pad=0.4)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── flip gallery: coarse good/bad tips that change under looser thresholds ────
def _flip_gallery(flips, path, *, plt, cells, n_good_def, n_good_low, total):
    if len(flips) > cells:
        flips = flips[:cells]
    ncol = 8
    nrow = max(1, int(np.ceil(len(flips) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 2.25))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for a, (rec, l0, l1) in zip(axes, flips):
        col = GREEN if l1 == "good" else RED
        a.axis("on")
        a.imshow(rec["thumb"], cmap="afmhot", vmin=0, vmax=255)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2.6)
        a.set_title(f"{l0}->{l1}\nN{rec['n_p']:.2f} Q{rec['q_score']:.0f} "
                    f"K{rec['k_p']:.2f} T{rec['t_p']:.2f}", fontsize=6.2, color=col, pad=1.5)
    fig.suptitle(f"Good/bad flips: old strict -> shipped default  "
                 f"(accepted {n_good_def} -> {n_good_low} of {total}; "
                 f"{len(flips)} shown)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.9, w_pad=0.4)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── sweep: accept-rate vs the two most impactful knobs ────────────────────────
def _sweep(records, strict_th, shipped_th, path, *, plt, VisionThresholds, fuse_coarse, decide_fine):
    rs = [{"n_p": r["n_p"], "k_p": r["k_p"], "t_p": r["t_p"],
           "s_axis_ratio": r["s_axis_ratio"], "q_score": r["q_score"]} for r in records]

    cuts = np.linspace(0.30, 0.70, 21)
    good_frac = [np.mean([fuse_coarse(r, VisionThresholds(coarse_good_threshold=c))[0] == "good"
                          for r in rs]) for c in cuts]
    napex = np.linspace(0.30, 0.80, 21)
    usable_frac = [np.mean([decide_fine(r, VisionThresholds(multi_apex_p_max=x))["is_usable"]
                            for r in rs]) for x in napex]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(cuts, good_frac, "-o", ms=3, color="#1565c0")
    ax1.axvline(strict_th.coarse_good_threshold, color=RED, ls="--", lw=1, label="old strict")
    ax1.axvline(shipped_th.coarse_good_threshold, color=GREEN, ls="--", lw=1, label="shipped")
    ax1.set_xlabel("coarse_good_threshold"); ax1.set_ylabel("fraction labelled GOOD")
    ax1.set_title("Accept-rate vs good/bad cut"); ax1.set_ylim(0, 1); ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.plot(napex, usable_frac, "-o", ms=3, color="#6a1b9a")
    ax2.axvline(strict_th.multi_apex_p_max, color=RED, ls="--", lw=1, label="old strict")
    ax2.axvline(shipped_th.multi_apex_p_max, color=GREEN, ls="--", lw=1, label="shipped")
    ax2.set_xlabel("multi_apex_p_max"); ax2.set_ylabel("fraction USABLE")
    ax2.set_title("Usable-rate vs multi-apex tolerance"); ax2.set_ylim(0, 1); ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="VIGIL v2.5 tip-quality threshold demo")
    ap.add_argument("--n", type=int, default=800, help="number of images to sample")
    ap.add_argument("--h5", default=DEFAULT_H5, help="path to sf09_real_512.h5")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--scan-size-nm", type=float, default=None,
                    help="explicit scan size for Head-Q scale conditioning (else model default)")
    ap.add_argument("--montage-cells", type=int, default=40, help="images per head montage")
    ap.add_argument("--flip-cells", type=int, default=48, help="max images in the flip gallery")
    args = ap.parse_args()

    root = _mastv2_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("MAST_VISION_BACKEND", "vigil")

    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mast.vision.thresholds import VisionThresholds, decide_fine, fuse_coarse
    from mast.vision.vigil_backend import VIGILBackend

    h5_path = Path(args.h5)
    if not h5_path.exists():
        print(f"[ERR] h5 not found: {h5_path}", file=sys.stderr)
        return 2

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root.parent / args.out_dir  # repo-root relative
    out.mkdir(parents=True, exist_ok=True)

    # ── load the deployed model ──
    print("[1/4] loading VIGIL v2.5 backend (cold load ~30s)...", flush=True)
    be = VIGILBackend()
    be.preload()
    if args.scan_size_nm:
        be.set_scan_size_nm(args.scan_size_nm)

    # ── sample + infer once per image; cache scalar heads + a display thumbnail ──
    records: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        ds = f["images_display"]  # (N, 512, 512) uint8
        n_total = int(ds.shape[0])
        idx = np.unique(np.linspace(0, n_total - 1, args.n).astype(int))
        print(f"[2/4] scoring {len(idx)} / {n_total} images...", flush=True)
        for c, i in enumerate(idx):
            raw = ds[i]                       # (512,512) uint8 — single slice read
            img = raw.astype(np.float32)      # fwd == bwd (single channel)
            if float(img.std()) < IMG_STD_MIN:
                continue                      # degenerate near-constant frame
            r = be._infer(img)                # raw head dict (one inference)
            records.append({
                "idx": int(i),
                "q_score": float(r["q_score"]),
                "n_p": float(r["n_p"]),
                "k_p": float(r["k_p"]),
                "t_p": float(r["t_p"]),
                "s_axis_ratio": float(r["s_axis_ratio"]),
                "s_asym_p": float(r["s_asym_p"]),
                "thumb": raw[::4, ::4].copy(),  # 512->128 uint8 for display
            })
            if (c + 1) % 100 == 0:
                print(f"    {c + 1}/{len(idx)} ...", flush=True)
    if not records:
        print("[ERR] no usable images sampled", file=sys.stderr)
        return 3
    print(f"    scored {len(records)} images", flush=True)

    # ── the two threshold sets we compare ──
    # OLD strict cuts (pre-2026-07-10) vs the SHIPPED default (VisionThresholds()
    # is now the loosened set). The flip gallery shows what the new default
    # accepts that the old strict cuts rejected.
    strict_th = VisionThresholds(
        coarse_good_threshold=0.50,
        m0_quality_min=72.0,
        multi_apex_p_max=0.50,
        contam_p_max=0.50,
        instability_p_max=0.50,
        m0_axis_ratio_min=0.60,
    )
    shipped_th = VisionThresholds()  # loosened shipped default

    # ── figure 1: per-head montages ──
    print("[3/4] rendering head montages...", flush=True)
    _head_montage(records, "n_p", shipped_th.multi_apex_p_max, False,
                  "Head N (multi-apex prob)", out / "head_montage_N.png",
                  plt=plt, cells=args.montage_cells)
    _head_montage(records, "q_score", shipped_th.m0_quality_min, True,
                  "Head Q (quality score)", out / "head_montage_Q.png",
                  plt=plt, cells=args.montage_cells)
    _head_montage(records, "k_p", shipped_th.contam_p_max, False,
                  "Head K (contamination prob)", out / "head_montage_K.png",
                  plt=plt, cells=args.montage_cells)
    _head_montage(records, "t_p", shipped_th.instability_p_max, False,
                  "Head T (instability prob)", out / "head_montage_T.png",
                  plt=plt, cells=args.montage_cells)
    _head_montage(records, "s_axis_ratio", shipped_th.m0_axis_ratio_min, True,
                  "Head S (apex axis ratio)", out / "head_montage_S.png",
                  plt=plt, cells=args.montage_cells)

    # ── figure 2: good/bad flip gallery (default -> lowered) ──
    print("[4/4] rendering flip gallery + sweep...", flush=True)
    n_good_strict = n_good_ship = 0
    flips: list[tuple] = []
    for rec in records:
        r = {"n_p": rec["n_p"], "k_p": rec["k_p"], "t_p": rec["t_p"],
             "s_axis_ratio": rec["s_axis_ratio"], "q_score": rec["q_score"]}
        l0, _ = fuse_coarse(r, strict_th)
        l1, _ = fuse_coarse(r, shipped_th)
        n_good_strict += l0 == "good"
        n_good_ship += l1 == "good"
        if l0 != l1:
            flips.append((rec, l0, l1))
    # borderline-first: sort flips by |n_p-cut| so the most instructive lead
    flips.sort(key=lambda x: abs(x[0]["n_p"] - shipped_th.multi_apex_p_max))
    _flip_gallery(flips, out / "flip_gallery.png", plt=plt, cells=args.flip_cells,
                  n_good_def=n_good_strict, n_good_low=n_good_ship, total=len(records))

    # ── figure 3: sweep ──
    _sweep(records, strict_th, shipped_th, out / "sweep.png",
           plt=plt, VisionThresholds=VisionThresholds,
           fuse_coarse=fuse_coarse, decide_fine=decide_fine)

    # ── usable-rate summary (fine path) ──
    def _usable(th):
        return sum(decide_fine(
            {"n_p": r["n_p"], "k_p": r["k_p"], "t_p": r["t_p"],
             "s_axis_ratio": r["s_axis_ratio"], "q_score": r["q_score"]}, th)["is_usable"]
            for r in records)
    usable_strict, usable_ship = int(_usable(strict_th)), int(_usable(shipped_th))

    summary = {
        "h5": str(h5_path),
        "n_sampled": len(records),
        "strict_thresholds": strict_th.to_mapping(),
        "shipped_thresholds": shipped_th.to_mapping(),
        "coarse_good": {"strict": n_good_strict, "shipped": n_good_ship,
                        "flips_bad_to_good": sum(1 for _, a, b in flips if a == "bad" and b == "good"),
                        "flips_good_to_bad": sum(1 for _, a, b in flips if a == "good" and b == "bad")},
        "fine_usable": {"strict": usable_strict, "shipped": usable_ship},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    n = len(records)
    print("\n=== SUMMARY (old strict -> shipped default) ===")
    print(f"sampled                : {n}")
    print(f"coarse GOOD  strict    : {n_good_strict}  ({100*n_good_strict/n:.1f}%)")
    print(f"coarse GOOD  shipped   : {n_good_ship}  ({100*n_good_ship/n:.1f}%)")
    print(f"  bad->good flips      : {summary['coarse_good']['flips_bad_to_good']}")
    print(f"  good->bad flips      : {summary['coarse_good']['flips_good_to_bad']}")
    print(f"fine USABLE  strict    : {usable_strict}  ({100*usable_strict/n:.1f}%)")
    print(f"fine USABLE  shipped   : {usable_ship}  ({100*usable_ship/n:.1f}%)")
    print(f"\noutput -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
