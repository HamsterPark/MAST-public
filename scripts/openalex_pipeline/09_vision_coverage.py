"""E: vision training-data material coverage vs literature material counts.

Maps each stm-datasets/ subdir to a coarse OpenAlex material category, then
compares to OpenAlex paper counts. Output is a markdown report at
docs/v2/benchmarks/material_coverage.md.

Goal: spot Phase-9 DINOv3 training-data gaps where literature interest is
high but our training corpus is thin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "stm-datasets"
COVERAGE_JSON = ROOT / "mast" / "knowledge" / "material_coverage.json"
OUT_MD = ROOT / "docs" / "v2" / "benchmarks" / "material_coverage.md"


# Manual mapping: dataset directory → (display name, mast_category, material_keywords)
DATASETS = {
    "asd-stm":      ("ASD-STM",       "general",            []),
    "asd-stm-data": ("ASD-STM-data",  "general",            []),
    "fast-spm":     ("Fast-SPM",      "general",            []),
    "deepspm":      ("DeepSPM",       "general",            []),
    "cyclegan-stm": ("CycleGAN-STM",  "general",            []),
    "smalley-wse2": ("Smalley-WSe2",  "2d_material",        ["WSe2"]),
    "stras":        ("DAS-STRAS",     "general",            []),
    "wm811k":       ("WM-811K",       "semiconductor",      ["Si wafer", "semiconductor wafer"]),
    "jarvis-stm":   ("JARVIS-STM",    "general",            []),
    "ppstm-examples": ("PPSTM examples", "general",         []),
    "quam-afm":     ("QUAM-AFM",      "afm-only",           []),
    "repos":        ("(supporting)",  "tooling",            []),
}


def main() -> int:
    if not COVERAGE_JSON.exists():
        print(f"ERROR: {COVERAGE_JSON} not found — run 08_material_coverage.py first")
        return 1
    coverage = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))
    by_cat = coverage["by_mast_category"]

    # Index datasets that exist on disk
    present_datasets = {p.name for p in DATASETS_DIR.iterdir() if p.is_dir()}
    download_status_path = DATASETS_DIR / "download_status.json"
    download_status = {}
    if download_status_path.exists():
        try:
            download_status = json.loads(download_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    lines = [
        "# STM Vision Training Data — Material Coverage Audit",
        "",
        "_Auto-generated from `mast/knowledge/material_coverage.json` (OpenAlex 36,830 papers)_",
        "_and on-disk `stm-datasets/` directory enumeration._",
        "",
        "## 1. Datasets present on disk",
        "",
        "| Dataset dir | Display | MAST category | Material tags | Downloaded? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dirname, (display, cat, tags) in DATASETS.items():
        present = dirname in present_datasets
        status = ""
        if download_status:
            ds_status = download_status.get(dirname, {})
            if isinstance(ds_status, dict):
                status = ds_status.get("status", "")
        if not status:
            status = "✓" if present else "✗"
        tag_str = ", ".join(tags) if tags else "_(unspecified)_"
        lines.append(f"| `{dirname}` | {display} | {cat} | {tag_str} | {status} |")
    lines.append("")

    lines.append("## 2. Literature interest vs training data")
    lines.append("")
    lines.append("| MAST category | Literature papers | Recent (≥2020) | Training datasets |")
    lines.append("| --- | ---: | ---: | --- |")
    cat_to_datasets: dict[str, list[str]] = {}
    for d, (display, cat, _) in DATASETS.items():
        if d in present_datasets:
            cat_to_datasets.setdefault(cat, []).append(display)

    cats_sorted = sorted(by_cat.items(), key=lambda kv: -kv[1]["n_papers"])
    for cat, st in cats_sorted:
        ds = ", ".join(cat_to_datasets.get(cat, [])) or "_(none)_"
        lines.append(f"| {cat} | {st['n_papers']:,} | {st['n_recent']:,} | {ds} |")
    lines.append("")

    # Gap analysis: large literature, no dataset
    lines.append("## 3. Coverage gaps (potential Phase-9 training data targets)")
    lines.append("")
    lines.append("Categories with **>1000 papers** but **no on-disk dataset**:")
    lines.append("")
    gaps = [
        (cat, st) for cat, st in cats_sorted
        if st["n_papers"] >= 1000
        and cat not in cat_to_datasets
        and cat not in ("other", "tooling", "afm-only", "general")
    ]
    if gaps:
        for cat, st in gaps:
            lines.append(f"- **{cat}**: {st['n_papers']:,} papers ({st['n_recent']} recent, "
                         f"{st['n_materials']} unique materials)")
    else:
        lines.append("_(none; all heavy-literature categories have at least one dataset)_")
    lines.append("")

    lines.append("## 4. Recommended actions")
    lines.append("")
    rec = []
    if any(c == "topological" and not ds for c, ds in cat_to_datasets.items()):
        pass
    if "topological" not in cat_to_datasets:
        rec.append("- **Topological materials** (8,186 lit. papers, 1,943 since 2020) — "
                   "no training set; highest-priority gap.")
    if "superconductor" not in cat_to_datasets:
        rec.append("- **Superconductors** (1,018 lit. papers, 318 since 2020) — "
                   "FeSe / cuprate / NbSe2 imaging would round out vortex / gap-mapping coverage.")
    if "molecular_adsorbate" not in cat_to_datasets:
        rec.append("- **Molecular adsorbates** (1,476 papers) — phthalocyanine / PTCDA / C60 "
                   "datasets would complement on-surface synthesis work.")
    if "magnetic_spm" not in cat_to_datasets:
        rec.append("- **Magnetic / SP-STM** (294 papers) — small but vision pipeline could "
                   "benefit from skyrmion / domain-wall examples.")

    if rec:
        lines.extend(rec)
    else:
        lines.append("_(no immediate gaps)_")
    lines.append("")
    lines.append("## 5. Methodology notes")
    lines.append("")
    lines.append("- Literature counts come from OpenAlex `stm_classified.parquet` "
                 "(LLM-tagged 36,830 papers).")
    lines.append("- MAST category mapping is keyword-heuristic (see "
                 "`scripts/openalex_pipeline/08_material_coverage.py`).")
    lines.append("- The `other` bucket holds the ~9k papers whose materials don't match "
                 "any of the 10 MAST sample-workflow categories — many are well-defined "
                 "samples (HOPG, graphite, generic 'metal surface') that warrant their own "
                 "category before final Phase-9 dataset planning.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} lines → {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
