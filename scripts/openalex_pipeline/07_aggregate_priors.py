"""F2: aggregate per-paper extractions → (material, phase) parameter priors.

Reads: artifacts/openalex_pipeline/params_extracted/params_regex.jsonl
Writes:
    mast/knowledge/literature_priors.json     (v1)
    MASTv2/mast/knowledge/literature_priors.json   (v2)

Schema:
    {
      "by_material_phase": {
        "Au(111)": {
          "imaging": {
            "n_papers": 145,
            "bias_v":      {"p25": 0.5, "p50": 1.0, "p75": 1.5, "min": -2.0, "max": 3.0, "n": 132},
            "setpoint_pa": {...},
            "temperature_k": {...},
            "scan_size_nm": {...}
          },
          "sts": {...}
        },
        ...
      },
      "by_phase": {
        "imaging": {  # global fallback if material not in by_material_phase
          "bias_v": {...},
          ...
        }
      },
      "metadata": {
        "source": "OpenAlex 32016 STM papers (regex extraction)",
        "n_records": 8882,
        "extraction_method": "regex (sonnet F1 deferred due to rate-limits)"
      }
    }
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "openalex_pipeline" / "params_extracted" / "params_regex.jsonl"
OUT_V1 = ROOT / "mast" / "knowledge" / "literature_priors.json"
OUT_V2 = ROOT / "MASTv2" / "mast" / "knowledge" / "literature_priors.json"

# Param fields we summarise
PARAM_FIELDS = ["bias_v", "setpoint_pa", "temperature_k", "scan_size_nm"]
# Min sample size to include a per-(material, phase) cell
MIN_PAPERS_PER_CELL = 3
# Min sample size to include a per-phase global cell
MIN_PAPERS_PER_GLOBAL = 5


def _flatten_value(v) -> list[float]:
    """A single value or [low, high] → list of floats for stat aggregation."""
    if isinstance(v, list):
        return [float(x) for x in v if isinstance(x, (int, float))]
    if isinstance(v, (int, float)):
        return [float(v)]
    return []


def _summarise(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    n = len(values)
    p25 = statistics.quantiles(values, n=4)[0] if n >= 4 else min(values)
    p75 = statistics.quantiles(values, n=4)[2] if n >= 4 else max(values)
    return {
        "p25": round(p25, 4),
        "p50": round(statistics.median(values), 4),
        "p75": round(p75, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
    }


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found")
        return 1

    # Buckets: {material: {phase: {param: [values...], "_papers": set}}}
    by_mp: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "_papers": set(),
        **{f: [] for f in PARAM_FIELDS},
    }))
    # Global by phase
    by_p: dict[str, dict] = defaultdict(lambda: {
        "_papers": set(),
        **{f: [] for f in PARAM_FIELDS},
    })

    n_records = 0
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_records += 1
            material = (rec.get("material") or "").strip()
            phase = (rec.get("phase") or "").strip()
            wid = rec.get("work_id", "")
            if not material or not phase:
                continue
            cell = by_mp[material][phase]
            cell["_papers"].add(wid)
            for f_name in PARAM_FIELDS:
                v = rec.get(f_name)
                if v is not None:
                    cell[f_name].extend(_flatten_value(v))
            gcell = by_p[phase]
            gcell["_papers"].add(wid)
            for f_name in PARAM_FIELDS:
                v = rec.get(f_name)
                if v is not None:
                    gcell[f_name].extend(_flatten_value(v))

    # Build the output
    by_material_phase: dict[str, dict[str, dict]] = {}
    for material, phases in by_mp.items():
        out_phases = {}
        for phase, cell in phases.items():
            n_papers = len(cell["_papers"])
            if n_papers < MIN_PAPERS_PER_CELL:
                continue
            entry = {"n_papers": n_papers}
            for f_name in PARAM_FIELDS:
                summ = _summarise(cell[f_name])
                if summ:
                    entry[f_name] = summ
            if any(f in entry for f in PARAM_FIELDS):
                out_phases[phase] = entry
        if out_phases:
            by_material_phase[material] = out_phases

    by_phase: dict[str, dict] = {}
    for phase, cell in by_p.items():
        n_papers = len(cell["_papers"])
        if n_papers < MIN_PAPERS_PER_GLOBAL:
            continue
        entry = {"n_papers": n_papers}
        for f_name in PARAM_FIELDS:
            summ = _summarise(cell[f_name])
            if summ:
                entry[f_name] = summ
        if any(f in entry for f in PARAM_FIELDS):
            by_phase[phase] = entry

    output = {
        "by_material_phase": dict(sorted(by_material_phase.items(), key=lambda kv: -sum(p["n_papers"] for p in kv[1].values()))),
        "by_phase": dict(sorted(by_phase.items())),
        "metadata": {
            "source": f"OpenAlex {n_records:,} extractions over 32016 STM papers",
            "n_records": n_records,
            "extraction_method": "regex (sonnet F1 deferred due to API rate-limits)",
            "min_papers_per_cell": MIN_PAPERS_PER_CELL,
            "min_papers_per_global": MIN_PAPERS_PER_GLOBAL,
        },
    }

    OUT_V1.parent.mkdir(parents=True, exist_ok=True)
    OUT_V2.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(output, indent=2, ensure_ascii=False)
    OUT_V1.write_text(text, encoding="utf-8")
    OUT_V2.write_text(text, encoding="utf-8")

    print(f"Records read:        {n_records:,}")
    print(f"Materials:           {len(by_material_phase)}")
    print(f"(material,phase) cells: {sum(len(v) for v in by_material_phase.values())}")
    print(f"Global phases:       {len(by_phase)}")
    print()
    print(f"Wrote:")
    print(f"  v1: {OUT_V1}")
    print(f"  v2: {OUT_V2}")
    print()
    # Show top materials
    print("Top 15 materials by total paper count:")
    for mat, phases in list(by_material_phase.items())[:15]:
        total = sum(p["n_papers"] for p in phases.values())
        ph_summary = ", ".join(f"{ph}({d['n_papers']})" for ph, d in phases.items())
        print(f"  {mat:30}  total={total:4}   phases: {ph_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
