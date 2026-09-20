"""F2-enhanced: merge sonnet F1 records with regex fallback → priors.json.

Sonnet records have:
  - phase classification (imaging/sts/mapping/...) likely more accurate
  - extra fields (tunneling_current_pa, frequency_hz, modulation_v,
    z_range_nm, scan_speed_nm_s)
  - emotion-tagged paper triage

Regex records cover ~26.6% of papers; sonnet completes only 17/50 chunks
due to Anthropic rate limits.

Strategy:
  * Take ALL sonnet records (high signal)
  * Take regex records for papers not yet covered by sonnet
  * Aggregate to per-(material, phase) {p25, p50, p75, min, max, n}
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED = ROOT / "artifacts" / "openalex_pipeline" / "params_extracted"
SONNET_GLOB = "params_chunk_*.jsonl"
REGEX_PATH = EXTRACTED / "params_regex.jsonl"
OUT_V1 = ROOT / "mast" / "knowledge" / "literature_priors.json"
OUT_V2 = ROOT / "MASTv2" / "mast" / "knowledge" / "literature_priors.json"

# Numeric fields we summarise — matches sonnet schema (regex is subset).
PARAM_FIELDS = [
    "bias_v", "setpoint_pa", "temperature_k", "scan_size_nm",
    "tunneling_current_pa", "frequency_hz", "z_range_nm",
    "modulation_v", "scan_speed_nm_s",
]
MIN_PAPERS_PER_CELL = 3
MIN_PAPERS_PER_GLOBAL = 5


def _flatten(v) -> list[float]:
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


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> int:
    sonnet_chunks = sorted(EXTRACTED.glob(SONNET_GLOB))
    print(f"sonnet chunks: {len(sonnet_chunks)}")
    sonnet_records: list[dict] = []
    for cp in sonnet_chunks:
        sonnet_records.extend(_read_jsonl(cp))
    print(f"  sonnet records: {len(sonnet_records):,}")

    regex_records = _read_jsonl(REGEX_PATH)
    print(f"  regex records:  {len(regex_records):,}")

    # Sonnet covers some papers; regex backs off for non-covered.
    sonnet_papers = {r.get("work_id", "") for r in sonnet_records if r.get("work_id")}
    print(f"  sonnet covers {len(sonnet_papers):,} unique papers")

    regex_complement = [
        r for r in regex_records
        if r.get("work_id", "") and r.get("work_id") not in sonnet_papers
    ]
    print(f"  regex complement: {len(regex_complement):,}")

    all_records = sonnet_records + regex_complement
    print(f"  combined records: {len(all_records):,}")

    # Buckets: {material: {phase: {param: [values...], "_papers": set, "_sources": Counter}}}
    by_mp: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "_papers": set(),
        "_sources": defaultdict(int),
        **{f: [] for f in PARAM_FIELDS},
    }))
    by_p: dict[str, dict] = defaultdict(lambda: {
        "_papers": set(),
        **{f: [] for f in PARAM_FIELDS},
    })

    n_sonnet = 0
    n_regex = 0
    for rec in sonnet_records:
        material = (rec.get("material") or "").strip()
        phase = (rec.get("phase") or "imaging").strip()
        wid = rec.get("work_id", "")
        if not material or not phase:
            continue
        cell = by_mp[material][phase]
        cell["_papers"].add(wid)
        cell["_sources"]["sonnet"] += 1
        n_sonnet += 1
        for f_name in PARAM_FIELDS:
            v = rec.get(f_name)
            if v is not None:
                cell[f_name].extend(_flatten(v))
        gcell = by_p[phase]
        gcell["_papers"].add(wid)
        for f_name in PARAM_FIELDS:
            v = rec.get(f_name)
            if v is not None:
                gcell[f_name].extend(_flatten(v))

    for rec in regex_complement:
        material = (rec.get("material") or "").strip()
        phase = (rec.get("phase") or "imaging").strip()
        wid = rec.get("work_id", "")
        if not material or not phase:
            continue
        cell = by_mp[material][phase]
        cell["_papers"].add(wid)
        cell["_sources"]["regex"] += 1
        n_regex += 1
        for f_name in PARAM_FIELDS:
            v = rec.get(f_name)
            if v is not None:
                cell[f_name].extend(_flatten(v))
        gcell = by_p[phase]
        gcell["_papers"].add(wid)
        for f_name in PARAM_FIELDS:
            v = rec.get(f_name)
            if v is not None:
                gcell[f_name].extend(_flatten(v))

    by_material_phase: dict[str, dict[str, dict]] = {}
    for material, phases in by_mp.items():
        out_phases = {}
        for phase, cell in phases.items():
            n_papers = len(cell["_papers"])
            if n_papers < MIN_PAPERS_PER_CELL:
                continue
            entry = {
                "n_papers": n_papers,
                "sources": dict(cell["_sources"]),
            }
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
        "by_material_phase": dict(sorted(
            by_material_phase.items(),
            key=lambda kv: -sum(p["n_papers"] for p in kv[1].values()),
        )),
        "by_phase": dict(sorted(by_phase.items())),
        "metadata": {
            "source": "OpenAlex 32016 STM papers (sonnet F1 + regex fallback)",
            "n_sonnet_records": n_sonnet,
            "n_regex_records": n_regex,
            "n_sonnet_chunks_completed": len(sonnet_chunks),
            "n_total_chunks": 50,
            "n_records_total": n_sonnet + n_regex,
            "min_papers_per_cell": MIN_PAPERS_PER_CELL,
            "min_papers_per_global": MIN_PAPERS_PER_GLOBAL,
        },
    }

    text = json.dumps(output, indent=2, ensure_ascii=False)
    OUT_V1.parent.mkdir(parents=True, exist_ok=True)
    OUT_V2.parent.mkdir(parents=True, exist_ok=True)
    OUT_V1.write_text(text, encoding="utf-8")
    OUT_V2.write_text(text, encoding="utf-8")

    print()
    print(f"Materials: {len(by_material_phase)}")
    print(f"(material,phase) cells: {sum(len(v) for v in by_material_phase.values())}")
    print(f"Global phases: {len(by_phase)}")
    print(f"sonnet records used: {n_sonnet}")
    print(f"regex records used (complement): {n_regex}")
    print()
    print("Top 15 materials by total paper count:")
    for mat, phases in list(by_material_phase.items())[:15]:
        total = sum(p["n_papers"] for p in phases.values())
        ph_summary = ", ".join(f"{ph}({d['n_papers']})" for ph, d in phases.items())
        print(f"  {mat:30}  total={total:4}   {ph_summary}")
    print()
    print(f"Wrote v1: {OUT_V1}")
    print(f"Wrote v2: {OUT_V2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
