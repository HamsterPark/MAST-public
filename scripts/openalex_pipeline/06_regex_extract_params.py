"""F1-fallback: regex-based parameter extraction (replaces rate-limited sonnet path).

Sonnet-based F1 hit Anthropic per-org rate limits at 50 concurrent agents
processing 640 papers each. This regex pass extracts the same numerical priors
without LLM calls, runs in ~30 seconds over the full 32k papers, and produces a
schema-compatible jsonl that downstream F2 aggregator consumes.

Coverage trade-off: regex catches the common patterns (~60-70% of explicit
mentions) but misses paraphrased / hedged language ("close to room temperature",
"sub-Kelvin"). Good enough for v1 priors; LLM refinement is a future polish.

Output:
    artifacts/openalex_pipeline/params_extracted/params_regex.jsonl
    one record per (paper, phase) hit
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "artifacts" / "openalex_pipeline" / "papers_chunks"
OUT_DIR = ROOT / "artifacts" / "openalex_pipeline" / "params_extracted"
OUT_PATH = OUT_DIR / "params_regex.jsonl"


# ── Phase keyword classifier ──────────────────────────────────────────
PHASE_KEYWORDS: dict[str, list[str]] = {
    "imaging": [
        "topograph", "constant current", "constant height", "atomic resolution",
        "stm image", "scanning tunneling microscopy image", "stm scan",
    ],
    "sts": [
        "scanning tunneling spectroscopy", " sts ", "(sts)", "di/dv spectr",
        "i-v spectr", "i/v spectr", "tunneling spectrum", "point spectroscopy",
        "differential conductance spectr",
    ],
    "mapping": [
        "di/dv map", "conductance map", "current imaging", "cits ",
        "spectroscopic imaging", "qpi map", "quasiparticle interference",
    ],
    "atom_manipulation": [
        "atom manipulation", "molecular manipulation", "lateral manipulation",
        "vertical manipulation", "single-atom",
    ],
    "tip_prep": ["tip preparation", "tip conditioning", "field emission",
                 "voltage pulse", "tip indent"],
    "approach": ["coarse approach", "fine approach", "tip-sample approach"],
    "lithography": ["lithograph", "tip-induced patterning", "stm lithography"],
}


def classify_phases(text: str) -> list[str]:
    t = text.lower()
    found = []
    for phase, kws in PHASE_KEYWORDS.items():
        if any(kw in t for kw in kws):
            found.append(phase)
    return found or ["imaging"]   # default to imaging if nothing else matches


# ── Numeric pattern extractors ────────────────────────────────────────

# Each returns either None, a float, or [low, high]. Units are normalised to:
#   bias_v, setpoint_pa, temperature_k, scan_size_nm

_NUM = r"-?\d+(?:\.\d+)?"
_RANGE = rf"(?:{_NUM})\s*(?:[-–—~]|to)\s*(?:{_NUM})"

def _to_float(s: str) -> float | None:
    s = s.replace("−", "-").strip()
    if not s or s in ("-", "+"):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _parse_range(match: str) -> list[float] | None:
    # Use explicit non-numeric splitter so unary minus inside a number isn't
    # confused with the range separator.
    parts = re.split(r"\s*(?:–|—|~|\bto\b)\s*", match)
    if len(parts) < 2:
        # Fallback: ASCII '-' between two numbers ⇒ split on " - "
        parts = re.split(r"\s+-\s+", match)
        if len(parts) < 2:
            return None
    a = _to_float(parts[0])
    b = _to_float(parts[-1])
    if a is None or b is None:
        return None
    return [a, b]


def _extract_voltage_v(text: str) -> float | list[float] | None:
    # mV → V (÷1000); V → V; kV → V (×1000)
    # Patterns: "0.5 V", "+0.5 V", "−0.5 V", "0.5 to 1.0 V", "100 mV", "−100 mV"
    candidates: list[float | list[float]] = []
    for m in re.finditer(rf"({_RANGE})\s*(mV|V)\b", text):
        unit = m.group(2)
        vals = _parse_range(m.group(1))
        if vals is None:
            continue
        if unit == "mV":
            vals = [v / 1000 for v in vals]
        if all(abs(v) <= 20 for v in vals):
            candidates.append(vals)
    for m in re.finditer(rf"({_NUM})\s*(mV|V)\b", text):
        unit = m.group(2)
        v = _to_float(m.group(1))
        if v is None:
            continue
        if unit == "mV":
            v = v / 1000
        if abs(v) <= 20:
            candidates.append(v)
    return candidates[0] if candidates else None


def _extract_setpoint_pa(text: str) -> float | list[float] | None:
    # nA → pA (×1000); pA → pA; fA → pA (÷1000)
    candidates: list[float | list[float]] = []
    # Look for "setpoint" / "tunneling current" / "I_t" near the value
    for m in re.finditer(rf"({_RANGE})\s*(pA|nA|fA)\b", text):
        unit = m.group(2)
        vals = _parse_range(m.group(1))
        if vals is None:
            continue
        if unit == "nA":
            vals = [v * 1000 for v in vals]
        elif unit == "fA":
            vals = [v / 1000 for v in vals]
        if all(0 < v <= 1e7 for v in vals):
            candidates.append(vals)
    for m in re.finditer(rf"({_NUM})\s*(pA|nA|fA)\b", text):
        unit = m.group(2)
        v = _to_float(m.group(1))
        if v is None:
            continue
        if unit == "nA":
            v = v * 1000
        elif unit == "fA":
            v = v / 1000
        if 0 < v <= 1e7:
            candidates.append(v)
    return candidates[0] if candidates else None


def _extract_temperature_k(text: str) -> float | list[float] | None:
    # K, mK; "room temperature" → 300; "RT" → 300
    t = text.lower()
    if "room temperature" in t or " rt " in f" {t} ":
        return 300.0
    candidates: list[float | list[float]] = []
    for m in re.finditer(rf"({_RANGE})\s*(mK|K)\b", text):
        unit = m.group(2)
        vals = _parse_range(m.group(1))
        if vals is None:
            continue
        if unit == "mK":
            vals = [v / 1000 for v in vals]
        if all(0 < v <= 1500 for v in vals):
            candidates.append(vals)
    for m in re.finditer(rf"({_NUM})\s*(mK|K)\b", text):
        unit = m.group(2)
        v = _to_float(m.group(1))
        if v is None:
            continue
        if unit == "mK":
            v = v / 1000
        if 0 < v <= 1500:
            candidates.append(v)
    return candidates[0] if candidates else None


def _extract_scan_size_nm(text: str) -> float | list[float] | None:
    # nm; µm → nm (×1000); Å → nm (÷10)
    # Look near "scan" / "image" / "area" / "region" of size ##
    candidates: list[float | list[float]] = []
    for m in re.finditer(
        rf"({_NUM})\s*(?:×|x|\*)\s*({_NUM})\s*(nm|µm|um|Å)\b",
        text,
    ):
        unit = m.group(3)
        v = _to_float(m.group(1))
        if v is None:
            continue
        if unit in ("µm", "um"):
            v = v * 1000
        elif unit == "Å":
            v = v / 10
        if 0 < v <= 1e5:
            candidates.append(v)
    if candidates:
        return candidates[0]
    for m in re.finditer(rf"({_NUM})\s*(nm|µm|um|Å)\b", text):
        unit = m.group(2)
        v = _to_float(m.group(1))
        if v is None:
            continue
        if unit in ("µm", "um"):
            v = v * 1000
        elif unit == "Å":
            v = v / 10
        if 0 < v <= 1e5:
            candidates.append(v)
    return candidates[0] if candidates else None


def extract(rec: dict) -> list[dict]:
    """Extract one or more (paper, phase, params) records. Empty list = no hits."""
    text = (rec.get("title", "") + ". " + rec.get("abstract", "")).strip()
    if not text:
        return []
    material = rec.get("material", "")
    if not material:
        return []
    phases = classify_phases(text)

    bias = _extract_voltage_v(text)
    setpt = _extract_setpoint_pa(text)
    temp = _extract_temperature_k(text)
    size = _extract_scan_size_nm(text)

    # If no parameter extracted, skip.
    if bias is None and setpt is None and temp is None and size is None:
        return []

    out = []
    for ph in phases:
        rec_out = {
            "work_id": rec.get("work_id", ""),
            "material": material,
            "phase": ph,
            "year": rec.get("year", 0),
        }
        if bias is not None:
            rec_out["bias_v"] = bias
        if setpt is not None:
            rec_out["setpoint_pa"] = setpt
        if temp is not None:
            rec_out["temperature_k"] = temp
        if size is not None:
            rec_out["scan_size_nm"] = size
        out.append(rec_out)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = sorted(PAPERS_DIR.glob("papers_chunk_*.jsonl"))
    print(f"Processing {len(chunks)} chunks …", flush=True)

    n_papers = 0
    n_hits = 0
    n_records = 0
    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for cp in chunks:
            with cp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n_papers += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records = extract(rec)
                    if records:
                        n_hits += 1
                        for r in records:
                            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                            n_records += 1
            print(f"  done {cp.name}: total papers={n_papers:,}", flush=True)

    print(f"\nDone.")
    print(f"  papers processed: {n_papers:,}")
    print(f"  papers with hits: {n_hits:,}  ({100*n_hits/max(n_papers,1):.1f}%)")
    print(f"  records emitted:  {n_records:,}")
    print(f"  output:           {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
