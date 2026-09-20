"""A2: merge 50 enriched yaml chunks into a single stm_glossary.yaml.

Output (also mirrored to MASTv2):
    mast/knowledge/stm_glossary.yaml
    MASTv2/mast/knowledge/stm_glossary.yaml

Rules:
    * drop entries with skip: true
    * dedupe by lowercase canonical_en — when duplicated, merge `zh` and
      `asr_normalize` arrays (set-union, preserve order)
    * sort first by domain, then by canonical_en
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_DIR = ROOT / "artifacts" / "openalex_pipeline" / "enriched"
OUT_V1 = ROOT / "mast" / "knowledge" / "stm_glossary.yaml"
OUT_V2 = ROOT / "MASTv2" / "mast" / "knowledge" / "stm_glossary.yaml"


def _norm_str(s) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _merge_lists(a, b) -> list:
    """Set-union preserving order from a, then b."""
    out = []
    seen = set()
    for x in list(a or []) + list(b or []):
        s = _norm_str(x)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def main() -> int:
    chunks = sorted(ENRICHED_DIR.glob("chunk_*.yaml"))
    if not chunks:
        print(f"ERROR: no chunks at {ENRICHED_DIR}")
        return 1
    print(f"Found {len(chunks)} chunks", flush=True)

    merged: OrderedDict[str, dict] = OrderedDict()
    n_loaded = 0
    n_kept = 0
    n_skipped = 0
    n_duplicates = 0
    parse_errors: list[str] = []

    for cp in chunks:
        try:
            data = yaml.safe_load(cp.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            parse_errors.append(f"{cp.name}: {e}")
            continue
        if not isinstance(data, list):
            parse_errors.append(f"{cp.name}: top level not a list")
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            n_loaded += 1
            if entry.get("skip"):
                n_skipped += 1
                continue
            cen = _norm_str(entry.get("canonical_en"))
            if not cen:
                continue
            key = cen.lower()
            if key in merged:
                n_duplicates += 1
                existing = merged[key]
                existing["zh"] = _merge_lists(existing.get("zh"), entry.get("zh"))
                existing["asr_normalize"] = _merge_lists(
                    existing.get("asr_normalize"), entry.get("asr_normalize")
                )
                # Prefer existing abbrev; else take new
                if not existing.get("abbrev") and entry.get("abbrev"):
                    existing["abbrev"] = entry["abbrev"]
                # Prefer existing tts_read_as_zh
                if not existing.get("tts_read_as_zh") and entry.get("tts_read_as_zh"):
                    existing["tts_read_as_zh"] = entry["tts_read_as_zh"]
                # Domain: keep existing if any
                continue
            n_kept += 1
            merged[key] = {
                "canonical_en": cen,
                "abbrev": _norm_str(entry.get("abbrev")),
                "zh": _merge_lists(entry.get("zh"), []),
                "asr_normalize": _merge_lists(entry.get("asr_normalize"), []),
                "tts_read_as_zh": _norm_str(entry.get("tts_read_as_zh")),
                "domain": _norm_str(entry.get("domain")) or "other",
            }

    if parse_errors:
        print(f"WARN: {len(parse_errors)} parse errors:")
        for e in parse_errors[:5]:
            print(f"  {e}")

    print()
    print(f"Loaded entries: {n_loaded}")
    print(f"  skip:true:    {n_skipped}")
    print(f"  duplicates:   {n_duplicates}")
    print(f"  unique kept:  {n_kept}")
    print(f"Final count:    {len(merged)}")

    # Sort: domain, then canonical_en
    final = sorted(
        merged.values(),
        key=lambda d: (d.get("domain", ""), d.get("canonical_en", "").lower()),
    )

    # Distribution by domain
    from collections import Counter
    domain_count = Counter(d["domain"] for d in final)
    print()
    print("By domain:")
    for d, n in sorted(domain_count.items(), key=lambda x: -x[1]):
        print(f"  {d:20} {n}")

    # Write yaml — block style for readability
    text_lines = [
        "# MAST STM glossary — auto-generated from OpenAlex 36,830 STM papers",
        "# Source pipeline: scripts/openalex_pipeline/{01_extract,02_split,A1 sonnet agents x50,04_merge}",
        f"# Total entries: {len(final)}",
        "# Schema: {canonical_en, abbrev, zh[], asr_normalize[], tts_read_as_zh, domain}",
        "#   - zh[0] is the canonical Chinese translation",
        "#   - asr_normalize lists common ASR transcription errors that should map to canonical_en/abbrev",
        "#   - tts_read_as_zh: feed this to TTS instead of canonical_en for natural Chinese pronunciation",
        "",
    ]
    body = yaml.dump(
        final,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )
    out_text = "\n".join(text_lines) + body

    OUT_V1.parent.mkdir(parents=True, exist_ok=True)
    OUT_V2.parent.mkdir(parents=True, exist_ok=True)
    OUT_V1.write_text(out_text, encoding="utf-8")
    OUT_V2.write_text(out_text, encoding="utf-8")
    print()
    print(f"Wrote {len(final)} entries:")
    print(f"  v1: {OUT_V1}")
    print(f"  v2: {OUT_V2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
