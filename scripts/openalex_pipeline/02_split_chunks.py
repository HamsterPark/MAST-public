"""Split candidate_terms.jsonl into N=50 chunks of 60 terms each.

Output: artifacts/openalex_pipeline/chunks/chunk_NN.jsonl  (00..49)
        artifacts/openalex_pipeline/chunks/_index.json     (manifest)

The downstream A1 step starts 50 parallel sonnet agents, each handling one
chunk.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "openalex_pipeline" / "candidate_terms.jsonl"
OUT_DIR = ROOT / "artifacts" / "openalex_pipeline" / "chunks"
N_CHUNKS = 50


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: candidate file not found: {SRC}")
        return 1

    items: list[dict] = []
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    total = len(items)
    chunk_size = (total + N_CHUNKS - 1) // N_CHUNKS  # ceil
    print(f"Total candidates: {total}, chunks: {N_CHUNKS}, per chunk: {chunk_size}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for i in range(N_CHUNKS):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            break
        chunk = items[start:end]
        path = OUT_DIR / f"chunk_{i:02d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in chunk:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest.append({"chunk_id": i, "path": str(path), "n_terms": len(chunk)})
        print(f"  chunk_{i:02d}: {len(chunk)} terms")

    (OUT_DIR / "_index.json").write_text(
        json.dumps({"n_chunks": len(manifest), "chunks": manifest}, indent=2),
        encoding="utf-8",
    )
    print(f"Done: {len(manifest)} chunks → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
