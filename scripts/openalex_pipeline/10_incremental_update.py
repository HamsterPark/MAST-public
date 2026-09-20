"""Incremental update of the OpenAlex STM corpus.

Pull all OpenAlex Works tagged with the STM concept (`C2778049076`) that were
published or updated **after** the last build, append them to the local
parquet files, and (optionally) extend the embedding index in place.

Usage:
    python 10_incremental_update.py [--since 2026-01-01]
                                    [--rebuild-index]
                                    [--dry-run]

Where:
  * `--since` YYYY-MM-DD: defaults to MAX(publication_year) + month detection
                          from the existing stm_papers.parquet, fall back to
                          today − 30 days.
  * `--rebuild-index`:    rebuild vectors.npy / metadata.parquet from scratch
                          via 03_build_embedding_index.py instead of in-place
                          appending (safer, slower).
  * `--dry-run`:          fetch + count, do not write anything.

Output:
  * Updates `stm_papers.parquet` (cleaned/) in place (additive, dedup by DOI)
  * Updates `stm_classified.parquet` minimally (sets material="<unclassified>"
    for new papers — re-run upstream LLM classifier later if you want them
    properly tagged)
  * Optional: extends MASTv2/artifacts/literature_index/{vectors.npy,
              metadata.parquet} with new rows

Politeness: passes a `mailto=` query parameter to enter OpenAlex's polite pool,
read from the `OPENALEX_MAILTO` environment variable. Without it the requests
still work but fall back to the shared pool, which is rate-limited far harder.
Uses cursor pagination (200 per page).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import numpy as np
import pandas as pd

from mast.knowledge.openalex_loader import (
    cleaned_parquet, classified_parquet,
    restore_abstract,
)

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "MASTv2" / "artifacts" / "literature_index"
VECTORS_PATH = INDEX_DIR / "vectors.npy"
METADATA_PATH = INDEX_DIR / "metadata.parquet"

OPENALEX_CONCEPT_STM = "C2778049076"
#: OpenAlex polite pool 要一个联系邮箱。它是**站点配置**,不是代码常量 ——
#: 写死一个人的地址意味着每台装了这份代码的机器都以他的名义发请求。
#: 留空就退回共享池:更慢,但不会冒名。
MAILTO = os.environ.get("OPENALEX_MAILTO", "").strip()
EMBEDDING_MODEL = "text-embedding-v3"
DIM = 1024


# ── OpenAlex fetch ────────────────────────────────────────────────────

def _fetch_works_since(since_date: str) -> Iterator[dict]:
    """Yield works updated since *since_date* (YYYY-MM-DD)."""
    base = "https://api.openalex.org/works"
    cursor = "*"
    page = 0
    fields = (
        "id,doi,title,publication_year,abstract_inverted_index,"
        "primary_location,authorships,concepts,keywords,cited_by_count,"
        "from_updated_date"
    )
    with httpx.Client(timeout=60.0) as client:
        while cursor:
            params = {
                "filter": (
                    f"concepts.id:{OPENALEX_CONCEPT_STM},"
                    f"from_updated_date:{since_date}"
                ),
                "select": fields,
                "per_page": 200,
                "cursor": cursor,
            }
            if MAILTO:
                params["mailto"] = MAILTO
            data = None
            for attempt in range(6):
                try:
                    r = client.get(base, params=params)
                    if r.status_code == 429:
                        wait = 2 ** attempt + 2
                        print(f"  HTTP 429, sleeping {wait}s before retry {attempt+1}/6", flush=True)
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    break
                except httpx.HTTPStatusError as exc:
                    print(f"  error {exc.response.status_code} on attempt {attempt+1}/6", flush=True)
                    time.sleep(2 ** attempt)
                except Exception as exc:
                    print(f"  network error on attempt {attempt+1}/6: {exc}", flush=True)
                    time.sleep(2 ** attempt)
            if data is None:
                raise RuntimeError("OpenAlex fetch failed after 6 retries")
            results = data.get("results") or []
            for w in results:
                yield w
            page += 1
            meta = data.get("meta") or {}
            next_cursor = meta.get("next_cursor")
            if not next_cursor or not results:
                break
            cursor = next_cursor
            print(f"  page {page}: {len(results)} (cursor advancing)", flush=True)
            time.sleep(0.5)   # be polite


def _project(rec: dict) -> dict:
    """Project an OpenAlex Work into the stm_papers.parquet schema."""
    primary = (rec.get("primary_location") or {}).get("source") or {}
    auths = rec.get("authorships") or []
    first_author = ""
    institutions = []
    countries = []
    if auths:
        first_author = (auths[0].get("author") or {}).get("display_name", "") or ""
        for a in auths:
            for inst in (a.get("institutions") or []):
                if inst.get("display_name"):
                    institutions.append(inst["display_name"])
                if inst.get("country_code"):
                    countries.append(inst["country_code"])
    abstract = restore_abstract(rec.get("abstract_inverted_index"))
    concepts_str = ";".join(
        c.get("display_name", "")
        for c in (rec.get("concepts") or [])
        if c.get("display_name")
    )
    keywords_str = ";".join(
        k.get("keyword", "")
        for k in (rec.get("keywords") or [])
        if k.get("keyword")
    )
    return {
        "work_id": rec.get("id", ""),
        "doi": rec.get("doi", "") or "",
        "title": rec.get("title", "") or "",
        "abstract": abstract,
        "publication_year": int(rec.get("publication_year") or 0),
        "journal": primary.get("display_name", "") or "",
        "authors": ";".join((a.get("author") or {}).get("display_name", "") for a in auths),
        "first_author": first_author,
        "institutions": ";".join(dict.fromkeys(institutions)),
        "countries": ";".join(dict.fromkeys(countries)),
        "concepts": concepts_str,
        "keywords": keywords_str,
        "cited_by_count": int(rec.get("cited_by_count") or 0),
        "has_abstract": bool(abstract),
        "work_type": "journal-article",
    }


# ── Embedding update ─────────────────────────────────────────────────

def _embed_new_rows(new_rows: list[dict], api_key: str) -> np.ndarray:
    """Embed *new_rows* (titles + abstracts) → (N, 1024) float32 via DashScope."""
    endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    out = np.zeros((len(new_rows), DIM), dtype=np.float32)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60.0, headers=headers) as client:
        for i in range(0, len(new_rows), 10):
            batch = new_rows[i : i + 10]
            texts = [
                (r["title"] + "\n\n" + r["abstract"][:6000]).strip()[:8000]
                for r in batch
            ]
            r = client.post(endpoint, json={"model": EMBEDDING_MODEL, "input": texts, "dimensions": DIM})
            r.raise_for_status()
            data = r.json()
            vecs = np.array([d["embedding"] for d in data["data"]], dtype=np.float32)
            out[i : i + len(batch)] = vecs
            time.sleep(0.1)
    return out


def _load_dashscope_key() -> str:
    val = (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("ALIYUN_BAILIAN_API_KEY", "").strip()
    )
    if val:
        return val
    p = ROOT / "api key" / "dashscope.env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise RuntimeError("No DashScope key found.")


# ── Main ─────────────────────────────────────────────────────────────

def _resolve_since(arg: str | None) -> str:
    if arg:
        return arg
    cleaned = cleaned_parquet()
    if cleaned.exists():
        df = pd.read_parquet(cleaned)
        if "publication_year" in df.columns and len(df):
            max_year = int(df["publication_year"].max())
            return f"{max_year}-01-01"
    return (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="YYYY-MM-DD; default = max year in existing parquet, or today-30")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Rebuild vectors.npy from scratch (slower, safer)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + count; do not write anything")
    args = ap.parse_args()

    since = _resolve_since(args.since)
    print(f"Fetching OpenAlex Works updated since {since} …", flush=True)

    new_records: list[dict] = []
    for w in _fetch_works_since(since):
        new_records.append(_project(w))

    print(f"  fetched {len(new_records):,} works", flush=True)
    if not new_records:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        print(f"DRY-RUN: would append {len(new_records):,} rows.")
        for r in new_records[:3]:
            print(f"  · {r['publication_year']} {r['title'][:80]}")
        return 0

    # Append to cleaned parquet (dedup by work_id)
    cleaned = cleaned_parquet()
    df_old = pd.read_parquet(cleaned)
    new_df = pd.DataFrame(new_records)
    merged = pd.concat([df_old, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["work_id"], keep="last")
    truly_new = len(merged) - len(df_old)
    print(f"  truly new (after dedup): {truly_new:,}")
    merged.to_parquet(cleaned, index=False)
    print(f"  wrote {cleaned}")

    # Append placeholder to classified parquet (mark "<unclassified>")
    classified = classified_parquet()
    if classified.exists():
        cls_df = pd.read_parquet(classified)
        existing_wids = set(cls_df["work_id"])
        new_cls = [
            {
                "work_id": r["work_id"],
                "categories": "[]",
                "primary_category": -1,
                "material": "<unclassified>",
                "confidence": 0.0,
                "parse_success": False,
                "raw_response": "",
            }
            for r in new_records
            if r["work_id"] and r["work_id"] not in existing_wids
        ]
        if new_cls:
            cls_df = pd.concat([cls_df, pd.DataFrame(new_cls)], ignore_index=True)
            cls_df.to_parquet(classified, index=False)
            print(f"  wrote {classified} (+{len(new_cls)} unclassified placeholders)")
            print("  → run upstream LLM classifier to label the new rows when convenient.")

    # Embedding index update
    if VECTORS_PATH.exists() and not args.rebuild_index:
        print(f"Embedding {len(new_records):,} new rows for in-place index extension …")
        api_key = _load_dashscope_key()
        new_vecs = _embed_new_rows(new_records, api_key)
        old_vecs = np.load(VECTORS_PATH)
        appended = np.concatenate([old_vecs, new_vecs], axis=0).astype(np.float32)
        np.save(VECTORS_PATH, appended)

        meta_df = pd.read_parquet(METADATA_PATH)
        new_meta = pd.DataFrame([
            {
                "work_id": r["work_id"], "doi": r["doi"], "title": r["title"],
                "year": r["publication_year"], "journal": r["journal"],
                "cited": r["cited_by_count"],
            }
            for r in new_records
        ])
        meta_df = pd.concat([meta_df, new_meta], ignore_index=True)
        meta_df.to_parquet(METADATA_PATH, index=False)
        print(f"  index now {len(appended):,} vectors × {appended.shape[1]} dims")
    elif args.rebuild_index:
        print("--rebuild-index given: re-run scripts/openalex_pipeline/03_build_embedding_index.py")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
