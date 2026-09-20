"""B1: build a literature embedding index over the 36,830 cleaned STM papers.

Embeddings are computed via DashScope `text-embedding-v3` (1024-d) through the
OpenAI-compatible /v1/embeddings endpoint. DashScope limits inputs to **10 per
call**, so we run with batch=10 and 8 concurrent worker threads.

Expected runtime: ~36830/10 = 3683 sequential calls / 8 workers ≈ 460 batches,
~2-3 s/batch = 15-25 min on a typical connection.

Output:
    MASTv2/artifacts/literature_index/vectors.npy        # (N, 1024) float32
    MASTv2/artifacts/literature_index/metadata.parquet   # work_id, doi, title, year, journal, cited
    MASTv2/artifacts/literature_index/_progress.json     # checkpoint for resume

Resume: re-running the script picks up where it left off.

CLI:
    python 03_build_embedding_index.py [--batch 25] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mast.knowledge.openalex_loader import load_cleaned

# ── Output paths ──────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parents[2] / "MASTv2" / "artifacts" / "literature_index"
VECTORS_PATH = OUT_DIR / "vectors.npy"
METADATA_PATH = OUT_DIR / "metadata.parquet"
PROGRESS_PATH = OUT_DIR / "_progress.json"

# ── DashScope endpoint ────────────────────────────────────────────────
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
MODEL = "text-embedding-v3"
DIM = 1024
MAX_INPUT_LEN = 8000   # chars; stay below tokenizer limits


def _load_key() -> str:
    """Load DashScope key (same logic as v1 mast.config / v2 MASTv2 config)."""
    key = (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("ALIYUN_BAILIAN_API_KEY", "").strip()
    )
    if key:
        return key
    p = Path(__file__).resolve().parents[2] / "api key" / "dashscope.env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise RuntimeError("No DashScope key found. Set DASHSCOPE_API_KEY or write api key/dashscope.env.")


def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Project the parquet to the rows + columns we want to embed."""
    cols_we_need = [
        "work_id", "doi", "title", "abstract", "publication_year",
        "journal", "cited_by_count",
    ]
    keep = [c for c in cols_we_need if c in df.columns]
    proj = df[keep].copy()
    proj = proj.rename(columns={"publication_year": "year", "cited_by_count": "cited"})
    # Build the text we embed: title + first 6000 chars of abstract
    def _make_text(row):
        t = (row.get("title") or "")
        a = (row.get("abstract") or "")
        if isinstance(t, float):
            t = ""
        if isinstance(a, float):
            a = ""
        s = f"{str(t).strip()}\n\n{str(a).strip()[:6000]}"
        return s.strip()[:MAX_INPUT_LEN]
    proj["embed_text"] = proj.apply(_make_text, axis=1)
    # drop rows with no text at all
    proj = proj[proj["embed_text"].str.len() > 30].reset_index(drop=True)
    return proj


def _embed_batch(client: httpx.Client, texts: list[str], retries: int = 5) -> np.ndarray:
    body = {"model": MODEL, "input": texts, "dimensions": DIM}
    last_exc = None
    for attempt in range(retries):
        try:
            r = client.post(ENDPOINT, json=body)
            if r.status_code == 429:
                # rate limited — backoff
                wait = 2 ** attempt + 1
                print(f"    HTTP 429, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            vecs = np.array(
                [d["embedding"] for d in data["data"]],
                dtype=np.float32,
            )
            assert vecs.shape == (len(texts), DIM), f"unexpected shape {vecs.shape}"
            return vecs
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"    error: {type(e).__name__}: {str(e)[:100]} (retry in {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"embedding failed after {retries} retries: {last_exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10, help="inputs per /embeddings call (DashScope max 10)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent worker threads")
    ap.add_argument("--limit", type=int, default=0, help="cap rows for testing (0=all)")
    ap.add_argument("--every", type=int, default=500, help="checkpoint every N rows")
    args = ap.parse_args()

    key = _load_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading stm_papers.parquet …", flush=True)
    df = load_cleaned()
    print(f"  raw rows: {len(df):,}", flush=True)
    proj = prepare_corpus(df)
    if args.limit > 0:
        proj = proj.head(args.limit)
    n = len(proj)
    print(f"  rows to embed: {n:,}", flush=True)
    print(f"  batches: {(n + args.batch - 1) // args.batch}", flush=True)

    # Resume: load progress + existing vectors
    start_row = 0
    vectors: np.ndarray = np.zeros((n, DIM), dtype=np.float32)
    if PROGRESS_PATH.exists() and VECTORS_PATH.exists():
        progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        if progress.get("total") == n and progress.get("model") == MODEL:
            done = int(progress.get("done", 0))
            existing = np.load(VECTORS_PATH)
            if existing.shape == (n, DIM):
                vectors = existing
                start_row = done
                print(f"  RESUMING from row {start_row:,}", flush=True)

    # Save metadata once (independent of progress)
    if not METADATA_PATH.exists():
        meta_cols = [c for c in ("work_id","doi","title","year","journal","cited") if c in proj.columns]
        proj[meta_cols].to_parquet(METADATA_PATH, index=False)
        print(f"  wrote metadata: {METADATA_PATH}", flush=True)

    # ── Concurrent embedding with thread pool ──
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Each worker has its own httpx client (httpx.Client is thread-safe but
    # connection pooling is more efficient with a small per-thread client).
    thread_local = threading.local()

    def _get_client() -> httpx.Client:
        c = getattr(thread_local, "client", None)
        if c is None:
            c = httpx.Client(
                timeout=httpx.Timeout(120.0, connect=30.0),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            thread_local.client = c
        return c

    def _process(batch_start: int) -> tuple[int, int, np.ndarray]:
        batch_end = min(batch_start + args.batch, n)
        texts = proj["embed_text"].iloc[batch_start:batch_end].tolist()
        vecs = _embed_batch(_get_client(), texts)
        return batch_start, batch_end, vecs

    batch_starts = list(range(start_row, n, args.batch))
    completed = 0
    t0 = time.time()
    save_lock = threading.Lock()
    last_save = start_row

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_process, bs) for bs in batch_starts]
        for fut in as_completed(futures):
            try:
                batch_start, batch_end, vecs = fut.result()
            except Exception as e:
                print(f"  ERROR in batch: {e}", flush=True)
                continue
            with save_lock:
                vectors[batch_start:batch_end] = vecs
                completed += (batch_end - batch_start)
                done_total = start_row + completed
                if done_total - last_save >= args.every or completed == len(batch_starts) * args.batch:
                    np.save(VECTORS_PATH, vectors)
                    PROGRESS_PATH.write_text(
                        json.dumps({"total": n, "done": done_total, "model": MODEL, "dim": DIM}),
                        encoding="utf-8",
                    )
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (n - done_total) / max(rate, 1e-6)
                    print(
                        f"  {done_total:,}/{n:,}  ({100*done_total/n:.1f}%)  "
                        f"rate={rate:.1f}/s  eta={eta/60:.1f}min",
                        flush=True,
                    )
                    last_save = done_total

    # Final save
    np.save(VECTORS_PATH, vectors)
    PROGRESS_PATH.write_text(
        json.dumps({"total": n, "done": n, "model": MODEL, "dim": DIM, "status": "done"}),
        encoding="utf-8",
    )
    print(f"\nDone. {n:,} vectors → {VECTORS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
