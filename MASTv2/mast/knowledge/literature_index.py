"""Literature search over the OpenAlex 36,792-paper embedding index (B2).

Index lives at:
    MASTv2/artifacts/literature_index/vectors.npy   (N, 1024) float32
    MASTv2/artifacts/literature_index/metadata.parquet
    MASTv2/artifacts/literature_index/abstracts.parquet   (optional enrichment)
    MASTv2/artifacts/literature_index/classified.parquet  (optional enrichment)
    MASTv2/artifacts/literature_index/manifest.json       (optional provenance)

Usage:
    >>> from mast.knowledge.literature_index import search
    >>> hits = search("Au(111) Kondo single magnetic atom", k=10)
    >>> for h in hits:
    ...     print(h["score"], h["title"], h["abstract_excerpt"][:80])

Both v1 and v2 read the same index files (path resolved off the repo root,
not the package). The vectors are loaded once into memory (~150 MB) on first
use and reused.

The query is embedded via DashScope `text-embedding-v3` (1024-d) using the
same model used to build the index. Numpy bruteforce cosine similarity over
36k vectors takes ~50 ms — fast enough to inline into chat tool calls.

Enrichment (design §2.1 / §4.8):
    `metadata.parquet` is the index-aligned hot table (row i ↔ vectors.npy[i])
    and carries only `work_id, doi, title, year, journal, cited` (+ an optional
    `source` column once provisioned). `abstracts.parquet` and
    `classified.parquet` are keyed by `work_id` and joined lazily ON DEMAND only
    when surfacing results, so the hot path (cosine + 6 small columns) is
    unchanged. All three enrichment files are OPTIONAL: if they are missing, the
    enrichment fields are gracefully omitted (or defaulted) and the search never
    crashes — abstracts simply aren't surfaced until the provisioning step
    (repo-root scripts/openalex_pipeline/12_provision_litlib.py) runs.

`manifest.json` may carry a `merge_map` ({"local:<hash>": "<work_id>"}) that
re-resolves user-contributed `local:<hash>` rows that were auto-merged into a
real OpenAlex `work_id` at re-provisioning time (Decision 1, design §3.2/§7.5).
`search()` applies this map so a hit that matched an old `local:<hash>` row
reports the merged canonical `work_id`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Index paths ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
# this file: {repo}/mast/knowledge/literature_index.py    OR
#            {repo}/MASTv2/mast/knowledge/literature_index.py
# The 205 MB index ships out-of-band (gitignored, NOT bundled in the PYZ), as a
# BINARY-adjacent asset under <base>/MASTv2/artifacts/literature_index/:
#   dev    → base = repo root (holds both mast/ and MASTv2/)
#   frozen → base = the EXECUTABLE's dir (C:\MAST2), where the install carries
#            MASTv2/artifacts/ — NOT project_root(), which may be a custom
#            data_dir.txt user-data location that has no index.
def _index_base() -> Path:
    """Delegates to the ONE shared resolver (``knowledge/paths.py``), which keeps
    exactly this rule — frozen ⇒ exe dir, dev ⇒ repo root — so five copies of the
    walk became one. Same answer as the walk it replaces (verified 2026-07-29)."""
    from mast.knowledge.paths import base_dir
    return base_dir()


# NOTE these five stay module-level constants, unlike the rest of the literature
# subsystem: ``_load_index`` and its caches close over them and the test suite
# monkeypatches ``_REPO_ROOT``. Consequence to know about: setting
# ``MAST_LITERATURE_INDEX_DIR`` **after** importing this module has no effect
# here (it does work for ``ingest`` / ``libraries`` / ``fetch_board``, which
# resolve per call). Set it before import, or patch ``_REPO_ROOT``.
_REPO_ROOT = _index_base()
_INDEX_DIR = _REPO_ROOT / "MASTv2" / "artifacts" / "literature_index"
VECTORS_PATH = _INDEX_DIR / "vectors.npy"
METADATA_PATH = _INDEX_DIR / "metadata.parquet"
ABSTRACTS_PATH = _INDEX_DIR / "abstracts.parquet"
CLASSIFIED_PATH = _INDEX_DIR / "classified.parquet"
MANIFEST_PATH = _INDEX_DIR / "manifest.json"

# Length cap for the abstract excerpt surfaced in search results (design §4.8).
_EXCERPT_MAX = 400
# Length cap for the FULL `abstract` field (the 2026-05-31 50k index ships
# abstracts inline in metadata.parquet; we surface a capped full abstract too).
_ABSTRACT_MAX = 2000

# DashScope embedding endpoint
_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
_MODEL = "text-embedding-v3"
_DIM = 1024


# ── Cache ────────────────────────────────────────────────────────────
_lock = threading.Lock()
_cache: dict[str, Any] = {
    "vectors": None,    # np.ndarray (N, 1024) float32, L2-normalised lazily
    "metadata": None,   # pd.DataFrame
    "norms_done": False,
}

# Enrichment caches (lazy, keyed by work_id) — loaded only on first need.
_enrich_lock = threading.Lock()
_enrich: dict[str, Any] = {
    "abstracts_by_id": None,   # dict[str, dict]  (work_id → abstract row dict)
    "abstracts_loaded": False,  # tri-state: have we *attempted* a load?
    "classified_by_id": None,  # dict[str, dict]  (work_id → classified row dict)
    "classified_loaded": False,
    "merge_map": None,         # dict[str, str]   (local:<hash> → work_id)
    "manifest_loaded": False,
    "meta_by_id": None,        # dict[str, dict]  (work_id → metadata.parquet row)
    "meta_by_id_loaded": False,
}

# ── Query-embedding cache (LRU) ──────────────────────────────────────
# Repeated/identical queries shouldn't each pay the remote DashScope round-trip
# (the dominant search latency). Keyed by (query, dim); a query's vector is
# stable for a fixed model, so this never needs busting on index change.
_QUERY_CACHE_MAX = 256
_query_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_qcache_lock = threading.Lock()


def _load_index() -> None:
    if _cache["vectors"] is not None:
        return
    with _lock:
        if _cache["vectors"] is not None:
            return
        if not VECTORS_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Literature index missing. Build it via:\n"
                f"  .venv/Scripts/python.exe scripts/openalex_pipeline/03_build_embedding_index.py\n"
                f"Expected at: {_INDEX_DIR}"
            )
        vec = np.load(VECTORS_PATH)
        meta = pd.read_parquet(METADATA_PATH)
        if vec.shape[0] != len(meta):
            raise RuntimeError(
                f"Index size mismatch: vectors={vec.shape[0]} metadata={len(meta)}"
            )
        # L2-normalise for cosine via dot
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vec = (vec / norms).astype(np.float32)
        _cache["vectors"] = vec
        _cache["metadata"] = meta
        _cache["norms_done"] = True
        logger.info("Loaded literature index: %d vectors × %d dims", vec.shape[0], vec.shape[1])


def _index_snapshot() -> tuple[np.ndarray, pd.DataFrame]:
    """Return a consistent ``(vectors, metadata)`` pair from the same generation.

    The big index cache can be invalidated concurrently (``invalidate_caches``
    nulls both fields under ``_lock``). A reader that grabs ``_cache["vectors"]``
    and ``_cache["metadata"]`` in two separate statements can therefore see a
    torn view: ``None`` (TypeError on ``vecs @ qv``) or two different
    generations (row misalignment). We re-load if needed and read both fields
    under a single lock hold so the returned pair is always coherent.
    """
    # Fast path: ensure something is loaded (idempotent; takes _lock internally).
    _load_index()
    with _lock:
        vecs = _cache["vectors"]
        meta = _cache["metadata"]
    if vecs is None or meta is None:
        # Invalidated between _load_index() and the lock grab — reload once more.
        _load_index()
        with _lock:
            vecs = _cache["vectors"]
            meta = _cache["metadata"]
    return vecs, meta


def corpus_size() -> int:
    """Live row count of the loaded index (0 if unavailable).

    Lets callers report the REAL corpus size instead of a hardcoded number that
    drifts as the index is regenerated (审查: the tool output said
    "36,792 papers" long after the index grew to ~50k). Best-effort: returns 0
    rather than raising if the index can't be loaded.
    """
    try:
        _load_index()
        vec = _cache.get("vectors")
        return int(vec.shape[0]) if vec is not None else 0
    except Exception:  # noqa: BLE001 — never let a count crash a tool call
        return 0


# ── Enrichment loaders (lazy, graceful) ──────────────────────────────

def _int_or_zero(v: Any) -> int:
    """Coerce a (possibly NaN/None) numeric cell to int, 0 when missing.

    The obvious ``int(v or 0)`` does NOT work here, and that is exactly how the
    field crash happened ( twice in one session)::

        bool(float('nan'))  ->  True          # NaN is truthy
        float('nan') or 0   ->  nan           # so the guard never fires
        int(nan)            ->  ValueError: cannot convert float NaN to integer

    ``or 0`` defends against None, which is not what a pandas column gives you
    for a missing year — it gives NaN. The whole search then died with a
    ValueError that surfaced to the agent as an opaque failure string, and its
    only recovery was to retry the same query.

    Companion to :func:`_str_or_empty`, which already did this for text columns.
    """
    if v is None:
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if f != f or f in (float("inf"), float("-inf")):   # NaN / ±inf
        return 0
    return int(f)


def _str_or_empty(v: Any) -> str:
    """Coerce a (possibly NaN/None) cell to a stripped str, '' when missing."""
    if v is None:
        return ""
    # pandas NaN is a float; guard with a scalar isna without importing extras
    try:
        if isinstance(v, float) and np.isnan(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v)
    if s.lower() == "nan":
        return ""
    return s


def _load_abstracts() -> dict[str, dict]:
    """Lazily build a `work_id → row dict` map from abstracts.parquet.

    Returns an EMPTY dict (never raises) if the file is absent or unreadable, so
    callers degrade gracefully to "no abstract surfaced". Loaded at most once;
    subsequent calls reuse the cache.
    """
    if _enrich["abstracts_loaded"]:
        return _enrich["abstracts_by_id"] or {}
    with _enrich_lock:
        if _enrich["abstracts_loaded"]:
            return _enrich["abstracts_by_id"] or {}
        by_id: dict[str, dict] = {}
        try:
            if ABSTRACTS_PATH.exists():
                df = pd.read_parquet(ABSTRACTS_PATH)
                if "work_id" in df.columns:
                    # to_dict('records') keeps column-name keys; index by work_id
                    for rec in df.to_dict("records"):
                        wid = _str_or_empty(rec.get("work_id"))
                        if wid:
                            by_id[wid] = rec
                    logger.info("Loaded literature abstracts: %d rows", len(by_id))
                else:
                    logger.warning(
                        "abstracts.parquet missing 'work_id' column; abstracts omitted"
                    )
            else:
                logger.debug("abstracts.parquet absent — abstracts omitted from results")
        except Exception as e:  # pragma: no cover - defensive, must not crash search
            logger.warning("Failed to load abstracts.parquet (%s); abstracts omitted", e)
            by_id = {}
        _enrich["abstracts_by_id"] = by_id
        _enrich["abstracts_loaded"] = True
        return by_id


def _load_classified() -> dict[str, dict]:
    """Lazily build a `work_id → row dict` map from classified.parquet.

    Empty dict (never raises) when the file is absent — the `material` filter
    then simply matches nothing extra and search still works on the base index.
    """
    if _enrich["classified_loaded"]:
        return _enrich["classified_by_id"] or {}
    with _enrich_lock:
        if _enrich["classified_loaded"]:
            return _enrich["classified_by_id"] or {}
        by_id: dict[str, dict] = {}
        try:
            if CLASSIFIED_PATH.exists():
                df = pd.read_parquet(CLASSIFIED_PATH)
                if "work_id" in df.columns:
                    for rec in df.to_dict("records"):
                        wid = _str_or_empty(rec.get("work_id"))
                        if wid:
                            by_id[wid] = rec
                    logger.info("Loaded literature classification: %d rows", len(by_id))
                else:
                    logger.warning(
                        "classified.parquet missing 'work_id' column; classification omitted"
                    )
            else:
                logger.debug("classified.parquet absent — material filter inert")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to load classified.parquet (%s); classification omitted", e)
            by_id = {}
        _enrich["classified_by_id"] = by_id
        _enrich["classified_loaded"] = True
        return by_id


def _load_merge_map() -> dict[str, str]:
    """Lazily read manifest.json's `merge_map` (local:<hash> → work_id).

    Empty dict (never raises) when manifest.json is absent or has no merge_map.
    """
    if _enrich["manifest_loaded"]:
        return _enrich["merge_map"] or {}
    with _enrich_lock:
        if _enrich["manifest_loaded"]:
            return _enrich["merge_map"] or {}
        merge_map: dict[str, str] = {}
        try:
            if MANIFEST_PATH.exists():
                data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                raw = data.get("merge_map") if isinstance(data, dict) else None
                if isinstance(raw, dict):
                    # keep only str→str entries
                    merge_map = {
                        str(k): str(v)
                        for k, v in raw.items()
                        if isinstance(k, str) and v
                    }
                    if merge_map:
                        logger.info("Loaded literature merge_map: %d entries", len(merge_map))
            else:
                logger.debug("manifest.json absent — no merge_map applied")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to read manifest.json merge_map (%s)", e)
            merge_map = {}
        _enrich["merge_map"] = merge_map
        _enrich["manifest_loaded"] = True
        return merge_map


def _resolve_work_id(work_id: str, merge_map: dict[str, str]) -> str:
    """Re-resolve a `local:<hash>` id through the merge_map (Decision 1).

    A merged `local:<hash>` reports its canonical OpenAlex work_id. Plain
    OpenAlex ids (and unmapped locals) pass through unchanged. Resolves
    transitively but is bounded against accidental cycles.
    """
    if not work_id or not merge_map:
        return work_id
    seen: set[str] = set()
    cur = work_id
    while cur in merge_map and cur not in seen:
        seen.add(cur)
        cur = merge_map[cur]
    return cur


# An OpenAlex work id: an uppercase/lowercase `W` followed by >=2 digits. The
# corpus parquet files key rows by the FULL URL form ("https://openalex.org/W…")
# but agents/users routinely pass the BARE id ("W…") — the tool docstrings even
# use the bare form as the example. An exact-keyed lookup must therefore accept
# both forms interchangeably.  (lib_search surfaces URL-form
# ids; the model echoed them back bare → fetch_abstract missed every row.)
_OPENALEX_ID_RE = re.compile(r"[Ww]\d{2,}")
#: 整串校验用的形态。``\d+``（而不是 ``\d{2,}``）是刻意的：配 ``fullmatch`` 用，
#: 不可能截断，所以下限放宽只会让**短 id 的双形去重也正确**（``W3`` 与
#: ``https://openalex.org/W3`` 此前会被存成两个成员，而「删除报成功却什么都没删」
#: 是很坏的形状）。真实 OpenAlex id 是 8–10 位，短 id 只出现在测试里 —— 但测试 id
#: 确实泄漏进过实机数据。
_OPENALEX_ID_FULL_RE = re.compile(r"[Ww]\d+")
_OPENALEX_URL_PREFIX = "https://openalex.org/"


def _workid_candidates(work_id: str) -> list[str]:
    """Return the equivalent key forms of *work_id* to try for an exact lookup.

    Order-preserving, de-duplicated. For an OpenAlex id this yields BOTH the
    bare canonical form ("W123…") and the full URL form
    ("https://openalex.org/W123…"), regardless of which one the caller passed,
    so a lookup against a table keyed by either form still hits. `local:<hash>`
    ids and anything without an embedded OpenAlex id pass through unchanged.
    """
    wid = (work_id or "").strip()
    if not wid:
        return []
    cands = [wid]
    # local:<hash> (and any non-OpenAlex id) must not be rewritten.
    if wid.lower().startswith("local:"):
        return cands
    m = _OPENALEX_ID_RE.search(wid)
    if m:
        bare = "W" + m.group(0)[1:]            # force canonical uppercase W…
        url = f"{_OPENALEX_URL_PREFIX}{bare}"
        for c in (bare, url):
            if c not in cands:
                cands.append(c)
    return cands


def canonical_work_id(work_id: str) -> str:
    """Collapse an OpenAlex id to its bare canonical form ("W…") for equality
    comparison, tolerating the full URL form. `local:<hash>` and any
    non-OpenAlex id are returned stripped-but-unchanged.

    Public helper so id comparisons across the curation layer (library
    membership) and the index agree on ONE form regardless of whether an id was
    supplied bare or as a URL — the two forms the corpus and its callers use
    interchangeably.
    """
    wid = (work_id or "").strip()
    if not wid or wid.lower().startswith("local:"):
        return wid
    # 只剥已知的 URL 前缀，然后**整串校验**。
    #
    # 原实现是 ``search`` + 取匹配段，对「看起来像但不是」的输入会**静默截断**：
    # ``canonical_work_id("W1998Barth") == "W1998"`` —— 后缀被丢掉，变成一个
    # **不同的** id。这在本轮成了关键路径：``experiment_library._fold`` 每次读
    # ``members.jsonl`` 都跑它，截断后的 id 会成为文件夹权威的 key，再传播进
    # registry；而 registry 是 gitignored、无备份、on-load migration 还会自己写回
    # 磁盘 —— 一次截断就不可逆，事后也无法诊断（2026-07-29 架构审查 M4）。
    #
    # 校验不过就**原样返回**：认不出来的 id 不该被改写成一个我们猜的形状。
    bare = wid
    for prefix in ("https://openalex.org/", "http://openalex.org/",
                   "https://api.openalex.org/works/", "openalex.org/"):
        if bare.lower().startswith(prefix):
            bare = bare[len(prefix):]
            break
    bare = bare.strip().strip("/")
    if _OPENALEX_ID_FULL_RE.fullmatch(bare):
        return "W" + bare[1:]
    return wid


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.I)


def work_id_for_doi(doi: str) -> str:
    """Resolve a DOI to work_id, or return an empty string when absent.

    Accept a bare DOI, a doi.org URL or a doi: prefix. Search results should
    expose work_id directly; this resolver also supports callers that only
    hold the identifier printed on a publication.
    """
    m = _DOI_RE.search(doi or "")
    if not m:
        return ""
    want = m.group(0).rstrip(".").lower()
    for wid, row in _metadata_by_id().items():
        cand = _str_or_empty(row.get("doi")).lower()
        if not cand:
            continue
        cm = _DOI_RE.search(cand)
        if cm and cm.group(0).rstrip(".") == want:
            return wid
    return ""


def _metadata_by_id() -> dict[str, dict]:
    """Lazy `work_id → metadata.parquet row dict` (incl. the inline `abstract`
    column on the 2026-05-31 50k index), loaded WITHOUT pulling the 205 MB
    vectors.npy into memory. Used by :func:`fetch_abstract` so a single lookup
    doesn't force a full index load. Degrades to {} if metadata is absent.
    """
    if _enrich["meta_by_id_loaded"]:
        return _enrich["meta_by_id"] or {}
    with _enrich_lock:
        if _enrich["meta_by_id_loaded"]:
            return _enrich["meta_by_id"] or {}
        by_id: dict[str, dict] = {}
        try:
            if METADATA_PATH.exists():
                df = pd.read_parquet(METADATA_PATH)
                if "work_id" in df.columns:
                    for rec in df.to_dict("records"):
                        wid = _str_or_empty(rec.get("work_id"))
                        if wid:
                            by_id[wid] = rec
                    logger.info("Loaded literature metadata-by-id: %d rows", len(by_id))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("metadata-by-id load failed (%s); fetch_abstract fallback off", e)
        _enrich["meta_by_id"] = by_id
        _enrich["meta_by_id_loaded"] = True
        return by_id


def _excerpt(text: str, limit: int = _EXCERPT_MAX) -> str:
    """Trim *text* to <= *limit* chars on a word/sentence-friendly boundary."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # prefer to break at the last whitespace so we don't slice mid-word
    sp = cut.rfind(" ")
    if sp >= limit * 0.6:   # only if it doesn't truncate too aggressively
        cut = cut[:sp]
    return cut.rstrip() + "…"


def _abstract_fields(row: dict | None, *, meta_abstract: str = "") -> dict:
    """Build the keep-both abstract fields (design §4.8 Decision 2).

    The BASE OpenAlex abstract now comes inline from the index-aligned
    ``metadata.parquet`` ``abstract`` column (the 2026-05-31 50k index ships
    abstracts inline — 87.8% coverage). ``abstracts.parquet``, when present,
    SUPPLEMENTS it with a user-contributed (keep-both) abstract + explicit
    provenance. ``meta_abstract`` is the metadata-column value for this row;
    ``row`` (if any) is the abstracts.parquet enrichment row.
    """
    openalex_abs = ""
    user_abs = ""
    if row is not None:
        # OpenAlex abstract lives in the `abstract` column.
        openalex_abs = _str_or_empty(row.get("abstract"))
        # User-contributed abstract: prefer `user_abstract`, fall back to
        # `fulltext_excerpt` (design §2.1 names both as the keep-both slot).
        user_abs = _str_or_empty(row.get("user_abstract")) or _str_or_empty(
            row.get("fulltext_excerpt")
        )
    # Inline-index fallback: the index-aligned metadata.parquet abstract column
    # is the base abstract when abstracts.parquet has none for this row.
    if not openalex_abs:
        openalex_abs = _str_or_empty(meta_abstract)

    # Provenance: trust an explicit column if present, else infer from content.
    prov = _str_or_empty(row.get("abstract_provenance")) if row is not None else ""
    if not prov:
        if openalex_abs and user_abs:
            prov = "openalex+user"
        elif user_abs:
            prov = "user"
        elif openalex_abs:
            prov = "openalex"
        else:
            prov = ""

    out: dict[str, Any] = {
        # Full (capped) abstract — surfaced so callers can read h["abstract"];
        # abstract_excerpt stays the short (≤400) preview for compact displays.
        "abstract": _excerpt(openalex_abs, _ABSTRACT_MAX) if openalex_abs else "",
        "abstract_excerpt": _excerpt(openalex_abs) if openalex_abs else "",
        "abstract_provenance": prov,
    }
    # Only surface the user excerpt key when a user abstract exists, so existing
    # callers and the common (base-only) case stay clean.
    if user_abs:
        out["user_abstract_excerpt"] = _excerpt(user_abs)
    return out


def _row_source(meta_row: pd.Series, abs_row: dict | None) -> str:
    """Resolve the `source` attribution for a result row (Decision 1).

    Order: metadata.parquet `source` column (index-aligned, authoritative) →
    abstracts.parquet `source` → default "openalex".
    """
    src = ""
    try:
        if "source" in meta_row.index:
            src = _str_or_empty(meta_row.get("source"))
    except Exception:
        src = ""
    if not src and abs_row is not None:
        src = _str_or_empty(abs_row.get("source"))
    return src or "openalex"


def _load_dashscope_key() -> str:
    """DashScope key via the CANONICAL ``config`` loader (env → ``dashscope.env``).

    This used to resolve ``api key/dashscope.env``
    off ``_REPO_ROOT`` — i.e. off :func:`_index_base`, which is deliberately the
    EXECUTABLE's directory in a frozen build because that is where the 205 MB
    index ships. But the key does not live there: the Inno installer writes a
    ``data_dir.txt`` pointing at a user-chosen data root, the launcher exports it
    as ``MAST2_PROJECT_ROOT``, and ``api key/`` lives under THAT root. So a
    normal install probed ``<install>/api key/dashscope.env``, found nothing, and
    every single search silently fell back to keyword matching — 68 times in one
    session, with a WARNING nobody reads and no UI signal at all.

    Index paths stay on ``_REPO_ROOT``; only the KEY moves to the shared
    resolver — the two genuinely live in different roots. See
    ``mast.knowledge._dashscope_key`` for the resolution order and for why the
    canonical ``config._load_provider_key`` alone is not sufficient.
    """
    from mast.knowledge._dashscope_key import require_dashscope_key

    return require_dashscope_key()


def _embed_query(query: str, *, timeout: float = 30.0) -> np.ndarray:
    """Embed *query* via DashScope text-embedding-v3 → (1024,) float32.

    LRU-cached so repeated/identical queries skip the remote round-trip (the
    dominant search latency). Raises on missing key / network error — callers
    (search) fall back to keyword retrieval rather than surfacing the exception.
    """
    ckey = (query, _DIM)
    with _qcache_lock:
        hit = _query_cache.get(ckey)
        if hit is not None:
            _query_cache.move_to_end(ckey)
            return hit.copy()
    key = _load_dashscope_key()
    body = {"model": _MODEL, "input": [query], "dimensions": _DIM}
    with httpx.Client(timeout=timeout) as c:
        r = c.post(
            _ENDPOINT,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
        )
        r.raise_for_status()
        data = r.json()
    vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    with _qcache_lock:
        _query_cache[ckey] = vec.copy()
        _query_cache.move_to_end(ckey)
        while len(_query_cache) > _QUERY_CACHE_MAX:
            _query_cache.popitem(last=False)
    return vec


def _keyword_search(
    query: str,
    k: int,
    *,
    year_min: int = 0,
    year_max: int = 9999,
    material: str = "",
    include_abstract: bool = True,
) -> list[dict]:
    """Backwards-compatible wrapper — hits only, diagnostics discarded."""
    hits, _diag = _keyword_search_diag(
        query, k, year_min=year_min, year_max=year_max,
        material=material, include_abstract=include_abstract)
    return hits


def _keyword_search_diag(
    query: str,
    k: int,
    *,
    year_min: int = 0,
    year_max: int = 9999,
    material: str = "",
    include_abstract: bool = True,
) -> tuple[list[dict], dict]:
    """Offline DEGRADED retrieval for when the embed endpoint is unavailable
    (no DashScope key / network down) — keeps search usable instead of raising.

    Ranks index rows by how many query terms occur in the title (weighted ×2)
    and the inline abstract (×1). Returns the SAME result schema as
    :func:`search`, plus ``degraded=True`` / ``retrieval="keyword"`` so callers
    can flag it in the UI. Never raises (returns [] when there is no usable
    text or no term match).

    Also returns a diagnostics dict so a caller can tell the difference between
    "keyword retrieval worked" and "keyword retrieval produced noise". A Chinese query
    ``'STM 偏压依赖成像 表面态 电子结构'`` split into four terms of which only
    ``'stm'`` occurs in this English corpus, so 50k papers tied at score 3.000
    and the top 8 were an arbitrary slice — yet the agent read them as
    "较相关". The tie and the unmatched CJK terms are both knowable HERE; the
    diagnostics carry them out.

    Diagnostics keys:
        ``matched_terms`` / ``unmatched_terms`` — query terms that do / don't
        occur anywhere in the corpus text;
        ``degenerate_ranking`` — every returned hit shares one score, i.e. the
        ordering carries no information;
        ``tied_corpus_rows`` — how many index rows share that same top score.
    """
    diag: dict[str, Any] = {
        "terms": [], "matched_terms": [], "unmatched_terms": [],
        "degenerate_ranking": False, "tied_corpus_rows": 0,
    }
    _, meta = _index_snapshot()
    if meta is None or len(meta) == 0:
        return [], diag
    terms = [t for t in re.split(r"[^0-9A-Za-z一-鿿]+", (query or "").lower())
             if len(t) >= 2]
    diag["terms"] = list(terms)
    if not terms:
        return [], diag
    title_l = meta["title"].fillna("").astype(str).str.lower()
    abs_l = (meta["abstract"].fillna("").astype(str).str.lower()
             if "abstract" in meta.columns else None)
    score = pd.Series(0.0, index=meta.index)
    for t in terms:
        in_title = title_l.str.contains(t, regex=False)
        in_abs = abs_l.str.contains(t, regex=False) if abs_l is not None else None
        # A term that occurs NOWHERE contributes nothing to the ranking. With a
        # Chinese query against an English corpus that is most of the query, and
        # what is left cannot discriminate — record it rather than hide it.
        hit_any = bool(in_title.any()) or bool(in_abs is not None and in_abs.any())
        (diag["matched_terms"] if hit_any else diag["unmatched_terms"]).append(t)
        score = score + in_title.astype(float) * 2.0
        if in_abs is not None:
            score = score + in_abs.astype(float)
    if year_min > 0 or year_max < 9999:
        yr = meta["year"].fillna(0).astype(int)
        score = score.where((yr >= year_min) & (yr <= year_max), 0.0)
    material_q = (material or "").strip().lower()
    classified = _load_classified() if material_q else {}
    merge_map = _load_merge_map()
    out: list[dict] = []
    seen: set[str] = set()
    for idx, sc in score.sort_values(ascending=False).items():
        if len(out) >= k or sc <= 0:
            break
        row = meta.loc[idx]
        raw_id = _str_or_empty(row.get("work_id"))
        work_id = _resolve_work_id(raw_id, merge_map)
        dedup_key = work_id or raw_id
        if dedup_key and dedup_key in seen:
            continue
        if material_q:
            cls = classified.get(work_id) or classified.get(raw_id)
            cat = _str_or_empty(cls.get("primary_category")).lower() if cls else ""
            mat = _str_or_empty(cls.get("material")).lower() if cls else ""
            if material_q not in cat and material_q not in mat:
                continue
        result: dict[str, Any] = {
            "score": round(float(sc), 4),
            "title": _str_or_empty(row.get("title")),
            "year": _int_or_zero(row.get("year")),
            "journal": _str_or_empty(row.get("journal")),
            "doi": _str_or_empty(row.get("doi")),
            "work_id": work_id,
            "cited": _int_or_zero(row.get("cited")),
            "source": "openalex",
            "degraded": True,
            "retrieval": "keyword",
        }
        if include_abstract:
            # Same key surface as the semantic path (_abstract_fields): callers
            # read h["abstract"] and must not KeyError just because the embedder
            # is down and we degraded to keyword retrieval.
            ab = _str_or_empty(row.get("abstract"))
            result["abstract"] = _excerpt(ab, _ABSTRACT_MAX) if ab else ""
            result["abstract_excerpt"] = _excerpt(ab) if ab else ""
            result["abstract_provenance"] = "openalex" if ab else ""
        if dedup_key:
            seen.add(dedup_key)
        out.append(result)

    # Degenerate ranking: every hit we are about to return carries the SAME
    # score, so their ORDER is arbitrary — the "top 8" is a slice of whatever
    # `sort_values` happened to put first among thousands of equals. Count how
    # many corpus rows sit at that score so the caller can say how arbitrary.
    if len(out) > 1:
        top = out[0]["score"]
        if all(abs(h["score"] - top) < 1e-9 for h in out):
            diag["degenerate_ranking"] = True
            try:
                diag["tied_corpus_rows"] = int((score == top).sum())
            except Exception:  # pragma: no cover - defensive
                diag["tied_corpus_rows"] = len(out)
    return out, diag


# ── Public API ───────────────────────────────────────────────────────

def search(
    query: str,
    k: int = 20,
    *,
    year_min: int = 0,
    year_max: int = 9999,
    material: str = "",
    include_abstract: bool = True,
) -> list[dict]:
    """Return top-*k* OpenAlex papers for *query*.

    Each result is a JSON-serializable dict:
        {"score", "title", "year", "journal", "doi", "work_id", "cited",
         "source",                                  # Decision 1 attribution
         "abstract_excerpt",                        # ≤400 chars, "" if none
         "abstract_provenance",                     # openalex|user|openalex+user|""
         "user_abstract_excerpt"?,                  # only when a user abstract exists
         "matched_abstract"?}                       # only when ambiguous

    The enrichment fields (`abstract_excerpt`, `source`, `abstract_provenance`,
    `user_abstract_excerpt`) come from a lazy left-join against
    `abstracts.parquet` keyed by `work_id`. If that file is absent the abstract
    fields are omitted/empty and the call still returns the base result set
    (backward compatible: pre-existing callers ignore the extra keys).

    Args:
        query:    natural language query (English or Chinese; embedding is
                  multilingual)
        k:        number of results
        year_min, year_max: optional filter on publication_year (inclusive)
        material: optional filter — keep only papers whose `classified.parquet`
                  `primary_category`/`material` matches (case-insensitive
                  substring). Inert if classified.parquet is absent.
        include_abstract: when False, skip the abstract left-join entirely
                  (cheaper; surfaces only base + source fields).

    NOTE: this thin wrapper DROPS the retrieval status. Anything that shows
    results to a human or feeds them to an LLM should call
    :func:`search_with_status` instead — otherwise a degraded keyword slice is
    indistinguishable from a semantic ranking.
    """
    hits, _status = search_with_status(
        query, k, year_min=year_min, year_max=year_max,
        material=material, include_abstract=include_abstract)
    return hits


def search_with_status(
    query: str,
    k: int = 20,
    *,
    year_min: int = 0,
    year_max: int = 9999,
    material: str = "",
    include_abstract: bool = True,
) -> tuple[list[dict], dict]:
    """:func:`search`, plus an explicit account of HOW the results were obtained.

    The semantic embedder was unavailable for an
    entire session (68 fallbacks). Degrading instead of raising was right; doing
    it *invisibly* was not. The only trace was a WARNING in a log nobody reads —
    the returned rows looked exactly like semantic hits (title, DOI, citation
    count), so the literature agent read a tied-score keyword slice and reported
    "第一批结果较相关". Callers that show results to a human or an LLM must use
    THIS function and surface ``status`` alongside the hits.

    ``status`` keys:
        ``retrieval``          "semantic" | "keyword"
        ``degraded``           True when the semantic path was unavailable
        ``reason``             why it degraded ("" when not degraded)
        ``trustworthy``        False when the ranking carries no information —
                               either degenerate (all scores tied) or built from
                               query terms that mostly don't occur in the corpus
        ``matched_terms`` / ``unmatched_terms`` / ``degenerate_ranking`` /
        ``tied_corpus_rows``   keyword-path diagnostics (see
                               :func:`_keyword_search_diag`)
    """
    status: dict[str, Any] = {
        "retrieval": "semantic", "degraded": False, "reason": "",
        "trustworthy": True, "matched_terms": [], "unmatched_terms": [],
        "degenerate_ranking": False, "tied_corpus_rows": 0,
    }
    if not query or not query.strip():
        return [], status
    _load_index()
    try:
        qv = _embed_query(query)
    except Exception as exc:  # missing key / network → degraded keyword search
        logger.warning("semantic embed unavailable (%s); using keyword fallback", exc)
        hits, diag = _keyword_search_diag(
            query, k, year_min=year_min, year_max=year_max,
            material=material, include_abstract=include_abstract)
        status.update(diag)
        status["retrieval"] = "keyword"
        status["degraded"] = True
        status["reason"] = str(exc)
        # A ranking is worthless when its order is arbitrary (everything tied),
        # or when the terms that survived are a minority of what was asked for
        # (a Chinese query against an English corpus: 1 of 4 terms matched).
        n_terms = len(diag.get("terms") or ())
        n_matched = len(diag.get("matched_terms") or ())
        status["trustworthy"] = bool(
            hits
            and not diag.get("degenerate_ranking")
            and (n_terms == 0 or n_matched * 2 >= n_terms)
        )
        return hits, status

    # Take a CONSISTENT (vecs, meta) snapshot. `_embed_query` above made a remote
    # round-trip; during it `invalidate_caches()` may have nulled the cache (→
    # `vecs @ qv` TypeError on None) or swapped in a different generation (→
    # vecs/meta row misalignment). Re-load (idempotent) and grab both under one
    # lock so the pair is the same generation; never operate on a torn cache.
    vecs, meta = _index_snapshot()

    merge_map = _load_merge_map()
    abstracts = _load_abstracts() if include_abstract else {}
    # classified is only needed when filtering by material
    classified = _load_classified() if material and material.strip() else {}
    material_q = material.strip().lower()

    # Cosine similarity (vecs is L2-normalised)
    sims = vecs @ qv
    # Apply year filter if any
    if year_min > 0 or year_max < 9999:
        mask = ((meta["year"].fillna(0).astype(int) >= year_min) &
                (meta["year"].fillna(9999).astype(int) <= year_max))
        sims = np.where(mask.values, sims, -1.0)

    # We over-fetch when a material filter is active (post-join drops rows), so
    # the requested k can still be filled. Bounded to the corpus size.
    fetch_k = k
    if material_q:
        fetch_k = min(len(sims), max(k * 8, k + 50))

    # Top-k via argpartition + sort
    if fetch_k >= len(sims):
        idx = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, fetch_k)[:fetch_k]
        idx = part[np.argsort(-sims[part])]

    out: list[dict] = []
    seen_ids: set[str] = set()
    for i in idx:
        if len(out) >= k:
            break
        score = float(sims[int(i)])
        if score <= -0.5:
            continue
        row = meta.iloc[int(i)]

        raw_id = _str_or_empty(row.get("work_id"))
        # Re-resolve a merged local:<hash> to its canonical OpenAlex work_id.
        work_id = _resolve_work_id(raw_id, merge_map)

        # Dedup to one entry per (resolved) work_id — keep-both can produce two
        # vector rows pointing at the same work_id (§4.8).
        dedup_key = work_id or raw_id
        if dedup_key and dedup_key in seen_ids:
            continue

        abs_row = abstracts.get(work_id) or abstracts.get(raw_id) if abstracts else None

        # Material filter (join classified.parquet): keep only matches.
        if material_q:
            cls = classified.get(work_id) or classified.get(raw_id)
            cat = ""
            mat = ""
            if cls is not None:
                cat = _str_or_empty(cls.get("primary_category")).lower()
                mat = _str_or_empty(cls.get("material")).lower()
            if material_q not in cat and material_q not in mat:
                continue

        result: dict[str, Any] = {
            "score":    round(score, 4),
            "title":    _str_or_empty(row.get("title")),
            "year":     _int_or_zero(row.get("year")),
            "journal":  _str_or_empty(row.get("journal")),
            "doi":      _str_or_empty(row.get("doi")),
            "work_id":  work_id,
            "cited":    _int_or_zero(row.get("cited")),
            "source":   _row_source(row, abs_row),
        }

        if include_abstract:
            af = _abstract_fields(abs_row, meta_abstract=_str_or_empty(row.get("abstract")))
            result.update(af)
            # matched_abstract: only meaningful when both abstracts exist.
            if af.get("abstract_provenance") == "openalex+user":
                # We cannot tell from a base left-join which vector row matched;
                # default to "openalex" (the base block vector). The promotion
                # path / per-row kind marker would refine this when present.
                kind = ""
                if abs_row is not None:
                    kind = _str_or_empty(abs_row.get("kind"))
                result["matched_abstract"] = "user" if kind == "user_abstract" else "openalex"

        if dedup_key:
            seen_ids.add(dedup_key)
        out.append(result)
    return out, status


def fetch_abstract(work_id: str) -> dict:
    """Exact `work_id` lookup into `abstracts.parquet` (design §4.2).

    Returns the full abstract + bibliographic context for a single paper,
    independent of the vector index — so it works even for the ~38 papers that
    are in the corpus but absent from `vectors.npy`.

    Returns a dict:
        {"work_id", "found": bool, "abstract", "user_abstract",
         "abstract_provenance", "authors", "first_author", "year"?, "journal"?,
         "concepts", "keywords", "work_type", "source", "cited_by_count"?,
         "primary_category"?, "material"?}

    Graceful degradation:
        - abstracts.parquet absent  → {"work_id", "found": False, "abstract": "",
          "note": "abstracts.parquet not provisioned"}
        - work_id unknown           → {"work_id", "found": False, "abstract": ""}

    Read-only; no state change; never raises on missing data.
    """
    wid = (work_id or "").strip()
    if not wid:
        return {"work_id": "", "found": False, "abstract": ""}

    # Re-resolve a merged local:<hash> to canonical work_id first.
    merge_map = _load_merge_map()
    resolved = _resolve_work_id(wid, merge_map)

    abstracts = _load_abstracts()

    # Candidate key forms (bare ↔ URL) for BOTH the merge-resolved id and the
    # original, so an exact lookup hits whether the corpus keys rows by the bare
    # "W…" id or the full "https://openalex.org/W…" URL (the
    # 50k corpus keys every row by the URL form, but agents pass the bare id).
    cand_keys: list[str] = []
    for base in (resolved, wid):
        for c in _workid_candidates(base):
            if c not in cand_keys:
                cand_keys.append(c)

    def _first_hit(table: dict[str, dict]) -> dict | None:
        for c in cand_keys:
            r = table.get(c)
            if r is not None:
                return r
        return None

    row = _first_hit(abstracts) if abstracts else None

    if row is None:
        # Fallback to the index-aligned metadata.parquet `abstract` column — the
        # SAME source lib_search draws its excerpt from (the 2026-05-31 50k index
        # ships abstracts inline; abstracts.parquet may also simply lack a row
        # the vector index has). This makes fetch_abstract answer for every paper
        # the search can surface, MARKED so the caller knows the abstract came
        # from the index-aligned metadata rather than the full abstracts table.
        meta_by_id = _metadata_by_id()
        mrow = _first_hit(meta_by_id)
        if mrow is not None:
            abs_text = _str_or_empty(mrow.get("abstract"))
            mout: dict[str, Any] = {
                "work_id": resolved, "found": True,
                "abstract": abs_text, "user_abstract": "",
                "abstract_provenance": "openalex" if abs_text else "",
                "authors": "", "first_author": "", "concepts": "",
                "keywords": "", "work_type": "", "source": "openalex",
                "title": _str_or_empty(mrow.get("title")),
                # Provenance marker: index-aligned metadata excerpt (lib_search's
                # source), NOT the richer abstracts enrichment table.
                "abstract_source": "index_metadata",
            }
            # _int_or_zero, not `int(… or 0)`: this cell comes straight off
            # metadata.parquet, where a missing count is NaN, and NaN is truthy
            # — so `or 0` never fires and int() raises. The try/except turned
            # that into a SILENT DROP of cited_by_count rather than a crash,
            # which is why it survived the 2026-07-27 fix of the same shape.
            mout["cited_by_count"] = _int_or_zero(mrow.get("cited"))
            for opt in ("year", "journal", "doi"):
                v = _str_or_empty(mrow.get(opt))
                if v:
                    mout[opt] = v
            cls = _first_hit(_load_classified())
            if cls is not None:
                pc = _str_or_empty(cls.get("primary_category"))
                mt = _str_or_empty(cls.get("material"))
                if pc:
                    mout["primary_category"] = pc
                if mt:
                    mout["material"] = mt
            return mout
        # Neither enrichment source has it.
        note = ({"note": "literature index not provisioned"}
                if not abstracts and not meta_by_id else {})
        return {"work_id": resolved, "found": False, "abstract": "", **note}

    out: dict[str, Any] = {
        "work_id":             resolved,
        "found":               True,
        "abstract":            _str_or_empty(row.get("abstract")),
        "user_abstract":       _str_or_empty(row.get("user_abstract"))
                               or _str_or_empty(row.get("fulltext_excerpt")),
        "abstract_provenance": _str_or_empty(row.get("abstract_provenance")),
        "authors":             _str_or_empty(row.get("authors")),
        "first_author":        _str_or_empty(row.get("first_author")),
        "concepts":            _str_or_empty(row.get("concepts")),
        "keywords":            _str_or_empty(row.get("keywords")),
        "work_type":           _str_or_empty(row.get("work_type")),
        "source":              _str_or_empty(row.get("source")) or "openalex",
        # Provenance marker: the full abstracts enrichment table (rich fields).
        "abstract_source":     "abstracts_table",
    }
    # Optional numeric/context columns when present.
    if "cited_by_count" in row:
        out["cited_by_count"] = _int_or_zero(row.get("cited_by_count"))
    # `title` is carried on abstracts.parquet rows (OpenAlex + user-contributed /
    # manual entries) — surface it so a hand-entered paper round-trips with its
    # title, not just its abstract .
    for opt in ("title", "year", "journal"):
        v = _str_or_empty(row.get(opt))
        if v:
            out[opt] = v

    # Fill in provenance if the column was empty but content tells us.
    if not out["abstract_provenance"]:
        if out["abstract"] and out["user_abstract"]:
            out["abstract_provenance"] = "openalex+user"
        elif out["user_abstract"]:
            out["abstract_provenance"] = "user"
        elif out["abstract"]:
            out["abstract_provenance"] = "openalex"

    # Best-effort classification join (optional).
    cls = _first_hit(_load_classified())
    if cls is not None:
        pc = _str_or_empty(cls.get("primary_category"))
        mt = _str_or_empty(cls.get("material"))
        if pc:
            out["primary_category"] = pc
        if mt:
            out["material"] = mt
    return out


def invalidate_caches() -> None:
    """Drop the in-memory enrichment caches (abstracts/classified/merge_map).

    Called by the promotion path (design §7.5 step 4) after appending a
    user-contributed paper so the new abstract / merge_map becomes visible
    without a process restart. Cheap: the next search/fetch re-reads the parquet.

    The big ~150 MB `vectors.npy`/`metadata.parquet` cache is also dropped so a
    grown tail is reloaded on the next search.
    """
    with _enrich_lock:
        _enrich["abstracts_by_id"] = None
        _enrich["abstracts_loaded"] = False
        _enrich["classified_by_id"] = None
        _enrich["classified_loaded"] = False
        _enrich["merge_map"] = None
        _enrich["manifest_loaded"] = False
        _enrich["meta_by_id"] = None
        _enrich["meta_by_id_loaded"] = False
    with _lock:
        _cache["vectors"] = None
        _cache["metadata"] = None
        _cache["norms_done"] = False


def ensure_manifest() -> dict:
    """Write a provenance ``manifest.json`` (with an empty ``merge_map``) next to
    the index if one is absent, so the build script's copy isn't a no-op and the
    frozen app ships provenance + a merge_map slot. A later re-provisioning run
    overwrites ``merge_map`` with real ``local:<hash>`` → ``work_id`` links.

    Returns the manifest dict (existing one if already present). Never raises.
    """
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - corrupt → rewrite below
            pass
    n = 0
    try:
        n = int(len(pd.read_parquet(METADATA_PATH)))
    except Exception:  # pragma: no cover
        pass
    manifest = {
        "source_repo": "openalex-stm",
        "model": _MODEL,
        "dim": _DIM,
        "n_papers": n,
        "merge_map": {},
        "note": ("provenance + merge_map; merge_map stays empty until a "
                 "re-provisioning run links user-contributed local:<hash> rows "
                 "to canonical OpenAlex work_ids."),
    }
    try:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, MANIFEST_PATH)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("ensure_manifest write failed: %s", exc)
    return manifest


def index_stats() -> dict:
    """Return index size + path info; loads on demand."""
    _load_index()
    vec: np.ndarray = _cache["vectors"]
    meta: pd.DataFrame = _cache["metadata"]
    return {
        "n_papers": int(len(vec)),
        "dim": int(vec.shape[1]),
        "model": _MODEL,
        "vectors_path": str(VECTORS_PATH),
        "metadata_path": str(METADATA_PATH),
        "abstracts_present": bool(ABSTRACTS_PATH.exists()),
        "classified_present": bool(CLASSIFIED_PATH.exists()),
        # _int_or_zero on both ends: on an empty or all-missing `year` column
        # pandas returns NaN (numpy) or pd.NA (Arrow) and bare int() raises —
        # which would take out the whole stats endpoint over a cosmetic field.
        "year_range": [_int_or_zero(meta["year"].min()),
                       _int_or_zero(meta["year"].max())],
    }


__all__ = ["search", "search_with_status", "fetch_abstract", "index_stats",
           "invalidate_caches", "canonical_work_id"]
