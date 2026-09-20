"""Searching *inside* the full texts we already hold.

``ingest_pdf`` has always written, for every paper it takes in:

    <papers>/<slug>/chunks.parquet      chunk_id, page, char_start, char_end, text, n_tokens
    <papers>/<slug>/chunk_vectors.npy   (n_chunks, 1024) float32, L2-normalised
    <papers>/<slug>/abstract_vector.npy (1, 1024)

embedded with the same DashScope model as the big index. Nothing ever read them.
So the only ways to consult a paper's body were a regex over a heading
(``read_paper_section``) or a full LLM read of the whole thing
(``deep_read_papers``) — one too blunt to find anything phrased unexpectedly, the
other costing a minute and tens of thousands of tokens per paper.

This module is the missing middle: ask a question, get back the handful of
passages that answer it, across one paper or every paper on disk. The vectors are
already there and already normalised, so a search is one query embedding plus a
dot product.

Granularity, so callers know what a "passage" is: despite the 800-token/120-overlap
parameter names, ``chunk_text`` splits at page and heading boundaries first, and a
page-section almost always fits inside one window. Chunks therefore come out
page-section sized and **do not overlap**, which makes each hit a coherent unit
but means a sentence spanning a page break is split across two chunks.

Honesty contract, inherited from ``literature_index.search_with_status`` and for
the same reason: when the embedder is unavailable this degrades to keyword
matching, and degraded results look exactly like semantic ones. Callers that show
results to a human or an LLM must surface ``status`` alongside the hits.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = ["available_papers", "search_chunks", "paper_title", "clear_cache"]

#: How many papers' vectors to keep in memory. A paper is a few hundred KB of
#: float32, so this is bounded at tens of MB.
_CACHE_MAX = 64
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_cache_lock = threading.Lock()

#: Directory names under <papers> that are not papers (see ``literature/tools``).
_NON_PAPER_DIRS = {"attachments", "_incoming"}


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _papers_dir() -> "Path | None":
    try:
        from mast.knowledge.paths import papers_dir
        d = papers_dir()
        return d if d.is_dir() else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("papers_dir unavailable: %s", exc)
        return None


def available_papers() -> list[str]:
    """Slugs of every paper whose full text has been chunked and embedded."""
    base = _papers_dir()
    if base is None:
        return []
    out: list[str] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name in _NON_PAPER_DIRS:
            continue
        if (d / "chunks.parquet").is_file() and (d / "chunk_vectors.npy").is_file():
            out.append(d.name)
    return out


def paper_title(slug: str) -> str:
    base = _papers_dir()
    if base is None:
        return ""
    try:
        import json
        meta = base / slug / "meta.json"
        if meta.is_file():
            return str(json.loads(meta.read_text(encoding="utf-8")).get("title", "") or "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _text(v) -> str:
    """A parquet cell as text. None / NaN / pd.NA → ``""``.

    ``str(v or "")`` does NOT do this and the difference is not cosmetic:
    ``float('nan')`` is TRUTHY, so ``or ""`` never fires and the caller gets the
    literal string ``"nan"`` — a passage of text that reads as content. Same
    shape as the ``(title or "")`` / ``.astype(str)`` pair that took down
    autonomous full-text ingest on 2026-08-03 (KNOWN_ISSUES §2.1 + 附二).
    """
    if v is None:
        return ""
    if isinstance(v, float) and v != v:      # NaN
        return ""
    s = str(v)
    return "" if s in ("nan", "None", "<NA>", "NaT") else s


def _page(v) -> int:
    """A parquet page number as int. Missing → 0.

    ``int(v or 0)`` raises ``ValueError: cannot convert float NaN to integer``
    on a chunk row with no page — uncaught here, so ONE such chunk killed the
    whole full-text search. Companion to ``literature_index._int_or_zero``,
    which exists for exactly this and was never applied on this path.
    """
    if v is None:
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if f != f or f in (float("inf"), float("-inf")):
        return 0
    return int(f)


def _load_paper(slug: str):
    """``(rows, vectors)`` for one paper, cached on (path, mtime, size).

    Keyed on the file's own stamp rather than just the slug: a re-ingest rewrites
    both files, and a cache keyed on the name alone would keep serving passages
    from the copy that no longer exists on disk.
    """
    base = _papers_dir()
    if base is None:
        return [], None
    pq = base / slug / "chunks.parquet"
    vf = base / slug / "chunk_vectors.npy"
    try:
        st_pq, st_vf = pq.stat(), vf.stat()
    except OSError:
        return [], None
    key = (str(pq), st_pq.st_mtime_ns, st_pq.st_size, st_vf.st_mtime_ns, st_vf.st_size)

    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    try:
        import numpy as np
        import pandas as pd
        df = pd.read_parquet(pq)
        vecs = np.load(vf)
        rows = df.to_dict("records")
        if len(rows) != len(vecs):
            # A mismatch means the two files came from different runs; trusting
            # the pairing would attach one passage's text to another's score.
            logger.warning("chunk/vector count mismatch for %s (%d vs %d) — skipping",
                           slug, len(rows), len(vecs))
            return [], None
    except Exception as exc:  # noqa: BLE001 — one unreadable paper is not fatal
        logger.info("chunk load failed for %s: %s", slug, exc)
        return [], None

    with _cache_lock:
        _cache[key] = (rows, vecs)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return rows, vecs


def _embed_query(query: str):
    """The same cached, same-model query embedding the big index uses."""
    from mast.knowledge.literature_index import _embed_query as _eq
    return _eq(query)


_TERM_RE = re.compile(r"[A-Za-z0-9一-鿿]+")


def _terms(query: str) -> list[str]:
    return [t.lower() for t in _TERM_RE.findall(query or "") if len(t) > 1]


def _keyword_score(text: str, terms: list[str]) -> tuple[float, int]:
    """``(score, n_matched_terms)`` for the degraded path."""
    low = (text or "").lower()
    matched = 0
    total = 0
    for t in terms:
        c = low.count(t)
        if c:
            matched += 1
            total += c
    return (float(total), matched)


def search_chunks(
    query: str,
    slugs: "list[str] | None" = None,
    k: int = 8,
    *,
    per_paper_cap: int = 3,
    embedder: "Callable[[str], Any] | None" = None,
) -> "tuple[list[dict], dict]":
    """Find the passages that answer ``query``.

    Args:
        query:  free text, English or Chinese (the embedding is multilingual).
        slugs:  restrict to these papers; ``None`` searches every chunked paper.
        k:      how many passages to return.
        per_paper_cap: at most this many passages from any one paper, so a single
            verbose paper cannot fill the whole answer and hide the others.
        embedder: injection point for tests; takes the query, returns a vector.

    Returns ``(hits, status)``. Each hit carries ``slug / title / chunk_id /
    page / score / text``. ``status`` mirrors
    :func:`literature_index.search_with_status`: ``retrieval`` (semantic|keyword),
    ``degraded``, ``reason``, ``trustworthy``, plus ``n_papers`` / ``n_chunks``.
    """
    status: dict[str, Any] = {
        "retrieval": "semantic", "degraded": False, "reason": "",
        "trustworthy": True, "n_papers": 0, "n_chunks": 0,
        "matched_terms": [], "unmatched_terms": [],
    }
    if not (query or "").strip():
        return [], status

    wanted = [s for s in (slugs or available_papers()) if s]
    if not wanted:
        status.update(reason="本地没有任何已分块的全文", trustworthy=False)
        return [], status

    loaded: list[tuple[str, list, Any]] = []
    for slug in wanted:
        rows, vecs = _load_paper(slug)
        if rows and vecs is not None:
            loaded.append((slug, rows, vecs))
    status["n_papers"] = len(loaded)
    status["n_chunks"] = sum(len(r) for _s, r, _v in loaded)
    if not loaded:
        status.update(reason="这些论文没有可检索的全文块（可能未 ingest 或没有文本层）",
                      trustworthy=False)
        return [], status

    qv = None
    try:
        qv = embedder(query) if embedder is not None else _embed_query(query)
    except Exception as exc:  # noqa: BLE001 — degrade, but say so
        logger.warning("full-text semantic embed unavailable (%s); keyword fallback", exc)
        status.update(retrieval="keyword", degraded=True,
                      reason=f"语义 embedding 不可用（{type(exc).__name__}）")

    scored: list[dict] = []
    if qv is not None and not status["degraded"]:
        import numpy as np
        q = np.asarray(qv, dtype=np.float32).reshape(-1)
        for slug, rows, vecs in loaded:
            # Both sides are L2-normalised at write time, so the dot product IS
            # the cosine — no renormalising, no division.
            sims = np.asarray(vecs, dtype=np.float32) @ q
            for i, row in enumerate(rows):
                scored.append({"slug": slug, "chunk_id": _text(row.get("chunk_id")),
                               "page": _page(row.get("page")),
                               "text": _text(row.get("text")),
                               "score": float(sims[i])})
    else:
        terms = _terms(query)
        seen_terms: set[str] = set()
        for slug, rows, _vecs in loaded:
            for row in rows:
                text = _text(row.get("text"))
                sc, matched = _keyword_score(text, terms)
                if sc <= 0:
                    continue
                for t in terms:
                    if t in text.lower():
                        seen_terms.add(t)
                scored.append({"slug": slug, "chunk_id": _text(row.get("chunk_id")),
                               "page": _page(row.get("page")),
                               "text": text, "score": sc})
        status["matched_terms"] = sorted(seen_terms)
        status["unmatched_terms"] = sorted(set(terms) - seen_terms)
        # A ranking built from terms that do not occur anywhere carries no
        # information, and must not be presented as if it did.
        if not scored or not seen_terms:
            status["trustworthy"] = False

    if not scored:
        return [], status

    scored.sort(key=lambda h: -h["score"])
    if len({round(h["score"], 6) for h in scored[:max(k, 5)]}) <= 1:
        status["trustworthy"] = False   # degenerate ranking

    out: list[dict] = []
    per_paper: dict[str, int] = {}
    for h in scored:
        if per_paper.get(h["slug"], 0) >= max(1, per_paper_cap):
            continue
        per_paper[h["slug"]] = per_paper.get(h["slug"], 0) + 1
        h["title"] = paper_title(h["slug"])
        out.append(h)
        if len(out) >= max(1, k):
            break
    return out, status
