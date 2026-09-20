"""OpenAlex STM corpus loader — read 62,997 papers off the local corpus dump.

The corpus lives outside the MAST repo (786 MB merged jsonl + classified
parquet), so its location is site-specific and MUST come from the
environment — never from a path baked into the source, which would only ever
resolve on the machine it was written on. Resolution:
    1. ``MAST_OPENALEX_DIR`` env var (absolute path to the ``data/`` dir)
    2. ``<MAST2_PROJECT_ROOT>/openalex_stm/data`` (follows the user data dir)

Three things to do here, all language-agnostic:
    * `restore_abstract(inv_index)` — invert OpenAlex's `abstract_inverted_index`
      back to a normal whitespace-joined string.
    * `iter_works(jsonl_path)` — stream-read merged/all_works.jsonl without
      pulling 786 MB into RAM.
    * `read_parquet(path)` — thin pyarrow wrapper that returns a pandas.DataFrame.

No dependence on `mast.config` or any other MAST module — this file is pure
data plumbing so v1 (`mast/knowledge/`) and v2 (`MASTv2/mast/knowledge/`) can
vendor it identically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

#: Corpus location relative to the user data dir, used when the explicit env
#: var is unset. Deliberately relative: an absolute default is a path that only
#: exists on one machine, and it ships to every other one inside the bundle.
_RELATIVE_DIR = Path("openalex_stm") / "data"


def corpus_dir() -> Path:
    """Return the OpenAlex data directory.

    Resolution: ``MAST_OPENALEX_DIR`` > ``<MAST2_PROJECT_ROOT>/openalex_stm/data``.
    Raises FileNotFoundError if the resolved path doesn't exist.
    """
    p = os.environ.get("MAST_OPENALEX_DIR", "").strip()
    if p:
        base = Path(p)
    else:
        root = os.environ.get("MAST2_PROJECT_ROOT", "").strip()
        base = (Path(root) if root else Path.cwd()) / _RELATIVE_DIR
    if not base.exists():
        raise FileNotFoundError(
            f"OpenAlex corpus not found at {base}. "
            f"Set MAST_OPENALEX_DIR to the corpus 'data' directory, "
            f"or place it under <MAST2_PROJECT_ROOT>/{_RELATIVE_DIR.as_posix()}."
        )
    return base


def merged_jsonl() -> Path:
    return corpus_dir() / "raw" / "merged" / "all_works.jsonl"


def cleaned_parquet() -> Path:
    return corpus_dir() / "cleaned" / "stm_papers.parquet"


def classified_parquet() -> Path:
    return corpus_dir() / "classified" / "stm_classified.parquet"


# ── Abstract restoration ──────────────────────────────────────────────

def restore_abstract(inv_index: dict[str, list[int]] | None) -> str:
    """Restore an OpenAlex inverted-index abstract back to plain text.

    Input shape: ``{"word": [pos, pos, ...]}``. Higher positions mean later
    in the abstract. Empty / None inputs return empty string.
    """
    if not inv_index:
        return ""
    # Build (position, word) pairs; sort by position; whitespace-join.
    pairs: list[tuple[int, str]] = []
    for word, positions in inv_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pairs)


# ── Streaming JSONL reader ────────────────────────────────────────────

def iter_works(
    path: str | Path | None = None,
    *,
    require_abstract: bool = True,
    yield_restored: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield one work dict per line from all_works.jsonl.

    Args:
        path:              override the default merged/all_works.jsonl path
        require_abstract:  skip records that have no abstract_inverted_index
        yield_restored:    if True, attach an `abstract` key with the
                           restored plain-text abstract (and drop the inv index)

    Yields:
        dict per record. Already-decoded JSON; skip silently on decode errors.
    """
    p = Path(path) if path else merged_jsonl()
    if not p.exists():
        raise FileNotFoundError(f"jsonl not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            inv = rec.get("abstract_inverted_index")
            if require_abstract and not inv:
                continue
            if yield_restored:
                rec["abstract"] = restore_abstract(inv)
                rec.pop("abstract_inverted_index", None)
            yield rec


def count_works(path: str | Path | None = None) -> int:
    """Quick line count without parsing JSON. Returns total records in jsonl."""
    p = Path(path) if path else merged_jsonl()
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


# ── Parquet wrappers ──────────────────────────────────────────────────

def read_parquet(path: str | Path):
    """Thin pyarrow → pandas wrapper. Raises ImportError if pyarrow missing."""
    import pandas as pd  # noqa: F401 — required by pd.read_parquet
    return _pd_read_parquet(path)


def _pd_read_parquet(path: str | Path):
    import pandas as pd
    return pd.read_parquet(path)


def load_classified() -> Any:
    """Load the LLM-classified stm_classified.parquet (DataFrame)."""
    return read_parquet(classified_parquet())


def load_cleaned() -> Any:
    """Load the deduplicated stm_papers.parquet (DataFrame)."""
    return read_parquet(cleaned_parquet())


# ── Light-weight projection helpers ───────────────────────────────────

def authorship_summary(rec: dict) -> dict:
    """Pull authors + first institution + country from authorships list."""
    authors = rec.get("authorships") or []
    names = [a.get("author", {}).get("display_name", "") for a in authors[:5]]
    inst = ""
    country = ""
    if authors:
        insts = authors[0].get("institutions") or []
        if insts:
            inst = insts[0].get("display_name", "")
            country = insts[0].get("country_code", "") or ""
    return {"authors": names, "first_institution": inst, "country": country}


def project_record(rec: dict) -> dict:
    """Flatten one OpenAlex record to a small dict suitable for embedding/index."""
    primary = (rec.get("primary_location") or {}).get("source") or {}
    return {
        "id": rec.get("id", ""),
        "doi": rec.get("doi", "") or "",
        "title": rec.get("title", "") or "",
        "abstract": rec.get("abstract", "") or "",
        "year": rec.get("publication_year", 0) or 0,
        "journal": primary.get("display_name", "") or "",
        "concepts": [
            c.get("display_name", "")
            for c in (rec.get("concepts") or [])
            if c.get("display_name")
        ],
        "keywords": [
            k.get("keyword", "")
            for k in (rec.get("keywords") or [])
            if k.get("keyword")
        ],
        "cited_by_count": rec.get("cited_by_count", 0) or 0,
        **authorship_summary(rec),
    }


__all__ = [
    "corpus_dir",
    "merged_jsonl",
    "cleaned_parquet",
    "classified_parquet",
    "restore_abstract",
    "iter_works",
    "count_works",
    "read_parquet",
    "load_classified",
    "load_cleaned",
    "authorship_summary",
    "project_record",
]
