"""B12: provision the MASTv2 literature library's derived artifacts.

Derives, from the openalex-stm producer outputs, the three NEW files the
literature library design (`docs/v2/design/literature_library_design.md` §3.2)
requires next to the already-built `vectors.npy` / `metadata.parquet`:

    MASTv2/artifacts/literature_index/
      abstracts.parquet   # work_id -> abstract (+ authors, first_author,
                          #   concepts, keywords, work_type, has_abstract,
                          #   cited_by_count) + source / user_abstract /
                          #   fulltext_excerpt / abstract_provenance (Decision 2)
      classified.parquet  # work_id -> categories, primary_category, material,
                          #   confidence
      manifest.json       # provenance: source repo+commit, build date, model,
                          #   dim, n_base / n_user / n_corpus row split,
                          #   sha256 of the BASE block of vectors.npy, and the
                          #   local:<hash> -> work_id auto-merge map (Decision 1)

Inputs:
    D:/.../openalex-stm/data/cleaned/stm_papers.parquet
    D:/.../openalex-stm/data/classified/stm_classified.parquet
    (and the already-built vectors.npy / metadata.parquet, which are NOT
     modified — they pin the base block row count + sha256.)

The base block (`source="openalex"`) is written deterministically (sorted by
work_id) so re-running on the same inputs reproduces byte-identical parquet.
A previously-provisioned user tail (`source in {user_pdf, user_url}`) is
PRESERVED across re-provisioning and AUTO-MERGED into freshly-arrived OpenAlex
rows (match by normalized DOI, then high-confidence title+year). The auto-merge
is idempotent: a second run finds no still-matching `local:` row and is a no-op.

The heavy lifting is exposed as pure, importable functions so it can be unit
tested with tiny synthetic parquet:

    provision(papers_path, classified_path, out_dir, ...) -> ProvisionResult
    auto_merge(user_rows, base_rows) -> AutoMergeResult

CLI:
    python 12_provision_litlib.py [--papers PATH] [--classified PATH]
                                  [--out-dir PATH] [--vectors PATH]
                                  [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# pandas / pyarrow / numpy are optional at import time so a bare `--help` or an
# import-for-testing never explodes on a machine missing the heavy deps. The
# functions that actually need them call _require_deps() and raise a clear,
# actionable error instead of a raw ImportError deep in the call stack.
try:  # pragma: no cover - exercised indirectly
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:  # pragma: no cover
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# ── Paths & constants ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
# {repo}/scripts/openalex_pipeline/12_provision_litlib.py -> repo root = parents[2]
REPO_ROOT = _HERE.parents[2]
INDEX_DIR = REPO_ROOT / "MASTv2" / "artifacts" / "literature_index"

# 语料仓在本仓之外，位置因机器而异 —— 从 OPENALEX_STM_REPO 取，缺省退回
# 「与本仓同级的 openalex-stm/」。写死一台机器的绝对路径等于把它发给每个
# clone 这个仓库的人，而那条路径在别处一次都解析不到。
DEFAULT_SOURCE_REPO_DIR = Path(
    os.environ.get("OPENALEX_STM_REPO", "").strip()
    or REPO_ROOT.parent / "openalex-stm")
DEFAULT_PAPERS = DEFAULT_SOURCE_REPO_DIR / "data" / "cleaned" / "stm_papers.parquet"
DEFAULT_CLASSIFIED = DEFAULT_SOURCE_REPO_DIR / "data" / "classified" / "stm_classified.parquet"

EMBEDDING_MODEL = "text-embedding-v3"
DIM = 1024

# Columns selected for abstracts.parquet (design §3.2 step 1).
_ABSTRACT_BASE_COLS = [
    "work_id", "abstract", "authors", "first_author",
    "concepts", "keywords", "work_type", "has_abstract", "cited_by_count",
]
# Decision-2 keep-both columns added on top of the OpenAlex base columns.
_ABSTRACT_EXTRA_COLS = ["source", "user_abstract", "fulltext_excerpt", "abstract_provenance"]
_ABSTRACT_ALL_COLS = _ABSTRACT_BASE_COLS + _ABSTRACT_EXTRA_COLS

_CLASSIFIED_COLS = ["work_id", "categories", "primary_category", "material", "confidence"]

_USER_SOURCES = ("user_pdf", "user_url")


# ── Result containers ─────────────────────────────────────────────────
@dataclass
class AutoMergeResult:
    """Outcome of folding the carried-over user tail into a fresh base block."""
    merged_user_rows: list[dict] = field(default_factory=list)   # post-merge user tail
    merge_map: dict[str, str] = field(default_factory=dict)       # local:<hash> -> work_id
    # local:<hash> -> {work_id, matched_on, original_local_ids:[...]}
    merge_detail: dict[str, dict] = field(default_factory=dict)


@dataclass
class ProvisionResult:
    out_dir: Path
    n_base: int = 0
    n_user: int = 0
    n_corpus: int = 0
    base_sha256: str = ""
    merge_map: dict[str, str] = field(default_factory=dict)
    coverage_gap: list[str] = field(default_factory=list)
    abstracts_path: Path | None = None
    classified_path: Path | None = None
    manifest_path: Path | None = None
    dry_run: bool = False


# ── Dependency / IO guards ────────────────────────────────────────────
def _require_deps() -> None:
    missing = [n for n, m in (("numpy", np), ("pandas", pd)) if m is None]
    if missing:
        raise RuntimeError(
            "12_provision_litlib needs: " + ", ".join(missing) + ".\n"
            "Install in the v2 venv:\n"
            "  .venv-v2-py313/Scripts/python.exe -m pip install pandas pyarrow numpy"
        )


def _read_parquet(path: Path) -> "pd.DataFrame":
    _require_deps()
    if not Path(path).exists():
        raise FileNotFoundError(f"input parquet not found: {path}")
    return pd.read_parquet(path)


# ── Normalization helpers (pure, dep-free) ────────────────────────────
def normalize_doi(doi: Any) -> str:
    """Lowercase, strip the https://doi.org/ prefix and surrounding noise.

    Returns "" for empty / NaN / non-DOI input so it never spuriously matches.
    """
    if doi is None:
        return ""
    s = str(doi).strip().lower()
    if not s or s in ("nan", "none"):
        return ""
    # strip common URL prefixes
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()


def normalize_title(title: Any) -> str:
    """Aggressively normalize a title for high-confidence matching.

    Lowercase, collapse whitespace, drop all non-alphanumeric chars. This makes
    "Kondo Effect in Au(111)" and "kondo effect in au(111)!" compare equal while
    still being conservative enough to avoid mis-merging different papers.
    """
    if title is None:
        return ""
    s = str(title).strip().lower()
    if not s or s == "nan":
        return ""
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _first_author(row: dict) -> str:
    fa = row.get("first_author")
    if fa:
        return str(fa).strip().lower()
    authors = row.get("authors") or ""
    if authors:
        return str(authors).split(";")[0].strip().lower()
    return ""


def _year_of(row: dict) -> str:
    for k in ("publication_year", "year"):
        v = row.get(k)
        if v not in (None, "", "nan"):
            try:
                return str(int(float(v)))
            except (TypeError, ValueError):
                return str(v).strip()
    return ""


def synthetic_local_id(title: Any, first_author: Any, year: Any) -> str:
    """Mint the synthetic stable id `local:<sha16(title+first_author+year)>`.

    Mirrors the ingestion-side minting (design §2.1 / §7.4) so the provisioning
    script and the runtime promotion path agree on the canonical key.
    """
    t = normalize_title(title)
    fa = str(first_author or "").strip().lower()
    y = ""
    if year not in (None, "", "nan"):
        try:
            y = str(int(float(year)))
        except (TypeError, ValueError):
            y = str(year).strip()
    payload = f"{t}|{fa}|{y}".encode("utf-8")
    return "local:" + hashlib.sha256(payload).hexdigest()[:16]


# ── Auto-merge (Decision 1, design §3.2 step 4 / §7.5) ────────────────
def auto_merge(user_rows: list[dict], base_rows: list[dict]) -> AutoMergeResult:
    """Fold the carried-over user tail into a freshly-built base block.

    For each user/local row (``source in {user_pdf, user_url}``) try to match it
    to a base row by normalized DOI first, then high-confidence title+year. On a
    match, record ``local:<hash> -> work_id`` in the merge map and DROP the now
    duplicate local row (its abstract/full-text refs are expected to be folded
    onto the base row by the caller, keeping BOTH abstracts — Decision 2). Many
    locals resolving to one work_id collapse into a single merge_detail entry.
    Unmatched user rows are re-appended unchanged.

    Idempotent: a row that already carries a real (non ``local:``) work_id never
    matches by its own id again; a second run with already-merged ids is a no-op.

    Pure / dep-free: operates on plain dict lists, so it is trivially testable.
    """
    # Index base rows by normalized DOI and by (title-key, year).
    by_doi: dict[str, dict] = {}
    by_title_year: dict[tuple[str, str], dict] = {}
    for b in base_rows:
        d = normalize_doi(b.get("doi"))
        if d and d not in by_doi:
            by_doi[d] = b
        tk = normalize_title(b.get("title"))
        yr = _year_of(b)
        if tk and yr:
            by_title_year.setdefault((tk, yr), b)

    merge_map: dict[str, str] = {}
    # work_id -> aggregated detail (supports many-locals-to-one collapse)
    collapsed: dict[str, dict] = {}
    survivors: list[dict] = []

    for u in user_rows:
        uid = str(u.get("work_id") or "")
        is_local = uid.startswith("local:")
        # Only `local:` rows are merge candidates. A user row that already has a
        # real work_id (an earlier run merged it) is carried over unchanged.
        matched_base = None
        matched_on = ""
        if is_local:
            d = normalize_doi(u.get("doi"))
            if d and d in by_doi:
                matched_base = by_doi[d]
                matched_on = "doi"
            else:
                tk = normalize_title(u.get("title"))
                yr = _year_of(u)
                if tk and yr and (tk, yr) in by_title_year:
                    matched_base = by_title_year[(tk, yr)]
                    matched_on = "title_year"

        if matched_base is not None:
            real_wid = str(matched_base.get("work_id") or "")
            merge_map[uid] = real_wid
            detail = collapsed.setdefault(
                real_wid,
                {"work_id": real_wid, "matched_on": matched_on, "original_local_ids": []},
            )
            if uid not in detail["original_local_ids"]:
                detail["original_local_ids"].append(uid)
            # DOI match is stronger than title_year; remember the strongest.
            if matched_on == "doi":
                detail["matched_on"] = "doi"
            # The user row itself is dropped (its abstract is folded onto base by
            # the caller). We do NOT re-append it to survivors.
        else:
            survivors.append(dict(u))

    merge_detail = dict(collapsed)
    return AutoMergeResult(
        merged_user_rows=survivors,
        merge_map=merge_map,
        merge_detail=merge_detail,
    )


def _fold_user_abstract_onto_base(
    base_abs: "pd.DataFrame",
    user_rows: list[dict],
    merge_detail: dict[str, dict],
) -> "pd.DataFrame":
    """Keep BOTH abstracts (Decision 2) for auto-merged rows.

    For every base row that absorbed one or more ``local:<hash>`` user papers,
    copy the (richest) user abstract into the base row's ``user_abstract`` /
    ``fulltext_excerpt`` slot (never overwriting the OpenAlex ``abstract``), tag
    the provenance ``openalex+user``, and record ``original_local_id`` (a list,
    for the many-locals-to-one collapse) + ``matched_on`` for audit. Returns the
    augmented DataFrame (two extra audit columns added when any merge occurred).
    """
    _require_deps()
    if not merge_detail:
        return base_abs
    user_by_id = {str(u.get("work_id") or ""): u for u in user_rows}
    df = base_abs.copy()
    if "original_local_id" not in df.columns:
        df["original_local_id"] = ""
    if "matched_on" not in df.columns:
        df["matched_on"] = ""
    wid_to_pos = {str(w): i for i, w in enumerate(df["work_id"].tolist())}

    for real_wid, detail in merge_detail.items():
        pos = wid_to_pos.get(str(real_wid))
        if pos is None:
            continue
        local_ids = detail.get("original_local_ids", [])
        # pick the richest (longest) user abstract among the collapsed locals
        best_abs, best_excerpt = "", ""
        for lid in local_ids:
            u = user_by_id.get(lid, {})
            cand = str(u.get("user_abstract") or u.get("abstract") or "")
            if len(cand) > len(best_abs):
                best_abs = cand
                best_excerpt = str(u.get("fulltext_excerpt") or "")
        if best_abs:
            df.iat[pos, df.columns.get_loc("user_abstract")] = best_abs
            df.iat[pos, df.columns.get_loc("fulltext_excerpt")] = best_excerpt
            df.iat[pos, df.columns.get_loc("abstract_provenance")] = "openalex+user"
        df.iat[pos, df.columns.get_loc("original_local_id")] = ";".join(local_ids)
        df.iat[pos, df.columns.get_loc("matched_on")] = detail.get("matched_on", "")
    return df


# ── Base block builders ───────────────────────────────────────────────
def build_base_abstracts(papers_df: "pd.DataFrame") -> "pd.DataFrame":
    """Project stm_papers.parquet -> the BASE abstracts block (source=openalex).

    Deterministic: deduped by work_id (keep first) and sorted by work_id so the
    output is byte-identical across runs on the same input.
    """
    _require_deps()
    keep = [c for c in _ABSTRACT_BASE_COLS if c in papers_df.columns]
    proj = papers_df[keep].copy()
    # ensure every expected base column exists (older snapshots may lack some)
    for c in _ABSTRACT_BASE_COLS:
        if c not in proj.columns:
            proj[c] = "" if c not in ("has_abstract", "cited_by_count") else 0
    proj = proj.dropna(subset=["work_id"])
    proj["work_id"] = proj["work_id"].astype(str)
    proj = proj[proj["work_id"] != ""]
    proj = proj.drop_duplicates(subset=["work_id"], keep="first")
    # Decision-2 keep-both columns: base rows have only the OpenAlex abstract.
    proj["source"] = "openalex"
    proj["user_abstract"] = ""
    proj["fulltext_excerpt"] = ""
    proj["abstract_provenance"] = "openalex"
    proj = proj[_ABSTRACT_ALL_COLS]
    proj = proj.sort_values("work_id", kind="stable").reset_index(drop=True)
    return proj


def _build_match_view(papers_df: "pd.DataFrame") -> list[dict]:
    """Project papers -> per-work_id match keys {work_id, doi, title, year}.

    The abstracts projection deliberately drops doi/title/year to keep the
    index-aligned table lean, so auto_merge needs this separate view (built off
    the same source papers) to resolve a carried-over local:<hash> by DOI then
    title+year. Deduped by work_id (keep first), aligned with build_base_abstracts.
    """
    _require_deps()
    keep = [c for c in ("work_id", "doi", "title", "publication_year") if c in papers_df.columns]
    proj = papers_df[keep].copy()
    proj = proj.dropna(subset=["work_id"])
    proj["work_id"] = proj["work_id"].astype(str)
    proj = proj[proj["work_id"] != ""]
    proj = proj.drop_duplicates(subset=["work_id"], keep="first")
    rows = []
    for r in proj.to_dict(orient="records"):
        rows.append({
            "work_id": r.get("work_id", ""),
            "doi": r.get("doi", "") or "",
            "title": r.get("title", "") or "",
            "year": r.get("publication_year", "") or "",
        })
    return rows


def build_classified(classified_df: "pd.DataFrame") -> "pd.DataFrame":
    """Project stm_classified.parquet -> classified.parquet (deterministic)."""
    _require_deps()
    keep = [c for c in _CLASSIFIED_COLS if c in classified_df.columns]
    proj = classified_df[keep].copy()
    for c in _CLASSIFIED_COLS:
        if c not in proj.columns:
            proj[c] = "" if c in ("categories", "material") else 0
    proj = proj.dropna(subset=["work_id"])
    proj["work_id"] = proj["work_id"].astype(str)
    proj = proj[proj["work_id"] != ""]
    proj = proj.drop_duplicates(subset=["work_id"], keep="first")
    proj = proj[_CLASSIFIED_COLS]
    proj = proj.sort_values("work_id", kind="stable").reset_index(drop=True)
    return proj


# ── Carried-over user tail discovery ──────────────────────────────────
def _load_prev_user_tail(abstracts_path: Path) -> list[dict]:
    """Read the user tail (source in {user_pdf, user_url}) from a prior provision.

    Returns [] when no previous abstracts.parquet exists or it has no user rows.
    """
    if not Path(abstracts_path).exists():
        return []
    try:
        prev = pd.read_parquet(abstracts_path)
    except Exception:
        return []
    if "source" not in prev.columns:
        return []
    user = prev[prev["source"].isin(_USER_SOURCES)]
    if user.empty:
        return []
    return user.to_dict(orient="records")


def _sha256_base_block(vectors_path: Path, n_base: int) -> str:
    """sha256 of the BASE block (first n_base rows) of vectors.npy.

    The base sha is what `manifest.json` records so a stale/mismatched base
    vectors.npy is detectable at load time. The user tail is excluded so the sha
    stays stable as the tail grows. Returns "" if vectors.npy is absent.
    """
    if np is None or not Path(vectors_path).exists():
        return ""
    try:
        arr = np.load(vectors_path, mmap_mode="r")
    except Exception:
        return ""
    n = min(n_base, int(arr.shape[0]))
    base = np.ascontiguousarray(arr[:n])
    return hashlib.sha256(base.tobytes()).hexdigest()


def _detect_source_commit(repo_dir: Path) -> str:
    """Best-effort: read the openalex-stm git short commit, dep-free.

    Reads .git/HEAD + the ref file directly so we never shell out / block.
    """
    try:
        git_dir = Path(repo_dir) / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
            # packed-refs fallback
            packed = git_dir / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and line.endswith(ref):
                        return line.split()[0][:12]
        elif head:
            return head[:12]
    except Exception:
        pass
    return ""


# ── The provisioning step (design §3.2) ───────────────────────────────
def provision(
    papers_path: Path,
    classified_path: Path,
    out_dir: Path,
    *,
    vectors_path: Path | None = None,
    n_base_override: int | None = None,
    source_repo_dir: Path | None = None,
    strict_coverage: bool = True,
    dry_run: bool = False,
) -> ProvisionResult:
    """Derive abstracts/classified parquet + manifest (idempotent).

    Steps (design §3.2):
      1. base abstracts block from papers (source="openalex"), sorted by work_id
      2. classified block from the classifier parquet
      3. assert index coverage: metadata.work_id (the base vector index) is a
         subset of abstracts.work_id
      4. auto-merge the carried-over user tail (local:<hash> -> work_id)
      5. re-append the post-merge user tail after the base block
      6. write manifest.json with the base/user split + base sha256 + merge_map

    ``strict_coverage`` (default True, the §3.2 contract): the index-coverage
    check in step 3 is a hard AssertionError. With ``strict_coverage=False`` a
    coverage gap (the vector index references work_ids absent from the freshly
    derived abstracts — a stale-vs-current producer-snapshot drift) is downgraded
    to a recorded ``coverage_gap`` in the manifest + result instead of aborting.
    Either way the gap is reported; it never silently passes.

    Returns a ProvisionResult; with ``dry_run=True`` nothing is written.
    """
    _require_deps()
    out_dir = Path(out_dir)
    abstracts_path = out_dir / "abstracts.parquet"
    classified_out = out_dir / "classified.parquet"
    manifest_path = out_dir / "manifest.json"
    if vectors_path is None:
        vectors_path = out_dir / "vectors.npy"
    metadata_path = out_dir / "metadata.parquet"
    if source_repo_dir is None:
        source_repo_dir = DEFAULT_SOURCE_REPO_DIR

    # 1. base abstracts
    papers_df = _read_parquet(papers_path)
    base_abs = build_base_abstracts(papers_df)
    base_records = base_abs.to_dict(orient="records")

    # 2. classified
    classified_df = _read_parquet(classified_path)
    classified_out_df = build_classified(classified_df)

    # 3. index coverage assertion: every indexed work_id must have an abstract.
    base_wids = set(base_abs["work_id"].astype(str))
    coverage_gap: list[str] = []
    if metadata_path.exists():
        meta = pd.read_parquet(metadata_path)
        index_wids = set(meta["work_id"].astype(str))
        missing = sorted(index_wids - base_wids)
        if missing:
            msg = (
                f"index coverage gap: {len(missing)} indexed work_id(s) absent "
                f"from abstracts (e.g. {missing[:3]}). The vector index references "
                "papers the current stm_papers.parquet no longer contains "
                "(stale-vs-current producer snapshot). abstracts.parquet should be "
                "a superset of the vector index."
            )
            if strict_coverage:
                raise AssertionError(msg + " Re-run with strict_coverage=False / "
                                     "--lax-coverage to record the gap instead.")
            coverage_gap = missing

    # 4. auto-merge the carried-over user tail.
    #    The abstracts projection (base_records) drops doi/title/year, but the
    #    match keys live in the source papers parquet. Build a per-work_id match
    #    view {work_id, doi, title, year} from papers and feed THAT to auto_merge.
    base_match_rows = _build_match_view(papers_df)
    prev_user_tail = _load_prev_user_tail(abstracts_path)
    am = auto_merge(prev_user_tail, base_match_rows)
    # Decision-2: fold matched users' abstracts onto base rows (keep BOTH).
    if am.merge_detail:
        base_abs = _fold_user_abstract_onto_base(base_abs, prev_user_tail, am.merge_detail)

    n_base = n_base_override if n_base_override is not None else len(base_abs)

    # 5. re-append the (post-merge) user tail.
    survivors = am.merged_user_rows
    if survivors:
        survivor_df = pd.DataFrame(survivors)
        # align survivor columns to the abstracts schema (fill missing)
        for c in base_abs.columns:
            if c not in survivor_df.columns:
                survivor_df[c] = ""
        survivor_df = survivor_df[[c for c in base_abs.columns]]
        abstracts_final = pd.concat([base_abs, survivor_df], ignore_index=True)
    else:
        abstracts_final = base_abs
    n_user = len(survivors)
    n_corpus = len(abstracts_final)

    base_sha = _sha256_base_block(vectors_path, n_base)

    manifest = {
        "schema": "litlib_provision/1",
        "source_repo": "HamsterPark/openalex-stm",
        "source_repo_dir": str(source_repo_dir),
        "source_commit": _detect_source_commit(source_repo_dir),
        "build_date": datetime.now(timezone.utc).isoformat(),
        "model": EMBEDDING_MODEL,
        "dim": DIM,
        "n_base": int(n_base),
        "n_user": int(n_user),
        "n_corpus": int(n_corpus),
        "base_vectors_sha256": base_sha,
        "merge_map": am.merge_map,
        "merge_detail": am.merge_detail,
        "coverage_gap_count": len(coverage_gap),
        "coverage_gap_sample": coverage_gap[:20],
        "inputs": {
            "papers": str(papers_path),
            "classified": str(classified_path),
        },
    }

    result = ProvisionResult(
        out_dir=out_dir,
        n_base=int(n_base),
        n_user=int(n_user),
        n_corpus=int(n_corpus),
        base_sha256=base_sha,
        merge_map=am.merge_map,
        coverage_gap=coverage_gap,
        abstracts_path=abstracts_path,
        classified_path=classified_out,
        manifest_path=manifest_path,
        dry_run=dry_run,
    )

    if dry_run:
        return result

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(abstracts_final, abstracts_path)
    _atomic_write_parquet(classified_out_df, classified_out)
    _atomic_write_json(manifest, manifest_path)
    return result


# ── Atomic writers (temp + replace) ───────────────────────────────────
def _atomic_write_parquet(df: "pd.DataFrame", path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ── CLI ───────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", type=Path, default=DEFAULT_PAPERS,
                    help="stm_papers.parquet (openalex-stm cleaned output)")
    ap.add_argument("--classified", type=Path, default=DEFAULT_CLASSIFIED,
                    help="stm_classified.parquet (openalex-stm classifier output)")
    ap.add_argument("--out-dir", type=Path, default=INDEX_DIR,
                    help="MASTv2/artifacts/literature_index/")
    ap.add_argument("--vectors", type=Path, default=None,
                    help="vectors.npy (default: <out-dir>/vectors.npy)")
    ap.add_argument("--source-repo-dir", type=Path, default=DEFAULT_SOURCE_REPO_DIR,
                    help="openalex-stm repo dir (for source_commit provenance)")
    ap.add_argument("--lax-coverage", action="store_true",
                    help="downgrade the index-coverage assertion to a recorded "
                         "manifest warning (use when the vector index predates the "
                         "current stm_papers.parquet snapshot)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + report counts; write nothing")
    args = ap.parse_args(argv)

    try:
        _require_deps()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        res = provision(
            args.papers, args.classified, args.out_dir,
            vectors_path=args.vectors,
            source_repo_dir=args.source_repo_dir,
            strict_coverage=not args.lax_coverage,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, AssertionError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    tag = "DRY-RUN (nothing written)" if res.dry_run else "wrote"
    print(f"{tag}: n_base={res.n_base:,}  n_user={res.n_user:,}  n_corpus={res.n_corpus:,}")
    print(f"  base vectors sha256: {res.base_sha256[:16] or '(no vectors.npy)'}")
    if res.merge_map:
        print(f"  auto-merged {len(res.merge_map)} local id(s) -> work_id")
    if res.coverage_gap:
        print(f"  WARNING: {len(res.coverage_gap)} indexed work_id(s) lack an "
              f"abstract (stale index vs current papers) — recorded in manifest")
    if not res.dry_run:
        print(f"  abstracts:  {res.abstracts_path}")
        print(f"  classified: {res.classified_path}")
        print(f"  manifest:   {res.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
