"""Full-text ingestion pipeline (literature library, design §7).

PDF (on disk)
  1. EXTRACT  — PyMuPDF (fitz): page → text. Store fulltext.txt; capture
                per-page char offsets. (Extractor is injectable for tests.)
  2. RESOLVE  — bind to OpenAlex work_id (§7.4): explicit arg > embedded DOI
                > title fuzzy match vs abstracts.parquet. If nothing resolves,
                mint a synthetic ``local:<sha16(title+first_author+year)>`` id
                (Decision 1, §2.1) — never refuse.
  3. CHUNK    — split at page then heading boundaries; window (≤800 tokens,
                ~120 overlap) only what is left over. In practice a page-section
                fits in one window, so chunks come out page-section sized with
                NO overlap — see :func:`chunk_text`. Write chunks.parquet
                (chunk_id, page, char_start/end, text, n_tokens).
  4. EMBED    — DashScope text-embedding-v3 @1024-d, BATCHED (≤25 inputs/call),
                SAME model/endpoint as the index. Write chunk_vectors.npy +
                abstract_vector.npy, float32, L2-normalized. (Embedder is
                injectable; tests pass a fake — we never call DashScope here.)
  5. ATTACH   — **DORMANT since 2026-07-29** (design §4.6). Used to append
                (work_id, abstract + chunk) vectors to a per-library index under
                ``artifacts/literature_libs/<library_id>/``. Removed from the
                default flow: ``library_id`` now only means "add a member pointer
                (+ full-text ref)", never "build a second vector index".
                See :func:`_append_library_index` for why it was never live.
  6. PROMOTE  — (Decision 1, §7.5) upsert the abstract/metadata into the BIG
                library: append abstract_vector to vectors.npy, append
                metadata/abstracts rows with source ∈ {user_pdf, user_url}.
                Dedup by work_id/DOI first; keep BOTH abstracts (Decision 2).
  7. DEDUP    — §7.4: don't double-ingest the same work_id; sha256(pdf) guards
                re-upload.
  RETURN IngestResult {work_id, source, n_chunks, slug_dir, promoted, status}.

All disk-side: nothing here ever reaches ``MASTState`` except the small,
JSON-serializable ``IngestResult`` dict the ``ingest_pdf`` tool turns into a
``Command(update=…)``. No array/handle crosses into state (§7.3).

Pure sync I/O — no event-loop assumptions (the LangGraph node wraps this in
``asyncio.to_thread``; the GUI runs it in a worker thread). Degrades gracefully
when fitz / numpy / pandas are missing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from mast.knowledge.literature_index import _int_or_zero

logger = logging.getLogger(__name__)

# ── Optional heavy deps (graceful degradation) ───────────────────────
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - exercised only when numpy absent
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - exercised only when pandas absent
    pd = None  # type: ignore


# ── Constants (must match the index, literature_index.py:56-59) ──────
_MODEL = "text-embedding-v3"
_DIM = 1024
_DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
_EMBED_BATCH = 25  # ≤25 inputs/call (§7, step 4)
# Below this many non-whitespace extracted chars we treat the PDF as
# scanned / image-only (no text layer). ``ingest_pdf`` first tries the
# qwen-vl-ocr fallback (knowledge.ocr) to recover text; if OCR is unavailable or
# still comes up short we REFUSE to promote a blank-abstract junk row into the
# big index (honest ``warning:no_text_layer`` instead). Doubles as the OCR
# trigger AND the OCR acceptance bar — recovered text must clear the same
# non-whitespace-char threshold a real text layer must clear.
_MIN_INGEST_TEXT_CHARS = 200

# Chunking (§7, step 3): ~800-token sliding windows w/ ~120 overlap.
# We approximate tokens with whitespace words (cheap, deterministic, dep-free).
_CHUNK_TOKENS = 800
_CHUNK_OVERLAP = 120

# Heading regexes — mirror tools.py:204 _SECTION_PATTERNS so chunk boundaries
# respect section starts. (Copied, not imported: tools.py is an agent package
# and importing it from mast.knowledge would invert the dependency direction.)
_SECTION_PATTERNS: tuple[str, ...] = (
    r"^\s*abstract\b",
    r"^\s*summary\b",
    r"^\s*1\.?\s*introduction\b",
    r"^\s*introduction\b",
    r"^\s*\d?\.?\s*(methods|materials and methods|experimental(\s+section)?)\b",
    r"^\s*\d?\.?\s*results(\s+and\s+discussion)?\b",
    r"^\s*\d?\.?\s*discussion\b",
    r"^\s*\d?\.?\s*(conclusions?|concluding\s+remarks)\b",
    r"^\s*references\b",
    r"^\s*bibliography\b",
)
_HEADING_RE = re.compile("|".join(_SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)

# DOI extraction from PDF text (RESOLVE step).
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)


# ── Result type ──────────────────────────────────────────────────────
@dataclass
class IngestResult:
    """Serializable outcome of an ingestion run (the only thing reaching state)."""

    work_id: str
    source: str  # "user_pdf" | "user_url" | "agent"
    n_chunks: int
    slug_dir: str
    promoted: bool
    status: str  # "ingested" | "noop" | "replaced" | "error:<reason>"
    doi: str = ""
    title: str = ""
    sha256: str = ""
    detail: str = ""
    library_id: str = ""
    ocr_used: bool = False  # text recovered via qwen-vl-ocr (scanned PDF fallback)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Page-extraction container (extractor output) ─────────────────────
@dataclass
class ExtractedText:
    """Output of the EXTRACT step. ``pages`` is page text in order."""

    pages: list[str] = field(default_factory=list)
    doi: str = ""
    title: str = ""

    @property
    def full_text(self) -> str:
        return "\n".join(self.pages)


# ── Path resolution ──────────────────────────────────────────────────
# Delegated to the ONE shared resolver, ``knowledge/paths.py``. The private
# ``_find_repo_root()`` that used to live here was one of five near-identical
# copies across ``knowledge/*``; each also froze its answer into a module
# constant at import time, which is precisely how pointed every
# data path at a directory that does not exist in the frozen build.
#
# Resolved per call (env overrides ``MAST_LITERATURE_INDEX_DIR`` /
# ``MAST_PAPERS_DIR`` / ``MAST_LITERATURE_LIBS_DIR`` are honoured late, so a
# test or launcher can repoint them after import). The defaults resolve to the
# same directories the old walk produced — verified on 2026-07-29, because a
# silent move here reads to the operator as "my 205 MB library vanished".
def _big_index_dir() -> Path:
    from mast.knowledge.paths import big_index_dir
    return big_index_dir()


def _libs_dir() -> Path:
    from mast.knowledge.paths import libs_dir
    return libs_dir()


def _papers_dir() -> Path:
    from mast.knowledge.paths import papers_dir
    return papers_dir()


# ── Small helpers ────────────────────────────────────────────────────
def _slug_for_work_id(work_id: str) -> str:
    """Filesystem-safe slug from a work_id (§2.3: strip the URL prefix).

    Handles OpenAlex URLs (``https://openalex.org/W…`` → ``W…``) and synthetic
    ``local:<hash>`` ids. Always sanitised to ``[A-Za-z0-9._-]`` to defeat path
    traversal — the slug becomes a single directory name, never a path.
    """
    wid = (work_id or "").strip()
    # OpenAlex URL → bare id
    if "openalex.org/" in wid:
        wid = wid.rsplit("/", 1)[-1]
    # local:<hash> → local_<hash>; local:doi:<doi> → local_doi_<doi-sanitised>
    wid = wid.replace(":", "_").replace("/", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", wid)
    safe = safe.strip("._-") or "paper"
    return safe[:120]


def _as_text(value: Any) -> str:
    """任何单元格 → str。缺失值（NaN / None / pd.NA / NaT）→ ""。

    Arrow 支撑的 DataFrame 中，``astype(str)`` 不保证消除空值。回调仍可能收到
    ``float('nan')``；因此归一化函数必须自己识别 NaN、None、pd.NA 和 NaT。

    于是 ``_normalize_title`` 的 ``.lower()`` 抛 AttributeError,整条自主取全文的
    链路挂掉。``_normalize_doi`` 的 ``(doi or "")`` 也拦不住 —— ``float('nan')``
    是**真值**,所以它照样走到 ``.strip()``。

    真正的教训不是"少写了一个 None 判断",是**一个看起来在防护的写法悄悄停止防护
    了**,而没有任何东西会说出来。所以修在归一化函数自己身上:一个归一化器对缺失
    输入抛异常,无论调用方怎么写都是错的。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value != value:  # NaN
        return ""
    try:  # pd.NA / pd.NaT / np.nan(object 列里)
        import pandas as _pd

        if _pd.isna(value):
            return ""
    except (TypeError, ValueError, ImportError):
        pass  # 数组/非标量/没装 pandas —— 交给下面的 str()
    return str(value)


def _normalize_doi(doi: Any) -> str:
    d = _as_text(doi).strip().lower()
    d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
    d = d.replace("https://dx.doi.org/", "").replace("http://dx.doi.org/", "")
    d = d.lstrip("/")
    if d.startswith("doi:"):
        d = d[4:]
    return d.strip()


def _normalize_title(title: Any) -> str:
    t = _as_text(title).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _synthetic_id(title: str, first_author: str, year: Any, doi: str = "") -> str:
    """Mint a stable synthetic id (§2.1 / §7.4).

    ``local:doi:<doi>`` when a DOI exists but no OpenAlex match; otherwise
    ``local:<sha16(normalized-title+first-author+year)>``.
    """
    ndoi = _normalize_doi(doi)
    if ndoi:
        return f"local:doi:{ndoi}"
    basis = f"{_normalize_title(title)}|{(first_author or '').strip().lower()}|{year or ''}"
    return f"local:{_sha16(basis)}"


def _count_tokens(text: str) -> int:
    return len(text.split())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """temp + rename atomic write (§7.5 step 5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _atomic_write_npy(path: Path, arr: Any) -> None:
    if np is None:  # pragma: no cover - guarded by callers
        raise RuntimeError("numpy unavailable")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npy.tmp")
    os.close(fd)
    try:
        # np.save appends .npy if missing — write to an explicit name then rename
        tmp_named = tmp + ".npy"
        np.save(tmp_named, arr)
        os.replace(tmp_named, path)
    finally:
        for cand in (tmp, tmp + ".npy"):
            if os.path.exists(cand):
                try:
                    os.remove(cand)
                except OSError:
                    pass


def _atomic_write_parquet(path: Path, df: Any) -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas unavailable")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ── 1. EXTRACT ───────────────────────────────────────────────────────
def extract_pdf(pdf_path: str | Path) -> ExtractedText:
    """Extract page text + best-effort DOI/title from a PDF via PyMuPDF.

    Raises ``RuntimeError`` if fitz is unavailable (caller turns it into a
    graceful ``IngestResult`` status). Pure sync I/O.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover - exercised only when fitz absent
        raise RuntimeError(f"PyMuPDF (fitz) unavailable: {e}") from e

    p = Path(pdf_path)
    if not p.is_file():
        raise FileNotFoundError(f"PDF not found: {p}")

    pages: list[str] = []
    title = ""
    doc = fitz.open(str(p))
    try:
        try:
            md = doc.metadata or {}
            title = (md.get("title") or "").strip()
        except Exception:
            title = ""
        for page in doc:
            try:
                pages.append(page.get_text() or "")
            except Exception:
                pages.append("")
    finally:
        doc.close()

    full = "\n".join(pages)
    doi = ""
    m = _DOI_RE.search(full)
    if m:
        doi = m.group(0).rstrip(".,;)")

    if not title:
        # crude title fallback: first non-empty line of page 1
        for line in (pages[0] if pages else "").splitlines():
            if line.strip():
                title = line.strip()[:300]
                break

    return ExtractedText(pages=pages, doi=doi, title=title)


def _extracted_from_ocr(ocr_text: str, prior: ExtractedText) -> ExtractedText:
    """Wrap qwen-vl-ocr output into an :class:`ExtractedText` for the pipeline.

    The OCR text becomes a single logical page (page-level offsets are meaningless
    for a scanned doc — chunking still windows within it and section headings still
    split it). DOI / title are re-derived from the OCR text but PREFER whatever the
    real text-layer extract already found (embedded metadata title, a DOI that
    leaked into the text layer) — OCR only fills the gaps.
    """
    text = ocr_text or ""
    doi = prior.doi
    if not doi:
        m = _DOI_RE.search(text)
        if m:
            doi = m.group(0).rstrip(".,;)")
    title = prior.title
    if not title:
        for line in text.splitlines():
            if line.strip():
                title = line.strip()[:300]
                break
    return ExtractedText(pages=[text], doi=doi, title=title)


# ── 2. RESOLVE ───────────────────────────────────────────────────────
def _load_big_abstracts() -> Optional["pd.DataFrame"]:  # type: ignore[name-defined]
    """Lazy-load the big-library abstracts table (work_id keyed). None if absent."""
    if pd is None:
        return None
    path = _big_index_dir() / "abstracts.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # pragma: no cover
        logger.warning("could not read abstracts.parquet: %s", e)
        return None


def resolve_work_id(
    extracted: ExtractedText,
    *,
    explicit_work_id: str = "",
    abstracts_df: Optional["pd.DataFrame"] = None,  # type: ignore[name-defined]
    first_author: str = "",
    year: Any = "",
) -> tuple[str, str, str]:
    """Resolve (work_id, doi, title) per §7.4.

    Priority: explicit arg > embedded DOI match > title fuzzy match >
    synthetic ``local:<hash>``. Never refuses.

    ``abstracts_df`` is injectable (tests pass a tiny frame); when None we lazy
    load the big library's abstracts.parquet (None when it doesn't exist).
    """
    doi = _normalize_doi(extracted.doi)
    title = extracted.title or ""

    if explicit_work_id and explicit_work_id.strip():
        return explicit_work_id.strip(), doi, title

    df = abstracts_df if abstracts_df is not None else _load_big_abstracts()

    if df is not None and len(df) and pd is not None:
        cols = set(df.columns)
        # DOI match
        if doi and "doi" in cols:
            # 没有 .astype(str):它在 Arrow 支撑的列上不填空,是个会误导人的假防护
            # （见 _as_text）。缺失值的处理归归一化函数自己。
            norm = df["doi"].map(_normalize_doi)
            hit = df[norm == doi]
            if len(hit):
                row = hit.iloc[0]
                return str(row.get("work_id") or ""), doi, title or str(row.get("title") or "")
        # high-confidence title match (exact normalized)
        if title and "title" in cols:
            ntitle = _normalize_title(title)
            if ntitle:
                norm = df["title"].map(_normalize_title)  # 同上，_as_text 兜底
                hit = df[norm == ntitle]
                if len(hit):
                    row = hit.iloc[0]
                    return str(row.get("work_id") or ""), doi, title

    # nothing resolved → synthetic stable id (never refuse)
    return _synthetic_id(title, first_author, year, doi), doi, title


# ── 3. CHUNK ─────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str
    page: int
    char_start: int
    char_end: int
    text: str
    n_tokens: int


def chunk_text(
    extracted: ExtractedText,
    *,
    chunk_tokens: int = _CHUNK_TOKENS,
    overlap: int = _CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a paper for retrieval (§7 step 3).

    Splits at **page boundaries first, then at section headings**, and only
    windows what is left over inside each segment.

    What that means in practice, measured on real papers: a page of body text
    runs 110–440 words, well under the 800-token window, so most segments emit a
    single chunk and ``overlap`` never comes into play. The effective granularity
    is **one page-section per chunk, with no overlap between chunks** — the
    windowing only bites on unusually long segments.

    This is a reasonable retrieval granularity (a section is a coherent unit of
    meaning, which a blind 800-word window is not), but it has a consequence
    worth knowing: a sentence spanning a page break is cut in two, so a method
    described across a page boundary can be missed by passage search. Said
    plainly here because the parameter names promise a sliding window and the
    files on disk are not one.

    Token counting is whitespace-word based (deterministic, dep-free). Character
    offsets are exact: ``full_text[c.char_start:c.char_end] == c.text``.
    """
    chunks: list[Chunk] = []
    if overlap >= chunk_tokens:
        overlap = max(0, chunk_tokens // 4)

    char_base = 0  # running char offset across the joined full text ("\n".join)
    idx = 0
    for page_no, page_text in enumerate(extracted.pages):
        # Split page into heading-delimited segments so headings start a chunk.
        segments = _split_on_headings(page_text)
        seg_offset = 0  # char offset within this page
        for seg in segments:
            if not seg.strip():
                seg_offset += len(seg)
                continue
            sub = _window_segment(
                seg,
                page_no,
                char_base + seg_offset,
                chunk_tokens,
                overlap,
            )
            for ch in sub:
                ch.chunk_id = f"c{idx:05d}"
                idx += 1
                chunks.append(ch)
            seg_offset += len(seg)
        # +1 for the "\n" join separator between pages
        char_base += len(page_text) + 1
    return chunks


def _split_on_headings(text: str) -> list[str]:
    """Split *text* so each section heading begins a new segment."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [text]
    segs: list[str] = []
    prev = 0
    for m in matches:
        start = m.start()
        if start > prev:
            segs.append(text[prev:start])
        prev = start
    segs.append(text[prev:])
    return segs


def _window_segment(
    seg: str, page_no: int, char_base: int, chunk_tokens: int, overlap: int
) -> list[Chunk]:
    """Sliding word-window over one segment; preserves char offsets."""
    # Tokenize with offsets so we can map back to char positions.
    tokens: list[tuple[str, int, int]] = []  # (word, start, end) char offsets
    for m in re.finditer(r"\S+", seg):
        tokens.append((m.group(0), m.start(), m.end()))
    out: list[Chunk] = []
    if not tokens:
        return out
    step = max(1, chunk_tokens - overlap)
    i = 0
    n = len(tokens)
    while i < n:
        window = tokens[i : i + chunk_tokens]
        c_start = window[0][1]
        c_end = window[-1][2]
        text = seg[c_start:c_end]
        out.append(
            Chunk(
                chunk_id="",  # set by caller
                page=page_no,
                char_start=char_base + c_start,
                char_end=char_base + c_end,
                text=text,
                n_tokens=len(window),
            )
        )
        if i + chunk_tokens >= n:
            break
        i += step
    return out


# ── 4. EMBED ─────────────────────────────────────────────────────────
# An embedder is ``Callable[[list[str]], <array-like (N, 1024)>]``. The default
# calls DashScope; tests inject a deterministic fake. We never call DashScope in
# this module unless the default embedder is actually used (and a key exists).
Embedder = Callable[[Sequence[str]], Any]


def _l2_normalize(arr: Any) -> Any:
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy unavailable")
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (a / norms).astype(np.float32)


def _default_dashscope_embedder(texts: Sequence[str]) -> Any:
    """Batched DashScope text-embedding-v3 → (N, 1024) float32, L2-normalized.

    Only used when no embedder is injected. Imports httpx lazily and resolves
    the key the same way literature_index does. Raises on missing key/dep so the
    caller degrades to an ``error:`` status rather than silently faking vectors.
    """
    if np is None:
        raise RuntimeError("numpy unavailable")
    import httpx  # lazy

    key = _load_dashscope_key()
    rows: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = list(texts[start : start + _EMBED_BATCH])
        body = {"model": _MODEL, "input": batch, "dimensions": _DIM}
        with httpx.Client(timeout=60.0) as c:
            r = c.post(
                _DASHSCOPE_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        # API may return out of order; sort by index
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        for it in items:
            rows.append(it["embedding"])
    return _l2_normalize(np.asarray(rows, dtype=np.float32))


def _load_dashscope_key() -> str:
    """DashScope key via the CANONICAL ``config`` loader (env → ``dashscope.env``).

    Same fix as ``literature_index._load_dashscope_key``: the index/exe root (``paths.base_dir()``) is NOT where ``api key/``
    lives — that sits under ``MAST2_PROJECT_ROOT`` (the installer's
    ``data_dir.txt`` data root). Probing the wrong root made ingest embedding
    fail on a machine that *has* a key.
    """
    from mast.knowledge._dashscope_key import require_dashscope_key

    return require_dashscope_key()


def embed_texts(texts: Sequence[str], *, embedder: Optional[Embedder] = None) -> Any:
    """Embed *texts* → (N, 1024) float32 L2-normalized, via injected/default embedder."""
    if not texts:
        if np is None:  # pragma: no cover
            raise RuntimeError("numpy unavailable")
        return np.zeros((0, _DIM), dtype=np.float32)
    fn = embedder or _default_dashscope_embedder
    arr = fn(texts)
    return _l2_normalize(arr)


# ── 5+6. ATTACH (per-library) + PROMOTE (big library) ────────────────
def _append_library_index(
    library_id: str,
    work_id: str,
    source: str,
    abstract_vec: Any,
    chunk_vecs: Any,
    chunks: list[Chunk],
    slug: str,
) -> Optional[str]:
    """ATTACH (§7 step 5) — **DORMANT: not called by any default flow.**

    Appends abstract + chunk vectors to a per-library index under
    ``artifacts/literature_libs/<library_id>/`` and updates its metadata.

    Why it is dormant rather than deleted (design §4.6, 2026-07-29). The feature
    was dead three times over and had been for its whole life:

      * the frontend always passed ``library_id=""``;
      * this function returns ``None`` immediately on an empty ``library_id``;
      * both libraries on the real machine had ``index_dir = null``.

    So no per-library index was ever built, and member-filtered search (search the
    big library, then filter to the ≤500 member pointers) has been carrying the
    load all along at an accepted cost. Chunk-level retrieval material already
    lives in ``data/papers/`` independent of any library index.

    It is kept — with its tests — because it is correct code for a decision that
    could plausibly be revisited (a library big enough for the member filter to
    hurt). **We do not fix, tune, or wire a path nobody walks**; if you revive it,
    also revive writing ``index_dir`` on the library record.

    Returns the library index dir path string (or None when no library_id).
    Idempotent-ish: removes any prior rows for this work_id before appending so
    a re-ingest replaces rather than duplicates.
    """
    if not library_id or np is None or pd is None:
        return None
    lib_dir = _libs_dir() / _slug_for_work_id(library_id)
    vec_path = lib_dir / "vectors.npy"
    meta_path = lib_dir / "metadata.parquet"

    # Build the new rows for this work_id.
    new_vecs = [abstract_vec.reshape(1, -1)]
    new_meta_rows = [
        {
            "work_id": work_id,
            "kind": "abstract",
            "chunk_id": "",
            "source_path": f"data/papers/{slug}/",
            "source": source,
        }
    ]
    for ch, _ in zip(chunks, range(len(chunks))):
        new_meta_rows.append(
            {
                "work_id": work_id,
                "kind": "chunk",
                "chunk_id": ch.chunk_id,
                "source_path": f"data/papers/{slug}/",
                "source": source,
            }
        )
    if len(chunks):
        new_vecs.append(np.asarray(chunk_vecs, dtype=np.float32))
    new_vec = np.vstack(new_vecs).astype(np.float32)
    new_meta = pd.DataFrame(new_meta_rows)

    # Load existing, drop any prior rows for this work_id (replace semantics).
    if vec_path.exists() and meta_path.exists():
        try:
            old_vec = np.load(vec_path)
            old_meta = pd.read_parquet(meta_path)
            n_old = min(len(old_meta), old_vec.shape[0])
            if n_old < len(old_meta) or n_old < old_vec.shape[0]:
                # vec/meta lengths drifted (a prior crash mid-write, etc.).
                # Do NOT discard the whole library — that would silently lose
                # every OTHER work_id's vectors. Truncate both to the aligned
                # prefix so we keep the rows we can trust, then proceed.
                logger.warning(
                    "per-library index (%s) vec/meta length mismatch "
                    "(vec=%d meta=%d); truncating to aligned prefix %d to "
                    "preserve existing data",
                    library_id, old_vec.shape[0], len(old_meta), n_old,
                )
                old_vec = old_vec[:n_old]
                old_meta = old_meta.iloc[:n_old].reset_index(drop=True)
            if n_old:
                keep = (old_meta["work_id"].astype(str) != work_id).values
                old_vec = old_vec[keep]
                old_meta = old_meta[keep].reset_index(drop=True)
                merged_vec = np.vstack([old_vec, new_vec]).astype(np.float32)
                merged_meta = pd.concat([old_meta, new_meta], ignore_index=True)
            else:
                merged_vec, merged_meta = new_vec, new_meta
        except Exception as e:  # pragma: no cover - corrupt index → rebuild
            logger.warning("rebuilding per-library index (%s): %s", library_id, e)
            merged_vec, merged_meta = new_vec, new_meta
    else:
        merged_vec, merged_meta = new_vec, new_meta

    _atomic_write_npy(vec_path, merged_vec)
    _atomic_write_parquet(meta_path, merged_meta)
    _atomic_write_json(
        lib_dir / "manifest.json",
        {
            "library_id": library_id,
            "model": _MODEL,
            "dim": _DIM,
            "n_rows": int(merged_vec.shape[0]),
        },
    )
    return str(lib_dir)


def _promote_to_big(
    work_id: str,
    doi: str,
    title: str,
    source: str,
    abstract_text: str,
    abstract_vec: Any,
    year: Any = 0,
) -> bool:
    """PROMOTE (§7.5, Decision 1 + keep-both Decision 2): upsert the abstract /
    metadata into the BIG library, append the abstract vector to the tail of
    ``vectors.npy``, index-aligned rows to ``metadata.parquet`` and a work_id
    row to ``abstracts.parquet``.

    Dedup by work_id then normalized DOI: if the paper already exists, keep BOTH
    abstracts (set ``user_abstract`` / ``fulltext_excerpt`` without overwriting
    the OpenAlex ``abstract``) and append a *second* vector row marked
    ``kind="user_abstract"``. A genuinely new paper occupies the ``abstract``
    slot with ``abstract_provenance="user"``.

    Returns True when a promotion/append happened, False when skipped (deps
    missing, or an in-place metadata refresh with no new vector needed).

    Atomicity (§7.5 step 5): all writes are temp+rename; before extending we
    truncate ``vectors.npy`` to the metadata length if a prior crash left them
    out of sync.
    """
    if np is None or pd is None:
        return False

    vec_path = _big_index_dir() / "vectors.npy"
    meta_path = _big_index_dir() / "metadata.parquet"
    abs_path = _big_index_dir() / "abstracts.parquet"

    # When the big index hasn't been provisioned, we still maintain abstracts +
    # a vectors/metadata pair so user contributions are not lost. (Base block is
    # never touched here; we only append.)
    has_vec = vec_path.exists()
    has_meta = meta_path.exists()

    # ── metadata.parquet (index-aligned) ──
    if has_meta:
        meta = pd.read_parquet(meta_path)
    else:
        meta = pd.DataFrame(
            columns=["work_id", "doi", "title", "year", "journal", "cited", "source", "kind"]
        )
    for col, default in (("source", "openalex"), ("kind", "abstract")):
        if col not in meta.columns:
            meta[col] = default

    # ── vectors.npy ──
    if has_vec:
        vec = np.load(vec_path)
        # crash-recovery truncation (§7.5 step 5)
        if vec.shape[0] != len(meta):
            keep = min(vec.shape[0], len(meta))
            logger.warning(
                "big index vec/meta mismatch (%d vs %d) → truncating to %d",
                vec.shape[0], len(meta), keep,
            )
            vec = vec[:keep]
            meta = meta.iloc[:keep].reset_index(drop=True)
    else:
        vec = np.zeros((0, _DIM), dtype=np.float32)

    avec = _l2_normalize(abstract_vec)  # (1, dim)

    ndoi = _normalize_doi(doi)
    existing_mask = (meta["work_id"].astype(str) == work_id)
    if not existing_mask.any() and ndoi and "doi" in meta.columns:
        # 同 resolve_work_id：不用 .astype(str)，缺失值由 _normalize_doi/_as_text 兜
        existing_mask = meta["doi"].map(_normalize_doi) == ndoi

    is_existing = bool(existing_mask.any())

    if is_existing:
        # KEEP BOTH (Decision 2): append a second vector row marked user_abstract
        new_meta_row = {
            "work_id": work_id,
            "doi": doi,
            "title": title,
            "year": _int_or_zero(year),
            "journal": "",
            "cited": 0,
            "source": source,
            "kind": "user_abstract",
        }
        kind = "user_abstract"
    else:
        new_meta_row = {
            "work_id": work_id,
            "doi": doi,
            "title": title,
            "year": _int_or_zero(year),
            "journal": "",
            "cited": 0,
            "source": source,
            "kind": "abstract",
        }
        kind = "abstract"

    new_vec = np.vstack([vec, avec.astype(np.float32)]).astype(np.float32)
    new_meta = pd.concat([meta, pd.DataFrame([new_meta_row])], ignore_index=True)

    _update_big_abstracts(abs_path, work_id, doi, title, abstract_text, is_existing, year)

    # Write metadata first, then vectors — on a crash between them, the load-time
    # truncation above repairs (extra meta row dropped on next load? no: we want
    # vec ≤ meta, so write meta first then vec, and truncate vec→meta on recovery
    # which would drop the new vec row, leaving a harmless dangling meta row that
    # the next promotion overwrites). Simpler + safe: write both atomically here.
    _atomic_write_parquet(meta_path, new_meta)
    _atomic_write_npy(vec_path, new_vec)
    logger.info("promoted %s to big library (kind=%s, n=%d)", work_id, kind, new_vec.shape[0])
    return True


def _update_big_abstracts(
    abs_path: Path,
    work_id: str,
    doi: str,
    title: str,
    abstract_text: str,
    is_existing: bool,
    year: Any,
) -> None:
    """Upsert into abstracts.parquet keeping BOTH abstracts (Decision 2)."""
    if pd is None:
        return
    if abs_path.exists():
        adf = pd.read_parquet(abs_path)
    else:
        adf = pd.DataFrame(
            columns=[
                "work_id", "doi", "title", "abstract", "user_abstract",
                "fulltext_excerpt", "abstract_provenance", "source", "year",
            ]
        )
    for col in (
        "work_id", "doi", "title", "abstract", "user_abstract",
        "fulltext_excerpt", "abstract_provenance", "source", "year",
    ):
        if col not in adf.columns:
            adf[col] = ""

    excerpt = (abstract_text or "")[:4000]
    row_mask = adf["work_id"].astype(str) == work_id
    if row_mask.any():
        # existing work_id row → add user_abstract WITHOUT overwriting `abstract`
        i = adf.index[row_mask][0]
        adf.at[i, "user_abstract"] = excerpt
        adf.at[i, "fulltext_excerpt"] = excerpt
        has_oa = bool(str(adf.at[i, "abstract"] or "").strip())
        adf.at[i, "abstract_provenance"] = "openalex+user" if has_oa else "user"
    else:
        # new work_id row → user abstract occupies the `abstract` slot
        new = {
            "work_id": work_id,
            "doi": doi,
            "title": title,
            "abstract": "" if is_existing else excerpt,
            "user_abstract": excerpt,
            "fulltext_excerpt": excerpt,
            "abstract_provenance": "openalex+user" if is_existing else "user",
            "source": "user_pdf",
            "year": _int_or_zero(year),
        }
        adf = pd.concat([adf, pd.DataFrame([new])], ignore_index=True)

    _atomic_write_parquet(abs_path, adf)


# ── Full-text back-reference (revives two dead schema fields) ────────
def record_fulltext(library_id: str, work_id: str, slug: str, *,
                    experiment_id: str = "", status: str = "ingested") -> str:
    """Point a library member at the full text we just stored. Returns the ref.

    ``fulltext_status`` / ``fulltext_ref`` were **schema slots that no production
    code had ever written** (2026-07-29 audit): ``ingest_pdf`` stored the PDF under
    ``data/papers/<slug>/`` and promoted the abstract into the big library, but
    never told the member row about it. So "do we have the full text of this one?"
    was unanswerable from the library — the operator had to go look in the folder.
    This is the one place that closes that loop; ``fetch_board`` fulfilment is the
    other caller.

    The ref is stored **relative** (``papers/<slug>``): full text is a machine-level
    asset that does NOT travel with the experiment folder, so an absolute path
    would break the moment the folder is copied to another machine, while the
    relative form still resolves against ``paths.papers_dir()``.

    Target library: explicit ``library_id`` > the effective library (that
    experiment's own library, else the manual pointer). Experiment libraries get
    an event in their ``members.jsonl``; global / custom libraries get the two
    fields written on the registry member. Best-effort throughout — a curation
    hiccup must never fail an ingest whose bytes are already safely on disk.
    """
    wid = (work_id or "").strip()
    if not wid or not slug:
        return ""
    ref = f"papers/{slug}"
    try:
        from mast.knowledge import experiment_library as expl
        from mast.knowledge import libraries as lib_mod

        target = (library_id or "").strip()
        eid = (experiment_id or "").strip()
        if not target and not eid:
            # Whoever asked for this paper decides where it is filed, not whoever
            # happens to be active now (trap ⑯). A fetch request raised during
            # experiment A can be fulfilled days later while B is running; the
            # experiment_id frozen on the request row is the right drawer.
            try:
                from mast.knowledge.fetch_board import open_experiment_ids
                waiting = [x for x in open_experiment_ids(wid) if x]
                if waiting:
                    eid = waiting[0]
            except Exception as exc:  # noqa: BLE001 — board absent → current scope
                logger.debug("fetch-board lookup for %s skipped: %r", wid, exc)
        if not target:
            # No ensure: knowing WHICH library is a pure lookup (the id is derived
            # from the experiment id). ``set_fulltext`` below creates it if needed.
            target, source = expl.resolve_effective_library(
                experiment_id=eid or None)
            if not eid and source == "experiment":
                eid = expl.active_experiment_id()
        if not eid:
            try:
                rec = lib_mod.get_library(target)
                eid = str(rec.get("experiment_id") or "")
            except Exception:  # noqa: BLE001 — unknown library → registry path
                eid = ""
        if eid:
            out = expl.set_fulltext(eid, wid, status, ref)
            if not out.get("ok"):
                logger.info("full-text ref not recorded for %s: %s",
                            wid, out.get("error"))
                return ""
            return ref
        if lib_mod.get_registry().set_member_fulltext(target, wid, status, ref):
            return ref
        # Not a member of the target library — the paper is still ingested and in
        # the big library; there is simply no pointer row to annotate. Say so in
        # the log rather than inventing one (adding a member here would make an
        # ingest silently curate).
        logger.info("full-text ref skipped: %s is not a member of %r", wid, target)
    except Exception as exc:  # noqa: BLE001 — never fail an ingest over curation
        logger.warning("record_fulltext(%s) failed: %r", wid, exc)
    return ""


# ── Top-level pipeline ───────────────────────────────────────────────
def ingest_pdf(
    pdf_path: str,
    work_id: str = "",
    library_id: str = "",
    *,
    embedder: Optional[Embedder] = None,
    extractor: Optional[Callable[[str | Path], ExtractedText]] = None,
    source: str = "user_pdf",
    promote: bool = True,
    first_author: str = "",
    year: Any = "",
    abstracts_df: Optional["pd.DataFrame"] = None,  # type: ignore[name-defined]
    ocr: bool = True,
    ocr_fn: Optional[Callable[[str | Path], str]] = None,
    experiment_id: str = "",
) -> IngestResult:
    """Run the full §7 ingestion pipeline on a PDF already on disk.

    EXTRACT → RESOLVE → CHUNK → EMBED → PROMOTE → DEDUP → record full text.
    (ATTACH is dormant — see the module docstring, step 5.)

    Args:
        pdf_path:    path to a PDF on disk (copied into data/papers/<slug>/).
        work_id:     explicit OpenAlex work_id; resolved if empty (§7.4), and a
                     synthetic ``local:<hash>`` minted if nothing resolves.
        library_id:  the library whose member should get the full-text
                     back-reference (``fulltext_status="ingested"``,
                     ``fulltext_ref="papers/<slug>"``). Empty → the effective
                     library (current experiment's, else the manual pointer).
                     It no longer triggers any per-library index build.
        experiment_id: pin the target library to one experiment instead of asking
                     "who is active now". ``fetch_board`` passes the id frozen on
                     the REQUEST row: a request raised during experiment A may be
                     fulfilled days later while B is active, and filing the paper
                     under B would be simply the wrong drawer.
        embedder:    injectable ``Callable[[list[str]], (N,1024)]``. None → the
                     default DashScope embedder (needs a key + httpx). Tests pass
                     a deterministic fake — DashScope is never called here.
        extractor:   injectable ``Callable[[path], ExtractedText]``. None →
                     ``extract_pdf`` (needs fitz). Tests pass a fake to avoid a
                     real PDF dependency.
        source:      "user_pdf" | "user_url" | "agent".
        promote:     whether to promote the abstract into the big library (§7.5).
        first_author/year: hints used only when minting a synthetic id.
        abstracts_df: injectable big-library abstracts frame for RESOLVE (tests).
        ocr:         when the text layer is (near-)empty, try qwen-vl-ocr to
                     recover text before refusing (scanned-PDF fallback). Set
                     False to disable (keeps the old honest-warning behaviour).
        ocr_fn:      injectable ``Callable[[path], str]`` for the OCR fallback.
                     None → ``knowledge.ocr.ocr_pdf`` (needs fitz + a DashScope
                     key). Tests pass a fake so DashScope is never called here.

    Returns a serializable ``IngestResult``. Never raises for missing deps /
    bad input — returns an ``error:<reason>`` status instead (graceful, §1.4).
    """
    src = pdf_path
    p = Path(pdf_path)
    if not p.is_file():
        return IngestResult(
            work_id=work_id, source=source, n_chunks=0, slug_dir="",
            promoted=False, status="error:pdf_not_found",
            detail=f"PDF not found: {pdf_path}", library_id=library_id,
        )

    if np is None or pd is None:
        return IngestResult(
            work_id=work_id, source=source, n_chunks=0, slug_dir="",
            promoted=False, status="error:numpy_or_pandas_missing",
            detail="numpy/pandas unavailable — cannot embed/index.",
            library_id=library_id,
        )

    # ── 1. EXTRACT ──
    extract = extractor or extract_pdf
    try:
        extracted = extract(src)
    except Exception as e:
        return IngestResult(
            work_id=work_id, source=source, n_chunks=0, slug_dir="",
            promoted=False, status="error:extract_failed",
            detail=str(e), library_id=library_id,
        )

    # ── 1b. OCR FALLBACK (scanned / image-only PDF — no/short text layer) ──
    # Runs BEFORE RESOLVE so an OCR-recovered title/DOI feed work-id resolution
    # and the slug/DEDUP path stay stable across re-uploads. Fully graceful: any
    # OCR failure (no key / no dep / network) leaves `extracted` untouched and the
    # honest `warning:no_text_layer` guard below still fires. Injectable for tests.
    ocr_used = False
    _extract_nonspace = "".join((extracted.full_text or "").split())
    if ocr and len(_extract_nonspace) < _MIN_INGEST_TEXT_CHARS:
        ocr_call = ocr_fn
        if ocr_call is None:
            try:
                from mast.knowledge import ocr as _ocr_mod

                ocr_call = _ocr_mod.ocr_pdf
            except Exception as e:  # pragma: no cover - ocr module import guard
                logger.warning("OCR module unavailable: %s", e)
                ocr_call = None
        if ocr_call is not None:
            try:
                ocr_text = ocr_call(src) or ""
            except Exception as e:  # ocr_pdf is documented never-raise; defensive
                logger.warning("OCR fallback raised for %s: %s", src, e)
                ocr_text = ""
            # Accept OCR only when it clears the SAME bar a real text layer must —
            # a partial/garbled result must not sneak a junk row into the index.
            if len("".join(ocr_text.split())) >= _MIN_INGEST_TEXT_CHARS:
                extracted = _extracted_from_ocr(ocr_text, extracted)
                ocr_used = True
                logger.info(
                    "OCR recovered %d chars from scanned PDF %s",
                    len("".join(ocr_text.split())), Path(pdf_path).name,
                )

    # ── 2. RESOLVE ──
    resolved_id, doi, title = resolve_work_id(
        extracted,
        explicit_work_id=work_id,
        abstracts_df=abstracts_df,
        first_author=first_author,
        year=year,
    )
    if not resolved_id:
        resolved_id = _synthetic_id(title, first_author, year, doi)
    slug = _slug_for_work_id(resolved_id)
    slug_dir = _papers_dir() / slug

    # ── 7. DEDUP (sha256 re-upload guard, §7.4) — compute early ──
    try:
        pdf_sha = _sha256_file(p)
    except OSError as e:
        return IngestResult(
            work_id=resolved_id, source=source, n_chunks=0, slug_dir="",
            promoted=False, status="error:pdf_read_failed", detail=str(e),
            doi=doi, title=title, library_id=library_id, ocr_used=ocr_used,
        )

    meta_json_path = slug_dir / "meta.json"
    if meta_json_path.exists():
        try:
            prior = json.loads(meta_json_path.read_text(encoding="utf-8"))
        except Exception:
            prior = {}
        if prior.get("sha256") == pdf_sha and (slug_dir / "chunks.parquet").exists():
            # identical re-upload for the same work_id → no-op (§7.4)
            try:
                n_prior = int(pd.read_parquet(slug_dir / "chunks.parquet").shape[0])
            except Exception:
                n_prior = 0
            return IngestResult(
                work_id=resolved_id, source=source, n_chunks=n_prior,
                slug_dir=str(slug_dir), promoted=bool(prior.get("promoted_to_big")),
                status="noop", detail="identical PDF already ingested",
                doi=doi, title=title, sha256=pdf_sha, library_id=library_id,
                ocr_used=ocr_used,
            )
        replaced = True  # different PDF for same work_id → replace (§7.4)
    else:
        replaced = False

    # ── 3. CHUNK ──
    chunks = chunk_text(extracted)
    abstract_text = _derive_abstract(extracted, chunks)

    # ── Guard: scanned / no-text-layer PDF → honest warning, do NOT promote ──
    # An image-only PDF extracts (near-)empty text; embedding "" and appending a
    # blank-abstract row would silently pollute the big index while the UI shows
    # a green "success". By here the OCR fallback (step 1b) has already run and
    # either upgraded `extracted` to recovered text (guard passes) or came up
    # short / was unavailable — in which case we refuse with a clear status.
    #
    # The canonical signal is the document-wide non-whitespace char count. We do
    # NOT also require an empty `abstract_text`: `_derive_abstract` falls back to
    # `full[:1500]`, so a scanned PDF whose text layer leaks a few stray glyphs
    # (watermark, page number, garbled OCR) yields a non-empty abstract_text and
    # would short-circuit this guard — letting the junk PDF into the big index.
    _nonspace = "".join((extracted.full_text or "").split())
    if len(_nonspace) < _MIN_INGEST_TEXT_CHARS:
        _ocr_note = (
            "（已尝试 OCR 但仍无足量文本）" if ocr
            else "（OCR 未启用）"
        )
        return IngestResult(
            work_id=resolved_id, source=source, n_chunks=0, slug_dir="",
            promoted=False, status="warning:no_text_layer",
            detail=(f"提取到的文本过少（{len(_nonspace)} 个非空白字符）{_ocr_note}，疑似扫描版 / "
                    "图片型 PDF 或无文本层，未嵌入大库。请提供文字版 PDF，或配置 DashScope "
                    "key 后重试 OCR。"),
            doi=doi, title=title, sha256=pdf_sha, library_id=library_id,
            ocr_used=ocr_used,
        )

    # ── 4. EMBED ──
    try:
        to_embed = [abstract_text] + [c.text for c in chunks]
        all_vecs = embed_texts(to_embed, embedder=embedder)
    except Exception as e:
        return IngestResult(
            work_id=resolved_id, source=source, n_chunks=len(chunks),
            slug_dir="", promoted=False, status="error:embed_failed",
            detail=str(e), doi=doi, title=title, sha256=pdf_sha,
            library_id=library_id, ocr_used=ocr_used,
        )
    abstract_vec = all_vecs[0:1]
    chunk_vecs = all_vecs[1:] if len(chunks) else np.zeros((0, _DIM), dtype=np.float32)

    # ── Persist per-paper artifacts (data/papers/<slug>/) ──
    try:
        slug_dir.mkdir(parents=True, exist_ok=True)
        # copy the source PDF in (atomic)
        _atomic_write_bytes(slug_dir / "source.pdf", p.read_bytes())
        _atomic_write_text(slug_dir / "fulltext.txt", extracted.full_text)
        chunks_df = pd.DataFrame(
            [
                {
                    "chunk_id": c.chunk_id, "page": c.page,
                    "char_start": c.char_start, "char_end": c.char_end,
                    "text": c.text, "n_tokens": c.n_tokens,
                }
                for c in chunks
            ],
            columns=["chunk_id", "page", "char_start", "char_end", "text", "n_tokens"],
        )
        _atomic_write_parquet(slug_dir / "chunks.parquet", chunks_df)
        _atomic_write_npy(slug_dir / "chunk_vectors.npy", np.asarray(chunk_vecs, dtype=np.float32))
        _atomic_write_npy(slug_dir / "abstract_vector.npy", np.asarray(abstract_vec, dtype=np.float32))
    except Exception as e:
        return IngestResult(
            work_id=resolved_id, source=source, n_chunks=len(chunks),
            slug_dir="", promoted=False, status="error:write_failed",
            detail=str(e), doi=doi, title=title, sha256=pdf_sha,
            library_id=library_id, ocr_used=ocr_used,
        )

    # ── 5. ATTACH — dormant (design §4.6). ``library_id`` now means "member
    # pointer + full-text ref", never "build a second vector index".
    # ``_append_library_index`` is kept but no longer called; see its docstring.
    lib_dir = None

    # ── 6. PROMOTE (big library) ──
    promoted = False
    if promote:
        try:
            promoted = _promote_to_big(
                resolved_id, doi, title, source, abstract_text,
                np.asarray(abstract_vec, dtype=np.float32), year=year,
            )
            # CRITICAL: _promote_to_big appends to the on-disk big index, but the
            # in-memory search cache (vectors + abstracts) stays stale, so a
            # just-ingested paper was NOT findable until a full restart. Drop the
            # cache so the next search re-reads the updated index — closes the
            # "user adds → goes into 大库 → findable this session" loop.
            if promoted:
                try:
                    from mast.knowledge import literature_index
                    literature_index.invalidate_caches()
                except Exception as _e:  # pragma: no cover - non-fatal
                    logger.warning("invalidate_caches after promote failed: %s", _e)
        except Exception as e:  # pragma: no cover - non-fatal
            logger.warning("PROMOTE failed for %s: %s", resolved_id, e)

    # ── meta.json ──
    meta = {
        "work_id": resolved_id,
        "doi": doi,
        "title": title,
        "sha256": pdf_sha,
        "ingested_at": _utc_now(),
        "source": source,
        "promoted_to_big": bool(promoted),
        "n_chunks": len(chunks),
        "library_id": library_id,
        "ocr_used": bool(ocr_used),
    }
    try:
        _atomic_write_json(meta_json_path, meta)
    except Exception as e:  # pragma: no cover
        logger.warning("meta.json write failed for %s: %s", resolved_id, e)

    # ── full-text back-reference on the library member ──
    ft_ref = record_fulltext(library_id, resolved_id, slug,
                             experiment_id=experiment_id)

    _detail = f"fulltext_ref={ft_ref}" if ft_ref else ""
    if ocr_used:
        _detail = (_detail + "；" if _detail else "") + "扫描件，已用 qwen-vl-ocr 提取文本"
    return IngestResult(
        work_id=resolved_id,
        source=source,
        n_chunks=len(chunks),
        slug_dir=str(slug_dir),
        promoted=bool(promoted),
        status="replaced" if replaced else "ingested",
        doi=doi,
        title=title,
        sha256=pdf_sha,
        library_id=library_id,
        detail=_detail,
        ocr_used=ocr_used,
    )


def _write_manual_abstract_row(
    work_id: str,
    *,
    doi: str = "",
    title: str = "",
    abstract: str = "",
    authors: str = "",
    first_author: str = "",
    journal: str = "",
    year: Any = "",
    source: str = "user_manual",
) -> bool:
    """Upsert a RICH row into the big library's ``abstracts.parquet`` for a
    typed (no-PDF) manual entry — the field the operator sees back via
    ``fetch_abstract`` (title / abstract / authors / first_author / year).

    Keyed by ``work_id`` (NOT index-aligned), so it is safe to write on its own —
    it never touches ``vectors.npy`` / ``metadata.parquet`` alignment. Returns
    True on write, False when pandas is unavailable.
    """
    if pd is None:
        return False
    abs_path = _big_index_dir() / "abstracts.parquet"
    cols = [
        "work_id", "doi", "title", "abstract", "user_abstract",
        "fulltext_excerpt", "abstract_provenance", "source", "year",
        "authors", "first_author", "journal",
    ]
    if abs_path.exists():
        adf = pd.read_parquet(abs_path)
    else:
        adf = pd.DataFrame(columns=cols)
    for col in cols:
        if col not in adf.columns:
            adf[col] = ""
    excerpt = (abstract or "")[:4000]
    row_mask = adf["work_id"].astype(str) == work_id
    values = {
        "work_id": work_id,
        "doi": doi,
        "title": title,
        # A manual entry's text IS the abstract (user-authored) — occupies the
        # `abstract` slot with provenance "user".
        "abstract": excerpt,
        "user_abstract": excerpt,
        "fulltext_excerpt": excerpt,
        "abstract_provenance": "user",
        "source": source,
        "year": _int_or_zero(year),
        "authors": authors,
        "first_author": first_author,
        "journal": journal,
    }
    if row_mask.any():
        i = adf.index[row_mask][0]
        for k, v in values.items():
            adf.at[i, k] = v
    else:
        adf = pd.concat([adf, pd.DataFrame([values])], ignore_index=True)
    _atomic_write_parquet(abs_path, adf)
    return True


def ingest_manual(
    *,
    title: str = "",
    abstract: str = "",
    work_id: str = "",
    doi: str = "",
    first_author: str = "",
    authors: str = "",
    year: Any = "",
    journal: str = "",
    source: str = "user_manual",
    embedder: Optional[Embedder] = None,
    promote: bool = True,
) -> IngestResult:
    """Register a paper from TYPED metadata — NO PDF (#125 '直接管理条目').

    The operator hand-enters title + abstract (+ optional doi / authors / year)
    for a paper that is not in OpenAlex and has no PDF at hand. We:

      1. resolve a ``work_id`` — explicit arg > synthetic ``local:<hash>`` from
         title/first_author/year/doi (never refuse, §7.4);
      2. write a rich ``abstracts.parquet`` row so ``fetch_abstract`` returns the
         entry with its title / abstract / authors — works with NO DashScope key;
      3. BEST-EFFORT: if numpy/pandas + an embedder/key are available, embed the
         abstract and append a vector+metadata row to the big index so the entry
         is *semantically searchable* too (``_promote_to_big``). When embedding is
         unavailable the entry is still stored + retrievable, just not vector-
         searchable — status ``ingested_no_embed`` says so honestly.

    Returns a serializable :class:`IngestResult` (``status`` in
    ``ingested`` | ``ingested_no_embed`` | ``error:*``). Never raises.
    """
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if not title and not abstract:
        return IngestResult(
            work_id="", source=source, n_chunks=0, slug_dir="",
            promoted=False, status="error:empty",
            detail="需要至少提供标题或摘要。",
        )
    if pd is None:
        return IngestResult(
            work_id=work_id, source=source, n_chunks=0, slug_dir="",
            promoted=False, status="error:numpy_or_pandas_missing",
            detail="pandas 不可用 — 无法写入文献库。",
        )

    ndoi = _normalize_doi(doi)
    resolved_id = (work_id or "").strip() or _synthetic_id(
        title, first_author, year, ndoi
    )

    # 2. rich abstracts row (retrievable even with no embed key)
    try:
        _write_manual_abstract_row(
            resolved_id, doi=ndoi, title=title, abstract=abstract,
            authors=authors, first_author=first_author, journal=journal,
            year=year, source=source,
        )
    except Exception as e:
        return IngestResult(
            work_id=resolved_id, source=source, n_chunks=0, slug_dir="",
            promoted=False, status="error:write_failed", detail=str(e),
            doi=ndoi, title=title,
        )

    # 3. best-effort embed + promote (semantic search)
    promoted = False
    embed_note = ""
    if promote and np is not None:
        embed_text = abstract or title
        try:
            avec = embed_texts([embed_text], embedder=embedder)  # (1, dim)
            promoted = _promote_to_big(
                resolved_id, ndoi, title, source, abstract or title,
                np.asarray(avec, dtype=np.float32), year=year,
            )
            # _promote_to_big rewrites a THIN abstracts row → restore the rich one.
            _write_manual_abstract_row(
                resolved_id, doi=ndoi, title=title, abstract=abstract,
                authors=authors, first_author=first_author, journal=journal,
                year=year, source=source,
            )
        except Exception as e:  # no key / network / dep → keep the light entry
            embed_note = f"（未能向量化，仅登记为可按 work_id 检索：{e}）"
            logger.info("ingest_manual embed/promote skipped for %s: %s", resolved_id, e)

    # refresh the search caches so the new entry is visible this session
    try:
        from mast.knowledge import literature_index
        literature_index.invalidate_caches()
    except Exception:  # pragma: no cover - non-fatal
        pass

    status = "ingested" if promoted else "ingested_no_embed"
    detail = "已登记并向量化，可语义检索。" if promoted else (
        "已登记（可按 work_id 检索摘要）。" + embed_note
        if embed_note else "已登记（可按 work_id 检索摘要；未向量化，语义检索暂不可见）。"
    )
    return IngestResult(
        work_id=resolved_id, source=source, n_chunks=0, slug_dir="",
        promoted=bool(promoted), status=status, detail=detail,
        doi=ndoi, title=title,
    )


def _derive_abstract(extracted: ExtractedText, chunks: list[Chunk]) -> str:
    """Best-effort abstract text for promotion: the text under an ``abstract``
    heading if present, else the leading ~1500 chars of the full text."""
    full = extracted.full_text
    m = re.search(r"(?im)^\s*abstract\b", full)
    if m:
        start = m.end()
        nxt = _HEADING_RE.search(full, start + 1)
        end = nxt.start() if nxt else min(len(full), start + 2500)
        seg = full[start:end].strip(" :.\n\t")
        if seg:
            return seg[:2500]
    return full[:1500].strip()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: Public alias — the SI-attachment code needs the same work_id → directory
#: mapping ingest uses, and a second implementation of it would be a second
#: place for attachments and papers to disagree about where a paper lives.
slug_for_work_id = _slug_for_work_id

__all__ = [
    "IngestResult",
    "ExtractedText",
    "Chunk",
    "ingest_pdf",
    "ingest_manual",
    "extract_pdf",
    "resolve_work_id",
    "chunk_text",
    "embed_texts",
    "slug_for_work_id",
]
