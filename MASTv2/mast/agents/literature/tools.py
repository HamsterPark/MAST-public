"""Literature Reading agent tool list — Phase 6 real implementations + P1 library curation.

Live tools:
  1. search_papers(query, max_results)         — PyMuPDF full-text scan over
                                                  data/papers/ (or paths from
                                                  MAST_PAPER_CORPUS env)
  2. read_paper_section(paper_id, section)     — PyMuPDF page → heading slice
  3. extract_protocol(paper_id)                — heuristic protocol fields
                                                  pulled from methods section
  4. web_search(query, max_results)            — Tavily client (key from
                                                  MASTv2/../mast/api key/tavily.env
                                                  or TAVILY_API_KEY env)
  5. search_local_corpus(query, ...)           — semantic search over the big
                                                  OpenAlex 36,792-paper index
  6. buffer tools (if buf supplied)            — live tip / scan reads
  7. handoff_to_experiment_design              — forward prior-art to XD
  8. handoff_to_supervisor                     — return to orchestrator

P1 library-curation + read-only knowledge tools (no hardware, no instrument
safety path — bounded curation per design G8):
  9. lib_list()                                — enumerate libraries
 10. lib_create(name, scope)                   — mint a custom library
 11. lib_switch(library_id)                     — set the active library
 12. lib_add(work_ids, library_id, reason)     — add work_id pointers to a library
 13. lib_remove(work_ids, library_id)          — drop work_id pointers
 14. lib_search(query, library_id, k)          — big-index search, member-filtered
 15. fetch_paper_abstract(work_id)             — exact abstract lookup
 16. propose_citations(section_text, k)        — citation candidates (read-only)
 17. literature_priors(material, mode)         — quantitative param percentiles

Design (owner G): the big OpenAlex index is the ONE true library; every other
library is just a SET OF work_id pointers into it. There is no per-library
content store. In-library search = big-index search filtered down to the
library's member work_ids (no secondary index is ever built).

Cross-agent import rule: this file must NOT import from any other agent
package. Only mast.agents._shared.*, mast.agents.state and mast.knowledge.*
are permitted.

If the paper corpus is empty / missing, search_papers and read_paper_section
return informative "no corpus" notes — the tools still answer, the operator
just learns the corpus needs to be populated. Tavily fails closed (returns a
note) if the API key is missing. The library tools and abstract/citation/priors
tools are fully offline (registry.json + local parquet) and never need a key.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from langchain_core.tools import InjectedToolCallId, tool

from mast.agents._shared.artifact_channel import (
    ArtifactToolReturn,
    doc_ref_from_save,
)
from mast.agents._shared.buffer_tools import make_buffer_tools
from mast.agents._shared.handoff import make_handoff

if TYPE_CHECKING:
    from mast.buffer.service import BufferService

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Corpus + key resolution helpers
# ─────────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    """Walk up from this file to the repo root (the directory holding `MASTv2/`).

    Was "the directory holding both `mast/` and `MASTv2/`" — v1's `mast/` was
    archived in 2026-06, so that condition stopped matching anything and every
    call fell through to the `parents[4]` fallback. Same answer, by accident.
    Now it looks for what actually exists.
    """
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return p
        p = p.parent
    # fallback: 4 levels up (MASTv2/mast/agents/literature/tools.py → repo root)
    return Path(__file__).resolve().parents[4]


#: Sub-directory name that holds a paper's SI attachments. Files under it are
#: supplementary material OF the paper in the parent directory — never papers in
#: their own right, so corpus scans must skip them (see ``_all_pdfs``).
ATTACHMENTS_DIRNAME = "attachments"

#: Directory names under <papers> that are NOT papers. ``_incoming`` is where
#: ``knowledge.fetch`` stages a download before ingest gives it a work_id: the
#: staged copy is keyed by DOI and the ingested one by work_id, so without this
#: the same paper appears twice in the corpus under two different paper_ids.
_NON_PAPER_DIRS = frozenset({ATTACHMENTS_DIRNAME, "_incoming"})


def _corpus_dirs() -> list[Path]:
    """Every directory scanned for paper PDFs, most-canonical first.

    Three separate places used to disagree about where papers live, and the
    result was that a paper the system had just ingested could not be read back:

      * ``knowledge/paths.papers_dir()`` (``MASTv2/data/papers``, env
        ``MAST_PAPERS_DIR``) is where ``ingest_pdf`` WRITES;
      * this module used to look only at ``<repo>/data/papers`` (env
        ``MAST_PAPER_CORPUS``), which does not exist;
      * the operator's actual PDFs sit in ``<repo>/papers``.

    So ingest's output landed in a directory the reading tools never scanned.
    The fix is to scan all of them: ``papers_dir()`` is ALWAYS included, and
    ``MAST_PAPER_CORPUS`` now **adds** directories rather than replacing the
    canonical one (a corpus override must not be able to hide ingested papers).
    Legacy locations are included only when they exist, so they cost nothing on
    installs that never had them.
    """
    out: list[Path] = []

    def _add(d: Path) -> None:
        try:
            rd = d.expanduser().resolve()
        except Exception:  # noqa: BLE001 — an unresolvable path is just skipped
            return
        if rd not in out:
            out.append(rd)

    try:
        from mast.knowledge.paths import papers_dir
        _add(papers_dir())
    except Exception as exc:  # noqa: BLE001 — never let path resolution kill a tool
        logger.debug("papers_dir() unavailable: %s", exc)

    env = os.environ.get("MAST_PAPER_CORPUS", "")
    if env:
        sep = ";" if os.name == "nt" else ":"
        for part in env.split(sep):
            if part.strip():
                _add(Path(part))
    else:
        # Legacy corpora — only when present, and only when the operator has not
        # pointed MAST_PAPER_CORPUS somewhere explicit (tests set it to an empty
        # tmp dir precisely to keep the real corpus out of reach).
        root = _repo_root()
        for legacy in (root / "data" / "papers", root / "papers"):
            if legacy.is_dir():
                _add(legacy)
    return out


def _all_pdfs() -> list[Path]:
    """Every PDF that counts as a *paper*.

    Attachments and fetch staging are skipped — both hold files that belong to a
    paper rather than being one, and listing them turns each into a phantom paper
    with its own paper_id.
    """
    pdfs: list[Path] = []
    for d in _corpus_dirs():
        if not d.is_dir():
            continue
        for p in d.rglob("*.pdf"):
            if _NON_PAPER_DIRS & set(p.parts):
                continue
            pdfs.append(p)
    return pdfs


def _paper_id(p: Path) -> str:
    """The stable identifier for a PDF on disk.

    ``ingest_pdf`` stores every paper as ``<papers>/<slug>/source.pdf``, so the
    file stem is ``source`` for ALL of them — every ingested paper would answer
    to the same paper_id and ``_find_paper`` would return whichever came first.
    For that layout the slug (the directory name) is the identity. Hand-dropped
    PDFs keep their stem, so existing references still resolve.
    """
    if p.name.lower() == "source.pdf" and p.parent is not None:
        return p.parent.name
    return p.stem


def _find_paper(paper_id: str) -> Path | None:
    for p in _all_pdfs():
        if _paper_id(p) == paper_id:
            return p
    return None


#: Legacy naming for supplementary material dropped next to a paper by hand:
#: ``Foo.pdf`` + ``Foo_SI.pdf`` / ``Foo-SI.pdf`` / ``Foo_supplementary.pdf``.
_LEGACY_SI_RX = re.compile(r"[_-](si|supp(l(ementary|ement)?)?(\s*info(rmation)?)?)\d*$",
                           re.IGNORECASE)


def _is_legacy_si(p: Path) -> bool:
    return bool(_LEGACY_SI_RX.search(p.stem))


def _legacy_si_siblings(pdf: Path) -> list[Path]:
    """Hand-dropped SI files that belong to ``pdf`` (``Foo.pdf`` → ``Foo_SI.pdf``)."""
    if pdf is None or not pdf.parent.is_dir():
        return []
    stem = pdf.stem.lower()
    out: list[Path] = []
    for sib in sorted(pdf.parent.glob("*.pdf")):
        if sib == pdf or not _is_legacy_si(sib):
            continue
        base = _LEGACY_SI_RX.sub("", sib.stem).lower()
        if base == stem:
            out.append(sib)
    return out


def _ingested_dir(paper_id: str) -> Path | None:
    """``<papers>/<slug>/`` for an ingested paper, else None."""
    try:
        from mast.knowledge.paths import papers_dir
        d = papers_dir() / paper_id
    except Exception:  # noqa: BLE001
        return None
    return d if d.is_dir() else None


def _cached_fulltext(paper_id: str) -> str:
    """The text ``ingest_pdf`` extracted for this paper, or ''.

    Prefer this over re-opening the PDF: for a scanned paper the text layer is
    empty and the only readable text is what OCR produced at ingest time and
    wrote to ``fulltext.txt``. Re-extracting with PyMuPDF would silently yield
    nothing and read like "the paper says nothing about it".
    """
    d = _ingested_dir(paper_id)
    if d is None:
        return ""
    f = d / "fulltext.txt"
    try:
        return f.read_text(encoding="utf-8", errors="replace") if f.is_file() else ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("fulltext.txt unreadable for %s: %s", paper_id, exc)
        return ""


def _paper_text(paper_id: str) -> tuple[str, str, str]:
    """``(text, kind, detail)`` for a paper: cached full text, else live PDF parse.

    ``kind`` is one of:

      * ``"fulltext"`` — served from the cached ``fulltext.txt`` (authoritative
        for scanned papers, whose only readable text came from ingest-time OCR);
      * ``"pdf"``      — parsed live out of the PDF;
      * ``"missing"``  — no such paper / no readable text (a fact ABOUT the corpus);
      * ``"error"``    — the read itself blew up (a fact about our machinery).

    The last two must stay distinguishable all the way to the caller. Collapsing
    them lets "we could not read this paper" get reported as "this paper does not
    mention it", which is a statement about the science that we have no basis for.
    """
    cached = _cached_fulltext(paper_id)
    if cached.strip():
        return cached, "fulltext", "fulltext.txt"

    pdf = _find_paper(paper_id)
    if pdf is None:
        return "", "missing", f"paper '{paper_id}' not found in corpus"
    try:
        import fitz
    except ImportError:
        return "", "error", "PyMuPDF not installed"
    doc = None
    try:
        doc = fitz.open(pdf)
        return "\n".join(page.get_text("text") for page in doc), "pdf", "pdf"
    except Exception as e:  # noqa: BLE001
        return "", "error", f"{type(e).__name__}: {e}"
    finally:
        if doc is not None:
            doc.close()


def _read_tavily_key() -> str | None:
    """Resolve the Tavily API key.

    Checks (in order):
      1. TAVILY_API_KEY environment variable
      2. <repo_root>/api key/tavily.env (gitignored .env-style file)

    The path used to be ``<repo_root>/mast/api key/tavily.env`` — a v1-era
    location under the since-archived ``mast/`` package, so the lookup could
    never succeed regardless of whether a key existed. Both the bare-key and
    ``KEY=value`` file shapes are accepted (the other provider key files in
    ``api key/`` store the bare key on the first line).
    """
    env = os.environ.get("TAVILY_API_KEY", "").strip()
    if env:
        return env
    candidate = _repo_root() / "api key" / "tavily.env"
    if not candidate.is_file():
        return None
    try:
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                if k.strip() == "TAVILY_API_KEY":
                    return v.strip()
                continue
            return line  # bare key on its own line
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Failed to read tavily.env: %s", e)
    return None


def _truncate(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + "...[truncated]"


# ─────────────────────────────────────────────────────────────────────
# search_papers — full-text scan over PDF corpus
# ─────────────────────────────────────────────────────────────────────

@tool("search_papers")
def search_papers(query: str, max_results: int = 5) -> str:
    """Search the local paper corpus (PyMuPDF scan).

    Args:
        query:       Free-text. Multiple words form an AND query (case-insensitive):
                     a paper is a hit only if EVERY term appears somewhere in it.
        max_results: Maximum number of hits (default 5).

    Returns a numbered list of matching papers with paper_id (file stem),
    a short matched-snippet, and the page number of the first match. If the
    corpus directory is missing or empty, returns a note pointing to the
    expected location and the MAST_PAPER_CORPUS env variable.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "search_papers: PyMuPDF not installed."

    pdfs = _all_pdfs()
    if not pdfs:
        dirs = ", ".join(str(d) for d in _corpus_dirs())
        return (
            f"search_papers: no PDFs found in [{dirs}]. "
            "Drop PDFs there or set MAST_PAPER_CORPUS to a colon-separated list "
            "of directories to enable corpus search."
        )

    q_terms = [t.lower() for t in re.split(r"\s+", query.strip()) if t]
    if not q_terms:
        return "search_papers: empty query."

    hits: list[tuple[str, int, str, int]] = []  # (paper_id, score, snippet, page)
    for pdf_path in pdfs:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.debug("Failed to open %s: %s", pdf_path, e)
            continue
        try:
            score = 0
            first_page = -1
            first_snippet = ""
            # AND semantics: a paper is a hit only if EVERY query term appears
            # somewhere in the document. We accumulate per-term doc-wide counts
            # and require all terms to be present.
            term_total: dict[str, int] = {t: 0 for t in q_terms}
            for i, page in enumerate(doc):
                text = page.get_text("text").lower()
                page_score = sum(text.count(t) for t in q_terms)
                for t in q_terms:
                    term_total[t] += text.count(t)
                if page_score > 0 and first_page < 0:
                    first_page = i + 1
                    # crude snippet: 80 chars around first match
                    idx = min((text.find(t) for t in q_terms if t in text), default=-1)
                    if idx >= 0:
                        start = max(0, idx - 40)
                        end = min(len(text), idx + 80)
                        first_snippet = text[start:end].replace("\n", " ").strip()
                score += page_score
            # AND: require every term to occur at least once across the doc.
            if all(term_total[t] > 0 for t in q_terms):
                hits.append((_paper_id(pdf_path), score, first_snippet, first_page))
        finally:
            doc.close()

    if not hits:
        return f"search_papers: no matches for '{query}' across {len(pdfs)} papers."

    hits.sort(key=lambda h: -h[1])
    hits = hits[:max_results]

    lines = [f"search_papers: top {len(hits)} of {len(pdfs)} papers matching '{query}':"]
    for n, (pid, score, snippet, page) in enumerate(hits, 1):
        # Flag hand-dropped supplements so the agent does not mistake a hit in
        # "Foo_SI" for a hit in a separate paper.
        si = "  (补充材料 SI)" if _LEGACY_SI_RX.search(pid) else ""
        lines.append(f"  ({n}) {pid}{si}  [score={score}, first match p.{page}]")
        if snippet:
            lines.append(f"        ...{snippet}...")
    return _truncate("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# read_paper_section — page-range extraction with heading detection
# ─────────────────────────────────────────────────────────────────────

_SECTION_PATTERNS: dict[str, list[str]] = {
    "abstract":     [r"^\s*abstract\b", r"^\s*summary\b"],
    "introduction": [r"^\s*1\.?\s*introduction\b", r"^\s*introduction\b"],
    "methods":      [r"^\s*\d?\.?\s*(methods|materials and methods|experimental(\s+section)?)\b"],
    "results":      [r"^\s*\d?\.?\s*results(\s+and\s+discussion)?\b"],
    "discussion":   [r"^\s*\d?\.?\s*discussion\b"],
    "conclusion":   [r"^\s*\d?\.?\s*(conclusions?|concluding\s+remarks)\b"],
    "references":   [r"^\s*references\b", r"^\s*bibliography\b"],
}


@tool("read_paper_section")
def read_paper_section(paper_id: str, section: str) -> str:
    """Read one named section of a paper from the corpus.

    Args:
        paper_id: Paper identifier as returned by search_papers (file stem).
        section:  One of "abstract", "introduction", "methods", "results",
                  "discussion", "conclusion", "references". Case-insensitive.

    Returns the matched section's plain text (truncated to ~2000 chars).
    Falls back to "section X not found" if the heading regexes do not match
    any line — in that case the operator should use search_papers to locate
    the paper and read manually.
    """
    sec_key = section.strip().lower()
    pats = _SECTION_PATTERNS.get(sec_key)
    if pats is None:
        return (
            f"read_paper_section: unknown section '{section}'. Use one of: "
            + ", ".join(_SECTION_PATTERNS.keys())
        )

    text, kind, detail = _paper_text(paper_id)
    if not text:
        # "failed:" is reserved for machinery faults — extract_protocol and the
        # agent both key off the distinction ().
        if kind == "error":
            return f"read_paper_section failed: {detail}"
        return f"read_paper_section: {detail}."
    lines = text.split("\n")

    # Find first matching heading
    start = -1
    compiled = [re.compile(p, re.IGNORECASE) for p in pats]
    for i, line in enumerate(lines):
        if any(rx.search(line) for rx in compiled):
            start = i
            break
    if start < 0:
        return (
            f"read_paper_section: section '{sec_key}' heading not detected in "
            f"'{paper_id}'. Try search_papers to locate keywords directly."
        )

    # Find end: next any-section heading
    other_pats = [
        re.compile(p, re.IGNORECASE)
        for k, plist in _SECTION_PATTERNS.items() if k != sec_key
        for p in plist
    ]
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if any(rx.search(lines[j]) for rx in other_pats):
            end = j
            break

    body = "\n".join(lines[start:end]).strip()
    return _truncate(f"## {sec_key.title()} of {paper_id}\n\n{body}")


# ─────────────────────────────────────────────────────────────────────
# extract_protocol — regex pull of bias / setpoint / scan-rate from methods
# ─────────────────────────────────────────────────────────────────────

# Each pattern tries a few common phrasings for STM scan parameters.
# Numbers are matched flexibly (decimal + optional sign + optional unit).
_PROTOCOL_PATTERNS: dict[str, list[str]] = {
    "bias":      [r"bias[^.\n]*?(-?\d+(?:\.\d+)?)\s*(mV|V)\b"],
    "setpoint":  [r"set[- ]?point[^.\n]*?(\d+(?:\.\d+)?)\s*(pA|nA|µA|uA)\b",
                  r"current[^.\n]*?(\d+(?:\.\d+)?)\s*(pA|nA|µA|uA)\b"],
    "scan_rate": [r"scan(?:\s+rate)?[^.\n]*?(\d+(?:\.\d+)?)\s*(Hz|s/line|line/s)\b"],
    "temperature": [r"(\d+(?:\.\d+)?)\s*(K|kelvin|°C)\b"],
    "anneal":    [r"anneal[^.\n]*?(\d+)\s*°?C\b[^.\n]*?(\d+)\s*(min|h|hours?)\b"],
}


@tool("extract_protocol")
def extract_protocol(paper_id: str) -> str:
    """Extract experimental protocol fields (bias, setpoint, etc.) from a paper.

    Args:
        paper_id: Paper identifier as returned by search_papers (file stem).

    Pulls numeric values from the methods section using regular expressions.
    This is heuristic — for high-precision extraction the operator should
    read the paper. If a field is missing it is reported as such.
    """
    methods = read_paper_section.invoke({"paper_id": paper_id, "section": "methods"})
    # A genuine read fault must NOT be treated as parseable methods text — that
    # would swallow the error and report a misleading "no parameters found".
    if methods.startswith("read_paper_section failed:"):
        return f"extract_protocol: could not read '{paper_id}' — {methods}"
    if methods.startswith("read_paper_section:"):
        # The methods heading was simply not detected → scan the whole paper.
        whole, kind, detail = _paper_text(paper_id)
        if kind == "error":
            return f"extract_protocol failed: {detail}"
        if not whole.strip():
            return methods
        methods = whole

    found: dict[str, str] = {}
    for field, pats in _PROTOCOL_PATTERNS.items():
        for p in pats:
            m = re.search(p, methods, re.IGNORECASE)
            if m:
                groups = [g for g in m.groups() if g]
                found[field] = " ".join(groups)
                break

    if not found:
        return (
            f"extract_protocol: no quantitative parameters found in '{paper_id}' "
            "methods. Try read_paper_section directly."
        )

    lines = [f"Protocol extracted from '{paper_id}':"]
    for k, v in found.items():
        lines.append(f"  {k}: {v}")
    missing = [k for k in _PROTOCOL_PATTERNS if k not in found]
    if missing:
        lines.append(f"  (not detected: {', '.join(missing)})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# web_search — Tavily-backed external lookup
# ─────────────────────────────────────────────────────────────────────

@tool("web_search")
def web_search(query: str, max_results: int = 5) -> str:
    """Search the external web for recent papers and resources via Tavily.

    Args:
        query:       Free text (title, keyword, author + year, DOI, etc.).
        max_results: Maximum results returned. Default 5.

    Returns a numbered list of (title, URL, snippet). The Tavily API key is
    read from TAVILY_API_KEY env or ``mast/api key/tavily.env``. If neither
    is present the tool returns a note explaining how to provision the key.
    """
    key = _read_tavily_key()
    if not key:
        return (
            "web_search: TAVILY_API_KEY not configured. Either set the env "
            "variable or create mast/api key/tavily.env with the line: "
            "TAVILY_API_KEY=<your-key>."
        )
    try:
        from tavily import TavilyClient
    except ImportError:
        return "web_search: tavily-python not installed."

    try:
        client = TavilyClient(api_key=key)
        resp = client.search(
            query=query,
            max_results=max(1, min(max_results, 10)),
            search_depth="basic",
        )
    except Exception as e:
        return f"web_search failed: {type(e).__name__}: {e}"

    results = resp.get("results", []) if isinstance(resp, dict) else []
    if not results:
        return f"web_search: no Tavily results for '{query}'."

    lines = [f"web_search Tavily results for '{query}':"]
    for n, r in enumerate(results[:max_results], 1):
        title = r.get("title", "(untitled)")
        url = r.get("url", "")
        snippet = (r.get("content", "") or "")[:200].replace("\n", " ").strip()
        lines.append(f"  ({n}) {title}\n        {url}\n        {snippet}")
    return _truncate("\n".join(lines), n=2500)


# ─────────────────────────────────────────────────────────────────────
# search_local_corpus — semantic search over the OpenAlex STM index (~50k papers)
# ─────────────────────────────────────────────────────────────────────

@tool("search_local_corpus")
def search_local_corpus(
    query: str,
    max_results: int = 10,
    year_min: int = 0,
    year_max: int = 9999,
) -> str:
    """Semantic search over the local OpenAlex STM corpus (~50k papers).

    Args:
        query:        Free-text — English or Chinese (cross-lingual embedding).
        max_results:  Top-k results (default 10).
        year_min:     Optional inclusive lower year filter.
        year_max:     Optional inclusive upper year filter.

    Returns a numbered list of (title, year, journal, DOI, similarity).
    Use this BEFORE Tavily web_search — the local corpus is curated, dedup'd,
    DOI-resolved, and offline. Fall back to web_search only when the local
    corpus has no relevant hits or you need post-2024 work.
    """
    try:
        from mast.knowledge.literature_index import search_with_status
    except ImportError as e:
        return f"search_local_corpus: literature index module not available ({e})."

    try:
        from mast.knowledge.literature_index import corpus_size
    except ImportError:
        corpus_size = lambda: 0  # noqa: E731

    try:
        hits, status = search_with_status(
            query, k=max_results, year_min=year_min, year_max=year_max)
    except FileNotFoundError as e:
        return (
            "search_local_corpus: literature index missing — build it via "
            "scripts/openalex_pipeline/03_build_embedding_index.py. "
            f"Detail: {e}"
        )
    except Exception as e:
        return f"search_local_corpus failed: {type(e).__name__}: {e}"

    if not hits:
        return f"search_local_corpus: no matches for '{query}' in the local corpus."

    # ── honesty gate () ──────────────────────
    # The semantic embedder was down for a whole session; every search silently
    # degraded to keyword matching. The rows still LOOKED like search results —
    # title, year, DOI, citation count — so this agent read a slice of 50k
    # papers tied at score 3.000 and reported "第一批结果较相关". Nothing in the
    # tool output had told it otherwise. Two rules now:
    #   1. a degraded result set always says so, in the FIRST line;
    #   2. a ranking that carries no information is not dressed up as one —
    #      we withhold the list and say retrieval is unavailable.
    if status.get("degraded"):
        why = _degradation_note(status)
        if not status.get("trustworthy"):
            return (
                "search_local_corpus 检索不可用：语义检索已降级为关键词匹配，"
                f"而本次关键词匹配的结果没有排序意义。{why}\n"
                "【不要把下面这类结果当成相关文献 —— 本次没有可用结果。】\n"
                "可行的替代：(a) 换成英文关键词重新检索（本语料标题/摘要是英文）；"
                "(b) 用 web_search 走联网检索；"
                "(c) 请用户配置 DASHSCOPE_API_KEY 以恢复语义检索。"
            )
        header_prefix = f"[降级:关键词匹配,非语义相关度] {why} "
    else:
        header_prefix = ""

    _total = corpus_size()
    _total_str = f"{_total:,}" if _total else "the"
    lines = [
        f"{header_prefix}search_local_corpus: top {len(hits)} of {_total_str} "
        f"OpenAlex STM papers for '{query}':"
    ]
    for n, h in enumerate(hits, 1):
        doi = h["doi"] or "(no DOI)"
        title = h["title"][:120]
        lines.append(
            f"  ({n}) [{h['score']:.3f}] {h['year']} {title}"
        )
        # Always expose work_id: fetch_paper_abstract requires the index id.
        # A DOI alone is not a substitute for that identifier.
        lines.append(
            f"        work_id={h.get('work_id', '?')}  cited={h['cited']}  "
            f"journal={h['journal'][:50]}  doi={doi}"
        )
    return _truncate("\n".join(lines), n=3000)


def _degradation_note(status: dict) -> str:
    """One-line, checkable account of WHY a degraded result set is untrustworthy.

    Says only what the diagnostics actually establish (which terms matched, how
    many rows tied) — never guesses at the operator's intent.
    """
    bits: list[str] = []
    reason = (status.get("reason") or "").strip()
    if reason:
        bits.append(f"降级原因: {reason}")
    unmatched = list(status.get("unmatched_terms") or ())
    matched = list(status.get("matched_terms") or ())
    if unmatched:
        bits.append(
            f"查询词 {unmatched} 在语料中完全不出现（只有 {matched or '零个词'} 命中）"
        )
    if status.get("degenerate_ranking"):
        tied = status.get("tied_corpus_rows") or 0
        bits.append(
            f"返回结果得分完全相同，语料中有 {tied} 条并列 —— 排序等同于任意取样"
        )
    return "；".join(bits) + ("。" if bits else "")


# ─────────────────────────────────────────────────────────────────────
# Library curation tools (P1) — registry.json CRUD + member-filtered search.
#
# Design (owner G): the big OpenAlex index is the ONE true library. Every other
# library is a SET OF work_id pointers into it — no per-library content store.
# lib_search = big-index search filtered to the library's member work_ids.
# All of these are READ / curation tools: they touch ONLY the JSON registry and
# the local parquet index, never the instrument, so they do NOT go through the
# SafetyGate / HITL instrument-safety path (design G8).
# ─────────────────────────────────────────────────────────────────────


def _effective_library() -> tuple[str, str]:
    """``(library_id, source)`` for the library an omitted ``library_id`` hits.

    Resolved fresh on every call — it follows the active experiment with no
    subscription anywhere. Returns ``("", "")`` if the backend is unavailable, so
    callers degrade to a plain message rather than naming a target they are not
    sure of.
    """
    try:
        from mast.knowledge.experiment_library import resolve_effective_library
        return resolve_effective_library()
    except Exception:  # pragma: no cover — defensive
        return "", ""


_SOURCE_ZH = {
    "experiment": "当前实验的专属库",
    "manual": "手动指针（当前没有活跃实验）",
    "fallback": "reading 兜底库（手动指针不可用）",
}


def _effective_note() -> str:
    """One line naming the default target and why it is the default."""
    lib_id, source = _effective_library()
    if not lib_id:
        return "（有效库解析不可用）"
    return f"有效库 = {lib_id}（{_SOURCE_ZH.get(source, source)}）"


@tool("lib_list")
def lib_list() -> str:
    """List all literature libraries and say which one is the EFFECTIVE library.

    A library is a named set of ``work_id`` pointers into the big OpenAlex index
    (the one true library). The undeletable global "reading library" always
    exists, and every experiment has its own library.

    The **effective library** is the one lib_add / lib_search target when you omit
    ``library_id``. It is NOT simply the flagged active one: with an experiment
    active it is that experiment's own library, and the ``*ACTIVE*`` flag is only
    the manual pointer that gets used when no experiment is running.
    """
    from mast.knowledge import libraries as libs

    try:
        rows = libs.list_libraries()
    except Exception as e:  # pragma: no cover — defensive
        return f"lib_list failed: {type(e).__name__}: {e}"
    if not rows:
        return "lib_list: no libraries (registry empty)."
    eff_id, eff_source = _effective_library()
    lines = [f"lib_list: {len(rows)} libraries（{_effective_note()}）:"]
    for n, r in enumerate(rows, 1):
        flags = ""
        if r.get("library_id") == eff_id:
            flags += " ←有效库(默认落点)"
        if r.get("is_active"):
            flags += " [手动指针]"
        exp = f", experiment={r.get('experiment_id')}" if r.get("experiment_id") else ""
        lines.append(
            f"  ({n}) {r.get('library_id')}  \"{r.get('name')}\"  "
            f"[scope={r.get('scope')}, members={r.get('member_count')}{exp}]{flags}"
        )
    if eff_source == "experiment":
        lines.append(
            "  说明：当前实验绑定它自己的专属库，书目存在实验文件夹里（复制实验"
            "文件夹就带走书目）。要检索别的库用 lib_search(library_id=…)；要把"
            "别的库的内容引进当前实验用 lib_copy(src_library_id)。"
        )
    return "\n".join(lines)


@tool("lib_create")
def lib_create(name: str, scope: str = "custom") -> str:
    """Create a new literature library (a named set of work_id pointers).

    Args:
        name:  Display name (non-empty, <=200 chars).
        scope: "custom" (default) or "experiment". "global" is reserved — the
               single global reading library is auto-created and cannot be minted.

    Returns the new library_id (a stable slug). The library starts empty; add
    papers with lib_add. The big OpenAlex index is unchanged — a library is just
    a pointer set into it.
    """
    from mast.knowledge import libraries as libs

    try:
        rec = libs.create_library(name, scope)
    except libs.LibraryError as e:
        return f"lib_create rejected: {e}"
    except Exception as e:  # pragma: no cover — defensive
        return f"lib_create failed: {type(e).__name__}: {e}"
    return (
        f"lib_create: created library '{rec['library_id']}' "
        f"(name=\"{rec['name']}\", scope={rec['scope']}). It is empty — use "
        f"lib_add(work_ids, library_id='{rec['library_id']}') to populate it."
    )


@tool("lib_switch")
def lib_switch(library_id: str) -> str:
    """Set the MANUAL library pointer — only used while no experiment is active.

    Args:
        library_id: An existing library_id (see lib_list).

    This does NOT redirect the default target while an experiment is running. Each
    experiment is bound to its own library and that binding is not switchable: it
    is what makes "copy the experiment folder, get its bibliography" true. When an
    experiment is active this call still records your choice (it takes effect
    later, once no experiment is active) and tells you what to use instead:

      * to SEARCH another library → lib_search(query, library_id="…")
      * to BRING another library's papers into this experiment → lib_copy("…")
    """
    from mast.knowledge import libraries as libs

    try:
        rec = libs.set_active_library(library_id)
    except libs.LibraryError as e:
        return f"lib_switch rejected: {e}"
    except Exception as e:  # pragma: no cover — defensive
        return f"lib_switch failed: {type(e).__name__}: {e}"
    eff_id, eff_source = _effective_library()
    msg = (f"lib_switch: 手动指针已设为 '{rec['library_id']}' (\"{rec['name']}\")。")
    if eff_source == "experiment":
        msg += (
            f"\n  注意：当前有活跃实验，默认落点仍然是它的专属库 {eff_id} —— "
            f"你刚设的指针只在**没有活跃实验**时才起作用，现在不生效。"
            f"\n  想在 '{rec['library_id']}' 里检索：lib_search(query, "
            f"library_id='{rec['library_id']}')。"
            f"\n  想把 '{rec['library_id']}' 的文献引进当前实验："
            f"lib_copy('{rec['library_id']}')。"
        )
    return msg


@tool("lib_copy")
def lib_copy(src_library_id: str, to_experiment_id: str = "") -> str:
    """Copy another library's papers into an experiment's own library.

    Args:
        src_library_id:   The library to copy FROM (see lib_list).
        to_experiment_id: Target experiment. Omit for the current experiment.

    Libraries are never shared between experiments — you take a copy. Growth in
    library count is fine and expected. The big OpenAlex index is untouched: both
    libraries end up pointing at the same papers (and the same stored full text).

    Papers already in the target are left exactly as they are, including their
    existing reason notes. If the target hits the 500-member cap the leftovers are
    reported, never silently dropped.
    """
    try:
        from mast.knowledge import experiment_library as expl
    except Exception as e:  # pragma: no cover — defensive
        return f"lib_copy failed: {type(e).__name__}: {e}"
    try:
        res = expl.copy_library((src_library_id or "").strip(),
                                (to_experiment_id or "").strip())
    except Exception as e:  # pragma: no cover — library fns don't raise, but @tool must not either
        return f"lib_copy failed: {type(e).__name__}: {e}"
    if not res.get("ok"):
        return f"lib_copy rejected: {res.get('error') or '未知原因'}"
    parts = [
        f"lib_copy: 从 '{res['src_library_id']}' 复制了 {len(res.get('copied') or [])} 篇"
        f"到 '{res['library_id']}'（现有 {res.get('member_count', 0)} 篇）。"
    ]
    if res.get("skipped"):
        parts.append(f"  {len(res['skipped'])} 篇目标库里已有，原样保留（未覆盖它们的 reason）。")
    if res.get("rejected"):
        parts.append(f"  {len(res['rejected'])} 篇没进去：{', '.join(res['rejected'][:10])}")
    if res.get("at_cap"):
        parts.append("  注意：目标库已到 500 成员上限，超出的没有加入。")
    if res.get("note"):
        parts.append(f"  {res['note']}")
    return "\n".join(parts)


@tool("lib_add")
def lib_add(work_ids: list[str], library_id: str | None = None, reason: str = "") -> str:
    """Add paper pointers (work_ids) to a library.

    Args:
        work_ids:   One or more OpenAlex work_ids (e.g. "W2041234567") or
                    "local:<hash>" ids, as returned by lib_search /
                    search_local_corpus.
        library_id: Target library. **Omit it** to use the effective library —
                    with an experiment active that is THAT EXPERIMENT'S OWN
                    library (not a global "current library"); with no experiment
                    it is the manual pointer. The reply always names where the
                    papers actually landed.
        reason:     Short note on WHY this paper matters — it is kept with the
                    pointer and is what makes the bibliography readable months
                    later. Worth writing.

    The big OpenAlex index is NOT modified — only the library's pointer set
    grows. Dedup is automatic (the bare "W123" and URL forms are the same paper);
    the set is bounded (max 500 members). Returns a summary of added /
    already-present / rejected ids.
    """
    from mast.knowledge import libraries as libs

    if isinstance(work_ids, str):
        work_ids = [work_ids]
    try:
        res = libs.add_members(work_ids, library_id, reason=reason, added_by="agent")
    except libs.LibraryError as e:
        return f"lib_add rejected: {e}"
    except Exception as e:  # pragma: no cover — defensive
        return f"lib_add failed: {type(e).__name__}: {e}"
    where = "" if library_id else f"（{_effective_note()}）"
    parts = [
        f"lib_add → '{res['library_id']}'{where}: added {len(res['added'])}, "
        f"already-present {len(res['skipped'])}, rejected {len(res['rejected'])} "
        f"(library now holds {res['member_count']} papers)."
    ]
    if res["added"]:
        parts.append(f"  added: {', '.join(res['added'])}")
    if res["rejected"]:
        parts.append(f"  rejected (invalid id or at cap): {', '.join(res['rejected'])}")
    if res.get("at_cap"):
        parts.append("  note: library is at the 500-member cap.")
    if res.get("folder") is False and not res.get("error"):
        parts.append("  note: 该实验的文件夹解析不到，指针只记在 registry 里"
                     "（内容没丢；实验行恢复后会补齐文件夹书目）。")
    if res.get("error"):
        parts.append(f"  note: {res['error']}")
    return "\n".join(parts)


@tool("lib_remove")
def lib_remove(work_ids: list[str], library_id: str | None = None) -> str:
    """Remove paper pointers (work_ids) from a library.

    Args:
        work_ids:   work_ids to drop.
        library_id: Target library. Omit to use the effective library (the current
                    experiment's own library, or the manual pointer if none).

    Only the library's pointer set shrinks; the big OpenAlex index and the paper
    itself are untouched. In an experiment library the removal is recorded as an
    event rather than an erasure, so the history of what was once considered
    stays readable. Returns a summary.
    """
    from mast.knowledge import libraries as libs

    if isinstance(work_ids, str):
        work_ids = [work_ids]
    try:
        res = libs.remove_members(work_ids, library_id)
    except libs.LibraryError as e:
        return f"lib_remove rejected: {e}"
    except Exception as e:  # pragma: no cover — defensive
        return f"lib_remove failed: {type(e).__name__}: {e}"
    removed = res.get("removed", [])
    where = "" if library_id else f"（{_effective_note()}）"
    msg = (
        f"lib_remove → '{res['library_id']}'{where}: removed {res.get('n_removed', 0)} "
        f"(library now holds {res['member_count']} papers)."
    )
    if removed:
        msg += f"\n  removed: {', '.join(removed)}"
    if res.get("error"):
        msg += f"\n  note: {res['error']}"
    return msg


@tool("lib_search")
def lib_search(query: str, library_id: str | None = None, k: int = 8) -> str:
    """Semantic search, optionally filtered to one library's members.

    Args:
        query:      Free-text (English or Chinese; multilingual embedding).
        library_id: Restrict results to that library's member work_ids — the big
                    OpenAlex index is searched and then FILTERED to the library's
                    pointer set (no secondary index is built; the big index is the
                    only real library). Pass ``"*"`` to search the WHOLE corpus.
                    Omit it and you search the effective library — the current
                    experiment's own library, i.e. only what has been curated for
                    this experiment. **To find new papers, use
                    search_local_corpus (or lib_search with "*").**
        k:          Number of results (default 8).

    Returns a numbered list of title / year / doi / abstract_excerpt / source.
    """
    from mast.knowledge.literature_index import canonical_work_id, search_with_status

    members: set[str] | None = None
    lib_label = "the whole corpus"
    # "*" is an explicit "whole corpus"; omitted means "what this experiment has
    # curated". Defaulting an omitted argument to the whole 50k corpus would make
    # lib_search and search_local_corpus the same tool.
    explicit = bool(library_id) and library_id != "*"
    if library_id == "*":
        library_id = None
    elif not library_id:
        eff_id, _src = _effective_library()
        library_id = eff_id or None
    if library_id:
        from mast.knowledge import libraries as libs

        empty_hint = (
            f"lib_search: 库 '{library_id}' 里还没有文献 —— 没有可检索的内容。"
            f"要搜全部 50k 大库用 search_local_corpus（或 "
            f"lib_search(query, library_id='*')），然后把值得留下的用 lib_add 收进来。"
        )
        try:
            rec = libs.get_library(library_id)
        except libs.LibraryError as e:
            # An experiment's own library is created lazily, so "not in the
            # registry yet" and "empty" are the SAME situation to the caller — the
            # id is derived from the experiment, it is theirs either way. Only an
            # id the caller typed can be genuinely wrong.
            if explicit:
                return f"lib_search rejected: {e}"
            return empty_hint
        except Exception as e:  # pragma: no cover — defensive
            return f"lib_search failed: {type(e).__name__}: {e}"
        # Normalise member ids to the canonical bare form so the filter matches
        # regardless of whether members were added bare ("W…") or as the full
        # URL — the hits carry the corpus's URL form.
        members = {canonical_work_id(m["work_id"]) for m in rec.get("members", [])}
        lib_label = f"library '{library_id}' (\"{rec.get('name', '')}\")"
        if not members:
            return empty_hint

    if not query or not query.strip():
        return "lib_search: empty query."

    # Over-fetch when filtering so the post-filter can still fill k.
    fetch_k = k if members is None else max(k * 8, k + 40)
    try:
        hits, status = search_with_status(query, k=fetch_k)
    except FileNotFoundError as e:
        return (
            "lib_search: literature index missing — build it via "
            f"scripts/openalex_pipeline/03_build_embedding_index.py. Detail: {e}"
        )
    except Exception as e:  # pragma: no cover — defensive
        return f"lib_search failed: {type(e).__name__}: {e}"

    # Same honesty gate as search_local_corpus (): never hand back an
    # information-free ranking that reads like a relevance ranking.
    if status.get("degraded") and not status.get("trustworthy"):
        return (
            f"lib_search 检索不可用：语义检索已降级为关键词匹配，本次结果没有排序意义。"
            f"{_degradation_note(status)}\n"
            "【本次没有可用结果 —— 不要当成相关文献。】改用英文关键词重查，"
            "或请用户配置 DASHSCOPE_API_KEY 恢复语义检索。"
        )

    if members is not None:
        hits = [h for h in hits if canonical_work_id(h.get("work_id")) in members]
    hits = hits[:k]

    if not hits:
        return f"lib_search: no matches for '{query}' in {lib_label}."

    _prefix = ("[降级:关键词匹配,非语义相关度] " if status.get("degraded") else "")
    lines = [f"{_prefix}lib_search: top {len(hits)} hits for '{query}' in {lib_label}:"]
    for n, h in enumerate(hits, 1):
        doi = h.get("doi") or "(no DOI)"
        title = (h.get("title") or "")[:120]
        excerpt = (h.get("abstract_excerpt") or "").replace("\n", " ").strip()
        lines.append(
            f"  ({n}) {h.get('work_id')}  {h.get('year')}  {title}"
        )
        lines.append(
            f"        doi={doi}  source={h.get('source', 'openalex')}"
        )
        if excerpt:
            lines.append(f"        abstract: {excerpt[:200]}")
    return _truncate("\n".join(lines), n=3000)


@tool("fetch_paper_abstract")
def fetch_paper_abstract(work_id: str) -> str:
    """Fetch the full abstract + bibliographic context for one paper by work_id.

    Args:
        work_id: An OpenAlex work_id in EITHER form — the bare id ("W2041234567")
                 or the full URL ("https://openalex.org/W2041234567") exactly as
                 lib_search / search_local_corpus print it — or a "local:<hash>"
                 id. A **DOI** is also accepted (bare "10.1103/…", a doi.org URL,
                 or "doi:…") and resolved to its work_id. Prefer the work_id:
                 search_local_corpus prints ``work_id=`` on every hit.
                 Exact lookup; works even for papers absent from the vector index.

    Returns the abstract, authors, provenance and (when available) classification.
    When the abstracts enrichment table has no row for the id, it falls back to
    the index-aligned metadata abstract (the SAME text lib_search shows) and
    marks the result as such. Read-only; never raises on a missing id.
    """
    from mast.knowledge.literature_index import fetch_abstract, work_id_for_doi

    try:
        rec = fetch_abstract(work_id)
        # A DOI is a legitimate thing to be holding — it is the id a paper
        # carries everywhere outside this system, and the search listing used to
        # print ONLY the DOI. Resolve it instead of dead-ending (2026-07-27:
        # five consecutive failed calls, all of them passing a doi.org URL).
        resolved = ""
        if not rec.get("found"):
            resolved = work_id_for_doi(work_id)
            if resolved:
                rec = fetch_abstract(resolved)
    except Exception as e:  # pragma: no cover — defensive
        return f"fetch_paper_abstract failed: {type(e).__name__}: {e}"
    if not rec.get("found"):
        note = rec.get("note")
        if note:
            return f"fetch_paper_abstract: '{work_id}' not available ({note})."
        return (
            f"fetch_paper_abstract: '{work_id}' not found in the literature index "
            "(tried the bare 'W…' id, the full OpenAlex URL form, and — if it "
            "looked like a DOI — a DOI→work_id lookup). "
            "search_local_corpus prints work_id= on every hit; pass that."
        )

    lines = [f"Abstract for {rec.get('work_id')}:"]
    if rec.get("first_author"):
        lines.append(f"  first author: {rec['first_author']}")
    if rec.get("authors"):
        lines.append(f"  authors: {rec['authors']}")
    for opt in ("year", "journal"):
        if rec.get(opt):
            lines.append(f"  {opt}: {rec[opt]}")
    lines.append(f"  source: {rec.get('source', 'openalex')}  "
                 f"provenance: {rec.get('abstract_provenance', '')}")
    if rec.get("abstract_source") == "index_metadata":
        lines.append(
            "  [注] 摘要来自索引对齐的 metadata 摘要（与 lib_search 同源）；"
            "abstracts 富集表无此条目，故缺作者/概念等字段。"
        )
    if rec.get("primary_category") or rec.get("material"):
        lines.append(
            f"  classified: category={rec.get('primary_category', '')} "
            f"material={rec.get('material', '')}"
        )
    if rec.get("abstract"):
        lines.append(f"\n  {rec['abstract']}")
    if rec.get("user_abstract"):
        lines.append(f"\n  [user-contributed] {rec['user_abstract']}")
    return _truncate("\n".join(lines), n=3000)


@tool("propose_citations")
def propose_citations(section_text: str, k: int = 8) -> str:
    """Propose literature citations for a draft passage (read-only recall).

    Args:
        section_text: A draft paragraph / claim to find supporting citations for.
        k:            Number of candidate citations (default 8).

    Returns a numbered list of candidate papers with a BibTeX-style key, title,
    year, DOI and similarity. This is a READ-ONLY recall tool over the big index
    — it does not modify any library.
    """
    from mast.knowledge.citations import propose_citations as _propose

    if not section_text or not section_text.strip():
        return "propose_citations: empty section text."
    try:
        cands = _propose(section_text, k=k)
    except FileNotFoundError as e:
        return (
            "propose_citations: literature index missing — build it via "
            f"scripts/openalex_pipeline/03_build_embedding_index.py. Detail: {e}"
        )
    except Exception as e:  # pragma: no cover — defensive
        return f"propose_citations failed: {type(e).__name__}: {e}"
    if not cands:
        return "propose_citations: no candidate citations found."
    lines = [f"propose_citations: {len(cands)} candidates:"]
    for n, c in enumerate(cands, 1):
        doi = c.get("doi") or "(no DOI)"
        lines.append(
            f"  ({n}) [{c.get('bibtex_key')}] {c.get('year')} "
            f"{(c.get('title') or '')[:110]}"
        )
        lines.append(
            f"        work_id={c.get('work_id')}  doi={doi}  "
            f"score={c.get('score')}  cited={c.get('cited')}"
        )
    return _truncate("\n".join(lines), n=3000)


@tool("literature_priors")
def literature_priors(material: str, mode: str = "") -> str:
    """Quantitative literature parameter percentiles for a material / phase.

    Args:
        material: Material/phase key, e.g. "Au(111)".
        mode:     Measurement phase, e.g. "imaging" or "sts". Omit to use the
                  first available phase for the material.

    Returns p25/p50/p75/min/max + paper count per parameter (bias, setpoint,
    scan size, temperature, …).

    IMPORTANT: these are LITERATURE REFERENCE VALUES aggregated across papers,
    NOT authoritative defaults — they vary by instrument and sample. Treat them
    as a starting prior to be confirmed, never as a guaranteed setting.
    """
    from mast.knowledge.priors import summarize_priors

    if not material or not material.strip():
        return "literature_priors: material is required (e.g. 'Au(111)')."
    try:
        summary = summarize_priors(material, mode)
    except Exception as e:  # pragma: no cover — defensive
        return f"literature_priors failed: {type(e).__name__}: {e}"
    if not summary or not summary.get("params"):
        return (
            f"literature_priors: no aggregated priors for material='{material}'"
            + (f", mode='{mode}'" if mode else "")
            + ". (literature_priors.json may be missing or lack this material.)"
        )
    lines = [
        f"literature_priors for {summary['material']} / {summary['mode']} "
        f"(n_papers={summary.get('n_papers', 0)}) — "
        f"LITERATURE REFERENCE VALUES, NOT authoritative defaults:"
    ]
    for param, pct in summary["params"].items():
        lines.append(
            f"  {param}: p25={pct.get('p25')} p50={pct.get('p50')} "
            f"p75={pct.get('p75')} (min={pct.get('min')} max={pct.get('max')} "
            f"n={pct.get('n')})"
        )
    return "\n".join(lines)


@tool("search_fulltext")
def search_fulltext(query: str, paper_refs: list[str] | None = None,
                    k: int = 8) -> str:
    """在**已入库的全文里**按语义找段落（比精读便宜得多，比读章节准得多）。

    ingest 时每篇论文的全文已经切块并做了向量化，这个工具直接检索那些块：
    问一句话，拿回最相关的几段原文（带论文、页码）。中英文 query 都行。

    **三个读全文的工具怎么选**：
      - 知道要哪一节（methods/results…）→ read_paper_section（本地解析，最快）
      - **不知道在哪一节、只知道想找什么**（"这几篇怎么处理针尖的？"）→ **就用这个**
        （一次 embedding 往返，几十秒；和 search_local_corpus 同一量级）
      - 要成套参数表 / 逐篇对比 → deep_read_papers（几分钟 + 大量 token）

    检索粒度是**页/小节**、块与块不重叠，所以跨页断开的句子可能被切成两段 ——
    命中某段后若上下文不全，用 read_paper_section 读那一节的完整原文。

    Args:
        query:      要找什么。写成问题或关键描述都行。
        paper_refs: 限定在这几篇里找（work_id 或 paper_id）。**留空＝在本地所有
                    已入库全文里找**。
        k:          返回几段（默认 8；同一篇最多占 3 段，免得一篇刷屏）。

    检索降级（embedding 不可用）时会在第一行明说，那批结果是关键词命中、没有相关度
    排序意义 —— 不要把它当语义结果汇报。
    """
    q = (query or "").strip()
    if not q:
        return "search_fulltext: 需要 query（想找什么）。"

    slugs: list[str] | None = None
    if paper_refs:
        slugs = []
        for ref in paper_refs:
            r = str(ref or "").strip()
            if not r:
                continue
            resolved = r
            try:
                from mast.knowledge.ingest import slug_for_work_id
                resolved = slug_for_work_id(r) or r
            except Exception:  # pragma: no cover — ingest deps missing
                pass
            for cand in (resolved, r):
                if cand and cand not in slugs:
                    slugs.append(cand)

    try:
        from mast.knowledge.fulltext_search import search_chunks
        hits, status = search_chunks(q, slugs, k=max(1, int(k or 8)))
    except Exception as e:  # pragma: no cover — @tool 层永不抛
        return f"search_fulltext failed: {type(e).__name__}: {e}"

    if not hits:
        base = f"search_fulltext: 没有命中「{q}」。"
        if status.get("n_papers"):
            return (base + f"（已检索 {status['n_papers']} 篇 / "
                    f"{status['n_chunks']} 个全文块）换个说法再试一次，或用 "
                    f"search_local_corpus 找别的论文。")
        # No searchable chunks. Distinguish "we hold nothing" from "we hold PDFs
        # that were never indexed" — they need opposite next steps, and reporting
        # the second as the first sends the agent off to re-fetch papers the
        # operator already put on the machine.
        n_pdfs = 0
        try:
            n_pdfs = len(_all_pdfs())
        except Exception:  # pragma: no cover — defensive
            n_pdfs = 0
        if n_pdfs:
            return (base + f"本地有 {n_pdfs} 篇 PDF，但它们**没有建过全文检索索引**"
                    f"（只有经 ingest 入库的论文才有）。请用户在「文献库 → 摄取与取文」"
                    f"里摄取它们，或先用 read_paper_section / deep_read_papers 直接读。")
        reason = str(status.get("reason") or "")
        return (base + (reason or "本地还没有任何全文") + "—— 先用 fetch_fulltext_oa 取文，"
                "或 request_fulltext 请用户上传。")

    if status.get("degraded") and not status.get("trustworthy"):
        return (f"search_fulltext 检索不可用：{status.get('reason', '')}。"
                f"关键词也没有有效命中，本次**没有可用结果**——不要当成相关段落。"
                f"可请用户检查 DASHSCOPE_API_KEY。")

    prefix = "[降级:关键词匹配,非语义相关度] " if status.get("degraded") else ""
    lines = [f"{prefix}search_fulltext：「{q}」的 {len(hits)} 段命中"
             f"（检索了 {status.get('n_papers', 0)} 篇 / "
             f"{status.get('n_chunks', 0)} 个全文块）"]
    for n, h in enumerate(hits, 1):
        who = h.get("title") or h.get("slug", "")
        lines.append(f"  ({n}) {who}  [{h.get('slug')}  p.{int(h.get('page', 0)) + 1}"
                     f"  score={h.get('score', 0):.3f}]")
        body = " ".join(str(h.get("text", "")).split())
        lines.append(f"        {body[:600]}")
    lines.append("需要这几篇的成套参数表时用 deep_read_papers；"
                 "只要某一节的原文用 read_paper_section。")
    return _truncate("\n".join(lines), n=5000)


@tool("deep_read_papers")
def deep_read_papers(paper_refs: list[str], focus: str = "",
                     tool_call_id: Annotated[str, InjectedToolCallId] = "",
                     ) -> "ArtifactToolReturn | str":
    """**并行精读**几篇论文的全文（含 SI 补充材料），产出逐篇结构化笔记。

    每篇论文交给一个独立的精读子任务**同时**进行：完整读一遍全文 + 它的补充材料，
    产出「实验体系 / 方法流程 / 关键参数表（每个值标注出自正文还是 SI）/ 结论证据 /
    可信度」的笔记，落盘到该论文目录，并把汇总存成一份文献报告文档。

    **什么时候用**：用户明确要求「仔细读 / 精读 / 逐篇比较方法与参数」，或者需要成套
    的实验条件而摘要和单节阅读凑不齐时。
    **什么时候不要用**：随手检索、只要一个数值 —— 那用 read_paper_section /
    extract_protocol 就够了。本工具每篇要花一次完整的 LLM 精读（几十秒、真实 token），
    **一次最多 4 篇**，整批最长约五分钟；期间你不会有别的输出。

    全文不在本地的论文会被跳过并提示先 fetch_fulltext_oa / request_fulltext；
    某一篇失败或超时不影响其余几篇。

    Args:
        paper_refs: 论文标识列表（work_id、或 search_papers 返回的 paper_id）。最多 4 个。
        focus:      本次精读的关注点（如「针尖处理与制样条件」）。留空＝通用精读，
                    方法与参数优先。**写清楚它**——它决定笔记里额外回答什么。
    """
    refs = [str(r).strip() for r in (paper_refs or []) if str(r).strip()]
    if not refs:
        return "deep_read_papers: 需要至少一个 paper_refs（work_id 或 paper_id）。"
    try:
        from mast.agents.literature.deep_read import deep_read_batch
        notes = deep_read_batch(refs, focus=focus)
    except Exception as e:  # pragma: no cover — @tool 层永不抛
        return f"deep_read_papers failed: {type(e).__name__}: {e}"
    if not notes:
        return "deep_read_papers: 没有可精读的论文。"

    ok = [n for n in notes if n.status == "ok"]
    bad = [n for n in notes if n.status != "ok"]

    lines = [f"精读完成：{len(ok)}/{len(notes)} 篇成功"
             + (f"（关注点：{focus.strip()}）" if focus.strip() else "")]
    for n in ok:
        head = f"\n【{n.title or n.slug or n.ref}】"
        if n.si_files:
            head += f"（含 SI：{', '.join(n.si_files)}）"
        if n.truncated:
            head += "（全文过长，只读了前一部分）"
        lines.append(head)
        lines.append(f"  {n.brief}")
        if n.note_path:
            lines.append(f"  笔记：{n.note_path}")
    for n in bad:
        why = {"timeout": "超时", "not_found": "本地没有全文",
               "failed": "失败"}.get(n.status, n.status)
        lines.append(f"\n【{n.ref}】{why}：{n.error}")

    res = _save_deep_read_digest(notes, focus)
    if res is not None and getattr(res, "ok", False):
        lines.append(f"\n汇总已存为文档「{res.title}」v{res.version}"
                     f"（doc_id={res.doc_id}）：{res.path}")
    lines.append("\n完整笔记见上面各篇的笔记路径；需要引用具体数值时以笔记里的"
                 "「关键参数表」为准（每个值都标了出自正文还是 SI）。")
    summary = _truncate("\n".join(lines), n=6000)

    ref = doc_ref_from_save(res, kind="literature_report",
                            produced_by="literature",
                            summary="; ".join(n.brief[:120] for n in ok)[:400])
    if ref is None:
        return summary
    return ArtifactToolReturn(summary, {"literature_report": ref},
                              tool_call_id=tool_call_id,
                              name="deep_read_papers")


def _save_deep_read_digest(notes: list, focus: str):
    """Persist the batch as a document; returns the save result or None.

    A reading that lives only in the conversation is gone the moment the context
    is compacted — which is precisely when someone asks "what did that paper say
    about the setpoint again".

    Saved as ``literature_report`` rather than a new document kind on purpose:
    ``documents.model.KINDS`` is a closed set and ``normalize_kind`` silently
    rewrites anything unknown to ``experiment_report``, so an invented kind would
    file these notes under the wrong heading without ever complaining.
    """
    ok = [n for n in notes if getattr(n, "status", "") == "ok"]
    if not ok:
        return None
    body = [f"# 精读笔记汇总（{len(ok)} 篇）", ""]
    if focus.strip():
        body.append(f"**关注点**：{focus.strip()}")
        body.append("")
    for n in ok:
        body.append(f"## {n.title or n.slug or n.ref}")
        if n.si_files:
            body.append(f"*SI 附件：{', '.join(n.si_files)}*")
        if n.truncated:
            body.append("*注意：全文过长，仅精读了前一部分。*")
        body.append("")
        body.append(n.note_md)
        body.append("")
    skipped = [n for n in notes if getattr(n, "status", "") != "ok"]
    if skipped:
        body.append("## 未能精读的论文")
        for n in skipped:
            body.append(f"- {n.ref}：{n.status} — {n.error}")
    try:
        from mast.documents import store
        res = store().save(text="\n".join(body), kind="literature_report",
                           title=f"精读笔记汇总（{len(ok)} 篇）",
                           created_by="agent:literature")
    except Exception as exc:  # pragma: no cover — the notes on disk still stand
        logger.info("deep-read digest save failed: %s", exc)
        return None
    return res


@tool("fetch_fulltext_oa")
def fetch_fulltext_oa(doi_or_url: str, work_id: str = "") -> str:
    """自己走**开源渠道**取全文并入库（拿不到再用 request_fulltext 找用户）。

    需要某篇论文的全文时**先调它**：经 OpenAlex / Unpaywall 定位开放获取（OA）副本，
    遵守 robots.txt 与付费墙边界合法下载，成功后自动促进进大库。成功即可立刻用
    read_paper_section / extract_protocol 读它的章节，或 deep_read_papers 精读。

    **合法性边界**：绝不绕过付费墙或登录墙 —— 拿不到就是拿不到，它会明说。
    那时才用 request_fulltext(work_id, reason) 请用户上传，然后**继续用摘要工作**。

    含网络请求，单次最长约一分钟。**一篇最多试一次**：失败了不要换个写法重试，
    照返回话术走下一步。

    Args:
        doi_or_url: 论文的 DOI（如 10.1103/PhysRevLett.123.456）或直链。检索结果里
                    都带 DOI。必填。
        work_id:    该论文在大库里的 work_id。**强烈建议传** —— 入库时用它精确绑定
                    身份，避免生成重复条目；取文失败后发 request_fulltext 也要用它。
    """
    raw = (doi_or_url or "").strip()
    if not raw:
        return ("fetch_fulltext_oa: 需要 doi_or_url（例如 10.1103/PhysRevLett.123.456）。"
                "检索结果里带 DOI，先拿到它再调。")
    wid = (work_id or "").strip()
    hint = (f"用 request_fulltext(work_id=\"{wid}\", reason=…) 请用户获取"
            if wid else "用 request_fulltext(work_id=…, reason=…) 请用户获取")

    try:
        from mast.knowledge.fetch import try_fetch_fulltext
        res = try_fetch_fulltext(raw, auto_oa=True, timeout=15.0) or {}
    except Exception as e:  # pragma: no cover — @tool 层永不抛
        return f"fetch_fulltext_oa failed: {type(e).__name__}: {e}"

    status = str(res.get("status", "") or "")
    message = str(res.get("message", "") or "").strip()

    if status == "ok_pdf":
        pdf_path = str(res.get("pdf_path", "") or "")
        try:
            from mast.knowledge.ingest import ingest_pdf
            ir = ingest_pdf(pdf_path, work_id=wid, library_id="",
                            source="agent", promote=True)
        except Exception as e:
            # 日志保留 traceback 供排查；模型只接收简短错误消息，避免
            # 用完整堆栈占用上下文。类型与消息本身不足以定位失败调用。
            logger.exception(
                "fetch_fulltext_oa: ingest_pdf failed for %s (work_id=%r)",
                pdf_path, wid)
            return (f"已合法下载 PDF（存于 {pdf_path}），但入库失败："
                    f"{type(e).__name__}: {e}。请把该路径告知用户从「文献库 → 摄取与"
                    f"取文」入库；先继续用摘要完成手头的任务。"
                    f"（完整栈已写入服务日志，搜 `ingest_pdf failed`。）")

        ir_status = str(getattr(ir, "status", "") or "")
        if ir_status not in ("ingested", "replaced", "noop"):
            detail = str(getattr(ir, "detail", "") or ir_status)
            return (f"已合法下载 PDF（存于 {pdf_path}），但入库失败：{detail}。"
                    f"请把该路径告知用户从「文献库 → 摄取与取文」入库；"
                    f"先继续用摘要完成手头的任务。")

        got_id = str(getattr(ir, "work_id", "") or wid)
        slug = ""
        try:
            slug = Path(str(getattr(ir, "slug_dir", "") or "")).name
        except Exception:  # pragma: no cover — defensive
            slug = ""
        cur: dict = {}
        try:
            from mast.knowledge.fulfilment import curate_ingested_fulltext
            cur = curate_ingested_fulltext(
                got_id, source="agent", reason="agent OA fetch",
                slug=slug, added_by="agent") or {}
        except Exception as exc:  # pragma: no cover — curation never blocks
            logger.info("curation after agent fetch failed (%s): %s", got_id, exc)

        # ingest copied the bytes into <papers>/<work_id>/; the staged download
        # is now a second copy of the same paper under a different name.
        try:
            import shutil
            staged = Path(pdf_path).parent
            if staged.name and staged.parent.name == "_incoming":
                shutil.rmtree(staged, ignore_errors=True)
        except Exception as exc:  # pragma: no cover — leftover is harmless
            logger.debug("staging cleanup skipped for %s: %s", pdf_path, exc)

        title = str(getattr(ir, "title", "") or "").strip()
        n_chunks = int(getattr(ir, "n_chunks", 0) or 0)
        lines = [
            "已通过开源渠道获取全文并入库：",
            f"  work_id={got_id}" + (f"  «{title}»" if title else "")
            + (f"  （{n_chunks} 个全文块，已促进进大库）" if n_chunks else "（已促进进大库）"),
        ]
        if cur.get("pointer_library_id"):
            lines.append(f"  归入库：{cur['pointer_library_id']}"
                         f"（{cur.get('pointer_library_source', '')}）")
        if int(cur.get("fulfilled_requests", 0) or 0) > 0:
            lines.append(f"  同时满足并关闭了取文板上 {cur['fulfilled_requests']} 条请求。")
        lines.append(
            f"现在可以直接用 read_paper_section(\"{slug or got_id}\", section) / "
            f"extract_protocol 读它的全文章节。继续你手头的任务。")
        return "\n".join(lines)

    if status == "blocked":
        return (f"这篇在付费墙 / robots 限制内，按对方条款**不绕过**"
                f"{('：' + message) if message else ''}。"
                f"确实需要全文就{hint}，提交后**继续用摘要**完成当前任务，不要等待。")

    if status in ("ok_metadata", "unavailable"):
        return (f"开源渠道没有可合法获取的全文副本"
                f"{('：' + message) if message else ''}。"
                f"确实需要全文就{hint}，提交后**继续用摘要**完成当前任务，不要等待。")

    return (f"取文失败{('：' + message) if message else ''}。"
            f"不要反复重试 —— {hint}，然后继续用摘要完成当前任务。")


@tool("request_fulltext")
def request_fulltext(work_id: str, reason: str = "", doi: str = "",
                     title: str = "") -> str:
    """请求用户获取某篇论文的全文（HITL 取文请求板，设计 §8）。

    （通常应先试过 fetch_fulltext_oa 自取，开源渠道拿不到时再来挂板。）

    当语义检索命中一篇你需要全文（而大库只有摘要）的论文时，发一条取文请求；
    它会出现在 GUI「文献库」标签的取文请求板。用户在请求行**直接上传 PDF**
    即可满足它（论文随即入库）；DOI 取文按钮只负责下载，还要再摄取一次才算数。

    **用户满足请求后，本会话会自动继续**——你会收到一条「取文请求已满足」的消息，
    接着把当初因为缺全文而搁置的事做完即可。所以 reason 要写清楚**为什么需要这篇**：
    请求可能几天后才被满足，到时候全靠这句话说明当初想干什么。
    也可以用 list_fetch_requests 主动查看处理结果。

    请求会记下**当前是哪个实验**（创建时冻结）。这一点重要：请求可能几天后才被满足，
    那时活跃实验可能已经换了，按「当时的活跃实验」入库就会归错档。

    work_id: 论文的 work_id（或 local:<hash>）。reason: 为什么需要全文。
    doi/title: 可选，便于用户识别。"""
    from mast.knowledge.fetch_board import post_request
    if not work_id or not work_id.strip():
        return "request_fulltext: work_id is required."
    eid = ""
    try:
        from mast.documents.paths import current_scope
        eid = str(current_scope()[0] or "")
    except Exception:  # pragma: no cover — 拿不到作用域不该阻塞请求
        eid = ""
    # Which conversation is asking — so fulfilling the request can wake it back
    # up instead of leaving the answer sitting on a board nobody re-reads.
    # Empty in a background run (no turn context), which degrades to "just close
    # the board", i.e. the behaviour before auto-resume existed.
    origin = ""
    try:
        from mast.core.turn_context import current_turn
        origin = str((current_turn() or {}).get("conversation_id") or "")
    except Exception:  # pragma: no cover — provenance is best-effort
        origin = ""
    try:
        rec = post_request(work_id.strip(), reason=reason, doi=doi, title=title,
                           requested_by="literature", experiment_id=eid or None,
                           origin_conversation_id=origin)
    except Exception as e:  # pragma: no cover — defensive
        return f"request_fulltext failed: {type(e).__name__}: {e}"
    if rec.get("error"):
        return f"request_fulltext: {rec['error']}"
    scope_note = ("满足后会归进当前实验的专属库。" if rec.get("experiment_id")
                  else "当前没有活跃实验，满足后归进手动指针指向的库。")
    return (f"已提交取文请求 {rec['request_id']} (work_id={rec['work_id']}, "
            f"状态={rec['status']})。{scope_note}用户将在文献库标签处理；用 "
            f"list_fetch_requests 查看进展。")


@tool("save_literature_report")
def save_literature_report(title: str, markdown_text: str, doc_id: str = "",
                           key_findings: str = "",
                           tool_call_id: Annotated[str, InjectedToolCallId] = "",
                           ) -> "ArtifactToolReturn | str":
    """把文献综述/调研报告**存成文档**（落当前实验的 reports/，永不覆盖）。

    在这个工具存在之前，你写的综述只活在对话流里 —— 对话一压缩就没了，用户事后
    翻不出来，也没法给别人。存下来才算交付。

    存盘同时会把这份报告**登记为本次协作的上游产物**：下游智能体（实验设计等）
    会在自己的上下文里直接看到它的 doc_id 与要点，不必你在交接语里复述。

    Args:
        title:         报告标题（显示名；改标题不影响文档身份）。
        markdown_text: 报告正文（markdown）。
        doc_id:        **修订同一份报告时必须回传上次拿到的 doc_id**，这样会存成
                       同一个文档的新版本（v002、v003…）。留空 = 另立一份新文档。
                       身份看 doc_id，不看标题 —— 换个措辞不会分叉，两份同名报告
                       也不会被误合并。
        key_findings:  给下游智能体看的**要点**（1–3 句，含具体参数区间就更好）。
                       它会被放进下游的上下文；留空则下游只看到标题。
                       写你希望「设计实验的人一眼就该知道」的那几句，不要复述目录。

    版本永不覆盖：每次保存都是一个新的 vNNN.md，旧版本原样留着。
    """
    if not (markdown_text or "").strip():
        return "save_literature_report: 正文为空，没有保存。"
    try:
        from mast.documents import store
        res = store().save(text=markdown_text, kind="literature_report",
                           title=(title or "").strip() or "文献综述",
                           doc_id=(doc_id or "").strip(),
                           created_by="agent:literature")
    except Exception as e:  # pragma: no cover — @tool 层永不抛
        return f"save_literature_report failed: {type(e).__name__}: {e}"
    if not res.ok:
        return f"save_literature_report 未保存：{res.error}"
    lines = [
        f"已保存文献报告「{res.title}」 v{res.version}（doc_id={res.doc_id}）",
        f"  路径：{res.path}",
        f"  **下次修订这份报告时把 doc_id={res.doc_id} 传回来**，否则会另立一份新文档。",
    ]
    if res.doc_id_unknown:
        lines.append(
            "  注意：你传的 doc_id 找不到对应文档，所以内容存成了一份**新文档**"
            "（没有丢），上面这个 doc_id 才是它的身份。")
    if res.root_kind == "unfiled":
        lines.append(
            "  注意：当前没有活跃实验，报告存在 _unfiled/ 里，还没有归属实验。"
            "建议先选择或新建实验，之后可以把它认领过去。")
    summary = "\n".join(lines)

    # Register the report as this agent's product so the NEXT agent is told it
    # exists — the whole reason experiment_design's prompt could talk about "any
    # LIT summary" while never actually receiving one.
    ref = doc_ref_from_save(res, kind="literature_report",
                            produced_by="literature",
                            summary=(key_findings or "").strip())
    if ref is None:
        return summary
    return ArtifactToolReturn(summary, {"literature_report": ref},
                              tool_call_id=tool_call_id,
                              name="save_literature_report")


@tool("list_fetch_requests")
def list_fetch_requests(status: str = "") -> str:
    """列出取文请求板上的请求（可按 status 过滤：pending/fulfilled/failed/dismissed）。

    用于查看你之前 request_fulltext 提交的请求是否已被用户满足
    （fulfilled = 全文已促进进大库，可用 fetch_paper_abstract / lib_search 取用）。"""
    from mast.knowledge.fetch_board import list_requests
    try:
        rows = list_requests(status or None)
    except Exception as e:  # pragma: no cover
        return f"list_fetch_requests failed: {type(e).__name__}: {e}"
    if not rows:
        return "取文请求板：无请求。" if not status else f"取文请求板：无 {status} 请求。"
    lines = [f"取文请求板（{len(rows)} 条）："]
    for r in rows[:40]:
        t = f" «{r['title']}»" if r.get("title") else ""
        note = f" — {r['note']}" if r.get("note") else ""
        lines.append(f"  [{r['status']}] {r['request_id']} work_id={r['work_id']}{t}{note}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Top-level assembly
# ─────────────────────────────────────────────────────────────────────

# Read-only search/abstract/citation/priors + library-curation tools.
LIBRARY_TOOLS: list = [
    search_fulltext,
    deep_read_papers,
    lib_list,
    lib_create,
    lib_switch,
    lib_copy,
    lib_add,
    lib_remove,
    lib_search,
    fetch_paper_abstract,
    propose_citations,
    literature_priors,
    request_fulltext,
    list_fetch_requests,
    save_literature_report,
]

AGENT_TOOLS: list = [
    search_papers,
    read_paper_section,
    extract_protocol,
    search_local_corpus,
    web_search,
    # Lives here rather than in LIBRARY_TOOLS because build_tools() only swaps
    # AGENT_TOOLS for "unavailable" placeholders — a network tool with no httpx
    # must be able to announce itself as dead rather than fail per call.
    fetch_fulltext_oa,
]


# buf=None fallbacks: SAME tool names as make_buffer_tools, so the system prompt
# never advertises a tool the model can't call when no BufferService is attached
# (it gets an honest "unavailable" placeholder instead of an unknown-tool error).
# The live orchestrator path always passes a real buf, so these are only used in
# offline/standalone construction.
@tool("read_latest_tip_status")
def _unavailable_tip_status() -> dict:
    """Read the most recent tip assessment from the vision buffer.

    (视觉缓冲当前未连接到本 agent — 返回不可用占位,不报错。)
    """
    return {"seqno": -1, "tip": None, "note": "视觉缓冲不可用（未连接 buffer）"}


@tool("get_scan_progress")
def _unavailable_scan_progress() -> dict:
    """Get current scan progress (line index / total / ETA seconds).

    (视觉缓冲当前未连接 — 返回不可用占位,不报错。)
    """
    return {"seqno": -1, "progress": None, "note": "视觉缓冲不可用（未连接 buffer）"}


@tool("get_tip_history_since")
def _unavailable_tip_history(since_seq: int) -> list:
    """Get tip-status entries newer than `since_seq`.

    (视觉缓冲当前未连接 — 返回空列表,不报错。)
    """
    return []


_UNAVAILABLE_BUFFER_TOOLS: list = [
    _unavailable_tip_status, _unavailable_scan_progress, _unavailable_tip_history,
]


# ─────────────────────────────────────────────────────────────────────
# Dead-tool detection: a tool whose hard dependency is missing
# is dead for the WHOLE session. Probe once at build time, tell the agent up
# front, and swap the dead tool for a same-named placeholder that fail-fasts with
# one clear "known constraint — do not retry" line. In the field, web_search
# (no TAVILY_API_KEY) failed 10× and search_papers (empty PDF corpus) 9×, because
# nothing told the agent the tool was permanently unavailable — it kept trying.
# ─────────────────────────────────────────────────────────────────────

def tool_availability() -> dict[str, str]:
    """Return ``{tool_name: reason}`` for the domain tools that are UNAVAILABLE
    right now (missing hard dependency). An available tool is absent from the map.

    Probed at agent-build time only. Currently covers the dependency-gated
    tools from the * ``web_search``  — needs a Tavily API key
      * ``search_papers`` — needs a non-empty local PDF corpus
      * ``fetch_fulltext_oa`` — needs an HTTP stack (httpx)
    The offline tools (search_local_corpus / lib_* / priors) are never gated here.

    Note what is deliberately NOT probed: numpy/pandas for the ingest half of
    ``fetch_fulltext_oa``. A missing ingest dependency still lets the fetch half
    legally retrieve the PDF, and the tool then reports where it landed so the
    operator can file it — more useful than pre-emptively declaring the whole
    tool dead.
    """
    out: dict[str, str] = {}
    if _read_tavily_key() is None:
        out["web_search"] = "TAVILY_API_KEY 未配置（联网检索不可用）"
    if not _all_pdfs():
        dirs = ", ".join(str(d) for d in _corpus_dirs())
        out["search_papers"] = f"本地 PDF 全文语料为空（{dirs} 下无 PDF）"
    try:
        from mast.knowledge import fetch as _fetch_mod
        if getattr(_fetch_mod, "httpx", None) is None:
            out["fetch_fulltext_oa"] = "网络组件 httpx 不可用（无法自主取文）"
    except Exception as exc:  # pragma: no cover — probe never breaks a build
        out["fetch_fulltext_oa"] = f"取文模块不可用（{type(exc).__name__}）"
    return out


def _make_unavailable_tool(name: str, reason: str):
    """A same-named stand-in for a dead domain tool. Keeps the tool NAME (so the
    system prompt stays valid and there is no unknown-tool error) but its
    description and its single return value both say, loudly, that it is
    unavailable and must not be retried."""
    from langchain_core.tools import StructuredTool

    msg = (
        f"{name} 当前不可用：{reason}。这是本会话的**已知约束**，请不要重复调用该"
        f"工具；改用 search_local_corpus / lib_search 等本地可用工具完成检索，"
        f"需要联网/PDF 全文时用 request_fulltext 请用户补充。"
    )

    # Every gated tool's parameters appear here: the placeholder must ACCEPT the
    # call the agent would have made and answer it with the message, because a
    # schema mismatch surfaces as a validation error the agent reads as "wrong
    # arguments" and retries — the exact loop the placeholder exists to end.
    def _fn(query: str = "", max_results: int = 5,
            doi_or_url: str = "", work_id: str = "",
            paper_id: str = "", section: str = "") -> str:
        return msg

    return StructuredTool.from_function(
        func=_fn, name=name, description="[当前不可用] " + msg,
    )


def unavailable_tools_note(unavailable: "dict[str, str] | None" = None) -> str:
    """A one-time system-prompt appendix naming the tools that are unavailable
    this session, so the agent never spends a turn (or ten) rediscovering it per
    call. Empty string when everything is available (no-op append)."""
    if unavailable is None:
        unavailable = tool_availability()
    if not unavailable:
        return ""
    lines = ["\n\n## 当前不可用的工具（已知约束，请勿调用）"]
    for nm, why in unavailable.items():
        lines.append(f"- `{nm}`：{why}")
    lines.append(
        "以上工具本会话不可用——**不要反复调用**。改用 search_local_corpus / "
        "lib_search 做本地检索；需要联网结果或 PDF 全文时，用 request_fulltext "
        "请用户补充，并在文献不足时如实说明。"
    )
    return "\n".join(lines)


def build_tools(
    buf: "BufferService | None",
    unavailable: "dict[str, str] | None" = None,
) -> list:
    # Dead-tool detection: swap any dependency-missing domain tool for a
    # same-named "unavailable" placeholder. Name + count are
    # preserved so the prompt stays valid and nothing downstream breaks.
    if unavailable is None:
        unavailable = tool_availability()
    tools: list = [
        _make_unavailable_tool(t.name, unavailable[t.name])
        if getattr(t, "name", "") in unavailable else t
        for t in AGENT_TOOLS
    ]
    # Library curation + read-only knowledge tools (P1) — offline, no hardware.
    tools = tools + list(LIBRARY_TOOLS)
    if buf is not None:
        tools = tools + make_buffer_tools(buf)
    else:
        # No buffer attached → bind no-op stand-ins under the same names so the
        # prompt's advertised buffer tools are always callable (honest
        # "unavailable" instead of an unknown-tool error).
        tools = tools + _UNAVAILABLE_BUFFER_TOOLS
    tools = tools + [
        make_handoff(
            "experiment_design",
            (
                "Hand prior-art summary to the Experiment Design agent so it "
                "can propose scan parameters informed by existing literature."
            ),
        ),
        make_handoff(
            "supervisor",
            (
                "Return control to the orchestrator/supervisor. Use when the "
                "literature search is complete and no immediate XD hand-off is "
                "needed, or when the corpus contains no relevant results."
            ),
        ),
    ]
    logger.info("literature: built %d tools (buf=%s)", len(tools), buf is not None)
    return tools


__all__ = [
    "search_papers",
    "read_paper_section",
    "extract_protocol",
    "search_local_corpus",
    "web_search",
    # P1 library curation + read-only knowledge tools
    "lib_list",
    "lib_create",
    "lib_switch",
    "lib_copy",
    "lib_add",
    "lib_remove",
    "lib_search",
    "fetch_paper_abstract",
    "propose_citations",
    "literature_priors",
    "request_fulltext",
    "list_fetch_requests",
    "save_literature_report",
    "LIBRARY_TOOLS",
    "AGENT_TOOLS",
    "build_tools",
    "tool_availability",
    "unavailable_tools_note",
]
