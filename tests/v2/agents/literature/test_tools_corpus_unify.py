"""The literature agent's PDF tools must see what ``ingest_pdf`` wrote.

Before this, three code paths disagreed about where papers live:

  * ``ingest_pdf`` writes ``<MAST_PAPERS_DIR>/<slug>/source.pdf`` + ``fulltext.txt``
  * ``search_papers`` / ``read_paper_section`` scanned ``<repo>/data/papers``
    (env ``MAST_PAPER_CORPUS``), a directory that does not exist
  * the operator's own PDFs sit in ``<repo>/papers``

so a paper the system had just fetched and ingested was unreadable by the very
tools whose job is to read it, and every ingested paper answered to the paper_id
``source`` because they are all named ``source.pdf``.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/literature/test_tools_corpus_unify.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (canonical block for tests/v2/) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest

from mast.agents.literature import tools as littools

fitz = pytest.importorskip("fitz", reason="PyMuPDF needed to build corpus fixtures")


# ── helpers ──────────────────────────────────────────────────────────────

def _write_pdf(path: Path, text: str) -> Path:
    """A real one-page PDF with a text layer (so PyMuPDF finds the words)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def papers_root(tmp_path, monkeypatch):
    """Point ``knowledge.paths.papers_dir()`` at a sandbox this test controls.

    The autouse ``_isolate_literature_data`` fixture already sets both vars; this
    re-points them (explicitly requested fixtures run after autouse ones, so this
    wins).

    ``MAST_PAPER_CORPUS`` is re-pointed, never *unset*: with it unset
    ``_corpus_dirs()`` falls back to the legacy ``<repo>/papers``, which on a real
    machine is the operator's own PDF collection — an early draft of this file did
    exactly that and the assertion came back holding 24 of their papers. The
    legacy-fallback behaviour is covered below with a faked repo root instead.
    """
    d = tmp_path / "papers"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MAST_PAPERS_DIR", str(d))
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(tmp_path / "empty_corpus"))
    return d


# ── P-1: ingested layout is reachable ────────────────────────────────────

def test_papers_dir_is_always_scanned(papers_root):
    """The canonical ingest destination is in the corpus list without any env."""
    dirs = littools._corpus_dirs()
    assert papers_root.resolve() in dirs


def test_search_papers_finds_an_ingested_paper(papers_root):
    _write_pdf(papers_root / "w2001_slug" / "source.pdf",
               "tunneling spectroscopy of graphene on iridium")
    out = littools.search_papers.invoke(
        {"query": "graphene iridium", "max_results": 5})
    assert "w2001_slug" in out
    assert "no PDFs found" not in out


def test_legacy_corpora_scanned_only_when_present_and_env_unset(tmp_path, monkeypatch):
    """With no corpus env, the operator's hand-managed folders are searched too.

    ``<repo>/papers`` is where this operator actually keeps PDFs, and
    ``search_papers`` never looked there — which is why it reported an empty
    corpus on a machine holding two dozen papers. The repo root is faked here so
    the assertion cannot reach the real collection.
    """
    fake_repo = tmp_path / "repo"
    (fake_repo / "MASTv2").mkdir(parents=True)
    (fake_repo / "papers").mkdir()
    (fake_repo / "data" / "papers").mkdir(parents=True)
    monkeypatch.setattr(littools, "_repo_root", lambda: fake_repo)
    monkeypatch.setenv("MAST_PAPERS_DIR", str(tmp_path / "canonical"))
    monkeypatch.delenv("MAST_PAPER_CORPUS", raising=False)

    dirs = littools._corpus_dirs()
    assert (fake_repo / "papers").resolve() in dirs
    assert (fake_repo / "data" / "papers").resolve() in dirs

    # A legacy dir that does not exist costs nothing.
    (fake_repo / "data" / "papers").rmdir()
    assert (fake_repo / "data" / "papers").resolve() not in littools._corpus_dirs()


def test_explicit_corpus_env_suppresses_legacy_fallback(tmp_path, monkeypatch):
    """Pointing the var somewhere explicit is how tests stay off the real corpus."""
    fake_repo = tmp_path / "repo"
    (fake_repo / "papers").mkdir(parents=True)
    monkeypatch.setattr(littools, "_repo_root", lambda: fake_repo)
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(tmp_path / "only_this"))
    assert (fake_repo / "papers").resolve() not in littools._corpus_dirs()


def test_paper_corpus_env_adds_rather_than_replaces(tmp_path, monkeypatch):
    """An extra corpus dir must not hide the canonical ingest destination.

    ``MAST_PAPER_CORPUS`` used to REPLACE the search path entirely, so pointing
    it at a scratch folder made every ingested paper invisible.
    """
    canonical = tmp_path / "canonical"
    extra = tmp_path / "extra"
    canonical.mkdir()
    extra.mkdir()
    monkeypatch.setenv("MAST_PAPERS_DIR", str(canonical))
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(extra))

    _write_pdf(canonical / "slug_a" / "source.pdf", "alpha keyword here")
    _write_pdf(extra / "hand_dropped.pdf", "alpha keyword here")

    dirs = littools._corpus_dirs()
    assert canonical.resolve() in dirs and extra.resolve() in dirs
    out = littools.search_papers.invoke({"query": "alpha", "max_results": 5})
    assert "slug_a" in out and "hand_dropped" in out


# ── P-2: paper_id identity ───────────────────────────────────────────────

def test_ingested_paper_id_is_the_slug_not_source(papers_root):
    p = _write_pdf(papers_root / "w42_slug" / "source.pdf", "x")
    assert littools._paper_id(p) == "w42_slug"


def test_hand_dropped_paper_keeps_its_stem(papers_root):
    p = _write_pdf(papers_root / "Nguyen2019_AuSTM.pdf", "x")
    assert littools._paper_id(p) == "Nguyen2019_AuSTM"


def test_two_ingested_papers_do_not_collide(papers_root):
    """Both are ``source.pdf``; each must still resolve to its own file."""
    _write_pdf(papers_root / "slug_one" / "source.pdf", "first paper unique_one")
    _write_pdf(papers_root / "slug_two" / "source.pdf", "second paper unique_two")

    a = littools._find_paper("slug_one")
    b = littools._find_paper("slug_two")
    assert a is not None and b is not None and a != b
    assert "unique_one" in littools._paper_text("slug_one")[0]
    assert "unique_two" in littools._paper_text("slug_two")[0]


def test_missing_and_broken_reads_stay_distinguishable(papers_root, monkeypatch):
    """"we cannot read it" and "it is not there" are different facts."""
    assert littools._paper_text("nope")[1] == "missing"

    _write_pdf(papers_root / "boom_slug" / "source.pdf", "text")
    import types
    mod = types.ModuleType("fitz")

    def _boom(*_a, **_k):
        raise RuntimeError("simulated read fault")

    mod.open = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fitz", mod)
    assert littools._paper_text("boom_slug")[1] == "error"


# ── fulltext.txt is preferred over re-parsing the PDF ────────────────────

def test_reads_cached_fulltext_when_pdf_has_no_text_layer(papers_root):
    """A scanned paper's only readable text is the OCR output ingest cached.

    Re-extracting with PyMuPDF yields nothing, which would read as "the paper
    does not mention it" rather than "we cannot read this paper".
    """
    slug = papers_root / "scanned_slug"
    _write_pdf(slug / "source.pdf", "")          # image-only stand-in: no words
    (slug / "fulltext.txt").write_text(
        "Methods\nSTM images were recorded at a bias of -0.8 V and a setpoint "
        "of 50 pA.\nResults\nWe observe a moire.",
        encoding="utf-8")

    text, kind, _detail = littools._paper_text("scanned_slug")
    assert kind == "fulltext" and "setpoint of 50 pA" in text

    out = littools.read_paper_section.invoke(
        {"paper_id": "scanned_slug", "section": "methods"})
    assert "-0.8 V" in out

    proto = littools.extract_protocol.invoke({"paper_id": "scanned_slug"})
    assert "0.8 V" in proto and "50 pA" in proto


def test_absent_paper_says_absent_not_no_parameters(papers_root):
    """A paper we do not have is a corpus fact, reported as such."""
    out = littools.extract_protocol.invoke({"paper_id": "does_not_exist"})
    assert "not found in corpus" in out
    assert "no quantitative parameters found" not in out


def test_read_fault_reports_failure_not_absence(papers_root, monkeypatch):
    """extract_protocol must never turn "cannot read" into "no parameters"."""
    _write_pdf(papers_root / "boom_slug" / "source.pdf", "text")
    import types
    mod = types.ModuleType("fitz")

    def _boom(*_a, **_k):
        raise RuntimeError("simulated read fault")

    mod.open = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fitz", mod)

    out = littools.extract_protocol.invoke({"paper_id": "boom_slug"})
    assert "could not read" in out and "simulated read fault" in out
    assert "no quantitative parameters found" not in out


# ── SI attachments are supplements, not papers ───────────────────────────

def test_attachments_are_excluded_from_the_corpus(papers_root):
    _write_pdf(papers_root / "slug_a" / "source.pdf", "main paper body")
    _write_pdf(papers_root / "slug_a" / "attachments" / "si_1.pdf", "supplementary")

    ids = {littools._paper_id(p) for p in littools._all_pdfs()}
    assert ids == {"slug_a"}
    assert littools._find_paper("si_1") is None


def test_legacy_si_sibling_is_associated(papers_root):
    main = _write_pdf(papers_root / "Chen2020.pdf", "main")
    _write_pdf(papers_root / "Chen2020_SI.pdf", "supplement")
    _write_pdf(papers_root / "Other2020.pdf", "unrelated")

    sibs = littools._legacy_si_siblings(main)
    assert [p.name for p in sibs] == ["Chen2020_SI.pdf"]


def test_fetch_staging_is_not_a_second_copy_of_the_paper(papers_root):
    """``fetch`` stages a download by DOI; ``ingest`` files it by work_id.

    Both used to land directly under <papers>, so a fetched-then-ingested paper
    sat in the corpus twice under two different paper_ids — once complete, once
    as a bare PDF with no full text.
    """
    _write_pdf(papers_root / "W123" / "source.pdf", "the ingested copy")
    _write_pdf(papers_root / "_incoming" / "10.1-x-abc123" / "source.pdf",
               "the staged download")

    ids = {littools._paper_id(p) for p in littools._all_pdfs()}
    assert ids == {"W123"}
    assert littools._find_paper("10.1-x-abc123") is None


def test_fetch_writes_into_the_staging_directory(papers_root):
    from mast.knowledge import fetch as fetch_mod
    path = Path(fetch_mod._save_pdf(b"%PDF-1.4 x", "doi-slug-abc"))
    assert path.parent.parent.name == "_incoming"
    assert path.is_file() and path.name == "source.pdf"


def test_legacy_si_is_flagged_in_search_results(papers_root):
    _write_pdf(papers_root / "Chen2020.pdf", "moire superlattice")
    _write_pdf(papers_root / "Chen2020_SI.pdf", "moire superlattice details")
    out = littools.search_papers.invoke({"query": "moire", "max_results": 5})
    assert "Chen2020_SI" in out and "SI)" in out


# ── P-5: tavily key path ─────────────────────────────────────────────────

def test_tavily_key_read_from_repo_api_key_dir(tmp_path, monkeypatch):
    """The lookup used to point inside the archived v1 ``mast/`` package."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    fake_repo = tmp_path / "repo"
    (fake_repo / "api key").mkdir(parents=True)
    (fake_repo / "api key" / "tavily.env").write_text(
        "# comment line\ntvly-secret-value\n", encoding="utf-8")
    monkeypatch.setattr(littools, "_repo_root", lambda: fake_repo)
    assert littools._read_tavily_key() == "tvly-secret-value"


def test_tavily_key_accepts_key_equals_value_form(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    fake_repo = tmp_path / "repo"
    (fake_repo / "api key").mkdir(parents=True)
    (fake_repo / "api key" / "tavily.env").write_text(
        "TAVILY_API_KEY=tvly-kv\n", encoding="utf-8")
    monkeypatch.setattr(littools, "_repo_root", lambda: fake_repo)
    assert littools._read_tavily_key() == "tvly-kv"


def test_tavily_absent_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(littools, "_repo_root", lambda: tmp_path / "empty")
    assert littools._read_tavily_key() is None


def test_repo_root_resolves_to_the_dir_holding_mastv2():
    root = littools._repo_root()
    assert (root / "MASTv2").is_dir()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
