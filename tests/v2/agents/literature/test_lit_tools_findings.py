"""Regression tests for literature/tools.py + graph.py review findings.

Covers four findings against the literature agent:

  #48 search_papers AND-vs-OR: the docstring promises AND semantics
       (a paper is a hit only if EVERY query term appears). Verify a paper
       containing only SOME of the terms is NOT a hit, while one containing
       all terms IS.
  #49 extract_protocol must NOT swallow a PyMuPDF read failure: when
       read_paper_section returns the "read_paper_section failed: ..." error
       string, extract_protocol must surface it (not silently report
       "no parameters found").
  #50 graph.py docstring must reflect the real default model (Kimi K2.6),
       not "Opus 4.7", and must state that prompt caching only attaches to a
       real ChatAnthropic model.
  #96 read_paper_section / extract_protocol must close the PyMuPDF doc handle
       even when get_text raises mid-read (try/finally).

All tests run fully offline: no API key, no network, no real index. The fitz
read paths are exercised either with a real tiny PyMuPDF PDF or with a fake
fitz module monkeypatched in to simulate read failures and track close().

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/literature/test_lit_tools_findings.py -q -p no:randomly
"""
from __future__ import annotations

# ── path bootstrap BEFORE any mast.* imports (canonical tests/v2/ block) ──
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

import pytest  # noqa: E402

from mast.agents.literature import graph as litgraph  # noqa: E402
from mast.agents.literature import tools as littools  # noqa: E402
from mast.agents.literature.tools import (  # noqa: E402
    extract_protocol,
    read_paper_section,
    search_papers,
)


# ──────────────────────────────────────────────────────────────────────
# Real-PDF fixtures (PyMuPDF must be present in the v2 venv).
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def and_corpus(tmp_path, monkeypatch):
    """Two PDFs:

      both.pdf — contains BOTH "graphene" and "kagome"
      one.pdf  — contains "graphene" but NOT "kagome"

    For the query "graphene kagome", AND semantics must hit only both.pdf.
    """
    fitz = pytest.importorskip("fitz")

    def _write(name: str, text: str) -> None:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 60), text, fontsize=10)
        doc.save(tmp_path / f"{name}.pdf")
        doc.close()

    _write(
        "both",
        "Abstract\nWe study graphene grown on a kagome lattice substrate.\n",
    )
    _write(
        "onlygraphene",
        "Abstract\nWe study graphene on a hexagonal boron nitride substrate.\n",
    )
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(tmp_path))
    yield tmp_path


@pytest.fixture
def methods_corpus(tmp_path, monkeypatch):
    """A single PDF with a methods section carrying a bias value."""
    fitz = pytest.importorskip("fitz")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 60),
        "Abstract\nWSe2 by STM.\n\n2. Methods\nBias voltage was 200 mV.\n",
        fontsize=10,
    )
    doc.save(tmp_path / "paperX.pdf")
    doc.close()
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(tmp_path))
    yield tmp_path


# ──────────────────────────────────────────────────────────────────────
# search_papers must implement AND, not OR.
# ──────────────────────────────────────────────────────────────────────

class TestSearchPapersAndSemantics:
    def test_multi_term_is_and_not_or(self, and_corpus):
        """'graphene kagome' must hit only the doc containing BOTH terms."""
        result = search_papers.invoke({"query": "graphene kagome"})
        assert "both" in result, f"AND hit (both terms) missing:\n{result}"
        assert "onlygraphene" not in result, (
            "OR leak: a paper with only one of the two terms was returned, "
            f"violating the documented AND semantics:\n{result}"
        )

    def test_single_term_still_matches(self, and_corpus):
        """A single-term query matches every doc containing it (AND of one term)."""
        result = search_papers.invoke({"query": "graphene"})
        assert "both" in result and "onlygraphene" in result, (
            f"single-term query should match both docs:\n{result}"
        )

    def test_absent_term_excludes_all(self, and_corpus):
        """If one AND term is absent everywhere, no hits."""
        result = search_papers.invoke({"query": "graphene zzzznotpresent"})
        assert "no matches" in result.lower(), result

    def test_docstring_promises_and(self):
        """Docstring must explicitly describe AND semantics (alignment with impl)."""
        doc = search_papers.func.__doc__ or ""
        assert "AND query" in doc, doc
        assert "EVERY term" in doc, doc


# ──────────────────────────────────────────────────────────────────────
# extract_protocol must surface a read failure, not swallow it.
# ──────────────────────────────────────────────────────────────────────

class _FakeReadTool:
    """Stand-in for the read_paper_section StructuredTool that returns a fixed
    string from .invoke (the StructuredTool itself is a frozen pydantic model,
    so its .invoke cannot be monkeypatched directly)."""

    def __init__(self, ret: str):
        self._ret = ret

    def invoke(self, _payload):
        return self._ret


class TestExtractProtocolReadFailure:
    def test_read_failure_is_surfaced(self, monkeypatch, methods_corpus):
        """If read_paper_section returns a 'failed:' error, extract_protocol
        must propagate it (NOT report 'no parameters found')."""
        err = "read_paper_section failed: RuntimeError: corrupt page tree"

        # extract_protocol resolves `read_paper_section` from the module globals
        # at call time, so replacing the module attribute is sufficient.
        monkeypatch.setattr(littools, "read_paper_section", _FakeReadTool(err))
        result = extract_protocol.invoke({"paper_id": "paperX"})
        assert "failed" in result.lower() or "could not read" in result.lower(), (
            f"read failure was swallowed — got:\n{result}"
        )
        assert "no quantitative parameters" not in result.lower(), (
            f"read failure misreported as 'no parameters':\n{result}"
        )

    def test_methods_missing_still_falls_back(self, monkeypatch, methods_corpus):
        """A genuine 'methods absent' note ('read_paper_section: ...') must still
        trigger whole-paper fallback and find the bias value — NOT be treated as
        a failure."""
        note = (
            "read_paper_section: section 'methods' heading not detected in "
            "'paperX'. Try search_papers to locate keywords directly."
        )
        monkeypatch.setattr(littools, "read_paper_section", _FakeReadTool(note))
        result = extract_protocol.invoke({"paper_id": "paperX"})
        # Whole-paper fallback should still pull the bias from the PDF body.
        assert "200" in result and "bias" in result.lower(), (
            f"fallback path broke — got:\n{result}"
        )

    def test_real_methods_extraction_unaffected(self, methods_corpus):
        """Sanity: with a real methods section present, bias is extracted."""
        result = extract_protocol.invoke({"paper_id": "paperX"})
        assert "bias" in result.lower() and "200" in result, result


# ──────────────────────────────────────────────────────────────────────
# PyMuPDF doc handle must be closed even on mid-read error.
#
# Use a fake `fitz` module so we can (a) raise mid-iteration and (b) record
# whether close() was called. The fake is injected via sys.modules so the
# `import fitz` inside the tools picks it up.
# ──────────────────────────────────────────────────────────────────────

class _FakeDoc:
    """A fake PyMuPDF document.

    If raise_on_page >= 0, the page at that index raises on get_text().
    Records close() invocations so the test can assert the handle was freed.
    """

    def __init__(self, n_pages: int, raise_on_page: int, closed_log: list):
        self._n_pages = n_pages
        self._raise_on_page = raise_on_page
        self._closed_log = closed_log
        self.closed = False

    def __iter__(self):
        for i in range(self._n_pages):
            yield _FakePage(i, self._raise_on_page)

    def close(self):
        self.closed = True
        self._closed_log.append(id(self))


class _FakePage:
    def __init__(self, index: int, raise_on_page: int):
        self._index = index
        self._raise_on_page = raise_on_page

    def get_text(self, *_args, **_kwargs):
        if self._index == self._raise_on_page:
            raise RuntimeError("simulated PyMuPDF read fault")
        return f"page {self._index} text"


def _install_fake_fitz(monkeypatch, n_pages: int, raise_on_page: int):
    """Install a fake `fitz` module; returns the close-log list."""
    import types

    closed_log: list = []
    docs_created: list = []
    mod = types.ModuleType("fitz")

    def _open(_path=None, *_a, **_k):
        d = _FakeDoc(n_pages, raise_on_page, closed_log)
        docs_created.append(d)
        return d

    mod.open = _open  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fitz", mod)
    return closed_log, docs_created


class TestDocHandleClosed:
    def test_read_paper_section_closes_on_read_error(
        self, monkeypatch, methods_corpus
    ):
        """read_paper_section must close the doc even when get_text raises."""
        closed_log, docs = _install_fake_fitz(
            monkeypatch, n_pages=3, raise_on_page=1
        )
        result = read_paper_section.invoke(
            {"paper_id": "paperX", "section": "methods"}
        )
        assert "failed" in result.lower(), result
        assert docs, "fake fitz.open was never called"
        assert all(d.closed for d in docs), (
            "doc handle leaked on read error in read_paper_section"
        )
        assert closed_log, "close() was never invoked"

    def test_read_paper_section_closes_on_success(
        self, monkeypatch, methods_corpus
    ):
        """Handle is also closed on the normal success path (no leak)."""
        closed_log, docs = _install_fake_fitz(
            monkeypatch, n_pages=2, raise_on_page=-1
        )
        read_paper_section.invoke({"paper_id": "paperX", "section": "abstract"})
        assert docs and all(d.closed for d in docs), (
            "doc handle leaked on the success path"
        )

    def test_extract_protocol_fallback_closes_on_read_error(
        self, monkeypatch, methods_corpus
    ):
        """extract_protocol whole-paper fallback must close the doc even when
        the fallback get_text raises."""
        # Force the 'methods absent' note so extract_protocol takes the
        # whole-paper fallback branch (the one that opens its own doc).
        monkeypatch.setattr(
            littools,
            "read_paper_section",
            _FakeReadTool("read_paper_section: heading not detected"),
        )
        closed_log, docs = _install_fake_fitz(
            monkeypatch, n_pages=3, raise_on_page=1
        )
        result = extract_protocol.invoke({"paper_id": "paperX"})
        assert "failed" in result.lower(), result
        assert docs and all(d.closed for d in docs), (
            "doc handle leaked in extract_protocol fallback on read error"
        )


# ──────────────────────────────────────────────────────────────────────
# graph.py docstring honesty (GLM default, not Opus 4.7).
# ──────────────────────────────────────────────────────────────────────

class TestGraphDocstringHonesty:
    def test_module_docstring_does_not_claim_opus(self):
        mod_doc = litgraph.__doc__ or ""
        assert "Opus 4.7" not in mod_doc, (
            "graph.py module docstring still claims Opus 4.7 as the default; "
            "the real default is Kimi K3 (AGENT_MODEL['literature'])."
        )
        assert "Kimi K3" in mod_doc, (
            "graph.py module docstring should name the real default Kimi K3."
        )

    def test_build_docstring_does_not_claim_opus(self):
        build_doc = litgraph.build.__doc__ or ""
        assert "Opus 4.7" not in build_doc, (
            "build() docstring still claims Opus 4.7; default is Kimi K3."
        )

    def test_default_model_is_actually_kimi(self):
        """Cross-check the claim against the model registry (source of truth)."""
        from mast.agents._shared.models import AGENT_MODEL, KIMI_K3, provider_for

        assert AGENT_MODEL["literature"] == KIMI_K3
        # And it routes to a non-anthropic (OpenAI-compatible) provider, so the
        # AnthropicPromptCachingMiddleware branch is genuinely Claude-only.
        assert provider_for(AGENT_MODEL["literature"]) != "anthropic"

    def test_docstring_explains_cache_is_claude_only(self):
        """Docstring should make the prompt-caching condition explicit/honest."""
        mod_doc = litgraph.__doc__ or ""
        assert "ChatAnthropic" in mod_doc, (
            "docstring should explain caching attaches only to ChatAnthropic."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
