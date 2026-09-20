"""The agent-facing side of full-text passage search.

Sits between ``read_paper_section`` (one paper, one named section, regex) and
``deep_read_papers`` (a full LLM read, minutes and tens of thousands of tokens).
Its job is to answer "where in these papers does it talk about X" cheaply — and,
when retrieval has degraded, to say so instead of handing back keyword hits that
look like semantic ones.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/literature/test_search_fulltext_tool.py -q
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
from mast.agents.literature.tools import search_fulltext
from mast.knowledge import fulltext_search as fts


def _patch_search(monkeypatch, hits, status, seen: dict | None = None):
    def _fake(query, slugs=None, k=8, **kw):
        if seen is not None:
            seen.update(query=query, slugs=slugs, k=k)
        return hits, status

    monkeypatch.setattr(fts, "search_chunks", _fake)


_OK_STATUS = {"retrieval": "semantic", "degraded": False, "reason": "",
              "trustworthy": True, "n_papers": 2, "n_chunks": 40}


def _hit(slug="W1", title="A paper", page=3, score=0.81,
         text="The tip was prepared by field emission at 500 V."):
    return {"slug": slug, "title": title, "chunk_id": "c00007", "page": page,
            "score": score, "text": text}


# ── the normal answer ────────────────────────────────────────────────────

def test_reports_passage_paper_and_page(monkeypatch):
    _patch_search(monkeypatch, [_hit()], _OK_STATUS)
    out = search_fulltext.invoke({"query": "针尖怎么处理的"})
    assert "A paper" in out and "W1" in out
    assert "p.4" in out, "pages are 0-based on disk and must be shown 1-based"
    assert "field emission" in out
    assert "0.81" in out


def test_searches_everything_when_no_papers_named(monkeypatch):
    seen: dict = {}
    _patch_search(monkeypatch, [_hit()], _OK_STATUS, seen)
    search_fulltext.invoke({"query": "q"})
    assert seen["slugs"] is None


def test_paper_refs_are_resolved_to_slugs(monkeypatch):
    """A work_id URL and a bare id are the same paper on disk."""
    seen: dict = {}
    _patch_search(monkeypatch, [_hit()], _OK_STATUS, seen)
    search_fulltext.invoke({"query": "q",
                            "paper_refs": ["https://openalex.org/W42", "W_live_a"]})
    assert "W42" in seen["slugs"] and "W_live_a" in seen["slugs"]


def test_k_is_passed_through(monkeypatch):
    seen: dict = {}
    _patch_search(monkeypatch, [_hit()], _OK_STATUS, seen)
    search_fulltext.invoke({"query": "q", "k": 3})
    assert seen["k"] == 3


def test_points_at_the_right_neighbouring_tool(monkeypatch):
    _patch_search(monkeypatch, [_hit()], _OK_STATUS)
    out = search_fulltext.invoke({"query": "q"})
    assert "deep_read_papers" in out and "read_paper_section" in out


# ── the honest empties ───────────────────────────────────────────────────

def test_no_local_fulltext_says_how_to_get_some(monkeypatch):
    _patch_search(monkeypatch, [], {**_OK_STATUS, "n_papers": 0, "n_chunks": 0,
                                    "reason": ""})
    out = search_fulltext.invoke({"query": "q"})
    assert "fetch_fulltext_oa" in out and "request_fulltext" in out


def test_searched_but_found_nothing_says_how_many_were_searched(monkeypatch):
    _patch_search(monkeypatch, [], _OK_STATUS)
    out = search_fulltext.invoke({"query": "q"})
    assert "2 篇" in out and "40" in out


def test_explicit_reason_is_surfaced(monkeypatch):
    _patch_search(monkeypatch, [], {**_OK_STATUS, "n_papers": 0,
                                    "reason": "这些论文没有可检索的全文块"})
    assert "没有可检索的全文块" in search_fulltext.invoke({"query": "q"})


def test_empty_query_is_rejected():
    assert "需要 query" in search_fulltext.invoke({"query": "  "})


# ── degradation ──────────────────────────────────────────────────────────

def test_degraded_but_usable_is_prefixed(monkeypatch):
    _patch_search(monkeypatch, [_hit()],
                  {**_OK_STATUS, "degraded": True, "retrieval": "keyword",
                   "trustworthy": True})
    out = search_fulltext.invoke({"query": "q"})
    assert out.startswith("[降级:关键词匹配"), out[:80]


def test_degraded_and_worthless_withholds_the_list(monkeypatch):
    """An information-free ranking must not be handed over looking like results."""
    _patch_search(monkeypatch, [_hit()],
                  {**_OK_STATUS, "degraded": True, "retrieval": "keyword",
                   "trustworthy": False, "reason": "语义 embedding 不可用"})
    out = search_fulltext.invoke({"query": "q"})
    assert "没有可用结果" in out
    assert "field emission" not in out, "worthless hits must not be shown"
    assert "DASHSCOPE_API_KEY" in out


def test_backend_exception_never_escapes(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("kaput")

    monkeypatch.setattr(fts, "search_chunks", _boom)
    out = search_fulltext.invoke({"query": "q"})
    assert out.startswith("search_fulltext failed:") and "kaput" in out


# ── wiring ───────────────────────────────────────────────────────────────

def test_tool_is_registered():
    names = [getattr(t, "name", "") for t in littools.build_tools(None)]
    assert "search_fulltext" in names


def test_prompt_teaches_the_three_way_choice():
    from mast.agents.literature.prompts import SYSTEM_PROMPT
    assert "search_fulltext" in SYSTEM_PROMPT
    assert "read_paper_section" in SYSTEM_PROMPT and "deep_read_papers" in SYSTEM_PROMPT


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
