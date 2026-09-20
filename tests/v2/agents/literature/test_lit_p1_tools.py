"""v2 tests for the literature agent P1 library-curation + knowledge tools.

Covers (design owner G — big index is the ONE true library, others are pointer
sets into it; in-library search = big-index search filtered to member work_ids):

  - lib_create / lib_switch / lib_add / lib_remove / lib_list against a
    synthetic registry.json (process-wide singleton repointed at a temp dir)
  - lib_search restricted to a library's members (big-index filter)
  - lib_search over the whole corpus (no library_id)
  - fetch_paper_abstract exact lookup
  - propose_citations (read-only recall)
  - literature_priors (clearly labelled non-authoritative)
  - build_tools attaches all 9 library tools, no cross-agent import

Synthetic fixtures only: a tiny vectors.npy + parquet + manifest.json index
(reusing the same builder shape as test_literature_index_search.py) with the
DashScope embed stubbed, plus a temp registry + temp priors.json. No network,
no API key, no real 150 MB index.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/literature/test_lit_p1_tools.py -q -p no:randomly
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

import json

import numpy as np
import pandas as pd
import pytest

import mast.knowledge.libraries as libmod
import mast.knowledge.literature_index as li
import mast.knowledge.priors as priors
from mast.agents.literature import tools as littools


# ──────────────────────────────────────────────────────────────────────
# Synthetic literature index (same row shape as test_literature_index_search)
# ──────────────────────────────────────────────────────────────────────

_DIM = li._DIM


def _vec(seed: int) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[seed % _DIM] = 1.0
    v[(seed * 7 + 3) % _DIM] = 0.3
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32)


def _build_index(tmp: Path) -> Path:
    idx = tmp / "literature_index"
    idx.mkdir(parents=True, exist_ok=True)
    vectors = np.stack([_vec(0), _vec(1), _vec(2), _vec(3)]).astype(np.float32)
    np.save(idx / "vectors.npy", vectors)
    meta_rows = [
        {"work_id": "W100", "doi": "https://doi.org/10.1/a", "title": "Kondo on Au(111)",
         "year": 2018, "journal": "PRL", "cited": 42},
        {"work_id": "W200", "doi": "https://doi.org/10.1/b", "title": "Graphene growth",
         "year": 2015, "journal": "Nature", "cited": 99},
        {"work_id": "W300", "doi": "", "title": "User contributed STM study",
         "year": 2021, "journal": "JoVE", "cited": 0},
        {"work_id": "W400", "doi": "https://doi.org/10.1/d", "title": "Merged paper",
         "year": 2020, "journal": "ACS Nano", "cited": 5},
    ]
    pd.DataFrame(meta_rows).to_parquet(idx / "metadata.parquet")
    abs_rows = [
        {"work_id": "W100", "abstract": "We study the Kondo effect of a single magnetic "
         "atom adsorbed on Au(111) using low-temperature STM and STS.",
         "user_abstract": "", "fulltext_excerpt": "", "abstract_provenance": "openalex",
         "authors": "A; B", "first_author": "A", "concepts": "Kondo; STM",
         "keywords": "kondo;au111", "work_type": "article", "cited_by_count": 42,
         "source": "openalex", "kind": "openalex"},
        {"work_id": "W200", "abstract": "Epitaxial graphene growth on SiC studied by STM.",
         "user_abstract": "", "fulltext_excerpt": "", "abstract_provenance": "openalex",
         "authors": "C", "first_author": "C", "concepts": "graphene", "keywords": "graphene;sic",
         "work_type": "article", "cited_by_count": 99, "source": "openalex", "kind": "openalex"},
        {"work_id": "W300", "abstract": "Operator-contributed abstract about tip preparation.",
         "user_abstract": "Operator-contributed abstract about tip preparation.",
         "fulltext_excerpt": "", "abstract_provenance": "user", "authors": "Operator",
         "first_author": "Operator", "concepts": "", "keywords": "", "work_type": "user_pdf",
         "cited_by_count": 0, "source": "user_pdf", "kind": "user_abstract"},
        {"work_id": "W400", "abstract": "OpenAlex abstract for the merged paper.",
         "user_abstract": "", "fulltext_excerpt": "", "abstract_provenance": "openalex",
         "authors": "D; E", "first_author": "D", "concepts": "merge", "keywords": "merge",
         "work_type": "article", "cited_by_count": 5, "source": "openalex", "kind": "openalex"},
    ]
    pd.DataFrame(abs_rows).to_parquet(idx / "abstracts.parquet")
    cls_rows = [
        {"work_id": "W100", "primary_category": "magnetic_spm", "material": "Au(111)",
         "confidence": 0.9},
    ]
    pd.DataFrame(cls_rows).to_parquet(idx / "classified.parquet")
    manifest = {"source_repo": "synthetic", "model": li._MODEL, "dim": _DIM,
                "n_base": 4, "n_user": 0}
    (idx / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return idx


def _repoint_index(monkeypatch, idx: Path) -> None:
    monkeypatch.setattr(li, "_INDEX_DIR", idx, raising=True)
    monkeypatch.setattr(li, "VECTORS_PATH", idx / "vectors.npy", raising=True)
    monkeypatch.setattr(li, "METADATA_PATH", idx / "metadata.parquet", raising=True)
    monkeypatch.setattr(li, "ABSTRACTS_PATH", idx / "abstracts.parquet", raising=True)
    monkeypatch.setattr(li, "CLASSIFIED_PATH", idx / "classified.parquet", raising=True)
    monkeypatch.setattr(li, "MANIFEST_PATH", idx / "manifest.json", raising=True)
    li.invalidate_caches()


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def synth_index(tmp_path, monkeypatch):
    """Synthetic index + stubbed embedder. Embedder returns _vec(seed) so the
    caller controls which row is the top hit by routing query → seed."""
    idx = _build_index(tmp_path)
    _repoint_index(monkeypatch, idx)

    # Map known query keywords to a target row seed; default = row 0.
    def _fake_embed(q, **kw):
        ql = (q or "").lower()
        if "graphene" in ql:
            return _vec(1)
        if "tip" in ql:
            return _vec(2)
        if "merged" in ql:
            return _vec(3)
        return _vec(0)

    monkeypatch.setattr(li, "_embed_query", _fake_embed)
    return idx


@pytest.fixture
def synth_registry(tmp_path, monkeypatch):
    """Repoint the process-wide library registry singleton at a temp dir.

    Via the ``MAST_LITERATURE_LIBS_DIR`` env override published by
    ``knowledge/paths.py`` (the old ``_DEFAULT_LIBS_DIR`` module constant is gone —
    all five private repo-walks in ``knowledge/*`` converged onto one resolver on
    2026-07-29). The assert is deliberate: if isolation ever silently stops
    working, this test file writes into the operator's real registry.
    """
    libs_dir = tmp_path / "literature_libs"
    monkeypatch.setenv("MAST_LITERATURE_LIBS_DIR", str(libs_dir))
    # No active experiment: since 2026-07-29 an omitted library_id resolves to the
    # EFFECTIVE library, which prefers the active experiment's own one. Without
    # this the tests below read the operator's live active_scope and assert against
    # whatever experiment happens to be open on the machine.
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(tmp_path / "experiments"))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "exp.db"))
    libmod.reset_default_registry()
    assert libmod.get_registry()._dir == libs_dir
    yield libs_dir
    libmod.reset_default_registry()


@pytest.fixture
def synth_priors(tmp_path, monkeypatch):
    """Write a tiny literature_priors.json and point the priors module at it."""
    p = tmp_path / "literature_priors.json"
    data = {
        "by_material_phase": {
            "Au(111)": {
                "imaging": {
                    "n_papers": 7,
                    "sources": {},
                    "bias_v": {"p25": 0.05, "p50": 0.1, "p75": 0.5, "min": 0.01,
                               "max": 1.0, "n": 7},
                    "temperature_k": {"p25": 4.2, "p50": 5.0, "p75": 77.0, "min": 4.2,
                                      "max": 300.0, "n": 7},
                }
            }
        }
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    priors._set_priors_path_for_test(p)
    yield p
    priors._set_priors_path_for_test(None)


# Helper: most tools return a string; .invoke({...}) routes through the schema.
def _inv(tool, **kwargs):
    return tool.invoke(kwargs)


# ──────────────────────────────────────────────────────────────────────
# Library CRUD
# ──────────────────────────────────────────────────────────────────────

def test_lib_list_starts_with_global(synth_registry):
    out = _inv(littools.lib_list)
    assert "reading" in out       # the auto-created global reading library
    # With no experiment active, reading is BOTH the manual pointer and the
    # effective (default) target. The list names the effective one explicitly —
    # "which library does an add land in" must never be a guess.
    assert "手动指针" in out
    assert "有效库" in out


def test_lib_create_and_switch(synth_registry):
    out = _inv(littools.lib_create, name="Kondo studies", scope="custom")
    assert "created library" in out
    # slug derived from the name
    assert "kondo_studies" in out

    # lib_switch sets the MANUAL pointer (used only while no experiment is
    # active) — it no longer claims to switch a global "current library".
    sw = _inv(littools.lib_switch, library_id="kondo_studies")
    assert "手动指针已设为 'kondo_studies'" in sw

    listed = _inv(littools.lib_list)
    kondo_line = next(ln for ln in listed.splitlines() if "kondo_studies" in ln)
    assert "手动指针" in kondo_line
    # and with no experiment active the manual pointer IS the effective target
    assert "有效库" in kondo_line


def test_lib_create_rejects_global_scope(synth_registry):
    out = _inv(littools.lib_create, name="bad", scope="global")
    assert "rejected" in out.lower()


def test_lib_switch_unknown_id(synth_registry):
    out = _inv(littools.lib_switch, library_id="does_not_exist")
    assert "rejected" in out.lower()


def test_lib_add_and_remove(synth_registry):
    _inv(littools.lib_create, name="Kondo", scope="custom")
    add = _inv(littools.lib_add, work_ids=["W100", "W200"], library_id="kondo",
               reason="prior art")
    assert "added 2" in add
    assert "library now holds 2" in add

    # idempotent re-add → already-present
    add2 = _inv(littools.lib_add, work_ids=["W100"], library_id="kondo")
    assert "added 0" in add2
    assert "already-present 1" in add2

    rem = _inv(littools.lib_remove, work_ids=["W200"], library_id="kondo")
    assert "removed 1" in rem
    assert "library now holds 1" in rem


def test_lib_add_uses_manual_pointer_when_no_experiment_active(synth_registry):
    """No experiment active ⇒ the effective library IS the manual pointer.

    This is the unchanged half of the 2026-07-29 semantics: only when an
    experiment is running does an omitted library_id go somewhere else (that
    experiment's own library) — see test_experiment_library.py.
    """
    _inv(littools.lib_create, name="Active Lib", scope="custom")
    _inv(littools.lib_switch, library_id="active_lib")
    add = _inv(littools.lib_add, work_ids=["W300"])  # no library_id → effective
    assert "→ 'active_lib'" in add
    assert "added 1" in add


def test_lib_add_rejects_invalid_work_id(synth_registry):
    _inv(littools.lib_create, name="L", scope="custom")
    add = _inv(littools.lib_add, work_ids=["", "W123"], library_id="l")
    assert "added 1" in add        # W123 ok
    assert "rejected 1" in add     # "" rejected


# ──────────────────────────────────────────────────────────────────────
# lib_search: big-index filtered to a library's members
# ──────────────────────────────────────────────────────────────────────

def test_lib_search_filters_to_members(synth_index, synth_registry):
    # Build a library holding ONLY W100. A graphene query is closest to W200,
    # but member-filtering to {W100} must drop W200 → only W100 (or nothing).
    _inv(littools.lib_create, name="OnlyKondo", scope="custom")
    _inv(littools.lib_add, work_ids=["W100"], library_id="onlykondo")

    out = _inv(littools.lib_search, query="graphene", library_id="onlykondo", k=8)
    assert "W200" not in out, "non-member W200 must be filtered out"
    # The library has W100, which won't match a graphene query well, but the
    # filter is what we assert: no non-members leak in.
    for ln in out.splitlines():
        assert "W200" not in ln and "W300" not in ln and "W400" not in ln


def test_lib_search_returns_member_hit(synth_index, synth_registry):
    _inv(littools.lib_create, name="KondoLib", scope="custom")
    _inv(littools.lib_add, work_ids=["W100", "W200"], library_id="kondolib")
    # Kondo query → row 0 = W100, which IS a member → must surface.
    out = _inv(littools.lib_search, query="kondo au111", library_id="kondolib", k=8)
    assert "W100" in out
    assert "doi=" in out


def test_lib_search_whole_corpus_with_star(synth_index, synth_registry):
    """``library_id="*"`` is the explicit whole-corpus search.

    An OMITTED library_id used to mean "all 50k papers", which made lib_search a
    duplicate of search_local_corpus and meant the tool advertised as "search this
    library" quietly searched everything. It now means the effective library.
    """
    out = _inv(littools.lib_search, query="graphene", k=8, library_id="*")
    assert "whole corpus" in out
    assert "W200" in out  # graphene query → W200 surfaces from the full index


def test_lib_search_without_library_id_scopes_to_the_effective_library(
        synth_index, synth_registry):
    """Omitted library_id searches only what has been curated."""
    _inv(littools.lib_create, name="Only Kondo", scope="custom")
    _inv(littools.lib_switch, library_id="only_kondo")
    _inv(littools.lib_add, work_ids=["W100"], library_id="only_kondo")
    out = _inv(littools.lib_search, query="graphene", k=8)
    assert "only_kondo" in out
    assert "W200" not in out  # the closest match is NOT a member → filtered out


def test_lib_search_empty_library_note(synth_index, synth_registry):
    _inv(littools.lib_create, name="Empty", scope="custom")
    out = _inv(littools.lib_search, query="anything", library_id="empty", k=8)
    assert "还没有文献" in out
    # and it points at the way to actually find some, rather than dead-ending
    assert "search_local_corpus" in out


def test_lib_search_unknown_library(synth_index, synth_registry):
    out = _inv(littools.lib_search, query="kondo", library_id="nope", k=8)
    assert "rejected" in out.lower()


def test_lib_search_member_filter_tolerates_bare_vs_url(tmp_path, monkeypatch,
                                                        synth_registry):
    """the corpus keys rows by the full OpenAlex URL, but the
    model adds library members as the BARE id. The member filter must match
    across the two forms, else a library-scoped search silently returns nothing."""
    idx = tmp_path / "literature_index"
    idx.mkdir(parents=True, exist_ok=True)
    np.save(idx / "vectors.npy", np.stack([_vec(0), _vec(1)]).astype(np.float32))
    pd.DataFrame([
        {"work_id": "https://openalex.org/W3041272829", "doi": "https://doi.org/10.1/z",
         "title": "URL-keyed Kondo paper", "year": 2019, "journal": "PRB", "cited": 7,
         "abstract": "Kondo abstract."},
        {"work_id": "https://openalex.org/W4312129937", "doi": "",
         "title": "Other paper", "year": 2022, "journal": "Nano", "cited": 3,
         "abstract": "Other abstract."},
    ]).to_parquet(idx / "metadata.parquet")
    _repoint_index(monkeypatch, idx)
    monkeypatch.setattr(li, "_embed_query", lambda q, **kw: _vec(0))  # → row 0

    _inv(littools.lib_create, name="BareLib", scope="custom")
    # member added in BARE form (as the model echoes it back from lib_search)
    _inv(littools.lib_add, work_ids=["W3041272829"], library_id="barelib")

    out = _inv(littools.lib_search, query="kondo", library_id="barelib", k=8)
    assert "no matches" not in out.lower()
    assert "W3041272829" in out  # the member surfaces despite bare-vs-URL form


# ──────────────────────────────────────────────────────────────────────
# fetch_paper_abstract
# ──────────────────────────────────────────────────────────────────────

def test_fetch_paper_abstract_found(synth_index):
    out = _inv(littools.fetch_paper_abstract, work_id="W100")
    assert "W100" in out
    assert "Kondo" in out
    assert "first author: A" in out


def test_fetch_paper_abstract_unknown(synth_index):
    out = _inv(littools.fetch_paper_abstract, work_id="W_nope")
    assert "not found" in out.lower()


def test_fetch_paper_abstract_bare_id_for_urlkeyed_index(tmp_path, monkeypatch):
    """lib_search prints URL-form ids; the model echoes the
    BARE id into fetch_paper_abstract. The real 50k corpus keys rows by the URL
    form, so the tool must accept the bare id and still return the abstract —
    here with no abstracts.parquet it falls back to the index-aligned metadata
    abstract (lib_search's source) and flags that provenance."""
    idx = tmp_path / "literature_index"
    idx.mkdir(parents=True, exist_ok=True)
    np.save(idx / "vectors.npy", np.stack([_vec(0)]).astype(np.float32))
    pd.DataFrame([
        {"work_id": "https://openalex.org/W3041272829",
         "doi": "https://doi.org/10.1/z", "title": "URL-keyed paper",
         "year": 2019, "journal": "PRB", "cited": 7,
         "abstract": "Inline metadata abstract lib_search would show."},
    ]).to_parquet(idx / "metadata.parquet")
    # No abstracts.parquet → forces the index-metadata fallback path.
    _repoint_index(monkeypatch, idx)

    out = _inv(littools.fetch_paper_abstract, work_id="W3041272829")  # bare id
    assert "not found" not in out.lower()
    assert "Inline metadata abstract" in out
    assert "索引对齐" in out  # index_metadata provenance note surfaced


# ──────────────────────────────────────────────────────────────────────
# propose_citations
# ──────────────────────────────────────────────────────────────────────

def test_propose_citations_returns_candidates(synth_index):
    out = _inv(littools.propose_citations, section_text="Kondo effect on Au(111)", k=4)
    assert "candidates" in out
    assert "W100" in out
    assert "doi=" in out


def test_propose_citations_empty_text(synth_index):
    out = _inv(littools.propose_citations, section_text="   ", k=4)
    assert "empty section text" in out


# ──────────────────────────────────────────────────────────────────────
# literature_priors
# ──────────────────────────────────────────────────────────────────────

def test_literature_priors_returns_percentiles(synth_priors):
    out = _inv(littools.literature_priors, material="Au(111)", mode="imaging")
    assert "bias_v" in out
    assert "p50=0.1" in out
    # must be clearly flagged as non-authoritative
    assert "NOT authoritative" in out


def test_literature_priors_unknown_material(synth_priors):
    out = _inv(littools.literature_priors, material="Unobtanium")
    assert "no aggregated priors" in out


def test_literature_priors_requires_material(synth_priors):
    out = _inv(littools.literature_priors, material="")
    assert "required" in out.lower()


# ──────────────────────────────────────────────────────────────────────
# build_tools wiring + no cross-agent import
# ──────────────────────────────────────────────────────────────────────

def test_build_tools_includes_all_library_tools():
    tools = littools.build_tools(buf=None)
    names = {t.name for t in tools}
    expected = {
        "lib_list", "lib_create", "lib_switch", "lib_add", "lib_remove",
        "lib_search", "fetch_paper_abstract", "propose_citations", "literature_priors",
    }
    assert expected <= names, f"missing library tools: {expected - names}"


def test_library_tools_constant_count():
    # 9 P1 library/knowledge tools + 2 fetch-request-board tools (P5)
    # + lib_copy and save_literature_report (experiment-scoped libraries, 2026-07-29)
    # + deep_read_papers (parallel full-text reading, 2026-08-01)
    # + search_fulltext (per-paper chunk retrieval, 2026-08-01)
    assert len(littools.LIBRARY_TOOLS) == 15
    names = {t.name for t in littools.LIBRARY_TOOLS}
    assert {"request_fulltext", "list_fetch_requests"} <= names
    # save_literature_report is the literature agent's FIRST write-to-disk tool —
    # before it, every review it wrote lived only in the conversation stream.
    assert {"lib_copy", "save_literature_report"} <= names


def test_tools_module_imports_only_allowed_packages():
    """No cross-agent import: tools.py must not import another agent package."""
    src = Path(littools.__file__).read_text(encoding="utf-8")
    # crude but effective: no `from mast.agents.<other>` / `import mast.agents.<other>`
    import re
    bad = re.findall(r"(?:from|import)\s+mast\.agents\.(\w+)", src)
    allowed = {"_shared", "state", "literature"}
    offenders = [m for m in bad if m not in allowed]
    assert not offenders, f"cross-agent import detected: {offenders}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
