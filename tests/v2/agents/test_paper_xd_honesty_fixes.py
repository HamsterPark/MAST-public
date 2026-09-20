"""Honesty-fix regression tests for the paper_writing / paper_review /
experiment_design agents.

All tests run offline — no LLM, no network. They exercise the real tool code
paths with fake inputs / tmp dirs and assert on docstring/prompt text.

  #58  paper_writing.draft_section returns an honest [FILL: …] scaffold (no
       silently-unfilled {placeholder} prose); lookup_citation is honest on a
       miss and never fabricates a reference.
  #104 paper_review/paper_writing/experiment_design graph docstrings no longer
       claim an Anthropic model (Opus/Sonnet) when the real default is Kimi.
  #105 experiment_design prompt references the REAL lookup_sample "No match
       found" output, not the phantom "not yet ported" branch.
  #106 paper_writing.embed_figure disambiguates same-named-but-different
       figures with a content hash instead of silently overwriting.
"""
from __future__ import annotations

# ── path bootstrap (force MASTv2 mast.* to win over any v1 copy) ───────
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
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

from mast.agents.paper_writing.tools import (  # noqa: E402
    draft_section,
    embed_figure,
    lookup_citation,
)
from mast.agents.paper_writing import graph as pw_graph  # noqa: E402
from mast.agents.paper_review import graph as pr_graph  # noqa: E402
from mast.agents.experiment_design import graph as xd_graph  # noqa: E402
from mast.agents.experiment_design.prompts import SYSTEM_PROMPT as XD_PROMPT  # noqa: E402
from mast.agents.experiment_design.tools import lookup_sample_tool  # noqa: E402


# ═════════════════════════════════════════════════════════════════════
# #58 — draft_section: honest fill-in scaffold, no unfilled {placeholders}
# ═════════════════════════════════════════════════════════════════════

class TestDraftSectionScaffold:
    _SECTIONS = ("introduction", "methods", "results", "discussion", "conclusion")

    def test_no_bare_brace_placeholders_remain(self):
        """The old templates emitted literal {topic}/{sample_prep}/... braces
        that were never substituted. The honest scaffold must NOT contain any
        such single-token brace placeholder."""
        import re
        brace_token = re.compile(r"\{[a-z_]+\}")
        for sec in self._SECTIONS:
            out = draft_section.invoke({"section": sec, "context": "ctx"})
            leftover = brace_token.findall(out)
            assert not leftover, (
                f"{sec}: unfilled brace placeholders leaked into output: {leftover}"
            )

    def test_emits_visible_fill_markers(self):
        """Every section must expose explicit [FILL: …] slots so the writer
        (and a human reader) can see what is still missing."""
        for sec in self._SECTIONS:
            out = draft_section.invoke({"section": sec, "context": "ctx"})
            assert "[FILL:" in out, f"{sec}: no [FILL: …] markers in scaffold"

    def test_marks_itself_as_scaffold_only(self):
        out = draft_section.invoke({"section": "methods", "context": "ctx"})
        assert "SCAFFOLD ONLY" in out

    def test_section_heading_present(self):
        # Backwards-compatible: still emits a "## <Section>" heading.
        assert "## Methods" in draft_section.invoke(
            {"section": "methods", "context": "x"}
        )
        assert "## Results" in draft_section.invoke(
            {"section": "results", "context": "x"}
        )

    def test_context_appended_verbatim(self):
        for sec in self._SECTIONS:
            out = draft_section.invoke(
                {"section": sec, "context": f"unique-ctx-{sec}"}
            )
            assert f"unique-ctx-{sec}" in out

    def test_unknown_section_is_honest_error(self):
        out = draft_section.invoke({"section": "appendix", "context": "x"})
        assert "unknown section" in out.lower()

    def test_makes_no_llm_call(self):
        """draft_section must be pure (no network / model). We assert it runs
        with no API keys present by simply calling it — if it tried to reach a
        model it would error. (Belt-and-braces: the tool body has no model
        import.)"""
        out = draft_section.invoke({"section": "introduction", "context": "ok"})
        assert "## Introduction" in out


# ═════════════════════════════════════════════════════════════════════
# #58 — lookup_citation: honest on a miss, never fabricates
# ═════════════════════════════════════════════════════════════════════

class TestLookupCitationHonesty:
    def test_known_key_formats_reference(self):
        out = lookup_citation.invoke({"key": "krull2020"})
        assert "Krull" in out and "2020" in out

    def test_unknown_key_is_honest_and_lists_known(self):
        out = lookup_citation.invoke({"key": "totally-made-up-2099"})
        low = out.lower()
        assert "not found" in low
        assert "curated" in low
        # lists at least one real known key
        assert "smalley2024" in out
        # explicitly refuses to invent
        assert "do not invent" in low

    def test_unknown_key_does_not_emit_fake_acs_line(self):
        """A miss must not return an ACS-style 'Author (year) "title."' line."""
        out = lookup_citation.invoke({"key": "ghost2030"})
        # An ACS line contains a quoted title; the honest-miss note does not.
        assert '."' not in out


# ═════════════════════════════════════════════════════════════════════
# #106 — embed_figure: no silent overwrite on same-name-different-bytes
# ═════════════════════════════════════════════════════════════════════

_PNG_A = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\xdacd\xf8\xcf\x00\x00\x00\x03\x00\x01\xb1\xc1\xa6\x91"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
# Different content (different IDAT payload byte) → different sha256.
_PNG_B = _PNG_A.replace(b"\xcf\x00\x00\x00\x03", b"\xcf\x00\x00\x00\x04")


@pytest.fixture
def figures_env(tmp_path, monkeypatch, documents_root):
    """embed_figure's destination is the EXPERIMENT's figure pool, not the global
    ``data/figures`` (2026-07-29). With no active experiment that is
    ``_unfiled/documents/_assets/`` — the fail-open落点, so a figure is never
    dropped for lack of a scope.

    ``MAST_FIGURES_DIR`` is still set because it is the SOURCE pool ``list_figures``
    enumerates; it is no longer where anything is written."""
    from mast.documents.paths import unfiled_docs_dir

    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    return tmp_path, unfiled_docs_dir(create=True) / "_assets"


def _src(tmp_path: Path, name: str, data: bytes) -> str:
    d = tmp_path / "src"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_bytes(data)
    return str(p)


class TestEmbedFigureCollision:
    def test_the_link_is_relative_to_the_document_not_the_repo(self, figures_env):
        """The link is resolved by whatever renders the .md, against the FILE's own
        directory. Emitting ``../figures/x.png`` only worked for a draft sitting in
        ``data/drafts/``; a document lives in ``<exp>/reports/<doc-dir>/`` and its
        pool one level up, so the correct link is ``../_assets/x.png`` — and it
        keeps resolving after the experiment folder is moved to another machine."""
        tmp_path, _figdir = figures_env
        out = embed_figure.invoke({"scan_path": _src(tmp_path, "topo.png", _PNG_A),
                                   "caption": "形貌"})
        link = out.split("](")[1].split(")")[0]
        assert link == "../_assets/topo.png", f"broken figure link: {link}"

    def test_same_name_different_bytes_does_not_overwrite(self, figures_env):
        tmp_path, figdir = figures_env
        src1 = _src(tmp_path, "fig.png", _PNG_A)
        out1 = embed_figure.invoke({"scan_path": src1, "caption": "A"})
        assert "fig.png" in out1

        # Second source: SAME name, DIFFERENT bytes, from a different dir.
        src2_dir = tmp_path / "other"
        src2_dir.mkdir()
        src2 = src2_dir / "fig.png"
        src2.write_bytes(_PNG_B)
        out2 = embed_figure.invoke({"scan_path": str(src2), "caption": "B"})

        # Original copy must still exist with original bytes.
        assert (figdir / "fig.png").read_bytes() == _PNG_A, (
            "embed_figure silently overwrote the existing figure!"
        )
        # Second figure must have been written under a hash-disambiguated name.
        assert "fig.png" not in out2.split("](")[1].split(")")[0] or \
            "fig." in out2  # the embed line points at the hashed file
        copies = sorted(p.name for p in figdir.iterdir())
        assert len(copies) == 2, f"expected 2 distinct figure files, got {copies}"
        # The hashed file holds the B bytes.
        hashed = [p for p in figdir.iterdir() if p.name != "fig.png"]
        assert len(hashed) == 1
        assert hashed[0].read_bytes() == _PNG_B
        # Output names the disambiguation honestly.
        assert "avoid overwriting" in out2.lower()

    def test_same_name_same_bytes_is_reused_not_duplicated(self, figures_env):
        tmp_path, figdir = figures_env
        src = _src(tmp_path, "topo.png", _PNG_A)
        embed_figure.invoke({"scan_path": src, "caption": "first"})
        out2 = embed_figure.invoke({"scan_path": src, "caption": "second"})
        # Only one file in the figures dir — identical bytes reused.
        copies = [p for p in figdir.iterdir() if p.is_file()]
        assert len(copies) == 1, f"identical figure duplicated: {copies}"
        assert "reusing" in out2.lower() or "already present" in out2.lower()

    def test_distinct_names_coexist(self, figures_env):
        tmp_path, figdir = figures_env
        a = _src(tmp_path, "a.png", _PNG_A)
        b = _src(tmp_path, "b.png", _PNG_B)
        embed_figure.invoke({"scan_path": a, "caption": "a"})
        embed_figure.invoke({"scan_path": b, "caption": "b"})
        names = sorted(p.name for p in figdir.iterdir())
        assert names == ["a.png", "b.png"]

    def test_missing_source_still_errors(self, figures_env):
        out = embed_figure.invoke(
            {"scan_path": "/no/such/file.png", "caption": "x"}
        )
        assert "source file not found" in out.lower()


# ═════════════════════════════════════════════════════════════════════
# #104 — graph docstrings no longer claim an Anthropic model by default
# ═════════════════════════════════════════════════════════════════════

class TestGraphDocstringsHonest:
    @pytest.mark.parametrize(
        "mod",
        [pr_graph, pw_graph, xd_graph],
        ids=["paper_review", "paper_writing", "experiment_design"],
    )
    def test_module_docstring_does_not_claim_anthropic_default(self, mod):
        doc = (mod.__doc__ or "")
        build_doc = (mod.build.__doc__ or "")
        text = doc + "\n" + build_doc
        # Must not assert the default IS Opus/Sonnet (the real default is Kimi K3).
        for bad in ("(Opus 4.7)", "(Sonnet 4.6)"):
            assert bad not in text, (
                f"{mod.__name__}: docstring still claims default model {bad}, "
                "but AGENT_MODEL defaults to Kimi K3"
            )
        # And it should honestly name Kimi K3 as the default.
        assert "Kimi K3" in text, (
            f"{mod.__name__}: docstring should name Kimi K3 as the default model"
        )

    def test_agent_model_really_is_kimi(self):
        from mast.agents._shared.models import AGENT_MODEL, KIMI_K3
        for agent in ("paper_review", "paper_writing", "experiment_design"):
            assert AGENT_MODEL[agent] == KIMI_K3


# ═════════════════════════════════════════════════════════════════════
# #105 — XD prompt aligns with the REAL lookup_sample output
# ═════════════════════════════════════════════════════════════════════

class TestExperimentDesignPromptAlignment:
    def test_prompt_does_not_reference_phantom_branch(self):
        assert "not yet ported" not in XD_PROMPT, (
            "XD prompt references a 'not yet ported' lookup_sample branch that "
            "the real tool never emits"
        )

    def test_prompt_references_real_miss_string(self):
        assert "No match found" in XD_PROMPT, (
            "XD prompt should key off the real lookup_sample miss text "
            "('No match found')"
        )

    def test_real_lookup_sample_miss_matches_prompt_wording(self):
        """Ground truth: the tool's miss output contains the exact phrase the
        prompt now tells the agent to look for."""
        lt = lookup_sample_tool()
        out = lt.invoke({"query": "unobtainium-ZZZ-not-a-real-material"})
        assert "No match found" in out
        # The phantom phrase must NOT appear in the real output either.
        assert "not yet ported" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v", "-p", "no:randomly"])
