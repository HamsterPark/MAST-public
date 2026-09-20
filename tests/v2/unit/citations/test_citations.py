"""v2 unit tests for mast.citations.

Ported from tests/unit/test_citations.py — 19 test functions covering:
  - Citation data model (BibTeX/plaintext rendering)
  - Database lookups (get_citations_for)
  - Per-experiment citation collection (CitationManager.for_experiment)
  - Output formats (BibTeX / text / Markdown / file save / report append)
  - for_skills (no experiment needed)

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/citations/test_citations.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import tempfile

import pytest

from mast.core.types import (
    ActionRecord,
    Citation,
    CitationType,
    NanonisCallRecord,
    SkillResult,
)
from mast.citations.database import (
    CITATION_DB,
    MAST,
    NANONIS_SPM,
    DEEPSPM,
    BO_AUTOSTM,
    CLAUDE,
    SCIKIT_IMAGE,
    get_citations_for,
)
from mast.citations.manager import CitationManager
from mast.logging.storage import ExperimentStorage


@pytest.fixture
def storage(tmp_path):
    return ExperimentStorage(tmp_path / "test_citations.db")


@pytest.fixture
def manager(storage):
    return CitationManager(storage)


# ── Citation data model ───────────────────────────────────────────────

def test_citation_to_bibtex():
    bib = DEEPSPM.to_bibtex()
    assert "@article{DeepSPM_2020," in bib
    assert "Krull" in bib
    assert "2020" in bib
    assert "Communications Physics" in bib


def test_citation_to_bibtex_software():
    bib = MAST.to_bibtex()
    assert "@software{MAST_2026," in bib


def test_citation_to_plaintext():
    text = DEEPSPM.to_plaintext()
    assert "Krull" in text
    assert "2020" in text
    assert "Communications Physics" in text


# ── Database lookups ──────────────────────────────────────────────────

def test_get_citations_for_known_skill():
    cites = get_citations_for("AssessTipFromImage_VGG")
    assert len(cites) == 1
    assert cites[0].key == "DeepSPM_2020"


def test_get_citations_for_builtin_skill():
    """Basic builtin skills have no algorithm-specific citations."""
    cites = get_citations_for("GetBias")
    assert cites == []


def test_get_citations_for_unknown():
    cites = get_citations_for("NonExistentSkill")
    assert cites == []


def test_multi_citation_skill():
    """BondSelectiveReaction should cite both papers."""
    cites = get_citations_for("BondSelectiveReaction")
    assert len(cites) == 2


# ── Per-experiment citation collection ────────────────────────────────

def test_empty_experiment_cites_mast_only(manager, storage):
    """An experiment with no actions should still cite MAST."""
    exp_id = storage.create_experiment("empty test")
    citations = manager.for_experiment(exp_id)
    assert len(citations) == 1
    assert citations[0].key == "MAST_2026"


def test_basic_experiment_cites_infrastructure(manager, storage):
    """Basic skills with Nanonis calls → MAST + nanonis-spm + hardware."""
    exp_id = storage.create_experiment("basic scan")
    action = ActionRecord(
        experiment_id=exp_id,
        skill_name="GetBias",
        nanonis_calls=[NanonisCallRecord(method="Bias_Get")],
    )
    storage.log_action(action)

    citations = manager.for_experiment(exp_id)
    keys = [c.key for c in citations]
    assert "MAST_2026" in keys
    assert "nanonis_spm_2024" in keys
    assert "Nanonis_V5e" in keys
    # No algorithm-specific citations for GetBias
    assert len(citations) == 3


def test_experiment_with_advanced_skill(manager, storage):
    """Using BO optimization → should cite BO-for-AutoSTM paper."""
    exp_id = storage.create_experiment("optimize resolution")
    storage.log_action(ActionRecord(
        experiment_id=exp_id,
        skill_name="OptimizeResolution_BO",
        nanonis_calls=[NanonisCallRecord(method="Bias_Set")],
    ))
    citations = manager.for_experiment(exp_id)
    keys = [c.key for c in citations]
    assert "Narasimha_2024" in keys
    assert "MAST_2026" in keys


def test_experiment_with_llm(manager, storage):
    """LLM-driven experiment → should cite Claude."""
    exp_id = storage.create_experiment("llm test")
    storage.log_action(ActionRecord(
        experiment_id=exp_id,
        skill_name="SetBias",
        context="MissionPlanner: set bias to -0.5V",
        nanonis_calls=[NanonisCallRecord(method="Bias_Set")],
    ))
    citations = manager.for_experiment(exp_id)
    keys = [c.key for c in citations]
    assert "Anthropic_Claude_2025" in keys


def test_experiment_deduplication(manager, storage):
    """Multiple uses of same skill → citation appears only once."""
    exp_id = storage.create_experiment("repeated scans")
    for _ in range(5):
        storage.log_action(ActionRecord(
            experiment_id=exp_id,
            skill_name="AssessTipFromImage_VGG",
            nanonis_calls=[NanonisCallRecord(method="Scan_Action")],
        ))
    citations = manager.for_experiment(exp_id)
    keys = [c.key for c in citations]
    assert keys.count("DeepSPM_2020") == 1


def test_mixed_experiment(manager, storage):
    """Realistic mixed experiment: scan + BO + DeepSPM tip assessment."""
    exp_id = storage.create_experiment("Au(111) atomic resolution")
    # Basic scan
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="StartScan",
        nanonis_calls=[NanonisCallRecord(method="Scan_Action")],
    ))
    # Tip assessment
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="AssessTipFromImage_VGG",
        nanonis_calls=[NanonisCallRecord(method="Scan_Action")],
    ))
    # Tip conditioning
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="ConditionTip_RuleBased",
        nanonis_calls=[NanonisCallRecord(method="Bias_Set")],
    ))
    # BO optimization
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="OptimizeResolution_BO",
        nanonis_calls=[NanonisCallRecord(method="Bias_Set")],
    ))

    citations = manager.for_experiment(exp_id)
    keys = [c.key for c in citations]

    # Should have: MAST, nanonis-spm, nanonis hardware, DeepSPM, Scanbot, BO-for-AutoSTM
    assert "MAST_2026" in keys
    assert "DeepSPM_2020" in keys
    assert "Scanbot_2024" in keys
    assert "Narasimha_2024" in keys
    # Should NOT have unrelated citations
    assert "Anthropic_Claude_2025" not in keys  # no LLM used
    assert "Chen_2022" not in keys  # no atom manipulation


# ── Output formats ────────────────────────────────────────────────────

def test_generate_bibtex(manager, storage):
    exp_id = storage.create_experiment("bibtex test")
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="OptimizeResolution_BO",
        nanonis_calls=[NanonisCallRecord(method="Bias_Set")],
    ))
    citations = manager.for_experiment(exp_id)
    bibtex = manager.generate_bibtex(citations)
    assert "@software{MAST_2026," in bibtex
    assert "@article{Narasimha_2024," in bibtex
    # Each entry should be complete
    assert bibtex.count("@") >= 3  # MAST + nanonis-spm + Nanonis + BO


def test_generate_text(manager, storage):
    exp_id = storage.create_experiment("text test")
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="GetBias",
        nanonis_calls=[NanonisCallRecord(method="Bias_Get")],
    ))
    citations = manager.for_experiment(exp_id)
    text = manager.generate_text(citations)
    assert "[1]" in text
    assert "MAST" in text


def test_generate_markdown(manager, storage):
    exp_id = storage.create_experiment("md test")
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="AssessTipFromImage_VGG",
        nanonis_calls=[NanonisCallRecord(method="Scan_Action")],
    ))
    citations = manager.for_experiment(exp_id)
    md = manager.generate_markdown(citations)
    assert "## Recommended Citations" in md
    assert "### Papers" in md
    assert "### Software" in md
    assert "Krull" in md


def test_save_bibtex_file(manager, storage, tmp_path):
    exp_id = storage.create_experiment("save test")
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="GetBias",
        nanonis_calls=[NanonisCallRecord(method="Bias_Get")],
    ))
    out = tmp_path / "refs.bib"
    manager.save_bibtex(exp_id, str(out))
    assert out.exists()
    content = out.read_text()
    assert "MAST_2026" in content


def test_append_to_report(manager, storage):
    exp_id = storage.create_experiment("report test")
    storage.log_action(ActionRecord(
        experiment_id=exp_id, skill_name="GP_AdaptiveSTS",
        nanonis_calls=[NanonisCallRecord(method="Bias_SpectrStart")],
    ))
    report = "# My Experiment\n\nSome results here."
    updated = manager.append_to_report(exp_id, report)
    assert "# My Experiment" in updated
    assert "## Recommended Citations" in updated
    assert "Thomas" in updated  # gpSTS author


# ── for_skills (no experiment needed) ─────────────────────────────────

def test_for_skills():
    storage = ExperimentStorage(Path(tempfile.mkdtemp()) / "test.db")
    manager = CitationManager(storage)
    citations = manager.for_skills(["AssessTipFromImage_VGG", "ConditionTip_DQN"])
    keys = [c.key for c in citations]
    # Both cite DeepSPM — should appear only once
    assert keys.count("DeepSPM_2020") == 1
    assert "MAST_2026" in keys


# ── Regression: empty-author rendering (finding #108) ─────────────────

def test_empty_author_plaintext_no_stray_dot():
    """MAST has authors="" — to_plaintext() must not start with a bare '. '."""
    text = MAST.to_plaintext()
    assert not text.startswith(". ")
    assert not text.startswith(".")
    # The title (quoted) should lead instead.
    assert text.lstrip().startswith('"')


def test_empty_author_markdown_software_no_stray_dot():
    """Software entry with empty author must not render '1. . *Title*'."""
    storage = ExperimentStorage(Path(tempfile.mkdtemp()) / "t.db")
    manager = CitationManager(storage)
    # MAST (empty author, SOFTWARE) is always collected.
    md = manager.generate_markdown([MAST])
    assert ". . *" not in md
    assert "1. *MAST" in md  # number, then straight into the italic title


def test_nonempty_author_still_rendered():
    """Guard must not drop real authors."""
    text = DEEPSPM.to_plaintext()
    assert text.startswith("Krull")


# ── Regression: acknowledgment numbering + et al. (finding #109) ──────

def test_acknowledgment_uses_numeric_markers_not_bibtex_keys():
    """Inline reference markers must be [n] matching the numbered list,
    never the raw BibTeX key like [Narasimha_2024]."""
    storage = ExperimentStorage(Path(tempfile.mkdtemp()) / "t.db")
    manager = CitationManager(storage)
    # BO_AUTOSTM is a multi-author PAPER with a note → produces an algo line.
    cites = [MAST, NANONIS_SPM, BO_AUTOSTM]
    ack = manager.generate_acknowledgment(cites)
    assert ack  # non-empty
    # Raw key must not appear as an inline marker.
    assert "[Narasimha_2024]" not in ack
    # The numeric index of BO_AUTOSTM in the list (position 3) must appear.
    idx = cites.index(BO_AUTOSTM) + 1
    assert f"[{idx}]" in ack
    # References list is still numbered.
    assert "References:" in ack
    assert "[1]" in ack


def test_acknowledgment_marker_matches_reference_list():
    """The [n] inline marker must point at the right entry in References."""
    storage = ExperimentStorage(Path(tempfile.mkdtemp()) / "t.db")
    manager = CitationManager(storage)
    cites = [MAST, NANONIS_SPM, BO_AUTOSTM]
    ack = manager.generate_acknowledgment(cites)
    idx = cites.index(BO_AUTOSTM) + 1
    # The narrative cites [idx]; the References block lists that same number
    # with the BO paper's lead author.
    assert f"[{idx}]" in ack
    assert f"[{idx}] Narasimha" in ack


def test_acknowledgment_lead_author_et_al_for_multi():
    """Multi-author papers get 'Surname et al.'."""
    assert CitationManager._lead_author(BO_AUTOSTM.authors) == "Narasimha et al."
    assert CitationManager._lead_author(DEEPSPM.authors) == "Krull et al."


def test_acknowledgment_lead_author_no_et_al_for_single():
    """A single-author entry must NOT get a spurious 'et al.'."""
    assert CitationManager._lead_author("Smith, J.") == "Smith"
    assert CitationManager._lead_author("Anthropic") == "Anthropic"
    # 'others' marker counts as multi even without an explicit second name.
    assert CitationManager._lead_author("Ziatdinov, M. and others") == "Ziatdinov et al."


def test_acknowledgment_lead_author_empty():
    assert CitationManager._lead_author("") == "the authors"


# ── Regression: BibTeX LaTeX escaping (finding #110) ─────────────────

def test_bibtex_escapes_special_chars():
    """& % # _ in free-text fields must be backslash-escaped for LaTeX."""
    c = Citation(
        key="Test_2026",
        authors="Smith, J. & Doe, A.",
        title="Coverage 50% with C_2 #defects",
        year=2026,
        journal="Phys & Chem",
        note="100% done",
    )
    bib = c.to_bibtex()
    assert r"\&" in bib
    assert r"\%" in bib
    assert r"\#" in bib
    assert r"\_" in bib
    # The raw unescaped forms must not survive in those fields.
    assert "J. & Doe" not in bib
    assert "50% with" not in bib
    assert "C_2" not in bib


def test_bibtex_preserves_intentional_latex_accents():
    """Backslash accents in curated entries must NOT be mangled."""
    c = Citation(
        key="Accent_2026",
        authors=r'Jestil\"a, J. S.',
        title="On-surface synthesis",
        year=2026,
    )
    bib = c.to_bibtex()
    assert r'Jestil\"a' in bib  # backslash + quote untouched


def test_bibtex_does_not_escape_url_or_doi_underscores():
    """URLs/DOIs contain underscores that must stay literal (links/ids)."""
    c = Citation(
        key="UrlTest_2026",
        authors="Doe, J.",
        title="Title",
        year=2026,
        citation_type=CitationType.SOFTWARE,
        url="https://example.com/some_path_here",
        doi="10.1000/some_doi_2026",
    )
    bib = c.to_bibtex()
    assert "https://example.com/some_path_here" in bib
    assert "10.1000/some_doi_2026" in bib


def test_existing_db_bibtex_unchanged_for_plain_entries():
    """Entries with no special chars must render identically (no over-escaping)."""
    bib = DEEPSPM.to_bibtex()
    # DeepSPM has hyphenated pages '54' and no &/%/#/_ in text fields.
    assert "\\&" not in bib
    assert "\\%" not in bib
    assert "\\_" not in bib
    assert "Communications Physics" in bib


# ── Regression: no DATASET dead branch (finding #111) ─────────────────

def test_no_dataset_citations_in_db():
    """Guards the removal of the unreachable '### Datasets' markdown branch:
    if a DATASET citation is ever added, this fails so the renderer is fixed."""
    for name, cites in CITATION_DB.items():
        for c in cites:
            assert c.citation_type != CitationType.DATASET, (
                f"{name} -> {c.key} is a DATASET; restore the Datasets render branch"
            )


def test_markdown_has_no_empty_dataset_section():
    """generate_markdown must never emit a stray '### Datasets' header."""
    storage = ExperimentStorage(Path(tempfile.mkdtemp()) / "t.db")
    manager = CitationManager(storage)
    md = manager.generate_markdown([MAST, DEEPSPM])
    assert "### Datasets" not in md
    # Sanity: the live sections are still present.
    assert "### Papers" in md
    assert "### Software" in md
