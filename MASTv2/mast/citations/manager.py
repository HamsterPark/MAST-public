"""CitationManager: generate recommended citations based on actual experiment usage.

Like R's citation() but per-experiment — only cites what you actually used.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from mast.core.types import Citation, CitationType
from mast.citations.database import (
    CITATION_DB,
    MAST,
    NANONIS_SPM,
    NANONIS_HARDWARE,
    CLAUDE,
    SCIPY,
    SCIKIT_IMAGE,
)
from mast.logging.storage import ExperimentStorage


class CitationManager:
    """Query experiment history and generate context-appropriate citation lists."""

    def __init__(self, storage: ExperimentStorage):
        self._storage = storage

    # ── Per-experiment citations ───────────────────────────────────────

    def for_experiment(self, experiment_id: str) -> list[Citation]:
        """Collect citations for all skills/functions used in a specific experiment.

        Logic:
        1. Always include MAST itself
        2. If any Nanonis TCP call was made → include nanonis-spm + Nanonis hardware
        3. For each unique skill_name in actions → look up CITATION_DB
        4. If LLM planner was used (context contains 'MissionPlanner') → include Claude
        5. Deduplicate by citation key, preserving order
        """
        actions = self._storage.get_actions(experiment_id)
        if not actions:
            return [MAST]

        used_skills: set[str] = set()
        had_nanonis_calls = False
        had_llm = False

        for action in actions:
            used_skills.add(action.skill_name)
            # The TCP call list lives on the action OR inside its SkillResult
            # (skills fill `SkillResult.nanonis_calls`; the records layer copies
            # it up to the action). Checking only the action-level list made this
            # condition permanently false for the whole agent path — the recorder
            # dropped both copies, so 0 of 27 actions in a real session carried
            # any calls and MAST never cited Nanonis for a session that had run
            # hundreds of TCP commands ().
            if action.nanonis_calls or getattr(action.result, "nanonis_calls", None):
                had_nanonis_calls = True
            if action.context and "MissionPlanner" in action.context:
                had_llm = True
            # Also check approval_source for LLM involvement
            if action.approval_source == "llm":
                had_llm = True

        return self._collect(used_skills, had_nanonis_calls, had_llm)

    def for_skills(self, skill_names: list[str]) -> list[Citation]:
        """Get citations for a specific set of skill names (no experiment needed)."""
        had_nanonis = True  # Assume hardware was used if skills are specified
        return self._collect(set(skill_names), had_nanonis, had_llm=False)

    def for_all_registered(self) -> list[Citation]:
        """Get citations for ALL known skills/tools (full bibliography)."""
        all_names = set(CITATION_DB.keys()) - {"_mast", "_nanonis_spm", "_claude_llm"}
        return self._collect(all_names, had_nanonis_calls=True, had_llm=True)

    # ── Output Formats ────────────────────────────────────────────────

    def generate_bibtex(self, citations: list[Citation]) -> str:
        """Generate BibTeX file content from citation list."""
        return "\n\n".join(c.to_bibtex() for c in citations)

    def generate_text(self, citations: list[Citation]) -> str:
        """Generate numbered plain-text citation list."""
        lines = []
        for i, c in enumerate(citations, 1):
            lines.append(f"[{i}] {c.to_plaintext()}")
        return "\n".join(lines)

    def generate_markdown(self, citations: list[Citation]) -> str:
        """Generate Markdown citation section suitable for experiment reports."""
        # NOTE: only PAPER and SOFTWARE citations are produced by the citation
        # database — there are no DATASET entries. A "### Datasets" section used
        # to live here but was unreachable dead code; if a DATASET citation is
        # ever added to mast.citations.database, restore a rendering branch here
        # (test_no_dataset_citations_in_db guards this assumption).
        papers = [c for c in citations if c.citation_type == CitationType.PAPER]
        software = [c for c in citations if c.citation_type == CitationType.SOFTWARE]

        lines = ["## Recommended Citations", ""]

        if papers:
            lines.append("### Papers")
            lines.append("")
            for i, c in enumerate(papers, 1):
                line = f"{i}. "
                if c.authors:
                    line += f"{c.authors} "
                line += f'"{c.title}." '
                if c.journal:
                    line += f"*{c.journal}*"
                    if c.volume:
                        line += f" **{c.volume}**"
                    if c.pages:
                        line += f", {c.pages}"
                line += f" ({c.year})."
                if c.doi:
                    line += f" DOI: [{c.doi}](https://doi.org/{c.doi})"
                lines.append(line)
            lines.append("")

        if software:
            lines.append("### Software")
            lines.append("")
            for i, c in enumerate(software, 1):
                # Empty author (e.g. MAST has a TODO author) must not render
                # as a stray ". " before the title.
                line = f"{i}. "
                if c.authors:
                    line += f"{c.authors}. "
                line += f"*{c.title}*"
                if c.url:
                    line += f" [{c.url}]({c.url})"
                line += f" ({c.year})."
                lines.append(line)
            lines.append("")

        return "\n".join(lines)

    def generate_acknowledgment(self, citations: list[Citation]) -> str:
        """Generate a natural-language acknowledgment paragraph.

        Example output:
        "STM measurements were performed using the MAST framework [1] with a
        Nanonis V5e controller [2]. Tip quality was assessed using the DeepSPM
        CNN approach [3]. Bayesian optimization followed the method of
        Narasimha et al. [4]. Image drift was corrected using phase
        cross-correlation from scikit-image [5]."
        """
        # Reference markers must match the numbered list rendered below, so map
        # each citation key to its 1-based position. The previous version
        # emitted the raw BibTeX key (e.g. "[Narasimha_2024]") which did not
        # correspond to any "[n]" entry in the References list.
        index_of = {c.key: i for i, c in enumerate(citations, 1)}

        parts = []
        # Infrastructure
        infra = [c for c in citations if c.key in ("MAST_2026", "Nanonis_V5e", "nanonis_spm_2024")]
        if infra:
            parts.append(
                "STM measurements were performed using the MAST framework "
                "with a Nanonis V5e controller via the nanonis-spm Python interface"
            )

        # LLM
        if any(c.key == "Anthropic_Claude_2025" for c in citations):
            parts.append(
                "Experiment planning and execution were driven by Claude (Anthropic) "
                "through an agentic tool-use loop"
            )

        # Algorithm-specific
        algo_citations = [
            c for c in citations
            if c.citation_type == CitationType.PAPER
            and c.key not in ("MAST_2026",)
        ]
        for c in algo_citations:
            # Use the note field to describe what it was used for
            if c.note:
                lead = self._lead_author(c.authors)
                marker = index_of.get(c.key)
                ref = f" [{marker}]" if marker is not None else ""
                parts.append(f"{c.note} followed {lead}{ref}")

        if not parts:
            return ""

        text = ". ".join(parts) + "."
        # Append numbered reference list
        text += "\n\nReferences:\n"
        for i, c in enumerate(citations, 1):
            text += f"[{i}] {c.to_plaintext()}\n"

        return text

    @staticmethod
    def _lead_author(authors: str) -> str:
        """Return a natural-language lead-author phrase for an acknowledgment.

        Multi-author entries get "Surname et al."; single-author entries get
        just the surname (no spurious "et al."). The author string follows the
        BibTeX convention "Last, First and Last, First and ...". Empty author
        strings fall back to "the authors".
        """
        authors = (authors or "").strip()
        if not authors:
            return "the authors"
        # Split into individual authors on the BibTeX " and " separator.
        author_list = [a.strip() for a in authors.split(" and ") if a.strip()]
        first = author_list[0] if author_list else authors
        # Surname is the text before the first comma ("Krull, P." -> "Krull");
        # if there is no comma, use the whole token ("Anthropic").
        surname = first.split(",")[0].strip() or first
        # More than one author — or an explicit "others"/"et al." marker —
        # warrants "et al.".
        multi = (
            len(author_list) > 1
            or "others" in authors.lower()
            or "et al" in authors.lower()
        )
        return f"{surname} et al." if multi else surname

    # ── Report Integration ────────────────────────────────────────────

    def append_to_report(
        self, experiment_id: str, report_md: str, format: str = "markdown"
    ) -> str:
        """Append citation section to an existing Markdown experiment report."""
        citations = self.for_experiment(experiment_id)
        if format == "markdown":
            section = self.generate_markdown(citations)
        elif format == "bibtex":
            section = "## BibTeX\n\n```bibtex\n" + self.generate_bibtex(citations) + "\n```\n"
        else:
            section = "## References\n\n" + self.generate_text(citations) + "\n"
        return report_md.rstrip() + "\n\n---\n\n" + section

    def save_bibtex(self, experiment_id: str, output_path: str) -> str:
        """Generate and save BibTeX file for an experiment."""
        citations = self.for_experiment(experiment_id)
        bibtex = self.generate_bibtex(citations)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(bibtex, encoding="utf-8")
        return bibtex

    # ── Internal ──────────────────────────────────────────────────────

    def _collect(
        self,
        skill_names: set[str],
        had_nanonis_calls: bool,
        had_llm: bool,
    ) -> list[Citation]:
        """Collect and deduplicate citations based on what was used."""
        seen: OrderedDict[str, Citation] = OrderedDict()

        def _add(c: Citation) -> None:
            if c.key not in seen:
                seen[c.key] = c

        # 1. Always cite MAST
        _add(MAST)

        # 2. Nanonis infrastructure if hardware was touched
        if had_nanonis_calls:
            _add(NANONIS_SPM)
            _add(NANONIS_HARDWARE)

        # 3. LLM if used
        if had_llm:
            _add(CLAUDE)

        # 4. Per-skill citations
        for name in sorted(skill_names):
            for c in CITATION_DB.get(name, []):
                _add(c)

        return list(seen.values())
