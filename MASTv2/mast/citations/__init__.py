"""Citation management: track which tools/papers to cite per experiment."""

from __future__ import annotations

from mast.citations.database import CITATION_DB, get_citations_for
from mast.citations.manager import CitationManager

__all__ = [
    "CITATION_DB",
    "CitationManager",
    "get_citations_for",
]
