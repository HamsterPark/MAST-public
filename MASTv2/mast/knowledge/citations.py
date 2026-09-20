"""D: paper_writing citation suggestion + paper_review novelty check.

Built on top of mast.knowledge.literature_index (the 36,792-paper embedding
index). v1 and v2 both call the same module (file is identical in
mast/knowledge/ and MASTv2/mast/knowledge/).

Two operations:

  * `propose_citations(section_text, k=10)` — given a draft section / claim,
    return top-k literature matches as candidate citations.

  * `check_priority_claim(claim, year)` — given a "first-to-do" claim
    written in year *year*, return any prior matching papers from the corpus.
    Useful for paper_review novelty checks.
"""

from __future__ import annotations

import logging
from typing import Any

from mast.knowledge.literature_index import search

logger = logging.getLogger(__name__)


def propose_citations(
    section_text: str,
    k: int = 10,
    *,
    year_min: int = 0,
    year_max: int = 9999,
) -> list[dict]:
    """Recall top-*k* literature candidates for citing in *section_text*.

    Each result includes BibTeX-friendly fields:
        {"title", "year", "journal", "doi", "work_id", "score", "cited",
         "bibtex_key"}

    The bibtex_key is auto-generated as
        firstauthor_year (e.g. "binnig_1982")
    or "noauthor_year_XXXX" if no author info is available.
    """
    if not section_text or not section_text.strip():
        return []

    hits = search(section_text, k=k, year_min=year_min, year_max=year_max)
    out: list[dict] = []
    for h in hits:
        # Build a bibtex-style key from year + first 6 chars of title
        title = (h.get("title") or "").strip()
        year = h.get("year") or 0
        first_word = "".join(c for c in title.split(" ", 1)[0].lower() if c.isalnum())
        key = f"{first_word[:8]}_{year}" if first_word else f"paper_{year}"
        out.append({
            **h,
            "bibtex_key": key,
        })
    return out


def check_priority_claim(
    claim: str,
    year: int,
    *,
    k: int = 10,
    similarity_threshold: float = 0.55,
) -> dict[str, Any]:
    """Verify a "first-to-do" claim against the literature corpus.

    Args:
        claim: prose describing the contribution (e.g. "we report the first
               STM observation of single-atom Kondo splitting on Pt(111)")
        year:  publication year being claimed
        k:     how many candidates to consider
        similarity_threshold: prior papers below this similarity are not
                              treated as challenges to the claim

    Returns:
        {"verdict": "novel" | "potentially_disputed" | "indeterminate",
         "challenger_count": int,
         "challengers": [...]}
    """
    if not claim or not claim.strip() or year < 1980:
        return {"verdict": "indeterminate", "challenger_count": 0, "challengers": []}

    # Look at papers strictly older than the claim year
    hits = search(claim, k=k * 2, year_max=year - 1)

    # `search` returns TWO incompatible `score` semantics:
    #   * semantic mode  → cosine similarity in ~[-1, 1] (comparable to the
    #     cosine `similarity_threshold`);
    #   * keyword fallback (DashScope key missing / network down) → a weighted
    #     term-FREQUENCY COUNT (title hit ×2 + abstract hit ×1), which is an
    #     unbounded count, NOT a cosine. Such hits carry retrieval="keyword" /
    #     degraded=True. Comparing a count of 2.0 against a 0.55 cosine threshold
    #     would flag every keyword hit as a challenger and wrongly dispute a
    #     genuinely novel claim. We cannot make a similarity-threshold novelty
    #     judgment without semantic scores → report indeterminate honestly.
    if any(h.get("retrieval") == "keyword" or h.get("degraded") for h in hits):
        return {
            "verdict": "indeterminate",
            "challenger_count": 0,
            "challengers": [],
            "degraded": True,
            "retrieval": "keyword",
            "note": (
                "语义检索不可用（嵌入服务缺失/网络故障），仅有关键词回退结果，"
                "其分数为词频计数而非余弦相似度，无法据相似度阈值判定优先权；"
                "结果不确定，请在语义检索恢复后重试。"
            ),
        }

    challengers = [h for h in hits if h.get("score", 0.0) >= similarity_threshold][:k]

    if not challengers:
        return {
            "verdict": "novel",
            "challenger_count": 0,
            "challengers": [],
            "note": "No prior matches above threshold in the corpus.",
        }

    return {
        "verdict": "potentially_disputed",
        "challenger_count": len(challengers),
        "challengers": challengers,
        "note": (
            f"Found {len(challengers)} prior paper(s) before {year} matching "
            f"the claim above similarity threshold {similarity_threshold:.2f}."
        ),
    }


__all__ = ["propose_citations", "check_priority_claim"]
