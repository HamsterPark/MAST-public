"""Extract ~3000 candidate STM terms from the OpenAlex corpus.

Source: 36,830 cleaned papers (stm_papers.parquet) — fields used:
    title, abstract, concepts (OpenAlex auto-tags), keywords (author keywords)

Strategy (multi-source, then dedupe):
    1. Author keywords        — all of them, frequency-weighted (highest signal)
    2. OpenAlex concepts      — all distinct concept display_names (mid signal)
    3. Title + abstract n-grams (1-4) with simple stopword filter and a
       minimum-frequency cutoff. We do NOT do TF-IDF against a generic corpus
       (overhead high; the keyword + concept channels already give us a strong
       domain prior). The n-gram channel is just for filling gaps.

Output: candidate_terms.jsonl, one record per line:
    {"canonical_en": "...", "freq": int, "channels": ["keyword","concept","ngram"],
     "example_sentence": "..."}

Sorted by frequency descending. The downstream A1 step splits this list into
50 chunks of ~60 terms each for parallel sonnet enrichment.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Bootstrap path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mast.knowledge.openalex_loader import load_cleaned  # noqa: E402

# ── Output target ─────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "openalex_pipeline"
OUT_PATH = OUT_DIR / "candidate_terms.jsonl"

# ── Tunables ──────────────────────────────────────────────────────────
TARGET_TERM_COUNT = 3000          # final list cap
MIN_NGRAM_FREQ = 30               # n-gram must appear in ≥ N papers
MIN_NGRAM_LEN = 3                 # token char minimum
MAX_NGRAM_LEN = 4                 # up to 4-grams

# Stop-list — keep short; the goal is NOT to cleanse for NLP, just to drop
# obvious non-terms. STM-specific terms like "tip" / "scan" are intentionally
# kept; they are useful canonical English forms.
_STOP = frozenset("""
the a an of in on at to for from by with as is are was were be been being
this that these those it its we us our their his her he she
which who whom whose what
and or but if not no nor so than then thus
all any some many much several few each every both either neither one two three four five
new used using uses use shown show showed showing
also however thus therefore moreover hence furthermore additionally
based recently here there where when how why
results result observed observe found find finding study studies investigation investigate
present presented presenting demonstrated demonstrate report reports reported reporting
high low large small great greater less wider width height range
data figure figures table tables work paper article articles
have has had having do does did doing does doing
can could may might shall should will would must
between within without through throughout into onto upon
been being above below over under near around about
much more most least less few many several
also still yet just only even already
different various certain particular specific general common typical
allow allows allowed allowing show shows showed showing reveal reveals revealed
described describe describing
made make makes making take takes took taken taking
provide provides provided providing
recent recently early earlier later late
case cases part parts type types kind kinds
single multi multiple two three four five six seven eight nine ten
respect respective respectively
form forms formation
nature natural
view views point points
follow follows following followed
real reals
discuss discusses discussing discussed
suggest suggests suggesting suggested
indicate indicates indicating indicated
allow allows allowing allowed
contain contains containing contained
include includes including included
require requires requiring required
""".split())

_PURE_NUM = re.compile(r"^[0-9.,\-+_/×()]+$")
_HAS_LETTER = re.compile(r"[a-zA-Z]")


# ── Tokeniser ─────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase tokens; keep chemical-formula friendly chars."""
    if not text:
        return []
    # Keep alphanum + parens + dashes for terms like Au(111), Bi2Te3, dI/dV.
    raw = re.findall(r"[A-Za-z][A-Za-z0-9\-]*\(\d{2,}\)|[A-Za-z][A-Za-z0-9\-]*", text)
    out: list[str] = []
    for tok in raw:
        # Strip trailing punctuation and lowercase
        t = tok.strip(".,;:()[]{}").lower()
        if not t or len(t) < 2:
            continue
        if _PURE_NUM.match(t):
            continue
        if not _HAS_LETTER.search(t):
            continue
        out.append(t)
    return out


def is_meaningful_ngram(words: tuple[str, ...]) -> bool:
    """Reject obvious garbage n-grams."""
    if any(w in _STOP for w in words):
        return False
    # No edge-only-stop checks needed — we already drop stops outright.
    if any(len(w) < MIN_NGRAM_LEN for w in words):
        return False
    # Reject if the whole gram is just numbers / single chars
    if all(_PURE_NUM.match(w) for w in words):
        return False
    return True


# ── Main extraction ───────────────────────────────────────────────────

def main() -> int:
    print("Loading stm_papers.parquet …", flush=True)
    df = load_cleaned()
    print(f"  rows: {len(df):,}", flush=True)

    keyword_counter: Counter = Counter()
    concept_counter: Counter = Counter()
    ngram_counter: Counter = Counter()
    examples: dict[str, str] = {}

    print("Streaming through papers …", flush=True)
    for i, row in enumerate(df.itertuples(index=False)):
        if i % 5000 == 0:
            print(f"  {i:,} / {len(df):,}", flush=True)

        _title = getattr(row, "title", "")
        _abstract = getattr(row, "abstract", "")
        title = (str(_title) if _title is not None else "").strip()
        abstract = (str(_abstract) if _abstract is not None else "").strip()
        # NaN coerced to "nan" — drop it
        if title == "nan":
            title = ""
        if abstract == "nan":
            abstract = ""

        # Channel 1: author keywords. Stored as a single string with ';' separator.
        kw_raw = getattr(row, "keywords", "") or ""
        for kw in str(kw_raw).split(";"):
            s = kw.strip()
            if not s or len(s) < 3 or s.lower() == "nan":
                continue
            keyword_counter[s] += 1
            examples.setdefault(s, title or abstract[:200])

        # Channel 2: OpenAlex concepts (also ';'-separated string).
        cn_raw = getattr(row, "concepts", "") or ""
        for cn in str(cn_raw).split(";"):
            s = cn.strip()
            if not s or len(s) < 3 or s.lower() == "nan":
                continue
            concept_counter[s] += 1
            examples.setdefault(s, title or abstract[:200])

        # Channel 3: n-grams from title + abstract (combined text)
        text = f"{title} {abstract}"
        toks = tokenize(text)
        seen_in_paper: set[tuple[str, ...]] = set()
        for n in (1, 2, 3, 4):
            if n > MAX_NGRAM_LEN:
                continue
            for j in range(len(toks) - n + 1):
                gram = tuple(toks[j : j + n])
                if not is_meaningful_ngram(gram):
                    continue
                if gram in seen_in_paper:
                    continue
                seen_in_paper.add(gram)
                ngram_counter[gram] += 1
                key = " ".join(gram)
                examples.setdefault(key, title or abstract[:200])

    print()
    print(f"Channel counts:")
    print(f"  keywords     : {len(keyword_counter):,} unique")
    print(f"  concepts     : {len(concept_counter):,} unique")
    print(f"  ngrams (raw) : {len(ngram_counter):,}")

    # ── Merge channels with weights ───────────────────────────────────
    # Weighting (informal):
    #   keyword ×3 (curated by authors)
    #   concept ×2 (curated by OpenAlex tagging)
    #   ngram   ×1 (raw frequency)
    merged: dict[str, dict] = {}

    def _add(term: str, n: int, channel: str, weight: int) -> None:
        if not term or len(term) < 3:
            return
        # Lowercase for dedupe key, but preserve original casing of the most
        # frequent variant
        key = term.lower()
        slot = merged.setdefault(key, {
            "canonical_en": term,
            "freq": 0,
            "channels": set(),
            "best_freq": 0,
        })
        slot["freq"] += n * weight
        slot["channels"].add(channel)
        if n > slot["best_freq"]:
            slot["canonical_en"] = term
            slot["best_freq"] = n

    for term, n in keyword_counter.items():
        _add(term, n, "keyword", 3)
    for term, n in concept_counter.items():
        _add(term, n, "concept", 2)
    for gram, n in ngram_counter.items():
        if n < MIN_NGRAM_FREQ:
            continue
        _add(" ".join(gram), n, "ngram", 1)

    # Sort by weighted frequency desc
    ranked = sorted(merged.values(), key=lambda d: d["freq"], reverse=True)
    cap = ranked[:TARGET_TERM_COUNT]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in cap:
            f.write(json.dumps({
                "canonical_en": rec["canonical_en"],
                "freq": rec["freq"],
                "channels": sorted(list(rec["channels"])),
                "example_sentence": examples.get(rec["canonical_en"].lower(), "")[:300],
            }, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {len(cap):,} candidate terms → {OUT_PATH}")
    print()
    print("Top 30 by weighted frequency:")
    for rec in cap[:30]:
        print(f"  {rec['freq']:>7}  [{','.join(rec['channels']):20}]  {rec['canonical_en']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
