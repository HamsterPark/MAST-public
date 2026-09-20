// Lightweight CN/EN fuzzy matcher — zero deps (the project pins deps).
//
// Used by pickers that will grow long (skills / composites). Scores a query
// against a text:
//   • exact substring  → highest, earlier match ranks higher;
//   • subsequence      → every query char appears in order, fewer gaps = better;
//   • no match         → 0.
// CJK is matched per-character (each 汉字 is its own char), so 中文 search works
// without tokenisation, and English fuzzy ("cndtip" → "ConditionTip") works too.

export function fuzzyScore(query: string, text: string): number {
  const q = query.trim().toLowerCase();
  if (!q) return 1;
  const t = (text || "").toLowerCase();
  if (!t) return 0;

  // exact substring → high score; earlier position ranks higher
  const idx = t.indexOf(q);
  if (idx >= 0) return 10000 - Math.min(idx, 9000);

  // subsequence: every query char appears in order
  let ti = 0;
  let gaps = 0;
  for (const ch of q) {
    const found = t.indexOf(ch, ti);
    if (found < 0) return 0; // not a subsequence → no match
    gaps += found - ti;
    ti = found + 1;
  }
  // matched as a subsequence — fewer gaps (more contiguous) ranks higher
  return Math.max(1, 1000 - gaps);
}

/** Best score of a query across several text fields (name / description / tags). */
export function fuzzyScoreFields(
  query: string,
  ...fields: (string | null | undefined)[]
): number {
  let best = 0;
  for (const f of fields) {
    if (!f) continue;
    const s = fuzzyScore(query, f);
    if (s > best) best = s;
  }
  return best;
}
