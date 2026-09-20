// 数据 tab 的翻页累积。
//
// The listing used to ask for a fixed 60 files and print "只显示最近 60 个"
// underneath. File 61 was unreachable by any interaction on the page — the
// number was not a page size, it was a ceiling. The server now takes an
// `offset`, and these helpers hold the accumulated pages.
//
// Pure functions, no React: components cannot be rendered in this repo's test
// setup (`node --test`), so anything worth asserting has to live here.

export interface PagedScan {
  path: string;
}

/**
 * Append the next page, dropping entries already present by path.
 *
 * The de-dup is not paranoia. This listing is mtime-desc over a directory the
 * instrument is still writing to: a scan saved between page 1 and page 2 shifts
 * every later entry one slot down, so the first row of page 2 is the row the
 * operator already saw at the bottom of page 1. Without this the same file
 * appears twice, which — on a tab whose whole point this round was to STOP
 * showing one measurement twice — would read as the copy bug coming back.
 *
 * (The reverse shift, an entry skipped across the boundary, cannot be fixed
 * here; it comes back on the next refresh.)
 */
export function appendScans<T extends PagedScan>(prev: readonly T[], next: readonly T[]): T[] {
  const seen = new Set(prev.map((s) => s.path));
  const out = prev.slice();
  for (const s of next) {
    if (seen.has(s.path)) continue;
    seen.add(s.path);
    out.push(s);
  }
  return out;
}

/**
 * How many entries to skip to ask for what comes after what we hold.
 *
 * Derived from the accumulated length rather than counting button presses:
 * a page that arrived short (because entries were de-duped) must not leave the
 * offset running ahead of the list, which would skip files silently.
 */
export function nextOffset(accumulated: readonly PagedScan[]): number {
  return accumulated.length;
}

/** Text for the "showing X of Y" line. Empty string when there is nothing to say. */
export function pageSummary(shown: number, collapsed: number, rawFiles: number): string {
  if (!collapsed) return "";
  const parts = [`显示 ${shown} / ${collapsed} 项`];
  // Only mention copies when there ARE copies: on a machine with the ingest off
  // the two numbers are equal and the extra clause is noise.
  if (rawFiles > collapsed) {
    parts.push(`已折叠 ${rawFiles - collapsed} 个重复副本（共 ${rawFiles} 个文件）`);
  }
  return parts.join(" · ");
}
