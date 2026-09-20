// 新仪器初始化 — which items the page shows.
//
// The page has to serve two jobs that pull in opposite directions:
//
//   1. first install — "what is still missing", a to-do list;
//   2. going back to change ONE number — "where is 退针方向".
//
// It only ever served (1). Answered items stay in the payload, but they end up
// inside a group whose header reads「完成」and which therefore collapses, so from
// the operator's side they have vanished. On 2026-08-04 that cost a round trip
// through the API to flip a retract direction — the single field where being
// wrong drives the tip INTO the sample, i.e. the one most likely to need
// changing after someone measures it.
//
// So: a scope switch, and a search that ignores the scope entirely. Search has
// to reach an answered item or it does not solve (2).
//
// Pure, and tested, because a filter that silently hides the wrong thing looks
// exactly like a filter that works.

// Explicit `.ts`: the unit tests run this file under `node --test` (see
// package.json), and Node's ESM resolver does not guess extensions. tsconfig has
// allowImportingTsExtensions, and Vite resolves it the same way.
import { stripBold } from "./inlineBold.ts";

export interface InitItemLike {
  id: string;
  group: string;
  label: string;
  key: string;
  store: string;
  severity: string;
  complete: boolean;
  status: string;
  hint?: string | null;
  consequence?: string | null;
  unit?: string | null;
}

export type InitScope = "todo" | "all";

/**
 * Everything the search box looks at for one item.
 *
 * The prose is stripped of its `**` markup first — otherwise searching for a
 * phrase that happens to be emphasised in the catalog ("运行时有兜底") misses,
 * and the operator concludes the search is broken rather than that they typed
 * across an asterisk they cannot see.
 */
function haystack(item: InitItemLike): string {
  return [
    item.label, item.key, item.store, item.id, item.unit,
    // `hint` 是可搜的:用户记得住的往往是「铭牌」「面板」这类**去哪儿找**,
    // 而不是这一项叫什么。它进 haystack 之后「铭牌」能搜到前放那两项。
    // 合并自从前的 what + where(#39/#40),两个字段都进过 haystack,这里没有
    // 少掉任何可搜的字。
    stripBold(item.hint), stripBold(item.consequence),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export function matchesQuery(item: InitItemLike, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const hay = haystack(item);
  // Every whitespace-separated term must appear. Chinese has no word breaks, so
  // a single term is the normal case and this degrades to plain substring.
  return q.split(/\s+/).every((term) => hay.includes(term));
}

/**
 * `todo` = anything not yet answered, at ANY severity — plus items still on a
 * factory default, which the operator has not looked at either.
 *
 * `n/a` items never appear in 待办: they were ruled out by an earlier answer,
 * so listing them as outstanding work would be a lie.
 */
export function isOutstanding(item: InitItemLike): boolean {
  if (item.status === "n/a") return false;
  return !item.complete;
}

export function visibleItems(
  items: readonly InitItemLike[],
  opts: { scope: InitScope; query: string },
): InitItemLike[] {
  const q = opts.query.trim();
  // A search reaches across the whole catalog on purpose — the answered item
  // the operator is hunting for is by definition not in 待办.
  if (q) return items.filter((i) => matchesQuery(i, q));
  if (opts.scope === "all") return [...items];
  return items.filter(isOutstanding);
}

/** Group ids that should be open, given what survived the filter. */
export function groupsToOpen(
  visible: readonly InitItemLike[],
  opts: { query: string },
): Set<string> {
  const open = new Set<string>();
  if (opts.query.trim()) {
    // A hit the operator cannot see is not a hit.
    for (const i of visible) open.add(i.group);
    return open;
  }
  for (const i of visible) {
    if (i.severity === "required" && isOutstanding(i)) open.add(i.group);
  }
  return open;
}
