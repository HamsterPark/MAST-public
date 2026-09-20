import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

/**
 * Number of UNRESOLVED fetch requests on the literature fetch board.
 *
 * The literature agent posts a request when it needs a full text the corpus
 * only has an abstract for, and tells itself "用户将在文献库标签处理". On
 * 2026-07-27 it posted three (fr-1, fr-2, fr-3) and all three sat untouched:
 * the board exists and works, but **nothing tells the operator a request is
 * waiting**, and nobody opens a tab to check for work they don't know about.
 * So the agent's promise was, in practice, an empty one.
 *
 * …and then this hook shipped with the same bug in a second form: it counted
 * rows whose status was `"open"`, a value the backend never produces. The board's
 * open state is `"pending"` (`knowledge/fetch_board.py` `_OPEN`), so the filter
 * matched nothing, the badge never appeared, and the fix for "nothing tells the
 * operator" told the operator nothing. It now reads `pending_count` straight off
 * the same response the board page uses — one number, computed server-side, with
 * no client-side guess about which status strings mean "waiting".
 *
 * Fails silently to 0 — a badge is an affordance, never a reason for the shell
 * to show an error or stall. `retry: false` keeps a dead endpoint from
 * retrying on every poll.
 */
const POLL_MS = 60_000;

export function useOpenFetchRequests(): number {
  const q = useQuery({
    queryKey: ["literature", "fetch-board", "open-count"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/literature/fetch-board");
      if (error) throw error;
      return data;
    },
    refetchInterval: POLL_MS,
    refetchOnWindowFocus: true,
    retry: false,
    staleTime: POLL_MS / 2,
  });

  const d = q.data as
    | { pending_count?: number | null; degraded?: boolean }
    | undefined;
  if (!d || d.degraded) return 0;
  const n = Number(d.pending_count ?? 0);
  return Number.isFinite(n) && n > 0 ? n : 0;
}
