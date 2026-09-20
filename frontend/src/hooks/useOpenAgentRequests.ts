import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

/**
 * Number of unanswered agent→operator requests on the 心愿单 board.
 *
 * The board has the better machinery of the two ask-the-operator channels — an
 * answer is injected into the agent's next turn automatically, no polling — and
 * the worse visibility: nothing anywhere told the operator a request had
 * arrived. So an agent could ask "请到机台更换样品" and wait forever on someone
 * who had no reason to open that tab. The fetch board had exactly this problem
 * and got a badge; this is the same fix for the same reason.
 *
 * Reads `pending_count` off the response rather than filtering rows client-side.
 * That is not a style preference: the fetch-request badge shipped filtering for
 * a status string the backend never emits, so it counted zero forever and the
 * fix for "nobody is told" told nobody.
 *
 * Fails silently to 0 — a badge is an affordance, never a reason for the shell
 * to show an error or stall.
 */
const POLL_MS = 60_000;

export function useOpenAgentRequests(): number {
  const q = useQuery({
    queryKey: ["wishlist", "open-request-count"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/wishlist");
      if (error) throw error;
      return data;
    },
    refetchInterval: POLL_MS,
    refetchOnWindowFocus: true,
    retry: false,
    staleTime: POLL_MS / 2,
  });

  const d = q.data as { pending_count?: number | null; degraded?: boolean } | undefined;
  if (!d || d.degraded) return 0;
  const n = Number(d.pending_count ?? 0);
  return Number.isFinite(n) && n > 0 ? n : 0;
}
