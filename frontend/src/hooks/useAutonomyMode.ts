import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { settingsWriteProblem, type SettingsPatch } from "@/lib/settingsWrite";

export type AutonomyMode = "safe" | "semi" | "auto";

const SETTINGS_KEY = ["settings"] as const;

/** A mode switch is a tiny settings write. If it has not landed in 8 s it is not
 *  going to — fail it, so the control comes back to life. */
const MODE_POST_TIMEOUT_MS = 8000;

/**
 * Global operating mode (safe / semi / auto), backed by GET/POST /api/settings
 * (`autonomy_mode`). The BACKEND is the source of truth — it drives the
 * instrument-control agent's tip-processing gate (safe = no tip work + belief,
 * semi = pulses→HITL + shallow shaping, auto = all). So we do NOT persist to
 * localStorage; we read via react-query and write with an optimistic update.
 * Shares the ["settings"] cache with SettingsPage, so both stay in sync.
 */
export function useAutonomyMode() {
  const qc = useQueryClient();

  const q = useQuery({
    queryKey: SETTINGS_KEY,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings");
      if (error) throw error;
      return data;
    },
  });

  const raw = (q.data as { autonomy_mode?: string | null } | undefined)?.autonomy_mode;
  // Unknown / unset → "auto" (matches the backend's fail-open default).
  const mode: AutonomyMode = raw === "safe" || raw === "semi" ? raw : "auto";

  const setMode = useMutation({
    mutationFn: async (m: AutonomyMode) => {
      // BOUND THE REQUEST — an unbounded await behind the mode switch could hang.
      // TopBar disables all three mode buttons while this is pending. With no
      // timeout, a POST that never settles leaves isPending true FOREVER — the
      // operating-mode control goes permanently dead, with no error and no way
      // back. An unbounded await behind a disabled control is a freeze.
      const ctrl = new AbortController();
      const timer = window.setTimeout(() => ctrl.abort(), MODE_POST_TIMEOUT_MS);
      try {
        const body: SettingsPatch = { autonomy_mode: m };
        const { data, error } = await api.POST("/api/settings", {
          body,
          signal: ctrl.signal,
        });
        if (error) throw error;
        // 一次被拒绝的写入(`ok:false`)以前会走 onSuccess:乐观更新留在屏幕上,
        // 直到 onSettled 的重取悄悄把它换回去 —— 用户看到模式先变后变回来,
        // 没有任何一句话说为什么。抛出来,让 onError 立刻回滚。
        const problem = settingsWriteProblem(data);
        if (problem) throw new Error(problem);
        return data;
      } finally {
        window.clearTimeout(timer);
      }
    },
    // Optimistic: reflect the new mode immediately (border + control), roll back
    // on error, and re-fetch to reconcile with the persisted value.
    onMutate: async (m) => {
      await qc.cancelQueries({ queryKey: SETTINGS_KEY });
      const prev = qc.getQueryData(SETTINGS_KEY);
      qc.setQueryData(SETTINGS_KEY, (old: unknown) => ({
        ...((old as object | null) ?? {}),
        autonomy_mode: m,
      }));
      return { prev };
    },
    onError: (_e, _m, ctx) => {
      if (ctx?.prev !== undefined) qc.setQueryData(SETTINGS_KEY, ctx.prev);
    },
    onSettled: () => qc.invalidateQueries({ queryKey: SETTINGS_KEY }),
  });

  return { mode, setMode, isLoading: q.isLoading };
}
