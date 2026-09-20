import { useCallback, useState } from "react";

/**
 * A text input whose draft survives tab switches / page navigation
 * (previously, switching tabs did not persist the draft).
 *
 * Backed by sessionStorage: drafts live for the browser-tab session (a full
 * app restart intentionally clears them — half-typed commands from yesterday
 * should not reappear). Same call signature as useState<string>.
 */
export function useDraft(key: string): [string, (v: string) => void] {
  const storageKey = `mast.draft.${key}`;
  const [value, setValue] = useState<string>(() => {
    try {
      return sessionStorage.getItem(storageKey) ?? "";
    } catch {
      return "";
    }
  });
  const set = useCallback(
    (v: string) => {
      setValue(v);
      try {
        if (v) sessionStorage.setItem(storageKey, v);
        else sessionStorage.removeItem(storageKey);
      } catch {
        /* storage unavailable (private mode quota) — draft is memory-only */
      }
    },
    [storageKey],
  );
  return [value, set];
}
