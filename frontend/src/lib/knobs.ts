/**
 * Operator-knob helpers shared by every settings section that renders a backend
 * knob CATALOGUE (`/api/monitoring/config`, `/api/env-history/config`, …).
 *
 * Lives here rather than in `lib/monitoring.ts` because the whole-replace trap
 * below is a property of the SETTINGS STORE, not of any one subsystem — the
 * second subsystem to grow a knob catalogue would otherwise have copied the
 * guard, and a copied guard is one that drifts.
 *
 * `lib/monitoring.ts` re-exports these so existing imports keep working.
 *
 * ZERO IMPORTS, and it has to stay that way. `lib/monitoring.ts` is loaded
 * directly by `node --test` (no bundler, no path aliases), so anything it pulls
 * in — including through this file — must resolve under plain Node. A single
 * `@/…` specifier here takes the whole monitoring test file down with it.
 */

export interface KnobLike {
  key: string;
  min?: number;
  max?: number;
  value?: number;
  default?: number;
  is_bool?: boolean;
}

function pickFinite(...vals: unknown[]): number {
  for (const v of vals) {
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return 0;
}

/**
 * Build the FULL knob dict to POST — every knob, not just the one that changed.
 *
 * WHOLE-REPLACE GUARD. The settings store replaces the whole dict, and the
 * backend's `from_mapping` fills any absent key with its DEFAULT. So posting
 * `{cm_keep_gb: 8}` alone does not mean "change one knob" — it means "set
 * cm_keep_gb to 8 and reset the other eighteen", including alert thresholds
 * someone spent a session calibrating. Exactly the same is true of the
 * environment-history dict, where the key that would silently snap back is
 * `eh_raw_keep_days` — the one knob in this system that deletes anything.
 *
 * Precedence per key: the explicit `change` → the persisted value → the
 * catalogue's effective `value` → its `default`. Persisted outranks the
 * catalogue because a just-saved value can reach the settings query before the
 * config query has refetched.
 */
export function buildKnobPayload(
  knobs: readonly KnobLike[],
  persisted: Readonly<Record<string, unknown>> | null | undefined,
  change?: { key: string; value: number } | null,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const k of knobs) {
    if (!k || typeof k.key !== "string" || !k.key) continue;
    const raw =
      change && change.key === k.key
        ? change.value
        : pickFinite(persisted?.[k.key], k.value, k.default);
    out[k.key] = clampKnob(k, raw);
  }
  // A change for a key the catalogue does not list still goes out: the only way
  // that happens is a frontend holding a stale catalogue, and dropping the
  // operator's edit silently would be worse than passing it to a backend that
  // validates anyway.
  if (change && !(change.key in out) && Number.isFinite(change.value)) {
    out[change.key] = change.value;
  }
  return out;
}

/** Coerce one knob's value into what the backend will accept: booleans become
 *  exactly 0 or 1, numbers are held inside the catalogue's bounds. */
export function clampKnob(k: KnobLike, value: number): number {
  if (!Number.isFinite(value)) return pickFinite(k.value, k.default);
  if (k.is_bool) return value >= 0.5 ? 1 : 0;
  let v = value;
  if (typeof k.min === "number" && Number.isFinite(k.min)) v = Math.max(k.min, v);
  if (typeof k.max === "number" && Number.isFinite(k.max) && k.max > 0) v = Math.min(k.max, v);
  return v;
}

/**
 * Text for a knob's input box.
 *
 * NOT SettingsPage's `_fmtThreshold`, which is `Math.round(v * 1000) / 1000`.
 * That is fine for the classical thresholds (all O(0.01–100)) and catastrophic
 * here: these knobs are currents in amperes, so a 20 pA warn threshold (2e-11)
 * renders as "0" and the row's own blur handler then commits that 0 back. The
 * operator would watch a calibrated threshold silently zero itself by looking
 * at it.
 *
 * `String(v)` is JavaScript's shortest round-tripping form — "2e-11" stays
 * "2e-11" and parses back to exactly the same double.
 */
export function fmtKnobValue(v: number): string {
  if (!Number.isFinite(v)) return "";
  return String(v);
}
