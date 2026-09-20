import type { ReactNode } from "react";
import clsx from "clsx";

/** Small presentational helpers shared by the Records domain panels. These are
 *  RecordsPage-local; they intentionally do not touch the global ui.tsx set. */

/** Status pill with a tone derived from common experiment/campaign/action
 *  status vocabulary. Falls back to a neutral accent tone. */
const STATUS_TONE: Record<string, string> = {
  running: "bg-mast-info-bg text-mast-info",
  active: "bg-mast-info-bg text-mast-info",
  in_progress: "bg-mast-info-bg text-mast-info",
  completed: "bg-mast-auto-bg text-mast-auto",
  done: "bg-mast-auto-bg text-mast-auto",
  success: "bg-mast-auto-bg text-mast-auto",
  ok: "bg-mast-auto-bg text-mast-auto",
  failed: "bg-mast-danger-bg text-mast-danger",
  error: "bg-mast-danger-bg text-mast-danger",
  aborted: "bg-mast-danger-bg text-mast-danger",
  cancelled: "bg-mast-warn-bg text-mast-warn",
  paused: "bg-mast-warn-bg text-mast-warn",
  pending: "bg-mast-warn-bg text-mast-warn",
};

export function StatusPill({ status }: { status?: string | null }) {
  const s = (status ?? "").toLowerCase();
  const tone = STATUS_TONE[s] ?? "bg-mast-accent/15 text-mast-accent";
  return (
    <span className={clsx("inline-block rounded px-1.5 py-0.5 text-xs", tone)}>
      {status || "—"}
    </span>
  );
}

/** boolean success → 成功 / 失败 / — pill */
export function SuccessPill({ success }: { success?: boolean | null }) {
  if (success === null || success === undefined) {
    return <span className="text-xs text-mast-muted">—</span>;
  }
  return (
    <span
      className={clsx(
        "inline-block rounded px-1.5 py-0.5 text-xs",
        success ? "bg-mast-auto-bg text-mast-auto" : "bg-mast-danger-bg text-mast-danger",
      )}
    >
      {success ? "成功" : "失败"}
    </span>
  );
}

/** A label : value definition row used in detail panes. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-3 py-1 text-sm">
      <div className="w-28 shrink-0 text-mast-muted">{label}</div>
      <div className="min-w-0 flex-1 break-words text-mast-text">{children ?? "—"}</div>
    </div>
  );
}

/** Pretty-print a small JSON object (params / state_delta / meta). */
export function JsonBlock({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="text-mast-muted">—</span>;
  if (typeof value === "object" && Object.keys(value as object).length === 0) {
    return <span className="text-mast-muted">{"{}"}</span>;
  }
  return (
    <pre className="max-h-64 overflow-auto rounded border border-mast-border bg-mast-bg/60 p-2 text-xs">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

/** Format an ISO/HLC-ish timestamp compactly; pass through if not parseable. */
export function fmtTime(ts?: string | null): string {
  if (!ts) return "—";
  const d = new Date(ts);
  if (!Number.isNaN(d.getTime())) {
    return d.toLocaleString("zh-CN", { hour12: false });
  }
  // HLC display strings ('2026-03-22T15:42:08Z-0009-main') etc. — show raw.
  return ts;
}
