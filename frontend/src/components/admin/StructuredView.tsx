import { type ReactNode, useState } from "react";
import clsx from "clsx";
import { SubTabs } from "@/components/controls";

/** StructuredView — a smart, recursive read-only renderer that turns arbitrary
 *  nested JSON (knowledge ktypes + encyclopedia sections) into a readable,
 *  reference-style UI instead of a raw JSON blob. The shaping rules:
 *
 *    · top-level / nested DICT  → each key is a titled section (collapsible at
 *      the top level, key→value rows when the values are scalars).
 *    · LIST of objects          → a DataTable (columns = union of keys; scalar
 *      cells inline, nested objects/lists collapse to a small expandable cell).
 *    · LIST of scalars          → chips / bullets.
 *    · scalar                   → plain value (numbers use tabular-nums, long
 *      text wraps).
 *
 *  All colours use mast-* tokens (text-mast-text for body) so it reads in BOTH
 *  light and dark themes. Pure presentational — no fetching, never freezes. */

type Json = unknown;

const isPlainObject = (v: Json): v is Record<string, Json> =>
  v != null && typeof v === "object" && !Array.isArray(v);

const isScalar = (v: Json): v is string | number | boolean | null =>
  v == null || ["string", "number", "boolean"].includes(typeof v);

const isScalarList = (v: Json[]): boolean => v.every(isScalar);
const isObjectList = (v: Json[]): boolean =>
  v.length > 0 && v.every((x) => isPlainObject(x));

/** Humanize a snake_case / camelCase key into a readable Chinese-friendly label
 *  (we only de-underscore + title the latin word fragments; non-latin passes
 *  through untouched). */
function humanizeKey(key: string): string {
  if (/[一-鿿]/.test(key)) return key; // already CJK — leave as-is
  return key
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

// ── scalar rendering ────────────────────────────────────────────────────────
function ScalarValue({ value }: { value: string | number | boolean | null }) {
  if (value == null) return <span className="text-mast-muted">—</span>;
  if (typeof value === "boolean")
    return (
      <span
        className={clsx(
          "inline-flex items-center gap-1 rounded-mast-badge border px-1.5 py-0.5 font-mono text-xs",
          value
            ? "border-mast-auto-border bg-mast-auto-bg text-mast-auto"
            : "border-mast-border bg-mast-panel-2 text-mast-muted",
        )}
      >
        {value ? "true" : "false"}
      </span>
    );
  if (typeof value === "number")
    return <span className="font-mono tabular-nums text-mast-accent">{value}</span>;
  // string — wrap long text; keep short tokens compact
  return <span className="whitespace-pre-wrap break-words font-mono text-mast-text">{value}</span>;
}

// ── chips for a list of scalars ─────────────────────────────────────────────
function ScalarChips({ items }: { items: Json[] }) {
  if (!items.length) return <span className="text-mast-muted">（空）</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((it, i) => (
        <span
          key={i}
          className="rounded-mast-badge border border-mast-border bg-mast-panel-2 px-2 py-0.5 text-xs text-mast-text"
        >
          {it == null ? "—" : typeof it === "number" ? (
            <span className="font-mono tabular-nums">{it}</span>
          ) : (
            String(it)
          )}
        </span>
      ))}
    </div>
  );
}

// ── a collapsed-by-default expandable for nested cells inside a table ────────
function CellExpandable({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 rounded-mast-badge border border-mast-border bg-mast-panel-2 px-1.5 py-0.5 text-xs text-mast-muted transition-colors hover:text-mast-text"
      >
        <span>{open ? "▾" : "▸"}</span>
        <span>{label}</span>
      </button>
      {open && <div className="mt-1.5 border-l-2 border-mast-border pl-2">{children}</div>}
    </div>
  );
}

/** Render a single table cell value: scalars inline, nested structures as a
 *  small expandable that recursively re-enters StructuredValue. */
function TableCell({ value }: { value: Json }) {
  if (isScalar(value)) return <ScalarValue value={value} />;
  if (Array.isArray(value)) {
    if (!value.length) return <span className="text-mast-muted">—</span>;
    if (isScalarList(value)) return <ScalarChips items={value} />;
    return (
      <CellExpandable label={`${value.length} 项`}>
        <StructuredValue value={value} depth={2} />
      </CellExpandable>
    );
  }
  // nested object
  const keys = Object.keys(value as Record<string, Json>);
  return (
    <CellExpandable label={`${keys.length} 字段`}>
      <StructuredValue value={value} depth={2} />
    </CellExpandable>
  );
}

// ── a list of objects → table ───────────────────────────────────────────────
function ObjectTable({ rows }: { rows: Record<string, Json>[] }) {
  // union of keys, preserving first-seen order
  const columns: string[] = [];
  for (const r of rows) for (const k of Object.keys(r)) if (!columns.includes(k)) columns.push(k);

  return (
    <div className="overflow-hidden rounded-mast-card border border-mast-border shadow-mast">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-mast-panel-2">
            <tr>
              {columns.map((c) => (
                <th
                  key={c}
                  className="whitespace-nowrap px-3.5 py-2 text-left text-xs font-medium tracking-wide text-mast-faint"
                >
                  {humanizeKey(c)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r, i) => (
              <tr
                key={i}
                className="border-t border-mast-border align-top transition-colors hover:bg-mast-panel-2/40"
              >
                {columns.map((c) => (
                  <td key={c} className="px-3.5 py-2.5 align-top">
                    {c in r ? (
                      <TableCell value={r[c]} />
                    ) : (
                      <span className="text-mast-muted">—</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// A dict is "tabbable" only when it is a FEW (2–8) named SECTIONS each holding a
// sizable container (e.g. experiment_design's 5 big sub-docs) — those read well
// as one-section-per-tab. Everything else (a long id→record/lookup map like
// 复合层级 13 / 技能验证 50, or any dict of scalar/list values) is rendered as a
// TABLE so the DETAILS are visible — NOT 13/50 useless tabs.
function isTabbable(entries: [string, Json][]): boolean {
  if (entries.length < 2 || entries.length > 8) return false;
  return entries.every(
    ([, v]) =>
      (isPlainObject(v) && Object.keys(v).length > 3) ||
      (Array.isArray(v) && v.length > 5),
  );
}

/** A dict keyed by id/name → a table: the KEY is the first column; an object
 *  value spreads into field columns, a scalar/list value goes in a single 值
 *  column (rendered by TableCell — scalars inline, lists as chips, nested as an
 *  expandable). So 复合层级 (键→步骤列表) and 技能验证 (键→字符串) show details. */
function RecordMapTable({ obj, keyLabel = "名称" }: { obj: Record<string, Json>; keyLabel?: string }) {
  const rows = Object.entries(obj).map(([k, v]) =>
    isPlainObject(v) ? { [keyLabel]: k, ...v } : { [keyLabel]: k, 值: v },
  );
  return <ObjectTable rows={rows as Record<string, Json>[]} />;
}

// ── key→value rows for a dict of scalars (or shallow mixed dict) ────────────
function KeyValueRows({ obj }: { obj: Record<string, Json> }) {
  const entries = Object.entries(obj);
  if (!entries.length) return <p className="text-sm text-mast-muted">（空）</p>;
  return (
    <dl className="divide-y divide-mast-border overflow-hidden rounded-mast-card border border-mast-border shadow-mast">
      {entries.map(([k, v]) => (
        <div
          key={k}
          className="flex flex-col gap-1 bg-mast-panel px-3.5 py-2.5 transition-colors hover:bg-mast-panel-2/40 sm:flex-row sm:gap-4"
        >
          <dt className="shrink-0 text-sm font-medium text-mast-faint sm:w-48">{humanizeKey(k)}</dt>
          <dd className="min-w-0 flex-1 text-sm">
            <StructuredValue value={v} depth={2} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** The core recursive dispatcher for a value (not a top-level section). */
function StructuredValue({ value, depth = 0 }: { value: Json; depth?: number }) {
  if (isScalar(value)) return <ScalarValue value={value} />;

  if (Array.isArray(value)) {
    if (!value.length) return <span className="text-mast-muted">（空列表）</span>;
    if (isScalarList(value)) return <ScalarChips items={value} />;
    if (isObjectList(value)) return <ObjectTable rows={value as Record<string, Json>[]} />;
    // mixed / nested list — render each item stacked
    return (
      <div className="space-y-2">
        {value.map((it, i) => (
          <div key={i} className="rounded-mast-ctl border border-mast-border bg-mast-panel-2/40 p-2">
            <div className="mb-1 font-mono text-xs text-mast-faint">#{i + 1}</div>
            <StructuredValue value={it} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }

  // object — a multi-entry record/lookup map → a table (key as first column); a
  // tiny dict → key/value rows (recurses per value).
  const obj = value as Record<string, Json>;
  if (Object.keys(obj).length >= 3) return <RecordMapTable obj={obj} />;
  return <KeyValueRows obj={obj} />;
}

// ── top-level collapsible section (one per top-level dict key) ──────────────
function TopSection({
  title,
  count,
  defaultOpen,
  children,
}: {
  title: string;
  count?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen ?? true);
  return (
    <div className="overflow-hidden rounded-mast-card border border-mast-border bg-mast-panel shadow-mast">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 bg-mast-panel-2 px-3.5 py-2 text-left transition-colors hover:bg-mast-panel-2/70"
      >
        <span className="flex items-center gap-2">
          <span className="text-mast-faint">{open ? "▾" : "▸"}</span>
          <span className="text-sm font-medium text-mast-text">{title}</span>
        </span>
        {count && <span className="font-mono text-xs tabular-nums text-mast-faint">{count}</span>}
      </button>
      {open && <div className="border-t border-mast-border p-3">{children}</div>}
    </div>
  );
}

function describeValue(v: Json): string | undefined {
  if (Array.isArray(v)) return `${v.length} 项`;
  if (isPlainObject(v)) return `${Object.keys(v).length} 字段`;
  return undefined;
}

/** StructuredView — the public entry. Top-level dict ⇒ one collapsible section
 *  per key; top-level list ⇒ a single table/chips; scalar ⇒ inline. The first
 *  few sections open by default so the surface is immediately readable. */
export function StructuredView({ value }: { value: Json }) {
  if (value == null) return <p className="text-sm text-mast-muted">（无默认值）</p>;

  // top-level scalar
  if (isScalar(value)) return <ScalarValue value={value} />;

  // top-level list → render directly (table / chips / stacked)
  if (Array.isArray(value)) {
    if (!value.length) return <p className="text-sm text-mast-muted">（空）</p>;
    return <StructuredValue value={value} />;
  }

  // top-level dict → ONE TAB per key (so a big multi-section block is browsed one
  // section at a time instead of a giant scroll). A single-key dict renders直接.
  const entries = Object.entries(value as Record<string, Json>);
  if (!entries.length) return <p className="text-sm text-mast-muted">（空）</p>;
  if (entries.length === 1) {
    const [k, v] = entries[0]!;
    return (
      <TopSection title={humanizeKey(k)} count={describeValue(v)} defaultOpen>
        <StructuredValue value={v} depth={1} />
      </TopSection>
    );
  }
  // A few big heterogeneous sections (experiment_design's 5 sub-docs) → tabs.
  // Everything else — a long id→record/lookup map (复合层级 / 技能验证), or any
  // dict of scalar/list values — → ONE table with details visible.
  if (isTabbable(entries)) return <DictTabs entries={entries} />;
  return <RecordMapTable obj={value as Record<string, Json>} />;
}

/** Top-level dict as tabs — one tab per key, each showing a count badge. Keeps a
 *  large reference doc (e.g. experiment_design's 5 big sections) browsable one
 *  section at a time. */
function DictTabs({ entries }: { entries: [string, Json][] }) {
  const [tab, setTab] = useState(entries[0]![0]);
  const active = entries.find(([k]) => k === tab) ?? entries[0]!;
  return (
    <div>
      <SubTabs
        value={tab}
        onChange={setTab}
        tabs={entries.map(([k, v]) => ({
          id: k,
          label: humanizeKey(k),
          badge: Array.isArray(v) ? v.length : isPlainObject(v) ? Object.keys(v).length : undefined,
        }))}
      />
      <StructuredValue value={active[1]} depth={1} />
    </div>
  );
}
