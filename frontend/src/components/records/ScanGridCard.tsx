import { useState } from "react";
import type { components } from "@/api/schema";
import { copyBadge, copyLocations, locationKindLabel, parentFolder } from "@/lib/scanCopies";
import { fmtTime } from "./shared";
import { useScanPreview, type FlattenMode } from "./scanPreviewQuery";

// 数据 tab 的大图标卡片。
//
// The listing used to be three lines of text per file — name, extension+size,
// timestamp — for files whose names are `unnamed0003.sxm`. Picking the right one
// meant clicking through them one at a time and watching the preview pane. A
// thumbnail answers "which scan is this" without any clicks at all, which is the
// whole reason 大图标模式 was asked for.

type ScanFileEntry = components["schemas"]["ScanFileEntry"];

function fmtBytes(b?: number | null): string {
  if (b == null) return "—";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function fmtMtime(m?: number | null): string {
  if (m == null) return "—";
  return fmtTime(new Date(m * 1000).toISOString());
}

/** 副本明细。Expands in place — a modal for four lines of paths is heavier than
 *  the question deserves, and this has to stay readable next to the picture. */
function CopyDetail({ entry }: { entry: ScanFileEntry }) {
  return (
    <div className="mt-1 space-y-1 rounded border border-mast-border bg-mast-bg/60 p-1.5 text-[11px]">
      <div className="text-mast-muted">同一份数据存在 {entry.copies} 处：</div>
      {copyLocations(entry).map((loc) => (
        <div key={loc.path} className="break-all font-mono text-mast-faint" title={loc.path}>
          <span className="mr-1 font-sans text-mast-muted">[{locationKindLabel(loc.kind)}]</span>
          {loc.path}
        </div>
      ))}
    </div>
  );
}

export function ScanGridCard({
  entry,
  selected,
  flatten,
  onSelect,
}: {
  entry: ScanFileEntry;
  selected: boolean;
  flatten: FlattenMode;
  onSelect: () => void;
}) {
  const [showCopies, setShowCopies] = useState(false);
  // Thumbnails are small on purpose: 160 px is what the grid cell shows, and
  // asking for the 1024 px version to scale it down would move megabytes per
  // screen for pixels nobody sees.
  const prev = useScanPreview(entry.path, { size: 160, flatten });
  const badge = copyBadge(entry);

  return (
    <div
      className={
        "rounded-md border p-2 text-xs transition-colors " +
        (selected ? "border-mast-accent bg-mast-accent/10" : "border-mast-border hover:bg-mast-bg/60")
      }
    >
      <button type="button" onClick={onSelect} className="block w-full text-left">
        <div className="relative">
          {prev.data?.image ? (
            <img
              src={prev.data.image}
              alt={entry.name}
              /* contain, not cover: a scan is not decorative — cropping it to a
                 square hides the part of the frame the operator is looking for. */
              className="aspect-square w-full rounded border border-mast-border bg-black/40 object-contain"
            />
          ) : (
            <div className="flex aspect-square w-full items-center justify-center rounded border border-dashed border-mast-border bg-black/20 px-2 text-center text-[11px] leading-snug text-mast-muted">
              {prev.isPending
                ? "渲染中…"
                : prev.data?.detail
                  ? `无法预览：${prev.data.detail}`
                  : "无法预览"}
            </div>
          )}
          {badge && (
            <span
              title="这份数据在磁盘上有多个副本"
              className="absolute right-1 top-1 rounded-mast-badge bg-mast-bg/90 px-1.5 py-0.5 text-[11px] font-medium text-mast-muted"
            >
              {badge}
            </span>
          )}
        </div>
        <div className="mt-1.5 truncate font-medium text-mast-text" title={entry.path}>
          {entry.name}
        </div>
        {/* The session folder. Nanonis names files per session, so 20260319 and
            20260327 both hold an `unnamed0033.sxm` — two different measurements,
            one name. Without this the two cards are indistinguishable text. */}
        {parentFolder(entry.path) && (
          <div className="truncate text-mast-faint" title={entry.path}>
            {parentFolder(entry.path)}
          </div>
        )}
        <div className="flex justify-between text-mast-muted tabular-nums">
          <span className="uppercase">{entry.ext.replace(".", "")}</span>
          <span>{fmtBytes(entry.size_bytes)}</span>
        </div>
        <div className="text-mast-muted tabular-nums">{fmtMtime(entry.mtime)}</div>
      </button>

      {badge && (
        <>
          <button
            type="button"
            onClick={() => setShowCopies((v) => !v)}
            className="mt-1 text-[11px] text-mast-accent hover:underline"
          >
            {showCopies ? "收起副本" : `查看 ${entry.copies} 个副本位置`}
          </button>
          {showCopies && <CopyDetail entry={entry} />}
        </>
      )}
    </div>
  );
}
