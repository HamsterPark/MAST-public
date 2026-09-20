import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import type { components } from "../../api/schema";
import { DegradedNote, EmptyNote, ErrorNote, Section, Spinner } from "../ui";
import { Button } from "../controls";
import { fmtTime } from "./shared";
import { KIND_EXTS, extParam, groupScans, otherCount } from "@/lib/scanKinds";
import { appendScans, pageSummary } from "@/lib/scanPager";
import { copyBadge, parentFolder } from "@/lib/scanCopies";
import { useStickyTab } from "@/hooks/useStickyTab";
import { ScanGridCard } from "./ScanGridCard";
import { ScanPreviewPanel } from "./ScanPreviewPanel";
import { ScanGroupBrowser } from "./ScanGroupBrowser";
import { GRID_FLATTEN_MODES, flattenLabel, type FlattenMode } from "./scanPreviewQuery";

// 数据 sub-tab — reproduces the old scan_preview "Data strip":
//   list recent .sxm/.dat/.3ds scans (GET /api/scans/latest) + "Load Latest"
//   + per-file base64 PNG preview (GET /api/scans/preview).
//
// ── layout  ──────────────────────────────────────────────────
// The preview used to sit BELOW the whole file grid, so with 60 files on a
// three-column grid the operator scrolled past twenty rows to see the picture
// of the file they had just clicked — and scrolled back up to click the next
// one. The list and the preview are now side by side, the preview sticky, and
// the list scrolls inside its own box rather than growing the page. On a narrow
// screen there is no room for two columns, so the preview goes ABOVE the list
// instead: the point is never having to scroll to reach it.
//
// ── filtering (同一组件的另一处 #53 同形状) ────────────────────────────────
// The type filter used to be a `scans.filter(...)` over a list the SERVER had
// already truncated to `n`. Selecting SXM could therefore show nothing while
// .sxm files sat on disk, because the newest `n` were all .dat — literally
// , moved to a different tab. The filter now goes to the server as
// `ext`, which applies it before the slice; the chip counts come from
// `counts_by_ext`, which the server computes over the FULL discovery result so
// a chip cannot read 0 for a type that is merely filtered out right now.
//
// ── 2026-08-21：图标视图 / 翻页 / 副本折叠 / 去衬底 ────────────────────────
// Four operator complaints, one screen:
//   * 「数据只是平铺预览，不直观」 → thumbnails in a grid (default), with the
//     old text rows still available as 列表, and 分组 for browsing by experiment.
//   * 「只能看 60 张图」 → the 60 was a hardcoded ceiling with no way past it.
//     The server takes an offset now and this pages with 加载更多.
//   * 「数据自动拷贝导致看到重复数据」 → the server folds byte-identical copies
//     into one entry; the card says ×N and lists where they are. The copying
//     itself is untouched — it is what makes an experiment folder self-contained.
//   * 「至少应该提供去衬底或不去衬底」 → a flatten switch, defaulting to 逐行平场.

type ScanFileEntry = components["schemas"]["ScanFileEntry"];

/** One page. Not a ceiling — 加载更多 asks for the next one. */
const PAGE = 30;

const VIEWS = ["grid", "list", "groups"] as const;
type ViewMode = (typeof VIEWS)[number];
const VIEW_LABEL: Record<ViewMode, string> = {
  grid: "网格",
  list: "列表",
  groups: "分组",
};

function useLatestScans(kind: string, countKeys: string[], offset: number) {
  const ext = extParam(kind, countKeys);
  return useQuery({
    queryKey: ["scans", "latest", PAGE, offset, ext ?? ""],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/scans/latest", {
        params: { query: ext ? { n: PAGE, offset, ext } : { n: PAGE, offset } },
      });
      if (error) throw error;
      return data;
    },
    // Chip counts describe the whole disk, not this filter, so keeping the last
    // response on screen while the next one loads stops the row from collapsing
    // and re-expanding on every click.
    placeholderData: (prev) => prev,
  });
}

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

export function ScanDataPane() {
  const [kind, setKind] = useState<string>("all");
  const [selected, setSelected] = useState<string | null>(null);
  // Counts survive a filter change (they describe the disk, not the filter), so
  // the chip row keeps its labels while the filtered page is in flight.
  const [countsByExt, setCountsByExt] = useState<Record<string, number>>({});
  const [offset, setOffset] = useState(0);
  const [acc, setAcc] = useState<ScanFileEntry[]>([]);

  // Both remembered across navigations: an operator who works in 列表 should not
  // be handed 网格 every time they come back, and the flatten mode is a property
  // of how they read scans, not of this visit.
  const [view, setView] = useStickyTab<ViewMode>("records.data.view", VIEWS, "grid");
  const [flatten, setFlatten] = useStickyTab<FlattenMode>(
    "records.data.flatten",
    ["raw", "plane", "line", "auto"],
    "line",
  );

  // `extParam` reads the count KEYS to expand 其他, and the counts come from the
  // query — so the keys have to be held in state rather than read straight off
  // `q.data`, which would make the query key depend on its own result.
  const q = useLatestScans(kind, Object.keys(countsByExt), offset);

  const page: ScanFileEntry[] = useMemo(() => q.data?.scans ?? [], [q.data]);
  // Accumulate pages. `appendScans` drops paths already held, which matters
  // because the rig writes to these directories while the operator pages: a new
  // scan shifts every later entry down one slot, so page 2 starts with the row
  // page 1 ended on.
  useEffect(() => {
    if (!q.data) return;
    setAcc((prev) => (offset === 0 ? page : appendScans(prev, page)));
  }, [q.data, page, offset]);

  // Page one renders straight from the response so the first paint is not a
  // frame behind the effect that fills `acc`.
  const scans = offset === 0 ? page : acc;

  const fresh = q.data?.counts_by_ext as Record<string, number> | undefined;
  const freshKey = fresh ? JSON.stringify(fresh) : "";
  useEffect(() => {
    if (fresh) setCountsByExt(fresh);
    // freshKey, not `fresh`: the object identity changes on every refetch even
    // when the disk has not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [freshKey]);

  // Display always prefers the live response; the state is only the fallback
  // while a filter change is in flight.
  const counts = fresh ?? countsByExt;
  // Chip labels count ENTRIES, not files: the number on the chip has to match
  // the number of cards clicking it produces, or the copy-fold reads as files
  // going missing.
  const collapsedCounts = q.data?.counts_by_ext_collapsed ?? counts;
  const totalAll = Object.values(counts).reduce((a, b) => a + b, 0);
  const totalEntries = Object.values(collapsedCounts).reduce((a, b) => a + b, 0);
  const nOther = otherCount(collapsedCounts);
  const matched = q.data?.total_matched ?? scans.length;
  const collapsed = q.data?.total_collapsed ?? scans.length;

  const resetPaging = () => {
    setOffset(0);
    setAcc([]);
  };

  // A selection the current filter hides must not keep driving the preview —
  // otherwise the highlighted card is invisible and the preview looks stuck.
  const effective =
    (selected && scans.some((s) => s.path === selected) ? selected : null) ||
    scans[0]?.path ||
    null;
  const effectiveEntry = scans.find((s) => s.path === effective) ?? null;

  // Grouped, not just sorted. See lib/scanKinds.groupScans: the source-side
  // exclusion is a blocklist and has now been outrun twice; a heading per kind
  // is what keeps scans visible when the NEXT telemetry writer appears.
  const groups = groupScans(scans);

  const chips: { id: string; label: string }[] = [
    { id: "all", label: `全部 (${totalEntries})` },
    // Keep a chip for the ACTIVE filter even when its count is 0, so the
    // operator has something highlighted to click away from.
    ...(kind !== "all" && kind !== "other" && !collapsedCounts[kind]
      ? [{ id: kind, label: `${kind} (0)` }]
      : []),
    ...KIND_EXTS.filter((e) => collapsedCounts[e]).map((e) => ({
      id: e as string,
      label: `${e.replace(".", "").toUpperCase()} (${collapsedCounts[e]})`,
    })),
    ...(nOther || kind === "other" ? [{ id: "other", label: `其他 (${nOther})` }] : []),
  ];

  const preview = effective && (
    <ScanPreviewPanel
      entry={effectiveEntry}
      path={effective}
      flatten={flatten}
      onFlattenChange={setFlatten}
    />
  );

  const cardGrid = (files: ScanFileEntry[]) => (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-3 xl:grid-cols-4">
      {files.map((s) => (
        <ScanGridCard
          key={s.path}
          entry={s}
          selected={effective === s.path}
          // The grid never asks for `auto`: it measures each frame (~3 s the
          // first time) and a screen of cards would queue thirty measurements.
          flatten={flatten === "auto" ? "line" : flatten}
          onSelect={() => setSelected(s.path)}
        />
      ))}
    </div>
  );

  const textRows = (files: ScanFileEntry[]) => (
    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-2">
      {files.map((s) => (
        <button
          key={s.path}
          onClick={() => setSelected(s.path)}
          className={
            "rounded-md border p-2 text-left text-xs transition-colors " +
            (effective === s.path
              ? "border-mast-accent bg-mast-accent/10"
              : "border-mast-border hover:bg-mast-bg/60")
          }
        >
          <div className="flex items-baseline gap-1">
            <div className="truncate font-medium text-mast-text" title={s.path}>
              {s.name}
            </div>
            {/* Which session folder — Nanonis reuses names across sessions, so
                two rows can read `unnamed0033.sxm` and be different scans. */}
            {parentFolder(s.path) && (
              <span className="shrink-0 text-mast-faint">{parentFolder(s.path)}</span>
            )}
            {copyBadge(s) && (
              <span className="ml-auto shrink-0 text-mast-muted" title="磁盘上有多个副本">
                {copyBadge(s)}
              </span>
            )}
          </div>
          <div className="mt-1 flex justify-between text-mast-muted tabular-nums">
            <span className="uppercase">{s.ext.replace(".", "")}</span>
            <span>{fmtBytes(s.size_bytes)}</span>
          </div>
          <div className="text-mast-muted tabular-nums">{fmtMtime(s.mtime)}</div>
        </button>
      ))}
    </div>
  );

  return (
    <Section
      title={`扫描数据${totalAll ? ` (${scans.length}/${collapsed})` : ""}`}
      actions={
        <div className="flex items-center gap-2">
          {/* 全量数据的预处理、标记与系列在「数据图库」那一段；这一页只看此刻磁盘上的最新文件。 */}
          <Link
            to="/records/gallery"
            className="whitespace-nowrap text-xs text-mast-accent hover:underline"
            title="全部数据的缩略图、自动判据、键盘打分、标签、系列与谱↔前后帧对照"
          >
            数据图库（预处理·标记·系列）→
          </Link>
          <div className="inline-flex overflow-hidden rounded-mast-ctl border border-mast-border">
            {VIEWS.map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={
                  "px-2.5 py-1 text-xs " +
                  (view === v
                    ? "bg-mast-accent-soft font-medium text-mast-accent"
                    : "text-mast-muted hover:text-mast-text")
                }
              >
                {VIEW_LABEL[v]}
              </button>
            ))}
          </div>
          <Button
            variant="primary"
            onClick={() => {
              resetPaging();
              q.refetch();
              setSelected(null);
            }}
          >
            Load Latest Scan
          </Button>
        </div>
      }
    >
      {/* Type filter. The 数据 tab mixes .sxm topography with .dat spectra and
          whatever generic numeric text happens to sit in a search dir; without
          this the newest thing wins the whole screen. */}
      {chips.length > 2 && (
        <div className="mb-2 flex flex-wrap items-center gap-1">
          {chips.map((c) => (
            <button
              key={c.id}
              onClick={() => {
                setKind(c.id);
                resetPaging();
              }}
              className={
                "rounded-mast-badge border px-2 py-0.5 text-xs " +
                (kind === c.id
                  ? "border-mast-accent bg-mast-accent-soft font-medium text-mast-accent"
                  : "border-mast-border text-mast-muted hover:text-mast-text")
              }
            >
              {c.label}
            </button>
          ))}
          {/* Thumbnail processing, for every card at once. The panel on the
              right has its own switch including 自动; this one is what the grid
              renders with. */}
          {view !== "list" && (
            <span className="ml-auto flex items-center gap-1 text-xs text-mast-muted">
              缩略图
              <span className="inline-flex overflow-hidden rounded-mast-ctl border border-mast-border">
                {GRID_FLATTEN_MODES.map((m) => (
                  <button
                    key={m}
                    onClick={() => setFlatten(m)}
                    className={
                      "px-2 py-0.5 " +
                      (flatten === m
                        ? "bg-mast-accent-soft font-medium text-mast-accent"
                        : "text-mast-muted hover:text-mast-text")
                    }
                  >
                    {flattenLabel(m)}
                  </button>
                ))}
              </span>
            </span>
          )}
        </div>
      )}
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="扫描数据" />}
      {q.data?.search_dirs && q.data.search_dirs.length > 0 && (
        <div className="mb-2 text-xs text-mast-muted">
          搜索目录：{q.data.search_dirs.join(" · ")}
        </div>
      )}

      {/* OLD 数据 empty state (build_data_empty_html) — icon + title + desc. */}
      {q.data && !q.data.degraded && totalAll === 0 && (
        <div className="rounded-md border border-mast-border bg-mast-bg/40 px-4 py-8 text-center">
          <div className="text-3xl">📡</div>
          <div className="mt-2 font-medium text-mast-text">No Scan Data</div>
          <div className="mt-1 text-sm text-mast-muted">
            Scan files (.sxm, .dat, .3ds) will appear here automatically.
          </div>
        </div>
      )}

      {totalAll > 0 && scans.length === 0 && (
        <EmptyNote
          label={`当前筛选（${kind}）下没有文件；共 ${totalEntries} 项。点「全部」看所有。`}
        />
      )}

      {view === "groups" ? (
        <ScanGroupBrowser
          scans={scans}
          scansDegraded={q.data?.degraded ?? false}
          flatten={flatten === "auto" ? "line" : flatten}
          selected={effective}
          onSelect={setSelected}
        />
      ) : (
        scans.length > 0 && (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,440px)]">
            {/* Preview first in the DOM on narrow screens (order-first), second
                on wide ones — so it is never below a long list either way. */}
            {/* Sticky at EVERY width, not only >= lg. #60 made this a sticky
                right-hand column, but the `lg:` prefix meant that below 1024 px
                the layout collapsed to one column with NO sticky at all — so on a
                laptop lid the operator still scrolled the preview off the top of
                the screen to reach the list, which is verbatim.
                `max-h`+`overflow` on the wrapper keeps a tall preview from eating
                the whole viewport on a short window. */}
            <div className="order-first max-h-[52vh] overflow-y-auto sticky top-0 z-10 bg-mast-panel pb-2 lg:order-last lg:max-h-none lg:overflow-visible lg:top-4 lg:self-start">
              <div className="mb-1 text-xs font-medium text-mast-muted">预览</div>
              {preview}
            </div>

            <div className="max-h-[70vh] overflow-y-auto pr-1">
              {groups.map(({ group, files }) => (
                <div key={group.id} className="mb-3 last:mb-0">
                  {groups.length > 1 && (
                    <div
                      title={group.hint}
                      className="mb-1 flex items-baseline gap-2 text-xs font-medium text-mast-muted"
                    >
                      {group.label}
                      <span className="font-normal text-mast-faint tabular-nums">
                        {files.length}
                      </span>
                    </div>
                  )}
                  {view === "grid" ? cardGrid(files) : textRows(files)}
                </div>
              ))}

              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-mast-faint">
                <span>{pageSummary(scans.length, collapsed, matched)}</span>
                {scans.length < collapsed && (
                  <Button
                    onClick={() => setOffset(scans.length)}
                    disabled={q.isFetching}
                  >
                    {q.isFetching ? "加载中…" : "加载更多"}
                  </Button>
                )}
              </div>
            </div>
          </div>
        )
      )}
    </Section>
  );
}
