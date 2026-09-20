import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Badge, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, useToast } from "@/components/controls";
import { TimeTraceChart } from "@/components/experimental/TimeTraceChart";
import { FftChart } from "@/components/vision/FftChart";
import { fmtBytes, fmtSeconds, verdictView } from "@/lib/monitoring";

type SegmentRow = components["schemas"]["SegmentRow"];

// 段浏览器 — the corpus side of the page. Every second of recorded current is a
// row here; pinning exempts one from the retention sweep and labelling records
// a human verdict on it (which pins it too, backend-side: a judged segment with
// its waveform swept away would be a label with nothing to train on).

const PAGE = 25;

const FILTERS = [
  { key: "all", label: "全部" },
  { key: "pinned", label: "已钉住" },
  { key: "good", label: "好针尖" },
  { key: "bad", label: "坏针尖" },
  { key: "unlabeled", label: "未标注" },
] as const;

type FilterKey = (typeof FILTERS)[number]["key"];

function filterParams(f: FilterKey): { pinned?: boolean; label?: string } {
  if (f === "pinned") return { pinned: true };
  if (f === "unlabeled") return { label: "unlabeled" };
  if (f === "good" || f === "bad") return { label: f };
  return {};
}

function LabelBadge({ label }: { label: string | null | undefined }) {
  if (!label) return null;
  if (label === "good") return <Badge tone="AUTO">好针尖</Badge>;
  if (label === "bad") return <Badge tone="DANGEROUS">坏针尖</Badge>;
  return <Badge tone="INFO">{label}</Badge>;
}

// ── detail pane ─────────────────────────────────────────────────────────────

function SegmentDetail({ segId }: { segId: number }) {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const [note, setNote] = useState("");

  const q = useQuery({
    queryKey: ["monitoring", "segment-data", segId],
    // A segment is immutable once written — the only thing that ever changes is
    // whether the .npy still exists, and the retention sweep is slow enough that
    // a refetch on mount is not worth the decimation work on the backend.
    staleTime: Infinity,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/segments/{seg_id}/data", {
        params: { path: { seg_id: segId }, query: { max_points: 4000, include_psd: true } },
      });
      if (error) throw error;
      return data;
    },
  });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["monitoring", "segments"] });
    qc.invalidateQueries({ queryKey: ["monitoring", "status"] });
  };

  const pin = useMutation({
    mutationFn: async (pinned: boolean) => {
      const { data, error } = await api.POST("/api/monitoring/segments/{seg_id}/pin", {
        params: { path: { seg_id: segId } },
        body: { pinned, reason: "manual" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      toast(d?.pinned ? "已钉住，不会被保留策略清理" : "已取消钉住", d?.degraded ? "err" : "ok");
      invalidate();
    },
    onError: (e) => toast(`操作失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const label = useMutation({
    mutationFn: async (value: string | null) => {
      const { data, error } = await api.POST("/api/monitoring/segments/{seg_id}/label", {
        params: { path: { seg_id: segId } },
        body: { label: value, note: note || null },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      toast(d?.label ? `已标注为「${d.label}」并自动钉住` : "已清除标注", d?.degraded ? "err" : "ok");
      setNote("");
      invalidate();
    },
    onError: (e) => toast(`标注失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  if (q.isPending) return <Spinner label="载入段数据…" />;
  if (q.isError) return <ErrorNote error={q.error} />;

  const d = q.data;
  if (!d?.ok) {
    return <EmptyNote label={d?.detail || "该段数据不可用。"} />;
  }
  const meta = d.meta;
  const envelope = d.source === "envelope";
  const swept = meta ? !meta.has_file : false;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm text-mast-text">第 {d.seg_id} 段</span>
        {meta && <Badge tone={verdictView(meta.verdict).tone}>{verdictView(meta.verdict).label}</Badge>}
        {meta?.pinned && <Badge tone="INFO">已钉住</Badge>}
        <LabelBadge label={meta?.label} />
        <span className="text-xs text-mast-muted">
          {new Date(d.t0 * 1000).toLocaleString("zh-CN", { hour12: false })} · {d.fs_hz.toFixed(0)} Hz ·{" "}
          {d.n_samples_raw} 点
          {meta?.gap_s ? ` · 前置缺口 ${fmtSeconds(meta.gap_s)}` : ""}
        </span>
      </div>

      {(envelope || swept) && (
        <div className="rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2 text-xs text-mast-warn">
          原始数据已按保留策略清理，当前显示的是<strong>包络</strong>（每桶最小/最大值）。
          形状与尖峰仍然可信，但细节与频谱不再可得。钉住的段不会被清理。
        </div>
      )}

      <TimeTraceChart
        timestampsS={d.t_s ?? []}
        samples={d.i_a ?? []}
        unit="A"
        channelName={meta?.channel_name || "Current"}
      />

      {d.psd ? (
        <div>
          <div className="mb-1 text-sm font-medium text-mast-text">功率谱</div>
          <FftChart fft={d.psd} logY logX={false} dropDc />
        </div>
      ) : (
        <p className="text-xs text-mast-muted">
          {envelope ? "包络无法反演频谱，故不显示。" : "该段未提供频谱。"}
        </p>
      )}

      <div className="rounded-mast-card border border-mast-border bg-mast-panel-2 p-3">
        <div className="mb-2 text-sm font-medium text-mast-text">人工判定</div>
        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="备注（可选）：当时在做什么、为什么这么判"
          className="mb-2 w-full rounded-mast-ctl border border-mast-border bg-mast-bg px-2 py-1.5 text-sm text-mast-text placeholder:text-mast-faint"
        />
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" onClick={() => label.mutate("good")} loading={label.isPending}>
            好针尖
          </Button>
          <Button variant="danger" onClick={() => label.mutate("bad")} loading={label.isPending}>
            坏针尖
          </Button>
          <Button variant="ghost" onClick={() => label.mutate(null)} loading={label.isPending}>
            清除标注
          </Button>
          <span className="ml-auto">
            <Button onClick={() => pin.mutate(!meta?.pinned)} loading={pin.isPending}>
              {meta?.pinned ? "取消钉住" : "钉住此段"}
            </Button>
          </span>
        </div>
        {meta?.label_note && (
          <p className="mt-2 text-xs text-mast-muted">已有备注：{meta.label_note}</p>
        )}
      </div>
      {node}
    </div>
  );
}

// ── list + pane ─────────────────────────────────────────────────────────────

export function SegmentBrowser({ degraded }: { degraded: boolean }) {
  const [filter, setFilter] = useState<FilterKey>("all");
  const [offset, setOffset] = useState(0);
  // `?seg=` deep link (2026-08-04, ). A current-monitor event tile links
  // here to show the full waveform behind its thumbnail curve. The detail pane
  // fetches by id and does NOT read the list, so a segment that is not on the
  // current page still opens — which matters, because the linked one is usually
  // old enough to have paged out.
  const [params, setParams] = useSearchParams();
  const linked = Number(params.get("seg"));
  const [selected, setSelected] = useState<number | null>(
    Number.isSafeInteger(linked) && linked > 0 ? linked : null,
  );

  // The panel sits far below the fold on this page, so an arriving deep link
  // has to bring itself into view — otherwise the click looks like it did
  // nothing, which the house rule about dead links forbids just as much as a
  // freeze does.
  const rootRef = useRef<HTMLDivElement | null>(null);
  // Only for an ARRIVING link. Scrolling on every selection would yank the page
  // under an operator who is just clicking down the list.
  const arrived = useRef(selected != null);
  useEffect(() => {
    if (!arrived.current || !rootRef.current) return;
    arrived.current = false;
    rootRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const select = (id: number | null) => {
    setSelected(id);
    // Keep the URL honest so a refresh (or a shared link) lands on the same
    // segment, and drop the param when the operator navigates away from it.
    setParams(
      (p) => {
        const next = new URLSearchParams(p);
        if (id == null) next.delete("seg");
        else next.set("seg", String(id));
        return next;
      },
      { replace: true },
    );
  };

  const list = useQuery({
    queryKey: ["monitoring", "segments", filter, offset],
    refetchInterval: 15_000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/segments", {
        params: { query: { ...filterParams(filter), limit: PAGE, offset } },
      });
      if (error) throw error;
      return data;
    },
  });

  const segments: SegmentRow[] = list.data?.segments ?? [];
  const total = list.data?.total ?? 0;
  // 传进来的 `degraded` 说的是 **status** 端点；段索引是另一个端点，它有自己的。
  // 少了后半句，一次 `segments_query` 失败会画成「该筛选下没有段 · 共 0 段」——
  // 三个数都是正面断言，而真相是一条都没读到。
  const unreadable = degraded || (list.data?.degraded ?? false);

  const pick = (f: FilterKey) => {
    setFilter(f);
    setOffset(0);
    select(null);
  };

  return (
    <div className="space-y-3" ref={rootRef}>
      <div className="flex flex-wrap items-center gap-2">
        <div className="inline-flex overflow-hidden rounded-mast-ctl border border-mast-border">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => pick(f.key)}
              className={
                "px-2.5 py-1 text-xs " +
                (filter === f.key
                  ? "bg-mast-accent-soft font-medium text-mast-accent"
                  : "text-mast-muted hover:text-mast-text")
              }
            >
              {f.label}
            </button>
          ))}
        </div>
        <span className="text-xs text-mast-muted">
          {unreadable
            ? "段索引读不到"
            : `共 ${total} 段 · 已钉住 ${list.data?.pinned_count ?? 0} · 已标注 ${list.data?.labeled_count ?? 0}`}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        {/* left: list */}
        <div className="space-y-2">
          {unreadable ? (
            <EmptyNote
              label={
                list.data?.detail
                  ? `读不到段索引：${list.data.detail}`
                  : "读不到段索引（监控模块未装载，或查询失败）。"
              }
            />
          ) : list.isPending ? (
            <Spinner />
          ) : list.isError ? (
            <ErrorNote error={list.error} />
          ) : segments.length === 0 ? (
            <EmptyNote label="该筛选下没有段。" />
          ) : (
            <div className="max-h-[560px] overflow-y-auto rounded-mast-card border border-mast-border">
              {segments.map((s) => {
                const v = verdictView(s.verdict);
                return (
                  <button
                    key={s.seg_id}
                    onClick={() => select(s.seg_id)}
                    className={clsx(
                      "flex w-full flex-col items-start gap-0.5 border-b border-mast-border px-3 py-2 text-left last:border-b-0",
                      selected === s.seg_id ? "bg-mast-accent-soft" : "hover:bg-mast-panel-2",
                    )}
                  >
                    <span className="flex w-full items-center gap-2">
                      <span className="font-mono text-xs text-mast-text">#{s.seg_id}</span>
                      <Badge tone={v.tone}>{v.label}</Badge>
                      {s.pinned && <span className="text-xs text-mast-info">钉</span>}
                      <LabelBadge label={s.label} />
                      <span className="ml-auto text-[11px] text-mast-muted">
                        {new Date(s.t_start * 1000).toLocaleTimeString("zh-CN", { hour12: false })}
                      </span>
                    </span>
                    <span className="text-[11px] text-mast-faint">
                      {s.n_samples} 点 · {s.fs_hz.toFixed(0)} Hz ·{" "}
                      {s.has_file ? fmtBytes(s.file_bytes) : "仅包络"}
                      {s.discontinuity && " · 有断点"}
                    </span>
                  </button>
                );
              })}
            </div>
          )}

          <div className="flex items-center gap-2 text-xs">
            <Button onClick={() => setOffset(Math.max(0, offset - PAGE))} disabled={offset === 0}>
              上一页
            </Button>
            <Button
              onClick={() => setOffset(offset + PAGE)}
              disabled={offset + PAGE >= total}
            >
              下一页
            </Button>
            <span className="text-mast-muted">
              {total === 0 ? 0 : offset + 1}–{Math.min(offset + PAGE, total)} / {total}
            </span>
          </div>
        </div>

        {/* right: detail */}
        <div>
          {selected == null ? (
            <EmptyNote label="从左侧选一段查看波形、频谱，并做人工判定。" />
          ) : (
            <SegmentDetail key={selected} segId={selected} />
          )}
        </div>
      </div>
    </div>
  );
}
