import { useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, Spinner, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";
import { Sparkline } from "@/components/vision/Sparkline";
import { Button } from "@/components/controls";

// 视觉缓冲 — full Vision-Buffer event panel. Typed replacement for
// gui/vision_buffer.py: stats bar + events/s sparkline, kind checkbox filter,
// severity & time-range dropdowns, pause/refresh, color-coded event list with
// expandable payload JSON + thumbnail stub, CSV/JSON export.
// Reads GET /api/vision/buffer (events + stats). Pure client-side filtering.

type BufferEvent = {
  event_id: string;
  seqno: number;
  kind: string;
  severity: string;
  payload?: Record<string, unknown>;
  cause_ref?: string | null;
  t_mono_ns: number;
  t_wall: number;
};

// kind → CN label + color bucket (mirrors _KIND_META).
const KIND_META: { kind: string; cn: string; bucket: "info" | "warn" | "crit" | "set" }[] = [
  { kind: "tip_quality_drop", cn: "针尖质量下降", bucket: "warn" },
  { kind: "feature_of_interest", cn: "特征发现", bucket: "info" },
  { kind: "scan_complete", cn: "扫描完成", bucket: "info" },
  { kind: "sensor_fault", cn: "传感器故障", bucket: "crit" },
  { kind: "emergency_retract_needed", cn: "紧急退针", bucket: "crit" },
  { kind: "vision_error", cn: "视觉模型错误", bucket: "warn" },
  { kind: "setpoint_change", cn: "设定点变更", bucket: "set" },
  { kind: "e_stop", cn: "急停", bucket: "crit" },
  { kind: "tip_shape_verdict", cn: "针尖整形", bucket: "info" },
];
const KIND_CN: Record<string, string> = Object.fromEntries(KIND_META.map((m) => [m.kind, m.cn]));

const RANGE_OPTIONS: { label: string; secs: number | null }[] = [
  { label: "最近 1 分钟", secs: 60 },
  { label: "最近 5 分钟", secs: 300 },
  { label: "最近 15 分钟", secs: 900 },
  { label: "最近 1 小时", secs: 3600 },
  { label: "全部", secs: null },
];

const SEV_OPTIONS: { label: string; value: string }[] = [
  { label: "全部", value: "all" },
  { label: "INFO", value: "info" },
  { label: "WARN", value: "warn" },
  { label: "CRITICAL", value: "critical" },
];

function edgeColor(kind: string, severity: string, payload?: Record<string, unknown>): string {
  const sev = (severity || "").toLowerCase();
  if (sev === "critical") return "var(--mast-danger)";
  // SAFE mode demotes a tip verdict to a record-only INFO. Colour it as the
  // information it now is — otherwise the kind's own bucket ("针尖质量下降" =
  // warn) paints a yellow edge on an event that is explicitly not asking for
  // anything, and the panel contradicts the badge next to it.
  if (payload?.["safe_mode_suppressed"]) return "var(--mast-accent)";
  const bucket = KIND_META.find((m) => m.kind === kind)?.bucket ?? "info";
  if (bucket === "crit") return "var(--mast-danger)";
  if (bucket === "warn") return "var(--mast-warn)";
  if (bucket === "set") return "var(--mast-warn)";
  return "var(--mast-accent)";
}

function severityCn(severity: string): string {
  return (
    { info: "信息", warn: "警告", warning: "警告", critical: "严重" }[
      (severity || "").toLowerCase()
    ] || severity || "?"
  );
}
function sevBadgeCls(severity: string): string {
  const sev = (severity || "").toLowerCase();
  if (sev === "critical") return "bg-mast-danger-bg text-mast-danger";
  if (sev === "warn" || sev === "warning") return "bg-mast-warn-bg text-mast-warn";
  return "bg-mast-info-bg text-mast-info";
}

function fmtWall(tWall: number): string {
  if (!tWall) return "—";
  try {
    const d = new Date(tWall * 1000);
    const hms = d.toLocaleTimeString();
    return `${hms}.${String(d.getMilliseconds()).padStart(3, "0")}`;
  } catch {
    return "—";
  }
}

function shortRepr(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toPrecision(4);
  const s = typeof v === "string" || typeof v === "boolean" ? String(v) : JSON.stringify(v);
  return s.length <= 80 ? s : s.slice(0, 77) + "…";
}
// The human-readable verdict fields. These are SENTENCES written for the
// operator ("反馈振荡/振铃（55 周期/行）——50% 处,建议降增益"), and squeezing them
// through shortRepr in a 4-key summary cut them mid-word with no way to read the
// rest (—「这里显示不全而且没有展开功能」). They get their own line, in
// full; everything else stays in the compact key list.
// `summary_zh` is the key scan_monitor actually writes; the rest are what other
// publishers use. Verified against the producers, not guessed — the backend had
// a sibling bug where it read `advice`/`summary`, which nothing writes.
const NARRATIVE_KEYS = [
  "summary_zh", "advice", "summary", "message", "reason", "detail",
] as const;

function narrative(payload?: Record<string, unknown>): string {
  if (!payload) return "";
  const parts: string[] = [];
  for (const k of NARRATIVE_KEYS) {
    const v = payload[k];
    if (typeof v === "string" && v.trim()) parts.push(v.trim());
  }
  return parts.join(" ");
}

function inlinePayload(payload?: Record<string, unknown>): string {
  if (!payload) return "";
  const keys = Object.keys(payload).filter(
    (k) => !(NARRATIVE_KEYS as readonly string[]).includes(k) && k !== "frame_path",
  );
  const items = keys.slice(0, 4).map((k) => `${k}=${shortRepr(payload[k])}`);
  if (keys.length > 4) items.push(`… (+${keys.length - 4} more)`);
  return items.join(" · ");
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function csvCell(v: unknown): string {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function eventsToCsv(events: BufferEvent[]): string {
  const head = ["t_iso", "seqno", "kind", "severity", "cause_ref", "event_id", "payload_json"];
  const lines = [head.join(",")];
  for (const e of events) {
    lines.push(
      [
        new Date(e.t_wall * 1000).toISOString(),
        e.seqno,
        e.kind,
        e.severity,
        e.cause_ref ?? "",
        e.event_id,
        JSON.stringify(e.payload ?? {}),
      ]
        .map(csvCell)
        .join(","),
    );
  }
  return lines.join("\n");
}
function eventsToJson(events: BufferEvent[]): string {
  return JSON.stringify(
    events.map((e) => ({
      t_iso: new Date(e.t_wall * 1000).toISOString(),
      seqno: e.seqno,
      kind: e.kind,
      severity: e.severity,
      cause_ref: e.cause_ref,
      event_id: e.event_id,
      payload: e.payload ?? {},
    })),
    null,
    2,
  );
}

function StatChip({ label, value, alert }: { label: string; value: string | number; alert?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-xs text-mast-muted">{label}</span>
      <span className={`text-sm font-medium tabular-nums ${alert ? "text-mast-danger" : "text-mast-text"}`}>
        {value}
      </span>
    </span>
  );
}

function EventRow({ ev }: { ev: BufferEvent }) {
  const [open, setOpen] = useState(false);
  const [noFrame, setNoFrame] = useState(false);
  const filePath =
    (ev.payload?.["file_path"] as string | undefined) ?? (ev.payload?.["path"] as string | undefined);
  // The PNG of the frame THIS event judged. Rendered lazily straight from
  // /api/vision/event-frame/{id} — the row used to draw a 24×24 dashed stub
  // here, which is why every pulse read "建议降增益" with nothing to look at
  // (). A missing frame says so; it never borrows another image.
  const hasFrame = Boolean(ev.payload?.["frame_path"]) && !noFrame;
  return (
    <div
      className="rounded-md border border-mast-border bg-mast-bg/40"
      style={{ borderLeft: `3px solid ${edgeColor(ev.kind, ev.severity, ev.payload)}` }}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-3 py-2 text-sm">
        <span className="font-mono text-xs text-mast-muted tabular-nums">{fmtWall(ev.t_wall)}</span>
        <span className="font-mono text-xs">{ev.kind}</span>
        <span className="text-mast-muted">{KIND_CN[ev.kind] ?? ev.kind}</span>
        <span className={`rounded px-1.5 py-0.5 text-xs ${sevBadgeCls(ev.severity)}`}>
          {severityCn(ev.severity)}
        </span>
        <span className="font-mono text-xs text-mast-muted">#{ev.seqno}</span>
        {Boolean(ev.payload?.["safe_mode_suppressed"]) && (
          <span
            className="rounded bg-mast-warn-bg px-1.5 py-0.5 text-xs text-mast-warn"
            title="安全模式：本条针尖判定已被降级为仅记录，不中止实验、不请求修针。原始判定见展开的 payload。"
          >
            安全模式抑制
          </span>
        )}
        {ev.cause_ref && (
          <span className="font-mono text-xs text-mast-muted" title="cause_ref · 视觉缓冲游标">
            ↗ {ev.cause_ref}
          </span>
        )}
      </div>
      {narrative(ev.payload) && (
        <div className="whitespace-pre-wrap break-words px-3 pb-1.5 text-[13px] leading-relaxed text-mast-text">
          {narrative(ev.payload)}
        </div>
      )}
      {inlinePayload(ev.payload) && (
        <div className="px-3 pb-1.5 font-mono text-xs text-mast-muted">{inlinePayload(ev.payload)}</div>
      )}
      {hasFrame && (
        <div className="px-3 pb-2">
          <img
            src={`/api/vision/event-frame/${encodeURIComponent(ev.event_id)}`}
            alt={`事件 #${ev.seqno} 判定时的画面`}
            loading="lazy"
            onError={() => setNoFrame(true)}
            className="max-h-64 w-auto max-w-full rounded border border-mast-border"
          />
        </div>
      )}
      {filePath && (
        <div className="flex items-center gap-2 px-3 pb-1.5 text-xs">
          <span className="truncate font-mono text-mast-muted" title={filePath}>
            {filePath.split(/[\\/]/).pop()}
          </span>
        </div>
      )}
      <div className="px-3 pb-2">
        <button
          onClick={() => setOpen((o) => !o)}
          className="text-xs text-mast-accent hover:underline"
        >
          {open ? "收起 Payload JSON" : "展开 Payload JSON"}
        </button>
        {open && (
          <pre className="mt-1 max-h-64 overflow-auto rounded bg-mast-bg p-2 font-mono text-xs text-mast-text">
            {JSON.stringify(ev.payload ?? {}, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

// Bucket events into events/s rates over the last 60s for the sparkline.
function bucketRates(events: BufferEvent[], windowS = 60, nBins = 40): number[] {
  if (!events.length) return [];
  const now = Date.now() / 1000;
  const binS = windowS / nBins;
  const counts = new Array(nBins).fill(0);
  for (const ev of events) {
    const age = now - ev.t_wall;
    if (age < 0 || age > windowS) continue;
    const b = nBins - 1 - Math.floor(age / binS);
    if (b >= 0 && b < nBins) counts[b] += 1;
  }
  return counts.map((c) => c / binS);
}

export function VisionBufferPanel() {
  const [paused, setPaused] = useState(false);
  const [activeKinds, setActiveKinds] = useState<string[]>(KIND_META.map((m) => m.kind));
  const [severity, setSeverity] = useState("all");
  const [rangeS, setRangeS] = useState<number | null>(300);

  const q = useQuery({
    queryKey: ["vision", "buffer"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/vision/buffer", {
        params: { query: { since: -1, limit: 500 } },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: paused ? false : 2000,
  });

  // Hold last good payload across refetch so paused/filtering stays stable.
  const lastRef = useRef<typeof q.data | null>(null);
  if (q.data) lastRef.current = q.data;
  const data = q.data ?? lastRef.current;

  const allEvents = (data?.events ?? []) as BufferEvent[];
  const stats = data?.stats;

  const filtered = useMemo(() => {
    const now = Date.now() / 1000;
    const kindSet = new Set(activeKinds);
    return allEvents.filter((ev) => {
      if (!kindSet.has(ev.kind)) return false;
      if (severity !== "all" && (ev.severity || "").toLowerCase() !== severity) return false;
      if (rangeS != null && now - ev.t_wall > rangeS) return false;
      return true;
    });
  }, [allEvents, activeKinds, severity, rangeS]);

  const rates = useMemo(() => bucketRates(allEvents), [allEvents]);

  const toggleKind = (k: string) =>
    setActiveKinds((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]));

  const dropped = stats?.events_dropped_oldest ?? 0;

  return (
    <div className="space-y-3">
      <p className="text-xs text-mast-muted">
        视觉模型（DINOv3 / 旧 ResNet18+UNet+DQN）推理后写入 BufferService 的所有事件 —— 轮询 2 秒。
        事件类型与 payload schema 见 <code>mast.buffer.schemas</code>。
      </p>

      {q.isPending && !data && <Spinner />}
      {q.isError && !data && <ErrorNote error={q.error} />}
      {data?.degraded && <DegradedNote what="视觉缓冲" />}

      {data && !data.degraded && (
        <>
          {/* Stats bar + sparkline */}
          <Card className="flex flex-wrap items-center gap-x-5 gap-y-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs ${
                paused ? "bg-mast-bg text-mast-muted" : "bg-mast-auto-bg text-mast-auto"
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${paused ? "bg-mast-muted" : "bg-mast-auto"}`} />
              {paused ? "已暂停" : "实时"}
            </span>
            <StatChip label="已发布" value={stats?.events_published ?? 0} />
            <StatChip label="丢弃 oldest" value={dropped} alert={dropped > 0} />
            <StatChip label="扇出失败" value={stats?.events_fanout_failed ?? 0} />
            <StatChip label="WAL" value={stats?.wal_event_writes ?? 0} />
            <StatChip label="订阅者" value={stats?.subscribers_active ?? 0} />
            <StatChip label="显示" value={`${filtered.length} / ${allEvents.length}`} />
            <span className="ml-auto flex items-center gap-2">
              <span className="text-xs text-mast-muted">events/s · 最近 60 s</span>
              <Sparkline
                values={rates}
                color={dropped > 0 ? "var(--mast-danger)" : "var(--mast-accent)"}
                width={200}
                height={28}
                fill={false}
              />
            </span>
          </Card>

          {/* Filter row: event-kind checkboxes */}
          <Card className="space-y-3">
            <div>
              <div className="mb-1.5 text-xs text-mast-muted">事件类型</div>
              <div className="flex flex-wrap gap-2">
                {KIND_META.map((m) => (
                  <label
                    key={m.kind}
                    className="inline-flex cursor-pointer items-center gap-1.5 rounded border border-mast-border px-2 py-1 text-xs"
                  >
                    <input
                      type="checkbox"
                      checked={activeKinds.includes(m.kind)}
                      onChange={() => toggleKind(m.kind)}
                    />
                    <span style={{ color: edgeColor(m.kind, "") }}>●</span>
                    {m.cn}
                    <span className="font-mono text-mast-muted">{m.kind}</span>
                  </label>
                ))}
              </div>
            </div>
            <div className="flex flex-wrap items-end gap-3">
              <label className="flex flex-col gap-1 text-xs">
                <span className="text-mast-muted">严重度</span>
                <select
                  value={severity}
                  onChange={(e) => setSeverity(e.target.value)}
                  className="rounded-md border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
                >
                  {SEV_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs">
                <span className="text-mast-muted">时间范围</span>
                <select
                  value={String(rangeS)}
                  onChange={(e) => setRangeS(e.target.value === "null" ? null : Number(e.target.value))}
                  className="rounded-md border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
                >
                  {RANGE_OPTIONS.map((o) => (
                    <option key={String(o.secs)} value={String(o.secs)}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
              <Button onClick={() => setPaused((p) => !p)}>{paused ? "▶ 继续" : "⏸ 暂停"}</Button>
              <Button onClick={() => q.refetch()}>立即刷新</Button>
              <Button onClick={() => setActiveKinds(KIND_META.map((m) => m.kind))}>重置筛选</Button>
              <div className="ml-auto flex gap-2">
                <Button
                  onClick={() => downloadText(`vision_buffer_${Date.now()}.csv`, eventsToCsv(filtered))}
                  disabled={!filtered.length}
                >
                  导出 CSV
                </Button>
                <Button
                  onClick={() => downloadText(`vision_buffer_${Date.now()}.json`, eventsToJson(filtered))}
                  disabled={!filtered.length}
                >
                  导出 JSON
                </Button>
              </div>
            </div>
          </Card>

          {/* Event list (newest first) */}
          {filtered.length === 0 ? (
            <EmptyNote label="尚无符合筛选条件的事件。视觉模型写入 BufferService 后会自动出现于此。" />
          ) : (
            <div className="space-y-2">
              {[...filtered].reverse().map((ev) => (
                <EventRow key={ev.event_id || ev.seqno} ev={ev} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
