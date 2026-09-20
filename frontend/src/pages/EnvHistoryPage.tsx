import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Section, Spinner } from "@/components/ui";
import { Button, Field, RadioGroup, SelectField } from "@/components/controls";
import { EnvSeriesChart } from "@/components/envhistory/EnvSeriesChart";
import { SpectrumViewer, type SpectrumLike } from "@/components/envhistory/SpectrumViewer";
import {
  RANGES,
  channelLabel,
  coverageOf,
  fmtBytes,
  fmtSpan,
  fmtTs,
  rangeWindow,
  spectrumPickerLabel,
  statusLabel,
  statusTone,
  worstOf,
} from "@/lib/envHistory";
import { ENV_HISTORY_SERIES_POLL_MS, ENV_HISTORY_STATUS_POLL_MS } from "@/lib/pollRates";
import { useStickyTab } from "@/hooks/useStickyTab";

type Status = components["schemas"]["EnvHistoryStatus"];
type Series = components["schemas"]["EnvSeriesResponse"];
type SpectraList = components["schemas"]["SpectraListResponse"];

// 环境历史 — the long-term record of the things that are true of the machine
// rather than of the measurement: temperature, vacuum, helium, field, the
// current the tip sits at while nobody is driving it, and the noise spectra.
//
// ── where each number comes from ───────────────────────────────────────────
// Nothing on this page touches hardware. The recorder aggregates the existing
// 2-second environment stream into permanent one-minute buckets and archives a
// spectrum every half hour off the current monitor's segment stream; these
// endpoints read that SQLite store and nothing else. Watching a year of
// temperature costs the instrument zero packets.
//
// ── what the curves mean ───────────────────────────────────────────────────
// Every point is a bucket, drawn as a mean line inside its true min/max band.
// When the server merges buckets for a wide window it re-aggregates by weight —
// the extremes and the worst status survive the merge, so zooming out can never
// hide an excursion that really happened.
//
// 隧道电流 is gated: only readings taken while nothing was driving the
// instrument are counted, because during a scan that signal is topography, not
// environment. The gaps are real and `instrument_quiet` explains them.

const STATUS_KEY = ["env-history", "status"] as const;

/**
 * A series' unit, from the sensor list.
 *
 * Used to decide what may be overlaid on what: only same-unit series are
 * offered as a comparison, because they are the only ones that can honestly
 * share one Y axis.
 */
function unitOf(list: { sensors?: { sensor: string; unit?: string | null }[] } | undefined,
                name: string): string {
  return list?.sensors?.find((s) => s.sensor === name)?.unit ?? "";
}

/** Series the operator most likely wants first, in order of preference. */
const PREFERRED = ["temperature", "vacuum", "helium_level", "tunnel_current"];

type View = "series" | "spectra";

const VIEWS: { value: View; label: string }[] = [
  { value: "series", label: "趋势" },
  { value: "spectra", label: "噪声谱" },
];

export default function EnvHistoryPage() {
  // 噪声谱是「看一眼别处再回来接着看」的典型,每次回来弹回「趋势」
  // 等于每次重选。这一页的切换器是 RadioGroup 不是 SubTabs,记忆机制无关。
  const [view, setView] = useStickyTab<View>(
    "env-history.view", VIEWS.map((v) => v.value), "series");
  const [sensor, setSensor] = useState<string>("");
  // 对比序列。SPM 和 Magnet 不是同时保存——
  // 后端一直是**一个 tick 读全部传感器、一个时间基准落库**，两条曲线本来就该
  // 落在同一个桶格上。但这一页一次只画一条，用户只能来回切换看，
  // 而来回切换看到的正是「两段互不重叠的数据」——那是 COM13 句柄争用的表征，
  // 不是排程。叠起来才看得见它们同不同时。
  const [compareSensor, setCompareSensor] = useState<string>("");
  // 存的是**跨度本身**，不是 RANGES 里的下标。加 48 小时那一档时这里原本写的是
  // `useState(1) // 24 小时` —— 一个由数组行序承重的默认值，往中间插一项就会静默
  // 改掉默认档，而注释仍然理直气壮地写着旧标签。`null` 是「全部」，是合法取值。
  const [rangeS, setRangeS] = useState<number | null>(86400);
  const [channel, setChannel] = useState<string>("current");
  const [primaryId, setPrimaryId] = useState<number | null>(null);
  const [compareId, setCompareId] = useState<number | null>(null);

  const status = useQuery({
    queryKey: STATUS_KEY,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/env-history/status");
      if (error) throw error;
      return data as Status;
    },
    refetchInterval: ENV_HISTORY_STATUS_POLL_MS,
  });

  const sensors = useQuery({
    queryKey: ["env-history", "sensors"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/env-history/sensors");
      if (error) throw error;
      return data;
    },
    refetchInterval: ENV_HISTORY_SERIES_POLL_MS,
  });

  const names = useMemo(
    () => (sensors.data?.sensors ?? []).map((s) => s.sensor).filter(Boolean),
    [sensors.data],
  );

  // Pick a default series once the list arrives, preferring the ones an
  // operator actually opens this page for.
  useEffect(() => {
    if (sensor || !names.length) return;
    const pick = PREFERRED.find((p) => names.includes(p)) ?? names[0];
    if (pick) setSensor(pick);
  }, [names, sensor]);

  const window = useMemo(() => rangeWindow(rangeS, Date.now()), [rangeS]);

  const series = useQuery({
    queryKey: ["env-history", "series", sensor, window.since ?? "all"],
    enabled: !!sensor,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/env-history/series", {
        params: { query: { sensor, since: window.since, max_points: 2000 } },
      });
      if (error) throw error;
      return data as Series;
    },
    refetchInterval: ENV_HISTORY_SERIES_POLL_MS,
  });

  // 同一个窗口、同一套参数取第二条 —— 只有这样两条曲线才可比。
  const compareSeries = useQuery({
    queryKey: ["env-history", "series", compareSensor, window.since ?? "all"],
    enabled: !!compareSensor && compareSensor !== sensor,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/env-history/series", {
        params: { query: { sensor: compareSensor, since: window.since, max_points: 2000 } },
      });
      if (error) throw error;
      return data as Series;
    },
    refetchInterval: ENV_HISTORY_SERIES_POLL_MS,
  });

  const spectra = useQuery({
    queryKey: ["env-history", "spectra", channel],
    enabled: view === "spectra",
    queryFn: async () => {
      const { data, error } = await api.GET("/api/env-history/spectra", {
        params: { query: { channel, limit: 200 } },
      });
      if (error) throw error;
      return data as SpectraList;
    },
    refetchInterval: ENV_HISTORY_SERIES_POLL_MS,
  });

  const metas = spectra.data?.spectra ?? [];
  useEffect(() => {
    if (!metas.length) {
      setPrimaryId(null);
      return;
    }
    const newest = metas[0];
    if (newest && (primaryId == null || !metas.some((m) => m.id === primaryId))) {
      setPrimaryId(newest.id); // newest first from the backend
    }
  }, [metas, primaryId]);

  const primary = useSpectrum(primaryId);
  const compare = useSpectrum(compareId);

  const st = status.data;
  const degraded = !!st?.degraded;
  const points = series.data?.points ?? [];
  const worst = worstOf(points);
  const cover = coverageOf(points);
  const unit = series.data?.unit ?? "";
  // 只提供**同单位**的对比项。开尔文对开尔文可比，开尔文对帕斯卡不可比，
  // 而后者要么得配第二根 Y 轴（相对高低由画图人定，会诱导出没人测过的相关性），
  // 要么就得把两个量硬塞进一根轴。所以在选项里就不给。
  const comparePoints =
    compareSensor && compareSensor !== sensor ? (compareSeries.data?.points ?? []) : [];
  const compareCover = coverageOf(comparePoints);

  return (
    <div className="p-5">
      <Section
        title="环境历史"
        subtitle="温度 / 真空 / 液氦 / 磁场 / 隧道电流的长期记录，以及噪声谱快照。全部读本地库，不占用仪器通信。"
        actions={
          <RadioGroup value={view} onChange={setView} options={VIEWS} />
        }
      >
        {degraded && <DegradedNote what="环境历史" />}
        <RecorderStatusCard status={st} loading={status.isLoading} />
      </Section>

      {view === "series" ? (
        <Section title="趋势">
          <Card>
            <div className="mb-3 flex flex-wrap items-end gap-3">
              <Field label="序列">
                <SelectField
                  value={sensor}
                  onChange={setSensor}
                  options={names.map((n) => ({ value: n, label: n }))}
                />
              </Field>
              <Field label="叠加对比">
                <SelectField
                  value={compareSensor}
                  onChange={setCompareSensor}
                  options={[
                    { value: "", label: "（不对比）" },
                    ...names
                      .filter((n) => n !== sensor && unitOf(sensors.data, n) === unit)
                      .map((n) => ({ value: n, label: n })),
                  ]}
                />
              </Field>
              <div>
                <div className="mb-1 text-xs text-mast-faint">时间范围</div>
                <div className="flex flex-wrap gap-1">
                  {RANGES.map((r) => (
                    <Button
                      key={r.label}
                      variant={r.s === rangeS ? "primary" : "ghost"}
                      onClick={() => setRangeS(r.s)}
                    >
                      {r.label}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="ml-auto flex items-center gap-2 text-xs">
                {/* 色调按严重度给，不再把 unavailable（最轻的非 ok）画成和 alarm
                    一样的红。 */}
                {worst !== "ok" && <Badge tone={statusTone(worst)}>{statusLabel(worst)}</Badge>}
                {cover.excluded > 0 && (
                  <span
                    className="text-mast-faint"
                    title={
                      "被排除的读数：传感器当时没给出有效值（多个通道共用一个串口时，" +
                      "同一时刻只有一个拿得到句柄，另一个就是空的），" +
                      "或者（隧道电流）仪器当时正在被驱动 —— 那种情况下这个信号是形貌不是环境。" +
                      "它们不进统计。"
                    }
                  >
                    已计入 {cover.counted} 条 · 未计入 {cover.excluded} 条
                  </span>
                )}
              </div>
            </div>

            {/* 两条序列的计入/未计入并排 —— 这就是「同不同时保存」的可查答案。
                共用一个串口时，同一时刻只有一个拿得到句柄，于是两边的
                已计入/未计入会**恰好互补**；都在涨才说明两路都在记。 */}
            {compareSensor && compareSensor !== sensor && (
              <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-mast-muted">
                <span>
                  {sensor}：已计入 <span className="tabular-nums">{cover.counted}</span> · 未计入{" "}
                  <span className="tabular-nums">{cover.excluded}</span>
                </span>
                <span>
                  {compareSensor}：已计入{" "}
                  <span className="tabular-nums">{compareCover.counted}</span> · 未计入{" "}
                  <span className="tabular-nums">{compareCover.excluded}</span>
                </span>
                {cover.counted > 0 && compareCover.counted > 0 ? (
                  <span className="text-mast-auto">两路都在记录</span>
                ) : (
                  <span className="text-mast-warn">
                    只有一路有读数——多为共用串口时另一路拿不到句柄
                  </span>
                )}
              </div>
            )}

            {series.isLoading && <Spinner />}
            {series.error != null && <ErrorNote error={series.error} />}
            {!series.isLoading && !series.error && !names.length && (
              <EmptyNote label="还没有任何历史。记录器需要跑够一个统计桶（默认一分钟）才会出现第一个点。" />
            )}
            {/* 有桶但一个值都画不出来 —— 从前这里渲染一张空白图表，不说任何话。
                「这个传感器这段时间没给出过有效读数」是一个事实，不是空状态。 */}
            {!series.isLoading && !series.error && !!names.length && !!sensor &&
              cover.buckets > 0 && cover.drawable === 0 && (
                <EmptyNote
                  label={
                    `${sensor} 在这个时间范围内没有任何有效读数（${cover.buckets} 个统计桶，` +
                    `${cover.excluded} 条读数全部被排除）。记录器一直在跑 —— 是这个传感器没读到值。`
                  }
                />
              )}
            {!series.isLoading && !series.error && !!names.length && (
              <EnvSeriesChart
                sensor={sensor}
                unit={series.data?.unit ?? ""}
                points={points}
                bucketS={series.data?.bucket_s_effective ?? 60}
                thinned={!!series.data?.thinned}
                compare={
                  compareSensor && compareSensor !== sensor
                    ? { sensor: compareSensor, points: comparePoints }
                    : null
                }
              />
            )}
          </Card>
        </Section>
      ) : (
        <Section title="噪声谱">
          <Card>
            <div className="mb-3 flex flex-wrap items-end gap-3">
              <Field label="通道">
                <SelectField
                  value={channel}
                  onChange={setChannel}
                  options={[
                    { value: "current", label: channelLabel("current") },
                    { value: "z", label: channelLabel("z") },
                  ]}
                />
              </Field>
              <Field label="快照">
                <SelectField
                  value={primaryId == null ? "" : String(primaryId)}
                  onChange={(v) => setPrimaryId(v ? Number(v) : null)}
                  options={metas.map((m) => ({
                    value: String(m.id),
                    label: spectrumPickerLabel(m, fmtTs),
                  }))}
                />
              </Field>
              <Field label="对比">
                <SelectField
                  value={compareId == null ? "" : String(compareId)}
                  onChange={(v) => setCompareId(v ? Number(v) : null)}
                  options={[
                    { value: "", label: "（不对比）" },
                    ...metas
                      .filter((m) => m.id !== primaryId)
                      .map((m) => ({
                        value: String(m.id),
                        label: spectrumPickerLabel(m, fmtTs),
                      })),
                  ]}
                />
              </Field>
            </div>

            {spectra.isLoading && <Spinner />}
            {spectra.error != null && <ErrorNote error={spectra.error} />}
            {!spectra.isLoading && !spectra.error && (
              <SpectrumViewer primary={primary} compare={compare} />
            )}
            {channel === "z" && !metas.length && (
              <p className="mt-2 text-xs text-mast-faint">
                Z 噪声谱默认关闭（设置 → 环境历史 → 记录 Z 噪声谱）。它是这套记录里唯一会主动
                占用仪器通信的部分，需要 Osci2T 双通道模块。
              </p>
            )}
          </Card>
        </Section>
      )}
    </div>
  );
}

function useSpectrum(id: number | null) {
  const q = useQuery({
    queryKey: ["env-history", "spectrum", id],
    enabled: id != null,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/env-history/spectra/{spectrum_id}", {
        params: { path: { spectrum_id: id as number } },
      });
      if (error) throw error;
      return data;
    },
    // A stored spectrum never changes — fetch once and keep it, so flipping
    // back and forth between two snapshots to compare them is instant.
    staleTime: Infinity,
  });
  const d = q.data;
  if (!d || !d.found) return null;
  return {
    id: d.id,
    ts: d.ts,
    channel: d.channel,
    unit: d.unit,
    fs_hz: d.fs_hz,
    span_s: d.span_s,
    n_segments: d.n_segments,
    freqs_hz: d.freqs_hz ?? [],
    psd: d.psd ?? [],
  } satisfies SpectrumLike;
}

function RecorderStatusCard({ status, loading }: { status?: Status; loading: boolean }) {
  if (loading) return <Spinner />;
  if (!status) return null;
  const sink = (status.sink ?? {}) as Record<string, unknown>;
  const store = (status.store ?? {}) as Record<string, unknown>;
  const sweep = (status.sweep ?? {}) as Record<string, unknown>;
  const cur = status.spectra?.current;
  const z = status.spectra?.z;
  const disabled = !!sink.disabled;

  return (
    <Card className="mt-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <span className="flex items-center gap-2">
          <Badge tone={status.recording ? "AUTO" : "INFO"}>
            {status.recording ? "记录中" : status.enabled ? "已暂停" : "已关闭"}
          </Badge>
          {disabled && (
            <span className="text-mast-warn" title="写入失败后自动静默 60 秒，随后自行恢复。实时监控与 CSV 不受影响。">
              写入暂时禁用
            </span>
          )}
        </span>
        <Stat label="聚合粒度" value={fmtSpan(status.bucket_s ?? 0)} />
        <Stat label="统计桶" value={`${num(store.bucket_rows)} 条`} />
        <Stat label="噪声谱" value={`${num(store.spectra_rows)} 条`} />
        <Stat label="库大小" value={fmtBytes(num(store.db_bytes))} />
        {cur && (
          <Stat
            label="电流谱"
            value={
              cur.last_ts
                ? `${fmtTs(cur.last_ts)}（攒 ${cur.n_accum ?? 0} 段）`
                : `攒 ${cur.n_accum ?? 0} 段`
            }
            hint={
              cur.insufficient_quiet
                ? "窗口早已到点但安静段一直不够——仪器长时间在忙。窗口会自动延长。"
                : undefined
            }
          />
        )}
        {status.z_enabled && z && (
          <Stat label="Z 谱" value={z.last_ts ? fmtTs(z.last_ts) : String((z.last_result as Record<string, unknown> | undefined)?.skipped ?? "等待安静窗口")} />
        )}
        {sweep.cutoff != null && (
          <Stat
            label="上次清扫"
            value={`删 ${num(sweep.rows_pruned)} 条`}
            hint={`早于 ${String(sweep.cutoff).slice(0, 19)} 的逐条读数已清理（告警行永久保留）`}
          />
        )}
      </div>
    </Card>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <span className="flex items-baseline gap-1.5" title={hint}>
      <span className="text-xs text-mast-faint">{label}</span>
      <span className="tabular-nums">{value}</span>
      {hint && <span className="text-xs text-mast-warn">*</span>}
    </span>
  );
}

function num(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}
