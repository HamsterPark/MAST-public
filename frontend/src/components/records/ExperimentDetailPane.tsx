import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import clsx from "clsx";
import { api } from "../../api/client";
import type { components } from "../../api/schema";
import { Card, DegradedNote, EmptyNote, ErrorNote, Section, Spinner } from "../ui";
import { DataTable } from "../DataTable";
import { Field, StatusPill, SuccessPill, fmtTime } from "./shared";
import { ExperimentDocuments } from "./ExperimentDocuments";
import { useCurrentScope, useSwitchSample } from "@/api/scope";
import { TipHistoryDialog } from "@/components/scope/TipControls";

type ExperimentDetail = components["schemas"]["ExperimentDetail"];
type SampleSummary = components["schemas"]["SampleSummary"];
type ActionSummary = components["schemas"]["ActionSummary"];
type MapMarker = components["schemas"]["MapMarker"];
type FeedbackEntry = components["schemas"]["FeedbackEntry"];
type TipInService = components["schemas"]["TipInService"];

function useExperimentDetail(id: string) {
  return useQuery({
    queryKey: ["experiment", id],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experiments/{experiment_id}", {
        params: { path: { experiment_id: id } },
      });
      if (error) throw error;
      return data;
    },
  });
}

const MARKER_COLUMNS: ColumnDef<MapMarker, any>[] = [
  { accessorKey: "kind", header: "类型" },
  { accessorKey: "skill_name", header: "技能", cell: (c) => c.getValue() || "—" },
  { accessorKey: "label", header: "标签", cell: (c) => c.getValue() || "—" },
  {
    accessorKey: "x_m",
    header: "X (nm)",
    cell: (c) => {
      const v = c.getValue() as number | null;
      return v == null ? "—" : (v * 1e9).toFixed(1);
    },
  },
  {
    accessorKey: "y_m",
    header: "Y (nm)",
    cell: (c) => {
      const v = c.getValue() as number | null;
      return v == null ? "—" : (v * 1e9).toFixed(1);
    },
  },
  {
    accessorKey: "status",
    header: "状态",
    cell: (c) => <StatusPill status={c.getValue() as string | null} />,
  },
  { accessorKey: "source", header: "来源" },
];

const FEEDBACK_COLUMNS: ColumnDef<FeedbackEntry, any>[] = [
  { accessorKey: "kind", header: "类别" },
  { accessorKey: "rating", header: "评分", cell: (c) => c.getValue() || "—" },
  { accessorKey: "comment", header: "评论", cell: (c) => c.getValue() || "—" },
  { accessorKey: "agent", header: "Agent", cell: (c) => c.getValue() || "—" },
  {
    accessorKey: "timestamp",
    header: "时间",
    cell: (c) => fmtTime(c.getValue() as string | null),
  },
];

export function ExperimentDetailPane({
  experimentId,
  onBack,
  onSelectAction,
}: {
  experimentId: string;
  onBack: () => void;
  onSelectAction: (actionId: string) => void;
}) {
  const q = useExperimentDetail(experimentId);

  const actionColumns: ColumnDef<ActionSummary, any>[] = [
    {
      accessorKey: "skill_name",
      header: "技能",
      cell: (c) => (
        <button
          className="text-mast-accent hover:underline"
          onClick={() => onSelectAction(c.row.original.id)}
        >
          {c.getValue() || "(未知)"}
        </button>
      ),
    },
    {
      accessorKey: "success",
      header: "结果",
      cell: (c) => <SuccessPill success={c.getValue() as boolean | null} />,
    },
    {
      accessorKey: "duration_s",
      header: "耗时(s)",
      cell: (c) => (c.getValue() as number)?.toFixed(2) ?? "0.00",
    },
    {
      accessorKey: "error",
      header: "错误",
      cell: (c) => {
        const v = c.getValue() as string | null;
        return v ? <span className="text-mast-danger">{v}</span> : "—";
      },
    },
    {
      accessorKey: "timestamp",
      header: "时间",
      cell: (c) => fmtTime(c.getValue() as string | null),
    },
  ];

  return (
    <div className="space-y-6">
      <button onClick={onBack} className="text-sm text-mast-accent hover:underline">
        ← 返回实验列表
      </button>

      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data && <DetailBody data={q.data} actionColumns={actionColumns} />}
    </div>
  );
}

// Sentinel for the "全部样品" (all samples) navigation option — empty string,
// distinct from any real sample id, so the scope key stays a plain string.
const ALL_SAMPLES = "";

const sampleLabel = (s: SampleSummary) => s.name || `${s.id.slice(0, 8)}…`;

/** Compact "起 → 止" time range for a sample. Prefers the sample's own
 *  start/end columns; falls back to the min/max timestamp of its actions so a
 *  sample with no recorded range still shows a meaningful span. */
function sampleTimeRange(
  s: SampleSummary,
  actions: ActionSummary[],
): { start: string | null; end: string | null } {
  let start = s.start_time ?? null;
  let end = s.end_time ?? null;
  if (start && end) return { start, end };
  const ts = actions
    .filter((a) => a.sample_id === s.id && a.timestamp)
    .map((a) => a.timestamp as string)
    .sort();
  if (ts.length) {
    start = start ?? ts[0] ?? null;
    end = end ?? ts[ts.length - 1] ?? null;
  }
  return { start, end };
}

/** One sample navigation card — the prominent, clickable per-sample entry that
 *  scopes the timeline / markers / feedback below. Shows name + type + status +
 *  action count + time range so samples read as a real navigation dimension. */

/** 「设为当前样品」。样品可以来回切 —— 换样品不会结束上一个。 */
function SetCurrentSample({
  experimentId,
  sampleId,
  name,
}: {
  experimentId: string;
  sampleId: string;
  name: string;
}) {
  const scope = useCurrentScope();
  const switchSmp = useSwitchSample();
  const isCurrent = scope.data?.sample?.id === sampleId;
  if (isCurrent) {
    return <span className="text-mast-accent">当前样品</span>;
  }
  return (
    <button
      className="rounded-mast-ctl border border-mast-border-strong px-2 py-0.5 text-xs hover:bg-mast-panel-2 disabled:opacity-50"
      disabled={switchSmp.isPending}
      onClick={() => switchSmp.mutate({ experimentId, sampleId })}
      title={`把「${name}」设为当前样品`}
    >
      {switchSmp.isPending ? "切换中…" : "设为当前"}
    </button>
  );
}

function SampleNavCard({
  active,
  title,
  subtitle,
  status,
  count,
  range,
  onClick,
}: {
  active: boolean;
  title: string;
  subtitle?: string | null;
  status?: string | null;
  count: number;
  range?: { start: string | null; end: string | null };
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={clsx(
        "flex w-full flex-col gap-1.5 rounded-lg border p-3 text-left transition-colors",
        active
          ? "border-mast-accent bg-mast-accent/10"
          : "border-mast-border bg-mast-bg hover:border-mast-accent/50",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span
          className={clsx(
            "min-w-0 truncate text-sm font-medium",
            active ? "text-mast-accent" : "text-mast-text",
          )}
        >
          {title}
        </span>
        <span className="shrink-0 rounded bg-mast-border/40 px-1.5 py-0.5 text-xs text-mast-muted">
          {count} 动作
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs text-mast-muted">
        {subtitle ? <span className="truncate">{subtitle}</span> : null}
        {status ? <StatusPill status={status} /> : null}
      </div>
      {range && (range.start || range.end) ? (
        <div className="text-xs text-mast-muted">
          {fmtTime(range.start)} → {fmtTime(range.end)}
        </div>
      ) : null}
    </button>
  );
}

function DetailBody({
  data,
  actionColumns,
}: {
  data: ExperimentDetail;
  actionColumns: ColumnDef<ActionSummary, any>[];
}) {
  // Hooks must run unconditionally — declare before the early returns below.
  const [sampleFilter, setSampleFilter] = useState<string>(ALL_SAMPLES);

  const samples = data.samples ?? [];
  const actions = data.actions ?? [];
  const markers = data.map_markers ?? [];
  const feedback = data.feedback ?? [];

  // Guard against a stale selection (sample disappeared after a refetch) so the
  // scope and the displayed data always agree.
  const effectiveFilter = samples.some((s) => s.id === sampleFilter)
    ? sampleFilter
    : ALL_SAMPLES;

  // Per-sample action counts derived from the action list — authoritative for
  // the nav cards even when a SampleSummary.action_count is stale/0.
  const countBySample = useMemo(() => {
    const m = new Map<string, number>();
    for (const a of actions) {
      if (a.sample_id) m.set(a.sample_id, (m.get(a.sample_id) ?? 0) + 1);
    }
    return m;
  }, [actions]);
  const sampleCount = (s: SampleSummary) =>
    s.action_count || countBySample.get(s.id) || 0;
  // Actions with no sample attribution — surfaced under 全部样品 so nothing is lost.
  const unscopedActions = actions.filter((a) => !a.sample_id).length;

  // 按样品过滤动作 / 地图标记 / 反馈。"全部样品" 时原样返回。
  const filtered = useMemo(() => {
    if (!effectiveFilter) return { actions, markers, feedback };
    return {
      actions: actions.filter((a) => a.sample_id === effectiveFilter),
      markers: markers.filter((m) => m.sample_id === effectiveFilter),
      feedback: feedback.filter((f) => f.sample_id === effectiveFilter),
    };
  }, [effectiveFilter, actions, markers, feedback]);

  if (data.degraded) return <DegradedNote what="实验详情" />;
  if (data.found === false) {
    return <EmptyNote label={`未找到实验 ${data.id}`} />;
  }

  const selectedSample =
    effectiveFilter ? samples.find((s) => s.id === effectiveFilter) : undefined;
  const scopeSuffix = effectiveFilter
    ? ` · 样品 ${selectedSample ? sampleLabel(selectedSample) : effectiveFilter.slice(0, 8) + "…"}`
    : "";

  return (
    <>
      <Section title={data.name || "(未命名实验)"}>
        <Card>
          <Field label="ID">{data.id}</Field>
          <Field label="目标">{data.goal_text || "—"}</Field>
          {/* 「状态」降级为历史遗留字段 (2026-07-28)：实验没有生命周期状态，
              「哪个是当前」由服务端指针回答。旧库里已有的 superseded/completed
              仍然展示出来，因为用户要能读懂历史数据。 */}
          {data.status && data.status !== "running" && (
            <Field label="历史状态（旧版）">
              <StatusPill status={data.status} />
            </Field>
          )}
          <Field label="开始">{fmtTime(data.start_time)}</Field>
          {data.end_time && <Field label="结束">{fmtTime(data.end_time)}</Field>}
          {data.notes && <Field label="备注">{data.notes}</Field>}
        </Card>
      </Section>

      {/* 本实验期间在役的针尖（「针尖记录等是不是也应该在实验记录中?」）。
          放在样品之前 —— 读一份实验记录时,「当时用的是哪根针」和「样品是什么」
          是同一层的前提。 */}
      <TipsInService tips={data.tips ?? []} />

      {/* 样品 = 实验内的一级导航维度。选中某样品后，下方动作时间线 / 地图标记 /
          反馈仅显示该样品；"全部样品" 显示整个实验。 */}
      <Section title={`样品 (${samples.length})`}>
        {samples.length === 0 ? (
          <EmptyNote label="该实验暂无样品" />
        ) : (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            <SampleNavCard
              active={effectiveFilter === ALL_SAMPLES}
              title="全部样品"
              subtitle={`${samples.length} 个样品${
                unscopedActions ? ` · ${unscopedActions} 个未归属动作` : ""
              }`}
              count={actions.length}
              range={{ start: data.start_time ?? null, end: data.end_time ?? null }}
              onClick={() => setSampleFilter(ALL_SAMPLES)}
            />
            {samples.map((s) => {
              const subtitle = [s.sample_type, s.sample_subtype]
                .filter(Boolean)
                .join(" / ") || null;
              return (
                <SampleNavCard
                  key={s.id}
                  active={effectiveFilter === s.id}
                  title={sampleLabel(s)}
                  subtitle={subtitle}
                  status={s.status}
                  count={sampleCount(s)}
                  range={sampleTimeRange(s, actions)}
                  onClick={() => setSampleFilter(s.id)}
                />
              );
            })}
          </div>
        )}
        {effectiveFilter ? (
          <div className="mt-3 flex items-center gap-2 text-sm text-mast-muted">
            <span>
              当前查看：
              <span className="text-mast-text">
                {selectedSample ? sampleLabel(selectedSample) : effectiveFilter}
              </span>
            </span>
            <button
              type="button"
              onClick={() => setSampleFilter(ALL_SAMPLES)}
              className="rounded border border-mast-border px-2 py-0.5 text-xs text-mast-muted hover:text-mast-text"
            >
              查看全部样品
            </button>
          </div>
        ) : null}
      </Section>

      {selectedSample && (
        <Section title={`样品详情 · ${sampleLabel(selectedSample)}`}>
          <Card>
            <Field label="ID">{selectedSample.id}</Field>
            <Field label="类型">{selectedSample.sample_type ?? "—"}</Field>
            <Field label="子类型">{selectedSample.sample_subtype ?? "—"}</Field>
            {selectedSample.status && selectedSample.status !== "active" && (
              <Field label="历史状态（旧版）">
                <StatusPill status={selectedSample.status} />
              </Field>
            )}
            <Field label="动作数">{sampleCount(selectedSample)}</Field>
            <Field label="开始">{fmtTime(selectedSample.start_time)}</Field>
            <Field label="当前">
              <SetCurrentSample
                experimentId={String(data.id)}
                sampleId={String(selectedSample.id)}
                name={sampleLabel(selectedSample)}
              />
            </Field>
            {selectedSample.description && (
              <Field label="描述">{selectedSample.description}</Field>
            )}
          </Card>
        </Section>
      )}

      <ExperimentDocuments experimentId={String(data.id)} />

      <Section title={`动作时间线 (${filtered.actions.length})${scopeSuffix}`}>
        <DataTable
          data={filtered.actions}
          columns={actionColumns}
          empty={effectiveFilter ? "该样品暂无动作" : "暂无动作"}
        />
      </Section>

      <Section title={`地图标记 (${filtered.markers.length})${scopeSuffix}`}>
        <DataTable
          data={filtered.markers}
          columns={MARKER_COLUMNS}
          empty={effectiveFilter ? "该样品暂无地图标记" : "暂无地图标记"}
        />
      </Section>

      <Section title={`用户反馈 (${filtered.feedback.length})${scopeSuffix}`}>
        <DataTable
          data={filtered.feedback}
          columns={FEEDBACK_COLUMNS}
          empty={effectiveFilter ? "该样品暂无反馈" : "暂无反馈"}
        />
      </Section>
    </>
  );
}


// ── 本实验期间在役的针尖 ─────────────────────────────────────────
//
// 「针尖记录等是不是也应该在实验记录中?」—— 是,但**不是搬家**。
//
// 针尖是仪器域的真源(`tips` 表):一根针跨很多次实验,一次实验也可能换好几根,
// 所以它与实验/样品**并列**而不是它们的下级 —— 右栏那个针尖卡片旁边的注释里
// 早就写着这句话。真正缺的是这一侧:读一份实验记录的时候,「当时用的是哪根针」
// 答不出来。
//
// 所以这里是**展示层聚合**,零写入、零新表。时间窗交集在服务端算(SQL,两张表在
// 同一个库里),因为 NULL 的口径(实验没结束 / 针尖还在役 / 装入时刻是回填的纯日期)
// 每一条写错都是**静默变空**,而空区块读起来像「这次实验没换过针」。
// 那些判断只该有一处,而且该在写这些字符串的那一侧。

function TipsInService({ tips }: { tips: TipInService[] }) {
  const [historyFor, setHistoryFor] = useState<string | null>(null);

  return (
    <Section title={`在役针尖 (${tips.length})`}>
      {tips.length === 0 ? (
        // 「查不到」和「确实没换过针」措辞上要分得开:这一段时间里 tips 表没有
        // 任何一行覆盖得上,最常见的原因是那时候还没开始登记针尖。
        <EmptyNote label="这段时间里没有登记过在役的针尖(可能是登记功能启用之前的实验)" />
      ) : (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {tips.map((t) => (
            <Card key={t.id} className="space-y-1 px-3 py-2 text-xs">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-semibold text-mast-text">
                  {t.tip_index ? `T${String(t.tip_index).padStart(2, "0")} ` : ""}
                  {t.name || "未命名"}
                </span>
                <span className={t.in_service_now ? "text-mast-auto" : "text-mast-faint"}>
                  {t.in_service_now ? "在用" : "已取出"}
                </span>
              </div>
              <div className="text-mast-muted">
                {[t.material, t.fabrication, t.form].filter(Boolean).join(" · ") || "—"}
              </div>
              <div className="text-mast-faint">
                {t.installed_at ? `${t.installed_at.slice(0, 10)} 装入` : "装入日期未记录"}
                {t.removed_at ? ` → ${t.removed_at.slice(0, 10)} 取出` : ""}
              </div>
              {!t.overlap_exact && (
                // 「确实在役」与「大概在役」必须看得出区别。装入时刻是回填的纯日期
                // (等价于当天 00:00)或者压根没填(退回用建档时刻)时,时间窗对上的
                // 那一端是个**下界** —— 偏早,宁可多列一根也不漏掉一根。
                // 不说出来的话,一个推断会被印得和一个事实一模一样。
                <div className="text-mast-warn">
                  装入时刻只精确到天(或未记录),与本实验的重叠是按下界推的
                </div>
              )}
              <button
                type="button"
                onClick={() => setHistoryFor(t.id)}
                className="text-mast-accent hover:underline"
              >
                在换针史中查看 →
              </button>
            </Card>
          ))}
        </div>
      )}
      {/* 链到**真正的**那张针尖卡片(全仓唯一一份渲染),而不是在这里再画一份像它
          的东西 —— 针尖卡片上写的是材料/制法/服役期这类会被拿去做判断的事实,
          两份会各自演化。针尖没有 per-tip 路由,所以「链接」的形式是把换针史打开
          并圈出这一行。 */}
      <TipHistoryDialog
        open={historyFor !== null}
        highlightId={historyFor}
        onClose={() => setHistoryFor(null)}
      />
    </Section>
  );
}
