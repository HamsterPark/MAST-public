import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { api } from "@/api/client";
import { useCurrentScope, useSwitchExperiment } from "@/api/scope";
import { NewExperimentDialog } from "@/components/scope/NewScopeDialogs";
import type { components } from "@/api/schema";
import { DegradedNote, ErrorNote, Section, Spinner } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { Button, SubTabs } from "@/components/controls";
import { fmtTime } from "@/components/records/shared";
import { ExperimentDetailPane } from "@/components/records/ExperimentDetailPane";
import { ActionDetailPane } from "@/components/records/ActionDetailPane";
import { TimelinePane } from "@/components/records/TimelinePane";
import { ScanDataPane } from "@/components/records/ScanDataPane";
import { GroupChatHistoryPane } from "@/components/records/GroupChatHistoryPane";
import { DocumentsPane } from "@/components/records/DocumentsPane";
import { FeedbackPane } from "@/components/records/FeedbackPane";
import { DiagnosticsPane } from "@/components/records/DiagnosticsPane";
import { TrajectoryExportPane } from "@/components/records/TrajectoryExportPane";
import { ExportAllPane } from "@/components/records/ExportAllPane";
import {
  FeedbackButton,
} from "@/components/records/ExperimentLifecycle";
import { useStickyTab } from "@/hooks/useStickyTab";

// Domain D — 实验记录 (Records). Full-parity rebuild of the freeze-prone old
// Gradio Records tab. Flat in-page SubTabs (no nested gr.Tabs → no freeze).
//
// Sub-tabs match the OLD Gradio build verbatim, same names + order:
//   实验记录  records list (GET /api/experiments) → detail → action detail;
//             + 新建/结束实验 + 反馈. (OLD = Provenance iframe over this data.)
//   数据      recent .sxm/.dat/.3ds scans + Load Latest + base64 PNG preview
//             (GET /api/scans/latest + /api/scans/preview).
//   日志      experiment + sample dropdowns → action timeline incl. per-action
//             TCP calls + state diff (GET /api/experiments/{id}/timeline).
//   训练日志  JSONL / SFT / DPO / 失败挖掘 (POST /api/trajectories/export).
//   导出全部  full-history ZIP manifest (POST /api/experiments/export).
//
// Every read renders loading / error / degraded / empty; edits toast.

type ExperimentSummary = components["schemas"]["ExperimentSummary"];

type TopTab =
  | "experiments"
  | "data"
  | "timeline"
  | "diagnostics"
  | "groupchats"
  | "documents"
  | "feedback"
  | "training"
  | "export";

const TABS: { id: TopTab; label: string }[] = [
  { id: "experiments", label: "实验记录" },
  { id: "data", label: "数据" },
  { id: "timeline", label: "日志" },
  { id: "groupchats", label: "群聊记录" },
  { id: "diagnostics", label: "诊断" },
  { id: "documents", label: "报告" },
  { id: "feedback", label: "反馈" },
  { id: "training", label: "训练日志" },
  { id: "export", label: "导出全部" },
];

// Selection state machine for the experiments drill-down.
type Selection =
  | { kind: "list" }
  | { kind: "experiment"; experimentId: string }
  | { kind: "action"; experimentId: string; actionId: string };

export default function RecordsPage() {
  // 大栏目要记住上次停在哪个子页。九个子页里「导出全部」「训练日志」
  // 都是回去接着看的地方,每次回来被打回「实验记录」等于每次重新找一遍。
  const [tab, setTab] = useStickyTab<TopTab>(
    "records", TABS.map((t) => t.id), "experiments");
  const [sel, setSel] = useState<Selection>({ kind: "list" });

  return (
    <div className="space-y-6">
      <SubTabs tabs={TABS} value={tab} onChange={setTab} />

      {tab === "experiments" && <ExperimentsView sel={sel} setSel={setSel} />}
      {tab === "data" && <ScanDataPane />}
      {tab === "timeline" && <TimelinePane />}
      {tab === "groupchats" && <GroupChatHistoryPane />}
      {tab === "diagnostics" && <DiagnosticsPane />}
      {tab === "documents" && <DocumentsPane />}
      {tab === "feedback" && <FeedbackPane />}
      {tab === "training" && <TrajectoryExportPane />}
      {tab === "export" && <ExportAllPane />}
    </div>
  );
}

function ExperimentsView({
  sel,
  setSel,
}: {
  sel: Selection;
  setSel: (s: Selection) => void;
}) {
  if (sel.kind === "action") {
    return (
      <ActionDetailPane
        actionId={sel.actionId}
        onBack={() => setSel({ kind: "experiment", experimentId: sel.experimentId })}
      />
    );
  }
  if (sel.kind === "experiment") {
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          {/* 「结束实验」已移除 (2026-07-28)：实验没有终态，永远可以继续做。
              取而代之的是「切换到此实验」—— 回到一个旧实验继续，才是这里真正
              需要的动作。 */}
          <ActivateExperimentButton experimentId={sel.experimentId} />
          <FeedbackButton experimentId={sel.experimentId} />
        </div>
        <ExperimentDetailPane
          experimentId={sel.experimentId}
          onBack={() => setSel({ kind: "list" })}
          onSelectAction={(actionId) =>
            setSel({ kind: "action", experimentId: sel.experimentId, actionId })
          }
        />
      </div>
    );
  }
  return (
    <ExperimentsList onSelect={(id) => setSel({ kind: "experiment", experimentId: id })} />
  );
}

function useExperiments() {
  return useQuery({
    queryKey: ["experiments"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experiments");
      if (error) throw error;
      return data;
    },
  });
}

/** 相对时间。「上次活动」比「开始时间」更能回答"我要切回哪个"。 */
function relTime(iso?: string | null): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 90) return "刚刚";
  if (s < 3600) return `${Math.round(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.round(s / 3600)} 小时前`;
  return `${Math.round(s / 86400)} 天前`;
}


/** 「切换到此实验」——记录页详情的主动作，取代原来的「结束实验」。 */
function ActivateExperimentButton({ experimentId }: { experimentId: string }) {
  const scope = useCurrentScope();
  const switchExp = useSwitchExperiment();
  const isCurrent = scope.data?.experiment?.id === experimentId;
  if (isCurrent) {
    return (
      <span className="rounded-mast-badge bg-mast-accent-soft px-2 py-1 text-xs text-mast-accent">
        当前实验
      </span>
    );
  }
  return (
    <Button
      variant="primary"
      loading={switchExp.isPending}
      onClick={() => switchExp.mutate({ experimentId })}
    >
      切换到此实验
    </Button>
  );
}

/** 新建实验（带重名检测的确认弹窗）。 */
function NewExperimentLauncher() {
  const [open, setOpen] = useState(false);
  const scope = useCurrentScope();
  return (
    <>
      <Button onClick={() => setOpen(true)}>新建实验</Button>
      <NewExperimentDialog
        open={open}
        onClose={() => setOpen(false)}
        currentName={scope.data?.experiment?.name}
      />
    </>
  );
}

function ExperimentsList({ onSelect }: { onSelect: (id: string) => void }) {
  const q = useExperiments();
  const scope = useCurrentScope();
  const switchExp = useSwitchExperiment();
  const currentId = scope.data?.experiment?.id;

  const columns: ColumnDef<ExperimentSummary, any>[] = [
    {
      accessorKey: "name",
      header: "名称",
      cell: (c) => (
        <span className="flex items-center gap-2">
          <button
            className="text-mast-accent hover:underline"
            onClick={() => onSelect(c.row.original.id)}
          >
            {c.getValue() || "(未命名)"}
          </button>
          {c.row.original.id === currentId && (
            <span className="rounded-mast-badge bg-mast-accent-soft px-1.5 py-px text-[10.5px] text-mast-accent">
              当前
            </span>
          )}
        </span>
      ),
    },
    { accessorKey: "goal", header: "目标", cell: (c) => c.getValue() || "—" },
    {
      // 「状态」列换成「切换」。experiments.status 已经是历史遗留的展示列
      // （2026-07-28）：实验没有生命周期状态，「哪个是当前」由服务端的单一
      // 指针回答。历史 status 值降级到详情页展示。
      id: "switch",
      header: "",
      cell: (c) =>
        c.row.original.id === currentId ? (
          <span className="text-xs text-mast-faint">—</span>
        ) : (
          <button
            className="rounded-mast-ctl border border-mast-border-strong px-2 py-1 text-xs hover:bg-mast-panel-2 disabled:opacity-50"
            disabled={switchExp.isPending}
            onClick={() => switchExp.mutate({ experimentId: c.row.original.id })}
          >
            切换到此实验
          </button>
        ),
    },
    {
      accessorKey: "start_time",
      header: "开始时间",
      cell: (c) => fmtTime(c.getValue() as string | null),
    },
    {
      id: "last_active",
      header: "最近活动",
      cell: (c) => (
        <span className="text-xs text-mast-muted">
          {relTime(
            (c.row.original as { last_active_at?: string | null }).last_active_at ??
              c.row.original.start_time,
          )}
        </span>
      ),
    },
    {
      accessorKey: "id",
      header: "ID",
      cell: (c) => (
        <span className="font-mono text-xs text-mast-muted">{c.getValue() as string}</span>
      ),
    },
  ];

  return (
    <>
      {/* OLD Records header markdown (verbatim) restored for parity. */}
      <p className="mb-3 text-sm text-mast-muted">
        <strong className="text-mast-text">实验记录 Provenance</strong> — campaigns / actions /
        observations / claims 溯源图。数据来自 <code className="font-mono">mast_experiments_v2.db</code>
        ；运行实验后点「刷新数据」拉取最新记录。
      </p>
    <Section
      title={`实验列表${q.data?.count ? ` (${q.data.count})` : ""}`}
      actions={
        <div className="flex items-center gap-2">
          {/* Old Records iframe had a 刷新数据 button to re-pull live records
              after running experiments; restored for parity. */}
          <button
            type="button"
            disabled={q.isFetching}
            onClick={() => q.refetch()}
            className="rounded border border-mast-border px-3 py-1 text-sm text-mast-muted hover:text-mast-text disabled:opacity-50"
          >
            {q.isFetching ? "刷新中…" : "刷新数据"}
          </button>
          <NewExperimentLauncher />
        </div>
      }
    >
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="实验记录" />}
      {q.data && !q.data.degraded && (
        <DataTable
          data={q.data.experiments ?? []}
          columns={columns}
          empty="暂无实验记录"
        />
      )}
    </Section>
    </>
  );
}
