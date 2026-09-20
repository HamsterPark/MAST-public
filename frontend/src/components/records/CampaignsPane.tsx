import { useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { api } from "../../api/client";
import type { components } from "../../api/schema";
import { DegradedNote, ErrorNote, Section, Spinner } from "../ui";
import { DataTable } from "../DataTable";
import { StatusPill, fmtTime } from "./shared";
// 这一列的文案是纯函数，和 conduct 面板共用一份 —— 三态的措辞不许有第二处。
import { badgeTone, goalProgressText } from "../../lib/conduct";

type CampaignSummary = components["schemas"]["CampaignSummary"];

const PAGE_SIZE = 50;

interface Filters {
  status: string;
  from: string;
  to: string;
  agent: string;
}

function useCampaigns(filters: Filters, page: number) {
  return useQuery({
    queryKey: ["records", "campaigns", filters, page],
    placeholderData: keepPreviousData,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/records/campaigns", {
        params: {
          query: {
            status: filters.status || null,
            from: filters.from || null,
            to: filters.to || null,
            agent: filters.agent || null,
            page,
            page_size: PAGE_SIZE,
          },
        },
      });
      if (error) throw error;
      return data;
    },
  });
}

const inputCls =
  "rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text placeholder:text-mast-muted";

export function CampaignsPane() {
  const [draft, setDraft] = useState<Filters>({ status: "", from: "", to: "", agent: "" });
  const [filters, setFilters] = useState<Filters>(draft);
  const [page, setPage] = useState(1);
  const q = useCampaigns(filters, page);

  const apply = () => {
    setPage(1);
    setFilters(draft);
  };
  const reset = () => {
    const cleared = { status: "", from: "", to: "", agent: "" };
    setDraft(cleared);
    setFilters(cleared);
    setPage(1);
  };

  const columns: ColumnDef<CampaignSummary, any>[] = [
    { accessorKey: "title", header: "标题", cell: (c) => c.getValue() || "(未命名)" },
    {
      accessorKey: "hypothesis",
      header: "假设",
      cell: (c) => c.getValue() || "—",
    },
    {
      accessorKey: "status",
      header: "状态",
      cell: (c) => <StatusPill status={c.getValue() as string | null} />,
    },
    { accessorKey: "created_by", header: "Agent", cell: (c) => c.getValue() || "—" },
    {
      accessorKey: "created_at",
      header: "创建时间",
      cell: (c) => fmtTime(c.getValue() as string | null),
    },
    {
      id: "goal",
      header: "目标判据",
      cell: (c) => {
        const g = goalProgressText(c.row.original.goal_progress);
        return (
          <span
            className={"rounded px-1.5 py-0.5 text-xs " + badgeTone(g.tone)}
            title={c.row.original.goal_progress?.reason || ""}
          >
            {g.text}
          </span>
        );
      },
    },
    {
      id: "experiments",
      header: "实验",
      cell: (c) => c.row.original.stats?.experiments ?? 0,
    },
    {
      id: "actions",
      header: "动作",
      cell: (c) => c.row.original.stats?.actions ?? 0,
    },
    {
      id: "observations",
      header: "观测",
      cell: (c) => c.row.original.stats?.observations ?? 0,
    },
    {
      id: "scans",
      header: "扫描",
      cell: (c) => c.row.original.stats?.scans ?? 0,
    },
  ];

  const total = q.data?.count ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <Section title="Campaigns（研究活动）">
      <div className="mb-3 flex flex-wrap items-end gap-2">
        <label className="flex flex-col text-xs text-mast-muted">
          状态
          <input
            className={inputCls}
            value={draft.status}
            placeholder="running / completed …"
            onChange={(e) => setDraft({ ...draft, status: e.target.value })}
          />
        </label>
        <label className="flex flex-col text-xs text-mast-muted">
          起始日期
          <input
            className={inputCls}
            value={draft.from}
            placeholder="YYYY-MM-DD"
            onChange={(e) => setDraft({ ...draft, from: e.target.value })}
          />
        </label>
        <label className="flex flex-col text-xs text-mast-muted">
          截止日期
          <input
            className={inputCls}
            value={draft.to}
            placeholder="YYYY-MM-DD"
            onChange={(e) => setDraft({ ...draft, to: e.target.value })}
          />
        </label>
        <label className="flex flex-col text-xs text-mast-muted">
          Agent
          <input
            className={inputCls}
            value={draft.agent}
            placeholder="created_by"
            onChange={(e) => setDraft({ ...draft, agent: e.target.value })}
          />
        </label>
        <button
          onClick={apply}
          className="rounded bg-mast-accent/20 px-3 py-1 text-sm text-mast-accent hover:bg-mast-accent/30"
        >
          筛选
        </button>
        <button
          onClick={reset}
          className="rounded border border-mast-border px-3 py-1 text-sm text-mast-muted hover:text-mast-text"
        >
          重置
        </button>
      </div>

      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="Campaigns 记录" />}
      {q.data && !q.data.degraded && (
        <>
          <DataTable
            data={q.data.campaigns ?? []}
            columns={columns}
            empty="无符合条件的 campaign"
          />
          <div className="mt-3 flex items-center justify-between text-sm text-mast-muted">
            <span>
              共 {total} 条 · 第 {page} / {pageCount} 页
            </span>
            <div className="flex gap-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="rounded border border-mast-border px-3 py-1 disabled:opacity-40 hover:text-mast-text"
              >
                上一页
              </button>
              <button
                disabled={page >= pageCount}
                onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
                className="rounded border border-mast-border px-3 py-1 disabled:opacity-40 hover:text-mast-text"
              >
                下一页
              </button>
            </div>
          </div>
        </>
      )}
    </Section>
  );
}
