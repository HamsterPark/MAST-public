import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { DegradedNote, ErrorNote, Section, Spinner } from "@/components/ui";
import { DocList, type DocRow } from "@/components/records/ExperimentDocuments";
import { useCurrentScope } from "@/api/scope";

// 报告 — the autonomous pipeline's WRITTEN
// outputs. A full run could end "全流程完成…论文评审(ACCEPT)" while
// nothing was visible on screen: paper_writing had no save tool at all,
// and paper_review's load_draft pointed at a directory that never existed.
//
// 2026-07-29：文档现在住在**实验文件夹**里（reports/ 与 plans/），带真正的版本
// 历史和归属。这个面板是跨实验的总览；单个实验的文档在 实验记录 → 实验详情 里。
// 正文改用 markdown 渲染 —— 此前是 <pre> 吐源码，表格和公式全是原始符号。

const KIND_TABS: { value: string; label: string }[] = [
  { value: "", label: "全部" },
  { value: "experiment_report", label: "实验报告" },
  { value: "paper_draft", label: "论文草稿" },
  { value: "literature_report", label: "文献报告" },
  { value: "experiment_plan", label: "实验计划" },
  { value: "review", label: "评审" },
];

export function DocumentsPane() {
  const [kind, setKind] = useState("");
  const [scopedOnly, setScopedOnly] = useState(false);
  const [showDiscarded, setShowDiscarded] = useState(false);
  const scope = useCurrentScope();
  const currentExperimentId = scope.data?.experiment?.id ?? null;

  const q = useQuery({
    queryKey: ["documents", kind, scopedOnly ? currentExperimentId : null, showDiscarded],
    queryFn: async () => {
      const query: Record<string, string | boolean> = {};
      if (kind) query.kind = kind;
      if (scopedOnly && currentExperimentId) query.experiment_id = currentExperimentId;
      if (showDiscarded) query.include_discarded = true;
      const { data, error } = await api.GET("/api/documents", {
        params: { query },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: 30_000,
  });

  if (q.isLoading) return <Spinner label="加载文档…" />;
  if (q.isError) return <ErrorNote error={q.error} />;
  const payload = q.data as
    | { documents?: DocRow[]; degraded?: boolean | null }
    | undefined;
  if (payload?.degraded) return <DegradedNote what="文档库" />;

  const docs = (payload?.documents ?? []) as DocRow[];

  return (
    <Section
      title={`文档 (${docs.length})`}
      subtitle="文献报告 / 实验计划 / 实验报告 / 论文草稿 / 评审 —— 每次修订都是新版本，永不覆盖。点击展开全文。"
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {KIND_TABS.map((t) => (
          <button
            key={t.value}
            onClick={() => setKind(t.value)}
            // `accent-ink` is the foreground for a SOLID accent fill (white on
            // cyan in light, near-black on cyan in dark). Pairing it with
            // `accent-soft` — a 9% tint that stays almost the page background —
            // put white text on white in light mode: the active chip vanished
            // . Every other accent-soft surface in this app pairs
            // with `text-mast-accent`; this one was the outlier.
            className={
              "rounded-full border px-2.5 py-1 text-xs " +
              (kind === t.value
                ? "border-mast-accent-line bg-mast-accent-soft font-medium text-mast-accent"
                : "border-mast-border text-mast-muted hover:bg-mast-bg/40")
            }
          >
            {t.label}
          </button>
        ))}
        <label
          className="ml-auto flex cursor-pointer items-center gap-1.5 text-xs text-mast-muted"
          title="废弃的文档没有被删除，只是搬进了 _discarded 区，随时可恢复"
        >
          <input
            type="checkbox"
            checked={showDiscarded}
            onChange={(e) => setShowDiscarded(e.target.checked)}
          />
          含已废弃
        </label>
        {currentExperimentId && (
          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-mast-muted">
            <input
              type="checkbox"
              checked={scopedOnly}
              onChange={(e) => setScopedOnly(e.target.checked)}
            />
            只看当前实验
          </label>
        )}
      </div>

      {docs.length === 0 ? (
        <div className="space-y-1 text-sm text-mast-muted">
          <p>没有匹配的文档。</p>
          <p className="text-xs">
            智能体写出的报告、综述、计划和论文草稿会保存到各自实验文件夹的{" "}
            <code className="font-mono text-[11px]">reports/</code> 与{" "}
            <code className="font-mono text-[11px]">plans/</code> 下。
            没有活跃实验时保存的文档落在{" "}
            <code className="font-mono text-[11px]">_unfiled/</code>，可以稍后认领。
          </p>
        </div>
      ) : (
        <DocList rows={docs} showOwner />
      )}
    </Section>
  );
}
