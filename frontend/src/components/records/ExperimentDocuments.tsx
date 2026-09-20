import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, ErrorNote, Section, Spinner } from "@/components/ui";
import { MarkdownView } from "@/components/MarkdownView";
import { fmtTime } from "@/components/records/shared";

// 实验详情页的文档区。文档主归属在实验的文件夹里 —— 报告、计划、
// 论文草稿都与对应实验关联，而不是躺在实验之外的全局 data/ 目录里。
//
// 两个概念要在界面上分清（设计 §3，既定的归属模型）：
//   主归属 —— 文档存在这个实验的文件夹里（一个实验可以有 0..N 份，一对多是常态）
//   关联   —— 文档主归属在别的实验，但声明了与本实验相关（一篇论文常综合多个实验）

const KIND_TONE: Record<string, "INFO" | "WARN" | "AUTO" | "default"> = {
  literature_report: "INFO",
  experiment_plan: "WARN",
  experiment_report: "default",
  paper_draft: "AUTO",
  review: "INFO",
};

function verdictTone(v?: string | null) {
  if (v === "ACCEPT") return "AUTO" as const;
  if (v === "REJECT") return "DANGEROUS" as const;
  if (v === "REVISE") return "WARN" as const;
  return "INFO" as const;
}

export type DocRow = {
  doc_id: string;
  kind: string;
  kind_label?: string | null;
  title: string;
  version?: number | null;
  versions_count?: number | null;
  relation?: string | null;
  experiment_id?: string | null;
  experiment_name?: string | null;
  verdict?: string | null;
  words?: number | null;
  updated_at?: string | null;
  created_by?: string | null;
  conversation_id?: string | null;
  root_kind?: string | null;
  path?: string | null;
};

export function ExperimentDocuments({ experimentId }: { experimentId: string }) {
  const q = useQuery({
    queryKey: ["documents", "experiment", experimentId],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/documents", {
        params: { query: { experiment_id: experimentId } },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: 60_000,
  });

  if (q.isLoading) return <Section title="文档"><Spinner label="加载文档…" /></Section>;
  if (q.isError) return <Section title="文档"><ErrorNote error={q.error} /></Section>;

  const docs = ((q.data as { documents?: DocRow[] } | undefined)?.documents ??
    []) as DocRow[];
  const primary = docs.filter((d) => (d.relation ?? "primary") === "primary");
  const related = docs.filter((d) => d.relation === "related");

  return (
    <Section
      title={`文档 (${docs.length})`}
      subtitle="文献报告 / 实验计划 / 实验报告 / 论文草稿 / 评审 —— 都存在这个实验的文件夹里，每次修订都是新版本，永不覆盖。"
    >
      {docs.length === 0 ? (
        <div className="space-y-1 text-sm text-mast-muted">
          <p>这个实验还没有文档。</p>
          <p className="text-xs">
            让智能体写一份报告或文献综述后，它会保存到实验文件夹的{" "}
            <code className="font-mono text-[11px]">reports/</code>{" "}
            下；实验计划保存到{" "}
            <code className="font-mono text-[11px]">plans/</code>。
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <DocList rows={primary} />
          {related.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-mast-muted">
                关联文档（主归属在其他实验，声明了与本实验相关）
              </p>
              <DocList rows={related} showOwner />
            </div>
          )}
        </div>
      )}
    </Section>
  );
}

/** 文档列表 + 点开读全文。实验详情页和全局「报告」tab 共用这一份。 */
export function DocList({ rows, showOwner = false }: { rows: DocRow[]; showOwner?: boolean }) {
  const [openId, setOpenId] = useState<string | null>(null);
  if (rows.length === 0) return null;
  return (
    <ul className="space-y-2">
      {rows.map((d) => (
        <li key={d.doc_id} className="rounded-lg border border-mast-border">
          <button
            className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left hover:bg-mast-bg/40"
            onClick={() => setOpenId(openId === d.doc_id ? null : d.doc_id)}
          >
            <Badge tone={KIND_TONE[d.kind] ?? "default"}>
              {d.kind_label || d.kind}
            </Badge>
            <span className="text-sm font-medium text-mast-text">{d.title}</span>
            <span
              className="font-mono text-[11px] text-mast-muted"
              title={`共 ${d.versions_count ?? d.version ?? 1} 个版本`}
            >
              v{String(d.version ?? 1).padStart(3, "0")}
            </span>
            {d.verdict && <Badge tone={verdictTone(d.verdict)}>{d.verdict}</Badge>}
            {typeof d.words === "number" && (
              <span className="text-[11px] text-mast-muted">{d.words} 词</span>
            )}
            {showOwner && d.experiment_name && (
              <span className="text-[11px] text-mast-muted">
                属于《{d.experiment_name}》
              </span>
            )}
            {d.created_by && (
              <span className="text-[11px] text-mast-muted" title="谁写的这一版">
                {d.created_by.startsWith("agent:")
                  ? d.created_by.slice("agent:".length)
                  : d.created_by === "operator"
                    ? "用户"
                    : d.created_by}
              </span>
            )}
            <span className="ml-auto text-[11px] text-mast-muted">
              {d.updated_at ? fmtTime(d.updated_at) : ""}
            </span>
          </button>
          {openId === d.doc_id && (
            <DocBody
              docId={d.doc_id}
              path={d.path ?? ""}
              conversationId={d.conversation_id ?? null}
              rootKind={d.root_kind ?? null}
            />
          )}
        </li>
      ))}
    </ul>
  );
}

function DocBody({
  docId,
  path,
  conversationId = null,
  rootKind = null,
}: {
  docId: string;
  path: string;
  conversationId?: string | null;
  rootKind?: string | null;
}) {
  const [version, setVersion] = useState<number | null>(null);
  const q = useQuery({
    queryKey: ["document", docId, version],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/documents/{doc_id}", {
        params: {
          path: { doc_id: docId },
          ...(version ? { query: { version } } : {}),
        },
      });
      if (error) throw error;
      return data;
    },
  });

  if (q.isLoading) return <Spinner label="加载全文…" />;
  if (q.isError) return <ErrorNote error={q.error} />;

  const d = q.data as
    | { content?: string | null; body?: string | null; detail?: string | null;
        versions?: { version: number; created_by?: string | null;
                     created_at?: string | null; words?: number | null }[] }
    | undefined;
  // 正文字段名兼容：新契约是 content，旧的是 body。
  const text = d?.content ?? d?.body ?? "";
  const versions = d?.versions ?? [];

  if (!text && d?.detail) {
    return <p className="px-3 pb-3 text-xs text-mast-danger">{d.detail}</p>;
  }

  return (
    <div className="border-t border-mast-border">
      <div className="flex flex-wrap items-center gap-2 border-b border-mast-border px-3 py-1.5">
        {versions.length > 1 && (
          <label className="flex items-center gap-1 text-[11px] text-mast-muted">
            版本
            <select
              className="rounded border border-mast-border bg-transparent px-1 py-0.5 font-mono text-[11px]"
              value={version ?? versions.at(-1)?.version ?? 1}
              onChange={(e) => setVersion(Number(e.target.value))}
            >
              {[...versions].reverse().map((v) => (
                <option key={v.version} value={v.version}>
                  v{String(v.version).padStart(3, "0")}
                  {v.created_by ? ` · ${v.created_by}` : ""}
                </option>
              ))}
            </select>
          </label>
        )}
        {/* 磁盘上的完整路径 —— 用户一直想要的那个东西 */}
        <p
          className="select-all font-mono text-[10px] text-mast-muted"
          title="文件在磁盘上的完整路径（可直接复制）"
        >
          {path}
        </p>
        {/* 来自哪次对话。转录在实验文件夹的 chats/ 下，按这个 id 前 8 位开头。 */}
        {conversationId && (
          <p
            className="select-all font-mono text-[10px] text-mast-muted"
            title="产出这份文档的对话 id —— 转录在实验文件夹的 chats/ 里，文件名以它的前 8 位开头"
          >
            对话 {conversationId.slice(0, 8)}
          </p>
        )}
      </div>
      <div className="max-h-[520px] overflow-auto px-3 py-2">
        <MarkdownView text={text} />
      </div>
      <ExportRow docId={docId} version={version} />
      <DiscardControls docId={docId} rootKind={rootKind} />
    </div>
  );
}

/** 导出三格式。**看的那份和被导出的那份必须是同一版** —— 所以带上 version。 */
function ExportRow({ docId, version }: { docId: string; version: number | null }) {
  // 不设 a.download：让服务端 content-disposition 的
  // `filename*=UTF-8''<标题>_vNNN.<ext>` 生效。ArtifactEditor 那边硬编码
  // `${id}.txt`，下载下来是一堆过一周谁也认不出的文件。
  function go(format: "md" | "html" | "docx") {
    const q = new URLSearchParams({ format });
    if (version) q.set("version", String(version));
    const a = document.createElement("a");
    a.href = `/api/documents/${encodeURIComponent(docId)}/export?${q}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-mast-border px-3 py-1.5">
      <span className="text-[11px] text-mast-muted">导出</span>
      <button
        className="rounded border border-mast-border px-2 py-0.5 text-[11px] text-mast-muted hover:bg-mast-bg/40"
        onClick={() => go("md")}
        title="原始 markdown —— 工作格式，图是相对链接（离开实验文件夹会断）"
      >
        Markdown
      </button>
      <button
        className="rounded border border-mast-border px-2 py-0.5 text-[11px] text-mast-muted hover:bg-mast-bg/40"
        onClick={() => go("html")}
        title="自包含 HTML —— 图 base64 内嵌，单个文件双击就开，发给别人「看」"
      >
        HTML
      </button>
      <button
        className="rounded border border-mast-border px-2 py-0.5 text-[11px] text-mast-muted hover:bg-mast-bg/40"
        onClick={() => go("docx")}
        title="Word 文档 —— 全文挂 Word 内置样式，收件人可用修订/批注来「改」，也能套期刊模板"
      >
        Word
      </button>
      <span className="text-[11px] text-mast-muted">
        HTML 发给别人看，Word 发给别人改。每次导出都在实验的{" "}
        <code className="font-mono text-[10px]">exports/</code> 里留一份带时间戳的档。
      </span>
    </div>
  );
}

/** 废弃 / 恢复。**没有「删除」** —— 废弃只是把目录搬进 `_discarded`，字节一个不少。 */
function DiscardControls({
  docId,
  rootKind,
}: {
  docId: string;
  rootKind?: string | null;
}) {
  const qc = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const [notice, setNotice] = useState("");
  const invalidate = () => qc.invalidateQueries({ queryKey: ["documents"] });

  const discard = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/documents/{doc_id}/discard", {
        params: { path: { doc_id: docId }, query: { reason: "用户在界面上废弃" } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (r) => {
      setConfirming(false);
      setNotice((r as { detail?: string })?.detail || "已废弃（可恢复）");
      invalidate();
    },
    onError: (e) => setNotice(`废弃失败：${String((e as Error)?.message ?? e)}`),
  });

  const restore = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/documents/{doc_id}/restore", {
        params: { path: { doc_id: docId }, query: { experiment_id: "" } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (r) => {
      setNotice((r as { detail?: string })?.detail || "已恢复到未归属区，可再认领");
      invalidate();
    },
    onError: (e) => setNotice(`恢复失败：${String((e as Error)?.message ?? e)}`),
  });

  const discarded = rootKind === "discarded";
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-mast-border px-3 py-1.5">
      {discarded ? (
        <button
          className="rounded border border-mast-border px-2 py-0.5 text-[11px] text-mast-muted hover:bg-mast-bg/40"
          onClick={() => restore.mutate()}
          disabled={restore.isPending}
        >
          恢复这份文档
        </button>
      ) : confirming ? (
        <>
          <button
            className="rounded border border-mast-warn-border bg-mast-warn-bg px-2 py-0.5 text-[11px] text-mast-warn"
            onClick={() => discard.mutate()}
            disabled={discard.isPending}
          >
            确认废弃
          </button>
          <button
            className="rounded border border-mast-border px-2 py-0.5 text-[11px] text-mast-muted hover:bg-mast-bg/40"
            onClick={() => setConfirming(false)}
          >
            取消
          </button>
          <span className="text-[11px] text-mast-muted">
            不会删除任何文件 —— 只是从列表里收起来，随时可恢复。
          </span>
        </>
      ) : (
        <button
          className="rounded border border-mast-border px-2 py-0.5 text-[11px] text-mast-muted hover:bg-mast-bg/40"
          onClick={() => setConfirming(true)}
          title="搬进 _discarded 区；不删除任何字节，随时可恢复"
        >
          废弃
        </button>
      )}
      {notice && <span className="text-[11px] text-mast-muted">{notice}</span>}
    </div>
  );
}
