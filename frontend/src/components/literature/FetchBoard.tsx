import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, Badge, Spinner, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";
import { Button, useToast } from "@/components/controls";
import type { components } from "@/api/schema";

type IngestResponse = components["schemas"]["IngestResponse"];

// 取文请求板 — GET /api/literature/fetch-board. The literature agent posts a
// full-text request here when the big library only has the abstract; the
// operator satisfies it by uploading the PDF right on the request row.
// Resolve via POST /api/literature/fetch-board/{request_id}/resolve
// {action:done|dismissed}.
//
// Uploading in place, rather than sending the operator to the 摄取与取文 sub-page
// with a copied work_id, matters for more than convenience: the upload carries
// the request's own work_id, so the paper is filed against the experiment frozen
// on that request and the agent that asked gets woken up. A hand-retyped id in
// another form is where that chain used to break.

const STATUS_ICON: Record<string, string> = {
  pending: "🟡",
  fulfilled: "✅",
  failed: "🔴",
  dismissed: "⚪",
};

const STATUS_TONE: Record<string, string> = {
  pending: "WARN",
  fulfilled: "AUTO",
  failed: "DANGEROUS",
  dismissed: "INFO",
};

const FILTERS = [
  { id: "", label: "全部" },
  { id: "pending", label: "待处理" },
  { id: "fulfilled", label: "已满足" },
  { id: "failed", label: "失败" },
  { id: "dismissed", label: "已忽略" },
] as const;

/** Satisfy one request in place: upload its full text, then optionally its SI. */
function SatisfyRow({
  workId,
  onDone,
}: {
  workId: string;
  onDone: (msg: string, ok: boolean) => void;
}) {
  const qc = useQueryClient();
  const pdfInput = useRef<HTMLInputElement>(null);
  const siInput = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState<"" | "pdf" | "si">("");

  const uploadPdf = async (file: File) => {
    setBusy("pdf");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("work_id", workId);
      fd.append("promote", "true");
      const resp = await fetch("/api/literature/upload-pdf", { method: "POST", body: fd });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const d = (await resp.json()) as IngestResponse;
      if (d.ok) {
        const woke = (d.fulfilled_requests ?? 0) > 0;
        onDone(
          woke
            ? "全文已入库，取文请求已关闭 —— 文献 agent 会自动继续之前的工作。"
            : "全文已入库。",
          true,
        );
      } else {
        onDone(d.detail || d.status || "入库失败", false);
      }
      qc.invalidateQueries({ queryKey: ["literature", "fetch-board"] });
      qc.invalidateQueries({ queryKey: ["literature", "libraries"] });
    } catch (e) {
      onDone(`上传失败：${String((e as Error)?.message ?? e)}`, false);
    } finally {
      setBusy("");
      if (pdfInput.current) pdfInput.current.value = "";
    }
  };

  const uploadSi = async (file: File) => {
    setBusy("si");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("work_id", workId);
      fd.append("label", file.name);
      const resp = await fetch("/api/literature/attach-si", { method: "POST", body: fd });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const d = (await resp.json()) as { ok?: boolean; error?: string; duplicate?: boolean };
      if (d.ok) onDone(d.duplicate ? "该补充材料已经挂过了。" : "补充材料已附加。", true);
      else onDone(d.error || "附加失败", false);
      qc.invalidateQueries({ queryKey: ["literature", "attachments", workId] });
    } catch (e) {
      onDone(`附加失败：${String((e as Error)?.message ?? e)}`, false);
    } finally {
      setBusy("");
      if (siInput.current) siInput.current.value = "";
    }
  };

  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs">
      <input
        ref={pdfInput}
        type="file"
        accept="application/pdf,.pdf"
        disabled={!workId || busy !== ""}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void uploadPdf(f);
        }}
        className="max-w-[15rem] text-xs text-mast-muted file:mr-2 file:rounded file:border-0 file:bg-mast-accent/20 file:px-2 file:py-0.5 file:text-xs file:text-mast-accent disabled:opacity-40"
        title="选择这篇论文的全文 PDF，直接满足该请求"
      />
      <input
        ref={siInput}
        type="file"
        accept="application/pdf,.pdf"
        disabled={!workId || busy !== ""}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void uploadSi(f);
        }}
        className="max-w-[15rem] text-xs text-mast-muted file:mr-2 file:rounded file:border-0 file:bg-mast-bg/60 file:px-2 file:py-0.5 file:text-xs file:text-mast-muted disabled:opacity-40"
        title="附加补充材料（SI）—— 需要先上传正文"
      />
      <span className="text-mast-muted">
        {busy === "pdf" ? "上传全文中…" : busy === "si" ? "附加 SI 中…" : "左：全文 PDF ｜ 右：补充材料 SI"}
      </span>
    </div>
  );
}

export function FetchBoard() {
  const [status, setStatus] = useState<string>("");
  const { toast, node: toastNode } = useToast();
  const qc = useQueryClient();

  const resolveM = useMutation({
    mutationFn: async (vars: { requestId: string; action: "done" | "dismissed" }) => {
      const { data, error } = await api.POST(
        "/api/literature/fetch-board/{request_id}/resolve",
        {
          params: { path: { request_id: vars.requestId } },
          body: { action: vars.action, note: "" },
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: (d, vars) => {
      if (d?.degraded) {
        toast("取文请求板后端不可用，未能处理。", "err");
      } else if (d?.ok) {
        toast(vars.action === "done" ? "已标记完成" : "已忽略", "ok");
      } else {
        toast(d?.message || "未能处理该请求", "err");
      }
      qc.invalidateQueries({ queryKey: ["literature", "fetch-board"] });
    },
    onError: (e) =>
      toast(`处理失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const copyWorkId = async (wid: string) => {
    if (!wid) return;
    try {
      await navigator.clipboard.writeText(wid);
      toast(`已复制 work_id：${wid}`, "ok");
    } catch {
      toast("复制失败（剪贴板不可用）", "err");
    }
  };

  const q = useQuery({
    queryKey: ["literature", "fetch-board", status],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/literature/fetch-board", {
        params: { query: { status: status || null } },
      });
      if (error) throw error;
      return data;
    },
  });

  const reqs = q.data?.requests ?? [];

  return (
    <Card>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">取文请求板</h3>
          {q.data && q.data.pending_count > 0 && (
            <Badge tone="WARN">{q.data.pending_count} 待处理</Badge>
          )}
        </div>
        <div className="flex flex-wrap gap-1">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setStatus(f.id)}
              className={
                "rounded px-2 py-1 text-xs " +
                (status === f.id
                  ? "bg-mast-accent/20 text-mast-accent"
                  : "text-mast-muted hover:bg-mast-bg/40 hover:text-mast-text")
              }
            >
              {f.label}
            </button>
          ))}
          <Button variant="ghost" onClick={() => q.refetch()}>
            刷新
          </Button>
        </div>
      </div>

      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="取文请求板" />}
      {q.data && !q.data.degraded && reqs.length === 0 && (
        <EmptyNote label="取文请求板为空。文献 agent 需要某篇全文时会在此提交请求，你上传 PDF 或取文即可满足。" />
      )}

      {q.data && !q.data.degraded && reqs.length > 0 && (
        <div className="divide-y divide-mast-border overflow-hidden rounded border border-mast-border">
          {reqs.map((r) => (
            <div key={r.request_id || r.work_id} className="px-3 py-2 text-sm">
              <div className="flex flex-wrap items-center gap-1.5">
                <span>{STATUS_ICON[r.status] ?? "•"}</span>
                <code className="text-mast-accent">{r.work_id || "(无 work_id)"}</code>
                {r.title && <span className="text-mast-text">«{r.title}»</span>}
                <Badge tone={STATUS_TONE[r.status] ?? "INFO"}>{r.status}</Badge>
                <span className="text-xs text-mast-muted">
                  {r.requested_by}
                  {r.request_id && (
                    <>
                      {" · "}
                      <code>{r.request_id}</code>
                    </>
                  )}
                </span>
              </div>
              {r.doi && (
                <div className="mt-0.5 text-xs text-mast-muted">
                  DOI <code>{r.doi}</code>
                </div>
              )}
              {r.reason && (
                <div className="mt-0.5 text-mast-muted">理由：{r.reason}</div>
              )}
              {r.note && (
                <div className="mt-0.5 text-xs text-mast-muted">{r.note}</div>
              )}
              <div className="mt-1.5 flex flex-wrap gap-1">
                <button
                  onClick={() => copyWorkId(r.work_id)}
                  disabled={!r.work_id}
                  className="rounded px-2 py-0.5 text-xs text-mast-accent hover:bg-mast-accent/20 disabled:opacity-30"
                >
                  复制 work_id
                </button>
                <button
                  onClick={() =>
                    r.request_id &&
                    resolveM.mutate({ requestId: r.request_id, action: "done" })
                  }
                  disabled={!r.request_id || r.status !== "pending" || resolveM.isPending}
                  title={
                    r.request_id
                      ? "标记此取文请求为已完成"
                      : "该请求缺少 request_id，无法处理"
                  }
                  className="rounded px-2 py-0.5 text-xs text-mast-accent hover:bg-mast-accent/20 disabled:opacity-30"
                >
                  标记完成
                </button>
                <button
                  onClick={() =>
                    r.request_id &&
                    resolveM.mutate({
                      requestId: r.request_id,
                      action: "dismissed",
                    })
                  }
                  disabled={!r.request_id || r.status !== "pending" || resolveM.isPending}
                  title={
                    r.request_id
                      ? "忽略此取文请求"
                      : "该请求缺少 request_id，无法处理"
                  }
                  className="rounded px-2 py-0.5 text-xs text-mast-muted hover:bg-mast-bg/40 hover:text-mast-text disabled:opacity-30"
                >
                  忽略
                </button>
              </div>
              {r.status === "pending" && (
                <SatisfyRow
                  workId={r.work_id ?? ""}
                  onDone={(msg, ok) => toast(msg, ok ? "ok" : "err")}
                />
              )}
            </div>
          ))}
        </div>
      )}

      <p className="mt-3 px-1 text-xs text-mast-muted">
        提示：直接在请求行选择全文 PDF 即可满足它 —— 论文会入大库、请求自动关闭，提出请求的文献
        agent 会自动继续之前的工作。右侧可再附加补充材料（SI，需先传正文），精读时会一并读。
        也可切到「摄取与取文」子页按 DOI 取文；无需取文则点「忽略」（仅对待处理请求可用）。
        「标记完成」只改状态、不代表全文已入库，因此不会唤醒 agent。
      </p>
      {toastNode}
    </Card>
  );
}
