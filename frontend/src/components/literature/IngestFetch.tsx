import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, Badge, Spinner, ErrorNote, DegradedNote } from "@/components/ui";
import { Field, TextField, Toggle, Button } from "@/components/controls";
import { LibraryRef, LibraryTargetNote } from "@/components/literature/LibraryTargetNote";
import type { components } from "@/api/schema";

type IngestResponse = components["schemas"]["IngestResponse"];

// 摄取就绪 — readiness probe (GET /api/literature/ingest-status). Surfaces the
// dependency/key requirements BEFORE the user spends an upload (mirrors
// literature_panel.ingest_readiness_html).

export function IngestReadiness() {
  const q = useQuery({
    queryKey: ["literature", "ingest-status"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/literature/ingest-status");
      if (error) throw error;
      return data;
    },
  });

  const s = q.data;
  return (
    <Card>
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold">摄取就绪</h3>
        <Button variant="ghost" onClick={() => q.refetch()}>
          刷新
        </Button>
      </div>
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {s?.degraded && <DegradedNote what="摄取就绪探针" />}
      {s && !s.degraded && (
        <div className="space-y-2">
          <p
            className={
              "text-sm " + (s.available ? "text-mast-auto" : "text-mast-warn")
            }
          >
            {s.detail}
          </p>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge tone={s.available ? "AUTO" : "DANGEROUS"}>
              numpy/pandas {s.available ? "✓" : "✗"}
            </Badge>
            <Badge tone={s.can_extract_pdf ? "AUTO" : "WARN"}>
              PyMuPDF {s.can_extract_pdf ? "✓" : "✗"}
            </Badge>
            <Badge tone={s.has_embedder_key ? "AUTO" : "WARN"}>
              DashScope key {s.has_embedder_key ? "✓" : "✗"}
            </Badge>
            <Badge tone={s.can_fetch ? "AUTO" : "WARN"}>
              取文(httpx) {s.can_fetch ? "✓" : "✗"}
            </Badge>
          </div>
        </div>
      )}
    </Card>
  );
}

// PDF 摄取（服务器路径，高级）— POST /api/literature/ingest. Takes a
// SERVER-visible pdf_path (e.g. a file already on the machine, or a path an
// agent fetched). For the common case of uploading a PDF from your browser, use
// the「上传 PDF」panel above (POST /api/literature/upload-pdf, multipart) — this
// server-path route stays for files that are already on disk.

export function IngestPanel({ libraryId }: { libraryId: string | null }) {
  const qc = useQueryClient();
  const [pdfPath, setPdfPath] = useState("");
  const [workId, setWorkId] = useState("");
  const [firstAuthor, setFirstAuthor] = useState("");
  const [year, setYear] = useState("");
  const [promote, setPromote] = useState(true);

  const m = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/literature/ingest", {
        body: {
          pdf_path: pdfPath.trim(),
          work_id: workId.trim(),
          library_id: libraryId ?? "",
          source: "user_pdf",
          promote,
          first_author: firstAuthor.trim(),
          year: year.trim(),
        },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["literature", "libraries"] });
      qc.invalidateQueries({ queryKey: ["literature", "fetch-board"] });
    },
  });

  const res = m.data;
  return (
    <Card>
      <h3 className="mb-2 text-sm font-semibold">PDF 摄取（服务器路径 · 高级）</h3>
      <p className="mb-3 text-xs text-mast-muted">
        摄取一篇 PDF（服务器可见路径；浏览器上传请用上方「上传 PDF」）→ 摘要促进进大库
        {libraryId ? (
          <>
            ，并作为指针加入库 <code>{libraryId}</code>
          </>
        ) : (
          "（未指定库 → 见下方落点）"
        )}
        。
      </p>
      <LibraryTargetNote libraryId={libraryId} />
      <div className="space-y-3">
        <Field label="PDF 路径（服务器可见）" hint="例如：D:/papers/foo.pdf">
          <TextField value={pdfPath} onChange={setPdfPath} placeholder="…/paper.pdf" mono />
        </Field>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <Field label="work_id（可选）" hint="留空 → 解析/铸造">
            <TextField value={workId} onChange={setWorkId} placeholder="W…" mono />
          </Field>
          <Field label="第一作者（铸造提示，可选）">
            <TextField value={firstAuthor} onChange={setFirstAuthor} />
          </Field>
          <Field label="年份（铸造提示，可选）">
            <TextField value={year} onChange={setYear} />
          </Field>
        </div>
        <label className="flex items-center gap-2 text-sm text-mast-muted">
          <Toggle checked={promote} onChange={setPromote} label="promote" />
          促进摘要进大库
        </label>
        <Button
          variant="primary"
          disabled={!pdfPath.trim() || m.isPending}
          onClick={() => m.mutate()}
        >
          {m.isPending ? "摄取中…" : "摄取 PDF"}
        </Button>
      </div>
      <IngestResult result={res} isError={m.isError} error={m.error} />
    </Card>
  );
}

function IngestResult({
  result,
  isError,
  error,
}: {
  result?: IngestResponse;
  isError: boolean;
  error: unknown;
}) {
  if (isError)
    return (
      <p className="mt-3 text-sm text-mast-danger">
        摄取失败：{String((error as Error)?.message ?? error)}
      </p>
    );
  const r = result;
  if (!r) return null;
  return (
    <div className="mt-3 space-y-1 text-sm">
      <div className="flex items-center gap-2">
        <Badge tone={r.ok ? "AUTO" : r.degraded ? "WARN" : "DANGEROUS"}>
          {r.status || (r.ok ? "ingested" : "failed")}
        </Badge>
        {r.promoted && <Badge tone="INFO">已促进</Badge>}
      </div>
      {r.degraded && <DegradedNote what="摄取后端" />}
      {r.work_id && (
        <p className="text-mast-muted">
          work_id <code>{r.work_id}</code>
          {r.title && <> · {r.title}</>}
          {typeof r.n_chunks === "number" && r.n_chunks > 0 && (
            <> · {r.n_chunks} 块</>
          )}
          {r.library_id && (
            <>
              {" "}
              · 库 <LibraryRef libraryId={r.library_id} />
            </>
          )}
        </p>
      )}
      {r.detail && <p className="text-mast-muted">{r.detail}</p>}
    </div>
  );
}

// DOI 取文 — POST /api/literature/fetch. Best-effort, ToS-respecting full-text
// fetch (auto_oa resolves an OA PDF from the DOI). No paywall bypass exists in
// the core. On success it returns a pdf_path the user can then ingest above.

export function FetchPanel({ onFetchedPath }: { onFetchedPath?: (path: string) => void }) {
  const [doiOrUrl, setDoiOrUrl] = useState("");
  const [oaUrl, setOaUrl] = useState("");
  const [autoOa, setAutoOa] = useState(true);

  const m = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/literature/fetch", {
        body: {
          doi_or_url: doiOrUrl.trim(),
          oa_url: oaUrl.trim(),
          auto_oa: autoOa,
        },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d?.pdf_path && onFetchedPath) onFetchedPath(d.pdf_path);
    },
  });

  const res = m.data;
  return (
    <Card>
      <h3 className="mb-2 text-sm font-semibold">DOI / URL 取文（实验性）</h3>
      <p className="mb-3 text-xs text-mast-muted">
        尽力且尊重 robots/ToS 地获取开放获取全文；不绕付费墙。取到后可在上方摄取。
      </p>
      <div className="space-y-3">
        <Field label="DOI 或 URL" hint="10.xxxx/… 或 https://…">
          <TextField value={doiOrUrl} onChange={setDoiOrUrl} placeholder="10.1103/PhysRevLett…" mono />
        </Field>
        <Field label="已知 OA PDF URL（可选，优先尝试）">
          <TextField value={oaUrl} onChange={setOaUrl} placeholder="https://…/paper.pdf" mono />
        </Field>
        <label className="flex items-center gap-2 text-sm text-mast-muted">
          <Toggle checked={autoOa} onChange={setAutoOa} label="auto_oa" />
          从 DOI 自动解析 OA PDF
        </label>
        <Button
          variant="primary"
          disabled={!doiOrUrl.trim() || m.isPending}
          onClick={() => m.mutate()}
        >
          {m.isPending ? "取文中…" : "取文"}
        </Button>
      </div>
      {m.isError && (
        <p className="mt-3 text-sm text-mast-danger">
          取文失败：{String((m.error as Error)?.message ?? m.error)}
        </p>
      )}
      {res && (
        <div className="mt-3 space-y-1 text-sm">
          <div className="flex items-center gap-2">
            <Badge tone={res.pdf_path ? "AUTO" : res.degraded ? "WARN" : "WARN"}>
              {res.status}
            </Badge>
            {res.source && <Badge tone="INFO">{res.source}</Badge>}
          </div>
          {res.degraded && <DegradedNote what="取文后端" />}
          {res.pdf_path && (
            <p className="text-mast-muted">
              已取到全文 → <code>{res.pdf_path}</code>
              {onFetchedPath && "（已填入上方 PDF 路径）"}
            </p>
          )}
          {res.message && <p className="text-mast-muted">{res.message}</p>}
          {res.url && (
            <p className="text-xs text-mast-muted">
              来源：<code>{res.url}</code>
            </p>
          )}
        </div>
      )}
    </Card>
  );
}
