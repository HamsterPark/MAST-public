import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, Badge, DegradedNote } from "@/components/ui";
import { Field, TextField, Toggle, Button } from "@/components/controls";
import { LibraryRef, LibraryTargetNote } from "@/components/literature/LibraryTargetNote";
import type { components } from "@/api/schema";

type IngestResponse = components["schemas"]["IngestResponse"];

// #125 — 用户直接管理条目 + 附加 PDF.
//   PdfUploadPanel   — pick a PDF in the browser → multipart POST
//                      /api/literature/upload-pdf → ingested into the big library
//                      + a pointer added to the chosen library.
//   ManualEntryPanel — hand-enter title/abstract/authors → POST
//                      /api/literature/manual-entry for a paper not in OpenAlex
//                      and with no PDF (mints a local:<hash> work_id).

// ── shared result renderer ───────────────────────────────────────────────────
function IngestResultView({
  result,
  isError,
  error,
  label,
}: {
  result?: IngestResponse;
  isError: boolean;
  error: unknown;
  label: string;
}) {
  if (isError)
    return (
      <p className="mt-3 text-sm text-mast-danger">
        {label}失败：{String((error as Error)?.message ?? error)}
      </p>
    );
  const r = result;
  if (!r) return null;
  const ok = r.ok;
  return (
    <div className="mt-3 space-y-1 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={ok ? "AUTO" : r.degraded ? "WARN" : "DANGEROUS"}>
          {r.status || (ok ? "ingested" : "failed")}
        </Badge>
        {r.promoted && <Badge tone="INFO">已促进大库</Badge>}
        {r.pointer_added && <Badge tone="INFO">已加入库指针</Badge>}
        {r.fulfilled_requests > 0 && (
          <Badge tone="INFO">满足 {r.fulfilled_requests} 条取文请求</Badge>
        )}
      </div>
      {r.degraded && <DegradedNote what="文献库后端" />}
      {r.work_id && (
        <p className="text-mast-muted">
          work_id <code>{r.work_id}</code>
          {r.title && <> · {r.title}</>}
          {/* 回显**实际**落库，不是请求里那个（可能为空的）值 —— 空 library_id 现在
              解析成有效库，不说清楚用户就不知道论文进了哪儿。 */}
          {r.pointer_library_id && (
            <>
              {" "}
              · 已入库{" "}
              <LibraryRef
                libraryId={r.pointer_library_id}
                source={r.pointer_library_source}
              />
              {r.pointer_library_source === "experiment" && "（当前实验的专属库）"}
              {r.pointer_library_source === "manual" && "（手动指定的库）"}
              {r.pointer_library_source === "fallback" && "（默认兜底库）"}
            </>
          )}
          {r.fulltext_ref && (
            <>
              {" "}
              · 全文 <code>{r.fulltext_ref}</code>
            </>
          )}
        </p>
      )}
      {r.detail && <p className="text-mast-muted">{r.detail}</p>}
    </div>
  );
}

// ── PDF 上传 ──────────────────────────────────────────────────────────────────
export function PdfUploadPanel({ libraryId }: { libraryId: string | null }) {
  const qc = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [workId, setWorkId] = useState("");
  const [firstAuthor, setFirstAuthor] = useState("");
  const [year, setYear] = useState("");
  const [promote, setPromote] = useState(true);
  const fileInput = useRef<HTMLInputElement>(null);

  const m = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("先选择一个 PDF 文件");
      // The multipart body carries a File → a raw fetch with FormData is the
      // clean path (openapi-fetch types the binary field as string). The
      // response is the same typed IngestResponse the JSON endpoints return.
      const fd = new FormData();
      fd.append("file", file);
      fd.append("library_id", libraryId ?? "");
      fd.append("work_id", workId.trim());
      fd.append("promote", String(promote));
      fd.append("first_author", firstAuthor.trim());
      fd.append("year", year.trim());
      const resp = await fetch("/api/literature/upload-pdf", {
        method: "POST",
        body: fd,
      });
      if (!resp.ok) throw new Error(`上传失败 (HTTP ${resp.status})`);
      return (await resp.json()) as IngestResponse;
    },
    onSuccess: (d) => {
      if (d?.ok) {
        setFile(null);
        setWorkId("");
        if (fileInput.current) fileInput.current.value = "";
      }
      qc.invalidateQueries({ queryKey: ["literature", "libraries"] });
      qc.invalidateQueries({ queryKey: ["literature", "fetch-board"] });
    },
  });

  return (
    <Card>
      <h3 className="mb-2 text-sm font-semibold">上传 PDF</h3>
      <p className="mb-3 text-xs text-mast-muted">
        从本机选一个 PDF 上传 → 提取摘要/元数据并促进进大库
        {libraryId ? (
          <>
            ，并作为指针加入库 <code>{libraryId}</code>
          </>
        ) : (
          "（未指定库 → 见下方落点）"
        )}
        。扫描版/无文本层 PDF 会被识别并拒绝（不会污染大库）。
      </p>
      <LibraryTargetNote libraryId={libraryId} />
      <div className="space-y-3">
        <Field label="PDF 文件">
          <input
            ref={fileInput}
            type="file"
            accept="application/pdf,.pdf"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block w-full text-sm text-mast-muted file:mr-3 file:rounded file:border-0 file:bg-mast-accent/20 file:px-3 file:py-1.5 file:text-sm file:text-mast-accent hover:file:bg-mast-accent/30"
          />
        </Field>
        {file && (
          <p className="text-xs text-mast-muted">
            已选：<code>{file.name}</code>（{(file.size / 1024 / 1024).toFixed(2)} MB）
          </p>
        )}
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
          disabled={!file || m.isPending}
          onClick={() => m.mutate()}
        >
          {m.isPending ? "上传摄取中…" : "上传并摄取"}
        </Button>
      </div>
      <IngestResultView result={m.data} isError={m.isError} error={m.error} label="上传" />
    </Card>
  );
}

// ── 手动新增条目 ──────────────────────────────────────────────────────────────
export function ManualEntryPanel({ libraryId }: { libraryId: string | null }) {
  const qc = useQueryClient();
  const [title, setTitle] = useState("");
  const [abstract, setAbstract] = useState("");
  const [authors, setAuthors] = useState("");
  const [year, setYear] = useState("");
  const [doi, setDoi] = useState("");
  const [journal, setJournal] = useState("");

  const m = useMutation({
    mutationFn: async () => {
      if (!title.trim() && !abstract.trim())
        throw new Error("至少填写标题或摘要");
      const { data, error } = await api.POST("/api/literature/manual-entry", {
        body: {
          title: title.trim(),
          abstract: abstract.trim(),
          work_id: "",
          first_author: "",
          authors: authors.trim(),
          year: year.trim(),
          doi: doi.trim(),
          journal: journal.trim(),
          library_id: libraryId ?? "",
        },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d?.ok) {
        setTitle("");
        setAbstract("");
        setAuthors("");
        setYear("");
        setDoi("");
        setJournal("");
      }
      qc.invalidateQueries({ queryKey: ["literature", "libraries"] });
    },
  });

  return (
    <Card>
      <h3 className="mb-2 text-sm font-semibold">手动新增条目</h3>
      <p className="mb-3 text-xs text-mast-muted">
        为不在 OpenAlex、也没有 PDF 的论文手动建条目（会铸造一个 <code>local:</code> work_id）。
        填了摘要即可按 work_id 检索；有嵌入 key 时还会向量化，进而支持语义检索。
        {libraryId ? (
          <>
            {" "}
            条目会作为指针加入库 <code>{libraryId}</code>。
          </>
        ) : (
          "（未指定库 → 见下方落点）"
        )}
      </p>
      <LibraryTargetNote libraryId={libraryId} />
      <div className="space-y-3">
        <Field label="标题">
          <TextField value={title} onChange={setTitle} placeholder="论文标题" />
        </Field>
        <Field label="摘要">
          <textarea
            value={abstract}
            onChange={(e) => setAbstract(e.target.value)}
            rows={4}
            placeholder="摘要 / 概要（用于检索）"
            className="w-full rounded-md border border-mast-border bg-mast-bg px-2.5 py-1.5 text-sm text-mast-text outline-none focus:border-mast-accent"
          />
        </Field>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Field label="作者（可选）">
            <TextField value={authors} onChange={setAuthors} placeholder="A. Bee, C. Dee" />
          </Field>
          <Field label="期刊 / 会议（可选）">
            <TextField value={journal} onChange={setJournal} placeholder="Phys. Rev. Lett." />
          </Field>
          <Field label="年份（可选）">
            <TextField value={year} onChange={setYear} placeholder="2021" />
          </Field>
          <Field label="DOI（可选）">
            <TextField value={doi} onChange={setDoi} placeholder="10.1103/…" mono />
          </Field>
        </div>
        <Button
          variant="primary"
          disabled={(!title.trim() && !abstract.trim()) || m.isPending}
          onClick={() => m.mutate()}
        >
          {m.isPending ? "新增中…" : "新增条目"}
        </Button>
      </div>
      <IngestResultView result={m.data} isError={m.isError} error={m.error} label="新增" />
    </Card>
  );
}
