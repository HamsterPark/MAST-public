import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, Field, Modal, SubTabs, useToast } from "@/components/controls";
import { agentLabel } from "./registry";

// ════════════════════════════════════════════════════════════════════════════
// ArtifactEditor — operator editor for a single workspace artifact (Parity F2).
//
// The old Gradio 对象编辑 button was DEAD (called a route that never existed).
// Wave F1 built the real backend; this wires it:
//
//   load body   GET  /api/artifacts/{id}/export?format=json  → {body, source}
//   保存        POST   /api/artifacts/{id}/edit  {body}
//   还原        DELETE /api/artifacts/{id}/edit
//   历史        GET    /api/artifacts/{id}/history  → {entries:[{body,t}], current}
//   查看 diff   GET    /api/artifacts/{id}/diff     → {diff, changed, has_*}
//   导出        GET    /api/artifacts/{id}/export   → text/plain download
//
// Opened as a Modal from each artifact row in ArtifactsView. Degrade-safe: every
// read/write surfaces degraded / error / empty and never freezes the page.
// ════════════════════════════════════════════════════════════════════════════

type Tab = "edit" | "diff" | "history";

function fmtTime(t?: number | null): string {
  if (t == null) return "—";
  try {
    return new Date(t * 1000).toLocaleString();
  } catch {
    return "—";
  }
}

/** Render a unified diff with +/- line coloring. */
function DiffView({ diff }: { diff: string }) {
  const lines = diff.split("\n");
  return (
    <pre className="max-h-[420px] overflow-auto rounded-md border border-mast-border bg-mast-bg/60 p-3 font-mono text-xs leading-relaxed">
      {lines.map((ln, i) => {
        let cls = "text-mast-muted";
        if (ln.startsWith("+++") || ln.startsWith("---")) cls = "text-mast-info font-semibold";
        else if (ln.startsWith("@@")) cls = "text-mast-dream";
        else if (ln.startsWith("+")) cls = "bg-mast-auto-bg text-mast-auto";
        else if (ln.startsWith("-")) cls = "bg-mast-danger-bg text-mast-danger";
        return (
          <div key={i} className={cls}>
            {ln || " "}
          </div>
        );
      })}
    </pre>
  );
}

export function ArtifactEditor({
  artifactId,
  producer,
  open = true,
  onClose,
  inline = false,
}: {
  artifactId: string | null;
  producer?: string | null;
  /** Modal mode: drives the Modal open state. Ignored when `inline`. */
  open?: boolean;
  onClose?: () => void;
  /** Render inline (on its own sub-tab) instead of inside a Modal. */
  inline?: boolean;
}) {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const [tab, setTab] = useState<Tab>("edit");
  const [draft, setDraft] = useState("");
  const [loadedFor, setLoadedFor] = useState<string | null>(null);

  const id = artifactId ?? "";

  // current effective body (operator edit if present, else produced/original).
  // The export route streams text/plain by default and has no typed response_model
  // (OpenAPI → `unknown`), so we request format=json and narrow the JSON shape.
  const body = useQuery({
    queryKey: ["artifact-edit", "body", id],
    enabled: open && !!artifactId,
    queryFn: async (): Promise<{
      body?: string | null;
      source?: string | null;
      filename?: string | null;
      degraded?: boolean;
      detail?: string | null;
    }> => {
      const { data, error } = await api.GET("/api/artifacts/{artifact_id}/export", {
        params: { path: { artifact_id: id }, query: { format: "json" } },
      });
      if (error) throw error;
      return (data ?? {}) as {
        body?: string | null;
        source?: string | null;
        filename?: string | null;
        degraded?: boolean;
        detail?: string | null;
      };
    },
  });

  // seed the textarea from the loaded body once per artifact open
  useEffect(() => {
    if (!open) {
      setLoadedFor(null);
      return;
    }
    if (body.data && loadedFor !== id) {
      setDraft(body.data.body ?? "");
      setLoadedFor(id);
    }
  }, [open, id, body.data, loadedFor]);

  // reset tab whenever a new artifact opens
  useEffect(() => {
    if (open) setTab("edit");
  }, [open, id]);

  const diff = useQuery({
    queryKey: ["artifact-edit", "diff", id],
    enabled: open && !!artifactId && tab === "diff",
    queryFn: async () => {
      const { data, error } = await api.GET("/api/artifacts/{artifact_id}/diff", {
        params: { path: { artifact_id: id } },
      });
      if (error) throw error;
      return data;
    },
  });

  const history = useQuery({
    queryKey: ["artifact-edit", "history", id],
    enabled: open && !!artifactId && tab === "history",
    queryFn: async () => {
      const { data, error } = await api.GET("/api/artifacts/{artifact_id}/history", {
        params: { path: { artifact_id: id } },
      });
      if (error) throw error;
      return data;
    },
  });

  function invalidateAll() {
    qc.invalidateQueries({ queryKey: ["artifact-edit", "body", id] });
    qc.invalidateQueries({ queryKey: ["artifact-edit", "diff", id] });
    qc.invalidateQueries({ queryKey: ["artifact-edit", "history", id] });
    qc.invalidateQueries({ queryKey: ["artifacts"] });
  }

  const save = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/artifacts/{artifact_id}/edit", {
        params: { path: { artifact_id: id } },
        body: { body: draft },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.ok && !data.degraded) {
        // The detail carries the new version's filename — the operator needs to
        // know their edit became a version, not an overwrite.
        toast(data.detail ?? "已保存为新版本", "ok");
        setLoadedFor(null); // re-seed from server on next render
        invalidateAll();
      } else {
        toast(data?.detail ?? "保存未生效", "err");
      }
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  const revert = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.DELETE("/api/artifacts/{artifact_id}/edit", {
        params: { path: { artifact_id: id } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.ok || (data && !data.degraded)) {
        toast(data.detail ?? (data.reverted ? "已回退到上一版本" : "没有可回退的历史"), "ok");
        setLoadedFor(null);
        invalidateAll();
      } else {
        toast(data?.detail ?? "回退未生效", "err");
      }
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  // 导出 — stream the current effective body as a text/plain download.
  function exportBody() {
    if (!artifactId) return;
    const url = `/api/artifacts/${encodeURIComponent(id)}/export?format=text`;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${id}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  const histEntries = history.data?.entries ?? [];

  const inner = (
    <>
      {node}
      {artifactId && (
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs text-mast-muted">
          {producer && <span>产出 · {agentLabel(producer)}</span>}
          {body.data?.filename && (
            <Badge tone="INFO">当前版本 · {body.data.filename}</Badge>
          )}
        </div>
      )}

      <SubTabs<Tab>
        tabs={[
          { id: "edit", label: "编辑" },
          { id: "diff", label: "差异" },
          { id: "history", label: "历史" },
        ]}
        value={tab}
        onChange={setTab}
      />

      {/* ── 编辑 ── */}
      {tab === "edit" && (
        <div className="space-y-3">
          {body.isPending && <Spinner />}
          {body.isError && <ErrorNote error={body.error} />}
          {body.data && body.data.degraded ? (
            // Non-editable / missing file: show ONLY the honest reason. Rendering
            // an editable-looking textarea + 保存 over a body the backend refuses
            // to save is exactly the write-only "black hole" this editor replaced
            // — the caller (AgentsPage) also stops such artifacts being clickable,
            // this is the defense-in-depth for any other opener.
            <EmptyNote label={body.data.detail ?? "该产物不可手工编辑"} />
          ) : (
            <>
              <Field
                label="文档正文"
                hint="保存 = 存为新版本（不覆盖智能体的版本）；下一个打开该文档的智能体会读到你的修改。回退 = 把上一版内容重新存为新版本，不删除任何历史。"
              >
                <textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  rows={16}
                  className="w-full resize-y rounded-md border border-mast-border bg-mast-bg px-3 py-2 font-mono text-xs text-mast-text outline-none focus:border-mast-accent"
                />
              </Field>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="primary"
                  onClick={() => save.mutate()}
                  disabled={save.isPending}
                >
                  {save.isPending ? "保存中…" : "保存"}
                </Button>
                <Button
                  variant="danger"
                  onClick={() => {
                    // No data is lost, so say so — a scary confirm on a non-destructive
                    // action just teaches people to click through the real ones.
                    if (window.confirm("回退到上一版本？上一版会被重新存为新版本，任何历史版本都不会被删除。"))
                      revert.mutate();
                  }}
                  disabled={revert.isPending}
                >
                  {revert.isPending ? "回退中…" : "回退上一版"}
                </Button>
                <Button variant="default" onClick={exportBody}>
                  导出
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── 差异 ── */}
      {tab === "diff" && (
        <div className="space-y-3">
          {diff.isPending && <Spinner />}
          {diff.isError && <ErrorNote error={diff.error} />}
          {diff.data && diff.data.degraded && (
            <EmptyNote label={diff.data.detail ?? "该产物没有可对比的版本"} />
          )}
          {diff.data && !diff.data.degraded && (
            <>
              <div className="flex flex-wrap items-center gap-2 text-xs text-mast-muted">
                <Badge tone={diff.data.changed ? "WARN" : "INFO"}>
                  {diff.data.changed ? "有改动" : "无改动"}
                </Badge>
                {!diff.data.has_original && <span>（只有一个版本，没有可对比的先前版本）</span>}
              </div>
              {diff.data.changed && diff.data.diff ? (
                <DiffView diff={diff.data.diff} />
              ) : (
                <EmptyNote label="上一版本与当前版本一致，无差异。" />
              )}
            </>
          )}
        </div>
      )}

      {/* ── 历史 ── */}
      {tab === "history" && (
        <div className="space-y-3">
          {history.isPending && <Spinner />}
          {history.isError && <ErrorNote error={history.error} />}
          {history.data && history.data.degraded && (
            <DegradedNote what="编辑历史（需活动会话）" />
          )}
          {history.data && !history.data.degraded && (
            <>
              {history.data.current && (
                <div className="rounded-md border border-mast-accent/40 bg-mast-accent/5 p-3">
                  <div className="mb-1 flex items-center gap-2 text-xs">
                    <Badge tone="WARN">当前编辑</Badge>
                    <span className="font-mono text-mast-muted">
                      {fmtTime(history.data.current.t)}
                    </span>
                  </div>
                  <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-mast-text">
                    {history.data.current.body}
                  </pre>
                </div>
              )}
              {histEntries.length === 0 && !history.data.current && (
                <EmptyNote label="暂无编辑历史。" />
              )}
              {histEntries.length > 0 && (
                <div className="space-y-2">
                  <h5 className="text-xs font-medium text-mast-muted">
                    历史版本（共 {history.data.count}）
                  </h5>
                  {histEntries
                    .slice()
                    .reverse()
                    .map((h, i) => (
                      <div
                        key={i}
                        className="rounded-md border border-mast-border p-3"
                      >
                        <div className="mb-1 flex items-center gap-2 text-xs">
                          <span className="font-mono text-mast-muted">{fmtTime(h.t)}</span>
                          <Button
                            variant="ghost"
                            onClick={() => {
                              setDraft(h.body);
                              setTab("edit");
                              toast("已载入该历史版本到编辑框（点保存以应用）", "ok");
                            }}
                          >
                            载入到编辑框
                          </Button>
                        </div>
                        <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-mast-muted">
                          {h.body}
                        </pre>
                      </div>
                    ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </>
  );

  if (inline) {
    return (
      <div className="rounded-lg border border-mast-border bg-mast-panel px-4 py-4">
        {inner}
      </div>
    );
  }

  return (
    <Modal open={open} onClose={onClose ?? (() => {})} title={`对象编辑 · ${id}`} wide>
      {inner}
    </Modal>
  );
}
