import { useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { type ColumnDef } from "@tanstack/react-table";
import { api } from "@/api/client";
import { DataTable } from "@/components/DataTable";
import {
  Section,
  Card,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
  EmptyNote,
} from "@/components/ui";
import { Field, TextField, Button, Modal, useToast } from "@/components/controls";
import { CompositeDag } from "./CompositeDag";
import type { SpecNode } from "./compositeGraph";

// ── 复合技能 — list + react-flow DAG + version history + clone/restore/delete ──
// Reproduces the old composite_panel.py surface:
//   composite list (dropdown→here a table) · metadata + 流程图(DAG) + 参数 ·
//   版本历史 + 一键回滚(restore) · 克隆为新技能(clone) · 删除(delete).
// NOTE: there is NO delete endpoint in the API (composite_panel.delete_composite
// has no /api/composites/{name} DELETE). Delete is flagged as unavailable.

const SAFETY_TONE: Record<string, string> = {
  auto: "AUTO",
  confirm: "WARN",
  dangerous: "DANGEROUS",
  AUTO: "AUTO",
  WARN: "WARN",
  DANGEROUS: "DANGEROUS",
};

type CompRow = {
  name: string;
  version: number;
  description: string;
  safety_level: string;
  n_nodes: number;
  tags?: unknown[];
};

function useComposites() {
  return useQuery({
    queryKey: ["composites"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/composites");
      if (error) throw error;
      return data;
    },
  });
}

function useComposite(name: string | null) {
  return useQuery({
    enabled: !!name,
    queryKey: ["composite", name],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/composites/{name}", {
        params: { path: { name: name! } },
      });
      if (error) throw error;
      return data;
    },
  });
}

export function CompositesManager() {
  const qc = useQueryClient();
  const { toast, node: toastNode } = useToast();
  const q = useComposites();

  const [text, setText] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  // clone modal
  const [cloneOpen, setCloneOpen] = useState(false);
  const [cloneName, setCloneName] = useState("");
  const [cloneAuthor, setCloneAuthor] = useState("");

  const rows = (q.data?.composites ?? []) as CompRow[];

  const cloneMut = useMutation({
    mutationFn: async () => {
      if (!selected) throw new Error("未选择复合技能");
      if (!cloneName.trim()) throw new Error("请填写新技能名称");
      const { data, error } = await api.POST("/api/composites/{name}/clone", {
        params: { path: { name: selected } },
        body: { new_name: cloneName.trim(), author: cloneAuthor.trim() },
      });
      if (error) throw error;
      if (!data?.ok) throw new Error(data?.error || "克隆失败");
      return data;
    },
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: ["composites"] });
      setCloneOpen(false);
      setCloneName("");
      setCloneAuthor("");
      toast(d?.message || "已克隆", "ok");
      if (d?.name) setSelected(d.name);
    },
    onError: (e) => toast(e instanceof Error ? e.message : "克隆失败", "err"),
  });

  const restoreMut = useMutation({
    mutationFn: async (version: number) => {
      if (!selected) throw new Error("未选择复合技能");
      const { data, error } = await api.POST(
        "/api/composites/{name}/restore/{version}",
        { params: { path: { name: selected, version } } },
      );
      if (error) throw error;
      if (!data?.ok) throw new Error(data?.error || "回滚失败");
      return data;
    },
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: ["composites"] });
      qc.invalidateQueries({ queryKey: ["composite", selected] });
      toast(d?.message || "已回滚", "ok");
    },
    onError: (e) => toast(e instanceof Error ? e.message : "回滚失败", "err"),
  });

  const columns = useMemo<ColumnDef<CompRow, any>[]>(
    () => [
      {
        accessorKey: "name",
        header: "名称",
        cell: (c) => <span className="font-medium text-mast-text">{c.getValue() as string}</span>,
      },
      { accessorKey: "version", header: "版本", cell: (c) => `v${c.getValue() as number}` },
      {
        accessorKey: "safety_level",
        header: "安全等级",
        cell: (c) => (
          <Badge tone={SAFETY_TONE[c.getValue() as string] ?? "default"}>{c.getValue() as string}</Badge>
        ),
      },
      { accessorKey: "n_nodes", header: "节点数" },
      { accessorKey: "description", header: "说明" },
      {
        id: "open",
        header: "",
        cell: (c) => (
          <button
            onClick={() => setSelected(c.row.original.name)}
            className="rounded border border-mast-border px-2 py-0.5 text-xs text-mast-accent hover:bg-mast-accent/10"
          >
            查看
          </button>
        ),
      },
    ],
    [],
  );

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="复合技能列表" />;

  return (
    <div className="space-y-4">
      {toastNode}
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="搜索复合技能…"
          className="w-64 rounded border border-mast-border bg-mast-bg px-3 py-1.5 text-sm text-mast-text placeholder:text-mast-muted"
        />
        <span className="text-xs text-mast-muted">共 {rows.length} 个</span>
      </div>

      <DataTable data={rows} columns={columns} globalFilter={text} empty="暂无复合技能" />

      {selected && (
        <CompositeDetailPanel
          name={selected}
          onClose={() => setSelected(null)}
          onClone={() => {
            setCloneName("");
            setCloneAuthor("");
            setCloneOpen(true);
          }}
          onRestore={(v) => restoreMut.mutate(v)}
          restoring={restoreMut.isPending}
        />
      )}

      <Modal open={cloneOpen} onClose={() => setCloneOpen(false)} title={`克隆为新复合技能（源：${selected}）`}>
        <div className="space-y-3">
          <Field label="新技能名称" hint="将作为新蓝本（v1），可在列表中选择并编辑">
            <TextField value={cloneName} onChange={setCloneName} placeholder="MyNewWorkflow" />
          </Field>
          <Field label="作者（可选）">
            <TextField value={cloneAuthor} onChange={setCloneAuthor} placeholder="" />
          </Field>
          <div className="flex justify-end gap-2 pt-1">
            <Button variant="ghost" onClick={() => setCloneOpen(false)}>取消</Button>
            <Button variant="primary" disabled={cloneMut.isPending} onClick={() => cloneMut.mutate()}>
              {cloneMut.isPending ? "克隆中…" : "克隆"}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

function CompositeDetailPanel({
  name,
  onClose,
  onClone,
  onRestore,
  restoring,
}: {
  name: string;
  onClose: () => void;
  onClone: () => void;
  onRestore: (version: number) => void;
  restoring: boolean;
}) {
  const qc = useQueryClient();
  const { toast, node: toastNode } = useToast();
  const q = useComposite(name);

  const validateMut = useMutation({
    mutationFn: async () => {
      const spec = (q.data?.spec ?? {}) as Record<string, unknown>;
      const { data, error } = await api.POST("/api/composites/{name}/validate", {
        params: { path: { name } },
        body: { spec },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d?.degraded) toast("校验后端不可用（degraded）", "err");
      else if (d?.ok) toast("校验通过，无问题", "ok");
      else toast(`校验发现 ${d?.problems?.length ?? 0} 个问题`, "err");
      qc.setQueryData(["composite", name, "validate"], d);
    },
    onError: (e) => toast(e instanceof Error ? e.message : "校验失败", "err"),
  });

  return (
    <Section
      title={`复合技能详情：${name}`}
      actions={
        <button onClick={onClose} className="text-xs text-mast-muted hover:text-mast-text">关闭</button>
      }
    >
      {toastNode}
      {q.isPending ? (
        <Spinner />
      ) : q.isError ? (
        <ErrorNote error={q.error} />
      ) : q.data?.degraded ? (
        <DegradedNote what="复合技能详情" />
      ) : !q.data?.found ? (
        <EmptyNote label={`未找到复合技能：${name}`} />
      ) : (
        (() => {
          const d = q.data!;
          const spec = (d.spec ?? null) as SpecNode | null;
          const version = spec ? Number(spec.version ?? 0) : 0;
          const safety = spec ? String(spec.safety_level ?? "") : "";
          const description = spec ? String(spec.description ?? "") : "";
          const tags = spec && Array.isArray(spec.tags) ? (spec.tags as unknown[]) : [];
          const params = spec && Array.isArray(spec.params) ? (spec.params as Record<string, unknown>[]) : [];
          const valReport = qc.getQueryData(["composite", name, "validate"]) as
            | { ok?: boolean; problems?: string[]; steps?: { id: string; skill: string; errors: string[]; warnings: string[] }[]; degraded?: boolean }
            | undefined;

          return (
            <div className="space-y-4">
              <Card className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-base font-semibold text-mast-text">{d.name}</h3>
                  {version > 0 && <Badge>v{version}</Badge>}
                  {safety && <Badge tone={SAFETY_TONE[safety] ?? "default"}>{safety}</Badge>}
                </div>
                {description && <p className="text-sm text-mast-text/90">{description}</p>}
                {!!tags.length && (
                  <div className="flex flex-wrap gap-1.5">
                    {tags.map((t, i) => (
                      <span key={i} className="rounded bg-mast-bg px-1.5 py-0.5 text-xs text-mast-muted">{String(t)}</span>
                    ))}
                  </div>
                )}
                {!!params.length && (
                  <div>
                    <h4 className="mb-1 text-sm font-medium text-mast-text">输入参数</h4>
                    <div className="flex flex-wrap gap-2 text-xs text-mast-muted">
                      {params.map((p, i) => (
                        <span key={i} className="rounded border border-mast-border px-2 py-1">
                          {String(p.name ?? "")}
                          <span className="opacity-60"> : {String(p.type ?? "")}</span>
                          {p.default != null && <span className="opacity-60"> = {String(p.default)}</span>}
                          {p.required ? <span className="ml-1 text-mast-danger">*</span> : null}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="flex flex-wrap gap-2 pt-1">
                  <Button variant="primary" onClick={onClone}>克隆为新技能</Button>
                  <Button variant="default" disabled={validateMut.isPending} onClick={() => validateMut.mutate()}>
                    {validateMut.isPending ? "校验中…" : "校验流程"}
                  </Button>
                </div>
              </Card>

              {valReport && (
                <Card className="space-y-2">
                  <h4 className="text-sm font-medium text-mast-text">
                    校验结果 {valReport.ok ? <Badge tone="AUTO">通过</Badge> : <Badge tone="WARN">有问题</Badge>}
                  </h4>
                  {!!valReport.problems?.length && (
                    <ul className="list-disc space-y-0.5 pl-5 text-xs text-mast-danger">
                      {valReport.problems.map((p, i) => <li key={i}>{p}</li>)}
                    </ul>
                  )}
                  {!!valReport.steps?.length && (
                    <div className="overflow-auto rounded border border-mast-border">
                      <table className="w-full text-xs">
                        <thead className="bg-mast-bg text-mast-muted">
                          <tr>
                            <th className="px-2 py-1.5 text-left">节点</th>
                            <th className="px-2 py-1.5 text-left">技能</th>
                            <th className="px-2 py-1.5 text-left">错误</th>
                            <th className="px-2 py-1.5 text-left">警告</th>
                          </tr>
                        </thead>
                        <tbody>
                          {valReport.steps.map((s) => (
                            <tr key={s.id} className="border-t border-mast-border">
                              <td className="px-2 py-1.5 font-mono">{s.id}</td>
                              <td className="px-2 py-1.5 font-mono">{s.skill}</td>
                              <td className="px-2 py-1.5 text-mast-danger">{s.errors.join("; ") || "—"}</td>
                              <td className="px-2 py-1.5 text-mast-warn">{s.warnings.join("; ") || "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </Card>
              )}

              <Card>
                <h4 className="mb-2 text-sm font-medium text-mast-text">控制流 DAG</h4>
                <CompositeDag spec={spec} />
              </Card>

              <Card>
                <h4 className="mb-2 text-sm font-medium text-mast-text">版本历史</h4>
                {!d.versions?.length ? (
                  <EmptyNote label="暂无版本历史" />
                ) : (
                  <div className="overflow-auto rounded border border-mast-border">
                    <table className="w-full text-xs">
                      <thead className="bg-mast-bg text-mast-muted">
                        <tr>
                          <th className="px-2 py-1.5 text-left">版本</th>
                          <th className="px-2 py-1.5 text-left">保存时间</th>
                          <th className="px-2 py-1.5 text-left">节点数</th>
                          <th className="px-2 py-1.5 text-left">说明</th>
                          <th className="px-2 py-1.5 text-left">操作</th>
                        </tr>
                      </thead>
                      <tbody>
                        {d.versions.map((v) => (
                          <tr key={v.version} className="border-t border-mast-border">
                            <td className="px-2 py-1.5 font-medium">v{v.version}</td>
                            <td className="px-2 py-1.5 text-mast-muted">{v.saved_at || "—"}</td>
                            <td className="px-2 py-1.5 text-mast-muted tabular-nums">{v.n_nodes}</td>
                            <td className="px-2 py-1.5 text-mast-muted">{v.description || "—"}</td>
                            <td className="px-2 py-1.5">
                              <button
                                disabled={restoring}
                                onClick={() => onRestore(v.version)}
                                className="rounded border border-mast-border px-2 py-0.5 text-xs text-mast-accent hover:bg-mast-accent/10 disabled:opacity-50"
                              >
                                回滚到此版本
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            </div>
          );
        })()
      )}
    </Section>
  );
}
