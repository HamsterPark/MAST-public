import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Section, Spinner } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { StructuredView } from "@/components/admin/StructuredView";

/** 参考/规则 — read-only safety + knowledge reference tables.
 *
 *  These are advisory code constants the old Gradio admin tabs showed read-only
 *  (never editable / never merged with overrides):
 *    · GET /api/safety/checks/effective — _GLOBAL_CHECKS merged with the override
 *      layer, each row tagged default / modified / addition (the effective rule
 *      set the SafetyGate actually enforces).
 *    · GET /api/knowledge/reference/{kind} — one verbatim knowledge constant
 *      (safety_constraints / scan_speed_rule / constant_height_prerequisites).
 *
 *  All reads render loading / error / degraded / empty; nothing here writes. */

type EffectiveCheckRow = components["schemas"]["EffectiveCheckRow"];

const ORIGIN_TONE: Record<string, string> = {
  default: "AUTO",
  modified: "WARN",
  addition: "INFO",
};
const ORIGIN_LABEL: Record<string, string> = {
  default: "默认",
  modified: "已修改",
  addition: "新增",
};

// ── 全局检查规则（生效值，只读）──────────────────────────────────────────────
function EffectiveChecksTable() {
  const q = useQuery({
    queryKey: ["safety", "checks", "effective"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/safety/checks/effective");
      if (error) throw error;
      return data;
    },
  });

  const columns: ColumnDef<EffectiveCheckRow, any>[] = [
    { accessorKey: "pattern", header: "字段匹配 (pattern)", cell: (c) => <span className="font-mono text-xs">{c.row.original.pattern}</span> },
    { accessorKey: "unit", header: "单位" },
    { accessorKey: "min_attr", header: "下限属性", cell: (c) => <span className="font-mono text-xs">{c.row.original.min_attr}</span> },
    { accessorKey: "max_attr", header: "上限属性", cell: (c) => <span className="font-mono text-xs">{c.row.original.max_attr}</span> },
    {
      id: "origin",
      header: "来源",
      accessorFn: (r) => r.origin,
      cell: (c) => (
        <Badge tone={ORIGIN_TONE[c.row.original.origin] ?? "INFO"}>
          {ORIGIN_LABEL[c.row.original.origin] ?? c.row.original.origin}
        </Badge>
      ),
    },
  ];

  return (
    <Section title="全局检查规则（生效值，只读）">
      <Card>
        <p className="mb-3 text-xs text-mast-muted">
          代码默认 _GLOBAL_CHECKS 与覆盖层合并后的最终规则集（SafetyGate 实际校验依据）。
          每行标注来源：默认 / 已修改 / 新增。编辑请到「安全 → 全局检查规则」。
        </p>
        {q.isPending && <Spinner />}
        {q.error && <ErrorNote error={q.error} />}
        {q.data?.degraded && <DegradedNote what="全局检查规则" />}
        {q.data && !q.data.degraded && (
          <>
            <div className="mb-2 flex items-center gap-2 text-xs text-mast-muted">
              <span>共 {q.data.count ?? (q.data.rows?.length ?? 0)} 条</span>
              {q.data.has_override ? <Badge tone="INFO">含覆盖</Badge> : <Badge tone="AUTO">纯默认</Badge>}
            </div>
            <DataTable data={q.data.rows ?? []} columns={columns} empty="无检查规则" />
          </>
        )}
      </Card>
    </Section>
  );
}

// ── 知识库参考常量（只读）────────────────────────────────────────────────────
const REF_KINDS: { id: string; label: string; desc: string }[] = [
  { id: "safety_constraints", label: "材料安全约束", desc: "材料/样品安全约束参考（safety_constraints）。" },
  { id: "scan_speed_rule", label: "扫描速度规则", desc: "扫描速度经验规则（scan_speed_rule）。" },
  { id: "constant_height_prerequisites", label: "恒高模式前置条件", desc: "恒高扫描前置条件（constant_height_prerequisites）。" },
];

function KnowledgeReferenceTable() {
  const [kind, setKind] = useState(REF_KINDS[0]!.id);
  const cur = REF_KINDS.find((k) => k.id === kind)!;

  const q = useQuery({
    queryKey: ["knowledge", "reference", kind],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/knowledge/reference/{kind}", {
        params: { path: { kind } },
      });
      if (error) throw error;
      return data;
    },
  });

  const refData = q.data?.data ?? null;

  return (
    <Section title="知识库参考常量（只读）">
      <div className="mb-3 flex flex-wrap gap-2">
        {REF_KINDS.map((k) => (
          <button
            key={k.id}
            onClick={() => setKind(k.id)}
            className={
              "rounded-md border px-3 py-1.5 text-sm " +
              (kind === k.id
                ? "border-mast-accent bg-mast-accent/15 text-mast-accent"
                : "border-mast-border text-mast-muted hover:text-mast-text")
            }
          >
            {k.label}
          </button>
        ))}
      </div>
      <Card>
        <p className="mb-3 text-xs text-mast-muted">{cur.desc} 这些是顾问性代码常量，从不与覆盖合并，仅供参考。</p>
        {q.isPending && <Spinner />}
        {q.error && <ErrorNote error={q.error} />}
        {q.data?.degraded && <DegradedNote what="知识库参考常量" />}
        {q.data && !q.data.degraded && (
          refData == null ? (
            <EmptyNote label="无参考数据" />
          ) : (
            <>
              <div className="mb-2 text-xs text-mast-muted">共 {q.data.count ?? 0} 项</div>
              <StructuredView value={refData} />
            </>
          )
        )}
      </Card>
    </Section>
  );
}

export function ReferenceTables() {
  return (
    <div className="space-y-6">
      <EffectiveChecksTable />
      <KnowledgeReferenceTable />
    </div>
  );
}
