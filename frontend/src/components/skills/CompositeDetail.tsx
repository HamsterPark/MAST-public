import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Card, Badge, Spinner, ErrorNote, DegradedNote, EmptyNote } from "../ui";
import { CompositeDag } from "./CompositeDag";
import type { SpecNode } from "./compositeGraph";

/** One composite: spec metadata + read-only DAG + version history.
 *  Consumes GET /api/composites/{name}. */
export function CompositeDetail({ name }: { name: string }) {
  const q = useQuery({
    queryKey: ["composite", name],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/composites/{name}", {
        params: { path: { name } },
      });
      if (error) throw error;
      return data;
    },
  });

  if (q.isPending) return <Card><Spinner /></Card>;
  if (q.isError) return <Card><ErrorNote error={q.error} /></Card>;
  const d = q.data;
  if (!d) return <Card><EmptyNote /></Card>;
  if (d.degraded) return <DegradedNote what="工作流详情" />;
  if (!d.found) {
    return (
      <Card>
        <EmptyNote label={`未找到工作流：${name}`} />
      </Card>
    );
  }

  const spec = (d.spec ?? null) as SpecNode | null;
  const version = spec ? Number(spec.version ?? 0) : 0;
  const safety = spec ? String(spec.safety_level ?? "") : "";
  const description = spec ? String(spec.description ?? "") : "";
  const tags = spec && Array.isArray(spec.tags) ? (spec.tags as unknown[]) : [];
  const params = spec && Array.isArray(spec.params) ? (spec.params as Record<string, unknown>[]) : [];

  const SAFETY_TONE: Record<string, string> = {
    auto: "AUTO",
    confirm: "WARN",
    dangerous: "DANGEROUS",
  };

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
              <span key={i} className="rounded bg-mast-bg px-1.5 py-0.5 text-xs text-mast-muted">
                {String(t)}
              </span>
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
                </span>
              ))}
            </div>
          </div>
        )}
      </Card>

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
                </tr>
              </thead>
              <tbody>
                {d.versions.map((v) => (
                  <tr key={v.version} className="border-t border-mast-border">
                    <td className="px-2 py-1.5 font-medium">v{v.version}</td>
                    <td className="px-2 py-1.5 text-mast-muted">{v.saved_at || "—"}</td>
                    <td className="px-2 py-1.5 text-mast-muted tabular-nums">{v.n_nodes}</td>
                    <td className="px-2 py-1.5 text-mast-muted">{v.description || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
