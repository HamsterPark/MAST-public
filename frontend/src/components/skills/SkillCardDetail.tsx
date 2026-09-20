import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import {
  Card,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
  EmptyNote,
} from "../ui";

const SAFETY_TONE: Record<string, string> = {
  AUTO: "AUTO",
  INFO: "INFO",
  WARN: "WARN",
  DANGEROUS: "DANGEROUS",
  auto: "AUTO",
  confirm: "WARN",
  dangerous: "DANGEROUS",
};

/** Full skill card detail, fetched on demand from GET /api/skills/{name}. */
export function SkillCardDetail({ name }: { name: string }) {
  const q = useQuery({
    queryKey: ["skill", name],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skills/{name}", {
        params: { path: { name } },
      });
      if (error) throw error;
      return data;
    },
  });

  if (q.isPending) return <Card><Spinner /></Card>;
  if (q.isError) return <Card><ErrorNote error={q.error} /></Card>;
  const card = q.data;
  if (!card) return <Card><EmptyNote /></Card>;
  if (card.degraded) return <DegradedNote what="技能详情" />;
  if (!card.found) {
    return (
      <Card>
        <EmptyNote label={`未找到技能：${name}`} />
      </Card>
    );
  }

  return (
    <Card className="space-y-4">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-base font-semibold text-mast-text">{card.name}</h3>
          {card.zh && <span className="text-sm text-mast-muted">{card.zh}</span>}
          <Badge tone={SAFETY_TONE[card.safety] ?? "default"}>{card.safety || "—"}</Badge>
          <Badge tone="INFO">L{card.level}</Badge>
          {card.version && <Badge>v{card.version}</Badge>}
        </div>
        <div className="mt-1 flex flex-wrap gap-3 text-xs text-mast-muted">
          <span>领域：{card.domain}</span>
          <span>来源：{card.source_zh || card.source}</span>
          {card.category && <span>分类：{card.category}</span>}
          {card.estimated_duration_s != null && (
            <span>预计耗时：{card.estimated_duration_s}s</span>
          )}
          {card.rollback_skill && <span>回滚：{card.rollback_skill}</span>}
        </div>
      </div>

      {(card.description_zh || card.description) && (
        <p className="text-sm text-mast-text/90">
          {card.description_zh || card.description}
        </p>
      )}

      {!!card.tags?.length && (
        <div className="flex flex-wrap gap-1.5">
          {card.tags.map((t) => (
            <span
              key={t}
              className="rounded bg-mast-bg px-1.5 py-0.5 text-xs text-mast-muted"
            >
              {t}
            </span>
          ))}
        </div>
      )}

      {!!card.parameters?.length && (
        <div>
          <h4 className="mb-2 text-sm font-medium text-mast-text">参数</h4>
          <div className="overflow-auto rounded border border-mast-border">
            <table className="w-full text-xs">
              <thead className="bg-mast-bg text-mast-muted">
                <tr>
                  <th className="px-2 py-1.5 text-left">名称</th>
                  <th className="px-2 py-1.5 text-left">类型</th>
                  <th className="px-2 py-1.5 text-left">单位</th>
                  <th className="px-2 py-1.5 text-left">必填</th>
                  <th className="px-2 py-1.5 text-left">默认</th>
                  <th className="px-2 py-1.5 text-left">范围</th>
                  <th className="px-2 py-1.5 text-left">说明</th>
                </tr>
              </thead>
              <tbody>
                {card.parameters.map((p) => (
                  <tr key={p.name} className="border-t border-mast-border">
                    <td className="px-2 py-1.5 font-medium">{p.name}</td>
                    <td className="px-2 py-1.5 text-mast-muted">{p.type}</td>
                    <td className="px-2 py-1.5 text-mast-muted">{p.unit ?? "—"}</td>
                    <td className="px-2 py-1.5">{p.required ? "是" : "否"}</td>
                    <td className="px-2 py-1.5 text-mast-muted">
                      {p.default == null ? "—" : String(p.default)}
                    </td>
                    <td className="px-2 py-1.5 text-mast-muted">
                      {p.allowed_values?.length
                        ? p.allowed_values.map((v) => String(v)).join(" / ")
                        : p.min != null || p.max != null
                          ? `${p.min ?? "−∞"} ~ ${p.max ?? "+∞"}`
                          : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-mast-muted">{p.description || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        {!!card.preconditions?.length && (
          <div>
            <h4 className="mb-1 text-sm font-medium text-mast-text">前置条件</h4>
            <ul className="list-disc space-y-0.5 pl-5 text-xs text-mast-muted">
              {card.preconditions.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}
        {!!card.postconditions?.length && (
          <div>
            <h4 className="mb-1 text-sm font-medium text-mast-text">后置条件</h4>
            <ul className="list-disc space-y-0.5 pl-5 text-xs text-mast-muted">
              {card.postconditions.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {!!card.outputs?.length && (
        <div>
          <h4 className="mb-1 text-sm font-medium text-mast-text">输出</h4>
          <pre className="overflow-auto rounded border border-mast-border bg-mast-bg p-2 text-xs text-mast-muted">
            {JSON.stringify(card.outputs, null, 2)}
          </pre>
        </div>
      )}
    </Card>
  );
}
