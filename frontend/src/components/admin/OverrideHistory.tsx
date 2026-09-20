import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, useToast } from "@/components/controls";

/** 覆盖历史 — every override category's current payload + timestamped backups,
 *  with restore-by-timestamp. Mirrors the old ConfigOverrideRegistry history.
 *  GET /api/admin/overrides/{category}, /history; POST .../restore/{ts}. */

const CATEGORIES: { id: string; label: string }[] = [
  { id: "safety_limits", label: "安全限值" },
  { id: "checks", label: "全局检查规则" },
  { id: "constraints", label: "材料约束" },
  { id: "skill", label: "技能覆盖" },
  { id: "knowledge", label: "知识库" },
  { id: "guidance", label: "技能指导" },
  { id: "encyclopedia", label: "百科配置" },
  { id: "agent", label: "智能体" },
];

export function OverrideHistory() {
  const queryClient = useQueryClient();
  const { toast, node } = useToast();
  const [category, setCategory] = useState(CATEGORIES[0]!.id);

  const ovrQ = useQuery({
    queryKey: ["override", category],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/overrides/{category}", {
        params: { path: { category } },
      });
      if (error) throw error;
      return data;
    },
  });

  const histQ = useQuery({
    queryKey: ["override", category, "history"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/overrides/{category}/history", {
        params: { path: { category } },
      });
      if (error) throw error;
      return data;
    },
  });

  const restore = useMutation({
    mutationFn: async (ts: string) => {
      const { data, error } = await api.POST("/api/admin/overrides/{category}/restore/{ts}", {
        params: { path: { category, ts } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["override", category] });
      queryClient.invalidateQueries({ queryKey: ["override", category, "history"] });
      queryClient.invalidateQueries({ queryKey: ["safety", "limits"] });
      res?.degraded ? toast("恢复未生效（内核未接入）。", "err")
        : toast(res?.reloaded ? "已恢复并热重载。" : "已恢复。", "ok");
    },
    onError: (e) => toast(`恢复失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2">
        {CATEGORIES.map((c) => (
          <button
            key={c.id}
            onClick={() => setCategory(c.id)}
            className={
              "rounded-md border px-3 py-1.5 text-sm " +
              (category === c.id
                ? "border-mast-accent bg-mast-accent/15 text-mast-accent"
                : "border-mast-border text-mast-muted hover:text-mast-text")
            }
          >
            {c.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <div className="mb-2 flex items-center gap-2">
            <span className="text-sm font-medium text-mast-text">当前覆盖内容</span>
            {ovrQ.data && !ovrQ.data.degraded &&
              (ovrQ.data.has_override ? <Badge tone="INFO">已覆盖</Badge> : <Badge tone="AUTO">使用默认</Badge>)}
          </div>
          {ovrQ.isPending && <Spinner />}
          {ovrQ.error && <ErrorNote error={ovrQ.error} />}
          {ovrQ.data?.degraded && <DegradedNote what="覆盖读取" />}
          {ovrQ.data && !ovrQ.data.degraded && (
            <pre className="max-h-96 overflow-auto rounded-md bg-mast-bg p-3 text-xs text-mast-text">
              {JSON.stringify(ovrQ.data.data ?? {}, null, 2)}
            </pre>
          )}
        </Card>

        <Card>
          <span className="mb-2 block text-sm font-medium text-mast-text">历史备份</span>
          {histQ.isPending && <Spinner />}
          {histQ.error && <ErrorNote error={histQ.error} />}
          {histQ.data?.degraded && <DegradedNote what="历史读取" />}
          {histQ.data && !histQ.data.degraded && (
            (histQ.data.entries ?? []).length === 0 ? (
              <EmptyNote label="暂无历史备份" />
            ) : (
              <ul className="space-y-1">
                {histQ.data.entries!.map((e) => (
                  <li key={e.timestamp} className="flex items-center justify-between gap-2 rounded-md border border-mast-border px-2 py-1.5 text-sm">
                    <div className="min-w-0">
                      <div className="font-mono text-xs text-mast-text">{e.timestamp}</div>
                      <div className="truncate text-xs text-mast-muted">{e.filename}</div>
                    </div>
                    <Button variant="primary" disabled={restore.isPending} onClick={() => restore.mutate(e.timestamp)}>
                      恢复
                    </Button>
                  </li>
                ))}
              </ul>
            )
          )}
        </Card>
      </div>
      {node}
    </div>
  );
}
