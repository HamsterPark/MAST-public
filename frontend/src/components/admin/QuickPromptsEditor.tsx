import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, TextField, useToast } from "@/components/controls";

/** 快速提示编辑 — chat quick-prompt list editor (label + prompt rows), mirroring
 *  the old encyclopedia_tab quick-prompts sub-tab. GET/POST
 *  /api/admin/quick-prompts. Empty list ⇒ reset to code defaults. NO Dataframe. */

type Prompt = { label: string; prompt: string };

export function QuickPromptsEditor() {
  const queryClient = useQueryClient();
  const { toast, node } = useToast();

  const q = useQuery({
    queryKey: ["quick-prompts"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/quick-prompts");
      if (error) throw error;
      return data;
    },
  });

  const stored: Prompt[] = ((q.data?.prompts ?? []) as any[]).map((p) => ({
    label: String(p.label ?? p.name ?? ""),
    prompt: String(p.prompt ?? p.text ?? ""),
  }));

  const [rows, setRows] = useState<Prompt[] | null>(null);
  useEffect(() => { setRows(null); }, [q.data]);
  const view = rows ?? stored;

  const save = useMutation({
    mutationFn: async (prompts: Prompt[]) => {
      const { data, error } = await api.POST("/api/admin/quick-prompts", {
        body: { prompts: prompts as never },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      setRows(null);
      queryClient.invalidateQueries({ queryKey: ["quick-prompts"] });
      res?.degraded ? toast("写入未生效（内核未接入）。", "err") : toast("已保存快速提示。", "ok");
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const persist = (next: Prompt[]) => {
    const clean = next.filter((r) => r.label.trim() || r.prompt.trim());
    save.mutate(clean);
  };

  return (
    <Card>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-medium text-mast-text">聊天快速提示</span>
        {q.data?.has_override ? <Badge tone="INFO">已覆盖</Badge> : <Badge tone="AUTO">默认</Badge>}
      </div>
      <p className="mb-3 text-xs text-mast-muted">编辑主聊天界面的快捷提示按钮（按钮名称 + 发送内容）。留空保存=恢复代码默认。</p>
      {q.isPending && <Spinner />}
      {q.error && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="快速提示" />}
      {q.data && !q.data.degraded && (
        <div className="space-y-2">
          <div className="grid grid-cols-[1fr_2fr_auto] gap-2 text-xs text-mast-muted">
            <span>按钮名称</span><span>发送内容</span><span />
          </div>
          {view.map((r, i) => (
            <div key={i} className="grid grid-cols-[1fr_2fr_auto] items-center gap-2">
              <TextField value={r.label} onChange={(v) => { const n = [...view]; n[i] = { ...r, label: v }; setRows(n); }} />
              <TextField value={r.prompt} onChange={(v) => { const n = [...view]; n[i] = { ...r, prompt: v }; setRows(n); }} />
              <Button variant="danger" onClick={() => setRows(view.filter((_, j) => j !== i))}>删除</Button>
            </div>
          ))}
          <Button variant="default" onClick={() => setRows([...view, { label: "", prompt: "" }])}>+ 新增提示</Button>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button variant="primary" disabled={save.isPending} onClick={() => persist(view)}>
              {save.isPending ? "保存中…" : "保存"}
            </Button>
            {rows && <Button variant="ghost" onClick={() => setRows(null)}>撤销编辑</Button>}
            <Button variant="danger" disabled={save.isPending} onClick={() => persist([])}>恢复默认</Button>
          </div>
        </div>
      )}
      {node}
    </Card>
  );
}
