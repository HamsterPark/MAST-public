import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge } from "@/components/ui";
import { useToast } from "@/components/controls";
import { useWsEvent } from "@/hooks/useWsEvents";
import { marketWriteProblem, type MarketWriteResult } from "@/lib/skillMarket";

// ── agent 的订阅推荐，就地确认 ────────────────────────────────────────────────
//
// 为什么是**页面上的一条**而不是聊天气泡里的卡片：气泡的 HTML 是后端预渲染、前端
// 用 dangerouslySetInnerHTML 塞进去的（`chat/render.py`），里面挂不上 React 事件
// 处理器。硬塞的话要么把渲染搬到前端（那是另一件大事），要么在气泡里放一个假按钮。
// 所以这条横幅贴在输入框上方 —— 对用户来说效果是一样的：**不用离开对话去点头**，
// 而那正是「主动推荐 + 用户确认」里最容易流失的一步。
//
// 数据源是 `GET /api/skill-market/recommendations`，不是 WS 帧。帧只做触发（那条
// 总线只重放 100 条，按帧累积列表的客户端迟早显示一份错的）。

type Rec = {
  id: string;
  skill: string;
  by_agent: string;
  reason: string;
  status: string;
};

export function PendingRecommendations({ compact = false }: { compact?: boolean }) {
  const qc = useQueryClient();
  const { toast, node: toastNode } = useToast();

  const q = useQuery({
    queryKey: ["skill-market", "recommendations"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-market/recommendations");
      if (error) throw error;
      return data as unknown as { pending: Rec[]; degraded: boolean };
    },
  });

  useWsEvent("skill_recommendation", () => {
    qc.invalidateQueries({ queryKey: ["skill-market", "recommendations"] });
  });

  const resolve = useMutation({
    mutationFn: async (v: { id: string; accept: boolean }) => {
      const { data, error } = await api.POST("/api/skill-market/recommendations/{rec_id}/resolve", { params: { path: { rec_id: v.id } }, body: { accept: v.accept } });
      if (error) throw error;
      return { res: data as unknown as MarketWriteResult, accept: v.accept };
    },
    onSuccess: ({ res, accept }) => {
      qc.invalidateQueries({ queryKey: ["skill-market"] });
      qc.invalidateQueries({ queryKey: ["agents", "tools"] });
      qc.invalidateQueries({ queryKey: ["builder", "catalog"] });
      const problem = marketWriteProblem(res);
      if (problem) {
        // 三态压进两档 toast 时一律走 err：「已排队」被读成「已生效」的代价，
        // 比一个看起来重了点的提示大得多。权威显示在 技能 → 市场 的状态条。
        toast(problem.message, "err");
        return;
      }
      toast(accept ? (res.rebuild_note || "已加入你的技能面") : "已拒绝，记录保留", "ok");
    },
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  const pending = q.data?.pending ?? [];
  if (q.isPending || q.data?.degraded || pending.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5">
      {toastNode}
      {pending.map((r) => (
        <div
          key={r.id}
          className="flex flex-wrap items-center gap-2 rounded border border-mast-warn-border bg-mast-warn-bg px-2.5 py-1.5 text-xs"
        >
          <Badge tone="WARN">技能推荐</Badge>
          <span className="text-mast-text">
            <b>{r.by_agent || "agent"}</b> 想用{" "}
            <code className="font-mono">{r.skill}</code>
            {r.reason ? <span className="text-mast-muted">：{r.reason}</span> : null}
          </span>
          <span className="flex-1" />
          <button
            type="button"
            disabled={resolve.isPending}
            onClick={() => resolve.mutate({ id: r.id, accept: true })}
            className="rounded border border-mast-accent bg-mast-accent px-2 py-0.5 text-mast-accent-ink disabled:opacity-50"
          >
            加入我的技能面
          </button>
          <button
            type="button"
            disabled={resolve.isPending}
            onClick={() => resolve.mutate({ id: r.id, accept: false })}
            className="rounded border border-mast-border px-2 py-0.5 text-mast-muted disabled:opacity-50"
          >
            不用
          </button>
          {!compact && (
            <span className="text-[10px] text-mast-faint">
              在「技能 → 市场」可以看到全部
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
