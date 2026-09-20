import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, useToast } from "@/components/controls";
import { DegradedNote } from "@/components/ui";
import { agentLabel } from "@/components/agents/registry";
import { interruptPanelView } from "@/lib/interruptPanel";
import {
  InterruptCard,
  fromPollRow,
  type ResolveArgs,
} from "@/components/agents/InterruptCard";

// 私聊的人工介入面 — pending interrupts polled from
// GET /api/agents/{id}/interrupts, resolved via
// POST /api/agents/{id}/interrupts/{event_id}/resolve.
//
// 覆盖两个方向：用户**批准**一个智能体想跑的 DANGEROUS 技能，以及用户
// **回答**智能体提的问题（ask_user）。卡片本体与群聊那一面共用，所以两处长得
// 一样、行为也一样。
//
// ── 为什么这里是内联而不是弹窗────────────────────────────
// 「人工介入(HITL)的时候看不到后面的界面显示了，这不对」。
//
// 从前这是一个 `<Modal>`，而且**一有中断就自动弹**。`Modal` 是
// `fixed inset-0 z-50 bg-black/60 backdrop-blur-sm` —— 整个界面被压黑加模糊。
// 于是恰恰在最需要看仪器状态的那一刻（有人在问「这一步要不要批准」），偏压、
// 电流、Z、扫描进度、告警全被自己的审批框挡住了。要判断该不该批准，靠的正是
// 那些被挡住的读数。
//
// 自动弹窗当初的理由是真的，不能一删了事：**发起这个中断的那一轮是阻塞的**，
// 把它藏在一个角标后面，就是「对话安静了十五分钟然后报一个没人看见的超时」。
// 所以这里保留「不可能错过」，只去掉「盖住屏幕」——两者本来就不必绑在一起：
// 内联卡片在文档流里，它把内容**推开**而不是**盖住**。
//
// 群聊那一面（RunTaskPanel）本来就是内联的，仪器 chat 是全仓唯一的例外。
// 这次是让它跟上已有的做法，不是发明新形态。
//
// 收起之后仍然留一行「N 项等待处理」——**可关闭 ≠ 可消失**。真把它藏干净，
// 上面那个十五分钟的失败模式就原样回来了。

export function useInterrupts(agentId: string) {
  return useQuery({
    queryKey: ["chat", "interrupts", agentId],
    queryFn: async () => {
      const { api } = await import("@/api/client");
      const { data, error } = await api.GET("/api/agents/{agent_id}/interrupts", {
        params: { path: { agent_id: agentId } },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: 2500,
  });
}

/**
 * 提交一个决定。
 *
 * 抽成 hook 是因为这段语义有三种结局而不是两种：成功、**后端刻意不接受**
 * （answer_invalid / route_not_allowed —— worker 仍然阻塞着，等用户改完重交）、
 * 以及降级未生效。第二种要求卡片保持可交互。复制一份迟早会漏掉中间那种。
 */
export function useResolveInterrupt(onDone?: () => void) {
  const qc = useQueryClient();
  const { toast, node } = useToast();

  const resolve = useMutation({
    mutationFn: async (vars: { agent_id: string; interrupt_id: string; args: ResolveArgs }) => {
      const { api } = await import("@/api/client");
      const { data: res, error } = await api.POST(
        "/api/agents/{agent_id}/interrupts/{interrupt_id}/resolve",
        {
          params: { path: { agent_id: vars.agent_id, interrupt_id: vars.interrupt_id } },
          body: {
            decision: vars.args.decision,
            edited_args: vars.args.editedArgs ?? null,
            comment: vars.args.comment ?? null,
            selected: vars.args.selected ?? null,
            custom_text: vars.args.customText ?? null,
          },
        },
      );
      if (error) throw error;
      return res;
    },
    onSuccess: (res, vars) => {
      const d = vars.args.decision;
      const label =
        d === "approve" ? "批准"
          : d === "reject" ? "拒绝"
          : d === "edit" ? "编辑并批准"
          : d === "answer" ? "提交回答" : `选择「${d}」`;
      if (res?.ok && res?.applied) {
        toast(`已${label}`, "ok");
      } else if (res?.status === "answer_invalid" || res?.status === "route_not_allowed") {
        toast(`未提交：${res?.detail ?? "回答无效"}，请修改后重试`, "err");
      } else {
        toast(`未生效（${res?.status ?? "degraded"}）：${res?.detail ?? "后端降级或已处理"}`, "err");
      }
      qc.invalidateQueries({ queryKey: ["chat", "interrupts", vars.agent_id] });
      onDone?.();
    },
    // 一个失败的提交必须**说**它失败了。静默是这个界面唯一不许有的结局
    // （「这里的批准和拒绝都点不了」就是这么来的）。
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  return { resolve, toastNode: node };
}

export interface PendingInterruptsHandle {
  /** 滚到面板并展开它。角标按钮用 —— 转录很长时它是唯一的锚点。 */
  reveal: () => void;
}

export const PendingInterrupts = forwardRef<
  PendingInterruptsHandle,
  {
    data: ReturnType<typeof useInterrupts>["data"];
    agentId: string;
  }
>(function PendingInterrupts({ data, agentId }, ref) {
  const [collapsed, setCollapsed] = useState(false);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const { resolve, toastNode } = useResolveInterrupt();

  useImperativeHandle(ref, () => ({
    reveal: () => {
      setCollapsed(false);
      // 最坏情况是「没反应」，绝不能抛。ref 还没挂上、或浏览器不支持平滑滚动，
      // 都不该让点击变成一个红色的控制台异常。
      try {
        boxRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      } catch {
        /* 滚不动就算了，面板已经展开了，这才是要紧的那一半 */
      }
    },
  }), []);

  const interrupts = data?.interrupts ?? [];
  // 显示判定在 lib/interruptPanel.ts —— 「收起只收卡片、那一行永远留着」是一条
  // 被要求过的性质，写在 JSX 里没人守得住。
  const view = interruptPanelView(data, collapsed);
  if (!view.render) return null;

  return (
    <div
      ref={boxRef}
      className="rounded-mast-card border border-mast-warn-border bg-mast-warn-bg p-3"
    >
      {toastNode}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-mast-warn">
          {view.headline}
          {view.count > 0 && <span className="ml-1.5 tabular-nums">（{view.count}）</span>}
        </span>
        <span className="text-xs text-mast-muted">
          这一轮<b className="font-medium">停在这里</b>，处理完才会继续。
        </span>
        <div className="ml-auto">
          <Button variant="ghost" onClick={() => setCollapsed((c) => !c)}>
            {collapsed ? "展开" : "收起"}
          </Button>
        </div>
      </div>

      {data?.degraded && <DegradedNote what="中断队列" />}

      {/* 收起只收卡片。上面那一行「还有 N 项等着你」永远留着 —— 能被彻底关掉的
          提示等于没有提示，而这一轮是真的卡住的。 */}
      {view.showCards && (
        <div className="mt-3 flex flex-col gap-3">
          <p className="text-xs text-mast-muted">
            危险技能需要你批准 / 拒绝 / 改参数后批准（改过的参数会由核心 SafetyGate
            重新校验）；智能体的提问需要你作答后它才会继续。
          </p>
          {interrupts.map((it) => {
            const agent = it.agent_id || agentId;
            return (
              <InterruptCard
                key={it.event_id}
                it={fromPollRow(it, agent)}
                busy={resolve.isPending && resolve.variables?.interrupt_id === it.event_id}
                agentLabel={agentLabel}
                compact
                onResolve={(args) =>
                  resolve.mutate({ agent_id: agent, interrupt_id: it.event_id, args })
                }
              />
            );
          })}
        </div>
      )}
    </div>
  );
});
