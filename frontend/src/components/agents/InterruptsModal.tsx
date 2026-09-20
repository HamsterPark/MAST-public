import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Modal, useToast } from "@/components/controls";
import { agentLabel } from "./registry";
import { InterruptCard, fromPollRow, type ResolveArgs } from "./InterruptCard";

// HITL interrupts viewer + RESOLVER — reproduces the old DANGEROUS / escalate
// modal seam and (Wave E2) restores operator approval of DANGEROUS ops.
//
// Reads GET /api/agents/{id}/interrupts (poll) and resolves via
// POST /api/agents/{id}/interrupts/{event_id}/resolve. The card body is the
// SHARED InterruptCard, so a new interrupt kind (an agent's ask_user question,
// a workflow route choice) renders here without a second implementation — the
// three approval surfaces used to be three copies, and all three carried the
// same bug that left route-style interrupts with no buttons at all.
//
// LIVE-only: when the resolve response (or the list) reports degraded=true we
// surface a small note and never freeze — the buttons stay clickable and report
// an honest status.

export function InterruptsModal({
  open,
  agentId,
  onClose,
}: {
  open: boolean;
  agentId: string | null;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { toast, node } = useToast();

  const q = useQuery({
    queryKey: ["agents", "interrupts", agentId],
    enabled: open && !!agentId,
    refetchInterval: open ? 4000 : false,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/{agent_id}/interrupts", {
        params: { path: { agent_id: agentId as string } },
      });
      if (error) throw error;
      return data;
    },
  });

  const resolve = useMutation({
    mutationFn: async (vars: { agent_id: string; interrupt_id: string; args: ResolveArgs }) => {
      const { data, error } = await api.POST(
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
      return data;
    },
    onSuccess: (data, vars) => {
      const d = vars.args.decision;
      const label =
        d === "approve" ? "批准"
          : d === "reject" ? "拒绝"
          : d === "edit" ? "编辑并批准"
          : d === "answer" ? "回答" : `选择「${d}」`;
      if (data?.ok && data?.applied) {
        toast(`已${label}中断 ${vars.interrupt_id}`, "ok");
      } else if (data?.status === "answer_invalid" || data?.status === "route_not_allowed") {
        // The backend left the worker blocked on purpose so this can be fixed —
        // do NOT let it read as "resolved".
        toast(`未提交：${data?.detail ?? "回答无效"}，请修改后重试`, "err");
      } else {
        toast(`中断未生效（${data?.status ?? "degraded"}）：${data?.detail ?? "后端降级或中断已处理"}`, "err");
      }
      qc.invalidateQueries({ queryKey: ["agents", "interrupts", vars.agent_id] });
      qc.invalidateQueries({ queryKey: ["agents", "snapshot"] });
    },
    onError: (e) => toast(String((e as Error)?.message ?? e), "err"),
  });

  const interrupts = q.data?.interrupts ?? [];

  return (
    <Modal open={open} onClose={onClose} title={`人机交互中断 · ${agentId ? agentLabel(agentId) : ""}`} wide>
      {node}
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data && q.data.degraded && <DegradedNote what="中断队列" />}
      {q.data && !q.data.degraded && (
        <div className="mb-3 flex items-center gap-3 text-sm text-mast-muted">
          <span>共 {q.data.count} 个待处理中断</span>
          {q.data.interrupt_gating && <Badge tone="WARN">门控开启</Badge>}
        </div>
      )}
      {q.data && !q.data.degraded && interrupts.length === 0 && (
        <EmptyNote label="当前无待处理中断。" />
      )}
      <div className="space-y-3">
        {interrupts.map((it) => {
          const agent = it.agent_id || (agentId as string);
          return (
            <InterruptCard
              key={it.event_id}
              it={fromPollRow(it, agent)}
              busy={resolve.isPending && resolve.variables?.interrupt_id === it.event_id}
              agentLabel={agentLabel}
              onResolve={(args) =>
                resolve.mutate({ agent_id: agent, interrupt_id: it.event_id, args })
              }
            />
          );
        })}
      </div>
    </Modal>
  );
}
