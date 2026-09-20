import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Badge, DegradedNote, EmptyNote, ErrorNote, Section, Spinner } from "@/components/ui";
import { Button, TextField, useToast } from "@/components/controls";
import { fmtTime } from "./shared";

type FeedbackEntry = components["schemas"]["FeedbackEntry"];

// 用户反馈 — the list of everything filed from the feedback float.
//
// ── why this page exists at all ────────────────────────────────────────────
// `GET /api/feedback` has existed for a while and NOTHING consumed it: the
// float is write-only, and the per-experiment drill-down filters by
// experiment_id, so the float's own rows (which have none) appeared nowhere.
// The operator could file feedback and never see it again.
//
// That is the mechanism behind「有些已经处理过了」: with no visible list, no
// timestamps and no processed marker, every batch of feedback arrived as an
// undated pile mixed old with new, and working out which entries were already
// closed meant an archaeology pass over git log, KNOWN_ISSUES and the
// machine-test checklists. One batch got that wrong in BOTH directions —
// an item re-reported after it had been fixed, and an item assumed fixed that
// was only half-fixed.
//
// So this list shows three things the pile did not: the row id (so a reference
// like「#12」 stops being ambiguous between two numbering spaces), the filing
// time, and whether it has been processed and in which version.

const RESOLVED_ROW = "opacity-55";

function useFeedback() {
  return useQuery({
    queryKey: ["feedback", "list"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/feedback", {
        params: { query: { limit: 500 } },
      });
      if (error) throw error;
      return data;
    },
  });
}

export function FeedbackPane() {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const q = useFeedback();
  // Pre-filled with the version being prepared, because typing it per row is
  // what would make people skip it — and a resolved row with no version is
  // barely better than no marker at all next time someone asks "已经修过了吗".
  const [version, setVersion] = useState("v6.2.1");
  const [showResolved, setShowResolved] = useState(true);

  const mark = useMutation({
    mutationFn: async (v: { id: number; resolved: boolean }) => {
      const { data, error } = await api.POST("/api/feedback/{feedback_id}/resolved", {
        params: { path: { feedback_id: v.id } },
        body: { resolved: v.resolved, version: v.resolved ? version : "", note: "" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d, v) => {
      // `ok=false` means the row was not there — say so rather than showing a
      // success toast over a write that did not happen.
      if (!d?.ok) toast(`第 ${v.id} 条没有更新（这一行可能已不存在）`);
      void qc.invalidateQueries({ queryKey: ["feedback", "list"] });
    },
    onError: () => toast("标记失败"),
  });

  const items: FeedbackEntry[] = q.data?.items ?? [];
  // Newest first: the reason anyone opens this is the batch that just came in.
  const rows = [...items].reverse().filter((f) => showResolved || !f.resolved_at);
  const nOpen = items.filter((f) => !f.resolved_at).length;

  return (
    <Section
      title={`用户反馈${items.length ? ` (未处理 ${nOpen} / 共 ${items.length})` : ""}`}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-mast-muted">标记版本</span>
          <div className="w-28">
            <TextField value={version} onChange={setVersion} placeholder="v6.2.1" mono />
          </div>
          <Button onClick={() => setShowResolved((v) => !v)}>
            {showResolved ? "只看未处理" : "显示全部"}
          </Button>
        </div>
      }
    >
      {node}
      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="用户反馈" />}
      {q.data && !q.data.degraded && rows.length === 0 && (
        <EmptyNote
          label={
            items.length
              ? "没有未处理的反馈——全部已标记。"
              : "还没有反馈。右下角的反馈浮窗提交的内容会出现在这里。"
          }
        />
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-mast-muted">
                <th className="py-1 pr-3 font-normal">#</th>
                <th className="py-1 pr-3 font-normal">时间</th>
                <th className="py-1 pr-3 font-normal">内容</th>
                <th className="py-1 pr-3 font-normal">来自</th>
                <th className="py-1 font-normal">状态</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((f) => (
                <tr
                  key={f.id}
                  className={
                    "border-t border-mast-border align-top " +
                    (f.resolved_at ? RESOLVED_ROW : "")
                  }
                >
                  {/* The id, shown. An operator saying「#12」 was ambiguous
                      between two numbering spaces that both reached #12 and
                      meant different things; a row id is not. */}
                  <td className="py-1.5 pr-3 font-mono text-xs text-mast-muted tabular-nums">
                    {f.id}
                  </td>
                  <td className="whitespace-nowrap py-1.5 pr-3 text-xs text-mast-muted tabular-nums">
                    {f.timestamp ? fmtTime(f.timestamp) : "—"}
                  </td>
                  <td className="py-1.5 pr-3 text-mast-text">
                    {f.comment || <span className="text-mast-muted">（只有评分）</span>}
                    {f.rating && (
                      <span className="ml-2 text-xs text-mast-muted">评分 {f.rating}</span>
                    )}
                  </td>
                  <td className="py-1.5 pr-3 text-xs text-mast-muted">
                    {String((f.meta as Record<string, unknown>)?.page ?? f.agent ?? "—")}
                  </td>
                  <td className="py-1.5">
                    {f.resolved_at ? (
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone="AUTO">
                          已在 {f.resolved_version || "某版本"} 处理
                        </Badge>
                        <button
                          type="button"
                          onClick={() => mark.mutate({ id: f.id, resolved: false })}
                          className="text-xs text-mast-muted hover:text-mast-text"
                          title="这个症状又回来了 —— 撤销标记，重新当作未处理"
                        >
                          又回来了
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => mark.mutate({ id: f.id, resolved: true })}
                        className="rounded-mast-ctl border border-mast-border px-2 py-0.5 text-xs text-mast-muted hover:text-mast-text"
                      >
                        标为已处理
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}
