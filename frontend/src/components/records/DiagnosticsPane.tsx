import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, Badge, Spinner, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";

// 诊断 — the refusal ledger. WHY nothing happened.
//
// The system records what SUCCEEDS. Until 2026-07-12 it recorded nothing about
// what it REFUSED, and the field trial fell straight into that gap:
//
//   「进针功能调用失败」            ← refused by WHICH layer? on what state?
//   「STS第五点停下来了」            ← ran and failed? skipped? aborted?
//   「…在预条件或安全门上反复失败而空转」
//                                     ← the operator diagnosed this themselves,
//                                       from the symptom, because we could not.
//
// This is the window onto core/diagnostics.py. Empty means nothing has been
// refused — a true statement, not a broken one.

const KIND_STYLE: Record<
  string,
  { label: string; tone: React.ComponentProps<typeof Badge>["tone"] }
> = {
  precondition_block: { label: "前置条件", tone: "WARN" },
  safety_block: { label: "安全门", tone: "DANGEROUS" },
  mode_block: { label: "模式门", tone: "WARN" },
  abort_block: { label: "中止拦截", tone: "DANGEROUS" },
  hitl_reject: { label: "人工拒绝", tone: "WARN" },
  step_skip: { label: "跳过", tone: "INFO" },
  step_fail: { label: "步骤失败", tone: "DANGEROUS" },
  step_abort: { label: "步骤中止", tone: "DANGEROUS" },
  stall: { label: "空转", tone: "DANGEROUS" },
  note: { label: "记录", tone: "INFO" },
  // ⑰（2026-08-08）本来会打断工作、现在只通知的那一类：缓冲区关键事件不再弹确认
  // 框，DANGEROUS 技能不再等审批。tone 是 INFO 而不是 WARN —— 它记的是「放行了」，
  // 把它染成告警色会让面板看起来像出了一堆事。
  notice_only: { label: "只通知", tone: "INFO" },
};

const GROUPS = [
  { id: "", label: "全部" },
  { id: "refusals", label: "拒绝（前置/安全/模式/中止）" },
  { id: "steps", label: "步骤（跳过/失败/中止）" },
  { id: "stall", label: "空转" },
  // 「本来会拦我几次」是验收时的问题，所以它得是一次点击，不是一次 grep。
  { id: "notices", label: "只通知（本来会打断）" },
] as const;

function fmtTs(t: number): string {
  const d = new Date(t * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export function DiagnosticsPane() {
  const [group, setGroup] = useState<string>("");
  const [subject, setSubject] = useState("");

  const q = useQuery({
    queryKey: ["diagnostics", group, subject],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/diagnostics", {
        params: { query: { group, subject, limit: 200 } },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  const data = q.data!;
  if (data.degraded) return <DegradedNote what="诊断台账" />;

  const top = data.summary?.top_refusals ?? [];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-baseline gap-3">
        <h2 className="text-lg font-semibold tracking-tight">诊断</h2>
        <span className="text-xs text-mast-muted">
          每一次<strong className="text-mast-text">被拒绝</strong>或
          <strong className="text-mast-text">被跳过</strong>的动作都记在这里——
          「为什么什么都没发生」的答案。
        </span>
      </div>

      {/* 形状先于细节：某一个 subject 独占计数，那就是空转本身 */}
      {top.length > 0 && (
        <Card className="space-y-2">
          <div className="text-xs font-medium text-mast-muted">最常被拒绝的动作</div>
          <div className="flex flex-wrap gap-2">
            {top.slice(0, 6).map((r) => {
              const what = String(r.what ?? "");
              const [kind, subj] = what.split(":");
              const st = KIND_STYLE[kind ?? ""] ?? { label: kind ?? "?", tone: "INFO" as const };
              return (
                <button
                  key={what}
                  type="button"
                  onClick={() => setSubject(subj ?? "")}
                  className="flex items-center gap-1.5 rounded border border-mast-border px-2 py-1 text-xs hover:border-mast-accent"
                >
                  <Badge tone={st.tone}>{st.label}</Badge>
                  <span className="font-mono text-mast-text">{subj}</span>
                  <span className="font-mono tabular-nums text-mast-muted">×{String(r.count ?? "")}</span>
                </button>
              );
            })}
          </div>
        </Card>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {GROUPS.map((g) => (
          <button
            key={g.id}
            type="button"
            onClick={() => setGroup(g.id)}
            className={
              "rounded border px-2.5 py-1 text-xs " +
              (group === g.id
                ? "border-mast-accent bg-mast-accent/10 text-mast-accent"
                : "border-mast-border text-mast-muted hover:text-mast-text")
            }
          >
            {g.label}
          </button>
        ))}
        <input
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          placeholder="按技能 / 步骤名过滤"
          className="ml-auto min-w-0 flex-1 rounded border border-mast-border bg-mast-bg px-3 py-1 text-xs text-mast-text placeholder:text-mast-muted focus:border-mast-accent focus:outline-none sm:max-w-xs"
        />
      </div>

      {data.count === 0 ? (
        <EmptyNote label="尚无拒绝记录——没有任何动作被拦下或跳过。" />
      ) : (
        <div className="space-y-1.5">
          {(data.entries ?? []).map((e) => {
            const st = KIND_STYLE[e.kind] ?? { label: e.kind, tone: "INFO" as const };
            const extra = Object.entries(e.fields ?? {}).filter(
              ([, v]) => v !== null && v !== "" && !(Array.isArray(v) && v.length === 0),
            );
            return (
              <Card key={e.seq} className="space-y-1 py-2">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-mono tabular-nums text-mast-muted">{fmtTs(e.t)}</span>
                  <Badge tone={st.tone}>{st.label}</Badge>
                  <span className="font-mono text-sm text-mast-text">{e.subject}</span>
                  {e.run_id && (
                    <span className="ml-auto font-mono text-[11px] text-mast-muted">
                      run {e.run_id}
                    </span>
                  )}
                </div>
                <p className="whitespace-pre-wrap text-sm text-mast-text">{e.reason}</p>
                {extra.length > 0 && (
                  <div className="flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-[11px] text-mast-muted">
                    {extra.map(([k, v]) => (
                      <span key={k}>
                        {k}={typeof v === "object" ? JSON.stringify(v) : String(v)}
                      </span>
                    ))}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}

      {data.summary?.log_path && (
        <p className="text-[11px] text-mast-muted">
          完整历史（跨重启）：
          <code className="ml-1 select-all font-mono text-mast-text">
            {data.summary.log_path}
          </code>
        </p>
      )}
    </div>
  );
}
