// 出图任务：发起、顶栏小标签（兼盯「在跑 → 结束」的边沿）、出图页的任务条。
//
// 出图只有一个任务槽（设计 D21）。任务可以从大图、系列页、底部选中条、已标记页发起，
// 所以「图出好了」的提示挂在顶栏这一层——人不在出图页时也要知道。

import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { ErrorNote } from "@/components/ui";
import { viewSearch } from "@/lib/gallery/route";
import { JOB_PHASE, KIND_LABEL, describeStart, type FigureJobStatus, type FigureKind } from "@/lib/gallery/figures";
import { toast } from "../toast";
import { SMALL_BTN } from "../bits";
import { FIGURE_KEYS, useFigureStatus } from "./useFigures";

const FIGURES_LINK = { search: `?${viewSearch({ v: "figures" })}` };

export interface FigureJobRequest {
  kind: FigureKind;
  ids?: string[];
  series?: string[];
  options?: Record<string, unknown>;
}

/** 发起一个出图任务，结果用提示条说清楚（包括「已有任务在跑」这种请求成功但没开始的情况）。 */
export async function startFigureJob(qc: QueryClient, req: FigureJobRequest): Promise<boolean> {
  const prev = qc.getQueryData<FigureJobStatus>(FIGURE_KEYS.status);
  try {
    const { data, error } = await api.POST("/api/gallery/figures/run", {
      body: { kind: req.kind, ids: req.ids ?? [], series: req.series ?? [], options: req.options ?? {} },
    });
    if (error || !data) throw new Error("请求失败");
    if (!data.degraded) qc.setQueryData(FIGURE_KEYS.status, data);
    const d = describeStart(prev, data, req.kind);
    if (d.ok) {
      toast(
        <>
          {d.text} · <Link to={FIGURES_LINK}>去出图页</Link>
        </>,
      );
    } else {
      toast(d.text, "err");
    }
    return d.ok;
  } catch (e) {
    toast(`出图没有开始：${(e as Error).message}`, "err");
    return false;
  }
}

export async function cancelFigureJob(qc: QueryClient): Promise<void> {
  try {
    const { data } = await api.POST("/api/gallery/figures/cancel");
    if (data && !data.degraded) qc.setQueryData(FIGURE_KEYS.status, data);
  } catch {
    /* 下一次轮询会看到真实状态 */
  }
}

/** 顶栏上的「出图中 n/N」；任务从在跑变成不在跑的那一刻刷新产物列表并提示。 */
export function FigureJobChip() {
  const qc = useQueryClient();
  const st = useFigureStatus();
  const running = st.data?.running;
  const was = useRef<boolean | null>(null);
  useEffect(() => {
    if (running === undefined) return;
    if (was.current && !running) {
      void qc.invalidateQueries({ queryKey: FIGURE_KEYS.list });
      const d = st.data;
      if (d?.phase === "error") {
        toast(`出图出错：${d.message || d.detail || ""}`, "err");
      } else if (d?.phase === "cancelled") {
        toast("出图已取消");
      } else {
        const nErr = d?.errors?.length ?? 0;
        toast(
          <>
            出图完成：{d?.made?.length ?? 0} 张{nErr ? `，失败 ${nErr} 个` : ""} · <Link to={FIGURES_LINK}>去出图页</Link>
          </>,
          nErr ? "err" : "ok",
        );
      }
    }
    was.current = running;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  if (!running || !st.data) return null;
  const kind = st.data.kind;
  return (
    <Link
      to={FIGURES_LINK}
      className="rounded-mast-badge bg-mast-info-bg px-1.5 text-xs text-mast-info hover:no-underline"
      title={`正在出图${kind ? `：${KIND_LABEL[kind]}` : ""}`}
    >
      出图中 {st.data.done}/{st.data.total}
    </Link>
  );
}

/** 出图页上的任务条：阶段、进度、已出张数、失败、最近日志、取消。 */
export function FigureJobBar() {
  const qc = useQueryClient();
  const q = useFigureStatus();
  const st = q.data;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (!st) return null;
  const pct = st.total ? Math.round((100 * st.done) / st.total) : 0;
  const errors = st.errors ?? [];
  const log = st.log ?? [];
  return (
    <div className="mt-2.5 rounded-[3px] border border-mast-border bg-mast-panel px-3 py-2.5 text-[13px]">
      {st.degraded && (
        <div className="mb-1 text-mast-warn">出图后端不可用：{st.detail || "未知原因"}。发起的出图会提示「没有开始」。</div>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span>
          出图任务：<b>{JOB_PHASE[st.phase] ?? st.phase}</b>
          {st.kind ? ` · ${KIND_LABEL[st.kind]}` : ""}
          {st.started ? ` · 开始 ${st.started}` : ""}
          {st.finished ? ` · 结束 ${st.finished}` : ""}
        </span>
        <span className="font-mono text-xs text-mast-muted">已出 {st.made?.length ?? 0} 张</span>
        <button type="button" className={SMALL_BTN} disabled={!st.running} onClick={() => void cancelFigureJob(qc)}>
          取消
        </button>
      </div>
      {st.total > 0 && (
        <div className="mt-1.5">
          <div className="h-1.5 overflow-hidden rounded bg-mast-panel-2">
            <div className="h-full bg-mast-accent transition-[width]" style={{ width: `${pct}%` }} />
          </div>
          <div className="mt-0.5 font-mono text-xs text-mast-muted">
            {st.done} / {st.total}（{pct}%）
          </div>
        </div>
      )}
      {st.message && <div className="mt-1">{st.message}</div>}
      {errors.length > 0 && (
        <details className="mt-1.5">
          <summary className="cursor-pointer text-mast-danger">失败 {errors.length} 个</summary>
          <ul className="mt-1 max-h-48 overflow-auto font-mono text-xs">
            {errors.map((e, i) => (
              <li key={`${e.id}:${i}`}>
                {e.id} — {e.why}
              </li>
            ))}
          </ul>
        </details>
      )}
      {log.length > 0 && (
        <pre className="mt-1.5 max-h-44 overflow-auto whitespace-pre-wrap rounded bg-mast-code-bg p-2 font-mono text-xs text-mast-text">
          {log.join("\n")}
        </pre>
      )}
    </div>
  );
}
