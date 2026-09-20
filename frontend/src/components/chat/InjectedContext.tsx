import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { api } from "@/api/client";
import {
  matchInjectedContext,
  nextTurnAfter,
  splitInjected,
  type CaptureList,
  type CapturedMsg,
} from "@/lib/injectedContext";

/** 「我这句话发出去之后，系统往模型那儿塞了什么」—— 挂在每条用户消息下面的折叠块。
 *
 *  对话里用一个可折叠/展开的按钮展示系统自动注入的
 *  上下文，便于调试。
 *
 *  ## 一个请求都不发，直到有人点开
 *
 *  展开之前这个组件**不查任何接口**（两个 query 都 `enabled: open`）。转录里有几十
 *  条用户消息，每条都挂一个轮询就是几十路无人看的流量。折叠状态下它只是一个按钮。
 *
 *  ## 展示的是真实请求，不是重放
 *
 *  数据来自 `mast/prompts/capture.py` 的环形缓冲 —— 模型**真正收到**的那份消息列表。
 *  不做离线重放：重放复现不了当次的硬件读数、对话历史和中间件顺序，产出的是「像」
 *  注入而不是「是」注入，而那正是拿它调试的人最不需要的东西。
 *
 *  代价是它可能对不上（进程重启过、这一轮被挤出缓冲、捕获被关掉）。**每一种对不上
 *  都单独说**，见 `lib/injectedContext.ts` —— 一句笼统的「暂无数据」会让五种原因
 *  看起来都像「再点一次说不定就有了」。
 */

function fmtChars(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k 字符`;
  return `${n} 字符`;
}

/** 展开之后的正文：按 seq 取那一次请求的完整消息列表。 */
function CaptureBody({ seq }: { seq: number }) {
  const q = useQuery({
    // 按 **seq** 取，不按 index —— index 每来一次模型调用就整体挪位，
    // 而这个面板正是在 agent 边跑边看的时候打开的。
    queryKey: ["prompt-capture", "by-seq", seq],
    queryFn: async () => {
      const { data, error } = await api.GET(
        "/api/admin/prompt-capture/by-seq/{seq}",
        { params: { path: { seq } } },
      );
      if (error) throw error;
      return data;
    },
    staleTime: Infinity,   // 一次请求的内容不会再变
  });

  if (q.isPending) {
    return <p className="px-3 py-2 text-[11px] text-mast-muted">读取快照…</p>;
  }
  if (q.error || !q.data) {
    return (
      <p className="px-3 py-2 text-[11px] text-mast-warn">
        读不到这份快照（{String(q.error ?? "无回包")}）。
      </p>
    );
  }
  const d = q.data;
  if (!d.found) {
    return (
      <p className="px-3 py-2 text-[11px] text-mast-warn">
        这一份已经被后面的请求挤出记录了 —— 不是没有注入过。
      </p>
    );
  }

  const messages = (d.messages ?? []) as CapturedMsg[];
  const { injected, history } = splitInjected(messages);

  return (
    <div className="space-y-2 px-3 pb-3 pt-2">
      <p className="text-[11px] leading-relaxed text-mast-faint">
        {new Date((d.ts ?? 0) * 1000).toLocaleString()} · {d.source} ·{" "}
        {d.model_id}
        {(d.dropped_messages ?? 0) > 0 && (
          <span className="text-mast-warn">
            {" "}
            · 另有 {d.dropped_messages} 条消息因体积上限未记录
          </span>
        )}
      </p>
      {/* 边界要说出来：一个以为这就是全部载荷的人，会对 token 花销和「模型
          当时看得见什么」得出错的结论。工具定义绑在模型上，不走消息列表。 */}
      {/* JSX 里 `**粗体**` 是**字面星号** —— 它不过 markdown。要强调就用标签。 */}
      <p className="text-[11px] leading-relaxed text-mast-faint">
        这里只有<strong className="text-mast-muted">消息列表</strong>
        。那几百个仪器控制工具的定义是绑在模型上的，不走消息，所以不在这份记录里
        （要看去 Agents → 工具）。
      </p>

      {injected.length === 0 ? (
        <p className="text-[11px] text-mast-warn">
          这次请求里一条 system 消息都没有 —— 也就是说这一轮
          <strong>没有系统注入</strong>。
        </p>
      ) : (
        injected.map((m, i) => (
          <div
            key={`sys-${i}`}
            className="rounded-mast-ctl border border-mast-border bg-mast-bg"
          >
            <div className="flex items-center justify-between border-b border-mast-border px-2.5 py-1 text-[10px]">
              <span className="font-mono text-mast-accent">{m.role}</span>
              <span className="font-mono text-mast-muted">
                {fmtChars(m.chars ?? 0)}
                {m.truncated && (
                  <span className="ml-2 text-mast-warn">已截断显示</span>
                )}
              </span>
            </div>
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words p-2.5 font-mono text-[11px] leading-relaxed text-mast-text">
              {m.content}
            </pre>
          </div>
        ))
      )}

      {history.length > 0 && (
        <details className="rounded-mast-ctl border border-mast-border bg-mast-bg">
          <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[11px] text-mast-muted">
            同时带过去的对话历史 {history.length} 条（不是注入，屏幕上就看得见）
          </summary>
          <div className="space-y-1.5 px-2.5 pb-2.5">
            {history.map((m, i) => (
              <div key={`hist-${i}`}>
                <div className="font-mono text-[10px] text-mast-faint">
                  {m.role} · {fmtChars(m.chars ?? 0)}
                </div>
                <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-mast-muted">
                  {m.content}
                </pre>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

export function InjectedContext({
  t,
  turnTimes,
  agentId,
}: {
  /** 这条用户消息的 epoch 秒。没有就永远对不上，而那也要说出来。 */
  t?: number | null;
  /** **整份**转录里所有用户消息的时间（升序）—— 见 `userTurnTimes` 的自述。 */
  turnTimes: number[];
  agentId: string;
}) {
  const [open, setOpen] = useState(false);

  const listQ = useQuery({
    queryKey: ["prompt-capture", "list"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/prompt-capture");
      if (error) throw error;
      return data as CaptureList;
    },
    enabled: open,        // 折叠着就一个请求都不发
    staleTime: 2000,
  });

  const body = () => {
    if (listQ.isPending) {
      return <p className="px-3 py-2 text-[11px] text-mast-muted">对齐这一轮…</p>;
    }
    if (listQ.error || !listQ.data) {
      return (
        <p className="px-3 py-2 text-[11px] text-mast-warn">
          读不到注入记录（{String(listQ.error ?? "无回包")}）。
        </p>
      );
    }
    const m = matchInjectedContext(
      t, nextTurnAfter(turnTimes, t), listQ.data, agentId);
    if (m.kind === "none") {
      return (
        <p className="px-3 py-2 text-[11px] leading-relaxed text-mast-muted">
          {m.why}
        </p>
      );
    }
    return (
      <>
        <p className="px-3 pt-2 text-[11px] leading-relaxed text-mast-faint">
          这一轮发出 {m.calls} 次模型调用；下面是
          <strong className="text-mast-muted">第一次</strong>带过去的那份
          （system {fmtChars(m.systemChars)}，整个请求 {fmtChars(m.totalChars)}）。
          {m.foreignSource && (
            <span className="text-mast-warn">
              {" "}
              ⚠️ 这一次是 {m.source} 发的，不是本对话的 {agentId} —— 时间窗里没有
              后者的请求，展示的是同一时刻别人的那次。
            </span>
          )}
        </p>
        <CaptureBody seq={m.seq} />
      </>
    );
  };

  return (
    <div className="mt-1.5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={clsx(
          "inline-flex items-center gap-1 rounded-mast-badge px-1.5 py-0.5",
          "font-mono text-[10px] text-mast-faint transition-colors",
          "hover:bg-mast-panel-2 hover:text-mast-muted",
        )}
        aria-expanded={open}
      >
        <span className={clsx("transition-transform", open && "rotate-90")}>
          ▸
        </span>
        注入的上下文
      </button>
      {open && (
        <div className="mt-1 rounded-mast-ctl border border-mast-border bg-mast-panel text-left">
          {body()}
        </div>
      )}
    </div>
  );
}
