import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import {
  Badge,
  Card,
  DegradedNote,
  EmptyNote,
  ErrorNote,
  Section,
  Spinner,
} from "@/components/ui";
import { Button, useToast } from "@/components/controls";

// 高级管理 → 系统/全局 → 上下文注入
//
// Every agent call carries a pile of text nobody could read in one place: a
// static SYSTEM_PROMPT per agent plus whatever the middlewares append. The
// 2026-07-27 coordinate incident (the model wrote 1.2531 for 1.2531e-6 m) came
// out of OUR OWN injection — live_state printing "(= 1253.1 nm)" beside the
// metre value — and went unnoticed because there was nowhere to look.
//
// TWO HALVES, deliberately unequal in prominence:
//   A. the inventory + editor (default view)
//   B. "agent 实际收到了什么" — real captured requests, DEFAULT COLLAPSED and
//      lazily fetched. Operator's words: 一般用户不让他们关心这个。
//
// THE HONESTY RULE: a block that needs live hardware or per-request state is
// shown as unavailable WITH the reason, never as a plausible sample. This page
// only has value if everything on it is the real text.

type PromptSummary = components["schemas"]["PromptSummary"];
type PromptDetail = components["schemas"]["PromptDetail"];
type CaptureSummary = components["schemas"]["PromptCaptureSummary"];

const CATEGORY_LABEL: Record<string, string> = {
  agent_system: "Agent 系统提示",
  routing: "路由",
  middleware: "运行时注入 (middleware)",
  sub_llm: "辅助模型提示",
};
const CATEGORY_ORDER = ["agent_system", "routing", "middleware", "sub_llm"];

type Availability = { label: string; tone: string };
const AVAILABILITY: Record<string, Availability> = {
  static: { label: "固定文本", tone: "AUTO" },
  live: { label: "实时渲染", tone: "INFO" },
  needs_hardware: { label: "需实时硬件", tone: "WARN" },
  needs_request: { label: "需运行时上下文", tone: "WARN" },
};
const UNKNOWN_AVAILABILITY: Availability = { label: "未知", tone: "INFO" };

function availabilityOf(key: string | undefined): Availability {
  return AVAILABILITY[key ?? ""] ?? UNKNOWN_AVAILABILITY;
}

function fmtChars(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k 字符` : `${n} 字符`;
}

function fmtAge(sec: number): string {
  if (sec < 60) return `${Math.round(sec)} 秒前`;
  if (sec < 3600) return `${Math.round(sec / 60)} 分钟前`;
  return `${(sec / 3600).toFixed(1)} 小时前`;
}

// ── A. inventory + editor ────────────────────────────────────────────────────

function PromptDetailPanel({ promptId, onClose }: { promptId: string; onClose: () => void }) {
  const queryClient = useQueryClient();
  const { toast, node: toastNode } = useToast();
  const [draft, setDraft] = useState<string | null>(null);

  const q = useQuery({
    queryKey: ["admin", "prompt", promptId],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/prompts/{prompt_id}", {
        params: { path: { prompt_id: promptId } },
      });
      if (error) throw error;
      return data as PromptDetail;
    },
  });

  const save = useMutation({
    mutationFn: async (text: string) => {
      const { data, error } = await api.POST("/api/admin/prompts/{prompt_id}", {
        params: { path: { prompt_id: promptId } },
        body: { text },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      setDraft(null);
      toast(data?.message || "已保存", data?.ok ? "ok" : "err");
      queryClient.invalidateQueries({ queryKey: ["admin", "prompt", promptId] });
      queryClient.invalidateQueries({ queryKey: ["admin", "prompts"] });
    },
    onError: () => toast("保存失败", "err"),
  });

  const reset = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.DELETE("/api/admin/prompts/{prompt_id}", {
        params: { path: { prompt_id: promptId } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      setDraft(null);
      toast(data?.message || "已恢复默认", "ok");
      queryClient.invalidateQueries({ queryKey: ["admin", "prompt", promptId] });
      queryClient.invalidateQueries({ queryKey: ["admin", "prompts"] });
    },
    onError: () => toast("恢复默认失败", "err"),
  });

  const d = q.data;
  const editable = !!d?.overridable;
  const value = draft ?? d?.effective_text ?? "";
  const dirty = draft !== null && draft !== (d?.effective_text ?? "");
  const avail = availabilityOf(d?.availability);

  return (
    <Card className="mt-3">
      {toastNode}
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-base font-semibold text-mast-text">{d?.label ?? promptId}</h3>
            <Badge tone={avail.tone}>{avail.label}</Badge>
            {d?.overridden && <Badge tone="WARN">已覆写</Badge>}
          </div>
          <p className="mt-1 font-mono text-[0.7rem] text-mast-muted">{d?.source || promptId}</p>
        </div>
        <Button variant="ghost" onClick={onClose}>收起</Button>
      </div>

      {q.isPending && <Spinner />}
      {q.error && <ErrorNote error={q.error} />}
      {d?.degraded && <DegradedNote what="上下文注入" />}

      {d && !d.degraded && (
        <>
          {d.note && <p className="mb-3 text-xs leading-relaxed text-mast-muted">{d.note}</p>}

          {/* The honesty surface: no body, and we say exactly why. */}
          {!d.effective_text && d.unavailable_reason && (
            <div className="mb-3 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2.5 text-sm text-mast-warn">
              {d.unavailable_reason}
            </div>
          )}

          {d.effective_text && (
            <>
              <div className="mb-1 flex items-center justify-between text-xs text-mast-muted">
                <span>{editable ? "生效中的文本（可编辑）" : "生效中的文本（只读）"}</span>
                <span className="font-mono">{fmtChars(d.effective_chars ?? 0)}</span>
              </div>
              <textarea
                value={value}
                readOnly={!editable}
                spellCheck={false}
                onChange={(e) => setDraft(e.target.value)}
                rows={18}
                className="w-full resize-y rounded-mast-ctl border border-mast-border bg-mast-bg p-3 font-mono text-[12.5px] leading-relaxed text-mast-text outline-none focus:border-mast-accent read-only:text-mast-muted"
              />
            </>
          )}

          {!editable && d.effective_text && (
            <p className="mt-2 text-xs text-mast-muted">
              这一块的内容由运行时状态计算，没有可替换的固定文本 —— 只读。
            </p>
          )}

          {editable && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Button
                variant="primary"
                loading={save.isPending}
                disabled={!dirty}
                onClick={() => save.mutate(draft ?? "")}
              >
                保存覆写
              </Button>
              <Button
                loading={reset.isPending}
                disabled={!d.overridden && !dirty}
                onClick={() => (dirty ? setDraft(null) : reset.mutate())}
              >
                {dirty ? "撤销未保存的修改" : "恢复默认"}
              </Button>
              {d.overridden && (
                <span className="text-xs text-mast-muted">
                  代码默认 {fmtChars(d.default_chars ?? 0)}，当前覆写 {fmtChars(d.effective_chars ?? 0)}。
                </span>
              )}
            </div>
          )}

          {d.overridden && d.default_text && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-mast-muted hover:text-mast-text">
                查看代码默认文本（对照用）
              </summary>
              <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-mast-ctl border border-mast-border bg-mast-bg p-3 font-mono text-[12px] leading-relaxed text-mast-muted">
                {d.default_text}
              </pre>
            </details>
          )}
        </>
      )}
    </Card>
  );
}

function PromptInventory() {
  const [query, setQuery] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);

  const q = useQuery({
    queryKey: ["admin", "prompts"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/prompts");
      if (error) throw error;
      return data;
    },
  });

  const groups = useMemo(() => {
    const items = (q.data?.items ?? []) as PromptSummary[];
    const s = query.trim().toLowerCase();
    const hit = (i: PromptSummary) =>
      !s ||
      [i.id, i.label, i.agent, i.source, i.note, i.preview]
        .some((f) => String(f ?? "").toLowerCase().includes(s));
    const filtered = items.filter(hit);
    return CATEGORY_ORDER
      .map((cat) => ({ cat, rows: filtered.filter((i) => i.category === cat) }))
      .filter((g) => g.rows.length > 0);
  }, [q.data, query]);

  return (
    <div>
      <p className="mb-3 text-xs leading-relaxed text-mast-muted">
        每次调用 agent 前注入上下文的<strong className="text-mast-text">全部话术</strong>。
        标注为「固定文本」的可以编辑并持久化（保存在 <code>config/overrides/prompt_overrides.json</code>，
        重启后仍在，且进入「覆盖历史」）；标注为「需实时硬件 / 需运行时上下文」的<strong className="text-mast-text">
        不会伪造示例文本</strong> —— 要看它们的真实内容，请展开页面下方的真实请求快照。
      </p>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="搜索…（agent 名、话术 ID、中英文关键词）"
        className="mb-3 w-full rounded-mast-ctl border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent"
      />

      {q.isPending && <Spinner label="读取注入清单…" />}
      {q.error && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="上下文注入清单" />}
      {q.data && !q.data.degraded && groups.length === 0 && <EmptyNote label="没有匹配的话术" />}

      {q.data && !q.data.degraded && (
        <>
          <div className="mb-3 text-xs text-mast-muted">
            共 {q.data.count} 条
            {(q.data.overridden_count ?? 0) > 0 && (
              <> · <span className="text-mast-warn">{q.data.overridden_count} 条已被覆写</span></>
            )}
          </div>

          {groups.map((g) => (
            <div key={g.cat} className="mb-4">
              <h3 className="mb-2 text-sm font-semibold text-mast-text">
                {CATEGORY_LABEL[g.cat] ?? g.cat}
                <span className="ml-2 text-xs font-normal text-mast-muted">{g.rows.length}</span>
              </h3>
              <div className="space-y-1.5">
                {g.rows.map((row) => {
                  const avail = availabilityOf(row.availability);
                  const isOpen = openId === row.id;
                  return (
                    <div key={row.id}>
                      <button
                        type="button"
                        onClick={() => setOpenId(isOpen ? null : row.id)}
                        className="w-full rounded-mast-ctl border border-mast-border bg-mast-panel px-3 py-2.5 text-left hover:bg-mast-panel-2"
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-mast-accent">{isOpen ? "▾" : "▸"}</span>
                          <span className="text-sm text-mast-text">{row.label}</span>
                          <Badge tone={avail.tone}>{avail.label}</Badge>
                          {row.overridden && <Badge tone="WARN">已覆写</Badge>}
                          <span className="ml-auto font-mono text-[0.7rem] text-mast-muted">
                            {row.effective_chars ? fmtChars(row.effective_chars) : "—"}
                          </span>
                        </div>
                        <p className="mt-1 truncate font-mono text-[0.7rem] text-mast-faint">{row.id}</p>
                        {row.preview ? (
                          <p className="mt-1 line-clamp-2 text-xs text-mast-muted">{row.preview}</p>
                        ) : (
                          <p className="mt-1 line-clamp-2 text-xs text-mast-warn">
                            {row.unavailable_reason}
                          </p>
                        )}
                      </button>
                      {isOpen && (
                        <PromptDetailPanel promptId={row.id} onClose={() => setOpenId(null)} />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

// ── B. "agent 实际收到了什么" — DEFAULT COLLAPSED, lazily loaded ─────────────

function CaptureDetail({ index }: { index: number }) {
  const q = useQuery({
    queryKey: ["admin", "prompt-capture", index],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/prompt-capture/{index}", {
        params: { path: { index } },
      });
      if (error) throw error;
      return data;
    },
  });

  if (q.isPending) return <Spinner label="读取快照…" />;
  if (q.error) return <ErrorNote error={q.error} />;
  const d = q.data;
  if (!d || !d.found) return <EmptyNote label="这条快照已被新的请求挤出缓冲区" />;

  return (
    <div className="mt-2 space-y-2">
      <p className="text-xs text-mast-muted">
        {new Date((d.ts ?? 0) * 1000).toLocaleString()} · {fmtAge(d.age_s ?? 0)} ·{" "}
        {d.source} · {d.model_id}
        {(d.dropped_messages ?? 0) > 0 && (
          <span className="text-mast-warn"> · 另有 {d.dropped_messages} 条消息因体积上限未记录</span>
        )}
      </p>
      {(d.messages ?? []).map((m, i) => (
        <div key={i} className="rounded-mast-ctl border border-mast-border bg-mast-bg">
          <div className="flex items-center justify-between border-b border-mast-border px-3 py-1.5 text-xs">
            <span className="font-mono text-mast-accent">{m.role}</span>
            <span className="font-mono text-mast-muted">
              {fmtChars(m.chars ?? 0)}
              {m.truncated && <span className="ml-2 text-mast-warn">已截断显示</span>}
            </span>
          </div>
          <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words p-3 font-mono text-[12px] leading-relaxed text-mast-text">
            {m.content}
          </pre>
        </div>
      ))}
    </div>
  );
}

function CapturePanel() {
  const queryClient = useQueryClient();
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const q = useQuery({
    queryKey: ["admin", "prompt-capture"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/prompt-capture");
      if (error) throw error;
      return data;
    },
  });

  const clear = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.DELETE("/api/admin/prompt-capture");
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      setOpenIndex(null);
      queryClient.invalidateQueries({ queryKey: ["admin", "prompt-capture"] });
    },
  });

  const items = (q.data?.items ?? []) as CaptureSummary[];

  return (
    <div>
      <p className="mb-3 text-xs leading-relaxed text-mast-muted">
        这里记录的是<strong className="text-mast-text">真实发生过的模型请求</strong> ——
        provider 收到的原始消息列表，含所有 middleware 拼进去的内容。
        不是模拟渲染：模拟无法复现当次的硬件读数、对话历史与 middleware 顺序，
        看起来像却不是真的东西比没有更糟。代价是它<strong className="text-mast-text">可能过期</strong>
        （每条都带时间戳），且本进程还没发生过调用时是空的。
        仅保存在内存里最近 {q.data?.capacity ?? 0} 次，不落盘。
      </p>
      <p className="mb-3 text-xs leading-relaxed text-mast-faint">
        范围说明：这里只有<strong className="text-mast-muted">消息列表</strong>。
        工具定义（IC 的 ~250 个工具 schema）是绑在模型上的，不走消息，因此不在这份记录里 ——
        要看工具清单去「Agents → 工具」。
      </p>

      <div className="mb-3 flex items-center gap-3">
        <Button onClick={() => q.refetch()} loading={q.isFetching}>刷新</Button>
        <Button
          variant="ghost"
          disabled={items.length === 0}
          loading={clear.isPending}
          onClick={() => clear.mutate()}
        >
          清空
        </Button>
        {(q.data?.total_seen ?? 0) > 0 && (
          <span className="text-xs text-mast-muted">本次运行累计 {q.data?.total_seen} 次调用</span>
        )}
      </div>

      {q.isPending && <Spinner label="读取快照列表…" />}
      {q.error && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="真实请求快照" />}
      {q.data && !q.data.degraded && items.length === 0 && (
        <EmptyNote label={q.data.note || "暂无快照"} />
      )}

      <div className="space-y-1.5">
        {items.map((row) => {
          const isOpen = openIndex === row.index;
          return (
            <div key={row.index}>
              <button
                type="button"
                onClick={() => setOpenIndex(isOpen ? null : (row.index ?? 0))}
                className="w-full rounded-mast-ctl border border-mast-border bg-mast-panel px-3 py-2.5 text-left hover:bg-mast-panel-2"
              >
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="text-mast-accent">{isOpen ? "▾" : "▸"}</span>
                  <span className="text-mast-text">{row.source}</span>
                  <Badge tone="INFO">{row.model_id || "未知模型"}</Badge>
                  <span className="text-xs text-mast-muted">{fmtAge(row.age_s ?? 0)}</span>
                  <span className="ml-auto font-mono text-[0.7rem] text-mast-muted">
                    {row.message_count} 条消息 · system {fmtChars(row.system_chars ?? 0)} · 合计{" "}
                    {fmtChars(row.total_chars ?? 0)}
                  </span>
                </div>
              </button>
              {isOpen && <CaptureDetail index={row.index ?? 0} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The whole 上下文注入 surface. Part B is collapsed until asked for — it does
 *  not render, and does not fetch, while closed. */
export function PromptInspector() {
  const [showCapture, setShowCapture] = useState(false);

  return (
    <div>
      <Section title="上下文注入话术">
        <PromptInventory />
      </Section>

      <div className="overflow-hidden rounded-mast-card border border-mast-border bg-mast-panel">
        <button
          type="button"
          onClick={() => setShowCapture((v) => !v)}
          className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-mast-panel-2"
        >
          <span className="text-sm text-mast-muted">
            <span className="text-mast-accent">{showCapture ? "▾" : "▸"}</span>{" "}
            agent 实际收到了什么（真实请求快照）
            <span className="ml-2 text-xs text-mast-faint">排障用，一般不需要打开</span>
          </span>
          <span className="text-mast-muted">{showCapture ? "收起" : "展开"}</span>
        </button>
        {showCapture && (
          <div className="border-t border-mast-border p-4">
            <CapturePanel />
          </div>
        )}
      </div>
    </div>
  );
}

export default PromptInspector;
