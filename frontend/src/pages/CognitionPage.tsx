import { useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import {
  Section,
  Card,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
  EmptyNote,
} from "@/components/ui";
import { Accordion } from "@/components/controls";
import { PhaseSummaries } from "@/components/cognition/PhaseSummaries";

// CognitionPage — Domain F (记忆 / 认知). Full-parity port of the old Gradio
// 记忆 tab (mast.gui.cognition_panel) over the typed FastAPI seam.
//
// The old tab was ONE single-column stack (NOT sub-tabs, NOT 2-column):
//   header markdown
//   filter row: 命名空间 + 类型 + 搜索 + ↻ 刷新
//   memory list (full width)
//   Accordion「查看 / 编辑一条记忆」(折叠) — path + 载入 + detail + editor
//   Accordion「🌙 做梦整合 / 🧠 头脑风暴」(折叠)
//   Accordion「对话阶段摘要(分片)」(折叠)
// Reproduced here with plain React state (no nested gr.Tabs → no freeze).

import type { components } from "@/api/schema";

type MemoryEntry = components["schemas"]["MemoryEntry"];
type BrainstormTurn = components["schemas"]["BrainstormTurn"];
type DreamEntry = components["schemas"]["DreamEntry"];

// Memory kinds (mirrors mast.memory.store.KINDS used by cognition_panel).
const KINDS = [
  "note",
  "insight",
  "summary",
  "hypothesis",
  "protocol",
  "dream",
  "brainstorm",
] as const;

// Editor kind choices — old _mem_savekind only allowed user-writable kinds.
const SAVE_KINDS = ["note", "insight", "summary", "hypothesis", "protocol"] as const;

const KIND_ICON: Record<string, string> = {
  note: "📝",
  insight: "💡",
  summary: "🗂",
  hypothesis: "🔬",
  protocol: "📐",
  dream: "🌙",
  brainstorm: "🧠",
};

// ── queries ───────────────────────────────────────────────────────────

function useNamespaces() {
  return useQuery({
    queryKey: ["memory", "namespaces"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/memory/namespaces");
      if (error) throw error;
      return data;
    },
  });
}

function useMemoryList(ns: string, kind: string, query: string) {
  return useQuery({
    queryKey: ["memory", "list", ns, kind, query],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/memory/{ns}", {
        params: {
          path: { ns },
          query: {
            kind: kind || null,
            query: query || null,
          },
        },
      });
      if (error) throw error;
      return data;
    },
    enabled: !!ns,
  });
}

function useMemoryDetail(ns: string, path: string | null) {
  return useQuery({
    queryKey: ["memory", "detail", ns, path],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/memory/{ns}/{path}", {
        params: { path: { ns, path: path as string } },
      });
      if (error) throw error;
      return data;
    },
    enabled: !!ns && !!path,
  });
}

// ── page ──────────────────────────────────────────────────────────────

export default function CognitionPage() {
  const namespaces = useNamespaces();

  const [ns, setNs] = useState<string>("global");
  const [kind, setKind] = useState<string>("");
  const [query, setQuery] = useState<string>("");
  const [selected, setSelected] = useState<string | null>(null);

  // The default namespace becomes whatever the backend lists first ("global").
  const nsOptions = namespaces.data?.namespaces ?? ["global"];
  const effectiveNs = nsOptions.includes(ns) ? ns : nsOptions[0] ?? "global";

  return (
    <div>
      <Section title="持久化记忆">
        <p className="mb-4 text-sm text-mast-muted">
          <strong>持久化记忆</strong> — agent 跨 session 读写的结构化记忆（依附实验记录）。📌
          置顶优先注入上下文。🌙 做梦 / 🧠 头脑风暴产物均标注「非实测」。
        </p>

        {namespaces.isPending && <Spinner />}
        {namespaces.isError && <ErrorNote error={namespaces.error} />}
        {namespaces.data?.degraded && <DegradedNote what="记忆存储" />}

        {namespaces.data && !namespaces.data.degraded && (
          <>
            <MemoryBrowser
              ns={effectiveNs}
              nsOptions={nsOptions}
              kind={kind}
              query={query}
              selected={selected}
              onNs={(v) => {
                setNs(v);
                setSelected(null);
              }}
              onKind={setKind}
              onQuery={setQuery}
              onSelect={setSelected}
            />

            <Accordion title="查看 / 编辑一条记忆" defaultOpen={false}>
              <MemoryEditorPanel
                ns={effectiveNs}
                path={selected}
                onCleared={() => setSelected(null)}
              />
            </Accordion>

            <Accordion title="🌙 做梦整合 / 🧠 头脑风暴" defaultOpen={false}>
              <DreamCard ns={effectiveNs} />
              <div className="mt-4 border-t border-mast-border pt-4">
                <BrainstormCard />
              </div>
            </Accordion>

            <Accordion title="对话阶段摘要(分片)" defaultOpen={false}>
              <PhaseSummaries />
            </Accordion>
          </>
        )}
      </Section>
    </div>
  );
}

// ── memory browser (filter row + list) ────────────────────────────────

function MemoryBrowser({
  ns,
  nsOptions,
  kind,
  query,
  selected,
  onNs,
  onKind,
  onQuery,
  onSelect,
}: {
  ns: string;
  nsOptions: string[];
  kind: string;
  query: string;
  selected: string | null;
  onNs: (v: string) => void;
  onKind: (v: string) => void;
  onQuery: (v: string) => void;
  onSelect: (path: string) => void;
}) {
  const list = useMemoryList(ns, kind, query);
  const entries: MemoryEntry[] = list.data?.entries ?? [];

  return (
    <Card className="mb-3">
      {/* Old single filter row: 命名空间 | 类型 | 搜索 | ↻ 刷新 */}
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-mast-muted">
          命名空间
          <select
            value={ns}
            onChange={(e) => onNs(e.target.value)}
            className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
          >
            {nsOptions.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-mast-muted">
          类型
          <select
            value={kind}
            onChange={(e) => onKind(e.target.value)}
            className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
          >
            <option value="">(全部)</option>
            {KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-1 flex-col gap-1 text-xs text-mast-muted">
          搜索
          <input
            value={query}
            onChange={(e) => onQuery(e.target.value)}
            placeholder="标题/正文/标签"
            className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
          />
        </label>
        <button
          onClick={() => list.refetch()}
          className="rounded border border-mast-border px-3 py-1.5 text-sm text-mast-text hover:bg-mast-bg/60"
        >
          ↻ 刷新
        </button>
      </div>

      {list.isPending && <Spinner />}
      {list.isError && <ErrorNote error={list.error} />}
      {list.data?.degraded && <DegradedNote what="记忆存储" />}
      {list.data && !list.data.degraded && entries.length === 0 && (
        <EmptyNote label={`命名空间 ${ns} 暂无记忆。`} />
      )}
      {list.data && !list.data.degraded && entries.length > 0 && (
        <div className="space-y-1">
          <p className="px-1 text-xs text-mast-muted">
            {list.data.count} 条 · 命名空间 <code>{ns}</code>
          </p>
          <div className="divide-y divide-mast-border overflow-hidden rounded border border-mast-border">
            {entries.map((e) => {
              const path = e.path ?? "";
              const title =
                e.title || (e.content ?? "").slice(0, 60).replace(/\n/g, " ");
              const isSel = selected === path;
              return (
                <button
                  key={`${e.id ?? path}`}
                  onClick={() => onSelect(path)}
                  className={
                    "block w-full px-3 py-2 text-left text-sm hover:bg-mast-bg/60 " +
                    (isSel ? "bg-mast-accent/10" : "")
                  }
                >
                  <div className="flex items-center gap-1.5">
                    {e.pinned && <span title="已置顶">📌</span>}
                    <span>{KIND_ICON[e.kind] ?? "📄"}</span>
                    <code className="text-mast-accent">{path}</code>
                    <span className="text-xs text-mast-muted">
                      ({e.kind}
                      {e.author ? ` · ${e.author}` : ""})
                    </span>
                  </div>
                  {title && (
                    <div className="mt-0.5 truncate text-mast-muted">{title}</div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </Card>
  );
}

// ── detail + editor ───────────────────────────────────────────────────

function MemoryEditorPanel({
  ns,
  path,
  onCleared,
}: {
  ns: string;
  path: string | null;
  onCleared: () => void;
}) {
  const detail = useMemoryDetail(ns, path);
  const entry = detail.data?.entry ?? null;

  return (
    <div>
      {path && detail.isPending && <Spinner />}
      {path && detail.isError && <ErrorNote error={detail.error} />}
      {path && detail.data?.degraded && <DegradedNote what="记忆存储" />}

      {/* editor — also used to create a new entry when nothing is selected */}
      {(!path || (detail.data && !detail.data.degraded)) && (
        <MemoryEditor
          ns={ns}
          path={path}
          entry={path && detail.data?.found ? entry : null}
          onDeleted={onCleared}
        />
      )}
    </div>
  );
}

function MemoryEditor({
  ns,
  path,
  entry,
  onDeleted,
}: {
  ns: string;
  path: string | null;
  entry: MemoryEntry | null;
  onDeleted: () => void;
}) {
  const queryClient = useQueryClient();
  const isExisting = !!path;

  // Editable fields; reset when the selected entry changes (see formKey below).
  const initial = useMemo(
    () => ({
      path: path ?? "",
      title: entry?.title ?? "",
      kind: entry?.kind ?? "note",
      content: entry?.content ?? "",
      tags: (entry?.tags ?? []).join(", "),
      pin: !!entry?.pinned,
    }),
    [path, entry],
  );

  const [form, setForm] = useState(initial);
  const [formKey, setFormKey] = useState("");
  const curKey = `${ns}::${path ?? ""}::${entry?.updated_at ?? ""}`;
  if (formKey !== curKey) {
    // Reset the controlled form when the selected entry changes.
    setFormKey(curKey);
    setForm(initial);
  }

  const [status, setStatus] = useState<string>("");

  const invalidate = (p: string) => {
    queryClient.invalidateQueries({ queryKey: ["memory", "list", ns] });
    queryClient.invalidateQueries({ queryKey: ["memory", "detail", ns, p] });
    queryClient.invalidateQueries({ queryKey: ["memory", "namespaces"] });
  };

  const save = useMutation({
    mutationFn: async () => {
      const p = form.path.trim();
      if (!p) throw new Error("需要 path（如 insights/tip.md）");
      const { data, error } = await api.POST("/api/memory/{ns}/{path}", {
        params: { path: { ns, path: p } },
        body: {
          content: form.content,
          title: form.title,
          kind: form.kind,
          tags: form.tags
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean),
          pin: !!form.pin,
        },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.degraded) setStatus("⚠ 记忆存储未连接（写入已延后）");
      else
        setStatus(
          `✅ 已保存 ${data?.namespace ?? ns}/${data?.path ?? form.path.trim()}`,
        );
      invalidate(form.path.trim());
    },
    onError: (e) => setStatus(`❌ 保存失败: ${String((e as Error).message ?? e)}`),
  });

  const del = useMutation({
    mutationFn: async () => {
      if (!path) throw new Error("无选中记忆");
      const { data, error } = await api.DELETE("/api/memory/{ns}/{path}", {
        params: { path: { ns, path } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      setStatus(data?.degraded ? "⚠ 记忆存储未连接" : "🗑 已删除");
      invalidate(path ?? "");
      onDeleted();
    },
    onError: (e) => setStatus(`❌ 删除失败: ${String((e as Error).message ?? e)}`),
  });

  const busy = save.isPending || del.isPending;

  return (
    <div className="space-y-3">
      {/* Old row: path dropdown + 载入. Here path is the selected entry or a
          new-entry input; "＋ 新建" clears the selection. */}
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-1 flex-col gap-1 text-xs text-mast-muted">
          path
          <input
            value={form.path}
            disabled={isExisting}
            onChange={(e) => setForm({ ...form, path: e.target.value })}
            placeholder="insights/tip.md"
            className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text disabled:opacity-60"
          />
        </label>
        {isExisting && (
          <button
            onClick={onDeleted}
            className="rounded border border-mast-border px-3 py-1.5 text-sm text-mast-text hover:bg-mast-bg/60"
          >
            ＋ 新建
          </button>
        )}
      </div>

      <label className="flex flex-col gap-1 text-xs text-mast-muted">
        正文(markdown)
        <textarea
          value={form.content}
          onChange={(e) => setForm({ ...form, content: e.target.value })}
          rows={8}
          className="rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm text-mast-text"
        />
      </label>

      {/* Old row: kind + 置顶 + 保存 + 删除 */}
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-mast-muted">
          kind
          <select
            value={form.kind}
            onChange={(e) => setForm({ ...form, kind: e.target.value })}
            className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
          >
            {SAVE_KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 pb-1.5 text-sm text-mast-text">
          <input
            type="checkbox"
            checked={!!form.pin}
            onChange={(e) => setForm({ ...form, pin: e.target.checked })}
            className="h-4 w-4 accent-mast-accent"
          />
          置顶
        </label>
        <button
          onClick={() => save.mutate()}
          disabled={busy}
          className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
        >
          {save.isPending ? "保存中…" : "保存"}
        </button>
        {isExisting && (
          <button
            onClick={() => del.mutate()}
            disabled={busy}
            className="rounded border border-mast-danger-border px-3 py-1.5 text-sm text-mast-danger hover:bg-mast-danger-bg disabled:opacity-50"
          >
            {del.isPending ? "删除中…" : "删除"}
          </button>
        )}
      </div>

      {status && <p className="text-sm text-mast-muted">{status}</p>}
    </div>
  );
}

// ── dreaming trigger ──────────────────────────────────────────────────

function DreamCard({ ns }: { ns: string }) {
  // GET /api/cognition/dream/full — runs one consolidation pass and reads back
  // each written entry's FULL content. Degrade-safe.
  const dream = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.GET("/api/cognition/dream/full", {
        params: { query: { namespace: ns } },
      });
      if (error) throw error;
      return data;
    },
  });

  const result = dream.data;
  const entries: DreamEntry[] = result?.entries ?? [];

  return (
    <div>
      <button
        onClick={() => dream.mutate()}
        disabled={dream.isPending}
        className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
      >
        {dream.isPending ? "⏳ 做梦中…" : "🌙 立即做梦整合（后台读历史实验 → 固化记忆，非实测）"}
      </button>

      {dream.isError && (
        <p className="mt-3 text-sm text-mast-danger">
          ❌ 做梦失败: {String((dream.error as Error)?.message ?? dream.error)}
        </p>
      )}
      {result && (
        <div className="mt-3 space-y-2">
          {result.degraded && <DegradedNote what="认知后端" />}
          {!result.degraded && (
            <p className="text-sm text-mast-muted">
              {entries.length === 0
                ? "🌙 做梦完成：无新的可固化内容（或已固化、被去重）。"
                : `🌙 做梦完成：写入 ${result.count} 条记忆。标注「非实测，仅供参考」。`}
            </p>
          )}
          {!result.degraded && entries.length > 0 && (
            <ul className="space-y-2">
              {entries.map((e, i) => (
                <li
                  key={`${e.path || i}`}
                  className="rounded border border-mast-border bg-mast-bg/40 p-2"
                >
                  <div className="mb-1 flex items-center gap-1.5 text-xs">
                    <span>{KIND_ICON[e.kind] ?? "🌙"}</span>
                    <code className="text-mast-accent">{e.path}</code>
                    {e.title && (
                      <span className="text-mast-muted">· {e.title}</span>
                    )}
                  </div>
                  <Markdownish text={e.content} />
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

// ── brainstorm trigger ────────────────────────────────────────────────

type StreamFrame = {
  kind?: string;
  round?: number;
  speaker?: string;
  role?: string;
  viewpoint?: string | null;
  content?: string;
  message?: string;
  degraded?: boolean;
  streaming?: boolean;
  note?: string;
};

function BrainstormCard() {
  const [topic, setTopic] = useState("");
  const [viewpoints, setViewpoints] = useState("");
  const [experimentId, setExperimentId] = useState("");
  // Old run_brainstorm_panel default max_rounds=2.
  const rounds = 2;

  const [streaming, setStreaming] = useState(false);
  const [turns, setTurns] = useState<BrainstormTurn[]>([]);
  const [summary, setSummary] = useState<string>("");
  const [writtenIds, setWrittenIds] = useState<string[]>([]);
  const [degraded, setDegraded] = useState(false);
  const [note, setNote] = useState<string>("");
  const [err, setErr] = useState<string | null>(null);
  const [ran, setRan] = useState(false);
  const abortCtrl = useRef<AbortController | null>(null);

  function buildBody() {
    return {
      topic,
      viewpoints: viewpoints
        .split("\n")
        .map((v) => v.trim())
        .filter(Boolean),
      max_rounds: rounds,
      experiment_id: experimentId.trim() || null,
    };
  }

  async function runFull(body: ReturnType<typeof buildBody>) {
    const { data, error } = await api.POST("/api/cognition/brainstorm/full", {
      body,
    });
    if (error) throw error;
    if (!data) return;
    setDegraded(!!data.degraded);
    setNote(data.message ?? "");
    setSummary(data.summary ?? "");
    setTurns((data.transcript ?? []) as BrainstormTurn[]);
    setWrittenIds(data.written_memory_ids ?? []);
  }

  async function runStream(body: ReturnType<typeof buildBody>) {
    const ctrl = new AbortController();
    abortCtrl.current = ctrl;
    const resp = await fetch("/api/cognition/brainstorm/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const rawFrame = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const payload = rawFrame
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trim())
          .join("");
        if (!payload) continue;
        let frame: StreamFrame;
        try {
          frame = JSON.parse(payload) as StreamFrame;
        } catch {
          continue;
        }
        if (frame.kind === "turn") {
          setTurns((prev) => [
            ...prev,
            {
              round: frame.round ?? 0,
              speaker: frame.speaker ?? "",
              role: frame.role ?? "",
              viewpoint: frame.viewpoint ?? null,
              content: frame.content ?? "",
            },
          ]);
        } else if (frame.kind === "summary") {
          setSummary(frame.content ?? "");
        } else if (frame.kind === "error") {
          if (frame.degraded) {
            setDegraded(true);
            setNote(frame.message ?? "认知后端未连接");
          } else {
            setErr(frame.message ?? "头脑风暴出错");
          }
        } else if (frame.kind === "done") {
          if (frame.degraded) setDegraded(true);
          if (frame.note) setNote((n) => n || (frame.note as string));
        }
      }
    }
  }

  async function start() {
    if (streaming) return;
    setStreaming(true);
    setRan(true);
    setErr(null);
    setDegraded(false);
    setNote("");
    setTurns([]);
    setSummary("");
    setWrittenIds([]);
    try {
      await runStream(buildBody());
    } catch (e) {
      if ((e as Error)?.name === "AbortError") {
        // user-cancelled; leave whatever was accumulated.
      } else {
        try {
          await runFull(buildBody());
        } catch (e2) {
          setErr(String((e2 as Error)?.message ?? e2));
        }
      }
    } finally {
      setStreaming(false);
      abortCtrl.current = null;
    }
  }

  function stop() {
    abortCtrl.current?.abort();
  }

  const rounds_ = useMemo(() => {
    const byRound = new Map<number, BrainstormTurn[]>();
    for (const t of turns) {
      const r = t.round ?? 0;
      if (!byRound.has(r)) byRound.set(r, []);
      byRound.get(r)!.push(t);
    }
    return [...byRound.entries()].sort((a, b) => a[0] - b[0]);
  }, [turns]);

  return (
    <div>
      <p className="mb-3 text-sm text-mast-text">
        <strong>🧠 头脑风暴</strong> — 多 agent 就当前实验与进度讨论；你也可加观点。
      </p>
      <div className="space-y-3">
        {/* Old row: 议题 + 实验ID(可空=global) */}
        <div className="flex flex-wrap gap-3">
          <label className="flex flex-[2] flex-col gap-1 text-xs text-mast-muted">
            议题
            <input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="如：下一步如何提升分辨率"
              disabled={streaming}
              className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text disabled:opacity-60"
            />
          </label>
          <label className="flex flex-1 flex-col gap-1 text-xs text-mast-muted">
            实验ID(可空=global)
            <input
              value={experimentId}
              onChange={(e) => setExperimentId(e.target.value)}
              disabled={streaming}
              className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text disabled:opacity-60"
            />
          </label>
        </div>
        <label className="flex flex-col gap-1 text-xs text-mast-muted">
          你的观点(每行一条)
          <textarea
            value={viewpoints}
            onChange={(e) => setViewpoints(e.target.value)}
            rows={2}
            disabled={streaming}
            className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text disabled:opacity-60"
          />
        </label>
        <div className="flex items-center gap-2">
          <button
            onClick={start}
            disabled={streaming}
            className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
          >
            {streaming ? "⏳ 讨论进行中…" : "发起讨论"}
          </button>
          {streaming && (
            <button
              onClick={stop}
              className="rounded border border-mast-border px-3 py-1.5 text-sm text-mast-text hover:bg-mast-bg/60"
            >
              停止
            </button>
          )}
        </div>

        {err && <p className="text-sm text-mast-danger">失败：{err}</p>}
        {degraded && <DegradedNote what="认知后端" />}
        {note && !degraded && <p className="text-xs text-mast-muted">{note}</p>}

        {ran && !degraded && (
          <BrainstormTranscript
            rounds={rounds_}
            summary={summary}
            writtenIds={writtenIds}
            streaming={streaming}
          />
        )}
      </div>
    </div>
  );
}

// ── brainstorm transcript renderer (grouped by round) ─────────────────

function BrainstormTranscript({
  rounds,
  summary,
  writtenIds,
  streaming,
}: {
  rounds: [number, BrainstormTurn[]][];
  summary: string;
  writtenIds: string[];
  streaming: boolean;
}) {
  if (rounds.length === 0 && !summary) {
    return streaming ? (
      <p className="mt-2 text-xs text-mast-muted">讨论进行中…</p>
    ) : (
      <EmptyNote label="（空）" />
    );
  }
  return (
    <div className="mt-2 space-y-3">
      {rounds.map(([round, turns]) => (
        <div key={round}>
          <p className="mb-1 text-xs font-semibold text-mast-muted">第 {round} 轮</p>
          <ul className="space-y-2">
            {turns.map((t, i) => (
              <li
                key={i}
                className="rounded border border-mast-border bg-mast-bg/40 p-2"
              >
                <div className="mb-1 flex items-center gap-1.5 text-xs">
                  <span className="font-semibold text-mast-text">
                    {t.speaker || t.role || "发言"}
                  </span>
                  {t.viewpoint && <Badge tone="INFO">{t.viewpoint}</Badge>}
                </div>
                <Markdownish text={t.content} />
              </li>
            ))}
          </ul>
        </div>
      ))}

      {summary && (
        <div className="rounded border border-mast-accent/40 bg-mast-accent/5 p-2">
          <p className="mb-1 text-xs font-semibold text-mast-accent">综合摘要</p>
          <Markdownish text={summary} />
        </div>
      )}

      {writtenIds.length > 0 && (
        <div className="text-xs text-mast-muted">
          <span className="font-semibold">已写入记忆：</span>
          <ul className="ml-4 list-disc">
            {writtenIds.map((p) => (
              <li key={p}>
                <code>{p}</code>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── tiny markdown-ish renderer (paragraphs + bullet lists) ─────────────

function Markdownish({ text }: { text: string }) {
  const lines = (text ?? "").split("\n");
  const blocks: { type: "p" | "ul"; lines: string[] }[] = [];
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    const isBullet = /^\s*[-*]\s+/.test(line);
    const last = blocks[blocks.length - 1];
    if (isBullet) {
      if (last && last.type === "ul") last.lines.push(line.replace(/^\s*[-*]\s+/, ""));
      else blocks.push({ type: "ul", lines: [line.replace(/^\s*[-*]\s+/, "")] });
    } else {
      if (last && last.type === "p") last.lines.push(line);
      else blocks.push({ type: "p", lines: [line] });
    }
  }
  return (
    <div className="space-y-1 text-sm text-mast-text">
      {blocks.map((b, i) =>
        b.type === "ul" ? (
          <ul key={i} className="ml-4 list-disc space-y-0.5">
            {b.lines.map((l, j) => (
              <li key={j}>{renderInline(l)}</li>
            ))}
          </ul>
        ) : (
          <p key={i} className="whitespace-pre-wrap break-words">
            {b.lines.map((l, j) => (
              <span key={j}>
                {renderInline(l)}
                {j < b.lines.length - 1 ? "\n" : ""}
              </span>
            ))}
          </p>
        ),
      )}
    </div>
  );
}

function renderInline(text: string) {
  const parts: ReactNode[] = [];
  const re = /(\*\*([^*]+)\*\*|`([^`]+)`)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[2] !== undefined) {
      parts.push(<strong key={k++}>{m[2]}</strong>);
    } else if (m[3] !== undefined) {
      parts.push(
        <code key={k++} className="rounded bg-mast-bg px-1 text-mast-accent">
          {m[3]}
        </code>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}
