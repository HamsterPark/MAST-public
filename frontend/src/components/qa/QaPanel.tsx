import { useState } from "react";
import { create } from "zustand";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";
import { useDraft } from "@/hooks/useDraft";
import { Card, Badge, EmptyNote } from "@/components/ui";
import { Button } from "@/components/controls";

// 查询助手 — single-turn, read-only knowledge Q&A (parity rebuild of the old
// Gradio 查询助手 tab). The old tab drove an *independent* QuickAskAgent (its
// own model = qa_model, distinct from the main chat) and called
// `QuickAskAgent.one_shot(text, scope=...)` directly in-process: single-turn,
// no history persisted, never touches hardware, never writes the experiment
// log. Wired to the REAL backend POST /api/qa (QuickAskAgent.one_shot relay);
// degrade-safe (degraded=True surfaces an explanatory answer, never freezes).
// The qa_model picker lives in 设置 → 模型 → 查询助手模型 (NOT duplicated here)
// exactly as the old tab — only the per-query 查询范围 scope selector is inline.

// Query scope — restricts which agent's domain the QA reasons about. Mirrors the
// old gr.Radio (rendered inline as buttons, never a popup — the dropdown overlay
// tripped a Svelte cascade in real Edge/Chrome: "切换查询范围后就卡死").
const SCOPES: { value: string; label: string }[] = [
  { value: "all", label: "全部" },
  { value: "literature", label: "文献 LIT" },
  { value: "experiment_design", label: "实验 XD" },
  { value: "instrument_control", label: "仪器 IC" },
  { value: "data_processing", label: "数据 DP" },
  { value: "paper_writing", label: "写作 PW" },
  { value: "paper_review", label: "审稿 PR" },
  { value: "buffer_summarizer", label: "视觉 BUF" },
];

type Turn = { role: "user" | "assistant"; content: string };

// Module-scoped store: the Q&A history used to live in component useState, so
// ANY tab switch unmounted the panel and wiped it ("查询助手的历史会自动清空",
// ). Hoisted like runTaskStore — history survives
// navigation for the whole SPA session; 清空 clears it explicitly.
const useQaHistoryStore = create<{
  history: Turn[];
  push: (...turns: Turn[]) => void;
  clear: () => void;
}>((set) => ({
  history: [],
  push: (...turns) => set((s) => ({ history: [...s.history, ...turns] })),
  clear: () => set(() => ({ history: [] })),
}));

export function QaPanel() {
  const [scope, setScope] = useState<string>("all");
  const [input, setInput] = useDraft("qa-input");
  const history = useQaHistoryStore((s) => s.history);
  const pushHistory = useQaHistoryStore((s) => s.push);
  const clearHistory = useQaHistoryStore((s) => s.clear);

  // Wired to POST /api/qa (single-turn QuickAsk). Degrade-safe: when the live
  // core/LLM is absent the endpoint returns degraded=true + an explanatory
  // answer — the page never freezes (the user's hard rule).
  const askMut = useMutation({
    mutationFn: async (vars: { question: string; scope: string }) => {
      const { data, error } = await api.POST("/api/qa", {
        body: { question: vars.question, scope: vars.scope },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data, vars) => {
      const answer = (data?.answer ?? "（无回答）") +
        (data?.degraded ? "\n\n_（降级：查询助手未接入实时内核或缺少模型密钥。）_" : "");
      pushHistory(
        { role: "user", content: vars.question },
        { role: "assistant", content: answer },
      );
      setInput("");
    },
    onError: (err, vars) => {
      pushHistory(
        { role: "user", content: vars.question },
        { role: "assistant", content: `_查询失败：${String(err)}_` },
      );
    },
  });

  function ask() {
    const text = input.trim();
    if (!text || askMut.isPending) return;
    askMut.mutate({ question: text, scope });
  }

  function clear() {
    clearHistory();
  }

  return (
    <div className="space-y-3">
      {/* header — reproduces the old gr.Markdown intro */}
      <Card>
        <p className="text-sm leading-relaxed text-mast-text">
          <strong>查询助手</strong> — 单次对话、只读。可以在主对话运行任务时并行查询仪器状态、文献、
          技能描述等信息；不会修改仪器、不写实验日志、不保留历史。
        </p>
        <p className="mt-2 text-xs text-mast-muted">
          模型在「设置 → 模型 → 查询助手模型」中调整。
        </p>
      </Card>

      {/* query scope — inline buttons, no popup (Radio parity) */}
      <Card>
        <div className="mb-2 text-sm text-mast-muted">查询范围</div>
        <div className="flex flex-wrap gap-1.5">
          {SCOPES.map((s) => (
            <button
              key={s.value}
              type="button"
              onClick={() => setScope(s.value)}
              className={
                "rounded-md border px-3 py-1.5 text-sm transition-colors " +
                (scope === s.value
                  ? "border-mast-accent bg-mast-accent/15 text-mast-accent"
                  : "border-mast-border text-mast-muted hover:text-mast-text")
              }
            >
              {s.label}
            </button>
          ))}
        </div>
      </Card>

      {/* input row — FIRST so it stays above the fold (old layout note) */}
      <Card>
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask();
              }
            }}
            placeholder="例：查看当前偏压；Au(111) Kondo 相关文献推荐"
            rows={2}
            className="flex-1 resize-none rounded-md border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent"
          />
          <Button variant="primary" onClick={ask} disabled={!input.trim() || askMut.isPending}>
            {askMut.isPending ? "查询中…" : "提问"}
          </Button>
          <Button variant="default" onClick={clear} disabled={history.length === 0}>
            清空
          </Button>
        </div>
        <p className="mt-2 text-xs text-mast-muted">
          {askMut.isPending ? "查询中…" : "就绪"}
        </p>
      </Card>

      {/* answer area — markdown-ish transcript */}
      <Card className="min-h-[200px]">
        {history.length === 0 ? (
          <EmptyNote label="输入问题并点「提问」开始查询。" />
        ) : (
          <div className="space-y-4">
            {history.map((t, i) => (
              <div key={i}>
                <div className="mb-1">
                  <Badge tone={t.role === "user" ? undefined : "INFO"}>
                    {t.role === "user" ? "你" : "查询助手"}
                  </Badge>
                </div>
                <div className="whitespace-pre-wrap break-words text-sm leading-relaxed text-mast-text">
                  {t.content}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
