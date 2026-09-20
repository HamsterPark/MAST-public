import type { ReactNode } from "react";

// Shared agent registry for the Agents inspector parity rebuild.
// Mirrors the old agents-ui.jsx 7-agent roster + supervisor singleton.
// Colors / Chinese labels / roles reproduced from the old blueprint so the
// topology, chips and cards read identically. Live data (model/thinking/
// active/threads) always comes from /api/agents/snapshot and overrides any
// static placeholder here.

export const SUP_ID = "_supervisor";

// Stable per-agent palette key (drives the --mast-ag-* token + class map).
// BUF (视觉摘要) is a side channel with no dedicated hue → falls back to sup.
export type AgentColorKey = "sup" | "rd" | "lit" | "xd" | "ic" | "dp" | "pw" | "pr";

export type AgentDef = {
  id: string;
  short: string;
  cn: string;
  /** Resolved hex (live token value) — kept for SVG fills that need a raw color. */
  color: string;
  /** Palette key → --mast-ag-{key} token + static Tailwind class map. */
  ck: AgentColorKey;
  role: string;
};

// `color` mirrors the dark-theme --mast-ag-{ck} token (raw hex for SVG fills);
// `ck` is the canonical palette key. The two are kept in lock-step with the
// design tokens in src/index.css so chips/nodes read identically in both themes.
export const AGENTS: AgentDef[] = [
  { id: "research_director", short: "RD", cn: "科研策划", color: "#4f46e5", ck: "rd", role: "科研纲领：假设 / 目标 / 谱系 · 委托 XD 起草方案" },
  { id: "literature", short: "LIT", cn: "文献", color: "#2563eb", ck: "lit", role: "搜索 36 k 论文库 · 抽取实验协议" },
  { id: "experiment_design", short: "XD", cn: "实验设计", color: "#16a34a", ck: "xd", role: "研究问题 → ExperimentPlan JSON" },
  { id: "instrument_control", short: "IC", cn: "仪器控制", color: "#0d9488", ck: "ic", role: "执行 Nanonis · 拥有 249 个 skill (L0–L4)" },
  { id: "data_processing", short: "DP", cn: "数据处理", color: "#7c3aed", ck: "dp", role: "解析 .sxm/.dat/.3ds → 数值指标" },
  { id: "paper_writing", short: "PW", cn: "论文写作", color: "#d97706", ck: "pw", role: "起草 intro/methods/results/discussion" },
  { id: "paper_review", short: "PR", cn: "论文审稿", color: "#e11d48", ck: "pr", role: "三轴 rubric 同行评审" },
  { id: "buffer_summarizer", short: "BUF", cn: "视觉摘要", color: "#475569", ck: "sup", role: "DINOv3 视觉输出 → 中文一句话摘要 · 侧通道" },
];

export const SUPERVISOR: AgentDef = {
  id: SUP_ID,
  short: "SUP",
  cn: "编排与协调",
  color: "#475569",
  ck: "sup",
  role: "任务分解 · 工作流编排 · 异常处理 · 进度监控 · 人机交互",
};

export const AGENT_BY: Record<string, AgentDef> = Object.fromEntries(
  [...AGENTS, SUPERVISOR].map((a) => [a.id, a]),
);

// Pipeline order shown in the topology (skips BUF — side channel).
export const PIPELINE = [
  // RD 在最前：它回答「为什么做」，是这条链的上游而不是并列的第七个（2026-08-21）。
  "research_director",
  "literature",
  "experiment_design",
  "instrument_control",
  "data_processing",
  "paper_writing",
  "paper_review",
];

// Three-phase grouping reproduced from the old PHASES blueprint.
//
// `agents` 从定长二元组放宽成 string[]（2026-08-21）：加 research_director 时
// 那个 `[string, string]` 会让「一个阶段有三个成员」在类型层就写不出来 —— 而
// 分组本来就不该被成员数目锁死。定长在这里没有换来任何保护：越界索引不是这份
// 数据的失败模式，漏掉一个 agent 才是（那由 test_agents_topology 的覆盖断言管）。
export const PHASES: {
  id: string;
  cn: string;
  en: string;
  color: string;
  agents: string[];
}[] = [
  // Phase hue tracks its lead agent's --mast-ag-* token (research=LIT, execute=IC,
  // write=PW) so the bands recolor with the theme alongside the agent cards.
  { id: "research", cn: "调研", en: "research", color: "var(--mast-ag-lit)", agents: ["research_director", "literature", "experiment_design"] },
  { id: "execute", cn: "实验", en: "execute", color: "var(--mast-ag-ic)", agents: ["instrument_control", "data_processing"] },
  { id: "write", cn: "撰写", en: "write", color: "var(--mast-ag-pw)", agents: ["paper_writing", "paper_review"] },
];

export function agentDef(id: string): AgentDef {
  return (
    AGENT_BY[id] ?? { id, short: id.slice(0, 3).toUpperCase(), cn: id, color: "#64748b", ck: "sup", role: "" }
  );
}

// ── Per-agent color identity ────────────────────────────────────────────────
// Tailwind purges class names it can't see as complete string literals, so we
// MUST list every per-agent class statically (never build by string concat).
// Each entry maps a palette key → the matching --mast-ag-{ck} utilities.
type AgentClassSet = { text: string; bg: string; border: string; borderL: string; ring: string };

const AG_CLASSES: Record<AgentColorKey, AgentClassSet> = {
  rd: {
    text: "text-mast-ag-rd",
    bg: "bg-mast-ag-rd",
    border: "border-mast-ag-rd",
    borderL: "border-l-[3px] border-l-mast-ag-rd",
    ring: "ring-mast-ag-rd",
  },
  sup: {
    text: "text-mast-ag-sup",
    bg: "bg-mast-ag-sup",
    border: "border-mast-ag-sup",
    borderL: "border-l-[3px] border-l-mast-ag-sup",
    ring: "ring-mast-ag-sup",
  },
  lit: {
    text: "text-mast-ag-lit",
    bg: "bg-mast-ag-lit",
    border: "border-mast-ag-lit",
    borderL: "border-l-[3px] border-l-mast-ag-lit",
    ring: "ring-mast-ag-lit",
  },
  xd: {
    text: "text-mast-ag-xd",
    bg: "bg-mast-ag-xd",
    border: "border-mast-ag-xd",
    borderL: "border-l-[3px] border-l-mast-ag-xd",
    ring: "ring-mast-ag-xd",
  },
  ic: {
    text: "text-mast-ag-ic",
    bg: "bg-mast-ag-ic",
    border: "border-mast-ag-ic",
    borderL: "border-l-[3px] border-l-mast-ag-ic",
    ring: "ring-mast-ag-ic",
  },
  dp: {
    text: "text-mast-ag-dp",
    bg: "bg-mast-ag-dp",
    border: "border-mast-ag-dp",
    borderL: "border-l-[3px] border-l-mast-ag-dp",
    ring: "ring-mast-ag-dp",
  },
  pw: {
    text: "text-mast-ag-pw",
    bg: "bg-mast-ag-pw",
    border: "border-mast-ag-pw",
    borderL: "border-l-[3px] border-l-mast-ag-pw",
    ring: "ring-mast-ag-pw",
  },
  pr: {
    text: "text-mast-ag-pr",
    bg: "bg-mast-ag-pr",
    border: "border-mast-ag-pr",
    borderL: "border-l-[3px] border-l-mast-ag-pr",
    ring: "ring-mast-ag-pr",
  },
};

/** Static Tailwind classes (text / bg / border / left-border / ring) for an
 *  agent's color identity. Always a complete literal so Tailwind keeps them. */
export function agentClasses(id: string): AgentClassSet {
  return AG_CLASSES[agentDef(id).ck];
}

/** The CSS custom-property reference for an agent's hue, e.g. "var(--mast-ag-ic)".
 *  Use in inline style when a static class can't express the need (SVG fill,
 *  computed gradients). Never feed this into a className. */
export function agentColorVar(id: string): string {
  return `var(--mast-ag-${agentDef(id).ck})`;
}

// Inline-SVG icon set (lucide path data) — keyed to the canvas agent palette:
// sup=network, lit=book, xd=beaker, ic=wrench, dp=line-chart, pw=pen-line,
// pr=clipboard-check. Rendered with currentColor so the parent controls hue.
const AG_ICON_PATHS: Record<AgentColorKey, ReactNode> = {
  // 一个假设分叉成两条待验证的路 —— RD 的产出正是「可证伪的纲领 + 委托」
  rd: (
    <>
      <circle cx="12" cy="4" r="2.5" />
      <path d="M12 6.5v4M12 10.5l-5 4M12 10.5l5 4" />
      <circle cx="6" cy="17" r="2.5" />
      <circle cx="18" cy="17" r="2.5" />
    </>
  ),
  // network
  sup: (
    <>
      <rect x="16" y="16" width="6" height="6" rx="1" />
      <rect x="2" y="16" width="6" height="6" rx="1" />
      <rect x="9" y="2" width="6" height="6" rx="1" />
      <path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3" />
      <path d="M12 12V8" />
    </>
  ),
  // book
  lit: (
    <>
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    </>
  ),
  // beaker
  xd: (
    <>
      <path d="M4.5 3h15" />
      <path d="M6 3v16a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V3" />
      <path d="M6 14h12" />
    </>
  ),
  // wrench
  ic: (
    <>
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </>
  ),
  // line-chart
  dp: (
    <>
      <path d="M3 3v16a2 2 0 0 0 2 2h16" />
      <path d="m19 9-5 5-4-4-3 3" />
    </>
  ),
  // pen-line
  pw: (
    <>
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </>
  ),
  // clipboard-check
  pr: (
    <>
      <rect width="8" height="4" x="8" y="2" rx="1" ry="1" />
      <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
      <path d="m9 14 2 2 4-4" />
    </>
  ),
};

/** Agent glyph (lucide-equivalent inline SVG) drawn in `currentColor`. */
export function AgentIcon({ id, size = 14, className }: { id: string; size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {AG_ICON_PATHS[agentDef(id).ck]}
    </svg>
  );
}

export function agentLabel(id: string): string {
  return AGENT_BY[id]?.cn ?? id;
}

// Thinking levels offered by the per-agent picker (mirrors THINKING_LEVELS).
export const THINKING_LEVELS = ["off", "low", "medium", "high"];

// Safety-level → Badge tone (the ui.tsx Badge knows AUTO/INFO/WARN/DANGEROUS).
export function safetyTone(sl?: string | null): string {
  const s = (sl ?? "AUTO").toUpperCase();
  if (s.includes("DANGER")) return "DANGEROUS";
  if (s.includes("CONFIRM") || s.includes("WARN")) return "WARN";
  if (s.includes("INFO")) return "INFO";
  return "AUTO";
}

/** Colored avatar: a filled circle (rounded square for SUP) with the short id. */
export function Avatar({
  id,
  size = 28,
  active = false,
}: {
  id: string;
  size?: number;
  active?: boolean;
}) {
  const a = agentDef(id);
  const isSup = id === SUP_ID;
  const hue = agentColorVar(id);
  return (
    <span
      className="inline-flex shrink-0 items-center justify-center font-mono font-bold text-white"
      style={{
        width: size,
        height: size,
        background: hue,
        borderRadius: isSup ? size * 0.28 : "50%",
        fontSize: size * 0.36,
        boxShadow: active ? `0 0 0 2px color-mix(in srgb, ${hue} 40%, transparent)` : undefined,
      }}
    >
      {a.short}
    </span>
  );
}
