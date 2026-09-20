import { Badge, Card } from "@/components/ui";

/** One domain-agent card: human label + monogram, effective model alias, and an
 *  honest effective-thinking badge. Pure presentational; data comes from the
 *  /api/agents/models seam. */

export type AgentMeta = {
  label: string;
  monogram: string;
  blurb: string;
};

// Canonical roster (7 agents) — mirrors orchestrator + the inspector roster.
export const AGENT_META: Record<string, AgentMeta> = {
  instrument_control: { label: "仪器控制", monogram: "IC", blurb: "Nanonis 硬件技能执行" },
  literature: { label: "文献调研", monogram: "LIT", blurb: "检索 / 全文 / 知识库" },
  experiment_design: { label: "实验设计", monogram: "EXP", blurb: "方案规划与技能编排" },
  data_processing: { label: "数据处理", monogram: "DP", blurb: "图像分析 / 拼图 / FFT" },
  paper_writing: { label: "论文撰写", monogram: "PW", blurb: "结果整理与成稿" },
  paper_review: { label: "论文评审", monogram: "PR", blurb: "稿件审阅与反馈" },
  buffer_summarizer: { label: "缓冲摘要", monogram: "BUF", blurb: "尖端状态/分割摘要" },
};

export function agentLabel(agentId: string): string {
  return AGENT_META[agentId]?.label ?? agentId;
}

/** Effective-thinking → tone + label. The string is whatever the model actually
 *  runs at (e.g. "high (固定)", "off", null when registry omits it). */
function thinkingTone(thinking?: string | null): { tone: string; text: string } {
  const t = (thinking ?? "").toLowerCase();
  if (!thinking) return { tone: "default", text: "未知" };
  if (t.includes("off") || t === "none" || t.includes("无")) return { tone: "INFO", text: thinking };
  if (t.includes("固定") || t.includes("fixed")) return { tone: "WARN", text: thinking };
  return { tone: "AUTO", text: thinking };
}

export function AgentCard({
  agentId,
  model,
  thinking,
  toolCount,
  active,
  onSelect,
}: {
  agentId: string;
  model?: string;
  thinking?: string | null;
  toolCount?: number;
  active?: boolean;
  onSelect?: () => void;
}) {
  const meta = AGENT_META[agentId];
  const th = thinkingTone(thinking);
  return (
    <Card
      className={
        "cursor-pointer transition-colors hover:border-mast-accent/60 " +
        (active ? "border-mast-accent ring-1 ring-mast-accent/40" : "")
      }
    >
      <button type="button" onClick={onSelect} className="w-full text-left">
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded bg-mast-accent/15 text-xs font-bold text-mast-accent">
            {meta?.monogram ?? agentId.slice(0, 2).toUpperCase()}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-medium">{meta?.label ?? agentId}</span>
              {typeof toolCount === "number" && (
                <span className="shrink-0 text-xs text-mast-muted">{toolCount} 工具</span>
              )}
            </div>
            <p className="mt-0.5 truncate text-xs text-mast-muted">{meta?.blurb ?? agentId}</p>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="rounded bg-mast-bg px-2 py-0.5 font-mono text-xs text-mast-text">
            {model ?? "—"}
          </span>
          <span className="text-xs text-mast-muted">思考</span>
          <Badge tone={th.tone}>{th.text}</Badge>
        </div>
      </button>
    </Card>
  );
}
