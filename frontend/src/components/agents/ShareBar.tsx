import { agentColorVar } from "@/components/agents/registry";
import type { Share } from "@/lib/contextInjection";
import { fmtChars } from "@/lib/contextInjection";

// 一条横向堆叠条 + 图例。**不引图表库**：全仓没有 recharts/d3，为一条百分比条
// 引一个是纯负债；而且这里要的是「一眼看出谁最大」，不是可交互图表。
//
// 颜色只用已存在的设计 token（`designTokens.test.ts` 会核每个 var(--…) 真的
// 定义过）。静态提示用该 agent 自己的身份色，其余走一组固定的循环色 —— 循环色
// 是刻意的：块的数量会随版本变，写死一块一色的表迟早对不上。

const BLOCK_COLORS = [
  "var(--mast-info)",
  "var(--mast-accent)",
  "var(--mast-auto)",
  "var(--mast-warn)",
];

function colorOf(share: Share, agentId: string, blockIndex: number): string {
  if (share.key === "static") return agentColorVar(agentId);
  if (share.key === "history") return "var(--mast-muted)";
  if (share.key === "tools") return "var(--mast-border-strong)";
  if (share.key === "unattributed") return "var(--mast-faint)";
  return BLOCK_COLORS[blockIndex % BLOCK_COLORS.length] ?? "var(--mast-info)";
}

export function ShareBar({
  shares,
  agentId,
  caption,
}: {
  shares: Share[];
  agentId: string;
  caption?: string;
}) {
  if (shares.length === 0) return null;
  // 屏幕阅读器拿到的是同一份信息，而不是「一张图」。
  const summary = shares.map((s) => `${s.label} ${s.pct}%`).join(" · ");
  let blockIndex = -1;
  const colored = shares.map((s) => {
    if (s.key === "block") blockIndex += 1;
    return { share: s, color: colorOf(s, agentId, blockIndex) };
  });

  return (
    <div>
      <div
        role="img"
        aria-label={summary}
        className="flex h-3 w-full overflow-hidden rounded-mast-badge border border-mast-border"
      >
        {colored.map(({ share, color }, i) => (
          <span
            key={`${share.key}-${share.id ?? i}`}
            title={`${share.label} — ${fmtChars(share.chars)}（${share.pct}%）`}
            style={{
              width: `${share.pct}%`,
              background: color,
              // 估算的那一段画成斜纹：一个和实测长得一样的估算值会被当成实测读。
              backgroundImage: share.estimated
                ? "repeating-linear-gradient(45deg, rgba(255,255,255,.35) 0 3px, transparent 3px 6px)"
                : undefined,
            }}
          />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-mast-muted">
        {colored.map(({ share, color }, i) => (
          <li key={`lg-${share.key}-${share.id ?? i}`} className="flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm border border-mast-border"
              style={{ background: color }}
            />
            <span className="text-mast-text">{share.label}</span>
            <span>{fmtChars(share.chars)} · {share.pct}%</span>
          </li>
        ))}
      </ul>
      {caption && <p className="mt-1.5 text-xs text-mast-faint">{caption}</p>}
    </div>
  );
}
