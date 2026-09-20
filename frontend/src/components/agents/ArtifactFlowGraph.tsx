import { useState } from "react";
import { PIPELINE, agentColorVar, agentDef } from "./registry";
import type { components } from "@/api/schema";

// ════════════════════════════════════════════════════════════════════════════
// 数据流 · ARTIFACT FLOW
//
// Redrawn 2026-07-11 ("对象总览-访问拓扑不科学").
//
// The old picture was a static bipartite blob derived from a one-to-one
// {agent: artifact} table: every artifact had exactly ONE writer and every other
// agent was drawn as a reader. Both halves were fiction — figures are written by
// data_processing AND by paper_writing, the shared memory is written by all six,
// and paper_review never opens a scan file.
//
// This draws the REAL directed flow instead:
//   · production is the skeleton  — solid, in the writing agent's own hue;
//   · consumption is the detail   — dashed and quiet until you ask for it;
//   · multi-writer artifacts show TWO+ incoming solid edges (the case the old
//     model could not express at all) and are called out explicitly;
//   · the two artifacts that everyone touches are drawn as what they ARE — a
//     shared backplane and a side rail — rather than as nodes with 12 edges
//     each, which is what made the old graph unreadable.
// ════════════════════════════════════════════════════════════════════════════

type Perm = components["schemas"]["ArtifactPermission"];

const MONO = "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace";
const SANS = "ui-sans-serif, system-ui, 'Noto Sans SC', 'PingFang SC', sans-serif";

const W = 1120;
const H = 394;

const AG_W = 132;
const AG_H = 54;
const AG_GAP = 20;
const AG_X0 = 200;
const AG_Y = 22;
const agX = (i: number) => AG_X0 + i * (AG_W + AG_GAP);
const agCx = (i: number) => agX(i) + AG_W / 2;
const AG_BOTTOM = AG_Y + AG_H;

const AR_W = 128;
const AR_H = 74;
const AR_Y = 196;
const AR_MIN_X = 24;
const AR_MAX_X = W - AR_W - 24;

const BUS_Y = 306; // shared backplane (memory)
const RAIL_Y = 352; // side rail (vision buffer)

const soft = (v: string, pct: number) => `color-mix(in srgb, ${v} ${pct}%, transparent)`;

// Artifacts everybody touches are structurally a BUS, not a node.
const BUS_ID = "memory";
const RAIL_ID = "vision_buffer";

const KIND_GLYPH: Record<string, string> = {
  file: "▤",
  db: "▦",
  state: "◈",
  index: "▥",
};

export function ArtifactFlowGraph({
  permissions,
  selectedId,
  onSelect,
  onOpen,
}: {
  permissions: Perm[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
  /** Open the artifact editor (double-click / click-through). */
  onOpen?: (id: string) => void;
}) {
  // focus: an agent id or an artifact id — dims everything not on its edges
  const [focus, setFocus] = useState<string | null>(null);

  const bus = permissions.find((p) => p.artifact_id === BUS_ID);
  const rail = permissions.find((p) => p.artifact_id === RAIL_ID);
  const nodes = permissions.filter(
    (p) => p.artifact_id !== BUS_ID && p.artifact_id !== RAIL_ID,
  );

  // Order the artifacts by their PRODUCER, not by a centre-of-mass of everyone
  // who touches them: this is a production flow, so the row should read left→
  // right in the order things get made (文献 → 方案 → 扫描 → 分析 → 图表 → 稿件
  // → 评审). Sorting by centre-of-mass instead let a far-right *reader* drag an
  // early artifact rightwards — the literature index landed after the
  // experiment plan, and the write edges crossed each other for no reason.
  const idxOf = (a: string) => PIPELINE.indexOf(a);
  const desired = nodes.map((p) => {
    const writers = (p.writers ?? []).map(idxOf).filter((i) => i >= 0);
    const readers = (p.readers ?? []).map(idxOf).filter((i) => i >= 0);
    // rank: earliest producer first (a multi-writer artifact sits under the
    // LAST agent that writes it, since it isn't finished until then)
    const rank = writers.length
      ? Math.max(...writers) + Math.min(...writers) / 100
      : readers.length
        ? Math.min(...readers) - 0.5   // system-written: park it by its first reader
        : 99;
    // horizontal home = under the producer (mid-point for a multi-writer)
    const anchor = writers.length
      ? writers.reduce((s, i) => s + agCx(i), 0) / writers.length
      : readers.length
        ? agCx(Math.min(...readers))
        : W / 2;
    return { p, rank, want: anchor - AR_W / 2 };
  });
  desired.sort((a, b) => a.rank - b.rank || a.want - b.want);
  const placed: { p: Perm; x: number }[] = [];
  let cursor = AR_MIN_X;
  for (const d of desired) {
    const x = Math.max(cursor, Math.min(d.want, AR_MAX_X));
    placed.push({ p: d.p, x });
    cursor = x + AR_W + 12;
  }
  // if we ran past the right edge, push the whole row back left
  const overflow = cursor - 12 - (AR_MAX_X + AR_W);
  if (overflow > 0) {
    for (const q of placed) q.x = Math.max(AR_MIN_X, q.x - overflow);
  }
  const posOf = new Map(placed.map((q) => [q.p.artifact_id, q.x]));

  const dim = (onFocus: boolean) => (focus && !onFocus ? 0.05 : 1);

  return (
    <div className="overflow-x-auto rounded-xl border border-mast-border bg-mast-bg/60">
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        style={{ display: "block", minWidth: 880 }}
        role="img"
        aria-label="智能体与共享产物的数据流"
        onMouseLeave={() => setFocus(null)}
      >
        <defs>
          <pattern id="af-grid" width="22" height="22" patternUnits="userSpaceOnUse">
            <path d="M22 0 L0 0 0 22" fill="none" stroke="var(--mast-border)"
                  strokeWidth="0.5" opacity="0.45" />
          </pattern>
        </defs>
        <rect x={0} y={0} width={W} height={H} fill="url(#af-grid)" />

        {/* ── READ edges (consumption): quiet by default, lifted on focus ── */}
        {nodes.map((p) =>
          (p.readers ?? []).map((r) => {
            const ai = idxOf(r);
            const ax = posOf.get(p.artifact_id);
            if (ai < 0 || ax === undefined) return null;
            const on = focus === r || focus === p.artifact_id;
            const x1 = ax + AR_W / 2 + 22;
            const x2 = agCx(ai) + 20;
            const mid = (AR_Y + AG_BOTTOM) / 2;
            return (
              <g key={`r-${p.artifact_id}-${r}`} opacity={on ? 1 : dim(false) === 1 ? 0.2 : 0.05}>
                <path
                  d={`M ${x1} ${AR_Y} C ${x1} ${mid + 26}, ${x2} ${mid - 26}, ${x2} ${AG_BOTTOM + 2}`}
                  fill="none"
                  stroke="var(--mast-muted)"
                  strokeWidth={on ? 1.4 : 1}
                  strokeDasharray="3 4"
                />
                {/* arrow INTO the agent — consumption points at the consumer */}
                <path d={`M ${x2 - 3.2} ${AG_BOTTOM + 6} L ${x2 + 3.2} ${AG_BOTTOM + 6} L ${x2} ${AG_BOTTOM + 1.5} Z`}
                      fill="var(--mast-muted)" />
              </g>
            );
          }),
        )}

        {/* ── WRITE edges (production): the skeleton of the picture ─────── */}
        {nodes.map((p) =>
          (p.writers ?? []).map((w) => {
            const ai = idxOf(w);
            const ax = posOf.get(p.artifact_id);
            if (ai < 0 || ax === undefined) return null;
            const hue = agentColorVar(w);
            const on = !focus || focus === w || focus === p.artifact_id;
            const x1 = agCx(ai) - 20;
            const x2 = ax + AR_W / 2 - 22;
            const mid = (AR_Y + AG_BOTTOM) / 2;
            return (
              <g key={`w-${p.artifact_id}-${w}`} opacity={on ? 1 : 0.06}>
                <path
                  d={`M ${x1} ${AG_BOTTOM} C ${x1} ${mid - 22}, ${x2} ${mid + 22}, ${x2} ${AR_Y - 6}`}
                  fill="none"
                  stroke={hue}
                  strokeWidth={1.6}
                />
                {/* arrow INTO the artifact — production points at the product */}
                <path d={`M ${x2 - 3.6} ${AR_Y - 7} L ${x2 + 3.6} ${AR_Y - 7} L ${x2} ${AR_Y - 1} Z`}
                      fill={hue} />
              </g>
            );
          }),
        )}

        {/* ── agent modules (pipeline order) ───────────────────────────── */}
        {PIPELINE.map((id, i) => {
          const a = agentDef(id);
          const hue = agentColorVar(id);
          const on = !focus || focus === id;
          return (
            <g
              key={id}
              transform={`translate(${agX(i)} ${AG_Y})`}
              onMouseEnter={() => setFocus(id)}
              opacity={on ? 1 : 0.28}
              style={{ cursor: "pointer" }}
            >
              <title>{`${a.cn} — 悬停查看其读写连线`}</title>
              <rect x={0} y={0} width={AG_W} height={AG_H} rx={7}
                    className="fill-mast-panel" stroke={soft(hue, 62)} strokeWidth={1} />
              <rect x={0} y={0} width={AG_W} height={3} rx={1.5} fill={hue} />
              <text x={11} y={26} fontFamily={MONO} fontSize={14} fontWeight={700}
                    letterSpacing="1.3" fill={hue}>
                {a.short}
              </text>
              <text x={11} y={42} fontFamily={SANS} fontSize={10} className="fill-mast-muted">
                {a.cn}
              </text>
            </g>
          );
        })}

        {/* ── artifact nodes ──────────────────────────────────────────── */}
        {placed.map(({ p, x }) => {
          const on = !focus || focus === p.artifact_id ||
            (p.writers ?? []).includes(focus) || (p.readers ?? []).includes(focus);
          const sel = selectedId === p.artifact_id;
          const multi = !!p.multi_writer;
          return (
            <g
              key={p.artifact_id}
              transform={`translate(${x} ${AR_Y})`}
              opacity={on ? 1 : 0.24}
              onMouseEnter={() => setFocus(p.artifact_id)}
              onClick={() => onSelect?.(p.artifact_id)}
              onDoubleClick={() => onOpen?.(p.artifact_id)}
              style={{ cursor: "pointer" }}
            >
              <title>{`${p.label || p.artifact_id}\n存储：${p.store}\n写：${(p.writers ?? []).map((w) => agentDef(w).cn).join("、") || "系统"}\n读：${(p.readers ?? []).map((r) => agentDef(r).cn).join("、") || "—"}`}</title>
              {sel && (
                <rect x={-4} y={-4} width={AR_W + 8} height={AR_H + 8} rx={9} fill="none"
                      stroke="var(--mast-accent)" strokeWidth={1.6} opacity={0.7} />
              )}
              <rect x={0} y={0} width={AR_W} height={AR_H} rx={7}
                    className="fill-mast-panel"
                    stroke={multi ? "var(--mast-accent)" : "var(--mast-border)"}
                    strokeWidth={multi ? 1.4 : 1} />
              {/* a multi-writer artifact gets a DOUBLED left rule — the visual
                  tell that more than one producer legitimately writes it */}
              {multi ? (
                <>
                  <rect x={0} y={0} width={2.5} height={AR_H} rx={1.2} fill="var(--mast-accent)" />
                  <rect x={4} y={0} width={1.5} height={AR_H} rx={0.8} fill="var(--mast-accent)"
                        opacity={0.55} />
                </>
              ) : (
                <rect x={0} y={0} width={2.5} height={AR_H} rx={1.2}
                      fill={soft("var(--mast-muted)", 55)} />
              )}

              <text x={13} y={19} fontFamily={MONO} fontSize={11}
                    className="fill-mast-muted">
                {KIND_GLYPH[p.kind] ?? "◈"}
              </text>
              <text x={28} y={19} fontFamily={SANS} fontSize={11.5} fontWeight={600}
                    className="fill-mast-text">
                {p.label || p.artifact_id}
              </text>

              {/* the writers, named — production is the point of this graph */}
              <g transform="translate(13 30)">
                {(p.writers ?? []).slice(0, 3).map((w, k) => (
                  <g key={w} transform={`translate(${k * 30} 0)`}>
                    <rect x={0} y={0} width={26} height={13} rx={3}
                          fill={soft(agentColorVar(w), 16)}
                          stroke={soft(agentColorVar(w), 60)} strokeWidth={0.8} />
                    <text x={13} y={9.6} textAnchor="middle" fontFamily={MONO} fontSize={8}
                          fontWeight={700} fill={agentColorVar(w)}>
                      {agentDef(w).short}
                    </text>
                  </g>
                ))}
                {(p.writers ?? []).length === 0 && (
                  <text x={0} y={9.6} fontFamily={MONO} fontSize={8}
                        className="fill-mast-muted">
                    系统写入
                  </text>
                )}
              </g>
              {multi && (
                <text x={AR_W - 9} y={19} textAnchor="end" fontFamily={MONO} fontSize={8}
                      fontWeight={700} fill="var(--mast-accent)">
                  多写
                </text>
              )}

              {/* where it physically lives — a claim you can check on disk */}
              <text x={13} y={AR_H - 11} fontFamily={MONO} fontSize={7.5}
                    className="fill-mast-muted" opacity={0.85}>
                {truncate(p.store || "", 22)}
              </text>
            </g>
          );
        })}

        {/* ── shared backplane: the artifact EVERY agent reads and writes ── */}
        {bus && (
          <g
            onMouseEnter={() => setFocus(bus.artifact_id)}
            onClick={() => onSelect?.(bus.artifact_id)}
            opacity={!focus || focus === bus.artifact_id || PIPELINE.includes(focus) ? 1 : 0.3}
            style={{ cursor: "pointer" }}
          >
            <title>{`${bus.label} — 六个智能体全员读写（共享总线）\n存储：${bus.store}`}</title>
            {/* drawn as a BUS because that is what it is — a node with 12 edges
                would have been unreadable and would have said nothing true */}
            <rect x={24} y={BUS_Y} width={W - 48} height={26} rx={5}
                  fill={soft("var(--mast-accent)", 7)}
                  stroke={soft("var(--mast-accent)", 50)} strokeWidth={1} />
            {PIPELINE.map((_, i) => (
              <line key={i} x1={agCx(i)} y1={BUS_Y} x2={agCx(i)} y2={BUS_Y - 5}
                    stroke={soft("var(--mast-accent)", 45)} strokeWidth={1} />
            ))}
            <text x={36} y={BUS_Y + 17} fontFamily={SANS} fontSize={11} fontWeight={600}
                  fill="var(--mast-accent)">
              {bus.label}
            </text>
            <text x={36 + 90} y={BUS_Y + 17} fontFamily={MONO} fontSize={8.5}
                  fill={soft("var(--mast-accent)", 80)}>
              全员读写 · {bus.store}
            </text>
            <text x={W - 36} y={BUS_Y + 17} textAnchor="end" fontFamily={MONO} fontSize={8}
                  fontWeight={700} fill="var(--mast-accent)">
              多写 ×{(bus.writers ?? []).length}
            </text>
          </g>
        )}

        {/* ── side rail: written by the vision subsystem, agents only read ── */}
        {rail && (
          <g
            onMouseEnter={() => setFocus(rail.artifact_id)}
            onClick={() => onSelect?.(rail.artifact_id)}
            opacity={!focus || focus === rail.artifact_id ? 1 : 0.3}
            style={{ cursor: "pointer" }}
          >
            <title>{`${rail.label} — 视觉子系统写入，智能体只读（"agent 从不写缓冲区"不变式）`}</title>
            <line x1={24} y1={RAIL_Y} x2={W - 24} y2={RAIL_Y}
                  stroke="var(--mast-border)" strokeWidth={1} strokeDasharray="4 4" />
            <text x={36} y={RAIL_Y + 14} fontFamily={SANS} fontSize={10}
                  className="fill-mast-muted">
              {rail.label}
            </text>
            <text x={36 + 130} y={RAIL_Y + 14} fontFamily={MONO} fontSize={8}
                  className="fill-mast-muted" opacity={0.8}>
              系统写入 · 智能体只读（agent 从不写缓冲区）
            </text>
          </g>
        )}

        {/* ── legend ──────────────────────────────────────────────────── */}
        <g transform={`translate(24 ${AG_Y + 16})`} fontFamily={MONO} fontSize={9}>
          <line x1={0} y1={-4} x2={18} y2={-4} stroke="var(--mast-ag-ic)" strokeWidth={1.6} />
          <text x={24} y={-1} className="fill-mast-muted">写入（生产）</text>
          <line x1={0} y1={12} x2={18} y2={12} stroke="var(--mast-muted)" strokeWidth={1}
                strokeDasharray="3 4" />
          <text x={24} y={15} className="fill-mast-muted">读取（消费）</text>
          <text x={0} y={31} className="fill-mast-muted" opacity={0.75}>悬停可聚焦</text>
        </g>
      </svg>
    </div>
  );
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}
