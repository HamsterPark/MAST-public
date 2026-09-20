import { PIPELINE, SUP_ID, agentColorVar, agentDef } from "./registry";

// ════════════════════════════════════════════════════════════════════════════
// 调度拓扑 · DISPATCH BUS
//
// Redrawn 2026-07-11. The old graph was an org-chart of phase bands — it could
// not say the one thing that now matters most: SEVERAL AGENTS RUN AT ONCE.
//
// The metaphor here is the instrument's own signal chain, which is what this
// system actually is: the supervisor is a dispatch BUS, each agent hangs off it
// on a trace, and dispatching an agent ENERGISES its trace. An idle system is a
// quiet dark blueprint; a running one lights up exactly the lanes doing work —
// so a 3-way fan-out is legible at a glance instead of hidden behind one
// "active" dot. Colour is spent ONLY on live current: everything else is
// hairline graphite.
// ════════════════════════════════════════════════════════════════════════════

export type AgentLive = {
  model?: string | null;
  thinking?: string | null;
  active?: boolean;
  held?: boolean;
  threads?: number;
};

const MONO = "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace";
const SANS = "ui-sans-serif, system-ui, 'Noto Sans SC', 'PingFang SC', sans-serif";

const H = 286;

const SUP_BOX = { x: 16, y: 30, w: 132, h: 62 };
const BUS_Y = SUP_BOX.y + SUP_BOX.h / 2; // 61 — the rail the supervisor drives
const BUS_X0 = SUP_BOX.x + SUP_BOX.w;

const MOD_W = 132;
const MOD_H = 92;
const MOD_GAP = 20;
const MOD_Y = 158;
const ROW_X = 200;

const modX = (i: number) => ROW_X + i * (MOD_W + MOD_GAP);
const modCx = (i: number) => modX(i) + MOD_W / 2;

// The canvas width is DERIVED from how many modules there are — never a magic
// number. It used to be a literal `1120`, which fitted exactly six; adding
// research_director (2026-08-21) pushed paper_review's right edge to 1244 and
// the seventh module was **silently clipped off the viewBox**. Nothing errored,
// nothing warned: an SVG just doesn't paint what falls outside its coordinate
// box, so the only symptom was an operator saying「拓扑图有一部分被遮住了」.
//
// A hard-coded canvas makes every future agent a layout regression, and the
// registry is the one place people DO remember to edit. Pinned by
// frontend/test/topologyCanvas.test.ts.
const RIGHT_PAD = 20;
const W = Math.max(1120, modX(PIPELINE.length - 1) + MOD_W + RIGHT_PAD);
const BUS_X1 = W - RIGHT_PAD;

// color-mix lets a CSS var carry an alpha (SVG can't concat opacity onto a var)
const soft = (v: string, pct: number) => `color-mix(in srgb, ${v} ${pct}%, transparent)`;

const SUP_HUE = agentColorVar(SUP_ID);

export function TopologyGraph({
  selected,
  live,
  onPick,
  onToggleHold,
}: {
  selected: string | null;
  live: Record<string, AgentLive>;
  onPick: (id: string) => void;
  /** Optional 暂停/恢复 toggle (LIVE-only; parent owns the mutation + toast). */
  onToggleHold?: (id: string, held: boolean) => void;
}) {
  const activeIdx = PIPELINE.map((id, i) => (live[id]?.active ? i : -1)).filter((i) => i >= 0);
  const parallel = activeIdx.length > 1;
  const supLive = live[SUP_ID] ?? {};

  return (
    <div className="overflow-x-auto rounded-xl border border-mast-border bg-mast-bg/60">
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        style={{ display: "block", minWidth: 860 }}
        role="img"
        aria-label="多智能体调度拓扑"
      >
        <defs>
          {/* graph-paper ground — the blueprint the traces are etched on */}
          <pattern id="tp-grid" width="22" height="22" patternUnits="userSpaceOnUse">
            <path
              d="M22 0 L0 0 0 22"
              fill="none"
              stroke="var(--mast-border)"
              strokeWidth="0.5"
              opacity="0.45"
            />
          </pattern>
          {/* live current gets a soft bloom; idle traces get nothing */}
          <filter id="tp-bloom" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="3.2" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <rect x={0} y={0} width={W} height={H} fill="url(#tp-grid)" />

        {/* ── the dispatch bus ─────────────────────────────────────────── */}
        {/* a doubled rail, like a power trace on a board */}
        <line x1={BUS_X0} y1={BUS_Y - 2.5} x2={BUS_X1} y2={BUS_Y - 2.5}
              stroke={soft(SUP_HUE, 40)} strokeWidth={1} />
        <line x1={BUS_X0} y1={BUS_Y + 2.5} x2={BUS_X1} y2={BUS_Y + 2.5}
              stroke={soft(SUP_HUE, 40)} strokeWidth={1} />
        <text x={BUS_X1} y={BUS_Y - 10} textAnchor="end" fontFamily={MONO}
              fontSize={8.5} letterSpacing="1.6" fill={soft(SUP_HUE, 55)}>
          DISPATCH BUS
        </text>

        {/* Bus TAPS: one bright node per energised lane. Deliberately NOT a
            bracket spanning min..max — a span reads as "everything between
            these two is running", which is false the moment the fan-out skips
            an agent (IC ‖ LIT with XD idle in between). Discrete taps can only
            say what is true. */}
        {activeIdx.map((i) => (
          <g key={`tap-${i}`} filter="url(#tp-bloom)">
            <rect x={modCx(i) - 3} y={BUS_Y - 4} width={6} height={8} rx={1.5}
                  fill={agentColorVar(PIPELINE[i]!)} />
          </g>
        ))}
        {parallel && (
          <g filter="url(#tp-bloom)">
            {/* the count, floated ABOVE the rail so it never sits on the trace */}
            <rect
              x={(modCx(activeIdx[0]!) + modCx(activeIdx[activeIdx.length - 1]!)) / 2 - 44}
              y={BUS_Y - 30}
              width={88}
              height={16}
              rx={8}
              fill="var(--mast-bg)"
              stroke="var(--mast-auto)"
              strokeWidth={1}
            />
            <text
              x={(modCx(activeIdx[0]!) + modCx(activeIdx[activeIdx.length - 1]!)) / 2}
              y={BUS_Y - 19}
              textAnchor="middle"
              fontFamily={MONO}
              fontSize={9.5}
              fontWeight={700}
              letterSpacing="0.6"
              fill="var(--mast-auto)"
            >
              ‖ 并行 ×{activeIdx.length}
            </text>
          </g>
        )}

        {/* ── drop traces: bus → each agent ────────────────────────────── */}
        {PIPELINE.map((id, i) => {
          const lv = live[id] ?? {};
          const hue = agentColorVar(id);
          const cx = modCx(i);
          const d = `M ${cx} ${BUS_Y + 3} L ${cx} ${MOD_Y}`;
          if (lv.active) {
            return (
              <g key={`tr-${id}`}>
                {/* bloom underlay + core trace, dashes marching toward the agent */}
                <path d={d} fill="none" stroke={soft(hue, 45)} strokeWidth={5}
                      filter="url(#tp-bloom)" />
                <path d={d} fill="none" stroke={hue} strokeWidth={1.8}
                      strokeDasharray="7 5">
                  <animate attributeName="stroke-dashoffset" from="12" to="0"
                           dur="0.65s" repeatCount="indefinite" />
                </path>
                <circle cx={cx} cy={MOD_Y - 4} r={2.6} fill={hue} />
              </g>
            );
          }
          return (
            <path
              key={`tr-${id}`}
              d={d}
              fill="none"
              stroke={lv.held ? "var(--mast-warn)" : "var(--mast-border)"}
              strokeWidth={lv.held ? 1.4 : 1}
              strokeDasharray={lv.held ? "2 3" : "1 5"}
              opacity={lv.held ? 0.9 : 0.85}
            />
          );
        })}

        {/* ── supervisor module ────────────────────────────────────────── */}
        <g onClick={() => onPick(SUP_ID)} style={{ cursor: "pointer" }}>
          <title>编排与协调 · 任务分解 / 并行下发 / 异常处理</title>
          {selected === SUP_ID && (
            <rect x={SUP_BOX.x - 4} y={SUP_BOX.y - 4} width={SUP_BOX.w + 8}
                  height={SUP_BOX.h + 8} rx={10} fill="none" stroke={SUP_HUE}
                  strokeWidth={1.6} opacity={0.55} />
          )}
          <rect x={SUP_BOX.x} y={SUP_BOX.y} width={SUP_BOX.w} height={SUP_BOX.h}
                rx={7} className="fill-mast-panel" stroke={soft(SUP_HUE, 70)}
                strokeWidth={1.2} />
          {/* silkscreen index stripe */}
          <rect x={SUP_BOX.x} y={SUP_BOX.y} width={SUP_BOX.w} height={3} rx={1.5}
                fill={SUP_HUE} />
          <text x={SUP_BOX.x + 12} y={SUP_BOX.y + 26} fontFamily={MONO} fontSize={15}
                fontWeight={700} letterSpacing="1.2" fill={SUP_HUE}>
            SUP
          </text>
          <text x={SUP_BOX.x + 12} y={SUP_BOX.y + 41} fontFamily={SANS} fontSize={10.5}
                className="fill-mast-muted">
            编排与协调
          </text>
          {supLive.model && (
            <text x={SUP_BOX.x + 12} y={SUP_BOX.y + 54} fontFamily={MONO} fontSize={8.5}
                  fill={soft(SUP_HUE, 75)}>
              {String(supLive.model).slice(0, 16)}
            </text>
          )}
          <Led x={SUP_BOX.x + SUP_BOX.w - 13} y={SUP_BOX.y + 14}
               on={!!supLive.active} hue="var(--mast-auto)" />
        </g>

        {/* ── agent modules ────────────────────────────────────────────── */}
        {PIPELINE.map((id, i) => (
          <AgentModule
            key={id}
            id={id}
            i={i}
            live={live[id] ?? {}}
            selected={selected === id}
            onPick={onPick}
            onToggleHold={onToggleHold}
          />
        ))}

        {/* ── legend: the only place idle/live/held are spelled out ────── */}
        <g transform={`translate(${ROW_X} ${H - 16})`} fontFamily={MONO} fontSize={9}>
          <line x1={0} y1={-3} x2={16} y2={-3} stroke="var(--mast-border)"
                strokeWidth={1} strokeDasharray="1 5" />
          <text x={22} y={0} className="fill-mast-muted">空闲</text>
          <line x1={58} y1={-3} x2={74} y2={-3} stroke="var(--mast-auto)" strokeWidth={1.8}
                strokeDasharray="7 5" />
          <text x={80} y={0} className="fill-mast-muted">运行中（通电）</text>
          <line x1={170} y1={-3} x2={186} y2={-3} stroke="var(--mast-warn)"
                strokeWidth={1.4} strokeDasharray="2 3" />
          <text x={192} y={0} className="fill-mast-muted">已暂停</text>
          <text x={252} y={0} className="fill-mast-muted" opacity={0.75}>
            · 仪器控制独占硬件，与其它智能体并行安全
          </text>
        </g>
      </svg>
    </div>
  );
}

// ── a pulsing status LED (the one piece of motion on an idle board) ──────────
function Led({ x, y, on, hue }: { x: number; y: number; on: boolean; hue: string }) {
  if (!on) {
    return <circle cx={x} cy={y} r={3} fill="none" stroke="var(--mast-border)" strokeWidth={1} />;
  }
  return (
    <g filter="url(#tp-bloom)">
      <circle cx={x} cy={y} r={3.4} fill={hue}>
        <animate attributeName="opacity" values="1;0.35;1" dur="1.4s" repeatCount="indefinite" />
      </circle>
    </g>
  );
}

function AgentModule({
  id,
  i,
  live: lv,
  selected,
  onPick,
  onToggleHold,
}: {
  id: string;
  i: number;
  live: AgentLive;
  selected: boolean;
  onPick: (id: string) => void;
  onToggleHold?: (id: string, held: boolean) => void;
}) {
  const a = agentDef(id);
  const hue = agentColorVar(id);
  const x = modX(i);
  const active = !!lv.active;
  const held = !!lv.held;
  const edge = active ? hue : held ? "var(--mast-warn)" : soft("var(--mast-muted)", 38);

  return (
    <g transform={`translate(${x} ${MOD_Y})`} onClick={() => onPick(id)} style={{ cursor: "pointer" }}>
      <title>{`${a.cn} · ${a.role}`}</title>

      {/* live modules bloom; idle ones stay graphite */}
      {active && (
        <rect x={-2} y={-2} width={MOD_W + 4} height={MOD_H + 4} rx={9}
              fill={soft(hue, 12)} stroke={soft(hue, 55)} strokeWidth={1}
              filter="url(#tp-bloom)" />
      )}

      {/* scope-cursor corner ticks mark the selection (no heavy ring) */}
      {selected && (
        <g stroke={hue} strokeWidth={1.6} fill="none" opacity={0.9}>
          <path d={`M -5 8 L -5 -5 L 8 -5`} />
          <path d={`M ${MOD_W - 8} -5 L ${MOD_W + 5} -5 L ${MOD_W + 5} 8`} />
          <path d={`M -5 ${MOD_H - 8} L -5 ${MOD_H + 5} L 8 ${MOD_H + 5}`} />
          <path d={`M ${MOD_W - 8} ${MOD_H + 5} L ${MOD_W + 5} ${MOD_H + 5} L ${MOD_W + 5} ${MOD_H - 8}`} />
        </g>
      )}

      <rect x={0} y={0} width={MOD_W} height={MOD_H} rx={7}
            className="fill-mast-panel" stroke={edge} strokeWidth={active ? 1.5 : 1} />
      {/* index stripe — the agent's identity hue, always on (it's who it IS) */}
      <rect x={0} y={0} width={MOD_W} height={3} rx={1.5} fill={hue}
            opacity={active ? 1 : 0.5} />

      {/* silkscreen: mono short-id, large + tracked out */}
      <text x={11} y={26} fontFamily={MONO} fontSize={16} fontWeight={700}
            letterSpacing="1.4" fill={active ? hue : "var(--mast-text)"}>
        {a.short}
      </text>
      <text x={11} y={41} fontFamily={SANS} fontSize={10.5} className="fill-mast-muted">
        {a.cn}
      </text>

      <Led x={MOD_W - 13} y={14} on={active} hue={hue} />
      {held && !active && (
        <text x={MOD_W - 11} y={17} textAnchor="end" fontFamily={MONO} fontSize={9}
              fontWeight={700} fill="var(--mast-warn)">
          ⏸
        </text>
      )}

      {/* model — the technical footnote, in the panel's smallest type */}
      {lv.model && (
        <text x={11} y={MOD_H - 26} fontFamily={MONO} fontSize={8.5}
              fill={soft(hue, active ? 95 : 62)}>
          {String(lv.model).slice(0, 17)}
        </text>
      )}
      {typeof lv.threads === "number" && lv.threads > 0 && (
        <text x={MOD_W - 11} y={MOD_H - 26} textAnchor="end" fontFamily={MONO}
              fontSize={8.5} className="fill-mast-muted">
          {lv.threads}t
        </text>
      )}

      {/* hold control — a panel switch, not a button */}
      {onToggleHold && (
        <g
          transform={`translate(11 ${MOD_H - 18})`}
          onClick={(e) => {
            e.stopPropagation();
            onToggleHold(id, !held);
          }}
          style={{ cursor: "pointer" }}
        >
          <rect x={0} y={0} width={MOD_W - 22} height={13} rx={3}
                fill={held ? soft("var(--mast-auto)", 14) : "transparent"}
                stroke={held ? "var(--mast-auto)" : soft("var(--mast-muted)", 34)}
                strokeWidth={0.9} />
          <text x={(MOD_W - 22) / 2} y={9.6} textAnchor="middle" fontFamily={MONO}
                fontSize={8.5} letterSpacing="0.8" fontWeight={600}
                fill={held ? "var(--mast-auto)" : "var(--mast-muted)"}>
            {held ? "RESUME 恢复" : "HOLD 暂停"}
          </text>
        </g>
      )}
    </g>
  );
}
