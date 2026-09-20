/** 「这个 agent 每次调用到底收到了什么、各占多大」——纯逻辑。
 *
 *  ## 为什么单独一个文件
 *
 *  `.tsx` 在这个仓里跑不了单测（`npm run test:unit` 是 `node --test`，JSX 剥不
 *  掉），所以凡是**会算错**的东西都要住在 `.ts` 里。这里算的每一件事都有算错
 *  的方式：占比会加总到 99%，码点偏移会切错一位，未知大小的块会被当成 0 从而
 *  把条画得比实际短。
 *
 *  ## 几条不显然的约定
 *
 *  * **supervisor 前后端两个名字**：前端 `_supervisor`，后端 `orchestrator`
 *    （`make_chat_model("orchestrator")`、登记表的 `agent="orchestrator"`）。
 *    不映射的话 SUP 那一列永远是空的 —— 而空看起来像「编排器没有注入」。
 *  * **快照末尾那条 `ai:response` 是模型的输出**，不是输入。它不计进
 *    `total_chars`，历史条数与占比都要把它剔掉。
 *  * **偏移按码点**：后端 Python 的 `len()` 数码点，JS 的 `.length` 数 UTF-16
 *    码元。一个 emoji 就能让所有段错一位。
 *  * **`chars` 是截断前的原长**，`content` 可能已被截到 20 k。切分要钳位并说
 *    出来，不能假装没截。
 */

import type { components } from "@/api/schema";

// ── 契约类型（生成物的别名，别在这里重抄一份字段）─────────────────────────

export type InjectionBlock = components["schemas"]["InjectionBlock"];
export type AgentInfo = components["schemas"]["AgentInfo"];
export type InjectionMatrix = components["schemas"]["InjectionMatrixResponse"];
export type AgentInjection = components["schemas"]["AgentInjectionResponse"];
export type LatestCapture = components["schemas"]["LatestCaptureResponse"];
export type CapturedMessage = components["schemas"]["CapturedMessageModel"];
export type ToolSurface = components["schemas"]["ToolSurfaceInfo"];

/** 后端 registry 里「全员」的那个值。 */
export const ALL_AGENTS = "*";

/** 前端 supervisor 的 id（`registry.tsx` 的 `SUP_ID`）。 */
export const UI_SUPERVISOR = "_supervisor";
/** 后端同一个东西的 id。 */
export const BACKEND_SUPERVISOR = "orchestrator";

/** 前端 agent id → 后端 id。 */
export function toBackendAgentId(uiId: string): string {
  return uiId === UI_SUPERVISOR ? BACKEND_SUPERVISOR : uiId;
}

/** 后端 agent id → 前端 id。 */
export function toUiAgentId(backendId: string): string {
  return backendId === BACKEND_SUPERVISOR ? UI_SUPERVISOR : backendId;
}

// ── 词表 ────────────────────────────────────────────────────────────────

const WHEN_LABEL: Record<string, string> = {
  always: "每次调用",
  when_set: "设了才有",
  on_event: "出事才有",
  on_mode: "看模式",
  build_time: "建图时定",
};

const POSITION_LABEL: Record<string, string> = {
  system: "系统消息",
  last_human: "最后一条用户消息",
  new_human: "新插一条消息",
  state_messages: "改写消息列表",
  tool_result: "改写工具返回",
  tools: "工具面",
};

const AVAILABILITY_LABEL: Record<string, string> = {
  static: "固定文本",
  live: "按当前状态渲染",
  needs_hardware: "需要实时硬件",
  needs_request: "只存在于一次请求内",
};

/** 未知值**原样返回**，不写「未知」—— 后端加一个枚举值时，页面不该变成一片未知。 */
function labelOf(table: Record<string, string>, key: string): string {
  return table[key] ?? key;
}

export const whenLabel = (w: string) => labelOf(WHEN_LABEL, w);
export const positionLabel = (p: string) => labelOf(POSITION_LABEL, p);
export const availabilityLabel = (a: string) => labelOf(AVAILABILITY_LABEL, a);

/** 这一块的正文能直接显示吗。`needs_*` 一律不能 —— 那是诚实性铁律的类型化。 */
export function canShowText(block: Pick<InjectionBlock, "availability">): boolean {
  return block.availability === "static" || block.availability === "live";
}

/** 这一块会到 `agent` 手上吗。 */
export function appliesTo(
  block: Pick<InjectionBlock, "agents">,
  backendId: string,
): boolean {
  const list = block.agents ?? [];
  if (list.length === 0) return false;
  return list.includes(ALL_AGENTS) || list.includes(backendId);
}

// ── 消息拆分 ────────────────────────────────────────────────────────────

export type SplitMessages = {
  system: CapturedMessage[];
  history: CapturedMessage[];
  /** 模型自己的输出（快照末尾那条）。不是输入。 */
  response: CapturedMessage | null;
};

export function splitMessages(messages: CapturedMessage[] | undefined): SplitMessages {
  const all = messages ?? [];
  let response: CapturedMessage | null = null;
  const rest: CapturedMessage[] = [];
  for (const m of all) {
    if (m.role === "ai:response") response = m;
    else rest.push(m);
  }
  return {
    system: rest.filter((m) => m.role === "system"),
    history: rest.filter((m) => m.role !== "system"),
    response,
  };
}

// ── 按块切分一条消息 ────────────────────────────────────────────────────

export type Segment = {
  /** null = 这一段没有归属（注入 helper 之外的文本）。 */
  id: string | null;
  start: number;
  end: number;
  text: string;
  chars: number;
  /** 这一段落在了 `content` 之外（原文被截断过）。 */
  clipped: boolean;
};

export type SegmentIssue =
  | { kind: "beyond_content"; detail: string }
  | { kind: "overlap"; detail: string }
  | { kind: "chars_mismatch"; detail: string };

type BlockRef = { id: string; start: number; end: number; chars: number };

/** 把一条消息的文本按块切成带来源标签的分段。
 *
 *  空洞补成 `id: null` 的未归属段，而不是丢掉 —— 丢掉的话各段加起来不等于全文，
 *  而「加起来对不上」正是这个页面要暴露的那类问题。
 */
export function splitByBlocks(
  content: string,
  chars: number,
  blocks: BlockRef[] | null | undefined,
): { segments: Segment[]; issues: SegmentIssue[] } {
  // 按**码点**切：后端的偏移是 Python 的 len()，数的是码点。
  const cp = Array.from(content ?? "");
  const n = cp.length;
  const issues: SegmentIssue[] = [];
  const cut = (a: number, b: number) => cp.slice(a, b).join("");

  const list = [...(blocks ?? [])].sort((x, y) => x.start - y.start);
  if (list.length === 0) {
    return {
      segments: n === 0 ? [] : [{ id: null, start: 0, end: n, text: content,
                                  chars: n, clipped: false }],
      issues,
    };
  }

  const segments: Segment[] = [];
  let cursor = 0;
  let prevEnd = -1;
  for (const b of list) {
    if (prevEnd >= 0 && b.start < prevEnd) {
      issues.push({ kind: "overlap",
                    detail: `${b.id} 与前一块重叠（start ${b.start} < ${prevEnd}）` });
    }
    prevEnd = b.end;
    const start = Math.max(0, Math.min(b.start, n));
    const end = Math.max(start, Math.min(b.end, n));
    if (start > cursor) {
      segments.push({ id: null, start: cursor, end: start,
                      text: cut(cursor, start), chars: start - cursor,
                      clipped: false });
    }
    const clipped = b.end > n;
    if (clipped) {
      issues.push({ kind: "beyond_content",
                    detail: `${b.id} 的结束位置 ${b.end} 超出了可见文本长度 ${n}（原文被截断过）` });
    }
    segments.push({ id: b.id, start, end, text: cut(start, end),
                    chars: b.chars, clipped });
    cursor = Math.max(cursor, end);
  }
  if (cursor < n) {
    segments.push({ id: null, start: cursor, end: n, text: cut(cursor, n),
                    chars: n - cursor, clipped: false });
  }

  const claimed = list.reduce((s, b) => s + (b.chars || 0), 0);
  if (chars > 0 && claimed > chars) {
    issues.push({ kind: "chars_mismatch",
                  detail: `各块声称的 ${claimed} 字符多于这条消息的 ${chars}` });
  }
  return { segments, issues };
}

// ── 占比条 ──────────────────────────────────────────────────────────────

export type ShareKey = "static" | "block" | "unattributed" | "history" | "tools";

export type Share = {
  key: ShareKey;
  id?: string;
  label: string;
  chars: number;
  /** 百分比，**整数，加总恰好 100**。 */
  pct: number;
  /** 这一段是估算（工具面按建图时的口径量的）。 */
  estimated: boolean;
};

/** 最大余数法：各自 `Math.round` 会加总成 99 或 101，而一条加不满的占比条
 *  会让人以为有一块没被算进去。 */
function toPercents(values: number[]): number[] {
  const total = values.reduce((a, b) => a + b, 0);
  if (total <= 0) return values.map(() => 0);
  const exact = values.map((v) => (v * 100) / total);
  const floors = exact.map((v) => Math.floor(v));
  let remainder = 100 - floors.reduce((a, b) => a + b, 0);
  const order = exact
    .map((v, i) => ({ i, frac: v - Math.floor(v) }))
    .sort((a, b) => b.frac - a.frac);
  const out = [...floors];
  for (const { i } of order) {
    if (remainder <= 0) break;
    const cur = out[i];
    if (cur === undefined) continue;
    out[i] = cur + 1;
    remainder -= 1;
  }
  return out;
}

export function composeShares(args: {
  segments: Segment[];
  history: CapturedMessage[];
  toolsChars: number | null | undefined;
  labelFor: (id: string) => string;
}): Share[] {
  const parts: Omit<Share, "pct">[] = [];
  for (const seg of args.segments) {
    if (seg.chars <= 0) continue;
    if (seg.id === null) {
      parts.push({ key: "unattributed", label: "未归属", chars: seg.chars,
                   estimated: false });
    } else {
      const isStatic = seg.id.startsWith("agent.") || seg.id === "system.base";
      parts.push({ key: isStatic ? "static" : "block", id: seg.id,
                   label: args.labelFor(seg.id), chars: seg.chars,
                   estimated: false });
    }
  }
  // 历史用**原长** `chars`，不是可能被截过的 `content.length`。
  const histChars = args.history.reduce((s, m) => s + (m.chars || 0), 0);
  if (histChars > 0) {
    parts.push({ key: "history", label: `对话历史（${args.history.length} 条）`,
                 chars: histChars, estimated: false });
  }
  // null 与 0 不一样：null = 没量到（不画这一段），0 = 量到了、就是没有。
  if (args.toolsChars !== null && args.toolsChars !== undefined) {
    parts.push({ key: "tools", label: "工具面（估算）", chars: args.toolsChars,
                 estimated: true });
  }
  if (parts.length === 0) return [];
  const pcts = toPercents(parts.map((p) => p.chars));
  return parts.map((p, i) => ({ ...p, pct: pcts[i] ?? 0 }));
}

// ── 矩阵 ────────────────────────────────────────────────────────────────

export type MatrixRow = {
  block: InjectionBlock;
  cells: boolean[];
  shared: boolean;
  exclusive: boolean;
};

export function buildMatrix(
  matrix: InjectionMatrix | undefined,
  columns: string[],
): MatrixRow[] {
  const blocks = matrix?.blocks ?? [];
  const cells = matrix?.cells ?? {};
  return blocks.map((block) => {
    const row = cells[block.id] ?? {};
    const agents = block.agents ?? [];
    return {
      block,
      cells: columns.map((c) => Boolean(row[c])),
      shared: agents.includes(ALL_AGENTS),
      exclusive: Boolean(block.exclusive),
    };
  });
}

export type Footprint = {
  /** 能算出字符数的块。 */
  knownChars: number;
  /** 只在真实请求里才有内容的块 —— 计数，**不当成 0**。 */
  unknownCount: number;
  sharedCount: number;
  exclusiveCount: number;
};

export function agentFootprint(
  matrix: InjectionMatrix | undefined,
  backendId: string,
): Footprint {
  const out: Footprint = { knownChars: 0, unknownCount: 0, sharedCount: 0,
                           exclusiveCount: 0 };
  for (const block of matrix?.blocks ?? []) {
    if (!appliesTo(block, backendId)) continue;
    if (canShowText(block)) out.knownChars += block.effective_chars ?? 0;
    else out.unknownCount += 1;
    if ((block.agents ?? []).includes(ALL_AGENTS)) out.sharedCount += 1;
    else if (block.exclusive) out.exclusiveCount += 1;
  }
  return out;
}

// ── 杂项 ────────────────────────────────────────────────────────────────

export function fmtChars(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n < 1000) return `${n} 字符`;
  return `${(n / 1000).toFixed(1)}k 字符`;
}

export function fmtAge(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒前`;
  if (s < 3600) return `${Math.round(s / 60)} 分钟前`;
  return `${(s / 3600).toFixed(1)} 小时前`;
}

/** 段落锚点 id。`.` 保留 —— 换成 `-` 会让 `mw.a.b` 与 `mw.a_b` 撞成一个。 */
export function segmentAnchor(id: string): string {
  return `ctx-seg-${id.replace(/[^a-zA-Z0-9_.-]/g, "_")}`;
}

/** 快照拿不到时的说明。四种原因指向四个不同的动作，不能揉成一句「暂无数据」。 */
export function captureEmptyMessage(reasonCode: string, reason: string): string {
  return reason || `没有快照（${reasonCode || "原因不明"}）。`;
}
