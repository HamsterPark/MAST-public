// ════════════════════════════════════════════════════════════════════════════
// src/lib/narration.ts — 旁白插在转录的**哪一条后面**。
//
// 旁白（narration）是系统在向用户解说一个长任务正在做什么：「我们要打一发
// 10 V / 500 ms 的脉冲」「扫到 50%」。它**不是助手说的话** —— agent 看不见它，
// 它也不进 LangGraph 的 message channel。后端：mast/chat/narration.py。
//
// 为什么排序逻辑住在这里，而不是写在组件里：
//
//   两条道各自有序，难的是**把它们并到一起**。消息现在带 `t`（后端
//   `MessageClockMiddleware`，2026-08-12 起），旁白一直带 `t`，所以主判据是
//   时间；`anchor`（旁白发出那一刻已渲染的消息条数）退成兜底，只在某一边没有
//   时刻时才用 —— 它记的是「第几条」，而转录被压缩之后那个坐标系会被整个重写
//   （见 mergeNarration 的合并规则）。
//
//   ⚠️ 这段注释在 2026-08-23 之前写的是「render_history 没有时间戳，所以按
//   wall-clock 归并做不到」。那句话在 08-12 之后就不成立了，而代码照着它继续
//   只认 anchor —— **一句过期的注释把一个已经修好的前提锁在了原地**。
//
//   归并规则写成纯函数才能被断言。一个埋在 JSX 里的 `.filter()` 出了错的样子是
//   「旁白偶尔排在奇怪的位置」，没人会去查。
//
//     cd frontend && npm run test:unit
// ════════════════════════════════════════════════════════════════════════════

export interface NarrationItem {
  seq: number;
  t: number;
  text: string;
  nk?: string;
  /** 发出这条时已渲染的消息条数；-1 = 没有活跃回合 → 追加到末尾。 */
  anchor?: number;
  tone?: string;
  has_image?: boolean;
  fold?: number;
  facts?: Record<string, unknown>;
}

export interface MessageLike {
  role: string;
  content: string;
  /** 这条消息**发生的时刻**（epoch 秒），后端 `MessageClockMiddleware` 盖的。
   *
   *  **缺席有意义**：`undefined` / `null` = 「不知道它是什么时候说的」
   *  （重启前就在 checkpoint 里的历史、或被压缩摘要替换掉的那一段），
   *  **不是零时刻**。合并时按「未定时」处理，绝不当成 1970。 */
  t?: number | null;
}

/** 消息的**有效时刻**序列：`eff[i]` = 到第 i 条为止已知的最晚时刻。
 *
 *  · 未盖戳的消息**继承前一条**的有效时刻 —— 它排在那儿，就跟那儿同期；
 *  · 开头那一段全没戳（压缩摘要 / 重启前历史）⇒ `null` = 「比什么都早」，
 *    旁白不会被顶到它们上面去；
 *  · 用 `max` 而不是直接赋值：这样 `eff` **按构造非降**，
 *    于是「第一个比 T 晚的下标」可以直接扫出来，不必假设转录本身有序。
 */
function effectiveTimes(messages: readonly MessageLike[]): (number | null)[] {
  const out: (number | null)[] = [];
  let carried: number | null = null;
  for (const m of messages) {
    const t = typeof m?.t === "number" && Number.isFinite(m.t) ? m.t : null;
    if (t !== null) carried = carried === null ? t : Math.max(carried, t);
    out.push(carried);
  }
  return out;
}

export type Lane<M> =
  | { kind: "message"; message: M; index: number }
  | { kind: "narration"; item: NarrationItem };

/**
 * 把旁白与消息合并成扁平渲染序列。
 * 两边都有时间戳时按事件时刻归并，避免入队延迟和历史压缩扰乱顺序。
 * 缺时间戳的条目逐条回退到 anchor：n 放在第 n 条消息后，0 放开头；
 * 越界或 -1 放末尾。同一位置按 t 排序，t 相同时以 seq 排序。
 * 工具执行中的旁白可能早于工具返回后才盖戳的助手消息，这符合事件时序。
 */
export function mergeNarration<M extends MessageLike>(
  messages: readonly M[],
  items: readonly NarrationItem[],
): Lane<M>[] {
  const n = messages.length;
  const eff = effectiveTimes(messages);
  const dated = eff.some((t) => t !== null);
  // 桶：0..n。超界与 -1 一律归到 n（末尾）。
  const buckets = new Map<number, NarrationItem[]>();
  for (const item of items ?? []) {
    const at = placeOne(item, n, eff, dated);
    const list = buckets.get(at);
    if (list) list.push(item);
    else buckets.set(at, [item]);
  }
  for (const list of buckets.values()) {
    list.sort(byEventTime);
  }

  const out: Lane<M>[] = [];
  for (const item of buckets.get(0) ?? []) out.push({ kind: "narration", item });
  for (let i = 0; i < n; i++) {
    out.push({ kind: "message", message: messages[i]!, index: i });
    for (const item of buckets.get(i + 1) ?? []) out.push({ kind: "narration", item });
  }
  return out;
}

/**
 * 这一条旁白该插在**第几条消息之后**（返回 0..n）。
 *
 * 按时间：找到第一条「有效时刻晚于这条旁白」的消息，插在它前面。
 * `eff` 按构造非降（见 {@link effectiveTimes}），所以一趟线性扫就够，
 * 而且结果与转录本身是否严格有序无关。
 *
 * 退回 `anchor` 的两种情形，**都不是异常**：
 *  · 这条旁白自己没有可用的 `t`（后端老行数据）；
 *  · 整份转录一条戳都没有（重启前的存档 / 关掉了时钟中间件）。
 * 这两种下，行为与 2026-08-23 之前**逐字相同**。
 */
function placeOne(
  item: NarrationItem,
  n: number,
  eff: readonly (number | null)[],
  dated: boolean,
): number {
  const t = typeof item?.t === "number" && Number.isFinite(item.t) && item.t > 0
    ? item.t
    : null;
  if (t !== null && dated) {
    for (let i = 0; i < n; i++) {
      const e = eff[i];
      // `null` = 这条消息（以及它前面那些）没有已知时刻 —— 当成「比什么都早」，
      // 旁白不会被顶到压缩摘要 / 重启前历史的**上面**去。
      // `undefined` 走同一支：`noUncheckedIndexedAccess` 下越界读也是它，而
      // 「读不到」和「没有时刻」在这里要的动作相同 —— 都不能拿来做比较。
      if (typeof e === "number" && e > t) return i;
    }
    return n;
  }
  const raw = typeof item.anchor === "number" ? item.anchor : -1;
  return raw < 0 || raw > n ? n : raw;
}

/**
 * 增量游标：这次该从哪个 seq 之后拉。
 *
 * 只往前，永不后退 —— 后退会让同一批旁白被重复插进列表。会话一换必须归零，
 * 那是**另一条**转录（沿用 seq 会让新会话的前几条被当成已读而永远不显示）。
 */
export function nextCursor(current: number, latestSeq: number | undefined): number {
  const latest = typeof latestSeq === "number" && Number.isFinite(latestSeq) ? latestSeq : 0;
  return latest > current ? latest : current;
}

/**
 * 排序判据：**事件发生时刻**优先，同刻才看 `seq`。
 *
 * `seq` 留作平手裁决而不是被丢掉：它是 SQLite 原子分配的，永远唯一且稳定，
 * 所以两条同一毫秒的旁白不会因为浏览器排序不稳定而来回跳。
 *
 * `t` 缺失/垃圾 ⇒ 回落到 `seq` 的量级之外是危险的（会把它顶到最前面），
 * 所以缺失时**只比 seq**：不知道发生时刻的那条，就按它入库的位置待着。
 */
export function byEventTime(a: NarrationItem, b: NarrationItem): number {
  const ta = Number.isFinite(a?.t) ? a.t : null;
  const tb = Number.isFinite(b?.t) ? b.t : null;
  if (ta !== null && tb !== null && ta !== tb) return ta - tb;
  return (a?.seq ?? 0) - (b?.seq ?? 0);
}

/** 合并新拉到的一批，按 seq 去重（WS 推送 + 轮询兜底会送来同一条）。 */
export function mergeBatch(
  existing: readonly NarrationItem[],
  incoming: readonly NarrationItem[],
): NarrationItem[] {
  if (!incoming?.length) return existing as NarrationItem[];
  // 去重的键仍然是 `seq` —— 它是这条旁白的**身份**（同一条被 WS 和轮询各送
  // 一次时要能认出是同一条）。`t` 不能当身份：两条不同的旁白可以同刻发生。
  const bySeq = new Map<number, NarrationItem>();
  for (const it of existing ?? []) bySeq.set(it.seq, it);
  for (const it of incoming) bySeq.set(it.seq, it);
  return [...bySeq.values()].sort(byEventTime);
}

/**
 * 旁白上显示的时刻。`HH:MM:SS`，不显示毫秒。
 *
 * Agent 的发言也带上时间。带时间的用处不是
 * 精确计时，是**让顺序可核** —— 一条排在前面的旁白如果时间更晚，那就是排序
 * 出了问题，而在没有时间显示之前，那种错只能靠「感觉有点乱」被发现。
 *
 * `t` 不是有效数 ⇒ 返回 `""`（不显示），**不显示 1970**：一个假时间比没有
 * 时间坏，因为它会被当成真的去推理。
 */
export function fmtClock(t: unknown): string {
  const v = typeof t === "number" && Number.isFinite(t) && t > 0 ? t : null;
  if (v === null) return "";
  // 后端存的是 epoch **秒**（`time.time()`）。毫秒会变成 55000 年，
  // 所以这里顺手认一下量级 —— 判据是「1971 年之后」。
  const ms = v > 1e11 ? v : v * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return "";
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/** 这条 `chat_narration` 事件是不是「我正在看的那个会话」的。 */
export function isNarrationFor(
  data: { conversation_id?: unknown } | null | undefined,
  conversationId: string | null | undefined,
): boolean {
  if (!data || !conversationId) return false;
  return String((data as { conversation_id?: unknown }).conversation_id ?? "") === conversationId;
}

// 令牌必须是 tailwind.config.ts 里真有的那几个。写错一个不报错、不崩，只是
// **什么也不做**（元素保持继承来的样式），症状只有用户一句「看不清」——
// 树里曾经躺过好几个这样的死类名。判据在 test/designTokens.test.ts，
// 而它是按字符串扫的：**连注释里都不能出现一个不存在的令牌名**。
// 「good」用 auto（AUTO 安全档的绿）而不是另起一个语义色，不新增令牌。
const TONE_CLASS: Record<string, string> = {
  info: "border-mast-border text-mast-muted",
  good: "border-mast-auto-border text-mast-auto",
  warn: "border-mast-warn-border text-mast-warn",
};

/** tone → 配色。**只做配色，不做判断** —— 未知 tone 退回中性，不猜。 */
export function toneClass(tone: string | undefined): string {
  return TONE_CLASS[String(tone ?? "info")] ?? TONE_CLASS.info!;
}

/**
 * 缩略图 URL。`has_image` 为假时返回 null —— 调用方据此**不渲染 `<img>`**。
 *
 * 这样即使取图端点还没上线（分阶段实施），也不会有任何一个请求发出去：
 * 没有坏掉的图标，没有 404 噪音。
 */
export function narrationImageUrl(
  item: NarrationItem,
  conversationId: string | null | undefined,
): string | null {
  if (!item?.has_image || !conversationId) return null;
  return (
    `/api/chat/narration-image/${item.seq}` +
    `?conversation_id=${encodeURIComponent(conversationId)}&px=240`
  );
}

// ── 跨挂载留存 ──────────────────────────────────────────────────────────────
//
// 已知问题:切换标签页再回来的时候旁白会消失。
//
// 原因有两层,少修一层都不行:
//
//   ① 条目存在 `NarrationLane` 的 `useState` 里、游标存在 `useRef` 里 ——
//      切走 = 组件卸载 = 两样一起丢。
//   ② 回来时 react-query 先把**上一次增量请求的缓存**喂回来。那次请求带的是
//      `after_seq=120` 之类,响应里只有最后两条;`useEffect` 照单全收之后把
//      游标推到 `latest_seq`,于是**前面一百多条永远补不回来** ——
//      看起来就是「旁白没有了」。
//
// 存在模块级 Map 里:切走再回来,条目和游标都还在,而且不必重新拉一遍全量。
// 按会话分桶 —— 换会话是换一条转录,不能串。

export interface NarrationMemo<T> {
  items: T[];
  cursor: number;
}

const _memo = new Map<string, NarrationMemo<unknown>>();

/** 这条会话攒到哪儿了。没有就给一个空的(不写进表,读不该有副作用)。 */
export function recallNarration<T>(conversationId: string | null | undefined): NarrationMemo<T> {
  const k = String(conversationId || "");
  const got = k ? _memo.get(k) : undefined;
  return got ? (got as NarrationMemo<T>) : { items: [], cursor: 0 };
}

/** 记下这条会话攒到哪儿了。 */
export function rememberNarration<T>(
  conversationId: string | null | undefined,
  items: T[],
  cursor: number,
): void {
  const k = String(conversationId || "");
  if (!k) return;
  _memo.set(k, { items: items as unknown[], cursor });
  // 只留最近几条会话 —— 一条长任务几百条旁白,无上限会把内存吃掉。
  // 删最早插入的那个:Map 保序,`keys().next()` 就是它。
  while (_memo.size > NARRATION_MEMO_MAX) {
    const oldest = _memo.keys().next();
    if (oldest.done) break;
    _memo.delete(oldest.value);
  }
}

/** 记几条会话。3 = 主聊天 + 两个刚看过的,再多没有意义。 */
export const NARRATION_MEMO_MAX = 3;

/** 换样品/换实验之类需要整个忘掉时用(测试也用它隔离)。 */
export function forgetAllNarration(): void {
  _memo.clear();
}
