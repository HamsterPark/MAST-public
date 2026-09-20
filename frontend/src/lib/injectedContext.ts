/** 「我这句话发出去之后，系统往模型那儿塞了什么」——把一条用户消息对上一次真实请求。
 *
 *  对话里用一个可折叠/展开的按钮展示系统自动注入的
 *  上下文，便于调试。
 *
 *  ## 为什么是按时间对，而不是按 id 对
 *
 *  快照是在 `on_chat_model_start` 上抓的（`mast/prompts/capture.py`）——那个回调
 *  拿得到消息列表，拿不到「这是哪一轮对话」。要让它拿到，得把 conversation_id 从
 *  HTTP 路由一路穿到模型工厂，而中间隔着 LangGraph 的执行器：contextvar 在线程池
 *  里不保证传播，穿不过去的那些请求会**安静地不带标记**，于是「对不上」和「没记录」
 *  变成同一种表现。那正是这个功能要避免的东西。
 *
 *  所以按时间：用户消息带 `t`（`MessageClockMiddleware` 盖的 epoch 秒），快照带
 *  `ts`。一轮的请求 = 落在 [这条用户消息, 下一条用户消息) 之间的那些。同一台机器、
 *  同一个进程、同一个时钟，没有跨机偏移问题。
 *
 *  ## 对不上的时候必须说清是哪一种对不上
 *
 *  「没记录」有五种，指向的动作完全不同：捕获被关了（去开）、进程还没跑过模型调用
 *  （正常）、这一轮太老被挤掉了（调大 MAX_SNAPSHOTS 或早点看）、这一轮还没发出请求
 *  （等一下）、这条消息压根没有时间戳（重启前的历史，永远对不上）。
 *  一个统一的「暂无数据」把五种揉成一种，而它们没有一种能靠再点一次解决。
 */

/** 一条快照的元数据（`/api/admin/prompt-capture` 的 items 元素）。 */
export type CaptureItem = {
  index: number;
  /** 稳定 id。`index` 每来一次模型调用就整体挪位，别拿它去取详情。 */
  seq: number;
  ts: number;
  source: string;
  model_id: string;
  provider: string;
  message_count: number;
  total_chars: number;
  system_chars: number;
};

/** `/api/admin/prompt-capture` 的回包里这个功能用得上的部分。 */
export type CaptureList = {
  items: CaptureItem[];
  count: number;
  enabled: boolean;
  total_seen: number;
  capacity: number;
  degraded: boolean;
};

export type InjectedMatch =
  | {
      kind: "match";
      /** 这一轮的**第一次**模型调用——注入的上下文就是它带过去的那份。 */
      seq: number;
      ts: number;
      source: string;
      model_id: string;
      /** 这一轮一共发出去几次模型调用（每一轮工具调用都是一次）。 */
      calls: number;
      /** system 消息的总字符数，取自这一次调用。 */
      systemChars: number;
      totalChars: number;
      /** 匹配到的请求不是这个 agent 发的——仍然展示，但要说出来。 */
      foreignSource: boolean;
    }
  | { kind: "none"; why: string };

/** 没有时间戳的消息永远对不上——这不是错误，是它本来就没有可对的东西。 */
const NO_CLOCK =
  "这条消息没有时间戳（多半是重启前就在存档里的历史），对不上任何一次请求。";

function reasonWhenEmpty(list: CaptureList): string | null {
  if (list.degraded) return "注入记录这个模块没能加载，读不到。";
  if (!list.enabled)
    return "快照记录已关闭，所以这里是空的 —— 不是没有发生过模型调用。";
  if (list.count === 0)
    return "本进程还没有发生过模型调用。这里只记录真实请求，不做离线模拟渲染。";
  return null;
}

/**
 * 把一条用户消息对上它那一轮的第一次模型请求。
 *
 * @param t          这条用户消息的 epoch 秒（`ChatMsg.t`）。null/undefined = 没有。
 * @param nextUserT  **下一条**用户消息的 epoch 秒；没有下一条就传 null
 *                   （表示这一轮还在进行中，窗口右边界是无穷大）。
 * @param list       `/api/admin/prompt-capture` 的回包。
 * @param agentId    这个对话属于哪个 agent（快照的 `source`）。
 */
export function matchInjectedContext(
  t: number | null | undefined,
  nextUserT: number | null,
  list: CaptureList,
  agentId: string,
): InjectedMatch {
  const empty = reasonWhenEmpty(list);
  if (empty) return { kind: "none", why: empty };
  if (typeof t !== "number" || !Number.isFinite(t)) {
    return { kind: "none", why: NO_CLOCK };
  }

  // 窗口右开：下一条用户消息**当时**的请求属于下一轮。没有下一条就不封口。
  const hi = typeof nextUserT === "number" && Number.isFinite(nextUserT)
    ? nextUserT
    : Number.POSITIVE_INFINITY;
  const inWindow = list.items
    .filter((it) => it.ts >= t && it.ts < hi)
    // items 是**新的在前**；这一轮的注入要看第一次调用，所以按时间正序。
    .sort((a, b) => a.ts - b.ts);

  if (inWindow.length === 0) {
    // 分清「挤掉了」和「还没发」——两者都空，但一个是过去式一个是将来式。
    const oldest = list.items.reduce(
      (m, it) => (it.ts < m ? it.ts : m),
      Number.POSITIVE_INFINITY,
    );
    if (Number.isFinite(oldest) && oldest > t) {
      return {
        kind: "none",
        why:
          `这一轮的注入已经被后面的请求挤出记录了（只留最近 ${list.capacity} 次，` +
          `本进程至今 ${list.total_seen} 次）。`,
      };
    }
    return {
      kind: "none",
      why: "这一轮还没有发出模型请求（或者它没走会被记录的那条路）。",
    };
  }

  // 同一个窗口里可能有别的 agent / 后台调用。优先取本对话这个 agent 的；
  // 一个都没有就取最早的那次，并且**说出来它不是这个 agent 发的**——
  // 悄悄展示一份别人的注入，比不展示更坏。
  const own = inWindow.filter((it) => it.source === agentId);
  const pool = own.length > 0 ? own : inWindow;
  const picked = pool[0];
  // `inWindow.length === 0` 上面已经退出了，所以 pool 一定非空。写出来是给
  // `noUncheckedIndexedAccess` 看的 —— 而它是对的：一个 `pool[0]!` 会让这条
  // 不变量隐形，下一个人加一层 filter 时不会有任何东西提醒他。
  if (!picked) {
    return { kind: "none", why: "这一轮还没有发出模型请求。" };
  }
  return {
    kind: "match",
    seq: picked.seq,
    ts: picked.ts,
    source: picked.source,
    model_id: picked.model_id,
    calls: pool.length,
    systemChars: picked.system_chars,
    totalChars: picked.total_chars,
    foreignSource: picked.source !== agentId,
  };
}

/**
 * 每一轮的起点——**全部**用户消息的时间，升序。
 *
 * ⚠️ 必须从**整份**转录算，不能按下标算。`NarrationLane` 把消息切成好几段
 * 分别交给 `ChatBubbles`（旁白卡片插在段之间），一段里的「最后一条用户消息」
 * 在整份转录里往往后面还有。按下标算 → 那一条的窗口右端点变成无穷大 →
 * 它会把**后面所有轮**的请求都算进自己这一轮。
 *
 * 所以边界是一个与切片无关的集合，查的时候用 {@link nextTurnAfter}。
 */
export function userTurnTimes(
  msgs: { role: string; t?: number | null }[],
): number[] {
  return msgs
    .filter((m) => m.role === "user" && typeof m.t === "number"
      && Number.isFinite(m.t))
    .map((m) => m.t as number)
    .sort((a, b) => a - b);
}

/** 这一轮的右边界：`times` 里**严格大于** `t` 的最小值；没有就是 null。 */
export function nextTurnAfter(
  times: number[],
  t: number | null | undefined,
): number | null {
  if (typeof t !== "number" || !Number.isFinite(t)) return null;
  let best: number | null = null;
  for (const x of times) {
    // 严格大于：同一秒里的**另一条**用户消息不该把这一轮提前封口。
    if (x > t && (best === null || x < best)) best = x;
  }
  return best;
}

/** 注入块 = system 角色的消息。对话历史不算注入——它在屏幕上就看得见。 */
export type CapturedMsg = {
  role: string;
  content: string;
  chars: number;
  truncated: boolean;
};

export function splitInjected(messages: CapturedMsg[]): {
  injected: CapturedMsg[];
  history: CapturedMsg[];
} {
  const injected = messages.filter((m) => m.role === "system");
  return { injected, history: messages.filter((m) => m.role !== "system") };
}
