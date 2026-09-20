/**
 * 技能市场写入的判据 + manifest 解析 —— 一份判据，一个地方。
 *
 * ── 为什么不能只看 `ok` ──
 * 改订阅只改了一个 holder。agent 的工具表是在**建图时冻结**的，所以一次成功的
 * 写入之后，模型手上那张表可能还是旧的：任务在跑 ⇒ 重建排队；重建被调度了但失败；
 * 压根没有活的运行时（存下来了，下次启动才生效）。三条路径都返回 `ok:true`。
 *
 * 这与 `settingsWrite.ts` 是同一条纪律的延伸，但多了一层：那边只需回答「存进去
 * 了吗」，这边还要回答「**现在生效了吗**」。所以判据有三态：
 *
 *   null            → 存了，而且 agent 手上那张表已经跟上
 *   { kind: "pending" } → 存了，还没生效（要显示成黄色，不是绿色）
 *   { kind: "failed" }  → 没存进去
 *
 * `test/skillMarketWiring.test.ts` 有一条闸门守着「谁 POST 了市场端点就必须用它」。
 */

export type MarketWriteResult = {
  ok?: boolean;
  degraded?: boolean;
  reason?: string | null;
  rebuild_note?: string | null;
  /** agent 侧还没跟上（true = 现在还没生效）；null = 判断不了。 */
  agent_path_pending?: boolean | null;
  /** agent 手上那张表 == 注册表现在的样子；**null ≠ false**。 */
  fingerprint_matches?: boolean | null;
  skipped_mandatory?: string[] | null;
  unknown?: string[] | null;
};

export type MarketWriteProblem = {
  kind: "failed" | "pending" | "unsure";
  message: string;
};

/**
 * 写入的结果到底是什么。真的存了**并且**生效了返回 null。
 *
 * 顺序是有意的：先「没收到回应」，再 degraded（内核没接上），再 `ok !== true`
 * （被拒），最后才是三态的生效判断。少一层都会让某一类失败静静变成绿色。
 */
export function marketWriteProblem(
  res: MarketWriteResult | null | undefined,
): MarketWriteProblem | null {
  if (!res) return { kind: "failed", message: "保存失败：没有收到内核的回应。" };
  if (res.degraded) {
    return { kind: "failed", message: res.reason || "写入未生效（内核未接入）。" };
  }
  if (res.ok !== true) {
    return { kind: "failed", message: res.reason || "保存被拒绝（内核未说明原因）。" };
  }
  // 后端那句人话逐字带出来 —— 它说的是「排队了/重建了/下次启动生效」，
  // 前端自己改写会把三种情况压成一种。
  const note = (res.rebuild_note || "").trim();
  if (res.agent_path_pending === true) {
    return { kind: "pending", message: note || "已保存，但 agent 工具表尚未跟上。" };
  }
  if (res.agent_path_pending === null || res.agent_path_pending === undefined) {
    return { kind: "unsure", message: note || "已保存；是否已生效判断不了。" };
  }
  if (res.fingerprint_matches === false) {
    return {
      kind: "pending",
      message: note || "已保存，但 agent 手上那张工具表还是旧的。",
    };
  }
  // fingerprint_matches === null 时不报 pending：进程刚起、还没建过工具表，
  // 那是「判断不了」，而 agent_path_pending 已经说了 false（重建走完了）。
  return null;
}

/** 生效状态的三态标签 —— null / false / true 各说各话，不折叠。 */
export function liveBadge(
  matches: boolean | null | undefined,
): { tone: "ok" | "warn" | "unknown"; text: string } {
  if (matches === true) return { tone: "ok", text: "已生效" };
  if (matches === false) return { tone: "warn", text: "尚未跟上" };
  return { tone: "unknown", text: "无法确认" };
}

/** 必装项在界面上是锁着的开关 —— 判据与后端同源（用它返回的 mandatory 字段）。 */
export function canUnsubscribe(row: { mandatory?: boolean }): boolean {
  return !row.mandatory;
}

/**
 * 构建器 palette 要不要把这一条藏起来。
 *
 * **`=== false` 而不是 `!subscribed`**：字段缺席（旧响应、目录降级、订阅子系统读
 * 不出来）时**不过滤**。写成 `!e.subscribed` 的话，一个少了这个字段的响应会让整个
 * palette 空掉，而界面上只会显示「没有匹配的技能」—— 一个完全合法的错答案。
 * 这是本仓 [[unknown_is_not_an_answer]] 的形状：「不知道」被折叠成了「否」。
 */
export function paletteHides(entry: { subscribed?: boolean }, subOnly: boolean): boolean {
  return subOnly && entry.subscribed === false;
}

/** 被订阅过滤挡掉的条数。**要显示出来** —— 静默截断会被读成「本机没有那个技能」。 */
export function paletteHiddenCount(
  entries: { subscribed?: boolean }[],
  subOnly: boolean,
): number {
  return entries.filter((e) => paletteHides(e, subOnly)).length;
}

/**
 * 「随应用发布」以外的来源 —— 也就是**在这台机器上出现的**技能。
 *
 * 判据用 `classify_origin` 的闭集词汇（后端唯一真源）。`builtin` / `composite` /
 * `paper` / `agent_tool` 都是随应用一起来的，升级会带；这三个是本机后天长出来的：
 * 用户/agent 组合的工作流、自建 .py、覆盖层新增。
 */
export const LOCAL_ORIGINS = ["user_composite", "custom", "overlay"] as const;

/**
 * 本机长出来、却**不在订阅面上**的技能。
 *
 * 这个列表补的是一个真实的空档：定制过订阅之后，agent 用技能工坊新造的组合技能
 * 只进市场、不进工具面 —— 而**没有任何东西会告诉用户它出现了**。工坊那边的
 * 说法是「下一轮它会自己出现在工具表里」，在定制过的机器上那句话不成立。
 *
 * 刻意做成**派生视图而不是一道闸**：闸会把工具面重新变成安全边界（2026-08-20
 * 已经拆过一次），而这里真正缺的是**可见性**。所以它没有自己的状态、不会过期、
 * 也不需要 agent 那边配合埋点。
 */
export function locallyAuthoredUnsubscribed<
  T extends { name: string; source?: string; subscribed?: boolean },
>(rows: T[]): T[] {
  const local = new Set<string>(LOCAL_ORIGINS);
  return rows.filter((r) => r.subscribed === false && local.has(r.source ?? ""));
}

/** 角标只数待确认的，已裁决的不算。 */
export function pendingBadgeCount(
  recs: { status?: string }[] | null | undefined,
): number {
  return (recs ?? []).filter((r) => (r.status ?? "pending") === "pending").length;
}

// ── manifest ────────────────────────────────────────────────────────────────

export type ManifestEntry = { name: string; source?: string; version?: string; spec?: unknown };
export type SubscriptionManifest = {
  kind: string;
  schema_version?: number;
  exported_at?: string;
  machine?: string;
  customised?: boolean;
  entries: ManifestEntry[];
};

export const MANIFEST_KIND = "mast-skill-subscription";

/**
 * 解析一份别人发来的订阅列表文件。
 *
 * 失败返回一句中文原因而不是抛 —— 调用方是一个文件选择器，它拿到的东西可能是
 * 任何东西（一张图、一份 conduct 导出、一个截断的下载）。
 */
export function parseSubscriptionManifest(
  text: string,
): { manifest: SubscriptionManifest } | { error: string } {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { error: `这不是一个 JSON 文件：${(e as Error).message}` };
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { error: "文件内容不是一个对象。" };
  }
  const obj = raw as Record<string, unknown>;
  if (obj.kind !== MANIFEST_KIND) {
    return {
      error: `这不是一份订阅列表（kind = ${JSON.stringify(obj.kind ?? null)}，应为 "${MANIFEST_KIND}"）。`,
    };
  }
  if (!Array.isArray(obj.entries)) {
    return { error: "manifest 里没有 entries 列表。" };
  }
  const entries: ManifestEntry[] = [];
  for (const it of obj.entries) {
    if (it && typeof it === "object" && typeof (it as ManifestEntry).name === "string") {
      const e = it as ManifestEntry;
      if (e.name.trim()) entries.push(e);
    }
  }
  return {
    manifest: {
      kind: MANIFEST_KIND,
      schema_version: typeof obj.schema_version === "number" ? obj.schema_version : 1,
      exported_at: typeof obj.exported_at === "string" ? obj.exported_at : "",
      machine: typeof obj.machine === "string" ? obj.machine : "",
      customised: obj.customised === true,
      entries,
    },
  };
}

/**
 * 导入前的本地预览：这份清单里有多少本机有、多少没有。
 *
 * 这只是给用户看的**预览**。真正的判据在后端（`POST /import` 的 dry_run），
 * 因为内嵌的 composite 落地之后「有没有」的答案会变 —— 前端算的那份必然偏保守。
 */
export function diffManifestAgainstCatalog(
  manifest: SubscriptionManifest,
  known: Iterable<string>,
): { matched: string[]; missing: ManifestEntry[]; embedded: number } {
  const have = new Set(known);
  const matched: string[] = [];
  const missing: ManifestEntry[] = [];
  let embedded = 0;
  for (const e of manifest.entries) {
    if (e.spec) embedded += 1;
    if (have.has(e.name)) matched.push(e.name);
    else missing.push(e);
  }
  return { matched, missing, embedded };
}
