// ════════════════════════════════════════════════════════════════════════════
// conduct.ts —— conduct 面板的**纯**判断。
//
// 与 `pages/ConductPage.tsx` 的关系同 `lib/envPanel.ts` 与右栏:这里没有 React
// import,所以每一条判断都能在 node --test 里直接跑。
//
// **ZERO IMPORTS,而且必须保持。** `node --test` 直接加载这个 .ts(没有打包器、
// 没有路径别名),一个 `@/…` 说明符会让整份测试文件当场挂掉。
//
// ── 抠出来的理由 ────────────────────────────────────────────────────────────
//
// 面板上的判断几乎全是「什么时候**不**能按」「什么时候那个数不是它看起来的意思」:
//
//   · 一个在 PAUSED 状态下还能按的「暂停」按钮 —— 按下去后端记一条 op_rejected,
//     而屏幕上什么都不会发生;
//   · 一个空着的预算条 —— 读的人以为「没花钱」,真相是「没看」;
//   · 一个不显示 waive 标记的等待卡 —— 那个闸是人拿命令行放过去的,报告里有、
//     屏幕上没有;
//   · 一条把「正常长步」报成停滞的告警 —— 报久了就没人看,真停滞时也没人看。
//
// 这些错法**没有一个会崩、会被 typecheck 抓到、或者在截图里看出来**。
//
// ── 状态词表是闭集,而且它的真源在 Python 里 ─────────────────────────────────
//
// `mast/conduct/store.py` 的 `STATUSES` / `OPS` 是唯一真源。这里这份是**镜像**,
// 由 `frontend/test/conduct.test.ts` 直接读那个 .py 文件对账 —— 不是靠 openapi
// 中转(响应字段是 `str`,枚举根本进不了 schema),也不是靠人记得两边一起改。
// 「双端镜像一开始就配 parity 测试」是设计 §10-6 点名的教训,这就是那道闸。
//
// 未知状态**不是错误**:后端加一个新状态时,老前端要把它原样显示成中性色,
// 而不是渲染成空白。一个空白的状态栏看起来像「没在跑」。
// ════════════════════════════════════════════════════════════════════════════

/** 顶层状态。镜像 `store.STATUSES`,顺序照它。 */
export const CONDUCT_STATUSES = [
  "draft",
  "approved",
  "running",
  "waiting_operator",
  "waiting_condition",
  "yielding",
  "paused",
  "halted_estop",
  "recovery_pending",
  "completed",
  "aborted",
] as const;

export type ConductStatus = (typeof CONDUCT_STATUSES)[number];

/** 终态。镜像 `store.TERMINAL_STATUSES`。 */
export const TERMINAL_STATUSES: readonly string[] = ["completed", "aborted"];

/** 用户意图。镜像 `store.OPS`。 */
export const CONDUCT_OPS = [
  "pause",
  "resume",
  "abort",
  "takeover",
  "ack",
  "set_attended",
  "waive_condition",
  "override_decision",
] as const;

export type ConductOp = (typeof CONDUCT_OPS)[number];

/**
 * 面板配色档。
 *
 * 刻意用**语义名**而不是直接写 `components/ui.tsx` 的 Badge 词表:那份词表说的是
 * 安全等级(AUTO/DANGEROUS),而这里说的是 conduct 的处境。两者的映射写在
 * :func:`badgeTone` 一处,由测试对着 ui.tsx 的真词表钉住 —— 拼错一个 tone 名字
 * 的症状是徽章**悄悄退回默认色**,而默认色看起来完全正常。
 */
export type Tone = "ok" | "warn" | "crit" | "info" | "neutral";

/** 语义档 → `components/ui.tsx` 的 Badge tone。 */
export function badgeTone(tone: Tone): string {
  switch (tone) {
    case "ok":
      return "AUTO";
    case "warn":
      return "WARN";
    case "crit":
      return "DANGEROUS";
    case "info":
      return "INFO";
    default:
      return "default";
  }
}

const STATUS_LABEL: Record<ConductStatus, string> = {
  draft: "草稿",
  approved: "已批准",
  running: "执行中",
  waiting_operator: "等人",
  waiting_condition: "等条件",
  yielding: "让路中",
  paused: "已暂停",
  halted_estop: "急停闩挂着",
  recovery_pending: "恢复自检中",
  completed: "已完成",
  aborted: "已中止",
};

const STATUS_TONE: Record<ConductStatus, Tone> = {
  draft: "neutral",
  approved: "info",
  running: "ok",
  waiting_operator: "warn",
  waiting_condition: "warn",
  yielding: "warn",
  paused: "info",
  halted_estop: "crit",
  recovery_pending: "warn",
  completed: "ok",
  aborted: "neutral",
};

function known(status: string): status is ConductStatus {
  return (CONDUCT_STATUSES as readonly string[]).includes(status);
}

/**
 * 状态的人读名。
 *
 * **不认识的原样返回**,不返回空串:后端加了新状态而这个前端还没更新时,用户
 * 该看到 `some_new_state` 而不是一片空白 —— 空白看起来像「没在跑」。
 */
export function statusLabel(status: string): string {
  if (!status) return "（无状态）";
  return known(status) ? STATUS_LABEL[status] : status;
}

export function statusTone(status: string): Tone {
  return known(status) ? STATUS_TONE[status] : "neutral";
}

export function isTerminal(status: string): boolean {
  return TERMINAL_STATUSES.includes(status);
}

// ── 按钮能不能按 ─────────────────────────────────────────────────────────────
//
// 镜像 `director.OP_VALID_STATUSES`。**不在表里 = 后端会显式拒绝并记
// op_rejected**,不是静默 no-op —— 但用户看到的仍然是「我按了,什么都没发生」。
// 所以按钮在这一侧就该是灰的,而且**说得出为什么灰**。

/** Director 真正驱动着这份 conduct 的那些状态。 */
const DRIVEN: readonly string[] = [
  "running",
  "waiting_operator",
  "waiting_condition",
  "yielding",
  "recovery_pending",
];

/** 一切非终态。abort 必须**从任何非终态都够得着** —— 「能停不能解」的反面。 */
const NON_TERMINAL: readonly string[] = CONDUCT_STATUSES.filter(
  (s) => !TERMINAL_STATUSES.includes(s),
);

export const OP_VALID_STATUSES: Record<ConductOp, readonly string[]> = {
  pause: DRIVEN,
  takeover: DRIVEN,
  resume: ["paused"],
  abort: NON_TERMINAL,
  set_attended: NON_TERMINAL,
  ack: ["waiting_operator", "waiting_condition"],
  waive_condition: ["waiting_operator", "waiting_condition"],
  // 闸门判定停下来之后那条**留痕的解锁路**。只在 waiting_operator 有意义。
  //
  // ⚠️ 状态对了**还不够**:同一个 waiting_operator 有两个来源(一个 wait 步 /
  // 一次裁决转人),只有后者有可放行的判定。按钮的最终可用性还要看
  // `pending_decision` 非空 —— 见 `overrideEnabled`。
  override_decision: ["waiting_operator"],
};

const OP_LABEL: Record<ConductOp, string> = {
  pause: "暂停",
  resume: "继续",
  abort: "中止",
  takeover: "人工接管",
  ack: "我已确认",
  set_attended: "值守模式",
  waive_condition: "由我提供证据",
  override_decision: "我看过了,继续",
};

export function opLabel(op: ConductOp): string {
  return OP_LABEL[op];
}

/**
 * 「我看过了,继续」这个按钮能不能按 —— 比 `opEnabled` 多问一层。
 *
 * 状态是 `waiting_operator` **还不够**:同一个状态有两个来源 ——
 * 一个 `wait` 步(等 ack + 物理条件,该按「我已确认」)与一次裁决转人
 * (等人的判断)。只有后者有一个可放行的判定,后端也只认那一种。
 *
 * 灰的时候必须说得出为什么灰:一个没有理由的灰按钮和一个坏掉的按钮,在屏幕上
 * 长得一模一样。
 */
export function overrideEnabled(
  status: string,
  pending: { decision_id: number; gate_id: string } | null | undefined,
): { enabled: boolean; why: string } {
  const base = opEnabled("override_decision", status);
  if (!base.enabled) return base;
  if (!pending || !pending.decision_id) {
    return {
      enabled: false,
      why: "现在停的不是一次闸门判定 —— 等待步用「我已确认」,别的停法用中止或接管",
    };
  }
  return { enabled: true, why: "" };
}

/**
 * 这个按钮现在能不能按,以及不能按时**那句话**。
 *
 * 灰按钮必须自带理由。一个没有理由的灰按钮和一个坏掉的按钮,在屏幕上长得一模
 * 一样,而用户对前者的正确反应是「哦，现在不该按」,对后者是「报个 bug」。
 */
export function opEnabled(
  op: ConductOp,
  status: string,
): { enabled: boolean; why: string } {
  const allowed = OP_VALID_STATUSES[op];
  if (allowed.includes(status)) return { enabled: true, why: "" };
  if (isTerminal(status)) {
    return {
      enabled: false,
      why: `这份 conduct 已经${statusLabel(status)} —— 终态是终态,要接着做请新建一份`,
    };
  }
  return {
    enabled: false,
    why: `${opLabel(op)}在「${statusLabel(status)}」下没有意义（可用于：${allowed
      .map(statusLabel)
      .join(" / ")}）`,
  };
}

// ── 等待:缺哪个闸 ───────────────────────────────────────────────────────────

export interface ConditionLike {
  desc?: string | null;
  threshold?: number | null;
  stale_after_s?: number | null;
  current_value?: number | null;
  met?: boolean | null;
  met_since?: number | null;
  stale?: boolean | null;
  reading_age_s?: number | null;
  waived?: boolean | null;
  waived_by?: string | null;
  waive_reason?: string | null;
}

export interface WaitLike {
  wait_id?: string | null;
  kind?: string | null;
  message?: string | null;
  ack?: { required?: boolean | null; at?: number | null; by?: string | null } | null;
  condition?: ConditionLike | null;
  lacking?: string[] | null;
}

/** 一个闸在面板上的样子。 */
export interface GateRow {
  /** 「人的确认」/「物理条件」 */
  label: string;
  /** 这个闸过了没有。 */
  ok: boolean;
  /** **判不了**(读不到) —— 与「没过」是两件事,颜色和措辞都不同。 */
  unreadable: boolean;
  /** 一句给人看的话。 */
  detail: string;
}

/**
 * 双闸各自的状态。
 *
 * **两个证据回答两个问题,互不替代**:人确认了不等于降到温,降到温不等于样品
 * 换好了。所以这里永远返回两行(有条件闸的话),而不是一个「还差 1 项」的计数 ——
 * 计数答不出「差的是哪一个」,而那正是用户要做的下一件事。
 */
export function waitGateRows(wait: WaitLike | null | undefined): GateRow[] {
  if (!wait) return [];
  const rows: GateRow[] = [];
  const ack = wait.ack ?? {};
  if (ack.required) {
    const done = ack.at != null;
    rows.push({
      label: "人的确认",
      ok: done,
      unreadable: false,
      detail: done
        ? `${ack.by || "某人"} 已确认`
        : "等一句「我已确认」——按下面那个按钮",
    });
  }
  const cond = wait.condition;
  if (cond) {
    if (cond.waived) {
      rows.push({
        label: "物理条件",
        ok: true,
        unreadable: false,
        // waive 的标记**持续显示**,不是 ack 那一刻显示一次:这个闸是人拿命令
        // 放过去的,报告里带着它,屏幕上也必须一直带着。
        detail: `已由 ${cond.waived_by || "某人"} 提供证据放行${
          cond.waive_reason ? `：${cond.waive_reason}` : ""
        }`,
      });
    } else if (cond.stale) {
      // **stale = 读不到 ≠ 没到。** 干等下去是错的,要人来看。
      rows.push({
        label: "物理条件",
        ok: false,
        unreadable: true,
        detail: `读不到${
          cond.reading_age_s != null ? `（读数已 ${fmtDuration(cond.reading_age_s)}）` : ""
        } —— 读不到不等于没到，请人来看（温度采集程序还在跑吗）`,
      });
    } else {
      rows.push({
        label: "物理条件",
        ok: Boolean(cond.met),
        unreadable: false,
        detail: conditionText(cond),
      });
    }
  }
  return rows;
}

/** 条件那一行的正文:目标、当前读数、到位了多久。 */
export function conditionText(cond: ConditionLike | null | undefined): string {
  if (!cond) return "";
  const target = cond.desc || (cond.threshold != null ? `目标 ${cond.threshold}` : "");
  const now =
    cond.current_value != null ? `当前 ${trimNum(cond.current_value)}` : "当前读不到";
  const held = cond.met && cond.met_since != null ? "，已到位" : "";
  return [target, now].filter(Boolean).join("；") + held;
}

/** 还缺哪些闸(给等待大卡的标题用)。空数组 = 两个闸都齐了。 */
export function lackingLabels(wait: WaitLike | null | undefined): string[] {
  return waitGateRows(wait)
    .filter((r) => !r.ok)
    .map((r) => r.label);
}

// ── heartbeat 告警条 ─────────────────────────────────────────────────────────

export interface HeartbeatLike {
  at?: number | null;
  age_s?: number | null;
  stalled?: boolean | null;
  threshold_s?: number | null;
  in_step?: boolean | null;
  step_elapsed_s?: number | null;
  reason?: string | null;
}

export interface Banner {
  tone: Tone;
  title: string;
  detail: string;
}

/**
 * 停滞告警条。不该报的时候返回 `null` —— **告警报久了就没人看**,而那时真的
 * 停滞发生了也一样没人看。
 *
 * 后端已经把「正常长步」和「循环死了」分成两支判过了(设计 §6-1),这里**不重判**,
 * 只把它的结论和理由画出来。前端再判一次就会有第二套阈值,而两套阈值迟早会给出
 * 两个答案。
 */
export function heartbeatBanner(
  hb: HeartbeatLike | null | undefined,
  status: string,
): Banner | null {
  if (!hb || !hb.stalled) return null;
  if (isTerminal(status)) return null;
  const detail = hb.reason || "";
  if (hb.in_step) {
    return {
      tone: "warn",
      title: "这一步停太久了",
      detail:
        (hb.step_elapsed_s != null
          ? `当前步已经跑了 ${fmtDuration(hb.step_elapsed_s)}`
          : "当前步已经超时") +
        (hb.threshold_s != null ? `（阈值 ${fmtDuration(hb.threshold_s)}）` : "") +
        (detail ? `。${detail}` : "") +
        "。指挥线程**不会**去杀这一步：卡死的 TCP 事务杀不得，" +
        "强杀会永久损坏 Nanonis 端口。护针的是看门狗。",
    };
  }
  return {
    tone: "crit",
    title: "指挥线程可能停了",
    detail:
      (hb.age_s != null ? `心跳已经 ${fmtDuration(hb.age_s)} 没更新` : "读不到心跳") +
      (hb.threshold_s != null ? `（阈值 ${fmtDuration(hb.threshold_s)}）` : "") +
      (detail ? `。${detail}` : ""),
  };
}

// ── 预算 ─────────────────────────────────────────────────────────────────────

export interface BudgetLike {
  spent_usd?: number | null;
  cap_usd?: number | null;
  reason?: string | null;
  enforceable?: boolean | null;
  not_enforceable_why?: string | null;
  measured_by_currency?: Record<string, number> | null;
}

/**
 * 预算条的正文。
 *
 * **读不到 ≠ 花了 0。** 一个空着的、或者写着 `$0.00` 的预算条会被读成「没花钱」,
 * 而真相可能是「没有人在记账」。所以读不到的时候写的是「读不到」加一句为什么。
 *
 * **而「拦不住」是第三件事,与前两件都不同。** 上限拦不住的时候,连
 * 「上限 $20.00」这半句都不该照原样印 —— 一个 `读不到 / 上限 $20.00` 读起来
 * 仍然像「有一道上限在那儿,只是这一刻没读到花了多少」。真相是那道上限不存在:
 * 账本按 provider 原生币种实测,而它是 USD,合并需要汇率而本仓不自造汇率。
 */
export function budgetText(b: BudgetLike | null | undefined): {
  text: string;
  unreadable: boolean;
  hint: string;
} {
  const cap = b?.cap_usd ?? 0;
  const enforceable = b?.enforceable !== false; // 旧响应没有这个字段 ⇒ 按旧行为
  const capText = cap > 0 ? `上限 $${cap.toFixed(2)}` : "未设上限";
  // 逐币种实测:有就显示,**不折成一个数**(折成一个数就得有汇率)。
  const measured = Object.entries(b?.measured_by_currency ?? {});
  const measuredText = measured.length
    ? measured.map(([cur, v]) => `${cur} ${v.toFixed(2)}`).join(" + ")
    : "";

  if (!enforceable) {
    return {
      text: measuredText
        ? `${measuredText}(上限不生效)`
        : `上限已声明但不生效`,
      unreadable: true,
      hint:
        b?.not_enforceable_why ||
        b?.reason ||
        `声明了 ${capText},但没有任何东西会因为它停下来`,
    };
  }
  if (!b || b.spent_usd == null) {
    return {
      text: `读不到 / ${capText}`,
      unreadable: true,
      hint: b?.reason || "花销读不到——读不到不是 0",
    };
  }
  return {
    text: `$${b.spent_usd.toFixed(2)} / ${capText}`,
    unreadable: false,
    hint: b.reason || "",
  };
}

// ── approve 批不下去的两种 ───────────────────────────────────────────────────

export interface ApproveLike {
  ok?: boolean | null;
  validation_ok?: boolean | null;
  validation_complete?: boolean | null;
}

/**
 * 批不下去的**那一句**。返回空串 = 没被挡。
 *
 * 后端只看 `approvable = ok ∧ complete`,而这两个不成立的原因要做的事完全不同:
 *
 * * **发现了错误**(ok=false)—— 模板本身有问题,去改模板;
 * * **有检查根本没跑**(complete=false)—— 比如注册表读不到,规则③整条没执行。
 *   这时 `ok` 仍然是 true,而「没检查」与「检查通过」在屏幕上会长得一模一样。
 *
 * 把两者并成一句「校验失败」,就把这个区分又抹掉了 —— 而它正是这套校验器当初
 * 要报三态(ok / complete / approvable)的全部理由。
 */
export function approveBlockReason(o: ApproveLike | null | undefined): string {
  if (!o || o.ok) return "";
  if (o.validation_ok && o.validation_complete === false) {
    return "不是发现了错误，是**有检查根本没跑**（比如技能注册表读不到）——「没检查」不许长得像「检查通过」";
  }
  if (o.validation_ok === false) return "模板本身有问题，见下面的发现";
  return "批不下去，见下面的原因";
}

// ── WS 帧:只做触发 ──────────────────────────────────────────────────────────

/** 这三种帧的名字,逐字对应 `core/events.py` 的 EventType。 */
export const CONDUCT_FRAMES = [
  "conduct_status",
  "conduct_gate",
  "conduct_alert",
] as const;

/**
 * 一帧到了该做什么。答案只有两个:**refetch,或者忽略**。
 *
 * ⚠️ 这个函数**故意不从帧里读任何状态**,返回值里也没有任何 payload 字段。
 * 设计 §7 的判据是「帧只做触发,不做增量状态源」,理由是这条总线只重放最后 100
 * 条:一个跑三天的 conduct 必然丢帧,而按帧累积状态的客户端会带着一个**错的**
 * 状态一直显示下去 —— 比没有实时推送糟得多。面板唯一的数据源是
 * `GET /api/conducts/{id}`,一次响应 = 一个时刻。
 *
 * `conductId` 不匹配时忽略:同一条总线上跑着别的 conduct 的帧(以及重启前那份
 * 的重放),照单全收会让面板去抓一份根本没在看的 conduct。
 */
export function frameAction(
  type: string,
  data: unknown,
  conductId: string,
): "refetch" | "ignore" {
  if (!(CONDUCT_FRAMES as readonly string[]).includes(type)) return "ignore";
  if (!conductId) return "ignore";
  const d = data && typeof data === "object" ? (data as { conduct_id?: unknown }) : null;
  const id = d && typeof d.conduct_id === "string" ? d.conduct_id : "";
  // 帧里没有 conduct_id(旧后端/畸形帧)⇒ 保守地刷一次:多一次 GET 的代价,
  // 远小于漏掉一次「等你确认」的代价。
  if (!id) return "refetch";
  return id === conductId ? "refetch" : "ignore";
}

// ── 心愿单 ↔ 面板 ────────────────────────────────────────────────────────────

/** conduct 在心愿单里的 agent_id 前缀。镜像 `adapters.WISHLIST_AGENT_PREFIX`。 */
export const WISHLIST_AGENT_PREFIX = "conduct:";

/**
 * 心愿单里那条请求是哪份 conduct 发的。不是 conduct 发的就返回空串。
 *
 * 等人的时候有**两个**地方会亮:conduct 面板的等待大卡,和心愿单里一条
 * `conduct:<id>` 的请求。后者常常是用户先看到的那个(他本来就在看待办),
 * 而在此之前那条请求是一段**没有出口的文字** —— 想去处理它得先知道 conduct
 * 面板在「实验记录」下面。
 *
 * 这与 2026-08-04 仪器初始化那次不同:那次是两个入口**都不是导航**;这里导航是
 * 有的,补的是「从待办直接走过去」那一跳。
 */
export function conductIdFromAgentId(agentId: string | null | undefined): string {
  const s = String(agentId ?? "");
  if (!s.startsWith(WISHLIST_AGENT_PREFIX)) return "";
  return s.slice(WISHLIST_AGENT_PREFIX.length).trim();
}

/** 面板的深链。带 id 时可收藏、可转发给下一个班的人。 */
export function conductPanelHref(conductId?: string | null): string {
  const id = String(conductId ?? "").trim();
  return id ? `/records/conduct?id=${encodeURIComponent(id)}` : "/records/conduct";
}

// ── 小工具 ───────────────────────────────────────────────────────────────────

/** 秒 → 人读时长。面板上到处都是「多久」,拼两遍就会有两种写法。 */
export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return m ? `${h} 小时 ${m} 分` : `${h} 小时`;
}

/** epoch 秒 → 本地时刻。空值给「—」而不是 1970。 */
export function fmtClock(epoch: number | null | undefined): string {
  if (epoch == null || !Number.isFinite(epoch) || epoch <= 0) return "—";
  return new Date(epoch * 1000).toLocaleString();
}

function trimNum(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return Math.abs(v) >= 100 ? v.toFixed(0) : String(Number(v.toFixed(3)));
}

/** 时间线一行的进度文案(`3/7 步`)。 */
export function timelineProgress(done: number, total: number): string {
  return `${Math.max(0, Math.min(done, total))}/${total} 步`;
}

// ── 设置页 ↔ conduct 页的共用文案（2026-08-27） ──────────────────────

/** 设置页里那一节的标题。**两页引用同一个常量**。
 *
 * ConductPage 的「去设置里打开它」指引此前指向「设置 → 常规设置」——
 * 一个不存在的地方（SettingsPage 里 grep `conduct` 零命中）。用户照着做
 * 找不到，于是这条线只能靠 curl 打开。按构造保证两边一致，比再写一遍靠谱。
 */
export const CONDUCT_SETTINGS_TITLE = "conduct 指挥线程 Conduct director";

/** 「已保存」与「真的在跑」是两件事 —— 说清楚是哪一件。
 *
 * `cd_enabled` 写下去之后：拿得到 runtime 句柄时线程**当场**起停；拿不到
 * （standalone）才是「下次启动生效」。ConductPage 原来那句「重启后生效」
 * 两种情况都说成了后者。
 *
 * 四态都要有话说，包括那个不该出现的组合：关掉了却还在跑，是真出了事，
 * 不能和「关着」长得一样。
 */
export function directorStateNote(
  enabled: boolean,
  running: boolean,
): { text: string; tone: Tone } {
  if (enabled && running) return { text: "指挥线程正在运行。", tone: "ok" };
  if (enabled && !running)
    return {
      text:
        "已保存为启用，但线程还没起来：本次写入若没有运行中的服务句柄，则下次启动生效；" +
        "否则请查日志里的「ConductDirector 没能启动」。",
      tone: "warn",
    };
  if (!enabled && running)
    return {
      text: "已保存为关闭，但线程仍在运行 —— 这不该发生，请查日志并重启服务。",
      tone: "crit",
    };
  return {
    text: "关着。读端点照常可用：已有 conduct 的状态仍然读得到，中止也仍然按得下。",
    tone: "neutral",
  };
}

/** supervised 撤销窗的那一行。窗口没开 ⇒ 空串（**不渲染**，不是渲染一个空框）。
 *
 * 这一档存在的全部理由就是「人不在场，但来得及后悔」。窗口开着而面板不说，
 * 等于这一档又回到了它被接上之前的样子 —— 那时它在行为上等于 autonomous，
 * 而 403 的文案还在推荐它。
 */
export function ignitionText(
  ig: { by?: string; remaining_s?: number; delay_s?: number } | null | undefined,
): string {
  const left = Math.ceil(Number(ig?.remaining_s ?? 0));
  if (!ig || !Number.isFinite(left) || left <= 0) return "";
  const who = ig.by || "agent";
  const mins = Math.floor(left / 60);
  const when = mins >= 1 ? `${mins} 分 ${left % 60} 秒` : `${left} 秒`;
  return `${who} 已批准，${when}后点火 —— 这段时间里 abort 能把它撤回。`;
}

/** 一条纲领的目标判据，在列表里显示成什么。
 *
 * 三态各有各的话说，而且 **unknown 不许显示成 `0/N`** —— 「一条都没满足」和
 * 「读不到」是两件事，前者要接着做，后者要去看为什么读不到。折叠成同一个数字，
 * 用户会把一次库故障读成「还早着呢」。
 */
export function goalProgressText(
  g:
    | { verdict?: string; satisfied?: number; total?: number; reason?: string }
    | null
    | undefined,
): { text: string; tone: Tone } {
  const v = String(g?.verdict ?? "");
  const sat = Number(g?.satisfied ?? 0);
  const total = Number(g?.total ?? 0);
  if (v === "done") return { text: `已达成 ${sat}/${total}`, tone: "ok" };
  if (v === "not_done") return { text: `${sat}/${total}`, tone: "info" };
  if (v === "unknown") {
    // 后端把三种 unknown 分得很清楚（没写判据 / 判据读不懂 / 证据读不到），
    // 而它们驱动的下一步不同：第一个要人去写判据，第二个要人去修判据，第三个
    // 要人去看库。全折成「读不到」等于把那三件事又合回一件。
    const why = String(g?.reason ?? "");
    if (why.includes("没写")) return { text: "未设判据", tone: "neutral" };
    if (why.includes("读不懂")) return { text: "判据有误", tone: "warn" };
    return { text: "读不到", tone: "neutral" };
  }
  return { text: "—", tone: "neutral" };
}
