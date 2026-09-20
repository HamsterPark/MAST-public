// ════════════════════════════════════════════════════════════════════════════
// feedbackSubject — 「这条反馈是在说哪一页」。
//
// 反馈悬浮窗飘在**每一页**上，所以它必须自己看出来人在哪。这件事有过一次前科:
// agent 曾经是写死的，于是所有反馈全部被记成
// instrument_control —— 包括那些明明在说群聊 / 记录 / 视觉页的。
// 不崩、不报错，只是一张读起来完全正常、内容却错了的表。
//
// 和 lib/ws.ts ↔ hooks/useWsEvents.ts 同一个分法:这里没有 React、没有 JSX，
// 所以 node --test 直接跑得了（`frontend/test/feedbackSubject.test.ts`）。
// 留在 FeedbackFloat.tsx 里的话，测试根本 import 不进来（node 不剥 JSX），
// 而这几条判断恰恰是「错了也看不出来」的那一类。
// ════════════════════════════════════════════════════════════════════════════

/**
 * 路径前缀 → 反馈记在哪个「面」上。
 *
 * `page_*` 是**界面**，不是 LLM agent；feedback 表的 `agent` 列是自由文本标签，
 * 所以这是一次加宽，不是 schema 变更。
 *
 * 2026-08-06（#34 合并大标签）：原来的 /builder /cognition /admin 现在是
 * /skills/builder /records/memory /settings/admin。**光改字符串不够** ——
 * 见下面 {@link subjectForPath} 为什么必须按最长前缀匹配。
 *
 * 同时补齐了合并时暴露出来的几个从来没登记过的面（监控 / 环境历史 /
 * 仪器初始化 / 用量花销 / 光学台）—— 它们的反馈一直被记成 instrument_control。
 */
export const ROUTE_SUBJECT: { prefix: string; agent: string }[] = [
  { prefix: "/agents", agent: "orchestrator" },
  { prefix: "/qa", agent: "page_qa" },
  { prefix: "/skills/builder", agent: "page_builder" },
  { prefix: "/skills", agent: "page_skills" },
  { prefix: "/literature", agent: "literature" },
  { prefix: "/records/memory", agent: "page_cognition" },
  { prefix: "/records", agent: "page_records" },
  { prefix: "/wishlist", agent: "page_wishlist" },
  { prefix: "/settings/admin", agent: "page_admin" },
  { prefix: "/settings/setup", agent: "page_setup" },
  { prefix: "/settings/usage", agent: "page_usage" },
  { prefix: "/settings", agent: "page_settings" },
  { prefix: "/monitoring/env", agent: "page_env_history" },
  { prefix: "/monitoring", agent: "page_monitoring" },
  { prefix: "/experimental/optics", agent: "page_optics" },
  { prefix: "/experimental", agent: "page_experimental" },
];

/**
 * 这条路径上的反馈算在哪个面上。
 *
 * **最长前缀优先**，不是「表里第一条命中」。合并之后每个大标签底下都有二级页，
 * 而 `/skills/builder` 天然会先撞上 `/skills` —— 靠把长的写在前面来避开它，
 * 等于让一张表的**行序**承重，而行序是最容易在一次无关的整理里被打乱的东西，
 * 打乱之后也没有任何东西会报错。
 */
export function subjectForPath(pathname: string): string {
  let best: { prefix: string; agent: string } | null = null;
  for (const r of ROUTE_SUBJECT) {
    if (!pathname.startsWith(r.prefix)) continue;
    if (!best || r.prefix.length > best.prefix.length) best = r;
  }
  // "/" is the 仪器 Chat page — a real private chat with the IC agent.
  return best ? best.agent : "instrument_control";
}
