/**
 * `POST /api/settings` 的响应到底算不算「保存成功」—— 一份判据，一个地方。
 *
 * ── 为什么要有这个文件 ──
 * 2026-08-10 实测：「保存 Nanonis 连接配置」是个彻底的 no-op（五个字段不在写
 * schema 上，pydantic 默认 `extra='ignore'` 全丢），而按钮 toast 绿字「已保存」。
 * 直接原因是后端缺字段；但**它为什么没被任何人察觉**，是因为那个 onSuccess 只
 * 看了 `degraded`。这个端点的拒绝形状是 `ok:false, degraded:false`（端口越界、
 * scan_policy/zctrl_presets 结构非法、管理 PIN 不对），于是「拒绝」和「成功」在
 * 前端长得一模一样。
 *
 * 六个界面在 POST 这个端点，各自记得检查哪几个字段 —— 这个仓已经为「每一页各自
 * 记得」的接线付过好几次学费。所以判据不放在各页里，放在这里：一次 POST 之后只
 * 要问一句 `settingsWriteProblem(data)`，非 null 就是没存进去，字符串就是给操作
 * 员看的原因。`test/settingsWrite.test.ts` 里有一条闸门守着「谁 POST 了这个端点
 * 就必须用它」。
 *
 * 刻意**不**在这里 toast：有的调用方要先回滚乐观更新、有的要保留用户刚输入的
 * 内容，那是各页自己的事。这里只回答「成没成」。
 */

import type { components } from "@/api/schema";

/**
 * `POST /api/settings` 的请求体 —— **从后端 schema 生成，不是手抄的**。
 *
 * 后端 2026-08-10 把 `SettingsWriteRequest` 改成了 `extra='forbid'`：模型上没有
 * 的键不再被静默丢弃（那会返回 200 + `ok:true` 而值根本没写进去），而是 422。
 * 好处是「没落地的写」再也不会报成功；代价是**前端多送一个键会让整笔 422**。
 *
 * 所以调用方一律用这个类型，而不是 `Record<string, unknown>`：键名写错、后端
 * 把某个字段删了，都在 `npm run typecheck` 当场红，不用等到运行时。
 * 六处 POST 这个端点的地方以前各自是 `Record<string, unknown>`（其中两处还
 * `as never` 把类型整个抹掉）—— 那等于自己维护了一份不存在的名单。
 */
export type SettingsPatch = Partial<components["schemas"]["SettingsWriteRequest"]>;

export type SettingsWriteResult = {
  ok?: boolean;
  degraded?: boolean;
  /** 键 → 中文原因。非空 = 整笔被拒，一个字节都没写。 */
  rejected?: Record<string, string> | null;
  /** PIN 守卫的键（hardware_modules / advanced_capabilities / coarse_drive…）。 */
  pin_required?: boolean;
  /** PIN 失败时后端给的那句人话；也用于能力开关的重建说明。 */
  rebuild_note?: string | null;
};

/**
 * 写入没落地时返回一句中文原因；真的存进去了返回 null。
 *
 * 顺序是有意的：先 PIN（后端把原因放在 rebuild_note 里）、再 degraded（内核没
 * 接上，写入压根没到 store）、再 rejected（结构/取值非法，整笔不写）、最后兜底
 * 的 `ok !== true`。少一层都会让某一类失败静静变成「已保存」。
 */
export function settingsWriteProblem(
  res: SettingsWriteResult | null | undefined,
): string | null {
  if (!res) return "保存失败：没有收到内核的回应。";
  if (res.pin_required) return res.rebuild_note || "管理 PIN 不正确，未保存。";
  if (res.degraded) return "写入未生效（内核未接入）。";
  const reasons = Object.values(res.rejected ?? {}).filter(Boolean);
  if (reasons.length > 0) return reasons.join("；");
  if (res.ok !== true) return "保存被拒绝（内核未说明原因）。";
  return null;
}
