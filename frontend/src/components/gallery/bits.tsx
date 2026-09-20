// 数据图库的小件：标签色块、键帽、评级按钮的配色、未判定原因的中文。
//
// 配色一律走 --mast-* token（设计 §4.4 末尾）：原子 = auto、超结构 = dream、
// 未扫完/重复/片段 = warn、新 = info、系列 = accent、系定 = info；
// ★ = warn、✓ = auto、✗ = danger。大图固定暗底，所以它另有一套 dark* 常量。

import type { ReactNode } from "react";
import clsx from "clsx";

export type ChipKind =
  | "new" | "atom" | "half" | "part" | "dup" | "seg" | "copies" | "gridpart" | "li"
  | "user" | "ser" | "anc";

const CHIP: Record<ChipKind, string> = {
  new: "bg-mast-info-bg text-mast-info",
  atom: "bg-mast-auto-bg text-mast-auto",
  half: "bg-[color-mix(in_srgb,var(--mast-dream)_16%,transparent)] text-mast-dream",
  part: "bg-mast-warn-bg text-mast-warn",
  dup: "bg-mast-warn-bg text-mast-warn",
  seg: "bg-mast-warn-bg text-mast-warn",
  gridpart: "bg-mast-warn-bg text-mast-warn",
  copies: "border border-mast-border bg-mast-panel-2 text-mast-muted",
  li: "bg-mast-auto-bg text-mast-auto",
  user: "border border-mast-border bg-mast-panel-2 text-mast-text",
  ser: "bg-mast-accent-soft text-mast-accent hover:underline",
  anc: "bg-mast-info-bg text-mast-info",
};

export function chipClass(kind: ChipKind, extra?: string): string {
  return clsx(
    "mr-[3px] mt-px inline-block rounded-[2px] px-1 font-sans text-[10.5px] leading-[1.5]",
    CHIP[kind],
    extra,
  );
}

export function Chip({ kind, title, children }: { kind: ChipKind; title?: string; children: ReactNode }) {
  return (
    <span className={chipClass(kind)} title={title}>
      {children}
    </span>
  );
}

export function Kbd({ children, dark }: { children: ReactNode; dark?: boolean }) {
  return (
    <kbd className={clsx("ml-[3px] font-mono text-[10px]", dark ? "text-[#9aa3ad]" : "text-mast-faint")}>
      {children}
    </kbd>
  );
}

/** 卡片上 ✓★✗ 三个小按钮「按下」时的配色。 */
export function ratingOnClass(r: number): string {
  if (r === 2) return "border-mast-warn bg-mast-warn text-white";
  if (r === 1) return "border-mast-auto bg-mast-auto text-white";
  return "border-mast-danger bg-mast-danger text-white";
}

/** 大图侧栏（暗底）的评级按钮「按下」配色。 */
export function darkRatingOnClass(r: number): string {
  if (r === 2) return "border-[#f0b545] bg-[#b7791f]";
  if (r === 1) return "border-[#5fc39d] bg-[#1d7a5f]";
  return "border-[#e58585] bg-[#8b2c2c]";
}

export const DARK_BTN =
  "rounded-[3px] border border-[#444] bg-[#2a2e33] px-[9px] py-[3px] text-[#eee] hover:border-[#a897ee] disabled:opacity-50";

export const SMALL_SELECT =
  "rounded-[3px] border border-mast-border bg-mast-panel px-1 py-0.5 text-[13px] text-mast-text";

export const SMALL_BTN =
  "rounded-[3px] border border-mast-border bg-mast-panel px-[7px] py-px text-[13px] text-mast-text hover:border-mast-accent disabled:opacity-50";

/** 原子相判据「判不了」的原因（vision/atomic_phase.py 的 reasons）→ 给人看的话。 */
const AR_TEXT: Record<string, string> = {
  scale_gate: "像素太粗（> 0.05 nm/px），这个尺度上晶格分辨不出来",
  scale_reduced: "0.02–0.05 nm/px 过渡带，证据强度不够下正面结论",
  unknown_pixel_size: "不知道像素尺寸",
  incomplete_frame: "帧不完整",
  insufficient_data: "有效数据太少",
  too_few_periods: "视野里装不下足够多的周期",
  dead_flat: "图像几乎是平的",
  dependency_unavailable: "判据依赖不可用",
  too_few_rows: "有效行太少",
};

export function arText(ar: string | null | undefined): string {
  if (!ar) return "";
  return AR_TEXT[ar] ? `${AR_TEXT[ar]}（${ar}）` : ar;
}
