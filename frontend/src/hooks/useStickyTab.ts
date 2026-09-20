// ════════════════════════════════════════════════════════════════════════════
// useStickyTab — 记住一个栏目上次停在哪个子页。
//
// 纯逻辑在 `lib/stickyTab.ts`（读时校验那一条是要害）；这里只剩 React 绑定。
// ════════════════════════════════════════════════════════════════════════════

import { useCallback, useState } from "react";
import { readStickyTab, writeStickyTab } from "@/lib/stickyTab";

/**
 * `useState` 的替身，值会跨导航活下来。
 *
 * @param page   存储键（`mast.subtab.<page>`）。嵌套的子页用点分层，
 *               例如 `"agents"` 与 `"agents.chat.mode"` —— 让键长得像它在
 *               界面里的位置，下一个人打开 devtools 才认得出哪个是哪个。
 * @param valid  这个栏目**当前**认得的全部 id。它就是 tab 列表本身，
 *               别另抄一份 —— 抄的那份会和 tab 列表漂开，而漂开的症状是
 *               「某个子页记不住」，一个没人会去查存储键的症状。
 * @param fallback 没有记忆 / 记忆已过期时用哪个。
 *
 * 初值走惰性 `useState(fn)`：读 localStorage 是同步 I/O，写成
 * `useState(readStickyTab(...))` 会在**每一次渲染**都读一遍。
 */
export function useStickyTab<T extends string>(
  page: string,
  valid: readonly T[],
  fallback: T,
): [T, (id: T) => void] {
  const [tab, setTabState] = useState<T>(() => readStickyTab(page, valid, fallback));
  const setTab = useCallback(
    (id: T) => {
      setTabState(id);
      writeStickyTab(page, id);
    },
    [page],
  );
  return [tab, setTab];
}
