// selectionStore —— 数据图库里「选中了哪些」（旧版兼容格式 series.js 的 SEL）。
//
// 纯逻辑在 lib/gallery/selection.ts；这里只是 zustand 绑定。选中集合每次变都换新
// Set，卡片按 id 订阅 `s.ids.has(id)`，于是只有状态真的变了的卡片重渲。
// 视图切换时由 GalleryApp 清空（原版 route() 里的 selClear）。

import { create } from "zustand";
import { pickClick, pickKey, type HasId, type SelectionState } from "@/lib/gallery/selection";

interface SelectionStore extends SelectionState {
  clear: () => void;
  setOne: (id: string, on: boolean) => void;
  click: (list: readonly HasId[], j: number, shift: boolean, checked: boolean) => void;
  key: (list: readonly HasId[], j: number, code: "KeyS" | "BracketLeft" | "BracketRight") => void;
  removeMany: (ids: readonly string[]) => void;
}

export const useSelectionStore = create<SelectionStore>()((set, get) => ({
  ids: new Set<string>(),
  lastId: null,
  startId: null,
  clear: () => set({ ids: new Set(), lastId: null, startId: null }),
  setOne: (id, on) => {
    const ids = new Set(get().ids);
    if (on) ids.add(id);
    else ids.delete(id);
    set({ ids });
  },
  click: (list, j, shift, checked) => set((st) => pickClick(st, list, j, shift, checked)),
  key: (list, j, code) => set((st) => pickKey(st, list, j, code)),
  removeMany: (drop) => {
    const ids = new Set(get().ids);
    for (const id of drop) ids.delete(id);
    set({ ids });
  },
}));
