// filtersStore —— 数据图库的筛选条状态（旧版兼容格式 app.js 的全局 F）。
//
// 各视图共用一份（从某目录切到「全部帧」时偏压、标签这些筛选跟着走，原版如此），
// 存 localStorage["mast.gallery.filters"]，搜索词不存。读时校验在 lib/gallery/filters.ts。

import { create } from "zustand";
import {
  FILTERS_KEY,
  parseStoredFilters,
  serializeFilters,
  type Filters,
} from "@/lib/gallery/filters";

function readInitial(): Filters {
  try {
    return parseStoredFilters(localStorage.getItem(FILTERS_KEY));
  } catch {
    return parseStoredFilters(null);
  }
}

interface FiltersStore {
  F: Filters;
  setF: (patch: Partial<Filters>) => void;
}

export const useFiltersStore = create<FiltersStore>()((set, get) => ({
  F: readInitial(),
  setF: (patch) => {
    const F = { ...get().F, ...patch };
    set({ F });
    try {
      localStorage.setItem(FILTERS_KEY, serializeFilters(F));
    } catch {
      /* 记不住筛选也要能用 */
    }
  },
}));
