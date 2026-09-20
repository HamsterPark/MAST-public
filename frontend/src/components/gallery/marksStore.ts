// ════════════════════════════════════════════════════════════════════════════
// marksStore —— 数据图库的标记（设计 D12；旧版兼容格式 app.js 的 Store 逐项移植）。
//
// 服务端 marks.json 是唯一真源；这里是它在浏览器里的镜像 + 存盘驱动：
//   * 每次改动先落到本地文档（界面立刻变），同时进 pending 队列；
//   * 350 ms 去抖后发一批；**一次只发一个请求**，服务端就按顺序收到；
//   * 失败的一批放回队列（不盖掉路上又改的新版本），5 s 后补发；
//   * pending 同时写 localStorage["mast.gallery.pending"]——关掉页面、服务重启、
//     网线掉了，下次打开先把它应用上再补发；
//   * 回到页面（visibilitychange）且没有待存改动时，按 rev 看服务端变没变过。
//
// 卡片按 id 订阅（`useMarksStore(s => s.doc.items[id])`）：文档每次改动都换新对象，
// 但没被改的那一条仍是同一个对象引用，所以改一条只重渲一张卡。
//
// `epoch` 只在**整份文档被服务端版本替换**时加一（加载、别的标签页改过、导入之后）。
// 列表视图据此重算筛选；本页自己的逐条改动不动它——原版在翻页、打分的过程中不重排
// 列表（否则按 ✗ 之后那一张立刻从「隐藏 ✗」的列表里消失，下一张跳到手底下）。
// ════════════════════════════════════════════════════════════════════════════

import { create } from "zustand";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import type { DirMark, GalleryItem, Mark, MarksDoc, Patch, Series } from "@/lib/gallery/types";
import {
  PENDING_KEY,
  applyPatchLocal,
  composeMark,
  emptyPatch,
  mergePatch,
  normaliseDoc,
  parsePending,
  patchEmpty,
  requeue,
  toggledTags,
  tsClock,
  withDir,
  withMark,
  withSeries,
} from "@/lib/gallery/marks";
import { nowStr } from "@/lib/gallery/format";

/**
 * loading：还没拿到服务端文档 · idle：连上了、还没存过 · pending：保存中 ·
 * ok：已存盘 · bad：没存上（改动在浏览器里，会补存）· offline：标记服务不可用
 */
export type SaveState = "loading" | "idle" | "pending" | "ok" | "bad" | "offline";

interface MarksState {
  loaded: boolean;
  online: boolean;
  doc: MarksDoc;
  pending: Patch;
  saveState: SaveState;
  savedAt: string;
  busy: boolean;
  again: boolean;
  epoch: number;
  init: () => Promise<void>;
  setMark: (id: string, m: Mark | null) => void;
  updateMark: (it: GalleryItem, patch: Partial<Mark>) => void;
  toggleTag: (it: GalleryItem, t: string, on?: boolean) => void;
  setSeries: (sid: string, s: Series | null) => void;
  setDay: (d: string, v: DirMark) => void;
  setTags: (tags: string[]) => void;
  flush: () => Promise<void>;
  refresh: () => Promise<boolean>;
  reload: () => Promise<void>;
}

const nextTs = tsClock();
let timer: ReturnType<typeof setTimeout> | null = null;
let initPromise: Promise<void> | null = null;
/** 服务没起来时的重试定时器；只留一个，别随组件重复挂载叠出一串。 */
let retryTimer: ReturnType<typeof setTimeout> | null = null;

function lsGet(k: string): string | null {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
}
function lsSet(k: string, v: string): void {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* 存不了就算了：隐私模式里仍要能用 */
  }
}
function lsDel(k: string): void {
  try {
    localStorage.removeItem(k);
  } catch {
    /* 同上 */
  }
}

/** 把一批改动并进 localStorage 里「没存上的改动」。 */
function persistPending(p: Patch): void {
  if (patchEmpty(p)) return;
  lsSet(PENDING_KEY, JSON.stringify(mergePatch(parsePending(lsGet(PENDING_KEY)) ?? emptyPatch(), p)));
}

async function fetchDoc(): Promise<MarksDoc | null> {
  try {
    const { data, error } = await api.GET("/api/gallery/marks");
    if (error || !data || (data as { degraded?: boolean }).degraded) return null;
    return normaliseDoc(data);
  } catch {
    return null;
  }
}

/** 发出去的请求体：tags 为 null 时不带这个键（它的意思是「标签表没改」）。 */
function requestBody(p: Patch): components["schemas"]["GalleryMarksPatch"] {
  const body: Record<string, unknown> = { items: p.items, series: p.series, days: p.days };
  if (p.tags) body.tags = p.tags;
  // 生成的 GalleryMark 把带默认值的字段标成必填；删除条目 {del, ts} 与只含部分键的标记
  // 在线上都是合法的（服务端 exclude_unset）。见 lib/gallery/types.ts 顶部。
  return body as unknown as components["schemas"]["GalleryMarksPatch"];
}

export const useMarksStore = create<MarksState>()((set, get) => {
  const schedule = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => void get().flush(), 350);
    persistPending(get().pending);
    set({ saveState: "pending" });
  };

  return {
    loaded: false,
    online: false,
    doc: normaliseDoc(null),
    pending: emptyPatch(),
    saveState: "loading",
    savedAt: "",
    busy: false,
    again: false,
    epoch: 0,

    init: () => {
      if (initPromise) return initPromise;
      if (get().online) return Promise.resolve();
      initPromise = (async () => {
        const fetched = await fetchDoc();
        if (!fetched) {
          set({ loaded: true, online: false, saveState: "offline" });
          initPromise = null;
          // 服务没起来：5 s 后再试。期间的改动照样进队列、写 localStorage。
          if (!retryTimer) {
            retryTimer = setTimeout(() => {
              retryTimer = null;
              void get().init();
            }, 5000);
          }
          return;
        }
        let doc = fetched;
        let pending = get().pending;
        const stored = parsePending(lsGet(PENDING_KEY));
        if (stored) {
          // 上次没存上的改动：先应用到界面上，再并进队列补发。
          doc = applyPatchLocal(doc, stored);
          pending = mergePatch(stored, pending);
        }
        if (!patchEmpty(get().pending)) doc = applyPatchLocal(doc, get().pending);
        set((s) => ({
          doc,
          pending,
          loaded: true,
          online: true,
          epoch: s.epoch + 1,
          savedAt: fetched.updated ?? "",
          saveState: patchEmpty(pending) ? (fetched.updated ? "ok" : "idle") : "pending",
        }));
        if (!patchEmpty(pending)) void get().flush();
      })();
      return initPromise;
    },

    setMark: (id, m) => {
      const entry = m ?? { del: true as const, ts: nextTs() };
      set((s) => ({
        doc: withMark(s.doc, id, m),
        pending: { ...s.pending, items: { ...s.pending.items, [id]: entry } },
      }));
      schedule();
    },

    updateMark: (it, patch) => {
      const m = composeMark(get().doc.items[it.id], patch, it, nowStr(), nextTs());
      get().setMark(it.id, m);
    },

    toggleTag: (it, t, on) => {
      const tags = toggledTags(get().doc.items[it.id]?.tags ?? [], t, on);
      if (tags) get().updateMark(it, { tags });
    },

    setSeries: (sid, s) => {
      const stamped = s ? { ...s, t: nowStr(), ts: nextTs() } : null;
      const entry = stamped && stamped.ids.length ? stamped : { del: true as const, ts: nextTs() };
      set((st) => ({
        doc: withSeries(st.doc, sid, stamped),
        pending: { ...st.pending, series: { ...st.pending.series, [sid]: entry } },
      }));
      schedule();
    },

    setDay: (d, v) => {
      const vv: DirMark = { ...v, ts: nextTs() };
      set((st) => ({
        doc: withDir(st.doc, d, vv),
        pending: { ...st.pending, days: { ...st.pending.days, [d]: vv } },
      }));
      schedule();
    },

    setTags: (tags) => {
      set((st) => ({ doc: { ...st.doc, tags }, pending: { ...st.pending, tags } }));
      schedule();
    },

    flush: async () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (get().busy) {
        set({ again: true });
        return;
      }
      const body = get().pending;
      if (patchEmpty(body)) return;
      set({ pending: emptyPatch(), busy: true, saveState: "pending" });
      persistPending(body);
      try {
        const { data, error } = await api.POST("/api/gallery/marks/patch", { body: requestBody(body) });
        if (error || !data || !data.ok) throw new Error(data?.detail || "patch failed");
        set((s) => ({
          doc: { ...s.doc, rev: data.rev, updated: data.updated },
          saveState: "ok",
          savedAt: data.updated,
          online: true,
        }));
        if (patchEmpty(get().pending)) lsDel(PENDING_KEY);
      } catch {
        set((s) => ({ pending: requeue(s.pending, body), saveState: "bad" }));
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => void get().flush(), 5000);
      } finally {
        set({ busy: false });
        if (get().again) {
          set({ again: false });
          void get().flush();
        }
      }
    },

    refresh: async () => {
      const st = get();
      if (!st.online || st.busy || !patchEmpty(st.pending)) return false;
      const d = await fetchDoc();
      if (!d || d.rev === get().doc.rev) return false;
      if (!patchEmpty(get().pending) || get().busy) return false; // 取的路上又改了：别覆盖
      set((s) => ({ doc: d, savedAt: d.updated ?? s.savedAt, epoch: s.epoch + 1 }));
      return true;
    },

    reload: async () => {
      const d = await fetchDoc();
      if (!d) return;
      set((s) => ({
        doc: applyPatchLocal(d, s.pending),
        savedAt: d.updated ?? s.savedAt,
        online: true,
        epoch: s.epoch + 1,
      }));
    },
  };
});

/** 顶栏的保存状态文字（原版 setSave）。 */
export function saveText(state: SaveState, savedAt: string): string {
  switch (state) {
    case "ok":
      return `已存盘 ${(savedAt || "").slice(11)} → marks.json`;
    case "pending":
      return "保存中…";
    case "bad":
      return "⚠ 没存上，改动先记在浏览器，服务恢复后自动补存";
    case "offline":
      return "⚠ 标记服务不可用：改动先记在浏览器，连上后自动补存";
    case "idle":
      return "已连上存盘服务 → marks.json";
    default:
      return "读取标记…";
  }
}
