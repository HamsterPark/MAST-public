// ════════════════════════════════════════════════════════════════════════════
// 数据图库（实验记录 → 数据图库，/records/gallery）。
//
// 数据来自 /api/gallery/*；「出图」一栏从用户标记与系列生成图表。
//
// 视图在 URL query 里（lib/gallery/route.ts）：深链可收藏、前进后退可用。URL 没有 `v`
// 时回到上次停的视图（localStorage["mast.gallery.view"]，读时校验）。
//
// 这一层管的是跨视图的事：顶栏、标记加载与「回到页面时拉一次」、构建完成的那一刻让
// 索引与标记重拉、出图任务结束的提示、视图切换时清空选中、底部操作条、提示条。
//
// ⚠️ 视图主体（body）按「视图 + 索引」memo。这一层订阅了保存状态、构建状态——每打
// 一次分、构建时每 1.5 s 轮询一次都会重渲。主体若跟着重建，列表视图拿到的就是一个
// 新的 base 数组：卡片网格回到前 150 张、滚动位置丢失、而且会在打分过程中按新标记
// 重排（原版刻意不这样做）。所以主体只在视图或索引真的变了时才重建。
// ════════════════════════════════════════════════════════════════════════════

import { useEffect, useMemo, useRef } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { newBatch } from "@/lib/gallery/dirs";
import { parseView, navId, viewKey, viewSearch, type GalleryView } from "@/lib/gallery/route";
import { GALLERY_KEYS, useGalleryModel, useGalleryStatus } from "./useGalleryData";
import { saveText, useMarksStore } from "./marksStore";
import { useSelectionStore } from "./selectionStore";
import { DirsView } from "./DirsView";
import { ListView } from "./ListView";
import { AllHead, DirHead, NewHead } from "./ListHeads";
import { SeriesView } from "./SeriesView";
import { MarkedView } from "./MarkedView";
import { SetupView } from "./SetupView";
import { SelectionBar } from "./SelectionBar";
import { GalleryToast, toast } from "./toast";
import { FiguresView } from "./figures/FiguresView";
import { FigureJobChip } from "./figures/figureJob";

const LAST_VIEW_KEY = "mast.gallery.view";

function readLastView(): string | null {
  try {
    const raw = localStorage.getItem(LAST_VIEW_KEY);
    return raw && parseView(new URLSearchParams(raw)) ? raw : null;
  } catch {
    return null;
  }
}

const NAV: { id: string; label: string; view: GalleryView }[] = [
  { id: "dirs", label: "目录", view: { v: "dirs" } },
  { id: "all-f", label: "全部帧", view: { v: "all", k: "f" } },
  { id: "all-s", label: "全部谱", view: { v: "all", k: "s" } },
  { id: "all-g", label: "网格谱", view: { v: "all", k: "g" } },
  { id: "marked", label: "已标记", view: { v: "marked" } },
  { id: "figures", label: "出图", view: { v: "figures" } },
  { id: "setup", label: "数据根与构建", view: { v: "setup" } },
];

export function GalleryApp() {
  const [sp, setSp] = useSearchParams();
  const parsed = parseView(sp);
  const vs = parsed ? viewSearch(parsed) : "";
  const view = useMemo<GalleryView>(() => parseView(new URLSearchParams(vs)) ?? { v: "dirs" }, [vs]);
  const vk = viewKey(view);
  const qc = useQueryClient();

  // URL 没有视图：回到上次停的地方（替换历史，不多一步后退）。
  useEffect(() => {
    if (!vs) setSp(readLastView() ?? viewSearch({ v: "dirs" }), { replace: true });
  }, [vs, setSp]);
  useEffect(() => {
    if (!vs) return;
    try {
      localStorage.setItem(LAST_VIEW_KEY, vs);
    } catch {
      /* 记不住也能用 */
    }
  }, [vs]);

  const { q, model } = useGalleryModel();
  const status = useGalleryStatus();

  // 标记：第一次进页面加载；回到页面时若服务端变过（别的标签页改了）就拉。
  const init = useMarksStore((s) => s.init);
  useEffect(() => {
    void init();
    void useMarksStore.getState().refresh();
  }, [init]);
  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === "visible") void useMarksStore.getState().refresh();
    };
    const onUnload = () => void useMarksStore.getState().flush();
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("beforeunload", onUnload);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("beforeunload", onUnload);
      void useMarksStore.getState().flush();
    };
  }, []);

  // 构建从「在跑」变成「不在跑」的那一刻：索引与标记重拉。
  const wasRunning = useRef<boolean | null>(null);
  const running = status.data?.running;
  useEffect(() => {
    if (running === undefined) return;
    if (wasRunning.current && !running) {
      void qc.invalidateQueries({ queryKey: GALLERY_KEYS.index });
      void useMarksStore.getState().reload();
      const d = status.data;
      toast(
        d?.phase === "error"
          ? `图库构建出错：${d.message || d.detail || ""}`
          : d?.phase === "cancelled"
            ? "图库构建已取消"
            : `图库已更新：新 ${d?.n_new ?? 0} · 变动 ${d?.n_changed ?? 0} · 失败 ${d?.n_failed ?? 0}`,
        d?.phase === "error" ? "err" : "ok",
      );
    }
    wasRunning.current = running;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  // 视图变了就清空选中（原版 route() 的 selClear）。
  useEffect(() => {
    useSelectionStore.getState().clear();
  }, [vk]);

  // 顶栏高度 → CSS 变量，筛选条吸顶在它下面。
  const rootRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const head = headRef.current;
    const root = rootRef.current;
    if (!head || !root || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => root.style.setProperty("--gallery-head", `${head.offsetHeight}px`));
    ro.observe(head);
    return () => ro.disconnect();
  }, []);

  const saveState = useMarksStore((s) => s.saveState);
  const savedAt = useMarksStore((s) => s.savedAt);
  const nMarked = useMarksStore((s) => Object.keys(s.doc.items).length + Object.keys(s.doc.series).length);
  const active = navId(view);

  const indexData = q.data;
  const body = useMemo(() => {
    if (view.v === "setup") return <SetupView />;
    // 出图页不依赖本地索引是否建好：产物列表与任务状态走自己的端点。
    if (view.v === "figures") return <FiguresView />;
    if (q.isPending) return <Spinner label="读取图库索引…" />;
    if (q.isError) return <ErrorNote error={q.error} />;
    if (indexData?.degraded) {
      return (
        <div className="rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2.5 text-sm text-mast-warn">
          图库后端不可用：{indexData.detail || "未知原因"}。
          <Link to={{ search: `?${viewSearch({ v: "setup" })}` }} className="ml-1 underline">
            去「数据根与构建」看看
          </Link>
        </div>
      );
    }
    if (!model || !model.built) {
      return (
        <div className="space-y-2">
          <EmptyNote label="还没有构建过图库索引。" />
          <div className="text-sm text-mast-muted">
            先到
            <Link to={{ search: `?${viewSearch({ v: "setup" })}` }} className="mx-1 text-mast-accent hover:underline">
              数据根与构建
            </Link>
            加一个数据根（放 .sxm / .dat / .3ds 的目录），再点「增量更新」。
          </div>
        </div>
      );
    }
    switch (view.v) {
      case "dirs":
        return <DirsView model={model} />;
      case "dir": {
        const base = model.items.filter((it) => it.d === view.d);
        return (
          <ListView key={vk} view="dir" model={model} base={base} head={<DirHead key={view.d} d={view.d} base={base} />} />
        );
      }
      case "all":
        return (
          <ListView
            key={vk}
            view="all"
            model={model}
            base={model.items.filter((it) => it.k === view.k)}
            head={<AllHead k={view.k} />}
          />
        );
      case "new":
        return (
          <ListView
            key={vk}
            view="new"
            model={model}
            base={newBatch(model.items, model.lastBatch)}
            head={<NewHead lastBatch={model.lastBatch} />}
          />
        );
      case "series":
        return <SeriesView key={vk} sid={view.s} model={model} />;
      case "marked":
        return <MarkedView key={vk} model={model} />;
    }
    return null;
  }, [view, vk, q.isPending, q.isError, q.error, indexData, model]);

  return (
    <div ref={rootRef} className="relative pb-20">
      <header
        ref={headRef}
        className="sticky top-0 z-20 flex flex-wrap items-center gap-x-[18px] gap-y-1 border-b border-mast-border bg-mast-bg py-1.5"
      >
        <div>
          <Link to={{ search: `?${viewSearch({ v: "dirs" })}` }} className="text-[1.05rem] font-bold text-mast-text hover:no-underline">
            数据图库
          </Link>
          <span className="ml-2 text-xs text-mast-faint">缩略图 {model?.generated || "—"}</span>
        </div>
        <nav className="flex flex-wrap gap-0.5">
          {NAV.map((n) => (
            <Link
              key={n.id}
              to={{ search: `?${viewSearch(n.view)}` }}
              className={clsx(
                "rounded-[3px] px-[9px] py-[3px] text-sm hover:no-underline",
                active === n.id ? "bg-mast-text text-mast-bg" : "text-mast-text hover:bg-mast-panel-2",
              )}
            >
              {n.label}
              {n.id === "marked" && <b className="ml-1">{nMarked}</b>}
            </Link>
          ))}
        </nav>
        {status.data?.running && (
          <Link
            to={{ search: `?${viewSearch({ v: "setup" })}` }}
            className="rounded-mast-badge bg-mast-info-bg px-1.5 text-xs text-mast-info hover:no-underline"
            title="图库正在构建"
          >
            构建中 {status.data.done}/{status.data.total}
          </Link>
        )}
        <FigureJobChip />
        <div
          className={clsx(
            "ml-auto text-xs",
            saveState === "ok" && "text-mast-auto",
            (saveState === "bad" || saveState === "offline") && "font-semibold text-mast-danger",
            (saveState === "idle" || saveState === "pending" || saveState === "loading") && "text-mast-muted",
          )}
        >
          {saveText(saveState, savedAt)}
        </div>
      </header>

      <div className="pt-3">{body}</div>

      {model && <SelectionBar view={view} model={model} />}
      <GalleryToast />
    </div>
  );
}
