// 列表视图：某目录 / 全部X / 最新一批 / 系列（旧版兼容格式 app.js 的 renderList + renderCards
// + closeLB 的重排逻辑）。
//
// 列表**只在**筛选变了、关大图、服务端文档整份换了（epoch）时重算——不随每一次打分
// 重算。原版如此，理由写在它的注释里：「筛选依赖标记：关掉大图时再重排，免得翻页时
// 列表跳」。在「隐藏 ✗」下按 3，那一张若立刻从列表里消失，下一张就跳到了手底下。

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { GalleryItem } from "@/lib/gallery/types";
import { applyFilters, facets, sanitizeFilters } from "@/lib/gallery/filters";
import { seriesOf } from "@/lib/gallery/series";
import type { ViewName } from "@/lib/gallery/route";
import type { GalleryModel } from "./useGalleryData";
import { useFiltersStore } from "./filtersStore";
import { useMarksStore } from "./marksStore";
import { useSelectionStore } from "./selectionStore";
import { FilterBar } from "./FilterBar";
import { CardGrid } from "./CardGrid";
import { Lightbox } from "./Lightbox";

export function scrollToGid(id: string): void {
  try {
    document.querySelector(`[data-gid="${CSS.escape(id)}"]`)?.scrollIntoView({ block: "center" });
  } catch {
    /* 老浏览器没有 CSS.escape：不滚就算了 */
  }
}

export function ListView({
  view,
  model,
  base,
  head,
  below,
}: {
  view: ViewName;
  model: GalleryModel;
  base: GalleryItem[];
  head?: ReactNode;
  below?: ReactNode;
}) {
  const F = useFiltersStore((s) => s.F);
  const tags = useMarksStore((s) => s.doc.tags);
  const epoch = useMarksStore((s) => s.epoch);
  const hasSel = useSelectionStore((s) => s.ids.size > 0);

  const fac = useMemo(() => facets(base), [base]);
  // 按**内容**稳定：标签表换了个新对象但内容没变（别处存了一次标签表）时，不该让列表
  // 重算、卡片网格回到前 150 张。
  const effKey = JSON.stringify(sanitizeFilters(F, fac, tags));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const effective = useMemo(() => sanitizeFilters(F, fac, tags), [effKey]);
  const [listVersion, setListVersion] = useState(0);

  const list = useMemo(() => {
    const doc = useMarksStore.getState().doc;
    return applyFilters(base, effective, {
      view,
      mark: (id) => doc.items[id],
      seriesOf: (id) => seriesOf(doc.series, id),
    });
    // listVersion / epoch 是「现在按当前标记重排一次」的信号，本身不参与计算。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, effective, view, listVersion, epoch]);

  // 选中的值在这个列表里不存在：退回「全部」并记住（原版在 filterBar 里改全局 F）。
  useEffect(() => {
    if (effective.b !== F.b || effective.w !== F.w || effective.pf !== F.pf || effective.tag !== F.tag) {
      useFiltersStore.getState().setF({ b: effective.b, w: effective.w, pf: effective.pf, tag: effective.tag });
    }
  }, [effective, F]);

  const [lb, setLb] = useState<number | null>(null);
  const [minShown, setMinShown] = useState(0);
  const listRef = useRef(list);
  listRef.current = list;
  const lbRef = useRef(lb);
  lbRef.current = lb;
  const pendingScroll = useRef<{ t: number; desc: boolean } | null>(null);

  const onOpen = useCallback((j: number) => setLb(j), []);
  const onPick = useCallback((j: number, shift: boolean, checked: boolean) => {
    useSelectionStore.getState().click(listRef.current, j, shift, checked);
  }, []);

  const closeLb = useCallback(() => {
    const cur = lbRef.current;
    const it = cur != null ? listRef.current[cur] : undefined;
    setLb(null);
    const Fnow = useFiltersStore.getState().F;
    if (Fnow.m || Fnow.tag) {
      pendingScroll.current = { t: it?.t ?? it?.mt ?? 0, desc: Fnow.sort === "desc" };
      setListVersion((v) => v + 1);
    } else if (it) {
      requestAnimationFrame(() => scrollToGid(it.id));
    }
  }, []);

  // 重排之后滚到时间上最接近刚才那张的位置（原版 closeLB）。
  useEffect(() => {
    const ps = pendingScroll.current;
    if (!ps) return;
    pendingScroll.current = null;
    const j = list.findIndex((x) => (ps.desc ? (x.t ?? 0) <= ps.t : (x.t ?? 0) >= ps.t));
    const target = list[j];
    if (target) {
      setMinShown(j + 1);
      requestAnimationFrame(() => requestAnimationFrame(() => scrollToGid(target.id)));
    }
  }, [list]);

  return (
    <div>
      {head}
      {below}
      <FilterBar view={view} effective={effective} fac={fac} tags={tags} list={list} total={base.length} />
      <CardGrid
        list={list}
        cw={effective.cw}
        showDir={view !== "dir"}
        currentId={lb != null ? (list[lb]?.id ?? null) : null}
        minShown={Math.max(minShown, lb != null ? lb + 1 : 0)}
        lastBatch={model.lastBatch}
        numOf={model.numOf}
        onOpen={onOpen}
        onPick={onPick}
      />
      {hasSel && <div className="h-40" aria-hidden />}
      {lb != null && list.length > 0 && (
        <Lightbox list={list} index={lb} model={model} onIndex={setLb} onClose={closeLb} />
      )}
    </div>
  );
}
