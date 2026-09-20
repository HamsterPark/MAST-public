// 卡片网格 + 增量渲染（旧版兼容格式 app.js 的 renderCards / moreCards / pump）。
//
// 每批 150 张，哨兵离视口底部 1500 px 以内就再加一批。IntersectionObserver 只在相交
// 状态**变化**时回调：加完一批后哨兵可能仍在 1500 px 以内，所以每次「已显示」变化都
// 重新 observe 一次，让它立刻按当前状态再报一次——否则一屏很高的窗口会停在第一批。
//
// 滚动容器是 AppLayout 的 <main>，不是 window；observer 的 root 用视口即可（相交按
// 视口算，main 滚动时哨兵在视口里移动）。

import { useEffect, useRef, useState } from "react";
import type { GalleryItem } from "@/lib/gallery/types";
import { GalleryCard } from "./GalleryCard";

const BATCH = 150;

export function CardGrid({
  list,
  cw,
  showDir,
  currentId,
  minShown,
  lastBatch,
  numOf,
  onOpen,
  onPick,
}: {
  list: GalleryItem[];
  cw: number;
  showDir: boolean;
  currentId: string | null;
  /** 至少渲染到第几张（大图翻到后面、关大图后要滚到某一张时）。 */
  minShown: number;
  lastBatch: string;
  numOf: (id: string) => string | null;
  onOpen: (j: number) => void;
  onPick: (j: number, shift: boolean, checked: boolean) => void;
}) {
  // 换了列表就从第一批重新开始（在渲染期间重置，避免先闪一帧旧的数量）。
  const [st, setSt] = useState({ list, shown: BATCH });
  if (st.list !== list) setSt({ list, shown: BATCH });
  const shown = st.list === list ? st.shown : BATCH;
  const need = Math.min(list.length, Math.max(shown, minShown));

  const sentinel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = sentinel.current;
    if (!el || need >= list.length) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSt((s) => (s.list === list ? { list, shown: Math.min(list.length, s.shown + BATCH) } : s));
        }
      },
      { rootMargin: "0px 0px 1500px 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [list, need]);

  return (
    <>
      <div
        className="mt-3 grid gap-2.5"
        style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${cw}px, 1fr))` }}
      >
        {list.slice(0, need).map((it, j) => (
          <GalleryCard
            key={it.id}
            item={it}
            j={j}
            showDir={showDir}
            current={it.id === currentId}
            lastBatch={lastBatch}
            numOf={numOf}
            onOpen={onOpen}
            onPick={onPick}
          />
        ))}
      </div>
      <div ref={sentinel} className="mx-auto my-[18px] text-center text-sm text-mast-muted">
        {need < list.length
          ? `已显示 ${need} / ${list.length}，往下滚继续`
          : list.length
            ? ""
            : "没有符合条件的文件"}
      </div>
    </>
  );
}
