// 一张卡片（旧版兼容格式 app.js 的 cardEl + capHtml + mkHtml）。
//
// memo + 按 id 订阅标记与选中：五千张的「全部帧」里打一个 ★，只有这一张重渲。
// 回调由 CardGrid 传入且是稳定引用，否则 memo 形同虚设。

import { memo } from "react";
import { Link } from "react-router-dom";
import clsx from "clsx";
import type { GalleryItem } from "@/lib/gallery/types";
import { RSYM, autoTags, caption, dirShort } from "@/lib/gallery/format";
import { ratingToggle } from "@/lib/gallery/marks";
import { seriesOf, shortFn } from "@/lib/gallery/series";
import { viewSearch } from "@/lib/gallery/route";
import { useMarksStore } from "./marksStore";
import { useSelectionStore } from "./selectionStore";
import { Chip, chipClass, ratingOnClass } from "./bits";

export interface CardProps {
  item: GalleryItem;
  j: number;
  /** 非目录视图里在说明行后面带 `[MMDD]`。 */
  showDir: boolean;
  current: boolean;
  lastBatch: string;
  numOf: (id: string) => string | null;
  onOpen: (j: number) => void;
  onPick: (j: number, shift: boolean, checked: boolean) => void;
}

const RATINGS: [number, string, string][] = [
  [1, "✓", "可用"],
  [2, "★", "重点"],
  [-1, "✗", "排除"],
];

export const GalleryCard = memo(function GalleryCard({
  item,
  j,
  showDir,
  current,
  lastBatch,
  numOf,
  onOpen,
  onPick,
}: CardProps) {
  const m = useMarksStore((s) => s.doc.items[item.id]);
  const series = useMarksStore((s) => s.doc.series);
  const updateMark = useMarksStore((s) => s.updateMark);
  const selected = useSelectionStore((s) => s.ids.has(item.id));
  const sids = seriesOf(series, item.id);
  const cap = caption(item);
  const tags = autoTags(item, lastBatch, numOf);
  const r = m?.r ?? 0;
  const spectrumLike = item.k === "s" || item.k === "g";

  return (
    <div
      data-gid={item.id}
      className={clsx(
        "relative flex flex-col overflow-hidden rounded-[3px] border border-mast-border bg-mast-panel",
        item.k === "g" && "col-span-2",
        r === 2 && "outline outline-[3px] outline-mast-warn [outline-offset:-1px]",
        r === 1 && "outline outline-2 outline-mast-auto [outline-offset:-1px]",
        r === -1 && "opacity-[.42] hover:opacity-90",
        selected && "bg-mast-accent-soft outline-dashed outline-2 outline-mast-accent [outline-offset:-2px]",
        current && "shadow-[0_0_0_3px_var(--mast-accent)]",
      )}
    >
      <button
        type="button"
        onClick={() => onOpen(j)}
        className={clsx("relative block cursor-zoom-in", spectrumLike ? "bg-white" : "bg-[#181614]")}
        title="点开大图（键盘翻页、打分）"
      >
        <img loading="lazy" src={item.th} alt="" className="block h-auto w-full" />
        {r !== 0 && (
          <span
            className={clsx(
              "absolute left-1 top-1 rounded-[3px] px-[5px] py-[3px] text-[15px] leading-none text-white",
              r === 2 ? "bg-mast-warn" : r === 1 ? "bg-mast-auto" : "bg-black/60",
            )}
          >
            {RSYM[String(r)]}
          </span>
        )}
        {item.li ? (
          <span className="absolute right-1 top-1 rounded-[2px] bg-[rgba(40,120,90,.85)] px-[5px] text-[10.5px] text-white">
            +dI/dV
          </span>
        ) : null}
      </button>

      <div className="px-[7px] pb-[3px] pt-[5px] font-mono text-[11.5px] leading-[1.38] text-mast-muted">
        <b className="font-semibold text-mast-text">{cap.num}</b> {cap.when}
        {showDir && (
          <span title={`所在目录 ${item.d}`}> [{dirShort(item.d)}]</span>
        )}
        {cap.lines.map((line, i) => (
          <div key={i}>{line}</div>
        ))}
        {tags.length > 0 && (
          <div>
            {tags.map((t) => (
              <Chip key={t.kind} kind={t.kind} title={t.title}>
                {t.text}
              </Chip>
            ))}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-[3px] px-1.5 pb-1.5 pt-0.5">
        <input
          type="checkbox"
          checked={selected}
          title="选中；按住 Shift 连选一段"
          onChange={() => {
            /* 状态由 onClick 写进 selectionStore（要读 shiftKey） */
          }}
          onClick={(e) => onPick(j, e.shiftKey, e.currentTarget.checked)}
          className="mr-[3px] h-[15px] w-[15px] cursor-pointer accent-[var(--mast-accent)]"
        />
        {RATINGS.map(([rv, sym, label]) => (
          <button
            key={rv}
            type="button"
            title={label}
            onClick={() => updateMark(item, { r: ratingToggle(r, rv) })}
            className={clsx(
              "rounded-[3px] border px-1.5 text-xs leading-[1.5]",
              r === rv ? ratingOnClass(rv) : "border-mast-border bg-mast-panel text-mast-text hover:border-mast-accent",
            )}
          >
            {sym}
          </button>
        ))}
        {(m?.tags ?? []).map((t) => (
          <Chip key={t} kind="user">
            {t}
          </Chip>
        ))}
        {sids.map((sid) => (
          <Link
            key={sid}
            to={{ search: `?${viewSearch({ v: "series", s: sid })}` }}
            className={chipClass("ser")}
            title="系列"
          >
            ▤ {series[sid]?.name || sid}
          </Link>
        ))}
        {m?.anchor && (
          <Chip kind="anc" title={`位置系于这张帧：${m.anchor.desc || ""}`}>
            ⌖ {shortFn(m.anchor.fn)}
          </Chip>
        )}
        {(m?.note || "").trim() && (
          <div className="line-clamp-3 w-full whitespace-pre-wrap text-[11.5px] text-mast-text">{m?.note}</div>
        )}
      </div>
    </div>
  );
});
