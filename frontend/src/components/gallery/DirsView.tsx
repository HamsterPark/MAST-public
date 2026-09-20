// 目录总览（旧版兼容格式 app.js 的 renderDays）：每个目录一张卡片 + 「最新一批」。

import { useMemo } from "react";
import { Link } from "react-router-dom";
import clsx from "clsx";
import { dirLabel, dirnameOf, fmtBShort, fmtT } from "@/lib/gallery/format";
import { newBatch, shortList, summariseDirs } from "@/lib/gallery/dirs";
import { seriesCountByDir } from "@/lib/gallery/series";
import { viewSearch } from "@/lib/gallery/route";
import type { GalleryModel } from "./useGalleryData";
import { useMarksStore } from "./marksStore";
import { Chip } from "./bits";
import { H1, Sub } from "./ListHeads";

const N = "font-mono text-xs leading-[1.45] text-mast-muted";

export function DirsView({ model }: { model: GalleryModel }) {
  const items = useMarksStore((s) => s.doc.items);
  const days = useMarksStore((s) => s.doc.days);
  const series = useMarksStore((s) => s.doc.series);

  const dirs = useMemo(
    () => summariseDirs(model.items, model.lastBatch, (id) => items[id]),
    [model, items],
  );
  const nser = useMemo(() => seriesCountByDir(series, model.byId), [series, model]);
  const nNew = useMemo(() => newBatch(model.items, model.lastBatch).length, [model]);
  const rootText = model.roots.map((r) => `${r.name}（${r.path}）`).join("、") || "—";

  return (
    <div>
      <H1>数据图库</H1>
      <Sub>
        {rootText}：{model.counts.f} 帧（其中 {model.counts.dup} 张是同一次扫描的重复保存，默认隐藏）·{" "}
        {model.counts.s} 条谱 · {model.counts.g} 个网格谱（缩略图更新于 {model.generated || "—"}）。
        进某个目录逐张看、打标记、圈系列；标记写进 marks.json / marks.md / marks.csv。
      </Sub>
      <div className="grid gap-2.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))" }}>
        {nNew > 0 && (
          <Link
            to={{ search: `?${viewSearch({ v: "new" })}` }}
            className="block rounded-[3px] border border-mast-border bg-mast-panel px-3 py-[9px] text-mast-text hover:border-mast-accent hover:no-underline"
          >
            <b className="text-[1.02rem]">最新一批 · {nNew} 个文件</b>
            <div className={N}>加入于 {model.lastBatch}</div>
            <div className={N}>跨目录集中看这次新到的数据</div>
          </Link>
        )}
        {dirs.map((s) => {
          const dv = days[s.d];
          return (
            <Link
              key={s.d}
              to={{ search: `?${viewSearch({ v: "dir", d: s.d })}` }}
              title={dirnameOf(s.sample.p) || s.d}
              className={clsx(
                "block rounded-[3px] border border-mast-border bg-mast-panel px-3 py-[9px] text-mast-text hover:border-mast-accent hover:no-underline",
                dv?.done && "border-l-4 border-l-mast-auto",
              )}
            >
              <b className="text-[1.02rem]">{dirLabel(s.d)}</b>{" "}
              {s.nw > 0 && <Chip kind="new">+{s.nw} 新</Chip>}
              {s.prefixes.slice(0, 3).map((p) => (
                <Chip key={p} kind="copies">
                  {p}
                </Chip>
              ))}
              <div className={N}>
                {s.f} 帧{s.dup ? `（重复 ${s.dup}）` : ""} · 原子分辨 {s.atom} · 超结构 {s.half}
                {s.s ? ` · 谱 ${s.s}` : ""}
                {s.g ? ` · 网格 ${s.g}` : ""}
              </div>
              <div className={N}>
                {s.t1 ? `${fmtT(s.t0)} → ${fmtT(s.t1)}` : "没有时刻"}
              </div>
              {s.biases.length > 0 && <div className={N}>偏压 {shortList(s.biases, fmtBShort)} V</div>}
              {s.widths.length > 0 && <div className={N}>帧宽 {shortList(s.widths, (w) => String(+w.toFixed(1)))} nm</div>}
              <div className={N}>
                {s.mk ? `已标记 ${s.mk}${s.star ? ` · ★ ${s.star}` : ""}` : "未标记"}
                {nser[s.d] ? ` · 系列 ${nser[s.d]}` : ""}
                {dv?.done && <b className="text-mast-auto"> · 已过完</b>}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
