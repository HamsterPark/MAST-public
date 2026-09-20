// 谱 ↔ 前后帧对照（旧版兼容格式 context.js 的 panelHtml / ctxStage / paintSide）。
//
// 打开一条谱（或网格谱）时，大图区依次是：谱 | 前一张帧 | 后一张帧。前一张 = 谱开始
// 之前最后保存的帧；后一张 = 谱结束之后才开始扫描的第一张（跨着谱的帧、重复保存的帧
// 跳过）。帧上画出谱的位置（网格画范围），视野外改黄色虚线。◀ 更早 / 更晚 ▶ 换帧，
// 「⌖ 系于此帧」把位置记到那张帧上（大图里的 A / D 键同样生效）。
//
// 两侧当前摆的是哪张帧由 Lightbox 持有（键盘要用），这里只渲染。

import { Link } from "react-router-dom";
import clsx from "clsx";
import type { GalleryItem } from "@/lib/gallery/types";
import { fmtB, fmtTs, nm, num } from "@/lib/gallery/format";
import { fStart, inside, itemGeometry, relation, tBeg, tEnd, type UV } from "@/lib/gallery/context";
import { viewSearch } from "@/lib/gallery/route";
import { SpectrumChart } from "@/components/records/SpectrumChart";
import { FrameWithMarks } from "./FrameWithMarks";
import { DARK_BTN, Kbd } from "./bits";

export type Side = "prev" | "next";

/** 一侧的帧面板。系列页（亮底）与大图（暗底）共用。 */
export function FramePanel({
  side,
  frame,
  t0,
  t1,
  noun,
  pts,
  poly,
  anchoredId,
  posText,
  onStep,
  onAnchor,
  dark,
}: {
  side: Side;
  frame: GalleryItem | undefined;
  t0: number;
  t1: number;
  noun: string;
  pts: UV[];
  poly: UV[] | null;
  anchoredId: string | null | undefined;
  /** 脚注里「帧内位置」那一截（单条谱给坐标，系列给「n/N 条在视野内」）。 */
  posText: (frame: GalleryItem) => React.ReactNode;
  onStep: (side: Side, step: number) => void;
  onAnchor: ((side: Side) => void) | null;
  dark?: boolean;
}) {
  const lab = side === "prev" ? "前一张帧" : "后一张帧";
  const btn = dark ? DARK_BTN : "rounded-[3px] border border-mast-border bg-mast-panel px-[7px] py-px text-[13px] hover:border-mast-accent";
  const stepButtons = (anchor: React.ReactNode) => (
    <div className="flex flex-wrap gap-1">
      <button type="button" className={btn} onClick={() => onStep(side, -1)}>
        ◀ 更早
      </button>
      {anchor}
      <button type="button" className={btn} onClick={() => onStep(side, 1)}>
        更晚 ▶
      </button>
    </div>
  );

  if (!frame) {
    return (
      <div className="flex min-w-0 flex-col gap-[5px] text-[12.5px]">
        <div>
          <b className={dark ? "text-white" : "text-mast-text"}>{lab}</b>
        </div>
        <div className={clsx("py-10 text-center", dark ? "text-[#9aa4af]" : "text-mast-muted")}>
          {side === "prev" ? "之前" : "之后"}没有帧
        </div>
        {stepButtons(null)}
      </div>
    );
  }

  const r = relation(frame, t0, t1, noun);
  const anchored = anchoredId === frame.id;
  return (
    <div className={clsx("flex min-w-0 flex-col gap-[5px] text-[12.5px]", dark && "text-[#dde3ea]")}>
      <div>
        <b className={dark ? "text-white" : "text-mast-text"}>{lab}</b>{" "}
        <Link
          to={{ search: `?${viewSearch({ v: "dir", d: frame.d })}` }}
          title={frame.fn}
          className={dark ? "text-[#c4b5fd]" : "text-mast-accent"}
        >
          {num(frame)}
        </Link>{" "}
        · {r.desc}
      </div>
      <FrameWithMarks key={frame.id} frame={frame} pts={pts} poly={poly} anchored={anchored} />
      <div className={clsx("font-mono text-[11.5px]", dark ? "text-[#9aa4af]" : "text-mast-muted")}>
        扫描 {fmtTs(fStart(frame))}–{fmtTs(frame.mt)} · {nm(frame.w)} nm · {fmtB(frame.b)} ·{" "}
        {Math.round(frame.sp || 0)} pA{frame.ang ? ` · 转角 ${frame.ang}°` : ""}
        {posText(frame)}
      </div>
      {stepButtons(
        onAnchor ? (
          <button
            type="button"
            className={clsx(btn, anchored && "!border-[#22d3ee] !bg-[#0e7490] !text-white")}
            onClick={() => onAnchor(side)}
          >
            {anchored ? (
              "✓ 已系于此帧（再点取消）"
            ) : (
              <>
                ⌖ 系于此帧 <Kbd dark={dark}>{side === "prev" ? "A" : "D"}</Kbd>
              </>
            )}
          </button>
        ) : null,
      )}
    </div>
  );
}

/** 大图里的对照区：谱 | 前一张 | 后一张。 */
export function ContextStage({
  it,
  frames,
  sides,
  anchoredId,
  interactive,
  onStep,
  onAnchor,
}: {
  it: GalleryItem;
  frames: readonly GalleryItem[];
  sides: { prev: number; next: number };
  anchoredId: string | null | undefined;
  /** 谱栏改用交互曲线（MAST 现有的 SpectrumChart）。 */
  interactive: boolean;
  onStep: (side: Side, step: number) => void;
  onAnchor: (side: Side) => void;
}) {
  const t0 = tBeg(it);
  const t1 = tEnd(it);
  const panel = (side: Side) => {
    const f = frames[sides[side]];
    const geom = f ? itemGeometry(it, f) : { pts: [], poly: null };
    return (
      <FramePanel
        side={side}
        frame={f}
        t0={t0}
        t1={t1}
        noun="谱"
        pts={geom.pts}
        poly={geom.poly}
        anchoredId={anchoredId}
        onStep={onStep}
        onAnchor={onAnchor}
        dark
        posText={() => {
          const p = geom.pts[0];
          if (!p) return null;
          return (
            <>
              {` · 帧内 (${p.u.toFixed(2)}, ${p.v.toFixed(2)})`}
              {!inside(p) && <span className="font-semibold text-[#fbbf24]"> {it.k === "g" ? "网格中心" : "谱位置"}在视野外</span>}
            </>
          );
        }}
      />
    );
  };

  return (
    <div className="grid w-full grid-cols-3 items-start gap-3 self-start overflow-auto max-[760px]:grid-cols-1">
      <div className="flex min-w-0 flex-col gap-[5px] text-[12.5px] text-[#dde3ea]">
        <div>
          <b className="text-white">{it.k === "g" ? "网格谱" : "谱"}</b> {num(it)} · {fmtTs(t0)}–{fmtTs(t1)}
        </div>
        {interactive && it.k === "s" ? (
          <div className="rounded bg-mast-panel p-2 text-mast-text">
            <SpectrumChart path={it.p} />
          </div>
        ) : (
          <img src={it.th} alt="" className="w-full bg-white object-contain" />
        )}
      </div>
      {panel("prev")}
      {panel("next")}
    </div>
  );
}
