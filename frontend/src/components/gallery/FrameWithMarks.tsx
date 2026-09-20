// 帧缩略图 + SVG 标记层（旧版兼容格式 context.js 的 drawMarks）。
//
// 几何全在 lib/gallery/context.ts 的 overlayShapes：它按缩略图的**自然像素尺寸**算，
// SVG 用同一个 viewBox 盖在图上（preserveAspectRatio="none"，与 <img> 同宽同高比例），
// 线宽用 non-scaling-stroke，图缩放后线不变粗。尺寸要等图片加载完才知道。

import { useCallback, useState } from "react";
import clsx from "clsx";
import type { GalleryItem } from "@/lib/gallery/types";
import { overlayShapes, type OverlayPoint, type UV } from "@/lib/gallery/context";

const IN = "#22d3ee";
const OUT = "#fbbf24";

function SinglePoint({ p, width }: { p: OverlayPoint; width: number }) {
  const col = p.out ? OUT : IN;
  const dash = p.out ? "4 3" : undefined;
  const { x, y } = p;
  return (
    <g>
      <circle cx={x} cy={y} r={11} fill="none" stroke="#000" strokeWidth={4} vectorEffect="non-scaling-stroke" strokeDasharray={dash} />
      <circle cx={x} cy={y} r={11} fill="none" stroke={col} strokeWidth={2} vectorEffect="non-scaling-stroke" strokeDasharray={dash} />
      <path
        d={`M${x - 18} ${y}h10M${x + 8} ${y}h10M${x} ${y - 18}v10M${x} ${y + 8}v10`}
        stroke={col}
        strokeWidth={1.5}
        vectorEffect="non-scaling-stroke"
      />
      {p.out && (
        <text
          x={Math.min(x + 8, width - 60)}
          y={Math.max(y - 14, 14)}
          fontSize={13}
          fill={col}
          stroke="#000"
          strokeWidth={3}
          paintOrder="stroke"
        >
          视野外
        </text>
      )}
    </g>
  );
}

function ManyPoint({ p }: { p: OverlayPoint }) {
  const col = p.out ? OUT : IN;
  return (
    <g>
      <circle cx={p.x} cy={p.y} r={4} fill={col} stroke="#000" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      {p.label && (
        <text
          x={p.x + 5}
          y={p.y - 5}
          fontSize={12}
          fill={col}
          stroke="#000"
          strokeWidth={3}
          paintOrder="stroke"
          fontFamily="Consolas,monospace"
        >
          {p.label}
        </text>
      )}
    </g>
  );
}

export function FrameWithMarks({
  frame,
  pts,
  poly,
  anchored,
}: {
  frame: GalleryItem;
  pts: UV[];
  poly: UV[] | null;
  anchored?: boolean;
}) {
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  // 缓存命中的图可能在监听挂上之前就加载完了：ref 回调里再看一眼 complete。
  const imgRef = useCallback((img: HTMLImageElement | null) => {
    if (img && img.complete && img.naturalWidth) setNat({ w: img.naturalWidth, h: img.naturalHeight });
  }, []);
  const shapes = nat ? overlayShapes(frame, pts, poly, nat.w, nat.h) : null;

  return (
    <div className={clsx("relative leading-[0]", anchored && "outline outline-[3px] outline-[#22d3ee] outline-offset-1")}>
      <img
        ref={imgRef}
        src={frame.th}
        alt=""
        className="block w-full"
        onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
      />
      {shapes && (
        <svg
          viewBox={`0 0 ${shapes.width} ${shapes.height}`}
          preserveAspectRatio="none"
          className="pointer-events-none absolute left-0 top-0 h-full w-full overflow-visible"
        >
          {shapes.polygon && (
            <polygon
              points={shapes.polygon}
              fill="rgba(34,211,238,.12)"
              stroke={IN}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          )}
          {shapes.polyline && (
            <polyline
              points={shapes.polyline}
              fill="none"
              stroke="rgba(34,211,238,.55)"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
          )}
          {shapes.points.map((p, i) =>
            shapes.single ? <SinglePoint key={i} p={p} width={shapes.width} /> : <ManyPoint key={i} p={p} />,
          )}
        </svg>
      )}
    </div>
  );
}
