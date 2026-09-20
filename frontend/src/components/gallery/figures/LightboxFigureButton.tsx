// 大图侧栏里的出图入口（设计 D22）：帧 →「出对比图」，网格谱 →「出逐层图」。
//
// 单独一个组件而不是写进 Lightbox：Lightbox 在 `if (!it) return null` 之前要把全部 hook
// 调完，把 useQueryClient 塞进去就又多一个必须排在早退之前的调用。

import { useQueryClient } from "@tanstack/react-query";
import type { GalleryItem } from "@/lib/gallery/types";
import { DARK_BTN } from "../bits";
import { startFigureJob } from "./figureJob";

export function LightboxFigureButton({ it }: { it: GalleryItem }) {
  const qc = useQueryClient();
  if (it.k !== "f" && it.k !== "g") return null;
  const isFrame = it.k === "f";
  return (
    <div className="mt-2">
      <button
        type="button"
        className={DARK_BTN}
        title={
          isFrame
            ? "「原版（只减平面）| 逐行调平」并排的一张图，底部写时刻、偏压、电流、尺寸（绘图实现 draw_sheets.py）"
            : "形貌 + 每一个偏压层的 dI/dV 与电流，每层各自拉伸（绘图实现 draw_sheets.py）"
        }
        onClick={() =>
          void startFigureJob(qc, isFrame ? { kind: "frame_sheet", ids: [it.id] } : { kind: "grid_sheets", ids: [it.id] })
        }
      >
        {isFrame ? "出对比图" : "出逐层图"}
      </button>
    </div>
  );
}
