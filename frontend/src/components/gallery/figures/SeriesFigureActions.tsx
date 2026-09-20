// 系列页上的出图入口（设计 D22）。
//
// 帧系列 →「16:9 拼图」（一列一个扫描转角、一行一轮）与「叠加…」（锚点 + 视野）；
// 谱系列 →「拉线谱出图…」（弹层，见 StsLinesDialog）。混合系列两边的按钮都给——
// 服务端只取对应种类的成员。

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import type { GalleryItem } from "@/lib/gallery/types";
import type { GalleryModel } from "../useGalleryData";
import { SMALL_BTN, SMALL_SELECT } from "../bits";
import { toast } from "../toast";
import { startFigureJob } from "./figureJob";
import { StsLinesDialog } from "./StsLinesDialog";

const ANCHORS: readonly (readonly [string, string])[] = [
  ["darkest", "中心附近最暗处"],
  ["brightest", "中心附近最亮处"],
  ["center", "帧中心（不找锚点）"],
];

const ACC = "!border-mast-accent !bg-mast-accent !text-mast-accent-ink";

export function SeriesFigureActions({ sid, mem, model }: { sid: string; mem: GalleryItem[]; model: GalleryModel }) {
  const qc = useQueryClient();
  const [stackOpen, setStackOpen] = useState(false);
  const [anchor, setAnchor] = useState("darkest");
  const [fov, setFov] = useState(() => String(mem.find((it) => it.k === "f")?.w || ""));
  const [linesOpen, setLinesOpen] = useState(false);

  const nF = mem.filter((it) => it.k === "f").length;
  const nS = mem.filter((it) => it.k === "s").length;
  if (!nF && !nS) return null;

  const runStack = () => {
    const t = fov.trim();
    const f = /^\d*\.?\d+$/.test(t) ? Number(t) : NaN;
    if (!(f > 0)) {
      toast("视野要是正数（nm）", "err");
      return;
    }
    void startFigureJob(qc, { kind: "series_stack", series: [sid], options: { anchor, fov_nm: f } });
    setStackOpen(false);
  };

  return (
    <div className="my-1.5 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-[13px]">
      <span className="text-mast-muted">出图：</span>
      {nF > 0 && (
        <>
          <button
            type="button"
            className={SMALL_BTN}
            title="16:9 两页：一列一个扫描转角（从第 1 帧起按采集顺序），一行一轮；每帧转到第 1 帧朝向、以锚点为中心"
            onClick={() => void startFigureJob(qc, { kind: "series_slides", series: [sid] })}
          >
            16:9 拼图
          </button>
          <button
            type="button"
            className={clsx(SMALL_BTN, stackOpen && "!border-mast-accent")}
            title="刚性叠加 + 晶格仿射校正叠加，另出配准表（漂移轨迹）与 npy 数组"
            onClick={() => setStackOpen((o) => !o)}
          >
            叠加…
          </button>
        </>
      )}
      {nS > 0 && (
        <button
          type="button"
          className={SMALL_BTN}
          title="勾上同一条线的各个区组系列，出热图与瀑布（分区组比较 / 站位均值）"
          onClick={() => setLinesOpen(true)}
        >
          拉线谱出图…
        </button>
      )}
      {stackOpen && (
        <span className="inline-flex flex-wrap items-center gap-1.5 rounded-[3px] border border-mast-border bg-mast-panel px-2 py-1">
          锚点
          <select className={SMALL_SELECT} value={anchor} onChange={(e) => setAnchor(e.target.value)}>
            {ANCHORS.map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
          视野
          <input className={`${SMALL_SELECT} w-14`} value={fov} onChange={(e) => setFov(e.target.value)} />
          nm
          <button type="button" className={clsx(SMALL_BTN, ACC)} onClick={runStack}>
            开始叠加
          </button>
          <button type="button" className={SMALL_BTN} onClick={() => setStackOpen(false)}>
            收起
          </button>
        </span>
      )}
      {linesOpen && <StsLinesDialog sid={sid} model={model} onClose={() => setLinesOpen(false)} />}
    </div>
  );
}
