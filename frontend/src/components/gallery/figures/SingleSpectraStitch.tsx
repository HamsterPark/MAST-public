// 已标记页：不属于任何系列的单根谱，按目录拼接（绘图实现 draw_sts.py 的「单根谱拼接」）。
//
// 绘图实现的规则原样：带标记、是谱、不在任何系列里，按目录成组。拼不拼、按哪几条拼由人点。

import { useMemo, useState } from "react";
import { dirLabel, num } from "@/lib/gallery/format";
import { singleSpectraByDir } from "@/lib/gallery/figures";
import type { GalleryModel } from "../useGalleryData";
import { useMarksStore } from "../marksStore";
import { SMALL_BTN } from "../bits";
import { Sub } from "../ListHeads";
import { DEFAULT_KAPPA, KappaInput, useStitchStarter } from "./StitchControls";

export function SingleSpectraStitch({ model }: { model: GalleryModel }) {
  const items = useMarksStore((s) => s.doc.items);
  const series = useMarksStore((s) => s.doc.series);
  const groups = useMemo(() => singleSpectraByDir(items, series, model.byId), [items, series, model.byId]);
  const [kappa, setKappa] = useState(DEFAULT_KAPPA);
  const start = useStitchStarter();
  const total = groups.reduce((a, g) => a + g.ids.length, 0);

  return (
    <>
      <h2 className="mb-1.5 mt-[18px] text-[1.05rem] font-semibold">不属于任何系列的单根谱 · {total}</h2>
      {!groups.length ? (
        <Sub>没有：带标记、又不在任何系列里的谱才会列在这里。</Sub>
      ) : (
        <>
          <Sub>
            同一目录里几段不同偏压范围的谱可以拼成一条宽范围曲线：覆盖 0 V、sweep×点数最多的一段为核心（同范围同 Zoff 的几条先平均），
            其余各段按接缝附近的电流比乘系数接上。各段不在同一点时图上会写明相距多远，拼接只作参考。
          </Sub>
          <div className="mb-1.5 flex flex-wrap items-center gap-2 text-[13px]">
            <KappaInput value={kappa} onChange={setKappa} />
            <button type="button" className={SMALL_BTN} onClick={() => start([], kappa, true)}>
              全部按目录拼接
            </button>
          </div>
          <ul className="space-y-1 text-[13px]">
            {groups.map((g) => (
              <li key={g.d} className="flex flex-wrap items-center gap-2">
                <b>{dirLabel(g.d)}</b>
                <span className="text-xs text-mast-faint">{g.d}</span>
                <span className="font-mono text-mast-muted">
                  {g.ids
                    .map((id) => {
                      const it = model.byId.get(id);
                      return it ? num(it) : id;
                    })
                    .join(" · ")}
                </span>
                <button type="button" className={SMALL_BTN} onClick={() => start(g.ids, kappa)}>
                  拼接出图（{g.ids.length} 条）
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}
