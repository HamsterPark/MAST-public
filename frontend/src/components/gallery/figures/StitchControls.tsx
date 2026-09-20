// 单根谱拼接的发起（设计 D22；绘图实现 draw_sts.py 的 fig_combo）：底部选中条（选中的全是谱时）
// 与已标记页的「不属于任何系列的单根谱」共用。
//
// κ 只用来在图上标注 Zoff 差对应的理论倍数 exp(2κΔz)，系数本身按接缝附近的实测电流比算
// ——所以它是出图参数，不是仪器配置（设计 D23）。

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { parseKappa } from "@/lib/gallery/figures";
import { SMALL_BTN, SMALL_SELECT } from "../bits";
import { toast } from "../toast";
import { startFigureJob } from "./figureJob";

export const DEFAULT_KAPPA = "";

export function KappaInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <label
      className="flex items-center gap-1"
      title="隧穿衰减常数 κ：只用来在图上标注 Zoff 差的理论倍数 exp(2κΔz)；请填写适用于当前数据的值；留空不计算理论倍率"
    >
      κ
      <input className={`${SMALL_SELECT} w-14`} value={value} onChange={(e) => onChange(e.target.value)} />
      nm⁻¹
    </label>
  );
}

/** `ids` 为空且 `groupByDir` ⇒ 让服务端把全部「不属于系列的标记谱」按目录各拼一张。 */
export function useStitchStarter() {
  const qc = useQueryClient();
  return (ids: string[], kappaText: string, groupByDir = false) => {
    const kappa = parseKappa(kappaText);
    if (kappaText.trim() && kappa == null) {
      toast("κ 要是正数（nm⁻¹）", "err");
      return;
    }
    const options: Record<string, unknown> = kappa == null ? {} : { kappa_per_nm: kappa };
    if (groupByDir) options.group_by_dir = true;
    void startFigureJob(qc, { kind: "sts_stitch", ids, options });
  };
}

/** 底部选中条上的「拼接出图」。 */
export function StitchSelected({ ids }: { ids: string[] }) {
  const [kappa, setKappa] = useState(DEFAULT_KAPPA);
  const start = useStitchStarter();
  return (
    <span className="inline-flex items-center gap-1.5">
      <button
        type="button"
        className={SMALL_BTN}
        title="把选中的谱拼成一条宽范围曲线：覆盖 0 V、sweep×点数最多的一段为核心，其余各段按接缝附近的电流比乘系数接上"
        onClick={() => start(ids, kappa)}
      >
        拼接出图
      </button>
      <KappaInput value={kappa} onChange={setKappa} />
    </span>
  );
}
