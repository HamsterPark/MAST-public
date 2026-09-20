// 筛选条（旧版兼容格式 app.js 的 filterBar + bindBar），多一项「文件名前缀」。
//
// 显示的值是调用方算好的 `effective`（已按当前列表退掉不存在的值），改动写回
// filtersStore（持久化）。筛选条吸顶在图库顶栏下面：顶栏高度会随窗口宽度换行变化，
// 所以 top 用 GalleryApp 量出来写进 CSS 变量的 --gallery-head。

import type { ReactNode } from "react";
import type { GalleryItem } from "@/lib/gallery/types";
import type { ViewName } from "@/lib/gallery/route";
import { CARD_WIDTH_MAX, CARD_WIDTH_MIN, type Facets, type Filters } from "@/lib/gallery/filters";
import { fmtB, fullPath } from "@/lib/gallery/format";
import { useFiltersStore } from "./filtersStore";
import { CopyButton } from "./clipboard";
import { SMALL_SELECT } from "./bits";

function Sel<T extends string>({
  label,
  value,
  onChange,
  options,
  title,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: [T, string][];
  title?: string;
}) {
  return (
    <label className="flex items-center gap-1 whitespace-nowrap" title={title}>
      {label}
      <select className={SMALL_SELECT} value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map(([v, lab]) => (
          <option key={v} value={v}>
            {lab}
          </option>
        ))}
      </select>
    </label>
  );
}

export function FilterBar({
  view,
  effective,
  fac,
  tags,
  list,
  total,
  extra,
  showHint = true,
}: {
  view: ViewName;
  effective: Filters;
  fac: Facets;
  tags: readonly string[];
  /** 当前筛选结果（计数、复制路径用）。 */
  list: readonly GalleryItem[];
  total: number;
  extra?: ReactNode;
  showHint?: boolean;
}) {
  const setF = useFiltersStore((s) => s.setF);
  const F = effective;

  return (
    <div
      className="sticky z-10 flex flex-wrap items-center gap-x-3.5 gap-y-1.5 border-b border-mast-border bg-mast-bg py-[7px] text-[13px] text-mast-text"
      style={{ top: "var(--gallery-head, 0px)" }}
    >
      {view !== "all" && (
        <Sel
          label="类型"
          value={F.k}
          onChange={(k) => setF({ k })}
          options={[["all", "全部"], ["f", "帧"], ["s", "谱"], ["g", "网格"]]}
        />
      )}
      <Sel
        label="偏压"
        value={F.b}
        onChange={(b) => setF({ b })}
        options={[["", "全部"], ...fac.biases.map(([k, b]): [string, string] => [k, fmtB(b)])]}
      />
      <Sel
        label="帧宽"
        value={F.w}
        onChange={(w) => setF({ w })}
        options={[["", "全部"], ...fac.widths.map(([k, w]): [string, string] => [k, `${+w.toFixed(1)} nm`])]}
      />
      {fac.prefixes.length > 1 && (
        <Sel
          label="前缀"
          title="文件名去掉末尾编号之后的部分——通常就是 Nanonis 里起的扫描系列名"
          value={F.pf}
          onChange={(pf) => setF({ pf })}
          options={[["", "全部"], ...fac.prefixes.map(([p, n]): [string, string] => [p, `${p}（${n}）`])]}
        />
      )}
      <Sel
        label="衬度"
        value={F.c}
        onChange={(c) => setF({ c })}
        options={[["", "全部"], ["atom", "有原子分辨"], ["half", "有超结构"], ["li", "有 lock-in dI/dV"]]}
      />
      <label className="flex items-center gap-1 whitespace-nowrap">
        <input type="checkbox" checked={F.full} onChange={(e) => setF({ full: e.target.checked })} />
        只看扫完
      </label>
      {view !== "marked" && view !== "series" && (
        <label
          className="flex items-center gap-1 whitespace-nowrap"
          title="同一次扫描存了两份的文件（数据块逐字节相同），只留先存的那份；自己带标记或在系列里的照样显示"
        >
          <input type="checkbox" checked={F.nodup} onChange={(e) => setF({ nodup: e.target.checked })} />
          隐藏重复保存
        </label>
      )}
      <Sel
        label="标记"
        value={F.m}
        onChange={(m) => setF({ m })}
        options={[
          ["", "全部"], ["none", "未标记"], ["any", "已标记"], ["2", "★ 重点"], ["1", "✓ 可用"],
          ["-1", "✗ 排除"], ["hidex", "隐藏 ✗"], ["ser", "属于系列"], ["anc", "位置已系定"],
        ]}
      />
      <Sel
        label="标签"
        value={F.tag}
        onChange={(tag) => setF({ tag })}
        options={[["", "全部"], ...tags.map((t): [string, string] => [t, t])]}
      />
      <label className="flex items-center gap-1 whitespace-nowrap">
        搜索
        <input
          className={`${SMALL_SELECT} w-24`}
          placeholder="编号 / 备注"
          value={F.q}
          onChange={(e) => setF({ q: e.target.value })}
        />
      </label>
      <Sel
        label="排序"
        value={F.sort}
        onChange={(sort) => setF({ sort })}
        options={[["asc", "时间 ↑"], ["desc", "时间 ↓"]]}
      />
      <label className="flex items-center gap-1 whitespace-nowrap">
        卡片
        <input
          type="range"
          min={CARD_WIDTH_MIN}
          max={CARD_WIDTH_MAX}
          step={10}
          value={F.cw}
          onChange={(e) => setF({ cw: Number(e.target.value) })}
        />
      </label>
      <span className="text-xs text-mast-muted">
        显示 <b className="text-mast-text">{list.length}</b> / {total}
      </span>
      <CopyButton
        text={() => list.map(fullPath).join("\n")}
        label="复制这些文件的路径"
        className="!text-[13px]"
        title="每行一个完整路径"
      />
      {extra}
      {showHint && (
        <span className="text-xs text-mast-muted">勾选框选中，Shift 连选一段 → 底部可一起打分或存为系列</span>
      )}
    </div>
  );
}
