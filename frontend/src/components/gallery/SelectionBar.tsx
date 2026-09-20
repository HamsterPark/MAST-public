// 底部操作条（旧版兼容格式 series.js 的 selBarEnsure / selBarUpdate）。
//
// 选中若干张后出现：一起打分、一起加/去标签、「存为系列…」（填系列名、评级、标签、备注；
// 也可以「并入」已有系列）、系列页里「从本系列移除」、取消选择。目录总览、已标记页、
// 出图页、数据根页不显示（原版如此；出图页没有卡片可选）。
//
// 选中的全是谱时多一个「拼接出图」（设计 §10 D22：单根谱宽范围拼接）。

import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import clsx from "clsx";
import type { GalleryView } from "@/lib/gallery/route";
import { viewSearch } from "@/lib/gallery/route";
import { autoName, kindOf, mergedMemberIds, newSeriesId, rangeText } from "@/lib/gallery/series";
import type { GalleryModel } from "./useGalleryData";
import { useMarksStore } from "./marksStore";
import { useSelectionStore } from "./selectionStore";
import { toast } from "./toast";
import { SMALL_BTN, SMALL_SELECT } from "./bits";
import { StitchSelected } from "./figures/StitchControls";

const ACC = "!border-mast-accent !bg-mast-accent !text-mast-accent-ink";

export function SelectionBar({ view, model }: { view: GalleryView; model: GalleryModel }) {
  const ids = useSelectionStore((s) => s.ids);
  const clear = useSelectionStore((s) => s.clear);
  const tagTable = useMarksStore((s) => s.doc.tags);
  const series = useMarksStore((s) => s.doc.series);
  const nav = useNavigate();

  const its = useMemo(() => model.items.filter((it) => ids.has(it.id)), [ids, model.items]);

  const [tag, setTag] = useState("");
  const [formOpen, setFormOpen] = useState(false);
  const [target, setTarget] = useState("");
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [form, setForm] = useState<{ r: number; tags: string[] }>({ r: 0, tags: [] });

  const hidden =
    !its.length || view.v === "dirs" || view.v === "marked" || view.v === "figures" || view.v === "setup";
  useEffect(() => {
    if (hidden) setFormOpen(false);
  }, [hidden]);
  if (hidden) return null;

  const fillForm = (sid: string) => {
    const s = sid ? series[sid] : undefined;
    setName(s ? s.name : autoName(its));
    setNote(s ? s.note || "" : "");
    setForm({ r: s ? s.r || 0 : 0, tags: s ? [...(s.tags || [])] : [] });
  };
  const curTag = tag && tagTable.includes(tag) ? tag : (tagTable[0] ?? "");
  const allSpectra = its.every((it) => it.k === "s");

  const saveSeries = () => {
    if (!its.length) return;
    const old = target ? series[target] : undefined;
    const memberIds = mergedMemberIds(old?.ids ?? [], its, model.items);
    const members = memberIds.map((id) => model.byId.get(id)).filter((x): x is NonNullable<typeof x> => !!x);
    const sid = target || newSeriesId(Date.now());
    const finalName = name.trim() || autoName(members);
    useMarksStore.getState().setSeries(sid, {
      ...(old ?? {}),
      name: finalName,
      k: kindOf(members),
      ids: memberIds,
      r: form.r,
      tags: [...form.tags],
      note,
    });
    setFormOpen(false);
    toast(
      <>
        {old ? "已并入" : "已存为"}系列{" "}
        <Link to={{ search: `?${viewSearch({ v: "series", s: sid })}` }}>{finalName}</Link>（{memberIds.length} 个）
      </>,
    );
    clear();
  };

  const removeFromSeries = () => {
    if (view.v !== "series") return;
    const s = useMarksStore.getState().doc.series[view.s];
    if (!s) return;
    const left = s.ids.filter((id) => !ids.has(id));
    useMarksStore.getState().setSeries(view.s, left.length ? { ...s, ids: left } : null);
    clear();
    if (!left.length) nav({ search: `?${viewSearch({ v: "marked" })}` });
  };

  return (
    <div className="fixed inset-x-0 bottom-0 z-40 border-t-2 border-mast-accent bg-mast-panel px-[18px] py-2 text-[13px] text-mast-text shadow-[0_-4px_16px_rgba(0,0,0,.14)]">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <b>已选 {its.length} 个</b>
        <span className="text-mast-muted">{rangeText(its)}</span>
        <span className="inline-flex items-center gap-[3px]">
          一起打分
          {([
            [1, "✓"],
            [2, "★"],
            [-1, "✗"],
            [0, "清评级"],
          ] as const).map(([r, lab]) => (
            <button
              key={r}
              type="button"
              className={SMALL_BTN}
              onClick={() => {
                for (const it of its) useMarksStore.getState().updateMark(it, { r });
                toast(`已给 ${its.length} 个打分`);
              }}
            >
              {lab}
            </button>
          ))}
        </span>
        <span className="inline-flex items-center gap-[3px]">
          <select className={SMALL_SELECT} value={curTag} onChange={(e) => setTag(e.target.value)}>
            {tagTable.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          {([
            [true, "一起加标签"],
            [false, "去掉"],
          ] as const).map(([add, lab]) => (
            <button
              key={lab}
              type="button"
              className={SMALL_BTN}
              disabled={!curTag}
              onClick={() => {
                for (const it of its) useMarksStore.getState().toggleTag(it, curTag, add);
                toast(`${add ? "已加" : "已去掉"}标签「${curTag}」×${its.length}`);
              }}
            >
              {lab}
            </button>
          ))}
        </span>
        <button
          type="button"
          className={clsx(SMALL_BTN, ACC)}
          onClick={() => {
            if (!formOpen) fillForm(target);
            setFormOpen((o) => !o);
          }}
        >
          存为系列…
        </button>
        {allSpectra && <StitchSelected ids={its.map((it) => it.id)} />}
        {view.v === "series" && (
          <button type="button" className={SMALL_BTN} onClick={removeFromSeries}>
            从本系列移除
          </button>
        )}
        <button type="button" className={SMALL_BTN} onClick={clear}>
          取消选择
        </button>
      </div>

      {formOpen && (
        <div className="mt-[7px] flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-mast-border pt-[7px]">
          <select
            className={SMALL_SELECT}
            value={target}
            onChange={(e) => {
              setTarget(e.target.value);
              fillForm(e.target.value);
            }}
          >
            <option value="">新建系列</option>
            {Object.entries(series).map(([sid, s]) => (
              <option key={sid} value={sid}>
                并入：{s.name}
              </option>
            ))}
          </select>
          <input
            className={`${SMALL_SELECT} w-[26rem] max-w-full`}
            value={name}
            placeholder="系列名"
            autoFocus
            onChange={(e) => setName(e.target.value)}
          />
          <span className="inline-flex gap-[3px]">
            {([
              [1, "✓ 可用"],
              [2, "★ 重点"],
              [-1, "✗ 排除"],
            ] as const).map(([r, lab]) => (
              <button
                key={r}
                type="button"
                className={clsx(SMALL_BTN, form.r === r && ACC)}
                onClick={() => setForm((f) => ({ ...f, r: f.r === r ? 0 : r }))}
              >
                {lab}
              </button>
            ))}
          </span>
          <span className="inline-flex flex-wrap gap-[3px]">
            {tagTable.map((t) => (
              <button
                key={t}
                type="button"
                className={clsx(SMALL_BTN, "text-xs", form.tags.includes(t) && ACC)}
                onClick={() =>
                  setForm((f) => ({
                    ...f,
                    tags: f.tags.includes(t) ? f.tags.filter((x) => x !== t) : [...f.tags, t],
                  }))
                }
              >
                {t}
              </button>
            ))}
          </span>
          <input
            className={`${SMALL_SELECT} w-72 max-w-full`}
            value={note}
            placeholder="系列备注（这组数据是什么、为什么有用）"
            onChange={(e) => setNote(e.target.value)}
          />
          <button type="button" className={clsx(SMALL_BTN, ACC)} onClick={saveSeries}>
            保存系列
          </button>
          <button type="button" className={SMALL_BTN} onClick={() => setFormOpen(false)}>
            收起
          </button>
        </div>
      )}
    </div>
  );
}
