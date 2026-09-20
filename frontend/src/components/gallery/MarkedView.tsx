// 已标记页（旧版兼容格式 app.js 的 renderMarked / renderTable，与 series.js 的 seriesSectionHtml）。
//
// 计数、导出（服务端现场生成 md/csv）、导入 marks.json（服务端合并，可加前缀）、标签表
// 编辑、系列表、单条标记表（同一套筛选条；按目录分组，分组行带目录备注；点缩略图开大图）。
// 表格在这一页里**不随每次改动重算**：刚清掉的条目留一行，重进本页或关大图时才消失（原版如此）。

import { useCallback, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { GalleryItem } from "@/lib/gallery/types";
import { KLAB, RLAB, autoTags, dirLabel, fmtT, fullPath, meta, num } from "@/lib/gallery/format";
import { applyFilters, facets, sanitizeFilters } from "@/lib/gallery/filters";
import { parseTagList } from "@/lib/gallery/marks";
import { seriesOf, shortFn, sortSeriesEntries } from "@/lib/gallery/series";
import { viewSearch } from "@/lib/gallery/route";
import { GALLERY_KEYS, type GalleryModel } from "./useGalleryData";
import { useFiltersStore } from "./filtersStore";
import { useMarksStore } from "./marksStore";
import { FilterBar } from "./FilterBar";
import { Lightbox } from "./Lightbox";
import { CopyButton } from "./clipboard";
import { toast } from "./toast";
import { Chip, SMALL_BTN, SMALL_SELECT, chipClass } from "./bits";
import { H1, Sub } from "./ListHeads";
import { SingleSpectraStitch } from "./figures/SingleSpectraStitch";

/** 导入一份 marks.json（例如旧版兼容格式那份）。已标记页与数据根页共用。 */
export function ImportMarksForm({ defaultPrefix }: { defaultPrefix: string }) {
  const [prefix, setPrefix] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();
  const value = prefix ?? defaultPrefix;

  const run = async (file: File) => {
    setBusy(true);
    try {
      const doc: unknown = JSON.parse(await file.text());
      if (!doc || typeof doc !== "object") throw new Error("不是一个 JSON 对象");
      const { data, error } = await api.POST("/api/gallery/marks/import", {
        body: { doc: doc as Record<string, unknown>, key_prefix: value },
      });
      if (error || !data || !data.ok) throw new Error(data?.detail || "导入失败");
      toast(
        `导入：单条 ${data.items} · 系列 ${data.series} · 目录 ${data.days} · 新标签 ${data.tags_added}` +
          (data.unmatched ? ` · 其中 ${data.unmatched} 条在当前索引里找不到（照样导入了）` : ""),
      );
      await useMarksStore.getState().reload();
      void qc.invalidateQueries({ queryKey: GALLERY_KEYS.index });
    } catch (e) {
      toast(`导入失败：${(e as Error).message}`, "err");
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <label className="flex items-center gap-1 text-[13px]" title="旧版兼容格式 marks.json 的键是相对它的数据根的；导进来要加上这里的根名">
        键前缀
        <input className={`${SMALL_SELECT} w-24 font-mono`} value={value} onChange={(e) => setPrefix(e.target.value)} />
      </label>
      <button type="button" className={SMALL_BTN} disabled={busy} onClick={() => fileRef.current?.click()}>
        {busy ? "导入中…" : "导入 marks.json…"}
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void run(f);
        }}
      />
    </span>
  );
}

function TagTableEditor() {
  const tags = useMarksStore((s) => s.doc.tags);
  const setTags = useMarksStore((s) => s.setTags);
  const [text, setText] = useState<string | null>(null);
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <span className="text-xs text-mast-muted">标签表（逗号分隔，前 9 个有快捷键 Q–O）：</span>
      <input
        className={`${SMALL_SELECT} w-[32rem] max-w-full`}
        value={text ?? tags.join("，")}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        type="button"
        className={SMALL_BTN}
        onClick={() => {
          setTags(parseTagList(text ?? tags.join("，")));
          setText(null);
          toast("标签表已保存");
        }}
      >
        保存标签表
      </button>
    </span>
  );
}

function SeriesTable({ model }: { model: GalleryModel }) {
  const series = useMarksStore((s) => s.doc.series);
  const entries = useMemo(() => sortSeriesEntries(series, model.items), [series, model.items]);
  if (!entries.length) {
    return (
      <>
        <h2 className="mb-1.5 mt-[18px] text-[1.05rem] font-semibold">系列</h2>
        <Sub>还没有系列：在某个目录里勾选卡片（Shift 连选一段），再点底部「存为系列…」。</Sub>
      </>
    );
  }
  return (
    <>
      <h2 className="mb-1.5 mt-[18px] text-[1.05rem] font-semibold">系列 · {entries.length}</h2>
      <div className="overflow-x-auto">
        <table className="mt-2.5 w-full border-collapse text-[13px]">
          <thead>
            <tr className="text-left">
              {["预览", "系列", "评级", "标签", "备注", "成员", "位置系于"].map((h) => (
                <th key={h} className="border-b border-mast-border bg-mast-bg px-1.5 py-[5px] font-semibold">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map(({ sid, s, mem }) => {
              const a = mem[0];
              const b = mem[mem.length - 1];
              return (
                <tr key={sid} className="align-top">
                  <td className="border-b border-mast-border px-1.5 py-[5px]">
                    {mem.slice(0, 3).map((it) => (
                      <img key={it.id} loading="lazy" src={it.th} alt="" className="mr-0.5 inline-block w-16" />
                    ))}
                  </td>
                  <td className="border-b border-mast-border px-1.5 py-[5px]">
                    <Link to={{ search: `?${viewSearch({ v: "series", s: sid })}` }} className="text-mast-accent hover:underline">
                      <b>{s.name}</b>
                    </Link>
                  </td>
                  <td className="border-b border-mast-border px-1.5 py-[5px]">{s.r ? RLAB[String(s.r)] : ""}</td>
                  <td className="border-b border-mast-border px-1.5 py-[5px]">
                    {(s.tags || []).map((t) => (
                      <Chip key={t} kind="user">
                        {t}
                      </Chip>
                    ))}
                  </td>
                  <td className="max-w-[420px] whitespace-pre-wrap border-b border-mast-border px-1.5 py-[5px]">{s.note}</td>
                  <td className="border-b border-mast-border px-1.5 py-[5px] font-mono text-mast-muted">
                    {mem.length} 个
                    {a && b && (
                      <>
                        <br />
                        {num(a)} → {num(b)}
                        <br />
                        {fmtT(a.t)} → {fmtT(b.mt || b.t1 || b.t)}
                      </>
                    )}
                  </td>
                  <td className="border-b border-mast-border px-1.5 py-[5px] text-mast-muted">
                    {s.anchor && (
                      <>
                        ⌖ {shortFn(s.anchor.fn)}
                        <br />
                        {s.anchor.desc}
                      </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function MarkedView({ model }: { model: GalleryModel }) {
  const F = useFiltersStore((s) => s.F);
  const tags = useMarksStore((s) => s.doc.tags);
  const epoch = useMarksStore((s) => s.epoch);
  const nSeries = useMarksStore((s) => Object.keys(s.doc.series).length);
  const [version, setVersion] = useState(0);

  // 本页的底：进页面（或关大图、导入之后）那一刻有标记的条目。
  const base = useMemo(() => {
    const items = useMarksStore.getState().doc.items;
    return model.items.filter((it) => items[it.id]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model.items, version, epoch]);
  const fac = useMemo(() => facets(base), [base]);
  const effective = useMemo(() => sanitizeFilters(F, fac, tags), [F, fac, tags]);
  const list = useMemo(() => {
    const doc = useMarksStore.getState().doc;
    return applyFilters(base, effective, {
      view: "marked",
      mark: (id) => doc.items[id],
      seriesOf: (id) => seriesOf(doc.series, id),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, effective, version, epoch]);

  const [lb, setLb] = useState<number | null>(null);
  const closeLb = useCallback(() => {
    setLb(null);
    setVersion((v) => v + 1);
  }, []);

  const itemsNow = useMarksStore((s) => s.doc.items);
  const count = (r: number) => base.filter((it) => (itemsNow[it.id]?.r ?? 0) === r).length;
  const defaultPrefix = model.roots[0] ? `${model.roots[0].name}/` : "";

  return (
    <div>
      <H1>
        已标记 · {base.length} 个 · 系列 {nSeries} 个
      </H1>
      <Sub>
        ★ 重点 {count(2)} · ✓ 可用 {count(1)} · ✗ 排除 {count(-1)} · 只有标签/备注/系定 {count(0)}。 导出：
        {(["md", "csv", "series_csv", "json"] as const).map((fmt) => (
          <a key={fmt} href={`/api/gallery/marks/export/${fmt}`} download className="ml-2 text-mast-accent hover:underline">
            {fmt === "series_csv" ? "marks_series.csv" : `marks.${fmt}`}
          </a>
        ))}
        （服务端的图库状态目录里也有同名文件，每次改动都会重写）
      </Sub>
      <div className="my-1.5 flex flex-wrap items-center gap-2">
        <ImportMarksForm defaultPrefix={defaultPrefix} />
        <TagTableEditor />
      </div>

      <SeriesTable model={model} />

      <SingleSpectraStitch model={model} />

      <h2 className="mb-1.5 mt-[18px] text-[1.05rem] font-semibold">单条标记</h2>
      <FilterBar
        view="marked"
        effective={effective}
        fac={fac}
        tags={tags}
        list={list}
        total={base.length}
        showHint={false}
      />
      <div className="overflow-x-auto">
        <table className="mt-2.5 w-full border-collapse text-[13px]">
          <thead>
            <tr className="text-left">
              {["缩略图", "目录 · 编号", "评级", "标签", "备注", "参数", "路径"].map((h) => (
                <th key={h} className="border-b border-mast-border bg-mast-bg px-1.5 py-[5px] font-semibold">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <MarkRows list={list} model={model} onOpen={setLb} />
          </tbody>
        </table>
      </div>
      {lb != null && list.length > 0 && (
        <Lightbox list={list} index={lb} model={model} onIndex={setLb} onClose={closeLb} />
      )}
    </div>
  );
}

function MarkRows({ list, model, onOpen }: { list: GalleryItem[]; model: GalleryModel; onOpen: (j: number) => void }) {
  const items = useMarksStore((s) => s.doc.items);
  const days = useMarksStore((s) => s.doc.days);
  const series = useMarksStore((s) => s.doc.series);
  const rows: React.ReactNode[] = [];
  let last = "";
  list.forEach((it, j) => {
    const m = items[it.id] ?? { r: 0, tags: [], note: "" };
    if (it.d !== last) {
      const note = (days[it.d]?.note || "").trim();
      rows.push(
        <tr key={`dir:${it.d}:${j}`}>
          <td colSpan={7} className="border-b border-mast-border px-1.5 py-[5px]">
            <b>{dirLabel(it.d)}</b> <span className="text-xs text-mast-faint">{it.d}</span>
            {note && <span className="text-mast-muted"> — {note}</span>}
          </td>
        </tr>,
      );
      last = it.d;
    }
    const cell = "border-b border-mast-border px-1.5 py-[5px]";
    rows.push(
      <tr key={it.id} className="align-top">
        <td className={cell}>
          <button type="button" onClick={() => onOpen(j)} title="点开大图">
            <img loading="lazy" src={it.th} alt="" className="block w-[120px] cursor-zoom-in" />
          </button>
        </td>
        <td className={`${cell} font-mono`}>
          <b>{num(it)}</b>
          <br />
          {KLAB[it.k]}
        </td>
        <td className={cell}>{m.r ? RLAB[String(m.r)] : ""}</td>
        <td className={cell}>
          {(m.tags || []).map((t) => (
            <Chip key={t} kind="user">
              {t}
            </Chip>
          ))}
        </td>
        <td className={`${cell} max-w-[420px] whitespace-pre-wrap`}>{m.note}</td>
        <td className={`${cell} font-mono text-mast-muted`}>
          {meta(it)}
          <br />
          {autoTags(it, model.lastBatch, model.numOf).map((t) => (
            <Chip key={t.kind} kind={t.kind} title={t.title}>
              {t.text}
            </Chip>
          ))}
          {seriesOf(series, it.id).map((sid) => (
            <Link key={sid} to={{ search: `?${viewSearch({ v: "series", s: sid })}` }} className={chipClass("ser")}>
              ▤ {series[sid]?.name || sid}
            </Link>
          ))}
          {m.anchor && (
            <Chip kind="anc" title={m.anchor.desc}>
              ⌖ {shortFn(m.anchor.fn)}
            </Chip>
          )}
        </td>
        <td className={`${cell} break-all font-mono text-[11px] text-mast-muted`}>
          {fullPath(it)} <CopyButton text={fullPath(it)} />
        </td>
      </tr>,
    );
  });
  return <>{rows}</>;
}
