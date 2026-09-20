// 系列页（旧版兼容格式 series.js 的 renderSeriesView，与 context.js 的 ctxSeries）。
//
// 头部可改名、评级、标签、备注、删除（点两次确认，成员自己的单条标记不动）；系列里有
// 谱时，把全部成员的位置按时间编号画在系列前后的两张帧上，可以把整个系列系于某张帧，
// 勾「同时写到每一条谱」就逐条系定。下面是成员列表（同一套筛选条与大图）。

import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import clsx from "clsx";
import type { GalleryItem, Series } from "@/lib/gallery/types";
import { RLAB, dirLabel, fmtT, fmtTs, num } from "@/lib/gallery/format";
import { anchorOf, initialNeighbours, inside, itemGeometry, seriesAnchorOf, spanOf } from "@/lib/gallery/context";
import { seriesMembers, shortFn } from "@/lib/gallery/series";
import { viewSearch } from "@/lib/gallery/route";
import type { GalleryModel } from "./useGalleryData";
import { useMarksStore } from "./marksStore";
import { FramePanel, type Side } from "./ContextStage";
import { ListView } from "./ListView";
import { H1, Sub } from "./ListHeads";
import { SMALL_BTN } from "./bits";
import { SeriesFigureActions } from "./figures/SeriesFigureActions";

function SeriesHead({ sid, s, mem }: { sid: string; s: Series; mem: GalleryItem[] }) {
  const setSeries = useMarksStore((st) => st.setSeries);
  const tagTable = useMarksStore((st) => st.doc.tags);
  const nav = useNavigate();
  const [name, setName] = useState(s.name);
  const [note, setNote] = useState(s.note || "");
  const [armed, setArmed] = useState(false);
  const noteTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const noteRef = useRef(note);
  noteRef.current = note;

  const cur = () => useMarksStore.getState().doc.series[sid];
  const save = (patch: Partial<Series>) => {
    const c = cur();
    if (c) setSeries(sid, { ...c, ...patch });
  };
  const saveNote = () => {
    if (noteTimer.current) {
      clearTimeout(noteTimer.current);
      noteTimer.current = null;
    }
    const c = cur();
    if (c && (c.note || "") !== noteRef.current) save({ note: noteRef.current });
  };
  useEffect(() => () => {
    if (noteTimer.current) saveNote();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(t);
  }, [armed]);

  const c = { f: 0, s: 0, g: 0 };
  for (const it of mem) c[it.k]++;
  const a = mem[0];
  const b = mem[mem.length - 1];
  const missing = (s.ids || []).length - mem.length;

  return (
    <div>
      <Link to={{ search: `?${viewSearch({ v: "marked" })}` }} className="text-sm text-mast-accent hover:underline">
        ← 已标记
      </Link>
      {a && (
        <>
          {" · "}
          <Link to={{ search: `?${viewSearch({ v: "dir", d: a.d })}` }} className="text-sm text-mast-accent hover:underline">
            {dirLabel(a.d)}
          </Link>
        </>
      )}
      <H1>▤ {s.name}</H1>
      <Sub>
        {mem.length} 个（{c.f ? `帧 ${c.f} ` : ""}
        {c.s ? `谱 ${c.s} ` : ""}
        {c.g ? `网格 ${c.g}` : ""}）
        {a && b ? ` · ${num(a)} → ${num(b)} · ${fmtT(a.t)} → ${fmtT(b.mt || b.t1 || b.t)}` : ""} · 系列号 {sid}
        {missing > 0 && <span className="text-mast-danger"> · {missing} 个成员不在图库里</span>}
      </Sub>
      <div className="my-1.5 mb-2.5 flex flex-wrap items-center gap-x-2.5 gap-y-2">
        <input
          value={name}
          size={44}
          title="系列名，改完回车或点别处保存"
          onChange={(e) => setName(e.target.value)}
          onBlur={() => save({ name: name.trim() || s.name })}
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          }}
          className="rounded-[3px] border border-mast-border bg-mast-panel px-[5px] py-0.5 text-[13px] text-mast-text"
        />
        <span className="inline-flex gap-1">
          {[1, 2, -1].map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => save({ r: s.r === r ? 0 : r })}
              className={clsx(SMALL_BTN, s.r === r && "!border-mast-accent !bg-mast-accent !text-mast-accent-ink")}
            >
              {RLAB[String(r)]}
            </button>
          ))}
        </span>
        <span className="inline-flex flex-wrap gap-[3px]">
          {tagTable.map((t) => {
            const on = (s.tags || []).includes(t);
            return (
              <button
                key={t}
                type="button"
                onClick={() => save({ tags: on ? s.tags.filter((x) => x !== t) : [...(s.tags || []), t] })}
                className={clsx(SMALL_BTN, "text-xs", on && "!border-mast-accent !bg-mast-accent !text-mast-accent-ink")}
              >
                {t}
              </button>
            );
          })}
        </span>
        <textarea
          value={note}
          placeholder="系列备注"
          onChange={(e) => {
            setNote(e.target.value);
            if (noteTimer.current) clearTimeout(noteTimer.current);
            noteTimer.current = setTimeout(saveNote, 700);
          }}
          onBlur={saveNote}
          className="h-9 min-w-[240px] flex-1 resize-y rounded-[3px] border border-mast-border bg-mast-panel px-[5px] py-0.5 text-[13px] text-mast-text"
        />
        <button
          type="button"
          className={clsx(SMALL_BTN, armed && "!border-mast-danger !text-mast-danger")}
          onClick={() => {
            if (!armed) {
              setArmed(true);
              return;
            }
            setSeries(sid, null);
            nav({ search: `?${viewSearch({ v: "marked" })}` });
          }}
        >
          {armed ? "再点一次确认删除（成员的单条标记不动）" : "删除系列"}
        </button>
      </div>
      {s.anchor && (
        <Sub>
          ⌖ 位置系于 {shortFn(s.anchor.fn)}（{s.anchor.desc}）
        </Sub>
      )}
    </div>
  );
}

function SeriesContext({ sid, s, mem, model }: { sid: string; s: Series; mem: GalleryItem[]; model: GalleryModel }) {
  const setSeries = useMarksStore((st) => st.setSeries);
  const updateMark = useMarksStore((st) => st.updateMark);
  const sp = useMemo(() => mem.filter((it) => it.k === "s" || it.k === "g"), [mem]);
  const { t0, t1 } = spanOf(sp);
  const [sides, setSides] = useState(() => initialNeighbours(model.frames, t0, t1, s.anchor?.id));
  const [each, setEach] = useState(false);
  if (!sp.length) return null;

  const onStep = (side: Side, step: number) =>
    setSides((x) => ({ ...x, [side]: Math.max(0, Math.min(model.frames.length - 1, x[side] + step)) }));
  const onAnchor = (side: Side) => {
    const f = model.frames[sides[side]];
    const cur = useMarksStore.getState().doc.series[sid];
    if (!f || !cur) return;
    if (cur.anchor?.id === f.id) {
      setSeries(sid, { ...cur, anchor: null });
      return;
    }
    setSeries(sid, { ...cur, anchor: seriesAnchorOf(f, t0, t1) });
    if (each) for (const it of sp) updateMark(it, { anchor: anchorOf(it, f) });
  };

  const panel = (side: Side) => {
    const f = model.frames[sides[side]];
    const pts = f
      ? sp.map((it, i) => {
          const p = itemGeometry(it, f).pts[0] ?? { u: 0.5, v: 0.5 };
          return { ...p, lab: String(i + 1) };
        })
      : [];
    return (
      <FramePanel
        side={side}
        frame={f}
        t0={t0}
        t1={t1}
        noun="系列"
        pts={pts}
        poly={null}
        anchoredId={s.anchor?.id}
        onStep={onStep}
        onAnchor={onAnchor}
        posText={() => ` · ${pts.filter(inside).length}/${pts.length} 条在视野内`}
      />
    );
  };

  return (
    <div className="mb-2.5 mt-1.5 rounded-[3px] border border-mast-border bg-mast-panel px-3 py-2.5 text-mast-text">
      <div className="text-[12.5px]">
        <b className="font-semibold">系列里 {sp.length} 条谱的位置</b>（按时间编号 1…{sp.length}；{fmtTs(t0)}–{fmtTs(t1)}）{" "}
        <label className="text-mast-muted">
          <input type="checkbox" checked={each} onChange={(e) => setEach(e.target.checked)} /> 系定时同时写到每一条谱
        </label>
      </div>
      <div className="mt-2 grid max-w-[1000px] grid-cols-2 gap-3.5 max-[760px]:grid-cols-1">
        {panel("prev")}
        {panel("next")}
      </div>
    </div>
  );
}

export function SeriesView({ sid, model }: { sid: string; model: GalleryModel }) {
  const s = useMarksStore((st) => st.doc.series[sid]);
  // 按成员 id 的**内容** memo：改系列名、备注、评级都会换一个新的 series 对象，
  // 若成员表跟着换引用，下面的卡片网格就回到前 150 张、滚动位置丢失。
  const idsKey = s ? s.ids.join("\n") : "";
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const mem = useMemo(() => seriesMembers(s, model.items), [idsKey, model.items]);

  if (!s) {
    return (
      <div>
        <Link to={{ search: `?${viewSearch({ v: "marked" })}` }} className="text-sm text-mast-accent hover:underline">
          ← 已标记
        </Link>
        <H1>这个系列不存在或已删除</H1>
      </div>
    );
  }
  return (
    <ListView
      view="series"
      model={model}
      base={mem}
      head={
        <>
          <SeriesHead key={sid} sid={sid} s={s} mem={mem} />
          <SeriesFigureActions key={`fig:${sid}`} sid={sid} mem={mem} model={model} />
        </>
      }
      below={<SeriesContext key={sid} sid={sid} s={s} mem={mem} model={model} />}
    />
  );
}
