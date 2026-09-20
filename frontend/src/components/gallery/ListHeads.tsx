// 列表视图的头部：某目录（带「已过完」与目录备注）/ 全部X / 最新一批
// （旧版兼容格式 app.js renderList 开头那三段）。

import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { GalleryItem, Kind } from "@/lib/gallery/types";
import { KLAB, dirLabel, dirnameOf, isDateSeg, lastSeg } from "@/lib/gallery/format";
import { viewSearch } from "@/lib/gallery/route";
import { useMarksStore } from "./marksStore";

export function H1({ children }: { children: React.ReactNode }) {
  return <h1 className="mb-0.5 mt-1 text-[1.35rem] font-semibold text-mast-text">{children}</h1>;
}

export function Sub({ children }: { children: React.ReactNode }) {
  return <div className="mb-2.5 text-[13px] text-mast-muted">{children}</div>;
}

export function BackToDirs() {
  return (
    <Link to={{ search: `?${viewSearch({ v: "dirs" })}` }} className="text-sm text-mast-accent hover:underline">
      ← 目录
    </Link>
  );
}

/** 某目录的头：计数、原始目录、「已过完」、目录备注（700 ms 去抖，离开时补存）。 */
export function DirHead({ d, base }: { d: string; base: GalleryItem[] }) {
  const dv = useMarksStore((s) => s.doc.days[d]);
  const setDay = useMarksStore((s) => s.setDay);
  const [note, setNote] = useState(dv?.note ?? "");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const noteRef = useRef(note);
  noteRef.current = note;

  // 服务端文档换了（别的标签页改过）且没在输入时，把备注同步过来。
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current && !timer.current) setNote(dv?.note ?? "");
  }, [dv?.note]);

  const saveNote = () => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
    const cur = useMarksStore.getState().doc.days[d];
    if ((cur?.note ?? "") === noteRef.current) return;
    useMarksStore.getState().setDay(d, { done: cur?.done ?? false, note: noteRef.current });
  };
  useEffect(() => () => {
    if (timer.current) saveNote();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const c = { f: 0, s: 0, g: 0 };
  let nd = 0;
  for (const it of base) {
    c[it.k]++;
    if (it.dup) nd++;
  }
  const sample = base[0];
  const dayWord = isDateSeg(lastSeg(d)) ? "本日" : "本目录";

  return (
    <div>
      <BackToDirs />
      <H1>{dirLabel(d)}</H1>
      <Sub>
        {c.f} 帧{nd ? `（其中重复保存 ${nd}）` : ""} · {c.s} 条谱 · {c.g} 个网格谱 · 原始目录{" "}
        <span className="font-mono">{sample ? dirnameOf(sample.p) : d}</span> · 点图放大后可用键盘连续标记
      </Sub>
      <div className="my-1.5 flex items-start gap-2.5">
        <label className="flex shrink-0 items-center gap-1 whitespace-nowrap text-[13px] text-mast-text">
          <input
            type="checkbox"
            checked={!!dv?.done}
            onChange={(e) => setDay(d, { done: e.target.checked, note: noteRef.current })}
          />
          {dayWord}已过完
        </label>
        <textarea
          value={note}
          placeholder={`${dayWord === "本日" ? "这一天" : "这个目录"}的备注：针尖状态、做了什么、看到了什么…`}
          onFocus={() => (focused.current = true)}
          onBlur={() => {
            focused.current = false;
            saveNote();
          }}
          onChange={(e) => {
            setNote(e.target.value);
            if (timer.current) clearTimeout(timer.current);
            timer.current = setTimeout(saveNote, 700);
          }}
          className="h-9 min-h-9 flex-1 resize-y rounded-[3px] border border-mast-border bg-mast-panel px-[5px] py-0.5 text-[13px] text-mast-text"
        />
      </div>
    </div>
  );
}

export function AllHead({ k }: { k: Kind }) {
  return (
    <div>
      <H1>全部{KLAB[k]}</H1>
      <Sub>跨目录，按目录与时间排；方括号里是所在目录。</Sub>
    </div>
  );
}

export function NewHead({ lastBatch }: { lastBatch: string }) {
  return (
    <div>
      <BackToDirs />
      <H1>最新一批（{lastBatch} 加入）</H1>
      <Sub>这次更新新到的全部文件。</Sub>
    </div>
  );
}
