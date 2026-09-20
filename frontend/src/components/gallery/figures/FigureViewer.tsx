// 出图页的全尺寸查看：暗底遮罩，←/→ 在同一类别内翻，Esc 关，顶上是下载链接。
//
// 与大图（Lightbox）一样挂到 document.body：MAST 的滚动容器是 AppLayout 的 <main>，
// fixed 遮罩留在里面的话，滚轮会把底下的页面一起滚走。键盘监听只注册一次，「现在第几张」
// 从 latest ref 读——连按 → 时闭包里的旧序号会让翻页原地打转。

import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { fileLinks, fullImageUrl, KIND_LABEL, type FigureEntry } from "@/lib/gallery/figures";
import { DARK_BTN } from "../bits";

export function FigureViewer({
  entries,
  index,
  onIndex,
  onClose,
}: {
  entries: readonly FigureEntry[];
  index: number;
  onIndex: (j: number) => void;
  onClose: () => void;
}) {
  const j = Math.max(0, Math.min(entries.length - 1, index));
  const e = entries[j];
  const latest = useRef({ j, n: entries.length, onIndex, onClose });
  latest.current = { j, n: entries.length, onIndex, onClose };

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
      const cur = latest.current;
      if (ev.key === "Escape") {
        ev.preventDefault();
        cur.onClose();
      } else if (ev.key === "ArrowRight" || ev.key === "ArrowDown") {
        ev.preventDefault();
        if (cur.j < cur.n - 1) cur.onIndex(cur.j + 1);
      } else if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") {
        ev.preventDefault();
        if (cur.j > 0) cur.onIndex(cur.j - 1);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  // 列表刷新后这一类空了：自己关掉，不留一个黑屏。
  useEffect(() => {
    if (!entries.length) latest.current.onClose();
  }, [entries.length]);

  if (!e) return null;
  const url = fullImageUrl(e);

  const node = (
    <div className="fixed inset-0 z-[60] flex flex-col bg-[rgba(10,10,12,.93)] text-[#eee]">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-[#333] px-3.5 py-2 text-[13px]">
        <b className="break-all font-mono">{e.title || e.base}</b>
        <span className="text-[#9aa4af]">
          {KIND_LABEL[e.kind]} · {e.created || "—"}
        </span>
        <span className="font-mono text-[#9aa4af]">
          {j + 1} / {entries.length}
        </span>
        <span className="ml-auto flex flex-wrap items-center gap-1.5">
          {fileLinks(e.files).map((f) => (
            <a key={f.name} href={f.url} download={f.name} className={DARK_BTN} title={f.name}>
              {f.label}
            </a>
          ))}
          <button type="button" className={DARK_BTN} disabled={j === 0} onClick={() => onIndex(j - 1)}>
            ← 上一张
          </button>
          <button type="button" className={DARK_BTN} disabled={j >= entries.length - 1} onClick={() => onIndex(j + 1)}>
            下一张 →
          </button>
          <button type="button" className={DARK_BTN} title="关闭（Esc）" onClick={onClose}>
            ✕
          </button>
        </span>
      </div>
      <div
        className="flex min-h-0 flex-1 items-center justify-center overflow-auto overscroll-contain p-3.5"
        onClick={(ev) => {
          if (ev.target === ev.currentTarget) onClose();
        }}
      >
        {url ? (
          <img src={url} alt={e.base} className="max-h-full max-w-full bg-white object-contain" />
        ) : (
          <div className="text-[#9aa4af]">这一项没有 PNG/JPG（只有数据文件），用上面的链接下载。</div>
        )}
      </div>
    </div>
  );
  return createPortal(node, document.body);
}
