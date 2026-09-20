import { useState } from "react";
import { framePlaceholder, type FrameLike } from "@/lib/visionFrames";
import { SegmentCurveThumb } from "@/components/monitoring/SegmentCurveThumb";
import { Modal } from "@/components/controls";

// The thumbnail slot of a 近期标注帧 tile.
//
// Extracted because ChatPage and VisionPage each carried a byte-identical copy,
// and is about the text this renders — two copies means fixing it
// once and having the operator meet the old wording on the other page. #62 (the
// current-monitor curve) lands here for the same reason. #93 (click to enlarge)
// is the third: putting the lightbox in the two PAGES would have been the same
// mistake a third time.
//
// ── 点开放大 ────────────────────────────────────────────
// Could not enlarge. This rendered a bare <img> with
// no handler, and neither page wrapped it in one. The current-monitor tiles were
// the exception all along (SegmentCurveThumb is a <Link> to the full waveform),
// which is why only the PICTURE tiles were dead ends.
//
// The enlarged picture comes from `/api/vision/recent-frame/{seqno}` and NOT
// from scaling up `image_b64`: that thumbnail is 160 px. Stretching it in the
// browser would be a 放大 button that produces the same 160 px with bigger
// squares — the shape of "看着做了其实没做".
//
// The modal opens ONLY on click. It is never opened by an effect: an
// auto-opened full-screen overlay would hide the readings exactly when human
// intervention needs them visible, and a picture is never urgent enough to cover the
// readings. `test/recentFrameZoom.test.ts` holds that line.

function FrameLightbox({
  frame,
  onClose,
}: {
  frame: FrameLike;
  onClose: () => void;
}) {
  const [failed, setFailed] = useState(false);
  const seqno = frame.seqno;
  const name = frame.file_path?.split(/[\\/]/).pop();
  return (
    <Modal open onClose={onClose} wide title={`标注帧 #${seqno} · ${frame.kind || "事件"}`}>
      {failed ? (
        <div className="py-10 text-center text-sm text-mast-muted">
          这一帧的图取不回来了。
          <div className="mt-1 text-xs text-mast-faint">
            事件还在，图不在 —— 文件可能已被移动或删除。缩略图是之前渲染并缓存下来的。
          </div>
        </div>
      ) : (
        <img
          src={`/api/vision/recent-frame/${encodeURIComponent(String(seqno))}`}
          alt={frame.kind ? `标注帧 #${seqno}（${frame.kind}）` : `标注帧 #${seqno}`}
          onError={() => setFailed(true)}
          /* object-contain, not cover: the tile crops to a square grid cell, and
             seeing the whole frame is the entire point of opening this. */
          className="mx-auto max-h-[72vh] w-auto max-w-full object-contain"
        />
      )}
      {name && (
        <div className="mt-3 truncate text-center font-mono text-xs text-mast-muted" title={frame.file_path ?? ""}>
          {name}
        </div>
      )}
    </Modal>
  );
}

export function RecentFrameThumb({ frame, alt }: { frame: FrameLike; alt?: string }) {
  const [zoom, setZoom] = useState(false);
  const placeholder = framePlaceholder(frame);

  if (!placeholder) {
    // A tile whose seqno never arrived cannot be asked for by the endpoint, so
    // it stays a plain picture rather than a button that opens an empty box.
    const canZoom = frame.seqno != null;
    const img = (
      <img
        src={`data:image/png;base64,${frame.image_b64}`}
        alt={alt || frame.kind || "标注帧"}
        className="aspect-square w-full rounded border border-mast-border object-cover"
      />
    );
    if (!canZoom) return img;
    return (
      <>
        <button
          type="button"
          onClick={() => setZoom(true)}
          title="点击放大"
          className="block w-full rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-mast-accent"
        >
          {img}
        </button>
        {zoom && <FrameLightbox frame={frame} onClose={() => setZoom(false)} />}
      </>
    );
  }

  // No image, but this event points at data of its OWN. Draw that — never
  // something borrowed from a neighbouring event .
  if (placeholder.source?.kind === "current_monitor") {
    return <SegmentCurveThumb segId={placeholder.source.segmentId} hint={placeholder.hint} />;
  }

  return (
    <div
      title={placeholder.hint}
      className="flex aspect-square w-full items-center justify-center rounded border border-dashed border-mast-border px-2 text-center text-xs leading-snug text-mast-muted"
    >
      {placeholder.label}
    </div>
  );
}
