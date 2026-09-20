// 近期标注帧 — what to say (or draw) in a tile that has no picture.
//
// The grid used to print a bare「无图像」whenever the backend handed over a frame
// with neither a thumbnail nor a file path . Several very different
// situations reach that state and the operator cannot tell them apart:
//
//   · an in-scan pulse (`feature_of_interest`) that analysed a partial frame
//     nobody persisted. This one is CORRECT and deliberate — /api/vision/recent
//     refuses to borrow the previous scan's picture for it, because showing an
//     older image next to a live judgement is the falsified history of #76/#78.
//     There is nothing to fix here except saying so.
//   · a completed scan whose file could not be located. That one is a real
//     failure, and it should not read the same as the case above.
//   · an event that never had a picture because it did not come from the
//     camera at all — the current monitor publishes CRITICALs from a waveform
//     . It has no frame and never will, but it DOES have data, and
//     「无图像」 was answering a question nobody asked.
//
// So the tile explains which, and where a curve can honestly be drawn it says
// so via `source` instead of a caption. The rule lives here, as a pure function,
// because the failure mode is silent: a wrong-but-plausible caption looks
// exactly like a right one in a screenshot.

export interface FrameLike {
  kind?: string | null;
  image_b64?: string | null;
  file_path?: string | null;
  cause_ref?: string | null;
  /** Buffer event seqno — the key `/api/vision/recent-frame/{seqno}` takes. */
  seqno?: number | null;
}

/**
 * What produced an event, parsed from `cause_ref`.
 *
 * The producers, all verified in the Python (2026-08-04):
 *   · `current_monitor#<segment_id>` / bare `current_monitor` — monitoring/alerts.py
 *   · `scan#<scan_id>`                                        — vision/scan_monitor.py
 *   · `tip_status#<seqno>`                                    — buffer/service.py
 *   · `skill:<name>`                                          — webui/tip_shape_records.py
 */
export type CauseSource =
  | { kind: "current_monitor"; segmentId: number | null }
  | { kind: "scan"; scanId: string }
  | { kind: "tip_status"; seqno: number | null }
  | { kind: "skill"; name: string }
  | { kind: "unknown"; raw: string };

function intOrNull(s: string): number | null {
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return Number.isSafeInteger(n) ? n : null;
}

export function parseCauseRef(causeRef: string | null | undefined): CauseSource | null {
  const raw = (causeRef ?? "").trim();
  if (!raw) return null;
  if (raw === "current_monitor") return { kind: "current_monitor", segmentId: null };
  if (raw.startsWith("current_monitor#")) {
    return { kind: "current_monitor", segmentId: intOrNull(raw.slice("current_monitor#".length)) };
  }
  if (raw.startsWith("scan#")) return { kind: "scan", scanId: raw.slice("scan#".length) };
  if (raw === "tip_status") return { kind: "tip_status", seqno: null };
  if (raw.startsWith("tip_status#")) {
    return { kind: "tip_status", seqno: intOrNull(raw.slice("tip_status#".length)) };
  }
  if (raw.startsWith("skill:")) return { kind: "skill", name: raw.slice("skill:".length) };
  return { kind: "unknown", raw };
}

export interface FramePlaceholder {
  /** Short caption drawn in the tile. */
  label: string;
  /** Longer explanation, shown on hover. Empty when the label says it all. */
  hint: string;
  /**
   * Set when the tile should draw the event's OWN data instead of the caption.
   * Never a borrowed picture — only data this very event points at.
   */
  source?: { kind: "current_monitor"; segmentId: number };
}

/**
 * Caption for a tile with no rendered image, or null when the frame HAS an image
 * and the caller should draw it instead.
 */
export function framePlaceholder(frame: FrameLike): FramePlaceholder | null {
  if (frame.image_b64) return null;

  // A path but no thumbnail: the file is known and the render failed or was
  // skipped. That is a different sentence from "there is no picture".
  if (frame.file_path) {
    return {
      label: "缩略图未生成",
      hint: `文件在 ${frame.file_path}，但这一帧没有渲染出缩略图。`,
    };
  }

  // WHERE the event came from beats WHAT type it is: the current monitor
  // publishes tip_quality_drop, and so does the vision path. Same kind, opposite
  // answers to「有没有图」.
  const src = parseCauseRef(frame.cause_ref);
  if (src?.kind === "current_monitor") {
    if (src.segmentId !== null) {
      return {
        label: `电流监控 · 第 ${src.segmentId} 段`,
        hint:
          `这条判断来自隧道电流监控，不是扫描画面 —— 它没有、也不会有帧。` +
          `下面画的是第 ${src.segmentId} 段的电流波形，也就是判据本身读的那段数据。`,
        source: { kind: "current_monitor", segmentId: src.segmentId },
      };
    }
    return {
      label: "电流监控 · 无分段",
      hint:
        "这条判断来自隧道电流监控，本就没有扫描画面；这一条也没有记下段号，" +
        "所以连波形都定位不到。段号在 cause_ref 里，缺了它就只能这么说。",
    };
  }

  switch ((frame.kind ?? "").toLowerCase()) {
    case "feature_of_interest":
      return {
        label: "扫描中判读 · 未留存画面",
        hint:
          "这是扫描进行中对局部画面做的判读，那一帧没有被保存下来。" +
          "这里不会借用上一张扫描图充数 —— 拿旧图配新判断会让历史看起来" +
          "像是当时就长这样。",
      };
    case "scan_complete":
      return {
        label: "找不到对应扫描文件",
        hint:
          "这一帧记的是一次扫描完成，但事件里没有文件路径，也没能在最近的扫描文件里" +
          "找到对应的一张。文件可能已被移动、删除，或存在没有被搜索到的目录里。",
      };
    case "tip_quality_drop":
    case "tip_shape_verdict":
      return { label: "针尖判读 · 无画面", hint: "针尖状态的判读依据是电流/Z 读数，本来就不带图像。" };
    case "sensor_fault":
    case "vision_error":
      return { label: "故障事件 · 无画面", hint: "这是一条故障记录，没有随附图像。" };
    default:
      return {
        label: "无画面",
        hint: "这条事件没有随附图像，也没有可指向的文件。",
      };
  }
}
