// 自动拷贝副本的显示。
//
// MAST copies every scan into the experiment folder and leaves the original
// where Nanonis wrote it. That is worth keeping — it is what makes an experiment
// folder self-contained. What it did to the 数据 tab is show one measurement as
// several cards with an identical name, size and timestamp, and no way to tell
// which was which.
//
// The server folds them; this file is how the fold gets DISCLOSED. A collapse
// the operator cannot see is indistinguishable from files going missing, and
// this codebase has already recorded that failure twice from the other side
// (#53/#60 — a listing quietly showing less than the disk holds).

export interface ScanLocationLike {
  path: string;
  kind: string;
}

export interface ScanEntryLike {
  path: string;
  kind?: string;
  copies?: number;
  locations?: ScanLocationLike[];
}

/**
 * The folder a file sits in, e.g. `20260327` — what tells two cards apart.
 *
 * There are TWO kinds of repetition in this listing and they need opposite
 * treatment:
 *
 *   * byte-identical copies of one measurement (the automatic ingest) — folded
 *     into one entry by the server, disclosed by the ×N badge above;
 *   * **different measurements that merely share a name** — Nanonis names files
 *     `unnamed0033.sxm` per session, so 20260319 and 20260327 both have one.
 *     Folding those would hide real data, so they stay as separate cards — and
 *     then two cards read `unnamed0033.sxm` with nothing to choose between them.
 *
 * The thumbnail answers it visually; this answers it in words. Measured on this
 * machine: 36 basenames occur more than once across session folders, with
 * different sizes and timestamps every time.
 */
export function parentFolder(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.length >= 2 ? parts[parts.length - 2]! : "";
}

/** 位置类型的中文名。Vocabulary matches the server's `file_locations.root_kind`. */
export function locationKindLabel(kind: string | undefined): string {
  switch (kind) {
    case "experiment":
    case "experiment_folder":
      return "实验文件夹";
    case "quarantine":
      return "隔离区";
    case "origin":
      return "原始位置";
    default:
      // Not "未知位置": an unrecognised kind is a NEW kind we have not been
      // taught, and calling it unknown reads like the file is lost.
      return kind ? String(kind) : "位置";
  }
}

/** `×3`, or null when there is only one copy and nothing to disclose. */
export function copyBadge(entry: ScanEntryLike): string | null {
  const n = entry.copies ?? 1;
  return n > 1 ? `×${n}` : null;
}

/**
 * Every place this measurement exists, ordered original-first.
 *
 * Single-copy entries carry no `locations` (the server omits the list to keep
 * the payload small), so their one path is reconstructed here — a caller
 * rendering "where is this file" must not get an empty answer for a file that
 * plainly exists.
 */
export function copyLocations(entry: ScanEntryLike): ScanLocationLike[] {
  if (entry.locations?.length) {
    const rank = (k: string) => (k === "origin" ? 0 : k === "experiment" ? 1 : 2);
    return entry.locations.slice().sort((a, b) => rank(a.kind) - rank(b.kind));
  }
  return [{ path: entry.path, kind: entry.kind ?? "origin" }];
}

/**
 * True when nothing has ingested this measurement into an experiment folder.
 *
 * Used by the grouped view to show the leftovers: raw session data that belongs
 * to no experiment yet. Note the asymmetry — a file WITH an experiment copy is
 * definitely attributed, while "no experiment copy" only means no copy was
 * found on the paths we walked. The grouped view therefore labels this section
 * 未归属 rather than claiming the file belongs to nothing.
 */
export function isUnattributed(entry: ScanEntryLike): boolean {
  return !copyLocations(entry).some(
    (l) => l.kind === "experiment" || l.kind === "experiment_folder",
  );
}
