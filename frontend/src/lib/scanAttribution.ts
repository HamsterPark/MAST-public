// 按实验 / 样品分组浏览的数据整形。
//
// GET /api/scans/latest walks the disk, and a path alone cannot say which
// experiment claimed a file. GET /api/scans/attribution reads the v2
// `file_locations` table, which is the record of that. This file turns the flat
// row list into the experiment→sample→files shape the browser renders.

export interface AttributionEntryLike {
  sha256: string;
  rel_path: string;
  abs_path?: string | null;
  origin_path?: string | null;
  root_kind?: string;
  sample_id?: string | null;
  source?: string;
  size_bytes?: number;
  status?: string;
  ingested_at?: string | null;
}

export interface SampleGroup<T extends AttributionEntryLike = AttributionEntryLike> {
  sampleId: string | null;
  files: T[];
  bytes: number;
}

/** The path to render/preview for a row, or null when neither is usable. */
export function displayPath(entry: AttributionEntryLike): string | null {
  // abs_path is reconstructed from the experiment folder layout and is null when
  // the folder could not be located (root moved/renamed). origin_path is then
  // the only path we have — it points at the file the ingest copied FROM, which
  // may itself be gone, so a caller still has to survive a missing file.
  return entry.abs_path || entry.origin_path || null;
}

/** The file's name, from whichever path exists. */
export function fileName(entry: AttributionEntryLike): string {
  const p = displayPath(entry) || entry.rel_path || "";
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

/**
 * Group rows by sample, newest-ingested first within each group.
 *
 * Rows with no sample_id are kept in their own group rather than dropped: files
 * ingested before the sample layer existed, or while no sample was active, are
 * real files and hiding them would make the grouped view show less than the
 * experiment holds.
 */
export function groupBySample<T extends AttributionEntryLike>(
  rows: readonly T[],
): SampleGroup<T>[] {
  const order: (string | null)[] = [];
  const byId = new Map<string | null, T[]>();
  for (const r of rows) {
    const key = r.sample_id || null;
    if (!byId.has(key)) {
      byId.set(key, []);
      order.push(key);
    }
    byId.get(key)!.push(r);
  }
  // Un-sampled files last: they are the leftovers, not the headline.
  order.sort((a, b) => (a === null ? 1 : 0) - (b === null ? 1 : 0));
  return order.map((sampleId) => {
    const files = byId.get(sampleId)!;
    return {
      sampleId,
      files,
      bytes: files.reduce((sum, f) => sum + (f.size_bytes || 0), 0),
    };
  });
}

/**
 * Normalised keys under which this row's file might be known elsewhere.
 *
 * Case-folded because the two halves of the app reach the same file by
 * different spellings: one from config, one from the Nanonis session path. A
 * caller matching a disk listing against attribution rows has to compare on
 * these, not on the raw strings.
 */
export function pathKeys(entry: AttributionEntryLike): string[] {
  const out: string[] = [];
  for (const p of [entry.abs_path, entry.origin_path]) {
    if (p) out.push(p.replace(/\\/g, "/").toLowerCase());
  }
  return out;
}

/** Short human summary for a sample group heading. */
export function groupSummary(g: { files: readonly unknown[]; bytes: number }): string {
  const mb = g.bytes / (1024 * 1024);
  const size = mb >= 1 ? `${mb.toFixed(1)} MB` : `${(g.bytes / 1024).toFixed(0)} KB`;
  return `${g.files.length} 个文件 · ${size}`;
}
