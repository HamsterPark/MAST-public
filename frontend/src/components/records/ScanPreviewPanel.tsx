import { useState } from "react";
import type { components } from "@/api/schema";
import { Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Modal } from "@/components/controls";
import { copyLocations, locationKindLabel } from "@/lib/scanCopies";
import { SpectrumChart } from "./SpectrumChart";
import {
  ALL_FLATTEN_MODES,
  flattenLabel,
  useScanPreview,
  type FlattenMode,
} from "./scanPreviewQuery";

// 单个文件的预览面板（右栏）。
//
// Two things this gained on 2026-08-21:
//
//  * **去衬底 controls.** A raw topograph is mostly sample tilt — over a 100 nm
//    frame a fraction of a degree is nanometres of z, while the corrugation the
//    operator is looking for is picometres. The panel now says which background
//    was subtracted, and `auto` explains its choice in the instrument's own
//    words (it forwards `scan_prep.plan_for`'s reasons).
//  * **Spectra are curves.** A .dat opens the interactive chart instead of a
//    thumbnail of its first two columns.
//
// The lightbox opens ONLY on click, never from an effect — is a
// full-screen overlay appearing over readings the operator was reading, and
// `test/recentFrameZoom.test.ts` holds that line for the vision tiles.

type ScanFileEntry = components["schemas"]["ScanFileEntry"];

const SPECTRUM_EXTS = [".dat", ".txt", ".csv", ".asc", ".tsv"];

function fmtNm(v?: number | null): string {
  if (v == null) return "—";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} µm` : `${v.toFixed(1)} nm`;
}

function fmtBias(v?: number | null): string {
  if (v == null) return "—";
  return Math.abs(v) < 1 ? `${(v * 1000).toFixed(0)} mV` : `${v.toFixed(3)} V`;
}

/** 点开放大。Asks for the big render rather than scaling the small one up —
 *  stretching a 256 px thumbnail is a 放大 button that produces bigger squares. */
function PreviewLightbox({
  path,
  flatten,
  onClose,
}: {
  path: string;
  flatten: FlattenMode;
  onClose: () => void;
}) {
  const big = useScanPreview(path, { size: 1024, flatten });
  const name = path.split(/[\\/]/).pop();
  return (
    <Modal open onClose={onClose} wide title={name || "预览"}>
      {big.isPending && <Spinner />}
      {big.isError && <ErrorNote error={big.error} />}
      {big.data?.image ? (
        <img
          src={big.data.image}
          alt={name || path}
          className="mx-auto max-h-[72vh] w-auto max-w-full object-contain"
        />
      ) : (
        !big.isPending && (
          <div className="py-10 text-center text-sm text-mast-muted">
            这张图取不回来了{big.data?.detail ? `：${big.data.detail}` : ""}。
          </div>
        )
      )}
      <div className="mt-3 break-all text-center font-mono text-xs text-mast-muted">{path}</div>
    </Modal>
  );
}

export function ScanPreviewPanel({
  entry,
  path,
  flatten,
  onFlattenChange,
}: {
  /** The listing row, when the file came from one (carries copy info). */
  entry?: ScanFileEntry | null;
  path: string;
  flatten: FlattenMode;
  onFlattenChange: (m: FlattenMode) => void;
}) {
  const [zoom, setZoom] = useState(false);
  const ext = ("." + (path.split(".").pop() || "")).toLowerCase();
  const isSpectrum = SPECTRUM_EXTS.includes(ext);
  const prev = useScanPreview(path, { size: 512, flatten, });

  return (
    <Card>
      <div className="mb-2 break-all font-mono text-xs text-mast-muted">{path}</div>

      {isSpectrum ? (
        <SpectrumChart path={path} />
      ) : (
        <>
          {/* 去衬底 switch. `auto` is here and NOT in the grid: it measures the
              frame (~3 s the first time per file) and thirty cards would queue
              thirty measurements. */}
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
            <div className="inline-flex overflow-hidden rounded-mast-ctl border border-mast-border">
              {ALL_FLATTEN_MODES.map((m) => (
                <button
                  key={m}
                  onClick={() => onFlattenChange(m)}
                  title={m === "auto" ? "测量这一帧再决定怎么处理（首次约 3 秒）" : undefined}
                  className={
                    "px-2.5 py-1 " +
                    (flatten === m
                      ? "bg-mast-accent-soft font-medium text-mast-accent"
                      : "text-mast-muted hover:text-mast-text")
                  }
                >
                  {flattenLabel(m)}
                </button>
              ))}
            </div>
            {/* What was ACTUALLY applied. `auto` resolves to a concrete method,
                and a method that fails falls back to raw — either way the label
                must not keep claiming the frame was flattened. */}
            {prev.data?.flatten && prev.data.flatten !== flatten && (
              <span className="text-mast-muted">
                实际：{flattenLabel(prev.data.flatten)}
              </span>
            )}
            {prev.isFetching && <Spinner label="处理中…" />}
          </div>

          {prev.isError && <ErrorNote error={prev.error} />}
          {prev.data?.degraded && <DegradedNote what="扫描预览" />}
          {prev.data && !prev.data.rendered && !prev.data.degraded && (
            <EmptyNote
              label={
                prev.data.found
                  ? `无法渲染${prev.data.detail ? `：${prev.data.detail}` : ""}`
                  : "文件不存在"
              }
            />
          )}

          {prev.data?.image && (
            <button
              type="button"
              onClick={() => setZoom(true)}
              title="点击放大"
              className="block w-full rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-mast-accent"
            >
              <img
                src={prev.data.image}
                alt={path}
                className="max-h-[60vh] w-full rounded border border-mast-border bg-black/40 object-contain"
              />
            </button>
          )}

          {/* Frame scale + bias, straight off the file header. Without these the
              picture has no size: two frames of the same surface at 20 nm and
              200 nm look identical once the tilt is out. */}
          {prev.data?.rendered && (
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-mast-muted">
              {prev.data.channel && <span>通道 {prev.data.channel}</span>}
              {prev.data.width_nm != null && (
                <span>
                  {fmtNm(prev.data.width_nm)} × {fmtNm(prev.data.height_nm)}
                </span>
              )}
              {prev.data.bias_v != null && <span>偏压 {fmtBias(prev.data.bias_v)}</span>}
            </div>
          )}

          {/* Why auto chose what it chose — the instrument's own words. */}
          {(prev.data?.flatten_why?.length ?? 0) > 0 && (
            <ul className="mt-2 space-y-0.5 text-xs text-mast-faint">
              {(prev.data?.flatten_why ?? []).map((w) => (
                <li key={w}>· {w}</li>
              ))}
            </ul>
          )}

          {zoom && (
            <PreviewLightbox path={path} flatten={flatten} onClose={() => setZoom(false)} />
          )}
        </>
      )}

      {/* Copies of THIS measurement. Shown in the panel too, not only on the
          card, because this is where an operator lands after clicking. */}
      {(entry?.copies ?? 1) > 1 && (
        <div className="mt-2 space-y-1 rounded border border-mast-border bg-mast-bg/60 p-1.5 text-[11px]">
          <div className="text-mast-muted">自动拷贝：同一份数据存在 {entry!.copies} 处</div>
          {copyLocations(entry!).map((loc) => (
            <div key={loc.path} className="break-all font-mono text-mast-faint" title={loc.path}>
              <span className="mr-1 font-sans text-mast-muted">[{locationKindLabel(loc.kind)}]</span>
              {loc.path}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
