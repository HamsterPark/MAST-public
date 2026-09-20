import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, ErrorNote, DegradedNote, EmptyNote } from "@/components/ui";
import { Button } from "@/components/controls";

// 图像拼接 / Mosaic — typed replacement for gui/exp_puzzle.build_mosaic_panel.
// Point at a directory of .sxm scans of ONE sample, choose a channel, and the
// core assembles them into one big-canvas overview placed by real stage xy
// (scan_offset / scan_range), returning a b64 PNG (the canvas ndarray never
// crosses the wire). Switching samples = point at a new directory = fresh canvas.
// Reproduces every old control: 目录 / 通道 / 配色 / 逐行展平 / 递归子目录 / 样品标签.

// Mirrors exp_puzzle._CMAPS.
const CMAPS = ["viridis", "copper", "cividis", "inferno", "magma", "gray", "turbo", "afmhot"];

function fmtSize(w?: number | null, h?: number | null): string {
  if (w == null || h == null) return "—";
  return `${(w * 1e9).toFixed(0)}×${(h * 1e9).toFixed(0)} nm`;
}

export function MosaicPanel() {
  const [directory, setDirectory] = useState("");
  const [channel, setChannel] = useState("Z");
  const [cmap, setCmap] = useState("viridis");
  const [recursive, setRecursive] = useState(false);
  const [lineNormalize, setLineNormalize] = useState(true);
  const [label, setLabel] = useState("");

  const mut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/experimental/mosaic", {
        body: {
          directory,
          channel,
          recursive,
          line_normalize: lineNormalize,
          cmap,
          label,
        },
      });
      if (error) throw error;
      return data;
    },
  });

  const result = mut.data;

  return (
    <div className="space-y-3">
      <p className="text-xs text-mast-muted">
        图像拼接 / Mosaic — 把<strong>同一样品</strong>的多张 .sxm 扫描，按各自的真实 xy 台面坐标
        （scan_offset / scan_range）拼到一张大画布上，得到表面概览。换样品时换一个目录即生成新画布。返回画布
        PNG（ndarray 永不过线）。
      </p>

      <Card className="space-y-3">
        <label className="flex flex-col gap-1 text-xs">
          <span className="text-mast-muted">扫描目录（含 .sxm 文件，留空则用默认保存目录）</span>
          <input
            className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 font-mono text-sm"
            placeholder={"例如 D:\\data\\sampleA"}
            value={directory}
            onChange={(e) => setDirectory(e.target.value)}
          />
        </label>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">通道</span>
            <input
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={channel}
              onChange={(e) => setChannel(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">配色</span>
            <select
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={cmap}
              onChange={(e) => setCmap(e.target.value)}
            >
              {CMAPS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-mast-muted">样品标签（写入文件名）</span>
            <input
              className="rounded border border-mast-border bg-mast-bg px-2 py-1.5 text-sm"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
            />
          </label>
          <div className="flex flex-col justify-end gap-1.5 text-xs">
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={lineNormalize}
                onChange={(e) => setLineNormalize(e.target.checked)}
              />
              逐行展平 (Z 推荐)
            </label>
            <label className="inline-flex items-center gap-2">
              <input type="checkbox" checked={recursive} onChange={(e) => setRecursive(e.target.checked)} />
              递归子目录
            </label>
          </div>
        </div>

        <div>
          <Button variant="primary" onClick={() => mut.mutate()} disabled={mut.isPending}>
            {mut.isPending ? "拼接中…" : "生成拼图"}
          </Button>
        </div>

        {mut.isError && <ErrorNote error={mut.error} />}
        {result && result.degraded && <DegradedNote what="拼图" />}
        {result && !result.degraded && !result.ok && (
          <EmptyNote label={result.error || "未找到可拼接的有效扫描。"} />
        )}
        {result && !result.degraded && result.ok && (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-mast-muted">
              <span>
                已放置 {result.placed}/{result.n_input}
              </span>
              {result.canvas_px && (
                <span>
                  画布 {result.canvas_px[0]}×{result.canvas_px[1]} px
                </span>
              )}
              {result.res_m_per_px > 0 && (
                <span>分辨率 {(result.res_m_per_px * 1e9).toFixed(2)} nm/px</span>
              )}
              {result.angle_warning && (
                <span className="text-mast-warn">⚠ 含旋转扫描（按外接框放置，拼接为近似）</span>
              )}
            </div>
            {result.image_b64 ? (
              <>
                <img
                  src={`data:image/png;base64,${result.image_b64}`}
                  alt="mosaic canvas"
                  className="max-w-full rounded-lg border border-mast-border"
                />
                <div className="flex flex-wrap items-center gap-2">
                  <a
                    href={`data:image/png;base64,${result.image_b64}`}
                    download={`mosaic_${label || "canvas"}.png`}
                    className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30"
                  >
                    保存拼图 (PNG)
                  </a>
                  {/* Parity flag: OLD "保存拼图 (PNG + NPY)" also wrote a .npy/.json
                      of the raw canvas ndarray to disk via save_mosaic(); that array
                      never crosses the wire, so PNG download is the web equivalent and
                      the NPY/JSON sidecar is a desktop-core-only action (no endpoint). */}
                  <span className="text-xs text-mast-muted">
                    （NPY/JSON 原始画布另存仅在桌面端进行）
                  </span>
                </div>
              </>
            ) : (
              <EmptyNote label="拼接成功但未生成预览图。" />
            )}
            {result.scans_meta && result.scans_meta.length > 0 && (
              <details className="text-xs">
                <summary className="cursor-pointer text-mast-muted">
                  输入扫描 ({result.scans_meta.length})
                </summary>
                <ul className="mt-2 space-y-1 font-mono text-mast-muted">
                  {result.scans_meta.map((s, i) => (
                    <li key={`${s.path}-${i}`} className="truncate" title={s.path}>
                      {s.path} · {s.channel} · {fmtSize(s.w, s.h)}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
