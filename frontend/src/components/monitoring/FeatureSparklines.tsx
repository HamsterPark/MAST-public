import type { components } from "@/api/schema";
import { Sparkline } from "@/components/vision/Sparkline";
import { EmptyNote } from "@/components/ui";
import { fmtCurrent } from "@/lib/units";
import { METRIC_TILES, metricSeries, type MetricUnit } from "@/lib/monitoring";

type FeatureRow = components["schemas"]["FeatureRow"];

// 特征瓦片 — six numbers with their recent history.
//
// The series comes from /api/monitoring/features, NOT from status.latest: a
// sparkline needs history and `latest` is one point. It also means a WS push
// (which carries five of the ~35 columns) can never blank a tile — the tiles
// simply follow the REST series at its own, slower cadence.

function fmtMetric(v: number, unit: MetricUnit): string {
  switch (unit) {
    case "A":
      return fmtCurrent(v, { placeholder: "—" });
    case "hz":
      return `${v.toPrecision(3)} Hz`;
    case "ratio":
      return v.toFixed(3);
    default:
      return Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)
        ? v.toExponential(2)
        : v.toPrecision(4);
  }
}

export function FeatureSparklines({
  rows,
  degraded,
  detail,
}: {
  rows: FeatureRow[];
  degraded?: boolean;
  detail?: string | null;
}) {
  // 「还没采到」与「读不到」是两句话。前者是等待，后者是故障 —— 而两者的
  // `rows` 都是空数组。不分开的话，一次 `features_query` 失败会显示成
  // 「采集满一段后即可看到」，于是用户安心地等一个永远不会来的东西。
  if (degraded) {
    return (
      <EmptyNote
        label={detail ? `读不到特征数据：${detail}` : "读不到特征数据（不是「还没采到」）。"}
      />
    );
  }
  if (!rows.length) {
    return <EmptyNote label="尚无特征数据——采集满一段后即可看到。" />;
  }
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
      {METRIC_TILES.map((tile) => {
        const series = metricSeries(rows, tile.key);
        const last = series.length ? series[series.length - 1]! : null;
        return (
          <div
            key={tile.key}
            className="rounded-mast-card border border-mast-border bg-mast-panel p-3"
            title={tile.hint}
          >
            <div className="text-xs text-mast-muted">{tile.label}</div>
            <div className="mt-1 flex items-end justify-between gap-2">
              <span className="font-mono text-lg text-mast-text">
                {last == null ? "—" : fmtMetric(last, tile.unit)}
              </span>
              <Sparkline values={series} />
            </div>
            <div className="mt-1 text-[11px] leading-snug text-mast-faint">{tile.hint}</div>
          </div>
        );
      })}
    </div>
  );
}
