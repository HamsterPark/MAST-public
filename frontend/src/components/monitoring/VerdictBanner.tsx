import clsx from "clsx";
import type { components } from "@/api/schema";
import { fmtCurrent } from "@/lib/units";
import { fmtSeconds, verdictView } from "@/lib/monitoring";

type MonitoringStatus = components["schemas"]["MonitoringStatus"];

// 当前判定横幅 — the one thing on the page an operator reads without stopping.
//
// Its hardest requirement is negative: when the newest segment has aged past the
// stale threshold this must NOT keep showing the last verdict. A frozen "正常"
// is indistinguishable from a live "正常", and the whole point of the monitor is
// to be believed. Stale → neutral grey 「无数据」 with the age spelled out.

const TONE_BOX: Record<string, string> = {
  AUTO: "border-mast-auto-border bg-mast-auto-bg text-mast-auto",
  INFO: "border-mast-info-border bg-mast-info-bg text-mast-info",
  WARN: "border-mast-warn-border bg-mast-warn-bg text-mast-warn",
  DANGEROUS: "border-mast-danger-border bg-mast-danger-bg text-mast-danger",
  default: "border-mast-border bg-mast-panel-2 text-mast-muted",
};

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide opacity-70">{label}</div>
      <div className="font-mono text-sm">{value}</div>
    </div>
  );
}

export function VerdictBanner({
  status,
  stale,
  ageS,
}: {
  status: MonitoringStatus | undefined;
  /** Newest segment older than MONITORING_STALE_AFTER_MS. */
  stale: boolean;
  /** Seconds since the newest segment; null when there has never been one. */
  ageS: number | null;
}) {
  const latest = status?.latest ?? null;
  const noData = !latest || stale;

  if (noData) {
    const why = !latest
      ? "还没有采集到任何一段电流"
      : `最近一段已是 ${fmtSeconds(ageS)} 前的数据`;
    return (
      <div className={clsx("rounded-mast-card border px-4 py-3", TONE_BOX.default)}>
        <div className="flex items-baseline gap-3">
          <span className="text-lg font-semibold">无数据</span>
          <span className="text-sm">{why}</span>
        </div>
        <p className="mt-1 text-xs opacity-80">
          横幅不显示已冻结的旧判定——上一次的读数即使还在库里，也不能代表此刻的针尖状态。
        </p>
      </div>
    );
  }

  const v = verdictView(latest.verdict);
  const suppressed = latest.verdict === "suppressed";

  return (
    <div className={clsx("rounded-mast-card border px-4 py-3", TONE_BOX[v.tone] ?? TONE_BOX.default)}>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="text-lg font-semibold">{v.label}</span>
        <span className="text-sm">
          第 {latest.seg_id} 段 · {new Date(latest.ts * 1000).toLocaleTimeString("zh-CN", { hour12: false })}
        </span>
        {suppressed && (
          <span className="text-xs opacity-90">
            该段落在修针 / 电脉冲窗口内，噪声是我们自己造成的，故不报警
          </span>
        )}
      </div>
      <div className="mt-2.5 grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
        <Stat label="RMS 噪声" value={fmtCurrent(latest.rms_detrended_a, { placeholder: "—" })} />
        <Stat label="平均电流" value={fmtCurrent(latest.mean_a, { placeholder: "—" })} />
        <Stat label="最小" value={fmtCurrent(latest.min_a, { placeholder: "—" })} />
        <Stat label="最大" value={fmtCurrent(latest.max_a, { placeholder: "—" })} />
      </div>
    </div>
  );
}
