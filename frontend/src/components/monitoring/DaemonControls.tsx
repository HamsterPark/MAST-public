import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import type { components } from "@/api/schema";
import { Badge } from "@/components/ui";
import { Button } from "@/components/controls";
import { fmtBytes, fmtSeconds, stateView } from "@/lib/monitoring";

type MonitoringStatus = components["schemas"]["MonitoringStatus"];

// 采集守护进程的状态与启停。
//
// The stop button is not a hazard: stopping only releases the oscilloscope
// subscription, and the store stays fully readable afterwards — browsing history
// with the daemon down is a supported state, not a degraded one.

function Chip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="inline-flex items-baseline gap-1.5 rounded-mast-badge border border-mast-border bg-mast-panel-2 px-2 py-1 text-xs">
      <span className="text-mast-muted">{label}</span>
      <span className="font-mono text-mast-text">{children}</span>
    </span>
  );
}

export function DaemonControls({
  status,
  onStart,
  onStop,
  starting,
  stopping,
  wsWarn,
}: {
  status: MonitoringStatus | undefined;
  onStart: () => void;
  onStop: () => void;
  starting: boolean;
  stopping: boolean;
  /** /ws/events is down — push unavailable, polling carrying the page. */
  wsWarn: boolean;
}) {
  const degraded = status?.degraded ?? false;
  const sv = stateView(status?.state);
  const running = status?.running ?? false;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={sv.tone}>{sv.label}</Badge>
        {status?.connected === false && <Badge tone="DANGEROUS">未连接仪器</Badge>}
        {status?.enabled_in_settings === false && <Badge tone="default">设置中已关闭</Badge>}
        {status?.alerts_enabled === false && <Badge tone="default">告警已关闭</Badge>}

        <span className="ml-auto flex items-center gap-2">
          <Button
            variant="primary"
            onClick={onStart}
            loading={starting}
            disabled={degraded || running || starting || stopping}
          >
            启动采集
          </Button>
          <Button onClick={onStop} loading={stopping} disabled={degraded || !running || starting || stopping}>
            停止采集
          </Button>
          <Link
            to="/settings/general"
            className="rounded-mast-ctl px-2 py-1 text-xs text-mast-accent hover:bg-mast-accent-soft"
          >
            阈值与保留策略设置 →
          </Link>
        </span>
      </div>

      <p className="text-xs text-mast-muted">
        {sv.hint}
        {status?.detail ? ` — ${status.detail}` : ""}
        {status?.retry_in_s ? `（${fmtSeconds(status.retry_in_s)} 后重试）` : ""}
      </p>

      <div className="flex flex-wrap gap-2">
        <Chip label="采集策略">{status?.strategy ?? "—"}</Chip>
        <Chip label="采样率">{status?.fs_hz ? `${status.fs_hz.toFixed(0)} Hz` : "—"}</Chip>
        <Chip label="通道">{status?.channel_name || "—"}</Chip>
        <Chip label="段长">{status?.segment_seconds ? fmtSeconds(status.segment_seconds) : "—"}</Chip>
        <Chip label="本次已采">{status?.segments_done ?? 0} 段</Chip>
        <Chip label="累计缺口">{fmtSeconds(status?.gaps_total_s ?? 0)}</Chip>
        <Chip label="库内段数">
          {status?.segments_total ?? 0}
          {status?.segments_on_disk != null && `（${status.segments_on_disk} 段留有原始数据）`}
        </Chip>
        <Chip label="已钉住">{status?.pinned_count ?? 0} 段</Chip>
        <Chip label="占用">
          {fmtBytes(status?.store_bytes)}
          {status?.retention_gb ? ` / ${status.retention_gb} GB` : ""}
        </Chip>
        <Chip label="保留">
          {status?.retention_hours ? `${status.retention_hours} 小时` : "—"}
        </Chip>
      </div>

      {wsWarn && (
        <p className="text-xs text-mast-warn">
          实时推送不可用，已回退为轮询——数据仍在更新，只是最多慢几秒。
        </p>
      )}
    </div>
  );
}
