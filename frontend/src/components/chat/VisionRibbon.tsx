import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import clsx from "clsx";
import { api } from "@/api/client";

// Vision pulse ribbon — a thin status bar reflecting GET /api/vision/pulse
// (tip-quality DINO score, alert level, trend, recent event counts). Mirrors the
// old Gradio 视觉脉冲 ribbon. Read-only; polls every 4 s.

const ALERT_STYLE: Record<string, { bg: string; text: string; label: string }> = {
  idle: { bg: "bg-mast-panel/60", text: "text-mast-muted", label: "空闲" },
  info: { bg: "bg-mast-info-bg", text: "text-mast-info", label: "信息" },
  warn: { bg: "bg-mast-warn-bg", text: "text-mast-warn", label: "注意" },
  critical: { bg: "bg-mast-danger-bg", text: "text-mast-danger", label: "警报" },
};

const TREND_ARROW: Record<string, string> = {
  idle: "·",
  rising: "↗",
  steady: "→",
  falling: "↘",
};

export function VisionRibbon({ onOpenBuffer }: { onOpenBuffer?: () => void } = {}) {
  const q = useQuery({
    queryKey: ["chat", "vision-pulse"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/vision/pulse");
      if (error) throw error;
      return data;
    },
    refetchInterval: 4000,
  });

  const d = q.data;
  const alert = d?.alert_level ?? "idle";
  const style = ALERT_STYLE[alert] ?? ALERT_STYLE.idle ?? { bg: "bg-mast-panel/60", text: "text-mast-muted", label: alert };
  const degraded = d?.degraded ?? false;

  return (
    <div
      className={clsx(
        "flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-mast-border px-3 py-1.5 text-xs",
        style.bg,
      )}
    >
      <span className={clsx("font-semibold", style.text)}>
        视觉脉冲 · {style.label}
      </span>
      {degraded ? (
        <span className="text-mast-muted">（视觉缓冲未启用）</span>
      ) : (
        <>
          <span className="text-mast-muted">
            针尖质量：
            <span className="ml-1 font-mono tabular-nums text-mast-text">
              {d?.tip_quality ?? "unknown"}
            </span>
            {/* SAFE overrides every tip verdict to "good" at the producer — mark it,
                so this reads as a mode, not as a measurement. */}
            {d?.safe_mode && (
              <span
                className="ml-1 text-mast-warn"
                title="安全模式：针尖判定被统一覆写为「良好」，非实测值"
              >
                （安全模式覆写）
              </span>
            )}
          </span>
          <span className="text-mast-muted">
            DINO：
            <span className="ml-1 font-mono tabular-nums text-mast-text">
              {d?.dino_score != null ? d.dino_score.toFixed(3) : "—"}
            </span>
          </span>
          <span className="text-mast-muted">
            趋势：<span className="ml-1 text-mast-text">{TREND_ARROW[d?.trend ?? "idle"]}</span>
          </span>
          <span className="text-mast-muted">
            近窗事件：
            <span className="ml-1 font-mono tabular-nums text-mast-text">{d?.recent_count ?? 0}</span>
          </span>
          {(d?.critical_count ?? 0) > 0 && (
            <span className="font-mono tabular-nums text-mast-danger">
              严重 {d?.critical_count}
            </span>
          )}
          <span className="text-mast-muted">
            速率：
            <span className="ml-1 font-mono tabular-nums text-mast-text">
              {(d?.rate_per_s ?? 0).toFixed(2)}/s
            </span>
          </span>
        </>
      )}
      {/* When rendered inside the Chat page, this switches to the Vision Buffer
          sub-tab (old chat_subtabs jump); standalone it routes to /vision. */}
      {onOpenBuffer ? (
        <button
          type="button"
          onClick={onOpenBuffer}
          className="ml-auto text-mast-accent hover:underline"
        >
          打开 Vision Buffer →
        </button>
      ) : (
        <Link to="/vision" className="ml-auto text-mast-accent hover:underline">
          打开 Vision Buffer →
        </Link>
      )}
    </div>
  );
}
