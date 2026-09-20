import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Badge, EmptyNote, Spinner } from "@/components/ui";
import { verdictView } from "@/lib/monitoring";
import { SegmentCurveThumb } from "@/components/monitoring/SegmentCurveThumb";

type AlertRow = components["schemas"]["AlertRow"];

// 告警历史。The list endpoint deliberately omits the evidence PNG — a page of
// 50 alerts each carrying a base64 plot is a megabyte of payload nobody looked
// at. Opening a row fetches that one image, and react-query caches it forever
// because an alert's evidence is written once and never changes.

function EvidenceImage({ alertId }: { alertId: number }) {
  const q = useQuery({
    queryKey: ["monitoring", "alert-evidence", alertId],
    staleTime: Infinity,
    gcTime: 10 * 60_000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/alerts/{alert_id}/evidence", {
        params: { path: { alert_id: alertId } },
      });
      if (error) throw error;
      return data;
    },
  });

  if (q.isPending) return <Spinner label="载入证据图…" />;
  if (q.isError) return <p className="text-xs text-mast-danger">证据图载入失败。</p>;
  if (!q.data?.ok || !q.data.png_b64) {
    return <p className="text-xs text-mast-muted">{q.data?.detail || "该告警没有可用的证据图。"}</p>;
  }
  return (
    <img
      src={`data:image/png;base64,${q.data.png_b64}`}
      alt="告警证据：时域波形与频谱"
      className="max-w-full rounded-lg border border-mast-border bg-mast-bg"
    />
  );
}

/**
 * What to show when an alert row is opened. THREE outcomes, not two.
 *
 * tip_quality_drop 事件详情缺图像时，能不能换成曲线图顶上。
 *
 * `evidence_available` answers a narrower question than the UI was asking: it
 * means "a matplotlib PNG exists on disk", not "there is something to look at".
 * It is false for every WARN (deliberately — WARNs are frequent and rendering
 * each one would put matplotlib on the monitor's hot path), and false for a
 * CRITICAL whose render lost the 60 s rate-limit race. Meanwhile `seg_id` sits
 * in the same row object, and the segment behind it is drawable **forever**:
 * the raw `.npy` is swept at `cm_keep_hours`, but the min/max envelope lives in
 * the row and rows are permanent.
 *
 * So the old two-way branch showed a paragraph of apology while the waveform the
 * judgement was actually made on was one field away. The third case — no PNG
 * AND no segment — keeps the apology, because then there genuinely is nothing.
 */
function AlertEvidence({ alert }: { alert: AlertRow }) {
  if (alert.evidence_available) return <EvidenceImage alertId={alert.id} />;

  if (alert.seg_id != null) {
    return (
      <div className="space-y-1">
        <SegmentCurveThumb
          segId={alert.seg_id}
          hint={`${alert.rule} 判定所依据的电流段`}
          wide
        />
        <p className="text-xs text-mast-muted">
          这是判定所依据的那一段电流本身。没有渲染证据图
          {alert.level === "warn" ? "——WARN 一律不渲染，是刻意的" : ""}，
          但波形一直在，点开可看完整时域与频谱。
        </p>
      </div>
    );
  }

  return (
    <p className="text-xs text-mast-muted">
      该告警既没有证据图，也没有关联的电流段——文字判定仍然有效。
    </p>
  );
}

/**
 * 「谁知道了这件事」—— **两个主体,两个格子**。
 *
 * `delivered_agent` = 这条进过正在操作仪器的 agent 的上下文。
 * `acked`           = 人在这里点掉了。
 *
 * 合成一个「已处理」会让两个问题都问不出来:2026-08-08 那次事故里,监控规则
 * 报得又快又准,而 agent 一无所知又扫了 13 分钟 —— 当时**没有任何字段**能回答
 * 「agent 到底看见没看见」。所以这一列刻意分开显示,而且「没看见」是有颜色的。
 */
function KnownBy({ alert }: { alert: AlertRow }) {
  return (
    <span className="inline-flex items-center gap-1">
      {alert.delivered_agent ? (
        <span className="text-mast-accent" title="agent 已在上下文里看到这条">🤖</span>
      ) : alert.level === "critical" ? (
        <span className="text-mast-danger" title="agent 还没看到这条 CRITICAL">🤖✗</span>
      ) : (
        <span className="text-mast-muted" title="agent 未看到(WARN 可能被折叠或按配置不打扰)">·</span>
      )}
      {alert.acked ? (
        <span className="text-mast-muted" title="已由人确认">✓</span>
      ) : null}
    </span>
  );
}

function AckButton({ alert }: { alert: AlertRow }) {
  const qc = useQueryClient();
  const m = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/monitoring/alerts/{alert_id}/ack", {
        params: { path: { alert_id: alert.id } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["monitoring"] });
    },
  });

  if (alert.acked) {
    return <p className="text-xs text-mast-muted">已确认。</p>;
  }
  return (
    <div className="space-y-1">
      <button
        type="button"
        disabled={m.isPending}
        onClick={(e) => {
          e.stopPropagation();
          m.mutate();
        }}
        className="rounded-md border border-mast-border px-2 py-1 text-xs text-mast-text hover:bg-mast-panel-2 disabled:opacity-50"
      >
        {m.isPending ? "确认中…" : "我知道了(确认)"}
      </button>
      {/* 说清楚这个按钮**不**做什么 —— 否则用户会以为点掉就不打扰 agent 了。 */}
      <p className="text-xs text-mast-muted">
        只是记下「人已知悉」。**不会**让这条不再送给 agent,也不代表 agent 看过。
      </p>
      {m.isError ? <p className="text-xs text-mast-danger">确认失败,请重试。</p> : null}
    </div>
  );
}

export function AlertsTable({
  alerts,
  degraded,
  detail,
}: {
  alerts: AlertRow[];
  degraded: boolean;
  detail?: string | null;
}) {
  const [openId, setOpenId] = useState<number | null>(null);

  // ⚠️ **degraded 时不许说「无告警记录」。** 2026-08-15 起 `alerts_query` 查询
  // 失败会抛，路由把它翻成 `degraded=true` + `detail` —— 而在此之前它自吞异常
  // 回 `([], 0)`，这块面板于是把「库读不到」画成了「暂无告警——这是好消息」。
  // 原文案还替它认了一个死因（「监控模块未装载」），那是两件事里的一件。
  if (degraded) {
    return (
      <EmptyNote
        label={
          detail
            ? `读不到告警记录：${detail}`
            : "读不到告警记录（监控模块未装载，或查询失败）——这不等于没有告警。"
        }
      />
    );
  }
  if (!alerts.length) return <EmptyNote label="暂无告警——这是好消息。" />;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-mast-muted">
            <th className="py-1 pr-3 font-normal">时间</th>
            <th className="py-1 pr-3 font-normal">级别</th>
            <th className="py-1 pr-3 font-normal">规则</th>
            <th className="py-1 pr-3 font-normal">说明</th>
            <th className="py-1 pr-3 font-normal" title="🤖=agent 已看到 / ✓=人已确认">
              谁知道了
            </th>
            <th className="py-1 font-normal">段</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((a) => {
            const v = verdictView(a.level);
            const open = openId === a.id;
            return [
              <tr
                key={a.id}
                onClick={() => setOpenId(open ? null : a.id)}
                className="cursor-pointer border-t border-mast-border hover:bg-mast-panel-2"
                title={
                  a.evidence_available
                    ? "点击展开证据图"
                    : a.seg_id != null
                      ? "点击展开触发段的电流曲线"
                      : "点击展开"
                }
              >
                <td className="whitespace-nowrap py-1.5 pr-3 text-xs text-mast-muted">
                  {new Date(a.ts * 1000).toLocaleString("zh-CN", { hour12: false })}
                </td>
                <td className="py-1.5 pr-3">
                  <Badge tone={v.tone}>{v.label}</Badge>
                </td>
                <td className="py-1.5 pr-3 font-mono text-xs text-mast-text">{a.rule}</td>
                <td className="py-1.5 pr-3 text-mast-text">{a.summary_zh}</td>
                <td className="py-1.5 pr-3 text-xs">
                  <KnownBy alert={a} />
                </td>
                <td className="py-1.5 font-mono text-xs text-mast-muted">
                  {a.seg_id ?? "—"}
                  {/* 两种「有东西可看」要分得开：◧ 是渲染好的证据图，∿ 是
                      触发段的原始波形。都标成同一个符号，就等于把「这条
                      CRITICAL 有证据图」和「这条 WARN 只有波形」说成一回事。 */}
                  {a.evidence_available ? (
                    <span className="ml-1 text-mast-accent" title="有证据图">◧</span>
                  ) : a.seg_id != null ? (
                    <span className="ml-1 text-mast-muted" title="有触发段波形">∿</span>
                  ) : null}
                </td>
              </tr>,
              open ? (
                <tr key={`${a.id}-ev`} className="border-t border-mast-border bg-mast-bg/40">
                  <td colSpan={6} className="p-3">
                    <div className="space-y-3">
                      <AlertEvidence alert={a} />
                      <AckButton alert={a} />
                    </div>
                  </td>
                </tr>
              ) : null,
            ];
          })}
        </tbody>
      </table>
    </div>
  );
}
