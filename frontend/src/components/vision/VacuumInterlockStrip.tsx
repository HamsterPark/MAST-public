import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, ErrorNote } from "@/components/ui";

// 粗动真空互锁状态条 + 用户签署。
//
// 为什么签署只在这里、没有对应的 agent 工具：签署的内容是「MAST 观察不到的物理
// 状态」（腔体已通大气 / 已抽到高真空但真空计不可用）。让模型签它，只是把同一份
// 无知洗成了一个许可。这是用户的动作，按定义如此。
//
// 状态条在拿不到裁决时显示**红色**，不是灰色：一个算不出结论的互锁指示灯绝不能
// 长得像绿灯。

// Must mirror mast.core.vacuum_interlock.ATTESTATION_REASONS — the backend
// rejects an unknown reason rather than accepting a free-text one, because a
// signature that does not say WHAT was attested is not auditable afterwards.
const REASONS = [
  { value: "vented_to_atmosphere", label: "已通大气（腔体在常压）" },
  { value: "high_vacuum_gauge_unavailable", label: "已抽至高真空，但真空计不可用" },
] as const;

export function VacuumInterlockStrip() {
  const qc = useQueryClient();
  const [reason, setReason] = useState<string>(REASONS[0].value);
  const [signedBy, setSignedBy] = useState("");
  const [ttlHours, setTtlHours] = useState(8);
  const [open, setOpen] = useState(false);

  const q = useQuery({
    queryKey: ["coarse-map", "vacuum"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/coarse-map/vacuum");
      if (error) throw error;
      return data;
    },
    refetchInterval: 15000,
  });

  const invalidate = () => {
    void qc.invalidateQueries({ queryKey: ["coarse-map"] });
  };

  const attest = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/coarse-map/vacuum/attest", {
        body: { reason, signed_by: signedBy, ttl_hours: ttlHours, note: "" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: invalidate,
  });

  const revoke = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.DELETE("/api/coarse-map/vacuum/attest", {});
      if (error) throw error;
      return data;
    },
    onSuccess: invalidate,
  });

  const v = q.data;
  const allow = v?.allow === true;

  return (
    <Card className="space-y-2 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-semibold">粗动真空互锁</span>
        <span
          className={`rounded px-2 py-0.5 text-xs font-semibold ${
            allow
              ? "bg-mast-accent/15 text-mast-accent"
              : "bg-mast-warn/15 text-mast-warn"
          }`}
        >
          {allow ? "允许粗动" : "禁止粗动"}
        </span>
      </div>

      {q.isError && <ErrorNote error={q.error} />}

      <p className="text-xs text-mast-muted">{v?.reason || "读取中…"}</p>

      {v && (
        <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-mast-muted">
          {v.pressure_pa != null && (
            <span className="tabular-nums">
              压强 {v.pressure_pa.toExponential(2)} Pa
              {v.age_s != null ? `（${v.age_s.toFixed(0)} 秒前）` : ""}
            </span>
          )}
          {v.over_range && <span className="text-mast-warn">超量程</span>}
          <span>模式 {v.mode || "—"}</span>
          {v.attested && (
            <span className="text-mast-accent">
              已签署
              {v.attestation_remaining_h != null
                ? `（剩 ${v.attestation_remaining_h.toFixed(1)} h）`
                : ""}
            </span>
          )}
        </div>
      )}

      {/* 只在真空计答不出时才提供签署入口——有有效读数时签字是多余且危险的。 */}
      {v && !allow && v.mode !== "gauge_only" && (
        <div className="rounded border border-mast-border bg-mast-panel p-2">
          {!open ? (
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="text-xs text-mast-muted underline hover:text-mast-text"
            >
              签署「当前气压安全」…
            </button>
          ) : (
            <div className="space-y-2">
              <p className="text-xs text-mast-muted">
                只有你能确认 MAST 看不到的东西。签署<b>有时效</b>，
                并且<b>重启即失效</b>——重启本身就是一次换场，重签只要十秒，
                而一个活过了它所依据条件的许可，代价是一整个压电叠堆。
              </p>
              <select
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                className="w-full rounded border border-mast-border bg-mast-bg px-2 py-1 text-xs"
              >
                {REASONS.map((r) => (
                  <option key={r.value} value={r.value}>
                    {r.label}
                  </option>
                ))}
              </select>
              <div className="flex gap-2">
                <input
                  value={signedBy}
                  onChange={(e) => setSignedBy(e.target.value)}
                  placeholder="署名"
                  className="min-w-0 flex-1 rounded border border-mast-border bg-mast-bg px-2 py-1 text-xs"
                />
                <input
                  type="number"
                  min={0.5}
                  max={24}
                  step={0.5}
                  value={ttlHours}
                  onChange={(e) => setTtlHours(Number(e.target.value) || 8)}
                  className="w-20 rounded border border-mast-border bg-mast-bg px-2 py-1 text-xs tabular-nums"
                />
                <span className="self-center text-xs text-mast-muted">小时</span>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => attest.mutate()}
                  disabled={attest.isPending}
                  className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-xs hover:text-mast-text disabled:opacity-50"
                >
                  {attest.isPending ? "签署中…" : "签署"}
                </button>
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-xs text-mast-muted"
                >
                  取消
                </button>
              </div>
              {attest.isError && <ErrorNote error={attest.error} />}
            </div>
          )}
        </div>
      )}

      {v?.attested && (
        <button
          type="button"
          onClick={() => revoke.mutate()}
          disabled={revoke.isPending}
          className="text-xs text-mast-muted underline hover:text-mast-warn disabled:opacity-50"
        >
          {revoke.isPending ? "撤销中…" : "撤销签署（例如开始抽气时）"}
        </button>
      )}
    </Card>
  );
}
