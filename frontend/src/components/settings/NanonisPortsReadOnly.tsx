import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Badge, Card, DegradedNote, ErrorNote, Spinner } from "../ui";

// 设置 → 硬件连接 — READ-ONLY Nanonis host + 4 TCP port display (ITEM 5).
//
// The 硬件 editor (host/port edit + reconnect + sensors) lives behind 「高级管理
// → 硬件」 (PIN-gated, owned by the admin agent). Here in 设置 we only SHOW the
// resolved values so the operator can see them without unlocking admin. We read
// GET /api/nanonis/connection (now reports the defaults 6501/6502/6503/6504 even
// when no pool is wired) and render a small read-only table — host + 4 ports +
// live per-port status — plus the note pointing at 高级管理 for edits.
//
// We deliberately do NOT reuse admin/HardwareManager here: this Settings-side
// surface is owned by the settings agent and must never edit hardware.

// role (as returned by /api/nanonis/connection) → human label, canonical order
// matching the four Nanonis TCP roles. Mirrors the admin PORT_ROLES labels so the
// two surfaces read identically.
const PORT_ROLES: { key: string; label: string }[] = [
  { key: "main", label: "主控端口 (main)" },
  { key: "monitor", label: "监控端口 (monitor)" },
  { key: "data", label: "数据端口 (data)" },
  { key: "emergency", label: "急停端口 (emergency)" },
];

function StatusBadge({ connected }: { connected?: boolean | null }) {
  if (connected == null) return null;
  return connected ? <Badge tone="AUTO">已连</Badge> : <Badge tone="DANGEROUS">未连</Badge>;
}

export function NanonisPortsReadOnly() {
  // Slow poll so the live status badges stay fresh without hammering the API.
  // Degrade-safe: the endpoint never probes hardware from the standalone process.
  const connQ = useQuery({
    queryKey: ["nanonis", "connection"],
    refetchInterval: 5000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/nanonis/connection");
      if (error) throw error;
      return data;
    },
  });

  if (connQ.isPending) return <Spinner />;
  if (connQ.isError) return <ErrorNote error={connQ.error} />;

  const conn = connQ.data;
  const host = conn?.host && String(conn.host).trim() ? String(conn.host) : "—";
  const portByRole = new Map(
    (conn?.ports ?? []).map((p) => [p.role, p]),
  );

  return (
    <Card>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-medium text-mast-text">Nanonis 连接（只读）</span>
        {conn && !conn.degraded &&
          (conn.connected ? <Badge tone="AUTO">在线</Badge> : <Badge tone="WARN">离线</Badge>)}
      </div>

      {conn?.degraded && <DegradedNote what="Nanonis 连接" />}

      <div className="space-y-3">
        {/* 主机地址 — read-only */}
        <div className="flex flex-col gap-1 text-sm">
          <span className="text-mast-muted">主机地址 Host</span>
          <span className="rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 font-mono text-mast-text">
            {host}
          </span>
        </div>

        {/* 4 端口 — read-only values + live status */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {PORT_ROLES.map((r) => {
            const ps = portByRole.get(r.key);
            const portVal = ps?.port != null ? String(ps.port) : "—";
            return (
              <div key={r.key} className="flex flex-col gap-1 text-sm">
                <span className="text-mast-muted">{r.label}</span>
                <div className="flex items-center gap-2">
                  <span className="rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 font-mono text-mast-text">
                    {portVal}
                  </span>
                  <StatusBadge connected={ps?.connected} />
                </div>
              </div>
            );
          })}
        </div>

        <p className="rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 text-xs text-mast-muted">
          这些为只读默认值（端口 6501–6504）。如需修改主机 / 端口或重新连接，请前往
          「高级管理 → 硬件」（需 PIN 解锁）。
        </p>
      </div>
    </Card>
  );
}
