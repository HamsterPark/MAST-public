import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, Field, TextField, useToast } from "@/components/controls";
import { DeviceScanButton, InstrumentStateCard } from "@/components/admin/DeviceScanner";
import { settingsWriteProblem, type SettingsPatch } from "@/lib/settingsWrite";

/** 硬件 — Nanonis connection (editable host + 4 TCP ports + reconnect) and
 *  environment sensors (CRUD + rescan), mirroring the old dashboard hardware
 *  surface. Host/ports persist via POST /api/settings; reconnect via
 *  POST /api/nanonis/connect (graceful, 202 + poll GET /api/nanonis/connection).
 *  Sensors: POST /api/environment/sensors (add/update),
 *  DELETE /api/environment/sensors/{id}, GET /api/environment/sensors/rescan.
 *
 *  ITEM 13 — `readOnly`: when true (the 设置 → 硬件连接 surface) the port/host
 *  editors and sensor CRUD are presented read-only. Ports are edited ONLY in
 *  「高级管理 → 硬件」 (PIN-gated). In read-only mode the host + 4 ports are
 *  prefilled from GET /api/nanonis/connection (now 6501-6504) alongside live
 *  connection status, plus a note pointing to 高级管理. */

/** 设置键名不是 `string` 而是这四个字面量：body 现在是 `SettingsPatch`（从后端
 *  schema 生成），写错一个名字会在 typecheck 当场红，而不是在真机上变成一次
 *  422 —— 或者在 2026-08-10 之前那样，变成一句绿色的「已保存」。 */
type PortSetting =
  | "nanonis_port_main"
  | "nanonis_port_monitor"
  | "nanonis_port_data"
  | "nanonis_port_emergency";

const PORT_ROLES: { key: "main" | "monitor" | "data" | "emergency"; label: string; setting: PortSetting }[] = [
  { key: "main", label: "主端口", setting: "nanonis_port_main" },
  { key: "monitor", label: "监控端口", setting: "nanonis_port_monitor" },
  { key: "data", label: "数据端口", setting: "nanonis_port_data" },
  { key: "emergency", label: "急停端口", setting: "nanonis_port_emergency" },
];

/** Static read-only value box (visual parity with TextField, not editable). */
function ReadOnlyValue({ value, mono }: { value: string; mono?: boolean }) {
  return (
    <div
      className={`rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 text-sm text-mast-text ${
        mono ? "font-mono" : ""
      }`}
    >
      {value}
    </div>
  );
}

function NanonisCard({ readOnly = false }: { readOnly?: boolean }) {
  const queryClient = useQueryClient();
  const { toast, node } = useToast();

  const settingsQ = useQuery({
    queryKey: ["settings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings");
      if (error) throw error;
      return data;
    },
  });

  const connQ = useQuery({
    queryKey: ["nanonis", "connection"],
    refetchInterval: 3000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/nanonis/connection");
      if (error) throw error;
      return data;
    },
  });

  const [host, setHost] = useState<string | null>(null);
  const [ports, setPorts] = useState<Record<string, string>>({});
  useEffect(() => {
    setHost(null);
    setPorts({});
  }, [settingsQ.data]);

  const sv = settingsQ.data as Record<string, unknown> | undefined;
  // Live connection port (6501-6504) for a given role, used as the prefilled
  // read-only value and as a fallback when settings hasn't persisted a port.
  const connPort = (key: string): string => {
    const p = (connQ.data?.ports ?? []).find((x) => x.role === key)?.port;
    return p != null ? String(p) : "";
  };
  const hostVal =
    host ??
    (sv?.nanonis_host != null
      ? String(sv.nanonis_host)
      : connQ.data?.host != null
        ? String(connQ.data.host)
        : "");
  const portVal = (setting: string, key: string): string => {
    if (setting in ports) return ports[setting] ?? "";
    if (sv?.[setting] != null) return String(sv[setting]);
    return connPort(key); // prefill from the live connection (6501-6504)
  };

  const save = useMutation({
    mutationFn: async () => {
      // SettingsPatch（由后端 schema 生成）而不是 Record<string, unknown>：
      // 后端 2026-08-10 起是 extra='forbid'，多送一个键会让整笔 422，所以
      // 「键名对不对」必须在 typecheck 就答完，不能等到运行时。
      const body: SettingsPatch = {};
      if (host != null && host.trim()) body.nanonis_host = host.trim();
      for (const r of PORT_ROLES) {
        const raw = (ports[r.setting] ?? "").trim();
        if (raw !== "" && !Number.isNaN(Number(raw))) body[r.setting] = Number(raw);
      }
      const { data, error } = await api.POST("/api/settings", { body });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      // The old check read only `degraded`, so every refusal (`ok:false,
      // degraded:false` — port out of range, PIN) produced the green 「已保存」
      // toast. That is the same lie this button was already telling for a
      // different reason: the five keys were missing from the write schema
      // entirely, so every save was a silent no-op.
      // The edits are cleared ONLY on success — dropping them after a refusal
      // would refill the form with the old values as if nothing had happened.
      const problem = settingsWriteProblem(res);
      if (problem) { toast(problem, "err"); return; }
      setHost(null); setPorts({});
      queryClient.invalidateQueries({ queryKey: ["settings"] });
      toast("已保存 Nanonis 连接配置。", "ok");
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const reconnect = useMutation({
    mutationFn: async () => {
      const body: Record<string, unknown> = {};
      if (hostVal) body.host = hostVal;
      for (const r of PORT_ROLES) {
        const raw = portVal(r.setting, r.key).trim();
        if (raw !== "" && !Number.isNaN(Number(raw))) body[`port_${r.key}`] = Number(raw);
      }
      const { data, error } = await api.POST("/api/nanonis/connect", { body: body as never });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["nanonis", "connection"] });
      res?.degraded ? toast("重连未触发（内核未接入）。", "err") : toast("已发起重连（后台优雅连接）。", "ok");
    },
    onError: (e) => toast(`重连失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const portStatus = (key: string): boolean | undefined =>
    (connQ.data?.ports ?? []).find((p) => p.role === key)?.connected;

  return (
    <Card>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-medium text-mast-text">Nanonis 连接</span>
        {connQ.data && !connQ.data.degraded &&
          (connQ.data.connected ? <Badge tone="AUTO">在线</Badge> : <Badge tone="WARN">离线</Badge>)}
        {connQ.data?.host && <span className="text-xs text-mast-muted">{connQ.data.host}</span>}
      </div>
      {settingsQ.isPending && !readOnly && <Spinner />}
      {settingsQ.error && !readOnly && <ErrorNote error={settingsQ.error} />}
      {connQ.data?.degraded && <DegradedNote what="Nanonis 连接" />}
      {/* In read-only (设置) mode render from whatever is available (connection
          ports are prefilled 6501-6504); editing requires the settings query. */}
      {(readOnly ? connQ.data || settingsQ.data : settingsQ.data) && (
        <div className="space-y-3">
          <Field label="主机地址">
            {readOnly ? (
              <ReadOnlyValue value={hostVal || "—"} mono />
            ) : (
              <TextField value={hostVal} mono onChange={setHost} placeholder="127.0.0.1" />
            )}
          </Field>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {PORT_ROLES.map((r) => {
              const st = portStatus(r.key);
              return (
                <Field key={r.key} label={r.label}>
                  <div className="flex items-center gap-2">
                    {readOnly ? (
                      <ReadOnlyValue value={portVal(r.setting, r.key) || "—"} mono />
                    ) : (
                      <TextField
                        value={portVal(r.setting, r.key)}
                        onChange={(v) => setPorts((p) => ({ ...p, [r.setting]: v }))}
                      />
                    )}
                    {st != null && (st ? <Badge tone="AUTO">已连</Badge> : <Badge tone="DANGEROUS">未连</Badge>)}
                  </div>
                </Field>
              );
            })}
          </div>
          {readOnly ? (
            <p className="rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 text-xs text-mast-muted">
              端口在「高级管理 → 硬件」中修改。
            </p>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="primary" disabled={save.isPending} onClick={() => save.mutate()}>
                {save.isPending ? "保存中…" : "保存配置"}
              </Button>
              <Button variant="default" disabled={reconnect.isPending} onClick={() => reconnect.mutate()}>
                {reconnect.isPending ? "重连中…" : "重新连接"}
              </Button>
            </div>
          )}
        </div>
      )}
      {node}
    </Card>
  );
}

type SensorRow = { id: string; name: string; type: string; port: string; unit: string };

function SensorsCard({ readOnly = false }: { readOnly?: boolean }) {
  const queryClient = useQueryClient();
  const { toast, node } = useToast();

  const q = useQuery({
    queryKey: ["environment", "sensors"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/environment/sensors");
      if (error) throw error;
      return data;
    },
  });

  const [editing, setEditing] = useState<SensorRow | null>(null);

  const upsert = useMutation({
    mutationFn: async (row: SensorRow) => {
      const { data, error } = await api.POST("/api/environment/sensors", {
        body: {
          id: row.id,
          name: row.name || null,
          type: row.type || null,
          port: row.port || null,
          unit: row.unit || null,
          extra: {},
        } as never,
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      setEditing(null);
      queryClient.invalidateQueries({ queryKey: ["environment", "sensors"] });
      res?.degraded ? toast("写入未生效（内核未接入）。", "err") : toast("已保存传感器。", "ok");
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const del = useMutation({
    mutationFn: async (id: string) => {
      const { data, error } = await api.DELETE("/api/environment/sensors/{sensor_id}", {
        params: { path: { sensor_id: id } },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["environment", "sensors"] });
      res?.degraded ? toast("删除未生效（内核未接入）。", "err") : toast("已删除传感器。", "ok");
    },
    onError: (e) => toast(`删除失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const rescan = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.GET("/api/environment/sensors/rescan");
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["environment", "sensors"] });
      res?.degraded ? toast("重新探测未执行（内核未接入）。", "err") : toast(`重新探测完成：${res.count ?? 0} 个传感器。`, "ok");
    },
    onError: (e) => toast(`重新探测失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const sensors = q.data?.sensors ?? [];

  return (
    <Card>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-mast-text">环境传感器</span>
          {q.data && !q.data.degraded && (
            <Badge tone={q.data.autodetect ? "AUTO" : "INFO"}>{q.data.autodetect ? "自动探测" : "手动"}</Badge>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {/* Scan is offered on BOTH surfaces: it is read-only until the user
              confirms in the dialog, and the 设置 page is where people look for
              "接上我的温度计". Sensor row CRUD stays admin-only. */}
          <DeviceScanButton />
          {!readOnly && (
            <Button variant="default" disabled={rescan.isPending} onClick={() => rescan.mutate()}>
              {rescan.isPending ? "探测中…" : "重新探测"}
            </Button>
          )}
        </div>
      </div>
      {q.isPending && <Spinner />}
      {q.error && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="传感器配置" />}
      {q.data && !q.data.degraded && (
        <div className="space-y-3">
          {sensors.length === 0 ? (
            <EmptyNote label="暂无已配置传感器" />
          ) : (
            <div className="overflow-auto rounded-md border border-mast-border">
              <table className="w-full text-sm">
                <thead className="text-mast-muted">
                  <tr className="border-b border-mast-border">
                    <th className="px-2 py-1 text-left">ID</th>
                    <th className="px-2 py-1 text-left">名称</th>
                    <th className="px-2 py-1 text-left">类型</th>
                    <th className="px-2 py-1 text-left">端口</th>
                    <th className="px-2 py-1 text-left">单位</th>
                    {!readOnly && <th className="px-2 py-1" />}
                  </tr>
                </thead>
                <tbody>
                  {sensors.map((s) => (
                    <tr key={s.id} className="border-b border-mast-border/50">
                      <td className="px-2 py-1 font-mono text-xs">{s.id}</td>
                      <td className="px-2 py-1">{s.name ?? "—"}</td>
                      <td className="px-2 py-1">{s.type ?? "—"}</td>
                      <td className="px-2 py-1">{s.port ?? "—"}</td>
                      <td className="px-2 py-1">{s.unit ?? "—"}</td>
                      {!readOnly && (
                        <td className="px-2 py-1 text-right">
                          <div className="flex justify-end gap-1">
                            <Button variant="ghost" onClick={() => setEditing({
                              id: s.id, name: s.name ?? "", type: s.type ?? "", port: s.port ?? "", unit: s.unit ?? "",
                            })}>编辑</Button>
                            <Button variant="danger" onClick={() => del.mutate(s.id)}>删除</Button>
                          </div>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {readOnly ? null : editing ? (
            <div className="rounded-md border border-mast-accent/40 p-3">
              <div className="mb-2 text-sm font-medium text-mast-text">编辑 / 新增传感器</div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <Field label="ID"><TextField value={editing.id} mono onChange={(v) => setEditing({ ...editing, id: v })} /></Field>
                <Field label="名称"><TextField value={editing.name} onChange={(v) => setEditing({ ...editing, name: v })} /></Field>
                <Field label="类型"><TextField value={editing.type} onChange={(v) => setEditing({ ...editing, type: v })} /></Field>
                <Field label="端口"><TextField value={editing.port} onChange={(v) => setEditing({ ...editing, port: v })} /></Field>
                <Field label="单位"><TextField value={editing.unit} onChange={(v) => setEditing({ ...editing, unit: v })} /></Field>
              </div>
              <div className="mt-2 flex gap-2">
                <Button variant="primary" disabled={upsert.isPending || !editing.id.trim()} onClick={() => upsert.mutate(editing)}>
                  {upsert.isPending ? "保存中…" : "保存"}
                </Button>
                <Button variant="ghost" onClick={() => setEditing(null)}>取消</Button>
              </div>
            </div>
          ) : (
            <Button variant="default" onClick={() => setEditing({ id: "", name: "", type: "", port: "", unit: "" })}>
              + 新增传感器
            </Button>
          )}
        </div>
      )}
      {node}
    </Card>
  );
}

/** `readOnly` (设置 → 硬件连接) renders host + 4 ports prefilled from the live
 *  connection (6501-6504) + status as read-only, with a note pointing edits to
 *  「高级管理 → 硬件」; sensors become view-only. Admin passes the default
 *  (editable). */
export function HardwareManager({ readOnly = false }: { readOnly?: boolean }) {
  return (
    <div className="grid grid-cols-1 gap-4">
      <NanonisCard readOnly={readOnly} />
      <SensorsCard readOnly={readOnly} />
      {/* Read-only instrument readout (inputs + heater state). Shown on both
          surfaces — it is pure reference, and「加热器现在是不是开着」is exactly
          the thing someone walks over to the settings page to check. */}
      <InstrumentStateCard />
    </div>
  );
}
