import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, Checkbox, Modal, TextField, useToast } from "@/components/controls";

/** 扫描设备接口 — scan → confirm → adopt.
 *
 *  The old「重新探测」button probed the serial ports and immediately swapped
 *  whatever it found into the live monitor: no identification shown, no way to
 *  say "that's not right". This flow splits the two halves that were conflated:
 *
 *    GET  /api/environment/discover        identify only, persists NOTHING
 *    POST /api/environment/sensors/adopt   write the entries the user ticked
 *
 *  Adoption is what makes a port stick — the entry lands in
 *  environment_sensors.json, so every later boot rebuilds it before autodetect
 *  runs and the instrument is found without another scan.
 *
 *  InstrumentStateCard is the read-only readout of everything MAST can ask the
 *  instrument (inputs, sensor types, and the heater state). Every field behind
 *  it comes from a query command; there is no path from this panel to a setter. */

type Suggested = Record<string, unknown>;

const fmt = (v: number | null | undefined, digits = 3, unit = ""): string =>
  v == null ? "—" : `${v.toFixed(digits)}${unit ? " " + unit : ""}`;

/** One input's reading status — THREE states, never two.
 *
 *  `faults` is tri-state on the wire: `[]` = the instrument answered `RDGST?`
 *  and no fault bit is set, non-empty = it answered and the reading is NOT
 *  trustworthy, `null` = the status query never answered, so nothing at all is
 *  known about this reading's validity.
 *
 *  Both tables used to write `(c.faults ?? []).length > 0`, which reads `null`
 *  as clean and labels an unbacked reading「正常」— the very fold the backend
 *  just stopped doing, re-done one layer later. Prefer the backend's own
 *  `health` verdict so the three-way rule keeps ONE definition; the local
 *  derivation is only for a payload that predates the field, and it defaults to
 *  unknown rather than to clean. */
function ReadingStatus({ faults, health }: {
  faults?: string[] | null;
  health?: string | null;
}) {
  const verdict = health ?? (faults == null ? "unknown" : faults.length ? "faulty" : "clean");
  if (verdict === "faulty") {
    return <Badge tone="DANGEROUS">{(faults ?? []).join("、") || "读数不可信"}</Badge>;
  }
  if (verdict === "clean") {
    return <span className="text-mast-muted">正常</span>;
  }
  return (
    <Badge tone="WARN">
      <span title="RDGST? 状态查询没有回复 —— 这个温度没有仪器的故障位背书，不等于正常">
        状态位未读到
      </span>
    </Badge>
  );
}

/** One discovered instrument, with a tick + editable name per input. */
function DeviceBlock({
  device, picks, onToggle, onRename,
}: {
  device: Record<string, never> extends never ? any : any;
  picks: Record<string, { on: boolean; name: string }>;
  onToggle: (id: string, on: boolean) => void;
  onRename: (id: string, name: string) => void;
}) {
  if (!device.identified) {
    return (
      <div className="rounded-md border border-mast-border bg-mast-bg/40 px-3 py-2 text-xs text-mast-muted">
        <span className="font-mono">{device.port}</span>
        {device.description ? ` — ${device.description}` : ""} — 未识别到已知仪器
      </div>
    );
  }
  const suggested: Suggested[] = device.suggested_sensors ?? [];
  return (
    <div className="rounded-md border border-mast-accent/40 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-mast-text">
          {device.model || "未知型号"}
        </span>
        <span className="font-mono text-xs text-mast-muted">{device.port}</span>
        {device.firmware && <Badge tone="INFO">固件 {device.firmware}</Badge>}
        {device.already_configured && <Badge tone="AUTO">已接入</Badge>}
        {device.any_heater_on ? (
          <Badge tone="DANGEROUS">加热器开启</Badge>
        ) : (
          <Badge tone="AUTO">加热器关闭</Badge>
        )}
      </div>
      {device.idn && (
        <div className="mb-2 break-all font-mono text-[11px] text-mast-muted">{device.idn}</div>
      )}

      {(device.channels ?? []).length > 0 && (
        <div className="mb-3 overflow-auto rounded-md border border-mast-border">
          <table className="w-full text-sm">
            <thead className="text-mast-muted">
              <tr className="border-b border-mast-border">
                <th className="px-2 py-1 text-left">输入</th>
                <th className="px-2 py-1 text-left">仪器上的名字</th>
                <th className="px-2 py-1 text-left">当前读数</th>
                <th className="px-2 py-1 text-left">传感器类型</th>
                <th className="px-2 py-1 text-left">状态</th>
              </tr>
            </thead>
            <tbody>
              {(device.channels ?? []).map((c: any) => (
                <tr key={c.channel} className="border-b border-mast-border/50">
                  <td className="px-2 py-1 font-mono">{c.channel}</td>
                  <td className="px-2 py-1">{c.label || "—"}</td>
                  <td className="px-2 py-1 font-mono">{fmt(c.kelvin, 3, "K")}</td>
                  <td className="px-2 py-1">{c.sensor_type || "—"}</td>
                  <td className="px-2 py-1">
                    <ReadingStatus faults={c.faults} health={c.health} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="mb-1 text-xs text-mast-muted">
        勾选要接入 MAST 的通道（名字取自仪器，可改）：
      </div>
      <div className="space-y-2">
        {suggested.map((s) => {
          const id = String(s.id);
          const pick = picks[id] ?? { on: true, name: String(s.name ?? id) };
          return (
            <div key={id} className="flex flex-wrap items-center gap-2">
              <Checkbox
                checked={pick.on}
                onChange={(v) => onToggle(id, v)}
                label={<span className="font-mono text-xs">{String(s.channel ?? "")} →</span>}
              />
              <div className="min-w-[12rem] flex-1">
                <TextField value={pick.name} onChange={(v) => onRename(id, v)} />
              </div>
            </div>
          );
        })}
        {suggested.length === 0 && <EmptyNote label="该设备没有可接入的通道" />}
      </div>
    </div>
  );
}

export function DeviceScanButton() {
  const queryClient = useQueryClient();
  const { toast, node } = useToast();
  const [open, setOpen] = useState(false);
  const [devices, setDevices] = useState<any[]>([]);
  const [picks, setPicks] = useState<Record<string, { on: boolean; name: string }>>({});

  const scan = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.GET("/api/environment/discover");
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      if (res?.degraded) {
        toast("扫描未执行（内核未接入）。", "err");
        return;
      }
      const found = (res?.devices ?? []) as any[];
      setDevices(found);
      // Pre-tick everything identified, pre-filled with the instrument's own
      // input labels — the user confirms rather than retypes.
      const next: Record<string, { on: boolean; name: string }> = {};
      for (const d of found) {
        for (const s of (d.suggested_sensors ?? []) as Suggested[]) {
          next[String(s.id)] = { on: true, name: String(s.name ?? s.id) };
        }
      }
      setPicks(next);
      setOpen(true);
    },
    onError: (e) => toast(`扫描失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const adopt = useMutation({
    mutationFn: async () => {
      const chosen: Suggested[] = [];
      for (const d of devices) {
        for (const s of (d.suggested_sensors ?? []) as Suggested[]) {
          const pick = picks[String(s.id)];
          if (pick?.on) chosen.push({ ...s, name: pick.name.trim() || String(s.name ?? s.id) });
        }
      }
      const { data, error } = await api.POST("/api/environment/sensors/adopt", {
        body: { sensors: chosen } as never,
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["environment"] });
      if (!res?.ok) {
        toast(`未接入任何设备${res?.rejected?.length ? "：" + res.rejected.join("；") : "。"}`, "err");
        return;
      }
      setOpen(false);
      toast(`已接入 ${res.adopted} 个通道，开机后自动连接。`, "ok");
    },
    onError: (e) => toast(`接入失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const chosenCount = Object.values(picks).filter((p) => p.on).length;
  const identified = devices.filter((d) => d.identified).length;

  return (
    <>
      <Button variant="default" disabled={scan.isPending} onClick={() => scan.mutate()}>
        {scan.isPending ? "扫描中…" : "扫描设备接口"}
      </Button>

      <Modal open={open} onClose={() => setOpen(false)} title="扫描结果 — 请确认识别是否正确" wide>
        <div className="space-y-3">
          {devices.length === 0 ? (
            <EmptyNote label="没有找到任何串口" />
          ) : identified === 0 ? (
            <>
              <EmptyNote label="扫描到串口，但没有识别出已知仪器" />
              {devices.map((d) => (
                <DeviceBlock key={d.port} device={d} picks={picks} onToggle={() => {}} onRename={() => {}} />
              ))}
            </>
          ) : (
            devices.map((d) => (
              <DeviceBlock
                key={d.port}
                device={d}
                picks={picks}
                onToggle={(id, on) =>
                  setPicks((p) => ({ ...p, [id]: { ...(p[id] ?? { name: id }), on } }))
                }
                onRename={(id, name) =>
                  setPicks((p) => ({ ...p, [id]: { ...(p[id] ?? { on: true }), name } }))
                }
              />
            ))
          )}

          <p className="rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 text-xs text-mast-muted">
            扫描只是读取，不会改变任何配置。点「接入 MAST」后才写入本地设置，之后每次开机会自动连接这个端口。
            MAST 对温控仪只发查询命令，不会设定温度或开启加热器。
          </p>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              disabled={adopt.isPending || chosenCount === 0}
              onClick={() => adopt.mutate()}
            >
              {adopt.isPending ? "接入中…" : `接入 MAST（${chosenCount} 个通道）`}
            </Button>
            <Button variant="ghost" onClick={() => setOpen(false)}>取消</Button>
          </div>
        </div>
      </Modal>
      {node}
    </>
  );
}

/** 仪器设定 — the full read-only readout, for whoever wants to know. */
export function InstrumentStateCard() {
  const q = useQuery({
    queryKey: ["environment", "instrument-state"],
    // Goes over the serial bus (sharing the monitor's handle), so poll gently.
    refetchInterval: 15000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/environment/instrument-state");
      if (error) throw error;
      return data;
    },
  });

  const instruments = q.data?.instruments ?? [];

  return (
    <Card>
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-medium text-mast-text">仪器设定（只读）</span>
        {instruments.some((i) => i.any_heater_on) ? (
          <Badge tone="DANGEROUS">加热器开启</Badge>
        ) : instruments.length > 0 ? (
          <Badge tone="AUTO">加热器关闭</Badge>
        ) : null}
      </div>
      {q.isPending && <Spinner />}
      {q.error && <ErrorNote error={q.error} />}
      {q.data?.degraded && <DegradedNote what="仪器设定" />}
      {q.data && !q.data.degraded && instruments.length === 0 && (
        <EmptyNote label="未连接任何温控仪 —— 可用「扫描设备接口」查找" />
      )}
      <div className="space-y-4">
        {instruments.map((inst) => (
          <div key={inst.port || inst.idn} className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-mast-text">{inst.model || "仪器"}</span>
              <span className="font-mono text-xs text-mast-muted">{inst.port}</span>
              {inst.firmware && <Badge tone="INFO">固件 {inst.firmware}</Badge>}
              {inst.serial_number && (
                <span className="font-mono text-[11px] text-mast-muted">S/N {inst.serial_number}</span>
              )}
            </div>

            <div className="overflow-auto rounded-md border border-mast-border">
              <table className="w-full text-sm">
                <thead className="text-mast-muted">
                  <tr className="border-b border-mast-border">
                    <th className="px-2 py-1 text-left">输入</th>
                    <th className="px-2 py-1 text-left">名称</th>
                    <th className="px-2 py-1 text-left">温度 (K)</th>
                    <th className="px-2 py-1 text-left">温度 (°C)</th>
                    <th className="px-2 py-1 text-left">原始量</th>
                    <th className="px-2 py-1 text-left">传感器</th>
                    <th className="px-2 py-1 text-left">状态</th>
                  </tr>
                </thead>
                <tbody>
                  {(inst.channels ?? []).map((c) => (
                    <tr key={c.channel} className="border-b border-mast-border/50">
                      <td className="px-2 py-1 font-mono">{c.channel}</td>
                      <td className="px-2 py-1">{c.label || "—"}</td>
                      <td className="px-2 py-1 font-mono">{fmt(c.kelvin)}</td>
                      <td className="px-2 py-1 font-mono">{fmt(c.celsius, 2)}</td>
                      <td className="px-2 py-1 font-mono">{fmt(c.sensor_value, 4)}</td>
                      <td className="px-2 py-1">{c.sensor_type || "—"}</td>
                      <td className="px-2 py-1">
                        <ReadingStatus faults={c.faults} health={c.health} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {(inst.outputs ?? []).length > 0 && (
              <div className="overflow-auto rounded-md border border-mast-border">
                <table className="w-full text-sm">
                  <thead className="text-mast-muted">
                    <tr className="border-b border-mast-border">
                      <th className="px-2 py-1 text-left">控制输出</th>
                      <th className="px-2 py-1 text-left">加热档位</th>
                      <th className="px-2 py-1 text-left">输出</th>
                      <th className="px-2 py-1 text-left">设定值</th>
                      <th className="px-2 py-1 text-left">控制模式</th>
                      <th className="px-2 py-1 text-left">控制输入</th>
                      <th className="px-2 py-1 text-left">升温速率</th>
                      <th className="px-2 py-1 text-left">断电自恢复</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(inst.outputs ?? []).map((o) => (
                      <tr key={o.output} className="border-b border-mast-border/50">
                        <td className="px-2 py-1 font-mono">{o.output}</td>
                        <td className="px-2 py-1">
                          {o.heater_on ? (
                            <Badge tone="DANGEROUS">{o.range_label || o.range_code}</Badge>
                          ) : (
                            <span className="text-mast-muted">{o.range_label || "关闭"}</span>
                          )}
                        </td>
                        <td className="px-2 py-1 font-mono">{fmt(o.heater_pct, 1, "%")}</td>
                        <td className="px-2 py-1 font-mono">{fmt(o.setpoint, 2, "K")}</td>
                        <td className="px-2 py-1">{o.mode || "—"}</td>
                        <td className="px-2 py-1">{o.control_input || "—"}</td>
                        <td className="px-2 py-1 font-mono">
                          {o.ramping ? fmt(o.ramp_rate, 1, "K/min") : "关闭"}
                        </td>
                        <td className="px-2 py-1">{o.powerup_enabled ? "是" : "否"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
      </div>
      {instruments.length > 0 && (
        <p className="mt-2 rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5 text-xs text-mast-muted">
          以上全部为只读查询。MAST 不会设定温度、不会改变加热档位 —— 温控由仪器本身负责。
        </p>
      )}
    </Card>
  );
}
