import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { ErrorNote, Spinner } from "../ui";
import type { SettingsPatch } from "@/lib/settingsWrite";

// ── Z 参数组 ────────────────────────────────────────────────────────────────
//
// AI 设置 Z 反馈参数的方式是说出一个**组名**（ApplyZCtrlPreset），具体数值由代码
// 从这里列出的来源取出并写入硬件。这一节回答两个问题：现在有哪些组，以及每一组
// 的数会从哪里取。
//
// 按组名解析反馈参数，减少量纲与指数转写错误；界面展示实际来源和值。
//
// 刻意是只读 + 删除，没有编辑框：
//   · approach 组的值在上面「仪器档案 → 进针参数组」里改；
//   · scan 与各档名的值在下面「扫描档位表」里改；
//   · 自定义组由 AI 经人工审批卡新建。
// 每个数只有一个可编辑的地方，这一节是它们汇总后的样子。在这里再放一套输入框就
// 会出现「改了这里没生效」或者更糟——两处不一致而界面上看不出来。

type Preset = {
  name: string;
  kind: "reserved" | "custom" | "scan-tier";
  usable: boolean;
  why?: string | null;
  p_gain?: number | null;
  i_gain?: number | null;
  time_constant_s?: number | null;
  setpoint_a?: number | null;
  sources?: Record<string, string> | null;
};

type CustomPreset = {
  name: string;
  p_gain: string;
  i_gain: string;
  setpoint_a?: string | null;
  note?: string | null;
};

type PresetsPayload = {
  presets: Preset[];
  custom: CustomPreset[];
  reserved: string[];
  max_presets: number;
};

/** 3e-12 → "3p"。与后端 core/si_quantity.format_si 同一套写法（面板语言）。 */
function si(value: number | null | undefined, unit: string): string {
  if (value === null || value === undefined) return "—";
  const v = Number(value);
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return `0${unit}`;
  const steps: [string, number][] = [
    ["G", 1e9], ["M", 1e6], ["k", 1e3],
    ["m", 1e-3], ["u", 1e-6], ["n", 1e-9], ["p", 1e-12], ["f", 1e-15],
  ];
  for (const [prefix, factor] of steps) {
    const scaled = v / factor;
    if (Math.abs(scaled) >= 1 && Math.abs(scaled) < 1000) {
      return `${Number(scaled.toPrecision(6))}${prefix}${unit}`;
    }
  }
  return `${v.toPrecision(4)}${unit}`;
}

const KIND_LABEL: Record<Preset["kind"], string> = {
  reserved: "内置",
  "scan-tier": "扫描档位",
  custom: "自定义",
};

function usePresets() {
  return useQuery({
    queryKey: ["settings", "zctrl-presets"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings/zctrl-presets");
      if (error) throw error;
      return data as unknown as PresetsPayload;
    },
  });
}

export function ZCtrlPresetsSection({
  saving,
  save,
}: {
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const { data, isLoading, error } = usePresets();

  const removeCustom = (name: string) => {
    const rest = (data?.custom ?? []).filter((p) => p.name !== name);
    save({ zctrl_presets: rest });
  };

  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold">Z 参数组</h3>
        <p className="text-xs text-mast-muted">
          AI 设置 Z 反馈参数时只说组名，<b>具体数值由代码取出并写入硬件，不经过模型</b>，
          写完自动回读比对，不一致即报失败。这里是它能用的全部组，以及每组的数从哪里取。
        </p>
      </div>

      {isLoading && <Spinner />}
      {error != null && <ErrorNote error={error} />}

      {data != null && (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-mast-muted">
                <tr className="text-left">
                  <th className="py-1 pr-3 font-medium">组名</th>
                  <th className="py-1 pr-3 font-medium">来源</th>
                  <th className="py-1 pr-3 font-medium">P (m)</th>
                  <th className="py-1 pr-3 font-medium">I (m/s)</th>
                  <th className="py-1 pr-3 font-medium">T = P/I</th>
                  <th className="py-1 pr-3 font-medium">设定点</th>
                  <th className="py-1 font-medium" />
                </tr>
              </thead>
              <tbody>
                {data.presets.map((p) => (
                  <tr key={p.name} className="border-t border-mast-border/40 align-top">
                    <td className="py-1.5 pr-3">
                      <code className="font-medium">{p.name}</code>
                    </td>
                    <td className="py-1.5 pr-3 text-mast-muted">{KIND_LABEL[p.kind]}</td>
                    {p.usable ? (
                      <>
                        <td className="py-1.5 pr-3 tabular-nums">{si(p.p_gain, "")}</td>
                        <td className="py-1.5 pr-3 tabular-nums">{si(p.i_gain, "")}</td>
                        <td className="py-1.5 pr-3 tabular-nums">
                          {si(p.time_constant_s, "s")}
                        </td>
                        <td className="py-1.5 pr-3 tabular-nums">
                          {p.setpoint_a == null ? "不改" : si(p.setpoint_a, "A")}
                        </td>
                      </>
                    ) : (
                      <td className="py-1.5 pr-3 text-mast-muted" colSpan={4}>
                        {p.name === "scan"
                          ? "应用时按当前图幅自动选档"
                          : `未配置 — ${p.why ?? ""}`}
                      </td>
                    )}
                    <td className="py-1.5 text-right">
                      {p.kind === "custom" && (
                        <button
                          type="button"
                          disabled={saving}
                          onClick={() => removeCustom(p.name)}
                          className="text-mast-muted hover:text-red-400 disabled:opacity-50"
                        >
                          删除
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-[11px] text-mast-muted">
            改哪里：<code>approach</code> 在上面「仪器档案 → 进针参数组」；
            <code>scan</code> 与各扫描档名在下面「扫描档位表」；
            自定义组由 AI 新建（每次都要你在人工审批卡上过目，卡片上显示的就是将要存进去的值）。
            最多 {data.max_presets} 个自定义组。
          </p>
        </>
      )}
    </section>
  );
}
