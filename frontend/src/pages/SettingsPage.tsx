import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useUiStore } from "../store";
import { INSTRUMENT_INIT_KEY, useInstrumentInit } from "@/hooks/useInstrumentInit";
import { SignalIndexField } from "@/components/signals/SignalIndexField";
import { SIGNAL_AUTO_VALUE, SIGNAL_INDEX_KEYS } from "@/lib/signalChannels";
import {
  Badge,
  DegradedNote,
  EmptyNote,
  ErrorNote,
  Section,
  Spinner,
} from "../components/ui";
import {
  Accordion,
  Checkbox,
  Field,
  RadioGroup,
  SelectField,
  Toggle,
  useToast,
} from "../components/controls";
import { ModelCapabilityTable } from "../components/settings/ModelCapabilityTable";
import { AgentOverridesEditor } from "../components/settings/AgentOverridesEditor";
// 硬件连接 in 设置 is READ-ONLY: a small port table (settings-owned) shows the
// resolved host + 4 Nanonis TCP ports. Editing/reconnect/sensors live behind
// 「高级管理 → 硬件」 (admin-owned, PIN-gated). We do NOT reuse admin/HardwareManager
// here so the settings surface never edits hardware.
import { NanonisPortsReadOnly } from "../components/settings/NanonisPortsReadOnly";
import { DeviceScanButton, InstrumentStateCard } from "@/components/admin/DeviceScanner";
import { RemoteAccessSection } from "../components/settings/RemoteAccessSection";
import { MonitoringSettingsSection } from "../components/settings/MonitoringSettingsSection";
import { EnvHistorySettingsSection } from "../components/settings/EnvHistorySettingsSection";
import { ConductSettingsSection } from "@/components/settings/ConductSettingsSection";
import { CONDUCT_SETTINGS_TITLE } from "@/lib/conduct";
import { ScanPolicySection } from "../components/settings/ScanPolicySection";
import { ZCtrlPresetsSection } from "../components/settings/ZCtrlPresetsSection";
import { settingsWriteProblem, type SettingsPatch } from "@/lib/settingsWrite";

// ── FULL-PARITY 设置 page ────────────────────────────────────────────────────
// Rebuilds the old Gradio 设置 tab surface-for-surface over the typed API seam.
// The old tab was a stack of collapsible gr.Accordion sections (NOT sub-tabs):
//   模型 Model       — model_info + 全局对话模型 + Thinking(只读显示实际锁定值) + 查询助手模型
//                       + 各 Agent 模型 / Thinking（与 Agents 页同源）(嵌套, 折叠)
//   外观 Appearance  — 字体大小 Font(radio) + 主题 Theme(radio) + 技能 Codex 实时搜索
//   （知识 Knowledge   — 2026-08-24 删除：那个单选框写进的 config.knowledge.default_mode
//                       全仓零读者，v2 的知识走拉取式 query_knowledge。
//                       一个存得进去、界面说「已保存」、而注入内容一个
//                       字节都不变的开关，比没有这个开关更误导人。）
//   语音 Voice       — TTS 音色 Voice + 自动朗读助手回复(折叠)
//   其它 Other       — 只读说明(折叠)
// Hardware (Nanonis / sensors) lives behind 「高级管理」 (PIN-gated) here.
// Every editable surface persists via POST /api/settings (or model-override).

// ── shared queries ────────────────────────────────────────────────────────────
function useModels() {
  return useQuery({
    queryKey: ["config", "models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/config/models");
      if (error) throw error;
      return data;
    },
  });
}

function useSettings() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings");
      if (error) throw error;
      return data;
    },
  });
}

function useHardwareModules() {
  return useQuery({
    queryKey: ["settings", "hardware-modules"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings/hardware-modules");
      if (error) throw error;
      return data;
    },
  });
}

function useVoices() {
  return useQuery({
    queryKey: ["voice", "voices"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/voice/voices");
      if (error) throw error;
      return data;
    },
  });
}

// Old 查询助手 model choices (qa_model_selector). Independent of the global model.
const QA_MODEL_CHOICES: { value: string; label: string }[] = [
  { value: "kimi-k2.6", label: "kimi-k2.6  (Moonshot)" },
  { value: "kimi-k2.7-code", label: "kimi-k2.7-code  (Moonshot)" },
  { value: "deepseek-v4-pro", label: "deepseek-v4-pro  (DeepSeek)" },
  { value: "qwen3.7-max", label: "qwen3.7-max  (Qwen / DashScope)" },
  { value: "qwen3.7-max-preview", label: "qwen3.7-max-preview  (Qwen 快照)" },
  { value: "minimax-m3", label: "minimax-m3  (MiniMax)" },
  { value: "glm-5.2", label: "glm-5.2  (Zhipu)" },
];

// Old gr.Radio choices — verbatim labels (no English suffix in the old build).
const FONT_CHOICES = [
  { value: "小", label: "小" },
  { value: "中", label: "中" },
  { value: "大", label: "大" },
];
const THEME_CHOICES = [
  { value: "Light", label: "Light" },
  { value: "Dark", label: "Dark" },
];

// ── 外观 live-apply helpers (ITEM 12) ──────────────────────────────────────────
// font_scale persists as 小/中/大; map to a root font-size so every rem-based
// Tailwind size scales. We set documentElement.style.fontSize + a data attr the
// CSS could read — applied live (no reload) and re-applied on settings load.
const FONT_PX: Record<string, string> = { 小: "14px", 中: "16px", 大: "18px" };
function applyFontScale(scale: string) {
  const root = document.documentElement;
  root.style.fontSize = FONT_PX[scale] ?? "16px";
  root.setAttribute("data-font-scale", scale);
}

// settings theme is persisted as Light/Dark; the zustand store (top-bar toggle)
// uses lowercase dark/light. Map both ways so the 外观 control both reflects and
// drives useUiStore.theme (which AppLayout's effect live-applies to <html>).
type StoreTheme = "dark" | "light";
const settingsThemeToStore = (t: string): StoreTheme => (t === "Dark" ? "dark" : "light");
const storeThemeToSettings = (t: StoreTheme): string => (t === "dark" ? "Dark" : "Light");

// ── Thinking display (READ-ONLY, single source of truth) ──────────────────────
// Thinking is NOT user-selectable: the system LOCKS every model to its strongest
// supported value. This helper turns a model's thinking_mode (from
// GET /api/config/models) into the consistent read-only text/badge shown
// identically across 全局 / 查询助手 / 各 Agent — so the DISPLAY always MATCHES
// the ACTUAL locked value:
//   none    → 无思考           (no thinking parameter sent)
//   fixed   → 固定（服务端）    (reasoning model fixes it server-side)
//   tunable → max · 最强（锁定）(system locks tunable models to max)
export type ThinkingMode = "none" | "fixed" | "tunable";

export function thinkingDisplay(mode: ThinkingMode | string | null | undefined): {
  label: string;
  tone: string;
} {
  switch (mode) {
    case "none":
      return { label: "无思考", tone: "WARN" };
    case "fixed":
      return { label: "固定·始终开启（最强）", tone: "INFO" };
    case "tunable":
    default:
      return { label: "max · 最强（锁定）", tone: "AUTO" };
  }
}

export default function SettingsPage() {
  const models = useModels();
  const settings = useSettings();
  const voices = useVoices();
  const hwModules = useHardwareModules();
  const queryClient = useQueryClient();
  const { toast, node: toastNode } = useToast();

  // Unified write mutation → POST /api/settings (partial; None = leave unchanged).
  const saveSettings = useMutation({
    mutationFn: async (body: SettingsPatch) => {
      const { data, error } = await api.POST("/api/settings", { body });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      // 「保存失败」这四个字以前是这条路径上唯一的说明 —— 而后端在 `rejected`
      // 里给了逐键的中文原因（档位表区间重叠、参数组数值没带 SI 前缀…）。
      // 共用判据把那句话带出来。
      const problem = settingsWriteProblem(data);
      if (problem) {
        toast(problem, "err");
      } else {
        // A hardware-module toggle is persisted the moment this returns, but it is
        // NOT live until the agent's tool list is rebuilt — the list is frozen at
        // graph-build time. The backend tells us what will actually happen (rebuild
        // started / declined because a task is running / no live agent). Say that
        // verbatim rather than a bare "已保存", which would be a lie about liveness.
        const note = (data as { rebuild_note?: string })?.rebuild_note;
        toast(note ? `已保存${note}` : "已保存", "ok");
        queryClient.invalidateQueries({ queryKey: ["settings"] });
      }
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const save = (patch: SettingsPatch) => saveSettings.mutate(patch);

  return (
    <div>
      <Section title="设置">
        <p className="mb-4 text-sm text-mast-muted">
          <strong>设置</strong> — 统一配置：模型 / 外观 / 硬件连接 / 知识 / 语音。所有改动会自动持久化，重启后保留。
        </p>

        {/* 新仪器初始化的**手动入口**。自动弹出走顶部横幅（必填未齐 / 硬件指纹变了
            / 从没走过初始化时出现）；这里是「我想再过一遍」的那条路。 */}
        <SetupEntryRow toast={toast} />

        <Accordion title="模型 Model" defaultOpen>
          <ModelSection
            models={models}
            settings={settings}
            saving={saveSettings.isPending}
            save={save}
          />
          <div className="mt-4">
            <Accordion title="各 Agent 模型 / Thinking（与 Agents 页同源）" defaultOpen={false}>
              <AgentOverridesEditor toast={toast} />
            </Accordion>
          </div>
        </Accordion>

        <Accordion title="外观 Appearance" defaultOpen>
          <AppearanceSection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="硬件连接 Hardware" defaultOpen>
          {/* ITEM 5 — ports are READ-ONLY here; editing lives in 高级管理 → 硬件
              (PIN-gated). A settings-owned read-only table shows host + 4 ports
              (defaults 6501–6504 via GET /api/nanonis/connection) + live status. */}
          <NanonisPortsReadOnly />
        </Accordion>

        <Accordion title="温度计 / 真空计 Instruments" defaultOpen>
          {/* 扫描设备接口: scan the serial ports, show what was identified, and
              only connect it after the user confirms in the dialog. Adoption is
              persisted, so the port is found automatically on every later boot.
              Below it, the read-only instrument readout (inputs + HEATER state)
              — reference for anyone who wants to know what the controller is
              actually doing. MAST only ever queries it; it cannot set a
              temperature or switch a heater on. */}
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <DeviceScanButton />
              <span className="text-xs text-mast-muted">
                扫描串口上的温控仪 / 真空计，确认后接入，并记住到下次开机。
              </span>
            </div>
            <InstrumentStateCard />
          </div>
        </Accordion>

        <Accordion title="远程访问 Remote（Tailscale 跨网控制）" defaultOpen={false}>
          <RemoteAccessSection toast={toast} />
        </Accordion>

        <Accordion title="实验默认参数 Experiment defaults" defaultOpen>
          <ExperimentDefaultsSection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="Z 参数组 Z-control presets（AI 按名应用，数值由代码写）" defaultOpen={false}>
          <ZCtrlPresetsSection saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="扫描参数档位 Scan policy（按尺度：速度 / 像素 / 反馈）" defaultOpen={false}>
          <ScanPolicySection saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="仪器进退针 Instrument (退针方向 / dI/dV)" defaultOpen={false}>
          <InstrumentProfileSection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="语音 Voice" defaultOpen={false}>
          <VoiceSection voices={voices} settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="自治与成本 Autonomy & cost" defaultOpen={false}>
          <AutonomySection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title={CONDUCT_SETTINGS_TITLE} defaultOpen={false}>
          <ConductSettingsSection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="对话处理与归档 Conversation & archiving" defaultOpen={false}>
          <ConversationArchivingSection
            settings={settings}
            saving={saveSettings.isPending}
            save={save}
          />
        </Accordion>

        <Accordion title="视觉判别阈值 Vision" defaultOpen={false}>
          <VisionThresholdsSection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="经典针尖阈值 Classical（免模型 · 按仪器重标定）" defaultOpen={false}>
          <ClassicalThresholdsSection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="电流监控 Current Monitor（常开 / 保留策略 / 告警阈值）" defaultOpen={false}>
          <MonitoringSettingsSection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        <Accordion title="环境历史 Environment History（记录粒度 / 保留期 / 噪声谱）" defaultOpen={false}>
          <EnvHistorySettingsSection settings={settings} saving={saveSettings.isPending} save={save} />
        </Accordion>

        {/* 硬件模块的开关 2026-07-13 移到 高级 → 系统/全局 → 能力开关。
            理由:打开一个模块 = 把 DANGEROUS skill 交给 agent(激光/射频/会互撞的多探针),
            那和「改个字号」不是同一类操作。留一块指路牌——静悄悄消失比放在这里更糟。*/}
        <Accordion title="硬件模块 Hardware" defaultOpen={false}>
          <HardwareModulesMoved hwModules={hwModules} />
        </Accordion>

        <Accordion title="其它 Other" defaultOpen={false}>
          <OtherSection />
        </Accordion>

        <Accordion title="模型能力表 Capabilities" defaultOpen={false}>
          <CapabilitiesSection models={models} />
        </Accordion>
      </Section>
      {toastNode}
    </div>
  );
}

// ── 模型 ───────────────────────────────────────────────────────────────────────
/** 新仪器初始化的手动入口 + 完成状态。
 *
 *  「重新弹出」清掉的是**完成戳**，一个数值都不动 —— 它只让横幅回来。
 *  必填项的判据从来不看这个戳（那是内容判据，从值本身算），所以清戳既不会
 *  放松什么，也不会假装什么还没填。 */
function SetupEntryRow({ toast }: { toast: (t: string, tone?: "ok" | "err") => void }) {
  const qc = useQueryClient();
  const q = useInstrumentInit();
  const reopen = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/instrument-init/reopen", {});
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: INSTRUMENT_INIT_KEY });
      toast("已重新打开：初始化提醒横幅会回来。");
    },
    onError: () => toast("操作失败。", "err"),
  });

  const d = q.data;
  const req = d?.counts?.required ?? { total: 0, complete: 0 };
  const done = !d?.needs_setup;

  return (
    <div className="mb-4 flex flex-wrap items-center gap-3 rounded-mast-card border border-mast-border bg-mast-panel-2 px-4 py-3">
      <div className="min-w-0">
        <p className="text-sm font-semibold text-mast-text">新仪器初始化</p>
        <p className="mt-0.5 text-xs text-mast-muted">
          「装到一台新机器上，还差哪些数」的那份清单：前放倍数 / 偏压极性 /
          安全包络 / 进针参数组 / 进针撞不撞 / 有没有 XY 位移 / 修针破坏范围 …
          {d && (
            <span className={done ? "ml-1 text-mast-auto" : "ml-1 text-mast-danger"}>
              当前必填 {req.complete}/{req.total}。
            </span>
          )}
        </p>
      </div>
      <div className="ml-auto flex gap-2">
        <Link
          to="/settings/setup"
          className="inline-flex items-center rounded-mast-ctl bg-mast-accent px-3.5 py-2 text-sm font-semibold text-mast-accent-ink hover:opacity-90"
        >
          打开初始化页面
        </Link>
        {d?.completed_at ? (
          <button
            type="button"
            disabled={reopen.isPending}
            onClick={() => reopen.mutate()}
            className="inline-flex items-center rounded-mast-ctl border border-mast-border-strong bg-mast-panel px-3.5 py-2 text-sm hover:bg-mast-panel-2 disabled:opacity-50"
            title="清掉完成戳，让提醒横幅重新出现。不会改动任何数值。"
          >
            重新提醒
          </button>
        ) : null}
      </div>
    </div>
  );
}

function ModelSection({
  models,
  settings,
  saving,
  save,
}: {
  models: ReturnType<typeof useModels>;
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  if (models.isPending || settings.isPending) return <Spinner />;
  if (models.isError) return <ErrorNote error={models.error} />;
  if (settings.isError) return <ErrorNote error={settings.error} />;
  if (!models.data?.models?.length) return <EmptyNote label="无可用模型" />;

  const s = settings.data ?? {};
  const modelList = models.data.models;

  const currentAlias = s.model_alias ?? models.data.default_alias;
  const currentModel = modelList.find((m) => m.alias === currentAlias);
  // Thinking is LOCKED (not selectable): the system pins every model to its
  // strongest supported value. We only DISPLAY the actual locked value, derived
  // from the model's thinking_mode (see thinkingDisplay) — consistently across
  // 全局 / 查询助手 / 各 Agent.
  const thinkingMode = currentModel?.thinking_mode ?? "tunable";
  const thinkInfo = thinkingDisplay(thinkingMode);
  const currentQa = s.qa_model ?? "kimi-k2.6";
  // QA helper thinking mirrors the SELECTED qa_model's locked value the same way.
  const qaThinkingMode = modelList.find((m) => m.alias === currentQa)?.thinking_mode ?? "tunable";
  const qaThinkInfo = thinkingDisplay(qaThinkingMode);

  return (
    <div className="space-y-4">
      {/* model_info — parity with build_model_info_html: badge + desc + Thinking line */}
      {currentModel && (
        <div className="rounded-md border border-mast-border bg-mast-bg/40 p-3 text-sm">
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="rounded bg-mast-accent/20 px-2 py-0.5 font-semibold text-mast-accent">
              {currentModel.alias}
            </span>
            <span className="text-mast-muted">{currentModel.description}</span>
          </div>
          <div className="mt-1 flex items-center gap-1.5 text-mast-muted">
            <span>Thinking:</span>
            <Badge tone={thinkInfo.tone}>{thinkInfo.label}</Badge>
          </div>
        </div>
      )}

      <Field label="全局对话模型 Global model">
        <SelectField
          value={currentAlias}
          onChange={(v) => save({ model_alias: v })}
          options={modelList.map((m) => ({ value: m.alias, label: `${m.alias}  (${m.provider})` }))}
        />
      </Field>

      {/* Thinking — READ-ONLY display of the actual locked value (not selectable).
          Mirrors the model_info card and the per-agent rows for full consistency. */}
      <Field label="Thinking" hint="思考强度由系统锁定为最强，不可手动选择">
        <span className="inline-flex items-center gap-1.5 rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5">
          <Badge tone={thinkInfo.tone}>{thinkInfo.label}</Badge>
        </span>
      </Field>

      <Field label="查询助手模型 QA helper model">
        <SelectField value={currentQa} onChange={(v) => save({ qa_model: v })} options={QA_MODEL_CHOICES} />
      </Field>

      {/* 查询助手 Thinking — READ-ONLY display of the qa_model's locked value,
          identical style to the global Thinking display above. */}
      <Field label="查询助手 Thinking" hint="思考强度由系统锁定，仅显示实际生效值">
        <span className="inline-flex items-center gap-1.5 rounded-md border border-mast-border bg-mast-bg/40 px-2 py-1.5">
          <Badge tone={qaThinkInfo.tone}>{qaThinkInfo.label}</Badge>
        </span>
      </Field>

      {saving && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}

// ── 外观 ───────────────────────────────────────────────────────────────────────
function AppearanceSection({
  settings,
  saving,
  save,
}: {
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  // 主题: reflect + drive the SAME zustand store the top-bar toggle uses, so the
  // 外观 control live-applies (AppLayout's effect mirrors store.theme → <html>)
  // and stays in sync with the header toggle. We persist the Light/Dark string too.
  const storeTheme = useUiStore((st) => st.theme);
  const setStoreTheme = useUiStore((st) => st.setTheme);

  const s = settings.data ?? {};
  const font = s.font_scale && ["小", "中", "大"].includes(s.font_scale) ? s.font_scale : "中";
  // The radio reflects the live store theme (not just the persisted value), so a
  // top-bar toggle is mirrored here immediately.
  const theme = storeThemeToSettings(storeTheme);
  const codexLive = s.codex_live_search ?? true;

  // On settings load: apply the persisted font live, and reconcile the persisted
  // theme into the store once (store wins thereafter — it's the live source).
  useEffect(() => {
    if (settings.data) applyFontScale(font);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.data, font]);

  if (settings.isPending) return <Spinner />;
  if (settings.isError) return <ErrorNote error={settings.error} />;

  const onTheme = (v: string) => {
    setStoreTheme(settingsThemeToStore(v)); // live-apply via AppLayout effect
    save({ theme: v }); // persist Light/Dark
  };
  const onFont = (v: string) => {
    applyFontScale(v); // live-apply (no reload)
    save({ font_scale: v }); // persist
  };

  return (
    <div className="space-y-4">
      <Field label="字体大小 Font">
        <RadioGroup value={font} onChange={onFont} options={FONT_CHOICES} />
      </Field>
      <Field label="主题 Theme">
        <RadioGroup value={theme} onChange={onTheme} options={THEME_CHOICES} />
      </Field>
      <Checkbox
        checked={!!codexLive}
        onChange={(v) => save({ codex_live_search: v })}
        label="技能 Codex 实时搜索 Live skill search"
      />
      {saving && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}

// ── 语音 ───────────────────────────────────────────────────────────────────────
function VoiceSection({
  voices,
  settings,
  saving,
  save,
}: {
  voices: ReturnType<typeof useVoices>;
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  if (voices.isPending || settings.isPending) return <Spinner />;
  if (settings.isError) return <ErrorNote error={settings.error} />;
  if (voices.isError) return <ErrorNote error={voices.error} />;
  if (voices.data?.degraded) return <DegradedNote what="语音" />;

  const s = settings.data ?? {};
  const enabled = !!voices.data?.enabled;
  // Old: ["Off", *TTS_VOICES]
  const voiceOptions = [
    { value: "Off", label: "Off" },
    ...(voices.data?.voices ?? []).map((v) => ({ value: v, label: v })),
  ];
  const currentVoice = s.voice && voiceOptions.some((o) => o.value === s.voice) ? s.voice : "Off";
  const autoplay = !!s.voice_autoplay;
  const mode = s.voice_mode ?? "ptt";
  const narrate = s.voice_narrate ?? true;

  return (
    <div className="space-y-4">
      {!enabled && (
        <p className="rounded-md border border-mast-warn-border bg-mast-warn-bg p-2 text-xs text-mast-warn">
          语音子系统未就绪（缺少 DashScope 密钥）。设置仍会持久化，配置密钥后生效。
        </p>
      )}
      <Field label="TTS 音色 Voice">
        <SelectField value={currentVoice} onChange={(v) => save({ voice: v })} options={voiceOptions} />
      </Field>
      <Field label="默认语音模式 Mode">
        <SelectField
          value={mode}
          onChange={(v) => save({ voice_mode: v })}
          options={[
            { value: "ptt", label: "按住说话 PTT" },
            { value: "wake", label: "唤醒词 Wake（说「MAST」）" },
            { value: "duplex", label: "全双工 Duplex（连续对话）" },
          ]}
        />
      </Field>
      <div className="flex items-center justify-between">
        <span className="text-sm">播报执行过程 Narrate（如「扫描完成」）</span>
        <Toggle checked={narrate} onChange={(v) => save({ voice_narrate: v })} label="播报执行过程" />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-sm">自动朗读助手回复 Autoplay</span>
        <Toggle checked={autoplay} onChange={(v) => save({ voice_autoplay: v })} label="自动朗读助手回复 Autoplay" />
      </div>
      <p className="text-xs text-mast-faint">
        语音走流式 realtime（qwen3-*-flash-realtime），不可用时自动回退批量。全双工/唤醒用本地能量 VAD 断句与打断。
      </p>
      {saving && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}

// ── 其它（只读说明） ───────────────────────────────────────────────────────────
// ── 自治 Autonomy ───────────────────────────────────────────────────────────────
// One numeric setting row: local text state, commit on blur/Enter, empty = 用默认。
// Text (not <input type="number">) so a half-typed value never round-trips through
// Number() and rewrites itself under the cursor.
function NumericSettingRow({
  label,
  hint,
  placeholder,
  value,
  saving,
  onCommit,
  parse,
}: {
  label: string;
  hint: string;
  placeholder: string;
  value: number | null;
  saving: boolean;
  onCommit: (v: number) => void;
  parse: (raw: string) => number | null;
}) {
  const [val, setVal] = useState<string>(value == null ? "" : String(value));
  useEffect(() => {
    setVal(value == null ? "" : String(value));
  }, [value]);

  const commit = () => {
    const t = val.trim();
    if (t === "") {
      // 留空 = 保持不变（后端默认兜底）。NOT "set to 0" — for a ceiling those two
      // mean opposite things, and a blank box must never silently disable a guard.
      setVal(value == null ? "" : String(value));
      return;
    }
    const parsed = parse(t);
    if (parsed == null) {
      setVal(value == null ? "" : String(value));
      return;
    }
    if (parsed !== value) onCommit(parsed);
    setVal(String(parsed));
  };

  return (
    <Field label={label} hint={hint}>
      <input
        type="text"
        inputMode="decimal"
        value={val}
        placeholder={placeholder}
        disabled={saving}
        onChange={(e) => setVal(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur();
        }}
        className="w-32 rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text"
      />
    </Field>
  );
}

function AutonomySection({
  settings,
  saving,
  save,
}: {
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const recursion = settings.data?.orchestrator_recursion_limit ?? null;
  const runBudget = settings.data?.orchestrator_run_budget_usd ?? null;
  const dailyBudget = settings.data?.daily_budget_usd ?? null;
  const wakeMax = settings.data?.wake_max_per_day ?? null;

  return (
    <div className="max-w-md space-y-4">
      <NumericSettingRow
        label="任务步数上限 recursion_limit"
        hint="单次自治任务（run-task）的 super-step 预算，默认 250（下限 50）。越大越能跑完长 campaign；真正空转由跳数熔断和成本闸门更早拦截，这只是最终兜底。留空 = 用默认。"
        placeholder="250"
        value={recursion}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseInt(raw, 10);
          return Number.isNaN(n) ? null : Math.max(50, n);
        }}
        onCommit={(v) => save({ orchestrator_recursion_limit: v })}
      />

      {/* ── 成本闸门 ─────────────────────────────────────────────────
          这两道闸门量的是不同的东西，别用一个代替另一个：
          · 单次上限管「一个 run 跑飞」；
          · 每日上限管「run 的数量」——被唤醒的 agent 每次都是全新的 run，
            带着全新的单次预算，所以单次上限对它完全无感。
          成功步骤也可能形成重复唤醒循环，因此还需要累计成本闸门。 */}
      <NumericSettingRow
        label="单次任务成本上限 (USD)"
        hint="一次自治任务最多花多少钱，超了就停。默认 80；填 0 = 关闭这道闸门。上限应按任务规模设置。账本按时间窗计量，并发的其他调用也会计入，因此可能偏高。"
        placeholder="80"
        value={runBudget}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseFloat(raw);
          return Number.isNaN(n) ? null : Math.max(0, Math.round(n * 100) / 100);
        }}
        onCommit={(v) => save({ orchestrator_run_budget_usd: v })}
      />

      <NumericSettingRow
        label="每日总成本上限 (USD)"
        hint="今天（本地零点起）所有任务加起来最多花多少，超了就不再自动唤醒被搁置的 agent。默认 300；填 0 = 关闭。单次上限看不见「任务数量」，而自动唤醒放大的正是数量——十次唤醒各自都合法，加起来才是问题。"
        placeholder="300"
        value={dailyBudget}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseFloat(raw);
          return Number.isNaN(n) ? null : Math.max(0, Math.round(n * 100) / 100);
        }}
        onCommit={(v) => save({ daily_budget_usd: v })}
      />

      {/* 唤醒次数配额。注意这里的 0 与上面两道金额闸门的 0 含义相反：
          · 金额填 0 = 不设上限（0 就是一个金额设置没填时的样子）；
          · 次数填 0 = 一次都不许（用户往安全计数器里打 0 的意思是「别做这件事」）。
          界面**不提供「无限制」**，这是有意的：这个计数是产物驱动唤醒环唯一的边界，
          别的闸门全是 per-run 的，而每次唤醒都是一个全新的 run（上限全部重置），
          且每一环都成功，空转检测按「重复失败」判定，结构上看不见这种环。 */}
      <NumericSettingRow
        label="每个实验每天最多自动唤醒几次"
        hint="被搁置的 agent 每天最多自动醒来几次（按实验分别计）。默认 6——高于任何正当流水线（六阶段最多一阶段醒一次），远低于「互相唤醒」的死循环。填 0 = 完全不自动唤醒（搁置的 agent 只会等你手动处理）。这是自动唤醒唯一的次数边界，所以不提供「不限次」。"
        placeholder="6"
        value={wakeMax}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseInt(raw, 10);
          // 夹到 >= 0：负数在调度器里表示「关掉这个检查」，那是给测试用的，
          // 绝不能从设置页够到。
          return Number.isNaN(n) ? null : Math.max(0, n);
        }}
        onCommit={(v) => save({ wake_max_per_day: v })}
      />

      {/* ── 单次调用的模型/工具调用上限 ───────────────────────────────
          模型和工具调用次数需要可编辑入口。它们与 recursion_limit 不同：
          后者计 super-step，这里分别计模型与工具调用。 */}
      <NumericSettingRow
        label="单次对话的模型调用上限"
        hint="一轮里模型最多开口几次，默认 30。长的自治流程（修针→扫图→判读）很容易撞上它，撞上就当场中断。改完下次启动生效。不接受 0/负数：内核会把它换回 30。"
        placeholder="30"
        value={settings.data?.chat_model_calls_per_run ?? null}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseInt(raw, 10);
          return Number.isNaN(n) ? null : Math.max(1, n);
        }}
        onCommit={(v) => save({ chat_model_calls_per_run: v })}
      />

      <NumericSettingRow
        label="单次对话的工具调用上限"
        hint="同一轮里最多执行几个工具/技能，默认 80。改完下次启动生效。"
        placeholder="80"
        value={settings.data?.chat_tool_calls_per_run ?? null}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseInt(raw, 10);
          return Number.isNaN(n) ? null : Math.max(1, n);
        }}
        onCommit={(v) => save({ chat_tool_calls_per_run: v })}
      />

      {/* 跨会话累计上限。这里的 0 与上面两个 per-run 上限含义相反：
          per-run 填 0 会被内核换回出厂默认（所以后端直接拒），
          累计填 0 = 不限（这是它的出厂状态）。 */}
      <NumericSettingRow
        label="跨会话累计的模型调用上限"
        hint="一条会话从建立起累计的模型调用上限，0 = 不限（默认）。累计上限不会自己重置，设小了会把一条长期会话彻底卡死——除非你确切知道为什么要开它，留 0。"
        placeholder="0"
        value={settings.data?.chat_model_calls_per_thread ?? null}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseInt(raw, 10);
          return Number.isNaN(n) ? null : Math.max(0, n);
        }}
        onCommit={(v) => save({ chat_model_calls_per_thread: v })}
      />

      <NumericSettingRow
        label="跨会话累计的工具调用上限"
        hint="同上，0 = 不限（默认）。"
        placeholder="0"
        value={settings.data?.chat_tool_calls_per_thread ?? null}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseInt(raw, 10);
          return Number.isNaN(n) ? null : Math.max(0, n);
        }}
        onCommit={(v) => save({ chat_tool_calls_per_thread: v })}
      />
    </div>
  );
}

// ── 对话处理与归档 ───────────────────────────────────────────────────────────
// 八个开关，2026-08-10 之前**一个都写不进去**：它们在 SettingsStore.KNOWN_KEYS
// 里、core 每次都读，只是不在 SettingsWriteRequest 上，于是 POST 被 pydantic
// 静默丢掉而响应仍然 ok=true。补齐写入口之后才轮到这里 —— 一个存得进却没有
// 界面的键，用户同样够不着。
function ConversationArchivingSection({
  settings,
  saving,
  save,
}: {
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const s = settings.data;
  // 这一组的出厂默认全是「开」，而 absent 就是「开」——所以 `?? true`
  // 不是随手写的默认值，它是后端 `v is None → True` 的镜像。
  const on = (v: boolean | null | undefined) => v ?? true;

  return (
    <div className="max-w-xl space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm">
          工具返回精炼 Tool refine
          <span className="ml-2 text-xs text-mast-faint">
            压缩过长的工具返回再给模型看。关掉它模型读到的是原文。改完下次启动生效。
          </span>
        </span>
        <Toggle
          checked={on(s?.tool_refine_enabled)}
          onChange={(v) => save({ tool_refine_enabled: v })}
          label="工具返回精炼"
        />
      </div>

      <NumericSettingRow
        label="精炼阈值（字符）"
        hint="只精炼比这个长的工具返回，默认 600。想整个关掉请用上面的开关，不要填 0。"
        placeholder="600"
        value={s?.tool_refine_min_chars ?? null}
        saving={saving}
        parse={(raw) => {
          const n = Number.parseInt(raw, 10);
          return Number.isNaN(n) ? null : Math.max(1, n);
        }}
        onCommit={(v) => save({ tool_refine_min_chars: v })}
      />

      <Field
        label="压缩摘要模型 Compaction model"
        hint="对话压缩用哪个模型写摘要。留空 = 跟随当前 agent 的模型（默认，够用）。改完下次启动生效。"
      >
        <TextCommitField
          value={s?.compaction_model ?? ""}
          saving={saving}
          placeholder="（跟随当前模型）"
          onCommit={(v) => save({ compaction_model: v })}
        />
      </Field>

      <div className="flex items-center justify-between">
        <span className="text-sm">
          全文到货后自动续跑 Literature auto-resume
          <span className="ml-2 text-xs text-mast-faint">
            你上传了 agent 要的论文之后，自动把它停住的那一轮跑完。
          </span>
        </span>
        <Toggle
          checked={on(s?.literature_fetch_auto_resume)}
          onChange={(v) => save({ literature_fetch_auto_resume: v })}
          label="全文到货后自动续跑"
        />
      </div>

      <div className="border-t border-mast-border pt-4" />

      <div className="flex items-center justify-between">
        <span className="text-sm">
          归档 Nanonis 产物 Ingest
          <span className="ml-2 text-xs text-mast-faint">
            把测量文件复制进实验文件夹。总开关，改完下次启动生效。
          </span>
        </span>
        <Toggle
          checked={on(s?.ingest_enabled)}
          onChange={(v) => save({ ingest_enabled: v })}
          label="归档 Nanonis 产物"
        />
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm">
          后台兜底扫盘 Watcher
          <span className="ml-2 text-xs text-mast-faint">
            把你手动存的文件也收进来（source=manual）。当场生效 —— 但前提是上面那个
            总开关在**本次启动时**是开着的：关着的话归档整条链根本没建起来，这里开了也不动。
          </span>
        </span>
        <Toggle
          checked={on(s?.ingest_watcher_enabled)}
          onChange={(v) => save({ ingest_watcher_enabled: v })}
          label="后台兜底扫盘"
        />
      </div>

      <Field
        label="归档方式 Copy mode"
        hint="hardlink 不占额外磁盘，但 Nanonis 保存目录和实验文件夹必须在同一个卷上；跨卷时只能用 copy。"
      >
        <SelectField
          value={(s?.ingest_copy_mode as "copy" | "hardlink") ?? "copy"}
          onChange={(v) => save({ ingest_copy_mode: v })}
          options={[
            { value: "copy", label: "copy — 复制一份（任何情况都行）" },
            { value: "hardlink", label: "hardlink — 硬链接（省盘，仅限同卷）" },
          ]}
        />
      </Field>

      <div className="flex items-center justify-between">
        <span className="text-sm">
          环境读数同时写 CSV
          <span className="ml-2 text-xs text-mast-faint">
            温度等读数额外落一份 CSV，供 Excel / Origin 直接打开。
          </span>
        </span>
        <Toggle
          checked={on(s?.env_csv_enabled)}
          onChange={(v) => save({ env_csv_enabled: v })}
          label="环境读数同时写 CSV"
        />
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm">
          对话增量导出 Conversation export
          <span className="ml-2 text-xs text-mast-faint">
            导出成 jsonl + md。这是唯一能保住被 8000 行上限裁掉的那部分历史的地方。
          </span>
        </span>
        <Toggle
          checked={on(s?.conv_export_enabled)}
          onChange={(v) => save({ conv_export_enabled: v })}
          label="对话增量导出"
        />
      </div>
    </div>
  );
}

/** 文本输入，失焦/回车才提交（和 NumericSettingRow 同款节奏：不按键就存）。 */
function TextCommitField({
  value,
  saving,
  placeholder,
  onCommit,
}: {
  value: string;
  saving: boolean;
  placeholder?: string;
  onCommit: (v: string) => void;
}) {
  const [val, setVal] = useState(value);
  useEffect(() => {
    setVal(value);
  }, [value]);
  const commit = () => {
    const t = val.trim();
    if (t !== value) onCommit(t);
    setVal(t);
  };
  return (
    <input
      type="text"
      value={val}
      placeholder={placeholder}
      disabled={saving}
      onChange={(e) => setVal(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur();
      }}
      className="w-full rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm text-mast-text"
    />
  );
}

// ── 视觉判别阈值 Vision ──────────────────────────────────────────────────────
// The VIGIL tip-quality valves. Model weights are fixed; these only tune how
// strict the good/bad + usable judgement is. The 6 knobs = overall good/bad cut
// + per-dimension cuts (Q/N/K/T/S). C (segmentation) has no pass/fail cut.
//
// `def` MUST match the VisionThresholds defaults in
// MASTv2/mast/vision/thresholds.py (6 numbers, kept in sync by hand — the
// backend is authoritative; this only seeds the input when nothing is persisted
// yet). Persisted values always win over `def`.
const VISION_THRESHOLD_KNOBS: {
  key: string; label: string; hint: string; min: number; max: number; step: string; def: number;
}[] = [
  {
    key: "coarse_good_threshold", label: "总门槛 好/坏切分", def: 0.35, min: 0.3, max: 0.7, step: "0.05",
    hint: "融合分 ≥ 此值判「良好」。最关键——直接决定「针尖坏→处理」是否触发。调低 = 更宽松，更少触发针尖处理。（旧严值 0.5）",
  },
  {
    key: "m0_quality_min", label: "Q 质量下限", def: 62, min: 60, max: 85, step: "1",
    hint: "质量分 ≥ 此值（且轴比达标）才算最优形貌 M0。调低 = 更多针尖算优质 / 可用。（旧严值 72）",
  },
  {
    key: "multi_apex_p_max", label: "N 多针尖容忍", def: 0.7, min: 0.2, max: 0.8, step: "0.05",
    hint: "多针尖概率 > 此值判为多针尖（不可用）。最强坏信号。调高 = 更能容忍疑似多针尖。（旧严值 0.5）",
  },
  {
    key: "contam_p_max", label: "K 污染容忍", def: 0.7, min: 0.2, max: 0.8, step: "0.05",
    hint: "污染概率 > 此值判为污染（不可用）。调高 = 更能容忍疑似污染。（旧严值 0.5）",
  },
  {
    key: "instability_p_max", label: "T 不稳定容忍", def: 0.7, min: 0.2, max: 0.8, step: "0.05",
    hint: "不稳定概率 > 此值判为扰动（不可用）。调高 = 更能容忍疑似不稳定。（旧严值 0.5）",
  },
  {
    key: "m0_axis_ratio_min", label: "S 轴比下限", def: 0.45, min: 0.4, max: 0.9, step: "0.05",
    hint: "针尖轴比 ≥ 此值（且质量达标）才算最优形貌 M0。调低 = 更能接受略椭圆的针尖。（旧严值 0.6）",
  },
];

// Per-instrument knobs for the network-free classical tip-quality tools
// (mast/vision/classical_thresholds.py). No model involved — pure algorithms.
const CLASSICAL_THRESHOLD_KNOBS: {
  key: string; label: string; hint: string; min: number; max: number; step: string; def: number;
}[] = [
  {
    key: "tq_double_threshold", label: "双针尖判定阈值", def: 0.18, min: 0.05, max: 1.0, step: "0.01",
    hint: "自相关副本峰分超过此值时判双/多针尖。默认值仅为起点，需根据图像纹理和目标仪器独立验证。",
  },
  {
    key: "tq_instability_max", label: "正反扫描不稳容忍", def: 0.5, min: 0.1, max: 1.0, step: "0.05",
    hint: "trace/retrace 移位容忍互相关不稳度超过此值时判不稳定。阈值需按目标仪器独立验证。",
  },
  {
    key: "tq_sharpness_min", label: "FFT 锐度下限", def: 8.0, min: 1.0, max: 100.0, step: "1",
    hint: "FFT 锐度 < 此值且无晶格 → 判「无可分辨表面」（坏，堵住噪声→good）。调低 = 更宽松。",
  },
  {
    key: "tq_change_threshold", label: "中途换针 t 阈值", def: 6.0, min: 2.0, max: 30.0, step: "0.5",
    hint: "逐行统计跳变 t 值超过此值时判扫描中途换针。默认值为合成示例，需独立验证。",
  },
  {
    key: "oscillation_threshold", label: "反馈振荡峰比阈值", def: 3.0, min: 1.5, max: 50.0, step: "0.5",
    hint: "轴上/轴外 FFT 峰比 > 此值 → 判反馈振荡/振铃。调高 = 只报更强的条纹。",
  },
  {
    key: "iz_barrier_min", label: "I(z) 势垒下限 (eV)", def: 0.5, min: 0.0, max: 5.0, step: "0.1",
    hint: "I(z) 表观势垒低于此值不算「干净指数」。洁净金属针尖 ~4-5 eV。",
  },
  {
    key: "iz_barrier_max", label: "I(z) 势垒上限 (eV)", def: 8.0, min: 3.0, max: 15.0, step: "0.5",
    hint: "I(z) 表观势垒高于此值不算「干净指数」（异常大多为拟合/噪声问题）。",
  },
];

const _fmtThreshold = (v: number) => String(Math.round(v * 1000) / 1000);

function ThresholdRow({
  label, hint, value, min, max, step, disabled, onCommit,
}: {
  label: string; hint: string; value: number; min: number; max: number;
  step: string; disabled: boolean; onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(_fmtThreshold(value));
  useEffect(() => {
    setText(_fmtThreshold(value));
  }, [value]);

  const commit = () => {
    const t = text.trim();
    const n = Number.parseFloat(t);
    if (t === "" || Number.isNaN(n)) {
      setText(_fmtThreshold(value)); // revert junk
      return;
    }
    const clamped = Math.min(max, Math.max(min, n));
    setText(_fmtThreshold(clamped));
    if (clamped !== value) onCommit(clamped);
  };

  return (
    <Field label={label} hint={hint}>
      <input
        type="text"
        inputMode="decimal"
        value={text}
        step={step}
        disabled={disabled}
        placeholder={_fmtThreshold(value)}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur();
        }}
        className="w-28 rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
      />
    </Field>
  );
}

// ── 实验默认参数  ─────────────────────────────────────────────────────────
// The operator's usual "扫多大 / 扫多快" and the setpoint / bias they normally
// start from. These are PREFERENCES, not bounds — they are injected into the IC /
// experiment_design agents' context (via mast.agents._shared.experiment_prefs) as
// "prefer these defaults", and SafetyLimits still enforces the hard ranges. A
// blank field is simply not sent (the middleware skips it).
//
// `def` is only the placeholder shown when nothing is persisted yet; there is NO
// baked-in default value (an unset preference must stay unset so the agent falls
// back to its own judgement — we never invent a scan size on the operator's behalf).
const EXPERIMENT_DEFAULT_FIELDS: {
  key: string; label: string; hint: string; min: number; max: number; step: string; placeholder: string;
}[] = [
  {
    key: "scan_size_nm", label: "扫描尺寸 (边长)", placeholder: "如 50", min: 0, max: 1e6, step: "1",
    hint: "常用扫描帧的边长，单位 nm。新扫描默认从这个尺寸起。",
  },
  {
    key: "scan_speed_nm_s", label: "扫描速度", placeholder: "如 20", min: 0, max: 1e6, step: "1",
    hint: "常用扫描速度，单位 nm/s。",
  },
  {
    key: "line_time_s", label: "每线时间", placeholder: "如 0.5", min: 0, max: 1e4, step: "0.1",
    hint: "每条扫描线的时间，单位 s（扫描速度的另一种表述）。",
  },
  {
    key: "scan_lines", label: "扫描线数 (分辨率)", placeholder: "如 256", min: 1, max: 8192, step: "1",
    hint: "每帧线数 / 像素分辨率。",
  },
  {
    key: "scan_angle_deg", label: "扫描旋转角", placeholder: "如 0", min: -360, max: 360, step: "1",
    hint: "扫描帧旋转角度，单位 °。",
  },
  {
    key: "setpoint_pa", label: "电流设定点 setpoint", placeholder: "如 100", min: 0, max: 1e8, step: "1",
    hint: "常用隧道电流设定点，单位 pA。",
  },
  {
    key: "bias_v", label: "偏压 bias", placeholder: "如 0.5", min: -10, max: 10, step: "0.1",
    hint: "常用样品偏压，单位 V（安全范围 −10…+10 V）。",
  },
];
const EXPERIMENT_NOTES_KEY = "notes";
// Enumerated preference: scan direction. "" = not set (removed from the dict).
const SCAN_DIRECTION_KEY = "scan_direction";
const SCAN_DIRECTION_CHOICES = [
  { value: "", label: "（未设定）" },
  { value: "up", label: "从下往上 (up)" },
  { value: "down", label: "从上往下 (down)" },
];

function ExperimentDefaultsSection({
  settings,
  saving,
  save,
}: {
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const persisted = (settings.data?.experiment_defaults ?? {}) as Record<string, unknown>;
  const numAt = (key: string): number | undefined =>
    typeof persisted[key] === "number" ? (persisted[key] as number) : undefined;
  const notes = typeof persisted[EXPERIMENT_NOTES_KEY] === "string"
    ? (persisted[EXPERIMENT_NOTES_KEY] as string)
    : "";
  const scanDir = typeof persisted[SCAN_DIRECTION_KEY] === "string"
    ? (persisted[SCAN_DIRECTION_KEY] as string)
    : "";

  // Whole-replace guard: SettingsStore replaces the experiment_defaults dict
  // wholesale (it does not merge), so a save must carry EVERY currently-set
  // field plus the one change — otherwise the others would be wiped. We only
  // include fields that are actually set (a blank preference stays unset).
  const buildBase = (): Record<string, unknown> => {
    const next: Record<string, unknown> = {};
    for (const f of EXPERIMENT_DEFAULT_FIELDS) {
      const v = numAt(f.key);
      if (typeof v === "number") next[f.key] = v;
    }
    if (scanDir) next[SCAN_DIRECTION_KEY] = scanDir;
    if (notes.trim()) next[EXPERIMENT_NOTES_KEY] = notes.trim();
    return next;
  };

  const commitNum = (key: string, value: number | null) => {
    const next = buildBase();
    if (value === null) delete next[key];
    else next[key] = value;
    save({ experiment_defaults: next });
  };

  const commitScanDir = (value: string) => {
    const next = buildBase();
    if (value) next[SCAN_DIRECTION_KEY] = value;
    else delete next[SCAN_DIRECTION_KEY];
    save({ experiment_defaults: next });
  };

  const commitNotes = (value: string) => {
    const next = buildBase();
    if (value.trim()) next[EXPERIMENT_NOTES_KEY] = value.trim();
    else delete next[EXPERIMENT_NOTES_KEY];
    save({ experiment_defaults: next });
  };

  // Empty dict → clears every preference (the agent falls back to its own judgement).
  const resetDefaults = () => save({ experiment_defaults: {} });

  const anySet = Object.keys(buildBase()).length > 0;

  return (
    <div className="max-w-lg space-y-3">
      <p className="text-xs text-mast-muted">
        你惯用的实验起点：<b>扫多大、扫多快</b>、常用 setpoint / bias。设定后会作为<b>偏好</b>
        注入 IC / 实验设计 agent 的上下文——在你没有相反要求时，agent 优先采用它们。
        <b>它们是偏好，不是硬上限</b>，安全边界仍由 SafetyLimits 约束。留空的项不注入。
      </p>
      {EXPERIMENT_DEFAULT_FIELDS.map((f) => (
        <ExperimentDefaultRow
          key={f.key}
          label={f.label}
          hint={f.hint}
          value={numAt(f.key)}
          min={f.min}
          max={f.max}
          step={f.step}
          placeholder={f.placeholder}
          disabled={saving}
          onCommit={(v) => commitNum(f.key, v)}
        />
      ))}
      <Field label="扫描方向" hint="常用的慢扫描方向（Nanonis scan direction）。">
        <SelectField
          value={scanDir}
          onChange={commitScanDir}
          options={SCAN_DIRECTION_CHOICES}
        />
      </Field>
      <Field label="其他偏好 (自由文本)" hint="例如：优先常温、勿超过 1 V、避免高电流…（会原样告知 agent）">
        <textarea
          defaultValue={notes}
          key={notes}
          rows={2}
          disabled={saving}
          placeholder="可选"
          onBlur={(e) => {
            if (e.target.value.trim() !== notes.trim()) commitNotes(e.target.value);
          }}
          className="w-full rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text disabled:opacity-50"
        />
      </Field>
      <div className="flex items-center gap-3 pt-1">
        <button
          type="button"
          onClick={resetDefaults}
          disabled={saving || !anySet}
          className="rounded border border-mast-border bg-mast-panel px-3 py-1 text-xs text-mast-muted hover:text-mast-text disabled:opacity-50"
        >
          全部清空
        </button>
        {!anySet && <span className="text-xs text-mast-muted">（未设置任何偏好）</span>}
      </div>
      {saving && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}

// ── Instrument profile (换样品退针 / 进针 dI/dV) ──────────────────────────────
// Per-rig hardware facts the approach/retract logic depends on, plus a read-only
// view of the auto-learned dI/dV-at-contact calibration.
const INSTRUMENT_NUM_FIELDS: {
  key: string; label: string; hint: string; min: number; max: number; step: string; placeholder: string;
}[] = [
  { key: "z_recede_min_nm", label: "退针 Z 自检阈值", placeholder: "如 1", min: 0, max: 1e4, step: "0.1",
    hint: "退针每级开反馈后，Z 压电朝伸长方向移动超过此阈值（nm）才判定「确认在远离」。" },
  { key: "z_settle_timeout_s", label: "退针 Z 稳定预算", placeholder: "如 20", min: 0.5, max: 300, step: "0.5",
    hint: "退针自检等待 Z 反馈稳定的上限；稳定后立即读取。超时表示无法得出结论。请按目标仪器的响应时间设置上限。" },
  { key: "retract_total_steps", label: "退针总步数", placeholder: "如 3000", min: 1, max: 1e6, step: "1",
    hint: "换样品/关机退针的粗动马达总步数。分级 1→10→100→剩余 退完，每级 Z 自检。" },
  { key: "retract_step_max", label: "单次粗动上限", placeholder: "如 1000", min: 1, max: 1000, step: "1",
    hint: "单次 Motor_StartMove 的最大步数（Nanonis 硬件上限 1000）。" },
  { key: "lockin_signal_index", label: "dI/dV 信号索引", placeholder: "如 8", min: 0, max: 127, step: "1",
    hint: "lock-in R (dI/dV) 输出对应的 Nanonis 信号索引（Signals_ValGet 用）。进针 dI/dV 测距 + 退针辅助自检需要它；不填则跳过 dI/dV。" },
  { key: "lockin_mod_amp_v", label: "dI/dV 调制幅度", placeholder: "如 0.02", min: 0, max: 1, step: "0.001",
    hint: "lock-in 加在电流上的偏压调制幅度，单位 V（默认 20 mV = 0.02 V）。" },
  { key: "lockin_mod_freq_hz", label: "dI/dV 调制频率", placeholder: "如 973", min: 0, max: 1e5, step: "1",
    hint: "调制频率，单位 Hz。建议避开 50/60 Hz 工频谐波与机械共振。" },
  { key: "lockin_x_signal_index", label: "解调 X 信号索引", placeholder: "如 24", min: 0, max: 127, step: "1",
    hint: "解调器的 X 分量走哪一路 RT 信号。这是接线事实，软件读不出来；AutoPhase 缺它会拒绝对齐相位而不是猜（猜错读到的是另一路信号，算出来的角度看上去一样合理）。" },
  { key: "lockin_y_signal_index", label: "解调 Y 信号索引", placeholder: "如 25", min: 0, max: 127, step: "1",
    hint: "解调器的 Y 分量走哪一路 RT 信号。同 X，两个都填了 AutoPhase 才可用。" },
  { key: "approach_didv_engage_frac", label: "进针 dI/dV 判据比例", placeholder: "如 0.7", min: 0, max: 2, step: "0.05",
    hint: "进针时 dI/dV 达到「到样品标定值」的此比例，视为接触在望。" },
  // 进针参数组。这三个数以前没有持久化的地方，每次进针前都要
  // 由模型转成工具调用去写硬件——曾经发生过一次转写错误，把 3e-12 写成了 3（大一万亿倍）。
  // 填在这里之后，ApplyZCtrlPreset('approach') 由代码直接取值写入硬件，全程不经模型。
  { key: "approach_p_gain_m", label: "进针 Z 比例增益 P", placeholder: "如 3e-12", min: 1e-15, max: 1e-6, step: "1e-13",
    hint: "Nanonis 面板 Z-Controller → Proportional (m)。面板上的「3.000p」= 3e-12。进针与扫图通常用同一个 P。" },
  { key: "approach_i_gain_m_per_s", label: "进针 Z 积分增益 I", placeholder: "如 1.8e-7", min: 1e-12, max: 1e-3, step: "1e-9",
    hint: "Nanonis 面板 Z-Controller → Integral (m/s)。面板上的「180.000n」= 1.8e-7。进针比扫图快（环更快），扫图值在下方扫描档位表里单独设。" },
  { key: "approach_setpoint_a", label: "进针电流设定点", placeholder: "如 1.5e-10", min: 1e-12, max: 1e-7, step: "1e-12",
    hint: "进针用的隧道电流设定点，单位安培（150 pA = 1.5e-10）。越大进针越快。扫图设定点在扫描档位表里。" },
  { key: "approach_steps_per_cycle", label: "Auto Approach 每轮步数", placeholder: "如 3", min: 1, max: 1000, step: "1",
    hint: "Nanonis Auto Approach 面板的 Number of Pulses。仅作记录与核对，MAST 不改写面板设置。" },
  { key: "approach_expected_steps", label: "进针预计总步数", placeholder: "如 2500", min: 1, max: 1e6, step: "100",
    hint: "你的经验值。用于判断某次进针是否异常（远少于此值可能是提前触发，远多于可能是没在靠近）。" },
  // 扫描地图避让半径 —— 这些是用户自己的经验值：一次修针尖的碎屑实际散多远，
  // 取决于这根针、这块样品、这个温度，只有他们盯过。默认值只是起点。
  { key: "avoid_radius_tip_shape_nm", label: "修针尖避让半径", placeholder: "如 30", min: 0, max: 1e5, step: "1",
    hint: "修针尖点周围多大范围内不再扫图（nm）。碎屑散布范围随针尖/样品/温度变，按你的经验调。" },
  { key: "avoid_radius_pulse_nm", label: "电脉冲避让半径", placeholder: "如 150", min: 0, max: 1e5, step: "1",
    hint: "电脉冲点的避让半径（nm）。" },
  { key: "scan_spacing_factor", label: "计划扫描点间距", placeholder: "如 1.2", min: 1, max: 20, step: "0.1",
    hint: "相邻计划帧的中心相距几个扫描框。1.2 = 紧挨着排（默认，取样最密），3 = 每帧之间留两帧空。想让一片区域被更分散地采样时调大它——两种巡览策略都受它影响。" },
  { key: "avoid_radius_crash_nm", label: "撞针避让半径", placeholder: "如 150", min: 0, max: 1e5, step: "1",
    hint: "撞针点的避让半径（nm）。撞过的地方会永久留在地图上，不随内存熔断计时器过期。" },
  { key: "avoid_radius_approach_nm", label: "进针扎痕避让半径", placeholder: "如 200", min: 0, max: 1e5, step: "1",
    hint: "进针点的避让半径（nm）。仅当下方「进针是否留扎痕」不是「不会」时才生效。" },
  { key: "xy_motor_step_m", label: "XY 粗动单步位移", placeholder: "如 1e-7（可留空）", min: 0, max: 1e-3, step: "1e-9",
    hint: "可选标定，单位米。仅用于在粗动标记上注一句「大概走了多远」；粗动是开环、无位置反馈，没有任何选点逻辑依赖它。留空则不注。" },
  // 扫描几何 / 调平（AutoTilt + scan_resolver）。这四个是本机的物理事实，
  // 不填就一直是默认值——而默认值只是「任何 STM 都不会离谱」的起点。
  { key: "z_range_m", label: "Z 压电总量程", placeholder: "如 1.5e-6", min: 1e-9, max: 1e-4, step: "1e-9",
    hint: "调平触发判据的分母：判据是「这一帧的斜坡吃掉多少 Z 量程」，所以同一个倾角在 1 µm 帧和 10 nm 帧上自动给出不同的紧迫程度。" },
  { key: "tilt_limit_deg", label: "倾斜补偿上限（单轴）", placeholder: "如 5", min: 0, max: 45, step: "0.1",
    hint: "压电倾斜补偿的绝对上限（度）。补过头会吃掉 XY 行程并让 Z 在帧角上打满；5° 对任何 STM 都已是很大的失配角。" },
  { key: "v_tip_max_m_s", label: "针尖横向速度上限", placeholder: "如 2e-6", min: 1e-12, max: 1e-3, step: "1e-9",
    hint: "单位 m/s。像素数、每线时间、帧宽单独看都合法，乘起来才知道针尖扫得多快——这个组合约束由 scan_resolver 检查。" },
  { key: "z_noise_floor_m", label: "Z 噪声底", placeholder: "留空 = 每次从数据估计", min: 0, max: 1e-9, step: "1e-13",
    hint: "单位米。留空则每帧从数据现估（RANSAC 内点阈、圆拟合台阶否决、PI 整定验证都要用它）；填了就当本机基准值，省掉每次估计。" },
  // 横向粗动换区（RelocateCoarseXY）。默认值都是起点，真值只能实机标：
  // 移一次、扫一张图、看地貌是不是**完全**换了。
  { key: "xy_prewithdraw_steps", label: "横向移动前退针步数", placeholder: "如 100", min: 0, max: 100000, step: "1",
    hint: "横向粗动前用粗动马达退多少步清障。压电收到顶只有 1–2 µm 余量，而样品台侧滑时的垂直跳动、样品倾斜和针尖长度都远不止这个数——这一步压电退针替代不了。" },
  { key: "xy_move_chunk_steps", label: "横向移动分块步数", placeholder: "如 50", min: 1, max: 1000, step: "1",
    hint: "每移动这么多步就回读一次电流（+qPlus 振幅）并重评真空。块越小看护越密，但每块都有一次 TCP 往返。" },
  { key: "au_step_pm", label: "单原子台阶高度（本机读数）", placeholder: "如 207", min: 50, max: 500, step: "0.5",
    hint: "pm。多针尖判据使用的相邻台面高度单位。填目标仪器在合适参照样品上读出的标定值，不能把物理常量直接当作仪器读数。更换扫描器或重标 Z 后需要重测。" },
  { key: "xy_site_spacing_steps", label: "粗动站点最小间距", placeholder: "如 200", min: 1, max: 1000000, step: "1",
    hint: "两片工作区之间至少隔多少步。必须让新区域完全跳出压电量程（±1.5 µm），否则「换区」只是把旧区域挪进视野。" },
  { key: "xy_axis_step_budget", label: "单轴行程预算", placeholder: "如 5000", min: 1, max: 10000000, step: "1",
    hint: "单位步。粗动台行程有限，走到头只会空滑（无害）但位置认知会全部丢失。超出预算的落点会被规划器拒绝。" },
  { key: "xy_step_uncertainty_frac", label: "单步位移不确定度", placeholder: "如 0.3", min: 0, max: 1, step: "0.05",
    hint: "相对值。开环步进器的步长随驱动幅度/负载/温度漂移，所以大地图上的站点画成模糊斑而不是点；这个数决定斑有多大。低温下应当调大。" },
  // 粗动真空互锁。放电区（Paschen 极小值附近）约 0.1–1000 Pa = 1e-3–10 mbar，
  // 抽气/放气途中正好穿过它，而那也是最有人想动粗动的时候。
  { key: "coarse_motion_max_pressure_pa", label: "允许粗动的压强上限", placeholder: "如 1e-2", min: 0, max: 1e5, step: "1e-3",
    hint: "单位 Pa。高于此值一律拒绝粗动。默认 1e-2 Pa 比放电区下沿（0.1 Pa）低一个数量级，任何真实 UHV STM 都远低于它。" },
  { key: "vacuum_reading_max_age_s", label: "真空计读数有效期", placeholder: "如 60", min: 1, max: 3600, step: "1",
    hint: "单位秒。超过这个时长的读数不能给现在授权——抽气/放气时压强变化很快，几分钟前的数字说明不了此刻。" },
  { key: "vacuum_gauge_full_scale_pa", label: "真空计量程上限", placeholder: "如 1e-1（DL-7）", min: 1e-9, max: 1e6, step: "1e-3",
    hint: "单位 Pa。读数到满量程 80% 以上视为「超量程、判断不了」而非「刚好卡在阈值下」——DL-7 的帧解析不校验指数位，超量程可能解出一个看着合理的小数。换规时改这里。" },
  { key: "vacuum_gauge_min_pa", label: "真空计量程下限", placeholder: "如 5e-8（DL-7）", min: 1e-12, max: 1e6, step: "1e-9",
    hint: "单位 Pa。规触底时发出的数（下限值 / 0 / 噪声）看起来正好像「真空非常好」。在冷阴极规上无害（下限远低于允许上限，触底只意味着更安全）；在**粗糙真空规**（Pirani / 电容薄膜规，下限约 0.5–1 Pa）上则是漏洞——触底只说明「低于 1 Pa」，而那包含 0.5 Pa，正在放电区里。填对这个数，只有这只规的下限本身低于允许上限时，触底读数才会放行。" },
  // 信号链（2026-07-31，与针尖登记同批）。
  { key: "preamp_gain_v_per_a", label: "前置放大器跨阻增益", placeholder: "如 1e9", min: 1e3, max: 1e13, step: "1e3",
    hint: "单位 V/A：电流 = 读数电压 ÷ 此值。Nanonis 只给得到增益「索引」，索引对应多少 V/A 不在任何地方——只能在这里登记。填错一个量级，MAST 报出去的每一个电流值就整体错一个量级，而且完全静默。" },
  { key: "preamp_full_scale_a", label: "前置放大器满量程电流", placeholder: "如 1e-8（±10 nA）", min: 1e-12, max: 1e-2, step: "1e-9",
    hint: "单位 A：这台前放能测到的最大电流（面板/手册上写的「±10 nA」那个数）。与上面的增益是同一件事的两种说法（满量程 ≈ ±10 V ÷ 增益），两个都填就能互相验一次。它是**设定点上限**与**电流贴轨判据**的物理依据——高于前放量程的设定点物理上达不到，反馈环拿不到目标电流就会一路把 Z 推向样品直到撞针。新仪器初始化页面会据它给出这两条线的建议值。" },
];
// 自由文本字段（型号一类）。数值 spec 和枚举 spec 都装不下它们。
//
// ⚠️ 绝不能把它们塞进 INSTRUMENT_CALIB_KEYS 图省事：下面 buildBase() 对 calib
// 键只回显 typeof === "number"，字符串会在下一次任意设置保存时被静默抹掉——
// 那正是 2026-07-31 qPlus 基线事故的形状。TEXT 键必须有自己的回显分支。
const INSTRUMENT_TEXT_FIELDS: { key: string; label: string; hint: string; placeholder: string }[] = [
  { key: "preamp_model", label: "前置放大器型号", placeholder: "如 FEMTO DLPCA-200",
    hint: "记下来是为了让 AI 解释噪声与带宽时知道自己在跟什么打交道（不同前放的噪声底、带宽、最大输入差别很大）。" },
];
const RETRACT_DIR_KEY = "retract_motor_dir";
const RETRACT_DIR_CHOICES = [
  { value: "z+", label: "Z+ (Nanonis 标准: 远离样品)" },
  { value: "z-", label: "Z− (反向装置)" },
];
const Z_EXTEND_SIGN_KEY = "z_extend_sign";
const Z_EXTEND_SIGN_CHOICES = [
  { value: "+1", label: "伸长时 Z 增大 (+1)" },
  { value: "-1", label: "伸长时 Z 减小 (−1)" },
];
// 扫描地图选点能力（决定「换区」是否是可选项，以及走哪条扫描路径）。
const XY_COARSE_KEY = "xy_coarse_motion";
const XY_COARSE_CHOICES = [
  { value: "yes", label: "有 — 可粗动换到新区域" },
  { value: "no", label: "无 — 换区只能靠插拔样品" },
];
const APPROACH_DMG_KEY = "approach_damages_surface";
const APPROACH_DMG_CHOICES = [
  { value: "unknown", label: "未知 — 按保守处理（同「会」）" },
  { value: "yes", label: "会 — 避开进针点" },
  { value: "no", label: "不会（好机器）— 进针点只作历史记录" },
];
// 进针会不会**扎针**——与上面那条（会不会点伤**表面**）是两件事：一件的代价是
// 一片表面，另一件的代价是针 + 之后几小时的修针。它决定 AI 敢不敢自己发起进针。
// ⚠️ 目前只注入给模型，不是硬门：HITL 门在建图时从 safety_level 派生，运行时
// 填的值改不了它（见 core/instrument_profile._format_approach_supervision_line）。
const APPROACH_SUPERVISION_KEY = "approach_supervision";
const APPROACH_SUPERVISION_CHOICES = [
  { value: "unknown", label: "未知 — 按保守处理（同「必须有人在场」）" },
  { value: "attended", label: "必须有人在场 — 本机进针会撞/扎针" },
  { value: "unattended", label: "可无人值守 — 本机 Auto Approach 不扎针" },
];
const SCAN_PATH_KEY = "scan_path_strategy";
const SCAN_PATH_CHOICES = [
  { value: "auto", label: "自动 — 按有无 XY 粗动推导" },
  { value: "center_first", label: "中心优先 — 压电蠕变最小" },
  { value: "perimeter_inward", label: "外圈→内圈 — 可用面积利用最大化" },
];
// 拿不到可信压强时怎么办。默认拒绝，但用户可以签一个带有效期的字。
const VACUUM_MODE_KEY = "vacuum_interlock_mode";
const VACUUM_MODE_CHOICES = [
  { value: "gauge_or_attest", label: "真空计优先，读不到时可由用户签署（默认）" },
  { value: "gauge_only", label: "只认真空计 — 读不到就绝不粗动" },
  { value: "off", label: "关闭阻断 — 仍记录裁决，风险由人承担" },
];
// 偏压加在样品还是针尖——决定 dI/dV 谱里正偏压对应占据态还是空态。
// 默认 unknown 而不是 sample：虽然样品偏压是绝大多数机器的约定，但「按最常见
// 约定默认下去」正是这类错误的来源，而它在数据里完全看不出来。
// lock-in 解调形式。**不是显示口味，是判据的物理含义**：R 是幅度、恒为正、随接近
// 单调增大；X 是带符号投影，相位在接近途中转动时会**穿零**，于是「越近越大」会中途
// 掉下去 —— 而那是相位问题，不是针尖退开了。
// 可用信号形式取决于仪器配置，必须由使用者明确声明。
const LOCKIN_FORM_KEY = "lockin_readout_form";
const LOCKIN_FORM_CHOICES = [
  { value: "unknown", label: "未声明 — 只当绝对值读数，不声称它是幅度" },
  { value: "xy", label: "X / Y（带符号投影 — 需先把相位调到信号全落在 X 上）" },
  { value: "r_phi", label: "R / Φ（幅度 — 恒为正，随接近单调增大）" },
];
const BIAS_APPLIED_KEY = "bias_applied_to";
const BIAS_APPLIED_CHOICES = [
  { value: "unknown", label: "未声明 — AI 会明说自己不知道，不做能态归属" },
  { value: "sample", label: "样品 — 常规约定（正偏压探测样品空态）" },
  { value: "tip", label: "针尖 — 符号与常规整体反号" },
];
const INSTRUMENT_CHOICE_KEYS = [
  RETRACT_DIR_KEY, Z_EXTEND_SIGN_KEY, XY_COARSE_KEY, APPROACH_DMG_KEY, SCAN_PATH_KEY,
  VACUUM_MODE_KEY, BIAS_APPLIED_KEY, APPROACH_SUPERVISION_KEY, LOCKIN_FORM_KEY,
];
// Learned calibration keys — written by the runtime (set_calibration), shown
// read-only here. They MUST be echoed back on every save (whole-replace store).
//
// The qPlus pair is here for exactly that reason, not because it comes from
// set_calibration: the free-oscillation baseline is captured at runtime (after a
// verified retract) and is the DENOMINATOR of the amplitude crash criterion.
// Neither key was in any of the three lists, so editing any unrelated setting
// wiped the baseline and the detector fell back to "no_baseline" — reporting
// "cannot tell" forever while looking perfectly healthy (2026-07-31). Anything
// the runtime writes into instrument_profile belongs in one of these lists;
// tests/v2/unit/api/test_instrument_profile_frontend_parity.py enforces that.
const INSTRUMENT_CALIB_KEYS = [
  "didv_at_contact_v", "didv_cal_bias_v", "didv_cal_setpoint_a",
  "didv_cal_mod_amp_v", "didv_cal_updated_at",
  "qplus_amplitude_baseline", "qplus_amplitude_signal_index",
  // 倾斜响应矩阵 G（AutoTilt 实测标定）+ 条件数 + 时间戳。没有这个标定
  // AutoTilt 一律跳过，所以丢掉它 = 静默关掉自动调平。
  "tilt_cal_g11", "tilt_cal_g12", "tilt_cal_g21", "tilt_cal_g22",
  "tilt_cal_cond", "tilt_cal_updated_at",
  // qPlus 实测共振（AcquirePLLFreqSweep 写入）。与针尖登记里那支音叉的**标称**
  // f0/Q 是两回事：标称跟着针尖走，实测是当前装机这一支扫出来的，换针即清空。
  "qplus_f0_measured_hz", "qplus_q_measured", "qplus_fq_updated_at",
];
// What the「清除标定」button erases. A STRICT SUBSET of the above: echoing a key
// back on save and offering to delete it are different questions, and conflating
// them would let a button labelled "clear the dI/dV calibration" also throw away
// the qPlus baseline — which is captured by a different mechanism and re-earned
// only by a full retract.
const INSTRUMENT_CLEARABLE_CALIB_KEYS = [
  "didv_at_contact_v", "didv_cal_bias_v", "didv_cal_setpoint_a",
  "didv_cal_mod_amp_v", "didv_cal_updated_at",
];

function _fmtDidv(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e-3) return `${(v * 1e3).toFixed(3)} mV`;
  if (a >= 1e-6) return `${(v * 1e6).toFixed(3)} µV`;
  return `${v.toExponential(2)} V`;
}

function InstrumentProfileSection({
  settings,
  saving,
  save,
}: {
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const persisted = (settings.data?.instrument_profile ?? {}) as Record<string, unknown>;
  const numAt = (key: string): number | undefined =>
    typeof persisted[key] === "number" ? (persisted[key] as number) : undefined;
  const strAt = (key: string): string =>
    typeof persisted[key] === "string" ? (persisted[key] as string) : "";

  // Whole-replace guard: the store REPLACES instrument_profile wholesale. A save
  // must carry every set CONFIG field AND every learned CALIBRATION key, or a
  // config edit would wipe the auto-learned dI/dV calibration.
  const buildBase = (): Record<string, unknown> => {
    const next: Record<string, unknown> = {};
    for (const f of INSTRUMENT_NUM_FIELDS) {
      const v = numAt(f.key);
      if (typeof v === "number") next[f.key] = v;
    }
    // Every enumerated key, from ONE list — a choice missing here is silently
    // reset to its default the next time any other field is edited.
    for (const k of INSTRUMENT_CHOICE_KEYS) {
      const v = strAt(k);
      if (v) next[k] = v;
    }
    // Free-text fields need their OWN branch: the calibration loop below only
    // echoes numbers, so a model string routed through it would be dropped on
    // the next unrelated save — the exact shape of the qPlus baseline incident.
    for (const f of INSTRUMENT_TEXT_FIELDS) {
      const v = strAt(f.key);
      if (v) next[f.key] = v;
    }
    for (const k of INSTRUMENT_CALIB_KEYS) {
      if (typeof persisted[k] === "number") next[k] = persisted[k];
    }
    return next;
  };

  const commitNum = (key: string, value: number | null) => {
    const next = buildBase();
    if (value === null) delete next[key];
    else next[key] = value;
    save({ instrument_profile: next });
  };
  const commitChoice = (key: string, value: string) => {
    const next = buildBase();
    if (value) next[key] = value;
    else delete next[key];
    save({ instrument_profile: next });
  };
  const commitText = (key: string, value: string) => {
    const next = buildBase();
    const t = value.trim();
    if (t) next[key] = t;
    else delete next[key];
    save({ instrument_profile: next });
  };
  const clearCalibration = () => {
    const next = buildBase();
    for (const k of INSTRUMENT_CLEARABLE_CALIB_KEYS) delete next[k];
    save({ instrument_profile: next });
  };
  const resetAll = () => save({ instrument_profile: {} });

  const didv = numAt("didv_at_contact_v");
  const calBias = numAt("didv_cal_bias_v");
  const anySet = Object.keys(buildBase()).length > 0;

  const numRow = (key: string) => {
    const f = INSTRUMENT_NUM_FIELDS.find((x) => x.key === key)!;
    // 信号索引走下拉，不让人填裸数字。分派按 key，因为选项是从
    // 仪器读回来的**活**名单，不是这一页那三张静态 choice 表里的东西。
    if (SIGNAL_INDEX_KEYS.has(f.key)) {
      const v = numAt(f.key);
      return (
        <Field key={f.key} label={f.label} hint={f.hint}>
          <SignalIndexField
            value={v == null ? "" : String(v)}
            onCommit={(t) => {
              const trimmed = t.trim();
              if (trimmed === "") return commitNum(f.key, null);
              const n = Number.parseInt(trimmed, 10);
              if (Number.isNaN(n)) return;
              commitNum(f.key, Math.min(f.max, Math.max(f.min, n)));
            }}
            autoValue={SIGNAL_AUTO_VALUE[f.key]}
            placeholder={f.placeholder}
          />
        </Field>
      );
    }
    return (
      <ExperimentDefaultRow
        key={f.key}
        label={f.label}
        hint={f.hint}
        value={numAt(f.key)}
        min={f.min}
        max={f.max}
        step={f.step}
        placeholder={f.placeholder}
        disabled={saving}
        onCommit={(v) => commitNum(f.key, v)}
      />
    );
  };

  return (
    <div className="max-w-lg space-y-3">
      <p className="text-xs text-mast-muted">
        本机<b>装置事实</b>：换样品/关机<b>退针方向</b>、进针 <b>dI/dV</b>（lock-in）参数、以及自动学习的
        <b>到样品标定值</b>。退针会用粗动马达沿此方向分级后退并逐级回读 Z 压电自检（防方向搞反撞针）；
        进针可用 dI/dV 幅度判距离。<b>退针方向即使填错也有运行时 Z 自检兜底</b>——第一级只退 1 步就能发现——
        但请如实填写。
      </p>
      <p className="text-[11px] font-semibold text-mast-muted">退针（粗动马达）</p>
      <Field label="退针方向" hint="粗动马达「远离样品」的方向。多数 Nanonis 装置 Z+ = 远离；反向装置选 Z−。">
        <SelectField
          value={strAt(RETRACT_DIR_KEY) || "z+"}
          onChange={(v) => commitChoice(RETRACT_DIR_KEY, v)}
          options={RETRACT_DIR_CHOICES}
        />
      </Field>
      <Field label="压电伸长符号" hint="反馈让压电伸长（趋向样品）时 Z 读数是增大还是减小——退针自检用它判方向。">
        <SelectField
          value={strAt(Z_EXTEND_SIGN_KEY) || "+1"}
          onChange={(v) => commitChoice(Z_EXTEND_SIGN_KEY, v)}
          options={Z_EXTEND_SIGN_CHOICES}
        />
      </Field>
      {["z_recede_min_nm", "z_settle_timeout_s", "retract_total_steps", "retract_step_max"].map(numRow)}
      <p className="text-[11px] font-semibold text-mast-muted pt-1">进针 dI/dV（lock-in）</p>
      {["lockin_signal_index", "lockin_mod_amp_v", "lockin_mod_freq_hz",
        "lockin_x_signal_index", "lockin_y_signal_index",
        "approach_didv_engage_frac"].map(numRow)}
      <Field label="lock-in 解调形式" hint="上面那个信号索引指的是 X/Y 还是 R/Φ。R 是幅度、恒为正、随针尖接近单调增大；X 是对参考相位的带符号投影 —— 只有相位调到「信号全落在 X 上」时 |X| 才≈幅度，相位在接近途中转动（结电容随距离变）时 |X| 会穿零，于是「越近越大」中途掉下去，而那是相位问题不是针尖退开了。声明之后进针播报会用正确的符号并带上相位提醒。">
        <SelectField
          value={strAt(LOCKIN_FORM_KEY) || "unknown"}
          onChange={(v) => commitChoice(LOCKIN_FORM_KEY, v)}
          options={LOCKIN_FORM_CHOICES}
        />
      </Field>

      <p className="text-[11px] font-semibold text-mast-muted pt-1">进针参数组（Z 反馈 + 设定点）</p>
      <p className="text-xs text-mast-muted">
        进针时用的 Z 参数。填在这里之后，AI 只需要说「用 approach 这组参数」
        （<code>ApplyZCtrlPreset</code>），<b>具体数值由代码取出并写入硬件，全程不经过模型</b>，
        写完还会自动回读比对。<br />
        照抄 Nanonis 面板 <b>Z-Controller → Controller Adjustment</b> 上的两个数即可：
        面板显示的 <code>3.000p</code> 就是 <code>3e-12</code>，<code>180.000n</code> 就是{" "}
        <code>1.8e-7</code>。<br />
        <b>扫图那套参数不在这里</b>——它在下面的扫描档位表里，按图幅大小自动选档，
        <code>ScanAt</code> 每帧都会下发。同一个数只存一处，改了就到处都对。
      </p>
      {["approach_p_gain_m", "approach_i_gain_m_per_s", "approach_setpoint_a",
        "approach_steps_per_cycle", "approach_expected_steps"].map(numRow)}

      <p className="text-[11px] font-semibold text-mast-muted pt-1">表面预算与选点（扫描地图）</p>
      <p className="text-xs text-mast-muted">
        决定 AI 怎么选下一个扫描位置、什么时候建议换区。<b>没有 XY 粗动</b>的机器，
        当前压电范围内的表面就是「插拔样品之前你能看到的全部表面」——修针尖的污染会不可逆地
        吃掉可用面积，所以要省着用。避让半径是<b>你的经验值</b>，默认值只是起点。
      </p>
      <Field label="是否有 XY 粗动马达" hint="有粗动=一片表面用坏了可以换区；无粗动=只能靠插拔样品移位。也决定默认走哪条扫描路径。">
        <SelectField
          value={strAt(XY_COARSE_KEY) || "yes"}
          onChange={(v) => commitChoice(XY_COARSE_KEY, v)}
          options={XY_COARSE_CHOICES}
        />
      </Field>
      <Field label="进针是否留扎痕" hint="有些机器进针会在样品表面扎一下，好机器不会。「未知」按保守处理（当作会扎）——多让掉一点表面，比在扎痕上扫图划算。">
        <SelectField
          value={strAt(APPROACH_DMG_KEY) || "unknown"}
          onChange={(v) => commitChoice(APPROACH_DMG_KEY, v)}
          options={APPROACH_DMG_CHOICES}
        />
      </Field>
      <Field label="进针要不要有人在场" hint="上一项问的是会不会点伤表面（代价：一片表面）；这一项问的是这台机器的 Auto Approach 会不会扎针（代价：针，加之后几小时的修针）。它决定 AI 敢不敢在无人值守的自主流程里自己发起进针。「未知」按保守处理（＝必须有人在场）。注意：目前这一项只会注入给模型，不是硬性审批门。">
        <SelectField
          value={strAt(APPROACH_SUPERVISION_KEY) || "unknown"}
          onChange={(v) => commitChoice(APPROACH_SUPERVISION_KEY, v)}
          options={APPROACH_SUPERVISION_CHOICES}
        />
      </Field>
      <Field label="扫描选点策略" hint="中心优先：压电扫描管在量程中心蠕变最小，能换区的机器就该一直在中心扫、用坏了直接换区。外圈→内圈：不能换区时从边缘开始用，把最大的干净连续区域和低蠕变的中心留到最后。">
        <SelectField
          value={strAt(SCAN_PATH_KEY) || "auto"}
          onChange={(v) => commitChoice(SCAN_PATH_KEY, v)}
          options={SCAN_PATH_CHOICES}
        />
      </Field>
      {["avoid_radius_tip_shape_nm", "avoid_radius_pulse_nm", "avoid_radius_crash_nm",
        "avoid_radius_approach_nm", "scan_spacing_factor", "xy_motor_step_m"].map(numRow)}

      <p className="text-[11px] font-semibold text-mast-muted pt-1">横向粗动换区</p>
      <p className="text-xs text-mast-muted">
        一片表面用坏了、要粗动到新的一片时用的参数。<b>换区走 RelocateCoarseXY</b>，
        它会先用粗动马达退针清障（压电退针的 1–2 µm 余量不够）、逐级自检方向、
        确认电流归零后再分块横移。这几个默认值只是起点——
        <b>真值只能实机标</b>：移一次、扫一张图、看地貌是不是<b>完全</b>换了。
      </p>
      {["xy_prewithdraw_steps", "xy_move_chunk_steps", "xy_site_spacing_steps",
        "xy_axis_step_budget", "xy_step_uncertainty_frac"].map(numRow)}

      <p className="text-[11px] font-semibold text-mast-muted pt-1">多针尖判据的尺子</p>
      <p className="text-xs text-mast-muted">
        大图（≥100 nm）上会自动查「同一条台阶是不是被画了两遍」。判决用的是自相关
        重影强度（使用前须验证阈值适用性），这里这个数只影响报文里
        「几个台面能级」那一项 —— 但它是<b>尺子的零点</b>，填错会让台面统计整体平移。
      </p>
      {["au_step_pm"].map(numRow)}

      <p className="text-[11px] font-semibold text-mast-muted pt-1">粗动真空互锁</p>
      <p className="text-xs text-mast-muted">
        在<b>中间真空区</b>（约 0.1–1000 Pa = 1e-3–10 mbar，Paschen 极小值附近）给粗动压电加
        几百伏会<b>打火击穿叠堆</b>——抽气和放气途中正好穿过这个区间。MAST 只在拿到一个
        <b>量程内的有效读数</b>时才放行粗动；拿不到（没接真空计 / 计坏 / 超量程 / <b>触底</b> /
        读数过期）一律拒绝，此时需要用户在扫描地图页签署一次「当前气压安全」。
        <b>判据与规的型号无关</b>——型号相关的只有下面那两个量程数字。
        注意：<b>粗糙真空规（Pirani / 电容薄膜规）看不到那么低</b>，只装它的话粗动会一直被拒绝，
        这不是故障。
      </p>
      <Field label="互锁模式" hint="拿不到可信压强时怎么办。「关闭阻断」仍会计算并显示裁决——「我们选择不检查」要留得下痕迹。">
        <SelectField
          value={strAt(VACUUM_MODE_KEY) || "gauge_or_attest"}
          onChange={(v) => commitChoice(VACUUM_MODE_KEY, v)}
          options={VACUUM_MODE_CHOICES}
        />
      </Field>
      {["coarse_motion_max_pressure_pa", "vacuum_reading_max_age_s",
        "vacuum_gauge_full_scale_pa", "vacuum_gauge_min_pa"].map(numRow)}

      <p className="text-[11px] font-semibold text-mast-muted pt-1">扫描几何与调平</p>
      <p className="text-xs text-mast-muted">
        自动调平和扫描参数解算用的本机物理事实。<b>倾斜响应矩阵 G</b> 由 AutoTilt 实测标定；
        <b>没有它 AutoTilt 一律跳过</b>——绝不带着猜来的符号去动硬件。
      </p>
      {["z_range_m", "tilt_limit_deg", "v_tip_max_m_s", "z_noise_floor_m"].map(numRow)}

      <p className="text-[11px] font-semibold text-mast-muted pt-1">信号链（前放 / 偏压极性）</p>
      <p className="text-xs text-mast-muted">
        这两项决定 AI 怎么<b>解释</b>数据，而不是怎么操作仪器。
        <b>偏压加在哪一侧</b>决定正偏压对应样品的空态还是占据态——弄反了整套能态归属就是反的，
        而这种错在数据里完全看不出来。<b>前放增益</b>是电流的绝对标度，
        Nanonis 只给得到「索引」，索引对应多少 V/A 只能在这里登记。
      </p>
      <Field
        label="偏压加在哪一侧"
        hint="常规约定是加在样品上：正样品偏压时电子由针尖隧穿进样品空态。加在针尖上则符号整体反号。留「未声明」时 AI 会明确告诉你它不知道，而不是按最常见的约定替你断言。"
      >
        <SelectField
          value={strAt(BIAS_APPLIED_KEY) || "unknown"}
          onChange={(v) => commitChoice(BIAS_APPLIED_KEY, v)}
          options={BIAS_APPLIED_CHOICES}
        />
      </Field>
      {INSTRUMENT_TEXT_FIELDS.map((f) => (
        <Field key={f.key} label={f.label} hint={f.hint}>
          <TextRow
            value={strAt(f.key)}
            placeholder={f.placeholder}
            disabled={saving}
            onCommit={(v) => commitText(f.key, v)}
          />
        </Field>
      ))}
      {["preamp_gain_v_per_a", "preamp_full_scale_a"].map(numRow)}

      <div className="rounded border border-mast-border bg-mast-panel px-3 py-2 text-xs text-mast-muted">
        <div className="font-semibold text-mast-text">到样品 dI/dV 标定值（自动学习）</div>
        {typeof didv === "number" ? (
          <div className="mt-1">
            当前 ≈ <b className="text-mast-text">{_fmtDidv(didv)}</b>
            {typeof calBias === "number" ? <>（绑定 bias {calBias.toPrecision(3)} V）</> : null}
            ——每轮成功进针自动 EWMA 更新。
          </div>
        ) : (
          <div className="mt-1">尚未标定：首次成功进针后自动记录，供下轮判距离用。</div>
        )}
        {typeof didv === "number" && (
          <button
            type="button"
            onClick={clearCalibration}
            disabled={saving}
            className="mt-2 rounded border border-mast-border bg-mast-bg px-2 py-1 text-[11px] text-mast-muted hover:text-mast-text disabled:opacity-50"
          >
            清除学习标定
          </button>
        )}
      </div>
      <div className="flex items-center gap-3 pt-1">
        <button
          type="button"
          onClick={resetAll}
          disabled={saving || !anySet}
          className="rounded border border-mast-border bg-mast-panel px-3 py-1 text-xs text-mast-muted hover:text-mast-text disabled:opacity-50"
        >
          全部恢复默认（含标定）
        </button>
        {!anySet && <span className="text-xs text-mast-muted">（全部为默认值）</span>}
      </div>
      {saving && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}

// A numeric preference row: like ThresholdRow but the value may be UNSET (blank),
// and clearing the field commits null (removes the preference). Enter/blur commits.
function ExperimentDefaultRow({
  label, hint, value, min, max, step, placeholder, disabled, onCommit,
}: {
  label: string; hint: string; value: number | undefined; min: number; max: number;
  step: string; placeholder: string; disabled: boolean; onCommit: (v: number | null) => void;
}) {
  const shown = typeof value === "number" ? _fmtThreshold(value) : "";
  const [text, setText] = useState(shown);
  useEffect(() => {
    setText(shown);
  }, [shown]);

  const commit = () => {
    const t = text.trim();
    if (t === "") {
      if (typeof value === "number") onCommit(null); // cleared → drop the preference
      return;
    }
    const n = Number.parseFloat(t);
    if (Number.isNaN(n)) {
      setText(shown); // revert junk
      return;
    }
    const clamped = Math.min(max, Math.max(min, n));
    setText(_fmtThreshold(clamped));
    if (clamped !== value) onCommit(clamped);
  };

  return (
    <Field label={label} hint={hint}>
      <input
        type="text"
        inputMode="decimal"
        value={text}
        step={step}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur();
        }}
        className="w-28 rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
      />
    </Field>
  );
}

// A free-text row (型号一类). Enter/blur commits; blank clears the value. The
// caller supplies the <Field> wrapper so it can be reused inside a mapped list.
function TextRow({
  value, placeholder, disabled, onCommit,
}: {
  value: string; placeholder: string; disabled: boolean; onCommit: (v: string) => void;
}) {
  const [text, setText] = useState(value);
  useEffect(() => {
    setText(value);
  }, [value]);

  const commit = () => {
    const t = text.trim();
    if (t !== value) onCommit(t);
    else setText(value);
  };

  return (
    <input
      type="text"
      value={text}
      disabled={disabled}
      placeholder={placeholder}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur();
      }}
      className="w-56 rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text disabled:opacity-50"
    />
  );
}

function VisionThresholdsSection({
  settings,
  saving,
  save,
}: {
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const persisted = (settings.data?.vision_thresholds ?? {}) as Record<string, number>;
  const eff = (key: string, def: number) =>
    typeof persisted[key] === "number" ? persisted[key] : def;

  // Whole-replace guard: the store REPLACES the vision_thresholds dict wholesale
  // (it does not merge), so a save must POST the effective value of EVERY knob
  // plus the one change — otherwise the other valves would be wiped.
  const commit = (key: string, value: number) => {
    const next: Record<string, number> = { ...persisted };
    for (const k of VISION_THRESHOLD_KNOBS) next[k.key] = eff(k.key, k.def);
    next[key] = value;
    save({ vision_thresholds: next });
  };

  // Empty dict → backend from_mapping falls back to all defaults (a true reset).
  const resetDefaults = () => save({ vision_thresholds: {} });

  return (
    <div className="max-w-lg space-y-3">
      <p className="text-xs text-mast-muted">
        视觉模型「针尖是否合格」的判别阈值。<b>模型权重不变，只调判别的宽严。</b>
        默认偏严会把不少可用针尖判坏、触发不必要的针尖处理；
        <b>调低门槛 / 调高容忍 = 更宽松</b>。改动即时生效（无需重载模型）。
        C（图像分割）不参与好坏判定，故无阀门。
      </p>
      {VISION_THRESHOLD_KNOBS.map((k) => (
        <ThresholdRow
          key={k.key}
          label={k.label}
          hint={k.hint}
          value={eff(k.key, k.def)}
          min={k.min}
          max={k.max}
          step={k.step}
          disabled={saving}
          onCommit={(v) => commit(k.key, v)}
        />
      ))}
      <div className="pt-1">
        <button
          type="button"
          onClick={resetDefaults}
          disabled={saving}
          className="rounded border border-mast-border bg-mast-panel px-3 py-1 text-xs text-mast-muted hover:text-mast-text disabled:opacity-50"
        >
          恢复默认
        </button>
      </div>
      {saving && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}

function ClassicalThresholdsSection({
  settings,
  saving,
  save,
}: {
  settings: ReturnType<typeof useSettings>;
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const persisted = (settings.data?.classical_thresholds ?? {}) as Record<string, number>;
  const eff = (key: string, def: number) =>
    typeof persisted[key] === "number" ? persisted[key] : def;

  // Whole-replace guard: the store REPLACES the classical_thresholds dict
  // wholesale, so a save must POST every knob's effective value plus the change.
  const commit = (key: string, value: number) => {
    const next: Record<string, number> = { ...persisted };
    for (const k of CLASSICAL_THRESHOLD_KNOBS) next[k.key] = eff(k.key, k.def);
    next[key] = value;
    save({ classical_thresholds: next });
  };

  const resetDefaults = () => save({ classical_thresholds: {} });

  return (
    <div className="max-w-lg space-y-3">
      <p className="text-xs text-mast-muted">
        <b>免模型</b>的经典针尖判别工具（双针尖 / 正反扫描不稳 / 中途换针 / 反馈振荡 / I(z) 谱）的阈值，
        可<b>按仪器/样品重标定</b>。改动即时生效（无需重载任何模型）。这些是深度模型判定的<b>可解释旁路 + 性能地板</b>。
      </p>
      {CLASSICAL_THRESHOLD_KNOBS.map((k) => (
        <ThresholdRow
          key={k.key}
          label={k.label}
          hint={k.hint}
          value={eff(k.key, k.def)}
          min={k.min}
          max={k.max}
          step={k.step}
          disabled={saving}
          onCommit={(v) => commit(k.key, v)}
        />
      ))}
      <div className="pt-1">
        <button
          type="button"
          onClick={resetDefaults}
          disabled={saving}
          className="rounded border border-mast-border bg-mast-panel px-3 py-1 text-xs text-mast-muted hover:text-mast-text disabled:opacity-50"
        >
          恢复默认
        </button>
      </div>
      {saving && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}

// ── 硬件模块（已移至 高级）─────────────────────────────────────────────────────
// The toggles moved to 高级 → 系统/全局 → 能力开关 on 2026-07-13, behind the admin
// PIN. Switching a module ON hands the agent DANGEROUS skills — a laser, an RF
// amplifier, probes that can collide — which is not the same class of action as
// changing the font size, and should not sit next to it.
//
// A signpost stays here rather than nothing: a control that silently vanishes is
// worse than one that tells you where it went.
function HardwareModulesMoved({
  hwModules,
}: {
  hwModules: ReturnType<typeof useHardwareModules>;
}) {
  const n = hwModules.data?.modules?.length ?? 0;
  const on = hwModules.data?.enabled_count ?? 0;
  const gated = hwModules.data?.gated_skill_count ?? 0;
  return (
    <div className="max-w-lg space-y-2 text-sm text-mast-muted">
      <p>
        Nanonis 的选装模块（KPFM / 多探针 / 高分辨示波器 / 高速扫描器 / 射频 / 激光…），
        授权都有、硬件不一定装，<b>默认全关</b>。
      </p>
      <p>
        开关已移到 <b className="text-mast-text">高级 → 系统 / 全局 → 能力开关</b>，需要管理 PIN。
        理由：打开一个模块就等于把 <b className="text-mast-text">DANGEROUS skill</b> 交给 agent
        （激光、射频功放、会互撞的多探针）——那和「改个字号」不是同一类操作。
      </p>
      <p className="text-xs">
        当前：<b className="text-mast-text">{on}</b> / {n} 个模块启用；
        <b className="text-mast-text">{gated}</b> 个 skill 因模块关闭而未挂载。
      </p>
    </div>
  );
}

function OtherSection() {
  return (
    <div className="space-y-3 text-sm text-mast-muted">
      <p>
        局域网访问 (LAN)、自动更新 / 推送服务器等进程级设置在 <b>桌面启动器 (Tk Launcher)</b> 中配置，
        不在本 Web 界面内。请在启动器窗口调整。
      </p>
      <p>
        <i>
          系统配置编辑器（安全限值 / 技能元数据 / 知识库 / 技能指导 / 百科配置 / 复杂技能）以及
          硬件连接（Nanonis TCP / 真空计 / 温度计）在独立的「高级管理」标签内，需启动器设置的 PIN 解锁。
        </i>
      </p>
    </div>
  );
}

// ── 模型能力表（只读） ─────────────────────────────────────────────────────────
function CapabilitiesSection({ models }: { models: ReturnType<typeof useModels> }) {
  if (models.isPending) return <Spinner />;
  if (models.isError) return <ErrorNote error={models.error} />;
  if (!models.data?.models?.length) return <EmptyNote label="无可用模型" />;

  return (
    <div className="space-y-3">
      <p className="text-xs text-mast-muted">
        当前模型能力（只读）。思考列：可调 = Anthropic / MiniMax 档位真正生效；固定 = 推理模型服务端固定；
        无 = 不传 thinking 参数。
      </p>
      <ModelCapabilityTable models={models.data.models} />
    </div>
  );
}
