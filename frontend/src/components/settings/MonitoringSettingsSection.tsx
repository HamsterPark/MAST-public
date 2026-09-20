import type { SettingsPatch } from "@/lib/settingsWrite";
import { KnobCatalogSection } from "./KnobCatalogSection";

// 设置 → 电流监控 — the always-on tunnelling-current monitor's knobs.
// Rows come from `GET /api/monitoring/config`'s catalogue; the rendering, the
// whole-replace guard and the reset button live in KnobCatalogSection, shared
// with 环境历史.

export function MonitoringSettingsSection({
  settings,
  saving,
  save,
}: {
  settings: { data?: Record<string, unknown> | null };
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  return (
    <KnobCatalogSection
      configPath="/api/monitoring/config"
      queryKey={["monitoring", "config"]}
      settingsKey="current_monitor"
      settings={settings}
      saving={saving}
      save={save}
      unavailableNote={
        "监控模块未装载，读不到旋钮目录，因此不提供编辑入口——" +
        "这里宁可什么都不显示，也不给一份改了不生效的表单。"
      }
      blurb={
        <>
          隧道电流的<b>常开采集</b>：段长、保留策略与告警阈值。改动<b>即时生效</b>，
          不需要重启采集。阈值随仪器 / 前置放大器 / 样品变化，出厂值只是起点——
          噪声阈按你这台机器的真实底噪重标定才有意义。
        </>
      }
    />
  );
}
