import type { SettingsPatch } from "@/lib/settingsWrite";
import { KnobCatalogSection } from "./KnobCatalogSection";

// 设置 → 环境历史 — how much of the environment record is kept, and how coarse.
//
// Two knobs here deserve care, and the blurb says so in the operator's terms:
//   · eh_raw_keep_days is the ONLY setting in this recorder that deletes
//     anything. It drops per-reading detail older than N days from the
//     database. The permanent statistics buckets, the alarm rows and the
//     experiment-folder CSVs are all untouched, so lowering it costs
//     resolution, never the record.
//   · eh_z_enabled is the only one that talks to the instrument at all.

export function EnvHistorySettingsSection({
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
      configPath="/api/env-history/config"
      queryKey={["env-history", "config"]}
      settingsKey="env_history"
      settings={settings}
      saving={saving}
      save={save}
      unavailableNote={
        "环境历史模块未装载，读不到旋钮目录，因此不提供编辑入口——" +
        "宁可什么都不显示，也不给一份改了不生效的表单。"
      }
      blurb={
        <>
          后台<b>静默</b>记录温度 / 真空 / 液氦 / 磁场 / 隧道电流，并定期存一条噪声谱快照。
          改动<b>即时生效</b>。统计桶与噪声谱<b>永久保留</b>；
          「原始读数保留」只影响逐条明细，超期后仍然能看统计曲线，
          <b>告警读数永远不删</b>，实验文件夹里的 CSV 也从不被触碰。
          「记录 Z 噪声谱」是这里唯一会占用仪器通信的开关。
        </>
      }
    />
  );
}
