import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { CONDUCT_SETTINGS_TITLE, directorStateNote, badgeTone } from "@/lib/conduct";
import type { SettingsPatch } from "@/lib/settingsWrite";
import { KnobCatalogSection } from "./KnobCatalogSection";

// 设置 → conduct 指挥线程（2026-08-27）
//
// 在这一节存在之前，`cd_enabled` 只能靠直接 POST /api/settings 打开：后端读、写、
// 类型三侧早就通了，前端一个入口都没有，而 ConductPage 的提示还指向「设置 →
// 常规设置」这个不存在的地方。真机三步走（有人陪跑 → 半无人 → 无人过夜）因此
// 没法从界面开工。
//
// 两个旋钮值得单独说一句，措辞照用户的话：
//   · cd_enabled —— **默认关**，而且关着的时候线程不建、表不建，逐字节等于这个
//     功能落地之前。它不是「暂停」。
//   · cd_autonomy —— 三档管的是**谁能点头**，不是**什么可以做**。三档下安全包络
//     一个数都不放宽；放开的只是「夜里三点没有人在场」那道在场约束。

export function ConductSettingsSection({
  settings,
  saving,
  save,
}: {
  settings: { data?: Record<string, unknown> | null };
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  // 与 ConductPage 用**同一个** queryKey：那边保存后失效这里、这里保存后失效
  // 那边，两个页面不会各自记着一份过期的 enabled。
  const config = useQuery({
    queryKey: ["conduct", "config"],
    staleTime: 5_000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/conducts/config");
      if (error) throw error;
      return data as { enabled?: boolean; director_running?: boolean };
    },
  });

  const note = directorStateNote(
    Boolean(config.data?.enabled),
    Boolean(config.data?.director_running),
  );

  return (
    <>
      {config.data && (
        <p className={"mb-3 rounded-mast-ctl px-3 py-2 text-sm " + badgeTone(note.tone)}>
          {note.text}
        </p>
      )}
      <KnobCatalogSection
        configPath="/api/conducts/config"
        queryKey={["conduct", "config"]}
        settingsKey="conduct"
        settings={settings}
        saving={saving}
        save={save}
        unavailableNote={
          "conduct 模块未装载，读不到旋钮目录，因此不提供编辑入口——" +
          "宁可什么都不显示，也不给一份改了不生效的表单。"
        }
        blurb={
          <>
            多天执行编排（hour~day）的常驻线程：声明式阶段状态机 + 闸门 + 断点续跑。
            <b>出厂默认关</b>——关着的时候线程不建、表不建，逐字节等于这个功能落地之前；
            <b>关掉它不会关掉读端点</b>，已有 conduct 的状态照常读得到、中止照常按得下。
            「自主度」三档管的是<b>谁能点这个头</b>，不是<b>什么可以做</b>：
            三档下安全包络一个数都不放宽，放开的只是「夜里三点没有人在场」那道在场约束。
          </>
        }
      />
    </>
  );
}

export { CONDUCT_SETTINGS_TITLE };
