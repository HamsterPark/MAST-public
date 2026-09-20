import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";
import { applyTheme, useUiStore } from "@/store";
import { useAutonomyMode } from "@/hooks/useAutonomyMode";
import { useOpenFetchRequests } from "@/hooks/useOpenFetchRequests";
import { useOpenAgentRequests } from "@/hooks/useOpenAgentRequests";
import { TopBar } from "@/components/shell/TopBar";
import { NoSampleBanner } from "@/components/scope/ScopeControls";
import { SetupBanner } from "@/components/shell/SetupBanner";
import { RightPanel } from "@/components/shell/RightPanel";
import { FeedbackFloat } from "@/components/chat/FeedbackFloat";
import { TABS } from "@/lib/nav";

const RAIL_COLLAPSED_KEY = "mast.rightRail.collapsed";

/** A count on a nav tab: N things are waiting for the operator there. */
function PendingBadge({ n, label }: { n: number; label: string }) {
  return (
    <span
      className="ml-1.5 inline-flex min-w-[18px] items-center justify-center
                 rounded-full bg-mast-accent px-1.5 py-0.5 text-[11px]
                 font-semibold leading-none text-white"
      title={label}
      aria-label={label}
    >
      {n > 99 ? "99+" : n}
    </span>
  );
}

// Shell faithful to the old Gradio app.py build_ui():
//   header (TopBar) + main-grid[ left tabs column (scale 5) | right panel (scale 2) ]
// The 12 horizontal top-tabs mirror the old gr.Tabs order exactly.

// TABS 住在 lib/nav.ts —— 那是一份纯数据，而「某一页没在导航里」这种缺失
// 类型检查、测试和截图全都看不见（路由照样解析、组件照样渲染），只能靠断言。
// 见那个文件顶部：2026-08-04「初始化页只能进去一次」就是这个形状。

export default function AppLayout() {
  const theme = useUiStore((s) => s.theme);
  const activeConversationId = useUiStore((s) => s.activeConversationId);
  // Global operating mode → a whole-app colored border: safe=green, semi=amber,
  // auto=none. Rendered as an INSET ring (box-shadow) so it never shifts layout
  // or clips at the viewport edge, and stays put across navigation (AppLayout
  // never unmounts).
  const { mode } = useAutonomyMode();
  // Unresolved fetch requests → a count on the 文献库 tab. The literature agent
  // tells itself the operator "将在文献库标签处理", but nothing ever said a
  // request had arrived, so three of them sat untouched all session
  // (). Degrades to 0 = no badge.
  const openFetchRequests = useOpenFetchRequests();
  // Same for the 心愿单 tab, where the OTHER ask-the-operator channel waits.
  // It had no badge at all, so an agent asking the operator to do something
  // physical had no way of being noticed — the same invisibility the fetch board
  // suffered, on the channel that otherwise has the better plumbing.
  const openAgentRequests = useOpenAgentRequests();
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  // ITEM 3 — right rail collapse, persisted in localStorage. A thin vertical
  // handle at the rail edge toggles it; collapsed shows a slim strip + chevron.
  const [railCollapsed, setRailCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem(RAIL_COLLAPSED_KEY) === "1";
    } catch {
      return false;
    }
  });
  const toggleRail = () => {
    setRailCollapsed((c) => {
      const next = !c;
      try {
        localStorage.setItem(RAIL_COLLAPSED_KEY, next ? "1" : "0");
      } catch {
        /* private mode / no storage — collapse still works for the session */
      }
      return next;
    });
  };

  return (
    <div className="flex h-screen flex-col bg-mast-bg text-mast-text">
      <TopBar />
      {/* 无样品时的常驻提示：产数据的操作被拦，但对话/查询照常。
          只告知，不困住 —— 它不阻塞任何交互。 */}
      <NoSampleBanner />
      {/* 新仪器初始化：必填项没答完 / 硬件指纹变了 / 从没走过初始化 → 常驻横幅。
          刻意不是模态框 —— 一个必须填完才能关的全屏遮罩，在用户想按急停的
          时候就是灾难（项目规约「UI 绝不冻结」）。 */}
      <SetupBanner />
      <div className="flex min-h-0 flex-1">
        {/* left: top-tabs + content */}
        <div className="flex min-w-0 flex-1 flex-col">
          {/* 2026-08-06:17 项合并成 10 项（「大标签太多」）。
              `flex-wrap` 留着 —— 窄窗口下换行仍然好过横向滚动或者把项目藏进
              一个下拉里（项目规约：tab 溢出下拉禁用）。 */}
          <nav className="flex flex-wrap items-center gap-x-0.5 border-b border-mast-border bg-mast-panel px-3">
            {TABS.map((t) => (
              <NavLink
                key={t.to}
                to={t.to}
                end={t.end}
                className={({ isActive }) =>
                  clsx(
                    "-mb-px whitespace-nowrap border-b-2 px-[13px] py-[11px] text-[13px]",
                    isActive
                      ? "border-mast-accent bg-mast-accent-soft font-semibold text-mast-accent"
                      : "border-transparent text-mast-muted hover:text-mast-text",
                  )
                }
              >
                {t.label}
                {t.to === "/literature" && openFetchRequests > 0 && (
                  <PendingBadge
                    n={openFetchRequests}
                    label={`${openFetchRequests} 条取文请求等待处理`}
                  />
                )}
                {t.to === "/wishlist" && openAgentRequests > 0 && (
                  <PendingBadge
                    n={openAgentRequests}
                    label={`${openAgentRequests} 条智能体请求等待你处理`}
                  />
                )}
              </NavLink>
            ))}
          </nav>
          <main className="min-h-0 flex-1 overflow-auto">
            {/* Full-width content fills the left column (old Gradio scale=5 column
                had no centred max-width); pages cap their own width where needed. */}
            <div className="p-5">
              <Outlet />
            </div>
          </main>
        </div>
        {/* right: persistent instrument/environment/experiment/settings/system rail.
            ITEM 3 — collapsible. The thin handle sits at the rail's left edge;
            when collapsed only a slim strip with an expand chevron remains. */}
        {railCollapsed ? (
          <div className="hidden w-8 shrink-0 flex-col items-center border-l border-mast-border bg-mast-panel lg:flex">
            <button
              type="button"
              onClick={toggleRail}
              title="展开侧栏"
              aria-label="展开侧栏"
              aria-expanded={false}
              className="flex h-full w-full flex-col items-center gap-2 py-3 text-mast-muted hover:bg-mast-bg hover:text-mast-text"
            >
              <span className="text-sm leading-none">‹</span>
              <span
                className="mt-1 text-[10px] font-semibold uppercase tracking-wider"
                style={{ writingMode: "vertical-rl" }}
              >
                状态栏
              </span>
            </button>
          </div>
        ) : (
          <div className="hidden lg:flex">
            <button
              type="button"
              onClick={toggleRail}
              title="收起侧栏"
              aria-label="收起侧栏"
              aria-expanded={true}
              className="w-3 shrink-0 cursor-pointer border-l border-mast-border bg-mast-panel text-mast-muted hover:bg-mast-bg hover:text-mast-text"
            >
              <span className="text-xs leading-none">›</span>
            </button>
            <RightPanel />
          </div>
        )}
      </div>
      {/* 对话反馈悬浮窗 — mounted once at the shell so it shows on EVERY tab and
          keeps its drag position across navigation (AppLayout never unmounts).
          It portals to <body>, so placement here doesn't affect layout. Its
          conversationId comes from the UI store, set by ChatPage. */}
      <FeedbackFloat conversationId={activeConversationId} />
      {/* Global operating-mode frame — a FULL-VIEWPORT fixed border overlaying the
          whole screen (safe=green, semi=amber). fixed+inset-0+high z + no
          pointer events, so no panel/background can paint over it (the old
          ring-inset got covered by the TopBar/panels) and it never clips or
          scrolls. */}
      {mode === "safe" && (
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[9999] border-4 border-mast-auto"
        />
      )}
      {mode === "semi" && (
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[9999] border-4 border-mast-warn"
        />
      )}
    </div>
  );
}
