import type { ReactElement } from "react";
import { createBrowserRouter } from "react-router-dom";
import AppLayout from "@/layout/AppLayout";
import { GroupIndex, LegacyRedirect, TabGroup } from "@/layout/TabGroup";
import { LEGACY_REDIRECTS, TABS, sectionIds, sectionPath } from "@/lib/nav";
import ChatPage from "@/pages/ChatPage";
import AgentsPage from "@/pages/AgentsPage";
import QaPage from "@/pages/QaPage";
import SkillsPage from "@/pages/SkillsPage";
import BuilderPage from "@/pages/BuilderPage";
import CognitionPage from "@/pages/CognitionPage";
import LiteraturePage from "@/pages/LiteraturePage";
import RecordsPage from "@/pages/RecordsPage";
import GalleryPage from "@/pages/GalleryPage";
import ConductPage from "@/pages/ConductPage";
import WishlistPage from "@/pages/WishlistPage";
import AdminPage from "@/pages/AdminPage";
import SettingsPage from "@/pages/SettingsPage";
import SetupPage from "@/pages/SetupPage";
import ExperimentalPage from "@/pages/ExperimentalPage";
import DashboardPage from "@/pages/DashboardPage";
import VisionPage from "@/pages/VisionPage";
import OpticsPage from "@/pages/OpticsPage";
import UsagePage from "@/pages/UsagePage";
import MonitoringPage from "@/pages/MonitoringPage";
import EnvHistoryPage from "@/pages/EnvHistoryPage";

// ── 2026-08-06:17 个大标签合并成 10 个 ─────────────────────
//
// 合并的是**导航**，不是页面：下面每一个 element 都还是原来那个组件，一行没动，
// 只是从顶栏的一项变成某个大标签下的一段。
//
// **路由结构由 lib/nav.ts 那张表生成，不是手写第二遍。** 这一条是被自己的测试
// 逼出来的：第一版是手写嵌套字面量 + 一个按缩进解析 TSX 的测试去对账，而那个
// 解析器给出了一棵**看起来很有道理的错误的树**（它把 `/wishlist` 认成了 `/skills`
// 的父节点，因为多行对象的 `path:` 缩进比单行对象深一级）。一个用正则去重建
// 语法树的校验，验的是它自己的猜测。所以改成：路径只有一份真源，代码从它派生，
// 测试只需要对账「每一段是不是都配了组件」。
//
// 二级页做成真的路由段而不是页面内的 useState，唯一但足够的理由是**书签**:
// `/settings` `/admin` 这类地址在用户浏览器里存了几个月。旧地址一条不落地
// 重定向（LEGACY_REDIRECTS），并且**带着 query 一起** —— 监控页的 `?seg=` 深链
// 丢了不会报错，只会静静地把「那个具体段落」变成「监控页首页」。

/**
 * 每一个二级页的完整路径 → 渲染它的组件。
 *
 * key 是 `sectionPath()` 算出来的完整路径，不是段名 —— 段名会重（`/skills/library`
 * 和某个未来的 `/records/library`），而重了之后后写的那个会静默覆盖先写的。
 *
 * 导出给测试：nav.ts 里加了一段却忘了在这里配组件，症状是点进去一片空白，
 * 而路由照样解析、类型检查照样绿。
 */
export const SECTION_ELEMENTS: Record<string, ReactElement> = {
  "/skills/library": <SkillsPage />,
  "/skills/builder": <BuilderPage />,

  "/records/log": <RecordsPage />,
  "/records/gallery": <GalleryPage />,
  "/records/conduct": <ConductPage />,
  "/records/memory": <CognitionPage />,

  // 仪器初始化在这一组里**仍然有自己的位置**。2026-08-04 把它请进导航的理由
  // 一条都没变：横幅盖完成戳之后按设计消失，而设置页里的那个按钮不是导航。
  // 退针方向是装完之后才量、量完才改的值，它必须常驻可达 —— 只是那个常驻的
  // 位置现在在「设置」下面一层，而不是顶栏。
  "/settings/general": <SettingsPage />,
  "/settings/setup": <SetupPage />,
  "/settings/admin": <AdminPage />,
  "/settings/usage": <UsagePage />,

  "/monitoring/current": <MonitoringPage />,
  "/monitoring/env": <EnvHistoryPage />,

  "/experimental/tools": <ExperimentalPage />,
  "/experimental/optics": <OpticsPage />,
};

/** 没有二级页的大标签 / 刻意不做 tab 的页面。 */
const FLAT_ELEMENTS: Record<string, ReactElement> = {
  "/agents": <AgentsPage />,
  "/qa": <QaPage />,
  "/literature": <LiteraturePage />,
  "/wishlist": <WishlistPage />,
  // reachable, non-tab (see INTENTIONALLY_UNLISTED):
  "/dashboard": <DashboardPage />,
  "/vision": <VisionPage />,
  "/chat": <ChatPage />,
};

const strip = (p: string) => p.replace(/^\//, "");

/** 有 sections 的大标签 → 一个带 children 的组路由。全部从 TABS 派生。 */
const groupRoutes = TABS.filter((t) => t.sections?.length).map((t) => ({
  path: strip(t.to),
  element: <TabGroup to={t.to} />,
  children: [
    { index: true, element: <GroupIndex to={t.to} /> },
    ...sectionIds(t).map((seg) => ({
      path: seg,
      element: SECTION_ELEMENTS[sectionPath(t, seg)],
    })),
  ],
}));

/** 合并之前的顶栏地址。书签、别人发的链接、文档里写的地址都还能用。 */
const legacyRoutes = Object.entries(LEGACY_REDIRECTS).map(([from, to]) => ({
  path: strip(from),
  element: <LegacyRedirect to={to} />,
}));

const flatRoutes = Object.entries(FLAT_ELEMENTS).map(([p, element]) => ({
  path: strip(p),
  element,
}));

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <ChatPage /> },
      ...flatRoutes,
      ...groupRoutes,
      ...legacyRoutes,
    ],
  },
]);
