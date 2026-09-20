// ════════════════════════════════════════════════════════════════════════════
// TabGroup — 一个合并后的大标签的外壳。
//
// 顶栏标签太多、容易挤到换行。17 个挤到换行的顶栏合并成 10 个之后，五个大标签底下各自装着
// 原来的两到四页。这个组件就是那层壳：一条二级 tab 条 + <Outlet/>。
//
// **二级页是真的路由段**，不是页面内的 useState。理由只有一个，但它足够：
// 书签。`/settings`、`/admin` 这类地址在用户的浏览器里存了几个月，而一个存在
// localStorage 里的子页选择回答不了「这个链接指向哪一页」—— 分享出去的链接、
// 别人机器上打开的链接，全都会落在默认那一段。
//
// 记忆仍然有，只是只管一件事：**光敲大标签**（`/settings`）时去哪一段。
// 那是 #32 的那条纪律在这一层的延续，用的也是同一个 useStickyTab。
// ════════════════════════════════════════════════════════════════════════════

import { useEffect } from "react";
import { Navigate, Outlet, useLocation, useNavigate } from "react-router-dom";
import { SubTabs } from "@/components/controls";
import { useStickyTab } from "@/hooks/useStickyTab";
import { findTab, sectionIds, sectionPath, type NavTab } from "@/lib/nav";

/**
 * 当前 URL 落在这个大标签的哪一段上。
 *
 * 从 pathname 现算，**不**用 `useParams()`：子路由写的是 `path: "library"` 这样的
 * 字面量，不是 `:section`，所以那个参数根本不存在 —— 而 `useParams().section`
 * 读一个不存在的参数得到的是 `undefined`，不是错误。二级 tab 条会因此永远没有
 * 高亮段，看起来像样式没写对。
 */
function currentSection(pathname: string, to: string, ids: string[]): string | null {
  const base = to === "/" ? "" : to;
  if (!pathname.startsWith(`${base}/`)) return null;
  const seg = pathname.slice(base.length + 1).split("/")[0] ?? "";
  return ids.includes(seg) ? seg : null;
}

/**
 * 一个大标签的壳。`to` 是它在 {@link TABS} 里的 `to`。
 *
 * 路由形状（见 router.tsx）::
 *
 *     { path: "settings", element: <TabGroup to="/settings" />, children: [
 *         { index: true, element: <GroupIndex to="/settings" /> },
 *         { path: "general", element: <SettingsPage /> },
 *         … ] }
 */
export function TabGroup({ to }: { to: string }) {
  const tab = findTab(to);
  const nav = useNavigate();
  const { pathname } = useLocation();
  const ids = tab ? sectionIds(tab) : [];
  const section = currentSection(pathname, to, ids);
  const [, remember] = useStickyTab(`group.${to}`, ids, ids[0] ?? "");

  // 记下用户**实际停在**哪一段,好让下次光敲大标签时回到这里。
  // 写在 effect 里而不是 onChange 里:直接敲一个二级地址进来(书签、别人发的链接)
  // 也该被记住 —— 只记点击的话,「我明明上次在高级管理」会在他是从书签进来的
  // 那些次里落空,而那是最难复现的一类抱怨。
  useEffect(() => {
    if (section) remember(section);
  }, [section, remember]);

  if (!tab || !tab.sections?.length) {
    // 导航表里没有这一项、或者它根本没有二级页 —— 不该走到这里。别白屏，
    // 也别抛（项目规约「UI 绝不冻结」）：把人送回首页。
    return <Navigate to="/" replace />;
  }

  return (
    <div className="space-y-4">
      <SubTabs
        tabs={tab.sections.map((s) => ({ id: s.seg, label: s.label }))}
        value={section ?? ""}
        onChange={(seg) => nav(sectionPath(tab, seg))}
      />
      <Outlet />
    </div>
  );
}

/**
 * 光敲大标签时去哪一段：上次停的那一段，没有记忆就是第一段。
 *
 * `replace` —— 不在历史里留下这个中转地址。留着的话，从二级页按「后退」会回到
 * 大标签根，而它又立刻把人送回来，于是后退键失灵。这是重定向路由最经典的那个坑。
 */
export function GroupIndex({ to }: { to: string }) {
  const tab = findTab(to);
  const ids = tab ? sectionIds(tab) : [];
  const [remembered] = useStickyTab(`group.${to}`, ids, ids[0] ?? "");
  const { search, hash } = useLocation();
  if (!tab || !ids.length) return <Navigate to="/" replace />;
  return <Navigate to={`${sectionPath(tab, remembered)}${search}${hash}`} replace />;
}

/**
 * 一条合并之前的旧地址。把人送到它现在所在的位置。
 *
 * **`search` 与 `hash` 原样带过去。** 这不是锦上添花：监控页的段落浏览器就是靠
 * `?seg=` 深链定位的（`SegmentBrowser`），一个丢掉 query 的重定向会把
 * 「我发给同事的那个具体段落」变成「监控页首页」—— 而链接照样打得开，
 * 所以谁都不会意识到定位丢了。
 */
export function LegacyRedirect({ to }: { to: string }) {
  const { search, hash } = useLocation();
  return <Navigate to={`${to}${search}${hash}`} replace />;
}

export type { NavTab };
