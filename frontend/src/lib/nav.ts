// 顶栏导航表 — the only place a page becomes reachable without knowing its URL.
//
// Lives apart from AppLayout so it can be asserted against. A page missing from
// here is not broken in any way a type checker, a test run, or a screenshot can
// see: the route still resolves, the component still renders, every link that
// points at it still works. It is simply unreachable for someone who does not
// already know it exists — and the report that reaches you is not「这个页面没了」
// but「这个页面只能进去一次」, which sounds like a state bug and is not one.
//
// That is exactly what happened to 仪器初始化. Both of its entry
// points were wired and working: a banner (which by design disappears once the
// completion stamp is set) and a button on the 设置 page. Neither is navigation.
// Correcting 退针方向 — the field where a wrong answer drives
// the tip INTO the sample — had no way in through the UI, and it had to be written
// over the API instead.
//
// ── 17 个大标签合并成 10 个 ─────────────────────
//
// 分组方案：高级管理+设置+仪器初始化+用量花销 合并成一个;记忆 并进
// 实验记录;光学台 并进实验性功能;电流监控 和 环境历史 合二为一;技能 和
// 技能构建器 合并。五组分法是给定的,照做。
//
// **合并的是导航，不是页面**：每个原页面仍然是原来那个组件，一行没动，只是从
// 「顶栏的一项」变成「某个大标签下的一段」。这样做的代价是层级深了一级
// （高级管理那条链现在有五层），收益是顶栏从 17 项挤成 10 项 —— 而挤到换行的
// 顶栏正是用户在抱怨的那件事。
//
// **二级页是真的路由段**，不是页面内的 useState。理由是书签：`/settings` 这类
// 地址在用户的浏览器里存了几个月，合并之后它们必须还能落到对的地方，而一个
// 存在 localStorage 里的子页选择回答不了「这个链接指向哪一页」。
// 旧地址一条不落地重定向到新位置，见 LEGACY_REDIRECTS。

export interface NavSection {
  /** 二级路径段（拼在大标签的 `to` 后面）。 */
  seg: string;
  label: string;
}

export interface NavTab {
  to: string;
  label: string;
  /** react-router `end`: match this path exactly, not as a prefix. */
  end?: boolean;
  /**
   * 这个大标签底下的二级页。
   *
   * 有 sections 的大标签，其 `to` 本身**不渲染内容**，只把人送到某一段
   * （上次停的那一段，没有记忆就是第一段）。没有 sections 的就是一页到底。
   */
  sections?: NavSection[];
}

export const TABS: NavTab[] = [
  { to: "/", label: "仪器 Chat", end: true },
  { to: "/agents", label: "Agents" },
  { to: "/qa", label: "查询助手" },
  // 技能 + 技能构建器（#34 第五组）。构建器是「做一个新技能」，目录是「看现有
  // 技能」—— 同一件事的两端，本来就该在一起。
  {
    to: "/skills",
    label: "技能",
    sections: [
      { seg: "library", label: "技能目录" },
      { seg: "builder", label: "技能构建器" },
    ],
  },
  { to: "/literature", label: "文献库" },
  // 实验记录 + 记忆 + 多天 conduct（#34 第二组）。
  //
  // conduct 放这一组而不是新开第 11 个大标签：大标签数量要收紧，
  // 而一份 conduct **必须**绑定一个实验（`experiment_id` 必填，它的 spec 快照与
  // progress.jsonl 都落在那个实验的文件夹里），所以它本来就属于这一组。
  //
  // 它同时还有**第二个入口**：等人的时候心愿单页会出现一条 `conduct:<id>` 的
  // 请求。这不是冗余 —— 两个入口都不是导航
  // 就等于没有入口；这里一个是导航、一个是待办，两者互补。
  //
  // 数据图库（2026-09-13）也在这一组：它整理的是实验采到的数据（预处理 → 展示 → 人工
  // 标记/系列），与「记录」同属一件事；理由的完整版在 nav.test.ts 的 ADDED_LATER。
  {
    to: "/records",
    label: "实验记录",
    sections: [
      { seg: "log", label: "记录" },
      { seg: "gallery", label: "数据图库" },
      { seg: "conduct", label: "多天 conduct" },
      { seg: "memory", label: "记忆" },
    ],
  },
  { to: "/wishlist", label: "心愿单" },
  // 设置 + 仪器初始化 + 高级管理 + 用量花销（#34 第一组）。
  //
  // 组根沿用 `/settings` 而不是新造一个 `/system`：这是四个里最常被收藏、也最
  // 常被人直接敲的地址。换掉它等于让最常用的那个书签走一次重定向，白白多一跳。
  //
  // 仪器初始化仍然有自己的位置 —— 把它请进顶栏的理由（横幅会消失、
  // 设置页里的按钮不是导航）一条都没变，只是那个位置现在在「设置」下面一层。
  {
    to: "/settings",
    label: "设置",
    sections: [
      { seg: "general", label: "常规设置" },
      { seg: "setup", label: "仪器初始化" },
      { seg: "admin", label: "高级管理" },
      { seg: "usage", label: "用量花销" },
    ],
  },
  // 电流监控 + 环境历史（#34 第四组）。两者都是「仪器现在/最近怎么样」。
  {
    to: "/monitoring",
    label: "监控",
    sections: [
      { seg: "current", label: "电流监控" },
      { seg: "env", label: "环境历史" },
    ],
  },
  // 实验性功能 + 光学台（#34 第三组）。
  {
    to: "/experimental",
    label: "实验性功能",
    sections: [
      { seg: "tools", label: "仪器工具" },
      { seg: "optics", label: "光学台" },
    ],
  },
];

/**
 * 合并之前的顶栏地址 → 合并之后它去了哪里。
 *
 * 纯数据，因为它要被两边用：`router.tsx` 拿它生成重定向路由，测试拿它断言
 * 「一条都没漏、且每一条都指向真的存在的地方」。手写两份就会漏 —— 而漏掉的
 * 那一条的症状是用户的书签打开一个 404 白页，看起来像「这个功能被删了」。
 *
 * 这张表**只增不改**：一条旧地址一旦进来就永远留着。它的成本是路由表里多一行，
 * 而删掉它的成本是某个人存了半年的书签突然打不开，还查不出为什么。
 */
export const LEGACY_REDIRECTS: Record<string, string> = {
  "/builder": "/skills/builder",
  "/cognition": "/records/memory",
  "/admin": "/settings/admin",
  "/setup": "/settings/setup",
  "/usage": "/settings/usage",
  "/env-history": "/monitoring/env",
  "/optics": "/experimental/optics",
};

/**
 * Pages that are reachable by URL on purpose and deliberately have no tab.
 *
 * An explicit list, so that "no tab" is always a decision someone wrote down
 * rather than an omission nobody noticed. `/dashboard` is embedded elsewhere;
 * `/chat` is an alias of `/`.
 *
 * ── `/vision` 那一条的理由从「embedded elsewhere」换掉 ──
 *
 * 那句话是半句真话，剩下的半句是：对话页的「视觉缓冲」
 * 只嵌了 `/vision` 七个子页里的四个（扫描地图 / 视觉脉冲 / 近期帧 / 缓冲），
 * 而且嵌的那张扫描地图**没有粗动大地图**。信号捕获+FFT、长期监控、拼图三个子页
 * 一次都没有被嵌过。更糟的是：在加上那个链接之前，全仓渲染出来的链接里**没有
 * 一个指向 `/vision`** —— `VisionRibbon` 里那个 `<Link to="/vision">` 只在不传
 * `onOpenBuffer` 时才画，而唯一的调用方永远传。
 *
 * 于是它不是「没有 tab 但嵌在别处」，是「没有 tab、没有链接、只有知道地址的人
 * 进得去」。这类缺口不会表现成「导航里少了一项」，而是「某个具体功能找不到入口」。
 * 与仪器初始化那次同一个形状。
 *
 * 现在「视觉缓冲」页顶有一个指过去的链接，所以这条豁免重新成立 —— 靠的是那个
 * 链接，不是靠这段注释。它被 `test/scanMapPairing.test.ts` 钉住。
 */
export const INTENTIONALLY_UNLISTED = ["/dashboard", "/vision", "/chat"] as const;

/** 一个大标签下某一段的完整路径。 */
export function sectionPath(tab: NavTab, seg: string): string {
  return `${tab.to === "/" ? "" : tab.to}/${seg}`;
}

/** 这个大标签认得的全部二级段 id（喂给 useStickyTab 的合法名单）。 */
export function sectionIds(tab: NavTab): string[] {
  return (tab.sections ?? []).map((s) => s.seg);
}

/** 顶栏里的某一项，按 `to` 找。 */
export function findTab(to: string): NavTab | undefined {
  return TABS.find((t) => t.to === to);
}
