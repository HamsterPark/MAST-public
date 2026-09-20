// ════════════════════════════════════════════════════════════════════════════
// stickyTab — 「离开再回来，还停在刚才那一页吗」的**纯**部分。
//
// 「从代理对话离开再回来，希望还是代理对话栏目」。
//
// 症结在一句注释里早就写明白过（`pages/ChatPage.tsx`，）:路由元素由
// AppLayout 的 <Outlet/> 渲染，**每次回到这个栏目都是一次全新挂载**，所以
// `useState(默认值)` 每次都把子页复位。ChatPage 当时手写了一份 localStorage
// 读写。它是对的 —— 而另外六页一份都没有。
//
// 这个文件存在的理由不是「省几行」，是那个**读时校验**:存进去的 id 会过期
// （子页改名、合并、删掉），而一个页面拿着不认识的 id 去渲染，得到的是一片空白
// 或者一个所有子页都不高亮的 tab 条。那不像「配置过期」，像「这一页坏了」。
// 抄六份就是把这条校验交给六个人各记一次。
//
// 无 React import ⇒ node --test 直接跑（`frontend/test/stickyTab.test.ts`）。
// ════════════════════════════════════════════════════════════════════════════

/** 存储键的统一前缀。改它会让所有人的记忆一次性失效 —— 那也是唯一的迁移方式。 */
export const STICKY_TAB_PREFIX = "mast.subtab.";

/** `page` → 完整的 localStorage 键。 */
export function stickyTabKey(page: string): string {
  return `${STICKY_TAB_PREFIX}${page}`;
}

/**
 * 读回来的值能不能用。
 *
 * **不认识就退回默认值**，而不是原样返回。存进去的 id 会过期：子页被改名、
 * 被合并进别的栏目、干脆删掉（#34 就要合并五组大标签）。一个页面拿着不认识的
 * id 去渲染，得到的是一片空白，或者一条所有段都不高亮的 tab 条 —— 而用户
 * 看到的不是「我的偏好过期了」，是「这一页坏了」，报上来的也会是后面那句。
 *
 * `fallback` 不在 `valid` 里时仍然原样返回：调用方给的默认值是它自己的事，
 * 在这里悄悄改掉它只会让「默认值写错了」变成一个查不到的问题。
 */
export function pickValidTab<T extends string>(
  stored: string | null | undefined,
  valid: readonly T[],
  fallback: T,
): T {
  if (typeof stored !== "string" || !stored) return fallback;
  return (valid as readonly string[]).includes(stored) ? (stored as T) : fallback;
}

/**
 * 读一个记住的子页。storage 不可用（隐私模式、被禁用）时**不抛**，退默认值。
 *
 * 一个记不住偏好的浏览器仍然要能用这个软件 —— 这条和「UI 绝不冻结」是同一条
 * 纪律的两个面。
 */
export function readStickyTab<T extends string>(
  page: string,
  valid: readonly T[],
  fallback: T,
): T {
  try {
    return pickValidTab(localStorage.getItem(stickyTabKey(page)), valid, fallback);
  } catch {
    return fallback;
  }
}

/** 记住一个子页。失败就算了 —— 这一次的选择在本次会话里照样生效。 */
export function writeStickyTab(page: string, id: string): void {
  try {
    localStorage.setItem(stickyTabKey(page), id);
  } catch {
    /* private mode / storage disabled — the selection still works this session */
  }
}
