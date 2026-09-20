// 显式 `.ts` 后缀是本目录的约定：`node --test` 直接跑 TS 源码，ESM 解析器不会
// 替你补后缀（envHistory.ts / monitoring.ts / initFilter.ts 同）。
import { LEGACY_REDIRECTS } from "./nav.ts";

// ════════════════════════════════════════════════════════════════════════════
// 「用户此刻是不是正站在初始化页上」—— 横幅用它决定要不要闭嘴。
//
// 为什么单独一个纯函数：这个判断有一个**写在别的文件里的前提**。
//
// 2026-08-06初始化页从 `/setup` 挪到 `/settings/setup`，旧地址改成
// 重定向。于是「判 `/settings/setup` 就够了」成立，但它成立的理由是
// **「`/setup` 只作为重定向存在」** —— 而那句话住在 router.tsx 里，不在横幅里。
//
// 两件事因此是安静的：
//
//   1. `<Navigate>` 是在 effect 里跳的，不是渲染期。所以走旧书签进来时，有**一帧**
//      的 pathname 仍是 `/setup`，而 AppLayout（连同横幅）已经挂上了 —— 横幅会闪
//      一下，指着一个人正要去的地方。
//   2. 更要紧的是：哪天有人把 `/setup` 变回真路由、或者重定向改成服务端做，
//      横幅就会**结结实实地挂在初始化页自己头上**，而改动的人没有任何理由会想到
//      来看这个文件。
//
// 所以别名不写死，从 `LEGACY_REDIRECTS` **派生** —— 那张表是「哪些旧地址指向这里」
// 的真源，而且它「只增不改」。往表里再加一条指向初始化页的旧地址，这里自动跟上。
// ════════════════════════════════════════════════════════════════════════════

/** 初始化页的正规地址。与 router.tsx 的 `SECTION_ELEMENTS` 键、nav.ts 的段名同源，
 *  三处一致由 `tests/v2/unit/api/test_instrument_init_frontend_parity.py` 钉住。 */
export const SETUP_PATH = "/settings/setup";

/**
 * 所有会把人送到初始化页的地址：正规地址 + 全部指向它的旧地址。
 *
 * 导出给测试 —— 一个「本该有两条却只剩一条」的名单，从行为上看和正确的一模一样，
 * 直到有人真的用了那条旧书签。
 */
export const SETUP_ALIASES: string[] = [
  SETUP_PATH,
  ...Object.entries(LEGACY_REDIRECTS)
    .filter(([, to]) => to === SETUP_PATH)
    .map(([from]) => from),
];

/**
 * True 表示这个 pathname 就是初始化页（或它的某个旧地址）。
 *
 * 用前缀匹配而不是相等：将来若初始化页自己长出子段（`/settings/setup/xxx`），
 * 横幅仍然该闭嘴。`/settings/setupfoo` 这种**不**算 —— 边界必须落在路径分隔符上，
 * 否则一个名字碰巧同前缀的新页面会让横幅在那里神秘消失。
 */
export function isSetupPath(pathname: string): boolean {
  const p = (pathname || "").replace(/\/+$/, "") || "/";
  return SETUP_ALIASES.some(
    (a) => p === a || p.startsWith(a.endsWith("/") ? a : `${a}/`),
  );
}
