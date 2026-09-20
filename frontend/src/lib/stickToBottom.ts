// 「跟到底部」的纯判断 —— 不碰 DOM,所以能在 node --test 里跑。
//
// ## 出处
//
// 已知问题:chat 对话会自动锁定滚动条 —— 对话滚动条与
// 整个标签页的滚动条都被锁定,想向下拉查看历史就被自动拉回,
// 两个滚动条都是这样。
//
// 两个毛病,分别对应这个文件里的两半:
//
// **① 无条件跟随。** 原来是 `useEffect(() => endRef.scrollIntoView(), [messages])`
//    —— 每来一条消息就拉到底,**从不问用户此刻在看哪儿**。用户往上翻是想读历史,
//    而下一条消息把他弹回去。判断在 `isAtBottom()`。
//
// **② 连祖先一起滚。** `scrollIntoView()` 会把**每一个可滚动祖先**都滚到能看见
//    目标为止 —— 所以聊天面板的滚动条和整页的滚动条一起动,正是要求的「两个」。
//    修法是只动最近的那一个滚动容器(`nearestScrollable()` 找它),
//    对它设 `scrollTop`,不用 `scrollIntoView`。
//
// 阈值不写 0:内容重排、图片加载完、亚像素取整都会让 `scrollTop` 差上几个像素,
// 判 0 会让「明明在底部」被读成「用户翻上去了」,跟随就永远不触发了。

/** 一次滚动量测。字段名与 DOM 同名,方便直接从元素上取。 */
export interface ScrollMetrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

/**
 * 离底部多少像素以内**仍然算在底部**。
 *
 * 48 px ≈ 一行气泡的高度:用户往上翻不足一行,意图上仍然是「我在看最新的」。
 */
export const STICK_THRESHOLD_PX = 48;

/** 现在算不算贴着底部(跟随只在这个状态下发生)。 */
export function isAtBottom(m: ScrollMetrics, threshold = STICK_THRESHOLD_PX): boolean {
  const gap = m.scrollHeight - m.scrollTop - m.clientHeight;
  // 内容比容器还短 ⇒ 根本没有滚动条 ⇒ 永远算在底部(否则第一条消息就跟不上)。
  if (m.scrollHeight <= m.clientHeight) return true;
  // 负数出现在橡皮筋回弹(macOS/触屏)那一瞬 —— 那显然是在底部,不是「翻上去了」。
  return gap <= Math.max(0, threshold);
}

/**
 * 找**最近的**可滚动容器:先看 `el` 自己,再一路往上找祖先;没有就返回 null。
 *
 * 只找一个,找到就停 —— 这正是与 `scrollIntoView()` 的区别所在:
 * 那个会把找到的每一个都滚,而我们只想动装着这段对话的那一个。
 *
 * 先看自己,是为了让两种调用方式共用一个函数:聊天那边传的是转录末尾的一个空
 * 锚点(自己不可滚,要找祖先),而任务面板传的就是滚动容器本身。分成两个 API
 * 只会多出一处「传错了也不报错、只是不跟随」的地方。
 *
 * `getStyle` 可注入,是为了让这个函数在没有 DOM 的测试里也能跑。
 */
export function nearestScrollable(
  el: Element | null,
  getStyle: (e: Element) => { overflowY?: string } = (e) =>
    (typeof globalThis !== "undefined" && (globalThis as any).getComputedStyle
      ? (globalThis as any).getComputedStyle(e)
      : {}),
): Element | null {
  let cur: Element | null = el ?? null;
  while (cur) {
    const oy = String(getStyle(cur).overflowY || "");
    // `overlay` 是老 WebKit 的写法,行为等同 auto —— 漏了它就会一路找到 <body>,
    // 于是又变成「连整页一起滚」。
    if (oy === "auto" || oy === "scroll" || oy === "overlay") return cur;
    cur = cur.parentElement;
  }
  return null;
}
