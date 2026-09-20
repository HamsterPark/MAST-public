// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 大图里的键盘（设计 D14，原版 app.js 的 keydown 处理）。
//
// 表放在这里、组件只执行，是为了让「哪个键做什么」能被测试整张钉住：键盘标记是
// 这个页面最高频的操作，一个键悄悄换了含义（比如 3 从「排除」变成「重点」）不会
// 报任何错，只会让几百条标记打错。
//
// 按 `e.code` 而不是 `e.key`：与键盘布局、输入法、Shift 状态都无关——Q 永远是
// 第一个标签的那个物理键。
// ════════════════════════════════════════════════════════════════════════════

export const HOT_CODES = [
  "KeyQ", "KeyW", "KeyE", "KeyR", "KeyT", "KeyY", "KeyU", "KeyI", "KeyO",
] as const;

/** 标签按钮上显示的快捷键字母，与 HOT_CODES 一一对应。 */
export const HOT_LABELS = "QWERTYUIO";

export type KeyAction =
  | { t: "next" }
  | { t: "prev" }
  | { t: "first" }
  | { t: "last" }
  | { t: "rate"; r: number }
  | { t: "note" }
  | { t: "pick"; code: "KeyS" | "BracketLeft" | "BracketRight" }
  | { t: "anchor"; side: "prev" | "next" }
  | { t: "tag"; index: number }
  | { t: "close" }
  | { t: "blur" };

export interface KeyInput {
  code: string;
  ctrl?: boolean;
  meta?: boolean;
  alt?: boolean;
  /** 焦点在文本框、非勾选框的 input、或下拉框里。 */
  inField?: boolean;
  /** 当前条目是偏压谱或网格谱（有前后帧对照、A/D 才有意义）。 */
  spectrumLike?: boolean;
}

/**
 * 一次按键 → 动作；`null` 表示不处理（也不 preventDefault）。
 *
 * * 焦点在输入控件里：只认 Esc（让它失焦），其余键交还给输入框——正在写备注时
 *   按 1 必须是输入一个「1」，而不是给这张图打分。
 * * 带 Ctrl / Meta / Alt：不处理，把复制粘贴、浏览器快捷键留给浏览器。
 */
export function keyAction(k: KeyInput): KeyAction | null {
  if (k.inField) return k.code === "Escape" ? { t: "blur" } : null;
  if (k.ctrl || k.meta || k.alt) return null;
  switch (k.code) {
    case "ArrowRight":
    case "ArrowDown":
    case "PageDown":
      return { t: "next" };
    case "ArrowLeft":
    case "ArrowUp":
    case "PageUp":
      return { t: "prev" };
    case "Home":
      return { t: "first" };
    case "End":
      return { t: "last" };
    case "Digit1":
    case "Numpad1":
      return { t: "rate", r: 1 };
    case "Digit2":
    case "Numpad2":
      return { t: "rate", r: 2 };
    case "Digit3":
    case "Numpad3":
      return { t: "rate", r: -1 };
    case "Digit0":
    case "Numpad0":
    case "Backquote":
      return { t: "rate", r: 0 };
    case "KeyN":
      return { t: "note" };
    case "KeyS":
    case "BracketLeft":
    case "BracketRight":
      return { t: "pick", code: k.code };
    case "KeyA":
      return k.spectrumLike ? { t: "anchor", side: "prev" } : null;
    case "KeyD":
      return k.spectrumLike ? { t: "anchor", side: "next" } : null;
    case "Escape":
      return { t: "close" };
    default: {
      const i = (HOT_CODES as readonly string[]).indexOf(k.code);
      return i >= 0 ? { t: "tag", index: i } : null;
    }
  }
}
