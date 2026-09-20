// 复制文字（旧版兼容格式 app.js 的 copyText）。
//
// `navigator.clipboard` 只在安全上下文里有：本机 http://127.0.0.1 可以，局域网里用
// http://<IP> 打开 MAST 时就没有——那正是远程看数据的常见用法，所以退回到临时
// textarea + execCommand。两条都失败时按钮上说「复制失败」，而不是假装成功。

import { useState } from "react";
import clsx from "clsx";

export async function copyText(s: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(s);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = s;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

export function CopyButton({
  text,
  label = "复制",
  className,
  title,
}: {
  text: string | (() => string);
  label?: string;
  className?: string;
  title?: string;
}) {
  const [state, setState] = useState<"" | "ok" | "fail">("");
  return (
    <button
      type="button"
      title={title}
      className={clsx(
        "rounded border border-mast-border px-1.5 text-xs text-mast-muted hover:border-mast-accent hover:text-mast-text",
        className,
      )}
      onClick={async (e) => {
        e.stopPropagation();
        const ok = await copyText(typeof text === "function" ? text() : text);
        setState(ok ? "ok" : "fail");
        setTimeout(() => setState(""), 1200);
      }}
    >
      {state === "ok" ? "已复制" : state === "fail" ? "复制失败" : label}
    </button>
  );
}
