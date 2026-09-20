import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import clsx from "clsx";
import { useInstrumentInit } from "@/hooks/useInstrumentInit";
import { SETUP_PATH, isSetupPath } from "@/lib/setupNav";

// ── 新仪器初始化的「自动弹出」──────────────────────────────────────────────────
//
// 刻意是**常驻横幅**，不是模态框。
//
// 项目规约 的硬规则是「UI 绝不冻结」：一个盖住全屏、必须填完才能关的模态，
// 在用户只想按急停的时候就是灾难。所以它醒目、可点进去、可以折成一行小字，
// 但在必填齐之前**不消失**——折叠状态只活在这一次会话里（useState，不落盘），
// 刷新页面它就回来。
//
// 什么时候出现（判据在后端 GET /api/instrument-init）：
//   * 有必填项没有答案（**权威判据，纯内容判据**——从值本身算，不看任何标记，
//     所以升级覆盖了哪个文件都影响不到它）；
//   * 硬件指纹与上次完成时不同（换了机器 / 重做了标定）；
//   * 从来没盖过完成戳。
//
// 读不出来（degraded）时**不显示**：一份读不出来的清单如果表现成一条红色警告，
// 一周之内就会被无视，而那正好毁掉它在真的缺数时的作用。

export function SetupBanner() {
  const [collapsed, setCollapsed] = useState(false);
  const loc = useLocation();
  const q = useInstrumentInit();
  const d = q.data;

  // 已经在这一页上了就别再喊。
  //
  // 2026-08-06(#34)：初始化页从 /setup 挪到了 /settings/setup，旧地址改成重定向。
  // 判据不写死路径 —— 从 nav.ts 的 LEGACY_REDIRECTS 派生（见 lib/setupNav.ts）：
  // 「判新地址就够了」这句话成立的**前提**（/setup 只作为重定向存在）住在
  // router.tsx 里，不在这个文件里，而依赖一个写在别处的前提正是本仓反复栽的形状。
  if (isSetupPath(loc.pathname)) return null;
  if (q.isLoading || q.error || !d || d.degraded) return null;
  if (!d.should_prompt) return null;

  const req = d.counts?.required ?? { total: 0, complete: 0 };
  const missing = req.total - req.complete;
  const urgent = d.needs_setup;

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        className={clsx(
          "flex w-full items-center gap-2 border-b px-4 py-1 text-left text-xs",
          urgent
            ? "border-mast-danger-border bg-mast-danger-bg text-mast-danger"
            : "border-mast-warn-border bg-mast-warn-bg text-mast-warn",
        )}
      >
        <span>●</span>
        <span>
          {urgent
            ? `新仪器初始化：还有 ${missing} 项必填没填`
            : "新仪器初始化：建议过一遍"}
        </span>
        <span className="ml-auto underline">展开</span>
      </button>
    );
  }

  return (
    <div
      className={clsx(
        "border-b px-4 py-2.5 text-sm",
        urgent
          ? "border-mast-danger-border bg-mast-danger-bg text-mast-danger"
          : "border-mast-warn-border bg-mast-warn-bg text-mast-warn",
      )}
      role="status"
    >
      {/* 一句话，没有副标题。
          #39 要求删掉的就是那条副标题：「缺了它们，某些安全网是按出厂占位值
          在拦人的 —— 出厂值与任何一台真实机器都不对应。」它解释的是**为什么要有
          这块横幅**，而看见横幅的人已经不需要被说服了；他需要的是数字和那个按钮。
          三条分支的副标题一起删（另外两条同样是解释，不是信息）。 */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <b>
          {urgent
            ? `这台仪器还有 ${missing} 项必填参数没有确定`
            : d.fingerprint_changed
              ? "硬件指纹变了 —— 这可能是另一台仪器"
              : "这台仪器还没走过初始化"}
        </b>
        <div className="ml-auto flex items-center gap-2">
          <Link
            to={SETUP_PATH}
            className="rounded-mast-ctl border border-current px-2.5 py-1 text-xs font-semibold hover:opacity-80"
          >
            去填 →
          </Link>
          <button
            type="button"
            onClick={() => setCollapsed(true)}
            className="text-xs underline opacity-80 hover:opacity-100"
            title="折起来（只在这次会话里生效，刷新就回来）"
          >
            稍后
          </button>
        </div>
      </div>
    </div>
  );
}
