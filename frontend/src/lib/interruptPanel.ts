/**
 * 私聊「等待人工介入」面板的显示判定 —— pure, `node --test`able.
 *
 * 这几行本来可以直接写在 PendingInterrupts.tsx 的 JSX 里。拎出来是因为其中
 * 一条是**被明确要求过、又极容易被当成冗余删掉**的性质，而它一旦坏掉，症状是
 * 「有一轮对话永远卡着而没人知道」——不崩、不报错、截图看不出。
 *
 * ── 那条性质：可关闭 ≠ 可消失 ──────────────────────────────────────────
 * 要的是「人工介入别挡住后面的界面」。于是面板可以收起。
 * 但收起**只收卡片**，「还有 N 项等着你」那一行必须留着：发起中断的那一轮是
 * 阻塞的，一个能被彻底关干净的提示等于没有提示，而这正是从前用全屏弹窗
 * 自动怼脸想解决的问题。两个需求（别挡住 / 别错过）都要满足，不是二选一。
 *
 * 所以 `render` 与 `showCards` 是**两个**返回值而不是一个。没有名字的区分
 * 迟早被下一个人合并掉。
 */

export interface InterruptRowLike {
  kind?: string | null;
}

export interface InterruptDataLike {
  interrupts?: readonly InterruptRowLike[] | null;
  degraded?: boolean | null;
}

export interface InterruptPanelView {
  /** 整块面板画不画。 */
  render: boolean;
  /** 卡片画不画 —— 收起时为 false，而 `render` 仍是 true。 */
  showCards: boolean;
  /** 待处理条数。 */
  count: number;
  /** 标题。见 `interruptPanelView` 里那条分支为什么还留着。 */
  headline: string;
}

export function interruptPanelView(
  data: InterruptDataLike | null | undefined,
  collapsed: boolean,
): InterruptPanelView {
  const rows = data?.interrupts ?? [];
  const count = rows.length;
  const degraded = !!data?.degraded;
  // 降级时即使一条都读不到也要出面板 —— 「中断队列读不出来」和「没有中断」
  // 是两句话，而把前者显示成后者正是让人停止排查的那一句。
  const render = count > 0 || degraded;
  // 这条分支原本写的理由是「提问和审批是两件事」。⑰ 之后**审批那一半没有了**
  // （`HumanInTheLoopMiddleware` / `ModeGatedPulseHITLMiddleware` / `buffer_hitl`
  // 的确认框全部割除，DANGEROUS 改成只提醒），所以那个理由已经变成一句假话 ——
  // 照它去推理会以为 else 分支是死代码然后把它删掉。
  //
  // 真实情况：树里发 interrupt 的只剩三种 kind。
  //   · `ask_user`         —— `ask_tools.ask_user`，缺关键参数时的正当提问（明确
  //                           保留的一类）；
  //   · `ask_user`         —— supervisor 路由不出去时的 `_ask_operator_node`，同一形状；
  //   · `workflow_human`   —— 工作流的 `human` 节点。内建工作流一个都没用它，
  //                           只有用户自建技能可能有。
  // 前两种是**智能体在问你一件事**；第三种是**流程走到一个需要人的步骤**，不是提问。
  // 所以 else 分支既不是死的，措辞也正是它该有的那句。
  const asking = rows.some((r) => r?.kind === "ask_user");
  return {
    render,
    // 收起了就不画卡片；但注意 `render` 不受 collapsed 影响 —— 见文件头。
    showCards: render && !collapsed && count > 0,
    count,
    headline: asking ? "智能体在问你" : "等待人工介入",
  };
}
