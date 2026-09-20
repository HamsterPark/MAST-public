/**
 * 人工介入面板 —— 显示判定，以及「它不许再变回一个遮罩」的结构闸门。
 *
 * 已知问题：人工介入(HITL)期间看不到后面的界面显示。
 *
 * 根因是 ChatPage 一有中断就自动弹一个 `fixed inset-0 bg-black/60
 * backdrop-blur-sm` 的全屏 Modal —— 恰恰在要判断该不该批准的那一刻，把用来做这个
 * 判断的读数（偏压 / 电流 / Z / 扫描状态 / 告警）全盖住了。
 *
 * 下面两组测试守的是两件不同的事：
 *   · 纯函数组 —— 「收起只收卡片，那一行永远留着」（可关闭 ≠ 可消失）；
 *   · 结构组   —— 遮罩没有偷偷回来。
 * 第二组必须存在，因为第一组全绿的同时，有人完全可以在 ChatPage 里把那个
 * 自动弹窗原样加回去（它看起来很有道理：中断是紧急的）。
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

import { interruptPanelView } from "../src/lib/interruptPanel.ts";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");
const CHAT_PAGE = join(SRC, "pages", "ChatPage.tsx");
const PANEL = join(SRC, "components", "chat", "PendingInterrupts.tsx");

/**
 * 去掉注释，只留代码。
 *
 * 第一版没有这一步，于是闸门被 PendingInterrupts.tsx 里那句「从前这是一个
 * `<Modal>`」判红 —— 它检查的是**注释**。同一个形状在本轮已经咬过一次（改端点
 * 常量时改到了文件头注释里的那一份）：一个扫源码的判据必须明说它扫的是代码，
 * 否则「解释自己为什么不这么写」的注释会被当成「这么写了」。
 *
 * `//` 只在前面不是 `:` 时才算注释，免得把字符串里的 `https://` 拦腰截断。
 */
function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1");
}

const ask = { kind: "ask_user" };
const approve = { kind: "approval" };

describe("interruptPanelView", () => {
  it("没有待处理事项时整块不画", () => {
    assert.equal(interruptPanelView({ interrupts: [] }, false).render, false);
    assert.equal(interruptPanelView(undefined, false).render, false);
    assert.equal(interruptPanelView(null, false).render, false);
  });

  it("有事项时画，并报出条数", () => {
    const v = interruptPanelView({ interrupts: [approve, ask] }, false);
    assert.equal(v.render, true);
    assert.equal(v.showCards, true);
    assert.equal(v.count, 2);
  });

  // 这一条是整个 #38 的支点。收起是操作员要的（别挡路），但收起之后那一轮
  // **仍然阻塞着** —— 面板整块消失就等于回到「对话安静十五分钟然后超时」，
  // 也就是当初那个全屏弹窗想解决的问题。别挡住 / 别错过，两个都要。
  it("收起只收卡片 —— 面板本身仍然在（可关闭 ≠ 可消失）", () => {
    const v = interruptPanelView({ interrupts: [approve] }, true);
    assert.equal(v.render, true, "收起把整块面板也藏了 —— 这一轮会永远卡着没人知道");
    assert.equal(v.showCards, false, "收起了却还在画卡片");
    assert.equal(v.count, 1, "收起之后条数就说不出来了");
  });

  // 「读不出来」与「没有」必须是两句话。把前者显示成后者，正是让人停止排查的那句。
  it("降级时即使一条都没读到也要出面板", () => {
    const v = interruptPanelView({ interrupts: [], degraded: true }, false);
    assert.equal(v.render, true);
    assert.equal(v.showCards, false, "没有行可画就不该画卡片区");
  });

  it("有人在提问时标题说的是提问", () => {
    assert.equal(interruptPanelView({ interrupts: [ask] }, false).headline, "智能体在问你");
    assert.equal(
      interruptPanelView({ interrupts: [approve, ask] }, false).headline,
      "智能体在问你",
      "混合时应以提问为准 —— 提问不答，那一轮就动不了",
    );
  });

  // ⑰ 割掉 DANGEROUS 审批之后，非 `ask_user` 这条分支**看起来**成了死代码，
  // 而「这个分支永远走不到」正是下一个人删掉它的理由。它没死：工作流的 `human`
  // 节点发 `kind: "workflow_human"`（内建工作流没用它，操作员自建技能可能有）。
  //
  // 而且措辞恰恰不能合并 —— 流程走到一个需要人的步骤，不是智能体在问你一件事。
  it("工作流 human 节点不说成「智能体在问你」（那条分支没死）", () => {
    assert.equal(
      interruptPanelView({ interrupts: [{ kind: "workflow_human" }] }, false).headline,
      "等待人工介入",
    );
    assert.equal(
      interruptPanelView({ interrupts: [{ kind: "workflow_human" }, ask] }, false).headline,
      "智能体在问你",
      "混着来时提问优先 —— 它是那一轮真正卡住的原因",
    );
  });

  // kind 缺失 / 是个没见过的串，都不该冒充成提问。
  it("认不出的 kind 走保守措辞", () => {
    assert.equal(interruptPanelView({ interrupts: [{}] }, false).headline, "等待人工介入");
    assert.equal(interruptPanelView({ interrupts: [approve] }, false).headline, "等待人工介入");
  });
});

describe("人工介入不许再变成遮罩 ", () => {
  const chatPage = stripComments(readFileSync(CHAT_PAGE, "utf8"));
  const panel = stripComments(readFileSync(PANEL, "utf8"));

  // 闸门自检：读到的确实是那两个文件。路径写错会让下面每一条都在空字符串上
  // 通过，而那和「全部通过」长得一模一样。
  it("闸门读到的是真的那两个文件", () => {
    assert.match(chatPage, /PendingInterrupts/, "ChatPage.tsx 里找不到面板，路径或接线已经变了");
    assert.match(panel, /InterruptCard/, "PendingInterrupts.tsx 里找不到卡片，路径已经变了");
  });

  it("面板不是 Modal，也不铺全屏", () => {
    assert.doesNotMatch(panel, /<Modal\b/, "面板又变回 Modal 了 —— 它会压黑并模糊整个界面");
    assert.doesNotMatch(panel, /fixed inset-0/, "面板铺了全屏遮罩");
  });

  it("仪器 chat 不再自己弹人工介入的窗", () => {
    assert.doesNotMatch(
      chatPage, /setHitlOpen/,
      "自动弹窗的开关回来了 —— #38 报的就是它",
    );
    assert.doesNotMatch(
      chatPage, /<HitlModal\b/,
      "全屏 HITL 弹窗回到 ChatPage 了",
    );
  });

  // 「不可能错过」这一半也要守住：角标按钮必须还能把人带到面板那里。
  // 只删弹窗不留锚点，转录一长就真的看不见了。
  it("角标按钮仍然把人带到面板（reveal 而不是弹窗）", () => {
    assert.match(chatPage, /hitlRef\.current\?\.reveal\(\)/, "角标不再指向面板");
    assert.match(panel, /reveal:/, "面板没有暴露 reveal");
    assert.match(panel, /scrollIntoView/, "reveal 不会把人滚过去");
  });
});
