import { test, expect } from "@playwright/test";

/**
 * 聊天页里那个「注入的上下文」折叠块 —— **真渲染一次**。
 *
 * 单测钉的是对齐逻辑（`lib/injectedContext.ts`），闸门钉的是接线
 * （分段渲染必须传整份转录）。中间那一段 —— 「这个组件挂上去之后真的画得出来、
 * 点得开、内容对得上」 —— 两者都证明不了：一个 hook 用错、一个 import 写错，
 * 类型全绿、构建全绿，症状只有「点了没反应」。
 *
 * 后端全部打桩：这条用例要验的是**前端**，而让它去等一次真实的模型调用意味着
 * 要有 API key、要花钱、还要看运气 —— 那样的用例最后一定会被 skip 掉。
 * 打桩的数据形状照 `mast/api/schemas_admin.py` 的真模型写。
 */

const CONV = "conv-e2e-injected";
const T_USER = 1_755_000_000;      // 用户消息的时间戳（秒）
const SEQ = 42;

/** 把这条用例用到的每一个后端接口都钉住。 */
async function stubApi(page: import("@playwright/test").Page) {
  const json = (body: unknown) => ({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

  // ⚠️ **兜底必须先注册。** Playwright 的路由是后注册的优先匹配 —— 把
  // `**/api/**` 放在最后会把下面每一个具体桩全盖住，而症状是「页面渲染了，
  // 就是没有数据」，和组件没接上长得一模一样（第一版就栽在这儿）。
  //
  // 不能回 `{}`：一个空对象在类型上说得通，跑起来却让别的面板在
  // `undefined.toFixed()` 上炸掉，整页换成 React 的错误页。503 让它们各自进
  // 降级态（本仓硬规则：界面绝不冻结），于是这条用例失败时一定是本组件的问题。
  await page.route("**/api/**", (r) =>
    r.fulfill({ status: 503, contentType: "application/json", body: "{}" }));

  // 字段名照 openapi.json 的真模型（ConversationsResponse / MessagesResponse）。
  // 编一套「看起来对」的字段名，页面会安静地当成空表 —— 这正是它要验的坑。
  await page.route("**/api/agents/*/conversations", (r) =>
    r.fulfill(json({
      conversations: [{
        agent_id: "instrument_control", archived: false, conversation_id: CONV,
        created_at: T_USER, kind: "private", last_message_preview: "好的。",
        thread_id: CONV, title: "注入上下文验收", updated_at: T_USER,
      }],
      count: 1, degraded: false,
    })));

  await page.route("**/api/agents/*/messages**", (r) =>
    r.fulfill(json({
      conversation_id: CONV,
      messages: [
        { role: "user", content: "扫一张 100 nm 的图", t: T_USER },
        { role: "assistant", content: "好的。", t: T_USER + 3 },
      ],
      count: 2, degraded: false,
    })));

  await page.route("**/api/admin/prompt-capture", (r) =>
    r.fulfill(json({
      items: [{
        index: 0, seq: SEQ, ts: T_USER + 1, age_s: 5,
        source: "instrument_control", model_id: "kimi-k3", provider: "moonshot",
        message_count: 3, total_chars: 9000, system_chars: 7000,
      }],
      count: 1, enabled: true, total_seen: 1, capacity: 40, note: "", degraded: false,
    })));

  await page.route(`**/api/admin/prompt-capture/by-seq/${SEQ}`, (r) =>
    r.fulfill(json({
      index: 0, seq: SEQ, ts: T_USER + 1, age_s: 5,
      source: "instrument_control", model_id: "kimi-k3", provider: "moonshot",
      messages: [
        { role: "system", content: "【安全边界】偏压上限 10 V。", chars: 20, truncated: false },
        { role: "human", content: "扫一张 100 nm 的图", chars: 16, truncated: false },
        { role: "ai", content: "好的。", chars: 3, truncated: false },
      ],
      total_chars: 9000, dropped_messages: 0, found: true, degraded: false,
    })));
}

test("用户消息下面有折叠按钮，点开显示这一轮真正注入的 system 块", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  await stubApi(page);
  await page.goto("/chat");

  const toggle = page.getByRole("button", { name: /注入的上下文/ });
  await expect(toggle.first()).toBeVisible();

  // 折叠着的时候**不许**已经把内容画出来 —— 那样「折叠」只是视觉上的。
  await expect(page.getByText("【安全边界】偏压上限")).toHaveCount(0);

  await toggle.first().click();

  // 注入正文
  await expect(page.getByText("【安全边界】偏压上限")).toBeVisible();
  // 这一轮发了几次调用、注入多大 —— 折叠块的抬头
  await expect(page.getByText(/这一轮发出 1 次模型调用/)).toBeVisible();
  // 边界要说出来：工具定义不在这份记录里
  await expect(page.getByText(/不走消息/)).toBeVisible();
  // 对话历史与注入分开放：历史默认再折一层
  await expect(page.getByText(/同时带过去的对话历史 2 条/)).toBeVisible();

  // 再点一次收起来
  await toggle.first().click();
  await expect(page.getByText("【安全边界】偏压上限")).toHaveCount(0);

  expect(errors, `页面报错：${errors.join("\n")}`).toHaveLength(0);
});

test("对不上的时候说清是哪一种对不上，而不是一句「暂无数据」", async ({ page }) => {
  await stubApi(page);
  // 捕获被关掉 —— 和「没发生过模型调用」是两件事，指向的动作也不同。
  await page.route("**/api/admin/prompt-capture", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [], count: 0, enabled: false, total_seen: 0,
        capacity: 40, note: "", degraded: false,
      }),
    }));

  await page.goto("/chat");
  await page.getByRole("button", { name: /注入的上下文/ }).first().click();
  await expect(page.getByText(/已关闭/)).toBeVisible();
});
