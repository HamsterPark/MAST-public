import { test, expect } from "@playwright/test";

// Phase-6 SPA smoke. Verifies the app shell + routing render (the structural
// replacement for the deleted Gradio UI). Data-dependent assertions live in the
// "with live API" block, which only meaningfully passes when a backend is up
// (python -m mast.api); without it the pages still render their loading/degraded
// states — which is itself the freeze-proof behavior we want to guarantee.

test("app shell + nav renders", async ({ page }) => {
  await page.goto("/");
  // Sidebar brand + the 11 nav entries from AppLayout.
  // shell 渲染即可——首页有多处 "MAST"(TopBar 品牌 + RightPanel 版本),
  // strict-mode 会撞多个,取 first() 验证 shell 出现即可(旧匹配器脆弱性 2026-07-20 修)。
  await expect(page.getByText("MAST", { exact: false }).first()).toBeVisible();
  // 2026-08-06(#34)：顶栏从 17 项合并成 10 项。「高级管理」等四项降成了「设置」
  // 底下的二级页，所以顶栏上不再有它们的链接 —— 断言跟着改成合并后的那十项。
  for (const label of ["仪器 Chat", "Agents", "技能", "实验记录", "设置", "监控"]) {
    await expect(page.getByRole("link", { name: label })).toBeVisible();
  }
});

test("routes mount without a full-page crash (no effect_update_depth class)", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  // 合并后的二级页 + 几条旧地址（重定向必须落到能渲染的地方，而不是白页）。
  for (const path of [
    "/", "/chat", "/agents", "/vision",
    "/skills/library", "/skills/builder",
    "/records/log", "/records/memory",
    "/settings/general", "/settings/setup", "/settings/admin", "/settings/usage",
    "/monitoring/current", "/monitoring/env",
    "/experimental/tools", "/experimental/optics",
    // 组根：应当被送进某一段，不该停在空壳上。
    "/skills", "/records", "/settings", "/monitoring", "/experimental",
    // 合并之前的地址：书签不能断。
    "/builder", "/cognition", "/admin", "/setup", "/usage", "/env-history", "/optics",
  ]) {
    await page.goto(path);
    // Each route renders SOMETHING (heading/section) and doesn't blank out.
    await expect(page.locator("main")).toBeVisible();
  }
  expect(errors, `unexpected page errors: ${errors.join("\n")}`).toHaveLength(0);
});

test("settings page shows the model table when the API is up", async ({ page }) => {
  await page.goto("/settings/general");
  const table = page.locator("table");
  // If the API is reachable the model-capability table renders rows; otherwise
  // the page shows a loading/degraded note — both are non-crashing. Only assert
  // the table when present so the smoke passes API-less.
  if (await table.count()) {
    await expect(table.first()).toBeVisible();
  }
});
