import { test, expect } from "@playwright/test";

// True-parallel background-runs panel smoke. Mirrors smoke.spec.ts: asserts the
// 后台 sub-tab mounts and the panel renders its launcher WITHOUT a live backend
// (the list falls back to a loading/empty/degraded note — the freeze-proof
// behaviour we guarantee). Data-dependent assertions are gated on API presence.

test("agents page exposes the 后台 (background runs) sub-tab", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto("/agents");
  // the new sub-tab exists (exact so it doesn't match 「后台启动」 / 「后台任务」)
  const tab = page.getByRole("button", { name: "后台", exact: true });
  await expect(tab).toBeVisible();
  await tab.click();

  // panel header + the launcher control render API-less (they are static cards)
  await expect(page.getByText("后台任务 · 真并行")).toBeVisible();
  await expect(page.getByRole("button", { name: "后台启动" })).toBeVisible();
  // the conservative auto-background toggle is exposed
  await expect(page.getByText("自动后台化（保守 · 实验性）")).toBeVisible();
  // the instruction box is present and the launch button is disabled while empty
  await expect(page.getByPlaceholder(/综述|独立/)).toBeVisible();

  // no uncaught render crash (the effect_update_depth class of bug)
  expect(errors, `unexpected page errors: ${errors.join("\n")}`).toHaveLength(0);
});

test("background launcher enables only with an instruction", async ({ page }) => {
  await page.goto("/agents");
  await page.getByRole("button", { name: "后台", exact: true }).click();

  const launch = page.getByRole("button", { name: "后台启动" });
  await expect(launch).toBeDisabled(); // empty instruction → disabled

  await page.getByPlaceholder(/综述|独立/).fill("综述 Au(111) 上的单分子磁体 STM 研究");
  await expect(launch).toBeEnabled();
});
