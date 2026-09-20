// Screenshot the current MAST SPA for the design-materials package.
// Assumes a server is serving the SPA + /api at BASE (default the standalone
// degraded API on :7870, which mounts frontend/dist at "/"). Captures each route
// in dark (default) + a light variant of two key pages.
//
// Run:  node frontend/scripts/shoot.mjs
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const BASE = process.env.SHOOT_BASE ?? "http://127.0.0.1:7870";
const OUT = "design-materials/screenshots";
mkdirSync(OUT, { recursive: true });

// New IA: 12 top-tabs mirroring the old Gradio order (仪器 Chat is home).
const ROUTES = [
  ["chat", "/"], ["agents", "/agents"], ["qa", "/qa"], ["skills", "/skills"],
  ["builder", "/builder"], ["cognition", "/cognition"], ["literature", "/literature"],
  ["records", "/records"], ["wishlist", "/wishlist"], ["admin", "/admin"],
  ["settings", "/settings"], ["experimental", "/experimental"],
];

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();

async function shoot(name, path, theme) {
  await page.goto(BASE + path, { waitUntil: "networkidle", timeout: 25000 }).catch(() => {});
  // ensure theme class on <html>
  await page.evaluate((t) => document.documentElement.classList.toggle("dark", t === "dark"), theme);
  // let live data (TanStack Query: readings poll, lists, charts) settle
  await page.waitForTimeout(3500);
  const file = `${OUT}/${name}-${theme}.png`;
  await page.screenshot({ path: file, fullPage: true }).catch((e) => console.log("shoot fail", name, e.message));
  console.log("shot", file);
}

for (const [name, path] of ROUTES) await shoot(name, path, "dark");
// a couple of light-theme references
for (const [name, path] of [["chat", "/"], ["settings", "/settings"], ["records", "/records"]])
  await shoot(name, path, "light");

await browser.close();
console.log("done");
