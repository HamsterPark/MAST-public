import { defineConfig, devices } from "@playwright/test";

// E2E config. Full E2E needs BOTH the API (python -m mast.api, ideally
// MAST_API_LIVE=1) and the SPA running. `webServer` here starts the Vite preview
// of the built SPA; point MAST_API_TARGET at a running API (defaults to :7870 via
// the vite proxy in dev, or set PW_BASE_URL). Run:
//   npm run build && npx playwright install chromium && npm run test:e2e
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: true,
  reporter: [["list"]],
  use: {
    baseURL: process.env.PW_BASE_URL ?? "http://localhost:4173",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run preview -- --port 4173",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
