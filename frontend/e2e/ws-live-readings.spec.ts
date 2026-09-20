import { test, expect, type Page, type WebSocketRoute } from "@playwright/test";

// ════════════════════════════════════════════════════════════════════════════
// TopBar live readings over /ws/events — push, and the fallback under it.
//
// Operator 2026-07-28: 「上 WebSocket 推送」. The transport is covered headlessly
// in frontend/test/ws.reconnect.test.ts; what only a browser can answer is
// whether the REACT side switched — does a pushed frame reach the DOM, does the
// poll come back when push dies, and does the poll actually de-rate while push
// is healthy.
//
// Self-contained: no backend. `page.routeWebSocket` mocks /ws/events and
// `page.route` mocks the REST snapshot, and the two are made to report DIFFERENT
// bias values on purpose — with a real API both would agree and the assertions
// would pass no matter which source was on screen.
//
// ── Why /settings with the rail collapsed ──────────────────────────────────
// TopBar, RightPanel and DashboardPage share the ["hardware","live-readings"]
// cache entry, and react-query gives EACH observer its own refetch timer. While
// RightPanel is mounted it polls that key every 500 ms and overwrites whatever
// the push just wrote, twice a second — so on a page where it is mounted, no
// assertion can distinguish push from poll. (Found the hard way: the first
// version of this spec passed against a design that turned out not to work.)
// /settings has no readings query of its own, and collapsing the right rail
// unmounts RightPanel, leaving TopBar as the only observer. That isolation is
// the test rig; it is also exactly the state of the app once the other two views
// switch to push as well.
//
//   npm run build && npm run test:e2e -- ws-live-readings
// ════════════════════════════════════════════════════════════════════════════

const POLLED_BIAS = 0.1234;
const PUSHED_BIAS = 2.5678;

/** Unmount RightPanel so TopBar owns the shared readings cache alone. */
async function isolateTopBar(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem("mast.rightRail.collapsed", "1");
  });
}

/** The REST snapshot. Counts its own calls so the poll rate is measurable. */
async function mockPolling(page: Page): Promise<{ calls: () => number }> {
  let calls = 0;
  await page.route("**/api/hardware/live-readings", async (route) => {
    calls += 1;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        readings: {
          bias_v: POLLED_BIAS,
          current_a: 1e-10,
          z_m: 1e-8,
          setpoint_a: 1e-10,
          z_controller_on: true,
          scan_running: false,
          stale: false,
        },
        bias_history: [1, 2, 3],
        current_history: [],
        z_history: [],
        degraded: false,
        connected: true,
      }),
    });
  });
  return { calls: () => calls };
}

/**
 * Mock /ws/events and hand back a handle the test can drive.
 *
 * `refuseReconnects()` matters more than it looks: the client reconnects within
 * a few hundred ms, and `routeWebSocket` accepts every attempt, so a plain
 * `close()` is a blip the app recovers from rather than an outage. Testing the
 * fallback needs the server to STAY gone.
 */
async function mockSocket(page: Page): Promise<{
  server: () => WebSocketRoute;
  refuseReconnects: () => void;
}> {
  let server: WebSocketRoute | null = null;
  let refusing = false;
  await page.routeWebSocket("**/ws/events**", (ws) => {
    if (refusing) {
      ws.close({ code: 1006 });
      return;
    }
    server = ws; // this mock IS the server — never connect upstream
  });
  return {
    server: () => {
      if (!server) throw new Error("the app never opened /ws/events");
      return server;
    },
    refuseReconnects: () => {
      refusing = true;
    },
  };
}

const bias = (page: Page) => page.getByTestId("reading-bias");
const header = (page: Page) => page.locator("header[data-ws-status]");

function hardwareState(seq: number, bias_v: number) {
  return JSON.stringify({
    kind: "event",
    seq,
    type: "hardware_state",
    data: { bias_v, stale: false },
    ts: Date.now(),
  });
}

test("a pushed hardware_state frame reaches the top bar and stays there", async ({ page }) => {
  await isolateTopBar(page);
  await mockPolling(page);
  const { server } = await mockSocket(page);

  await page.goto("/settings");

  // The poll answers first — that value must be on screen before any push.
  await expect(bias(page)).toHaveText(POLLED_BIAS.toFixed(4));
  await expect(header(page)).toHaveAttribute("data-ws-status", "open");

  server().send(hardwareState(1, PUSHED_BIAS));

  await expect(bias(page)).toHaveText(PUSHED_BIAS.toFixed(4));
  await expect(header(page)).toHaveAttribute("data-readings-source", "ws");

  // STAYS. Without the poll de-rating, the 500 ms fetch would drag the display
  // back to POLLED_BIAS almost immediately — which is what the first attempt at
  // this feature actually did.
  await page.waitForTimeout(3000);
  await expect(bias(page)).toHaveText(PUSHED_BIAS.toFixed(4));
});

test("push de-rates the poll instead of leaving it at 500 ms", async ({ page }) => {
  await isolateTopBar(page);
  const rest = await mockPolling(page);
  const { server } = await mockSocket(page);

  await page.goto("/settings");
  await expect(bias(page)).toHaveText(POLLED_BIAS.toFixed(4));
  await expect(header(page)).toHaveAttribute("data-ws-status", "open");

  server().send(hardwareState(1, PUSHED_BIAS));
  await expect(bias(page)).toHaveText(PUSHED_BIAS.toFixed(4));

  const before = rest.calls();
  await page.waitForTimeout(4000);
  const during = rest.calls() - before;
  // 4 s at the 500 ms rate would be ~8 requests; at the 10 s keep-alive it is
  // 0 or 1. This is the traffic saving the whole change is for.
  expect(during, `expected the poll to de-rate, saw ${during} requests in 4 s`).toBeLessThanOrEqual(2);
});

test("unknown event types and junk frames do not break the socket", async ({ page }) => {
  await isolateTopBar(page);
  await mockPolling(page);
  const { server } = await mockSocket(page);

  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto("/settings");
  await expect(header(page)).toHaveAttribute("data-ws-status", "open");

  // The contract says `type` is an open set that will grow. A client that
  // predates a new type must ignore it, not fall over.
  server().send(JSON.stringify({ kind: "event", seq: 1, type: "invented_next_year", data: {} }));
  server().send(JSON.stringify({ kind: "ping" }));
  server().send("this is not json");
  server().send(JSON.stringify({ kind: "event", seq: 2, type: "hardware_state", data: null }));
  server().send(hardwareState(3, PUSHED_BIAS));

  // Still alive, still delivering.
  await expect(bias(page)).toHaveText(PUSHED_BIAS.toFixed(4));
  expect(errors, `unexpected page errors: ${errors.join("\n")}`).toHaveLength(0);
});

test("when the socket closes the poll takes the readout back", async ({ page }) => {
  await isolateTopBar(page);
  await mockPolling(page);
  const { server, refuseReconnects } = await mockSocket(page);

  await page.goto("/settings");
  await expect(header(page)).toHaveAttribute("data-ws-status", "open");
  server().send(hardwareState(5, PUSHED_BIAS));
  await expect(bias(page)).toHaveText(PUSHED_BIAS.toFixed(4));

  // A Tailscale roam that does close the connection.
  refuseReconnects();
  server().close();

  // 项目铁律「UI 绝不冻结」: the readout must not sit on the last pushed value.
  // The 500 ms poll resumes the moment the status leaves "open".
  await expect(bias(page)).toHaveText(POLLED_BIAS.toFixed(4), { timeout: 10_000 });
  await expect(header(page)).toHaveAttribute("data-readings-source", "poll");
});

test("a socket that goes SILENT without closing does not freeze the readout", async ({ page }) => {
  // The failure the keep-alive poll exists for, and the one a plain
  // "WS open → stop polling" switch gets wrong. A connection killed by a NAT
  // rebind is not closed — it just stops delivering — so `onclose` never fires
  // and the client believes it is live for up to WS_IDLE_MS (45 s). With polling
  // switched fully off that is 45 s of frozen numbers that look current, on the
  // readout used to judge whether a tip is about to crash.
  test.setTimeout(60_000);

  await isolateTopBar(page);
  await mockPolling(page);
  const { server } = await mockSocket(page);

  await page.goto("/settings");
  await expect(header(page)).toHaveAttribute("data-ws-status", "open");
  server().send(hardwareState(9, PUSHED_BIAS));
  await expect(bias(page)).toHaveText(PUSHED_BIAS.toFixed(4));

  // …and now nothing. No close, no ping, no event.
  // The 10 s keep-alive poll must put a real reading back on screen well before
  // the 45 s idle watchdog notices anything is wrong.
  await expect(bias(page)).toHaveText(POLLED_BIAS.toFixed(4), { timeout: 25_000 });
  // Still nominally connected — the point being that recovery did NOT depend on
  // the socket admitting it was dead.
  await expect(header(page)).toHaveAttribute("data-ws-status", "open");
});

test("a socket that never connects leaves the bar polling, then says so", async ({ page }) => {
  await isolateTopBar(page);
  await mockPolling(page);

  // Refuse every upgrade — "backend down" / "proxy strips the upgrade".
  await page.routeWebSocket("**/ws/events**", (ws) => {
    ws.close({ code: 1006 });
  });

  await page.goto("/settings");

  // Readings keep coming from HTTP throughout. The assertion that matters most:
  // a dead push channel must never mean a dead readout.
  await expect(bias(page)).toHaveText(POLLED_BIAS.toFixed(4));

  // …and once the retries have clearly failed, the operator is TOLD, rather than
  // left wondering why it feels slow over Tailscale.
  await expect(page.getByTestId("ws-degraded")).toBeVisible({ timeout: 60_000 });
  await expect(header(page)).toHaveAttribute("data-ws-status", "failed");
});
