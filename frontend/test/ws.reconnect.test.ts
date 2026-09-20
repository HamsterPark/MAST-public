// ════════════════════════════════════════════════════════════════════════════
// Transport tests for src/lib/ws.ts — reconnect, catch-up, fallback.
//
// NO TEST FRAMEWORK IS INSTALLED in this frontend (no vitest, no jest), and
// adding one is a dependency decision that is not ours to make. These run on
// `node --test` with Node's native TypeScript stripping — zero new packages:
//
//     cd frontend && npm run test:unit
//
// What that buys, and what it does not: this exercises the real singleton by
// substituting a fake `WebSocket` and a fake `location`, so refcounting, the
// backoff schedule, the `?since=` catch-up, the idle watchdog and the
// poll/no-poll decision are all covered as CODE. It cannot cover React binding
// or rendering — that is what the Playwright spec (e2e/ws-live-readings.spec.ts)
// is for.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { after, before, beforeEach, describe, it, mock } from "node:test";

import {
  WS_BACKOFF_MAX_MS,
  WS_FAIL_AFTER_ATTEMPTS,
  WS_IDLE_MS,
  backoffDelayMs,
  closeSocket,
  eventsUrl,
  getLastSeq,
  getStatus,
  nextLastSeq,
  parseFrame,
  shouldPoll,
  shouldWarn,
  subscribe,
  type WsEvent,
} from "../src/lib/ws.ts";

// ── a WebSocket we can drive ────────────────────────────────────────────────

class FakeSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  /** Every socket the module has constructed, in order — the reconnect trail. */
  static instances: FakeSocket[] = [];

  readyState = 0;
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeSocket.instances.push(this);
  }

  // -- driven by the test --
  accept(): void {
    this.readyState = 1;
    this.onopen?.();
  }
  deliver(payload: unknown): void {
    this.onmessage?.({ data: typeof payload === "string" ? payload : JSON.stringify(payload) });
  }
  drop(): void {
    this.readyState = 3;
    this.onclose?.();
  }
  close(): void {
    this.readyState = 3;
  }
}

const g = globalThis as unknown as Record<string, unknown>;
let savedWebSocket: unknown;
let savedLocation: unknown;

before(() => {
  savedWebSocket = g.WebSocket;
  savedLocation = g.location;
  g.WebSocket = FakeSocket;
  g.location = { protocol: "http:", host: "localhost:5173" };
});

after(() => {
  g.WebSocket = savedWebSocket;
  g.location = savedLocation;
});

// `closeSocket()` deliberately does NOT drop subscribers — that is what lets a
// reconnect resume delivering to the same consumers. So the harness has to
// unsubscribe them itself, or the refcount from one test keeps the next test's
// `subscribe()` from being the FIRST one and no socket is ever opened.
const subs: Array<() => void> = [];

function sub(type: string, cb: (event: WsEvent) => void): () => void {
  const off = subscribe(type, cb);
  subs.push(off);
  return off;
}

beforeEach(() => {
  while (subs.length) subs.pop()?.();
  closeSocket();
  FakeSocket.instances = [];
});

/** Latest socket the module opened. */
function live(): FakeSocket {
  const s = FakeSocket.instances.at(-1);
  assert.ok(s, "expected the module to have opened a socket");
  return s;
}

// ════════════════════════════════════════════════════════════════════════════
// Pure decisions
// ════════════════════════════════════════════════════════════════════════════

describe("backoff", () => {
  it("grows exponentially and is capped", () => {
    const worst = (n: number) => backoffDelayMs(n, { rng: () => 1 });
    assert.equal(worst(0), 500);
    assert.equal(worst(1), 1000);
    assert.equal(worst(2), 2000);
    assert.equal(worst(6), WS_BACKOFF_MAX_MS);
    // Still capped an hour into an outage — a laptop left open overnight must
    // still find the server within the cap, not in 2^40 ms.
    assert.equal(worst(40), WS_BACKOFF_MAX_MS);
    assert.equal(worst(4000), WS_BACKOFF_MAX_MS);
  });

  it("jitters across the whole window", () => {
    // Full jitter: every tab retrying in lockstep after a shared wake-up is the
    // failure this prevents, so the floor must really be 0, not base.
    assert.equal(backoffDelayMs(3, { rng: () => 0 }), 0);
    assert.equal(backoffDelayMs(3, { rng: () => 0.5 }), 2000);
    assert.equal(backoffDelayMs(3, { rng: () => 1 }), 4000);
  });

  it("treats a negative or fractional attempt as attempt 0", () => {
    assert.equal(backoffDelayMs(-5, { rng: () => 1 }), 500);
    assert.equal(backoffDelayMs(0.9, { rng: () => 1 }), 500);
  });
});

describe("fallback policy", () => {
  it("polls in every state except open", () => {
    assert.equal(shouldPoll("open"), false);
    for (const s of ["idle", "connecting", "reconnecting", "failed"] as const) {
      assert.equal(shouldPoll(s), true, `${s} must keep polling`);
    }
  });

  it("warns only once push is presumed gone", () => {
    assert.equal(shouldWarn("failed"), true);
    // A 400 ms blip must not flash a banner.
    for (const s of ["idle", "connecting", "open", "reconnecting"] as const) {
      assert.equal(shouldWarn(s), false, `${s} must not warn`);
    }
  });
});

describe("seq bookkeeping", () => {
  it("advances on newer events", () => {
    assert.equal(nextLastSeq(0, 1), 1);
    assert.equal(nextLastSeq(7, 8), 8);
  });

  it("adopts a LOWER seq as a server restart", () => {
    // The bug this exists to prevent: a plain max() pins lastSeq at the old
    // high-water mark, we reconnect with ?since=<huge>, and the client sits
    // there connected and permanently empty.
    assert.equal(nextLastSeq(9000, 3), 3);
  });

  it("ignores junk", () => {
    assert.equal(nextLastSeq(5, Number.NaN), 5);
    assert.equal(nextLastSeq(5, -1), 5);
  });
});

describe("frame parsing", () => {
  it("decodes an event", () => {
    const f = parseFrame('{"kind":"event","seq":12,"type":"hardware_state","data":{"bias_v":1.5}}');
    assert.equal(f.kind, "event");
    if (f.kind !== "event") return;
    assert.equal(f.event.seq, 12);
    assert.equal(f.event.type, "hardware_state");
    assert.deepEqual(f.event.data, { bias_v: 1.5 });
  });

  it("recognises the heartbeat", () => {
    assert.equal(parseFrame('{"kind":"ping"}').kind, "ping");
  });

  it("ignores rather than throws on anything unexpected", () => {
    // The contract says `type` is an open set. A backend that ships a new event
    // type must not break every client that predates it.
    for (const raw of [
      "not json at all",
      "null",
      "[1,2,3]",
      '{"kind":"something_new_in_2027","payload":1}',
      '{"kind":"event","seq":1}', // no type
      '{"kind":"event","type":123}', // type not a string
      "",
    ]) {
      const f = parseFrame(raw);
      assert.equal(f.kind, "ignored", `should ignore: ${raw}`);
    }
  });

  it("keeps an event whose seq is missing, marking it -1", () => {
    // Losing the payload would be worse than losing the ordering.
    const f = parseFrame('{"kind":"event","type":"anomaly","data":{"x":1}}');
    assert.equal(f.kind, "event");
    if (f.kind !== "event") return;
    assert.equal(f.event.seq, -1);
  });
});

describe("connect URL", () => {
  const loc = { protocol: "http:", host: "lab:7870" };

  it("omits ?since on a cold start", () => {
    assert.equal(eventsUrl(0, loc), "ws://lab:7870/ws/events");
  });

  it("asks for the catch-up on a reconnect", () => {
    assert.equal(eventsUrl(42, loc), "ws://lab:7870/ws/events?since=42");
  });

  it("follows the page to wss", () => {
    assert.equal(
      eventsUrl(1, { protocol: "https:", host: "box.tailnet.ts.net" }),
      "wss://box.tailnet.ts.net/ws/events?since=1",
    );
  });
});

// ════════════════════════════════════════════════════════════════════════════
// The singleton, driven through a fake socket
// ════════════════════════════════════════════════════════════════════════════

describe("connection lifecycle", () => {
  it("opens once for many subscribers and closes when the last leaves", () => {
    const a = sub("hardware_state", () => {});
    const b = sub("anomaly", () => {});
    assert.equal(FakeSocket.instances.length, 1, "one socket, not one per consumer");
    assert.equal(getStatus(), "connecting");

    live().accept();
    assert.equal(getStatus(), "open");

    a();
    assert.equal(getStatus(), "open", "still has a subscriber");
    b();
    assert.equal(getStatus(), "idle", "last unsubscribe closes it");
  });

  it("survives an unsubscribe being called twice (StrictMode)", () => {
    const off = sub("hardware_state", () => {});
    live().accept();
    off();
    off();
    assert.equal(getStatus(), "idle");
  });

  it("routes by type and never lets one listener's throw stop the others", () => {
    const seen: string[] = [];
    sub("hardware_state", () => {
      throw new Error("consumer bug");
    });
    sub("hardware_state", () => seen.push("hw"));
    sub("*", (e: WsEvent) => seen.push(`all:${e.type}`));
    sub("anomaly", () => seen.push("anomaly"));

    live().accept();
    live().deliver({ kind: "event", seq: 1, type: "hardware_state", data: {} });
    live().deliver({ kind: "event", seq: 2, type: "scan_complete", data: {} });
    // An event type nobody has ever heard of: delivered to "*", ignored otherwise.
    live().deliver({ kind: "event", seq: 3, type: "invented_next_year", data: {} });

    assert.deepEqual(seen, ["hw", "all:hardware_state", "all:scan_complete", "all:invented_next_year"]);
    assert.equal(getLastSeq(), 3);
  });

  it("ignores a ping for data purposes but still counts as traffic", () => {
    let events = 0;
    sub("*", () => (events += 1));
    live().accept();
    // NB `lastSeq` is module state that deliberately OUTLIVES a close — a
    // re-subscribe must resume from where it left off — so assert that the ping
    // left it alone rather than that it is zero.
    const before = getLastSeq();
    live().deliver({ kind: "ping" });
    assert.equal(events, 0, "a heartbeat is not an event");
    assert.equal(getLastSeq(), before, "a heartbeat must not advance the catch-up cursor");
  });
});

describe("reconnect", () => {
  beforeEach(() => {
    mock.timers.enable({ apis: ["setTimeout"] });
  });

  const cleanup = () => mock.timers.reset();

  it("retries after a drop and asks for what it missed", () => {
    try {
      sub("hardware_state", () => {});
      live().accept();
      live().deliver({ kind: "event", seq: 17, type: "hardware_state", data: { bias_v: 1 } });

      live().drop();
      assert.equal(getStatus(), "reconnecting");
      assert.equal(FakeSocket.instances.length, 1, "backoff first, not an instant hammer");

      mock.timers.tick(WS_BACKOFF_MAX_MS + 1);
      assert.equal(FakeSocket.instances.length, 2);
      assert.match(live().url, /\?since=17$/, "must resume from the last seq it saw");
    } finally {
      cleanup();
    }
  });

  it("escalates to failed after repeated failures, and keeps trying anyway", () => {
    try {
      sub("hardware_state", () => {});
      for (let i = 0; i < WS_FAIL_AFTER_ATTEMPTS; i += 1) {
        live().drop();
        mock.timers.tick(WS_BACKOFF_MAX_MS + 1);
      }
      assert.equal(getStatus(), "failed");
      assert.equal(shouldPoll(getStatus()), true, "failed MUST fall back to polling");
      assert.equal(shouldWarn(getStatus()), true, "and MUST tell the operator");

      // "failed" is a hint, not a surrender: the socket count keeps climbing.
      const before = FakeSocket.instances.length;
      live().drop();
      mock.timers.tick(WS_BACKOFF_MAX_MS + 1);
      assert.equal(FakeSocket.instances.length, before + 1);
    } finally {
      cleanup();
    }
  });

  it("clears the failure once a connection sticks", () => {
    try {
      sub("hardware_state", () => {});
      for (let i = 0; i < WS_FAIL_AFTER_ATTEMPTS; i += 1) {
        live().drop();
        mock.timers.tick(WS_BACKOFF_MAX_MS + 1);
      }
      assert.equal(getStatus(), "failed");
      live().accept();
      assert.equal(getStatus(), "open");
      assert.equal(shouldPoll(getStatus()), false, "push is back, poll may de-rate");
    } finally {
      cleanup();
    }
  });

  it("stops retrying once nobody is listening", () => {
    try {
      const off = sub("hardware_state", () => {});
      live().accept();
      off();
      const before = FakeSocket.instances.length;
      mock.timers.tick(WS_BACKOFF_MAX_MS * 4);
      assert.equal(FakeSocket.instances.length, before, "an unmounted app must not keep dialling");
      assert.equal(getStatus(), "idle");
    } finally {
      cleanup();
    }
  });

  it("reconnects a socket that went silent without closing", () => {
    try {
      sub("hardware_state", () => {});
      live().accept();
      const dead = live();

      // A NAT rebind / Tailscale re-route leaves a half-open socket: no close
      // event ever arrives, so without the watchdog the UI waits forever on a
      // connection that will never deliver again.
      mock.timers.tick(WS_IDLE_MS + 1);
      assert.equal(FakeSocket.instances.length, 2, "watchdog must force a new socket");
      assert.notEqual(live(), dead);
    } finally {
      cleanup();
    }
  });

  it("does not fire the watchdog while pings keep arriving", () => {
    try {
      sub("hardware_state", () => {});
      live().accept();
      // Server heartbeats every ~15 s; three beats must not trip a 45 s budget.
      for (let i = 0; i < 3; i += 1) {
        mock.timers.tick(15_000);
        live().deliver({ kind: "ping" });
      }
      assert.equal(FakeSocket.instances.length, 1, "a healthy link must not be reconnected");
      assert.equal(getStatus(), "open");
    } finally {
      cleanup();
    }
  });
});
