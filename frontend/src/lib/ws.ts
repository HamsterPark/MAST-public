// ════════════════════════════════════════════════════════════════════════════
// ws.ts — the ONE WebSocket the app keeps open to /ws/events.
//
// This module moves live readings onto WebSocket push. Tailscale is what makes push worth it: the box is
// reachable from anywhere, so the UI no longer has to poll a LAN address it can
// only see from the lab.
//
// It is also what makes reconnection non-optional. The same link roams between
// Wi-Fi / cellular / sleep, and a WireGuard re-route kills the TCP connection
// underneath a live socket. That is the same failure sse.ts was written for,
// and this module takes the same position:
//
//   a transport that dies quietly is worse than one that never connected,
//   because the screen keeps showing numbers that stopped being true.
//
// So every path here ends in a state the consumer can act on, and the consumer
// is always told whether to fall back to polling. 项目铁律「UI 绝不冻结」 means
// specifically: there is no combination of network failures that leaves the
// readings frozen with no polling and no notice.
//
// ── Contract (server side: MASTv2/mast/api/ws.py) ──────────────────────────
//   GET /ws/events?since=<seq>          WebSocket upgrade
//   ← {"kind":"event","seq":123,"type":"hardware_state","data":{…},"ts":…}
//   ← {"kind":"ping"}                   heartbeat, ~15 s
//   `seq` is monotonic; on reconnect we send the last seq we saw and the server
//   replays everything after it (EventBus keeps the most recent 100 events for
//   exactly this). `type` is an open set — an unknown one must be ignored, never
//   thrown on, or one new backend event type breaks every old client.
//
// ── Layering ───────────────────────────────────────────────────────────────
// Pure transport. No React, no react-query, no knowledge of what any event
// means. The decision functions (backoff, fallback, seq bookkeeping, frame
// parsing) are exported separately BECAUSE they are the parts worth testing
// without a browser — see frontend/test/ws.reconnect.test.ts.
// React bindings live in src/hooks/useWsEvents.ts.
// ════════════════════════════════════════════════════════════════════════════

/** Connection states a consumer may need to render or branch on. */
export type WsStatus =
  /** No subscribers — nothing is open, by design. */
  | "idle"
  /** First connection attempt in flight. */
  | "connecting"
  /** Live. This is the ONLY state in which polling may be relaxed. */
  | "open"
  /** Dropped; a retry is scheduled. Data may be stale RIGHT NOW. */
  | "reconnecting"
  /** Enough consecutive failures that push should be presumed unavailable.
   *  Retries CONTINUE — this is a hint to the UI, not a surrender. */
  | "failed";

/** A decoded server event. `data` is deliberately `unknown`: this module does
 *  not know any event's payload shape, and consumers must narrow it. */
export interface WsEvent {
  seq: number;
  type: string;
  data: unknown;
  ts?: number;
}

/** What one raw text frame turned out to be. `ignored` covers malformed JSON,
 *  unknown `kind`, and events missing the fields we need — all non-fatal by
 *  contract. */
export type ParsedFrame =
  | { kind: "event"; event: WsEvent }
  | { kind: "ping" }
  | { kind: "ignored"; why: string };

// ── tunables ────────────────────────────────────────────────────────────────

/** First retry delay. Short enough that a blip is invisible to the operator. */
export const WS_BACKOFF_BASE_MS = 500;

/** Ceiling on the retry delay. A UI that has been disconnected for an hour must
 *  still notice the server coming back within ~30 s, and 30 s of idle socket
 *  costs the box nothing. */
export const WS_BACKOFF_MAX_MS = 30_000;

/**
 * Silence budget before we call the socket dead and reconnect.
 *
 * The server heartbeats every ~15 s (`_PING_TIMEOUT_S` in api/ws.py), so this is
 * 3× the beat. sse.ts uses 5× for its stream, and the difference is deliberate:
 * a false positive there aborts a live agent turn, whereas here it costs one
 * reconnect that replays anything missed via `?since=`. Cheap to be wrong, so
 * we are allowed to be twice as suspicious.
 */
export const WS_IDLE_MS = 45_000;

/** Consecutive failures before the status escalates to `failed`. Six attempts
 *  is where the backoff reaches its ceiling — i.e. we say "presume unavailable"
 *  at the same moment we stop trying harder. */
export const WS_FAIL_AFTER_ATTEMPTS = 6;

// ════════════════════════════════════════════════════════════════════════════
// Pure decision functions — no sockets, no globals, no time. Tested directly.
// ════════════════════════════════════════════════════════════════════════════

/**
 * Retry delay for the Nth consecutive failure (`attempt` starts at 0).
 *
 * Exponential with FULL jitter: `random(0, min(cap, base·2^n))`. Jitter is not
 * decoration — several tabs on one operator's laptop plus RightPanel/Dashboard
 * all wake from sleep at the same instant, and un-jittered backoff makes them
 * retry in lockstep forever. `rng` is injectable so the schedule can be asserted.
 */
export function backoffDelayMs(
  attempt: number,
  opts: { base?: number; cap?: number; rng?: () => number } = {},
): number {
  const { base = WS_BACKOFF_BASE_MS, cap = WS_BACKOFF_MAX_MS, rng = Math.random } = opts;
  const n = Math.max(0, Math.floor(attempt));
  // 2**n overflows to Infinity long before it matters; Math.min pins it anyway.
  const ceiling = Math.min(cap, base * 2 ** n);
  return Math.round(ceiling * rng());
}

/**
 * THE fallback decision: may the consumer stop polling?
 *
 * Only `open` earns it. `connecting` and `reconnecting` deliberately keep
 * polling — the whole point is that the gap between "socket died" and "socket
 * noticed" is covered by HTTP, not by a frozen number. Written as a function
 * (rather than `status === "open"` at each call site) so there is exactly one
 * place this policy can be got wrong.
 */
export function shouldPoll(status: WsStatus): boolean {
  return status !== "open";
}

/**
 * Whether the operator should be TOLD that live push is unavailable.
 *
 * Distinct from `shouldPoll`: a 400 ms blip during a reconnect is not worth a
 * banner, but a socket that has failed six times running is — the operator is
 * often on the other end of a Tailscale link and "why is this not updating" is
 * a question the UI should answer before they ask it.
 */
export function shouldWarn(status: WsStatus): boolean {
  return status === "failed";
}

/**
 * Next value of `lastSeq` given what just arrived.
 *
 * Normally a max(). The case that matters is the server RESTARTING: its seq
 * counter resets to a small number, and a plain max() would pin `lastSeq` at the
 * old high-water mark forever — we would then reconnect with `?since=<huge>`,
 * the server would have nothing newer, and the client would sit there live and
 * empty. A seq that moves BACKWARD is therefore read as a new server generation
 * and adopted as-is.
 */
export function nextLastSeq(prev: number, incoming: number): number {
  if (!Number.isFinite(incoming) || incoming < 0) return prev;
  if (incoming < prev) return incoming; // server restarted; follow it down
  return incoming;
}

/**
 * Decode one text frame. Never throws — an unparseable or unknown frame is a
 * fact to be ignored, not an error to be propagated. A future `kind` the server
 * adds must not take the socket down.
 */
export function parseFrame(raw: string): ParsedFrame {
  let msg: unknown;
  try {
    msg = JSON.parse(raw);
  } catch {
    return { kind: "ignored", why: "malformed JSON" };
  }
  if (typeof msg !== "object" || msg === null) {
    return { kind: "ignored", why: "not an object" };
  }
  const m = msg as Record<string, unknown>;
  if (m.kind === "ping") return { kind: "ping" };
  if (m.kind !== "event") return { kind: "ignored", why: `unknown kind ${String(m.kind)}` };
  if (typeof m.type !== "string") return { kind: "ignored", why: "event without a type" };
  const seq = typeof m.seq === "number" && Number.isFinite(m.seq) ? m.seq : -1;
  return {
    kind: "event",
    event: {
      seq,
      type: m.type,
      data: m.data,
      ts: typeof m.ts === "number" ? m.ts : undefined,
    },
  };
}

/** Build the connect URL. Same-origin by construction (the SPA is served by the
 *  same FastAPI process in prod, and proxied by Vite in dev), so there is no
 *  host to configure and no CORS to get wrong. `since` is omitted on a cold
 *  start — asking for "everything after 0" would replay the whole retained
 *  history to a client that has no use for it. */
export function eventsUrl(lastSeq: number, loc: { protocol: string; host: string }): string {
  const proto = loc.protocol === "https:" ? "wss" : "ws";
  const base = `${proto}://${loc.host}/ws/events`;
  return lastSeq > 0 ? `${base}?since=${lastSeq}` : base;
}

// ════════════════════════════════════════════════════════════════════════════
// The singleton connection
// ════════════════════════════════════════════════════════════════════════════

type EventListener = (event: WsEvent) => void;

let socket: WebSocket | null = null;
let status: WsStatus = "idle";
let lastSeq = 0;
let attempt = 0;
let retryTimer: ReturnType<typeof setTimeout> | undefined;
let idleTimer: ReturnType<typeof setTimeout> | undefined;
/** Set while we are tearing a socket down on purpose, so its own `onclose`
 *  doesn't schedule a reconnect we didn't ask for. */
let closingDeliberately = false;

/** type → listeners. `"*"` receives every event, whatever its type. */
const listeners = new Map<string, Set<EventListener>>();
const statusListeners = new Set<() => void>();

export const WS_ALL = "*";

function setStatus(next: WsStatus): void {
  if (status === next) return;
  status = next;
  for (const cb of statusListeners) {
    try {
      cb();
    } catch {
      /* a broken status listener must not stop the others */
    }
  }
}

function clearTimers(): void {
  if (retryTimer !== undefined) clearTimeout(retryTimer);
  if (idleTimer !== undefined) clearTimeout(idleTimer);
  retryTimer = undefined;
  idleTimer = undefined;
}

function armIdleWatchdog(): void {
  if (idleTimer !== undefined) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    // A socket killed by a NAT rebind is not closed, it just stops delivering —
    // `onclose` never fires and we would wait forever. Close it ourselves so the
    // normal reconnect path runs.
    forceReconnect("idle");
  }, WS_IDLE_MS);
}

function totalListeners(): number {
  let n = 0;
  for (const set of listeners.values()) n += set.size;
  return n;
}

function scheduleReconnect(): void {
  if (totalListeners() === 0) {
    setStatus("idle");
    return;
  }
  attempt += 1;
  setStatus(attempt >= WS_FAIL_AFTER_ATTEMPTS ? "failed" : "reconnecting");
  const delay = backoffDelayMs(attempt - 1);
  if (retryTimer !== undefined) clearTimeout(retryTimer);
  retryTimer = setTimeout(() => {
    retryTimer = undefined;
    open();
  }, delay);
}

function dispatch(event: WsEvent): void {
  lastSeq = nextLastSeq(lastSeq, event.seq);
  const targeted = listeners.get(event.type);
  const all = listeners.get(WS_ALL);
  for (const set of [targeted, all]) {
    if (!set) continue;
    for (const cb of set) {
      try {
        cb(event);
      } catch (err) {
        // One consumer's render bug must not kill the transport for everyone.
        // eslint-disable-next-line no-console
        console.warn("[ws] listener threw for", event.type, err);
      }
    }
  }
}

function open(): void {
  if (typeof WebSocket === "undefined" || typeof location === "undefined") return;
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  setStatus(attempt === 0 ? "connecting" : status === "failed" ? "failed" : "reconnecting");

  let ws: WebSocket;
  try {
    ws = new WebSocket(eventsUrl(lastSeq, location));
  } catch {
    // Constructor throws on a malformed URL / blocked scheme. Treat like any
    // other failure so we still back off rather than dying here.
    scheduleReconnect();
    return;
  }
  socket = ws;
  closingDeliberately = false;

  ws.onopen = () => {
    if (socket !== ws) return;
    attempt = 0;
    setStatus("open");
    armIdleWatchdog();
  };

  ws.onmessage = (ev: MessageEvent) => {
    if (socket !== ws) return;
    // ANY traffic proves the link is alive, pings included — that is what makes
    // the idle watchdog safe to keep tight.
    armIdleWatchdog();
    if (typeof ev.data !== "string") return; // binary frames are not our contract
    const frame = parseFrame(ev.data);
    if (frame.kind === "event") dispatch(frame.event);
  };

  ws.onerror = () => {
    // Browsers give no detail here by design. `onclose` always follows, so the
    // reconnect is driven from there and this handler only exists to stop the
    // event surfacing as an unhandled error.
  };

  ws.onclose = () => {
    if (socket !== ws) return;
    socket = null;
    if (idleTimer !== undefined) {
      clearTimeout(idleTimer);
      idleTimer = undefined;
    }
    if (closingDeliberately) {
      closingDeliberately = false;
      return;
    }
    scheduleReconnect();
  };
}

function forceReconnect(_reason: "idle" | "wake"): void {
  const ws = socket;
  socket = null;
  if (ws) {
    closingDeliberately = true;
    try {
      ws.close();
    } catch {
      /* already closing */
    }
    closingDeliberately = false;
  }
  clearTimers();
  if (totalListeners() === 0) {
    setStatus("idle");
    return;
  }
  // A wake-from-sleep should retry NOW, not after the accumulated backoff — the
  // network that was missing is usually back the instant the lid opens.
  attempt = 0;
  open();
}

/**
 * Subscribe to one event `type` (or `WS_ALL`). Returns the unsubscribe.
 *
 * The socket is reference-counted: it opens on the first subscriber and closes
 * when the last one goes away, so an unmounted component leaves nothing behind
 * and no route in the app pays for a channel it doesn't read.
 */
export function subscribe(type: string, cb: EventListener): () => void {
  let set = listeners.get(type);
  if (!set) {
    set = new Set();
    listeners.set(type, set);
  }
  set.add(cb);
  if (totalListeners() === 1) {
    attempt = 0;
    open();
  }
  let done = false;
  return () => {
    if (done) return; // idempotent: React StrictMode double-invokes cleanups
    done = true;
    const s = listeners.get(type);
    if (s) {
      s.delete(cb);
      if (s.size === 0) listeners.delete(type);
    }
    if (totalListeners() === 0) closeSocket();
  };
}

/** Observe connection-state changes (for `useSyncExternalStore`). */
export function subscribeStatus(cb: () => void): () => void {
  statusListeners.add(cb);
  return () => {
    statusListeners.delete(cb);
  };
}

export function getStatus(): WsStatus {
  return status;
}

/** Highest event seq seen. Sent back as `?since=` so a reconnect catches up
 *  rather than silently skipping whatever happened while we were away. */
export function getLastSeq(): number {
  return lastSeq;
}

/** Close and stop retrying. Called automatically when the last subscriber
 *  leaves; exported for tests and for an explicit app-level teardown. */
export function closeSocket(): void {
  clearTimers();
  const ws = socket;
  socket = null;
  if (ws) {
    closingDeliberately = true;
    try {
      ws.close();
    } catch {
      /* already closing */
    }
    closingDeliberately = false;
  }
  attempt = 0;
  setStatus("idle");
}

// ── wake-ups ────────────────────────────────────────────────────────────────
// A laptop that slept through the backoff, or a Tailscale route that just came
// back, should not have to wait out a 30 s timer. Both signals are advisory:
// if the link is still dead the normal backoff resumes from the failed attempt.
if (typeof window !== "undefined") {
  window.addEventListener("online", () => {
    if (totalListeners() > 0 && status !== "open") forceReconnect("wake");
  });
  if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && totalListeners() > 0 && status !== "open") {
        forceReconnect("wake");
      }
    });
  }
}
